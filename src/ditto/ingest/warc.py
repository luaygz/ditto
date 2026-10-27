"""WARC input: find HTML responses in ``.warc`` / ``.warc.gz`` files and re-read one by offset."""

from __future__ import annotations

import io
import math
import os
import zlib
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from warcio.archiveiterator import ArchiveIterator

from ditto.ingest.capture import MAX_ERROR_SAMPLES, Capture, ScanResult, merge_scans
from ditto.ingest.encoding import charset_from_content_type, detect_encoding, sniff_html
from ditto.textutil import sha256_hex

WARC_SUFFIXES = (".warc", ".warc.gz")
HTML_MIMES = frozenset({"text/html", "application/xhtml+xml"})


def is_warc_path(path: Path | str) -> bool:
    return os.fspath(path).lower().endswith(WARC_SUFFIXES)


def find_warc_files(directory: Path) -> list[Path]:
    """Every WARC file under ``directory`` (recursive, hidden entries skipped), sorted."""
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(directory):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        found.extend(
            Path(dirpath, name)
            for name in filenames
            if not name.startswith(".") and is_warc_path(name)
        )
    return sorted(found)


def warc_timestamp(value: str | None) -> float:
    """Seconds since the epoch for a ``WARC-Date`` (naive dates are UTC); ``-inf`` if absent
    or unparsable, so such captures lose against any dated one."""
    if not value:
        return -math.inf
    try:
        parsed = datetime.fromisoformat(value.strip())
    except ValueError:
        return -math.inf
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.timestamp()


def scan_warc_files(paths: Sequence[str]) -> ScanResult:
    """pmap worker: scan each WARC file in ``paths`` (absolute paths)."""
    return merge_scans([scan_warc(path) for path in paths])


def scan_warc(path: str) -> ScanResult:
    """Every HTTP 200 HTML ``response`` record in one WARC file, with its record offset and
    length. Payloads are hashed and encoding-sniffed here and then dropped."""
    captures: list[Capture] = []
    counts: Counter[str] = Counter()
    errors: list[str] = []
    try:
        with open(path, "rb") as fh:
            records = ArchiveIterator(fh)
            for record in records:
                if record.format != "warc":  # warcio leniently parses junk as ARC records
                    raise ValueError(f"not a WARC file: found an {record.format} record")
                counts["records"] += 1
                if record.rec_type != "response":
                    counts["skipped_revisit" if record.rec_type == "revisit" else "other"] += 1
                    continue
                http = record.http_headers
                if http is None:
                    counts["skipped_non_html"] += 1
                    continue
                content_type = http.get_header("Content-Type") or ""
                mime = content_type.split(";", 1)[0].strip().lower()
                if mime and mime not in HTML_MIMES:
                    counts["skipped_non_html"] += 1
                    continue
                if http.get_statuscode() != "200":
                    counts["skipped_status"] += 1
                    continue
                url = record.rec_headers.get_header("WARC-Target-URI") or ""
                if not url:
                    counts["skipped_no_url"] += 1
                    continue
                try:
                    body = record.content_stream().read()
                except Exception as exc:  # corrupt chunked/gzip payload; keep scanning
                    counts["errors"] += 1
                    if len(errors) < MAX_ERROR_SAMPLES:
                        errors.append(f"{path}: {url}: {type(exc).__name__}: {exc}")
                    continue
                if not body or body.isspace():
                    counts["skipped_empty"] += 1
                    continue
                if not mime and not sniff_html(body):
                    counts["skipped_non_html"] += 1
                    continue
                date = record.rec_headers.get_header("WARC-Date")
                offset = records.get_record_offset()
                captures.append(
                    Capture(
                        url=url,
                        source=path,
                        sha256=sha256_hex(body),
                        fetched_at=date,
                        offset=offset,
                        length=records.get_record_length(),
                        encoding=detect_encoding(body, charset_from_content_type(content_type)),
                        rank=(warc_timestamp(date), path, offset),
                    )
                )
    except Exception as exc:  # unreadable or truncated archive: keep what was read so far
        counts["errors"] += 1
        errors.append(f"{path}: {type(exc).__name__}: {str(exc)[:200]}")
    return ScanResult(captures, counts, errors)


def read_warc_payload(path: str, offset: int, length: int) -> bytes:
    """HTTP payload body (transfer/content encodings removed) of the record at ``offset``."""
    with open(path, "rb") as fh:
        fh.seek(offset)
        raw = fh.read(length)
    body = plain_payload(raw)
    if body is not None:
        return body
    record = next(iter(ArchiveIterator(io.BytesIO(raw))))
    content: bytes = record.content_stream().read()
    return content


_BLANK_LINE = b"\r\n\r\n"
_CODED = frozenset({b"transfer-encoding", b"content-encoding"})


def _header_lines(block: bytes) -> list[bytes] | None:
    """CRLF header lines after the first line; None when any line has a bare CR or LF or a
    header is folded or has no colon (warcio's lenient parsing decides those)."""
    first, *lines = block.split(b"\r\n")
    if b"\n" in first or b"\r" in first:
        return None
    for line in lines:
        if b"\n" in line or b"\r" in line or line[:1] in (b" ", b"\t") or b":" not in line:
            return None
    return lines


def plain_payload(raw: bytes) -> bytes | None:
    """The HTTP body of one WARC ``response`` record (a gzip member or plain bytes) when it is
    plain: strict CRLF headers and no ``Transfer-Encoding``/``Content-Encoding``. That is the
    bytes after the HTTP header block, up to the WARC ``Content-Length``, exactly what warcio's
    ``content_stream()`` returns, without its per-record stream machinery. None for anything
    else, which warcio then reads."""
    data = raw
    if raw[:2] == b"\x1f\x8b":
        member = zlib.decompressobj(31)
        try:
            data = member.decompress(raw)
        except zlib.error:
            return None
        if not member.eof or member.unused_data:
            return None
    end = data.find(_BLANK_LINE)
    if end < 0 or not data.startswith(b"WARC/"):
        return None
    lines = _header_lines(data[:end])
    if lines is None:
        return None
    fields = {}
    for line in lines:
        name, _, value = line.partition(b":")
        fields[name.strip().lower()] = value.strip()
    if fields.get(b"warc-type") != b"response":
        return None
    if not fields.get(b"content-type", b"").lower().startswith(b"application/http"):
        return None
    try:
        size = int(fields[b"content-length"])
    except (KeyError, ValueError):
        return None
    block = data[end + len(_BLANK_LINE) : end + len(_BLANK_LINE) + size]
    http_end = block.find(_BLANK_LINE)
    if len(block) != size or http_end < 0 or not block.startswith(b"HTTP/"):
        return None
    http = _header_lines(block[:http_end])
    if http is None or any(line.partition(b":")[0].strip().lower() in _CODED for line in http):
        return None
    return block[http_end + len(_BLANK_LINE) :]
