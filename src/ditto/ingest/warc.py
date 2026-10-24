"""WARC input: find HTML responses in ``.warc`` / ``.warc.gz`` files and re-read one by offset."""

from __future__ import annotations

import io
import math
import os
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
    record = next(iter(ArchiveIterator(io.BytesIO(raw))))
    body: bytes = record.content_stream().read()
    return body
