"""Directory input: list HTML files, read the optional ``urls.tsv`` manifest, hash and store
each file as a content-addressed blob."""

from __future__ import annotations

import os
import posixpath
import time
from collections import Counter
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from pathlib import Path

from ditto.ingest.capture import MAX_ERROR_SAMPLES, Capture, ScanResult
from ditto.ingest.encoding import detect_encoding, sniff_html
from ditto.textutil import sha256_hex

HTML_SUFFIXES = frozenset({".html", ".htm", ".xhtml"})
MANIFEST_NAME = "urls.tsv"


@dataclass(frozen=True)
class Manifest:
    """``urls.tsv``: relative POSIX path -> URL."""

    urls: dict[str, str]
    invalid_lines: int


@dataclass(frozen=True)
class Listing:
    files: list[tuple[str, bool]]  # (relative POSIX path, needs content sniffing), sorted
    skipped_non_html: int


def normalize_relpath(path: str, root: Path) -> str:
    """Relative POSIX form of a manifest/file path (absolute paths are made relative to root)."""
    if os.path.isabs(path):
        path = os.path.relpath(path, os.path.abspath(root))
    return posixpath.normpath(path.replace("\\", "/"))


def read_manifest(path: Path, root: Path) -> Manifest:
    """Parse ``path<TAB>url`` lines; blank lines, ``#`` comments and a ``path<TAB>url`` header
    are ignored, later lines win for a repeated path, malformed lines are counted."""
    urls: dict[str, str] = {}
    invalid = 0
    with open(path, encoding="utf-8-sig", errors="replace", newline="") as fh:
        for number, raw in enumerate(fh):
            line = raw.rstrip("\r\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            rel, sep, url = line.partition("\t")
            rel, url = rel.strip(), url.strip()
            if not sep or not rel or not url:
                invalid += 1
                continue
            if number == 0 and rel.lower() == "path" and url.lower() == "url":
                continue
            urls[normalize_relpath(rel, root)] = url
    return Manifest(urls, invalid)


def list_html_files(
    root: Path,
    *,
    listed: Collection[str] = (),
    ignore: Collection[str] = (),
    exclude: Collection[str] = (),
) -> Listing:
    """Walk ``root`` recursively, skipping hidden entries, the relative file paths in ``ignore``
    and directories whose real path is in ``exclude``. Files with an HTML suffix are taken as
    is; extensionless files and files named in the manifest (``listed``) are taken subject to
    content sniffing; everything else counts as non-HTML."""
    files: list[tuple[str, bool]] = []
    skipped = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d
            for d in dirnames
            if not d.startswith(".")
            and not (exclude and os.path.realpath(os.path.join(dirpath, d)) in exclude)
        ]
        rel_dir = os.path.relpath(dirpath, root).replace(os.sep, "/")
        for name in filenames:
            rel = name if rel_dir == "." else f"{rel_dir}/{name}"
            if name.startswith(".") or rel in ignore:
                continue
            suffix = os.path.splitext(name)[1].lower()
            if suffix in HTML_SUFFIXES:
                files.append((rel, False))
            elif not suffix or rel in listed:
                files.append((rel, True))
            else:
                skipped += 1
    files.sort()
    return Listing(files, skipped)


def blob_relpath(sha256: str) -> str:
    """Blob location relative to the run root (``blobs/<sha[:2]>/<sha>``)."""
    return f"blobs/{sha256[:2]}/{sha256}"


def store_blob(src: str, data: bytes, dest: str) -> None:
    """Hardlink ``src`` to ``dest`` when possible, else write ``data`` atomically. Blobs are
    content-addressed, so an existing ``dest`` of the right size (left by a previous ingest
    into this run, or by another worker) already holds these bytes; a wrong-sized one is a
    hardlinked source edited in place after that ingest, and is replaced."""
    try:
        os.link(src, dest)
        return
    except FileExistsError:
        if os.stat(dest).st_size == len(data):
            return
    except OSError:
        pass  # cross-device, unsupported filesystem or permissions: copy instead
    tmp = f"{dest}.{os.getpid()}.tmp"
    with open(tmp, "wb") as fh:
        fh.write(data)
    os.replace(tmp, dest)


def scan_files(chunk: Sequence[tuple[str, bool]], root: str, run_root: str) -> ScanResult:
    """pmap worker: read, sniff, hash, detect encoding and store one chunk of files. The
    capture URL is the relative path; the caller substitutes manifest URLs."""
    captures: list[Capture] = []
    counts: Counter[str] = Counter()
    errors: list[str] = []
    for rel, needs_sniff in chunk:
        path = os.path.join(root, rel)
        try:
            with open(path, "rb") as fh:
                mtime = os.fstat(fh.fileno()).st_mtime
                data = fh.read()
            if not data or data.isspace():
                counts["skipped_empty"] += 1
                continue
            if needs_sniff and not sniff_html(data):
                counts["skipped_non_html"] += 1
                continue
            sha = sha256_hex(data)
            blob = blob_relpath(sha)
            store_blob(path, data, os.path.join(run_root, blob))
        except OSError as exc:
            counts["errors"] += 1
            if len(errors) < MAX_ERROR_SAMPLES:
                errors.append(f"{rel}: {exc}")
            continue
        captures.append(
            Capture(
                url=rel,
                source=blob,
                sha256=sha,
                fetched_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(mtime)),
                offset=None,
                length=len(data),
                encoding=detect_encoding(data),
                rank=(mtime, rel, 0),
            )
        )
    return ScanResult(captures, counts, errors)
