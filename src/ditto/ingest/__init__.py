"""Stage 1: ingest an HTML directory (optional ``urls.tsv``) or WARC files into a run directory,
and read pages back by dense ``page_id``."""

from __future__ import annotations

from ditto.ingest.encoding import detect_encoding
from ditto.ingest.stage import IngestResult, ingest
from ditto.ingest.store import PageRecord, PageStore, read_html, read_page

__all__ = [
    "IngestResult",
    "PageRecord",
    "PageStore",
    "detect_encoding",
    "ingest",
    "read_html",
    "read_page",
]
