"""Builders for ingest tests: small HTML directories and WARC files written with warcio."""

from __future__ import annotations

import io
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from warcio.statusandheaders import StatusAndHeaders
from warcio.warcwriter import WARCWriter


@dataclass
class Rec:
    """One WARC record to write. ``kind`` is response / revisit / request / warcinfo."""

    url: str
    body: bytes = b""
    date: str = "2024-01-01T00:00:00Z"
    kind: str = "response"
    status: str = "200 OK"
    headers: list[tuple[str, str]] = field(
        default_factory=lambda: [("Content-Type", "text/html; charset=utf-8")]
    )


def write_warc(path: Path, records: Sequence[Rec]) -> Path:
    """Write ``records`` to ``path`` (gzip per record iff the name ends in ``.gz``)."""
    with open(path, "wb") as fh:
        writer = WARCWriter(fh, gzip=path.name.endswith(".gz"))
        for rec in records:
            warc_headers = {"WARC-Date": rec.date}
            if rec.kind == "warcinfo":
                record = writer.create_warcinfo_record(path.name, {"software": "ditto-tests"})
            elif rec.kind == "request":
                http = StatusAndHeaders(
                    f"GET {rec.url} HTTP/1.1", [("Host", "example.com")], is_http_request=True
                )
                record = writer.create_warc_record(
                    rec.url, "request", http_headers=http, warc_headers_dict=warc_headers
                )
            elif rec.kind == "revisit":
                http = StatusAndHeaders(rec.status, rec.headers, protocol="HTTP/1.1")
                record = writer.create_revisit_record(
                    rec.url,
                    digest="sha1:AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
                    refers_to_uri=rec.url,
                    refers_to_date="2023-01-01T00:00:00Z",
                    http_headers=http,
                    warc_headers_dict=warc_headers,
                )
            else:
                http = StatusAndHeaders(rec.status, rec.headers, protocol="HTTP/1.1")
                record = writer.create_warc_record(
                    rec.url,
                    "response",
                    payload=io.BytesIO(rec.body),
                    http_headers=http,
                    warc_headers_dict=warc_headers,
                )
            writer.write_record(record)
    return path


def write_tree(root: Path, files: dict[str, bytes]) -> Path:
    for rel, data in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return root
