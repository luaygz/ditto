from __future__ import annotations

import codecs

import pytest

from ditto.ingest.encoding import (
    charset_from_content_type,
    declared_encoding,
    detect_encoding,
    normalize_encoding,
    sniff_html,
)

FRENCH = "L'été dernier, nous sommes allés à la plage près de Nîmes. Le café était très bon. "


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("utf-8", "utf-8"),
        ("UTF8", "utf-8"),
        ('"utf-8"', "utf-8"),
        ("latin-1", "cp1252"),  # WHATWG: iso-8859-1 labels decode as windows-1252
        ("ISO-8859-1", "cp1252"),
        ("us-ascii", "cp1252"),
        ("windows-1252", "cp1252"),
        ("Shift_JIS", "cp932"),
        ("gb2312", "gbk"),
        ("koi8-r", "koi8-r"),
        ("base64", None),  # a Python codec, but not a text encoding
        ("rot13", None),
        ("utf-7", None),
        ("x-user-defined", None),
        ("bogus", None),
        ("", None),
        (None, None),
    ],
)
def test_normalize_encoding(label: str | None, expected: str | None) -> None:
    assert normalize_encoding(label) == expected


def test_charset_from_content_type() -> None:
    assert charset_from_content_type("text/html; charset=ISO-8859-1") == "ISO-8859-1"
    assert charset_from_content_type('text/html;charset="utf-8"; foo=bar') == "utf-8"
    assert charset_from_content_type("text/html") is None
    assert charset_from_content_type(None) is None


def test_bom_wins_over_stale_meta_charset() -> None:
    # SWDE pages: UTF-8 with BOM but a stale <meta charset=iso-8859-1> from the original crawl.
    page = codecs.BOM_UTF8 + '<html><head><meta charset="iso-8859-1"></head>naïve</html>'.encode()
    encoding = detect_encoding(page, http_charset="iso-8859-1")
    assert encoding == "utf-8-sig"
    assert page.decode(encoding) == '<html><head><meta charset="iso-8859-1"></head>naïve</html>'


@pytest.mark.parametrize("codec", ["utf-16-le", "utf-16-be"])
def test_utf16_bom(codec: str) -> None:
    bom = codecs.BOM_UTF16_LE if codec.endswith("le") else codecs.BOM_UTF16_BE
    page = bom + "<html><body>Ünïcode</body></html>".encode(codec)
    encoding = detect_encoding(page)
    assert encoding == "utf-16"
    assert page.decode(encoding) == "<html><body>Ünïcode</body></html>"


def test_http_charset_beats_meta() -> None:
    page = b'<meta charset="utf-8"><p>caf\xe9</p>'
    assert detect_encoding(page, http_charset="ISO-8859-1") == "cp1252"
    assert detect_encoding(page, http_charset="nonsense") == "utf-8"


def test_meta_charset_forms() -> None:
    assert detect_encoding(b"<html><head><meta charset=latin-1></head>caf\xe9") == "cp1252"
    http_equiv = (
        b'<meta http-equiv="Content-Type" content="text/html; charset=windows-1251">'
        b"\xcf\xf0\xe8\xe2\xe5\xf2"
    )
    assert detect_encoding(http_equiv) == "cp1251"
    assert http_equiv.decode(detect_encoding(http_equiv)).endswith("Привет")
    xml = b'<?xml version="1.0" encoding="ISO-8859-15"?><html><body>\xa4</body></html>'
    assert detect_encoding(xml) == "iso8859-15"
    assert xml.decode("iso8859-15").endswith("<body>€</body></html>")


def test_meta_utf16_means_utf8_and_bad_labels_fall_through() -> None:
    assert declared_encoding(b'<meta charset="utf-16"><p>x</p>') == "utf-8"
    assert declared_encoding(b'<meta charset="no-such-thing"><p>x</p>') is None
    assert detect_encoding('<meta charset="bogus"><p>café</p>'.encode()) == "utf-8"


def test_meta_beyond_prescan_window_is_ignored() -> None:
    page = b"<html>" + b" " * 5000 + b'<meta charset="koi8-r"></html>'
    assert declared_encoding(page) is None
    assert detect_encoding(page) == "utf-8"


def test_utf8_without_declaration() -> None:
    assert detect_encoding("<html><body>café — ☃</body></html>".encode()) == "utf-8"
    assert detect_encoding(b"<html><body>plain ascii</body></html>") == "utf-8"


def test_undeclared_legacy_encodings_use_detector() -> None:
    french = f"<html><body><p>{FRENCH * 3}</p></body></html>".encode("cp1252")
    encoding = detect_encoding(french)
    assert encoding == "cp1252"  # the detector ties cp1250/cp1252/cp1257; cp1252 wins
    assert FRENCH in french.decode(encoding)
    russian = "<html><body><p>" + "Привет, мир! Это тестовая страница. " * 4 + "</p></body></html>"
    data = russian.encode("cp1251")
    assert data.decode(detect_encoding(data)) == russian


def test_sniff_html() -> None:
    assert sniff_html(b"  \n<!DOCTYPE html><title>x</title>")
    assert sniff_html(b"<!-- saved page --><html><body></body></html>")
    assert sniff_html(codecs.BOM_UTF8 + b"<html>")
    assert sniff_html(codecs.BOM_UTF16_LE + "<html><body>".encode("utf-16-le"))
    assert not sniff_html(b"\x89PNG\r\n\x1a\n<html>")
    assert not sniff_html(b"just some text mentioning <html> later")
    assert not sniff_html(b'<?xml version="1.0"?><rss></rss>')
    assert not sniff_html(b"")
