from __future__ import annotations

import pytest

from ditto.cluster import url_pattern


@pytest.mark.parametrize(
    ("url", "pattern"),
    [
        ("https://shop.example.com/products/blue-widget-42", "shop.example.com/products/{slug}"),
        ("http://www.Shop.example.com:8080/products/123", "shop.example.com/products/{n}"),
        ("https://ex.com/blog/2024/05/how-to-fold-a-shirt-77", "ex.com/blog/{n}/{n}/{slug}"),
        ("https://ex.com/jobs/481/senior_python_dev", "ex.com/jobs/{n}/{slug}"),
        ("https://ex.com/p/blue-widget.html", "ex.com/p/{slug}.html"),
        ("https://ex.com/page2/", "ex.com/page{n}/"),
        ("https://ex.com/p000123.html", "ex.com/p{n}.html"),
        ("https://ex.com/item/3f2a9c1e-77ab-4c1d-9e1f-0a1b2c3d4e5f", "ex.com/item/{id}"),
        ("https://ex.com/a/averyveryveryverylongtokenwithoutseparators", "ex.com/a/{slug}"),
        ("https://ex.com/search?q=red+shoes&page=3#top", "ex.com/search?page={n}&q={v}"),
        ("https://ex.com/list?b=&a=1", "ex.com/list?a={n}&b="),
        ("https://ex.com/", "ex.com/"),
        ("https://ex.com", "ex.com"),
        ("site/pages/item-12.html", "site/pages/{slug}.html"),
        ("https://ex.com/about", "ex.com/about"),
        ("https://ex.com/v1.2/docs", "ex.com/v{n}.{n}/docs"),
    ],
)
def test_url_pattern(url: str, pattern: str) -> None:
    assert url_pattern(url) == pattern


def test_same_template_urls_share_a_pattern() -> None:
    a = url_pattern("https://ex.com/products/red-shoe-1")
    b = url_pattern("https://ex.com/products/green-hat-99")
    c = url_pattern("https://ex.com/events/99")
    assert a == b != c


def test_malformed_url_does_not_raise() -> None:
    assert url_pattern("http://[::1/x7") == "http://[::{n}/x{n}"
