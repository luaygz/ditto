"""Hand-built pages for clustering tests: a product and an article template under shared chrome."""

from __future__ import annotations

import random

CHROME = """<header class="site"><a class="logo" href="/">Shop</a>
<nav class="menu"><ul><li><a href="/a">A</a></li><li><a href="/b">B</a></li></ul></nav></header>"""
FOOTER = """<footer class="site-footer"><p class="copy">(c) 2026</p>
<ul class="links"><li><a href="/about">About</a></li></ul></footer>"""


def page(content: str) -> str:
    return (
        "<html><head><title>x</title></head><body class='page'>"
        f"{CHROME}<main>{content}</main>{FOOTER}</body></html>"
    )


def product(i: int, rng: random.Random) -> str:
    sale = "<p class='sale'>Sale!</p>" if rng.random() < 0.5 else ""
    reviews = (
        "<section class='reviews'><h3>Reviews</h3><ul><li class='review'>ok</li></ul></section>"
        if rng.random() < 0.3
        else ""
    )
    return page(
        f"<div class='product'><h1 class='name'>Item {i}</h1><span class='price'>${i}</span>"
        f"{sale}<dl class='specs'><dt>Color</dt><dd>red</dd><dt>Size</dt><dd>L</dd></dl>"
        f"<div class='gallery'><img class='thumb' src='/{i}.png'></div>{reviews}</div>"
    )


def article(i: int, rng: random.Random) -> str:
    fig = "<figure class='lead'><img src='/f.png'><figcaption>c</figcaption></figure>"
    return page(
        f"<article class='post'><h1 class='headline'>News {i}</h1>"
        f"<p class='byline'><span class='author'>Ann</span><time>today</time></p>"
        f"{fig if rng.random() < 0.6 else ''}<div class='body'><p>a</p><p>b</p></div>"
        f"<aside class='related'><h4>More</h4><ol><li><a href='/n'>n</a></li></ol></aside>"
        f"</article>"
    )
