"""Hand-built product-page cluster exercising every analysis rule (no randomness)."""

from __future__ import annotations

from datetime import date, timedelta

N_PAGES = 100
NOISE_PAGE = 3


def email_on(i: int) -> bool:
    return i % 5 < 2  # 40% of pages


def color_on(i: int) -> bool:
    return i % 2 == 1  # 50% of pages


def n_pubs(i: int) -> int:
    return 1 + i % 4


def product_page(i: int) -> str:
    avail = "In stock" if i % 10 < 7 else "Out of stock"
    tags = "".join(f"<li>tag-{(i * 3 + k) % 17}</li>" for k in range(1 + (i * 7) % 4))
    badge = f'<p class="badge">Save {5 + i % 30}%</p>' if email_on(i) else ""
    email = (
        f"<dt>Email</dt><dd><a href='mailto:p{i}@example.org'>p{i}@example.org</a></dd>"
        if email_on(i)
        else ""
    )
    color = f"<dt>Color</dt><dd>{['red', 'blue', 'green'][i % 3]}</dd>" if color_on(i) else ""
    pubs = "".join(
        f"<tr><td><a href='/pub/{i}/{r}'>Paper {i}-{r}</a></td><td>{2000 + (i + r) % 20}</td></tr>"
        for r in range(n_pubs(i))
    )
    day0 = date(2020, 1, 1)
    log = "".join(
        f"<tr><td>{(day0 + timedelta(days=i * 3 + r)).isoformat()}</td><td>event {i}.{r}</td></tr>"
        for r in range(2 + i % 2)
    )
    results = "".join(
        f"<li class='result'><a class='title' href='/item/{i}/{k}'>Item {i}-{k}</a>"
        f"<span class='cost'>${i + k}.00</span></li>"
        for k in range(2 + i % 3)
    )
    noise = f'<p class="promo">Special offer {i}</p>' if i == NOISE_PAGE else ""
    return f"""<!doctype html>
<html><head><title>Widget {i} | Shop</title><script>var x = {i};</script></head>
<body>
<nav><ul><li><a href="/">Home</a></li><li><a href="/about">About</a></li></ul></nav>
<main>
<h1 class="name">Widget {i}</h1>
<div class="buy"><span class="lbl">Price:</span><span class="price">${10 + i}.99</span></div>
<p class="avail">{avail}</p>
{badge}
<ul class="tags">{tags}</ul>
<a class="more" href="/widgets/{i}/details">Read more</a>
<dl class="profile"><dt>Name</dt><dd>Person {i}</dd>{email}<dt>Phone</dt><dd>555-{i:04d}</dd>{color}</dl>
<table class="pubs"><thead><tr><th>Title</th><th>Year</th></tr></thead><tbody>{pubs}</tbody></table>
<table class="log">{log}</table>
<table class="facts"><tr><th>Material</th><td>{["oak", "pine", "steel", "glass"][i % 4]}</td></tr>
<tr><th>Origin</th><td>{["France", "Peru", "Japan"][i % 3]}</td></tr></table>
<ul class="results">{results}</ul>
<div class="desc"><h4>Description</h4><p class="lead">Lead {i}</p><p class="body">Body text {i}</p></div>
{noise}
</main>
<footer><p>Copyright Shop</p></footer>
</body></html>"""


def site_pages(n: int = N_PAGES) -> list[tuple[int, str]]:
    return [(i, product_page(i)) for i in range(n)]
