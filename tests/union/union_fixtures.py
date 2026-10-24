"""Builders for union page tests: a small "staff profile" template with real HTML."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ditto.analyze.models import Candidate, CandidateTable, Table

P = "html > body.page > main.profile"
CONTACT = f"{P} > div.contact"
SPECS = f"{P} > dl.specs"
PUBS = f"{P} > ul.pubs > li.pub"


def profile_page(
    name: str,
    *,
    phone: str = "555-0100",
    email: str | None = None,
    specs: Sequence[tuple[str, str]] = (("Department", "Sales"), ("Office", "B12")),
    pubs: Sequence[str | tuple[str, str]] = ("On things",),
    awards: Sequence[str] = (),
    bio: str | None = None,
) -> str:
    """A staff profile page. A pub given as ``(title, year)`` gets a ``span.year`` too."""
    email_html = (
        f'<span class="label">Email:</span>'
        f'<a class="email" href="mailto:{email}" onclick="track()">{email}</a>'
        if email
        else ""
    )
    specs_html = "".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in specs)
    items = []
    for pub in pubs:
        if isinstance(pub, tuple):
            title, year = pub
            items.append(
                f'<li class="pub"><span class="title">{title}</span>'
                f'<span class="year">{year}</span></li>'
            )
        else:
            items.append(f'<li class="pub"><span class="title">{pub}</span></li>')
    pubs_html = "".join(items)
    awards_html = (
        '<div class="awards"><h3>Awards</h3><ul>'
        + "".join(f'<li class="award">{a}</li>' for a in awards)
        + "</ul></div>"
        if awards
        else ""
    )
    bio_html = f'<p class="bio">{bio}</p>' if bio else ""
    return f"""<!doctype html>
<html><head><title>Profile of {name}</title><script>var x = 1;</script>
<style>.a{{color:red}}</style></head>
<body class="page">
<nav class="site-nav"><a href="/">Home</a> <a href="/staff">Staff</a> <a href="/about">About</a></nav>
<div class="cookie-banner" id="cookies">We use cookies to improve the site. <button>OK</button></div>
<main class="profile">
  <h1 class="name">{name}</h1>
  <div class="contact"><span class="label">Phone:</span><span class="phone">{phone}</span>{email_html}</div>
  {bio_html}
  <h2>Details</h2>
  <dl class="specs">{specs_html}</dl>
  <ul class="pubs">{pubs_html}</ul>
  {awards_html}
</main>
<footer class="site-footer">Copyright 2026 Example University</footer>
</body></html>"""


def cand(cid: str, path: str, cls: str, p: float, **kw: Any) -> Candidate:
    attr = path.rsplit(" @", 1)[1] if " @" in path else None
    data: dict[str, Any] = {"id": cid, "path": path, "class": cls, "p": p, "m": 0.1, "attr": attr}
    data.update(kw)
    return Candidate.model_validate(data)


def profile_table(*, email_p: float = 0.41, fax_p: float = 0.3) -> CandidateTable:
    candidates = [
        cand("c0", "html > head > title", "mandatory", 1.0, examples=["Profile of Ann Lee"]),
        cand("c1", f"{P} > h1.name", "mandatory", 1.0, examples=["Ann Lee"]),
        cand("c2", f"{CONTACT} > span.phone", "mandatory", 1.0, label_hint="Phone:"),
        cand(
            "c3",
            f"{CONTACT} > a.email",
            "optional",
            email_p,
            label_hint="Email:",
            examples=["a.khoury@example.org"],
        ),
        cand(
            "c4",
            f"{CONTACT} > a.email @href",
            "optional",
            email_p,
            label_hint="Email:",
            examples=["mailto:a.khoury@example.org"],
        ),
        cand("c5", f"{SPECS} > dt", "label", 1.0, m=1.0, table_id="t0", table_role="kv_key"),
        cand(
            "c6",
            f"{SPECS} > dd",
            "mandatory",
            1.0,
            label_hint="Department",
            table_id="t0",
            table_role="kv_value",
            column="Department",
            examples=["Sales"],
        ),
        cand(
            "c7",
            f"{SPECS} > dd",
            "mandatory",
            1.0,
            label_hint="Office",
            table_id="t0",
            table_role="kv_value",
            column="Office",
        ),
        cand(
            "c8",
            f"{SPECS} > dd",
            "optional",
            fax_p,
            label_hint="Fax",
            table_id="t0",
            table_role="kv_value",
            column="Fax",
            examples=["555-0199"],
        ),
        cand("c9", f"{PUBS} > span.title", "list", 1.0, repeat_min=1, repeat_max=30),
        cand("c10", f"{P} > div.awards > ul > li.award", "optional", 0.1, label_hint="Awards"),
        cand("c11", "html > body.page > nav.site-nav > a", "boilerplate", 1.0, m=1.0),
        cand("c12", "html > body.page > nav.site-nav > a @href", "boilerplate", 1.0, m=1.0),
        cand("c13", "html > body.page > div.cookie-banner", "boilerplate", 1.0, m=1.0),
        cand("c14", "html > body.page > div.cookie-banner > button", "boilerplate", 1.0, m=1.0),
        cand("c15", "html > body.page > footer.site-footer", "boilerplate", 1.0, m=1.0),
        cand("c16", f"{CONTACT} > span.label", "label", 1.0, m=1.0),
        cand("c17", f"{P} > h2", "label", 1.0, m=1.0),
        cand("c18", f"{P} > div.awards > h3", "label", 0.1, m=1.0),
    ]
    tables = [
        Table(
            id="t0",
            kind="key_value",
            layout="dl",
            container_path=SPECS,
            row_path=f"{SPECS} > dt",
            p=1.0,
            rows_min=2,
            rows_max=3,
            key_candidate="c5",
            keys_seen=["Department", "Office", "Fax"],
        )
    ]
    return CandidateTable(
        template_id="t_test",
        cluster_size=40,
        n_sampled=40,
        candidates=candidates,
        tables=tables,
    )
