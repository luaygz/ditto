from __future__ import annotations

import numpy as np

from ditto.fingerprint import fingerprint_html, fingerprint_tree
from ditto.fingerprint.dom import clean_html, iter_paths
from ditto.fingerprint.tree import TEXT_JOIN, PageFingerprint, signature_hash
from ditto.textutil import hash64

LIST_PAGE = """<html><head><title>Shop</title></head><body class="page">
<ul class="items">
  <li class="item"><a href="/p/1">One</a><span class="price">$1</span></li>
  <li class="item"><a href="/p/2">Two</a><span class="price">$2</span></li>
  <li class="item"><a href="/p/3">Three</a></li>
</ul>
</body></html>"""


def _paths(fp: PageFingerprint) -> list[str]:
    return [fp.paths[int(h)] for h in fp.signature]


def test_signature_is_sorted_unique_hashes_of_all_element_paths() -> None:
    root = clean_html(LIST_PAGE)
    fp = fingerprint_tree(root)
    expected = {p for _, p in iter_paths(root)}
    assert set(_paths(fp)) == expected
    assert fp.signature.dtype == np.uint64
    assert np.all(np.diff(fp.signature.astype(np.float64)) > 0) or len(fp.signature) < 2
    assert list(fp.signature) == sorted(hash64(p) for p in expected)


def test_repeated_siblings_collapse() -> None:
    three = fingerprint_html(LIST_PAGE)
    one = fingerprint_html(
        LIST_PAGE.replace(
            '<li class="item"><a href="/p/2">Two</a><span class="price">$2</span></li>', ""
        ).replace('<li class="item"><a href="/p/3">Three</a></li>', "")
    )
    many = fingerprint_html(
        LIST_PAGE.replace(
            '<li class="item"><a href="/p/3">Three</a></li>',
            '<li class="item"><a href="/p/3">Three</a></li>' * 40,
        )
    )
    assert np.array_equal(three.signature, one.signature)
    assert np.array_equal(three.signature, many.signature)
    assert three.signature_hash == one.signature_hash == many.signature_hash


def test_utility_and_hashed_classes_do_not_change_signature() -> None:
    a = fingerprint_html('<html><body><div class="card mt-2 css-9x8y7z">x</div></body></html>')
    b = fingerprint_html('<html><body><div class="card px-4 sc-Qw3rT1">y</div></body></html>')
    c = fingerprint_html('<html><body><div class="panel">y</div></body></html>')
    assert a.signature_hash == b.signature_hash
    assert a.signature_hash != c.signature_hash


def test_text_items_hash_joined_texts_in_document_order() -> None:
    fp = fingerprint_html(LIST_PAGE)
    by_path = {
        fp.paths[int(h)]: (int(t), int(c))
        for h, t, c in zip(fp.text_path, fp.text_hash, fp.text_count, strict=True)
    }
    link = "html > body.page > ul.items > li.item > a"
    assert by_path[link] == (hash64(TEXT_JOIN.join(["One", "Two", "Three"])), 3)
    assert by_path[link + " @href"] == (hash64(TEXT_JOIN.join(["/p/1", "/p/2", "/p/3"])), 3)
    assert by_path["html > body.page > ul.items > li.item > span.price"][1] == 2
    assert by_path["html > head > title"] == (hash64("Shop"), 1)
    # elements without own text are not text items
    assert "html > body.page > ul.items" not in by_path
    assert list(fp.text_path) == sorted(int(h) for h in fp.text_path)
    assert fp.text_count.dtype == np.uint32


def test_paths_vocabulary_covers_all_hashes() -> None:
    fp = fingerprint_html(LIST_PAGE)
    for h in list(fp.signature) + list(fp.text_path):
        assert hash64(fp.paths[int(h)]) == int(h)


def test_empty_fingerprint_and_signature_hash() -> None:
    empty = PageFingerprint.empty()
    assert len(empty.signature) == 0 and len(empty.text_path) == 0
    assert signature_hash(np.array([1, 2], dtype=np.uint64)) != signature_hash(
        np.array([2, 1], dtype=np.uint64)
    )
    assert signature_hash(np.array([5], dtype=np.uint64)) == signature_hash(
        np.array([5], dtype=np.uint64)
    )


MESSY = """<html lang=en><head><title>  A &amp; B </title><meta name=x></head>
<body class="page">
  text before <!-- c1 --> after comment
  <div class="row card">one<b>bold</b> tail one <i>it</i>tail two<br>
     <span>x</span>   </div>
  <div class="card">two&nbsp;&nbsp;spaces<img src=" /i.png "></div>
  <div class="card"><a href="">empty href</a><a href="  ">blank</a><a href="/z">z</a></div>
  <table><tr><td>1</td><td></td><td> 3 </td></tr><tr><td>4</td></tr></table>
  <p>caf\u00e9 \u2014 na\u00efve</p><?php echo 1; ?>
</body></html>"""


def _reference(root: object) -> tuple[set[str], dict[str, list[str]]]:
    from ditto.fingerprint.dom import iter_text_items

    paths = {p for _, p in iter_paths(root)}  # type: ignore[arg-type]
    texts: dict[str, list[str]] = {}
    for item in iter_text_items(root):  # type: ignore[arg-type]
        texts.setdefault(item.path, []).append(item.text)
    return paths, texts


def test_fingerprint_matches_reference_on_messy_html() -> None:
    from ditto.fingerprint.dom import clean_etree

    for root in (
        clean_html(MESSY),
        clean_etree(MESSY),
        clean_etree(MESSY.encode("utf-8"), "utf-8"),
    ):
        fp = fingerprint_tree(root)
        paths, texts = _reference(clean_html(MESSY))
        assert set(_paths(fp)) == paths
        got = {
            fp.paths[int(h)]: (int(t), int(c))
            for h, t, c in zip(fp.text_path, fp.text_hash, fp.text_count, strict=True)
        }
        assert got == {p: (hash64(TEXT_JOIN.join(v)), len(v)) for p, v in texts.items()}
    body = texts["html > body.page"]
    assert body == ["text before after comment"]
    assert texts["html > body.page > div.card"] == [
        "one tail one tail two",
        "two spaces",
    ]
    assert texts["html > body.page > div.card > a @href"] == ["/z"]
    assert texts["html > body.page > div.card > img @src"] == ["/i.png"]
