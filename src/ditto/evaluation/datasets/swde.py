"""SWDE adapter: on-disk layout, ground-truth parser, frozen split, and evaluation sites.

SWDE (Hao, Cai, Pang, Zhang, SIGIR 2011): 8 verticals x 10 sites, 124,291 detail pages, 32
attributes. ``eval/datasets/swde/download.py`` fetches and verifies it into ``data/swde/``::

    data/swde/<vertical>/<vertical>-<site>(<n_pages>)/<NNNN>.htm
    data/swde/groundtruth/<vertical>/<vertical>-<site>-<attribute>.txt

Pages are UTF-8 with a BOM; the first tag of every page is ``<base href="<source URL>"/>``, often
followed by a stale ``<meta charset>`` from the original crawl (the BOM is authoritative).

Ground-truth files are UTF-8 with a BOM and CRLF line ends. Line 1 is
``vertical<TAB>site<TAB>attribute``; line 2 holds four statistics (pages, pages with values,
values, unique values); every further line is one page:
``<page id><TAB><value count><TAB><value>...``, with the single value ``<NULL>`` when the count is
0. Values are the raw source text of DOM text nodes, so HTML entities (``&amp;``, ``&nbsp;``,
``&#150;``) are still encoded; :func:`decode_value` turns them into display text.

Split (frozen): per vertical, the two sites that sort first alphabetically are the development
slice (tuning allowed); the other eight are the test set. ``eval/datasets/swde/split.json``
records it and is hashed into the dataset manifest.

Page key = the page file stem (``"0001"``), i.e. its path relative to the site directory minus the
``.htm`` suffix.
"""

from __future__ import annotations

import html
import os
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from ditto.evaluation.datasets.base import (
    DatasetUnavailable,
    EvalSite,
    Selection,
    Split,
    select_sites,
)
from ditto.parallel import pmap

DATASET = "swde"
DEFAULT_DATA_ROOT = Path("data/swde")
DEFAULT_FIELDS_DIR = Path("eval/datasets/swde/fields")
DOWNLOAD_HINT = "run `uv run python eval/datasets/swde/download.py`"
DEV_SITES_PER_VERTICAL = 2
NULL_VALUE = "<NULL>"
PAGE_SUFFIX = ".htm"
URL_HOST_SUFFIX = ".swde.invalid"
URL_INDEX_DIR = "urls"

VERTICALS: tuple[str, ...] = (
    "auto",
    "book",
    "camera",
    "job",
    "movie",
    "nbaplayer",
    "restaurant",
    "university",
)

ATTRIBUTES: dict[str, tuple[str, ...]] = {
    "auto": ("model", "price", "engine", "fuel_economy"),
    "book": ("title", "author", "isbn_13", "publisher", "publication_date"),
    "camera": ("model", "price", "manufacturer"),
    "job": ("title", "company", "location", "date_posted"),
    "movie": ("title", "director", "genre", "mpaa_rating"),
    "nbaplayer": ("name", "team", "height", "weight"),
    "restaurant": ("name", "address", "phone", "cuisine"),
    "university": ("name", "phone", "website", "type"),
}

SITES: dict[str, tuple[str, ...]] = {
    "auto": (
        "aol",
        "autobytel",
        "automotive",
        "autoweb",
        "carquotes",
        "cars",
        "kbb",
        "motortrend",
        "msn",
        "yahoo",
    ),
    "book": (
        "abebooks",
        "amazon",
        "barnesandnoble",
        "bookdepository",
        "booksamillion",
        "borders",
        "buy",
        "christianbook",
        "deepdiscount",
        "waterstones",
    ),
    "camera": (
        "amazon",
        "beachaudio",
        "buy",
        "compsource",
        "ecost",
        "jr",
        "newegg",
        "onsale",
        "pcnation",
        "thenerds",
    ),
    "job": (
        "careerbuilder",
        "dice",
        "hotjobs",
        "job",
        "jobcircle",
        "jobtarget",
        "monster",
        "nettemps",
        "rightitjobs",
        "techcentric",
    ),
    "movie": (
        "allmovie",
        "amctv",
        "boxofficemojo",
        "hollywood",
        "iheartmovies",
        "imdb",
        "metacritic",
        "msn",
        "rottentomatoes",
        "yahoo",
    ),
    "nbaplayer": (
        "espn",
        "fanhouse",
        "foxsports",
        "msnca",
        "nba",
        "si",
        "slam",
        "usatoday",
        "wiki",
        "yahoo",
    ),
    "restaurant": (
        "fodors",
        "frommers",
        "gayot",
        "opentable",
        "pickarestaurant",
        "restaurantica",
        "tripadvisor",
        "urbanspoon",
        "usdiners",
        "zagat",
    ),
    "university": (
        "collegeboard",
        "collegenavigator",
        "collegeprowler",
        "collegetoolkit",
        "ecampustours",
        "embark",
        "matchcollege",
        "princetonreview",
        "studentaid",
        "usnews",
    ),
}

_SITE_DIR_RE = re.compile(r"^(?P<vertical>[a-z]+)-(?P<site>[a-z0-9]+)\((?P<n>\d+)\)$")
_PAGE_ID_RE = re.compile(r"^\d{4}$")
_BASE_HREF_RE = re.compile(rb'^(?:\xef\xbb\xbf)?\s*<base\s+href\s*=\s*"([^"]*)"', re.IGNORECASE)

PageLabels = dict[str, dict[str, list[str]]]
"""Page key -> attribute -> values (``[]`` when the page lacks the attribute)."""


class GroundTruthError(ValueError):
    """A ground-truth file or line does not follow the SWDE format."""


class LayoutError(RuntimeError):
    """The extracted dataset does not have the expected structure."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        shown = "; ".join(problems[:20])
        more = f" (+{len(problems) - 20} more)" if len(problems) > 20 else ""
        super().__init__(f"SWDE layout check failed: {shown}{more}")


# --------------------------------------------------------------------------- ground truth


@dataclass(frozen=True)
class GroundTruthStats:
    """Line 2 of a ground-truth file."""

    pages: int
    pages_with_values: int
    values: int
    unique_values: int


@dataclass(frozen=True)
class GroundTruth:
    """One parsed ``<vertical>-<site>-<attribute>.txt`` file (values raw, entities encoded)."""

    vertical: str
    site: str
    attribute: str
    stats: GroundTruthStats
    values: dict[str, list[str]]


def parse_groundtruth_line(line: str) -> tuple[str, list[str]]:
    """Parse one page line: ``(page_id, values)``; a count of 0 gives ``[]``."""
    parts = line.rstrip("\r\n").split("\t")
    if len(parts) < 2:
        raise GroundTruthError(f"page line needs an id and a count: {line!r}")
    page_id, count_text = parts[0], parts[1]
    if not _PAGE_ID_RE.match(page_id):
        raise GroundTruthError(f"bad page id {page_id!r} in {line!r}")
    try:
        count = int(count_text)
    except ValueError:
        raise GroundTruthError(f"bad value count {count_text!r} in {line!r}") from None
    values = parts[2:]
    if count == 0:
        if values not in ([], [NULL_VALUE]):
            raise GroundTruthError(f"count 0 but values present in {line!r}")
        return page_id, []
    if count < 0 or len(values) != count:
        raise GroundTruthError(f"count {count} but {len(values)} values in {line!r}")
    if NULL_VALUE in values:
        raise GroundTruthError(f"{NULL_VALUE} among {count} values in {line!r}")
    return page_id, values


def parse_groundtruth(text: str) -> GroundTruth:
    """Parse a whole ground-truth file and check it against its own statistics line."""
    lines = text.lstrip("\ufeff").splitlines()
    if len(lines) < 2:
        raise GroundTruthError("ground-truth file needs a header and a statistics line")
    header = lines[0].split("\t")
    if len(header) != 3:
        raise GroundTruthError(f"header must be vertical, site, attribute: {lines[0]!r}")
    try:
        stats = GroundTruthStats(*(int(x) for x in lines[1].split("\t")))
    except (TypeError, ValueError):
        raise GroundTruthError(f"statistics line must hold 4 integers: {lines[1]!r}") from None
    values: dict[str, list[str]] = {}
    for line in lines[2:]:
        if not line:
            continue
        page_id, page_values = parse_groundtruth_line(line)
        if page_id in values:
            raise GroundTruthError(f"duplicate page id {page_id}")
        values[page_id] = page_values
    found = GroundTruthStats(
        pages=len(values),
        pages_with_values=sum(1 for v in values.values() if v),
        values=sum(len(v) for v in values.values()),
        unique_values=len({x for v in values.values() for x in v}),
    )
    if found != stats:
        raise GroundTruthError(f"statistics line {stats} disagrees with the page lines {found}")
    return GroundTruth(header[0], header[1], header[2], stats, values)


def read_groundtruth(path: Path) -> GroundTruth:
    """Read and parse a ground-truth file (UTF-8, BOM optional)."""
    try:
        return parse_groundtruth(path.read_bytes().decode("utf-8-sig"))
    except GroundTruthError as exc:
        raise GroundTruthError(f"{path}: {exc}") from None


def decode_value(raw: str) -> str:
    """Display text of a raw label value: HTML entities decoded per HTML5 (``&nbsp;`` -> NBSP)."""
    return html.unescape(raw)


# --------------------------------------------------------------------------- sites and split


@dataclass(frozen=True, order=True)
class SiteRef:
    vertical: str
    site: str

    @property
    def name(self) -> str:
        """Unique site name across verticals, e.g. ``auto-aol``."""
        return f"{self.vertical}-{self.site}"


def all_sites() -> list[SiteRef]:
    return [SiteRef(v, s) for v in VERTICALS for s in sorted(SITES[v])]


def split_of(ref: SiteRef) -> Split:
    """``"dev"`` for the two alphabetically-first sites of a vertical, else ``"test"``."""
    return "dev" if ref.site in sorted(SITES[ref.vertical])[:DEV_SITES_PER_VERTICAL] else "test"


def sites(split: Split = "all") -> list[SiteRef]:
    """Sites of a split (``dev``, ``test`` or ``all``), ordered by vertical then site."""
    if split not in ("dev", "test", "all"):
        raise ValueError(f"unknown split {split!r}")
    return [r for r in all_sites() if split == "all" or split_of(r) == split]


def split_document() -> dict[str, Any]:
    """The content of ``split.json``."""
    return {
        "dataset": DATASET,
        "rule": (
            f"per vertical, the {DEV_SITES_PER_VERTICAL} sites that sort first alphabetically "
            "form the development slice (tuning allowed); the other sites are the frozen test set"
        ),
        "dev": {v: [r.site for r in sites("dev") if r.vertical == v] for v in VERTICALS},
        "test": {v: [r.site for r in sites("test") if r.vertical == v] for v in VERTICALS},
    }


def ref_for_name(name: str) -> SiteRef:
    """Parse ``auto-aol`` into a :class:`SiteRef` and check it exists."""
    vertical, _, site = name.partition("-")
    if vertical not in SITES or site not in SITES[vertical]:
        raise KeyError(f"unknown SWDE site {name!r}")
    return SiteRef(vertical, site)


# --------------------------------------------------------------------------- layout


def site_dir(root: Path, vertical: str, site: str) -> Path:
    """``root/<vertical>/<vertical>-<site>(<n>)``; raises ``FileNotFoundError`` if absent."""
    parent = root / vertical
    if parent.is_dir():
        for child in parent.iterdir():
            m = _SITE_DIR_RE.match(child.name)
            if m and m["vertical"] == vertical and m["site"] == site and child.is_dir():
                return child
    raise FileNotFoundError(f"no SWDE site directory for {vertical}-{site} under {parent}")


def declared_pages(path: Path) -> int:
    """Page count encoded in a site directory name, e.g. ``camera-amazon(1767)`` -> 1767."""
    m = _SITE_DIR_RE.match(path.name)
    if not m:
        raise ValueError(f"not a SWDE site directory name: {path.name!r}")
    return int(m["n"])


def groundtruth_path(root: Path, vertical: str, site: str, attribute: str) -> Path:
    return root / "groundtruth" / vertical / f"{vertical}-{site}-{attribute}.txt"


def page_paths(directory: Path) -> list[Path]:
    """Page files of a site directory, sorted by name (``scandir`` file types, no per-file stat)."""
    with os.scandir(directory) as entries:
        names = sorted(e.name for e in entries if e.name.endswith(PAGE_SUFFIX) and e.is_file())
    return [directory / name for name in names]


def page_key(path: Path) -> str:
    """Page key of a page file: its stem (``0001``), which is also its ground-truth page id."""
    return path.stem


def source_url(head: bytes) -> str | None:
    """Source URL from the leading ``<base href="...">`` tag of a page (first bytes suffice)."""
    m = _BASE_HREF_RE.match(head)
    return html.unescape(m.group(1).decode("utf-8", errors="replace")).strip() if m else None


def fallback_url(ref: SiteRef, key: str) -> str:
    """Stable placeholder URL for a page without a usable source URL."""
    return f"https://{ref.site}{URL_HOST_SUFFIX}/{ref.vertical}/{key}{PAGE_SUFFIX}"


def page_urls(ref: SiteRef, paths: Iterable[Path]) -> dict[str, str]:
    """Page key -> URL: the page's source URL when present and unique in the site, else the
    :func:`fallback_url` (so ingest never merges two pages under one URL)."""
    found: dict[str, str | None] = {}
    for path in paths:
        with path.open("rb") as fh:
            found[page_key(path)] = source_url(fh.read(4096))
    counts: dict[str, int] = {}
    for url in found.values():
        if url:
            counts[url] = counts.get(url, 0) + 1
    return {
        key: url if url and counts[url] == 1 else fallback_url(ref, key)
        for key, url in found.items()
    }


def load_labels(root: Path, vertical: str, site: str, *, decode: bool = True) -> PageLabels:
    """Labels of one site: page key -> attribute -> values (``[]`` where absent)."""
    labels: PageLabels = {}
    for attribute in ATTRIBUTES[vertical]:
        gt = read_groundtruth(groundtruth_path(root, vertical, site, attribute))
        if (gt.vertical, gt.site, gt.attribute) != (vertical, site, attribute):
            raise GroundTruthError(
                f"header {gt.vertical}/{gt.site}/{gt.attribute} != {vertical}/{site}/{attribute}"
            )
        for key, values in gt.values.items():
            labels.setdefault(key, {})[attribute] = (
                [decode_value(v) for v in values] if decode else list(values)
            )
    return labels


def load_site(
    vertical: str, site: str, *, root: Path = DEFAULT_DATA_ROOT, decode: bool = True
) -> tuple[Path, PageLabels]:
    """``(input_dir, labels)`` for one site; ``input_dir`` is what ditto ingests."""
    return site_dir(root, vertical, site), load_labels(root, vertical, site, decode=decode)


@dataclass
class SiteScan:
    vertical: str
    site: str
    directory: str
    declared_pages: int
    pages: int
    attributes: dict[str, dict[str, int]] = field(default_factory=dict)


def scan_layout(root: Path) -> list[SiteScan]:
    """Verify the extracted tree by structure and return per-site counts.

    Checks: exactly the 8 verticals with 10 site directories each, named
    ``<vertical>-<site>(<n>)`` with ``n`` pages ``0000.htm``..; one ground-truth file per
    (vertical, site, attribute) and no others; every ground-truth file parses, matches its own
    statistics line and covers exactly the site's page ids. Raises :class:`LayoutError` listing
    every problem.
    """
    problems: list[str] = []
    scans: list[SiteScan] = []
    found_verticals = (
        sorted(p.name for p in root.iterdir() if p.is_dir() and p.name in VERTICALS)
        if root.is_dir()
        else []
    )
    if found_verticals != sorted(VERTICALS):
        problems.append(f"verticals {found_verticals} != {sorted(VERTICALS)}")
    gt_root = root / "groundtruth"
    for vertical in VERTICALS:
        vdir = root / vertical
        if not vdir.is_dir():
            continue
        names = sorted(
            m["site"] for p in vdir.iterdir() if (m := _SITE_DIR_RE.match(p.name)) and p.is_dir()
        )
        if names != sorted(SITES[vertical]):
            problems.append(f"{vertical}: sites {names} != {sorted(SITES[vertical])}")
        expected_gt = {
            groundtruth_path(root, vertical, s, a).name
            for s in SITES[vertical]
            for a in ATTRIBUTES[vertical]
        }
        gt_dir = gt_root / vertical
        actual_gt = {p.name for p in gt_dir.glob("*.txt")} if gt_dir.is_dir() else set()
        for name in sorted(actual_gt ^ expected_gt):
            kind = "missing" if name in expected_gt else "unexpected"
            problems.append(f"{kind} ground-truth file {vertical}/{name}")
        for site in sorted(SITES[vertical]):
            try:
                sdir = site_dir(root, vertical, site)
            except FileNotFoundError as exc:
                problems.append(str(exc))
                continue
            keys = [page_key(p) for p in page_paths(sdir)]
            scan = SiteScan(vertical, site, sdir.name, declared_pages(sdir), len(keys))
            if keys != [f"{i:04d}" for i in range(len(keys))]:
                problems.append(f"{sdir.name}: page files are not 0000..{len(keys) - 1:04d}")
            if scan.pages != scan.declared_pages:
                problems.append(f"{sdir.name}: {scan.pages} pages, name says {scan.declared_pages}")
            for attribute in ATTRIBUTES[vertical]:
                path = groundtruth_path(root, vertical, site, attribute)
                if not path.is_file():
                    continue
                try:
                    gt = read_groundtruth(path)
                except GroundTruthError as exc:
                    problems.append(str(exc))
                    continue
                if (gt.vertical, gt.site, gt.attribute) != (vertical, site, attribute):
                    problems.append(f"{path.name}: header {gt.vertical}/{gt.site}/{gt.attribute}")
                if sorted(gt.values) != keys:
                    problems.append(f"{path.name}: page ids differ from the page files")
                scan.attributes[attribute] = {
                    "pages_with_values": gt.stats.pages_with_values,
                    "values": gt.stats.values,
                    "unique_values": gt.stats.unique_values,
                    "multi_value_pages": sum(1 for v in gt.values.values() if len(v) > 1),
                }
            scans.append(scan)
    if problems:
        raise LayoutError(problems)
    return scans


# --------------------------------------------------------------------------- eval adapter


def fields_path(vertical: str, fields_dir: Path = DEFAULT_FIELDS_DIR) -> Path:
    """Prompted-mode ``fields.yaml`` of a vertical."""
    return fields_dir / f"{vertical}.yaml"


def write_urls_tsv(path: Path, urls: dict[str, str]) -> None:
    """``urls.tsv`` for ingest: ``<NNNN>.htm<TAB><url>`` per page, path relative to the site dir."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{key}{PAGE_SUFFIX}\t{url}\n" for key, url in sorted(urls.items())]
    path.write_text("".join(lines), encoding="utf-8")


def read_urls_tsv(path: Path) -> dict[str, str]:
    """Page key -> URL from a ``urls.tsv`` written by :func:`write_urls_tsv`."""
    urls: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        rel, sep, url = line.partition("\t")
        if sep:
            urls[page_key(Path(rel))] = url
    return urls


def url_index_path(root: Path, ref: SiteRef) -> Path:
    """Precomputed ``urls.tsv`` of a site (``data/swde/urls/<vertical>-<site>.tsv``)."""
    return root / URL_INDEX_DIR / f"{ref.name}.tsv"


def _index_sites(chunk: Sequence[tuple[str, str, str]]) -> int:
    pages = 0
    for root_text, vertical, site in chunk:
        root, ref = Path(root_text), SiteRef(vertical, site)
        urls = page_urls(ref, page_paths(site_dir(root, vertical, site)))
        write_urls_tsv(url_index_path(root, ref), urls)
        pages += len(urls)
    return pages


def build_url_index(root: Path, refs: Sequence[SiteRef], *, workers: int | None = None) -> int:
    """Read every page's source URL once and write the per-site URL index; returns pages."""
    items = [(str(root), r.vertical, r.site) for r in refs]
    return sum(pmap(_index_sites, items, workers=workers, chunk_size=1))


def is_fallback_url(url: str) -> bool:
    return (urlsplit(url).hostname or "").endswith(URL_HOST_SUFFIX)


def site_urls(root: Path, ref: SiteRef, paths: Sequence[Path]) -> dict[str, str]:
    """Page key -> URL for a site: the URL index when it covers exactly the page files, else
    read from the pages."""
    index = url_index_path(root, ref)
    if index.is_file():
        urls = read_urls_tsv(index)
        if sorted(urls) == [page_key(p) for p in paths]:
            return urls
    return page_urls(ref, paths)


def load_sites(selection: Selection, work_dir: Path) -> list[EvalSite]:
    """Evaluation sites for the harness (``ditto.evaluation.datasets.base`` protocol).

    Options (``selection.options``): ``data_root`` (default ``data/swde``), ``fields_dir``
    (default ``eval/datasets/swde/fields``), ``decode_entities`` (default true). Site names are
    ``<vertical>-<site>``. Each site's ``urls.tsv`` is written under
    ``work_dir/swde/<vertical>-<site>/``.
    """
    opts = selection.options
    root = Path(opts.get("data_root", DEFAULT_DATA_ROOT))
    fields_dir = Path(opts.get("fields_dir", DEFAULT_FIELDS_DIR))
    decode = bool(opts.get("decode_entities", True))
    if not (root / "groundtruth").is_dir():
        raise DatasetUnavailable(f"SWDE not found under {root}: {DOWNLOAD_HINT}")
    refs = {r.name: r for r in sites(selection.split)}
    stubs = [EvalSite(r.name, r.vertical, [], "dir", None, {}, {}, meta={}) for r in refs.values()]
    chosen: list[EvalSite] = []
    for stub in select_sites(stubs, selection):
        ref = refs[stub.name]
        try:
            input_dir, labels = load_site(ref.vertical, ref.site, root=root, decode=decode)
        except FileNotFoundError as exc:
            raise DatasetUnavailable(f"{exc}: {DOWNLOAD_HINT}") from None
        paths = page_paths(input_dir)
        urls = site_urls(root, ref, paths)
        manifest = work_dir / DATASET / ref.name / "urls.tsv"
        write_urls_tsv(manifest, urls)
        schema = fields_path(ref.vertical, fields_dir)
        chosen.append(
            EvalSite(
                name=ref.name,
                vertical=ref.vertical,
                input_paths=[input_dir],
                input_kind="dir",
                fields_schema_path=schema if schema.is_file() else None,
                labels=dict(labels),
                url_to_key={url: key for key, url in urls.items()},
                manifest=manifest,
                meta={
                    "dataset": DATASET,
                    "site": ref.site,
                    "split": split_of(ref),
                    "pages": len(paths),
                    "attributes": list(ATTRIBUTES[ref.vertical]),
                    "fallback_urls": sum(1 for u in urls.values() if is_fallback_url(u)),
                    "entities_decoded": decode,
                },
            )
        )
    return chosen
