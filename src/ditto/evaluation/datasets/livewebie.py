"""LiveWeb-IE evaluation adapter (third-party modern check).

LiveWeb-IE: Yang, Kim, Choi, Kim, Yang, Park, Choo. "LiveWeb-IE: A Benchmark For Online Web
Information Extraction", ICLR 2026, arXiv 2603.13773. Release: https://github.com/sbY99/LiveWeb-IE
(non-commercial research use only; cite the paper; redistribution only with the license terms).

The release holds 342 natural-language queries over 15 sites and 46 page groups
(``dataset.json``), the URLs of each group (``<site>/group/group.jsonl``) and one ground-truth
file per (site, group, attribute) (``groundtruth/<domain>/<website>-<group>-<attribute>.txt``).
The paper evaluates on the live sites; ditto evaluates on a frozen snapshot of the referenced
pages written by ``eval/datasets/livewebie/download.py`` into ``data/livewebie/<website>.warc.gz``,
so numbers are not directly comparable with the paper's.

Mapping onto the evaluation harness:

* one :class:`EvalSite` per website; page groups are the true templates (``truth_templates``);
* a query's complexity type fixes the cardinality of its attributes: ``type_1``/``type_2`` ask
  for single values (field kind ``scalar``), ``type_3``/``type_4`` for lists (kind ``list``).
  When one site uses an attribute name both ways (a paper's ``title`` on detail pages, all
  ``title`` values on listing pages), the list-valued field is renamed ``<attribute>_list``;
* each page is labeled only with the fields annotated for its group; a group's field missing
  from a page's ground truth is left out (not scored), a ground-truth count of 0 is an empty
  label (``None`` / ``[]``);
* ``field_groups`` lists, per complexity type, the fields its queries ask for.

Ground-truth file format (verified against every file of the pinned release)::

    <domain> <website> <group_id> <attribute label>
    <n_pages> <n_values> <n_values>
    "<url>" <count> <value>\\t<value>...

Values are raw page text, ``@href`` for links and ``@src`` for images, and may contain
newlines, so a record runs until the next line that starts with ``"<url>" <count>``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from ditto.evaluation.datasets.base import (
    DatasetUnavailable,
    EvalSite,
    LabelKind,
    Selection,
    select_sites,
)

RELEASE_REPO = "sbY99/LiveWeb-IE"
RELEASE_COMMIT = "34b56496f221e41ae44d137b7c42e38e57702c70"
COMPLEXITY_TYPES: tuple[str, ...] = ("type_1", "type_2", "type_3", "type_4")
LIST_TYPES = frozenset({"type_3", "type_4"})
LIST_SUFFIX = "_list"

DEFAULT_DATA_DIR = Path("data/livewebie")
DEFAULT_FIELDS_DIR = Path("eval/datasets/livewebie/fields")
DEFAULT_MANIFEST = Path("eval/datasets/livewebie/manifest.json")

LICENSE_TERMS = (
    "The LiveWeb-IE benchmark is released strictly for non-commercial research purposes only. "
    "By downloading or using this dataset, you agree to the following terms: "
    "1. Non-Commercial Use: You may not use this dataset for any commercial purposes, including "
    "but not limited to training commercial models or enhancing commercial services. "
    "2. Attribution: Any publication or report using this dataset must cite the original paper "
    "(LiveWeb-IE: A Benchmark For Online Web Information Extraction, ICLR 2026). "
    "3. No Redistribution without License: You may not redistribute this dataset without "
    "including this license agreement. "
    "This restriction complies with the ethical agreements made with the website providers "
    "included in this benchmark."
)
CITATION = (
    "Seungbin Yang, Jihwan Kim, Jaemin Choi, Dongjin Kim, Soyoung Yang, ChaeHun Park, Jaegul "
    "Choo. LiveWeb-IE: A Benchmark For Online Web Information Extraction. ICLR 2026. "
    "arXiv:2603.13773. https://openreview.net/forum?id=bIODmQ1TNd"
)

Cardinality = Literal["scalar", "list"]
Category = Literal["text", "link", "image"]

_RECORD_START = re.compile(r'^"(https?://[^"]*)" (\d+)(?: |$)')
_IMAGE_TOKENS = frozenset({"image", "img", "badge", "logo", "poster", "fanart", "banner", "thumb"})


@dataclass(frozen=True)
class Query:
    """One row of ``dataset.json``."""

    sample_id: str
    website: str
    group_id: str
    query_id: str
    query: str
    attributes: tuple[str, ...]
    type: str
    labels: dict[str, str]
    """Attribute -> ground-truth path relative to the release root."""

    @property
    def cardinality(self) -> Cardinality:
        return "list" if self.type in LIST_TYPES else "scalar"


@dataclass
class GroundTruth:
    """One parsed ground-truth file: ``values[url]`` is the page's value list (empty if absent)."""

    domain: str
    website: str
    group_id: str
    label: str
    values: dict[str, list[str]]


@dataclass
class Release:
    """The parsed LiveWeb-IE release."""

    root: Path
    queries: list[Query]
    groups: dict[tuple[str, str], list[str]]
    """(website, group_id) -> page URLs in release order, duplicates removed."""
    domains: dict[str, str]
    """Website -> domain (the paper's vertical: academic, auto, book, ...)."""

    def websites(self) -> list[str]:
        seen: dict[str, None] = {}
        for q in self.queries:
            seen.setdefault(q.website, None)
        return list(seen)

    def site_queries(self, website: str) -> list[Query]:
        return [q for q in self.queries if q.website == website]

    def site_urls(self, website: str) -> list[str]:
        """Every page URL of a website over all its groups, first occurrence wins."""
        seen: dict[str, None] = {}
        for (site, _gid), urls in self.groups.items():
            if site == website:
                for url in urls:
                    seen.setdefault(url, None)
        return list(seen)

    def label_path(self, website: str, group_id: str, attribute: str) -> Path:
        for q in self.queries:
            if q.website == website and q.group_id == group_id and attribute in q.labels:
                return self.root / q.labels[attribute]
        raise KeyError((website, group_id, attribute))


@dataclass
class FieldDef:
    """One site-level field: an attribute of one cardinality over one or more groups."""

    name: str
    attribute: str
    kind: Cardinality
    category: Category
    groups: list[str] = field(default_factory=list)
    queries: list[Query] = field(default_factory=list)

    def description(self) -> str:
        """Prompted-mode description translated from the dataset's own queries."""
        single = _distinct(q.query for q in self.queries if q.attributes == (self.attribute,))
        if single:
            text = " / ".join(_sentence(s) for s in single[:2])
        else:
            first = self.queries[0].query.strip()
            text = f'{_humanize(self.attribute)} (asked for in the query "{first}").'
        if self.category == "image":
            text += " Value: the image URL exactly as written in the img src attribute."
        elif self.category == "link":
            text += " Value: the URL exactly as written in the a href attribute."
        if self.kind == "list":
            text += " One entry per item on the page."
        return text


def parse_groundtruth(text: str) -> GroundTruth:
    """Parse one ground-truth file; see the module docstring for the format."""
    lines = text.split("\n")
    if len(lines) < 2:
        raise ValueError("ground-truth file needs a header and a count line")
    header = lines[0].split(" ")
    if len(header) < 4:
        raise ValueError(f"bad ground-truth header: {lines[0]!r}")
    counts = [int(x) for x in lines[1].split()]
    records: list[tuple[str, int, list[str]]] = []
    for line in lines[2:]:
        m = _RECORD_START.match(line)
        if m:
            records.append((m.group(1), int(m.group(2)), [line[m.end() :]]))
        elif records:
            records[-1][2].append(line)
        elif line.strip():
            raise ValueError(f"ground-truth line before the first record: {line!r}")
    values: dict[str, list[str]] = {}
    for url, count, parts in records:
        body = "\n".join(parts).rstrip("\n")
        vals = body.split("\t") if count else []
        if len(vals) != count:
            raise ValueError(f"{url}: count {count} but {len(vals)} values")
        values[url] = vals
    if counts and counts[0] != len(records):
        raise ValueError(f"header says {counts[0]} pages, found {len(records)}")
    return GroundTruth(
        domain=header[0],
        website=header[1],
        group_id=header[2],
        label=" ".join(header[3:]),
        values=values,
    )


def load_release(root: Path) -> Release:
    """Load ``dataset.json`` and the group URL lists from an extracted release (``LiveWeb_IE``)."""
    dataset_path = root / "dataset.json"
    if not dataset_path.is_file():
        raise DatasetUnavailable(f"LiveWeb-IE release not found at {root}")
    raw = json.loads(dataset_path.read_text(encoding="utf-8"))
    queries: list[Query] = []
    group_files: dict[str, str] = {}
    domains: dict[str, str] = {}
    for rows in raw.values():
        for row in rows:
            labels = {a: str(p).removeprefix("./") for a, p in row["label"].items()}
            q = Query(
                sample_id=row["sample_id"],
                website=row["website"],
                group_id=row["group_id"],
                query_id=row["query_id"],
                query=row["query"],
                attributes=tuple(row["attribute"]),
                type=row["type"],
                labels=labels,
            )
            if q.type not in COMPLEXITY_TYPES:
                raise ValueError(f"{q.sample_id}: unknown type {q.type!r}")
            if set(q.attributes) != set(labels):
                raise ValueError(f"{q.sample_id}: attributes and labels differ")
            queries.append(q)
            group_files[q.website] = str(row["group_url"]).removeprefix("./")
            for path in labels.values():
                # groundtruth/<domain>/<website>-<group>-<attribute>.txt
                domains.setdefault(q.website, Path(path).parent.name)
    groups: dict[tuple[str, str], list[str]] = {}
    for website, rel in group_files.items():
        for line in (root / rel).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            for gid, entry in json.loads(line).items():
                groups[(website, gid)] = list(dict.fromkeys(entry["group_url"]))
    return Release(root=root, queries=queries, groups=groups, domains=domains)


def site_fields(release: Release, website: str) -> list[FieldDef]:
    """Site-level fields in first-query order, list-valued homonyms renamed ``<attr>_list``."""
    kinds: dict[str, set[Cardinality]] = {}
    for q in release.site_queries(website):
        for a in q.attributes:
            kinds.setdefault(a, set()).add(q.cardinality)
    fields: dict[str, FieldDef] = {}
    for q in release.site_queries(website):
        for a in q.attributes:
            name = field_name(a, q.cardinality, both=len(kinds[a]) > 1)
            if name != a and name in kinds:
                raise ValueError(f"{website}: renamed field {name!r} collides with an attribute")
            fd = fields.get(name)
            if fd is None:
                fd = fields[name] = FieldDef(name, a, q.cardinality, attribute_category(a))
            if q.group_id not in fd.groups:
                fd.groups.append(q.group_id)
            fd.queries.append(q)
    return list(fields.values())


def field_name(attribute: str, kind: Cardinality, *, both: bool) -> str:
    return attribute + LIST_SUFFIX if both and kind == "list" else attribute


def attribute_category(attribute: str) -> Category:
    """Data category from the attribute name, the way the paper's annotators named them."""
    tokens = set(attribute.lower().split("_"))
    if tokens & _IMAGE_TOKENS:
        return "image"
    if "link" in tokens or "url" in tokens:
        return "link"
    return "text"


def site_labels(
    release: Release, website: str, fields: Sequence[FieldDef]
) -> dict[str, dict[str, Any]]:
    """``{url: {field: value}}`` for every page with ground truth for its group's fields."""
    labels: dict[str, dict[str, Any]] = {}
    for fd in fields:
        for gid in fd.groups:
            gt = parse_groundtruth(
                release.label_path(website, gid, fd.attribute).read_text(encoding="utf-8")
            )
            for url, vals in gt.values.items():
                labels.setdefault(url, {})[fd.name] = _label_value(vals, fd.kind)
    return labels


def site_templates(release: Release, website: str) -> dict[str, str]:
    """``{url: group_id}``; a URL listed in two groups keeps the first."""
    out: dict[str, str] = {}
    for (site, gid), urls in release.groups.items():
        if site == website:
            for url in urls:
                out.setdefault(url, gid)
    return out


def complexity_groups(
    release: Release, website: str, fields: Sequence[FieldDef]
) -> dict[str, list[str]]:
    """Complexity type -> names of the fields asked for by that type's queries on the site."""
    out: dict[str, list[str]] = {}
    for t in COMPLEXITY_TYPES:
        names = [fd.name for fd in fields if any(q.type == t for q in fd.queries)]
        if names:
            out[t] = names
    return out


def load_sites(selection: Selection, work_dir: Path) -> list[EvalSite]:
    """Harness entry point: one :class:`EvalSite` per usable LiveWeb-IE website.

    Options (``dataset_options``): ``data_dir`` (snapshot + extracted release, default
    ``data/livewebie``), ``fields_dir`` (default ``eval/datasets/livewebie/fields``),
    ``manifest`` (default ``eval/datasets/livewebie/manifest.json``). LiveWeb-IE is a test-only
    set: ``split="dev"`` selects nothing.
    """
    del work_dir  # fields.yaml files are committed and frozen; nothing to derive
    opts = selection.options
    data_dir = _resolve(Path(opts.get("data_dir", DEFAULT_DATA_DIR)))
    fields_dir = _resolve(Path(opts.get("fields_dir", DEFAULT_FIELDS_DIR)))
    manifest_path = _resolve(Path(opts.get("manifest", DEFAULT_MANIFEST)))
    if selection.split == "dev":
        return []
    if not manifest_path.is_file():
        raise DatasetUnavailable(f"LiveWeb-IE manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") == "skipped":
        raise DatasetUnavailable(f"LiveWeb-IE skipped: {manifest.get('reason', '')}")
    release = load_release(data_dir / "release" / "LiveWeb_IE")
    snapshot_sites: dict[str, Any] = manifest.get("sites", {})
    sites: list[EvalSite] = []
    for website in release.websites():
        info = snapshot_sites.get(website, {})
        if not info.get("included", False):
            continue
        warc = data_dir / f"{website}.warc.gz"
        if not warc.is_file():
            raise DatasetUnavailable(
                f"LiveWeb-IE snapshot missing: {warc} (run eval/datasets/livewebie/download.py)"
            )
        fields = site_fields(release, website)
        fields_path = fields_dir / f"{website}.yaml"
        if not fields_path.is_file():
            raise DatasetUnavailable(f"LiveWeb-IE fields.yaml missing: {fields_path}")
        labels = site_labels(release, website, fields)
        url_to_key = {url: url for url in labels}
        url_to_key.update(_redirect_keys(data_dir / "fetchlog" / f"{website}.jsonl", labels))
        kinds: dict[str, LabelKind] = {fd.name: fd.kind for fd in fields}
        sites.append(
            EvalSite(
                name=website,
                vertical=release.domains[website],
                input_paths=[warc],
                input_kind="warc",
                fields_schema_path=fields_path,
                labels=labels,
                url_to_key=url_to_key,
                field_kinds=kinds,
                truth_templates=site_templates(release, website),
                field_groups=complexity_groups(release, website, fields),
                meta={
                    "dataset": "livewebie",
                    "release": f"{RELEASE_REPO}@{RELEASE_COMMIT}",
                    "snapshot": info,
                    "queries": [_query_meta(q, fields) for q in release.site_queries(website)],
                },
            )
        )
    return select_sites(sites, selection)


def fetch_log_entries(path: Path) -> list[dict[str, Any]]:
    """Rows of a snapshot fetch log (``data/livewebie/fetchlog/<website>.jsonl``)."""
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _redirect_keys(log_path: Path, labels: dict[str, Any]) -> dict[str, str]:
    """Final (post-redirect) URL -> benchmark URL for pages whose fetch was redirected."""
    out: dict[str, str] = {}
    for row in fetch_log_entries(log_path):
        url, final = row.get("url"), row.get("final_url")
        if isinstance(url, str) and isinstance(final, str) and final != url and url in labels:
            out.setdefault(final, url)
    return out


def _query_meta(q: Query, fields: Sequence[FieldDef]) -> dict[str, Any]:
    names = {(fd.attribute, fd.kind): fd.name for fd in fields}
    return {
        "sample_id": q.sample_id,
        "group_id": q.group_id,
        "type": q.type,
        "query": q.query,
        "fields": [names[(a, q.cardinality)] for a in q.attributes],
    }


def _label_value(vals: list[str], kind: Cardinality) -> Any:
    if kind == "list":
        return list(vals)
    if not vals:
        return None
    return vals[0] if len(vals) == 1 else list(vals)


def _resolve(path: Path) -> Path:
    """Absolute paths as given; relative ones against the cwd, else the repository root."""
    if path.is_absolute() or path.exists():
        return path
    repo = Path(__file__).resolve().parents[4]
    candidate = repo / path
    return candidate if candidate.exists() else path


def _distinct(items: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(s.strip() for s in items if s.strip()))


def _sentence(text: str) -> str:
    text = text.strip()
    return text if text.endswith((".", "?", "!")) else text + "."


def _humanize(attribute: str) -> str:
    return attribute.replace("_", " ").capitalize()
