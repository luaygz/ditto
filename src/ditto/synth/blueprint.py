"""Template blueprints: the schema, layout and class names of one synthetic page template.

A :class:`TemplateDef` is built once per template from a seeded RNG. Class names are resolved per
*slot* (``"root"``, ``"wrap:price"``, ``"label"``, ...) with an RNG derived from the template's
``class_seed`` and the slot name, so they are stable for the whole template build (like real
CSS-in-JS hashes) and independent of rendering order. A slot is hashed/utility-only with
probability ``hashed_share``; otherwise it gets a semantic class built from the template prefix.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Literal, TypeVar, cast

from ditto.fingerprint.dom import canonical_classes
from ditto.specgen.typecheck import ValueType
from ditto.synth import vocab as V
from ditto.synth.params import (
    ABSENT_RENDERINGS,
    KV_STYLES,
    MARKUP_STYLES,
    AbsentRendering,
    KVStyle,
    MarkupStyle,
    SynthParams,
)
from ditto.synth.values import CURRENCIES, DATE_FORMATS, TIME_FORMATS, ValueCtx

T = TypeVar("T", bound=str)
Naming = Literal["kebab", "bem", "camel"]
FieldScheme = Literal["wrapper", "value", "itemprop"]
FieldLayout = Literal["div", "p", "li", "row"]

UTILITY_TOKENS = (
    "flex",
    "items-center",
    "justify-between",
    "mt-2",
    "mb-4",
    "px-4",
    "py-2",
    "text-sm",
    "text-gray-700",
    "font-semibold",
    "rounded-lg",
    "shadow",
    "gap-2",
    "w-full",
    "grid",
    "grid-cols-2",
    "bg-white",
    "border",
    "leading-tight",
    "md:flex",
    "lg:w-1/2",
)
# Slots whose element often carries no class at all in real markup.
_BARE_SLOTS = ("item:", "th", "td", "col:", "kv_row", "related_item", "kv_key", "kv_val")
_ALNUM = "abcdefghijklmnopqrstuvwxyz0123456789"


@dataclass
class ScalarDef:
    name: str
    label: str
    gen: str
    vtype: ValueType
    desc: str
    tag: str = "span"
    optional: bool = False
    presence: float = 1.0
    group: str = "summary"  # summary | details | para


@dataclass
class ListDef:
    name: str
    heading: str
    gen: str
    vtype: ValueType
    desc: str
    ordered: bool = False
    link: bool = False


@dataclass
class ColumnDef:
    name: str
    label: str
    gen: str
    vtype: ValueType
    optional: bool = False
    presence: float = 1.0


@dataclass
class TableDef:
    name: str
    heading: str
    desc: str
    columns: list[ColumnDef]


@dataclass
class KVDef:
    name: str
    heading: str
    desc: str
    keys: list[tuple[str, str]]
    style: KVStyle
    key_tag: str = "th"  # table style: th or td for the key cell


@dataclass
class TemplateDef:
    tid: str
    index: int
    kind: str
    is_tail: bool
    segment: str
    url_style: str
    markup: MarkupStyle
    absent: AbsentRendering
    prefix: str
    naming: Naming
    field_scheme: FieldScheme
    field_layout: FieldLayout
    label_colon: bool
    root_tag: str
    react_depth: int
    react_h1: bool
    value_ctx: ValueCtx
    title: ScalarDef
    scalars: list[ScalarDef]
    lists: list[ListDef]
    tables: list[TableDef]
    kvs: list[KVDef]
    order: list[tuple[str, str]]
    related: bool
    related_heading: str
    class_seed: str
    hashed_share: float
    overrides: dict[str, str] = field(default_factory=dict)
    near_duplicate_of: str | None = None
    pages: int = 0
    _cache: dict[str, str] = field(default_factory=dict, repr=False, compare=False)

    # --- classes ------------------------------------------------------------------------------
    def cls(self, slot: str) -> str:
        """Class attribute value for ``slot`` (may be empty)."""
        cached = self._cache.get(slot)
        if cached is None:
            cached = self.overrides.get(slot)
            if cached is None:
                cached = _make_class(self, slot)
            self._cache[slot] = cached
        return cached

    def wrap_cls(self, name: str) -> str:
        """Class of field ``name``'s wrapper element (field-specific in the ``wrapper`` scheme)."""
        return self.cls(f"wrap:{name}" if self.field_scheme == "wrapper" else "wrap")

    def label_cls(self, name: str) -> str:
        """Class of field ``name``'s label; near-duplicate edits override ``label@<name>``."""
        return self.overrides.get(f"label@{name}") or self.cls("label")

    def value_cls(self, name: str) -> str:
        """Class of field ``name``'s value element (field-specific in the ``value`` scheme);
        near-duplicate edits override ``value@<name>``."""
        override = self.overrides.get(f"value@{name}")
        if override:
            return override
        return self.cls(f"val:{name}" if self.field_scheme == "value" else "value")

    # --- schema -------------------------------------------------------------------------------
    def optional_scalars(self) -> list[ScalarDef]:
        return [s for s in self.scalars if s.optional]

    def scalar(self, name: str) -> ScalarDef:
        return next(s for s in self.scalars if s.name == name)

    def field_names(self) -> list[str]:
        return (
            [self.title.name]
            + [s.name for s in self.scalars]
            + [x.name for x in self.lists]
            + [x.name for x in self.tables]
            + [x.name for x in self.kvs]
        )

    def clone(self) -> TemplateDef:
        out = copy.deepcopy(self)
        out._cache = {}
        return out


# --- class naming ---------------------------------------------------------------------------


def _camel(text: str) -> str:
    parts = [p for p in text.replace("_", "-").split("-") if p]
    return parts[0].lower() + "".join(p.capitalize() for p in parts[1:]) if parts else "x"


def _kebab(text: str) -> str:
    return text.replace("_", "-").lower()


def role_of(slot: str) -> str:
    """Semantic role word of a slot: ``wrap:price`` -> ``price``, ``item:tags`` -> ``tags-item``."""
    head, _, rest = slot.partition(":")
    if head in ("wrap", "val", "list", "kv", "table", "group"):
        return rest or head
    if head == "item":
        return f"{rest}-item"
    if head == "section":
        return f"{rest}-section"
    if head == "col":
        return rest.split(":", 1)[-1]
    if head == "empty":
        return f"{rest}-empty"
    if head == "inner":
        return "inner-" + rest.replace(":", "-")
    return {
        "wrap": "field",
        "kv_row": "row",
        "kv_key": "key",
        "kv_val": "val",
        "h2": "heading",
        "related_item": "related-item",
        "related_list": "related-list",
    }.get(slot, slot)


def ensure_semantic(token: str) -> str:
    """Make sure the class canonicalizer keeps ``token`` (no utility prefix, no hash look)."""
    if canonical_classes(token) != [token]:
        token = "s-" + token
    return token


def semantic_class(prefix: str, naming: Naming, role: str) -> str:
    if role == prefix:
        return ensure_semantic(prefix)
    if naming == "camel":
        return ensure_semantic(_camel(prefix) + _camel(role)[:1].upper() + _camel(role)[1:])
    if naming == "bem":
        return ensure_semantic(f"{_kebab(prefix)}__{_kebab(role)}")
    return ensure_semantic(f"{_kebab(prefix)}-{_kebab(role)}")


def _hash_chunk(rng: random.Random, n: int) -> str:
    chars = [rng.choice(_ALNUM) for _ in range(n)]
    chars[rng.randrange(n)] = rng.choice("0123456789")
    chars[rng.randrange(n)] = rng.choice("abcdefghijklmnopqrstuvwxyz")
    if chars[0].isdigit():  # keep at least one letter/digit mix inside the alnum run
        chars[0] = rng.choice("abcdefghijklmnopqrstuvwxyz")
    out = "".join(chars)
    if not (any(c.isdigit() for c in out) and any(c.isalpha() for c in out)):
        out = out[:-1] + "7"
    return out


def hashed_class(rng: random.Random, markup: MarkupStyle, role: str) -> str:
    """A generated or utility-only class string (every token is dropped by canonicalization)."""
    r = rng.random()
    if markup == "react":
        if r < 0.45:
            return f"css-{_hash_chunk(rng, 7)}"
        if r < 0.75:
            return f"sc-{_hash_chunk(rng, 6)} {_hash_chunk(rng, 6)}"
        word = _camel(role)
        return (
            f"{word[:1].upper()}{word[1:]}_{_kebab(role).replace('-', '')}__{_hash_chunk(rng, 5)}"
        )
    if r < 0.55:
        k = rng.randint(2, 4)
        return " ".join(rng.sample(UTILITY_TOKENS, k))
    if r < 0.8:
        return f"_{_hash_chunk(rng, 6)}"
    return f"jsx-{rng.randint(100000, 999999)}"


def _make_class(t: TemplateDef, slot: str) -> str:
    rng = random.Random(f"{t.class_seed}|{slot}")
    if slot.startswith(_BARE_SLOTS) and rng.random() < 0.5:
        return ""
    role = role_of(slot)
    if rng.random() < t.hashed_share:
        return hashed_class(rng, t.markup, role)
    if slot == "root":
        sem = semantic_class(t.prefix, t.naming, t.prefix)
    elif t.naming == "bem" and slot.startswith("wrap:"):
        base = semantic_class(t.prefix, "bem", "field")
        sem = f"{base} {ensure_semantic(f'{base}--{_kebab(role)}')}"
    else:
        sem = semantic_class(t.prefix, t.naming, role)
    if rng.random() < 0.25:
        sem += " " + " ".join(rng.sample(UTILITY_TOKENS, rng.randint(1, 2)))
    return sem


def class_is_hashed(value: str) -> bool:
    """True when a non-empty class attribute keeps no token after canonicalization."""
    return bool(value.split()) and not canonical_classes(value)


# --- building -------------------------------------------------------------------------------


@dataclass
class NameRegistry:
    """Site-wide uniqueness of class prefixes and URL segments."""

    prefixes: set[str] = field(default_factory=set)
    segments: set[str] = field(default_factory=set)

    def prefix(self, kind: V.KindProto, rng: random.Random) -> str:
        options = list(kind.prefixes)
        rng.shuffle(options)
        for p in options:
            if p not in self.prefixes:
                self.prefixes.add(p)
                return p
        mods = list(V.PREFIX_MODIFIERS)
        rng.shuffle(mods)
        for depth in range(1, 4):
            for base in options:
                for _ in range(len(mods)):
                    cand = "-".join([base, *rng.sample(mods, depth)])
                    if cand not in self.prefixes:
                        self.prefixes.add(cand)
                        return cand
        raise RuntimeError("ran out of class prefixes")

    def segment(self, kind: V.KindProto, rng: random.Random, skip_first: int = 0) -> str:
        options = list(kind.segments[skip_first:])
        mods = list(V.PREFIX_MODIFIERS)
        rng.shuffle(mods)
        options += [f"{kind.segments[0]}-{m}" for m in mods]
        options += [f"{kind.segments[0]}-{m}-{n}" for m in mods for n in mods if m != n]
        for cand in options:
            if cand not in self.segments:
                self.segments.add(cand)
                return cand
        raise RuntimeError("ran out of URL segments")


def _pick(rng: random.Random, choice: str, options: tuple[T, ...]) -> T:
    return rng.choice(options) if choice == "mixed" else cast(T, choice)


def _scalar(p: V.FieldProto, rng: random.Random, optional: bool, presence: float) -> ScalarDef:
    return ScalarDef(
        name=p.name,
        label=rng.choice(p.labels),
        gen=p.gen,
        vtype=p.vtype,
        desc=p.desc,
        tag=p.tag,
        optional=optional,
        presence=presence,
        group="para" if p.tag == "p" else "summary",
    )


def build_template(
    params: SynthParams,
    *,
    seed: int,
    index: int,
    tid: str,
    kind: V.KindProto,
    is_tail: bool,
    names: NameRegistry,
    attempt: int = 0,
) -> TemplateDef:
    rng = random.Random(f"{seed}|template|{index}|{attempt}")
    markup = _pick(rng, params.markup_style, MARKUP_STYLES)
    absent = _pick(rng, params.absent_rendering, ABSENT_RENDERINGS)
    layout: FieldLayout
    if markup == "server":
        layout = rng.choice(("div", "p", "li"))
        root_tag = rng.choice(("article", "div", "section"))
    elif markup == "react":
        layout, root_tag = "div", "div"
    else:
        layout, root_tag = "row", "table"
    naming: Naming = rng.choice(("kebab", "kebab", "bem", "camel"))
    scheme: FieldScheme = rng.choices(("wrapper", "value", "itemprop"), (0.45, 0.4, 0.15))[0]
    ctx = ValueCtx(
        currency=rng.randrange(len(CURRENCIES)),
        date_fmt=rng.choice(DATE_FORMATS),
        time_fmt=rng.choice(TIME_FORMATS),
    )

    used: set[str] = {kind.title.name}
    # scalars: the kind's own catalog first, then generic fields
    kind_pool = list(kind.scalars)
    rng.shuffle(kind_pool)
    generic = list(V.GENERIC_SCALARS)
    rng.shuffle(generic)
    pool = [p for p in kind_pool + generic if p.name not in used]
    n_mand = rng.randint(*params.mandatory_fields)
    mandatory: list[ScalarDef] = []
    optional: list[ScalarDef] = []
    lo, hi = params.optional_presence
    for p in pool:
        if p.name in used:
            continue
        if len(mandatory) < n_mand:
            mandatory.append(_scalar(p, rng, False, 1.0))
        elif len(optional) < params.optional_fields:
            optional.append(_scalar(p, rng, True, round(rng.uniform(lo, hi), 3)))
        else:
            break
        used.add(p.name)
    scalars = list(mandatory)
    for s in optional:  # optional fields interleave with mandatory ones at stable positions
        scalars.insert(rng.randint(0, len(scalars)), s)
    inline = [s for s in scalars if s.group != "para"]
    if len(inline) > 4 and rng.random() < 0.6:
        cut = rng.randint(2, len(inline) - 2)
        for s in inline[cut:]:
            s.group = "details"

    lists: list[ListDef] = []
    list_pool = list(kind.lists)
    rng.shuffle(list_pool)
    generic_lists = list(V.GENERIC_LISTS)
    rng.shuffle(generic_lists)
    for lp in list_pool + generic_lists:
        if len(lists) >= params.list_fields:
            break
        if lp.name in used:
            continue
        used.add(lp.name)
        lists.append(
            ListDef(
                lp.name, rng.choice(lp.headings), lp.gen, lp.vtype, lp.desc, lp.ordered, lp.link
            )
        )

    tables: list[TableDef] = []
    generic_tables = list(V.GENERIC_TABLES)
    rng.shuffle(generic_tables)
    for tp in list(kind.tables) + generic_tables:
        if len(tables) >= params.record_tables:
            break
        if tp.name in used:
            continue
        used.add(tp.name)
        cols = [ColumnDef(c.name, c.label, c.gen, c.vtype) for c in tp.columns]
        extras = list(tp.extra)
        rng.shuffle(extras)
        taken = {c.name for c in tp.columns} | {c.name for c in extras}
        extras += [c for c in V.GENERIC_COLUMNS if c.name not in taken]
        for c in extras[: params.optional_columns]:
            col = ColumnDef(c.name, c.label, c.gen, c.vtype, True, round(rng.uniform(lo, hi), 3))
            cols.insert(rng.randint(1, len(cols)), col)
        tables.append(TableDef(tp.name, rng.choice(tp.headings), tp.desc, cols))

    kvs: list[KVDef] = []
    generic_kvs = list(V.GENERIC_KVS)
    rng.shuffle(generic_kvs)
    for kp in list(kind.kvs) + generic_kvs:
        if len(kvs) >= params.kv_blocks:
            break
        if kp.name in used:
            continue
        used.add(kp.name)
        style = _pick(rng, params.kv_style, KV_STYLES)
        kvs.append(
            KVDef(
                kp.name,
                rng.choice(kp.headings),
                kp.desc,
                list(kp.keys),
                style,
                rng.choice(("th", "td")),
            )
        )

    order: list[tuple[str, str]] = []
    if any(s.group == "details" for s in scalars):
        order.append(("group", "details"))
    order += [("para", s.name) for s in scalars if s.group == "para"]
    order += [("list", x.name) for x in lists]
    order += [("kv", x.name) for x in kvs]
    order += [("table", x.name) for x in tables]
    rng.shuffle(order)
    order.insert(0, ("group", "summary"))

    url_style = "query" if markup == "table" and rng.random() < 0.5 else kind.url_style
    return TemplateDef(
        tid=tid,
        index=index,
        kind=kind.kind,
        is_tail=is_tail,
        segment=names.segment(kind, rng),
        url_style=url_style,
        markup=markup,
        absent=absent,
        prefix=names.prefix(kind, rng),
        naming=naming,
        field_scheme=scheme,
        field_layout=layout,
        label_colon=rng.random() < 0.5,
        root_tag=root_tag,
        react_depth=rng.randint(1, 2),
        react_h1=rng.random() < 0.5,
        value_ctx=ctx,
        title=ScalarDef(
            kind.title.name,
            "",
            kind.title.gen,
            kind.title.vtype,
            kind.title.desc,
            "h1",
            group="header",
        ),
        scalars=scalars,
        lists=lists,
        tables=tables,
        kvs=kvs,
        order=order,
        related=rng.random() < 0.5,
        related_heading=kind.related,
        class_seed=f"{seed}|classes|{index}|{attempt}",
        hashed_share=params.hashed_class_share,
    )


# --- near-duplicate derivation --------------------------------------------------------------

Op = tuple[str, str]  # (operation, argument)


def near_duplicate_ops(a: TemplateDef, rng: random.Random) -> tuple[list[Op], list[Op]]:
    """Coarse (whole field blocks) and fine (single element classes) edits that turn template
    ``a`` into a structurally different sibling. Coarse ops alternate adding a new mandatory
    field and dropping one of ``a``'s mandatory fields so the page size stays comparable."""
    kind = V.KIND_BY_NAME[a.kind]
    have = set(a.field_names())
    adds = [p.name for p in list(kind.scalars) + list(V.GENERIC_SCALARS) if p.name not in have]
    rng.shuffle(adds)
    drops = [s.name for s in a.scalars if not s.optional]
    rng.shuffle(drops)
    drops = drops[: max(0, len(drops) - 2)]  # keep at least two shared mandatory fields
    coarse: list[Op] = []
    for i in range(max(len(adds), len(drops))):
        if i < len(adds):
            coarse.append(("add", adds[i]))
        if i < len(drops):
            coarse.append(("drop", drops[i]))
    blocks = [x.name for x in a.lists] + [x.name for x in a.tables] + [x.name for x in a.kvs]
    coarse += [("slot", f"section:{name}") for name in blocks]
    fine: list[Op] = []
    for s in a.scalars:
        fine += [("value", s.name), ("label", s.name)]
    rng.shuffle(fine)
    return coarse, fine


def apply_op(t: TemplateDef, op: Op, rng: random.Random) -> TemplateDef:
    """Return a copy of ``t`` with one near-duplicate edit applied."""
    out = t.clone()
    what, arg = op
    if what == "drop":
        out.scalars = [s for s in out.scalars if s.name != arg]
        out.order = [o for o in out.order if o != ("para", arg)]
    elif what == "add":
        proto = next(
            p
            for p in list(V.KIND_BY_NAME[t.kind].scalars) + list(V.GENERIC_SCALARS)
            if p.name == arg
        )
        s = _scalar(proto, rng, False, 1.0)
        if s.group != "para" and any(x.group == "details" for x in out.scalars):
            s.group = rng.choice(("summary", "details"))
        out.scalars.insert(rng.randint(0, len(out.scalars)), s)
        if s.group == "para":
            out.order.insert(rng.randint(1, len(out.order)), ("para", s.name))
        # a new field gets its own semantic classes so it adds new structural paths
        out.overrides[f"wrap:{arg}"] = semantic_class(t.prefix, t.naming, arg)
        out.overrides[f"val:{arg}"] = semantic_class(t.prefix, t.naming, arg)
        out.overrides[f"label@{arg}"] = semantic_class(t.prefix, t.naming, f"{arg}-label")
        out.overrides[f"value@{arg}"] = semantic_class(t.prefix, t.naming, f"{arg}-value")
    elif what == "slot":
        out.overrides[arg] = semantic_class(t.prefix, t.naming, role_of(arg) + "-alt")
    elif what in ("value", "label"):
        out.overrides[f"{what}@{arg}"] = semantic_class(t.prefix, t.naming, f"{arg}-{what}-alt")
    else:  # pragma: no cover - programming error
        raise ValueError(f"unknown near-duplicate op {op}")
    return out
