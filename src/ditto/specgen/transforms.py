"""The closed set of value transforms a spec may use, applied in order.

Grammar (one string per transform)::

    strip | normalize_ws
    regex(<pattern>, <group>)       group is an int; pattern may be quoted '...' or "..."
    split(<sep>, <index>)           index is an int (negative counts from the end)
    replace(<old>, <new>)
    cast(int|float|date)
    attr(<name>)                    attribute value of the matched element (href, src, ...)

Arguments may be quoted with single or double quotes; inside quotes a backslash escapes the
quote character. An unquoted ``regex`` pattern extends to the last comma. Anything else fails.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from typing import Any

from lxml import etree

from ditto.textutil import normalize_ws

TRANSFORM_NAMES = ("strip", "normalize_ws", "regex", "split", "replace", "cast", "attr")
CAST_TYPES = ("int", "float", "date")
DATE_FORMATS = (
    "%Y-%m-%d",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%d %H:%M:%S",
    "%Y/%m/%d",
    "%m/%d/%Y",
    "%d/%m/%Y",
    "%d.%m.%Y",
    "%B %d, %Y",
    "%b %d, %Y",
    "%d %B %Y",
    "%d %b %Y",
    "%B %Y",
    "%b %Y",
    "%Y",
)

_CALL = re.compile(r"^\s*([a-z_]+)\s*(?:\((.*)\))?\s*$", re.DOTALL)


@dataclass(frozen=True)
class Transform:
    name: str
    args: tuple[str | int, ...] = ()

    def __str__(self) -> str:
        if not self.args:
            return self.name
        return f"{self.name}({', '.join(repr(a) if isinstance(a, str) else str(a) for a in self.args)})"


def _split_args(raw: str) -> list[tuple[str, bool]]:
    """Split on commas outside quotes. Returns ``(value, was_quoted)`` pairs."""
    args: list[tuple[str, bool]] = []
    buf: list[str] = []
    quote: str | None = None
    quoted = False
    i = 0
    while i < len(raw):
        ch = raw[i]
        if quote:
            if ch == "\\" and i + 1 < len(raw) and raw[i + 1] in (quote, "\\"):
                buf.append(raw[i + 1])
                i += 2
                continue
            if ch == quote:
                quote = None
            else:
                buf.append(ch)
        elif ch in "'\"" and not "".join(buf).strip():
            quote = ch
            quoted = True
            buf = []
        elif ch == ",":
            args.append(("".join(buf) if quoted else "".join(buf).strip(), quoted))
            buf, quoted = [], False
        else:
            if not (quoted and ch.isspace()):
                buf.append(ch)
        i += 1
    if quote:
        raise ValueError("unterminated quoted argument")
    args.append(("".join(buf) if quoted else "".join(buf).strip(), quoted))
    return args


def _int(value: str, what: str) -> int:
    try:
        return int(value)
    except ValueError:
        raise ValueError(f"{what} must be an integer, got {value!r}") from None


@lru_cache(maxsize=4096)
def parse_transform(text: str) -> Transform:
    m = _CALL.match(text)
    if not m:
        raise ValueError("not a transform call")
    name, raw = m.group(1), m.group(2)
    if name not in TRANSFORM_NAMES:
        raise ValueError(f"unknown transform {name!r}; allowed: {', '.join(TRANSFORM_NAMES)}")
    if name in ("strip", "normalize_ws"):
        if raw and raw.strip():
            raise ValueError(f"{name} takes no arguments")
        return Transform(name)
    if raw is None:
        raise ValueError(f"{name} needs arguments")
    if name == "regex":
        head, _, tail = raw.rpartition(",")
        if not head:
            pattern, group = raw, "0"
        else:
            parts = _split_args(raw)
            if len(parts) == 2:
                pattern, group = parts[0][0], parts[1][0]
            else:  # unquoted pattern containing commas: split on the last comma
                pattern, group = head.strip(), tail.strip()
        try:
            compiled = re.compile(pattern)
        except re.error as exc:
            raise ValueError(f"invalid regex: {exc}") from None
        g = _int(group, "regex group")
        if g > compiled.groups:
            raise ValueError(f"regex has {compiled.groups} groups, asked for {g}")
        return Transform(name, (pattern, g))
    parts = _split_args(raw)
    if name == "split":
        if len(parts) != 2:
            raise ValueError("split takes (sep, index)")
        if not parts[0][0]:
            raise ValueError("split separator is empty")
        return Transform(name, (parts[0][0], _int(parts[1][0], "split index")))
    if name == "replace":
        if len(parts) != 2:
            raise ValueError("replace takes (old, new)")
        return Transform(name, (parts[0][0], parts[1][0]))
    if name == "cast":
        if len(parts) != 1 or parts[0][0] not in CAST_TYPES:
            raise ValueError(f"cast takes one of {CAST_TYPES}")
        return Transform(name, (parts[0][0],))
    # attr
    if len(parts) != 1 or not re.fullmatch(r"[A-Za-z_:][\w:.-]*", parts[0][0]):
        raise ValueError("attr takes one attribute name")
    return Transform(name, (parts[0][0],))


def to_text(value: Any) -> str | None:
    """String value of an XPath result item (element -> normalized text content)."""
    if value is None:
        return None
    if isinstance(value, etree._Element):
        return normalize_ws("".join(str(t) for t in value.itertext()))
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else str(value)
    return str(value)


def _to_number(text: str, kind: str) -> int | float | None:
    cleaned = re.sub(r"[^\d.,\-]", "", text)
    if not re.search(r"\d", cleaned):
        return None
    if "," in cleaned and "." in cleaned:
        if cleaned.rfind(",") > cleaned.rfind("."):  # 1.234,56
            cleaned = cleaned.replace(".", "").replace(",", ".")
        else:
            cleaned = cleaned.replace(",", "")
    elif "," in cleaned:
        head, _, tail = cleaned.rpartition(",")
        cleaned = cleaned.replace(",", "") if len(tail) == 3 else head.replace(",", "") + "." + tail
    try:
        num = float(cleaned)
    except ValueError:
        return None
    return int(num) if kind == "int" else num


def _to_date(text: str) -> str | None:
    t = normalize_ws(text)
    for fmt in DATE_FORMATS:
        try:
            dt = datetime.strptime(t, fmt)
        except ValueError:
            continue
        return dt.date().isoformat()
    m = re.search(r"\d{4}-\d{2}-\d{2}", t)
    return m.group(0) if m else None


def apply_transforms(value: Any, transforms: list[str]) -> Any:
    """Apply a transform chain to one XPath result item. Returns str, int, float, or None."""
    current: Any = value
    for spec in transforms:
        t = parse_transform(spec)
        if current is None:
            return None
        if t.name == "attr":
            if not isinstance(current, etree._Element):
                return None
            current = current.get(str(t.args[0]))
            continue
        if t.name == "cast":
            text = to_text(current)
            if text is None:
                return None
            current = _to_date(text) if t.args[0] == "date" else _to_number(text, str(t.args[0]))
            continue
        text = current if isinstance(current, str) else to_text(current)
        if text is None:
            return None
        if t.name == "strip":
            current = text.strip()
        elif t.name == "normalize_ws":
            current = normalize_ws(text)
        elif t.name == "regex":
            m = re.search(str(t.args[0]), text)
            current = m.group(int(t.args[1])) if m else None
        elif t.name == "split":
            pieces = text.split(str(t.args[0]))
            idx = int(t.args[1])
            current = pieces[idx].strip() if -len(pieces) <= idx < len(pieces) else None
        elif t.name == "replace":
            current = text.replace(str(t.args[0]), str(t.args[1]))
    if isinstance(current, etree._Element):
        current = to_text(current)
    if isinstance(current, str) and not current.strip():
        return None
    return current
