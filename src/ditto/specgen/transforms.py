"""The closed set of value transforms a spec may use, applied in order.

Grammar (one string per transform)::

    transform := name | name "(" args ")"
    strip | normalize_ws                 no arguments
    regex(<pattern>, <group>)            <group> is a non-negative int; omitted means 0
    split(<sep>, <index>)                <index> is an int; negative counts from the end
    replace(<old>, <new>)                <old> must not be empty
    cast(int|float|date)
    attr(<name>)                         attribute value of the matched element

Arguments are separated by commas. An argument may be quoted with ``'...'`` or ``"..."``;
inside quotes ``\\'``/``\\"`` stands for the quote character and ``\\\\`` for one backslash, and
commas and spaces are literal. Unquoted arguments are trimmed and cannot contain commas, except
that an unquoted ``regex`` pattern extends to the last comma. Any other name, arity or argument
fails validation.

Values: each transform receives one XPath result item (an element, a string from ``text()`` or
``@attr``, or a number/boolean from an XPath expression) or the previous transform's output.
Elements are read as their whitespace-normalized text. ``attr`` needs an element. A step that
produces nothing (no regex match, index out of range, unparseable cast, missing attribute) makes
the whole chain produce ``null``. The final value is a trimmed string, an ``int``/``float`` from
``cast``, or ``null`` when empty.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, cast

from lxml import etree

from ditto.specgen.typecheck import parse_date, parse_int, parse_number


def _collapse_ws(text: str) -> str:
    """Same result as ``ditto.textutil.normalize_ws`` (``str.split`` splits on NBSP and every
    other Unicode space ``\\s`` matches), several times faster on long texts."""
    return " ".join(text.split())


TRANSFORM_NAMES = ("strip", "normalize_ws", "regex", "split", "replace", "cast", "attr")
CAST_TYPES = ("int", "float", "date")

#: ``name -> (signature, meaning)``; rendered into ``docs/spec-format.md``.
TRANSFORM_DOCS: dict[str, tuple[str, str]] = {
    "strip": ("strip", "Trim leading and trailing whitespace."),
    "normalize_ws": (
        "normalize_ws",
        "Collapse every whitespace run (including NBSP) to one space and trim.",
    ),
    "regex": (
        "regex(pattern, group)",
        "Search the text with the Python regular expression `pattern` and keep capture `group` "
        "(default 0, the whole match). No match gives null.",
    ),
    "split": (
        "split(sep, index)",
        "Split on the literal `sep` and keep piece `index` (negative counts from the end), "
        "trimmed. An index out of range gives null.",
    ),
    "replace": ("replace(old, new)", "Replace every literal occurrence of `old` with `new`."),
    "cast": (
        "cast(int|float|date)",
        "`int`/`float`: the first number in the text (`1,234.50`, `€1.234,56`, `1 234,5`); "
        "`int` truncates toward zero. `date`: the first date as ISO 8601 (`March 3, 2021` -> "
        "`2021-03-03`; `March 2021` -> `2021-03`). Unparseable text gives null.",
    ),
    "attr": (
        "attr(name)",
        "Attribute `name` of the matched element (`href`, `src`, `content`, ...). Gives null "
        "when the value is not an element or the attribute is missing.",
    ),
}

_CALL = re.compile(r"^\s*([A-Za-z_][\w-]*)\s*(?:\((.*)\))?\s*$", re.DOTALL)
_ATTR_NAME = re.compile(r"^[A-Za-z_:][\w:.-]*$")


@dataclass(frozen=True)
class Transform:
    name: str
    args: tuple[str | int, ...] = ()

    def __str__(self) -> str:
        """Canonical text form; ``parse_transform(str(t)) == t``."""
        if not self.args:
            return self.name
        return f"{self.name}({', '.join(_quote(a) for a in self.args)})"


def _quote(arg: str | int) -> str:
    if isinstance(arg, int):
        return str(arg)
    return "'" + arg.replace("\\", "\\\\").replace("'", "\\'") + "'"


def _split_args(raw: str) -> list[tuple[str, bool]]:
    """Split on commas outside quotes. Returns ``(value, was_quoted)`` pairs."""
    args: list[tuple[str, bool]] = []
    i, n = 0, len(raw)
    while True:
        while i < n and raw[i].isspace():
            i += 1
        if i < n and raw[i] in "'\"":
            quote, buf = raw[i], []
            i += 1
            while i < n and raw[i] != quote:
                if raw[i] == "\\" and i + 1 < n and raw[i + 1] in (quote, "\\"):
                    i += 1
                buf.append(raw[i])
                i += 1
            if i >= n:
                raise ValueError("unterminated quoted argument")
            i += 1
            while i < n and raw[i].isspace():
                i += 1
            if i < n and raw[i] != ",":
                raise ValueError("unexpected text after a quoted argument")
            args.append(("".join(buf), True))
        else:
            comma = raw.find(",", i)
            end = n if comma < 0 else comma
            args.append((raw[i:end].strip(), False))
            i = end
        if i >= n:
            return args
        i += 1  # the comma


def _int(value: str, what: str) -> int:
    try:
        return int(value)
    except ValueError:
        raise ValueError(f"{what} must be an integer, got {value!r}") from None


def _signature(name: str) -> str:
    return TRANSFORM_DOCS[name][0]


@lru_cache(maxsize=4096)
def parse_transform(text: str) -> Transform:
    """Parse and validate one transform string; raises ``ValueError`` with the reason."""
    m = _CALL.match(text)
    if not m:
        raise ValueError(f"not a transform call {text!r}; expected name or name(args)")
    name, raw = m.group(1), m.group(2)
    if name not in TRANSFORM_NAMES:
        raise ValueError(f"unknown transform {name!r}; allowed: {', '.join(TRANSFORM_NAMES)}")
    if name in ("strip", "normalize_ws"):
        if raw is not None and raw.strip():
            raise ValueError(f"{name} takes no arguments")
        return Transform(name)
    if raw is None or not raw.strip():
        raise ValueError(f"{name} needs arguments: {_signature(name)}")
    parts = _split_args(raw)
    if name == "regex":
        return _parse_regex(raw, parts)
    if name == "split":
        if len(parts) != 2:
            raise ValueError("split takes (sep, index); quote a ',' separator")
        if not parts[0][0]:
            raise ValueError("split separator is empty")
        return Transform(name, (parts[0][0], _int(parts[1][0], "split index")))
    if name == "replace":
        if len(parts) != 2:
            raise ValueError("replace takes (old, new); quote arguments containing ','")
        if not parts[0][0]:
            raise ValueError("replace needs a non-empty old string")
        return Transform(name, (parts[0][0], parts[1][0]))
    if name == "cast":
        if len(parts) != 1 or parts[0][0] not in CAST_TYPES:
            raise ValueError(f"cast takes one of {', '.join(CAST_TYPES)}")
        return Transform(name, (parts[0][0],))
    # attr
    if len(parts) != 1 or not _ATTR_NAME.match(parts[0][0]):
        raise ValueError("attr takes one attribute name")
    return Transform(name, (parts[0][0],))


def _parse_regex(raw: str, parts: list[tuple[str, bool]]) -> Transform:
    if len(parts) > 2 and not parts[0][1]:  # unquoted pattern with commas: last comma splits
        head, _, tail = raw.rpartition(",")
        parts = [(head.strip(), False), (tail.strip(), False)]
    if len(parts) > 2:
        raise ValueError("regex takes (pattern, group)")
    pattern = parts[0][0]
    group = _int(parts[1][0], "regex group") if len(parts) == 2 else 0
    if not pattern:
        raise ValueError("regex pattern is empty")
    try:
        compiled = _regex(pattern)
    except re.error as exc:
        raise ValueError(f"invalid regex: {exc}") from None
    if not 0 <= group <= compiled.groups:
        raise ValueError(f"regex has {compiled.groups} groups, asked for group {group}")
    return Transform("regex", (pattern, group))


@lru_cache(maxsize=4096)
def _regex(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern)


def to_text(value: Any) -> str | None:
    """String value of an XPath result item. An element gives its whitespace-normalized full
    text, the same value as ``ditto.fingerprint.dom.node_text``."""
    if value is None:
        return None
    if isinstance(value, etree._Element):
        # HTML trees yield str only; the stubs also allow bytes.
        return _collapse_ws("".join(cast(Iterator[str], value.itertext())))
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else str(value)
    return str(value)


def _cast(text: str, kind: str) -> int | float | str | None:
    if kind == "int":
        return parse_int(text)
    if kind == "float":
        return parse_number(text)
    return parse_date(text)


def apply_transforms(value: Any, transforms: list[str]) -> Any:
    """Apply a transform chain to one XPath result item. Returns str, int, float, or None."""
    current: Any = to_text(value) if isinstance(value, (bool, float)) else value
    for spec in transforms:
        if current is None:
            return None
        t = parse_transform(spec)
        if t.name == "attr":
            current = current.get(str(t.args[0])) if isinstance(current, etree._Element) else None
            continue
        text: str = current if isinstance(current, str) else to_text(current) or ""
        if t.name == "cast":
            current = _cast(text, str(t.args[0]))
        elif t.name == "strip":
            current = text.strip()
        elif t.name == "normalize_ws":
            current = _collapse_ws(text)
        elif t.name == "regex":
            m = _regex(str(t.args[0])).search(text)
            current = m.group(int(t.args[1])) if m else None
        elif t.name == "split":
            pieces = text.split(str(t.args[0]))
            idx = int(t.args[1])
            current = pieces[idx].strip() if -len(pieces) <= idx < len(pieces) else None
        else:  # replace
            current = text.replace(str(t.args[0]), str(t.args[1]))
    if isinstance(current, etree._Element):
        current = to_text(current)
    if isinstance(current, str):
        current = current.strip()
        return current or None
    return current
