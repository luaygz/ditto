"""Versioned prompt files: ``prompts/<name>.v<N>.md``.

A prompt is loaded by name and version; the id (``specgen.v1``) is written into every spec's
``generated_by.prompt_version`` and into the LLM cache key. A changed prompt is a new version
file, never an in-place edit of a committed one.

Lookup order: the installed package data ``ditto/_data/prompts`` (the wheel force-includes the
repo's ``prompts/`` there), then the source tree's ``prompts/`` directory (editable installs and
running from a checkout).

File format: optional ``# system`` and ``# user`` level-1 headings split the file into the two
chat messages (a file without them is a single user message). ``{{name}}`` placeholders are
filled by :meth:`Prompt.render` / :meth:`Prompt.messages`; every placeholder must be given and
no unknown name may be passed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from importlib.resources.abc import Traversable

    from ditto.llm.base import Message

_FILE = re.compile(r"^(?P<name>[a-z][a-z0-9_]*)\.v(?P<version>\d+)\.md$")
_PLACEHOLDER = re.compile(r"\{\{\s*([a-z_][a-z0-9_]*)\s*\}\}")
_SECTION = re.compile(r"^# (system|user)[ \t]*$", re.MULTILINE)

_REPO_PROMPTS = Path(__file__).resolve().parents[3] / "prompts"


@dataclass(frozen=True)
class Prompt:
    name: str
    version: int
    text: str

    @property
    def id(self) -> str:
        """``<name>.v<version>``, e.g. ``specgen.v1``."""
        return f"{self.name}.v{self.version}"

    @property
    def placeholders(self) -> set[str]:
        return set(_PLACEHOLDER.findall(self.text))

    def render(self, **values: str) -> str:
        """Fill every ``{{name}}`` placeholder of the whole file."""
        return _fill(self.text, values, self.placeholders, self.id)

    def messages(self, **values: str) -> list[Message]:
        """Split into ``system``/``user`` messages and fill placeholders.

        Splitting happens before substitution, so values may contain anything (HTML, JSON,
        markdown headings) without affecting the message structure.
        """
        names = self.placeholders
        parts = _SECTION.split(self.text)
        if len(parts) == 1:
            return [{"role": "user", "content": _fill(self.text, values, names, self.id)}]
        if parts[0].strip():
            raise ValueError(f"{self.id}: text before the first '# system'/'# user' heading")
        out: list[Message] = []
        for i in range(1, len(parts), 2):
            body = _fill(parts[i + 1], values, names, self.id).strip()
            if parts[i] == "system":
                out.append({"role": "system", "content": body})
            else:
                out.append({"role": "user", "content": body})
        return out


def _fill(text: str, values: dict[str, str], declared: set[str], prompt_id: str) -> str:
    unknown = set(values) - declared
    if unknown:
        raise ValueError(f"{prompt_id}: unknown placeholder value(s) {sorted(unknown)}")

    def sub(m: re.Match[str]) -> str:
        key = m.group(1)
        if key not in values:
            raise KeyError(f"{prompt_id}: no value for placeholder {{{{{key}}}}}")
        return values[key]

    return _PLACEHOLDER.sub(sub, text)


def _search_dirs() -> list[Traversable]:
    dirs: list[Traversable] = []
    try:
        packaged = resources.files("ditto") / "_data" / "prompts"
        if packaged.is_dir():
            dirs.append(packaged)
    except (ModuleNotFoundError, FileNotFoundError):  # pragma: no cover - ditto is importable
        pass
    if _REPO_PROMPTS.is_dir():
        dirs.append(_REPO_PROMPTS)
    return dirs


def available_prompts() -> dict[str, list[int]]:
    """Prompt name -> sorted versions found in the first directory that has the name."""
    found: dict[str, list[int]] = {}
    for d in _search_dirs():
        local: dict[str, list[int]] = {}
        for entry in d.iterdir():
            m = _FILE.match(entry.name)
            if m:
                local.setdefault(m.group("name"), []).append(int(m.group("version")))
        for name, versions in local.items():
            found.setdefault(name, sorted(versions))
    return found


@lru_cache(maxsize=64)
def load_prompt(name: str, version: int | None = None) -> Prompt:
    """Load ``prompts/<name>.v<version>.md``; the highest version when ``version`` is None."""
    for d in _search_dirs():
        if version is None:
            versions = [
                int(m.group("version"))
                for entry in d.iterdir()
                if (m := _FILE.match(entry.name)) and m.group("name") == name
            ]
            if not versions:
                continue
            chosen = max(versions)
        else:
            chosen = version
        entry = d / f"{name}.v{chosen}.md"
        if entry.is_file():
            return Prompt(name=name, version=chosen, text=entry.read_text(encoding="utf-8"))
    where = ", ".join(str(d) for d in _search_dirs()) or "(no prompt directory found)"
    wanted = f"{name}.v{version}" if version is not None else f"{name}.v*"
    raise FileNotFoundError(f"prompt {wanted}.md not found in {where}")
