"""Unprompted-mode checkpoint: show the proposed schema, let the user drop or rename fields.

``show_schema_proposal`` runs after unprompted generation and before verification. Edits are
written back to ``schema_proposal.json`` and applied to the draft specs
(``specs/draft/<tid>.json``) so verification runs on the confirmed schema.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field
from rich.console import Console
from rich.table import Table as RichTable

from ditto.checkpoints._common import InputFn, ask, make_console, short, split_names
from ditto.run import RunDir, read_json, write_json
from ditto.specgen.models import Spec

RESERVED = frozenset({"url", "page_id", "template_id"})
HELP = (
    "Commands: Enter = accept · drop <field>[, <field>...] · rename <old> <new> · show\n"
    "Prefix a field with a template id (t_3f9a:price) to change one template only."
)


class ProposalResult(BaseModel):
    confirmed_by: Literal["user", "yes"]
    kept: dict[str, list[str]] = Field(default_factory=dict)
    dropped: dict[str, list[str]] = Field(default_factory=dict)
    renamed: dict[str, dict[str, str]] = Field(default_factory=dict, description="old -> new")
    drafts_updated: list[str] = Field(default_factory=list)
    drafts_missing: list[str] = Field(default_factory=list)


def render_proposal(data: dict[str, Any], con: Console) -> None:
    templates: dict[str, Any] = data.get("templates") or {}
    if not templates:
        con.print("schema proposal: no templates")
        return
    for tid, entry in templates.items():
        table = RichTable(
            title=f"{tid} · {entry.get('size', '?')} pages · proposed fields", title_justify="left"
        )
        for col in ("field", "type", "check", "optional", "candidates", "examples"):
            table.add_column(col, overflow="fold")
        for name, f in (entry.get("fields") or {}).items():
            examples = f.get("examples") or []
            table.add_row(
                name,
                str(f.get("type", "")),
                str(f.get("type_check") or ""),
                "yes" if f.get("optional") else "",
                ", ".join(f.get("source_candidates") or []),
                " | ".join(short(e, 40) for e in examples[:3]),
            )
        con.print(table)


def _target(token: str) -> tuple[str | None, str]:
    tid, sep, name = token.partition(":")
    return (tid, name) if sep else (None, token)


class _Editor:
    """Applies drop/rename commands to the proposal, tracking each field's original name."""

    def __init__(self, data: dict[str, Any], con: Console) -> None:
        self.templates: dict[str, Any] = data.get("templates") or {}
        self.con = con
        # tid -> current name -> original name (insertion order = display order)
        self.origin: dict[str, dict[str, str]] = {
            tid: {name: name for name in (entry.get("fields") or {})}
            for tid, entry in self.templates.items()
        }
        self.original: dict[str, list[str]] = {t: list(n) for t, n in self.origin.items()}

    def _tids(self, tid: str | None, name: str) -> list[str]:
        if tid is not None:
            if tid not in self.templates:
                self.con.print(f"  [red]unknown template {tid}[/red]")
                return []
            tids = [tid] if name in self.origin[tid] else []
        else:
            tids = [t for t, names in self.origin.items() if name in names]
        if not tids:
            self.con.print(f"  [red]no template has a field {name!r}[/red]")
        return tids

    def drop(self, token: str) -> None:
        tid, name = _target(token)
        for t in self._tids(tid, name):
            del self.templates[t]["fields"][name]
            del self.origin[t][name]
            self.con.print(f"  dropped {t}:{name}")

    def rename(self, token: str, new: str) -> None:
        tid, name = _target(token)
        if new.startswith("_") or new in RESERVED:
            self.con.print(f"  [red]{new!r} is reserved for output metadata[/red]")
            return
        for t in self._tids(tid, name):
            if new in self.origin[t]:
                self.con.print(f"  [red]{t} already has a field {new!r}[/red]")
                continue
            fields = self.templates[t]["fields"]
            self.templates[t]["fields"] = {(new if k == name else k): v for k, v in fields.items()}
            self.origin[t] = {(new if k == name else k): v for k, v in self.origin[t].items()}
            self.con.print(f"  renamed {t}:{name} -> {new}")

    def changes(self, tid: str) -> tuple[list[str], dict[str, str]]:
        """``(dropped, renamed old->new)`` for one template, in original names."""
        kept = set(self.origin[tid].values())
        dropped = [n for n in self.original[tid] if n not in kept]
        renamed = {orig: cur for cur, orig in self.origin[tid].items() if cur != orig}
        return dropped, renamed


def apply_to_draft(spec: Spec, dropped: list[str], renamed: dict[str, str]) -> Spec:
    """Drop and rename fields of a draft spec; everything else is kept as it was. A field
    renamed to a name listed in ``not_present`` replaces that entry."""
    data = spec.model_dump()
    data["fields"] = {
        renamed.get(name, name): f for name, f in data["fields"].items() if name not in dropped
    }
    data["not_present"] = {k: v for k, v in data["not_present"].items() if k not in data["fields"]}
    return Spec.model_validate(data)


def show_schema_proposal(
    run: RunDir,
    *,
    yes: bool = False,
    input_fn: InputFn = input,
    console: Console | None = None,
) -> ProposalResult:
    """Show ``schema_proposal.json``; unless ``yes``, let the user drop or rename fields.

    Edits are written back to the proposal (with a ``checkpoint`` record) and applied to the
    draft specs so verification runs on the confirmed schema.
    """
    con = make_console(console)
    by: Literal["user", "yes"] = "yes" if yes else "user"
    if not run.schema_proposal.exists():
        con.print(f"no schema proposal at {run.schema_proposal}")
        return ProposalResult(confirmed_by=by)
    data: dict[str, Any] = read_json(run.schema_proposal)
    render_proposal(data, con)
    editor = _Editor(data, con)
    if not yes:
        con.print(HELP)
        while True:
            ans = ask(input_fn, "schema> ")
            cmd, _, rest = ans.partition(" ")
            cmd = cmd.lower()
            if cmd in ("", "y", "yes", "accept", "done"):
                break
            if cmd == "drop" and rest.strip():
                for token in split_names(rest):
                    editor.drop(token)
            elif cmd == "rename" and len(rest.split()) == 2:
                old, new = rest.split()
                editor.rename(old, new)
            elif cmd == "show":
                render_proposal(data, con)
            else:
                con.print(HELP)
    result = ProposalResult(
        confirmed_by=by, kept={tid: list(names) for tid, names in editor.origin.items()}
    )
    for tid in editor.origin:
        dropped, renamed = editor.changes(tid)
        if dropped:
            result.dropped[tid] = dropped
        if renamed:
            result.renamed[tid] = renamed
        if not dropped and not renamed:
            continue
        path = run.draft_spec(tid)
        if not path.exists():
            result.drafts_missing.append(tid)
            continue
        apply_to_draft(Spec.load(path), dropped, renamed).save(path)
        result.drafts_updated.append(tid)
    if result.dropped or result.renamed:
        data["checkpoint"] = {
            "confirmed_by": by,
            "confirmed_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "dropped": result.dropped,
            "renamed": result.renamed,
        }
        write_json(run.schema_proposal, data)
        con.print(
            f"schema: {sum(map(len, result.dropped.values()))} fields dropped, "
            f"{sum(map(len, result.renamed.values()))} renamed; "
            f"{len(result.drafts_updated)} draft specs updated"
        )
    return result
