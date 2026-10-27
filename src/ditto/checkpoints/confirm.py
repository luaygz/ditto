"""Stage 9b: confirm. Turn the probe proposal into ``mapping.json``.

``--yes`` (or a non-interactive caller) accepts ``probe.yaml`` as it stands (the probe's
proposal plus any hand edits). Interactively, ``$EDITOR`` opens ``probe.yaml`` and the saved file
is read back; without an editor each ambiguous field is asked in the terminal.

Every mapped field records its provenance: ``user`` when a person chose it (an interactive
answer, an explicit ``edits`` entry, a ``use`` that differs from the probe's proposal, or a field
marked ``confirmed: true`` in probe.yaml), else ``auto`` (the probe's proposal accepted unseen).
Generation binds ``user`` entries and treats ``auto`` ones as suggestions, so the probe's ranked
alternatives are kept with each field.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from collections.abc import Mapping as MappingABC
from typing import Any

import yaml
from rich.console import Console
from rich.table import Table as RichTable

from ditto.analyze.models import CandidateTable
from ditto.checkpoints._common import InputFn, ask, make_console, short, split_names
from ditto.checkpoints.match import usable_candidates
from ditto.checkpoints.probe import (
    CandidateInfo,
    FieldProbe,
    ProbeResult,
    TemplateProbe,
    dump_probe_yaml,
    write_probe_yaml,
)
from ditto.run import RunDir
from ditto.schema import ABSENT, Alternative, Mapping, Provenance, TemplateMapping

Edits = MappingABC[str, MappingABC[str, str | list[str]]]
ALTERNATIVES = 3  # ranked probe candidates kept per field in mapping.json

_ABSENT_WORDS = frozenset({"", ABSENT, "-", "none", "null", "no"})


class ConfirmError(ValueError):
    """``probe.yaml`` does not describe a valid mapping."""

    def __init__(self, problems: list[str]) -> None:
        super().__init__("probe.yaml has problems:\n  " + "\n  ".join(problems))
        self.problems = problems


class _Ids:
    """Candidate ids of one template that may (and may not) be mapped to a field."""

    def __init__(self, table: CandidateTable | None) -> None:
        self.known = table is not None
        self.usable: set[str] = set()
        self.cls: dict[str, str] = {}
        self.groups: dict[str, list[str]] = {}
        if table is None:
            return
        self.usable = {c.id for c in usable_candidates(table)}
        for c in table.candidates:
            role = c.table_role if c.table_role in ("kv_key", "record_header") else c.cls
            self.cls[c.id] = role
        for t in table.tables:
            if t.kind == "key_value":
                self.groups[t.id] = [i for i in (t.key_candidate, t.value_candidate) if i]
            else:
                self.groups[t.id] = [c.candidate_id for c in t.columns]

    def resolve(self, cid: str, problems: list[str], where: str) -> list[str]:
        if not self.known or cid in self.usable:
            return [cid]
        if cid in self.groups:
            return list(self.groups[cid])
        if cid in self.cls:
            problems.append(f"{where}: {cid} is a {self.cls[cid]}, which is never a field value")
        else:
            problems.append(f"{where}: unknown candidate id {cid!r}")
        return []


def parse_use(raw: Any, ids: _Ids, problems: list[str], where: str) -> str | list[str]:
    """A ``use`` value from probe.yaml or the prompt: id, ``[ids]``, ``"c4, c5"`` or absent."""
    if raw is None or raw is False:
        return ABSENT
    if isinstance(raw, str):
        if raw.strip().lower() in _ABSENT_WORDS:
            return ABSENT
        items: list[Any] = split_names(raw)
    elif isinstance(raw, list):
        items = raw
    else:
        problems.append(f"{where}: `use` must be a candidate id, a list of ids or absent")
        return ABSENT
    out: list[str] = []
    for item in items:
        if not isinstance(item, str) or not item.strip():
            problems.append(f"{where}: {item!r} is not a candidate id")
            continue
        for cid in ids.resolve(item.strip(), problems, where):
            if cid not in out:
                out.append(cid)
    if not out:
        return ABSENT
    return out[0] if len(out) == 1 and not isinstance(raw, list) else out


def alternative(ci: CandidateInfo) -> Alternative:
    """A probe candidate as stored in ``mapping.json``."""
    return Alternative(
        id=ci.id,
        use=ci.use,
        score=ci.score,
        kind=ci.cls,
        label=ci.key or ci.label_hint,
        examples=ci.examples[:3],
    )


def _key_notes(tp: TemplateProbe, fields: dict[str, str | list[str]]) -> list[str]:
    """Generation hints for fields matched as one key of a key/value table."""
    return [
        f"{name}: value of key '{fp.key}' in {fp.use}"
        for name, fp in tp.fields.items()
        if fp.key and fields.get(name) == fp.use
    ]


def mapping_from_yaml(probe: ProbeResult, data: Any, tables: dict[str, CandidateTable]) -> Mapping:
    """Build the mapping from (possibly edited) probe.yaml data; raise ``ConfirmError``.

    Templates or fields missing from the YAML keep the probe's proposal.
    """
    problems: list[str] = []
    ytemplates: dict[str, Any] = {}
    if not isinstance(data, dict) or not isinstance(data.get("templates", {}), dict):
        raise ConfirmError(["expected a top-level `templates:` mapping"])
    ytemplates = data.get("templates") or {}
    problems += [f"{tid}: unknown template" for tid in ytemplates if tid not in probe.templates]
    fields_wanted = probe.schema_fields
    out: dict[str, TemplateMapping] = {}
    for tid, tp in probe.templates.items():
        entry = ytemplates.get(tid)
        if entry is None:
            entry = {}
        if not isinstance(entry, dict):
            problems.append(f"{tid}: expected a mapping")
            continue
        include = entry.get("include", tp.include)
        if not isinstance(include, bool):
            problems.append(f"{tid}.include: expected true or false, got {include!r}")
            include = tp.include
        yfields = entry.get("fields") or {}
        if not isinstance(yfields, dict):
            problems.append(f"{tid}.fields: expected a mapping")
            yfields = {}
        problems += [f"{tid}.{f}: unknown field" for f in yfields if f not in fields_wanted]
        ids = _Ids(tables.get(tid))
        fields: dict[str, str | list[str]] = {}
        provenance: dict[str, Provenance] = {}
        alternatives: dict[str, list[Alternative]] = {}
        for name in fields_wanted:
            fp = tp.fields.get(name)
            default: Any = fp.use if fp is not None else ABSENT
            fentry = yfields.get(name)
            raw = fentry.get("use", default) if isinstance(fentry, dict) else fentry
            if fentry is None:
                raw = default
            where = f"{tid}.{name}"
            fields[name] = parse_use(raw, ids, problems, where)
            confirmed = fentry.get("confirmed", False) if isinstance(fentry, dict) else False
            if not isinstance(confirmed, bool):
                problems.append(f"{where}.confirmed: expected true or false, got {confirmed!r}")
                confirmed = False
            proposal = parse_use(default, ids, [], where)
            provenance[name] = "user" if confirmed or fields[name] != proposal else "auto"
            if fp is not None and fp.candidates:
                alternatives[name] = [alternative(ci) for ci in fp.candidates[:ALTERNATIVES]]
        notes = _key_notes(tp, fields)
        user_note = entry.get("note")
        if isinstance(user_note, str) and user_note.strip():
            notes.insert(0, user_note.strip())
        out[tid] = TemplateMapping(
            included=include,
            fields=fields,
            provenance=provenance,
            alternatives=alternatives,
            note="; ".join(notes) if notes else None,
        )
    if problems:
        raise ConfirmError(problems)
    return Mapping(schema_fields=fields_wanted, templates=out)


def read_yaml(path: os.PathLike[str] | str) -> Any:
    try:
        with open(path, encoding="utf-8") as fh:
            return yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        raise ConfirmError([f"not valid YAML: {exc}"]) from exc


def _load_tables(run: RunDir, probe: ProbeResult) -> dict[str, CandidateTable]:
    tables: dict[str, CandidateTable] = {}
    for tid in probe.templates:
        path = run.candidates(tid)
        if path.exists():
            tables[tid] = CandidateTable.load(path)
    return tables


def _yaml_template(data: dict[str, Any], tid: str) -> dict[str, Any]:
    """The (created if needed) entry of ``tid`` in probe.yaml data."""
    if not isinstance(data.get("templates"), dict):
        data["templates"] = {}
    templates: dict[str, Any] = data["templates"]
    entry = templates.get(tid)
    if not isinstance(entry, dict):
        entry = templates[tid] = {}
    return entry


def _set_use(data: dict[str, Any], tid: str, name: str, use: str | list[str]) -> None:
    """Write a person's choice into probe.yaml data (marked confirmed, so a later ``--yes``
    keeps it as a user decision)."""
    entry = _yaml_template(data, tid)
    if not isinstance(entry.get("fields"), dict):
        entry["fields"] = {}
    fentry = entry["fields"].get(name)
    if isinstance(fentry, dict):
        fentry["use"] = use
        fentry["confirmed"] = True
    else:
        entry["fields"][name] = {"use": use, "confirmed": True}


def apply_edits(data: dict[str, Any], edits: Edits) -> None:
    """Explicit ``{template_id: {field: id | [ids] | "absent"}}`` choices into probe.yaml data
    (validated later by :func:`mapping_from_yaml`)."""
    for tid, fields in edits.items():
        for name, use in fields.items():
            _set_use(data, tid, name, use)


def _candidate_table(fp: FieldProbe) -> RichTable:
    t = RichTable(show_header=True, header_style="bold", box=None, pad_edge=False)
    for col in ("#", "id", "score", "class", "label", "type", "examples"):
        t.add_column(col)
    for i, ci in enumerate(fp.candidates, 1):
        t.add_row(
            str(i),
            ci.id + (" [" + ", ".join(ci.use) + "]" if isinstance(ci.use, list) else ""),
            f"{ci.score:.2f}",
            ci.cls or "",
            short(ci.key or ci.label_hint or "", 30),
            ci.value_type or "",
            " | ".join(short(e, 30) for e in ci.examples[:3]),
        )
    return t


def _ask_field(
    name: str,
    fp: FieldProbe,
    default: str | list[str],
    ids: _Ids,
    input_fn: InputFn,
    con: Console,
) -> str | list[str]:
    con.print(_candidate_table(fp))
    shown = ", ".join(default) if isinstance(default, list) else default
    prompt = f"  {name} [1-{len(fp.candidates)}, candidate id, absent; Enter = {shown}]: "
    while True:
        ans = ask(input_fn, prompt)
        if not ans:
            return default
        if ans.isdigit() and 1 <= int(ans) <= len(fp.candidates):
            return fp.candidates[int(ans) - 1].use
        problems: list[str] = []
        value = parse_use(ans, ids, problems, name)
        if not problems:
            return value
        for p in problems:
            con.print(f"  [red]{p}[/red]")


def walk_ambiguous(
    probe: ProbeResult,
    data: dict[str, Any],
    tables: dict[str, CandidateTable],
    input_fn: InputFn,
    con: Console,
) -> None:
    """Ask about every ambiguous field of included templates, then about excluded templates;
    the answers are written into ``data`` (the probe.yaml content)."""
    current = mapping_from_yaml(probe, data, tables)
    for tid, tp in probe.templates.items():
        tm = current.templates[tid]
        ambiguous = [f for f, fp in tp.fields.items() if fp.status == "ambiguous"]
        if not tm.included or not ambiguous:
            continue
        urls = ", ".join(tp.url_examples[:2])
        con.print(f"\n[bold]{tid}[/bold] · {tp.size} pages · {urls}")
        ids = _Ids(tables.get(tid))
        for name in ambiguous:
            con.print(f"[yellow]{name}[/yellow] is ambiguous:")
            use = _ask_field(name, tp.fields[name], tm.fields[name], ids, input_fn, con)
            _set_use(data, tid, name, use)
    excluded = [tid for tid, tm in current.templates.items() if not tm.included]
    if not excluded:
        return
    con.print("\nProposed as excluded (no requested field found):")
    for tid in excluded:
        tp = probe.templates[tid]
        con.print(f"  {tid} · {tp.size} pages · {', '.join(tp.url_examples[:2])}")
    while True:
        ans = ask(input_fn, "Include any of them? [template ids, Enter = none]: ")
        unknown = [t for t in split_names(ans) if t not in excluded]
        if not unknown:
            break
        con.print(f"  [red]not an excluded template: {', '.join(unknown)}[/red]")
    for tid in split_names(ans):
        _yaml_template(data, tid)["include"] = True


def _edit_loop(
    run: RunDir,
    probe: ProbeResult,
    tables: dict[str, CandidateTable],
    editor: str,
    input_fn: InputFn,
    con: Console,
) -> Mapping:
    while True:
        cmd = [*shlex.split(editor), str(run.probe_yaml)]
        try:
            proc = subprocess.run(cmd, check=False)
        except OSError as exc:
            raise ConfirmError([f"cannot run editor {editor!r}: {exc}"]) from exc
        if proc.returncode != 0:
            raise ConfirmError([f"editor exited with status {proc.returncode}; nothing written"])
        try:
            return mapping_from_yaml(probe, read_yaml(run.probe_yaml), tables)
        except ConfirmError as exc:
            for p in exc.problems:
                con.print(f"[red]{p}[/red]")
            if ask(input_fn, "Re-open the editor to fix them? [Y/n]: ").lower() in ("n", "no"):
                raise


def summarize(mapping: Mapping) -> str:
    included = mapping.included_templates()
    mapped = sum(
        1 for tid in included for v in mapping.templates[tid].fields.values() if v != ABSENT
    )
    excluded = len(mapping.templates) - len(included)
    return (
        f"mapping: {len(included)} templates included, {excluded} excluded, {mapped} field mappings"
    )


def confirm_run(
    run: RunDir,
    *,
    yes: bool = False,
    interactive: bool = True,
    editor: str | None = None,
    input_fn: InputFn = input,
    console: Console | None = None,
    edits: Edits | None = None,
) -> Mapping:
    """Confirm the probe proposal and write ``mapping.json``.

    ``yes`` (or ``interactive=False``) accepts ``probe.yaml`` as it stands. Otherwise the editor
    (``editor`` or ``$EDITOR``; pass ``""`` to force the prompt) opens ``probe.yaml``, or, with
    no editor, each ambiguous field is asked through ``input_fn``. ``edits``
    (``{template_id: {field: id | [ids] | "absent"}}``) are explicit user choices written into
    ``probe.yaml`` first; their fields get provenance ``user``.
    """
    if not run.probe_json.exists():
        raise FileNotFoundError(f"{run.probe_json} not found: run the probe first")
    probe = ProbeResult.load(run.probe_json)
    if not run.probe_yaml.exists():
        write_probe_yaml(probe, run.probe_yaml)
    tables = _load_tables(run, probe)
    if edits:
        data = read_yaml(run.probe_yaml)
        if not isinstance(data, dict):
            raise ConfirmError(["expected a top-level `templates:` mapping"])
        apply_edits(data, edits)
        mapping_from_yaml(probe, data, tables)  # raises before anything is written
        run.probe_yaml.write_text(dump_probe_yaml(data), encoding="utf-8")
    con = make_console(console)
    if yes or not interactive:
        mapping = mapping_from_yaml(probe, read_yaml(run.probe_yaml), tables)
    else:
        editor = os.environ.get("EDITOR", "") if editor is None else editor
        if editor:
            mapping = _edit_loop(run, probe, tables, editor, input_fn, con)
        else:
            data = read_yaml(run.probe_yaml)
            walk_ambiguous(probe, data, tables, input_fn, con)
            run.probe_yaml.write_text(dump_probe_yaml(data), encoding="utf-8")
            mapping = mapping_from_yaml(probe, data, tables)
    mapping.save(run.mapping_json)
    con.print(summarize(mapping))
    return mapping
