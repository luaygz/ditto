"""confirm_run: probe.yaml round trip into mapping.json (--yes, hand edits, prompt, editor)."""

from __future__ import annotations

import json
import sys
import textwrap
from collections.abc import Callable
from pathlib import Path

import pytest
import yaml
from checkpoint_builders import clusters, job_schema, job_table, kv_table, listing_table, make_run
from rich.console import Console

from ditto.checkpoints import ConfirmError, confirm_run, probe_run
from ditto.run import RunDir
from ditto.schema import FieldsSchema, Mapping


def scripted(*answers: str) -> Callable[[str], str]:
    queue = list(answers)
    prompts: list[str] = []

    def input_fn(prompt: str) -> str:
        prompts.append(prompt)
        if not queue:
            raise AssertionError(f"unexpected prompt: {prompt!r}")
        return queue.pop(0)

    input_fn.prompts = prompts  # type: ignore[attr-defined]
    return input_fn


def quiet() -> Console:
    return Console(file=open("/dev/null", "w"), width=120)


@pytest.fixture
def run(tmp_path: Path) -> RunDir:
    cl = clusters({"t_job1": 120, "t_list1": 40})
    run = make_run(tmp_path / "run", [job_table(), listing_table()], cl)
    probe_run(run, job_schema())
    return run


def test_yes_accepts_the_proposal(run: RunDir) -> None:
    mapping = confirm_run(run, yes=True, console=quiet())
    on_disk = json.loads(run.mapping_json.read_text())
    assert on_disk["version"] == 1
    assert on_disk["schema_fields"] == job_schema().names()
    assert on_disk["templates"]["t_job1"] == {
        "included": True,
        "fields": {
            "title": "c2",
            "company": "c4",
            "location": "c5",
            "date_posted": "c6",
            "salary": "c7",  # ambiguous: the best guess is proposed
            "remote_policy": "absent",
        },
        "note": None,
    }
    assert on_disk["templates"]["t_list1"]["included"] is False
    assert Mapping.load(run.mapping_json) == mapping
    assert mapping.included_templates() == ["t_job1"]


def test_hand_edited_yaml_is_used_with_yes(run: RunDir) -> None:
    data = yaml.safe_load(run.probe_yaml.read_text())
    data["templates"]["t_job1"]["fields"]["salary"]["use"] = "absent"
    data["templates"]["t_job1"]["fields"]["title"] = "c1"  # short form: field: id
    data["templates"]["t_job1"]["note"] = "titles carry the site name"
    data["templates"]["t_list1"]["include"] = True
    del data["templates"]["t_job1"]["fields"]["company"]  # missing -> proposal kept
    run.probe_yaml.write_text(yaml.safe_dump(data))
    mapping = confirm_run(run, yes=True, console=quiet())
    job = mapping.templates["t_job1"]
    assert job.fields["salary"] == "absent"
    assert job.fields["title"] == "c1"
    assert job.fields["company"] == "c4"
    assert job.note == "titles carry the site name"
    assert mapping.templates["t_list1"].included is True


@pytest.mark.parametrize(
    ("use", "message"),
    [
        ("c3", "c3 is a label, which is never a field value"),
        ("c0", "c0 is a boilerplate"),
        ("c42", "unknown candidate id 'c42'"),
        (17, "`use` must be a candidate id, a list of ids or absent"),
        ([17], "17 is not a candidate id"),
    ],
)
def test_invalid_ids_are_rejected(run: RunDir, use: object, message: str) -> None:
    data = yaml.safe_load(run.probe_yaml.read_text())
    data["templates"]["t_job1"]["fields"]["company"]["use"] = use
    run.probe_yaml.write_text(yaml.safe_dump(data))
    with pytest.raises(ConfirmError) as exc:
        confirm_run(run, yes=True, console=quiet())
    assert any(message in p for p in exc.value.problems)
    assert not run.mapping_json.exists()


def test_structural_errors_are_reported(run: RunDir) -> None:
    data = yaml.safe_load(run.probe_yaml.read_text())
    data["templates"]["t_job1"]["include"] = "maybe"
    data["templates"]["t_job1"]["fields"]["colour"] = {"use": "c2"}
    data["templates"]["t_zzz"] = {}
    run.probe_yaml.write_text(yaml.safe_dump(data))
    with pytest.raises(ConfirmError) as exc:
        confirm_run(run, yes=True, console=quiet())
    problems = "\n".join(exc.value.problems)
    assert "t_job1.include: expected true or false" in problems
    assert "t_job1.colour: unknown field" in problems
    assert "t_zzz: unknown template" in problems
    run.probe_yaml.write_text("templates: [unclosed")
    with pytest.raises(ConfirmError, match="not valid YAML"):
        confirm_run(run, yes=True, console=quiet())


def test_interactive_walk_resolves_ambiguous_and_excluded(run: RunDir) -> None:
    # salary is the only ambiguous field; then the excluded listing template is offered
    input_fn = scripted("absent", "t_list1")
    mapping = confirm_run(run, editor="", input_fn=input_fn, console=quiet())
    assert input_fn.prompts[0].startswith("  salary [1-1, candidate id, absent; Enter = c7]")  # type: ignore[attr-defined]
    assert mapping.templates["t_job1"].fields["salary"] == "absent"
    assert mapping.templates["t_list1"].included is True
    # the decisions are written back into probe.yaml
    data = yaml.safe_load(run.probe_yaml.read_text())
    assert data["templates"]["t_job1"]["fields"]["salary"]["use"] == "absent"
    assert data["templates"]["t_list1"]["include"] is True
    assert run.probe_yaml.read_text().startswith("# ditto probe")
    # and a later --yes keeps them
    assert confirm_run(run, yes=True, console=quiet()) == mapping


def test_interactive_walk_retries_invalid_answers(run: RunDir) -> None:
    input_fn = scripted("c3", "c99", "1", "")
    mapping = confirm_run(run, editor="", input_fn=input_fn, console=quiet())
    assert mapping.templates["t_job1"].fields["salary"] == "c7"
    assert mapping.templates["t_list1"].included is False
    assert len(input_fn.prompts) == 4  # type: ignore[attr-defined]


def test_interactive_walk_accepts_other_candidate_ids(run: RunDir) -> None:
    mapping = confirm_run(run, editor="", input_fn=scripted("c9", ""), console=quiet())
    assert mapping.templates["t_job1"].fields["salary"] == "c9"


def test_editor_round_trip(run: RunDir, tmp_path: Path) -> None:
    script = tmp_path / "fake_editor.py"
    script.write_text(
        textwrap.dedent(
            """
            import sys, yaml
            path = sys.argv[1]
            data = yaml.safe_load(open(path))
            data["templates"]["t_job1"]["fields"]["salary"]["use"] = "c9"
            data["templates"]["t_list1"]["include"] = True
            open(path, "w").write(yaml.safe_dump(data))
            """
        )
    )
    mapping = confirm_run(run, editor=f"{sys.executable} {script}", console=quiet())
    assert mapping.templates["t_job1"].fields["salary"] == "c9"
    assert mapping.templates["t_list1"].included is True
    assert Mapping.load(run.mapping_json) == mapping


def test_editor_from_environment_and_reopen_on_error(
    run: RunDir, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # first save introduces a bad id, second save fixes it
    script = tmp_path / "flaky_editor.py"
    script.write_text(
        textwrap.dedent(
            f"""
            import sys, yaml, pathlib
            path = sys.argv[1]
            marker = pathlib.Path({str(tmp_path / "opened")!r})
            data = yaml.safe_load(open(path))
            use = "c77" if not marker.exists() else "c5"
            marker.write_text("x")
            data["templates"]["t_job1"]["fields"]["salary"]["use"] = use
            open(path, "w").write(yaml.safe_dump(data))
            """
        )
    )
    monkeypatch.setenv("EDITOR", f"{sys.executable} {script}")
    mapping = confirm_run(run, input_fn=scripted("y"), console=quiet())
    assert mapping.templates["t_job1"].fields["salary"] == "c5"


def test_editor_failure_writes_nothing(run: RunDir) -> None:
    with pytest.raises(ConfirmError, match="exited with status"):
        confirm_run(run, editor=f"{sys.executable} -c 'raise SystemExit(3)'", console=quiet())
    with pytest.raises(ConfirmError, match="cannot run editor"):
        confirm_run(run, editor="/nonexistent/editor", console=quiet())
    assert not run.mapping_json.exists()


def test_confirm_needs_a_probe(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="run the probe first"):
        confirm_run(RunDir(tmp_path), yes=True, console=quiet())


def test_group_fields_and_key_notes(tmp_path: Path) -> None:
    run = make_run(tmp_path / "r", [kv_table()], clusters({"t_kv1": 50}))
    schema = FieldsSchema.from_obj(
        {"salary": "pay", "reviews": {"type": "record_list", "description": "user reviews"}}
    )
    probe_run(run, schema)
    # a table id in probe.yaml expands to its cells
    data = yaml.safe_load(run.probe_yaml.read_text())
    assert data["templates"]["t_kv1"]["fields"]["reviews"]["use"] == ["c4", "c5"]
    data["templates"]["t_kv1"]["fields"]["reviews"]["use"] = "tb1"
    run.probe_yaml.write_text(yaml.safe_dump(data))
    mapping = confirm_run(run, yes=True, console=quiet())
    tm = mapping.templates["t_kv1"]
    assert tm.fields == {"salary": "c2", "reviews": ["c4", "c5"]}
    assert tm.note == "salary: value of key 'Salary' in c2"


def test_missing_probe_yaml_is_regenerated(run: RunDir) -> None:
    run.probe_yaml.unlink()
    mapping = confirm_run(run, yes=True, console=quiet())
    assert run.probe_yaml.exists()
    assert mapping.templates["t_job1"].fields["company"] == "c4"
