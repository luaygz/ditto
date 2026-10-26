"""Shared helpers for the CLI, Python API and pipeline tests: the 200-page fixture site and a
fake LLM that answers every role offline.

Generation answers are valid specs built from the template's candidate table with the same
path-induced selectors verification falls back to: in prompted mode for the candidates
``mapping.json`` assigns to each field (fields without a usable selector go to
``not_present``), in unprompted mode for the first field candidates. Probe answers mark every
asked field absent, repair answers change nothing, tail answers are empty rows.
"""

from __future__ import annotations

import json
import shutil
import threading
from pathlib import Path
from typing import Any

from ditto.analyze.models import CandidateTable
from ditto.llm.base import Completion, Message, Usage
from ditto.run import RunDir
from ditto.schema import ABSENT, Mapping
from ditto.verify.fallback import induced_field

FIXTURE = Path(__file__).parent / "fixtures" / "site"
SITE = FIXTURE / "site"
FIELDS = FIXTURE / "fields.yaml"
FIELD_NAMES = ("title", "published", "tags")
MODEL = "openai/gpt-6.1-sol"  # priced in config/prices.yaml, so budget checks see a real cost
SMALL = "openai/gpt-5.4-mini"
UNPROMPTED_FIELDS = 3


def spec_answer(run: RunDir, template_id: str, messages: list[Message]) -> dict[str, Any]:
    table = CandidateTable.load(run.candidates(template_id))
    user = "\n".join(m["content"] for m in messages)
    wanted: dict[str, list[str]] = {}
    if "None requested" in user:  # unprompted: name the first field candidates
        for c in table.candidates:
            if c.is_field and len(wanted) < UNPROMPTED_FIELDS:
                wanted[f"field_{c.id}"] = [c.id]
    elif run.mapping_json.exists():
        tm = Mapping.load(run.mapping_json).templates[template_id]
        for name, use in tm.fields.items():
            if use != ABSENT:
                wanted[name] = [use] if isinstance(use, str) else list(use)
    else:
        return {"fields": {}, "not_present": {n: "fake: no mapping" for n in FIELD_NAMES}}
    fields: dict[str, Any] = {}
    not_present: dict[str, str] = {}
    for name, ids in wanted.items():
        spec_field = induced_field(ids, table)
        if spec_field is None:
            not_present[name] = "fake: no selector for the mapped candidates"
        else:
            fields[name] = spec_field.model_dump(mode="json", exclude_none=True, by_alias=True)
    if "None requested" not in user:  # every requested name needs an answer (extras are dropped)
        for name in FIELD_NAMES:
            if name not in fields:
                not_present.setdefault(name, "fake: marked absent in mapping.json")
    return {"fields": fields, "not_present": not_present}


class FakeLLM:
    """Implements the ``ditto.llm.base.LLM`` protocol with no network; ``fail`` (an exception)
    is raised by every call instead of answering."""

    def __init__(
        self,
        run: RunDir | None,
        role: str = "generation",
        *,
        model: str = MODEL,
        fail: Exception | None = None,
    ) -> None:
        self.run = run
        self.role = role
        self.model = model
        self.fail = fail
        self.calls: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def complete(
        self,
        messages: list[Message],
        *,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int = 4096,
        prompt_version: str = "",
        template_id: str | None = None,
    ) -> Completion:
        with self._lock:
            self.calls.append({"prompt_version": prompt_version, "template_id": template_id})
        if self.fail is not None:
            raise self.fail
        prompt = prompt_version.split(".")[0]
        answer: dict[str, Any]
        if prompt == "specgen":
            assert self.run is not None and template_id is not None
            answer = spec_answer(self.run, template_id, messages)
        elif prompt == "probe":
            asked = (json_schema or {})["properties"]["fields"]["required"]
            answer = {"fields": {f: {"candidate": ABSENT, "reason": "fake"} for f in asked}}
        elif prompt == "repair":
            answer = {"fields": {}}
        else:  # tail_direct: an empty row (every field missing)
            answer = {}
        return Completion(text=json.dumps(answer), usage=Usage(1000, 200), model=self.model)


class FakeLLMs:
    """Stand-in for ``ditto.llm.get_llm``: hands out one :class:`FakeLLM` per request."""

    def __init__(self, fail: Exception | None = None) -> None:
        self.fail = fail
        self.made: list[FakeLLM] = []

    def __call__(
        self,
        role: str,
        *,
        settings: Any = None,
        run: RunDir | None = None,
        no_cache: bool = False,
        **_: Any,
    ) -> FakeLLM:
        model = SMALL if role in ("probe", "tail_direct") else MODEL
        llm = FakeLLM(run, role, model=model, fail=self.fail)
        self.made.append(llm)
        return llm

    def calls(self, role: str | None = None) -> list[dict[str, Any]]:
        return [c for llm in self.made if role in (None, llm.role) for c in llm.calls]


def copy_run(src: Path, dest: Path) -> RunDir:
    """A private copy of a prepared run directory (run directories are relocatable)."""
    shutil.copytree(src, dest)
    return RunDir(dest)
