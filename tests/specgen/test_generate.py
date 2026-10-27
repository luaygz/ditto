"""Unit tests for one-template spec generation (``generate_spec``) with a fake LLM."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from typing import Any

import pytest

from ditto.analyze.models import Candidate, CandidateTable, Table, TableColumn
from ditto.llm.base import CacheMiss, Completion, Message, Usage
from ditto.schema import Alternative, FieldRequest
from ditto.specgen.generate import (
    TemplateInput,
    build_messages,
    format_candidate_table,
    format_fields_section,
    format_tables,
    generate_spec,
    llm_output_schema,
    schema_text,
)
from ditto.specgen.models import Spec
from ditto.specgen.prompts import load_prompt

MODEL = "openai/gpt-6.1-sol"

PAGE = """<html><head><title>Dr. Amal Khoury - Staff</title></head><body>
<div id="profile"><h1 itemprop="name">Amal Khoury</h1>
<dl class="contact"><dt>Email</dt><dd><a href="mailto:a.khoury@example.org">a.khoury@example.org</a></dd>
<dt>Office</dt><dd>B-204</dd></dl>
<table id="pubs"><tr><th>Title</th><th>Year</th></tr>
<tr><td>Wrappers at scale</td><td>2021</td></tr><tr><td>Template drift</td><td>2023</td></tr></table>
</div><div class="related"><h2>Colleagues</h2><a href="/staff/l-haddad">L. Haddad</a></div>
</body></html>"""

UNION = """<html><body><div id="profile">
<!-- c1 · mandatory · on 100% of pages · e.g. Amal Khoury -->
<h1 itemprop="name">Amal Khoury</h1>
<dl class="contact"><dt>Email</dt>
<!-- c2 · optional · on 60% of pages · e.g. a.khoury@example.org -->
<dd><a href="mailto:a.khoury@example.org">a.khoury@example.org</a></dd>
<dt>Office</dt><!-- c3 · mandatory · on 100% of pages · e.g. B-204 --><dd>B-204</dd></dl>
<table id="pubs"><tr><th>Title</th><th>Year</th></tr>
<tr><!-- c4 · list · on 80% of pages · e.g. Wrappers at scale --><td>Wrappers at scale</td>
<!-- c5 · list · on 80% of pages · e.g. 2021 --><td>2021</td></tr>
<!-- collapsed: 6 more siblings like the one above (variant 1 of 1) -->
</table></div></body></html>"""


def candidate_table() -> CandidateTable:
    def cand(cid: str, path: str, cls: str, p: float, **kw: Any) -> Candidate:
        return Candidate.model_validate(
            {"id": cid, "path": path, "class": cls, "p": p, "m": 0.1, **kw}
        )

    return CandidateTable(
        template_id="t_aaaa",
        cluster_size=40,
        n_sampled=40,
        candidates=[
            cand("c0", "html > head > title", "label", 1.0, m=1.0, examples=["Staff"]),
            cand(
                "c1",
                "html > body > div#profile > h1[itemprop=name]",
                "mandatory",
                1.0,
                examples=["Amal Khoury", "Lina Haddad"],
            ),
            cand(
                "c2",
                "html > body > div#profile > dl.contact > dd > a",
                "optional",
                0.6,
                label_hint="Email",
                value_type="email",
                examples=["a.khoury@example.org"],
            ),
            cand(
                "c3",
                "html > body > div#profile > dl.contact > dd",
                "mandatory",
                1.0,
                label_hint="Office",
                examples=["B-204", "C-110"],
            ),
            cand(
                "c4",
                "html > body > div#profile > table#pubs > tr > td",
                "list",
                0.8,
                repeat_min=1,
                repeat_max=9,
                table_id="T1",
                table_role="record_column",
                column="Title",
                examples=["Wrappers at scale"],
            ),
            cand(
                "c5",
                "html > body > div#profile > table#pubs > tr > td",
                "list",
                0.8,
                repeat_min=1,
                repeat_max=9,
                table_id="T1",
                table_role="record_column",
                column="Year",
                examples=["2021", "2023"],
                value_type="number",
            ),
            cand("c6", "html > body > div.related > a @href", "optional", 0.5, attr="href"),
        ],
        tables=[
            Table(
                id="T1",
                kind="record_list",
                container_path="html > body > div#profile > table#pubs",
                row_path="html > body > div#profile > table#pubs > tr",
                p=0.8,
                rows_min=1,
                rows_max=9,
                columns=[
                    TableColumn(name="Title", candidate_id="c4"),
                    TableColumn(name="Year", candidate_id="c5"),
                ],
            ),
            Table(
                id="T2",
                kind="key_value",
                container_path="html > body > div#profile > dl.contact",
                row_path="html > body > div#profile > dl.contact > dt",
                p=1.0,
                rows_min=1,
                rows_max=2,
                key_candidate="c7",
                value_candidate="c3",
                keys_seen=["Email", "Office"],
            ),
        ],
    )


REQUESTED = [
    FieldRequest(name="name", description="full name"),
    FieldRequest(name="email", description="contact email", type_check="email"),
    FieldRequest(name="publications", type="record_list"),
    FieldRequest(name="phone", description="phone number"),
]

NAME = {
    "type": "scalar",
    "candidates": ["//h1[@itemprop='name']", "//div[@id='profile']//h1"],
    "source_candidates": ["c1"],
}
EMAIL = {
    "type": "scalar",
    "candidates": [
        "//dt[normalize-space()='Email']/following-sibling::dd[1]//a",
        "//a[starts-with(@href,'mailto:')]",
    ],
    "source_candidates": ["c2"],
}
PUBS = {
    "type": "record_list",
    "container": "//table[@id='pubs']//tr[td]",
    "columns": {
        "title": "./td[count(ancestor::table//th[normalize-space()='Title']/preceding-sibling::th)+1]",
        "year": "./td[count(ancestor::table//th[normalize-space()='Year']/preceding-sibling::th)+1]",
    },
    "column_source": "header_row",
    "transforms": {"year": ["cast(int)"]},
    "source_candidates": ["c4", "c5"],
}
BAD_EMAIL = {"type": "scalar", "candidates": ["/html/body/div/dl/dd[1]/a", "//dl/dd/a"]}


def answer(fields: dict[str, Any], not_present: dict[str, str] | None = None) -> str:
    return json.dumps({"fields": fields, "not_present": not_present or {}})


class FakeLLM:
    """Implements the ``ditto.llm.base.LLM`` protocol from canned answers."""

    def __init__(
        self,
        responses: list[str | Exception] | Callable[[list[Message], str | None], str],
        *,
        model: str = MODEL,
        role: str = "generation",
    ) -> None:
        self.model = model
        self.role = role
        self._responses = responses
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
            self.calls.append(
                {"messages": messages, "template_id": template_id, "prompt_version": prompt_version}
            )
            if callable(self._responses):
                out: str | Exception = self._responses(messages, template_id)
            else:
                out = self._responses.pop(0)
        if isinstance(out, Exception):
            raise out
        return Completion(
            text=out,
            usage=Usage(input_tokens=1000, output_tokens=200),
            model=self.model,
            cost_usd=0.004,
        )


def make_input(**kw: Any) -> TemplateInput:
    base: dict[str, Any] = {
        "template_id": "t_aaaa",
        "site": "example.org",
        "mode": "prompted",
        "table": candidate_table(),
        "union_html": UNION,
        "size": 40,
        "url_patterns": ["/staff/{slug}"],
        "url_examples": ["https://example.org/staff/a-khoury"],
        "pages_seen": ["https://example.org/staff/a-khoury", "https://example.org/staff/l-haddad"],
        "shown": {"c1", "c2", "c3", "c4", "c5"},
        "fields": list(REQUESTED),
    }
    base.update(kw)
    return TemplateInput(**base)


# -- prompt rendering ----------------------------------------------------------------------------


def test_output_schema_leaves_verifier_keys_out() -> None:
    schema = llm_output_schema()
    assert set(schema["properties"]) == {"fields", "not_present"}
    for name in ("ScalarField", "ListField", "RecordListField", "KeyValueField"):
        props = schema["$defs"][name]["properties"]
        assert not {"stats", "selected", "fallbacks", "degraded", "selection_reason"} & set(props)
        assert "type" in schema["$defs"][name]["required"]
    assert '"title":' not in schema_text()  # no schema titles
    assert len(schema_text()) < 8000  # ~2k tokens with the field descriptions


def test_candidate_table_is_compact_and_marks_unshown() -> None:
    text = format_candidate_table(candidate_table(), shown={"c1", "c2"})
    lines = text.splitlines()
    assert lines[0].startswith("id | class | p")
    assert not any(line.startswith("c0 ") for line in lines)  # labels are not field rows
    c2 = next(line for line in lines if line.startswith("c2 "))
    assert '"Email"' in c2 and "0.60" in c2 and "email" in c2
    c4 = next(line for line in lines if line.startswith("c4 "))
    assert "1-9" in c4 and "T1:record_column (Title)" in c4
    assert "not on union page; path: html > body > div#profile > dl.contact > dd" in text
    assert any(line.startswith("c6 @href") for line in lines)


def test_tables_and_fields_sections() -> None:
    tables = format_tables(candidate_table())
    assert "T1 record_list: 1-9 rows" in tables and "Title (c4, header cell)" in tables
    assert "keys seen: Email, Office" in tables
    inp = make_input(
        mapped={"name": ["c1"]},
        suggested={
            "email": [
                Alternative(
                    id="c3", use="c3", kind="mandatory", label="Office", examples=["B-204"]
                ),
                Alternative(id="T1", use=["c4", "c5"], kind="record_list table"),
            ]
        },
        unmatched=["phone"],
    )
    section = format_fields_section(inp)
    assert "`name`: full name. Confirmed mapping: c1" in section
    assert (
        "`email` (type_check email): contact email. Probe suggestions, best first: "
        'c3 (mandatory, label "Office", e.g. "B-204"); T1 = [c4, c5] (record_list table)'
    ) in section
    assert "`phone`: phone number. Probe found no candidate" in section
    assert "`publications` (record_list)\n" in section + "\n"  # no marker without a mapping
    assert "choose the entity fields" in format_fields_section(make_input(mode="unprompted"))


def test_v2_prompt_explains_the_mapping_markers() -> None:
    prompt = load_prompt("specgen")
    assert prompt.id == "specgen.v2"
    for marker in (
        "Confirmed mapping",
        "Probe suggestions, best first",
        "Probe found no candidate",
    ):
        assert marker in prompt.text


def test_messages_fill_every_placeholder() -> None:
    prompt = load_prompt("specgen")
    msgs = build_messages(prompt, make_input(), schema_text())
    assert [m["role"] for m in msgs] == ["system", "user"]
    system, user = msgs[0]["content"], msgs[1]["content"]
    assert "XPath 1.0 only" in system and '"not_present"' in system
    assert "{{" not in system + user
    assert "Mode: prompted" in user and "Template t_aaaa: 40 pages" in user
    assert UNION.strip() in user


# -- generate_spec -------------------------------------------------------------------------------


def test_prompted_happy_path() -> None:
    llm = FakeLLM(
        [answer({"name": NAME, "email": EMAIL, "publications": PUBS}, {"phone": "No phone"})]
    )
    gen = generate_spec(make_input(), llm=llm)
    assert gen.status == "ok" and gen.reason is None
    assert gen.calls == 1 and gen.input_tokens == 1000 and gen.cost_usd == pytest.approx(0.004)
    spec = gen.spec
    assert isinstance(spec, Spec)
    assert spec.template_id == "t_aaaa" and spec.site == "example.org"
    assert spec.generated_by.model == MODEL
    assert spec.generated_by.prompt_version == "specgen.v2"
    assert spec.generated_by.mode == "prompted"
    assert spec.generated_by.pages_seen[0] == "https://example.org/staff/a-khoury"
    assert spec.generated_by.created_at
    assert set(spec.fields) == {"name", "email", "publications"}
    # Pipeline fills what it knows: description/type_check from the request, optional from p.
    email = spec.fields["email"]
    assert email.description == "contact email" and email.optional is True
    assert getattr(email, "type_check", None) == "email"
    assert spec.fields["name"].optional is False
    assert spec.fields["publications"].optional is True
    assert llm.calls[0]["prompt_version"] == "specgen.v2"
    assert llm.calls[0]["template_id"] == "t_aaaa"
    # The draft spec works on a real page.
    row = spec.apply(PAGE)
    assert row["name"] == "Amal Khoury"
    assert row["email"] == "a.khoury@example.org"
    assert row["publications"] == [
        {"title": "Wrappers at scale", "year": 2021},
        {"title": "Template drift", "year": 2023},
    ]


def test_not_present_is_reported() -> None:
    llm = FakeLLM(
        [answer({"name": NAME, "email": EMAIL, "publications": PUBS}, {"phone": "no phone label"})]
    )
    gen = generate_spec(make_input(), llm=llm)
    assert gen.status == "ok"
    assert gen.spec is not None and gen.spec.not_present == {"phone": "no phone label"}
    assert gen.not_present == {"phone": "no phone label"}
    assert "phone" not in gen.spec.fields


def test_invalid_then_fixed_retry_carries_errors() -> None:
    llm = FakeLLM(
        [
            answer({"name": NAME, "email": BAD_EMAIL, "publications": PUBS}, {"phone": "none"}),
            answer({"name": NAME, "email": EMAIL, "publications": PUBS}, {"phone": "none"}),
        ]
    )
    gen = generate_spec(make_input(), llm=llm)
    assert gen.status == "ok" and gen.calls == 2
    retry = llm.calls[1]["messages"]
    assert len(retry) > len(llm.calls[0]["messages"])
    feedback = "\n".join(m["content"] for m in retry[len(llm.calls[0]["messages"]) :])
    assert "fields.email" in feedback and "purely positional" in feedback


def test_missing_requested_field_triggers_retry() -> None:
    llm = FakeLLM(
        [
            answer({"name": NAME, "email": EMAIL, "publications": PUBS}),
            answer({"name": NAME, "email": EMAIL, "publications": PUBS}, {"phone": "absent"}),
        ]
    )
    gen = generate_spec(make_input(), llm=llm)
    assert gen.status == "ok" and gen.calls == 2
    feedback = llm.calls[1]["messages"][-1]["content"]
    assert "requested field 'phone' is missing" in feedback


def test_double_failure_keeps_valid_fields_and_records_failed_ones() -> None:
    bad = answer({"name": NAME, "email": {**BAD_EMAIL, "source_candidates": ["c2"]}}, {})
    llm = FakeLLM([bad, bad])
    gen = generate_spec(make_input(mapped={"publications": ["c4", "c5"]}), llm=llm)
    assert gen.status == "partial" and gen.calls == 2
    assert gen.spec is not None and set(gen.spec.fields) == {"name"}
    assert set(gen.failed_fields) == {"email", "publications", "phone"}
    email = gen.failed_fields["email"]
    assert email.source_candidates == ["c2"] and email.type == "scalar"
    assert email.type_check == "email" and email.description == "contact email"
    assert any("purely positional" in e for e in email.errors)
    pubs = gen.failed_fields["publications"]
    assert pubs.source_candidates == ["c4", "c5"] and pubs.type == "record_list"
    assert gen.to_json()["failed_fields"]["email"]["source_candidates"] == ["c2"]
    assert gen.reason is not None and gen.reason.startswith("invalid after retry")


def test_double_failure_unparseable_fails_template_with_mapped_candidates() -> None:
    llm = FakeLLM(["Sorry, I cannot help.", "still not json"])
    gen = generate_spec(make_input(mapped={"name": ["c1"]}), llm=llm)
    assert gen.status == "failed" and gen.spec is None
    assert set(gen.failed_fields) == {"name", "email", "publications", "phone"}
    assert gen.failed_fields["name"].source_candidates == ["c1"]
    assert gen.failed_fields["publications"].type == "record_list"


def test_llm_error_fails_template() -> None:
    llm = FakeLLM([CacheMiss("offline and not cached")])
    gen = generate_spec(make_input(), llm=llm)
    assert gen.status == "failed" and gen.reason is not None and "CacheMiss" in gen.reason
    assert set(gen.failed_fields) == {"name", "email", "publications", "phone"}


def test_prompted_drops_unrequested_and_matches_names_case_insensitively() -> None:
    llm = FakeLLM(
        [
            answer(
                {
                    "Name": NAME,
                    "email": {**EMAIL, "stats": {"presence_rate": 1.0}, "selected": 1},
                    "publications": PUBS,
                    "office": {
                        **NAME,
                        "candidates": ["//dt[.='Office']/following-sibling::dd[1]", "//dl//dd[2]"],
                    },
                },
                {"phone": "none", "fax": "none"},
            )
        ]
    )
    gen = generate_spec(make_input(), llm=llm)
    assert gen.status == "ok"
    assert gen.spec is not None and set(gen.spec.fields) == {"name", "email", "publications"}
    assert gen.spec.fields["email"].stats.presence_rate is None  # verifier-owned keys dropped
    assert getattr(gen.spec.fields["email"], "selected", None) == 0
    assert set(gen.dropped_fields) == {"office", "fax"}


def test_requested_type_is_enforced() -> None:
    pubs_as_list = {
        "type": "list",
        "candidates": ["//table[@id='pubs']//td", "//table[@id='pubs']//tr/td"],
    }
    llm = FakeLLM(
        [
            answer({"name": NAME, "email": EMAIL, "publications": pubs_as_list}, {"phone": "none"}),
            answer({"name": NAME, "email": EMAIL, "publications": PUBS}, {"phone": "none"}),
        ]
    )
    gen = generate_spec(make_input(), llm=llm)
    assert gen.status == "ok" and gen.calls == 2
    assert "type must be 'record_list'" in llm.calls[1]["messages"][-1]["content"]


def test_unknown_source_candidates_are_dropped_with_warning() -> None:
    llm = FakeLLM(
        [
            answer(
                {
                    "name": {**NAME, "source_candidates": ["c1", "c99"]},
                    "email": EMAIL,
                    "publications": PUBS,
                },
                {"phone": "x"},
            )
        ]
    )
    gen = generate_spec(make_input(), llm=llm)
    assert gen.spec is not None and gen.spec.fields["name"].source_candidates == ["c1"]
    assert any("c99" in w for w in gen.warnings)


def test_mapping_absent_fields_go_to_not_present() -> None:
    fields = [f for f in REQUESTED if f.name != "phone"]
    llm = FakeLLM([answer({"name": NAME, "email": EMAIL, "publications": PUBS})])
    gen = generate_spec(make_input(fields=fields, absent=["phone"]), llm=llm)
    assert gen.status == "ok" and gen.spec is not None
    assert gen.spec.not_present["phone"].startswith("marked absent")


def test_unprompted_requires_snake_case_names() -> None:
    llm = FakeLLM(
        [
            answer({"Full Name": NAME, "email": EMAIL}),
            answer({"full_name": NAME, "email": EMAIL}, {"ignored": "x"}),
        ]
    )
    gen = generate_spec(make_input(mode="unprompted", fields=[]), llm=llm)
    assert gen.status == "ok" and gen.calls == 2
    assert "snake_case" in llm.calls[1]["messages"][-1]["content"]
    assert gen.spec is not None and set(gen.spec.fields) == {"full_name", "email"}
    assert gen.spec.not_present == {} and gen.spec.generated_by.mode == "unprompted"


def test_unprompted_empty_answer_fails() -> None:
    llm = FakeLLM([answer({}), answer({})])
    gen = generate_spec(make_input(mode="unprompted", fields=[]), llm=llm)
    assert gen.status == "failed" and gen.spec is None and gen.failed_fields == {}


def test_code_fenced_answer_is_accepted() -> None:
    body = answer({"name": NAME, "email": EMAIL, "publications": PUBS}, {"phone": "none"})
    llm = FakeLLM([f"```json\n{body}\n```"])
    gen = generate_spec(make_input(), llm=llm)
    assert gen.status == "ok"
