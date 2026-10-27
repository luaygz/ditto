# Python API

```python
from ditto import FieldsSchema, Mapping, Probe, Site, Spec
```

The API runs the same stages as the CLI on the same run directory, so a run can be started in
Python and continued with `ditto <command>` (or the other way round). Stages that need an LLM read
the same `.env` settings as the CLI ([cli.md](cli.md#configuration)). `ditto.__version__` is the
installed version.

Page-level work runs in worker processes. Where they are spawned rather than forked (macOS and
Windows by default, or `DITTO_MP_START=spawn`), each worker re-imports your script, so a script
that calls the API must keep those calls under `if __name__ == "__main__":`, as usual with
`multiprocessing`.

## A whole site

```python
from ditto import Site

site = Site.from_dir("pages/", run_dir="runs/example")    # HTML directory (+ optional urls.tsv)
# site = Site.from_warc(["crawl-1.warc.gz", "crawl-2.warc.gz"], run_dir="runs/example")
# site = Site.open("runs/example")                       # reopen an existing run directory

site.cluster()                                           # Clusters (clusters.json)
site.analyze()                                           # {template_id: CandidateTable}
probe = site.probe("fields.yaml")                        # Probe; probe.result is the ProbeResult
mapping = probe.confirm()                                # Mapping (mapping.json)
site.generate("prompted", schema="fields.yaml", mapping=mapping)   # 1 LLM call per template
site.verify(max_repair_calls=3)                          # {template_id: VerifyReport}
site.review()                                            # ReviewResult; golden/<template>.jsonl
site.extract()                                           # ExtractSummary; out/, report.json/.md

for record in site.records():                            # every row of every out/<site>/*.jsonl
    print(record["url"], record["template_id"], record["title"])
specs = site.specs()                                     # {template_id: Spec}, final specs
```

## Reference

### `Site`

| Member | Does |
| --- | --- |
| `Site.from_dir(path, manifest=None, *, run_dir=None, workers=None)` | Ingests an HTML directory (`manifest`: a `urls.tsv` of `path<TAB>url`; default `path/urls.tsv` when present) into `run_dir` (default `./ditto-runs/<directory name>`). |
| `Site.from_warc(paths, *, run_dir=None, workers=None)` | Ingests one WARC/WARC.gz path or a list of them (directories holding only WARC files work too) into `run_dir` (default `./ditto-runs/<stem of the first path>`). |
| `Site.open(run_dir)` / `Site(run_dir)` | Wraps an existing run directory; `site.run` is its `RunDir` (every file name). |
| `site.cluster(*, min_cluster_size=20, workers=None)` | Fingerprints and clusters; returns `Clusters`. |
| `site.analyze(*, include_tail=False, workers=None)` | Candidate tables and union pages; returns `{template_id: CandidateTable}`. |
| `site.probe(schema, *, use_llm=False)` | Maps the fields of `schema` (a `FieldsSchema` or a `fields.yaml` path) to candidates of every non-tail template; `use_llm=True` resolves ambiguous fields with the small model. Returns a `Probe`. |
| `site.generate(mode, schema=None, mapping=None, llm=None, *, min_cluster_size=20)` | One spec-generation call per template (`mode`: `"prompted"` or `"unprompted"`); prompted mode reads `mapping.json` when `mapping` is not given and skips the templates the mapping excludes or does not list. Returns a `GenerationResult`. Raises `ditto.llm.BudgetExceeded` before any call when the projected cost exceeds the remaining budget; a budget that runs out mid-run returns normally with `result.aborted` and `result.unprocessed` set. |
| `site.verify(max_repair_calls=3, *, llm=None)` | Scores every candidate on every page, repairs, may merge or split clusters (rewriting `clusters.json`), writes final specs; `max_repair_calls=0` makes no LLM call. Returns `{template_id: VerifyReport}`. |
| `site.review(*, n=5, seed=0)` | Non-interactive review: samples `n` records per template, saves golden samples, returns a `ReviewResult`. |
| `site.extract(*, workers=None)` | Runs the final specs over every page (no LLM) and writes the report; returns an `ExtractSummary` (pages, seconds, pages per second per core, per-template pages and filled counts per field). |
| `site.records()` | Iterates every extracted row of every `out/<site>/*.jsonl` (including `_tail.jsonl`). |
| `site.specs()` | `{template_id: Spec}` of the final specs. |

`llm=` takes any object implementing the `ditto.llm.base.LLM` protocol; by default the
`generation` and `repair` roles (and `probe` for `site.probe(use_llm=True)`) come from
`ditto.llm.get_llm`.

### `Probe.confirm(edits=None)`

Accepts `probe.yaml` as it stands (the probe's proposal plus any hand edits) and writes
`mapping.json`. `edits` overrides it per template and field: `{"t_0c6f": {"tags": "c12",
"price": "absent"}}`, where a value is a candidate id, a list of ids (or a table id) for record
and key-value groups, or `"absent"`. Edits go through `probe.yaml` and are validated like a hand
edit: an unknown template, field or candidate raises `ditto.checkpoints.ConfirmError` and nothing
is written. Edited fields count as chosen by a person (provenance `user`, binding for
generation); the rest of the accepted proposal is recorded as `auto` and passed to the LLM as
suggestions ([how-it-works.md](how-it-works.md#6-generate-a-spec-one-llm-call)).

### `Mapping`

`Mapping.load(path)` / `mapping.save(path)` read and write `mapping.json`. `mapping.templates`
maps a template id to a `TemplateMapping`: `included`, `fields` (field → candidate id, list of ids
or `"absent"`), `provenance` (field → `"user"` or `"auto"`; a missing entry reads as `"user"`),
`alternatives` (the probe's top three per field) and `note`. `mapping.included_templates()` lists
the included template ids.

### `FieldsSchema`

`FieldsSchema.load("fields.yaml")` (or `FieldsSchema.from_obj(data)`) reads the prompted-mode
schema: a mapping of field names, optionally under a top-level `fields:` key. Each field is either
a one-line description or a mapping:

```yaml
title: the main title or headline shown as the page heading
price: the listed price, including currency symbol
email: {description: contact email, type: scalar, examples: [a@b.org], type_check: email}
```

`type` is one of `scalar`, `list`, `record_list`, `key_value` (default: the LLM decides);
`type_check` is one of `email`, `url`, `number`, `date`, `price`, `text`.

## One spec on one page

A finished spec needs nothing but the page's HTML; this is the call to embed in your own scraper:

```python
from ditto import Spec

spec = Spec.load("runs/example/specs/t_b071.json")
row = spec.apply(html)          # html: str or bytes of one page
# {'title': 'The Year', 'published': 'Jan 14, 2019', 'tags': ['music'],
#  '_meta': {'missing': [], 'spec_version': '1'}}
```

`apply` cleans the page exactly as extraction does (`<script>`, `<style>`, `<noscript>`, `<svg>`,
`<iframe>`, `<template>`, comments and `<head>` except `<title>` removed), evaluates each field's
selected XPath with its fallbacks and transforms (`normalize-space()` treats no-break spaces as
spaces, as everywhere in ditto), and returns the fields plus `_meta.missing` (fields with no value
on this page). No LLM, no network and no other file is involved. `Spec` is a Pydantic model:
`Spec.load(path)`, `spec.save(path)`, and the format in [spec-format.md](spec-format.md) (JSON
Schema: [spec.schema.json](spec.schema.json)).
