# CLI reference

`ditto` is a [Typer](https://typer.tiangolo.com/) app installed by `pip install ditto-extract`.
Every option below is listed by `ditto <command> --help`. All stages read and write one *run
directory* (layout at the end of this page), so any stage can be rerun alone; the usual path is
`ditto run`, which chains them.

| Command | What it does | LLM |
| --- | --- | --- |
| [`ditto ingest`](#ditto-ingest) | reads an HTML directory (optional `urls.tsv`) or WARC files into a run directory | no |
| [`ditto cluster`](#ditto-cluster) | fingerprints and clusters pages into templates | no |
| [`ditto analyze`](#ditto-analyze) | candidate tables and one union page per template | no |
| [`ditto probe`](#ditto-probe) | maps requested fields to candidates per template | only with `--probe-llm` (small model) |
| [`ditto confirm`](#ditto-confirm) | edits or accepts the probe's mapping | no |
| [`ditto generate`](#ditto-generate) | one spec-generation call per template; draft specs | yes (main model) |
| [`ditto verify`](#ditto-verify) | scores candidates on every page, repairs, finalizes specs | repair calls only (main model) |
| [`ditto review`](#ditto-review) | five extracted records per template for sign-off; golden samples | no |
| [`ditto extract`](#ditto-extract) | runs the final specs over every page; JSONL and report | only with `--tail direct` (small model) |
| [`ditto run`](#ditto-run) | all of the above in order | yes |
| [`ditto eval run`](#ditto-eval-run) | runs one benchmark config into `eval/results/` | depends on the config |
| [`ditto synth generate`](#ditto-synth-generate) | builds a synthetic site with ground truth | no |

Conventions shared by every command:

- `RUN` is a run directory created by `ditto ingest` (or `ditto run`); a directory without
  `pages.sqlite` is rejected as a usage error.
- `--workers N` sets the worker processes for page-level work (default: all cores, or
  `DITTO_WORKERS`).
- `--no-cache` skips the LLM response cache lookup under `.cache/llm/` (fresh answers are still
  stored). It cannot be combined with `DITTO_LLM_OFFLINE=1`.
- `.env` in the current directory (or a parent) is read at startup; see
  [Configuration](#configuration).
- Exit status: `0` on success; `1` with a one-line `error: ...` message for expected failures (a
  missing file, an invalid `fields.yaml`, a provider error, a checkpoint without a terminal, the
  LLM budget, which the message reports with its totals); `2` for usage errors (an unknown
  option, a missing required argument, an invalid `--mode` or `--tail` value, a `RUN` that is not
  a run directory).

## ditto ingest

```text
ditto ingest SRC... --out RUN [--manifest PATH] [--site NAME] [--workers N]
```

| Argument / option | Default | Meaning |
| --- | --- | --- |
| `SRC...` (required) | | An HTML directory, or one or more WARC/WARC.gz files. |
| `--out`, `-o` (required) | | Run directory to create. |
| `--manifest` | `SRC/urls.tsv` when present | `urls.tsv` manifest (`path<TAB>url` per line) for a directory. |
| `--site` | inferred from the URLs | Site name (the `out/<site>/` directory). |
| `--workers` | all cores | Worker processes. |

Keeps the latest capture per URL, skips non-HTML records, drops byte-identical duplicates, and
detects each page's encoding. Without a manifest, URLs fall back to file paths. Writes
`pages.sqlite`, `blobs/` (directory input; WARC input is read in place by offset) and `run.json`.

## ditto cluster

```text
ditto cluster RUN [--min-cluster-size 20] [--workers N]
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--min-cluster-size` | `20` | Templates below this size are the long tail. |
| `--workers` | all cores | Worker processes. |

Fingerprints every page, then clusters them into templates. Writes `fingerprints/`,
`clusters.json` and `cluster_report.md` (sizes, URL patterns and examples per template).

## ditto analyze

```text
ditto analyze RUN [--include-tail | --no-include-tail] [--max-tokens 30000] [--workers N]
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--include-tail` / `--no-include-tail` | `--no-include-tail` | Also analyze tail clusters (needed for `generate --tail spec`). |
| `--max-tokens` | `30000` | Union page token cap (tiktoken `o200k_base`). |
| `--workers` | all cores | Worker processes. |

Writes `candidates/<template>.json` (the candidate table) and `union/<template>.html` plus
`union/<template>.json` (token count, base page, donors, compression log).

## ditto probe

```text
ditto probe RUN --schema fields.yaml [--probe-llm] [--no-cache]
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--schema` (required) | | `fields.yaml` with the requested fields. |
| `--probe-llm` | off | Resolve ambiguous fields with the small model. |
| `--no-cache` | off | Bypass the LLM response cache. |

Scores every candidate against every requested field without an LLM and marks each field
`found`, `ambiguous` or `missing` per template. Writes `probe.json` and the editable
`probe.yaml`.

## ditto confirm

```text
ditto confirm RUN [--yes]
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--yes`, `-y` | off | Accept the proposal. |

Without `--yes`, opens `probe.yaml` in `$EDITOR` or, when `EDITOR` is unset, walks through the
ambiguous fields in the terminal; it needs an interactive terminal and otherwise exits with an
error asking for `--yes`. Writes `mapping.json` (template → field → candidate id or `absent`).

## ditto generate

```text
ditto generate RUN [--mode prompted|unprompted] [--schema fields.yaml] [--min-cluster-size 20]
               [--tail skip|spec] [--concurrency 4] [--no-cache]
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--mode` | `prompted` | `prompted` (extract the fields in `--schema`) or `unprompted` (the LLM names the fields). |
| `--schema` | | `fields.yaml`; required in prompted mode. |
| `--min-cluster-size` | `20` | Templates below this size are the long tail. |
| `--tail` | `skip` | `skip`, or `spec` to generate specs for tail clusters too. Per-page tail extraction is `ditto extract --tail direct`. |
| `--concurrency` | `4` | Concurrent LLM calls. |
| `--no-cache` | off | Bypass the LLM response cache. |

One main-model call per template. Prompted runs use `mapping.json` when it exists. Writes
`specs/draft/<template>.json`, `generation.json` and, in unprompted mode,
`schema_proposal.json`. If the LLM budget runs out mid-way, the drafts written so far are kept,
the unprocessed templates are listed and the command exits with status 1.

## ditto verify

```text
ditto verify RUN [--max-repair-calls 3] [--workers N] [--no-cache]
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--max-repair-calls` | `3` | Repair calls per template (`0`: no LLM at all). |
| `--workers` | all cores | Worker processes. |
| `--no-cache` | off | Bypass the LLM response cache. |

Runs every candidate selector on every page of its cluster, picks the winners, repairs failing
fields within the call budget and may merge or split clusters. Writes the final
`specs/<template>.json` and `verify/<template>.json`.

## ditto review

```text
ditto review RUN [--yes]
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--yes`, `-y` | off | Skip sign-off. |

Shows five extracted records per template beside their source values and asks for sign-off
(interactive terminal required without `--yes`). Writes `golden/<template>.jsonl` and
`review.json`.

## ditto extract

```text
ditto extract RUN [--workers N] [--tail skip|direct] [--schema fields.yaml] [--no-cache]
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--workers` | all cores | Worker processes. |
| `--tail` | `skip` | `skip`, or `direct`: one small-model call per long-tail page. |
| `--schema` | | `fields.yaml`; required with `--tail direct`. |
| `--no-cache` | off | Bypass the LLM response cache. |

Runs the final specs over every page with no LLM. Writes `out/<site>/<template>.jsonl` (one row
per page: `url`, `page_id`, `template_id`, the fields, `_meta: {missing, spec_version}`),
`report.json` and `report.md`. `--tail direct` rows carry `template_id: null`.

## ditto run

```text
ditto run SRC... --out RUN [--mode prompted|unprompted] [--schema fields.yaml] [--yes]
          [--tail skip|direct|spec] [--min-cluster-size 20] [--probe-llm]
          [--max-repair-calls 3] [--workers N] [--no-cache]
```

| Argument / option | Default | Meaning |
| --- | --- | --- |
| `SRC...` (required) | | An HTML directory, or WARC/WARC.gz files. |
| `--out`, `-o` (required) | | Run directory to create. |
| `--mode` | `prompted` | `prompted` or `unprompted`. |
| `--schema` | | `fields.yaml`; required in prompted mode. |
| `--yes`, `-y` | off | Accept both checkpoints. |
| `--tail` | `skip` | `skip`, `direct` (prompted mode only) or `spec`. |
| `--min-cluster-size` | `20` | Templates below this size are the long tail. |
| `--probe-llm` | off | Resolve ambiguous fields with the small model. |
| `--max-repair-calls` | `3` | Repair calls per template. |
| `--workers` | all cores | Worker processes. |
| `--no-cache` | off | Bypass the LLM response cache. |

Order: ingest, cluster, analyze, then (prompted) probe and the mapping checkpoint, generate,
(unprompted) the schema checkpoint, verify, the review checkpoint, extract, and the tail when
`--tail direct`. Without `--yes` the checkpoints ask in the terminal; without a terminal the
command exits with an error asking for `--yes`.

## ditto eval run

```text
ditto eval run CONFIG [--results-dir eval/results] [--dry-run | --no-dry-run]
```

| Argument / option | Default | Meaning |
| --- | --- | --- |
| `CONFIG` (required) | | `eval/configs/<name>.yaml`. |
| `--results-dir` | `eval/results` | Where results are written. |
| `--dry-run` / `--no-dry-run` | `--no-dry-run` | Validate the config and list the planned sites (writes `plan.json`, no stages run). |

Writes `<results-dir>/<name>/<UTC timestamp>/` with `results.json`, `results.csv`,
`per_site.csv` and `meta.json` (model ids, prompt versions, git commit, dataset manifest hashes,
machine, spend). A dataset that is not on disk is recorded as `not_run` with the reason. The
Crawl4AI baseline needs the `eval` extra (`pip install 'ditto-extract[eval]'`), as do the charts
of `eval/report.py`. See [benchmarks.md](benchmarks.md).

## ditto synth generate

```text
ditto synth generate --params PATH [--seed 1] [--out synth] [--workers N]
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--params` (required) | | Parameter YAML, e.g. `eval/synth/default.yaml`. |
| `--seed` | `1` | Random seed. |
| `--out`, `-o` | `synth` | Output directory. |
| `--workers` | all cores | Worker processes. |

Needs the `synth` extra (`pip install 'ditto-extract[synth]'`, for Faker). Writes `site/*.html`
and `site/urls.tsv` (what ditto ingests), `truth/<page>.json` sidecars, `truth/templates.json`,
`truth/fields/<template>.yaml` (prompted-mode schemas) and `synth.json`. The output depends only
on the parameters and the seed.

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `OPENAI_API_KEY` | | Key for the OpenAI-compatible adapter (required for LLM stages). |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | Any OpenAI-compatible endpoint (OpenRouter, vLLM, Ollama, Together). |
| `OPENAI_MODEL` / `OPENAI_MODEL_SMALL` | auto-selected from `GET /models` | Main model (generation, repair, baseline) and small model (probe, tail). |
| `ANTHROPIC_API_KEY` | | For `anthropic/<model>` model references. |
| `DITTO_MODEL_<ROLE>` | | Per-role model as `provider/model` (`GENERATION`, `REPAIR`, `PROBE`, `TAIL_DIRECT`, `BASELINE`). |
| `LLM_BUDGET_USD` | `40` | Total budget over the spend ledger; runs projected above the remainder are refused. |
| `DITTO_RUN_BUDGET_USD` | none | Optional budget per run directory. |
| `DITTO_LLM_TIMEOUT` | `600` | Request timeout in seconds. |
| `DITTO_CACHE_DIR` | `.cache` | Response cache and spend ledger location. |
| `DITTO_LLM_OFFLINE` | off | Serve recorded responses only; a cache miss is an error. |
| `DITTO_WORKERS` | all cores | Default worker processes. |

More detail in [CONTRIBUTING.md](../CONTRIBUTING.md#environment-variables).

## Run directory

```text
run.json  pages.sqlite  blobs/                  ingest
fingerprints/  clusters.json  cluster_report.md  cluster
candidates/<t>.json  union/<t>.html  union/<t>.json   analyze
probe.json  probe.yaml  mapping.json             probe, confirm
specs/draft/<t>.json  generation.json  schema_proposal.json   generate
specs/<t>.json  verify/<t>.json                  verify
golden/<t>.jsonl  review.json                    review
out/<site>/<t>.jsonl  report.json  report.md     extract
usage.jsonl  timings.json                        every LLM call; every stage's timing
```
