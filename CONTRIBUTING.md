# Contributing

## Setup

```bash
uv sync --all-extras          # Python 3.11+, creates .venv from uv.lock
cp .env.example .env          # only needed for live LLM tests and real runs
make check                    # ruff, ruff format --check, mypy --strict, pytest with coverage >= 80%
```

The test suite needs no API key and makes no network calls. Tests marked `@pytest.mark.live`
run only when `OPENAI_API_KEY` is set and cost a few cents.

## Environment variables

Read from the environment or a `.env` file (searched from the working directory upward; the
environment wins). `.env.example` lists the common ones.

| Variable | Default | Meaning |
| --- | --- | --- |
| `OPENAI_API_KEY` | | Key for the OpenAI-compatible adapter; required for every LLM call that is not served from the cache. |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible endpoint (OpenRouter, vLLM, Ollama, Together). |
| `OPENAI_MODEL` | auto-selected | Main model (roles `generation`, `repair`, `baseline`); when unset, picked from `GET {base_url}/models` and cached. |
| `OPENAI_MODEL_SMALL` | auto-selected | Small model (roles `probe`, `tail_direct`). |
| `ANTHROPIC_API_KEY` | | Needed for `anthropic/<model>` references (native Anthropic adapter). |
| `DITTO_MODEL_<ROLE>` | | Per-role override as `provider/model`, e.g. `DITTO_MODEL_GENERATION=anthropic/<model>`; roles `GENERATION`, `REPAIR`, `PROBE`, `TAIL_DIRECT`, `BASELINE`; a bare id means `openai/`. Wins over `OPENAI_MODEL*`. |
| `LLM_BUDGET_USD` | `40` | Global budget over the spend ledger `<cache dir>/ledger.jsonl`; a run whose projected cost exceeds the remainder is refused with the totals. |
| `DITTO_RUN_BUDGET_USD` | none | Optional extra budget per run directory, over that run's `usage.jsonl`. |
| `DITTO_LLM_TIMEOUT` | `600` | Request timeout in seconds. |
| `DITTO_CACHE_DIR` | `.cache` | Response cache (`<dir>/llm/`) and spend ledger. The integration test points it at `tests/fixtures/llm`. |
| `DITTO_LLM_OFFLINE` | off | `1`/`true`: answer only from the cache; a miss raises `CacheMiss` instead of calling the provider. Cannot be combined with `--no-cache`. |
| `DITTO_WORKERS` | all cores | Default worker processes for page-level work (`--workers` wins). |
| `EDITOR` | | Editor `ditto confirm` opens `probe.yaml` in; unset means a terminal walk-through. |

### Service tiers

ditto's own calls use the provider's default tier. The naive per-page and AutoScraper baselines
(`eval/baselines/`, constant `SERVICE_TIER = "flex"`) request OpenAI's Flex tier, which bills
about half and is slower (timeout raised to 15 minutes); when Flex has no capacity
(`flex_unavailable`/`resource_unavailable` 429) the request is sent once more on the default
tier. The tier only changes billing, not the model or its answers, so cache keys are shared. The
ledger bills each call at the tier the response reports, using the multipliers under `tiers:` in
`config/prices.yaml` (`flex` and `batch`: 0.5); `list_cost_usd` and every cost table stay at
standard-tier list prices so methods compare fairly.

## Layout

`src/ditto/<stage>/` holds one pipeline stage each (ingest, fingerprint, cluster, analyze, union,
specgen, verify, extract, checkpoints, llm, synth); `ditto.evaluation` holds metrics and the
`ditto eval run` harness; `eval/` holds dataset scripts, baselines, configs and committed results.
Stage contracts (file names in a run directory, entry-point signatures) live in `ditto.run` and
`ditto.pipeline`. `docs/how-it-works.md` describes the data flow.

## Rules

- A spec is data: XPath 1.0 plus the closed transform set. Never add a code path that executes
  model output.
- Prompts are versioned files `prompts/<name>.v<N>.md`. A changed prompt is a new version file;
  never edit a released one in place. The version is recorded in every spec.
- Every stage except spec generation, repair, the opt-in probe call and the opt-in
  `--tail direct` mode must run without an LLM.
- Never commit datasets, downloaded HTML, `.env`, keys, or LLM caches containing page content.
  `data/` and `.cache/` are gitignored. The only committed LLM responses are the recorded ones
  for the synthetic fixture site under `tests/fixtures/llm/`.
- Benchmark numbers in the README and `docs/benchmarks.md` come only from
  `ditto eval run eval/configs/<name>.yaml` outputs under `eval/results/`. The tables between
  `<!-- results:<section>:start -->` and `<!-- results:<section>:end -->` are written by
  `uv run python eval/report.py` (which also draws `docs/img/*.png`); never edit them by hand.
  `uv run python eval/report.py --check` fails when they are out of date.

## Re-recording the fixture LLM responses

The integration test runs `ditto run --yes` on `tests/fixtures/site/` with
`DITTO_LLM_OFFLINE=1` and `DITTO_CACHE_DIR=tests/fixtures/llm`. After changing a prompt or the
union page format, re-record with a key:

```bash
rm -rf tests/fixtures/llm/llm
DITTO_CACHE_DIR=tests/fixtures/llm uv run ditto run tests/fixtures/site/site \
  --out /tmp/fixture-run --schema tests/fixtures/site/fields.yaml --yes
```

## Releasing

1. Update `CHANGELOG.md` and the version in `pyproject.toml`; run `make check`.
2. `make build` produces the sdist and wheel in `dist/`. Verify the wheel in a clean environment:
   `python -m venv /tmp/v && /tmp/v/bin/pip install dist/*.whl && /tmp/v/bin/ditto --help`.
3. Tag `vX.Y.Z` and push the tag; CI must be green on the tagged commit.
4. Publishing needs a PyPI API token for the `ditto-extract` project: export it as
   `UV_PUBLISH_TOKEN` and run `make publish` (which runs `uv publish`). The distribution name is
   `ditto-extract` because `ditto` is taken on PyPI; the import name and CLI are `ditto`.
