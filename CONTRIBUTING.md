# Contributing

## Setup

```bash
uv sync --all-extras          # Python 3.11+, creates .venv from uv.lock
cp .env.example .env          # only needed for live LLM tests and real runs
make check                    # ruff, ruff format --check, mypy --strict, pytest with coverage >= 80%
```

The test suite needs no API key and makes no network calls. Tests marked `@pytest.mark.live`
run only when `OPENAI_API_KEY` is set and cost a few cents.

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
- Every stage except spec generation, repair and the opt-in probe call must run without an LLM.
- Never commit datasets, downloaded HTML, `.env`, keys, or LLM caches containing page content.
  `data/` and `.cache/` are gitignored. The only committed LLM responses are the recorded ones
  for the synthetic fixture site under `tests/fixtures/llm/`.
- Benchmark numbers in the README come only from `ditto eval run eval/configs/<name>.yaml`
  outputs under `eval/results/`.

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
