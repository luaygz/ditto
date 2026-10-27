# Status

_Updated 2026-10-28._

## Current phase

**Released v1.0.0** — all six gates passed and tagged (v0.1.0–v1.0.0).

## Gates

| Gate | Criterion | Status |
| --- | --- | --- |
| v0.1 | ARI ≥ 0.95 on synthetic default site at 10k pages; 100k (or scaled) run completes | **passed** — ARI 1.0 at 10k and 100k (`eval/results/synth_default`, `synth_scale`) |
| v0.2 | every optional synthetic field in its union page; SWDE dev union pages < 30k tokens | **passed** — 48/48 optional fields; max 29,822 tokens over 23 SWDE dev templates (`eval/results/union_gate_*`) |
| v0.3 | SWDE dev F1 within 5 points of naive baseline at < 1% of its tokens/page | **passed** — ditto 0.876 vs naive 0.858 (norm F1, same 50-page samples); 40.6 vs 11,385 tokens/page (0.36%) (`eval/results/swde_dev_ditto/20261027T102344Z`) |
| v0.4 | schema-precision ≥ 0.8 on 10 dev sites; clean install + README example works | **passed** — precision 0.908 (187/206 fields, build-agent judgment; `eval/results/swde_dev_unprompted/20261027T115205Z`); wheel in a fresh venv ran the README example |
| v0.5 | every README number regenerates from `eval run`; spend under budget | **passed** — all 18 configs rerun at a clean commit with pinned models, reruns reproduce earlier numbers, `eval/report.py --check` up to date; ledger $86.76 (plan $85; overrun explained in BUILD_REPORT.md) |
| v1.0 | release checklist | **passed** — wheel 1.0.0 installs in a clean venv and runs the fixture offline (recorded responses) and with a key; CI green on all 6 jobs + nightly; BUILD_REPORT.md |

## Done

- Repo skeleton: `pyproject.toml` (uv, hatchling, src layout), `Makefile` (`make check`), CI workflow, MIT LICENSE, `.gitignore` (data/, .cache/, .env).
- PyPI name check: `ditto` taken → distribution `ditto-extract` (see DECISIONS.md).
- API key check passed; main model `gpt-6.1-sol`, small model `gpt-5.4-mini`; prices recorded in `config/prices.yaml`.
- Shared contracts: run-directory layout (`ditto.run`), structural paths and page cleaning (`ditto.fingerprint.dom`), spec models + XPath lint + transforms + runner (`Spec.apply`), candidate-table, clusters and mapping models, LLM interface.

## v0.1 task list

- [x] ingest (directory + manifest, WARC) into `pages.sqlite`
- [x] fingerprinting (signatures, text hashes) at 100k-page scale
- [x] clustering (exact groups, chrome filter, MinHash LSH, Jaccard confirm, URL tie-break)
- [x] synthetic generator with default parameters + 200-page fixture site
- [x] unit tests for all of the above (`make check`: 1,152 tests, 93.6% coverage)
- [x] gate: ARI 1.0 at 10k; 100k run completes (ARI 1.0, ~60 s end to end on 16 cores)

All other stages also exist (analysis, union page, spec generation, verification/repair, extraction, checkpoints, LLM layer, evaluation harness, baselines); the fixture site runs end to end through `ditto run --yes` for $0.053.

## v0.2 task list

- [x] gate: every optional synthetic field appears in its template's union page (48/48)
- [x] gate: SWDE development-slice union pages under 30k tokens each (max 29,822; book-amazon needed budget-aware splicing)

## Next

Done. Post-v1 ideas are in BUILD_REPORT.md ("Next"). Publishing to PyPI needs the owner's token (`make publish`).

## Known blockers

None. The OpenAI account is close to empty; the release steps need no LLM.

## LLM spend so far

$86.76 (ledger) of the $85 plan; per phase and per evaluation in BUILD_REPORT.md and the README.
