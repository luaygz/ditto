# Status

_Updated 2026-10-26._

## Current phase

**v0.2 Analysis and union page** — in progress (v0.1 passed and tagged).

## Gates

| Gate | Criterion | Status |
| --- | --- | --- |
| v0.1 | ARI ≥ 0.95 on synthetic default site at 10k pages; 100k (or scaled) run completes | **passed** — ARI 1.0 at 10k and 100k (`eval/results/synth_default`, `synth_scale`) |
| v0.2 | every optional synthetic field in its union page; SWDE dev union pages < 30k tokens | pending |
| v0.3 | SWDE dev F1 within 5 points of naive baseline at < 1% of its tokens/page | pending |
| v0.4 | schema-precision ≥ 0.8 on 10 dev sites; clean install + README example works | pending |
| v0.5 | every README number regenerates from `eval run`; spend under budget | pending |
| v1.0 | release checklist | pending |

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

- [ ] gate: every optional synthetic field appears in its template's union page
- [ ] gate: SWDE development-slice union pages under 30k tokens each

## Next

v0.2 gate measurement; probe/mapping robustness (auto-accepted mappings become hints); SWDE dev accuracy tuning toward the v0.3 gate; Common Crawl download (~6 h, running).

## Known blockers

None.

## LLM spend so far

$0.36 (agent smoke tests and the fixture run); budget $60.
