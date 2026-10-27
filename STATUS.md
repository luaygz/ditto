# Status

_Updated 2026-10-27._

## Current phase

**v0.2 Analysis and union page** — in progress (v0.1 passed and tagged).

## Gates

| Gate | Criterion | Status |
| --- | --- | --- |
| v0.1 | ARI ≥ 0.95 on synthetic default site at 10k pages; 100k (or scaled) run completes | **passed** — ARI 1.0 at 10k and 100k (`eval/results/synth_default`, `synth_scale`) |
| v0.2 | every optional synthetic field in its union page; SWDE dev union pages < 30k tokens | **passed** — 48/48 optional fields; max 29,822 tokens over 23 SWDE dev templates (`eval/results/union_gate_*`); tag after the checkpoint work lands (HEAD must pass `make check`) |
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

- [x] gate: every optional synthetic field appears in its template's union page (48/48)
- [x] gate: SWDE development-slice union pages under 30k tokens each (max 29,822; book-amazon needed budget-aware splicing)

## Next

- Accuracy-2: mapping provenance (`user`/`auto`), specgen.v2, probe fixes, SWDE dev tuning toward the v0.3 gate (ditto F1 >= 0.808 vs naive 0.858 on the same 50-page samples), integration test with recorded responses.
- Then: unprompted mode + schema-precision judgment (v0.4), SWDE test / Common Crawl / LiveWeb-IE evaluations and baselines (v0.5), release.
- Common Crawl polite download running (1 req/s, cap 2,500/domain, ~5 h); labels + noise check run automatically afterwards.

## Known blockers

None blocking. Common Crawl's CDN blocked this IP after ~440 MB per run at 4 concurrent requests; the download now runs at 1 request/s with back-off (see DECISIONS.md).

## LLM spend so far

$26.34 of an $85 budget (owner topped up the account; full evaluation plan estimated at ≈ $81 total).
