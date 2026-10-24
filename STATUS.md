# Status

_Updated 2026-10-24._

## Current phase

**v0.1 Foundations** — in progress.

## Gates

| Gate | Criterion | Status |
| --- | --- | --- |
| v0.1 | ARI ≥ 0.95 on synthetic default site at 10k pages; 100k (or scaled) run completes | pending |
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

- [ ] ingest (directory + manifest, WARC) into `pages.sqlite`
- [ ] fingerprinting (signatures, text hashes) at 100k-page scale
- [ ] clustering (exact groups, chrome filter, MinHash LSH, Jaccard confirm, URL tie-break)
- [ ] synthetic generator with default parameters + 200-page fixture site
- [ ] unit tests for all of the above
- [ ] gate: ARI ≥ 0.95 at 10k pages; 100k run with clustering metrics recorded

## Next

Parallel build of all pipeline stages against the shared contracts; dataset acquisition (SWDE, Common Crawl, LiveWeb-IE) runs alongside.

## Known blockers

None.

## LLM spend so far

$0.00 (two key/model checks, < $0.001).
