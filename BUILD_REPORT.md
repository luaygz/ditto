# ditto v1.0 build report

What was built, what each gate measured, every waiver and shortfall, what the LLM cost, how the
build ran, and what to do next. Numbers are quoted with the file they come from; the README
tables are regenerated from `eval/results/` by `eval/report.py` and are the reference. Spots
marked **[INTEGRATOR]** wait for the final Common Crawl evaluations.

## What was built

**The pipeline** (`src/ditto/`, one package per stage, each writing into a run directory so any
stage reruns alone):

1. `ingest`: HTML directories (optional `urls.tsv`) and WARC files into `pages.sqlite` plus
   content-addressed blobs; latest capture per URL, byte-identical duplicates dropped, HTML5
   encoding detection order.
2. `fingerprint`: one lxml parse per page, scripts/styles/head stripped, utility and hashed
   classes canonicalized, structural paths, sorted path-hash signatures in a CSR store.
3. `cluster`: exact signature groups, a 90% chrome filter, MinHash LSH with exact Jaccard
   confirmation, URL-pattern tie-break.
4. `analyze`: presence and modal share per path, candidate classes, label hints, key-value and
   record-table detection.
5. `union`: one annotated union page per template (base page, aligned splicing, collapse by
   variant, budget-aware splicing, a hard 30k-token cap).
6. `specgen`: one LLM call per template in prompted or unprompted mode, versioned prompts, the
   Pydantic spec model (JSON Schema in `docs/spec.schema.json`), XPath 1.0 lint, closed
   transform grammar.
7. `verify`: every candidate on every page, winner by presence × multiplicity × type pass,
   fallbacks, repair calls within a budget, path-induced fallback, cluster merge and split.
8. `extract`: JSONL per template and a report with no LLM; `Spec.load(path).apply(html)` for use
   in other scrapers.

**Checkpoints** (`checkpoints/`): `probe` (offline matcher, optional small-model call), `confirm`
(mapping with provenance: `user` choices bind generation, `auto` ones are ranked suggestions),
`review` (five records per template, golden samples, repair of rejected fields), and the
unprompted schema proposal; `ditto run --yes` passes all of them.

**LLM layer** (`llm/`): OpenAI-compatible and Anthropic adapters behind one interface, model
auto-selection, roles (generation, repair, probe, tail_direct, baseline), a response cache keyed
on prompt version and messages, retries with back-off, a usage ledger, a budget guard that
reserves a worst-case projection while a call runs, structured JSON with one validated retry,
Flex billing at the tier multiplier.

**Interfaces**: Typer CLI (`ditto ingest … run`, `eval run`, `synth generate`), the Python
`Site` API, `make check` / `report` / `install-check` / `publish`, CI on Ubuntu and macOS for
Python 3.11 and 3.12 plus a nightly synthetic run. `uv run pytest --collect-only` lists 1,490
tests (2026-10-27).

**Evaluation harness** (`src/ditto/evaluation/`, `eval/`): YAML configs with pinned models,
budgets, page samples and seeds; per-site scoring with three match columns (`raw`, `norm`,
`canon`), unseen pages only, optional-field recall, unprompted alignment and judged schema
precision, clustering metrics, cost from tokens at list prices, stage speed; results as JSON
and CSV with git commit, model ids, prompt versions, metric version and dataset hashes;
`eval/report.py` writes every README/docs table and chart, pairs every baseline with ditto on
the same sites and pages, and prints the ledger per build phase.

**Datasets** (`eval/datasets/`): SWDE (download, structural verification, ground-truth parser,
dev/test split, field descriptions written before tuning); a Common Crawl silver set (8 domains
in 4 verticals chosen by a scripted procedure, capture index, polite downloader, JSON-LD labels,
noise check); a LiveWeb-IE snapshot (release check, permissions, polite fetcher, audit); a
synthetic site generator with ground-truth sidecars and stress parameters.

**Baselines** (`eval/baselines/`): naive per-page LLM, Crawl4AI `generate_schema`, AutoScraper
(upstream code at a pinned commit, model calls through ditto's LLM layer, its `eval` replaced),
hand-written XPath wrappers for five SWDE test sites written by the build agent.

## Gates

| gate | criterion | measured | result | source |
| --- | --- | --- | --- | --- |
| v0.1 | ARI ≥ 0.95 on the synthetic default site at 10k pages; a 100k run completes | 15 templates, seed 1 | passed: ARI 1.0, purity 1.0, no over- or under-merge at 10,000 and at 100,000 pages | `eval/results/synth_default/20261026T095112Z`, `eval/results/synth_scale/20261026T095119Z` |
| v0.2 | every optional synthetic field in its union page; SWDE dev union pages < 30k tokens | 12 synthetic templates; 23 templates on the 16 dev sites | passed: 48 of 48 optional fields; SWDE dev union pages 1,955–29,822 tokens, none over the cap | `eval/results/union_gate_synth/20261027T101206Z`, `eval/results/union_gate_swde_dev/20261027T101223Z` |
| v0.3 | SWDE dev F1 within 5 points of the naive baseline on its 200-page samples, at < 1% of its tokens per page | both on the same seeded **50-page** samples (waiver below) | passed: normalized F1 0.876 (ditto) vs 0.858 (naive); 33.5 vs 11,385 tokens per page pooled (0.29%; 40.6 tokens, 0.36%, as the mean over sites used when the gate was recorded); $0.073 vs $23.13 per 1,000 pages | `eval/results/swde_dev_ditto/20261027T102344Z` (`accuracy_sample`), `eval/results/swde_dev_naive/20261027T085024Z` |
| v0.4 | schema precision ≥ 0.8 on 10 dev sites; clean install + README example | precision judged per field by the build agent; wheel in a fresh Python 3.11 venv | passed: 0.908 (187 of 206 proposed fields); the README example ran from a scratch directory with only `.env` and reproduced the `t_0c6f` row | `eval/results/swde_dev_unprompted/20261027T115205Z`, `eval/judgments/swde_dev_schema_precision.json`; install check in `DECISIONS.md` (2026-10-27) |
| v0.5 | every README number regenerates from `eval run` with a pinned model id; spend under budget with a per-evaluation breakdown | final cached reruns of every config, `make report`, `eval/report.py --check` | **[INTEGRATOR]** after the Common Crawl evaluations and the final reruns | `eval/results/*`, README "LLM spend" |
| v1.0 | release checklist | wheel in a clean venv, CI green on the tag, `BUILD_REPORT.md`, tag `v1.0.0` | **[INTEGRATOR]** | — |

The v0.3 and v0.4 figures come from runs scored before metric version 2 (no `canon` column);
the final cached reruns rescore them with identical `norm` and `raw` predictions and add
`canon`.

## Results at release

From the latest runs at the time of writing (metric version 1, before the final reruns):

- SWDE test split, 64 sites: ditto normalized F1 0.846, raw 0.771, $0.0777 per 1,000 pages,
  1.45 LLM calls per template (`eval/results/swde_test_ditto/20261027T112642Z`). On the naive
  baseline's seeded 10-page samples: ditto 0.843, naive 0.845 at $22.59 per 1,000 pages
  (extrapolated) (`eval/results/swde_test_naive/20261027T102925Z`). AutoScraper on 8 sites:
  0.880 against ditto's 0.733 on the same pages (`eval/results/swde_test_autoscraper/20261027T112853Z`).
  Hand-written wrappers on 5 sites: 1.000 against ditto's 0.906 on the same pages
  (`eval/results/swde_test_handwritten/20261027T102047Z`). Crawl4AI on its 36 sites:
  **[INTEGRATOR: from the README table after the final rerun of `swde_test_crawl4ai`]**.
- SWDE dev slice, unprompted mode: recall of the labeled attributes 0.847 after name alignment
  (`eval/results/swde_dev_unprompted/20261027T115205Z`).
- LiveWeb-IE snapshot, 9 of 15 sites: normalized F1 0.724 (`eval/results/livewebie_ditto/20261027T122202Z`).
- Common Crawl silver set: ditto **[INTEGRATOR]**, naive per-page baseline on 20-page samples
  **[INTEGRATOR]**, Crawl4AI **[INTEGRATOR]** (`eval/results/cc_silver_*`); pages downloaded
  per domain and the final noise check **[INTEGRATOR]**
  (`eval/datasets/commoncrawl_silver/domains.yaml`, `noise_check.json`).
- Clustering: minimum ARI 0.994 over 62 synthetic stress points; whole templates merge only for
  the near-duplicate pair at Jaccard ≥ 0.8 (`eval/results/synth_curves/20261027T093851Z`).
- Extraction speed: the evaluation runs measured 64 (SWDE dev, unprompted) to 177 (LiveWeb-IE)
  pages per second per core (mean over sites, 16 worker processes, pool start-up per site
  included; `eval/results/*/results.json` `speed_pages_per_sec_per_core.extract`). The bench
  (`uv run python eval/bench/speed.py`, single process, parse included) measured 542 pages/s on
  `auto-aol` (49.6 KB pages) and 475 on `book-abebooks` (63.2 KB pages) on an AMD Ryzen
  Threadripper 1900X (8 cores, 16 threads); source: the Perf-2 agent's bench run (commit
  `79f6a0b`; no results file is committed for the bench).

## Waivers and shortfalls

Each one is also in the README limitations (generated from the results where it has numbers).

- **v0.3 gate on 50-page samples, not 200.** The measured naive cost (about 11.3k tokens, $0.024
  per SWDE page) put 200-page samples on every site near $450 against a $40 default budget. The
  owner raised the budget to $60, then $85 for the full plan; samples were sized to fit: 50 pages
  per site on the SWDE dev slice, 10 on the test split, 20 per Common Crawl domain
  (`DECISIONS.md`, 2026-10-24 and 2026-10-27). The per-page baseline's cost per 1,000 pages is
  extrapolated from these samples.
- **Crawl4AI on 36 of 64 SWDE test sites.** It cost about $0.3–1 per site (many templates per
  site, about four calls per template with its validation loop), more than estimated;
  it was stopped to keep budget for Common Crawl, and its config lists the finished sites so it
  reruns from cache. Its usage hook also recorded less than the ledger billed for its calls, so
  its cost columns are likely understated.
- **AutoScraper on 8 SWDE test sites (one per vertical), 10-page samples.** Budget: it makes
  dozens of calls per template. Its cost per 1,000 pages scales the sample cost linearly though
  its rules are per template, so on whole sites it would cost far less per page.
- **Hand-written wrappers on 5 sites**, as the brief planned; written by the build agent from
  each site's first ten pages, which are also scored.
- **Common Crawl capped at 2,500 pages per domain instead of 20,000.** CloudFront's firewall
  blocked this machine's IP ("Request blocked", HTTP 403) after about 2,250 records in each of
  two runs at 4 concurrent requests. Following Common Crawl's guidance, the downloader now makes
  one request at a time at least one second apart with back-off; 160,000 captures would have
  taken about 44 hours. `argos.co.uk` keeps the 6,701 detail pages fetched before the block plus
  500 others (7,201 planned).
- **Silver labels.** Common Crawl labels are the sites' own JSON-LD; the noise check (build agent,
  automated) found several fields whose labels rarely match the visible text (on the selection
  sample: `ikea.com` name, the job sites' `datePosted`, `marmiton.org` recipeYield, among others;
  `eval/datasets/commoncrawl_silver/noise_check.json`). **[INTEGRATOR: confirm against the
  final noise check on the downloaded pages.]**
- **LiveWeb-IE on 9 of 15 sites.** Excluded: `dp.la` (HTTP 403 for every page),
  `fueleconomy.gov` (robots.txt unreachable, so disallowed per RFC 9309), `iclr.cc`, `icml.cc`,
  `neurips.cc` (most ground-truth values absent from the pages as the extractor sees them),
  `thesportsdb.com` (`Content-Signal: ai-input=no`); `eval/datasets/livewebie/manifest.json`.
  The release is licensed for non-commercial research; ditto is scored on a snapshot with field
  definitions, so its numbers are not comparable with the paper's.
- **Synthetic sites are separable by construction**, so the clustering curves say little about
  real sites beyond scale and markup-noise robustness.
- **`canon` column added late** (metric version 2): rules derived from SWDE dev misses only and
  frozen before the test sets were rescored; results written before it show `–` until rerun.
  The same analysis estimates that SWDE label problems (truncated, over-long or missing labels)
  are about 72% of ditto's remaining dev misses under `norm`, from 84 misses classified by the
  build agent (`eval/analysis/swde_dev_misses.json`).
- **Judgments by the build agent, not a human**: schema precision, the Common Crawl noise check,
  the SWDE miss classification, the hand-written wrappers.
- **Union pages of each template are built from more than three pages when needed.** The brief
  says specs are generated from 1–3 pages per template; 84 of the 118 specs behind the SWDE test,
  SWDE dev and LiveWeb-IE ditto results drew on one to three pages (`generated_by.pages_seen` in
  the run directories under `data/eval/`), but donors for rare optional fields raise that: the
  `auto-automotive` spec drew on 134 pages (its `pages_excluded` in
  `eval/results/swde_test_ditto/20261027T112642Z/results.json`). Every page a union page drew on
  is excluded from scoring.
- **Extraction speed** in the evaluation runs is below the 500 pages/s/core target; the bench
  reaches it on 50 KB pages but not on 63 KB pages (above).
- **Small templates' union pages below the 8k-token minimum are not padded** (all synthetic
  union pages are 1,225–2,077 tokens; `eval/results/union_gate_synth/20261027T101206Z`).
- **Pinned models.** `gpt-6.1-terra` ($1 / $5 per million tokens) appeared after the build
  started and a fresh install auto-selects it; every evaluation stays on `gpt-6.1-sol` and
  `gpt-5.4-mini`, the ids selected at build time. Terra was not evaluated.
- **OpenAI Flex** was mostly without capacity, so baseline calls fell back to standard prices.
- **Anthropic adapter** unit-tested, never called live (no key).
- **Not on PyPI.** `make publish` needs the owner's token.
- **Stage-7 merge rule**: two templates merge when one spec reaches its presence rates on the
  other's pages, so near-duplicate page types share a spec and a JSONL file.
- **Incremental re-crawls and drift detection** are post-v1; specs store the per-field
  statistics they would need.

## LLM spend

Ledger (`.cache/ledger.jsonl`, `cost_usd`) per phase, phases bounded by the commit dates of the
tagged commits (`git log -1 --format=%cI v0.X.0`), as of the last ledger line
(2026-10-27T13:34:53Z). Regenerate with `uv run python eval/report.py --spend-by-phase`.

| phase | until | calls | billed (ledger) | by role |
| --- | --- | --- | --- | --- |
| before v0.1.0 | 2026-10-26T11:55:56+02:00 | 27 | $0.30 | generation $0.17, baseline $0.13 |
| v0.1.0–v0.2.0 | 2026-10-27T12:23:39+02:00 | 968 | $26.96 | baseline $18.49, generation $7.30, repair $1.17 |
| v0.2.0–v0.3.0 | 2026-10-27T12:32:46+02:00 | 140 | $3.80 | baseline $3.80 |
| v0.3.0–v0.4.0 | 2026-10-27T14:06:55+02:00 | 1,180 | $26.94 | baseline $18.34, generation $7.34, repair $1.27 |
| v0.4.0–now | — | 623 | $12.90 | baseline $10.59, generation $2.05, repair $0.26 |
| total | | 2,938 | $70.91 | |

**[INTEGRATOR: rerun `uv run python eval/report.py --spend-by-phase` after the Common Crawl
evaluations and replace the last two rows.]**

Reading the table: the tags were placed on the commits that recorded each gate, and those were
committed together late in the build, so the phases are uneven. "v0.1.0–v0.2.0" holds the SWDE
dev naive baseline (with its two partial runs) and most of the `specgen.v2` tuning on the dev
slice; "v0.3.0–v0.4.0" holds the SWDE test naive baseline, Crawl4AI, AutoScraper and the
unprompted runs; "v0.4.0–now" holds the SWDE test and LiveWeb-IE ditto runs and the rest of
Crawl4AI. List prices equal the billed figures: Flex discounts almost never applied.

Budget: `LLM_BUDGET_USD` defaulted to $40, was raised by the owner to $60 after the naive
baseline's cost was measured and to $85 for the full evaluation plan; the owner topped up the
account by $35 and $15. A billing check found the provider charging $37.70 while the ledger
recorded $30.73 over the same period (about 23% more; cause not found: possibly calls billed but
failed client-side, Crawl4AI calls its hook missed, or other use of the account;
`DECISIONS.md`). The ledger total is under the $85 budget; applying the same undercount to the
whole ledger would put real billing near $87 (an estimate, not a measurement). **[INTEGRATOR:
final total after the Common Crawl evaluations.]** The LLM cost behind each published number
is in the README "LLM spend" table.

## Process notes

- **Parallel agents against fixed contracts.** The main agent wrote the shared contracts first
  (run directory, structural paths and page cleaning, spec models, candidate/cluster/mapping
  models, the LLM interface) and split the build into 15 slice agents. All 15 hit an Anthropic
  API rate limit after about 34 minutes and were resumed in two batches of at most 8. Wave 2 ran
  at most 6 agents with exclusive file ownership (Accuracy, Perf, UnionGate, Docs, Install, then
  Datasets, Unprompted, Metrics and this release slice). Agents never committed; the main agent
  integrated, committed per module, ran the official evaluations and kept `DECISIONS.md` and
  `STATUS.md`.
- **Two owner pauses.** The owner paused the build after wave 1, when 10 of the 15 slices were
  committed; the five running agents were cancelled and their work resumed from disk. The owner
  paused again during wave 2: the Accuracy, Perf and UnionGate agents were killed
  and the Common Crawl download cancelled; replacement agents resumed from the files on disk and
  their predecessors' transcripts.
- **Machine move.** During the first pause the session moved from a laptop (12 cores, 30 GB) to
  a desktop (16 logical CPUs, 62 GB). The copied `.venv` pointed at a missing interpreter and
  was rebuilt from `uv.lock`; every speed number in the results comes from the desktop.
- **External limits met along the way**: Common Crawl's firewall (download cap), Flex capacity
  (standard billing), Crawl4AI's cost per site (fewer sites), a new default model mid-build
  (pinning), and the ledger/billing gap (re-estimates with the observed factor). The owner
  approved a private GitHub repository (`luaygz/ditto`) so CI runs on GitHub Actions; tags
  v0.1.0–v0.4.0 are pushed and CI was green on all six jobs and the nightly run.

## Known issues

Found while checking the docs against the code for this release; not fixed:

- `cli.main()` calls `load_dotenv()` without `usecwd=True`, so for an installed package
  `DITTO_WORKERS` and `DITTO_MP_START` in the working directory's `.env` are ignored (LLM
  settings are read correctly).
- A YAML syntax error in `fields.yaml` prints a traceback instead of an `error:` line; so does a
  refused overwrite in `ditto synth generate`.
- `--workers` is not passed to union-page building.
- Prompted mode drops tail templates: `probe` covers non-tail templates only, so
  `ditto run --mode prompted --tail spec` produces no tail specs; after `--tail spec`,
  `extract --tail direct` still sends every tail page to the model.
- Review repairs call `get_llm("repair")` directly and ignore `--no-cache` and the run's pinned
  models.
- `Site.from_dir` names the default run directory from `Path.stem` (`pages.v2` becomes `pages`).
- `DECISIONS.md` says `Retry-After` is honoured up to 300 s; the code caps it at 60 s.
- **[INTEGRATOR]** `eval/configs/swde_test_crawl4ai.yaml` lists a bogus site `_dataset`
  (reported to the main agent).

## Next

1. **Re-run the comparisons at full size on the current default model.** Every baseline on all
   64 SWDE test sites with the planned 200-page samples, the Common Crawl set at the planned
   cap over a slower download window, AutoScraper's cost measured on whole sites, and the
   evaluations repeated with `gpt-6.1-terra`, which a fresh install now selects.
2. **Replace the build agent's judgments with human ones.** Human verdicts for schema precision,
   a human pass over the Common Crawl label-noise check and over the SWDE label-problem sample,
   so the claims that rest on them stand on their own.
3. **Close the product gaps the release found**: tail templates in prompted mode, the known
   issues above, end-to-end extraction speed (one worker pool per run instead of per site and
   stage), and template-drift detection for incremental re-crawls from the per-field statistics
   specs already store.
