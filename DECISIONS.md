# Decisions

One line per decision, with the reason. Newest at the bottom of each section.

## Naming, license, environment

- 2026-10-24 — PyPI name check: `https://pypi.org/pypi/ditto/json` returned HTTP 200 (taken); `ditto-extract` returned 404. Distribution name is `ditto-extract`; module and CLI stay `ditto`.
- 2026-10-24 — License MIT; copyright line `Luay Gharzeddine` from `git config user.name`.
- 2026-10-24 — Python pinned to 3.11 via `.python-version` (uv-managed CPython 3.11.14); CI also runs 3.12.
- 2026-10-24 — Build backend hatchling; `prompts/` and `config/prices.yaml` are force-included into the wheel as `ditto/_data/` so an installed package finds them without the source tree.

## Models and budget

- 2026-10-24 — API key verified with one minimal chat completion (HTTP 200).
- 2026-10-24 — Main model `gpt-6.1-sol`: highest version among `gpt-` ids from `GET /v1/models` after dropping mini/nano/realtime/audio/image/embedding/tts/transcribe/search/preview; accepted a JSON-only prompt (`{"ok": true}`).
- 2026-10-24 — Small model `gpt-5.4-mini` (resolves to `gpt-5.4-mini-2026-03-17`): highest version id containing `mini` after the same exclusions (`gpt-realtime-2.1-mini` is excluded by `realtime`); accepted a JSON-only prompt.
- 2026-10-24 — Prices read from https://developers.openai.com/api/docs/pricing on 2026-10-24: gpt-6.1-sol $2.00 in / $10.00 out per 1M tokens, gpt-5.4-mini $0.75 / $4.50. Unknown model ids fall back to a deliberately high estimated price ($10 / $50).
- 2026-10-24 — Budget `LLM_BUDGET_USD` unset in `.env`, so the default $40 applies.

## Formats and contracts

- 2026-10-24 — Structural path step format `tag#id.cls1.cls2[itemprop=v][data-testid=v][role=v]`, separator ` > `, attribute paths end in ` @href`/` @src`; defined once in `ditto.fingerprint.dom`.
- 2026-10-24 — `id` and `data-testid` values containing any digit are dropped from paths: page-specific ids (`post-1234`) would otherwise split every page into its own signature.
- 2026-10-24 — State classes (`active`, `selected`, `odd`, `even`, `first`, `last`, `is-*`, `has-*`, …) are treated like utility classes: they vary by page or position and would break path stability.
- 2026-10-24 — A node's analysis text is its *own* text (text plus children's tails, whitespace-normalized); a selector's extracted value is the node's full string value. Analysis needs per-path units, extraction needs what users see.
- 2026-10-24 — Positional lint: a selector made only of bare-tag steps is rejected when rooted at `/`, when any step has `[n]`, or when it ends on a generic tag (`div`, `span`, `td`, …). `//header//h1` (the brief's own example) stays legal.
- 2026-10-24 — `record_list`/`key_value` keep the brief's single `container`/`pairs` selector and carry the "two or three candidates" rule through optional `container_alternates`/`pairs_alternates` (0–2); column selectors are single XPaths addressed by header text.
- 2026-10-24 — Relative column selectors may be positional only when `column_source` is `llm` (headerless tables have no header text to anchor on).
- 2026-10-24 — Scalar/list fields carry `fallbacks`: candidate indices verification found to agree with the winner; extraction tries them only when the selected candidate matches nothing.
- 2026-10-24 — Page ids are dense integers from 0 in ingest order; `pages.sqlite` also stores `encoding`.
- 2026-10-24 — Owner raised the LLM budget to $60 (`LLM_BUDGET_USD=60` added to `.env`) after the measured naive-baseline cost (≈11.3k tokens ≈ $0.024 per SWDE page) showed the brief's 200-page samples on every site would cost ≈$450.
- 2026-10-24 — OpenAI Flex tier (`service_tier: flex`, 50% off, verified working for gpt-6.1-sol) may be used for bulk baseline calls; README cost tables are computed from token counts at standard-tier prices for every method so comparisons stay fair, and actual spend is reported separately.

## Process

- 2026-10-24 — The build is split into parallel slice agents against fixed contracts (`ditto.run`, `ditto.fingerprint.dom`, spec models, candidate/cluster/mapping models, LLM interface); all 15 hit an Anthropic API rate limit after ~34 minutes and were resumed in two batches of ≤8 to stay under it.

## Module decisions (reported by the slice builds; condensed)

### Ingest
- Encoding order: BOM, HTTP charset, meta/XML declaration, strict UTF-8, charset_normalizer, cp1252 (HTML5 order; SWDE pages carry a UTF-8 BOM with a stale `iso-8859-1` meta).
- WHATWG superset names (latin-1 → cp1252, gb2312 → gbk, shift_jis → cp932, …); non-text codecs (utf-7, base64, rot13) ignored; a meta declaring UTF-16/32 means UTF-8.
- WARC filters (HTTP 200, HTML content type, sniffing when missing) run before "latest capture per URL", so a newer 404 never hides an older good capture; ties go to the later file/offset.
- Blob paths stored relative to the run root so run directories can move; re-ingest keeps valid blobs and prunes the rest.
- Mixed inputs (HTML dir + WARC, two dirs, manifest with WARC) raise `ValueError`.
- Blob layout stays one file per page (contract); on btrfs the first ingest of 20k files measured ~980 pages/s (hardlink-bound), 100k-page synthetic ingest measured 1,375 pages/s/core.

### LLM layer
- Model ranking drops `-pro`, `-chat-latest`, `-codex`, `-deep-research`, `-instruct` to last resort; highest numeric version wins; undated alias beats dated snapshots.
- Cache key = SHA-256 of `{v:1, prompt_version, provider/model, messages, max_tokens, json_schema}`; `base_url` excluded (use another `DITTO_CACHE_DIR` per endpoint).
- Structured output never delegated: the JSON schema is rendered into the system prompt, not sent as `response_format`.
- Retries: up to 6 attempts, exponential backoff with equal jitter (1 s doubling, capped at 60 s), `Retry-After` honoured as the minimum wait up to that 60 s cap (header values over 300 s are ignored), `insufficient_quota` not retried, SDK retries disabled.
- Budget guard reserves a worst-case projection (input chars ÷ 3 + `max_tokens` output) while a call runs, so concurrent calls cannot overspend; scopes: global ledger vs `LLM_BUDGET_USD`, optional per-run `DITTO_RUN_BUDGET_USD`.
- Cached calls record `cost_usd: 0` and `list_cost_usd` (uncached price) for cost tables.
- Flex-tier calls are billed at the tier multiplier from `prices.yaml` (0.5) in the ledger; `list_cost_usd` stays at standard prices.

### Spec format
- Lint uses a real XPath 1.0 tokenizer; rejects namespace prefixes and `$variables`; positional forms include `[position()=n]`, `[last()]`, axis steps, `(…)[n]`, and every branch of `|`.
- Row-relative column selectors fail only when positional or stepping through `td`/`th`/`*` without an anchor.
- Reserved field names: `url`, `page_id`, `template_id`, anything starting with `_`.
- The runner recognises the brief's header-text column pattern and resolves the column index once per row group (identical results; quadratic → linear per table).
- `cast(date)` yields `YYYY-MM` for month-year, `YYYY` for a bare year; a lone `,` is a thousands separator only before exactly three digits.

### Analysis
- Key-value tables produce one candidate per key (shared path, `column` = key); record tables one candidate per column (`<th>` text, else `colN`).
- Key-value test: ≥70% of key cells recur on ≥2 pages and ≥2% of pages, some values vary, some page has ≥2 rows; keys capped at 50 characters.
- Layout tables (nested table or 60+ elements in a cell) are ignored.
- Constant text on < 95% of pages is boilerplate/label, not optional (the brief's optional rule requires m < 0.98).
- Two-pass analysis: pass 1 (≤500 evenly spaced pages) decides tables and invariant texts, pass 2 gathers stats on every sampled page.
- Label hint = adjacent invariant text only (never nav/lists/tables/blocks > 6 items; never climbs into `body`).

### Union page
- Candidates located with the analysis' own `locate_candidates`, so matching agrees with the analysis exactly.
- Every field candidate missing from the base is spliced (not only optional ones); donors chosen greedily.
- `colspan`/`rowspan` kept beyond the brief's attribute list (numbered columns depend on them).
- Collapse has four aggressiveness levels; key-value lists never collapse; over budget: collapse, then drop candidate-free subtrees, then cut candidate text to 120 characters, then drop least-frequent candidates (logged).

### Spec generation
- Prompt files are split into chat messages by `# system` / `# user` headings with `{{placeholder}}` values; all prompt text lives in versioned files.
- Both mode instructions sit in the system message so the system prefix is identical across templates (provider prompt cache applies).
- The schema shown to the model is derived from the Pydantic `Spec` minus verification-owned keys.
- The pipeline overrides `optional` from the analysis (max p of source candidates < 0.95) and fills `type_check`/description from `fields.yaml`.
- After a second validation failure, individually valid fields are kept (draft `partial`); failed fields go to `generation.json` for the path-induced fallback.
- Budget: refuse before any call when the projection exceeds the remainder; mid-run exhaustion returns normally listing unprocessed templates.

### Checkpoints
- Probe score = strongest evidence (label similarity, description overlap, user examples, type) × factors that only lower it (type, example shape, scalar/list shape, presence); found ≥ 0.6 with a 0.15 lead, ambiguous ≥ 0.35.
- Labels, boilerplate, noise, and table key/header cells can never be fields.
- `confirm --yes` accepts `probe.yaml` as it stands (hand edits kept).
- Review applies the spec to sampled pages directly; rejected fields go to `ditto.verify.repair_rejected`.

### Evaluation
- A field key missing from a page's label dict is not scored on that page; present-but-empty means the page lacks it.
- A null prediction on an empty label is a true negative; a wrong value counts as both a false positive and a miss.
- Scalar labels given as lists are alternatives; list predictions are scored top-1 against scalar labels (standard SWDE practice).
- The normalized column also canonicalizes numbers, reads C1 controls as cp1252, strips leading separators (SWDE labels carry `": Canon"`).
- Pages in a spec's `pages_seen` are excluded from accuracy.
- `usd_per_1k_pages` uses standard-tier list prices for every method; `spend_usd` is what was billed.
- Naive-baseline samples: 50 pages/site SWDE dev, 20 SWDE test, 40 Common Crawl (budget); ditto is scored on the same seeded pages.

### Synthetic generator
- `templates` counts regular templates; 3 extra tail templates (1–19 pages) always exist.
- Faker value pools built once per seed; each page's RNG derives from (seed, page index), so output is identical for any worker count.
- Near-duplicate Jaccard measured on rendered pages (worst case over nav variants); templates re-rolled until pairwise Jaccard ≤ 0.5 except the designated pair.

### Baselines
- Naive model input: `clean_html` plus attribute pruning to the union-page set (+ microdata `content`/`datetime`, `alt`/`title`).
- crawl4ai forced to temperature 1 (gpt-6.1-sol rejects 0.01); its token usage captured by wrapping its completion helper and recorded via `record_external_usage`.
- AutoScraper's `eval()` on model output replaced with a digits-only parser (no generated code runs); 3 seed pages per template; ≤40 calls per (template, field).
- Hand-written wrappers for auto-yahoo, book-barnesandnoble, movie-imdb, nbaplayer-yahoo, university-usnews (test split), written from pages 0000–0009 before any generated spec existed.

### Common Crawl
- Crawls CC-MAIN-2026-43 and CC-MAIN-2026-39; 8 domains: argos.co.uk, ikea.com (product), careerbuilder.com, dice.com (job), food.ndtv.com, marmiton.org (recipe), skiddle.com, ticketweb.com (event); selection evidence in `domains.yaml` and `selection_log.json`.
- Download measured at ~7.5 captures/s with the 4-request limit, so 160k captures take ~6 h; cap kept at 20,000 per domain.

## Gates

- 2026-10-26 — v0.1 passed on the desktop (16 cores, 62 GB; no scaling needed): ARI 1.0, purity 1.0, 0 over/under-merges at 10k and 100k pages (seed 1). Results: `eval/results/synth_default/20261026T095112Z`, `eval/results/synth_scale/20261026T095119Z`.

## Process (continued)

- 2026-10-26 — Work moved from the laptop (12 cores, 30 GB) to the desktop (16 cores, 62 GB); the copied `.venv` pointed at a missing interpreter and was rebuilt from `uv.lock`. Speed numbers in results come from the desktop.
- 2026-10-27 — SWDE dev naive baseline complete in `eval/results/swde_dev_naive/20261027T085024Z` (16/16 sites). The two earlier partial runs (7/16 before the budget cap; 10/16 after an integrator edit to a baseline file broke 5 sites mid-run and an XML control character broke movie-amctv) are kept for the audit trail; reports read the latest run.
- 2026-10-27 — Common Crawl download: CloudFront's WAF blocked this IP ("Request blocked", HTTP 403) after ~2,250 records (~440 MB) in each of two runs at 4 concurrent requests. Per Common Crawl's guidance (403 = slow down; retry at most once per second) the downloader now makes one request at a time, ≥ 1 s apart, backs off 60 s → 30 min on 403/429/503, and retries errored captures on resume. At that pace 160k captures would take ~44 h, so the cap is 2,500 pages per domain (2,000 detail + 500 other); argos.co.uk keeps the 6,701 detail pages fetched before the block plus 500 others. Actual counts are recorded in `domains.yaml`.
- 2026-10-27 — XPath evaluation in ditto's runner treats NBSP as whitespace in `normalize-space()`, so label predicates such as `normalize-space()='Price:'` match `Price:&nbsp;`; documented in `docs/spec-format.md`. A plain-lxml consumer of a spec gets standard XPath 1.0 behavior instead.
- 2026-10-27 — Owner chose the full evaluation plan after the estimate (≈ $55 more on top of $26.14 spent; project total ≈ $81) and will top up the OpenAI account; `LLM_BUDGET_USD` raised to 85. Caps: SWDE test naive $16 (10 pages/site), Crawl4AI $5 (SWDE test) / $1.5 (CC), AutoScraper $5 on 8 SWDE test sites (one per vertical), CC naive $6 (20 pages/domain).
- 2026-10-27 — Owner topped up the OpenAI account by $35 (balance $62.54 at that point; ledger $26.14 spent).
- 2026-10-27 — A new model, `gpt-6.1-terra` ($1.00 / $5.00 per 1M tokens, cached $0.05), appeared after the build started; a fresh install's auto-selection now picks it. All published evaluations stay pinned to `gpt-6.1-sol` (main) and `gpt-5.4-mini` (small), the ids selected at build time, so tuning, baselines and results share one model. Price entry added; ledger lines from the clean-install check re-priced from the estimated default.
- 2026-10-27 — v0.4 clean-install check: wheel built with `uv build`, installed into a fresh Python 3.11 venv, README 30-second example run from a scratch directory with only `.env`: 3 JSONL files (the near-duplicate article template merged by the stage-7 rule), row for t_0c6f identical to the README.
- 2026-10-27 — Billing check: between the first top-up (balance $62.54, ledger $26.14) and a later balance of $24.84, the ledger recorded $30.73 while the account was charged $37.70, so ditto's ledger undercounts real billing by ~23% (unexplained: possibly calls billed but failed client-side, Crawl4AI calls its usage hook missed, or other usage on the same account). Remaining work was re-estimated with that factor (~$23 real); the owner topped up another $15 (balance $39.05). README spend figures are ledger figures and say so.
- 2026-10-27 — Crawl4AI cost ~$0.20 per SWDE test site (more than estimated) and stopped at 28/64 sites at its $5 cap; cap raised to $9 and the config rerun so every method is scored on the same 64 sites (the 28 finished sites come from cache).
- 2026-10-27 — Owner approved creating the private GitHub repository luaygz/ditto; main and tags v0.1.0–v0.4.0 pushed so CI runs on GitHub Actions.
- 2026-10-27 — Crawl4AI on SWDE test costs ~$0.3–1 per site (many templates per site, ~4 calls per template with its validation/refinement loop; its usage hook also under-reported spend by ~1.5× against the ledger). It was stopped after 36 of 64 test sites (4–5 per vertical, round-robin) to keep the budget for the Common Crawl evaluations; the config now lists those 36 sites so it reruns from cache at $0. The README compares it with ditto and naive on the same 36 sites.
