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
