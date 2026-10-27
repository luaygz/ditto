# SWDE dev-slice misses and the `canon` match column

Regenerate with `uv run python eval/analysis/swde_dev_misses.py` (offline, saved predictions, no
LLM). Inputs: ditto run `swde_dev_ditto/20261027T102344Z` (every labeled page the LLM did not
see) and naive baseline run `swde_dev_naive/20261027T085024Z` (its seeded 50-page sample per
site), dev slice only (two sites per vertical). The script first checks that its rescored `norm`
and `raw` numbers equal both runs' committed `results.json` (they do). Counts and examples:
`swde_dev_misses.json`; the per-miss classification: `swde_dev_misses_classified.jsonl`; F1 per
vertical: `swde_dev_rescore.json`.

## Misses under `norm`

A miss is a (page, field) unit with a false positive or a false negative.

| method | misses | wrong value | missing | extra (label empty) |
|---|---|---|---|---|
| ditto | 11,656 | 10,084 | 160 | 1,412 |
| naive | 401 | 356 | 0 | 45 |

A stratified sample of 158 misses (up to 12 per method and vertical, round-robin over the site's
fields, seed 1; university has no misses) was classified by hand by the build agent, not a human.
The population estimate weights each sampled miss by its site field's misses / samples.

| method | sampled | formatting | real error | label problem | estimate: formatting / real / label |
|---|---|---|---|---|---|
| ditto | 84 | 6 | 44 | 34 | 1,975 / 1,271 / 8,410 |
| naive | 74 | 16 | 18 | 40 | 70 / 85 / 246 |

- **Formatting false negatives** (the value is right, its representation differs): a trailing
  `" |"` in SWDE restaurant labels (`907/243-0507` vs `907/243-0507 |`), diacritics (`Sacks Cafe`
  vs `Sacks Café`; the page shows both), and several labeled values joined in one string
  (`Comedy Western` vs `[Comedy, Western]`, `Seafood,American`).
- **Real errors**: missing values (ditto: job-careerbuilder, camera-amazon manufacturer, auto-aol
  price), wrong nodes (an engine summary as `fuel_economy`, a seller's condition note as
  `publisher`), extra content (`323/653-3300 | Reserve Online`; the naive baseline's MPAA reasons,
  which `fields.yaml` asks for but SWDE does not label), and camera model codes (`DMC-ZR3K`)
  where SWDE labels the full product name.
- **Label problems**: SWDE labels that are truncated (allmovie MPAA `P` for `PG13`, amctv
  directors without first names, `BRIDGE` for `BRIDGEwater Bistro`), carry extra content
  (`Ray Allen #20 SG`; book-amazon `publisher` and `publication_date` are both labeled with the
  whole line `Knopf; 1st edition (May 25, 2010)`), or are missing while the page states the value
  (auto-aol engines in the overview prose, abebooks publishers in the seller description).

Most of ditto's remaining misses are label problems in six site fields: amctv director (1,978),
book-amazon publisher (1,969) and publication date (1,968), auto-aol engine (1,117), allmovie MPAA
rating (885) and espn player name (433).

## Rules added (`canon`, metric version 2)

Only the formatting kinds above, plus one seen in earlier dev runs, became rules (exact
definitions in `docs/benchmarks.md`, Metrics):

1. diacritics dropped (U+0300–U+036F after canonical decomposition);
2. a trailing `|` stripped like trailing punctuation;
3. digit groups joined by hyphens or spaces compared without them when at least ten characters
   remain (ISBNs): earlier ditto dev runs `20261026T100631Z` and `20261026T102101Z` lost 1,975
   and 1,958 `book-amazon.isbn_13` units to `9780307269997` vs `978-0307269997`;
4. on scalar fields, two or more distinct label alternatives joined by `,` `;` `/` `|` or
   whitespace (also `New American, American`, 36 units in ditto run `20261026T100631Z`).

No rule for label problems or real errors: each would need containment or fuzzy matching. No
rule for dates, leading field labels or units: no dev miss showed one. In the classified sample
`canon` credits all 22 formatting misses and none of the 136 others. Over all misses it credits
1,974 for ditto (fodors address 1,839, fodors name 135; the one fodors name it misses is
`WLe Meritage` vs `W Le Meritage`, a lost space) and 70 for naive (fodors address 44, phone 13,
allmovie genre 12, fodors cuisine 1); no unit that matches under `norm` misses under `canon`.

## F1 before and after (dev slice)

| run | vertical | F1 norm | F1 canon | change | F1 raw |
|---|---|---|---|---|---|
| ditto | auto | 0.867 | 0.867 | +0.000 | 0.867 |
| ditto | book | 0.797 | 0.797 | +0.000 | 0.797 |
| ditto | camera | 0.804 | 0.804 | +0.000 | 0.804 |
| ditto | job | 0.998 | 0.998 | +0.000 | 0.998 |
| ditto | movie | 0.820 | 0.820 | +0.000 | 0.820 |
| ditto | nbaplayer | 0.875 | 0.875 | +0.000 | 0.750 |
| ditto | restaurant | 0.855 | 0.978 | +0.123 | 0.855 |
| ditto | university | 1.000 | 1.000 | +0.000 | 1.000 |
| ditto | overall | 0.877 | 0.892 | +0.015 | 0.861 |
| ditto, naive's 50-page samples | overall | 0.876 | 0.890 | +0.015 | 0.860 |
| naive | auto | 0.875 | 0.875 | +0.000 | 0.875 |
| naive | book | 0.799 | 0.799 | +0.000 | 0.799 |
| naive | camera | 0.823 | 0.823 | +0.000 | 0.823 |
| naive | job | 0.873 | 0.873 | +0.000 | 0.873 |
| naive | movie | 0.765 | 0.795 | +0.030 | 0.765 |
| naive | nbaplayer | 0.875 | 0.875 | +0.000 | 0.750 |
| naive | restaurant | 0.855 | 1.000 | +0.145 | 0.855 |
| naive | university | 1.000 | 1.000 | +0.000 | 1.000 |
| naive | overall | 0.858 | 0.880 | +0.022 | 0.842 |

On the same 50-page samples ditto leads by 0.018 F1 under `norm` and by 0.010 under `canon`. The
rules were frozen before any SWDE test, Common Crawl or LiveWeb-IE result was rescored, and no
miss from those sets was looked at.
