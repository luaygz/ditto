# Synthetic fixture site

A 200-page synthetic site produced by `ditto synth` (it is generated, not scraped): four
templates of 25-70 pages, one of them a near-duplicate pair (structural Jaccard 0.7, different
URL patterns), plus three long-tail templates of 2-5 pages.

- `site/*.html`, `site/urls.tsv` — the input ditto ingests (`path<TAB>url`).
- `truth/<page>.json` — ground-truth sidecar per page (never read by the extractor).
- `truth/templates.json` — template definitions: fields, types, value types, presence rates.
- `truth/fields/<template>.yaml` — prompted-mode `fields.yaml` per template.
- `synth.json` — parameters, seed, counts, measurements, generation timing.

Regenerate (byte-identical apart from the `generation` timing block of `synth.json`;
`tests/synth/test_synth_fixture.py` checks this):

```sh
uv run ditto synth generate --params eval/synth/fixture.yaml --seed 1 --out tests/fixtures/site
# equivalent without the CLI:
uv run python -c "from ditto.synth import synth_main; synth_main('eval/synth/fixture.yaml', 1, 'tests/fixtures/site')"
```
