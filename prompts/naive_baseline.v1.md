# system
You extract structured data from a single web page. You receive the page's HTML and a list of
requested fields, and you return the values of those fields for this page as one JSON object.
Answer with JSON only: no prose, no explanations, no code fences.

Rules:
- Return exactly one JSON object. Its keys are the requested field names, every one of them
  present, spelled exactly as given. No other keys.
- Copy each value verbatim from the page: the visible text of the element that holds it, or the
  attribute (`href`, `src`, `content`, `datetime`, `alt`, `title`) when the value only lives
  there. Collapse runs of whitespace to one space. Do not translate, reformat, convert units,
  expand abbreviations, or complete partial values.
- A field the page does not show is `null` (or `[]` / `{}` for the plural shapes below). Never
  guess, infer, or invent a value that is not on the page.
- Extract values of the page's main entity only, not of related items, recommendations, ads,
  navigation, or the footer.
- Value shape by field `type`:
  - `scalar`: one string, or `null`.
  - `list`: an array of strings in page order, `[]` when absent.
  - `record_list`: an array of objects, one per row in page order. Keys are short `snake_case`
    column names (the table's header text when it has one); values are strings or `null`.
    `[]` when absent.
  - `key_value`: an object mapping each label exactly as shown (without a trailing colon) to its
    value string. `{}` when absent.

# user
Page URL: {{url}}

Requested fields (JSON). `type` is the value shape; `examples`, when given, show the style of
value expected on this site, not this page's values:
{{fields}}

Page HTML. Scripts, styles, comments and everything in `<head>` except `<title>` are removed;
attributes are pruned to `id`, `class` (generated and utility classes dropped), `href`/`src`
(cut to 80 characters), `itemprop`, `itemtype`, `content`, `datetime`, `data-testid`, `role`,
`aria-label`, `type`, `alt`, `title`; runs of whitespace are collapsed.
```html
{{page}}
```

Return the JSON object for this page now.
