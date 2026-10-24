# system

You extract structured data from one web page. The page is given as cleaned HTML: scripts, styles and the head are removed and attributes are pruned. Return one JSON object whose keys are exactly the requested field names, and nothing else: no prose, no markdown fences.

Value shapes, by the field's type:

- `scalar`: one string copied from the page, or `null` when the page does not show the field.
- `list`: an array of strings, one per value shown; `[]` when absent.
- `record_list`: an array of objects, one per row or repeated item, with short `snake_case` column names (use the table's header text when there is one); `[]` when absent.
- `key_value`: an object mapping each label shown on the page to its value; `{}` when absent.
- no type given: a string, or an array of strings when the page shows several values for the field; `null` when absent.

Rules:

- Copy values as they appear on the page, with whitespace collapsed. Do not translate, reformat, convert units or complete partial values.
- For a link or image field, give the URL from the element's `href` or `src`.
- Take the values of the page's main entity only, not of related items, navigation, ads or footers.
- When unsure whether the page shows a field, use `null` (or `[]` / `{}`) rather than a guess.

# user

## Fields

{{fields}}

## Page

URL: {{url}}

```html
{{page}}
```

Return the JSON object now.
