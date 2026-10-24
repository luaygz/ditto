"""AutoScraper baseline (Huang et al., EMNLP 2024; github.com/EZ-hwh/AutoScraper).

The upstream code is cloned, unmodified, into ``eval/baselines/autoscraper/vendor/`` (gitignored)
at :data:`PINNED_COMMIT`; :func:`ensure_vendor` clones it when missing. We run the paper's method,
``module.stepback_crawler.StepbackCrawler`` (``--pattern autocrawler``), the way upstream's
``crawler_generation.py`` / ``crawler_extraction.py`` drive it, with these adaptations:

* **Model calls**: the vendor's ``api(query) -> str`` hook (upstream: an Azure/OpenAI GPT-4
  client) is a function over ditto's LLM layer, role ``baseline`` (the same main model as ditto
  and the naive baseline), one user message per call, ``prompt_version`` ``autoscraper@<commit>``
  so responses are cached like every other ditto call. ``max_tokens`` is 4096 instead of
  upstream's 512 because the main model spends output tokens on reasoning.
* **Safety**: upstream picks the best rule with ``eval(res['number'])`` on model output. ditto
  never executes generated code, so the vendor module's ``eval`` name is shadowed by an integer
  parser (anything else raises, which upstream already handles by taking rule 0).
* **Cost bounds**: ``request_parse`` retries unparseable answers up to ``error_max_times``
  (upstream 15); we pass 3. A cap of :data:`MAX_CALLS_PER_FIELD` model calls per (template,
  field) stops runaway step-back loops; a field that hits it gets no rule.
* **Granularity**: ditto's clusters stand in for upstream's per-website loop. Per template,
  :data:`N_SEEDS` seeded random pages are the seed pages (upstream README: ``--seed_website
  3``); one rule (an XPath action sequence) per requested field; the rule is applied to every
  page of the template. Tail pages without a template get empty predictions.
* **Instruction** per field: ``"Here's a webpage with detail information. Please extract the
  <name>: <description>."`` plus upstream's SWDE meta sentence.
* **Values**: upstream returns the list of matched text values; ``scalar`` fields take the first
  non-empty one, ``list`` fields all of them. AutoScraper writes single-node XPaths only, so
  ``record_list`` / ``key_value`` fields are not attempted (empty, noted).
* Vendor ``print`` output goes to ``<out_dir>/autoscraper.log`` instead of stdout.
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import json
import random
import subprocess
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, TextIO

from ditto.evaluation.baseline_api import BaselineOutput, NotRun, PageInput

if TYPE_CHECKING:
    from ditto.llm.base import LLM
    from ditto.schema import FieldsSchema


def _load_common() -> ModuleType:
    name = "ditto_eval_baseline_common"
    module = sys.modules.get(name)
    if module is None:
        spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name("common.py"))
        if spec is None or spec.loader is None:  # pragma: no cover - file ships with this one
            raise ImportError("eval/baselines/common.py not found")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return module


common = _load_common()

NAME = "autoscraper"
REPO_URL = "https://github.com/EZ-hwh/AutoScraper"
PINNED_COMMIT = "3d78f8ab2c026d067be8b04b6b9247741fe207d1"
VENDOR_DIR = Path(__file__).with_name("autoscraper") / "vendor"
PROMPT_VERSION = f"autoscraper@{PINNED_COMMIT[:7]}"
N_SEEDS = 3
MAX_TOKEN = 8000  # upstream default chunk size (cl100k tokens) for domlm_parse
ERROR_MAX_TIMES = 3
MAX_OUTPUT_TOKENS = 4096
SERVICE_TIER: str | None = "flex"  # see common.baseline_llm
MAX_CALLS_PER_FIELD = 40
SUPPORTED_KINDS = ("scalar", "list")
SWDE_META = (
    "It's worth noticing that the candidate attribute values are the non-empty strings "
    "contained in text nodes in the corresponding DOM tree, and one page may contain multiple "
    "distinct values that correspond to an attribute."
)
_VENDOR_MODULES = ("utils", "module")


class CallBudgetExhausted(RuntimeError):
    """Raised from the api hook when a field exceeds :data:`MAX_CALLS_PER_FIELD`."""


# --------------------------------------------------------------------------- vendor code


def ensure_vendor(vendor_dir: Path = VENDOR_DIR, *, clone: bool = True) -> Path:
    """Make sure the upstream checkout exists (clone at the pinned commit if allowed)."""
    marker = vendor_dir / "module" / "stepback_crawler.py"
    if marker.is_file():
        return vendor_dir
    if not clone:
        raise NotRun(f"AutoScraper vendor checkout missing at {vendor_dir}")
    vendor_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["git", "clone", "--quiet", REPO_URL, str(vendor_dir)],
            check=True,
            capture_output=True,
            timeout=300,
        )
        subprocess.run(
            ["git", "-C", str(vendor_dir), "checkout", "--quiet", PINNED_COMMIT],
            check=True,
            capture_output=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise NotRun(f"could not clone {REPO_URL} at {PINNED_COMMIT}: {exc}") from exc
    if not marker.is_file():
        raise NotRun(f"{REPO_URL} checkout has no module/stepback_crawler.py")
    return vendor_dir


_VENDOR_CACHE: dict[str, ModuleType] = {}


def _safe_index(value: Any) -> int:
    """Stand-in for the vendor's ``eval(res['number'])``: digits only, never code."""
    text = str(value).strip()
    if not text.isdigit():
        raise ValueError(f"not a rule index: {text!r}")
    return int(text)


class _VendorLog:
    """Replacement for ``print`` inside vendor modules; writes to a log file when bound."""

    def __init__(self) -> None:
        self.stream: TextIO | None = None

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        if self.stream is not None:
            print(*args, file=self.stream, flush=False)


VENDOR_LOG = _VendorLog()


def load_vendor(vendor_dir: Path = VENDOR_DIR) -> ModuleType:
    """Import ``module.stepback_crawler`` from the checkout without leaking its top-level
    ``utils``/``module`` package names into ``sys.modules``."""
    key = str(vendor_dir.resolve())
    if key in _VENDOR_CACHE:
        return _VENDOR_CACHE[key]
    saved = {
        name: mod
        for name, mod in sys.modules.items()
        if name in _VENDOR_MODULES or name.startswith(tuple(f"{m}." for m in _VENDOR_MODULES))
    }
    for name in saved:
        del sys.modules[name]
    sys.path.insert(0, key)
    try:
        crawler_mod = importlib.import_module("module.stepback_crawler")
        html_utils = importlib.import_module("utils.html_utils")
    except ImportError as exc:
        raise NotRun(f"AutoScraper vendor code does not import: {exc}") from exc
    finally:
        sys.path.remove(key)
        for name in [
            n
            for n in sys.modules
            if n in _VENDOR_MODULES or n.startswith(tuple(f"{m}." for m in _VENDOR_MODULES))
        ]:
            del sys.modules[name]
        sys.modules.update(saved)
    # Module globals shadow builtins: no eval() of model output, prints go to the run log.
    crawler_mod.eval = _safe_index  # type: ignore[attr-defined]
    crawler_mod.print = VENDOR_LOG  # type: ignore[attr-defined]
    html_utils.print = VENDOR_LOG  # type: ignore[attr-defined]
    _VENDOR_CACHE[key] = crawler_mod
    return crawler_mod


# --------------------------------------------------------------------------- model hook


def instruction_for(name: str, description: str) -> str:
    target = f"{name}: {description}" if description else name
    return f"Here's a webpage with detail information. Please extract the {target}. {SWDE_META}"


class ApiHook:
    """The vendor's ``api(query) -> str`` over a ditto ``LLM``; counts calls per field."""

    def __init__(self, llm: LLM, *, max_calls: int = MAX_CALLS_PER_FIELD) -> None:
        self.llm = common.RecordingLLM(llm)
        self.max_calls = max_calls
        self.template_id: str | None = None
        self.field_calls = 0

    def start_field(self, template_id: str) -> None:
        self.template_id = template_id
        self.field_calls = 0

    def __call__(self, query: str) -> str:
        if self.field_calls >= self.max_calls:
            raise CallBudgetExhausted(f"more than {self.max_calls} calls for one field")
        self.field_calls += 1
        completion = self.llm.complete(
            [{"role": "user", "content": query}],
            max_tokens=MAX_OUTPUT_TOKENS,
            prompt_version=PROMPT_VERSION,
            template_id=self.template_id,
        )
        return str(completion.text)


# --------------------------------------------------------------------------- extraction


def rule_values(crawler: Any, simplified_html: str, rule: list[str]) -> list[str]:
    """Upstream ``extract_with_sequence`` on an already simplified page."""
    if not rule:
        return []
    values = crawler.extract_with_sequence(simplified_html, rule)
    return [v for v in values if isinstance(v, str) and v.strip()]


def to_prediction(values: dict[str, list[str]], schema: FieldsSchema) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for f in schema.fields:
        kind = common.field_kind(f)
        raw = values.get(f.name, [])
        if kind in SUPPORTED_KINDS:  # scalar: first non-empty match; list: all matches
            out[f.name] = common.coerce_value(raw, kind)
        else:
            out[f.name] = common.empty_value(kind)
    return out


def _extract_chunk(
    chunk: Sequence[tuple[str, str, str, str]],
) -> list[tuple[str, dict[str, list[str]]]]:
    """pmap worker: ``(key, html, rules_json, vendor_dir)`` -> raw values per field."""
    out: list[tuple[str, dict[str, list[str]]]] = []
    crawler = None
    for key, html, rules_json, vendor_dir in chunk:
        mod = load_vendor(Path(vendor_dir))
        if crawler is None:
            crawler = mod.StepbackCrawler(api=_no_api, simplify=False, verbose=False)
        simplified = mod.simplify_html(html)
        rules: dict[str, list[str]] = json.loads(rules_json)
        out.append(
            (key, {name: rule_values(crawler, simplified, rule) for name, rule in rules.items()})
        )
    return out


def _no_api(query: str) -> str:  # pragma: no cover - extraction never calls the model
    raise RuntimeError("extraction must not call the model")


@contextlib.contextmanager
def _vendor_log(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        VENDOR_LOG.stream = fh
        try:
            yield
        finally:
            VENDOR_LOG.stream = None


# --------------------------------------------------------------------------- entry point


def synthesize_rules(
    crawler: Any,
    hook: ApiHook,
    template_id: str,
    seeds: list[str],
    schema: FieldsSchema,
) -> tuple[dict[str, list[str]], dict[str, str]]:
    """One AutoScraper rule per supported field from the seed pages: ``(rules, errors)``."""
    rules: dict[str, list[str]] = {}
    errors: dict[str, str] = {}
    for f in schema.fields:
        if common.field_kind(f) not in SUPPORTED_KINDS:
            continue
        hook.start_field(template_id)
        try:
            rule = crawler.rule_synthesis(
                template_id, seeds, instruction_for(f.name, f.description), max_token=MAX_TOKEN
            )
        except CallBudgetExhausted as exc:
            rule, errors[f.name] = [], str(exc)
        rules[f.name] = [str(x) for x in rule] if isinstance(rule, list) else []
    return rules, errors


def run_baseline(
    pages: list[PageInput],
    schema: FieldsSchema,
    *,
    out_dir: Path,
    settings: Any = None,
    max_pages: int | None = None,
    seed: int = 0,
    site: str | None = None,
    llm: LLM | None = None,
    n_seeds: int = N_SEEDS,
    service_tier: str | None = SERVICE_TIER,
    workers: int | None = None,
    vendor_dir: Path = VENDOR_DIR,
    clone: bool = True,
    progress: Callable[[str], None] | None = None,
) -> BaselineOutput:
    """Synthesize AutoScraper rules per template from ``n_seeds`` pages, apply to all pages."""
    from ditto.parallel import pmap_flat

    vendor = load_vendor(ensure_vendor(vendor_dir, clone=clone))
    out_dir = Path(out_dir)
    selected = common.sample_pages(pages, max_pages, seed)
    groups, tail = common.group_by_template(selected)
    base_llm = llm if llm is not None else common.baseline_llm(settings, service_tier=service_tier)
    hook = ApiHook(base_llm)
    crawler = vendor.StepbackCrawler(api=hook, error_max_times=ERROR_MAX_TIMES)
    rng = random.Random(seed)

    template_rows: list[dict[str, Any]] = []
    work: list[tuple[str, str, str, str]] = []
    with _vendor_log(out_dir / f"{NAME}.log"):
        for tid in sorted(groups):
            group = groups[tid]
            seed_pages = rng.sample(group, min(n_seeds, len(group)))
            if progress is not None:
                progress(f"{tid}: {len(group)} pages, seeds {[p.key for p in seed_pages]}")
            started = time.perf_counter()
            rules, errors = synthesize_rules(
                crawler, hook, tid, [p.html for p in seed_pages], schema
            )
            template_rows.append(
                {
                    "template_id": tid,
                    "pages": len(group),
                    "seed_keys": [p.key for p in seed_pages],
                    "rules": rules,
                    "errors": errors,
                    "seconds": round(time.perf_counter() - started, 2),
                }
            )
            rules_json = json.dumps(rules)
            work.extend((p.key, p.html, rules_json, str(vendor_dir)) for p in group)

    tally = common.UsageTally()
    for completion in hook.llm.drain():
        common.tally_completion(tally, completion)

    predictions: dict[str, dict[str, Any]] = {}
    started = time.perf_counter()
    for key, values in pmap_flat(_extract_chunk, work, workers=workers, chunk_size=16):
        predictions[key] = to_prediction(values, schema)
    extract_s = time.perf_counter() - started
    for page in tail:
        predictions[page.key] = to_prediction({}, schema)

    unsupported = [f.name for f in schema.fields if common.field_kind(f) not in SUPPORTED_KINDS]
    notes = [
        f"AutoScraper StepbackCrawler @ {PINNED_COMMIT[:7]} with {base_llm.model}: {n_seeds} seed "
        f"pages per template ({len(groups)} templates), rules applied to {len(work)} pages",
        f"cost_usd prices all tokens from config/prices.yaml; actually spent this run: "
        f"${tally.spent_usd:.4f} ({tally.cached_calls}/{tally.calls} calls from cache)",
        f"extraction: {len(work)} pages in {extract_s:.2f}s",
        f"service tier: {service_tier or 'default' if llm is None else 'caller-supplied LLM'}; "
        "spend is recorded at the standard price (Flex bills ~50% of it)",
    ]
    if unsupported:
        notes.append(f"record_list/key_value fields not attempted by AutoScraper: {unsupported}")
    if tail:
        notes.append(f"{len(tail)} tail page(s) without a template got empty predictions")
    capped = [(r["template_id"], f) for r in template_rows for f in r["errors"]]
    if capped:
        notes.append(f"fields stopped at the {MAX_CALLS_PER_FIELD}-call cap: {capped}")
    output = BaselineOutput(
        predictions=predictions,
        calls=tally.calls,
        input_tokens=tally.input_tokens,
        output_tokens=tally.output_tokens,
        cost_usd=tally.nominal_usd,
        notes=notes,
        model=base_llm.model,
        pages_run=len(selected),
        spend_usd=tally.spent_usd,
    )
    common.write_jsonl(out_dir / f"{NAME}.templates.jsonl", template_rows)
    common.write_jsonl(
        out_dir / f"{NAME}.predictions.jsonl", common.predictions_rows(selected, predictions)
    )
    common.write_summary(
        out_dir,
        NAME,
        output,
        {
            "commit": PINNED_COMMIT,
            "seed": seed,
            "n_seeds": n_seeds,
            "usage": tally.as_dict(),
            "extract_s": extract_s,
        },
    )
    return output
