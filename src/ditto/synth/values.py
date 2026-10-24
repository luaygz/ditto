"""Realistic field values: Faker-seeded pools plus fast per-page generators.

Faker is slow (hundreds of microseconds per name), so a :class:`Pools` object is built once per
seed from seeded Faker instances (names, companies, streets, cities, phones, domains, ISBNs, ...)
and pages draw from it with their own ``random.Random``. Pools depend only on the seed, so every
worker process rebuilds (or inherits through fork) exactly the same pools.
"""

from __future__ import annotations

import random
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache

from ditto.synth import vocab as V

MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
DATE_FORMATS = ("iso", "us_long", "eu_short", "us_num", "us_short")
TIME_FORMATS = ("12h", "24h")
# (symbol, position, thousands separator, decimal separator)
CURRENCIES: tuple[tuple[str, str, str, str], ...] = (
    ("$", "pre", ",", "."),
    ("$", "pre", ",", "."),
    ("$", "pre", ",", "."),
    ("£", "pre", ",", "."),
    ("€", "post", ".", ","),
    ("USD ", "pre", ",", "."),
)
_LOCALES = (("en_US", 1.0), ("de_DE", 0.15), ("fr_FR", 0.15), ("es_ES", 0.15))


@dataclass(frozen=True)
class ValueCtx:
    """Per-template formatting conventions."""

    currency: int = 0
    date_fmt: str = "iso"
    time_fmt: str = "12h"


@dataclass(frozen=True)
class Pools:
    first_names: tuple[str, ...]
    last_names: tuple[str, ...]
    companies: tuple[str, ...]
    streets: tuple[str, ...]
    cities: tuple[str, ...]
    states: tuple[str, ...]
    phones: tuple[str, ...]
    domains: tuple[str, ...]
    free_domains: tuple[str, ...]
    jobs: tuple[str, ...]
    colors: tuple[str, ...]
    words: tuple[str, ...]
    isbns: tuple[str, ...]
    countries: tuple[str, ...]


def _uniq(items: list[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(i.strip() for i in items if i and i.strip()))


@lru_cache(maxsize=4)
def build_pools(seed: int) -> Pools:
    """Value pools for ``seed`` (deterministic for a given Faker version)."""
    from faker import Faker

    fakers = {}
    for n, (loc, _) in enumerate(_LOCALES):
        fk = Faker(loc)
        fk.seed_instance(seed * 1009 + n)
        fakers[loc] = fk

    def draw(fn: str, base: int) -> list[str]:
        out: list[str] = []
        for loc, share in _LOCALES:
            fk = fakers[loc]
            out.extend(str(getattr(fk, fn)()) for _ in range(max(1, int(base * share))))
        return out

    en = fakers["en_US"]
    return Pools(
        first_names=_uniq(draw("first_name", 400)),
        last_names=_uniq(draw("last_name", 400)),
        companies=_uniq(draw("company", 260)),
        streets=_uniq([en.street_address() for _ in range(300)]),
        cities=_uniq(draw("city", 240)),
        states=_uniq([en.state_abbr() for _ in range(200)]),
        phones=_uniq([en.phone_number() for _ in range(300)]),
        domains=_uniq([en.domain_name() for _ in range(260)]),
        free_domains=_uniq([en.free_email_domain() for _ in range(60)]),
        jobs=_uniq([en.job() for _ in range(400)]),
        colors=_uniq([en.color_name() for _ in range(300)]) or V.COLORS_FALLBACK,
        words=_uniq([w for w in en.get_words_list() if w.isalpha() and len(w) > 2]),
        isbns=_uniq([en.isbn13() for _ in range(300)]),
        countries=_uniq([en.country() for _ in range(300)]),
    )


# --- helpers ----------------------------------------------------------------------------------


def ascii_fold(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def slugify(text: str) -> str:
    out: list[str] = []
    dash = False
    for ch in ascii_fold(text).lower():
        if ch.isalnum():
            out.append(ch)
            dash = False
        elif not dash and out:
            out.append("-")
            dash = True
    return "".join(out).strip("-")[:60] or "item"


def _letters(rng: random.Random, n: int) -> str:
    return "".join(rng.choice("ABCDEFGHJKLMNPRSTUVWXYZ") for _ in range(n))


def _digits(rng: random.Random, n: int) -> str:
    return str(rng.randrange(1, 10)) + "".join(rng.choice("0123456789") for _ in range(n - 1))


def _money(amount: float, ctx: ValueCtx, cents: bool) -> str:
    sym, pos, thou, dec = CURRENCIES[ctx.currency % len(CURRENCIES)]
    text = f"{amount:,.2f}" if cents else f"{round(amount):,}"
    if thou != ",":
        text = text.replace(",", "\x00").replace(".", dec).replace("\x00", thou)
    return f"{sym}{text}" if pos == "pre" else f"{text} {sym.strip()}"


def _date(rng: random.Random, ctx: ValueCtx, lo: int = 2015, hi: int = 2026) -> str:
    y, m, d = rng.randint(lo, hi), rng.randint(1, 12), rng.randint(1, 28)
    fmt = ctx.date_fmt
    if fmt == "us_long":
        return f"{MONTHS[m - 1]} {d}, {y}"
    if fmt == "eu_short":
        return f"{d} {MONTHS[m - 1][:3]} {y}"
    if fmt == "us_num":
        return f"{m:02d}/{d:02d}/{y}"
    if fmt == "us_short":
        return f"{MONTHS[m - 1][:3]} {d}, {y}"
    return f"{y}-{m:02d}-{d:02d}"


def _time(rng: random.Random, ctx: ValueCtx) -> str:
    h, mi = rng.randint(7, 22), rng.choice((0, 15, 30, 45))
    if ctx.time_fmt == "24h":
        return f"{h:02d}:{mi:02d}"
    return f"{(h - 1) % 12 + 1}:{mi:02d} {'AM' if h < 12 else 'PM'}"


def _sentence(rng: random.Random, p: Pools, lo: int = 6, hi: int = 14) -> str:
    words = [rng.choice(p.words) for _ in range(rng.randint(lo, hi))]
    return " ".join(words).capitalize() + "."


def _title_words(rng: random.Random, p: Pools, n: int) -> str:
    return " ".join(rng.choice(p.words).capitalize() for _ in range(n))


def _person(rng: random.Random, p: Pools) -> str:
    return f"{rng.choice(p.first_names)} {rng.choice(p.last_names)}"


def _email(rng: random.Random, p: Pools) -> str:
    first = slugify(rng.choice(p.first_names)).replace("-", "")
    last = slugify(rng.choice(p.last_names)).replace("-", "")
    domain = rng.choice(p.domains) if rng.random() < 0.7 else rng.choice(p.free_domains)
    style = rng.randrange(4)
    if style == 0:
        local = f"{first}.{last}"
    elif style == 1:
        local = f"{first[:1]}{last}"
    elif style == 2:
        local = f"{first}{rng.randint(1, 99)}"
    else:
        local = f"{last}.{first[:1]}"
    return f"{local}@{domain}"


def _url(rng: random.Random, p: Pools) -> str:
    domain = rng.choice(p.domains)
    if rng.random() < 0.5:
        return f"https://www.{domain}/"
    return f"https://{domain}/{rng.choice(p.words)}/{rng.choice(p.words)}"


def _city(rng: random.Random, p: Pools) -> str:
    return f"{rng.choice(p.cities)}, {rng.choice(p.states)}"


def _address(rng: random.Random, p: Pools) -> str:
    return (
        f"{rng.choice(p.streets)}, {rng.choice(p.cities)}, {rng.choice(p.states)} {_digits(rng, 5)}"
    )


def _count(rng: random.Random, lo: int, hi: int) -> str:
    return f"{rng.randint(lo, hi):,}"


def _choice(options: tuple[str, ...]) -> Gen:
    def gen(rng: random.Random, _p: Pools, _c: ValueCtx) -> str:
        return rng.choice(options)

    return gen


Gen = Callable[[random.Random, Pools, ValueCtx], str]


def _g(fn: Callable[[random.Random, Pools], str]) -> Gen:
    def gen(rng: random.Random, p: Pools, _c: ValueCtx) -> str:
        return fn(rng, p)

    return gen


def _gr(fn: Callable[[random.Random], str]) -> Gen:
    def gen(rng: random.Random, _p: Pools, _c: ValueCtx) -> str:
        return fn(rng)

    return gen


GENERATORS: dict[str, Gen] = {
    # people, organisations, places, contact
    "person_name": _g(_person),
    "reviewer": _g(lambda r, p: f"{r.choice(p.first_names)} {r.choice(p.last_names)[:1]}."),
    "company": _g(lambda r, p: r.choice(p.companies)),
    "email": _g(_email),
    "url": _g(_url),
    "phone": _g(lambda r, p: r.choice(p.phones)),
    "address": _g(_address),
    "city": _g(_city),
    "country": _g(lambda r, p: r.choice(p.countries)),
    "job_title": _g(lambda r, p: r.choice(p.jobs)),
    "color": _g(lambda r, p: r.choice(p.colors)),
    "isbn": _g(lambda r, p: r.choice(p.isbns)),
    "venue": _g(lambda r, p: f"{r.choice(p.last_names)} {r.choice(V.VENUE_NOUNS)}"),
    "property_title": _g(lambda r, p: f"{r.choice(p.streets)}, {r.choice(p.cities)}"),
    "character": _g(lambda r, p: f"{r.choice(V.CHARACTER_TITLES)} {r.choice(p.last_names)}"),
    # money, numbers, dates
    "price": lambda r, _p, c: _money(r.uniform(4, 900), c, cents=True),
    "price_small": lambda r, _p, c: _money(r.uniform(1, 80), c, cents=True),
    "price_large": lambda r, _p, c: _money(r.randrange(80, 2500) * 1000, c, cents=False),
    "salary": lambda r, _p, c: _money(r.randrange(30, 220) * 1000, c, cents=False),
    "revenue": lambda r, _p, c: _money(r.randrange(1, 900) * 100000, c, cents=False),
    "date": lambda r, _p, c: _date(r, c),
    "time": lambda r, _p, c: _time(r, c),
    "count": _gr(lambda r: _count(r, 10, 48000)),
    "count_small": _gr(lambda r: str(r.randint(1, 12))),
    "year": _gr(lambda r: str(r.randint(1950, 2026))),
    "rating": _gr(lambda r: f"{r.uniform(1, 5):.1f}"),
    "stars": _gr(lambda r: str(r.randint(1, 5))),
    "calories": _gr(lambda r: str(r.randint(80, 950))),
    "percent": _gr(lambda r: f"{r.randint(1, 99)}%"),
    "discount": _gr(lambda r: f"{r.choice((5, 10, 15, 20, 25, 30, 40, 50))}%"),
    # codes
    "sku": _gr(lambda r: f"{_letters(r, 3)}-{_digits(r, 5)}"),
    "reference": _gr(lambda r: f"{_letters(r, 2)}-{_digits(r, 6)}"),
    "vin": _gr(lambda r: "".join(r.choice("ABCDEFGHJKLMNPRSTUVWXYZ0123456789") for _ in range(17))),
    "ticker": _gr(lambda r: _letters(r, r.randint(3, 4))),
    "vat": _gr(lambda r: f"{r.choice(('DE', 'FR', 'GB', 'NL', 'ES'))}{_digits(r, 9)}"),
    "model_code": _gr(lambda r: f"{_letters(r, 2)}-{_digits(r, 3)}{_letters(r, 1)}"),
    # measures
    "weight": _gr(lambda r: f"{r.uniform(0.1, 40):.1f} {r.choice(('kg', 'lb'))}"),
    "dimensions": _gr(
        lambda r: f"{r.randint(5, 200)} x {r.randint(5, 120)} x {r.randint(1, 90)} cm"
    ),
    "power": _gr(lambda r: f"{r.choice((5, 10, 40, 60, 100, 600, 1200, 2000))} W"),
    "capacity": _gr(lambda r: f"{r.choice((0.5, 1, 1.5, 2, 5, 20, 45))} L"),
    "battery": _gr(lambda r: f"{r.randint(2, 40)} hours"),
    "warranty": _gr(lambda r: r.choice(("1 year", "2 years", "3 years", "6 months", "Lifetime"))),
    "area": _gr(lambda r: f"{r.randint(450, 5200):,} sq ft"),
    "area_m2": _gr(lambda r: f"{r.randint(12, 90)} m²"),
    "lot_size": _gr(lambda r: f"{r.uniform(0.05, 3):.2f} acres"),
    "mileage": _gr(lambda r: f"{r.randint(500, 180000):,} mi"),
    "runtime": _gr(lambda r: f"{r.randint(78, 195)} min"),
    "duration_min": _gr(
        lambda r: r.choice((f"{r.randint(5, 55)} min", f"1 hr {r.randint(5, 55)} min"))
    ),
    "duration_weeks": _gr(lambda r: f"{r.randint(2, 16)} weeks"),
    "duration_months": _gr(lambda r: f"{r.randint(3, 24)} months"),
    "duration_hours": _gr(lambda r: f"{r.randint(1, 9)} hours"),
    "effort": _gr(lambda r: f"{r.randint(2, 5)}-{r.randint(6, 12)} hours per week"),
    "read_time": _gr(lambda r: f"{r.randint(2, 25)} min read"),
    "years": _gr(lambda r: f"{r.randint(1, 35)} years"),
    "horsepower": _gr(lambda r: f"{r.randint(90, 520)} hp"),
    "mpg": _gr(lambda r: f"{r.randint(15, 45)} city / {r.randint(20, 55)} hwy"),
    "engine": _gr(
        lambda r: f"{r.uniform(1.0, 5.7):.1f}L {r.choice(('I4', 'V6', 'V8', 'Turbo I4'))}"
    ),
    "amount_g": _gr(lambda r: f"{r.randint(1, 120)} {r.choice(('g', 'mg'))}"),
    "file_size": _gr(lambda r: f"{r.uniform(0.1, 48):.1f} MB"),
    "hours": _gr(
        lambda r: (
            f"{r.choice(('Mon–Fri', 'Daily', 'Tue–Sun'))} {r.randint(7, 11)}:00 AM – "
            f"{r.randint(5, 11)}:00 PM"
        )
    ),
    "week": _gr(lambda r: f"Week {r.randint(1, 16)}"),
    "room": _gr(lambda r: f"Room {_letters(r, 1)}{r.randint(1, 12)}"),
    "month": _gr(lambda r: r.choice(MONTHS)),
    "guests": _choice(V.GUESTS),
    # composed names and texts
    "product_name": _gr(
        lambda r: f"{r.choice(V.ADJECTIVES)} {r.choice(V.MATERIALS)} {r.choice(V.PRODUCT_NOUNS)}"
    ),
    "headline": _g(
        lambda r, p: (
            f"{r.choice(p.cities)} {r.choice(V.HEADLINE_VERBS)} "
            f"{_title_words(r, p, r.randint(2, 5))}"
        )
    ),
    "event_name": _g(
        lambda r, p: (
            f"{r.choice(V.EVENT_THEMES)} {r.choice(p.words).capitalize()} "
            f"{r.choice(V.EVENT_TYPES)} {r.randint(2024, 2027)}"
        )
    ),
    "recipe_name": _gr(
        lambda r: (
            f"{r.choice(V.DISH_MODS)} {r.choice(V.INGREDIENTS).title()} {r.choice(V.DISH_BASES)}"
        )
    ),
    "hotel_name": _g(
        lambda r, p: r.choice(
            (
                f"{r.choice(V.HOTEL_MODS)} {r.choice(p.cities)} {r.choice(V.HOTEL_NOUNS)}",
                f"The {r.choice(p.last_names)} {r.choice(V.HOTEL_NOUNS)}",
            )
        )
    ),
    "restaurant_name": _g(
        lambda r, p: r.choice(
            (
                f"{r.choice(p.last_names)}'s Kitchen",
                f"Bistro {r.choice(p.words).capitalize()}",
                f"{r.choice(p.words).capitalize()} & {r.choice(p.words).capitalize()}",
                f"The {r.choice(p.words).capitalize()} {r.choice(('Table', 'Room', 'House'))}",
            )
        )
    ),
    "book_title": _g(
        lambda r, p: r.choice(
            (
                f"The {_title_words(r, p, 1)} of {_title_words(r, p, r.randint(1, 2))}",
                f"{_title_words(r, p, r.randint(2, 4))}",
                f"A {_title_words(r, p, 1)} in {r.choice(p.cities)}",
            )
        )
    ),
    "movie_title": _g(
        lambda r, p: r.choice(
            (
                f"{_title_words(r, p, r.randint(1, 3))}",
                f"The {_title_words(r, p, r.randint(1, 2))}",
                f"{_title_words(r, p, 1)}: {_title_words(r, p, 2)}",
            )
        )
    ),
    "course_title": _gr(
        lambda r: r.choice(
            (
                f"Introduction to {r.choice(V.SKILLS)}",
                f"{r.choice(V.SKILLS)} for Professionals",
                f"Advanced {r.choice(V.SKILLS)}",
                f"{r.choice(V.SKILLS)} Fundamentals",
            )
        )
    ),
    "car_title": _gr(
        lambda r: (
            f"{r.randint(2008, 2026)} {r.choice(V.CAR_MAKES)} {r.choice(V.CAR_MODELS)} "
            f"{r.choice(V.CAR_TRIMS)}"
        )
    ),
    "session_title": _g(lambda r, p: f"{_title_words(r, p, 2)} in {r.choice(V.SKILLS)}"),
    "dish": _gr(lambda r: f"{r.choice(V.DISH_MODS)} {r.choice(V.DISH_BASES)}"),
    "dish_desc": _g(lambda r, p: _sentence(r, p, 4, 9)),
    "ingredient": _gr(
        lambda r: (
            f"{r.choice(('1', '2', '3', '1/2', '1/4', '4', '200'))} {r.choice(V.UNITS)} "
            f"{r.choice(V.INGREDIENTS)}"
        )
    ),
    "step": _g(
        lambda r, p: (
            f"{r.choice(('Mix', 'Stir', 'Bake', 'Whisk', 'Chop', 'Simmer', 'Season'))} "
            f"the {r.choice(V.INGREDIENTS)} {' '.join(r.choice(p.words) for _ in range(r.randint(3, 8)))}."
        )
    ),
    "requirement": _gr(
        lambda r: (
            f"{r.choice(V.REQUIREMENT_OPENERS)} {r.choice(V.SKILLS)}"
            + r.choice(("", f" ({r.randint(1, 8)}+ years)"))
        )
    ),
    "feature": _gr(lambda r: f"{r.choice(V.FEATURE_VERBS)} {r.choice(V.FEATURE_OBJECTS)}"),
    "sentence": _g(lambda r, p: _sentence(r, p)),
    "paragraph": _g(lambda r, p: " ".join(_sentence(r, p) for _ in range(r.randint(2, 3)))),
    "tag": _g(lambda r, p: r.choice(p.words)),
    "skill": _choice(V.SKILLS),
    "document_name": _gr(lambda r: f"{r.choice(V.DOC_NOUNS)} {r.randint(2019, 2026)}"),
    "office_name": _choice(V.OFFICE_NAMES),
    "degree": _choice(V.DEGREES),
    # categorical (low cardinality on purpose: modal share must not hide them)
    "availability": _choice(V.AVAILABILITY),
    "product_category": _choice(V.CATEGORIES),
    "size": _choice(V.SIZES),
    "stock": _choice(V.STOCK),
    "material": _choice(V.MATERIALS),
    "department": _choice(V.DEPARTMENTS),
    "employment_type": _choice(V.EMPLOYMENT_TYPES),
    "experience_level": _choice(V.EXPERIENCE_LEVELS),
    "work_mode": _choice(V.WORK_MODES),
    "job_category": _choice(V.CATEGORY_WORDS_JOB),
    "schedule": _choice(V.SCHEDULES),
    "travel": _choice(V.TRAVEL),
    "timezone": _choice(V.TIMEZONES),
    "pronouns": _choice(V.PRONOUNS),
    "certificate": _choice(V.CERTIFICATES),
    "benefit": _choice(V.BENEFITS),
    "event_category": _choice(V.EVENT_TYPES),
    "age_limit": _choice(V.AGE_LIMITS),
    "dress_code": _choice(V.DRESS_CODES),
    "accessibility": _choice(V.ACCESSIBILITY),
    "refund_policy": _choice(V.REFUND),
    "event_format": _choice(V.EVENT_FORMATS),
    "ticket_type": _choice(V.TICKET_TYPES),
    "ticket_status": _choice(V.TICKET_STATUS),
    "track": _choice(V.TRACKS),
    "cuisine": _choice(V.CUISINES),
    "difficulty": _choice(V.DIFFICULTY),
    "meal_course": _choice(V.MEAL_COURSES),
    "nutrient": _choice(V.NUTRIENTS),
    "diet": _choice(V.DIETS),
    "equipment": _choice(V.EQUIPMENT),
    "storage": _gr(lambda r: f"Keeps {r.randint(2, 7)} days in the fridge"),
    "season": _choice(V.SEASONS),
    "occasion": _choice(V.OCCASIONS),
    "section": _choice(V.SECTIONS),
    "license": _choice(V.LICENSES),
    "edition": _choice(V.EDITIONS),
    "series_name": _g(lambda r, p: f"The {_title_words(r, p, 1)} Series"),
    "industry": _choice(V.INDUSTRIES),
    "company_status": _choice(V.COMPANY_STATUS),
    "legal_form": _choice(V.LEGAL_FORMS),
    "exchange": _choice(V.EXCHANGES),
    "service": _choice(V.SERVICES),
    "property_type": _choice(V.PROPERTY_TYPES),
    "listing_event": _choice(V.LISTING_EVENTS),
    "listing_source": _choice(V.LISTING_SOURCES),
    "heating": _choice(V.HEATING),
    "cooling": _choice(V.COOLING),
    "flooring": _choice(V.FLOORING),
    "roof": _choice(V.ROOFS),
    "view": _choice(V.VIEWS),
    "energy": _choice(V.ENERGY),
    "parking": _choice(V.PARKING),
    "amenity": _choice(V.AMENITIES),
    "room_type": _choice(V.ROOM_TYPES),
    "room_avail": _choice(V.ROOM_AVAIL),
    "bed": _choice(V.BEDS),
    "pets": _choice(V.PETS),
    "smoking": _choice(V.SMOKING),
    "children_policy": _choice(V.CHILDREN),
    "payment": _choice(V.PAYMENT),
    "breakfast": _choice(V.BREAKFAST),
    "wifi": _choice(V.WIFI),
    "book_format": _choice(V.BOOK_FORMATS),
    "genre": _choice(V.GENRES),
    "language": _choice(V.LANGUAGES),
    "reading_age": _choice(V.READING_AGES),
    "aspect": _choice(V.ASPECTS),
    "sound": _choice(V.SOUND),
    "film_color": _choice(V.FILM_COLORS),
    "camera": _choice(V.CAMERAS),
    "credit": _choice(V.CREDITS),
    "level": _choice(V.LEVELS),
    "credential": _choice(V.CREDENTIALS),
    "assessment": _choice(V.ASSESSMENTS),
    "prereq": _choice(V.PREREQS),
    "price_range": _choice(V.PRICE_RANGES),
    "noise_level": _choice(V.NOISE_LEVELS),
    "yes_no": _choice(("Yes", "No")),
    "make": _choice(V.CAR_MAKES),
    "fuel": _choice(V.FUELS),
    "transmission": _choice(V.TRANSMISSIONS),
    "condition": _choice(V.CONDITIONS),
    "body_type": _choice(V.BODY_TYPES),
    "car_feature": _choice(V.CAR_FEATURES),
    "service_type": _choice(V.SERVICE_TYPES),
    "drivetrain": _choice(V.DRIVETRAINS),
    "history_event": _choice(V.HISTORY_EVENTS),
    "file_type": _choice(V.FILE_TYPES),
    "delivery_time": _choice(V.DELIVERY),
    "returns": _choice(V.RETURNS),
    "carrier": _choice(V.CARRIERS),
    "badge": _choice(V.BADGES),
    "promo": _choice(V.PROMOS),
    "verified": _choice(V.VERIFIED),
}


def generate(name: str, rng: random.Random, pools: Pools, ctx: ValueCtx) -> str:
    """One value of generator ``name`` (single line, no surrounding whitespace)."""
    return GENERATORS[name](rng, pools, ctx)
