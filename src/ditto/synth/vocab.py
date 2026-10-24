"""Entity kinds, field catalogs and word lists for the synthetic generator (pure data).

Each kind describes one entity schema (product, person, job, ...): a heading field, scalar fields
with visible label synonyms, list fields, record tables, and key-value blocks. Generators are
named by string and implemented in :mod:`ditto.synth.values`.
"""

from __future__ import annotations

from dataclasses import dataclass

from ditto.specgen.typecheck import ValueType


@dataclass(frozen=True)
class FieldProto:
    name: str
    labels: tuple[str, ...]
    gen: str
    vtype: ValueType
    desc: str
    tag: str = "span"  # semantic value tag in server markup: span | strong | p | h1


@dataclass(frozen=True)
class ListProto:
    name: str
    headings: tuple[str, ...]
    gen: str
    vtype: ValueType
    desc: str
    ordered: bool = False
    link: bool = False


@dataclass(frozen=True)
class ColumnProto:
    name: str
    label: str
    gen: str
    vtype: ValueType


@dataclass(frozen=True)
class TableProto:
    name: str
    headings: tuple[str, ...]
    columns: tuple[ColumnProto, ...]
    extra: tuple[ColumnProto, ...]  # optional-column candidates
    desc: str


@dataclass(frozen=True)
class KVProto:
    name: str
    headings: tuple[str, ...]
    keys: tuple[tuple[str, str], ...]  # (visible key label, generator)
    desc: str


@dataclass(frozen=True)
class KindProto:
    kind: str
    noun: str
    segments: tuple[str, ...]  # URL first segments; later ones serve near-duplicate partners
    url_style: str
    prefixes: tuple[str, ...]  # semantic class prefixes
    title: FieldProto
    scalars: tuple[FieldProto, ...]
    lists: tuple[ListProto, ...]
    tables: tuple[TableProto, ...]
    kvs: tuple[KVProto, ...]
    related: str  # heading of the "related items" block


F = FieldProto
C = ColumnProto


KINDS: tuple[KindProto, ...] = (
    KindProto(
        kind="product",
        noun="product",
        segments=("products", "bundles", "deals", "shop"),
        url_style="slug-id",
        prefixes=("product", "pdp", "item", "goods"),
        title=F(
            "name",
            ("Product",),
            "product_name",
            "text",
            "the product name shown as the page heading",
            "h1",
        ),
        scalars=(
            F(
                "price",
                ("Price", "Our price", "Sale price"),
                "price",
                "price",
                "the current selling price, including the currency symbol",
                "strong",
            ),
            F(
                "sku",
                ("SKU", "Item number", "Product code"),
                "sku",
                "text",
                "the stock keeping unit code",
            ),
            F(
                "brand",
                ("Brand", "Manufacturer", "Made by"),
                "company",
                "text",
                "the brand or manufacturer name",
            ),
            F(
                "availability",
                ("Availability", "Stock status"),
                "availability",
                "text",
                "stock status such as In stock",
            ),
            F(
                "rating",
                ("Rating", "Average rating", "Customer rating"),
                "rating",
                "number",
                "average customer rating out of 5",
            ),
            F(
                "review_count",
                ("Reviews", "Number of reviews"),
                "count",
                "number",
                "number of customer reviews",
            ),
            F("color", ("Color", "Colour", "Finish"), "color", "text", "the product color"),
            F(
                "weight",
                ("Weight", "Shipping weight"),
                "weight",
                "text",
                "product weight with unit",
            ),
            F("seller", ("Sold by", "Seller", "Vendor"), "company", "text", "name of the seller"),
            F(
                "seller_email",
                ("Seller email", "Contact seller"),
                "email",
                "email",
                "seller contact email address",
            ),
            F("warranty", ("Warranty", "Guarantee"), "warranty", "text", "warranty period"),
            F(
                "release_date",
                ("Release date", "Available since", "Launched"),
                "date",
                "date",
                "date the product was released",
            ),
            F(
                "original_price",
                ("Was", "List price", "Regular price"),
                "price",
                "price",
                "price before discount, including the currency symbol",
            ),
            F(
                "category",
                ("Category", "Department"),
                "product_category",
                "text",
                "product category",
            ),
            F(
                "ships_from",
                ("Ships from", "Dispatched from"),
                "city",
                "text",
                "city the item ships from",
            ),
            F(
                "description",
                ("Description", "About this item", "Product details"),
                "paragraph",
                "text",
                "the product description paragraph",
                "p",
            ),
        ),
        lists=(
            ListProto(
                "features",
                ("Features", "Highlights", "Key features"),
                "feature",
                "text",
                "product feature bullet points",
            ),
            ListProto(
                "tags", ("Tags", "Tagged"), "tag", "text", "tags attached to the product", link=True
            ),
        ),
        tables=(
            TableProto(
                "variants",
                ("Variants", "Available options", "Choose an option"),
                (
                    C("size", "Size", "size", "text"),
                    C("color", "Color", "color", "text"),
                    C("price", "Price", "price", "price"),
                    C("stock", "Stock", "stock", "text"),
                ),
                (C("sku", "SKU", "sku", "text"), C("discount", "Discount", "discount", "number")),
                "table of purchasable variants with size, color, price and stock",
            ),
        ),
        kvs=(
            KVProto(
                "specifications",
                ("Specifications", "Tech specs", "Product information"),
                (
                    ("Material", "material"),
                    ("Dimensions", "dimensions"),
                    ("Weight", "weight"),
                    ("Model", "model_code"),
                    ("Country of origin", "country"),
                    ("Power", "power"),
                    ("Capacity", "capacity"),
                    ("Battery life", "battery"),
                    ("Warranty", "warranty"),
                    ("Finish", "color"),
                ),
                "specification pairs such as material, dimensions and weight",
            ),
        ),
        related="You may also like",
    ),
    KindProto(
        kind="person",
        noun="person",
        segments=("people", "authors", "team", "members"),
        url_style="slug",
        prefixes=("profile", "person", "member", "bio"),
        title=F(
            "name",
            ("Name",),
            "person_name",
            "text",
            "the person's full name shown as the page heading",
            "h1",
        ),
        scalars=(
            F(
                "job_title",
                ("Title", "Position", "Role"),
                "job_title",
                "text",
                "the person's job title",
            ),
            F(
                "organization",
                ("Company", "Organization", "Employer"),
                "company",
                "text",
                "organization the person works for",
            ),
            F(
                "email",
                ("Email", "E-mail", "Contact email"),
                "email",
                "email",
                "the person's email address",
            ),
            F(
                "phone",
                ("Phone", "Telephone", "Mobile"),
                "phone",
                "text",
                "the person's phone number",
            ),
            F(
                "location",
                ("Location", "Based in", "City"),
                "city",
                "text",
                "city where the person is based",
            ),
            F("website", ("Website", "Homepage", "Web"), "url", "url", "personal website URL"),
            F("member_since", ("Member since", "Joined"), "date", "date", "date the person joined"),
            F("department", ("Department", "Team"), "department", "text", "department or team"),
            F("office", ("Office", "Address"), "address", "text", "office postal address"),
            F(
                "experience",
                ("Experience", "Years of experience"),
                "years",
                "text",
                "years of professional experience",
            ),
            F(
                "languages",
                ("Languages", "Speaks"),
                "language",
                "text",
                "language the person speaks",
            ),
            F(
                "bio",
                ("About", "Biography", "Bio"),
                "paragraph",
                "text",
                "biography paragraph",
                "p",
            ),
        ),
        lists=(
            ListProto(
                "skills",
                ("Skills", "Expertise", "Areas of focus"),
                "skill",
                "text",
                "the person's skills",
                link=True,
            ),
        ),
        tables=(
            TableProto(
                "work_history",
                ("Work history", "Career", "Experience"),
                (
                    C("role", "Role", "job_title", "text"),
                    C("company", "Company", "company", "text"),
                    C("start", "From", "year", "number"),
                    C("end", "To", "year", "number"),
                ),
                (
                    C("location", "Location", "city", "text"),
                    C("team", "Team", "department", "text"),
                ),
                "table of previous roles with company and years",
            ),
        ),
        kvs=(
            KVProto(
                "details",
                ("Details", "At a glance", "Profile facts"),
                (
                    ("Nationality", "country"),
                    ("Education", "degree"),
                    ("Languages", "language"),
                    ("Time zone", "timezone"),
                    ("Availability", "work_mode"),
                    ("Hourly rate", "price_small"),
                    ("Certification", "certificate"),
                    ("Pronouns", "pronouns"),
                ),
                "profile detail pairs such as education and time zone",
            ),
        ),
        related="Other people",
    ),
    KindProto(
        kind="job",
        noun="job posting",
        segments=("jobs", "internships", "careers", "vacancies"),
        url_style="id-slug",
        prefixes=("job", "posting", "vacancy", "opening"),
        title=F(
            "title", ("Job",), "job_title", "text", "the job title shown as the page heading", "h1"
        ),
        scalars=(
            F(
                "company",
                ("Company", "Employer", "Hiring company"),
                "company",
                "text",
                "the hiring company",
            ),
            F(
                "location",
                ("Location", "Job location", "Where"),
                "city",
                "text",
                "job location city",
            ),
            F(
                "salary",
                ("Salary", "Pay", "Compensation"),
                "salary",
                "price",
                "annual salary including the currency symbol",
                "strong",
            ),
            F(
                "employment_type",
                ("Employment type", "Job type", "Contract"),
                "employment_type",
                "text",
                "employment type such as Full-time",
            ),
            F(
                "date_posted",
                ("Posted", "Date posted", "Published"),
                "date",
                "date",
                "date the job was posted",
            ),
            F(
                "deadline",
                ("Apply by", "Closing date", "Deadline"),
                "date",
                "date",
                "application deadline",
            ),
            F(
                "contact_email",
                ("Contact", "Recruiter email", "Questions"),
                "email",
                "email",
                "recruiter contact email",
            ),
            F("department", ("Department", "Team"), "department", "text", "department of the role"),
            F(
                "experience_level",
                ("Experience level", "Seniority", "Level"),
                "experience_level",
                "text",
                "seniority level",
            ),
            F(
                "work_mode",
                ("Workplace", "Work mode", "Remote"),
                "work_mode",
                "text",
                "on-site, hybrid or remote",
            ),
            F(
                "apply_url",
                ("Apply", "Application link", "Apply online"),
                "url",
                "url",
                "URL of the application form",
            ),
            F(
                "reference",
                ("Reference", "Job ID", "Requisition"),
                "reference",
                "text",
                "job reference code",
            ),
            F(
                "openings",
                ("Openings", "Positions"),
                "count_small",
                "number",
                "number of open positions",
            ),
            F(
                "description",
                ("Job description", "About the role"),
                "paragraph",
                "text",
                "job description paragraph",
                "p",
            ),
        ),
        lists=(
            ListProto(
                "requirements",
                ("Requirements", "Qualifications", "What you bring"),
                "requirement",
                "text",
                "list of job requirements",
            ),
            ListProto(
                "benefits",
                ("Benefits", "Perks", "What we offer"),
                "benefit",
                "text",
                "list of benefits",
            ),
        ),
        tables=(
            TableProto(
                "open_positions",
                ("Open positions", "Locations", "Where we are hiring"),
                (
                    C("location", "Location", "city", "text"),
                    C("team", "Team", "department", "text"),
                    C("positions", "Positions", "count_small", "number"),
                ),
                (
                    C("start", "Start", "date", "date"),
                    C("manager", "Hiring manager", "person_name", "text"),
                ),
                "table of open positions per location",
            ),
        ),
        kvs=(
            KVProto(
                "job_details",
                ("Job details", "Overview", "The role at a glance"),
                (
                    ("Category", "job_category"),
                    ("Schedule", "schedule"),
                    ("Travel", "travel"),
                    ("Visa sponsorship", "yes_no"),
                    ("Start date", "date"),
                    ("Contract length", "duration_months"),
                    ("Reports to", "job_title"),
                    ("Team size", "count_small"),
                ),
                "job detail pairs such as schedule and travel",
            ),
        ),
        related="Similar jobs",
    ),
    KindProto(
        kind="event",
        noun="event",
        segments=("events", "webinars", "meetups", "shows"),
        url_style="id",
        prefixes=("event", "happening", "gig", "session"),
        title=F(
            "name",
            ("Event",),
            "event_name",
            "text",
            "the event name shown as the page heading",
            "h1",
        ),
        scalars=(
            F("date", ("Date", "When", "Event date"), "date", "date", "date of the event"),
            F("start_time", ("Starts", "Start time", "Doors open"), "time", "text", "start time"),
            F("venue", ("Venue", "Where", "Location"), "venue", "text", "venue name"),
            F("address", ("Address", "Venue address"), "address", "text", "venue street address"),
            F(
                "price",
                ("Tickets from", "Price", "Admission"),
                "price",
                "price",
                "lowest ticket price including the currency symbol",
                "strong",
            ),
            F(
                "organizer",
                ("Organizer", "Hosted by", "Presented by"),
                "company",
                "text",
                "event organizer",
            ),
            F(
                "organizer_email",
                ("Organizer email", "Contact"),
                "email",
                "email",
                "organizer contact email",
            ),
            F("capacity", ("Capacity", "Seats"), "count", "number", "venue capacity"),
            F("category", ("Category", "Event type"), "event_category", "text", "event category"),
            F("website", ("Website", "Event page", "More info"), "url", "url", "event website URL"),
            F("end_time", ("Ends", "End time"), "time", "text", "end time"),
            F("age_limit", ("Ages", "Age restriction"), "age_limit", "text", "age restriction"),
            F(
                "description",
                ("About this event", "Details"),
                "paragraph",
                "text",
                "event description paragraph",
                "p",
            ),
        ),
        lists=(
            ListProto(
                "lineup",
                ("Lineup", "Speakers", "Featuring"),
                "person_name",
                "text",
                "performers or speakers",
            ),
        ),
        tables=(
            TableProto(
                "schedule",
                ("Schedule", "Programme", "Agenda"),
                (
                    C("time", "Time", "time", "text"),
                    C("session", "Session", "session_title", "text"),
                    C("speaker", "Speaker", "person_name", "text"),
                ),
                (C("room", "Room", "room", "text"), C("track", "Track", "track", "text")),
                "event schedule table with time, session and speaker",
            ),
            TableProto(
                "tickets",
                ("Tickets", "Ticket options"),
                (
                    C("ticket", "Ticket", "ticket_type", "text"),
                    C("price", "Price", "price", "price"),
                    C("status", "Status", "ticket_status", "text"),
                ),
                (C("fee", "Fee", "price_small", "price"),),
                "ticket types with price and status",
            ),
        ),
        kvs=(
            KVProto(
                "good_to_know",
                ("Good to know", "Event info"),
                (
                    ("Dress code", "dress_code"),
                    ("Parking", "yes_no"),
                    ("Accessibility", "accessibility"),
                    ("Language", "language"),
                    ("Refunds", "refund_policy"),
                    ("Duration", "duration_hours"),
                    ("Format", "event_format"),
                ),
                "practical event information pairs",
            ),
        ),
        related="More events",
    ),
    KindProto(
        kind="recipe",
        noun="recipe",
        segments=("recipes", "cocktails", "bakes", "dishes"),
        url_style="slug",
        prefixes=("recipe", "dish", "cook", "kitchen"),
        title=F(
            "name",
            ("Recipe",),
            "recipe_name",
            "text",
            "the recipe name shown as the page heading",
            "h1",
        ),
        scalars=(
            F("author", ("By", "Recipe by", "Author"), "person_name", "text", "recipe author"),
            F("prep_time", ("Prep time", "Prep"), "duration_min", "text", "preparation time"),
            F("cook_time", ("Cook time", "Cook"), "duration_min", "text", "cooking time"),
            F("total_time", ("Total time", "Ready in"), "duration_min", "text", "total time"),
            F(
                "servings",
                ("Servings", "Serves", "Yield"),
                "count_small",
                "number",
                "number of servings",
            ),
            F("calories", ("Calories", "Energy"), "calories", "number", "calories per serving"),
            F("cuisine", ("Cuisine",), "cuisine", "text", "cuisine"),
            F("difficulty", ("Difficulty", "Skill level"), "difficulty", "text", "difficulty"),
            F("published", ("Published", "Posted on"), "date", "date", "publication date"),
            F(
                "rating",
                ("Rating", "Average rating"),
                "rating",
                "number",
                "average rating out of 5",
            ),
            F("course", ("Course", "Meal"), "meal_course", "text", "meal course"),
            F(
                "description",
                ("About this recipe", "Introduction"),
                "paragraph",
                "text",
                "recipe introduction paragraph",
                "p",
            ),
        ),
        lists=(
            ListProto(
                "ingredients",
                ("Ingredients",),
                "ingredient",
                "text",
                "list of ingredients with quantities",
            ),
            ListProto(
                "steps",
                ("Method", "Directions", "Instructions"),
                "step",
                "text",
                "ordered preparation steps",
                ordered=True,
            ),
        ),
        tables=(
            TableProto(
                "nutrition",
                ("Nutrition facts", "Nutrition per serving"),
                (
                    C("nutrient", "Nutrient", "nutrient", "text"),
                    C("amount", "Amount", "amount_g", "text"),
                    C("daily_value", "Daily value", "percent", "number"),
                ),
                (C("per_100g", "Per 100 g", "amount_g", "text"),),
                "nutrition table with nutrient, amount and daily value",
            ),
        ),
        kvs=(
            KVProto(
                "recipe_info",
                ("Recipe info", "At a glance"),
                (
                    ("Diet", "diet"),
                    ("Equipment", "equipment"),
                    ("Storage", "storage"),
                    ("Freezer friendly", "yes_no"),
                    ("Season", "season"),
                    ("Occasion", "occasion"),
                    ("Origin", "country"),
                ),
                "recipe information pairs such as diet and equipment",
            ),
        ),
        related="More recipes",
    ),
    KindProto(
        kind="article",
        noun="article",
        segments=("blog", "news", "stories", "insights"),
        url_style="date-slug",
        prefixes=("article", "story", "post", "entry"),
        title=F(
            "headline",
            ("Headline",),
            "headline",
            "text",
            "the article headline shown as the page heading",
            "h1",
        ),
        scalars=(
            F("author", ("By", "Author", "Written by"), "person_name", "text", "article author"),
            F("published", ("Published", "Posted", "Date"), "date", "date", "publication date"),
            F("updated", ("Updated", "Last updated"), "date", "date", "last update date"),
            F("section", ("Section", "Category", "Filed under"), "section", "text", "site section"),
            F(
                "read_time",
                ("Reading time", "Read time"),
                "read_time",
                "text",
                "estimated reading time",
            ),
            F("word_count", ("Words", "Word count"), "count", "number", "number of words"),
            F(
                "author_email",
                ("Contact the author", "Email"),
                "email",
                "email",
                "author email address",
            ),
            F(
                "summary",
                ("Summary", "In brief", "Standfirst"),
                "sentence",
                "text",
                "one-sentence summary",
                "p",
            ),
            F(
                "photo_credit",
                ("Photo", "Image credit"),
                "person_name",
                "text",
                "photographer credit",
            ),
            F(
                "dateline",
                ("Dateline", "Reporting from"),
                "city",
                "text",
                "city the story is reported from",
            ),
            F("comment_count", ("Comments",), "count_small", "number", "number of comments"),
        ),
        lists=(
            ListProto("tags", ("Tags", "Topics"), "tag", "text", "article tags", link=True),
            ListProto(
                "key_points",
                ("Key points", "Takeaways"),
                "sentence",
                "text",
                "key points of the article",
            ),
        ),
        tables=(
            TableProto(
                "comments",
                ("Comments", "Reader comments"),
                (
                    C("author", "Author", "person_name", "text"),
                    C("date", "Date", "date", "date"),
                    C("comment", "Comment", "sentence", "text"),
                ),
                (
                    C("likes", "Likes", "count_small", "number"),
                    C("location", "Location", "city", "text"),
                ),
                "reader comments with author, date and text",
            ),
        ),
        kvs=(
            KVProto(
                "article_info",
                ("Article info", "About this story"),
                (
                    ("Section", "section"),
                    ("Language", "language"),
                    ("License", "license"),
                    ("Edition", "edition"),
                    ("Series", "series_name"),
                    ("Word count", "count"),
                ),
                "article metadata pairs",
            ),
        ),
        related="Related stories",
    ),
    KindProto(
        kind="company",
        noun="company",
        segments=("companies", "brands", "suppliers", "partners"),
        url_style="slug",
        prefixes=("company", "org", "firm", "business"),
        title=F(
            "name",
            ("Company",),
            "company",
            "text",
            "the company name shown as the page heading",
            "h1",
        ),
        scalars=(
            F("industry", ("Industry", "Sector"), "industry", "text", "industry sector"),
            F("founded", ("Founded", "Established", "Since"), "year", "number", "year founded"),
            F(
                "employees",
                ("Employees", "Company size", "Staff"),
                "count",
                "number",
                "number of employees",
            ),
            F(
                "headquarters",
                ("Headquarters", "HQ", "Head office"),
                "address",
                "text",
                "headquarters address",
            ),
            F("website", ("Website", "Web"), "url", "url", "company website URL"),
            F("email", ("Email", "Contact email"), "email", "email", "company contact email"),
            F("phone", ("Phone", "Telephone"), "phone", "text", "company phone number"),
            F(
                "ceo",
                ("CEO", "Chief executive", "Led by"),
                "person_name",
                "text",
                "chief executive",
            ),
            F(
                "revenue",
                ("Revenue", "Annual revenue"),
                "revenue",
                "price",
                "annual revenue including the currency symbol",
            ),
            F("ticker", ("Ticker", "Stock symbol"), "ticker", "text", "stock ticker symbol"),
            F(
                "status",
                ("Status", "Company status"),
                "company_status",
                "text",
                "registration status",
            ),
            F(
                "registration",
                ("Registration no.", "Company number"),
                "reference",
                "text",
                "registration number",
            ),
            F(
                "description",
                ("About", "Company profile"),
                "paragraph",
                "text",
                "company description paragraph",
                "p",
            ),
        ),
        lists=(
            ListProto(
                "services",
                ("Services", "What we do", "Products"),
                "service",
                "text",
                "services offered",
            ),
        ),
        tables=(
            TableProto(
                "offices",
                ("Offices", "Locations"),
                (
                    C("office", "Office", "office_name", "text"),
                    C("city", "City", "city", "text"),
                    C("phone", "Phone", "phone", "text"),
                ),
                (
                    C("employees", "Employees", "count_small", "number"),
                    C("opened", "Opened", "year", "number"),
                ),
                "office locations with city and phone",
            ),
        ),
        kvs=(
            KVProto(
                "company_facts",
                ("Company facts", "Key facts"),
                (
                    ("Legal form", "legal_form"),
                    ("Fiscal year end", "month"),
                    ("Auditor", "company"),
                    ("Parent company", "company"),
                    ("Stock exchange", "exchange"),
                    ("VAT ID", "vat"),
                ),
                "company fact pairs such as legal form and auditor",
            ),
        ),
        related="Similar companies",
    ),
    KindProto(
        kind="property",
        noun="property listing",
        segments=("homes", "rentals", "listings", "properties"),
        url_style="id-slug",
        prefixes=("listing", "property", "home", "estate"),
        title=F(
            "address",
            ("Address",),
            "property_title",
            "text",
            "the property address shown as the page heading",
            "h1",
        ),
        scalars=(
            F(
                "price",
                ("Price", "Asking price", "Offers over"),
                "price_large",
                "price",
                "asking price including the currency symbol",
                "strong",
            ),
            F("bedrooms", ("Bedrooms", "Beds"), "count_small", "number", "number of bedrooms"),
            F("bathrooms", ("Bathrooms", "Baths"), "count_small", "number", "number of bathrooms"),
            F(
                "area",
                ("Floor area", "Living area", "Size"),
                "area",
                "text",
                "floor area with unit",
            ),
            F(
                "year_built",
                ("Year built", "Built"),
                "year",
                "number",
                "year the property was built",
            ),
            F("property_type", ("Type", "Property type"), "property_type", "text", "property type"),
            F(
                "listed",
                ("Listed", "Date listed", "On market since"),
                "date",
                "date",
                "date listed",
            ),
            F("agent", ("Agent", "Listed by"), "person_name", "text", "listing agent"),
            F(
                "agent_email",
                ("Agent email", "Email agent"),
                "email",
                "email",
                "agent email address",
            ),
            F("agent_phone", ("Agent phone", "Call"), "phone", "text", "agent phone number"),
            F(
                "hoa_fee",
                ("HOA fee", "Service charge"),
                "price_small",
                "price",
                "monthly HOA fee including the currency symbol",
            ),
            F("lot_size", ("Lot size", "Plot"), "lot_size", "text", "lot size"),
            F("parking", ("Parking", "Garage"), "parking", "text", "parking"),
            F("mls", ("MLS #", "Listing ID"), "reference", "text", "listing id"),
            F(
                "description",
                ("Description", "About this home"),
                "paragraph",
                "text",
                "property description paragraph",
                "p",
            ),
        ),
        lists=(
            ListProto(
                "amenities", ("Amenities", "Features"), "amenity", "text", "list of amenities"
            ),
        ),
        tables=(
            TableProto(
                "price_history",
                ("Price history", "Sale history"),
                (
                    C("date", "Date", "date", "date"),
                    C("event", "Event", "listing_event", "text"),
                    C("price", "Price", "price_large", "price"),
                ),
                (
                    C("source", "Source", "listing_source", "text"),
                    C("change", "Change", "percent", "number"),
                ),
                "price history with date, event and price",
            ),
        ),
        kvs=(
            KVProto(
                "home_features",
                ("Property features", "Interior and exterior", "Facts and features"),
                (
                    ("Heating", "heating"),
                    ("Cooling", "cooling"),
                    ("Flooring", "flooring"),
                    ("Roof", "roof"),
                    ("Basement", "yes_no"),
                    ("View", "view"),
                    ("Pool", "yes_no"),
                    ("Energy rating", "energy"),
                ),
                "property feature pairs such as heating and flooring",
            ),
        ),
        related="Nearby homes",
    ),
    KindProto(
        kind="hotel",
        noun="hotel",
        segments=("hotels", "stays", "resorts", "inns"),
        url_style="slug-id",
        prefixes=("hotel", "stay", "lodging", "resort"),
        title=F(
            "name",
            ("Hotel",),
            "hotel_name",
            "text",
            "the hotel name shown as the page heading",
            "h1",
        ),
        scalars=(
            F("address", ("Address", "Location"), "address", "text", "hotel street address"),
            F("stars", ("Stars", "Star rating", "Class"), "stars", "number", "star rating"),
            F(
                "price_per_night",
                ("From", "Price per night", "Nightly rate"),
                "price",
                "price",
                "lowest nightly price including the currency symbol",
                "strong",
            ),
            F("phone", ("Phone", "Reception"), "phone", "text", "hotel phone number"),
            F("email", ("Email", "Reservations email"), "email", "email", "hotel email address"),
            F("website", ("Website", "Official site"), "url", "url", "hotel website URL"),
            F("check_in", ("Check-in", "Check in from"), "time", "text", "check-in time"),
            F("check_out", ("Check-out", "Check out by"), "time", "text", "check-out time"),
            F("rating", ("Guest rating", "Score"), "rating", "number", "guest rating"),
            F("room_count", ("Rooms", "Number of rooms"), "count", "number", "number of rooms"),
            F("chain", ("Brand", "Hotel chain"), "company", "text", "hotel chain"),
            F(
                "description",
                ("About the hotel", "Overview"),
                "paragraph",
                "text",
                "hotel description paragraph",
                "p",
            ),
        ),
        lists=(
            ListProto(
                "amenities", ("Amenities", "Facilities"), "amenity", "text", "hotel amenities"
            ),
        ),
        tables=(
            TableProto(
                "rooms",
                ("Rooms", "Room types", "Availability"),
                (
                    C("room_type", "Room type", "room_type", "text"),
                    C("guests", "Guests", "guests", "text"),
                    C("price", "Price", "price", "price"),
                    C("availability", "Availability", "room_avail", "text"),
                ),
                (C("bed", "Bed", "bed", "text"), C("size", "Size", "area_m2", "text")),
                "room types with guests, price and availability",
            ),
        ),
        kvs=(
            KVProto(
                "policies",
                ("Policies", "House rules"),
                (
                    ("Pets", "pets"),
                    ("Smoking", "smoking"),
                    ("Children", "children_policy"),
                    ("Cancellation", "refund_policy"),
                    ("Payment", "payment"),
                    ("Breakfast", "breakfast"),
                    ("Parking", "parking"),
                    ("Wi-Fi", "wifi"),
                ),
                "hotel policy pairs such as pets and cancellation",
            ),
        ),
        related="Hotels nearby",
    ),
    KindProto(
        kind="book",
        noun="book",
        segments=("books", "ebooks", "titles", "library"),
        url_style="slug-id",
        prefixes=("book", "title", "volume", "tome"),
        title=F(
            "title",
            ("Book",),
            "book_title",
            "text",
            "the book title shown as the page heading",
            "h1",
        ),
        scalars=(
            F("author", ("Author", "By", "Written by"), "person_name", "text", "book author"),
            F("isbn", ("ISBN", "ISBN-13"), "isbn", "text", "ISBN-13"),
            F("publisher", ("Publisher", "Published by"), "company", "text", "publisher"),
            F("published", ("Published", "Publication date"), "date", "date", "publication date"),
            F("pages", ("Pages", "Print length"), "count", "number", "number of pages"),
            F("language", ("Language",), "language", "text", "language"),
            F("format", ("Format", "Binding"), "book_format", "text", "format such as Paperback"),
            F(
                "price",
                ("Price", "Buy for"),
                "price",
                "price",
                "price including the currency symbol",
                "strong",
            ),
            F(
                "rating",
                ("Rating", "Average rating"),
                "rating",
                "number",
                "average rating out of 5",
            ),
            F("series", ("Series",), "series_name", "text", "series name"),
            F("translator", ("Translated by", "Translator"), "person_name", "text", "translator"),
            F(
                "description",
                ("Synopsis", "About the book", "Blurb"),
                "paragraph",
                "text",
                "book synopsis paragraph",
                "p",
            ),
        ),
        lists=(
            ListProto("genres", ("Genres", "Subjects"), "genre", "text", "book genres", link=True),
        ),
        tables=(
            TableProto(
                "editions",
                ("Editions", "Other formats"),
                (
                    C("format", "Format", "book_format", "text"),
                    C("year", "Year", "year", "number"),
                    C("isbn", "ISBN", "isbn", "text"),
                    C("price", "Price", "price", "price"),
                ),
                (
                    C("publisher", "Publisher", "company", "text"),
                    C("pages", "Pages", "count", "number"),
                ),
                "editions with format, year, ISBN and price",
            ),
        ),
        kvs=(
            KVProto(
                "book_details",
                ("Product details", "Book details"),
                (
                    ("Publisher", "company"),
                    ("Language", "language"),
                    ("Dimensions", "dimensions"),
                    ("Reading age", "reading_age"),
                    ("Weight", "weight"),
                    ("Edition", "edition"),
                ),
                "book detail pairs such as dimensions and reading age",
            ),
        ),
        related="Readers also bought",
    ),
    KindProto(
        kind="movie",
        noun="movie",
        segments=("movies", "films", "titles-film", "cinema"),
        url_style="id-slug",
        prefixes=("movie", "film", "feature", "picture"),
        title=F(
            "title",
            ("Movie",),
            "movie_title",
            "text",
            "the movie title shown as the page heading",
            "h1",
        ),
        scalars=(
            F("director", ("Director", "Directed by"), "person_name", "text", "director"),
            F("release_date", ("Release date", "Released"), "date", "date", "release date"),
            F("runtime", ("Runtime", "Running time", "Length"), "runtime", "text", "running time"),
            F("rating", ("Rating", "User score"), "rating", "number", "user rating"),
            F("genre", ("Genre",), "genre", "text", "primary genre"),
            F("studio", ("Studio", "Production"), "company", "text", "studio"),
            F(
                "budget",
                ("Budget",),
                "price_large",
                "price",
                "production budget including the currency symbol",
            ),
            F(
                "box_office",
                ("Box office", "Gross"),
                "price_large",
                "price",
                "box office gross including the currency symbol",
            ),
            F(
                "language",
                ("Language", "Original language"),
                "language",
                "text",
                "original language",
            ),
            F("country", ("Country", "Country of origin"), "country", "text", "country of origin"),
            F("certificate", ("Rated", "Certificate"), "certificate", "text", "age certificate"),
            F(
                "description",
                ("Synopsis", "Plot", "Storyline"),
                "paragraph",
                "text",
                "plot synopsis paragraph",
                "p",
            ),
        ),
        lists=(
            ListProto(
                "keywords", ("Keywords", "Plot keywords"), "tag", "text", "plot keywords", link=True
            ),
        ),
        tables=(
            TableProto(
                "cast",
                ("Cast", "Cast and crew"),
                (
                    C("actor", "Actor", "person_name", "text"),
                    C("character", "Role", "character", "text"),
                ),
                (
                    C("episodes", "Episodes", "count_small", "number"),
                    C("credit", "Credited as", "credit", "text"),
                ),
                "cast list with actor and role",
            ),
        ),
        kvs=(
            KVProto(
                "tech_specs",
                ("Technical specs", "Details"),
                (
                    ("Aspect ratio", "aspect"),
                    ("Sound mix", "sound"),
                    ("Color", "film_color"),
                    ("Filming location", "city"),
                    ("Camera", "camera"),
                    ("Production company", "company"),
                ),
                "technical specification pairs",
            ),
        ),
        related="More like this",
    ),
    KindProto(
        kind="course",
        noun="course",
        segments=("courses", "classes", "programs", "learn"),
        url_style="slug",
        prefixes=("course", "class", "program", "lesson"),
        title=F(
            "title",
            ("Course",),
            "course_title",
            "text",
            "the course title shown as the page heading",
            "h1",
        ),
        scalars=(
            F("instructor", ("Instructor", "Taught by"), "person_name", "text", "instructor"),
            F(
                "price",
                ("Price", "Tuition", "Fee"),
                "price",
                "price",
                "course price including the currency symbol",
                "strong",
            ),
            F("duration", ("Duration", "Length"), "duration_weeks", "text", "course duration"),
            F("level", ("Level", "Difficulty"), "level", "text", "course level"),
            F("language", ("Language", "Taught in"), "language", "text", "teaching language"),
            F("start_date", ("Starts", "Start date", "Next session"), "date", "date", "start date"),
            F(
                "enrolled",
                ("Enrolled", "Students"),
                "count",
                "number",
                "number of enrolled students",
            ),
            F("rating", ("Rating", "Course rating"), "rating", "number", "rating out of 5"),
            F(
                "provider",
                ("Offered by", "Provider", "School"),
                "company",
                "text",
                "course provider",
            ),
            F("contact_email", ("Contact", "Questions"), "email", "email", "contact email"),
            F(
                "credential",
                ("Certificate", "Credential"),
                "credential",
                "text",
                "certificate awarded",
            ),
            F("effort", ("Effort", "Hours per week"), "effort", "text", "weekly effort"),
            F(
                "description",
                ("About this course", "Overview"),
                "paragraph",
                "text",
                "course description paragraph",
                "p",
            ),
        ),
        lists=(
            ListProto(
                "topics",
                ("What you'll learn", "Topics", "Skills you'll gain"),
                "skill",
                "text",
                "topics covered",
            ),
        ),
        tables=(
            TableProto(
                "syllabus",
                ("Syllabus", "Curriculum"),
                (
                    C("week", "Week", "week", "text"),
                    C("topic", "Topic", "session_title", "text"),
                    C("hours", "Hours", "count_small", "number"),
                ),
                (
                    C("assessment", "Assessment", "assessment", "text"),
                    C("reading", "Reading", "book_title", "text"),
                ),
                "syllabus table with week, topic and hours",
            ),
        ),
        kvs=(
            KVProto(
                "course_facts",
                ("Course details", "At a glance"),
                (
                    ("Format", "event_format"),
                    ("Prerequisites", "prereq"),
                    ("Credits", "count_small"),
                    ("Assessment", "assessment"),
                    ("Subtitles", "language"),
                    ("Certificate", "credential"),
                ),
                "course fact pairs such as format and prerequisites",
            ),
        ),
        related="Students also viewed",
    ),
    KindProto(
        kind="restaurant",
        noun="restaurant",
        segments=("restaurants", "places", "eat", "dining"),
        url_style="slug-id",
        prefixes=("venue", "restaurant", "eatery", "bistro"),
        title=F(
            "name",
            ("Restaurant",),
            "restaurant_name",
            "text",
            "the restaurant name shown as the page heading",
            "h1",
        ),
        scalars=(
            F("cuisine", ("Cuisine", "Food"), "cuisine", "text", "cuisine"),
            F("address", ("Address", "Find us"), "address", "text", "street address"),
            F("phone", ("Phone", "Call"), "phone", "text", "phone number"),
            F("website", ("Website", "Online"), "url", "url", "restaurant website URL"),
            F(
                "price_range",
                ("Price range", "Prices"),
                "price_range",
                "text",
                "price range such as $$",
            ),
            F("rating", ("Rating", "Diner rating"), "rating", "number", "rating out of 5"),
            F("hours", ("Hours", "Opening hours", "Open"), "hours", "text", "opening hours"),
            F("email", ("Email", "Bookings email"), "email", "email", "contact email"),
            F(
                "reservations",
                ("Reservations", "Booking"),
                "yes_no",
                "text",
                "whether reservations are taken",
            ),
            F("neighborhood", ("Neighborhood", "Area"), "city", "text", "neighborhood"),
            F("chef", ("Chef", "Head chef"), "person_name", "text", "head chef"),
            F("delivery", ("Delivery", "Takeout"), "yes_no", "text", "delivery available"),
            F(
                "description",
                ("About", "Our story"),
                "paragraph",
                "text",
                "restaurant description paragraph",
                "p",
            ),
        ),
        lists=(
            ListProto(
                "dishes", ("Popular dishes", "Signature dishes"), "dish", "text", "popular dishes"
            ),
        ),
        tables=(
            TableProto(
                "menu",
                ("Menu", "Featured menu"),
                (
                    C("dish", "Dish", "dish", "text"),
                    C("description", "Description", "dish_desc", "text"),
                    C("price", "Price", "price_small", "price"),
                ),
                (
                    C("calories", "Calories", "calories", "number"),
                    C("diet", "Diet", "diet", "text"),
                ),
                "menu table with dish, description and price",
            ),
        ),
        kvs=(
            KVProto(
                "restaurant_info",
                ("Restaurant info", "Details", "Good to know"),
                (
                    ("Payment", "payment"),
                    ("Parking", "parking"),
                    ("Wheelchair access", "yes_no"),
                    ("Outdoor seating", "yes_no"),
                    ("Dress code", "dress_code"),
                    ("Noise level", "noise_level"),
                    ("Good for", "occasion"),
                ),
                "restaurant information pairs",
            ),
        ),
        related="Nearby restaurants",
    ),
    KindProto(
        kind="vehicle",
        noun="vehicle listing",
        segments=("cars", "used-cars", "vehicles", "auto"),
        url_style="html",
        prefixes=("vehicle", "car", "auto", "motor"),
        title=F(
            "title",
            ("Vehicle",),
            "car_title",
            "text",
            "the vehicle title shown as the page heading",
            "h1",
        ),
        scalars=(
            F(
                "price",
                ("Price", "Our price"),
                "price_large",
                "price",
                "price including the currency symbol",
                "strong",
            ),
            F("mileage", ("Mileage", "Odometer"), "mileage", "text", "mileage"),
            F("year", ("Year", "Model year"), "year", "number", "model year"),
            F("make", ("Make", "Manufacturer"), "make", "text", "make"),
            F("fuel", ("Fuel", "Fuel type"), "fuel", "text", "fuel type"),
            F("transmission", ("Transmission", "Gearbox"), "transmission", "text", "transmission"),
            F("color", ("Exterior color", "Color"), "color", "text", "exterior color"),
            F("vin", ("VIN",), "vin", "text", "vehicle identification number"),
            F("dealer", ("Dealer", "Seller"), "company", "text", "dealer name"),
            F(
                "dealer_phone",
                ("Dealer phone", "Call dealer"),
                "phone",
                "text",
                "dealer phone number",
            ),
            F("listed", ("Listed", "Date listed"), "date", "date", "date listed"),
            F("condition", ("Condition",), "condition", "text", "condition"),
            F("body_type", ("Body style", "Body"), "body_type", "text", "body style"),
            F(
                "description",
                ("Seller's notes", "Description"),
                "paragraph",
                "text",
                "seller description paragraph",
                "p",
            ),
        ),
        lists=(
            ListProto(
                "features", ("Features", "Options"), "car_feature", "text", "vehicle features"
            ),
        ),
        tables=(
            TableProto(
                "service_history",
                ("Service history", "Maintenance"),
                (
                    C("date", "Date", "date", "date"),
                    C("service", "Service", "service_type", "text"),
                    C("mileage", "Mileage", "mileage", "text"),
                ),
                (C("cost", "Cost", "price_small", "price"), C("shop", "Shop", "company", "text")),
                "service history with date, service and mileage",
            ),
        ),
        kvs=(
            KVProto(
                "vehicle_specs",
                ("Specifications", "Vehicle specs"),
                (
                    ("Engine", "engine"),
                    ("Horsepower", "horsepower"),
                    ("Drivetrain", "drivetrain"),
                    ("Doors", "count_small"),
                    ("Seats", "count_small"),
                    ("Fuel economy", "mpg"),
                    ("Interior color", "color"),
                ),
                "vehicle specification pairs such as engine and drivetrain",
            ),
        ),
        related="Similar vehicles",
    ),
)

KIND_BY_NAME: dict[str, KindProto] = {k.kind: k for k in KINDS}

# Fields any template may borrow when its kind's catalog runs out.
GENERIC_SCALARS: tuple[FieldProto, ...] = (
    F("support_email", ("Questions?", "Support email"), "email", "email", "support email address"),
    F(
        "last_updated",
        ("Last updated", "Updated on"),
        "date",
        "date",
        "date the page was last updated",
    ),
    F("reference_code", ("Ref.", "Reference code"), "reference", "text", "internal reference code"),
    F("views", ("Views", "Page views"), "count", "number", "number of views"),
    F("promotion", ("Offer", "Promotion"), "promo", "text", "current promotion"),
    F("support_phone", ("Support", "Hotline"), "phone", "text", "support phone number"),
    F("notice", ("Note", "Notice"), "sentence", "text", "a notice sentence"),
    F("badge", ("Badge", "Label"), "badge", "text", "badge label"),
    F("permalink", ("Permalink", "Share link"), "url", "url", "permanent link URL"),
    F("region", ("Region", "Area served"), "country", "text", "region or country"),
    F("verification", ("Verified", "Verification"), "verified", "text", "verification status"),
    F("listing_date", ("Added", "Date added"), "date", "date", "date the entry was added"),
)
GENERIC_LISTS: tuple[ListProto, ...] = (
    ListProto("tags", ("Tags",), "tag", "text", "tags", link=True),
    ListProto("keywords", ("Keywords", "Related searches"), "tag", "text", "keywords", link=True),
    ListProto(
        "highlights", ("Highlights", "Why choose this"), "sentence", "text", "highlight sentences"
    ),
)
GENERIC_TABLES: tuple[TableProto, ...] = (
    TableProto(
        "reviews",
        ("Reviews", "Customer reviews", "What people say"),
        (
            C("reviewer", "Reviewer", "reviewer", "text"),
            C("date", "Date", "date", "date"),
            C("rating", "Rating", "rating", "number"),
            C("comment", "Comment", "sentence", "text"),
        ),
        (
            C("verified", "Verified", "yes_no", "text"),
            C("helpful", "Helpful", "count_small", "number"),
        ),
        "reviews with reviewer, date, rating and comment",
    ),
    TableProto(
        "history",
        ("History", "Change log", "Updates"),
        (
            C("date", "Date", "date", "date"),
            C("event", "Event", "history_event", "text"),
            C("by", "By", "person_name", "text"),
        ),
        (C("note", "Note", "sentence", "text"),),
        "history of updates with date, event and author",
    ),
    TableProto(
        "documents",
        ("Documents", "Downloads", "Attachments"),
        (
            C("name", "Name", "document_name", "text"),
            C("type", "Type", "file_type", "text"),
            C("size", "Size", "file_size", "text"),
        ),
        (C("updated", "Updated", "date", "date"),),
        "downloadable documents with name, type and size",
    ),
)
# Optional record-table columns any table may add when its own extras run out.
GENERIC_COLUMNS: tuple[ColumnProto, ...] = (
    C("notes", "Notes", "sentence", "text"),
    C("ref", "Ref.", "reference", "text"),
    C("updated", "Updated", "date", "date"),
)
GENERIC_KVS: tuple[KVProto, ...] = (
    KVProto(
        "additional_info",
        ("Additional information", "More details"),
        (
            ("Reference", "reference"),
            ("Last updated", "date"),
            ("Language", "language"),
            ("Region", "country"),
            ("Category", "badge"),
            ("Contact", "email"),
        ),
        "additional information pairs",
    ),
    KVProto(
        "shipping",
        ("Shipping and returns", "Delivery"),
        (
            ("Delivery", "delivery_time"),
            ("Returns", "returns"),
            ("Shipping cost", "price_small"),
            ("Ships to", "country"),
            ("Carrier", "carrier"),
        ),
        "shipping and returns pairs",
    ),
    KVProto(
        "contact_info",
        ("Contact information", "Get in touch"),
        (
            ("Phone", "phone"),
            ("Email", "email"),
            ("Address", "address"),
            ("Website", "url"),
            ("Hours", "hours"),
        ),
        "contact information pairs",
    ),
)

# Words appended to a class prefix when several templates share a kind (no digits: a digit in a
# 5+ alnum run would make the class look hashed to the canonicalizer).
PREFIX_MODIFIERS: tuple[str, ...] = (
    "alt",
    "plus",
    "pro",
    "classic",
    "lite",
    "prime",
    "max",
    "mini",
    "next",
    "neo",
    "core",
    "edge",
    "nova",
    "zen",
    "aero",
    "terra",
    "luna",
    "sol",
    "vista",
    "nord",
    "west",
    "east",
    "north",
    "south",
    "blue",
    "red",
    "green",
    "gold",
    "silver",
    "iron",
    "oak",
    "pine",
)

# Site navigation and footer vocabulary.
NAV_WORDS: tuple[str, ...] = (
    "Home",
    "New arrivals",
    "Bestsellers",
    "Sale",
    "Brands",
    "Gift cards",
    "Stores",
    "Blog",
    "Help",
    "About",
    "Careers",
    "Contact",
    "Events",
    "Deals",
    "Community",
    "Guides",
    "Press",
    "Partners",
    "Pricing",
    "Features",
    "Resources",
    "Support",
    "Shop",
    "Explore",
    "Popular",
    "Collections",
    "Trending",
    "Outlet",
    "Membership",
    "Rewards",
    "Locations",
    "Services",
)
FOOTER_HEADINGS: tuple[str, ...] = (
    "Company",
    "Customer service",
    "Resources",
    "Legal",
    "Follow us",
    "Shop",
    "Explore",
    "Community",
    "Partners",
    "Support",
    "Discover",
    "More",
)
FOOTER_LINKS: tuple[str, ...] = (
    "About us",
    "Careers",
    "Press",
    "Investors",
    "Contact",
    "Privacy policy",
    "Terms of use",
    "Cookie settings",
    "Accessibility",
    "Sitemap",
    "Help center",
    "Shipping",
    "Returns",
    "Order status",
    "Gift cards",
    "Affiliates",
    "Advertise",
    "Developers",
    "Status",
    "Security",
    "Newsletter",
    "Store locator",
    "Sustainability",
    "Blog",
    "Podcast",
    "Events",
    "FAQ",
    "Warranty",
    "Feedback",
    "Do not sell my info",
)

# Value vocabularies (Faker supplies names, companies, addresses, phones, domains, ISBNs).
ADJECTIVES = (
    "Classic",
    "Compact",
    "Deluxe",
    "Ergonomic",
    "Essential",
    "Everyday",
    "Lightweight",
    "Modern",
    "Portable",
    "Premium",
    "Rustic",
    "Sleek",
    "Smart",
    "Sturdy",
    "Ultra",
    "Vintage",
    "Wireless",
    "Heavy-Duty",
    "Eco",
    "Foldable",
    "Insulated",
    "Adjustable",
    "Handcrafted",
)
MATERIALS = (
    "Steel",
    "Oak",
    "Leather",
    "Cotton",
    "Bamboo",
    "Ceramic",
    "Glass",
    "Aluminum",
    "Wool",
    "Linen",
    "Walnut",
    "Copper",
    "Marble",
    "Silicone",
    "Canvas",
    "Granite",
    "Titanium",
)
PRODUCT_NOUNS = (
    "Chair",
    "Lamp",
    "Backpack",
    "Kettle",
    "Desk",
    "Speaker",
    "Headphones",
    "Jacket",
    "Mug",
    "Notebook",
    "Blender",
    "Sofa",
    "Watch",
    "Bottle",
    "Tent",
    "Bicycle",
    "Camera",
    "Keyboard",
    "Monitor",
    "Pan",
    "Rug",
    "Shelf",
    "Toaster",
    "Umbrella",
    "Wallet",
    "Sneakers",
    "Drill",
    "Router",
    "Pillow",
    "Blanket",
    "Vase",
    "Clock",
    "Mirror",
    "Stool",
    "Grill",
)
CATEGORIES = (
    "Home & Kitchen",
    "Electronics",
    "Outdoors",
    "Furniture",
    "Fashion",
    "Sports",
    "Garden",
    "Office",
    "Toys & Games",
    "Beauty",
    "Tools",
    "Automotive",
    "Books",
    "Pet supplies",
)
AVAILABILITY = ("In stock", "Out of stock", "Only a few left", "Pre-order", "Ships in 2-3 weeks")
SIZES = ("XS", "S", "M", "L", "XL", "XXL", "One size", "32", "34", "36", "38", "40", "42")
STOCK = ("In stock", "Low stock", "Sold out", "Backorder", "Available")
DEPARTMENTS = (
    "Engineering",
    "Marketing",
    "Sales",
    "Finance",
    "Operations",
    "Design",
    "Legal",
    "Customer Success",
    "Research",
    "Human Resources",
    "Product",
    "Data",
    "Security",
)
EMPLOYMENT_TYPES = ("Full-time", "Part-time", "Contract", "Temporary", "Internship", "Freelance")
EXPERIENCE_LEVELS = ("Entry level", "Junior", "Mid-level", "Senior", "Lead", "Director")
WORK_MODES = ("On-site", "Hybrid", "Remote", "Remote (US only)", "Flexible")
SKILLS = (
    "Python",
    "Data analysis",
    "Project management",
    "Public speaking",
    "SQL",
    "Machine learning",
    "Negotiation",
    "UX research",
    "Copywriting",
    "Cloud architecture",
    "Accounting",
    "Leadership",
    "Photography",
    "Video editing",
    "Kubernetes",
    "Statistics",
    "Customer support",
    "Sales",
    "Product strategy",
    "Graphic design",
    "Rust",
    "TypeScript",
    "Supply chain",
    "Mentoring",
)
EVENT_TYPES = (
    "Concert",
    "Conference",
    "Workshop",
    "Festival",
    "Meetup",
    "Exhibition",
    "Webinar",
    "Comedy",
    "Theatre",
    "Sports",
)
EVENT_THEMES = (
    "Summer",
    "Winter",
    "Night",
    "Open",
    "Annual",
    "Spring",
    "Global",
    "Local",
    "Future",
    "City",
)
VENUE_NOUNS = (
    "Hall",
    "Arena",
    "Theatre",
    "Pavilion",
    "Center",
    "Gardens",
    "Club",
    "Studio",
    "Forum",
    "Warehouse",
)
CUISINES = (
    "Italian",
    "Mexican",
    "Japanese",
    "Thai",
    "Indian",
    "French",
    "Greek",
    "Korean",
    "Lebanese",
    "Spanish",
    "Vietnamese",
    "American",
    "Ethiopian",
    "Peruvian",
)
DISH_BASES = (
    "Risotto",
    "Tacos",
    "Ramen",
    "Curry",
    "Salad",
    "Pasta",
    "Stew",
    "Burger",
    "Soup",
    "Pie",
    "Tart",
    "Bowl",
    "Skewers",
    "Flatbread",
    "Dumplings",
    "Paella",
)
DISH_MODS = (
    "Roasted",
    "Spicy",
    "Creamy",
    "Smoky",
    "Herbed",
    "Grilled",
    "Braised",
    "Crispy",
    "Lemon",
    "Garlic",
    "Honey",
    "Wild Mushroom",
    "Coconut",
    "Chili",
)
INGREDIENTS = (
    "flour",
    "sugar",
    "butter",
    "eggs",
    "milk",
    "olive oil",
    "garlic",
    "onion",
    "tomatoes",
    "salt",
    "black pepper",
    "lemon juice",
    "parsley",
    "rice",
    "chicken thighs",
    "chickpeas",
    "coconut milk",
    "basil",
    "parmesan",
    "spinach",
    "honey",
    "cumin",
    "paprika",
    "yogurt",
)
UNITS = ("cups", "tbsp", "tsp", "g", "ml", "cloves", "pinch of", "handful of", "oz")
DIFFICULTY = ("Easy", "Medium", "Hard", "Beginner", "Advanced")
MEAL_COURSES = ("Breakfast", "Lunch", "Dinner", "Dessert", "Snack", "Starter", "Side dish")
NUTRIENTS = (
    "Calories",
    "Fat",
    "Saturates",
    "Carbohydrates",
    "Sugars",
    "Fibre",
    "Protein",
    "Salt",
    "Sodium",
    "Cholesterol",
)
DIETS = (
    "Vegetarian",
    "Vegan",
    "Gluten-free",
    "Dairy-free",
    "Low-carb",
    "Keto",
    "Pescatarian",
    "None",
)
EQUIPMENT = (
    "Oven",
    "Blender",
    "Cast-iron pan",
    "Stand mixer",
    "Slow cooker",
    "Food processor",
    "Grill",
)
SEASONS = ("Spring", "Summer", "Autumn", "Winter", "All year")
OCCASIONS = ("Weeknight", "Party", "Holiday", "Date night", "Family", "Picnic", "Brunch")
SECTIONS = (
    "World",
    "Business",
    "Technology",
    "Science",
    "Culture",
    "Sports",
    "Opinion",
    "Travel",
    "Health",
    "Politics",
    "Climate",
)
HEADLINE_VERBS = (
    "Reveals",
    "Explains",
    "Rethinks",
    "Faces",
    "Plans",
    "Tests",
    "Launches",
    "Questions",
    "Embraces",
    "Rebuilds",
)
INDUSTRIES = (
    "Software",
    "Retail",
    "Manufacturing",
    "Healthcare",
    "Logistics",
    "Energy",
    "Finance",
    "Media",
    "Hospitality",
    "Construction",
    "Education",
    "Agriculture",
)
COMPANY_STATUS = ("Active", "Dormant", "Acquired", "Private", "Public")
LEGAL_FORMS = ("LLC", "Corporation", "GmbH", "Ltd", "S.A.", "Partnership", "Cooperative")
EXCHANGES = ("NYSE", "NASDAQ", "LSE", "Euronext", "TSX", "Not listed")
PROPERTY_TYPES = (
    "Single family",
    "Condo",
    "Townhouse",
    "Apartment",
    "Multi-family",
    "Bungalow",
    "Loft",
)
LISTING_EVENTS = (
    "Listed for sale",
    "Price change",
    "Sold",
    "Pending",
    "Relisted",
    "Listing removed",
)
LISTING_SOURCES = ("MLS", "Agent", "Public record", "Owner")
HEATING = ("Forced air", "Radiant", "Heat pump", "Baseboard", "Gas furnace", "None")
COOLING = ("Central air", "Window units", "Ductless mini-split", "None")
FLOORING = ("Hardwood", "Carpet", "Tile", "Laminate", "Vinyl", "Concrete")
ROOFS = ("Asphalt shingle", "Metal", "Tile", "Slate", "Flat")
VIEWS = ("City", "Mountain", "Water", "Park", "None")
ENERGY = ("A", "B", "C", "D", "E", "F")
PARKING = ("Garage, 2 spaces", "Street parking", "Driveway", "Carport", "None", "Valet")
AMENITIES = (
    "Pool",
    "Gym",
    "Free Wi-Fi",
    "Spa",
    "Restaurant",
    "Bar",
    "Airport shuttle",
    "Laundry",
    "Balcony",
    "Fireplace",
    "Garden",
    "Air conditioning",
    "Elevator",
    "Pet friendly",
    "Room service",
)
HOTEL_NOUNS = ("Hotel", "Inn", "Suites", "Lodge", "Resort", "House", "Residences")
HOTEL_MODS = (
    "Grand",
    "Royal",
    "Harbor",
    "Park",
    "Riverside",
    "Central",
    "Garden",
    "Plaza",
    "Summit",
    "Bay",
)
ROOM_TYPES = (
    "Standard Double",
    "Deluxe King",
    "Twin Room",
    "Junior Suite",
    "Family Room",
    "Studio",
    "Penthouse Suite",
)
GUESTS = ("1 adult", "2 adults", "2 adults, 1 child", "4 guests", "2 adults, 2 children")
ROOM_AVAIL = ("Available", "Only 1 left", "Sold out", "Request")
BEDS = ("1 king bed", "2 twin beds", "1 queen bed", "1 sofa bed", "2 double beds")
PETS = ("Allowed", "Not allowed", "On request", "Allowed, fees apply")
SMOKING = ("Non-smoking", "Designated areas", "Not allowed")
CHILDREN = ("Welcome", "Over 12 only", "Not allowed")
PAYMENT = ("Cards and cash", "Card only", "Cash only", "Visa, Mastercard, Amex")
BREAKFAST = ("Included", "Not included", "Available, fees apply", "Continental")
WIFI = ("Free", "Paid", "Lobby only")
REFUND = (
    "Full refund up to 48 hours before",
    "Non-refundable",
    "Partial refund",
    "Free cancellation",
)
BOOK_FORMATS = ("Hardcover", "Paperback", "Ebook", "Audiobook", "Mass market paperback")
GENRES = (
    "Fiction",
    "Mystery",
    "Science fiction",
    "Fantasy",
    "Biography",
    "History",
    "Romance",
    "Thriller",
    "Poetry",
    "Self-help",
    "Drama",
    "Comedy",
    "Horror",
    "Documentary",
)
LANGUAGES = (
    "English",
    "Spanish",
    "French",
    "German",
    "Italian",
    "Portuguese",
    "Japanese",
    "Arabic",
    "Dutch",
    "Korean",
    "Swedish",
)
READING_AGES = ("3 - 5 years", "8 - 12 years", "Young adult", "Adult")
EDITIONS = (
    "First edition",
    "Second edition",
    "Revised edition",
    "Anniversary edition",
    "Illustrated edition",
)
CERTIFICATES = ("G", "PG", "PG-13", "R", "12A", "15", "18")
ASPECTS = ("1.85 : 1", "2.39 : 1", "1.33 : 1", "16 : 9")
SOUND = ("Dolby Digital", "DTS", "Dolby Atmos", "Stereo", "Mono")
FILM_COLORS = ("Color", "Black and White")
CAMERAS = ("Arri Alexa", "Red Dragon", "Panavision", "Sony Venice", "Super 16")
CHARACTER_TITLES = (
    "Detective",
    "Captain",
    "Doctor",
    "Agent",
    "Professor",
    "Mayor",
    "Officer",
    "Sister",
)
CREDITS = ("uncredited", "voice", "as herself", "as himself", "archive footage")
LEVELS = ("Beginner", "Intermediate", "Advanced", "All levels")
CREDENTIALS = (
    "Certificate of completion",
    "Professional certificate",
    "None",
    "Diploma",
    "Digital badge",
)
ASSESSMENTS = ("Quiz", "Project", "Peer review", "Exam", "Essay", "None")
PREREQS = (
    "None",
    "Basic algebra",
    "Introductory programming",
    "High school biology",
    "Prior course",
)
EVENT_FORMATS = ("In person", "Online", "Hybrid", "Self-paced")
PRICE_RANGES = ("$", "$$", "$$$", "$$$$")
NOISE_LEVELS = ("Quiet", "Moderate", "Lively", "Loud")
DRESS_CODES = ("Casual", "Smart casual", "Business", "Formal", "None")
ACCESSIBILITY = ("Step-free access", "Wheelchair accessible", "Hearing loop", "Limited access")
TICKET_TYPES = ("General admission", "VIP", "Early bird", "Student", "Group of 4", "Day pass")
TICKET_STATUS = ("On sale", "Sold out", "Few left", "Coming soon")
TRACKS = ("Main stage", "Workshop track", "Business", "Technical", "Community")
CAR_MAKES = (
    "Toyota",
    "Honda",
    "Ford",
    "Volkswagen",
    "BMW",
    "Hyundai",
    "Kia",
    "Subaru",
    "Mazda",
    "Audi",
    "Tesla",
    "Volvo",
    "Nissan",
)
CAR_MODELS = (
    "Camry",
    "Civic",
    "Focus",
    "Golf",
    "3 Series",
    "Elantra",
    "Sportage",
    "Outback",
    "CX-5",
    "A4",
    "Model 3",
    "XC60",
    "Leaf",
    "Corolla",
    "Accord",
)
CAR_TRIMS = ("SE", "LE", "Sport", "Touring", "Limited", "Base", "Premium", "XLE")
FUELS = ("Gasoline", "Diesel", "Hybrid", "Electric", "Plug-in hybrid")
TRANSMISSIONS = ("Automatic", "Manual", "CVT", "Dual-clutch")
CONDITIONS = ("Used", "Certified pre-owned", "New", "Like new", "Fair")
BODY_TYPES = ("Sedan", "SUV", "Hatchback", "Coupe", "Wagon", "Pickup", "Convertible", "Minivan")
CAR_FEATURES = (
    "Backup camera",
    "Bluetooth",
    "Heated seats",
    "Sunroof",
    "Navigation",
    "Apple CarPlay",
    "Lane assist",
    "Cruise control",
    "Keyless entry",
    "Third-row seating",
    "Tow package",
)
SERVICE_TYPES = (
    "Oil change",
    "Tire rotation",
    "Brake service",
    "Inspection",
    "Battery replacement",
    "Transmission service",
    "Alignment",
)
DRIVETRAINS = ("FWD", "RWD", "AWD", "4WD")
HISTORY_EVENTS = (
    "Created",
    "Edited",
    "Price updated",
    "Photos added",
    "Verified",
    "Archived",
    "Republished",
)
FILE_TYPES = ("PDF", "DOCX", "XLSX", "ZIP", "CSV", "PNG")
DOC_NOUNS = (
    "Manual",
    "Brochure",
    "Datasheet",
    "Guide",
    "Report",
    "Checklist",
    "Floor plan",
    "Price list",
)
DELIVERY = ("2-4 business days", "Next day", "1-2 weeks", "Same day", "3-5 business days")
RETURNS = ("30 days", "14 days", "No returns", "60 days", "Free returns")
CARRIERS = ("UPS", "FedEx", "DHL", "USPS", "Royal Mail", "Local courier")
BADGES = ("Bestseller", "Staff pick", "New", "Featured", "Limited", "Top rated", "Eco choice")
PROMOS = (
    "10% off with code SAVE10",
    "Free shipping today",
    "Buy one get one",
    "Members save 15%",
    "Bundle and save",
)
VERIFIED = ("Verified", "Pending verification", "Unverified")
LICENSES = ("All rights reserved", "CC BY 4.0", "CC BY-SA 4.0", "Syndicated")
COLORS_FALLBACK = (
    "Black",
    "White",
    "Navy",
    "Olive",
    "Red",
    "Sand",
    "Charcoal",
    "Teal",
    "Burgundy",
    "Grey",
)
CATEGORY_WORDS_JOB = (
    "Engineering",
    "Marketing",
    "Operations",
    "Healthcare",
    "Education",
    "Retail",
    "Finance",
)
SCHEDULES = ("Monday to Friday", "Shift work", "Weekends", "Flexible hours", "Night shift")
TRAVEL = ("None", "Up to 25%", "Up to 50%", "Frequent")
TIMEZONES = ("UTC-8", "UTC-5", "UTC+0", "UTC+1", "UTC+2", "UTC+5:30", "UTC+9")
DEGREES = (
    "BSc Computer Science",
    "MBA",
    "BA History",
    "MSc Physics",
    "PhD Biology",
    "BA Economics",
    "MFA Design",
)
PRONOUNS = ("she/her", "he/him", "they/them")
BENEFITS = (
    "Health insurance",
    "401(k) matching",
    "Flexible hours",
    "Remote work budget",
    "Parental leave",
    "Learning stipend",
    "Gym membership",
    "Stock options",
    "Paid time off",
    "Commuter benefits",
)
REQUIREMENT_OPENERS = (
    "Experience with",
    "Strong knowledge of",
    "Familiarity with",
    "A track record in",
    "Comfort with",
)
SERVICES = (
    "Consulting",
    "Custom manufacturing",
    "Logistics",
    "Software development",
    "Installation",
    "Maintenance",
    "Training",
    "Design",
    "Research",
    "Wholesale",
    "Financing",
    "Support plans",
)
OFFICE_NAMES = (
    "Headquarters",
    "Regional office",
    "Sales office",
    "R&D center",
    "Warehouse",
    "Showroom",
)
AGE_LIMITS = ("All ages", "18+", "21+", "16+ with adult")
FEATURE_VERBS = ("Keeps", "Holds", "Fits", "Supports", "Charges", "Folds", "Cleans", "Protects")
FEATURE_OBJECTS = (
    "up to 2 liters",
    "most laptops",
    "drinks cold for 24 hours",
    "a full day of use",
    "into a compact case",
    "in under 10 minutes",
    "against scratches",
    "up to 120 kg",
)
