"""Stage 5: union page synthesis and compression (one annotated page per template)."""

from __future__ import annotations

from ditto.union.page import UnionPage, build_union_page
from ditto.union.run import build_union_run
from ditto.union.tokens import count_tokens

__all__ = ["UnionPage", "build_union_page", "build_union_run", "count_tokens"]
