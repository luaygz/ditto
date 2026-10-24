"""Token counting for union pages (tiktoken ``o200k_base``)."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import tiktoken

ENCODING = "o200k_base"


@lru_cache(maxsize=1)
def _encoding() -> Any:
    return tiktoken.get_encoding(ENCODING)


def count_tokens(text: str) -> int:
    """Number of ``o200k_base`` tokens in ``text`` (special-token text counts as plain text)."""
    tokens: list[int] = _encoding().encode_ordinary(text)
    return len(tokens)
