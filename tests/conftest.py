"""Shared pytest configuration: live tests run only with an API key; no test touches the network otherwise."""

from __future__ import annotations

import os

import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if os.environ.get("OPENAI_API_KEY"):
        return
    skip_live = pytest.mark.skip(reason="live test: OPENAI_API_KEY not set")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)
