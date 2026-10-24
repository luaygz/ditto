from __future__ import annotations

from types import ModuleType

import pytest
from baseline_helpers import FakeLLM, load_module


@pytest.fixture(scope="session")
def common() -> ModuleType:
    return load_module("common")


@pytest.fixture
def fake_llm_factory() -> type[FakeLLM]:
    return FakeLLM
