from __future__ import annotations

from pathlib import Path

import pytest
from lwie_fixtures import build_release


@pytest.fixture
def release_root(tmp_path: Path) -> Path:
    return build_release(tmp_path)
