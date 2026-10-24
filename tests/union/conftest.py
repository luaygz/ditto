from __future__ import annotations

import pytest
from union_fixtures import profile_table

from ditto.analyze.models import CandidateTable


@pytest.fixture
def table() -> CandidateTable:
    return profile_table()
