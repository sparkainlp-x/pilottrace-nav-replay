# Copyright (C) 2026 Jean-François Brisson / Spark AI NLP. SPDX-License-Identifier: AGPL-3.0-only
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pilottrace_nav.replay import run_replay  # noqa: E402


@pytest.fixture(scope="session")
def full_run(tmp_path_factory):
    """One full replay shared by the tests that inspect the evidence (about 5 s)."""
    out = tmp_path_factory.mktemp("replay")
    evidence = run_replay(out)
    return out, evidence
