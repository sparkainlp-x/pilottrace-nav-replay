# Copyright (C) 2026 Jean-François Brisson, Spark AI NLP. SPDX-License-Identifier: AGPL-3.0-only
"""Tests for the static Pages builder (scripts/build_site.py)."""

import importlib.util
import re

import pytest
from conftest import ROOT

SPEC = importlib.util.spec_from_file_location("build_site", ROOT / "scripts" / "build_site.py")
build_site = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build_site)


@pytest.fixture(scope="module")
def page():
    return build_site.render(names=["nominal", "spoof_ramp"])


def test_page_is_self_contained(page):
    assert "<script" not in page.lower()
    urls = set(re.findall(r'(?:src|href)="(https?://[^"]+)"', page))
    assert urls == {"https://github.com/sparkainlp-x/pilottrace-nav-replay"}
    assert "Content-Security-Policy" in page


def test_page_states_boundaries_and_misses(page):
    assert "SYNTHETIC" in page
    assert "No physical sensor or other hardware is involved" in page
    assert "NOT latched during event" in page  # the slow spoof is a miss and must be shown as one
    assert "quantum" not in page.lower()


def test_page_has_one_accessible_chart_per_scenario(page):
    assert page.count('role="img"') == 2
    assert page.count("<desc ") == 2


def test_build_is_deterministic(page):
    assert build_site.render(names=["nominal", "spoof_ramp"]) == page


def test_main_writes_index(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(build_site, "render", lambda: "<html></html>")
    assert build_site.main([str(tmp_path)]) == 0
    assert (tmp_path / "index.html").read_text() == "<html></html>"
    assert (tmp_path / ".nojekyll").exists()
    assert "wrote" in capsys.readouterr().out
