# Copyright (C) 2026 Jean-François Brisson / Spark AI NLP. SPDX-License-Identifier: AGPL-3.0-only
import math
import random

import pytest

from pilottrace_nav._deps import load_pilottrace
from pilottrace_nav.integrations import oes_adapter as oa
from pilottrace_nav.integrations.oes_adapter import FailClosedError


def test_formula_by_hand():
    x = [0.0] * 32
    x[0] = 2.0
    expected = 0.45 * 2.0 + 0.35 * math.sqrt(4.0 / 32) + 0.20 * (2.0 / 32)
    assert oa.latch_score(x) == pytest.approx(expected, abs=1e-15)
    assert oa.residual_max(x) == 2.0
    assert oa.latch_score([0.0] * 32) == 0.0
    assert oa.latch_score([1.0] * 32) == pytest.approx(1.0, abs=1e-15)


def test_reference_vector_is_subtracted():
    x = [1.0] * 32
    assert oa.residual_max(x, [1.0] * 32) == 0.0
    assert oa.latch_score(x, [0.5] * 32) == pytest.approx(0.5, abs=1e-15)
    assert oa.residual_max([0] * 32, [-3] * 32) == 3.0


def test_single_unit_spike_latches_at_default_tau():
    x = [0.0] * 32
    x[5] = 1.0  # one missing-sensor epoch in the window
    score = oa.latch_score(x)
    assert score == pytest.approx(0.45 + 0.35 / math.sqrt(32) + 0.20 / 32)
    assert oa.latches(score) and score >= oa.DEFAULT_TAU


@pytest.mark.parametrize(
    "bad",
    [
        [float("nan")] + [0.0] * 31,
        [float("inf")] + [0.0] * 31,
        [float("-inf")] + [0.0] * 31,
        [True] + [0.0] * 31,
        ["1"] + [0.0] * 31,
        [None] + [0.0] * 31,
        [10**400] + [0.0] * 31,
        [0.0] * 31,
        [0.0] * 33,
        "x" * 32,
        b"x" * 32,
        {i: 0.0 for i in range(32)},
        42,
        None,
    ],
)
def test_fail_closed_on_bad_block(bad):
    with pytest.raises(FailClosedError):
        oa.latch_score(bad)
    with pytest.raises(FailClosedError):
        oa.residual_max(bad)


def test_fail_closed_on_bad_reference_and_overflow():
    with pytest.raises(FailClosedError):
        oa.residual_max([0.0] * 32, [float("nan")] * 32)
    with pytest.raises(FailClosedError):
        oa.latch_score([0.0] * 32, [0.0] * 31)
    with pytest.raises(FailClosedError, match="overflow"):
        oa.residual_max([1.7e308] * 32, [-1.7e308] * 32)


@pytest.mark.parametrize("tau", [float("nan"), float("inf"), -0.1, True, "0.5", None])
def test_latches_rejects_bad_tau(tau):
    with pytest.raises(FailClosedError):
        oa.latches(0.7, tau)


@pytest.mark.parametrize("score", [float("nan"), float("inf"), True, None])
def test_latches_rejects_bad_score(score):
    with pytest.raises(FailClosedError):
        oa.latches(score)


def test_latch_boundary_is_inclusive():
    assert oa.latches(0.5, 0.5)
    assert not oa.latches(math.nextafter(0.5, 0.0), 0.5)
    assert oa.latches(0.0, 0.0)


def test_build_frame_mapping_is_block_equals_channel_oldest_first():
    window = [[100.0 * b + j for b in range(16)] for j in range(32)]  # epoch j, channel b
    frame = oa.build_frame(window)
    assert len(frame) == 512
    for b in range(16):
        for j in range(32):
            assert frame[32 * b + j] == 100.0 * b + j


@pytest.mark.parametrize(
    "window",
    [
        [[0.0] * 16] * 31,
        [[0.0] * 16] * 33,
        [[0.0] * 15] + [[0.0] * 16] * 31,
        [[float("nan")] + [0.0] * 15] + [[0.0] * 16] * 31,
        "not a window",
        None,
    ],
)
def test_build_frame_fails_closed(window):
    with pytest.raises(FailClosedError):
        oa.build_frame(window)


def test_frames_from_residuals_windows():
    residuals = [[float(t)] + [0.0] * 15 for t in range(40)]
    frames = oa.frames_from_residuals(residuals)
    assert [t for t, _ in frames] == list(range(31, 40))
    assert frames[0][1][:32] == [float(t) for t in range(32)]
    assert frames[-1][1][:32] == [float(t) for t in range(8, 40)]
    with pytest.raises(FailClosedError):
        oa.frames_from_residuals(residuals[:31])


def test_score_frame_and_frame_shape():
    frame = [0.0] * 512
    frame[32 * 6] = 1.0
    result = oa.score_frame(frame)
    assert result.latched and result.latched_blocks == (6,)
    assert result.residual_max[6] == 1.0
    assert not oa.score_frame([0.0] * 512).latched
    for bad in ([0.0] * 511, [0.0] * 511 + [float("nan")]):
        with pytest.raises(FailClosedError):
            oa.score_frame(bad)


def test_adapter_matches_pilottrace_score_block():
    pilottrace, _ = load_pilottrace()
    rng = random.Random(42)
    for _ in range(500):
        scale = 10 ** rng.uniform(-6, 6)
        block = [rng.gauss(0.0, scale) for _ in range(32)]
        ours, theirs = oa.latch_score(block), pilottrace.score_block(block)
        assert abs(ours - theirs) <= 1e-12 * max(1.0, abs(theirs))


def test_channel_table():
    table = oa.channel_table()
    assert len(table) == oa.BLOCK_COUNT == 16
    assert len({row["name"] for row in table}) == 16
    assert table[15]["frame_range"] == [480, 512]
    assert all(row["scale"] > 0 for row in table)
    assert oa.CHANNEL_INDEX["gnss_unavailable"] == 6


def test_fmt_helper_large_and_none():
    from pilottrace_nav.provenance.evidence_report import _fmt

    assert _fmt(None) == "n/a" and _fmt(2.5e7) == "2.5e+07" and _fmt(1.230) == "1.23" and _fmt(3) == "3"
