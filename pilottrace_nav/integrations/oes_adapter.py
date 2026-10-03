# Copyright (C) 2026 Jean-François Brisson / Spark AI NLP. SPDX-License-Identifier: AGPL-3.0-only
"""Explicit mapping of navigation residuals into OES-32 blocks and OES-512 (16 x 32) frames.

Mapping (fixed, declared, no learned parameters):

* There are 16 residual **channels** (``CHANNELS``). Channel ``b`` is divided by its declared scale, so
  ``|value| = 1`` means "one declared tolerance"; the scales are SYNTHETIC modelling choices.
* Block ``b`` of the frame for epoch ``t`` holds channel ``b`` over the 32 most recent epochs, oldest
  first: ``frame[32*b + j] = residual[t - 31 + j][b]`` for ``j = 0..31``. A frame exists only once 32
  epochs are available (no padding, no imputation).
* The classical OES-32 residual is ``R = max|x - x_ref|`` over one 32-value block; here ``x`` is the
  windowed normalised residual and ``x_ref`` the zero reference ("no residual"), unless a caller supplies
  another reference.
* The OES latch score is ``S = 0.45·Peak + 0.35·RMS + 0.20·MeanAbs`` of ``x - x_ref``; a block latches when
  ``S >= tau`` (default 0.50). A frame latches when any block latches.

Fail-closed: any NaN, infinity, boolean, non-number or wrong shape raises :class:`FailClosedError`; nothing
is scored, clipped or imputed. Missing measurements are handled upstream, explicitly: the residual is set
to 0.0 and the matching availability channel is set to 1.0, so a missing sensor always shows up as a
latching block rather than as a silent zero.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

BLOCK_WIDTH = 32
BLOCK_COUNT = 16
FRAME_WIDTH = BLOCK_WIDTH * BLOCK_COUNT
DEFAULT_TAU = 0.50
WEIGHTS = (0.45, 0.35, 0.20)


class FailClosedError(ValueError):
    """Input rejected: non-finite, non-numeric, or wrong shape. Nothing was scored."""


@dataclass(frozen=True)
class Channel:
    index: int
    name: str
    unit: str
    scale: float
    description: str


_CHANNEL_ROWS = (
    ("gnss_pos_innov_e", "m", 20.0, "GNSS east position minus dead-reckoning prediction (0 when no fix)"),
    ("gnss_pos_innov_n", "m", 20.0, "GNSS north position minus dead-reckoning prediction (0 when no fix)"),
    ("gnss_vel_innov_e", "m/s", 1.2, "GNSS east velocity minus dead-reckoning velocity (0 when no fix)"),
    ("gnss_vel_innov_n", "m/s", 1.2, "GNSS north velocity minus dead-reckoning velocity (0 when no fix)"),
    ("gnss_cn0_deficit", "dB-Hz", 30.0, "nominal C/N0 (45) minus reported C/N0 (0 when no fix)"),
    ("gnss_sat_deficit", "satellites", 10.0, "nominal satellite count (10) minus tracked count (0 when no fix)"),
    ("gnss_unavailable", "flag", 1.0, "1 when no GNSS fix is available, else 0"),
    ("gnss_rejected", "flag", 1.0, "1 when the fix failed the innovation gate, else 0"),
    ("heading_resid", "deg", 15.0, "gyro-propagated heading minus compass, wrapped to +/-180 (0 when no compass)"),
    ("speed_resid", "m/s", 1.5, "speed log minus GNSS speed over ground (0 when either is missing)"),
    ("gyro_unavailable", "flag", 1.0, "1 when the gyro sample is missing (last rate held), else 0"),
    ("speed_log_unavailable", "flag", 1.0, "1 when the speed log is missing (last speed held), else 0"),
    ("compass_unavailable", "flag", 1.0, "1 when the compass is missing, else 0"),
    ("reserved_13", "none", 1.0, "reserved; always 0"),
    ("reserved_14", "none", 1.0, "reserved; always 0"),
    ("reserved_15", "none", 1.0, "reserved; always 0"),
)
CHANNELS: tuple[Channel, ...] = tuple(Channel(i, *row) for i, row in enumerate(_CHANNEL_ROWS))
CHANNEL_INDEX: dict[str, int] = {channel.name: channel.index for channel in CHANNELS}
assert len(CHANNELS) == BLOCK_COUNT


def _finite_vector(values: Any, length: int, what: str) -> list[float]:
    if isinstance(values, (str, bytes, dict)) or not isinstance(values, Sequence):
        raise FailClosedError(f"{what} must be a sequence of {length} numbers")
    if len(values) != length:
        raise FailClosedError(f"{what} must contain exactly {length} values (got {len(values)})")
    out = []
    for i, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise FailClosedError(f"{what}[{i}] is not a real number")
        try:
            number = float(value)
        except OverflowError:
            raise FailClosedError(f"{what}[{i}] is outside the finite float range") from None
        if not math.isfinite(number):
            raise FailClosedError(f"{what}[{i}] is not finite (NaN/inf are rejected)")
        out.append(number)
    return out


def _difference(x: Sequence[float], x_ref: Sequence[float] | None) -> list[float]:
    values = _finite_vector(x, BLOCK_WIDTH, "x")
    reference = [0.0] * BLOCK_WIDTH if x_ref is None else _finite_vector(x_ref, BLOCK_WIDTH, "x_ref")
    diff = [a - b for a, b in zip(values, reference, strict=True)]
    if not all(math.isfinite(d) for d in diff):
        raise FailClosedError("x - x_ref overflowed")
    return diff


def residual_max(x: Sequence[float], x_ref: Sequence[float] | None = None) -> float:
    """Classical OES-32 residual ``R = max|x - x_ref|`` over one 32-value block."""
    return max(abs(d) for d in _difference(x, x_ref))


def latch_score(x: Sequence[float], x_ref: Sequence[float] | None = None) -> float:
    """``S = 0.45·Peak + 0.35·RMS + 0.20·MeanAbs`` of ``x - x_ref`` (one 32-value block)."""
    magnitudes = [abs(d) for d in _difference(x, x_ref)]
    peak = max(magnitudes)
    if peak == 0.0:
        return 0.0
    rms = math.sqrt(math.fsum(m * m for m in magnitudes) / BLOCK_WIDTH)
    mean_abs = math.fsum(magnitudes) / BLOCK_WIDTH
    return WEIGHTS[0] * peak + WEIGHTS[1] * rms + WEIGHTS[2] * mean_abs


def latches(score: float, tau: float = DEFAULT_TAU) -> bool:
    if isinstance(tau, bool) or not isinstance(tau, (int, float)) or not math.isfinite(tau) or tau < 0:
        raise FailClosedError("tau must be a finite, non-negative number")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
        raise FailClosedError("score must be finite")
    return score >= tau


def build_frame(window: Sequence[Sequence[float]]) -> list[float]:
    """32 epochs x 16 channels (oldest first) -> one 512-value frame, block ``b`` = channel ``b``."""
    if isinstance(window, (str, bytes)) or not isinstance(window, Sequence) or len(window) != BLOCK_WIDTH:
        raise FailClosedError(f"window must contain exactly {BLOCK_WIDTH} epochs")
    rows = [_finite_vector(row, BLOCK_COUNT, f"window[{i}]") for i, row in enumerate(window)]
    return [rows[j][b] for b in range(BLOCK_COUNT) for j in range(BLOCK_WIDTH)]


def frame_blocks(frame: Sequence[float]) -> list[list[float]]:
    values = _finite_vector(frame, FRAME_WIDTH, "frame")
    return [values[b * BLOCK_WIDTH : (b + 1) * BLOCK_WIDTH] for b in range(BLOCK_COUNT)]


def frames_from_residuals(residuals: Sequence[Sequence[float]]) -> list[tuple[int, list[float]]]:
    """Sliding 32-epoch windows -> ``[(end_epoch, frame)]``; the first frame ends at epoch 31."""
    if len(residuals) < BLOCK_WIDTH:
        raise FailClosedError(f"need at least {BLOCK_WIDTH} epochs of residuals (got {len(residuals)})")
    return [(t, build_frame(residuals[t - BLOCK_WIDTH + 1 : t + 1])) for t in range(BLOCK_WIDTH - 1, len(residuals))]


@dataclass(frozen=True)
class FrameScore:
    scores: tuple[float, ...]
    residual_max: tuple[float, ...]
    latched_blocks: tuple[int, ...]

    @property
    def latched(self) -> bool:
        return bool(self.latched_blocks)


def score_frame(frame: Sequence[float], tau: float = DEFAULT_TAU) -> FrameScore:
    """Fail-closed reference scoring of one 512-value frame (16 blocks)."""
    blocks = frame_blocks(frame)
    scores = tuple(latch_score(block) for block in blocks)
    r_max = tuple(residual_max(block) for block in blocks)
    return FrameScore(scores, r_max, tuple(b for b, s in enumerate(scores) if latches(s, tau)))


def channel_table() -> list[dict[str, Any]]:
    return [
        {
            "block": c.index,
            "frame_range": [c.index * BLOCK_WIDTH, (c.index + 1) * BLOCK_WIDTH],
            "name": c.name,
            "unit": c.unit,
            "scale": c.scale,
            "description": c.description,
        }
        for c in CHANNELS
    ]
