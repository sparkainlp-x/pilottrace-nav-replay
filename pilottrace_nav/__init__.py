# Copyright (C) 2026 Jean-François Brisson / Spark AI NLP. SPDX-License-Identifier: AGPL-3.0-only
"""PilotTrace Quantum Navigation Replay Pack.

A deterministic, fully SYNTHETIC software replay of GNSS-denied maritime navigation. Navigation
residuals are mapped explicitly into 16 x 32 = 512-value frames and scored through the existing
``pilottrace`` package. No physical sensor (quantum or otherwise) is involved; quantum advantage,
calibration and operational validation are NOT established.
"""

__version__ = "0.1.0"

from .integrations.oes_adapter import (  # noqa: E402
    BLOCK_COUNT,
    BLOCK_WIDTH,
    CHANNELS,
    FRAME_WIDTH,
    FailClosedError,
    latch_score,
    residual_max,
)
from .replay import run_replay, run_scenario  # noqa: E402
from .scenarios import SCENARIOS, SimConfig  # noqa: E402

__all__ = [
    "__version__",
    "BLOCK_COUNT",
    "BLOCK_WIDTH",
    "CHANNELS",
    "FRAME_WIDTH",
    "FailClosedError",
    "SCENARIOS",
    "SimConfig",
    "latch_score",
    "residual_max",
    "run_replay",
    "run_scenario",
]
