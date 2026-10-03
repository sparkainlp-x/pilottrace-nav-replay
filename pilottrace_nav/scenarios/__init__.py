# Copyright (C) 2026 Jean-François Brisson / Spark AI NLP. SPDX-License-Identifier: AGPL-3.0-only
"""Deterministic SYNTHETIC maritime navigation scenarios (seed 42)."""

from .catalog import INERTIAL_PROFILES, SCENARIOS, Event, InertialProfile, ScenarioSpec, SimConfig
from .route import WAYPOINTS, TruthEpoch, truth_trajectory
from .simulate import EpochRecord, simulate

__all__ = [
    "INERTIAL_PROFILES",
    "SCENARIOS",
    "WAYPOINTS",
    "EpochRecord",
    "Event",
    "InertialProfile",
    "ScenarioSpec",
    "SimConfig",
    "TruthEpoch",
    "simulate",
    "truth_trajectory",
]
