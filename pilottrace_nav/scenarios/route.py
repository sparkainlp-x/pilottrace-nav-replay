# Copyright (C) 2026 Jean-François Brisson / Spark AI NLP. SPDX-License-Identifier: AGPL-3.0-only
"""SYNTHETIC maritime route ground truth in a local East-North plane (metres, 1 Hz).

The route is invented: a vessel leaving a fictional harbour and following five waypoints. It is not a
real chart, port or vessel. Heading is measured clockwise from north (degrees), so the velocity is
``(v·sin h, v·cos h)`` in (East, North).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

#: Fictional waypoints (East, North) in metres.
WAYPOINTS: tuple[tuple[float, float], ...] = (
    (0.0, 0.0),
    (1500.0, 400.0),
    (2600.0, 1800.0),
    (3000.0, 3400.0),
    (2200.0, 4600.0),
    (600.0, 5200.0),
)
CRUISE_SPEED = 6.0  # m/s (about 11.7 knots)
SPEED_SWING = 0.3  # m/s, slow sinusoidal speed variation
SPEED_PERIOD = 300.0  # s
MAX_TURN_RATE = 1.0  # deg/s
WAYPOINT_RADIUS = 100.0  # m


@dataclass(frozen=True)
class TruthEpoch:
    t: int
    east: float
    north: float
    heading_deg: float  # [0, 360)
    speed: float  # m/s
    yaw_rate_dps: float  # deg/s over the step that ends at this epoch


def wrap180(angle_deg: float) -> float:
    """Wrap an angle to [-180, 180)."""
    return (angle_deg + 180.0) % 360.0 - 180.0


def bearing_deg(east: float, north: float, to_east: float, to_north: float) -> float:
    return math.degrees(math.atan2(to_east - east, to_north - north)) % 360.0


def truth_trajectory(epochs: int, dt: float = 1.0) -> list[TruthEpoch]:
    """Rate-limited waypoint following; deterministic and noise-free."""
    if epochs < 1 or dt <= 0:
        raise ValueError("epochs must be >= 1 and dt > 0")
    east, north = WAYPOINTS[0]
    target = 1
    heading = bearing_deg(east, north, *WAYPOINTS[1])
    out = [TruthEpoch(0, east, north, heading, CRUISE_SPEED, 0.0)]
    for t in range(1, epochs):
        if target < len(WAYPOINTS) - 1 and math.hypot(WAYPOINTS[target][0] - east, WAYPOINTS[target][1] - north) < (
            WAYPOINT_RADIUS
        ):
            target += 1
        desired = bearing_deg(east, north, *WAYPOINTS[target])
        turn = max(-MAX_TURN_RATE * dt, min(MAX_TURN_RATE * dt, wrap180(desired - heading)))
        heading = (heading + turn) % 360.0
        speed = CRUISE_SPEED + SPEED_SWING * math.sin(2.0 * math.pi * t * dt / SPEED_PERIOD)
        rad = math.radians(heading)
        east += speed * math.sin(rad) * dt
        north += speed * math.cos(rad) * dt
        out.append(TruthEpoch(t, east, north, heading, speed, turn / dt))
    return out
