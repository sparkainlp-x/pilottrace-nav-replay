# Copyright (C) 2026 Jean-François Brisson / Spark AI NLP. SPDX-License-Identifier: AGPL-3.0-only
"""Scenario catalogue: configuration, inertial parameter profiles and labelled degradation events.

All values are SYNTHETIC modelling assumptions chosen for a readable software replay. None is a
measurement of a real receiver, inertial unit, vessel or jammer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

EVENT_KINDS = (
    "gnss_dropout",
    "jamming",
    "spoof_ramp",
    "spoof_step",
    "gyro_drift",
    "gyro_dropout",
    "speed_log_dropout",
    "compass_dropout",
)


@dataclass(frozen=True)
class InertialProfile:
    """Gyro error model for heading propagation (degrees per second)."""

    name: str
    gyro_bias_dps: float
    gyro_noise_dps: float
    note: str


INERTIAL_PROFILES: dict[str, InertialProfile] = {
    "classical_dr": InertialProfile(
        "classical_dr",
        gyro_bias_dps=0.002,
        gyro_noise_dps=0.01,
        note="Illustrative low-cost dead-reckoning gyro (bias 7.2 deg/h). SYNTHETIC assumption.",
    ),
    "hypothetical_low_drift": InertialProfile(
        "hypothetical_low_drift",
        gyro_bias_dps=0.00002,
        gyro_noise_dps=0.001,
        note=(
            "Hypothetical low-drift parameter set (bias 0.072 deg/h), standing in for a future "
            "low-drift inertial sensor. TARGET-style assumption, not a measurement of any device; "
            "no sensor is modelled physically and no benefit is established."
        ),
    ),
}


@dataclass(frozen=True)
class Event:
    """A labelled degradation over epochs ``start <= t < end``."""

    kind: str
    start: int
    end: int
    params: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in EVENT_KINDS:
            raise ValueError(f"unknown event kind {self.kind!r}")
        if not 0 <= self.start < self.end:
            raise ValueError(f"event {self.kind}: need 0 <= start < end")

    def param(self, key: str, default: float) -> float:
        return dict(self.params).get(key, default)

    def active(self, t: int) -> bool:
        return self.start <= t < self.end


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    title: str
    description: str
    events: tuple[Event, ...] = ()
    expected_channels: tuple[str, ...] = ()


@dataclass(frozen=True)
class SimConfig:
    seed: int = 42
    epochs: int = 900
    dt: float = 1.0
    window: int = 32
    tau: float = 0.50
    inertial_profile: str = "classical_dr"
    filter_gain: float = 0.25
    innovation_gate_m: float = 30.0
    reset_after_rejections: int = 10
    gnss_pos_sigma_m: float = 2.0
    gnss_vel_sigma_mps: float = 0.08
    gnss_cn0_nominal: float = 45.0
    gnss_cn0_sigma: float = 0.8
    gnss_sats_nominal: int = 10
    speed_log_sigma_mps: float = 0.08
    compass_sigma_deg: float = 1.0
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if type(self.seed) is not int:
            raise ValueError("seed must be an int")
        if self.window != 32:
            raise ValueError("window must be 32 (one OES block = 32 epochs of one channel)")
        if self.epochs <= self.window:
            raise ValueError("epochs must exceed the 32-epoch window")
        if self.inertial_profile not in INERTIAL_PROFILES:
            raise ValueError(f"unknown inertial profile {self.inertial_profile!r}")
        if not (0.0 < self.filter_gain <= 1.0) or self.dt <= 0 or self.tau < 0:
            raise ValueError("filter_gain must be in (0, 1], dt > 0 and tau >= 0")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


SCENARIOS: dict[str, ScenarioSpec] = {
    spec.name: spec
    for spec in (
        ScenarioSpec(
            "nominal",
            "Nominal GNSS",
            "GNSS, gyro, speed log and compass all nominal for the whole route; no events.",
        ),
        ScenarioSpec(
            "total_dropout",
            "Total GNSS dropout",
            "No GNSS fix for 300 s; the estimate coasts on gyro heading and speed log (dead reckoning).",
            (Event("gnss_dropout", 300, 600),),
            ("gnss_unavailable",),
        ),
        ScenarioSpec(
            "jamming",
            "Jamming-like interference",
            "For 180 s: C/N0 falls about 18 dB-Hz, 3-5 satellites are lost, position and velocity noise grow "
            "and 25% of fixes are lost.",
            (
                Event(
                    "jamming",
                    300,
                    480,
                    (("cn0_drop", 18.0), ("pos_sigma_m", 12.0), ("vel_sigma_mps", 0.6), ("fix_loss_prob", 0.25)),
                ),
            ),
            ("gnss_cn0_deficit", "gnss_sat_deficit", "gnss_unavailable"),
        ),
        ScenarioSpec(
            "spoof_ramp",
            "Spoofing: slow position ramp",
            "For 300 s a consistent spoofer drags GNSS position east at 0.5 m/s with matching velocity, then stops.",
            (Event("spoof_ramp", 300, 600, (("rate_east_mps", 0.5),)),),
            ("gnss_vel_innov_e", "gnss_pos_innov_e", "speed_resid"),
        ),
        ScenarioSpec(
            "spoof_step",
            "Spoofing: abrupt position offset",
            "For 150 s GNSS position jumps 120 m north (velocity unchanged), then returns.",
            (Event("spoof_step", 400, 550, (("offset_north_m", 120.0),)),),
            ("gnss_pos_innov_n", "gnss_rejected"),
        ),
        ScenarioSpec(
            "inertial_drift",
            "Inertial (gyro) drift",
            "From t=300 s to the end the gyro bias grows by 0.03 deg/s; heading is not re-aligned.",
            (Event("gyro_drift", 300, 900, (("extra_bias_dps", 0.03),)),),
            ("heading_resid", "gnss_vel_innov_e", "gnss_vel_innov_n"),
        ),
        ScenarioSpec(
            "sensor_dropouts",
            "Aiding-sensor dropouts",
            "Gyro samples missing for 20 s, speed log for 40 s and compass for 30 s, at different times.",
            (
                Event("gyro_dropout", 250, 270),
                Event("speed_log_dropout", 450, 490),
                Event("compass_dropout", 650, 680),
            ),
            ("gyro_unavailable", "speed_log_unavailable", "compass_unavailable"),
        ),
    )
}
