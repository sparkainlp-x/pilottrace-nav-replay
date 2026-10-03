# Copyright (C) 2026 Jean-François Brisson / Spark AI NLP. SPDX-License-Identifier: AGPL-3.0-only
"""Sensor synthesis, a deliberately simple navigation estimator, and residual channels.

Everything is SYNTHETIC and deterministic for a given seed. Each sensor draws its nominal noise from its
own seeded stream on every epoch (whether or not the sample is used), so scenarios share identical
nominal noise outside their events; event randomness uses a separate per-scenario stream.

Estimator (documented simplifications, not a production filter):

* heading = free-running integral of the gyro rate (no compass or GNSS re-alignment); the last rate is
  held while gyro samples are missing;
* speed = speed log (water speed, no current); the last value is held while the log is missing;
* prediction = previous estimate + speed·dt along the heading (dead reckoning);
* a GNSS fix is accepted when its position innovation is within ``innovation_gate_m``; then the estimate
  moves a fixed fraction ``filter_gain`` toward the fix. After ``reset_after_rejections`` consecutive
  rejected fixes the estimate is re-initialised to the fix (a common, spoofable recovery rule).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from ..integrations.oes_adapter import CHANNEL_INDEX, CHANNELS
from .catalog import INERTIAL_PROFILES, ScenarioSpec, SimConfig
from .route import truth_trajectory, wrap180


@dataclass(frozen=True)
class EpochRecord:
    t: int
    truth_east: float
    truth_north: float
    est_east: float
    est_north: float
    nav_error_m: float
    degraded: bool
    events: tuple[str, ...]
    raw: tuple[float, ...]  # 16 residual channels in physical units (see CHANNELS)
    residuals: tuple[float, ...]  # raw / declared scale, rounded to 9 decimals


def _streams(seed: int, scenario: str) -> dict[str, random.Random]:
    names = ("gnss", "gyro", "speed_log", "compass")
    streams = {name: random.Random(f"pilottrace-nav:{seed}:{name}") for name in names}
    streams["events"] = random.Random(f"pilottrace-nav:{seed}:events:{scenario}")
    return streams


def simulate(spec: ScenarioSpec, config: SimConfig | None = None) -> list[EpochRecord]:
    """Run one scenario; returns one record per epoch."""
    config = config or SimConfig()
    profile = INERTIAL_PROFILES[config.inertial_profile]
    truth = truth_trajectory(config.epochs, config.dt)
    rng = _streams(config.seed, spec.name)
    dt = config.dt

    est_e, est_n = truth[0].east, truth[0].north
    heading = truth[0].heading_deg
    last_rate = 0.0
    last_speed = truth[0].speed
    rejections = 0
    records: list[EpochRecord] = []

    for tr in truth:
        t = tr.t
        active = [event for event in spec.events if event.active(t)]
        kinds = tuple(sorted({event.kind for event in active}))

        # ---- nominal noise draws (always drawn, so streams stay aligned across scenarios)
        g = rng["gnss"]
        noise_pos = (g.gauss(0.0, 1.0), g.gauss(0.0, 1.0))
        noise_vel = (g.gauss(0.0, 1.0), g.gauss(0.0, 1.0))
        noise_cn0 = g.gauss(0.0, config.gnss_cn0_sigma)
        sat_jitter = g.randint(-1, 1)
        gyro_noise = rng["gyro"].gauss(0.0, profile.gyro_noise_dps)
        log_noise = rng["speed_log"].gauss(0.0, config.speed_log_sigma_mps)
        compass_noise = rng["compass"].gauss(0.0, config.compass_sigma_deg)

        # ---- GNSS measurement
        rad = math.radians(tr.heading_deg)
        true_ve, true_vn = tr.speed * math.sin(rad), tr.speed * math.cos(rad)
        pos_sigma, vel_sigma = config.gnss_pos_sigma_m, config.gnss_vel_sigma_mps
        cn0 = config.gnss_cn0_nominal + noise_cn0
        sats = config.gnss_sats_nominal + sat_jitter
        gnss_ok = True
        off_e = off_n = off_ve = off_vn = 0.0
        for event in active:
            if event.kind == "gnss_dropout":
                gnss_ok = False
            elif event.kind == "jamming":
                ev = rng["events"]
                cn0 -= event.param("cn0_drop", 18.0) + abs(ev.gauss(0.0, 2.0))
                sats -= ev.randint(3, 5)
                pos_sigma = event.param("pos_sigma_m", 12.0)
                vel_sigma = event.param("vel_sigma_mps", 0.6)
                if ev.random() < event.param("fix_loss_prob", 0.25):
                    gnss_ok = False
            elif event.kind == "spoof_ramp":
                rate = event.param("rate_east_mps", 0.5)
                off_e += rate * (t - event.start + 1) * dt
                off_ve += rate
            elif event.kind == "spoof_step":
                off_n += event.param("offset_north_m", 120.0)
        gnss_e = tr.east + off_e + pos_sigma * noise_pos[0]
        gnss_n = tr.north + off_n + pos_sigma * noise_pos[1]
        gnss_ve = true_ve + off_ve + vel_sigma * noise_vel[0]
        gnss_vn = true_vn + off_vn + vel_sigma * noise_vel[1]

        # ---- aiding sensors
        bias = profile.gyro_bias_dps + sum(e.param("extra_bias_dps", 0.0) for e in active if e.kind == "gyro_drift")
        gyro_ok = "gyro_dropout" not in kinds
        log_ok = "speed_log_dropout" not in kinds
        compass_ok = "compass_dropout" not in kinds
        if t > 0:
            if gyro_ok:
                last_rate = tr.yaw_rate_dps + bias + gyro_noise
            heading = (heading + last_rate * dt) % 360.0
        if log_ok:
            last_speed = tr.speed + log_noise
        compass = (tr.heading_deg + compass_noise) % 360.0

        # ---- dead-reckoning prediction and GNSS update
        hrad = math.radians(heading)
        dr_ve, dr_vn = last_speed * math.sin(hrad), last_speed * math.cos(hrad)
        pred_e, pred_n = (est_e + dr_ve * dt, est_n + dr_vn * dt) if t > 0 else (est_e, est_n)
        rejected = False
        innov_e = innov_n = 0.0
        if gnss_ok:
            innov_e, innov_n = gnss_e - pred_e, gnss_n - pred_n
            if math.hypot(innov_e, innov_n) > config.innovation_gate_m:
                rejected = True
                rejections += 1
                if rejections >= config.reset_after_rejections:
                    est_e, est_n, rejections = gnss_e, gnss_n, 0
                else:
                    est_e, est_n = pred_e, pred_n
            else:
                rejections = 0
                est_e = pred_e + config.filter_gain * innov_e
                est_n = pred_n + config.filter_gain * innov_n
        else:
            est_e, est_n = pred_e, pred_n

        # ---- residual channels (missing measurement -> residual 0.0 AND availability flag 1.0)
        raw = [0.0] * len(CHANNELS)
        if gnss_ok:
            raw[CHANNEL_INDEX["gnss_pos_innov_e"]] = innov_e
            raw[CHANNEL_INDEX["gnss_pos_innov_n"]] = innov_n
            raw[CHANNEL_INDEX["gnss_vel_innov_e"]] = gnss_ve - dr_ve
            raw[CHANNEL_INDEX["gnss_vel_innov_n"]] = gnss_vn - dr_vn
            raw[CHANNEL_INDEX["gnss_cn0_deficit"]] = config.gnss_cn0_nominal - cn0
            raw[CHANNEL_INDEX["gnss_sat_deficit"]] = float(config.gnss_sats_nominal - sats)
            raw[CHANNEL_INDEX["gnss_rejected"]] = 1.0 if rejected else 0.0
            if log_ok:
                raw[CHANNEL_INDEX["speed_resid"]] = last_speed - math.hypot(gnss_ve, gnss_vn)
        else:
            raw[CHANNEL_INDEX["gnss_unavailable"]] = 1.0
        if compass_ok:
            raw[CHANNEL_INDEX["heading_resid"]] = wrap180(heading - compass)
        raw[CHANNEL_INDEX["gyro_unavailable"]] = 0.0 if gyro_ok else 1.0
        raw[CHANNEL_INDEX["speed_log_unavailable"]] = 0.0 if log_ok else 1.0
        raw[CHANNEL_INDEX["compass_unavailable"]] = 0.0 if compass_ok else 1.0
        residuals = tuple(round(value / channel.scale, 9) + 0.0 for value, channel in zip(raw, CHANNELS, strict=True))

        records.append(
            EpochRecord(
                t=t,
                truth_east=tr.east,
                truth_north=tr.north,
                est_east=est_e,
                est_north=est_n,
                nav_error_m=math.hypot(est_e - tr.east, est_n - tr.north),
                degraded=bool(kinds),
                events=kinds,
                raw=tuple(raw),
                residuals=residuals,
            )
        )
    return records
