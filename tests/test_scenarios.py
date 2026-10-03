# Copyright (C) 2026 Jean-François Brisson / Spark AI NLP. SPDX-License-Identifier: AGPL-3.0-only
import math
from dataclasses import replace

import pytest

from pilottrace_nav.integrations.oes_adapter import CHANNEL_INDEX
from pilottrace_nav.scenarios import SCENARIOS, WAYPOINTS, Event, SimConfig, simulate, truth_trajectory
from pilottrace_nav.scenarios.route import wrap180


def test_truth_is_deterministic_and_plausible():
    a, b = truth_trajectory(900), truth_trajectory(900)
    assert a == b
    assert (a[0].east, a[0].north) == WAYPOINTS[0]
    for prev, cur in zip(a, a[1:], strict=False):
        step = math.hypot(cur.east - prev.east, cur.north - prev.north)
        assert step == pytest.approx(cur.speed, rel=1e-12)
        assert abs(cur.yaw_rate_dps) <= 1.0 + 1e-12
        assert 0.0 <= cur.heading_deg < 360.0


def test_truth_rejects_bad_arguments():
    with pytest.raises(ValueError):
        truth_trajectory(0)
    with pytest.raises(ValueError):
        truth_trajectory(10, dt=0.0)


def test_route_reaches_later_waypoints():
    traj = truth_trajectory(1400)
    closest = min(math.hypot(t.east - WAYPOINTS[3][0], t.north - WAYPOINTS[3][1]) for t in traj)
    assert closest < 150.0


def test_wrap180():
    assert wrap180(190.0) == -170.0
    assert wrap180(-190.0) == 170.0
    assert wrap180(180.0) == -180.0


def test_event_validation():
    with pytest.raises(ValueError):
        Event("meteor", 0, 10)
    with pytest.raises(ValueError):
        Event("gnss_dropout", 10, 10)
    with pytest.raises(ValueError):
        Event("gnss_dropout", -1, 10)
    event = Event("spoof_step", 5, 8, (("offset_north_m", 3.0),))
    assert event.param("offset_north_m", 0.0) == 3.0 and event.param("missing", 7.0) == 7.0
    assert event.active(5) and event.active(7) and not event.active(8)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"seed": 4.2},
        {"window": 16},
        {"epochs": 32},
        {"inertial_profile": "warp_drive"},
        {"filter_gain": 0.0},
        {"dt": 0.0},
        {"tau": -1.0},
    ],
)
def test_config_validation(kwargs):
    with pytest.raises(ValueError):
        SimConfig(**kwargs)


def test_simulation_is_deterministic():
    spec = SCENARIOS["jamming"]
    assert simulate(spec) == simulate(spec)
    other_seed = simulate(spec, SimConfig(seed=43))
    assert other_seed[100].residuals != simulate(spec)[100].residuals


def test_nominal_noise_is_shared_outside_events():
    nominal = simulate(SCENARIOS["nominal"])
    jamming = simulate(SCENARIOS["jamming"])
    assert [r.residuals for r in nominal[:300]] == [r.residuals for r in jamming[:300]]
    assert nominal[300].residuals != jamming[300].residuals


def test_labels_follow_events():
    records = simulate(SCENARIOS["total_dropout"])
    assert [r.t for r in records if r.degraded] == list(range(300, 600))
    assert records[450].events == ("gnss_dropout",)
    assert not any(r.degraded for r in simulate(SCENARIOS["nominal"]))


def test_missing_measurements_are_flagged_not_silent():
    records = simulate(SCENARIOS["total_dropout"])
    r = records[450]
    assert r.residuals[CHANNEL_INDEX["gnss_unavailable"]] == 1.0
    for name in ("gnss_pos_innov_e", "gnss_pos_innov_n", "gnss_vel_innov_e", "gnss_cn0_deficit", "speed_resid"):
        assert r.residuals[CHANNEL_INDEX[name]] == 0.0
    dropouts = simulate(SCENARIOS["sensor_dropouts"])
    assert dropouts[260].residuals[CHANNEL_INDEX["gyro_unavailable"]] == 1.0
    assert dropouts[470].residuals[CHANNEL_INDEX["speed_log_unavailable"]] == 1.0
    assert dropouts[470].residuals[CHANNEL_INDEX["speed_resid"]] == 0.0
    assert dropouts[660].residuals[CHANNEL_INDEX["compass_unavailable"]] == 1.0
    assert dropouts[660].residuals[CHANNEL_INDEX["heading_resid"]] == 0.0


def test_reserved_channels_are_zero_and_values_finite():
    for spec in SCENARIOS.values():
        for r in simulate(spec):
            assert r.residuals[13:] == (0.0, 0.0, 0.0)
            assert all(math.isfinite(v) for v in r.residuals)
            assert len(r.residuals) == len(r.raw) == 16


def test_spoof_step_and_gate_reset():
    records = simulate(SCENARIOS["spoof_step"])
    assert records[400].residuals[CHANNEL_INDEX["gnss_rejected"]] == 1.0
    # after 10 consecutive rejections the estimate is re-initialised to the (spoofed) fix
    assert records[420].nav_error_m > 100.0


def test_dropout_error_grows_and_depends_on_inertial_profile():
    spec = SCENARIOS["total_dropout"]
    classical = simulate(spec)
    low_drift = simulate(spec, replace(SimConfig(), inertial_profile="hypothetical_low_drift"))
    assert classical[599].nav_error_m > classical[300].nav_error_m
    assert low_drift[599].nav_error_m < classical[599].nav_error_m
