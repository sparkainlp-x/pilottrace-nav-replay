# Copyright (C) 2026 Jean-François Brisson / Spark AI NLP. SPDX-License-Identifier: AGPL-3.0-only
import json
from pathlib import Path

import pytest

from pilottrace_nav import replay
from pilottrace_nav.integrations.oes_adapter import FailClosedError
from pilottrace_nav.scenarios import SCENARIOS, SimConfig

ROOT = Path(__file__).resolve().parents[1]
COMMITTED = ROOT / "examples" / "output"


def by_name(evidence):
    return {s["name"]: s for s in evidence["scenarios"]}


def test_committed_evidence_is_reproduced_byte_for_byte(full_run):
    out, _ = full_run
    for name in ("evidence_report.json", "evidence_report.md", "provenance_manifest.json"):
        assert (out / name).read_bytes() == (COMMITTED / name).read_bytes(), name


def test_nominal_has_no_latches(full_run):
    nominal = by_name(full_run[1])["nominal"]
    assert nominal["frames"] == 869 and nominal["frames_latched"] == 0
    assert nominal["false_alarm_frames"] == 0
    assert nominal["pilottrace_descriptive_metrics"]["descriptive_specificity"] == 1.0


def test_every_scenario_cross_checks_against_pilottrace(full_run):
    for s in full_run[1]["scenarios"]:
        assert s["cross_check"]["latch_disagreements"] == 0
        assert s["cross_check"]["max_abs_score_difference"] <= replay.SCORE_TOLERANCE
        assert s["cross_check"]["frames_compared"] == s["frames"]


def test_event_outcomes_match_the_documented_synthetic_results(full_run):
    s = by_name(full_run[1])
    for name in ("total_dropout", "jamming", "spoof_step", "sensor_dropouts"):
        for event in s[name]["events"]:
            assert event["detected_during_event"] and event["detection_latency_s"] == 0.0, (name, event["kind"])
            assert event["expected_channel_latched_during_event"]
    drift = s["inertial_drift"]["events"][0]
    assert drift["detected_during_event"] and drift["detection_latency_s"] > 60
    # Documented negative result: the slow consistent spoof is NOT latched while it is active.
    ramp = s["spoof_ramp"]["events"][0]
    assert not ramp["detected_during_event"] and ramp["detection_latency_s"] is None
    assert ramp["latched_frames_after_end"] > 0
    assert ramp["nav_error_at_event_end_m"] > 100.0


def test_dropout_profile_comparison(full_run):
    rows = {r["inertial_profile"]: r for r in full_run[1]["dropout_profile_comparison"]}
    assert set(rows) == {"classical_dr", "hypothetical_low_drift"}
    low, classical = rows["hypothetical_low_drift"], rows["classical_dr"]
    assert low["nav_error_at_dropout_end_m"] < classical["nav_error_at_dropout_end_m"]
    assert "not a measurement" in rows["hypothetical_low_drift"]["note"]


def test_outputs_and_pilottrace_artifacts(full_run):
    out, _ = full_run
    manifest = json.loads((out / "provenance_manifest.json").read_text())
    assert manifest["seed"] == 42 and manifest["evidence_tag"] == "SYNTHETIC"
    assert len(manifest["outputs"]) == 3 * len(SCENARIOS) + 2
    report = json.loads((out / "traces" / "jamming_pilottrace_report.json").read_text())
    assert report["threshold"] == 0.5 and report["input_summary"]["frames_rejected"] == 0
    trace = json.loads((out / "traces" / "jamming_trace.json").read_text())
    assert len(trace["frames"]) == 869 and len(trace["frames"][0]["values"]) == 512


def test_evidence_text_states_boundaries(full_run):
    md = (full_run[0] / "evidence_report.md").read_text()
    assert "Every number in this report is SYNTHETIC" in md
    assert "No sensor benefit, calibration or fitness for operational use is established" in md
    assert "No physical sensor or other hardware is involved" in md
    assert "quantum" not in md.lower()


def test_run_subset_and_validation(tmp_path):
    evidence = replay.run_replay(tmp_path, ["nominal"], SimConfig(epochs=100))
    assert [s["name"] for s in evidence["scenarios"]] == ["nominal"]
    assert evidence["scenarios"][0]["frames"] == 69
    with pytest.raises(ValueError):
        replay.run_replay(tmp_path, ["nominal", "nominal"])
    with pytest.raises(ValueError):
        replay.run_replay(tmp_path, ["atlantis"])


def test_pilottrace_manifest_declarations_are_explicit():
    manifest = replay.pilottrace_manifest(SimConfig(), SCENARIOS["nominal"])
    assert manifest["threshold"] == 0.5 and manifest["schema_version"] == 1
    assert all(isinstance(v, str) and v for k, v in manifest.items() if k not in {"threshold", "schema_version"})


def _small_frames():
    return [(31, [0.0] * 512)]


def test_cross_check_fails_closed_on_rejection_and_disagreement():
    rejected = {"input_summary": {"frames_rejected": 1}, "frames": []}
    with pytest.raises(FailClosedError, match="rejected"):
        replay._cross_check(_small_frames(), rejected, 0.5)
    blocks = [{"score": 0.0} for _ in range(16)]
    disagree = {
        "input_summary": {"frames_rejected": 0},
        "frames": [{"blocks": blocks, "flagged_block_indices": [3]}],
    }
    with pytest.raises(FailClosedError, match="disagree"):
        replay._cross_check(_small_frames(), disagree, 0.5)
    blocks[0] = {"score": 1e-9}
    off = {"input_summary": {"frames_rejected": 0}, "frames": [{"blocks": blocks, "flagged_block_indices": []}]}
    with pytest.raises(FailClosedError, match="disagree"):
        replay._cross_check(_small_frames(), off, 0.5)


def test_events_beyond_a_shortened_run(tmp_path):
    evidence = replay.run_replay(tmp_path, ["inertial_drift", "total_dropout"], SimConfig(epochs=400))
    drift = by_name(evidence)["inertial_drift"]["events"][0]
    assert drift["within_run"] and drift["end_epoch_exclusive"] == 900 and drift["nav_error_at_event_end_m"] > 0
    assert evidence["dropout_profile_comparison"] == []  # 300-600 dropout does not fit in 400 epochs
    short = replay.run_replay(tmp_path, ["spoof_step"], SimConfig(epochs=200))
    assert short["scenarios"][0]["events"][0] == {
        "kind": "spoof_step", "start_epoch": 400, "end_epoch_exclusive": 550,
        "params": {"offset_north_m": 120.0}, "within_run": False,
    }
    assert "outside the run" in (tmp_path / "evidence_report.md").read_text()
