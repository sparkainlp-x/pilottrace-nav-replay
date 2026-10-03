# Copyright (C) 2026 Jean-François Brisson / Spark AI NLP. SPDX-License-Identifier: AGPL-3.0-only
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from pilottrace_nav import cli
from pilottrace_nav._deps import DependencyError

ROOT = Path(__file__).resolve().parents[1]


def test_list(capsys):
    assert cli.main(["list"]) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "spoof_ramp" in out and "hypothetical_low_drift" in out


def test_run_subset_and_verify(tmp_path, capsys):
    assert cli.main(["run", "--out-dir", str(tmp_path), "--scenarios", "nominal,sensor_dropouts"]) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "SYNTHETIC" in out and "sensor_dropouts" in out
    manifest = tmp_path / "provenance_manifest.json"
    assert cli.main(["verify", str(manifest)]) == cli.EXIT_OK
    (tmp_path / "evidence_report.md").write_text("changed")
    assert cli.main(["verify", str(manifest)]) == cli.EXIT_MISMATCH
    assert "mismatch" in capsys.readouterr().err


def test_quiet_run_with_profile(tmp_path, capsys):
    args = ["run", "--out-dir", str(tmp_path), "--scenarios", "nominal", "--inertial-profile",
            "hypothetical_low_drift", "--quiet"]
    assert cli.main(args) == cli.EXIT_OK
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("args", [["--scenarios", "atlantis"], ["--tau", "-1"], ["--tau", "nan"]])
def test_usage_errors_exit_2(tmp_path, args):
    assert cli.main(["run", "--out-dir", str(tmp_path), *args]) == cli.EXIT_USAGE


def test_io_error_exit_1(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    assert cli.main(["run", "--out-dir", str(blocker), "--scenarios", "nominal"]) == cli.EXIT_IO


def test_dependency_error_exit_3(tmp_path, monkeypatch):
    def boom(*_args, **_kwargs):
        raise DependencyError("pilottrace missing")

    monkeypatch.setattr(cli, "run_replay", boom)
    assert cli.main(["run", "--out-dir", str(tmp_path)]) == cli.EXIT_DEPENDENCY


def test_verify_missing_manifest_exit_1(tmp_path):
    assert cli.main(["verify", str(tmp_path / "none.json")]) == cli.EXIT_IO


def test_version_and_module_entry(tmp_path):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
    result = subprocess.run([sys.executable, "-m", "pilottrace_nav", "list"], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0 and "nominal" in result.stdout


def test_example_script(tmp_path):
    script = ROOT / "examples" / "quantum_navigation_replay.py"
    result = subprocess.run([sys.executable, str(script), str(tmp_path)], capture_output=True, text=True, cwd=ROOT)
    assert result.returncode == 0, result.stderr
    assert "NOT latched during event" in result.stdout and "[SYNTHETIC]" in result.stdout
    for name in ("evidence_report.json", "evidence_report.md", "provenance_manifest.json"):
        assert (tmp_path / name).read_bytes() == (ROOT / "examples" / "output" / name).read_bytes()
    shutil.rmtree(tmp_path / "traces")
