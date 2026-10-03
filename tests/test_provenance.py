# Copyright (C) 2026 Jean-François Brisson / Spark AI NLP. SPDX-License-Identifier: AGPL-3.0-only
import json
import shutil

import pytest

from pilottrace_nav.provenance import manifest as mf


def test_canonical_json_is_stable_and_rejects_nan():
    assert mf.canonical_json({"b": 1, "a": [1.5, "é"]}) == '{"a":[1.5,"é"],"b":1}'.encode()
    with pytest.raises(ValueError):
        mf.canonical_json({"x": float("nan")})


def test_source_hashes_cover_the_package():
    hashes = mf.source_hashes()
    assert "integrations/oes_adapter.py" in hashes and "replay.py" in hashes
    assert all(len(v) == 64 for v in hashes.values())
    assert mf.tree_hash(hashes) == mf.tree_hash(dict(reversed(list(hashes.items()))))


def test_build_manifest_is_deterministic():
    kwargs = dict(config={"seed": 42}, scenarios={"s": 1}, outputs={"a.txt": b"x"}, pilottrace_hashes={"api.py": "0"})
    first, second = mf.build_manifest(**kwargs), mf.build_manifest(**kwargs)
    assert mf.pretty_json(first) == mf.pretty_json(second)
    assert first["outputs"]["a.txt"] == {"sha256": mf.sha256_bytes(b"x"), "bytes": 1}
    assert first["dependency"]["modified"] is False


@pytest.fixture()
def copied_run(full_run, tmp_path):
    out, _ = full_run
    target = tmp_path / "run"
    shutil.copytree(out, target)
    return target


def test_verify_ok_and_tamper_detection(copied_run):
    path = copied_run / "provenance_manifest.json"
    assert mf.verify_manifest(path) == []
    (copied_run / "evidence_report.md").write_text("tampered")
    (copied_run / "traces" / "nominal_trace.json").unlink()
    problems = mf.verify_manifest(path)
    assert any("evidence_report.md" in p for p in problems)
    assert any("nominal_trace.json: missing" in p for p in problems)


def test_verify_rejects_escaping_paths_schema_and_code_changes(copied_run, monkeypatch):
    path = copied_run / "provenance_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["outputs"]["../outside.txt"] = {"sha256": "0" * 64, "bytes": 0}
    manifest["manifest_schema_version"] = 99
    path.write_text(json.dumps(manifest))
    problems = mf.verify_manifest(path)
    assert any("escapes" in p for p in problems)
    assert any("schema" in p for p in problems)
    monkeypatch.setattr(mf, "source_hashes", lambda: {"changed.py": "1" * 64})
    assert any(p.startswith("code:") for p in mf.verify_manifest(path))
