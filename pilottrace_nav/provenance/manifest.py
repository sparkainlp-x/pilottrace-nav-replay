# Copyright (C) 2026 Jean-François Brisson / Spark AI NLP. SPDX-License-Identifier: AGPL-3.0-only
"""Deterministic provenance manifest: hashes of config, code, dependency pin, seed and outputs.

The manifest has no timestamps, hostnames or paths outside the output directory, so identical inputs
and code give byte-identical manifests. ``verify_manifest`` re-hashes every listed output file.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .. import __version__
from .._deps import PILOTTRACE_COMMIT, PILOTTRACE_REPOSITORY, PILOTTRACE_VERSION

MANIFEST_SCHEMA_VERSION = 1
PACKAGE_DIR = Path(__file__).resolve().parents[1]


def canonical_json(obj: Any) -> bytes:
    """Sorted keys, compact separators, UTF-8, NaN forbidden."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def pretty_json(obj: Any) -> bytes:
    return (json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def source_hashes(package_dir: Path = PACKAGE_DIR) -> dict[str, str]:
    """SHA-256 of every ``.py`` file in the package, keyed by POSIX path relative to the package."""
    return {
        path.relative_to(package_dir).as_posix(): sha256_bytes(path.read_bytes())
        for path in sorted(package_dir.rglob("*.py"))
        if "__pycache__" not in path.parts
    }


def tree_hash(hashes: Mapping[str, str]) -> str:
    return sha256_bytes(canonical_json(dict(sorted(hashes.items()))))


def build_manifest(
    *,
    config: Mapping[str, Any],
    scenarios: Mapping[str, Any],
    outputs: Mapping[str, bytes],
    pilottrace_hashes: Mapping[str, str],
) -> dict[str, Any]:
    code = source_hashes()
    return {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "evidence_tag": "SYNTHETIC",
        "project": "pilottrace-nav-replay",
        "code_version": __version__,
        "seed": config["seed"],
        "config_sha256": sha256_bytes(canonical_json(config)),
        "config": dict(config),
        "scenarios_sha256": sha256_bytes(canonical_json(scenarios)),
        "code": {"package_tree_sha256": tree_hash(code), "files": code},
        "dependency": {
            "name": "pilottrace",
            "repository": PILOTTRACE_REPOSITORY,
            "commit": PILOTTRACE_COMMIT,
            "version": PILOTTRACE_VERSION,
            "files_sha256": dict(sorted(pilottrace_hashes.items())),
            "modified": False,
        },
        "outputs": {name: {"sha256": sha256_bytes(data), "bytes": len(data)} for name, data in sorted(outputs.items())},
        "note": "Deterministic: no timestamps or host data. Regenerate with `python -m pilottrace_nav run`.",
    }


def verify_manifest(path: str | Path) -> list[str]:
    """Return a list of problems (empty = every output file matches its recorded hash and size)."""
    manifest_path = Path(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    problems = []
    if manifest.get("manifest_schema_version") != MANIFEST_SCHEMA_VERSION:
        problems.append("unsupported manifest_schema_version")
    for name, entry in sorted(manifest.get("outputs", {}).items()):
        target = (manifest_path.parent / name).resolve()
        if manifest_path.parent.resolve() not in target.parents:
            problems.append(f"{name}: path escapes the manifest directory")
            continue
        try:
            data = target.read_bytes()
        except OSError:
            problems.append(f"{name}: missing")
            continue
        if sha256_bytes(data) != entry.get("sha256") or len(data) != entry.get("bytes"):
            problems.append(f"{name}: hash or size mismatch")
    if manifest.get("code", {}).get("package_tree_sha256") != tree_hash(source_hashes()):
        problems.append("code: the installed package differs from the code that produced this manifest")
    return problems
