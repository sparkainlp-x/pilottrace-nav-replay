# Copyright (C) 2026 Jean-François Brisson / Spark AI NLP. SPDX-License-Identifier: AGPL-3.0-only
import importlib
import shutil
import types
from pathlib import Path

import pytest

from pilottrace_nav import _deps


def test_pinned_pilottrace_loads_and_verifies():
    module, hashes = _deps.load_pilottrace()
    assert module.__version__ == _deps.PILOTTRACE_VERSION
    assert hashes == _deps.PILOTTRACE_FILE_SHA256


def _copy_pilottrace(tmp_path: Path) -> Path:
    module, _ = _deps.load_pilottrace()
    src = Path(module.__file__).parent
    dst = tmp_path / "pilottrace"
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__"))
    return dst


def test_modified_pilottrace_is_refused(tmp_path):
    dst = _copy_pilottrace(tmp_path)
    (dst / "scoring.py").write_text((dst / "scoring.py").read_text().replace("0.45", "0.50"))
    fake = types.SimpleNamespace(__file__=str(dst / "__init__.py"), __version__=_deps.PILOTTRACE_VERSION)
    with pytest.raises(_deps.DependencyError, match="scoring.py"):
        _deps.verify_pilottrace(fake)


def test_wrong_version_and_missing_files_are_refused(tmp_path):
    dst = _copy_pilottrace(tmp_path)
    with pytest.raises(_deps.DependencyError, match="version"):
        _deps.verify_pilottrace(types.SimpleNamespace(__file__=str(dst / "__init__.py"), __version__="9.9"))
    (dst / "api.py").unlink()
    with pytest.raises(_deps.DependencyError, match="cannot read"):
        _deps.pilottrace_file_hashes(dst)


def test_missing_pilottrace_gives_instructions(monkeypatch, tmp_path):
    def fail(name):
        raise ImportError(name)

    monkeypatch.setattr(importlib, "import_module", fail)
    monkeypatch.setattr(_deps, "_LOCAL_CHECKOUT", tmp_path / "nowhere")
    with pytest.raises(_deps.DependencyError, match="git clone"):
        _deps.load_pilottrace()


def test_local_checkout_fallback(monkeypatch, tmp_path):
    checkout = tmp_path / "checkout"
    _copy_pilottrace(checkout)
    real = importlib.import_module
    calls = []

    def first_fails(name):
        calls.append(name)
        if len(calls) == 1:
            raise ImportError(name)
        return real(name)

    monkeypatch.setattr(importlib, "import_module", first_fails)
    monkeypatch.setattr(_deps, "_LOCAL_CHECKOUT", checkout)
    monkeypatch.setattr(_deps.sys, "path", list(_deps.sys.path))
    module, hashes = _deps.load_pilottrace()
    assert _deps.sys.path[0] == str(checkout)
    assert hashes == _deps.PILOTTRACE_FILE_SHA256 and module.__version__ == "0.1.0"
