# Copyright (C) 2026 Jean-François Brisson / Spark AI NLP. SPDX-License-Identifier: AGPL-3.0-only
"""Locate and verify the pinned ``pilottrace`` dependency (fail closed on mismatch).

``sparkainlp-x/pilottrace`` is used unmodified at a pinned commit. It is not a pip package, so it is
imported from the import path (for example ``PYTHONPATH=.deps/pilottrace``) or from ``.deps/pilottrace``
next to this repository. The source files that define scoring, validation and reporting are
hash-checked against the pin before use, so a silently changed upstream cannot alter the evidence.
"""

from __future__ import annotations

import hashlib
import importlib
import sys
from pathlib import Path
from types import ModuleType

PILOTTRACE_REPOSITORY = "https://github.com/sparkainlp-x/pilottrace"
PILOTTRACE_COMMIT = "d402c09cfdd3282262efc3cecc290905623ad98d"
PILOTTRACE_VERSION = "0.1.0"
PILOTTRACE_FILE_SHA256 = {
    "__init__.py": "761b07f5d9aca53a3e3b78383e343db7e306acbc57b51131c67209302bf67d53",
    "api.py": "34b1975bc04155e901b7c5ac36ad4d4e5bd5b2a0d951dda8a5255b687de6a9f6",
    "scoring.py": "b853aaa4283ea56d911032be17196c8540a2e5ece7104d899da6520dbe8a2cd2",
    "validation.py": "a661ca9e8a1a72931c1e6d7737ee7f9be6870d7e119c20acf017f72162071f8d",
}
_LOCAL_CHECKOUT = Path(__file__).resolve().parents[1] / ".deps" / "pilottrace"


class DependencyError(RuntimeError):
    """``pilottrace`` is missing or does not match the pinned source hashes."""


def pilottrace_file_hashes(package_dir: Path) -> dict[str, str]:
    hashes = {}
    for name in sorted(PILOTTRACE_FILE_SHA256):
        path = package_dir / name
        try:
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as exc:
            raise DependencyError(f"cannot read pilottrace source file {path}: {exc}") from exc
    return hashes


def verify_pilottrace(module: ModuleType) -> dict[str, str]:
    """Return the verified file hashes, or raise :class:`DependencyError` (fail closed)."""
    package_dir = Path(module.__file__ or "").resolve().parent
    hashes = pilottrace_file_hashes(package_dir)
    mismatched = sorted(name for name, digest in hashes.items() if digest != PILOTTRACE_FILE_SHA256[name])
    if mismatched:
        raise DependencyError(
            f"pilottrace at {package_dir} does not match the pinned commit {PILOTTRACE_COMMIT[:12]} "
            f"(changed: {', '.join(mismatched)}); refusing to score"
        )
    if getattr(module, "__version__", None) != PILOTTRACE_VERSION:
        raise DependencyError(f"pilottrace version {getattr(module, '__version__', None)!r} != {PILOTTRACE_VERSION}")
    return hashes


def load_pilottrace() -> tuple[ModuleType, dict[str, str]]:
    """Import ``pilottrace`` (falling back to ``.deps/pilottrace``) and verify it against the pin."""
    try:
        module = importlib.import_module("pilottrace")
    except ImportError:
        if not (_LOCAL_CHECKOUT / "pilottrace" / "__init__.py").is_file():
            raise DependencyError(
                "pilottrace is not importable. Run: git clone "
                f"{PILOTTRACE_REPOSITORY} .deps/pilottrace && git -C .deps/pilottrace checkout {PILOTTRACE_COMMIT}"
            ) from None
        sys.path.insert(0, str(_LOCAL_CHECKOUT))
        module = importlib.import_module("pilottrace")
    return module, verify_pilottrace(module)
