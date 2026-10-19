#!/usr/bin/env python3
# Copyright (C) 2026 Jean-François Brisson / Spark AI NLP. SPDX-License-Identifier: AGPL-3.0-only
"""Run the full SYNTHETIC replay pack and refresh the committed evidence in examples/output/.

Usage (from the repository root, with pilottrace available; see README):

    python examples/navigation_replay.py [OUT_DIR]

The large per-scenario traces and pilottrace reports go to OUT_DIR/traces/ (git-ignored); their hashes
are recorded in OUT_DIR/provenance_manifest.json. No physical sensor is involved, and no sensor benefit,
calibration or fitness for operational use is established.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pilottrace_nav.replay import run_replay  # noqa: E402


def main(argv: list[str]) -> int:
    out_dir = Path(argv[0]) if argv else ROOT / "examples" / "output"
    evidence = run_replay(out_dir)
    for s in evidence["scenarios"]:
        events = "; ".join(
            f"{e['kind']}: {'latched' if e['detected_during_event'] else 'NOT latched'} during event"
            for e in s["events"]
        ) or "no events"
        print(f"[SYNTHETIC] {s['name']}: {events}; max nav error {s['nav_error_m']['max']:.1f} m")
    print(f"evidence: {out_dir / 'evidence_report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
