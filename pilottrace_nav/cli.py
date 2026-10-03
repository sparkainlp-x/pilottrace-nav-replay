# Copyright (C) 2026 Jean-François Brisson / Spark AI NLP. SPDX-License-Identifier: AGPL-3.0-only
"""Command line: ``pilottrace-nav-replay {list,run,verify}`` (or ``python -m pilottrace_nav``).

Exit codes: 0 ok; 1 I/O error; 2 invalid arguments or fail-closed input; 3 pilottrace missing or not the
pinned version; 4 ``verify`` found a hash mismatch.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from . import __version__
from ._deps import DependencyError
from .provenance.manifest import verify_manifest
from .replay import run_replay
from .scenarios import INERTIAL_PROFILES, SCENARIOS, SimConfig

EXIT_OK, EXIT_IO, EXIT_USAGE, EXIT_DEPENDENCY, EXIT_MISMATCH = 0, 1, 2, 3, 4


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pilottrace-nav-replay",
        description="Deterministic SYNTHETIC GNSS-denied maritime navigation replay, scored through pilottrace.",
        epilog="SYNTHETIC data only; no physical quantum sensor; not a navigation or safety system.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="list scenarios and inertial profiles")
    run = sub.add_parser("run", help="run the replay pack and write evidence + provenance")
    run.add_argument("--out-dir", default="replay-output", help="output directory (default: replay-output)")
    run.add_argument("--scenarios", default="", help="comma-separated subset (default: all)")
    run.add_argument("--inertial-profile", default="classical_dr", choices=sorted(INERTIAL_PROFILES))
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--tau", type=float, default=0.50, help="latch threshold (default 0.50)")
    run.add_argument("--quiet", action="store_true")
    verify = sub.add_parser("verify", help="re-hash the outputs listed in a provenance manifest")
    verify.add_argument("manifest", help="path to provenance_manifest.json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "list":
            for spec in SCENARIOS.values():
                print(f"{spec.name:16s} {spec.title}: {spec.description}")
            for profile in INERTIAL_PROFILES.values():
                print(f"profile {profile.name}: {profile.note}")
            return EXIT_OK
        if args.command == "verify":
            problems = verify_manifest(args.manifest)
            for problem in problems:
                print(f"mismatch: {problem}", file=sys.stderr)
            if not problems:
                print("verify: all outputs match the manifest")
            return EXIT_MISMATCH if problems else EXIT_OK
        names = [n.strip() for n in args.scenarios.split(",") if n.strip()] or None
        config = SimConfig(seed=args.seed, tau=args.tau, inertial_profile=args.inertial_profile)
        evidence = run_replay(args.out_dir, names, config)
        if not args.quiet:
            for s in evidence["scenarios"]:
                print(
                    f"{s['name']:16s} latched {s['frames_latched']:4d}/{s['frames']} frames, "
                    f"false-alarm frames {s['false_alarm_frames']:3d}, max nav error {s['nav_error_m']['max']:.1f} m "
                    "(SYNTHETIC)"
                )
            print(f"wrote {args.out_dir}/evidence_report.md and provenance_manifest.json")
        return EXIT_OK
    except DependencyError as exc:
        print(f"pilottrace-nav-replay: {exc}", file=sys.stderr)
        return EXIT_DEPENDENCY
    except (ValueError, KeyError) as exc:
        print(f"pilottrace-nav-replay: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except OSError as exc:
        print(f"pilottrace-nav-replay: {exc}", file=sys.stderr)
        return EXIT_IO
