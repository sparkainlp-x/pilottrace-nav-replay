# Changelog

All notable changes are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.1.0] - 2026-10-03 (not released; private repository)

### Added
- Seven deterministic SYNTHETIC maritime scenarios (seed 42): nominal GNSS, total GNSS dropout, jamming-like interference, slow and abrupt spoofing, gyro drift, and aiding-sensor dropouts, with labelled degradation events.
- A simple documented estimator (dead reckoning plus a gated constant-gain GNSS update) and 16 residual channels.
- `integrations/oes_adapter.py`: an explicit 16 × 32 mapping, the classical `R = max|x − x_ref|`, the latch `S = 0.45·Peak + 0.35·RMS + 0.20·MeanAbs ≥ τ` (default 0.50), and fail-closed validation.
- Scoring through the unmodified, hash-pinned `pilottrace` package, cross-checked frame by frame against the adapter.
- `provenance/manifest.py` (hashes of config, code, dependency, seed and outputs; `verify`) and `provenance/evidence_report.py` (JSON and Markdown).
- CLI (`pilottrace-nav-replay list|run|verify`), `examples/quantum_navigation_replay.py`, 92 tests, CI on Python 3.10–3.13, Dependabot, and the community files.
