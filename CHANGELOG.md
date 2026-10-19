# Changelog

All notable changes are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.1.0] - unreleased (date set when tagged)

### Changed
- Retitled from "PilotTrace Quantum Navigation Replay Pack" to "PilotTrace Navigation Replay Pack"; the repository name `pilottrace-nav-replay` is unchanged. `examples/quantum_navigation_replay.py` is now `examples/navigation_replay.py`.
- Boundary text no longer uses quantum wording: "No physical sensor or other hardware is involved" and "No sensor benefit, calibration or fitness for operational use is established by this replay." The committed evidence in `examples/output/` was regenerated; only wording and hashes changed, no number.
- README badges removed; README gains a Pages link, Limitations, How to cite (DOI placeholder) and a License section with the copyright line.
- .zenodo.json: community `spark-ai-nlp`, full affiliation, `references` the pilottrace concept DOI 10.5281/zenodo.23187278 (also in CITATION.cff).
- Tests treat warnings as errors; CI runs on every branch.

### Added
- Static GitHub Pages view (`scripts/build_site.py`, `.github/workflows/pages.yml`): every scenario replayed at build time through the pinned pilottrace, with navigation error, labelled event windows and latched frames as inline SVG. No JavaScript, no external requests. 5 builder tests (97 tests total).
- Seven deterministic SYNTHETIC maritime scenarios (seed 42): nominal GNSS, total GNSS dropout, jamming-like interference, slow and abrupt spoofing, gyro drift, and aiding-sensor dropouts, with labelled degradation events.
- A simple documented estimator (dead reckoning plus a gated constant-gain GNSS update) and 16 residual channels.
- `integrations/oes_adapter.py`: an explicit 16 × 32 mapping, the classical `R = max|x − x_ref|`, the latch `S = 0.45·Peak + 0.35·RMS + 0.20·MeanAbs ≥ τ` (default 0.50), and fail-closed validation.
- Scoring through the unmodified, hash-pinned `pilottrace` package, cross-checked frame by frame against the adapter.
- `provenance/manifest.py` (hashes of config, code, dependency, seed and outputs; `verify`) and `provenance/evidence_report.py` (JSON and Markdown).
- CLI (`pilottrace-nav-replay list|run|verify`), the example replay script, 92 tests, CI on Python 3.10–3.13, Dependabot, and the community files.
