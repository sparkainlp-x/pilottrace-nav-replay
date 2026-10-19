## What and why

## Checklist
- [ ] `ruff check .` and `python -m pytest --cov` pass (coverage ≥ 95%)
- [ ] If outputs changed: regenerated `examples/output/` with `python examples/navigation_replay.py` and explained the change
- [ ] Every new number is tagged SYNTHETIC / REPORTED / TARGET / UNRUN
- [ ] No physical-sensor, sensor-benefit, calibration, operational, medical or readiness claims
- [ ] Fail-closed behaviour preserved (no clipping or imputation in the adapter)
- [ ] pilottrace not modified; README and CHANGELOG updated
