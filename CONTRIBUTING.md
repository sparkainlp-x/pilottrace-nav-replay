# Contributing

Thank you for helping. Please read the [Code of Conduct](CODE_OF_CONDUCT.md) first. Report security issues privately as described in [SECURITY.md](SECURITY.md), not as public issues.

## Ground rules

- **Synthetic data only.** Never commit real navigation recordings, vessel tracks, receiver logs, or private, personal or customer data, not even "just for a test".
- **No physical-sensor claims.** This is a software replay. Do not present its numbers as evidence about a physical sensor, receiver, vessel or navigation system. The low-drift inertial profile is a hypothetical parameter set; no sensor benefit, calibration or fitness for operational use is established.
- **Tag every number** as SYNTHETIC, REPORTED (measured on named, authorised real data, with provenance), TARGET (a goal or assumption) or UNRUN. Report misses and negative results as clearly as successes.
- **Fail closed.** Never clip, impute or silently drop non-finite or malformed values in the adapter. Missing measurements must stay visible through an availability channel.
- **Do not modify pilottrace.** It is used unmodified at a pinned commit. Changing the pin requires updating the hashes in `pilottrace_nav/_deps.py` and explaining why.
- **Licence.** The project is AGPL-3.0-only (see [LICENSE](LICENSE) and [COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md)). By contributing, you agree that your contribution is licensed under the same terms.

## Before you open a pull request

1. Run `ruff check .` and `python -m pytest --cov` (coverage of at least 95%; CI runs Python 3.10–3.13).
2. If you change scenarios, the estimator, the mapping or the report, run `python examples/navigation_replay.py` and commit the regenerated `examples/output/*.json|md`. Explain in the PR why the numbers changed.
3. Update the README and `CHANGELOG.md` for any user-visible change.
