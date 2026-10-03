# Security policy

PilotTrace Quantum Navigation Replay Pack is research software that runs on synthetic data only. It processes no personal data, opens no network connections at runtime, and reads and writes files only under the output directory you choose. It imports the `pilottrace` package from your import path or from `.deps/pilottrace`, and refuses to run if pilottrace's source files do not match the pinned SHA-256 hashes.

It is **not** a navigation, integrity-monitoring or safety system. Do not use it to make navigation decisions.

## Supported versions

Only the `main` branch receives fixes.

## Reporting a vulnerability

Please **do not open a public issue** for a security problem. Contact the maintainer, Jean-François Brisson, privately via the contact details on the GitHub profile (https://github.com/sparkainlp-x) or the website (https://sparkainlpx.xyz). Please include:

- the affected commit;
- steps to reproduce; and
- the impact you expect.

You should get an acknowledgement within 7 days.
