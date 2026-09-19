# Security Policy

## Supported Versions

This is a research/backtesting repository. Only the latest release on the
`main` branch is supported.

| Version | Supported          |
| ------- | ------------------ |
| main    | :white_check_mark: |
| < 0.1.0 | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability — for example in data fetching,
environment handling, or anything that could leak credentials when the
scripts are deployed — please report it **privately**:

1. Open a [GitHub Security Advisory](https://github.com/tytaiii/stable-anfen-strategy/security/advisories/new)
   (visible only to maintainers until published).
2. Describe the affected component, the impact, and steps to reproduce.
3. Please do **not** open a public issue for security problems.

You can expect an initial response within 7 days and, if confirmed, a fix
release as soon as practical.

## Security Notes for Deployers

- `stable_anfen_strategy.py` and the bundled scripts never contain API keys.
  Keep any credentials in environment variables (see the strategy's
  `ANFEN_DATA_DIR` pattern as an example).
- This repository contains **no live-trading executor**. It performs
  backtesting only; do not wire the signal output to real orders without
  independent risk controls.
- Binance/Deribit data is fetched over HTTPS via public endpoints. No
  authentication is involved.
