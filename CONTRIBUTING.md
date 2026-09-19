# Contributing to Stable-Anfen Strategy

Thanks for your interest in contributing! This guide covers how to set up,
test, and submit changes.

## Development Setup

Requirements: Python 3.10+, and the packages in `requirements.txt`.

```bash
git clone https://github.com/tytaiii/stable-anfen-strategy.git
cd stable-anfen-strategy
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install pytest               # test dependency
```

## Running Tests

```bash
pytest -v
```

The test suite covers feature engineering, the trading loop, the scout gun,
the momentum rotation engine, and statistics — all with small synthetic
datasets, so the full suite runs in seconds.

## Making Changes

1. Fork the repository and create a topic branch:
   `git checkout -b fix/your-change`
2. Make your change. Follow the existing style: 4-space indent, English
   comments, minimal dependencies.
3. If you change strategy logic, add or update a test that pins the behavior.
   Run `pytest -v` locally before submitting.
4. Keep the [changelog](./CHANGELOG.md) up to date for user-visible changes.
5. Open a pull request. Use the PR template checklist and describe what
   problem your change solves and how it was verified.

## Pull Request Checklist

- [ ] Tests added/updated and passing (`pytest -v`)
- [ ] No new dependencies without justification
- [ ] CHANGELOG.md updated for user-visible changes
- [ ] Comments/documentation updated where behavior changed

## Code of Conduct

All participants are expected to follow the
[Code of Conduct](./CODE_OF_CONDUCT.md).
