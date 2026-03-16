## Setup
Dev setup (linting & hooks)
```bash
pip install -e ".[dev]"
pre-commit install  # sets up git hooks
# on demand: pre-commit run --all-files
```

## Tests

- Run the main suite: `python -m pytest tests`
- Run API-level tests only: `python -m pytest tests/api`

## Releases

- Release process guide: `docs/releasing.md`
- Create and push a release tag: `scripts/release.sh 0.2.0 --push`
- Release versions come from Git tags via `hatch-vcs`
- PyPI publishing uses GitHub Actions Trusted Publishing
