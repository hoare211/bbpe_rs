# Contributing

Thanks for considering a contribution. This document covers the local setup and
the checks a pull request must pass.

## Prerequisites

- Rust 1.85 or newer (`rustup toolchain install stable`)
- Python 3.8+ and `maturin` for the Python bindings
- `pytest` for the Python test suite

## Build and test

```bash
# CLI only (no pyo3, fastest loop)
cargo build --release --no-default-features

# Rust tests, CLI feature set
cargo test --locked --no-default-features

# Formatting and lints (both are enforced in CI)
cargo fmt --all --check
cargo clippy --locked --no-default-features --all-targets -- -D warnings
```

Python bindings:

```bash
python -m venv .venv
source .venv/bin/activate
pip install maturin pytest jinja2
maturin develop --release
python -m pytest tests/ -q
```

`cargo test` without `--no-default-features` compiles the pyo3 binding layer and
therefore needs the Python development headers (`libpython3.x-dev` on Debian or
Ubuntu). CI covers the binding layer through the wheel build instead.

## Making changes

- Keep changes focused; unrelated refactors make review harder.
- Add or update tests for behaviour changes. Rust integration tests live in
  `tests/`, Python tests in `tests/test_*.py`.
- Do not introduce `unsafe` code in `src/`.
- User-facing behaviour changes need a `CHANGELOG.md` entry under
  `[Unreleased]`.
- Document model-format or CLI contract changes in `docs/使用说明.md`.

## Commit messages

Use conventional-commit prefixes, matching the existing history:

```text
feat(train): add --sample-seed
fix(encode): reject empty special tokens
docs: clarify held-out acceptance
test(release): cover version consistency
```

Keep the subject under ~72 characters, in the imperative mood.

## Pull request checklist

- [ ] `cargo fmt --all --check` passes
- [ ] `cargo clippy --locked --no-default-features --all-targets -- -D warnings` passes
- [ ] `cargo test --locked --no-default-features` passes
- [ ] Python tests pass if the binding layer changed
- [ ] `CHANGELOG.md` updated for user-visible changes
- [ ] No new `unsafe`, no secrets, no local absolute paths in committed files

## License

By submitting a pull request you agree that your contribution is dual licensed
under MIT and Apache-2.0, as described in the README.
