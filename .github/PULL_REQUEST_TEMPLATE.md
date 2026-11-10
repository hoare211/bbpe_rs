## What changed

<!-- One or two sentences. Link the issue if there is one. -->

## Why

<!-- The problem this solves. Include benchmarks if it is a performance change. -->

## Checklist

- [ ] `cargo fmt --all --check` passes
- [ ] `cargo clippy --locked --no-default-features --all-targets -- -D warnings` passes
- [ ] `cargo test --locked --no-default-features` passes
- [ ] Python tests pass if the binding layer changed
- [ ] `CHANGELOG.md` updated for user-visible changes
- [ ] No new `unsafe`, no secrets, no local absolute paths

## Notes for reviewers

<!-- Anything worth calling out: trade-offs, follow-ups, deliberately out of scope. -->
