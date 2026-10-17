## Summary

<!-- What does this change, and why? Link the related issue: "Closes #123". -->

## Type of change

- [ ] Bug fix
- [ ] New feature (opt-in, default output unchanged)
- [ ] Change to the numerical output (see below)
- [ ] Documentation
- [ ] Build, CI or tooling

## Parity

<!--
The default pipeline must stay identical to spotify/basic-pitch v0.4.0 (and main for
the trim mode). If this pull request changes any number that reaches a note, a
pitch bend or a MIDI message, explain why and how it is covered by a fixture.
-->

- [ ] `cargo test --manifest-path crates/parity/Cargo.toml --release` passes.
- [ ] This change does not alter the default numerical output, or it adds or updates a fixture generated with `reference/generate_fixtures.py` and explains why.

## Checklist

- [ ] `cargo fmt --all --check` and `cargo clippy --workspace --all-targets --all-features -- -D warnings` are clean.
- [ ] Tests cover the change.
- [ ] Public items are documented, and `CHANGELOG.md` has an entry under "Unreleased" for user-facing changes.
- [ ] Files ported from the Python code keep their license header (see CONTRIBUTING.md).
