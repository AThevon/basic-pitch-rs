# Security policy

## Supported versions

basic-pitch-rs is in its 0.x series. Security fixes land on the latest released
version only; please upgrade before reporting.

| Version          | Supported |
| ---------------- | --------- |
| latest 0.x       | yes       |
| older releases   | no        |

## Reporting a vulnerability

Please do not report security vulnerabilities through public issues, discussions
or pull requests.

Report them privately with GitHub's private vulnerability reporting:
[open a draft security advisory](https://github.com/AThevon/basic-pitch-rs/security/advisories/new).
Only the maintainer can see it.

Please include:

- the affected version or commit,
- the kind of issue (for example a panic, an out-of-bounds read or unbounded
  memory use on a crafted audio, MIDI or model file),
- the steps or the input file that reproduce it,
- the impact you expect.

You can expect an acknowledgement within 7 days. Once the issue is confirmed, a
fix and a coordinated disclosure date are agreed with you, and the advisory
credits you unless you ask otherwise.

## Scope

In scope: the `basic-pitch-rs` library and the `basic-pitch-rs` command-line
tool, including how they decode untrusted audio files and load model files.

Out of scope, to be reported upstream:

- vulnerabilities in dependencies (tract, ONNX Runtime, symphonia, libsoxr...),
  unless basic-pitch-rs uses them in an unsafe way;
- the behavior of the Basic Pitch model itself, which belongs to
  [spotify/basic-pitch](https://github.com/spotify/basic-pitch).
