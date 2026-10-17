# basic-pitch-cli

[![crates.io](https://img.shields.io/crates/v/basic-pitch-cli?style=flat-square&labelColor=131518&color=6fcf7c)](https://crates.io/crates/basic-pitch-cli)
[![CI](https://img.shields.io/github/actions/workflow/status/AThevon/basic-pitch-rs/ci.yml?branch=main&style=flat-square&label=CI&labelColor=131518)](https://github.com/AThevon/basic-pitch-rs/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-6fcf7c?style=flat-square&labelColor=131518)](https://github.com/AThevon/basic-pitch-rs/blob/main/LICENSE)

The command-line tool of [basic-pitch-rs](https://github.com/AThevon/basic-pitch-rs), an
unofficial, parity-tested Rust port of Spotify's
[Basic Pitch](https://github.com/spotify/basic-pitch). It turns a recording of any
instrument or voice into notes with pitch bends, written as MIDI, JSON or CSV. One
binary, model included, no Python.

![The contour posteriorgram of a 10 s synthetic guitar recording in gray, with the 18 notes basic-pitch-rs detects and their pitch bends drawn in green over it.](https://raw.githubusercontent.com/AThevon/basic-pitch-rs/main/assets/hero.svg)

## Install

```sh
cargo install basic-pitch-cli      # builds the `basic-pitch-rs` command
cargo binstall basic-pitch-cli     # or downloads a prebuilt binary
```

The crate is named `basic-pitch-cli`; the command it installs is `basic-pitch-rs`.
Prebuilt binaries for Linux, macOS and Windows are attached to the
[GitHub releases](https://github.com/AThevon/basic-pitch-rs/releases).

## Use

```sh
basic-pitch-rs song.wav                     # writes song_basic_pitch.mid
basic-pitch-rs -o midi/ takes/*.flac        # several files into midi/
basic-pitch-rs riff.m4a --midi --json       # MIDI and JSON notes
basic-pitch-rs riff.m4a --json --output - | jq '.notes[].note_name'
basic-pitch-rs bass.wav --minimum-frequency 30 --maximum-frequency 400
```

Inputs: WAV, AIFF, CAF, FLAC, MP3, Ogg Vorbis, MP4/M4A/MOV (AAC, ALAC) and Matroska/WebM.
AAC files are decoded gapless: the encoder delay and padding are removed, as ffmpeg
does, so the notes are not shifted by the 2112 priming samples.

| Output | Flag | Content |
| --- | --- | --- |
| MIDI | `--midi` (default) | Written like Python Basic Pitch writes it (pretty_midi): the same notes give the same events |
| JSON | `--json` | Notes sorted by time with note names, velocities and pitch bends, plus the input, model and parameters. Described by a [JSON Schema](https://github.com/AThevon/basic-pitch-rs/blob/main/crates/basic-pitch-cli/schema/transcription.v1.schema.json) |
| CSV | `--csv` | The `--save-note-events` file of Python Basic Pitch, in the same format |
| NumPy | `--posteriorgrams` | The raw model outputs, as float32 arrays in a `.npz` archive |

The note options of the Python tool keep their names and defaults
(`--onset-threshold`, `--frame-threshold`, `--minimum-note-length`,
`--minimum-frequency`, `--maximum-frequency`, `--multiple-pitch-bends`, `--no-melodia`,
`--midi-tempo`), plus:

| Option | Effect |
| --- | --- |
| `--bend-offset -1` | Centers the pitch bends of in-tune notes on 0 ([spotify/basic-pitch#87](https://github.com/spotify/basic-pitch/issues/87)) |
| `--no-infer-onsets`, `--no-pitch-bends` | The remaining switches of the Python library |
| `--trim v0.4.0` | Reproduces the PyPI release, which drops the last frames of each file |
| `-j`, `--threads N` | Model threads (default: one per core); the results do not depend on it |
| `--backend ort` | ONNX Runtime instead of tract (build with `--features ort`) |
| `--resampler soxr` | libsoxr, like librosa (build with `--features soxr`, LGPL-2.1) |

Existing files are never overwritten without `--force`. Exit codes: 0 on success, 1 if
an input failed (the others are still processed), 2 for invalid arguments.

## Parity

From the same audio files, the tool gives the same 1650 notes as Python Basic Pitch on
the test suite (6 recordings, 8 parameter sets, 2 trim modes): frames, times, pitches and
pitch bends. Files that are not at 22 050 Hz are resampled first, with rubato by default
(within about 1e-6 of librosa from 44.1 and 48 kHz on voice and guitar) or with libsoxr
through the `soxr` feature (within 4.2e-7). Details are in the
[repository README](https://github.com/AThevon/basic-pitch-rs#parity-with-python).

## License

Apache-2.0, like Basic Pitch. Not affiliated with or endorsed by Spotify. The bundled
model is `nmp.onnx` from spotify/basic-pitch v0.4.0, unmodified; see
[NOTICE](https://github.com/AThevon/basic-pitch-rs/blob/main/NOTICE). The `soxr`
feature statically links libsoxr (LGPL-2.1-or-later) and is never part of the release
binaries.
