# Python reference

This directory holds the official Python [Basic Pitch](https://github.com/spotify/basic-pitch)
pinned in two versions, and the scripts that generate the parity fixtures in
[`tests/fixtures`](../tests/fixtures). It is not part of the Rust build and is
never published. Day-to-day development does not need it: the fixtures are
committed.

| Path | Content |
| --- | --- |
| `pyproject.toml`, `uv.lock` | basic-pitch 0.4.0 from PyPI: the `v0.4.0` variant |
| `main/pyproject.toml`, `main/uv.lock` | spotify/basic-pitch at commit `fa5997a`: the `main` variant |
| `make_synth.py` | Writes the synthetic test signals (`synth`, `pluck`, `silence`, `short`) |
| `generate_fixtures.py` | Runs the reference pipeline step by step and saves every stage |
| `generate_adversarial.py` | Runs the note creation on synthetic posteriorgrams built to reach its corners |
| `bench_python.py` | Times the Python pipeline on the inputs of the Rust benchmarks |
| `render_readme_assets.py` | Draws `assets/hero.svg` and `assets/social-preview.png` from the output of the Rust command-line tool |

The format of the fixtures is described in the docstring of
`generate_fixtures.py` and in [tests/fixtures/README.md](../tests/fixtures/README.md).

## Usage

You need [uv](https://docs.astral.sh/uv/). It installs the right Python and the
locked dependencies by itself. From this directory:

```sh
uv run --locked python -I make_synth.py ../tests/fixtures/audio
uv run --locked python -I generate_fixtures.py ../tests/fixtures ../tests/fixtures/audio/*.wav
uv run --locked python -I generate_adversarial.py ../tests/fixtures/adversarial
cd main
uv run --locked python -I ../generate_fixtures.py ../../tests/fixtures ../../tests/fixtures/audio/*.wav --variant-only
```

`--variant-only` skips what does not depend on the variant (decoded audio, model
windows, raw model outputs), which the first run already wrote.

To compare speeds with the Rust benchmarks (`cargo bench -p basic-pitch-rs`, see
`crates/basic-pitch-rs/benches`), on the same inputs:

```sh
uv run --locked python -I bench_python.py ../tests/fixtures
```

The images of the README are drawn from what the Rust command-line tool returns on
`tests/fixtures/audio/pluck.wav`, never by hand. The script builds and runs the tool,
then needs `resvg` and `usvg` (`cargo install resvg usvg --locked`) and the static
Familjen Grotesk fonts (SIL Open Font License) in a directory of your choice; its
docstring says where to get them:

```sh
uv run --locked python -I render_readme_assets.py --fonts DIR
```

## Why the model is forced to ONNX

basic-pitch picks its default model at import time, in this order: TensorFlow,
CoreML, TFLite, ONNX (`basic_pitch/__init__.py`). On macOS, `coremltools` is a
mandatory dependency of basic-pitch 0.4.0, so a plain `basic-pitch` run on a Mac
uses the CoreML model, not ONNX. CoreML is another export of the network, run
by another runtime that can compute in reduced precision on Apple hardware: it
is not the graph that basic-pitch-rs runs.

`generate_fixtures.py` therefore loads `saved_models/icassp_2022/nmp.onnx`
explicitly, runs it with onnxruntime 1.31.0 on the CPU, and asserts that the
ONNX backend is in use. That file is byte-identical to the model bundled in
`crates/basic-pitch-rs/model/nmp.onnx` (SHA-256
`2c3c1d144bfa61ad236e92e169c13535c880469a12a047d4e73451f2c059a0ec`, recorded in
`meta-*.json`). It also checks that its step-by-step results match the official
`predict()` exactly, so the fixtures are what a user of basic-pitch gets with
the ONNX model.

The same trap exists on the Rust side: ONNX Runtime 1.31 can also send part of
the graph to CoreML on macOS. The `ort` backend of basic-pitch-rs forces the CPU
for that reason.

## Why Python 3.11

basic-pitch 0.4.0 supports Python 3.8 to 3.11 (its classifiers stop at 3.11, and
its TensorFlow requirement, `<2.15.1`, has no wheels for newer Pythons). 3.11 is
the newest version it supports. The ONNX path itself does not need TensorFlow,
and `basic-pitch[onnx]` resolves cleanly on 3.11. Both projects pin
`requires-python = "==3.11.*"`.

## Why `setuptools<82`

basic-pitch pins `resampy<0.4.3`. resampy 0.4.2 still imports `pkg_resources`
(`resampy/filters.py`), which setuptools removed in version 82.0.0. Without the
pin, `import basic_pitch` fails with `ModuleNotFoundError: No module named
'pkg_resources'`. The lockfiles resolve to setuptools 81.0.0, which only prints
a deprecation warning.

## The two variants: v0.4.0 and main

- **v0.4.0** is the latest release on PyPI (2024-08-16). It is what
  `pip install basic-pitch` gives.
- **main** is spotify/basic-pitch at `fa5997a` (2025-11-13), the latest commit
  when the fixtures were made. Its package still reports version 0.4.0.

Between the two, the only change that alters any output is the trimming in
`unwrap_output` (spotify/basic-pitch#179):

- v0.4.0 keeps `floor(n_samples * 86 / 22050)` frames, using the rounded frame
  rate of 86 frames per second, so it drops the last frames of the file
  (5 frames on a 10 s file);
- main keeps `int(n_samples / 36164 * 142)` frames, derived from the real hop
  between windows, and keeps the end of the file.

The other differences (code restructuring in `constrain_frequency`, CUDA
provider selection, constants, training code) do not change the output of the
ONNX pipeline. basic-pitch-rs implements both trim modes and uses the one of
main by default.

Other pins (numpy 2.4.6, librosa 0.11.0 with the `soxr_hq` resampler,
soxr 1.1.0, soundfile 0.14.0, pretty-midi 0.2.11.post0, mido 1.3.3) are fixed so
that a regeneration gives the same bytes. Change them only together with a
regeneration of the fixtures, in the same pull request.
