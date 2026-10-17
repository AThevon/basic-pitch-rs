# basic-pitch-rs in WebAssembly

Polyphonic audio-to-MIDI transcription that runs entirely in the browser (or in Node):
the model is embedded in the `.wasm` module and executed by [tract](https://github.com/sonos/tract),
in pure Rust. No server, no native library, and the audio never leaves the page.

This directory holds:

| file | what it is |
| --- | --- |
| `src/lib.rs` | the `wasm-bindgen` bindings: `Transcriber`, `Prediction`, `transcribe()` |
| `index.html`, `main.js`, `worker.js` | a demo page: drop a file, see a piano roll, download the MIDI file |
| `parity/` | the parity check against Python Basic Pitch, for Node and for the browser |
| `build.sh` | builds the module and its bindings into `pkg/` |

## Build

```sh
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.129 --locked   # must match Cargo.lock
examples/wasm/build.sh
```

The script builds with WebAssembly SIMD (`simd128`), which every current browser and
Node support and which makes inference about 2.4 times faster; `SIMD=0` turns it off. It
uses the `wasm-release` profile of the workspace, optimized for size.

## Run the demo

Serve the repository root with any static file server, then open `/examples/wasm/`:

```sh
python3 -m http.server 8000   # then http://localhost:8000/examples/wasm/
```

The page decodes the file with the Web Audio API, resampled to 22 050 Hz by the
browser, and transcribes it in a Web Worker. Changing a setting only decodes the notes
again, which takes a few milliseconds: the model does not run twice.

## Use it from JavaScript

```js
import init, { Transcriber } from "./pkg/basic_pitch_wasm.js";

await init();
const transcriber = new Transcriber(); // keep it: loading the model takes about 50 ms

// `samples` is a mono Float32Array at 22 050 Hz.
const notes = JSON.parse(transcriber.transcribe(samples));
// [{ start_s, end_s, start_frame, end_frame, pitch, amplitude, pitch_bends }, ...]

// Or keep the model outputs and decode them as often as needed:
const prediction = transcriber.predict(samples);
const fewer = JSON.parse(prediction.notes(JSON.stringify({ onset_threshold: 0.7 })));
const midiBytes = prediction.midi(null, JSON.stringify({ midi_tempo: 90 })); // Uint8Array
prediction.free();
```

Parameters are JSON objects with the field names of
[`NoteParams`](https://docs.rs/basic-pitch-rs/latest/basic_pitch/struct.NoteParams.html)
and `MidiParams` (snake case); missing fields keep their default and unknown fields are
rejected, so a typo cannot go unnoticed. `new Transcriber("v0.4.0")` trims the
output like the PyPI release instead of the current `main` branch.

## Parity with Python

```sh
node examples/wasm/parity/node.mjs
```

or open `/examples/wasm/parity/` in a browser, served from the repository root. Both
run the module on the six parity fixtures, with the eight post-processing
configurations and both trim modes, and compare the results with the official Python
package.

Measured on an Apple M2 Pro (Node 24.20 and Chromium 155 give the same results):

- **notes**: 824 of 825 identical per trim mode (times, pitch, pitch bends). The other
  one has a single pitch bend frame off by one third of a semitone, with
  the low thresholds of the `thresholds` configuration. Amplitudes differ by at most
  6.7e-5, which never changes a MIDI velocity.
- **model outputs**: within 5.3e-4 of `onnxruntime`, against 1.3e-6 for native builds.

The difference does not come from WebAssembly itself: tract's WebAssembly kernels, like
its portable fallback, do not use fused multiply-add, while its native ARM and x86
kernels and ONNX Runtime do. A native build with only the portable kernels
(`TRACT_CPU_ISA=-aarch64`) shows exactly the same 2.3e-4 on the `short` fixture. Use the
native library when bit-level parity matters.

## Size and speed

| build | `.wasm` | gzipped | 10 s of audio |
| --- | --- | --- | --- |
| `wasm-release` with SIMD (default of `build.sh`) | 9.3 MB | 2.3 MB | 0.45 s |
| `release` with SIMD | 17.7 MB | 3.9 MB | 0.36 s |
| `release` without SIMD | 17.6 MB | 3.8 MB | 0.87 s |

Most of the module is tract's operator library; the model itself is 230 kB. Running
`wasm-opt -O3` (binaryen) on the `release` build saves about 7% gzipped, for minutes
of build time. Inference is single-threaded, about 20 times faster than real time in
Node and Chromium, and loading the model takes about 50 ms.
