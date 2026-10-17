// Copyright 2026 Adrien Thevon
// SPDX-License-Identifier: Apache-2.0

// Runs basic-pitch-rs off the main thread. The page sends mono samples at 22 050 Hz;
// the worker keeps the model outputs, so new settings only decode the notes again.

import init, { Transcriber, version } from "./pkg/basic_pitch_wasm.js";

let transcriber;
let prediction;

const ready = (async () => {
  const t0 = performance.now();
  await init();
  transcriber = new Transcriber();
  postMessage({ type: "ready", version: version(), loadMs: performance.now() - t0 });
})();

function decode(params) {
  const t0 = performance.now();
  const notes = JSON.parse(prediction.notes(JSON.stringify(params)));
  return { notes, decodeMs: performance.now() - t0 };
}

onmessage = async ({ data }) => {
  try {
    await ready;
    if (data.type === "transcribe") {
      prediction?.free();
      prediction = undefined;
      const t0 = performance.now();
      prediction = transcriber.predict(data.audio);
      const inferenceMs = performance.now() - t0;
      postMessage({ type: "notes", inferenceMs, nFrames: prediction.nFrames, ...decode(data.params) });
    } else if (data.type === "decode" && prediction) {
      postMessage({ type: "notes", ...decode(data.params) });
    } else if (data.type === "midi" && prediction) {
      const bytes = prediction.midi(JSON.stringify(data.params));
      postMessage({ type: "midi", bytes }, [bytes.buffer]);
    }
  } catch (error) {
    postMessage({ type: "error", message: String(error?.message ?? error) });
  }
};
