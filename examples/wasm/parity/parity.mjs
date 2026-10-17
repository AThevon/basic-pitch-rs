// Copyright 2026 Adrien Thevon
// SPDX-License-Identifier: Apache-2.0

// Compares the WebAssembly build with the official Python Basic Pitch, on the parity
// fixtures of tests/fixtures. Shared by the Node runner (node.mjs) and the browser
// runner (index.html).
//
// - Posteriorgrams: largest absolute difference with <case>/v0.4.0/{note,onset,contour}.npy.
// - Notes: for every fixture configuration and both trim modes, the decoded notes are
//   compared with <case>/<variant>/<config>.json, in order: times, pitch, pitch bends
//   and float32 amplitude.
//
// Without FMA, tract's WebAssembly kernels round differently from its native ones, and
// the model outputs drift from Python by up to about 5e-4 (1e-6 natively). Amplitudes
// then differ in the last digits, and a pitch bend can flip on a near tie. A run fails
// if a note appears, disappears, or changes its times or pitch, or if a frame count
// differs; amplitude and pitch bend differences are only reported.

export const CASES = ["pluck", "short", "silence", "synth", "vocadito_10", "vocadito_14"];
export const CONFIGS = [
  "default",
  "thresholds",
  "freq_range",
  "multiple_bends",
  "no_melodia",
  "no_infer_onsets",
  "no_bends",
  "tempo_90",
];
const VARIANTS = ["v0.4.0", "main"];
const SAMPLE_RATE = 22050;

/** Parses a little-endian float32 or float64 .npy file in C order. */
export function parseNpy(bytes, name = "array") {
  const text = (start, end) => String.fromCharCode(...bytes.subarray(start, end));
  if (text(1, 6) !== "NUMPY") throw new Error(`${name}: not a .npy file`);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const v1 = bytes[6] === 1;
  const headerStart = v1 ? 10 : 12;
  const headerEnd = headerStart + (v1 ? view.getUint16(8, true) : view.getUint32(8, true));
  const header = text(headerStart, headerEnd);
  if (!header.includes("'fortran_order': False")) throw new Error(`${name}: Fortran order`);
  const data = bytes.slice(headerEnd).buffer;
  if (header.includes("'<f4'")) return new Float32Array(data);
  if (header.includes("'<f8'")) return new Float64Array(data);
  throw new Error(`${name}: unsupported dtype in ${header}`);
}

/** Maps the Python parameter names of a fixture to the fields of NoteParams. */
function noteParams(p) {
  return JSON.stringify({
    onset_threshold: p.onset_thresh,
    frame_threshold: p.frame_thresh,
    minimum_note_length_ms: p.min_note_len_ms,
    minimum_frequency: p.min_freq,
    maximum_frequency: p.max_freq,
    infer_onsets: p.infer_onsets,
    melodia_trick: p.melodia_trick,
    include_pitch_bends: p.include_pitch_bends,
  });
}

const sameBends = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const velocity = (amplitude) => Math.round(127 * amplitude);
const median = (xs) => [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)];

function maxAbsDiff(a, b) {
  if (a.length !== b.length) throw new Error(`length ${a.length} != ${b.length}`);
  let max = 0;
  for (let i = 0; i < a.length; i++) max = Math.max(max, Math.abs(a[i] - b[i]));
  return max;
}

/**
 * Runs the comparison.
 *
 * @param {object} options
 * @param {Function} options.Transcriber  the class exported by the wasm-bindgen module
 * @param {(path: string) => Promise<Uint8Array>} options.read  reads a file of tests/fixtures
 * @param {(line: string) => void} options.log
 * @param {number} [options.repeats]  timed runs per case (the median is reported)
 * @returns {Promise<object>} a summary; `failures` is 0 when the notes match
 */
export async function runParity({ Transcriber, read, log, repeats = 3 }) {
  const readJson = async (path) => JSON.parse(new TextDecoder().decode(await read(path)));
  const meta = await readJson("meta-main.json");
  const summary = { failures: 0, worstDiff: 0, variants: {}, timings: {} };

  for (const variant of VARIANTS) {
    let t0 = performance.now();
    const transcriber = new Transcriber(variant);
    const loadMs = performance.now() - t0;
    log(`\n${variant} trim (model loaded in ${loadMs.toFixed(0)} ms)`);
    const totals = { exact: 0, amplitudeOnly: 0, total: 0, amplitudeDiff: 0, velocity: 0 };
    summary.variants[variant] = totals;

    for (const name of CASES) {
      const audio = parseNpy(await read(`${name}/audio_22050.npy`), name);
      const runs = [];
      let prediction;
      for (let i = 0; i < repeats; i++) {
        prediction?.free();
        t0 = performance.now();
        prediction = transcriber.predict(audio);
        runs.push(performance.now() - t0);
      }
      const nFrames = prediction.nFrames;
      const predictMs = median(runs);
      summary.timings[`${variant}/${name}`] = { seconds: audio.length / SAMPLE_RATE, predictMs };

      let diffs = "";
      if (variant === "v0.4.0") {
        const d = [];
        for (const k of ["note", "onset", "contour"]) {
          const expected = parseNpy(await read(`${name}/${variant}/${k}.npy`), k);
          d.push(maxAbsDiff(prediction[k](), expected));
        }
        summary.worstDiff = Math.max(summary.worstDiff, ...d);
        diffs = `, max |diff| note/onset/contour ${d.map((x) => x.toExponential(2)).join(" / ")}`;
      } else if (meta.cases[name].n_frames !== nFrames) {
        log(`  ${name}: ${nFrames} frames, Python ${meta.cases[name].n_frames}`);
        summary.failures++;
      }

      const stats = { exact: 0, amplitudeOnly: 0, total: 0, amplitudeDiff: 0, velocity: 0 };
      const details = [];
      for (const config of CONFIGS) {
        const fixture = await readJson(`${name}/${variant}/${config}.json`);
        const expected = fixture.events;
        const ours = JSON.parse(prediction.notes(noteParams(fixture.params)));
        stats.total += Math.max(ours.length, expected.length);
        if (ours.length !== expected.length) {
          details.push(`${config}: ${ours.length} notes, Python ${expected.length}`);
          summary.failures++;
        }
        for (let i = 0; i < Math.min(ours.length, expected.length); i++) {
          const [startS, endS, pitch, amplitude, bends] = expected[i];
          const note = ours[i];
          const differ = [];
          if (note.start_s !== startS || note.end_s !== endS) differ.push("times");
          if (note.pitch !== pitch) differ.push("pitch");
          if (differ.length) summary.failures++;
          if (!sameBends(note.pitch_bends, bends)) {
            const n = bends?.filter((b, k) => b !== note.pitch_bends?.[k]).length;
            differ.push(`pitch bends (${n} of ${bends?.length} frames)`);
          }
          if (differ.length) {
            details.push(`${config} note #${i} (MIDI ${pitch} at ${startS.toFixed(3)} s): ${differ.join(", ")}`);
            continue;
          }
          if (Math.fround(note.amplitude) === Math.fround(amplitude)) stats.exact++;
          else stats.amplitudeOnly++;
          stats.amplitudeDiff = Math.max(stats.amplitudeDiff, Math.abs(note.amplitude - amplitude));
          if (velocity(note.amplitude) !== velocity(amplitude)) stats.velocity++;
        }
      }
      prediction.free();
      for (const [k, v] of Object.entries(stats)) {
        totals[k] = k === "amplitudeDiff" ? Math.max(totals[k], v) : totals[k] + v;
      }

      log(
        `  ${name.padEnd(12)} ${(audio.length / SAMPLE_RATE).toFixed(1).padStart(5)} s, ` +
          `${String(nFrames).padStart(4)} frames, predict ${predictMs.toFixed(0).padStart(4)} ms ` +
          `(median of ${repeats})${diffs}`,
      );
      log(
        `${" ".repeat(15)}notes: ${stats.exact + stats.amplitudeOnly}/${stats.total} identical, ` +
          `${stats.exact} with the same float32 amplitude ` +
          `(max |diff| ${stats.amplitudeDiff.toExponential(1)}, ${stats.velocity} velocities differ)`,
      );
      for (const d of details.slice(0, 5)) log(`${" ".repeat(17)}differs: ${d}`);
    }
    transcriber.free();
  }

  log("");
  for (const [variant, t] of Object.entries(summary.variants)) {
    log(
      `${variant}: ${t.exact + t.amplitudeOnly}/${t.total} notes identical (times, pitch, bends), ` +
        `${t.exact} also with the same float32 amplitude; max amplitude difference ` +
        `${t.amplitudeDiff.toExponential(1)}, ${t.velocity} different velocities`,
    );
  }
  log(`worst posteriorgram difference: ${summary.worstDiff.toExponential(2)}`);
  log(summary.failures ? `FAILED: ${summary.failures} differences in notes or frames` : "OK");
  return summary;
}
