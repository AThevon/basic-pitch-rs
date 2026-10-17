// Copyright 2026 Adrien Thevon
// SPDX-License-Identifier: Apache-2.0

// The demo page: decodes a file with the Web Audio API, sends the samples to the
// worker, draws the notes as a piano roll and offers the MIDI file.

const SAMPLE_RATE = 22050;
const $ = (id) => document.getElementById(id);

const worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
const state = { ready: false, busy: false, duration: 0, notes: [], name: "transcription" };

const controls = {
  onset: { input: $("onset"), output: $("onset-out"), format: (v) => v.toFixed(2) },
  frame: { input: $("frame"), output: $("frame-out"), format: (v) => v.toFixed(2) },
  length: { input: $("length"), output: $("length-out"), format: (v) => `${Math.round(v)} ms` },
};

function params() {
  return {
    onset_threshold: Number(controls.onset.input.value),
    frame_threshold: Number(controls.frame.input.value),
    minimum_note_length_ms: Number(controls.length.input.value),
    melodia_trick: $("melodia").checked,
    bend_offset: $("center").checked ? -1 : 0,
  };
}

function setStatus(text, isError = false) {
  $("status").textContent = text;
  $("status").classList.toggle("error", isError);
}

/** Decodes any format the browser supports to mono samples at 22 050 Hz. */
async function decode(file) {
  // decodeAudioData resamples to the sample rate of the context.
  const context = new OfflineAudioContext(1, 1, SAMPLE_RATE);
  const buffer = await context.decodeAudioData(await file.arrayBuffer());
  const mono = new Float32Array(buffer.length);
  for (let c = 0; c < buffer.numberOfChannels; c++) {
    const channel = buffer.getChannelData(c);
    for (let i = 0; i < mono.length; i++) mono[i] += channel[i];
  }
  if (buffer.numberOfChannels > 1) {
    for (let i = 0; i < mono.length; i++) mono[i] /= buffer.numberOfChannels;
  }
  return mono;
}

async function transcribe(file) {
  if (!state.ready || state.busy) return;
  state.busy = true;
  state.name = file.name.replace(/\.[^.]+$/, "") || "transcription";
  try {
    setStatus(`decoding ${file.name}...`);
    const audio = await decode(file);
    state.duration = audio.length / SAMPLE_RATE;
    setStatus(`transcribing ${state.duration.toFixed(1)} s of audio...`);
    worker.postMessage({ type: "transcribe", audio, params: params() }, [audio.buffer]);
  } catch (error) {
    state.busy = false;
    setStatus(`could not decode ${file.name}: ${error.message ?? error}`, true);
  }
}

worker.onmessage = ({ data }) => {
  switch (data.type) {
    case "ready":
      state.ready = true;
      setStatus(`basic-pitch-rs ${data.version} ready (loaded in ${Math.round(data.loadMs)} ms)`);
      break;
    case "notes": {
      state.busy = false;
      state.notes = data.notes;
      draw();
      $("download").disabled = false;
      const timing =
        data.inferenceMs === undefined
          ? `decoded in ${data.decodeMs.toFixed(1)} ms`
          : `transcribed in ${Math.round(data.inferenceMs + data.decodeMs)} ms, ` +
            `${(state.duration / ((data.inferenceMs + data.decodeMs) / 1000)).toFixed(0)}x real time`;
      setStatus(`${data.notes.length} notes in ${state.duration.toFixed(1)} s, ${timing}`);
      break;
    }
    case "midi": {
      const url = URL.createObjectURL(new Blob([data.bytes], { type: "audio/midi" }));
      const link = Object.assign(document.createElement("a"), { href: url, download: `${state.name}.mid` });
      link.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
      break;
    }
    case "error":
      state.busy = false;
      setStatus(data.message, true);
      break;
  }
};

/** Draws the notes: time on x, pitch on y, velocity as opacity, pitch bends as a line. */
function draw() {
  const canvas = $("roll");
  const ratio = window.devicePixelRatio || 1;
  const { width, height } = canvas.getBoundingClientRect();
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  const g = canvas.getContext("2d");
  g.scale(ratio, ratio);
  g.clearRect(0, 0, width, height);
  $("empty").hidden = state.notes.length > 0;
  if (!state.notes.length) return;

  const css = getComputedStyle(document.documentElement);
  const accent = css.getPropertyValue("--accent").trim();
  const line = css.getPropertyValue("--line").trim();
  const muted = css.getPropertyValue("--muted").trim();

  const pitches = state.notes.map((n) => n.pitch);
  const low = Math.min(...pitches) - 2;
  const high = Math.max(...pitches) + 2;
  const left = 34;
  const bottom = 18;
  const x = (s) => left + (s / state.duration) * (width - left);
  const rowHeight = (height - bottom) / (high - low + 1);
  const y = (pitch) => (high - pitch) * rowHeight;

  g.font = `11px ${css.fontFamily}`;
  g.fillStyle = muted;
  g.strokeStyle = line;
  g.lineWidth = 1;
  for (let p = low; p <= high; p++) {
    if (p % 12 !== 0) continue;
    g.beginPath();
    g.moveTo(left, y(p) + rowHeight + 0.5);
    g.lineTo(width, y(p) + rowHeight + 0.5);
    g.stroke();
    g.fillText(`C${p / 12 - 1}`, 2, y(p) + rowHeight - 2);
  }
  const step = state.duration > 60 ? 10 : state.duration > 20 ? 5 : 1;
  for (let s = 0; s <= state.duration; s += step) {
    g.fillText(`${s}s`, x(s) + 2, height - 4);
  }

  for (const note of state.notes) {
    const x0 = x(note.start_s);
    const w = Math.max(1, x(note.end_s) - x0);
    g.globalAlpha = 0.35 + 0.65 * Math.min(1, note.amplitude);
    g.fillStyle = accent;
    g.fillRect(x0, y(note.pitch) + 1, w, Math.max(1, rowHeight - 2));
    const bends = note.pitch_bends;
    if (bends?.length > 1 && rowHeight >= 4) {
      g.globalAlpha = 0.9;
      g.strokeStyle = "#e9f7eb";
      g.beginPath();
      bends.forEach((bend, i) => {
        const px = x0 + (i / (bends.length - 1)) * w;
        const py = y(note.pitch + bend / 3) + rowHeight / 2;
        if (i) g.lineTo(px, py);
        else g.moveTo(px, py);
      });
      g.stroke();
    }
  }
  g.globalAlpha = 1;
}

for (const { input, output, format } of Object.values(controls)) {
  const show = () => (output.textContent = format(Number(input.value)));
  show();
  input.addEventListener("input", show);
}
for (const element of [...Object.values(controls).map((c) => c.input), $("melodia"), $("center")]) {
  element.addEventListener("change", () => {
    if (state.notes.length || state.duration) worker.postMessage({ type: "decode", params: params() });
  });
}

const drop = $("drop");
$("file").addEventListener("change", (event) => {
  const [file] = event.target.files;
  if (file) transcribe(file);
  event.target.value = "";
});
drop.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    $("file").click();
  }
});
drop.addEventListener("dragover", (event) => {
  event.preventDefault();
  drop.classList.add("over");
});
drop.addEventListener("dragleave", () => drop.classList.remove("over"));
drop.addEventListener("drop", (event) => {
  event.preventDefault();
  drop.classList.remove("over");
  const [file] = event.dataTransfer.files;
  if (file) transcribe(file);
});
$("download").addEventListener("click", () => worker.postMessage({ type: "midi", params: params() }));
new ResizeObserver(draw).observe($("roll"));
draw();

// Test hook: transcribe samples that are already mono at 22 050 Hz.
window.basicPitchDemo = {
  transcribeSamples(audio, name = "samples") {
    state.busy = true;
    state.name = name;
    state.duration = audio.length / SAMPLE_RATE;
    worker.postMessage({ type: "transcribe", audio, params: params() });
  },
  get notes() {
    return state.notes;
  },
};
