"""Times the official Python Basic Pitch (ONNX model, onnxruntime on the CPU), on the same
inputs as the Rust benchmarks of crates/basic-pitch-rs/benches:

  window           one model window: the second window of vocadito_14
  load             loading the ONNX model
  postprocessing   model_output_to_notes on the posteriorgrams of vocadito_14 (v0.4.0),
                   notes only, then notes and the PrettyMIDI object
  transcribe 60 s  predict() on one minute of audio: the fixture recordings vocadito_14,
                   pluck, vocadito_10 and synth (audio_22050.npy), concatenated and
                   repeated, written to a float32 WAV at 22 050 Hz, so that librosa
                   decodes it without resampling. The time of librosa.load alone is also
                   reported.

Inference runs twice: with onnxruntime's default thread pool (one thread per core, like
basic-pitch) and with a single thread, like tract. Each timing is the median of
--repeats runs, after one warm-up run.

Usage: python -I bench_python.py FIXTURES_DIR [--repeats 5]
"""

import argparse
import contextlib
import io
import os
import pathlib
import platform
import statistics
import subprocess
import tempfile
import time

import numpy as np

CASES_FOR_ONE_MINUTE = ["vocadito_14", "pluck", "vocadito_10", "synth"]


def median_time(fn, repeats):
    fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def fmt(seconds):
    return f"{seconds * 1000:9.2f} ms"


def cpu_name():
    try:
        return subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return platform.processor() or platform.machine()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("fixtures", type=pathlib.Path)
    p.add_argument("--repeats", type=int, default=5)
    args = p.parse_args()

    import librosa
    import onnxruntime as ort
    import soundfile as sf
    from basic_pitch import FilenameSuffix, build_icassp_2022_model_path
    from basic_pitch import inference as bp_inf
    from basic_pitch import note_creation as nc
    from basic_pitch.constants import AUDIO_N_SAMPLES, AUDIO_SAMPLE_RATE, FFT_HOP

    model_path = build_icassp_2022_model_path(FilenameSuffix.onnx)
    print(f"{cpu_name()}, {os.cpu_count()} logical CPUs, {platform.platform()}")
    print(f"Python {platform.python_version()}, onnxruntime {ort.__version__}, numpy {np.__version__}")
    print(f"median of {args.repeats} runs after a warm-up\n")

    overlap_len = 30 * FFT_HOP
    hop_size = AUDIO_N_SAMPLES - overlap_len
    vocadito = np.load(args.fixtures / "vocadito_14" / "audio_22050.npy")
    padded = np.concatenate([np.zeros(overlap_len // 2, dtype=np.float32), vocadito])
    window = padded[hop_size : hop_size + AUDIO_N_SAMPLES][np.newaxis, :, np.newaxis]

    sources = [np.load(args.fixtures / c / "audio_22050.npy") for c in CASES_FOR_ONE_MINUTE]
    one_minute = np.resize(np.concatenate(sources), 60 * AUDIO_SAMPLE_RATE).astype(np.float32)

    model = bp_inf.Model(model_path)
    assert model.model_type == bp_inf.Model.MODEL_TYPES.ONNX, model.model_type
    single = ort.SessionOptions()
    single.intra_op_num_threads = 1
    single.inter_op_num_threads = 1
    model_1t = bp_inf.Model(model_path)
    model_1t.model = ort.InferenceSession(str(model_path), single, providers=["CPUExecutionProvider"])

    print(f"load            {fmt(median_time(lambda: bp_inf.Model(model_path), args.repeats))}")
    for label, m in [("default threads", model), ("1 thread", model_1t)]:
        t = median_time(lambda: m.predict(window), max(args.repeats, 20))
        print(f"window          {fmt(t)}  ({label})")

    output = {
        k: np.load(args.fixtures / "vocadito_14" / "v0.4.0" / f"{k}.npy") for k in ["note", "onset", "contour"]
    }
    min_note_len = int(np.round(127.70 / 1000 * (AUDIO_SAMPLE_RATE / FFT_HOP)))

    def notes_only():
        events = nc.output_to_notes_polyphonic(
            output["note"], output["onset"], onset_thresh=0.5, frame_thresh=0.3, infer_onsets=True,
            min_note_len=min_note_len, min_freq=None, max_freq=None, melodia_trick=True,
        )
        events = nc.get_pitch_bends(output["contour"], events)
        times = nc.model_frames_to_time(output["contour"].shape[0])
        return [(times[e[0]], times[e[1]], e[2], e[3], e[4]) for e in events]

    def notes_and_midi():
        return nc.model_output_to_notes(output, onset_thresh=0.5, frame_thresh=0.3, min_note_len=min_note_len)

    t = median_time(notes_only, max(args.repeats, 20))
    print(f"postprocessing  {fmt(t)}  (vocadito_14, notes)")
    t = median_time(notes_and_midi, max(args.repeats, 20))
    print(f"postprocessing  {fmt(t)}  (vocadito_14, notes and PrettyMIDI)")

    with tempfile.TemporaryDirectory() as tmp:
        wav = pathlib.Path(tmp) / "one_minute.wav"
        sf.write(wav, one_minute, AUDIO_SAMPLE_RATE, subtype="FLOAT")
        loaded, _ = librosa.load(str(wav), sr=AUDIO_SAMPLE_RATE, mono=True)
        assert np.array_equal(loaded, one_minute)
        t = median_time(lambda: librosa.load(str(wav), sr=AUDIO_SAMPLE_RATE, mono=True), args.repeats)
        print(f"librosa.load    {fmt(t)}  (60 s, float32 WAV at 22 050 Hz)")
        for label, m in [("default threads", model), ("1 thread", model_1t)]:
            def predict():
                with contextlib.redirect_stdout(io.StringIO()):  # predict() prints the file name
                    return bp_inf.predict(wav, m)

            t = median_time(predict, args.repeats)
            print(f"transcribe 60 s {fmt(t)}  (predict(), {label}, {60 / t:.0f}x real time)")


if __name__ == "__main__":
    main()
