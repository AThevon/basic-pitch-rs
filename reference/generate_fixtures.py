"""Generates the parity fixtures of basic-pitch-rs from the official Python Basic Pitch.

The model is always the ONNX one (nmp.onnx, run on the CPU by onnxruntime): on macOS the
default backend of basic-pitch is CoreML, which would not compare the same graph.

Every intermediate step is saved, so each Rust layer can be tested on its own:

  <case>/audio_22050.npy      f32 (n,)                  librosa.load(sr=22050, mono=True)
  <case>/model_input.npy      f32 (n_win, 43844)        windows exactly as fed to the model (WINDOW_CASES only)
  <case>/raw_<out>.npy        f32 (n_win, 172, bins)    raw model outputs (out = note, onset, contour)
  <case>/<variant>/<out>.npy  f32 (n_frames, bins)      after unwrap_output (v0.4.0 only, main is checked
                                                        through its notes, the arrays are a longer copy)
  <case>/<variant>/times.npy  f64 (n_frames,)           model_frames_to_time(n_frames)
  <case>/<variant>/<config>.json and .mid               notes for one set of parameters

<variant> is "v0.4.0" (PyPI) or "main" (spotify/basic-pitch@fa5997a): the two only differ in
how unwrap_output trims the end of the file. Raw arrays are written by the v0.4.0 run only.

Each <config>.json holds:
  params             the post-processing parameters
  polyphonic_events  output_to_notes_polyphonic, in Python order: [start_frame, end_frame, pitch, amplitude]
  frame_events       after get_pitch_bends (or None bends): [start_frame, end_frame, pitch, amplitude, bends]
  events             model_output_to_notes: [start_s, end_s, pitch, amplitude, bends]
  midi               the .mid written by pretty_midi, parsed by mido: ticks_per_beat and, per track,
                     [absolute_tick, message] with message as mido's dict

Usage: python -I generate_fixtures.py FIXTURES_DIR WAV [WAV ...] [--variant-only]
  --variant-only   skip audio and raw arrays (used by the main run)
"""

import argparse
import hashlib
import importlib.metadata as md
import inspect
import json
import pathlib
import platform

import numpy as np

# model_input.npy is a plain slice of audio_22050.npy: two cases are enough to pin the windowing.
WINDOW_CASES = {"short", "synth"}

CONFIGS = {
    "default": {},
    "thresholds": {"onset_thresh": 0.3, "frame_thresh": 0.2, "min_note_len_ms": 58.0},
    "freq_range": {"min_freq": 80.0, "max_freq": 1000.0},
    "multiple_bends": {"multiple_pitch_bends": True},
    "no_melodia": {"melodia_trick": False},
    "no_infer_onsets": {"infer_onsets": False},
    "no_bends": {"include_pitch_bends": False},
    "tempo_90": {"midi_tempo": 90.0},
}

DEFAULTS = {
    "onset_thresh": 0.5,
    "frame_thresh": 0.3,
    "min_note_len_ms": 127.70,
    "infer_onsets": True,
    "min_freq": None,
    "max_freq": None,
    "include_pitch_bends": True,
    "multiple_pitch_bends": False,
    "melodia_trick": True,
    "midi_tempo": 120.0,
}


def sha256(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def to_py(x):
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


def event(e):
    out = [to_py(v) for v in e[:4]]
    if len(e) > 4:
        out.append(None if e[4] is None else [int(b) for b in e[4]])
    return out


def parse_midi(path):
    import mido

    mid = mido.MidiFile(str(path))
    tracks = []
    for track in mid.tracks:
        tick = 0
        msgs = []
        for msg in track:
            tick += msg.time
            d = msg.dict()
            d.pop("time", None)
            msgs.append([tick, d])
        tracks.append(msgs)
    return {"type": mid.type, "ticks_per_beat": mid.ticks_per_beat, "tracks": tracks}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("fixtures")
    p.add_argument("wavs", nargs="+")
    p.add_argument("--variant-only", action="store_true")
    args = p.parse_args()

    import librosa
    from basic_pitch import FilenameSuffix, build_icassp_2022_model_path
    from basic_pitch import inference as bp_inf
    from basic_pitch import note_creation as nc
    from basic_pitch.constants import AUDIO_N_SAMPLES, AUDIO_SAMPLE_RATE, FFT_HOP

    model_path = build_icassp_2022_model_path(FilenameSuffix.onnx)
    model = bp_inf.Model(model_path)
    assert model.model_type == bp_inf.Model.MODEL_TYPES.ONNX, model.model_type

    unwrap_takes_hop = "hop_size" in inspect.signature(bp_inf.unwrap_output).parameters
    variant = "main" if unwrap_takes_hop else "v0.4.0"
    n_overlapping_frames = 30
    overlap_len = n_overlapping_frames * FFT_HOP
    hop_size = AUDIO_N_SAMPLES - overlap_len

    fixtures = pathlib.Path(args.fixtures)
    summary = {}
    for wav in args.wavs:
        wav = pathlib.Path(wav)
        case = fixtures / wav.stem
        vdir = case / variant
        vdir.mkdir(parents=True, exist_ok=True)

        audio, _ = librosa.load(str(wav), sr=AUDIO_SAMPLE_RATE, mono=True)
        audio = audio.astype(np.float32, copy=False)
        original_length = audio.shape[0]
        padded = np.concatenate([np.zeros((overlap_len // 2,), dtype=np.float32), audio])
        windows = np.stack([w[:, 0] for w, _ in bp_inf.window_audio_file(padded, hop_size)]).astype(np.float32)

        raw = {"note": [], "onset": [], "contour": []}
        for w in windows:
            for k, v in model.predict(w[np.newaxis, :, np.newaxis]).items():
                raw[k].append(np.asarray(v, dtype=np.float32))
        raw = {k: np.concatenate(v, axis=0) for k, v in raw.items()}

        if not args.variant_only:
            np.save(case / "audio_22050.npy", audio)
            if wav.stem in WINDOW_CASES:
                np.save(case / "model_input.npy", windows)
            for k, v in raw.items():
                np.save(case / f"raw_{k}.npy", v)

        extra = (hop_size,) if unwrap_takes_hop else ()
        unwrapped = {k: bp_inf.unwrap_output(v, original_length, n_overlapping_frames, *extra) for k, v in raw.items()}
        if not args.variant_only:
            for k, v in unwrapped.items():
                np.save(vdir / f"{k}.npy", v.astype(np.float32, copy=False))
        n_frames = unwrapped["contour"].shape[0]
        np.save(vdir / "times.npy", nc.model_frames_to_time(n_frames).astype(np.float64))

        # Sanity check: the step-by-step run must match the official predict().
        official, _, official_events = bp_inf.predict(wav, model)
        for k in unwrapped:
            assert np.array_equal(official[k], unwrapped[k]), f"{wav.name}: {k} differs from predict()"

        counts = {}
        for name, overrides in CONFIGS.items():
            prm = {**DEFAULTS, **overrides}
            min_note_len = int(np.round(prm["min_note_len_ms"] / 1000 * (AUDIO_SAMPLE_RATE / FFT_HOP)))
            out = {k: v.copy() for k, v in unwrapped.items()}  # constrain_frequency mutates in place
            poly = nc.output_to_notes_polyphonic(
                out["note"],
                out["onset"],
                onset_thresh=prm["onset_thresh"],
                frame_thresh=prm["frame_thresh"],
                infer_onsets=prm["infer_onsets"],
                min_note_len=min_note_len,
                min_freq=prm["min_freq"],
                max_freq=prm["max_freq"],
                melodia_trick=prm["melodia_trick"],
            )
            if prm["include_pitch_bends"]:
                frame_events = nc.get_pitch_bends(out["contour"], poly)
            else:
                frame_events = [(e[0], e[1], e[2], e[3], None) for e in poly]

            out = {k: v.copy() for k, v in unwrapped.items()}
            midi, events = nc.model_output_to_notes(
                out,
                onset_thresh=prm["onset_thresh"],
                frame_thresh=prm["frame_thresh"],
                infer_onsets=prm["infer_onsets"],
                min_note_len=min_note_len,
                min_freq=prm["min_freq"],
                max_freq=prm["max_freq"],
                include_pitch_bends=prm["include_pitch_bends"],
                multiple_pitch_bends=prm["multiple_pitch_bends"],
                melodia_trick=prm["melodia_trick"],
                midi_tempo=prm["midi_tempo"],
            )
            mid_path = vdir / f"{name}.mid"
            midi.write(str(mid_path))
            if name == "default":
                assert [event(e) for e in events] == [event(e) for e in official_events], f"{wav.name}: events != predict()"

            with open(vdir / f"{name}.json", "w") as f:
                json.dump(
                    {
                        "params": {**prm, "min_note_len_frames": min_note_len},
                        "polyphonic_events": [event(e) for e in poly],
                        "frame_events": [event(e) for e in frame_events],
                        "events": [event(e) for e in events],
                        "midi": parse_midi(mid_path),
                    },
                    f,
                    indent=None,
                    separators=(",", ":"),
                )
            counts[name] = len(events)

        summary[wav.stem] = {
            "input_sha256": sha256(wav),
            "original_length_22050": int(original_length),
            "n_windows": int(windows.shape[0]),
            "n_frames": int(n_frames),
            "notes": counts,
        }
        print(f"{variant} {wav.stem}: {windows.shape[0]} windows, {n_frames} frames, notes {counts}")

    meta = {
        "variant": variant,
        "model_sha256": sha256(model_path),
        "model_backend": model.model_type.name,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            n: (md.version(n) if n in {d.metadata["Name"].lower() for d in md.distributions()} else None)
            for n in ["basic-pitch", "numpy", "librosa", "soxr", "soundfile", "onnxruntime", "pretty-midi", "mido"]
        },
        "librosa_load_res_type": inspect.signature(librosa.load).parameters["res_type"].default,
        "cases": summary,
    }
    with open(fixtures / f"meta-{variant}.json", "w") as f:
        json.dump(meta, f, indent=1)


if __name__ == "__main__":
    main()
