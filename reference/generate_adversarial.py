"""Generates the adversarial post-processing fixtures of basic-pitch-rs.

The audio fixtures only cover what the model produces on a few recordings. These cases
feed note_creation.py with synthetic posteriorgrams that reach the corners of the code:
thresholds at 0 and 1, plateaus and ties, frequency bounds that become negative numpy
indices, notes longer than numpy's 8192-element reduction buffer, edge keys and contour
bins, more than 15 pitches with multiple_pitch_bends, extreme tempos, custom pitch bend
tolerances, NaN activations, and notes that pretty_midi refuses to write.

Each case is written to <out>/<case>.json:

  n_frames                  number of frames
  note, onset, contour      {"fill": value, "entries": [[frame, bin, value], ...]}: every
                            cell is `fill` except the listed ones (float32 values)
  runs                      one object per parameter set:
    name, params            the parameters, named as in generate_fixtures.py
    polyphonic              output_to_notes_polyphonic: [start_frame, end_frame, pitch, amplitude]
    events, midi            model_output_to_notes and the parsed .mid (as in generate_fixtures.py),
                            or {"error": ...} for a .mid that pretty_midi cannot write
    error                   instead of events and midi, when model_output_to_notes raises
  bend_tolerances           optional: get_pitch_bends with other n_bins_tolerance values, on
                            the notes of the first run: [{"tolerance", "notes", "bends"}]

Floats that JSON cannot hold are strings: "NaN", "Infinity", "-Infinity".

Usage: python -I generate_adversarial.py OUT_DIR
"""

import json
import math
import pathlib
import sys

import numpy as np

# `python -I` does not put the script's directory on sys.path.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from generate_fixtures import DEFAULTS, event, parse_midi  # noqa: E402

N_NOTE = 88
N_CONTOUR = 264


def num(x):
    x = float(x)
    if math.isnan(x):
        return "NaN"
    if math.isinf(x):
        return "Infinity" if x > 0 else "-Infinity"
    return x


def encode_events(events):
    out = []
    for e in events:
        e = event(e)
        out.append([num(v) if isinstance(v, float) else v for v in e])
    return out


def sparse(m):
    m = np.asarray(m, dtype=np.float32)
    values, counts = np.unique(m, return_counts=True, equal_nan=True)
    fill = values[np.argmax(counts)]
    if np.isnan(fill):
        mask = ~np.isnan(m)
    else:
        mask = (m != fill) | np.isnan(m)
    entries = [[int(t), int(b), num(m[t, b])] for t, b in zip(*np.nonzero(mask))]
    return {"fill": num(fill), "entries": entries}


def run(nc, out, name, overrides, tmp):
    prm = {**DEFAULTS, **overrides}
    min_note_len = int(np.round(prm["min_note_len_ms"] / 1000 * (22050 / 256)))
    result = {"name": name, "params": {**prm, "min_note_len_frames": min_note_len}}
    copy = {k: v.copy() for k, v in out.items()}  # constrain_frequency mutates its inputs
    poly = nc.output_to_notes_polyphonic(
        copy["note"],
        copy["onset"],
        onset_thresh=prm["onset_thresh"],
        frame_thresh=prm["frame_thresh"],
        infer_onsets=prm["infer_onsets"],
        min_note_len=min_note_len,
        min_freq=prm["min_freq"],
        max_freq=prm["max_freq"],
        melodia_trick=prm["melodia_trick"],
    )
    result["polyphonic"] = encode_events(poly)
    copy = {k: v.copy() for k, v in out.items()}
    try:
        midi, events = nc.model_output_to_notes(
            copy,
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
    except Exception as e:  # noqa: BLE001 - the error itself is the expected result
        result["error"] = repr(e)
        return result, poly
    result["events"] = encode_events(events)
    path = tmp / "run.mid"
    try:
        midi.write(str(path))
        result["midi"] = parse_midi(path)
    except Exception as e:  # noqa: BLE001
        result["midi"] = {"error": repr(e)}
    return result, poly


def zeros(n):
    return (
        np.zeros((n, N_NOTE), np.float32),
        np.zeros((n, N_NOTE), np.float32),
        np.zeros((n, N_CONTOUR), np.float32),
    )


def random_case(rng, n, quantize=None):
    note = (rng.uniform(0, 0.25, (n, N_NOTE)) * (rng.uniform(size=(n, N_NOTE)) < 0.3)).astype(np.float32)
    onset = (rng.uniform(0, 0.3, (n, N_NOTE)) * (rng.uniform(size=(n, N_NOTE)) < 0.3)).astype(np.float32)
    contour = (rng.uniform(0, 0.2, (n, N_CONTOUR)) * (rng.uniform(size=(n, N_CONTOUR)) < 0.25)).astype(np.float32)
    for _ in range(40):
        b = int(rng.integers(0, N_NOTE))
        s = int(rng.integers(0, n))
        e = min(n, s + int(rng.integers(1, 60)))
        note[s:e, b] = rng.uniform(0.25, 1.0, e - s)
        if rng.uniform() < 0.7:
            onset[s, b] = rng.uniform(0.3, 1.0)
        cb = min(max(3 * b + int(rng.integers(-4, 5)), 0), N_CONTOUR - 1)
        contour[s:e, cb] = rng.uniform(0.3, 1.0, e - s)
        if rng.uniform() < 0.3:
            contour[s:e, min(max(cb + int(rng.integers(-30, 31)), 0), N_CONTOUR - 1)] = 1.0
    if quantize:
        note, onset, contour = (np.round(m * quantize) / quantize for m in (note, onset, contour))
    return note.astype(np.float32), onset.astype(np.float32), contour.astype(np.float32)


FREQ_BOUNDS = [
    (80.0, 1000.0),
    (1000.0, 200.0),
    (20.0, None),
    (None, 20.0),
    (5.0, 26.0),
    (27.5, 27.5),
    (4300.0, None),
    (None, 4500.0),
    (13.75, 30.0),
]


def cases():
    rng = np.random.default_rng(20261017)

    note, onset, contour = random_case(rng, 150)
    runs = [
        ("default", {}),
        ("thresholds_low", {"onset_thresh": 0.1, "frame_thresh": 0.1, "min_note_len_ms": 58.0}),
        ("thresholds_one", {"onset_thresh": 1.0, "frame_thresh": 1.0}),
        ("no_infer_onsets", {"infer_onsets": False}),
        ("no_melodia", {"melodia_trick": False}),
        ("no_bends", {"include_pitch_bends": False}),
        ("multiple_bends", {"multiple_pitch_bends": True}),
        ("min_len_300ms", {"min_note_len_ms": 300.0}),
        ("min_len_11_6ms", {"min_note_len_ms": 11.6}),
        ("min_len_1_5_frames", {"min_note_len_ms": 1.5 * 256 / 22050 * 1000}),
        ("tempo_33_3", {"midi_tempo": 33.3}),
        ("tempo_97_3", {"midi_tempo": 97.3}),
        ("tempo_1e5", {"midi_tempo": 1e5}),
        ("tempo_3", {"midi_tempo": 3.0}),
    ]
    runs += [(f"freq_{lo}_{hi}", {"min_freq": lo, "max_freq": hi}) for lo, hi in FREQ_BOUNDS]
    yield "random", (note, onset, contour), runs, [0, 1, 5, 24, 26, 100, 263, 264, 300, 1000]

    # Thresholds at zero turn almost every cell into a note: a short case is enough.
    note, onset, contour = random_case(rng, 40)
    yield "random_short", (note, onset, contour), [
        ("thresholds_zero", {"onset_thresh": 0.0, "frame_thresh": 0.0, "min_note_len_ms": 0.0}),
        ("frame_thresh_zero", {"frame_thresh": 0.0}),
    ], []

    note, onset, contour = random_case(rng, 172, quantize=8)
    yield "random_quantized", (note, onset, contour), [
        ("default", {}),
        ("thresholds_quarter", {"onset_thresh": 0.25, "frame_thresh": 0.25}),
        ("no_infer_onsets", {"infer_onsets": False}),
        ("multiple_bends", {"multiple_pitch_bends": True}),
    ], []

    # Lowest and highest keys, contour peaks on the first and last bins.
    note, onset, contour = zeros(60)
    for b in [0, 1, 86, 87]:
        note[5:40, b] = 0.8
        onset[5, b] = 0.9
    contour[:, 0] = 0.9
    contour[:, 263] = 0.9
    contour[:, 2] = 0.5
    contour[:, 260] = 0.5
    yield "edges", (note, onset, contour), [("default", {}), ("multiple_bends", {"multiple_pitch_bends": True})], [
        0,
        3,
        264,
    ]

    # 22 overlapping pitches: more instruments than MIDI channels.
    note, onset, contour = zeros(120)
    contour[:] = (rng.uniform(0, 1, (120, N_CONTOUR)) * (rng.uniform(size=(120, N_CONTOUR)) < 0.1)).astype(np.float32)
    for j, b in enumerate(range(2, 88, 4)):
        note[3 + j : 60 + j, b] = 0.6 + 0.01 * j
        onset[3 + j, b] = 0.95
        contour[3 + j : 60 + j, 3 * b + 1] = 0.8
    yield "many_pitches", (note, onset, contour), [
        ("multiple_bends", {"multiple_pitch_bends": True, "midi_tempo": 97.3}),
        ("single_track", {"midi_tempo": 33.3}),
    ], []

    # Ties everywhere: numpy's first-index argmax decides.
    yield "plateau", (np.full((50, N_NOTE), 0.5, np.float32), np.zeros((50, N_NOTE), np.float32), np.full((50, N_CONTOUR), 0.5, np.float32)), [
        ("default", {}),
        ("no_infer_onsets", {"infer_onsets": False}),
    ], []
    yield "plateau_onsets", (np.full((50, N_NOTE), 0.5, np.float32), np.full((50, N_NOTE), 0.5, np.float32), np.full((50, N_CONTOUR), 0.5, np.float32)), [
        ("default", {}),
    ], []

    # Silence: the inferred onsets are 0/0 = NaN.
    yield "silence", zeros(50), [("default", {}), ("no_infer_onsets", {"infer_onsets": False})], []

    for n in [1, 2, 3]:
        full = (np.full((n, N_NOTE), 0.9, np.float32), np.full((n, N_NOTE), 0.9, np.float32), np.full((n, N_CONTOUR), 0.5, np.float32))
        yield f"tiny_{n}", full, [("min_len_0", {"min_note_len_ms": 0.0})], []

    # Velocities right at rounding boundaries.
    note, onset, contour = zeros(60)
    for k, amp in enumerate([0.5, 0.3976378, 0.003937008, 1.0, 0.99999994, 0.30000001]):
        b = 10 + 12 * k
        note[2:30, b] = np.float32(amp)
        onset[2, b] = 0.9
    yield "velocity_rounding", (note, onset, contour), [
        ("frame_thresh_0", {"frame_thresh": 0.0}),
        ("default", {}),
    ], []

    # A note longer than numpy's 8192-element reduction buffer.
    n = 9000
    note, onset, contour = zeros(n)
    note[:, 40] = rng.uniform(0.3001, 1.0, n).astype(np.float32)
    onset[1, 40] = 0.9
    contour[:, 3 * 40 + 1] = 0.7
    # Pitch bends would add one value per frame to the fixture without testing more.
    yield "long_note", (note, onset, contour), [
        ("no_melodia", {"melodia_trick": False, "include_pitch_bends": False}),
        ("melodia_only", {"infer_onsets": False, "include_pitch_bends": False}),
    ], []

    # Invalid activations, which the model never outputs but a custom backend could.
    note, onset, contour = zeros(80)
    note[20:60, 40] = 0.8
    note[30, 40] = np.nan
    onset[20, 40] = 0.9
    yield "nan_note", (note, onset, contour), [
        ("no_infer_onsets", {"infer_onsets": False}),
        ("default", {}),
    ], []

    note, onset, contour = zeros(80)
    note[20:60, 40] = 0.8
    onset[20, 40] = 0.9
    onset[50, 60] = np.nan
    yield "nan_onset", (note, onset, contour), [("default", {}), ("no_infer_onsets", {"infer_onsets": False})], []

    note, onset, contour = zeros(80)
    note[20:60, 40] = 1.5
    onset[20, 40] = 0.9
    yield "amplitude_above_1", (note, onset, contour), [("default", {})], []

    # Low frames inside a note (fewer than energy_tol in a row) pull its mean below 0.
    note, onset, contour = zeros(80)
    note[20:60, 40] = 0.5
    for start in range(22, 58, 11):
        note[start : start + 10, 40] = -5.0
    onset[20, 40] = 0.9
    yield "amplitude_below_0", (note, onset, contour), [("no_melodia", {"melodia_trick": False})], []


def main():
    import tempfile

    from basic_pitch import note_creation as nc

    out_dir = pathlib.Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = pathlib.Path(tempfile.mkdtemp())
    for name, (note, onset, contour), runs, tolerances in cases():
        out = {"note": note.astype(np.float32), "onset": onset.astype(np.float32), "contour": contour.astype(np.float32)}
        doc = {
            "n_frames": int(note.shape[0]),
            "note": sparse(out["note"]),
            "onset": sparse(out["onset"]),
            "contour": sparse(out["contour"]),
            "runs": [],
        }
        first_poly = None
        for run_name, overrides in runs:
            result, poly = run(nc, out, run_name, overrides, tmp)
            doc["runs"].append(result)
            if first_poly is None:
                first_poly = poly
        if tolerances:
            doc["bend_tolerances"] = [
                {
                    "tolerance": tol,
                    "notes": encode_events(first_poly),
                    "bends": [e[4] and [int(b) for b in e[4]] for e in nc.get_pitch_bends(out["contour"], first_poly, n_bins_tolerance=tol)],
                }
                for tol in tolerances
            ]
        with open(out_dir / f"{name}.json", "w") as f:
            json.dump(doc, f, indent=None, separators=(",", ":"), allow_nan=False)
        errors = sum("error" in r or "error" in r.get("midi", {}) for r in doc["runs"])
        notes = sum(len(r.get("events", r["polyphonic"])) for r in doc["runs"])
        print(f"{name}: {note.shape[0]} frames, {len(runs)} runs, {notes} notes, {errors} expected errors")


if __name__ == "__main__":
    main()
