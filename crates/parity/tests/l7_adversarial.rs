// Copyright 2026 Adrien Thevon
// SPDX-License-Identifier: Apache-2.0

//! L7: the post-processing on synthetic posteriorgrams built to reach the corners of
//! `note_creation.py` (`reference/generate_adversarial.py`): thresholds at 0 and 1, ties,
//! frequency bounds that become negative numpy indices, very long notes, edge keys,
//! extreme tempos, custom pitch bend tolerances, NaN activations, and notes that
//! `pretty_midi` refuses to write. Where Python raises, Rust must return an error.

use std::path::Path;

use basic_pitch::midi::note_events_to_midi_bytes;
use basic_pitch::notes::{
    FrameNote, NoteEvent, get_pitch_bends, model_output_to_notes, output_to_notes_polyphonic,
};
use basic_pitch::{Posteriorgram, Posteriorgrams};
use parity::{Report, fixtures_dir, midi_to_json, params, read_json};
use serde_json::Value;

/// A float of the fixture: a JSON number, or "NaN" / "Infinity" / "-Infinity".
fn float(v: &Value) -> f64 {
    match v {
        Value::Number(n) => n.as_f64().expect("finite number"),
        Value::String(s) => match s.as_str() {
            "NaN" => f64::NAN,
            "Infinity" => f64::INFINITY,
            "-Infinity" => f64::NEG_INFINITY,
            other => panic!("unexpected float {other:?}"),
        },
        other => panic!("unexpected float {other}"),
    }
}

#[allow(
    clippy::cast_possible_truncation,
    reason = "the fixture holds float32 values"
)]
fn matrix(v: &Value, n_frames: usize, n_bins: usize) -> Posteriorgram {
    let mut m = Posteriorgram::zeros(n_frames, n_bins);
    m.as_mut_slice().fill(float(&v["fill"]) as f32);
    for e in v["entries"].as_array().expect("entries") {
        let (t, b) = (index(&e[0]), index(&e[1]));
        m.as_mut_slice()[t * n_bins + b] = float(&e[2]) as f32;
    }
    m
}

fn index(v: &Value) -> usize {
    usize::try_from(v.as_u64().expect("index")).expect("index fits")
}

fn same_f32(a: f32, b: f64) -> bool {
    #[allow(clippy::cast_possible_truncation, reason = "float32 values")]
    let b = b as f32;
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

fn same_f64(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

fn bends(v: &Value) -> Option<Vec<i32>> {
    v.as_array().map(|b| {
        b.iter()
            .map(|x| i32::try_from(x.as_i64().expect("bend")).expect("bend fits"))
            .collect()
    })
}

fn check_polyphonic(ours: &[FrameNote], theirs: &Value, ctx: &str) {
    let theirs = theirs.as_array().expect("polyphonic");
    assert_eq!(ours.len(), theirs.len(), "{ctx}: note count");
    for (i, (o, t)) in ours.iter().zip(theirs).enumerate() {
        assert_eq!(
            (o.start_frame, o.end_frame, u64::from(o.pitch)),
            (index(&t[0]), index(&t[1]), t[2].as_u64().expect("pitch")),
            "{ctx}: note {i}"
        );
        assert!(
            same_f32(o.amplitude, float(&t[3])),
            "{ctx}: note {i} amplitude {} vs {}",
            o.amplitude,
            t[3]
        );
    }
}

fn check_events(ours: &[NoteEvent], theirs: &Value, ctx: &str) {
    let theirs = theirs.as_array().expect("events");
    assert_eq!(ours.len(), theirs.len(), "{ctx}: event count");
    for (i, (o, t)) in ours.iter().zip(theirs).enumerate() {
        assert!(
            same_f64(o.start_s, float(&t[0])) && same_f64(o.end_s, float(&t[1])),
            "{ctx}: event {i} times"
        );
        assert_eq!(
            u64::from(o.pitch),
            t[2].as_u64().expect("pitch"),
            "{ctx}: event {i} pitch"
        );
        assert!(
            same_f32(o.amplitude, float(&t[3])),
            "{ctx}: event {i} amplitude"
        );
        assert_eq!(o.pitch_bends, bends(&t[4]), "{ctx}: event {i} bends");
    }
}

fn check_case(path: &Path, report: &mut Report) -> (usize, usize, usize) {
    let case = path.file_stem().unwrap().to_string_lossy().into_owned();
    let doc = read_json(path);
    let n = index(&doc["n_frames"]);
    let output = Posteriorgrams::new(
        matrix(&doc["note"], n, 88),
        matrix(&doc["onset"], n, 88),
        matrix(&doc["contour"], n, 264),
    )
    .expect("valid shapes");

    let (mut notes, mut errors, mut midi_files) = (0, 0, 0);
    for run in doc["runs"].as_array().expect("runs") {
        let ctx = format!("{case}/{}", run["name"].as_str().unwrap());
        let (note_params, midi_params) = params(run);

        let poly = output_to_notes_polyphonic(&output.note, &output.onset, &note_params)
            .unwrap_or_else(|e| panic!("{ctx}: {e}"));
        check_polyphonic(&poly, &run["polyphonic"], &ctx);
        notes += poly.len();

        let events = model_output_to_notes(&output, &note_params);
        if let Some(error) = run.get("error") {
            // Python fails while building its MIDI object: so must Rust, at the latest
            // when it writes the MIDI file.
            let failed = events
                .map(|events| note_events_to_midi_bytes(&events, &midi_params).is_err())
                .unwrap_or(true);
            assert!(failed, "{ctx}: Python raised {error}, Rust did not fail");
            errors += 1;
            continue;
        }
        let events = events.unwrap_or_else(|e| panic!("{ctx}: {e}"));
        check_events(&events, &run["events"], &ctx);

        let midi = note_events_to_midi_bytes(&events, &midi_params);
        if let Some(error) = run["midi"].get("error") {
            assert!(
                midi.is_err(),
                "{ctx}: pretty_midi raised {error}, Rust wrote the file"
            );
            errors += 1;
        } else {
            let bytes = midi.unwrap_or_else(|e| panic!("{ctx}: {e}"));
            assert_eq!(midi_to_json(&bytes), run["midi"], "{ctx}: MIDI events");
            midi_files += 1;
        }
    }

    for item in doc
        .get("bend_tolerances")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let tolerance = index(&item["tolerance"]);
        let frame_notes: Vec<FrameNote> = item["notes"]
            .as_array()
            .expect("notes")
            .iter()
            .map(|t| FrameNote {
                start_frame: index(&t[0]),
                end_frame: index(&t[1]),
                pitch: u8::try_from(t[2].as_u64().expect("pitch")).expect("pitch"),
                #[allow(clippy::cast_possible_truncation, reason = "float32 values")]
                amplitude: float(&t[3]) as f32,
            })
            .collect();
        let ours = get_pitch_bends(&output.contour, &frame_notes, tolerance).expect("bends");
        let theirs: Vec<Vec<i32>> = item["bends"]
            .as_array()
            .expect("bends")
            .iter()
            .map(|b| bends(b).unwrap_or_default())
            .collect();
        assert_eq!(
            ours, theirs,
            "{case}: pitch bends with tolerance {tolerance}"
        );
    }

    report.line(format!(
        "{case:<18} {n:>5} frames: {notes:>5} notes, {midi_files:>2} MIDI files, {errors} expected errors"
    ));
    (notes, midi_files, errors)
}

#[test]
fn adversarial_cases_match_python() {
    let dir = fixtures_dir().join("adversarial");
    let mut paths: Vec<_> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
        .map(|entry| entry.expect("entry").path())
        .filter(|p| p.extension().is_some_and(|e| e == "json"))
        .collect();
    paths.sort();
    assert!(paths.len() >= 15, "missing adversarial fixtures");

    let mut report = Report::new("L7 adversarial post-processing");
    let mut totals = (0, 0, 0);
    for path in &paths {
        let (notes, midi, errors) = check_case(path, &mut report);
        totals = (totals.0 + notes, totals.1 + midi, totals.2 + errors);
    }
    report.line(format!(
        "total: {} cases, {} notes, {} MIDI files identical, {} errors where Python fails",
        paths.len(),
        totals.0,
        totals.1,
        totals.2
    ));
    report.print();
}
