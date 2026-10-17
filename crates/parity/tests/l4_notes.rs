// Copyright 2026 Adrien Thevon
// SPDX-License-Identifier: Apache-2.0

//! L4: notes decoded from the Python posteriorgrams, for every case, configuration and
//! variant: frames, pitches and pitch bends exactly, in Python's order.

use basic_pitch::notes::{
    FrameNote, NoteEvent, get_pitch_bends, model_output_to_notes, output_to_notes_polyphonic,
};
use parity::{CASES, CONFIGS, Report, Variant, config_fixture, params, python_posteriorgrams};
use serde_json::Value;

/// Relative tolerance on amplitudes (numpy and Rust both sum in float32, pairwise).
const AMPLITUDE_RTOL: f64 = 1e-6;
/// Absolute tolerance on times, in seconds.
const TIME_ATOL: f64 = 1e-9;

#[derive(Default)]
struct Tally {
    notes: usize,
    bends: usize,
    amplitude_bit_exact: usize,
    max_amplitude_rel: f64,
    times_bit_exact: usize,
    max_time_diff: f64,
}

fn as_usize(v: &Value) -> usize {
    usize::try_from(v.as_u64().expect("integer")).expect("fits")
}

fn bends_of(v: &Value) -> Option<Vec<i32>> {
    v.as_array().map(|a| {
        a.iter()
            .map(|b| i32::try_from(b.as_i64().expect("integer bend")).expect("fits"))
            .collect()
    })
}

fn check_amplitude(t: &mut Tally, ours: f32, theirs: &Value, ctx: &str) {
    let theirs = theirs.as_f64().expect("amplitude");
    let rel = ((f64::from(ours) - theirs) / theirs).abs();
    assert!(
        rel <= AMPLITUDE_RTOL,
        "{ctx}: amplitude {ours} vs {theirs} (rel {rel:e})"
    );
    t.max_amplitude_rel = t.max_amplitude_rel.max(rel);
    if f64::from(ours).to_bits() == theirs.to_bits() {
        t.amplitude_bit_exact += 1;
    }
}

fn check_poly(ours: &[FrameNote], theirs: &Value, t: &mut Tally, ctx: &str) {
    let theirs = theirs.as_array().expect("polyphonic_events");
    assert_eq!(ours.len(), theirs.len(), "{ctx}: number of notes");
    for (i, (n, e)) in ours.iter().zip(theirs).enumerate() {
        let ctx = format!("{ctx} note {i}");
        assert_eq!(
            (n.start_frame, n.end_frame, usize::from(n.pitch)),
            (as_usize(&e[0]), as_usize(&e[1]), as_usize(&e[2])),
            "{ctx}: frames or pitch"
        );
        check_amplitude(t, n.amplitude, &e[3], &ctx);
    }
}

fn check_events(ours: &[NoteEvent], theirs: &Value, t: &mut Tally, ctx: &str) {
    let theirs = theirs.as_array().expect("events");
    assert_eq!(ours.len(), theirs.len(), "{ctx}: number of events");
    for (i, (n, e)) in ours.iter().zip(theirs).enumerate() {
        let ctx = format!("{ctx} event {i}");
        for (ours, theirs) in [(n.start_s, &e[0]), (n.end_s, &e[1])] {
            let theirs = theirs.as_f64().expect("time");
            let d = (ours - theirs).abs();
            assert!(d <= TIME_ATOL, "{ctx}: time {ours} vs {theirs}");
            t.max_time_diff = t.max_time_diff.max(d);
            if ours.to_bits() == theirs.to_bits() {
                t.times_bit_exact += 1;
            }
        }
        assert_eq!(usize::from(n.pitch), as_usize(&e[2]), "{ctx}: pitch");
        check_amplitude(t, n.amplitude, &e[3], &ctx);
        assert_eq!(n.pitch_bends, bends_of(&e[4]), "{ctx}: pitch bends");
        t.notes += 1;
        t.bends += n.pitch_bends.as_ref().map_or(0, Vec::len);
    }
}

#[test]
fn notes_match_python_exactly() {
    let mut report = Report::new("L4 notes from the Python posteriorgrams");
    let mut total = Tally::default();
    for variant in Variant::ALL {
        let mut tally = Tally::default();
        for case in CASES {
            let output = python_posteriorgrams(case, variant);
            for config in CONFIGS {
                let ctx = format!("{variant:?} {case} {config}");
                let fixture = config_fixture(case, variant, config);
                let (note_params, _) = params(&fixture);

                let poly = output_to_notes_polyphonic(&output.note, &output.onset, &note_params)
                    .expect("notes");
                check_poly(&poly, &fixture["polyphonic_events"], &mut tally, &ctx);

                let frame_events = fixture["frame_events"].as_array().expect("frame_events");
                let bends = if note_params.include_pitch_bends {
                    get_pitch_bends(&output.contour, &poly, 25)
                        .expect("bends")
                        .into_iter()
                        .map(Some)
                        .collect()
                } else {
                    vec![None; poly.len()]
                };
                for (i, (b, e)) in bends.iter().zip(frame_events).enumerate() {
                    assert_eq!(*b, bends_of(&e[4]), "{ctx} frame event {i}: bends");
                }

                let events = model_output_to_notes(&output, &note_params).expect("events");
                check_events(&events, &fixture["events"], &mut tally, &ctx);
            }
        }
        report.line(format!(
            "{:<6}: {} notes over 6 cases x 8 configs, {} pitch bend values; amplitudes bit-exact {}/{} (max rel {:.1e}); times bit-exact {}/{} (max {:.1e} s)",
            variant.dir(),
            tally.notes,
            tally.bends,
            tally.amplitude_bit_exact,
            2 * tally.notes,
            tally.max_amplitude_rel,
            tally.times_bit_exact,
            2 * tally.notes,
            tally.max_time_diff
        ));
        total.notes += tally.notes;
    }
    assert!(total.notes > 0, "no note compared");
    report.print();
}
