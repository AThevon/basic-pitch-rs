// Copyright 2026 Adrien Thevon
// SPDX-License-Identifier: Apache-2.0

//! L6: the whole pipeline with the tract backend, from the 22 050 Hz audio to notes.
//!
//! tract and onnxruntime differ by 1e-6 to 5e-4 on the posteriorgrams, depending on the
//! matrix kernels tract picks for the CPU. Notes can then only differ in three ways, each
//! bounded by that difference: an amplitude (a mean of note activations), a pitch bend
//! frame whose two best contour bins are nearly equal, or a decision of the decoder that
//! sits within that distance of a threshold. Any other difference fails the test.

use basic_pitch::notes::{NoteEvent, NoteParams, infer_onsets};
use basic_pitch::{BasicPitch, Posteriorgrams};
use parity::{
    CASES, CONFIGS, Report, Variant, audio, config_fixture, max_abs_diff, params,
    python_posteriorgrams,
};
use serde_json::Value;

const N_BINS: usize = 88;

fn python_bends(e: &Value) -> Option<Vec<i32>> {
    e[4].as_array().map(|b| {
        b.iter()
            .map(|x| i32::try_from(x.as_i64().unwrap()).unwrap())
            .collect()
    })
}

/// Same frames, times and pitch, and an amplitude within `amp_tol`.
fn same_but_bends(a: &NoteEvent, e: &Value, amp_tol: f64) -> bool {
    (a.start_s - e[0].as_f64().unwrap()).abs() < 1e-9
        && (a.end_s - e[1].as_f64().unwrap()).abs() < 1e-9
        && u64::from(a.pitch) == e[2].as_u64().unwrap()
        && (f64::from(a.amplitude) - e[3].as_f64().unwrap()).abs() <= amp_tol
}

fn same_note(a: &NoteEvent, e: &Value, amp_tol: f64) -> bool {
    same_but_bends(a, e, amp_tol) && a.pitch_bends == python_bends(e)
}

/// The pitch bends differ in exactly one frame, by one bin.
fn one_bend_frame_apart(a: &NoteEvent, e: &Value) -> bool {
    let (Some(ours), Some(theirs)) = (a.pitch_bends.as_deref(), python_bends(e)) else {
        return false;
    };
    let diffs: Vec<i32> = ours
        .iter()
        .zip(&theirs)
        .filter(|(x, y)| x != y)
        .map(|(x, y)| x - y)
        .collect();
    ours.len() == theirs.len() && diffs.len() == 1 && diffs[0].abs() == 1
}

/// Smallest distance to a decoder threshold among the activations that decide the notes
/// around `pitch` (its bin and both neighbours), or anywhere when `pitch` is `None`.
fn threshold_margin(p: &Posteriorgrams, params: &NoteParams, pitch: Option<u8>) -> f64 {
    let bins = match pitch {
        Some(pitch) => {
            let bin = usize::from(pitch) - 21;
            bin.saturating_sub(1)..(bin + 2).min(N_BINS)
        }
        None => 0..N_BINS,
    };
    let onsets: Vec<f64> = if params.infer_onsets {
        infer_onsets(&p.onset, &p.note).unwrap()
    } else {
        p.onset.as_slice().iter().map(|&x| f64::from(x)).collect()
    };
    let mut margin = f64::INFINITY;
    for t in 0..p.n_frames() {
        for b in bins.clone() {
            let i = t * N_BINS + b;
            margin = margin.min((f64::from(p.note.as_slice()[i]) - params.frame_threshold).abs());
            if onsets[i].is_finite() {
                margin = margin.min((onsets[i] - params.onset_threshold).abs());
            }
        }
    }
    margin
}

#[test]
fn tract_pipeline_reproduces_python_notes() {
    let mut report = Report::new("L6 end to end with tract (audio_22050 -> notes)");
    let mut total = (0, 0, 0, 0); // identical notes, python notes, explained differences, velocity mismatches
    for variant in Variant::ALL {
        let transcriber = BasicPitch::new()
            .expect("model")
            .with_trim_mode(variant.trim());
        for case in CASES {
            let ours = transcriber.predict(&audio(case)).expect("inference");
            let python = python_posteriorgrams(case, variant);
            let delta = [
                max_abs_diff(&ours.note, &python.note),
                max_abs_diff(&ours.onset, &python.onset),
                max_abs_diff(&ours.contour, &python.contour),
            ];
            let delta_max = f64::from(delta.into_iter().fold(0.0, f32::max));
            // An amplitude is a mean of note activations, each within delta_max.
            let amp_tol = delta_max.max(1e-5);
            // With default parameters, the closest activation to a threshold anywhere:
            // when it is farther than the posteriorgram difference, no threshold decision
            // can flip.
            let margin = threshold_margin(&python, &NoteParams::default(), None);
            let mut line = format!(
                "{:<6} {case:<12} max |diff| {:.1e}/{:.1e}/{:.1e}, default threshold margin {margin:.1e}; notes per config",
                variant.dir(),
                delta[0],
                delta[1],
                delta[2]
            );
            for config in CONFIGS {
                let fixture = config_fixture(case, variant, config);
                let (note_params, _) = params(&fixture);
                let theirs = fixture["events"].as_array().unwrap();
                let notes = basic_pitch::notes::model_output_to_notes(&ours, &note_params).unwrap();
                let identical = notes.len() == theirs.len()
                    && notes
                        .iter()
                        .zip(theirs)
                        .all(|(a, e)| same_note(a, e, amp_tol));
                let n_same = notes
                    .iter()
                    .zip(theirs)
                    .filter(|(a, e)| same_note(a, e, amp_tol))
                    .count();
                // A velocity may only move when Python's amplitude is within amp_tol of a
                // rounding boundary.
                let velocity_mismatch = notes
                    .iter()
                    .zip(theirs)
                    .filter(|(a, e)| {
                        let scaled = 127.0 * e[3].as_f64().unwrap();
                        #[allow(clippy::cast_possible_truncation, reason = "a float32 amplitude")]
                        let v = (127.0_f32 * e[3].as_f64().unwrap() as f32).round_ties_even();
                        let near_boundary =
                            ((scaled - scaled.floor()) - 0.5).abs() <= 127.0 * amp_tol;
                        same_note(a, e, amp_tol) && f32::from(a.velocity()) != v && !near_boundary
                    })
                    .count();
                let bend_frames = notes
                    .iter()
                    .zip(theirs)
                    .filter(|(a, e)| same_but_bends(a, e, amp_tol) && one_bend_frame_apart(a, e))
                    .count();
                if bend_frames > 0 {
                    line.push_str(&format!(
                        " ({bend_frames} with one pitch bend frame off by one)"
                    ));
                }
                total.0 += n_same;
                total.1 += theirs.len();
                total.3 += velocity_mismatch;
                line.push_str(&format!(" {n_same}/{}", theirs.len()));
                if !identical {
                    // Every pitch involved in a difference must have an activation within
                    // the posteriorgram difference of a threshold.
                    let differing = notes
                        .iter()
                        .zip(theirs)
                        .filter(|(a, e)| {
                            !same_note(a, e, amp_tol)
                                && !(same_but_bends(a, e, amp_tol) && one_bend_frame_apart(a, e))
                        })
                        .flat_map(|(a, e)| [a.pitch, u8::try_from(e[2].as_u64().unwrap()).unwrap()])
                        .chain(notes.iter().skip(theirs.len()).map(|a| a.pitch))
                        .chain(
                            theirs
                                .iter()
                                .skip(notes.len())
                                .map(|e| u8::try_from(e[2].as_u64().unwrap()).unwrap()),
                        );
                    for pitch in differing {
                        line.push_str(&format!(" [pitch {pitch} differs]"));
                        let margin = threshold_margin(&python, &note_params, Some(pitch));
                        assert!(
                            margin <= delta_max,
                            "{variant:?} {case} {config}: pitch {pitch} differs, but its closest activation is {margin:e} from a threshold (posteriorgram diff {delta_max:e})"
                        );
                        line.push_str(&format!(
                            " (pitch {pitch}: margin {margin:.1e} <= {delta_max:.1e})"
                        ));
                    }
                    total.2 += 1;
                }
            }
            report.line(line);
        }
    }
    report.line(format!(
        "identical notes {}/{}; configurations with an explained difference: {}; velocity mismatches on identical notes: {}",
        total.0, total.1, total.2, total.3
    ));
    report.print();
    assert_eq!(total.3, 0, "velocity mismatch");
}

/// Running the windows on several threads shares one tract plan between them: the
/// posteriorgrams must stay bit-identical to a single-threaded run.
#[test]
fn threads_give_bit_identical_posteriorgrams() {
    use std::num::NonZeroUsize;

    let single = BasicPitch::new().expect("model");
    let multi = BasicPitch::new()
        .expect("model")
        .with_threads(NonZeroUsize::new(4).unwrap());
    let mut report = Report::new("L6 threads");
    for case in ["pluck", "vocadito_14"] {
        let audio = audio(case);
        let a = single.predict(&audio).expect("predict");
        let b = multi.predict(&audio).expect("predict");
        for (x, y) in [
            (&a.note, &b.note),
            (&a.onset, &b.onset),
            (&a.contour, &b.contour),
        ] {
            assert!(
                parity::bit_identical(x, y),
                "{case}: threads changed the output"
            );
        }
        report.line(format!(
            "{case:<12}: {} frames, bit-identical on 1 and 4 threads",
            a.n_frames()
        ));
    }
    report.print();
}
