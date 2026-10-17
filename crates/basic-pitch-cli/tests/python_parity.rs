// Copyright 2026 Adrien Thevon
// SPDX-License-Identifier: Apache-2.0

//! The command-line tool against the official Python Basic Pitch, from the audio files.
//!
//! These tests start from `tests/fixtures/audio/*.wav`, like a user would, and compare:
//!
//! - the decoded, downmixed and resampled audio with `librosa.load` (`audio_22050.npy`);
//! - the notes, CSV and MIDI written by `basic-pitch-rs` with what basic-pitch wrote, for
//!   every parameter set of the fixtures and both trim modes.
//!
//! They run once per resampler compiled in (rubato, and soxr with `--features soxr`).
//! Run with `--nocapture` to see the measured differences.

use std::path::{Path, PathBuf};
use std::process::Command;

use basic_pitch_cli::audio;
use basic_pitch_cli::resample::Resampler;
use parity::{CASES, CONFIGS, Report, Variant, config_fixture, fixtures_dir, midi_to_json};
use serde_json::Value;

fn resamplers() -> Vec<Resampler> {
    let mut all = vec![Resampler::Rubato];
    if cfg!(feature = "soxr") {
        all.push(Resampler::Soxr);
    }
    all
}

fn wav(case: &str) -> PathBuf {
    fixtures_dir().join("audio").join(format!("{case}.wav"))
}

/// Largest absolute difference with librosa allowed for each resampler.
fn audio_tolerance(resampler: Resampler) -> f32 {
    match resampler {
        // The synthetic sawtooth has energy up to 22 kHz, where the anti-aliasing filters
        // differ most; music and voice stay around 1e-6.
        Resampler::Rubato => 1e-3,
        Resampler::Soxr => 1e-6,
    }
}

#[test]
fn decoded_audio_matches_librosa() {
    for resampler in resamplers() {
        let mut report = Report::new(&format!("audio vs librosa.load, {resampler}"));
        for case in CASES {
            let expected: Vec<f32> = parity::audio(case);
            let loaded = audio::load(&wav(case), resampler).unwrap();
            assert_eq!(loaded.samples.len(), expected.len(), "{case}: length");
            let (max, rms) = diff(&loaded.samples, &expected);
            let used = loaded.resampler.map_or("none (22050 Hz)", |r| r.name());
            report.line(format!(
                "{case:<12} {} Hz, {} ch, resampler {used:<15} max |diff| {max:.2e}, rms {rms:.2e}",
                loaded.source.sample_rate, loaded.source.channels
            ));
            if loaded.source.sample_rate == 22_050 {
                // No resampling: decoding and downmixing must be bit-exact.
                assert!(
                    loaded
                        .samples
                        .iter()
                        .zip(&expected)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "{case}: decoding is not bit-identical to libsndfile"
                );
            }
            assert!(
                max <= audio_tolerance(resampler),
                "{case}: {resampler} differs from librosa by {max:e}"
            );
        }
        report.print();
    }
}

#[allow(clippy::cast_precision_loss, reason = "statistics")]
fn diff(a: &[f32], b: &[f32]) -> (f32, f64) {
    let max = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max);
    let sum: f64 = a.iter().zip(b).map(|(x, y)| f64::from(x - y).powi(2)).sum();
    (max, (sum / a.len().max(1) as f64).sqrt())
}

/// Command-line flags equivalent to the parameters of a fixture.
fn flags(fixture: &Value) -> Vec<String> {
    let p = &fixture["params"];
    let num = |k: &str| {
        p[k].as_f64()
            .unwrap_or_else(|| panic!("params.{k}"))
            .to_string()
    };
    let on = |k: &str| p[k].as_bool().unwrap_or_else(|| panic!("params.{k}"));
    let mut args = vec![
        "--onset-threshold".into(),
        num("onset_thresh"),
        "--frame-threshold".into(),
        num("frame_thresh"),
        "--minimum-note-length".into(),
        num("min_note_len_ms"),
        "--midi-tempo".into(),
        num("midi_tempo"),
    ];
    for (key, flag) in [
        ("min_freq", "--minimum-frequency"),
        ("max_freq", "--maximum-frequency"),
    ] {
        if let Some(hz) = p[key].as_f64() {
            args.extend([flag.into(), hz.to_string()]);
        }
    }
    for (key, flag, when) in [
        ("melodia_trick", "--no-melodia", false),
        ("infer_onsets", "--no-infer-onsets", false),
        ("include_pitch_bends", "--no-pitch-bends", false),
        ("multiple_pitch_bends", "--multiple-pitch-bends", true),
    ] {
        if on(key) == when {
            args.push(flag.into());
        }
    }
    args
}

/// One expected note: the fixture's frame event and timed event, which share an order.
#[derive(Debug)]
struct Expected {
    start_frame: u64,
    end_frame: u64,
    start_s: f64,
    end_s: f64,
    pitch: u64,
    amplitude: f64,
    bends: Option<Vec<i64>>,
}

fn expected_notes(fixture: &Value) -> Vec<Expected> {
    let frames = fixture["frame_events"].as_array().unwrap();
    let timed = fixture["events"].as_array().unwrap();
    assert_eq!(frames.len(), timed.len());
    frames
        .iter()
        .zip(timed)
        .map(|(f, t)| Expected {
            start_frame: f[0].as_u64().unwrap(),
            end_frame: f[1].as_u64().unwrap(),
            start_s: t[0].as_f64().unwrap(),
            end_s: t[1].as_f64().unwrap(),
            pitch: f[2].as_u64().unwrap(),
            amplitude: f[3].as_f64().unwrap(),
            bends: f[4]
                .as_array()
                .map(|b| b.iter().map(|x| x.as_i64().unwrap()).collect()),
        })
        .collect()
}

/// The CSV basic-pitch writes for the fixture's events, in its order.
#[allow(
    clippy::cast_possible_truncation,
    reason = "amplitudes are float32 values"
)]
fn expected_csv(fixture: &Value) -> String {
    let py = |x: f64| {
        let s = x.to_string();
        if s.contains('.') { s } else { s + ".0" }
    };
    let mut csv = String::from("start_time_s,end_time_s,pitch_midi,velocity,pitch_bend\r\n");
    for e in fixture["events"].as_array().unwrap() {
        let amplitude = e[3].as_f64().unwrap() as f32;
        let velocity = (127.0_f32 * amplitude).round_ties_even();
        csv += &format!(
            "{},{},{},{velocity}",
            py(e[0].as_f64().unwrap()),
            py(e[1].as_f64().unwrap()),
            e[2]
        );
        for bend in e[4].as_array().into_iter().flatten() {
            csv += &format!(",{bend}");
        }
        csv += "\r\n";
    }
    csv
}

#[derive(Default)]
struct Tally {
    notes: usize,
    identical_notes: usize,
    max_amplitude_diff: f64,
    velocity_diffs: usize,
    files: usize,
    identical_csv: usize,
    identical_midi: usize,
    /// CSV or MIDI files that only differ by a velocity of plus or minus one, or by the
    /// order of notes whose energies are nearly equal.
    near_identical: Vec<String>,
    mismatches: Vec<String>,
}

fn run_cli(out: &Path, resampler: Resampler, variant: Variant, extra: &[String]) {
    let status = Command::new(env!("CARGO_BIN_EXE_basic-pitch-rs"))
        .args(CASES.iter().map(|c| wav(c)))
        .arg("--output-dir")
        .arg(out)
        .args([
            "--midi",
            "--json",
            "--csv",
            "--quiet",
            "--resampler",
            resampler.name(),
        ])
        .args(["--trim", variant.dir()])
        .args(extra)
        .status()
        .expect("the binary runs");
    assert!(status.success(), "basic-pitch-rs failed with {extra:?}");
}

/// CSV rows with the velocity set aside, sorted: equal when two files hold the same notes
/// in any order.
fn csv_rows(csv: &str) -> Vec<(String, i64)> {
    let mut rows: Vec<(String, i64)> = csv
        .split("\r\n")
        .skip(1)
        .filter(|r| !r.is_empty())
        .map(|r| {
            let mut fields: Vec<&str> = r.split(',').collect();
            let velocity = fields.remove(3).parse().unwrap();
            (fields.join(","), velocity)
        })
        .collect();
    rows.sort();
    rows
}

fn velocities_close(a: &[(String, i64)], b: &[(String, i64)]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(x, y)| x.0 == y.0 && (x.1 - y.1).abs() <= 1)
}

/// MIDI events with the note velocities set to zero, and the velocities apart.
fn midi_without_velocities(midi: &Value) -> (Value, Vec<i64>) {
    let mut midi = midi.clone();
    let mut velocities = Vec::new();
    for track in midi["tracks"].as_array_mut().unwrap() {
        for event in track.as_array_mut().unwrap() {
            let message = &mut event[1];
            if message["type"] == "note_on" {
                velocities.push(message["velocity"].as_i64().unwrap());
                message["velocity"] = 0.into();
            }
        }
    }
    (midi, velocities)
}

fn check_outputs(out: &Path, variant: Variant, config: &str, tally: &mut Tally) {
    for case in CASES {
        let fixture = config_fixture(case, variant, config);
        let label = format!("{case}/{}/{config}", variant.dir());
        let read =
            |ext: &str| std::fs::read(out.join(format!("{case}_basic_pitch.{ext}"))).unwrap();

        let doc: Value = serde_json::from_slice(&read("json")).unwrap();
        let notes = doc["notes"].as_array().unwrap();
        let mut expected = expected_notes(&fixture);
        expected.sort_by_key(|e| (e.start_frame, e.pitch, e.end_frame));
        tally.notes += expected.len().max(notes.len());
        if notes.len() != expected.len() {
            tally.mismatches.push(format!(
                "{label}: {} notes, Python has {}",
                notes.len(),
                expected.len()
            ));
        }
        for (n, e) in notes.iter().zip(&expected) {
            let bends = n["pitch_bends"]
                .as_array()
                .map(|b| b.iter().map(|x| x.as_i64().unwrap()).collect::<Vec<_>>());
            let amplitude = n["amplitude"].as_f64().unwrap();
            let same = n["start_frame"].as_u64() == Some(e.start_frame)
                && n["end_frame"].as_u64() == Some(e.end_frame)
                && n["start_s"].as_f64().map(f64::to_bits) == Some(e.start_s.to_bits())
                && n["end_s"].as_f64().map(f64::to_bits) == Some(e.end_s.to_bits())
                && n["pitch"].as_u64() == Some(e.pitch)
                && bends == e.bends;
            if same {
                tally.identical_notes += 1;
                tally.max_amplitude_diff = tally
                    .max_amplitude_diff
                    .max((amplitude - e.amplitude).abs());
                #[allow(clippy::cast_possible_truncation, reason = "a float32 amplitude")]
                let velocity = (127.0_f32 * e.amplitude as f32).round_ties_even();
                if n["velocity"].as_f64() != Some(f64::from(velocity)) {
                    tally.velocity_diffs += 1;
                }
            } else {
                tally.mismatches.push(format!("{label}: {n} vs {e:?}"));
            }
        }

        tally.files += 1;
        let csv = String::from_utf8(read("csv")).unwrap();
        let expected_csv = expected_csv(&fixture);
        if csv == expected_csv {
            tally.identical_csv += 1;
        } else if velocities_close(&csv_rows(&csv), &csv_rows(&expected_csv)) {
            tally.near_identical.push(format!("{label}: CSV"));
        } else {
            tally.mismatches.push(format!("{label}: CSV differs"));
        }

        let midi = midi_to_json(&read("mid"));
        if midi == fixture["midi"] {
            tally.identical_midi += 1;
        } else {
            let (events, velocities) = midi_without_velocities(&midi);
            let (expected_events, expected_velocities) = midi_without_velocities(&fixture["midi"]);
            if events == expected_events
                && velocities
                    .iter()
                    .zip(&expected_velocities)
                    .all(|(a, b)| (a - b).abs() <= 1)
            {
                tally.near_identical.push(format!("{label}: MIDI"));
            } else {
                tally
                    .mismatches
                    .push(format!("{label}: MIDI events differ"));
            }
        }
    }
}

/// Every output of the tool, for 6 files x 8 parameter sets x 2 trim modes, against
/// basic-pitch's.
///
/// The notes (frames, times, pitches and pitch bends) must be identical with either
/// resampler. Amplitudes are not bit-identical: the model reacts to differences of 1e-7
/// in quiet passages (its input is a log spectrogram), and tract differs from ONNX
/// Runtime by about 1e-6. An amplitude within 1e-4 of a rounding boundary can then move a
/// velocity by one, and two melodia notes of nearly equal energy can swap places in the
/// CSV, which keeps Python's unsorted order. Those differences are reported and allowed;
/// anything else fails.
#[test]
fn notes_csv_and_midi_match_python() {
    let mut failures = Vec::new();
    for resampler in resamplers() {
        let tmp = tempfile::tempdir().unwrap();
        let runs: Vec<(Variant, &str, PathBuf)> = Variant::ALL
            .iter()
            .flat_map(|&v| CONFIGS.iter().map(move |&c| (v, c)))
            .map(|(v, c)| (v, c, tmp.path().join(format!("{}-{c}", v.dir()))))
            .collect();

        // Each run transcribes the six files; running them in parallel keeps the test fast
        // in debug builds.
        std::thread::scope(|scope| {
            for (variant, config, out) in &runs {
                scope.spawn(move || {
                    let fixture = config_fixture(CASES[0], *variant, config);
                    run_cli(out, resampler, *variant, &flags(&fixture));
                });
            }
        });

        let mut tally = Tally::default();
        for (variant, config, out) in &runs {
            check_outputs(out, *variant, config, &mut tally);
        }

        let mut report = Report::new(&format!(
            "basic-pitch-rs vs Python basic-pitch, from the audio files, {resampler}"
        ));
        report.line(format!(
            "notes: {}/{} identical (frames, times, pitch, pitch bends); max amplitude diff {:.2e}; {} velocities off by one",
            tally.identical_notes, tally.notes, tally.max_amplitude_diff, tally.velocity_diffs
        ));
        report.line(format!(
            "CSV files: {}/{} identical; MIDI files: {}/{} with identical events",
            tally.identical_csv, tally.files, tally.identical_midi, tally.files
        ));
        for m in &tally.near_identical {
            report.line(format!("velocity +-1 or note order only: {m}"));
        }
        for m in tally.mismatches.iter().take(20) {
            report.line(format!("DIFFERENT: {m}"));
        }
        report.print();

        if !tally.mismatches.is_empty() {
            failures.push(format!(
                "{resampler}: {} differences",
                tally.mismatches.len()
            ));
        }
    }
    assert!(failures.is_empty(), "{failures:?}");
}
