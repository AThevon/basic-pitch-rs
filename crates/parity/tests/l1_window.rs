// Copyright 2026 Adrien Thevon
// SPDX-License-Identifier: Apache-2.0

//! L1: audio windows, bit for bit.

use basic_pitch::window::{window_count, windows};
use parity::{CASES, Report, Variant, WINDOW_CASES, audio, fixtures_dir, meta, read_f32};

#[test]
fn windows_match_the_model_input_of_python() {
    let mut report = Report::new("L1 windows");
    for case in WINDOW_CASES {
        let expected = read_f32(&fixtures_dir().join(case).join("model_input.npy"));
        let audio = audio(case);
        let ours: Vec<Vec<f32>> = windows(&audio).collect();
        assert_eq!(ours.len(), expected.shape()[0], "{case}: window count");
        for (i, (w, e)) in ours.iter().zip(expected.outer_iter()).enumerate() {
            assert_eq!(w.len(), e.len(), "{case}: window {i} length");
            let same = w
                .iter()
                .zip(e.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits());
            assert!(same, "{case}: window {i} differs");
        }
        report.line(format!("{case}: {} windows bit-identical", ours.len()));
    }
    report.print();
}

#[test]
fn window_counts_match_every_case() {
    let meta = meta(Variant::V040);
    for case in CASES {
        let n = audio(case).len();
        let m = &meta["cases"][case];
        assert_eq!(
            n as u64,
            m["original_length_22050"].as_u64().unwrap(),
            "{case}: length"
        );
        assert_eq!(
            window_count(n) as u64,
            m["n_windows"].as_u64().unwrap(),
            "{case}: windows"
        );
    }
}
