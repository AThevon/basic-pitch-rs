// Copyright 2026 Adrien Thevon
// SPDX-License-Identifier: Apache-2.0

//! L2: raw model outputs of the inference backends against onnxruntime 1.31 (Python).

use basic_pitch::model::{Backend, TractBackend, run_windows};
use parity::{
    CASES, Report, WINDOW_CASES, audio, fixtures_dir, max_abs_diff, raw_outputs, read_f32,
};

/// Tolerance of the native tract backend against onnxruntime.
///
/// tract picks its matrix kernels at run time from the CPU. With fused multiply-add
/// kernels (Apple M2, for example) the outputs stay within 1.3e-6 of onnxruntime; the
/// kernels selected on GitHub's runners give up to 2.4e-4, the same order as the
/// WebAssembly build. The notes are checked against Python in L6, which explains any
/// difference by a threshold margin.
const TRACT_TOLERANCE: f32 = 1e-3;

fn check_backend(name: &str, backend: &dyn Backend, tolerance: f32) -> f32 {
    let mut report = Report::new(&format!(
        "L2 {name} vs onnxruntime (max |diff| note / onset / contour)"
    ));
    let mut worst = 0.0_f32;

    // The exact windows fed to Python's model, where they were saved.
    let mut worst_input = 0.0_f32;
    for case in WINDOW_CASES {
        let input = read_f32(&fixtures_dir().join(case).join("model_input.npy"));
        let expected = raw_outputs(case);
        for (i, w) in input.outer_iter().enumerate() {
            let w: Vec<f32> = w.iter().copied().collect();
            let out = backend.predict_window(&w).expect("inference");
            let rows = i * 172..(i + 1) * 172;
            for (ours, theirs) in [
                (&out.note, &expected.note),
                (&out.onset, &expected.onset),
                (&out.contour, &expected.contour),
            ] {
                let bins = theirs.n_bins();
                let slice = &theirs.as_slice()[rows.start * bins..rows.end * bins];
                let d = ours
                    .as_slice()
                    .iter()
                    .zip(slice)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0, f32::max);
                assert!(
                    d <= tolerance,
                    "{name} {case} window {i}: {d:e} > {tolerance:e}"
                );
                worst_input = worst_input.max(d);
            }
        }
    }
    report.line(format!(
        "saved model inputs (short, synth): max {worst_input:.2e}"
    ));
    worst = worst.max(worst_input);

    // Every case, through the crate's own windowing.
    for case in CASES {
        let ours = run_windows(backend, &audio(case)).expect("inference");
        let theirs = raw_outputs(case);
        let d = [
            max_abs_diff(&ours.note, &theirs.note),
            max_abs_diff(&ours.onset, &theirs.onset),
            max_abs_diff(&ours.contour, &theirs.contour),
        ];
        report.line(format!(
            "{case:<12} {:>2} windows: {:.2e} / {:.2e} / {:.2e}",
            ours.n_frames() / 172,
            d[0],
            d[1],
            d[2]
        ));
        for x in d {
            assert!(x <= tolerance, "{name} {case}: {x:e} > {tolerance:e}");
            worst = worst.max(x);
        }
    }
    report.line(format!(
        "max over all cases: {worst:.2e} (tolerance {tolerance:.0e})"
    ));
    report.print();
    worst
}

#[test]
fn tract_matches_onnxruntime() {
    let backend = TractBackend::new().expect("model loads");
    check_backend("tract", &backend, TRACT_TOLERANCE);
}

#[cfg(feature = "ort")]
#[test]
fn ort_matches_onnxruntime_bit_for_bit() {
    use basic_pitch::model::OrtBackend;

    let backend = OrtBackend::new().expect("ONNX Runtime starts");
    let worst = check_backend("ort", &backend, TRACT_TOLERANCE);
    // Same engine and version as Python, on the CPU: identical outputs are expected.
    // A non-zero value here means another execution provider (CoreML) was selected.
    assert!(worst < 1e-6, "ort drifted from onnxruntime: {worst:e}");
}
