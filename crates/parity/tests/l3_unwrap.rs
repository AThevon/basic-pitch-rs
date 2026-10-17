// Copyright 2026 Adrien Thevon
// SPDX-License-Identifier: Apache-2.0

//! L3: unwrapping the window outputs, and frame times.

use basic_pitch::notes::model_frames_to_time;
use basic_pitch::window::{TrimMode, unwrap_output};
use basic_pitch::{Posteriorgram, Posteriorgrams};
use parity::{
    CASES, Report, Variant, audio, bit_identical, fixtures_dir, meta, raw_outputs, read_f64,
    unwrapped_v040,
};

fn unwrap(raw: &Posteriorgrams, n: usize, trim: TrimMode) -> Posteriorgrams {
    let u = |p: &Posteriorgram| unwrap_output(p, n, trim).expect("unwrap");
    Posteriorgrams::new(u(&raw.note), u(&raw.onset), u(&raw.contour)).expect("shapes")
}

#[test]
fn unwrap_v040_is_bit_identical() {
    let mut report = Report::new("L3 unwrap");
    let meta_main = meta(Variant::Main);
    for case in CASES {
        let n = audio(case).len();
        let raw = raw_outputs(case);
        let expected = unwrapped_v040(case);

        let v040 = unwrap(&raw, n, TrimMode::V0_4_0);
        assert!(bit_identical(&v040.note, &expected.note), "{case}: note");
        assert!(bit_identical(&v040.onset, &expected.onset), "{case}: onset");
        assert!(
            bit_identical(&v040.contour, &expected.contour),
            "{case}: contour"
        );

        let main = unwrap(&raw, n, TrimMode::Main);
        let n_main = meta_main["cases"][case]["n_frames"].as_u64().unwrap() as usize;
        assert_eq!(main.n_frames(), n_main, "{case}: main frame count");
        // main only keeps more frames at the end: the v0.4.0 output is a prefix.
        for (m, v) in [
            (&main.note, &v040.note),
            (&main.onset, &v040.onset),
            (&main.contour, &v040.contour),
        ] {
            let prefix = &m.as_slice()[..v.as_slice().len()];
            assert!(
                prefix
                    .iter()
                    .zip(v.as_slice())
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "{case}: prefix"
            );
        }
        report.line(format!(
            "{case:<12} v0.4.0 {:>4} frames bit-identical, main {:>4} frames",
            v040.n_frames(),
            main.n_frames()
        ));
    }
    report.print();
}

#[test]
fn frame_times_are_bit_identical() {
    for variant in Variant::ALL {
        for case in CASES {
            let expected = read_f64(
                &fixtures_dir()
                    .join(case)
                    .join(variant.dir())
                    .join("times.npy"),
            );
            let ours = model_frames_to_time(expected.len());
            assert!(
                ours.iter()
                    .zip(expected.iter())
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "{variant:?} {case}: times differ"
            );
        }
    }
}
