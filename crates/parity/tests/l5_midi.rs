// Copyright 2026 Adrien Thevon
// SPDX-License-Identifier: Apache-2.0

//! L5: MIDI events, compared after parsing (ticks, types and values), as pretty_midi
//! writes them.

use basic_pitch::midi::note_events_to_midi_bytes;
use basic_pitch::notes::model_output_to_notes;
use parity::{
    CASES, CONFIGS, Report, Variant, config_fixture, midi_to_json, params, python_posteriorgrams,
};

#[test]
fn midi_events_match_pretty_midi() {
    let mut report = Report::new("L5 MIDI");
    for variant in Variant::ALL {
        let (mut files, mut messages) = (0, 0);
        for case in CASES {
            let output = python_posteriorgrams(case, variant);
            for config in CONFIGS {
                let fixture = config_fixture(case, variant, config);
                let (note_params, midi_params) = params(&fixture);
                let events = model_output_to_notes(&output, &note_params).expect("events");
                let bytes = note_events_to_midi_bytes(&events, &midi_params).expect("midi");
                let ours = midi_to_json(&bytes);
                let theirs = &fixture["midi"];
                assert_eq!(
                    ours["type"], theirs["type"],
                    "{variant:?} {case} {config}: format"
                );
                assert_eq!(
                    ours["ticks_per_beat"], theirs["ticks_per_beat"],
                    "{variant:?} {case} {config}: resolution"
                );
                let (ot, tt) = (
                    ours["tracks"].as_array().unwrap(),
                    theirs["tracks"].as_array().unwrap(),
                );
                assert_eq!(
                    ot.len(),
                    tt.len(),
                    "{variant:?} {case} {config}: track count"
                );
                for (i, (o, t)) in ot.iter().zip(tt).enumerate() {
                    let (o, t) = (o.as_array().unwrap(), t.as_array().unwrap());
                    for (j, (a, b)) in o.iter().zip(t).enumerate() {
                        assert_eq!(a, b, "{variant:?} {case} {config}: track {i} event {j}");
                    }
                    assert_eq!(
                        o.len(),
                        t.len(),
                        "{variant:?} {case} {config}: track {i} length"
                    );
                    messages += o.len();
                }
                files += 1;
            }
        }
        report.line(format!(
            "{:<6}: {files} files, {messages} events identical",
            variant.dir()
        ));
    }
    report.print();
}
