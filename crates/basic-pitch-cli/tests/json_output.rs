// Copyright 2026 Adrien Thevon
// SPDX-License-Identifier: Apache-2.0

//! The JSON output against its published schema
//! (`crates/basic-pitch-cli/schema/transcription.v1.schema.json`).

use std::process::Command;

use basic_pitch::model::MODEL_SHA256;
use basic_pitch_cli::output::{
    JSON_SCHEMA, JSON_SCHEMA_URL, JSON_SCHEMA_VERSION, JsonDocument, note_name,
};
use parity::fixtures_dir;
use serde_json::Value;

fn schema() -> Value {
    serde_json::from_str(JSON_SCHEMA).expect("the schema is valid JSON")
}

/// Runs `basic-pitch-rs <case>.wav --json --output - [extra]` and parses its stdout.
fn transcribe(case: &str, extra: &[&str]) -> Value {
    let out = Command::new(env!("CARGO_BIN_EXE_basic-pitch-rs"))
        .arg(fixtures_dir().join("audio").join(format!("{case}.wav")))
        .args(["--json", "--output", "-", "--quiet"])
        .args(extra)
        .output()
        .expect("the binary runs");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(out.stderr.is_empty(), "--quiet must keep stderr empty");
    serde_json::from_slice(&out.stdout).expect("stdout is one JSON document")
}

fn assert_valid(doc: &Value) {
    let validator = jsonschema::validator_for(&schema()).expect("the schema compiles");
    let errors: Vec<String> = validator
        .iter_errors(doc)
        .map(|e| format!("{} at {}", e, e.instance_path()))
        .collect();
    assert!(
        errors.is_empty(),
        "the output does not follow the schema: {errors:#?}"
    );
}

/// Fails if `value` has a field that the schema does not document.
fn assert_documented(value: &Value, node: &Value, root: &Value, path: &str) {
    let node = match node.get("$ref").and_then(Value::as_str) {
        Some(reference) => {
            let pointer = reference.strip_prefix('#').expect("local reference");
            root.pointer(pointer).expect("the reference resolves")
        }
        None => node,
    };
    match value {
        Value::Object(map) => {
            let properties = node["properties"]
                .as_object()
                .unwrap_or_else(|| panic!("{path}: the schema documents no properties"));
            for (key, child) in map {
                let child_schema = properties
                    .get(key)
                    .unwrap_or_else(|| panic!("{path}/{key} is not in the schema"));
                assert_documented(child, child_schema, root, &format!("{path}/{key}"));
            }
        }
        Value::Array(items) => {
            if let Some(item_schema) = node.get("items") {
                for (i, item) in items.iter().enumerate() {
                    assert_documented(item, item_schema, root, &format!("{path}/{i}"));
                }
            }
        }
        _ => {}
    }
}

#[test]
fn schema_is_a_valid_json_schema() {
    let schema = schema();
    jsonschema::meta::validate(&schema).expect("valid against the 2020-12 meta-schema");
    assert_eq!(schema["$id"], JSON_SCHEMA_URL);
    assert_eq!(
        schema["properties"]["schema_version"]["const"],
        JSON_SCHEMA_VERSION
    );
}

#[test]
fn output_follows_the_schema() {
    let root = schema();
    for (case, extra) in [
        ("short", &[][..]),
        ("silence", &[][..]),
        (
            "pluck",
            &["--no-pitch-bends", "--minimum-frequency", "80"][..],
        ),
        (
            "vocadito_10",
            &["--trim", "v0.4.0", "--bend-offset", "-1"][..],
        ),
    ] {
        let doc = transcribe(case, extra);
        assert_valid(&doc);
        assert_documented(&doc, &root, &root, "");
        // The document also parses into the Rust type, which rejects unknown fields.
        let _: JsonDocument = serde_json::from_value(doc).expect("strict layout");
    }
}

#[test]
fn document_describes_the_run() {
    let doc: JsonDocument = serde_json::from_value(transcribe("short", &[])).unwrap();
    assert_eq!(doc.schema, JSON_SCHEMA_URL);
    assert_eq!(doc.schema_version, 1);
    assert_eq!(doc.generator.name, "basic-pitch-rs");
    assert_eq!(doc.generator.version, env!("CARGO_PKG_VERSION"));
    assert_eq!(doc.model.sha256, MODEL_SHA256);
    assert_eq!(doc.model.backend, "tract");
    assert_eq!(doc.input.sample_rate, 22_050);
    assert_eq!(doc.input.channels, 1);
    assert_eq!(doc.input.frames, 13_230);
    assert_eq!(doc.input.codec, "pcm_s16le");
    assert_eq!(
        doc.analysis.resampler, None,
        "short.wav is already at 22 050 Hz"
    );
    assert_eq!(doc.analysis.samples, 13_230);
    assert_eq!(doc.analysis.frames, 51);
    assert_eq!(doc.analysis.trim_mode, "main");
    assert_eq!(doc.params.onset_threshold, 0.5);
    assert_eq!(doc.params.minimum_frequency, None);

    let pluck: JsonDocument = serde_json::from_value(transcribe("pluck", &[])).unwrap();
    assert_eq!(pluck.input.sample_rate, 48_000);
    assert_eq!(pluck.input.channels, 2);
    assert_eq!(
        pluck.analysis.resampler.as_deref(),
        Some(basic_pitch_cli::resample::Resampler::DEFAULT.name())
    );
    assert!(!pluck.notes.is_empty());
    for pair in pluck.notes.windows(2) {
        assert!(
            (pair[0].start_s, pair[0].pitch) <= (pair[1].start_s, pair[1].pitch),
            "notes are sorted by start time, then pitch"
        );
    }
    for note in &pluck.notes {
        assert_eq!(note.note_name, note_name(note.pitch));
        assert!(note.end_frame > note.start_frame && note.end_s > note.start_s);
        let bends = note
            .pitch_bends
            .as_ref()
            .expect("pitch bends are on by default");
        assert_eq!(bends.len(), note.end_frame - note.start_frame);
        let semitones = note.pitch_bends_semitones.as_ref().unwrap();
        for (b, s) in bends.iter().zip(semitones) {
            assert_eq!(*s, f64::from(*b) / 3.0);
        }
    }
}

#[test]
fn bend_offset_shifts_every_bend() {
    let plain: JsonDocument = serde_json::from_value(transcribe("synth", &[])).unwrap();
    let shifted: JsonDocument =
        serde_json::from_value(transcribe("synth", &["--bend-offset", "-1"])).unwrap();
    assert_eq!(shifted.params.bend_offset, -1);
    assert_eq!(plain.notes.len(), shifted.notes.len());
    for (a, b) in plain.notes.iter().zip(&shifted.notes) {
        let (a, b) = (
            a.pitch_bends.as_ref().unwrap(),
            b.pitch_bends.as_ref().unwrap(),
        );
        assert!(a.iter().zip(b).all(|(x, y)| y - x == -1));
    }
}
