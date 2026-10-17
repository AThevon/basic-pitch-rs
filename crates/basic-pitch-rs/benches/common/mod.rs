// Copyright 2026 Adrien Thevon
// SPDX-License-Identifier: Apache-2.0

//! Inputs shared by the benchmarks, read from the parity fixtures in `tests/fixtures`.

#![allow(dead_code, reason = "each benchmark uses part of this module")]

use std::path::{Path, PathBuf};

use basic_pitch::constants::{AUDIO_SAMPLE_RATE, N_FREQ_BINS_CONTOURS, N_FREQ_BINS_NOTES};
use basic_pitch::{Posteriorgram, Posteriorgrams};

/// The recordings concatenated into [`one_minute`], in order.
const MINUTE_SOURCES: [&str; 4] = ["vocadito_14", "pluck", "vocadito_10", "synth"];

pub fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures")
}

/// Reads a little-endian float32 `.npy` file (any shape, C order) as a flat vector.
pub fn read_npy_f32(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    assert_eq!(
        &bytes[1..6],
        b"NUMPY",
        "{}: not a .npy file",
        path.display()
    );
    let (header_start, header_len) = match bytes[6] {
        1 => (10, usize::from(u16::from_le_bytes([bytes[8], bytes[9]]))),
        _ => (
            12,
            u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
        ),
    };
    let header =
        std::str::from_utf8(&bytes[header_start..header_start + header_len]).expect("ASCII header");
    assert!(
        header.contains("'descr': '<f4'"),
        "{}: {header}",
        path.display()
    );
    assert!(
        header.contains("'fortran_order': False"),
        "{}",
        path.display()
    );
    let (values, rest) = bytes[header_start + header_len..].as_chunks::<4>();
    assert!(rest.is_empty(), "{}: truncated data", path.display());
    values.iter().map(|&b| f32::from_le_bytes(b)).collect()
}

/// The audio of a fixture case, mono at 22 050 Hz.
pub fn audio(case: &str) -> Vec<f32> {
    read_npy_f32(&fixtures_dir().join(case).join("audio_22050.npy"))
}

/// The posteriorgrams computed by Python Basic Pitch 0.4.0 for a fixture case.
pub fn posteriorgrams(case: &str) -> Posteriorgrams {
    let dir = fixtures_dir().join(case).join("v0.4.0");
    let load = |name: &str, bins: usize| {
        Posteriorgram::from_vec(read_npy_f32(&dir.join(format!("{name}.npy"))), bins)
            .expect("fixture shape")
    };
    Posteriorgrams::new(
        load("note", N_FREQ_BINS_NOTES),
        load("onset", N_FREQ_BINS_NOTES),
        load("contour", N_FREQ_BINS_CONTOURS),
    )
    .expect("consistent fixtures")
}

/// Exactly 60 s of real and synthetic music: the fixture recordings, concatenated and
/// repeated. `reference/` builds the same signal for the Python timings.
pub fn one_minute() -> Vec<f32> {
    let len = 60 * AUDIO_SAMPLE_RATE as usize;
    let sources: Vec<Vec<f32>> = MINUTE_SOURCES.iter().map(|case| audio(case)).collect();
    sources
        .iter()
        .flatten()
        .copied()
        .cycle()
        .take(len)
        .collect()
}
