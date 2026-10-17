// Copyright 2026 Adrien Thevon
// SPDX-License-Identifier: Apache-2.0

//! The `basic-pitch-rs` command-line tool: audio files in, MIDI, JSON or CSV notes out.
//!
//! Install it with `cargo install basic-pitch-cli`, which provides the `basic-pitch-rs`
//! command, and run `basic-pitch-rs --help` for the options.
//!
//! This library target only exists for the tool's own tests and has no public API. To
//! transcribe audio from Rust, use the [`basic_pitch`] crate (`basic-pitch-rs` on
//! crates.io).
//!
//! # Features
//!
//! | feature | default | description |
//! |---------|---------|-------------|
//! | `soxr`  | no      | Resample with libsoxr, like librosa, for sample-level parity with Python. Statically links libsoxr, which is LGPL-2.1-or-later: binaries built with it must meet the LGPL's terms. |
//! | `ort`   | no      | Adds the ONNX Runtime backend (`--backend ort`). |

#![deny(unsafe_code)]
#![deny(missing_docs)]

// The modules are public for the integration tests only, and hidden from the
// documentation: they are not an API.
#[doc(hidden)]
pub mod audio;
#[doc(hidden)]
pub mod caf;
#[doc(hidden)]
pub mod cli;
#[doc(hidden)]
pub mod mp4;
#[doc(hidden)]
pub mod npz;
#[doc(hidden)]
pub mod output;
mod progress;
#[doc(hidden)]
pub mod resample;
