// Copyright 2026 Adrien Thevon
// SPDX-License-Identifier: Apache-2.0

//! `basic-pitch-rs`: polyphonic audio-to-MIDI transcription from the command line.
//!
//! Run `basic-pitch-rs --help` for the options.

#![forbid(unsafe_code)]

fn main() -> std::process::ExitCode {
    basic_pitch_cli::cli::main()
}
