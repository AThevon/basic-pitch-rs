// Copyright 2026 Adrien Thevon
// SPDX-License-Identifier: Apache-2.0

//! The error type of the crate.

use std::fmt;

/// A boxed error coming from an inference backend.
pub type BoxError = Box<dyn std::error::Error + Send + Sync + 'static>;

/// Errors returned by `basic_pitch`.
///
/// Every public function that can fail on user input returns this type: the crate
/// does not panic on empty, short or malformed input.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// The audio contains a NaN or infinite sample.
    #[error("audio sample {index} is not finite ({value})")]
    NonFiniteAudio {
        /// Index of the first offending sample.
        index: usize,
        /// Its value.
        value: f32,
    },

    /// A window passed to a backend does not have [`AUDIO_N_SAMPLES`](crate::constants::AUDIO_N_SAMPLES) samples.
    #[error("a model window must hold {expected} samples, got {actual}")]
    WindowLength {
        /// Required number of samples.
        expected: usize,
        /// Number of samples received.
        actual: usize,
    },

    /// An array does not have the shape an operation requires.
    #[error("shape mismatch for {what}: expected {expected}, got {actual}")]
    Shape {
        /// Which array is wrong.
        what: &'static str,
        /// The expected shape, human readable.
        expected: String,
        /// The actual shape, human readable.
        actual: String,
    },

    /// A parameter is outside its valid range.
    #[error("invalid parameter `{name}`: {reason}")]
    InvalidParameter {
        /// Parameter name.
        name: &'static str,
        /// Why it was rejected.
        reason: String,
    },

    /// The inference backend failed to load the model or to run it.
    #[error("{backend} backend: {source}")]
    Backend {
        /// Backend name, such as `"tract"` or `"ort"`.
        backend: &'static str,
        /// The underlying error.
        #[source]
        source: BoxError,
    },

    /// Writing the MIDI file failed.
    #[cfg(feature = "midi")]
    #[error("MIDI encoding failed: {0}")]
    Midi(String),
}

impl Error {
    pub(crate) fn shape(
        what: &'static str,
        expected: impl fmt::Display,
        actual: impl fmt::Display,
    ) -> Self {
        Self::Shape {
            what,
            expected: expected.to_string(),
            actual: actual.to_string(),
        }
    }

    pub(crate) fn invalid(name: &'static str, reason: impl Into<String>) -> Self {
        Self::InvalidParameter {
            name,
            reason: reason.into(),
        }
    }

    /// Wraps an error coming from an inference backend.
    ///
    /// Custom [`Backend`](crate::model::Backend) implementations use this to report
    /// their own failures.
    pub fn backend(backend: &'static str, source: impl Into<BoxError>) -> Self {
        Self::Backend {
            backend,
            source: source.into(),
        }
    }
}

/// Shorthand for `Result<T, basic_pitch::Error>`.
pub type Result<T, E = Error> = std::result::Result<T, E>;
