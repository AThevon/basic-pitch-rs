// Copyright 2026 Adrien Thevon
// SPDX-License-Identifier: Apache-2.0

//! WebAssembly bindings of basic-pitch-rs, built with `wasm-bindgen`.
//!
//! Everything runs in the page (or in Node): the model is embedded in the `.wasm` file
//! and executed by tract, with no server and no native library.
//!
//! ```js
//! import init, { Transcriber } from "./pkg/basic_pitch_wasm.js";
//!
//! await init();
//! const transcriber = new Transcriber();         // or new Transcriber("v0.4.0")
//! const notes = JSON.parse(transcriber.transcribe(samples)); // mono Float32Array at 22 050 Hz
//! ```
//!
//! Parameters are passed as JSON, with the field names of [`NoteParams`]; missing fields
//! keep their default, for example `JSON.stringify({ onset_threshold: 0.6 })`.

#![forbid(unsafe_code)]

use basic_pitch::constants::AUDIO_SAMPLE_RATE;
use basic_pitch::{BasicPitch, MidiParams, NoteEvent, NoteParams, Posteriorgrams, TrimMode};
use wasm_bindgen::prelude::*;

/// The sample rate the model expects, in Hz (22 050).
#[wasm_bindgen(js_name = sampleRate)]
#[must_use]
pub fn sample_rate() -> u32 {
    AUDIO_SAMPLE_RATE
}

/// The version of the basic-pitch-rs library inside this module.
#[wasm_bindgen]
#[must_use]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_owned()
}

/// Transcribes `audio` (mono, 22 050 Hz) with the default settings and returns the notes
/// as JSON.
///
/// This loads the model on every call; keep a [`Transcriber`] to transcribe several
/// files.
///
/// # Errors
///
/// If the audio contains NaN or infinite samples.
#[wasm_bindgen]
pub fn transcribe(audio: &[f32]) -> Result<String, JsError> {
    Transcriber::new(None)?.transcribe(audio, None)
}

/// A loaded model, reusable for any number of transcriptions.
#[wasm_bindgen]
pub struct Transcriber {
    inner: BasicPitch,
}

#[wasm_bindgen]
impl Transcriber {
    /// Loads the embedded model.
    ///
    /// `trim` selects how the output is trimmed to the audio length: `"main"` (default,
    /// keeps the end of the file) or `"v0.4.0"` (identical to the PyPI release).
    ///
    /// # Errors
    ///
    /// If `trim` is not one of these values.
    #[wasm_bindgen(constructor)]
    pub fn new(trim: Option<String>) -> Result<Transcriber, JsError> {
        let trim = match trim.as_deref() {
            None => TrimMode::default(),
            Some(name) => name.parse::<TrimMode>()?,
        };
        Ok(Self {
            inner: BasicPitch::new()?.with_trim_mode(trim),
        })
    }

    /// Runs the network on `audio` (mono, 22 050 Hz) and keeps its outputs, from which
    /// notes can then be decoded with any parameters.
    ///
    /// # Errors
    ///
    /// If the audio contains NaN or infinite samples.
    pub fn predict(&self, audio: &[f32]) -> Result<Prediction, JsError> {
        Ok(Prediction {
            inner: self.inner.predict(audio)?,
        })
    }

    /// Transcribes `audio` (mono, 22 050 Hz) and returns the notes as a JSON array.
    ///
    /// Each note has `start_s`, `end_s`, `start_frame`, `end_frame`, `pitch` (MIDI
    /// number), `amplitude` and `pitch_bends` (one value per frame, in thirds of a
    /// semitone, or `null`), in Basic Pitch's order.
    ///
    /// # Errors
    ///
    /// If the audio contains NaN or infinite samples, or `params` is not valid.
    pub fn transcribe(&self, audio: &[f32], params: Option<String>) -> Result<String, JsError> {
        self.predict(audio)?.notes(params)
    }
}

/// The model outputs for one recording: note, onset and contour activations, one frame
/// every 256 samples.
#[wasm_bindgen]
pub struct Prediction {
    inner: Posteriorgrams,
}

#[wasm_bindgen]
impl Prediction {
    /// Number of frames.
    #[wasm_bindgen(getter, js_name = nFrames)]
    #[must_use]
    pub fn n_frames(&self) -> usize {
        self.inner.n_frames()
    }

    /// Note activations, row-major: `nFrames` rows of 88 bins (A0 to C8).
    #[must_use]
    pub fn note(&self) -> Vec<f32> {
        self.inner.note.as_slice().to_vec()
    }

    /// Onset activations, row-major: `nFrames` rows of 88 bins.
    #[must_use]
    pub fn onset(&self) -> Vec<f32> {
        self.inner.onset.as_slice().to_vec()
    }

    /// Pitch contour, row-major: `nFrames` rows of 264 bins (a third of a semitone each).
    #[must_use]
    pub fn contour(&self) -> Vec<f32> {
        self.inner.contour.as_slice().to_vec()
    }

    /// Decodes the notes, as a JSON array (see [`Transcriber::transcribe`]).
    ///
    /// # Errors
    ///
    /// If `params` is not valid.
    pub fn notes(&self, params: Option<String>) -> Result<String, JsError> {
        Ok(serde_json::to_string(&self.decode(params.as_deref())?)?)
    }

    /// Decodes the notes and encodes them as a Standard MIDI File, like Basic Pitch's
    /// `.mid` output.
    ///
    /// `midi_params` is a JSON object with the fields of `MidiParams`, for example
    /// `{"midi_tempo": 90}`.
    ///
    /// # Errors
    ///
    /// If a parameter is not valid.
    pub fn midi(
        &self,
        params: Option<String>,
        midi_params: Option<String>,
    ) -> Result<Vec<u8>, JsError> {
        let notes = self.decode(params.as_deref())?;
        let midi_params: MidiParams = parse_or_default(midi_params.as_deref())?;
        Ok(basic_pitch::midi::note_events_to_midi_bytes(
            &notes,
            &midi_params,
        )?)
    }
}

impl Prediction {
    fn decode(&self, params: Option<&str>) -> Result<Vec<NoteEvent>, JsError> {
        let params: NoteParams = parse_or_default(params)?;
        Ok(basic_pitch::notes::model_output_to_notes(
            &self.inner,
            &params,
        )?)
    }
}

fn parse_or_default<T: Default + serde::de::DeserializeOwned>(
    json: Option<&str>,
) -> Result<T, JsError> {
    match json {
        None => Ok(T::default()),
        Some(json) => serde_json::from_str(json)
            .map_err(|e| JsError::new(&format!("invalid parameters: {e}"))),
    }
}
