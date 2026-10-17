// Copyright 2026 Adrien Thevon
// SPDX-License-Identifier: Apache-2.0

//! Inference with tract, in pure Rust.

use std::sync::Arc;

use tract_onnx::prelude::{
    Datum, Framework, InferenceFact, InferenceModelExt, IntoRunnable, Tensor, TypedSimplePlan, tvec,
};

use super::{Backend, check_window, window_output};
use crate::constants::{
    AUDIO_N_SAMPLES, MODEL_OUTPUT_CONTOUR, MODEL_OUTPUT_NOTE, MODEL_OUTPUT_ONSET,
};
use crate::error::{Error, Result};
use crate::posteriorgram::Posteriorgrams;

const NAME: &str = "tract";

/// Runs the model with [tract](https://github.com/sonos/tract), a pure Rust ONNX engine.
///
/// The default backend. It needs no native library, works on every target Rust
/// supports, including `wasm32-unknown-unknown`, and matches ONNX Runtime to about
/// 1e-6 on the model outputs.
///
/// The backend is cheap to share: [`predict_window`](Backend::predict_window) takes
/// `&self` and can be called from several threads.
///
/// ```no_run
/// # #[cfg(feature = "bundled-model")] {
/// use basic_pitch::model::{Backend, TractBackend};
/// use basic_pitch::constants::AUDIO_N_SAMPLES;
///
/// let backend = TractBackend::new()?;
/// let out = backend.predict_window(&vec![0.0; AUDIO_N_SAMPLES])?;
/// assert_eq!(out.n_frames(), 172);
/// # }
/// # Ok::<(), basic_pitch::Error>(())
/// ```
#[derive(Clone, Debug)]
pub struct TractBackend {
    plan: Arc<TypedSimplePlan>,
}

impl TractBackend {
    /// Loads the bundled model.
    ///
    /// # Errors
    ///
    /// [`Error::Backend`] if tract cannot optimize the graph (not expected for the
    /// bundled model).
    #[cfg(feature = "bundled-model")]
    #[cfg_attr(docsrs, doc(cfg(feature = "bundled-model")))]
    pub fn new() -> Result<Self> {
        Self::from_onnx_bytes(super::MODEL_ONNX)
    }

    /// Loads a Basic Pitch model from the bytes of an ONNX file.
    ///
    /// The model must have the interface of `nmp.onnx`: one input of shape
    /// `[batch, 43844, 1]` and the outputs `StatefulPartitionedCall:{0,1,2}`.
    ///
    /// # Errors
    ///
    /// [`Error::Backend`] if the bytes are not a loadable ONNX model with that interface.
    pub fn from_onnx_bytes(bytes: &[u8]) -> Result<Self> {
        let err = |e: tract_onnx::prelude::TractError| Error::backend(NAME, e);
        let mut model = tract_onnx::onnx()
            .model_for_read(&mut &*bytes)
            .map_err(err)?;
        // Outputs are selected by name: their order in the graph is onset, note, contour.
        let outlets = model.output_outlets().map_err(err)?.to_vec();
        let find = |name: &str| {
            outlets
                .iter()
                .copied()
                .find(|&o| model.outlet_label(o) == Some(name))
                .ok_or_else(|| {
                    Error::backend(NAME, format!("the model has no output named {name}"))
                })
        };
        let selected = [
            find(MODEL_OUTPUT_NOTE)?,
            find(MODEL_OUTPUT_ONSET)?,
            find(MODEL_OUTPUT_CONTOUR)?,
        ];
        model.select_output_outlets(&selected).map_err(err)?;
        // A concrete batch of 1 is required: with a symbolic batch, tract cannot type the
        // Reshape of the normalized log layer.
        let plan = model
            .with_input_fact(
                0,
                InferenceFact::dt_shape(f32::datum_type(), [1, AUDIO_N_SAMPLES, 1]),
            )
            .and_then(InferenceModelExt::into_optimized)
            .and_then(IntoRunnable::into_runnable)
            .map_err(err)?;
        Ok(Self { plan })
    }
}

impl Backend for TractBackend {
    fn predict_window(&self, window: &[f32]) -> Result<Posteriorgrams> {
        check_window(window)?;
        let err = |e: tract_onnx::prelude::TractError| Error::backend(NAME, e);
        let input = Tensor::from_shape(&[1, AUDIO_N_SAMPLES, 1], window).map_err(err)?;
        let outputs = self.plan.run(tvec!(input.into())).map_err(err)?;
        let mut flat = Vec::with_capacity(3);
        for out in &outputs {
            let view = out.to_plain_array_view::<f32>().map_err(err)?;
            flat.push(view.iter().copied().collect::<Vec<f32>>());
        }
        let [note, onset, contour]: [Vec<f32>; 3] = flat.try_into().map_err(|v: Vec<_>| {
            Error::backend(NAME, format!("expected 3 outputs, got {}", v.len()))
        })?;
        window_output(note, onset, contour)
    }
}
