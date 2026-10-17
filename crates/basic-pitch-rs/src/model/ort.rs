// Copyright 2026 Adrien Thevon
// SPDX-License-Identifier: Apache-2.0

//! Inference with ONNX Runtime, through the `ort` crate.

use std::sync::{Mutex, PoisonError};

use ort::environment::Environment;
use ort::session::Session;
use ort::session::builder::AutoDevicePolicy;
use ort::value::TensorRef;

use super::{Backend, check_window, window_output};
use crate::constants::{
    AUDIO_N_SAMPLES, MODEL_INPUT_NAME, MODEL_OUTPUT_CONTOUR, MODEL_OUTPUT_NOTE, MODEL_OUTPUT_ONSET,
};
use crate::error::{Error, Result};
use crate::posteriorgram::Posteriorgrams;

const NAME: &str = "ort";

/// Wraps an `ort` error, keeping it as the source.
#[allow(clippy::needless_pass_by_value, reason = "used as a map_err function")]
fn backend_error(e: ort::Error) -> Error {
    Error::backend(NAME, e)
}

/// The environment created by [`OrtBackend::new`] and [`OrtBackend::from_onnx_bytes`].
///
/// ONNX Runtime allows one environment per process, so it is created once and shared.
static ENVIRONMENT: Mutex<Option<Environment>> = Mutex::new(None);

fn shared_environment() -> Result<Environment> {
    let mut slot = ENVIRONMENT.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some(env) = slot.as_ref() {
        return Ok(env.clone());
    }
    let env = ort::init()
        .with_name("basic-pitch-rs")
        .with_telemetry(false)
        .build()
        .map_err(backend_error)?;
    *slot = Some(env.clone());
    Ok(env)
}

/// Runs the model with [ONNX Runtime](https://onnxruntime.ai) through the
/// [`ort`](https://docs.rs/ort) crate.
///
/// About 3 times faster than the tract backend on native targets, and bit-identical to
/// Python's `onnxruntime` 1.31 on the CPU. With the `ort-download` feature, the build
/// downloads a prebuilt ONNX Runtime 1.31 library (about 80 MB) and links it statically;
/// with `ort` alone, it links the ONNX Runtime you provide (see the crate's features).
///
/// ONNX Runtime already spreads each window over every core. One session runs one window
/// at a time: calls to [`predict_window`](Backend::predict_window) from several threads
/// wait for each other. For parallel transcriptions, create one `OrtBackend` per thread.
///
/// On macOS, the build script also links clang's runtime library (`libclang_rt.osx.a`
/// from the active Xcode toolchain), which the prebuilt library needs when the final
/// binary is linked with LTO.
///
/// # CPU only, on purpose
///
/// The session is always created with
/// [`AutoDevicePolicy::PreferCPU`](ort::session::builder::AutoDevicePolicy::PreferCPU).
/// By default, `ort` 2.0.0-rc.14 asks ONNX Runtime 1.31 to select devices for maximum
/// efficiency, which on macOS sends most of the graph to CoreML in float16: the outputs
/// then drift by up to 0.15 and notes change. Registering only the CPU execution
/// provider does not prevent that selection; the device policy does
/// (pykeio/ort#662). The network is small, so the CPU is fast anyway.
///
/// ```no_run
/// # #[cfg(feature = "bundled-model")] {
/// use basic_pitch::BasicPitch;
/// use basic_pitch::model::OrtBackend;
///
/// let transcriber = BasicPitch::with_backend(OrtBackend::new()?);
/// # }
/// # Ok::<(), basic_pitch::Error>(())
/// ```
#[derive(Debug)]
pub struct OrtBackend {
    session: Mutex<Session>,
}

impl OrtBackend {
    /// Loads the bundled model in a process-wide ONNX Runtime environment.
    ///
    /// # Errors
    ///
    /// [`Error::Backend`] if ONNX Runtime fails to start, for instance because the
    /// application already created its own environment: use
    /// [`OrtBackend::with_environment`] then.
    #[cfg(feature = "bundled-model")]
    #[cfg_attr(docsrs, doc(cfg(feature = "bundled-model")))]
    pub fn new() -> Result<Self> {
        Self::from_onnx_bytes(super::MODEL_ONNX)
    }

    /// Loads a Basic Pitch model from the bytes of an ONNX file.
    ///
    /// # Errors
    ///
    /// [`Error::Backend`] if ONNX Runtime fails to start or to load the model.
    pub fn from_onnx_bytes(bytes: &[u8]) -> Result<Self> {
        Self::with_environment(&shared_environment()?, bytes)
    }

    /// Loads a model in an environment created by the application.
    ///
    /// # Errors
    ///
    /// [`Error::Backend`] if ONNX Runtime fails to load the model.
    pub fn with_environment(environment: &Environment, bytes: &[u8]) -> Result<Self> {
        let mut builder = Session::builder(environment)
            .map_err(backend_error)?
            .with_auto_device(AutoDevicePolicy::PreferCPU)
            .map_err(|e| backend_error(e.into()))?;
        let session = builder.commit_from_memory(bytes).map_err(backend_error)?;
        Ok(Self {
            session: Mutex::new(session),
        })
    }
}

impl Backend for OrtBackend {
    fn predict_window(&self, window: &[f32]) -> Result<Posteriorgrams> {
        check_window(window)?;
        let err = backend_error;
        let input =
            TensorRef::from_array_view(([1_usize, AUDIO_N_SAMPLES, 1], window)).map_err(err)?;
        let mut session = self.session.lock().unwrap_or_else(PoisonError::into_inner);
        let outputs = session
            .run(ort::inputs![MODEL_INPUT_NAME => input])
            .map_err(err)?;
        let extract = |name: &str| -> Result<Vec<f32>> {
            let value = outputs.get(name).ok_or_else(|| {
                Error::backend(NAME, format!("the model has no output named {name}"))
            })?;
            let (_, data) = value.try_extract_tensor::<f32>().map_err(err)?;
            Ok(data.to_vec())
        };
        window_output(
            extract(MODEL_OUTPUT_NOTE)?,
            extract(MODEL_OUTPUT_ONSET)?,
            extract(MODEL_OUTPUT_CONTOUR)?,
        )
    }
}
