// Copyright 2026 Adrien Thevon
// SPDX-License-Identifier: Apache-2.0

//! A progress bar on stderr, advanced by the model once per window.

use std::io::IsTerminal;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use basic_pitch::model::Backend;
use basic_pitch::{Posteriorgrams, Result};
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};

/// The current progress bar, shared between the command loop and the backend.
///
/// The bar is hidden unless stderr is a terminal and the run is not quiet, so
/// redirected output and logs never contain progress lines.
#[derive(Clone, Debug)]
pub(crate) struct Progress {
    bar: Arc<Mutex<ProgressBar>>,
    visible: bool,
}

impl Progress {
    pub(crate) fn new(quiet: bool) -> Self {
        Self {
            bar: Arc::new(Mutex::new(ProgressBar::hidden())),
            visible: !quiet && std::io::stderr().is_terminal(),
        }
    }

    /// Shows a spinner while a file is decoded and resampled.
    pub(crate) fn decoding(&self, label: &str) {
        self.replace(None, "{prefix:.bold} {spinner:.green} {msg:.dim}", label);
        self.lock().set_message("decoding");
    }

    /// Starts a new bar for one file: `windows` model windows, labelled `label`.
    pub(crate) fn start(&self, label: &str, windows: usize) {
        self.replace(
            Some(windows as u64),
            "{prefix:.bold} {bar:28.green/dim} {pos}/{len} windows {elapsed:.dim}",
            label,
        );
    }

    /// Clears the current bar, then shows a new one (a spinner when `len` is `None`).
    fn replace(&self, len: Option<u64>, template: &str, label: &str) {
        let mut current = self.lock();
        current.finish_and_clear();
        *current = if self.visible {
            let bar = ProgressBar::with_draw_target(len, ProgressDrawTarget::stderr_with_hz(10));
            bar.set_style(
                ProgressStyle::with_template(template)
                    .expect("the template is valid")
                    .progress_chars("━╸─"),
            );
            bar.set_prefix(label.to_owned());
            bar.enable_steady_tick(Duration::from_millis(100));
            bar
        } else {
            ProgressBar::hidden()
        };
    }

    /// Removes the bar from the terminal.
    pub(crate) fn finish(&self) {
        self.lock().finish_and_clear();
    }

    /// Prints a line on stderr without breaking the bar.
    pub(crate) fn println(&self, line: &str) {
        self.lock().suspend(|| eprintln!("{line}"));
    }

    fn inc(&self) {
        self.lock().inc(1);
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, ProgressBar> {
        self.bar.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// A backend that advances the progress bar after each window.
pub(crate) struct ProgressBackend<B> {
    pub(crate) inner: B,
    pub(crate) progress: Progress,
}

impl<B: Backend> Backend for ProgressBackend<B> {
    fn predict_window(&self, window: &[f32]) -> Result<Posteriorgrams> {
        let out = self.inner.predict_window(window);
        self.progress.inc();
        out
    }
}
