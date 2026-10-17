// Copyright 2026 Adrien Thevon
// SPDX-License-Identifier: Apache-2.0

//! Time-frequency activation matrices produced by the model.

use std::ops::Index;

use crate::constants::{N_FREQ_BINS_CONTOURS, N_FREQ_BINS_NOTES};
use crate::error::{Error, Result};

/// A row-major `frames x bins` matrix of model activations in `[0, 1]`.
///
/// Row `t` holds the activations of frame `t`; column `f` is a frequency bin. The note
/// and onset outputs have 88 bins (one per piano key, MIDI 21 to 108), the contour
/// output has 264 (three per semitone).
///
/// ```
/// use basic_pitch::Posteriorgram;
///
/// let p = Posteriorgram::from_vec(vec![0.0, 0.5, 1.0, 0.25], 2)?;
/// assert_eq!((p.n_frames(), p.n_bins()), (2, 2));
/// assert_eq!(p[(1, 0)], 1.0);
/// assert_eq!(p.row(0), Some(&[0.0, 0.5][..]));
/// # Ok::<(), basic_pitch::Error>(())
/// ```
#[derive(Clone, Debug, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(try_from = "serde_repr::PosteriorgramRepr"))]
pub struct Posteriorgram {
    data: Vec<f32>,
    n_frames: usize,
    n_bins: usize,
}

impl Posteriorgram {
    /// Wraps row-major data with `n_bins` columns.
    ///
    /// Zero bins are only accepted with no data: that is the empty matrix of
    /// [`Posteriorgram::default`].
    ///
    /// # Errors
    ///
    /// [`Error::Shape`] if `n_bins` does not divide `data.len()`.
    pub fn from_vec(data: Vec<f32>, n_bins: usize) -> Result<Self> {
        if n_bins == 0 && data.is_empty() {
            return Ok(Self::default());
        }
        if n_bins == 0 || !data.len().is_multiple_of(n_bins) {
            return Err(Error::shape(
                "posteriorgram",
                format!("a multiple of {n_bins} values"),
                data.len(),
            ));
        }
        Ok(Self {
            n_frames: data.len() / n_bins,
            data,
            n_bins,
        })
    }

    /// A matrix of zeros.
    ///
    /// # Panics
    ///
    /// If `n_frames * n_bins` overflows `usize`, or if the allocation fails, like
    /// [`vec!`].
    #[must_use]
    pub fn zeros(n_frames: usize, n_bins: usize) -> Self {
        let len = n_frames
            .checked_mul(n_bins)
            .unwrap_or_else(|| panic!("a {n_frames}x{n_bins} posteriorgram overflows usize"));
        Self {
            data: vec![0.0; len],
            n_frames,
            n_bins,
        }
    }

    /// Number of frames (rows).
    #[must_use]
    pub fn n_frames(&self) -> usize {
        self.n_frames
    }

    /// Number of frequency bins (columns).
    #[must_use]
    pub fn n_bins(&self) -> usize {
        self.n_bins
    }

    /// `true` if the matrix has no frame.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.n_frames == 0
    }

    /// The activation at `(frame, bin)`, or `None` out of bounds.
    #[must_use]
    pub fn get(&self, frame: usize, bin: usize) -> Option<f32> {
        (frame < self.n_frames && bin < self.n_bins).then(|| self.data[frame * self.n_bins + bin])
    }

    /// The activations of one frame, or `None` out of bounds.
    #[must_use]
    pub fn row(&self, frame: usize) -> Option<&[f32]> {
        (frame < self.n_frames).then(|| &self.data[frame * self.n_bins..(frame + 1) * self.n_bins])
    }

    /// Iterates over the frames.
    pub fn rows(&self) -> std::slice::ChunksExact<'_, f32> {
        // `max(1)` keeps a zero-bin default value iterable (it then yields nothing).
        self.data.chunks_exact(self.n_bins.max(1))
    }

    /// The underlying row-major data.
    #[must_use]
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    /// The underlying row-major data, mutable.
    #[must_use]
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// Consumes the matrix and returns its row-major data.
    #[must_use]
    pub fn into_vec(self) -> Vec<f32> {
        self.data
    }

    pub(crate) fn expect_bins(&self, what: &'static str, n_bins: usize) -> Result<()> {
        if self.n_bins == n_bins {
            Ok(())
        } else {
            Err(Error::shape(
                what,
                format!("{n_bins} bins"),
                format!("{} bins", self.n_bins),
            ))
        }
    }
}

impl Index<(usize, usize)> for Posteriorgram {
    type Output = f32;

    /// # Panics
    ///
    /// If `frame` or `bin` is out of bounds, like slice indexing.
    fn index(&self, (frame, bin): (usize, usize)) -> &f32 {
        assert!(
            frame < self.n_frames && bin < self.n_bins,
            "index ({frame}, {bin}) out of bounds for a {}x{} posteriorgram",
            self.n_frames,
            self.n_bins
        );
        &self.data[frame * self.n_bins + bin]
    }
}

/// The three outputs of the model, aligned on the same frames.
///
/// After [`unwrap_output`](crate::window::unwrap_output), frame `t` starts at
/// [`model_frames_to_time`](crate::notes::model_frames_to_time)`(n)[t]` seconds.
///
/// The default value has no frame, with the right number of bins. Deserialization checks
/// the shapes like [`Posteriorgrams::new`].
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(try_from = "serde_repr::PosteriorgramsRepr"))]
pub struct Posteriorgrams {
    /// Note (frame) activations, `n_frames x 88`: how likely each key sounds.
    pub note: Posteriorgram,
    /// Onset activations, `n_frames x 88`: how likely each key starts.
    pub onset: Posteriorgram,
    /// Pitch contour, `n_frames x 264`: fine pitch at a third of a semitone.
    pub contour: Posteriorgram,
}

impl Default for Posteriorgrams {
    fn default() -> Self {
        Self {
            note: Posteriorgram::zeros(0, N_FREQ_BINS_NOTES),
            onset: Posteriorgram::zeros(0, N_FREQ_BINS_NOTES),
            contour: Posteriorgram::zeros(0, N_FREQ_BINS_CONTOURS),
        }
    }
}

impl Posteriorgrams {
    /// Bundles three matrices, checking their shapes.
    ///
    /// # Errors
    ///
    /// [`Error::Shape`] if `note` or `onset` does not have 88 bins, `contour` does not
    /// have 264, or the frame counts differ.
    pub fn new(note: Posteriorgram, onset: Posteriorgram, contour: Posteriorgram) -> Result<Self> {
        let p = Self {
            note,
            onset,
            contour,
        };
        p.validate()?;
        Ok(p)
    }

    /// Number of frames shared by the three matrices.
    #[must_use]
    pub fn n_frames(&self) -> usize {
        self.note.n_frames()
    }

    /// Checks the shapes described in [`Posteriorgrams::new`].
    ///
    /// # Errors
    ///
    /// [`Error::Shape`] on the first inconsistency found.
    pub fn validate(&self) -> Result<()> {
        self.note
            .expect_bins("note posteriorgram", N_FREQ_BINS_NOTES)?;
        self.onset
            .expect_bins("onset posteriorgram", N_FREQ_BINS_NOTES)?;
        self.contour
            .expect_bins("contour posteriorgram", N_FREQ_BINS_CONTOURS)?;
        let n = self.note.n_frames();
        for (what, other) in [
            ("onset frames", &self.onset),
            ("contour frames", &self.contour),
        ] {
            if other.n_frames() != n {
                return Err(Error::shape(what, n, other.n_frames()));
            }
        }
        Ok(())
    }
}

#[cfg(feature = "serde")]
mod serde_repr {
    use super::{Posteriorgram, Posteriorgrams};
    use crate::error::Error;

    #[derive(serde::Deserialize)]
    pub(super) struct PosteriorgramsRepr {
        note: Posteriorgram,
        onset: Posteriorgram,
        contour: Posteriorgram,
    }

    impl TryFrom<PosteriorgramsRepr> for Posteriorgrams {
        type Error = Error;

        fn try_from(r: PosteriorgramsRepr) -> Result<Self, Error> {
            Posteriorgrams::new(r.note, r.onset, r.contour)
        }
    }

    #[derive(serde::Deserialize)]
    pub(super) struct PosteriorgramRepr {
        data: Vec<f32>,
        n_frames: usize,
        n_bins: usize,
    }

    impl TryFrom<PosteriorgramRepr> for Posteriorgram {
        type Error = Error;

        fn try_from(r: PosteriorgramRepr) -> Result<Self, Error> {
            let p = Posteriorgram::from_vec(r.data, r.n_bins)?;
            if p.n_frames() == r.n_frames {
                Ok(p)
            } else {
                Err(Error::shape(
                    "posteriorgram frames",
                    r.n_frames,
                    p.n_frames(),
                ))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_ragged_data() {
        assert!(Posteriorgram::from_vec(vec![0.0; 5], 2).is_err());
        assert!(Posteriorgram::from_vec(vec![0.0; 5], 0).is_err());
        assert_eq!(
            Posteriorgram::from_vec(vec![], 0).unwrap(),
            Posteriorgram::default()
        );
        assert!(Posteriorgram::from_vec(vec![], 88).unwrap().is_empty());
    }

    #[test]
    #[should_panic(expected = "overflows usize")]
    fn zeros_checks_its_size() {
        let _ = Posteriorgram::zeros(usize::MAX / 2, 3);
    }

    #[test]
    fn default_outputs_are_valid() {
        let p = Posteriorgrams::default();
        assert!(p.validate().is_ok());
        assert_eq!(p.n_frames(), 0);
        assert_eq!((p.note.n_bins(), p.contour.n_bins()), (88, 264));
    }

    #[cfg(feature = "serde")]
    #[test]
    fn serde_round_trips_and_validates() {
        let p = Posteriorgram::default();
        let json = serde_json::to_string(&p).unwrap();
        assert_eq!(serde_json::from_str::<Posteriorgram>(&json).unwrap(), p);

        let ps = Posteriorgrams::new(
            Posteriorgram::zeros(2, 88),
            Posteriorgram::zeros(2, 88),
            Posteriorgram::zeros(2, 264),
        )
        .unwrap();
        let json = serde_json::to_string(&ps).unwrap();
        assert_eq!(serde_json::from_str::<Posteriorgrams>(&json).unwrap(), ps);
        let json = serde_json::to_string(&Posteriorgrams::default()).unwrap();
        assert!(serde_json::from_str::<Posteriorgrams>(&json).is_ok());

        let bad = r#"{
            "note": {"data": [], "n_frames": 0, "n_bins": 1},
            "onset": {"data": [], "n_frames": 0, "n_bins": 3},
            "contour": {"data": [], "n_frames": 0, "n_bins": 2}
        }"#;
        assert!(serde_json::from_str::<Posteriorgrams>(bad).is_err());
    }

    #[test]
    fn accessors_are_bounds_checked() {
        let p = Posteriorgram::zeros(3, 4);
        assert_eq!(p.get(2, 3), Some(0.0));
        assert_eq!(p.get(3, 0), None);
        assert_eq!(p.row(3), None);
        assert_eq!(p.rows().len(), 3);
        assert_eq!(Posteriorgram::default().rows().len(), 0);
    }

    #[test]
    fn validates_output_shapes() {
        let ok = Posteriorgrams::new(
            Posteriorgram::zeros(4, 88),
            Posteriorgram::zeros(4, 88),
            Posteriorgram::zeros(4, 264),
        );
        assert!(ok.is_ok());
        let bad = Posteriorgrams::new(
            Posteriorgram::zeros(4, 88),
            Posteriorgram::zeros(5, 88),
            Posteriorgram::zeros(4, 264),
        );
        assert!(matches!(bad, Err(Error::Shape { .. })));
    }
}
