// Copyright 2022 Spotify AB (original: basic_pitch/note_creation.py)
// Copyright 2026 Adrien Thevon (Rust port, modified)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Ported from spotify/basic-pitch v0.4.0. Changes: translated to Rust, and the MIDI file
// is built directly with midly. The tick conversion and the event ordering reproduce
// what pretty_midi 0.2.11 (MIT License, Colin Raffel) writes for these notes.

//! Standard MIDI File export (feature `midi`).
//!
//! [`note_events_to_midi`] produces the same MIDI events as Basic Pitch, which builds its
//! file with `pretty_midi`: format 1, 220 ticks per quarter note, a tempo track, then one
//! track of Electric Piano 1 (program 4) holding the notes and their pitch bends. Pitch
//! bends cover plus or minus 2 semitones (4096 units per semitone) and are spread evenly
//! from the start to the end of each note.
//!
//! The files are not byte-identical to `pretty_midi`'s (encoders differ, for instance in
//! running status), but every parsed event, tick and value is.
//!
//! ```
//! use basic_pitch::midi::{MidiParams, note_events_to_midi_bytes};
//! use basic_pitch::notes::NoteEvent;
//!
//! let notes = [NoteEvent {
//!     start_frame: 0, end_frame: 86, start_s: 0.0, end_s: 0.9985,
//!     pitch: 69, amplitude: 0.8, pitch_bends: None,
//! }];
//! let bytes = note_events_to_midi_bytes(&notes, &MidiParams::default())?;
//! assert_eq!(&bytes[..4], b"MThd");
//! # Ok::<(), basic_pitch::Error>(())
//! ```

use std::cmp::Ordering;

use midly::num::{u4, u7, u15, u24, u28};
use midly::{
    Format, Header, MetaMessage, MidiMessage, PitchBend, Smf, Timing, TrackEvent, TrackEventKind,
};

/// The `midly` crate, whose [`Smf`] type [`note_events_to_midi`] returns.
pub use midly;

use crate::constants::{CONTOURS_BINS_PER_SEMITONE, DEFAULT_MIDI_TEMPO};
use crate::error::{Error, Result};
use crate::notes::NoteEvent;

/// Ticks per quarter note of the files written by Basic Pitch (`pretty_midi`'s default).
pub const TICKS_PER_BEAT: u16 = 220;

/// General MIDI program of the notes: Electric Piano 1 (zero-based).
pub const PROGRAM: u8 = 4;

/// Pitch bend units per semitone: the pitch bend range is plus or minus 2 semitones.
pub const PITCH_BEND_UNITS_PER_SEMITONE: i64 = 4096;

const N_PITCH_BEND_TICKS: i64 = 8192;

/// Parameters of the MIDI export, with the defaults of `basic_pitch.inference.predict`.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default, deny_unknown_fields))]
pub struct MidiParams {
    /// Put each pitch on its own track and channel, so that overlapping notes can keep
    /// their pitch bends (default `false`).
    ///
    /// When `false`, notes that overlap another note lose their pitch bends, since a
    /// MIDI channel has a single pitch wheel. When `true`, channels are assigned in order
    /// of first appearance, skipping the drum channel 9; beyond 15 distinct pitches,
    /// channels are reused, as in `pretty_midi`.
    pub multiple_pitch_bends: bool,
    /// Tempo written to the file, in beats per minute (default 120). It only changes the
    /// tick grid: note times stay in seconds.
    pub midi_tempo: f64,
}

impl Default for MidiParams {
    fn default() -> Self {
        Self {
            multiple_pitch_bends: false,
            midi_tempo: DEFAULT_MIDI_TEMPO,
        }
    }
}

/// Converts a pitch bend in thirds of a semitone to 14-bit MIDI pitch wheel units:
/// `round(bend * 4096 / 3)` with ties to even, clipped to `[-8192, 8191]`.
///
/// ```
/// use basic_pitch::midi::bend_to_pitch_wheel;
///
/// assert_eq!(bend_to_pitch_wheel(1), 1365);
/// assert_eq!(bend_to_pitch_wheel(3), 4096); // one semitone
/// assert_eq!(bend_to_pitch_wheel(6), 8191); // clipped
/// ```
#[must_use]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "the value is clipped to the 14-bit range"
)]
pub fn bend_to_pitch_wheel(bend: i32) -> i16 {
    let units = (i64::from(bend) * PITCH_BEND_UNITS_PER_SEMITONE) as f64
        / CONTOURS_BINS_PER_SEMITONE as f64;
    (units.round_ties_even() as i64).clamp(-N_PITCH_BEND_TICKS, N_PITCH_BEND_TICKS - 1) as i16
}

/// Removes the pitch bends of every note that overlaps another one
/// (`drop_overlapping_pitch_bends`).
///
/// Returns a copy sorted like Python sorts its note tuples: by start time, end time,
/// pitch, amplitude, then pitch bends.
#[must_use]
pub fn drop_overlapping_pitch_bends(events: &[NoteEvent]) -> Vec<NoteEvent> {
    let mut sorted = events.to_vec();
    sorted.sort_by(python_tuple_order);
    for i in 0..sorted.len().saturating_sub(1) {
        for j in i + 1..sorted.len() {
            if sorted[j].start_s >= sorted[i].end_s {
                break;
            }
            sorted[i].pitch_bends = None;
            sorted[j].pitch_bends = None;
        }
    }
    sorted
}

/// Order of Python's `sorted` on `(start_s, end_s, pitch, amplitude, bends)` tuples.
fn python_tuple_order(a: &NoteEvent, b: &NoteEvent) -> Ordering {
    a.start_s
        .total_cmp(&b.start_s)
        .then(a.end_s.total_cmp(&b.end_s))
        .then(a.pitch.cmp(&b.pitch))
        .then(a.amplitude.total_cmp(&b.amplitude))
        .then_with(|| a.pitch_bends.cmp(&b.pitch_bends))
}

/// Builds the MIDI file Basic Pitch writes for `events` (`note_events_to_midi`).
///
/// # Errors
///
/// [`Error::InvalidParameter`] if the tempo is not a positive finite number that MIDI
/// can store (above about 3.6 BPM), or [`Error::Midi`] if a note cannot be written: a
/// pitch above 127, an amplitude whose velocity `round(127 * amplitude)` is not in
/// `0..=127` (NaN included), or a time that is NaN or does not fit in a MIDI file (more
/// than 2^28 ticks). Python fails on the same notes, when `pretty_midi` writes the file.
pub fn note_events_to_midi(events: &[NoteEvent], params: &MidiParams) -> Result<Smf<'static>> {
    for (i, note) in events.iter().enumerate() {
        check_note(note).map_err(|reason| Error::Midi(format!("note {i}: {reason}")))?;
    }
    let tempo = params.midi_tempo;
    if !tempo.is_finite() || tempo <= 0.0 {
        return Err(Error::invalid(
            "midi_tempo",
            format!("must be a positive number of BPM, got {tempo}"),
        ));
    }
    let resolution = f64::from(TICKS_PER_BEAT);
    let tick_scale = 60.0 / (tempo * resolution);
    // pretty_midi: int(6e7 / (60. / (tick_scale * resolution)))
    let mpqn = 6e7 / (60.0 / (tick_scale * resolution));
    let mpqn = u24::try_from(float_to_u32(mpqn.trunc()).unwrap_or(u32::MAX)).ok_or_else(|| {
        Error::invalid(
            "midi_tempo",
            format!("{tempo} BPM is too slow for a MIDI tempo"),
        )
    })?;

    let mut tracks = vec![finish_track(vec![
        (0, Event::Tempo(mpqn)),
        (0, Event::TimeSignature),
    ])?];

    let dropped;
    let events = if params.multiple_pitch_bends {
        events
    } else {
        dropped = drop_overlapping_pitch_bends(events);
        &dropped
    };

    // One instrument per pitch (or a single one), in order of first appearance.
    let mut instruments: Vec<(u8, Vec<(u64, Event)>)> = Vec::new();
    let time_to_tick = |t: f64| time_to_tick(t, tick_scale);
    for note in events {
        let key = if params.multiple_pitch_bends {
            note.pitch
        } else {
            0
        };
        let idx = instruments
            .iter()
            .position(|(k, _)| *k == key)
            .unwrap_or_else(|| {
                instruments.push((key, Vec::new()));
                instruments.len() - 1
            });
        let (_, track) = &mut instruments[idx];
        track.push((
            time_to_tick(note.start_s)?,
            Event::NoteOn(note.pitch, note.velocity()),
        ));
        track.push((time_to_tick(note.end_s)?, Event::NoteOn(note.pitch, 0)));
    }
    // pretty_midi appends all notes, then all pitch bends, then sorts stably.
    for note in events {
        let Some(bends) = note.pitch_bends.as_deref().filter(|b| !b.is_empty()) else {
            continue;
        };
        let key = if params.multiple_pitch_bends {
            note.pitch
        } else {
            0
        };
        let Some((_, track)) = instruments.iter_mut().find(|(k, _)| *k == key) else {
            continue;
        };
        for (t, &bend) in linspace(note.start_s, note.end_s, bends.len())
            .into_iter()
            .zip(bends)
        {
            track.push((
                time_to_tick(t)?,
                Event::PitchWheel(bend_to_pitch_wheel(bend)),
            ));
        }
    }

    // Channels 0..=15 without the drum channel 9, assigned round robin.
    let channels: Vec<u8> = (0..16).filter(|&c| c != 9).collect();
    for (n, (_, notes)) in instruments.into_iter().enumerate() {
        let channel = channels[n % channels.len()];
        let mut events = vec![(0, Event::ProgramChange(PROGRAM))];
        events.extend(notes);
        tracks.push(finish_track_on(events, channel)?);
    }

    Ok(Smf {
        header: Header::new(Format::Parallel, Timing::Metrical(u15::new(TICKS_PER_BEAT))),
        tracks,
    })
}

/// Rejects what `pretty_midi` cannot write: MIDI data bytes are 7-bit.
fn check_note(note: &NoteEvent) -> std::result::Result<(), String> {
    if note.pitch > 127 {
        return Err(format!("pitch {} is above 127", note.pitch));
    }
    // numpy computes 127 * amplitude in float32, then int(np.round(...)).
    let velocity = (127.0_f32 * note.amplitude).round_ties_even();
    if !(0.0..=127.0).contains(&velocity) {
        return Err(format!(
            "amplitude {} gives velocity {velocity}, outside 0..=127",
            note.amplitude
        ));
    }
    Ok(())
}

/// [`note_events_to_midi`], encoded as the bytes of a `.mid` file.
///
/// # Errors
///
/// Same as [`note_events_to_midi`].
pub fn note_events_to_midi_bytes(events: &[NoteEvent], params: &MidiParams) -> Result<Vec<u8>> {
    let smf = note_events_to_midi(events, params)?;
    let mut bytes = Vec::new();
    smf.write(&mut bytes)
        .map_err(|e| Error::Midi(e.to_string()))?;
    Ok(bytes)
}

/// `PrettyMIDI.time_to_tick` for a file with a single tempo.
fn time_to_tick(time: f64, tick_scale: f64) -> Result<u64> {
    if time.is_nan() {
        return Err(Error::Midi("a note time is NaN".into()));
    }
    if time <= 0.0 {
        return Ok(0);
    }
    let tick = (time / tick_scale).round_ties_even();
    if tick > f64::from(u28::max_value().as_int()) {
        return Err(Error::Midi(format!(
            "time {time} s is beyond the range of a MIDI file"
        )));
    }
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "range checked above"
    )]
    Ok(tick as u64)
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "range checked"
)]
fn float_to_u32(x: f64) -> Option<u32> {
    (x >= 0.0 && x <= f64::from(u32::MAX)).then_some(x as u32)
}

/// `np.linspace(start, stop, num)` with the endpoint, bit for bit.
#[allow(clippy::cast_precision_loss, reason = "small counts")]
fn linspace(start: f64, end: f64, num: usize) -> Vec<f64> {
    if num == 0 {
        return Vec::new();
    }
    let div = (num - 1) as f64;
    let delta = end - start;
    if num == 1 {
        return vec![0.0 * delta + start];
    }
    let step = delta / div;
    let mut y: Vec<f64> = if step == 0.0 {
        (0..num).map(|i| (i as f64 / div) * delta + start).collect()
    } else {
        (0..num).map(|i| i as f64 * step + start).collect()
    };
    y[num - 1] = end;
    y
}

/// A MIDI event before encoding, carrying what `pretty_midi`'s sort looks at.
#[derive(Clone, Copy, Debug)]
enum Event {
    Tempo(u24),
    TimeSignature,
    ProgramChange(u8),
    PitchWheel(i16),
    NoteOn(u8, u8),
    EndOfTrack,
}

impl Event {
    /// Secondary sort key of `PrettyMIDI.write` for events on the same tick.
    fn secondary(self) -> i64 {
        const SPACING: i64 = 256 * 256;
        match self {
            Self::Tempo(_) => SPACING,
            Self::TimeSignature => 2 * SPACING,
            Self::ProgramChange(_) => 6 * SPACING,
            Self::PitchWheel(p) => 7 * SPACING + i64::from(p),
            Self::NoteOn(note, vel) => 10 * SPACING + i64::from(note) * 256 + i64::from(vel),
            Self::EndOfTrack => 11 * SPACING,
        }
    }

    fn kind(self, channel: u8) -> TrackEventKind<'static> {
        let channel = u4::new(channel);
        match self {
            Self::Tempo(t) => TrackEventKind::Meta(MetaMessage::Tempo(t)),
            // 4/4, 24 MIDI clocks per click, 8 32nd notes per quarter: mido's defaults.
            Self::TimeSignature => TrackEventKind::Meta(MetaMessage::TimeSignature(4, 2, 24, 8)),
            Self::EndOfTrack => TrackEventKind::Meta(MetaMessage::EndOfTrack),
            Self::ProgramChange(program) => TrackEventKind::Midi {
                channel,
                message: MidiMessage::ProgramChange {
                    program: u7::new(program),
                },
            },
            Self::PitchWheel(p) => TrackEventKind::Midi {
                channel,
                message: MidiMessage::PitchBend {
                    bend: PitchBend::from_int(p),
                },
            },
            Self::NoteOn(note, vel) => TrackEventKind::Midi {
                channel,
                message: MidiMessage::NoteOn {
                    key: u7::new(note),
                    vel: u7::new(vel),
                },
            },
        }
    }
}

fn finish_track(events: Vec<(u64, Event)>) -> Result<Vec<TrackEvent<'static>>> {
    finish_track_on(events, 0)
}

/// Sorts events like `PrettyMIDI.write`, appends the end of track one tick after the
/// last event, and converts absolute ticks to deltas.
fn finish_track_on(mut events: Vec<(u64, Event)>, channel: u8) -> Result<Vec<TrackEvent<'static>>> {
    events.sort_by_key(|&(tick, e)| (tick, e.secondary()));
    // pretty_midi then swaps a note-on followed by a note-off of the same pitch on the
    // same tick. After the sort above this never happens; it is kept for fidelity.
    let snapshot = events.clone();
    for (n, pair) in snapshot.windows(2).enumerate() {
        if let [(t1, Event::NoteOn(n1, v1)), (t2, Event::NoteOn(n2, v2))] = pair
            && t1 == t2
            && n1 == n2
            && *v1 != 0
            && *v2 == 0
        {
            events.swap(n, n + 1);
        }
    }
    let last = events.last().map_or(0, |&(t, _)| t);
    events.push((last + 1, Event::EndOfTrack));

    let mut previous = 0;
    events
        .into_iter()
        .map(|(tick, e)| {
            let delta = u32::try_from(tick - previous)
                .ok()
                .and_then(u28::try_from)
                .ok_or_else(|| Error::Midi("event delta beyond the range of a MIDI file".into()))?;
            previous = tick;
            Ok(TrackEvent {
                delta,
                kind: e.kind(channel),
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn note(start_s: f64, end_s: f64, pitch: u8, bends: Option<Vec<i32>>) -> NoteEvent {
        NoteEvent {
            start_frame: 0,
            end_frame: 0,
            start_s,
            end_s,
            pitch,
            amplitude: 0.5,
            pitch_bends: bends,
        }
    }

    #[test]
    fn pitch_wheel_rounding_and_clipping() {
        let got: Vec<i16> = [-7, -6, -1, 0, 1, 2, 5, 6]
            .map(bend_to_pitch_wheel)
            .to_vec();
        assert_eq!(got, [-8192, -8192, -1365, 0, 1365, 2731, 6827, 8191]);
    }

    #[test]
    fn linspace_matches_numpy() {
        assert_eq!(linspace(1.0, 2.0, 5), [1.0, 1.25, 1.5, 1.75, 2.0]);
        assert_eq!(linspace(3.0, 4.0, 1), [3.0]);
        assert!(linspace(0.0, 1.0, 0).is_empty());
        assert_eq!(linspace(2.0, 2.0, 3), [2.0, 2.0, 2.0]);
    }

    #[test]
    fn overlapping_notes_lose_their_bends() {
        let events = [
            note(1.0, 2.0, 60, Some(vec![1])),
            note(0.0, 1.5, 64, Some(vec![1])),
            note(3.0, 4.0, 67, Some(vec![1])),
        ];
        let out = drop_overlapping_pitch_bends(&events);
        assert_eq!(
            out.iter().map(|n| n.pitch).collect::<Vec<_>>(),
            [64, 60, 67]
        );
        assert_eq!(out[0].pitch_bends, None);
        assert_eq!(out[1].pitch_bends, None);
        assert_eq!(out[2].pitch_bends, Some(vec![1]));
    }

    #[test]
    fn tempo_is_validated_and_truncated() {
        assert!(
            note_events_to_midi(
                &[],
                &MidiParams {
                    midi_tempo: 0.0,
                    ..MidiParams::default()
                }
            )
            .is_err()
        );
        assert!(
            note_events_to_midi(
                &[],
                &MidiParams {
                    midi_tempo: 1.0,
                    ..MidiParams::default()
                }
            )
            .is_err()
        );
        let smf = note_events_to_midi(
            &[],
            &MidiParams {
                midi_tempo: 90.0,
                ..MidiParams::default()
            },
        )
        .unwrap();
        assert_eq!(smf.tracks.len(), 1);
        assert_eq!(
            smf.tracks[0][0].kind,
            TrackEventKind::Meta(MetaMessage::Tempo(u24::new(666_666)))
        );
    }

    #[test]
    fn notes_that_midi_cannot_hold_are_rejected() {
        let params = MidiParams::default();
        let mut bad = Vec::new();
        for amplitude in [f32::NAN, f32::INFINITY, -0.01, 1.01, 1.5] {
            bad.push(NoteEvent {
                amplitude,
                ..note(0.0, 1.0, 60, None)
            });
        }
        bad.push(note(0.0, 1.0, 128, None));
        bad.push(note(0.0, 1.0, 200, None));
        for n in bad {
            let err = note_events_to_midi(std::slice::from_ref(&n), &params).unwrap_err();
            assert!(matches!(err, Error::Midi(_)), "{n:?}: {err}");
        }
        // Rounding keeps these in range, as in numpy: 127 * 1.0039 rounds to 127, and
        // -0.0039 * 127 rounds to -0.
        for amplitude in [0.0, 1.0, 1.0039, -0.0039] {
            let n = NoteEvent {
                amplitude,
                ..note(0.0, 1.0, 127, None)
            };
            assert!(note_events_to_midi(&[n], &params).is_ok(), "{amplitude}");
        }
    }

    #[test]
    fn one_track_per_pitch_skips_the_drum_channel() {
        let events: Vec<NoteEvent> = (0..11)
            .map(|i| note(f64::from(i), f64::from(i) + 0.5, 40 + i as u8, None))
            .collect();
        let params = MidiParams {
            multiple_pitch_bends: true,
            ..MidiParams::default()
        };
        let smf = note_events_to_midi(&events, &params).unwrap();
        assert_eq!(smf.tracks.len(), 12);
        let channel_of = |t: &[TrackEvent]| match t[0].kind {
            TrackEventKind::Midi { channel, .. } => channel.as_int(),
            _ => unreachable!(),
        };
        assert_eq!(channel_of(&smf.tracks[9]), 8);
        assert_eq!(channel_of(&smf.tracks[10]), 10);
    }
}
