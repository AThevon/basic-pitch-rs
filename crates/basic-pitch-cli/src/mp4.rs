// Copyright 2026 Adrien Thevon
// SPDX-License-Identifier: Apache-2.0

//! Gapless trimming of MP4, M4A and MOV audio.
//!
//! AAC encoders add priming samples before the audio (usually 2112) and padding after it,
//! and record how much in the file: in the edit list of the track (`elst`, written by
//! AVFoundation, so by iPhone recordings and videos) or in an `iTunSMPB` tag (written by
//! iTunes and `afconvert`). Players and ffmpeg (which librosa uses for these files) drop
//! those samples; without that, every note would come out about 48 ms late.
//!
//! The demuxer of symphonia 0.6 reads neither, so this module finds them with a minimal
//! box parser. It only reads the `moov` box.

use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::Path;

/// The part of a decoded track that holds the real audio, in samples.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Gapless {
    /// Total number of samples in the encoded track, priming and padding included.
    pub encoded: u64,
    /// Priming samples to drop at the start.
    pub skip: u64,
    /// Number of samples of audio after `skip`, when known.
    pub length: Option<u64>,
}

impl Gapless {
    /// Trims `samples` (one per frame) if they are still the whole encoded track.
    ///
    /// Leaves them untouched when their length is not [`encoded`](Self::encoded), which
    /// means the decoder already trimmed them (or decoded something else).
    pub fn apply(&self, samples: &mut Vec<f32>) {
        if samples.len() as u64 != self.encoded {
            return;
        }
        let start = usize::try_from(self.skip)
            .unwrap_or(usize::MAX)
            .min(samples.len());
        let end = match self.length {
            Some(length) => usize::try_from(self.skip.saturating_add(length))
                .unwrap_or(usize::MAX)
                .min(samples.len()),
            None => samples.len(),
        };
        samples.truncate(end);
        samples.drain(..start);
    }
}

/// Largest `moov` box read, to bound the memory used on malformed files.
const MAX_MOOV: u64 = 64 << 20;

/// Finds the gapless information of the first audio track of an MP4 file, at
/// `sample_rate` Hz.
///
/// Returns `None` for other files, for tracks without edit list or `iTunSMPB` tag, and
/// for anything this parser does not understand: the audio is then used as decoded.
#[must_use]
pub fn gapless(path: &Path, sample_rate: u32) -> Option<Gapless> {
    let mut file = File::open(path).ok()?;
    let moov = read_moov(&mut file).ok()??;
    parse_moov(&moov, sample_rate)
}

/// Reads the body of the top-level `moov` box, if the file is an ISO media file.
fn read_moov(file: &mut File) -> io::Result<Option<Vec<u8>>> {
    let len = file.metadata()?.len();
    let mut pos = 0;
    let mut first = true;
    while pos + 8 <= len {
        file.seek(SeekFrom::Start(pos))?;
        let mut header = [0; 16];
        file.read_exact(&mut header[..8])?;
        let mut size = u64::from(u32::from_be_bytes(header[..4].try_into().expect("4 bytes")));
        let kind: [u8; 4] = header[4..8].try_into().expect("4 bytes");
        let mut header_len = 8;
        if size == 1 {
            file.read_exact(&mut header[8..16])?;
            size = u64::from_be_bytes(header[8..16].try_into().expect("8 bytes"));
            header_len = 16;
        } else if size == 0 {
            size = len - pos;
        }
        if first && &kind != b"ftyp" {
            return Ok(None);
        }
        first = false;
        if size < header_len {
            return Ok(None);
        }
        if &kind == b"moov" {
            let body_len = size - header_len;
            if body_len > MAX_MOOV {
                return Ok(None);
            }
            let mut body = vec![0; usize::try_from(body_len).map_err(io::Error::other)?];
            file.read_exact(&mut body)?;
            return Ok(Some(body));
        }
        pos += size;
    }
    Ok(None)
}

/// Iterates over the boxes of `data`: `(type, body)`.
fn boxes(mut data: &[u8]) -> impl Iterator<Item = (&[u8], &[u8])> {
    std::iter::from_fn(move || {
        if data.len() < 8 {
            return None;
        }
        let size = u32::from_be_bytes(data[..4].try_into().ok()?) as usize;
        let (header, size) = match size {
            0 => (8, data.len()),
            1 => {
                let large = u64::from_be_bytes(data.get(8..16)?.try_into().ok()?);
                (16, usize::try_from(large).ok()?)
            }
            n => (8, n),
        };
        if size < header || size > data.len() {
            return None;
        }
        let item = (&data[4..8], &data[header..size]);
        data = &data[size..];
        Some(item)
    })
}

fn child<'a>(data: &'a [u8], kind: &[u8]) -> Option<&'a [u8]> {
    boxes(data).find(|(k, _)| *k == kind).map(|(_, body)| body)
}

fn be_u32(data: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_be_bytes(data.get(at..at + 4)?.try_into().ok()?))
}

fn be_u64(data: &[u8], at: usize) -> Option<u64> {
    Some(u64::from_be_bytes(data.get(at..at + 8)?.try_into().ok()?))
}

/// `(timescale, duration)` of an `mvhd` or `mdhd` box.
fn timescale_and_duration(body: &[u8]) -> Option<(u32, u64)> {
    match body.first()? {
        0 => Some((be_u32(body, 12)?, u64::from(be_u32(body, 16)?))),
        1 => Some((be_u32(body, 20)?, be_u64(body, 24)?)),
        _ => None,
    }
}

/// The first edit of an `elst` box that plays media: `(segment_duration, media_time)`.
fn first_media_edit(body: &[u8]) -> Option<(u64, u64)> {
    let version = *body.first()?;
    let count = be_u32(body, 4)? as usize;
    let entry_len = if version == 1 { 20 } else { 12 };
    (0..count).find_map(|i| {
        let at = 8 + i * entry_len;
        let (duration, media_time) = if version == 1 {
            (
                be_u64(body, at)?,
                i64::from_be_bytes(body.get(at + 8..at + 16)?.try_into().ok()?),
            )
        } else {
            (
                u64::from(be_u32(body, at)?),
                i64::from(i32::from_be_bytes(
                    body.get(at + 4..at + 8)?.try_into().ok()?,
                )),
            )
        };
        // A media time of -1 is an empty edit (a pause), not a part of the track.
        u64::try_from(media_time).ok().map(|t| (duration, t))
    })
}

/// The `iTunSMPB` tag: `(priming, padding, length)` in samples.
fn itunsmpb(moov: &[u8]) -> Option<(u64, u64, u64)> {
    let meta = child(child(moov, b"udta")?, b"meta")?;
    // `meta` is a full box: a version and flags precede its children.
    let ilst = child(meta.get(4..)?, b"ilst")?;
    boxes(ilst)
        .filter(|(kind, _)| *kind == b"----")
        .find_map(|(_, tag)| {
            let name = child(tag, b"name")?;
            if name.get(4..)? != b"iTunSMPB" {
                return None;
            }
            let data = child(tag, b"data")?;
            let text = std::str::from_utf8(data.get(8..)?).ok()?;
            let fields: Vec<u64> = text
                .split_whitespace()
                .map(|f| u64::from_str_radix(f, 16))
                .collect::<Result<_, _>>()
                .ok()?;
            Some((*fields.get(1)?, *fields.get(2)?, *fields.get(3)?))
        })
}

/// Converts `value` from `from` units per second to `to` units per second.
fn rescale(value: u64, from: u32, to: u32) -> Option<u64> {
    if from == 0 {
        return None;
    }
    u64::try_from(u128::from(value) * u128::from(to) / u128::from(from)).ok()
}

fn parse_moov(moov: &[u8], sample_rate: u32) -> Option<Gapless> {
    let (movie_timescale, _) = timescale_and_duration(child(moov, b"mvhd")?)?;
    let trak = boxes(moov)
        .filter(|(k, _)| *k == b"trak")
        .map(|(_, b)| b)
        .find(|trak| {
            child(trak, b"mdia")
                .and_then(|mdia| child(mdia, b"hdlr"))
                .and_then(|hdlr| hdlr.get(8..12))
                == Some(b"soun".as_slice())
        })?;
    let (media_timescale, media_duration) =
        timescale_and_duration(child(child(trak, b"mdia")?, b"mdhd")?)?;
    let encoded = rescale(media_duration, media_timescale, sample_rate)?;

    let edit = child(trak, b"edts")
        .and_then(|edts| child(edts, b"elst"))
        .and_then(first_media_edit);
    if let Some((segment, media_time)) = edit {
        let skip = rescale(media_time, media_timescale, sample_rate)?;
        // A zero duration means "until the end of the media".
        let length = (segment > 0)
            .then(|| rescale(segment, movie_timescale, sample_rate))
            .flatten();
        if skip > 0 || length.is_some_and(|l| l < encoded) {
            return Some(Gapless {
                encoded,
                skip,
                length,
            });
        }
    }
    let (priming, _padding, length) = itunsmpb(moov)?;
    Some(Gapless {
        encoded,
        skip: priming,
        length: Some(length),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bx(kind: &[u8; 4], body: &[u8]) -> Vec<u8> {
        let mut out = u32::try_from(body.len() + 8)
            .unwrap()
            .to_be_bytes()
            .to_vec();
        out.extend_from_slice(kind);
        out.extend_from_slice(body);
        out
    }

    fn header_v0(timescale: u32, duration: u32) -> Vec<u8> {
        let mut body = vec![0; 12];
        body.extend(timescale.to_be_bytes());
        body.extend(duration.to_be_bytes());
        body.extend([0; 8]);
        body
    }

    fn hdlr(kind: &[u8; 4]) -> Vec<u8> {
        let mut body = vec![0; 8];
        body.extend_from_slice(kind);
        body.extend([0; 13]);
        bx(b"hdlr", &body)
    }

    fn elst(entries: &[(u32, i32)]) -> Vec<u8> {
        let mut body = vec![0; 4];
        body.extend(u32::try_from(entries.len()).unwrap().to_be_bytes());
        for &(duration, media_time) in entries {
            body.extend(duration.to_be_bytes());
            body.extend(media_time.to_be_bytes());
            body.extend([0, 1, 0, 0]);
        }
        bx(b"edts", &bx(b"elst", &body))
    }

    fn trak(handler: &[u8; 4], edts: Option<Vec<u8>>, duration: u32) -> Vec<u8> {
        let mdia = [bx(b"mdhd", &header_v0(44_100, duration)), hdlr(handler)].concat();
        let mut body = edts.unwrap_or_default();
        body.extend(bx(b"mdia", &mdia));
        bx(b"trak", &body)
    }

    fn itunes(tag: &str) -> Vec<u8> {
        let mut name = vec![0; 4];
        name.extend_from_slice(b"iTunSMPB");
        let mut data = vec![0, 0, 0, 1, 0, 0, 0, 0];
        data.extend_from_slice(tag.as_bytes());
        let freeform = bx(b"----", &[bx(b"name", &name), bx(b"data", &data)].concat());
        let mut meta = vec![0; 4];
        meta.extend(bx(b"ilst", &freeform));
        bx(b"udta", &bx(b"meta", &meta))
    }

    #[test]
    fn edit_list_skips_the_priming_samples() {
        let moov = [
            bx(b"mvhd", &header_v0(600, 0)),
            trak(b"vide", None, 1000),
            trak(b"soun", Some(elst(&[(7320, 2112)])), 540_672),
        ]
        .concat();
        let g = parse_moov(&moov, 44_100).unwrap();
        assert_eq!(
            g,
            Gapless {
                encoded: 540_672,
                skip: 2112,
                length: Some(7320 * 44_100 / 600)
            }
        );
    }

    #[test]
    fn empty_edits_are_ignored() {
        let moov = [
            bx(b"mvhd", &header_v0(44_100, 0)),
            trak(b"soun", Some(elst(&[(500, -1), (100_000, 1024)])), 102_400),
        ]
        .concat();
        let g = parse_moov(&moov, 44_100).unwrap();
        assert_eq!((g.skip, g.length), (1024, Some(100_000)));
    }

    #[test]
    fn itunsmpb_tag_is_read() {
        let tag = " 00000000 00000840 0000027C 0000000000083544 00000000 00000000";
        let moov = [
            bx(b"mvhd", &header_v0(44_100, 540_672)),
            trak(b"soun", None, 540_672),
            itunes(tag),
        ]
        .concat();
        let g = parse_moov(&moov, 44_100).unwrap();
        assert_eq!(
            g,
            Gapless {
                encoded: 540_672,
                skip: 2112,
                length: Some(537_924)
            }
        );
    }

    #[test]
    fn nothing_to_trim_without_metadata() {
        let moov = [
            bx(b"mvhd", &header_v0(44_100, 0)),
            trak(b"soun", None, 1000),
        ]
        .concat();
        assert_eq!(parse_moov(&moov, 44_100), None);
        // An edit that plays the whole track from the start trims nothing.
        let moov = [
            bx(b"mvhd", &header_v0(44_100, 0)),
            trak(b"soun", Some(elst(&[(1000, 0)])), 1000),
        ]
        .concat();
        assert_eq!(parse_moov(&moov, 44_100), None);
    }

    #[test]
    fn malformed_boxes_are_ignored() {
        assert_eq!(parse_moov(&[0, 0, 0, 200, b'm', b'v'], 44_100), None);
        assert_eq!(boxes(&[0, 0, 0, 4, b'a', b'b', b'c', b'd']).count(), 0);
    }

    #[test]
    fn trimming_needs_the_untrimmed_length() {
        let g = Gapless {
            encoded: 10,
            skip: 2,
            length: Some(5),
        };
        let mut samples: Vec<f32> = (0..10_u8).map(f32::from).collect();
        g.apply(&mut samples);
        assert_eq!(samples, [2.0, 3.0, 4.0, 5.0, 6.0]);
        // Already trimmed (or not the expected track): left alone.
        let mut other = vec![0.0; 7];
        g.apply(&mut other);
        assert_eq!(other.len(), 7);
    }

    #[test]
    fn non_mp4_files_are_skipped() {
        let wav =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/audio/short.wav");
        assert_eq!(gapless(&wav, 22_050), None);
    }
}
