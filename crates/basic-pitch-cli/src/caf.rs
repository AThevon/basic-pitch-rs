// Copyright 2026 Adrien Thevon
// SPDX-License-Identifier: Apache-2.0

//! Gapless trimming of CAF audio.
//!
//! A Core Audio Format file with compressed audio (AAC from `afconvert` or
//! `AVAudioRecorder`, for instance) records its encoder delay in the packet table chunk
//! (`pakt`): priming frames at the start, remainder frames at the end. Core Audio drops
//! them when it decodes the file, and so does Python Basic Pitch on macOS (librosa reads
//! these files through `audioread`). symphonia 0.6 reads the chunk but leaves the
//! samples in, which would make every note about 48 ms late.

use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::Path;

use crate::mp4::Gapless;

/// Largest number of chunks visited, to bound the work on malformed files.
const MAX_CHUNKS: usize = 4096;

/// Finds the priming and remainder frames of a CAF file.
///
/// Returns `None` for other files, for files without a `pakt` chunk or with nothing to
/// trim, and for anything this parser does not understand: the audio is then used as
/// decoded.
#[must_use]
pub fn gapless(path: &Path) -> Option<Gapless> {
    let mut file = File::open(path).ok()?;
    read_pakt(&mut file)
        .ok()?
        .and_then(|pakt| parse_pakt(&pakt))
}

/// Reads the first 24 bytes of the `pakt` chunk, if the file is a CAF file.
fn read_pakt(file: &mut File) -> io::Result<Option<[u8; 24]>> {
    let len = file.metadata()?.len();
    let mut header = [0; 8];
    file.read_exact(&mut header)?;
    if &header[..4] != b"caff" {
        return Ok(None);
    }
    let mut pos = 8;
    for _ in 0..MAX_CHUNKS {
        if pos + 12 > len {
            break;
        }
        file.seek(SeekFrom::Start(pos))?;
        let mut chunk = [0; 12];
        file.read_exact(&mut chunk)?;
        let kind = &chunk[..4];
        let size = i64::from_be_bytes(chunk[4..].try_into().expect("8 bytes"));
        if kind == b"pakt" {
            let mut body = [0; 24];
            file.read_exact(&mut body)?;
            return Ok(Some(body));
        }
        // Only the audio data chunk may have an unknown size (-1), and it runs to the
        // end of the file.
        let Ok(size) = u64::try_from(size) else {
            break;
        };
        pos = pos.saturating_add(12).saturating_add(size);
    }
    Ok(None)
}

/// Decodes the header of a `pakt` chunk: number of packets, valid frames, priming frames
/// and remainder frames.
fn parse_pakt(body: &[u8; 24]) -> Option<Gapless> {
    let valid = i64::from_be_bytes(body[8..16].try_into().ok()?);
    let priming = i32::from_be_bytes(body[16..20].try_into().ok()?);
    let remainder = i32::from_be_bytes(body[20..24].try_into().ok()?);
    let (valid, priming, remainder) = (
        u64::try_from(valid).ok()?,
        u64::try_from(priming).ok()?,
        u64::try_from(remainder).ok()?,
    );
    if priming == 0 && remainder == 0 {
        return None;
    }
    Some(Gapless {
        encoded: valid.checked_add(priming)?.checked_add(remainder)?,
        skip: priming,
        length: Some(valid),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pakt(packets: i64, valid: i64, priming: i32, remainder: i32) -> [u8; 24] {
        let mut body = [0; 24];
        body[..8].copy_from_slice(&packets.to_be_bytes());
        body[8..16].copy_from_slice(&valid.to_be_bytes());
        body[16..20].copy_from_slice(&priming.to_be_bytes());
        body[20..24].copy_from_slice(&remainder.to_be_bytes());
        body
    }

    fn chunk(kind: &[u8; 4], body: &[u8], size: i64) -> Vec<u8> {
        let mut out = kind.to_vec();
        out.extend(size.to_be_bytes());
        out.extend_from_slice(body);
        out
    }

    #[test]
    fn packet_table_gives_the_priming_and_remainder() {
        // afconvert -d aac on 480 000 frames: 471 packets of 1024 frames.
        let g = parse_pakt(&pakt(471, 480_000, 2112, 192)).unwrap();
        assert_eq!(
            g,
            Gapless {
                encoded: 482_304,
                skip: 2112,
                length: Some(480_000)
            }
        );
        assert_eq!(parse_pakt(&pakt(10, 10_240, 0, 0)), None);
        assert_eq!(parse_pakt(&pakt(10, -1, 2112, 0)), None);
    }

    #[test]
    fn chunks_are_walked_to_the_packet_table() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.caf");
        let mut bytes = b"caff\x00\x01\x00\x00".to_vec();
        bytes.extend(chunk(b"desc", &[0; 32], 32));
        bytes.extend(chunk(b"free", &[0; 5], 5));
        bytes.extend(chunk(b"pakt", &pakt(3, 3000, 2112, 960), 24));
        bytes.extend(chunk(b"data", &[0; 4], -1));
        std::fs::write(&path, &bytes).unwrap();
        assert_eq!(
            gapless(&path),
            Some(Gapless {
                encoded: 6072,
                skip: 2112,
                length: Some(3000)
            })
        );

        // The data chunk of unknown size ends the walk.
        let mut bytes = b"caff\x00\x01\x00\x00".to_vec();
        bytes.extend(chunk(b"data", &[0; 4], -1));
        bytes.extend(chunk(b"pakt", &pakt(3, 3000, 2112, 960), 24));
        std::fs::write(&path, &bytes).unwrap();
        assert_eq!(gapless(&path), None);
    }

    #[test]
    fn other_files_are_skipped() {
        let wav =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/audio/short.wav");
        assert_eq!(gapless(&wav), None);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("truncated.caf");
        std::fs::write(&path, b"caff\x00\x01\x00\x00pakt").unwrap();
        assert_eq!(gapless(&path), None);
    }
}
