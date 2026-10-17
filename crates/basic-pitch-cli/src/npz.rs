// Copyright 2026 Adrien Thevon
// SPDX-License-Identifier: Apache-2.0

//! A minimal writer of NumPy `.npz` archives: uncompressed ZIP files of `.npy` arrays,
//! as written by `numpy.savez`.
//!
//! Only what the posteriorgrams need: little-endian `float32` matrices, ZIP entries
//! stored without compression, archives below 4 GiB.

use std::io::{self, Write};

/// A `float32` matrix in row-major order.
#[derive(Clone, Copy, Debug)]
pub struct Matrix<'a> {
    /// Name of the array in the archive (`<name>.npy`).
    pub name: &'a str,
    /// Number of rows.
    pub rows: usize,
    /// Number of columns.
    pub cols: usize,
    /// `rows * cols` values, row after row.
    pub data: &'a [f32],
}

/// Encodes a matrix as a `.npy` file (format version 1.0).
fn npy(m: &Matrix<'_>) -> Vec<u8> {
    const MAGIC: &[u8] = b"\x93NUMPY\x01\x00";
    let mut header = format!(
        "{{'descr': '<f4', 'fortran_order': False, 'shape': ({}, {}), }}",
        m.rows, m.cols
    );
    // numpy pads the header with spaces and a newline so that the data starts on a
    // multiple of 64 bytes.
    let unpadded = MAGIC.len() + 2 + header.len() + 1;
    header.extend(std::iter::repeat_n(
        ' ',
        unpadded.next_multiple_of(64) - unpadded,
    ));
    header.push('\n');

    let mut out = Vec::with_capacity(MAGIC.len() + 2 + header.len() + 4 * m.data.len());
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(
        &u16::try_from(header.len())
            .expect("the header is short")
            .to_le_bytes(),
    );
    out.extend_from_slice(header.as_bytes());
    for x in m.data {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

/// CRC-32 (IEEE 802.3), as used by ZIP.
fn crc32(data: &[u8]) -> u32 {
    const TABLE: [u32; 256] = {
        let mut table = [0_u32; 256];
        let mut i = 0;
        while i < 256 {
            #[allow(clippy::cast_possible_truncation, reason = "i < 256")]
            let mut c = i as u32;
            let mut k = 0;
            while k < 8 {
                c = if c & 1 == 1 {
                    0xEDB8_8320 ^ (c >> 1)
                } else {
                    c >> 1
                };
                k += 1;
            }
            table[i] = c;
            i += 1;
        }
        table
    };
    !data.iter().fold(!0_u32, |c, &b| {
        TABLE[usize::from(u8::try_from(c & 0xFF).expect("masked")) ^ usize::from(b)] ^ (c >> 8)
    })
}

fn too_large() -> io::Error {
    io::Error::other("the posteriorgrams exceed 4 GiB, the limit of the .npz writer")
}

/// Writes `matrices` as an uncompressed `.npz` archive.
///
/// # Errors
///
/// Any error of `out`, or an error if the archive would exceed 4 GiB.
///
/// # Panics
///
/// If a matrix does not hold `rows * cols` values.
pub fn write_npz(matrices: &[Matrix<'_>], mut out: impl Write) -> io::Result<()> {
    // DOS date of 1980-01-01, 00:00: the archive does not depend on the clock.
    const DOS_TIME: u16 = 0;
    const DOS_DATE: u16 = (1 << 5) | 1;
    const VERSION: u16 = 20;

    let mut central = Vec::new();
    let mut offset: u32 = 0;
    for m in matrices {
        assert_eq!(m.data.len(), m.rows * m.cols, "{}: shape", m.name);
        let name = format!("{}.npy", m.name);
        let name_len = u16::try_from(name.len()).map_err(|_| too_large())?;
        let body = npy(m);
        let size = u32::try_from(body.len()).map_err(|_| too_large())?;
        let crc = crc32(&body);

        let mut local = Vec::with_capacity(30 + name.len());
        local.extend_from_slice(&0x0403_4b50_u32.to_le_bytes());
        local.extend_from_slice(&VERSION.to_le_bytes());
        local.extend_from_slice(&0_u16.to_le_bytes()); // flags
        local.extend_from_slice(&0_u16.to_le_bytes()); // stored
        local.extend_from_slice(&DOS_TIME.to_le_bytes());
        local.extend_from_slice(&DOS_DATE.to_le_bytes());
        local.extend_from_slice(&crc.to_le_bytes());
        local.extend_from_slice(&size.to_le_bytes()); // compressed
        local.extend_from_slice(&size.to_le_bytes()); // uncompressed
        local.extend_from_slice(&name_len.to_le_bytes());
        local.extend_from_slice(&0_u16.to_le_bytes()); // extra field
        local.extend_from_slice(name.as_bytes());
        out.write_all(&local)?;
        out.write_all(&body)?;

        central.extend_from_slice(&0x0201_4b50_u32.to_le_bytes());
        central.extend_from_slice(&VERSION.to_le_bytes()); // made by
        central.extend_from_slice(&local[4..30]); // same fields as the local header
        central.extend_from_slice(&0_u16.to_le_bytes()); // comment
        central.extend_from_slice(&0_u16.to_le_bytes()); // disk
        central.extend_from_slice(&0_u16.to_le_bytes()); // internal attributes
        central.extend_from_slice(&0_u32.to_le_bytes()); // external attributes
        central.extend_from_slice(&offset.to_le_bytes());
        central.extend_from_slice(name.as_bytes());

        let entry_len = u32::try_from(local.len()).map_err(|_| too_large())?;
        offset = offset
            .checked_add(entry_len)
            .and_then(|o| o.checked_add(size))
            .ok_or_else(too_large)?;
    }

    let count = u16::try_from(matrices.len()).map_err(|_| too_large())?;
    let central_len = u32::try_from(central.len()).map_err(|_| too_large())?;
    out.write_all(&central)?;
    let mut end = Vec::with_capacity(22);
    end.extend_from_slice(&0x0605_4b50_u32.to_le_bytes());
    end.extend_from_slice(&0_u16.to_le_bytes()); // this disk
    end.extend_from_slice(&0_u16.to_le_bytes()); // disk of the central directory
    end.extend_from_slice(&count.to_le_bytes());
    end.extend_from_slice(&count.to_le_bytes());
    end.extend_from_slice(&central_len.to_le_bytes());
    end.extend_from_slice(&offset.to_le_bytes());
    end.extend_from_slice(&0_u16.to_le_bytes()); // comment
    out.write_all(&end)?;
    out.flush()
}

/// Reads back the entries of an archive written by [`write_npz`]: `(file name, .npy
/// bytes)`, in order. It only supports stored (uncompressed) entries.
///
/// # Errors
///
/// [`io::ErrorKind::InvalidData`] if the archive is malformed, compressed, or an entry
/// fails its CRC check.
pub fn read_npz(zip: &[u8]) -> io::Result<Vec<(String, Vec<u8>)>> {
    let invalid = |what: &str| io::Error::new(io::ErrorKind::InvalidData, what.to_owned());
    let u16_at = |at: usize| -> io::Result<usize> {
        let b = zip
            .get(at..at + 2)
            .ok_or_else(|| invalid("truncated archive"))?;
        Ok(usize::from(u16::from_le_bytes([b[0], b[1]])))
    };
    let u32_at = |at: usize| -> io::Result<u32> {
        let b = zip
            .get(at..at + 4)
            .ok_or_else(|| invalid("truncated archive"))?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    };
    let end = zip
        .len()
        .checked_sub(22)
        .ok_or_else(|| invalid("not a ZIP archive"))?;
    if u32_at(end)? != 0x0605_4b50 {
        return Err(invalid("no end of central directory"));
    }
    let mut at = u32_at(end + 16)? as usize;
    let mut entries = Vec::new();
    for _ in 0..u16_at(end + 10)? {
        if u32_at(at)? != 0x0201_4b50 || u16_at(at + 10)? != 0 {
            return Err(invalid("unsupported central directory entry"));
        }
        let crc = u32_at(at + 16)?;
        let size = u32_at(at + 20)? as usize;
        let name_len = u16_at(at + 28)?;
        let local = u32_at(at + 42)? as usize;
        let name = zip
            .get(at + 46..at + 46 + name_len)
            .and_then(|n| String::from_utf8(n.to_vec()).ok())
            .ok_or_else(|| invalid("bad file name"))?;
        if u32_at(local)? != 0x0403_4b50 {
            return Err(invalid("bad local header"));
        }
        let data_at = local + 30 + u16_at(local + 26)? + u16_at(local + 28)?;
        let data = zip
            .get(data_at..data_at + size)
            .ok_or_else(|| invalid("truncated entry"))?
            .to_vec();
        if crc32(&data) != crc {
            return Err(invalid("CRC mismatch"));
        }
        entries.push((name, data));
        at += 46 + name_len + u16_at(at + 30)? + u16_at(at + 32)?;
    }
    Ok(entries)
}

#[cfg(test)]
mod tests {
    use ndarray_npy::ReadNpyExt;

    use super::*;

    fn entries(zip: &[u8]) -> Vec<(String, Vec<u8>)> {
        read_npz(zip).unwrap()
    }

    #[test]
    fn crc32_matches_the_standard_check_value() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        assert_eq!(crc32(b""), 0);
    }

    #[test]
    fn npy_header_is_aligned() {
        let m = Matrix {
            name: "x",
            rows: 2,
            cols: 3,
            data: &[0.0; 6],
        };
        let bytes = npy(&m);
        let header_len = usize::from(u16::from_le_bytes([bytes[8], bytes[9]]));
        assert_eq!((10 + header_len) % 64, 0);
        assert_eq!(bytes[10 + header_len - 1], b'\n');
        assert_eq!(bytes.len(), 10 + header_len + 24);
    }

    #[test]
    fn archive_round_trips() {
        let a: Vec<f32> = (0..12_u8).map(f32::from).collect();
        let b = [f32::NAN, -0.0, 1e-30, f32::MAX];
        let mut zip = Vec::new();
        write_npz(
            &[
                Matrix {
                    name: "note",
                    rows: 3,
                    cols: 4,
                    data: &a,
                },
                Matrix {
                    name: "contour",
                    rows: 1,
                    cols: 4,
                    data: &b,
                },
            ],
            &mut zip,
        )
        .unwrap();

        let entries = entries(&zip);
        assert_eq!(entries[0].0, "note.npy");
        assert_eq!(entries[1].0, "contour.npy");
        let note = ndarray::Array2::<f32>::read_npy(entries[0].1.as_slice()).unwrap();
        assert_eq!(note.dim(), (3, 4));
        assert_eq!(note.as_slice().unwrap(), a.as_slice());
        let contour = ndarray::Array2::<f32>::read_npy(entries[1].1.as_slice()).unwrap();
        let bits: Vec<u32> = contour.iter().map(|x| x.to_bits()).collect();
        assert_eq!(bits, b.map(f32::to_bits));
    }

    #[test]
    fn empty_matrices_are_fine() {
        let mut zip = Vec::new();
        write_npz(
            &[Matrix {
                name: "note",
                rows: 0,
                cols: 88,
                data: &[],
            }],
            &mut zip,
        )
        .unwrap();
        let entries = entries(&zip);
        let note = ndarray::Array2::<f32>::read_npy(entries[0].1.as_slice()).unwrap();
        assert_eq!(note.dim(), (0, 88));
    }

    #[test]
    fn corrupt_archives_are_rejected() {
        let mut zip = Vec::new();
        let m = Matrix {
            name: "x",
            rows: 1,
            cols: 2,
            data: &[1.0, 2.0],
        };
        write_npz(&[m], &mut zip).unwrap();
        let mut flipped = zip.clone();
        flipped[100] ^= 1;
        assert!(read_npz(&flipped).is_err());
        assert!(read_npz(&zip[..20]).is_err());
        assert!(read_npz(b"").is_err());
    }
}
