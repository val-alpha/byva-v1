//! Minimal zip archive reader for PyTorch checkpoint files.
//!
//! `torch.save` (new zipfile serialization) writes every entry with the
//! stored (method 0) compression method; this reader supports exactly that
//! subset and rejects anything else with a typed error. Sizes are taken from
//! the central directory; zip64 extra fields are parsed when the 32-bit
//! fields carry the 0xFFFFFFFF sentinel.

use crate::error::ConvertError;

const EOCD_SIGNATURE: u32 = 0x0605_4b50;
const CENTRAL_SIGNATURE: u32 = 0x0201_4b50;
const LOCAL_SIGNATURE: u32 = 0x0403_4b50;
const EOCD_MIN_SIZE: u64 = 22;
const MAX_EOCD_SCAN: u64 = 65_571;
const ZIP64_MARKER: u32 = 0xFFFF_FFFF;
const ZIP64_EXTRA_ID: u16 = 0x0001;

/// One stored zip entry resolved to a byte range of the archive.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ZipEntry {
    name: String,
    offset: u64,
    size: u64,
}

impl ZipEntry {
    /// Entry name as stored in the archive.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Uncompressed (stored) byte length.
    pub const fn size(&self) -> u64 {
        self.size
    }
}

/// Parses the central directory of a stored-only zip archive.
///
/// # Errors
/// Fails when the file is not a zip archive, the directory is malformed, an
/// entry uses compression other than stored, or a name is not UTF-8.
pub fn read_directory(bytes: &[u8]) -> Result<Vec<ZipEntry>, ConvertError> {
    let eocd = find_eocd(bytes)?;
    let mut cursor = central_directory_offset(bytes, eocd)?;
    let mut entries = Vec::new();
    loop {
        if cursor + 4 > bytes.len() as u64 {
            break;
        }
        let signature =
            read_u32(bytes, cursor).ok_or(ConvertError::BadCentralDirectory { offset: cursor })?;
        if signature != CENTRAL_SIGNATURE {
            break;
        }
        let header = read_slice(bytes, cursor, 46)
            .ok_or(ConvertError::BadCentralDirectory { offset: cursor })?;
        let method = u16::from_le_bytes([header[10], header[11]]);
        let mut uncompressed_size =
            u32::from_le_bytes([header[24], header[25], header[26], header[27]]) as u64;
        let name_len = u16::from_le_bytes([header[28], header[29]]) as u64;
        let extra_len = u16::from_le_bytes([header[30], header[31]]) as u64;
        let comment_len = u16::from_le_bytes([header[32], header[33]]) as u64;
        let local_offset =
            u32::from_le_bytes([header[42], header[43], header[44], header[45]]) as u64;
        let name_start = cursor + 46;
        let name_bytes = read_slice(bytes, name_start, name_len)
            .ok_or(ConvertError::BadCentralDirectory { offset: cursor })?;
        let name = std::str::from_utf8(name_bytes).map_err(|_| ConvertError::InvalidEntryName)?;
        if method != 0 {
            return Err(ConvertError::UnsupportedZipMethod {
                entry: name.to_string(),
                method,
            });
        }
        let extra = read_slice(bytes, name_start + name_len, extra_len).unwrap_or(&[]);
        if uncompressed_size == ZIP64_MARKER as u64 {
            uncompressed_size = zip64_size(extra)?;
        }
        entries.push(ZipEntry {
            name: name.to_string(),
            offset: local_offset,
            size: uncompressed_size,
        });
        cursor = name_start + name_len + extra_len + comment_len;
    }
    Ok(entries)
}

/// Extracts one entry's stored bytes, resolving its local header.
///
/// # Errors
/// Fails when the local header is malformed or the declared size exceeds
/// the archive.
pub fn read_entry<'a>(bytes: &'a [u8], entry: &ZipEntry) -> Result<&'a [u8], ConvertError> {
    let header = read_slice(bytes, entry.offset, 30).ok_or(ConvertError::BadCentralDirectory {
        offset: entry.offset,
    })?;
    let signature = u32::from_le_bytes([header[0], header[1], header[2], header[3]]);
    if signature != LOCAL_SIGNATURE {
        return Err(ConvertError::BadCentralDirectory {
            offset: entry.offset,
        });
    }
    let name_len = u16::from_le_bytes([header[26], header[27]]) as u64;
    let extra_len = u16::from_le_bytes([header[28], header[29]]) as u64;
    let data_start = entry.offset + 30 + name_len + extra_len;
    read_slice(bytes, data_start, entry.size)
        .ok_or(ConvertError::BadCentralDirectory { offset: data_start })
}

/// Finds the entry whose name ends with `suffix` (PyTorch prefixes entries
/// with the archive stem, e.g. `ckpt/data.pkl`).
///
/// # Errors
/// Fails when no exactly-one matching entry exists.
pub fn find_entry<'a>(entries: &'a [ZipEntry], suffix: &str) -> Result<&'a ZipEntry, ConvertError> {
    let matches: Vec<&ZipEntry> = entries
        .iter()
        .filter(|entry| entry.name().ends_with(suffix))
        .collect();
    match matches.as_slice() {
        [entry] => Ok(entry),
        [] => Err(ConvertError::MissingEntry {
            name: suffix.to_string(),
        }),
        [..] => Err(ConvertError::MissingEntry {
            name: suffix.to_string(),
        }),
    }
}

const CENTRAL_OFFSET_OFFSET: u64 = 16;
const EOCD64_LOCATOR_SIGNATURE: u32 = 0x0706_4b50;
const EOCD64_SIGNATURE: u32 = 0x0606_4b50;

fn central_directory_offset(bytes: &[u8], eocd: u64) -> Result<u64, ConvertError> {
    let offset = read_u32(bytes, eocd + CENTRAL_OFFSET_OFFSET).ok_or(ConvertError::NotAZipFile)?;
    if offset != ZIP64_MARKER {
        return Ok(u64::from(offset));
    }
    let locator = eocd.checked_sub(20).ok_or(ConvertError::NotAZipFile)?;
    let signature = read_u32(bytes, locator).ok_or(ConvertError::NotAZipFile)?;
    if signature != EOCD64_LOCATOR_SIGNATURE {
        return Err(ConvertError::NotAZipFile);
    }
    let eocd64_offset = read_u64(bytes, locator + 8).ok_or(ConvertError::NotAZipFile)?;
    let signature = read_u32(bytes, eocd64_offset).ok_or(ConvertError::NotAZipFile)?;
    if signature != EOCD64_SIGNATURE {
        return Err(ConvertError::BadCentralDirectory {
            offset: eocd64_offset,
        });
    }
    read_u64(bytes, eocd64_offset + 48).ok_or(ConvertError::BadCentralDirectory {
        offset: eocd64_offset,
    })
}

fn read_u64(bytes: &[u8], offset: u64) -> Option<u64> {
    let chunk = read_slice(bytes, offset, 8)?;
    let mut raw = [0u8; 8];
    raw.copy_from_slice(chunk);
    Some(u64::from_le_bytes(raw))
}

fn find_eocd(bytes: &[u8]) -> Result<u64, ConvertError> {
    if bytes.len() < EOCD_MIN_SIZE as usize {
        return Err(ConvertError::NotAZipFile);
    }
    let lowest = bytes.len() as u64 - EOCD_MIN_SIZE;
    let highest_scan = lowest.saturating_sub(MAX_EOCD_SCAN);
    let mut cursor = lowest;
    while cursor >= highest_scan {
        if read_u32(bytes, cursor) == Some(EOCD_SIGNATURE) {
            return Ok(cursor);
        }
        if cursor == 0 {
            break;
        }
        cursor -= 1;
    }
    Err(ConvertError::NotAZipFile)
}

fn zip64_size(extra: &[u8]) -> Result<u64, ConvertError> {
    let malformed = || ConvertError::BadCentralDirectory { offset: 0 };
    let mut cursor = 0usize;
    while cursor + 4 <= extra.len() {
        let id = u16::from_le_bytes([extra[cursor], extra[cursor + 1]]);
        let len = u16::from_le_bytes([extra[cursor + 2], extra[cursor + 3]]) as usize;
        let start = cursor + 4;
        let end = start.checked_add(len).ok_or_else(malformed)?;
        if id == ZIP64_EXTRA_ID && end <= extra.len() && len >= 8 {
            let chunk: [u8; 8] = extra[start..start + 8]
                .try_into()
                .map_err(|_| malformed())?;
            return Ok(u64::from_le_bytes(chunk));
        }
        cursor = end;
    }
    Err(malformed())
}

fn read_u32(bytes: &[u8], offset: u64) -> Option<u32> {
    let chunk = read_slice(bytes, offset, 4)?;
    Some(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
}

fn read_slice(bytes: &[u8], offset: u64, len: u64) -> Option<&[u8]> {
    let start = usize::try_from(offset).ok()?;
    let end = start.checked_add(usize::try_from(len).ok()?)?;
    bytes.get(start..end)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stored_zip_with_method(entries: &[(&str, &[u8])], method: u16) -> Vec<u8> {
        let method_bytes = method.to_le_bytes();
        let mut out = Vec::new();
        let mut central = Vec::new();
        let mut offsets = Vec::new();
        for (name, data) in entries {
            offsets.push((name.as_bytes(), data.len() as u32, out.len() as u32));
            out.extend_from_slice(&LOCAL_SIGNATURE.to_le_bytes());
            out.extend_from_slice(&[20, 0]);
            out.extend_from_slice(&[0, 0]);
            out.extend_from_slice(&method_bytes);
            out.extend_from_slice(&[0, 0, 0, 0]);
            out.extend_from_slice(&[0, 0, 0, 0]);
            out.extend_from_slice(&(data.len() as u32).to_le_bytes());
            out.extend_from_slice(&(data.len() as u32).to_le_bytes());
            out.extend_from_slice(&(name.len() as u16).to_le_bytes());
            out.extend_from_slice(&[0, 0]);
            out.extend_from_slice(name.as_bytes());
            out.extend_from_slice(data);
        }
        let cd_start = out.len() as u32;
        for (name, size, offset) in &offsets {
            central.extend_from_slice(&CENTRAL_SIGNATURE.to_le_bytes());
            central.extend_from_slice(&[20, 0]);
            central.extend_from_slice(&[20, 0]);
            central.extend_from_slice(&[0, 0]);
            central.extend_from_slice(&method_bytes);
            central.extend_from_slice(&[0, 0, 0, 0]);
            central.extend_from_slice(&[0, 0, 0, 0]);
            central.extend_from_slice(&size.to_le_bytes());
            central.extend_from_slice(&size.to_le_bytes());
            central.extend_from_slice(&(name.len() as u16).to_le_bytes());
            central.extend_from_slice(&[0, 0]);
            central.extend_from_slice(&[0, 0]);
            central.extend_from_slice(&[0, 0]);
            central.extend_from_slice(&[0, 0]);
            central.extend_from_slice(&[0, 0, 0, 0]);
            central.extend_from_slice(&offset.to_le_bytes());
            central.extend_from_slice(name);
        }
        out.extend_from_slice(&central);
        let cd_size = (out.len() as u32).wrapping_sub(cd_start);
        out.extend_from_slice(&EOCD_SIGNATURE.to_le_bytes());
        out.extend_from_slice(&[0, 0]);
        out.extend_from_slice(&[0, 0]);
        out.extend_from_slice(&(offsets.len() as u16).to_le_bytes());
        out.extend_from_slice(&(offsets.len() as u16).to_le_bytes());
        out.extend_from_slice(&cd_size.to_le_bytes());
        out.extend_from_slice(&cd_start.to_le_bytes());
        out.extend_from_slice(&[0, 0]);
        out
    }

    fn stored_zip(entries: &[(&str, &[u8])]) -> Vec<u8> {
        stored_zip_with_method(entries, 0)
    }

    #[test]
    fn directory_and_entries_round_trip() {
        let zip = stored_zip(&[
            ("stem/data.pkl", b"pickle-bytes"),
            ("stem/data/0", &[1, 2, 3, 4]),
        ]);
        let entries =
            read_directory(&zip).unwrap_or_else(|error| panic!("valid zip rejected: {error}"));
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].name(), "stem/data.pkl");
        let found = find_entry(&entries, "/data/0")
            .unwrap_or_else(|error| panic!("entry not found: {error}"));
        assert_eq!(found.size(), 4);
        assert_eq!(
            read_entry(&zip, found).expect("entry readable"),
            &[1u8, 2, 3, 4][..]
        );
    }

    #[test]
    fn suffix_match_requires_exact_boundary() {
        let zip = stored_zip(&[("stem/data/10", b"abc"), ("stem/data/0", b"de")]);
        let entries = read_directory(&zip).expect("valid zip");
        assert_eq!(find_entry(&entries, "/data/0").expect("entry").size(), 2);
        assert_eq!(find_entry(&entries, "/data/10").expect("entry").size(), 3);
        assert!(matches!(
            find_entry(&entries, "/data/1"),
            Err(ConvertError::MissingEntry { .. })
        ));
        assert!(matches!(
            find_entry(&entries, "/data/2"),
            Err(ConvertError::MissingEntry { .. })
        ));
    }

    #[test]
    fn non_zip_bytes_are_rejected() {
        let mut bytes = vec![0u8; 64];
        bytes[63] = 0x50;
        assert!(matches!(
            read_directory(&bytes),
            Err(ConvertError::NotAZipFile)
        ));
        assert!(matches!(
            read_directory(&[]),
            Err(ConvertError::NotAZipFile)
        ));
    }

    #[test]
    fn deflated_entries_are_rejected() {
        let zip = stored_zip_with_method(&[("stem/data.pkl", b"abc")], 8);
        assert!(matches!(
            read_directory(&zip),
            Err(ConvertError::UnsupportedZipMethod { method: 8, .. })
        ));
    }

    #[test]
    fn truncated_central_directory_is_rejected() {
        let zip = stored_zip(&[("stem/data.pkl", b"abc")]);
        let truncated = &zip[..zip.len() - 10];
        assert!(matches!(
            read_directory(truncated),
            Err(ConvertError::NotAZipFile | ConvertError::BadCentralDirectory { .. })
        ));
    }
}
