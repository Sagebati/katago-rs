//! Glue for the upstream files replaced by crates (port-manifest.toml [[ignored]]
//! `cpp/core/sha2.*` -> `sha2`, `cpp/core/md5.*` -> `md-5`). Same digests and the same
//! output layouts as upstream's `SHA2::get256/get512` and `MD5::get` (PORTING.md §10).

use md5::Md5;
use sha2::Digest;
use sha2::Sha256;
use sha2::Sha512;

use crate::port::cpp;

/// `SHA2::get256(const char* msg, uint64_t hash[4])`: big-endian u64 words.
pub fn sha2_get256_u64(msg: &str) -> [u64; 4] {
  let d = Sha256::digest(cpp::c_str(msg).as_bytes());
  let mut hash = [0u64; 4];
  for (i, h) in hash.iter_mut().enumerate() {
    *h = u64::from_be_bytes(d[8 * i..8 * i + 8].try_into().expect("8 bytes"));
  }
  hash
}

/// `SHA2::get256(const char* msg, char hash[65])`: lowercase hex.
pub fn sha2_get256_hex(msg: &str) -> String {
  sha2_get256_hex_bytes(cpp::c_str(msg).as_bytes())
}

/// [`sha2_get256_hex`] of raw bytes (no NUL truncation): lowercase hex of the SHA-256 of `data`.
pub fn sha2_get256_hex_bytes(data: &[u8]) -> String {
  let d = Sha256::digest(data);
  let mut s = String::with_capacity(64);
  for b in d.iter() {
    s.push_str(&format!("{:02x}", b));
  }
  s
}

/// `SHA2::get512(const char* msg, char hash[129])`: lowercase hex.
pub fn sha2_get512_hex(msg: &str) -> String {
  let d = Sha512::digest(cpp::c_str(msg).as_bytes());
  let mut s = String::with_capacity(128);
  for b in d.iter() {
    s.push_str(&format!("{:02x}", b));
  }
  s
}

/// `MD5::get(const char* data, size_t len, uint32_t hash[4])`: the state words,
/// little-endian (PORTING.md §10).
pub fn md5_get(data: &[u8]) -> [u32; 4] {
  let d = Md5::digest(data);
  let mut hash = [0u32; 4];
  for (i, h) in hash.iter_mut().enumerate() {
    *h = u32::from_le_bytes(d[4 * i..4 * i + 4].try_into().expect("4 bytes"));
  }
  hash
}
