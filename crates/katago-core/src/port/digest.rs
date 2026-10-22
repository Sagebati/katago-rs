//! Glue for the upstream files replaced by crates (port-manifest.toml [[ignored]]
//! `cpp/core/sha2.*` -> `sha2`, `cpp/core/md5.*` -> `md-5`). Same digests and the same
//! output layouts as upstream's `SHA2::get256/get512` and `MD5::get` (PORTING.md §10).

use md5::Md5;
use sha2::Digest;
use sha2::Sha256;
use sha2::Sha512;

/// The bytes a `const char*` would see: up to the first NUL.
fn c_str_bytes(msg: &str) -> &[u8] {
  let b = msg.as_bytes();
  match b.iter().position(|&c| c == 0) {
    Some(p) => &b[..p],
    None => b,
  }
}

/// `SHA2::get256(const char* msg, uint64_t hash[4])`: big-endian u64 words.
pub fn sha2_get256_u64(msg: &str) -> [u64; 4] {
  let d = Sha256::digest(c_str_bytes(msg));
  let mut hash = [0u64; 4];
  for (i, h) in hash.iter_mut().enumerate() {
    *h = u64::from_be_bytes(d[8 * i..8 * i + 8].try_into().expect("8 bytes"));
  }
  hash
}

/// `SHA2::get256(const char* msg, char hash[65])`: lowercase hex.
pub fn sha2_get256_hex(msg: &str) -> String {
  let d = Sha256::digest(c_str_bytes(msg));
  let mut s = String::with_capacity(64);
  for b in d.iter() {
    s.push_str(&format!("{:02x}", b));
  }
  s
}

/// `SHA2::get512(const char* msg, char hash[129])`: lowercase hex.
pub fn sha2_get512_hex(msg: &str) -> String {
  let d = Sha512::digest(c_str_bytes(msg));
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
