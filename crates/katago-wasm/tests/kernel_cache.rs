//! The blob format of the WebGPU kernel cache (third_party/cubecl-wgpu, `wgsl_cache`) that the worker keeps
//! in the Cache API: a blob written by hand imports, exports back byte for byte, and damaged or truncated
//! blobs are refused whole. (Lookups and recording run in the browser: `just web-test`, "kernel cache".)
//! One test function: the cache is a process global.

use cubecl_wgpu::wgsl_cache;

fn put_u32(out: &mut Vec<u8>, x: u32) {
  out.extend_from_slice(&x.to_le_bytes());
}

fn put_str(out: &mut Vec<u8>, s: &str) {
  put_u32(out, s.len() as u32);
  out.extend_from_slice(s.as_bytes());
}

fn fnv1a(bytes: &[u8]) -> u64 {
  bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &b| (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3))
}

/// One entry: device properties hash, kernel id stable hash, stable format, entry point, WGSL, buffer
/// visibilities (Uniform, Read, ReadWrite = 0, 1, 2), buffer IO (ReadOnly, WriteOnly, ReadWrite, Dead = 0..3),
/// shared memory size; then the FNV-1a of everything before it.
fn blob() -> Vec<u8> {
  let mut b = b"CCLWGSL2".to_vec();
  put_u32(&mut b, 1);
  b.extend_from_slice(&7u64.to_le_bytes());
  b.extend_from_slice(&0x0123_4567_89ab_cdef_0011_2233_4455_6677u128.to_le_bytes());
  put_str(&mut b, "my::Kernel-u32-Dim3 { x: 64, y: 1, z: 1 }-Checked-None");
  put_str(&mut b, "main_01234567");
  put_str(&mut b, "@compute @workgroup_size(64, 1, 1) fn main_01234567() {}");
  put_u32(&mut b, 3);
  b.extend_from_slice(&[1, 2, 0]);
  put_u32(&mut b, 4);
  b.extend_from_slice(&[0, 1, 2, 3]);
  b.extend_from_slice(&1024u64.to_le_bytes());
  let checksum = fnv1a(&b);
  b.extend_from_slice(&checksum.to_le_bytes());
  b
}

#[test]
fn blob_round_trip_and_damage() {
  let good = blob();
  let mut flipped = good.clone();
  let mid = flipped.len() / 2;
  flipped[mid] ^= 1;
  assert!(wgsl_cache::import(&flipped).is_err(), "a flipped bit is refused");
  assert!(wgsl_cache::import(&good[..good.len() - 1]).is_err(), "a truncated blob is refused");
  assert!(wgsl_cache::import(b"").is_err());
  assert_eq!(wgsl_cache::stats().0, 0, "nothing of a refused blob is kept");

  assert_eq!(wgsl_cache::import(&good), Ok(1));
  assert_eq!(wgsl_cache::stats(), (1, 0, 0, 0));
  assert_eq!(wgsl_cache::export(), good, "export writes back what import read");
  assert_eq!(wgsl_cache::import(&good), Ok(1), "importing again keeps one entry");
  assert_eq!(wgsl_cache::stats().0, 1);
}
