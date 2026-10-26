//! port::math::tanhf against glibc (fixtures/upstream-v1.16.5/unit/oracle/math_tanhf.txt, made by
//! math_tanhf.cpp next to it). Regenerates the oracle's 2^20 pseudo-random inputs, folds the result
//! bits per 65536-input chunk and compares the hashes, then checks the special-value lines. Runs
//! natively and on wasm32 (`just test-wasm`), where the std `f32::tanh` is not glibc's.

#[macro_use]
mod common;

use common::Cases;
use common::fixture;
use katago_core::port::math;

struct Sm(u64);
impl Sm {
  fn sm(&mut self) -> u64 {
    self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = self.0;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
  }
  fn u(&mut self) -> f64 {
    (self.sm() >> 11) as f64 * f64::from_bits(0x3ca0000000000000) // 0x1p-53
  }
  fn rbf(&mut self) -> f32 {
    f32::from_bits(self.sm() as u32)
  }
}

fn bitsf(f: f32) -> u64 {
  if f.is_nan() { 0x7FF8000000000000 } else { f.to_bits() as u64 }
}
fn mix(h: u64, v: u64) -> u64 {
  (h ^ v).wrapping_mul(1099511628211)
}
fn hex(s: &str) -> u64 {
  u64::from_str_radix(s, 16).expect("hex in math_tanhf.txt")
}

const CHUNK: i64 = 65536;
const CHUNKS: i64 = 16;

// Mirrors genTanhf in math_tanhf.cpp.
fn gen_tanhf(r: &mut Sm, i: i64) -> f32 {
  match i % 6 {
    0 => r.rbf(),
    1 => (r.u() * 40.0 - 20.0) as f32,
    2 => (r.u() * 4.0 - 2.0) as f32,
    3 => ((r.u() - 0.5) * f64::from_bits(0x3F60000000000000)) as f32, // 0x1p-9
    4 => (8.0 + r.u() * 3.0) as f32,
    _ => (-(r.u() * 0.75)) as f32,
  }
}

wasm_test! {
fn tanhf_chunk_hashes_vs_glibc() {
  let expected = fixture("oracle", "math_tanhf.txt");
  let exp_lines: Vec<&str> = expected.lines().filter(|l| l.starts_with("C ")).collect();
  let mut r = Sm(5);
  let mut actual: Vec<String> = Vec::new();
  for c in 0..CHUNKS {
    let mut h: u64 = 1469598103934665603;
    for k in 0..CHUNK {
      h = mix(h, bitsf(math::tanhf(gen_tanhf(&mut r, c * CHUNK + k))));
    }
    actual.push(format!("C tanhf {} {:016X}", c, h));
  }
  let mut cases = Cases::new("tanhf_chunk_hashes_vs_glibc (65536 inputs each)");
  assert_eq!(actual.len(), exp_lines.len(), "chunk count");
  let mut bad: Vec<String> = Vec::new();
  for (a, e) in actual.iter().zip(exp_lines.iter()) {
    cases.ran();
    if a != e {
      bad.push(format!("rust {} / glibc {}", a, e));
    }
  }
  assert!(bad.is_empty(), "{} of {} chunks differ from glibc:\n{}", bad.len(), exp_lines.len(), bad.join("\n"));
}
}

wasm_test! {
fn tanhf_special_values_vs_glibc() {
  let expected = fixture("oracle", "math_tanhf.txt");
  let mut cases = Cases::new("tanhf_special_values_vs_glibc");
  let mut bad: Vec<String> = Vec::new();
  for line in expected.lines().filter(|l| l.starts_with("S ")) {
    let t: Vec<&str> = line.split(' ').collect();
    assert_eq!(t[1], "tanhf", "{}", line);
    let x = hex(t[2]);
    // NaN inputs are printed canonicalised; any NaN maps to a NaN result.
    let xf = if x == 0x7FF8000000000000 { f32::NAN } else { f32::from_bits(x as u32) };
    let got = bitsf(math::tanhf(xf));
    cases.ran();
    if got != hex(t[3]) {
      bad.push(format!("{} -> rust {:016X}", line, got));
    }
  }
  assert!(bad.is_empty(), "{} special values differ from glibc:\n{}", bad.len(), bad.join("\n"));
}
}

/// Direct comparison with the host libm where it is the oracle (glibc >= 2.41: correctly
/// rounded tanhf). The full 2^32 sweep was run once outside the test suite (see port/math.rs).
#[cfg(all(target_arch = "x86_64", target_os = "linux", target_env = "gnu"))]
#[test]
#[allow(clippy::disallowed_methods)]
fn tanhf_vs_host_glibc() {
  let mut r = Sm(77);
  let n: i64 = 1 << 20;
  let mut bad = 0;
  for i in 0..n {
    let x = gen_tanhf(&mut r, i);
    bad += (bitsf(x.tanh()) != bitsf(math::tanhf(x))) as i32;
  }
  assert_eq!(bad, 0, "port::math::tanhf differs from host glibc");
  eprintln!("tanhf_vs_host_glibc: {} inputs", n);
}

/// Every f32 input: tanhf (fast f64 path + rounding check) is bit-identical to tanhf_reference (the
/// double-double port of glibc's correctly rounded tanhf). ~4 min natively on 12 threads (release); not on wasm32.
#[cfg(not(target_arch = "wasm32"))]
#[test]
#[ignore = "exhaustive (2^32 inputs): cargo nextest run -p katago-core --test math_tanhf --run-ignored only"]
fn tanhf_fast_path_exhaustive() {
  let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4) as u64;
  let chunk = (1u64 << 32).div_ceil(threads);
  let mismatches: u64 = std::thread::scope(|s| {
    let handles: Vec<_> = (0..threads)
      .map(|t| {
        s.spawn(move || {
          let (lo, hi) = (t * chunk, ((t + 1) * chunk).min(1u64 << 32));
          let mut bad = 0u64;
          for bits in lo..hi {
            let x = f32::from_bits(bits as u32);
            let (a, b) = (math::tanhf(x), math::tanhf_reference(x));
            if a.to_bits() != b.to_bits() && !(a.is_nan() && b.is_nan()) {
              if bad < 10 {
                eprintln!("tanhf mismatch at {bits:#010x}: fast {:#010x} reference {:#010x}", a.to_bits(), b.to_bits());
              }
              bad += 1;
            }
          }
          bad
        })
      })
      .collect();
    handles.into_iter().map(|h| h.join().unwrap()).sum()
  });
  eprintln!("tanhf_fast_path_exhaustive: 4294967296 inputs, {mismatches} mismatches");
  assert_eq!(mismatches, 0);
}
