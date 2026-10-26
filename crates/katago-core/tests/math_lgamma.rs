//! port::math::{lgamma_r, lgamma, expm1} against glibc (fixtures/upstream-v1.16.5/unit/oracle/
//! math_lgamma.txt, made by math_lgamma.cpp next to it). Regenerates the oracle's 2^20
//! pseudo-random inputs per function, folds the result bits (and lgamma's signgam) per
//! 65536-input chunk and compares the hashes, then checks the special-value lines. Runs natively
//! and on wasm32 (`just test-wasm`), where Rust's libm is not glibc's.

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
  fn rb(&mut self) -> f64 {
    f64::from_bits(self.sm())
  }
}

/// 2^e for -1022 <= e <= 1023, exactly (`p2` in math_lgamma.cpp).
fn p2(e: i32) -> f64 {
  f64::from_bits(((1023 + e) as u64) << 52)
}
fn bits(d: f64) -> u64 {
  if d.is_nan() { 0x7FF8000000000000 } else { d.to_bits() }
}
fn mix(h: u64, v: u64) -> u64 {
  (h ^ v).wrapping_mul(1099511628211)
}
fn hex(s: &str) -> u64 {
  u64::from_str_radix(s, 16).expect("hex in math_lgamma.txt")
}

const CHUNK: i64 = 65536;
const CHUNKS: i64 = 16;

// Mirrors genLgamma in math_lgamma.cpp (one draw per statement, same order).
fn gen_lgamma(r: &mut Sm, i: i64) -> f64 {
  match i % 12 {
    0 => r.rb(),
    1 => r.u() * 400.0 - 200.0,
    2 => r.u() * 24.0 - 12.0,
    3 => {
      let k = (r.sm() % 300) as f64;
      let e = (r.sm() % 60) as i32 + 1;
      let v = r.u();
      -k + (v - 0.5) * p2(-e)
    }
    4 => {
      let k = 1.0 + (r.sm() % 2) as f64;
      let e = (r.sm() % 60) as i32 + 1;
      let v = r.u();
      k + (v - 0.5) * p2(-e)
    }
    5 => {
      let e = r.sm() % 2047;
      let m = r.sm() & 0x000fffffffffffff;
      let s = r.sm() & (1u64 << 63);
      f64::from_bits(s | (e << 52) | m)
    }
    6 => f64::from_bits(r.sm() & 0x800fffffffffffff),
    7 => r.u() * 16.0 - 10.5,
    8 => {
      let x = (r.sm() % 400) as f64 * 0.5 - 100.0;
      let c = r.sm() & 1;
      let v = r.u();
      if c == 0 { x } else { x + (v - 0.5) * 1e-12 }
    }
    9 => (r.u() - 0.5) * 2.0,
    10 => {
      let e = (r.sm() % 64) as i32 + 960;
      let c = r.sm() & 1;
      let v = r.u();
      (if c == 0 { 1.0 } else { -1.0 }) * (1.0 + v) * p2(e)
    }
    _ => -(r.u() * 11.0),
  }
}

// Mirrors genExpm1 in math_lgamma.cpp.
fn gen_expm1(r: &mut Sm, i: i64) -> f64 {
  match i % 8 {
    0 => r.rb(),
    1 => r.u() * 80.0 - 60.0,
    2 => r.u() * 2.0 - 1.0,
    3 => (r.u() - 0.5) * 1e-3,
    4 => r.u() * 1440.0 - 720.0,
    5 => (r.u() - 0.5) * 2.08,
    6 => {
      let e = (r.sm() % 70) as i32;
      let v = r.u();
      (v - 0.5) * p2(-e)
    }
    _ => r.u() * 50.0,
  }
}

fn compare_chunks(label: &'static str, name: &str, actual: Vec<String>) {
  let expected = fixture("oracle", "math_lgamma.txt");
  let prefix = format!("C {} ", name);
  let exp_lines: Vec<&str> = expected.lines().filter(|l| l.starts_with(&prefix)).collect();
  let mut cases = Cases::new(label);
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

wasm_test! {
fn lgamma_chunk_hashes_vs_glibc() {
  let mut r = Sm(11);
  let mut actual: Vec<String> = Vec::new();
  for c in 0..CHUNKS {
    let mut h: u64 = 1469598103934665603;
    for k in 0..CHUNK {
      let x = gen_lgamma(&mut r, c * CHUNK + k);
      let mut sg: i32 = 0;
      let y = math::lgamma_r(x, &mut sg);
      h = mix(h, bits(y));
      h = mix(h, sg as i64 as u64);
    }
    actual.push(format!("C lgamma {} {:016X}", c, h));
  }
  compare_chunks("lgamma_chunk_hashes_vs_glibc (65536 inputs each)", "lgamma", actual);
}
}

wasm_test! {
fn expm1_chunk_hashes_vs_glibc() {
  let mut r = Sm(12);
  let mut actual: Vec<String> = Vec::new();
  for c in 0..CHUNKS {
    let mut h: u64 = 1469598103934665603;
    for k in 0..CHUNK {
      h = mix(h, bits(math::expm1(gen_expm1(&mut r, c * CHUNK + k))));
    }
    actual.push(format!("C expm1 {} {:016X}", c, h));
  }
  compare_chunks("expm1_chunk_hashes_vs_glibc (65536 inputs each)", "expm1", actual);
}
}

wasm_test! {
fn lgamma_expm1_special_values_vs_glibc() {
  let expected = fixture("oracle", "math_lgamma.txt");
  let mut cases = Cases::new("lgamma_expm1_special_values_vs_glibc");
  let mut bad: Vec<String> = Vec::new();
  for line in expected.lines().filter(|l| l.starts_with("S ")) {
    let t: Vec<&str> = line.split(' ').collect();
    let x = f64::from_bits(hex(t[2]));
    let ok = match t[1] {
      "lgamma" => {
        let mut sg: i32 = 0;
        let y = math::lgamma_r(x, &mut sg);
        // NaN inputs are printed canonicalised; glibc and the port both set signgam = 1 for them.
        bits(y) == hex(t[3]) && sg == t[4].parse::<i32>().expect("signgam") && bits(math::lgamma(x)) == hex(t[3])
      }
      "expm1" => bits(math::expm1(x)) == hex(t[3]),
      other => panic!("unknown function {}", other),
    };
    cases.ran();
    if !ok {
      bad.push(line.to_string());
    }
  }
  assert!(bad.is_empty(), "{} special values differ from glibc:\n{}", bad.len(), bad.join("\n"));
}
}

/// Direct comparison with the host libm where it is the oracle: glibc 2.44 on x86-64 with FMA
/// (lgamma is correctly rounded since glibc 2.43; expm1 needs the FMA variant). A run over
/// 1.2*10^9 inputs per function was done outside the test suite (see port/math.rs).
#[cfg(all(target_arch = "x86_64", target_os = "linux", target_env = "gnu"))]
#[test]
fn lgamma_expm1_vs_host_glibc() {
  unsafe extern "C" {
    fn lgamma_r(x: f64, signgamp: *mut i32) -> f64;
    fn expm1(x: f64) -> f64;
  }
  if !std::arch::is_x86_feature_detected!("fma") {
    eprintln!("lgamma_expm1_vs_host_glibc: host CPU has no FMA, glibc uses its non-FMA expm1; skipped");
    return;
  }
  let mut r = Sm(77);
  let n: i64 = 1 << 20;
  let mut bad = 0;
  for i in 0..n {
    let x = gen_lgamma(&mut r, i);
    let (mut s1, mut s2) = (0i32, 0i32);
    // SAFETY: plain libm call with a valid out-pointer.
    let g = unsafe { lgamma_r(x, &mut s1) };
    let m = math::lgamma_r(x, &mut s2);
    bad += (bits(g) != bits(m) || s1 != s2) as i32;
    let y = gen_expm1(&mut r, i);
    // SAFETY: plain libm call.
    let g = unsafe { expm1(y) };
    bad += (bits(g) != bits(math::expm1(y))) as i32;
  }
  assert_eq!(bad, 0, "port::math::{{lgamma_r, expm1}} differ from host glibc");
  eprintln!("lgamma_expm1_vs_host_glibc: {} inputs x 2 functions", n);
}
