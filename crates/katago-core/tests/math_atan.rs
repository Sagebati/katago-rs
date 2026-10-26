//! port::math::{cr_atan, atan} (DECISIONS.md D13, option b) against fixtures/upstream-v1.16.5/
//! unit/oracle/math_atan.txt, made by math_atan.cpp next to it:
//! - `cr_atan` (CORE-MATH port) vs MPFR's correctly rounded atan: regenerates the oracle's 2^20
//!   pseudo-random inputs, folds the result bits per 65536-input chunk and compares the hashes;
//! - `atan` vs glibc 2.44 (`__atan_fma`) on all 50,531 ScoreValue table arguments (hashes per
//!   4096 arguments), and `GLIBC_ATAN_OVERRIDES` vs the list of glibc's misroundings there;
//! - special inputs, line by line.
//!
//! Runs natively and on wasm32 (`just test-wasm`); natively also directly against the host glibc.

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
}

fn bits(d: f64) -> u64 {
  if d.is_nan() { 0x7FF8000000000000 } else { d.to_bits() }
}
fn mix(h: u64, v: u64) -> u64 {
  (h ^ v).wrapping_mul(1099511628211)
}
fn hex(s: &str) -> u64 {
  u64::from_str_radix(s, 16).expect("hex in math_atan.txt")
}

const CHUNK: i64 = 65536;
const CHUNKS: i64 = 16;
const TCHUNK: i32 = 4096;
const TABLE_LO: i32 = -25265;
const TABLE_HI: i32 = 25265;

// Mirrors genAtan in math_atan.cpp (one draw per statement, same order).
fn gen_atan(r: &mut Sm, i: i64) -> f64 {
  match i % 8 {
    0 => f64::from_bits(r.sm()),
    1 => r.u() * 16.0 - 8.0,
    2 => {
      let e = r.sm() % 120;
      let m = r.sm();
      f64::from_bits((m & 0x800FFFFFFFFFFFFF) | ((e + 1023 - 60) << 52))
    }
    3 => {
      let v = r.sm();
      f64::from_bits((v & 0x800FFFFFFFFFFFFF) | (((v % 51) + 1023 - 30) << 52))
    }
    4 => {
      let k = ((r.sm() % 200001) as i64) as f64 - 100000.0;
      let s = r.u();
      (k / 10.0) / ((s * 2.0 + 0.1) * 19.0)
    }
    5 => {
      let v = r.sm();
      f64::from_bits((v & 0x800FFFFFFFFFFFFF) | (((v % 21) + 1023 - 28) << 52))
    }
    6 => {
      let v = r.sm();
      f64::from_bits((v & 0x800FFFFFFFFFFFFF) | (((v % 50) + 1023 + 6) << 52))
    }
    _ => r.u() * 2.0 - 1.0,
  }
}

/// ScoreValue::initTables' atan argument for step i: what
/// `white_score_value_of_score_smooth_no_draw_adjust(i / 10.0, 0.0, 1.0, 19.0)` passes to atan.
fn table_arg(i: i32) -> f64 {
  let mean = i as f64 / 10.0;
  (mean - 0.0) / (1.0 * 19.0)
}

fn lines_with(prefix: &str) -> Vec<String> {
  fixture("oracle", "math_atan.txt").lines().filter(|l| l.starts_with(prefix)).map(|l| l.to_string()).collect()
}

wasm_test! {
fn cr_atan_chunk_hashes_vs_mpfr() {
  let expected = lines_with("C ");
  assert_eq!(expected.len() as i64, CHUNKS, "C lines");
  let mut cases = Cases::new("cr_atan_chunk_hashes_vs_mpfr (65536 inputs each)");
  let mut r = Sm(21);
  let mut bad: Vec<String> = Vec::new();
  for c in 0..CHUNKS {
    let mut h: u64 = 1469598103934665603;
    for k in 0..CHUNK {
      h = mix(h, bits(math::cr_atan(gen_atan(&mut r, c * CHUNK + k))));
    }
    let actual = format!("C {} {:016X}", c, h);
    cases.ran();
    if actual != expected[c as usize] {
      bad.push(format!("rust {} / mpfr {}", actual, expected[c as usize]));
    }
  }
  assert!(bad.is_empty(), "{} of {} chunks differ from MPFR:\n{}", bad.len(), CHUNKS, bad.join("\n"));
}
}

wasm_test! {
/// The point of D13 (b): on every target, `atan` is glibc's atan on every ScoreValue table argument.
fn atan_table_args_vs_glibc() {
  let expected = lines_with("T ");
  let mut cases = Cases::new("atan_table_args_vs_glibc (4096 table arguments each)");
  let mut bad: Vec<String> = Vec::new();
  let mut n = 0;
  let mut c = 0usize;
  let mut lo = TABLE_LO;
  while lo <= TABLE_HI {
    let mut h: u64 = 1469598103934665603;
    for i in lo..(lo + TCHUNK).min(TABLE_HI + 1) {
      h = mix(h, bits(math::atan(table_arg(i))));
      n += 1;
    }
    let actual = format!("T {} {:016X}", c, h);
    cases.ran();
    if expected.get(c) != Some(&actual) {
      bad.push(format!("rust {} / glibc {:?}", actual, expected.get(c)));
    }
    c += 1;
    lo += TCHUNK;
  }
  assert_eq!(n, 50531, "table arguments");
  assert_eq!(c, expected.len(), "T lines");
  assert!(bad.is_empty(), "{} of {} chunks differ from glibc:\n{}", bad.len(), c, bad.join("\n"));
}
}

wasm_test! {
/// `GLIBC_ATAN_OVERRIDES` is exactly the set of table arguments where glibc is not correctly
/// rounded (D lines: x, glibc, MPFR), and `cr_atan` is the MPFR value there.
fn atan_overrides_are_glibc_misroundings() {
  let d: Vec<(u64, u64, u64)> = lines_with("D ")
    .iter()
    .map(|l| {
      let t: Vec<&str> = l.split(' ').collect();
      (hex(t[1]), hex(t[2]), hex(t[3]))
    })
    .collect();
  let ov = &math::GLIBC_ATAN_OVERRIDES;
  assert!(ov.windows(2).all(|w| w[0].0 < w[1].0), "GLIBC_ATAN_OVERRIDES must be sorted by x bits (binary search)");
  let from_fixture: Vec<(u64, u64)> = d.iter().map(|&(x, g, _)| (x, g)).collect();
  assert_eq!(ov.to_vec(), from_fixture, "GLIBC_ATAN_OVERRIDES vs the D lines of math_atan.txt");
  for &(x, g, m) in &d {
    let x = f64::from_bits(x);
    assert_eq!(bits(math::cr_atan(x)), m, "cr_atan({:e}) vs MPFR", x);
    assert_eq!(bits(math::atan(x)), g, "atan({:e}) vs glibc", x);
  }
}
}

wasm_test! {
fn atan_special_values() {
  let ov = &math::GLIBC_ATAN_OVERRIDES;
  let mut cases = Cases::new("atan_special_values");
  let mut bad: Vec<String> = Vec::new();
  for line in lines_with("S ") {
    let t: Vec<&str> = line.split(' ').collect();
    let xb = hex(t[1]);
    let x = f64::from_bits(xb);
    let (g, m) = (hex(t[2]), hex(t[3]));
    let want = if ov.iter().any(|e| e.0 == xb) { g } else { m };
    cases.ran();
    if bits(math::cr_atan(x)) != m || bits(math::atan(x)) != want {
      bad.push(format!("{} -> cr_atan {:016X}, atan {:016X}", line, bits(math::cr_atan(x)), bits(math::atan(x))));
    }
  }
  assert!(bad.is_empty(), "{} special values differ:\n{}", bad.len(), bad.join("\n"));
}
}

/// Direct comparison with the host glibc 2.44 on x86-64 with FMA (`__atan_fma`): every table
/// argument bit-exact, and 2^20 random inputs within 1 ulp (glibc is not correctly rounded; a
/// 6*10^9-input run of `cr_atan` against the C CORE-MATH code was done outside the test suite,
/// see port/math/atan.rs).
#[cfg(all(target_arch = "x86_64", target_os = "linux", target_env = "gnu"))]
#[test]
fn atan_vs_host_glibc() {
  unsafe extern "C" {
    fn atan(x: f64) -> f64;
  }
  if !std::arch::is_x86_feature_detected!("fma") {
    eprintln!("atan_vs_host_glibc: host CPU has no FMA, glibc uses its non-FMA atan; skipped");
    return;
  }
  let mut bad = 0;
  for i in TABLE_LO..=TABLE_HI {
    let x = table_arg(i);
    // SAFETY: plain libm call.
    let g = unsafe { atan(x) };
    bad += (bits(g) != bits(math::atan(x))) as i32;
  }
  assert_eq!(bad, 0, "port::math::atan differs from host glibc on ScoreValue table arguments");
  let mut r = Sm(5);
  let (mut differ, mut over1) = (0, 0);
  let n: i64 = 1 << 20;
  for i in 0..n {
    let x = gen_atan(&mut r, i);
    // SAFETY: plain libm call.
    let g = unsafe { atan(x) };
    let a = math::atan(x);
    if bits(g) != bits(a) {
      differ += 1;
      over1 += ((g.to_bits() as i64).wrapping_sub(a.to_bits() as i64).abs() > 1) as i32;
    }
  }
  assert_eq!(over1, 0, "port::math::atan more than 1 ulp from host glibc");
  eprintln!("atan_vs_host_glibc: 50531 table arguments exact; {} random inputs, {} differ by 1 ulp (glibc misrounding)", n, differ);
}
