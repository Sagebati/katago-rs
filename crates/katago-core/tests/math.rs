//! port::math against glibc (fixtures/upstream-v1.16.5/unit/oracle/math.txt, made by math.cpp next
//! to it). Regenerates the oracle's 2^20 pseudo-random inputs per function, folds the result bits
//! per 65536-input chunk and compares the hashes, then checks the special-value table line by
//! line. Runs natively and on wasm32 (`just test-wasm`), which is the point: on wasm the std
//! methods are not glibc's.

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
  fn rbf(&mut self) -> f32 {
    f32::from_bits(self.sm() as u32)
  }
}

fn bits(d: f64) -> u64 {
  if d.is_nan() { 0x7FF8000000000000 } else { d.to_bits() }
}
fn bitsf(f: f32) -> u64 {
  if f.is_nan() { 0x7FF8000000000000 } else { f.to_bits() as u64 }
}
fn mix(h: u64, v: u64) -> u64 {
  (h ^ v).wrapping_mul(1099511628211)
}

const CHUNK: i64 = 65536;
const CHUNKS: i64 = 16;

// Mirrors genExp / genLog / genPow / genExpf in math.cpp.
fn gen_exp(r: &mut Sm, i: i64) -> f64 {
  match i % 6 {
    0 => r.rb(),
    1 => r.u() * 1500.0 - 750.0,
    2 => r.u() * 2.0 - 1.0,
    3 => (r.u() - 0.5) * 1e-10,
    4 => 700.0 + r.u() * 20.0,
    _ => -745.0 + r.u() * 40.0,
  }
}
fn gen_log(r: &mut Sm, i: i64) -> f64 {
  match i % 6 {
    0 => r.rb(),
    1 => r.u() * 1e3,
    2 => 0.9 + r.u() * 0.2,
    3 => r.u() * 1e-300,
    4 => r.u(),
    _ => r.u() * 1e-310,
  }
}
fn gen_pow(r: &mut Sm, i: i64) -> (f64, f64) {
  match i % 7 {
    0 => {
      let x = r.rb();
      let y = r.rb();
      (x, y)
    }
    1 => {
      let x = r.u() * 10.0;
      let y = r.u() * 20.0 - 10.0;
      (x, y)
    }
    2 => {
      let x = r.u();
      let y = 1.0 / (0.03 + r.u() * 3.0);
      (x, y)
    }
    3 => {
      let x = -(r.u() * 10.0);
      let y = ((r.u() * 40.0) as i64 - 20) as f64;
      (x, y)
    }
    4 => {
      let x = 1.0 + (r.u() - 0.5) * 1e-6;
      let y = r.u() * 1e9;
      (x, y)
    }
    5 => {
      let x = r.u() * 1e-310;
      let y = r.u() * 2.0;
      (x, y)
    }
    _ => {
      let x = 1.0 + r.u();
      let y = (r.u() * 2.0 - 1.0) * 1500.0;
      (x, y)
    }
  }
}
fn gen_expf(r: &mut Sm, i: i64) -> f32 {
  match i % 4 {
    0 => r.rbf(),
    1 => (r.u() * 220.0 - 115.0) as f32,
    2 => (r.u() * 2.0 - 1.0) as f32,
    _ => (-103.97 + r.u() * 0.7) as f32,
  }
}

fn chunk_hashes(name: &str, seed: u64, f: &mut dyn FnMut(&mut Sm, i64) -> u64) -> Vec<String> {
  let mut r = Sm(seed);
  let mut out = Vec::new();
  for c in 0..CHUNKS {
    let mut h: u64 = 1469598103934665603;
    for k in 0..CHUNK {
      h = mix(h, f(&mut r, c * CHUNK + k));
    }
    out.push(format!("C {} {} {:016X}", name, c, h));
  }
  out
}

fn hex(s: &str) -> u64 {
  u64::from_str_radix(s, 16).expect("hex in math.txt")
}

wasm_test! {
fn math_chunk_hashes_vs_glibc() {
  let expected = fixture("oracle", "math.txt");
  let exp_lines: Vec<&str> = expected.lines().filter(|l| l.starts_with("C ")).collect();
  let mut actual: Vec<String> = Vec::new();
  actual.extend(chunk_hashes("exp", 1, &mut |r, i| bits(math::exp(gen_exp(r, i)))));
  actual.extend(chunk_hashes("ln", 2, &mut |r, i| bits(math::ln(gen_log(r, i)))));
  actual.extend(chunk_hashes("powf", 3, &mut |r, i| {
    let (x, y) = gen_pow(r, i);
    bits(math::powf(x, y))
  }));
  actual.extend(chunk_hashes("expf", 4, &mut |r, i| bitsf(math::expf(gen_expf(r, i)))));
  let mut cases = Cases::new("math_chunk_hashes_vs_glibc (65536 inputs each)");
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
fn math_special_values_vs_glibc() {
  let expected = fixture("oracle", "math.txt");
  let mut cases = Cases::new("math_special_values_vs_glibc");
  let mut bad: Vec<String> = Vec::new();
  for line in expected.lines().filter(|l| l.starts_with("S ")) {
    let t: Vec<&str> = line.split(' ').collect();
    let got: u64 = match t[1] {
      "exp" => bits(math::exp(f64::from_bits(hex(t[2])))),
      "ln" => bits(math::ln(f64::from_bits(hex(t[2])))),
      "expf" => {
        let x = hex(t[2]);
        // NaN inputs are printed canonicalised; any NaN maps to a NaN result.
        let xf = if x == 0x7FF8000000000000 { f32::NAN } else { f32::from_bits(x as u32) };
        bitsf(math::expf(xf))
      }
      "powf" => bits(math::powf(f64::from_bits(hex(t[2])), f64::from_bits(hex(t[3])))),
      other => panic!("unknown function {}", other),
    };
    let want = hex(t[t.len() - 1]);
    cases.ran();
    if got != want {
      bad.push(format!("{} -> rust {:016X}", line, got));
    }
  }
  assert!(bad.is_empty(), "{} special values differ from glibc:\n{}", bad.len(), bad.join("\n"));
}
}

/// Direct comparison with the host libm, only where the host libm *is* the oracle: glibc on
/// x86-64 with FMA (the std methods call glibc's exp/log/pow/expf there).
#[cfg(all(target_arch = "x86_64", target_os = "linux", target_env = "gnu"))]
#[test]
#[allow(clippy::disallowed_methods)]
fn math_vs_host_glibc() {
  if !std::arch::is_x86_feature_detected!("fma") {
    eprintln!("math_vs_host_glibc: host CPU has no FMA, glibc uses its non-FMA variant; skipped");
    return;
  }
  let mut r = Sm(99);
  let n: i64 = 1 << 20;
  let mut bad = 0;
  for i in 0..n {
    let x = gen_exp(&mut r, i);
    bad += (bits(x.exp()) != bits(math::exp(x))) as i32;
    let y = gen_log(&mut r, i);
    bad += (bits(y.ln()) != bits(math::ln(y))) as i32;
    let (a, b) = gen_pow(&mut r, i);
    bad += (bits(a.powf(b)) != bits(math::powf(a, b))) as i32;
    let f = gen_expf(&mut r, i);
    bad += (bitsf(f.exp()) != bitsf(math::expf(f))) as i32;
  }
  assert_eq!(bad, 0, "port::math differs from host glibc");
  eprintln!("math_vs_host_glibc: {} inputs x 4 functions", n);
}

wasm_test! {
/// D13 (b): `atan` is CORE-MATH's correctly rounded atan (+ glibc's misroundings on ScoreValue's
/// table arguments, see tests/math_atan.rs), so on every target it equals glibc wherever glibc is
/// correctly rounded: here everywhere but the 2 of 2048 inputs where glibc 2.44 is 1 ulp off
/// (checked with MPFR). `tan` uses the platform libm: within 1 ulp of glibc (exact natively).
fn atan_tan_within_one_ulp_of_glibc() {
  common::init();
  let table = common::fixture("oracle", "math_ulp.txt");
  let (mut n, mut differ_atan, mut differ_tan) = (0, 0, 0);
  for line in table.lines() {
    let parts: Vec<&str> = line.split_whitespace().collect();
    let x = f64::from_bits(hex(parts[1]));
    let want = f64::from_bits(hex(parts[2]));
    let (got, differ) = match parts[0] {
      "atan" => (katago_core::port::math::atan(x), &mut differ_atan),
      "tan" => (katago_core::port::math::tan(x), &mut differ_tan),
      other => panic!("unknown fn {other}"),
    };
    let ulps = (got.to_bits() as i64 - want.to_bits() as i64).abs();
    assert!(ulps <= 1, "{line}: got {:016X} ({ulps} ulp)", got.to_bits());
    if ulps != 0 {
      *differ += 1;
    }
    n += 1;
  }
  assert_eq!(n, 4096, "math_ulp.txt cases");
  report!("atan/tan: {n} cases, {differ_atan} atan and {differ_tan} tan differ by 1 ulp");
  // glibc misrounds atan at x = BFCD1486C8F6B8A0 and BFF85EB42EEA89F8; the port is correctly rounded there.
  assert_eq!(differ_atan, 2, "atan must equal glibc except where glibc is not correctly rounded");
  #[cfg(not(target_arch = "wasm32"))]
  assert_eq!(differ_tan, 0, "native tan must be glibc itself");
}
}
