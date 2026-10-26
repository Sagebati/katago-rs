//! port::math::log10 against glibc (fixtures/upstream-v1.18.2/unit/oracle/math_log10.txt, made by
//! math_log10.cpp next to it). Regenerates the oracle's 2^20 pseudo-random inputs, folds the result
//! bits per 65536-input chunk and compares the hashes, then checks the special-value lines (every
//! power of ten and its neighbours: what Global::roundDynamic's floor(log10(x)) depends on).
//! Runs natively and on wasm32 (`just test-wasm`).

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

fn bits(d: f64) -> u64 {
  if d.is_nan() { 0x7FF8000000000000 } else { d.to_bits() }
}
fn mix(h: u64, v: u64) -> u64 {
  (h ^ v).wrapping_mul(1099511628211)
}

// Mirrors gen() in math_log10.cpp.
fn gen_input(r: &mut Sm, i: i64) -> f64 {
  match i % 5 {
    0 => r.rb(),
    1 => r.u() * 1e3,
    2 => r.u() * 2.0,
    3 => f64::from_bits(r.sm() >> 12),
    _ => (r.u() - 0.5) * 1e-3 + 1.0,
  }
}

wasm_test! {
fn log10_matches_glibc() {
  let text = fixture("oracle", "math_log10.txt");
  let mut cases = Cases::new("math_log10");
  let mut r = Sm(0x4c4f473130);
  let mut chunk: i64 = 0;
  for line in text.lines() {
    let parts: Vec<&str> = line.split(' ').collect();
    match parts[0] {
      "C" => {
        let c: i64 = parts[1].parse().unwrap();
        assert_eq!(c, chunk);
        let mut h: u64 = 1469598103934665603;
        for j in 0..65536i64 {
          let x = gen_input(&mut r, c * 65536 + j);
          h = mix(h, bits(math::log10(x)));
        }
        assert_eq!(format!("{:016X}", h), parts[2], "log10 chunk {}", c);
        chunk += 1;
        cases.ran();
      }
      "S" => {
        let x = f64::from_bits(u64::from_str_radix(parts[1], 16).unwrap());
        let expected = u64::from_str_radix(parts[2], 16).unwrap();
        assert_eq!(bits(math::log10(x)), expected, "log10({:e}) bits {}", x, parts[1]);
        cases.ran();
      }
      _ => panic!("bad line {}", line),
    }
  }
  assert_eq!(chunk, 16);
}
}
