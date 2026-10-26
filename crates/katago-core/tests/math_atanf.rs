//! port::math::atanf against glibc 2.44's correctly rounded atanf (CORE-MATH).
//!
//! - `atanf_sample_hashes_vs_glibc` (native and wasm32): 2^20 pseudo-random inputs plus special values, their
//!   result bits folded per 65536-input chunk, against fixtures/upstream-v1.16.5/unit/oracle/math_atanf.txt.
//! - `atanf_exhaustive_vs_glibc` (native, ignored): every f32 against the platform's atanf, and the ties of
//!   `cr_atan(x) as f32` (port::math::ATANF_TIES) listed.
//! - `regen_math_atanf_fixture` (native, ignored): writes the fixture from the platform's atanf (glibc 2.44).

mod common;

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

fn bitsf(f: f32) -> u64 {
  if f.is_nan() { 0x7FF8000000000000 } else { f.to_bits() as u64 }
}
fn mix(h: u64, v: u64) -> u64 {
  (h ^ v).wrapping_mul(1099511628211)
}

const CHUNK: u64 = 65536;
const CHUNKS: u64 = 16;

/// Inputs: raw bit patterns, the range NNEvaluator feeds (v3 score logits, |x| up to ~50), small and huge values.
fn gen_input(r: &mut Sm, i: u64) -> f32 {
  match i % 5 {
    0 => f32::from_bits(r.sm() as u32),
    1 => (r.u() * 100.0 - 50.0) as f32,
    2 => (r.u() * 4.0 - 2.0) as f32,
    3 => ((r.u() - 0.5) * f64::from_bits(0x3F60000000000000)) as f32, // 0x1p-9
    _ => (r.u() * 1e8) as f32,
  }
}

const SPECIALS: [u32; 12] = [
  0x00000000, 0x80000000, 0x00000001, 0x80000001, 0x3f800000, 0xbf800000, 0x7f7fffff, 0xff7fffff, 0x7f800000,
  0xff800000, 0x7fc00000, 0x39000000,
];

/// The fixture's lines: one `C <chunk> <hash>` per chunk, then `S <input bits> <result bits>` per special value.
fn lines(f: impl Fn(f32) -> f32) -> Vec<String> {
  let mut r = Sm(7);
  let mut out = Vec::new();
  for c in 0..CHUNKS {
    let mut h: u64 = 1469598103934665603;
    for i in 0..CHUNK {
      h = mix(h, bitsf(f(gen_input(&mut r, c * CHUNK + i))));
    }
    out.push(format!("C {c} {h:016x}"));
  }
  for s in SPECIALS {
    out.push(format!("S {s:08x} {:016x}", bitsf(f(f32::from_bits(s)))));
  }
  out
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn atanf_sample_hashes_vs_glibc() {
  let expected = fixture("oracle", "math_atanf.txt");
  let expected: Vec<&str> = expected.lines().filter(|l| l.starts_with("C ") || l.starts_with("S ")).collect();
  let actual = lines(math::atanf);
  assert_eq!(actual.len(), expected.len(), "fixture line count");
  for (a, e) in actual.iter().zip(&expected) {
    assert_eq!(a, e, "atanf differs from glibc");
  }
}

#[cfg(not(target_arch = "wasm32"))]
unsafe extern "C" {
  fn atanf(x: f32) -> f32;
}

/// One thread's share: (mismatch count, the first mismatches (input, port, glibc), the tie inputs with glibc's result).
#[cfg(not(target_arch = "wasm32"))]
type Share = (u64, Vec<(u32, u32, u32)>, Vec<(u32, u32)>);

#[cfg(not(target_arch = "wasm32"))]
fn glibc_atanf(x: f32) -> f32 {
  // SAFETY: libm's atanf is a pure function of its argument.
  unsafe { atanf(x) }
}

/// Every f32 input against the platform atanf (glibc 2.44 here: CORE-MATH, correctly rounded), and the inputs
/// where `cr_atan(x) as f32` is a tie (they must be exactly the entries of ATANF_TIES). Release, ~1-2 min on 12
/// threads.
#[cfg(not(target_arch = "wasm32"))]
#[test]
#[ignore = "exhaustive (2^32 inputs): cargo nextest run --release -p katago-core --test math_atanf --run-ignored only"]
fn atanf_exhaustive_vs_glibc() {
  let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4) as u64;
  let chunk = (1u64 << 32).div_ceil(threads);
  let results: Vec<Share> = std::thread::scope(|s| {
    let handles: Vec<_> = (0..threads)
      .map(|t| {
        s.spawn(move || {
          let (lo, hi) = (t * chunk, ((t + 1) * chunk).min(1u64 << 32));
          let mut bad = 0u64;
          let mut shown = Vec::new();
          let mut ties = Vec::new();
          for bits in lo..hi {
            let x = f32::from_bits(bits as u32);
            let (a, g) = (math::atanf(x), glibc_atanf(x));
            if !x.is_nan() {
              let r = math::cr_atan(x as f64);
              let f = r as f32;
              let (fd, up, down) = (f as f64, f32::from_bits(f.to_bits() + 1) as f64, f32::from_bits(f.to_bits().wrapping_sub(1)) as f64);
              if f.is_finite() && f != 0.0 && (r == (fd + up) * 0.5 || r == (fd + down) * 0.5) {
                ties.push((bits as u32, g.to_bits()));
              }
            }
            if a.to_bits() != g.to_bits() && !(a.is_nan() && g.is_nan()) {
              if shown.len() < 20 {
                shown.push((bits as u32, a.to_bits(), g.to_bits()));
              }
              bad += 1;
            }
          }
          (bad, shown, ties)
        })
      })
      .collect();
    handles.into_iter().map(|h| h.join().unwrap()).collect()
  });
  let mismatches: u64 = results.iter().map(|r| r.0).sum();
  for (_, shown, _) in &results {
    for (x, a, g) in shown {
      eprintln!("atanf mismatch at {x:#010x}: port {a:#010x} glibc {g:#010x}");
    }
  }
  let mut ties: Vec<(u32, u32)> = results.iter().flat_map(|r| r.2.iter().copied()).collect();
  ties.sort_unstable();
  for (x, g) in &ties {
    eprintln!("tie: ({x:#010x}, {g:#010x}),");
  }
  eprintln!("atanf_exhaustive_vs_glibc: 4294967296 inputs, {mismatches} mismatches, {} ties", ties.len());
  assert_eq!(mismatches, 0);
  let table: Vec<(u32, u32)> = math::ATANF_TIES.to_vec();
  assert_eq!(table, ties, "ATANF_TIES must be exactly the tie inputs, with glibc's results");
}

/// Writes fixtures/upstream-v1.16.5/unit/oracle/math_atanf.txt from the platform atanf (run on glibc 2.44).
#[cfg(not(target_arch = "wasm32"))]
#[test]
#[ignore = "regenerates the oracle: cargo nextest run -p katago-core --test math_atanf --run-ignored only -E 'test(regen)'"]
fn regen_math_atanf_fixture() {
  let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../fixtures/upstream-v1.16.5/unit/oracle/math_atanf.txt");
  let mut text = String::from("# glibc 2.44 atanf (CORE-MATH, correctly rounded) on tests/math_atanf.rs's inputs; regen_math_atanf_fixture\n");
  for l in lines(glibc_atanf) {
    text.push_str(&l);
    text.push('\n');
  }
  std::fs::write(path, text).unwrap();
}
