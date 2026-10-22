//! search/distributiontable (DistributionTable) against the real C++:
//! fixtures/upstream-v1.16.5/unit/oracle/distributiontable.txt from oracle/distributiontable.cpp (same order;
//! the oracle's header has the build command). Doubles are compared as bits (16 hex digits).
//! The oracle's pdf/cdf use only + - * / sqrt and exp (via port::math::exp, glibc-identical).
//!
//! Not exercised: z with `(int)d` overflowing (|d| >= 2^31, NaN): C++ UB (INT_MIN, then an out-of-bounds
//! table read); the port panics on the index there (TODO(port) in distributiontable.rs).

#![allow(clippy::needless_range_loop, clippy::approx_constant)]

mod common;

use std::fmt::Write as _;

use common::Cases;
use common::compare_lines;
use common::fixture;
use katago_core::core::global::Global;
use katago_core::port::math;
use katago_core::search::distributiontable::DistributionTable;

fn bits(x: f64) -> String {
  format!("{:016X}", x.to_bits())
}

fn rat_pdf(z: f64) -> f64 {
  1.0 / (1.0 + z * z) / 3.14
}
fn rat_cdf(z: f64) -> f64 {
  0.5 + z / (2.0 * (1.0 + (if z < 0.0 { -z } else { z })))
}
fn sqrt_pdf(z: f64) -> f64 {
  0.5 / ((1.0 + z * z) * (1.0 + z * z).sqrt())
}
fn sqrt_cdf(z: f64) -> f64 {
  0.5 * (1.0 + z / (1.0 + z * z).sqrt())
}
fn exp_pdf(z: f64) -> f64 {
  math::exp(-0.5 * z * z) / 2.5066282746310002
}
fn exp_cdf(z: f64) -> f64 {
  0.5 * (1.0 + z / (1.0 + z * z).sqrt()) * math::exp(-0.001 * z * z)
}

fn dump(out: &mut String, name: &str, pdf: &dyn Fn(f64) -> f64, cdf: &dyn Fn(f64) -> f64, min_z: f64, max_z: f64, size: i32) {
  let t = DistributionTable::new(pdf, cdf, min_z, max_z, size);
  writeln!(out, "table {} size {} minZ {} maxZ {}", name, t.size, bits(t.min_z), bits(t.max_z)).unwrap();
  let mut fnv: u64 = 1469598103934665603;
  for i in 0..size as usize {
    for v in [t.pdf_table[i], t.cdf_table[i]] {
      let u = v.to_bits();
      for b in 0..8 {
        fnv ^= (u >> (8 * b)) & 0xFF;
        fnv = fnv.wrapping_mul(1099511628211);
      }
    }
  }
  writeln!(out, "tablehash {}", Global::uint64_to_hex_string(fnv)).unwrap();
  if size <= 200 {
    for i in 0..size as usize {
      writeln!(out, "entry {} {} {}", i, bits(t.pdf_table[i]), bits(t.cdf_table[i])).unwrap();
    }
  } else {
    for i in [0, 1, 2, size / 4, size / 2, size / 2 + 1, size - 3, size - 2, size - 1] {
      writeln!(out, "entry {} {} {}", i, bits(t.pdf_table[i as usize]), bits(t.cdf_table[i as usize])).unwrap();
    }
  }

  let range = max_z - min_z;
  let mut nq = 0;
  let mut query = |out: &mut String, z: f64| {
    let mut p = 0.0;
    let mut c = 0.0;
    t.get_pdf_cdf(z, &mut p, &mut c);
    writeln!(out, "q {} {} {} {} {}", bits(z), bits(p), bits(c), bits(t.get_pdf(z)), bits(t.get_cdf(z))).unwrap();
    nq += 1;
  };
  // a grid reaching well outside [minZ,maxZ]
  const K: i32 = 400;
  let lo = min_z - 0.1 * range;
  let hi = max_z + 0.1 * range;
  for k in 0..=K {
    let z = lo + k as f64 * (hi - lo) / K as f64;
    query(out, z);
  }
  // the table nodes and their neighbourhoods (where idx changes)
  let step = if size <= 101 { 1 } else { 7 };
  let mut i = 0;
  while i < size {
    let z = min_z + i as f64 * (max_z - min_z) / (size - 1) as f64;
    query(out, z);
    query(out, z + 1e-9 * range);
    query(out, z - 1e-9 * range);
    query(out, z + 0.5 * range / (size - 1) as f64);
    i += step;
  }
  // exact boundaries
  query(out, min_z);
  query(out, max_z);
  query(out, 0.0);
  query(out, -0.0);
  query(out, 1e6);
  query(out, -1e300);
  query(out, min_z - 1e-300);
  assert!(nq > 400);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn distributiontable_vs_cpp() {
  let mut cases = Cases::new("distributiontable::distributiontable_vs_cpp");
  let mut out = String::new();
  dump(&mut out, "rational 2000", &rat_pdf, &rat_cdf, -50.0, 50.0, 2000);
  cases.ran();
  dump(&mut out, "rational 101", &rat_pdf, &rat_cdf, -5.0, 5.0, 101);
  cases.ran();
  dump(&mut out, "sqrt 2", &sqrt_pdf, &sqrt_cdf, 0.0, 1.0, 2);
  cases.ran();
  dump(&mut out, "sqrt 3", &sqrt_pdf, &sqrt_cdf, 0.0, 1.0, 3);
  cases.ran();
  dump(&mut out, "sqrt 7", &sqrt_pdf, &sqrt_cdf, -1.0, 1.0, 7);
  cases.ran();
  dump(&mut out, "rational 1000 asym", &rat_pdf, &rat_cdf, -3.5, 12.25, 1000);
  cases.ran();
  dump(&mut out, "exp 2000", &exp_pdf, &exp_cdf, -50.0, 50.0, 2000);
  cases.ran();
  dump(&mut out, "exp 64", &exp_pdf, &exp_cdf, -8.0, 8.0, 64);
  cases.ran();
  compare_lines("distributiontable.txt", &out, &fixture("oracle", "distributiontable.txt"));
}
