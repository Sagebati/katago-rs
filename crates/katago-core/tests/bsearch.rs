//! upstream: BSearch::runTests (bsearch.cpp), as a #[test] (port-manifest.toml notes).

#![allow(non_upper_case_globals)]

mod common;

use common::Cases;
use katago_core::core::bsearch::BSearch;
use katago_core::testAssert;

#[test]
fn bsearch_run_tests() {
  let mut cases = Cases::new("bsearch::bsearch_run_tests");
  const len: usize = 13;
  let arr: [f64; len] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
  for i in 0..len {
    testAssert!(BSearch::find_first_gt(&arr, i as f64, 0, len) == i + 1);
    testAssert!(BSearch::find_first_gt(&arr, i as f64 + 0.7, 0, len) == i + 1);
    testAssert!(BSearch::find_first_gt(&arr, i as f64 + 0.99, 0, len) == i + 1);
    testAssert!(BSearch::find_first_gt(&arr, i as f64 - 0.01, 0, len) == i);
    cases.ran();
  }
}
