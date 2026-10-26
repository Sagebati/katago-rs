//! Replays upstream cpp/tests/testconfig.cpp @ v1.16.5, Tests::runInlineConfigTests (the part that needs no
//! file IO), against fixtures/upstream-v1.16.5/unit/testconfig/ (the `R"%%(...)%%"` config texts, extracted
//! by tools/extract_raw_blocks.py: blocks 0..27 in source order). The assertions are upstream's own
//! `testAssert`s, in order.
//!
//! Not ported: Tests::runConfigTests (reads cpp/tests/data/configs/*.cfg and drives KataGoCommandLine;
//! the @include behaviour it checks is covered by tests/config_parser.rs against the real C++),
//! Tests::runParseAllConfigsTest (walks ./configs/ on disk) and Tests::runTaskParsingTests
//! (BUILD_DISTRIBUTED only: distributed/client).

#[macro_use]
mod common;

use common::Cases;
use common::fixture;
use katago_core::core::config_parser::ConfigParser;
use katago_core::testAssert;

const F: &str = "testconfig";

/// The i-th extracted raw block of runInlineConfigTests.
fn block(i: usize) -> String {
  fixture(F, &format!("runinlineconfigtests.input{}.txt", i))
}

wasm_test! {
fn run_inline_config_tests() {
  let mut cases = Cases::new("testconfig::run_inline_config_tests");
  {
    let s = block(0);
    let cfg = ConfigParser::new_in(&s, false, true).unwrap();
    let expected = "";
    testAssert!(expected == cfg.get_all_key_vals());
    cases.ran();
  }
  {
    let s = block(1);
    let cfg = ConfigParser::new_in(&s, false, true).unwrap();
    let expected = concat!(
      "_a = quoted", "\n",
      "_b = quoted ", "\n",
      "_c =  quoted ", "\n",
      "_c_ = 43", "\n",
      "_d =  some # symbols \" yay ", "\n",
      "_e = \"\"\\", "\n",
      "a1 = k2", "\n",
      "abc = def", "\n",
      "b1 = c5", "\n",
      "bcd = g", "\n",
      "c-de = g", "\n",
      "d_ = 5", "\n",
      "e = 6", "\n",
      "f = 7", "\n",
      "key = with spaces", "\n",
      "quotes = i'm a value \" with \" quotes!", "\n",
      "test = back\\slashes don't \\escape \\\\here\\", "\n",
      "test2 = back\\slashes don't \\escape \\\\here\\", "\n",
    );
    testAssert!(expected == cfg.get_all_key_vals());
    cases.ran();
  }

  // `isCfgFail`: the constructor throws a StringError (any of the five types)
  let is_cfg_fail = |s: &str| -> bool {
    match ConfigParser::new_in(s, false, true) {
      Ok(_) => false,
      Err(e) => {
        testAssert!(e.is_string_error());
        true
      }
    }
  };

  // (block, whether upstream asserts isCfgFail)
  let fail_blocks: [(usize, bool); 22] = [
    (2, true),
    (3, true),
    (4, true),
    (5, true),
    (6, true),
    (7, true),
    (8, true),
    (9, false),
    (10, false),
    (11, true),
    (12, true),
    (13, true),
    (14, true),
    (15, true),
    (16, true),
    (17, true),
    (18, true),
    (19, true),
    (20, false),
    (21, false),
    (22, true),
    (23, false),
  ];
  for (i, fails) in fail_blocks {
    let s = block(i);
    if fails {
      testAssert!(is_cfg_fail(&s));
    } else {
      testAssert!(!is_cfg_fail(&s));
    }
    cases.ran();
  }
  {
    let s = block(24);
    testAssert!(!is_cfg_fail(&s));
    cases.ran();
  }

  // Test that getInt/getFloat/getDouble without range args accept the full range of values
  {
    let s = block(25);
    let mut cfg = ConfigParser::new_in(&s, false, true).unwrap();

    testAssert!(cfg.get_int("negInt", i32::MIN, i32::MAX).unwrap() == -42);
    testAssert!(cfg.get_int("zeroInt", i32::MIN, i32::MAX).unwrap() == 0);
    testAssert!(cfg.get_int("posInt", i32::MIN, i32::MAX).unwrap() == 100);
    testAssert!(cfg.get_int("bigInt", i32::MIN, i32::MAX).unwrap() == 2000000000);

    testAssert!(cfg.get_int64("negInt64", i64::MIN, i64::MAX).unwrap() == -9000000000000i64);
    testAssert!(cfg.get_int64("zeroInt64", i64::MIN, i64::MAX).unwrap() == 0);

    testAssert!(cfg.get_uint64("posUInt64", 0, u64::MAX).unwrap() == 18000000000000u64);

    testAssert!(cfg.get_float("negFloat", f32::MIN, f32::MAX).unwrap() == -3.5f32);
    testAssert!(cfg.get_float("zeroFloat", f32::MIN, f32::MAX).unwrap() == 0.0f32);
    testAssert!(cfg.get_float("posFloat", f32::MIN, f32::MAX).unwrap() == 1.5e20f32);

    testAssert!(cfg.get_double("negDouble", f64::MIN, f64::MAX).unwrap() == -1.0e100);
    testAssert!(cfg.get_double("zeroDouble", f64::MIN, f64::MAX).unwrap() == 0.0);
    testAssert!(cfg.get_double("posDouble", f64::MIN, f64::MAX).unwrap() == 1.0e100);
    cases.ran();
  }

  // Test that getInt/getFloat/getDouble with range args reject out-of-range values
  {
    let s = block(26);
    let mut cfg = ConfigParser::new_in(&s, false, true).unwrap();

    testAssert!(cfg.get_int("val", 0, 10).unwrap() == 5);
    testAssert!(cfg.get_float("fval", 0.0f32, 5.0f32).unwrap() == 2.5f32);
    testAssert!(cfg.get_double("fval", 0.0, 5.0).unwrap() == 2.5);

    let mut failed = false;
    if let Err(e) = cfg.get_int("val", 10, 20) {
      testAssert!(e.is_io_error());
      failed = true;
    }
    testAssert!(failed);

    failed = false;
    if let Err(e) = cfg.get_float("fval", 5.0f32, 10.0f32) {
      testAssert!(e.is_io_error());
      failed = true;
    }
    testAssert!(failed);

    failed = false;
    if let Err(e) = cfg.get_double("fval", 5.0, 10.0) {
      testAssert!(e.is_io_error());
      failed = true;
    }
    testAssert!(failed);
    cases.ran();
  }

  // Test that getInts/getFloats/getDoubles without range args accept the full range
  {
    let s = block(27);
    let mut cfg = ConfigParser::new_in(&s, false, true).unwrap();

    let ints: Vec<i32> = cfg.get_ints("ints", i32::MIN, i32::MAX).unwrap();
    testAssert!(ints.len() == 3);
    testAssert!(ints[0] == -100);
    testAssert!(ints[1] == 0);
    testAssert!(ints[2] == 100);

    let floats: Vec<f32> = cfg.get_floats("floats", f32::MIN, f32::MAX).unwrap();
    testAssert!(floats.len() == 3);
    testAssert!(floats[0] == -1.5f32);
    testAssert!(floats[1] == 0.0f32);
    testAssert!(floats[2] == 1.5f32);

    let doubles: Vec<f64> = cfg.get_doubles("doubles", f64::MIN, f64::MAX).unwrap();
    testAssert!(doubles.len() == 3);
    testAssert!(doubles[0] == -1.0e50);
    testAssert!(doubles[1] == 0.0);
    testAssert!(doubles[2] == 1.0e50);
    cases.ran();
  }
}
}
