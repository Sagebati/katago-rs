//! Replays upstream `DateTime::runTests` (cpp/core/datetime.cpp, SimpleDate) call for call; the two
//! expected blocks are fixtures/upstream-v1.16.5/unit/datetime/runtests.expected{0,1}.txt.
#![allow(clippy::bool_comparison, clippy::nonminimal_bool, clippy::identity_op, clippy::eq_op, clippy::neg_cmp_op_on_partial_ord)]

mod common;

use common::Cases;
use common::fixture;
use katago_core::core::datetime::SimpleDate;
use katago_core::core::rand::Rand;
use katago_core::core::test::TestCommon;
use katago_core::testAssert;

fn sd(y: i32, m: i32, d: i32) -> SimpleDate {
  SimpleDate::new_y_m_d(y, m, d).expect("valid date")
}
fn ss(s: &str) -> SimpleDate {
  SimpleDate::new_s(s).expect("valid date string")
}

/// upstream: DateTime::runTests (datetime.cpp)
#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn datetime_run_tests() {
  let mut cases = Cases::new("datetime_run_tests");
  testAssert!(sd(1999, 1, 11) == sd(1999, 1, 11));
  testAssert!(sd(1999, 1, 11) != sd(1999, 2, 11));
  testAssert!(sd(1999, 1, 11) != sd(1999, 1, 21));
  testAssert!(sd(1999, 1, 11) != sd(1998, 1, 11));
  testAssert!(!(sd(1999, 1, 11) != sd(1999, 1, 11)));
  testAssert!(!(sd(1999, 1, 11) == sd(1999, 2, 11)));
  testAssert!(!(sd(1999, 1, 11) == sd(1999, 1, 21)));
  testAssert!(!(sd(1999, 1, 11) == sd(1998, 1, 11)));

  testAssert!(sd(1965, 6, 21) > sd(1945, 8, 25));
  testAssert!(sd(1965, 6, 21) > sd(1965, 4, 25));
  testAssert!(sd(1965, 6, 21) > sd(1965, 6, 20));
  testAssert!(sd(1965, 6, 21) >= sd(1945, 8, 25));
  testAssert!(sd(1965, 6, 21) >= sd(1965, 4, 25));
  testAssert!(sd(1965, 6, 21) >= sd(1965, 6, 20));
  testAssert!(sd(1965, 6, 21) >= sd(1965, 6, 21));

  testAssert!(!(sd(1965, 6, 21) <= sd(1945, 8, 25)));
  testAssert!(!(sd(1965, 6, 21) <= sd(1965, 4, 25)));
  testAssert!(!(sd(1965, 6, 21) <= sd(1965, 6, 20)));
  testAssert!(!(sd(1965, 6, 21) < sd(1945, 8, 25)));
  testAssert!(!(sd(1965, 6, 21) < sd(1965, 4, 25)));
  testAssert!(!(sd(1965, 6, 21) < sd(1965, 6, 20)));
  testAssert!(!(sd(1965, 6, 21) < sd(1965, 6, 21)));

  testAssert!(sd(1965, 6, 21) < sd(1985, 4, 20));
  testAssert!(sd(1965, 6, 21) < sd(1965, 8, 20));
  testAssert!(sd(1965, 6, 21) < sd(1965, 6, 22));
  testAssert!(sd(1965, 6, 21) <= sd(1985, 4, 20));
  testAssert!(sd(1965, 6, 21) <= sd(1965, 8, 20));
  testAssert!(sd(1965, 6, 21) <= sd(1965, 6, 22));
  testAssert!(sd(1965, 6, 21) <= sd(1965, 6, 21));

  testAssert!(!(sd(1965, 6, 21) >= sd(1985, 4, 20)));
  testAssert!(!(sd(1965, 6, 21) >= sd(1965, 8, 20)));
  testAssert!(!(sd(1965, 6, 21) >= sd(1965, 6, 22)));
  testAssert!(!(sd(1965, 6, 21) > sd(1985, 4, 20)));
  testAssert!(!(sd(1965, 6, 21) > sd(1965, 8, 20)));
  testAssert!(!(sd(1965, 6, 21) > sd(1965, 6, 22)));
  testAssert!(!(sd(1965, 6, 21) > sd(1965, 6, 21)));

  testAssert!(sd(1234, 5, 7) == ss("1234-05-07"));
  testAssert!(SimpleDate::new() == ss("1970-01-01"));
  testAssert!(sd(475, 1, 31) == ss("0475-01-31"));

  testAssert!(sd(1234, 5, 7).to_string() == "1234-05-07");
  testAssert!(SimpleDate::new().to_string() == "1970-01-01");

  testAssert!(SimpleDate::new() + 365 == ss("1971-01-01"));
  testAssert!(SimpleDate::new() + 365 * 2 == ss("1972-01-01"));
  testAssert!(SimpleDate::new() + 365 * 3 == ss("1972-12-31"));

  testAssert!(ss("1895-01-01") + 365 == ss("1896-01-01"));
  testAssert!(ss("1895-01-01") + 365 * 2 == ss("1896-12-31"));
  testAssert!(ss("1895-01-01") + 365 * 3 == ss("1897-12-31"));

  testAssert!(ss("1899-01-01") + 365 == ss("1900-01-01"));
  testAssert!(ss("1899-01-01") + 365 * 2 == ss("1901-01-01"));
  testAssert!(ss("1899-01-01") + 365 * 3 == ss("1902-01-01"));

  testAssert!(ss("1999-01-01") + 365 == ss("2000-01-01"));
  testAssert!(ss("1999-01-01") + 365 * 2 == ss("2000-12-31"));
  testAssert!(ss("1999-01-01") + 365 * 3 == ss("2001-12-31"));

  let mut out = String::new();
  let mut rand = Rand::new_seed("SimpleDate tests");
  for i in 0..100000 {
    let mut date1 = SimpleDate::new();
    let mut date2: SimpleDate;
    let d1: i32 = rand.next_int_a_b(-10000000, 10000000);
    let mut d2: i32 = rand.next_int_a_b(-20, 20);
    d2 *= if rand.next_bool(0.1) { 100 } else { 1 };
    let mut d3: i32 = rand.next_int_a_b(-20, 20);
    d3 = d3 * d3 * if rand.next_bool(0.5) { -1 } else { 1 };
    d3 *= if rand.next_bool(0.1) { 37 } else { 1 };

    date1 += d1;
    testAssert!(SimpleDate::new() + d1 == date1);
    testAssert!(d1 + SimpleDate::new() == date1);
    testAssert!(SimpleDate::new() - (-d1) == date1);
    testAssert!(date1 - d1 == SimpleDate::new());
    testAssert!(date1.num_days_after(&SimpleDate::new()) == d1);

    date2 = date1;
    date2 += d2;
    testAssert!(date1 + d2 == date2);
    testAssert!(d2 + date1 == date2);
    testAssert!(date1 - (-d2) == date2);
    testAssert!(date2 - d2 == date1);
    testAssert!(date2.num_days_after(&date1) == d2);

    testAssert!((date1 == SimpleDate::new()) == (d1 == 0));
    testAssert!((date1 != SimpleDate::new()) == (d1 != 0));
    testAssert!((date1 > SimpleDate::new()) == (d1 > 0));
    testAssert!((date1 >= SimpleDate::new()) == (d1 >= 0));
    testAssert!((date1 < SimpleDate::new()) == (d1 < 0));
    testAssert!((date1 <= SimpleDate::new()) == (d1 <= 0));

    testAssert!((date2 == date1) == (d2 == 0));
    testAssert!((date2 != date1) == (d2 != 0));
    testAssert!((date2 > date1) == (d2 > 0));
    testAssert!((date2 >= date1) == (d2 >= 0));
    testAssert!((date2 < date1) == (d2 < 0));
    testAssert!((date2 <= date1) == (d2 <= 0));

    let mut date3: SimpleDate = date2;
    date3 += d3;
    testAssert!(date3.num_days_after(&date2) == d3);
    testAssert!((date3 == date2) == (d3 == 0));
    testAssert!((date3 != date2) == (d3 != 0));
    testAssert!((date3 > date2) == (d3 > 0));
    testAssert!((date3 >= date2) == (d3 >= 0));
    testAssert!((date3 < date2) == (d3 < 0));
    testAssert!((date3 <= date2) == (d3 <= 0));

    testAssert!((date1 + d2) + d3 == date3);
    testAssert!((date1 + d3) + d2 == date3);
    testAssert!(date1 + (d2 + d3) == date3);

    testAssert!(((((((SimpleDate::new() + d1) + d2) - d1) + d3) - d2) - d3) == SimpleDate::new());

    if i < 30 {
      out += &(date1.to_string() + " " + &date2.to_string() + " " + &date3.to_string() + "\n");
    }
    cases.ran();
  }

  let expected = fixture("datetime", "runtests.expected0.txt");
  TestCommon::expect_string("Sample Random Dates", &mut out, &expected);

  let mut i = 0;
  while i < 370 {
    if i == 80 {
      i = 340;
    }
    out += &((ss("2011-12-20") + i).to_string() + " " + &(ss("2012-12-20") + i).to_string() + "\n");
    i += 1;
  }

  let expected = fixture("datetime", "runtests.expected1.txt");
  TestCommon::expect_string("Counting forward dates", &mut out, &expected);
}

/// The constructors' error paths (upstream throws StringError with these texts).
#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn datetime_invalid_dates() {
  let mut cases = Cases::new("datetime_invalid_dates");
  for (y, m, d) in [(2001, 2, 29), (2000, 13, 1), (2000, 0, 1), (2000, 4, 31), (2000, 1, 0)] {
    let e = SimpleDate::new_y_m_d(y, m, d).unwrap_err();
    assert!(e.is_string_error());
    assert_eq!(e.to_string(), format!("SimpleDate: Invalid year month day:{},{},{}", y, m, d));
    cases.ran();
  }
  testAssert!(sd(2000, 2, 29).is_during_leap_year());
  testAssert!(!sd(1900, 2, 28).is_during_leap_year());
  for s in ["2001-02-29", "2001-2-28", "2001/02/28", "20010-2-28", "2001-0a-28", "2001-02-28 ", ""] {
    let e = SimpleDate::new_s(s).unwrap_err();
    assert_eq!(e.to_string(), format!("SimpleDate: Unable to parse as ISO8601 date: {}", s));
    cases.ran();
  }
}
