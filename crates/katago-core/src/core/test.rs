// Ported from KataGo cpp/core/test.h @ fd0723fdbc0e9d82cf269c9630af8c27c57c07c4
//
// cpp/core/test.cpp only includes test.h ("ensure it compiles on its own"): nothing to port.

use crate::core::global::Global;

/// upstream: testAssert (test.h)
/// A version of assert that's always defined, regardless of NDEBUG
/// PORT NOTE: always on (PORTING.md §7). The upstream macro calls
/// TestCommon::testAssertFailed(#EX, __FILE__, __LINE__), which is reproduced here.
#[macro_export]
macro_rules! testAssert {
  ($ex:expr) => {
    if !($ex) {
      $crate::core::test::TestCommon::test_assert_failed(stringify!($ex), file!(), line!() as i32);
    }
  };
}

/// upstream: namespace TestCommon (test.h)
pub struct TestCommon;

impl TestCommon {
  /// upstream: TestCommon::testAssertFailed (test.h)
  pub fn test_assert_failed(msg: &str, file: &str, line: i32) -> ! {
    Global::fatal_error(
      &(String::from("Failed test assert: ")
        + msg
        + "\n"
        + "file: "
        + file
        + "\n"
        + "line: "
        + &Global::int_to_string(line)),
    );
  }

  /// upstream: TestCommon::expect (test.h)
  /// PORT NOTE: the `(const char*, std::ostringstream&, const std::string&)` overload, which
  /// also clears the stream, becomes `expect_string` (overload rules 3+4: same params and
  /// names, declared later, so it appends the Rust type of `actual`, `&mut String`).
  /// Upstream prints the diff to cout and `exit(1)`s; the port puts the same text in a panic.
  pub fn expect(name: &str, actual: &str, expected: &str) {
    let a = Global::trim(actual, Global::TRIM_DEFAULT_DELIM);
    let e = Global::trim(expected, Global::TRIM_DEFAULT_DELIM);
    let alines = Global::split_delim(&a, '\n');
    let elines = Global::split_delim(&e, '\n');

    let mut matches = true;
    let mut first_line_diff: i32 = 0;
    let n = std::cmp::max(alines.len(), elines.len());
    let mut i: usize = 0;
    while i < n {
      if i >= alines.len() || i >= elines.len() {
        first_line_diff = i as i32;
        matches = false;
        break;
      }
      if Global::trim(&alines[i], Global::TRIM_DEFAULT_DELIM) != Global::trim(&elines[i], Global::TRIM_DEFAULT_DELIM) {
        first_line_diff = i as i32;
        matches = false;
        break;
      }
      i += 1;
    }

    if !matches {
      let mut msg = String::new();
      msg += "Expect test failure!\n";
      msg += name;
      msg += "\n";
      msg += "Expected===============================================================\n";
      msg += expected;
      msg += "\n";
      msg += "Got====================================================================\n";
      msg += actual;
      msg += "\n";

      msg += "=======================================================================\n";
      msg += &format!("First line different (0-indexed) = {}\n", first_line_diff);
      let actual_line = if first_line_diff as usize >= alines.len() { String::new() } else { alines[first_line_diff as usize].clone() };
      let expected_line = if first_line_diff as usize >= elines.len() { String::new() } else { elines[first_line_diff as usize].clone() };
      msg += &format!("Actual  : {}\n", actual_line);
      msg += &format!("Expected: {}\n", expected_line);
      let (ab, eb) = (actual_line.as_bytes(), expected_line.as_bytes());
      let mut j: usize = 0;
      while j < ab.len() && j < eb.len() {
        if ab[j] != eb[j] {
          break;
        }
        j += 1;
      }
      msg += &format!("Char {} differs\n", j);

      panic!("{}", msg);
    }
  }

  /// upstream: TestCommon::expect (test.h)
  /// The `std::ostringstream&` overload: compares, then clears `actual`.
  pub fn expect_string(name: &str, actual: &mut String, expected: &str) {
    TestCommon::expect(name, actual, expected);
    actual.clear();
  }
}
