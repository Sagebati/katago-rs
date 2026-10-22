//! C++ semantics shims (docs/PORTING.md appendix). Not a port of any upstream file.

use std::cell::Cell;
use std::cmp::Ordering;
use std::ops::Add;

/// `std::min(a,b)`: returns `b` only if `b < a`, so NaN and ties behave like C++.
pub fn cpp_min<T: PartialOrd + Copy>(a: T, b: T) -> T {
  if b < a { b } else { a }
}

/// `std::max(a,b)`: returns `b` only if `a < b`, so NaN and ties behave like C++.
pub fn cpp_max<T: PartialOrd + Copy>(a: T, b: T) -> T {
  if a < b { b } else { a }
}

/// Turns a C++ strict-weak-order "less than" into a Rust `Ordering` comparator.
pub fn cmp_from_lt<T>(lt: impl Fn(&T, &T) -> bool) -> impl Fn(&T, &T) -> Ordering {
  move |a, b| {
    if lt(a, b) {
      Ordering::Less
    } else if lt(b, a) {
      Ordering::Greater
    } else {
      Ordering::Equal
    }
  }
}

/// `atomic.compare_exchange_strong(expected, desired)`: on failure writes the current value
/// into `expected`, like C++. Never fails spuriously.
pub fn cell_cas<T: Copy + PartialEq>(c: &Cell<T>, expected: &mut T, desired: T) -> bool {
  let cur = c.get();
  if cur == *expected {
    c.set(desired);
    true
  } else {
    *expected = cur;
    false
  }
}

/// `atomic.fetch_add(d)`: returns the old value.
pub fn cell_fetch_add<T: Copy + Add<Output = T>>(c: &Cell<T>, d: T) -> T {
  let old = c.get();
  c.set(old + d);
  old
}

// ---------------------------------------------------------------------------------------
// istream >> number (libstdc++ num_get semantics, "C" locale).
// Used by Global::tryStringTo* which do `istringstream in(trim(str)); in >> val;` and then
// require `!in.fail() && in.peek() == EOF`, i.e. the whole string is one number.
// ---------------------------------------------------------------------------------------

/// Length of the longest prefix of `s` that libstdc++'s num_get would accumulate for an
/// integer in base 10: optional sign, then decimal digits.
fn istream_int_prefix(s: &[u8]) -> (usize, bool) {
  let mut i = 0;
  if i < s.len() && (s[i] == b'+' || s[i] == b'-') {
    i += 1;
  }
  let digits_start = i;
  while i < s.len() && s[i].is_ascii_digit() {
    i += 1;
  }
  (i, i > digits_start)
}

/// `istringstream in(s); in >> (int64_t)x;` then `!fail && peek()==EOF`.
/// Leading whitespace is skipped like `operator>>` does. Overflow fails (C++11 sets failbit).
pub fn istream_parse_i64_whole(s: &str) -> Option<i64> {
  let t = s.trim_start_matches([' ', '\t', '\n', '\r', '\x0b', '\x0c']).as_bytes();
  let (len, has_digits) = istream_int_prefix(t);
  if !has_digits || len != t.len() {
    return None;
  }
  std::str::from_utf8(&t[..len]).ok()?.parse::<i64>().ok()
}

/// `istringstream in(s); in >> (int)x;` then `!fail && peek()==EOF`.
pub fn istream_parse_i32_whole(s: &str) -> Option<i32> {
  let t = s.trim_start_matches([' ', '\t', '\n', '\r', '\x0b', '\x0c']).as_bytes();
  let (len, has_digits) = istream_int_prefix(t);
  if !has_digits || len != t.len() {
    return None;
  }
  std::str::from_utf8(&t[..len]).ok()?.parse::<i32>().ok()
}

/// `istringstream in(s); in >> (uint64_t)x;` then `!fail && peek()==EOF`.
/// TODO(port): libstdc++ accepts a leading '-' for unsigned and negates modulo 2^64; upstream's
/// only caller (tryStringToUInt64) rejects '-' before parsing, so this rejects it too.
pub fn istream_parse_u64_whole(s: &str) -> Option<u64> {
  let t = s.trim_start_matches([' ', '\t', '\n', '\r', '\x0b', '\x0c']).as_bytes();
  let (len, has_digits) = istream_int_prefix(t);
  if !has_digits || len != t.len() || t[0] == b'-' {
    return None;
  }
  std::str::from_utf8(&t[..len]).ok()?.parse::<u64>().ok()
}

/// `istringstream in(s); in >> std::hex >> (uint64_t)x;` then `!fail && peek()==EOF`.
/// Upstream's only caller (tryHexStringToUInt64) has already checked every char is a hex digit.
pub fn istream_parse_hex_u64_whole(s: &str) -> Option<u64> {
  let t = s.as_bytes();
  if t.is_empty() || !t.iter().all(|c| c.is_ascii_hexdigit()) {
    return None;
  }
  u64::from_str_radix(s, 16).ok()
}

/// Length of the longest prefix libstdc++'s num_get accumulates for a floating point value:
/// `[+-]? digits* ('.' digits*)? ([eE] [+-]? digits*)?`. Returns (len, has_mantissa_digit).
fn istream_float_prefix(s: &[u8]) -> (usize, bool) {
  let mut i = 0;
  let mut has_digit = false;
  if i < s.len() && (s[i] == b'+' || s[i] == b'-') {
    i += 1;
  }
  while i < s.len() && s[i].is_ascii_digit() {
    i += 1;
    has_digit = true;
  }
  if i < s.len() && s[i] == b'.' {
    i += 1;
    while i < s.len() && s[i].is_ascii_digit() {
      i += 1;
      has_digit = true;
    }
  }
  if has_digit && i < s.len() && (s[i] == b'e' || s[i] == b'E') {
    i += 1;
    if i < s.len() && (s[i] == b'+' || s[i] == b'-') {
      i += 1;
    }
    while i < s.len() && s[i].is_ascii_digit() {
      i += 1;
    }
  }
  (i, has_digit)
}

/// `istringstream in(s); in >> (double)x;` then `!fail && peek()==EOF`.
/// libstdc++ hands the accumulated chars to strtod and fails unless all were consumed, and
/// fails on overflow (±HUGE_VAL). "inf"/"nan" are not accepted by num_get.
pub fn istream_parse_f64_whole(s: &str) -> Option<f64> {
  let t = s.trim_start_matches([' ', '\t', '\n', '\r', '\x0b', '\x0c']).as_bytes();
  let (len, has_digit) = istream_float_prefix(t);
  if !has_digit || len != t.len() {
    return None;
  }
  let v = std::str::from_utf8(&t[..len]).ok()?.parse::<f64>().ok()?;
  if v.is_infinite() {
    return None;
  }
  Some(v)
}

/// `istringstream in(s); in >> (float)x;` then `!fail && peek()==EOF` (strtof, correctly rounded).
pub fn istream_parse_f32_whole(s: &str) -> Option<f32> {
  let t = s.trim_start_matches([' ', '\t', '\n', '\r', '\x0b', '\x0c']).as_bytes();
  let (len, has_digit) = istream_float_prefix(t);
  if !has_digit || len != t.len() {
    return None;
  }
  let v = std::str::from_utf8(&t[..len]).ok()?.parse::<f32>().ok()?;
  if v.is_infinite() {
    return None;
  }
  Some(v)
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn istream_parsing() {
    assert_eq!(istream_parse_i32_whole("+5"), Some(5));
    assert_eq!(istream_parse_i32_whole("-0012"), Some(-12));
    assert_eq!(istream_parse_i32_whole("1e3"), None);
    assert_eq!(istream_parse_i32_whole("-"), None);
    assert_eq!(istream_parse_i32_whole(""), None);
    assert_eq!(istream_parse_i32_whole("99999999999"), None);
    assert_eq!(istream_parse_f32_whole("7.5"), Some(7.5));
    assert_eq!(istream_parse_f32_whole(".5"), Some(0.5));
    assert_eq!(istream_parse_f32_whole("5."), Some(5.0));
    assert_eq!(istream_parse_f32_whole("1.2.3"), None);
    assert_eq!(istream_parse_f32_whole("inf"), None);
    assert_eq!(istream_parse_f32_whole("1e"), None);
    assert_eq!(istream_parse_f64_whole("1e400"), None);
    assert!(cpp_max(f64::NAN, 1.0).is_nan());
    assert_eq!(cpp_max(1.0, f64::NAN), 1.0);
    assert_eq!(cpp_min(1.0, 2.0), 1.0);
  }
}

/// `(int)x` / `static_cast<int>(x)` of a `double` as compiled for x86-64 (`cvttsd2si`).
/// C++ UB for NaN and out-of-range values; x86-64 gives INT_MIN there, while Rust `as`
/// saturates (NaN -> 0). In range it truncates toward zero like `as`.
pub fn static_cast_i32_f64(x: f64) -> i32 {
  if x > -2147483649.0 && x < 2147483648.0 { x as i32 } else { i32::MIN }
}

/// `(int)x` of a `float` as compiled for x86-64 (`cvttss2si`): see static_cast_i32_f64.
pub fn static_cast_i32_f32(x: f32) -> i32 {
  static_cast_i32_f64(x as f64)
}
