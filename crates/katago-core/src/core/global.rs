// Ported from KataGo cpp/core/global.cpp, cpp/core/global.h @ ba938676d7f42d70950b3a535af2466fb642008c
//
// PARTIAL PORT (pilot): only the Global:: helpers reached from hash/rand/rules/board and their
// tests. See port-manifest.toml notes for the list.

use crate::port::cpp;
use crate::port::fmt::cpp_ostream_f64;
use crate::port::fmt::cpp_ostream_f64_prec;

//GLOBAL DEFINES AND FLAGS----------------------------------------------------

// upstream: ASSERT_UNREACHABLE (global.h)
// PORT NOTE: `ASSERT_UNREACHABLE` -> `unreachable!()` at call sites (PORTING.md §7); no Rust item.

//GLOBAL FUNCTIONS------------------------------------------------------------

/// upstream: StringError, IOError, ConfigParsingError, ValueError, CommandError (global.h)
/// PORT NOTE: the exception hierarchy becomes one enum (PORTING.md §7). `NlohmannJson` stands
/// for `nlohmann::detail::exception`, which is not a StringError.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum KataError {
  StringError(String),
  IOError(String),
  ConfigParsingError(String),
  ValueError(String),
  CommandError(String),
  NlohmannJson(String),
}

impl KataError {
  /// `catch(const StringError&)` matches all five upstream exception types.
  pub fn is_string_error(&self) -> bool {
    !matches!(self, KataError::NlohmannJson(_))
  }
  /// `catch(const IOError&)` matches IOError and its subclass ConfigParsingError.
  pub fn is_io_error(&self) -> bool {
    matches!(self, KataError::IOError(_) | KataError::ConfigParsingError(_))
  }
  /// `e.what()`
  /// PORT NOTE: `what()` returns `message.c_str()`, so a C++ reader stops at the first NUL of a
  /// message built from user input; the full message stays in the variant.
  pub fn what(&self) -> &str {
    let m: &str = match self {
      KataError::StringError(m)
      | KataError::IOError(m)
      | KataError::ConfigParsingError(m)
      | KataError::ValueError(m)
      | KataError::CommandError(m)
      | KataError::NlohmannJson(m) => m,
    };
    match m.find('\0') {
      Some(i) => &m[..i],
      None => m,
    }
  }
}

impl std::fmt::Display for KataError {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.write_str(self.what())
  }
}

impl std::error::Error for KataError {}

/// `Global::strprintf(fmt, ...)`: C printf into a String (PORTING.md §9). Keep the C format
/// string verbatim.
#[macro_export]
macro_rules! strprintf {
  ($fmt:expr $(, $arg:expr)* $(,)?) => {
    $crate::port::fmt::printf($fmt, &[$($crate::port::fmt::PrintfArg::from($arg)),*])
  };
}

/// upstream: namespace Global (global.h)
pub struct Global;

impl Global {
  //ERRORS----------------------------------

  /// upstream: Global::fatalError (global.cpp)
  /// Report fatal error message and exit
  /// PORT NOTE: panics with the upstream text instead of `quick_exit` (PORTING.md §7).
  pub fn fatal_error(s: &str) -> ! {
    panic!("\nFATAL ERROR:\n{}", s);
  }

  //STRINGS---------------------------------

  /// upstream: Global::boolToString (global.cpp)
  pub fn bool_to_string(b: bool) -> String {
    if b { String::from("true") } else { String::from("false") }
  }

  /// upstream: Global::intToString (global.cpp)
  pub fn int_to_string(x: i32) -> String {
    x.to_string()
  }

  /// upstream: Global::floatToString (global.cpp)
  pub fn float_to_string(x: f32) -> String {
    cpp_ostream_f64(x as f64)
  }

  /// upstream: Global::doubleToString (global.cpp)
  pub fn double_to_string(x: f64) -> String {
    cpp_ostream_f64(x)
  }

  /// upstream: Global::doubleToStringHighPrecision (global.cpp)
  pub fn double_to_string_high_precision(x: f64) -> String {
    cpp_ostream_f64_prec(x, 17)
  }

  /// upstream: Global::int64ToString (global.cpp)
  pub fn int64_to_string(x: i64) -> String {
    x.to_string()
  }

  /// upstream: Global::uint32ToString (global.cpp)
  pub fn uint32_to_string(x: u32) -> String {
    x.to_string()
  }

  /// upstream: Global::uint64ToString (global.cpp)
  pub fn uint64_to_string(x: u64) -> String {
    x.to_string()
  }

  /// upstream: Global::uint32ToHexString (global.cpp)
  pub fn uint32_to_hex_string(x: u32) -> String {
    const digits: &[u8; 16] = b"0123456789ABCDEF";
    let hex_len: usize = std::mem::size_of::<u32>() * 2;
    let mut s = vec![b'0'; hex_len];
    for i in 0..hex_len {
      s[hex_len - i - 1] = digits[((x >> (i * 4)) & 0x0f) as usize];
    }
    String::from_utf8(s).expect("hex digits are ASCII")
  }

  /// upstream: Global::uint64ToHexString (global.cpp)
  pub fn uint64_to_hex_string(x: u64) -> String {
    const digits: &[u8; 16] = b"0123456789ABCDEF";
    let hex_len: usize = std::mem::size_of::<u64>() * 2;
    let mut s = vec![b'0'; hex_len];
    for i in 0..hex_len {
      s[hex_len - i - 1] = digits[((x >> (i * 4)) & 0x0f) as usize];
    }
    String::from_utf8(s).expect("hex digits are ASCII")
  }

  /// upstream: Global::sizeToString (global.cpp)
  pub fn size_to_string(x: usize) -> String {
    x.to_string()
  }

  /// upstream: Global::tryStringToInt (global.cpp)
  pub fn try_string_to_int(str: &str, x: &mut i32) -> bool {
    // PORT NOTE: `istringstream in(trim(str)); in >> val;` + fail/peek check -> port::cpp helper.
    match cpp::istream_parse_i32_whole(&Global::trim(str, Global::TRIM_DEFAULT_DELIM)) {
      None => false,
      Some(val) => {
        *x = val;
        true
      }
    }
  }

  /// upstream: Global::stringToInt (global.cpp)
  pub fn string_to_int(str: &str) -> Result<i32, KataError> {
    match cpp::istream_parse_i32_whole(&Global::trim(str, Global::TRIM_DEFAULT_DELIM)) {
      None => Err(KataError::IOError(String::from("could not parse int: ") + str)),
      Some(val) => Ok(val),
    }
  }

  /// upstream: Global::tryStringToInt64 (global.cpp)
  pub fn try_string_to_int64(str: &str, x: &mut i64) -> bool {
    match cpp::istream_parse_i64_whole(&Global::trim(str, Global::TRIM_DEFAULT_DELIM)) {
      None => false,
      Some(val) => {
        *x = val;
        true
      }
    }
  }

  /// upstream: Global::stringToInt64 (global.cpp)
  pub fn string_to_int64(str: &str) -> Result<i64, KataError> {
    match cpp::istream_parse_i64_whole(&Global::trim(str, Global::TRIM_DEFAULT_DELIM)) {
      None => Err(KataError::IOError(String::from("could not parse int: ") + str)),
      Some(val) => Ok(val),
    }
  }

  /// upstream: Global::tryStringToBool (global.cpp)
  pub fn try_string_to_bool(str: &str, x: &mut bool) -> bool {
    let s = Global::to_lower(&Global::trim(str, Global::TRIM_DEFAULT_DELIM));
    if s == "false" {
      *x = false;
      return true;
    }
    if s == "true" {
      *x = true;
      return true;
    }
    false
  }

  /// upstream: Global::stringToBool (global.cpp)
  pub fn string_to_bool(str: &str) -> Result<bool, KataError> {
    let s = Global::to_lower(&Global::trim(str, Global::TRIM_DEFAULT_DELIM));
    if s == "false" {
      return Ok(false);
    }
    if s == "true" {
      return Ok(true);
    }

    Err(KataError::IOError(String::from("could not parse bool: ") + str))
  }

  /// upstream: Global::tryStringToUInt64 (global.cpp)
  pub fn try_string_to_uint64(str: &str, x: &mut u64) -> bool {
    let s = Global::trim(str, Global::TRIM_DEFAULT_DELIM);
    if !s.is_empty() && s.as_bytes()[0] == b'-' {
      return false;
    }
    match cpp::istream_parse_u64_whole(&s) {
      None => false,
      Some(val) => {
        *x = val;
        true
      }
    }
  }

  /// upstream: Global::tryHexStringToUInt64 (global.cpp)
  pub fn try_hex_string_to_uint64(str: &str, x: &mut u64) -> bool {
    for &c in str.as_bytes() {
      if !(c.is_ascii_digit()) && !((b'A'..=b'F').contains(&c)) && !((b'a'..=b'f').contains(&c)) {
        return false;
      }
    }
    // PORT NOTE: `in >> std::hex >> val` + fail/peek check -> port::cpp helper.
    match cpp::istream_parse_hex_u64_whole(str) {
      None => false,
      Some(val) => {
        *x = val;
        true
      }
    }
  }

  /// upstream: Global::stringToUInt64 (global.cpp)
  pub fn string_to_uint64(str: &str) -> Result<u64, KataError> {
    let mut val: u64 = 0;
    let suc = Global::try_string_to_uint64(str, &mut val);
    if !suc {
      return Err(KataError::IOError(String::from("could not parse uint64: ") + str));
    }
    Ok(val)
  }

  /// upstream: Global::hexStringToUInt64 (global.cpp)
  pub fn hex_string_to_uint64(str: &str) -> Result<u64, KataError> {
    let mut val: u64 = 0;
    let suc = Global::try_hex_string_to_uint64(str, &mut val);
    if !suc {
      return Err(KataError::IOError(String::from("could not parse uint64 from hex: ") + str));
    }
    Ok(val)
  }

  /// upstream: Global::tryStringToFloat (global.cpp)
  pub fn try_string_to_float(str: &str, x: &mut f32) -> bool {
    match cpp::istream_parse_f32_whole(&Global::trim(str, Global::TRIM_DEFAULT_DELIM)) {
      None => false,
      Some(val) => {
        *x = val;
        true
      }
    }
  }

  /// upstream: Global::stringToFloat (global.cpp)
  pub fn string_to_float(str: &str) -> Result<f32, KataError> {
    match cpp::istream_parse_f32_whole(&Global::trim(str, Global::TRIM_DEFAULT_DELIM)) {
      None => Err(KataError::IOError(String::from("could not parse float: ") + str)),
      Some(val) => Ok(val),
    }
  }

  /// upstream: Global::tryStringToDouble (global.cpp)
  pub fn try_string_to_double(str: &str, x: &mut f64) -> bool {
    match cpp::istream_parse_f64_whole(&Global::trim(str, Global::TRIM_DEFAULT_DELIM)) {
      None => false,
      Some(val) => {
        *x = val;
        true
      }
    }
  }

  /// upstream: Global::stringToDouble (global.cpp)
  pub fn string_to_double(str: &str) -> Result<f64, KataError> {
    match cpp::istream_parse_f64_whole(&Global::trim(str, Global::TRIM_DEFAULT_DELIM)) {
      None => Err(KataError::IOError(String::from("could not parse double: ") + str)),
      Some(val) => Ok(val),
    }
  }

  /// upstream: Global::isWhitespace (global.cpp)
  /// PORT NOTE: overloads (PORTING.md §1 rule 3): `isWhitespace(char c)` is declared first and
  /// keeps the base name; `isWhitespace(const std::string& s)` appends its param name: `_s`.
  pub fn is_whitespace(c: u8) -> bool {
    contains_str_char(" \t\r\n\x0b\x0c", c)
  }

  /// upstream: Global::isWhitespace (global.cpp)
  pub fn is_whitespace_s(s: &str) -> bool {
    s.bytes().all(|c| b" \t\r\n\x0b\x0c".contains(&c))
  }

  /// upstream: Global::isPrefix (global.cpp)
  pub fn is_prefix(s: &str, prefix: &str) -> bool {
    if s.len() < prefix.len() {
      return false;
    }
    s.as_bytes()[..prefix.len()] == *prefix.as_bytes()
  }

  /// upstream: Global::isSuffix (global.cpp)
  pub fn is_suffix(s: &str, suffix: &str) -> bool {
    if s.len() < suffix.len() {
      return false;
    }
    s.as_bytes()[s.len() - suffix.len()..] == *suffix.as_bytes()
  }

  /// upstream: Global::trim (global.h) default argument `delims`
  /// PORT NOTE: the default `" \t\r\n\v\f"` becomes this const; every caller passes it
  /// explicitly (PORTING.md §5, default-argument exception).
  pub const TRIM_DEFAULT_DELIM: &'static str = " \t\r\n\x0b\x0c";

  /// upstream: Global::trim (global.cpp)
  /// Trim whitespace off both ends of string
  pub fn trim(s: &str, delims: &str) -> String {
    let b = s.as_bytes();
    let d = delims.as_bytes();
    let p2 = b.iter().rposition(|c| !d.contains(c));
    let p2 = match p2 {
      None => return String::new(),
      Some(p) => p,
    };
    let p1 = b.iter().position(|c| !d.contains(c)).unwrap_or(0);

    s[p1..p2 + 1].to_string()
  }

  /// upstream: Global::split (global.cpp)
  /// Split string into tokens, trimming off whitespace
  pub fn split(s: &str) -> Vec<String> {
    // `while(in >> token)`: whitespace-separated tokens.
    let mut tokens = Vec::new();
    for token in s.split([' ', '\t', '\n', '\r', '\x0b', '\x0c']) {
      if token.is_empty() {
        continue;
      }
      let token = Global::trim(token, Global::TRIM_DEFAULT_DELIM);
      tokens.push(token);
    }
    tokens
  }

  /// upstream: Global::split (global.cpp)
  /// Split string based on the given delim, no trimming
  pub fn split_delim(s: &str, delim: char) -> Vec<String> {
    // `while(getline(in,token,delim))`: like str::split, except that getline does not produce
    // a final empty token when the input ends with the delimiter (or is empty).
    let mut tokens: Vec<String> = s.split(delim).map(|t| t.to_string()).collect();
    if tokens.last().is_some_and(|t| t.is_empty()) {
      tokens.pop();
    }
    tokens
  }

  /// upstream: Global::toUpper (global.cpp)
  pub fn to_upper(s: &str) -> String {
    // PORT NOTE: `toupper` in the "C" locale only maps ASCII.
    s.to_ascii_uppercase()
  }

  /// upstream: Global::toLower (global.cpp)
  pub fn to_lower(s: &str) -> String {
    s.to_ascii_lowercase()
  }

  /// upstream: Global::isEqualCaseInsensitive (global.cpp)
  pub fn is_equal_case_insensitive(s0: &str, s1: &str) -> bool {
    let len = s0.len();
    if len != s1.len() {
      return false;
    }
    let (b0, b1) = (s0.as_bytes(), s1.as_bytes());
    for i in 0..len {
      if b0[i].to_ascii_lowercase() != b1[i].to_ascii_lowercase() {
        return false;
      }
    }
    true
  }

  /// upstream: Global::isDigit (global.cpp)
  pub fn is_digit(c: u8) -> bool {
    c >= b'0' && c <= b'9'
  }

  /// upstream: Global::isAlpha (global.cpp)
  pub fn is_alpha(c: u8) -> bool {
    (c >= b'A' && c <= b'Z') || (c >= b'a' && c <= b'z')
  }
}

/// upstream: contains (global.cpp)
/// `contains(const char* str, char c)` = `strchr(str,c) != NULL`.
/// PORT NOTE: strchr also finds the terminating NUL, so `c == 0` is true in C++.
pub fn contains_str_char(str: &str, c: u8) -> bool {
  c == 0 || str.as_bytes().contains(&c)
}
