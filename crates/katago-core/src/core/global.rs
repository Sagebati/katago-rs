// Ported from KataGo cpp/core/global.cpp, cpp/core/global.h @ ba938676d7f42d70950b3a535af2466fb642008c
//
// Not ported (port-manifest.toml notes): getDateString (declared, never defined upstream),
// pauseForKey (console IO).

use std::borrow::Borrow;
use std::collections::BTreeMap;
use std::collections::BTreeSet;

use crate::port::cpp;
use crate::port::fmt::cpp_ostream_f64;
use crate::port::fmt::cpp_ostream_f64_prec;

//GLOBAL DEFINES AND FLAGS----------------------------------------------------

// upstream: PUREFUNC (global.h)
// PORT NOTE: `__attribute__ ((pure))` is an optimisation hint only; no Rust item.

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
/// PORT NOTE: `fmt` is a `const char*`: it ends at its first NUL (port::cpp::c_str).
#[macro_export]
macro_rules! strprintf {
  ($fmt:expr $(, $arg:expr)* $(,)?) => {
    $crate::port::fmt::printf($crate::port::cpp::c_str($fmt), &[$($crate::port::fmt::PrintfArg::from($arg)),*])
  };
}

/// upstream: namespace Global (global.h)
pub struct Global;

impl Global {
  //ERRORS----------------------------------

  /// upstream: Global::fatalError (global.cpp)
  /// Report fatal error message and exit
  /// PORT NOTE: panics with the upstream text instead of `quick_exit` (PORTING.md §7). The
  /// `const char*` and `const std::string&` overloads collapse; `cout << s` of a std::string prints
  /// its NULs, of a `const char*` stops at the first one (port::cpp::c_str at the call site).
  pub fn fatal_error(s: &str) -> ! {
    panic!("\nFATAL ERROR:\n{}", s);
  }

  //TIME------------------------------------

  // upstream: Global::getDateString (global.h)
  //Get string describing the current date, suitable for filenames
  // PORT NOTE: declared in global.h but never defined in global.cpp (DateTime::getDateString in
  // core/datetime.cpp is the live one); nothing to port.

  //STRINGS---------------------------------

  /// upstream: Global::boolToString (global.cpp)
  pub fn bool_to_string(b: bool) -> String {
    if b { String::from("true") } else { String::from("false") }
  }

  /// upstream: Global::charToString (global.cpp)
  pub fn char_to_string(c: u8) -> String {
    // `char buf[2] = {c, 0}; return string(buf);`: a NUL char gives the empty string.
    if c == 0 {
      return String::new();
    }
    // TODO(port): a non-ASCII byte cannot be a one-byte Rust String; U+FFFD here (never hit: callers pass ASCII).
    String::from_utf8_lossy(&[c]).into_owned()
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

  /// upstream: Global::chopPrefix (global.cpp)
  pub fn chop_prefix(s: &str, prefix: &str) -> Result<String, KataError> {
    if !Global::is_prefix(s, prefix) {
      return Err(KataError::StringError(String::from("Global::chopPrefix: \n") + prefix + "\nis not a prefix of\n" + s));
    }
    // `s.substr(prefix.size())`: `prefix` is valid UTF-8 and equals the first bytes of `s`, so the
    // cut is at a char boundary of `s` and this slice cannot panic.
    Ok(s[prefix.len()..].to_string())
  }
  /// upstream: Global::chopSuffix (global.cpp)
  pub fn chop_suffix(s: &str, suffix: &str) -> Result<String, KataError> {
    if !Global::is_suffix(s, suffix) {
      return Err(KataError::StringError(String::from("Global::chopSuffix: \n") + suffix + "\nis not a suffix of\n" + s));
    }
    // `s.substr(0,s.size()-suffix.size())`: `suffix` is valid UTF-8 and equals the last bytes of `s`,
    // so the cut is at a char boundary of `s` and this slice cannot panic.
    Ok(s[..s.len() - suffix.len()].to_string())
  }

  /// upstream: Global::trim (global.h) default argument `delims`
  /// PORT NOTE: the default `" \t\r\n\v\f"` becomes this const; every caller passes it
  /// explicitly (PORTING.md §5, default-argument exception).
  pub const TRIM_DEFAULT_DELIM: &'static str = " \t\r\n\x0b\x0c";

  /// upstream: Global::trim (global.cpp)
  /// Trim whitespace off both ends of string
  /// PORT NOTE: `delims` is a `const char*`: it ends at its first NUL (port::cpp::c_str), while `s` is
  /// a std::string and keeps its NULs. find_last_not_of/find_first_of work on bytes, so a non-ASCII
  /// delimiter can cut a UTF-8 character in half (`trim("é", "©")` is the single byte "\xC3" in C++).
  pub fn trim(s: &str, delims: &str) -> String {
    let delims = cpp::c_str(delims);
    let b = s.as_bytes();
    let d = delims.as_bytes();
    let p2 = b.iter().rposition(|c| !d.contains(c));
    let p2 = match p2 {
      None => return String::new(),
      Some(p) => p,
    };
    let p1 = b.iter().position(|c| !d.contains(c)).unwrap_or(0);

    // TODO(port): std::string is bytes; when the cut leaves invalid UTF-8 the String gets U+FFFD
    // in place of the stray bytes (raw-byte output would need a Vec<u8> return). Never hit with
    // ASCII delimiters, which is every upstream caller.
    String::from_utf8_lossy(&b[p1..p2 + 1]).into_owned()
  }

  /// upstream: Global::concat (global.cpp)
  //Join strings with a delimiter between each one, from [start,end)
  /// PORT NOTE: overloads (PORTING.md §1): `concat(const vector<string>&, delim)` has the fewest
  /// params and is declared before the set one, so it keeps the base name; `(strs, len, delim)`
  /// -> `concat_len`, `(strs, delim, start, end)` -> `concat_start_end`, `(const set<string>&,
  /// delim)` -> `concat_set` (rule 4, type of its first differing param).
  /// PORT NOTE: `strs[i]` and `delim` are `const char*`: each ends at its first NUL (port::cpp::c_str).
  pub fn concat_len(strs: &[&str], len: usize, delim: &str) -> String {
    let delim = cpp::c_str(delim);
    let mut total_len: usize = 0;
    let delim_len: usize = delim.len();
    for i in 0..len {
      if i > 0 {
        total_len += delim_len;
      }
      total_len += cpp::c_str(strs[i]).len();
    }
    let mut s = String::with_capacity(total_len);
    for i in 0..len {
      if i > 0 {
        s += delim;
      }
      s += cpp::c_str(strs[i]);
    }
    s
  }

  /// upstream: Global::concat (global.cpp)
  pub fn concat(strs: &[String], delim: &str) -> String {
    Global::concat_start_end(strs, delim, 0, strs.len())
  }

  /// upstream: Global::concat (global.cpp)
  /// PORT NOTE: `delim` is a `const char*`: it ends at its first NUL (port::cpp::c_str); the
  /// std::string elements keep theirs. `concat` and `concat_set` pass `delim` through to here.
  pub fn concat_start_end(strs: &[String], delim: &str, start: usize, end: usize) -> String {
    let delim = cpp::c_str(delim);
    let mut total_len: usize = 0;
    let delim_len: usize = delim.len();
    for i in start..end {
      if i > start {
        total_len += delim_len;
      }
      total_len += strs[i].len();
    }
    let mut s = String::with_capacity(total_len);
    for i in start..end {
      if i > start {
        s += delim;
      }
      s += &strs[i];
    }
    s
  }

  /// upstream: Global::concat (global.cpp)
  pub fn concat_set(strs: &BTreeSet<String>, delim: &str) -> String {
    let mut v: Vec<String> = Vec::new();
    for x in strs.iter() {
      v.push(x.clone());
    }
    Global::concat_start_end(&v, delim, 0, v.len())
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

  /// upstream: Global::isDigits (global.cpp)
  //Check if a string consists entirely of digits, and parse the integer, checking for overflow
  pub fn is_digits(str: &str) -> bool {
    Global::is_digits_start_end(str, 0, str.len())
  }

  /// upstream: Global::isDigits (global.cpp)
  pub fn is_digits_start_end(str: &str, start: usize, end: usize) -> bool {
    //Too long to fit in integer for sure?
    if end <= start {
      return false;
    }
    if end - start > 9 {
      return false;
    }

    let size: usize = str.len();
    let b = str.as_bytes();
    let mut value: i64 = 0;
    let mut i = start;
    while i < end && i < size {
      let c: u8 = b[i];
      if !Global::is_digit(c) {
        return false;
      }
      value = value * 10 + (c as i64 - b'0' as i64);
      i += 1;
    }

    if (value & 0x7FFFFFFFi64) != value {
      return false;
    }

    true
  }

  /// upstream: Global::parseDigits (global.cpp)
  pub fn parse_digits(str: &str) -> Result<i32, KataError> {
    Global::parse_digits_start_end(str, 0, str.len())
  }

  /// upstream: Global::parseDigits (global.cpp)
  pub fn parse_digits_start_end(str: &str, start: usize, end: usize) -> Result<i32, KataError> {
    //Too long to fit in integer for sure?
    if end <= start {
      return Err(KataError::IOError(String::from("Could not parse digits, end <= start, or empty string")));
    }
    if end - start > 9 {
      return Err(KataError::IOError(String::from("Could not parse digits, overflow: ") + &substr(str, start, end - start)));
    }

    let size: usize = str.len();
    let b = str.as_bytes();
    let mut value: i64 = 0;
    let mut i = start;
    while i < end && i < size {
      let c: u8 = b[i];
      if !Global::is_digit(c) {
        return Ok(0);
      }
      value = value * 10 + (c as i64 - b'0' as i64);
      i += 1;
    }

    if (value & 0x7FFFFFFFi64) != value {
      return Err(KataError::IOError(String::from("Could not parse digits, overflow: ") + &substr(str, start, end - start)));
    }

    Ok(value as i32)
  }

  /// upstream: Global::isDigit (global.cpp)
  pub fn is_digit(c: u8) -> bool {
    c >= b'0' && c <= b'9'
  }

  /// upstream: Global::isAlpha (global.cpp)
  pub fn is_alpha(c: u8) -> bool {
    (c >= b'A' && c <= b'Z') || (c >= b'a' && c <= b'z')
  }

  /// upstream: Global::stringCharsAllAllowed (global.cpp)
  //Check if every char in the string is in the allowed list
  /// PORT NOTE: `allowedChars` is a `const char*`; contains_str_char cuts it at its first NUL, and
  /// (strchr) treats a NUL byte of `str` as allowed.
  pub fn string_chars_all_allowed(str: &str, allowed_chars: &str) -> bool {
    let b = str.as_bytes();
    for i in 0..b.len() {
      if !contains_str_char(allowed_chars, b[i]) {
        return false;
      }
    }
    true
  }

  /// upstream: Global::readKeyValues (global.cpp)
  //Key value pairs are of the form "x=y" or "x = y".
  //Multiple key value pairs are allowed on one line if comma separated.
  //Key value pairs are also broken by newlines.
  pub fn read_key_values(contents: &str) -> Result<BTreeMap<String, String>, KataError> {
    let mut key_values: BTreeMap<String, String> = BTreeMap::new();
    // `while(getline(lineIn,line))`
    for line in getline_split(contents, '\n') {
      if line.is_empty() {
        continue;
      }
      // `while(getline(commaIn,commaChunk,','))`
      for comma_chunk in getline_split(&line, ',') {
        if comma_chunk.is_empty() {
          continue;
        }
        let equals_pos = match comma_chunk.find('=') {
          None => continue,
          Some(p) => p,
        };
        let left_chunk = Global::trim(&comma_chunk[..equals_pos], Global::TRIM_DEFAULT_DELIM);
        let right_chunk = Global::trim(&comma_chunk[equals_pos + 1..], Global::TRIM_DEFAULT_DELIM);
        if left_chunk.is_empty() {
          return Err(KataError::IOError(String::from("readKeyValues: key value pair without key: ") + &line));
        }
        if right_chunk.is_empty() {
          return Err(KataError::IOError(String::from("readKeyValues: key value pair without value: ") + &line));
        }
        if key_values.contains_key(&left_chunk) {
          return Err(KataError::IOError(String::from("readKeyValues: duplicate key: ") + &left_chunk));
        }
        key_values.insert(left_chunk, right_chunk);
      }
    }
    Ok(key_values)
  }

  /// upstream: Global::stripComments (global.cpp)
  //Strips "#" rest-of-line style comments from a string
  pub fn strip_comments(str: &str) -> String {
    if !str.contains('#') {
      return str.to_string();
    }

    //Turn str into a stream so we can go line by line
    let mut result = String::new();

    for line in getline_split(str, '\n') {
      match line.find('#') {
        Some(pos) => result += &line[..pos],
        None => result += &line,
      }
      result += "\n";
    }
    result
  }

  /// upstream: Global::readMem (global.cpp)
  //Read a memory value, like 16G or 256K.
  /// PORT NOTE: the `const char*` and `const std::string&` overloads collapse (PORTING.md §1 rule 1).
  /// This is the std::string one (a NUL is an ordinary byte); the `const char*` one is
  /// `readMem(string(str))`, i.e. `read_mem(port::cpp::c_str(str))`.
  pub fn read_mem(str: &str) -> Result<u64, KataError> {
    if str.len() < 2 {
      return Err(KataError::IOError(String::from("Global::readMem: Could not parse amount of memory: ") + str));
    }

    let end: usize = str.len() - 1;
    let snd: usize = str.len() - 2;

    let numeric_part: String;
    let shift_factor: i32;
    if find_first_of(str, "K") == Some(end) {
      shift_factor = 10;
      numeric_part = str[..end].to_string();
    } else if find_first_of(str, "KB") == Some(snd) {
      shift_factor = 10;
      numeric_part = str[..snd].to_string();
    } else if find_first_of(str, "M") == Some(end) {
      shift_factor = 20;
      numeric_part = str[..end].to_string();
    } else if find_first_of(str, "MB") == Some(snd) {
      shift_factor = 20;
      numeric_part = str[..snd].to_string();
    } else if find_first_of(str, "G") == Some(end) {
      shift_factor = 30;
      numeric_part = str[..end].to_string();
    } else if find_first_of(str, "GB") == Some(snd) {
      shift_factor = 30;
      numeric_part = str[..snd].to_string();
    } else if find_first_of(str, "T") == Some(end) {
      shift_factor = 40;
      numeric_part = str[..end].to_string();
    } else if find_first_of(str, "TB") == Some(snd) {
      shift_factor = 40;
      numeric_part = str[..snd].to_string();
    } else if find_first_of(str, "P") == Some(end) {
      shift_factor = 50;
      numeric_part = str[..end].to_string();
    } else if find_first_of(str, "PB") == Some(snd) {
      shift_factor = 50;
      numeric_part = str[..snd].to_string();
    } else if find_first_of(str, "B") == Some(end) {
      shift_factor = 0;
      numeric_part = str[..end].to_string();
    } else {
      shift_factor = 0;
      numeric_part = str.to_string();
    }

    if !Global::is_digits(&numeric_part) {
      return Err(KataError::IOError(String::from("Global::readMem: Could not parse amount of memory: ") + str));
    }
    // PORT NOTE: `istringstream in(numericPart); in >> mem; if(in.bad())`: numericPart is 1-9
    // digits that fit in 31 bits (isDigits), so the parse cannot fail.
    let mut mem: u64 = match cpp::istream_parse_u64_whole(&numeric_part) {
      Some(v) => v,
      None => return Err(KataError::IOError(String::from("Global::readMem: Could not parse amount of memory: ") + str)),
    };

    for _i in 0..shift_factor {
      let new_mem: u64 = mem << 1;
      if new_mem < mem {
        return Err(KataError::IOError(String::from("Global::readMem: Could not parse amount of memory (too large): ") + str));
      }
      mem = new_mem;
    }
    Ok(mem)
  }

  // upstream: Global::pauseForKey (global.cpp)
  //Display a message and ask the user to press a key to continue
  // PORT NOTE: not ported: reads stdin (no console IO in katago-core, PORTING.md §12).

  /// upstream: Global::roundStatic (global.cpp)
  //Round x to the nearest multiple of 1/inverseScale
  pub fn round_static(x: f64, inverse_scale: f64) -> f64 {
    (x * inverse_scale).round() / inverse_scale
  }

  /// upstream: Global::roundDynamic (global.cpp)
  //Round x to this many decimal digits of precision
  pub fn round_dynamic(x: f64, precision: i32) -> f64 {
    let absx: f64 = x.abs();
    if absx <= 1e-60 {
      return x;
    }
    // PORT NOTE: C++ UB for infinite/NaN x; x86-64 gives INT_MIN (PORTING.md §6 item 5).
    let order_of_magnitude: i32 = crate::port::cpp::static_cast_i32_f64(crate::port::math::log10(absx).floor());
    // PORT NOTE (PORTING.md §6b): with orderOfMagnitude = INT_MIN (x infinite or NaN) the subtraction
    // overflows (C++ UB). GCC folds `orderOfMagnitude - precision >= 0` into `orderOfMagnitude >= precision`
    // (no overflow is assumed), so the test is made on the operands, and the difference and its negation
    // below wrap like x86-64: an infinite x reaches roundStatic(x, pow(10, ...) = 0) and returns NaN
    // (checked against the fixture build by oracle/search_select_diff.cpp, R rows).
    let rounding_magnitude: i32 = order_of_magnitude.wrapping_sub(precision);
    if order_of_magnitude >= precision {
      return x.round();
    }
    let inverse_scale: f64 = crate::port::math::powf(10.0, rounding_magnitude.wrapping_neg() as f64);
    Global::round_static(x, inverse_scale)
  }
}

/// upstream: contains (global.cpp)
/// `contains(const char* str, char c)` = `strchr(str,c) != NULL`; also the
/// `contains(const std::string& str, char c)` overload (`strchr(str.c_str(),c)`, PORTING.md §1 rule 1).
/// PORT NOTE: strchr also finds the terminating NUL, so `c == 0` is true in C++. Both overloads
/// search a C string (`str.c_str()` for the std::string one), which ends at the first NUL
/// (port::cpp::c_str): `contains(string("a\0b",3),'b')` is false.
pub fn contains_str_char(str: &str, c: u8) -> bool {
  c == 0 || cpp::c_str(str).as_bytes().contains(&c)
}

/// `std::string::substr(pos, n)`: clamps `n` to the end of the string.
/// TODO(port): C++ throws std::out_of_range when pos > size(); this clamps to "" instead
/// (only reachable from parseDigits with start > size, never by upstream callers).
fn substr(s: &str, pos: usize, n: usize) -> String {
  let b = s.as_bytes();
  if pos > b.len() {
    return String::new();
  }
  let e = std::cmp::min(b.len(), pos.saturating_add(n));
  String::from_utf8_lossy(&b[pos..e]).into_owned()
}

/// `std::string::find_first_of(chars)` (byte semantics).
fn find_first_of(s: &str, chars: &str) -> Option<usize> {
  s.as_bytes().iter().position(|c| chars.as_bytes().contains(c))
}

/// `while(getline(in, token, delim))` over a string: like `split`, except that no final empty
/// token is produced when the input ends with the delimiter (or is empty).
fn getline_split(s: &str, delim: char) -> Vec<String> {
  Global::split_delim(s, delim)
}

//SHORTCUTS FOR std::map and other containers------------------------------------------------

/// upstream: contains (global.h)
/// PORT NOTE: the template `contains(const std::vector<A>&, const A&)` and
/// `contains(const std::vector<std::string>&, const char*)` collapse into one generic (the
/// const char* overload is the A = std::string case, PORTING.md §1 rule 1); named by rule 4.
/// The generic cannot tell the two apart, so it compares whole `&str`s like the std::string
/// template; a caller porting a `const char*` argument that may hold a NUL passes
/// `port::cpp::c_str(x)`. The same holds for index_of, contains_set and contains_map.
pub fn contains_vector<A: PartialEq<Q>, Q: ?Sized>(vec: &[A], elt: &Q) -> bool {
  for x in vec {
    if x == elt {
      return true;
    }
  }
  false
}

/// upstream: indexOf (global.h)
/// `std::string::npos` when absent.
/// PORT NOTE: the template and the `(const std::vector<std::string>&, const char*)` overload collapse.
pub fn index_of<A: PartialEq<Q>, Q: ?Sized>(vec: &[A], elt: &Q) -> usize {
  let size: usize = vec.len();
  for i in 0..size {
    if vec[i] == *elt {
      return i;
    }
  }
  NPOS
}

/// `std::string::npos`
pub const NPOS: usize = usize::MAX;

/// upstream: contains (global.h)
/// PORT NOTE: `contains(const std::set<A>&, const A&)` and `contains(const std::set<std::string>&, const char*)`.
pub fn contains_set<A: Ord + Borrow<Q>, Q: Ord + ?Sized>(set: &BTreeSet<A>, elt: &Q) -> bool {
  set.contains(elt)
}

/// upstream: contains (global.h)
/// PORT NOTE: `contains(const std::map<A,B>&, const A&)` and `contains(const std::map<std::string,B>&, const char*)`.
pub fn contains_map<A: Ord + Borrow<Q>, Q: Ord + ?Sized, B>(m: &BTreeMap<A, B>, key: &Q) -> bool {
  m.contains_key(key)
}

/// upstream: map_get (global.h)
/// The `template<A,B> map_get(const std::map<A,B>&, const A&)` overload.
pub fn map_get<A: Ord, B: Clone>(m: &BTreeMap<A, B>, key: &A) -> B {
  match m.get(key) {
    None => Global::fatal_error("map_get: key not found"),
    Some(v) => v.clone(),
  }
}

/// upstream: map_get (global.h)
/// PORT NOTE: the `map_get(const std::map<std::string,B>&, const char*)` overload, which has its
/// own message; `_str` because rule 4 gives no readable name for a template specialisation.
/// `key` is a `const char*`: it ends at its first NUL (port::cpp::c_str).
pub fn map_get_str<B: Clone>(m: &BTreeMap<String, B>, key: &str) -> B {
  let key = cpp::c_str(key);
  match m.get(key) {
    None => Global::fatal_error(&(String::from("map_get: key \"") + key + "\" not found")),
    Some(v) => v.clone(),
  }
}

/// upstream: map_get_defaulting (global.h)
pub fn map_get_defaulting<A: Ord, B: Clone>(m: &BTreeMap<A, B>, key: &A, def: &B) -> B {
  match m.get(key) {
    None => def.clone(),
    Some(v) => v.clone(),
  }
}

/// upstream: unique_ptr_void (global.h)
/// PORT NOTE: `std::unique_ptr<void, void(*)(const void*)>` (type-erased owner) -> `Box<dyn Any>`.
pub type unique_ptr_void = Box<dyn std::any::Any>;

/// upstream: make_unique_void (global.h)
pub fn make_unique_void<T: 'static>(ptr: Box<T>) -> unique_ptr_void {
  ptr
}

/// upstream: WrappedWithDeleter (global.h)
/// PORT NOTE: the `DeleterRet (*deleter)(T)` template parameter becomes a field (Rust has no
/// function-pointer const generics). `operator T&` / `operator T` -> `get_mut` / `get`,
/// `operator=(const T&)` -> `assign`. `T val;` default-initialised -> `T: Default`.
pub struct WrappedWithDeleter<T: Copy + Default, DeleterRet> {
  pub assigned: bool,
  pub val: T,
  deleter: fn(T) -> DeleterRet,
}

impl<T: Copy + Default, DeleterRet> WrappedWithDeleter<T, DeleterRet> {
  /// upstream: WrappedWithDeleter::WrappedWithDeleter (global.h)
  pub fn new(deleter: fn(T) -> DeleterRet) -> Self {
    WrappedWithDeleter { assigned: false, val: T::default(), deleter }
  }
  /// upstream: WrappedWithDeleter::WrappedWithDeleter (global.h)
  pub fn new_v(v: T, deleter: fn(T) -> DeleterRet) -> Self {
    WrappedWithDeleter { assigned: true, val: v, deleter }
  }
  /// upstream: WrappedWithDeleter::operator T& (global.h)
  pub fn get_mut(&mut self) -> &mut T {
    &mut self.val
  }
  /// upstream: WrappedWithDeleter::operator T (global.h)
  pub fn get(&self) -> T {
    self.val
  }
  /// upstream: WrappedWithDeleter::operator= (global.h)
  pub fn assign(&mut self, v: T) -> &mut Self {
    self.assigned = true;
    self.val = v;
    self
  }
}

/// upstream: WrappedWithDeleter::~WrappedWithDeleter (global.h)
impl<T: Copy + Default, DeleterRet> Drop for WrappedWithDeleter<T, DeleterRet> {
  fn drop(&mut self) {
    if self.assigned {
      (self.deleter)(self.val);
    }
  }
}

/// upstream: Global::CustomScopeGuard (global.h)
/// PORT NOTE: lives at module level (Rust has no struct inside a unit-struct namespace); copy and
/// move are deleted upstream, so no Clone.
pub struct CustomScopeGuard<F: FnMut()> {
  func: F,
}

impl<F: FnMut()> CustomScopeGuard<F> {
  /// upstream: Global::CustomScopeGuard::CustomScopeGuard (global.h)
  pub fn new(f: F) -> Self {
    CustomScopeGuard { func: f }
  }
}

/// upstream: Global::CustomScopeGuard::~CustomScopeGuard (global.h)
impl<F: FnMut()> Drop for CustomScopeGuard<F> {
  fn drop(&mut self) {
    (self.func)();
  }
}

// upstream: STRUCT_NAMED_SINGLE, STRUCT_NAMED_PAIR, STRUCT_NAMED_TRIPLE, STRUCT_NAMED_QUAD (global.h)
// PORT NOTE: no Rust macro: each use becomes a hand-written struct with pub fields + `new`
// (PORTING.md §3), e.g. `Move` (board.rs), `EncoreKoCapture` (boardhistory.rs).
