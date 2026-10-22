//! C printf / `std::ostream` number formatting with glibc's output (docs/PORTING.md §9).
//! Not a port of any upstream file.
//!
//! `printf`/`printf_bytes` implement `%d %i %u %ld %lld %zu %x %X %o %f %F %e %E %g %G %s %c %%` with flags
//! `- + 0 space #`, width and precision (including `*`). Floating point digits come from Rust's
//! exact formatter (`{:.N}` / `{:.Ne}`), which prints the exact binary value rounded
//! half-to-even, like glibc in the default rounding mode. `tests/port_fmt.rs` checks a table
//! generated from glibc.

/// One C vararg. Integer variants keep the C type's width and signedness, because `%u`/`%x` of a
/// negative `int` print it modulo 2^32 while the same value as `int64_t` prints modulo 2^64.
#[derive(Clone, Copy, Debug)]
pub enum PrintfArg<'a> {
  I32(i32),
  I64(i64),
  U32(u32),
  U64(u64),
  /// `float` varargs are promoted to `double`.
  F64(f64),
  Str(&'a str),
  /// A C `char` (byte).
  Char(u8),
}

macro_rules! impl_from_arg {
  ($t:ty, $v:ident, $conv:ty) => {
    impl<'a> From<$t> for PrintfArg<'a> {
      fn from(x: $t) -> Self {
        PrintfArg::$v(x as $conv)
      }
    }
  };
}
// C integer promotion: char/short promote to int.
impl_from_arg!(i8, I32, i32);
impl_from_arg!(i16, I32, i32);
impl_from_arg!(i32, I32, i32);
impl_from_arg!(u16, I32, i32);
impl_from_arg!(i64, I64, i64);
impl_from_arg!(u32, U32, u32);
impl_from_arg!(u64, U64, u64);
impl_from_arg!(usize, U64, u64);
impl_from_arg!(f32, F64, f64);
impl_from_arg!(f64, F64, f64);

impl<'a> From<&'a str> for PrintfArg<'a> {
  fn from(x: &'a str) -> Self {
    PrintfArg::Str(x)
  }
}
impl<'a> From<&'a String> for PrintfArg<'a> {
  fn from(x: &'a String) -> Self {
    PrintfArg::Str(x.as_str())
  }
}
impl<'a> From<char> for PrintfArg<'a> {
  fn from(x: char) -> Self {
    PrintfArg::Char(x as u32 as u8)
  }
}
impl<'a> From<u8> for PrintfArg<'a> {
  fn from(x: u8) -> Self {
    PrintfArg::Char(x)
  }
}
impl<'a> From<bool> for PrintfArg<'a> {
  fn from(x: bool) -> Self {
    PrintfArg::I32(x as i32)
  }
}

impl<'a> PrintfArg<'a> {
  fn as_signed(&self) -> i64 {
    match *self {
      PrintfArg::I32(x) => x as i64,
      PrintfArg::I64(x) => x,
      PrintfArg::U32(x) => x as i32 as i64,
      PrintfArg::U64(x) => x as i64,
      PrintfArg::Char(x) => x as i8 as i64,
      PrintfArg::F64(_) | PrintfArg::Str(_) => panic!("printf: integer conversion given a non-integer argument"),
    }
  }
  fn as_unsigned(&self) -> u64 {
    match *self {
      PrintfArg::I32(x) => x as u32 as u64,
      PrintfArg::I64(x) => x as u64,
      PrintfArg::U32(x) => x as u64,
      PrintfArg::U64(x) => x,
      PrintfArg::Char(x) => x as i8 as i32 as u32 as u64,
      PrintfArg::F64(_) | PrintfArg::Str(_) => panic!("printf: integer conversion given a non-integer argument"),
    }
  }
  fn as_f64(&self) -> f64 {
    match *self {
      PrintfArg::F64(x) => x,
      _ => panic!("printf: floating point conversion given a non-double argument"),
    }
  }
}

#[derive(Default, Clone, Copy)]
struct Spec {
  left: bool,
  plus: bool,
  space: bool,
  zero: bool,
  alt: bool,
  width: usize,
  precision: Option<usize>,
}

fn pad(out: &mut Vec<u8>, spec: &Spec, sign: &str, prefix: &str, body: &[u8], zero_pad_ok: bool) {
  let len = sign.len() + prefix.len() + body.len();
  let fill = spec.width.saturating_sub(len);
  if spec.left {
    out.extend_from_slice(sign.as_bytes());
    out.extend_from_slice(prefix.as_bytes());
    out.extend_from_slice(body);
    for _ in 0..fill {
      out.push(b' ');
    }
  } else if spec.zero && zero_pad_ok {
    out.extend_from_slice(sign.as_bytes());
    out.extend_from_slice(prefix.as_bytes());
    for _ in 0..fill {
      out.push(b'0');
    }
    out.extend_from_slice(body);
  } else {
    for _ in 0..fill {
      out.push(b' ');
    }
    out.extend_from_slice(sign.as_bytes());
    out.extend_from_slice(prefix.as_bytes());
    out.extend_from_slice(body);
  }
}

fn sign_str(negative: bool, spec: &Spec) -> &'static str {
  if negative {
    "-"
  } else if spec.plus {
    "+"
  } else if spec.space {
    " "
  } else {
    ""
  }
}

fn fmt_int_digits(mut digits: String, spec: &Spec) -> String {
  if let Some(p) = spec.precision {
    if p == 0 && digits == "0" {
      digits = String::new();
    }
    while digits.len() < p {
      digits.insert(0, '0');
    }
  }
  digits
}

/// `%.{prec}e` digits of a finite non-negative value: (mantissa "d.ddd", exponent).
fn exp_parts(x: f64, prec: usize) -> (String, i32) {
  let s = format!("{:.*e}", prec, x);
  let (m, e) = s.split_once('e').expect("Rust {:e} output always contains 'e'");
  (m.to_string(), e.parse::<i32>().expect("Rust {:e} exponent is an integer"))
}

fn exp_suffix(e: i32, upper: bool) -> String {
  let mut s = String::new();
  s.push(if upper { 'E' } else { 'e' });
  s.push(if e < 0 { '-' } else { '+' });
  let a = e.unsigned_abs();
  if a < 10 {
    s.push('0');
  }
  s.push_str(&a.to_string());
  s
}

fn strip_trailing_zeros(s: &mut String) {
  if s.contains('.') {
    while s.ends_with('0') {
      s.pop();
    }
    if s.ends_with('.') {
      s.pop();
    }
  }
}

/// Body (no sign) of `%f %e %g` for a finite non-negative value.
fn fmt_float_body(x: f64, conv: u8, prec: usize, alt: bool) -> String {
  let upper = conv.is_ascii_uppercase();
  match conv.to_ascii_lowercase() {
    b'f' => {
      let mut s = format!("{:.*}", prec, x);
      if alt && prec == 0 {
        s.push('.');
      }
      s
    }
    b'e' => {
      let (mut m, e) = exp_parts(x, prec);
      if alt && prec == 0 {
        m.push('.');
      }
      m + &exp_suffix(e, upper)
    }
    b'g' => {
      let p = if prec == 0 { 1 } else { prec };
      let (_, xexp) = exp_parts(x, p - 1);
      // glibc quirk (checked against the table): with '#', when the exact value's exponent
      // picked %f style but rounding carried into a new decade so that %e is needed, glibc keeps
      // no fraction digits: printf("%#g", 999999.5) is "1.e+06", not C99's "1.00000e+06".
      // 25 digits are enough to read the exact exponent: a double is never within 1e-25
      // (relative) below a power of ten without being equal to it.
      if alt && xexp == p as i32 && exp_parts(x, 25).1 == p as i32 - 1 {
        return String::from("1.") + &exp_suffix(xexp, upper);
      }
      if (p as i32) > xexp && xexp >= -4 {
        let mut s = format!("{:.*}", (p as i32 - 1 - xexp) as usize, x);
        if !alt {
          strip_trailing_zeros(&mut s);
        } else if !s.contains('.') {
          s.push('.');
        }
        s
      } else {
        let (mut m, e) = exp_parts(x, p - 1);
        if !alt {
          strip_trailing_zeros(&mut m);
        } else if !m.contains('.') {
          m.push('.');
        }
        m + &exp_suffix(e, upper)
      }
    }
    _ => unreachable!(),
  }
}

fn fmt_float(out: &mut Vec<u8>, x: f64, conv: u8, spec: &Spec) {
  let negative = x.is_sign_negative();
  let sign = sign_str(negative, spec);
  let upper = conv.is_ascii_uppercase();
  if x.is_nan() || x.is_infinite() {
    let body = match (x.is_nan(), upper) {
      (true, false) => "nan",
      (true, true) => "NAN",
      (false, false) => "inf",
      (false, true) => "INF",
    };
    pad(out, spec, sign, "", body.as_bytes(), false);
    return;
  }
  let prec = spec.precision.unwrap_or(6);
  let body = fmt_float_body(x.abs(), conv, prec, spec.alt);
  pad(out, spec, sign, "", body.as_bytes(), true);
}

/// C `snprintf` into a `String`, glibc output. Panics on a malformed format or an argument
/// whose type does not match its conversion (UB in C).
/// TODO(port): C writes bytes. `%c` of a byte >= 0x80, or `%.Ns` cutting a multi-byte UTF-8
/// character, produce bytes that are not UTF-8; this `String` wrapper replaces such a sequence
/// with U+FFFD. Callers that need the exact bytes use [`printf_bytes`].
pub fn printf(fmt: &str, args: &[PrintfArg]) -> String {
  match String::from_utf8(printf_bytes(fmt, args)) {
    Ok(s) => s,
    Err(e) => String::from_utf8_lossy(e.as_bytes()).into_owned(),
  }
}

/// C `snprintf` with byte semantics: exactly the bytes glibc writes. `%c` writes one raw byte
/// (`(unsigned char)arg`), `%.Ns` writes at most N bytes even inside a UTF-8 character, and `%s`
/// stops at the first NUL like C reading a `char*`.
pub fn printf_bytes(fmt: &str, args: &[PrintfArg]) -> Vec<u8> {
  let f = fmt.as_bytes();
  let mut out: Vec<u8> = Vec::new();
  let mut argi = 0usize;
  let next_arg = |argi: &mut usize| -> PrintfArg {
    let a = args.get(*argi).copied().expect("printf: too few arguments");
    *argi += 1;
    a
  };
  let mut i = 0usize;
  let mut lit_start = 0usize;
  while i < f.len() {
    if f[i] != b'%' {
      i += 1;
      continue;
    }
    out.extend_from_slice(&f[lit_start..i]);
    i += 1;
    let mut spec = Spec::default();
    // flags
    while i < f.len() {
      match f[i] {
        b'-' => spec.left = true,
        b'+' => spec.plus = true,
        b' ' => spec.space = true,
        b'0' => spec.zero = true,
        b'#' => spec.alt = true,
        _ => break,
      }
      i += 1;
    }
    // width
    if i < f.len() && f[i] == b'*' {
      let w = next_arg(&mut argi).as_signed();
      if w < 0 {
        spec.left = true;
        spec.width = (-w) as usize;
      } else {
        spec.width = w as usize;
      }
      i += 1;
    } else {
      while i < f.len() && f[i].is_ascii_digit() {
        spec.width = spec.width * 10 + (f[i] - b'0') as usize;
        i += 1;
      }
    }
    // precision
    if i < f.len() && f[i] == b'.' {
      i += 1;
      if i < f.len() && f[i] == b'*' {
        let p = next_arg(&mut argi).as_signed();
        spec.precision = if p < 0 { None } else { Some(p as usize) };
        i += 1;
      } else {
        let mut p = 0usize;
        while i < f.len() && f[i].is_ascii_digit() {
          p = p * 10 + (f[i] - b'0') as usize;
          i += 1;
        }
        spec.precision = Some(p);
      }
    }
    // length modifiers: the argument's own Rust type already fixes the width.
    while i < f.len() && matches!(f[i], b'h' | b'l' | b'L' | b'q' | b'j' | b'z' | b't') {
      i += 1;
    }
    let conv = *f.get(i).expect("printf: format ends inside a conversion");
    i += 1;
    match conv {
      b'%' => out.push(b'%'),
      b'd' | b'i' => {
        let v = next_arg(&mut argi).as_signed();
        let digits = fmt_int_digits(v.unsigned_abs().to_string(), &spec);
        pad(&mut out, &spec, sign_str(v < 0, &spec), "", digits.as_bytes(), spec.precision.is_none());
      }
      b'u' => {
        let v = next_arg(&mut argi).as_unsigned();
        let digits = fmt_int_digits(v.to_string(), &spec);
        pad(&mut out, &spec, "", "", digits.as_bytes(), spec.precision.is_none());
      }
      b'x' | b'X' | b'o' => {
        let v = next_arg(&mut argi).as_unsigned();
        let raw = match conv {
          b'x' => format!("{:x}", v),
          b'X' => format!("{:X}", v),
          _ => format!("{:o}", v),
        };
        let digits = fmt_int_digits(raw, &spec);
        // '#': 0x/0X only for a nonzero value; for %o the precision grows just enough to make
        // the first digit 0, which also covers value 0 with precision 0 (`%#.0o` of 0 is "0").
        let prefix = if spec.alt {
          match conv {
            b'x' if v != 0 => "0x",
            b'X' if v != 0 => "0X",
            b'o' if !digits.starts_with('0') => "0",
            _ => "",
          }
        } else {
          ""
        };
        pad(&mut out, &spec, "", prefix, digits.as_bytes(), spec.precision.is_none());
      }
      b'f' | b'F' | b'e' | b'E' | b'g' | b'G' => {
        let v = next_arg(&mut argi).as_f64();
        fmt_float(&mut out, v, conv, &spec);
      }
      b's' => {
        let a = next_arg(&mut argi);
        let s = match a {
          PrintfArg::Str(s) => s,
          _ => panic!("printf: %s given a non-string argument"),
        };
        // C reads a char*: bytes up to the first NUL, and at most `precision` bytes.
        let s: &[u8] = s.as_bytes();
        let s: &[u8] = match s.iter().position(|&b| b == 0) {
          Some(n) => &s[..n],
          None => s,
        };
        let s: &[u8] = match spec.precision {
          Some(p) if p < s.len() => &s[..p],
          _ => s,
        };
        pad(&mut out, &spec, "", "", s, false);
      }
      b'c' => {
        let a = next_arg(&mut argi);
        let c = match a {
          PrintfArg::Char(c) => c,
          other => other.as_signed() as u8,
        };
        // One raw byte, `(unsigned char)arg`.
        pad(&mut out, &spec, "", "", &[c], false);
      }
      // TODO(port): %a/%A/%p/%n are not implemented (no upstream caller in ported code yet).
      other => panic!("printf: unsupported conversion %{}", other as char),
    }
    lit_start = i;
  }
  out.extend_from_slice(&f[lit_start..]);
  out
}

/// `ostream << double` with the default precision 6 and no `fixed`/`scientific`: `%g`.
/// `ostream << float` is the same after widening to double.
pub fn cpp_ostream_f64(x: f64) -> String {
  printf("%g", &[PrintfArg::F64(x)])
}

/// `ss.precision(p); ss << x;`: `%.{p}g`.
pub fn cpp_ostream_f64_prec(x: f64, prec: usize) -> String {
  printf("%.*g", &[PrintfArg::I32(prec as i32), PrintfArg::F64(x)])
}

/// `std::to_string(double)`: `%f`.
pub fn cpp_to_string_f64(x: f64) -> String {
  printf("%f", &[PrintfArg::F64(x)])
}
