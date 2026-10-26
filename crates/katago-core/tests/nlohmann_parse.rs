//! port::nlohmann::parse against the vendored nlohmann/json 3.8.0
//! (fixtures/upstream-v1.18.2/unit/oracle/nlohmann_parse.cpp; inputs in nlohmann_parse.input.txt).
//! Per input the oracle prints a `D` line (json::parse: dump() or e.what()) and an `S` line
//! (json::sax_parse event stream, which shows the lexer's unsigned/integer/float classification).

mod common;

use common::Cases;
use common::fixture;
use katago_core::port::nlohmann;

fn unesc(s: &str) -> Vec<u8> {
  let b = s.as_bytes();
  let mut o = Vec::new();
  let mut i = 0;
  while i < b.len() {
    if b[i] == b'\\' && i + 1 < b.len() {
      i += 1;
      match b[i] {
        b'n' => o.push(b'\n'),
        b't' => o.push(b'\t'),
        b'r' => o.push(b'\r'),
        b'0' => o.push(0),
        b'x' => {
          o.push(u8::from_str_radix(&s[i + 1..i + 3], 16).expect("hex escape"));
          i += 2;
        }
        c => o.push(c),
      }
    } else {
      o.push(b[i]);
    }
    i += 1;
  }
  o
}

fn esc(s: &[u8]) -> String {
  let mut o = String::new();
  for &c in s {
    if c == b'\\' {
      o.push_str("\\\\");
    } else if !(0x20..0x7F).contains(&c) {
      o.push_str(&format!("\\x{:02x}", c));
    } else {
      o.push(c as char);
    }
  }
  o
}

/// Records the SAX events like `Rec` in nlohmann_parse.cpp.
struct Rec {
  ev: String,
}

impl Rec {
  fn put(&mut self, s: &str) {
    if !self.ev.is_empty() {
      self.ev.push(' ');
    }
    self.ev.push_str(s);
  }
}

impl nlohmann::Sax for Rec {
  fn null(&mut self) -> bool {
    self.put("n");
    true
  }
  fn boolean(&mut self, val: bool) -> bool {
    self.put(if val { "b1" } else { "b0" });
    true
  }
  fn number_integer(&mut self, val: i64) -> bool {
    self.put(&format!("i{}", val));
    true
  }
  fn number_unsigned(&mut self, val: u64) -> bool {
    self.put(&format!("u{}", val));
    true
  }
  fn number_float(&mut self, val: f64, s: &[u8]) -> bool {
    self.put(&format!("f{:016X}/{}", val.to_bits(), esc(s)));
    true
  }
  fn string(&mut self, val: &mut Vec<u8>) -> bool {
    let e = format!("s{}", esc(val));
    self.put(&e);
    true
  }
  fn start_object(&mut self, _len: usize) -> bool {
    self.put("{");
    true
  }
  fn key(&mut self, val: &mut Vec<u8>) -> bool {
    let e = format!("k{}", esc(val));
    self.put(&e);
    true
  }
  fn end_object(&mut self) -> bool {
    self.put("}");
    true
  }
  fn start_array(&mut self, _len: usize) -> bool {
    self.put("[");
    true
  }
  fn end_array(&mut self) -> bool {
    self.put("]");
    true
  }
  fn parse_error(&mut self, pos: usize, last_token: &[u8], ex: &nlohmann::Exception) -> Result<bool, katago_core::core::global::KataError> {
    let e = format!("E{}|{}|{}", pos, esc(last_token), esc(ex.what()));
    self.put(&e);
    Ok(false)
  }
}

fn inputs() -> Vec<String> {
  let inp = fixture("oracle", "nlohmann_parse.input.txt");
  inp
    .lines()
    .map(|l| String::from_utf8(unesc(l)).expect("corpus inputs are valid UTF-8 (parse takes &str)"))
    .collect()
}

#[test]
fn parse_dom_matches_nlohmann() {
  // dump() and serde_json's Value drop recurse once per nesting level (dump does upstream too);
  // the corpus has a 3000-deep array, which overflows the 2 MiB test thread in debug builds.
  std::thread::Builder::new()
    .stack_size(64 << 20)
    .spawn(parse_dom_matches_nlohmann_body)
    .expect("spawn")
    .join()
    .expect("parse_dom_matches_nlohmann_body");
}

fn parse_dom_matches_nlohmann_body() {
  let mut cases = Cases::new("nlohmann_parse::parse_dom_matches_nlohmann");
  let expected = fixture("oracle", "nlohmann_parse.txt");
  // TODO(port) in nlohmann.rs (Exception::into_kata_error): KataError holds a String, so a what()
  // that ends inside a multi-byte UTF-8 character carries U+FFFD there. Apply the same lossy
  // conversion to the C++ text; the exact bytes are checked by sax_parse_matches_nlohmann.
  let expected_d: Vec<String> = expected
    .lines()
    .filter(|l| l.starts_with("D "))
    .map(|l| esc(String::from_utf8_lossy(&unesc(l)).as_bytes()))
    .collect();
  let ins = inputs();
  assert_eq!(ins.len(), expected_d.len());
  let mut bad = 0;
  for (i, s) in ins.iter().enumerate() {
    let actual = match nlohmann::parse(s) {
      Ok(v) => esc(format!("D OK {}", nlohmann::dump(&v)).as_bytes()),
      Err(e) => esc(format!("D ERR {}", e.what()).as_bytes()),
    };
    if actual != expected_d[i] {
      bad += 1;
      if bad <= 10 {
        eprintln!("input {} {:?}\n  rust: {}\n  cpp:  {}", i + 1, &s[..s.len().min(80)], &actual[..actual.len().min(300)], &expected_d[i][..expected_d[i].len().min(300)]);
      }
    }
    cases.ran();
  }
  assert_eq!(bad, 0, "{} of {} inputs differ from nlohmann", bad, ins.len());
}

#[test]
fn sax_parse_matches_nlohmann() {
  let mut cases = Cases::new("nlohmann_parse::sax_parse_matches_nlohmann");
  let expected = fixture("oracle", "nlohmann_parse.txt");
  let expected_s: Vec<&str> = expected.lines().filter(|l| l.starts_with("S ")).collect();
  let ins = inputs();
  assert_eq!(ins.len(), expected_s.len());
  let mut bad = 0;
  for (i, s) in ins.iter().enumerate() {
    let mut rec = Rec { ev: String::new() };
    let ok = nlohmann::sax_parse(s, &mut rec, true).expect("Rec::parse_error never throws");
    let actual = format!("S {} {}", if ok { 1 } else { 0 }, rec.ev);
    if actual != expected_s[i] {
      bad += 1;
      if bad <= 10 {
        eprintln!("input {} {:?}\n  rust: {}\n  cpp:  {}", i + 1, &s[..s.len().min(80)], &actual[..actual.len().min(300)], &expected_s[i][..expected_s[i].len().min(300)]);
      }
    }
    cases.ran();
  }
  assert_eq!(bad, 0, "{} of {} inputs differ from nlohmann", bad, ins.len());
}

/// `j.get<int>()` of a float is `static_cast<int>(double)`: out of range or NaN gives INT_MIN on
/// x86-64 (cvttsd2si), which is what the upstream binaries do.
#[test]
fn get_i32_out_of_range_float_is_int_min() {
  let mut cases = Cases::new("nlohmann_parse::get_i32_out_of_range_float_is_int_min");
  let f = |x: f64| nlohmann::get_i32(&serde_json::Value::from(x)).expect("number");
  for (x, e) in [
    (3.9, 3),
    (-3.9, -3),
    (2147483647.9, 2147483647),
    (-2147483648.9, i32::MIN),
    (2147483648.0, i32::MIN),
    (1e300, i32::MIN),
    (-1e300, i32::MIN),
    (-2147483649.0, i32::MIN),
  ] {
    assert_eq!(f(x), e, "get<int>({})", x);
    cases.ran();
  }
  // NaN cannot be stored in a serde_json Value (nor parsed by nlohmann); the cast helper covers it.
  assert_eq!(nlohmann::static_cast_i32_f64(f64::NAN), i32::MIN);
  cases.ran();
}
