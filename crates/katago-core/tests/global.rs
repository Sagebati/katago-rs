//! core/global: the functions the pilot left out, against the real C++
//! (fixtures/upstream-v1.18.2/unit/oracle/global.txt from oracle/global.cpp; same order).
#![allow(clippy::approx_constant)] // 3.14159265 is a deliberate test input, not a pi approximation

#[macro_use]
mod common;

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::fmt::Write as _;

use common::Cases;
use common::compare_lines;
use common::fixture;
use katago_core::core::global::*;
use katago_core::port::fmt::cpp_ostream_f64;

fn q(s: &str) -> String {
  format!("[{}]", s)
}

fn b(x: bool) -> i32 {
  x as i32
}

wasm_test! {
fn global_vs_cpp() {
  let mut cases = Cases::new("global::global_vs_cpp");
  let mut out = String::new();
  // charToString
  for c in [0u8, b'a', b' ', b'~', b'0'] {
    let s = Global::char_to_string(c);
    writeln!(out, "charToString {} {} {}", c, s.len(), q(&s)).unwrap();
  }
  // chopPrefix / chopSuffix (C++ prints the " -> " prefix before the exception escapes)
  let chops = [("abcdef", "abc"), ("abcdef", ""), ("abc", "abcdef"), ("abcdef", "bcd"), ("", ""), ("abcdef", "abcdef"), ("abcdef", "def")];
  for (a, p) in chops {
    write!(out, "chopPrefix {}{} -> ", q(a), q(p)).unwrap();
    match Global::chop_prefix(a, p) {
      Ok(r) => writeln!(out, "{}", q(&r)).unwrap(),
      Err(e) => writeln!(out, "chopPrefix {}{} throws {}", q(a), q(p), q(e.what())).unwrap(),
    }
    write!(out, "chopSuffix {}{} -> ", q(a), q(p)).unwrap();
    match Global::chop_suffix(a, p) {
      Ok(r) => writeln!(out, "{}", q(&r)).unwrap(),
      Err(e) => writeln!(out, "chopSuffix {}{} throws {}", q(a), q(p), q(e.what())).unwrap(),
    }
  }
  // concat
  {
    let arr: [&str; 4] = ["a", "", "bc", "def"];
    for len in 0..=4usize {
      writeln!(out, "concat_len {} {} {}", len, q(&Global::concat_len(&arr, len, ", ")), q(&Global::concat_len(&arr, len, ""))).unwrap();
    }
    let v: Vec<String> = vec!["x".into(), "yy".into(), "".into(), "zzz".into()];
    writeln!(out, "concat {} {}", q(&Global::concat(&v, "|")), q(&Global::concat(&[], "|"))).unwrap();
    for s in 0..=4usize {
      for e in s..=4usize {
        writeln!(out, "concat_start_end {} {} {}", s, e, q(&Global::concat_start_end(&v, "-", s, e))).unwrap();
      }
    }
    let st: BTreeSet<String> = ["b", "a", "c", "aa", "B"].iter().map(|x| x.to_string()).collect();
    writeln!(out, "concat_set {} {}", q(&Global::concat_set(&st, ",")), q(&Global::concat_set(&BTreeSet::new(), ","))).unwrap();
  }
  // isDigits / parseDigits
  let digs = ["", "0", "7", "123", "000000000", "999999999", "1234567890", "2147483647", "12a", "a12", "-1", " 1", "1 ", "00000000000", "0x1", "١"];
  for s in digs {
    write!(out, "isDigits {} {}", q(s), b(Global::is_digits(s))).unwrap();
    write!(out, " parseDigits ").unwrap();
    match Global::parse_digits(s) {
      Ok(v) => write!(out, "{}", v).unwrap(),
      Err(e) => write!(out, " parseDigits throws {}", q(e.what())).unwrap(),
    }
    writeln!(out).unwrap();
    for a in 0..=s.len() + 1 {
      for bb in 0..=s.len() + 1 {
        if a > s.len() {
          continue; // substr(a,...) with a > size throws std::out_of_range (not mirrored)
        }
        write!(out, "isDigits_se {} {} {} {}", q(s), a, bb, b(Global::is_digits_start_end(s, a, bb))).unwrap();
        write!(out, " parseDigits_se ").unwrap();
        match Global::parse_digits_start_end(s, a, bb) {
          Ok(v) => write!(out, "{}", v).unwrap(),
          Err(e) => write!(out, " parseDigits_se throws {}", q(e.what())).unwrap(),
        }
        writeln!(out).unwrap();
      }
    }
  }
  // stringCharsAllAllowed
  let allowed = [("abc", "abc"), ("abc", "ab"), ("", "x"), ("aaa", "a"), ("a b", "ab "), ("xyz", "")];
  for (s, a) in allowed {
    writeln!(out, "stringCharsAllAllowed {}{} {}", q(s), q(a), b(Global::string_chars_all_allowed(s, a))).unwrap();
  }
  // stripComments
  let comments = ["", "abc", "a#b", "#", "x = 1 # c\ny = 2\n#full\n\nz=3", "no newline # end", "a\nb#\n", "\n\n#\n"];
  for s in comments {
    writeln!(out, "stripComments {} -> {}", q(s), q(&Global::strip_comments(s))).unwrap();
  }
  // readKeyValues
  let kvs = ["", "a=1", "a = 1, b= 2\nc=3", "a=1\na=2", "=1", "a=", "a", "a=1,,b=2", "x=y=z", " k = v w , m=n\n\n,\np=q", "a=1\r\nb=2", "a=1,b"];
  for s in kvs {
    write!(out, "readKeyValues {} ->", q(s)).unwrap();
    match Global::read_key_values(s) {
      Ok(m) => {
        for (k, v) in m.iter() {
          write!(out, " {}={}", q(k), q(v)).unwrap();
        }
      }
      Err(e) => write!(out, " throws {}", q(e.what())).unwrap(),
    }
    writeln!(out).unwrap();
  }
  // readMem
  let mems = [
    "", "1", "12", "16G", "16GB", "256K", "256KB", "3M", "3MB", "1T", "1TB", "2P", "2PB", "100B", "100", "K", "KB", "1.5G", "-1G", "999999999P",
    "8191P", "8192P", "16383T", "1GG", "1BK", "123456789", "1234567890", "1kB", "G1",
  ];
  for s in mems {
    write!(out, "readMem {}", q(s)).unwrap();
    write!(out, " ").unwrap();
    match Global::read_mem(s) {
      Ok(v) => write!(out, "{}", v).unwrap(),
      Err(e) => write!(out, " throws {}", q(e.what())).unwrap(),
    }
    writeln!(out).unwrap();
  }
  // roundStatic
  let xs = [0.0, 1.0, -1.0, 0.5, -0.5, 1.25, 2.675, -2.675, 123.456789, 1e-9, 3.14159265];
  let scales = [1.0, 10.0, 100.0, 1000.0, 0.1, 4.0];
  for x in xs {
    for sc in scales {
      writeln!(
        out,
        "roundStatic {} {} {}",
        Global::double_to_string_high_precision(x),
        cpp_ostream_f64(sc),
        Global::double_to_string_high_precision(Global::round_static(x, sc))
      )
      .unwrap();
    }
  }
  // contains / indexOf / map helpers
  let vs: Vec<String> = vec!["a".into(), "bb".into(), "".into(), "a".into()];
  writeln!(
    out,
    "contains_vec {}{}{}{}",
    b(contains_vector(&vs, "a")),
    b(contains_vector(&vs, "")),
    b(contains_vector(&vs, "c")),
    b(contains_vector(&vs, &String::from("bb")))
  )
  .unwrap();
  writeln!(out, "indexOf {} {} {}", index_of(&vs, "a"), index_of(&vs, ""), b(index_of(&vs, "zz") == NPOS)).unwrap();
  let vi: Vec<i32> = vec![3, 1, 4, 1, 5];
  writeln!(out, "contains_vec_int {}{} indexOf {} {}", b(contains_vector(&vi, &1)), b(contains_vector(&vi, &2)), index_of(&vi, &1), index_of(&vi, &5)).unwrap();
  let ss: BTreeSet<String> = ["x", "y"].iter().map(|x| x.to_string()).collect();
  writeln!(out, "contains_set {}{}{}", b(contains_set(&ss, "x")), b(contains_set(&ss, "z")), b(contains_set(&ss, &String::from("y")))).unwrap();
  let mut ms: BTreeMap<String, i32> = BTreeMap::new();
  ms.insert("k".into(), 7);
  ms.insert("j".into(), 8);
  writeln!(out, "contains_map {}{}{}", b(contains_map(&ms, "k")), b(contains_map(&ms, "q")), b(contains_map(&ms, &String::from("j")))).unwrap();
  writeln!(
    out,
    "map_get {} {} {} {}",
    map_get_str(&ms, "k"),
    map_get(&ms, &String::from("j")),
    map_get_defaulting(&ms, &String::from("q"), &-1),
    map_get_defaulting(&ms, &String::from("k"), &-1)
  )
  .unwrap();
  writeln!(
    out,
    "containsChar {}{}{}{}",
    b(contains_str_char("abc", b'b')),
    b(contains_str_char("abc", b'z')),
    b(contains_str_char("abc", b'c')),
    b(contains_str_char("abc", 0))
  )
  .unwrap();


  // NUL bytes: `const char*` parameters end at the first NUL, std::string ones do not.
  writeln!(
    out,
    "nul_contains {}{}{}{}",
    b(contains_str_char("a\0b", b'b')),
    b(contains_str_char("a\0b", b'a')),
    b(contains_str_char("a\0b", 0)),
    b(contains_str_char("a\0b", b'b'))
  )
  .unwrap();
  writeln!(
    out,
    "nul_allowed {}{}{}{}",
    b(Global::string_chars_all_allowed("b", "a\0b")),
    b(Global::string_chars_all_allowed("a", "a\0b")),
    b(Global::string_chars_all_allowed("a\0", "a")),
    b(Global::string_chars_all_allowed("a\0b", "ab"))
  )
  .unwrap();
  {
    let arr: [&str; 2] = ["a\0b", "c"];
    let c = Global::concat_len(&arr, 2, ",");
    writeln!(out, "nul_concat_len {} {}", c.len(), hx(&c)).unwrap();
    let c = Global::concat_len(&arr, 2, ";\0;");
    writeln!(out, "nul_concat_len_delim {} {}", c.len(), hx(&c)).unwrap();
    let v: Vec<String> = vec!["x\0y".into(), "z".into()];
    let c = Global::concat(&v, "+\0-");
    writeln!(out, "nul_concat {} {}", c.len(), hx(&c)).unwrap();
    let c = Global::concat_start_end(&v, "+\0-", 0, 2);
    writeln!(out, "nul_concat_start_end {} {}", c.len(), hx(&c)).unwrap();
    let st: BTreeSet<String> = ["p", "q"].iter().map(|x| x.to_string()).collect();
    let c = Global::concat_set(&st, "+\0-");
    writeln!(out, "nul_concat_set {} {}", c.len(), hx(&c)).unwrap();
  }
  {
    let t = Global::trim("x\0 ", " \0");
    writeln!(out, "nul_trim {} {}", t.len(), hx(&t)).unwrap();
    let t = Global::trim(" \0x\0 ", " \0x");
    writeln!(out, "nul_trim2 {} {}", t.len(), hx(&t)).unwrap();
    let t = Global::trim("\0x\0", Global::TRIM_DEFAULT_DELIM);
    writeln!(out, "nul_trim3 {} {}", t.len(), hx(&t)).unwrap();
  }
  writeln!(out, "nul_map_get {}", map_get_str(&ms, "k\0x")).unwrap();
  {
    let t = katago_core::strprintf!("a\0%d", 5);
    writeln!(out, "nul_strprintf {} {}", t.len(), hx(&t)).unwrap();
  }
  // Non-ASCII bytes. C++ results that are not valid UTF-8 ("lossy" lines) cannot be a Rust String:
  // Global::trim returns them with U+FFFD in place of the stray bytes (String::from_utf8_lossy), and
  // they are checked against the C++ bytes below instead of line by line.
  let trims: [(&str, &str); 8] = [
    ("\u{e9}", "\u{a9}"), // "é", "©": cuts the shared A9 byte -> "\xC3"
    ("  \u{e9}  ", " \t\r\n\x0b\x0c"),
    ("\u{e9}", "\u{e9}"),
    ("a\u{e9}", "\u{a9}"),
    ("\u{e9}a\u{e9}", "\u{e9}"),
    // C++ delims "\xC3" (a lone byte, not a valid &str); "\u{c3}" = C3 83 acts the same here since
    // 83 does not occur in "\xC3\xA9a".
    ("\u{e9}a", "\u{c3}"),
    // "€" = E2 82 AC. C++ delims "\xE2\xAC"; "\u{2020}\u{ac}" = E2 80 A0 C2 AC acts the same (no 80/A0/C2 in "€").
    ("\u{20ac}", "\u{2020}\u{ac}"),
    ("\u{20ac}", "\u{a9}"),
  ];
  let mut lossy: Vec<(String, String)> = Vec::new();
  for (i, (t0, d0)) in trims.iter().enumerate() {
    let t = Global::trim(t0, d0);
    let name = format!("trim{}", i);
    if t.contains('\u{fffd}') {
      lossy.push((name, t));
    } else {
      writeln!(out, "bytes {} {} {}", name, t.len(), hx(&t)).unwrap();
    }
  }
  {
    let t = Global::chop_prefix("\u{e9}x", "\u{e9}").unwrap();
    writeln!(out, "bytes chopPrefix {} {}", t.len(), hx(&t)).unwrap();
    let t = Global::chop_suffix("x\u{e9}", "\u{e9}").unwrap();
    writeln!(out, "bytes chopSuffix {} {}", t.len(), hx(&t)).unwrap();
  }

  let expected = fixture("oracle", "global.txt");
  let mut expected_exact = String::new();
  let mut expected_lossy: Vec<(String, Vec<u8>)> = Vec::new();
  for line in expected.split_inclusive('\n') {
    if let Some(rest) = line.strip_prefix("lossy ") {
      let f: Vec<&str> = rest.trim_end_matches('\n').split(' ').collect();
      expected_lossy.push((f[0].to_string(), unhex(f[2])));
    } else {
      expected_exact += line;
    }
  }
  compare_lines("oracle/global.txt", &out, &expected_exact);
  assert_eq!(lossy.len(), expected_lossy.len(), "lossy cases: {:?} vs C++ {:?}", lossy, expected_lossy);
  for ((name, t), (cpp_name, cpp_bytes)) in lossy.iter().zip(expected_lossy.iter()) {
    assert_eq!(name, cpp_name);
    assert_eq!(*t, String::from_utf8_lossy(cpp_bytes), "{}: Rust must be the C++ bytes with U+FFFD for invalid UTF-8", name);
  }
  cases.ran();
}
}

fn hx(s: &str) -> String {
  s.bytes().map(|c| format!("{:02X}", c)).collect()
}

fn unhex(h: &str) -> Vec<u8> {
  (0..h.len()).step_by(2).map(|i| u8::from_str_radix(&h[i..i + 2], 16).unwrap()).collect()
}

wasm_test! {
/// CustomScopeGuard / WrappedWithDeleter / make_unique_void: no C++ output to compare; checks the
/// destructor semantics (guard runs once at scope end; deleter only when assigned).
fn global_raii_helpers() {
  let mut cases = Cases::new("global::global_raii_helpers");
  let ran = std::cell::Cell::new(0);
  {
    let _guard = CustomScopeGuard::new(|| ran.set(ran.get() + 1));
    assert_eq!(ran.get(), 0);
  }
  assert_eq!(ran.get(), 1);
  cases.ran();

  thread_local! { static DELETED: std::cell::Cell<i64> = const { std::cell::Cell::new(0) }; }
  fn deleter(x: i32) -> i32 {
    DELETED.with(|d| d.set(d.get() * 100 + x as i64));
    0
  }
  {
    let _w: WrappedWithDeleter<i32, i32> = WrappedWithDeleter::new(deleter);
  }
  assert_eq!(DELETED.with(|d| d.get()), 0);
  {
    let mut w: WrappedWithDeleter<i32, i32> = WrappedWithDeleter::new(deleter);
    w.assign(5);
    *w.get_mut() += 1;
    assert_eq!(w.get(), 6);
  }
  assert_eq!(DELETED.with(|d| d.get()), 6);
  {
    let _w: WrappedWithDeleter<i32, i32> = WrappedWithDeleter::new_v(7, deleter);
  }
  assert_eq!(DELETED.with(|d| d.get()), 607);
  cases.ran();

  let p: unique_ptr_void = make_unique_void(Box::new(42i32));
  assert_eq!(p.downcast_ref::<i32>(), Some(&42));
  cases.ran();
}
}
