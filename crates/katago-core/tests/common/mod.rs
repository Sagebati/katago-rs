//! Shared helpers for the fixture-driven tests (not a port of upstream code).
#![allow(dead_code)]

use std::path::PathBuf;

use katago_core::game::board::Board;

/// `fixtures/upstream-v1.16.5/unit`
pub fn fixture_dir() -> PathBuf {
  PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/upstream-v1.16.5/unit")
}

/// Reads `fixtures/upstream-v1.16.5/unit/<stem>/<file>`; panics if missing (an empty glob must fail).
#[cfg(not(target_arch = "wasm32"))]
pub fn fixture(stem: &str, file: &str) -> String {
  let p = fixture_dir().join(stem).join(file);
  std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("missing fixture {}: {}", p.display(), e))
}

/// wasm32 (`just test-wasm`, run under node by wasm-bindgen-test-runner): read the same file
/// through node's `fs`; the path is absolute at compile time and the runner is on the same host.
#[cfg(target_arch = "wasm32")]
pub fn fixture(stem: &str, file: &str) -> String {
  let p = fixture_dir().join(stem).join(file);
  let path = p.to_str().expect("fixture path is UTF-8");
  node_fs::read_file_sync(path, "utf8").unwrap_or_else(|e| panic!("missing fixture {}: {:?}", path, e))
}

#[cfg(target_arch = "wasm32")]
mod node_fs {
  use wasm_bindgen::prelude::*;
  #[wasm_bindgen(module = "fs")]
  extern "C" {
    #[wasm_bindgen(js_name = readFileSync, catch)]
    pub fn read_file_sync(path: &str, encoding: &str) -> Result<String, JsValue>;
  }
}

/// `eprintln!` that also reaches the node console on wasm32 (stderr is a no-op there).
#[macro_export]
macro_rules! report {
  ($($t:tt)*) => {{
    #[cfg(not(target_arch = "wasm32"))]
    eprintln!($($t)*);
    #[cfg(target_arch = "wasm32")]
    wasm_bindgen_test::console_log!($($t)*);
  }};
}

/// Upstream's test driver calls Board::initHash() at program start.
pub fn init() {
  Board::init_hash();
}

/// `out << endl`
pub fn endl(out: &mut String) {
  out.push('\n');
}

/// Counts cases run by one #[test] and reports them (PORTING.md §13: tests must prove they ran).
pub struct Cases {
  name: &'static str,
  n: usize,
}

impl Cases {
  pub fn new(name: &'static str) -> Cases {
    Cases { name, n: 0 }
  }
  pub fn ran(&mut self) {
    self.n += 1;
  }
}

impl Drop for Cases {
  fn drop(&mut self) {
    if !std::thread::panicking() {
      assert!(self.n > 0, "{}: no cases ran", self.name);
      crate::report!("{}: {} cases", self.name, self.n);
    }
  }
}

/// Exact equality against an oracle fixture: the same bytes, including every line ending (`\r`
/// kept) and the trailing newline (or its absence). On a mismatch it reports the first differing
/// line, with escapes so that `\r` and missing newlines are visible.
pub fn compare_lines(label: &str, actual: &str, expected: &str) {
  if actual == expected {
    return;
  }
  let a: Vec<&str> = actual.split_inclusive('\n').collect();
  let e: Vec<&str> = expected.split_inclusive('\n').collect();
  for i in 0..std::cmp::max(a.len(), e.len()) {
    let al = a.get(i).copied();
    let el = e.get(i).copied();
    if al != el {
      panic!(
        "{}: line {} differs (actual {} lines / {} bytes, expected {} lines / {} bytes)\n  actual: {:?}\nexpected: {:?}",
        label,
        i + 1,
        a.len(),
        actual.len(),
        e.len(),
        expected.len(),
        al,
        el
      );
    }
  }
  unreachable!("{}: strings differ but every line is equal", label);
}

/// `dbits(x)` in oracle.cpp: the f64 bit pattern as 16 uppercase hex digits.
pub fn dbits(x: f64) -> String {
  katago_core::core::global::Global::uint64_to_hex_string(x.to_bits())
}
