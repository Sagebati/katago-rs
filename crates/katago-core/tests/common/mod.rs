//! Shared helpers for the fixture-driven tests (not a port of upstream code).
#![allow(dead_code)]

// Declared first: macro_rules! are textually scoped, so a test binary that uses `wasm_test!`
// declares this module with `#[macro_use] mod common;`.

/// A `#[test]` that also runs under wasm-bindgen-test on wasm32 (`just test-wasm`).
#[allow(unused_macros)]
macro_rules! wasm_test {
  ($(#[$m:meta])* fn $name:ident() $body:block) => {
    $(#[$m])*
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn $name() $body
  };
}

use std::path::PathBuf;

use katago_core::game::board::Board;

/// `fixtures/upstream-v1.18.2/unit`
pub fn fixture_dir() -> PathBuf {
  PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/upstream-v1.18.2/unit")
}

/// Reads `fixtures/upstream-v1.18.2/unit/<stem>/<file>`; panics if missing (an empty glob must fail).
#[cfg(not(target_arch = "wasm32"))]
pub fn fixture(stem: &str, file: &str) -> String {
  let p = fixture_dir().join(stem).join(file);
  std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("missing fixture {}: {}", p.display(), e))
}

/// `fixture(stem, file)` and `false`; or, natively, when the environment variable `var` names a file,
/// that file's text and `true` (exploration of another oracle output: callers skip their case counts).
pub fn fixture_or_env(var: &str, stem: &str, file: &str) -> (String, bool) {
  #[cfg(not(target_arch = "wasm32"))]
  if let Ok(p) = std::env::var(var) {
    return (std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {}", p, e)), true);
  }
  #[cfg(target_arch = "wasm32")]
  let _ = var;
  (fixture(stem, file), false)
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

/// Deterministic clock for `core::timer::set_clock`: the searches need one installed, and no time
/// limit is set where this is used.
pub fn fixed_clock() -> f64 {
  1000.0
}

/// The `message` of a `tracing` event.
struct Message(String);

impl tracing::field::Visit for Message {
  fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
    if field.name() == "message" {
      use std::fmt::Write as _;
      let _ = write!(self.0, "{value:?}");
    }
  }
}

struct CaptureLogs {
  target_prefix: &'static str,
  sink: fn(&str),
}

impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for CaptureLogs {
  fn on_event(&self, event: &tracing::Event<'_>, _ctx: tracing_subscriber::layer::Context<'_, S>) {
    if event.metadata().target().starts_with(self.target_prefix) {
      let mut msg = Message(String::new());
      event.record(&mut msg);
      (self.sink)(&msg.0);
    }
  }
}

/// Where the oracles' Logger lines go (the port logs through `tracing`): calls `sink(message)` for every event
/// whose target starts with `target_prefix`, on this thread until the returned guard drops.
pub fn capture_logs(target_prefix: &'static str, sink: fn(&str)) -> tracing::subscriber::DefaultGuard {
  use tracing_subscriber::layer::SubscriberExt as _;
  tracing::subscriber::set_default(tracing_subscriber::registry().with(CaptureLogs { target_prefix, sink }))
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

/// Watchdog for async tests (no upstream counterpart): drives `fut` to completion on `pool` like
/// `LocalPool::run_until`, but never parks the thread. After each poll of `fut` it runs every other
/// ready task (`run_until_stalled`); if the pool then stalls while `fut` is pending and nothing woke
/// it, nothing ever will (these tests have no outside events), so it panics with `label` instead of
/// hanging: a lost wakeup fails the test.
pub fn run_or_stall_panic<F: std::future::Future>(pool: &mut futures::executor::LocalPool, label: &str, fut: F) -> F::Output {
  use std::sync::Arc;
  use std::sync::atomic::AtomicBool;
  use std::sync::atomic::Ordering;
  use std::task::Context;
  use std::task::Poll;
  struct Flag(AtomicBool);
  impl futures::task::ArcWake for Flag {
    fn wake_by_ref(arc_self: &Arc<Self>) {
      arc_self.0.store(true, Ordering::SeqCst);
    }
  }
  let flag = Arc::new(Flag(AtomicBool::new(true)));
  let waker = futures::task::waker(flag.clone());
  let mut cx = Context::from_waker(&waker);
  let mut fut = std::pin::pin!(fut);
  loop {
    flag.0.store(false, Ordering::SeqCst);
    if let Poll::Ready(v) = fut.as_mut().poll(&mut cx) {
      return v;
    }
    pool.run_until_stalled();
    if !flag.0.load(Ordering::SeqCst) {
      panic!("{}: the executor stalled with the future pending and not woken (lost wakeup / deadlock)", label);
    }
  }
}
