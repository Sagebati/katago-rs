// Ported from KataGo cpp/core/timer.cpp, cpp/core/timer.h @ ba938676d7f42d70950b3a535af2466fb642008c
//
// Divergence (PORTING.md §10, manifest notes): no `std::chrono::steady_clock` / `Instant` in core
// (they panic on wasm32-unknown-unknown). The clock is injected once at startup with `set_clock`.
// The Windows (`GetTickCount64`) implementation of timer.cpp is not ported: one implementation,
// the UNIX one, with its clock replaced by the injected one.

/*
 * timer.h
 * Author: David Wu
 *
 * A simple class for getting elapsed runtime. Should be wall time, not cpu time.
 * Should also be threadsafe for concurrent calls to getSeconds()
 *
 */

use std::sync::OnceLock;

use crate::core::global::Global;

/// PORT NOTE: the injected monotonic clock that replaces `std::chrono::steady_clock::now()`.
/// It returns SECONDS since an arbitrary fixed origin as an `f64` (wasm host: `performance.now() /
/// 1000.0`; katago-cli: seconds since an `Instant` taken at startup). Installed once by `set_clock`.
static CLOCK: OnceLock<fn() -> f64> = OnceLock::new();

/// PORT NOTE: installs the clock used by every `ClockTimer` (once, at startup). Returns false and
/// keeps the first clock if one is already installed (`OnceLock`), so repeated installation of the
/// same clock by several tests/hosts is harmless.
pub fn set_clock(clock: fn() -> f64) -> bool {
  CLOCK.set(clock).is_ok()
}

/// PORT NOTE: `std::chrono::steady_clock::now().time_since_epoch()` as integer nanoseconds
/// (`duration<int64_t,std::nano>`), from the injected clock.
fn now_nanos() -> i64 {
  match CLOCK.get() {
    Some(clock) => {
      let seconds: f64 = clock();
      // PORT NOTE: in range because a monotonic clock value in seconds is far below 2^63 / 1e9
      // (292 years), so `as` truncation is the integer the C++ duration holds.
      (seconds * 1000000000.0) as i64
    }
    // TODO(port): no clock installed. Upstream always has one; a silent 0 would make every
    // maxTime / reportDuringSearchEvery check never fire, so this is fatal instead. The host
    // (katago-cli, katago-wasm, tests that build a Search) must call timer::set_clock first.
    None => Global::fatal_error("ClockTimer: no clock installed, call core::timer::set_clock once at startup"),
  }
}

/// upstream: ClockTimer (timer.h)
/// PORT NOTE: copy constructor and assignment are deleted upstream, so no Clone.
pub struct ClockTimer {
  initial_time: i64,
}

impl ClockTimer {
  /// upstream: ClockTimer::ClockTimer (timer.cpp)
  pub fn new() -> ClockTimer {
    let mut s = ClockTimer { initial_time: 0 };
    s.reset();
    s
  }

  // upstream: ClockTimer::~ClockTimer (timer.cpp)
  // PORT NOTE: empty destructor, omitted.

  /// upstream: ClockTimer::reset (timer.cpp)
  pub fn reset(&mut self) {
    let d: i64 = now_nanos();
    self.initial_time = d;
  }

  /// upstream: ClockTimer::getSeconds (timer.cpp)
  pub fn get_seconds(&self) -> f64 {
    let new_time: i64 = now_nanos();
    (new_time - self.initial_time) as f64 / 1000000000.0
  }

  /// upstream: ClockTimer::getPrecisionSystemTime (timer.cpp)
  //Return some integer indicating the current system time (for seeds/hashes), may vary with OS.
  pub fn get_precision_system_time() -> i64 {
    let new_time: i64 = now_nanos();
    new_time
  }
}
