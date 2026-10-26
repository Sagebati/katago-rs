//! Helpers shared by nn_parity.rs, wasm_human.rs (wasm32 under node: nothing here may touch std::fs), search_smoke.rs,
//! search_tolerance.rs (through common_oracle) and the testsearch* suites (through common_search): the lost-wakeup
//! watchdog for the single-threaded pool, the hex float tokens of the fixtures, and the `pos` line of the shared
//! position format (nneval_eigen.cpp) with its replay into a board and history. Not a port of upstream code.

#![allow(dead_code)]

use std::future::Future;
use std::time::Duration;

use futures::executor::LocalPool;

use katago_core::game::board::Board;
use katago_core::game::board::Location;
use katago_core::game::boardhistory::BoardHistory;
use katago_core::game::boardhistory::BoardHistoryModes;
use katago_core::game::rules::Rules;

/// Deterministic clock for `ClockTimer` (no time limit is set anywhere in these tests).
pub fn fake_clock() -> f64 {
  1000.0
}

/// Drives `fut` on `pool` like `LocalPool::run_until`, but never blocks forever: after each poll it
/// runs the other ready tasks (`run_until_stalled`); if `fut` is then pending and not woken, it
/// waits up to `grace` for a wakeup from outside the pool (a GPU readback callback) and panics
/// after that: a lost wakeup fails the test. flex completes everything inside the pool (grace 0).
/// Shared by nn_parity, search_smoke, search_tolerance and (through common_search) the testsearch* suites.
pub fn run_watchdog<F: Future>(pool: &mut LocalPool, label: &str, grace: Duration, fut: F) -> F::Output {
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
    let start = std::time::Instant::now();
    while !flag.0.load(Ordering::SeqCst) {
      if start.elapsed() >= grace {
        panic!("{label}: the executor stalled with the future pending and not woken for {grace:?} (lost wakeup)");
      }
      std::thread::sleep(Duration::from_millis(1));
      pool.run_until_stalled();
    }
  }
}

pub fn hex_u64(s: &str) -> u64 {
  u64::from_str_radix(s, 16).unwrap()
}
pub fn hex_u32(s: &str) -> u32 {
  u32::from_str_radix(s, 16).unwrap()
}

/// A `pos` line of the shared position format (see the oracle's header).
pub struct Pos {
  pub x_size: i32,
  pub y_size: i32,
  pub rules: Rules,
  pub encore_phase: i32,
  pub initial_pla: i8,
  pub assume_multi_handicap: bool,
  pub setup: Vec<(i32, i32, i8)>,
  pub moves: Vec<(i32, i32, i8, bool)>,
}

/// Parses the tokens of one `pos` line (the leading `pos <id>` included).
pub fn parse_pos_tokens(t: &[&str]) -> Pos {
  let n = |k: usize| -> i32 { t[k].parse().unwrap() };
  let rules = Rules::new_full(n(4), n(5), n(6), n(7) != 0, n(8) != 0, n(9), n(10) != 0, f32::from_bits(hex_u32(t[11])));
  let mut k = 15;
  let n_setup = n(k) as usize;
  k += 1;
  let mut setup = Vec::new();
  for _ in 0..n_setup {
    setup.push((n(k), n(k + 1), n(k + 2) as i8));
    k += 3;
  }
  let n_moves = n(k) as usize;
  k += 1;
  let mut moves = Vec::new();
  for _ in 0..n_moves {
    moves.push((n(k), n(k + 1), n(k + 2) as i8, n(k + 3) != 0));
    k += 4;
  }
  assert_eq!(k, t.len());
  Pos {
    x_size: n(2),
    y_size: n(3),
    rules,
    encore_phase: n(12),
    initial_pla: n(13) as i8,
    assume_multi_handicap: n(14) != 0,
    setup,
    moves,
  }
}

/// A `pos` line, replayed into a board and history.
pub fn replay(p: &Pos) -> (Board, BoardHistory) {
  let mut board = Board::new_x_y(p.x_size, p.y_size).unwrap();
  for &(x, y, c) in &p.setup {
    assert!(board.set_stone(Location::get_loc(x, y, p.x_size), c));
  }
  let mut hist = BoardHistory::new_full(&board, p.initial_pla, &p.rules, p.encore_phase, &BoardHistoryModes::new());
  hist.set_assume_multiple_starting_black_moves_are_handicap(p.assume_multi_handicap);
  for &(x, y, pla, prevent_encore) in &p.moves {
    let loc = if x == -1 { Board::PASS_LOC } else { Location::get_loc(x, y, p.x_size) };
    assert!(hist.is_legal(&board, loc, pla));
    hist.make_board_move_assume_legal(&mut board, loc, pla, None, prevent_encore);
  }
  (board, hist)
}
