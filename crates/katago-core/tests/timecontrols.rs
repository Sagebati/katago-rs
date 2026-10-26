//! search/timecontrols (TimeControls) against the real C++: fixtures/upstream-v1.18.2/unit/oracle/
//! timecontrols{,_cases,_testtime}.txt from oracle/timecontrols.cpp (same order; the oracle's header
//! has the build command).
//!
//! - `testtime_replay`: upstream's own cpp/tests/testtime.cpp (Tests::runTimeControlsTests), replayed
//!   call for call (the 53 time controls x 4 boards; board inputs are the extracted
//!   fixtures/.../testtime/*.input*.txt blocks) and compared byte-exact with the output of the
//!   unmodified upstream test.
//! - `fixed_sequence_and_sweep`: the factories, errors, isEffectivelyUnlimitedTime, getTime,
//!   roundUpTimeLimitIfNeeded and both toDebugString forms on named time controls, then on 4000
//!   generated cases whose fields are read back from timecontrols_cases.txt as exact hex bits.
//! - `clock_timer`: core/timer.rs with an injected fake clock (no upstream counterpart to compare).
//!
//! Doubles are compared as bits (16 hex digits; "nan" for any NaN), as in the oracle.

#![allow(clippy::bool_comparison, clippy::needless_range_loop, clippy::vec_init_then_push)]

#[macro_use]
mod common;

use std::fmt::Write as _;

use common::Cases;
use common::compare_lines;
use common::fixture;
use common::init;
use katago_core::core::global::KataError;
use katago_core::game::board::*;
use katago_core::game::boardhistory::*;
use katago_core::game::rules::Rules;
use katago_core::port::fmt::cpp_ostream_f64;
use katago_core::search::timecontrols::TimeControls;

fn bits(x: f64) -> String {
  if x.is_nan() {
    return String::from("nan");
  }
  format!("{:016X}", x.to_bits())
}

fn q(s: &str) -> String {
  format!("[{}]", s)
}

fn fields(t: &TimeControls) -> String {
  format!(
    "{} {} {} {} {} {} {} {} {} {} {} {}",
    bits(t.original_main_time),
    bits(t.increment),
    bits(t.main_time_limit),
    bits(t.max_time_per_move),
    t.original_num_periods,
    t.num_stones_per_period,
    bits(t.per_period_time),
    bits(t.main_time_left),
    t.in_overtime as i32,
    t.num_periods_left_including_current,
    t.num_stones_left_in_period,
    bits(t.time_left_in_period)
  )
}

/// `makeBoard` of the oracle: raster-order stones from a fixed colour pattern.
fn make_board(xs: i32, ys: i32, n: i32) -> Board {
  let mut b = Board::new_x_y(xs, ys).unwrap();
  let mut placed = 0;
  for y in 0..ys {
    for x in 0..xs {
      if placed >= n {
        break;
      }
      let loc = Location::get_loc(x, y, xs);
      let c = if (x * 7 + y * 3) % 5 < 2 { C_BLACK } else { C_WHITE };
      b.set_stone(loc, c);
      placed += 1;
    }
  }
  b
}

/// A board and its history, built in a frame of their own: debug wasm has a 1 MiB stack and every
/// temporary of a Board/BoardHistory (4.7 KB / 34 KB) gets its own slot (PORTING.md §13 "wasm stack").
fn boxed_history(board: Board) -> (Box<Board>, Box<BoardHistory>) {
  let board = Box::new(board);
  let hist = Box::new(BoardHistory::new_full(&board, P_BLACK, &Rules::new(), 0));
  (board, hist)
}

fn evaluate(tc: &TimeControls, board: &Board, hist: &BoardHistory, lag_buffer: f64) -> String {
  let mut o = String::new();
  write!(o, "stones {} unlimited {}", board.num_stones_on_board(), tc.is_effectively_unlimited_time() as i32).unwrap();
  let mut min_time = -7.0;
  let mut rec = -7.0;
  let mut max_time = -7.0;
  let mut lim = 7.5;
  match tc.get_time(board, hist, lag_buffer, &mut min_time, &mut rec, &mut max_time) {
    Ok(()) => {
      write!(o, " getTime {} {} {}", bits(min_time), bits(rec), bits(max_time)).unwrap();
      lim = rec;
    }
    Err(e) => {
      assert!(e.is_string_error());
      write!(o, " getTime throws {}", q(e.what())).unwrap();
    }
  }
  let rrec0 = tc.round_up_time_limit_if_needed(lag_buffer, 0.0, lim);
  let rreclimit = tc.round_up_time_limit_if_needed(lag_buffer, lim - 0.000001, lim);
  let rreclimit2 = tc.round_up_time_limit_if_needed(lag_buffer, rreclimit - 0.000001, rreclimit);
  let rhalf = tc.round_up_time_limit_if_needed(lag_buffer, lim * 0.5, lim);
  let rover = tc.round_up_time_limit_if_needed(lag_buffer, lim + 3.0, lim);
  let rmain = tc.round_up_time_limit_if_needed(lag_buffer, tc.main_time_left * 0.9, lim * 3.0 + 1.0);
  write!(o, " round {} {} {} {} {} {}", bits(rrec0), bits(rreclimit), bits(rreclimit2), bits(rhalf), bits(rover), bits(rmain)).unwrap();
  write!(o, " debug {}", q(&tc.to_debug_string())).unwrap();
  match tc.to_debug_string_board_hist_lag_buffer(board, hist, lag_buffer) {
    Ok(s) => write!(o, " debugfull {}", q(&s)).unwrap(),
    Err(e) => write!(o, " debugfull throws {}", q(e.what())).unwrap(),
  }
  o
}

// ===================================================================== testtime.cpp (Tests::runTimeControlsTests)

// PORTING.md §13 "wasm stack": Board/BoardHistory are big, so they are boxed (debug wasm has a 1 MiB stack).
struct Boards {
  board9_early: Box<Board>,
  hist9_early: Box<BoardHistory>,
  board9_late: Box<Board>,
  hist9_late: Box<BoardHistory>,
  board19_early: Box<Board>,
  hist19_early: Box<BoardHistory>,
  board19_late: Box<Board>,
  hist19_late: Box<BoardHistory>,
}

/// `tryTimeControlsOnBoard` lambda
fn try_time_controls_on_board(out: &mut String, s: &str, time_controls: &TimeControls, board: &Board, hist: &BoardHistory, lag_buffer: f64) {
  let mut min_time = 0.0;
  let mut recommended_time = 0.0;
  let mut max_time = 0.0;
  time_controls.get_time(board, hist, lag_buffer, &mut min_time, &mut recommended_time, &mut max_time).unwrap();
  //Rounded time limit recommendation at the start of search
  let rrec0 = time_controls.round_up_time_limit_if_needed(lag_buffer, 0.0, recommended_time);
  //Rounded time limit recommendation as we're just about to hit limit
  let rreclimit = time_controls.round_up_time_limit_if_needed(lag_buffer, recommended_time - 0.000001, recommended_time);
  //Rounded time limit recommendation as we're just about to hit rounded limit
  let rreclimit2 = time_controls.round_up_time_limit_if_needed(lag_buffer, rreclimit - 0.000001, rreclimit);

  writeln!(
    out,
    "{} min rec max = {} {} {} roundedrec(used0) {} roundedrec(usedlimit) {} roundedrec(usedlimit2) {}",
    s,
    cpp_ostream_f64(min_time),
    cpp_ostream_f64(recommended_time),
    cpp_ostream_f64(max_time),
    cpp_ostream_f64(rrec0),
    cpp_ostream_f64(rreclimit),
    cpp_ostream_f64(rreclimit2)
  )
  .unwrap();
}

/// `tryTimeControlsOnBoards` lambda
fn try_time_controls_on_boards(out: &mut String, boards: &Boards, time_controls: &TimeControls, lag_buffer: f64) {
  try_time_controls_on_board(out, "board9Early", time_controls, &boards.board9_early, &boards.hist9_early, lag_buffer);
  try_time_controls_on_board(out, "board9Late", time_controls, &boards.board9_late, &boards.hist9_late, lag_buffer);
  try_time_controls_on_board(out, "board19Early", time_controls, &boards.board19_early, &boards.hist19_early, lag_buffer);
  try_time_controls_on_board(out, "board19Late", time_controls, &boards.board19_late, &boards.hist19_late, lag_buffer);
}

/// One `{ cout << ===...; cout << title; TimeControls timeControls; <assignments>; double lagBuffer; try...OnBoards }`
/// block of runTimeControlsTests; `set` holds the block's field assignments in order.
fn case(out: &mut String, boards: &Boards, title: &str, lag_buffer: f64, set: impl FnOnce(&mut TimeControls)) {
  let rule = "===================================================================";
  writeln!(out, "{}", rule).unwrap();
  writeln!(out, "{}", title).unwrap();
  writeln!(out, "{}", rule).unwrap();

  let mut time_controls = TimeControls::new();
  set(&mut time_controls);
  try_time_controls_on_boards(out, boards, &time_controls, lag_buffer);
}

wasm_test! {
fn testtime_replay() {
  init();
  let mut cases = Cases::new("timecontrols::testtime_replay");
  let f = "testtime";
  //Board board9Early = Board::parseBoard(9,9,R"%%(...)%%"); BoardHistory hist9Early(board9Early,P_BLACK,Rules(),0); etc.
  let (board9_early, hist9_early) = boxed_history(Board::parse_board(9, 9, &fixture(f, "runtimecontrolstests.input0.txt"), '\n').unwrap());
  let (board9_late, hist9_late) = boxed_history(Board::parse_board(9, 9, &fixture(f, "runtimecontrolstests.input1.txt"), '\n').unwrap());
  let (board19_early, hist19_early) = boxed_history(Board::parse_board(19, 19, &fixture(f, "runtimecontrolstests.input2.txt"), '\n').unwrap());
  let (board19_late, hist19_late) = boxed_history(Board::parse_board(19, 19, &fixture(f, "runtimecontrolstests.input3.txt"), '\n').unwrap());
  let boards = Boards { board9_early, hist9_early, board9_late, hist9_late, board19_early, hist19_early, board19_late, hist19_late };

  let mut out = String::new();
  case(&mut out, &boards, "Unlimited time controls", 0.0, |_t| {});
  case(&mut out, &boards, "Basic 1h absolute time controls, all time left", 1.0, |t| { t.original_main_time = 3600.0; t.increment = 0.0; t.original_num_periods = 0; t.num_stones_per_period = 0; t.per_period_time = 0.0; t.main_time_left = 3600.0; t.in_overtime = false; t.num_periods_left_including_current = 0; t.num_stones_left_in_period = 0; t.time_left_in_period = 0.0; });
  case(&mut out, &boards, "Basic 1h absolute time controls, 10m left", 1.0, |t| { t.original_main_time = 3600.0; t.increment = 0.0; t.original_num_periods = 0; t.num_stones_per_period = 0; t.per_period_time = 0.0; t.main_time_left = 600.0; t.in_overtime = false; t.num_periods_left_including_current = 0; t.num_stones_left_in_period = 0; t.time_left_in_period = 0.0; });
  case(&mut out, &boards, "Basic 1h fischer time controls, 10m left, 10s increment", 1.0, |t| { t.original_main_time = 3600.0; t.increment = 10.0; t.original_num_periods = 0; t.num_stones_per_period = 0; t.per_period_time = 0.0; t.main_time_left = 600.0; t.in_overtime = false; t.num_periods_left_including_current = 0; t.num_stones_left_in_period = 0; t.time_left_in_period = 0.0; });
  case(&mut out, &boards, "Basic 1h fischer time controls, 10m left, 10s increment, larger lag buffer", 5.0, |t| { t.original_main_time = 3600.0; t.increment = 10.0; t.original_num_periods = 0; t.num_stones_per_period = 0; t.per_period_time = 0.0; t.main_time_left = 600.0; t.in_overtime = false; t.num_periods_left_including_current = 0; t.num_stones_left_in_period = 0; t.time_left_in_period = 0.0; });
  case(&mut out, &boards, "Basic 10m fischer time controls, 10m left, 10s increment, main time limit 10m", 1.0, |t| { t.original_main_time = 600.0; t.increment = 10.0; t.main_time_limit = 600.0; t.original_num_periods = 0; t.num_stones_per_period = 0; t.per_period_time = 0.0; t.main_time_left = 600.0; t.in_overtime = false; t.num_periods_left_including_current = 0; t.num_stones_left_in_period = 0; t.time_left_in_period = 0.0; });
  case(&mut out, &boards, "Basic 10m fischer time controls, 10m left, 10s increment, main time limit 10m+5s", 1.0, |t| { t.original_main_time = 600.0; t.increment = 10.0; t.main_time_limit = 605.0; t.original_num_periods = 0; t.num_stones_per_period = 0; t.per_period_time = 0.0; t.main_time_left = 600.0; t.in_overtime = false; t.num_periods_left_including_current = 0; t.num_stones_left_in_period = 0; t.time_left_in_period = 0.0; });
  case(&mut out, &boards, "Basic 1h fischer time controls, 10m left, 10s increment, max time per move 20s", 1.0, |t| { t.original_main_time = 3600.0; t.increment = 10.0; t.max_time_per_move = 20.0; t.original_num_periods = 0; t.num_stones_per_period = 0; t.per_period_time = 0.0; t.main_time_left = 600.0; t.in_overtime = false; t.num_periods_left_including_current = 0; t.num_stones_left_in_period = 0; t.time_left_in_period = 0.0; });
  case(&mut out, &boards, "Basic 1h fischer time controls, 10m left, 10s increment, max time per move 30s", 1.0, |t| { t.original_main_time = 3600.0; t.increment = 10.0; t.max_time_per_move = 30.0; t.original_num_periods = 0; t.num_stones_per_period = 0; t.per_period_time = 0.0; t.main_time_left = 600.0; t.in_overtime = false; t.num_periods_left_including_current = 0; t.num_stones_left_in_period = 0; t.time_left_in_period = 0.0; });
  case(&mut out, &boards, "Basic 1h fischer time controls, 15s left, 10s increment", 1.0, |t| { t.original_main_time = 3600.0; t.increment = 10.0; t.original_num_periods = 0; t.num_stones_per_period = 0; t.per_period_time = 0.0; t.main_time_left = 15.0; t.in_overtime = false; t.num_periods_left_including_current = 0; t.num_stones_left_in_period = 0; t.time_left_in_period = 0.0; });
  case(&mut out, &boards, "Basic 1h fischer time controls, 15s left, 10s increment, max time per move 11s", 1.0, |t| { t.original_main_time = 3600.0; t.increment = 10.0; t.max_time_per_move = 11.0; t.original_num_periods = 0; t.num_stones_per_period = 0; t.per_period_time = 0.0; t.main_time_left = 15.0; t.in_overtime = false; t.num_periods_left_including_current = 0; t.num_stones_left_in_period = 0; t.time_left_in_period = 0.0; });
  case(&mut out, &boards, "Basic 1h fischer time controls, 15s left, 10s increment, max time per move 7s", 1.0, |t| { t.original_main_time = 3600.0; t.increment = 10.0; t.max_time_per_move = 7.0; t.original_num_periods = 0; t.num_stones_per_period = 0; t.per_period_time = 0.0; t.main_time_left = 15.0; t.in_overtime = false; t.num_periods_left_including_current = 0; t.num_stones_left_in_period = 0; t.time_left_in_period = 0.0; });
  case(&mut out, &boards, "Basic 1h fischer time controls, 5s left, 10s increment", 1.0, |t| { t.original_main_time = 3600.0; t.increment = 10.0; t.original_num_periods = 0; t.num_stones_per_period = 0; t.per_period_time = 0.0; t.main_time_left = 5.0; t.in_overtime = false; t.num_periods_left_including_current = 0; t.num_stones_left_in_period = 0; t.time_left_in_period = 0.0; });
  case(&mut out, &boards, "Basic 5s fischer time controls, 5s left, 10s increment, 6s main time limit", 1.0, |t| { t.original_main_time = 5.0; t.increment = 10.0; t.main_time_limit = 6.0; t.original_num_periods = 0; t.num_stones_per_period = 0; t.per_period_time = 0.0; t.main_time_left = 5.0; t.in_overtime = false; t.num_periods_left_including_current = 0; t.num_stones_left_in_period = 0; t.time_left_in_period = 0.0; });
  case(&mut out, &boards, "Basic 1h fischer time controls, -1s left, 10s increment", 1.0, |t| { t.original_main_time = 3600.0; t.increment = 10.0; t.original_num_periods = 0; t.num_stones_per_period = 0; t.per_period_time = 0.0; t.main_time_left = -1.0; t.in_overtime = false; t.num_periods_left_including_current = 0; t.num_stones_left_in_period = 0; t.time_left_in_period = 0.0; });
  case(&mut out, &boards, "Basic 10s fischer time controls, -1s left, 10s increment, 10s main time limit", 1.0, |t| { t.original_main_time = 10.0; t.increment = 10.0; t.main_time_limit = 10.0; t.original_num_periods = 0; t.num_stones_per_period = 0; t.per_period_time = 0.0; t.main_time_left = -1.0; t.in_overtime = false; t.num_periods_left_including_current = 0; t.num_stones_left_in_period = 0; t.time_left_in_period = 0.0; });
  case(&mut out, &boards, "Basic 1h byo yomi time controls, all time left, 1 period of 30s", 1.0, |t| { t.original_main_time = 3600.0; t.increment = 0.0; t.original_num_periods = 1; t.num_stones_per_period = 1; t.per_period_time = 30.0; t.main_time_left = 3600.0; t.in_overtime = false; t.num_periods_left_including_current = 1; t.num_stones_left_in_period = 1; t.time_left_in_period = 0.0; });
  case(&mut out, &boards, "Basic 1h byo yomi time controls, all time left, 3 periods of 30s", 1.0, |t| { t.original_main_time = 3600.0; t.increment = 0.0; t.original_num_periods = 3; t.num_stones_per_period = 1; t.per_period_time = 30.0; t.main_time_left = 3600.0; t.in_overtime = false; t.num_periods_left_including_current = 3; t.num_stones_left_in_period = 1; t.time_left_in_period = 0.0; });
  case(&mut out, &boards, "Basic 1h byo yomi time controls, all time left, 5 periods of 30s", 1.0, |t| { t.original_main_time = 3600.0; t.increment = 0.0; t.original_num_periods = 5; t.num_stones_per_period = 1; t.per_period_time = 30.0; t.main_time_left = 3600.0; t.in_overtime = false; t.num_periods_left_including_current = 5; t.num_stones_left_in_period = 1; t.time_left_in_period = 0.0; });
  case(&mut out, &boards, "Basic 1h byo yomi time controls, all time left, 6 periods of 30s", 1.0, |t| { t.original_main_time = 3600.0; t.increment = 0.0; t.original_num_periods = 6; t.num_stones_per_period = 1; t.per_period_time = 30.0; t.main_time_left = 3600.0; t.in_overtime = false; t.num_periods_left_including_current = 6; t.num_stones_left_in_period = 1; t.time_left_in_period = 0.0; });
  case(&mut out, &boards, "Basic 1h byo yomi time controls, all time left, 7 periods of 30s", 1.0, |t| { t.original_main_time = 3600.0; t.increment = 0.0; t.original_num_periods = 7; t.num_stones_per_period = 1; t.per_period_time = 30.0; t.main_time_left = 3600.0; t.in_overtime = false; t.num_periods_left_including_current = 7; t.num_stones_left_in_period = 1; t.time_left_in_period = 0.0; });
  case(&mut out, &boards, "Basic 1h byo yomi time controls, all time left, 3 moves canadian in 30s", 1.0, |t| { t.original_main_time = 3600.0; t.increment = 0.0; t.original_num_periods = 1; t.num_stones_per_period = 3; t.per_period_time = 30.0; t.main_time_left = 3600.0; t.in_overtime = false; t.num_periods_left_including_current = 1; t.num_stones_left_in_period = 3; t.time_left_in_period = 0.0; });
  case(&mut out, &boards, "Basic 1h byo yomi time controls, no time left, 1 periods of 30s", 1.0, |t| { t.original_main_time = 3600.0; t.increment = 0.0; t.original_num_periods = 1; t.num_stones_per_period = 1; t.per_period_time = 30.0; t.main_time_left = 0.0; t.in_overtime = false; t.num_periods_left_including_current = 1; t.num_stones_left_in_period = 1; t.time_left_in_period = 0.0; });
  case(&mut out, &boards, "Basic 1h byo yomi time controls, no time left, 5 periods of 30s", 1.0, |t| { t.original_main_time = 3600.0; t.increment = 0.0; t.original_num_periods = 5; t.num_stones_per_period = 1; t.per_period_time = 30.0; t.main_time_left = 0.0; t.in_overtime = false; t.num_periods_left_including_current = 5; t.num_stones_left_in_period = 1; t.time_left_in_period = 0.0; });
  case(&mut out, &boards, "Basic 1h byo yomi time controls, no time left, 6 periods of 30s", 1.0, |t| { t.original_main_time = 3600.0; t.increment = 0.0; t.original_num_periods = 6; t.num_stones_per_period = 1; t.per_period_time = 30.0; t.main_time_left = 0.0; t.in_overtime = false; t.num_periods_left_including_current = 6; t.num_stones_left_in_period = 1; t.time_left_in_period = 0.0; });
  case(&mut out, &boards, "Basic 1h byo yomi time controls, no time left, 7 periods of 30s", 1.0, |t| { t.original_main_time = 3600.0; t.increment = 0.0; t.original_num_periods = 7; t.num_stones_per_period = 1; t.per_period_time = 30.0; t.main_time_left = 0.0; t.in_overtime = false; t.num_periods_left_including_current = 7; t.num_stones_left_in_period = 1; t.time_left_in_period = 0.0; });
  case(&mut out, &boards, "Basic 1h byo yomi time controls, no time left, 3 moves canadian in 30s", 1.0, |t| { t.original_main_time = 3600.0; t.increment = 0.0; t.original_num_periods = 1; t.num_stones_per_period = 3; t.per_period_time = 30.0; t.main_time_left = 0.0; t.in_overtime = false; t.num_periods_left_including_current = 1; t.num_stones_left_in_period = 3; t.time_left_in_period = 0.0; });
  case(&mut out, &boards, "Basic 1h byo yomi time controls, 15s left, 1 periods of 30s", 1.0, |t| { t.original_main_time = 3600.0; t.increment = 0.0; t.original_num_periods = 1; t.num_stones_per_period = 1; t.per_period_time = 30.0; t.main_time_left = 15.0; t.in_overtime = false; t.num_periods_left_including_current = 1; t.num_stones_left_in_period = 1; t.time_left_in_period = 0.0; });
  case(&mut out, &boards, "Basic 1h byo yomi time controls, 15s left, 2 periods of 30s", 1.0, |t| { t.original_main_time = 3600.0; t.increment = 0.0; t.original_num_periods = 2; t.num_stones_per_period = 1; t.per_period_time = 30.0; t.main_time_left = 15.0; t.in_overtime = false; t.num_periods_left_including_current = 2; t.num_stones_left_in_period = 1; t.time_left_in_period = 0.0; });
  case(&mut out, &boards, "Basic 1h byo yomi time controls, 15s left, 3 moves canadian in 30s", 1.0, |t| { t.original_main_time = 3600.0; t.increment = 0.0; t.original_num_periods = 1; t.num_stones_per_period = 3; t.per_period_time = 30.0; t.main_time_left = 15.0; t.in_overtime = false; t.num_periods_left_including_current = 1; t.num_stones_left_in_period = 3; t.time_left_in_period = 0.0; });
  case(&mut out, &boards, "Basic 1h byo yomi time controls, 45s left, 1 periods of 30s", 1.0, |t| { t.original_main_time = 3600.0; t.increment = 0.0; t.original_num_periods = 1; t.num_stones_per_period = 1; t.per_period_time = 30.0; t.main_time_left = 45.0; t.in_overtime = false; t.num_periods_left_including_current = 1; t.num_stones_left_in_period = 1; t.time_left_in_period = 0.0; });
  case(&mut out, &boards, "Basic 1h byo yomi time controls, 45s left, 2 periods of 30s", 1.0, |t| { t.original_main_time = 3600.0; t.increment = 0.0; t.original_num_periods = 2; t.num_stones_per_period = 1; t.per_period_time = 30.0; t.main_time_left = 45.0; t.in_overtime = false; t.num_periods_left_including_current = 2; t.num_stones_left_in_period = 1; t.time_left_in_period = 0.0; });
  case(&mut out, &boards, "Basic 1h byo yomi time controls, 45s left, 2 periods of 30s, max time per move 57s", 1.0, |t| { t.original_main_time = 3600.0; t.increment = 0.0; t.max_time_per_move = 57.0; t.original_num_periods = 2; t.num_stones_per_period = 1; t.per_period_time = 30.0; t.main_time_left = 45.0; t.in_overtime = false; t.num_periods_left_including_current = 2; t.num_stones_left_in_period = 1; t.time_left_in_period = 0.0; });
  case(&mut out, &boards, "Basic 1h byo yomi time controls, 45s left, 3 moves canadian in 30s", 1.0, |t| { t.original_main_time = 3600.0; t.increment = 0.0; t.original_num_periods = 1; t.num_stones_per_period = 3; t.per_period_time = 30.0; t.main_time_left = 45.0; t.in_overtime = false; t.num_periods_left_including_current = 1; t.num_stones_left_in_period = 3; t.time_left_in_period = 0.0; });
  case(&mut out, &boards, "Basic 1h byo yomi time controls, 61s left, 1 periods of 30s", 1.0, |t| { t.original_main_time = 3600.0; t.increment = 0.0; t.original_num_periods = 1; t.num_stones_per_period = 1; t.per_period_time = 30.0; t.main_time_left = 61.0; t.in_overtime = false; t.num_periods_left_including_current = 1; t.num_stones_left_in_period = 1; t.time_left_in_period = 0.0; });
  case(&mut out, &boards, "Basic 1h byo yomi time controls, 61s left, 2 periods of 30s", 1.0, |t| { t.original_main_time = 3600.0; t.increment = 0.0; t.original_num_periods = 2; t.num_stones_per_period = 1; t.per_period_time = 30.0; t.main_time_left = 61.0; t.in_overtime = false; t.num_periods_left_including_current = 2; t.num_stones_left_in_period = 1; t.time_left_in_period = 0.0; });
  case(&mut out, &boards, "Basic 1h byo yomi time controls, 61s left, 3 moves canadian in 30s", 1.0, |t| { t.original_main_time = 3600.0; t.increment = 0.0; t.original_num_periods = 1; t.num_stones_per_period = 3; t.per_period_time = 30.0; t.main_time_left = 61.0; t.in_overtime = false; t.num_periods_left_including_current = 1; t.num_stones_left_in_period = 3; t.time_left_in_period = 0.0; });
  case(&mut out, &boards, "Basic 1h byo yomi time controls, 70s left, 1 periods of 30s", 1.0, |t| { t.original_main_time = 3600.0; t.increment = 0.0; t.original_num_periods = 1; t.num_stones_per_period = 1; t.per_period_time = 30.0; t.main_time_left = 70.0; t.in_overtime = false; t.num_periods_left_including_current = 1; t.num_stones_left_in_period = 1; t.time_left_in_period = 0.0; });
  case(&mut out, &boards, "Basic 1h byo yomi time controls, 70s left, 2 periods of 30s", 1.0, |t| { t.original_main_time = 3600.0; t.increment = 0.0; t.original_num_periods = 2; t.num_stones_per_period = 1; t.per_period_time = 30.0; t.main_time_left = 70.0; t.in_overtime = false; t.num_periods_left_including_current = 2; t.num_stones_left_in_period = 1; t.time_left_in_period = 0.0; });
  case(&mut out, &boards, "Basic 1h byo yomi time controls, 70s left, 3 moves canadian in 30s", 1.0, |t| { t.original_main_time = 3600.0; t.increment = 0.0; t.original_num_periods = 1; t.num_stones_per_period = 3; t.per_period_time = 30.0; t.main_time_left = 70.0; t.in_overtime = false; t.num_periods_left_including_current = 1; t.num_stones_left_in_period = 3; t.time_left_in_period = 0.0; });
  case(&mut out, &boards, "Basic 1h byo yomi time controls, 1 periods of 30s, just entered overtime", 1.0, |t| { t.original_main_time = 3600.0; t.increment = 0.0; t.original_num_periods = 1; t.num_stones_per_period = 1; t.per_period_time = 30.0; t.main_time_left = 0.0; t.in_overtime = true; t.num_periods_left_including_current = 1; t.num_stones_left_in_period = 1; t.time_left_in_period = 30.0; });
  case(&mut out, &boards, "Basic 1h byo yomi time controls, 3 periods of 30s, just entered overtime", 1.0, |t| { t.original_main_time = 3600.0; t.increment = 0.0; t.original_num_periods = 3; t.num_stones_per_period = 1; t.per_period_time = 30.0; t.main_time_left = 0.0; t.in_overtime = true; t.num_periods_left_including_current = 3; t.num_stones_left_in_period = 1; t.time_left_in_period = 30.0; });
  case(&mut out, &boards, "Basic 1h byo yomi time controls, 5 periods of 30s, just entered overtime", 1.0, |t| { t.original_main_time = 3600.0; t.increment = 0.0; t.original_num_periods = 5; t.num_stones_per_period = 1; t.per_period_time = 30.0; t.main_time_left = 0.0; t.in_overtime = true; t.num_periods_left_including_current = 5; t.num_stones_left_in_period = 1; t.time_left_in_period = 30.0; });
  case(&mut out, &boards, "Basic 1h byo yomi time controls, 6 periods of 30s, just entered overtime", 1.0, |t| { t.original_main_time = 3600.0; t.increment = 0.0; t.original_num_periods = 6; t.num_stones_per_period = 1; t.per_period_time = 30.0; t.main_time_left = 0.0; t.in_overtime = true; t.num_periods_left_including_current = 6; t.num_stones_left_in_period = 1; t.time_left_in_period = 30.0; });
  case(&mut out, &boards, "Basic 1h byo yomi time controls, 7 periods of 30s, just entered overtime", 1.0, |t| { t.original_main_time = 3600.0; t.increment = 0.0; t.original_num_periods = 7; t.num_stones_per_period = 1; t.per_period_time = 30.0; t.main_time_left = 0.0; t.in_overtime = true; t.num_periods_left_including_current = 7; t.num_stones_left_in_period = 1; t.time_left_in_period = 30.0; });
  case(&mut out, &boards, "Basic 1h byo yomi time controls, 3 moves canadian in 30s, just entered overtime", 1.0, |t| { t.original_main_time = 3600.0; t.increment = 0.0; t.original_num_periods = 1; t.num_stones_per_period = 3; t.per_period_time = 30.0; t.main_time_left = 0.0; t.in_overtime = true; t.num_periods_left_including_current = 1; t.num_stones_left_in_period = 3; t.time_left_in_period = 30.0; });
  case(&mut out, &boards, "Basic 1h byo yomi time controls, 1 periods of 30s, entered overtime 15s used", 1.0, |t| { t.original_main_time = 3600.0; t.increment = 0.0; t.original_num_periods = 1; t.num_stones_per_period = 1; t.per_period_time = 30.0; t.main_time_left = 0.0; t.in_overtime = true; t.num_periods_left_including_current = 1; t.num_stones_left_in_period = 1; t.time_left_in_period = 15.0; });
  case(&mut out, &boards, "Basic 1h byo yomi time controls, 5 periods of 30s, entered overtime 15s used", 1.0, |t| { t.original_main_time = 3600.0; t.increment = 0.0; t.original_num_periods = 5; t.num_stones_per_period = 1; t.per_period_time = 30.0; t.main_time_left = 0.0; t.in_overtime = true; t.num_periods_left_including_current = 5; t.num_stones_left_in_period = 1; t.time_left_in_period = 15.0; });
  case(&mut out, &boards, "Basic 1h byo yomi time controls, 6 periods of 30s, entered overtime 15s used", 1.0, |t| { t.original_main_time = 3600.0; t.increment = 0.0; t.original_num_periods = 6; t.num_stones_per_period = 1; t.per_period_time = 30.0; t.main_time_left = 0.0; t.in_overtime = true; t.num_periods_left_including_current = 6; t.num_stones_left_in_period = 1; t.time_left_in_period = 15.0; });
  case(&mut out, &boards, "Basic 1h byo yomi time controls, 7 periods of 30s, entered overtime 15s used", 1.0, |t| { t.original_main_time = 3600.0; t.increment = 0.0; t.original_num_periods = 7; t.num_stones_per_period = 1; t.per_period_time = 30.0; t.main_time_left = 0.0; t.in_overtime = true; t.num_periods_left_including_current = 7; t.num_stones_left_in_period = 1; t.time_left_in_period = 15.0; });
  case(&mut out, &boards, "Basic 1h byo yomi time controls, 3 moves canadian in 30s, entered overtime 15s used", 1.0, |t| { t.original_main_time = 3600.0; t.increment = 0.0; t.original_num_periods = 1; t.num_stones_per_period = 3; t.per_period_time = 30.0; t.main_time_left = 0.0; t.in_overtime = true; t.num_periods_left_including_current = 1; t.num_stones_left_in_period = 3; t.time_left_in_period = 15.0; });
  case(&mut out, &boards, "Basic 1h byo yomi time controls, 3 moves canadian in 30s, entered overtime 15s used, 2 moves left", 1.0, |t| { t.original_main_time = 3600.0; t.increment = 0.0; t.original_num_periods = 1; t.num_stones_per_period = 3; t.per_period_time = 30.0; t.main_time_left = 0.0; t.in_overtime = true; t.num_periods_left_including_current = 1; t.num_stones_left_in_period = 2; t.time_left_in_period = 15.0; });
  case(&mut out, &boards, "Basic 1h byo yomi time controls, 3 moves canadian in 30s, entered overtime 15s used, 1 moves left", 1.0, |t| { t.original_main_time = 3600.0; t.increment = 0.0; t.original_num_periods = 1; t.num_stones_per_period = 3; t.per_period_time = 30.0; t.main_time_left = 0.0; t.in_overtime = true; t.num_periods_left_including_current = 1; t.num_stones_left_in_period = 1; t.time_left_in_period = 15.0; });

  let expected = fixture("oracle", "timecontrols_testtime.txt");
  assert_eq!(out.matches("min rec max").count(), 53 * 4);
  for _ in 0..53 {
    cases.ran();
  }
  compare_lines("timecontrols_testtime.txt", &out, &expected);
}
}

// ===================================================================== oracle sequence (timecontrols.cpp)

fn parse_bits(s: &str) -> f64 {
  f64::from_bits(u64::from_str_radix(s, 16).unwrap_or_else(|e| panic!("bad hex {:?}: {}", s, e)))
}

/// `fromFields`: the 12 field tokens of a cases line (as printed by `fields`).
fn from_fields(tok: &[&str]) -> TimeControls {
  let mut t = TimeControls::new();
  t.original_main_time = parse_bits(tok[0]);
  t.increment = parse_bits(tok[1]);
  t.main_time_limit = parse_bits(tok[2]);
  t.max_time_per_move = parse_bits(tok[3]);
  t.original_num_periods = tok[4].parse().unwrap();
  t.num_stones_per_period = tok[5].parse().unwrap();
  t.per_period_time = parse_bits(tok[6]);
  t.main_time_left = parse_bits(tok[7]);
  t.in_overtime = tok[8] == "1";
  t.num_periods_left_including_current = tok[9].parse().unwrap();
  t.num_stones_left_in_period = tok[10].parse().unwrap();
  t.time_left_in_period = parse_bits(tok[11]);
  t
}

wasm_test! {
fn fixed_sequence_and_sweep() {
  init();
  let mut cases = Cases::new("timecontrols::fixed_sequence_and_sweep");
  let mut res = String::new();

  {
    let (b9, h9) = boxed_history(make_board(9, 9, 0));
    let (b19, h19) = boxed_history(make_board(19, 19, 150));

    writeln!(
      res,
      "constants {} {} {} {}",
      bits(TimeControls::UNLIMITED_TIME_THRESHOLD),
      bits(TimeControls::MAX_USER_INPUT_TIME),
      bits(TimeControls::UNLIMITED_TIME_DEFAULT),
      bits(TimeControls::UNLIMITED_TIME_DEFAULT_LARGE)
    )
    .unwrap();

    let mut named: Vec<(String, TimeControls)> = Vec::new();
    named.push(("default".to_string(), TimeControls::new()));
    named.push(("absolute 600".to_string(), TimeControls::absolute_time(600.0)));
    named.push(("absolute -3".to_string(), TimeControls::absolute_time(-3.0)));
    named.push(("absolute 1e21".to_string(), TimeControls::absolute_time(1e21)));
    named.push(("fischer 600 10".to_string(), TimeControls::fischer_time(600.0, 10.0)));
    named.push(("fischer 0 5".to_string(), TimeControls::fischer_time(0.0, 5.0)));
    named.push(("fischer 1e30 5".to_string(), TimeControls::fischer_time(1e30, 5.0)));
    named.push(("capped 600 10 700 30".to_string(), TimeControls::fischer_capped_time(600.0, 10.0, 700.0, 30.0).unwrap()));
    named.push(("capped 600 10 600 1e30".to_string(), TimeControls::fischer_capped_time(600.0, 10.0, 600.0, 1e30).unwrap()));
    named.push(("capped 600 0 1e25 1e25".to_string(), TimeControls::fischer_capped_time(600.0, 0.0, 1e25, 1e25).unwrap()));
    named.push(("byo 600 30 5 1".to_string(), TimeControls::canadian_or_byo_yomi_time(600.0, 30.0, 5, 1)));
    named.push(("byo 0 30 1 1".to_string(), TimeControls::canadian_or_byo_yomi_time(0.0, 30.0, 1, 1)));
    named.push(("canadian 3600 300 1 25".to_string(), TimeControls::canadian_or_byo_yomi_time(3600.0, 300.0, 1, 25)));
    named.push(("byo 1e30 30 3 1".to_string(), TimeControls::canadian_or_byo_yomi_time(1e30, 30.0, 3, 1)));
    {
      let mut t = TimeControls::canadian_or_byo_yomi_time(10.0, 30.0, 5, 1);
      t.main_time_left = 0.0;
      t.in_overtime = true;
      t.num_periods_left_including_current = 3;
      t.num_stones_left_in_period = 1;
      t.time_left_in_period = 1e30;
      named.push(("byo overtime unlimited period".to_string(), t));
    }
    {
      let mut t = TimeControls::new();
      t.in_overtime = true;
      t.time_left_in_period = 1e21;
      t.main_time_left = 5.0;
      named.push(("unlimited via overtime period".to_string(), t));
    }
    {
      let mut t = TimeControls::new();
      t.max_time_per_move = 10.0;
      named.push(("default with maxTimePerMove".to_string(), t));
    }
    {
      let mut t = TimeControls::new();
      t.main_time_left = 1e20;
      named.push(("mainTimeLeft exactly threshold".to_string(), t));
    }
    {
      let mut t = TimeControls::new();
      t.main_time_left = 1.0000001e20;
      t.max_time_per_move = 1.0000001e20;
      named.push(("just over threshold".to_string(), t));
    }
    for i in 0..named.len() {
      writeln!(res, "named {} fields {}", q(&named[i].0), fields(&named[i].1)).unwrap();
      writeln!(res, "named {} b9 lag1 {}", q(&named[i].0), evaluate(&named[i].1, &b9, &h9, 1.0)).unwrap();
      writeln!(res, "named {} b19 lag0 {}", q(&named[i].0), evaluate(&named[i].1, &b19, &h19, 0.0)).unwrap();
      writeln!(res, "named {} b19 lag7 {}", q(&named[i].0), evaluate(&named[i].1, &b19, &h19, 7.0)).unwrap();
      cases.ran();
    }

    // fischerCappedTime error
    let args: [[f64; 4]; 4] = [[600.0, 10.0, 599.0, 30.0], [600.0, 10.0, 600.0, 30.0], [0.0, 0.0, -1.0, 0.0], [5.0, 1.0, 1e40, 1e40]];
    for a in args.iter() {
      match TimeControls::fischer_capped_time(a[0], a[1], a[2], a[3]) {
        Ok(t) => writeln!(res, "capped {} {} {} {} ok {}", bits(a[0]), bits(a[1]), bits(a[2]), bits(a[3]), fields(&t)).unwrap(),
        Err(e) => {
          assert!(matches!(e, KataError::StringError(_)));
          writeln!(res, "capped {} {} {} {} throws {}", bits(a[0]), bits(a[1]), bits(a[2]), bits(a[3]), q(e.what())).unwrap()
        }
      }
      cases.ran();
    }
  }

  // sweep
  let cases_text = fixture("oracle", "timecontrols_cases.txt");
  let mut n = 0;
  for (i, line) in cases_text.lines().enumerate() {
    let tok: Vec<&str> = line.split(' ').collect();
    assert_eq!(tok.len(), 4 + 12, "case line {}", i);
    let xs: i32 = tok[0].parse().unwrap();
    let ys: i32 = tok[1].parse().unwrap();
    let nstones: i32 = tok[2].parse().unwrap();
    let lag = parse_bits(tok[3]);
    let tc = from_fields(&tok[4..]);
    assert_eq!(fields(&tc), tok[4..].join(" "), "case {} fields round trip", i);

    let (board, hist) = boxed_history(make_board(xs, ys, nstones));
    writeln!(res, "case {} {}", i, evaluate(&tc, &board, &hist, lag)).unwrap();
    n += 1;
    cases.ran();
  }
  assert_eq!(n, 4000);

  let expected = fixture("oracle", "timecontrols.txt");
  compare_lines("timecontrols.txt", &res, &expected);
}
}

// ===================================================================== core/timer.rs

thread_local! {
  static FAKE_NOW: std::cell::Cell<f64> = const { std::cell::Cell::new(0.0) };
}

fn fake_clock() -> f64 {
  FAKE_NOW.with(|c| c.get())
}

wasm_test! {
fn clock_timer() {
  use katago_core::core::timer::ClockTimer;
  use katago_core::core::timer::set_clock;
  // The clock is process-wide and set once: every test of this binary that needs it installs the same fn.
  set_clock(fake_clock);
  FAKE_NOW.with(|c| c.set(100.0));
  let mut timer = ClockTimer::new();
  assert_eq!(timer.get_seconds(), 0.0);
  FAKE_NOW.with(|c| c.set(100.5));
  assert_eq!(timer.get_seconds(), 0.5);
  FAKE_NOW.with(|c| c.set(103.25));
  assert_eq!(timer.get_seconds(), 3.25);
  assert_eq!(ClockTimer::get_precision_system_time(), 103250000000);
  timer.reset();
  assert_eq!(timer.get_seconds(), 0.0);
  FAKE_NOW.with(|c| c.set(104.0));
  assert_eq!(timer.get_seconds(), 0.75);
}
}
