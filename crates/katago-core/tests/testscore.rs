//! Replays upstream cpp/tests/testscore.cpp @ v1.16.5 (Tests::runScoreTests). It prints to cout
//! during `katago runoutputtests`, so the oracle is its part of upstream
//! cpp/tests/results/runOutputTests.txt
//! (fixtures/upstream-v1.18.2/unit/runoutputtests/score-and-utility-tests.expected0.txt, extracted
//! by tools/extract_runoutputtests.py); the boards are the testscore.cpp raw blocks
//! (fixtures/upstream-v1.18.2/unit/testscore/, inputs only).
//! ScoreValue uses atan/tan (port::math::{atan, tan}: platform libm, <= 1 ulp from glibc on wasm,
//! DECISIONS.md D13); the printed precision (6 significant digits, "%.3f") hides that.

// Test code keeps the shapes of upstream's tests and oracle programs (PORTING.md §11b).
#![allow(clippy::needless_range_loop)]

#[macro_use]
mod common;

use common::Cases;
use common::endl;
use common::fixture;
use common::init;
use katago_core::core::test::TestCommon;
use katago_core::game::board::*;
use katago_core::game::boardhistory::BoardHistory;
use katago_core::game::rules::Rules;
use katago_core::neuralnet::nninputs::ScoreValue;
use katago_core::port::cpp::cpp_max;
use katago_core::port::fmt::cpp_ostream_f64;
use katago_core::strprintf;

const F: &str = "testscore";

/// `ostream << double` (default precision 6); floats are widened first.
fn d(x: f64) -> String {
  cpp_ostream_f64(x)
}

/// upstream: the `printScoreStats` lambda in Tests::runScoreTests
fn print_score_stats(out: &mut String, board: &Board, hist: &BoardHistory) {
  *out += &format!("Black self komi wins/draw=0.5: {}\n", d(hist.current_self_komi(P_BLACK, 0.5) as f64));
  *out += &format!("White self komi wins/draw=0.5: {}\n", d(hist.current_self_komi(P_WHITE, 0.5) as f64));
  *out += &format!("Black self komi wins/draw=0.25: {}\n", d(hist.current_self_komi(P_BLACK, 0.25) as f64));
  *out += &format!("White self komi wins/draw=0.25: {}\n", d(hist.current_self_komi(P_WHITE, 0.25) as f64));
  *out += &format!("Black self komi wins/draw=0.75: {}\n", d(hist.current_self_komi(P_BLACK, 0.75) as f64));
  *out += &format!("White self komi wins/draw=0.75: {}\n", d(hist.current_self_komi(P_WHITE, 0.75) as f64));

  *out += &format!("Winner: {}\n", PlayerIO::color_to_char(hist.winner));
  let score: f64 = hist.final_white_minus_black_score as f64;
  *out += &format!("Final score: {}\n", d(score));

  let draw_equivs_to_try: [f64; 4] = [0.5, 0.3, 0.7, 1.0];
  for i in 0..4 {
    let draw_equiv = draw_equivs_to_try[i];
    let s = strprintf!("%.1f", draw_equiv);
    let score_adjusted = ScoreValue::white_score_draw_adjust(score, draw_equiv, hist);
    let stdev = cpp_max(0.0, ScoreValue::white_score_mean_sq_of_score_gridded(score, draw_equiv) - score_adjusted * score_adjusted).sqrt();
    let sqrt_board_area = board.sqrt_board_area();
    let expected_score_value = ScoreValue::expected_white_score_value(score_adjusted, stdev, 0.0, 2.0, sqrt_board_area);
    *out += &format!("WL Wins wins/draw={}: {}\n", s, d(ScoreValue::white_wins_of_winner(hist.winner, draw_equiv)));
    *out += &format!("Score wins/draw={}: {}\n", s, d(score_adjusted));
    *out += &format!("Score Stdev wins/draw={}: {}\n", s, d(stdev));
    *out += &format!(
      "Score Util Smooth  wins/draw={}: {}\n",
      s,
      d(ScoreValue::white_score_value_of_score_smooth(score, 0.0, 2.0, draw_equiv, sqrt_board_area, hist))
    );
    *out += &format!(
      "Score Util SmootND wins/draw={}: {}\n",
      s,
      d(ScoreValue::white_score_value_of_score_smooth_no_draw_adjust(score, 0.0, 2.0, sqrt_board_area))
    );
    *out += &format!("Score Util Gridded wins/draw={}: {}\n", s, d(expected_score_value));
    *out += &format!(
      "Score Util GridInv wins/draw={}: {}\n",
      s,
      d(ScoreValue::approx_white_score_of_score_value_smooth(expected_score_value, 0.0, 2.0, sqrt_board_area))
    );
  }
}

wasm_test! {
/// upstream: Tests::runScoreTests (testscore.cpp). `cout` is `cout_buf`; `out` is the
/// ostringstream that each block prints after its name (and never clears, as upstream).
fn run_score_tests() {
  init();
  ScoreValue::init_tables();
  let mut cases = Cases::new("testscore::run_score_tests");
  let mut cout = String::new();
  cout += "Running score and utility tests\n";
  let mut out = String::new();

  // (name, slug, x, y, komi override)
  let blocks: [(&str, &str, i32, i32, Option<f32>); 4] = [
    ("On-board even 9x9, komi 7.5", "on-board-even-9x9-komi-7-5", 9, 9, None),
    ("On-board even 9x9, komi 7", "on-board-even-9x9-komi-7", 9, 9, Some(7.0)),
    ("On-board black ahead 7 9x9, komi 7", "on-board-black-ahead-7-9x9-komi-7", 9, 9, Some(7.0)),
    ("On-board even 5x5, komi 7", "on-board-even-5x5-komi-7", 5, 5, Some(7.0)),
  ];
  for (name, slug, x, y, komi) in blocks.iter() {
    let board = Board::parse_board(*x, *y, &fixture(F, &format!("{}.input0.txt", slug)), '\n').unwrap();
    let mut rules = Rules::get_tromp_taylorish();
    if let Some(k) = komi {
      rules.komi = *k;
    }
    let mut hist = BoardHistory::new_full(&board, P_BLACK, &rules, 0);
    hist.end_and_score_game_now(&board);
    print_score_stats(&mut out, &board, &hist);
    cout += name;
    endl(&mut cout);
    cout += &out;
    endl(&mut cout);
    endl(&mut cout);
    cases.ran();
  }
  {
    let name = "Score value tables";
    cout += name;
    endl(&mut cout);
    let mut center = 0;
    while center <= 5 {
      for scale in 1..=2 {
        for b in 0..5 {
          let x_sizes: [i32; 5] = [9, 13, 13, 13, 19];
          let y_sizes: [i32; 5] = [9, 9, 13, 19, 19];
          let board = Board::new_x_y(x_sizes[b], y_sizes[b]).unwrap();
          cout += &format!("center {} scale {} x {} y {}\n", center, scale, x_sizes[b], y_sizes[b]);
          for stdev in 0..=5 {
            let mut dd: f64 = -8.0;
            while dd <= 8.0 {
              let score_value =
                ScoreValue::expected_white_score_value(dd, stdev as f64, center as f64, scale as f64, board.sqrt_board_area());
              cout += &strprintf!("%.3f ", score_value);
              dd += 0.5;
            }
            endl(&mut cout);
          }
          endl(&mut cout);
          cases.ran();
        }
        endl(&mut cout);
      }
      center += 5;
    }
  }
  let expected = fixture("runoutputtests", "score-and-utility-tests.expected0.txt");
  TestCommon::expect("Score and utility tests", &cout, &expected);
}
}
