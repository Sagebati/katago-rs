//! Replays upstream cpp/tests/testrules.cpp @ v1.16.5 (Tests::runRulesTests) against
//! fixtures/upstream-v1.16.5/unit/testrules/. Each upstream `{ const char* name = ...; }` block is
//! one #[test] (the blocks share nothing but `out`, which every `expect` clears), in upstream order.
//! Board inputs are copied verbatim from the C++ (`R"%%(...)%%"` → `r"..."`); expected blocks are
//! read from the fixtures.
//!
//! Covered elsewhere: "Test some roundtripping of rules strings" and "Rules parsing bug" (tests/rules.rs).
//! Not ported: "Test basic game" needs CompactSgf (dataio/sgf, not M1): TODO(port) at its position.

// Lints the verbatim transcription triggers (`testAssert(hist.isGameFinished == false)`, indexed
// loops), as in lib.rs (PORTING.md §11b); test crates do not inherit the lib's list.
#![allow(clippy::bool_comparison, clippy::needless_range_loop, clippy::needless_late_init, clippy::manual_memcpy)]

mod common;

use std::fmt::Write as _;

use common::Cases;
use common::fixture;
use common::init;
use katago_core::core::hash::Hash128;
use katago_core::core::rand::Rand;
use katago_core::core::test::TestCommon;
use katago_core::game::board::*;
use katago_core::game::boardhistory::*;
use katago_core::game::rules::Rules;
use katago_core::port::fmt::cpp_ostream_f64;
use katago_core::testAssert;

macro_rules! wasm_test {
  ($(#[$m:meta])* fn $name:ident() $body:block) => {
    $(#[$m])*
    #[cfg_attr(not(target_arch = "wasm32"), test)]
    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    fn $name() $body
  };
}

// ===================================================================== helpers (testrules.cpp statics and runRulesTests lambdas)

const F: &str = "testrules";

/// `expect(name,out,expected)` with the expected block read from fixtures/.../testrules/<file>.
fn expect(name: &str, out: &mut String, expected_file: &str) {
  let expected = fixture(F, expected_file);
  TestCommon::expect_string(name, out, &expected);
}

/// `Board::parseBoard(x,y,s)` (default line delimiter '\n').
fn parse(x: i32, y: i32, s: &str) -> Board {
  Board::parse_board(x, y, s, '\n').unwrap()
}

/// `out << board << endl`
fn out_board(out: &mut String, board: &Board) {
  writeln!(out, "{}", board).unwrap();
}

/// `out << (float)x` (ostream default precision).
fn fs(x: f32) -> String {
  cpp_ostream_f64(x as f64)
}

/// `out << (double)x` (ostream default precision).
fn ds(x: f64) -> String {
  cpp_ostream_f64(x)
}

/// upstream: checkKoHashConsistency (testrules.cpp) static
fn check_ko_hash_consistency(hist: &BoardHistory, board: &Board, next_pla: Player) {
  testAssert!(!hist.ko_hash_history.is_empty());
  let mut expected = board.pos_hash;
  if hist.encore_phase > 0 {
    expected ^= Board::statics().ZOBRIST_PLAYER_HASH[next_pla as usize];
    for y in 0..board.y_size {
      for x in 0..board.x_size {
        let loc = Location::get_loc(x, y, board.x_size);
        if hist.ko_recap_blocked[loc as usize] {
          expected ^= Board::statics().ZOBRIST_KO_MARK_HASH[loc as usize][P_BLACK as usize]
            ^ Board::statics().ZOBRIST_KO_MARK_HASH[loc as usize][P_WHITE as usize];
        }
      }
    }
  } else if hist.rules.ko_rule == Rules::KO_SITUATIONAL || hist.rules.ko_rule == Rules::KO_SIMPLE {
    expected ^= Board::statics().ZOBRIST_PLAYER_HASH[next_pla as usize];
  }
  testAssert!(expected == hist.ko_hash_history[hist.ko_hash_history.len() - 1]);
}

/// upstream: makeMoveAssertLegal(hist,board,loc,pla,line,preventEncore,table) (testrules.cpp) static
/// `throw StringError(..)` -> panic (nothing catches it upstream).
fn make_move_assert_legal_pe_t(
  hist: &mut BoardHistory,
  board: &mut Board,
  loc: Loc,
  pla: Player,
  line: u32,
  prevent_encore: bool,
  table: Option<&KoHashTable>,
) {
  let phase_would_end = hist.pass_would_end_phase(board, pla);
  let old_phase = hist.encore_phase;

  if !hist.is_legal(board, loc, pla) {
    panic!("Illegal move on line {}", line);
  }
  if !hist.is_legal_tolerant(board, loc, pla) {
    panic!("Tolerant illegal move on line {}", line);
  }
  hist.make_board_move_assume_legal(board, loc, pla, table, prevent_encore);
  check_ko_hash_consistency(hist, board, get_opp(pla));

  if loc == Board::PASS_LOC {
    let new_phase = hist.encore_phase;
    if (phase_would_end && !prevent_encore) != (new_phase != old_phase || hist.is_game_finished) {
      panic!("hist.passWouldEndPhase returned different answer than what actually happened after a pass");
    }
  }
}

/// upstream: makeMoveAssertLegal(hist,board,loc,pla,line,preventEncore) (testrules.cpp) static
fn make_move_assert_legal_pe(hist: &mut BoardHistory, board: &mut Board, loc: Loc, pla: Player, line: u32, prevent_encore: bool) {
  make_move_assert_legal_pe_t(hist, board, loc, pla, line, prevent_encore, None);
}

/// upstream: makeMoveAssertLegal(hist,board,loc,pla,line,table) (testrules.cpp) static
fn make_move_assert_legal_t(hist: &mut BoardHistory, board: &mut Board, loc: Loc, pla: Player, line: u32, table: Option<&KoHashTable>) {
  make_move_assert_legal_pe_t(hist, board, loc, pla, line, false, table);
}

/// upstream: makeMoveAssertLegal(hist,board,loc,pla,line) (testrules.cpp) static
fn make_move_assert_legal(hist: &mut BoardHistory, board: &mut Board, loc: Loc, pla: Player, line: u32) {
  make_move_assert_legal_pe_t(hist, board, loc, pla, line, false, None);
}

/// upstream: finalScoreIfGameEndedNow (testrules.cpp) static
fn final_score_if_game_ended_now(base_hist: &BoardHistory, base_board: &Board) -> f64 {
  let mut pla: Player = P_BLACK;
  let mut board = base_board.clone();
  let mut hist = base_hist.clone();
  if !hist.move_history.is_empty() {
    pla = get_opp(hist.move_history[hist.move_history.len() - 1].pla);
  }
  while !hist.is_game_finished {
    hist.make_board_move_assume_legal(&mut board, Board::PASS_LOC, pla, None, false);
    pla = get_opp(pla);
  }

  let score: f64 = hist.final_white_minus_black_score as f64;

  hist.end_and_score_game_now(&board);
  testAssert!(hist.final_white_minus_black_score as f64 == score);

  let mut hist2 = base_hist.clone();
  hist2.end_and_score_game_now(base_board);
  testAssert!(hist2.final_white_minus_black_score as f64 == score);

  score
}

/// upstream: the `printIllegalMoves` lambda in Tests::runRulesTests
fn print_illegal_moves(o: &mut String, board: &Board, hist: &BoardHistory, pla: Player) {
  for y in 0..board.y_size {
    for x in 0..board.x_size {
      let loc = Location::get_loc(x, y, board.x_size);
      if board.colors[loc as usize] == C_EMPTY
        && !board.is_illegal_suicide(loc, pla, hist.rules.multi_stone_suicide_legal)
        && !hist.is_legal(board, loc, pla)
      {
        writeln!(o, "Illegal: {} {}", Location::to_string_mach(loc, board.x_size), PlayerIO::color_to_char(pla)).unwrap();
      }
      if hist.ko_recap_blocked[loc as usize] {
        writeln!(o, "Ko-recap-blocked: {}", Location::to_string_mach(loc, board.x_size)).unwrap();
      }
    }
  }
}

/// upstream: the `printEncoreKoBlock` lambda in Tests::runRulesTests
fn print_encore_ko_block(o: &mut String, board: &Board, hist: &BoardHistory) {
  for y in 0..board.y_size {
    for x in 0..board.x_size {
      let loc = Location::get_loc(x, y, board.x_size);
      if hist.ko_recap_blocked[loc as usize] {
        writeln!(o, "Ko recap blocked at {}", Location::to_string(loc, board)).unwrap();
      }
    }
  }
}

/// upstream: the `printGameResult` lambda in Tests::runRulesTests
fn print_game_result(o: &mut String, hist: &BoardHistory) {
  if !hist.is_game_finished {
    writeln!(o, "Game is not over").unwrap();
  } else {
    writeln!(o, "Winner: {}", PlayerIO::player_to_string(hist.winner)).unwrap();
    writeln!(o, "W-B Score: {}", fs(hist.final_white_minus_black_score)).unwrap();
    writeln!(o, "isNoResult: {}", hist.is_no_result as i32).unwrap();
    writeln!(o, "isResignation: {}", hist.is_resignation as i32).unwrap();
    testAssert!(hist.is_no_result as i32 + hist.is_resignation as i32 + hist.is_scored as i32 == hist.is_game_finished as i32);
  }
}

/// `makeMoveAssertLegal(hist, board, loc, pla, __LINE__[, preventEncore][, table])`.
/// A macro so that `loc` (usually `Location::getLoc(x,y,board.x_size)`) is evaluated before
/// `board` is mutably borrowed.
macro_rules! mmal {
  ($hist:expr, $board:expr, $loc:expr, $pla:expr) => {{
    let loc_: Loc = $loc;
    make_move_assert_legal(&mut $hist, &mut $board, loc_, $pla, line!())
  }};
  ($hist:expr, $board:expr, $loc:expr, $pla:expr, pe = $pe:expr) => {{
    let loc_: Loc = $loc;
    make_move_assert_legal_pe(&mut $hist, &mut $board, loc_, $pla, line!(), $pe)
  }};
  ($hist:expr, $board:expr, $loc:expr, $pla:expr, t = $t:expr) => {{
    let loc_: Loc = $loc;
    make_move_assert_legal_t(&mut $hist, &mut $board, loc_, $pla, line!(), $t)
  }};
  ($hist:expr, $board:expr, $loc:expr, $pla:expr, pe = $pe:expr, t = $t:expr) => {{
    let loc_: Loc = $loc;
    make_move_assert_legal_pe_t(&mut $hist, &mut $board, loc_, $pla, line!(), $pe, $t)
  }};
}

/// `hist.makeBoardMoveAssumeLegal(board, loc, pla, table[, preventEncore])` with `loc` evaluated first.
macro_rules! mbmal {
  ($hist:expr, $board:expr, $loc:expr, $pla:expr, $t:expr) => {{
    let loc_: Loc = $loc;
    $hist.make_board_move_assume_legal(&mut $board, loc_, $pla, $t, false)
  }};
  ($hist:expr, $board:expr, $loc:expr, $pla:expr, $t:expr, $pe:expr) => {{
    let loc_: Loc = $loc;
    $hist.make_board_move_assume_legal(&mut $board, loc_, $pla, $t, $pe)
  }};
}

// ===================================================================== runRulesTests blocks, in upstream order

wasm_test! {
fn area_rules() {
  init();
  let mut cases = Cases::new("testrules::area_rules");
  let mut out = String::new();
  {
    let name = "Area rules";
    let mut board = parse(4, 4, r"
....
....
....
....
");
    let mut rules = Rules::new();
    rules.ko_rule = Rules::KO_POSITIONAL;
    rules.scoring_rule = Rules::SCORING_AREA;
    rules.komi = 0.5f32;
    rules.multi_stone_suicide_legal = true;
    rules.tax_rule = Rules::TAX_NONE;
    let mut hist = BoardHistory::new_full(&board, P_BLACK, &rules, 0);

    mmal!(hist, board, Location::get_loc(1, 1, board.x_size), P_BLACK);
    mmal!(hist, board, Location::get_loc(2, 2, board.x_size), P_WHITE);
    mmal!(hist, board, Location::get_loc(1, 2, board.x_size), P_BLACK);
    mmal!(hist, board, Location::get_loc(2, 1, board.x_size), P_WHITE);
    mmal!(hist, board, Location::get_loc(1, 3, board.x_size), P_BLACK);
    mmal!(hist, board, Location::get_loc(2, 3, board.x_size), P_WHITE);
    mmal!(hist, board, Location::get_loc(1, 0, board.x_size), P_BLACK);
    mmal!(hist, board, Location::get_loc(2, 0, board.x_size), P_WHITE);
    testAssert!(hist.is_game_finished == false);
    mmal!(hist, board, Board::PASS_LOC, P_BLACK);
    testAssert!(hist.is_game_finished == false);
    mmal!(hist, board, Board::PASS_LOC, P_WHITE);
    testAssert!(hist.is_game_finished == true);
    testAssert!(hist.winner == P_WHITE);
    testAssert!(hist.final_white_minus_black_score == 0.5f32);
    //Resurrecting the board after game over with another pass
    mmal!(hist, board, Board::PASS_LOC, P_BLACK);
    testAssert!(hist.is_game_finished == true);
    testAssert!(hist.winner == P_WHITE);
    testAssert!(hist.final_white_minus_black_score == 0.5f32);
    //And then some real moves followed by more passes
    mmal!(hist, board, Location::get_loc(3, 2, board.x_size), P_WHITE);
    testAssert!(hist.is_game_finished == false);
    mmal!(hist, board, Board::PASS_LOC, P_BLACK);
    testAssert!(hist.is_game_finished == false);
    mmal!(hist, board, Board::PASS_LOC, P_WHITE);
    testAssert!(hist.is_game_finished == true);
    testAssert!(hist.winner == P_WHITE);
    testAssert!(hist.final_white_minus_black_score == 0.5f32);
    out_board(&mut out, &board);
    expect(name, &mut out, "area-rules.expected0.txt");
    cases.ran();
  }
}
}

wasm_test! {
fn territory_rules() {
  init();
  let mut cases = Cases::new("testrules::territory_rules");
  let mut out = String::new();
  {
    let name = "Territory rules";
    let mut board = parse(4, 4, r"
....
....
....
....
");
    let mut rules = Rules::new();
    rules.ko_rule = Rules::KO_POSITIONAL;
    rules.scoring_rule = Rules::SCORING_TERRITORY;
    rules.komi = 0.5f32;
    rules.multi_stone_suicide_legal = true;
    rules.tax_rule = Rules::TAX_SEKI;
    let mut hist = BoardHistory::new_full(&board, P_BLACK, &rules, 0);

    mmal!(hist, board, Location::get_loc(1, 1, board.x_size), P_BLACK);
    mmal!(hist, board, Location::get_loc(2, 2, board.x_size), P_WHITE);
    mmal!(hist, board, Location::get_loc(1, 2, board.x_size), P_BLACK);
    mmal!(hist, board, Location::get_loc(2, 1, board.x_size), P_WHITE);
    mmal!(hist, board, Location::get_loc(1, 3, board.x_size), P_BLACK);
    mmal!(hist, board, Location::get_loc(2, 3, board.x_size), P_WHITE);
    mmal!(hist, board, Location::get_loc(1, 0, board.x_size), P_BLACK);
    mmal!(hist, board, Location::get_loc(2, 0, board.x_size), P_WHITE);
    mmal!(hist, board, Board::PASS_LOC, P_BLACK);
    mmal!(hist, board, Location::get_loc(3, 2, board.x_size), P_WHITE);
    mmal!(hist, board, Board::PASS_LOC, P_BLACK);
    testAssert!(hist.encore_phase == 0);
    testAssert!(hist.is_game_finished == false);
    mmal!(hist, board, Board::PASS_LOC, P_WHITE);
    testAssert!(hist.encore_phase == 1);
    testAssert!(hist.is_game_finished == false);
    mmal!(hist, board, Board::PASS_LOC, P_BLACK);
    testAssert!(hist.encore_phase == 1);
    testAssert!(hist.is_game_finished == false);
    mmal!(hist, board, Board::PASS_LOC, P_WHITE);
    testAssert!(hist.encore_phase == 2);
    testAssert!(hist.is_game_finished == false);
    mmal!(hist, board, Board::PASS_LOC, P_BLACK);
    testAssert!(hist.encore_phase == 2);
    testAssert!(hist.is_game_finished == false);
    mmal!(hist, board, Board::PASS_LOC, P_WHITE);
    testAssert!(hist.encore_phase == 2);
    testAssert!(hist.is_game_finished == true);
    testAssert!(hist.winner == P_BLACK);
    testAssert!(hist.final_white_minus_black_score == -0.5f32);
    out_board(&mut out, &board);

    //Resurrecting the board after pass to have black throw in a dead stone, since second encore, should make no difference
    mmal!(hist, board, Location::get_loc(3, 1, board.x_size), P_BLACK);
    mmal!(hist, board, Board::PASS_LOC, P_WHITE);
    mmal!(hist, board, Board::PASS_LOC, P_BLACK);
    testAssert!(hist.encore_phase == 2);
    testAssert!(hist.is_game_finished == true);
    testAssert!(hist.winner == P_BLACK);
    testAssert!(hist.final_white_minus_black_score == -0.5f32);
    out_board(&mut out, &board);

    //Resurrecting again to have white throw in a junk stone that makes it unclear if black has anything
    //White gets a point for playing, but it's not there second encore, so again no difference
    mmal!(hist, board, Location::get_loc(0, 1, board.x_size), P_WHITE);
    mmal!(hist, board, Board::PASS_LOC, P_BLACK);
    mmal!(hist, board, Board::PASS_LOC, P_WHITE);
    testAssert!(hist.encore_phase == 2);
    testAssert!(hist.is_game_finished == true);
    testAssert!(hist.winner == P_WHITE);
    testAssert!(hist.final_white_minus_black_score == 3.5f32);
    out_board(&mut out, &board);

    //Resurrecting again to have black solidfy his group and prove it pass-alive
    mmal!(hist, board, Location::get_loc(0, 2, board.x_size), P_BLACK);
    mmal!(hist, board, Location::get_loc(3, 0, board.x_size), P_WHITE);
    mmal!(hist, board, Board::PASS_LOC, P_BLACK);
    mmal!(hist, board, Board::PASS_LOC, P_WHITE);
    //Back to the original result
    testAssert!(hist.encore_phase == 2);
    testAssert!(hist.is_game_finished == true);
    testAssert!(hist.winner == P_BLACK);
    testAssert!(hist.final_white_minus_black_score == -0.5f32);
    out_board(&mut out, &board);

    expect(name, &mut out, "territory-rules.expected0.txt");
    cases.ran();
  }
}
}

wasm_test! {
fn simple_ko_rules() {
  init();
  let mut cases = Cases::new("testrules::simple_ko_rules");
  let mut out = String::new();
  //Ko rule testing with a regular ko and a sending two returning 1
  {
    let base_board = parse(6, 5, r"
.o.xxo
oxxxo.
o.x.oo
xx.oo.
oooo.o
");

    let mut base_rules = Rules::new();
    base_rules.ko_rule = Rules::KO_POSITIONAL;
    base_rules.scoring_rule = Rules::SCORING_TERRITORY;
    base_rules.komi = 0.5f32;
    base_rules.multi_stone_suicide_legal = false;
    base_rules.tax_rule = Rules::TAX_SEKI;

    {
      let name = "Simple ko rules";
      let mut board = base_board.clone();
      let mut rules = base_rules;
      rules.ko_rule = Rules::KO_SIMPLE;
      let mut hist = BoardHistory::new_full(&board, P_BLACK, &rules, 0);

      mmal!(hist, board, Location::get_loc(5, 1, board.x_size), P_BLACK);
      out += "After black ko capture:\n";
      print_illegal_moves(&mut out, &board, &hist, P_WHITE);

      mmal!(hist, board, Board::PASS_LOC, P_WHITE);
      out += "After black ko capture and one pass:\n";
      print_illegal_moves(&mut out, &board, &hist, P_BLACK);

      testAssert!(hist.pass_would_end_phase(&board, P_BLACK));
      testAssert!(!hist.pass_would_end_game(&board, P_BLACK));
      mmal!(hist, board, Location::get_loc(2, 3, board.x_size), P_BLACK);
      testAssert!(hist.encore_phase == 0);
      testAssert!(hist.is_game_finished == false);

      out += "After black ko capture and one pass and black other move:\n";
      print_illegal_moves(&mut out, &board, &hist, P_WHITE);

      mmal!(hist, board, Location::get_loc(5, 0, board.x_size), P_WHITE);
      out += "White recapture:\n";
      print_illegal_moves(&mut out, &board, &hist, P_BLACK);

      mmal!(hist, board, Location::get_loc(3, 2, board.x_size), P_BLACK);

      out += "Beginning sending two returning one cycle\n";
      mmal!(hist, board, Location::get_loc(2, 0, board.x_size), P_WHITE);
      print_illegal_moves(&mut out, &board, &hist, P_BLACK);
      mmal!(hist, board, Location::get_loc(0, 0, board.x_size), P_BLACK);
      print_illegal_moves(&mut out, &board, &hist, P_WHITE);
      mmal!(hist, board, Location::get_loc(1, 0, board.x_size), P_WHITE);
      print_illegal_moves(&mut out, &board, &hist, P_BLACK);
      mmal!(hist, board, Board::PASS_LOC, P_BLACK);
      print_illegal_moves(&mut out, &board, &hist, P_WHITE);
      mmal!(hist, board, Location::get_loc(2, 0, board.x_size), P_WHITE);
      print_illegal_moves(&mut out, &board, &hist, P_BLACK);
      mmal!(hist, board, Location::get_loc(0, 0, board.x_size), P_BLACK);
      print_illegal_moves(&mut out, &board, &hist, P_WHITE);
      mmal!(hist, board, Location::get_loc(1, 0, board.x_size), P_WHITE);
      print_illegal_moves(&mut out, &board, &hist, P_BLACK);
      testAssert!(hist.encore_phase == 0);
      testAssert!(hist.is_game_finished == false);
      //Spight ending condition cuts this cycle a bit shorter
      mmal!(hist, board, Board::PASS_LOC, P_BLACK);
      print_illegal_moves(&mut out, &board, &hist, P_WHITE);
      testAssert!(hist.encore_phase == 1);
      testAssert!(hist.is_game_finished == false);

      mmal!(hist, board, Board::PASS_LOC, P_WHITE);
      mmal!(hist, board, Board::PASS_LOC, P_BLACK);
      mmal!(hist, board, Board::PASS_LOC, P_WHITE);
      mmal!(hist, board, Board::PASS_LOC, P_BLACK);
      testAssert!(hist.encore_phase == 2);
      print_game_result(&mut out, &hist);

      expect(name, &mut out, "simple-ko-rules.expected0.txt");
      cases.ran();
    }
  }
}
}

wasm_test! {
fn positional_ko_rules() {
  init();
  let mut cases = Cases::new("testrules::positional_ko_rules");
  let mut out = String::new();
  //Ko rule testing with a regular ko and a sending two returning 1
  {
    let base_board = parse(6, 5, r"
.o.xxo
oxxxo.
o.x.oo
xx.oo.
oooo.o
");

    let mut base_rules = Rules::new();
    base_rules.ko_rule = Rules::KO_POSITIONAL;
    base_rules.scoring_rule = Rules::SCORING_TERRITORY;
    base_rules.komi = 0.5f32;
    base_rules.multi_stone_suicide_legal = false;
    base_rules.tax_rule = Rules::TAX_SEKI;

    {
      let name = "Positional ko rules";
      let mut board = base_board.clone();
      let mut rules = base_rules;
      rules.ko_rule = Rules::KO_POSITIONAL;
      let mut hist = BoardHistory::new_full(&board, P_BLACK, &rules, 0);

      mmal!(hist, board, Location::get_loc(5, 1, board.x_size), P_BLACK);
      out += "After black ko capture:\n";
      print_illegal_moves(&mut out, &board, &hist, P_WHITE);

      mmal!(hist, board, Board::PASS_LOC, P_WHITE);
      out += "After black ko capture and one pass:\n";
      print_illegal_moves(&mut out, &board, &hist, P_BLACK);

      //On tmp board and hist, verify that the main phase ends if black passes now
      let mut tmpboard = board.clone();
      let mut tmphist = hist.clone();
      mmal!(tmphist, tmpboard, Board::PASS_LOC, P_BLACK);
      testAssert!(tmphist.encore_phase == 1);
      testAssert!(tmphist.is_game_finished == false);

      mmal!(hist, board, Location::get_loc(3, 2, board.x_size), P_BLACK);
      out += "Beginning sending two returning one cycle\n";

      mmal!(hist, board, Location::get_loc(2, 0, board.x_size), P_WHITE);
      out += "After white sends two?\n";
      print_illegal_moves(&mut out, &board, &hist, P_BLACK);

      mmal!(hist, board, Location::get_loc(0, 0, board.x_size), P_BLACK);
      out += "Can white recapture?\n";
      print_illegal_moves(&mut out, &board, &hist, P_WHITE);

      mmal!(hist, board, Location::get_loc(5, 0, board.x_size), P_WHITE);
      out += "After white recaptures the other ko instead\n";
      print_illegal_moves(&mut out, &board, &hist, P_BLACK);

      mmal!(hist, board, Board::PASS_LOC, P_BLACK);
      out += "After white recaptures the other ko instead and black passes\n";
      print_illegal_moves(&mut out, &board, &hist, P_WHITE);

      mmal!(hist, board, Location::get_loc(1, 0, board.x_size), P_WHITE);
      out += "After white now returns 1\n";
      print_illegal_moves(&mut out, &board, &hist, P_BLACK);

      mmal!(hist, board, Board::PASS_LOC, P_BLACK);
      out += "After white now returns 1 and black passes\n";
      print_illegal_moves(&mut out, &board, &hist, P_WHITE);

      mmal!(hist, board, Location::get_loc(2, 0, board.x_size), P_WHITE);
      out += "After white sends 2 again\n";
      print_illegal_moves(&mut out, &board, &hist, P_BLACK);
      testAssert!(hist.encore_phase == 0);
      testAssert!(hist.is_game_finished == false);
      testAssert!(hist.num_consec_valid_turns_this_game == 10);

      testAssert!(!hist.is_legal(&board, Location::get_loc(0, 0, board.x_size), P_BLACK));
      testAssert!(hist.is_legal_tolerant(&board, Location::get_loc(0, 0, board.x_size), P_BLACK));
      mbmal!(hist, board, Location::get_loc(0, 0, board.x_size), P_BLACK, None);
      testAssert!(hist.num_consec_valid_turns_this_game == 0);
      check_ko_hash_consistency(&hist, &board, P_WHITE);


      expect(name, &mut out, "positional-ko-rules.expected0.txt");
      cases.ran();
    }
  }
}
}

wasm_test! {
fn situational_ko_rules() {
  init();
  let mut cases = Cases::new("testrules::situational_ko_rules");
  let mut out = String::new();
  //Ko rule testing with a regular ko and a sending two returning 1
  {
    let base_board = parse(6, 5, r"
.o.xxo
oxxxo.
o.x.oo
xx.oo.
oooo.o
");

    let mut base_rules = Rules::new();
    base_rules.ko_rule = Rules::KO_POSITIONAL;
    base_rules.scoring_rule = Rules::SCORING_TERRITORY;
    base_rules.komi = 0.5f32;
    base_rules.multi_stone_suicide_legal = false;
    base_rules.tax_rule = Rules::TAX_SEKI;

    {
      let name = "Situational ko rules";
      let mut board = base_board.clone();
      let mut rules = base_rules;
      rules.ko_rule = Rules::KO_SITUATIONAL;
      let mut hist = BoardHistory::new_full(&board, P_BLACK, &rules, 0);

      mmal!(hist, board, Location::get_loc(5, 1, board.x_size), P_BLACK);
      out += "After black ko capture:\n";
      print_illegal_moves(&mut out, &board, &hist, P_WHITE);

      mmal!(hist, board, Board::PASS_LOC, P_WHITE);
      out += "After black ko capture and one pass:\n";
      print_illegal_moves(&mut out, &board, &hist, P_BLACK);

      //On tmp board and hist, verify that the main phase ends if black passes now
      let mut tmpboard = board.clone();
      let mut tmphist = hist.clone();
      mmal!(tmphist, tmpboard, Board::PASS_LOC, P_BLACK);
      testAssert!(tmphist.encore_phase == 1);
      testAssert!(tmphist.is_game_finished == false);

      mmal!(hist, board, Location::get_loc(3, 2, board.x_size), P_BLACK);
      out += "Beginning sending two returning one cycle\n";

      mmal!(hist, board, Location::get_loc(2, 0, board.x_size), P_WHITE);
      out += "After white sends two?\n";
      print_illegal_moves(&mut out, &board, &hist, P_BLACK);

      mmal!(hist, board, Location::get_loc(0, 0, board.x_size), P_BLACK);
      out += "Can white recapture?\n";
      print_illegal_moves(&mut out, &board, &hist, P_WHITE);

      mmal!(hist, board, Location::get_loc(5, 0, board.x_size), P_WHITE);
      out += "After white recaptures the other ko instead\n";
      print_illegal_moves(&mut out, &board, &hist, P_BLACK);

      mmal!(hist, board, Board::PASS_LOC, P_BLACK);
      out += "After white recaptures the other ko instead and black passes\n";
      print_illegal_moves(&mut out, &board, &hist, P_WHITE);

      mmal!(hist, board, Location::get_loc(1, 0, board.x_size), P_WHITE);
      out += "After white now returns 1\n";
      print_illegal_moves(&mut out, &board, &hist, P_BLACK);

      mmal!(hist, board, Board::PASS_LOC, P_BLACK);
      out += "After white now returns 1 and black passes\n";
      print_illegal_moves(&mut out, &board, &hist, P_WHITE);

      mmal!(hist, board, Location::get_loc(2, 0, board.x_size), P_WHITE);
      out += "After white sends 2 again\n";
      print_illegal_moves(&mut out, &board, &hist, P_BLACK);
      testAssert!(hist.encore_phase == 0);
      testAssert!(hist.is_game_finished == false);

      expect(name, &mut out, "situational-ko-rules.expected0.txt");
      cases.ran();
    }
  }
}
}

wasm_test! {
fn spight_ko_rules() {
  init();
  let mut cases = Cases::new("testrules::spight_ko_rules");
  let mut out = String::new();
  //Ko rule testing with a regular ko and a sending two returning 1
  {
    let base_board = parse(6, 5, r"
.o.xxo
oxxxo.
o.x.oo
xx.oo.
oooo.o
");

    let mut base_rules = Rules::new();
    base_rules.ko_rule = Rules::KO_POSITIONAL;
    base_rules.scoring_rule = Rules::SCORING_TERRITORY;
    base_rules.komi = 0.5f32;
    base_rules.multi_stone_suicide_legal = false;
    base_rules.tax_rule = Rules::TAX_SEKI;

    {
      let name = "Spight ko rules";
      let mut board = base_board.clone();
      let suc = { let loc_ = Location::get_loc(2, 3, board.x_size); board.set_stone(loc_, C_BLACK) };
      testAssert!(suc);
      let mut rules = base_rules;
      rules.ko_rule = Rules::KO_SPIGHT;
      let mut hist = BoardHistory::new_full(&board, P_BLACK, &rules, 0);

      mmal!(hist, board, Location::get_loc(5, 1, board.x_size), P_BLACK);
      out += "After black ko capture:\n";
      print_illegal_moves(&mut out, &board, &hist, P_WHITE);

      mmal!(hist, board, Board::PASS_LOC, P_WHITE);
      out += "After black ko capture and one pass:\n";
      print_illegal_moves(&mut out, &board, &hist, P_BLACK);

      //On tmp board and hist, verify that the main phase does not end if black passes now
      let mut tmpboard = board.clone();
      let mut tmphist = hist.clone();
      mmal!(tmphist, tmpboard, Board::PASS_LOC, P_BLACK);
      testAssert!(tmphist.encore_phase == 0);
      testAssert!(tmphist.is_game_finished == false);
      out += "If black were to pass as well??\n";
      print_illegal_moves(&mut out, &tmpboard, &tmphist, P_WHITE);

      mmal!(hist, board, Location::get_loc(3, 2, board.x_size), P_BLACK);
      out += "Beginning sending two returning one cycle\n";

      mmal!(hist, board, Location::get_loc(2, 0, board.x_size), P_WHITE);
      out += "After white sends two?\n";
      print_illegal_moves(&mut out, &board, &hist, P_BLACK);

      mmal!(hist, board, Location::get_loc(0, 0, board.x_size), P_BLACK);
      out += "Can white recapture?\n";
      print_illegal_moves(&mut out, &board, &hist, P_WHITE);

      mmal!(hist, board, Location::get_loc(5, 0, board.x_size), P_WHITE);
      out += "After white recaptures the other ko instead\n";
      print_illegal_moves(&mut out, &board, &hist, P_BLACK);

      mmal!(hist, board, Board::PASS_LOC, P_BLACK);
      out += "After white recaptures the other ko instead and black passes\n";
      print_illegal_moves(&mut out, &board, &hist, P_WHITE);

      mmal!(hist, board, Location::get_loc(1, 0, board.x_size), P_WHITE);
      out += "After white now returns 1\n";
      print_illegal_moves(&mut out, &board, &hist, P_BLACK);

      mmal!(hist, board, Board::PASS_LOC, P_BLACK);
      out += "After white now returns 1 and black passes\n";
      print_illegal_moves(&mut out, &board, &hist, P_WHITE);

      mmal!(hist, board, Location::get_loc(2, 0, board.x_size), P_WHITE);
      out += "After white sends 2 again\n";
      print_illegal_moves(&mut out, &board, &hist, P_BLACK);

      mmal!(hist, board, Location::get_loc(0, 0, board.x_size), P_BLACK);
      out += "Can white recapture?\n";
      print_illegal_moves(&mut out, &board, &hist, P_WHITE);

      mmal!(hist, board, Board::PASS_LOC, P_WHITE);
      out += "After pass\n";
      print_illegal_moves(&mut out, &board, &hist, P_BLACK);
      testAssert!(hist.encore_phase == 0);
      testAssert!(hist.is_game_finished == false);

      //This is actually black's second pass in this position!
      mmal!(hist, board, Board::PASS_LOC, P_BLACK);
      out += "After pass\n";
      print_illegal_moves(&mut out, &board, &hist, P_WHITE);
      testAssert!(hist.encore_phase == 1);
      testAssert!(hist.is_game_finished == false);

      mmal!(hist, board, Board::PASS_LOC, P_WHITE);
      mmal!(hist, board, Board::PASS_LOC, P_BLACK);
      mmal!(hist, board, Board::PASS_LOC, P_WHITE);
      mmal!(hist, board, Board::PASS_LOC, P_BLACK);
      testAssert!(hist.encore_phase == 2);
      print_game_result(&mut out, &hist);

      expect(name, &mut out, "spight-ko-rules.expected0.txt");
      cases.ran();
    }
  }
}
}

wasm_test! {
fn suicide_ko_testing() {
  init();
  let mut cases = Cases::new("testrules::suicide_ko_testing");
  let mut out = String::new();
  //Testing superko with suicide
  {
    let base_board = parse(6, 5, r"
.oxo.x
oxxooo
xx....
......
......
");

    let mut base_rules = Rules::new();
    base_rules.ko_rule = Rules::KO_POSITIONAL;
    base_rules.scoring_rule = Rules::SCORING_AREA;
    base_rules.komi = 0.5f32;
    base_rules.multi_stone_suicide_legal = true;
    base_rules.tax_rule = Rules::TAX_NONE;

    let ko_rules_to_test: [i32; 3] = [Rules::KO_POSITIONAL, Rules::KO_SITUATIONAL, Rules::KO_SPIGHT];
    let name = "Suicide ko testing";
    for i in 0..3
    {
      out += "------------------------------\n";
      let mut board = base_board.clone();
      let mut rules = base_rules;
      rules.ko_rule = ko_rules_to_test[i];
      let mut hist = BoardHistory::new_full(&board, P_BLACK, &rules, 0);

      mmal!(hist, board, Location::get_loc(4, 0, board.x_size), P_BLACK);
      mmal!(hist, board, Board::PASS_LOC, P_WHITE);
      out += "After black suicide and white pass\n";
      print_illegal_moves(&mut out, &board, &hist, P_BLACK);

      mmal!(hist, board, Location::get_loc(4, 0, board.x_size), P_BLACK);
      mmal!(hist, board, Location::get_loc(0, 0, board.x_size), P_WHITE);
      mmal!(hist, board, Location::get_loc(5, 0, board.x_size), P_BLACK);
      mmal!(hist, board, Location::get_loc(1, 0, board.x_size), P_WHITE);
      mmal!(hist, board, Board::PASS_LOC, P_BLACK);
      out += "After a little looping\n";
      print_illegal_moves(&mut out, &board, &hist, P_WHITE);

      mmal!(hist, board, Location::get_loc(0, 0, board.x_size), P_WHITE);
      mmal!(hist, board, Location::get_loc(4, 0, board.x_size), P_BLACK);
      out += "Filling in a bit more\n";
      print_illegal_moves(&mut out, &board, &hist, P_WHITE);

      //Illegal under non-spight superkos, but still should be handled gracefully
      mbmal!(hist, board, Location::get_loc(0, 1, board.x_size), P_WHITE, None);
      mbmal!(hist, board, Location::get_loc(5, 0, board.x_size), P_BLACK, None);
      mbmal!(hist, board, Location::get_loc(1, 0, board.x_size), P_WHITE, None);
      mbmal!(hist, board, Location::get_loc(4, 0, board.x_size), P_BLACK, None);
      out += "Looped some more\n";
      print_illegal_moves(&mut out, &board, &hist, P_WHITE);
      out_board(&mut out, &board);

    }
    expect(name, &mut out, "suicide-ko-testing.expected0.txt");
      cases.ran();
  }
}
}

wasm_test! {
fn eternal_life() {
  init();
  let mut cases = Cases::new("testrules::eternal_life");
  let mut out = String::new();
  {
    let name = "Eternal life";
    let mut board = parse(8, 5, r"
........
oooooo..
xxxxxo..
xoooxxoo
.o.x.ox.
");
    let mut rules = Rules::new();
    rules.ko_rule = Rules::KO_SIMPLE;
    rules.scoring_rule = Rules::SCORING_AREA;
    rules.komi = 0.5f32;
    rules.multi_stone_suicide_legal = false;
    rules.tax_rule = Rules::TAX_NONE;
    let mut hist = BoardHistory::new_full(&board, P_BLACK, &rules, 0);

    mmal!(hist, board, Location::get_loc(2, 4, board.x_size), P_BLACK);
    mmal!(hist, board, Location::get_loc(4, 4, board.x_size), P_WHITE);
    mmal!(hist, board, Location::get_loc(3, 4, board.x_size), P_BLACK);
    mmal!(hist, board, Location::get_loc(5, 4, board.x_size), P_WHITE);
    mmal!(hist, board, Location::get_loc(2, 4, board.x_size), P_BLACK);
    mmal!(hist, board, Location::get_loc(4, 4, board.x_size), P_WHITE);
    mmal!(hist, board, Location::get_loc(3, 4, board.x_size), P_BLACK);
    testAssert!(hist.is_game_finished == false);
    mmal!(hist, board, Location::get_loc(5, 4, board.x_size), P_WHITE);
    testAssert!(hist.is_game_finished == true);
    print_game_result(&mut out, &hist);

    expect(name, &mut out, "eternal-life.expected0.txt");
      cases.ran();
  }
}
}

wasm_test! {
fn triple_ko_simple() {
  init();
  let mut cases = Cases::new("testrules::triple_ko_simple");
  let mut out = String::new();
  {
    let name = "Triple ko simple";
    let mut board = parse(7, 6, r"
ooooooo
oxo.o.o
x.xoxox
xxxxxxx
ooooooo
.......
");
    let mut rules = Rules::new();
    rules.ko_rule = Rules::KO_SIMPLE;
    rules.scoring_rule = Rules::SCORING_AREA;
    rules.komi = 0.5f32;
    rules.multi_stone_suicide_legal = false;
    rules.tax_rule = Rules::TAX_NONE;
    let mut hist = BoardHistory::new_full(&board, P_BLACK, &rules, 0);

    mmal!(hist, board, Location::get_loc(3, 1, board.x_size), P_BLACK);
    mmal!(hist, board, Location::get_loc(1, 2, board.x_size), P_WHITE);
    mmal!(hist, board, Location::get_loc(5, 1, board.x_size), P_BLACK);
    mmal!(hist, board, Location::get_loc(3, 2, board.x_size), P_WHITE);
    mmal!(hist, board, Location::get_loc(1, 1, board.x_size), P_BLACK);
    mmal!(hist, board, Location::get_loc(5, 2, board.x_size), P_WHITE);
    mmal!(hist, board, Location::get_loc(3, 1, board.x_size), P_BLACK);
    mmal!(hist, board, Location::get_loc(1, 2, board.x_size), P_WHITE);
    mmal!(hist, board, Location::get_loc(5, 1, board.x_size), P_BLACK);
    mmal!(hist, board, Location::get_loc(3, 2, board.x_size), P_WHITE);
    mmal!(hist, board, Location::get_loc(1, 1, board.x_size), P_BLACK);
    testAssert!(hist.is_game_finished == false);
    mmal!(hist, board, Location::get_loc(5, 2, board.x_size), P_WHITE);
    testAssert!(hist.is_game_finished == true);
    print_game_result(&mut out, &hist);

    expect(name, &mut out, "triple-ko-simple.expected0.txt");
      cases.ran();
  }
}
}

wasm_test! {
fn triple_ko_superko() {
  init();
  let mut cases = Cases::new("testrules::triple_ko_superko");
  let mut out = String::new();
  {
    let name = "Triple ko superko";
    let mut board = parse(7, 6, r"
ooooooo
oxo.o.o
x.xoxox
xxxxxxx
ooooooo
.......
");
    let mut rules = Rules::new();
    rules.ko_rule = Rules::KO_POSITIONAL;
    rules.scoring_rule = Rules::SCORING_AREA;
    rules.komi = 0.5f32;
    rules.multi_stone_suicide_legal = false;
    rules.tax_rule = Rules::TAX_NONE;
    let mut hist = BoardHistory::new_full(&board, P_BLACK, &rules, 0);

    mmal!(hist, board, Location::get_loc(3, 1, board.x_size), P_BLACK);
    mmal!(hist, board, Location::get_loc(1, 2, board.x_size), P_WHITE);
    mmal!(hist, board, Location::get_loc(5, 1, board.x_size), P_BLACK);
    mmal!(hist, board, Location::get_loc(3, 2, board.x_size), P_WHITE);
    mmal!(hist, board, Location::get_loc(1, 1, board.x_size), P_BLACK);
    print_illegal_moves(&mut out, &board, &hist, P_WHITE);
    expect(name, &mut out, "triple-ko-superko.expected0.txt");
      cases.ran();
  }
}
}

wasm_test! {
fn triple_ko_encore() {
  init();
  let mut cases = Cases::new("testrules::triple_ko_encore");
  let mut out = String::new();
  {
    let name = "Triple ko encore";
    let mut board = parse(7, 6, r"
ooooooo
oxo.o.o
x.xoxox
xxxxxxx
ooooooo
.......
");
    let mut rules = Rules::new();
    rules.ko_rule = Rules::KO_POSITIONAL;
    rules.scoring_rule = Rules::SCORING_TERRITORY;
    rules.komi = 0.5f32;
    rules.multi_stone_suicide_legal = false;
    rules.tax_rule = Rules::TAX_SEKI;
    let mut hist = BoardHistory::new_full(&board, P_BLACK, &rules, 0);

    mmal!(hist, board, Board::PASS_LOC, P_BLACK);
    mmal!(hist, board, Board::PASS_LOC, P_WHITE);
    mmal!(hist, board, Location::get_loc(3, 1, board.x_size), P_BLACK);
    mmal!(hist, board, Location::get_loc(1, 2, board.x_size), P_WHITE);
    mmal!(hist, board, Location::get_loc(5, 1, board.x_size), P_BLACK);
    //Pass for ko
    mmal!(hist, board, Location::get_loc(3, 2, board.x_size), P_WHITE);
    //Should be a complete capture
    mmal!(hist, board, Location::get_loc(1, 1, board.x_size), P_BLACK);
    out_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);

    expect(name, &mut out, "triple-ko-encore.expected0.txt");
      cases.ran();
  }
}
}

wasm_test! {
fn encore_own_throwin_that_temporarily_breaks_the_ko_shape_should_not_clear_the_ko_recap_block() {
  init();
  let mut cases = Cases::new("testrules::encore_own_throwin_that_temporarily_breaks_the_ko_shape_should_not_clear_the_ko_recap_block");
  let mut out = String::new();
  {
    let name = "Encore - own throwin that temporarily breaks the ko shape should not clear the ko recap block";
    let mut board = parse(7, 6, r"
..o....
...o...
.xoxo..
..x.x..
...x...
.......
");
    let mut rules = Rules::new();
    rules.ko_rule = Rules::KO_POSITIONAL;
    rules.scoring_rule = Rules::SCORING_TERRITORY;
    rules.komi = 0.5f32;
    rules.multi_stone_suicide_legal = false;
    rules.tax_rule = Rules::TAX_SEKI;
    let mut hist = BoardHistory::new_full(&board, P_WHITE, &rules, 0);

    mmal!(hist, board, Board::PASS_LOC, P_WHITE);
    mmal!(hist, board, Board::PASS_LOC, P_BLACK);
    mmal!(hist, board, Location::get_loc(3, 3, board.x_size), P_WHITE);
    out_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    mmal!(hist, board, Location::get_loc(2, 1, board.x_size), P_BLACK);
    out_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    mmal!(hist, board, Location::get_loc(1, 1, board.x_size), P_WHITE);
    out_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);

    expect(name, &mut out, "encore-own-throwin-that-temporarily-breaks-the-ko-shape-should-not-clear-the-ko-recap-block.expected0.txt");
      cases.ran();
  }
}
}

wasm_test! {
fn encore_ko_recap_block_does_not_stop_non_ko_capture() {
  init();
  let mut cases = Cases::new("testrules::encore_ko_recap_block_does_not_stop_non_ko_capture");
  let mut out = String::new();
  {
    let name = "Encore - ko recap block does not stop non-ko-capture";
    let mut board = parse(7, 6, r"
..o....
...o...
.xoxo..
..x.x..
...x...
.......
");
    let mut rules = Rules::new();
    rules.ko_rule = Rules::KO_POSITIONAL;
    rules.scoring_rule = Rules::SCORING_TERRITORY;
    rules.komi = 0.5f32;
    rules.multi_stone_suicide_legal = false;
    rules.tax_rule = Rules::TAX_SEKI;
    let mut hist = BoardHistory::new_full(&board, P_WHITE, &rules, 0);

    mmal!(hist, board, Board::PASS_LOC, P_WHITE);
    mmal!(hist, board, Board::PASS_LOC, P_BLACK);
    mmal!(hist, board, Location::get_loc(3, 3, board.x_size), P_WHITE);
    out_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    mmal!(hist, board, Location::get_loc(2, 1, board.x_size), P_BLACK);
    out_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    mmal!(hist, board, Location::get_loc(0, 0, board.x_size), P_WHITE);
    out_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    mmal!(hist, board, Location::get_loc(3, 2, board.x_size), P_BLACK);
    out_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);

    expect(name, &mut out, "encore-ko-recap-block-does-not-stop-non-ko-capture.expected0.txt");
      cases.ran();
  }
}
}

wasm_test! {
fn encore_once_only_rule_doesn_t_prevent_the_opponent_moving_there_filling_ko() {
  init();
  let mut cases = Cases::new("testrules::encore_once_only_rule_doesn_t_prevent_the_opponent_moving_there_filling_ko");
  let mut out = String::new();
  {
    let name = "Encore - once only rule doesn't prevent the opponent moving there (filling ko)";
    let mut board = parse(7, 6, r"
..o....
...o...
.xoxo..
..x.x..
...x...
.......
");
    let mut rules = Rules::new();
    rules.ko_rule = Rules::KO_POSITIONAL;
    rules.scoring_rule = Rules::SCORING_TERRITORY;
    rules.komi = 0.5f32;
    rules.multi_stone_suicide_legal = false;
    rules.tax_rule = Rules::TAX_SEKI;
    let mut hist = BoardHistory::new_full(&board, P_WHITE, &rules, 0);

    mmal!(hist, board, Board::PASS_LOC, P_WHITE);
    mmal!(hist, board, Board::PASS_LOC, P_BLACK);
    mmal!(hist, board, Location::get_loc(3, 3, board.x_size), P_WHITE);
    out_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    //Pass for ko
    mmal!(hist, board, Location::get_loc(3, 2, board.x_size), P_BLACK);
    out_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    //Pass
    mmal!(hist, board, Board::PASS_LOC, P_WHITE);
    out_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    //Take ko
    mmal!(hist, board, Location::get_loc(3, 2, board.x_size), P_BLACK);
    out_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    //Pass
    mmal!(hist, board, Board::PASS_LOC, P_WHITE);
    out_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    //Fill ko
    mmal!(hist, board, Location::get_loc(3, 3, board.x_size), P_BLACK);
    out_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);

    mmal!(hist, board, Location::get_loc(1, 3, board.x_size), P_WHITE);
    mmal!(hist, board, Board::PASS_LOC, P_BLACK);
    mmal!(hist, board, Location::get_loc(2, 4, board.x_size), P_WHITE);
    mmal!(hist, board, Board::PASS_LOC, P_BLACK);
    mmal!(hist, board, Location::get_loc(3, 5, board.x_size), P_WHITE);
    mmal!(hist, board, Board::PASS_LOC, P_BLACK);
    mmal!(hist, board, Location::get_loc(4, 4, board.x_size), P_WHITE);
    mmal!(hist, board, Board::PASS_LOC, P_BLACK);
    out_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    mmal!(hist, board, Location::get_loc(5, 3, board.x_size), P_WHITE);
    out_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);

    expect(name, &mut out, "encore-once-only-rule-doesn-t-prevent-the-opponent-moving-there-filling-ko.expected0.txt");
      cases.ran();
  }
}
}

wasm_test! {
fn area_scoring_in_the_main_phase() {
  init();
  let mut cases = Cases::new("testrules::area_scoring_in_the_main_phase");
  let mut out = String::new();
  {
    let name = "Area scoring in the main phase";

    let tax_rules: [i32; 3] = [Rules::TAX_NONE, Rules::TAX_SEKI, Rules::TAX_ALL];
    for which_tax_rule in 0..3 {
      let mut board = parse(7, 7, r"
ox.ooo.
oxxxxxx
ooooooo
.xoxx..
ooox...
x.oxxxx
.xox...
");
      let mut rules = Rules::new();
      rules.ko_rule = Rules::KO_POSITIONAL;
      rules.scoring_rule = Rules::SCORING_AREA;
      rules.komi = 0.5f32;
      rules.multi_stone_suicide_legal = false;
      rules.tax_rule = tax_rules[which_tax_rule];
      let mut hist = BoardHistory::new_full(&board, P_BLACK, &rules, 0);

      writeln!(out, "Score: {}", ds(final_score_if_game_ended_now(&hist, &board))).unwrap();
      mmal!(hist, board, Location::get_loc(5, 3, board.x_size), P_BLACK);
      writeln!(out, "Score: {}", ds(final_score_if_game_ended_now(&hist, &board))).unwrap();
      mmal!(hist, board, Location::get_loc(6, 3, board.x_size), P_WHITE);
      writeln!(out, "Score: {}", ds(final_score_if_game_ended_now(&hist, &board))).unwrap();
      mmal!(hist, board, Location::get_loc(6, 4, board.x_size), P_BLACK);
      writeln!(out, "Score: {}", ds(final_score_if_game_ended_now(&hist, &board))).unwrap();
      mmal!(hist, board, Location::get_loc(5, 4, board.x_size), P_WHITE);
      writeln!(out, "Score: {}", ds(final_score_if_game_ended_now(&hist, &board))).unwrap();
      mmal!(hist, board, Location::get_loc(4, 4, board.x_size), P_BLACK);
      writeln!(out, "Score: {}", ds(final_score_if_game_ended_now(&hist, &board))).unwrap();
      mmal!(hist, board, Location::get_loc(0, 3, board.x_size), P_WHITE);
      writeln!(out, "Score: {}", ds(final_score_if_game_ended_now(&hist, &board))).unwrap();
      mmal!(hist, board, Location::get_loc(6, 6, board.x_size), P_BLACK);
      writeln!(out, "Score: {}", ds(final_score_if_game_ended_now(&hist, &board))).unwrap();
      out.push('\n');
    }
    expect(name, &mut out, "area-scoring-in-the-main-phase.expected0.txt");
    cases.ran();
  }
}
}

wasm_test! {
fn territory_scoring_in_the_main_phase() {
  init();
  let mut cases = Cases::new("testrules::territory_scoring_in_the_main_phase");
  let mut out = String::new();
  {
    let name = "Territory scoring in the main phase";

    let tax_rules: [i32; 3] = [Rules::TAX_NONE, Rules::TAX_SEKI, Rules::TAX_ALL];
    for which_tax_rule in 0..3 {
      let mut board = parse(7, 7, r"
ox.ooo.
oxxxxxx
ooooooo
.xoxx..
ooox...
x.oxxxx
.xox...
");
      let mut rules = Rules::new();
      rules.ko_rule = Rules::KO_POSITIONAL;
      rules.scoring_rule = Rules::SCORING_TERRITORY;
      rules.komi = 0.5f32;
      rules.multi_stone_suicide_legal = false;
      rules.tax_rule = tax_rules[which_tax_rule];
      let mut hist = BoardHistory::new_full(&board, P_BLACK, &rules, 0);

      writeln!(out, "Score: {}", ds(final_score_if_game_ended_now(&hist, &board))).unwrap();
      mmal!(hist, board, Location::get_loc(5, 3, board.x_size), P_BLACK);
      writeln!(out, "Score: {}", ds(final_score_if_game_ended_now(&hist, &board))).unwrap();
      mmal!(hist, board, Location::get_loc(6, 3, board.x_size), P_WHITE);
      writeln!(out, "Score: {}", ds(final_score_if_game_ended_now(&hist, &board))).unwrap();
      mmal!(hist, board, Location::get_loc(6, 4, board.x_size), P_BLACK);
      writeln!(out, "Score: {}", ds(final_score_if_game_ended_now(&hist, &board))).unwrap();
      mmal!(hist, board, Location::get_loc(5, 4, board.x_size), P_WHITE);
      writeln!(out, "Score: {}", ds(final_score_if_game_ended_now(&hist, &board))).unwrap();
      mmal!(hist, board, Location::get_loc(4, 4, board.x_size), P_BLACK);
      writeln!(out, "Score: {}", ds(final_score_if_game_ended_now(&hist, &board))).unwrap();
      mmal!(hist, board, Location::get_loc(0, 3, board.x_size), P_WHITE);
      writeln!(out, "Score: {}", ds(final_score_if_game_ended_now(&hist, &board))).unwrap();
      mmal!(hist, board, Location::get_loc(6, 6, board.x_size), P_BLACK);
      writeln!(out, "Score: {}", ds(final_score_if_game_ended_now(&hist, &board))).unwrap();
      out.push('\n');
    }
    expect(name, &mut out, "territory-scoring-in-the-main-phase.expected0.txt");
    cases.ran();
  }
}
}

wasm_test! {
fn territory_scoring_in_encore_1() {
  init();
  let mut cases = Cases::new("testrules::territory_scoring_in_encore_1");
  let mut out = String::new();
  {
    let name = "Territory scoring in encore 1";

    let tax_rules: [i32; 3] = [Rules::TAX_NONE, Rules::TAX_SEKI, Rules::TAX_ALL];
    for which_tax_rule in 0..3 {
      let mut board = parse(7, 7, r"
ox.ooo.
oxxxxxx
ooooooo
.xoxx..
ooox...
x.oxxxx
.xox...
");
      let mut rules = Rules::new();
      rules.ko_rule = Rules::KO_POSITIONAL;
      rules.scoring_rule = Rules::SCORING_TERRITORY;
      rules.komi = 0.5f32;
      rules.multi_stone_suicide_legal = false;
      rules.tax_rule = tax_rules[which_tax_rule];
      let mut hist = BoardHistory::new_full(&board, P_BLACK, &rules, 0);

      writeln!(out, "Score: {}", ds(final_score_if_game_ended_now(&hist, &board))).unwrap();
      mmal!(hist, board, Location::get_loc(5, 3, board.x_size), P_BLACK);
      writeln!(out, "Score: {}", ds(final_score_if_game_ended_now(&hist, &board))).unwrap();
      mmal!(hist, board, Location::get_loc(6, 3, board.x_size), P_WHITE);
      writeln!(out, "Score: {}", ds(final_score_if_game_ended_now(&hist, &board))).unwrap();
      mmal!(hist, board, Location::get_loc(6, 4, board.x_size), P_BLACK);
      writeln!(out, "Score: {}", ds(final_score_if_game_ended_now(&hist, &board))).unwrap();
      mmal!(hist, board, Board::PASS_LOC, P_WHITE);
      mmal!(hist, board, Board::PASS_LOC, P_BLACK);
      mmal!(hist, board, Location::get_loc(5, 4, board.x_size), P_WHITE);
      writeln!(out, "Score: {}", ds(final_score_if_game_ended_now(&hist, &board))).unwrap();
      mmal!(hist, board, Location::get_loc(4, 4, board.x_size), P_BLACK);
      writeln!(out, "Score: {}", ds(final_score_if_game_ended_now(&hist, &board))).unwrap();
      mmal!(hist, board, Location::get_loc(0, 3, board.x_size), P_WHITE);
      writeln!(out, "Score: {}", ds(final_score_if_game_ended_now(&hist, &board))).unwrap();
      mmal!(hist, board, Location::get_loc(6, 6, board.x_size), P_BLACK);
      writeln!(out, "Score: {}", ds(final_score_if_game_ended_now(&hist, &board))).unwrap();
      out.push('\n');
    }
    expect(name, &mut out, "territory-scoring-in-encore-1.expected0.txt");
    cases.ran();
  }
}
}

wasm_test! {
fn territory_scoring_in_encore_2() {
  init();
  let mut cases = Cases::new("testrules::territory_scoring_in_encore_2");
  let mut out = String::new();
  {
    let name = "Territory scoring in encore 2";

    let tax_rules: [i32; 3] = [Rules::TAX_NONE, Rules::TAX_SEKI, Rules::TAX_ALL];
    for which_tax_rule in 0..3 {
      let mut board = parse(7, 7, r"
ox.ooo.
oxxxxxx
ooooooo
.xoxx..
ooox...
x.oxxxx
.xox...
");
      let mut rules = Rules::new();
      rules.ko_rule = Rules::KO_POSITIONAL;
      rules.scoring_rule = Rules::SCORING_TERRITORY;
      rules.komi = 0.5f32;
      rules.multi_stone_suicide_legal = false;
      rules.tax_rule = tax_rules[which_tax_rule];
      let mut hist = BoardHistory::new_full(&board, P_BLACK, &rules, 0);

      writeln!(out, "Score: {}", ds(final_score_if_game_ended_now(&hist, &board))).unwrap();
      mmal!(hist, board, Location::get_loc(5, 3, board.x_size), P_BLACK);
      writeln!(out, "Score: {}", ds(final_score_if_game_ended_now(&hist, &board))).unwrap();
      mmal!(hist, board, Location::get_loc(6, 3, board.x_size), P_WHITE);
      writeln!(out, "Score: {}", ds(final_score_if_game_ended_now(&hist, &board))).unwrap();
      mmal!(hist, board, Location::get_loc(6, 4, board.x_size), P_BLACK);
      writeln!(out, "Score: {}", ds(final_score_if_game_ended_now(&hist, &board))).unwrap();
      mmal!(hist, board, Board::PASS_LOC, P_WHITE);
      mmal!(hist, board, Board::PASS_LOC, P_BLACK);
      mmal!(hist, board, Board::PASS_LOC, P_WHITE);
      mmal!(hist, board, Board::PASS_LOC, P_BLACK);
      mmal!(hist, board, Location::get_loc(5, 4, board.x_size), P_WHITE);
      writeln!(out, "Score: {}", ds(final_score_if_game_ended_now(&hist, &board))).unwrap();
      mmal!(hist, board, Location::get_loc(4, 4, board.x_size), P_BLACK);
      writeln!(out, "Score: {}", ds(final_score_if_game_ended_now(&hist, &board))).unwrap();
      mmal!(hist, board, Location::get_loc(0, 3, board.x_size), P_WHITE);
      writeln!(out, "Score: {}", ds(final_score_if_game_ended_now(&hist, &board))).unwrap();
      mmal!(hist, board, Location::get_loc(6, 6, board.x_size), P_BLACK);
      writeln!(out, "Score: {}", ds(final_score_if_game_ended_now(&hist, &board))).unwrap();
      out.push('\n');
    }
    expect(name, &mut out, "territory-scoring-in-encore-2.expected0.txt");
    cases.ran();
  }
}
}

wasm_test! {
fn fill_seki_liberties_in_main_phase() {
  init();
  let mut cases = Cases::new("testrules::fill_seki_liberties_in_main_phase");
  let mut out = String::new();
  {
    let name = "Fill seki liberties in main phase";

    let tax_rules: [i32; 3] = [Rules::TAX_NONE, Rules::TAX_SEKI, Rules::TAX_ALL];
    for which_tax_rule in 0..3 {
      let mut board = parse(7, 7, r"
...oxx.
oooox.x
xxxxoxx
o.xoooo
.oxox.o
oxxo.x.
o.xoo.x
");
      let mut rules = Rules::new();
      rules.ko_rule = Rules::KO_POSITIONAL;
      rules.scoring_rule = Rules::SCORING_TERRITORY;
      rules.komi = -0.5f32;
      rules.multi_stone_suicide_legal = false;
      rules.tax_rule = tax_rules[which_tax_rule];
      let mut hist = BoardHistory::new_full(&board, P_WHITE, &rules, 0);

      writeln!(out, "Score: {}", ds(final_score_if_game_ended_now(&hist, &board))).unwrap();
      mmal!(hist, board, Location::get_loc(6, 5, board.x_size), P_WHITE);
      writeln!(out, "Score: {}", ds(final_score_if_game_ended_now(&hist, &board))).unwrap();
      mmal!(hist, board, Location::get_loc(5, 6, board.x_size), P_BLACK);
      writeln!(out, "Score: {}", ds(final_score_if_game_ended_now(&hist, &board))).unwrap();
      mmal!(hist, board, Location::get_loc(0, 4, board.x_size), P_WHITE);
      writeln!(out, "Score: {}", ds(final_score_if_game_ended_now(&hist, &board))).unwrap();
      mmal!(hist, board, Location::get_loc(6, 0, board.x_size), P_BLACK);
      writeln!(out, "Score: {}", ds(final_score_if_game_ended_now(&hist, &board))).unwrap();
      mmal!(hist, board, Location::get_loc(1, 0, board.x_size), P_WHITE);
      writeln!(out, "Score: {}", ds(final_score_if_game_ended_now(&hist, &board))).unwrap();
      mmal!(hist, board, Location::get_loc(4, 5, board.x_size), P_BLACK);
      writeln!(out, "Score: {}", ds(final_score_if_game_ended_now(&hist, &board))).unwrap();
      mmal!(hist, board, Location::get_loc(5, 4, board.x_size), P_WHITE);
      writeln!(out, "Score: {}", ds(final_score_if_game_ended_now(&hist, &board))).unwrap();
      out.push('\n');
    }
    expect(name, &mut out, "fill-seki-liberties-in-main-phase.expected0.txt");
    cases.ran();
  }
}
}

wasm_test! {
fn fill_seki_liberties_in_encore_2() {
  init();
  let mut cases = Cases::new("testrules::fill_seki_liberties_in_encore_2");
  let mut out = String::new();
  {
    let name = "Fill seki liberties in encore 2";

    let tax_rules: [i32; 3] = [Rules::TAX_NONE, Rules::TAX_SEKI, Rules::TAX_ALL];
    for which_tax_rule in 0..3 {
      let mut board = parse(7, 7, r"
...oxx.
oooox.x
xxxxoxx
o.xoooo
.oxox.o
oxxo.x.
o.xoo.x
");
      let mut rules = Rules::new();
      rules.ko_rule = Rules::KO_POSITIONAL;
      rules.scoring_rule = Rules::SCORING_TERRITORY;
      rules.komi = -0.5f32;
      rules.multi_stone_suicide_legal = false;
      rules.tax_rule = tax_rules[which_tax_rule];
      let mut hist = BoardHistory::new_full(&board, P_WHITE, &rules, 0);

      mmal!(hist, board, Board::PASS_LOC, P_WHITE);
      mmal!(hist, board, Board::PASS_LOC, P_BLACK);
      mmal!(hist, board, Board::PASS_LOC, P_WHITE);
      mmal!(hist, board, Board::PASS_LOC, P_BLACK);
      writeln!(out, "Score: {}", ds(final_score_if_game_ended_now(&hist, &board))).unwrap();
      mmal!(hist, board, Location::get_loc(6, 5, board.x_size), P_WHITE);
      writeln!(out, "Score: {}", ds(final_score_if_game_ended_now(&hist, &board))).unwrap();
      mmal!(hist, board, Location::get_loc(5, 6, board.x_size), P_BLACK);
      writeln!(out, "Score: {}", ds(final_score_if_game_ended_now(&hist, &board))).unwrap();
      mmal!(hist, board, Location::get_loc(0, 4, board.x_size), P_WHITE);
      writeln!(out, "Score: {}", ds(final_score_if_game_ended_now(&hist, &board))).unwrap();
      mmal!(hist, board, Location::get_loc(6, 0, board.x_size), P_BLACK);
      writeln!(out, "Score: {}", ds(final_score_if_game_ended_now(&hist, &board))).unwrap();
      mmal!(hist, board, Location::get_loc(1, 0, board.x_size), P_WHITE);
      writeln!(out, "Score: {}", ds(final_score_if_game_ended_now(&hist, &board))).unwrap();
      mmal!(hist, board, Location::get_loc(4, 5, board.x_size), P_BLACK);
      writeln!(out, "Score: {}", ds(final_score_if_game_ended_now(&hist, &board))).unwrap();
      mmal!(hist, board, Location::get_loc(5, 4, board.x_size), P_WHITE);
      writeln!(out, "Score: {}", ds(final_score_if_game_ended_now(&hist, &board))).unwrap();
      out.push('\n');
    }
    expect(name, &mut out, "fill-seki-liberties-in-encore-2.expected0.txt");
    cases.ran();
  }
}
}

wasm_test! {
fn area_scoring_with_button() {
  init();
  let mut cases = Cases::new("testrules::area_scoring_with_button");
  let mut out = String::new();
  {
    let name = "Area scoring with button";

    let button_rule: [bool; 2] = [false, true];
    for which_rule in 0..2 {
      let mut board = parse(7, 7, r"
..x.xo.
..xxoo.
...xo..
..xxo..
..x.o..
..xxo..
...xo..
");
      let mut rules = Rules::new();
      rules.ko_rule = Rules::KO_SIMPLE;
      rules.scoring_rule = Rules::SCORING_AREA;
      rules.tax_rule = Rules::TAX_NONE;
      rules.multi_stone_suicide_legal = false;
      rules.komi = 2.5f32;
      rules.has_button = button_rule[which_rule];
      let mut hist = BoardHistory::new_full(&board, P_BLACK, &rules, 0);

      writeln!(out, "Score: {}", ds(final_score_if_game_ended_now(&hist, &board))).unwrap();
      mmal!(hist, board, Location::get_loc(3, 4, board.x_size), P_BLACK);
      writeln!(out, "Score: {}", ds(final_score_if_game_ended_now(&hist, &board))).unwrap();
      mmal!(hist, board, Location::get_loc(3, 0, board.x_size), P_WHITE);
      writeln!(out, "Score: {}", ds(final_score_if_game_ended_now(&hist, &board))).unwrap();
      mmal!(hist, board, Location::get_loc(1, 2, board.x_size), P_BLACK);
      writeln!(out, "Score: {}", ds(final_score_if_game_ended_now(&hist, &board))).unwrap();
      mmal!(hist, board, Location::get_loc(4, 0, board.x_size), P_WHITE);
      writeln!(out, "Score: {}", ds(final_score_if_game_ended_now(&hist, &board))).unwrap();
      mmal!(hist, board, Board::PASS_LOC, P_BLACK);
      writeln!(out, "Score: {}", ds(final_score_if_game_ended_now(&hist, &board))).unwrap();
      mmal!(hist, board, Location::get_loc(6, 2, board.x_size), P_WHITE);
      writeln!(out, "Score: {}", ds(final_score_if_game_ended_now(&hist, &board))).unwrap();
      mmal!(hist, board, Board::PASS_LOC, P_BLACK);
      writeln!(out, "Score: {}", ds(final_score_if_game_ended_now(&hist, &board))).unwrap();
      mmal!(hist, board, Board::PASS_LOC, P_WHITE);
      testAssert!(hist.is_game_finished);
      writeln!(out, "Score: {}", ds(final_score_if_game_ended_now(&hist, &board))).unwrap();
      out.push('\n');
    }
    expect(name, &mut out, "area-scoring-with-button.expected0.txt");
    cases.ran();
  }
}
}

wasm_test! {
// upstream overwrites hashc twice in a row (testrules.cpp:1794-1796).
#[allow(unused_assignments)]
fn pass_for_ko() {
  init();
  let mut cases = Cases::new("testrules::pass_for_ko");
  let mut out = String::new();
  {
    let name = "Pass for ko";
    let mut board = parse(7, 7, r"
..ox.oo
..oxxxo
...oox.
....oxx
..o.oo.
.......
.......
");
    let mut rules = Rules::new();
    rules.ko_rule = Rules::KO_POSITIONAL;
    rules.scoring_rule = Rules::SCORING_TERRITORY;
    rules.komi = 0.5f32;
    rules.multi_stone_suicide_legal = false;
    rules.tax_rule = Rules::TAX_SEKI;
    let mut hist = BoardHistory::new_full(&board, P_BLACK, &rules, 0);
    let mut hasha: Hash128;
    let hashb: Hash128;
    let mut hashc: Hash128;
    let mut hashd: Hash128;
    let hashe: Hash128;
    let hashf: Hash128;

    mmal!(hist, board, Board::PASS_LOC, P_BLACK);
    mmal!(hist, board, Board::PASS_LOC, P_WHITE);
    testAssert!(hist.encore_phase == 1);
    mmal!(hist, board, Board::PASS_LOC, P_BLACK);
    mmal!(hist, board, Location::get_loc(6, 2, board.x_size), P_WHITE);
    mmal!(hist, board, Location::get_loc(4, 0, board.x_size), P_BLACK);
    mmal!(hist, board, Location::get_loc(6, 1, board.x_size), P_WHITE);
    mmal!(hist, board, Location::get_loc(6, 0, board.x_size), P_BLACK);
    mmal!(hist, board, Location::get_loc(5, 0, board.x_size), P_WHITE);
    writeln!(out, "Black can't retake").unwrap();
    print_illegal_moves(&mut out, &board, &hist, P_BLACK);
    mmal!(hist, board, Location::get_loc(2, 2, board.x_size), P_BLACK);
    mmal!(hist, board, Location::get_loc(1, 2, board.x_size), P_WHITE);
    writeln!(out, "Ko threat shouldn't work in the encore").unwrap();
    print_illegal_moves(&mut out, &board, &hist, P_BLACK);
    mmal!(hist, board, Board::PASS_LOC, P_BLACK);
    mmal!(hist, board, Location::get_loc(0, 6, board.x_size), P_WHITE);
    writeln!(out, "Regular pass shouldn't work in the encore").unwrap();
    print_illegal_moves(&mut out, &board, &hist, P_BLACK);
    writeln!(out, "Pass for ko! (Should not affect the board stones)").unwrap();
    mmal!(hist, board, Location::get_loc(6, 0, board.x_size), P_BLACK);
    out_board(&mut out, &board);
    mmal!(hist, board, Location::get_loc(0, 5, board.x_size), P_WHITE);
    hashd = hist.ko_hash_history[hist.ko_hash_history.len() - 1];
    writeln!(out, "Now black can retake, and white's retake isn't legal").unwrap();
    mmal!(hist, board, Location::get_loc(6, 0, board.x_size), P_BLACK);
    print_illegal_moves(&mut out, &board, &hist, P_WHITE);
    hasha = hist.ko_hash_history[hist.ko_hash_history.len() - 1];
    mmal!(hist, board, Location::get_loc(5, 0, board.x_size), P_WHITE);
    hashb = hist.ko_hash_history[hist.ko_hash_history.len() - 1];
    mmal!(hist, board, Board::PASS_LOC, P_BLACK);
    hashc = hist.ko_hash_history[hist.ko_hash_history.len() - 1];
    testAssert!(hasha != hashb);
    testAssert!(hasha != hashc);
    testAssert!(hashb != hashc);
    writeln!(out, "White's retake is legal after passing for ko").unwrap();
    print_illegal_moves(&mut out, &board, &hist, P_WHITE);
    mmal!(hist, board, Location::get_loc(5, 0, board.x_size), P_WHITE);
    writeln!(out, "Black's retake is illegal again").unwrap();
    print_illegal_moves(&mut out, &board, &hist, P_BLACK);
    mmal!(hist, board, Location::get_loc(6, 0, board.x_size), P_BLACK);
    mmal!(hist, board, Board::PASS_LOC, P_WHITE);
    testAssert!(hashd == hist.ko_hash_history[hist.ko_hash_history.len() - 1]);
    writeln!(out, "And is still illegal due to only-once").unwrap();
    print_illegal_moves(&mut out, &board, &hist, P_BLACK);
    mmal!(hist, board, Location::get_loc(1, 1, board.x_size), P_BLACK);
    mmal!(hist, board, Location::get_loc(2, 3, board.x_size), P_WHITE);
    writeln!(out, "But a ko threat fixes that").unwrap();
    print_illegal_moves(&mut out, &board, &hist, P_BLACK);
    mmal!(hist, board, Location::get_loc(6, 0, board.x_size), P_BLACK);
    writeln!(out, "White illegal now").unwrap();
    print_illegal_moves(&mut out, &board, &hist, P_WHITE);
    testAssert!(hist.encore_phase == 1);
    hasha = hist.ko_hash_history[hist.ko_hash_history.len() - 1];
    mmal!(hist, board, Board::PASS_LOC, P_WHITE);
    hashc = hist.ko_hash_history[hist.ko_hash_history.len() - 1];
    mmal!(hist, board, Board::PASS_LOC, P_BLACK);
    hashc = hist.ko_hash_history[hist.ko_hash_history.len() - 1];
    testAssert!(hist.encore_phase == 2);
    testAssert!(hasha != hashb);
    testAssert!(hasha != hashc);
    testAssert!(hashb != hashc);
    writeln!(out, "Legal again in second encore").unwrap();
    print_illegal_moves(&mut out, &board, &hist, P_WHITE);
    mmal!(hist, board, Location::get_loc(5, 0, board.x_size), P_WHITE);
    writeln!(out, "Lastly, try black ko threat one more time").unwrap();
    mmal!(hist, board, Location::get_loc(1, 0, board.x_size), P_BLACK);
    mmal!(hist, board, Location::get_loc(2, 2, board.x_size), P_WHITE);
    print_illegal_moves(&mut out, &board, &hist, P_BLACK);
    writeln!(out, "And a pass for ko").unwrap();
    hashd = hist.ko_hash_history[hist.ko_hash_history.len() - 1];
    mmal!(hist, board, Location::get_loc(6, 0, board.x_size), P_BLACK);
    hashe = hist.ko_hash_history[hist.ko_hash_history.len() - 1];
    mmal!(hist, board, Board::PASS_LOC, P_WHITE);
    hashf = hist.ko_hash_history[hist.ko_hash_history.len() - 1];
    print_illegal_moves(&mut out, &board, &hist, P_BLACK);
    writeln!(out, "And repeat with white").unwrap();
    mmal!(hist, board, Location::get_loc(6, 0, board.x_size), P_BLACK);
    mmal!(hist, board, Location::get_loc(5, 0, board.x_size), P_WHITE);
    mmal!(hist, board, Board::PASS_LOC, P_BLACK);
    mmal!(hist, board, Location::get_loc(5, 0, board.x_size), P_WHITE);
    testAssert!(hashd == hist.ko_hash_history[hist.ko_hash_history.len() - 1]);
    mmal!(hist, board, Location::get_loc(6, 0, board.x_size), P_BLACK);
    testAssert!(hashe == hist.ko_hash_history[hist.ko_hash_history.len() - 1]);
    mmal!(hist, board, Board::PASS_LOC, P_WHITE);
    testAssert!(hashf == hist.ko_hash_history[hist.ko_hash_history.len() - 1]);
    writeln!(out, "And see the only-once for black").unwrap();
    print_illegal_moves(&mut out, &board, &hist, P_BLACK);

    expect(name, &mut out, "pass-for-ko.expected0.txt");
    cases.ran();
  }
}
}

wasm_test! {
fn two_step_ko_in_encore() {
  init();
  let mut cases = Cases::new("testrules::two_step_ko_in_encore");
  let mut out = String::new();
  {
    let name = "Two step ko in encore";
    let mut board = parse(7, 5, r"
x.x....
.xx....
xox....
ooo....
.......
");
    let mut rules = Rules::new();
    rules.ko_rule = Rules::KO_SITUATIONAL;
    rules.scoring_rule = Rules::SCORING_TERRITORY;
    rules.komi = 0.5f32;
    rules.multi_stone_suicide_legal = true;
    rules.tax_rule = Rules::TAX_SEKI;
    let mut hist = BoardHistory::new_full(&board, P_WHITE, &rules, 0);

    mmal!(hist, board, Board::PASS_LOC, P_WHITE);
    mmal!(hist, board, Board::PASS_LOC, P_BLACK);
    testAssert!(hist.encore_phase == 1);
    mmal!(hist, board, Location::get_loc(0, 1, board.x_size), P_WHITE);
    writeln!(out, "After first cap").unwrap();
    print_illegal_moves(&mut out, &board, &hist, P_BLACK);
    mmal!(hist, board, Board::PASS_LOC, P_BLACK);
    mmal!(hist, board, Location::get_loc(1, 0, board.x_size), P_WHITE);
    writeln!(out, "After second cap").unwrap();
    print_illegal_moves(&mut out, &board, &hist, P_BLACK);
    mmal!(hist, board, Location::get_loc(0, 0, board.x_size), P_BLACK);
    writeln!(out, "Just after black pass for ko").unwrap();
    print_illegal_moves(&mut out, &board, &hist, P_WHITE);
    out_board(&mut out, &board);

    mmal!(hist, board, Board::PASS_LOC, P_WHITE);
    writeln!(out, "After another white pass").unwrap();
    print_illegal_moves(&mut out, &board, &hist, P_BLACK);
    mmal!(hist, board, Location::get_loc(0, 0, board.x_size), P_BLACK);
    writeln!(out, "After first cap").unwrap();
    print_illegal_moves(&mut out, &board, &hist, P_WHITE);
    out_board(&mut out, &board);
    mmal!(hist, board, Board::PASS_LOC, P_WHITE);
    mmal!(hist, board, Location::get_loc(0, 2, board.x_size), P_BLACK);
    writeln!(out, "After second pass for ko").unwrap();
    print_illegal_moves(&mut out, &board, &hist, P_WHITE);
    out_board(&mut out, &board);
    mmal!(hist, board, Board::PASS_LOC, P_WHITE);
    mmal!(hist, board, Location::get_loc(0, 2, board.x_size), P_BLACK);
    writeln!(out, "After second cap").unwrap();
    print_illegal_moves(&mut out, &board, &hist, P_WHITE);
    out_board(&mut out, &board);
    mmal!(hist, board, Location::get_loc(0, 1, board.x_size), P_WHITE);
    mmal!(hist, board, Board::PASS_LOC, P_BLACK);
    writeln!(out, "After pass for ko").unwrap();
    print_illegal_moves(&mut out, &board, &hist, P_WHITE);
    out_board(&mut out, &board);

    expect(name, &mut out, "two-step-ko-in-encore.expected0.txt");
    cases.ran();
  }
}
}

wasm_test! {
fn throw_in_that_destroys_the_ko_momentarily_does_not_clear_ko_recap_block() {
  init();
  let mut cases = Cases::new("testrules::throw_in_that_destroys_the_ko_momentarily_does_not_clear_ko_recap_block");
  let mut out = String::new();
  {
    let name = "Throw in that destroys the ko momentarily does not clear ko recap block";
    let mut board = parse(7, 5, r"
x......
oxx....
.o.....
oo.....
.......
");
    let mut rules = Rules::new();
    rules.ko_rule = Rules::KO_SITUATIONAL;
    rules.scoring_rule = Rules::SCORING_TERRITORY;
    rules.komi = 0.5f32;
    rules.multi_stone_suicide_legal = true;
    rules.tax_rule = Rules::TAX_SEKI;
    let mut hist = BoardHistory::new_full(&board, P_BLACK, &rules, 0);

    mmal!(hist, board, Board::PASS_LOC, P_BLACK);
    mmal!(hist, board, Board::PASS_LOC, P_WHITE);
    mmal!(hist, board, Board::PASS_LOC, P_BLACK);
    mmal!(hist, board, Board::PASS_LOC, P_WHITE);
    testAssert!(hist.encore_phase == 2);
    mmal!(hist, board, Location::get_loc(0, 2, board.x_size), P_BLACK);
    print_illegal_moves(&mut out, &board, &hist, P_WHITE);
    mmal!(hist, board, Location::get_loc(1, 0, board.x_size), P_WHITE);
    mmal!(hist, board, Location::get_loc(2, 0, board.x_size), P_BLACK);
    out_board(&mut out, &board);
    print_illegal_moves(&mut out, &board, &hist, P_WHITE);

    expect(name, &mut out, "throw-in-that-destroys-the-ko-momentarily-does-not-clear-ko-recap-block.expected0.txt");
    cases.ran();
  }
}
}

wasm_test! {
fn various_komis() {
  init();
  let mut cases = Cases::new("testrules::various_komis");
  let mut out = String::new();
  {
    let name = "Various komis";
    let mut board = parse(7, 6, r"
.......
.......
ooooooo
xxxxxxx
.......
.......
");
    let mut rules = Rules::new();
    rules.ko_rule = Rules::KO_SIMPLE;
    rules.scoring_rule = Rules::SCORING_AREA;
    rules.komi = 0.5f32;
    rules.multi_stone_suicide_legal = false;
    rules.tax_rule = Rules::TAX_NONE;
    let mut hist = BoardHistory::new_full(&board, P_BLACK, &rules, 0);

    mmal!(hist, board, Board::PASS_LOC, P_BLACK);
    mmal!(hist, board, Board::PASS_LOC, P_WHITE);
    mmal!(hist, board, Board::PASS_LOC, P_BLACK);
    mmal!(hist, board, Board::PASS_LOC, P_WHITE);
    testAssert!(hist.is_game_finished == true);
    print_game_result(&mut out, &hist);

    hist.set_komi(0.0f32);
    mmal!(hist, board, Board::PASS_LOC, P_BLACK);
    mmal!(hist, board, Board::PASS_LOC, P_WHITE);
    testAssert!(hist.is_game_finished == true);
    print_game_result(&mut out, &hist);

    hist.set_komi(-0.5f32);
    mmal!(hist, board, Board::PASS_LOC, P_BLACK);
    mmal!(hist, board, Board::PASS_LOC, P_WHITE);
    testAssert!(hist.is_game_finished == true);
    print_game_result(&mut out, &hist);

    expect(name, &mut out, "various-komis.expected0.txt");
    cases.ran();
  }
}
}

wasm_test! {
fn group_tax_seki_scoring() {
  init();
  let mut cases = Cases::new("testrules::group_tax_seki_scoring");
  let mut out = String::new();
  {
    let name = "GroupTaxSekiScoring";
    let mut board = parse(9, 9, r"
.x.xo.o.x
...xooox.
.xxxxxxoo
xoooooxo.
xo.o.oxoo
xoooooxxx
xxxo...oo
.xxxoooo.
.x.xo.o.o
");
    let mut rules = Rules::new();
    rules.komi = 0.5f32;
    rules.ko_rule = Rules::KO_POSITIONAL;
    rules.multi_stone_suicide_legal = false;

    {
      rules.scoring_rule = Rules::SCORING_AREA;
      rules.tax_rule = Rules::TAX_NONE;
      let mut hist = BoardHistory::new_full(&board, P_BLACK, &rules, 0);
      mmal!(hist, board, Board::PASS_LOC, P_BLACK);
      mmal!(hist, board, Board::PASS_LOC, P_WHITE);
      testAssert!(hist.is_game_finished == true);
      print_game_result(&mut out, &hist);
    }
    {
      rules.scoring_rule = Rules::SCORING_AREA;
      rules.tax_rule = Rules::TAX_SEKI;
      let mut hist = BoardHistory::new_full(&board, P_BLACK, &rules, 0);
      mmal!(hist, board, Board::PASS_LOC, P_BLACK);
      mmal!(hist, board, Board::PASS_LOC, P_WHITE);
      testAssert!(hist.is_game_finished == true);
      print_game_result(&mut out, &hist);
    }
    {
      rules.scoring_rule = Rules::SCORING_AREA;
      rules.tax_rule = Rules::TAX_ALL;
      let mut hist = BoardHistory::new_full(&board, P_BLACK, &rules, 0);
      mmal!(hist, board, Board::PASS_LOC, P_BLACK);
      mmal!(hist, board, Board::PASS_LOC, P_WHITE);
      testAssert!(hist.is_game_finished == true);
      print_game_result(&mut out, &hist);
    }
    {
      rules.scoring_rule = Rules::SCORING_TERRITORY;
      rules.tax_rule = Rules::TAX_NONE;
      let mut hist = BoardHistory::new_full(&board, P_BLACK, &rules, 0);
      mmal!(hist, board, Board::PASS_LOC, P_BLACK);
      mmal!(hist, board, Board::PASS_LOC, P_WHITE);
      mmal!(hist, board, Board::PASS_LOC, P_BLACK);
      mmal!(hist, board, Board::PASS_LOC, P_WHITE);
      mmal!(hist, board, Board::PASS_LOC, P_BLACK);
      mmal!(hist, board, Board::PASS_LOC, P_WHITE);
      testAssert!(hist.is_game_finished == true);
      print_game_result(&mut out, &hist);
    }
    {
      rules.scoring_rule = Rules::SCORING_TERRITORY;
      rules.tax_rule = Rules::TAX_SEKI;
      let mut hist = BoardHistory::new_full(&board, P_BLACK, &rules, 0);
      mmal!(hist, board, Board::PASS_LOC, P_BLACK);
      mmal!(hist, board, Board::PASS_LOC, P_WHITE);
      mmal!(hist, board, Board::PASS_LOC, P_BLACK);
      mmal!(hist, board, Board::PASS_LOC, P_WHITE);
      mmal!(hist, board, Board::PASS_LOC, P_BLACK);
      mmal!(hist, board, Board::PASS_LOC, P_WHITE);
      testAssert!(hist.is_game_finished == true);
      print_game_result(&mut out, &hist);
    }
    {
      rules.scoring_rule = Rules::SCORING_TERRITORY;
      rules.tax_rule = Rules::TAX_ALL;
      let mut hist = BoardHistory::new_full(&board, P_BLACK, &rules, 0);
      mmal!(hist, board, Board::PASS_LOC, P_BLACK);
      mmal!(hist, board, Board::PASS_LOC, P_WHITE);
      mmal!(hist, board, Board::PASS_LOC, P_BLACK);
      mmal!(hist, board, Board::PASS_LOC, P_WHITE);
      mmal!(hist, board, Board::PASS_LOC, P_BLACK);
      mmal!(hist, board, Board::PASS_LOC, P_WHITE);
      testAssert!(hist.is_game_finished == true);
      print_game_result(&mut out, &hist);
    }

    expect(name, &mut out, "grouptaxsekiscoring.expected0.txt");
    cases.ran();
  }
}
}

wasm_test! {
fn group_tax_seki_scoring2() {
  init();
  let mut cases = Cases::new("testrules::group_tax_seki_scoring2");
  let mut out = String::new();
  {
    let name = "GroupTaxSekiScoring2";
    let mut board = parse(9, 9, r"
.x.xo.o.x
...xooox.
.xxxxxxoo
xoooooxo.
xo.o.oxoo
xoooooxxx
xxxoxxxoo
.xxxoooo.
.x.xo.o.o
");
    let mut rules = Rules::new();
    rules.komi = 0.5f32;
    rules.ko_rule = Rules::KO_POSITIONAL;
    rules.multi_stone_suicide_legal = false;

    {
      rules.scoring_rule = Rules::SCORING_AREA;
      rules.tax_rule = Rules::TAX_NONE;
      let mut hist = BoardHistory::new_full(&board, P_BLACK, &rules, 0);
      mmal!(hist, board, Board::PASS_LOC, P_BLACK);
      mmal!(hist, board, Board::PASS_LOC, P_WHITE);
      testAssert!(hist.is_game_finished == true);
      print_game_result(&mut out, &hist);
    }
    {
      rules.scoring_rule = Rules::SCORING_AREA;
      rules.tax_rule = Rules::TAX_SEKI;
      let mut hist = BoardHistory::new_full(&board, P_BLACK, &rules, 0);
      mmal!(hist, board, Board::PASS_LOC, P_BLACK);
      mmal!(hist, board, Board::PASS_LOC, P_WHITE);
      testAssert!(hist.is_game_finished == true);
      print_game_result(&mut out, &hist);
    }
    {
      rules.scoring_rule = Rules::SCORING_AREA;
      rules.tax_rule = Rules::TAX_ALL;
      let mut hist = BoardHistory::new_full(&board, P_BLACK, &rules, 0);
      mmal!(hist, board, Board::PASS_LOC, P_BLACK);
      mmal!(hist, board, Board::PASS_LOC, P_WHITE);
      testAssert!(hist.is_game_finished == true);
      print_game_result(&mut out, &hist);
    }
    {
      rules.scoring_rule = Rules::SCORING_TERRITORY;
      rules.tax_rule = Rules::TAX_NONE;
      let mut hist = BoardHistory::new_full(&board, P_BLACK, &rules, 0);
      mmal!(hist, board, Board::PASS_LOC, P_BLACK);
      mmal!(hist, board, Board::PASS_LOC, P_WHITE);
      mmal!(hist, board, Board::PASS_LOC, P_BLACK);
      mmal!(hist, board, Board::PASS_LOC, P_WHITE);
      mmal!(hist, board, Board::PASS_LOC, P_BLACK);
      mmal!(hist, board, Board::PASS_LOC, P_WHITE);
      testAssert!(hist.is_game_finished == true);
      print_game_result(&mut out, &hist);
    }
    {
      rules.scoring_rule = Rules::SCORING_TERRITORY;
      rules.tax_rule = Rules::TAX_SEKI;
      let mut hist = BoardHistory::new_full(&board, P_BLACK, &rules, 0);
      mmal!(hist, board, Board::PASS_LOC, P_BLACK);
      mmal!(hist, board, Board::PASS_LOC, P_WHITE);
      mmal!(hist, board, Board::PASS_LOC, P_BLACK);
      mmal!(hist, board, Board::PASS_LOC, P_WHITE);
      mmal!(hist, board, Board::PASS_LOC, P_BLACK);
      mmal!(hist, board, Board::PASS_LOC, P_WHITE);
      testAssert!(hist.is_game_finished == true);
      print_game_result(&mut out, &hist);
    }
    {
      rules.scoring_rule = Rules::SCORING_TERRITORY;
      rules.tax_rule = Rules::TAX_ALL;
      let mut hist = BoardHistory::new_full(&board, P_BLACK, &rules, 0);
      mmal!(hist, board, Board::PASS_LOC, P_BLACK);
      mmal!(hist, board, Board::PASS_LOC, P_WHITE);
      mmal!(hist, board, Board::PASS_LOC, P_BLACK);
      mmal!(hist, board, Board::PASS_LOC, P_WHITE);
      mmal!(hist, board, Board::PASS_LOC, P_BLACK);
      mmal!(hist, board, Board::PASS_LOC, P_WHITE);
      testAssert!(hist.is_game_finished == true);
      print_game_result(&mut out, &hist);
    }

    expect(name, &mut out, "grouptaxsekiscoring2.expected0.txt");
    cases.ran();
  }
}
}

wasm_test! {
fn prevent_encore() {
  init();
  let mut cases = Cases::new("testrules::prevent_encore");
  let mut out = String::new();
  {
    let name = "PreventEncore";
    let mut board = parse(3, 3, r"
.x.
xxo
.o.
");
    let mut rules = Rules::new();
    rules.komi = 0.5f32;
    rules.ko_rule = Rules::KO_POSITIONAL;
    rules.multi_stone_suicide_legal = false;
    rules.scoring_rule = Rules::SCORING_TERRITORY;
    rules.tax_rule = Rules::TAX_NONE;

    {
      let mut hist = BoardHistory::new_full(&board, P_BLACK, &rules, 0);
      mmal!(hist, board, Board::PASS_LOC, P_BLACK);
      mmal!(hist, board, Board::PASS_LOC, P_WHITE);
      hist.print_debug_info(&mut out, &board);
      mmal!(hist, board, Board::PASS_LOC, P_BLACK);
      hist.print_debug_info(&mut out, &board);
      mmal!(hist, board, Board::PASS_LOC, P_WHITE);
      hist.print_debug_info(&mut out, &board);
      mmal!(hist, board, Board::PASS_LOC, P_BLACK);
      hist.print_debug_info(&mut out, &board);
      mmal!(hist, board, Board::PASS_LOC, P_WHITE);
      hist.print_debug_info(&mut out, &board);
      mbmal!(hist, board, Board::PASS_LOC, P_BLACK, None);
      hist.print_debug_info(&mut out, &board);
      out.push('\n');
    }
    {
      out.push_str("-----------------------\n");
      out.push_str("Preventing encore\n");
      let mut hist = BoardHistory::new_full(&board, P_BLACK, &rules, 0);
      mmal!(hist, board, Board::PASS_LOC, P_BLACK, pe = true);
      mmal!(hist, board, Board::PASS_LOC, P_WHITE, pe = true);
      hist.print_debug_info(&mut out, &board);
      mmal!(hist, board, Board::PASS_LOC, P_BLACK, pe = true);
      hist.print_debug_info(&mut out, &board);
      mmal!(hist, board, Board::PASS_LOC, P_WHITE, pe = true);
      hist.print_debug_info(&mut out, &board);
    }

    expect(name, &mut out, "preventencore.expected0.txt");
    cases.ran();
  }
}
}

/// upstream: the `stressTest` lambda in the "Stress test on tiny boards" block of Tests::runRulesTests
/// PORT NOTE: captures (out, baseRand) passed explicitly (PORTING.md §5).
fn stress_test(out: &mut String, base_rand: &mut Rand, mut board: Board, mut hist: BoardHistory, mut next_pla: Player, prolong_game: bool) {
  let mut rand = Rand::new_seed_u64(base_rand.next_uint64());
  for _i in 0..1000 {
    let mut num_legal: i32 = 0;
    const MAX_LEGAL_MOVES: i32 = Board::MAX_PLAY_SIZE + 1;
    let mut legal_moves: [Loc; MAX_LEGAL_MOVES as usize] = [0; MAX_LEGAL_MOVES as usize];
    let mut mv: Loc;

    for y in 0..board.y_size {
      for x in 0..board.x_size {
        mv = Location::get_loc(x, y, board.x_size);
        if hist.is_legal(&board, mv, next_pla) {
          legal_moves[num_legal as usize] = mv;
          num_legal += 1;
        }
      }
    }
    mv = Board::PASS_LOC;
    if hist.is_legal(&board, mv, next_pla) {
      legal_moves[num_legal as usize] = mv;
      num_legal += 1;
    }

    out.push_str(&num_legal.to_string());
    out.push(' ');
    for y in 0..board.y_size {
      for x in 0..board.x_size {
        out.push(PlayerIO::color_to_char(board.colors[Location::get_loc(x, y, board.x_size) as usize]));
      }
    }
    out.push_str(" NP");
    out.push(PlayerIO::color_to_char(next_pla));
    out.push_str(&format!(" PS{}", hist.consecutive_ending_passes));
    out.push_str(&format!(" E{}", hist.encore_phase));
    out.push(' ');
    out.push(' ');
    for y in 0..board.y_size {
      for x in 0..board.x_size {
        out.push_str(&(hist.ko_recap_blocked[Location::get_loc(x, y, board.x_size) as usize] as i32).to_string());
      }
    }
    out.push(' ');
    for y in 0..board.y_size {
      for x in 0..board.x_size {
        out.push_str(&(hist.second_encore_start_colors[Location::get_loc(x, y, board.x_size) as usize] as i32).to_string());
      }
    }

    out.push('\n');

    if hist.is_game_finished {
      break;
    }

    testAssert!(num_legal > 0);
    mv = legal_moves[rand.next_uint_n(num_legal as u32) as usize];
    if prolong_game && mv == Board::PASS_LOC {
      mv = legal_moves[rand.next_uint_n(num_legal as u32) as usize];
    }
    mmal!(hist, board, mv, next_pla);
    next_pla = get_opp(next_pla);
  }
  out.push_str(&format!("White bonus score {}\n", fs(hist.white_bonus_score)));
  print_game_result(out, &hist);
}

wasm_test! {
fn stress_test_on_tiny_boards() {
  init();
  let mut cases = Cases::new("testrules::stress_test_on_tiny_boards");
  let mut out = String::new();
  {
    let name = "Stress test on tiny boards";

    let mut base_rand = Rand::new_seed("Tiny board stress test");

    let empty_board22 = parse(2, 2, r"
..
..
");

    let mut rules = Rules::new();

    rules.ko_rule = Rules::KO_SIMPLE;
    rules.scoring_rule = Rules::SCORING_TERRITORY;
    rules.komi = 0.5f32;
    rules.tax_rule = Rules::TAX_SEKI;
    rules.multi_stone_suicide_legal = false;
    stress_test(&mut out, &mut base_rand, empty_board22.clone(), BoardHistory::new_full(&empty_board22, P_BLACK, &rules, 0), P_BLACK, true);
    rules.multi_stone_suicide_legal = true;
    stress_test(&mut out, &mut base_rand, empty_board22.clone(), BoardHistory::new_full(&empty_board22, P_BLACK, &rules, 0), P_BLACK, true);

    expect(name, &mut out, "stress-test-on-tiny-boards.expected0.txt");
    cases.ran();

    rules.ko_rule = Rules::KO_SIMPLE;
    rules.scoring_rule = Rules::SCORING_AREA;
    rules.komi = 0.5f32;
    rules.multi_stone_suicide_legal = false;
    rules.tax_rule = Rules::TAX_NONE;
    stress_test(&mut out, &mut base_rand, empty_board22.clone(), BoardHistory::new_full(&empty_board22, P_BLACK, &rules, 0), P_BLACK, false);
    stress_test(&mut out, &mut base_rand, empty_board22.clone(), BoardHistory::new_full(&empty_board22, P_BLACK, &rules, 0), P_BLACK, false);
    rules.multi_stone_suicide_legal = true;
    stress_test(&mut out, &mut base_rand, empty_board22.clone(), BoardHistory::new_full(&empty_board22, P_BLACK, &rules, 0), P_BLACK, false);
    stress_test(&mut out, &mut base_rand, empty_board22.clone(), BoardHistory::new_full(&empty_board22, P_BLACK, &rules, 0), P_BLACK, false);

    expect(name, &mut out, "stress-test-on-tiny-boards.expected1.txt");
    cases.ran();

    let ko_board71 = parse(7, 1, r"
.o.ox.o
");
    let ko_board41 = parse(4, 1, r"
....
");

    rules.ko_rule = Rules::KO_SIMPLE;
    rules.scoring_rule = Rules::SCORING_TERRITORY;
    rules.komi = 0.5f32;
    rules.multi_stone_suicide_legal = false;
    rules.tax_rule = Rules::TAX_SEKI;
    stress_test(&mut out, &mut base_rand, ko_board71.clone(), BoardHistory::new_full(&ko_board71, P_BLACK, &rules, 0), P_BLACK, true);

    expect(name, &mut out, "stress-test-on-tiny-boards.expected2.txt");
    cases.ran();

    rules.ko_rule = Rules::KO_POSITIONAL;
    rules.scoring_rule = Rules::SCORING_TERRITORY;
    rules.komi = 0.5f32;
    rules.multi_stone_suicide_legal = false;
    rules.tax_rule = Rules::TAX_SEKI;
    stress_test(&mut out, &mut base_rand, ko_board41.clone(), BoardHistory::new_full(&ko_board41, P_BLACK, &rules, 0), P_BLACK, true);
    expect(name, &mut out, "stress-test-on-tiny-boards.expected3.txt");
    cases.ran();

    rules.ko_rule = Rules::KO_SITUATIONAL;
    rules.scoring_rule = Rules::SCORING_TERRITORY;
    rules.komi = 0.5f32;
    rules.multi_stone_suicide_legal = false;
    rules.tax_rule = Rules::TAX_SEKI;
    stress_test(&mut out, &mut base_rand, ko_board41.clone(), BoardHistory::new_full(&ko_board41, P_BLACK, &rules, 0), P_BLACK, true);
    expect(name, &mut out, "stress-test-on-tiny-boards.expected4.txt");
    cases.ran();


    rules.ko_rule = Rules::KO_SIMPLE;
    rules.scoring_rule = Rules::SCORING_AREA;
    rules.komi = 0.5f32;
    rules.multi_stone_suicide_legal = false;
    rules.tax_rule = Rules::TAX_NONE;
    stress_test(&mut out, &mut base_rand, ko_board41.clone(), BoardHistory::new_full(&ko_board41, P_BLACK, &rules, 0), P_BLACK, true);

    expect(name, &mut out, "stress-test-on-tiny-boards.expected5.txt");
    cases.ran();

    rules.ko_rule = Rules::KO_POSITIONAL;
    rules.scoring_rule = Rules::SCORING_AREA;
    rules.komi = 0.5f32;
    rules.multi_stone_suicide_legal = false;
    rules.tax_rule = Rules::TAX_NONE;
    stress_test(&mut out, &mut base_rand, ko_board41.clone(), BoardHistory::new_full(&ko_board41, P_BLACK, &rules, 0), P_BLACK, true);
    expect(name, &mut out, "stress-test-on-tiny-boards.expected6.txt");
    cases.ran();

    rules.ko_rule = Rules::KO_SITUATIONAL;
    rules.scoring_rule = Rules::SCORING_AREA;
    rules.komi = 0.5f32;
    rules.multi_stone_suicide_legal = false;
    rules.tax_rule = Rules::TAX_NONE;
    stress_test(&mut out, &mut base_rand, ko_board41.clone(), BoardHistory::new_full(&ko_board41, P_BLACK, &rules, 0), P_BLACK, true);
    expect(name, &mut out, "stress-test-on-tiny-boards.expected7.txt");
    cases.ran();


    rules.ko_rule = Rules::KO_POSITIONAL;
    rules.scoring_rule = Rules::SCORING_TERRITORY;
    rules.komi = 0.5f32;
    rules.multi_stone_suicide_legal = false;
    rules.tax_rule = Rules::TAX_SEKI;
    base_rand.init_seed("123");
    stress_test(&mut out, &mut base_rand, ko_board71.clone(), BoardHistory::new_full(&ko_board71, P_BLACK, &rules, 0), P_BLACK, false);
    expect(name, &mut out, "stress-test-on-tiny-boards.expected8.txt");
    cases.ran();
  }
}
}

/// upstream: the `compareHists` lambda in the "Board history clearing directly to the encore" block
/// PORT NOTE: captures (out, hist, hist2, board) passed explicitly (PORTING.md §5).
fn compare_hists(out: &mut String, hist: &BoardHistory, hist2: &BoardHistory, board: &Board) {
  out.push_str(&format!("{} {}\n", hist.move_history.len(), hist2.move_history.len()));
  out.push_str(&format!("{} {}\n", hist.ko_hash_history.len(), hist2.ko_hash_history.len()));
  out.push_str(&format!("{} {}\n", hist.ko_hash_history[0], hist2.ko_hash_history[0]));
  out.push_str(&format!("{} {}\n", hist.first_turn_idx_with_ko_history, hist2.first_turn_idx_with_ko_history));
  out.push_str(&format!("{} {}\n", hist.get_recent_board(0).pos_hash, hist2.get_recent_board(0).pos_hash));
  out.push_str(&format!("{} {}\n", hist.get_recent_board(1).pos_hash, hist2.get_recent_board(1).pos_hash));
  out.push_str(&format!("{} {}\n", hist.get_recent_board(2).pos_hash, hist2.get_recent_board(2).pos_hash));
  out.push_str(&format!("{} {}\n", hist.get_recent_board(3).pos_hash, hist2.get_recent_board(3).pos_hash));
  out.push_str(&format!("{} {}\n", hist.get_recent_board(4).pos_hash, hist2.get_recent_board(4).pos_hash));
  out.push_str(&format!("{} {}\n", hist.get_recent_board(5).pos_hash, hist2.get_recent_board(5).pos_hash));

  for i in 0..Board::MAX_ARR_SIZE as usize {
    testAssert!(hist.was_ever_occupied_or_played[i] == hist2.was_ever_occupied_or_played[i]);
  }
  for i in 0..Board::MAX_ARR_SIZE as usize {
    testAssert!(hist.super_ko_banned[i] == false);
  }
  for i in 0..Board::MAX_ARR_SIZE as usize {
    testAssert!(hist2.super_ko_banned[i] == false);
  }

  out.push_str(&format!("{} {}\n", hist.consecutive_ending_passes, hist2.consecutive_ending_passes));
  out.push_str(&format!("{} {}\n", hist.hashes_before_black_pass.len(), hist2.hashes_before_black_pass.len()));
  out.push_str(&format!("{} {}\n", hist.hashes_before_white_pass.len(), hist2.hashes_before_white_pass.len()));
  out.push_str(&format!("{} {}\n", hist.encore_phase, hist2.encore_phase));

  for i in 0..Board::MAX_ARR_SIZE as usize {
    testAssert!(hist.ko_recap_blocked[i] == false);
  }
  for i in 0..Board::MAX_ARR_SIZE as usize {
    testAssert!(hist2.ko_recap_blocked[i] == false);
  }

  out.push_str(&format!("{} {}\n", hist.ko_recap_block_hash, hist2.ko_recap_block_hash));
  out.push_str(&format!("{} {}\n", hist.ko_captures_in_encore.len(), hist2.ko_captures_in_encore.len()));

  for y in 0..board.y_size {
    for x in 0..board.x_size {
      out.push_str(&(hist.second_encore_start_colors[Location::get_loc(x, y, board.x_size) as usize] as i32).to_string());
    }
  }
  out.push('\n');
  for y in 0..board.y_size {
    for x in 0..board.x_size {
      out.push_str(&(hist2.second_encore_start_colors[Location::get_loc(x, y, board.x_size) as usize] as i32).to_string());
    }
  }
  out.push('\n');

  out.push_str(&format!("{} {}\n", fs(hist.white_bonus_score), fs(hist2.white_bonus_score)));
  out.push_str(&format!("{} {}\n", hist.is_game_finished as i32, hist2.is_game_finished as i32));
  out.push_str(&format!("{} {}\n", hist.winner as i32, hist2.winner as i32));
  out.push_str(&format!("{} {}\n", fs(hist.final_white_minus_black_score), fs(hist2.final_white_minus_black_score)));
  out.push_str(&format!("{} {}\n", hist.is_no_result as i32, hist2.is_no_result as i32));
}

wasm_test! {
fn board_history_clearing_directly_to_the_encore() {
  init();
  let mut cases = Cases::new("testrules::board_history_clearing_directly_to_the_encore");
  let mut out = String::new();
  {
    let name = "Board history clearing directly to the encore";
    let board = parse(4, 4, r"
..o.
.o.o
.xox
..xx
");
    let mut rules = Rules::new();
    rules.ko_rule = Rules::KO_SIMPLE;
    rules.scoring_rule = Rules::SCORING_TERRITORY;
    rules.komi = 0.5f32;
    rules.multi_stone_suicide_legal = true;
    rules.tax_rule = Rules::TAX_SEKI;
    let mut hist = BoardHistory::new_full(&board, P_BLACK, &rules, 0);
    let mut hist2 = BoardHistory::new_full(&board, P_BLACK, &rules, 0);

    let mut copy = board.clone();
    mmal!(hist, copy, Board::PASS_LOC, P_BLACK);
    mmal!(hist, copy, Board::PASS_LOC, P_WHITE);

    let hist2_rules = hist2.rules;
    hist2.clear(&board, P_BLACK, &hist2_rules, 1);

    compare_hists(&mut out, &hist, &hist2, &board);
    expect(name, &mut out, "board-history-clearing-directly-to-the-encore.expected0.txt");
    cases.ran();

    mmal!(hist, copy, Board::PASS_LOC, P_BLACK);
    mmal!(hist, copy, Board::PASS_LOC, P_WHITE);

    let hist2_rules = hist2.rules;
    hist2.clear(&board, P_BLACK, &hist2_rules, 2);

    compare_hists(&mut out, &hist, &hist2, &board);
    expect(name, &mut out, "board-history-clearing-directly-to-the-encore.expected1.txt");
    cases.ran();
  }
}
}

wasm_test! {
fn test_case_failing_in_search_before() {
  init();
  let mut cases = Cases::new("testrules::test_case_failing_in_search_before");
  let mut out = String::new();
  {
    let name = "Test case failing in search before";
    let mut board = parse(9, 9, r"
XXXXXXXXX
X.OXXXXXX
XXXXOXXXX
XXX.OOXX.
OXXXOOXXX
.OXXXXXXO
XXXX.XOOO
XXXOXOOOO
XXXOO.OOO
");
    let mut rules = Rules::new();
    rules.ko_rule = Rules::KO_SIMPLE;
    rules.scoring_rule = Rules::SCORING_TERRITORY;
    rules.komi = 0.5f32;
    rules.multi_stone_suicide_legal = false;
    rules.tax_rule = Rules::TAX_SEKI;
    let mut hist = BoardHistory::new_full(&board, P_BLACK, &rules, 0);

    mmal!(hist, board, Board::PASS_LOC, P_BLACK);
    mmal!(hist, board, Board::PASS_LOC, P_WHITE);

    testAssert!(hist.encore_phase == 1);
    mmal!(hist, board, Location::get_loc(8, 3, board.x_size), P_BLACK);
    mmal!(hist, board, Location::get_loc(4, 6, board.x_size), P_WHITE);
    mmal!(hist, board, Location::get_loc(4, 7, board.x_size), P_BLACK);
    mmal!(hist, board, Location::get_loc(4, 7, board.x_size), P_WHITE);
    out_board(&mut out, &board);

    expect(name, &mut out, "test-case-failing-in-search-before.expected0.txt");
    cases.ran();
  }
}
}

// "Test basic game" (testrules.cpp lines 3459-3531)
// TODO(port): needs dataio/sgf (CompactSgf::parse, setupBoardAndHistAssumeLegal), not in M1.
// Upstream parses a 12x12 SGF (RU[koSIMPLEscoreTERRITORYtaxSEKIsui1], KM[7.5], RE[B+1.5]) whose
// game ends in the second encore, sets up board+hist for all moves, prints hist.printDebugInfo
// and the area from hist.endAndScoreGameNow(board,area), and compares with
// test-basic-game.expected0.txt (fixture kept).

wasm_test! {
fn sending_two_returning_one_initial_pass_without_button_simple() {
  init();
  let mut cases = Cases::new("testrules::sending_two_returning_one_initial_pass_without_button_simple");
  let mut out = String::new();
  {
    let name = "Sending two returning one initial pass WITHOUT button, simple";

    let mut board = parse(5, 6, r"
.....
..xxx
xx.oo
xooo.
xo.ox
xoxx.
");
    let mut rules = Rules::new();
    rules.ko_rule = Rules::KO_SIMPLE;
    rules.scoring_rule = Rules::SCORING_AREA;
    rules.tax_rule = Rules::TAX_SEKI;
    rules.multi_stone_suicide_legal = false;
    rules.komi = 0.0f32;
    rules.has_button = false;
    let mut hist = BoardHistory::new_full(&board, P_BLACK, &rules, 0);

    mmal!(hist, board, Location::get_loc(2, 2, board.x_size), P_BLACK);
    mmal!(hist, board, Board::PASS_LOC, P_WHITE);
    mmal!(hist, board, Location::get_loc(4, 3, board.x_size), P_BLACK);
    mmal!(hist, board, Location::get_loc(4, 5, board.x_size), P_WHITE);
    mmal!(hist, board, Location::get_loc(4, 4, board.x_size), P_BLACK);
    testAssert!(!hist.is_game_finished);
    mmal!(hist, board, Board::PASS_LOC, P_WHITE);
    testAssert!(hist.is_game_finished);
    print_game_result(&mut out, &hist);
    out.push('\n');
    expect(name, &mut out, "sending-two-returning-one-initial-pass-without-button-simple.expected0.txt");
    cases.ran();
  }
}
}

wasm_test! {
fn sending_two_returning_one_initial_pass_with_button_simple() {
  init();
  let mut cases = Cases::new("testrules::sending_two_returning_one_initial_pass_with_button_simple");
  let mut out = String::new();
  {
    let name = "Sending two returning one initial pass WITH button, simple";

    let mut board = parse(5, 6, r"
.....
..xxx
xx.oo
xooo.
xo.ox
xoxx.
");
    let mut rules = Rules::new();
    rules.ko_rule = Rules::KO_SIMPLE;
    rules.scoring_rule = Rules::SCORING_AREA;
    rules.tax_rule = Rules::TAX_SEKI;
    rules.multi_stone_suicide_legal = false;
    rules.komi = 0.5f32;
    rules.has_button = true;
    let mut hist = BoardHistory::new_full(&board, P_BLACK, &rules, 0);

    mmal!(hist, board, Location::get_loc(2, 2, board.x_size), P_BLACK);
    mmal!(hist, board, Board::PASS_LOC, P_WHITE);
    mmal!(hist, board, Location::get_loc(4, 3, board.x_size), P_BLACK);
    mmal!(hist, board, Location::get_loc(4, 5, board.x_size), P_WHITE);
    mmal!(hist, board, Location::get_loc(4, 4, board.x_size), P_BLACK);
    testAssert!(!hist.is_game_finished);
    mmal!(hist, board, Board::PASS_LOC, P_WHITE);
    testAssert!(!hist.is_game_finished);
    mmal!(hist, board, Location::get_loc(4, 3, board.x_size), P_BLACK);
    mmal!(hist, board, Location::get_loc(4, 5, board.x_size), P_WHITE);
    mmal!(hist, board, Location::get_loc(4, 4, board.x_size), P_BLACK);
    testAssert!(!hist.is_game_finished);
    mmal!(hist, board, Board::PASS_LOC, P_WHITE);
    testAssert!(hist.is_game_finished);
    print_game_result(&mut out, &hist);
    out.push('\n');
    expect(name, &mut out, "sending-two-returning-one-initial-pass-with-button-simple.expected0.txt");
    cases.ran();
  }
}
}

wasm_test! {
fn sending_two_returning_one_no_initial_pass_without_button_ssk() {
  init();
  let mut cases = Cases::new("testrules::sending_two_returning_one_no_initial_pass_without_button_ssk");
  let mut out = String::new();
  {
    let name = "Sending two returning one NO initial pass WITHOUT button, SSK";

    let mut board = parse(5, 6, r"
.....
..xxx
xxxoo
xooo.
xo.ox
xoxx.
");
    let mut rules = Rules::new();
    rules.ko_rule = Rules::KO_SITUATIONAL;
    rules.scoring_rule = Rules::SCORING_AREA;
    rules.tax_rule = Rules::TAX_SEKI;
    rules.multi_stone_suicide_legal = false;
    rules.komi = 0.0f32;
    rules.has_button = false;
    let mut hist = BoardHistory::new_full(&board, P_BLACK, &rules, 0);

    mmal!(hist, board, Location::get_loc(4, 3, board.x_size), P_BLACK);
    mmal!(hist, board, Location::get_loc(4, 5, board.x_size), P_WHITE);
    mmal!(hist, board, Location::get_loc(4, 4, board.x_size), P_BLACK);
    mmal!(hist, board, Board::PASS_LOC, P_WHITE);
    print_illegal_moves(&mut out, &board, &hist, P_BLACK);
    testAssert!(!hist.is_game_finished);
    out.push('\n');
    expect(name, &mut out, "sending-two-returning-one-no-initial-pass-without-button-ssk.expected0.txt");
    cases.ran();
  }
}
}

wasm_test! {
fn sending_two_returning_one_no_initial_pass_with_button_ssk() {
  init();
  let mut cases = Cases::new("testrules::sending_two_returning_one_no_initial_pass_with_button_ssk");
  let mut out = String::new();
  {
    let name = "Sending two returning one NO initial pass WITH button, SSK";

    let mut board = parse(5, 6, r"
.....
..xxx
xxxoo
xooo.
xo.ox
xoxx.
");
    let mut rules = Rules::new();
    rules.ko_rule = Rules::KO_SITUATIONAL;
    rules.scoring_rule = Rules::SCORING_AREA;
    rules.tax_rule = Rules::TAX_SEKI;
    rules.multi_stone_suicide_legal = false;
    rules.komi = 0.5f32;
    rules.has_button = true;
    let mut hist = BoardHistory::new_full(&board, P_BLACK, &rules, 0);

    mmal!(hist, board, Location::get_loc(4, 3, board.x_size), P_BLACK);
    mmal!(hist, board, Location::get_loc(4, 5, board.x_size), P_WHITE);
    mmal!(hist, board, Location::get_loc(4, 4, board.x_size), P_BLACK);
    mmal!(hist, board, Board::PASS_LOC, P_WHITE);
    print_illegal_moves(&mut out, &board, &hist, P_BLACK);
    testAssert!(!hist.is_game_finished);
    out.push_str("--\n");
    testAssert!(!hist.is_game_finished);
    mmal!(hist, board, Location::get_loc(4, 3, board.x_size), P_BLACK);
    mmal!(hist, board, Location::get_loc(4, 5, board.x_size), P_WHITE);
    mmal!(hist, board, Location::get_loc(4, 4, board.x_size), P_BLACK);
    mmal!(hist, board, Board::PASS_LOC, P_WHITE);
    print_illegal_moves(&mut out, &board, &hist, P_BLACK);
    out.push('\n');
    expect(name, &mut out, "sending-two-returning-one-no-initial-pass-with-button-ssk.expected0.txt");
    cases.ran();
  }
}
}

wasm_test! {
fn sending_two_returning_one_initial_pass_with_button_ssk() {
  init();
  let mut cases = Cases::new("testrules::sending_two_returning_one_initial_pass_with_button_ssk");
  let mut out = String::new();
  {
    let name = "Sending two returning one initial pass WITH button, SSK";

    let mut board = parse(5, 6, r"
.....
..xxx
xx.oo
xooo.
xo.ox
xoxx.
");
    let mut rules = Rules::new();
    rules.ko_rule = Rules::KO_SITUATIONAL;
    rules.scoring_rule = Rules::SCORING_AREA;
    rules.tax_rule = Rules::TAX_SEKI;
    rules.multi_stone_suicide_legal = false;
    rules.komi = 0.5f32;
    rules.has_button = true;
    let mut hist = BoardHistory::new_full(&board, P_BLACK, &rules, 0);

    mmal!(hist, board, Location::get_loc(2, 2, board.x_size), P_BLACK);
    mmal!(hist, board, Board::PASS_LOC, P_WHITE);
    mmal!(hist, board, Location::get_loc(4, 3, board.x_size), P_BLACK);
    mmal!(hist, board, Location::get_loc(4, 5, board.x_size), P_WHITE);
    mmal!(hist, board, Location::get_loc(4, 4, board.x_size), P_BLACK);
    mmal!(hist, board, Board::PASS_LOC, P_WHITE);
    print_illegal_moves(&mut out, &board, &hist, P_BLACK);
    testAssert!(!hist.is_game_finished);
    out.push('\n');
    expect(name, &mut out, "sending-two-returning-one-initial-pass-with-button-ssk.expected0.txt");
    cases.ran();
  }
}
}


wasm_test! {
fn double_ko_death_1a() {
  init();
  let mut cases = Cases::new("testrules::double_ko_death_1a");
  let mut out = String::new();
  {
    let name = "Double ko death 1a";

    let mut board = parse(9, 9, r"
   A B C D E F G H J
 9 . . . . X O X . X
 8 . . . . X O O X X
 7 . . X . X X O X O
 6 . . . . O X O O .
 5 . . . O O X X O O
 4 . X X X X . X O .
 3 X X O O O X X O O
 2 O O O X O O X X X
 1 . . . . . O O O O
");
    let mut rules = Rules::new();
    rules.ko_rule = Rules::KO_SITUATIONAL;
    rules.scoring_rule = Rules::SCORING_TERRITORY;
    rules.tax_rule = Rules::TAX_NONE;
    rules.multi_stone_suicide_legal = false;
    rules.komi = 6.5f32;
    rules.has_button = false;
    let mut hist = BoardHistory::new_full(&board, P_WHITE, &rules, 0);

    mmal!(hist, board, Board::PASS_LOC, P_WHITE);
    mmal!(hist, board, Board::PASS_LOC, P_BLACK);
    mmal!(hist, board, Board::PASS_LOC, P_WHITE);
    mmal!(hist, board, Location::get_loc(8, 3, board.x_size), P_BLACK);
    mmal!(hist, board, Location::get_loc(7, 0, board.x_size), P_WHITE);
    out_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    mmal!(hist, board, Location::get_loc(7, 0, board.x_size), P_BLACK);
    out_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    mmal!(hist, board, Location::get_loc(8, 3, board.x_size), P_WHITE);
    out_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    testAssert!(!hist.is_game_finished);
    hist.print_debug_info(&mut out, &board);
    out.push('\n');
    expect(name, &mut out, "double-ko-death-1a.expected0.txt");
    cases.ran();
  }
}
}

wasm_test! {
fn double_ko_death_1b() {
  init();
  let mut cases = Cases::new("testrules::double_ko_death_1b");
  let mut out = String::new();
  {
    let name = "Double ko death 1b";

    let mut board = parse(9, 9, r"
   A B C D E F G H J
 9 . . . . X O X . X
 8 . . . . X O O X X
 7 . . X . X X O X O
 6 . . . . O X O O .
 5 . . . O O X X O O
 4 . X X X X . X O .
 3 X X O O O X X O O
 2 O O O X O O X X X
 1 . . . . . O O O O
");
    let mut rules = Rules::new();
    rules.ko_rule = Rules::KO_SITUATIONAL;
    rules.scoring_rule = Rules::SCORING_TERRITORY;
    rules.tax_rule = Rules::TAX_NONE;
    rules.multi_stone_suicide_legal = false;
    rules.komi = 6.5f32;
    rules.has_button = false;
    let mut hist = BoardHistory::new_full(&board, P_WHITE, &rules, 0);

    mmal!(hist, board, Board::PASS_LOC, P_WHITE);
    mmal!(hist, board, Board::PASS_LOC, P_BLACK);
    mmal!(hist, board, Board::PASS_LOC, P_WHITE);
    mmal!(hist, board, Location::get_loc(8, 3, board.x_size), P_BLACK);
    mmal!(hist, board, Location::get_loc(7, 0, board.x_size), P_WHITE);
    out_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    mmal!(hist, board, Location::get_loc(7, 0, board.x_size), P_BLACK);
    out_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    mmal!(hist, board, Location::get_loc(8, 2, board.x_size), P_WHITE);
    out_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    testAssert!(!hist.is_game_finished);
    hist.print_debug_info(&mut out, &board);
    out.push('\n');
    expect(name, &mut out, "double-ko-death-1b.expected0.txt");
    cases.ran();
  }
}
}

wasm_test! {
fn double_ko_death_1c() {
  init();
  let mut cases = Cases::new("testrules::double_ko_death_1c");
  let mut out = String::new();
  {
    let name = "Double ko death 1c";

    let mut board = parse(9, 9, r"
   A B C D E F G H J
 9 . . . . X O X . X
 8 . . . . X O O X X
 7 . . X . X X O X O
 6 . . . . O X O O .
 5 . . . O O X X O O
 4 . X X X X . X O .
 3 X X O O O X X O O
 2 O O O X O O X X X
 1 . . . . . O O O O
");
    let mut rules = Rules::new();
    rules.ko_rule = Rules::KO_SITUATIONAL;
    rules.scoring_rule = Rules::SCORING_TERRITORY;
    rules.tax_rule = Rules::TAX_NONE;
    rules.multi_stone_suicide_legal = false;
    rules.komi = 6.5f32;
    rules.has_button = false;
    let mut hist = BoardHistory::new_full(&board, P_WHITE, &rules, 0);

    mmal!(hist, board, Board::PASS_LOC, P_WHITE);
    mmal!(hist, board, Board::PASS_LOC, P_BLACK);
    mmal!(hist, board, Board::PASS_LOC, P_WHITE);
    mmal!(hist, board, Location::get_loc(8, 3, board.x_size), P_BLACK);
    mmal!(hist, board, Location::get_loc(7, 0, board.x_size), P_WHITE);
    out_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    mmal!(hist, board, Location::get_loc(7, 0, board.x_size), P_BLACK);
    out_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    mmal!(hist, board, Location::get_loc(8, 3, board.x_size), P_WHITE);
    out_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    mmal!(hist, board, Location::get_loc(6, 0, board.x_size), P_BLACK);
    out_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    mmal!(hist, board, Location::get_loc(8, 2, board.x_size), P_WHITE);
    out_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    mmal!(hist, board, Location::get_loc(8, 3, board.x_size), P_BLACK);
    out_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    mmal!(hist, board, Board::PASS_LOC, P_WHITE);
    out_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    print_illegal_moves(&mut out, &board, &hist, P_BLACK);

    out.push('\n');
    expect(name, &mut out, "double-ko-death-1c.expected0.txt");
    cases.ran();
  }
}
}

wasm_test! {
fn double_ko_death_2a() {
  init();
  let mut cases = Cases::new("testrules::double_ko_death_2a");
  let mut out = String::new();
  {
    let name = "Double ko death 2a";

    let mut board = parse(9, 9, r"
   A B C D E F G H J
 9 . . . X O X . X .
 8 . . . X O O X O X
 7 . . . X X O O . O
 6 . . . . X X O O .
 5 . . . . . X X O O
 4 . . . . . . X X X
 3 . . . . . . . . .
 2 . . . . . . . . .
 1 . . . . . . . . .
");
    let mut rules = Rules::new();
    rules.ko_rule = Rules::KO_SIMPLE;
    rules.scoring_rule = Rules::SCORING_TERRITORY;
    rules.tax_rule = Rules::TAX_ALL;
    rules.multi_stone_suicide_legal = true;
    rules.komi = 6.5f32;
    rules.has_button = false;
    let mut hist = BoardHistory::new_full(&board, P_BLACK, &rules, 0);

    mmal!(hist, board, Board::PASS_LOC, P_BLACK);
    mmal!(hist, board, Board::PASS_LOC, P_WHITE);
    mmal!(hist, board, Location::get_loc(7, 2, board.x_size), P_BLACK);
    out_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    mmal!(hist, board, Location::get_loc(6, 0, board.x_size), P_WHITE);
    out_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    mmal!(hist, board, Location::get_loc(8, 3, board.x_size), P_BLACK);
    out_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    mmal!(hist, board, Location::get_loc(7, 1, board.x_size), P_WHITE);
    out_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    mmal!(hist, board, Location::get_loc(6, 1, board.x_size), P_BLACK);
    out_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    testAssert!(hist.is_legal(&board, Location::get_loc(7, 2, board.x_size), P_WHITE));
    mmal!(hist, board, Location::get_loc(8, 2, board.x_size), P_WHITE);
    out_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    hist.print_debug_info(&mut out, &board);

    out.push('\n');
    expect(name, &mut out, "double-ko-death-2a.expected0.txt");
    cases.ran();
  }
}
}

wasm_test! {
fn double_ko_death_2b() {
  init();
  let mut cases = Cases::new("testrules::double_ko_death_2b");
  let mut out = String::new();
  {
    let name = "Double ko death 2b";

    let mut board = parse(9, 9, r"
   A B C D E F G H J
 9 . . . X O X . X .
 8 . . . X O O X O X
 7 . . . X X O O . O
 6 . . . . X X O O .
 5 . . . . . X X O O
 4 . . . . . . X X X
 3 . . . . . . . . .
 2 . . . . . . . . .
 1 . . . . . . . . .
");
    let mut rules = Rules::new();
    rules.ko_rule = Rules::KO_SIMPLE;
    rules.scoring_rule = Rules::SCORING_TERRITORY;
    rules.tax_rule = Rules::TAX_ALL;
    rules.multi_stone_suicide_legal = true;
    rules.komi = 6.5f32;
    rules.has_button = false;
    let mut hist = BoardHistory::new_full(&board, P_BLACK, &rules, 0);

    mmal!(hist, board, Board::PASS_LOC, P_BLACK);
    mmal!(hist, board, Board::PASS_LOC, P_WHITE);
    mmal!(hist, board, Location::get_loc(7, 2, board.x_size), P_BLACK);
    out_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    mmal!(hist, board, Location::get_loc(6, 0, board.x_size), P_WHITE);
    out_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    mmal!(hist, board, Location::get_loc(8, 3, board.x_size), P_BLACK);
    out_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    mmal!(hist, board, Location::get_loc(7, 1, board.x_size), P_WHITE);
    out_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    mmal!(hist, board, Location::get_loc(7, 1, board.x_size), P_BLACK);
    out_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    testAssert!(hist.is_legal(&board, Location::get_loc(7, 2, board.x_size), P_WHITE));
    mmal!(hist, board, Location::get_loc(8, 3, board.x_size), P_WHITE);
    out_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    mmal!(hist, board, Location::get_loc(6, 1, board.x_size), P_BLACK);
    out_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    mmal!(hist, board, Location::get_loc(8, 2, board.x_size), P_WHITE);
    out_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    mmal!(hist, board, Location::get_loc(5, 0, board.x_size), P_BLACK);
    out_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    mmal!(hist, board, Location::get_loc(7, 2, board.x_size), P_WHITE);
    out_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    mmal!(hist, board, Location::get_loc(5, 0, board.x_size), P_BLACK);
    out_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    mmal!(hist, board, Location::get_loc(7, 1, board.x_size), P_WHITE);
    out_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    print_illegal_moves(&mut out, &board, &hist, P_BLACK);
    testAssert!(hist.is_legal(&board, Location::get_loc(7, 1, board.x_size), P_BLACK));
    mmal!(hist, board, Location::get_loc(7, 2, board.x_size), P_BLACK);
    out_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    mmal!(hist, board, Board::PASS_LOC, P_WHITE);
    out_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    print_illegal_moves(&mut out, &board, &hist, P_BLACK);

    hist.print_debug_info(&mut out, &board);

    out.push('\n');
    expect(name, &mut out, "double-ko-death-2b.expected0.txt");
    cases.ran();
  }
}
}

wasm_test! {
fn double_ko_death_2c() {
  init();
  let mut cases = Cases::new("testrules::double_ko_death_2c");
  let mut out = String::new();
  {
    let name = "Double ko death 2c";

    let mut board = parse(9, 9, r"
   A B C D E F G H J
 9 . . . X O X . X .
 8 . . . X O O X O X
 7 . . . X X O O . O
 6 . . . . X X O O .
 5 . . . . . X X O O
 4 . . . . . . X X X
 3 . . . . . . . . .
 2 . . . . . . . . .
 1 . . . . . . . . .
");
    let mut rules = Rules::new();
    rules.ko_rule = Rules::KO_SIMPLE;
    rules.scoring_rule = Rules::SCORING_TERRITORY;
    rules.tax_rule = Rules::TAX_ALL;
    rules.multi_stone_suicide_legal = true;
    rules.komi = 6.5f32;
    rules.has_button = false;
    let mut hist = BoardHistory::new_full(&board, P_BLACK, &rules, 0);

    mmal!(hist, board, Board::PASS_LOC, P_BLACK);
    mmal!(hist, board, Board::PASS_LOC, P_WHITE);
    mmal!(hist, board, Board::PASS_LOC, P_BLACK);
    mmal!(hist, board, Board::PASS_LOC, P_WHITE);
    mmal!(hist, board, Location::get_loc(7, 2, board.x_size), P_BLACK);
    out_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    mmal!(hist, board, Location::get_loc(6, 0, board.x_size), P_WHITE);
    out_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    mmal!(hist, board, Location::get_loc(8, 3, board.x_size), P_BLACK);
    out_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    mmal!(hist, board, Location::get_loc(7, 1, board.x_size), P_WHITE);
    out_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    testAssert!(!hist.is_legal(&board, Location::get_loc(6, 0, board.x_size), P_BLACK));
    mmal!(hist, board, Location::get_loc(7, 1, board.x_size), P_BLACK);
    hist.print_debug_info(&mut out, &board);
    out.push('\n');
    expect(name, &mut out, "double-ko-death-2c.expected0.txt");
    cases.ran();
  }
}
}

wasm_test! {
fn double_ko_death_2d() {
  init();
  let mut cases = Cases::new("testrules::double_ko_death_2d");
  let mut out = String::new();
  {
    let name = "Double ko death 2d";

    let mut board = parse(9, 9, r"
   A B C D E F G H J
 9 . . . X O X . X .
 8 . . . X O O X O X
 7 . . . X X O O . O
 6 . . . . X X O O .
 5 . . . . . X X O O
 4 . . . . . . X X X
 3 . . . . . . . . .
 2 . . . . . . . . .
 1 . . . . . . . . .
");
    let mut rules = Rules::new();
    rules.ko_rule = Rules::KO_SIMPLE;
    rules.scoring_rule = Rules::SCORING_TERRITORY;
    rules.tax_rule = Rules::TAX_ALL;
    rules.multi_stone_suicide_legal = true;
    rules.komi = 6.5f32;
    rules.has_button = false;
    let mut hist = BoardHistory::new_full(&board, P_BLACK, &rules, 0);

    mmal!(hist, board, Board::PASS_LOC, P_BLACK);
    mmal!(hist, board, Board::PASS_LOC, P_WHITE);
    mmal!(hist, board, Location::get_loc(7, 2, board.x_size), P_BLACK);
    out_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    mmal!(hist, board, Location::get_loc(6, 0, board.x_size), P_WHITE);
    out_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    mmal!(hist, board, Location::get_loc(8, 3, board.x_size), P_BLACK);
    out_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    mmal!(hist, board, Location::get_loc(7, 1, board.x_size), P_WHITE);
    out_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    mmal!(hist, board, Location::get_loc(6, 1, board.x_size), P_BLACK);
    out_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    mmal!(hist, board, Location::get_loc(7, 2, board.x_size), P_WHITE);
    out_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    mmal!(hist, board, Location::get_loc(6, 1, board.x_size), P_BLACK);
    out_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    mmal!(hist, board, Location::get_loc(8, 2, board.x_size), P_WHITE);
    out_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    mmal!(hist, board, Location::get_loc(5, 0, board.x_size), P_BLACK);
    out_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    hist.print_debug_info(&mut out, &board);

    out.push('\n');
    expect(name, &mut out, "double-ko-death-2d.expected0.txt");
    cases.ran();
  }
}
}

wasm_test! {
fn double_ko_death_2e() {
  init();
  let mut cases = Cases::new("testrules::double_ko_death_2e");
  let mut out = String::new();
  {
    let name = "Double ko death 2e";

    let mut board = parse(9, 9, r"
   A B C D E F G H J
 9 . . . X O X . X .
 8 . . . X O O X O X
 7 . . . X X O O . O
 6 . . . . X X O O .
 5 . . . . . X X O O
 4 . . . . . . X X X
 3 . . . . . . . . .
 2 . . . . . . . . .
 1 . . . . . . . . .
");
    let mut rules = Rules::new();
    rules.ko_rule = Rules::KO_SIMPLE;
    rules.scoring_rule = Rules::SCORING_TERRITORY;
    rules.tax_rule = Rules::TAX_ALL;
    rules.multi_stone_suicide_legal = true;
    rules.komi = 6.5f32;
    rules.has_button = false;
    let mut hist = BoardHistory::new_full(&board, P_BLACK, &rules, 0);

    mmal!(hist, board, Board::PASS_LOC, P_BLACK);
    mmal!(hist, board, Board::PASS_LOC, P_WHITE);
    mmal!(hist, board, Board::PASS_LOC, P_BLACK);
    mmal!(hist, board, Board::PASS_LOC, P_WHITE);
    mmal!(hist, board, Location::get_loc(7, 2, board.x_size), P_BLACK);
    out_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    mmal!(hist, board, Location::get_loc(6, 0, board.x_size), P_WHITE);
    out_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    mmal!(hist, board, Location::get_loc(5, 0, board.x_size), P_BLACK);
    out_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    mmal!(hist, board, Location::get_loc(7, 1, board.x_size), P_WHITE);
    out_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    hist.print_debug_info(&mut out, &board);

    out.push('\n');
    expect(name, &mut out, "double-ko-death-2e.expected0.txt");
    cases.ran();
  }
}
}

wasm_test! {
fn double_ko_death_2f() {
  init();
  let mut cases = Cases::new("testrules::double_ko_death_2f");
  let mut out = String::new();
  {
    let name = "Double ko death 2f";

    let mut board = parse(9, 9, r"
   A B C D E F G H J
 9 . . . X O X . X .
 8 . . . X O O X O X
 7 . . . X X O O . O
 6 . . . . X X O O .
 5 . . . . . X X O O
 4 . . . . . . X X X
 3 . . . . . . . . .
 2 . . . . . . . . .
 1 . . . . . . . . .
");
    let mut rules = Rules::new();
    rules.ko_rule = Rules::KO_SIMPLE;
    rules.scoring_rule = Rules::SCORING_TERRITORY;
    rules.tax_rule = Rules::TAX_ALL;
    rules.multi_stone_suicide_legal = true;
    rules.komi = 6.5f32;
    rules.has_button = false;
    let mut hist = BoardHistory::new_full(&board, P_BLACK, &rules, 0);

    mmal!(hist, board, Board::PASS_LOC, P_BLACK);
    mmal!(hist, board, Board::PASS_LOC, P_WHITE);
    mmal!(hist, board, Board::PASS_LOC, P_BLACK);
    mmal!(hist, board, Board::PASS_LOC, P_WHITE);
    mmal!(hist, board, Location::get_loc(7, 2, board.x_size), P_BLACK);
    out_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    mmal!(hist, board, Location::get_loc(6, 0, board.x_size), P_WHITE);
    out_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    mmal!(hist, board, Location::get_loc(5, 0, board.x_size), P_BLACK);
    out_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    mmal!(hist, board, Location::get_loc(7, 2, board.x_size), P_WHITE);
    out_board(&mut out, &board);
    print_encore_ko_block(&mut out, &board, &hist);
    hist.print_debug_info(&mut out, &board);

    out.push('\n');
    expect(name, &mut out, "double-ko-death-2f.expected0.txt");
    cases.ran();
  }
}
}

wasm_test! {
fn brute_force_testing_of_ko_hash_table() {
  init();
  let mut cases = Cases::new("testrules::brute_force_testing_of_ko_hash_table");
  {
    let name = "Brute force testing of ko hash table";
    let rules: Vec<Rules> = vec![
      Rules::new_full(Rules::KO_SIMPLE, Rules::SCORING_AREA, Rules::TAX_NONE, false, false, Rules::WHB_ZERO, false, 1.0f32),
      Rules::new_full(Rules::KO_POSITIONAL, Rules::SCORING_AREA, Rules::TAX_NONE, false, true, Rules::WHB_ZERO, false, 3.0f32),
      Rules::new_full(Rules::KO_SITUATIONAL, Rules::SCORING_AREA, Rules::TAX_NONE, true, false, Rules::WHB_N, false, 5.5f32),
      Rules::new_full(Rules::KO_SIMPLE, Rules::SCORING_AREA, Rules::TAX_SEKI, false, false, Rules::WHB_ZERO, false, 1.0f32),
      Rules::new_full(Rules::KO_POSITIONAL, Rules::SCORING_AREA, Rules::TAX_SEKI, true, false, Rules::WHB_N_MINUS_ONE, false, 4.5f32),
      Rules::new_full(Rules::KO_SITUATIONAL, Rules::SCORING_TERRITORY, Rules::TAX_NONE, true, true, Rules::WHB_ZERO, false, 6.5f32),
      Rules::new_full(Rules::KO_SIMPLE, Rules::SCORING_TERRITORY, Rules::TAX_SEKI, false, true, Rules::WHB_ZERO, false, 2.0f32),
      Rules::new_full(Rules::KO_SITUATIONAL, Rules::SCORING_TERRITORY, Rules::TAX_SEKI, true, true, Rules::WHB_N_MINUS_ONE, false, 5.5f32),
      Rules::new_full(Rules::KO_SIMPLE, Rules::SCORING_AREA, Rules::TAX_ALL, true, false, Rules::WHB_N, false, 2.5f32),
      Rules::new_full(Rules::KO_POSITIONAL, Rules::SCORING_AREA, Rules::TAX_ALL, false, true, Rules::WHB_ZERO, false, 3.0f32),
      Rules::new_full(Rules::KO_POSITIONAL, Rules::SCORING_TERRITORY, Rules::TAX_ALL, false, true, Rules::WHB_ZERO, false, 4.0f32),
      Rules::new_full(Rules::KO_SITUATIONAL, Rules::SCORING_TERRITORY, Rules::TAX_ALL, true, false, Rules::WHB_N, false, 6.5f32),
    ];
    let mut base_rand = Rand::new_seed(name);

    const NUM_BOARDS: usize = 6; // upstream: numBoards
    // `Board boards[6] = {5 initializers}`: the sixth is default-constructed (Board(), 19x19).
    let boards: [Board; NUM_BOARDS] = [
      Board::new_x_y(2, 2).unwrap(),
      Board::new_x_y(5, 1).unwrap(),
      Board::new_x_y(6, 1).unwrap(),
      Board::new_x_y(2, 3).unwrap(),
      Board::new_x_y(4, 2).unwrap(),
      Board::new(),
    ];
    for i in 0..NUM_BOARDS {
      for j in 0..rules.len() {
        let mut next_pla: Player = P_BLACK;
        let mut board = boards[i].clone();
        let mut hist = BoardHistory::new_full(&board, next_pla, &rules[j], 0);
        let mut board2 = boards[i].clone();
        let mut hist2 = BoardHistory::new_full(&board2, next_pla, &rules[j], 0);
        let mut table = KoHashTable::new();

        let mut rand = Rand::new_seed_u64(base_rand.next_uint64());
        let mut k = 0;
        while k < 300 {
          let mut num_legal: i32 = 0;
          const MAX_LEGAL_MOVES: i32 = Board::MAX_PLAY_SIZE + 1;
          let mut legal_moves: [Loc; MAX_LEGAL_MOVES as usize] = [0; MAX_LEGAL_MOVES as usize];
          for y in 0..board.y_size {
            for x in 0..board.x_size {
              let mv: Loc = Location::get_loc(x, y, board.x_size);
              let is_legal = hist.is_legal(&board, mv, next_pla);
              let is_legal2 = hist2.is_legal(&board2, mv, next_pla);
              testAssert!(is_legal == is_legal2);
              if is_legal {
                legal_moves[num_legal as usize] = mv;
                num_legal += 1;
              }
            }
          }
          {
            let mv: Loc = Board::PASS_LOC;
            let is_legal = hist.is_legal(&board, mv, next_pla);
            let is_legal2 = hist2.is_legal(&board2, mv, next_pla);
            testAssert!(is_legal == is_legal2);
            if is_legal {
              legal_moves[num_legal as usize] = mv;
              num_legal += 1;
            }
          }
          if hist.is_game_finished {
            break;
          }

          testAssert!(num_legal > 0);
          let mut mv: Loc = legal_moves[rand.next_uint_n(num_legal as u32) as usize];
          if mv == Board::PASS_LOC {
            mv = legal_moves[rand.next_uint_n(num_legal as u32) as usize];
          }
          if mv == Board::PASS_LOC {
            mv = legal_moves[rand.next_uint_n(num_legal as u32) as usize];
          }
          make_move_assert_legal(&mut hist, &mut board, mv, next_pla, line!());
          make_move_assert_legal_t(&mut hist2, &mut board2, mv, next_pla, line!(), Some(&table));
          next_pla = get_opp(next_pla);

          let mut just_recomputed = false;
          if rand.next_bool(0.10) {
            table.recompute(&hist2).expect("history shorter than 1000000000");
            just_recomputed = true;
          }

          for m in 0..hist.ko_hash_history.len() {
            let hash = hist.ko_hash_history[m];
            testAssert!(
              hist.number_of_ko_hash_occurrences_in_history(hash, None)
                == hist2.number_of_ko_hash_occurrences_in_history(hash, Some(&table))
            );
            if just_recomputed {
              testAssert!(hist.number_of_ko_hash_occurrences_in_history(hash, None) == table.number_of_occurrences_of_hash(hash));
            }
          }
          k += 1;
        }
        // `delete table` -> drop
        cases.ran();
      }
    }
  }
}
}
