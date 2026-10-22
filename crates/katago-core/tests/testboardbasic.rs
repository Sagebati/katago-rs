//! Replays upstream cpp/tests/testboardbasic.cpp @ v1.16.5 against the extracted
//! expected blocks in fixtures/upstream-v1.16.5/unit/testboardbasic/.
//!
//! Tests::runBoardReplayTest uses a test-local copy of PlayUtils::chooseRandomLegalMove
//! (program/playutils.rs is M3 and only ports getHackedLCBForWinrate), and also checks its Rand
//! stream against oracle/boardreplay_rng.cpp (the upstream test asserts only replay invariants).

#![allow(non_upper_case_globals)]

mod common;

use common::Cases;
use common::endl;
use common::fixture;
use common::init;
use katago_core::core::global::Global;
use katago_core::core::hash::Hash128;
use katago_core::core::rand::Rand;
use katago_core::core::test::TestCommon;
use katago_core::game::board::*;
use katago_core::game::boardhistory::BoardHistory;
use katago_core::game::graphhash::GraphHash;
use katago_core::game::rules::Rules;
use katago_core::strprintf;
use katago_core::testAssert;

const F: &str = "testboardbasic";

fn parse(x: i32, y: i32, s: &str) -> Board {
  Board::parse_board(x, y, s, '\n').unwrap()
}

/// upstream: TestCommon::boardsSeemEqual (tests/testcommon.cpp)
fn boards_seem_equal(b1: &Board, b2: &Board) -> bool {
  for i in 0..Board::MAX_ARR_SIZE as usize {
    if b1.colors[i] != b2.colors[i] {
      return false;
    }
  }
  if b1.num_black_captures != b2.num_black_captures {
    return false;
  }
  if b1.num_white_captures != b2.num_white_captures {
    return false;
  }
  true
}

fn expect(name: &str, out: &mut String, expected_file: &str) {
  let expected = fixture(F, expected_file);
  TestCommon::expect_string(name, out, &expected);
}

// ===================================================================== runBoardIOTests

#[test]
fn location_parse_test() {
  init();
  let mut cases = Cases::new("testboardbasic::location_parse_test");
  let mut out = String::new();
  let name = "Location parse test";
  let test_loc = |out: &mut String, s: &str, x_size: i32, y_size: i32| match Location::of_string_x_size_y_size(s, x_size, y_size) {
    Ok(loc) => {
      out.push_str(&format!(
        "{} {} x {} y {}\n",
        s,
        Location::to_string_x_size_y_size(loc, x_size, y_size),
        Location::get_x(loc, x_size),
        Location::get_y(loc, x_size)
      ));
    }
    Err(e) if e.is_string_error() => {
      out.push_str(e.what());
      endl(out);
    }
    Err(e) => panic!("{}", e),
  };

  const numSizes: usize = 4;
  let sizes: [i32; numSizes] = [9, 19, 26, 70];
  for i in 0..numSizes as i32 {
    for j in 0..numSizes as i32 {
      if i - j > 1 || j - i > 1 {
        continue;
      }
      let x_size = sizes[i as usize];
      let y_size = sizes[j as usize];
      out += "----------------------------------\n";
      out += &format!("{} {}\n", x_size, y_size);

      for s in [
        "A1", "A0", "B2", "b2", "A", "B", "1", "pass", "H9", "I9", "J9", "J10", "K8", "k19", "a22", "y1", "z1", "aa1", "AA26", "AZ26",
        "BC50",
      ] {
        test_loc(&mut out, s, x_size, y_size);
      }
    }
  }
  expect(name, &mut out, "location-parse-test.expected0.txt");
  cases.ran();
}

#[test]
fn parse_test() {
  init();
  let mut cases = Cases::new("testboardbasic::parse_test");
  let mut out = String::new();
  let name = "Parse test";
  let mut board = parse(6, 5, &fixture(F, "parse-test.input0.txt"));
  let mut board2 = parse(6, 5, &fixture(F, "parse-test.input1.txt"));

  let l = Location::of_string("B2", &board).unwrap();
  board.play_move(l, P_BLACK, true);
  let l = Location::of_string("F1", &board).unwrap();
  board2.play_move(l, P_WHITE, true);
  out += &format!("{}\n", board);
  out += &format!("{}\n", board2);

  expect(name, &mut out, "parse-test.expected0.txt");
  cases.ran();
}

// ===================================================================== runBoardBasicTests

#[test]
fn liberties() {
  init();
  let mut cases = Cases::new("testboardbasic::liberties");
  let mut out = String::new();
  let board = parse(9, 9, &fixture(F, "liberties.input0.txt"));
  endl(&mut out);
  for y in 0..board.y_size {
    for x in 0..board.x_size {
      let loc = Location::get_loc(x, y, board.x_size);
      if board.colors[loc as usize] != C_EMPTY {
        out += &board.get_num_liberties(loc).to_string();
      } else {
        out += ".";
      }
    }
    endl(&mut out);
  }
  endl(&mut out);
  expect("Liberties", &mut out, "liberties.expected0.txt");
  cases.ran();
}

/// The shape shared by the five "Liberties after move*" cases.
fn libs_after_move_case(name: &str, slug: &str, max: i32) {
  let mut out = String::new();
  let board = parse(9, 9, &fixture(F, &format!("{}.input0.txt", slug)));
  endl(&mut out);
  for (label, pla) in [("After black", P_BLACK), ("After white", P_WHITE)] {
    out += label;
    endl(&mut out);
    for y in 0..board.y_size {
      for x in 0..board.x_size {
        let loc = Location::get_loc(x, y, board.x_size);
        if board.colors[loc as usize] == C_EMPTY {
          out += &board.get_num_liberties_after_play(loc, pla, max).to_string();
        } else {
          out += "-";
        }
      }
      endl(&mut out);
    }
    endl(&mut out);
  }
  expect(name, &mut out, &format!("{}.expected0.txt", slug));
}

#[test]
fn liberties_after_move() {
  init();
  let mut cases = Cases::new("testboardbasic::liberties_after_move*");
  libs_after_move_case("Liberties after move", "liberties-after-move", 100);
  cases.ran();
  libs_after_move_case("Liberties after move capped at 2", "liberties-after-move-capped-at-2", 2);
  cases.ran();
  libs_after_move_case("Liberties after move capped at 3", "liberties-after-move-capped-at-3", 3);
  cases.ran();
  libs_after_move_case("Liberties after move 2", "liberties-after-move-2", 100);
  cases.ran();
  libs_after_move_case("Liberties after move 2 capped at 3", "liberties-after-move-2-capped-at-3", 3);
  cases.ran();
}

#[test]
fn distance() {
  init();
  let mut cases = Cases::new("testboardbasic::distance");
  let mut out = String::new();
  let board = Board::new_x_y(17, 12).unwrap();
  let test_distance = |out: &mut String, x0: i32, y0: i32, x1: i32, y1: i32| {
    *out += &format!(
      "distance ({},{}) ({},{}) = {}\n",
      x0,
      y0,
      x1,
      y1,
      Location::distance(Location::get_loc(x0, y0, board.x_size), Location::get_loc(x1, y1, board.x_size), board.x_size)
    );
  };
  let test_euclidean_distance = |out: &mut String, x0: i32, y0: i32, x1: i32, y1: i32| {
    *out += &format!(
      "euclideanSq ({},{}) ({},{}) = {}\n",
      x0,
      y0,
      x1,
      y1,
      Location::euclidean_distance_squared(Location::get_loc(x0, y0, board.x_size), Location::get_loc(x1, y1, board.x_size), board.x_size)
    );
  };
  let targets: [(i32, i32); 24] = [
    (12, 3), (12, 4), (12, 5), (12, 6), (12, 7), (13, 3), (13, 4), (13, 5), (13, 6), (13, 7), (14, 3), (14, 4), (14, 5), (14, 6), (14, 7),
    (15, 3), (15, 4), (15, 5), (15, 6), (15, 7), (0, 0), (16, 11), (0, 11), (16, 0),
  ];
  for (x1, y1) in targets {
    test_distance(&mut out, 13, 6, x1, y1);
  }
  for (x1, y1) in targets {
    test_euclidean_distance(&mut out, 13, 6, x1, y1);
  }
  expect("Distance", &mut out, "distance.expected0.txt");
  cases.ran();
}

/// Shared shape of IsAdjacentToPla / wouldBeCapture / wouldBeKoCapture.
fn per_point_bool_case(name: &str, slug: &str, labels: [&str; 2], f: impl Fn(&Board, Loc, Player) -> bool) {
  let mut out = String::new();
  let board = parse(9, 9, &fixture(F, &format!("{}.input0.txt", slug)));
  endl(&mut out);
  for (label, pla) in [(labels[0], P_BLACK), (labels[1], P_WHITE)] {
    out += label;
    endl(&mut out);
    for y in 0..board.y_size {
      for x in 0..board.x_size {
        let loc = Location::get_loc(x, y, board.x_size);
        out += &(f(&board, loc, pla) as i32).to_string();
      }
      endl(&mut out);
    }
    endl(&mut out);
  }
  expect(name, &mut out, &format!("{}.expected0.txt", slug));
}

#[test]
fn is_adjacent_to_pla() {
  init();
  let mut cases = Cases::new("testboardbasic::is_adjacent_to_pla");
  per_point_bool_case("IsAdjacentToPla", "isadjacenttopla", ["Adj black", "Adj white"], |b, l, p| b.is_adjacent_to_pla(l, p));
  cases.ran();
}

#[test]
fn would_be_capture() {
  init();
  let mut cases = Cases::new("testboardbasic::would_be_capture");
  per_point_bool_case("wouldBeCapture", "wouldbecapture", ["WouldBeCapture black", "WouldBeCapture white"], |b, l, p| {
    b.would_be_capture(l, p)
  });
  cases.ran();
}

#[test]
fn would_be_ko_capture() {
  init();
  let mut cases = Cases::new("testboardbasic::would_be_ko_capture");
  per_point_bool_case("wouldBeKoCapture", "wouldbekocapture", ["WouldBeKo black", "WouldBeKo white"], |b, l, p| {
    b.would_be_ko_capture(l, p)
  });
  cases.ran();
}

/// `if(colors != EMPTY) out << (int)search(loc); else out << ".";` for every point of one board.
fn ladder_grid(out: &mut String, board: &mut Board, attacker_first_2_libs: bool, buf: &mut Vec<Loc>, buf2: &mut Vec<Loc>) {
  for y in 0..board.y_size {
    for x in 0..board.x_size {
      let loc = Location::get_loc(x, y, board.x_size);
      if board.colors[loc as usize] != C_EMPTY {
        let r = if attacker_first_2_libs {
          board.search_is_ladder_captured_attacker_first2_libs(loc, buf, buf2)
        } else {
          board.search_is_ladder_captured(loc, true, buf)
        };
        *out += &(r as i32).to_string();
      } else {
        *out += ".";
      }
    }
    endl(out);
  }
  endl(out);
}

fn ladder_case(name: &str, slug: &str, x: i32, y: i32, attacker_first_2_libs: bool, check_start: bool) {
  let mut out = String::new();
  let mut buf: Vec<Loc> = Vec::new();
  let mut buf2: Vec<Loc> = Vec::new();
  let mut board = parse(x, y, &fixture(F, &format!("{}.input0.txt", slug)));
  endl(&mut out);
  let start_board = board.clone();
  ladder_grid(&mut out, &mut board, attacker_first_2_libs, &mut buf, &mut buf2);
  board.check_consistency().unwrap();
  if check_start {
    testAssert!(boards_seem_equal(&board, &start_board));
  }
  expect(name, &mut out, &format!("{}.expected0.txt", slug));
}

#[test]
fn ladders_1_lib() {
  init();
  let mut cases = Cases::new("testboardbasic::ladders_1_lib");
  ladder_case("Ladders 1 Lib", "ladders-1-lib", 9, 9, false, true);
  cases.ran();
}

#[test]
fn ladders_2_libs() {
  init();
  let mut cases = Cases::new("testboardbasic::ladders_2_libs");
  ladder_case("Ladders 2 Libs", "ladders-2-libs", 9, 9, true, true);
  cases.ran();
}

#[test]
fn ladders_ko_1() {
  init();
  let mut cases = Cases::new("testboardbasic::ladders_ko_1");
  ladder_case("LaddersKo-1", "laddersko-1", 18, 9, true, true);
  cases.ran();
}

#[test]
fn ladders_ko_2() {
  init();
  let mut cases = Cases::new("testboardbasic::ladders_ko_2");
  let mut out = String::new();
  let mut buf: Vec<Loc> = Vec::new();
  let mut buf2: Vec<Loc> = Vec::new();
  let mut board = parse(18, 9, &fixture(F, "laddersko-2.input0.txt"));
  let mut board2 = parse(18, 9, &fixture(F, "laddersko-2.input1.txt"));
  let mut board3 = parse(18, 9, &fixture(F, "laddersko-2.input2.txt"));
  endl(&mut out);
  ladder_grid(&mut out, &mut board, true, &mut buf, &mut buf2);
  ladder_grid(&mut out, &mut board2, true, &mut buf, &mut buf2);
  ladder_grid(&mut out, &mut board3, true, &mut buf, &mut buf2);
  board.check_consistency().unwrap();
  board2.check_consistency().unwrap();
  board3.check_consistency().unwrap();
  expect("LaddersKo-2", &mut out, "laddersko-2.expected0.txt");
  cases.ran();
}

#[test]
fn ladders_multi_ko() {
  init();
  let mut cases = Cases::new("testboardbasic::ladders_multi_ko");
  ladder_case("LaddersMultiKo", "laddersmultiko", 9, 9, true, false);
  cases.ran();
}

#[test]
fn ladders_big_board() {
  init();
  let mut cases = Cases::new("testboardbasic::ladders_big_board");
  ladder_case("LaddersBigBoard", "laddersbigboard", 19, 19, true, true);
  cases.ran();
}

#[test]
fn whole_board_ladder() {
  init();
  let mut cases = Cases::new("testboardbasic::whole_board_ladder");
  ladder_case("WholeBoardLadder", "wholeboardladder", 19, 19, true, true);
  cases.ran();
}

#[test]
fn cubic_ladder_not_far_from_max_node_budget() {
  init();
  let mut cases = Cases::new("testboardbasic::cubic_ladder_not_far_from_max_node_budget");
  ladder_case("CubicLadder not far from max node budget", "cubicladder-not-far-from-max-node-budget", 19, 19, true, true);
  cases.ran();
}

#[test]
fn failing_polynomial_ladder_due_to_max_node_budget() {
  init();
  let mut cases = Cases::new("testboardbasic::failing_polynomial_ladder_due_to_max_node_budget");
  //Requires a node budget of somewhere between 2.5M and 25M nodes
  ladder_case(
    "Failing polynomial ladder due to max node budget",
    "failing-polynomial-ladder-due-to-max-node-budget",
    19,
    19,
    true,
    true,
  );
  cases.ran();
}

fn simple_repetition_bound_case(name: &str, slug: &str) {
  let mut out = String::new();
  let board = parse(19, 19, &fixture(F, &format!("{}.input0.txt", slug)));
  let start_board = board.clone();
  for pla in 0..=2 as Player {
    for y in 0..board.y_size {
      for x in 0..board.x_size {
        let loc = Location::get_loc(x, y, board.x_size);
        if board.colors[loc as usize] != C_EMPTY {
          if pla != C_EMPTY {
            out += &format!("   {}", PlayerIO::color_to_char(board.colors[loc as usize]));
          } else {
            let mut bound: i32 = 0;
            while bound < 362 {
              if !board.simple_repetition_bound_gt(loc, bound) {
                break;
              }
              bound += 1;
            }
            out += &strprintf!(" %3d", bound);
            while bound < 362 {
              testAssert!(!board.simple_repetition_bound_gt(loc, bound));
              bound += 1;
            }
          }
          board.check_consistency().unwrap();
        } else if pla == C_EMPTY {
          out += "   .";
        } else {
          let mut board_copy = board.clone();
          board_copy.clear_simple_ko_loc();
          let is_multi_stone_suicide_legal = true;
          let suc = board_copy.play_move(loc, pla, is_multi_stone_suicide_legal);
          if !suc {
            out += "   .";
          } else {
            let mut bound: i32 = 0;
            while bound < 362 {
              if !board_copy.simple_repetition_bound_gt(loc, bound) {
                break;
              }
              bound += 1;
            }
            out += &strprintf!(" %3d", bound);
            while bound < 362 {
              testAssert!(!board_copy.simple_repetition_bound_gt(loc, bound));
              bound += 1;
            }
          }
          board_copy.check_consistency().unwrap();
        }
      }
      endl(&mut out);
    }
    endl(&mut out);
  }
  board.check_consistency().unwrap();
  testAssert!(boards_seem_equal(&board, &start_board));
  expect(name, &mut out, &format!("{}.expected0.txt", slug));
}

#[test]
fn simple_repetition_bound_open() {
  init();
  let mut cases = Cases::new("testboardbasic::simple_repetition_bound_open");
  simple_repetition_bound_case("simpleRepetitionBoundOpen", "simplerepetitionboundopen");
  cases.ran();
}

#[test]
fn simple_repetition_bound() {
  init();
  let mut cases = Cases::new("testboardbasic::simple_repetition_bound");
  simple_repetition_bound_case("simpleRepetitionBound", "simplerepetitionbound");
  cases.ran();
}

// ===================================================================== runBoardUndoTest

#[test]
fn board_undo_test() {
  init();
  let mut cases = Cases::new("testboardbasic::board_undo_test");
  let mut rand = Rand::new_seed("runBoardUndoTests");

  let mut suicide_count: i32 = 0;
  let mut ko_capture_count: i32 = 0;
  let mut pass_count: i32 = 0;
  let mut regular_move_count: i32 = 0;
  let mut run = |start_board: &Board, multi_stone_suicide_legal: bool| {
    const steps: usize = 1000;
    let mut boards: Vec<Board> = vec![Board::new(); steps + 1];
    let mut records: Vec<MoveRecord> = vec![MoveRecord::default(); steps];
    boards[0] = start_board.clone();
    for n in 1..=steps {
      boards[n] = boards[n - 1].clone();
      let mut loc: Loc;
      let mut pla: Player;
      loop {
        pla = if rand.next_uint_n(2) == 0 { P_BLACK } else { P_WHITE };
        //Maximum range of board location values when 19x19:
        let num_locs: i32 = (19 + 1) * (19 + 2) + 1;
        loc = rand.next_uint_n(num_locs as u32) as Loc;
        if boards[n].is_legal(loc, pla, multi_stone_suicide_legal) {
          break;
        }
      }

      records[n - 1] = boards[n].play_move_recorded(loc, pla);
      if loc == Board::PASS_LOC {
        pass_count += 1;
      } else if boards[n - 1].is_suicide(loc, pla) {
        suicide_count += 1;
      } else {
        if boards[n].ko_loc != Board::NULL_LOC {
          ko_capture_count += 1;
        }
        regular_move_count += 1;
      }
    }

    let mut board = boards[steps].clone();
    for n in (0..steps).rev() {
      board.undo(records[n]);
      testAssert!(boards_seem_equal(&boards[n], &board));
      board.check_consistency().unwrap();
    }
  };

  run(&Board::new_x_y(19, 19).unwrap(), true);
  run(&Board::new_x_y(4, 4).unwrap(), true);
  run(&Board::new_x_y(4, 4).unwrap(), false);

  let mut out = String::new();
  endl(&mut out);
  out += &format!("regularMoveCount {}\n", regular_move_count);
  out += &format!("passCount {}\n", pass_count);
  out += &format!("koCaptureCount {}\n", ko_capture_count);
  out += &format!("suicideCount {}\n", suicide_count);

  expect("Board undo test move counts", &mut out, "runboardundotest.expected0.txt");
  cases.ran();
}

// ===================================================================== runBoardHandicapTest

#[test]
fn board_handicap_test() {
  init();
  let mut cases = Cases::new("testboardbasic::board_handicap_test");
  {
    let mut board = Board::new_x_y(19, 19).unwrap();
    let next_pla: Player = P_BLACK;
    let rules = Rules::parse_rules("chinese").unwrap();
    let mut hist = BoardHistory::new_full(&board, next_pla, &rules, 0);

    testAssert!(hist.compute_num_handicap_stones() == 0);
    testAssert!(hist.compute_white_handicap_bonus() == 0);
    let loc = Location::get_loc(3, 3, board.x_size);
    hist.make_board_move_assume_legal(&mut board, loc, P_BLACK, None, false);
    testAssert!(hist.compute_num_handicap_stones() == 0);
    testAssert!(hist.compute_white_handicap_bonus() == 0);
    let loc = Location::get_loc(3, 4, board.x_size);
    hist.make_board_move_assume_legal(&mut board, loc, P_BLACK, None, false);
    testAssert!(hist.compute_num_handicap_stones() == 0);
    testAssert!(hist.compute_white_handicap_bonus() == 0);
    let loc = Location::get_loc(3, 5, board.x_size);
    hist.make_board_move_assume_legal(&mut board, loc, P_BLACK, None, false);
    testAssert!(hist.compute_num_handicap_stones() == 0);
    testAssert!(hist.compute_white_handicap_bonus() == 0);
    hist.set_assume_multiple_starting_black_moves_are_handicap(true);
    testAssert!(hist.compute_num_handicap_stones() == 3);
    testAssert!(hist.compute_white_handicap_bonus() == 3);
  }
  cases.ran();

  {
    let mut board = Board::new_x_y(19, 19).unwrap();
    let next_pla: Player = P_BLACK;
    let rules = Rules::parse_rules("chinese").unwrap();
    let mut hist = BoardHistory::new_full(&board, next_pla, &rules, 0);

    hist.set_assume_multiple_starting_black_moves_are_handicap(true);
    testAssert!(hist.compute_num_handicap_stones() == 0);
    let loc = Location::get_loc(3, 3, board.x_size);
    hist.make_board_move_assume_legal(&mut board, loc, P_BLACK, None, false);
    testAssert!(hist.compute_num_handicap_stones() == 0);
    testAssert!(hist.compute_white_handicap_bonus() == 0);
    let loc = Location::get_loc(3, 4, board.x_size);
    hist.make_board_move_assume_legal(&mut board, loc, P_BLACK, None, false);
    testAssert!(hist.compute_num_handicap_stones() == 2);
    testAssert!(hist.compute_white_handicap_bonus() == 2);
    let loc = Location::get_loc(3, 5, board.x_size);
    hist.make_board_move_assume_legal(&mut board, loc, P_BLACK, None, false);
    testAssert!(hist.compute_num_handicap_stones() == 3);
    testAssert!(hist.compute_white_handicap_bonus() == 3);
  }
  cases.ran();

  {
    let mut board = Board::new_x_y(19, 19).unwrap();
    let next_pla: Player = P_BLACK;
    let rules = Rules::parse_rules("aga").unwrap();
    let mut hist = BoardHistory::new_full(&board, next_pla, &rules, 0);

    hist.set_assume_multiple_starting_black_moves_are_handicap(true);
    testAssert!(hist.compute_num_handicap_stones() == 0);
    let loc = Location::get_loc(3, 3, board.x_size);
    hist.make_board_move_assume_legal(&mut board, loc, P_BLACK, None, false);
    testAssert!(hist.compute_num_handicap_stones() == 0);
    testAssert!(hist.compute_white_handicap_bonus() == 0);
    let loc = Location::get_loc(3, 4, board.x_size);
    hist.make_board_move_assume_legal(&mut board, loc, P_BLACK, None, false);
    testAssert!(hist.compute_num_handicap_stones() == 2);
    testAssert!(hist.compute_white_handicap_bonus() == 1);
    let loc = Location::get_loc(3, 5, board.x_size);
    hist.make_board_move_assume_legal(&mut board, loc, P_BLACK, None, false);
    testAssert!(hist.compute_num_handicap_stones() == 3);
    testAssert!(hist.compute_white_handicap_bonus() == 2);
  }
  cases.ran();

  {
    let mut board = Board::new_x_y(19, 19).unwrap();
    let next_pla: Player = P_BLACK;
    let rules = Rules::parse_rules("aga").unwrap();
    let mut hist = BoardHistory::new_full(&board, next_pla, &rules, 0);

    hist.set_assume_multiple_starting_black_moves_are_handicap(true);
    testAssert!(hist.compute_num_handicap_stones() == 0);
    let loc = Location::get_loc(3, 3, board.x_size);
    hist.make_board_move_assume_legal(&mut board, loc, P_BLACK, None, false);
    testAssert!(hist.compute_num_handicap_stones() == 0);
    testAssert!(hist.compute_white_handicap_bonus() == 0);
    hist.make_board_move_assume_legal(&mut board, Board::PASS_LOC, P_WHITE, None, false);
    testAssert!(hist.compute_num_handicap_stones() == 0);
    testAssert!(hist.compute_white_handicap_bonus() == 0);
    let loc = Location::get_loc(3, 4, board.x_size);
    hist.make_board_move_assume_legal(&mut board, loc, P_BLACK, None, false);
    testAssert!(hist.compute_num_handicap_stones() == 2);
    testAssert!(hist.compute_white_handicap_bonus() == 1);
    hist.make_board_move_assume_legal(&mut board, Board::PASS_LOC, P_WHITE, None, false);
    testAssert!(hist.compute_num_handicap_stones() == 2);
    testAssert!(hist.compute_white_handicap_bonus() == 1);
    let loc = Location::get_loc(3, 5, board.x_size);
    hist.make_board_move_assume_legal(&mut board, loc, P_BLACK, None, false);
    testAssert!(hist.compute_num_handicap_stones() == 3);
    testAssert!(hist.compute_white_handicap_bonus() == 2);
    let loc = Location::get_loc(3, 6, board.x_size);
    hist.make_board_move_assume_legal(&mut board, loc, P_WHITE, None, false);
    testAssert!(hist.compute_num_handicap_stones() == 3);
    testAssert!(hist.compute_white_handicap_bonus() == 2);
    let loc = Location::get_loc(3, 7, board.x_size);
    hist.make_board_move_assume_legal(&mut board, loc, P_BLACK, None, false);
    testAssert!(hist.compute_num_handicap_stones() == 3);
    testAssert!(hist.compute_white_handicap_bonus() == 2);
  }
  cases.ran();

  {
    let mut board = Board::new_x_y(19, 19).unwrap();
    let next_pla: Player = P_BLACK;
    let rules = Rules::parse_rules("chinese").unwrap();
    let mut hist = BoardHistory::new_full(&board, next_pla, &rules, 0);

    hist.set_assume_multiple_starting_black_moves_are_handicap(true);
    testAssert!(hist.compute_num_handicap_stones() == 0);
    let loc = Location::get_loc(3, 3, board.x_size);
    hist.make_board_move_assume_legal(&mut board, loc, P_BLACK, None, false);
    testAssert!(hist.compute_num_handicap_stones() == 0);
    testAssert!(hist.compute_white_handicap_bonus() == 0);
    let loc = Location::get_loc(3, 4, board.x_size);
    hist.make_board_move_assume_legal(&mut board, loc, P_WHITE, None, false);
    testAssert!(hist.compute_num_handicap_stones() == 0);
    testAssert!(hist.compute_white_handicap_bonus() == 0);
    let loc = Location::get_loc(3, 5, board.x_size);
    hist.make_board_move_assume_legal(&mut board, loc, P_BLACK, None, false);
    testAssert!(hist.compute_num_handicap_stones() == 0);
    testAssert!(hist.compute_white_handicap_bonus() == 0);
    hist.make_board_move_assume_legal(&mut board, Board::PASS_LOC, P_WHITE, None, false);
    testAssert!(hist.compute_num_handicap_stones() == 0);
    testAssert!(hist.compute_white_handicap_bonus() == 0);
    let loc = Location::get_loc(3, 6, board.x_size);
    hist.make_board_move_assume_legal(&mut board, loc, P_BLACK, None, false);
    testAssert!(hist.compute_num_handicap_stones() == 0);
    testAssert!(hist.compute_white_handicap_bonus() == 0);
    let loc = Location::get_loc(3, 7, board.x_size);
    hist.make_board_move_assume_legal(&mut board, loc, P_BLACK, None, false);
    testAssert!(hist.compute_num_handicap_stones() == 0);
    testAssert!(hist.compute_white_handicap_bonus() == 0);
  }
  cases.ran();

}

// ===================================================================== runBoardStressTest

#[test]
fn board_stress_test() {
  init();
  let mut cases = Cases::new("testboardbasic::board_stress_test");
  {
    let mut rand = Rand::new_seed("runBoardStressTests");
    const numBoards: usize = 4;
    let mut boards: Vec<Board> = vec![Board::new(), Board::new(), Board::new(), Board::new()];
    boards[0] = Board::new();
    boards[1] = Board::new_x_y(9, 16).unwrap();
    boards[2] = Board::new_x_y(13, 7).unwrap();
    boards[3] = Board::new_x_y(4, 4).unwrap();
    let multi_stone_suicide_legal: [bool; 4] = [false, false, true, false];
    let mut copies: Vec<Board> = vec![Board::new(), Board::new(), Board::new(), Board::new()];
    let mut pla: Player = C_BLACK;
    let mut suicide_count: i32 = 0;
    let mut ko_ban_count: i32 = 0;
    let mut ko_capture_count: i32 = 0;
    let mut pass_count: i32 = 0;
    let mut regular_move_count: i32 = 0;
    for n in 0..20000 {
      let mut locs: [Loc; numBoards] = [0; numBoards];
      let mut empty_buf: [Loc; Board::MAX_ARR_SIZE as usize] = [0; Board::MAX_ARR_SIZE as usize];

      //Sometimes generate garbage input
      if n % 2 == 0 {
        locs[0] = rand.next_int_a_b(-10, 500) as Loc;
        locs[1] = rand.next_int_a_b(-10, 250) as Loc;
        locs[2] = rand.next_int_a_b(-10, 200) as Loc;
        locs[3] = rand.next_int_a_b(-10, 50) as Loc;
      }
      //Sometimes select only from empty points, to greatly increase the chance of real moves
      else {
        for i in 0..numBoards {
          let mut empty_count: i32 = 0;
          let end: Loc = Location::get_loc(boards[i].x_size - 1, boards[i].y_size - 1, boards[i].x_size);
          let mut j: Loc = 0;
          while j <= end {
            if boards[i].colors[j as usize] == C_EMPTY {
              empty_buf[empty_count as usize] = j;
              empty_count += 1;
            }
            j += 1;
          }
          testAssert!(empty_count > 0);
          locs[i] = empty_buf[rand.next_uint_n(empty_count as u32) as usize];
        }
      }

      for i in 0..numBoards {
        copies[i] = boards[i].clone();
      }

      let mut is_legal: [bool; numBoards] = [false; numBoards];
      let mut suc: [bool; numBoards] = [false; numBoards];
      for i in 0..numBoards {
        is_legal[i] = boards[i].is_legal(locs[i], pla, multi_stone_suicide_legal[i]);
        testAssert!(boards_seem_equal(&copies[i], &boards[i]));
        suc[i] = boards[i].play_move(locs[i], pla, multi_stone_suicide_legal[i]);
      }

      for i in 0..numBoards {
        testAssert!(is_legal[i] == suc[i]);
        boards[i].check_consistency().unwrap();

        let board: &Board = &boards[i];
        let copy: &Board = &copies[i];
        let loc: Loc = locs[i];
        if !suc[i] {
          if board.is_on_board(loc) {
            testAssert!(boards_seem_equal(copy, board));
            testAssert!(
              loc < 0
                || (loc as i32) >= Board::MAX_ARR_SIZE
                || board.colors[loc as usize] != C_EMPTY
                || board.is_illegal_suicide(loc, pla, multi_stone_suicide_legal[i])
                || board.is_ko_banned(loc)
            );
            if board.is_ko_banned(loc) {
              testAssert!(board.colors[loc as usize] == C_EMPTY);
              testAssert!(board.would_be_ko_capture(loc, C_BLACK) || board.would_be_ko_capture(loc, C_WHITE));
              testAssert!(board.would_be_capture(loc, C_BLACK) || board.would_be_capture(loc, C_WHITE));
              if board.is_adjacent_to_pla(loc, get_opp(pla)) {
                testAssert!(board.would_be_ko_capture(loc, pla));
                testAssert!(board.would_be_capture(loc, pla));
              }
              ko_ban_count += 1;
            }
          }
        } else if loc == Board::PASS_LOC {
          testAssert!(boards_seem_equal(copy, board));
          testAssert!(board.ko_loc == Board::NULL_LOC);
          pass_count += 1;
        } else if copy.is_suicide(loc, pla) {
          testAssert!(board.colors[loc as usize] == C_EMPTY);
          testAssert!(board.is_legal(loc, pla, multi_stone_suicide_legal[i]));
          testAssert!(multi_stone_suicide_legal[i]);
          testAssert!(!copy.would_be_capture(loc, pla));
          suicide_count += 1;
        } else {
          testAssert!(board.colors[loc as usize] == pla);
          testAssert!(board.get_num_liberties(loc) == copy.get_num_liberties_after_play(loc, pla, 1000));
          testAssert!(std::cmp::min(2, board.get_num_liberties(loc)) == copy.get_num_liberties_after_play(loc, pla, 2));
          testAssert!(std::cmp::min(4, board.get_num_liberties(loc)) == copy.get_num_liberties_after_play(loc, pla, 4));
          if board.ko_loc != Board::NULL_LOC {
            ko_capture_count += 1;
            testAssert!(copy.would_be_ko_capture(loc, pla));
            testAssert!(copy.would_be_capture(loc, pla));
          } else {
            testAssert!(!copy.would_be_ko_capture(loc, pla));
          }
          if !board.is_adjacent_to_pla(loc, get_opp(pla)) && copy.is_adjacent_to_pla(loc, get_opp(pla)) {
            testAssert!(copy.would_be_capture(loc, pla));
          }
          if !copy.is_adjacent_to_pla(loc, get_opp(pla)) {
            testAssert!(!copy.would_be_capture(loc, pla));
          }

          regular_move_count += 1;
        }
      }

      pla = if rand.next_uint_n(2) == 0 { get_opp(pla) } else { pla };
    }

    let mut out = String::new();
    endl(&mut out);
    out += &format!("regularMoveCount {}\n", regular_move_count);
    out += &format!("passCount {}\n", pass_count);
    out += &format!("koCaptureCount {}\n", ko_capture_count);
    out += &format!("koBanCount {}\n", ko_ban_count);
    out += &format!("suicideCount {}\n", suicide_count);

    for i in 0..4 {
      out += &format!("Caps {} {}\n", boards[i].num_black_captures, boards[i].num_white_captures);
    }
    expect("Board stress test move counts", &mut out, "runboardstresstest.expected0.txt");
    cases.ran();
  }

  {
    let rand = std::cell::RefCell::new(Rand::new_seed("runBoardSetStoneTests"));

    let board_to_placements = |b: &Board| -> Vec<Move> {
      let mut placements: Vec<Move> = Vec::new();
      for y in 0..b.y_size {
        for x in 0..b.x_size {
          let loc = Location::get_loc(x, y, b.x_size);
          placements.push(Move::new(loc, b.colors[loc as usize]));
        }
      }
      for i in 1..placements.len() {
        let r = rand.borrow_mut().next_uint_n((i + 1) as u32) as usize;
        placements.swap(i, r);
      }
      placements
    };
    let board_to_non_empty_placements = |b: &Board| -> Vec<Move> {
      let mut placements: Vec<Move> = Vec::new();
      for y in 0..b.y_size {
        for x in 0..b.x_size {
          let loc = Location::get_loc(x, y, b.x_size);
          if b.colors[loc as usize] != C_EMPTY {
            placements.push(Move::new(loc, b.colors[loc as usize]));
          }
        }
      }
      for i in 1..placements.len() {
        let r = rand.borrow_mut().next_uint_n((i + 1) as u32) as usize;
        placements.swap(i, r);
      }
      placements
    };

    for _rep in 0..1000 {
      let bx = 5 + rand.borrow_mut().next_uint_n(14) as i32;
      let by = 5 + rand.borrow_mut().next_uint_n(14) as i32;
      let mut board0 = Board::new_x_y(bx, by).unwrap();
      let mut board1 = board0.clone();
      let mut board2 = board0.clone();
      let mut board3 = board0.clone();
      let mut board4 = board0.clone();
      let mut board5 = board0.clone();
      let mut board6 = board0.clone();
      let num_moves1 = rand.borrow_mut().next_uint_n(1000) as i32;
      let num_moves2 = rand.borrow_mut().next_uint_n(1000) as i32;
      for _i in 0..num_moves1 {
        let lx = rand.borrow_mut().next_uint_n(board1.x_size as u32) as i32;
        let ly = rand.borrow_mut().next_uint_n(board1.y_size as u32) as i32;
        let loc = Location::get_loc(lx, ly, board1.x_size);
        let pla: Player = if rand.borrow_mut().next_bool(0.5) { P_BLACK } else { P_WHITE };
        if board1.is_legal(loc, pla, true) {
          let suc4 = board4.set_stone_fail_if_no_libs(loc, pla);
          testAssert!(suc4 == !(board1.would_be_capture(loc, pla) || board1.is_suicide(loc, pla)));
          if !suc4 {
            board4.play_move_assume_legal(loc, pla);
          } else {
            testAssert!(board4.colors[loc as usize] == pla);
          }

          board1.play_move_assume_legal(loc, pla);
          let suc3 = board3.set_stone(loc, pla);
          testAssert!(suc3);
        } else {
          let old_color: Color = board4.colors[loc as usize];
          let suc4 = board4.set_stone_fail_if_no_libs(loc, pla);
          if suc4 {
            testAssert!(board4.colors[loc as usize] == pla);
            let suc4_2 = board4.set_stone_fail_if_no_libs(loc, old_color);
            testAssert!(suc4_2);
          }
        }
      }
      board1.check_consistency().unwrap();
      board3.check_consistency().unwrap();
      board4.check_consistency().unwrap();
      testAssert!(board1.pos_hash == board3.pos_hash);
      testAssert!(board1.pos_hash == board4.pos_hash);

      let empty_prob: f64 = rand.borrow_mut().next_double() * 0.5;
      for _i in 0..num_moves2 {
        let lx = rand.borrow_mut().next_uint_n(board2.x_size as u32) as i32;
        let ly = rand.borrow_mut().next_uint_n(board2.y_size as u32) as i32;
        let loc = Location::get_loc(lx, ly, board2.x_size);
        let color: Color = if rand.borrow_mut().next_bool(empty_prob) {
          C_EMPTY
        } else if rand.borrow_mut().next_bool(0.5) {
          P_BLACK
        } else {
          P_WHITE
        };
        let suc2 = board2.set_stone(loc, color);
        testAssert!(suc2);
        let suc5 = board5.set_stone_fail_if_no_libs(loc, color);
        let suc6 = board6.set_stone_fail_if_no_libs(loc, color);
        if color == C_EMPTY {
          testAssert!(suc5);
          testAssert!(suc6);
        }
        if !suc5 {
          board5.set_stone(loc, color);
        } else {
          testAssert!(board5.colors[loc as usize] == color);
        }
        if suc6 {
          testAssert!(board6.colors[loc as usize] == color);
        } else {
          testAssert!(board6.colors[loc as usize] != color);
        }
        board2.check_consistency().unwrap();
        board5.check_consistency().unwrap();
        board6.check_consistency().unwrap();
        testAssert!(board2.pos_hash == board5.pos_hash);
      }

      let p = board_to_non_empty_placements(&board1);
      let suc0 = board0.set_stones_fail_if_no_libs(&p);
      testAssert!(suc0);
      board0.check_consistency().unwrap();
      testAssert!(board0.pos_hash == board1.pos_hash);

      let p = board_to_placements(&board2);
      let suc3 = board3.set_stones_fail_if_no_libs(&p);
      testAssert!(suc3);
      board3.check_consistency().unwrap();
      testAssert!(board3.pos_hash == board2.pos_hash);
    }
    cases.ran();
  }
  {
    let mut rand = Rand::new_seed("runBoardSetStoneTests2");
    for _rep in 0..1000 {
      let bx = 1 + rand.next_uint_n(18) as i32;
      let by = 1 + rand.next_uint_n(18) as i32;
      let mut board = Board::new_x_y(bx, by).unwrap();
      let mut placements: Vec<Move> = Vec::new();
      for _i in 0..1000 {
        let lx = rand.next_uint_n(board.x_size as u32) as i32;
        let ly = rand.next_uint_n(board.y_size as u32) as i32;
        let loc = Location::get_loc(lx, ly, board.x_size);
        let pla: Player = if rand.next_bool(0.5) { P_BLACK } else { P_WHITE };
        if board.is_legal(loc, pla, true) {
          placements.push(Move::new(loc, pla));
          let any_caps = board.would_be_capture(loc, pla) || board.is_suicide(loc, pla);
          board.play_move_assume_legal(loc, pla);
          let mut copy = Board::new_x_y(board.x_size, board.y_size).unwrap();
          let suc = copy.set_stones_fail_if_no_libs(&placements);
          testAssert!(suc == !any_caps);
          copy.check_consistency().unwrap();
          if !suc {
            break;
          }
          testAssert!(board.pos_hash == copy.pos_hash);
        }
      }
    }
    cases.ran();
  }
  {
    let mut rand = Rand::new_seed("runBoardSetStoneTests3");
    for _rep in 0..1000 {
      let bx = 1 + rand.next_uint_n(18) as i32;
      let by = 1 + rand.next_uint_n(18) as i32;
      let mut board = Board::new_x_y(bx, by).unwrap();
      for _i in 0..300 {
        let lx = rand.next_uint_n(board.x_size as u32) as i32;
        let ly = rand.next_uint_n(board.y_size as u32) as i32;
        let loc = Location::get_loc(lx, ly, board.x_size);
        let color: Color = if rand.next_bool(0.25) {
          C_EMPTY
        } else if rand.next_bool(0.5) {
          P_BLACK
        } else {
          P_WHITE
        };
        board.set_stone(loc, color);
      }

      let orig = board.clone();
      let mut prev_placed_locs: std::collections::BTreeSet<Loc> = std::collections::BTreeSet::new();
      let mut placements: Vec<Move> = Vec::new();
      for i in 0..1000 {
        let lx = rand.next_uint_n(board.x_size as u32) as i32;
        let ly = rand.next_uint_n(board.y_size as u32) as i32;
        let loc = Location::get_loc(lx, ly, board.x_size);
        let color: Color = if rand.next_bool(0.25) {
          C_EMPTY
        } else if rand.next_bool(0.5) {
          P_BLACK
        } else {
          P_WHITE
        };

        placements.push(Move::new(loc, color));
        if prev_placed_locs.contains(&loc) {
          let mut copy = Board::new_x_y(board.x_size, board.y_size).unwrap();
          let suc = copy.set_stones_fail_if_no_libs(&placements);
          testAssert!(!suc);
          placements.pop();
        } else {
          prev_placed_locs.insert(loc);
          let prev = board.clone();
          board.set_stone(loc, color);

          let mut any_caps = false;
          for y in 0..board.y_size {
            for x in 0..board.x_size {
              let l = Location::get_loc(x, y, board.x_size);
              if l != loc && board.colors[l as usize] == C_EMPTY && prev.colors[l as usize] != C_EMPTY {
                any_caps = true;
              }
            }
          }

          let mut copy = orig.clone();
          let suc = copy.set_stones_fail_if_no_libs(&placements);
          if !(suc == (!any_caps && board.colors[loc as usize] == color)) {
            eprintln!("{} {} {}", suc as i32, any_caps as i32, (board.colors[loc as usize] == color) as i32);
            eprintln!("{}", orig);
            eprintln!("{}", prev);
            eprintln!("{}", board);
            eprintln!("{}", i);
            for placement in &placements {
              eprintln!("{} {}", Location::to_string(placement.loc, &board), placement.pla as i32);
            }
          }
          testAssert!(suc == (!any_caps && board.colors[loc as usize] == color));
          if !suc {
            break;
          }
        }
      }
    }
    cases.ran();
  }
  let _ = Global::int_to_string(0);
}

// ===================================================================== runBoardReplayTest

/// upstream: PlayUtils::chooseRandomLegalMove (program/playutils.cpp)
/// Test-local copy: program/playutils.rs (M3) only ports getHackedLCBForWinrate, and this
/// test needs nothing else from it.
fn choose_random_legal_move(board: &Board, hist: &BoardHistory, pla: Player, game_rand: &mut Rand, ban_move: Loc) -> Loc {
  let mut num_legal_moves: i32 = 0;
  let mut locs: [Loc; Board::MAX_ARR_SIZE as usize] = [0; Board::MAX_ARR_SIZE as usize];
  testAssert!(pla == hist.presumed_next_move_pla);
  for loc in 0..Board::MAX_ARR_SIZE {
    let loc = loc as Loc;
    if hist.is_legal(board, loc, pla) && loc != ban_move {
      locs[num_legal_moves as usize] = loc;
      num_legal_moves += 1;
    }
  }
  if num_legal_moves > 0 {
    let n: i32 = game_rand.next_uint_n(num_legal_moves as u32) as i32;
    return locs[n as usize];
  }
  Board::NULL_LOC
}

/// upstream: Board::isAdjacentToPla (board.cpp), as called by runBoardReplayTest.
/// PORT NOTE: upstream UB (isAdjacentToPla(PASS_LOC) reads colors[-9], out of bounds).
/// chooseRandomLegalMove scans every loc from 0 and PASS_LOC is always legal, so upstream calls
/// isAdjacentToPla(PASS_LOC, opp) (and would call it with NULL_LOC if nothing were legal). FOREACHADJ
/// then reads colors[loc-(x_size+1)], colors[loc-1], colors[loc+1], colors[loc+x_size+1]: for PASS_LOC
/// = 1 that is colors[-9] (outside the Board), colors[0], colors[2] and colors[x_size+2] (= the
/// on-board point (0,0)). Rust would panic on the negative index, so only the adjacent indices >= 0
/// are checked: the result is true exactly when (0,0) holds `pla`. That keeps the RNG stream of the
/// upstream binary, whose out-of-bounds byte never equals pla there: oracle/boardreplay_rng.cpp runs
/// the real isAdjacentToPla next to this in-bounds variant and counts 0 disagreements over the 6000
/// reps, and board_replay_test asserts its Rand/move fingerprints. On-board locs are unaffected.
fn is_adjacent_to_pla_in_bounds(board: &Board, loc: Loc, pla: Player, fp: &mut ReplayFingerprint) -> bool {
  let mut r = false;
  for adjoffset in [-(board.x_size + 1), -1, 1, board.x_size + 1] {
    let adj: i32 = loc as i32 + adjoffset;
    if adj >= 0 && board.colors[adj as usize] == pla {
      r = true;
      break;
    }
  }
  if loc == Board::PASS_LOC || loc == Board::NULL_LOC {
    fp.pass_adj_calls += 1;
    if r {
      fp.pass_adj_true += 1;
    }
  }
  r
}

/// Not upstream: fingerprints of the Rand stream and of the selected moves, compared with
/// oracle/boardreplay_rng.txt (the same loop built from the C++ sources, oracle/boardreplay_rng.cpp).
/// The upstream test asserts only replay invariants, so without this a different draw sequence
/// would still pass.
struct ReplayFingerprint {
  moves: i64,
  pass_adj_calls: i64,
  pass_adj_true: i64,
  move_fnv: u64,
}

impl ReplayFingerprint {
  fn note_move(&mut self, loc: Loc) {
    self.moves += 1;
    self.move_fnv ^= loc as i64 as u64;
    self.move_fnv = self.move_fnv.wrapping_mul(1099511628211);
  }
}

#[test]
fn board_replay_test() {
  init();
  let mut cases = Cases::new("testboardbasic::board_replay_test");
  // cout << "Running board replay test" << endl;
  let mut rand = Rand::new_seed("runBoardReplayTest");

  let base = parse(9, 5, &fixture(F, "runboardreplaytest.input0.txt"));
  let mut fp = ReplayFingerprint { moves: 0, pass_adj_calls: 0, pass_adj_true: 0, move_fnv: 1469598103934665603 };
  let mut reps = 0;

  for _rep in 0..6000 {
    let mut board = base.clone();
    let mut pla: Player = if rand.next_bool(0.5) { P_BLACK } else { P_WHITE };
    let mut initial_encore_phase: i32 = if rand.next_bool(0.9) { 0 } else if rand.next_bool(0.5) { 1 } else { 2 };
    let mut rules: Rules = if rand.next_bool(0.8) { Rules::parse_rules("japanese").unwrap() } else { Rules::parse_rules("chinese").unwrap() };
    if rules.scoring_rule == Rules::SCORING_AREA {
      initial_encore_phase = 0;
    }
    if rand.next_bool(0.1) {
      rules.ko_rule = if rand.next_bool(0.5) { Rules::KO_SITUATIONAL } else { Rules::KO_POSITIONAL };
    }
    if rand.next_bool(0.2) {
      rules.tax_rule = if rand.next_bool(0.5) { Rules::TAX_SEKI } else if rand.next_bool(0.5) { Rules::TAX_NONE } else { Rules::TAX_ALL };
    }
    let mut hist = BoardHistory::new_full(&board, pla, &rules, initial_encore_phase);
    hist.set_initial_turn_number(rand.next_int_a_b(0, 40) as i64);
    hist.set_assume_multiple_starting_black_moves_are_handicap(rand.next_bool(0.5));

    let draw_equivalent_wins_for_white: f64 = 0.7;
    let rep_bound: i32 = rand.next_int_a_b(1, 5);

    let mut graph_hash: Hash128 = GraphHash::get_graph_hash_from_scratch(&hist, pla, rep_bound, draw_equivalent_wins_for_white);

    let mut any_komi_set = false;
    let pass_prob: f64 = rand.next_double_a_b(0.05, 0.80);
    let num_steps: i32 = rand.next_int_a_b(6, 15);
    for _i in 0..num_steps {
      let tmp_hist = BoardHistory::new_full(&board, pla, &rules, hist.encore_phase);
      let mut move_loc: Loc;
      if rand.next_bool(pass_prob) {
        move_loc = Board::PASS_LOC;
      } else {
        move_loc = choose_random_legal_move(&board, &tmp_hist, pla, &mut rand, Board::NULL_LOC);
        //Resample to increase likelihood of capture moves.
        if !board.would_be_capture(move_loc, pla) {
          move_loc = choose_random_legal_move(&board, &tmp_hist, pla, &mut rand, Board::NULL_LOC);
        }
        //Resample to decrease likelihood of moves not adjacent to opponent
        if !is_adjacent_to_pla_in_bounds(&board, move_loc, get_opp(pla), &mut fp) {
          move_loc = choose_random_legal_move(&board, &tmp_hist, pla, &mut rand, Board::NULL_LOC);
        }
      }
      fp.note_move(move_loc);
      let prevent_encore: bool = rand.next_bool(0.5);
      if rand.next_bool(0.5) {
        // if(!hist.isLegal(board,moveLoc,pla))
        //   cout << "AAAA" << endl;
        let suc: bool = hist.make_board_move_tolerant_prevent_encore(&mut board, move_loc, pla, prevent_encore);
        testAssert!(suc);
      } else {
        if hist.is_legal(&board, move_loc, pla) {
          hist.make_board_move_assume_legal(&mut board, move_loc, pla, None, prevent_encore);
        } else {
          continue;
        }
      }
      pla = get_opp(pla);
      if rand.next_bool(0.1) {
        pla = get_opp(pla);
      }
      if rand.next_bool(0.025) {
        any_komi_set = true;
        let new_komi: f32 = hist.rules.komi + (if rand.next_bool(0.5) { -0.5 } else { 0.5 }) as f32;
        hist.set_komi(new_komi);
      }

      graph_hash = GraphHash::get_graph_hash(graph_hash, &hist, pla, rep_bound, draw_equivalent_wins_for_white);
    }

    let mut hist_copy = hist.copy_to_initial();
    let mut board_copy = hist_copy.get_recent_board(0).clone();
    for i in 0..hist.move_history.len() {
      hist_copy.make_board_move_assume_legal(&mut board_copy, hist.move_history[i].loc, hist.move_history[i].pla, None, hist.prevent_encore_history[i]);
    }
    if rand.next_bool(0.05) {
      hist_copy = hist.clone();
    }
    //if(rep < 100)
    //  hist.printDebugInfo(cout,board);

    testAssert!(board_copy.is_equal_for_testing(&board, true, true).unwrap());
    testAssert!(board_copy.is_equal_for_testing(hist_copy.get_recent_board(0), true, true).unwrap());
    testAssert!(hist_copy.get_recent_board(0).is_equal_for_testing(hist.get_recent_board(0), true, true).unwrap());
    testAssert!(
      BoardHistory::get_situation_rules_and_ko_hash(&board_copy, &hist_copy, pla, draw_equivalent_wins_for_white)
        == BoardHistory::get_situation_rules_and_ko_hash(&board, &hist, pla, draw_equivalent_wins_for_white)
    );
    testAssert!(
      hist_copy.current_self_komi(P_BLACK, draw_equivalent_wins_for_white) == hist.current_self_komi(P_BLACK, draw_equivalent_wins_for_white)
    );
    testAssert!(
      hist_copy.current_self_komi(P_WHITE, draw_equivalent_wins_for_white) == hist.current_self_komi(P_WHITE, draw_equivalent_wins_for_white)
    );
    testAssert!(hist_copy.initial_turn_number == hist.initial_turn_number);
    testAssert!(hist_copy.presumed_next_move_pla == hist.presumed_next_move_pla);
    testAssert!(hist_copy.assume_multiple_starting_black_moves_are_handicap == hist.assume_multiple_starting_black_moves_are_handicap);
    testAssert!(
      GraphHash::get_state_hash(&hist_copy, pla, draw_equivalent_wins_for_white) == GraphHash::get_state_hash(&hist, pla, draw_equivalent_wins_for_white)
    );
    testAssert!(
      GraphHash::get_graph_hash_from_scratch(&hist_copy, pla, rep_bound, draw_equivalent_wins_for_white)
        == GraphHash::get_graph_hash_from_scratch(&hist, pla, rep_bound, draw_equivalent_wins_for_white)
    );
    testAssert!(any_komi_set || graph_hash == GraphHash::get_graph_hash_from_scratch(&hist, pla, rep_bound, draw_equivalent_wins_for_white));
    cases.ran();
    reps += 1;
  }

  // Not upstream: the draw sequence must be the C++ one (oracle/boardreplay_rng.cpp).
  let fin: u64 = rand.next_uint64();
  let actual = format!(
    "reps {} moves {} passadj {} {} mismatch 0 movefnv {} final {}\n",
    reps,
    fp.moves,
    fp.pass_adj_calls,
    fp.pass_adj_true,
    Global::uint64_to_hex_string(fp.move_fnv),
    Global::uint64_to_hex_string(fin)
  );
  assert_eq!(actual, fixture("oracle", "boardreplay_rng.txt"), "runBoardReplayTest RNG stream differs from C++");
}
