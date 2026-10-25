//! Replays upstream cpp/tests/testsymmetries.cpp @ v1.16.5 (Tests::runBasicSymmetryTests,
//! Tests::runBoardSymmetryTests, Tests::runSymmetryDifferenceTests) against
//! fixtures/upstream-v1.16.5/unit/testsymmetries/ (all 7 expected blocks), and
//! Tests::runNNSymmetryTests (cpp/tests/testnn.cpp), which prints to cout during
//! `katago runoutputtests`, against its part of upstream cpp/tests/results/runOutputTests.txt
//! (fixtures/upstream-v1.16.5/unit/runoutputtests/nn-symmetry-tests.expected0.txt, extracted by
//! tools/extract_runoutputtests.py).
#![allow(clippy::needless_range_loop, clippy::bool_comparison)]

// Test code keeps the shapes of upstream's tests and oracle programs (PORTING.md §11b).
#![allow(clippy::type_complexity)]

mod common;

use std::fmt::Write;

use common::Cases;
use common::endl;
use common::fixture;
use common::init;
use katago_core::core::global::Global;
use katago_core::core::test::TestCommon;
use katago_core::game::board::*;
use katago_core::game::boardhistory::BoardHistory;
use katago_core::game::rules::Rules;
use katago_core::neuralnet::nninputs::SymmetryHelpers;
use katago_core::port::fmt::cpp_ostream_f64;
use katago_core::strprintf;
use katago_core::testAssert;

const F: &str = "testsymmetries";

fn parse(x: i32, y: i32, s: &str) -> Board {
  Board::parse_board(x, y, s, '\n').unwrap()
}

fn input(slug: &str, k: usize) -> String {
  fixture(F, &format!("{}.input{}.txt", slug, k))
}

/// `out << board << endl`
fn print_board(out: &mut String, board: &Board) {
  write!(out, "{}", board).unwrap();
  endl(out);
}

/// `ostream << float` (default precision 6)
fn f(x: f32) -> String {
  cpp_ostream_f64(x as f64)
}

/// upstream: Tests::runBasicSymmetryTests (testsymmetries.cpp)
#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn run_basic_symmetry_tests() {
  init();
  let mut cases = Cases::new("testsymmetries::run_basic_symmetry_tests");
  let mut out = String::new();

  {
    let name = "Basic flipping of tensors and boards and locations";
    let slug = "basic-flipping-of-tensors-and-boards-and-locations";
    {
      let board = parse(7, 7, &input(slug, 0));
      #[rustfmt::skip]
      let buf: [f32; 49] = [
        0.0,0.0,0.0,0.0,0.0,0.0,0.0,
        0.0,0.0,0.0,0.0,0.0,0.0,0.0,
        0.0,1.0,1.0,1.0,0.0,0.0,0.0,
        0.0,0.0,0.0,0.0,2.0,0.0,0.0,
        0.0,0.0,0.0,0.0,2.0,0.0,0.0,
        0.0,0.0,0.0,0.0,2.0,0.0,0.0,
        0.0,0.0,0.0,0.0,0.0,0.0,0.0,
      ];

      for symmetry in 0..SymmetryHelpers::NUM_SYMMETRIES {
        let mut dst: [f32; 49] = [0.0; 49];
        SymmetryHelpers::copy_inputs_with_symmetry(&buf, &mut dst, 1, 7, 7, 1, true, symmetry);
        let sym_board = SymmetryHelpers::get_sym_board(&board, symmetry).unwrap();
        let sym_loc_with_x = SymmetryHelpers::get_sym_loc_x_y(1, 2, &board, symmetry);
        out += &format!("SYMMETRY {}\n", symmetry);
        print_board(&mut out, &sym_board);
        out += &Global::bool_to_string(sym_board.colors[sym_loc_with_x as usize] == P_BLACK);
        endl(&mut out);
        for y in 0..7 {
          for x in 0..7 {
            out += &f(dst[x + y * 7]);
            out += " ";
          }
          endl(&mut out);
        }
      }
    }

    {
      let mut board = parse(7, 5, &input(slug, 1));
      let ko = Location::get_loc(5, 0, board.x_size);
      board.set_simple_ko_loc(ko);

      for symmetry in 0..SymmetryHelpers::NUM_SYMMETRIES {
        let sym_board = SymmetryHelpers::get_sym_board(&board, symmetry).unwrap();
        out += &format!("SYMMETRY {}\n", symmetry);
        print_board(&mut out, &sym_board);
        out += &format!("{}\n", sym_board.get_sit_hash_with_simple_ko(P_BLACK));
        let sym_loc_with_x = SymmetryHelpers::get_sym_loc_x_y(1, 1, &board, symmetry);
        let sym_loc_with_x2 = SymmetryHelpers::get_sym_loc_x_y(1, 2, &board, symmetry);
        let sym_loc_with_o = SymmetryHelpers::get_sym_loc_x_y(1, 3, &board, symmetry);
        let sym_loc_with_o2 = SymmetryHelpers::get_sym_loc_x_y(2, 3, &board, symmetry);
        out += &format!("{}\n", Global::bool_to_string(sym_board.colors[sym_loc_with_x as usize] == P_BLACK));
        out += &format!("{}\n", Global::bool_to_string(sym_board.colors[sym_loc_with_x2 as usize] == P_BLACK));
        out += &format!("{}\n", Global::bool_to_string(sym_board.colors[sym_loc_with_o as usize] == P_WHITE));
        out += &format!("{}\n", Global::bool_to_string(sym_board.colors[sym_loc_with_o2 as usize] == P_WHITE));
        out += &format!(
          "{}\n",
          Global::bool_to_string(sym_board.ko_loc == SymmetryHelpers::get_sym_loc_x_y(5, 0, &board, symmetry))
        );
      }
    }

    TestCommon::expect_string(name, &mut out, &fixture(F, &format!("{}.expected0.txt", slug)));
    cases.ran();
  }

  {
    let board = parse(12, 7, &input("basic-flipping-of-tensors-and-boards-and-locations", 2));
    for symmetry in 0..SymmetryHelpers::NUM_SYMMETRIES {
      testAssert!(
        SymmetryHelpers::get_sym_loc_x_y(1, 2, &board, symmetry)
          == SymmetryHelpers::get_sym_loc(Location::get_loc(1, 2, board.x_size), &board, symmetry)
      );
      testAssert!(
        SymmetryHelpers::get_sym_loc_x_y(1, 2, &board, symmetry)
          == SymmetryHelpers::get_sym_loc_x_size_y_size(Location::get_loc(1, 2, board.x_size), 12, 7, symmetry)
      );
      testAssert!(
        SymmetryHelpers::get_sym_loc_x_y(1, 2, &board, symmetry)
          == SymmetryHelpers::get_sym_loc_x_y_x_size_y_size(1, 2, 12, 7, symmetry)
      );
    }
    cases.ran();
  }

  {
    let name = "Symmetry composition tests";
    let slug = "symmetry-composition-tests";
    let board = parse(9, 7, &input(slug, 0));

    let loc = Location::get_loc(1, 2, board.x_size);
    for symmetry1 in 0..SymmetryHelpers::NUM_SYMMETRIES {
      for symmetry2 in 0..SymmetryHelpers::NUM_SYMMETRIES {
        let symmetry_composed = SymmetryHelpers::compose(symmetry1, symmetry2);
        let sym_board_comb = SymmetryHelpers::get_sym_board(&board, symmetry_composed).unwrap();
        let sym_board_comb_manual = SymmetryHelpers::get_sym_board(&SymmetryHelpers::get_sym_board(&board, symmetry1).unwrap(), symmetry2).unwrap();
        let sym_loc_comb = SymmetryHelpers::get_sym_loc(loc, &board, symmetry_composed);
        let sym_loc_comb_manual = SymmetryHelpers::get_sym_loc(
          SymmetryHelpers::get_sym_loc(loc, &board, symmetry1),
          &SymmetryHelpers::get_sym_board(&board, symmetry1).unwrap(),
          symmetry2,
        );
        out += &format!("Symmetry {} + {} = {}\n", symmetry1, symmetry2, symmetry_composed);
        testAssert!(sym_board_comb_manual.is_equal_for_testing(&sym_board_comb, true, true).unwrap());
        testAssert!(sym_loc_comb == sym_loc_comb_manual);
      }
    }
    TestCommon::expect_string(name, &mut out, &fixture(F, &format!("{}.expected0.txt", slug)));
    cases.ran();
  }

  {
    let name = "Symmetry inverse tests";
    for symmetry1 in 0..SymmetryHelpers::NUM_SYMMETRIES {
      for symmetry2 in 0..SymmetryHelpers::NUM_SYMMETRIES {
        out += &format!(
          "Symmetry {} : {} inverses {}\n",
          symmetry1,
          symmetry2,
          Global::bool_to_string(SymmetryHelpers::invert(symmetry1) == symmetry2)
        );
        testAssert!((SymmetryHelpers::compose(symmetry1, symmetry2) == 0) == (SymmetryHelpers::invert(symmetry1) == symmetry2));
      }
    }
    TestCommon::expect_string(name, &mut out, &fixture(F, "symmetry-inverse-tests.expected0.txt"));
    cases.ran();
  }

  {
    let name = "Symmetry and move commute";
    let slug = "symmetry-and-move-commute";
    let board = parse(9, 7, &input(slug, 0));

    let loc = Location::get_loc(8, 4, board.x_size);
    for symmetry in 0..SymmetryHelpers::NUM_SYMMETRIES {
      let mut board_a = SymmetryHelpers::get_sym_board(&board, symmetry).unwrap();
      board_a.play_move(SymmetryHelpers::get_sym_loc(loc, &board, symmetry), P_BLACK, true);

      let mut board_b = board.clone();
      board_b.play_move(loc, P_BLACK, true);
      board_b = SymmetryHelpers::get_sym_board(&board_b, symmetry).unwrap();

      out += &format!("SYMMETRY {}\n", symmetry);
      print_board(&mut out, &board_a);
      print_board(&mut out, &board_b);
      testAssert!(board_a.is_equal_for_testing(&board_b, true, true).unwrap());
    }
    TestCommon::expect_string(name, &mut out, &fixture(F, &format!("{}.expected0.txt", slug)));
    cases.ran();
  }
}

/// upstream: the `printMarkedSymDupArea` lambda in Tests::runBoardSymmetryTests
fn print_marked_sym_dup_area(out: &mut String, board: &Board, is_sym_dup_loc: &[bool], valid_symmetries: &[i32]) {
  *out += "Symmetries: ";
  for symmetry in valid_symmetries {
    *out += &format!("{} ", symmetry);
  }
  endl(out);
  for y in 0..board.y_size {
    for x in 0..board.x_size {
      let loc = Location::get_loc(x, y, board.x_size);
      if is_sym_dup_loc[loc as usize] {
        out.push('x');
      } else {
        out.push('.');
      }
    }
    endl(out);
  }
  endl(out);
}

/// upstream: the `computeAndPrintMarkedSymDupArea` lambda in Tests::runBoardSymmetryTests
fn compute_and_print_marked_sym_dup_area(out: &mut String, board: &Board, pla: Player, only_symmetries: Option<&[i32]>) {
  let hist = BoardHistory::new_full(board, pla, &Rules::get_tromp_taylorish(), 0);
  let mut is_sym_dup_loc: [bool; Board::MAX_ARR_SIZE as usize] = [false; Board::MAX_ARR_SIZE as usize];
  let mut valid_symmetries: Vec<i32> = Vec::new();
  let avoid_moves: Vec<i32> = Vec::new();
  SymmetryHelpers::mark_duplicate_move_locs(board, &hist, only_symmetries, &avoid_moves, &mut is_sym_dup_loc, &mut valid_symmetries);
  print_board(out, board);
  print_marked_sym_dup_area(out, board, &is_sym_dup_loc, &valid_symmetries);
}

/// upstream: Tests::runBoardSymmetryTests (testsymmetries.cpp)
#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn run_board_symmetry_tests() {
  init();
  let mut cases = Cases::new("testsymmetries::run_board_symmetry_tests");
  let mut out = String::new();
  let name = "Testing SymmetryHelpers::markDuplicateMoveLocs";
  let slug = "testing-symmetryhelpers-markduplicatemovelocs";

  // (label, x, y, next player, onlySymmetries), in upstream order; input k is the k-th board.
  let only_no_56: Vec<i32> = vec![0, 1, 2, 3, 4, 7];
  let only_no_1247: Vec<i32> = vec![0, 3, 5, 6];
  let only_hflip: Vec<i32> = vec![0, 2];
  let boards: [(&str, i32, i32, Player, Option<&[i32]>); 24] = [
    ("Fully symmetric board", 9, 9, P_BLACK, None),
    ("Fully symmetric board white next", 9, 9, P_WHITE, None),
    ("X-flip symmetry", 9, 9, P_BLACK, None),
    ("Y-flip symmetry", 9, 9, P_BLACK, None),
    ("Diagonal-flips symmetry", 9, 9, P_BLACK, None),
    ("2 fold-rotational symmetry", 9, 9, P_BLACK, None),
    ("Square-flips symmetry", 9, 9, P_BLACK, None),
    ("Diagonal-flips symmetry", 9, 9, P_BLACK, None),
    ("2 fold-rotational symmetry", 9, 9, P_BLACK, None),
    ("4 fold-rotational symmetry", 9, 9, P_BLACK, None),
    ("4 fold-rotational symmetry white next", 9, 9, P_WHITE, None),
    ("UR-only diagonal flip symmetry", 9, 9, P_BLACK, None),
    ("UL-only diagonal flip symmetry", 9, 9, P_BLACK, None),
    ("Not symmetric", 9, 9, P_BLACK, None),
    ("Empty rectangular board", 5, 7, P_BLACK, None),
    ("Fully symmetric rectangular board", 5, 7, P_BLACK, None),
    ("X-flip rectangular board", 5, 7, P_BLACK, None),
    ("Y-flip rectangular board", 5, 7, P_BLACK, None),
    ("Both-flip rectangular board", 5, 7, P_BLACK, None),
    ("Both-flip rectangular board white next", 5, 7, P_WHITE, None),
    ("Not symmetric", 5, 7, P_BLACK, None),
    ("4 fold-rotational symmetry, no 5 6", 9, 9, P_BLACK, Some(only_no_56.as_slice())),
    ("Empty board, no 1,2,4,7", 9, 9, P_BLACK, Some(only_no_1247.as_slice())),
    ("Empty board, only hflip", 9, 9, P_BLACK, Some(only_hflip.as_slice())),
  ];
  for (k, (label, x, y, pla, only)) in boards.iter().enumerate() {
    out += label;
    endl(&mut out);
    let board = parse(*x, *y, &input(slug, k));
    compute_and_print_marked_sym_dup_area(&mut out, &board, *pla, *only);
    cases.ran();
  }
  TestCommon::expect_string(name, &mut out, &fixture(F, &format!("{}.expected0.txt", slug)));
}

/// upstream: the `testSymmetryDifferences` lambda in Tests::runSymmetryDifferenceTests
fn test_symmetry_differences(out: &mut String, board: &Board, other: &Board, max_difference_to_report: f64) {
  let mut diffs: [f64; SymmetryHelpers::NUM_SYMMETRIES as usize] = [0.0; SymmetryHelpers::NUM_SYMMETRIES as usize];
  SymmetryHelpers::get_symmetry_differences(board, other, max_difference_to_report, &mut diffs);
  *out += &strprintf!(
    "%.1f %.1f %.1f %.1f %.1f %.1f %.1f %.1f",
    diffs[0],
    diffs[1],
    diffs[2],
    diffs[3],
    diffs[4],
    diffs[5],
    diffs[6],
    diffs[7]
  );
  endl(out);
}

/// upstream: Tests::runSymmetryDifferenceTests (testsymmetries.cpp)
#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn run_symmetry_difference_tests() {
  init();
  let mut cases = Cases::new("testsymmetries::run_symmetry_difference_tests");
  let mut out = String::new();
  {
    let name = "Simple difference tests";
    let slug = "simple-difference-tests";
    let board = parse(5, 5, &input(slug, 0));
    let board2 = parse(5, 5, &input(slug, 1));
    let board3 = parse(5, 5, &input(slug, 2));
    let board4 = parse(5, 5, &input(slug, 3));
    test_symmetry_differences(&mut out, &board, &board, 20.0);
    test_symmetry_differences(&mut out, &board, &board2, 20.0);
    test_symmetry_differences(&mut out, &board, &board3, 20.0);
    test_symmetry_differences(&mut out, &board, &board4, 20.0);
    test_symmetry_differences(&mut out, &board, &board4, 7.0);
    TestCommon::expect_string(name, &mut out, &fixture(F, &format!("{}.expected0.txt", slug)));
    cases.ran();
  }
  {
    let name = "All 8 symmetries difference tests";
    let slug = "all-8-symmetries-difference-tests";
    let board5 = parse(5, 5, &input(slug, 0));
    for i in 0..8 {
      let b = SymmetryHelpers::get_sym_board(&board5, i).unwrap();
      test_symmetry_differences(&mut out, &board5, &b, 99.0);
    }
    TestCommon::expect_string(name, &mut out, &fixture(F, &format!("{}.expected0.txt", slug)));
    cases.ran();
  }
}

/// upstream: NCHWtoNHWC (testnn.cpp) static
fn nchw_to_nhwc(vec: &[f32], n_size: usize, c_size: usize, y_size: usize, x_size: usize) -> Vec<f32> {
  let mut ret: Vec<f32> = vec![0.0; vec.len()];
  let cyx_size = c_size * y_size * x_size;
  let yx_size = y_size * x_size;
  let xc_size = x_size * c_size;
  for n in 0..n_size {
    for c in 0..c_size {
      for y in 0..y_size {
        for x in 0..x_size {
          ret[n * cyx_size + y * xc_size + x * c_size + c] = vec[n * cyx_size + c * yx_size + y * x_size + x];
        }
      }
    }
  }
  ret
}

/// upstream: the `testConfigurations` lambda in Tests::runNNSymmetryTests (testnn.cpp)
fn test_configurations(out: &mut String, label: &str, batch_size: i32, num_channels: i32, nn_x_len: i32, nn_y_len: i32, input: &[f32]) {
  for use_nhwc in 0..=1 {
    for symmetry in 0..8 {
      let input_this_loop: Vec<f32> = if use_nhwc != 0 {
        nchw_to_nhwc(input, batch_size as usize, num_channels as usize, nn_y_len as usize, nn_x_len as usize)
      } else {
        input.to_vec()
      };
      let mut output_this_loop: Vec<f32> = vec![0.0; input_this_loop.len()];
      // Upstream passes (nnXLen, nnYLen) as (hSize, wSize); kept as is.
      SymmetryHelpers::copy_inputs_with_symmetry(
        &input_this_loop,
        &mut output_this_loop,
        batch_size,
        nn_x_len,
        nn_y_len,
        num_channels,
        use_nhwc != 0,
        symmetry,
      );
      *out += &format!("{} useNHWC {} {}\n", label, use_nhwc, symmetry);
      for i in 0..output_this_loop.len() {
        *out += &f(output_this_loop[i]);
        *out += " ";
      }
      endl(out);
    }
  }
  for symmetry in 0..8 {
    let input_this_loop: Vec<f32> = input.to_vec();
    let mut output_this_loop: Vec<f32> = vec![0.0; input_this_loop.len()];
    SymmetryHelpers::copy_outputs_with_symmetry(&input_this_loop, &mut output_this_loop, batch_size * num_channels, nn_x_len, nn_y_len, symmetry);
    *out += &format!("{} OUTPUT \n", label);
    for i in 0..output_this_loop.len() {
      *out += &f(output_this_loop[i]);
      *out += " ";
    }
    endl(out);
  }
}

/// upstream: Tests::runNNSymmetryTests (testnn.cpp); its cout is part of runOutputTests.txt.
#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn run_nn_symmetry_tests() {
  init();
  let mut cases = Cases::new("testsymmetries::run_nn_symmetry_tests");
  let mut out = String::new();
  {
    {
      //NCHW
      #[rustfmt::skip]
      let input: Vec<f32> = vec![
        0.0,1.0,2.0,
        3.0,4.0,5.0,
        6.0,7.0,8.0,

        3.0,0.0,4.0,
        0.0,5.0,0.0,
        0.0,6.0,0.0,

        1.0,0.0,0.0,
        1.0,1.0,1.0,
        1.0,0.0,1.0,
      ];
      test_configurations(&mut out, "Symmetry 3-1-3-3", 3, 1, 3, 3, &input);
      cases.ran();
      test_configurations(&mut out, "Symmetry 1-3-3-3", 1, 3, 3, 3, &input);
      cases.ran();
    }

    {
      //NCHW
      #[rustfmt::skip]
      let input: Vec<f32> = vec![
        0.0,1.0,2.0,3.0,
        4.0,5.0,6.0,7.0,
        8.0,9.0,10.0,11.0,

        12.0,13.0,14.0,15.0,
        16.0,17.0,18.0,19.0,
        20.0,21.0,22.0,23.0,
      ];
      test_configurations(&mut out, "Symmetry 2-1-3-4", 2, 1, 3, 4, &input);
      cases.ran();
      test_configurations(&mut out, "Symmetry 2-3-2-2", 2, 3, 2, 2, &input);
      cases.ran();
    }
  }
  let expected = fixture("runoutputtests", "nn-symmetry-tests.expected0.txt");
  common::compare_lines("runNNSymmetryTests", &out, &expected);
}
