//! Replays upstream cpp/tests/testboardarea.cpp @ v1.16.5 (Tests::runBoardAreaTests) against
//! fixtures/upstream-v1.16.5/unit/testboardarea/ (all 22 expected blocks; the three
//! groupTaxScoring blocks use NNInputs::fillScoring, M2).

mod common;

use common::Cases;
use common::endl;
use common::fixture;
use common::init;
use katago_core::core::test::TestCommon;
use katago_core::game::board::*;
use katago_core::neuralnet::nninputs::NNInputs;
use katago_core::strprintf;
use katago_core::testAssert;

const F: &str = "testboardarea";

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

fn b(v: bool) -> i32 {
  v as i32
}

/// upstream: the `printAreas` lambda in Tests::runBoardAreaTests
fn print_areas(out: &mut String, board: &Board, result: &mut [Color; Board::MAX_ARR_SIZE as usize]) {
  let safe_big_territories_buf: [bool; 4] = [false, true, true, true];
  let unsafe_big_territories_buf: [bool; 4] = [false, false, true, true];
  let non_pass_alive_stones_buf: [bool; 4] = [false, false, false, true];

  for mode in 0..8 {
    let multi_stone_suicide_legal = mode % 2 == 1;
    let safe_big_territories = safe_big_territories_buf[mode / 2];
    let unsafe_big_territories = unsafe_big_territories_buf[mode / 2];
    let non_pass_alive_stones = non_pass_alive_stones_buf[mode / 2];
    let copy = board.clone();
    copy.calculate_area(result, non_pass_alive_stones, safe_big_territories, unsafe_big_territories, multi_stone_suicide_legal);
    *out += &format!(
      "Safe big territories {} Unsafe big territories {} Non pass alive stones {} Suicide {}\n",
      b(safe_big_territories),
      b(unsafe_big_territories),
      b(non_pass_alive_stones),
      b(multi_stone_suicide_legal)
    );
    for y in 0..copy.y_size {
      for x in 0..copy.x_size {
        let loc = Location::get_loc(x, y, copy.x_size);
        out.push(PlayerIO::color_to_char(result[loc as usize]));
      }
      endl(out);
    }
    for i in 0..Board::MAX_ARR_SIZE {
      if !board.is_on_board(i as Loc) {
        testAssert!(result[i as usize] == C_EMPTY);
      }
    }
    endl(out);
    testAssert!(boards_seem_equal(&copy, board));
    copy.check_consistency().unwrap();
  }
}

/// upstream: the `printIndependentLifeAreas` lambda in Tests::runBoardAreaTests
fn print_independent_life_areas(out: &mut String, board: &Board, result: &mut [Color; Board::MAX_ARR_SIZE as usize]) {
  let keep_territories_buf: [bool; 4] = [false, true, false, true];
  let keep_stones_buf: [bool; 4] = [false, false, true, true];

  for mode in 0..8 {
    let multi_stone_suicide_legal = mode % 2 == 1;
    let keep_territories = keep_territories_buf[mode / 2];
    let keep_stones = keep_stones_buf[mode / 2];
    let mut white_minus_black_independent_life_region_count: i32 = 0;
    let copy = board.clone();
    copy.calculate_independent_life_area(
      result,
      &mut white_minus_black_independent_life_region_count,
      keep_territories,
      keep_stones,
      multi_stone_suicide_legal,
    );
    *out += &format!("Keep Territories {} Keep Stones {} Suicide {}\n", b(keep_territories), b(keep_stones), b(multi_stone_suicide_legal));
    *out += &format!("whiteMinusBlackIndependentLifeRegionCount {}\n", white_minus_black_independent_life_region_count);
    for y in 0..copy.y_size {
      for x in 0..copy.x_size {
        let loc = Location::get_loc(x, y, copy.x_size);
        out.push(PlayerIO::color_to_char(result[loc as usize]));
      }
      endl(out);
    }
    for i in 0..Board::MAX_ARR_SIZE {
      if !board.is_on_board(i as Loc) {
        testAssert!(result[i as usize] == C_EMPTY);
      }
    }
    endl(out);
    testAssert!(boards_seem_equal(&copy, board));
    copy.check_consistency().unwrap();
  }
}

/// One case: parse each (x, y, input file), print with `f`, compare with `<slug>.expected0.txt`.
fn case(name: &str, slug: &str, boards: &[(i32, i32)], independent_life: bool, out_sep: Option<&str>) {
  let mut out = String::new();
  let mut result: [Color; Board::MAX_ARR_SIZE as usize] = [0; Board::MAX_ARR_SIZE as usize];
  for (k, (x, y)) in boards.iter().enumerate() {
    if k > 0 {
      if let Some(sep) = out_sep {
        out += sep;
      }
    }
    let board = parse(*x, *y, &fixture(F, &format!("{}.input{}.txt", slug, k)));
    if independent_life {
      print_independent_life_areas(&mut out, &board, &mut result);
    } else {
      print_areas(&mut out, &board, &mut result);
    }
  }
  let expected = fixture(F, &format!("{}.expected0.txt", slug));
  TestCommon::expect_string(name, &mut out, &expected);
}

#[test]
fn area_cases() {
  init();
  let mut cases = Cases::new("testboardarea::area_cases");
  case("Area 1", "area-1", &[(9, 9)], false, None);
  cases.ran();
  // `printAreas(board); out << "-----" << endl; printAreas(board2);`
  case("Area 2", "area-2", &[(9, 9), (9, 9)], false, Some("-----\n"));
  cases.ran();
  case("Area 3", "area-3", &[(19, 19)], false, None);
  cases.ran();
  case("Area 4", "area-4", &[(19, 19)], false, None);
  cases.ran();
  case("Area 5", "area-5", &[(19, 13)], false, None);
  cases.ran();
  case("Area Rect", "area-rect", &[(12, 3)], false, None);
  cases.ran();
  case("Pass alive bug-derived test case", "pass-alive-bug-derived-test-case", &[(9, 9)], false, None);
  cases.ran();
  case("One more simple test case", "one-more-simple-test-case", &[(7, 7)], false, None);
  cases.ran();
  case(
    "Bug-derived test case, in more colors and orientations",
    "bug-derived-test-case-in-more-colors-and-orientations",
    &[(9, 9)],
    false,
    None,
  );
  cases.ran();
  case("More tests for recursive safe", "more-tests-for-recursive-safe", &[(19, 19)], false, None);
  cases.ran();
}

#[test]
fn is_non_pass_alive_self_connection() {
  init();
  let mut cases = Cases::new("testboardarea::is_non_pass_alive_self_connection");
  let mut out = String::new();
  let name = "isNonPassAliveSelfConnection";
  let mut result: [Color; Board::MAX_ARR_SIZE as usize] = [0; Board::MAX_ARR_SIZE as usize];
  let board = parse(9, 9, &fixture(F, "isnonpassaliveselfconnection.input0.txt"));

  let multi_stone_suicide_legal = false;
  let non_pass_alive_stones = false;
  let safe_big_territories = true;
  let unsafe_big_territories = false;
  board.calculate_area(&mut result, non_pass_alive_stones, safe_big_territories, unsafe_big_territories, multi_stone_suicide_legal);

  endl(&mut out);
  for (label, pla) in [("NonPassAliveSelfConn black", P_BLACK), ("NonPassAliveSelfConn white", P_WHITE)] {
    out += label;
    endl(&mut out);
    for y in 0..board.y_size {
      for x in 0..board.x_size {
        let loc = Location::get_loc(x, y, board.x_size);
        if board.colors[loc as usize] == C_EMPTY {
          out += &b(board.is_non_pass_alive_self_connection(loc, pla, &result)).to_string();
        } else {
          testAssert!(board.is_non_pass_alive_self_connection(loc, pla, &result) == false);
          out += "-";
        }
      }
      endl(&mut out);
    }
    endl(&mut out);
  }

  let expected = fixture(F, "isnonpassaliveselfconnection.expected0.txt");
  TestCommon::expect_string(name, &mut out, &expected);
  cases.ran();
}

#[test]
fn independent_life_cases() {
  init();
  let mut cases = Cases::new("testboardarea::independent_life_cases");
  case("IndependentLife 1", "independentlife-1", &[(19, 19)], true, None);
  cases.ran();
  case("IndependentLife 2", "independentlife-2", &[(19, 19)], true, None);
  cases.ran();
  case("IndependentLife 3,4,5", "independentlife-3-4-5", &[(15, 4), (15, 4), (15, 4)], true, None);
  cases.ran();
  case("IndependentLife 6", "independentlife-6", &[(15, 4), (15, 4)], true, None);
  cases.ran();
  case("IndependentLife 7", "independentlife-7", &[(15, 5)], true, None);
  cases.ran();
  case("IndependentLife 8", "independentlife-8", &[(15, 5)], true, None);
  cases.ran();
  case("IndependentLife 9", "independentlife-9", &[(15, 5)], true, None);
  cases.ran();
  case("IndependentLife 10", "independentlife-10", &[(15, 5)], true, None);
  cases.ran();
}

/// upstream: the three groupTaxScoring blocks of Tests::runBoardAreaTests: calculateArea with the
/// given flags, then NNInputs::fillScoring without and with group tax, printed with "%4.0f ".
fn group_tax_case(
  out: &mut String,
  name: &str,
  slug: &str,
  multi_stone_suicide_legal: bool,
  non_pass_alive_stones: bool,
  safe_big_territories: bool,
  unsafe_big_territories: bool,
) {
  let mut result: [Color; Board::MAX_ARR_SIZE as usize] = [0; Board::MAX_ARR_SIZE as usize];
  let board = parse(13, 13, &fixture(F, &format!("{}.input0.txt", slug)));
  board.calculate_area(&mut result, non_pass_alive_stones, safe_big_territories, unsafe_big_territories, multi_stone_suicide_legal);
  let mut scoring: [f32; Board::MAX_ARR_SIZE as usize] = [0.0; Board::MAX_ARR_SIZE as usize];

  endl(out);
  *out += "No group tax";
  endl(out);
  NNInputs::fill_scoring(&board, &result, false, &mut scoring);
  for y in 0..board.y_size {
    for x in 0..board.x_size {
      let loc = Location::get_loc(x, y, board.x_size);
      *out += &strprintf!("%4.0f ", scoring[loc as usize] * 100.0f32);
    }
    endl(out);
  }
  endl(out);
  *out += "Group tax";
  endl(out);
  NNInputs::fill_scoring(&board, &result, true, &mut scoring);
  for y in 0..board.y_size {
    for x in 0..board.x_size {
      let loc = Location::get_loc(x, y, board.x_size);
      *out += &strprintf!("%4.0f ", scoring[loc as usize] * 100.0f32);
    }
    endl(out);
  }
  endl(out);

  let expected = fixture(F, &format!("{}.expected0.txt", slug));
  TestCommon::expect_string(name, out, &expected);
}

#[test]
fn group_tax_scoring_cases() {
  init();
  let mut cases = Cases::new("testboardarea::group_tax_scoring_cases");
  let mut out = String::new();
  group_tax_case(&mut out, "groupTaxScoring", "grouptaxscoring", false, true, true, true);
  cases.ran();
  group_tax_case(&mut out, "groupTaxScoring2", "grouptaxscoring2", true, true, true, false);
  cases.ran();
  group_tax_case(
    &mut out,
    "groupTaxScoring2 but with unsafe territories on",
    "grouptaxscoring2-but-with-unsafe-territories-on",
    true,
    true,
    true,
    true,
  );
  cases.ran();
}
