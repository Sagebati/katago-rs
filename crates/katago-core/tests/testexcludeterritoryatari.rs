//! Replays the game/ parts of upstream cpp/tests/testexcludeterritoryatari.cpp @ v1.18.2
//! (Tests::runExcludeTerritoryAtariModeTests): BoardHistoryModes::excludeTerritoryAdjacentToAtari in
//! Board::calculateIndependentLifeArea, BoardHistory scoring, getSituationRulesAndKoHash (both overloads),
//! setModes, and the copy / assignment / clear() / copyToInitial preservation of the modes. Upstream
//! prints only a completion line; every check is a testAssert, so the oracle is the asserts themselves.
//!
//! Not ported here (TODO(v1.18.2 sync), they need code other areas re-sync): "MiscNNInputParams override
//! behavior for the nn cache hash" (MiscNNInputParams::excludeTerritoryAdjAtariOverride /
//! getExcludeTerritoryAdjacentToAtari, NNInputs::getHash), "Search maintains the invariant ..." (Search::
//! resolveExcludeTerritoryAdjacentToAtari, SearchParams::excludeTerritoryAdjacentToAtari, NNEvaluator::
//! modelPreferExcludeTerritoryAdjacentToAtari), "Books record the BoardHistoryModes ..." (book/, not ported),
//! "The nn cache cannot mix results across featurization modes" (NNEvaluator::evaluate with overrides),
//! "NN input features 18/19 ..." (fillRowV7 with the mode).

// Test code keeps the shapes of upstream's tests (PORTING.md §11b).
#![allow(clippy::needless_range_loop, clippy::redundant_clone)]

mod common;

use common::Cases;
use common::init;
use katago_core::game::board::*;
use katago_core::game::boardhistory::*;
use katago_core::game::rules::Rules;
use katago_core::testAssert;

fn parse(x: i32, y: i32, s: &str) -> Board {
  Board::parse_board(x, y, s, '\n').unwrap()
}

/// The whole-board seki with an unfilled ko mouth at the top of runExcludeTerritoryAtariModeTests.
fn seki_board() -> Board {
  //A whole-board seki with an unfilled ko mouth. The top-right black group's only liberty is its
  //eye at (5,1), so the group is in atari and under rules v3 (excludeTerritoryAdjacentToAtari)
  //the eye is not a point for black under territory scoring with TaxRule NONE, while under
  //rules v2 it is.
  parse(7, 7, r"
...oxxx
oooox.x
xxxxoxx
o.xoooo
ooxox.o
oxxo.xo
o.xooxx
")
}

fn terr_no_tax_rules() -> Rules {
  let mut terr_no_tax_rules = Rules::new();
  terr_no_tax_rules.ko_rule = Rules::KO_SIMPLE;
  terr_no_tax_rules.scoring_rule = Rules::SCORING_TERRITORY;
  terr_no_tax_rules.tax_rule = Rules::TAX_NONE;
  terr_no_tax_rules.multi_stone_suicide_legal = false;
  terr_no_tax_rules.komi = -0.5f32;
  terr_no_tax_rules
}

fn modes(a: bool, b: bool) -> BoardHistoryModes {
  BoardHistoryModes::new_full(a, b)
}

#[test]
fn exclude_territory_atari_mode_game_tests() {
  init();
  let mut cases = Cases::new("testexcludeterritoryatari::exclude_territory_atari_mode_game_tests");
  let board = seki_board();
  let terr_no_tax_rules = terr_no_tax_rules();

  let eye_loc: Loc = Location::get_loc(5, 1, board.x_size);
  testAssert!(board.colors[eye_loc as usize] == C_EMPTY);
  testAssert!(board.get_num_liberties(Location::get_loc(5, 0, board.x_size)) == 1);

  //Sanity check that this position really does discriminate the two computations.
  {
    let mut area_v2 = [0 as Color; Board::MAX_ARR_SIZE as usize];
    let mut area_v3 = [0 as Color; Board::MAX_ARR_SIZE as usize];
    let mut count: i32 = 0;
    board.calculate_independent_life_area(&mut area_v2, &mut count, true, false, false, false);
    board.calculate_independent_life_area(&mut area_v3, &mut count, true, false, true, false);
    testAssert!(area_v2[eye_loc as usize] == C_BLACK);
    testAssert!(area_v3[eye_loc as usize] == C_EMPTY);
    cases.ran();
  }

  //Basic flag behavior and effect on scoring
  {
    let hist = BoardHistory::new_full(&board, P_BLACK, &terr_no_tax_rules, 0, &modes(false, false));
    testAssert!(!hist.modes.exclude_territory_adjacent_to_atari);

    let mut hist_flagged = BoardHistory::new_full(&board, P_BLACK, &terr_no_tax_rules, 0, &modes(false, false));
    hist_flagged.set_modes(&modes(false, true));
    testAssert!(hist_flagged.modes.exclude_territory_adjacent_to_atari);

    //The flag changes the situation-and-rules hash under territory scoring with TaxRule NONE
    let hash_off = BoardHistory::get_situation_rules_and_ko_hash(&board, &hist, P_BLACK, 0.5);
    let hash_on = BoardHistory::get_situation_rules_and_ko_hash(&board, &hist_flagged, P_BLACK, 0.5);
    testAssert!(hash_off != hash_on);

    //And changes territory scoring by exactly the ko mouth point (black loses 1 point)
    let mut hist_score = BoardHistory::new_full(&board, P_BLACK, &terr_no_tax_rules, 0, &modes(false, false));
    let board_copy = board.clone();
    hist_score.end_and_score_game_now(&board_copy);
    let mut hist_score_flagged = BoardHistory::new_full(&board, P_BLACK, &terr_no_tax_rules, 0, &modes(false, true));
    let board_copy2 = board.clone();
    hist_score_flagged.end_and_score_game_now(&board_copy2);
    testAssert!(hist_score.is_scored && hist_score_flagged.is_scored);
    testAssert!(hist_score_flagged.final_white_minus_black_score == hist_score.final_white_minus_black_score + 1.0f32);

    //Copying and clear() preserve the flag
    let copied = hist_flagged.clone();
    testAssert!(copied.modes.exclude_territory_adjacent_to_atari);
    let mut assigned = BoardHistory::new();
    assigned.clone_from(&hist_flagged);
    testAssert!(assigned.modes.exclude_territory_adjacent_to_atari);
    let mut cleared = hist_flagged.clone();
    cleared.clear(&board, P_BLACK, &terr_no_tax_rules, 0);
    testAssert!(cleared.modes.exclude_territory_adjacent_to_atari);
    testAssert!(hist_flagged.copy_to_initial().modes.exclude_territory_adjacent_to_atari);
    cases.ran();
  }

  //Under any other scoring/tax rules, the flag is a no-op for hashing and scoring
  {
    //Area scoring with TaxRule NONE
    let mut area_rules = terr_no_tax_rules;
    area_rules.scoring_rule = Rules::SCORING_AREA;
    let mut hist = BoardHistory::new_full(&board, P_BLACK, &area_rules, 0, &modes(false, false));
    let mut hist_flagged = BoardHistory::new_full(&board, P_BLACK, &area_rules, 0, &modes(false, true));
    testAssert!(
      BoardHistory::get_situation_rules_and_ko_hash(&board, &hist, P_BLACK, 0.5) ==
      BoardHistory::get_situation_rules_and_ko_hash(&board, &hist_flagged, P_BLACK, 0.5)
    );
    let board_copy = board.clone();
    hist.end_and_score_game_now(&board_copy);
    let board_copy2 = board.clone();
    hist_flagged.end_and_score_game_now(&board_copy2);
    testAssert!(hist.final_white_minus_black_score == hist_flagged.final_white_minus_black_score);

    //Territory scoring with TaxRule SEKI
    let mut terr_seki_rules = terr_no_tax_rules;
    terr_seki_rules.tax_rule = Rules::TAX_SEKI;
    let mut hist2 = BoardHistory::new_full(&board, P_BLACK, &terr_seki_rules, 0, &modes(false, false));
    let mut hist2_flagged = BoardHistory::new_full(&board, P_BLACK, &terr_seki_rules, 0, &modes(false, true));
    testAssert!(
      BoardHistory::get_situation_rules_and_ko_hash(&board, &hist2, P_BLACK, 0.5) ==
      BoardHistory::get_situation_rules_and_ko_hash(&board, &hist2_flagged, P_BLACK, 0.5)
    );
    let board_copy3 = board.clone();
    hist2.end_and_score_game_now(&board_copy3);
    let board_copy4 = board.clone();
    hist2_flagged.end_and_score_game_now(&board_copy4);
    testAssert!(hist2.final_white_minus_black_score == hist2_flagged.final_white_minus_black_score);
    cases.ran();
  }

  //The two flags hash independently
  {
    let hist = BoardHistory::new_full(&board, P_BLACK, &terr_no_tax_rules, 0, &modes(false, false));
    let h00 = BoardHistory::get_situation_rules_and_ko_hash_modes(&board, &hist, P_BLACK, 0.5, &modes(false, false));
    let h01 = BoardHistory::get_situation_rules_and_ko_hash_modes(&board, &hist, P_BLACK, 0.5, &modes(false, true));
    let h10 = BoardHistory::get_situation_rules_and_ko_hash_modes(&board, &hist, P_BLACK, 0.5, &modes(true, false));
    let h11 = BoardHistory::get_situation_rules_and_ko_hash_modes(&board, &hist, P_BLACK, 0.5, &modes(true, true));
    testAssert!(h00 != h01 && h00 != h10 && h00 != h11);
    testAssert!(h01 != h10 && h01 != h11 && h10 != h11);
    cases.ran();
  }

  // TODO(v1.18.2 sync): "MiscNNInputParams override behavior for the nn cache hash",
  // "Search maintains the invariant ...", "Books record the BoardHistoryModes ...",
  // "The nn cache cannot mix results across featurization modes" (see the file header).

  //A pass-alive white group with both phenomena at once: a dead black throw-in stone in atari
  //(A5, with liberty A4) inside its pass-alive territory, and a ko mouth (D7, the sole liberty
  //of the isolated white ko stone E7). The exclusion applies only to chains of the territory
  //owner's own color, so the throw-in does NOT block A4/A5 from counting for white, while the
  //own-color ko stone in atari DOES block the mouth D7 - the only point the modes disagree on.
  {
    let tboard = parse(7, 7, r"
.oo.ox.
oo.oxx.
xoooox.
.oooox.
oooo.x.
o.ooox.
ooooox.
");
    let throw_in_loc: Loc = Location::get_loc(0, 2, tboard.x_size);
    let liberty_loc: Loc = Location::get_loc(0, 3, tboard.x_size);
    let mouth_loc: Loc = Location::get_loc(3, 0, tboard.x_size);
    let ko_stone_loc: Loc = Location::get_loc(4, 0, tboard.x_size);
    testAssert!(tboard.colors[throw_in_loc as usize] == C_BLACK);
    testAssert!(tboard.get_num_liberties(throw_in_loc) == 1);
    testAssert!(tboard.colors[ko_stone_loc as usize] == C_WHITE);
    testAssert!(tboard.get_num_liberties(ko_stone_loc) == 1);

    let mut area_v2 = [0 as Color; Board::MAX_ARR_SIZE as usize];
    let mut area_v3 = [0 as Color; Board::MAX_ARR_SIZE as usize];
    let mut count: i32 = 0;
    tboard.calculate_independent_life_area(&mut area_v2, &mut count, true, false, false, false);
    tboard.calculate_independent_life_area(&mut area_v3, &mut count, true, false, true, false);
    //Opposing-color atari does not block the territory...
    testAssert!(area_v2[liberty_loc as usize] == C_WHITE);
    testAssert!(area_v3[liberty_loc as usize] == C_WHITE);
    testAssert!(area_v2[throw_in_loc as usize] == C_WHITE);
    testAssert!(area_v3[throw_in_loc as usize] == C_WHITE);
    //...but the own-color ko stone in atari blocks the ko mouth, and nothing else differs.
    testAssert!(area_v2[mouth_loc as usize] == C_WHITE);
    testAssert!(area_v3[mouth_loc as usize] == C_EMPTY);
    for i in 0..Board::MAX_ARR_SIZE {
      testAssert!(area_v2[i as usize] == area_v3[i as usize] || i == mouth_loc as i32);
    }

    let mut hist_v2 = BoardHistory::new_full(&tboard, P_BLACK, &terr_no_tax_rules, 0, &modes(false, false));
    let tb1 = tboard.clone();
    hist_v2.end_and_score_game_now(&tb1);
    let mut hist_v3 = BoardHistory::new_full(&tboard, P_BLACK, &terr_no_tax_rules, 0, &modes(false, true));
    let tb2 = tboard.clone();
    hist_v3.end_and_score_game_now(&tb2);
    testAssert!(hist_v2.final_white_minus_black_score == hist_v3.final_white_minus_black_score + 1.0f32);
    cases.ran();
  }

  // TODO(v1.18.2 sync): "NN input features 18/19 (current territory) reflect the flag ..." needs
  // NNInputs::fillRowV7 with MiscNNInputParams::getExcludeTerritoryAdjacentToAtari (neuralnet/nninputs).

  //Torazu sanmoku ("three points without capturing")
  {
    let tboard = parse(9, 9, r"
.xxo.....
oxxo.....
xooo.....
xxxoooooo
xxxxxxxxx
.........
.........
.........
.........
");
    let shared_lib_loc: Loc = Location::get_loc(0, 0, tboard.x_size);
    let dango_loc: Loc = Location::get_loc(1, 0, tboard.x_size);
    let lone_stone_loc: Loc = Location::get_loc(0, 1, tboard.x_size);
    testAssert!(tboard.colors[shared_lib_loc as usize] == C_EMPTY);
    //Mutual atari: black's four-stone dango and white's lone stone share the 1-1 point.
    testAssert!(tboard.colors[dango_loc as usize] == C_BLACK);
    testAssert!(tboard.get_num_liberties(dango_loc) == 1);
    testAssert!(tboard.colors[lone_stone_loc as usize] == C_WHITE);
    testAssert!(tboard.get_num_liberties(lone_stone_loc) == 1);

    let mut area_v2 = [0 as Color; Board::MAX_ARR_SIZE as usize];
    let mut area_v3 = [0 as Color; Board::MAX_ARR_SIZE as usize];
    let mut count: i32 = 0;
    tboard.calculate_independent_life_area(&mut area_v2, &mut count, true, false, false, false);
    tboard.calculate_independent_life_area(&mut area_v3, &mut count, true, false, true, false);
    //The shared liberty is nobody's territory either way, and nothing else differs.
    testAssert!(area_v2[shared_lib_loc as usize] == C_EMPTY);
    testAssert!(area_v3[shared_lib_loc as usize] == C_EMPTY);
    for i in 0..Board::MAX_ARR_SIZE as usize {
      testAssert!(area_v2[i] == area_v3[i]);
    }

    let mut hist_v2 = BoardHistory::new_full(&tboard, P_BLACK, &terr_no_tax_rules, 0, &modes(false, false));
    let tb1 = tboard.clone();
    hist_v2.end_and_score_game_now(&tb1);
    let mut hist_v3 = BoardHistory::new_full(&tboard, P_BLACK, &terr_no_tax_rules, 0, &modes(false, true));
    let tb2 = tboard.clone();
    hist_v3.end_and_score_game_now(&tb2);
    testAssert!(hist_v2.final_white_minus_black_score == hist_v3.final_white_minus_black_score);
    cases.ran();
  }
}
