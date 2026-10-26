//! Replays the game/ parts of upstream cpp/tests/testpassalivesuicide.cpp @ v1.18.2
//! (Tests::runPassAliveSuicideModeTests): BoardHistoryModes::alwaysComputePassAliveUnderSuicideRules in
//! BoardHistory::suicideLegalForPassAlive, area and territory scoring, getSituationRulesAndKoHash, setModes,
//! and the copy / assignment / clear() / copyToInitial preservation of the modes. Upstream prints only a
//! completion line; every check is a testAssert, so the oracle is the asserts themselves.
//!
//! Not ported here (TODO(v1.18.2 sync), they need code other areas re-sync): "MiscNNInputParams override
//! behavior for the nn cache hash" (MiscNNInputParams::passAliveSuicideRulesOverride /
//! getSuicideLegalForPassAlive, NNInputs::getHash), "Search maintains the invariant ..." (Search::
//! resolveAlwaysComputePassAliveUnderSuicideRules, SearchParams::alwaysComputePassAliveUnderSuicideRules,
//! NNEvaluator::modelPreferPassAliveUnderSuicideRules), "Books record the BoardHistoryModes ..." (book/, not
//! ported), "The nn cache cannot mix results across featurization modes" (NNEvaluator::evaluate with overrides).

mod common;

use common::Cases;
use common::init;
use katago_core::game::board::*;
use katago_core::game::boardhistory::*;
use katago_core::game::rules::Rules;
use katago_core::testAssert;

fn modes(a: bool, b: bool) -> BoardHistoryModes {
  BoardHistoryModes::new_full(a, b)
}

#[test]
fn pass_alive_suicide_mode_game_tests() {
  init();
  let mut cases = Cases::new("testpassalivesuicide::pass_alive_suicide_mode_game_tests");
  //A position where pass-alive computation differs between suicide-legal and suicide-illegal modes.
  //Under suicide-illegal computation, the three white stones are pass-dead inside black
  //pass-alive territory (an ecosystem of mutually-supporting black chains). Under suicide-legal
  //computation, none of the black chains are pass-alive at all.
  let board = Board::parse_board(4, 5, r"
...x
xx..
oxxx
oox.
.x.x
", '\n').unwrap();

  let rules = Rules::parse_rules("chinese").unwrap();
  testAssert!(!rules.multi_stone_suicide_legal);

  //Sanity check that this position really does discriminate the two computations.
  {
    let mut area_no_suicide = [0 as Color; Board::MAX_ARR_SIZE as usize];
    let mut area_suicide = [0 as Color; Board::MAX_ARR_SIZE as usize];
    board.calculate_area(&mut area_no_suicide, false, false, false, false);
    board.calculate_area(&mut area_suicide, false, false, false, true);
    let white_loc: Loc = Location::get_loc(0, 2, board.x_size);
    testAssert!(board.colors[white_loc as usize] == C_WHITE);
    testAssert!(area_no_suicide[white_loc as usize] == C_BLACK);
    testAssert!(area_suicide[white_loc as usize] == C_EMPTY);
    cases.ran();
  }

  //Basic flag behavior and effect on scoring
  {
    let hist = BoardHistory::new_full(&board, P_BLACK, &rules, 0, &modes(false, false));
    testAssert!(!hist.modes.always_compute_pass_alive_under_suicide_rules);
    testAssert!(!hist.suicide_legal_for_pass_alive());

    let mut hist_flagged = BoardHistory::new_full(&board, P_BLACK, &rules, 0, &modes(false, false));
    hist_flagged.set_modes(&modes(true, false));
    testAssert!(hist_flagged.modes.always_compute_pass_alive_under_suicide_rules);
    testAssert!(hist_flagged.suicide_legal_for_pass_alive());

    //The flag changes the situation-and-rules hash exactly when the rules don't already have suicide legal
    let hash_off = BoardHistory::get_situation_rules_and_ko_hash(&board, &hist, P_BLACK, 0.5);
    let hash_on = BoardHistory::get_situation_rules_and_ko_hash(&board, &hist_flagged, P_BLACK, 0.5);
    testAssert!(hash_off != hash_on);

    //And changes area scoring on this position
    let mut hist_score = BoardHistory::new_full(&board, P_BLACK, &rules, 0, &modes(false, false));
    let board_copy = board.clone();
    hist_score.end_and_score_game_now(&board_copy);
    let mut hist_score_flagged = BoardHistory::new_full(&board, P_BLACK, &rules, 0, &modes(false, false));
    hist_score_flagged.set_modes(&modes(true, false));
    let board_copy2 = board.clone();
    hist_score_flagged.end_and_score_game_now(&board_copy2);
    testAssert!(hist_score.is_scored && hist_score_flagged.is_scored);
    testAssert!(hist_score.final_white_minus_black_score != hist_score_flagged.final_white_minus_black_score);

    //And changes territory scoring too (the countTerritoryAreaScoreWhiteMinusBlack path)
    let jap_rules = Rules::parse_rules("japanese").unwrap();
    testAssert!(!jap_rules.multi_stone_suicide_legal);
    let mut hist_terr_score = BoardHistory::new_full(&board, P_BLACK, &jap_rules, 0, &modes(false, false));
    let board_copy3 = board.clone();
    hist_terr_score.end_and_score_game_now(&board_copy3);
    let mut hist_terr_score_flagged = BoardHistory::new_full(&board, P_BLACK, &jap_rules, 0, &modes(false, false));
    hist_terr_score_flagged.set_modes(&modes(true, false));
    let board_copy4 = board.clone();
    hist_terr_score_flagged.end_and_score_game_now(&board_copy4);
    testAssert!(hist_terr_score.is_scored && hist_terr_score_flagged.is_scored);
    testAssert!(hist_terr_score.final_white_minus_black_score != hist_terr_score_flagged.final_white_minus_black_score);

    //Copying and clear() preserve the flag
    let copied = hist_flagged.clone();
    testAssert!(copied.modes.always_compute_pass_alive_under_suicide_rules);
    let mut assigned = BoardHistory::new();
    assigned.clone_from(&hist_flagged);
    testAssert!(assigned.modes.always_compute_pass_alive_under_suicide_rules);
    let mut cleared = hist_flagged.clone();
    cleared.clear(&board, P_BLACK, &rules, 0);
    testAssert!(cleared.modes.always_compute_pass_alive_under_suicide_rules);
    testAssert!(hist_flagged.copy_to_initial().modes.always_compute_pass_alive_under_suicide_rules);
    cases.ran();
  }

  //When the rules already have suicide legal, the flag is a no-op for hashing and scoring
  {
    let tromp = Rules::parse_rules("tromp-taylor").unwrap();
    testAssert!(tromp.multi_stone_suicide_legal);
    let hist = BoardHistory::new_full(&board, P_BLACK, &tromp, 0, &modes(false, false));
    let mut hist_flagged = BoardHistory::new_full(&board, P_BLACK, &tromp, 0, &modes(false, false));
    hist_flagged.set_modes(&modes(true, false));
    testAssert!(hist.suicide_legal_for_pass_alive() && hist_flagged.suicide_legal_for_pass_alive());
    let hash_off = BoardHistory::get_situation_rules_and_ko_hash(&board, &hist, P_BLACK, 0.5);
    let hash_on = BoardHistory::get_situation_rules_and_ko_hash(&board, &hist_flagged, P_BLACK, 0.5);
    testAssert!(hash_off == hash_on);
    cases.ran();
  }

  // TODO(v1.18.2 sync): "MiscNNInputParams override behavior for the nn cache hash",
  // "Search maintains the invariant ...", "Books record the BoardHistoryModes ...",
  // "The nn cache cannot mix results across featurization modes" (see the file header).
}
