//! Transcription (a test, not a port) of upstream cpp/tests/testsearchv9.cpp `Tests::runSearchTestsV9` (and its
//! `runV9Positions`) @ v1.16.5, with the TestSearchCommon helpers as test-local copies (common_search/mod.rs).
//!
//! It replays the same calls in the same order (same params, seeds and printing) through the ported AsyncBot /
//! Search / NNEvaluator with the REAL network g170-b6c96 on the burn flex backend, and compares the captured
//! stdout with our Eigen rerun of upstream (fixtures/upstream-v1.16.5/unit/testsearchv9/
//! runsearchtestsv9-g170-bin.expected0.txt = stdout of `katago runsearchtestsv9 g170-b6c96.bin.gz true true false`,
//! see fixtures/upstream-v1.16.5/unit/testsearchcommon/regen.sh). The board diagrams are the extracted
//! `R"%%(...)%%"` blocks `runv9positions.input<k>.txt` (k = order of appearance in the C++).
//!
//! Comparison: section by section, with the numeric tolerance of common_search (integers exact, decimals within
//! one unit of the last printed digit); see `ALLOWLIST` for the sections that are known to diverge and why.
//! `inputsNHWC=true useNHWC=true` of the Eigen run are `false` here (burn is NCHW only; outputs do not depend on
//! them, inventory §1.5).
//!
//! Heavy (11680 NN rows with b6c96 plus the searches): ~4 min in release (`cargo test --release -p katago-nn-burn
//! --test testsearchv9 -- --nocapture`), ~8 min in debug (burn's kernels are optimized in the dev profile).
//! Set KATAGO_SEARCH_TEST_DUMP=<dir> to write the captured stdout there.
#![cfg(all(feature = "flex", feature = "net-b6c96"))]
#![allow(clippy::field_reassign_with_default, clippy::needless_range_loop, clippy::too_many_arguments, clippy::needless_bool)]

#[macro_use]
mod common_search;

use std::rc::Rc;
use std::time::Instant;

use common_search::*;
use katago_core::game::board::*;
use katago_core::game::boardhistory::BoardHistory;
use katago_core::game::rules::Rules;
use katago_core::neuralnet::nneval::NNEvaluator;
use katago_core::neuralnet::nneval::NNResultBuf;
use katago_core::neuralnet::nninputs::MiscNNInputParams;
use katago_core::neuralnet::nninputs::NNInputs;
use katago_core::neuralnet::nninputs::NNOutput;
use katago_core::neuralnet::nninputs::NNPos;
use katago_core::port::fmt::cpp_ostream_f64;
use katago_core::program::playutils::PlayUtils;
use katago_core::search::asyncbot::AsyncBot;
use katago_core::search::searchparams::SearchParams;

const STEM: &str = "testsearchv9";
const SLUG: &str = "runv9positions";

fn board_input(k: usize, x_size: i32, y_size: i32) -> Board {
  parse_board_input(STEM, SLUG, k, x_size, y_size)
}

fn new_bot(params: &SearchParams, nn_eval: &Rc<NNEvaluator>, seed: &str) -> Box<AsyncBot> {
  Box::new(AsyncBot::new(params, nn_eval.clone(), seed))
}

/// `nnEval->evaluate(board,hist,nextPla,nnInputParams,buf,skipCache=true,includeOwnerMap); buf.result->debugPrint(...)`
fn evaluate_and_print(ctx: &mut Ctx, nn_eval: &Rc<NNEvaluator>, board: &Board, hist: &BoardHistory, next_pla: Player, symmetry: i32, include_owner_map: bool) {
  let mut nn_input_params = MiscNNInputParams::default();
  nn_input_params.symmetry = symmetry;
  let buf: Rc<NNResultBuf> = Rc::new(NNResultBuf::new());
  let skip_cache = true;
  ctx.run("evaluate", nn_eval.evaluate(board, hist, next_pla, &nn_input_params, &buf, skip_cache, include_owner_map)).unwrap();
  coutln!("Symmetry {}", symmetry);
  buf.result.borrow().as_ref().expect("evaluated").debug_print(&mut Cout, board);
}

/// `PlayUtils::getFullSymmetryNNOutput(...)` then `nnOutput->debugPrint(cout, board)`
fn full_symmetry_and_print(ctx: &mut Ctx, nn_eval: &Rc<NNEvaluator>, board: &Board, hist: &BoardHistory, next_pla: Player, include_owner_map: bool) {
  let nn_output: Rc<NNOutput> =
    ctx.run("getFullSymmetryNNOutput", PlayUtils::get_full_symmetry_nn_output(board, hist, next_pla, include_owner_map, None, nn_eval)).unwrap();
  coutln!("Symmetry all");
  nn_output.debug_print(&mut Cout, board);
}

/// upstream: the `printLadderFeaturesV7` lambda of runV9Positions (testsearchv9.cpp)
fn print_ladder_features_v7(board: &Board, hist: &BoardHistory, conservative_pass: bool) {
  let nn_x_len = 7;
  let nn_y_len = 7;
  let inputs_use_nhwc = false;
  let mut row_bin: Vec<f32> = vec![0.0; (NNInputs::NUM_FEATURES_SPATIAL_V7 * nn_x_len * nn_y_len) as usize];
  let mut row_global: Vec<f32> = vec![0.0; NNInputs::NUM_FEATURES_GLOBAL_V7 as usize];

  let mut nn_input_params = MiscNNInputParams::default();
  nn_input_params.draw_equivalent_wins_for_white = 0.5;
  nn_input_params.conservative_pass_and_is_root = conservative_pass;
  NNInputs::fill_row_v7(board, hist, hist.presumed_next_move_pla, &nn_input_params, nn_x_len, nn_y_len, inputs_use_nhwc, &mut row_bin, &mut row_global);

  for f in [14, 15, 16] {
    coutln!("Ladder feature {}", f);
    for y in 0..nn_y_len {
      for x in 0..nn_x_len {
        cout!("{} ", cpp_ostream_f64(row_bin[(f * nn_x_len * nn_y_len + y * nn_x_len + x) as usize] as f64));
      }
      coutln!();
    }
  }
}

/// upstream: runV9Positions (testsearchv9.cpp) static
fn run_v9_positions(ctx: &mut Ctx, nn_eval: &Rc<NNEvaluator>) {
  {
    let mut params = SearchParams::for_tests_v2();
    params.max_visits = 100;

    let mut bot = new_bot(&params, nn_eval, &get_search_rand_seed());
    let rules = Rules::parse_rules("Japanese").unwrap();
    let mut opts = TestSearchOptions::new();
    opts.print_play_selection_values = true;
    opts.print_sharp_score_and_error = true;

    {
      coutln!("Flying dagger fight with V2 params and much variety ==========================================================================");
      coutln!();

      let sgf_str = "(;GM[1]FF[4]CA[UTF-8]AP[CGoban:3]ST[2]RU[Japanese]SZ[19]KM[6.50]PW[White]PB[Black];B[pd];W[dd];B[pp];W[dp];B[cq];W[dq];B[cp];W[cn];B[bn];W[bm];B[co];W[dn];B[do];W[eo];B[ep];W[er];B[fp];W[fo];B[go];W[gn];B[gq];W[cr];B[bo];W[cl];B[br];W[gr];B[hr];W[bs];B[bq];W[fr];B[hn];W[ho];B[gp];W[ir];B[hs];W[ap];B[fn];W[en];B[gm];W[nc];B[dl];W[dk];B[em];W[el];B[al];W[bk];B[am];W[ao])";

      for i in 15..47 {
        params.use_non_buggy_lcb = i % 3 != 0;
        params.root_noise_enabled = i % 5 == 0;
        params.lcb_stdevs = if i % 7 == 0 { 2.5 } else { 5.0 };
        params.min_visit_prop_for_lcb = if i % 7 == 0 { 0.03 } else { 0.15 };
        params.use_lcb_for_selection = if i % 7 == 1 { false } else { true };
        params.playout_doubling_advantage = if i % 11 == 0 { 0.75 } else { 0.0 };
        bot.set_params(&params);
        run_bot_on_sgf(ctx, &mut bot, sgf_str, &rules, i, 6.5, &opts);
      }
      coutln!();
      coutln!();
    }

    {
      coutln!("3-r pincer fight with V2 params and much variety ==========================================================================");
      coutln!();

      let sgf_str = "(;GM[1]FF[4]CA[UTF-8]AP[CGoban:3]ST[2]RU[Japanese]SZ[19]KM[6.50]PW[White]PB[Black];B[pd];W[dd];B[pp];W[cp];B[eq];W[hq];B[do];W[dp];B[ep];W[eo];B[dn];W[en];B[bo];W[bp];B[dm];W[em];B[dl];W[dq];B[gp];W[hp];B[er];W[go];B[cr];W[dr];B[ds];W[br];B[bs];W[aq];B[gr];W[hr])";

      for i in 11..30 {
        params.use_non_buggy_lcb = i % 3 != 0;
        params.root_noise_enabled = i % 5 == 0;
        params.lcb_stdevs = if i % 7 < 4 { 2.5 } else { 5.0 };
        params.min_visit_prop_for_lcb = if i % 7 < 4 { 0.03 } else { 0.15 };
        params.use_lcb_for_selection = if i % 7 > 5 { false } else { true };
        params.playout_doubling_advantage = if i % 11 < 3 { 0.75 } else { 0.0 };
        bot.set_params(&params);
        run_bot_on_sgf(ctx, &mut bot, sgf_str, &rules, i, 6.5, &opts);
      }
      coutln!();
      coutln!();
    }

    drop(bot);
  }

  {
    coutln!("Pruned root values test ==========================================================================");
    coutln!();
    let board = board_input(0, 13, 13);
    let next_pla = P_WHITE;
    let rules = Rules::parse_rules("Chinese").unwrap();
    let hist = Box::new(BoardHistory::new_full(&board, next_pla, &rules, 0));

    {
      let mut params = SearchParams::for_tests_v2();
      params.max_visits = 600;
      params.root_fpu_reduction_max = 0.0;
      let mut bot = new_bot(&params, nn_eval, &get_search_rand_seed());
      let mut opts = TestSearchOptions::new();
      opts.print_play_selection_values = true;
      opts.print_root_values = true;
      opts.print_pruned_root_values = true;
      run_bot_on_position(ctx, &mut bot, &board, next_pla, &hist, &opts);
      drop(bot);
    }
    {
      let mut params = SearchParams::for_tests_v2();
      params.max_visits = 600;
      params.root_noise_enabled = true;
      params.root_fpu_reduction_max = 0.0;
      let mut bot = new_bot(&params, nn_eval, &get_search_rand_seed());
      let mut opts = TestSearchOptions::new();
      opts.print_play_selection_values = true;
      opts.print_root_values = true;
      opts.print_pruned_root_values = true;
      run_bot_on_position(ctx, &mut bot, &board, next_pla, &hist, &opts);
      drop(bot);
    }
  }

  {
    coutln!("Conservative passing with pass as hint loc ==========================================================================");
    coutln!();

    let board = board_input(1, 9, 9);
    let next_pla = P_WHITE;

    for (rules_str, conservative_pass, label) in [
      ("Chinese", false, "Conservative pass false"),
      ("Chinese", true, "Conservative pass true"),
      ("Japanese", false, "Conservative pass false jp rules"),
      ("Japanese", true, "Conservative pass true jp rules"),
    ] {
      let rules = Rules::parse_rules(rules_str).unwrap();
      let hist = Box::new(BoardHistory::new_full(&board, next_pla, &rules, 0));

      let mut params = SearchParams::for_tests_v2();
      params.max_visits = 75;
      params.root_fpu_reduction_max = 0.0;
      params.conservative_pass = conservative_pass;
      let mut bot = new_bot(&params, nn_eval, &get_search_rand_seed());
      let mut opts = TestSearchOptions::new();
      opts.num_moves_in_a_row = 25;
      opts.root_hint_loc = Board::PASS_LOC;
      coutln!("{}", label);
      run_bot_on_position(ctx, &mut bot, &board, next_pla, &hist, &opts);
      drop(bot);
    }
  }

  {
    coutln!("Phase change truncating ladder history in spight rules ==========================================================================");
    coutln!();

    //One `board` for both blocks, as upstream: the first block's moves mutate it (and the second block replays the
    //same moves on it).
    let mut board = board_input(2, 7, 7);
    let next_pla = P_BLACK;

    for conservative_pass in [false, true] {
      let mut rules = Rules::parse_rules("Japanese").unwrap();
      rules.ko_rule = Rules::KO_SPIGHT;
      let mut hist = Box::new(BoardHistory::new_full(&board, next_pla, &rules, 0));
      play(&mut hist, &mut board, "G3", P_BLACK);
      play(&mut hist, &mut board, "G1", P_WHITE);
      play(&mut hist, &mut board, "G2", P_BLACK);
      play(&mut hist, &mut board, "pass", P_WHITE);
      play(&mut hist, &mut board, "G3", P_BLACK);
      play(&mut hist, &mut board, "G1", P_WHITE);
      play(&mut hist, &mut board, "G2", P_BLACK);
      play(&mut hist, &mut board, "pass", P_WHITE);

      let mut params = SearchParams::for_tests_v2();
      params.max_visits = 100;
      params.root_fpu_reduction_max = 0.0;
      params.conservative_pass = conservative_pass;
      let mut bot = new_bot(&params, nn_eval, &get_search_rand_seed());
      let opts = TestSearchOptions::new();
      //Both blocks print "Conservative pass false" upstream.
      coutln!("Conservative pass false");
      run_bot_on_position(ctx, &mut bot, &board, next_pla, &hist, &opts);
      print_ladder_features_v7(&board, &hist, conservative_pass);
      drop(bot);
    }
  }

  {
    let next_pla = P_WHITE;
    let rules = Rules::get_tromp_taylorish();
    let board = board_input(3, 9, 9);
    let hist = Box::new(BoardHistory::new_full(&board, next_pla, &rules, 0));

    let mut avoid_move_until_by_loc: Vec<i32> = vec![0; Board::MAX_ARR_SIZE as usize];
    avoid_move_until_by_loc[loc_of("D5", &board) as usize] = 1;
    let mut params = SearchParams::for_tests_v2();
    params.max_visits = 200;
    let mut bot_a = new_bot(&params, nn_eval, "avoidMoveUntilRescaleRoot test");
    let mut bot_b = new_bot(&params, nn_eval, "avoidMoveUntilRescaleRoot test");
    let mut bot_c = new_bot(&params, nn_eval, "avoidMoveUntilRescaleRoot test");
    let mut bot_d = new_bot(&params, nn_eval, "avoidMoveUntilRescaleRoot test");

    bot_a.set_position(next_pla, &board, &hist).unwrap();
    bot_b.set_position(next_pla, &board, &hist).unwrap();
    bot_c.set_position(next_pla, &board, &hist).unwrap();
    bot_d.set_position(next_pla, &board, &hist).unwrap();

    bot_b.set_avoid_move_until_rescale_root(true);
    bot_c.set_avoid_move_until_by_loc(&avoid_move_until_by_loc, &avoid_move_until_by_loc);
    bot_d.set_avoid_move_until_rescale_root(true);
    bot_d.set_avoid_move_until_by_loc(&avoid_move_until_by_loc, &avoid_move_until_by_loc);

    let mut opts = TestSearchOptions::new();
    opts.ignore_position = true; // Avoid clearing the avoids we set.
    coutln!("Baseline");
    run_bot_on_position(ctx, &mut bot_a, &board, next_pla, &hist, &opts);
    coutln!("setAvoidMoveUntilRescaleRoot (should be same)");
    run_bot_on_position(ctx, &mut bot_b, &board, next_pla, &hist, &opts);
    coutln!("Banning extend");
    run_bot_on_position(ctx, &mut bot_c, &board, next_pla, &hist, &opts);
    coutln!("Banning extend and setAvoidMoveUntilRescaleRoot");
    run_bot_on_position(ctx, &mut bot_d, &board, next_pla, &hist, &opts);
    coutln!();
    coutln!();

    drop(bot_a);
    drop(bot_b);
    drop(bot_c);
    drop(bot_d);
  }

  {
    coutln!("Passing details ==========================================================================");
    coutln!();

    let board = board_input(4, 8, 8);
    let next_pla = P_BLACK;

    for (label, friendly_pass_ok, enable_passing_hacks) in [
      ("Area scoring no friendly pass ok", false, false),
      ("Area scoring yes friendly pass ok", true, false),
      ("Area scoring no friendly pass ok but pass hacks", false, true),
    ] {
      coutln!("{}", label);
      let mut rules = Rules::parse_rules("Chinese").unwrap();
      rules.friendly_pass_ok = friendly_pass_ok;
      let hist = Box::new(BoardHistory::new_full(&board, next_pla, &rules, 0));

      let mut params = SearchParams::for_tests_v2();
      params.max_visits = 50;
      params.root_fpu_reduction_max = 0.0;
      if enable_passing_hacks {
        params.enable_passing_hacks = true;
      }
      let mut bot = new_bot(&params, nn_eval, &get_search_rand_seed());
      let mut opts = TestSearchOptions::new();
      opts.root_hint_loc = Board::PASS_LOC;
      opts.print_more = true;
      run_bot_on_position(ctx, &mut bot, &board, next_pla, &hist, &opts);
      drop(bot);
    }
  }

  {
    coutln!("Single symmetry and full symmetry raw nets");
    let mut board = board_input(5, 8, 8);
    let mut next_pla = P_WHITE;
    let rules = Rules::parse_rules("Chinese").unwrap();
    let mut hist = Box::new(BoardHistory::new_full(&board, next_pla, &rules, 0));
    play(&mut hist, &mut board, "E5", P_WHITE);
    next_pla = get_opp(next_pla);

    let include_owner_map = true;
    coutln!("{}", board);
    for sym in [0, 1, 2, 4] {
      evaluate_and_print(ctx, nn_eval, &board, &hist, next_pla, sym, include_owner_map);
    }
    full_symmetry_and_print(ctx, nn_eval, &board, &hist, next_pla, include_owner_map);
  }

  {
    coutln!("Single symmetry and full symmetry raw nets");
    let mut board = board_input(6, 8, 5);
    let mut next_pla = P_WHITE;
    let rules = Rules::parse_rules("Japanese").unwrap();
    let mut hist = Box::new(BoardHistory::new_full(&board, next_pla, &rules, 0));
    play(&mut hist, &mut board, "E4", P_WHITE);
    next_pla = get_opp(next_pla);

    let include_owner_map = true;
    coutln!("{}", board);
    for sym in [0, 1, 2, 4] {
      evaluate_and_print(ctx, nn_eval, &board, &hist, next_pla, sym, include_owner_map);
    }
    full_symmetry_and_print(ctx, nn_eval, &board, &hist, next_pla, include_owner_map);
  }

  {
    coutln!("Single symmetry and full symmetry raw nets");
    let mut board = Board::new_x_y(19, 19).unwrap();
    let mut next_pla = P_BLACK;
    let mut rules = Rules::parse_rules("Japanese").unwrap();
    rules.komi = -4.0;
    let mut hist = Box::new(BoardHistory::new_full(&board, next_pla, &rules, 0));
    play(&mut hist, &mut board, "A1", P_BLACK);
    play(&mut hist, &mut board, "B2", P_WHITE);
    play(&mut hist, &mut board, "C3", P_BLACK);
    play(&mut hist, &mut board, "D4", P_WHITE);
    play(&mut hist, &mut board, "E5", P_BLACK);
    next_pla = get_opp(next_pla);

    let include_owner_map = true;
    coutln!("{}", board);
    full_symmetry_and_print(ctx, nn_eval, &board, &hist, next_pla, include_owner_map);
  }

  {
    coutln!("Very low visit search discretization ==========================================================================");
    coutln!();

    let mut board = board_input(7, 9, 9);
    {
      let mut rules = Rules::parse_rules("Chinese").unwrap();
      rules.friendly_pass_ok = false;
      rules.komi = 5.5;
      let mut hist = Box::new(BoardHistory::new_full(&board, P_BLACK, &rules, 0));
      play(&mut hist, &mut board, "E5", P_BLACK);
      let next_pla = P_WHITE;

      let mut params = SearchParams::for_tests_v2();
      params.max_visits = 20;
      params.root_symmetry_pruning = true;
      params.root_fpu_reduction_max = 0.0;
      let mut bot = new_bot(&params, nn_eval, &get_search_rand_seed());
      let opts = TestSearchOptions::new();
      run_bot_on_position(ctx, &mut bot, &board, next_pla, &hist, &opts);
      drop(bot);
    }
  }

  for (title, k, start_pla) in [
    ("More passing hack ==========================================================================", 8, P_WHITE),
    ("More passing hack 2 ==========================================================================", 9, P_BLACK),
  ] {
    coutln!("{}", title);
    coutln!();

    let mut board = board_input(k, 9, 9);
    {
      let mut rules = Rules::parse_rules("Chinese").unwrap();
      rules.friendly_pass_ok = false;
      rules.komi = 7.5;
      let mut hist = Box::new(BoardHistory::new_full(&board, start_pla, &rules, 0));

      play(&mut hist, &mut board, "pass", start_pla);
      let next_pla = get_opp(start_pla);

      let mut params = SearchParams::for_tests_v2();

      let mut opts = TestSearchOptions::new();
      opts.print_more_more_more = true;
      opts.print_ending_score_value_bonus = true;
      opts.print_play_selection_values = true;

      let mut bot = new_bot(&params, nn_eval, &get_search_rand_seed());
      for more_passing_hacks in [false, true] {
        params.enable_more_passing_hacks = more_passing_hacks;
        coutln!("params.enableMorePassingHacks = {}", if more_passing_hacks { "true" } else { "false" });
        for max_visits in [5, 20, 100] {
          params.max_visits = max_visits;
          bot.set_params(&params);
          run_bot_on_position(ctx, &mut bot, &board, next_pla, &hist, &opts);
        }
      }
      drop(bot);
    }
  }
}

/// upstream: Tests::runSearchTestsV9 (testsearchv9.cpp)
fn run_search_tests_v9() {
  //TestCommon::overrideForBackends(inputsNHWC, useNHWC); (no-op for the burn backend)
  coutln!("Running v9 search tests");
  //NeuralNet::globalInitialize(); (no-op for cpu)
  reset_search_rand_seed();
  let mut ctx = Ctx::new();

  let symmetry = 4;
  let nn_eval = start_nn_eval(&mut ctx, MODEL_FILE, "", NNPos::MAX_BOARD_LEN, NNPos::MAX_BOARD_LEN, symmetry, false, false, false, false, false);
  run_v9_positions(&mut ctx, &nn_eval);
  delete_nn_eval(&mut ctx, nn_eval);

  //NeuralNet::globalCleanup();
  coutln!("Done");
}

//------------------------------------------------------------------------------------------------
// Driver

/// Sub-block titles the test prints without '=' (finer sections, so that one diverging search does not hide the
/// others).
const EXTRA_HEADERS: &[&str] = &[
  "Conservative pass false",
  "Conservative pass true",
  "Conservative pass false jp rules",
  "Conservative pass true jp rules",
  "Baseline",
  "setAvoidMoveUntilRescaleRoot (should be same)",
  "Banning extend",
  "Banning extend and setAvoidMoveUntilRescaleRoot",
  "Area scoring no friendly pass ok",
  "Area scoring yes friendly pass ok",
  "Area scoring no friendly pass ok but pass hacks",
  "Single symmetry and full symmetry raw nets",
  "params.enableMorePassingHacks = false",
  "params.enableMorePassingHacks = true",
];

/// Sections that are known to diverge from the Eigen rerun, with the reason (first diverging line in the report).
const ALLOWLIST: &[Allowed] = &[];

fn policy_of(_label: &str) -> Policy {
  Policy::Full
}

#[test]
fn search_tests_v9() {
  let t0 = Instant::now();
  let expected_text = fixture(STEM, "runsearchtestsv9-g170-bin.expected0.txt");
  let extra: &[&str] = EXTRA_HEADERS;
  let expected = split_sections(&expected_text, extra);
  assert!(expected.len() > 10, "expected sections");

  let (actual_text, panic_msg) = run_capturing("testsearchv9", run_search_tests_v9);
  let secs = t0.elapsed().as_secs_f64();
  if let Ok(dir) = std::env::var("KATAGO_SEARCH_TEST_DUMP") {
    let p = std::path::Path::new(&dir).join("testsearchv9.actual.txt");
    std::fs::write(&p, &actual_text).unwrap();
    eprintln!("wrote {}", p.display());
  }
  let actual = split_sections(&actual_text, extra);
  let (report, num_bad, num_compared) = compare_all("testsearchv9", &expected, &actual, &policy_of, ALLOWLIST, panic_msg.as_deref());
  eprintln!("{}", report);
  eprintln!("testsearchv9: {} sections compared, run took {:.1} s", num_compared, secs);
  assert!(num_compared >= 1, "no section compared");
  assert!(panic_msg.is_none(), "the transcription panicked: {}", panic_msg.unwrap_or_default());
  assert_eq!(num_bad, 0, "{} sections diverged from the Eigen rerun and are not allowlisted", num_bad);
}
