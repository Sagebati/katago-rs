//! Transcription (a test, not a port) of upstream cpp/tests/testsearchv8.cpp `Tests::runSearchTestsV8` and its
//! sub-functions (runV8TestsSize9, runV8TestsRandomSym, runV8Tests, runMoreV8Tests, runMoreV8Tests2,
//! runMoreV8TestsRandomizedNNEvals, runV8SearchMultithreadTest) @ v1.16.5, with the TestSearchCommon helpers as
//! test-local copies (common_search/mod.rs).
//!
//! It replays the same calls in the same order (same params, seeds and printing) through the ported AsyncBot /
//! Search / NNEvaluator with the REAL network g170-b6c96 on the burn flex backend, and compares the captured stdout
//! with our Eigen rerun of upstream (fixtures/upstream-v1.16.5/unit/testsearchv8/runsearchtestsv8-g170-bin.expected0.txt
//! = stdout of `katago runsearchtestsv8 g170-b6c96.bin.gz true true false`, see
//! fixtures/upstream-v1.16.5/unit/testsearchcommon/regen.sh). Board diagrams: the extracted `R"%%(...)%%"` blocks
//! `<function>.input<k>.txt`.
//!
//! runSearchTestsV8 creates its evaluators in seven groups (one per sub-function); each group is one #[test] here
//! (`v8_group_<k>`), comparing the slice of the expected stdout from that group's `startNNEval` log lines up to the
//! next group's. Each group test starts from a fresh process state (upstream runs them in one process; the only
//! process-wide state they share is the Logger, and getSearchRandSeed is not used by V8).
//!
//! Comparison: section by section, with the numeric tolerance of common_search (integers exact, decimals within one
//! unit of the last printed digit); `policy_of` lists the summary-only
//! sections, `ALLOWLIST` the sections known to diverge and why.
//! `inputsNHWC=true useNHWC=true` of the Eigen run are `false` here (burn is NCHW only; outputs do not depend on them).
//! 9x9 evaluators run through common_search's `PaddedNet` (the burn graph is fixed at 19x19).
//!
//! Heavy: run in release, `cargo test --release -p katago-nn-burn --test testsearchv8 -- --nocapture` (~15 min, group 3
//! dominates). Groups 3 and 6 are ignored in debug builds (TODO(port) at the tests); the other groups run in debug.
//! KATAGO_SEARCH_TEST_DUMP=<dir> writes each group's captured stdout there.
#![cfg(all(feature = "flex", feature = "net-b6c96"))]
#![allow(clippy::field_reassign_with_default, clippy::needless_range_loop, clippy::too_many_arguments, clippy::type_complexity, clippy::bool_comparison)]

#[macro_use]
mod common_search;

use std::rc::Rc;
use std::time::Instant;

use common_search::*;
use katago_core::core::commontypes::enabled_t;
use katago_core::game::board::*;
use katago_core::game::boardhistory::BoardHistory;
use katago_core::game::rules::Rules;
use katago_core::neuralnet::nneval::NNEvaluator;
use katago_core::neuralnet::nninputs::NNPos;
use katago_core::port::fmt::cpp_ostream_f64;
use katago_core::program::playutils::PlayUtils;
use katago_core::search::asyncbot::AsyncBot;
use katago_core::search::search::Search;
use katago_core::search::searchparams::SearchParams;
use katago_core::search::searchprint::PrintTreeOptions;
use katago_core::search::timecontrols::TimeControls;
use katago_core::testAssert;

const STEM: &str = "testsearchv8";

fn board_input(slug: &str, k: usize, x_size: i32, y_size: i32) -> Board {
  parse_board_input(STEM, slug, k, x_size, y_size)
}

fn new_bot(params: &SearchParams, nn_eval: &Rc<NNEvaluator>, seed: &str) -> Box<AsyncBot> {
  Box::new(AsyncBot::new(params, nn_eval.clone(), seed))
}

fn new_search(params: &SearchParams, nn_eval: &Rc<NNEvaluator>, seed: &str) -> Box<Search> {
  Box::new(Search::new(params, nn_eval.clone(), seed))
}

/// `std::unique_ptr<CompactSgf> sgf = CompactSgf::parse(sgfStr); ... sgf->setupBoardAndHistAssumeLegal(initialRules, board, nextPla, hist, turnIdx);`
fn setup_from_sgf(sgf: &CompactSgf, initial_rules: &Rules, turn_idx: i64) -> (Board, Player, Box<BoardHistory>) {
  let mut board = Board::new();
  let mut next_pla: Player = P_BLACK;
  let mut hist = Box::new(BoardHistory::new());
  sgf.setup_board_and_hist_assume_legal(initial_rules, &mut board, &mut next_pla, &mut hist, turn_idx);
  (board, next_pla, hist)
}

fn clear_cache_and_stats(nn_eval: &NNEvaluator) {
  nn_eval.clear_cache();
  nn_eval.clear_stats();
}

/// `cout << "===...===" << endl << title << endl << "===...===" << endl;`
fn header_trio(title: &str) {
  coutln!("===================================================================");
  coutln!("{}", title);
  coutln!("===================================================================");
}

//------------------------------------------------------------------------------------------------
// runV8TestsSize9

/// upstream: runV8TestsSize9 (testsearchv8.cpp) static
fn run_v8_tests_size9(ctx: &mut Ctx, nn_eval: &Rc<NNEvaluator>, nn_eval9: &Rc<NNEvaluator>, nn_eval9_exact: &Rc<NNEvaluator>) {
  {
    coutln!("TEST EXACT (NO MASKING) VS MASKED 9x9 ==========================================================================");
    let sgf_str = "(;FF[4]GM[1]SZ[9]HA[0]KM[7]RU[stonescoring];B[ef];W[ed];B[ge])";
    let sgf = CompactSgf::parse(sgf_str);

    let initial_rules = sgf.get_rules_or_fail();
    let (board, next_pla, hist) = setup_from_sgf(&sgf, &initial_rules, 3);

    let mut params = SearchParams::for_tests_v1();
    params.max_visits = 200;
    let mut bot_a = new_bot(&params, nn_eval, "test exact again");
    let mut bot_b = new_bot(&params, nn_eval9, "test exact again");
    let mut bot_c = new_bot(&params, nn_eval9_exact, "test exact again");

    let opts = TestSearchOptions::new();
    coutln!("BASIC");
    run_bot_on_position(ctx, &mut bot_a, &board, next_pla, &hist, &opts);
    coutln!("BASIC9");
    run_bot_on_position(ctx, &mut bot_b, &board, next_pla, &hist, &opts);
    coutln!("EXACT");
    run_bot_on_position(ctx, &mut bot_c, &board, next_pla, &hist, &opts);

    coutln!();
    coutln!();
  }
}

//------------------------------------------------------------------------------------------------
// runV8TestsRandomSym

/// upstream: runV8TestsRandomSym (testsearchv8.cpp) static
fn run_v8_tests_random_sym(ctx: &mut Ctx, nn_eval: &Rc<NNEvaluator>, nn_eval19_exact: &Rc<NNEvaluator>) {
  {
    coutln!("TEST EXACT (NO MASKING) VS MASKED ==========================================================================");

    let sgf_str = "(;GM[1]FF[4]CA[UTF-8]RU[Japanese]SZ[19]KM[6.5];B[dd];W[qd];B[pq];W[dp];B[oc];W[pe];B[fq];W[jp];B[ph];W[cf];B[ck])";
    let sgf = CompactSgf::parse(sgf_str);
    let initial_rules = sgf.get_rules_or_fail();
    let (board, next_pla, hist) = setup_from_sgf(&sgf, &initial_rules, 11);

    let mut params = SearchParams::for_tests_v1();
    params.max_visits = 200;
    let mut bot_a = new_bot(&params, nn_eval, "test exact");
    let mut bot_b = new_bot(&params, nn_eval19_exact, "test exact");

    let opts = TestSearchOptions::new();
    coutln!("BASIC");
    run_bot_on_position(ctx, &mut bot_a, &board, next_pla, &hist, &opts);
    coutln!("EXACT");
    run_bot_on_position(ctx, &mut bot_b, &board, next_pla, &hist, &opts);

    coutln!();
    coutln!();
  }

  {
    coutln!("TEST SYMMETRY AVGING ==========================================================================");
    //Reset random seeds for nnEval
    respawn_server_threads(ctx, nn_eval);
    respawn_server_threads(ctx, nn_eval19_exact);

    let sgf_str = "(;GM[1]FF[4]CA[UTF-8]RU[Japanese]SZ[19]KM[6.5];B[dd];W[qd];B[od];W[pq];B[dq];W[do];B[eo];W[oe])";
    let sgf = CompactSgf::parse(sgf_str);
    let initial_rules = sgf.get_rules_or_fail();
    let (board, next_pla, hist) = setup_from_sgf(&sgf, &initial_rules, 8);

    let mut params = SearchParams::for_tests_v1();
    params.root_num_symmetries_to_sample = 8;
    params.max_visits = 200;
    let mut bot_a = new_bot(&params, nn_eval, "test exact");
    let mut bot_b = new_bot(&params, nn_eval19_exact, "test exact");

    let opts = TestSearchOptions::new();
    coutln!("BASIC");
    run_bot_on_position(ctx, &mut bot_a, &board, next_pla, &hist, &opts);
    coutln!("EXACT");
    run_bot_on_position(ctx, &mut bot_b, &board, next_pla, &hist, &opts);

    coutln!();
    coutln!();

    drop(bot_a);
    drop(bot_b);

    //Reset random seeds for nnEval
    respawn_server_threads(ctx, nn_eval);
    respawn_server_threads(ctx, nn_eval19_exact);
  }

  {
    coutln!("TEST NN TEMPERATURE ==========================================================================");
    let sgf_str = "(;GM[1]FF[4]CA[UTF-8]RU[AGA]SZ[19]KM[7.0];B[dd];W[pd];B[dp];W[pp];B[qc];W[qd];B[pc];W[nc];B[nb])";
    let sgf = CompactSgf::parse(sgf_str);
    let initial_rules = sgf.get_rules_or_fail();
    let (board, next_pla, hist) = setup_from_sgf(&sgf, &initial_rules, 8);

    let mut params_a = SearchParams::for_tests_v1();
    let mut params_b = SearchParams::for_tests_v1();
    let mut params_c = SearchParams::for_tests_v1();
    params_a.max_visits = 200;
    params_b.max_visits = 200;
    params_c.max_visits = 200;
    params_b.nn_policy_temperature = 1.5;
    params_c.nn_policy_temperature = 0.5;
    let mut bot_a = new_bot(&params_a, nn_eval, "test exact");
    let mut bot_b = new_bot(&params_b, nn_eval, "test exact");
    let mut bot_c = new_bot(&params_c, nn_eval, "test exact");
    let mut bot_a2 = new_bot(&params_a, nn_eval19_exact, "test exact");
    let mut bot_b2 = new_bot(&params_b, nn_eval19_exact, "test exact");
    let mut bot_c2 = new_bot(&params_c, nn_eval19_exact, "test exact");

    let mut opts = TestSearchOptions::new();
    opts.print_more = true;
    clear_cache_and_stats(nn_eval);
    opts.no_clear_cache = true;
    coutln!("BASELINE");
    //Reset random seeds for nnEval
    respawn_server_threads(ctx, nn_eval);
    respawn_server_threads(ctx, nn_eval19_exact);
    run_bot_on_position(ctx, &mut bot_a, &board, next_pla, &hist, &opts);
    run_bot_on_position(ctx, &mut bot_a2, &board, next_pla, &hist, &opts);

    coutln!("TEMP 1.5");
    //Reset random seeds for nnEval
    respawn_server_threads(ctx, nn_eval);
    respawn_server_threads(ctx, nn_eval19_exact);
    run_bot_on_position(ctx, &mut bot_b, &board, next_pla, &hist, &opts);
    run_bot_on_position(ctx, &mut bot_b2, &board, next_pla, &hist, &opts);

    coutln!("TEMP 0.5");
    //Reset random seeds for nnEval
    respawn_server_threads(ctx, nn_eval);
    respawn_server_threads(ctx, nn_eval19_exact);
    run_bot_on_position(ctx, &mut bot_c, &board, next_pla, &hist, &opts);
    run_bot_on_position(ctx, &mut bot_c2, &board, next_pla, &hist, &opts);
    coutln!();
    coutln!();
  }
}

//------------------------------------------------------------------------------------------------
// runV8Tests

/// upstream: the `printSearchResults` lambda of runV8Tests (testsearchv8.cpp)
fn print_search_results(search: &Search) {
  coutln!("{}", search.root_board);
  cout!("Root visits: {}\n", search.get_root_visits());
  cout!("Last search playouts: {}\n", search.last_search_num_playouts);
  coutln!("NN rows: {}", search.nn_evaluator.num_rows_processed());
  coutln!("NN batches: {}", search.nn_evaluator.num_batches_processed());
  coutln!("NN avg batch size: {}", cpp_ostream_f64(search.nn_evaluator.average_processed_batch_size()));
  if search.search_params.playout_doubling_advantage != 0.0 {
    coutln!(
      "PlayoutDoublingAdvantage: {}",
      cpp_ostream_f64(if search.get_root_pla() == get_opp(search.get_playout_doubling_advantage_pla()) {
        -search.search_params.playout_doubling_advantage
      } else {
        search.search_params.playout_doubling_advantage
      })
    );
  }
  cout!("PV: ");
  let root = search.get_root_node();
  search.print_pv(&mut Cout, root.as_deref(), 25);
  cout!("\n");
  cout!("Tree:\n");
  let mut options = PrintTreeOptions::new();
  options = options.max_depth(1);
  print_tree(search, &options, P_WHITE);
}

/// `search->runWholeSearchAndGetMove(nextPla[,pondering])`
fn search_and_get_move(ctx: &mut Ctx, search: &mut Search, pla: Player, pondering: bool) -> Loc {
  ctx.run("runWholeSearchAndGetMove", search.run_whole_search_and_get_move_pondering(pla, pondering)).unwrap()
}

/// `search->runWholeSearch(nextPla[,pondering])`
fn search_whole(ctx: &mut Ctx, search: &mut Search, pla: Player, pondering: bool) {
  ctx.run("runWholeSearch", search.run_whole_search_pondering(pla, pondering)).unwrap()
}

/// upstream: runV8Tests (testsearchv8.cpp) static
fn run_v8_tests(ctx: &mut Ctx, nn_eval: &Rc<NNEvaluator>) {
  {
    header_trio("Testing PDA + pondering, p200 v400");

    let board = board_input("runv8tests", 0, 13, 13);

    let start_pla: Player = P_WHITE;
    let rules = Rules::get_tromp_taylorish();
    let hist = Box::new(BoardHistory::new_full(&board, start_pla, &rules, 0));
    let mut base_params = SearchParams::for_tests_v1();
    base_params.max_visits = 400;
    base_params.max_visits_pondering = 600;
    base_params.max_playouts = 200;
    base_params.max_playouts_pondering = 300;

    clear_cache_and_stats(nn_eval);

    {
      let mut params = base_params.clone();
      params.playout_doubling_advantage = 1.5;
      coutln!("Basic search with PDA 1.5, no player");

      let mut search = new_search(&params, nn_eval, "autoSearchRandSeed3");
      let mut next_pla: Player;

      search.set_position(start_pla, &board, &hist).unwrap();
      next_pla = start_pla;

      let mut move_loc = search_and_get_move(ctx, &mut search, next_pla, false);
      print_search_results(&search);

      coutln!("Search next player - should clear tree and flip PDA");
      search.make_move(move_loc, next_pla).unwrap();
      next_pla = get_opp(next_pla);
      move_loc = search_and_get_move(ctx, &mut search, next_pla, false);
      print_search_results(&search);

      coutln!("Search next player - should clear tree and flip PDA");
      search.make_move(move_loc, next_pla).unwrap();
      next_pla = get_opp(next_pla);
      search_whole(ctx, &mut search, next_pla, false);
      print_search_results(&search);

      drop(search);
      clear_cache_and_stats(nn_eval);
    }

    for (force_pla, label) in [(P_BLACK, "black"), (P_WHITE, "white")] {
      let mut params = base_params.clone();
      params.playout_doubling_advantage = 1.5;
      params.playout_doubling_advantage_pla = force_pla;
      coutln!("Basic search with PDA 1.5, force {}", label);

      let mut search = new_search(&params, nn_eval, "autoSearchRandSeed3");
      let mut next_pla: Player;

      search.set_position(start_pla, &board, &hist).unwrap();
      next_pla = start_pla;

      let mut move_loc = search_and_get_move(ctx, &mut search, next_pla, false);
      print_search_results(&search);

      coutln!("Search next player - should preserve tree");
      search.make_move(move_loc, next_pla).unwrap();
      next_pla = get_opp(next_pla);
      move_loc = search_and_get_move(ctx, &mut search, next_pla, false);
      print_search_results(&search);

      coutln!("Search next player - should preserve tree");
      search.make_move(move_loc, next_pla).unwrap();
      next_pla = get_opp(next_pla);
      move_loc = search_and_get_move(ctx, &mut search, next_pla, false);
      print_search_results(&search);

      coutln!("Search next player PONDERING - should preserve tree");
      search.make_move(move_loc, next_pla).unwrap();
      next_pla = get_opp(next_pla);
      let mut pondering = true;
      move_loc = search_and_get_move(ctx, &mut search, next_pla, pondering);
      print_search_results(&search);

      if force_pla == P_WHITE {
        coutln!("Search next player PONDERING - an extra time, should preserve tree");
        search.make_move(move_loc, next_pla).unwrap();
        next_pla = get_opp(next_pla);
        pondering = true;
        move_loc = search_and_get_move(ctx, &mut search, next_pla, pondering);
        print_search_results(&search);
      }
      let _ = pondering;

      coutln!("Search next player - should preserve tree");
      search.make_move(move_loc, next_pla).unwrap();
      next_pla = get_opp(next_pla);
      search_whole(ctx, &mut search, next_pla, false);
      print_search_results(&search);

      drop(search);
      clear_cache_and_stats(nn_eval);
    }

    {
      let mut params = base_params.clone();
      params.playout_doubling_advantage = 1.5;
      coutln!("Basic search with PDA 1.5, no player");

      let mut search = new_search(&params, nn_eval, "autoSearchRandSeed3");
      let mut next_pla: Player;

      search.set_position(start_pla, &board, &hist).unwrap();
      next_pla = start_pla;

      let mut move_loc = search_and_get_move(ctx, &mut search, next_pla, false);
      print_search_results(&search);

      coutln!("Search next player PONDERING - should keep prior tree and PDA");
      search.make_move(move_loc, next_pla).unwrap();
      next_pla = get_opp(next_pla);
      let pondering = true;
      move_loc = search_and_get_move(ctx, &mut search, next_pla, pondering);
      print_search_results(&search);

      coutln!("Search next player - should keep tree from ponder");
      search.make_move(move_loc, next_pla).unwrap();
      next_pla = get_opp(next_pla);
      search_whole(ctx, &mut search, next_pla, false);
      print_search_results(&search);

      drop(search);
      clear_cache_and_stats(nn_eval);
    }

    {
      let mut params = base_params.clone();
      params.playout_doubling_advantage = 1.5;
      coutln!("Basic search with PDA 1.5, no player");

      let mut search = new_search(&params, nn_eval, "autoSearchRandSeed3");
      let mut next_pla: Player;

      search.set_position(start_pla, &board, &hist).unwrap();
      next_pla = start_pla;

      let mut move_loc = search_and_get_move(ctx, &mut search, next_pla, false);
      print_search_results(&search);

      coutln!("Search next player PONDERING - should keep prior tree and PDA");
      search.make_move(move_loc, next_pla).unwrap();
      next_pla = get_opp(next_pla);
      let mut pondering = true;
      move_loc = search_and_get_move(ctx, &mut search, next_pla, pondering);
      print_search_results(&search);

      coutln!("Search next player PONDERING an extra time - should keep prior tree and PDA");
      search.make_move(move_loc, next_pla).unwrap();
      next_pla = get_opp(next_pla);
      pondering = true;
      move_loc = search_and_get_move(ctx, &mut search, next_pla, pondering);
      print_search_results(&search);

      coutln!("Search next player - now should lose the tree and PDA, because the player it is for is different");
      search.make_move(move_loc, next_pla).unwrap();
      next_pla = get_opp(next_pla);
      search_whole(ctx, &mut search, next_pla, false);
      print_search_results(&search);

      drop(search);
      clear_cache_and_stats(nn_eval);
    }

    {
      let mut params = base_params.clone();
      params.playout_doubling_advantage = 1.5;
      coutln!("Basic search with PDA 1.5, no player");

      let mut search = new_search(&params, nn_eval, "autoSearchRandSeed3");
      let mut next_pla: Player;

      search.set_position(start_pla, &board, &hist).unwrap();
      next_pla = start_pla;

      let mut move_loc = search_and_get_move(ctx, &mut search, next_pla, false);
      print_search_results(&search);

      coutln!("Search next player PONDERING - should keep prior tree and PDA");
      search.make_move(move_loc, next_pla).unwrap();
      next_pla = get_opp(next_pla);
      let mut pondering = true;
      move_loc = search_and_get_move(ctx, &mut search, next_pla, pondering);
      print_search_results(&search);

      coutln!("Search next player PONDERING - should still keep prior tree and PDA");
      search.make_move(move_loc, next_pla).unwrap();
      next_pla = get_opp(next_pla);
      pondering = true;
      search_whole(ctx, &mut search, next_pla, pondering);
      print_search_results(&search);

      coutln!("Without making a move - convert ponder to regular search, should still keep tree and PDA");
      search_whole(ctx, &mut search, next_pla, false);
      print_search_results(&search);

      clear_cache_and_stats(nn_eval);

      coutln!("Set position to original, search PONDERING");
      search.set_position(start_pla, &board, &hist).unwrap();
      next_pla = start_pla;
      pondering = true;
      search_whole(ctx, &mut search, next_pla, pondering);
      print_search_results(&search);

      coutln!("Without making a move, convert to regular search, should not keep tree");
      coutln!("and should not benefit from cache, since search would guess the opponent as 'our' side");
      move_loc = search_and_get_move(ctx, &mut search, next_pla, false);
      print_search_results(&search);

      coutln!("But should be fine thereafter. Make two moves and continue");
      search.make_move(move_loc, next_pla).unwrap();
      next_pla = get_opp(next_pla);
      search.make_move(loc_of("D4", &board), next_pla).unwrap();
      next_pla = get_opp(next_pla);
      search_whole(ctx, &mut search, next_pla, false);
      print_search_results(&search);

      clear_cache_and_stats(nn_eval);

      coutln!("Set position to original, search PONDERING");
      search.set_position(start_pla, &board, &hist).unwrap();
      next_pla = start_pla;
      pondering = true;
      move_loc = search_and_get_move(ctx, &mut search, next_pla, pondering);
      print_search_results(&search);

      coutln!("Play that move and real search on the next position, should keep tree because correct guess of side");
      search.make_move(move_loc, next_pla).unwrap();
      next_pla = get_opp(next_pla);
      search_whole(ctx, &mut search, next_pla, false);
      print_search_results(&search);

      drop(search);
      clear_cache_and_stats(nn_eval);
    }
  }

  {
    //Reset random seeds for nnEval
    respawn_server_threads(ctx, nn_eval);

    let board = board_input("runv8tests", 1, 19, 19);

    let next_pla = P_BLACK;
    let rules = Rules::parse_rules("Chinese").unwrap();
    let hist = Box::new(BoardHistory::new_full(&board, next_pla, &rules, 0));

    let mut params = SearchParams::for_tests_v1();
    params.max_visits = 400;
    params.root_noise_enabled = true;
    params.root_policy_temperature = 1.2;
    params.root_policy_temperature_early = 1.2;
    params.root_num_symmetries_to_sample = 2;

    let opts = TestSearchOptions::new();
    let mut opts_continue = TestSearchOptions::new();
    opts_continue.ignore_position = true;
    let _ = opts_continue;

    {
      header_trio("Test real hintloc T16 (with symmetry sampling)");
      let mut bot = new_bot(&params, nn_eval, "hintloc");
      bot.set_root_hint_loc(loc_of("T16", &board));
      run_bot_on_position(ctx, &mut bot, &board, next_pla, &hist, &opts);
      drop(bot);
    }
    {
      header_trio("Test bad hintloc O18 (with symmetry sampling)");
      let mut bot = new_bot(&params, nn_eval, "hintloc");
      bot.set_root_hint_loc(loc_of("O18", &board));
      run_bot_on_position(ctx, &mut bot, &board, next_pla, &hist, &opts);
      drop(bot);
    }

    //Reset random seeds for nnEval
    respawn_server_threads(ctx, nn_eval);
  }

  {
    coutln!("AntiMirror white ==========================================================================");

    let sgf_str = "(;KM[7.5]SZ[19];B[pd];W[dp];B[pp];W[dd];B[cc];W[qq];B[dc];W[pq];B[op];W[ed];B[qp];W[cd];B[ec];W[oq];B[nq];W[fc];B[mp];W[gd];B[rp];W[bd];B[fq];W[nc];B[pi];W[dk];B[fe];W[no];B[cq];W[qc];B[pc];W[dq];B[cp];W[qd];B[do];W[pe];B[oe];W[eo];B[en];W[of];B[nf];W[fn];B[fd];W[np];B[mo];W[ge];B[fo];W[ne];B[od];W[ep];B[gn];W[mf];B[ng];W[fm];B[dn];W[pf];B[ff];W[nn];B[nd];W[fp];B[go];W[me];B[mr];W[gb];B[md];W[gp];B[gm];W[mg];B[mh];W[gl];B[hp];W[ld];B[lc];W[hq];B[fl];W[nh];B[gq];W[mc];B[pb];W[dr];B[hr];W[lb];B[kc];W[iq];B[ir];W[kb];B[hc];W[lq];B[og];W[em];B[hl];W[lh];B[mi];W[gk];B[le];W[ho];B[in];W[kf];B[jq];W[jc];B[pg];W[dm];B[kd];W[ip];B[mq];W[gc];B[bn];W[rf];B[cm];W[qg];B[qh];W[cl];B[rg];W[bm];B[cn];W[qf];B[li];W[hk];B[il];W[kh];B[rh];W[bl];B[bo];W[re];B[ik];W[ki];B[kj];W[ij];B[jj])";
    let sgf = CompactSgf::parse(sgf_str);

    for turn_idx in [24, 32, 124] {
      let initial_rules = sgf.get_rules_or_fail_allow_unspecified(&Rules::get_tromp_taylorish());
      let (board, next_pla, hist) = setup_from_sgf(&sgf, &initial_rules, turn_idx);
      let mut params = SearchParams::for_tests_v1();
      params.max_visits = 200;
      params.anti_mirror = true;
      let mut bot = new_bot(&params, nn_eval, "antimirrortest");

      let opts = TestSearchOptions::new();
      run_bot_on_position(ctx, &mut bot, &board, next_pla, &hist, &opts);
      drop(bot);
    }

    coutln!();
    coutln!();
  }

  {
    coutln!("AntiMirror black negkomi ==========================================================================");

    let sgf_str = "(;SZ[19]KM[-3.50];B[jj];W[pd];B[dp];W[dd];B[pp];W[cn];B[qf];W[nq];B[fc];W[qn];B[cf];W[df];B[pn];W[pm];B[dg];W[po];B[de];W[fd];B[np];W[mp];B[gd];W[ed];B[op];W[on];B[ef];W[gc];B[mq];W[lq];B[hc];W[jk];B[ji];W[ik];B[ki];W[ij];B[kj];W[gb];B[mr];W[ic];B[kq];W[pf];B[dn];W[do];B[pe];W[qe];B[co];W[lp];B[hd];W[eo];B[oe];W[qg];B[cm];W[bn];B[rf];W[qd];B[cp];W[hb];B[lr];W[bm];B[rg];W[of];B[en];W[fn];B[nf];W[qh];B[cl];W[ck];B[qi];W[ne];B[fo];W[ep];B[od];W[ng];B[fm];W[gn];B[mf];W[mg];B[gm];W[lf];B[hn];W[rh];B[bl];W[me];B[go];W[ii];B[kk];W[kl])";
    let sgf = CompactSgf::parse(sgf_str);

    for turn_idx in [29, 83] {
      let initial_rules = sgf.get_rules_or_fail_allow_unspecified(&Rules::get_tromp_taylorish());
      let (board, next_pla, hist) = setup_from_sgf(&sgf, &initial_rules, turn_idx);
      let mut params = SearchParams::for_tests_v1();
      params.max_visits = 200;
      params.anti_mirror = true;
      let mut bot = new_bot(&params, nn_eval, "antimirrortest");

      let opts = TestSearchOptions::new();
      run_bot_on_position(ctx, &mut bot, &board, next_pla, &hist, &opts);
      drop(bot);
    }

    coutln!();
    coutln!();
  }
}

//------------------------------------------------------------------------------------------------
// runMoreV8Tests

const MORE: &str = "runmorev8tests";

/// upstream: the `runTest` lambda of the "Multithreaded tree updating" block of runMoreV8Tests (testsearchv8.cpp)
fn multithreaded_tree_updating_run_test(
  ctx: &mut Ctx,
  nn_eval: &Rc<NNEvaluator>,
  board_base: &Board,
  next_pla_base: Player,
  hist_base: &BoardHistory,
  params_base: &SearchParams,
  num_visits: i64,
  num_threads: i32,
  subtree_value_bias: bool,
  graph_search: bool,
) {
  let mut board = board_base.clone();
  let mut next_pla = next_pla_base;
  let mut hist = Box::new(hist_base.clone());

  let mut params = params_base.clone();
  params.max_visits = num_visits;
  if subtree_value_bias {
    params.subtree_value_bias_factor = 0.35;
    params.subtree_value_bias_weight_exponent = 0.8;
    params.subtree_value_bias_free_prop = 0.0;
  }
  if graph_search {
    params.use_graph_search = true;
  }

  clear_cache_and_stats(nn_eval);
  let mut search = new_search(&params, nn_eval, "multithreaded tree updating");
  search.set_position(next_pla, &board, &hist).unwrap();
  search_whole(ctx, &mut search, next_pla, false);

  params.num_threads = num_threads;
  search.set_params_no_clearing(&params);

  let mut options = PrintTreeOptions::new();
  options = options.max_depth(1);
  print_basic_stuff_after_search(&board, &hist, &search, &options);

  let move_loc = loc_of("E2", &board);
  search.make_move(move_loc, next_pla).unwrap();
  hist.make_board_move_assume_legal(&mut board, move_loc, next_pla, None, false);
  next_pla = get_opp(next_pla);
  let _ = next_pla;

  coutln!("Just after move");
  print_tree(&search, &options, P_WHITE);
  let pondering = false;
  ctx.run("beginSearch", search.begin_search(pondering)).unwrap();
  coutln!("Just after begin search");
  if subtree_value_bias {
    coutln!("Skipping since exact values are nondeterministic due to subtree value bias float update order");
  } else {
    print_tree(&search, &options, P_WHITE);
  }

  drop(search);
  clear_cache_and_stats(nn_eval);
}

/// upstream: runMoreV8Tests (testsearchv8.cpp) static
fn run_more_v8_tests(ctx: &mut Ctx, nn_eval: &Rc<NNEvaluator>) {
  {
    coutln!("TEST VALUE BIAS ==========================================================================");

    let sgf_str = "(;GM[1]FF[4]CA[UTF-8]RU[Japanese]SZ[9]KM[0];B[dc];W[ef];B[df];W[de];B[dg];W[eg];B[eh];W[fh];B[ee])";
    let sgf = CompactSgf::parse(sgf_str);
    let initial_rules = sgf.get_rules_or_fail();
    let (board, next_pla, hist) = setup_from_sgf(&sgf, &initial_rules, 8);

    let mut params = SearchParams::for_tests_v1();
    params.max_visits = 20;
    params.chosen_move_temperature = 0.0;
    let mut bot_a = new_bot(&params, nn_eval, "valuebias test");
    params.subtree_value_bias_factor = 0.5;
    let mut bot_b = new_bot(&params, nn_eval, "valuebias test");
    params.max_visits = 300;
    let mut bot_c = new_bot(&params, nn_eval, "valuebias test");

    let mut opts = TestSearchOptions::new();
    opts.print_more_more_more = true;
    opts.num_moves_in_a_row = 3;
    opts.print_after_begun = true;
    coutln!("BASE");
    run_bot_on_position(ctx, &mut bot_a, &board, next_pla, &hist, &opts);
    coutln!("VALUE BIAS 0.5");
    run_bot_on_position(ctx, &mut bot_b, &board, next_pla, &hist, &opts);

    opts.print_more_more_more = false;
    run_bot_on_position(ctx, &mut bot_c, &board, next_pla, &hist, &opts);

    coutln!();
    coutln!();
  }

  {
    let board = board_input(MORE, 0, 11, 11);

    let mut params = SearchParams::for_tests_v1();
    params.max_visits = 300;
    params.fpu_reduction_max = 0.0;
    params.root_fpu_reduction_max = 0.0;
    params.root_ending_bonus_points = 0.0;
    params.root_policy_temperature = 1.5;
    params.root_policy_temperature_early = 1.5;
    let mut params2 = params.clone();
    params2.root_ending_bonus_points = 0.5;

    let mut opts = TestSearchOptions::new();
    opts.print_ending_score_value_bonus = true;

    for (title, next_pla, rules_str, has_button, encore) in [
      ("Ending bonus points in area scoring with selfatari moves, white first", P_WHITE, "Chinese", false, 0),
      ("Ending bonus points in area scoring with selfatari moves, white first, button", P_WHITE, "Chinese", true, 0),
      ("Ending bonus points in area scoring with selfatari moves, black first", P_BLACK, "Chinese", false, 0),
      ("Ending bonus points in area scoring with selfatari moves, black first, button", P_BLACK, "Chinese", true, 0),
      ("Ending bonus points in territory scoring with selfatari moves, black first", P_BLACK, "Japanese", false, 0),
      ("Ending bonus points in territory scoring with selfatari moves, black first, encore 2", P_BLACK, "Japanese", false, 2),
    ] {
      header_trio(title);

      let mut rules = Rules::parse_rules(rules_str).unwrap();
      if has_button {
        rules.has_button = true;
      }
      let hist = Box::new(BoardHistory::new_full(&board, next_pla, &rules, encore));
      {
        coutln!("Without root ending bonus pts===================");
        coutln!();
        let mut bot = new_bot(&params, nn_eval, "async bot ending bonus points seed");
        run_bot_on_position(ctx, &mut bot, &board, next_pla, &hist, &opts);
        drop(bot);
      }
      {
        coutln!("With root ending bonus pts===================");
        coutln!();
        let mut bot = new_bot(&params2, nn_eval, "async bot ending bonus points seed");
        run_bot_on_position(ctx, &mut bot, &board, next_pla, &hist, &opts);
        drop(bot);
      }
    }
  }

  {
    coutln!("TEST Ending bonus points ==========================================================================");

    let board = board_input(MORE, 1, 11, 11);

    let mut params = SearchParams::for_tests_v1();
    params.max_visits = 300;
    params.fpu_reduction_max = 0.0;
    params.root_fpu_reduction_max = 0.0;
    params.root_ending_bonus_points = 0.0;
    params.root_policy_temperature = 1.5;
    params.root_policy_temperature_early = 1.5;
    let mut params2 = params.clone();
    params2.root_ending_bonus_points = 0.5;

    let mut opts = TestSearchOptions::new();
    opts.print_ending_score_value_bonus = true;

    {
      header_trio("Ending bonus points in area scoring with selfatari moves, one more fancy position, white first");

      let next_pla = P_WHITE;
      let rules = Rules::parse_rules("Chinese").unwrap();
      let hist = Box::new(BoardHistory::new_full(&board, next_pla, &rules, 0));
      {
        let mut bot = new_bot(&params2, nn_eval, "async bot ending bonus points seed");
        run_bot_on_position(ctx, &mut bot, &board, next_pla, &hist, &opts);
        drop(bot);
      }
    }
  }

  for (title, k, max_visits, max_playouts) in [
    ("TEST futileVisitsThreshold ==========================================================================", 2, 400, None),
    ("TEST futileVisitsThreshold with playouts ==========================================================================", 3, 10000, Some(200)),
  ] {
    coutln!("{}", title);

    let next_pla = P_BLACK;
    let rules = Rules::get_tromp_taylorish();
    let board = board_input(MORE, k, 9, 9);
    let hist = Box::new(BoardHistory::new_full(&board, next_pla, &rules, 0));

    let mut params = SearchParams::for_tests_v1();
    params.max_visits = max_visits;
    if let Some(p) = max_playouts {
      params.max_playouts = p;
    }
    let mut bot_a = new_bot(&params, nn_eval, "futileVisitsThreshold test");
    params.futile_visits_threshold = 0.15;
    let mut bot_b = new_bot(&params, nn_eval, "futileVisitsThreshold test");
    params.futile_visits_threshold = 0.4;
    let mut bot_c = new_bot(&params, nn_eval, "futileVisitsThreshold test");

    let opts = TestSearchOptions::new();
    coutln!("BASE");
    run_bot_on_position(ctx, &mut bot_a, &board, next_pla, &hist, &opts);
    coutln!("futileVisitsThreshold 0.15");
    run_bot_on_position(ctx, &mut bot_b, &board, next_pla, &hist, &opts);
    coutln!("futileVisitsThreshold 0.4");
    run_bot_on_position(ctx, &mut bot_c, &board, next_pla, &hist, &opts);
    coutln!();
    coutln!();
  }

  {
    coutln!("TEST Hintlocs ==========================================================================");

    //Reset random seeds for nnEval
    respawn_server_threads(ctx, nn_eval);

    let board = board_input(MORE, 4, 19, 19);

    let next_pla = P_BLACK;
    let rules = Rules::parse_rules("Chinese").unwrap();
    let hist = Box::new(BoardHistory::new_full(&board, next_pla, &rules, 0));

    let mut params_fast = SearchParams::for_tests_v1();
    params_fast.max_visits = 5;
    let mut params_slow = SearchParams::for_tests_v1();
    params_slow.max_visits = 200;
    let mut params_fast_noised = params_fast.clone();
    params_fast_noised.root_noise_enabled = true;
    let mut params_slow_noised = params_slow.clone();
    params_slow_noised.root_noise_enabled = true;
    //Note - symmetry sampling here will randomize the root even though the nnEval is set to not randomize
    //because the root symmetry sampling is done in the search
    testAssert!(nn_eval.get_do_randomize() == false);
    let mut params_fast_sym = params_fast_noised.clone();
    params_fast_sym.root_num_symmetries_to_sample = 4;
    let mut params_slow_sym = params_slow_noised.clone();
    params_slow_sym.root_num_symmetries_to_sample = 4;

    let mut params_slow_sym_no_noise = params_slow_sym.clone();
    params_slow_sym_no_noise.root_noise_enabled = false;

    let mut opts = TestSearchOptions::new();
    opts.no_clear_bot = true;
    let mut opts_continue = TestSearchOptions::new();
    opts_continue.no_clear_bot = true;
    opts_continue.ignore_position = true;

    let c1 = loc_of("C1", &board);
    {
      header_trio("Test hintloc C1");
      let mut bot = new_bot(&params_slow, nn_eval, "hintloc");
      bot.set_root_hint_loc(c1);
      run_bot_on_position(ctx, &mut bot, &board, next_pla, &hist, &opts);
      drop(bot);
    }
    {
      header_trio("Test hintloc C1, again");
      let mut bot = new_bot(&params_slow, nn_eval, "hintloc");
      bot.set_root_hint_loc(c1);
      run_bot_on_position(ctx, &mut bot, &board, next_pla, &hist, &opts);
      drop(bot);
    }
    {
      header_trio("Test hintloc C1 after attempting same-turn tree reuse");
      let mut bot = new_bot(&params_fast, nn_eval, "hintloc");
      run_bot_on_position(ctx, &mut bot, &board, next_pla, &hist, &opts);
      bot.set_params_no_clearing(&params_slow);
      bot.set_root_hint_loc(c1); //This should actually clear the tree even though we didn't say to do so!
      run_bot_on_position(ctx, &mut bot, &board, next_pla, &hist, &opts_continue);
      drop(bot);
    }
    {
      header_trio("Test hintloc C1 dirichlet noise");
      let mut bot = new_bot(&params_slow_noised, nn_eval, "hintloc");
      bot.set_root_hint_loc(c1);
      run_bot_on_position(ctx, &mut bot, &board, next_pla, &hist, &opts);
      drop(bot);
    }
    {
      header_trio("Test hintloc C1 dirichlet noise after attempting same-turn tree reuse");
      let mut bot = new_bot(&params_fast_noised, nn_eval, "hintloc");
      run_bot_on_position(ctx, &mut bot, &board, next_pla, &hist, &opts);
      bot.set_params_no_clearing(&params_slow_noised);
      bot.set_root_hint_loc(c1); //This should actually clear the tree even though we didn't say to do so!
      run_bot_on_position(ctx, &mut bot, &board, next_pla, &hist, &opts_continue);
      drop(bot);
    }
    {
      header_trio("Test hintloc C1 dirichlet noise and symmetry sampling");
      let mut bot = new_bot(&params_slow_sym, nn_eval, "hintloc");
      bot.set_root_hint_loc(c1);
      run_bot_on_position(ctx, &mut bot, &board, next_pla, &hist, &opts);
      drop(bot);
    }
    {
      header_trio("Test hintloc C1 dirichlet noise and symmetry sampling, again (same due to matching search seed)");
      let mut bot = new_bot(&params_slow_sym, nn_eval, "hintloc");
      bot.set_root_hint_loc(c1);
      run_bot_on_position(ctx, &mut bot, &board, next_pla, &hist, &opts);
      drop(bot);
    }
    {
      header_trio("Test hintloc C1 dirichlet noise and symmetry sampling after attempting same-turn tree reuse");
      let mut bot = new_bot(&params_fast_sym, nn_eval, "hintloc");
      run_bot_on_position(ctx, &mut bot, &board, next_pla, &hist, &opts);
      bot.set_params_no_clearing(&params_slow_sym);
      bot.set_root_hint_loc(c1); //This should actually clear the tree even though we didn't say to do so!
      run_bot_on_position(ctx, &mut bot, &board, next_pla, &hist, &opts_continue);
      drop(bot);
    }
    {
      header_trio("Test hintloc C1 symmetry sampling alone, new seed");
      let mut bot = new_bot(&params_slow_sym_no_noise, nn_eval, "abc");
      bot.set_root_hint_loc(c1);
      run_bot_on_position(ctx, &mut bot, &board, next_pla, &hist, &opts);
      drop(bot);
    }
    {
      header_trio("Test hintloc C1 symmetry sampling alone, new seed 2");
      let mut bot = new_bot(&params_slow_sym_no_noise, nn_eval, "abc2");
      bot.set_root_hint_loc(c1);
      run_bot_on_position(ctx, &mut bot, &board, next_pla, &hist, &opts);
      drop(bot);
    }

    //Reset random seeds for nnEval
    respawn_server_threads(ctx, nn_eval);
  }

  {
    coutln!("Mix pruning, subtracting, dirichlet noise, value weight exponent =========================================");

    let board = board_input(MORE, 5, 19, 19);

    let next_pla = P_WHITE;
    let rules = Rules::parse_rules("Chinese").unwrap();
    let hist = Box::new(BoardHistory::new_full(&board, next_pla, &rules, 0));

    let mut params_base = SearchParams::for_tests_v1();
    params_base.max_visits = 500;
    let opts = TestSearchOptions::new();

    // (title, maxVisits override, rootNoiseEnabled, chosenMovePrune/Subtract, valueWeightExponent)
    let cases: [(&str, Option<i64>, bool, Option<(f64, f64)>, Option<f64>); 9] = [
      ("Base", None, false, None, None),
      ("Dirichlet noise", None, true, None, None),
      ("Dirichlet noise Prune 10 sub 7", None, true, Some((10.0, 7.0)), None),
      ("Dirichlet noise Value weight exponent 0.8", None, true, None, Some(0.8)),
      ("Dirichlet noise Value weight exponent 0.8 prune 12 sub 5", None, true, Some((12.0, 5.0)), Some(0.8)),
      ("Dirichlet noise Value weight exponent 0.8 more visits", Some(2500), true, None, Some(0.8)),
      ("Dirichlet noise Value weight exponent 0.8 prune 12 sub 5 more visits", Some(2500), true, Some((12.0, 5.0)), Some(0.8)),
      ("Dirichlet noise Value weight exponent 0.0 more visits", Some(2500), true, None, Some(0.0)),
      ("Dirichlet noise Value weight exponent 0.0 prune 7 sub 3 more visits", Some(2500), true, Some((12.0, 5.0)), Some(0.0)),
    ];
    for (title, max_visits, root_noise, prune_sub, vwe) in cases {
      header_trio(title);
      let mut params = params_base.clone();
      if let Some(v) = max_visits {
        params.max_visits = v;
      }
      if root_noise {
        params.root_noise_enabled = true;
      }
      if let Some((prune, sub)) = prune_sub {
        params.chosen_move_prune = prune;
        params.chosen_move_subtract = sub;
      }
      if let Some(e) = vwe {
        params.value_weight_exponent = e;
      }
      let mut bot = new_bot(&params, nn_eval, "mix");
      run_bot_on_position(ctx, &mut bot, &board, next_pla, &hist, &opts);
      drop(bot);
    }
  }

  {
    coutln!("Fill dame before pass =========================================");

    let board = board_input(MORE, 6, 9, 10);

    let mut params_base = SearchParams::for_tests_v1();
    params_base.max_visits = 600;
    let opts = TestSearchOptions::new();

    for (title, next_pla, fill_dame) in [
      ("Base, white to play", P_WHITE, false),
      ("Fill dame before pass, white to play", P_WHITE, true),
      ("Base, black to play", P_BLACK, false),
      ("Fill dame before pass, black to play", P_BLACK, true),
    ] {
      let rules = Rules::parse_rules("Japanese").unwrap();
      let hist = Box::new(BoardHistory::new_full(&board, next_pla, &rules, 0));

      header_trio(title);
      let mut params = params_base.clone();
      if fill_dame {
        params.fill_dame_before_pass = true;
      }
      let mut bot = new_bot(&params, nn_eval, "fill dame before pass");
      run_bot_on_position(ctx, &mut bot, &board, next_pla, &hist, &opts);
      drop(bot);
    }
  }

  {
    coutln!("Conservative pass =========================================");

    let board = board_input(MORE, 7, 9, 9);

    let mut params_base = SearchParams::for_tests_v1();
    params_base.max_visits = 600;
    let mut opts = TestSearchOptions::new();
    opts.no_clear_bot = true;

    for (next_pla, komi) in [(P_WHITE, Some(14.0f32)), (P_BLACK, None)] {
      let mut rules = Rules::parse_rules("Chinese").unwrap();
      if let Some(k) = komi {
        rules.komi = k;
      }
      let hist = Box::new(BoardHistory::new_full(&board, next_pla, &rules, 0));

      //Both blocks print "White to play" upstream.
      header_trio("White to play");
      let params = params_base.clone();
      let mut bot = new_bot(&params, nn_eval, "conservative pass");
      run_bot_on_position(ctx, &mut bot, &board, next_pla, &hist, &opts);

      let move_loc = bot.get_search_stop_and_wait().get_chosen_move_loc();
      let move_loc2 = PlayUtils::maybe_cleanup_before_pass(enabled_t::True, enabled_t::False, next_pla, move_loc, &bot);
      coutln!("Move loc: {}", Location::to_string(move_loc, &board));
      coutln!("Conservative pass: {}", Location::to_string(move_loc2, &board));
      drop(bot);
    }
  }

  {
    coutln!("Basic graph search 7x7 fight =========================================");

    let board = board_input(MORE, 8, 7, 7);
    let next_pla = P_WHITE;

    let mut params_base = SearchParams::for_tests_v1();
    params_base.max_visits = 1000;
    params_base.use_graph_search = true;
    let mut opts = TestSearchOptions::new();
    opts.num_moves_in_a_row = 3;
    opts.print_post_order_node_count = true;

    {
      let mut rules = Rules::parse_rules("Japanese").unwrap();
      rules.komi = 8.0;
      let hist = Box::new(BoardHistory::new_full(&board, next_pla, &rules, 0));

      header_trio("White to play");
      let params = params_base.clone();
      let mut bot = new_bot(&params, nn_eval, "conservative pass");
      run_bot_on_position(ctx, &mut bot, &board, next_pla, &hist, &opts);
      drop(bot);
    }
  }

  {
    coutln!("Friendly pass =========================================");

    let mut params_base = SearchParams::for_tests_v1();
    params_base.max_visits = 600;
    let mut opts = TestSearchOptions::new();
    opts.no_clear_bot = true;

    for (k, pass_pla, komi, title) in [(9, P_BLACK, 4.0f32, "White to play"), (10, P_WHITE, 7.0f32, "Black to play")] {
      let mut board = board_input(MORE, k, 9, 9);

      let mut next_pla = pass_pla;
      let mut rules = Rules::parse_rules("Chinese").unwrap();
      rules.komi = komi;
      let mut hist = Box::new(BoardHistory::new_full(&board, next_pla, &rules, 0));
      hist.make_board_move_assume_legal(&mut board, Board::PASS_LOC, next_pla, None, false);
      next_pla = get_opp(pass_pla);

      header_trio(title);
      let params = params_base.clone();
      let mut bot = new_bot(&params, nn_eval, "friendly pass");
      run_bot_on_position(ctx, &mut bot, &board, next_pla, &hist, &opts);

      let move_loc = bot.get_search_stop_and_wait().get_chosen_move_loc();
      let move_loc2 = ctx
        .run("maybeFriendlyPass", PlayUtils::maybe_friendly_pass(enabled_t::False, enabled_t::True, next_pla, move_loc, bot.get_search_stop_and_wait(), 50))
        .unwrap();
      coutln!("Move loc: {}", Location::to_string(move_loc, &board));
      coutln!("Friendly pass: {}", Location::to_string(move_loc2, &board));
      drop(bot);
    }
  }

  {
    coutln!("Multithreaded tree updating =========================================");

    let board_base = board_input(MORE, 11, 19, 19);

    let next_pla_base = P_BLACK;
    let mut rules = Rules::parse_rules("Japanese").unwrap();
    rules.komi = 6.5;
    let hist_base = Box::new(BoardHistory::new_full(&board_base, next_pla_base, &rules, 0));

    let mut params_base = SearchParams::for_tests_v1();
    params_base.max_visits = 1000;

    let run_test = |ctx: &mut Ctx, num_visits: i64, num_threads: i32, subtree_value_bias: bool, graph_search: bool| {
      multithreaded_tree_updating_run_test(ctx, nn_eval, &board_base, next_pla_base, &hist_base, &params_base, num_visits, num_threads, subtree_value_bias, graph_search);
    };

    header_trio("Baseline 1k visits");
    run_test(ctx, 1000, 1, false, false);

    header_trio("Baseline 1k visits, graph search");
    run_test(ctx, 1000, 1, false, true);

    header_trio("4 threads for search updates, 1k visits");
    run_test(ctx, 1000, 4, false, false);

    header_trio("4 threads for search updates, subtree value bias stability (but 0 free prop), 1k visits");
    run_test(ctx, 1000, 4, true, false);

    header_trio("Baseline 8k visits");
    run_test(ctx, 8000, 1, false, false);

    header_trio("4 threads for search updates, 8k visits");
    run_test(ctx, 8000, 4, false, false);

    header_trio("30 threads for search updates, 8k visits");
    run_test(ctx, 8000, 30, false, false);

    header_trio("30 threads for search updates, 8k visits, graph search");
    run_test(ctx, 8000, 30, false, true);

    header_trio("30 threads for search updates, subtree value bias stability (but 0 free prop), 8k visits");
    run_test(ctx, 8000, 4, true, false);
  }

  {
    coutln!("Pattern bonus =========================================");

    let mut board = board_input(MORE, 12, 19, 19);

    let mut next_pla = P_BLACK;
    let mut rules = Rules::parse_rules("Japanese").unwrap();
    rules.komi = 6.5;
    let mut hist = Box::new(BoardHistory::new_full(&board, next_pla, &rules, 0));

    let mut params0 = SearchParams::for_tests_v1();
    params0.max_visits = 1000;
    let mut params1 = params0.clone();
    params1.avoid_repeated_pattern_utility = 0.2;

    let mut opts = TestSearchOptions::new();
    opts.no_clear_bot = true;

    for m in ["R3", "Q3", "R4", "Q5", "S6", "F17", "C14"] {
      play(&mut hist, &mut board, m, next_pla);
      next_pla = get_opp(next_pla);
    }

    let mut bot0 = new_bot(&params0, nn_eval, "pattern bonus");
    let mut bot1 = new_bot(&params1, nn_eval, "pattern bonus");
    let mut bot2 = new_bot(&params1, nn_eval, "pattern bonus");
    coutln!("Bot0: baseline");
    run_bot_on_position(ctx, &mut bot0, &board, next_pla, &hist, &opts);
    coutln!("Bot1: avoid repeat 0.2");
    run_bot_on_position(ctx, &mut bot1, &board, next_pla, &hist, &opts);
    coutln!("Bot2: avoid repeat 0.2 but will only play one side");
    run_bot_on_position(ctx, &mut bot2, &board, next_pla, &hist, &opts);

    //We manually make the bots play through the next moves
    opts.ignore_position = true;
    play(&mut hist, &mut board, "D18", next_pla);
    bot0.make_move(loc_of("D18", &board), next_pla).unwrap();
    bot1.make_move(loc_of("D18", &board), next_pla).unwrap();
    bot2.make_move(loc_of("D18", &board), next_pla).unwrap();
    next_pla = get_opp(next_pla);

    coutln!("Bot0: baseline");
    run_bot_on_position(ctx, &mut bot0, &board, next_pla, &hist, &opts);
    coutln!("Bot1: avoid repeat 0.2");
    run_bot_on_position(ctx, &mut bot1, &board, next_pla, &hist, &opts);
    coutln!("Bot2: avoid repeat 0.2 but will only play one side, skipping");
    //runBotOnPosition(bot2, board, nextPla, hist, opts);

    play(&mut hist, &mut board, "M4", next_pla);
    bot0.make_move(loc_of("M4", &board), next_pla).unwrap();
    bot1.make_move(loc_of("M4", &board), next_pla).unwrap();
    bot2.make_move(loc_of("M4", &board), next_pla).unwrap();
    next_pla = get_opp(next_pla);

    coutln!("Bot0: baseline");
    run_bot_on_position(ctx, &mut bot0, &board, next_pla, &hist, &opts);
    coutln!("Bot1: avoid repeat 0.2");
    run_bot_on_position(ctx, &mut bot1, &board, next_pla, &hist, &opts);
    coutln!("Bot2: avoid repeat 0.2 but will only played one side");
    run_bot_on_position(ctx, &mut bot2, &board, next_pla, &hist, &opts);

    drop(bot0);
    drop(bot1);
    drop(bot2);
  }

  {
    coutln!("Pattern bonus does not care about ko =========================================");

    let mut board = board_input(MORE, 13, 19, 19);

    let mut next_pla = P_BLACK;
    let mut rules = Rules::parse_rules("Japanese").unwrap();
    rules.komi = 6.5;
    let mut hist = Box::new(BoardHistory::new_full(&board, next_pla, &rules, 0));

    let mut params0 = SearchParams::for_tests_v1();
    params0.max_visits = 1000;
    let mut params1 = params0.clone();
    params1.avoid_repeated_pattern_utility = 0.2;

    let mut opts = TestSearchOptions::new();
    opts.no_clear_bot = true;

    for m in ["R17", "D15", "C15", "Q17", "C5", "B5"] {
      play(&mut hist, &mut board, m, next_pla);
      next_pla = get_opp(next_pla);
    }

    let mut bot0 = new_bot(&params0, nn_eval, "pattern bonus");
    let mut bot1 = new_bot(&params1, nn_eval, "pattern bonus");
    coutln!("Bot0: baseline");
    run_bot_on_position(ctx, &mut bot0, &board, next_pla, &hist, &opts);
    coutln!("Bot1: avoid repeat 0.2");
    run_bot_on_position(ctx, &mut bot1, &board, next_pla, &hist, &opts);

    drop(bot0);
    drop(bot1);
  }

  {
    coutln!("Pattern bonus does not multi-count shapes =========================================");

    let mut board = board_input(MORE, 14, 19, 19);

    let mut next_pla = P_WHITE;
    let mut rules = Rules::parse_rules("Japanese").unwrap();
    rules.komi = 25.5;
    let mut hist = Box::new(BoardHistory::new_full(&board, next_pla, &rules, 0));

    let mut params0 = SearchParams::for_tests_v1();
    params0.max_visits = 1000;
    params0.avoid_repeated_pattern_utility = 0.2;

    let mut opts = TestSearchOptions::new();
    opts.no_clear_bot = false;
    let mut bot0 = new_bot(&params0, nn_eval, "pattern bonus");
    run_bot_on_position(ctx, &mut bot0, &board, next_pla, &hist, &opts);

    for pair in [["R14", "O17"], ["F17", "C14"], ["C6", "F3"]] {
      for m in pair {
        play(&mut hist, &mut board, m, next_pla);
        bot0.make_move(loc_of(m, &board), next_pla).unwrap();
        next_pla = get_opp(next_pla);
      }
      run_bot_on_position(ctx, &mut bot0, &board, next_pla, &hist, &opts);
    }

    drop(bot0);
  }
}

//------------------------------------------------------------------------------------------------
// runMoreV8Tests2

/// upstream: runMoreV8Tests2 (testsearchv8.cpp) static
fn run_more_v8_tests2(ctx: &mut Ctx, nn_eval: &Rc<NNEvaluator>) {
  {
    coutln!("TEST ownership endgame ==========================================================================");

    let next_pla = P_WHITE;
    let rules = Rules::get_tromp_taylorish();
    let board = board_input("runmorev8tests2", 0, 7, 9);
    let hist = Box::new(BoardHistory::new_full(&board, next_pla, &rules, 0));

    let mut params = SearchParams::for_tests_v1();
    params.max_visits = 100;
    params.futile_visits_threshold = 0.4;
    let mut bot = new_bot(&params, nn_eval, "Endgame ownership test");

    let mut opts = TestSearchOptions::new();
    opts.print_ownership = true;
    run_bot_on_position(ctx, &mut bot, &board, next_pla, &hist, &opts);
    drop(bot);
  }
}

//------------------------------------------------------------------------------------------------
// runMoreV8TestsRandomizedNNEvals

/// `std::set<double>::insert`
fn set_insert(set: &mut Vec<f64>, x: f64) {
  if !set.contains(&x) {
    set.push(x);
    set.sort_by(|a, b| a.partial_cmp(b).unwrap());
  }
}

/// upstream: runMoreV8TestsRandomizedNNEvals (testsearchv8.cpp) static
fn run_more_v8_tests_randomized_nn_evals(ctx: &mut Ctx, nn_eval: &Rc<NNEvaluator>) {
  {
    coutln!("TEST sampled symmetries ==========================================================================");
    let board = board_input("runmorev8testsrandomizednnevals", 0, 15, 15);

    let next_pla = P_BLACK;
    let rules = Rules::parse_rules("AGA").unwrap();
    let hist = Box::new(BoardHistory::new_full(&board, next_pla, &rules, 0));

    let mut params = SearchParams::for_tests_v1();
    params.root_num_symmetries_to_sample = 8;
    params.max_visits = 1;

    let mut opts = TestSearchOptions::new();
    opts.print_root_policy = true;
    for seed in ["sample", "sample2", "sample3"] {
      header_trio(
        "Repeatedly run bot with 8 root symmetries sampled, should be deterministic (except for scoreUtility due to dynamic score centering)",
      );
      let mut bot = new_bot(&params, nn_eval, seed);
      run_bot_on_position(ctx, &mut bot, &board, next_pla, &hist, &opts);
      drop(bot);
    }

    {
      header_trio("Repeatedly run bot with 2 root symmetries sampled");
      let mut params2 = params.clone();
      params2.root_num_symmetries_to_sample = 2;
      let mut bot = new_bot(&params2, nn_eval, "two root syms");
      bot.set_position(next_pla, &board, &hist).unwrap();
      let mut policy_samples: Vec<f64> = Vec::new();
      let mut wl_samples: Vec<f64> = Vec::new();

      let mut policy_probs = [0.0f32; NNPos::MAX_NN_POLICY_SIZE as usize];
      for _ in 0..500 {
        ctx.run("genMoveSynchronous", bot.gen_move_synchronous(next_pla, &TimeControls::new())).unwrap();
        bot.get_search().get_policy(&mut policy_probs);
        set_insert(&mut policy_samples, policy_probs[NNPos::xy_to_pos(2, 4, bot.get_search().nn_x_len) as usize] as f64);
        set_insert(&mut wl_samples, bot.get_search().get_root_values_require_success().unwrap().win_loss_value);
        bot.clear_search();
      }
      drop(bot);
      let mut i;
      i = 0;
      coutln!("Policy samples");
      for x in &policy_samples {
        coutln!("{} {}", i, cpp_ostream_f64(*x));
        i += 1;
      }
      i = 0;
      coutln!("WL samples");
      for x in &wl_samples {
        coutln!("{} {}", i, cpp_ostream_f64(*x));
        i += 1;
      }
    }
  }
}

//------------------------------------------------------------------------------------------------
// runV8SearchMultithreadTest

/// `cout << "Chosen move: " ...; cout << "<what> near <v>: " << (std::fabs(... - (c)) < tol) << endl;` x 2
fn print_multithread_summary(search: &Search, board: &Board, move_loc: Loc, wl_label: &str, wl_c: f64, wl_tol: f64, lead_label: &str, lead_c: f64, lead_tol: f64) {
  coutln!("Chosen move: {}", Location::to_string(move_loc, board));
  coutln!("{}: {}", wl_label, ((search.get_root_values_require_success().unwrap().win_loss_value - wl_c).abs() < wl_tol) as i32);
  coutln!("{}: {}", lead_label, ((search.get_root_values_require_success().unwrap().lead - lead_c).abs() < lead_tol) as i32);
}

/// upstream: runV8SearchMultithreadTest (testsearchv8.cpp) static
fn run_v8_search_multithread_test(ctx: &mut Ctx, nn_eval: &Rc<NNEvaluator>) {
  {
    coutln!("Multithreaded search test =========================================");

    let mut board = board_input("runv8searchmultithreadtest", 0, 19, 19);

    let mut next_pla = P_BLACK;
    let mut rules = Rules::parse_rules("Japanese").unwrap();
    rules.komi = 8.5;
    let mut hist = Box::new(BoardHistory::new_full(&board, next_pla, &rules, 0));

    let mut params = SearchParams::for_tests_v1();
    params.max_visits = 16000;
    params.subtree_value_bias_factor = 0.35;
    params.subtree_value_bias_weight_exponent = 0.8;
    params.subtree_value_bias_free_prop = 0.8;
    params.chosen_move_temperature = 0.0;
    params.chosen_move_temperature_early = 0.0;
    params.use_noise_pruning = true;
    params.num_threads = 40;

    let mut move_loc: Loc;

    let mut search = new_search(&params, nn_eval, "multithreaded test");
    search.set_position(next_pla, &board, &hist).unwrap();
    search_whole(ctx, &mut search, next_pla, false);
    move_loc = search.get_chosen_move_loc();
    print_multithread_summary(&search, &board, move_loc, "Winloss near 0.05", 0.05, 0.05, "Lead near 1", 1.0, 1.0);

    for _ in 0..2 {
      search.make_move(move_loc, next_pla).unwrap();
      hist.make_board_move_assume_legal(&mut board, move_loc, next_pla, None, false);
      next_pla = get_opp(next_pla);
      search_whole(ctx, &mut search, next_pla, false);
      move_loc = search.get_chosen_move_loc();
      print_multithread_summary(&search, &board, move_loc, "Winloss near 0.05", 0.05, 0.05, "Lead near 1", 1.0, 1.0);
    }

    //Enumerate the tree and make sure every node is indeed hit exactly once in postorder.
    verify_tree_post_order(&search, 16000);

    //With 16000 visits per move and three searches, we very likely have about that many nn evals (this shouldn't be a heavily transposing position)
    //and despite doing 3x such searches, we should have caching and tree reuse keep it not much more than that.
    let num_rows_processed = nn_eval.num_rows_processed() as i64;
    coutln!("numRowsProcessed as expected: {}", (num_rows_processed > 14000 && num_rows_processed < 28000) as i32);
    eprintln!("(Multithreaded search test: numRowsProcessed {})", num_rows_processed);

    drop(search);
    clear_cache_and_stats(nn_eval);
  }

  {
    coutln!("Multithreaded graph search test =========================================");

    let mut board = board_input("runv8searchmultithreadtest", 1, 19, 19);

    let mut next_pla = P_WHITE;
    let mut rules = Rules::parse_rules("Chinese").unwrap();
    rules.komi = 6.5;
    let mut hist = Box::new(BoardHistory::new_full(&board, next_pla, &rules, 0));

    let mut params = SearchParams::for_tests_v1();
    params.max_visits = 8000;
    params.use_graph_search = true;
    params.subtree_value_bias_factor = 0.35;
    params.subtree_value_bias_weight_exponent = 0.8;
    params.subtree_value_bias_free_prop = 0.8;
    params.chosen_move_temperature = 0.0;
    params.chosen_move_temperature_early = 0.0;
    params.use_noise_pruning = true;
    params.num_threads = 40;

    let mut move_loc: Loc;

    let mut search = new_search(&params, nn_eval, "multithreaded test");
    search.set_position(next_pla, &board, &hist).unwrap();
    search_whole(ctx, &mut search, next_pla, false);
    move_loc = search.get_chosen_move_loc();
    print_multithread_summary(&search, &board, move_loc, "Winloss near 0.63", 0.63, 0.1, "Lead near 15.5", 15.8, 1.8);

    //Enumerate the tree and make sure every node is indeed hit exactly once in postorder.
    verify_tree_post_order(&search, 5000);

    search.make_move(move_loc, next_pla).unwrap();
    hist.make_board_move_assume_legal(&mut board, move_loc, next_pla, None, false);
    next_pla = get_opp(next_pla);
    search_whole(ctx, &mut search, next_pla, false);
    move_loc = search.get_chosen_move_loc();
    print_multithread_summary(&search, &board, move_loc, "Winloss near 0.63", 0.63, 0.1, "Lead near 15.5", 15.8, 1.8);

    //Enumerate the tree and make sure every node is indeed hit exactly once in postorder.
    verify_tree_post_order(&search, 5000);

    //With 8000 visits per move and 2 searches, we very likely have about that many nn evals
    let num_rows_processed = nn_eval.num_rows_processed() as i64;
    coutln!("numRowsProcessed as expected: {}", (num_rows_processed > 5000 && num_rows_processed < 13000) as i32);
    eprintln!("(Multithreaded graph search test: numRowsProcessed {})", num_rows_processed);

    drop(search);
    clear_cache_and_stats(nn_eval);
  }
}

//------------------------------------------------------------------------------------------------
// Tests::runSearchTestsV8, one #[test] per evaluator group

/// `startNNEval(modelFile,logger,"v8seed",x,y,sym,inputsNHWC,useNHWC,useFP16,false,exact)` (NHWC/FP16: see the header)
fn start(ctx: &mut Ctx, x: i32, y: i32, sym: i32, exact: bool) -> Rc<NNEvaluator> {
  start_nn_eval(ctx, MODEL_FILE, "v8seed", x, y, sym, false, false, false, false, exact)
}

/// Group 0: the preamble, the 19/9/9-exact evaluators (symmetry 1), runV8TestsSize9.
fn group0() {
  //TestCommon::overrideForBackends(inputsNHWC, useNHWC);
  coutln!("Running search tests introduced after v8 nets");
  //NeuralNet::globalInitialize();
  let mut ctx = Ctx::new();
  let nn_eval = start(&mut ctx, 19, 19, 1, false);
  let nn_eval9 = start(&mut ctx, 9, 9, 1, false);
  let nn_eval9_exact = start(&mut ctx, 9, 9, 1, true);
  run_v8_tests_size9(&mut ctx, &nn_eval, &nn_eval9, &nn_eval9_exact);
  delete_nn_eval(&mut ctx, nn_eval);
  delete_nn_eval(&mut ctx, nn_eval9);
  delete_nn_eval(&mut ctx, nn_eval9_exact);
}

/// Group 1: two randomized (symmetry -1) 19x19 evaluators, runV8TestsRandomSym.
fn group1() {
  let mut ctx = Ctx::new();
  let nn_eval = start(&mut ctx, 19, 19, -1, false);
  let nn_eval19_exact = start(&mut ctx, 19, 19, -1, true);
  run_v8_tests_random_sym(&mut ctx, &nn_eval, &nn_eval19_exact);
  delete_nn_eval(&mut ctx, nn_eval);
  delete_nn_eval(&mut ctx, nn_eval19_exact);
}

/// Group 2: symmetry 2, runV8Tests.
fn group2() {
  let mut ctx = Ctx::new();
  let nn_eval = start(&mut ctx, 19, 19, 2, false);
  run_v8_tests(&mut ctx, &nn_eval);
  delete_nn_eval(&mut ctx, nn_eval);
}

/// Group 3: symmetry 5, runMoreV8Tests.
fn group3() {
  let mut ctx = Ctx::new();
  let nn_eval = start(&mut ctx, 19, 19, 5, false);
  run_more_v8_tests(&mut ctx, &nn_eval);
  delete_nn_eval(&mut ctx, nn_eval);
}

/// Group 4: symmetry 5, runMoreV8Tests2.
fn group4() {
  let mut ctx = Ctx::new();
  let nn_eval = start(&mut ctx, 19, 19, 5, false);
  run_more_v8_tests2(&mut ctx, &nn_eval);
  delete_nn_eval(&mut ctx, nn_eval);
}

/// Group 5: randomized, runMoreV8TestsRandomizedNNEvals.
fn group5() {
  let mut ctx = Ctx::new();
  let nn_eval = start(&mut ctx, 19, 19, -1, false);
  run_more_v8_tests_randomized_nn_evals(&mut ctx, &nn_eval);
  delete_nn_eval(&mut ctx, nn_eval);
}

/// Group 6: randomized, runV8SearchMultithreadTest, then the end of runSearchTestsV8.
fn group6() {
  let mut ctx = Ctx::new();
  let nn_eval = start(&mut ctx, 19, 19, -1, false);
  run_v8_search_multithread_test(&mut ctx, &nn_eval);
  // Suppress some nondeterministc messages about number of batches
  LOGGER_DISABLED.with(|d| d.set(true));
  delete_nn_eval(&mut ctx, nn_eval);

  //NeuralNet::globalCleanup();
  coutln!("Done");
}

//------------------------------------------------------------------------------------------------
// Comparison policy

/// Lines compared in the 40-thread sections (upstream: values are thread-timing dependent; the test itself only
/// prints booleans and the post-order check). "Chosen move" is reported in the dump but not compared.
const MULTITHREAD_SUMMARY: &[&str] = &["Winloss near", "Lead near", "Post order", "numRowsProcessed as expected"];

fn policy_of(label: &str) -> Policy {
  let title = label.split_once(' ').map(|(_, t)| t).unwrap_or(label);
  match title {
    "Multithreaded search test" | "Multithreaded graph search test" => Policy::SummaryOnly(MULTITHREAD_SUMMARY),
    _ => Policy::Full,
  }
}

/// The 8000-visit runs of "Multithreaded tree updating": a long single-thread search where the ~1e-5 burn-vs-Eigen NN
/// difference shifts a few visits between near-tied grandchildren (same root children, same root child visit counts,
/// E8 4062 vs 4064, H3 2655 vs 2656 below E2; NN rows 4781 vs 4794), as the 1000-visit analysis goldens do between
/// CUDA and Eigen (docs/M3-test-inventory.md §2 #7/#8, §5). Not structural: same moves, PVs, values within ~0.05c.
/// (Measured 2026-10, release; the four 1000-visit runs of the same section match within tolerance.)
const REASON_8K: &str = "8000-visit search: near-tie visit drift from ~1e-5 NN differences (NN rows 4781 vs 4794, grandchild visits +-2..20); \
                         same root children/PVs, values within 0.05c (inventory §5)";

/// Sections that are known to diverge from the Eigen rerun, with the reason (first diverging line in the report).
const ALLOWLIST: &[Allowed] = &[
  Allowed { label: "#76 Baseline 8k visits", reason: REASON_8K },
  Allowed { label: "#77 4 threads for search updates, 8k visits", reason: REASON_8K },
  Allowed { label: "#78 30 threads for search updates, 8k visits", reason: REASON_8K },
  Allowed { label: "#79 30 threads for search updates, 8k visits, graph search", reason: REASON_8K },
  // Pattern bonus sections (enabled with patternbonustable.rs, 2026-10; release, group 3). Measured: every line
  // matches except the ones named here.
  Allowed {
    label: "#81 Pattern bonus",
    reason: "one visit moves between near-tied root children (L4 27 -> 26, C4 80 -> 81) in Bot1's tree-reuse search after D18 \
             (1000 visits, ~1e-5 NN differences). The C++ result itself depends on the NN cache contents left by earlier \
             sections: the same search run standalone in C++ gives C4 81 / L4 27, in the full group C4 80 / L4 27. The other \
             7 searches of the section, including all pattern-bonus ones and Bot1's next reuse search, are exact",
  },
  Allowed {
    label: "#82 Pattern bonus does not care about ko",
    reason: "a single visit at a depth-10 near-tie (O18 vs R19) goes elsewhere; NN outputs agree to 1e-6. Bot0 (baseline, no \
             pattern bonus) already differs the same way; Bot1 (avoid repeat 0.2) prints exactly Bot0's lines in both builds, \
             which is what the section checks",
  },
  Allowed {
    label: "#83 Pattern bonus does not multi-count shapes",
    reason: "only the group's closing 'GPU 0 finishing, processed N rows' line differs (55469 vs 55432): the evaluator's \
             cumulative row count, shifted by #81/#82; every search of the section is exact",
  },
];

/// The expected sections of group `g`: from the g-th "<startNNEval>" (group 0: from the start) up to the next one.
fn expected_group(g: usize) -> Vec<Section> {
  let text = fixture(STEM, "runsearchtestsv8-g170-bin.expected0.txt");
  let all = split_sections(&text, &[]);
  let starts: Vec<usize> = all.iter().enumerate().filter(|(_, s)| s.title == "<startNNEval>").map(|(i, _)| i).collect();
  assert_eq!(starts.len(), 7, "startNNEval groups in the expected text");
  let lo = if g == 0 { 0 } else { starts[g] };
  let hi = if g + 1 < starts.len() { starts[g + 1] } else { all.len() };
  all[lo..hi].to_vec()
}

fn run_group(g: usize, f: fn()) {
  let t0 = Instant::now();
  let expected = expected_group(g);
  assert!(!expected.is_empty());
  let name = format!("testsearchv8 group {}", g);
  let (actual_text, panic_msg) = run_capturing(&format!("testsearchv8-g{}", g), f);
  let secs = t0.elapsed().as_secs_f64();
  if let Ok(dir) = std::env::var("KATAGO_SEARCH_TEST_DUMP") {
    let p = std::path::Path::new(&dir).join(format!("testsearchv8-g{}.actual.txt", g));
    std::fs::write(&p, &actual_text).unwrap();
    eprintln!("wrote {}", p.display());
  }
  let actual = split_sections(&actual_text, &[]);
  let (report, num_bad, num_compared) = compare_all(&name, &expected, &actual, &policy_of, ALLOWLIST, panic_msg.as_deref());
  eprintln!("{}", report);
  eprintln!("{}: {} sections compared, run took {:.1} s", name, num_compared, secs);
  assert!(num_compared >= 1, "no section compared");
  assert!(panic_msg.is_none(), "the transcription panicked: {}", panic_msg.unwrap_or_default());
  assert_eq!(num_bad, 0, "{} sections diverged from the Eigen rerun and are not allowlisted", num_bad);
}

#[test]
fn v8_group0_size9() {
  run_group(0, group0);
}

#[test]
fn v8_group1_random_sym() {
  run_group(1, group1);
}

#[test]
fn v8_group2_v8_tests() {
  run_group(2, group2);
}

// TODO(port): heavy, ~15 min release (8000-visit and 2500-visit searches), ~2x that in debug; run in release (`cargo test --release -p katago-nn-burn --test testsearchv8`).
#[test]
#[cfg_attr(debug_assertions, ignore = "TODO(port): heavy, release only (just search-tests)")]
fn v8_group3_more_v8_tests() {
  run_group(3, group3);
}

#[test]
fn v8_group4_more_v8_tests2() {
  run_group(4, group4);
}

#[test]
fn v8_group5_randomized_nn_evals() {
  run_group(5, group5);
}

// TODO(port): heavy, ~6 min release (40 virtual threads, 16000 + 8000 visits), ~2x that in debug; run in release (`cargo test --release -p katago-nn-burn --test testsearchv8`).
#[test]
#[cfg_attr(debug_assertions, ignore = "TODO(port): heavy, release only (just search-tests)")]
fn v8_group6_multithread() {
  run_group(6, group6);
}
