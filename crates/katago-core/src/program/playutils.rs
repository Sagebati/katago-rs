// Ported from KataGo cpp/program/playutils.cpp, cpp/program/playutils.h @ ba938676d7f42d70950b3a535af2466fb642008c
//
// M3 subset (port-manifest.toml): getHackedLCBForWinrate (search/searchresults.cpp), printGenmoveLog
// (command/analysis.cpp), and what the search tests use: getNoiselessParams, computeOwnership,
// computeAnticipatedStatusesSimple, computeAnticipatedStatusesWithOwnership, maybeCleanupBeforePass,
// maybeFriendlyPass, getFullSymmetryNNOutput. Everything else in playutils.cpp/.h is not ported yet.
// The functions that run a search or an NN evaluation are async (PORTING.md §11).

use std::rc::Rc;

use crate::core::commontypes::enabled_t;
use crate::core::global::KataError;
use crate::game::board::get_opp;
use crate::game::board::Board;
use crate::game::board::Color;
use crate::game::board::Loc;
use crate::game::board::Location;
use crate::game::board::Player;
use crate::game::board::C_BLACK;
use crate::game::board::C_EMPTY;
use crate::game::board::C_WHITE;
use crate::game::board::P_BLACK;
use crate::game::board::P_WHITE;
use crate::game::boardhistory::BoardHistory;
use crate::neuralnet::nneval::NNEvaluator;
use crate::neuralnet::nneval::NNResultBuf;
use crate::neuralnet::nninputs::MiscNNInputParams;
use crate::neuralnet::nninputs::NNOutput;
use crate::neuralnet::nninputs::NNPos;
use crate::neuralnet::nninputs::SymmetryHelpers;
use crate::neuralnet::sgfmetadata::SGFMetadata;
use crate::port::cpp;
use crate::port::fmt::cpp_ostream_f64;
use crate::search::analysisdata::AnalysisData;
use crate::search::asyncbot::AsyncBot;
use crate::search::reportedsearchvalues::ReportedSearchValues;
use crate::search::search::Search;
use crate::search::searchnode::SearchNode;
use crate::search::searchparams::SearchParams;
use crate::search::searchprint::PrintTreeOptions;
use crate::testAssert;

/// upstream: namespace PlayUtils (playutils.h)
pub struct PlayUtils;

impl PlayUtils {
  /// upstream: PlayUtils::getHackedLCBForWinrate (playutils.cpp)
  pub fn get_hacked_lcb_for_winrate(search: &Search, data: &AnalysisData, pla: Player) -> f64 {
    let winrate: f64 = 0.5 * (1.0 + data.win_loss_value);
    //Super hacky - in KataGo, lcb is on utility (i.e. also weighting score), not winrate, but if you're using
    //lz-analyze you probably don't know about utility and expect LCB to be about winrate. So we apply the LCB
    //radius to the winrate in order to get something reasonable to display, and also scale it proportionally
    //by how much winrate is supposed to matter relative to score.
    let mut radius_scale_hack_factor: f64 = search.search_params.win_loss_utility_factor
      / (search.search_params.win_loss_utility_factor
        + search.search_params.static_score_utility_factor
        + search.search_params.dynamic_score_utility_factor
        + 1.0e-20); //avoid divide by 0
    //Also another factor of 0.5 because winrate goes from only 0 to 1 instead of -1 to 1 when it's part of utility
    radius_scale_hack_factor *= 0.5;
    let lcb: f64 = if pla == P_WHITE {
      winrate - data.radius * radius_scale_hack_factor
    } else {
      winrate + data.radius * radius_scale_hack_factor
    };
    lcb
  }

  /// upstream: PlayUtils::computeOwnership (playutils.cpp)
  /// PORT NOTE: async (runs a search); throws -> Result.
  pub async fn compute_ownership(bot: &mut Search, board: &Board, hist: &BoardHistory, pla: Player, num_visits: i64) -> Result<Vec<f64>, KataError> {
    testAssert!(num_visits > 0);
    let old_always_include_owner_map: bool = bot.always_include_owner_map;
    bot.set_always_include_owner_map(true);

    let old_params: SearchParams = bot.search_params.clone();
    let mut new_params: SearchParams = get_noiseless_params(&old_params, num_visits);
    new_params.playout_doubling_advantage_pla = C_EMPTY;
    new_params.playout_doubling_advantage = 0.0;
    //Make sure the search is always from a state where the game isn't believed to end with another pass
    new_params.conservative_pass = true;

    bot.set_params(&new_params);
    bot.set_position(pla, board, hist)?;
    bot.run_whole_search(pla).await?;

    let ownerships: Vec<f64> = bot.get_average_tree_ownership(None)?;

    bot.set_params(&old_params);
    bot.set_always_include_owner_map(old_always_include_owner_map);
    bot.clear_search();

    Ok(ownerships)
  }

  /// upstream: PlayUtils::computeAnticipatedStatusesSimple (playutils.cpp)
  //Tromp-taylor-like scoring, except recognizes pass-dead stones.
  pub fn compute_anticipated_statuses_simple(board: &Board, hist: &BoardHistory) -> Vec<bool> {
    let mut is_alive: Vec<bool> = vec![false; Board::MAX_ARR_SIZE as usize];

    //Treat all stones as alive under a no result
    if hist.is_game_finished && hist.is_no_result {
      for y in 0..board.y_size {
        for x in 0..board.x_size {
          let loc: Loc = Location::get_loc(x, y, board.x_size);
          if board.colors[loc as usize] != C_EMPTY {
            is_alive[loc as usize] = true;
          }
        }
      }
    }
    //Else use Tromp-taylorlike scoring, except recognizing pass-dead stones.
    else {
      let mut area: [Color; Board::MAX_ARR_SIZE as usize] = [C_EMPTY; Board::MAX_ARR_SIZE as usize];
      let mut hist_copy: BoardHistory = hist.clone();
      hist_copy.end_and_score_game_now_area(board, &mut area);
      for y in 0..board.y_size {
        for x in 0..board.x_size {
          let loc: Loc = Location::get_loc(x, y, board.x_size);
          if board.colors[loc as usize] != C_EMPTY {
            is_alive[loc as usize] = board.colors[loc as usize] == area[loc as usize];
          }
        }
      }
    }
    is_alive
  }

  /// upstream: PlayUtils::computeAnticipatedStatusesWithOwnership (playutils.cpp)
  //Always non-tromp-taylorlike in the main phase of the game, this is the ownership that users would want.
  /// PORT NOTE: async (computeOwnership); throws -> Result.
  pub async fn compute_anticipated_statuses_with_ownership(
    bot: &mut Search,
    board: &Board,
    hist: &BoardHistory,
    pla: Player,
    num_visits: i64,
    ownerships_buf: &mut Vec<f64>,
  ) -> Result<Vec<bool>, KataError> {
    let mut is_alive: Vec<bool> = vec![false; Board::MAX_ARR_SIZE as usize];
    let mut solved: [bool; Board::MAX_ARR_SIZE as usize] = [false; Board::MAX_ARR_SIZE as usize];
    for i in 0..Board::MAX_ARR_SIZE {
      is_alive[i as usize] = false;
      solved[i as usize] = false;
    }

    *ownerships_buf = PlayUtils::compute_ownership(bot, board, hist, pla, num_visits).await?;
    let ownerships: &Vec<f64> = ownerships_buf;
    let nn_x_len: i32 = bot.nn_x_len;
    let nn_y_len: i32 = bot.nn_y_len;

    //Heuristic:
    //Stones are considered dead if their average ownership is less than 0.2 equity in their own color,
    //or if the worst equity in the chain is less than -0.6 equity in their color.
    const AVG_THRESHOLD_FOR_LIFE: f64 = 0.2;
    const WORST_THRESHOLD_FOR_LIFE: f64 = -0.6;

    for y in 0..board.y_size {
      for x in 0..board.x_size {
        let loc: Loc = Location::get_loc(x, y, board.x_size);
        if solved[loc as usize] {
          continue;
        }

        if board.colors[loc as usize] == P_WHITE || board.colors[loc as usize] == P_BLACK {
          let mut pos: i32 = NNPos::loc_to_pos(loc, board.x_size, nn_x_len, nn_y_len);
          let mut min_ownership: f64 = ownerships[pos as usize];
          let mut max_ownership: f64 = ownerships[pos as usize];
          let mut ownership_sum: f64 = 0.0;
          let mut count: f64 = 0.0;

          //Run through the whole chain
          let mut cur: Loc = loc;
          loop {
            pos = NNPos::loc_to_pos(cur, board.x_size, nn_x_len, nn_y_len);
            min_ownership = cpp::cpp_min(ownerships[pos as usize], min_ownership);
            max_ownership = cpp::cpp_max(ownerships[pos as usize], max_ownership);
            ownership_sum += ownerships[pos as usize];
            count += 1.0;
            cur = board.next_in_chain[cur as usize];
            if !(cur != loc) {
              break;
            }
          }

          let avg_ownership: f64 = ownership_sum / count;
          let alive: bool;
          if board.colors[loc as usize] == P_WHITE {
            alive = avg_ownership > AVG_THRESHOLD_FOR_LIFE && min_ownership > WORST_THRESHOLD_FOR_LIFE;
          } else {
            alive = avg_ownership < -AVG_THRESHOLD_FOR_LIFE && max_ownership < -WORST_THRESHOLD_FOR_LIFE;
          }

          //Run through the whole chain again, recording the result
          cur = loc;
          loop {
            is_alive[cur as usize] = alive;
            solved[cur as usize] = true;
            cur = board.next_in_chain[cur as usize];
            if !(cur != loc) {
              break;
            }
          }
        }
      }
    }
    Ok(is_alive)
  }

  /// upstream: PlayUtils::printGenmoveLog (playutils.cpp)
  pub fn print_genmove_log(
    out: &mut dyn std::fmt::Write,
    search: &Search,
    nn_eval: &NNEvaluator,
    move_loc: Loc,
    time_taken: f64,
    perspective: Player,
    log_search_info_for_chosen_move: bool,
  ) -> Result<(), KataError> {
    let board: &Board = search.get_root_board();
    Board::print_board(out, board, move_loc, Some(&search.get_root_hist().move_history));
    write!(out, "{}\n", search.get_root_hist().rules).unwrap();
    if !time_taken.is_nan() {
      write!(out, "Time taken: {}\n", cpp_ostream_f64(time_taken)).unwrap();
    }
    write!(out, "Root visits: {}\n", search.get_root_visits()).unwrap();
    write!(out, "New playouts: {}\n", search.last_search_num_playouts).unwrap();
    writeln!(out, "NN rows: {}", nn_eval.num_rows_processed()).unwrap();
    writeln!(out, "NN batches: {}", nn_eval.num_batches_processed()).unwrap();
    writeln!(out, "NN avg batch size: {}", cpp_ostream_f64(nn_eval.average_processed_batch_size())).unwrap();
    if search.search_params.playout_doubling_advantage != 0.0 {
      writeln!(
        out,
        "PlayoutDoublingAdvantage: {}",
        cpp_ostream_f64(if search.get_root_pla() == get_opp(search.get_playout_doubling_advantage_pla()) {
          -search.search_params.playout_doubling_advantage
        } else {
          search.search_params.playout_doubling_advantage
        })
      )
      .unwrap();
    }
    write!(out, "PV: ").unwrap();
    let root_node: Option<Rc<SearchNode>> = search.get_root_node();
    search.print_pv(out, root_node.as_deref(), 25);
    write!(out, "\n").unwrap();
    write!(out, "Tree:\n").unwrap();
    if log_search_info_for_chosen_move && move_loc != Board::NULL_LOC {
      search.print_tree(
        out,
        root_node.as_deref(),
        &PrintTreeOptions::new().max_depth(1).max_children_to_show(10).also_branch(board, &Location::to_string(move_loc, board))?,
        perspective,
      );
    } else {
      search.print_tree(out, root_node.as_deref(), &PrintTreeOptions::new().max_depth(1).max_children_to_show(10), perspective);
    }
    Ok(())
  }

  /// upstream: PlayUtils::maybeCleanupBeforePass (playutils.cpp)
  pub fn maybe_cleanup_before_pass(cleanup_before_pass: enabled_t, friendly_pass: enabled_t, pla: Player, mut move_loc: Loc, bot: &AsyncBot) -> Loc {
    if friendly_pass == enabled_t::True {
      return move_loc;
    }
    let hist: &BoardHistory = bot.get_root_hist();
    testAssert!(pla == hist.presumed_next_move_pla);
    let rules = &hist.rules;
    let do_cleanup_before_pass: bool = if cleanup_before_pass == enabled_t::True {
      true
    } else if cleanup_before_pass == enabled_t::False {
      false
    } else {
      rules.friendly_pass_ok == false && rules.scoring_rule == crate::game::rules::Rules::SCORING_AREA
    };
    if do_cleanup_before_pass && move_loc == Board::PASS_LOC && hist.is_final_phase() && !hist.has_button {
      let board: &Board = bot.get_root_board();
      let safe_area: &[Color] = &bot.get_search().root_safe_area[..];
      // PORT NOTE: `testAssert(safeArea != NULL)`: rootSafeArea is an owned array here, never null.
      //Scan the board for any spot that is adjacent to an opponent group that is part of our pass-alive territory.
      for y in 0..board.y_size {
        for x in 0..board.x_size {
          let other_loc: Loc = Location::get_loc(x, y, board.x_size);
          if move_loc == Board::PASS_LOC
            && board.colors[other_loc as usize] == C_EMPTY
            && safe_area[other_loc as usize] == pla
            && board.is_adjacent_to_pla(other_loc, get_opp(pla))
            && hist.is_legal(board, other_loc, pla)
          {
            move_loc = other_loc;
          }
        }
      }
    }
    move_loc
  }

  /// upstream: PlayUtils::maybeFriendlyPass (playutils.cpp)
  /// PORT NOTE: async (runs searches); throws -> Result.
  pub async fn maybe_friendly_pass(
    cleanup_before_pass: enabled_t,
    friendly_pass: enabled_t,
    pla: Player,
    mut move_loc: Loc,
    bot: &mut Search,
    num_visits: i64,
  ) -> Result<Loc, KataError> {
    if cleanup_before_pass == enabled_t::True {
      return Ok(move_loc);
    }
    let should_proceed: bool;
    {
      let board: &Board = bot.get_root_board();
      let hist: &BoardHistory = bot.get_root_hist();
      let rules = &hist.rules;
      let do_friendly_pass: bool = if friendly_pass == enabled_t::True {
        true
      } else if friendly_pass == enabled_t::False {
        false
      } else {
        rules.friendly_pass_ok == true && rules.scoring_rule == crate::game::rules::Rules::SCORING_AREA
      };
      should_proceed = do_friendly_pass
        && move_loc != Board::PASS_LOC
        && rules.scoring_rule == crate::game::rules::Rules::SCORING_AREA
        && hist.is_final_phase()
        && !hist.has_button
        && hist.pass_would_end_phase(board, pla)
        && hist.move_history.len() > 0
        && hist.move_history[hist.move_history.len() - 1].pla == get_opp(pla)
        && hist.move_history[hist.move_history.len() - 1].loc == Board::PASS_LOC;
    }
    if !should_proceed {
      return Ok(move_loc);
    }

    //Make absolutely sure we can restore the bot's old state
    let old_pla: Player = bot.get_root_pla();
    let old_board: Board = bot.get_root_board().clone();
    let old_hist: BoardHistory = bot.get_root_hist().clone();

    let board: Board = bot.get_root_board().clone();
    let hist: BoardHistory = bot.get_root_hist().clone();
    testAssert!(old_pla == pla);

    if !hist.is_legal(&board, move_loc, pla) {
      return Err(KataError::StringError("PlayUtils::maybeFriendlyPass called on illegal move ".to_string() + &Location::to_string(move_loc, &board)));
    }

    let mut ownerships: Vec<f64> = Vec::new();
    let is_alive: Vec<bool> = PlayUtils::compute_anticipated_statuses_with_ownership(bot, &board, &hist, pla, num_visits, &mut ownerships).await?;

    //Delete all dead groups from board
    let mut clean_board: Board = board.clone();
    for y in 0..board.y_size {
      for x in 0..board.x_size {
        let loc: Loc = Location::get_loc(x, y, board.x_size);
        if board.colors[loc as usize] != C_EMPTY && !is_alive[loc as usize] {
          clean_board.set_stone(loc, C_EMPTY);
        }
      }
    }

    //Now, check if naive floodfilling agrees that all highly-owned spots are surrounded (finished borders)
    let mut area: [Color; Board::MAX_ARR_SIZE as usize] = [C_EMPTY; Board::MAX_ARR_SIZE as usize];
    {
      let non_pass_alive_stones: bool = true;
      let safe_big_territories: bool = true;
      let unsafe_big_territories: bool = true;
      let is_multi_stone_suicide_legal: bool = hist.rules.multi_stone_suicide_legal;
      clean_board.calculate_area(&mut area, non_pass_alive_stones, safe_big_territories, unsafe_big_territories, is_multi_stone_suicide_legal);
    }
    const HIGH_OWNERSHIP_THRESHOLD: f64 = 0.75;
    let nn_x_len: i32 = bot.nn_x_len;
    let nn_y_len: i32 = bot.nn_y_len;
    let mut found_unsurrounded_spot: bool = false;
    let mut y: i32 = 0;
    while y < board.y_size && !found_unsurrounded_spot {
      let mut x: i32 = 0;
      while x < board.x_size && !found_unsurrounded_spot {
        let loc: Loc = Location::get_loc(x, y, board.x_size);
        let pos: i32 = NNPos::loc_to_pos(loc, board.x_size, nn_x_len, nn_y_len);
        if ownerships[pos as usize] > HIGH_OWNERSHIP_THRESHOLD && area[loc as usize] != C_WHITE {
          found_unsurrounded_spot = true;
        }
        if ownerships[pos as usize] < -HIGH_OWNERSHIP_THRESHOLD && area[loc as usize] != C_BLACK {
          found_unsurrounded_spot = true;
        }
        x += 1;
      }
      y += 1;
    }

    //Also, the location that the bot currently wants to move must not be adjacent to an opponent's living stone or a self stone in atari by living stones.
    //This means it's a safe dame or protective fill the bot is trying. It's not a very exhaustive check (since there are also prep connect moves)
    //and maybe has some false positives but it's a free extra check.
    if !found_unsurrounded_spot && !clean_board.is_adjacent_to_pla(move_loc, get_opp(pla)) && !clean_board.would_be_capture(move_loc, get_opp(pla)) {
      //And also, the bot must rate passing as not significantly worse than playing its move
      let old_params: SearchParams = bot.search_params.clone();
      let mut new_params: SearchParams = get_noiseless_params(&old_params, num_visits);
      new_params.playout_doubling_advantage_pla = C_EMPTY;
      new_params.playout_doubling_advantage = 0.0;
      //Conservative pass makes sure we evaluate the opponent's turn as one in which the game is neither finished nor the opponent
      //will be able to end the game via passing.
      new_params.conservative_pass = true;
      bot.set_params(&new_params);

      let values_after_pass: ReportedSearchValues;
      let values_after_move: ReportedSearchValues;

      {
        let mut board_after_pass: Board = board.clone();
        let mut hist_after_pass: BoardHistory = hist.clone();
        hist_after_pass.make_board_move_assume_legal(&mut board_after_pass, Board::PASS_LOC, pla, None, false);
        let pla_after_pass: Player = get_opp(pla);
        bot.set_position(pla_after_pass, &board_after_pass, &hist_after_pass)?;
        bot.run_whole_search(pla_after_pass).await?;
        values_after_pass = bot.get_root_values_require_success()?;
      }
      {
        let mut board_after_move: Board = board.clone();
        let mut hist_after_move: BoardHistory = hist.clone();
        hist_after_move.make_board_move_assume_legal(&mut board_after_move, move_loc, pla, None, false);
        let pla_after_move: Player = get_opp(pla);
        bot.set_position(pla_after_move, &board_after_move, &hist_after_move)?;
        bot.run_whole_search(pla_after_move).await?;
        values_after_move = bot.get_root_values_require_success()?;
      }

      bot.set_params(&old_params);
      bot.clear_search();

      if pla == P_WHITE
        && values_after_pass.utility > values_after_move.utility - 0.1
        && values_after_pass.expected_score > values_after_move.expected_score - 0.25
        && values_after_pass.lead > values_after_move.lead - 0.25
      {
        move_loc = Board::PASS_LOC;
      } else if pla == P_BLACK
        && values_after_pass.utility < values_after_move.utility + 0.1
        && values_after_pass.expected_score < values_after_move.expected_score + 0.25
        && values_after_pass.lead < values_after_move.lead + 0.25
      {
        move_loc = Board::PASS_LOC;
      }
    }
    //Restore
    bot.set_position(old_pla, &old_board, &old_hist)?;

    Ok(move_loc)
  }

  /// upstream: PlayUtils::getFullSymmetryNNOutput (playutils.cpp)
  /// PORT NOTE: async (NN evaluations); throws -> Result; `const SGFMetadata*` -> Option<&SGFMetadata>.
  pub async fn get_full_symmetry_nn_output(
    board: &Board,
    hist: &BoardHistory,
    pla: Player,
    include_owner_map: bool,
    sgf_meta: Option<&SGFMetadata>,
    nn_eval: &NNEvaluator,
  ) -> Result<Rc<NNOutput>, KataError> {
    let mut ptrs: Vec<Rc<NNOutput>> = Vec::new();
    for sym in 0..SymmetryHelpers::NUM_SYMMETRIES {
      let mut nn_input_params: MiscNNInputParams = MiscNNInputParams::new();
      nn_input_params.symmetry = sym;
      let buf: Rc<NNResultBuf> = Rc::new(NNResultBuf::new());
      let skip_cache: bool = true; //Always ignore cache so that we use the desired symmetry
      nn_eval.evaluate_sgf_meta(board, hist, pla, sgf_meta, &nn_input_params, &buf, skip_cache, include_owner_map).await?;
      let result: Rc<NNOutput> = buf.result.borrow_mut().take().expect("evaluate sets a result");
      ptrs.push(result);
    }
    let result: Rc<NNOutput> = Rc::new(NNOutput::new_others(&ptrs));
    Ok(result)
  }
}

/// upstream: getNoiselessParams (playutils.cpp) static
fn get_noiseless_params(old_params: &SearchParams, num_visits: i64) -> SearchParams {
  let mut new_params: SearchParams = old_params.clone();
  new_params.max_visits = num_visits;
  new_params.max_playouts = num_visits;
  new_params.root_noise_enabled = false;
  new_params.root_policy_temperature = 1.0;
  new_params.root_policy_temperature_early = 1.0;
  new_params.root_fpu_reduction_max = new_params.fpu_reduction_max;
  new_params.root_fpu_loss_prop = new_params.fpu_loss_prop;
  new_params.root_desired_per_child_visits_coeff = 0.0;
  new_params.root_num_symmetries_to_sample = 1;
  new_params.search_factor_after_one_pass = 1.0;
  new_params.search_factor_after_two_pass = 1.0;
  if new_params.num_threads as i64 > (num_visits + 7) / 8 {
    // PORT NOTE: in range because it is less than numThreads, an int.
    new_params.num_threads = ((num_visits + 7) / 8) as i32;
  }
  new_params
}
