// Ported from KataGo cpp/game/graphhash.cpp, cpp/game/graphhash.h @ fd0723fdbc0e9d82cf269c9630af8c27c57c07c4

use crate::core::hash::Hash;
use crate::core::hash::Hash128;
use crate::game::board::*;
use crate::game::boardhistory::BoardHistory;
use crate::testAssert;

/// upstream: namespace GraphHash (graphhash.h)
pub struct GraphHash;

impl GraphHash {
  /// upstream: GraphHash::getStateHash (graphhash.cpp)
  //Hash taking into account all state relevant for move legality, the rules, and some immediate other info like effect of passing.
  //Does NOT take into account more complex history beyond immediate ko and superko bans.
  pub fn get_state_hash(hist: &BoardHistory, next_player: Player, draw_equivalent_wins_for_white: f64) -> Hash128 {
    let board: &Board = hist.get_recent_board(0);
    let mut hash: Hash128 = BoardHistory::get_situation_rules_and_ko_hash(board, hist, next_player, draw_equivalent_wins_for_white);

    // Fold in whether a pass ends this phase
    let pass_ends_phase: bool = hist.pass_would_end_phase(board, next_player);
    if pass_ends_phase {
      hash ^= Board::ZOBRIST_PASS_ENDS_PHASE;
    }
    // Fold in whether the game is over or not
    if hist.is_game_finished {
      hash ^= Board::ZOBRIST_GAME_IS_OVER;
    }

    // Fold in consecutive pass count. Probably usually redundant with history tracking. Use some standard LCG constants.
    const CONSECPASS_MULT0: u64 = 2862933555777941757u64;
    const CONSECPASS_MULT1: u64 = 3202034522624059733u64;
    hash.hash0 = hash.hash0.wrapping_add(CONSECPASS_MULT0.wrapping_mul(hist.consecutive_ending_passes as u64));
    hash.hash1 = hash.hash1.wrapping_add(CONSECPASS_MULT1.wrapping_mul(hist.consecutive_ending_passes as u64));
    hash
  }

  /// upstream: GraphHash::getGraphHash (graphhash.cpp)
  //Call this AFTER making a move, to update a hash suitable for superko-safe tranpositions, given the previous graph hash.
  //Will guard against cycles up to repBound in size and possibly some slightly larger cycles.
  pub fn get_graph_hash(prev_graph_hash: Hash128, hist: &BoardHistory, next_player: Player, rep_bound: i32, draw_equivalent_wins_for_white: f64) -> Hash128 {
    let board: &Board = hist.get_recent_board(0);
    let prev_move_loc: Loc =
      if hist.move_history.is_empty() { Board::NULL_LOC } else { hist.move_history[hist.move_history.len() - 1].loc };
    if prev_move_loc == Board::NULL_LOC || board.simple_repetition_bound_gt(prev_move_loc, rep_bound) {
      GraphHash::get_state_hash(hist, next_player, draw_equivalent_wins_for_white)
    } else {
      let mut new_hash: Hash128 = prev_graph_hash;
      new_hash.hash0 = Hash::split_mix64(new_hash.hash0 ^ new_hash.hash1);
      new_hash.hash1 = Hash::nasam(new_hash.hash1).wrapping_add(new_hash.hash0);
      let state_hash: Hash128 = GraphHash::get_state_hash(hist, next_player, draw_equivalent_wins_for_white);
      new_hash.hash0 = new_hash.hash0.wrapping_add(state_hash.hash0);
      new_hash.hash1 = new_hash.hash1.wrapping_add(state_hash.hash1);
      new_hash
    }
  }

  /// upstream: GraphHash::getGraphHashFromScratch (graphhash.cpp)
  //Compute graph hash from scratch by replaying the whole history.
  pub fn get_graph_hash_from_scratch(hist_orig: &BoardHistory, next_player: Player, rep_bound: i32, draw_equivalent_wins_for_white: f64) -> Hash128 {
    let mut hist: BoardHistory = hist_orig.copy_to_initial();
    let mut board: Board = hist.get_recent_board(0).clone();
    let mut graph_hash: Hash128 = Hash128::default();

    for i in 0..hist_orig.move_history.len() {
      graph_hash = GraphHash::get_graph_hash(graph_hash, &hist, hist_orig.move_history[i].pla, rep_bound, draw_equivalent_wins_for_white);
      let suc: bool = hist.make_board_move_tolerant_prevent_encore(
        &mut board,
        hist_orig.move_history[i].loc,
        hist_orig.move_history[i].pla,
        hist_orig.prevent_encore_history[i],
      );
      testAssert!(suc);
    }
    testAssert!(
      BoardHistory::get_situation_rules_and_ko_hash(&board, &hist, next_player, draw_equivalent_wins_for_white)
        == BoardHistory::get_situation_rules_and_ko_hash(hist_orig.get_recent_board(0), hist_orig, next_player, draw_equivalent_wins_for_white)
    );

    graph_hash = GraphHash::get_graph_hash(graph_hash, &hist, next_player, rep_bound, draw_equivalent_wins_for_white);
    graph_hash
  }
}
