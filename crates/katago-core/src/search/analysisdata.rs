// Ported from KataGo cpp/search/analysisdata.cpp, cpp/search/analysisdata.h @ ba938676d7f42d70950b3a535af2466fb642008c

use crate::game::board::get_opp;
use crate::game::board::Board;
use crate::game::board::Loc;
use crate::game::board::Location;
use crate::game::board::Player;
use crate::game::boardhistory::BoardHistory;
use crate::port::node_arena::NodeIdx;
use crate::testAssert;

/// upstream: AnalysisData (analysisdata.h)
/// PORT NOTE: the copy constructor and copy assignment are `Clone` (they copy every field); the
/// move constructor and move assignment are Rust moves.
#[derive(Clone, Debug)]
pub struct AnalysisData {
  //Utilities and scores should all be from white's perspective
  pub move_: Loc,
  pub num_visits: i64,
  pub play_selection_value: f64, //Similar units to visits, but might have LCB adjustments
  pub lcb: f64,                  //In units of utility
  pub radius: f64,               //In units of utility
  pub utility: f64,              //From -1 to 1 or -1.25 to -1.25 or other similar bounds, depending on score utility
  pub result_utility: f64,       //Utility from winloss result
  pub score_utility: f64,        //Utility from score. Summing with resultUtility gives utility.
  pub win_loss_value: f64,       //From -1 to 1
  pub no_result_value: f64,      //From 0 to 1
  pub policy_prior: f64,         //From 0 to 1
  pub score_mean: f64,           //In units of points
  pub score_stdev: f64,          //In units of points
  pub lead: f64,                 //In units of points
  pub ess: f64,                  //Effective sample size taking into account weighting, could be somewhat smaller than visits
  pub weight_factor: f64,        //Due to child value weighting
  pub weight_sum: f64,           //Internal value that is used instead of visits for everything
  pub weight_sq_sum: f64,        //Sum of squares
  pub utility_sq_avg: f64,
  pub score_mean_sq_avg: f64,
  pub child_visits: i64,      //Visits of the underlying child node
  pub child_weight_sum: f64,  //Weight sum of the underlying child node
  pub order: i32,             //Preference order of the moves, 0 is best
  pub is_symmetry_of: Loc,    //If not Board::NULL_LOC, this move is a duplicate analysis data reflected from isSymmetryOf
  pub symmetry: i32,          //The symmetry applied to isSymmetryOf to get move, or 0.
  pub pv: Vec<Loc>,
  pub pv_visits: Vec<i64>,
  pub pv_edge_visits: Vec<i64>,

  /// PORT NOTE (OWNERSHIP.tsv): `const SearchNode*` -> arena index; resolve with
  /// `search.node_arena.get(idx)`.
  pub node: Option<NodeIdx>, //ONLY valid so long as search is not cleared
}

/// Builds a boxed copy in a small frame (Board ~4.7 KB, BoardHistory ~34 KB; PORTING.md §13 wasm stack).
fn boxed_copy<T: Clone>(t: &T) -> Box<T> {
  Box::new(t.clone())
}

impl AnalysisData {
  /// upstream: AnalysisData::AnalysisData (analysisdata.cpp)
  /// PORT NOTE: the field `move` is `move_` (`move` is a Rust keyword).
  pub fn new() -> AnalysisData {
    AnalysisData {
      move_: Board::NULL_LOC,
      num_visits: 0,
      play_selection_value: 0.0,
      lcb: 0.0,
      radius: 0.0,
      utility: 0.0,
      result_utility: 0.0,
      score_utility: 0.0,
      win_loss_value: 0.0,
      no_result_value: 0.0,
      policy_prior: 0.0,
      score_mean: 0.0,
      score_stdev: 0.0,
      lead: 0.0,
      ess: 0.0,
      weight_factor: 0.0,
      weight_sum: 0.0,
      weight_sq_sum: 0.0,
      utility_sq_avg: 0.0,
      score_mean_sq_avg: 0.0,
      child_visits: 0,
      child_weight_sum: 0.0,
      order: 0,
      is_symmetry_of: Board::NULL_LOC,
      symmetry: 0,
      pv: Vec::new(),
      pv_visits: Vec::new(),
      pv_edge_visits: Vec::new(),
      node: None,
    }
  }

  /// upstream: AnalysisData::pvContainsPass (analysisdata.cpp)
  pub fn pv_contains_pass(&self) -> bool {
    for i in 0..self.pv.len() {
      if self.pv[i] == Board::PASS_LOC {
        return true;
      }
    }
    false
  }

  /// upstream: AnalysisData::writePV (analysisdata.cpp)
  pub fn write_pv(&self, out: &mut dyn std::fmt::Write, board: &Board) {
    for j in 0..self.pv.len() {
      if j > 0 {
        write!(out, " ").unwrap();
      }
      write!(out, "{}", Location::to_string(self.pv[j], board)).unwrap();
    }
  }

  /// upstream: AnalysisData::writePVVisits (analysisdata.cpp)
  pub fn write_pv_visits(&self, out: &mut dyn std::fmt::Write) {
    for j in 0..self.pv_visits.len() {
      if j > 0 {
        write!(out, " ").unwrap();
      }
      write!(out, "{}", self.pv_visits[j]).unwrap();
    }
  }

  /// upstream: AnalysisData::writePVEdgeVisits (analysisdata.cpp)
  pub fn write_pv_edge_visits(&self, out: &mut dyn std::fmt::Write) {
    for j in 0..self.pv_edge_visits.len() {
      if j > 0 {
        write!(out, " ").unwrap();
      }
      write!(out, "{}", self.pv_edge_visits[j]).unwrap();
    }
  }

  /// upstream: AnalysisData::getPVLenUpToPhaseEnd (analysisdata.cpp)
  pub fn get_pv_len_up_to_phase_end(&self, initial_board: &Board, initial_hist: &BoardHistory, initial_pla: Player) -> i32 {
    let mut board: Box<Board> = boxed_copy(initial_board);
    let mut hist: Box<BoardHistory> = boxed_copy(initial_hist);
    let mut next_pla: Player = initial_pla;
    let mut j: i32 = 0;
    while (j as usize) < self.pv.len() {
      hist.make_board_move_assume_legal(&mut board, self.pv[j as usize], next_pla, None, false);
      next_pla = get_opp(next_pla);
      if hist.encore_phase != initial_hist.encore_phase {
        break;
      }
      j += 1;
    }
    j
  }

  /// upstream: AnalysisData::writePVUpToPhaseEnd (analysisdata.cpp)
  pub fn write_pv_up_to_phase_end(&self, out: &mut dyn std::fmt::Write, initial_board: &Board, initial_hist: &BoardHistory, initial_pla: Player) {
    let mut board: Box<Board> = boxed_copy(initial_board);
    let mut hist: Box<BoardHistory> = boxed_copy(initial_hist);
    let mut next_pla: Player = initial_pla;
    for j in 0..self.pv.len() {
      if j > 0 {
        write!(out, " ").unwrap();
      }
      write!(out, "{}", Location::to_string(self.pv[j], &board)).unwrap();

      hist.make_board_move_assume_legal(&mut board, self.pv[j], next_pla, None, false);
      next_pla = get_opp(next_pla);
      if hist.encore_phase != initial_hist.encore_phase {
        break;
      }
    }
  }

  /// upstream: AnalysisData::writePVVisitsUpToPhaseEnd (analysisdata.cpp)
  pub fn write_pv_visits_up_to_phase_end(&self, out: &mut dyn std::fmt::Write, initial_board: &Board, initial_hist: &BoardHistory, initial_pla: Player) {
    let mut board: Box<Board> = boxed_copy(initial_board);
    let mut hist: Box<BoardHistory> = boxed_copy(initial_hist);
    let mut next_pla: Player = initial_pla;
    testAssert!(self.pv.len() == self.pv_visits.len());
    for j in 0..self.pv.len() {
      if j > 0 {
        write!(out, " ").unwrap();
      }
      write!(out, "{}", self.pv_visits[j]).unwrap();

      hist.make_board_move_assume_legal(&mut board, self.pv[j], next_pla, None, false);
      next_pla = get_opp(next_pla);
      if hist.encore_phase != initial_hist.encore_phase {
        break;
      }
    }
  }

  /// upstream: AnalysisData::writePVEdgeVisitsUpToPhaseEnd (analysisdata.cpp)
  pub fn write_pv_edge_visits_up_to_phase_end(
    &self,
    out: &mut dyn std::fmt::Write,
    initial_board: &Board,
    initial_hist: &BoardHistory,
    initial_pla: Player,
  ) {
    let mut board: Box<Board> = boxed_copy(initial_board);
    let mut hist: Box<BoardHistory> = boxed_copy(initial_hist);
    let mut next_pla: Player = initial_pla;
    testAssert!(self.pv.len() == self.pv_edge_visits.len());
    for j in 0..self.pv.len() {
      if j > 0 {
        write!(out, " ").unwrap();
      }
      write!(out, "{}", self.pv_edge_visits[j]).unwrap();

      hist.make_board_move_assume_legal(&mut board, self.pv[j], next_pla, None, false);
      next_pla = get_opp(next_pla);
      if hist.encore_phase != initial_hist.encore_phase {
        break;
      }
    }
  }
}

// ~AnalysisData() is empty: omitted.

/// upstream: operator<(const AnalysisData&, const AnalysisData&) (analysisdata.cpp)
pub fn operator_lt(a0: &AnalysisData, a1: &AnalysisData) -> bool {
  // Sort all 0-visit moves to the end.
  if a0.num_visits > 0 && a1.num_visits == 0 {
    return true;
  }
  if a1.num_visits > 0 && a0.num_visits == 0 {
    return false;
  }
  // Then sort by play selection value, the normal value for chosing moves to play.
  if a0.play_selection_value > a1.play_selection_value {
    return true;
  }
  if a0.play_selection_value < a1.play_selection_value {
    return false;
  }
  // Then by visits
  if a0.num_visits > a1.num_visits {
    return true;
  }
  if a0.num_visits < a1.num_visits {
    return false;
  }
  // Then just by raw policy
  else {
    return a0.policy_prior > a1.policy_prior;
  }
}
