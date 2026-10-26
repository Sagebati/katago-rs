// Ported from KataGo cpp/search/searchresults.cpp @ fd0723fdbc0e9d82cf269c9630af8c27c57c07c4
//
//-------------------------------------------------------------------------------------
//This file contains various functions for extracting stats and results from the search, choosing a move, etc
//-------------------------------------------------------------------------------------
//
// PORT NOTE (PORTING.md §4, OWNERSHIP.tsv): nodes are reached through `self.node_arena`; a
// `const SearchNode*` that may be NULL is `Option<&SearchNode>`, a node stored in a result
// (AnalysisData::node) is a NodeIdx, returned nodes are `Option<Rc<SearchNode>>`; pointer equality
// (`&node == rootNode`) is index equality. `std::unordered_set<const SearchNode*>` graph paths are
// `HashSet<NodeIdx>`. `double buf[NNPos::MAX_NN_POLICY_SIZE]` locals are Vecs (PERF(port): heap
// instead of stack, keeps the 1 MiB debug wasm stack free). JSON goes into a serde_json::Value
// (PORTING.md §9) that callers serialize with port::nlohmann::dump.

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::collections::HashSet;
use std::rc::Rc;

use serde_json::Value;

use crate::core::global::Global;
use crate::core::global::KataError;
use crate::core::hash::Hash128;
use crate::game::board::Board;
use crate::game::board::Loc;
use crate::game::board::Location;
use crate::game::board::Player;
use crate::game::board::PlayerIO;
use crate::game::board::P_BLACK;
use crate::game::board::P_WHITE;
use crate::game::boardhistory::BoardHistory;
use crate::neuralnet::nninputs::NNOutput;
use crate::neuralnet::nninputs::NNPos;
use crate::neuralnet::nninputs::ScoreValue;
use crate::neuralnet::nninputs::SymmetryHelpers;
use crate::port::cpp;
use crate::port::fmt::cpp_ostream_f64;
use crate::port::math;
use crate::port::node_arena::NodeIdx;
use crate::program::playutils::PlayUtils;
use crate::search::analysisdata::operator_lt;
use crate::search::analysisdata::AnalysisData;
use crate::search::reportedsearchvalues::ReportedSearchValues;
use crate::search::search::Search;
use crate::search::searchnode::MoreNodeStats;
use crate::search::searchnode::NodeStats;
use crate::search::searchnode::SearchChildPointer;
use crate::search::searchnode::SearchChildrenSizes;
use crate::search::searchnode::SearchNode;
use crate::search::searchnode::SearchNodeChildrenReference;
use crate::search::searchnode::SearchNodeState;
use crate::search::searchprint::PrintTreeOptions;
use crate::strprintf;
use crate::testAssert;

/// The policy array of a possibly-NULL nnOutput (`const float* policyProbs = ... : NULL`).
/// PORT NOTE: upstream dereferences it without a NULL check where a node with children or the
/// root must have an nnOutput; here that is an expect.
fn pp(policy_probs: Option<&[f32]>) -> &[f32] {
  policy_probs.expect("a node with children has an nnOutput (upstream dereferences policyProbs here)")
}

impl Search {
  /// upstream: Search::getRootVisits (searchresults.cpp)
  //Get the number of visits recorded for the root node
  pub fn get_root_visits(&self) -> i64 {
    let root_node: NodeIdx = match self.root_node {
      None => return 0,
      Some(r) => r,
    };
    let n: i64 = self.node_arena.get(root_node).stats.visits.get() /*acquire*/;
    n
  }

  /// upstream: Search::getPlaySelectionValues(vector<Loc>&, vector<double>&, double) (searchresults.cpp)
  //Get the vector of values (e.g. modified visit counts) used to select a move.
  //Does take into account chosenMoveSubtract but does NOT apply temperature.
  //If somehow the max value is less than scaleMaxToAtLeast, scale it to at least that value.
  //Always returns false in the case where no actual legal moves are found or there is no nnOutput or no root node.
  //If returning true, the is at least one loc and playSelectionValue.
  pub fn get_play_selection_values(&self, locs: &mut Vec<Loc>, play_selection_values: &mut Vec<f64>, scale_max_to_at_least: f64) -> bool {
    let root_node: Rc<SearchNode> = match self.root_node {
      None => {
        locs.clear();
        play_selection_values.clear();
        return false;
      }
      Some(r) => self.node_arena.get(r),
    };
    let allow_direct_policy_moves: bool = true;
    self.get_play_selection_values_node(&root_node, locs, play_selection_values, None, scale_max_to_at_least, allow_direct_policy_moves)
  }

  /// upstream: Search::getPlaySelectionValues(vector<Loc>&, vector<double>&, vector<double>*, double) (searchresults.cpp)
  pub fn get_play_selection_values_ret_visit_counts(
    &self,
    locs: &mut Vec<Loc>,
    play_selection_values: &mut Vec<f64>,
    ret_visit_counts: Option<&mut Vec<f64>>,
    scale_max_to_at_least: f64,
  ) -> bool {
    let root_node: Rc<SearchNode> = match self.root_node {
      None => {
        locs.clear();
        play_selection_values.clear();
        if let Some(ret_visit_counts) = ret_visit_counts {
          ret_visit_counts.clear();
        }
        return false;
      }
      Some(r) => self.node_arena.get(r),
    };
    let allow_direct_policy_moves: bool = true;
    self.get_play_selection_values_node(&root_node, locs, play_selection_values, ret_visit_counts, scale_max_to_at_least, allow_direct_policy_moves)
  }

  /// upstream: Search::getPlaySelectionValues(const SearchNode&, ..., bool allowDirectPolicyMoves) (searchresults.cpp)
  //Same, but works on a node within the search, not just the root
  /// PORT NOTE: name from PORTING.md Appendix B.
  pub fn get_play_selection_values_node(
    &self,
    node: &SearchNode,
    locs: &mut Vec<Loc>,
    play_selection_values: &mut Vec<f64>,
    ret_visit_counts: Option<&mut Vec<f64>>,
    scale_max_to_at_least: f64,
    allow_direct_policy_moves: bool,
  ) -> bool {
    let mut lcb_buf: Vec<f64> = vec![0.0; NNPos::MAX_NN_POLICY_SIZE as usize];
    let mut radius_buf: Vec<f64> = vec![0.0; NNPos::MAX_NN_POLICY_SIZE as usize];
    let result: bool = self.get_play_selection_values_node_lcb(
      node,
      locs,
      play_selection_values,
      ret_visit_counts,
      scale_max_to_at_least,
      allow_direct_policy_moves,
      false,
      false,
      Some(&mut lcb_buf),
      Some(&mut radius_buf),
    );
    result
  }

  /// upstream: Search::getPlaySelectionValues(const SearchNode&, ..., bool alwaysComputeLcb, bool neverUseLcb, double lcbBuf[], double radiusBuf[]) (searchresults.cpp)
  /// PORT NOTE: name from PORTING.md Appendix B. The buffers are `Option<&mut [f64]>` (length
  /// MAX_NN_POLICY_SIZE) because upstream passes NULL together with neverUseLcb = true.
  pub(crate) fn get_play_selection_values_node_lcb(
    &self,
    node: &SearchNode,
    locs: &mut Vec<Loc>,
    play_selection_values: &mut Vec<f64>,
    mut ret_visit_counts: Option<&mut Vec<f64>>,
    scale_max_to_at_least: f64,
    allow_direct_policy_moves: bool,
    always_compute_lcb: bool,
    never_use_lcb: bool,
    //Note: lcbBuf is signed from the player to move's perspective
    lcb_buf: Option<&mut [f64]>,
    radius_buf: Option<&mut [f64]>,
  ) -> bool {
    locs.clear();
    play_selection_values.clear();
    if let Some(ret_visit_counts) = ret_visit_counts.as_deref_mut() {
      ret_visit_counts.clear();
    }

    let nn_output: Option<Rc<NNOutput>> = node.get_nn_output();
    let policy_probs: Option<&[f32]> = match &nn_output {
      Some(o) => Some(&o.get_policy_probs_maybe_noised()[..]),
      None => None,
    };

    let mut total_child_weight: f64 = 0.0;
    let suppress_pass: bool = self.should_suppress_pass(Some(node));

    //Store up basic weights
    let children: SearchNodeChildrenReference = node.get_children();
    let children_capacity: i32 = children.get_capacity();
    for i in 0..children_capacity {
      let child_pointer: &SearchChildPointer = children.index(i);
      let child: Rc<SearchNode> = match child_pointer.get_if_allocated() {
        None => break,
        Some(c) => self.node_arena.get(c),
      };
      let move_loc: Loc = child_pointer.get_move_loc_relaxed();

      let edge_visits: i64 = child_pointer.get_edge_visits();
      let child_weight: f64 = child.stats.get_child_weight(edge_visits);

      locs.push(move_loc);
      total_child_weight += child_weight;

      // If the move appears to be outright illegal in policy probs, zero out the selection value.
      // Also if we're suppressing passes.
      // We always push a value on to playSelectionValues even if that value is 0,
      // because some callers rely on this to line up with the raw indices in the children array of the node.
      if (suppress_pass && move_loc == Board::PASS_LOC) || pp(policy_probs)[self.get_pos(move_loc) as usize] < 0.0 {
        play_selection_values.push(0.0);
        if let Some(ret_visit_counts) = ret_visit_counts.as_deref_mut() {
          ret_visit_counts.push(0.0);
        }
      } else {
        play_selection_values.push(child_weight);
        if let Some(ret_visit_counts) = ret_visit_counts.as_deref_mut() {
          ret_visit_counts.push(edge_visits as f64);
        }
      }
    }

    let mut num_children: i32 = play_selection_values.len() as i32;

    //Find the best child before LCB for pruning. Intended to be the most stably explored child.
    //This is the most weighted child, except with a tiny adjustment so that
    //at very low playouts, variable child weights and discretization doesn't do crazy things.
    let mut non_lcb_best_idx: i32 = 0;
    let mut non_lcb_best_child_weight: f64 = -1e30;
    {
      let mut max_goodness: f64 = -1e30;
      for i in 0..num_children {
        let weight: f64 = play_selection_values[i as usize];
        let child_pointer: &SearchChildPointer = children.index(i);
        let edge_visits: f64 = child_pointer.get_edge_visits() as f64;
        let move_loc: Loc = child_pointer.get_move_loc_relaxed();
        let policy_prob: f64 = pp(policy_probs)[self.get_pos(move_loc) as usize] as f64;

        //Small weight on raw policy, and discount one visit's worth of weight since the most recent
        //visit could be overweighted.
        let g: f64 = weight * cpp::cpp_max(0.0, edge_visits - 1.0) / cpp::cpp_max(1.0, edge_visits) + 2.0 * policy_prob;
        if g > max_goodness {
          max_goodness = g;
          non_lcb_best_child_weight = weight;
          non_lcb_best_idx = i;
        }
      }
    }

    //Possibly reduce weight on children that we spend too many visits on in retrospect
    if Some(node.idx) == self.root_node && num_children > 0 {
      let best_child_pointer: &SearchChildPointer = children.index(non_lcb_best_idx);
      let best_child: Option<NodeIdx> = best_child_pointer.get_if_allocated();
      let best_child_edge_visits: i64 = best_child_pointer.get_edge_visits();
      let best_move_loc: Loc = best_child_pointer.get_move_loc_relaxed();
      testAssert!(best_child.is_some());
      let best_child: Rc<SearchNode> = self.node_arena.get(best_child.expect("testAssert above"));
      let is_root: bool = true;
      let policy_prob_mass_visited: f64 = 1.0; //doesn't matter, since fpu value computed from it isn't used here
      let mut parent_utility: f64 = 0.0;
      let mut parent_weight_per_visit: f64 = 0.0;
      let mut parent_utility_stdev_factor: f64 = 0.0;
      let fpu_value: f64 = self.get_fpu_value_for_children_assume_visited(
        node,
        self.root_pla,
        is_root,
        policy_prob_mass_visited,
        &mut parent_utility,
        &mut parent_weight_per_visit,
        &mut parent_utility_stdev_factor,
      );

      let is_during_search: bool = false;

      let explore_scaling: f64 = self.get_explore_scaling(total_child_weight, parent_utility_stdev_factor);

      testAssert!(nn_output.is_some());
      let count_edge_visit: bool = true;
      let best_child_explore_selection_value: f64 = self.get_explore_selection_value_of_child(
        node,
        pp(policy_probs),
        &best_child,
        best_move_loc,
        explore_scaling,
        total_child_weight,
        best_child_edge_visits,
        fpu_value,
        parent_utility,
        parent_weight_per_visit,
        is_during_search,
        false,
        non_lcb_best_child_weight,
        count_edge_visit,
        None,
      );

      for i in 0..num_children {
        let child_pointer: &SearchChildPointer = children.index(i);
        let child: Option<NodeIdx> = child_pointer.get_if_allocated();
        let move_loc: Loc = child_pointer.get_move_loc_relaxed();
        if suppress_pass && move_loc == Board::PASS_LOC {
          play_selection_values[i as usize] = 0.0;
          continue;
        }
        if i != non_lcb_best_idx {
          let child: Rc<SearchNode> = self.node_arena.get(child.expect("children below numChildren are allocated"));
          let edge_visits: i64 = child_pointer.get_edge_visits();
          let reduced: f64 = self.get_reduced_play_selection_weight(
            node,
            pp(policy_probs),
            &child,
            move_loc,
            explore_scaling,
            edge_visits,
            best_child_explore_selection_value,
          );
          play_selection_values[i as usize] = reduced.ceil();
        }
      }
    }

    //Now compute play selection values taking into account LCB
    if !never_use_lcb && (always_compute_lcb || (self.search_params.use_lcb_for_selection && num_children > 0)) {
      let lcb_buf: &mut [f64] = lcb_buf.expect("lcbBuf is non-NULL unless neverUseLcb");
      let radius_buf: &mut [f64] = radius_buf.expect("radiusBuf is non-NULL unless neverUseLcb");
      let mut best_lcb: f64 = -1e10;
      let mut best_lcb_index: i32 = -1;
      for i in 0..num_children {
        let child_pointer: &SearchChildPointer = children.index(i);
        let child: Rc<SearchNode> = self.node_arena.get(child_pointer.get_if_allocated().expect("children below numChildren are allocated"));
        let edge_visits: i64 = child_pointer.get_edge_visits();
        let move_loc: Loc = child_pointer.get_move_loc_relaxed();
        self.get_self_utility_lcb_and_radius(node, &child, edge_visits, move_loc, &mut lcb_buf[i as usize], &mut radius_buf[i as usize]);
        //Check if this node is eligible to be considered for best LCB
        let weight: f64 = play_selection_values[i as usize];
        if weight > 0.0 && weight >= self.search_params.min_visit_prop_for_lcb * non_lcb_best_child_weight {
          if lcb_buf[i as usize] > best_lcb {
            best_lcb = lcb_buf[i as usize];
            best_lcb_index = i;
          }
        }
      }

      if self.search_params.use_lcb_for_selection
        && num_children > 0
        && (if self.search_params.use_non_buggy_lcb { best_lcb_index >= 0 } else { best_lcb_index > 0 })
      {
        //Best LCB move gets a bonus that ensures it is large enough relative to every other child
        let mut adjusted_weight: f64 = play_selection_values[best_lcb_index as usize];
        for i in 0..num_children {
          if i != best_lcb_index {
            let excess_value: f64 = best_lcb - lcb_buf[i as usize];
            //This move is actually worse lcb than some other move, it's just that the other
            //move failed its checks for having enough minimum weight. So don't actually
            //try to compute how much better this one is than that one, because it's not better.
            if excess_value < 0.0 {
              continue;
            }

            let radius: f64 = radius_buf[i as usize];
            //How many times wider would the radius have to be before the lcb would be worse?
            //Add adjust the denom so that we cannot possibly gain more than a factor of 5, just as a guard
            let radius_factor: f64 = (radius + excess_value) / (radius + 0.20 * excess_value);

            //That factor, squared, is the number of "weight" more that we should pretend we have, for
            //the purpose of selection, since normally stdev is proportional to 1/weight^2.
            let lbound: f64 = radius_factor * radius_factor * play_selection_values[i as usize];
            if lbound > adjusted_weight {
              adjusted_weight = lbound;
            }
          }
        }
        play_selection_values[best_lcb_index as usize] = adjusted_weight;
      }
    }

    let is_okay_raw_policy_move_at_root = |move_loc: Loc, policy_prob: f64, obey_allowed_root_move: bool| -> bool {
      if !self.root_history.is_legal(&self.root_board, move_loc, self.root_pla)
        || policy_prob < 0.0
        || (obey_allowed_root_move && !self.is_allowed_root_move(move_loc))
      {
        return false;
      }
      let avoid_move_until_by_loc: &Vec<i32> =
        if self.root_pla == P_BLACK { &self.avoid_move_until_by_loc_black } else { &self.avoid_move_until_by_loc_white };
      if !avoid_move_until_by_loc.is_empty() {
        testAssert!(avoid_move_until_by_loc.len() >= Board::MAX_ARR_SIZE as usize);
        let until_depth: i32 = avoid_move_until_by_loc[move_loc as usize];
        if until_depth > 0 {
          return false;
        }
      }
      true
    };

    //If we have no children, then use the policy net directly. Only for the root, though, if calling this on any subtree
    //then just require that we have children, for implementation simplicity (since it requires that we have a board and a boardhistory too)
    //(and we also use isAllowedRootMove and avoidMoveUntilByLoc)
    if num_children == 0 {
      if nn_output.is_none() || Some(node.idx) != self.root_node || !allow_direct_policy_moves {
        return false;
      }

      let mut obey_allowed_root_move: bool = true;
      loop {
        for move_pos in 0..self.policy_size {
          let move_loc: Loc = NNPos::pos_to_loc(move_pos, self.root_board.x_size, self.root_board.y_size, self.nn_x_len, self.nn_y_len);
          let mut policy_prob: f64 = pp(policy_probs)[move_pos as usize] as f64;
          if !is_okay_raw_policy_move_at_root(move_loc, policy_prob, obey_allowed_root_move) {
            continue;
          }
          if suppress_pass && move_loc == Board::PASS_LOC {
            policy_prob = 0.0;
          }
          locs.push(move_loc);
          play_selection_values.push(policy_prob);
          num_children += 1;
        }
        //Still no children? Then at this point just ignore isAllowedRootMove.
        if num_children == 0 && obey_allowed_root_move {
          obey_allowed_root_move = false;
          continue;
        }
        break;
      }
    }

    //Might happen absurdly rarely if we both have no children and don't properly have an nnOutput
    //but have a hash collision or something so we "found" an nnOutput anyways.
    //Could also happen if we have avoidMoveUntilByLoc pruning all the allowed moves.
    if num_children == 0 {
      return false;
    }

    let mut max_value: f64 = 0.0;
    for i in 0..num_children {
      if play_selection_values[i as usize] > max_value {
        max_value = play_selection_values[i as usize];
      }
    }

    if max_value <= 1e-50 {
      //If we reached this point we have nonzero many children but the children are all weightless.
      //In that case, at least set each one to be weighted by its policy.
      for i in 0..num_children {
        play_selection_values[i as usize] = cpp::cpp_max(0.0, pp(policy_probs)[self.get_pos(locs[i as usize]) as usize] as f64);
      }
      //Recompute max
      for i in 0..num_children {
        if play_selection_values[i as usize] > max_value {
          max_value = play_selection_values[i as usize];
        }
      }
      if max_value <= 1e-50 {
        return false;
      }
    }

    //Sanity check - if somehow we had more than this, something must have overflowed or gone wrong
    testAssert!(max_value < 1e40);

    let amount_to_subtract: f64 = cpp::cpp_min(self.search_params.chosen_move_subtract, max_value / 64.0);
    let amount_to_prune: f64 = cpp::cpp_min(self.search_params.chosen_move_prune, max_value / 64.0);
    for i in 0..num_children {
      if play_selection_values[i as usize] < amount_to_prune {
        play_selection_values[i as usize] = 0.0;
      } else {
        play_selection_values[i as usize] -= amount_to_subtract;
        if play_selection_values[i as usize] <= 0.0 {
          play_selection_values[i as usize] = 0.0;
        }
      }
    }

    // Average in human policy
    if self.human_evaluator.is_some()
      && (self.search_params.human_sl_profile.initialized || !self.human_evaluator.as_ref().expect("checked is_some").requires_sgf_metadata())
      && self.search_params.human_sl_chosen_move_prop > 0.0
    {
      let human_output: Option<Rc<NNOutput>> = node.get_human_output();
      let human_probs: Option<&[f32]> = match &human_output {
        Some(o) => Some(&o.get_policy_probs_maybe_noised()[..]),
        None => None,
      };
      if let Some(human_probs) = human_probs {
        // First, take a pass to just fill out all the legal/allowed moves into the play selection values, if allowed, and if at root.
        if Some(node.idx) == self.root_node && allow_direct_policy_moves {
          let mut locs_set: BTreeSet<Loc> = locs.iter().copied().collect();
          for move_pos in 0..self.policy_size {
            let move_loc: Loc = NNPos::pos_to_loc(move_pos, self.root_board.x_size, self.root_board.y_size, self.nn_x_len, self.nn_y_len);
            let human_prob: f64 = human_probs[move_pos as usize] as f64;
            let obey_allowed_root_move: bool = true;
            if !is_okay_raw_policy_move_at_root(move_loc, human_prob, obey_allowed_root_move) {
              continue;
            }
            if locs_set.contains(&move_loc) {
              continue;
            }
            locs.push(move_loc);
            locs_set.insert(move_loc);
            play_selection_values.push(0.0); // Pushing zeros since we're just filling in
            num_children += 1;
          }
        }

        // Grab utility on the moves we have utilities for.
        let mut shifted_policy: BTreeMap<Loc, f64> = BTreeMap::new();
        let mut self_utilities: BTreeMap<Loc, f64> = BTreeMap::new();
        let mut self_utility_max: f64 = -1e10;
        let mut self_utility_sum: f64 = 0.0;
        for i in 0..children_capacity {
          let child_pointer: &SearchChildPointer = children.index(i);
          let child: Rc<SearchNode> = match child_pointer.get_if_allocated() {
            None => break,
            Some(c) => self.node_arena.get(c),
          };
          let move_loc: Loc = child_pointer.get_move_loc_relaxed();
          let mut human_prob: f64 = human_probs[self.get_pos(move_loc) as usize] as f64;
          if (suppress_pass && move_loc == Board::PASS_LOC) || human_prob < 0.0 {
            human_prob = 0.0;
          }

          *shifted_policy.entry(move_loc).or_insert(0.0) = human_prob;
          *self_utilities.entry(move_loc).or_insert(0.0) =
            (if self.root_pla == P_WHITE { 1 } else { -1 }) as f64 * child.stats.utility_avg.get() /*acquire*/;
          self_utility_max = cpp::cpp_max(self_utility_max, *self_utilities.entry(move_loc).or_insert(0.0));
          self_utility_sum += *self_utilities.entry(move_loc).or_insert(0.0);
        }
        // Straight linear average. Use this to complete the remaining utilities, i.e. for fpu
        let self_utility_avg: f64 = self_utility_sum / std::cmp::max(1usize, self_utilities.len()) as f64;
        self_utility_max = cpp::cpp_max(self_utility_max, self_utility_avg); // In case of 0 size
        for &loc in locs.iter() {
          if !shifted_policy.contains_key(&loc) {
            let mut human_prob: f64 = human_probs[self.get_pos(loc) as usize] as f64;
            if (suppress_pass && loc == Board::PASS_LOC) || human_prob < 0.0 {
              human_prob = 0.0;
            }
            *shifted_policy.entry(loc).or_insert(0.0) = human_prob;
            *self_utilities.entry(loc).or_insert(0.0) = self_utility_avg;
          }
        }
        // Perform shift
        for &loc in locs.iter() {
          let su: f64 = *self_utilities.entry(loc).or_insert(0.0);
          *shifted_policy.entry(loc).or_insert(0.0) *= math::exp((su - self_utility_max) / self.search_params.human_sl_chosen_move_pikl_lambda);
        }

        let mut shifted_policy_sum: f64 = 0.0;
        for &loc in locs.iter() {
          shifted_policy_sum += *shifted_policy.entry(loc).or_insert(0.0);
        }

        // Renormalize and average in to current play selection values, scaling up to the current sum scale of playSelectionValues.
        if shifted_policy_sum > 0.0 {
          for &loc in locs.iter() {
            *shifted_policy.entry(loc).or_insert(0.0) /= shifted_policy_sum;
          }

          let mut play_selection_value_sum: f64 = 0.0;
          let mut play_selection_value_non_pass_sum: f64 = 0.0;
          for i in 0..num_children {
            play_selection_value_sum += play_selection_values[i as usize];
            if locs[i as usize] != Board::PASS_LOC {
              play_selection_value_non_pass_sum += play_selection_values[i as usize];
            }
          }

          if self.search_params.human_sl_chosen_move_ignore_pass {
            let mut shifted_policy_non_pass_sum: f64 = 0.0;
            for &loc in locs.iter() {
              if loc != Board::PASS_LOC {
                shifted_policy_non_pass_sum += *shifted_policy.entry(loc).or_insert(0.0);
              }
            }
            if shifted_policy_non_pass_sum > 0.0 {
              for &loc in locs.iter() {
                if loc != Board::PASS_LOC {
                  let v: f64 = *shifted_policy.entry(loc).or_insert(0.0);
                  *shifted_policy.entry(loc).or_insert(0.0) =
                    v / shifted_policy_non_pass_sum * play_selection_value_non_pass_sum / play_selection_value_sum;
                } else {
                  *shifted_policy.entry(loc).or_insert(0.0) = (play_selection_value_sum - play_selection_value_non_pass_sum) / play_selection_value_sum;
                }
              }
            }
          }

          for i in 0..num_children {
            let sp: f64 = *shifted_policy.entry(locs[i as usize]).or_insert(0.0);
            play_selection_values[i as usize] +=
              self.search_params.human_sl_chosen_move_prop * (play_selection_value_sum * sp - play_selection_values[i as usize]);
          }
        }
      }
    }

    max_value = 0.0;
    for i in 0..num_children {
      if play_selection_values[i as usize] > max_value {
        max_value = play_selection_values[i as usize];
      }
    }
    testAssert!(max_value > 0.0);
    if max_value < scale_max_to_at_least {
      for i in 0..num_children {
        play_selection_values[i as usize] *= scale_max_to_at_least / max_value;
      }
    }

    true
  }

  /// upstream: Search::getRootValues (searchresults.cpp)
  //Get the values recorded for the root node, if possible.
  pub fn get_root_values(&self, values: &mut ReportedSearchValues) -> bool {
    let root_node: Option<Rc<SearchNode>> = self.get_root_node();
    self.get_node_values(root_node.as_deref(), values)
  }

  /// upstream: Search::getRootValuesRequireSuccess (searchresults.cpp)
  //Same, same, but throws an exception if no values could be obtained
  /// PORT NOTE: throws -> Result.
  pub fn get_root_values_require_success(&self) -> Result<ReportedSearchValues, KataError> {
    let mut values: ReportedSearchValues = ReportedSearchValues::new();
    let root_node: Rc<SearchNode> = match self.root_node {
      None => return Err(KataError::StringError("Bug? Bot search root was null".to_string())),
      Some(r) => self.node_arena.get(r),
    };
    let success: bool = self.get_node_values(Some(&root_node), &mut values);
    if !success {
      return Err(KataError::StringError("Bug? Bot search returned no root values".to_string()));
    }
    Ok(values)
  }

  /// upstream: Search::getRootRawNNValues (searchresults.cpp)
  //Same, but based only on the single raw neural net evaluation.
  pub fn get_root_raw_nn_values(&self, values: &mut ReportedSearchValues) -> bool {
    let root_node: Rc<SearchNode> = match self.root_node {
      None => return false,
      Some(r) => self.node_arena.get(r),
    };
    self.get_node_raw_nn_values(&root_node, values)
  }

  /// upstream: Search::getRootRawNNValuesRequireSuccess (searchresults.cpp)
  /// PORT NOTE: throws -> Result.
  pub fn get_root_raw_nn_values_require_success(&self) -> Result<ReportedSearchValues, KataError> {
    let mut values: ReportedSearchValues = ReportedSearchValues::new();
    let root_node: Rc<SearchNode> = match self.root_node {
      None => return Err(KataError::StringError("Bug? Bot search root was null".to_string())),
      Some(r) => self.node_arena.get(r),
    };
    let success: bool = self.get_node_raw_nn_values(&root_node, &mut values);
    if !success {
      return Err(KataError::StringError("Bug? Bot search returned no root values".to_string()));
    }
    Ok(values)
  }

  /// upstream: Search::getNodeRawNNValues (searchresults.cpp)
  pub fn get_node_raw_nn_values(&self, node: &SearchNode, values: &mut ReportedSearchValues) -> bool {
    let nn_output: Rc<NNOutput> = match node.get_nn_output() {
      None => return false,
      Some(o) => o,
    };

    values.win_value = nn_output.white_win_prob as f64;
    values.loss_value = nn_output.white_loss_prob as f64;
    values.no_result_value = nn_output.white_no_result_prob as f64;

    let score_mean: f64 = nn_output.white_score_mean as f64;
    let score_mean_sq: f64 = nn_output.white_score_mean_sq as f64;
    let score_stdev: f64 = ScoreValue::get_score_stdev(score_mean, score_mean_sq);
    let sqrt_board_area: f64 = self.root_board.sqrt_board_area();
    values.static_score_value = ScoreValue::expected_white_score_value(score_mean, score_stdev, 0.0, 2.0, sqrt_board_area);
    values.dynamic_score_value = ScoreValue::expected_white_score_value(
      score_mean,
      score_stdev,
      self.recent_score_center,
      self.search_params.dynamic_score_center_scale,
      sqrt_board_area,
    );
    values.expected_score = score_mean;
    values.expected_score_stdev = score_stdev;
    values.lead = nn_output.white_lead as f64;

    //Sanity check
    testAssert!(values.win_value >= 0.0);
    testAssert!(values.loss_value >= 0.0);
    testAssert!(values.no_result_value >= 0.0);
    testAssert!(values.win_value + values.loss_value + values.no_result_value < 1.001);

    let mut win_loss_value: f64 = values.win_value - values.loss_value;
    if win_loss_value > 1.0 {
      win_loss_value = 1.0;
    }
    if win_loss_value < -1.0 {
      win_loss_value = -1.0;
    }
    values.win_loss_value = win_loss_value;

    values.weight = self.compute_weight_from_nn_output(&nn_output);
    values.visits = 1;

    true
  }

  /// upstream: Search::getNodeValues (searchresults.cpp)
  //Same, but works on a node within the search, not just the root
  pub fn get_node_values(&self, node: Option<&SearchNode>, values: &mut ReportedSearchValues) -> bool {
    let node: &SearchNode = match node {
      None => return false,
      Some(n) => n,
    };
    let visits: i64 = node.stats.visits.get() /*acquire*/;
    let weight_sum: f64 = node.stats.weight_sum.get() /*acquire*/;
    let win_loss_value_avg: f64 = node.stats.win_loss_value_avg.get() /*acquire*/;
    let no_result_value_avg: f64 = node.stats.no_result_value_avg.get() /*acquire*/;
    let score_mean_avg: f64 = node.stats.score_mean_avg.get() /*acquire*/;
    let score_mean_sq_avg: f64 = node.stats.score_mean_sq_avg.get() /*acquire*/;
    let lead_avg: f64 = node.stats.lead_avg.get() /*acquire*/;
    let utility_avg: f64 = node.stats.utility_avg.get() /*acquire*/;

    if weight_sum <= 0.0 {
      return false;
    }
    debug_assert!(visits >= 0);
    if Some(node.idx) == self.root_node {
      //For terminal nodes, we may have no nnoutput and yet we have legitimate visits and terminal evals.
      //But for the root, the root is never treated as a terminal node and always gets an nneval, so if
      //it has visits and weight, it has an nnoutput unless something has gone wrong.
      let nn_output: Option<Rc<NNOutput>> = node.get_nn_output();
      testAssert!(nn_output.is_some());
      let _ = nn_output;
    }

    *values = ReportedSearchValues::new_search(
      self,
      win_loss_value_avg,
      no_result_value_avg,
      score_mean_avg,
      score_mean_sq_avg,
      lead_avg,
      utility_avg,
      weight_sum,
      visits,
    );
    true
  }

  /// upstream: Search::getRootNode (searchresults.cpp)
  /// PORT NOTE (OWNERSHIP.tsv): returns the node's Rc.
  pub fn get_root_node(&self) -> Option<Rc<SearchNode>> {
    self.root_node.map(|r| self.node_arena.get(r))
  }

  /// upstream: Search::getChildForMove (searchresults.cpp)
  /// PORT NOTE (OWNERSHIP.tsv): returns the child's Rc.
  pub fn get_child_for_move(&self, node: Option<&SearchNode>, move_loc: Loc) -> Option<Rc<SearchNode>> {
    let node: &SearchNode = node?;
    let children: SearchNodeChildrenReference = node.get_children();
    let children_capacity: i32 = children.get_capacity();
    for i in 0..children_capacity {
      let child_pointer: &SearchChildPointer = children.index(i);
      let child: NodeIdx = child_pointer.get_if_allocated()?;
      let child_move_loc: Loc = child_pointer.get_move_loc_relaxed();
      if move_loc == child_move_loc {
        return Some(self.node_arena.get(child));
      }
    }
    None
  }

  /// upstream: Search::getChosenMoveLoc (searchresults.cpp)
  //Choose a move at the root of the tree, with randomization, if possible.
  //Might return Board::NULL_LOC if there is no root, or no legal moves that aren't forcibly pruned, etc.
  /// PORT NOTE: non-const upstream (it draws from nonSearchRand); `&self` here because
  /// nonSearchRand is a RefCell (search.rs).
  pub fn get_chosen_move_loc(&self) -> Loc {
    if self.root_node.is_none() {
      return Board::NULL_LOC;
    }

    let mut locs: Vec<Loc> = Vec::new();
    let mut play_selection_values: Vec<f64> = Vec::new();
    let suc: bool = self.get_play_selection_values(&mut locs, &mut play_selection_values, 0.0);
    if !suc {
      return Board::NULL_LOC;
    }

    testAssert!(locs.len() == play_selection_values.len());

    let temperature: f64 = self.interpolate_early(
      self.search_params.chosen_move_temperature_halflife,
      self.search_params.chosen_move_temperature_early,
      self.search_params.chosen_move_temperature,
    );

    let idx_chosen: u32 = Search::choose_index_with_temperature(
      &mut self.non_search_rand.borrow_mut(),
      &play_selection_values,
      play_selection_values.len() as i32,
      temperature,
      self.search_params.chosen_move_temperature_only_below_prob,
      None,
    );
    locs[idx_chosen as usize]
  }

  /// upstream: Search::getPolicy(float[]) (searchresults.cpp)
  //Get the root node's policy prediction
  pub fn get_policy(&self, policy_probs: &mut [f32; NNPos::MAX_NN_POLICY_SIZE as usize]) -> bool {
    let root_node: Option<Rc<SearchNode>> = self.get_root_node();
    self.get_policy_node(root_node.as_deref(), policy_probs)
  }

  /// upstream: Search::getPolicy(const SearchNode*, float[]) (searchresults.cpp)
  pub fn get_policy_node(&self, node: Option<&SearchNode>, policy_probs: &mut [f32; NNPos::MAX_NN_POLICY_SIZE as usize]) -> bool {
    let node: &SearchNode = match node {
      None => return false,
      Some(n) => n,
    };
    let nn_output: Rc<NNOutput> = match node.get_nn_output() {
      None => return false,
      Some(o) => o,
    };

    policy_probs.copy_from_slice(&nn_output.policy_probs[..]);
    true
  }

  /// upstream: Search::getPolicySurprise (searchresults.cpp)
  //Safe to call concurrently with search
  pub fn get_policy_surprise(&self) -> f64 {
    let mut surprise: f64 = 0.0;
    let mut search_entropy: f64 = 0.0;
    let mut policy_entropy: f64 = 0.0;
    if self.get_policy_surprise_and_entropy(&mut surprise, &mut search_entropy, &mut policy_entropy) {
      return surprise;
    }
    0.0
  }

  /// upstream: Search::getPolicySurpriseAndEntropy(double&, double&, double&) (searchresults.cpp)
  //Get the surprisingness (kl-divergence) of the search result given the policy prior, as well as the entropy of each.
  //Returns false if could not be computed.
  pub fn get_policy_surprise_and_entropy(&self, surprise_ret: &mut f64, search_entropy_ret: &mut f64, policy_entropy_ret: &mut f64) -> bool {
    let root_node: Option<Rc<SearchNode>> = self.get_root_node();
    self.get_policy_surprise_and_entropy_node(surprise_ret, search_entropy_ret, policy_entropy_ret, root_node.as_deref())
  }

  /// upstream: Search::getPolicySurpriseAndEntropy(double&, double&, double&, const SearchNode*) (searchresults.cpp)
  //Safe to call concurrently with search
  pub fn get_policy_surprise_and_entropy_node(
    &self,
    surprise_ret: &mut f64,
    search_entropy_ret: &mut f64,
    policy_entropy_ret: &mut f64,
    node: Option<&SearchNode>,
  ) -> bool {
    let node: &SearchNode = match node {
      None => return false,
      Some(n) => n,
    };
    let nn_output: Rc<NNOutput> = match node.get_nn_output() {
      None => return false,
      Some(o) => o,
    };

    let mut locs: Vec<Loc> = Vec::new();
    let mut play_selection_values: Vec<f64> = Vec::new();
    let allow_direct_policy_moves: bool = true;
    let always_compute_lcb: bool = false;
    let mut lcb_buf: Vec<f64> = vec![0.0; NNPos::MAX_NN_POLICY_SIZE as usize];
    let mut radius_buf: Vec<f64> = vec![0.0; NNPos::MAX_NN_POLICY_SIZE as usize];
    let suc: bool = self.get_play_selection_values_node_lcb(
      node,
      &mut locs,
      &mut play_selection_values,
      None,
      1.0,
      allow_direct_policy_moves,
      always_compute_lcb,
      false,
      Some(&mut lcb_buf),
      Some(&mut radius_buf),
    );
    if !suc {
      return false;
    }

    let mut policy_probs_from_nn_buf: Vec<f32> = vec![0.0; NNPos::MAX_NN_POLICY_SIZE as usize];
    {
      let policy_probs_from_nn: &[f32; NNPos::MAX_NN_POLICY_SIZE as usize] = nn_output.get_policy_probs_maybe_noised();
      policy_probs_from_nn_buf.copy_from_slice(&policy_probs_from_nn[..]);
    }

    let mut sum_play_selection_values: f64 = 0.0;
    for i in 0..play_selection_values.len() {
      sum_play_selection_values += play_selection_values[i];
    }

    let mut surprise: f64 = 0.0;
    let mut search_entropy: f64 = 0.0;
    for i in 0..play_selection_values.len() {
      let pos: i32 = self.get_pos(locs[i]);
      let policy: f64 = cpp::cpp_max(policy_probs_from_nn_buf[pos as usize] as f64, 1e-100);
      let target: f64 = play_selection_values[i] / sum_play_selection_values;
      if target > 1e-100 {
        let log_target: f64 = math::ln(target);
        let log_policy: f64 = math::ln(policy);
        surprise += target * (log_target - log_policy);
        search_entropy += -target * log_target;
      }
    }

    let mut policy_entropy: f64 = 0.0;
    for pos in 0..NNPos::MAX_NN_POLICY_SIZE {
      let policy: f64 = policy_probs_from_nn_buf[pos as usize] as f64;
      if policy > 1e-100 {
        policy_entropy += -policy * math::ln(policy);
      }
    }

    //Just in case, guard against float imprecision
    if surprise < 0.0 {
      surprise = 0.0;
    }
    if search_entropy < 0.0 {
      search_entropy = 0.0;
    }
    if policy_entropy < 0.0 {
      policy_entropy = 0.0;
    }

    *surprise_ret = surprise;
    *search_entropy_ret = search_entropy;
    *policy_entropy_ret = policy_entropy;

    true
  }

  /// upstream: Search::printRootOwnershipMap (searchresults.cpp)
  pub fn print_root_ownership_map(&self, out: &mut dyn std::fmt::Write, perspective: Player) {
    let root_node: Rc<SearchNode> = match self.root_node {
      None => return,
      Some(r) => self.node_arena.get(r),
    };
    let nn_output: Rc<NNOutput> = match root_node.get_nn_output() {
      None => return,
      Some(o) => o,
    };
    let white_owner_map: &Vec<f32> = match &nn_output.white_owner_map {
      None => return,
      Some(m) => m,
    };

    let perspective_to_use: Player = if perspective != P_BLACK && perspective != P_WHITE { self.root_pla } else { perspective };
    let perspective_factor: f64 = if perspective_to_use == P_BLACK { -1.0 } else { 1.0 };

    for y in 0..self.root_board.y_size {
      for x in 0..self.root_board.x_size {
        let pos: i32 = NNPos::xy_to_pos(x, y, nn_output.nn_x_len);
        write!(out, "{}", strprintf!("%6.1f ", perspective_factor * white_owner_map[pos as usize] as f64 * 100.0)).unwrap();
      }
      writeln!(out).unwrap();
    }
    writeln!(out).unwrap();
  }

  /// upstream: Search::printRootPolicyMap (searchresults.cpp)
  pub fn print_root_policy_map(&self, out: &mut dyn std::fmt::Write) {
    let root_node: Rc<SearchNode> = match self.root_node {
      None => return,
      Some(r) => self.node_arena.get(r),
    };
    let nn_output: Rc<NNOutput> = match root_node.get_nn_output() {
      None => return,
      Some(o) => o,
    };

    let policy_probs: &[f32; NNPos::MAX_NN_POLICY_SIZE as usize] = nn_output.get_policy_probs_maybe_noised();
    for y in 0..self.root_board.y_size {
      for x in 0..self.root_board.x_size {
        let pos: i32 = NNPos::xy_to_pos(x, y, nn_output.nn_x_len);
        // PORT NOTE: float*int is computed in float, then promoted for the vararg (PORTING.md §6 item 1).
        write!(out, "{}", strprintf!("%6.1f ", policy_probs[pos as usize] * 100.0f32)).unwrap();
      }
      writeln!(out).unwrap();
    }
    writeln!(out).unwrap();
  }

  /// upstream: Search::printRootEndingScoreValueBonus (searchresults.cpp)
  pub fn print_root_ending_score_value_bonus(&self, out: &mut dyn std::fmt::Write) {
    let root_node: Rc<SearchNode> = match self.root_node {
      None => return,
      Some(r) => self.node_arena.get(r),
    };
    let nn_output: Rc<NNOutput> = match root_node.get_nn_output() {
      None => return,
      Some(o) => o,
    };
    if nn_output.white_owner_map.is_none() {
      return;
    }

    let children: SearchNodeChildrenReference = root_node.get_children();
    let children_capacity: i32 = children.get_capacity();
    for i in 0..children_capacity {
      let child: Rc<SearchNode> = match children.index(i).get_if_allocated() {
        None => break,
        Some(c) => self.node_arena.get(c),
      };

      let edge_visits: i64 = children.index(i).get_edge_visits();
      let move_loc: Loc = children.index(i).get_move_loc_relaxed();
      let child_visits: i64 = child.stats.visits.get() /*acquire*/;
      let score_mean_avg: f64 = child.stats.score_mean_avg.get() /*acquire*/;
      let score_mean_sq_avg: f64 = child.stats.score_mean_sq_avg.get() /*acquire*/;
      let utility_avg: f64 = child.stats.utility_avg.get() /*acquire*/;

      let utility_no_bonus: f64 = utility_avg;
      let ending_score_bonus: f64 = self.get_ending_white_score_bonus(&root_node, move_loc);
      let utility_diff: f64 = self.get_score_utility_diff(score_mean_avg, score_mean_sq_avg, ending_score_bonus);
      let utility_with_bonus: f64 = utility_no_bonus + utility_diff;

      // PORT NOTE: upstream passes int64_t to %d (UB); x86-64 prints the low 32 bits as int, hence `as i32`.
      write!(
        out,
        "{} {}",
        Location::to_string(move_loc, &self.root_board),
        strprintf!(
          "visits %d edgeVisits %d utilityNoBonus %.2fc utilityWithBonus %.2fc endingScoreBonus %.2f",
          child_visits as i32,
          edge_visits as i32,
          utility_no_bonus * 100.0,
          utility_with_bonus * 100.0,
          ending_score_bonus
        )
      )
      .unwrap();
      writeln!(out).unwrap();
    }
  }

  /// upstream: Search::appendPV (searchresults.cpp)
  //Append the PV from node n onward (not including the move if any that reached node n)
  pub fn append_pv(
    &self,
    buf: &mut Vec<Loc>,
    visits_buf: &mut Vec<i64>,
    edge_visits_buf: &mut Vec<i64>,
    scratch_locs: &mut Vec<Loc>,
    scratch_values: &mut Vec<f64>,
    node: Option<&SearchNode>,
    max_depth: i32,
  ) {
    self.append_pv_for_move(buf, visits_buf, edge_visits_buf, scratch_locs, scratch_values, node, Board::NULL_LOC, max_depth);
  }

  /// upstream: Search::appendPVForMove (searchresults.cpp)
  //Append the PV from node n for specified move, assuming move is a child move of node n
  /// PORT NOTE: the walk re-reads each node from the arena by index to hold it while descending.
  pub fn append_pv_for_move(
    &self,
    buf: &mut Vec<Loc>,
    visits_buf: &mut Vec<i64>,
    edge_visits_buf: &mut Vec<i64>,
    scratch_locs: &mut Vec<Loc>,
    scratch_values: &mut Vec<f64>,
    node: Option<&SearchNode>,
    move_: Loc,
    max_depth: i32,
  ) {
    let mut node: Rc<SearchNode> = match node {
      None => return,
      Some(n) => self.node_arena.get(n.idx),
    };

    for depth in 0..max_depth {
      let allow_direct_policy_moves: bool = true;
      let success: bool = self.get_play_selection_values_node(&node, scratch_locs, scratch_values, None, 1.0, allow_direct_policy_moves);
      if !success {
        return;
      }

      let mut max_selection_value: f64 = Search::POLICY_ILLEGAL_SELECTION_VALUE;
      let mut best_child_idx: i32 = -1;
      let mut best_child_move_loc: Loc = Board::NULL_LOC;

      for i in 0..scratch_values.len() {
        let move_loc: Loc = scratch_locs[i];
        let selection_value: f64 = scratch_values[i];

        if depth == 0 && move_loc == move_ {
          max_selection_value = selection_value;
          best_child_idx = i as i32;
          best_child_move_loc = move_loc;
          break;
        }

        if selection_value > max_selection_value {
          max_selection_value = selection_value;
          best_child_idx = i as i32;
          best_child_move_loc = move_loc;
        }
      }
      let _ = max_selection_value;

      if best_child_idx < 0 || best_child_move_loc == Board::NULL_LOC {
        return;
      }
      if depth == 0 && move_ != Board::NULL_LOC && best_child_move_loc != move_ {
        return;
      }

      let (child, edge_visits): (Rc<SearchNode>, i64) = {
        let children: SearchNodeChildrenReference = node.get_children();
        let children_capacity: i32 = children.get_capacity();
        //Direct policy move
        if best_child_idx >= children_capacity {
          buf.push(best_child_move_loc);
          visits_buf.push(0);
          edge_visits_buf.push(0);
          return;
        }
        let child: Option<NodeIdx> = children.index(best_child_idx).get_if_allocated();
        //Direct policy move
        let child: NodeIdx = match child {
          None => {
            buf.push(best_child_move_loc);
            visits_buf.push(0);
            edge_visits_buf.push(0);
            return;
          }
          Some(c) => c,
        };
        // PORT NOTE: upstream reads the edge visits after `node = child` through the (still parent's)
        // children reference; same value.
        (self.node_arena.get(child), children.index(best_child_idx).get_edge_visits())
      };

      node = child;

      let visits: i64 = node.stats.visits.get() /*acquire*/;

      buf.push(best_child_move_loc);
      visits_buf.push(visits);
      edge_visits_buf.push(edge_visits);
    }
  }

  /// upstream: Search::printPV(ostream&, const SearchNode*, int) (searchresults.cpp)
  pub fn print_pv(&self, out: &mut dyn std::fmt::Write, n: Option<&SearchNode>, max_depth: i32) {
    let mut buf: Vec<Loc> = Vec::new();
    let mut visits_buf: Vec<i64> = Vec::new();
    let mut edge_visits_buf: Vec<i64> = Vec::new();
    let mut scratch_locs: Vec<Loc> = Vec::new();
    let mut scratch_values: Vec<f64> = Vec::new();
    self.append_pv(&mut buf, &mut visits_buf, &mut edge_visits_buf, &mut scratch_locs, &mut scratch_values, n, max_depth);
    self.print_pv_buf(out, &buf);
  }

  /// upstream: Search::printPV(ostream&, const vector<Loc>&) (searchresults.cpp)
  /// PORT NOTE: name from PORTING.md Appendix B.
  pub(crate) fn print_pv_buf(&self, out: &mut dyn std::fmt::Write, buf: &[Loc]) {
    let mut printed_anything: bool = false;
    for i in 0..buf.len() {
      if printed_anything {
        write!(out, " ").unwrap();
      }
      if buf[i] == Board::NULL_LOC {
        continue;
      }
      write!(out, "{}", Location::to_string(buf[i], &self.root_board)).unwrap();
      printed_anything = true;
    }
  }

  /// upstream: Search::getAnalysisDataOfSingleChild (searchresults.cpp)
  //Child should NOT be locked.
  pub(crate) fn get_analysis_data_of_single_child(
    &self,
    child: Option<&SearchNode>,
    edge_visits: i64,
    scratch_locs: &mut Vec<Loc>,
    scratch_values: &mut Vec<f64>,
    move_: Loc,
    policy_prob: f64,
    fpu_value: f64,
    parent_utility: f64,
    parent_win_loss_value: f64,
    parent_score_mean: f64,
    parent_score_stdev: f64,
    parent_lead: f64,
    max_pv_depth: i32,
  ) -> AnalysisData {
    let mut child_visits: i64 = 0;
    let mut win_loss_value_avg: f64 = 0.0;
    let mut no_result_value_avg: f64 = 0.0;
    let mut score_mean_avg: f64 = 0.0;
    let mut score_mean_sq_avg: f64 = 0.0;
    let mut lead_avg: f64 = 0.0;
    let mut utility_avg: f64 = 0.0;
    let mut utility_sq_avg: f64 = 0.0;
    let mut weight_sum: f64 = 0.0;
    let mut weight_sq_sum: f64 = 0.0;
    let mut child_weight_sum: f64 = 0.0;

    if let Some(child) = child {
      child_visits = child.stats.visits.get() /*acquire*/;
      win_loss_value_avg = child.stats.win_loss_value_avg.get() /*acquire*/;
      no_result_value_avg = child.stats.no_result_value_avg.get() /*acquire*/;
      score_mean_avg = child.stats.score_mean_avg.get() /*acquire*/;
      score_mean_sq_avg = child.stats.score_mean_sq_avg.get() /*acquire*/;
      lead_avg = child.stats.lead_avg.get() /*acquire*/;
      utility_avg = child.stats.utility_avg.get() /*acquire*/;
      utility_sq_avg = child.stats.utility_sq_avg.get() /*acquire*/;
      weight_sum = child.stats.get_child_weight_child_visits(edge_visits, child_visits);
      weight_sq_sum = child.stats.get_child_weight_sq_child_visits(edge_visits, child_visits);
      child_weight_sum = child.stats.weight_sum.get() /*acquire*/;
    }

    let mut data: AnalysisData = AnalysisData::new();
    data.move_ = move_;
    data.num_visits = edge_visits;
    if child_visits <= 0 || child_weight_sum <= 1e-30 {
      data.utility = fpu_value;
      data.score_utility = self.get_score_utility(parent_score_mean, parent_score_mean * parent_score_mean + parent_score_stdev * parent_score_stdev);
      data.result_utility = fpu_value - data.score_utility;
      data.win_loss_value =
        if self.search_params.win_loss_utility_factor == 1.0 { parent_win_loss_value + (fpu_value - parent_utility) } else { 0.0 };
      data.no_result_value = 0.0;
      // Make sure winloss values due to FPU don't go out of bounds for purposes of reporting to UI
      if data.win_loss_value < -1.0 {
        data.win_loss_value = -1.0;
      }
      if data.win_loss_value > 1.0 {
        data.win_loss_value = 1.0;
      }
      data.score_mean = parent_score_mean;
      data.score_stdev = parent_score_stdev;
      data.lead = parent_lead;
      data.ess = 0.0;
      data.weight_sum = 0.0;
      data.weight_sq_sum = 0.0;
      data.utility_sq_avg = data.utility * data.utility;
      data.score_mean_sq_avg = parent_score_mean * parent_score_mean + parent_score_stdev * parent_score_stdev;
      data.child_visits = child_visits;
      data.child_weight_sum = child_weight_sum;
    } else {
      data.utility = utility_avg;
      data.result_utility = self.get_result_utility(win_loss_value_avg, no_result_value_avg);
      data.score_utility = self.get_score_utility(score_mean_avg, score_mean_sq_avg);
      data.win_loss_value = win_loss_value_avg;
      data.no_result_value = no_result_value_avg;
      data.score_mean = score_mean_avg;
      data.score_stdev = ScoreValue::get_score_stdev(score_mean_avg, score_mean_sq_avg);
      data.lead = lead_avg;
      data.ess = weight_sum * weight_sum / cpp::cpp_max(1e-8, weight_sq_sum);
      data.weight_sum = weight_sum;
      data.weight_sq_sum = weight_sq_sum;
      data.utility_sq_avg = utility_sq_avg;
      data.score_mean_sq_avg = score_mean_sq_avg;
      data.child_visits = child_visits;
      data.child_weight_sum = child_weight_sum;
    }

    data.policy_prior = policy_prob;
    data.order = 0;

    data.pv.clear();
    data.pv.push(move_);
    data.pv_visits.clear();
    data.pv_visits.push(child_visits);
    data.pv_edge_visits.clear();
    data.pv_edge_visits.push(edge_visits);
    self.append_pv(&mut data.pv, &mut data.pv_visits, &mut data.pv_edge_visits, scratch_locs, scratch_values, child, max_pv_depth);

    data.node = child.map(|c| c.idx);

    data
  }

  /// upstream: Search::getAnalysisData(vector<AnalysisData>&, int, bool, int, bool) (searchresults.cpp)
  //Get detailed analysis data, designed for lz-analyze and kata-analyze commands.
  pub fn get_analysis_data(
    &self,
    buf: &mut Vec<AnalysisData>,
    min_moves_to_try_to_get: i32,
    include_weight_factors: bool,
    max_pv_depth: i32,
    duplicate_for_symmetries: bool,
  ) {
    buf.clear();
    let root_node: Rc<SearchNode> = match self.root_node {
      None => return,
      Some(r) => self.node_arena.get(r),
    };
    self.get_analysis_data_node(&root_node, buf, min_moves_to_try_to_get, include_weight_factors, max_pv_depth, duplicate_for_symmetries);
  }

  /// upstream: Search::getAnalysisData(const SearchNode&, vector<AnalysisData>&, int, bool, int, bool) (searchresults.cpp)
  pub fn get_analysis_data_node(
    &self,
    node: &SearchNode,
    buf: &mut Vec<AnalysisData>,
    min_moves_to_try_to_get: i32,
    include_weight_factors: bool,
    max_pv_depth: i32,
    duplicate_for_symmetries: bool,
  ) {
    buf.clear();
    let mut children: Vec<Rc<SearchNode>> = Vec::new();
    let mut children_edge_visits: Vec<i64> = Vec::new();
    let mut children_move_locs: Vec<Loc> = Vec::new();
    children.reserve((self.root_board.x_size * self.root_board.y_size + 1) as usize);
    children_edge_visits.reserve((self.root_board.x_size * self.root_board.y_size + 1) as usize);
    children_move_locs.reserve((self.root_board.x_size * self.root_board.y_size + 1) as usize);

    let num_children: i32;
    let mut scratch_locs: Vec<Loc> = Vec::new();
    let mut scratch_values: Vec<f64> = Vec::new();
    let mut lcb_buf: Vec<f64> = vec![0.0; NNPos::MAX_NN_POLICY_SIZE as usize];
    let mut radius_buf: Vec<f64> = vec![0.0; NNPos::MAX_NN_POLICY_SIZE as usize];
    let mut policy_probs: Vec<f32> = vec![0.0; NNPos::MAX_NN_POLICY_SIZE as usize];
    {
      let children_arr: SearchNodeChildrenReference = node.get_children();
      let children_capacity: i32 = children_arr.get_capacity();
      for i in 0..children_capacity {
        let child_pointer: &SearchChildPointer = children_arr.index(i);
        let child: Rc<SearchNode> = match child_pointer.get_if_allocated() {
          None => break,
          Some(c) => self.node_arena.get(c),
        };
        children.push(child);
        children_edge_visits.push(child_pointer.get_edge_visits());
        children_move_locs.push(child_pointer.get_move_loc_relaxed());
      }
      num_children = children.len() as i32;

      if num_children <= 0 {
        return;
      }
      debug_assert!(num_children <= NNPos::MAX_NN_POLICY_SIZE);

      let always_compute_lcb: bool = true;
      let got_play_selection_values: bool = self.get_play_selection_values_node_lcb(
        node,
        &mut scratch_locs,
        &mut scratch_values,
        None,
        1.0,
        false,
        always_compute_lcb,
        false,
        Some(&mut lcb_buf),
        Some(&mut radius_buf),
      );

      // No play selection values - then fill with values consistent with all 0 visits.
      // We want it to be possible to get analysis data even when all visits are weightless.
      if !got_play_selection_values {
        for i in 0..num_children {
          scratch_locs.push(children_move_locs[i as usize]);
          scratch_values.push(0.0);
        }
        let mut lcb_buf_value: f64 = 0.0;
        let mut radius_buf_value: f64 = 0.0;
        self.get_self_utility_lcb_and_radius_zero_visits(&mut lcb_buf_value, &mut radius_buf_value);
        lcb_buf[..num_children as usize].fill(lcb_buf_value);
        radius_buf[..num_children as usize].fill(radius_buf_value);
      }

      let nn_output: Rc<NNOutput> = node.get_nn_output().expect("a node with children has an nnOutput (upstream dereferences it)");
      let policy_probs_from_nn: &[f32; NNPos::MAX_NN_POLICY_SIZE as usize] = nn_output.get_policy_probs_maybe_noised();
      for i in 0..NNPos::MAX_NN_POLICY_SIZE {
        policy_probs[i as usize] = policy_probs_from_nn[i as usize];
      }
    }

    //Copy to make sure we keep these values so we can reuse scratch later for PV
    let play_selection_values: Vec<f64> = scratch_values.clone();

    let mut policy_prob_mass_visited: f64 = 0.0;
    {
      for i in 0..num_children {
        policy_prob_mass_visited += cpp::cpp_max(0.0, policy_probs[self.get_pos(children_move_locs[i as usize]) as usize] as f64);
      }
      //Probability mass should not sum to more than 1, giving a generous allowance
      //for floating point error.
      debug_assert!(policy_prob_mass_visited <= 1.0001);
    }

    let parent_win_loss_value: f64;
    let parent_score_mean: f64;
    let parent_score_stdev: f64;
    let parent_lead: f64;
    {
      let weight_sum: f64 = node.stats.weight_sum.get() /*acquire*/;
      let win_loss_value_avg: f64 = node.stats.win_loss_value_avg.get() /*acquire*/;
      let score_mean_avg: f64 = node.stats.score_mean_avg.get() /*acquire*/;
      let score_mean_sq_avg: f64 = node.stats.score_mean_sq_avg.get() /*acquire*/;
      let lead_avg: f64 = node.stats.lead_avg.get() /*acquire*/;
      debug_assert!(weight_sum > 0.0);
      let _ = weight_sum;

      parent_win_loss_value = win_loss_value_avg;
      parent_score_mean = score_mean_avg;
      parent_score_stdev = ScoreValue::get_score_stdev(parent_score_mean, score_mean_sq_avg);
      parent_lead = lead_avg;
    }

    let mut parent_utility: f64 = 0.0;
    let mut parent_weight_per_visit: f64 = 0.0;
    let mut parent_utility_stdev_factor: f64 = 0.0;
    let fpu_value: f64 = self.get_fpu_value_for_children_assume_visited(
      node,
      node.next_pla,
      true,
      policy_prob_mass_visited,
      &mut parent_utility,
      &mut parent_weight_per_visit,
      &mut parent_utility_stdev_factor,
    );

    let mut stats_buf: Vec<MoreNodeStats> = vec![MoreNodeStats::new(); num_children as usize];
    for i in 0..num_children {
      let child: &SearchNode = &children[i as usize];
      let edge_visits: i64 = children_edge_visits[i as usize];
      let move_loc: Loc = children_move_locs[i as usize];
      let policy_prob: f64 = policy_probs[self.get_pos(move_loc) as usize] as f64;
      let mut data: AnalysisData = self.get_analysis_data_of_single_child(
        Some(child),
        edge_visits,
        &mut scratch_locs,
        &mut scratch_values,
        move_loc,
        policy_prob,
        fpu_value,
        parent_utility,
        parent_win_loss_value,
        parent_score_mean,
        parent_score_stdev,
        parent_lead,
        max_pv_depth,
      );
      data.play_selection_value = play_selection_values[i as usize];
      //Make sure data.lcb is from white's perspective, for consistency with everything else
      //In lcbBuf, it's from self perspective, unlike values at nodes.
      data.lcb = if node.next_pla == P_BLACK { -lcb_buf[i as usize] } else { lcb_buf[i as usize] };
      data.radius = radius_buf[i as usize];
      let data_utility: f64 = data.utility;
      buf.push(data);

      if include_weight_factors {
        let stats: &mut MoreNodeStats = &mut stats_buf[i as usize];
        stats.stats = NodeStats::new_other(&child.stats);
        stats.self_utility = if node.next_pla == P_WHITE { data_utility } else { -data_utility };
        stats.weight_adjusted = stats.stats.get_child_weight(edge_visits);
        stats.prev_move_loc = move_loc;
      }
    }

    //Find all children and compute weighting of the children based on their values
    if include_weight_factors {
      let mut total_child_weight: f64 = 0.0;
      for i in 0..num_children {
        total_child_weight += stats_buf[i as usize].weight_adjusted;
      }
      if self.search_params.use_noise_pruning {
        let mut policy_probs_buf: Vec<f64> = vec![0.0; NNPos::MAX_NN_POLICY_SIZE as usize];
        for i in 0..num_children {
          policy_probs_buf[i as usize] = cpp::cpp_max(1e-30, policy_probs[self.get_pos(stats_buf[i as usize].prev_move_loc) as usize] as f64);
        }
        total_child_weight = self.prune_noise_weight(&mut stats_buf, num_children, total_child_weight, &policy_probs_buf);
      }
      let amount_to_subtract: f64 = 0.0;
      let amount_to_prune: f64 = 0.0;
      self.downweight_bad_children_and_normalize_weight(
        num_children,
        total_child_weight,
        total_child_weight,
        amount_to_subtract,
        amount_to_prune,
        &mut stats_buf,
      );
      for i in 0..num_children {
        buf[i as usize].weight_factor = stats_buf[i as usize].weight_adjusted;
      }
    }

    //Fill the rest of the moves directly from policy
    if num_children < min_moves_to_try_to_get {
      //A bit inefficient, but no big deal
      for _i in 0..(min_moves_to_try_to_get - num_children) {
        let mut best_pos: i32 = -1;
        let mut best_policy: f64 = -1.0;
        for pos in 0..NNPos::MAX_NN_POLICY_SIZE {
          if (policy_probs[pos as usize] as f64) < best_policy {
            continue;
          }

          let mut already_used: bool = false;
          for j in 0..buf.len() {
            if self.get_pos(buf[j].move_) == pos {
              already_used = true;
              break;
            }
          }
          if already_used {
            continue;
          }

          best_pos = pos;
          best_policy = policy_probs[pos as usize] as f64;
        }
        if best_pos < 0 || best_policy < 0.0 {
          break;
        }

        let best_move: Loc = NNPos::pos_to_loc(best_pos, self.root_board.x_size, self.root_board.y_size, self.nn_x_len, self.nn_y_len);
        let data: AnalysisData = self.get_analysis_data_of_single_child(
          None,
          0,
          &mut scratch_locs,
          &mut scratch_values,
          best_move,
          best_policy,
          fpu_value,
          parent_utility,
          parent_win_loss_value,
          parent_score_mean,
          parent_score_stdev,
          parent_lead,
          max_pv_depth,
        );
        buf.push(data);
      }
    }
    // PORT NOTE: std::stable_sort with operator< (PORTING.md §5).
    buf.sort_by(cpp::cmp_from_lt(operator_lt));

    if duplicate_for_symmetries && self.search_params.root_symmetry_pruning && self.root_symmetries.len() > 1 {
      let mut new_buf: Vec<AnalysisData> = Vec::new();
      let mut is_done: BTreeSet<Loc> = BTreeSet::new();
      for i in 0..buf.len() {
        let data: &AnalysisData = &buf[i];
        for &symmetry in self.root_symmetries.iter() {
          let sym_move: Loc = SymmetryHelpers::get_sym_loc(data.move_, &self.root_board, symmetry);
          if is_done.contains(&sym_move) {
            continue;
          }
          let avoid_move_until_by_loc: &Vec<i32> =
            if self.root_pla == P_BLACK { &self.avoid_move_until_by_loc_black } else { &self.avoid_move_until_by_loc_white };
          if !avoid_move_until_by_loc.is_empty() && avoid_move_until_by_loc[sym_move as usize] > 0 {
            continue;
          }

          is_done.insert(sym_move);
          new_buf.push(data.clone());
          //Replace the fields that need to be adjusted for symmetry
          let new_data: &mut AnalysisData = new_buf.last_mut().expect("just pushed");
          new_data.move_ = sym_move;
          if symmetry != 0 {
            new_data.is_symmetry_of = data.move_;
          }
          new_data.symmetry = symmetry;
          for j in 0..new_data.pv.len() {
            new_data.pv[j] = SymmetryHelpers::get_sym_loc(new_data.pv[j], &self.root_board, symmetry);
          }
        }
      }
      *buf = new_buf;
    }

    for i in 0..buf.len() {
      buf[i].order = i as i32;
    }
  }

  /// upstream: Search::printPVForMove (searchresults.cpp)
  pub fn print_pv_for_move(&self, out: &mut dyn std::fmt::Write, n: Option<&SearchNode>, move_: Loc, max_depth: i32) {
    let mut buf: Vec<Loc> = Vec::new();
    let mut visits_buf: Vec<i64> = Vec::new();
    let mut edge_visits_buf: Vec<i64> = Vec::new();
    let mut scratch_locs: Vec<Loc> = Vec::new();
    let mut scratch_values: Vec<f64> = Vec::new();
    self.append_pv_for_move(&mut buf, &mut visits_buf, &mut edge_visits_buf, &mut scratch_locs, &mut scratch_values, n, move_, max_depth);
    for i in 0..buf.len() {
      if i > 0 {
        write!(out, " ").unwrap();
      }
      write!(out, "{}", Location::to_string(buf[i], &self.root_board)).unwrap();
    }
  }

  /// upstream: Search::printTree (searchresults.cpp)
  pub fn print_tree(&self, out: &mut dyn std::fmt::Write, node: Option<&SearchNode>, options: &PrintTreeOptions, mut perspective: Player) {
    let node: &SearchNode = match node {
      None => return,
      Some(n) => n,
    };
    let mut prefix: String = String::new();
    let mut data: AnalysisData;
    {
      let mut scratch_locs: Vec<Loc> = Vec::new();
      let mut scratch_values: Vec<f64> = Vec::new();
      //Use dummy values for parent
      let policy_prob: f64 = f64::NAN;
      let fpu_value: f64 = 0.0;
      let parent_utility: f64 = 0.0;
      let parent_win_loss_value: f64 = 0.0;
      let parent_score_mean: f64 = 0.0;
      let parent_score_stdev: f64 = 0.0;
      let parent_lead: f64 = 0.0;
      //Since we don't have an edge from another parent we are following, we just use the visits on the node itself as the edge visits.
      let edge_visits: i64 = node.stats.visits.get() /*acquire*/;
      data = self.get_analysis_data_of_single_child(
        Some(node),
        edge_visits,
        &mut scratch_locs,
        &mut scratch_values,
        Board::NULL_LOC,
        policy_prob,
        fpu_value,
        parent_utility,
        parent_win_loss_value,
        parent_score_mean,
        parent_score_stdev,
        parent_lead,
        options.max_pv_depth_,
      );
      data.weight_factor = f64::NAN;
    }
    perspective = if perspective != P_BLACK && perspective != P_WHITE { node.next_pla } else { perspective };
    self.print_tree_helper(out, Some(node), options, &mut prefix, 0, 0, &data, perspective);
  }

  /// upstream: Search::printTreeHelper (searchresults.cpp)
  pub(crate) fn print_tree_helper(
    &self,
    out: &mut dyn std::fmt::Write,
    n: Option<&SearchNode>,
    options: &PrintTreeOptions,
    prefix: &mut String,
    mut orig_visits: i64,
    depth: i32,
    data: &AnalysisData,
    perspective: Player,
  ) {
    let n: &SearchNode = match n {
      None => return,
      Some(n) => n,
    };

    let node: &SearchNode = n;

    let perspective_to_use: Player = if perspective != P_BLACK && perspective != P_WHITE { n.next_pla } else { perspective };
    let perspective_factor: f64 = if perspective_to_use == P_BLACK { -1.0 } else { 1.0 };

    if depth == 0 {
      orig_visits = data.num_visits;
    }

    //Output for this node
    {
      write!(out, "{}", prefix).unwrap();
      // PORT NOTE: `char buf[128]` + sprintf: each strprintf! below is one sprintf into buf, then `out << buf`.

      write!(out, ": ").unwrap();

      if data.child_visits > 0 {
        write!(out, "{}", strprintf!("T %6.2fc ", perspective_factor * data.utility * 100.0)).unwrap();
        write!(out, "{}", strprintf!("W %6.2fc ", perspective_factor * data.result_utility * 100.0)).unwrap();
        write!(
          out,
          "{}",
          strprintf!(
            "S %6.2fc (%+5.1f L %+5.1f) ",
            perspective_factor * data.score_utility * 100.0,
            perspective_factor * data.score_mean,
            perspective_factor * data.lead
          )
        )
        .unwrap();
      }

      // bool hasNNValue = false;
      // double nnResultValue;
      // double nnTotalValue;
      // lock.lock();
      // if(node.nnOutput != nullptr) {
      //   nnResultValue = getResultUtilityFromNN(*node.nnOutput);
      //   nnTotalValue = getUtilityFromNN(*node.nnOutput);
      //   hasNNValue = true;
      // }
      // lock.unlock();

      // if(hasNNValue) {
      //   sprintf(buf,"VW %6.2fc VS %6.2fc ", nnResultValue * 100.0, (nnTotalValue - nnResultValue) * 100.0);
      //   out << buf;
      // }
      // else {
      //   sprintf(buf,"VW ---.--c VS ---.--c ");
      //   out << buf;
      // }

      if depth > 0 && !data.lcb.is_nan() {
        write!(out, "{}", strprintf!("LCB %7.2fc ", perspective_factor * data.lcb * 100.0)).unwrap();
      }

      if !data.policy_prior.is_nan() {
        write!(out, "{}", strprintf!("P %5.2f%% ", data.policy_prior * 100.0)).unwrap();
      }
      if !data.weight_factor.is_nan() {
        write!(out, "{}", strprintf!("WF %5.1f ", data.weight_factor)).unwrap();
      }
      if data.play_selection_value >= 0.0 && depth > 0 {
        write!(out, "{}", strprintf!("PSV %7.0f ", data.play_selection_value)).unwrap();
      }

      if options.print_sqs_ {
        write!(
          out,
          "{}",
          strprintf!("SMSQ %5.1f USQ %7.5f W %6.2f WSQ %8.2f ", data.score_mean_sq_avg, data.utility_sq_avg, data.weight_sum, data.weight_sq_sum)
        )
        .unwrap();
      }

      if options.print_avg_shortterm_error_ {
        let wl_and_score_error: (f64, f64) = self.get_shallow_average_shortterm_wl_and_score_error(Some(node));
        write!(out, "{}", strprintf!("STWL %6.2fc STS %5.1f ", wl_and_score_error.0 * 100.0, wl_and_score_error.1)).unwrap();
      }

      // Using child visits here instead of edge visits because edge visits is at least
      // semi-reflected in WF and PSV.
      // PORT NOTE: "N %7" PRIu64: the int64_t is printed as unsigned.
      write!(out, "{}", strprintf!("N %7lu  --  ", data.child_visits as u64)).unwrap();

      self.print_pv_buf(out, &data.pv);
      writeln!(out).unwrap();
    }

    // PORT NOTE: `depth >= options.branch_.size()` and `depth >= options.maxDepth_ + options.branch_.size()`
    // compare in size_t (unsigned, 64-bit): the int operands convert and the sum wraps.
    let branch_len: u64 = options.branch_.len() as u64;
    if depth as u64 >= branch_len {
      if depth as u64 >= (options.max_depth_ as i64 as u64).wrapping_add(branch_len) {
        return;
      }
      if data.num_visits < options.min_visits_to_expand_ {
        return;
      }
      if (data.num_visits as f64) < orig_visits as f64 * options.min_visits_prop_to_expand_ {
        return;
      }
    }
    if (options.also_branch_ && depth == 0) || (!options.also_branch_ && depth as u64 == branch_len) {
      writeln!(
        out,
        "---{}({})---",
        PlayerIO::player_to_string(node.next_pla),
        if node.next_pla == perspective_to_use { "^" } else { "v" }
      )
      .unwrap();
    }

    let mut analysis_data: Vec<AnalysisData> = Vec::new();
    let duplicate_for_symmetries: bool = false;
    self.get_analysis_data_node(node, &mut analysis_data, 0, true, options.max_pv_depth_, duplicate_for_symmetries);

    let num_children: i32 = analysis_data.len() as i32;

    //Apply filtering conditions, but include children that don't match the filtering condition
    //but where there are children afterward that do, in case we ever use something more complex
    //than plain visits as a filter criterion. Do this by finding the last child that we want as the threshold.
    let mut last_idx_with_enough_visits: i32 = num_children - 1;
    loop {
      if last_idx_with_enough_visits <= 0 {
        break;
      }

      let child_visits: i64 = analysis_data[last_idx_with_enough_visits as usize].num_visits;
      let has_enough_visits: bool =
        child_visits >= options.min_visits_to_show_ && child_visits as f64 >= orig_visits as f64 * options.min_visits_prop_to_show_;
      if has_enough_visits {
        break;
      }
      last_idx_with_enough_visits -= 1;
    }

    let mut num_children_to_recurse_on: i32 = num_children;
    if options.max_children_to_show_ < num_children_to_recurse_on {
      num_children_to_recurse_on = options.max_children_to_show_;
    }
    if last_idx_with_enough_visits + 1 < num_children_to_recurse_on {
      num_children_to_recurse_on = last_idx_with_enough_visits + 1;
    }

    for i in 0..num_children {
      let child: Option<Rc<SearchNode>> = analysis_data[i as usize].node.map(|c| self.node_arena.get(c));
      let move_loc: Loc = analysis_data[i as usize].move_;

      if (depth as u64 >= branch_len && i < num_children_to_recurse_on)
        || ((depth as u64) < branch_len && move_loc == options.branch_[depth as usize])
        || ((depth as u64) < branch_len && options.also_branch_ && i < num_children_to_recurse_on)
      {
        let old_len: usize = prefix.len();
        let loc_str: String = Location::to_string(move_loc, &self.root_board);
        if loc_str == "pass" {
          *prefix += "pss";
        } else {
          *prefix += &loc_str;
        }
        *prefix += " ";
        while prefix.len() < old_len + 4 {
          *prefix += " ";
        }
        let mut next_depth: i32 = depth + 1;
        if (depth as u64) < branch_len && move_loc != options.branch_[depth as usize] {
          next_depth = options.branch_.len() as i32 + 1;
        }
        self.print_tree_helper(out, child.as_deref(), options, prefix, orig_visits, next_depth, &analysis_data[i as usize], perspective);
        prefix.truncate(old_len);
      }
    }
  }

  /// upstream: Search::getShallowAverageShorttermWLAndScoreError (searchresults.cpp)
  pub fn get_shallow_average_shortterm_wl_and_score_error(&self, node: Option<&SearchNode>) -> (f64, f64) {
    let root_holder: Option<Rc<SearchNode>>;
    let node: &SearchNode = match node {
      Some(n) => n,
      None => {
        root_holder = self.get_root_node();
        match &root_holder {
          None => return (0.0, 0.0),
          Some(r) => r,
        }
      }
    };
    if !self.nn_evaluator.supports_shortterm_error() {
      return (-1.0, -1.0);
    }
    let mut graph_path: HashSet<NodeIdx> = HashSet::new();
    let mut policy_probs_buf: Vec<f64> = vec![0.0; NNPos::MAX_NN_POLICY_SIZE as usize];
    let mut wl_error: f64 = 0.0;
    let mut score_error: f64 = 0.0;
    // Stop deepening when we hit a node whose proportion in the final average would be less than this.
    // Sublinear in visits so that the cost of this grows more slowly than overall search depth.
    let visits: i64 = node.stats.visits.get() /*acquire*/;
    let min_prop: f64 = 0.25 / math::powf(cpp::cpp_max(1.0, visits as f64), 0.625);
    let desired_prop: f64 = 1.0;
    self.get_shallow_average_shortterm_wl_and_score_error_helper(
      node,
      &mut graph_path,
      &mut policy_probs_buf,
      min_prop,
      desired_prop,
      &mut wl_error,
      &mut score_error,
    );
    (wl_error, score_error)
  }

  /// upstream: Search::getShallowAverageShorttermWLAndScoreErrorHelper (searchresults.cpp)
  pub(crate) fn get_shallow_average_shortterm_wl_and_score_error_helper(
    &self,
    node: &SearchNode,
    graph_path: &mut HashSet<NodeIdx>,
    policy_probs_buf: &mut [f64],
    min_prop: f64,
    desired_prop: f64,
    wl_error: &mut f64,
    score_error: &mut f64,
  ) {
    let nn_output: Rc<NNOutput> = match node.get_nn_output() {
      None => {
        // Accumulate nothing. This will be correct for terminal nodes, which have no uncertainty.
        // Not quite correct for multithreading, but no big deal, this value isn't used for anything critical
        // and currently isn't called while multithreaded. Yay code debt.
        return;
      }
      Some(o) => o,
    };

    if desired_prop < min_prop {
      // We don't track the average errors on nodes, so just use the error of this node's raw nn output.
      *wl_error += desired_prop * nn_output.shortterm_winloss_error as f64;
      *score_error += desired_prop * nn_output.shortterm_score_error as f64;
      return;
    }

    let inserted: bool = graph_path.insert(node.idx);
    // No insertion, node was already there, this means we hit a cycle in the graph
    if !inserted {
      //Just treat it as base case and immediately terminate.
      *wl_error += desired_prop * nn_output.shortterm_winloss_error as f64;
      *score_error += desired_prop * nn_output.shortterm_score_error as f64;
      return;
    }

    let children: SearchNodeChildrenReference = node.get_children();
    let children_capacity: i32 = children.get_capacity();

    let mut stats_buf: Vec<MoreNodeStats> = Vec::new();
    for i in 0..children_capacity {
      let child_pointer: &SearchChildPointer = children.index(i);
      let child: Rc<SearchNode> = match child_pointer.get_if_allocated() {
        None => break,
        Some(c) => self.node_arena.get(c),
      };
      let edge_visits: i64 = child_pointer.get_edge_visits();
      let move_loc: Loc = child_pointer.get_move_loc_relaxed();
      let mut stats: MoreNodeStats = MoreNodeStats::new();
      stats.stats = NodeStats::new_other(&child.stats);
      stats.self_utility = if node.next_pla == P_WHITE { stats.stats.utility_avg } else { -stats.stats.utility_avg };
      stats.weight_adjusted = stats.stats.get_child_weight(edge_visits);
      stats.prev_move_loc = move_loc;
      stats_buf.push(stats);
    }
    let num_children: i32 = stats_buf.len() as i32;

    // Find all children and compute weighting of the children based on their values
    {
      let mut total_child_weight: f64 = 0.0;
      for i in 0..num_children {
        total_child_weight += stats_buf[i as usize].weight_adjusted;
      }
      let policy_probs: &[f32; NNPos::MAX_NN_POLICY_SIZE as usize] = nn_output.get_policy_probs_maybe_noised();
      if self.search_params.use_noise_pruning {
        for i in 0..num_children {
          policy_probs_buf[i as usize] = cpp::cpp_max(1e-30, policy_probs[self.get_pos(stats_buf[i as usize].prev_move_loc) as usize] as f64);
        }
        total_child_weight = self.prune_noise_weight(&mut stats_buf, num_children, total_child_weight, policy_probs_buf);
      }
      let amount_to_subtract: f64 = 0.0;
      let amount_to_prune: f64 = 0.0;
      self.downweight_bad_children_and_normalize_weight(
        num_children,
        total_child_weight,
        total_child_weight,
        amount_to_subtract,
        amount_to_prune,
        &mut stats_buf,
      );
    }

    //What we actually weight the children by for averaging
    let mut relative_children_weight_sum: f64 = 0.0;
    //What the weights of the children sum to from the search.
    let mut children_weight_sum: f64 = 0.0;
    for i in 0..num_children {
      let child_weight: f64 = stats_buf[i as usize].weight_adjusted;
      relative_children_weight_sum += child_weight;
      children_weight_sum += child_weight;
    }
    let mut parent_nn_weight: f64 = self.compute_weight_from_nn_output(&nn_output);
    parent_nn_weight = cpp::cpp_max(parent_nn_weight, 1e-10);
    let desired_prop_from_children: f64 = desired_prop * children_weight_sum / (children_weight_sum + parent_nn_weight);
    let mut self_prop: f64 = desired_prop * parent_nn_weight / (children_weight_sum + parent_nn_weight);

    // In multithreading we may sometimes have children but with no weight at all yet, in that case just use parent alone.
    if desired_prop_from_children <= 0.0 || relative_children_weight_sum <= 0.0 {
      self_prop += desired_prop_from_children;
    } else {
      for i in 0..num_children {
        let child: Rc<SearchNode> = self.node_arena.get(children.index(i).get_if_allocated().expect("counted children are allocated"));
        let child_weight: f64 = stats_buf[i as usize].weight_adjusted;
        let desired_prop_from_child: f64 = child_weight / relative_children_weight_sum * desired_prop_from_children;
        self.get_shallow_average_shortterm_wl_and_score_error_helper(
          &child,
          graph_path,
          policy_probs_buf,
          min_prop,
          desired_prop_from_child,
          wl_error,
          score_error,
        );
      }
    }

    graph_path.remove(&node.idx);

    // Also add in the direct evaluation of this node.
    {
      *wl_error += self_prop * nn_output.shortterm_winloss_error as f64;
      *score_error += self_prop * nn_output.shortterm_score_error as f64;
    }
  }

  /// upstream: Search::getSharpScore (searchresults.cpp)
  pub fn get_sharp_score(&self, node: Option<&SearchNode>, ret: &mut f64) -> bool {
    let root_holder: Option<Rc<SearchNode>>;
    let node: &SearchNode = match node {
      Some(n) => n,
      None => {
        root_holder = self.get_root_node();
        match &root_holder {
          None => return false,
          Some(r) => r,
        }
      }
    };

    let visits: i64 = node.stats.visits.get() /*acquire*/;
    // Stop deepening when we hit a node whose proportion in the final average would be less than this.
    // Sublinear in visits so that the cost of this grows more slowly than overall search depth.
    let min_prop: f64 = 0.25 / math::powf(cpp::cpp_max(1.0, visits as f64), 0.5);
    let desired_prop: f64 = 1.0;

    // Store initial value so we can start accumulating
    *ret = 0.0;

    let mut graph_path: HashSet<NodeIdx> = HashSet::new();

    let mut policy_probs_buf: Vec<f64> = vec![0.0; NNPos::MAX_NN_POLICY_SIZE as usize];
    if Some(node.idx) != self.root_node {
      return self.get_sharp_score_helper(node, &mut graph_path, &mut policy_probs_buf, min_prop, desired_prop, ret);
    }

    let nn_output: Rc<NNOutput> = match node.get_nn_output() {
      None => return false,
      Some(o) => o,
    };

    let mut play_selection_values: Vec<f64> = Vec::new();
    let mut locs: Vec<Loc> = Vec::new(); // not used
    let allow_direct_policy_moves: bool = false;
    let always_compute_lcb: bool = false;
    let never_use_lcb: bool = true;
    let suc: bool = self.get_play_selection_values_node_lcb(
      node,
      &mut locs,
      &mut play_selection_values,
      None,
      1.0,
      allow_direct_policy_moves,
      always_compute_lcb,
      never_use_lcb,
      None,
      None,
    );
    // If there are no children, or otherwise values could not be computed, then fall back to the normal case
    if !suc {
      let mut values: ReportedSearchValues = ReportedSearchValues::new();
      if self.get_node_values(Some(node), &mut values) {
        *ret = values.expected_score;
        return true;
      }
      return false;
    }

    let num_children: i32 = play_selection_values.len() as i32;

    let children: SearchNodeChildrenReference = node.get_children();

    //What we actually weight the children by for averaging sharp score, sharper than the plain weight.
    let mut relative_children_weight_sum: f64 = 0.0;
    //What the weights of the children sum to from the search.
    let mut children_weight_sum: f64 = 0.0;
    for i in 0..num_children {
      let child_weight: f64 = play_selection_values[i as usize];
      relative_children_weight_sum += child_weight * child_weight * child_weight;
      children_weight_sum += child_weight;
    }
    let mut parent_nn_weight: f64 = self.compute_weight_from_nn_output(&nn_output);
    parent_nn_weight = cpp::cpp_max(parent_nn_weight, 1e-10);
    let desired_prop_from_children: f64 = desired_prop * children_weight_sum / (children_weight_sum + parent_nn_weight);
    let mut self_prop: f64 = desired_prop * parent_nn_weight / (children_weight_sum + parent_nn_weight);

    // In multithreading we may sometimes have children but with no weight at all yet, in that case just use parent alone.
    if desired_prop_from_children <= 0.0 || relative_children_weight_sum <= 0.0 {
      self_prop += desired_prop_from_children;
    } else {
      graph_path.insert(node.idx);

      for i in 0..num_children {
        let child: Rc<SearchNode> = self.node_arena.get(children.index(i).get_if_allocated().expect("children below numChildren are allocated"));
        let child_weight: f64 = play_selection_values[i as usize];
        let desired_prop_from_child: f64 = child_weight * child_weight * child_weight / relative_children_weight_sum * desired_prop_from_children;
        let accumulated: bool = self.get_sharp_score_helper(&child, &mut graph_path, &mut policy_probs_buf, min_prop, desired_prop_from_child, ret);
        if !accumulated {
          self_prop += desired_prop_from_child;
        }
      }

      graph_path.remove(&node.idx);
    }

    // Also add in the direct evaluation of this node.
    {
      let score_mean: f64 = nn_output.white_score_mean as f64;
      // cout << "Accumulating " << scoreMean << " " << selfProp << endl;
      *ret += score_mean * self_prop;
    }
    true
  }

  /// upstream: Search::getSharpScoreHelper (searchresults.cpp)
  pub(crate) fn get_sharp_score_helper(
    &self,
    node: &SearchNode,
    graph_path: &mut HashSet<NodeIdx>,
    policy_probs_buf: &mut [f64],
    min_prop: f64,
    desired_prop: f64,
    ret: &mut f64,
  ) -> bool {
    let nn_output: Option<Rc<NNOutput>> = node.get_nn_output();
    if nn_output.is_none() || desired_prop < min_prop {
      let stats: NodeStats = NodeStats::new_other(&node.stats);
      if stats.visits <= 0 {
        return false;
      }
      // cout << "Accumulating " << stats.scoreMeanAvg << " " << desiredProp << endl;
      *ret += stats.score_mean_avg * desired_prop;
      return true;
    }
    let nn_output: Rc<NNOutput> = nn_output.expect("checked above");

    let children: SearchNodeChildrenReference = node.get_children();
    let children_capacity: i32 = children.get_capacity();

    if children_capacity <= 0 {
      let score_mean: f64 = nn_output.white_score_mean as f64;
      // cout << "Accumulating " << scoreMean << " " << desiredProp << endl;
      *ret += score_mean * desired_prop;
      return true;
    }

    let inserted: bool = graph_path.insert(node.idx);
    // No insertion, node was already there, this means we hit a cycle in the graph
    if !inserted {
      // Just treat it as base case and immediately terminate.
      let score_mean: f64 = nn_output.white_score_mean as f64;
      // cout << "Accumulating " << scoreMean << " " << desiredProp << endl;
      *ret += score_mean * desired_prop;
      return true;
    }

    let mut stats_buf: Vec<MoreNodeStats> = Vec::new();
    for i in 0..children_capacity {
      let child_pointer: &SearchChildPointer = children.index(i);
      let child: Rc<SearchNode> = match child_pointer.get_if_allocated() {
        None => break,
        Some(c) => self.node_arena.get(c),
      };
      let edge_visits: i64 = child_pointer.get_edge_visits();
      let move_loc: Loc = child_pointer.get_move_loc_relaxed();
      let mut stats: MoreNodeStats = MoreNodeStats::new();
      stats.stats = NodeStats::new_other(&child.stats);
      stats.self_utility = if node.next_pla == P_WHITE { stats.stats.utility_avg } else { -stats.stats.utility_avg };
      stats.weight_adjusted = stats.stats.get_child_weight(edge_visits);
      stats.prev_move_loc = move_loc;
      stats_buf.push(stats);
    }
    let num_children: i32 = stats_buf.len() as i32;

    //Find all children and compute weighting of the children based on their values
    {
      let mut total_child_weight: f64 = 0.0;
      for i in 0..num_children {
        total_child_weight += stats_buf[i as usize].weight_adjusted;
      }
      let policy_probs: &[f32; NNPos::MAX_NN_POLICY_SIZE as usize] = nn_output.get_policy_probs_maybe_noised();
      if self.search_params.use_noise_pruning {
        for i in 0..num_children {
          policy_probs_buf[i as usize] = cpp::cpp_max(1e-30, policy_probs[self.get_pos(stats_buf[i as usize].prev_move_loc) as usize] as f64);
        }
        total_child_weight = self.prune_noise_weight(&mut stats_buf, num_children, total_child_weight, policy_probs_buf);
      }
      let amount_to_subtract: f64 = 0.0;
      let amount_to_prune: f64 = 0.0;
      self.downweight_bad_children_and_normalize_weight(
        num_children,
        total_child_weight,
        total_child_weight,
        amount_to_subtract,
        amount_to_prune,
        &mut stats_buf,
      );
    }

    // What we actually weight the children by for averaging sharp score, sharper than the plain weight.
    let mut relative_children_weight_sum: f64 = 0.0;
    // What the weights of the children sum to from the search.
    let mut children_weight_sum: f64 = 0.0;
    for i in 0..num_children {
      if stats_buf[i as usize].stats.visits <= 0 {
        continue;
      }
      let child_weight: f64 = stats_buf[i as usize].weight_adjusted;
      relative_children_weight_sum += child_weight * child_weight * child_weight;
      children_weight_sum += child_weight;
    }
    let mut parent_nn_weight: f64 = self.compute_weight_from_nn_output(&nn_output);
    parent_nn_weight = cpp::cpp_max(parent_nn_weight, 1e-10);
    let desired_prop_from_children: f64 = desired_prop * children_weight_sum / (children_weight_sum + parent_nn_weight);
    let mut self_prop: f64 = desired_prop * parent_nn_weight / (children_weight_sum + parent_nn_weight);

    // In multithreading we may sometimes have children but with no weight at all yet, in that case just use parent alone.
    if desired_prop_from_children <= 0.0 || relative_children_weight_sum <= 0.0 {
      self_prop += desired_prop_from_children;
    } else {
      for i in 0..num_children {
        let child: Rc<SearchNode> = self.node_arena.get(children.index(i).get_if_allocated().expect("counted children are allocated"));
        let child_weight: f64 = stats_buf[i as usize].weight_adjusted;
        let desired_prop_from_child: f64 = child_weight * child_weight * child_weight / relative_children_weight_sum * desired_prop_from_children;
        let accumulated: bool = self.get_sharp_score_helper(&child, graph_path, policy_probs_buf, min_prop, desired_prop_from_child, ret);
        if !accumulated {
          self_prop += desired_prop_from_child;
        }
      }
    }

    graph_path.remove(&node.idx);

    // Also add in the direct evaluation of this node.
    {
      let score_mean: f64 = nn_output.white_score_mean as f64;
      // cout << "Accumulating " << scoreMean << " " << selfProp << endl;
      *ret += score_mean * self_prop;
    }
    true
  }

  /// upstream: Search::getAverageTreeOwnership(const SearchNode*) (searchresults.cpp)
  //Get the ownership map averaged throughout the search tree.
  //Must have ownership present on all neural net evals.
  //Safe to call DURING search, but NOT necessarily safe to call multithreadedly when updating the root position
  //or changing parameters or clearing search.
  //If node is not provided, defaults to using the root node.
  /// PORT NOTE: throws -> Result. With no node and no root, upstream dereferences NULL (UB) once past the
  /// alwaysIncludeOwnerMap check; here a panic there.
  pub fn get_average_tree_ownership(&self, node: Option<&SearchNode>) -> Result<Vec<f64>, KataError> {
    let root_holder: Option<Rc<SearchNode>>;
    let node: Option<&SearchNode> = match node {
      Some(n) => Some(n),
      None => {
        root_holder = self.get_root_node();
        root_holder.as_deref()
      }
    };
    if !self.always_include_owner_map {
      return Err(KataError::StringError("Called Search::getAverageTreeOwnership when alwaysIncludeOwnerMap is false".to_string()));
    }
    let node: &SearchNode = node.expect("getAverageTreeOwnership: upstream dereferences a NULL root (UB)");
    let mut vec: Vec<f64> = vec![0.0; (self.nn_x_len * self.nn_y_len) as usize];
    let nn_area: i32 = self.nn_x_len * self.nn_y_len;
    let mut accumulate = |ownership: &[f32], self_prop: f64| {
      for pos in 0..nn_area {
        vec[pos as usize] += self_prop * ownership[pos as usize] as f64;
      }
    };
    let visits: i64 = node.stats.visits.get() /*acquire*/;
    //Stop deepening when we hit a node whose proportion in the final average would be less than this.
    //Sublinear in visits so that the cost of this grows more slowly than overall search depth.
    let min_prop: f64 = 0.5 / math::powf(cpp::cpp_max(1.0, visits as f64), 0.75);
    //Entirely drop a node with weight less than this
    let prune_prop: f64 = min_prop * 0.01;
    let mut graph_path: HashSet<NodeIdx> = HashSet::new();
    self.traverse_tree_for_ownership(min_prop, prune_prop, 1.0, Some(node), &mut graph_path, &mut accumulate);
    Ok(vec)
  }

  /// upstream: Search::getAverageAndStandardDeviationTreeOwnership(const SearchNode*) (searchresults.cpp)
  /// PORT NOTE: with no node and no root, upstream dereferences NULL (UB); here a panic.
  pub fn get_average_and_standard_deviation_tree_ownership(&self, node: Option<&SearchNode>) -> (Vec<f64>, Vec<f64>) {
    let root_holder: Option<Rc<SearchNode>>;
    let node: &SearchNode = match node {
      Some(n) => n,
      None => {
        root_holder = self.get_root_node();
        root_holder.as_deref().expect("getAverageAndStandardDeviationTreeOwnership: upstream dereferences a NULL root (UB)")
      }
    };
    let mut average: Vec<f64> = vec![0.0; (self.nn_x_len * self.nn_y_len) as usize];
    let mut stdev: Vec<f64> = vec![0.0; (self.nn_x_len * self.nn_y_len) as usize];
    let nn_area: i32 = self.nn_x_len * self.nn_y_len;
    {
      let mut accumulate = |ownership: &[f32], self_prop: f64| {
        for pos in 0..nn_area {
          let value: f64 = ownership[pos as usize] as f64;
          average[pos as usize] += self_prop * value;
          stdev[pos as usize] += self_prop * value * value;
        }
      };
      let visits: i64 = node.stats.visits.get() /*acquire*/;
      // Stop deepening when we hit a node whose proportion in the final average would be less than this.
      // Sublinear in visits so that the cost of this grows more slowly than overall search depth.
      let min_prop: f64 = 0.5 / math::powf(cpp::cpp_max(1.0, visits as f64), 0.75);
      // Entirely drop a node with weight less than this
      let prune_prop: f64 = min_prop * 0.01;
      let mut graph_path: HashSet<NodeIdx> = HashSet::new();
      self.traverse_tree_for_ownership(min_prop, prune_prop, 1.0, Some(node), &mut graph_path, &mut accumulate);
    }
    for pos in 0..nn_area {
      let avg: f64 = average[pos as usize];
      stdev[pos as usize] = cpp::cpp_max(stdev[pos as usize] - avg * avg, 0.0).sqrt();
    }
    (average, stdev)
  }

  /// upstream: Search::traverseTreeForOwnership (searchresults.cpp)
  // Returns true if anything was accumulated, false otherwise.
  pub(crate) fn traverse_tree_for_ownership<F: FnMut(&[f32], f64)>(
    &self,
    min_prop: f64,
    prune_prop: f64,
    desired_prop: f64,
    node: Option<&SearchNode>,
    graph_path: &mut HashSet<NodeIdx>,
    accumulate: &mut F,
  ) -> bool {
    let node: &SearchNode = match node {
      None => return false,
      Some(n) => n,
    };

    let nn_output: Rc<NNOutput> = match node.get_nn_output() {
      None => return false,
      Some(o) => o,
    };

    // Base case
    if desired_prop < min_prop {
      let owner_map: &Vec<f32> = nn_output.white_owner_map.as_ref().expect("ownership present on all evals (upstream asserts)");
      accumulate(owner_map, desired_prop);
      return true;
    }

    let children: SearchNodeChildrenReference = node.get_children();
    let children_capacity: i32 = children.get_capacity();

    if children_capacity <= 0 {
      let owner_map: &Vec<f32> = nn_output.white_owner_map.as_ref().expect("ownership present on all evals (upstream asserts)");
      accumulate(owner_map, desired_prop);
      return true;
    }

    let inserted: bool = graph_path.insert(node.idx);
    // No insertion, node was already there, this means we hit a cycle in the graph
    if !inserted {
      //Just treat it as base case and immediately terminate.
      let owner_map: &Vec<f32> = nn_output.white_owner_map.as_ref().expect("ownership present on all evals (upstream asserts)");
      accumulate(owner_map, desired_prop);
      return true;
    }

    let self_prop: f64;
    let parent_nn_weight: f64 = self.compute_weight_from_nn_output(&nn_output);
    if children_capacity <= SearchChildrenSizes::SIZE0TOTAL {
      let mut child_weight_buf: [f64; SearchChildrenSizes::SIZE0TOTAL as usize] = [0.0; SearchChildrenSizes::SIZE0TOTAL as usize];
      self_prop = self.traverse_tree_for_ownership_children(
        min_prop,
        prune_prop,
        desired_prop,
        parent_nn_weight,
        children,
        &mut child_weight_buf,
        children_capacity,
        graph_path,
        accumulate,
      );
    } else {
      let mut child_weight_buf: Vec<f64> = vec![0.0; children_capacity as usize];
      self_prop = self.traverse_tree_for_ownership_children(
        min_prop,
        prune_prop,
        desired_prop,
        parent_nn_weight,
        children,
        &mut child_weight_buf,
        children_capacity,
        graph_path,
        accumulate,
      );
    }

    graph_path.remove(&node.idx);

    let owner_map: &Vec<f32> = nn_output.white_owner_map.as_ref().expect("ownership present on all evals (upstream asserts)");
    accumulate(owner_map, self_prop);
    true
  }

  /// upstream: Search::traverseTreeForOwnershipChildren (searchresults.cpp)
  // Returns the prop that the parent node should be weighted.
  // Not guaranteed to be <= the parent's weightsum due to multithreading.
  pub(crate) fn traverse_tree_for_ownership_children<F: FnMut(&[f32], f64)>(
    &self,
    min_prop: f64,
    prune_prop: f64,
    desired_prop: f64,
    mut parent_nn_weight: f64,
    children: SearchNodeChildrenReference<'_>,
    child_weight_buf: &mut [f64],
    children_capacity: i32,
    graph_path: &mut HashSet<NodeIdx>,
    accumulate: &mut F,
  ) -> f64 {
    let mut num_children: i32 = 0;
    for i in 0..children_capacity {
      let child_pointer: &SearchChildPointer = children.index(i);
      let child: Rc<SearchNode> = match child_pointer.get_if_allocated() {
        None => break,
        Some(c) => self.node_arena.get(c),
      };
      let edge_visits: i64 = child_pointer.get_edge_visits();
      let child_weight: f64 = child.stats.get_child_weight(edge_visits);
      child_weight_buf[i as usize] = child_weight;
      num_children += 1;
    }

    // What we actually weight the children by for averaging ownership, sharper than the plain weight.
    let mut relative_children_weight_sum: f64 = 0.0;
    // What the weights of the children sum to from the search.
    let mut children_weight_sum: f64 = 0.0;
    for i in 0..num_children {
      let child_weight: f64 = child_weight_buf[i as usize];
      relative_children_weight_sum += child_weight * child_weight;
      children_weight_sum += child_weight;
    }

    // Just in case
    parent_nn_weight = cpp::cpp_max(parent_nn_weight, 1e-10);
    let desired_prop_from_children: f64 = desired_prop * children_weight_sum / (children_weight_sum + parent_nn_weight);
    let mut self_prop: f64 = desired_prop * parent_nn_weight / (children_weight_sum + parent_nn_weight);

    // Recurse
    // In multithreading we may sometimes have children but with no weight at all yet, in that case just use parent alone.
    if desired_prop_from_children <= 0.0 || relative_children_weight_sum <= 0.0 {
      self_prop += desired_prop_from_children;
    } else {
      for i in 0..num_children {
        let child_weight: f64 = child_weight_buf[i as usize];
        let child: Rc<SearchNode> = self.node_arena.get(children.index(i).get_if_allocated().expect("counted children are allocated"));
        let desired_prop_from_child: f64 = child_weight * child_weight / relative_children_weight_sum * desired_prop_from_children;
        if desired_prop_from_child < prune_prop {
          self_prop += desired_prop_from_child;
        } else {
          let accumulated: bool = self.traverse_tree_for_ownership(min_prop, prune_prop, desired_prop_from_child, Some(&child), graph_path, accumulate);
          if !accumulated {
            self_prop += desired_prop_from_child;
          }
        }
      }
    }

    self_prop
  }

  /// upstream: Search::getAverageTreeOwnership(const Player, const SearchNode*, int) (searchresults.cpp)
  //Same, but applies symmetry and perspective
  /// PORT NOTE: name from PORTING.md Appendix B; throws -> Result.
  pub fn get_average_tree_ownership_perspective(&self, perspective: Player, node: Option<&SearchNode>, symmetry: i32) -> Result<Vec<f64>, KataError> {
    let ownership: Vec<f64> = self.get_average_tree_ownership(node)?;
    let board: &Board = &self.root_board;
    let mut ownership_to_output: Vec<f64> = vec![0.0; (board.y_size * board.x_size) as usize];

    for y in 0..board.y_size {
      for x in 0..board.x_size {
        let pos: i32 = NNPos::xy_to_pos(x, y, self.nn_x_len);
        let sym_loc: Loc = SymmetryHelpers::get_sym_loc_x_y(x, y, board, symmetry);
        let sym_pos: i32 = Location::get_y(sym_loc, board.x_size) * board.x_size + Location::get_x(sym_loc, board.x_size);
        debug_assert!(sym_pos >= 0 && sym_pos < board.y_size * board.x_size);

        let o: f64;
        if perspective == P_BLACK || (perspective != P_BLACK && perspective != P_WHITE && self.root_pla == P_BLACK) {
          o = -ownership[pos as usize];
        } else {
          o = ownership[pos as usize];
        }
        // Round to 10^-6 to limit the size of output.
        // No guarantees that the serializer actually outputs something of this length rather than longer due to float wonkiness, but it should usually be true.
        ownership_to_output[sym_pos as usize] = Global::round_static(o, 1000000.0);
      }
    }
    Ok(ownership_to_output)
  }

  /// upstream: Search::getAverageAndStandardDeviationTreeOwnership(const Player, const SearchNode*, int) (searchresults.cpp)
  /// PORT NOTE: name from PORTING.md Appendix B.
  pub fn get_average_and_standard_deviation_tree_ownership_perspective(
    &self,
    perspective: Player,
    node: Option<&SearchNode>,
    symmetry: i32,
  ) -> (Vec<f64>, Vec<f64>) {
    let ownership_average_and_stdev: (Vec<f64>, Vec<f64>) = self.get_average_and_standard_deviation_tree_ownership(node);
    let board: &Board = &self.root_board;
    let ownership: &Vec<f64> = &ownership_average_and_stdev.0;
    let ownership_stdev: &Vec<f64> = &ownership_average_and_stdev.1;
    let mut ownership_to_output: Vec<f64> = vec![0.0; (board.y_size * board.x_size) as usize];
    let mut ownership_stdev_to_output: Vec<f64> = vec![0.0; (board.y_size * board.x_size) as usize];

    for y in 0..board.y_size {
      for x in 0..board.x_size {
        let pos: i32 = NNPos::xy_to_pos(x, y, self.nn_x_len);
        let sym_loc: Loc = SymmetryHelpers::get_sym_loc_x_y(x, y, board, symmetry);
        let sym_pos: i32 = Location::get_y(sym_loc, board.x_size) * board.x_size + Location::get_x(sym_loc, board.x_size);
        debug_assert!(sym_pos >= 0 && sym_pos < board.y_size * board.x_size);

        let o: f64;
        if perspective == P_BLACK || (perspective != P_BLACK && perspective != P_WHITE && self.root_pla == P_BLACK) {
          o = -ownership[pos as usize];
        } else {
          o = ownership[pos as usize];
        }
        // Round to 10^-6 to limit the size of output.
        // No guarantees that the serializer actually outputs something of this length rather than longer due to float wonkiness, but it should usually be true.
        ownership_to_output[sym_pos as usize] = Global::round_static(o, 1000000.0);
        ownership_stdev_to_output[sym_pos as usize] = Global::round_static(ownership_stdev[pos as usize], 1000000.0);
      }
    }
    (ownership_to_output, ownership_stdev_to_output)
  }

  /// upstream: Search::getAnalysisJson (searchresults.cpp)
  //Fill json with analysis engine format information about search results
  /// PORT NOTE: `nlohmann::json& ret` is a serde_json::Value (serialize with port::nlohmann::dump,
  /// PORTING.md §9); `json[key]` on null/object is `Value`'s IndexMut, which also turns null into an
  /// object. throws (getAverageTreeOwnership) -> Result.
  pub fn get_analysis_json(
    &self,
    perspective: Player,
    analysis_pv_len: i32,
    prevent_encore: bool,
    include_policy: bool,
    include_ownership: bool,
    include_ownership_stdev: bool,
    include_moves_ownership: bool,
    include_moves_ownership_stdev: bool,
    include_pv_visits: bool,
    include_no_result_value: bool,
    ret: &mut Value,
  ) -> Result<bool, KataError> {
    let mut buf: Vec<AnalysisData> = Vec::new();
    const MIN_MOVES: i32 = 0;
    const OUTPUT_PRECISION: i32 = 8;

    let board: &Board = &self.root_board;
    let hist: &BoardHistory = &self.root_history;
    let duplicate_for_symmetries: bool = true;
    self.get_analysis_data(&mut buf, MIN_MOVES, false, analysis_pv_len, duplicate_for_symmetries);

    let mut nn_output: Option<Rc<NNOutput>> = None;
    let mut human_output: Option<Rc<NNOutput>> = None;
    if let Some(root_node) = self.get_root_node() {
      nn_output = root_node.get_nn_output();
      human_output = root_node.get_human_output();
    }

    // Stats for all the individual moves
    let mut move_infos: Vec<Value> = Vec::new();
    for i in 0..buf.len() {
      let data: &AnalysisData = &buf[i];
      let mut winrate: f64 = 0.5 * (1.0 + data.win_loss_value);
      let mut utility: f64 = data.utility;
      let mut lcb: f64 = PlayUtils::get_hacked_lcb_for_winrate(self, data, self.root_pla);
      let mut utility_lcb: f64 = data.lcb;
      let mut score_mean: f64 = data.score_mean;
      let mut lead: f64 = data.lead;
      if perspective == P_BLACK || (perspective != P_BLACK && perspective != P_WHITE && self.root_pla == P_BLACK) {
        winrate = 1.0 - winrate;
        lcb = 1.0 - lcb;
        utility = -utility;
        score_mean = -score_mean;
        lead = -lead;
        utility_lcb = -utility_lcb;
      }

      let mut move_info: Value = Value::Null;
      move_info["move"] = Value::from(Location::to_string(data.move_, board));
      move_info["visits"] = Value::from(data.child_visits);
      move_info["weight"] = Value::from(Global::round_dynamic(data.child_weight_sum, OUTPUT_PRECISION));
      move_info["utility"] = Value::from(Global::round_dynamic(utility, OUTPUT_PRECISION));
      move_info["winrate"] = Value::from(Global::round_dynamic(winrate, OUTPUT_PRECISION));
      // We report lead for scoreMean here so that a bunch of legacy tools that use KataGo use lead instead, which
      // is usually a better field for user applications. We report scoreMean instead as scoreSelfplay
      move_info["scoreMean"] = Value::from(Global::round_dynamic(lead, OUTPUT_PRECISION));
      move_info["scoreSelfplay"] = Value::from(Global::round_dynamic(score_mean, OUTPUT_PRECISION));
      move_info["scoreLead"] = Value::from(Global::round_dynamic(lead, OUTPUT_PRECISION));
      move_info["scoreStdev"] = Value::from(Global::round_dynamic(data.score_stdev, OUTPUT_PRECISION));
      if include_no_result_value {
        move_info["noResultValue"] = Value::from(Global::round_dynamic(data.no_result_value, OUTPUT_PRECISION));
      }
      move_info["prior"] = Value::from(Global::round_dynamic(data.policy_prior, OUTPUT_PRECISION));
      if let Some(human_output) = &human_output {
        move_info["humanPrior"] = Value::from(Global::round_dynamic(
          cpp::cpp_max(0.0, human_output.policy_probs[self.get_pos(data.move_) as usize] as f64),
          OUTPUT_PRECISION,
        ));
      }
      move_info["lcb"] = Value::from(Global::round_dynamic(lcb, OUTPUT_PRECISION));
      move_info["utilityLcb"] = Value::from(Global::round_dynamic(utility_lcb, OUTPUT_PRECISION));
      move_info["order"] = Value::from(data.order);
      if data.is_symmetry_of != Board::NULL_LOC {
        move_info["isSymmetryOf"] = Value::from(Location::to_string(data.is_symmetry_of, board));
      }
      move_info["edgeVisits"] = Value::from(data.num_visits);
      move_info["edgeWeight"] = Value::from(Global::round_dynamic(data.weight_sum, OUTPUT_PRECISION));
      move_info["playSelectionValue"] = Value::from(Global::round_dynamic(data.play_selection_value, OUTPUT_PRECISION));

      let mut pv: Vec<Value> = Vec::new();
      let pv_len: i32 =
        if prevent_encore && data.pv_contains_pass() { data.get_pv_len_up_to_phase_end(board, hist, self.root_pla) } else { data.pv.len() as i32 };
      for j in 0..pv_len {
        pv.push(Value::from(Location::to_string(data.pv[j as usize], board)));
      }
      move_info["pv"] = Value::Array(pv);

      if include_pv_visits {
        // PORT NOTE: size_t >= int compares in size_t.
        testAssert!(data.pv_visits.len() as u64 >= pv_len as i64 as u64);
        let mut pv_visits: Vec<Value> = Vec::new();
        for j in 0..pv_len {
          pv_visits.push(Value::from(data.pv_visits[j as usize]));
        }
        move_info["pvVisits"] = Value::Array(pv_visits);

        testAssert!(data.pv_edge_visits.len() as u64 >= pv_len as i64 as u64);
        let mut pv_edge_visits: Vec<Value> = Vec::new();
        for j in 0..pv_len {
          pv_edge_visits.push(Value::from(data.pv_edge_visits[j as usize]));
        }
        move_info["pvEdgeVisits"] = Value::Array(pv_edge_visits);
      }

      let data_node: Option<Rc<SearchNode>> = data.node.map(|n| self.node_arena.get(n));
      if include_moves_ownership && include_moves_ownership_stdev {
        let ownership_and_stdev: (Vec<f64>, Vec<f64>) =
          self.get_average_and_standard_deviation_tree_ownership_perspective(perspective, data_node.as_deref(), data.symmetry);
        move_info["ownership"] = json_of_doubles(&ownership_and_stdev.0);
        move_info["ownershipStdev"] = json_of_doubles(&ownership_and_stdev.1);
      } else if include_moves_ownership_stdev {
        let ownership_and_stdev: (Vec<f64>, Vec<f64>) =
          self.get_average_and_standard_deviation_tree_ownership_perspective(perspective, data_node.as_deref(), data.symmetry);
        move_info["ownershipStdev"] = json_of_doubles(&ownership_and_stdev.1);
      } else if include_moves_ownership {
        move_info["ownership"] = json_of_doubles(&self.get_average_tree_ownership_perspective(perspective, data_node.as_deref(), data.symmetry)?);
      }

      move_infos.push(move_info);
    }
    ret["moveInfos"] = Value::Array(move_infos);

    // Stats for root position
    {
      let mut root_vals: ReportedSearchValues = ReportedSearchValues::new();
      let suc: bool = self.get_pruned_root_values(&mut root_vals);
      if !suc {
        return Ok(false);
      }

      let winloss: f64 = root_vals.win_loss_value;
      let score_mean: f64 = root_vals.expected_score;
      let lead: f64 = root_vals.lead;
      let utility: f64 = root_vals.utility;
      let flip_factor: f64 =
        if perspective == P_BLACK || (perspective != P_BLACK && perspective != P_WHITE && self.root_pla == P_BLACK) { -1.0 } else { 1.0 };

      let mut root_info: Value = Value::Null;
      root_info["visits"] = Value::from(root_vals.visits);
      root_info["weight"] = Value::from(root_vals.weight);
      root_info["winrate"] = Value::from(Global::round_dynamic(0.5 + 0.5 * winloss * flip_factor, OUTPUT_PRECISION));
      root_info["scoreSelfplay"] = Value::from(Global::round_dynamic(score_mean * flip_factor, OUTPUT_PRECISION));
      root_info["scoreLead"] = Value::from(Global::round_dynamic(lead * flip_factor, OUTPUT_PRECISION));
      root_info["scoreStdev"] = Value::from(Global::round_dynamic(root_vals.expected_score_stdev, OUTPUT_PRECISION));
      root_info["utility"] = Value::from(Global::round_dynamic(utility * flip_factor, OUTPUT_PRECISION));

      if let Some(nn_output) = &nn_output {
        // PORT NOTE: float operands stay float until they meet a double (PORTING.md §6 item 1).
        root_info["rawWinrate"] =
          Value::from(Global::round_dynamic(0.5 + 0.5 * (nn_output.white_win_prob - nn_output.white_loss_prob) as f64 * flip_factor, OUTPUT_PRECISION));
        root_info["rawLead"] = Value::from(Global::round_dynamic(nn_output.white_lead as f64 * flip_factor, OUTPUT_PRECISION));
        root_info["rawScoreSelfplay"] = Value::from(Global::round_dynamic(nn_output.white_score_mean as f64 * flip_factor, OUTPUT_PRECISION));
        let wsm: f64 = nn_output.white_score_mean as f64;
        root_info["rawScoreSelfplayStdev"] =
          Value::from(Global::round_dynamic(cpp::cpp_max(0.0, nn_output.white_score_mean_sq as f64 - wsm * wsm).sqrt(), OUTPUT_PRECISION));
        root_info["rawNoResultProb"] = Value::from(Global::round_dynamic(nn_output.white_no_result_prob as f64, OUTPUT_PRECISION));
        root_info["rawStWrError"] = Value::from(Global::round_dynamic(nn_output.shortterm_winloss_error as f64 * 0.5, OUTPUT_PRECISION));
        root_info["rawStScoreError"] = Value::from(Global::round_dynamic(nn_output.shortterm_score_error as f64, OUTPUT_PRECISION));
        root_info["rawVarTimeLeft"] = Value::from(Global::round_dynamic(nn_output.var_time_left as f64, OUTPUT_PRECISION));
      }
      if let Some(human_output) = &human_output {
        root_info["humanWinrate"] = Value::from(Global::round_dynamic(
          0.5 + 0.5 * (human_output.white_win_prob - human_output.white_loss_prob) as f64 * flip_factor,
          OUTPUT_PRECISION,
        ));
        root_info["humanScoreMean"] = Value::from(Global::round_dynamic(human_output.white_score_mean as f64 * flip_factor, OUTPUT_PRECISION));
        let wsm: f64 = human_output.white_score_mean as f64;
        root_info["humanScoreStdev"] =
          Value::from(Global::round_dynamic(cpp::cpp_max(0.0, human_output.white_score_mean_sq as f64 - wsm * wsm).sqrt(), OUTPUT_PRECISION));
        root_info["humanStWrError"] = Value::from(Global::round_dynamic(human_output.shortterm_winloss_error as f64 * 0.5, OUTPUT_PRECISION));
        root_info["humanStScoreError"] = Value::from(Global::round_dynamic(human_output.shortterm_score_error as f64, OUTPUT_PRECISION));
      }

      let mut this_hash: Hash128 = Hash128::default();
      let mut sym_hash: Hash128 = Hash128::default();
      for symmetry in 0..SymmetryHelpers::NUM_SYMMETRIES {
        // PORT NOTE: getSymBoard cannot throw here (the root board's sizes are valid, transposed or not).
        let sym_board: Board = SymmetryHelpers::get_sym_board(board, symmetry).expect("upstream: getSymBoard cannot throw for a valid board");
        let hash: Hash128 = sym_board.get_sit_hash_with_simple_ko(self.root_pla);
        if symmetry == 0 {
          this_hash = hash;
          sym_hash = hash;
        } else {
          if hash < sym_hash {
            sym_hash = hash;
          }
        }
      }
      root_info["thisHash"] = Value::from(Global::uint64_to_hex_string(this_hash.hash1) + &Global::uint64_to_hex_string(this_hash.hash0));
      root_info["symHash"] = Value::from(Global::uint64_to_hex_string(sym_hash.hash1) + &Global::uint64_to_hex_string(sym_hash.hash0));
      root_info["currentPlayer"] = Value::from(PlayerIO::player_to_string_short(self.root_pla));

      ret["rootInfo"] = root_info;
    }

    // Raw policy prior
    if include_policy {
      {
        let mut policy_probs: Box<[f32; NNPos::MAX_NN_POLICY_SIZE as usize]> = Box::new([0.0; NNPos::MAX_NN_POLICY_SIZE as usize]);
        let suc: bool = self.get_policy(&mut policy_probs);
        if !suc {
          return Ok(false);
        }
        let mut policy: Vec<Value> = Vec::new();
        for y in 0..board.y_size {
          for x in 0..board.x_size {
            let pos: i32 = NNPos::xy_to_pos(x, y, self.nn_x_len);
            policy.push(Value::from(Global::round_dynamic(policy_probs[pos as usize] as f64, OUTPUT_PRECISION)));
          }
        }

        let pass_pos: i32 = NNPos::loc_to_pos(Board::PASS_LOC, board.x_size, self.nn_x_len, self.nn_y_len);
        policy.push(Value::from(Global::round_dynamic(policy_probs[pass_pos as usize] as f64, OUTPUT_PRECISION)));
        ret["policy"] = Value::Array(policy);
      }

      if let Some(human_output) = &human_output {
        let policy_probs: &[f32; NNPos::MAX_NN_POLICY_SIZE as usize] = human_output.get_policy_probs_maybe_noised();
        let mut policy: Vec<Value> = Vec::new();
        for y in 0..board.y_size {
          for x in 0..board.x_size {
            let pos: i32 = NNPos::xy_to_pos(x, y, self.nn_x_len);
            policy.push(Value::from(Global::round_dynamic(policy_probs[pos as usize] as f64, OUTPUT_PRECISION)));
          }
        }
        let pass_pos: i32 = NNPos::loc_to_pos(Board::PASS_LOC, board.x_size, self.nn_x_len, self.nn_y_len);
        policy.push(Value::from(Global::round_dynamic(policy_probs[pass_pos as usize] as f64, OUTPUT_PRECISION)));
        ret["humanPolicy"] = Value::Array(policy);
      }
    }

    // Average tree ownership
    let root_node: Option<Rc<SearchNode>> = self.get_root_node();
    if include_ownership && include_ownership_stdev {
      let symmetry: i32 = 0;
      let ownership_and_stdev: (Vec<f64>, Vec<f64>) =
        self.get_average_and_standard_deviation_tree_ownership_perspective(perspective, root_node.as_deref(), symmetry);
      ret["ownership"] = json_of_doubles(&ownership_and_stdev.0);
      ret["ownershipStdev"] = json_of_doubles(&ownership_and_stdev.1);
    } else if include_ownership_stdev {
      let symmetry: i32 = 0;
      let ownership_and_stdev: (Vec<f64>, Vec<f64>) =
        self.get_average_and_standard_deviation_tree_ownership_perspective(perspective, root_node.as_deref(), symmetry);
      ret["ownershipStdev"] = json_of_doubles(&ownership_and_stdev.1);
    } else if include_ownership {
      let symmetry: i32 = 0;
      ret["ownership"] = json_of_doubles(&self.get_average_tree_ownership_perspective(perspective, root_node.as_deref(), symmetry)?);
    }

    Ok(true)
  }

  /// upstream: Search::getPrunedRootValues (searchresults.cpp)
  //Compute all the stats of the node based on its children, pruning weights such that they are as expected
  //based on policy and utility. This is used to give accurate rootInfo even with a lot of wide root noise
  pub fn get_pruned_root_values(&self, values: &mut ReportedSearchValues) -> bool {
    let root_node: Option<Rc<SearchNode>> = self.get_root_node();
    self.get_pruned_node_values(root_node.as_deref(), values)
  }

  /// upstream: Search::getPrunedNodeValues (searchresults.cpp)
  pub fn get_pruned_node_values(&self, node_ptr: Option<&SearchNode>, values: &mut ReportedSearchValues) -> bool {
    let node: &SearchNode = match node_ptr {
      None => return false,
      Some(n) => n,
    };

    let children: SearchNodeChildrenReference = node.get_children();
    let children_capacity: i32 = children.get_capacity();

    let mut play_selection_values: Vec<f64> = Vec::new();
    let mut locs: Vec<Loc> = Vec::new(); // not used
    let allow_direct_policy_moves: bool = false;
    let always_compute_lcb: bool = false;
    let never_use_lcb: bool = true;
    let suc: bool = self.get_play_selection_values_node_lcb(
      node,
      &mut locs,
      &mut play_selection_values,
      None,
      1.0,
      allow_direct_policy_moves,
      always_compute_lcb,
      never_use_lcb,
      None,
      None,
    );
    //If there are no children, or otherwise values could not be computed,
    //then fall back to the normal case and just listen to the values on the node rather than trying
    //to recompute things.
    if !suc {
      return self.get_node_values(node_ptr, values);
    }

    let mut win_loss_value_sum: f64 = 0.0;
    let mut no_result_value_sum: f64 = 0.0;
    let mut score_mean_sum: f64 = 0.0;
    let mut score_mean_sq_sum: f64 = 0.0;
    let mut lead_sum: f64 = 0.0;
    let mut utility_sum: f64 = 0.0;
    let mut weight_sum: f64 = 0.0;
    for i in 0..children_capacity {
      let child_pointer: &SearchChildPointer = children.index(i);
      let child: Rc<SearchNode> = match child_pointer.get_if_allocated() {
        None => break,
        Some(c) => self.node_arena.get(c),
      };
      let edge_visits: i64 = child_pointer.get_edge_visits();
      let stats: NodeStats = NodeStats::new_other(&child.stats);

      if stats.visits <= 0 || stats.weight_sum <= 0.0 || edge_visits <= 0 {
        continue;
      }
      let weight: f64 = play_selection_values[i as usize];
      win_loss_value_sum += weight * stats.win_loss_value_avg;
      no_result_value_sum += weight * stats.no_result_value_avg;
      score_mean_sum += weight * stats.score_mean_avg;
      score_mean_sq_sum += weight * stats.score_mean_sq_avg;
      lead_sum += weight * stats.lead_avg;
      utility_sum += weight * stats.utility_avg;
      weight_sum += weight;
    }

    //Also add in the direct evaluation of this node.
    {
      //If somehow the nnOutput is still null here, skip
      let nn_output: Rc<NNOutput> = match node.get_nn_output() {
        None => return false,
        Some(o) => o,
      };
      let win_prob: f64 = nn_output.white_win_prob as f64;
      let loss_prob: f64 = nn_output.white_loss_prob as f64;
      let no_result_prob: f64 = nn_output.white_no_result_prob as f64;
      let score_mean: f64 = nn_output.white_score_mean as f64;
      let score_mean_sq: f64 = nn_output.white_score_mean_sq as f64;
      let lead: f64 = nn_output.white_lead as f64;
      let utility: f64 = self.get_result_utility(win_prob - loss_prob, no_result_prob) + self.get_score_utility(score_mean, score_mean_sq);

      let weight: f64 = self.compute_weight_from_nn_output(&nn_output);
      win_loss_value_sum += (win_prob - loss_prob) * weight;
      no_result_value_sum += no_result_prob * weight;
      score_mean_sum += score_mean * weight;
      score_mean_sq_sum += score_mean_sq * weight;
      lead_sum += lead * weight;
      utility_sum += utility * weight;
      weight_sum += weight;
    }
    *values = ReportedSearchValues::new_search(
      self,
      win_loss_value_sum / weight_sum,
      no_result_value_sum / weight_sum,
      score_mean_sum / weight_sum,
      score_mean_sq_sum / weight_sum,
      lead_sum / weight_sum,
      utility_sum / weight_sum,
      node.stats.weight_sum.get(), /*acquire*/
      node.stats.visits.get(),     /*acquire*/
    );
    true
  }

  /// upstream: Search::debugPrintChildrenSummary (searchresults.cpp)
  /// PORT NOTE: non-const upstream but only reads; `&self` so the playout loop can call it.
  pub(crate) fn debug_print_children_summary(&self, out: &mut dyn std::fmt::Write, node: &SearchNode, nn_output: &NNOutput) {
    let node_state: SearchNodeState = node.state.get() /*acquire*/;
    let mut num_children: i32 = 0;
    let children: SearchNodeChildrenReference = node.get_children_state(node_state);
    let children_capacity: i32 = children.get_capacity();
    writeln!(out, "childrenCapacity {}", children_capacity).unwrap();
    let policy_probs: &[f32; NNPos::MAX_NN_POLICY_SIZE as usize] = nn_output.get_policy_probs_maybe_noised();
    for i in 0..children_capacity {
      let child_pointer: &SearchChildPointer = children.index(i);
      let child: Rc<SearchNode> = match child_pointer.get_if_allocated() {
        None => break,
        Some(c) => self.node_arena.get(c),
      };
      num_children += 1;
      let move_loc: Loc = child_pointer.get_move_loc_relaxed();
      let move_pos: i32 = self.get_pos(move_loc);
      let nn_policy_prob: f32 = policy_probs[move_pos as usize];
      let edge_visits: i64 = child_pointer.get_edge_visits();
      let child_weight: f64 = child.stats.get_child_weight(edge_visits);
      writeln!(
        out,
        "{} {} {} {} {}",
        i,
        Location::to_string(move_loc, &self.root_board),
        cpp_ostream_f64(nn_policy_prob as f64),
        edge_visits,
        cpp_ostream_f64(child_weight)
      )
      .unwrap();
    }
    writeln!(out, "numChildren {}", num_children).unwrap();
  }
}

/// `json(std::vector<double>)`: a JSON array of numbers (NaN/inf become null, as nlohmann dumps them).
fn json_of_doubles(v: &[f64]) -> Value {
  let mut arr: Vec<Value> = Vec::with_capacity(v.len());
  for &x in v {
    arr.push(Value::from(x));
  }
  Value::Array(arr)
}
