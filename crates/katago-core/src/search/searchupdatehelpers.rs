// Ported from KataGo cpp/search/searchupdatehelpers.cpp @ ba938676d7f42d70950b3a535af2466fb642008c
//
// PORT NOTE: playout-path helpers take &self and write the node stats through their Cells
// (PORTING.md §3/§11). The statsLock and the SubtreeValueBiasEntry entryLock spin locks are removed
// (single-threaded), their bodies kept.

use std::rc::Rc;

use crate::game::board::get_opp;
use crate::game::board::Loc;
use crate::game::board::C_EMPTY;
use crate::game::board::P_WHITE;
use crate::neuralnet::nninputs::NNOutput;
use crate::neuralnet::nninputs::NNPos;
use crate::port::cpp;
use crate::port::math;
use crate::port::node_arena::NodeIdx;
use crate::search::search::Search;
use crate::search::search::SearchThread;
use crate::search::evalcache::EvalCacheEntry;
use crate::search::searchnode::MoreNodeStats;
use crate::search::searchnode::NodeStats;
use crate::search::searchnode::SearchChildPointer;
use crate::search::searchnode::SearchNode;
use crate::search::searchnode::SearchNodeChildrenReference;
use crate::search::subtreevaluebiastable::SubtreeValueBiasEntry;

impl Search {
  /// upstream: Search::addLeafValue (searchupdatehelpers.cpp)
  pub(crate) fn add_leaf_value(
    &self,
    node: &SearchNode,
    win_loss_value: f64,
    no_result_value: f64,
    score_mean: f64,
    score_mean_sq: f64,
    lead: f64,
    weight: f64,
    is_terminal: bool,
    assume_no_existing_weight: bool,
  ) {
    let mut utility: f64 = self.get_result_utility(win_loss_value, no_result_value) + self.get_score_utility(score_mean, score_mean_sq);

    if self.search_params.subtree_value_bias_factor != 0.0 && !is_terminal {
      let entry_opt: Option<Rc<SubtreeValueBiasEntry>> = node.subtree_value_bias_table_entry.borrow().clone();
      if let Some(entry) = entry_opt {
        // PORT NOTE: entryLock spin lock removed (single-threaded).
        let new_entry_delta_utility_sum: f64 = entry.delta_utility_sum.get();
        let new_entry_weight_sum: f64 = entry.weight_sum.get();
        //This is the amount of the direct evaluation of this node that we are going to bias towards the table entry
        let bias_factor: f64 = self.search_params.subtree_value_bias_factor;
        if new_entry_weight_sum > 0.001 {
          utility += bias_factor * new_entry_delta_utility_sum / new_entry_weight_sum;
        }
      }
    }

    utility += self.get_pattern_bonus(node.pattern_bonus_hash.get(), get_opp(node.next_pla));

    let utility_sq: f64 = utility * utility;
    let weight_sq: f64 = weight * weight;

    if assume_no_existing_weight {
      // PORT NOTE: statsLock spin lock removed (single-threaded).
      node.stats.win_loss_value_avg.set(win_loss_value) /*release*/;
      node.stats.no_result_value_avg.set(no_result_value) /*release*/;
      node.stats.score_mean_avg.set(score_mean) /*release*/;
      node.stats.score_mean_sq_avg.set(score_mean_sq) /*release*/;
      node.stats.lead_avg.set(lead) /*release*/;
      node.stats.utility_avg.set(utility) /*release*/;
      node.stats.utility_sq_avg.set(utility_sq) /*release*/;
      node.stats.weight_sq_sum.set(weight_sq) /*release*/;
      node.stats.weight_sum.set(weight) /*release*/;
      let old_visits: i64 = cpp::cell_fetch_add(&node.stats.visits, 1i64) /*release*/;
      // This should only be possible in the extremely rare case that we transpose to a terminal node from a non-terminal node probably due to
      // a hash collision, or that we have a graph history interaction that somehow changes whether a particular path ends the game or not, despite
      // our simpleRepetitionBoundGt logic... such that the node managed to get visits as a terminal node despite not having an nn eval. There's
      // nothing reasonable to do here once we have such a bad collision, so just at least don't crash.
      if old_visits != 0 {
        log::info!("WARNING: assumeNoExistingWeight for leaf but leaf already has visits");
      }
    } else {
      // PORT NOTE: statsLock spin lock removed (single-threaded).
      let old_weight_sum: f64 = node.stats.weight_sum.get() /*relaxed*/;
      let new_weight_sum: f64 = old_weight_sum + weight;

      node.stats.win_loss_value_avg.set((node.stats.win_loss_value_avg.get() * old_weight_sum + win_loss_value * weight) / new_weight_sum) /*release*/;
      node.stats.no_result_value_avg.set((node.stats.no_result_value_avg.get() * old_weight_sum + no_result_value * weight) / new_weight_sum) /*release*/;
      node.stats.score_mean_avg.set((node.stats.score_mean_avg.get() * old_weight_sum + score_mean * weight) / new_weight_sum) /*release*/;
      node.stats.score_mean_sq_avg.set((node.stats.score_mean_sq_avg.get() * old_weight_sum + score_mean_sq * weight) / new_weight_sum) /*release*/;
      node.stats.lead_avg.set((node.stats.lead_avg.get() * old_weight_sum + lead * weight) / new_weight_sum) /*release*/;
      node.stats.utility_avg.set((node.stats.utility_avg.get() * old_weight_sum + utility * weight) / new_weight_sum) /*release*/;
      node.stats.utility_sq_avg.set((node.stats.utility_sq_avg.get() * old_weight_sum + utility_sq * weight) / new_weight_sum) /*release*/;
      node.stats.weight_sq_sum.set(node.stats.weight_sq_sum.get() + weight_sq) /*release*/;
      node.stats.weight_sum.set(new_weight_sum) /*release*/;
      cpp::cell_fetch_add(&node.stats.visits, 1i64) /*release*/;
    }
  }

  /// upstream: Search::addCurrentNNOutputAsLeafValue (searchupdatehelpers.cpp)
  pub(crate) fn add_current_nn_output_as_leaf_value(&self, node: &SearchNode, assume_no_existing_weight: bool) {
    let nn_output: Rc<NNOutput> = node.get_nn_output().expect("addCurrentNNOutputAsLeafValue needs an nnOutput");
    //Values in the search are from the perspective of white positive always
    let mut win_loss_value: f64 = nn_output.white_win_prob as f64 - nn_output.white_loss_prob as f64;
    let mut no_result_prob: f64 = nn_output.white_no_result_prob as f64;
    let mut score_mean: f64 = nn_output.white_score_mean as f64;
    let mut score_mean_sq: f64 = nn_output.white_score_mean_sq as f64;
    let mut lead: f64 = nn_output.white_lead as f64;

    //Note that root node does not use eval cache for its aggregate values
    if self.search_params.use_eval_cache
      && self.search_params.use_graph_search
      && node.eval_cache_entry.borrow().is_some()
      && self.mirroring_pla == C_EMPTY
      && (Some(node.idx) != self.root_node)
      && !node.force_non_terminal
    {
      // For raw NN value for purposes of cache weighting, always treat the node as 1 visit since
      // repeating the same nn eval over and over is only 1 visit of eval info.
      let this_node_visits_for_cache: i64 = 1;
      let entry: Rc<EvalCacheEntry> = node.eval_cache_entry.borrow().clone().expect("checked is_some");
      self.adjust_evals_from_cache_helper(
        &entry,
        this_node_visits_for_cache,
        &mut win_loss_value,
        &mut no_result_prob,
        &mut score_mean,
        &mut score_mean_sq,
        &mut lead,
        None,
      );
    }

    let weight: f64 = self.compute_weight_from_nn_output(&nn_output);
    self.add_leaf_value(node, win_loss_value, no_result_prob, score_mean, score_mean_sq, lead, weight, false, assume_no_existing_weight);
  }

  /// upstream: Search::computeWeightFromNNOutput (searchupdatehelpers.cpp)
  /// PORT NOTE: `pub` (upstream private, PORTING.md §1 would give pub(crate)) so the crafted parity oracle
  /// tests/search_select_diff.rs can call it.
  pub fn compute_weight_from_nn_output(&self, nn_output: &NNOutput) -> f64 {
    if !self.search_params.use_uncertainty {
      return 1.0;
    }
    if !self.nn_evaluator.supports_shortterm_error() {
      return 1.0;
    }

    let score_mean: f64 = nn_output.white_score_mean as f64;
    let utility_uncertainty_wl: f64 = self.search_params.win_loss_utility_factor * nn_output.shortterm_winloss_error as f64;
    let utility_uncertainty_score: f64 = self.get_approx_score_utility_derivative(score_mean) * nn_output.shortterm_score_error as f64;
    let utility_uncertainty: f64 = utility_uncertainty_wl + utility_uncertainty_score;

    let powered_uncertainty: f64;
    if self.search_params.uncertainty_exponent == 1.0 {
      powered_uncertainty = utility_uncertainty;
    } else if self.search_params.uncertainty_exponent == 0.5 {
      powered_uncertainty = utility_uncertainty.sqrt();
    } else {
      powered_uncertainty = math::powf(utility_uncertainty, self.search_params.uncertainty_exponent);
    }

    let baseline_uncertainty: f64 = self.search_params.uncertainty_coeff / self.search_params.uncertainty_max_weight;
    let weight: f64 = self.search_params.uncertainty_coeff / (powered_uncertainty + baseline_uncertainty);
    weight
  }

  /// upstream: Search::updateStatsAfterPlayout (searchupdatehelpers.cpp)
  pub(crate) fn update_stats_after_playout(&self, node: &SearchNode, thread: &mut SearchThread, is_root: bool) {
    //The thread that grabs a 0 from this peforms the recomputation of stats.
    let mut old_dirty_counter: i32 = cpp::cell_fetch_add(&node.dirty_counter, 1) /*acq_rel*/;
    debug_assert!(old_dirty_counter >= 0);
    //If we atomically grab a nonzero, then we know another thread must already be doing the work, so we can skip the update ourselves.
    if old_dirty_counter > 0 {
      return;
    }
    let mut num_visits_completed: i32 = 1;
    loop {
      //Perform update
      self.recompute_node_stats(node, thread, num_visits_completed, is_root);
      //Now attempt to undo the counter
      old_dirty_counter = cpp::cell_fetch_add(&node.dirty_counter, -num_visits_completed) /*acq_rel*/;
      let new_dirty_counter: i32 = old_dirty_counter - num_visits_completed;
      //If no other threads incremented it in the meantime, so our decrement hits zero, we're done.
      if new_dirty_counter <= 0 {
        debug_assert!(new_dirty_counter == 0);
        break;
      }
      //Otherwise, more threads incremented this more in the meantime. So we need to loop again and add their visits, recomputing again.
      num_visits_completed = new_dirty_counter;
      continue;
    }
  }

  /// upstream: Search::recomputeNodeStats (searchupdatehelpers.cpp)
  //Recompute all the stats of this node based on its children, except its visits and virtual losses, which are not child-dependent and
  //are updated in the manner specified.
  //Assumes this node has an nnOutput
  pub(crate) fn recompute_node_stats(&self, node: &SearchNode, thread: &mut SearchThread, num_visits_to_add: i32, is_root: bool) {
    //Find all children and compute weighting of the children based on their values
    let stats_buf: &mut Vec<MoreNodeStats> = &mut thread.stats_buf;
    let mut num_good_children: i32 = 0;

    let children: SearchNodeChildrenReference = node.get_children();
    let children_capacity: i32 = children.get_capacity();
    let mut orig_total_child_weight: f64 = 0.0;
    let mut this_node_visits: i64 = 1; // 1 = own visit
    for i in 0..children_capacity {
      let child_pointer: &SearchChildPointer = children.index(i);
      let child: NodeIdx = match child_pointer.get_if_allocated() {
        None => break,
        Some(c) => c,
      };
      let stats: &mut MoreNodeStats = &mut stats_buf[num_good_children as usize];

      let move_loc: Loc = child_pointer.get_move_loc_relaxed();
      let edge_visits: i64 = child_pointer.get_edge_visits();
      stats.stats = NodeStats::new_other(&self.node_arena.get(child).stats);

      if stats.stats.visits <= 0 || stats.stats.weight_sum <= 0.0 || edge_visits <= 0 {
        continue;
      }

      let child_utility: f64 = stats.stats.utility_avg;
      stats.self_utility = if node.next_pla == P_WHITE { child_utility } else { -child_utility };
      stats.weight_adjusted = stats.stats.get_child_weight(edge_visits);
      stats.prev_move_loc = move_loc;

      orig_total_child_weight += stats.weight_adjusted;
      this_node_visits += edge_visits;
      num_good_children += 1;
    }

    //Always tracks the sum of statsBuf[i].weightAdjusted across the children.
    let mut current_total_child_weight: f64 = orig_total_child_weight;

    if self.search_params.use_noise_pruning && num_good_children > 0 && !(self.search_params.anti_mirror && self.mirroring_pla != C_EMPTY) {
      let mut policy_probs_buf: [f64; NNPos::MAX_NN_POLICY_SIZE as usize] = [0.0; NNPos::MAX_NN_POLICY_SIZE as usize];
      {
        let nn_output: Rc<NNOutput> = node.get_nn_output().expect("recomputeNodeStats needs an nnOutput");
        let policy_probs: &[f32] = &nn_output.get_policy_probs_maybe_noised()[..];
        for i in 0..num_good_children as usize {
          policy_probs_buf[i] = cpp::cpp_max(1e-30, policy_probs[self.get_pos(stats_buf[i].prev_move_loc) as usize] as f64);
        }
      }
      current_total_child_weight = self.prune_noise_weight(stats_buf, num_good_children, current_total_child_weight, &policy_probs_buf);
    }

    {
      let mut amount_to_subtract: f64 = 0.0;
      let mut amount_to_prune: f64 = 0.0;
      if is_root && self.search_params.root_noise_enabled && !self.search_params.use_noise_pruning {
        let mut max_child_weight: f64 = 0.0;
        for i in 0..num_good_children as usize {
          if stats_buf[i].weight_adjusted > max_child_weight {
            max_child_weight = stats_buf[i].weight_adjusted;
          }
        }
        amount_to_subtract = cpp::cpp_min(self.search_params.chosen_move_subtract, max_child_weight / 64.0);
        amount_to_prune = cpp::cpp_min(self.search_params.chosen_move_prune, max_child_weight / 64.0);
      }

      self.downweight_bad_children_and_normalize_weight(
        num_good_children,
        current_total_child_weight,
        current_total_child_weight,
        amount_to_subtract,
        amount_to_prune,
        stats_buf,
      );
    }

    let mut win_loss_value_sum: f64 = 0.0;
    let mut no_result_value_sum: f64 = 0.0;
    let mut score_mean_sum: f64 = 0.0;
    let mut score_mean_sq_sum: f64 = 0.0;
    let mut lead_sum: f64 = 0.0;
    let mut utility_sum: f64 = 0.0;
    let mut utility_sq_sum: f64 = 0.0;
    let mut weight_sq_sum: f64 = 0.0;
    let mut weight_sum: f64 = current_total_child_weight;
    for i in 0..num_good_children as usize {
      let stats: &NodeStats = &stats_buf[i].stats;

      let desired_weight: f64 = stats_buf[i].weight_adjusted;
      let weight_scaling: f64 = desired_weight / stats.weight_sum;

      win_loss_value_sum += desired_weight * stats.win_loss_value_avg;
      no_result_value_sum += desired_weight * stats.no_result_value_avg;
      score_mean_sum += desired_weight * stats.score_mean_avg;
      score_mean_sq_sum += desired_weight * stats.score_mean_sq_avg;
      lead_sum += desired_weight * stats.lead_avg;
      utility_sum += desired_weight * stats.utility_avg;
      utility_sq_sum += desired_weight * stats.utility_sq_avg;
      weight_sq_sum += weight_scaling * weight_scaling * stats.weight_sq_sum;
    }

    //Also add in the direct evaluation of this node.
    {
      let nn_output: Rc<NNOutput> = node.get_nn_output().expect("recomputeNodeStats needs an nnOutput");
      let win_prob: f64 = nn_output.white_win_prob as f64;
      let loss_prob: f64 = nn_output.white_loss_prob as f64;
      let no_result_prob: f64 = nn_output.white_no_result_prob as f64;
      let score_mean: f64 = nn_output.white_score_mean as f64;
      let score_mean_sq: f64 = nn_output.white_score_mean_sq as f64;
      let lead: f64 = nn_output.white_lead as f64;
      let mut utility: f64 = self.get_result_utility(win_prob - loss_prob, no_result_prob) + self.get_score_utility(score_mean, score_mean_sq);

      if self.search_params.subtree_value_bias_factor != 0.0 {
        let entry_opt: Option<Rc<SubtreeValueBiasEntry>> = node.subtree_value_bias_table_entry.borrow().clone();
        if let Some(entry) = entry_opt {
          let new_entry_delta_utility_sum: f64;
          let new_entry_weight_sum: f64;

          if current_total_child_weight > 1e-10 {
            let utility_children: f64 = utility_sum / current_total_child_weight;
            let subtree_value_bias_weight: f64 = math::powf(orig_total_child_weight, self.search_params.subtree_value_bias_weight_exponent);
            let subtree_value_bias_delta_sum: f64 = (utility_children - utility) * subtree_value_bias_weight;

            // PORT NOTE: entryLock spin lock removed (single-threaded).
            entry.delta_utility_sum.set(entry.delta_utility_sum.get() + (subtree_value_bias_delta_sum - node.last_subtree_value_bias_delta_sum.get()));
            entry.weight_sum.set(entry.weight_sum.get() + (subtree_value_bias_weight - node.last_subtree_value_bias_weight.get()));
            new_entry_delta_utility_sum = entry.delta_utility_sum.get();
            new_entry_weight_sum = entry.weight_sum.get();
            node.last_subtree_value_bias_delta_sum.set(subtree_value_bias_delta_sum);
            node.last_subtree_value_bias_weight.set(subtree_value_bias_weight);
          } else {
            // PORT NOTE: entryLock spin lock removed (single-threaded).
            new_entry_delta_utility_sum = entry.delta_utility_sum.get();
            new_entry_weight_sum = entry.weight_sum.get();
          }

          //This is the amount of the direct evaluation of this node that we are going to bias towards the table entry
          let bias_factor: f64 = self.search_params.subtree_value_bias_factor;
          if new_entry_weight_sum > 0.001 {
            utility += bias_factor * new_entry_delta_utility_sum / new_entry_weight_sum;
          }
          //This is the amount by which we need to scale desiredSelfWeight such that if the table entry were actually equal to
          //the current difference between the direct eval and the children, we would perform a no-op... unless a noop is actually impossible
          //Then we just take what we can get.
          //desiredSelfWeight *= weightSum / (1.0-biasFactor) / std::max(0.001, (weightSum + desiredSelfWeight - desiredSelfWeight / (1.0-biasFactor)));
        }
      }

      let weight: f64 = self.compute_weight_from_nn_output(&nn_output);
      win_loss_value_sum += (win_prob - loss_prob) * weight;
      no_result_value_sum += no_result_prob * weight;
      score_mean_sum += score_mean * weight;
      score_mean_sq_sum += score_mean_sq * weight;
      lead_sum += lead * weight;
      utility_sum += utility * weight;
      utility_sq_sum += utility * utility * weight;
      weight_sq_sum += weight * weight;
      weight_sum += weight;
    }

    let mut win_loss_value_avg: f64 = win_loss_value_sum / weight_sum;
    let mut no_result_value_avg: f64 = no_result_value_sum / weight_sum;
    let mut score_mean_avg: f64 = score_mean_sum / weight_sum;
    let mut score_mean_sq_avg: f64 = score_mean_sq_sum / weight_sum;
    let mut lead_avg: f64 = lead_sum / weight_sum;
    let mut utility_avg: f64 = utility_sum / weight_sum;
    let mut utility_sq_avg: f64 = utility_sq_sum / weight_sum;

    let old_utility_avg: f64 = utility_avg;
    utility_avg += self.get_pattern_bonus(node.pattern_bonus_hash.get(), get_opp(node.next_pla));
    //Note that root node does not use eval cache for its aggregate values
    if self.search_params.use_eval_cache
      && self.search_params.use_graph_search
      && node.eval_cache_entry.borrow().is_some()
      && self.mirroring_pla == C_EMPTY
      && (Some(node.idx) != self.root_node)
      && !node.force_non_terminal
    {
      let entry: Rc<EvalCacheEntry> = node.eval_cache_entry.borrow().clone().expect("checked is_some");
      self.adjust_evals_from_cache_helper(
        &entry,
        this_node_visits,
        &mut win_loss_value_avg,
        &mut no_result_value_avg,
        &mut score_mean_avg,
        &mut score_mean_sq_avg,
        &mut lead_avg,
        Some(&mut utility_avg),
      );
    }
    utility_sq_avg = utility_sq_avg + (utility_avg * utility_avg - old_utility_avg * old_utility_avg);

    //TODO statslock may be unnecessary now with the dirtyCounter mechanism?
    // PORT NOTE: statsLock spin lock removed (single-threaded).
    node.stats.win_loss_value_avg.set(win_loss_value_avg) /*release*/;
    node.stats.no_result_value_avg.set(no_result_value_avg) /*release*/;
    node.stats.score_mean_avg.set(score_mean_avg) /*release*/;
    node.stats.score_mean_sq_avg.set(score_mean_sq_avg) /*release*/;
    node.stats.lead_avg.set(lead_avg) /*release*/;
    node.stats.utility_avg.set(utility_avg) /*release*/;
    node.stats.utility_sq_avg.set(utility_sq_avg) /*release*/;
    node.stats.weight_sq_sum.set(weight_sq_sum) /*release*/;
    node.stats.weight_sum.set(weight_sum) /*release*/;
    cpp::cell_fetch_add(&node.stats.visits, num_visits_to_add as i64) /*release*/;
  }

  /// upstream: Search::adjustEvalsFromCacheHelper (searchupdatehelpers.cpp)
  /// PORT NOTE: `&self` (upstream non-const; it mutates only its out-params, PORTING.md §3).
  pub(crate) fn adjust_evals_from_cache_helper(
    &self,
    eval_cache_entry: &Rc<EvalCacheEntry>,
    this_node_visits: i64,
    win_loss_value_avg: &mut f64,
    no_result_value_avg: &mut f64,
    score_mean_avg: &mut f64,
    score_mean_sq_avg: &mut f64,
    lead_avg: &mut f64,
    utility_avg: Option<&mut f64>,
  ) {
    let cache_avg_win_loss: f64 = eval_cache_entry.avg_win_loss as f64;
    let cache_avg_no_result: f64 = eval_cache_entry.avg_no_result as f64;
    let cache_avg_score_mean: f64 = eval_cache_entry.avg_score_mean as f64;
    let cache_avg_lead: f64 = eval_cache_entry.avg_lead as f64;
    let mut cache_weight: f64 = eval_cache_entry.cache_weight as f64;
    // Squish down very heavily-weighted entries that are likely to have cache entries ahead of them, to somewhat
    // reduce the impact of double-counting since child nodes will also apply their own averaging.
    if cache_weight > self.search_params.eval_cache_min_visits as f64 {
      cache_weight = (self.search_params.eval_cache_min_visits as f64 * cache_weight).sqrt();
    }
    let visits_to_cache_ratio: f64 = this_node_visits as f64 / cache_weight;
    let cache_frac: f64 = 1.0 / (1.0 + 3.0 * visits_to_cache_ratio * (1.0 + 2.0 * visits_to_cache_ratio * visits_to_cache_ratio));

    *win_loss_value_avg += cache_frac * (cache_avg_win_loss - *win_loss_value_avg);
    *no_result_value_avg += cache_frac * (cache_avg_no_result - *no_result_value_avg);
    let old_score_mean_avg: f64 = *score_mean_avg;
    *score_mean_avg += cache_frac * (cache_avg_score_mean - *score_mean_avg);
    *score_mean_sq_avg = cpp::cpp_max(0.0, *score_mean_sq_avg - old_score_mean_avg * old_score_mean_avg + *score_mean_avg * *score_mean_avg);
    *lead_avg += cache_frac * (cache_avg_lead - *lead_avg);

    if let Some(utility_avg) = utility_avg {
      let cache_avg_utility: f64 = self.get_result_utility(cache_avg_win_loss, cache_avg_no_result)
        + self.get_score_utility(
          cache_avg_score_mean,
          cpp::cpp_max(0.0, *score_mean_sq_avg - *score_mean_avg * *score_mean_avg + cache_avg_score_mean * cache_avg_score_mean),
        );
      *utility_avg += cache_frac * (cache_avg_utility - *utility_avg);
    }
  }

  /// upstream: Search::downweightBadChildrenAndNormalizeWeight (searchupdatehelpers.cpp)
  /// PORT NOTE: `pub` (upstream private, PORTING.md §1 would give pub(crate)) so the crafted parity oracle
  /// tests/search_select_diff.rs can call it.
  pub fn downweight_bad_children_and_normalize_weight(
    &self,
    num_children: i32,
    mut current_total_weight: f64, //The current sum of statsBuf[i].weightAdjusted
    desired_total_weight: f64,     //What statsBuf[i].weightAdjusted should sum up to after this function is done.
    amount_to_subtract: f64,
    amount_to_prune: f64,
    stats_buf: &mut Vec<MoreNodeStats>,
  ) {
    if num_children <= 0 || current_total_weight <= 0.0 {
      return;
    }

    if self.search_params.value_weight_exponent == 0.0 || self.mirroring_pla != C_EMPTY {
      for i in 0..num_children as usize {
        if stats_buf[i].weight_adjusted < amount_to_prune {
          current_total_weight -= stats_buf[i].weight_adjusted;
          stats_buf[i].weight_adjusted = 0.0;
          continue;
        }
        let new_weight: f64 = stats_buf[i].weight_adjusted - amount_to_subtract;
        if new_weight <= 0.0 {
          current_total_weight -= stats_buf[i].weight_adjusted;
          stats_buf[i].weight_adjusted = 0.0;
        } else {
          current_total_weight -= amount_to_subtract;
          stats_buf[i].weight_adjusted = new_weight;
        }
      }

      if current_total_weight != desired_total_weight {
        let factor: f64 = desired_total_weight / current_total_weight;
        for i in 0..num_children as usize {
          stats_buf[i].weight_adjusted *= factor;
        }
      }
      return;
    }

    debug_assert!(num_children <= NNPos::MAX_NN_POLICY_SIZE);
    let mut stdevs: [f64; NNPos::MAX_NN_POLICY_SIZE as usize] = [0.0; NNPos::MAX_NN_POLICY_SIZE as usize];
    let mut simple_value_sum: f64 = 0.0;
    for i in 0..num_children as usize {
      let num_visits: i64 = stats_buf[i].stats.visits;
      debug_assert!(num_visits >= 0);
      if num_visits == 0 {
        continue;
      }

      let weight: f64 = stats_buf[i].weight_adjusted;
      let precision: f64 = 1.5 * weight.sqrt();

      //Ensure some minimum variance for stability regardless of how we change the above formula
      const MIN_VARIANCE: f64 = 0.00000001;
      stdevs[i] = (MIN_VARIANCE + 1.0 / precision).sqrt();
      simple_value_sum += stats_buf[i].self_utility * weight;
    }

    let simple_value: f64 = simple_value_sum / current_total_weight;

    let mut total_new_unnorm_weight: f64 = 0.0;
    for i in 0..num_children as usize {
      if stats_buf[i].stats.visits == 0 {
        continue;
      }

      if stats_buf[i].weight_adjusted < amount_to_prune {
        current_total_weight -= stats_buf[i].weight_adjusted;
        stats_buf[i].weight_adjusted = 0.0;
        continue;
      }
      let new_weight: f64 = stats_buf[i].weight_adjusted - amount_to_subtract;
      if new_weight <= 0.0 {
        current_total_weight -= stats_buf[i].weight_adjusted;
        stats_buf[i].weight_adjusted = 0.0;
      } else {
        current_total_weight -= amount_to_subtract;
        stats_buf[i].weight_adjusted = new_weight;
      }

      let z: f64 = (stats_buf[i].self_utility - simple_value) / stdevs[i];
      //Also just for numeric sanity, make sure everything has some tiny minimum value.
      let p: f64 = self.value_weight_distribution.get_cdf(z) + 0.0001;
      stats_buf[i].weight_adjusted *= math::powf(p, self.search_params.value_weight_exponent);
      total_new_unnorm_weight += stats_buf[i].weight_adjusted;
    }
    let _ = current_total_weight;

    //Post-process and normalize to sum to the desired weight
    debug_assert!(total_new_unnorm_weight > 0.0);
    let factor: f64 = desired_total_weight / total_new_unnorm_weight;
    for i in 0..num_children as usize {
      stats_buf[i].weight_adjusted *= factor;
    }
  }

  /// upstream: Search::pruneNoiseWeight (searchupdatehelpers.cpp)
  //Returns the new sum of weightAdjusted
  /// PORT NOTE: `pub` (upstream private, PORTING.md §1 would give pub(crate)) so the crafted parity oracle
  /// tests/search_select_diff.rs can call it.
  pub fn prune_noise_weight(&self, stats_buf: &mut Vec<MoreNodeStats>, num_children: i32, total_child_weight: f64, policy_probs_buf: &[f64]) -> f64 {
    if num_children <= 1 || total_child_weight <= 0.00001 {
      return total_child_weight;
    }

    // Children are normally sorted in policy order in KataGo.
    // But this is not guaranteed, because at the root, we might recompute the nnoutput, or when finding the best new child, we have hacks like antiMirror policy
    // and other adjustments. For simplicity, we just consider children in sorted order anyways for this pruning, since it will be close.

    // For any child, if its own utility is lower than the weighted average utility of the children before it, it's downweighted if it exceeds much more than a
    // raw-policy share of the weight.
    let mut utility_sum_so_far: f64 = 0.0;
    let mut weight_sum_so_far: f64 = 0.0;
    //double rawPolicyUtilitySumSoFar = 0;
    let mut raw_policy_sum_so_far: f64 = 0.0;
    for i in 0..num_children as usize {
      let utility: f64 = stats_buf[i].self_utility;
      let old_weight: f64 = stats_buf[i].weight_adjusted;
      let raw_policy: f64 = policy_probs_buf[i];

      let mut new_weight: f64 = old_weight;
      if weight_sum_so_far > 0.0 && raw_policy_sum_so_far > 0.0 {
        let avg_utility_so_far: f64 = utility_sum_so_far / weight_sum_so_far;
        let utility_gap: f64 = avg_utility_so_far - utility;
        if utility_gap > 0.0 {
          let weight_share_from_raw_policy: f64 = weight_sum_so_far * raw_policy / raw_policy_sum_so_far;
          //If the child is more than double its proper share of the weight
          let lenient_weight_share_from_raw_policy: f64 = 2.0 * weight_share_from_raw_policy;
          if old_weight > lenient_weight_share_from_raw_policy {
            let excess_weight: f64 = old_weight - lenient_weight_share_from_raw_policy;
            let mut weight_to_subtract: f64 = excess_weight * (1.0 - math::exp(-utility_gap / self.search_params.noise_prune_utility_scale));
            if weight_to_subtract > self.search_params.noise_pruning_cap {
              weight_to_subtract = self.search_params.noise_pruning_cap;
            }

            new_weight = old_weight - weight_to_subtract;
            stats_buf[i].weight_adjusted = new_weight;
          }
        }
      }
      utility_sum_so_far += utility * new_weight;
      weight_sum_so_far += new_weight;
      //rawPolicyUtilitySumSoFar += utility * rawPolicy;
      raw_policy_sum_so_far += raw_policy;
    }
    weight_sum_so_far
  }
}
