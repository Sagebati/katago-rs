// Ported from KataGo cpp/search/searchexplorehelpers.cpp @ ba938676d7f42d70950b3a535af2466fb642008c
//
// PORT NOTE: playout-path helpers take &self (PORTING.md §3/§11). `SearchThread* thread` is
// nullable (searchresults.cpp passes NULL outside search): `Option<&mut SearchThread>`.
// Anti-mirror (searchmirror.cpp) and the eval cache (evalcache.cpp) are deferred (D7): their call
// sites are kept behind their upstream conditions, which cannot hold in the port.

use std::rc::Rc;

use crate::game::board::Board;
use crate::game::board::Loc;
use crate::game::board::Player;
use crate::game::board::C_EMPTY;
use crate::game::board::P_BLACK;
use crate::game::board::P_WHITE;
use crate::neuralnet::nninputs::NNOutput;
use crate::neuralnet::nninputs::NNPos;
use crate::port::cpp;
use crate::port::math;
use crate::port::node_arena::NodeIdx;
use crate::search::search::Search;
use crate::search::search::SearchThread;
use crate::search::searchnode::SearchChildPointer;
use crate::search::searchnode::SearchNode;
use crate::search::searchnode::SearchNodeChildrenReference;
use crate::search::searchnode::SearchNodeState;
use crate::search::searchparams::SearchParams;

/// upstream: cpuctExploration (searchexplorehelpers.cpp) static
fn cpuct_exploration(total_child_weight: f64, search_params: &SearchParams) -> f64 {
  search_params.cpuct_exploration
    + search_params.cpuct_exploration_log
      * math::ln((total_child_weight + search_params.cpuct_exploration_base) / search_params.cpuct_exploration_base)
}

/// upstream: cpuctExplorationHuman (searchexplorehelpers.cpp) static
fn cpuct_exploration_human(total_child_weight: f64, search_params: &SearchParams) -> f64 {
  search_params.human_sl_cpuct_exploration + search_params.human_sl_cpuct_permanent * total_child_weight.sqrt()
}

/// upstream: TOTALCHILDWEIGHT_PUCT_OFFSET (searchexplorehelpers.cpp) static
//Tiny constant to add to numerator of puct formula to make it positive
//even when visits = 0.
const TOTALCHILDWEIGHT_PUCT_OFFSET: f64 = 0.01;

/// upstream: maybeApplyWideRootNoise (searchexplorehelpers.cpp) static
fn maybe_apply_wide_root_noise(
  child_utility: &mut f64,
  nn_policy_prob: &mut f32,
  search_params: &SearchParams,
  thread: &mut SearchThread,
  parent: &SearchNode,
) {
  //For very large wideRootNoise, go ahead and also smooth out the policy
  *nn_policy_prob = math::powf(*nn_policy_prob as f64, 1.0 / (4.0 * search_params.wide_root_noise + 1.0)) as f32;
  if thread.rand.next_bool(0.5) {
    let bonus: f64 = search_params.wide_root_noise * thread.rand.next_gaussian().abs();
    if parent.next_pla == P_WHITE {
      *child_utility += bonus;
    } else {
      *child_utility -= bonus;
    }
  }
}

impl Search {
  /// upstream: Search::getExploreScaling (searchexplorehelpers.cpp)
  pub(crate) fn get_explore_scaling(&self, total_child_weight: f64, parent_utility_stdev_factor: f64) -> f64 {
    cpuct_exploration(total_child_weight, &self.search_params) * (total_child_weight + TOTALCHILDWEIGHT_PUCT_OFFSET).sqrt() * parent_utility_stdev_factor
  }

  /// upstream: Search::getExploreScalingHuman (searchexplorehelpers.cpp)
  pub(crate) fn get_explore_scaling_human(&self, total_child_weight: f64) -> f64 {
    cpuct_exploration_human(total_child_weight, &self.search_params) * (total_child_weight + TOTALCHILDWEIGHT_PUCT_OFFSET).sqrt()
  }

  /// upstream: Search::getExploreSelectionValue (searchexplorehelpers.cpp)
  /// PORT NOTE: `pub` (upstream private, PORTING.md §1 would give pub(crate)) so the crafted parity oracle
  /// tests/search_select_diff.rs can call it.
  pub fn get_explore_selection_value(&self, explore_scaling: f64, nn_policy_prob: f64, child_weight: f64, child_utility: f64, pla: Player) -> f64 {
    if nn_policy_prob < 0.0 {
      return Search::POLICY_ILLEGAL_SELECTION_VALUE;
    }

    let explore_component: f64 = explore_scaling * nn_policy_prob / (1.0 + child_weight);

    //At the last moment, adjust value to be from the player's perspective, so that players prefer values in their favor
    //rather than in white's favor
    let value_component: f64 = if pla == P_WHITE { child_utility } else { -child_utility };
    explore_component + value_component
  }

  /// upstream: Search::getExploreSelectionValueInverse (searchexplorehelpers.cpp)
  //Return the childWeight that would make Search::getExploreSelectionValue return the given explore selection value.
  //Or return 0, if it would be less than 0.
  /// PORT NOTE: search.h declares the first two parameters as (exploreScaling, exploreSelectionValue)
  /// but searchexplorehelpers.cpp defines them as (exploreSelectionValue, exploreScaling), and the body
  /// and the caller use the definition's order; the Rust names follow the definition.
  /// PORT NOTE: `pub` (upstream private, PORTING.md §1 would give pub(crate)) so the crafted parity oracle
  /// tests/search_select_diff.rs can call it.
  pub fn get_explore_selection_value_inverse(
    &self,
    explore_selection_value: f64,
    explore_scaling: f64,
    nn_policy_prob: f64,
    child_utility: f64,
    pla: Player,
  ) -> f64 {
    if nn_policy_prob < 0.0 {
      return 0.0;
    }
    let value_component: f64 = if pla == P_WHITE { child_utility } else { -child_utility };

    let explore_component: f64 = explore_selection_value - value_component;
    let explore_component_scaling: f64 = explore_scaling * nn_policy_prob;

    //Guard against float weirdness
    if explore_component <= 0.0 {
      return 1e100;
    }

    let mut child_weight: f64 = explore_component_scaling / explore_component - 1.0;
    if child_weight < 0.0 {
      child_weight = 0.0;
    }
    child_weight
  }

  /// upstream: Search::getExploreSelectionValueOfChild (searchexplorehelpers.cpp)
  /// PORT NOTE: `const SearchNode* child` is never NULL at the callers: `&SearchNode`.
  pub(crate) fn get_explore_selection_value_of_child(
    &self,
    parent: &SearchNode,
    parent_policy_probs: &[f32],
    child: &SearchNode,
    move_loc: Loc,
    explore_scaling: f64,
    total_child_weight: f64,
    child_edge_visits: i64,
    fpu_value: f64,
    parent_utility: f64,
    parent_weight_per_visit: f64,
    is_during_search: bool,
    anti_mirror: bool,
    max_child_weight: f64,
    count_edge_visit: bool,
    mut thread: Option<&mut SearchThread>,
  ) -> f64 {
    let _ = parent_utility;
    let move_pos: i32 = self.get_pos(move_loc);
    let mut nn_policy_prob: f32 = parent_policy_probs[move_pos as usize];

    let child_virtual_losses: i32 = child.virtual_losses.get() /*acquire*/;
    let child_visits: i64 = child.stats.visits.get() /*acquire*/;
    let utility_avg: f64 = child.stats.utility_avg.get() /*acquire*/;
    let score_mean_avg: f64 = child.stats.score_mean_avg.get() /*acquire*/;
    let score_mean_sq_avg: f64 = child.stats.score_mean_sq_avg.get() /*acquire*/;
    let mut child_weight: f64;
    if count_edge_visit {
      child_weight = child.stats.get_child_weight_child_visits(child_edge_visits, child_visits);
    } else {
      child_weight = child.stats.weight_sum.get() /*acquire*/;
    }

    //It's possible that childVisits is actually 0 here with multithreading because we're visiting this node while a child has
    //been expanded but its thread not yet finished its first visit.
    //It's also possible that we observe childWeight <= 0 even though childVisits >= due to multithreading, the two could
    //be out of sync briefly since they are separate atomics.
    let mut child_utility: f64;
    if child_visits <= 0 || child_weight <= 0.0 {
      child_utility = fpu_value;
    } else {
      child_utility = utility_avg;

      //Tiny adjustment for passing
      let ending_score_bonus: f64 = self.get_ending_white_score_bonus(parent, move_loc);
      if ending_score_bonus != 0.0 {
        child_utility += self.get_score_utility_diff(score_mean_avg, score_mean_sq_avg, ending_score_bonus);
      }
    }

    //Virtual losses to direct threads down different paths
    if child_virtual_losses > 0 {
      let virtual_loss_weight: f64 = child_virtual_losses as f64 * self.search_params.num_virtual_losses_per_thread;

      let utility_radius: f64 =
        self.search_params.win_loss_utility_factor + self.search_params.static_score_utility_factor + self.search_params.dynamic_score_utility_factor;
      let virtual_loss_utility: f64 = if parent.next_pla == P_WHITE { -utility_radius } else { utility_radius };
      let virtual_loss_weight_frac: f64 = virtual_loss_weight / (virtual_loss_weight + cpp::cpp_max(0.25, child_weight));
      child_utility = child_utility + (virtual_loss_utility - child_utility) * virtual_loss_weight_frac;
      child_weight += virtual_loss_weight;
    }

    if is_during_search && (Some(parent.idx) == self.root_node) && count_edge_visit {
      //Futile visits pruning - skip this move if the amount of time we have left to search is too small, assuming
      //its average weight per visit is maintained.
      //We use childVisits rather than childEdgeVisits for the final estimate since when childEdgeVisits < childVisits, adding new visits is instant.
      if self.search_params.futile_visits_threshold > 0.0 {
        let required_weight: f64 = self.search_params.futile_visits_threshold * max_child_weight;
        //Avoid divide by 0 by adding a prior equal to the parent's weight per visit
        let average_visits_per_weight: f64 = (child_edge_visits as f64 + 1.0) / (child_weight + parent_weight_per_visit);
        let estimated_required_visits: f64 = required_weight * average_visits_per_weight;
        let upper_bound_visits_left: f64 = thread.as_deref().expect("isDuringSearch implies a thread").upper_bound_visits_left;
        if child_visits as f64 + upper_bound_visits_left < estimated_required_visits {
          return Search::FUTILE_VISITS_PRUNE_VALUE;
        }
      }
      //Hack to get the root to funnel more visits down child branches
      if self.search_params.root_desired_per_child_visits_coeff > 0.0 {
        if nn_policy_prob > 0.0
          && child_weight < (nn_policy_prob as f64 * total_child_weight * self.search_params.root_desired_per_child_visits_coeff).sqrt()
        {
          return 1e20;
        }
      }
      //Hack for hintloc - must search this move almost as often as the most searched move
      if self.root_hint_loc != Board::NULL_LOC && move_loc == self.root_hint_loc {
        let average_weight_per_visit: f64 = (child_weight + parent_weight_per_visit) / (child_visits as f64 + 1.0);
        let children: SearchNodeChildrenReference = parent.get_children();
        let children_capacity: i32 = children.get_capacity();
        for i in 0..children_capacity {
          let child_pointer: &SearchChildPointer = children.index(i);
          let c: NodeIdx = match child_pointer.get_if_allocated() {
            None => break,
            Some(c) => c,
          };
          let c_edge_visits: i64 = child_pointer.get_edge_visits();
          let c_weight: f64 = self.node_arena.get(c).stats.get_child_weight(c_edge_visits);
          if child_weight + average_weight_per_visit < c_weight * 0.8 {
            return 1e20;
          }
        }
      }

      if self.search_params.wide_root_noise > 0.0 && nn_policy_prob >= 0.0 {
        maybe_apply_wide_root_noise(
          &mut child_utility,
          &mut nn_policy_prob,
          &self.search_params,
          thread.as_mut().expect("isDuringSearch implies a thread"),
          parent,
        );
      }
    }
    if is_during_search && anti_mirror && nn_policy_prob >= 0.0 && count_edge_visit {
      // TODO(port): deferred (searchmirror.cpp):
      // maybeApplyAntiMirrorPolicy(nnPolicyProb, moveLoc, parentPolicyProbs, parent.nextPla, thread);
      // maybeApplyAntiMirrorForcedExplore(childUtility, parentUtility, moveLoc, parentPolicyProbs, childWeight, totalChildWeight, parent.nextPla, thread, parent);
      // Unreachable: antiMirror is false at every caller while mirroringPla stays C_EMPTY.
      unreachable!("anti-mirror is deferred (searchmirror.cpp)");
    }

    self.get_explore_selection_value(explore_scaling, nn_policy_prob as f64, child_weight, child_utility, parent.next_pla)
  }

  /// upstream: Search::getNewExploreSelectionValue (searchexplorehelpers.cpp)
  pub(crate) fn get_new_explore_selection_value(
    &self,
    parent: &SearchNode,
    explore_scaling: f64,
    mut nn_policy_prob: f32,
    fpu_value: f64,
    parent_weight_per_visit: f64,
    max_child_weight: f64,
    count_edge_visit: bool,
    mut thread: Option<&mut SearchThread>,
  ) -> f64 {
    let child_weight: f64 = 0.0;
    let mut child_utility: f64 = fpu_value;
    if Some(parent.idx) == self.root_node && count_edge_visit {
      //Futile visits pruning - skip this move if the amount of time we have left to search is too small
      if self.search_params.futile_visits_threshold > 0.0 {
        //Avoid divide by 0 by adding a prior equal to the parent's weight per visit
        let average_visits_per_weight: f64 = 1.0 / parent_weight_per_visit;
        let required_weight: f64 = self.search_params.futile_visits_threshold * max_child_weight;
        let estimated_required_visits: f64 = required_weight * average_visits_per_weight;
        if thread.as_deref().expect("thread is non-NULL at every caller").upper_bound_visits_left < estimated_required_visits {
          return Search::FUTILE_VISITS_PRUNE_VALUE;
        }
      }
      if self.search_params.wide_root_noise > 0.0 {
        maybe_apply_wide_root_noise(
          &mut child_utility,
          &mut nn_policy_prob,
          &self.search_params,
          thread.as_mut().expect("thread is non-NULL at every caller"),
          parent,
        );
      }
    }
    self.get_explore_selection_value(explore_scaling, nn_policy_prob as f64, child_weight, child_utility, parent.next_pla)
  }

  /// upstream: Search::getReducedPlaySelectionWeight (searchexplorehelpers.cpp)
  /// PORT NOTE: `const SearchNode* child` is never NULL at the callers: `&SearchNode`.
  pub(crate) fn get_reduced_play_selection_weight(
    &self,
    parent: &SearchNode,
    parent_policy_probs: &[f32],
    child: &SearchNode,
    move_loc: Loc,
    explore_scaling: f64,
    child_edge_visits: i64,
    best_child_explore_selection_value: f64,
  ) -> f64 {
    debug_assert!(Some(parent.idx) == self.root_node);
    let move_pos: i32 = self.get_pos(move_loc);
    let nn_policy_prob: f32 = parent_policy_probs[move_pos as usize];

    let child_visits: i64 = child.stats.visits.get() /*acquire*/;
    let score_mean_avg: f64 = child.stats.score_mean_avg.get() /*acquire*/;
    let score_mean_sq_avg: f64 = child.stats.score_mean_sq_avg.get() /*acquire*/;
    let utility_avg: f64 = child.stats.utility_avg.get() /*acquire*/;
    let child_weight: f64 = child.stats.get_child_weight_child_visits(child_edge_visits, child_visits);

    //Child visits may be 0 if this function is called in a multithreaded context, such as during live analysis
    //Child weight may also be 0 if it's out of sync.
    if child_visits <= 0 || child_weight <= 0.0 {
      return 0.0;
    }

    //Tiny adjustment for passing
    let ending_score_bonus: f64 = self.get_ending_white_score_bonus(parent, move_loc);
    let mut child_utility: f64 = utility_avg;
    if ending_score_bonus != 0.0 {
      child_utility += self.get_score_utility_diff(score_mean_avg, score_mean_sq_avg, ending_score_bonus);
    }

    let child_weight_we_retrospectively_wanted: f64 =
      self.get_explore_selection_value_inverse(best_child_explore_selection_value, explore_scaling, nn_policy_prob as f64, child_utility, parent.next_pla);
    if child_weight > child_weight_we_retrospectively_wanted {
      return child_weight_we_retrospectively_wanted;
    }
    child_weight
  }

  /// upstream: Search::getFpuValueForChildrenAssumeVisited (searchexplorehelpers.cpp)
  pub(crate) fn get_fpu_value_for_children_assume_visited(
    &self,
    node: &SearchNode,
    pla: Player,
    is_root: bool,
    policy_prob_mass_visited: f64,
    parent_utility: &mut f64,
    parent_weight_per_visit: &mut f64,
    parent_utility_stdev_factor: &mut f64,
  ) -> f64 {
    let visits: i64 = node.stats.visits.get() /*acquire*/;
    let weight_sum: f64 = node.stats.weight_sum.get() /*acquire*/;
    let utility_avg: f64 = node.stats.utility_avg.get() /*acquire*/;
    let mut utility_sq_avg: f64 = node.stats.utility_sq_avg.get() /*acquire*/;

    debug_assert!(visits > 0);
    debug_assert!(weight_sum > 0.0);
    *parent_weight_per_visit = weight_sum / visits as f64;
    *parent_utility = utility_avg;
    let variance_prior: f64 = self.search_params.cpuct_utility_stdev_prior * self.search_params.cpuct_utility_stdev_prior;
    let variance_prior_weight: f64 = self.search_params.cpuct_utility_stdev_prior_weight;
    let parent_utility_stdev: f64;
    if visits <= 0 || weight_sum <= 1.0 {
      parent_utility_stdev = self.search_params.cpuct_utility_stdev_prior;
    } else {
      let utility_sq: f64 = *parent_utility * *parent_utility;
      //Make sure we're robust to numerical precision issues or threading desync of these values, so we don't observe negative variance
      if utility_sq_avg < utility_sq {
        utility_sq_avg = utility_sq;
      }
      parent_utility_stdev = cpp::cpp_max(
        0.0,
        ((utility_sq + variance_prior) * variance_prior_weight + utility_sq_avg * weight_sum) / (variance_prior_weight + weight_sum - 1.0) - utility_sq,
      )
      .sqrt();
    }
    *parent_utility_stdev_factor =
      1.0 + self.search_params.cpuct_utility_stdev_scale * (parent_utility_stdev / self.search_params.cpuct_utility_stdev_prior - 1.0);

    let mut parent_utility_for_fpu: f64 = *parent_utility;
    if self.search_params.fpu_parent_weight_by_visited_policy {
      let avg_weight: f64 = cpp::cpp_min(1.0, math::powf(policy_prob_mass_visited, self.search_params.fpu_parent_weight_by_visited_policy_pow));
      parent_utility_for_fpu = avg_weight * *parent_utility
        + (1.0 - avg_weight) * self.get_utility_from_nn(&node.get_nn_output().expect("an expanded node has an nnOutput"));
    } else if self.search_params.fpu_parent_weight > 0.0 {
      parent_utility_for_fpu = self.search_params.fpu_parent_weight * self.get_utility_from_nn(&node.get_nn_output().expect("an expanded node has an nnOutput"))
        + (1.0 - self.search_params.fpu_parent_weight) * *parent_utility;
    }

    let mut fpu_value: f64;
    {
      let fpu_reduction_max: f64 = if is_root { self.search_params.root_fpu_reduction_max } else { self.search_params.fpu_reduction_max };
      let fpu_loss_prop: f64 = if is_root { self.search_params.root_fpu_loss_prop } else { self.search_params.fpu_loss_prop };
      let utility_radius: f64 =
        self.search_params.win_loss_utility_factor + self.search_params.static_score_utility_factor + self.search_params.dynamic_score_utility_factor;

      let reduction: f64 = fpu_reduction_max * policy_prob_mass_visited.sqrt();
      fpu_value = if pla == P_WHITE { parent_utility_for_fpu - reduction } else { parent_utility_for_fpu + reduction };
      let loss_value: f64 = if pla == P_WHITE { -utility_radius } else { utility_radius };
      fpu_value = fpu_value + (loss_value - fpu_value) * fpu_loss_prop;
    }

    fpu_value
  }

  /// upstream: Search::selectBestChildToDescend (searchexplorehelpers.cpp)
  pub(crate) fn select_best_child_to_descend(
    &self,
    thread: &mut SearchThread,
    node: &SearchNode,
    node_state: SearchNodeState,
    num_children_found: &mut i32,
    best_child_idx: &mut i32,
    best_child_move_loc: &mut Loc,
    count_edge_visit: &mut bool,
    is_root: bool,
  ) {
    debug_assert!(thread.pla == node.next_pla);

    let mut max_selection_value: f64 = Search::POLICY_ILLEGAL_SELECTION_VALUE;
    *best_child_idx = -1;
    *best_child_move_loc = Board::NULL_LOC;
    *count_edge_visit = true;

    let children: SearchNodeChildrenReference = node.get_children_state(node_state);
    let children_capacity: i32 = children.get_capacity();

    let mut policy_prob_mass_visited: f64 = 0.0;
    let mut max_child_weight: f64 = 0.0;
    let mut total_child_weight: f64 = 0.0;
    let mut total_child_edge_visits: i64 = 0;
    // PORT NOTE: the Rcs that `nnOutput` / `humanOutput` point to are held here so `policyProbs` can borrow them.
    let nn_output: Rc<NNOutput> = node.get_nn_output().expect("an expanded node has an nnOutput");
    let human_output_holder: Option<Rc<NNOutput>>;
    let mut policy_probs: &[f32] = &nn_output.get_policy_probs_maybe_noised()[..];
    for i in 0..children_capacity {
      let child_pointer: &SearchChildPointer = children.index(i);
      let child: NodeIdx = match child_pointer.get_if_allocated() {
        None => break,
        Some(c) => c,
      };
      let move_loc: Loc = child_pointer.get_move_loc_relaxed();
      let move_pos: i32 = self.get_pos(move_loc);
      let nn_policy_prob: f32 = policy_probs[move_pos as usize];
      if nn_policy_prob < 0.0 {
        continue;
      }
      policy_prob_mass_visited += nn_policy_prob as f64;

      let edge_visits: i64 = child_pointer.get_edge_visits();
      let child_weight: f64 = self.node_arena.get(child).stats.get_child_weight(edge_visits);

      total_child_weight += child_weight;
      if child_weight > max_child_weight {
        max_child_weight = child_weight;
      }
      total_child_edge_visits += edge_visits;
    }

    let mut use_human_sl: bool = false;
    if let Some(human_evaluator) = &self.human_evaluator {
      if self.search_params.human_sl_profile.initialized || !human_evaluator.requires_sgf_metadata() {
        human_output_holder = node.get_human_output();
        if human_output_holder.is_some() {
          let weightless_prob: f64;
          let weightful_prob: f64;
          if is_root {
            weightless_prob = self.search_params.human_sl_root_explore_prob_weightless;
            weightful_prob = self.search_params.human_sl_root_explore_prob_weightful;
          } else if thread.pla == self.root_pla {
            weightless_prob = self.search_params.human_sl_pla_explore_prob_weightless;
            weightful_prob = self.search_params.human_sl_pla_explore_prob_weightful;
          } else {
            weightless_prob = self.search_params.human_sl_opp_explore_prob_weightless;
            weightful_prob = self.search_params.human_sl_opp_explore_prob_weightful;
          }

          let total_human_prob: f64 = weightless_prob + weightful_prob;
          if total_human_prob > 0.0 {
            let r: f64 = thread.rand.next_double();
            if r < weightless_prob {
              use_human_sl = true;
              *count_edge_visit = false;
            } else if r < total_human_prob {
              use_human_sl = true;
            }
          }
        }

        // Swap out policy and also recompute policy prob mass visited
        if use_human_sl {
          policy_prob_mass_visited = 0.0;
          let human_output: &NNOutput = human_output_holder.as_deref().expect("useHumanSL implies a humanOutput");
          policy_probs = &human_output.get_policy_probs_maybe_noised()[..];
          for i in 0..children_capacity {
            let child_pointer: &SearchChildPointer = children.index(i);
            if child_pointer.get_if_allocated().is_none() {
              break;
            }
            let move_loc: Loc = child_pointer.get_move_loc_relaxed();
            let move_pos: i32 = self.get_pos(move_loc);
            let nn_policy_prob: f32 = policy_probs[move_pos as usize];
            if nn_policy_prob < 0.0 {
              continue;
            }
            policy_prob_mass_visited += nn_policy_prob as f64;
          }
        }
      }
    }

    //Probability mass should not sum to more than 1, giving a generous allowance
    //for floating point error.
    debug_assert!(policy_prob_mass_visited <= 1.0001);

    //If we're doing a weightless visit, then we should redo PUCT to operate on child node weight, not child edge weight
    if !*count_edge_visit {
      total_child_weight = 0.0;
      max_child_weight = 0.0;
      for i in 0..children_capacity {
        let child_pointer: &SearchChildPointer = children.index(i);
        let child: NodeIdx = match child_pointer.get_if_allocated() {
          None => break,
          Some(c) => c,
        };
        let child_weight: f64 = self.node_arena.get(child).stats.weight_sum.get() /*acquire*/;
        total_child_weight += child_weight;
        if child_weight > max_child_weight {
          max_child_weight = child_weight;
        }
      }
    }

    //First play urgency
    let mut parent_utility: f64 = 0.0;
    let mut parent_weight_per_visit: f64 = 0.0;
    let mut parent_utility_stdev_factor: f64 = 0.0;
    let fpu_value: f64 = self.get_fpu_value_for_children_assume_visited(
      node,
      thread.pla,
      is_root,
      policy_prob_mass_visited,
      &mut parent_utility,
      &mut parent_weight_per_visit,
      &mut parent_utility_stdev_factor,
    );

    let mut poses_with_child_buf: [bool; NNPos::MAX_NN_POLICY_SIZE as usize] = [false; NNPos::MAX_NN_POLICY_SIZE as usize]; // Initialize all to false
    // TODO(port): deferred (searchmirror.cpp): the last conjunct is `isMirroringSinceSearchStart(thread.history,0)`;
    // unreachable while updateMirroring is deferred (mirroringPla stays C_EMPTY).
    let anti_mirror: bool = self.search_params.anti_mirror
      && self.mirroring_pla != C_EMPTY
      && unreachable!("anti-mirror is deferred (searchmirror.cpp)");

    let explore_scaling: f64;
    if use_human_sl {
      explore_scaling = self.get_explore_scaling_human(total_child_weight);
    } else {
      explore_scaling = self.get_explore_scaling(total_child_weight, parent_utility_stdev_factor);
    }

    //Try all existing children
    //Also count how many children we actually find
    *num_children_found = 0;
    for i in 0..children_capacity {
      let child_pointer: &SearchChildPointer = children.index(i);
      let child: NodeIdx = match child_pointer.get_if_allocated() {
        None => break,
        Some(c) => c,
      };
      *num_children_found += 1;
      let child_edge_visits: i64 = child_pointer.get_edge_visits();

      let move_loc: Loc = child_pointer.get_move_loc_relaxed();
      let is_during_search: bool = true;
      let child_node: Rc<SearchNode> = self.node_arena.get(child);
      let selection_value: f64 = self.get_explore_selection_value_of_child(
        node,
        policy_probs,
        &child_node,
        move_loc,
        explore_scaling,
        total_child_weight,
        child_edge_visits,
        fpu_value,
        parent_utility,
        parent_weight_per_visit,
        is_during_search,
        anti_mirror,
        max_child_weight,
        *count_edge_visit,
        Some(&mut *thread),
      );
      if selection_value > max_selection_value {
        // if(child->state.load(std::memory_order_seq_cst) == SearchNode::STATE_EVALUATING) {
        //   selectionValue -= EVALUATING_SELECTION_VALUE_PENALTY;
        //   if(isRoot && child->prevMoveLoc == Location::ofString("K4",thread.board)) {
        //     out << "ouch" << "\n";
        //   }
        // }
        max_selection_value = selection_value;
        *best_child_idx = i;
        *best_child_move_loc = move_loc;
      }

      poses_with_child_buf[self.get_pos(move_loc) as usize] = true;
    }

    let avoid_move_until_by_loc: &Vec<i32> =
      if thread.pla == P_BLACK { &self.avoid_move_until_by_loc_black } else { &self.avoid_move_until_by_loc_white };

    //Try all the things in the eval cache that are moves we haven't visited yet.
    //Use the normal new explore selection value for them but with their eval cache utility instead of FPU.
    //Might explore things out of descending policy order!
    if self.search_params.use_eval_cache
      && self.search_params.use_graph_search
      && node.eval_cache_entry.borrow().is_some()
      && self.mirroring_pla == C_EMPTY
      && !node.force_non_terminal
    {
      // TODO(port): deferred (evalcache.cpp): the loop over `node.evalCacheEntry->firstExploreEvals` that scores each
      // not-yet-tried legal (root-allowed, not avoided) move with getNewExploreSelectionValue using the cached
      // `getResultUtility(eval.avgWinLoss,0.0) + getScoreUtility(eval.avgScoreMean, eval.avgScoreMean * eval.avgScoreMean)`
      // instead of fpuValue, and takes it as the best child (index numChildrenFound) if it beats maxSelectionValue.
      // EvalCacheEntry has no values (D7), so this is unreachable.
      let entry: Rc<crate::search::searchnode::EvalCacheEntry> = node.eval_cache_entry.borrow().clone().expect("checked is_some");
      match *entry {}
    }

    //Try the new child with the best policy value
    let mut best_new_move_loc: Loc = Board::NULL_LOC;
    let mut best_new_nn_policy_prob: f32 = -1.0f32;
    for move_pos in 0..self.policy_size {
      let already_tried: bool = poses_with_child_buf[move_pos as usize];
      if already_tried {
        continue;
      }

      //Quit immediately for illegal moves
      let nn_policy_prob: f32 = policy_probs[move_pos as usize];
      if nn_policy_prob < 0.0 {
        continue;
      }

      let move_loc: Loc = NNPos::pos_to_loc(move_pos, thread.board.x_size, thread.board.y_size, self.nn_x_len, self.nn_y_len);
      if move_loc == Board::NULL_LOC {
        continue;
      }

      //Special logic for the root
      if is_root {
        debug_assert!(thread.board.pos_hash == self.root_board.pos_hash);
        debug_assert!(thread.pla == self.root_pla);
        if !self.is_allowed_root_move(move_loc) {
          continue;
        }
      }
      if !avoid_move_until_by_loc.is_empty() {
        debug_assert!(avoid_move_until_by_loc.len() >= Board::MAX_ARR_SIZE as usize);
        let until_depth: i32 = avoid_move_until_by_loc[move_loc as usize];
        // PORT NOTE: size_t arithmetic: the difference is unsigned and untilDepth converts to size_t (§6 item 8).
        if (thread.history.move_history.len() as u64).wrapping_sub(self.root_history.move_history.len() as u64) < until_depth as u64 {
          continue;
        }
      }

      if anti_mirror {
        // TODO(port): deferred (searchmirror.cpp): `maybeApplyAntiMirrorPolicy(nnPolicyProb, moveLoc, policyProbs, node.nextPla, &thread);`
        unreachable!("anti-mirror is deferred (searchmirror.cpp)");
      }

      if nn_policy_prob > best_new_nn_policy_prob {
        best_new_nn_policy_prob = nn_policy_prob;
        best_new_move_loc = move_loc;
      }
    }
    if best_new_move_loc != Board::NULL_LOC {
      let selection_value: f64 = self.get_new_explore_selection_value(
        node,
        explore_scaling,
        best_new_nn_policy_prob,
        fpu_value,
        parent_weight_per_visit,
        max_child_weight,
        *count_edge_visit,
        Some(&mut *thread),
      );
      if selection_value > max_selection_value {
        max_selection_value = selection_value;
        *best_child_idx = *num_children_found;
        *best_child_move_loc = best_new_move_loc;
      }
    }
    let _ = max_selection_value;

    if total_child_edge_visits >= 2
      && self.search_params.enable_more_passing_hacks
      && thread.history.pass_would_end_phase(&thread.board, thread.pla)
      && avoid_move_until_by_loc.is_empty()
    // Don't force playouts if there's any chance we're specifying specific moves since we don't want to force an avoided move
    {
      let mut has_pass_move: bool = false;
      let mut has_non_pass_move: bool = false;
      for i in 0..children_capacity {
        let child_pointer: &SearchChildPointer = children.index(i);
        if child_pointer.get_if_allocated().is_none() {
          break;
        }
        let move_loc: Loc = child_pointer.get_move_loc_relaxed();
        if move_loc == Board::PASS_LOC {
          has_pass_move = true;
        } else {
          has_non_pass_move = true;
        }
      }
      if !has_pass_move && *best_child_move_loc != Board::PASS_LOC && *best_child_move_loc != Board::NULL_LOC {
        *best_child_idx = *num_children_found;
        *best_child_move_loc = Board::PASS_LOC;
        *count_edge_visit = false;
        // Specifically for these special extra-pass search playouts, we don't count them for the purpose of visit/playout limits.
        thread.should_count_playout = false;
      } else if !has_non_pass_move
        && *best_child_move_loc == Board::PASS_LOC
        && best_new_move_loc != Board::PASS_LOC
        && best_new_move_loc != Board::NULL_LOC
      {
        *best_child_idx = *num_children_found;
        *best_child_move_loc = best_new_move_loc;
        *count_edge_visit = false;
        // Specifically for these special extra-pass search playouts, we don't count them for the purpose of visit/playout limits.
        thread.should_count_playout = false;
      }
    }
  }
}
