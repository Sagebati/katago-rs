// Ported from KataGo cpp/search/searchtimehelpers.cpp @ ba938676d7f42d70950b3a535af2466fb642008c
//
// PORT NOTE: numVisitsNeededToBeNonFutile, computeUpperBoundVisitsLeftDueToTime and
// recomputeSearchTimeLimit are non-const upstream but only read the Search; thread 0's search loop
// calls them through &self (PORTING.md §11), so they take &self.

use crate::core::global::KataError;
use crate::game::board::Loc;
use crate::port::cpp;
use crate::port::math;
use crate::search::search::Search;
use crate::search::timecontrols::TimeControls;

impl Search {
  /// upstream: Search::numVisitsNeededToBeNonFutile (searchtimehelpers.cpp)
  pub(crate) fn num_visits_needed_to_be_non_futile(&self, max_visits_move_visits: f64) -> f64 {
    let required_visits: f64 = self.search_params.futile_visits_threshold * max_visits_move_visits;
    //In the case where we're playing high temperature, also require that we can't get to more than a 1:100 odds of playing the move.
    let chosen_move_temperature: f64 = self.interpolate_early(
      self.search_params.chosen_move_temperature_halflife,
      self.search_params.chosen_move_temperature_early,
      self.search_params.chosen_move_temperature,
    );
    if chosen_move_temperature < 1e-3 {
      return required_visits;
    }
    let required_visits_due_to_temp: f64 = max_visits_move_visits * math::powf(0.01, chosen_move_temperature);
    cpp::cpp_min(required_visits, required_visits_due_to_temp)
  }

  /// upstream: Search::computeUpperBoundVisitsLeftDueToTime (searchtimehelpers.cpp)
  pub(crate) fn compute_upper_bound_visits_left_due_to_time(&self, root_visits: i64, time_used: f64, planned_time_limit: f64) -> f64 {
    if root_visits <= 1 {
      return 1e30;
    }
    let time_thought_so_far: f64 = self.effective_search_time_carried_over + time_used;
    let time_left_planned: f64 = planned_time_limit - time_used;
    //Require at least a tenth of a second of search to begin to trust an estimate of visits/time.
    if time_thought_so_far < 0.1 {
      return 1e30;
    }

    let proportion_of_time_thought_left: f64 = time_left_planned / time_thought_so_far;
    (proportion_of_time_thought_left * root_visits as f64 + self.search_params.num_threads as f64 - 1.0).ceil()
  }

  /// upstream: Search::recomputeSearchTimeLimit (searchtimehelpers.cpp)
  /// PORT NOTE: throws (TimeControls::getTime) -> Result.
  pub(crate) fn recompute_search_time_limit(&self, tc: &TimeControls, time_used: f64, search_factor: f64, root_visits: i64) -> Result<f64, KataError> {
    let mut tc_min: f64 = 0.0;
    let mut tc_rec: f64 = 0.0;
    let mut tc_max: f64 = 0.0;
    tc.get_time(&self.root_board, &self.root_history, self.search_params.lag_buffer, &mut tc_min, &mut tc_rec, &mut tc_max)?;

    tc_rec *= self.search_params.overallocate_time_factor;

    if self.search_params.midgame_time_factor != 1.0 {
      let board_area_scale: f64 = (self.root_board.x_size * self.root_board.y_size) as f64 / 361.0;
      let mut presumed_turn_number: f64 = self.root_history.get_current_turn_number() as f64;
      if presumed_turn_number < 0.0 {
        presumed_turn_number = 0.0;
      }

      let mut mid_game_weight: f64;
      if presumed_turn_number < self.search_params.midgame_turn_peak_time * board_area_scale {
        mid_game_weight = presumed_turn_number / (self.search_params.midgame_turn_peak_time * board_area_scale);
      } else {
        mid_game_weight = math::exp(
          -(presumed_turn_number - self.search_params.midgame_turn_peak_time * board_area_scale) / (self.search_params.endgame_turn_time_decay * board_area_scale),
        );
      }
      if mid_game_weight < 0.0 {
        mid_game_weight = 0.0;
      }
      if mid_game_weight > 1.0 {
        mid_game_weight = 1.0;
      }

      tc_rec *= 1.0 + mid_game_weight * (self.search_params.midgame_time_factor - 1.0);
    }

    if self.search_params.obvious_moves_time_factor < 1.0 {
      let mut surprise: f64 = 0.0;
      let mut search_entropy: f64 = 0.0;
      let mut policy_entropy: f64 = 0.0;
      let suc: bool = self.get_policy_surprise_and_entropy(&mut surprise, &mut search_entropy, &mut policy_entropy);
      if suc {
        //If the original policy was confident and the surprise is low, then this is probably an "obvious" move.
        let obviousness_by_entropy: f64 = math::exp(-policy_entropy / self.search_params.obvious_moves_policy_entropy_tolerance);
        let obviousness_by_surprise: f64 = math::exp(-surprise / self.search_params.obvious_moves_policy_surprise_tolerance);
        let obviousness_weight: f64 = cpp::cpp_min(obviousness_by_entropy, obviousness_by_surprise);
        tc_rec *= 1.0 + obviousness_weight * (self.search_params.obvious_moves_time_factor - 1.0);
      }
    }

    if tc_rec > 1e-20 {
      let remaining_time_needed: f64 = tc_rec - self.effective_search_time_carried_over;
      let remaining_time_needed_factor: f64 = remaining_time_needed / tc_rec;
      //TODO this is a bit conservative relative to old behavior, it might be of slightly detrimental value, needs testing.
      //Apply softplus so that we still do a tiny bit of search even in the presence of variable search time instead of instamoving,
      //there are some benefits from root-level search due to broader root exploration and the cost is small, also we may be over
      //counting the ponder benefit if search is faster on this node than on the previous turn.
      tc_rec = tc_rec * cpp::cpp_min(1.0, math::ln(1.0 + math::exp(remaining_time_needed_factor * 6.0)) / 6.0);
    }

    //Make sure we're not wasting time
    tc_rec = tc.round_up_time_limit_if_needed(self.search_params.lag_buffer, time_used, tc_rec);
    if tc_rec > tc_max {
      tc_rec = tc_max;
    }

    //After rounding up time, check if with our planned rounded time, anything is futile to search
    if self.search_params.futile_visits_threshold > 0.0 {
      let upper_bound_visits_left_due_to_time: f64 = self.compute_upper_bound_visits_left_due_to_time(root_visits, time_used, tc_rec);
      if upper_bound_visits_left_due_to_time < self.search_params.futile_visits_threshold * root_visits as f64 {
        let mut locs: Vec<Loc> = Vec::new();
        let mut play_selection_values: Vec<f64> = Vec::new();
        let mut visit_counts: Vec<f64> = Vec::new();
        let suc: bool = self.get_play_selection_values_ret_visit_counts(&mut locs, &mut play_selection_values, Some(&mut visit_counts), 1.0);
        if suc && !play_selection_values.is_empty() {
          //This may fail to hold if we have no actual visits and play selections are being pulled from stuff like raw policy
          if play_selection_values.len() == visit_counts.len() {
            let num_moves: i32 = play_selection_values.len() as i32;
            let mut max_visits_idx: i32 = 0;
            let mut best_move_idx: i32 = 0;
            for i in 1..num_moves {
              if play_selection_values[i as usize] > play_selection_values[best_move_idx as usize] {
                best_move_idx = i;
              }
              if visit_counts[i as usize] > visit_counts[max_visits_idx as usize] {
                max_visits_idx = i;
              }
            }
            if max_visits_idx == best_move_idx {
              let required_visits: f64 = self.num_visits_needed_to_be_non_futile(visit_counts[max_visits_idx as usize]);
              let mut found_possible_alternative_move: bool = false;
              for i in 0..num_moves {
                if i == best_move_idx {
                  continue;
                }
                if visit_counts[i as usize] + upper_bound_visits_left_due_to_time >= required_visits {
                  found_possible_alternative_move = true;
                  break;
                }
              }
              if !found_possible_alternative_move {
                //We should stop search now - set our desired thinking to very slightly smaller than what we used.
                tc_rec = time_used * (1.0 - (1e-10));
              }
            }
          }
        }
      }
    }

    //Make sure we're not wasting time, even after considering that we might want to stop early
    tc_rec = tc.round_up_time_limit_if_needed(self.search_params.lag_buffer, time_used, tc_rec);
    if tc_rec > tc_max {
      tc_rec = tc_max;
    }

    //Apply caps and search factor
    //Since searchFactor is mainly used for friendliness (like, play faster after many passes)
    //we allow it to violate the min time.
    if tc_rec < tc_min {
      tc_rec = tc_min;
    }
    tc_rec *= search_factor;
    if tc_rec > tc_max {
      tc_rec = tc_max;
    }

    Ok(tc_rec)
  }
}
