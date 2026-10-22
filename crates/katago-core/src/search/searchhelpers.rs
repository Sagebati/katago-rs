// Ported from KataGo cpp/search/searchhelpers.cpp @ ba938676d7f42d70950b3a535af2466fb642008c
//
// PORT NOTE: `std::shared_ptr<NNOutput>*` results are `Rc<NNOutput>`; maybeAddPolicyNoiseAndTemp,
// addDirichletNoise and computeDirichletAlphaDistribution return Result because
// computeDirichletAlphaDistribution throws and Rand::nextGamma throws (PORTING.md §7). Pattern bonus
// tables are deferred (D7): patternBonusTable is always None.

use std::rc::Rc;

use crate::core::global::KataError;
use crate::core::hash::Hash128;
use crate::core::rand::Rand;
use crate::game::board::get_opp;
use crate::game::board::Board;
use crate::game::board::Loc;
use crate::game::board::Player;
use crate::game::board::P_BLACK;
use crate::game::board::P_WHITE;
use crate::game::rules::Rules;
use crate::neuralnet::nninputs::NNOutput;
use crate::neuralnet::nninputs::NNPos;
use crate::neuralnet::nninputs::ScoreValue;
use crate::port::cpp;
use crate::port::math;
use crate::port::node_arena::NodeIdx;
use crate::search::search::Search;
use crate::search::search::SearchThread;
use crate::search::searchnode::SearchChildPointer;
use crate::search::searchnode::SearchNode;
use crate::search::searchnode::SearchNodeChildrenReference;
use crate::search::searchparams::SearchParams;
use crate::testAssert;

impl Search {
  /// upstream: Search::chooseIndexWithTemperature (searchhelpers.cpp)
  /// PORT NOTE: `processedRelProbsBuf` may be NULL (searchresults.cpp, playutils.cpp): Option.
  pub fn choose_index_with_temperature(
    rand: &mut Rand,
    relative_probs: &[f64],
    num_relative_probs: i32,
    temperature: f64,
    only_below_prob: f64,
    processed_rel_probs_buf: Option<&mut [f64]>,
  ) -> u32 {
    testAssert!(num_relative_probs > 0);
    testAssert!(num_relative_probs <= Board::MAX_ARR_SIZE); //We're just doing this on the stack
    let mut processed_rel_probs: [f64; Board::MAX_ARR_SIZE as usize] = [0.0; Board::MAX_ARR_SIZE as usize];
    let processed_rel_probs_buf: &mut [f64] = match processed_rel_probs_buf {
      Some(b) => b,
      None => &mut processed_rel_probs[..],
    };

    let mut max_rel_prob: f64 = 0.0;
    let mut sum_rel_prob: f64 = 0.0;
    for i in 0..num_relative_probs as usize {
      sum_rel_prob += cpp::cpp_max(0.0, relative_probs[i]);
      if relative_probs[i] > max_rel_prob {
        max_rel_prob = relative_probs[i];
      }
    }
    testAssert!(max_rel_prob > 0.0);
    testAssert!(sum_rel_prob > 0.0);

    //Temperature so close to 0 that we just calculate the max directly
    if temperature <= 1.0e-4 && only_below_prob >= 1.0 {
      let mut best_prob: f64 = relative_probs[0];
      let mut best_idx: i32 = 0;
      processed_rel_probs_buf[0] = 0.0;
      for i in 1..num_relative_probs {
        processed_rel_probs_buf[i as usize] = 0.0;
        if relative_probs[i as usize] > best_prob {
          best_prob = relative_probs[i as usize];
          best_idx = i;
        }
      }
      processed_rel_probs_buf[best_idx as usize] = 1.0;
      best_idx as u32
    }
    //Actual temperature
    else {
      let log_max_rel_prob: f64 = math::ln(max_rel_prob);
      let log_sum_rel_prob: f64 = math::ln(sum_rel_prob);
      let log_only_below_prob: f64 = math::ln(cpp::cpp_max(1e-50, only_below_prob));
      let mut sum: f64 = 0.0;
      for i in 0..num_relative_probs as usize {
        if relative_probs[i] <= 0.0 {
          processed_rel_probs_buf[i] = 0.0;
        } else {
          let log_rel_prob: f64 = math::ln(relative_probs[i]) - log_max_rel_prob;
          let log_rel_prob_threshold: f64 = cpp::cpp_min(0.0, log_only_below_prob + log_sum_rel_prob - log_max_rel_prob);
          let new_log_rel_prob: f64;
          if log_rel_prob > log_rel_prob_threshold {
            new_log_rel_prob = log_rel_prob;
          } else {
            new_log_rel_prob = (log_rel_prob - log_rel_prob_threshold) / temperature + log_rel_prob_threshold;
          }
          processed_rel_probs_buf[i] = math::exp(new_log_rel_prob);
        }
        sum += processed_rel_probs_buf[i];
      }
      testAssert!(sum > 0.0);
      let idx_chosen: u32 = rand.next_uint_rel_probs_n(processed_rel_probs_buf, num_relative_probs as usize);
      idx_chosen
    }
  }

  /// upstream: Search::computeDirichletAlphaDistribution (searchhelpers.cpp)
  /// PORT NOTE: throws -> Result.
  pub fn compute_dirichlet_alpha_distribution(policy_size: i32, policy_probs: &[f32], alpha_distr: &mut [f64]) -> Result<(), KataError> {
    let mut legal_count: i32 = 0;
    for i in 0..policy_size as usize {
      if policy_probs[i] >= 0.0 {
        legal_count += 1;
      }
    }

    if legal_count <= 0 {
      return Err(KataError::StringError(
        "computeDirichletAlphaDistribution: No move with nonnegative policy value - can't even pass?".to_string(),
      ));
    }

    //We're going to generate a gamma draw on each move with alphas that sum up to searchParams.rootDirichletNoiseTotalConcentration.
    //Half of the alpha weight are uniform.
    //The other half are shaped based on the log of the existing policy.
    let mut log_policy_sum: f64 = 0.0;
    for i in 0..policy_size as usize {
      if policy_probs[i] >= 0.0 {
        alpha_distr[i] = math::ln(cpp::cpp_min(0.01, policy_probs[i] as f64) + 1e-20);
        log_policy_sum += alpha_distr[i];
      }
    }
    let log_policy_mean: f64 = log_policy_sum / legal_count as f64;
    let mut alpha_prop_sum: f64 = 0.0;
    for i in 0..policy_size as usize {
      if policy_probs[i] >= 0.0 {
        alpha_distr[i] = cpp::cpp_max(0.0, alpha_distr[i] - log_policy_mean);
        alpha_prop_sum += alpha_distr[i];
      }
    }
    let uniform_prob: f64 = 1.0 / legal_count as f64;
    if alpha_prop_sum <= 0.0 {
      for i in 0..policy_size as usize {
        if policy_probs[i] >= 0.0 {
          alpha_distr[i] = uniform_prob;
        }
      }
    } else {
      for i in 0..policy_size as usize {
        if policy_probs[i] >= 0.0 {
          alpha_distr[i] = 0.5 * (alpha_distr[i] / alpha_prop_sum + uniform_prob);
        }
      }
    }
    Ok(())
  }

  /// upstream: Search::addDirichletNoise (searchhelpers.cpp)
  /// PORT NOTE: throws (computeDirichletAlphaDistribution, Rand::nextGamma) -> Result.
  pub fn add_dirichlet_noise(search_params: &SearchParams, rand: &mut Rand, policy_size: i32, policy_probs: &mut [f32]) -> Result<(), KataError> {
    let mut r: [f64; NNPos::MAX_NN_POLICY_SIZE as usize] = [0.0; NNPos::MAX_NN_POLICY_SIZE as usize];
    Search::compute_dirichlet_alpha_distribution(policy_size, policy_probs, &mut r)?;

    //r now contains the proportions with which we would like to split the alpha
    //The total of the alphas is searchParams.rootDirichletNoiseTotalConcentration
    //Generate gamma draw on each move
    let mut r_sum: f64 = 0.0;
    for i in 0..policy_size as usize {
      if policy_probs[i] >= 0.0 {
        r[i] = rand.next_gamma(r[i] * search_params.root_dirichlet_noise_total_concentration)?;
        r_sum += r[i];
      } else {
        r[i] = 0.0;
      }
    }

    //Normalized gamma draws -> dirichlet noise
    for i in 0..policy_size as usize {
      r[i] /= r_sum;
    }

    //At this point, r[i] contains a dirichlet distribution draw, so add it into the nnOutput.
    for i in 0..policy_size as usize {
      if policy_probs[i] >= 0.0 {
        let weight: f64 = search_params.root_dirichlet_noise_weight;
        policy_probs[i] = (r[i] * weight + policy_probs[i] as f64 * (1.0 - weight)) as f32;
      }
    }
    Ok(())
  }

  /// upstream: Search::maybeAddPolicyNoiseAndTemp (searchhelpers.cpp)
  /// PORT NOTE: returns the new Rc (None = NULL) wrapped in Result (addDirichletNoise throws).
  /// `oldNNOutput` is a reference: every caller passes a non-null output, so upstream's
  /// `if(oldNNOutput == NULL) return NULL;` cannot fire.
  pub(crate) fn maybe_add_policy_noise_and_temp(
    &self,
    thread: &mut SearchThread,
    is_root: bool,
    old_nn_output: &NNOutput,
  ) -> Result<Option<Rc<NNOutput>>, KataError> {
    if !is_root {
      return Ok(None);
    }
    if !self.search_params.root_noise_enabled
      && self.search_params.root_policy_temperature == 1.0
      && self.search_params.root_policy_temperature_early == 1.0
      && self.root_hint_loc == Board::NULL_LOC
      && !self.avoid_move_until_rescale_root
    {
      return Ok(None);
    }
    if old_nn_output.noised_policy_probs.is_some() {
      return Ok(None);
    }

    //Copy nnOutput as we're about to modify its policy to add noise or temperature
    let mut new_nn_output: NNOutput = old_nn_output.clone();

    // PORT NOTE: upstream stores the new array into newNNOutput->noisedPolicyProbs first and then
    // modifies it in place; here it is modified in a local Box and stored at the end (same values).
    let mut noised_policy_probs: Box<[f32; NNPos::MAX_NN_POLICY_SIZE as usize]> = Box::new(new_nn_output.policy_probs);

    if self.search_params.root_policy_temperature != 1.0 || self.search_params.root_policy_temperature_early != 1.0 {
      let root_policy_temperature: f64 = self.interpolate_early(
        self.search_params.chosen_move_temperature_halflife,
        self.search_params.root_policy_temperature_early,
        self.search_params.root_policy_temperature,
      );

      let mut max_value: f64 = 0.0;
      for i in 0..self.policy_size as usize {
        let prob: f64 = noised_policy_probs[i] as f64;
        if prob > max_value {
          max_value = prob;
        }
      }
      testAssert!(max_value > 0.0);

      let log_max_value: f64 = math::ln(max_value);
      let inv_temp: f64 = 1.0 / root_policy_temperature;
      let mut sum: f64 = 0.0;

      for i in 0..self.policy_size as usize {
        if noised_policy_probs[i] > 0.0 {
          //Numerically stable way to raise to power and normalize
          let p: f32 = math::exp((math::ln(noised_policy_probs[i] as f64) - log_max_value) * inv_temp) as f32;
          noised_policy_probs[i] = p;
          sum += p as f64;
        }
      }
      testAssert!(sum > 0.0);
      for i in 0..self.policy_size as usize {
        if noised_policy_probs[i] >= 0.0 {
          noised_policy_probs[i] = (noised_policy_probs[i] as f64 / sum) as f32;
        }
      }
    }

    if self.search_params.root_noise_enabled {
      Search::add_dirichlet_noise(&self.search_params, &mut thread.rand, self.policy_size, &mut noised_policy_probs[..])?;
    }

    if self.avoid_move_until_rescale_root {
      let avoid_move_until_by_loc: &Vec<i32> =
        if self.root_pla == P_BLACK { &self.avoid_move_until_by_loc_black } else { &self.avoid_move_until_by_loc_white };
      if !avoid_move_until_by_loc.is_empty() {
        testAssert!(avoid_move_until_by_loc.len() >= Board::MAX_ARR_SIZE as usize);
        let mut policy_sum: f64 = 0.0;
        let mut loc: Loc = 0;
        while (loc as i32) < Board::MAX_ARR_SIZE {
          if (self.root_board.is_on_board(loc) || loc == Board::PASS_LOC) && avoid_move_until_by_loc[loc as usize] <= 0 {
            let pos: i32 = self.get_pos(loc);
            if noised_policy_probs[pos as usize] > 0.0 {
              policy_sum += noised_policy_probs[pos as usize] as f64;
            }
          }
          loc += 1;
        }
        if policy_sum > 0.0 {
          for i in 0..self.policy_size as usize {
            if noised_policy_probs[i] > 0.0 {
              noised_policy_probs[i] = (noised_policy_probs[i] as f64 / policy_sum) as f32;
            }
          }
        }
      }
    }

    //Move a small amount of policy to the hint move, around the same level that noising it would achieve
    if self.root_hint_loc != Board::NULL_LOC {
      const PROP_TO_MOVE: f32 = 0.02f32;
      let pos: i32 = self.get_pos(self.root_hint_loc);
      if noised_policy_probs[pos as usize] >= 0.0 {
        let mut amount_to_move: f64 = 0.0;
        for i in 0..self.policy_size as usize {
          if noised_policy_probs[i] >= 0.0 {
            amount_to_move += (noised_policy_probs[i] * PROP_TO_MOVE) as f64;
            noised_policy_probs[i] *= 1.0f32 - PROP_TO_MOVE;
          }
        }
        noised_policy_probs[pos as usize] += amount_to_move as f32;
      }
    }

    new_nn_output.noised_policy_probs = Some(noised_policy_probs);
    Ok(Some(Rc::new(new_nn_output)))
  }

  /// upstream: Search::getResultUtility (searchhelpers.cpp)
  pub(crate) fn get_result_utility(&self, win_loss_value: f64, no_result_value: f64) -> f64 {
    win_loss_value * self.search_params.win_loss_utility_factor + no_result_value * self.search_params.no_result_utility_for_white
  }

  /// upstream: Search::getResultUtilityFromNN (searchhelpers.cpp)
  pub(crate) fn get_result_utility_from_nn(&self, nn_output: &NNOutput) -> f64 {
    // PORT NOTE: whiteWinProb - whiteLossProb is a float subtraction, widened afterwards (§6 item 1).
    (nn_output.white_win_prob - nn_output.white_loss_prob) as f64 * self.search_params.win_loss_utility_factor
      + nn_output.white_no_result_prob as f64 * self.search_params.no_result_utility_for_white
  }

  /// upstream: Search::getScoreUtility (searchhelpers.cpp)
  /// PORT NOTE: `pub` (upstream private, PORTING.md §1 would give pub(crate)) so the crafted parity oracle
  /// tests/search_select_diff.rs can call it.
  pub fn get_score_utility(&self, score_mean_avg: f64, score_mean_sq_avg: f64) -> f64 {
    let score_mean: f64 = score_mean_avg;
    let score_mean_sq: f64 = score_mean_sq_avg;
    let score_stdev: f64 = ScoreValue::get_score_stdev(score_mean, score_mean_sq);
    let sqrt_board_area: f64 = self.root_board.sqrt_board_area();
    let static_score_value: f64 = ScoreValue::expected_white_score_value(score_mean, score_stdev, 0.0, 2.0, sqrt_board_area);
    let dynamic_score_value: f64 =
      ScoreValue::expected_white_score_value(score_mean, score_stdev, self.recent_score_center, self.search_params.dynamic_score_center_scale, sqrt_board_area);
    static_score_value * self.search_params.static_score_utility_factor + dynamic_score_value * self.search_params.dynamic_score_utility_factor
  }

  /// upstream: Search::getScoreUtilityDiff (searchhelpers.cpp)
  /// PORT NOTE: `pub` (upstream private, PORTING.md §1 would give pub(crate)) so the crafted parity oracle
  /// tests/search_select_diff.rs can call it.
  pub fn get_score_utility_diff(&self, score_mean_avg: f64, score_mean_sq_avg: f64, delta: f64) -> f64 {
    let score_mean: f64 = score_mean_avg;
    let score_mean_sq: f64 = score_mean_sq_avg;
    let score_stdev: f64 = ScoreValue::get_score_stdev(score_mean, score_mean_sq);
    let sqrt_board_area: f64 = self.root_board.sqrt_board_area();
    let static_score_value_diff: f64 = ScoreValue::expected_white_score_value(score_mean + delta, score_stdev, 0.0, 2.0, sqrt_board_area)
      - ScoreValue::expected_white_score_value(score_mean, score_stdev, 0.0, 2.0, sqrt_board_area);
    let dynamic_score_value_diff: f64 = ScoreValue::expected_white_score_value(
      score_mean + delta,
      score_stdev,
      self.recent_score_center,
      self.search_params.dynamic_score_center_scale,
      sqrt_board_area,
    ) - ScoreValue::expected_white_score_value(score_mean, score_stdev, self.recent_score_center, self.search_params.dynamic_score_center_scale, sqrt_board_area);
    static_score_value_diff * self.search_params.static_score_utility_factor + dynamic_score_value_diff * self.search_params.dynamic_score_utility_factor
  }

  /// upstream: Search::getApproxScoreUtilityDerivative (searchhelpers.cpp)
  //Ignores scoreMeanSq's effect on the utility, since that's complicated
  /// PORT NOTE: `pub` (upstream private, PORTING.md §1 would give pub(crate)) so the crafted parity oracle
  /// tests/search_select_diff.rs can call it.
  pub fn get_approx_score_utility_derivative(&self, score_mean: f64) -> f64 {
    let sqrt_board_area: f64 = self.root_board.sqrt_board_area();
    let static_score_value_derivative: f64 = ScoreValue::white_d_score_value_d_score_smooth_no_draw_adjust(score_mean, 0.0, 2.0, sqrt_board_area);
    let dynamic_score_value_derivative: f64 = ScoreValue::white_d_score_value_d_score_smooth_no_draw_adjust(
      score_mean,
      self.recent_score_center,
      self.search_params.dynamic_score_center_scale,
      sqrt_board_area,
    );
    static_score_value_derivative * self.search_params.static_score_utility_factor
      + dynamic_score_value_derivative * self.search_params.dynamic_score_utility_factor
  }

  /// upstream: Search::getUtilityFromNN (searchhelpers.cpp)
  pub(crate) fn get_utility_from_nn(&self, nn_output: &NNOutput) -> f64 {
    let result_utility: f64 = self.get_result_utility_from_nn(nn_output);
    result_utility + self.get_score_utility(nn_output.white_score_mean as f64, nn_output.white_score_mean_sq as f64)
  }

  /// upstream: Search::isAllowedRootMove (searchhelpers.cpp)
  pub(crate) fn is_allowed_root_move(&self, move_loc: Loc) -> bool {
    debug_assert!(move_loc == Board::PASS_LOC || self.root_board.is_on_board(move_loc));

    //A bad situation that can happen that unnecessarily prolongs training games is where one player
    //repeatedly passes and the other side repeatedly fills the opponent's space and/or suicides over and over.
    //To mitigate some of this and save computation, we make it so that at the root, if the last four moves by the opponent
    //were passes, we will never play a move in either player's pass-alive area. In theory this could prune
    //a good move in situations like https://senseis.xmp.net/?1EyeFlaw, but this should be extraordinarly rare,
    if self.search_params.root_prune_useless_moves && !self.root_history.move_history.is_empty() && move_loc != Board::PASS_LOC {
      let last_idx: usize = self.root_history.move_history.len() - 1;
      let opp: Player = get_opp(self.root_pla);
      let mh = &self.root_history.move_history;
      if last_idx >= 6
        && mh[last_idx - 0].loc == Board::PASS_LOC
        && mh[last_idx - 2].loc == Board::PASS_LOC
        && mh[last_idx - 4].loc == Board::PASS_LOC
        && mh[last_idx - 6].loc == Board::PASS_LOC
        && mh[last_idx - 0].pla == opp
        && mh[last_idx - 2].pla == opp
        && mh[last_idx - 4].pla == opp
        && mh[last_idx - 6].pla == opp
        && (self.root_safe_area[move_loc as usize] == opp || self.root_safe_area[move_loc as usize] == self.root_pla)
      {
        return false;
      }
    }

    if self.search_params.root_symmetry_pruning && move_loc != Board::PASS_LOC && self.root_sym_dup_loc[move_loc as usize] {
      return false;
    }

    true
  }

  /// upstream: Search::getPatternBonus (searchhelpers.cpp)
  pub(crate) fn get_pattern_bonus(&self, pattern_bonus_hash: Hash128, prev_move_pla: Player) -> f64 {
    let pattern_bonus_table = match &self.pattern_bonus_table {
      None => return 0.0,
      Some(t) => t,
    };
    if prev_move_pla != self.pla_that_search_is_for {
      return 0.0;
    }
    let _ = pattern_bonus_hash;
    // TODO(port): deferred (patternbonustable.cpp): `return patternBonusTable->get(patternBonusHash).utilityBonus;`
    // PatternBonusTable has no values (D7), so this is unreachable.
    match **pattern_bonus_table {}
  }

  /// upstream: Search::getEndingWhiteScoreBonus (searchhelpers.cpp)
  pub(crate) fn get_ending_white_score_bonus(&self, parent: &SearchNode, move_loc: Loc) -> f64 {
    if Some(parent.idx) != self.root_node || move_loc == Board::NULL_LOC {
      return 0.0;
    }

    let nn_output: Rc<NNOutput> = match parent.get_nn_output() {
      None => return 0.0,
      Some(n) => n,
    };
    let white_owner_map: &Vec<f32> = match &nn_output.white_owner_map {
      None => return 0.0,
      Some(m) => m,
    };

    let is_area_ish: bool = self.root_history.rules.scoring_rule == Rules::SCORING_AREA
      || (self.root_history.rules.scoring_rule == Rules::SCORING_TERRITORY && self.root_history.encore_phase >= 2);
    testAssert!(nn_output.nn_x_len == self.nn_x_len);
    testAssert!(nn_output.nn_y_len == self.nn_y_len);

    const EXTREME: f64 = 0.95;
    const TAIL: f64 = 0.05;

    //Extra points from the perspective of the root player
    let mut extra_root_points: f64 = 0.0;
    if is_area_ish {
      //Areaish scoring - in an effort to keep the game short and slightly discourage pointless territory filling at the end
      //discourage any move that, except in case of ko, is either:
      // * On a spot that the opponent almost surely owns, unless it captures stones.
      // * On a spot that the player almost surely owns and it is not adjacent to opponent stones and is not a connection of non-pass-alive groups.
      //These conditions should still make it so that "cleanup" and dame-filling moves are not discouraged.
      // * When playing button go, very slightly discourage passing - so that if there are an even number of dame, filling a dame is still favored over passing.
      if move_loc != Board::PASS_LOC && self.root_board.ko_loc == Board::NULL_LOC {
        let pos: i32 = NNPos::loc_to_pos(move_loc, self.root_board.x_size, self.nn_x_len, self.nn_y_len);
        let pla_ownership: f64 =
          if self.root_pla == P_WHITE { white_owner_map[pos as usize] as f64 } else { -white_owner_map[pos as usize] as f64 };
        if pla_ownership <= -EXTREME {
          if !self.root_board.would_be_capture(move_loc, self.root_pla) {
            extra_root_points -= self.search_params.root_ending_bonus_points * ((-EXTREME - pla_ownership) / TAIL);
          }
        } else if pla_ownership >= EXTREME {
          if !self.root_board.is_adjacent_to_pla(move_loc, get_opp(self.root_pla))
            && !self.root_board.is_non_pass_alive_self_connection(move_loc, self.root_pla, &self.root_safe_area[..])
          {
            extra_root_points -= self.search_params.root_ending_bonus_points * ((pla_ownership - EXTREME) / TAIL);
          }
        }
      }
      if move_loc == Board::PASS_LOC && self.root_history.has_button {
        extra_root_points -= self.search_params.root_ending_bonus_points * 0.5;
      }
    } else {
      //Territorish scoring - slightly encourage dame-filling by discouraging passing, so that the player will try to do everything
      //non-point-losing first, like filling dame.
      //Human japanese rules often "want" you to fill the dame so this is a cosmetic adjustment to encourage the neural
      //net to learn to do so in the main phase rather than waiting until the encore.
      //But cosmetically, it's also not great if we just encourage useless threat moves in the opponent's territory to prolong the game.
      //So also discourage those moves except in cases of ko. Also similar to area scoring just to be symmetrical, discourage moves on spots
      //that the player almost surely owns that are not adjacent to opponent stones and are not a connection of non-pass-alive groups.
      if move_loc == Board::PASS_LOC {
        extra_root_points -= self.search_params.root_ending_bonus_points * (2.0 / 3.0);
      } else if self.root_board.ko_loc == Board::NULL_LOC {
        let pos: i32 = NNPos::loc_to_pos(move_loc, self.root_board.x_size, self.nn_x_len, self.nn_y_len);
        let pla_ownership: f64 =
          if self.root_pla == P_WHITE { white_owner_map[pos as usize] as f64 } else { -white_owner_map[pos as usize] as f64 };
        if pla_ownership <= -EXTREME {
          extra_root_points -= self.search_params.root_ending_bonus_points * ((-EXTREME - pla_ownership) / TAIL);
        } else if pla_ownership >= EXTREME {
          if !self.root_board.is_adjacent_to_pla(move_loc, get_opp(self.root_pla))
            && !self.root_board.is_non_pass_alive_self_connection(move_loc, self.root_pla, &self.root_safe_area[..])
          {
            extra_root_points -= self.search_params.root_ending_bonus_points * ((pla_ownership - EXTREME) / TAIL);
          }
        }
      }
    }

    if self.root_pla == P_WHITE {
      extra_root_points
    } else {
      -extra_root_points
    }
  }

  /// upstream: Search::shouldSuppressPass (searchhelpers.cpp)
  //Hack to encourage well-behaved dame filling behavior under territory scoring
  /// PORT NOTE: `pub` (upstream private, PORTING.md §1 would give pub(crate)) so the crafted parity oracle
  /// tests/search_select_diff.rs can call it.
  pub fn should_suppress_pass(&self, n: Option<&SearchNode>) -> bool {
    let n: &SearchNode = match n {
      None => return false,
      Some(n) => n,
    };
    if !self.search_params.fill_dame_before_pass || Some(n.idx) != self.root_node {
      return false;
    }
    if self.root_history.rules.scoring_rule != Rules::SCORING_TERRITORY || self.root_history.encore_phase > 0 {
      return false;
    }

    let node: &SearchNode = n;
    let nn_output: Rc<NNOutput> = match node.get_nn_output() {
      None => return false,
      Some(o) => o,
    };
    let white_owner_map: &Vec<f32> = match &nn_output.white_owner_map {
      None => return false,
      Some(m) => m,
    };
    testAssert!(nn_output.nn_x_len == self.nn_x_len);
    testAssert!(nn_output.nn_y_len == self.nn_y_len);

    //Find the pass move
    let mut pass_node: Option<Rc<SearchNode>> = None;
    let mut pass_edge_visits: i64 = 0;

    let children: SearchNodeChildrenReference = node.get_children();
    let children_capacity: i32 = children.get_capacity();
    for i in 0..children_capacity {
      let child_pointer: &SearchChildPointer = children.index(i);
      let child: NodeIdx = match child_pointer.get_if_allocated() {
        None => break,
        Some(c) => c,
      };
      let move_loc: Loc = child_pointer.get_move_loc_relaxed();
      if move_loc == Board::PASS_LOC {
        pass_node = Some(self.node_arena.get(child));
        pass_edge_visits = child_pointer.get_edge_visits();
        break;
      }
    }
    let pass_node: Rc<SearchNode> = match pass_node {
      None => return false,
      Some(p) => p,
    };

    let pass_weight: f64;
    let pass_utility: f64;
    let pass_score_mean: f64;
    let pass_lead: f64;
    {
      let pass_visits: i64 = pass_node.stats.visits.get() /*acquire*/;
      let score_mean_avg: f64 = pass_node.stats.score_mean_avg.get() /*acquire*/;
      let lead_avg: f64 = pass_node.stats.lead_avg.get() /*acquire*/;
      let utility_avg: f64 = pass_node.stats.utility_avg.get() /*acquire*/;
      let child_weight: f64 = pass_node.stats.get_child_weight_child_visits(pass_edge_visits, pass_visits);

      if pass_visits <= 0 || child_weight <= 1e-10 {
        return false;
      }
      pass_weight = child_weight;
      pass_utility = utility_avg;
      pass_score_mean = score_mean_avg;
      pass_lead = lead_avg;
    }

    const EXTREME: f64 = 0.95;

    //Suppress pass if we find a move that is not a spot that the opponent almost certainly owns
    //or that is adjacent to a pla owned spot, and is not greatly worse than pass.
    for i in 0..children_capacity {
      let child_pointer: &SearchChildPointer = children.index(i);
      let child_idx: NodeIdx = match child_pointer.get_if_allocated() {
        None => break,
        Some(c) => c,
      };
      let move_loc: Loc = child_pointer.get_move_loc_relaxed();
      if move_loc == Board::PASS_LOC {
        continue;
      }
      let pos: i32 = NNPos::loc_to_pos(move_loc, self.root_board.x_size, self.nn_x_len, self.nn_y_len);
      let pla_ownership: f64 = if self.root_pla == P_WHITE { white_owner_map[pos as usize] as f64 } else { -white_owner_map[pos as usize] as f64 };
      let opp_owned: bool = pla_ownership < -EXTREME;
      let mut adj_to_pla_owned: bool = false;
      for j in 0..4 {
        let adj: Loc = (move_loc as i32 + self.root_board.adj_offsets[j] as i32) as Loc;
        if self.root_board.is_on_board(adj) {
          let adj_pos: i32 = NNPos::loc_to_pos(adj, self.root_board.x_size, self.nn_x_len, self.nn_y_len);
          let adj_pla_ownership: f64 =
            if self.root_pla == P_WHITE { white_owner_map[adj_pos as usize] as f64 } else { -white_owner_map[adj_pos as usize] as f64 };
          if adj_pla_ownership > EXTREME {
            adj_to_pla_owned = true;
            break;
          }
        }
      }
      if opp_owned && !adj_to_pla_owned {
        continue;
      }

      let edge_visits: i64 = child_pointer.get_edge_visits();

      let child: Rc<SearchNode> = self.node_arena.get(child_idx);
      let score_mean_avg: f64 = child.stats.score_mean_avg.get() /*acquire*/;
      let lead_avg: f64 = child.stats.lead_avg.get() /*acquire*/;
      let utility_avg: f64 = child.stats.utility_avg.get() /*acquire*/;
      let child_weight: f64 = child.stats.get_child_weight(edge_visits);

      //Too few visits - reject move
      if (edge_visits <= 500 && child_weight <= 2.0 * pass_weight.sqrt()) || child_weight <= 1e-10 {
        continue;
      }

      let utility: f64 = utility_avg;
      let score_mean: f64 = score_mean_avg;
      let lead: f64 = lead_avg;

      if self.root_pla == P_WHITE && utility > pass_utility - 0.1 && score_mean > pass_score_mean - 0.5 && lead > pass_lead - 0.5 {
        return true;
      }
      if self.root_pla == P_BLACK && utility < pass_utility + 0.1 && score_mean < pass_score_mean + 0.5 && lead < pass_lead + 0.5 {
        return true;
      }
    }
    false
  }

  /// upstream: Search::interpolateEarly (searchhelpers.cpp)
  pub(crate) fn interpolate_early(&self, halflife: f64, early_value: f64, value: f64) -> f64 {
    let raw_halflives: f64 = self.root_history.get_current_turn_number() as f64 / halflife;
    let halflives: f64 = raw_halflives * 19.0 / ((self.root_board.x_size * self.root_board.y_size) as f64).sqrt();
    value + (early_value - value) * math::powf(0.5, halflives)
  }

  /// upstream: Search::getSelfUtilityLCBAndRadiusZeroVisits (searchhelpers.cpp)
  /// PORT NOTE: `pub` (upstream private, PORTING.md §1 would give pub(crate)) so the crafted parity oracle
  /// tests/search_select_diff.rs can call it.
  pub fn get_self_utility_lcb_and_radius_zero_visits(&self, lcb_buf: &mut f64, radius_buf: &mut f64) {
    // Max radius of the entire utility range
    let utility_range_radius: f64 =
      self.search_params.win_loss_utility_factor + self.search_params.static_score_utility_factor + self.search_params.dynamic_score_utility_factor;
    *radius_buf = 2.0 * utility_range_radius * self.search_params.lcb_stdevs;
    *lcb_buf = -*radius_buf;
  }

  /// upstream: Search::getSelfUtilityLCBAndRadius (searchhelpers.cpp)
  /// PORT NOTE: `const SearchNode* child` is never NULL at the callers: `&SearchNode`.
  pub(crate) fn get_self_utility_lcb_and_radius(
    &self,
    parent: &SearchNode,
    child: &SearchNode,
    edge_visits: i64,
    move_loc: Loc,
    lcb_buf: &mut f64,
    radius_buf: &mut f64,
  ) {
    let child_visits: i64 = child.stats.visits.get() /*acquire*/;
    let score_mean_avg: f64 = child.stats.score_mean_avg.get() /*acquire*/;
    let score_mean_sq_avg: f64 = child.stats.score_mean_sq_avg.get() /*acquire*/;
    let utility_avg: f64 = child.stats.utility_avg.get() /*acquire*/;
    let mut utility_sq_avg: f64 = child.stats.utility_sq_avg.get() /*acquire*/;
    let mut weight_sum: f64 = child.stats.get_child_weight_child_visits(edge_visits, child_visits);
    let mut weight_sq_sum: f64 = child.stats.get_child_weight_sq_child_visits(edge_visits, child_visits);

    // Max radius of the entire utility range
    let utility_range_radius: f64 =
      self.search_params.win_loss_utility_factor + self.search_params.static_score_utility_factor + self.search_params.dynamic_score_utility_factor;
    *radius_buf = 2.0 * utility_range_radius * self.search_params.lcb_stdevs;
    *lcb_buf = -*radius_buf;
    if child_visits <= 0 || weight_sum <= 0.0 || weight_sq_sum <= 0.0 {
      return;
    }

    // Effective sample size for weighted data
    let mut ess: f64 = weight_sum * weight_sum / weight_sq_sum;

    // To behave well at low playouts, we'd like a variance approximation that makes sense even with very small sample sizes.
    // We'd like to avoid using a T distribution approximation because we actually know a bound on the scale of the utilities
    // involved, namely utilityRangeRadius. So instead add a prior with a small weight that the variance is the largest it can be.
    // This should give a relatively smooth scaling that works for small discrete samples but diminishes for larger playouts.
    let prior_weight: f64 = weight_sum / (ess * ess * ess);
    utility_sq_avg = cpp::cpp_max(utility_sq_avg, utility_avg * utility_avg + 1e-8);
    utility_sq_avg = (utility_sq_avg * weight_sum + (utility_sq_avg + utility_range_radius * utility_range_radius) * prior_weight) / (weight_sum + prior_weight);
    weight_sum += prior_weight;
    weight_sq_sum += prior_weight * prior_weight;

    // Recompute effective sample size now that we have the prior
    ess = weight_sum * weight_sum / weight_sq_sum;

    let ending_score_bonus: f64 = self.get_ending_white_score_bonus(parent, move_loc);
    let utility_diff: f64 = self.get_score_utility_diff(score_mean_avg, score_mean_sq_avg, ending_score_bonus);
    let utility_with_bonus: f64 = utility_avg + utility_diff;
    let self_utility: f64 = if parent.next_pla == P_WHITE { utility_with_bonus } else { -utility_with_bonus };

    let utility_variance: f64 = utility_sq_avg - utility_avg * utility_avg;
    let estimate_stdev: f64 = (utility_variance / ess).sqrt();
    let radius: f64 = estimate_stdev * self.search_params.lcb_stdevs;

    *lcb_buf = self_utility - radius;
    *radius_buf = radius;
  }
}
