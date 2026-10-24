// Ported from KataGo cpp/search/searchmirror.cpp @ ba938676d7f42d70950b3a535af2466fb642008c

use crate::game::board::get_opp;
use crate::game::board::Board;
use crate::game::board::Color;
use crate::game::board::Loc;
use crate::game::board::Location;
use crate::game::board::Player;
use crate::game::board::C_EMPTY;
use crate::game::board::P_BLACK;
use crate::game::board::P_WHITE;
use crate::game::boardhistory::BoardHistory;
use crate::game::rules::Rules;
use crate::neuralnet::nninputs::NNOutput;
use crate::port::cpp;
use crate::port::math;
use crate::search::search::Search;
use crate::search::search::SearchThread;
use crate::search::searchnode::SearchNode;

impl Search {
  /// upstream: Search::updateMirroring (searchmirror.cpp)
  // Updates mirroringPla, mirrorAdvantage, mirrorCenterSymmetryError
  pub(crate) fn update_mirroring(&mut self) {
    self.mirroring_pla = C_EMPTY;
    self.mirror_advantage = 0.0;
    self.mirror_center_symmetry_error = 1e10;

    if self.search_params.anti_mirror {
      let board: &Board = &self.root_board;
      let hist: &BoardHistory = &self.root_history;
      let mut mirror_count: i32 = 0;
      let mut total_count: i32 = 0;
      let mut mirror_ewms: f64 = 0.0;
      let mut total_ewms: f64 = 0.0;
      let mut last_was_mirror: bool = false;
      for i in 1..hist.move_history.len() {
        if hist.move_history[i].pla != self.root_pla {
          last_was_mirror = false;
          if hist.move_history[i].loc == Location::get_mirror_loc(hist.move_history[i - 1].loc, board.x_size, board.y_size) {
            mirror_count += 1;
            mirror_ewms += 1.0;
            last_was_mirror = true;
          }
          total_count += 1;
          total_ewms += 1.0;
          mirror_ewms *= 0.75;
          total_ewms *= 0.75;
        }
      }
      //If at most of the moves in the game are mirror moves, and many of the recent moves were mirrors, and the last move
      //was a mirror, then the opponent is mirroring.
      if mirror_count as f64 >= 7.0 + 0.5 * total_count as f64 && mirror_ewms >= 0.45 * total_ewms && last_was_mirror {
        self.mirroring_pla = get_opp(self.root_pla);

        let mut black_extra_points: f64 = 0.0;
        let num_handicap_stones: i32 = hist.compute_num_handicap_stones();
        if hist.rules.scoring_rule == Rules::SCORING_AREA {
          if num_handicap_stones > 0 {
            black_extra_points += (num_handicap_stones - 1) as f64;
          }
          let black_gets_last_move: bool =
            (board.x_size % 2 == 1 && board.y_size % 2 == 1) == (num_handicap_stones == 0 || num_handicap_stones % 2 == 1);
          if black_gets_last_move {
            black_extra_points += 1.0;
          }
        }
        if num_handicap_stones > 0 && hist.rules.white_handicap_bonus_rule == Rules::WHB_N {
          black_extra_points -= num_handicap_stones as f64;
        }
        if num_handicap_stones > 0 && hist.rules.white_handicap_bonus_rule == Rules::WHB_N_MINUS_ONE {
          black_extra_points -= (num_handicap_stones - 1) as f64;
        }
        self.mirror_advantage = if self.mirroring_pla == P_BLACK {
          black_extra_points - hist.rules.komi as f64
        } else {
          hist.rules.komi as f64 - black_extra_points
        };
      }

      if board.x_size >= 7 && board.y_size >= 7 {
        self.mirror_center_symmetry_error = 0.0;
        let half_x: i32 = board.x_size / 2;
        let half_y: i32 = board.y_size / 2;
        let mut unmatched_mirror_pla_stones: i32 = 0;
        for dy in -3..=3 {
          for dx in -3..=3 {
            let loc: Loc = Location::get_loc(half_x + dx, half_y + dy, board.x_size);
            let mirror_loc: Loc = Location::get_mirror_loc(loc, board.x_size, board.y_size);
            if loc == mirror_loc {
              continue;
            }
            let c0: Color = board.colors[loc as usize];
            let c1: Color = board.colors[mirror_loc as usize];
            if c0 == get_opp(self.mirroring_pla) && c1 != self.mirroring_pla {
              self.mirror_center_symmetry_error += 1.0;
            }
            if c0 == self.mirroring_pla && c1 == C_EMPTY {
              unmatched_mirror_pla_stones += 1;
            }
          }
        }
        if self.mirror_center_symmetry_error > 0.0 {
          self.mirror_center_symmetry_error += 0.2 * unmatched_mirror_pla_stones as f64;
        }
        if self.mirror_center_symmetry_error >= 1.0 {
          self.mirror_center_symmetry_error = 0.5 * self.mirror_center_symmetry_error * (1.0 + self.mirror_center_symmetry_error);
        }
      }
    }
  }

  /// upstream: Search::isMirroringSinceSearchStart (searchmirror.cpp)
  pub(crate) fn is_mirroring_since_search_start(&self, thread_history: &BoardHistory, skip_recent: i32) -> bool {
    let x_size: i32 = thread_history.initial_board.x_size;
    let y_size: i32 = thread_history.initial_board.y_size;
    // PORT NOTE: size_t loop variable; `i+skipRecent` converts skipRecent (never negative here) to size_t.
    let mut i: usize = self.root_history.move_history.len() + 1;
    while i + (skip_recent as usize) < thread_history.move_history.len() {
      if thread_history.move_history[i].loc != Location::get_mirror_loc(thread_history.move_history[i - 1].loc, x_size, y_size) {
        return false;
      }
      i += 2;
    }
    true
  }

  /// upstream: Search::maybeApplyAntiMirrorPolicy (searchmirror.cpp)
  pub(crate) fn maybe_apply_anti_mirror_policy(
    &self,
    nn_policy_prob: &mut f32,
    move_loc: Loc,
    policy_probs: &[f32],
    move_pla: Player,
    thread: &SearchThread,
  ) {
    let x_size: i32 = thread.board.x_size;
    let y_size: i32 = thread.board.y_size;

    let mut weight: f64 = 0.0;

    //Put significant prior probability on the opponent continuing to mirror, at least for the next few turns.
    if move_pla == get_opp(self.root_pla) && !thread.history.move_history.is_empty() {
      let prev_loc: Loc = thread.history.move_history[thread.history.move_history.len() - 1].loc;
      if prev_loc == Board::PASS_LOC {
        return;
      }
      let mut mirror_loc: Loc = Location::get_mirror_loc(prev_loc, x_size, y_size);
      if policy_probs[self.get_pos(mirror_loc) as usize] < 0.0 {
        mirror_loc = Board::PASS_LOC;
      }
      if move_loc == mirror_loc {
        weight = 1.0;
        let center_loc: Loc = Location::get_center_loc_x_size_y_size(x_size, y_size);
        let is_difficult: bool =
          center_loc != Board::NULL_LOC && thread.board.colors[center_loc as usize] == self.mirroring_pla && self.mirror_advantage >= -0.5;
        if is_difficult {
          weight *= 3.0;
        }
      }
    }
    //Put a small prior on playing the center or attaching to center, bonusing moves that are relatively more likely.
    else if move_pla == self.root_pla && move_loc != Board::PASS_LOC {
      if Location::is_central(move_loc, x_size, y_size) {
        weight = 0.3;
      } else {
        if Location::is_near_central(move_loc, x_size, y_size) {
          weight = 0.05;
        }

        let center_loc: Loc = Location::get_center_loc_x_size_y_size(x_size, y_size);
        if center_loc != Board::NULL_LOC {
          if self.root_board.colors[center_loc as usize] == get_opp(move_pla) {
            if thread.board.is_adjacent_to_chain(move_loc, center_loc) {
              weight = 0.05;
            } else {
              let distance_sq: i32 = Location::euclidean_distance_squared(move_loc, center_loc, x_size);
              if distance_sq <= 2 {
                weight = 0.05;
              } else if distance_sq <= 4 {
                weight = 0.03;
              }
            }
          }
        }
      }
    }

    if weight > 0.0 {
      // PORT NOTE: size_t difference converted to double, as upstream (never negative during search).
      weight = weight
        / (1.0
          + ((thread.history.move_history.len() as u64).wrapping_sub(self.root_history.move_history.len() as u64) as f64).sqrt());
      *nn_policy_prob = *nn_policy_prob + (1.0f32 - *nn_policy_prob) * weight as f32;
    }
  }

  /// upstream: Search::maybeApplyAntiMirrorForcedExplore (searchmirror.cpp)
  //Force the search to dump playouts down a mirror move, so as to encourage moves that cause mirror moves
  //to have bad values, and also tolerate us playing certain countering moves even if their values are a bit worse.
  pub(crate) fn maybe_apply_anti_mirror_forced_explore(
    &self,
    child_utility: &mut f64,
    parent_utility: f64,
    move_loc: Loc,
    policy_probs: &[f32],
    this_child_weight: f64,
    total_child_weight: f64,
    move_pla: Player,
    thread: &SearchThread,
    parent: &SearchNode,
  ) {
    debug_assert!(self.mirroring_pla == get_opp(self.root_pla));

    let x_size: i32 = thread.board.x_size;
    let y_size: i32 = thread.board.y_size;
    let center_loc: Loc = Location::get_center_loc_x_size_y_size(x_size, y_size);
    //The difficult case is when the opponent has occupied tengen, and ALSO the komi favors them.
    //In such a case, we're going to have a hard time.
    //Technically there are other configurations (like if the opponent makes a diamond around tengen)
    //but we're not going to worry about breaking that.
    let is_difficult: bool =
      center_loc != Board::NULL_LOC && thread.board.colors[center_loc as usize] == self.mirroring_pla && self.mirror_advantage >= -0.5;
    // bool isSemiDifficult = !isDifficult && mirrorAdvantage >= 6.5;
    let is_root: bool = Some(parent.idx) == self.root_node;

    //Force mirroring pla to dump playouts down mirror moves
    if move_pla == self.mirroring_pla && !thread.history.move_history.is_empty() {
      let prev_loc: Loc = thread.history.move_history[thread.history.move_history.len() - 1].loc;
      if prev_loc == Board::PASS_LOC {
        return;
      }
      let mut mirror_loc: Loc = Location::get_mirror_loc(prev_loc, x_size, y_size);
      if policy_probs[self.get_pos(mirror_loc) as usize] < 0.0 {
        mirror_loc = Board::PASS_LOC;
      }
      if move_loc == mirror_loc {
        let mut proportion_to_dump: f64; // PORT NOTE: upstream initializes it to 0.0, overwritten on every branch
        let mut proportion_to_bias: f64; // PORT NOTE: upstream initializes it to 0.0, overwritten on every branch
        if is_difficult {
          proportion_to_dump = 0.20;
          if mirror_loc != Board::PASS_LOC {
            proportion_to_dump = cpp::cpp_max(
              proportion_to_dump,
              1.0 / (0.75 + 0.5 * (Location::euclidean_distance_squared(center_loc, mirror_loc, x_size) as f64).sqrt())
                / cpp::cpp_max(1.0, self.mirror_center_symmetry_error),
            );
          }
          proportion_to_bias = 0.75;
        } else if self.mirror_advantage >= 5.0 {
          proportion_to_dump = 0.15;
          proportion_to_bias = 0.50;
        } else if self.mirror_advantage >= -5.0 {
          proportion_to_dump = 0.10 + self.mirror_advantage;
          proportion_to_bias = 0.30 + self.mirror_advantage * 4.0;
        } else {
          proportion_to_dump = 0.05;
          proportion_to_bias = 0.10;
        }

        if mirror_loc == Board::PASS_LOC {
          proportion_to_dump *=
            if move_loc == center_loc { 0.35 } else { 0.35 / cpp::cpp_max(1.0, self.mirror_center_symmetry_error.sqrt()) };
        }
        if self.mirror_center_symmetry_error >= 1.0 {
          proportion_to_dump /= self.mirror_center_symmetry_error;
          proportion_to_bias /= self.mirror_center_symmetry_error;
        }

        if this_child_weight < proportion_to_dump * total_child_weight {
          *child_utility += if parent.next_pla == P_WHITE { 100.0 } else { -100.0 };
        }
        if this_child_weight < proportion_to_bias * total_child_weight {
          *child_utility +=
            (if parent.next_pla == P_WHITE { 0.18 } else { -0.18 }) * cpp::cpp_max(0.3, 1.0 - 0.7 * parent_utility * parent_utility);
        }
        if this_child_weight < 0.5 * proportion_to_bias * total_child_weight {
          *child_utility +=
            (if parent.next_pla == P_WHITE { 0.36 } else { -0.36 }) * cpp::cpp_max(0.3, 1.0 - 0.7 * parent_utility * parent_utility);
        }
      }
    }
    //Encourage us to find refuting moves, even if they look a little bad, in the difficult case
    //Force us to dump playouts down tengen if possible, to encourage us to make tengen into a good move.
    else if move_pla == self.root_pla && move_loc != Board::PASS_LOC {
      let mut proportion_to_dump: f64 = 0.0;
      if is_difficult {
        if thread.board.is_adjacent_to_chain(move_loc, center_loc) {
          *child_utility += (if parent.next_pla == P_WHITE { 0.75 } else { -0.75 })
            / (1.0 + thread.board.get_num_liberties(center_loc) as f64)
            / cpp::cpp_max(1.0, self.mirror_center_symmetry_error)
            * cpp::cpp_max(0.3, 1.0 - 0.7 * parent_utility * parent_utility);
          proportion_to_dump = 0.10 / thread.board.get_num_liberties(center_loc) as f64;
        }
        let distance_sq: i32 = Location::euclidean_distance_squared(move_loc, center_loc, x_size);
        if distance_sq <= 2 {
          proportion_to_dump = cpp::cpp_max(proportion_to_dump, 0.010);
        } else if distance_sq <= 4 {
          proportion_to_dump = cpp::cpp_max(proportion_to_dump, 0.005);
        }

        //proportionToDump *= (1.0 / (1.0 + sqrt(thread->history.moveHistory.size() - rootHistory.moveHistory.size())));
      }
      if move_loc == center_loc {
        if is_root {
          proportion_to_dump = 0.06;
        } else {
          proportion_to_dump = 0.12;
        }
      }

      let utility_loss: f64 = if parent.next_pla == P_WHITE { parent_utility - *child_utility } else { *child_utility - parent_utility };
      if utility_loss > 0.0 && utility_loss * proportion_to_dump > 0.03 {
        proportion_to_dump += 0.5 * (0.03 / utility_loss - proportion_to_dump);
      }

      if !thread.history.move_history.is_empty() {
        let prev_loc: Loc = thread.history.move_history[thread.history.move_history.len() - 1].loc;
        if prev_loc != Board::NULL_LOC && prev_loc != Board::PASS_LOC {
          let center_distance_squared: i32 = Location::euclidean_distance_squared(center_loc, prev_loc, x_size);
          if center_distance_squared <= 16 {
            proportion_to_dump *= 0.900;
          }
          if center_distance_squared <= 5 {
            proportion_to_dump *= 0.825;
          }
          if center_distance_squared <= 2 {
            proportion_to_dump *= 0.750;
          }
        }
      }

      if this_child_weight < proportion_to_dump * total_child_weight {
        *child_utility += if parent.next_pla == P_WHITE { 100.0 } else { -100.0 };
      }
    }
  }

  /// upstream: Search::hackNNOutputForMirror (searchmirror.cpp)
  /// PORT NOTE: `std::shared_ptr<NNOutput>& result` -> `&mut NNOutput`: the caller hacks its fresh,
  /// still-unshared copy (PORTING.md §4, NNOutput is immutable once shared).
  pub(crate) fn hack_nn_output_for_mirror(&self, result: &mut NNOutput) {
    // Root player gets a bonus/penalty based on the strength of the center.
    let center_pos: i32 = self.get_pos(Location::get_center_loc(&self.root_board));
    let total_wl_prob: f64 = (result.white_win_prob + result.white_loss_prob) as f64;
    let own_scale: f64 = if self.mirror_center_symmetry_error <= 0.0 { 0.7 } else { 0.3 };
    let mut wl: f64 = (result.white_win_prob - result.white_loss_prob) as f64 / (total_wl_prob + 1e-10);
    wl = cpp::cpp_min(cpp::cpp_max(wl, -1.0 + 1e-15), 1.0 - 1e-15);
    let white_owner_map: &Vec<f32> = result.white_owner_map.as_ref().expect("antiMirrorDifficult requests the owner map");
    wl = math::tanh(math::atanh(wl) + own_scale * white_owner_map[center_pos as usize] as f64);
    let mut white_new_win_prob: f64 = 0.5 + 0.5 * wl;
    white_new_win_prob = total_wl_prob * white_new_win_prob;

    result.white_win_prob = white_new_win_prob as f32;
    result.white_loss_prob = (total_wl_prob - white_new_win_prob) as f32;
  }
}
