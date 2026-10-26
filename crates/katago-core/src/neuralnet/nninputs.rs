// Ported from KataGo cpp/neuralnet/nninputs.cpp, cpp/neuralnet/nninputs.h @ fd0723fdbc0e9d82cf269c9630af8c27c57c07c4
//
// Not ported (port-manifest.toml notes): NNInputs::fillRowV3, fillRowV4, fillRowV5, fillRowV6 (input
// versions of pre-v8 nets; v8+ nets consume fillRowV7).
//
// Overload names (PORTING.md §1 rules 3-4):
//   SymmetryHelpers::getSymLoc(Loc, const Board&, int)              -> get_sym_loc (fewest params)
//   SymmetryHelpers::getSymLoc(int x, int y, const Board&, int)      -> get_sym_loc_x_y
//   SymmetryHelpers::getSymLoc(int x, int y, int xSize, int ySize, int) -> get_sym_loc_x_y_x_size_y_size
//   SymmetryHelpers::getSymLoc(Loc, int xSize, int ySize, int)       -> get_sym_loc_x_size_y_size
//   SymmetryHelpers::compose(int, int)                              -> compose
//   SymmetryHelpers::compose(int, int, int nextNextSymmetry)         -> compose_next_next_symmetry
//   NNOutput::NNOutput()                                            -> new
//   NNOutput::NNOutput(const std::vector<std::shared_ptr<NNOutput>>&) -> new_others
//   NNOutput copy ctor / operator=                                  -> impl Clone


use std::rc::Rc;
use std::sync::OnceLock;

use crate::core::global::KataError;
use crate::core::hash::Hash;
use crate::core::hash::Hash128;
use crate::game::board::*;
use crate::game::boardhistory::BoardHistory;
use crate::game::boardhistory::BoardHistoryModes;
use crate::game::rules::Rules;
use crate::port::cpp;
use crate::port::math;
use crate::strprintf;
use crate::testAssert;

/// upstream: NNPos (nninputs.h)
pub struct NNPos;

impl NNPos {
  /// upstream: NNPos::MAX_BOARD_LEN, MAX_BOARD_AREA, MAX_NN_POLICY_SIZE, EXTRA_SCORE_DISTR_RADIUS, KOMI_CLIP_RADIUS (nninputs.h)
  pub const MAX_BOARD_LEN: i32 = Board::MAX_LEN;
  pub const MAX_BOARD_AREA: i32 = NNPos::MAX_BOARD_LEN * NNPos::MAX_BOARD_LEN;
  //Policy output adds +1 for the pass move
  pub const MAX_NN_POLICY_SIZE: i32 = NNPos::MAX_BOARD_AREA + 1;
  //Extra score distribution radius, used for writing score in data rows and for the neural net score belief output
  pub const EXTRA_SCORE_DISTR_RADIUS: i32 = 60;
  //Used various places we clip komi beyond board area.
  pub const KOMI_CLIP_RADIUS: f32 = 20.0f32;

  /// upstream: NNPos::xyToPos (nninputs.h)
  #[inline]
  pub fn xy_to_pos(x: i32, y: i32, nn_x_len: i32) -> i32 {
    y * nn_x_len + x
  }
  /// upstream: NNPos::locToPos (nninputs.h)
  #[inline]
  pub fn loc_to_pos(loc: Loc, board_x_size: i32, nn_x_len: i32, nn_y_len: i32) -> i32 {
    if loc == Board::PASS_LOC {
      return nn_x_len * nn_y_len;
    } else if loc == Board::NULL_LOC {
      return nn_x_len * (nn_y_len + 1);
    }
    Location::get_y(loc, board_x_size) * nn_x_len + Location::get_x(loc, board_x_size)
  }
  /// upstream: NNPos::posToLoc (nninputs.h)
  #[inline]
  pub fn pos_to_loc(pos: i32, board_x_size: i32, board_y_size: i32, nn_x_len: i32, nn_y_len: i32) -> Loc {
    if pos == nn_x_len * nn_y_len {
      return Board::PASS_LOC;
    }
    let x: i32 = pos % nn_x_len;
    let y: i32 = pos / nn_x_len;
    if x < 0 || x >= board_x_size || y < 0 || y >= board_y_size {
      return Board::NULL_LOC;
    }
    Location::get_loc(x, y, board_x_size)
  }
  /// upstream: NNPos::getPassPos (nninputs.h)
  #[inline]
  pub fn get_pass_pos(nn_x_len: i32, nn_y_len: i32) -> i32 {
    nn_x_len * nn_y_len
  }
  /// upstream: NNPos::isPassPos (nninputs.h)
  #[inline]
  pub fn is_pass_pos(pos: i32, nn_x_len: i32, nn_y_len: i32) -> bool {
    pos == nn_x_len * nn_y_len
  }
  /// upstream: NNPos::getPolicySize (nninputs.h)
  #[inline]
  pub fn get_policy_size(nn_x_len: i32, nn_y_len: i32) -> i32 {
    nn_x_len * nn_y_len + 1
  }
}

/// upstream: NNInputs (nninputs.h)
pub struct NNInputs;

impl NNInputs {
  /// upstream: NNInputs::SYMMETRY_NOTSPECIFIED, NNInputs::SYMMETRY_ALL (nninputs.h)
  pub const SYMMETRY_NOTSPECIFIED: i32 = -1;
  pub const SYMMETRY_ALL: i32 = -2;

  /// upstream: NNInputs::NUM_FEATURES_SPATIAL_V3 .. NUM_FEATURES_GLOBAL_V7 (nninputs.h)
  pub const NUM_FEATURES_SPATIAL_V3: i32 = 22;
  pub const NUM_FEATURES_GLOBAL_V3: i32 = 14;

  pub const NUM_FEATURES_SPATIAL_V4: i32 = 22;
  pub const NUM_FEATURES_GLOBAL_V4: i32 = 14;

  pub const NUM_FEATURES_SPATIAL_V5: i32 = 13;
  pub const NUM_FEATURES_GLOBAL_V5: i32 = 12;

  pub const NUM_FEATURES_SPATIAL_V6: i32 = 22;
  pub const NUM_FEATURES_GLOBAL_V6: i32 = 16;

  pub const NUM_FEATURES_SPATIAL_V7: i32 = 22;
  pub const NUM_FEATURES_GLOBAL_V7: i32 = 19;
}

/// upstream: MiscNNInputParams (nninputs.h)
#[derive(Clone, Debug)]
pub struct MiscNNInputParams {
  pub draw_equivalent_wins_for_white: f64,
  pub conservative_pass_and_is_root: bool,
  pub enable_passing_hacks: bool,
  pub playout_doubling_advantage: f64,
  pub nn_policy_temperature: f32,
  pub avoid_myt_dagger_hack: bool,
  // If no symmetry is specified, it will use default or random based on config, unless node is already cached.
  pub symmetry: i32,
  pub policy_optimism: f64,
  pub max_history: i32,
  // -1 = no override: pass-alive featurization follows hist.modes.alwaysComputePassAliveUnderSuicideRules.
  // 0/1 = override that flag for featurization (and its contribution to the nn cache hash). Used when
  // evaluating with a secondary net (e.g. a human SL profile net) whose declared featurization mode
  // differs from the mode the search itself is using.
  pub pass_alive_suicide_rules_override: i32,
  // Same, but for hist.modes.excludeTerritoryAdjacentToAtari.
  pub exclude_territory_adj_atari_override: i32,
}

impl Default for MiscNNInputParams {
  /// upstream: MiscNNInputParams default member initializers (nninputs.h)
  fn default() -> Self {
    MiscNNInputParams {
      draw_equivalent_wins_for_white: 0.5,
      conservative_pass_and_is_root: false,
      enable_passing_hacks: false,
      playout_doubling_advantage: 0.0,
      nn_policy_temperature: 1.0f32,
      avoid_myt_dagger_hack: false,
      symmetry: NNInputs::SYMMETRY_NOTSPECIFIED,
      policy_optimism: 0.0,
      max_history: 1000,
      pass_alive_suicide_rules_override: -1,
      exclude_territory_adj_atari_override: -1,
    }
  }
}

impl MiscNNInputParams {
  /// upstream: MiscNNInputParams::MiscNNInputParams() (implicit; nninputs.h initializers)
  pub fn new() -> MiscNNInputParams {
    MiscNNInputParams::default()
  }

  /// upstream: MiscNNInputParams::ZOBRIST_* (nninputs.cpp)
  pub const ZOBRIST_CONSERVATIVE_PASS: Hash128 = Hash128::new(0x0c2b96f4b8ae2da9, 0x5a14dee208fec0ed);
  pub const ZOBRIST_FRIENDLY_PASS: Hash128 = Hash128::new(0xe750505a66f7c5c2, 0x7a83139bf632d6c4);
  pub const ZOBRIST_PASSING_HACKS: Hash128 = Hash128::new(0x9c89f4fd3ce5a92c, 0x268c9aff79c64d00);
  pub const ZOBRIST_PLAYOUT_DOUBLINGS: Hash128 = Hash128::new(0xa5e6114d380bfc1d, 0x4160557f1222f4ad);
  pub const ZOBRIST_NN_POLICY_TEMP: Hash128 = Hash128::new(0xebcbdfeec6f4334b, 0xb85e43ee243b5ad2);
  pub const ZOBRIST_AVOID_MYTDAGGER_HACK: Hash128 = Hash128::new(0x612d22ec402ce054, 0x0db915c49de527ae);
  pub const ZOBRIST_POLICY_OPTIMISM: Hash128 = Hash128::new(0x88415c85c2801955, 0x39bdf76b2aaa5eb1);
  pub const ZOBRIST_ZERO_HISTORY: Hash128 = Hash128::new(0x78f02afdd1aa4910, 0xda78d550486fe978);
}

//-----------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------

/// upstream: ScoreValue (nninputs.h)
//Utility functions for computing the "scoreValue", the unscaled utility of various numbers of points, prior to multiplication by
//staticScoreUtilityFactor or dynamicScoreUtilityFactor (see searchparams.h)
pub struct ScoreValue;

/// upstream: twoOverPi, piOverTwo (nninputs.cpp) static
const twoOverPi: f64 = 0.63661977236758134308;
const piOverTwo: f64 = 1.57079632679489661923;

/// upstream: svTableAssumedBSize, svTableMeanRadius, svTableMeanLen, svTableStdevLen (nninputs.cpp) static
const svTableAssumedBSize: i32 = NNPos::MAX_BOARD_LEN;
const svTableMeanRadius: i32 = svTableAssumedBSize * svTableAssumedBSize + NNPos::EXTRA_SCORE_DISTR_RADIUS;
const svTableMeanLen: i32 = svTableMeanRadius * 2;
const svTableStdevLen: i32 = svTableAssumedBSize * svTableAssumedBSize + NNPos::EXTRA_SCORE_DISTR_RADIUS;

/// upstream: scoreValueTablesInitialized, expectedSVTable (nninputs.cpp) static
/// PORT NOTE: runtime-initialized file statics -> one struct in a OnceLock (PORTING.md §8);
/// `scoreValueTablesInitialized` is `SCORE_VALUE_STATICS.get().is_some()`.
pub struct ScoreValueStatics {
  pub expectedSVTable: Vec<f64>,
}

static SCORE_VALUE_STATICS: OnceLock<ScoreValueStatics> = OnceLock::new();

/// upstream: inverse_atan (nninputs.cpp) static
fn inverse_atan(x: f64) -> f64 {
  if x >= piOverTwo - 1e-6 {
    return 1e6;
  }
  if x <= -piOverTwo + 1e-6 {
    return -1e6;
  }
  math::tan(x)
}

impl ScoreValue {
  /// upstream: ScoreValue::whiteWinsOfWinner (nninputs.cpp)
  //The number of wins a game result should count as
  pub fn white_wins_of_winner(winner: Player, draw_equivalent_wins_for_white: f64) -> f64 {
    if winner == P_WHITE {
      return 1.0;
    } else if winner == P_BLACK {
      return 0.0;
    }

    testAssert!(winner == C_EMPTY);
    draw_equivalent_wins_for_white
  }

  /// upstream: ScoreValue::whiteScoreDrawAdjust (nninputs.cpp)
  //The score difference that a game result should count as on average
  pub fn white_score_draw_adjust(final_white_minus_black_score: f64, draw_equivalent_wins_for_white: f64, hist: &BoardHistory) -> f64 {
    final_white_minus_black_score + hist.white_komi_adjustment_for_draws(draw_equivalent_wins_for_white) as f64
  }

  /// upstream: ScoreValue::whiteScoreValueOfScoreSmooth (nninputs.cpp)
  //The unscaled utility of achieving a certain score difference
  pub fn white_score_value_of_score_smooth(
    final_white_minus_black_score: f64,
    center: f64,
    scale: f64,
    draw_equivalent_wins_for_white: f64,
    sqrt_board_area: f64,
    hist: &BoardHistory,
  ) -> f64 {
    let adjusted_score: f64 =
      final_white_minus_black_score + hist.white_komi_adjustment_for_draws(draw_equivalent_wins_for_white) as f64 - center;
    math::atan(adjusted_score / (scale * sqrt_board_area)) * twoOverPi
  }

  /// upstream: ScoreValue::whiteScoreValueOfScoreSmoothNoDrawAdjust (nninputs.cpp)
  pub fn white_score_value_of_score_smooth_no_draw_adjust(final_white_minus_black_score: f64, center: f64, scale: f64, sqrt_board_area: f64) -> f64 {
    let adjusted_score: f64 = final_white_minus_black_score - center;
    math::atan(adjusted_score / (scale * sqrt_board_area)) * twoOverPi
  }

  /// upstream: ScoreValue::whiteDScoreValueDScoreSmoothNoDrawAdjust (nninputs.cpp)
  //The derviative of whiteScoreValueOfScoreSmoothNoDrawAdjust with respect to finalWhiteMinusBlackScore.
  pub fn white_d_score_value_d_score_smooth_no_draw_adjust(final_white_minus_black_score: f64, center: f64, scale: f64, sqrt_board_area: f64) -> f64 {
    let adjusted_score: f64 = final_white_minus_black_score - center;
    let scale_factor: f64;
    scale_factor = scale * sqrt_board_area;
    scale_factor / (scale_factor * scale_factor + adjusted_score * adjusted_score) * twoOverPi
  }

  /// upstream: ScoreValue::approxWhiteScoreOfScoreValueSmooth (nninputs.cpp)
  //Approximately invert whiteScoreValueOfScoreSmooth
  pub fn approx_white_score_of_score_value_smooth(score_value: f64, center: f64, scale: f64, sqrt_board_area: f64) -> f64 {
    testAssert!(score_value >= -1.0 && score_value <= 1.0);
    let score_unscaled: f64 = inverse_atan(score_value * piOverTwo);
    score_unscaled * (scale * sqrt_board_area) + center
  }

  /// upstream: ScoreValue::whiteScoreMeanSqOfScoreGridded (nninputs.cpp)
  //Compute what the scoreMeanSq should be for a final game result
  //It is NOT simply the same as finalWhiteMinusBlackScore^2 because for integer komi we model it as a distribution where with the appropriate probability
  //you gain or lose 0.5 point to achieve the desired drawEquivalentWinsForWhite, so it actually has some variance.
  pub fn white_score_mean_sq_of_score_gridded(final_white_minus_black_score: f64, draw_equivalent_wins_for_white: f64) -> f64 {
    // PORT NOTE: C++ UB for out-of-range/NaN; x86-64 gives INT_MIN (port::cpp::static_cast_i32_f64).
    testAssert!(cpp::static_cast_i32_f64(final_white_minus_black_score * 2.0) as f64 == final_white_minus_black_score * 2.0);
    let final_score_is_integer: bool = cpp::static_cast_i32_f64(final_white_minus_black_score) as f64 == final_white_minus_black_score;
    if !final_score_is_integer {
      return final_white_minus_black_score * final_white_minus_black_score;
    }

    let lower: f64 = final_white_minus_black_score - 0.5;
    let upper: f64 = final_white_minus_black_score + 0.5;
    let lower_sq: f64 = lower * lower;
    let upper_sq: f64 = upper * upper;

    lower_sq + (upper_sq - lower_sq) * draw_equivalent_wins_for_white
  }

  /// upstream: ScoreValue::freeTables (nninputs.cpp)
  /// PORT NOTE: no-op; the tables live in a OnceLock for the process lifetime (PORTING.md §8).
  pub fn free_tables() {}

  /// upstream: ScoreValue::initTables (nninputs.cpp)
  /// PORT NOTE: idempotent (OnceLock, PORTING.md §8) instead of testAssert(!scoreValueTablesInitialized).
  //MUST BE CALLED AT PROGRAM START!
  pub fn init_tables() {
    SCORE_VALUE_STATICS.get_or_init(compute_score_value_tables);
  }

  /// upstream: ScoreValue::expectedWhiteScoreValue (nninputs.cpp)
  //The expected unscaled utility of the final score difference, given the mean and stdev of the distribution of that difference,
  //assuming roughly a normal distribution.
  pub fn expected_white_score_value(white_score_mean: f64, white_score_stdev: f64, center: f64, scale: f64, sqrt_board_area: f64) -> f64 {
    let statics: Option<&ScoreValueStatics> = SCORE_VALUE_STATICS.get();
    testAssert!(statics.is_some());
    let expectedSVTable: &[f64] = &statics.expect("checked by testAssert above").expectedSVTable;

    let scale_factor: f64 = svTableAssumedBSize as f64 / (scale * sqrt_board_area);

    let mean_scaled: f64 = (white_score_mean - center) * scale_factor;
    let stdev_scaled: f64 = white_score_stdev * scale_factor;

    let mean_rounded: f64 = mean_scaled.round();
    let stdev_floored: f64 = stdev_scaled.floor();
    // PORT NOTE: C++ UB for out-of-range/NaN; x86-64 gives INT_MIN (port::cpp::static_cast_i32_f64).
    // PORT NOTE: `+ svTableMeanRadius` and `meanIdx0 + 1` wrap like x86-64 (C++ signed overflow UB when
    // meanRounded is within svTableMeanRadius of INT_MAX: the wrapped index is negative and clamps to 0;
    // checked against the C++ in oracle/nninputs_scorevalue.txt).
    let mut mean_idx0: i32 = cpp::static_cast_i32_f64(mean_rounded).wrapping_add(svTableMeanRadius);
    let mut stdev_idx0: i32 = cpp::static_cast_i32_f64(stdev_floored);
    let mut mean_idx1: i32 = mean_idx0.wrapping_add(1);
    // PORT NOTE: wrapping like x86-64 (C++ signed overflow UB only at stdevFloored == INT_MAX).
    let mut stdev_idx1: i32 = stdev_idx0.wrapping_add(1);

    if mean_idx0 < 0 {
      mean_idx0 = 0;
      mean_idx1 = 0;
    }
    if mean_idx1 >= svTableMeanLen {
      mean_idx0 = svTableMeanLen - 1;
      mean_idx1 = svTableMeanLen - 1;
    }
    testAssert!(stdev_idx0 >= 0);
    if stdev_idx1 >= svTableStdevLen {
      stdev_idx0 = svTableStdevLen - 1;
      stdev_idx1 = svTableStdevLen - 1;
    }

    let lambda_mean: f64 = mean_scaled - mean_rounded + 0.5;
    let lambda_stdev: f64 = stdev_scaled - stdev_floored;

    let a00: f64 = expectedSVTable[(mean_idx0 * svTableStdevLen + stdev_idx0) as usize];
    let a01: f64 = expectedSVTable[(mean_idx0 * svTableStdevLen + stdev_idx1) as usize];
    let a10: f64 = expectedSVTable[(mean_idx1 * svTableStdevLen + stdev_idx0) as usize];
    let a11: f64 = expectedSVTable[(mean_idx1 * svTableStdevLen + stdev_idx1) as usize];

    let b0: f64 = a00 + lambda_stdev * (a01 - a00);
    let b1: f64 = a10 + lambda_stdev * (a11 - a10);
    b0 + lambda_mean * (b1 - b0)
  }

  /// upstream: ScoreValue::getScoreStdev (nninputs.cpp)
  //Get the standard deviation of score given the E(score) and E(score^2)
  pub fn get_score_stdev(score_mean: f64, score_mean_sq: f64) -> f64 {
    let variance: f64 = score_mean_sq - score_mean * score_mean;
    if variance <= 0.0 {
      return 0.0;
    }
    variance.sqrt()
  }
}

/// upstream: ScoreValue::initTables (nninputs.cpp) body
fn compute_score_value_tables() -> ScoreValueStatics {
  let mut expectedSVTable: Vec<f64> = vec![0.0; (svTableMeanLen * svTableStdevLen) as usize];

  //Precompute normal PDF
  const stepsPerUnit: i32 = 10; //Must be divisible by 2. This is both the number of segments that we divide points into, and that we divide stdevs into
  const boundStdevs: i32 = 5;
  let min_stdev_steps: i32 = -boundStdevs * stepsPerUnit;
  let max_stdev_steps: i32 = boundStdevs * stepsPerUnit;
  let mut normal_pdf: Vec<f64> = vec![0.0; ((max_stdev_steps - min_stdev_steps) + 1) as usize];
  for i in min_stdev_steps..=max_stdev_steps {
    let x_in_stdevs: f64 = i as f64 / stepsPerUnit as f64;
    let w: f64 = math::exp(-0.5 * x_in_stdevs * x_in_stdevs);
    normal_pdf[(i - min_stdev_steps) as usize] = w;
  }
  //Precompute scorevalue at increments of 1/stepsPerUnit points
  let min_sv_steps: i32 = -(svTableMeanRadius * stepsPerUnit + stepsPerUnit / 2 + boundStdevs * svTableStdevLen * stepsPerUnit);
  let max_sv_steps: i32 = -min_sv_steps;
  let mut sv_precomp: Vec<f64> = vec![0.0; ((max_sv_steps - min_sv_steps) + 1) as usize];
  for i in min_sv_steps..=max_sv_steps {
    let mean: f64 = i as f64 / stepsPerUnit as f64;
    let sv: f64 = ScoreValue::white_score_value_of_score_smooth_no_draw_adjust(mean, 0.0, 1.0, svTableAssumedBSize as f64);
    sv_precomp[(i - min_sv_steps) as usize] = sv;
  }

  //Perform numeric integration
  for mean_idx in 0..svTableMeanLen {
    let mean_steps: i32 = (mean_idx - svTableMeanRadius) * stepsPerUnit - stepsPerUnit / 2;
    for stdev_idx in 0..svTableStdevLen {
      let mut w_sum: f64 = 0.0;
      let mut wsv_sum: f64 = 0.0;
      for i in min_stdev_steps..=max_stdev_steps {
        let x_steps: i32 = mean_steps + stdev_idx * i;
        let w: f64 = normal_pdf[(i - min_stdev_steps) as usize];
        debug_assert!(x_steps >= min_sv_steps && x_steps <= max_sv_steps);
        let sv: f64 = sv_precomp[(x_steps - min_sv_steps) as usize];
        w_sum += w;
        wsv_sum += w * sv;
      }
      expectedSVTable[(mean_idx * svTableStdevLen + stdev_idx) as usize] = wsv_sum / w_sum;
    }
  }

  ScoreValueStatics { expectedSVTable }
}

//-----------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------

impl NNInputs {
  /// upstream: NNInputs::fillScoring (nninputs.cpp)
  //If groupTax is specified, for each color region of area, reduce weight on empty spaces equally to reduce the total sum by 2.
  //(but should handle seki correctly)
  pub fn fill_scoring(board: &Board, area: &[Color], group_tax: bool, scoring: &mut [f32]) {
    if !group_tax {
      scoring[..Board::MAX_ARR_SIZE as usize].fill(0.0f32);
      for y in 0..board.y_size {
        for x in 0..board.x_size {
          let loc: Loc = Location::get_loc(x, y, board.x_size);
          let area_color: Color = area[loc as usize];
          if area_color == P_BLACK {
            scoring[loc as usize] = -1.0f32;
          } else if area_color == P_WHITE {
            scoring[loc as usize] = 1.0f32;
          } else {
            debug_assert!(area_color == C_EMPTY);
            scoring[loc as usize] = 0.0;
          }
        }
      }
    } else {
      let mut visited: [bool; Board::MAX_ARR_SIZE as usize] = [false; Board::MAX_ARR_SIZE as usize];
      let mut queue: [Loc; Board::MAX_ARR_SIZE as usize] = [0; Board::MAX_ARR_SIZE as usize];

      visited.fill(false);
      scoring[..Board::MAX_ARR_SIZE as usize].fill(0.0f32);
      for y in 0..board.y_size {
        for x in 0..board.x_size {
          let loc: Loc = Location::get_loc(x, y, board.x_size);
          if visited[loc as usize] {
            continue;
          }
          let area_color: Color = area[loc as usize];
          if area_color == P_BLACK || area_color == P_WHITE {
            let full_value: f32 = if area_color == P_WHITE { 1.0f32 } else { -1.0f32 };
            let mut queue_head: i32 = 0;
            let mut queue_tail: i32 = 1;
            queue[0] = loc;
            visited[loc as usize] = true;

            //First, count how many empty or opp locations there are
            let mut territory_count: i32 = 0;
            while queue_head < queue_tail {
              let next: Loc = queue[queue_head as usize];
              queue_head += 1;
              if board.colors[next as usize] != area_color {
                territory_count += 1;
              }
              //Push adjacent locations on to queue
              for i in 0..4 {
                let adj: Loc = (next as i32 + board.adj_offsets[i] as i32) as Loc;
                if area[adj as usize] == area_color && !visited[adj as usize] {
                  queue[queue_tail as usize] = adj;
                  queue_tail += 1;
                  visited[adj as usize] = true;
                }
              }
            }

            //Then, actually fill values
            let territory_value: f32 =
              if territory_count <= 2 { 0.0f32 } else { full_value * (territory_count as f32 - 2.0f32) / territory_count as f32 };
            for j in 0..queue_tail {
              let next: Loc = queue[j as usize];
              queue_head += 1;
              if board.colors[next as usize] != area_color {
                scoring[next as usize] = territory_value;
              } else {
                scoring[next as usize] = full_value;
              }
            }
            // upstream increments queueHead in the fill loop above without ever reading it again.
            let _ = queue_head;
          } else {
            debug_assert!(area_color == C_EMPTY);
            scoring[loc as usize] = 0.0;
          }
        }
      }
    }
  }
}

//-----------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------

/// upstream: NNOutput (nninputs.h)
/// PORT NOTE: copy ctor and operator= -> derived Clone (deep copies of whiteOwnerMap and
/// noisedPolicyProbs, as upstream); the destructor only frees memory.
#[derive(Clone, Debug)]
pub struct NNOutput {
  pub nn_hash: Hash128, //NNInputs - getHash

  //Initially from the perspective of the player to move at the time of the eval, fixed up later in nnEval.cpp
  //to be the value from white's perspective.
  //These three are categorial probabilities for each outcome.
  pub white_win_prob: f32,
  pub white_loss_prob: f32,
  pub white_no_result_prob: f32,

  //The first two moments of the believed distribution of the expected score at the end of the game, from white's perspective.
  pub white_score_mean: f32,
  pub white_score_mean_sq: f32,
  //Points to make game fair
  pub white_lead: f32,
  //Expected arrival time of remaining game variance, in turns, weighted by variance, only when modelVersion >= 9
  pub var_time_left: f32,
  //A metric indicating the "typical" error in the winloss value or the score that the net expects, relative to the
  //short-term future MCTS value.
  pub shortterm_winloss_error: f32,
  pub shortterm_score_error: f32,

  //Indexed by pos rather than loc
  //Values in here will be set to negative for illegal moves, including superko
  pub policy_probs: [f32; NNPos::MAX_NN_POLICY_SIZE as usize],
  //The optimism value used for this neural net evaluation.
  pub policy_optimism_used: f32,

  pub nn_x_len: i32,
  pub nn_y_len: i32,
  //If not NULL, then this contains a nnXLen*nnYLen-sized map of expected ownership on the board.
  pub white_owner_map: Option<Vec<f32>>,

  //If not NULL, then contains policy with dirichlet noise or any other noise adjustments for this node
  pub noised_policy_probs: Option<Box<[f32; NNPos::MAX_NN_POLICY_SIZE as usize]>>,
}

impl NNOutput {
  /// upstream: NNOutput::NNOutput() (nninputs.cpp)
  /// PORT NOTE: C++ leaves the scalar fields and policyProbs uninitialized; zero-initialized here.
  pub fn new() -> NNOutput {
    NNOutput {
      nn_hash: Hash128::default(),
      white_win_prob: 0.0,
      white_loss_prob: 0.0,
      white_no_result_prob: 0.0,
      white_score_mean: 0.0,
      white_score_mean_sq: 0.0,
      white_lead: 0.0,
      var_time_left: 0.0,
      shortterm_winloss_error: 0.0,
      shortterm_score_error: 0.0,
      policy_probs: [0.0; NNPos::MAX_NN_POLICY_SIZE as usize],
      policy_optimism_used: 0.0,
      nn_x_len: 0,
      nn_y_len: 0,
      white_owner_map: None,
      noised_policy_probs: None,
    }
  }

  /// upstream: NNOutput::NNOutput(const std::vector<std::shared_ptr<NNOutput>>&) (nninputs.cpp)
  //Averages the others. Others must be nonempty and share the same nnHash (i.e. be for the same board, except if somehow the hash collides)
  //Does NOT carry over noisedPolicyProbs.
  pub fn new_others(others: &[Rc<NNOutput>]) -> NNOutput {
    let mut s: NNOutput = NNOutput::new();
    testAssert!(others.len() < 1000000);
    let len: i32 = others.len() as i32;
    let float_len: f32 = len as f32;
    testAssert!(len > 0);
    for i in 1..len {
      testAssert!(others[i as usize].nn_hash == others[0].nn_hash);
    }
    s.nn_hash = others[0].nn_hash;

    s.white_win_prob = 0.0f32;
    s.white_loss_prob = 0.0f32;
    s.white_no_result_prob = 0.0f32;
    s.white_score_mean = 0.0f32;
    s.white_score_mean_sq = 0.0f32;
    s.white_lead = 0.0f32;
    s.var_time_left = 0.0f32;
    s.shortterm_winloss_error = 0.0f32;
    s.shortterm_score_error = 0.0f32;
    for i in 0..len {
      let other: &NNOutput = &others[i as usize];
      s.white_win_prob += other.white_win_prob;
      s.white_loss_prob += other.white_loss_prob;
      s.white_no_result_prob += other.white_no_result_prob;
      s.white_score_mean += other.white_score_mean;
      s.white_score_mean_sq += other.white_score_mean_sq;
      s.white_lead += other.white_lead;
      s.var_time_left += other.var_time_left;
      s.shortterm_winloss_error += other.shortterm_winloss_error;
      s.shortterm_score_error += other.shortterm_score_error;
    }
    s.white_win_prob /= float_len;
    s.white_loss_prob /= float_len;
    s.white_no_result_prob /= float_len;
    s.white_score_mean /= float_len;
    s.white_score_mean_sq /= float_len;
    s.white_lead /= float_len;
    s.var_time_left /= float_len;
    s.shortterm_winloss_error /= float_len;
    s.shortterm_score_error /= float_len;

    s.nn_x_len = others[0].nn_x_len;
    s.nn_y_len = others[0].nn_y_len;

    {
      let mut white_owner_map_count: f32 = 0.0f32;
      s.white_owner_map = None;
      let n: usize = (s.nn_x_len * s.nn_y_len) as usize;
      for i in 0..len {
        let other: &NNOutput = &others[i as usize];
        if let Some(other_map) = &other.white_owner_map {
          if s.white_owner_map.is_none() {
            s.white_owner_map = Some(vec![0.0f32; n]);
          }
          white_owner_map_count += 1.0f32;
          let map: &mut Vec<f32> = s.white_owner_map.as_mut().expect("allocated above");
          for pos in 0..n {
            map[pos] += other_map[pos];
          }
        }
      }
      if let Some(map) = &mut s.white_owner_map {
        testAssert!(white_owner_map_count > 0.0);
        for pos in 0..n {
          map[pos] /= white_owner_map_count;
        }
      }
    }

    s.noised_policy_probs = None;

    //For technical correctness in case of impossibly rare hash collisions:
    //Just give up if they don't all match in move legality
    {
      let mut mismatch: bool = false;
      s.policy_probs.fill(0.0f32);
      for i in 0..len {
        let other: &NNOutput = &others[i as usize];
        for pos in 0..NNPos::MAX_NN_POLICY_SIZE as usize {
          if i > 0 && (s.policy_probs[pos] < 0.0) != (other.policy_probs[pos] < 0.0) {
            mismatch = true;
          }
          s.policy_probs[pos] += other.policy_probs[pos];
        }
      }
      //In case of mismatch, just take the first one
      //This should basically never happen, only on true hash collisions
      if mismatch {
        let other: &NNOutput = &others[0];
        s.policy_probs.copy_from_slice(&other.policy_probs);
      } else {
        for pos in 0..NNPos::MAX_NN_POLICY_SIZE as usize {
          s.policy_probs[pos] /= float_len;
        }
      }
    }
    {
      let mut all_optimisms_match: bool = true;
      for i in 1..len {
        if others[i as usize].policy_optimism_used != others[0].policy_optimism_used {
          all_optimisms_match = false;
          break;
        }
      }
      if all_optimisms_match {
        s.policy_optimism_used = others[0].policy_optimism_used;
      } else {
        s.policy_optimism_used = 0.0;
        for i in 0..len {
          s.policy_optimism_used += others[i as usize].policy_optimism_used / len as f32;
        }
      }
    }
    s
  }

  /// upstream: NNOutput::getPolicyProbsMaybeNoised (nninputs.h)
  #[inline]
  pub fn get_policy_probs_maybe_noised(&self) -> &[f32; NNPos::MAX_NN_POLICY_SIZE as usize] {
    match &self.noised_policy_probs {
      Some(p) => p,
      None => &self.policy_probs,
    }
  }

  /// upstream: NNOutput::debugPrint (nninputs.cpp)
  pub fn debug_print(&self, out: &mut dyn std::fmt::Write, board: &Board) {
    writeln!(out, "Win {}", strprintf!("%.2fc", self.white_win_prob * 100.0f32)).unwrap();
    writeln!(out, "Loss {}", strprintf!("%.2fc", self.white_loss_prob * 100.0f32)).unwrap();
    writeln!(out, "NoResult {}", strprintf!("%.2fc", self.white_no_result_prob * 100.0f32)).unwrap();
    writeln!(out, "ScoreMean {}", strprintf!("%.2f", self.white_score_mean)).unwrap();
    writeln!(out, "ScoreMeanSq {}", strprintf!("%.1f", self.white_score_mean_sq)).unwrap();
    writeln!(out, "Lead {}", strprintf!("%.2f", self.white_lead)).unwrap();
    writeln!(out, "VarTimeLeft {}", strprintf!("%.1f", self.var_time_left)).unwrap();
    writeln!(out, "STWinlossError {}", strprintf!("%.2fc", self.shortterm_winloss_error * 100.0f32)).unwrap();
    writeln!(out, "STScoreError {}", strprintf!("%.2f", self.shortterm_score_error)).unwrap();
    writeln!(out, "OptimismUsed {}", strprintf!("%.2f", self.policy_optimism_used)).unwrap();

    writeln!(out, "Policy").unwrap();
    // PORT NOTE: round(float) is roundf; (int) of it is port::cpp::static_cast_i32_f32 (x86-64:
    // INT_MIN for NaN and out-of-int-range values, which debugPrint does print).
    writeln!(
      out,
      "Pass{}",
      strprintf!("%4d ", cpp::static_cast_i32_f32((self.policy_probs[NNPos::get_pass_pos(self.nn_x_len, self.nn_y_len) as usize] * 1000.0f32).round()))
    )
    .unwrap();
    for y in 0..board.y_size {
      for x in 0..board.x_size {
        let pos: i32 = NNPos::xy_to_pos(x, y, self.nn_x_len);
        let prob: f32 = self.policy_probs[pos as usize];
        if prob < 0.0 {
          write!(out, "   - ").unwrap();
        } else {
          write!(out, "{}", strprintf!("%4d ", cpp::static_cast_i32_f32((prob * 1000.0f32).round()))).unwrap();
        }
      }
      writeln!(out).unwrap();
    }

    if let Some(white_owner_map) = &self.white_owner_map {
      for y in 0..board.y_size {
        for x in 0..board.x_size {
          let pos: i32 = NNPos::xy_to_pos(x, y, self.nn_x_len);
          let white_own: f32 = white_owner_map[pos as usize];
          write!(out, "{}", strprintf!("%5d ", cpp::static_cast_i32_f32((white_own * 1000.0f32).round()))).unwrap();
        }
        writeln!(out).unwrap();
      }
      writeln!(out).unwrap();
    }
  }

  /// upstream: NNOutput::getPos (nninputs.h)
  #[inline]
  pub fn get_pos(&self, loc: Loc, board: &Board) -> i32 {
    NNPos::loc_to_pos(loc, board.x_size, self.nn_x_len, self.nn_y_len)
  }
}

//-------------------------------------------------------------------------------------------------------------

/// upstream: SymmetryHelpers (nninputs.h)
pub struct SymmetryHelpers;

/// upstream: copyWithSymmetry (nninputs.cpp) static
fn copy_with_symmetry(src: &[f32], dst: &mut [f32], n_size: i32, h_size: i32, w_size: i32, c_size: i32, use_nhwc: bool, symmetry: i32, reverse: bool) {
  let transpose: bool = (symmetry & 0x4) != 0 && h_size == w_size;
  let mut flip_x: bool = (symmetry & 0x2) != 0;
  let mut flip_y: bool = (symmetry & 0x1) != 0;
  if transpose && !reverse {
    std::mem::swap(&mut flip_x, &mut flip_y);
  }
  if use_nhwc {
    let n_stride: i32 = h_size * w_size * c_size;
    let h_stride: i32 = w_size * c_size;
    let w_stride: i32 = c_size;
    let mut h_base_new: i32 = 0;
    let mut h_stride_new: i32 = h_stride;
    let mut w_base_new: i32 = 0;
    let mut w_stride_new: i32 = w_stride;

    if flip_y {
      h_base_new = (h_size - 1) * h_stride_new;
      h_stride_new = -h_stride_new;
    }
    if flip_x {
      w_base_new = (w_size - 1) * w_stride_new;
      w_stride_new = -w_stride_new;
    }

    if transpose {
      std::mem::swap(&mut h_stride_new, &mut w_stride_new);
    }

    for n in 0..n_size {
      for h in 0..h_size {
        let nh_old: i32 = n * n_stride + h * h_stride;
        let nh_new: i32 = n * n_stride + h_base_new + h * h_stride_new;
        for w in 0..w_size {
          let nhw_old: i32 = nh_old + w * w_stride;
          let nhw_new: i32 = nh_new + w_base_new + w * w_stride_new;
          for c in 0..c_size {
            dst[(nhw_new + c) as usize] = src[(nhw_old + c) as usize];
          }
        }
      }
    }
  } else {
    let nc_size: i32 = n_size * c_size;
    let nc_stride: i32 = h_size * w_size;
    let h_stride: i32 = w_size;
    let w_stride: i32 = 1;
    let mut h_base_new: i32 = 0;
    let mut h_stride_new: i32 = h_stride;
    let mut w_base_new: i32 = 0;
    let mut w_stride_new: i32 = w_stride;

    if flip_y {
      h_base_new = (h_size - 1) * h_stride_new;
      h_stride_new = -h_stride_new;
    }
    if flip_x {
      w_base_new = (w_size - 1) * w_stride_new;
      w_stride_new = -w_stride_new;
    }

    if transpose {
      std::mem::swap(&mut h_stride_new, &mut w_stride_new);
    }

    for nc in 0..nc_size {
      for h in 0..h_size {
        let nch_old: i32 = nc * nc_stride + h * h_stride;
        let nch_new: i32 = nc * nc_stride + h_base_new + h * h_stride_new;
        for w in 0..w_size {
          let nchw_old: i32 = nch_old + w * w_stride;
          let nchw_new: i32 = nch_new + w_base_new + w * w_stride_new;
          dst[nchw_new as usize] = src[nchw_old as usize];
        }
      }
    }
  }
}

/// upstream: getSymmetryDifference (nninputs.cpp) static
fn get_symmetry_difference(board: &Board, other: &Board, symmetry: i32, max_difference_to_report: f64) -> f64 {
  let mut this_difference: f64 = 0.0;
  for y in 0..board.y_size {
    for x in 0..board.x_size {
      let loc: Loc = Location::get_loc(x, y, board.x_size);
      let sym_loc: Loc = SymmetryHelpers::get_sym_loc_x_y(x, y, board, symmetry);
      // Difference!
      if board.colors[loc as usize] != other.colors[sym_loc as usize] {
        // One of them was empty, the other was a stone
        if board.colors[loc as usize] == C_EMPTY || other.colors[sym_loc as usize] == C_EMPTY {
          this_difference += 1.0;
        }
        // Differing stones - triple the penalty
        else {
          this_difference += 3.0;
        }

        if this_difference > max_difference_to_report {
          return max_difference_to_report;
        }
      }
    }
  }
  this_difference
}

impl SymmetryHelpers {
  /// upstream: SymmetryHelpers::NUM_SYMMETRIES, NUM_SYMMETRIES_WITHOUT_TRANSPOSE (nninputs.h)
  //A symmetry is 3 bits flipY(bit 0), flipX(bit 1), transpose(bit 2). They are applied in that order.
  //The first four symmetries only reflect, and do not transpose X and Y.
  pub const NUM_SYMMETRIES: i32 = 8;
  pub const NUM_SYMMETRIES_WITHOUT_TRANSPOSE: i32 = 4;

  /// upstream: SymmetryHelpers::copyInputsWithSymmetry (nninputs.cpp)
  //These two IGNORE transpose if hSize and wSize do not match. So non-square transposes are disallowed.
  //copyOutputsWithSymmetry performs the inverse of symmetry.
  pub fn copy_inputs_with_symmetry(src: &[f32], dst: &mut [f32], n_size: i32, h_size: i32, w_size: i32, c_size: i32, use_nhwc: bool, symmetry: i32) {
    copy_with_symmetry(src, dst, n_size, h_size, w_size, c_size, use_nhwc, symmetry, false);
  }

  /// upstream: SymmetryHelpers::copyOutputsWithSymmetry (nninputs.cpp)
  pub fn copy_outputs_with_symmetry(src: &[f32], dst: &mut [f32], n_size: i32, h_size: i32, w_size: i32, symmetry: i32) {
    copy_with_symmetry(src, dst, n_size, h_size, w_size, 1, false, symmetry, true);
  }

  /// upstream: SymmetryHelpers::invert (nninputs.cpp)
  //Get the inverse of the specified symmetry
  pub fn invert(symmetry: i32) -> i32 {
    if symmetry == 5 {
      return 6;
    }
    if symmetry == 6 {
      return 5;
    }
    symmetry
  }

  /// upstream: SymmetryHelpers::compose (nninputs.cpp)
  //Get the symmetry equivalent to first applying firstSymmetry and then applying nextSymmetry.
  pub fn compose(first_symmetry: i32, mut next_symmetry: i32) -> i32 {
    if SymmetryHelpers::is_transpose(first_symmetry) {
      next_symmetry = (next_symmetry & 0x4) | ((next_symmetry & 0x2) >> 1) | ((next_symmetry & 0x1) << 1);
    }
    first_symmetry ^ next_symmetry
  }

  /// upstream: SymmetryHelpers::compose (nninputs.cpp)
  pub fn compose_next_next_symmetry(first_symmetry: i32, next_symmetry: i32, next_next_symmetry: i32) -> i32 {
    SymmetryHelpers::compose(SymmetryHelpers::compose(first_symmetry, next_symmetry), next_next_symmetry)
  }

  /// upstream: SymmetryHelpers::getSymLoc (nninputs.cpp)
  //Applies a symmetry to a location
  pub fn get_sym_loc_x_y_x_size_y_size(mut x: i32, mut y: i32, x_size: i32, y_size: i32, symmetry: i32) -> Loc {
    let transpose: bool = (symmetry & 0x4) != 0;
    let flip_x: bool = (symmetry & 0x2) != 0;
    let flip_y: bool = (symmetry & 0x1) != 0;
    if flip_x {
      x = x_size - x - 1;
    }
    if flip_y {
      y = y_size - y - 1;
    }

    if transpose {
      std::mem::swap(&mut x, &mut y);
    }
    Location::get_loc(x, y, if transpose { y_size } else { x_size })
  }

  /// upstream: SymmetryHelpers::getSymLoc (nninputs.cpp)
  pub fn get_sym_loc_x_y(x: i32, y: i32, board: &Board, symmetry: i32) -> Loc {
    SymmetryHelpers::get_sym_loc_x_y_x_size_y_size(x, y, board.x_size, board.y_size, symmetry)
  }

  /// upstream: SymmetryHelpers::getSymLoc (nninputs.cpp)
  pub fn get_sym_loc(loc: Loc, board: &Board, symmetry: i32) -> Loc {
    if loc == Board::NULL_LOC || loc == Board::PASS_LOC {
      return loc;
    }
    SymmetryHelpers::get_sym_loc_x_y(Location::get_x(loc, board.x_size), Location::get_y(loc, board.x_size), board, symmetry)
  }

  /// upstream: SymmetryHelpers::getSymLoc (nninputs.cpp)
  pub fn get_sym_loc_x_size_y_size(loc: Loc, x_size: i32, y_size: i32, symmetry: i32) -> Loc {
    if loc == Board::NULL_LOC || loc == Board::PASS_LOC {
      return loc;
    }
    SymmetryHelpers::get_sym_loc_x_y_x_size_y_size(Location::get_x(loc, x_size), Location::get_y(loc, x_size), x_size, y_size, symmetry)
  }

  /// upstream: SymmetryHelpers::getSymBoard (nninputs.cpp)
  //Applies a symmetry to a board
  pub fn get_sym_board(board: &Board, symmetry: i32) -> Result<Board, KataError> {
    let transpose: bool = (symmetry & 0x4) != 0;
    let flip_x: bool = (symmetry & 0x2) != 0;
    let flip_y: bool = (symmetry & 0x1) != 0;
    let mut sym_board: Board =
      Board::new_x_y(if transpose { board.y_size } else { board.x_size }, if transpose { board.x_size } else { board.y_size })?;
    let mut sym_ko_loc: Loc = Board::NULL_LOC;
    for y in 0..board.y_size {
      for x in 0..board.x_size {
        let loc: Loc = Location::get_loc(x, y, board.x_size);
        let mut sym_x: i32 = if flip_x { board.x_size - x - 1 } else { x };
        let mut sym_y: i32 = if flip_y { board.y_size - y - 1 } else { y };
        if transpose {
          std::mem::swap(&mut sym_x, &mut sym_y);
        }
        let sym_loc: Loc = Location::get_loc(sym_x, sym_y, sym_board.x_size);
        let suc: bool = sym_board.set_stone_fail_if_no_libs(sym_loc, board.colors[loc as usize]);
        testAssert!(suc);
        if loc == board.ko_loc {
          sym_ko_loc = sym_loc;
        }
      }
    }
    //Set only at the end because otherwise setStoneFailIfNoLibs clears it.
    if sym_ko_loc != Board::NULL_LOC {
      sym_board.set_simple_ko_loc(sym_ko_loc);
    }
    Ok(sym_board)
  }

  /// upstream: SymmetryHelpers::isTranspose (nninputs.h)
  #[inline]
  pub fn is_transpose(symmetry: i32) -> bool {
    (symmetry & 0x4) != 0
  }
  /// upstream: SymmetryHelpers::isFlipX (nninputs.h)
  #[inline]
  pub fn is_flip_x(symmetry: i32) -> bool {
    (symmetry & 0x2) != 0
  }
  /// upstream: SymmetryHelpers::isFlipY (nninputs.h)
  #[inline]
  pub fn is_flip_y(symmetry: i32) -> bool {
    (symmetry & 0x1) != 0
  }

  /// upstream: SymmetryHelpers::markDuplicateMoveLocs (nninputs.cpp)
  //Fill isSymDupLoc with true on all but one copy of each symmetrically equivalent move, and false everywhere else.
  //isSymDupLocs should be an array of size Board::MAX_ARR_SIZE
  //If onlySymmetries is not NULL, will only consider the symmetries specified there.
  //validSymmetries will be filled with all symmetries of the current board, including using history for checking ko/superko and some encore-related state.
  //This implementation is dependent on specific order of the symmetries (i.e. transpose is coded as 0x4)
  //Will pretend moves that have a nonzero value in avoidMoves do not exist.
  pub fn mark_duplicate_move_locs(
    board: &Board,
    hist: &BoardHistory,
    only_symmetries: Option<&[i32]>,
    avoid_moves: &[i32],
    is_sym_dup_loc: &mut [bool],
    valid_symmetries: &mut Vec<i32>,
  ) {
    is_sym_dup_loc[..Board::MAX_ARR_SIZE as usize].fill(false);
    valid_symmetries.clear();
    valid_symmetries.reserve(SymmetryHelpers::NUM_SYMMETRIES as usize);
    valid_symmetries.push(0);

    //The board should never be considered symmetric if any moves are banned by ko or superko
    if board.ko_loc != Board::NULL_LOC {
      return;
    }
    for y in 0..board.y_size {
      for x in 0..board.x_size {
        if hist.super_ko_banned[Location::get_loc(x, y, board.x_size) as usize] {
          return;
        }
      }
    }

    //If board has different sizes of x and y, we will not search symmetries involved with transpose.
    let symmetry_search_upper_bound: i32 =
      if board.x_size == board.y_size { SymmetryHelpers::NUM_SYMMETRIES } else { SymmetryHelpers::NUM_SYMMETRIES_WITHOUT_TRANSPOSE };

    for symmetry in 1..symmetry_search_upper_bound {
      if let Some(only) = only_symmetries {
        if !only.contains(&symmetry) {
          continue;
        }
      }

      let mut is_board_sym: bool = true;
      for y in 0..board.y_size {
        for x in 0..board.x_size {
          let loc: Loc = Location::get_loc(x, y, board.x_size);
          let sym_loc: Loc = SymmetryHelpers::get_sym_loc_x_y(x, y, board, symmetry);
          let is_stone_sym: bool = board.colors[loc as usize] == board.colors[sym_loc as usize];
          let is_ko_recap_blocked_sym: bool =
            if hist.encore_phase > 0 { hist.ko_recap_blocked[loc as usize] == hist.ko_recap_blocked[sym_loc as usize] } else { true };
          let is_second_encore_start_colors_sym: bool = if hist.encore_phase == 2 {
            hist.second_encore_start_colors[loc as usize] == hist.second_encore_start_colors[sym_loc as usize]
          } else {
            true
          };
          if !is_stone_sym || !is_ko_recap_blocked_sym || !is_second_encore_start_colors_sym {
            is_board_sym = false;
            break;
          }
        }
        if !is_board_sym {
          break;
        }
      }
      if is_board_sym {
        valid_symmetries.push(symmetry);
      }
    }

    //The way we iterate is to achieve https://senseis.xmp.net/?PlayingTheFirstMoveInTheUpperRightCorner%2FDiscussion
    //Reverse the iteration order for white, so that natural openings result in white on the left and black on the right
    //as is common now in SGFs
    if hist.presumed_next_move_pla == P_BLACK {
      let mut x: i32 = board.x_size - 1;
      while x >= 0 {
        for y in 0..board.y_size {
          let loc: Loc = Location::get_loc(x, y, board.x_size);
          if !avoid_moves.is_empty() && avoid_moves[loc as usize] > 0 {
            continue;
          }
          for &symmetry in valid_symmetries.iter() {
            if symmetry == 0 {
              continue;
            }
            let sym_loc: Loc = SymmetryHelpers::get_sym_loc_x_y(x, y, board, symmetry);
            if !is_sym_dup_loc[loc as usize] && loc != sym_loc {
              is_sym_dup_loc[sym_loc as usize] = true;
            }
          }
        }
        x -= 1;
      }
    } else {
      for x in 0..board.x_size {
        let mut y: i32 = board.y_size - 1;
        while y >= 0 {
          let loc: Loc = Location::get_loc(x, y, board.x_size);
          if !avoid_moves.is_empty() && avoid_moves[loc as usize] > 0 {
            y -= 1;
            continue;
          }
          for &symmetry in valid_symmetries.iter() {
            if symmetry == 0 {
              continue;
            }
            let sym_loc: Loc = SymmetryHelpers::get_sym_loc_x_y(x, y, board, symmetry);
            if !is_sym_dup_loc[loc as usize] && loc != sym_loc {
              is_sym_dup_loc[sym_loc as usize] = true;
            }
          }
          y -= 1;
        }
      }
    }
  }

  /// upstream: SymmetryHelpers::getSymmetryDifferences (nninputs.cpp)
  // For each symmetry, return a metric about the "amount" of difference that board would have with other
  // if symmetry were applied to board.
  pub fn get_symmetry_differences(
    board: &Board,
    other: &Board,
    max_difference_to_report: f64,
    symmetry_differences: &mut [f64; SymmetryHelpers::NUM_SYMMETRIES as usize],
  ) {
    for symmetry in 0..SymmetryHelpers::NUM_SYMMETRIES {
      symmetry_differences[symmetry as usize] = max_difference_to_report;
    }

    // Don't bother handling ultra-fancy transpose logic
    if board.x_size != other.x_size || board.y_size != other.y_size {
      return;
    }

    let mut num_symmetries: i32 = SymmetryHelpers::NUM_SYMMETRIES;
    if board.x_size != board.y_size {
      num_symmetries = SymmetryHelpers::NUM_SYMMETRIES_WITHOUT_TRANSPOSE;
    }

    for symmetry in 0..num_symmetries {
      symmetry_differences[symmetry as usize] = get_symmetry_difference(board, other, symmetry, max_difference_to_report);
    }
  }
}

//-------------------------------------------------------------------------------------------------------------

/// upstream: setRowBin (nninputs.cpp) static
#[inline]
fn set_row_bin(row_bin: &mut [f32], pos: i32, feature: i32, value: f32, pos_stride: i32, feature_stride: i32) {
  row_bin[(pos * pos_stride + feature * feature_stride) as usize] = value;
}

/// upstream: iterLadders (nninputs.cpp) static
/// PORT NOTE: `const std::function<void(Loc,int,const vector<Loc>&)>&` -> `&mut dyn FnMut` (PORTING.md §3:
/// called from one thread; the lambdas write rowBin).
//Calls f on each location that is part of an inescapable atari, or a group that can be put into inescapable atari
fn iter_ladders(board: &Board, nn_x_len: i32, f: &mut dyn FnMut(Loc, i32, &[Loc])) {
  let x_size: i32 = board.x_size;
  let y_size: i32 = board.y_size;

  let mut chain_heads_solved: [Loc; Board::MAX_PLAY_SIZE as usize] = [0; Board::MAX_PLAY_SIZE as usize];
  let mut chain_heads_solved_value: [bool; Board::MAX_PLAY_SIZE as usize] = [false; Board::MAX_PLAY_SIZE as usize];
  let mut num_chain_heads_solved: i32 = 0;
  let mut copy: Board = board.clone();
  let mut buf: Vec<Loc> = Vec::new();
  let mut working_moves: Vec<Loc> = Vec::new();

  for y in 0..y_size {
    for x in 0..x_size {
      let loc: Loc = Location::get_loc(x, y, x_size);
      let stone: Color = board.colors[loc as usize];
      if stone == P_BLACK || stone == P_WHITE {
        let libs: i32 = board.get_num_liberties(loc);
        if libs == 1 || libs == 2 {
          let mut already_solved: bool = false;
          let pos: i32 = NNPos::xy_to_pos(x, y, nn_x_len);
          let head: Loc = board.chain_head[loc as usize];
          for i in 0..num_chain_heads_solved {
            if chain_heads_solved[i as usize] == head {
              already_solved = true;
              if chain_heads_solved_value[i as usize] {
                working_moves.clear();
                f(loc, pos, &working_moves);
              }
              break;
            }
          }
          if !already_solved {
            //Perform search on copy so as not to mess up tracking of solved heads
            let laddered: bool;
            if libs == 1 {
              laddered = copy.search_is_ladder_captured(loc, true, &mut buf);
            } else {
              working_moves.clear();
              laddered = copy.search_is_ladder_captured_attacker_first2_libs(loc, &mut buf, &mut working_moves);
            }

            chain_heads_solved[num_chain_heads_solved as usize] = head;
            chain_heads_solved_value[num_chain_heads_solved as usize] = laddered;
            num_chain_heads_solved += 1;
            if laddered {
              f(loc, pos, &working_moves);
            }
          }
        }
      }
    }
  }
}

impl NNInputs {
  /// upstream: NNInputs::getHash (nninputs.cpp)
  //Currently does NOT depend on history (except for marking ko-illegal spots)
  pub fn get_hash(board: &Board, hist: &BoardHistory, next_player: Player, nn_input_params: &MiscNNInputParams) -> Hash128 {
    //Hash using the effective BoardHistoryModes for this eval, which are normally hist's own
    //but may be overridden per-query (e.g. for a secondary net whose declared featurization differs).
    let mut hash: Hash128 = BoardHistory::get_situation_rules_and_ko_hash_modes(
      board, hist, next_player, nn_input_params.draw_equivalent_wins_for_white,
      &nn_input_params.get_modes(hist)
    );

    //Fold in whether a pass ends this phase.
    if hist.pass_would_end_phase(board, next_player) {
      hash ^= Board::ZOBRIST_PASS_ENDS_PHASE;
      //Technically some of the below only apply when passing ends the game, but it's pretty harmless to use the more
      //conservative hashing including them when the phase would end too.

      //And in the case that a pass ends the phase, conservativePass also affects the result for the root node
      if nn_input_params.conservative_pass_and_is_root {
        hash ^= MiscNNInputParams::ZOBRIST_CONSERVATIVE_PASS;
      }

      //If we're in a ruleset where passing without capturing all the stones is okay, and as a result are suppressing
      //the game end effect of a pass during search, hash this in.
      if hist.should_suppress_end_game_from_friendly_pass(board, next_player) {
        hash ^= MiscNNInputParams::ZOBRIST_FRIENDLY_PASS;
      }

      //Passing hacks can also affect things at game or phase end.
      if nn_input_params.enable_passing_hacks {
        hash ^= MiscNNInputParams::ZOBRIST_PASSING_HACKS;
      }
    }
    //Fold in whether the game is over or not, since this affects how we compute input features
    //but is not a function necessarily of previous hashed values.
    //If the history is in a weird prolonged state, also treat it similarly.
    if hist.is_game_finished || hist.is_past_normal_phase_end {
      hash ^= Board::ZOBRIST_GAME_IS_OVER;
    }

    //Distinguish between forcing the history to always be empty and allow any nonempty amount
    //We already tolerate caching and reusing evals across distinct transpositions with different
    //history, for performance reasons, but the case of no history at all is probably distinct
    //enough that we should distinguish it.
    if nn_input_params.max_history <= 0 {
      hash.hash0 = hash.hash0.wrapping_add(MiscNNInputParams::ZOBRIST_ZERO_HISTORY.hash0);
      hash.hash1 = hash.hash1.wrapping_add(MiscNNInputParams::ZOBRIST_ZERO_HISTORY.hash1);
    }

    //Fold in asymmetric playout indicator
    if nn_input_params.playout_doubling_advantage != 0.0 {
      // PORT NOTE: C++ UB for out-of-range/NaN; x86-64 gives INT64_MIN (port::cpp::static_cast_i64_f64).
      let playout_doublings_discretized: i64 = cpp::static_cast_i64_f64(nn_input_params.playout_doubling_advantage * 256.0f32 as f64);
      hash.hash0 = hash.hash0.wrapping_add(Hash::split_mix64(playout_doublings_discretized as u64));
      hash.hash1 = hash.hash1.wrapping_add(Hash::basic_l_cong(playout_doublings_discretized as u64));
      hash ^= MiscNNInputParams::ZOBRIST_PLAYOUT_DOUBLINGS;
    }

    //Fold in policy temperature
    if nn_input_params.nn_policy_temperature != 1.0f32 {
      // PORT NOTE: C++ UB for out-of-range/NaN; x86-64 gives INT64_MIN (port::cpp::static_cast_i64_f32).
      let nn_policy_temperature_discretized: i64 = cpp::static_cast_i64_f32(nn_input_params.nn_policy_temperature * 2048.0f32);
      hash.hash0 ^= Hash::basic_l_cong2(nn_policy_temperature_discretized as u64);
      hash.hash1 = Hash::split_mix64(hash.hash1.wrapping_add(nn_policy_temperature_discretized as u64));
      hash.hash0 = hash.hash0.wrapping_add(hash.hash1);
      hash ^= MiscNNInputParams::ZOBRIST_NN_POLICY_TEMP;
    }

    if nn_input_params.avoid_myt_dagger_hack {
      hash ^= MiscNNInputParams::ZOBRIST_AVOID_MYTDAGGER_HACK;
    }

    //Fold in policy optimism
    if nn_input_params.policy_optimism > 0.0 {
      hash ^= MiscNNInputParams::ZOBRIST_POLICY_OPTIMISM;
      // PORT NOTE: C++ UB for out-of-range; x86-64 gives INT64_MIN (port::cpp::static_cast_i64_f64).
      let policy_optimism_discretized: i64 = cpp::static_cast_i64_f64(nn_input_params.policy_optimism * 1024.0);
      hash.hash0 = Hash::rrmxmx(Hash::split_mix64(hash.hash0).wrapping_add(policy_optimism_discretized as u64));
      hash.hash1 = Hash::rrmxmx(hash.hash1.wrapping_add(hash.hash0).wrapping_add(policy_optimism_discretized as u64));
    }

    hash
  }
}

impl MiscNNInputParams {
  /// upstream: MiscNNInputParams::getAlwaysComputePassAliveUnderSuicideRules (nninputs.cpp)
  // The effective value of modes.alwaysComputePassAliveUnderSuicideRules for featurization: the
  // override if one is set, else hist's own flag.
  pub fn get_always_compute_pass_alive_under_suicide_rules(&self, hist: &BoardHistory) -> bool {
    if self.pass_alive_suicide_rules_override >= 0 {
      return self.pass_alive_suicide_rules_override != 0;
    }
    hist.modes.always_compute_pass_alive_under_suicide_rules
  }

  /// upstream: MiscNNInputParams::getExcludeTerritoryAdjacentToAtari (nninputs.cpp)
  // The effective value of modes.excludeTerritoryAdjacentToAtari for featurization: the override
  // if one is set, else hist's own flag.
  pub fn get_exclude_territory_adjacent_to_atari(&self, hist: &BoardHistory) -> bool {
    if self.exclude_territory_adj_atari_override >= 0 {
      return self.exclude_territory_adj_atari_override != 0;
    }
    hist.modes.exclude_territory_adjacent_to_atari
  }

  /// upstream: MiscNNInputParams::getModes (nninputs.cpp)
  // The effective full modes for featurization, taking both overrides into account.
  pub fn get_modes(&self, hist: &BoardHistory) -> BoardHistoryModes {
    BoardHistoryModes::new_full(
      self.get_always_compute_pass_alive_under_suicide_rules(hist),
      self.get_exclude_territory_adjacent_to_atari(hist)
    )
  }

  /// upstream: MiscNNInputParams::getSuicideLegalForPassAlive (nninputs.cpp)
  // The suicide legality to use for pass-alive area computations in featurization, taking the
  // override and hist.modes.alwaysComputePassAliveUnderSuicideRules into account.
  pub fn get_suicide_legal_for_pass_alive(&self, hist: &BoardHistory) -> bool {
    hist.rules.multi_stone_suicide_legal || self.get_always_compute_pass_alive_under_suicide_rules(hist)
  }
}

impl NNInputs {
  //===========================================================================================
  //INPUTSVERSION 7
  //===========================================================================================

  /// upstream: NNInputs::fillRowV7 (nninputs.cpp)
  pub fn fill_row_v7(
    board: &Board,
    hist: &BoardHistory,
    next_player: Player,
    nn_input_params: &MiscNNInputParams,
    nn_x_len: i32,
    nn_y_len: i32,
    use_nhwc: bool,
    row_bin: &mut [f32],
    row_global: &mut [f32],
  ) {
    debug_assert!(nn_x_len <= NNPos::MAX_BOARD_LEN);
    debug_assert!(nn_y_len <= NNPos::MAX_BOARD_LEN);
    debug_assert!(board.x_size <= nn_x_len);
    debug_assert!(board.y_size <= nn_y_len);
    row_bin[..(NNInputs::NUM_FEATURES_SPATIAL_V7 * nn_x_len * nn_y_len) as usize].fill(0.0f32);
    row_global[..NNInputs::NUM_FEATURES_GLOBAL_V7 as usize].fill(0.0f32);

    let pla: Player = next_player;
    let opp: Player = get_opp(pla);
    let x_size: i32 = board.x_size;
    let y_size: i32 = board.y_size;

    let feature_stride: i32;
    let pos_stride: i32;
    if use_nhwc {
      feature_stride = 1;
      pos_stride = NNInputs::NUM_FEATURES_SPATIAL_V7;
    } else {
      feature_stride = nn_x_len * nn_y_len;
      pos_stride = 1;
    }

    for y in 0..y_size {
      for x in 0..x_size {
        let pos: i32 = NNPos::xy_to_pos(x, y, nn_x_len);
        let loc: Loc = Location::get_loc(x, y, x_size);

        //Feature 0 - on board
        set_row_bin(row_bin, pos, 0, 1.0f32, pos_stride, feature_stride);

        let stone: Color = board.colors[loc as usize];

        //Features 1,2 - pla,opp stone
        //Features 3,4,5 - 1,2,3 libs
        if stone == pla {
          set_row_bin(row_bin, pos, 1, 1.0f32, pos_stride, feature_stride);
        } else if stone == opp {
          set_row_bin(row_bin, pos, 2, 1.0f32, pos_stride, feature_stride);
        }

        if stone == pla || stone == opp {
          let libs: i32 = board.get_num_liberties(loc);
          if libs == 1 {
            set_row_bin(row_bin, pos, 3, 1.0f32, pos_stride, feature_stride);
          } else if libs == 2 {
            set_row_bin(row_bin, pos, 4, 1.0f32, pos_stride, feature_stride);
          } else if libs == 3 {
            set_row_bin(row_bin, pos, 5, 1.0f32, pos_stride, feature_stride);
          }
        }
      }
    }

    //Feature 6 - ko-ban locations, including possibly superko.
    if hist.encore_phase == 0 {
      if board.ko_loc != Board::NULL_LOC {
        let pos: i32 = NNPos::loc_to_pos(board.ko_loc, x_size, nn_x_len, nn_y_len);
        set_row_bin(row_bin, pos, 6, 1.0f32, pos_stride, feature_stride);
      }
      for y in 0..y_size {
        for x in 0..x_size {
          let loc: Loc = Location::get_loc(x, y, x_size);
          if hist.super_ko_banned[loc as usize] && loc != board.ko_loc {
            let pos: i32 = NNPos::loc_to_pos(loc, x_size, nn_x_len, nn_y_len);
            set_row_bin(row_bin, pos, 6, 1.0f32, pos_stride, feature_stride);
          }
        }
      }
    } else {
      //Feature 6,7,8 - in the encore, no-second-ko-capture locations, encore ko prohibitions where we have to pass for ko
      for y in 0..y_size {
        for x in 0..x_size {
          let loc: Loc = Location::get_loc(x, y, x_size);
          let pos: i32 = NNPos::loc_to_pos(loc, x_size, nn_x_len, nn_y_len);
          if hist.super_ko_banned[loc as usize] {
            set_row_bin(row_bin, pos, 6, 1.0f32, pos_stride, feature_stride);
          }
          if hist.ko_recap_blocked[loc as usize] {
            set_row_bin(row_bin, pos, 7, 1.0f32, pos_stride, feature_stride);
          }
        }
      }
    }

    //Features 18,19 - current territory, not counting group tax
    let mut area: [Color; Board::MAX_ARR_SIZE as usize] = [0; Board::MAX_ARR_SIZE as usize];
    let mut has_area_feature: bool = false;
    let mut group_tax_adjustment_for_pla: i32 = 0;
    if hist.rules.scoring_rule == Rules::SCORING_AREA && hist.rules.tax_rule == Rules::TAX_NONE {
      has_area_feature = true;
      let non_pass_alive_stones: bool = true;
      let safe_big_territories: bool = true;
      let unsafe_big_territories: bool = true;
      board.calculate_area(&mut area, non_pass_alive_stones, safe_big_territories, unsafe_big_territories, nn_input_params.get_suicide_legal_for_pass_alive(hist));
    } else {
      let mut keep_territories: bool = false;
      let mut keep_stones: bool = false;
      let mut white_minus_black_independent_life_region_count: i32 = 0;
      if hist.rules.scoring_rule == Rules::SCORING_AREA && (hist.rules.tax_rule == Rules::TAX_SEKI || hist.rules.tax_rule == Rules::TAX_ALL) {
        has_area_feature = true;
        keep_territories = false;
        keep_stones = true;
      } else if hist.rules.scoring_rule == Rules::SCORING_TERRITORY && hist.rules.tax_rule == Rules::TAX_NONE {
        //Territory scoring omits feature until we reach the stage where scoring matters
        if hist.encore_phase >= 2 {
          has_area_feature = true;
          keep_territories = true;
          keep_stones = false;
        }
      } else if hist.rules.scoring_rule == Rules::SCORING_TERRITORY
        && (hist.rules.tax_rule == Rules::TAX_SEKI || hist.rules.tax_rule == Rules::TAX_ALL)
      {
        //Territory scoring omits feature until we reach the stage where scoring matters
        if hist.encore_phase >= 2 {
          has_area_feature = true;
          keep_territories = false;
          keep_stones = false;
        }
      } else {
        unreachable!();
      }

      if has_area_feature {
        board.calculate_independent_life_area(
          &mut area,
          &mut white_minus_black_independent_life_region_count,
          keep_territories,
          keep_stones,
          nn_input_params.get_exclude_territory_adjacent_to_atari(hist),
          nn_input_params.get_suicide_legal_for_pass_alive(hist),
        );
        if hist.rules.tax_rule == Rules::TAX_ALL {
          group_tax_adjustment_for_pla = if pla == P_WHITE {
            -2 * white_minus_black_independent_life_region_count
          } else {
            2 * white_minus_black_independent_life_region_count
          };
        }
      }
    }

    let mut final_phase_and_game_end_would_not_be_win: bool = false;
    if has_area_feature {
      let mut board_score_for_pla: i32 = group_tax_adjustment_for_pla;
      for y in 0..y_size {
        for x in 0..x_size {
          let loc: Loc = Location::get_loc(x, y, x_size);
          let pos: i32 = NNPos::loc_to_pos(loc, x_size, nn_x_len, nn_y_len);
          if area[loc as usize] == pla {
            set_row_bin(row_bin, pos, 18, 1.0f32, pos_stride, feature_stride);
            board_score_for_pla += 1;
          } else if area[loc as usize] == opp {
            set_row_bin(row_bin, pos, 19, 1.0f32, pos_stride, feature_stride);
            board_score_for_pla -= 1;
          } else {
            if hist.rules.scoring_rule == Rules::SCORING_TERRITORY {
              //Also we must be in the second encore phase, based on the logic above.
              if board.colors[loc as usize] == pla && hist.second_encore_start_colors[loc as usize] == pla {
                set_row_bin(row_bin, pos, 18, 1.0f32, pos_stride, feature_stride);
                board_score_for_pla += 1;
              } else if board.colors[loc as usize] == opp && hist.second_encore_start_colors[loc as usize] == opp {
                set_row_bin(row_bin, pos, 19, 1.0f32, pos_stride, feature_stride);
                board_score_for_pla -= 1;
              }
            }
          }
        }
      }
      let self_komi: f32 = hist.current_self_komi(pla, nn_input_params.draw_equivalent_wins_for_white);
      let final_score_pla: f32 = board_score_for_pla as f32 + self_komi;
      // If the game ended here, and was scored instantly, it would be a loss or a draw?
      if final_score_pla as f64 <= 0.0 {
        final_phase_and_game_end_would_not_be_win = true;
      }
    }

    //Hide history from the net if a pass would end things and we're behaving as if a pass won't.
    //Or if the game is in fact over right now!
    let mut max_turns_of_history_to_include: i32 = 5;
    let mut suppress_pass_would_end_phase: bool = false;
    if hist.pass_would_end_game(board, next_player)
      && (
        //At the root, if assuming passing doesn't end the game, and it would, then need to mask that out.
        nn_input_params.conservative_pass_and_is_root ||
       //Deeper in the tree, we might not assume passes end the game in a friendly pass setting.
       hist.should_suppress_end_game_from_friendly_pass(board,next_player) ||
       //Passing hacks suppress the net to end the game when losing if it thinks a premature pass will lose by less.
       (nn_input_params.enable_passing_hacks && final_phase_and_game_end_would_not_be_win)
      )
    {
      max_turns_of_history_to_include = 0;
      suppress_pass_would_end_phase = true;
    } else if hist.is_game_finished || hist.is_past_normal_phase_end {
      // Include one of the passes, at the end of that sequence
      max_turns_of_history_to_include = 1;
    }
    max_turns_of_history_to_include = std::cmp::min(max_turns_of_history_to_include, nn_input_params.max_history);

    let mut num_turns_of_history_included: i32 = 0;

    //Features 9,10,11,12,13
    if max_turns_of_history_to_include > 0 {
      let move_history: &Vec<Move> = &hist.move_history;
      let move_history_len: usize = move_history.len();
      debug_assert!(move_history_len >= hist.num_approx_valid_turns_this_phase as usize);

      //Effectively wipe history as we change phase by also capping it
      let amount_of_history_to_try_to_use: i32 = std::cmp::min(max_turns_of_history_to_include, hist.num_approx_valid_turns_this_phase);

      if amount_of_history_to_try_to_use >= 1 && move_history[move_history_len - 1].pla == opp {
        let prev1_loc: Loc = move_history[move_history_len - 1].loc;
        num_turns_of_history_included = 1;
        if prev1_loc == Board::PASS_LOC {
          row_global[0] = 1.0f32;
        } else if prev1_loc != Board::NULL_LOC {
          let pos: i32 = NNPos::loc_to_pos(prev1_loc, x_size, nn_x_len, nn_y_len);
          set_row_bin(row_bin, pos, 9, 1.0f32, pos_stride, feature_stride);
        }
        if amount_of_history_to_try_to_use >= 2 && move_history[move_history_len - 2].pla == pla {
          let prev2_loc: Loc = move_history[move_history_len - 2].loc;
          num_turns_of_history_included = 2;
          if prev2_loc == Board::PASS_LOC {
            row_global[1] = 1.0f32;
          } else if prev2_loc != Board::NULL_LOC {
            let pos: i32 = NNPos::loc_to_pos(prev2_loc, x_size, nn_x_len, nn_y_len);
            set_row_bin(row_bin, pos, 10, 1.0f32, pos_stride, feature_stride);
          }
          if amount_of_history_to_try_to_use >= 3 && move_history[move_history_len - 3].pla == opp {
            let prev3_loc: Loc = move_history[move_history_len - 3].loc;
            num_turns_of_history_included = 3;
            if prev3_loc == Board::PASS_LOC {
              row_global[2] = 1.0f32;
            } else if prev3_loc != Board::NULL_LOC {
              let pos: i32 = NNPos::loc_to_pos(prev3_loc, x_size, nn_x_len, nn_y_len);
              set_row_bin(row_bin, pos, 11, 1.0f32, pos_stride, feature_stride);
            }
            if amount_of_history_to_try_to_use >= 4 && move_history[move_history_len - 4].pla == pla {
              let prev4_loc: Loc = move_history[move_history_len - 4].loc;
              num_turns_of_history_included = 4;
              if prev4_loc == Board::PASS_LOC {
                row_global[3] = 1.0f32;
              } else if prev4_loc != Board::NULL_LOC {
                let pos: i32 = NNPos::loc_to_pos(prev4_loc, x_size, nn_x_len, nn_y_len);
                set_row_bin(row_bin, pos, 12, 1.0f32, pos_stride, feature_stride);
              }
              if amount_of_history_to_try_to_use >= 5 && move_history[move_history_len - 5].pla == opp {
                let prev5_loc: Loc = move_history[move_history_len - 5].loc;
                num_turns_of_history_included = 5;
                if prev5_loc == Board::PASS_LOC {
                  row_global[4] = 1.0f32;
                } else if prev5_loc != Board::NULL_LOC {
                  let pos: i32 = NNPos::loc_to_pos(prev5_loc, x_size, nn_x_len, nn_y_len);
                  set_row_bin(row_bin, pos, 13, 1.0f32, pos_stride, feature_stride);
                }
              }
            }
          }
        }
      }
    }

    //Ladder features 14,15,16,17
    let mut add_ladder_feature = |loc: Loc, pos: i32, working_moves: &[Loc]| {
      debug_assert!(board.colors[loc as usize] == P_BLACK || board.colors[loc as usize] == P_WHITE);
      debug_assert!(pos >= 0 && pos < NNPos::MAX_BOARD_AREA);
      set_row_bin(row_bin, pos, 14, 1.0f32, pos_stride, feature_stride);
      if board.colors[loc as usize] == opp && board.get_num_liberties(loc) > 1 {
        for j in 0..working_moves.len() {
          let working_pos: i32 = NNPos::loc_to_pos(working_moves[j], x_size, nn_x_len, nn_y_len);
          set_row_bin(row_bin, working_pos, 17, 1.0f32, pos_stride, feature_stride);
        }
      }
    };

    iter_ladders(board, nn_x_len, &mut add_ladder_feature);

    let prev_board: &Board = if num_turns_of_history_included < 1 { board } else { hist.get_recent_board(1) };
    let mut add_prev_ladder_feature = |loc: Loc, pos: i32, working_moves: &[Loc]| {
      let _ = working_moves;
      let _ = loc;
      debug_assert!(prev_board.colors[loc as usize] == P_BLACK || prev_board.colors[loc as usize] == P_WHITE);
      debug_assert!(pos >= 0 && pos < NNPos::MAX_BOARD_AREA);
      set_row_bin(row_bin, pos, 15, 1.0f32, pos_stride, feature_stride);
    };
    iter_ladders(prev_board, nn_x_len, &mut add_prev_ladder_feature);

    let prev_prev_board: &Board = if num_turns_of_history_included < 2 { prev_board } else { hist.get_recent_board(2) };
    let mut add_prev_prev_ladder_feature = |loc: Loc, pos: i32, working_moves: &[Loc]| {
      let _ = working_moves;
      let _ = loc;
      debug_assert!(prev_prev_board.colors[loc as usize] == P_BLACK || prev_prev_board.colors[loc as usize] == P_WHITE);
      debug_assert!(pos >= 0 && pos < NNPos::MAX_BOARD_AREA);
      set_row_bin(row_bin, pos, 16, 1.0f32, pos_stride, feature_stride);
    };
    iter_ladders(prev_prev_board, nn_x_len, &mut add_prev_prev_ladder_feature);

    //Features 20, 21 - second encore starting stones
    if hist.encore_phase >= 2 {
      for y in 0..y_size {
        for x in 0..x_size {
          let loc: Loc = Location::get_loc(x, y, x_size);
          let pos: i32 = NNPos::loc_to_pos(loc, x_size, nn_x_len, nn_y_len);
          if hist.second_encore_start_colors[loc as usize] == pla {
            set_row_bin(row_bin, pos, 20, 1.0f32, pos_stride, feature_stride);
          } else if hist.second_encore_start_colors[loc as usize] == opp {
            set_row_bin(row_bin, pos, 21, 1.0f32, pos_stride, feature_stride);
          }
        }
      }
    }

    //Global features.
    //The first 5 of them were set already above to flag which of the past 5 moves were passes.

    //Komi and any score adjustments
    let mut self_komi: f32 = hist.current_self_komi(next_player, nn_input_params.draw_equivalent_wins_for_white);
    let b_area: f32 = (x_size * y_size) as f32;
    //Bound komi just in case
    if self_komi > b_area + NNPos::KOMI_CLIP_RADIUS {
      self_komi = b_area + NNPos::KOMI_CLIP_RADIUS;
    }
    if self_komi < -b_area - NNPos::KOMI_CLIP_RADIUS {
      self_komi = -b_area - NNPos::KOMI_CLIP_RADIUS;
    }
    row_global[5] = self_komi / 20.0f32;

    //Ko rule
    if hist.rules.ko_rule == Rules::KO_SIMPLE {
    } else if hist.rules.ko_rule == Rules::KO_POSITIONAL || hist.rules.ko_rule == Rules::KO_SPIGHT {
      row_global[6] = 1.0f32;
      row_global[7] = 0.5f32;
    } else if hist.rules.ko_rule == Rules::KO_SITUATIONAL {
      row_global[6] = 1.0f32;
      row_global[7] = -0.5f32;
    } else {
      unreachable!();
    }

    //Suicide
    if hist.rules.multi_stone_suicide_legal {
      row_global[8] = 1.0f32;
    }

    //Scoring
    if hist.rules.scoring_rule == Rules::SCORING_AREA {
    } else if hist.rules.scoring_rule == Rules::SCORING_TERRITORY {
      row_global[9] = 1.0f32;
    } else {
      unreachable!();
    }
    //Tax
    if hist.rules.tax_rule == Rules::TAX_NONE {
    } else if hist.rules.tax_rule == Rules::TAX_SEKI {
      row_global[10] = 1.0f32;
    } else if hist.rules.tax_rule == Rules::TAX_ALL {
      row_global[10] = 1.0f32;
      row_global[11] = 1.0f32;
    } else {
      unreachable!();
    }

    //Encore phase
    if hist.encore_phase > 0 {
      row_global[12] = 1.0f32;
    }
    if hist.encore_phase > 1 {
      row_global[13] = 1.0f32;
    }

    //Does a pass end the current phase given the ruleset and history?
    let pass_would_end_phase: bool = if suppress_pass_would_end_phase { false } else { hist.pass_would_end_phase(board, next_player) };
    row_global[14] = if pass_would_end_phase { 1.0f32 } else { 0.0f32 };

    //Used for handicap play
    //Parameter 15 is used because there's actually a discontinuity in how training behavior works when this is
    //nonzero, no matter how slightly.
    if nn_input_params.playout_doubling_advantage != 0.0 {
      row_global[15] = 1.0f32;
      row_global[16] = (0.5 * nn_input_params.playout_doubling_advantage) as f32;
    }

    //Button
    if hist.has_button {
      row_global[17] = 1.0f32;
    }

    //Provide parity information about the board size and komi
    //This comes from the following observation:
    //From white's perspective:
    //Komi = 0.0 - Draw possible
    //Komi = 0.5 - Win the games we would have drawn with komi 0.0
    //Komi = 1.0 - Usually no difference from komi 0.5
    //Komi = 1.5 - Usually no difference from komi 0.5
    //Komi = 2.0 - Draw possible
    //If we were to assign an "effective goodness" to these komis in order it would look like
    //0 1 1 1 2 3 3 3 4 5 5 5 6 ...
    //since when away from the right parity, increasing the komi doesn't help us except in cases of seki with odd numbers of dame.
    //If we were to add 0.5 times a vector like:
    //0 -1 0 1 0 -1 0 1 0 -1 0 ...
    //Then this would become a linear function and hopefully easier for a neural net to learn.
    //We expect that this is hard for a neural net to learn since it depends on the parity of the board size
    //and is very "xor"like.
    //So we provide it as an input.
    //Since we are using a model where games are jittered by 0.5 (see BoardHistory::whiteKomiAdjustmentForDraws)
    //in theory right thing to first order to provide should be a triangular wave with a period of 2 komi points:
    //  ../\........
    //  ./..\.......
    //  /....\..../.
    //  ......\../..
    //  .......\/...
    //The upsloping part of the wave is centered around the komi value where you could draw
    //since komi is extra valuable when it turns losses into draws into wins, peaking at the komi value where you could draw + 0.5.
    //It's downsloping around the komi value where you can't draw, since the marginal komi there is nearly useless, not causing you to win
    //more games except in case of odd-dame seki.

    if hist.rules.scoring_rule == Rules::SCORING_AREA || hist.encore_phase >= 2 {
      let board_area_is_even: bool = (x_size * y_size) % 2 == 0;

      //What is the parity of the komi values that can produce jigos?
      let drawable_komis_are_even: bool = board_area_is_even;

      //Find the difference between the komi viewed from our perspective and the nearest drawable komi below it.
      let komi_floor: f32;
      if drawable_komis_are_even {
        komi_floor = (self_komi / 2.0f32).floor() * 2.0f32;
      } else {
        komi_floor = ((self_komi - 1.0f32) / 2.0f32).floor() * 2.0f32 + 1.0f32;
      }

      //Cap just in case we have floating point weirdness
      let mut delta: f32 = self_komi - komi_floor;
      debug_assert!(delta >= -0.0001f32);
      debug_assert!(delta <= 2.0001f32);
      if delta < 0.0f32 {
        delta = 0.0f32;
      }
      if delta > 2.0f32 {
        delta = 2.0f32;
      }

      //Create the triangle wave based on the difference
      let wave: f32;
      if delta < 0.5f32 {
        wave = delta;
      } else if delta < 1.5f32 {
        wave = 1.0f32 - delta;
      } else {
        wave = delta - 2.0f32;
      }

      //NOTE: If ever changing which feature this is, must also update index in model.py where we multiply it into the scorebelief parity vector
      row_global[18] = wave;
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn symmetry_compose_invert() {
    for a in 0..8 {
      assert_eq!(SymmetryHelpers::compose(a, SymmetryHelpers::invert(a)), 0);
      for b in 0..8 {
        // Composing on locations agrees with compose().
        let c = SymmetryHelpers::compose(a, b);
        for (x, y) in [(0, 0), (1, 2), (4, 3)] {
          let l1 = SymmetryHelpers::get_sym_loc_x_y_x_size_y_size(x, y, 5, 5, a);
          let l2 = SymmetryHelpers::get_sym_loc_x_size_y_size(l1, 5, 5, b);
          assert_eq!(l2, SymmetryHelpers::get_sym_loc_x_y_x_size_y_size(x, y, 5, 5, c));
        }
      }
    }
  }

  #[test]
  fn copy_inputs_then_outputs_round_trips() {
    let src: Vec<f32> = (0..(2 * 5 * 5)).map(|i| i as f32).collect();
    for sym in 0..8 {
      let mut tmp = vec![0.0f32; src.len()];
      let mut back = vec![0.0f32; src.len()];
      SymmetryHelpers::copy_inputs_with_symmetry(&src, &mut tmp, 2, 5, 5, 1, false, sym);
      SymmetryHelpers::copy_outputs_with_symmetry(&tmp, &mut back, 2, 5, 5, sym);
      assert_eq!(src, back, "symmetry {}", sym);
    }
  }

  #[test]
  fn score_value_basics() {
    ScoreValue::init_tables();
    assert_eq!(ScoreValue::get_score_stdev(1.0, 0.5), 0.0);
    assert_eq!(ScoreValue::white_score_mean_sq_of_score_gridded(1.5, 0.5), 2.25);
    assert_eq!(ScoreValue::white_score_mean_sq_of_score_gridded(2.0, 0.5), 0.5 * (1.5 * 1.5) + 0.5 * (2.5 * 2.5));
    let v = ScoreValue::expected_white_score_value(0.0, 0.0, 0.0, 1.0, 19.0);
    assert!(v.abs() < 1e-9);
  }
}
