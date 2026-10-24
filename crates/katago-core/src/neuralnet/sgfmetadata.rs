// Ported from KataGo cpp/neuralnet/sgfmetadata.cpp, cpp/neuralnet/sgfmetadata.h @ ba938676d7f42d70950b3a535af2466fb642008c

use std::sync::OnceLock;

use crate::core::datetime::SimpleDate;
use crate::core::global::Global;
use crate::core::global::KataError;
use crate::core::hash::Hash;
use crate::core::hash::Hash128;
use crate::game::board::{P_BLACK, P_WHITE, Player};
use crate::port::cpp::{cpp_max, cpp_min};
use crate::port::math;
use crate::testAssert;

/// upstream: SGFMetadata (sgfmetadata.h)
/// PORT NOTE: the default member initializers are `Default` (gameDate: SimpleDate() = 1970-01-01).
#[derive(Clone, Debug, Default)]
pub struct SGFMetadata {
  pub initialized: bool,

  pub inverse_b_rank: i32, // KG = 0, 9d = 1, 8d = 2,... 1d = 9, 1k = 10, 2k = 11, ...
  pub inverse_w_rank: i32,
  pub b_is_unranked: bool, // KGS "-" rank
  pub w_is_unranked: bool,
  pub b_rank_is_unknown: bool, // No rank in file
  pub w_rank_is_unknown: bool,
  pub b_is_human: bool,
  pub w_is_human: bool,

  pub game_is_unrated: bool,
  pub game_ratedness_is_unknown: bool,

  //One-hot for all things with metadata
  pub tc_is_unknown: bool,
  pub tc_is_none: bool,
  pub tc_is_absolute: bool,
  pub tc_is_simple: bool,
  pub tc_is_byo_yomi: bool,
  pub tc_is_canadian: bool,
  pub tc_is_fischer: bool,

  pub main_time_seconds: f64,
  pub period_time_seconds: f64,
  pub byo_yomi_periods: i32,
  pub canadian_moves: i32,

  pub game_date: SimpleDate,

  pub source: i32,
}

impl PartialEq for SGFMetadata {
  /// upstream: SGFMetadata::operator==, SGFMetadata::operator!= (sgfmetadata.cpp)
  fn eq(&self, other: &SGFMetadata) -> bool {
    self.initialized == other.initialized
      && self.inverse_b_rank == other.inverse_b_rank
      && self.inverse_w_rank == other.inverse_w_rank
      && self.b_is_unranked == other.b_is_unranked
      && self.w_is_unranked == other.w_is_unranked
      && self.b_rank_is_unknown == other.b_rank_is_unknown
      && self.w_rank_is_unknown == other.w_rank_is_unknown
      && self.b_is_human == other.b_is_human
      && self.w_is_human == other.w_is_human
      && self.game_is_unrated == other.game_is_unrated
      && self.game_ratedness_is_unknown == other.game_ratedness_is_unknown
      && self.tc_is_unknown == other.tc_is_unknown
      && self.tc_is_none == other.tc_is_none
      && self.tc_is_absolute == other.tc_is_absolute
      && self.tc_is_simple == other.tc_is_simple
      && self.tc_is_byo_yomi == other.tc_is_byo_yomi
      && self.tc_is_canadian == other.tc_is_canadian
      && self.tc_is_fischer == other.tc_is_fischer
      && self.main_time_seconds == other.main_time_seconds
      && self.period_time_seconds == other.period_time_seconds
      && self.byo_yomi_periods == other.byo_yomi_periods
      && self.canadian_moves == other.canadian_moves
      && self.game_date == other.game_date
      && self.source == other.source
  }
}

impl SGFMetadata {
  /// upstream: SGFMetadata::SOURCE_OGS .. SOURCE_GO4GO (sgfmetadata.h)
  pub const SOURCE_OGS: i32 = 1;
  pub const SOURCE_KGS: i32 = 2;
  pub const SOURCE_FOX: i32 = 3;
  pub const SOURCE_TYGEM: i32 = 4;
  pub const SOURCE_GOGOD: i32 = 5;
  pub const SOURCE_GO4GO: i32 = 6;

  /// upstream: SGFMetadata::METADATA_INPUT_NUM_CHANNELS (sgfmetadata.h)
  pub const METADATA_INPUT_NUM_CHANNELS: i32 = 192;

  /// upstream: SGFMetadata::getHash (sgfmetadata.cpp)
  pub fn get_hash(&self, next_player: Player) -> Hash128 {
    if !self.initialized
      || self.inverse_b_rank < 0
      || self.inverse_b_rank >= 128
      || self.inverse_w_rank < 0
      || self.inverse_w_rank >= 128
      || self.source < 0
      || self.source >= 128
      || self.main_time_seconds < 0.0
      || self.period_time_seconds < 0.0
      || self.byo_yomi_periods < 0
      || self.canadian_moves < 0
    {
      Global::fatal_error("Invalid or uninitialized SGFMetadata for hash");
    }

    // PORT NOTE: uint32_t arithmetic wraps (wrapping_*); the operands are in range after the check above.
    let b: u32 = (self.inverse_b_rank as u32)
      .wrapping_add((self.b_is_unranked as u32).wrapping_mul(128u32))
      .wrapping_add((self.b_rank_is_unknown as u32).wrapping_mul(256u32))
      .wrapping_add((self.b_is_human as u32).wrapping_mul(512u32));
    let w: u32 = (self.inverse_w_rank as u32)
      .wrapping_add((self.w_is_unranked as u32).wrapping_mul(128u32))
      .wrapping_add((self.w_rank_is_unknown as u32).wrapping_mul(256u32))
      .wrapping_add((self.w_is_human as u32).wrapping_mul(512u32));

    let mut x0: u32 = 0;
    let mut x1: u32 = 0;
    let mut x2: u32 = 0;
    let mut x3: u32 = 0;

    if next_player == P_BLACK {
      x0 = x0.wrapping_add(b.wrapping_add(w << 10));
    } else {
      x0 = x0.wrapping_add(w.wrapping_add(b << 10));
    }

    x0 = x0.wrapping_add((self.game_is_unrated as u32) << 20);
    x0 = x0.wrapping_add((self.game_ratedness_is_unknown as u32) << 21);

    let mut which_tc: u32 = 0;
    if self.tc_is_unknown {
      which_tc = 1;
    } else if self.tc_is_none {
      which_tc = 2;
    } else if self.tc_is_absolute {
      which_tc = 3;
    } else if self.tc_is_simple {
      which_tc = 4;
    } else if self.tc_is_byo_yomi {
      which_tc = 5;
    } else if self.tc_is_canadian {
      which_tc = 6;
    } else if self.tc_is_fischer {
      which_tc = 7;
    }

    x0 = x0.wrapping_add(which_tc << 22);
    // 7 bits left for source going up to 128
    x0 = x0.wrapping_add((self.source as u32) << 25);

    if !self.main_time_seconds.is_finite() || !self.period_time_seconds.is_finite() {
      Global::fatal_error(
        &("Invalid SGFMetadata: non-finite time values, mainTimeSeconds=".to_string()
          + &Global::double_to_string(self.main_time_seconds)
          + " periodTimeSeconds="
          + &Global::double_to_string(self.period_time_seconds)),
      );
    }
    let main_time_seconds_capped: f64 = cpp_min(cpp_max(self.main_time_seconds, 0.0), 3.0 * 86400.0);
    // ~20 bits
    // PORT NOTE: in range because 0 <= main_time_seconds_capped * 4 <= 1036800.
    x1 = x1.wrapping_add((main_time_seconds_capped * 4.0) as u32);
    let period_time_seconds_capped: f64 = cpp_min(cpp_max(self.period_time_seconds, 0.0), 1.0 * 86400.0);
    // ~22 bits
    // PORT NOTE: in range because 0 <= period_time_seconds_capped * 32 <= 2764800.
    x2 = x2.wrapping_add((period_time_seconds_capped * 32.0) as u32);

    let byo_yomi_periods_capped: i32 = std::cmp::min(std::cmp::max(self.byo_yomi_periods, 0), 50);
    x1 = x1.wrapping_add((byo_yomi_periods_capped as u32) << 24);
    let canadian_moves_capped: i32 = std::cmp::min(std::cmp::max(self.canadian_moves, 0), 50);
    x2 = x2.wrapping_add((canadian_moves_capped as u32) << 24);

    let days_difference: i32 = self.game_date.num_days_after(&SimpleDate::new_y_m_d(1970, 1, 1).expect("upstream: cannot throw for constant input"));
    x3 = x3.wrapping_add(days_difference as u32);

    let mut h0: u64 = Hash::combine(x0, x1);
    let mut h1: u64 = Hash::combine(x2, x3);
    h0 = Hash::nasam(h0);
    h1 = h1.wrapping_add(h0);
    h1 = Hash::nasam(h1);
    h0 = h0.wrapping_add(h1);

    Hash128::new(h0, h1)
  }

  /// upstream: SGFMetadata::fillMetadataRow (sgfmetadata.cpp)
  /// PORT NOTE: `float* rowMetadata` -> `&mut [f32]` (at least METADATA_INPUT_NUM_CHANNELS long).
  pub fn fill_metadata_row(sgf_meta: Option<&SGFMetadata>, row_metadata: &mut [f32], next_player: Player, board_area: i32) {
    testAssert!(sgf_meta.is_some());
    let sgf_meta: &SGFMetadata = sgf_meta.expect("testAssert above");
    if !sgf_meta.initialized {
      Global::fatal_error("Invalid or uninitialized SGFMetadata");
    }

    for i in 0..SGFMetadata::METADATA_INPUT_NUM_CHANNELS as usize {
      row_metadata[i] = 0.0f32;
    }

    let pla_is_human: bool = if next_player == P_WHITE { sgf_meta.w_is_human } else { sgf_meta.b_is_human };
    let opp_is_human: bool = if next_player == P_WHITE { sgf_meta.b_is_human } else { sgf_meta.w_is_human };
    row_metadata[0] = if pla_is_human { 1.0f32 } else { 0.0f32 };
    row_metadata[1] = if opp_is_human { 1.0f32 } else { 0.0f32 };

    let pla_is_unranked: bool = if next_player == P_WHITE { sgf_meta.w_is_unranked } else { sgf_meta.b_is_unranked };
    let opp_is_unranked: bool = if next_player == P_WHITE { sgf_meta.b_is_unranked } else { sgf_meta.w_is_unranked };
    row_metadata[2] = if pla_is_unranked { 1.0f32 } else { 0.0f32 };
    row_metadata[3] = if opp_is_unranked { 1.0f32 } else { 0.0f32 };

    let pla_rank_is_unknown: bool = if next_player == P_WHITE { sgf_meta.w_rank_is_unknown } else { sgf_meta.b_rank_is_unknown };
    let opp_rank_is_unknown: bool = if next_player == P_WHITE { sgf_meta.b_rank_is_unknown } else { sgf_meta.w_rank_is_unknown };
    row_metadata[4] = if pla_rank_is_unknown { 1.0f32 } else { 0.0f32 };
    row_metadata[5] = if opp_rank_is_unknown { 1.0f32 } else { 0.0f32 };

    const RANK_START_IDX: i32 = 6;
    let inv_pla_rank: i32 = if next_player == P_WHITE { sgf_meta.inverse_w_rank } else { sgf_meta.inverse_b_rank };
    let inv_opp_rank: i32 = if next_player == P_WHITE { sgf_meta.inverse_b_rank } else { sgf_meta.inverse_w_rank };
    const RANK_LEN_PER_PLA: i32 = 34;
    if !pla_is_unranked {
      for i in 0..std::cmp::min(inv_pla_rank, RANK_LEN_PER_PLA) {
        row_metadata[(RANK_START_IDX + i) as usize] = 1.0f32;
      }
    }
    if !opp_is_unranked {
      for i in 0..std::cmp::min(inv_opp_rank, RANK_LEN_PER_PLA) {
        row_metadata[(RANK_START_IDX + RANK_LEN_PER_PLA + i) as usize] = 1.0f32;
      }
    }

    const _: () = assert!(74 == RANK_START_IDX + 2 * RANK_LEN_PER_PLA);
    row_metadata[74] = if sgf_meta.game_ratedness_is_unknown {
      0.5f32
    } else if sgf_meta.game_is_unrated {
      1.0f32
    } else {
      0.0f32
    };

    row_metadata[75] = if sgf_meta.tc_is_unknown { 1.0f32 } else { 0.0f32 };
    row_metadata[76] = if sgf_meta.tc_is_none { 1.0f32 } else { 0.0f32 };
    row_metadata[77] = if sgf_meta.tc_is_absolute { 1.0f32 } else { 0.0f32 };
    row_metadata[78] = if sgf_meta.tc_is_simple { 1.0f32 } else { 0.0f32 };
    row_metadata[79] = if sgf_meta.tc_is_byo_yomi { 1.0f32 } else { 0.0f32 };
    row_metadata[80] = if sgf_meta.tc_is_canadian { 1.0f32 } else { 0.0f32 };
    row_metadata[81] = if sgf_meta.tc_is_fischer { 1.0f32 } else { 0.0f32 };
    if row_metadata[75] + row_metadata[76] + row_metadata[77] + row_metadata[78] + row_metadata[79] + row_metadata[80] + row_metadata[81] != 1.0f32 {
      Global::fatal_error("SGFMetadata has invalid time control flags - exactly one must be set");
    }

    let main_time_seconds_capped: f64 = cpp_min(cpp_max(sgf_meta.main_time_seconds, 0.0), 3.0 * 86400.0);
    let period_time_seconds_capped: f64 = cpp_min(cpp_max(sgf_meta.period_time_seconds, 0.0), 1.0 * 86400.0);
    row_metadata[82] = (0.4 * (math::ln(main_time_seconds_capped + 60.0) - 6.5)) as f32;
    row_metadata[83] = (0.3 * (math::ln(period_time_seconds_capped + 1.0) - 3.0)) as f32;
    let byo_yomi_periods_capped: i32 = std::cmp::min(std::cmp::max(sgf_meta.byo_yomi_periods, 0), 50);
    let canadian_moves_capped: i32 = std::cmp::min(std::cmp::max(sgf_meta.canadian_moves, 0), 50);
    row_metadata[84] = (0.5 * (math::ln(byo_yomi_periods_capped as f64 + 2.0) - 1.5)) as f32;
    row_metadata[85] = (0.25 * (math::ln(canadian_moves_capped as f64 + 2.0) - 1.5)) as f32;

    row_metadata[86] = (0.5 * math::ln(board_area as f64 / 361.0)) as f32;

    let days_difference: f64 =
      sgf_meta.game_date.num_days_after(&SimpleDate::new_y_m_d(1970, 1, 1).expect("upstream: cannot throw for constant input")) as f64;
    const DATE_START_IDX: i32 = 87;
    const DATE_LEN: i32 = 32;
    // 7 because we're curious if there's a day-of-the-week effect
    // on gameplay...
    let mut period: f64 = 7.0;
    // PORT NOTE: function-local static -> fn-local OnceLock (PORTING.md §8).
    static FACTOR: OnceLock<f64> = OnceLock::new();
    let factor: f64 = *FACTOR.get_or_init(|| math::powf(80000.0, 1.0 / (DATE_LEN - 1) as f64));
    const twopi: f64 = 6.283185307179586476925;
    for i in 0..DATE_LEN {
      let num_revolutions: f64 = days_difference / period;
      // PORT NOTE: port::math::{cos, sin} are correctly rounded (CORE-MATH); glibc's differ from them by
      // 1 ulp on ~0.1% of these inputs, never after the cast to float (tests/math_sincos.rs).
      row_metadata[(DATE_START_IDX + i * 2 + 0) as usize] = math::cos(num_revolutions * twopi) as f32;
      row_metadata[(DATE_START_IDX + i * 2 + 1) as usize] = math::sin(num_revolutions * twopi) as f32;
      period *= factor;
    }
    const _: () = assert!(151 == DATE_START_IDX + 2 * DATE_LEN);

    if sgf_meta.source < 0 || sgf_meta.source >= 16 {
      Global::fatal_error(&("SGFMetadata has invalid source: ".to_string() + &Global::int_to_string(sgf_meta.source)));
    }
    row_metadata[(151 + sgf_meta.source) as usize] = 1.0f32;

    const _: () = assert!(151 + 16 < SGFMetadata::METADATA_INPUT_NUM_CHANNELS);
  }

  /// upstream: SGFMetadata::getProfile (sgfmetadata.cpp)
  /// PORT NOTE: throws -> Result.
  pub fn get_profile(human_sl_profile_name: &str) -> Result<SGFMetadata, KataError> {
    if human_sl_profile_name == "" || human_sl_profile_name == "_" || human_sl_profile_name == "\"\"" {
      return Ok(SGFMetadata::default());
    }

    if Global::is_prefix(human_sl_profile_name, "proyear_") {
      let year_str: String = Global::chop_prefix(human_sl_profile_name, "proyear_")?;
      let mut year: i32 = 0;
      let suc: bool = Global::try_string_to_int(&year_str, &mut year);
      // PORT NOTE: SimpleDate(year,6,1) cannot throw for these years.
      if suc && year >= 1800 && year <= 2020 {
        return Ok(make_historical_pro_profile(&SimpleDate::new_y_m_d(year, 6, 1).expect("upstream: valid date")));
      }
      if suc && year >= 2021 && year <= 2023 {
        return Ok(make_modern_pro_profile(&SimpleDate::new_y_m_d(year, 6, 1).expect("upstream: valid date")));
      }
    }
    if Global::is_prefix(human_sl_profile_name, "rank_") || Global::is_prefix(human_sl_profile_name, "preaz_") {
      let ranks_str: String;
      let pre_az: bool;
      if Global::is_prefix(human_sl_profile_name, "rank_") {
        ranks_str = Global::chop_prefix(human_sl_profile_name, "rank_")?;
        pre_az = false;
      } else {
        ranks_str = Global::chop_prefix(human_sl_profile_name, "preaz_")?;
        pre_az = true;
      }

      let get_inverse_rank = |rank_str: &str| -> i32 {
        if rank_str == "9d" {
          return 1;
        }
        if rank_str == "8d" {
          return 2;
        }
        if rank_str == "7d" {
          return 3;
        }
        if rank_str == "6d" {
          return 4;
        }
        if rank_str == "5d" {
          return 5;
        }
        if rank_str == "4d" {
          return 6;
        }
        if rank_str == "3d" {
          return 7;
        }
        if rank_str == "2d" {
          return 8;
        }
        if rank_str == "1d" {
          return 9;
        }
        if rank_str == "1k" {
          return 10;
        }
        if rank_str == "2k" {
          return 11;
        }
        if rank_str == "3k" {
          return 12;
        }
        if rank_str == "4k" {
          return 13;
        }
        if rank_str == "5k" {
          return 14;
        }
        if rank_str == "6k" {
          return 15;
        }
        if rank_str == "7k" {
          return 16;
        }
        if rank_str == "8k" {
          return 17;
        }
        if rank_str == "9k" {
          return 18;
        }
        if rank_str == "10k" {
          return 19;
        }
        if rank_str == "11k" {
          return 20;
        }
        if rank_str == "12k" {
          return 21;
        }
        if rank_str == "13k" {
          return 22;
        }
        if rank_str == "14k" {
          return 23;
        }
        if rank_str == "15k" {
          return 24;
        }
        if rank_str == "16k" {
          return 25;
        }
        if rank_str == "17k" {
          return 26;
        }
        if rank_str == "18k" {
          return 27;
        }
        if rank_str == "19k" {
          return 28;
        }
        if rank_str == "20k" {
          return 29;
        }
        -1
      };

      let inverse_rank: i32 = get_inverse_rank(&ranks_str);
      if inverse_rank != -1 {
        return Ok(make_basic_rank_profile(inverse_rank, inverse_rank, pre_az));
      }

      let pieces: Vec<String> = Global::split_delim(&ranks_str, '_');
      if pieces.len() == 2 {
        let inverse_rank_black: i32 = get_inverse_rank(&pieces[0]);
        let inverse_rank_white: i32 = get_inverse_rank(&pieces[1]);
        if inverse_rank_black != -1 && inverse_rank_white != -1 {
          return Ok(make_basic_rank_profile(inverse_rank_black, inverse_rank_white, pre_az));
        }
      }
    }

    Err(KataError::StringError("Unknown human SL network profile: ".to_string() + human_sl_profile_name))
  }
}

/// upstream: makeBasicRankProfile (sgfmetadata.cpp) static
fn make_basic_rank_profile(inverse_rank_black: i32, inverse_rank_white: i32, pre_az: bool) -> SGFMetadata {
  // KGS rating system is pretty reasonable, so let's use KGS as the source.
  let mut ret = SGFMetadata::default();
  ret.initialized = true;
  ret.inverse_b_rank = inverse_rank_black;
  ret.inverse_w_rank = inverse_rank_white;
  ret.b_is_human = true;
  ret.w_is_human = true;
  ret.game_ratedness_is_unknown = true;
  ret.tc_is_byo_yomi = true;
  ret.main_time_seconds = 1200.0;
  ret.period_time_seconds = 30.0;
  ret.byo_yomi_periods = 5;
  // PORT NOTE: constant valid dates (SimpleDate(int,int,int) cannot throw).
  if pre_az {
    ret.game_date = SimpleDate::new_y_m_d(2016, 9, 1).expect("upstream: cannot throw for constant input");
  } else {
    ret.game_date = SimpleDate::new_y_m_d(2020, 3, 1).expect("upstream: cannot throw for constant input");
  }
  ret.source = SGFMetadata::SOURCE_KGS;
  ret
}

/// upstream: makeHistoricalProProfile (sgfmetadata.cpp) static
fn make_historical_pro_profile(date: &SimpleDate) -> SGFMetadata {
  let mut ret = SGFMetadata::default();
  ret.initialized = true;
  ret.inverse_b_rank = 1;
  ret.inverse_w_rank = 1;
  ret.b_is_human = true;
  ret.w_is_human = true;
  ret.tc_is_unknown = true;
  ret.game_date = *date;
  ret.source = SGFMetadata::SOURCE_GOGOD;
  ret
}

/// upstream: makeModernProProfile (sgfmetadata.cpp) static
fn make_modern_pro_profile(date: &SimpleDate) -> SGFMetadata {
  let mut ret = SGFMetadata::default();
  ret.initialized = true;
  ret.inverse_b_rank = 1;
  ret.inverse_w_rank = 1;
  ret.b_is_human = true;
  ret.w_is_human = true;
  ret.tc_is_unknown = true;
  ret.game_date = *date;
  ret.source = SGFMetadata::SOURCE_GO4GO;
  ret
}
