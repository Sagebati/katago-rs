// Ported from KataGo cpp/neuralnet/sgfmetadata.h @ ba938676d7f42d70950b3a535af2466fb642008c
//
// M2 subset (port-manifest.toml: the module is M8): only the SGFMetadata struct and its constants,
// because NNEvaluator::evaluate takes a `const SGFMetadata*` and reads `initialized`.
// Not ported yet (TODO(port): deferred (sgfmetadata.cpp), M8): operator==/!=, getHash,
// fillMetadataRow (needs port::math sin/cos/log), getProfile beyond its "no profile" branch, and the `gameDate` field
// (SimpleDate, core/datetime.cpp, M8). They are only reached for human-SL nets
// (numInputMetaChannels > 0), which tools/bin2onnx rejects, so no backend can load one yet.

use crate::core::global::KataError;

/// upstream: SGFMetadata (sgfmetadata.h)
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

  // TODO(port): deferred (datetime.cpp SimpleDate, M8): `SimpleDate gameDate;`

  pub source: i32,
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

  /// upstream: SGFMetadata::getProfile (sgfmetadata.cpp)
  /// PORT NOTE (M3 wave 2b, Setup::loadParams reads humanSLProfile for every bot): only the first
  /// branch is ported, the one every config without a human-SL profile takes (an empty name, "_" or
  /// "\"\"" -> a default, uninitialized SGFMetadata). Throws -> Result.
  pub fn get_profile(human_sl_profile_name: &str) -> Result<SGFMetadata, KataError> {
    if human_sl_profile_name == "" || human_sl_profile_name == "_" || human_sl_profile_name == "\"\"" {
      return Ok(SGFMetadata::default());
    }

    // TODO(port): deferred (sgfmetadata.cpp, M8): the "proyear_<year>", "rank_<r>" / "preaz_<r>" and
    // "rank_<b>_<w>" profiles need makeHistoricalProProfile / makeModernProProfile / makeBasicRankProfile
    // (SimpleDate gameDate, core/datetime.cpp). Only human-SL nets use a profile, and tools/bin2onnx rejects
    // those, so this fails instead of guessing; upstream throws only for unknown names, with
    // "Unknown human SL network profile: " + name.
    Err(KataError::StringError("humanSLProfile ".to_string() + human_sl_profile_name + " is not supported yet (SGFMetadata::getProfile is M8 in katago-rs)"))
  }
}
