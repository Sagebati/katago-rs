// Ported from KataGo cpp/neuralnet/modelversion.cpp, cpp/neuralnet/modelversion.h @ ba938676d7f42d70950b3a535af2466fb642008c

use crate::core::global::Global;
use crate::core::global::KataError;
use crate::neuralnet::nninputs::NNInputs;

//Old model versions, no longer supported:
//0 = V1 features, with old head architecture using crelus (no longer supported)
//1 = V1 features, with new head architecture, no crelus
//2 = V2 features, no internal architecture change.

//Supported model versions:
//3 = V3 features, many architecture changes for new selfplay loop, including multiple board sizes
//4 = V3 features, scorebelief head
//5 = V4 features, changed current territory feature to just indicate pass-alive
//6 = V5 features, disable fancy features
//7 = V6 features, support new rules configurations
//8 = V7 features, unbalanced training, button go, lead and variance time
//9 = V7 features, shortterm value error
//10 = V7 features, shortterm value error done more properly
//11 = V7 features, supports mish activations by desc actually reading the activations
//12 = V7 features, optimisic policy head
//13 = V7 features, Adjusted scaling on shortterm score variance, and made C++ side read in scalings.
//14 = V7 features, Squared softplus for error variance predictions
//15 = V7 features, Extra nonlinearity for pass output
//16 = V7 features, Q value predictions in the policy head

/// upstream: NNModelVersion (modelversion.h)
// Model versions
pub struct NNModelVersion;

impl NNModelVersion {
  /// upstream: NNModelVersion::latestModelVersionImplemented, latestInputsVersionImplemented,
  /// defaultModelVersion, oldestModelVersionImplemented, oldestInputsVersionImplemented (modelversion.h)
  pub const latestModelVersionImplemented: i32 = 16;
  pub const latestInputsVersionImplemented: i32 = 7;
  pub const defaultModelVersion: i32 = 16;

  pub const oldestModelVersionImplemented: i32 = 3;
  pub const oldestInputsVersionImplemented: i32 = 3;

  /// upstream: NNModelVersion::getInputsVersion (modelversion.cpp)
  // Which V* feature version from NNInputs does a given model version consume?
  pub fn get_inputs_version(model_version: i32) -> Result<i32, KataError> {
    if model_version >= 8 && model_version <= 16 {
      return Ok(7);
    } else if model_version == 7 {
      return Ok(6);
    } else if model_version == 6 {
      return Ok(5);
    } else if model_version == 5 {
      return Ok(4);
    } else if model_version == 3 || model_version == 4 {
      return Ok(3);
    }

    fail(model_version)
  }

  /// upstream: NNModelVersion::getNumSpatialFeatures (modelversion.cpp)
  // Convenience functions, feeds forward the number of features and the size of
  // the row vector that the net takes as input
  pub fn get_num_spatial_features(model_version: i32) -> Result<i32, KataError> {
    if model_version >= 8 && model_version <= 16 {
      return Ok(NNInputs::NUM_FEATURES_SPATIAL_V7);
    } else if model_version == 7 {
      return Ok(NNInputs::NUM_FEATURES_SPATIAL_V6);
    } else if model_version == 6 {
      return Ok(NNInputs::NUM_FEATURES_SPATIAL_V5);
    } else if model_version == 5 {
      return Ok(NNInputs::NUM_FEATURES_SPATIAL_V4);
    } else if model_version == 3 || model_version == 4 {
      return Ok(NNInputs::NUM_FEATURES_SPATIAL_V3);
    }

    fail(model_version)
  }

  /// upstream: NNModelVersion::getNumGlobalFeatures (modelversion.cpp)
  pub fn get_num_global_features(model_version: i32) -> Result<i32, KataError> {
    if model_version >= 8 && model_version <= 16 {
      return Ok(NNInputs::NUM_FEATURES_GLOBAL_V7);
    } else if model_version == 7 {
      return Ok(NNInputs::NUM_FEATURES_GLOBAL_V6);
    } else if model_version == 6 {
      return Ok(NNInputs::NUM_FEATURES_GLOBAL_V5);
    } else if model_version == 5 {
      return Ok(NNInputs::NUM_FEATURES_GLOBAL_V4);
    } else if model_version == 3 || model_version == 4 {
      return Ok(NNInputs::NUM_FEATURES_GLOBAL_V3);
    }

    fail(model_version)
  }

  /// upstream: NNModelVersion::getNumInputMetaChannels (modelversion.cpp)
  // SGF metadata encoder input versions
  pub fn get_num_input_meta_channels(meta_encoder_version: i32) -> Result<i32, KataError> {
    if meta_encoder_version == 0 {
      return Ok(0);
    }
    if meta_encoder_version == 1 {
      return Ok(192);
    }
    Err(KataError::StringError(
      "NNModelVersion: metaEncoderVersion not currently implemented or supported: ".to_string()
        + &Global::int_to_string(meta_encoder_version),
    ))
  }
}

/// upstream: fail (modelversion.cpp) static
/// PORT NOTE: `throw` -> `Err`; the C++ `return -1` after `fail` is unreachable.
fn fail<T>(model_version: i32) -> Result<T, KataError> {
  Err(KataError::StringError(
    "NNModelVersion: Model version not currently implemented or supported: ".to_string() + &Global::int_to_string(model_version),
  ))
}

/// upstream: static_assert block (modelversion.cpp)
const _: () = assert!(NNModelVersion::oldestModelVersionImplemented == 3);
const _: () = assert!(NNModelVersion::oldestInputsVersionImplemented == 3);
const _: () = assert!(NNModelVersion::latestModelVersionImplemented == 16);
const _: () = assert!(NNModelVersion::latestInputsVersionImplemented == 7);
