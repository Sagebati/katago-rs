// Ported from KataGo cpp/neuralnet/desc.cpp, cpp/neuralnet/desc.h, cpp/neuralnet/activations.h @ ba938676d7f42d70950b3a535af2466fb642008c
//
// Scope (port-manifest.toml notes): the model-level header that NNEvaluator and the backends read
// (ModelDesc's scalar fields, ModelPostProcessParams, getSupportedRules) and activations.h.
// The layer descriptors (ConvLayerDesc .. ValueHeadDesc, SGFMetadataEncoderDesc), the istream
// parsers, iterConvLayers, maxConvChannels, getTrunkSpatialConvDepth, transformToReduceActivations,
// applyScale8ToReduceActivations and loadFromFileMaybeGZipped are NOT ported: the network itself
// goes through tools/bin2onnx (which ports the parsers) + burn-onnx.
// PORT NOTE: where the values come from. Upstream fills ModelDesc by parsing the whole `.bin.gz`
// (`ModelDesc(istream&, sha256, binaryFloats)`): the header gives name, version, input channel
// counts, the v13+ post-process multipliers and metaEncoderVersion, while numPolicyChannels,
// numValueChannels, numScoreValueChannels and numOwnershipChannels come from the head layers at
// the END of the file. A header-only reader cannot provide those, and the browser never downloads
// the `.bin.gz` (only burn's `.bpk` weights). So the backend builds the ModelDesc from the values
// bin2onnx's desc.cpp-faithful parser stored in the ONNX `metadata_props` (header values for v13+,
// the ModelPostProcessParams() defaults otherwise), embedded at build time by
// crates/katago-nn-burn/build.rs. `sha256` stays empty (no expectedSha256 check: TODO(port) if a
// host needs it).

use crate::game::rules::Rules;

/// upstream: ACTIVATION_IDENTITY, ACTIVATION_RELU, ACTIVATION_MISH, ACTIVATION_MISH_SCALE8 (activations.h)
pub const ACTIVATION_IDENTITY: i32 = 0;
pub const ACTIVATION_RELU: i32 = 1;
pub const ACTIVATION_MISH: i32 = 2;

pub const ACTIVATION_MISH_SCALE8: i32 = 12;

/// upstream: ModelPostProcessParams (desc.h)
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ModelPostProcessParams {
  pub td_score_multiplier: f64,
  pub score_mean_multiplier: f64,
  pub score_stdev_multiplier: f64,
  pub lead_multiplier: f64,
  pub variance_time_multiplier: f64,
  pub shortterm_value_error_multiplier: f64,
  pub shortterm_score_error_multiplier: f64,

  pub output_scale_multiplier: f32,
}

impl ModelPostProcessParams {
  /// upstream: ModelPostProcessParams::ModelPostProcessParams (desc.cpp)
  pub fn new() -> ModelPostProcessParams {
    ModelPostProcessParams {
      td_score_multiplier: 20.0,
      score_mean_multiplier: 20.0,
      score_stdev_multiplier: 20.0,
      lead_multiplier: 20.0,
      variance_time_multiplier: 40.0,
      shortterm_value_error_multiplier: 0.25,
      shortterm_score_error_multiplier: 30.0,
      output_scale_multiplier: 1.0f32,
    }
  }
}

/// upstream: ModelDesc (desc.h)
/// PORT NOTE: `trunk`, `policyHead`, `valueHead` are not ported (see the file header); the
/// fields derived from them upstream are plain fields filled by the backend.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelDesc {
  pub name: String,
  pub sha256: String,
  pub model_version: i32,
  pub num_input_channels: i32,
  pub num_input_global_channels: i32,
  pub num_input_meta_channels: i32,
  pub num_policy_channels: i32,
  pub num_value_channels: i32,
  pub num_score_value_channels: i32,
  pub num_ownership_channels: i32,

  pub meta_encoder_version: i32,

  pub post_process_params: ModelPostProcessParams,
}

impl ModelDesc {
  /// upstream: ModelDesc::ModelDesc() (desc.cpp)
  pub fn new() -> ModelDesc {
    ModelDesc {
      name: String::new(),
      sha256: String::new(),
      model_version: -1,
      num_input_channels: 0,
      num_input_global_channels: 0,
      num_input_meta_channels: 0,
      num_policy_channels: 0,
      num_value_channels: 0,
      num_score_value_channels: 0,
      num_ownership_channels: 0,
      meta_encoder_version: 0,
      post_process_params: ModelPostProcessParams::new(),
    }
  }

  /// upstream: ModelDesc::getSupportedRules (desc.cpp)
  //Return the "nearest" supported ruleset to desiredRules by this model.
  //Fills supported with true if desiredRules itself was exactly supported, false if some modifications had to be made.
  pub fn get_supported_rules(&self, desired_rules: &Rules, supported: &mut bool) -> Rules {
    let mut rules: Rules = *desired_rules;
    *supported = true;
    if self.model_version <= 6 {
      if rules.ko_rule == Rules::KO_SIMPLE || rules.ko_rule == Rules::KO_SPIGHT {
        rules.ko_rule = Rules::KO_SITUATIONAL;
        *supported = false;
      }
      if rules.scoring_rule == Rules::SCORING_TERRITORY {
        rules.scoring_rule = Rules::SCORING_AREA;
        *supported = false;
      }
      if rules.tax_rule != Rules::TAX_NONE {
        rules.tax_rule = Rules::TAX_NONE;
        *supported = false;
      }
      if rules.has_button {
        rules.has_button = false;
        *supported = false;
      }
    } else {
      if rules.ko_rule == Rules::KO_SPIGHT {
        rules.ko_rule = Rules::KO_SITUATIONAL;
        *supported = false;
      }
      if rules.has_button && rules.scoring_rule != Rules::SCORING_AREA {
        rules.has_button = false;
        *supported = false;
      }
    }

    return rules;
  }
}
