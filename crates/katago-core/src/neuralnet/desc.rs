// Ported from KataGo cpp/neuralnet/desc.cpp, cpp/neuralnet/desc.h, cpp/neuralnet/activations.h @ fd0723fdbc0e9d82cf269c9630af8c27c57c07c4
//
// Scope (port-manifest.toml notes): the model-level header that NNEvaluator and the backends read
// (ModelDesc's scalar fields incl. the v1.18 prefer* flags and ArchSummary, ModelPostProcessParams,
// getSupportedRules, the archSummary branches of getTrunkSpatialConvDepth / getNumParameters /
// hasAnyTransformerBlocks / hasAnyNestedBottleneckBlocks, getShortInfoString, checkNameValid) and
// activations.h.
// The layer descriptors (ConvLayerDesc .. ValueHeadDesc, RMSNormLayerDesc, the Transformer*Desc,
// SGFMetadataEncoderDesc, the TRUNK_NORM_KIND_* / *_BLOCK_KIND constants), the istream parsers,
// iterConvLayers, maxConvChannels, the layer walks behind the arch-summary getters,
// transformToReduceActivations, applyScale8ToReduceActivations, releaseWeights and
// loadFromFileMaybeGZipped are NOT ported: the network itself goes through tools/bin2onnx (which
// ports the parsers) + burn-onnx.
// PORT NOTE: where the values come from. Upstream fills ModelDesc by parsing the whole `.bin.gz`
// (`ModelDesc(istream&, sha256, binaryFloats)`): the header gives name, version, input channel
// counts, the v13+ post-process multipliers, metaEncoderVersion and (v15+ slots, v1.18) the two
// prefer* flags, while numPolicyChannels, numValueChannels, numScoreValueChannels and
// numOwnershipChannels come from the head layers at the END of the file. A header-only reader
// cannot provide those, and the browser never downloads the `.bin.gz` (only burn's `.bpk` weights).
// So the backend builds the ModelDesc from the values bin2onnx's desc.cpp-faithful parser stored in
// the ONNX `metadata_props` (header values for v13+, the ModelPostProcessParams() defaults
// otherwise), embedded at build time by crates/katago-nn-burn/build.rs, as upstream's own ONNX
// loader does (onnxmodelbuilder.cpp: a desc reconstructed from a .onnx file carries an ArchSummary
// with `present = true` instead of layers). `sha256` stays empty (no expectedSha256 check:
// TODO(port) if a host needs it).

use crate::core::global::Global;
use crate::core::global::KataError;
use crate::game::rules::Rules;

/// upstream: ACTIVATION_IDENTITY, ACTIVATION_RELU, ACTIVATION_MISH, ACTIVATION_SILU, ACTIVATION_MISH_SCALE8 (activations.h)
// x
pub const ACTIVATION_IDENTITY: i32 = 0;

// max(x,0)
pub const ACTIVATION_RELU: i32 = 1;

// x * tanh(softplus(x))
pub const ACTIVATION_MISH: i32 = 2;

// x / (1+exp(-x))
pub const ACTIVATION_SILU: i32 = 3;

// x * tanh(softplus(8x))
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

  //True if the model expects its pass-alive area input features to be computed as if
  //multi-stone suicide were always legal, regardless of the actual suicide rule.
  pub prefer_pass_alive_under_suicide_rules: bool,

  //True if the model expects territory scoring with TaxRule NONE (both for adjudication and for its
  //territory input features) to exclude empty points adjacent to chains in atari, per rules version 3.
  pub prefer_exclude_territory_adjacent_to_atari: bool,

  pub post_process_params: ModelPostProcessParams,

  //Architecture summary values that are normally derived by walking trunk/policyHead/valueHead.
  //Set (present = true) only for a desc reconstructed from a .onnx file, which has no layer
  //structure to walk; the summary getters below then report these values instead.
  pub arch_summary: ArchSummary,
}

/// upstream: ModelDesc::ArchSummary (desc.h)
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ArchSummary {
  pub present: bool,
  pub trunk_spatial_conv_depth: f64,
  pub num_parameters: i64,
  pub has_any_transformer_blocks: bool,
  pub has_any_nested_bottleneck_blocks: bool,
}

impl ArchSummary {
  /// upstream: ModelDesc::ArchSummary::ArchSummary (desc.cpp)
  pub fn new() -> ArchSummary {
    ArchSummary {
      present: false,
      trunk_spatial_conv_depth: 0.0,
      num_parameters: 0,
      has_any_transformer_blocks: false,
      has_any_nested_bottleneck_blocks: false,
    }
  }
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
      prefer_pass_alive_under_suicide_rules: false,
      prefer_exclude_territory_adjacent_to_atari: false,
      post_process_params: ModelPostProcessParams::new(),
      arch_summary: ArchSummary::new(),
    }
  }

  /// upstream: ModelDesc::checkNameValid (desc.cpp)
  /// PORT NOTE: throws -> Result. upstream calls it from the istream constructor (not ported); the
  /// burn backend calls it on the ONNX metadata name when loading.
  //Throws StringError if name is not usable as a model name. Model names get embedded into on-disk
  //cache filenames (e.g. the TensorRT plan cache), so they are restricted to a short
  //filesystem-safe character set.
  pub fn check_name_valid(name: &str) -> Result<(), KataError> {
    // PORT NOTE: `name.size() <= 0` on a size_t is `== 0` (clippy::absurd_extreme_comparisons).
    if name.len() == 0 {
      return Err(KataError::StringError("Model name is empty, a nonempty model name is required".to_string()));
    }
    if name.len() > 96 {
      return Err(KataError::StringError(
        "Model name is too long (".to_string() + &Global::int_to_string(name.len() as i32) + " chars, max 96): " + name,
      ));
    }
    for c in name.bytes() {
      let ok: bool = (c >= b'A' && c <= b'Z') || (c >= b'a' && c <= b'z') || (c >= b'0' && c <= b'9') || c == b'_' || c == b'-';
      if !ok {
        return Err(KataError::StringError("Model name must contain only alphanumeric characters, underscores, and hyphens: ".to_string() + name));
      }
    }
    Ok(())
  }

  /// upstream: ModelDesc::getTrunkSpatialConvDepth (desc.cpp)
  /// PORT NOTE: the layer walk (`trunk.getSpatialConvDepth()`) is not ported (file header): every
  /// ModelDesc the port loads comes from ONNX metadata with `archSummary.present`, as upstream's
  /// ONNX loader; a desc without one has no layers to walk.
  pub fn get_trunk_spatial_conv_depth(&self) -> f64 {
    if self.arch_summary.present {
      return self.arch_summary.trunk_spatial_conv_depth;
    }
    Global::fatal_error("ModelDesc::getTrunkSpatialConvDepth: no archSummary and the layer descs are not ported")
  }

  /// upstream: ModelDesc::getNumParameters (desc.cpp)
  /// PORT NOTE: the layer walk is not ported, as getTrunkSpatialConvDepth.
  //Total count of learnable parameters across the whole model.
  pub fn get_num_parameters(&self) -> i64 {
    if self.arch_summary.present {
      return self.arch_summary.num_parameters;
    }
    Global::fatal_error("ModelDesc::getNumParameters: no archSummary and the layer descs are not ported")
  }

  /// upstream: ModelDesc::hasAnyNestedBottleneckBlocks (desc.cpp)
  /// PORT NOTE: the layer walk is not ported, as getTrunkSpatialConvDepth.
  pub fn has_any_nested_bottleneck_blocks(&self) -> bool {
    if self.arch_summary.present {
      return self.arch_summary.has_any_nested_bottleneck_blocks;
    }
    Global::fatal_error("ModelDesc::hasAnyNestedBottleneckBlocks: no archSummary and the layer descs are not ported")
  }

  /// upstream: ModelDesc::getShortInfoString (desc.cpp)
  //Short human-readable summary of the model architecture kind and parameter count, e.g.
  //"nbt transformer, 12345678 params". Backends can append this in parentheses after the model name.
  pub fn get_short_info_string(&self) -> String {
    let is_transformer: bool = self.has_any_transformer_blocks();
    let is_nbt: bool = self.has_any_nested_bottleneck_blocks();
    let kind: String;
    if is_nbt {
      kind = (if is_transformer { "nbt transformer" } else { "nbt convnet" }).to_string();
    } else {
      kind = (if is_transformer { "transformer" } else { "convnet" }).to_string();
    }
    kind + ", " + &Global::int64_to_string(self.get_num_parameters()) + " params"
  }

  /// upstream: ModelDesc::hasAnyTransformerBlocks (desc.cpp)
  /// PORT NOTE: the layer walk (`trunk.hasAnyTransformerBlocks()`) is not ported, as
  /// getTrunkSpatialConvDepth.
  //True if the model's trunk contains any transformer (attention or ffn) block. Useful for callers
  //that want to report model stats or special-case transformer-only behavior (e.g. graph warmup).
  pub fn has_any_transformer_blocks(&self) -> bool {
    if self.arch_summary.present {
      return self.arch_summary.has_any_transformer_blocks;
    }
    Global::fatal_error("ModelDesc::hasAnyTransformerBlocks: no archSummary and the layer descs are not ported")
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
