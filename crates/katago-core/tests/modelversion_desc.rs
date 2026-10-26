//! Not a port of upstream code: neuralnet/modelversion.rs and neuralnet/desc.rs vs
//! oracle/modelversion_desc.cpp (the unmodified upstream NNModelVersion, activations.h,
//! ModelPostProcessParams() and ModelDesc::getSupportedRules), compared line by line.

#[macro_use]
mod common;

use common::Cases;
use common::fixture;
use katago_core::game::rules::Rules;
use katago_core::neuralnet::desc::ACTIVATION_IDENTITY;
use katago_core::neuralnet::desc::ACTIVATION_MISH;
use katago_core::neuralnet::desc::ACTIVATION_MISH_SCALE8;
use katago_core::neuralnet::desc::ACTIVATION_RELU;
use katago_core::neuralnet::desc::ModelDesc;
use katago_core::neuralnet::desc::ModelPostProcessParams;
use katago_core::neuralnet::modelversion::NNModelVersion;

fn v(name: &str, arg: i32, r: Result<i32, katago_core::core::global::KataError>) -> String {
  match r {
    Ok(x) => format!("V {} {} {}", name, arg, x),
    Err(e) => format!("V {} {} ERR {}", name, arg, e.what()),
  }
}

wasm_test! {
fn modelversion_and_desc_vs_cpp() {
  let mut cases = Cases::new("modelversion_desc::modelversion_and_desc_vs_cpp");
  let expected = fixture("oracle", "modelversion_desc.txt");
  let mut out: Vec<String> = Vec::new();
  out.push(format!("C latestModelVersionImplemented {}", NNModelVersion::latestModelVersionImplemented));
  out.push(format!("C latestInputsVersionImplemented {}", NNModelVersion::latestInputsVersionImplemented));
  out.push(format!("C defaultModelVersion {}", NNModelVersion::defaultModelVersion));
  out.push(format!("C oldestModelVersionImplemented {}", NNModelVersion::oldestModelVersionImplemented));
  out.push(format!("C oldestInputsVersionImplemented {}", NNModelVersion::oldestInputsVersionImplemented));
  out.push(format!("C ACTIVATION_IDENTITY {}", ACTIVATION_IDENTITY));
  out.push(format!("C ACTIVATION_RELU {}", ACTIVATION_RELU));
  out.push(format!("C ACTIVATION_MISH {}", ACTIVATION_MISH));
  out.push(format!("C ACTIVATION_MISH_SCALE8 {}", ACTIVATION_MISH_SCALE8));
  for mv in -2..=20 {
    out.push(v("getInputsVersion", mv, NNModelVersion::get_inputs_version(mv)));
    out.push(v("getNumSpatialFeatures", mv, NNModelVersion::get_num_spatial_features(mv)));
    out.push(v("getNumGlobalFeatures", mv, NNModelVersion::get_num_global_features(mv)));
  }
  for me in -1..=3 {
    out.push(v("getNumInputMetaChannels", me, NNModelVersion::get_num_input_meta_channels(me)));
  }
  let pp = ModelPostProcessParams::new();
  out.push(format!("P tdScoreMultiplier {:016X}", pp.td_score_multiplier.to_bits()));
  out.push(format!("P scoreMeanMultiplier {:016X}", pp.score_mean_multiplier.to_bits()));
  out.push(format!("P scoreStdevMultiplier {:016X}", pp.score_stdev_multiplier.to_bits()));
  out.push(format!("P leadMultiplier {:016X}", pp.lead_multiplier.to_bits()));
  out.push(format!("P varianceTimeMultiplier {:016X}", pp.variance_time_multiplier.to_bits()));
  out.push(format!("P shorttermValueErrorMultiplier {:016X}", pp.shortterm_value_error_multiplier.to_bits()));
  out.push(format!("P shorttermScoreErrorMultiplier {:016X}", pp.shortterm_score_error_multiplier.to_bits()));
  out.push(format!("P outputScaleMultiplier {:08X}", pp.output_scale_multiplier.to_bits()));

  for mv in 3..=16 {
    let mut desc = ModelDesc::new();
    desc.model_version = mv;
    for ko in 0..4 {
      for sc in 0..2 {
        for tax in 0..3 {
          for suic in 0..2 {
            for button in 0..2 {
              for whb in 0..3 {
                for fpok in 0..2 {
                  let rules = Rules::new_full(ko, sc, tax, suic != 0, button != 0, whb, fpok != 0, 6.5f32);
                  let mut supported = true;
                  let r = desc.get_supported_rules(&rules, &mut supported);
                  out.push(format!(
                    "R {} {} {} {} {} {} {} {} -> {} {} {} {} {} {} {} {:08X} {}",
                    mv,
                    ko,
                    sc,
                    tax,
                    suic,
                    button,
                    whb,
                    fpok,
                    r.ko_rule,
                    r.scoring_rule,
                    r.tax_rule,
                    r.multi_stone_suicide_legal as i32,
                    r.has_button as i32,
                    r.white_handicap_bonus_rule,
                    r.friendly_pass_ok as i32,
                    r.komi.to_bits(),
                    supported as i32
                  ));
                }
              }
            }
          }
        }
      }
    }
  }
  let exp: Vec<&str> = expected.lines().collect();
  assert_eq!(out.len(), exp.len(), "line count");
  for (a, e) in out.iter().zip(exp.iter()) {
    assert_eq!(a, e);
    cases.ran();
  }
}
}
