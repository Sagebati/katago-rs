//! A debugSkipNeuralNet NNEvaluator for the tests that replay upstream's
//! `TestSearchCommon::startNNEval(..., debugSkipNeuralNet = true, ...)` without a search
//! (testexcludeterritoryatari.rs, testpassalivesuicide.rs). Not a port of upstream code, except the
//! tagged test-local copy below.

#![allow(dead_code, clippy::too_many_arguments)]

use std::rc::Rc;

use futures::executor::LocalPool;
use katago_core::core::commontypes::enabled_t;
use katago_core::core::config_parser::ConfigParser;
use katago_core::core::global::KataError;
use katago_core::neuralnet::desc::ModelDesc;
use katago_core::neuralnet::nneval::NNEvaluator;
use katago_core::neuralnet::nneval::NNResultBuf;
use katago_core::neuralnet::nninputs::NNOutput;
use katago_core::neuralnet::nninterface::*;
use katago_core::port::executor::LocalBoxFuture;
use katago_core::port::executor::Spawner;

use crate::common::run_or_stall_panic;

/// debugSkipNeuralNet: NNEvaluator only asks the backend for getNumEffectiveDevices (dummybackend.cpp's
/// body); everything else is never reached.
pub struct NoBackend;

impl NeuralNet for NoBackend {
  fn global_initialize(&self) {}
  fn global_cleanup(&self) {}
  fn print_devices(&self) {}
  fn get_runtime_backend_detail(&self, _: &mut ConfigParser) -> String {
    String::new()
  }
  fn get_batch_policy(&self, _: &mut ConfigParser) -> BatchPolicy {
    BatchPolicy::Dynamic
  }
  fn get_num_effective_devices(&self, _: &mut ConfigParser, gpu_idx_by_server_thread: &[i32]) -> i32 {
    let distinct_devices: std::collections::BTreeSet<i32> = gpu_idx_by_server_thread.iter().copied().collect();
    std::cmp::max(1, distinct_devices.len() as i32)
  }
  fn set_is_warmup(&self, _: &dyn ComputeHandle, _: bool) -> bool {
    false
  }
  fn load_model_file(&self, _: &str, _: Vec<u8>, _: &str) -> Result<Box<dyn LoadedModel>, KataError> {
    panic!("debugSkipNeuralNet: no model is loaded")
  }
  fn get_model_desc<'a>(&self, _: &'a dyn LoadedModel) -> &'a ModelDesc {
    panic!("debugSkipNeuralNet")
  }
  fn create_compute_context(
    &self,
    _: &[i32],
    _: i32,
    _: i32,
    _: &str,
    _: enabled_t,
    _: &dyn LoadedModel,
    _: &mut ConfigParser,
  ) -> Result<Box<dyn ComputeContext>, KataError> {
    panic!("debugSkipNeuralNet")
  }
  fn create_compute_handle(&self, _: &dyn ComputeContext, _: &dyn LoadedModel, _: i32, _: bool, _: bool, _: i32, _: i32) -> Result<Box<dyn ComputeHandle>, KataError> {
    panic!("debugSkipNeuralNet")
  }
  fn is_using_fp16(&self, _: &dyn ComputeHandle) -> bool {
    panic!("debugSkipNeuralNet")
  }
  fn create_input_buffers(&self, _: &dyn LoadedModel, _: i32, _: i32, _: i32) -> Box<dyn InputBuffers> {
    panic!("debugSkipNeuralNet")
  }
  fn get_output<'a>(
    &'a self,
    _: &'a mut dyn ComputeHandle,
    _: &'a mut dyn InputBuffers,
    _: i32,
    _: &'a [Rc<NNResultBuf>],
    _: &'a mut [NNOutput],
  ) -> LocalBoxFuture<'a, Result<(), KataError>> {
    panic!("debugSkipNeuralNet")
  }
}

/// upstream: TestSearchCommon::startNNEval (testsearchcommon.cpp) — test-local copy
/// PORT NOTE: `Logger&` dropped (log crate); the backend is NoBackend, so only debugSkipNeuralNet = true works;
/// `spawnServerThreads()` is async, driven to completion on `pool`; the `sleep_for(0.1)` ordering hack is not
/// needed (one OS thread, deterministic).
pub fn start_nn_eval(
  pool: &mut LocalPool,
  spawner: &Rc<dyn Spawner>,
  model_file: &str,
  seed: &str,
  nn_x_len: i32,
  nn_y_len: i32,
  default_symmetry: i32,
  inputs_use_nhwc: bool,
  use_nhwc: bool,
  use_fp16: bool,
  debug_skip_neural_net: bool,
  require_exact_nn_len: bool,
) -> Rc<NNEvaluator> {
  let gpu_idx_by_server_thread: Vec<i32> = vec![0];
  let max_batch_size = 16;
  let nn_cache_size_power_of_two = 16;
  let nn_mutex_pool_size_power_of_two = 12;
  //bool debugSkipNeuralNet = false;
  let model_name = model_file;
  let home_data_dir_override = "";
  let mut cfg = ConfigParser::new(false, true);
  //NHWC layout is no longer a generic NNEvaluator option; only the CUDA backend reads it (off cfg).
  //Route the test's useNHWC param into a cudaUseNHWC override so it still drives the CUDA layout.
  cfg.override_key("cudaUseNHWC", if use_nhwc { "true" } else { "false" });
  //The ONNX backend requires an explicit provider choice in real configs, so tests set cpu.
  cfg.override_key("onnxProvider", "cpu");
  let num_nn_server_threads_per_model = 1;
  let mut nn_randomize = false;
  let nn_rand_seed = "runSearchTestsRandSeed".to_string() + seed;

  let mut default_symmetry = default_symmetry;
  if default_symmetry == -1 {
    nn_randomize = true;
    default_symmetry = 0;
  }

  let expected_sha256 = "";
  let nn_eval = NNEvaluator::new(
    model_name,
    model_file,
    expected_sha256,
    max_batch_size,
    nn_x_len,
    nn_y_len,
    require_exact_nn_len,
    inputs_use_nhwc,
    nn_cache_size_power_of_two,
    nn_mutex_pool_size_power_of_two,
    debug_skip_neural_net,
    home_data_dir_override,
    if use_fp16 { enabled_t::True } else { enabled_t::False },
    num_nn_server_threads_per_model,
    &gpu_idx_by_server_thread,
    &nn_rand_seed,
    nn_randomize,
    default_symmetry,
    &mut cfg,
    Rc::new(NoBackend),
    spawner.clone(),
    Vec::new(),
  )
  .unwrap();
  let nn_eval = Rc::new(nn_eval);

  run_or_stall_panic(pool, "spawnServerThreads", nn_eval.spawn_server_threads()).unwrap();
  nn_eval
}

/// The `sameOutputs` lambda of the "nn cache cannot mix results across featurization modes" sections.
pub fn same_outputs(a: &NNOutput, b: &NNOutput) -> bool {
  if a.white_win_prob != b.white_win_prob || a.white_loss_prob != b.white_loss_prob || a.white_score_mean != b.white_score_mean {
    return false;
  }
  for i in 0..katago_core::neuralnet::nninputs::NNPos::MAX_NN_POLICY_SIZE {
    if a.policy_probs[i as usize] != b.policy_probs[i as usize] {
      return false;
    }
  }
  true
}
