//! Setup::initializeNNEvaluators and the injected `NNEvaluatorHost` (D6): when upstream opens the
//! model file. Upstream's NNEvaluator constructor opens it (`NeuralNet::loadModelFile`) only when
//! `!debugSkipNeuralNet`, and only after its own argument checks (nnXLen/nnYLen too large,
//! maxBatchSize <= 0, gpuIdxByServerThread.size() != numThreads). The port's host hook
//! `load_model` stands in for that open, so it must follow the same two rules.
//! Not a port of upstream code.

#[macro_use]
mod common;

use std::cell::RefCell;
use std::rc::Rc;

use common::Cases;
use common::run_or_stall_panic;
use futures::executor::LocalPool;
use katago_core::core::config_parser::ConfigParser;
use katago_core::core::global::KataError;
use katago_core::core::rand::Rand;
use katago_core::neuralnet::nneval::NNEvaluator;
use katago_core::neuralnet::nninterface::BatchPolicy;
use katago_core::neuralnet::nninterface::NeuralNet;
use katago_core::port::executor::Spawner;
use katago_core::program::setup::MaxBatchSizeRequest;
use katago_core::program::setup::NNEvaluatorHost;
use katago_core::program::setup::Setup;
use katago_core::program::setup::setup_for_t;

/// `load_model` records its arguments and then fails with a recognisable error (or, with
/// `panic_on_load`, panics: the model must not even be asked for).
struct Host {
  prefix: String,
  spawner: Rc<dyn Spawner>,
  calls: RefCell<Vec<(String, String)>>,
  panic_on_load: bool,
  batch_policy: BatchPolicy,
}

impl NNEvaluatorHost for Host {
  fn backend_prefix(&self) -> String {
    self.prefix.clone()
  }
  fn global_initialize(&self) {}
  fn global_cleanup(&self) {}
  fn load_model(&self, nn_model_name: &str, nn_model_file: &str) -> Result<(Rc<dyn NeuralNet>, Vec<u8>), KataError> {
    assert!(!self.panic_on_load, "load_model({nn_model_name}, {nn_model_file}) called, but upstream does not open the model here");
    self.calls.borrow_mut().push((nn_model_name.to_string(), nn_model_file.to_string()));
    Err(KataError::IOError(format!("Could not open file {nn_model_file} - does not exist or invalid permissions?")))
  }
  fn spawner(&self) -> Rc<dyn Spawner> {
    self.spawner.clone()
  }
  fn get_batch_policy(&self, _: &mut ConfigParser) -> BatchPolicy {
    self.batch_policy
  }
  // dummybackend.cpp's: the number of distinct gpu indices.
  fn get_num_effective_devices(&self, _: &mut ConfigParser, gpu_idx_by_server_thread: &[i32]) -> i32 {
    let distinct: std::collections::BTreeSet<i32> = gpu_idx_by_server_thread.iter().copied().collect();
    std::cmp::max(1, distinct.len() as i32)
  }
}

type Evals = Result<Vec<Rc<NNEvaluator>>, KataError>;

const FC: MaxBatchSizeRequest = MaxBatchSizeRequest { kind: MaxBatchSizeRequest::FROM_CONCURRENCY, size: -1 };

fn init_evals(cfg_text: &str, files: &[&str], setup_for: setup_for_t, default_nnx_len: i32, max_batch_size_request: MaxBatchSizeRequest, panic_on_load: bool) -> (Evals, Vec<(String, String)>) {
  init_evals_full("dummybackend", BatchPolicy::Dynamic, 1, cfg_text, files, setup_for, default_nnx_len, max_batch_size_request, panic_on_load)
}

fn init_evals_prefix(
  prefix: &str,
  cfg_text: &str,
  files: &[&str],
  setup_for: setup_for_t,
  default_nnx_len: i32,
  max_batch_size_request: MaxBatchSizeRequest,
  panic_on_load: bool,
) -> (Evals, Vec<(String, String)>) {
  init_evals_full(prefix, BatchPolicy::Dynamic, 1, cfg_text, files, setup_for, default_nnx_len, max_batch_size_request, panic_on_load)
}

#[allow(clippy::too_many_arguments)]
fn init_evals_full(
  prefix: &str,
  batch_policy: BatchPolicy,
  expected_concurrent_evals: i32,
  cfg_text: &str,
  files: &[&str],
  setup_for: setup_for_t,
  default_nnx_len: i32,
  max_batch_size_request: MaxBatchSizeRequest,
  panic_on_load: bool,
) -> (Evals, Vec<(String, String)>) {
  common::init();
  let mut cfg = ConfigParser::new_in(cfg_text, false, true).unwrap();
  let mut seed_rand = Rand::new_seed("setup_host");
  let names: Vec<String> = (0..files.len()).map(|i| format!("model{i}")).collect();
  let files: Vec<String> = files.iter().map(|s| s.to_string()).collect();
  let mut pool = LocalPool::new();
  let host = Host { prefix: prefix.to_string(), spawner: Rc::new(pool.spawner()), calls: RefCell::new(Vec::new()), panic_on_load, batch_policy };
  let r = run_or_stall_panic(
    &mut pool,
    "initialize_nn_evaluators",
    Setup::initialize_nn_evaluators(&names, &files, &[], &mut cfg, &mut seed_rand, expected_concurrent_evals, default_nnx_len, 19, max_batch_size_request, false, false, setup_for, &host),
  );
  if let Ok(evals) = &r {
    for e in evals {
      e.kill_server_threads();
    }
    pool.run_until_stalled();
  }
  let calls = host.calls.borrow().clone();
  (r, calls)
}

fn err_text(r: Evals) -> String {
  match r {
    Ok(_) => panic!("expected an error"),
    Err(e) => e.what().to_string(),
  }
}

wasm_test! {
/// `debugSkipNeuralNet` (config key, or the "/dev/null" default): the model is never opened.
fn skip_neural_net_never_loads_the_model() {
  let mut cases = Cases::new("setup_host::skip_neural_net_never_loads_the_model");
  // The "/dev/null" default.
  let (r, calls) = init_evals("numSearchThreads = 1\n", &["/dev/null", "/dev/null"], setup_for_t::SETUP_FOR_GTP, 19, FC, true);
  let evals = r.unwrap();
  assert_eq!(evals.len(), 2);
  assert!(evals.iter().all(|e| e.is_neural_net_less()));
  assert!(calls.is_empty());
  cases.ran();
  // The config key, with a model file that does not exist.
  for sf in [setup_for_t::SETUP_FOR_GTP, setup_for_t::SETUP_FOR_ANALYSIS, setup_for_t::SETUP_FOR_MATCH, setup_for_t::SETUP_FOR_BENCHMARK] {
    let (r, calls) = init_evals("debugSkipNeuralNet = true\nnnMaxBatchSize = 4\n", &["no/such/net.bin.gz"], sf, 19, FC, true);
    let evals = r.unwrap_or_else(|e| panic!("{:?}: {}", sf, e.what()));
    assert!(evals[0].is_neural_net_less());
    assert_eq!(evals[0].get_model_file_name(), "no/such/net.bin.gz");
    assert!(calls.is_empty());
    cases.ran();
  }
  // Without the key and with a real file name the host is asked, once per model, in order, and its error is the result.
  let (r, calls) = init_evals("nnMaxBatchSize = 4\n", &["a.bin.gz", "b.bin.gz"], setup_for_t::SETUP_FOR_GTP, 19, FC, false);
  assert_eq!(err_text(r), "Could not open file a.bin.gz - does not exist or invalid permissions?");
  assert_eq!(calls, vec![("model0".to_string(), "a.bin.gz".to_string())]);
  cases.ran();
  // A config that turns skipping off for a "/dev/null" file asks for the model.
  let (r, calls) = init_evals("debugSkipNeuralNet = false\nnnMaxBatchSize = 4\n", &["/dev/null"], setup_for_t::SETUP_FOR_GTP, 19, FC, false);
  assert!(r.is_err());
  assert_eq!(calls, vec![("model0".to_string(), "/dev/null".to_string())]);
  cases.ran();
}
}

wasm_test! {
/// NNEvaluator's own argument checks come before it opens the model, so they win over a missing file.
fn nneval_argument_errors_come_before_the_model_open() {
  let mut cases = Cases::new("setup_host::nneval_argument_errors_come_before_the_model_open");
  // An explicit size is taken as is, and checked before the model is opened (v1.18.2: in Setup itself; NNEvaluator's
  // "maxBatchSize is negative" check cannot be reached from Setup any more).
  for (bad, text) in [(0, "Invalid explicit max batch size: 0"), (-3, "Invalid explicit max batch size: -3"), (65537, "Invalid explicit max batch size: 65537")] {
    let (r, calls) = init_evals("", &["a.bin.gz"], setup_for_t::SETUP_FOR_BENCHMARK, 19, MaxBatchSizeRequest::explicit_size(bad), false);
    assert_eq!(err_text(r), text);
    assert!(calls.is_empty(), "the model was opened before NNEvaluator's checks");
    cases.ran();
  }
  // The caller's defaultNNXLen is used when the config has no maxBoardXSizeForNNBuffer.
  let (r, calls) = init_evals("nnMaxBatchSize = 4\n", &["a.bin.gz"], setup_for_t::SETUP_FOR_GTP, 20, FC, false);
  assert_eq!(err_text(r), "Maximum supported nnEval board size is 19");
  assert!(calls.is_empty());
  cases.ran();
  // The same checks apply when the net is skipped.
  let (r, calls) = init_evals("debugSkipNeuralNet = true\n", &["a.bin.gz"], setup_for_t::SETUP_FOR_BENCHMARK, 19, MaxBatchSizeRequest::explicit_size(0), true);
  assert_eq!(err_text(r), "Invalid explicit max batch size: 0");
  assert!(calls.is_empty());
  cases.ran();
}
}

wasm_test! {
/// Setup::maybeWarnHumanSLParams: warns (and returns true) only when `humanSLProfile` is set and none of the models
/// takes SGF metadata; the text lists the models. (Not reachable from the setup oracle: it needs evaluators.)
fn maybe_warn_human_sl_params_texts() {
  use katago_core::search::searchparams::SearchParams;
  let mut cases = Cases::new("setup_host::maybe_warn_human_sl_params_texts");
  let (r, _) = init_evals("numSearchThreads = 1\n", &["/dev/null", "/dev/null"], setup_for_t::SETUP_FOR_GTP, 19, FC, true);
  let evals = r.unwrap();
  let (a, b) = (&evals[0], &evals[1]);
  assert!(!a.requires_sgf_metadata());
  let mut params = SearchParams::new();
  // No profile: nothing to warn about.
  let mut out = String::new();
  assert!(!Setup::maybe_warn_human_sl_params(&params, Some(a), Some(b), &mut out, false));
  assert_eq!(out, "");
  cases.ran();
  params.human_sl_profile.initialized = true;
  let mut out = String::new();
  assert!(Setup::maybe_warn_human_sl_params(&params, Some(a), None, &mut out, false));
  assert_eq!(out, "WARNING: humanSLProfile is specified as config param but model(s) don't use it: model0\n");
  cases.ran();
  let mut out = String::new();
  assert!(Setup::maybe_warn_human_sl_params(&params, Some(a), Some(b), &mut out, false));
  assert_eq!(out, "WARNING: humanSLProfile is specified as config param but model(s) don't use it: model0 and model1\n");
  cases.ran();
  let mut out = String::new();
  assert!(Setup::maybe_warn_human_sl_params(&params, None, Some(b), &mut out, false));
  assert_eq!(out, "WARNING: humanSLProfile is specified as config param but model(s) don't use it: model1\n");
  cases.ran();
  let mut out = String::new();
  assert!(Setup::maybe_warn_human_sl_params(&params, None, None, &mut out, false));
  assert_eq!(out, "WARNING: humanSLProfile is specified as config param but model(s) don't use it: \n");
  cases.ran();
}
}

wasm_test! {
/// The "burn" backend (katago-nn-burn) takes NCHW inputs like opencl / trt / metal, and reads its own prefixed keys;
/// "dummybackend" (NHWC) and the foreign backends' keys are ignored. (The setup oracle only builds upstream's
/// backends, so the port's own prefix is checked here.)
fn burn_backend_prefix_defaults_and_keys() {
  let mut cases = Cases::new("setup_host::burn_backend_prefix_defaults_and_keys");
  let nhwc = |prefix: &str, cfg: &str| -> bool {
    let (r, _) = init_evals_prefix(prefix, cfg, &["/dev/null"], setup_for_t::SETUP_FOR_GTP, 19, FC, true);
    r.unwrap()[0].inputs_use_nhwc
  };
  assert!(!nhwc("burn", ""), "burn takes NCHW inputs by default");
  assert!(nhwc("dummybackend", ""), "NHWC is upstream's default for the other backends");
  assert!(!nhwc("opencl", ""));
  cases.ran();
  assert!(nhwc("burn", "burnInputsUseNHWC = true\n"));
  assert!(nhwc("burn", "inputsUseNHWC0 = true\n"));
  assert!(!nhwc("burn", "openclInputsUseNHWC = true\ndummybackendInputsUseNHWC = true\n"), "other backends' keys are ignored");
  assert!(nhwc("dummybackend", "burnInputsUseNHWC = false\n"), "burn keys are ignored by the others");
  cases.ran();
}
}

wasm_test! {
/// v1.18.2's batch sizing (Setup::initializeNNEvaluators: MaxBatchSizeRequest x NeuralNet::BatchPolicy). The setup
/// oracle links upstream's Eigen backend (CpuLocal) only, so the Dynamic and FixedShape branches are checked here
/// against the formulas of setup.cpp.
fn max_batch_size_request_and_batch_policy() {
  let mut cases = Cases::new("setup_host::max_batch_size_request_and_batch_policy");
  let size = |policy: BatchPolicy, conc: i32, cfg: &str, sf: setup_for_t, req: MaxBatchSizeRequest| -> Result<i32, String> {
    let (r, _) = init_evals_full("dummybackend", policy, conc, cfg, &["/dev/null"], sf, 19, req, true);
    r.map(|e| e[0].get_max_batch_size()).map_err(|e| e.what().to_string())
  };
  let gtp = setup_for_t::SETUP_FOR_GTP;
  let fc = MaxBatchSizeRequest::from_concurrency();
  let strict = MaxBatchSizeRequest::from_concurrency_strict();
  let cfgreq = MaxBatchSizeRequest::require_from_config();
  // Dynamic: max(8, round up to 4), strict: max(1, c); a config value wins unless setupFor forces the request.
  assert_eq!(size(BatchPolicy::Dynamic, 1, "", gtp, fc), Ok(8));
  assert_eq!(size(BatchPolicy::Dynamic, 13, "", gtp, fc), Ok(16));
  assert_eq!(size(BatchPolicy::Dynamic, 0, "", gtp, strict), Ok(1));
  assert_eq!(size(BatchPolicy::Dynamic, 13, "", gtp, strict), Ok(13));
  assert_eq!(size(BatchPolicy::Dynamic, 13, "nnMaxBatchSize = 5\n", gtp, fc), Ok(5));
  for forced in [setup_for_t::SETUP_FOR_BENCHMARK, setup_for_t::SETUP_FOR_BENCHMARKNN, setup_for_t::SETUP_FOR_DISTRIBUTED] {
    assert_eq!(size(BatchPolicy::Dynamic, 13, "nnMaxBatchSize = 5\n", forced, fc), Ok(16), "{forced:?}");
  }
  cases.ran();
  // FixedShape: half the per-device concurrency, rounded up, at least 2.
  assert_eq!(size(BatchPolicy::FixedShape, 1, "", gtp, fc), Ok(2));
  assert_eq!(size(BatchPolicy::FixedShape, 13, "", gtp, fc), Ok(7));
  assert_eq!(size(BatchPolicy::FixedShape, 13, "", gtp, strict), Ok(7));
  assert_eq!(size(BatchPolicy::FixedShape, 13, "numNNServerThreadsPerModel = 2\ngpuToUseThread0 = 0\ngpuToUseThread1 = 1\n", gtp, fc), Ok(4));
  assert_eq!(Setup::compute_fixed_shape_max_batch_size(16, 0), 8);
  assert_eq!(Setup::compute_fixed_shape_max_batch_size(17, 2), 5);
  cases.ran();
  // CpuLocal: 2 whatever is asked, except an explicit size.
  assert_eq!(size(BatchPolicy::CpuLocal, 64, "nnMaxBatchSize = 5\n", gtp, fc), Ok(2));
  assert_eq!(size(BatchPolicy::CpuLocal, 64, "", gtp, cfgreq), Ok(2));
  assert_eq!(size(BatchPolicy::CpuLocal, 64, "nnMaxBatchSize = 5\n", gtp, MaxBatchSizeRequest::explicit_size(9)), Ok(9));
  cases.ran();
  // REQUIRE_FROM_CONFIG: the key is required (unless CpuLocal).
  assert_eq!(size(BatchPolicy::Dynamic, 64, "nnMaxBatchSize = 5\n", gtp, cfgreq), Ok(5));
  assert!(size(BatchPolicy::Dynamic, 64, "", gtp, cfgreq).unwrap_err().contains("nnMaxBatchSize"));
  cases.ran();
}
}
