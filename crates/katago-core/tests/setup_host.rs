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
use katago_core::neuralnet::nninterface::NeuralNet;
use katago_core::port::executor::Spawner;
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
}

type Evals = Result<Vec<Rc<NNEvaluator>>, KataError>;

fn init_evals(cfg_text: &str, files: &[&str], setup_for: setup_for_t, default_nnx_len: i32, default_max_batch_size: i32, panic_on_load: bool) -> (Evals, Vec<(String, String)>) {
  init_evals_prefix("dummybackend", cfg_text, files, setup_for, default_nnx_len, default_max_batch_size, panic_on_load)
}

fn init_evals_prefix(
  prefix: &str,
  cfg_text: &str,
  files: &[&str],
  setup_for: setup_for_t,
  default_nnx_len: i32,
  default_max_batch_size: i32,
  panic_on_load: bool,
) -> (Evals, Vec<(String, String)>) {
  common::init();
  let mut cfg = ConfigParser::new_in(cfg_text, false, true).unwrap();
  let mut seed_rand = Rand::new_seed("setup_host");
  let names: Vec<String> = (0..files.len()).map(|i| format!("model{i}")).collect();
  let files: Vec<String> = files.iter().map(|s| s.to_string()).collect();
  let mut pool = LocalPool::new();
  let host = Host { prefix: prefix.to_string(), spawner: Rc::new(pool.spawner()), calls: RefCell::new(Vec::new()), panic_on_load };
  let r = run_or_stall_panic(
    &mut pool,
    "initialize_nn_evaluators",
    Setup::initialize_nn_evaluators(&names, &files, &[], &mut cfg, &mut seed_rand, 1, default_nnx_len, 19, default_max_batch_size, false, false, setup_for, &host),
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
  let (r, calls) = init_evals("numSearchThreads = 1\n", &["/dev/null", "/dev/null"], setup_for_t::SETUP_FOR_GTP, 19, 4, true);
  let evals = r.unwrap();
  assert_eq!(evals.len(), 2);
  assert!(evals.iter().all(|e| e.is_neural_net_less()));
  assert!(calls.is_empty());
  cases.ran();
  // The config key, with a model file that does not exist.
  for sf in [setup_for_t::SETUP_FOR_GTP, setup_for_t::SETUP_FOR_ANALYSIS, setup_for_t::SETUP_FOR_MATCH, setup_for_t::SETUP_FOR_BENCHMARK] {
    let (r, calls) = init_evals("debugSkipNeuralNet = true\nnnMaxBatchSize = 4\n", &["no/such/net.bin.gz"], sf, 19, 4, true);
    let evals = r.unwrap_or_else(|e| panic!("{:?}: {}", sf, e.what()));
    assert!(evals[0].is_neural_net_less());
    assert_eq!(evals[0].get_model_file_name(), "no/such/net.bin.gz");
    assert!(calls.is_empty());
    cases.ran();
  }
  // Without the key and with a real file name the host is asked, once per model, in order, and its error is the result.
  let (r, calls) = init_evals("nnMaxBatchSize = 4\n", &["a.bin.gz", "b.bin.gz"], setup_for_t::SETUP_FOR_GTP, 19, 4, false);
  assert_eq!(err_text(r), "Could not open file a.bin.gz - does not exist or invalid permissions?");
  assert_eq!(calls, vec![("model0".to_string(), "a.bin.gz".to_string())]);
  cases.ran();
  // A config that turns skipping off for a "/dev/null" file asks for the model.
  let (r, calls) = init_evals("debugSkipNeuralNet = false\nnnMaxBatchSize = 4\n", &["/dev/null"], setup_for_t::SETUP_FOR_GTP, 19, 4, false);
  assert!(r.is_err());
  assert_eq!(calls, vec![("model0".to_string(), "/dev/null".to_string())]);
  cases.ran();
}
}

wasm_test! {
/// NNEvaluator's own argument checks come before it opens the model, so they win over a missing file.
fn nneval_argument_errors_come_before_the_model_open() {
  let mut cases = Cases::new("setup_host::nneval_argument_errors_come_before_the_model_open");
  // BENCHMARK takes the caller's defaultMaxBatchSize as is.
  for (bad, text) in [(0, "maxBatchSize is negative: 0"), (-3, "maxBatchSize is negative: -3")] {
    let (r, calls) = init_evals("", &["a.bin.gz"], setup_for_t::SETUP_FOR_BENCHMARK, 19, bad, false);
    assert_eq!(err_text(r), text);
    assert!(calls.is_empty(), "the model was opened before NNEvaluator's checks");
    cases.ran();
  }
  // The caller's defaultNNXLen is used when the config has no maxBoardXSizeForNNBuffer.
  let (r, calls) = init_evals("nnMaxBatchSize = 4\n", &["a.bin.gz"], setup_for_t::SETUP_FOR_GTP, 20, 4, false);
  assert_eq!(err_text(r), "Maximum supported nnEval board size is 19");
  assert!(calls.is_empty());
  cases.ran();
  // The same checks apply when the net is skipped.
  let (r, calls) = init_evals("debugSkipNeuralNet = true\n", &["a.bin.gz"], setup_for_t::SETUP_FOR_BENCHMARK, 19, 0, true);
  assert_eq!(err_text(r), "maxBatchSize is negative: 0");
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
  let (r, _) = init_evals("numSearchThreads = 1\n", &["/dev/null", "/dev/null"], setup_for_t::SETUP_FOR_GTP, 19, 4, true);
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
    let (r, _) = init_evals_prefix(prefix, cfg, &["/dev/null"], setup_for_t::SETUP_FOR_GTP, 19, 4, true);
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
