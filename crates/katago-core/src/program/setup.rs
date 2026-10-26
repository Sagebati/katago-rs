// Ported from KataGo cpp/program/setup.cpp, cpp/program/setup.h @ fd0723fdbc0e9d82cf269c9630af8c27c57c07c4
//
// Divergences (port-manifest.toml notes):
// - `Logger&` / `Logger*` parameters dropped (log crate, PORTING.md §3); `logger.write` -> `log::info!`.
// - initializeNNEvaluator(s): the compile-time backend (`#if defined(USE_*_BACKEND)`) and the model file
//   come from an injected `NNEvaluatorHost` (DECISIONS.md D6: backend trait objects, models as bytes;
//   core does no file IO). Async, because NNEvaluator::spawnServerThreads is (PORTING.md §11, D8). The
//   non-Eigen branches (`#ifndef USE_EIGEN_BACKEND`) are the ported ones: the burn backend is not Eigen.
// - Not ported: computeDefaultEigenBackendThreads (Eigen backend only, std::thread::hardware_concurrency),
//   saveAutoPatternBonusData and loadAndPruneAutoPatternBonusTables (file IO and SGF/pos files).
//   loadAvoidSgfPatternBonusTables reads every key as upstream; its PatternBonusTable::avoidRepeatedSgfMoves call
//   (SGF loading) is a TODO(port) IO/M8 gap: a fully configured avoid-SGF set is an error.

use std::collections::BTreeSet;
use std::rc::Rc;

use crate::core::config_parser::ConfigParser;
use crate::core::commontypes::enabled_t;
use crate::core::global::Global;
use crate::core::global::KataError;
use crate::core::rand::Rand;
use crate::game::board::Board;
use crate::game::board::Player;
use crate::game::board::PlayerIO;
use crate::game::board::C_EMPTY;
use crate::game::board::P_BLACK;
use crate::game::board::P_WHITE;
use crate::game::rules::Rules;
use crate::neuralnet::desc::ModelDesc;
use crate::neuralnet::nneval::NNEvaluator;
use crate::neuralnet::nneval::NNResultBuf;
use crate::neuralnet::nninputs::NNPos;
use crate::neuralnet::nninputs::SymmetryHelpers;
use crate::neuralnet::nninputs::NNOutput;
use crate::neuralnet::nninterface::*;
use crate::neuralnet::sgfmetadata::SGFMetadata;
use crate::port::cpp::cpp_min;
use crate::port::executor::LocalBoxFuture;
use crate::port::executor::Spawner;
use crate::search::patternbonustable::PatternBonusTable;
use crate::search::searchparams::SearchParams;
use crate::testAssert;

/// upstream: Setup::setup_for_t (setup.h)
#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum setup_for_t {
  SETUP_FOR_GTP,
  SETUP_FOR_BENCHMARK,
  SETUP_FOR_MATCH,
  SETUP_FOR_ANALYSIS,
  SETUP_FOR_OTHER,
  SETUP_FOR_DISTRIBUTED,
  // Pure NN forward benchmark - does not spawn NN server threads, the benchmark drives
  // compute handles itself. See NNEvaluator::benchmarkPureForward.
  SETUP_FOR_BENCHMARKNN,
}

/// upstream: Setup::MaxBatchSizeRequest::Kind (setup.h)
#[allow(clippy::upper_case_acronyms)]
#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
  // Size it for expectedConcurrentEvals, with room to spare: this is a search whose threads all
  // want evaluations at once, and a batch a little larger than strictly needed costs nothing on
  // a GPU. The overwhelmingly common case.
  FROM_CONCURRENCY,
  // Size it for exactly expectedConcurrentEvals, with no rounding up or floor. For callers
  // that already know their true concurrency and mean it exactly, rather than a search whose
  // thread count is only an upper bound on how many evals are really in flight.
  FROM_CONCURRENCY_STRICT,
  // No default at all: nnMaxBatchSize must come from the config, and it is an error if absent.
  // For the long-running multi-game tools whose configs are expected to set it.
  REQUIRE_FROM_CONFIG,
  // Use exactly this batch size, overriding every policy including CpuLocal's fixed size and
  // any config value. For benchmarking and testing tools whose number is already a batch size
  // rather than a concurrency, so no policy transform may be applied to it.
  EXPLICIT_SIZE,
}

/// upstream: Setup::MaxBatchSizeRequest (setup.h)
// How a caller wants nnMaxBatchSize decided. The caller says what it knows about the work it
// is about to do, and the backend's NeuralNet::BatchPolicy decides how that turns into a batch
// size, since the right answer differs between GPU backends, fixed-shape backends and CPU
// backends.
// A user-specified nnMaxBatchSize wins over the concurrency-based kinds, except where setupFor
// forces the caller's value (benchmarks sweeping thread counts, and distributed). It never
// applies under EXPLICIT_SIZE, nor for CpuLocal backends, which ignore it entirely.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MaxBatchSizeRequest {
  pub kind: Kind,
  pub size: i32, // Only meaningful for EXPLICIT_SIZE
}

impl MaxBatchSizeRequest {
  /// upstream: Setup::MaxBatchSizeRequest::FROM_CONCURRENCY .. EXPLICIT_SIZE (setup.h), as `MaxBatchSizeRequest::X` reads them.
  pub const FROM_CONCURRENCY: Kind = Kind::FROM_CONCURRENCY;
  pub const FROM_CONCURRENCY_STRICT: Kind = Kind::FROM_CONCURRENCY_STRICT;
  pub const REQUIRE_FROM_CONFIG: Kind = Kind::REQUIRE_FROM_CONFIG;
  pub const EXPLICIT_SIZE: Kind = Kind::EXPLICIT_SIZE;

  /// upstream: Setup::MaxBatchSizeRequest::fromConcurrency (setup.h)
  pub fn from_concurrency() -> MaxBatchSizeRequest {
    MaxBatchSizeRequest { kind: Kind::FROM_CONCURRENCY, size: -1 }
  }
  /// upstream: Setup::MaxBatchSizeRequest::fromConcurrencyStrict (setup.h)
  pub fn from_concurrency_strict() -> MaxBatchSizeRequest {
    MaxBatchSizeRequest { kind: Kind::FROM_CONCURRENCY_STRICT, size: -1 }
  }
  /// upstream: Setup::MaxBatchSizeRequest::requireFromConfig (setup.h)
  pub fn require_from_config() -> MaxBatchSizeRequest {
    MaxBatchSizeRequest { kind: Kind::REQUIRE_FROM_CONFIG, size: -1 }
  }
  /// upstream: Setup::MaxBatchSizeRequest::explicitSize (setup.h)
  pub fn explicit_size(size: i32) -> MaxBatchSizeRequest {
    MaxBatchSizeRequest { kind: Kind::EXPLICIT_SIZE, size }
  }
}

// The enum values as names in scope, as inside `namespace Setup`.
use setup_for_t::SETUP_FOR_ANALYSIS;
use setup_for_t::SETUP_FOR_BENCHMARK;
use setup_for_t::SETUP_FOR_BENCHMARKNN;
use setup_for_t::SETUP_FOR_DISTRIBUTED;
use setup_for_t::SETUP_FOR_GTP;
use setup_for_t::SETUP_FOR_MATCH;
use setup_for_t::SETUP_FOR_OTHER;

/// What upstream's Setup::initializeNNEvaluators gets from the build and the file system.
/// PORT NOTE (DECISIONS.md D6, PORTING.md §10): upstream picks the backend at compile time
/// (`USE_CUDA_BACKEND` ... -> `backendPrefix`, `NeuralNet::*`) and NNEvaluator's constructor opens
/// `nnModelFile`. Here the host supplies both, and the executor the NN server tasks run on.
pub trait NNEvaluatorHost {
  /// The `backendPrefix` of the backend ("cuda", "trt", "metal", "opencl", "eigen", "dummybackend"
  /// upstream; "burn" for katago-nn-burn), used to pick backend-specific config keys.
  fn backend_prefix(&self) -> String;
  /// `NeuralNet::globalInitialize()` (Setup::initializeSession).
  fn global_initialize(&self);
  /// `NeuralNet::globalCleanup()` (the end of the commands that called initializeSession).
  fn global_cleanup(&self);
  /// The backend (`NeuralNet::` functions) for this model, and the contents of `nn_model_file`
  /// (what NNEvaluator's constructor would read). Asked for only when the net is used: with
  /// debugSkipNeuralNet (the config key, or the "/dev/null" default) upstream never opens the model
  /// file, so this is not called; and, as upstream's constructor, only after NNEvaluator's own
  /// argument checks (board size, batch size) passed.
  fn load_model(&self, nn_model_name: &str, nn_model_file: &str) -> Result<(Rc<dyn NeuralNet>, Vec<u8>), KataError>;
  /// The host's local executor (NNEvaluator::spawnServerThreads spawns the server tasks on it).
  fn spawner(&self) -> Rc<dyn Spawner>;
  /// `NeuralNet::getBatchPolicy(cfg)` of the linked backend, which Setup::initializeNNEvaluators asks before
  /// any model is loaded (also with debugSkipNeuralNet). PORT NOTE (D6): a static function of the backend linked
  /// at build time upstream; here the host's backend answers.
  fn get_batch_policy(&self, cfg: &mut ConfigParser) -> BatchPolicy;
  /// `NeuralNet::getNumEffectiveDevices(cfg, gpuIdxByServerThread)` of the linked backend (FixedShape sizing in
  /// Setup::initializeNNEvaluators). PORT NOTE (D6): as `get_batch_policy`.
  fn get_num_effective_devices(&self, cfg: &mut ConfigParser, gpu_idx_by_server_thread: &[i32]) -> i32;
}

/// The `NeuralNet` of an evaluator with debugSkipNeuralNet: NNEvaluator never calls its backend then
/// (no model is loaded, no compute context or handle is made, nothing is evaluated), so there is
/// nothing to run and every function that would need a model fails.
/// PORT NOTE: port-only (D6). NNEvaluator::new takes the backend as a parameter; upstream links one
/// at build time and skips the net without ever touching it.
struct SkippedNeuralNet;

impl SkippedNeuralNet {
  fn skipped<T>() -> T {
    panic!("debugSkipNeuralNet: no neural net backend is used")
  }
}

impl NeuralNet for SkippedNeuralNet {
  fn global_initialize(&self) {}
  fn global_cleanup(&self) {}
  fn print_devices(&self) {}
  // PORT NOTE: NNEvaluator's constructor asks the linked backend for getNumEffectiveDevices even when it skips
  // the net; these four follow dummybackend.cpp, upstream's NN-less backend.
  fn get_runtime_backend_detail(&self, _: &mut ConfigParser) -> String {
    String::new()
  }
  fn get_batch_policy(&self, _: &mut ConfigParser) -> BatchPolicy {
    BatchPolicy::Dynamic
  }
  fn get_num_effective_devices(&self, _: &mut ConfigParser, gpu_idx_by_server_thread: &[i32]) -> i32 {
    let distinct_devices: BTreeSet<i32> = gpu_idx_by_server_thread.iter().copied().collect();
    std::cmp::max(1, distinct_devices.len() as i32)
  }
  fn set_is_warmup(&self, _: &dyn ComputeHandle, _: bool) -> bool {
    false
  }
  fn load_model_file(&self, _: &str, _: Vec<u8>, _: &str) -> Result<Box<dyn LoadedModel>, KataError> {
    SkippedNeuralNet::skipped()
  }
  fn get_model_desc<'a>(&self, _: &'a dyn LoadedModel) -> &'a ModelDesc {
    SkippedNeuralNet::skipped()
  }
  fn create_compute_context(&self, _: &[i32], _: i32, _: i32, _: &str, _: enabled_t, _: &dyn LoadedModel, _: &mut ConfigParser) -> Result<Box<dyn ComputeContext>, KataError> {
    SkippedNeuralNet::skipped()
  }
  fn create_compute_handle(&self, _: &dyn ComputeContext, _: &dyn LoadedModel, _: i32, _: bool, _: bool, _: i32, _: i32) -> Result<Box<dyn ComputeHandle>, KataError> {
    SkippedNeuralNet::skipped()
  }
  fn is_using_fp16(&self, _: &dyn ComputeHandle) -> bool {
    SkippedNeuralNet::skipped()
  }
  fn create_input_buffers(&self, _: &dyn LoadedModel, _: i32, _: i32, _: i32) -> Box<dyn InputBuffers> {
    SkippedNeuralNet::skipped()
  }
  fn get_output<'a>(
    &'a self,
    _: &'a mut dyn ComputeHandle,
    _: &'a mut dyn InputBuffers,
    _: i32,
    _: &'a [Rc<NNResultBuf>],
    _: &'a mut [NNOutput],
  ) -> LocalBoxFuture<'a, Result<(), KataError>> {
    SkippedNeuralNet::skipped()
  }
}

/// The checks at the top of upstream's NNEvaluator constructor, which run before it opens the model
/// file (nneval.cpp), repeated here because the port's model open (`NNEvaluatorHost::load_model`)
/// happens in Setup, before `NNEvaluator::new`. `NNEvaluator::new` checks them again.
fn check_nn_evaluator_args(max_batch_size: i32, nn_x_len: i32, nn_y_len: i32, num_threads: i32, gpu_idx_by_server_thread: &[i32]) -> Result<(), KataError> {
  if nn_x_len > NNPos::MAX_BOARD_LEN {
    return Err(KataError::StringError("Maximum supported nnEval board size is ".to_string() + &Global::int_to_string(NNPos::MAX_BOARD_LEN)));
  }
  if nn_y_len > NNPos::MAX_BOARD_LEN {
    return Err(KataError::StringError("Maximum supported nnEval board size is ".to_string() + &Global::int_to_string(NNPos::MAX_BOARD_LEN)));
  }
  if max_batch_size <= 0 {
    return Err(KataError::StringError("maxBatchSize is negative: ".to_string() + &Global::int_to_string(max_batch_size)));
  }
  if gpu_idx_by_server_thread.len() as u64 != num_threads as u64 {
    return Err(KataError::StringError("gpuIdxByServerThread.size() != numThreads".to_string()));
  }
  Ok(())
}

/// upstream: namespace Setup (setup.h)
//Some bits of initialization and main function logic shared between various programs
pub struct Setup;

impl Setup {
  /// upstream: Setup::SETUP_FOR_GTP .. SETUP_FOR_BENCHMARKNN (setup.h), as `Setup::X` call sites read them.
  pub const SETUP_FOR_GTP: setup_for_t = setup_for_t::SETUP_FOR_GTP;
  pub const SETUP_FOR_BENCHMARK: setup_for_t = setup_for_t::SETUP_FOR_BENCHMARK;
  pub const SETUP_FOR_MATCH: setup_for_t = setup_for_t::SETUP_FOR_MATCH;
  pub const SETUP_FOR_ANALYSIS: setup_for_t = setup_for_t::SETUP_FOR_ANALYSIS;
  pub const SETUP_FOR_OTHER: setup_for_t = setup_for_t::SETUP_FOR_OTHER;
  pub const SETUP_FOR_DISTRIBUTED: setup_for_t = setup_for_t::SETUP_FOR_DISTRIBUTED;
  pub const SETUP_FOR_BENCHMARKNN: setup_for_t = setup_for_t::SETUP_FOR_BENCHMARKNN;

  /// upstream: Setup::MAX_BOT_PARAMS_FROM_CFG, DEFAULT_ANALYSIS_WIDE_ROOT_NOISE, DEFAULT_ANALYSIS_IGNORE_PRE_ROOT_HISTORY (setup.h)
  pub const MAX_BOT_PARAMS_FROM_CFG: i32 = 4096;

  pub const DEFAULT_ANALYSIS_WIDE_ROOT_NOISE: f64 = 0.04;
  pub const DEFAULT_ANALYSIS_IGNORE_PRE_ROOT_HISTORY: bool = true;

  /// upstream: Setup::initializeSession (setup.cpp)
  /// PORT NOTE: `NeuralNet::globalInitialize()` is the injected host's (D6).
  pub fn initialize_session(cfg: &mut ConfigParser, host: &dyn NNEvaluatorHost) {
    let _ = cfg;
    host.global_initialize();
  }

  /// upstream: Setup::getBackendPrefixes (setup.cpp)
  //Return config prefixes for GPU backends.
  /// PORT NOTE: "burn" (katago-nn-burn) added, so that its keys are not marked used when another
  /// backend runs and the upstream prefixes are marked used when burn runs.
  pub fn get_backend_prefixes() -> Vec<String> {
    let mut prefixes: Vec<String> = Vec::new();
    prefixes.push("cuda".to_string());
    prefixes.push("trt".to_string());
    prefixes.push("metal".to_string());
    prefixes.push("opencl".to_string());
    prefixes.push("rocm".to_string());
    prefixes.push("eigen".to_string());
    prefixes.push("onnx".to_string());
    prefixes.push("dummybackend".to_string());
    prefixes.push("burn".to_string());
    prefixes
  }

  /// upstream: Setup::initializeNNEvaluator (setup.cpp)
  /// PORT NOTE: `Logger&` dropped; `host` added (D6); async (spawnServerThreads, §11); throws -> Result.
  pub async fn initialize_nn_evaluator(
    nn_model_name: &str,
    nn_model_file: &str,
    expected_sha256: &str,
    cfg: &mut ConfigParser,
    seed_rand: &mut Rand,
    expected_concurrent_evals: i32,
    default_nnx_len: i32,
    default_nny_len: i32,
    max_batch_size_request: MaxBatchSizeRequest,
    default_require_exact_nn_len: bool,
    disable_fp16: bool,
    setup_for: setup_for_t,
    host: &dyn NNEvaluatorHost,
  ) -> Result<Rc<NNEvaluator>, KataError> {
    let mut nn_evals: Vec<Rc<NNEvaluator>> = Setup::initialize_nn_evaluators(
      &[nn_model_name.to_string()],
      &[nn_model_file.to_string()],
      &[expected_sha256.to_string()],
      cfg,
      seed_rand,
      expected_concurrent_evals,
      default_nnx_len,
      default_nny_len,
      max_batch_size_request,
      default_require_exact_nn_len,
      disable_fp16,
      setup_for,
      host,
    )
    .await?;
    testAssert!(nn_evals.len() == 1);
    Ok(nn_evals.remove(0))
  }

  /// upstream: Setup::initializeNNEvaluators (setup.cpp)
  /// PORT NOTE: `Logger&` dropped; `host` added: it gives `backendPrefix` (upstream: `#if defined(USE_*_BACKEND)`),
  /// the backend and the model bytes for `NNEvaluator`'s new trailing parameters, and the spawner (D6). The
  /// `#ifndef USE_EIGEN_BACKEND` branches are ported (burn is not Eigen); the Eigen ones are not. Async
  /// (spawnServerThreads, §11); throws -> Result.
  pub async fn initialize_nn_evaluators(
    nn_model_names: &[String],
    nn_model_files: &[String],
    expected_sha256s: &[String],
    cfg: &mut ConfigParser,
    seed_rand: &mut Rand,
    expected_concurrent_evals: i32,
    default_nnx_len: i32,
    default_nny_len: i32,
    max_batch_size_request: MaxBatchSizeRequest,
    default_require_exact_nn_len: bool,
    disable_fp16: bool,
    setup_for: setup_for_t,
    host: &dyn NNEvaluatorHost,
  ) -> Result<Vec<Rc<NNEvaluator>>, KataError> {
    let mut nn_evals: Vec<Rc<NNEvaluator>> = Vec::new();
    testAssert!(nn_model_names.len() == nn_model_files.len());
    testAssert!(expected_sha256s.is_empty() || expected_sha256s.len() == nn_model_files.len());

    // PORT NOTE: `#if defined(USE_CUDA_BACKEND) ... string backendPrefix = "cuda"; ...` -> the host's.
    let backend_prefix: String = host.backend_prefix();

    // PORT NOTE: `#if defined(USE_ONNX_BACKEND)` -> the host's backend prefix is "onnx" (never true for the
    // port's own backends, kept for fidelity).
    if backend_prefix == "onnx" {
      // Distributed selfplay (contribute) uploads training data, which must never contain
      // FP16-overflow NaN rows, so always apply the scale8 workaround regardless of onnxSkipScale8.
      if setup_for == SETUP_FOR_DISTRIBUTED && cfg.contains("onnxSkipScale8") && cfg.get_bool("onnxSkipScale8")? {
        cfg.override_key("onnxSkipScale8", "false");
        log::info!(
          "{}",
          "WARNING: onnxSkipScale8 = true is not allowed for contribute (distributed selfplay); \
forcing it to false so FP16-overflow NaNs cannot poison contributed training data."
        );
      }
    }

    //Automatically flag keys that are for other backends as used so that we don't warn about unused keys
    //for those options
    for prefix in Setup::get_backend_prefixes() {
      if prefix != backend_prefix {
        cfg.mark_all_keys_used_with_prefix(&prefix);
      }
    }

    for i in 0..nn_model_files.len() {
      let idx_str: String = Global::uint64_to_string(i as u64);
      let nn_model_name: &String = &nn_model_names[i];
      let nn_model_file: &String = &nn_model_files[i];
      let expected_sha256: String = if !expected_sha256s.is_empty() { expected_sha256s[i].clone() } else { String::new() };

      let debug_skip_neural_net_default: bool = nn_model_file == "/dev/null";
      let debug_skip_neural_net: bool = if setup_for == SETUP_FOR_DISTRIBUTED {
        debug_skip_neural_net_default
      } else if cfg.contains("debugSkipNeuralNet") {
        cfg.get_bool("debugSkipNeuralNet")?
      } else {
        debug_skip_neural_net_default
      };

      let mut nn_x_len: i32 = std::cmp::max(default_nnx_len, 2);
      let mut nn_y_len: i32 = std::cmp::max(default_nny_len, 2);
      if setup_for != SETUP_FOR_DISTRIBUTED {
        if cfg.contains(&format!("maxBoardXSizeForNNBuffer{idx_str}")) {
          nn_x_len = cfg.get_int(&format!("maxBoardXSizeForNNBuffer{idx_str}"), 2, NNPos::MAX_BOARD_LEN)?;
        } else if cfg.contains("maxBoardXSizeForNNBuffer") {
          nn_x_len = cfg.get_int("maxBoardXSizeForNNBuffer", 2, NNPos::MAX_BOARD_LEN)?;
        } else if cfg.contains(&format!("maxBoardSizeForNNBuffer{idx_str}")) {
          nn_x_len = cfg.get_int(&format!("maxBoardSizeForNNBuffer{idx_str}"), 2, NNPos::MAX_BOARD_LEN)?;
        } else if cfg.contains("maxBoardSizeForNNBuffer") {
          nn_x_len = cfg.get_int("maxBoardSizeForNNBuffer", 2, NNPos::MAX_BOARD_LEN)?;
        }

        if cfg.contains(&format!("maxBoardYSizeForNNBuffer{idx_str}")) {
          nn_y_len = cfg.get_int(&format!("maxBoardYSizeForNNBuffer{idx_str}"), 2, NNPos::MAX_BOARD_LEN)?;
        } else if cfg.contains("maxBoardYSizeForNNBuffer") {
          nn_y_len = cfg.get_int("maxBoardYSizeForNNBuffer", 2, NNPos::MAX_BOARD_LEN)?;
        } else if cfg.contains(&format!("maxBoardSizeForNNBuffer{idx_str}")) {
          nn_y_len = cfg.get_int(&format!("maxBoardSizeForNNBuffer{idx_str}"), 2, NNPos::MAX_BOARD_LEN)?;
        } else if cfg.contains("maxBoardSizeForNNBuffer") {
          nn_y_len = cfg.get_int("maxBoardSizeForNNBuffer", 2, NNPos::MAX_BOARD_LEN)?;
        }
      }

      let mut require_exact_nn_len: bool = default_require_exact_nn_len;
      if setup_for != SETUP_FOR_DISTRIBUTED {
        if cfg.contains(&format!("requireMaxBoardSize{idx_str}")) {
          require_exact_nn_len = cfg.get_bool(&format!("requireMaxBoardSize{idx_str}"))?;
        } else if cfg.contains("requireMaxBoardSize") {
          require_exact_nn_len = cfg.get_bool("requireMaxBoardSize")?;
        }
      }

      //ROCm defaults to NHWC inputs like CUDA: its compute layout is NHWC in the default FP16 path
      //(transformers always, convnets on the archs where NHWC is faster), and unlike cuDNN, MIOpen
      //cannot consume mismatched input/compute layouts for free - it costs a device transpose.
      // PORT NOTE: "burn" added to the NCHW backends (the burn graph is NCHW, katago-nn-burn backend.rs).
      let mut inputs_use_nhwc: bool =
        if backend_prefix == "opencl" || backend_prefix == "trt" || backend_prefix == "metal" || backend_prefix == "onnx" || backend_prefix == "burn" { false } else { true };
      if cfg.contains(&format!("{backend_prefix}InputsUseNHWC{idx_str}")) {
        inputs_use_nhwc = cfg.get_bool(&format!("{backend_prefix}InputsUseNHWC{idx_str}"))?;
      } else if cfg.contains(&format!("inputsUseNHWC{idx_str}")) {
        inputs_use_nhwc = cfg.get_bool(&format!("inputsUseNHWC{idx_str}"))?;
      } else if cfg.contains(&format!("{backend_prefix}InputsUseNHWC")) {
        inputs_use_nhwc = cfg.get_bool(&format!("{backend_prefix}InputsUseNHWC"))?;
      } else if cfg.contains("inputsUseNHWC") {
        inputs_use_nhwc = cfg.get_bool("inputsUseNHWC")?;
      }

      let nn_randomize: bool = if setup_for == SETUP_FOR_DISTRIBUTED {
        true
      } else if cfg.contains("nnRandomize") {
        cfg.get_bool("nnRandomize")?
      } else {
        true
      };

      let nn_rand_seed: String;
      if setup_for == SETUP_FOR_DISTRIBUTED {
        nn_rand_seed = Global::uint64_to_string(seed_rand.next_uint64());
      } else if cfg.contains(&format!("nnRandSeed{idx_str}")) {
        nn_rand_seed = cfg.get_string(&format!("nnRandSeed{idx_str}"))?;
      } else if cfg.contains("nnRandSeed") {
        nn_rand_seed = cfg.get_string("nnRandSeed")?;
      } else {
        nn_rand_seed = Global::uint64_to_string(seed_rand.next_uint64());
      }
      log::info!("{}", "nnRandSeed".to_string() + &idx_str + " = " + &nn_rand_seed);

      // PORT NOTE: the `#ifndef USE_EIGEN_BACKEND` branch.
      cfg.mark_all_keys_used_with_prefix("numEigenThreadsPerModel");
      let num_nn_server_threads_per_model: i32 =
        if cfg.contains("numNNServerThreadsPerModel") { cfg.get_int("numNNServerThreadsPerModel", 1, 1024)? } else { 1 };

      let mut gpu_idx_by_server_thread: Vec<i32> = Vec::new();
      for j in 0..num_nn_server_threads_per_model {
        let thread_idx_str: String = Global::int_to_string(j);
        if cfg.contains(&format!("{backend_prefix}DeviceToUseModel{idx_str}Thread{thread_idx_str}")) {
          gpu_idx_by_server_thread.push(cfg.get_int(&format!("{backend_prefix}DeviceToUseModel{idx_str}Thread{thread_idx_str}"), 0, 1023)?);
        } else if cfg.contains(&format!("{backend_prefix}GpuToUseModel{idx_str}Thread{thread_idx_str}")) {
          gpu_idx_by_server_thread.push(cfg.get_int(&format!("{backend_prefix}GpuToUseModel{idx_str}Thread{thread_idx_str}"), 0, 1023)?);
        } else if cfg.contains(&format!("deviceToUseModel{idx_str}Thread{thread_idx_str}")) {
          gpu_idx_by_server_thread.push(cfg.get_int(&format!("deviceToUseModel{idx_str}Thread{thread_idx_str}"), 0, 1023)?);
        } else if cfg.contains(&format!("gpuToUseModel{idx_str}Thread{thread_idx_str}")) {
          gpu_idx_by_server_thread.push(cfg.get_int(&format!("gpuToUseModel{idx_str}Thread{thread_idx_str}"), 0, 1023)?);
        } else if cfg.contains(&format!("{backend_prefix}DeviceToUseModel{idx_str}")) {
          gpu_idx_by_server_thread.push(cfg.get_int(&format!("{backend_prefix}DeviceToUseModel{idx_str}"), 0, 1023)?);
        } else if cfg.contains(&format!("{backend_prefix}GpuToUseModel{idx_str}")) {
          gpu_idx_by_server_thread.push(cfg.get_int(&format!("{backend_prefix}GpuToUseModel{idx_str}"), 0, 1023)?);
        } else if cfg.contains(&format!("deviceToUseModel{idx_str}")) {
          gpu_idx_by_server_thread.push(cfg.get_int(&format!("deviceToUseModel{idx_str}"), 0, 1023)?);
        } else if cfg.contains(&format!("gpuToUseModel{idx_str}")) {
          gpu_idx_by_server_thread.push(cfg.get_int(&format!("gpuToUseModel{idx_str}"), 0, 1023)?);
        } else if cfg.contains(&format!("{backend_prefix}DeviceToUseThread{thread_idx_str}")) {
          gpu_idx_by_server_thread.push(cfg.get_int(&format!("{backend_prefix}DeviceToUseThread{thread_idx_str}"), 0, 1023)?);
        } else if cfg.contains(&format!("{backend_prefix}GpuToUseThread{thread_idx_str}")) {
          gpu_idx_by_server_thread.push(cfg.get_int(&format!("{backend_prefix}GpuToUseThread{thread_idx_str}"), 0, 1023)?);
        } else if cfg.contains(&format!("deviceToUseThread{thread_idx_str}")) {
          gpu_idx_by_server_thread.push(cfg.get_int(&format!("deviceToUseThread{thread_idx_str}"), 0, 1023)?);
        } else if cfg.contains(&format!("gpuToUseThread{thread_idx_str}")) {
          gpu_idx_by_server_thread.push(cfg.get_int(&format!("gpuToUseThread{thread_idx_str}"), 0, 1023)?);
        } else if cfg.contains(&format!("{backend_prefix}DeviceToUse")) {
          gpu_idx_by_server_thread.push(cfg.get_int(&format!("{backend_prefix}DeviceToUse"), 0, 1023)?);
        } else if cfg.contains(&format!("{backend_prefix}GpuToUse")) {
          gpu_idx_by_server_thread.push(cfg.get_int(&format!("{backend_prefix}GpuToUse"), 0, 1023)?);
        } else if cfg.contains("deviceToUse") {
          gpu_idx_by_server_thread.push(cfg.get_int("deviceToUse", 0, 1023)?);
        } else if cfg.contains("gpuToUse") {
          gpu_idx_by_server_thread.push(cfg.get_int("gpuToUse", 0, 1023)?);
        } else {
          gpu_idx_by_server_thread.push(-1);
        }
      }

      let home_data_dir_override: String = Setup::load_home_data_dir_override(cfg)?;

      // Backend-specific options (e.g. openclTunerFile, cudaDisableGraphSDPA) are read directly by the
      // relevant compute backend off of cfg (see createComputeContext). Because they follow the backend
      // prefix convention, the getBackendPrefixes() loop above already marks them used for the backends
      // that don't read them, so no explicit mark-used is needed here.
      // PORT NOTE (v1.18.2 nn-layer sync): the openclTunerFile / openclReTunePerBoardSize / useNHWC reads
      // were removed with the NNEvaluator constructor arguments they fed, as upstream v1.18.2 did.

      let mut use_fp16_mode: enabled_t = enabled_t::Auto;
      if cfg.contains(&format!("{backend_prefix}UseFP16-{idx_str}")) {
        use_fp16_mode = cfg.get_enabled(&format!("{backend_prefix}UseFP16-{idx_str}"))?;
      } else if cfg.contains(&format!("useFP16-{idx_str}")) {
        use_fp16_mode = cfg.get_enabled(&format!("useFP16-{idx_str}"))?;
      } else if cfg.contains(&format!("{backend_prefix}UseFP16")) {
        use_fp16_mode = cfg.get_enabled(&format!("{backend_prefix}UseFP16"))?;
      } else if cfg.contains("useFP16") {
        use_fp16_mode = cfg.get_enabled("useFP16")?;
      }

      let mut forced_symmetry: i32 = -1;
      if setup_for != SETUP_FOR_DISTRIBUTED && cfg.contains("nnForcedSymmetry") {
        forced_symmetry = cfg.get_int("nnForcedSymmetry", 0, SymmetryHelpers::NUM_SYMMETRIES - 1)?;
      }

      log::info!(
        "{}",
        "After dedups: nnModelFile".to_string() + &idx_str + " = " + nn_model_file + " useFP16 " + &use_fp16_mode.to_string()
      );

      let nn_cache_size_power_of_two: i32 = if cfg.contains("nnCacheSizePowerOfTwo") {
        cfg.get_int("nnCacheSizePowerOfTwo", -1, 48)?
      } else if setup_for == SETUP_FOR_GTP {
        20
      } else if setup_for == SETUP_FOR_BENCHMARK {
        20
      } else if setup_for == SETUP_FOR_BENCHMARKNN {
        16
      } else if setup_for == SETUP_FOR_DISTRIBUTED {
        19
      } else if setup_for == SETUP_FOR_MATCH {
        21
      } else if setup_for == SETUP_FOR_ANALYSIS {
        23
      } else {
        cfg.get_int("nnCacheSizePowerOfTwo", -1, 48)?
      };

      let nn_mutex_pool_size_power_of_two: i32 = if cfg.contains("nnMutexPoolSizePowerOfTwo") {
        cfg.get_int("nnMutexPoolSizePowerOfTwo", -1, 24)?
      } else if setup_for == SETUP_FOR_GTP {
        16
      } else if setup_for == SETUP_FOR_BENCHMARK {
        16
      } else if setup_for == SETUP_FOR_BENCHMARKNN {
        12
      } else if setup_for == SETUP_FOR_DISTRIBUTED {
        16
      } else if setup_for == SETUP_FOR_MATCH {
        17
      } else if setup_for == SETUP_FOR_ANALYSIS {
        17
      } else {
        cfg.get_int("nnMutexPoolSizePowerOfTwo", -1, 24)?
      };

      // Turn what the caller knows (maxBatchSizeRequest) into a batch size the way this backend wants
      // it (NeuralNet::getBatchPolicy), rather than having every call site bake in a formula that is
      // only right for GPU backends.
      // PORT NOTE (D6): `NeuralNet::getBatchPolicy` / `getNumEffectiveDevices` of the linked backend -> the host's.
      let batch_policy: BatchPolicy = host.get_batch_policy(cfg);

      let nn_max_batch_size: i32;
      if max_batch_size_request.kind == MaxBatchSizeRequest::EXPLICIT_SIZE {
        // The caller already has an exact batch size in hand (a benchmark or test tool's own number),
        // so no policy transform applies - not even CpuLocal's fixed size, since these tools exist
        // precisely to measure or exercise a specific batch size.
        nn_max_batch_size = max_batch_size_request.size;
        if nn_max_batch_size <= 0 || nn_max_batch_size > 65536 {
          return Err(KataError::StringError("Invalid explicit max batch size: ".to_string() + &Global::int_to_string(nn_max_batch_size)));
        }
      } else if batch_policy == BatchPolicy::CpuLocal {
        // Batching doesn't help this backend (see BatchPolicy::CpuLocal), so fix a small size that
        // saves memory, ignoring both the request and anything the user specified for GPUs.
        nn_max_batch_size = 2;
        cfg.mark_all_keys_used_with_prefix("nnMaxBatchSize");
      } else if max_batch_size_request.kind == MaxBatchSizeRequest::REQUIRE_FROM_CONFIG {
        // No default to fall back on, so the config has to say. Callers using this always pass a
        // setupFor that does not force the request, so there is nothing to force here.
        nn_max_batch_size = cfg.get_int("nnMaxBatchSize", 1, 65536)?;
      } else {
        let derived: i32;
        if batch_policy == BatchPolicy::FixedShape {
          // Every batch gets padded up to this size, so overshooting wastes real arithmetic on every
          // evaluation. Server threads sharing a device share its concurrency, hence dividing by the
          // number of distinct devices rather than by the thread count.
          let num_devices: i32 = host.get_num_effective_devices(cfg, &gpu_idx_by_server_thread);
          derived = Setup::compute_fixed_shape_max_batch_size(expected_concurrent_evals, num_devices);
        } else if max_batch_size_request.kind == MaxBatchSizeRequest::FROM_CONCURRENCY_STRICT {
          // The caller means exactly the concurrency it gave.
          derived = std::cmp::max(1, expected_concurrent_evals);
        } else {
          // Any batch size is fine here and larger ones amortize per-call overhead, so round up and
          // keep a floor.
          derived = std::cmp::max(8, ((expected_concurrent_evals + 3) / 4) * 4);
        }

        let force_request: bool =
          setup_for == SETUP_FOR_BENCHMARK || setup_for == SETUP_FOR_BENCHMARKNN || setup_for == SETUP_FOR_DISTRIBUTED;
        nn_max_batch_size = if !force_request && cfg.contains("nnMaxBatchSize") { cfg.get_int("nnMaxBatchSize", 1, 65536)? } else { derived };
      }

      let default_symmetry: i32 = if forced_symmetry >= 0 { forced_symmetry } else { 0 };
      if disable_fp16 {
        use_fp16_mode = enabled_t::False;
      }

      // PORT NOTE (D6): the backend and the model file's bytes come from the host. Upstream's NNEvaluator constructor opens
      // the model file only when !debugSkipNeuralNet, and only after its own argument checks, so the host is asked under
      // the same two conditions (a missing model file must not hide "maxBatchSize is negative", and a skipped net
      // never touches the file system). NNEvaluator::new does not call the backend when it skips the net.
      let (backend, model_bytes): (Rc<dyn NeuralNet>, Vec<u8>) = if debug_skip_neural_net {
        (Rc::new(SkippedNeuralNet), Vec::new())
      } else {
        check_nn_evaluator_args(nn_max_batch_size, nn_x_len, nn_y_len, num_nn_server_threads_per_model, &gpu_idx_by_server_thread)?;
        host.load_model(nn_model_name, nn_model_file)?
      };
      let nn_eval: Rc<NNEvaluator> = Rc::new(NNEvaluator::new(
        nn_model_name,
        nn_model_file,
        &expected_sha256,
        nn_max_batch_size,
        nn_x_len,
        nn_y_len,
        require_exact_nn_len,
        inputs_use_nhwc,
        nn_cache_size_power_of_two,
        nn_mutex_pool_size_power_of_two,
        debug_skip_neural_net,
        &home_data_dir_override,
        use_fp16_mode,
        num_nn_server_threads_per_model,
        &gpu_idx_by_server_thread,
        &nn_rand_seed,
        if forced_symmetry >= 0 { false } else { nn_randomize },
        default_symmetry,
        cfg,
        backend,
        host.spawner(),
        model_bytes,
      )?);

      // benchmarkPureForward drives compute handles itself, without server threads.
      if setup_for != SETUP_FOR_BENCHMARKNN {
        nn_eval.spawn_server_threads().await?;
      }

      nn_evals.push(nn_eval);
    }

    Ok(nn_evals)
  }

  /// upstream: Setup::computeFixedShapeMaxBatchSize (setup.cpp)
  // The default nnMaxBatchSize for backends whose BatchPolicy is FixedShape: half of the evals one
  // device sees at a time, rounded up, so that two batches can be in flight per device. Exposed so
  // that the benchmark can compute the same per-thread-count batch sizes this policy would pick.
  pub fn compute_fixed_shape_max_batch_size(expected_concurrent_evals: i32, num_devices: i32) -> i32 {
    let num_devices: i32 = std::cmp::max(1, num_devices);
    let per_device: i32 = (expected_concurrent_evals + num_devices - 1) / num_devices;
    // Floor of 2 rather than 1: batch size 1 makes OpenVINO's NPU inference produce nonfinite
    // policy outputs (a provider bug, observed with OpenVINO 2026.2.1), and a 2-row batch costs
    // low-thread-count configurations only a little on the other providers.
    std::cmp::max(2, (per_device + 1) / 2)
  }

  // upstream: Setup::computeDefaultEigenBackendThreads (setup.cpp)
  // PORT NOTE: not ported (Eigen backend only; std::thread::hardware_concurrency). See the manifest notes.

  /// upstream: Setup::loadHomeDataDirOverride (setup.cpp)
  pub fn load_home_data_dir_override(cfg: &mut ConfigParser) -> Result<String, KataError> {
    let mut home_data_dir_override: String = String::new();
    if cfg.contains("homeDataDir") {
      home_data_dir_override = cfg.get_string("homeDataDir")?;
    }
    Ok(home_data_dir_override)
  }

  /// upstream: Setup::loadSingleParams(ConfigParser&, setup_for_t) (setup.cpp)
  pub fn load_single_params(cfg: &mut ConfigParser, setup_for: setup_for_t) -> Result<SearchParams, KataError> {
    let has_human_model: bool = false;
    Setup::load_single_params_has_human_model(cfg, setup_for, has_human_model)
  }

  /// upstream: Setup::loadSingleParams(ConfigParser&, setup_for_t, bool) (setup.cpp)
  pub fn load_single_params_has_human_model(cfg: &mut ConfigParser, setup_for: setup_for_t, has_human_model: bool) -> Result<SearchParams, KataError> {
    let load_single_config_only: bool = true;
    let mut paramss: Vec<SearchParams> = Setup::load_params_full(cfg, setup_for, has_human_model, load_single_config_only)?;
    if paramss.len() != 1 {
      return Err(KataError::StringError(
        "Config contains parameters for multiple bot configurations, but this KataGo command only supports a single configuration".to_string(),
      ));
    }
    Ok(paramss.remove(0))
  }

  /// upstream: Setup::loadParams(ConfigParser&, setup_for_t) (setup.cpp)
  //Loads search parameters for bot from config, by bot idx.
  //Fails if no parameters are found.
  pub fn load_params(cfg: &mut ConfigParser, setup_for: setup_for_t) -> Result<Vec<SearchParams>, KataError> {
    let has_human_model: bool = false;
    let load_single_config_only: bool = false;
    Setup::load_params_full(cfg, setup_for, has_human_model, load_single_config_only)
  }

  /// upstream: Setup::loadParams(ConfigParser&, setup_for_t, bool) (setup.cpp)
  pub fn load_params_has_human_model(cfg: &mut ConfigParser, setup_for: setup_for_t, has_human_model: bool) -> Result<Vec<SearchParams>, KataError> {
    let load_single_config_only: bool = false;
    Setup::load_params_full(cfg, setup_for, has_human_model, load_single_config_only)
  }

  /// upstream: Setup::loadParams(ConfigParser&, setup_for_t, bool, bool) (setup.cpp)
  /// PORT NOTE: name from PORTING.md Appendix B; throws -> Result.
  pub fn load_params_full(
    cfg: &mut ConfigParser,
    setup_for: setup_for_t,
    has_human_model: bool,
    load_single_config_only: bool,
  ) -> Result<Vec<SearchParams>, KataError> {
    let mut paramss: Vec<SearchParams> = Vec::new();
    let mut num_bots: i32 = 1;
    if cfg.contains("numBots") {
      num_bots = cfg.get_int("numBots", 1, Setup::MAX_BOT_PARAMS_FROM_CFG)?;
    }

    if load_single_config_only {
      if num_bots != 1 {
        return Err(KataError::ConfigParsingError("The config for this command cannot have numBots > 0".to_string()));
      }
    }

    for i in 0..num_bots {
      let mut params: SearchParams = SearchParams::new();

      let idx_str: String = if load_single_config_only { String::new() } else { Global::int_to_string(i) };

      if cfg.contains(&format!("maxPlayouts{idx_str}")) {
        params.max_playouts = cfg.get_int64(&format!("maxPlayouts{idx_str}"), 1i64, 1i64 << 50)?;
      } else if cfg.contains("maxPlayouts") {
        params.max_playouts = cfg.get_int64("maxPlayouts", 1i64, 1i64 << 50)?;
      }
      if cfg.contains(&format!("maxVisits{idx_str}")) {
        params.max_visits = cfg.get_int64(&format!("maxVisits{idx_str}"), 1i64, 1i64 << 50)?;
      } else if cfg.contains("maxVisits") {
        params.max_visits = cfg.get_int64("maxVisits", 1i64, 1i64 << 50)?;
      }
      if cfg.contains(&format!("maxTime{idx_str}")) {
        params.max_time = cfg.get_double(&format!("maxTime{idx_str}"), 0.0, 1.0e20)?;
      } else if cfg.contains("maxTime") {
        params.max_time = cfg.get_double("maxTime", 0.0, 1.0e20)?;
      }

      if cfg.contains(&format!("maxPlayoutsPondering{idx_str}")) {
        params.max_playouts_pondering = cfg.get_int64(&format!("maxPlayoutsPondering{idx_str}"), 1i64, 1i64 << 50)?;
      } else if cfg.contains("maxPlayoutsPondering") {
        params.max_playouts_pondering = cfg.get_int64("maxPlayoutsPondering", 1i64, 1i64 << 50)?;
      } else {
        params.max_playouts_pondering = 1i64 << 50;
      }
      if cfg.contains(&format!("maxVisitsPondering{idx_str}")) {
        params.max_visits_pondering = cfg.get_int64(&format!("maxVisitsPondering{idx_str}"), 1i64, 1i64 << 50)?;
      } else if cfg.contains("maxVisitsPondering") {
        params.max_visits_pondering = cfg.get_int64("maxVisitsPondering", 1i64, 1i64 << 50)?;
      } else {
        params.max_visits_pondering = 1i64 << 50;
      }
      if cfg.contains(&format!("maxTimePondering{idx_str}")) {
        params.max_time_pondering = cfg.get_double(&format!("maxTimePondering{idx_str}"), 0.0, 1.0e20)?;
      } else if cfg.contains("maxTimePondering") {
        params.max_time_pondering = cfg.get_double("maxTimePondering", 0.0, 1.0e20)?;
      } else {
        params.max_time_pondering = 1.0e20;
      }

      if cfg.contains(&format!("lagBuffer{idx_str}")) {
        params.lag_buffer = cfg.get_double(&format!("lagBuffer{idx_str}"), 0.0, 3600.0)?;
      } else if cfg.contains("lagBuffer") {
        params.lag_buffer = cfg.get_double("lagBuffer", 0.0, 3600.0)?;
      } else {
        params.lag_buffer = 0.0;
      }

      if cfg.contains(&format!("searchFactorAfterOnePass{idx_str}")) {
        params.search_factor_after_one_pass = cfg.get_double(&format!("searchFactorAfterOnePass{idx_str}"), 0.0, 1.0)?;
      } else if cfg.contains("searchFactorAfterOnePass") {
        params.search_factor_after_one_pass = cfg.get_double("searchFactorAfterOnePass", 0.0, 1.0)?;
      }
      if cfg.contains(&format!("searchFactorAfterTwoPass{idx_str}")) {
        params.search_factor_after_two_pass = cfg.get_double(&format!("searchFactorAfterTwoPass{idx_str}"), 0.0, 1.0)?;
      } else if cfg.contains("searchFactorAfterTwoPass") {
        params.search_factor_after_two_pass = cfg.get_double("searchFactorAfterTwoPass", 0.0, 1.0)?;
      }

      if cfg.contains(&format!("numSearchThreads{idx_str}")) {
        params.num_threads = cfg.get_int(&format!("numSearchThreads{idx_str}"), 1, 4096)?;
      } else {
        params.num_threads = cfg.get_int("numSearchThreads", 1, 4096)?;
      }

      if cfg.contains(&format!("minPlayoutsPerThread{idx_str}")) {
        params.min_playouts_per_thread = cfg.get_double(&format!("minPlayoutsPerThread{idx_str}"), 0.0, 1.0e20)?;
      } else if cfg.contains("minPlayoutsPerThread") {
        params.min_playouts_per_thread = cfg.get_double("minPlayoutsPerThread", 0.0, 1.0e20)?;
      } else {
        params.min_playouts_per_thread = if setup_for == SETUP_FOR_ANALYSIS || setup_for == SETUP_FOR_GTP { 8.0 } else { 0.0 };
      }

      if cfg.contains(&format!("winLossUtilityFactor{idx_str}")) {
        params.win_loss_utility_factor = cfg.get_double(&format!("winLossUtilityFactor{idx_str}"), 0.0, 1.0)?;
      } else if cfg.contains("winLossUtilityFactor") {
        params.win_loss_utility_factor = cfg.get_double("winLossUtilityFactor", 0.0, 1.0)?;
      } else {
        params.win_loss_utility_factor = 1.0;
      }
      if cfg.contains(&format!("staticScoreUtilityFactor{idx_str}")) {
        params.static_score_utility_factor = cfg.get_double(&format!("staticScoreUtilityFactor{idx_str}"), 0.0, 1.0)?;
      } else if cfg.contains("staticScoreUtilityFactor") {
        params.static_score_utility_factor = cfg.get_double("staticScoreUtilityFactor", 0.0, 1.0)?;
      } else {
        params.static_score_utility_factor = 0.1;
      }
      if cfg.contains(&format!("dynamicScoreUtilityFactor{idx_str}")) {
        params.dynamic_score_utility_factor = cfg.get_double(&format!("dynamicScoreUtilityFactor{idx_str}"), 0.0, 1.0)?;
      } else if cfg.contains("dynamicScoreUtilityFactor") {
        params.dynamic_score_utility_factor = cfg.get_double("dynamicScoreUtilityFactor", 0.0, 1.0)?;
      } else {
        params.dynamic_score_utility_factor = 0.3;
      }
      if cfg.contains(&format!("noResultUtilityForWhite{idx_str}")) {
        params.no_result_utility_for_white = cfg.get_double(&format!("noResultUtilityForWhite{idx_str}"), -1.0, 1.0)?;
      } else if cfg.contains("noResultUtilityForWhite") {
        params.no_result_utility_for_white = cfg.get_double("noResultUtilityForWhite", -1.0, 1.0)?;
      } else {
        params.no_result_utility_for_white = 0.0;
      }
      if cfg.contains(&format!("drawEquivalentWinsForWhite{idx_str}")) {
        params.draw_equivalent_wins_for_white = cfg.get_double(&format!("drawEquivalentWinsForWhite{idx_str}"), 0.0, 1.0)?;
      } else if cfg.contains("drawEquivalentWinsForWhite") {
        params.draw_equivalent_wins_for_white = cfg.get_double("drawEquivalentWinsForWhite", 0.0, 1.0)?;
      } else {
        params.draw_equivalent_wins_for_white = 0.5;
      }

      if cfg.contains(&format!("dynamicScoreCenterZeroWeight{idx_str}")) {
        params.dynamic_score_center_zero_weight = cfg.get_double(&format!("dynamicScoreCenterZeroWeight{idx_str}"), 0.0, 1.0)?;
      } else if cfg.contains("dynamicScoreCenterZeroWeight") {
        params.dynamic_score_center_zero_weight = cfg.get_double("dynamicScoreCenterZeroWeight", 0.0, 1.0)?;
      } else {
        params.dynamic_score_center_zero_weight = 0.20;
      }
      if cfg.contains(&format!("dynamicScoreCenterScale{idx_str}")) {
        params.dynamic_score_center_scale = cfg.get_double(&format!("dynamicScoreCenterScale{idx_str}"), 0.2, 5.0)?;
      } else if cfg.contains("dynamicScoreCenterScale") {
        params.dynamic_score_center_scale = cfg.get_double("dynamicScoreCenterScale", 0.2, 5.0)?;
      } else {
        params.dynamic_score_center_scale = 0.75;
      }

      if cfg.contains(&format!("cpuctExploration{idx_str}")) {
        params.cpuct_exploration = cfg.get_double(&format!("cpuctExploration{idx_str}"), 0.0, 10.0)?;
      } else if cfg.contains("cpuctExploration") {
        params.cpuct_exploration = cfg.get_double("cpuctExploration", 0.0, 10.0)?;
      } else {
        params.cpuct_exploration = 1.0;
      }
      if cfg.contains(&format!("cpuctExplorationLog{idx_str}")) {
        params.cpuct_exploration_log = cfg.get_double(&format!("cpuctExplorationLog{idx_str}"), 0.0, 10.0)?;
      } else if cfg.contains("cpuctExplorationLog") {
        params.cpuct_exploration_log = cfg.get_double("cpuctExplorationLog", 0.0, 10.0)?;
      } else {
        params.cpuct_exploration_log = 0.45;
      }
      if cfg.contains(&format!("cpuctExplorationBase{idx_str}")) {
        params.cpuct_exploration_base = cfg.get_double(&format!("cpuctExplorationBase{idx_str}"), 10.0, 100000.0)?;
      } else if cfg.contains("cpuctExplorationBase") {
        params.cpuct_exploration_base = cfg.get_double("cpuctExplorationBase", 10.0, 100000.0)?;
      } else {
        params.cpuct_exploration_base = 500.0;
      }

      if cfg.contains(&format!("cpuctUtilityStdevPrior{idx_str}")) {
        params.cpuct_utility_stdev_prior = cfg.get_double(&format!("cpuctUtilityStdevPrior{idx_str}"), 1e-8, 10.0)?;
      } else if cfg.contains("cpuctUtilityStdevPrior") {
        params.cpuct_utility_stdev_prior = cfg.get_double("cpuctUtilityStdevPrior", 1e-8, 10.0)?;
      } else {
        params.cpuct_utility_stdev_prior = 0.40;
      }
      if cfg.contains(&format!("cpuctUtilityStdevPriorWeight{idx_str}")) {
        params.cpuct_utility_stdev_prior_weight = cfg.get_double(&format!("cpuctUtilityStdevPriorWeight{idx_str}"), 0.0, 100.0)?;
      } else if cfg.contains("cpuctUtilityStdevPriorWeight") {
        params.cpuct_utility_stdev_prior_weight = cfg.get_double("cpuctUtilityStdevPriorWeight", 0.0, 100.0)?;
      } else {
        params.cpuct_utility_stdev_prior_weight = 2.0;
      }
      if cfg.contains(&format!("cpuctUtilityStdevScale{idx_str}")) {
        params.cpuct_utility_stdev_scale = cfg.get_double(&format!("cpuctUtilityStdevScale{idx_str}"), 0.0, 1.0)?;
      } else if cfg.contains("cpuctUtilityStdevScale") {
        params.cpuct_utility_stdev_scale = cfg.get_double("cpuctUtilityStdevScale", 0.0, 1.0)?;
      } else {
        params.cpuct_utility_stdev_scale = if setup_for != SETUP_FOR_DISTRIBUTED && setup_for != SETUP_FOR_OTHER { 0.85 } else { 0.0 };
      }


      if cfg.contains(&format!("fpuReductionMax{idx_str}")) {
        params.fpu_reduction_max = cfg.get_double(&format!("fpuReductionMax{idx_str}"), 0.0, 2.0)?;
      } else if cfg.contains("fpuReductionMax") {
        params.fpu_reduction_max = cfg.get_double("fpuReductionMax", 0.0, 2.0)?;
      } else {
        params.fpu_reduction_max = 0.2;
      }
      if cfg.contains(&format!("fpuLossProp{idx_str}")) {
        params.fpu_loss_prop = cfg.get_double(&format!("fpuLossProp{idx_str}"), 0.0, 1.0)?;
      } else if cfg.contains("fpuLossProp") {
        params.fpu_loss_prop = cfg.get_double("fpuLossProp", 0.0, 1.0)?;
      } else {
        params.fpu_loss_prop = 0.0;
      }
      if cfg.contains(&format!("fpuParentWeightByVisitedPolicy{idx_str}")) {
        params.fpu_parent_weight_by_visited_policy = cfg.get_bool(&format!("fpuParentWeightByVisitedPolicy{idx_str}"))?;
      } else if cfg.contains("fpuParentWeightByVisitedPolicy") {
        params.fpu_parent_weight_by_visited_policy = cfg.get_bool("fpuParentWeightByVisitedPolicy")?;
      } else {
        params.fpu_parent_weight_by_visited_policy = setup_for != SETUP_FOR_DISTRIBUTED;
      }
      if params.fpu_parent_weight_by_visited_policy {
      if cfg.contains(&format!("fpuParentWeightByVisitedPolicyPow{idx_str}")) {
        params.fpu_parent_weight_by_visited_policy_pow = cfg.get_double(&format!("fpuParentWeightByVisitedPolicyPow{idx_str}"), 0.0, 5.0)?;
      } else if cfg.contains("fpuParentWeightByVisitedPolicyPow") {
        params.fpu_parent_weight_by_visited_policy_pow = cfg.get_double("fpuParentWeightByVisitedPolicyPow", 0.0, 5.0)?;
      } else {
        params.fpu_parent_weight_by_visited_policy_pow = 2.0;
      }
      } else {
      if cfg.contains(&format!("fpuParentWeight{idx_str}")) {
        params.fpu_parent_weight = cfg.get_double(&format!("fpuParentWeight{idx_str}"), 0.0, 1.0)?;
      } else if cfg.contains("fpuParentWeight") {
        params.fpu_parent_weight = cfg.get_double("fpuParentWeight", 0.0, 1.0)?;
      } else {
        params.fpu_parent_weight = 0.0;
      }
      }

      if cfg.contains(&format!("policyOptimism{idx_str}")) {
        params.policy_optimism = cfg.get_double(&format!("policyOptimism{idx_str}"), 0.0, 1.0)?;
      } else if cfg.contains("policyOptimism") {
        params.policy_optimism = cfg.get_double("policyOptimism", 0.0, 1.0)?;
      } else {
        params.policy_optimism = if setup_for != SETUP_FOR_DISTRIBUTED && setup_for != SETUP_FOR_OTHER { 1.0 } else { 0.0 };
      }

      if cfg.contains(&format!("valueWeightExponent{idx_str}")) {
        params.value_weight_exponent = cfg.get_double(&format!("valueWeightExponent{idx_str}"), 0.0, 1.0)?;
      } else if cfg.contains("valueWeightExponent") {
        params.value_weight_exponent = cfg.get_double("valueWeightExponent", 0.0, 1.0)?;
      } else {
        params.value_weight_exponent = 0.25;
      }
      if cfg.contains(&format!("useNoisePruning{idx_str}")) {
        params.use_noise_pruning = cfg.get_bool(&format!("useNoisePruning{idx_str}"))?;
      } else if cfg.contains("useNoisePruning") {
        params.use_noise_pruning = cfg.get_bool("useNoisePruning")?;
      } else {
        params.use_noise_pruning = setup_for != SETUP_FOR_DISTRIBUTED && setup_for != SETUP_FOR_OTHER;
      }
      if cfg.contains(&format!("noisePruneUtilityScale{idx_str}")) {
        params.noise_prune_utility_scale = cfg.get_double(&format!("noisePruneUtilityScale{idx_str}"), 0.001, 10.0)?;
      } else if cfg.contains("noisePruneUtilityScale") {
        params.noise_prune_utility_scale = cfg.get_double("noisePruneUtilityScale", 0.001, 10.0)?;
      } else {
        params.noise_prune_utility_scale = 0.15;
      }
      if cfg.contains(&format!("noisePruningCap{idx_str}")) {
        params.noise_pruning_cap = cfg.get_double(&format!("noisePruningCap{idx_str}"), 0.0, 1e50)?;
      } else if cfg.contains("noisePruningCap") {
        params.noise_pruning_cap = cfg.get_double("noisePruningCap", 0.0, 1e50)?;
      } else {
        params.noise_pruning_cap = 1e50;
      }


      if cfg.contains(&format!("useUncertainty{idx_str}")) {
        params.use_uncertainty = cfg.get_bool(&format!("useUncertainty{idx_str}"))?;
      } else if cfg.contains("useUncertainty") {
        params.use_uncertainty = cfg.get_bool("useUncertainty")?;
      } else {
        params.use_uncertainty = setup_for != SETUP_FOR_DISTRIBUTED && setup_for != SETUP_FOR_OTHER;
      }
      if cfg.contains(&format!("uncertaintyCoeff{idx_str}")) {
        params.uncertainty_coeff = cfg.get_double(&format!("uncertaintyCoeff{idx_str}"), 0.0001, 1.0)?;
      } else if cfg.contains("uncertaintyCoeff") {
        params.uncertainty_coeff = cfg.get_double("uncertaintyCoeff", 0.0001, 1.0)?;
      } else {
        params.uncertainty_coeff = 0.25;
      }
      if cfg.contains(&format!("uncertaintyExponent{idx_str}")) {
        params.uncertainty_exponent = cfg.get_double(&format!("uncertaintyExponent{idx_str}"), 0.0, 2.0)?;
      } else if cfg.contains("uncertaintyExponent") {
        params.uncertainty_exponent = cfg.get_double("uncertaintyExponent", 0.0, 2.0)?;
      } else {
        params.uncertainty_exponent = 1.0;
      }
      if cfg.contains(&format!("uncertaintyMaxWeight{idx_str}")) {
        params.uncertainty_max_weight = cfg.get_double(&format!("uncertaintyMaxWeight{idx_str}"), 1.0, 100.0)?;
      } else if cfg.contains("uncertaintyMaxWeight") {
        params.uncertainty_max_weight = cfg.get_double("uncertaintyMaxWeight", 1.0, 100.0)?;
      } else {
        params.uncertainty_max_weight = 8.0;
      }

      if cfg.contains(&format!("useGraphSearch{idx_str}")) {
        params.use_graph_search = cfg.get_bool(&format!("useGraphSearch{idx_str}"))?;
      } else if cfg.contains("useGraphSearch") {
        params.use_graph_search = cfg.get_bool("useGraphSearch")?;
      } else {
        params.use_graph_search = setup_for != SETUP_FOR_DISTRIBUTED;
      }
      if cfg.contains(&format!("graphSearchRepBound{idx_str}")) {
        params.graph_search_rep_bound = cfg.get_int(&format!("graphSearchRepBound{idx_str}"), 3, 50)?;
      } else if cfg.contains("graphSearchRepBound") {
        params.graph_search_rep_bound = cfg.get_int("graphSearchRepBound", 3, 50)?;
      } else {
        params.graph_search_rep_bound = 11;
      }
      if cfg.contains(&format!("graphSearchCatchUpLeakProb{idx_str}")) {
        params.graph_search_catch_up_leak_prob = cfg.get_double(&format!("graphSearchCatchUpLeakProb{idx_str}"), 0.0, 1.0)?;
      } else if cfg.contains("graphSearchCatchUpLeakProb") {
        params.graph_search_catch_up_leak_prob = cfg.get_double("graphSearchCatchUpLeakProb", 0.0, 1.0)?;
      } else {
        params.graph_search_catch_up_leak_prob = 0.0;
      }
      // if(cfg.contains("graphSearchCatchUpProp"+idxStr)) params.graphSearchCatchUpProp = cfg.getDouble("graphSearchCatchUpProp"+idxStr, 0.0, 1.0);
      // else if(cfg.contains("graphSearchCatchUpProp"))   params.graphSearchCatchUpProp = cfg.getDouble("graphSearchCatchUpProp", 0.0, 1.0);
      // else                                              params.graphSearchCatchUpProp = 0.0;

      if cfg.contains(&format!("rootNoiseEnabled{idx_str}")) {
        params.root_noise_enabled = cfg.get_bool(&format!("rootNoiseEnabled{idx_str}"))?;
      } else if cfg.contains("rootNoiseEnabled") {
        params.root_noise_enabled = cfg.get_bool("rootNoiseEnabled")?;
      } else {
        params.root_noise_enabled = false;
      }
      if cfg.contains(&format!("rootDirichletNoiseTotalConcentration{idx_str}")) {
        params.root_dirichlet_noise_total_concentration = cfg.get_double(&format!("rootDirichletNoiseTotalConcentration{idx_str}"), 0.001, 10000.0)?;
      } else if cfg.contains("rootDirichletNoiseTotalConcentration") {
        params.root_dirichlet_noise_total_concentration = cfg.get_double("rootDirichletNoiseTotalConcentration", 0.001, 10000.0)?;
      } else {
        params.root_dirichlet_noise_total_concentration = 10.83;
      }
      if cfg.contains(&format!("rootDirichletNoiseWeight{idx_str}")) {
        params.root_dirichlet_noise_weight = cfg.get_double(&format!("rootDirichletNoiseWeight{idx_str}"), 0.0, 1.0)?;
      } else if cfg.contains("rootDirichletNoiseWeight") {
        params.root_dirichlet_noise_weight = cfg.get_double("rootDirichletNoiseWeight", 0.0, 1.0)?;
      } else {
        params.root_dirichlet_noise_weight = 0.25;
      }

      if cfg.contains(&format!("rootPolicyTemperature{idx_str}")) {
        params.root_policy_temperature = cfg.get_double(&format!("rootPolicyTemperature{idx_str}"), 0.01, 100.0)?;
      } else if cfg.contains("rootPolicyTemperature") {
        params.root_policy_temperature = cfg.get_double("rootPolicyTemperature", 0.01, 100.0)?;
      } else {
        params.root_policy_temperature = 1.0;
      }
      if cfg.contains(&format!("rootPolicyTemperatureEarly{idx_str}")) {
        params.root_policy_temperature_early = cfg.get_double(&format!("rootPolicyTemperatureEarly{idx_str}"), 0.01, 100.0)?;
      } else if cfg.contains("rootPolicyTemperatureEarly") {
        params.root_policy_temperature_early = cfg.get_double("rootPolicyTemperatureEarly", 0.01, 100.0)?;
      } else {
        params.root_policy_temperature_early = params.root_policy_temperature;
      }
      if cfg.contains(&format!("rootFpuReductionMax{idx_str}")) {
        params.root_fpu_reduction_max = cfg.get_double(&format!("rootFpuReductionMax{idx_str}"), 0.0, 2.0)?;
      } else if cfg.contains("rootFpuReductionMax") {
        params.root_fpu_reduction_max = cfg.get_double("rootFpuReductionMax", 0.0, 2.0)?;
      } else {
        params.root_fpu_reduction_max = if params.root_noise_enabled { 0.0 } else { 0.1 };
      }
      if cfg.contains(&format!("rootFpuLossProp{idx_str}")) {
        params.root_fpu_loss_prop = cfg.get_double(&format!("rootFpuLossProp{idx_str}"), 0.0, 1.0)?;
      } else if cfg.contains("rootFpuLossProp") {
        params.root_fpu_loss_prop = cfg.get_double("rootFpuLossProp", 0.0, 1.0)?;
      } else {
        params.root_fpu_loss_prop = params.fpu_loss_prop;
      }
      if cfg.contains(&format!("rootNumSymmetriesToSample{idx_str}")) {
        params.root_num_symmetries_to_sample = cfg.get_int(&format!("rootNumSymmetriesToSample{idx_str}"), 1, SymmetryHelpers::NUM_SYMMETRIES)?;
      } else if cfg.contains("rootNumSymmetriesToSample") {
        params.root_num_symmetries_to_sample = cfg.get_int("rootNumSymmetriesToSample", 1, SymmetryHelpers::NUM_SYMMETRIES)?;
      } else {
        params.root_num_symmetries_to_sample = 1;
      }
      if cfg.contains(&format!("rootSymmetryPruning{idx_str}")) {
        params.root_symmetry_pruning = cfg.get_bool(&format!("rootSymmetryPruning{idx_str}"))?;
      } else if cfg.contains("rootSymmetryPruning") {
        params.root_symmetry_pruning = cfg.get_bool("rootSymmetryPruning")?;
      } else {
        params.root_symmetry_pruning = setup_for == SETUP_FOR_ANALYSIS || setup_for == SETUP_FOR_GTP;
      }

      if cfg.contains(&format!("rootDesiredPerChildVisitsCoeff{idx_str}")) {
        params.root_desired_per_child_visits_coeff = cfg.get_double(&format!("rootDesiredPerChildVisitsCoeff{idx_str}"), 0.0, 100.0)?;
      } else if cfg.contains("rootDesiredPerChildVisitsCoeff") {
        params.root_desired_per_child_visits_coeff = cfg.get_double("rootDesiredPerChildVisitsCoeff", 0.0, 100.0)?;
      } else {
        params.root_desired_per_child_visits_coeff = 0.0;
      }

      if cfg.contains(&format!("rootPolicyOptimism{idx_str}")) {
        params.root_policy_optimism = cfg.get_double(&format!("rootPolicyOptimism{idx_str}"), 0.0, 1.0)?;
      } else if cfg.contains("rootPolicyOptimism") {
        params.root_policy_optimism = cfg.get_double("rootPolicyOptimism", 0.0, 1.0)?;
      } else {
        params.root_policy_optimism = if setup_for != SETUP_FOR_DISTRIBUTED && setup_for != SETUP_FOR_OTHER { cpp_min(params.policy_optimism, 0.2) } else { 0.0 };
      }

      if cfg.contains(&format!("chosenMoveTemperature{idx_str}")) {
        params.chosen_move_temperature = cfg.get_double(&format!("chosenMoveTemperature{idx_str}"), 0.0, 5.0)?;
      } else if cfg.contains("chosenMoveTemperature") {
        params.chosen_move_temperature = cfg.get_double("chosenMoveTemperature", 0.0, 5.0)?;
      } else {
        params.chosen_move_temperature = 0.1;
      }
      if cfg.contains(&format!("chosenMoveTemperatureEarly{idx_str}")) {
        params.chosen_move_temperature_early = cfg.get_double(&format!("chosenMoveTemperatureEarly{idx_str}"), 0.0, 5.0)?;
      } else if cfg.contains("chosenMoveTemperatureEarly") {
        params.chosen_move_temperature_early = cfg.get_double("chosenMoveTemperatureEarly", 0.0, 5.0)?;
      } else {
        params.chosen_move_temperature_early = 0.5;
      }
      if cfg.contains(&format!("chosenMoveTemperatureHalflife{idx_str}")) {
        params.chosen_move_temperature_halflife = cfg.get_double(&format!("chosenMoveTemperatureHalflife{idx_str}"), 0.1, 100000.0)?;
      } else if cfg.contains("chosenMoveTemperatureHalflife") {
        params.chosen_move_temperature_halflife = cfg.get_double("chosenMoveTemperatureHalflife", 0.1, 100000.0)?;
      } else {
        params.chosen_move_temperature_halflife = 19.0;
      }
      if cfg.contains(&format!("chosenMoveTemperatureOnlyBelowProb{idx_str}")) {
        params.chosen_move_temperature_only_below_prob = cfg.get_double(&format!("chosenMoveTemperatureOnlyBelowProb{idx_str}"), 0.0, 1.0)?;
      } else if cfg.contains("chosenMoveTemperatureOnlyBelowProb") {
        params.chosen_move_temperature_only_below_prob = cfg.get_double("chosenMoveTemperatureOnlyBelowProb", 0.0, 1.0)?;
      } else {
        params.chosen_move_temperature_only_below_prob = 1.0;
      }
      if cfg.contains(&format!("chosenMoveSubtract{idx_str}")) {
        params.chosen_move_subtract = cfg.get_double(&format!("chosenMoveSubtract{idx_str}"), 0.0, 1.0e10)?;
      } else if cfg.contains("chosenMoveSubtract") {
        params.chosen_move_subtract = cfg.get_double("chosenMoveSubtract", 0.0, 1.0e10)?;
      } else {
        params.chosen_move_subtract = 0.0;
      }
      if cfg.contains(&format!("chosenMovePrune{idx_str}")) {
        params.chosen_move_prune = cfg.get_double(&format!("chosenMovePrune{idx_str}"), 0.0, 1.0e10)?;
      } else if cfg.contains("chosenMovePrune") {
        params.chosen_move_prune = cfg.get_double("chosenMovePrune", 0.0, 1.0e10)?;
      } else {
        params.chosen_move_prune = 1.0;
      }

      if cfg.contains(&format!("useLcbForSelection{idx_str}")) {
        params.use_lcb_for_selection = cfg.get_bool(&format!("useLcbForSelection{idx_str}"))?;
      } else if cfg.contains("useLcbForSelection") {
        params.use_lcb_for_selection = cfg.get_bool("useLcbForSelection")?;
      } else {
        params.use_lcb_for_selection = true;
      }
      if cfg.contains(&format!("lcbStdevs{idx_str}")) {
        params.lcb_stdevs = cfg.get_double(&format!("lcbStdevs{idx_str}"), 1.0, 12.0)?;
      } else if cfg.contains("lcbStdevs") {
        params.lcb_stdevs = cfg.get_double("lcbStdevs", 1.0, 12.0)?;
      } else {
        params.lcb_stdevs = 5.0;
      }
      if cfg.contains(&format!("minVisitPropForLCB{idx_str}")) {
        params.min_visit_prop_for_lcb = cfg.get_double(&format!("minVisitPropForLCB{idx_str}"), 0.0, 1.0)?;
      } else if cfg.contains("minVisitPropForLCB") {
        params.min_visit_prop_for_lcb = cfg.get_double("minVisitPropForLCB", 0.0, 1.0)?;
      } else {
        params.min_visit_prop_for_lcb = 0.15;
      }
      //For distributed and selfplay, we default to buggy LCB for the moment since it has effects on the policy training target.
      if cfg.contains(&format!("useNonBuggyLcb{idx_str}")) {
        params.use_non_buggy_lcb = cfg.get_bool(&format!("useNonBuggyLcb{idx_str}"))?;
      } else if cfg.contains("useNonBuggyLcb") {
        params.use_non_buggy_lcb = cfg.get_bool("useNonBuggyLcb")?;
      } else {
        params.use_non_buggy_lcb = setup_for != SETUP_FOR_DISTRIBUTED && setup_for != SETUP_FOR_OTHER;
      }


      if cfg.contains(&format!("rootEndingBonusPoints{idx_str}")) {
        params.root_ending_bonus_points = cfg.get_double(&format!("rootEndingBonusPoints{idx_str}"), -1.0, 1.0)?;
      } else if cfg.contains("rootEndingBonusPoints") {
        params.root_ending_bonus_points = cfg.get_double("rootEndingBonusPoints", -1.0, 1.0)?;
      } else {
        params.root_ending_bonus_points = 0.5;
      }
      if cfg.contains(&format!("rootPruneUselessMoves{idx_str}")) {
        params.root_prune_useless_moves = cfg.get_bool(&format!("rootPruneUselessMoves{idx_str}"))?;
      } else if cfg.contains("rootPruneUselessMoves") {
        params.root_prune_useless_moves = cfg.get_bool("rootPruneUselessMoves")?;
      } else {
        params.root_prune_useless_moves = true;
      }
      if cfg.contains(&format!("conservativePass{idx_str}")) {
        params.conservative_pass = cfg.get_bool(&format!("conservativePass{idx_str}"))?;
      } else if cfg.contains("conservativePass") {
        params.conservative_pass = cfg.get_bool("conservativePass")?;
      } else {
        params.conservative_pass = false;
      }
      if cfg.contains(&format!("fillDameBeforePass{idx_str}")) {
        params.fill_dame_before_pass = cfg.get_bool(&format!("fillDameBeforePass{idx_str}"))?;
      } else if cfg.contains("fillDameBeforePass") {
        params.fill_dame_before_pass = cfg.get_bool("fillDameBeforePass")?;
      } else {
        params.fill_dame_before_pass = false;
      }
      if cfg.contains(&format!("alwaysComputePassAliveUnderSuicideRules{idx_str}")) {
        params.always_compute_pass_alive_under_suicide_rules = cfg.get_enabled(&format!("alwaysComputePassAliveUnderSuicideRules{idx_str}"))?;
      } else if cfg.contains("alwaysComputePassAliveUnderSuicideRules") {
        params.always_compute_pass_alive_under_suicide_rules = cfg.get_enabled("alwaysComputePassAliveUnderSuicideRules")?;
      } else {
        params.always_compute_pass_alive_under_suicide_rules = enabled_t::Auto;
      }
      if cfg.contains(&format!("excludeTerritoryAdjacentToAtari{idx_str}")) {
        params.exclude_territory_adjacent_to_atari = cfg.get_enabled(&format!("excludeTerritoryAdjacentToAtari{idx_str}"))?;
      } else if cfg.contains("excludeTerritoryAdjacentToAtari") {
        params.exclude_territory_adjacent_to_atari = cfg.get_enabled("excludeTerritoryAdjacentToAtari")?;
      } else {
        params.exclude_territory_adjacent_to_atari = enabled_t::Auto;
      }
      //Controlled by GTP directly, not used in any other mode
      params.avoid_myt_dagger_hack_pla = C_EMPTY;
      if cfg.contains(&format!("wideRootNoise{idx_str}")) {
        params.wide_root_noise = cfg.get_double(&format!("wideRootNoise{idx_str}"), 0.0, 5.0)?;
      } else if cfg.contains("wideRootNoise") {
        params.wide_root_noise = cfg.get_double("wideRootNoise", 0.0, 5.0)?;
      } else {
        params.wide_root_noise = if setup_for == SETUP_FOR_ANALYSIS { Setup::DEFAULT_ANALYSIS_WIDE_ROOT_NOISE } else { 0.00 };
      }

      if cfg.contains(&format!("enablePassingHacks{idx_str}")) {
        params.enable_passing_hacks = cfg.get_bool(&format!("enablePassingHacks{idx_str}"))?;
      } else if cfg.contains("enablePassingHacks") {
        params.enable_passing_hacks = cfg.get_bool("enablePassingHacks")?;
      } else {
        params.enable_passing_hacks = if setup_for == SETUP_FOR_GTP || setup_for == SETUP_FOR_ANALYSIS { true } else { false };
      }

      if cfg.contains(&format!("enableMorePassingHacks{idx_str}")) {
        params.enable_more_passing_hacks = cfg.get_bool(&format!("enableMorePassingHacks{idx_str}"))?;
      } else if cfg.contains("enableMorePassingHacks") {
        params.enable_more_passing_hacks = cfg.get_bool("enableMorePassingHacks")?;
      } else {
        params.enable_more_passing_hacks = if setup_for == SETUP_FOR_GTP || setup_for == SETUP_FOR_ANALYSIS { true } else { false };
      }

      if cfg.contains(&format!("playoutDoublingAdvantage{idx_str}")) {
        params.playout_doubling_advantage = cfg.get_double(&format!("playoutDoublingAdvantage{idx_str}"), -3.0, 3.0)?;
      } else if cfg.contains("playoutDoublingAdvantage") {
        params.playout_doubling_advantage = cfg.get_double("playoutDoublingAdvantage", -3.0, 3.0)?;
      } else {
        params.playout_doubling_advantage = 0.0;
      }
      if cfg.contains(&format!("playoutDoublingAdvantagePla{idx_str}")) {
        params.playout_doubling_advantage_pla = parse_player("playoutDoublingAdvantagePla", &cfg.get_string(&format!("playoutDoublingAdvantagePla{idx_str}"))?)?;
      } else if cfg.contains("playoutDoublingAdvantagePla") {
        params.playout_doubling_advantage_pla = parse_player("playoutDoublingAdvantagePla", &cfg.get_string("playoutDoublingAdvantagePla")?)?;
      } else {
        params.playout_doubling_advantage_pla = C_EMPTY;
      }

      if cfg.contains(&format!("avoidRepeatedPatternUtility{idx_str}")) {
        params.avoid_repeated_pattern_utility = cfg.get_double(&format!("avoidRepeatedPatternUtility{idx_str}"), -3.0, 3.0)?;
      } else if cfg.contains("avoidRepeatedPatternUtility") {
        params.avoid_repeated_pattern_utility = cfg.get_double("avoidRepeatedPatternUtility", -3.0, 3.0)?;
      } else {
        params.avoid_repeated_pattern_utility = 0.0;
      }

      if cfg.contains(&format!("nnPolicyTemperature{idx_str}")) {
        params.nn_policy_temperature = cfg.get_float(&format!("nnPolicyTemperature{idx_str}"), 0.01f32, 5.0f32)?;
      } else if cfg.contains("nnPolicyTemperature") {
        params.nn_policy_temperature = cfg.get_float("nnPolicyTemperature", 0.01f32, 5.0f32)?;
      } else {
        params.nn_policy_temperature = 1.0f32;
      }

      if cfg.contains(&format!("antiMirror{idx_str}")) {
        params.anti_mirror = cfg.get_bool(&format!("antiMirror{idx_str}"))?;
      } else if cfg.contains("antiMirror") {
        params.anti_mirror = cfg.get_bool("antiMirror")?;
      } else {
        params.anti_mirror = false;
      }

      if cfg.contains(&format!("ignorePreRootHistory{idx_str}")) {
        params.ignore_pre_root_history = cfg.get_bool(&format!("ignorePreRootHistory{idx_str}"))?;
      } else if cfg.contains("ignorePreRootHistory") {
        params.ignore_pre_root_history = cfg.get_bool("ignorePreRootHistory")?;
      } else {
        params.ignore_pre_root_history = if setup_for == SETUP_FOR_ANALYSIS { Setup::DEFAULT_ANALYSIS_IGNORE_PRE_ROOT_HISTORY } else { false };
      }
      if cfg.contains(&format!("ignoreAllHistory{idx_str}")) {
        params.ignore_all_history = cfg.get_bool(&format!("ignoreAllHistory{idx_str}"))?;
      } else if cfg.contains("ignoreAllHistory") {
        params.ignore_all_history = cfg.get_bool("ignoreAllHistory")?;
      } else {
        params.ignore_all_history = false;
      }

      if cfg.contains(&format!("subtreeValueBiasFactor{idx_str}")) {
        params.subtree_value_bias_factor = cfg.get_double(&format!("subtreeValueBiasFactor{idx_str}"), 0.0, 1.0)?;
      } else if cfg.contains("subtreeValueBiasFactor") {
        params.subtree_value_bias_factor = cfg.get_double("subtreeValueBiasFactor", 0.0, 1.0)?;
      } else {
        params.subtree_value_bias_factor = 0.45;
      }
      if cfg.contains(&format!("subtreeValueBiasFreeProp{idx_str}")) {
        params.subtree_value_bias_free_prop = cfg.get_double(&format!("subtreeValueBiasFreeProp{idx_str}"), 0.0, 1.0)?;
      } else if cfg.contains("subtreeValueBiasFreeProp") {
        params.subtree_value_bias_free_prop = cfg.get_double("subtreeValueBiasFreeProp", 0.0, 1.0)?;
      } else {
        params.subtree_value_bias_free_prop = 0.8;
      }
      if cfg.contains(&format!("subtreeValueBiasWeightExponent{idx_str}")) {
        params.subtree_value_bias_weight_exponent = cfg.get_double(&format!("subtreeValueBiasWeightExponent{idx_str}"), 0.0, 1.0)?;
      } else if cfg.contains("subtreeValueBiasWeightExponent") {
        params.subtree_value_bias_weight_exponent = cfg.get_double("subtreeValueBiasWeightExponent", 0.0, 1.0)?;
      } else {
        params.subtree_value_bias_weight_exponent = 0.85;
      }

      if cfg.contains(&format!("useEvalCache{idx_str}")) {
        params.use_eval_cache = cfg.get_bool(&format!("useEvalCache{idx_str}"))?;
      } else if cfg.contains("useEvalCache") {
        params.use_eval_cache = cfg.get_bool("useEvalCache")?;
      } else {
        params.use_eval_cache = false;
      }

      if cfg.contains(&format!("evalCacheMinVisits{idx_str}")) {
        params.eval_cache_min_visits = cfg.get_int64(&format!("evalCacheMinVisits{idx_str}"), 1i64, 1i64 << 50)?;
      } else if cfg.contains("evalCacheMinVisits") {
        params.eval_cache_min_visits = cfg.get_int64("evalCacheMinVisits", 1i64, 1i64 << 50)?;
      } else {
        params.eval_cache_min_visits = 100;
      }

      if cfg.contains(&format!("nodeTableShardsPowerOfTwo{idx_str}")) {
        params.node_table_shards_power_of_two = cfg.get_int(&format!("nodeTableShardsPowerOfTwo{idx_str}"), 8, 24)?;
      } else if cfg.contains("nodeTableShardsPowerOfTwo") {
        params.node_table_shards_power_of_two = cfg.get_int("nodeTableShardsPowerOfTwo", 8, 24)?;
      } else {
        params.node_table_shards_power_of_two = 16;
      }
      if cfg.contains(&format!("numVirtualLossesPerThread{idx_str}")) {
        params.num_virtual_losses_per_thread = cfg.get_double(&format!("numVirtualLossesPerThread{idx_str}"), 0.01, 1000.0)?;
      } else if cfg.contains("numVirtualLossesPerThread") {
        params.num_virtual_losses_per_thread = cfg.get_double("numVirtualLossesPerThread", 0.01, 1000.0)?;
      } else {
        params.num_virtual_losses_per_thread = 1.0;
      }

      if cfg.contains(&format!("treeReuseCarryOverTimeFactor{idx_str}")) {
        params.tree_reuse_carry_over_time_factor = cfg.get_double(&format!("treeReuseCarryOverTimeFactor{idx_str}"), 0.0, 1.0)?;
      } else if cfg.contains("treeReuseCarryOverTimeFactor") {
        params.tree_reuse_carry_over_time_factor = cfg.get_double("treeReuseCarryOverTimeFactor", 0.0, 1.0)?;
      } else {
        params.tree_reuse_carry_over_time_factor = 0.0;
      }
      if cfg.contains(&format!("overallocateTimeFactor{idx_str}")) {
        params.overallocate_time_factor = cfg.get_double(&format!("overallocateTimeFactor{idx_str}"), 0.01, 100.0)?;
      } else if cfg.contains("overallocateTimeFactor") {
        params.overallocate_time_factor = cfg.get_double("overallocateTimeFactor", 0.01, 100.0)?;
      } else {
        params.overallocate_time_factor = 1.0;
      }
      if cfg.contains(&format!("midgameTimeFactor{idx_str}")) {
        params.midgame_time_factor = cfg.get_double(&format!("midgameTimeFactor{idx_str}"), 0.01, 100.0)?;
      } else if cfg.contains("midgameTimeFactor") {
        params.midgame_time_factor = cfg.get_double("midgameTimeFactor", 0.01, 100.0)?;
      } else {
        params.midgame_time_factor = 1.0;
      }
      if cfg.contains(&format!("midgameTurnPeakTime{idx_str}")) {
        params.midgame_turn_peak_time = cfg.get_double(&format!("midgameTurnPeakTime{idx_str}"), 0.0, 1000.0)?;
      } else if cfg.contains("midgameTurnPeakTime") {
        params.midgame_turn_peak_time = cfg.get_double("midgameTurnPeakTime", 0.0, 1000.0)?;
      } else {
        params.midgame_turn_peak_time = 130.0;
      }
      if cfg.contains(&format!("endgameTurnTimeDecay{idx_str}")) {
        params.endgame_turn_time_decay = cfg.get_double(&format!("endgameTurnTimeDecay{idx_str}"), 0.0, 1000.0)?;
      } else if cfg.contains("endgameTurnTimeDecay") {
        params.endgame_turn_time_decay = cfg.get_double("endgameTurnTimeDecay", 0.0, 1000.0)?;
      } else {
        params.endgame_turn_time_decay = 100.0;
      }
      if cfg.contains(&format!("obviousMovesTimeFactor{idx_str}")) {
        params.obvious_moves_time_factor = cfg.get_double(&format!("obviousMovesTimeFactor{idx_str}"), 0.01, 1.0)?;
      } else if cfg.contains("obviousMovesTimeFactor") {
        params.obvious_moves_time_factor = cfg.get_double("obviousMovesTimeFactor", 0.01, 1.0)?;
      } else {
        params.obvious_moves_time_factor = 1.0;
      }
      if cfg.contains(&format!("obviousMovesPolicyEntropyTolerance{idx_str}")) {
        params.obvious_moves_policy_entropy_tolerance = cfg.get_double(&format!("obviousMovesPolicyEntropyTolerance{idx_str}"), 0.001, 2.0)?;
      } else if cfg.contains("obviousMovesPolicyEntropyTolerance") {
        params.obvious_moves_policy_entropy_tolerance = cfg.get_double("obviousMovesPolicyEntropyTolerance", 0.001, 2.0)?;
      } else {
        params.obvious_moves_policy_entropy_tolerance = 0.30;
      }
      if cfg.contains(&format!("obviousMovesPolicySurpriseTolerance{idx_str}")) {
        params.obvious_moves_policy_surprise_tolerance = cfg.get_double(&format!("obviousMovesPolicySurpriseTolerance{idx_str}"), 0.001, 2.0)?;
      } else if cfg.contains("obviousMovesPolicySurpriseTolerance") {
        params.obvious_moves_policy_surprise_tolerance = cfg.get_double("obviousMovesPolicySurpriseTolerance", 0.001, 2.0)?;
      } else {
        params.obvious_moves_policy_surprise_tolerance = 0.15;
      }
      if cfg.contains(&format!("futileVisitsThreshold{idx_str}")) {
        params.futile_visits_threshold = cfg.get_double(&format!("futileVisitsThreshold{idx_str}"), 0.01, 1.0)?;
      } else if cfg.contains("futileVisitsThreshold") {
        params.futile_visits_threshold = cfg.get_double("futileVisitsThreshold", 0.01, 1.0)?;
      } else {
        params.futile_visits_threshold = 0.0;
      }

      // This does NOT report an error under throwHumanParsingError like the parameters below that expect a second model
      // because the user might be providing the human model as the MAIN model. In which case humanSLProfile is still a
      // valid param but the others are not.
      if setup_for != SETUP_FOR_DISTRIBUTED {
        let mut human_sl_profile_name: String = String::new();
        if cfg.contains(&format!("humanSLProfile{idx_str}")) {
          human_sl_profile_name = cfg.get_string(&format!("humanSLProfile{idx_str}"))?;
        } else if cfg.contains("humanSLProfile") {
          human_sl_profile_name = cfg.get_string("humanSLProfile")?;
        }
        params.human_sl_profile = SGFMetadata::get_profile(&human_sl_profile_name)?;
      }

      // PORT NOTE: the throwHumanParsingError lambda is the file-level fn throw_human_parsing_error, which
      // returns the error for the caller to `return Err(..)` (the lambda throws).

      if !has_human_model && cfg.contains(&format!("humanSLCpuctExploration{idx_str}")) {
        return Err(throw_human_parsing_error(&format!("humanSLCpuctExploration{idx_str}")));
      } else if !has_human_model && cfg.contains("humanSLCpuctExploration") {
        return Err(throw_human_parsing_error("humanSLCpuctExploration"));
      } else if cfg.contains(&format!("humanSLCpuctExploration{idx_str}")) {
        params.human_sl_cpuct_exploration = cfg.get_double(&format!("humanSLCpuctExploration{idx_str}"), 0.0, 1000.0)?;
      } else if cfg.contains("humanSLCpuctExploration") {
        params.human_sl_cpuct_exploration = cfg.get_double("humanSLCpuctExploration", 0.0, 1000.0)?;
      } else {
        params.human_sl_cpuct_exploration = 1.0;
      }
      if !has_human_model && cfg.contains(&format!("humanSLCpuctPermanent{idx_str}")) {
        return Err(throw_human_parsing_error(&format!("humanSLCpuctPermanent{idx_str}")));
      } else if !has_human_model && cfg.contains("humanSLCpuctPermanent") {
        return Err(throw_human_parsing_error("humanSLCpuctPermanent"));
      } else if cfg.contains(&format!("humanSLCpuctPermanent{idx_str}")) {
        params.human_sl_cpuct_permanent = cfg.get_double(&format!("humanSLCpuctPermanent{idx_str}"), 0.0, 1000.0)?;
      } else if cfg.contains("humanSLCpuctPermanent") {
        params.human_sl_cpuct_permanent = cfg.get_double("humanSLCpuctPermanent", 0.0, 1000.0)?;
      } else {
        params.human_sl_cpuct_permanent = 0.0;
      }
      if !has_human_model && cfg.contains(&format!("humanSLRootExploreProbWeightless{idx_str}")) {
        return Err(throw_human_parsing_error(&format!("humanSLRootExploreProbWeightless{idx_str}")));
      } else if !has_human_model && cfg.contains("humanSLRootExploreProbWeightless") {
        return Err(throw_human_parsing_error("humanSLRootExploreProbWeightless"));
      } else if cfg.contains(&format!("humanSLRootExploreProbWeightless{idx_str}")) {
        params.human_sl_root_explore_prob_weightless = cfg.get_double(&format!("humanSLRootExploreProbWeightless{idx_str}"), 0.0, 1.0)?;
      } else if cfg.contains("humanSLRootExploreProbWeightless") {
        params.human_sl_root_explore_prob_weightless = cfg.get_double("humanSLRootExploreProbWeightless", 0.0, 1.0)?;
      } else {
        params.human_sl_root_explore_prob_weightless = 0.0;
      }
      if !has_human_model && cfg.contains(&format!("humanSLRootExploreProbWeightful{idx_str}")) {
        return Err(throw_human_parsing_error(&format!("humanSLRootExploreProbWeightful{idx_str}")));
      } else if !has_human_model && cfg.contains("humanSLRootExploreProbWeightful") {
        return Err(throw_human_parsing_error("humanSLRootExploreProbWeightful"));
      } else if cfg.contains(&format!("humanSLRootExploreProbWeightful{idx_str}")) {
        params.human_sl_root_explore_prob_weightful = cfg.get_double(&format!("humanSLRootExploreProbWeightful{idx_str}"), 0.0, 1.0)?;
      } else if cfg.contains("humanSLRootExploreProbWeightful") {
        params.human_sl_root_explore_prob_weightful = cfg.get_double("humanSLRootExploreProbWeightful", 0.0, 1.0)?;
      } else {
        params.human_sl_root_explore_prob_weightful = 0.0;
      }
      if !has_human_model && cfg.contains(&format!("humanSLPlaExploreProbWeightless{idx_str}")) {
        return Err(throw_human_parsing_error(&format!("humanSLPlaExploreProbWeightless{idx_str}")));
      } else if !has_human_model && cfg.contains("humanSLPlaExploreProbWeightless") {
        return Err(throw_human_parsing_error("humanSLPlaExploreProbWeightless"));
      } else if cfg.contains(&format!("humanSLPlaExploreProbWeightless{idx_str}")) {
        params.human_sl_pla_explore_prob_weightless = cfg.get_double(&format!("humanSLPlaExploreProbWeightless{idx_str}"), 0.0, 1.0)?;
      } else if cfg.contains("humanSLPlaExploreProbWeightless") {
        params.human_sl_pla_explore_prob_weightless = cfg.get_double("humanSLPlaExploreProbWeightless", 0.0, 1.0)?;
      } else {
        params.human_sl_pla_explore_prob_weightless = 0.0;
      }
      if !has_human_model && cfg.contains(&format!("humanSLPlaExploreProbWeightful{idx_str}")) {
        return Err(throw_human_parsing_error(&format!("humanSLPlaExploreProbWeightful{idx_str}")));
      } else if !has_human_model && cfg.contains("humanSLPlaExploreProbWeightful") {
        return Err(throw_human_parsing_error("humanSLPlaExploreProbWeightful"));
      } else if cfg.contains(&format!("humanSLPlaExploreProbWeightful{idx_str}")) {
        params.human_sl_pla_explore_prob_weightful = cfg.get_double(&format!("humanSLPlaExploreProbWeightful{idx_str}"), 0.0, 1.0)?;
      } else if cfg.contains("humanSLPlaExploreProbWeightful") {
        params.human_sl_pla_explore_prob_weightful = cfg.get_double("humanSLPlaExploreProbWeightful", 0.0, 1.0)?;
      } else {
        params.human_sl_pla_explore_prob_weightful = 0.0;
      }
      if !has_human_model && cfg.contains(&format!("humanSLOppExploreProbWeightless{idx_str}")) {
        return Err(throw_human_parsing_error(&format!("humanSLOppExploreProbWeightless{idx_str}")));
      } else if !has_human_model && cfg.contains("humanSLOppExploreProbWeightless") {
        return Err(throw_human_parsing_error("humanSLOppExploreProbWeightless"));
      } else if cfg.contains(&format!("humanSLOppExploreProbWeightless{idx_str}")) {
        params.human_sl_opp_explore_prob_weightless = cfg.get_double(&format!("humanSLOppExploreProbWeightless{idx_str}"), 0.0, 1.0)?;
      } else if cfg.contains("humanSLOppExploreProbWeightless") {
        params.human_sl_opp_explore_prob_weightless = cfg.get_double("humanSLOppExploreProbWeightless", 0.0, 1.0)?;
      } else {
        params.human_sl_opp_explore_prob_weightless = 0.0;
      }
      if !has_human_model && cfg.contains(&format!("humanSLOppExploreProbWeightful{idx_str}")) {
        return Err(throw_human_parsing_error(&format!("humanSLOppExploreProbWeightful{idx_str}")));
      } else if !has_human_model && cfg.contains("humanSLOppExploreProbWeightful") {
        return Err(throw_human_parsing_error("humanSLOppExploreProbWeightful"));
      } else if cfg.contains(&format!("humanSLOppExploreProbWeightful{idx_str}")) {
        params.human_sl_opp_explore_prob_weightful = cfg.get_double(&format!("humanSLOppExploreProbWeightful{idx_str}"), 0.0, 1.0)?;
      } else if cfg.contains("humanSLOppExploreProbWeightful") {
        params.human_sl_opp_explore_prob_weightful = cfg.get_double("humanSLOppExploreProbWeightful", 0.0, 1.0)?;
      } else {
        params.human_sl_opp_explore_prob_weightful = 0.0;
      }
      if !has_human_model && cfg.contains(&format!("humanSLChosenMoveProp{idx_str}")) {
        return Err(throw_human_parsing_error(&format!("humanSLChosenMoveProp{idx_str}")));
      } else if !has_human_model && cfg.contains("humanSLChosenMoveProp") {
        return Err(throw_human_parsing_error("humanSLChosenMoveProp"));
      } else if cfg.contains(&format!("humanSLChosenMoveProp{idx_str}")) {
        params.human_sl_chosen_move_prop = cfg.get_double(&format!("humanSLChosenMoveProp{idx_str}"), 0.0, 1.0)?;
      } else if cfg.contains("humanSLChosenMoveProp") {
        params.human_sl_chosen_move_prop = cfg.get_double("humanSLChosenMoveProp", 0.0, 1.0)?;
      } else {
        params.human_sl_chosen_move_prop = 0.0;
      }
      if !has_human_model && cfg.contains(&format!("humanSLChosenMoveIgnorePass{idx_str}")) {
        return Err(throw_human_parsing_error(&format!("humanSLChosenMoveIgnorePass{idx_str}")));
      } else if !has_human_model && cfg.contains("humanSLChosenMoveIgnorePass") {
        return Err(throw_human_parsing_error("humanSLChosenMoveIgnorePass"));
      } else if cfg.contains(&format!("humanSLChosenMoveIgnorePass{idx_str}")) {
        params.human_sl_chosen_move_ignore_pass = cfg.get_bool(&format!("humanSLChosenMoveIgnorePass{idx_str}"))?;
      } else if cfg.contains("humanSLChosenMoveIgnorePass") {
        params.human_sl_chosen_move_ignore_pass = cfg.get_bool("humanSLChosenMoveIgnorePass")?;
      } else {
        params.human_sl_chosen_move_ignore_pass = false;
      }
      if !has_human_model && cfg.contains(&format!("humanSLChosenMovePiklLambda{idx_str}")) {
        return Err(throw_human_parsing_error(&format!("humanSLChosenMovePiklLambda{idx_str}")));
      } else if !has_human_model && cfg.contains("humanSLChosenMovePiklLambda") {
        return Err(throw_human_parsing_error("humanSLChosenMovePiklLambda"));
      } else if cfg.contains(&format!("humanSLChosenMovePiklLambda{idx_str}")) {
        params.human_sl_chosen_move_pikl_lambda = cfg.get_double(&format!("humanSLChosenMovePiklLambda{idx_str}"), 0.0, 1000000000.0)?;
      } else if cfg.contains("humanSLChosenMovePiklLambda") {
        params.human_sl_chosen_move_pikl_lambda = cfg.get_double("humanSLChosenMovePiklLambda", 0.0, 1000000000.0)?;
      } else {
        params.human_sl_chosen_move_pikl_lambda = 1000000000.0;
      }

      //On distributed, tolerate reading mutexPoolSize since older version configs use it.
      if setup_for == SETUP_FOR_DISTRIBUTED {
        cfg.mark_all_keys_used_with_prefix("mutexPoolSize");
      }

      paramss.push(params);
    }

    Ok(paramss)
  }

  /// upstream: Setup::maybeWarnHumanSLParams (setup.cpp)
  /// PORT NOTE: `Logger* logger` -> `logger: bool` (whether upstream passed a Logger, i.e. whether to
  /// log through the log crate); `NNEvaluator*` -> Option<&NNEvaluator>.
  pub fn maybe_warn_human_sl_params(
    params: &SearchParams,
    nn_eval: Option<&NNEvaluator>,
    human_eval: Option<&NNEvaluator>,
    out: &mut dyn std::fmt::Write,
    logger: bool,
  ) -> bool {
    if params.human_sl_profile.initialized {
      let has_any_sgf_meta_use: bool =
        (nn_eval.is_some() && nn_eval.expect("checked").requires_sgf_metadata()) || (human_eval.is_some() && human_eval.expect("checked").requires_sgf_metadata());
      if !has_any_sgf_meta_use {
        let mut model_names: String = String::new();
        if let Some(nn_eval) = nn_eval {
          model_names += &nn_eval.get_model_name();
        }
        if let Some(human_eval) = human_eval {
          if !model_names.is_empty() {
            model_names += " and ";
          }
          model_names += &human_eval.get_model_name();
        }
        if logger {
          log::info!("{}", "WARNING: humanSLProfile is specified as config param but model(s) don't use it: ".to_string() + &model_names);
        }
        writeln!(out, "WARNING: humanSLProfile is specified as config param but model(s) don't use it: {}", model_names).unwrap();
        return true;
      }
    }
    false
  }

  /// upstream: Setup::parseReportAnalysisWinrates (setup.cpp)
  pub fn parse_report_analysis_winrates(cfg: &mut ConfigParser, default_perspective: Player) -> Result<Player, KataError> {
    if !cfg.contains("reportAnalysisWinratesAs") {
      return Ok(default_perspective);
    }

    let s_orig: String = cfg.get_string("reportAnalysisWinratesAs")?;
    let s: String = Global::to_lower(&s_orig);
    if s == "b" || s == "black" {
      return Ok(P_BLACK);
    } else if s == "w" || s == "white" {
      return Ok(P_WHITE);
    } else if s == "sidetomove" {
      return Ok(C_EMPTY);
    }

    Err(KataError::StringError("Could not parse config value for reportAnalysisWinratesAs: ".to_string() + &s_orig))
  }

  /// upstream: Setup::loadSingleRules (setup.cpp)
  //Komi is just set to 7.5 and is not read in from cfg
  pub fn load_single_rules(cfg: &mut ConfigParser, load_komi: bool) -> Result<Rules, KataError> {
    let mut rules: Rules = Rules::new();

    if cfg.contains("rules") {
      if cfg.contains("koRule") {
        return Err(KataError::StringError("Cannot both specify 'rules' and individual rules like koRule".to_string()));
      }
      if cfg.contains("scoringRule") {
        return Err(KataError::StringError("Cannot both specify 'rules' and individual rules like scoringRule".to_string()));
      }
      if cfg.contains("multiStoneSuicideLegal") {
        return Err(KataError::StringError("Cannot both specify 'rules' and individual rules like multiStoneSuicideLegal".to_string()));
      }
      if cfg.contains("hasButton") {
        return Err(KataError::StringError("Cannot both specify 'rules' and individual rules like hasButton".to_string()));
      }
      if cfg.contains("taxRule") {
        return Err(KataError::StringError("Cannot both specify 'rules' and individual rules like taxRule".to_string()));
      }
      if cfg.contains("whiteHandicapBonus") {
        return Err(KataError::StringError("Cannot both specify 'rules' and individual rules like whiteHandicapBonus".to_string()));
      }
      if cfg.contains("friendlyPassOk") {
        return Err(KataError::StringError("Cannot both specify 'rules' and individual rules like friendlyPassOk".to_string()));
      }
      if cfg.contains("whiteBonusPerHandicapStone") {
        return Err(KataError::StringError("Cannot both specify 'rules' and individual rules like whiteBonusPerHandicapStone".to_string()));
      }

      rules = Rules::parse_rules(&cfg.get_string("rules")?)?;
    } else {
      let ko_rule: String = cfg.get_string_possibles("koRule", &Rules::ko_rule_strings())?;
      let scoring_rule: String = cfg.get_string_possibles("scoringRule", &Rules::scoring_rule_strings())?;
      let multi_stone_suicide_legal: bool = cfg.get_bool("multiStoneSuicideLegal")?;
      let has_button: bool = if cfg.contains("hasButton") { cfg.get_bool("hasButton")? } else { false };
      let komi: f32 = 7.5f32;

      rules.ko_rule = Rules::parse_ko_rule(&ko_rule)?;
      rules.scoring_rule = Rules::parse_scoring_rule(&scoring_rule)?;
      rules.multi_stone_suicide_legal = multi_stone_suicide_legal;
      rules.has_button = has_button;
      rules.komi = komi;

      if cfg.contains("taxRule") {
        let tax_rule: String = cfg.get_string_possibles("taxRule", &Rules::tax_rule_strings())?;
        rules.tax_rule = Rules::parse_tax_rule(&tax_rule)?;
      } else {
        rules.tax_rule = if rules.scoring_rule == Rules::SCORING_TERRITORY { Rules::TAX_SEKI } else { Rules::TAX_NONE };
      }

      if rules.has_button && rules.scoring_rule != Rules::SCORING_AREA {
        return Err(KataError::StringError("Config specifies hasButton=true on a scoring system other than AREA".to_string()));
      }

      //Also handles parsing of legacy option whiteBonusPerHandicapStone
      if cfg.contains("whiteBonusPerHandicapStone") && cfg.contains("whiteHandicapBonus") {
        return Err(KataError::StringError("May specify only one of whiteBonusPerHandicapStone and whiteHandicapBonus in config".to_string()));
      } else if cfg.contains("whiteHandicapBonus") {
        rules.white_handicap_bonus_rule =
          Rules::parse_white_handicap_bonus_rule(&cfg.get_string_possibles("whiteHandicapBonus", &Rules::white_handicap_bonus_rule_strings())?)?;
      } else if cfg.contains("whiteBonusPerHandicapStone") {
        let white_bonus_per_handicap_stone: i32 = cfg.get_int("whiteBonusPerHandicapStone", 0, 1)?;
        if white_bonus_per_handicap_stone == 0 {
          rules.white_handicap_bonus_rule = Rules::WHB_ZERO;
        } else {
          rules.white_handicap_bonus_rule = Rules::WHB_N;
        }
      } else {
        rules.white_handicap_bonus_rule = Rules::WHB_ZERO;
      }

      if cfg.contains("friendlyPassOk") {
        rules.friendly_pass_ok = cfg.get_bool("friendlyPassOk")?;
      }

      //Drop default komi to 6.5 for territory rules, and to 7.0 for button
      if rules.scoring_rule == Rules::SCORING_TERRITORY {
        rules.komi = 6.5f32;
      } else if rules.has_button {
        rules.komi = 7.0f32;
      }
    }

    if load_komi {
      rules.komi = cfg.get_float("komi", Rules::MIN_USER_KOMI, Rules::MAX_USER_KOMI)?;
    }

    Ok(rules)
  }

  /// upstream: Setup::loadDefaultBoardXYSize (setup.cpp)
  //Returns true if the user's config specified the size, false if it did not. If false, does not set defaultBoardXSizeRet or defaultBoardYSizeRet.
  /// PORT NOTE: `Logger&` dropped (log crate); throws -> Result.
  pub fn load_default_board_xy_size(cfg: &mut ConfigParser, default_board_x_size_ret: &mut i32, default_board_y_size_ret: &mut i32) -> Result<bool, KataError> {
    let default_board_x_size: i32 = if cfg.contains("defaultBoardXSize") {
      cfg.get_int("defaultBoardXSize", 2, Board::MAX_LEN)?
    } else if cfg.contains("defaultBoardSize") {
      cfg.get_int("defaultBoardSize", 2, Board::MAX_LEN)?
    } else {
      -1
    };
    let default_board_y_size: i32 = if cfg.contains("defaultBoardYSize") {
      cfg.get_int("defaultBoardYSize", 2, Board::MAX_LEN)?
    } else if cfg.contains("defaultBoardSize") {
      cfg.get_int("defaultBoardSize", 2, Board::MAX_LEN)?
    } else {
      -1
    };
    if (default_board_x_size == -1) != (default_board_y_size == -1) {
      log::info!("Warning: Config specified only one of defaultBoardXSize or defaultBoardYSize and no other board size parameter, ignoring it");
    }

    if default_board_x_size == -1 || default_board_y_size == -1 {
      return Ok(false);
    }
    *default_board_x_size_ret = default_board_x_size;
    *default_board_y_size_ret = default_board_y_size;
    Ok(true)
  }

  /// upstream: Setup::getMutexKeySets (setup.cpp)
  //Get sets of options that are mutually exclusive. Intended for use in configParser
  pub fn get_mutex_key_sets() -> Vec<(BTreeSet<String>, BTreeSet<String>)> {
    let to_set = |v: &[&str]| -> BTreeSet<String> { v.iter().map(|s| s.to_string()).collect() };
    let mutex_key_sets: Vec<(BTreeSet<String>, BTreeSet<String>)> = vec![(
      to_set(&["rules"]),
      to_set(&["koRule", "scoringRule", "multiStoneSuicideLegal", "taxRule", "hasButton", "whiteBonusPerHandicapStone", "friendlyPassOk", "whiteHandicapBonus"]),
    )];
    mutex_key_sets
  }

  /// upstream: Setup::loadAvoidSgfPatternBonusTables (setup.cpp)
  //Load pattern bonus tables that avoid repeating moves that the user supplied in external sgfs
  /// PORT NOTE: `Logger&` dropped; throws -> Result; `unique_ptr<PatternBonusTable>` -> Option<Box<..>>.
  pub fn load_avoid_sgf_pattern_bonus_tables(cfg: &mut ConfigParser) -> Result<Vec<Option<Box<PatternBonusTable>>>, KataError> {
    let mut num_bots: i32 = 1;
    if cfg.contains("numBots") {
      num_bots = cfg.get_int("numBots", 1, Setup::MAX_BOT_PARAMS_FROM_CFG)?;
    }

    let mut tables: Vec<Option<Box<PatternBonusTable>>> = Vec::new();
    for i in 0..num_bots {
      //Indexes different bots, such as in a match config
      let idx_str: String = Global::int_to_string(i);

      let mut pattern_bonus_table: Option<Box<PatternBonusTable>> = None;
      // PERF(port): 100000 iterations of string building and containsAny per bot, as upstream.
      for j in 1..100000 {
        //Indexes different sets of params for different sets of files, to combine into one bot.
        let set_str: String = if j == 1 { String::new() } else { Global::int_to_string(j) };
        let prefix: String = "avoidSgf".to_string() + &set_str;

        //Tries to find prefix+suffix+optional index
        //E.g. "avoidSgf"+"PatternUtility"+(optional integer indexing which bot for match)
        // PORT NOTE: captures passed explicitly (aliasing): `cfg` is also borrowed mutably by the getters.
        let contains = |cfg: &ConfigParser, suffix: &str| -> bool { cfg.contains_any(&[prefix.clone() + suffix + &idx_str, prefix.clone() + suffix]) };
        let find = |cfg: &ConfigParser, suffix: &str| -> Result<String, KataError> {
          cfg.first_found_or_fail(&[prefix.clone() + suffix + &idx_str, prefix.clone() + suffix])
        };

        if contains(cfg, "PatternUtility") {
          let penalty: f64 = cfg.get_double(&find(cfg, "PatternUtility")?, -3.0, 3.0)?;
          let lambda: f64 = if contains(cfg, "PatternLambda") { cfg.get_double(&find(cfg, "PatternLambda")?, 0.0, 1.0)? } else { 1.0 };
          let min_turn_number: i32 =
            if contains(cfg, "PatternMinTurnNumber") { cfg.get_int(&find(cfg, "PatternMinTurnNumber")?, 0, 1000000)? } else { 0 };
          let max_files: u64 = if contains(cfg, "PatternMaxFiles") { cfg.get_int(&find(cfg, "PatternMaxFiles")?, 1, 1000000)? as u64 } else { 1000000 };
          let allowed_player_names: Vec<String> =
            if contains(cfg, "PatternAllowedNames") { cfg.get_strings_non_empty_trim(&find(cfg, "PatternAllowedNames")?)? } else { Vec::new() };
          let sgf_dirs: Vec<String> = cfg.get_strings(&find(cfg, "PatternDirs")?)?;
          if pattern_bonus_table.is_none() {
            pattern_bonus_table = Some(Box::new(PatternBonusTable::new()));
          }
          let log_source: String = "bot ".to_string() + &idx_str;
          // TODO(port): IO/M8: `patternBonusTable->avoidRepeatedSgfMoves(sgfDirs,penalty,lambda,minTurnNumber,maxFiles,
          // allowedPlayerNames,logger,logSource);` collects and parses SGF files (FileHelpers, dataio/sgf: not ported,
          // core does no file IO), so a configured avoid-SGF set is an error here, after upstream's key checks.
          let _ = (penalty, lambda, min_turn_number, max_files, allowed_player_names, log_source, &pattern_bonus_table);
          return Err(KataError::ConfigParsingError(
            "avoidSgf".to_string()
              + &set_str
              + "PatternDirs ("
              + &sgf_dirs.join(",")
              + "): loading SGF files for pattern bonus tables is not supported in katago-rs (PatternBonusTable::avoidRepeatedSgfMoves, dataio/sgf)",
          ));
        }
      }
      tables.push(pattern_bonus_table);
    }
    Ok(tables)
  }

  // upstream: Setup::saveAutoPatternBonusData, Setup::loadAndPruneAutoPatternBonusTables (setup.cpp)
  // PORT NOTE: not ported (file IO: SGF / pos files; dataio/sgf is not ported, M8). See the manifest notes.
}

/// upstream: parsePlayer (setup.cpp) static
/// PORT NOTE: throws -> Result.
fn parse_player(field: &str, s: &str) -> Result<Player, KataError> {
  let mut pla: Player = C_EMPTY;
  let suc: bool = PlayerIO::try_parse_player(s, &mut pla);
  if !suc {
    return Err(KataError::StringError("Could not parse player in field ".to_string() + field + ", should be BLACK or WHITE"));
  }
  Ok(pla)
}

/// upstream: the `throwHumanParsingError` lambda of Setup::loadParams (setup.cpp)
/// PORT NOTE: returns the ConfigParsingError the lambda throws; the call sites `return Err(..)` it.
fn throw_human_parsing_error(param: &str) -> KataError {
  KataError::ConfigParsingError(
    "Provided parameter ".to_string() + param + " but no human model was specified (e.g -human-model b18c384nbt-humanv0.bin.gz)",
  )
}
