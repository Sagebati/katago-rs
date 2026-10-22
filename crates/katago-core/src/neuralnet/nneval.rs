// Ported from KataGo cpp/neuralnet/nneval.cpp, cpp/neuralnet/nneval.h @ ba938676d7f42d70950b3a535af2466fb642008c
//
// PORT NOTE (PORTING.md §11, DECISIONS.md D6/D8): server threads become `serve` tasks spawned on
// the host's local executor through an injected `Rc<dyn Spawner>`; `ThreadSafeQueue` becomes
// `port::executor::AsyncQueue`; condition variables become `WakerList`s; `bufferMutex`,
// `resultMutex` and the cache's MutexPool are removed (single-threaded) and the counters they
// guarded are `Cell`s. `evaluate`, `averageMultipleSymmetries`, `waitForNextNNEvalIfAny`,
// `spawnServerThreads` and `serve` are async. The `NeuralNet::` backend functions are reached
// through an injected `Rc<dyn NeuralNet>` (nninterface.rs). The `Logger*` is replaced by the log
// crate (ctor param and getLogger dropped).
// Not ported: NNEvaluator::getTrunkSpatialConvDepth (needs the layer descs, which are not ported:
// see desc.rs; nothing outside nneval.cpp calls it).

use std::cell::Cell;
use std::cell::RefCell;
use std::collections::BTreeSet;
use std::rc::Rc;

use crate::core::commontypes::enabled_t;
use crate::core::global::Global;
use crate::core::global::KataError;
use crate::core::hash::Hash128;
use crate::core::rand::Rand;
use crate::game::board::Board;
use crate::game::board::Location;
use crate::game::board::Loc;
use crate::game::board::Player;
use crate::game::board::get_opp;
use crate::game::board::C_EMPTY;
use crate::game::board::P_WHITE;
use crate::game::boardhistory::BoardHistory;
use crate::game::rules::Rules;
use crate::neuralnet::desc::ModelPostProcessParams;
use crate::neuralnet::modelversion::NNModelVersion;
use crate::neuralnet::nninputs::MiscNNInputParams;
use crate::neuralnet::nninputs::NNInputs;
use crate::neuralnet::nninputs::NNOutput;
use crate::neuralnet::nninputs::NNPos;
use crate::neuralnet::nninputs::ScoreValue;
use crate::neuralnet::nninputs::SymmetryHelpers;
use crate::neuralnet::nninterface::ComputeContext;
use crate::neuralnet::nninterface::ComputeHandle;
use crate::neuralnet::nninterface::InputBuffers;
use crate::neuralnet::nninterface::LoadedModel;
use crate::neuralnet::nninterface::NeuralNet;
use crate::neuralnet::sgfmetadata::SGFMetadata;
use crate::port::cpp;
use crate::port::executor::AsyncQueue;
use crate::port::executor::Spawner;
use crate::port::executor::WakerList;
use crate::port::math;
use crate::testAssert;

//-------------------------------------------------------------------------------------

/// upstream: NNCacheTable::Entry (nneval.h)
pub(crate) struct Entry {
  pub(crate) ptr: Option<Rc<NNOutput>>,
}

impl Entry {
  /// upstream: NNCacheTable::Entry::Entry (nneval.cpp)
  pub(crate) fn new() -> Entry {
    Entry { ptr: None }
  }
}

/// upstream: NNCacheTable (nneval.h)
/// PORT NOTE: `mutexPool`/`mutexPoolMask` removed (lock removed, single-threaded); `entries` is a
/// RefCell so get/set/clear take `&self` (called during search).
pub struct NNCacheTable {
  pub(crate) entries: RefCell<Box<[Entry]>>,
  pub(crate) table_size: u64,
  pub(crate) table_mask: u64,
}

//Each thread should allocate and re-use one of these
/// upstream: NNResultBuf (nneval.h)
/// PORT NOTE: shared as `Rc<NNResultBuf>` between the client and the eval queue, so every field is
/// a Cell/RefCell (OWNERSHIP.tsv); `resultMutex` removed, `clientWaitingForResult` is a WakerList.
pub struct NNResultBuf {
  pub client_waiting_for_result: WakerList,
  pub has_result: Cell<bool>,
  pub include_owner_map: Cell<bool>,
  pub board_x_size_for_server: Cell<i32>,
  pub board_y_size_for_server: Cell<i32>,
  pub row_spatial_buf: RefCell<Vec<f32>>,
  pub row_global_buf: RefCell<Vec<f32>>,
  pub row_meta_buf: RefCell<Vec<f32>>,
  pub has_row_meta: Cell<bool>,
  pub result: RefCell<Option<Rc<NNOutput>>>,
  pub error_log_lockout: Cell<bool>, //error flag to restrict log to 1 error to prevent spam
  pub symmetry: Cell<i32>,           //The symmetry to use for this eval
  pub policy_optimism: Cell<f64>,    //The policy optimism to use for this eval
}

//Each server thread should allocate and re-use one of these
/// upstream: NNServerBuf (nneval.h)
pub struct NNServerBuf {
  pub input_buffers: Option<Box<dyn InputBuffers>>,
}

/// upstream: NNEvaluator (nneval.h)
/// PORT NOTE: shared as `Rc<NNEvaluator>` (the serve tasks hold one), so every field written after
/// construction is a Cell/RefCell (OWNERSHIP.tsv), including `numThreads` / `gpuIdxByServerThread`
/// (setNumThreads) and `numServerThreadsEverSpawned`. `serverThreads` (vector<thread*>) becomes the
/// count `server_threads` of spawned serve tasks, plus `num_server_threads_alive` (decremented when
/// a serve task ends, which notifies `server_threads_exited`): awaiting it to reach 0 stands in for
/// join() (D8: killServerThreads' join is deferred to the next spawnServerThreads).
pub struct NNEvaluator {
  pub(crate) model_name: String,
  pub(crate) model_file_name: String,
  pub(crate) nn_x_len: i32,
  pub(crate) nn_y_len: i32,
  pub(crate) require_exact_nn_len: bool,
  pub(crate) policy_size: i32,
  pub(crate) inputs_use_nhwc: bool,
  pub(crate) using_fp16_mode: enabled_t,
  pub(crate) using_nhwc_mode: enabled_t,
  pub(crate) num_threads: Cell<i32>,
  pub(crate) gpu_idx_by_server_thread: RefCell<Vec<i32>>,
  pub(crate) rand_seed: String,
  pub(crate) debug_skip_neural_net: bool,

  pub(crate) compute_context: Option<Box<dyn ComputeContext>>,
  pub(crate) loaded_model: Option<Box<dyn LoadedModel>>,
  pub(crate) nn_cache_table: Option<Box<NNCacheTable>>,

  pub(crate) internal_model_name: String,
  pub(crate) model_version: i32,
  pub(crate) inputs_version: i32,
  pub(crate) num_input_meta_channels: i32,

  pub(crate) post_process_params: ModelPostProcessParams,

  pub(crate) num_server_threads_ever_spawned: Cell<i32>,
  pub(crate) server_threads: Cell<i32>,
  pub(crate) num_server_threads_alive: Cell<i32>,
  /// PORT NOTE: new (OWNERSHIP.tsv, D8): notified by each serve task as it ends; the deferred
  /// join in spawnServerThreads awaits `num_server_threads_alive == 0` on it.
  pub(crate) server_threads_exited: WakerList,

  pub(crate) max_batch_size: i32,

  //Counters for statistics
  pub(crate) m_num_rows_processed: Cell<u64>,
  pub(crate) m_num_batches_processed: Cell<u64>,

  //Everything in this section is protected under bufferMutex--------------------------------------------
  // PORT NOTE: bufferMutex removed (single-threaded).

  pub(crate) is_killed: Cell<bool>,                    //Flag used for killing server threads
  pub(crate) num_server_threads_starting_up: Cell<i32>, //Counter for waiting until server threads are spawned
  pub(crate) main_thread_waiting_for_spawn: WakerList, //Condvar for waiting until server threads are spawned

  pub(crate) server_threads_is_using_fp16: RefCell<Vec<i32>>,

  pub(crate) num_ongoing_evals: Cell<i32>,  //Current number of ongoing evals.
  pub(crate) num_waiting_evals: Cell<i32>,  //Current number of things waiting for finish.
  pub(crate) num_evals_to_awaken: Cell<i32>, //Current number of things waitingForFinish that should be woken up. Used to avoid spurious wakeups.
  pub(crate) waiting_for_finish: WakerList, //Condvar for waiting for at least one ongoing eval to finish.

  //-------------------------------------------------------------------------------------------------

  //Randomization settings for symmetries
  pub(crate) current_do_randomize: Cell<bool>,
  pub(crate) current_default_symmetry: Cell<i32>,
  //Modifiable batch size smaller than maxBatchSize
  pub(crate) current_batch_size: Cell<i32>,

  //Queued up requests
  pub(crate) query_queue: AsyncQueue<Rc<NNResultBuf>>,

  /// PORT NOTE: new (OWNERSHIP.tsv): the backend standing in for the `NeuralNet::` free functions.
  pub(crate) backend: Rc<dyn NeuralNet>,
  /// PORT NOTE: new (OWNERSHIP.tsv): the host's executor, used by spawnServerThreads.
  pub(crate) spawner: Rc<dyn Spawner>,
}

//-------------------------------------------------------------------------------------

impl NNResultBuf {
  /// upstream: NNResultBuf::NNResultBuf (nneval.cpp)
  pub fn new() -> NNResultBuf {
    NNResultBuf {
      client_waiting_for_result: WakerList::new(),
      has_result: Cell::new(false),
      include_owner_map: Cell::new(false),
      board_x_size_for_server: Cell::new(0),
      board_y_size_for_server: Cell::new(0),
      row_spatial_buf: RefCell::new(Vec::new()),
      row_global_buf: RefCell::new(Vec::new()),
      row_meta_buf: RefCell::new(Vec::new()),
      has_row_meta: Cell::new(false),
      result: RefCell::new(None),
      error_log_lockout: Cell::new(false),
      // If no symmetry is specified, it will use default or random based on config.
      symmetry: Cell::new(NNInputs::SYMMETRY_NOTSPECIFIED),
      policy_optimism: Cell::new(0.0),
    }
  }
}

//-------------------------------------------------------------------------------------

impl NNServerBuf {
  /// upstream: NNServerBuf::NNServerBuf (nneval.cpp)
  /// freeInputBuffers in the destructor is Drop of the Box.
  pub fn new(nn_eval: &NNEvaluator, model: Option<&dyn LoadedModel>) -> NNServerBuf {
    let mut input_buffers: Option<Box<dyn InputBuffers>> = None;
    let max_batch_size: i32 = nn_eval.get_max_batch_size();
    if let Some(model) = model {
      input_buffers = Some(nn_eval.backend.create_input_buffers(model, max_batch_size, nn_eval.get_nnx_len(), nn_eval.get_nny_len()));
    }
    NNServerBuf { input_buffers }
  }
}

//-------------------------------------------------------------------------------------

impl NNEvaluator {
  /// upstream: NNEvaluator::NNEvaluator (nneval.cpp)
  /// PORT NOTE: `Logger* logger` dropped (log crate). New trailing params (OWNERSHIP.tsv, D6):
  /// `backend` (the NeuralNet:: functions), `spawner` (spawnServerThreads), and `model_bytes`, the
  /// contents of `modelFileName`, which the backend loads instead of opening the file (no file IO
  /// in core). Throws -> Result.
  pub fn new(
    model_name: &str,
    model_file_name: &str,
    expected_sha256: &str,
    max_batch_size: i32,
    nn_x_len: i32,
    nn_y_len: i32,
    require_exact_nn_len: bool,
    inputs_use_nhwc: bool,
    nn_cache_size_power_of_two: i32,
    nn_mutex_pool_size_powerof_two: i32,
    debug_skip_neural_net: bool,
    open_cl_tuner_file: &str,
    home_data_dir_override: &str,
    open_cl_re_tune_per_board_size: bool,
    use_fp16_mode: enabled_t,
    use_nhwc_mode: enabled_t,
    num_threads: i32,
    gpu_idx_by_server_thread: &[i32],
    rand_seed: &str,
    do_randomize: bool,
    default_symmetry: i32,
    backend: Rc<dyn NeuralNet>,
    spawner: Rc<dyn Spawner>,
    model_bytes: Vec<u8>,
  ) -> Result<NNEvaluator, KataError> {
    let mut s = NNEvaluator {
      model_name: model_name.to_string(),
      model_file_name: model_file_name.to_string(),
      nn_x_len,
      nn_y_len,
      require_exact_nn_len,
      policy_size: NNPos::get_policy_size(nn_x_len, nn_y_len),
      inputs_use_nhwc,
      using_fp16_mode: use_fp16_mode,
      using_nhwc_mode: use_nhwc_mode,
      num_threads: Cell::new(num_threads),
      gpu_idx_by_server_thread: RefCell::new(gpu_idx_by_server_thread.to_vec()),
      rand_seed: rand_seed.to_string(),
      debug_skip_neural_net,
      compute_context: None,
      loaded_model: None,
      nn_cache_table: None,
      internal_model_name: String::new(),
      model_version: -1,
      inputs_version: -1,
      num_input_meta_channels: 0,
      post_process_params: ModelPostProcessParams::new(),
      num_server_threads_ever_spawned: Cell::new(0),
      server_threads: Cell::new(0),
      num_server_threads_alive: Cell::new(0),
      server_threads_exited: WakerList::new(),
      max_batch_size,
      m_num_rows_processed: Cell::new(0),
      m_num_batches_processed: Cell::new(0),
      is_killed: Cell::new(false),
      num_server_threads_starting_up: Cell::new(0),
      main_thread_waiting_for_spawn: WakerList::new(),
      server_threads_is_using_fp16: RefCell::new(Vec::new()),
      num_ongoing_evals: Cell::new(0),
      num_waiting_evals: Cell::new(0),
      num_evals_to_awaken: Cell::new(0),
      waiting_for_finish: WakerList::new(),
      current_do_randomize: Cell::new(do_randomize),
      current_default_symmetry: Cell::new(default_symmetry),
      current_batch_size: Cell::new(max_batch_size),
      query_queue: AsyncQueue::new(),
      backend,
      spawner,
    };
    if s.nn_x_len > NNPos::MAX_BOARD_LEN {
      return Err(KataError::StringError("Maximum supported nnEval board size is ".to_string() + &Global::int_to_string(NNPos::MAX_BOARD_LEN)));
    }
    if s.nn_y_len > NNPos::MAX_BOARD_LEN {
      return Err(KataError::StringError("Maximum supported nnEval board size is ".to_string() + &Global::int_to_string(NNPos::MAX_BOARD_LEN)));
    }
    if s.max_batch_size <= 0 {
      return Err(KataError::StringError("maxBatchSize is negative: ".to_string() + &Global::int_to_string(s.max_batch_size)));
    }
    // PORT NOTE: size_t != int converts numThreads to size_t (a negative count never matches).
    if s.gpu_idx_by_server_thread.borrow().len() as u64 != s.num_threads.get() as u64 {
      return Err(KataError::StringError("gpuIdxByServerThread.size() != numThreads".to_string()));
    }

    log::info!(
      "{}",
      "Initializing neural net buffer to be size ".to_string()
        + &Global::int_to_string(s.nn_x_len)
        + " * "
        + &Global::int_to_string(s.nn_y_len)
        + (if s.require_exact_nn_len { " exactly" } else { " allowing smaller boards" })
    );

    if nn_cache_size_power_of_two >= 0 {
      s.nn_cache_table = Some(Box::new(NNCacheTable::new(nn_cache_size_power_of_two, nn_mutex_pool_size_powerof_two)?));
    }

    if !s.debug_skip_neural_net {
      let mut gpu_idxs: Vec<i32> = s.gpu_idx_by_server_thread.borrow().clone();
      gpu_idxs.sort();
      gpu_idxs.dedup();
      let loaded_model: Box<dyn LoadedModel> = s.backend.load_model_file(&s.model_file_name, model_bytes, expected_sha256)?;
      {
        let desc = s.backend.get_model_desc(&*loaded_model);
        s.internal_model_name = desc.name.clone();
        s.model_version = desc.model_version;
        s.inputs_version = NNModelVersion::get_inputs_version(s.model_version)?;
        s.num_input_meta_channels = desc.num_input_meta_channels;
        s.post_process_params = desc.post_process_params;
      }
      s.compute_context = Some(s.backend.create_compute_context(
        &gpu_idxs,
        s.nn_x_len,
        s.nn_y_len,
        open_cl_tuner_file,
        home_data_dir_override,
        open_cl_re_tune_per_board_size,
        s.using_fp16_mode,
        s.using_nhwc_mode,
        &*loaded_model,
      )?);
      s.loaded_model = Some(loaded_model);
    } else {
      drop(model_bytes);
      s.internal_model_name = "random".to_string();
      s.model_version = NNModelVersion::defaultModelVersion;
      s.inputs_version = NNModelVersion::get_inputs_version(s.model_version)?;
    }

    //Reserve a decent amount above the batch size so that allocation is unlikely.
    s.query_queue.reserve((s.max_batch_size * 4) as usize * s.gpu_idx_by_server_thread.borrow().len());
    //Starts readonly. Becomes writable once we spawn server threads
    s.query_queue.set_read_only();
    Ok(s)
  }

  /// upstream: NNEvaluator::getModelName (nneval.cpp)
  pub fn get_model_name(&self) -> String {
    self.model_name.clone()
  }
  /// upstream: NNEvaluator::getModelFileName (nneval.cpp)
  pub fn get_model_file_name(&self) -> String {
    self.model_file_name.clone()
  }
  /// upstream: NNEvaluator::getInternalModelName (nneval.cpp)
  pub fn get_internal_model_name(&self) -> String {
    self.internal_model_name.clone()
  }

  /// upstream: NNEvaluator::getAbbrevInternalModelName (nneval.cpp)
  pub fn get_abbrev_internal_model_name(&self) -> String {
    let name: String = self.get_internal_model_name();
    let pieces: Vec<String> = Global::split_delim(&name, '-');
    let mut new_pieces: Vec<String> = Vec::new();
    for piece in &pieces {
      let mut buf = String::new();
      let pb = piece.as_bytes();
      if piece == "kata1" {
        // skip
      } else if pb.len() > 1 && pb[0] == b's' && try_abbreviate_step_string(piece, &mut buf) {
        new_pieces.push(buf);
      } else if pb.len() > 1 && pb[0] == b'd' && try_abbreviate_step_string(piece, &mut buf) {
        // skip
      } else {
        new_pieces.push(piece.clone());
      }
    }
    Global::concat(&new_pieces, "-")
  }

  /// upstream: NNEvaluator::isNeuralNetLess (nneval.cpp)
  pub fn is_neural_net_less(&self) -> bool {
    self.debug_skip_neural_net
  }
  /// upstream: NNEvaluator::getMaxBatchSize (nneval.cpp)
  pub fn get_max_batch_size(&self) -> i32 {
    self.max_batch_size
  }
  /// upstream: NNEvaluator::getCurrentBatchSize (nneval.cpp)
  pub fn get_current_batch_size(&self) -> i32 {
    self.current_batch_size.get() /*acquire*/
  }
  /// upstream: NNEvaluator::setCurrentBatchSize (nneval.cpp)
  pub fn set_current_batch_size(&self, batch_size: i32) -> Result<(), KataError> {
    if batch_size <= 0 || batch_size > self.max_batch_size {
      return Err(KataError::StringError("Invalid setting for batch size".to_string()));
    }
    self.current_batch_size.set(batch_size) /*release*/;
    Ok(())
  }
  /// upstream: NNEvaluator::requiresSGFMetadata (nneval.cpp)
  pub fn requires_sgf_metadata(&self) -> bool {
    self.num_input_meta_channels > 0
  }

  /// upstream: NNEvaluator::getNumGpus (nneval.cpp)
  /// PORT NOTE: `#ifdef USE_EIGEN_BACKEND` is not defined for our backend: the generic branch.
  pub fn get_num_gpus(&self) -> i32 {
    let mut gpu_idxs: BTreeSet<i32> = BTreeSet::new();
    let g = self.gpu_idx_by_server_thread.borrow();
    for i in 0..g.len() {
      gpu_idxs.insert(g[i]);
    }
    gpu_idxs.len() as i32
  }
  /// upstream: NNEvaluator::getNumServerThreads (nneval.cpp)
  pub fn get_num_server_threads(&self) -> i32 {
    self.gpu_idx_by_server_thread.borrow().len() as i32
  }
  /// upstream: NNEvaluator::getGpuIdxs (nneval.cpp)
  /// PORT NOTE: the non-Eigen branch, as getNumGpus.
  pub fn get_gpu_idxs(&self) -> BTreeSet<i32> {
    let mut gpu_idxs: BTreeSet<i32> = BTreeSet::new();
    let g = self.gpu_idx_by_server_thread.borrow();
    for i in 0..g.len() {
      gpu_idxs.insert(g[i]);
    }
    gpu_idxs
  }

  /// upstream: NNEvaluator::getNNXLen (nneval.cpp)
  pub fn get_nnx_len(&self) -> i32 {
    self.nn_x_len
  }
  /// upstream: NNEvaluator::getNNYLen (nneval.cpp)
  pub fn get_nny_len(&self) -> i32 {
    self.nn_y_len
  }
  /// upstream: NNEvaluator::getModelVersion (nneval.cpp)
  pub fn get_model_version(&self) -> i32 {
    self.model_version
  }
  // upstream: NNEvaluator::getTrunkSpatialConvDepth (nneval.cpp)
  // Not ported: needs ModelDesc's layer descs (desc.rs); no caller outside nneval.cpp.

  /// upstream: NNEvaluator::getUsingFP16Mode (nneval.cpp)
  pub fn get_using_fp16_mode(&self) -> enabled_t {
    self.using_fp16_mode
  }
  /// upstream: NNEvaluator::getUsingNHWCMode (nneval.cpp)
  pub fn get_using_nhwc_mode(&self) -> enabled_t {
    self.using_nhwc_mode
  }

  /// upstream: NNEvaluator::supportsShorttermError (nneval.cpp)
  //Check if the loaded neural net supports shorttermError fields
  pub fn supports_shortterm_error(&self) -> bool {
    self.model_version >= 9
  }

  /// upstream: NNEvaluator::getDoRandomize (nneval.cpp)
  pub fn get_do_randomize(&self) -> bool {
    self.current_do_randomize.get() /*acquire*/
  }
  /// upstream: NNEvaluator::getDefaultSymmetry (nneval.cpp)
  pub fn get_default_symmetry(&self) -> i32 {
    self.current_default_symmetry.get() /*acquire*/
  }
  /// upstream: NNEvaluator::setDoRandomize (nneval.cpp)
  pub fn set_do_randomize(&self, b: bool) {
    self.current_do_randomize.set(b) /*release*/;
  }
  /// upstream: NNEvaluator::setDefaultSymmetry (nneval.cpp)
  pub fn set_default_symmetry(&self, s: i32) {
    self.current_default_symmetry.set(s) /*release*/;
  }

  /// upstream: NNEvaluator::getSupportedRules (nneval.cpp)
  //Return the "nearest" supported ruleset to desiredRules by this model.
  //Fills supported with true if desiredRules itself was exactly supported, false if some modifications had to be made.
  pub fn get_supported_rules(&self, desired_rules: &Rules, supported: &mut bool) -> Rules {
    match &self.loaded_model {
      None => {
        *supported = true;
        *desired_rules
      }
      Some(loaded_model) => self.backend.get_model_desc(&**loaded_model).get_supported_rules(desired_rules, supported),
    }
  }

  /// upstream: NNEvaluator::numRowsProcessed (nneval.cpp)
  pub fn num_rows_processed(&self) -> u64 {
    self.m_num_rows_processed.get() /*relaxed*/
  }
  /// upstream: NNEvaluator::numBatchesProcessed (nneval.cpp)
  pub fn num_batches_processed(&self) -> u64 {
    self.m_num_batches_processed.get() /*relaxed*/
  }
  /// upstream: NNEvaluator::averageProcessedBatchSize (nneval.cpp)
  pub fn average_processed_batch_size(&self) -> f64 {
    self.num_rows_processed() as f64 / self.num_batches_processed() as f64
  }

  /// upstream: NNEvaluator::clearStats (nneval.cpp)
  pub fn clear_stats(&self) {
    self.m_num_rows_processed.set(0);
    self.m_num_batches_processed.set(0);
  }

  /// upstream: NNEvaluator::clearCache (nneval.cpp)
  //Clear all entires cached in the table
  pub fn clear_cache(&self) {
    if let Some(t) = &self.nn_cache_table {
      t.clear();
    }
  }

  /// upstream: NNEvaluator::isAnyThreadUsingFP16 (nneval.cpp)
  //After spawnServerThreads has returned, check if is was using FP16.
  pub fn is_any_thread_using_fp16(&self) -> bool {
    // PORT NOTE: lock removed (single-threaded)
    for is_using_fp16 in self.server_threads_is_using_fp16.borrow().iter() {
      if *is_using_fp16 != 0 {
        return true;
      }
    }
    false
  }

  /// upstream: NNEvaluator::setNumThreads (nneval.cpp)
  //Set the number of threads and what gpus they use. Only call this if threads are not spawned yet, or have been killed.
  pub fn set_num_threads(&self, gpu_idx_by_server_thr: &[i32]) -> Result<(), KataError> {
    if self.server_threads.get() != 0 {
      return Err(KataError::StringError("NNEvaluator::setNumThreads called when threads were already running!".to_string()));
    }
    self.num_threads.set(gpu_idx_by_server_thr.len() as i32);
    *self.gpu_idx_by_server_thread.borrow_mut() = gpu_idx_by_server_thr.to_vec();
    Ok(())
  }

  /// upstream: NNEvaluator::spawnServerThreads (nneval.cpp)
  /// PORT NOTE: async (D8): each server thread is a `serve` task spawned through the injected
  /// Spawner; the wait for their start-up (`mainThreadWaitingForSpawn`) is an await. Takes the
  /// evaluator's Rc because the tasks share it. It first completes the join that
  /// killServerThreads could not do (D8): it awaits until every serve task of a previous spawn has
  /// ended, then runs the rest of upstream's killServerThreads after join (`isKilled = false` and
  /// the post-join testAsserts), and only then reopens the queue. Without that a kill->spawn pair
  /// (testsearchv8) leaves the old task serving, since it never sees the queue read-only.
  //Actually spawn threads to handle evaluations.
  //If doRandomize, uses randSeed as a seed, further randomized per-thread
  //If not doRandomize, uses defaultSymmetry for all nn evaluations, unless a symmetry is requested in MiscNNInputParams.
  //This function itself is not threadsafe.
  pub async fn spawn_server_threads(self: &Rc<Self>) -> Result<(), KataError> {
    if self.server_threads.get() != 0 {
      return Err(KataError::StringError("NNEvaluator::spawnServerThreads called when threads were already running!".to_string()));
    }

    // PORT NOTE (D8): the deferred `serverThreads[i]->join()` of killServerThreads, then what
    // upstream's killServerThreads does after the join.
    self.server_threads_exited.wait_until(|| self.num_server_threads_alive.get() == 0).await;
    if self.is_killed.get() {
      //Can unset now that threads are dead
      self.is_killed.set(false);

      testAssert!(self.num_ongoing_evals.get() == 0);
      testAssert!(self.num_waiting_evals.get() == 0);
      testAssert!(self.num_evals_to_awaken.get() == 0);
    }

    {
      // PORT NOTE: lock removed (single-threaded)
      self.server_threads_is_using_fp16.borrow_mut().resize(self.num_threads.get() as usize, 0);
    }

    self.query_queue.unset_read_only();

    self.num_server_threads_starting_up.set(self.num_threads.get());
    for i in 0..self.num_threads.get() {
      let gpu_idx_for_this_thread: i32 = self.gpu_idx_by_server_thread.borrow()[i as usize];
      let rand_seed_this_thread: String =
        self.rand_seed.clone() + ":NNEvalServerThread:" + &Global::int_to_string(self.num_server_threads_ever_spawned.get());
      self.num_server_threads_ever_spawned.set(self.num_server_threads_ever_spawned.get() + 1);
      let nn_eval: Rc<NNEvaluator> = self.clone();
      self.num_server_threads_alive.set(self.num_server_threads_alive.get() + 1);
      self.spawner.spawn_local(Box::pin(serve_evals(rand_seed_this_thread, nn_eval, gpu_idx_for_this_thread, i)));
      self.server_threads.set(self.server_threads.get() + 1);
    }

    // PORT NOTE: lock removed; `while(numServerThreadsStartingUp > 0) mainThreadWaitingForSpawn.wait(lock)`
    self.main_thread_waiting_for_spawn.wait_until(|| self.num_server_threads_starting_up.get() <= 0).await;
    Ok(())
  }

  /// upstream: NNEvaluator::killServerThreads (nneval.cpp)
  /// PORT NOTE: no join here (PORTING.md §11, D8): it can't block. The serve tasks see the read-only
  /// queue on their next poll, finish whatever is still queued (as upstream's threads do before
  /// join returns) and end, dropping their Rc. The join and what follows it upstream (`isKilled =
  /// false`, the post-join testAsserts) happen right here only if no serve task is alive (never
  /// spawned, or Drop); otherwise in the next spawnServerThreads, which awaits the tasks' end. Until
  /// then `isKilled` stays true, as upstream's waiters see it during the join.
  /// An evaluator with live serve tasks is kept alive by them: hosts call this before dropping it.
  //Kill spawned server threads and join and free them. This function is not threadsafe, and along with spawnServerThreads
  //should have calls to it and spawnServerThreads singlethreaded.
  pub fn kill_server_threads(&self) {
    // PORT NOTE: lock removed (single-threaded)
    self.is_killed.set(true);
    self.query_queue.set_read_only();

    self.waiting_for_finish.notify_all();

    // PORT NOTE: `for(...) serverThreads[i]->join();` is deferred to spawnServerThreads (D8).
    self.server_threads.set(0);
    self.server_threads_is_using_fp16.borrow_mut().clear();

    if self.num_server_threads_alive.get() == 0 {
      //Can unset now that threads are dead
      self.is_killed.set(false);

      testAssert!(self.num_ongoing_evals.get() == 0);
      testAssert!(self.num_waiting_evals.get() == 0);
      testAssert!(self.num_evals_to_awaken.get() == 0);
    }
  }

  /// upstream: NNEvaluator::serve (nneval.cpp)
  /// PORT NOTE: async task body (PORTING.md §11); `waitPopUpToN` is awaited and returns whatever is
  /// queued when the task is polled. `NeuralNet::getOutput` is awaited.
  //Helper, for internal use only
  pub async fn serve(self: Rc<Self>, buf: &mut NNServerBuf, rand: &mut Rand, gpu_idx_for_this_thread: i32, server_thread_idx: i32) -> Result<(), KataError> {
    let mut num_batches_handled_this_thread: i64 = 0;
    let mut num_rows_handled_this_thread: i64 = 0;

    let mut gpu_handle: Option<Box<dyn ComputeHandle>> = None;
    if let Some(loaded_model) = &self.loaded_model {
      gpu_handle = Some(self.backend.create_compute_handle(
        &**self.compute_context.as_ref().expect("computeContext is created with loadedModel"),
        &**loaded_model,
        self.max_batch_size,
        self.require_exact_nn_len,
        self.inputs_use_nhwc,
        gpu_idx_for_this_thread,
        server_thread_idx,
      )?);
    }

    {
      // PORT NOTE: lock removed (single-threaded)
      testAssert!((server_thread_idx as usize) < self.server_threads_is_using_fp16.borrow().len());
      self.server_threads_is_using_fp16.borrow_mut()[server_thread_idx as usize] = match &gpu_handle {
        None => 0,
        Some(h) if self.backend.is_using_fp16(&**h) => 1,
        Some(_) => 0,
      };
      self.num_server_threads_starting_up.set(self.num_server_threads_starting_up.get() - 1);
      if self.num_server_threads_starting_up.get() <= 0 {
        self.main_thread_waiting_for_spawn.notify_all();
      }
    }

    let mut result_bufs: Vec<Rc<NNResultBuf>> = Vec::with_capacity(self.max_batch_size as usize);

    let mut output_buf: Vec<NNOutput> = Vec::new();

    loop {
      result_bufs.clear();
      let desired_batch_size: i32 = std::cmp::min(self.max_batch_size, self.current_batch_size.get() /*acquire*/);
      let got_anything: bool = self.query_queue.wait_pop_up_to_n(&mut result_bufs, desired_batch_size as usize).await;
      //Queue being closed is a signal that we're done.
      if !got_anything {
        break;
      }

      let num_rows: i32 = result_bufs.len() as i32;
      testAssert!(num_rows > 0);

      let do_randomize: bool = self.current_do_randomize.get() /*acquire*/;
      let default_symmetry: i32 = self.current_default_symmetry.get() /*acquire*/;

      if self.debug_skip_neural_net {
        for row in 0..num_rows {
          // PORT NOTE: `resultBufs[row] = NULL` is the clear() at the top of the loop.
          let result_buf: Rc<NNResultBuf> = result_bufs[row as usize].clone();

          let board_x_size: i32 = result_buf.board_x_size_for_server.get();
          let board_y_size: i32 = result_buf.board_y_size_for_server.get();

          testAssert!(result_buf.has_result.get() == false);
          let mut result: NNOutput = NNOutput::new();

          let policy_probs = &mut result.policy_probs;
          for i in 0..NNPos::MAX_NN_POLICY_SIZE {
            policy_probs[i as usize] = 0.0;
          }

          //At this point, these aren't probabilities, since this is before the postprocessing
          //that happens for each result. These just need to be unnormalized log probabilities.
          //Illegal move filtering happens later.
          for y in 0..board_y_size {
            for x in 0..board_x_size {
              let pos: i32 = NNPos::xy_to_pos(x, y, self.nn_x_len);
              policy_probs[pos as usize] = rand.next_gaussian() as f32;
            }
          }
          policy_probs[NNPos::loc_to_pos(Board::PASS_LOC, board_x_size, self.nn_x_len, self.nn_y_len) as usize] = rand.next_gaussian() as f32;

          result.nn_x_len = self.nn_x_len;
          result.nn_y_len = self.nn_y_len;
          if result_buf.include_owner_map.get() {
            let mut white_owner_map: Vec<f32> = vec![0.0f32; (self.nn_x_len * self.nn_y_len) as usize];
            for i in 0..self.nn_x_len * self.nn_y_len {
              white_owner_map[i as usize] = 0.0;
            }
            for y in 0..board_y_size {
              for x in 0..board_x_size {
                let pos: i32 = NNPos::xy_to_pos(x, y, self.nn_x_len);
                white_owner_map[pos as usize] = rand.next_gaussian() as f32 * 0.20f32;
              }
            }
            result.white_owner_map = Some(white_owner_map);
          } else {
            result.white_owner_map = None;
          }

          //These aren't really probabilities. Win/Loss/NoResult will get softmaxed later
          let white_win_prob: f64 = 0.0 + rand.next_gaussian() * 0.20;
          let white_loss_prob: f64 = 0.0 + rand.next_gaussian() * 0.20;
          let white_score_mean: f64 = 0.0 + rand.next_gaussian() * 0.20;
          let white_score_mean_sq: f64 = 0.0 + rand.next_gaussian() * 0.20;
          let white_no_result_prob: f64 = 0.0 + rand.next_gaussian() * 0.20;
          let var_time_left: f64 = 0.5 * board_x_size as f64 * board_y_size as f64;
          result.white_win_prob = white_win_prob as f32;
          result.white_loss_prob = white_loss_prob as f32;
          result.white_no_result_prob = white_no_result_prob as f32;
          result.white_score_mean = white_score_mean as f32;
          result.white_score_mean_sq = white_score_mean_sq as f32;
          result.white_lead = white_score_mean as f32;
          result.var_time_left = var_time_left as f32;
          result.shortterm_winloss_error = 0.0f32;
          result.shortterm_score_error = 0.0f32;
          result.policy_optimism_used = result_buf.policy_optimism.get() as f32;
          *result_buf.result.borrow_mut() = Some(Rc::new(result));
          result_buf.has_result.set(true);
          result_buf.client_waiting_for_result.notify_all();
        }
      } else {
        output_buf.clear();
        for row in 0..num_rows {
          let mut empty_output: NNOutput = NNOutput::new();
          empty_output.nn_x_len = self.nn_x_len;
          empty_output.nn_y_len = self.nn_y_len;
          if result_bufs[row as usize].include_owner_map.get() {
            empty_output.white_owner_map = Some(vec![0.0f32; (self.nn_x_len * self.nn_y_len) as usize]);
          } else {
            empty_output.white_owner_map = None;
          }
          output_buf.push(empty_output);
        }

        for row in 0..num_rows {
          if result_bufs[row as usize].symmetry.get() == NNInputs::SYMMETRY_NOTSPECIFIED {
            if do_randomize {
              result_bufs[row as usize].symmetry.set(rand.next_uint_n(SymmetryHelpers::NUM_SYMMETRIES as u32) as i32);
            } else {
              testAssert!(default_symmetry >= 0 && default_symmetry <= SymmetryHelpers::NUM_SYMMETRIES - 1);
              result_bufs[row as usize].symmetry.set(default_symmetry);
            }
          }
        }

        {
          let gpu_handle: &mut Box<dyn ComputeHandle> = gpu_handle.as_mut().expect("loadedModel is set when !debugSkipNeuralNet");
          let input_buffers: &mut Box<dyn InputBuffers> = buf.input_buffers.as_mut().expect("inputBuffers is created with loadedModel");
          self.backend.get_output(&mut **gpu_handle, &mut **input_buffers, num_rows, &result_bufs, &mut output_buf).await?;
        }
        testAssert!(output_buf.len() == num_rows as usize);

        cpp::cell_fetch_add(&self.m_num_rows_processed, num_rows as u64) /*relaxed*/;
        cpp::cell_fetch_add(&self.m_num_batches_processed, 1u64) /*relaxed*/;
        num_rows_handled_this_thread += num_rows as i64;
        num_batches_handled_this_thread += 1;

        for (row, output) in output_buf.drain(..).enumerate() {
          let result_buf: Rc<NNResultBuf> = result_bufs[row].clone();
          // PORT NOTE: `resultBufs[row] = NULL` is the clear() at the top of the loop.

          testAssert!(result_buf.has_result.get() == false);
          *result_buf.result.borrow_mut() = Some(Rc::new(output));
          result_buf.has_result.set(true);
          result_buf.client_waiting_for_result.notify_all();
        }
      }

      //Lock and update stats before looping again
      // PORT NOTE: lock removed (single-threaded)
      self.num_ongoing_evals.set(self.num_ongoing_evals.get() - num_rows);

      if self.num_waiting_evals.get() > 0 {
        self.num_evals_to_awaken.set(self.num_evals_to_awaken.get() + self.num_waiting_evals.get());
        self.num_waiting_evals.set(0);
        self.waiting_for_finish.notify_all();
      }
      continue;
    }

    // NeuralNet::freeComputeHandle(gpuHandle)
    drop(gpu_handle);
    log::info!(
      "{}",
      "GPU ".to_string()
        + &Global::int_to_string(gpu_idx_for_this_thread)
        + " finishing, processed "
        + &Global::int64_to_string(num_rows_handled_this_thread)
        + " rows "
        + &Global::int64_to_string(num_batches_handled_this_thread)
        + " batches"
    );
    Ok(())
  }

  /// upstream: NNEvaluator::waitForNextNNEvalIfAny (nneval.cpp)
  //If there is at least one evaluate ongoing, wait until at least one finishes.
  //Returns immediately if there isn't one ongoing right now.
  pub async fn wait_for_next_nn_eval_if_any(&self) {
    // PORT NOTE: lock removed (single-threaded)
    if self.num_ongoing_evals.get() <= 0 {
      return;
    }

    self.num_waiting_evals.set(self.num_waiting_evals.get() + 1);
    self.waiting_for_finish.wait_until(|| !(self.num_evals_to_awaken.get() <= 0 && !self.is_killed.get())).await;
    self.num_evals_to_awaken.set(self.num_evals_to_awaken.get() - 1);
  }

  /// upstream: NNEvaluator::averageMultipleSymmetries (nneval.cpp)
  /// PORT NOTE: returns the Rc instead of a heap-allocated `shared_ptr<NNOutput>*` (OWNERSHIP.tsv).
  pub async fn average_multiple_symmetries(
    &self,
    board: &Board,
    history: &BoardHistory,
    next_player: Player,
    sgf_meta: Option<&SGFMetadata>,
    base_nn_input_params: &MiscNNInputParams,
    buf: &Rc<NNResultBuf>,
    include_owner_map: bool,
    rand: &mut Rand,
    num_symmetries_to_sample: i32,
  ) -> Result<Rc<NNOutput>, KataError> {
    let mut nn_input_params: MiscNNInputParams = base_nn_input_params.clone();
    let mut ptrs: Vec<Rc<NNOutput>> = Vec::new();
    let mut symmetry_indexes: [i32; SymmetryHelpers::NUM_SYMMETRIES as usize] = [0; SymmetryHelpers::NUM_SYMMETRIES as usize];
    for (i, s) in symmetry_indexes.iter_mut().enumerate() {
      *s = i as i32;
    }
    for i in 0..num_symmetries_to_sample {
      let j: i32 = rand.next_int_a_b(i, SymmetryHelpers::NUM_SYMMETRIES - 1);
      symmetry_indexes.swap(i as usize, j as usize);
      nn_input_params.symmetry = symmetry_indexes[i as usize];
      let skip_cache_this_iteration: bool = true; //Skip cache since there's no guarantee which symmetry is in the cache
      self.evaluate_sgf_meta(board, history, next_player, sgf_meta, &nn_input_params, buf, skip_cache_this_iteration, include_owner_map).await?;
      ptrs.push(buf.result.borrow_mut().take().expect("evaluate always leaves a result in buf"));
    }
    Ok(Rc::new(NNOutput::new_others(&ptrs)))
  }

  /// upstream: NNEvaluator::evaluate (nneval.cpp)
  /// PORT NOTE: async (PORTING.md §11); the overload without `sgfMeta` keeps the base name.
  //Queue a position for the next neural net batch evaluation and wait for it. Upon evaluation, result
  //will be supplied in NNResultBuf& buf, the shared_ptr there can grabbed via std::move if desired.
  //logStream is for some error logging, can be NULL.
  //This function is threadsafe.
  pub async fn evaluate(
    &self,
    board: &Board,
    history: &BoardHistory,
    next_player: Player,
    nn_input_params: &MiscNNInputParams,
    buf: &Rc<NNResultBuf>,
    skip_cache: bool,
    include_owner_map: bool,
  ) -> Result<(), KataError> {
    self.evaluate_sgf_meta(board, history, next_player, None, nn_input_params, buf, skip_cache, include_owner_map).await
  }

  /// upstream: NNEvaluator::evaluate (nneval.cpp)
  /// PORT NOTE: async (PORTING.md §11); overload name per PORTING.md §1 rule 3.
  pub async fn evaluate_sgf_meta(
    &self,
    board: &Board,
    history: &BoardHistory,
    next_player: Player,
    sgf_meta: Option<&SGFMetadata>,
    nn_input_params_arg: &MiscNNInputParams,
    buf: &Rc<NNResultBuf>,
    skip_cache: bool,
    include_owner_map: bool,
  ) -> Result<(), KataError> {
    testAssert!(!self.is_killed.get());
    buf.has_result.set(false);

    if board.x_size > self.nn_x_len || board.y_size > self.nn_y_len {
      return Err(KataError::StringError(
        "NNEvaluator was configured with nnXLen = ".to_string()
          + &Global::int_to_string(self.nn_x_len)
          + " nnYLen = "
          + &Global::int_to_string(self.nn_y_len)
          + " but was asked to evaluate board with larger x or y size",
      ));
    }
    if self.require_exact_nn_len {
      if board.x_size != self.nn_x_len || board.y_size != self.nn_y_len {
        return Err(KataError::StringError(
          "NNEvaluator was configured with nnXLen = ".to_string()
            + &Global::int_to_string(self.nn_x_len)
            + " nnYLen = "
            + &Global::int_to_string(self.nn_y_len)
            + " and requireExactNNLen, but was asked to evaluate board with different x or y size",
        ));
      }
    }

    // Avoid using policy optimism for humanSL
    let mut nn_input_params: MiscNNInputParams = nn_input_params_arg.clone();
    if self.num_input_meta_channels > 0 {
      nn_input_params.policy_optimism = 0.0;
    }

    let nn_hash: Hash128 = NNInputs::get_hash(board, history, next_player, &nn_input_params);
    if self.num_input_meta_channels > 0 {
      match sgf_meta {
        None => Global::fatal_error(&("SGFMetadata is required for ".to_string() + &self.model_name + " but was not provided")),
        Some(sgf_meta) => {
          if !sgf_meta.initialized {
            Global::fatal_error(
              &("SGFMetadata is required for ".to_string()
                + &self.model_name
                + " but was not initialized. Did you specify humanSLProfile=... in katago's config or via overrides?"),
            );
          }
          // TODO(port): deferred (sgfmetadata.cpp, M8): `nnHash ^= sgfMeta->getHash(nextPlayer);`
          // SGFMetadata::getHash is not ported yet; only human-SL nets (numInputMetaChannels > 0)
          // get here, and tools/bin2onnx cannot convert them.
          Global::fatal_error("SGFMetadata::getHash is not ported yet (human-SL nets are not supported)");
        }
      }
    }

    let mut had_result_without_owner_map: bool = false;
    let mut result_without_owner_map: Option<Rc<NNOutput>> = None;
    if let Some(nn_cache_table) = &self.nn_cache_table {
      if !skip_cache && nn_cache_table.get(nn_hash, &mut buf.result.borrow_mut()) {
        let missing_owner_map: bool = buf.result.borrow().as_ref().expect("get() returned true").white_owner_map.is_none();
        if !(include_owner_map && missing_owner_map) {
          buf.has_result.set(true);
          return Ok(());
        } else {
          had_result_without_owner_map = true;
          result_without_owner_map = buf.result.borrow_mut().take();
          *buf.result.borrow_mut() = None;
        }
      }
    }
    buf.include_owner_map.set(include_owner_map);

    buf.board_x_size_for_server.set(board.x_size);
    buf.board_y_size_for_server.set(board.y_size);

    if !self.debug_skip_neural_net {
      let row_spatial_len: i32 = NNModelVersion::get_num_spatial_features(self.model_version)? * self.nn_x_len * self.nn_y_len;
      {
        let mut v = buf.row_spatial_buf.borrow_mut();
        if v.len() < row_spatial_len as usize {
          v.resize(row_spatial_len as usize, 0.0);
        }
      }
      let row_global_len: i32 = NNModelVersion::get_num_global_features(self.model_version)?;
      {
        let mut v = buf.row_global_buf.borrow_mut();
        if v.len() < row_global_len as usize {
          v.resize(row_global_len as usize, 0.0);
        }
      }
      let row_meta_len: i32 = self.num_input_meta_channels;
      {
        let mut v = buf.row_meta_buf.borrow_mut();
        if v.len() < row_meta_len as usize {
          v.resize(row_meta_len as usize, 0.0);
        }
      }

      const _: () = assert!(NNModelVersion::latestInputsVersionImplemented == 7);
      if self.inputs_version == 7 {
        NNInputs::fill_row_v7(
          board,
          history,
          next_player,
          &nn_input_params,
          self.nn_x_len,
          self.nn_y_len,
          self.inputs_use_nhwc,
          &mut buf.row_spatial_buf.borrow_mut(),
          &mut buf.row_global_buf.borrow_mut(),
        );
      } else {
        // TODO(port): fillRowV3..V6 are not ported (port-manifest.toml nninputs notes: input
        // versions of pre-v8 nets); upstream dispatches inputsVersion 3..6 to them here.
        return Err(KataError::StringError(
          "NNEvaluator: inputs version ".to_string() + &Global::int_to_string(self.inputs_version) + " (fillRowV3..V6) is not ported",
        ));
      }

      if row_meta_len > 0 {
        match sgf_meta {
          None => Global::fatal_error(&("SGFMetadata is required for ".to_string() + &self.model_name + " but was not provided")),
          Some(sgf_meta) => {
            if !sgf_meta.initialized {
              Global::fatal_error(
                &("SGFMetadata is required for ".to_string()
                  + &self.model_name
                  + " but was not initialized. Did you specify humanSLProfile=... in katago's config or via overrides?"),
              );
            }
            // TODO(port): deferred (sgfmetadata.cpp, M8): SGFMetadata::fillMetadataRow(sgfMeta,
            // buf.rowMetaBuf.data(), nextPlayer, board.x_size*board.y_size); buf.hasRowMeta = true;
            Global::fatal_error("SGFMetadata::fillMetadataRow is not ported yet (human-SL nets are not supported)");
          }
        }
      } else {
        buf.has_row_meta.set(false);
      }
    }

    buf.symmetry.set(nn_input_params.symmetry);
    buf.policy_optimism.set(nn_input_params.policy_optimism);

    // PORT NOTE: lock removed (single-threaded)
    self.num_ongoing_evals.set(self.num_ongoing_evals.get() + 1);

    // PORT NOTE: evaluate futures must not be cancelled (PORTING.md §11): a future dropped after
    // this push leaves `buf` queued, and the next evaluate with it would queue it twice.
    debug_assert!(!self.query_queue.any(|b| Rc::ptr_eq(b, buf)), "NNEvaluator::evaluate: buf is already queued (a cancelled evaluate future?)");
    let suc: bool = self.query_queue.force_push(buf.clone());
    testAssert!(suc);

    // PORT NOTE: `while(!buf.hasResult) buf.clientWaitingForResult.wait(resultLock);`
    buf.client_waiting_for_result.wait_until(|| buf.has_result.get()).await;

    // PORT NOTE: upstream post-processes `*buf.result` in place. The fresh result is unique (the
    // server dropped its reference), so it is mutated through Rc::get_mut while borrowed from the buf
    // (OWNERSHIP.tsv); nothing below awaits. On an error return the partially processed result stays
    // in buf.result, as upstream's does when it throws.
    let mut result_slot = buf.result.borrow_mut();
    let result: &mut NNOutput = Rc::get_mut(result_slot.as_mut().expect("the server sets buf.result before hasResult")).expect("unique: fresh server result");

    //Perform postprocessing on the result - turn the nn output into probabilities
    //As a hack though, if the only thing we were missing was the ownermap, just grab the old policy and values
    //and use those. This avoids recomputing in a randomly different orientation when we just need the ownermap
    //and causing policy weights to be different, which would reduce performance of successive searches in a game
    //by making the successive searches distribute their playouts less coherently and using the cache more poorly.
    if had_result_without_owner_map {
      let r_wo: &NNOutput = result_without_owner_map.as_ref().expect("set with hadResultWithoutOwnerMap");
      result.white_win_prob = r_wo.white_win_prob;
      result.white_loss_prob = r_wo.white_loss_prob;
      result.white_no_result_prob = r_wo.white_no_result_prob;
      result.white_score_mean = r_wo.white_score_mean;
      result.white_score_mean_sq = r_wo.white_score_mean_sq;
      result.white_lead = r_wo.white_lead;
      result.var_time_left = r_wo.var_time_left;
      result.shortterm_winloss_error = r_wo.shortterm_winloss_error;
      result.shortterm_score_error = r_wo.shortterm_score_error;
      result.policy_probs[..NNPos::MAX_NN_POLICY_SIZE as usize].copy_from_slice(&r_wo.policy_probs[..NNPos::MAX_NN_POLICY_SIZE as usize]);
      result.policy_optimism_used = r_wo.policy_optimism_used;
      result.nn_x_len = r_wo.nn_x_len;
      result.nn_y_len = r_wo.nn_y_len;
      testAssert!(result.white_owner_map.is_some());
    } else {
      let policy: &mut [f32] = &mut result.policy_probs[..];

      let policy_output_scaling: f32 = self.post_process_params.output_scale_multiplier / nn_input_params.nn_policy_temperature;

      let x_size: i32 = board.x_size;
      let y_size: i32 = board.y_size;

      let mut max_policy: f32 = -1e25f32;
      let mut is_legal: [bool; NNPos::MAX_NN_POLICY_SIZE as usize] = [false; NNPos::MAX_NN_POLICY_SIZE as usize];
      let mut legal_count: i32 = 0;
      testAssert!(next_player == history.presumed_next_move_pla);
      for i in 0..self.policy_size {
        let loc: Loc = NNPos::pos_to_loc(i, x_size, y_size, self.nn_x_len, self.nn_y_len);
        is_legal[i as usize] = history.is_legal(board, loc, next_player);
      }

      if nn_input_params.avoid_myt_dagger_hack && x_size >= 13 && y_size >= 13 {
        for symmetry in 0..8 {
          let mut banned: Loc = Board::NULL_LOC;
          if dagger_match(board, next_player, &mut banned, symmetry) {
            if banned != Board::NULL_LOC {
              is_legal[NNPos::loc_to_pos(banned, x_size, self.nn_x_len, self.nn_y_len) as usize] = false;
            }
          }
        }
      }

      for i in 0..self.policy_size {
        let policy_value: f32;
        if is_legal[i as usize] {
          legal_count += 1;
          policy_value = policy[i as usize] * policy_output_scaling;
        } else {
          policy_value = -1e30f32;
        }

        policy[i as usize] = policy_value;
        if policy_value > max_policy {
          max_policy = policy_value;
        }
      }

      testAssert!(legal_count > 0);

      let mut policy_sum: f32 = 0.0f32;

      if nn_input_params.enable_passing_hacks {
        //Cap passing prior policy at 95% (19x other moves)
        let max_pass_policy_sum_factor: f32 = 19.0f32;

        for i in 0..self.policy_size - 1 {
          policy[i as usize] = math::expf(policy[i as usize] - max_policy);
          policy_sum += policy[i as usize];
        }
        let pass_pos: i32 = NNPos::loc_to_pos(Board::PASS_LOC, x_size, self.nn_x_len, self.nn_y_len);
        testAssert!(pass_pos == self.policy_size - 1);
        let i: i32 = pass_pos;
        policy[i as usize] = cpp::cpp_max(1e-20f32, cpp::cpp_min(math::expf(policy[i as usize] - max_policy), policy_sum * max_pass_policy_sum_factor));
        policy_sum += policy[i as usize];
      } else {
        for i in 0..self.policy_size {
          policy[i as usize] = math::expf(policy[i as usize] - max_policy);
          policy_sum += policy[i as usize];
        }
      }

      if !policy_sum.is_finite() {
        // PORT NOTE: `cout << ...; history.printDebugInfo(cout,board);` -> log::error!
        let mut dbg = String::new();
        history.print_debug_info(&mut dbg, board);
        log::error!("Got nonfinite for policy sum\n{}", dbg);
        return Err(KataError::StringError("Got nonfinite for policy sum".to_string()));
      }

      //Somehow all legal moves rounded to 0 probability
      if (policy_sum as f64) <= 0.0 {
        if !buf.error_log_lockout.get() {
          buf.error_log_lockout.set(true);
          log::warn!("{}", "Warning: all legal moves rounded to 0 probability for ".to_string() + &self.model_file_name);
        }
        let uniform: f32 = 1.0f32 / legal_count as f32;
        for i in 0..self.policy_size {
          policy[i as usize] = if is_legal[i as usize] { uniform } else { -1.0f32 };
        }
      }
      //Normal case
      else {
        for i in 0..self.policy_size {
          policy[i as usize] = if is_legal[i as usize] { policy[i as usize] / policy_sum } else { -1.0f32 };
        }
      }

      //Fill everything out-of-bounds too, for robustness.
      for i in self.policy_size..NNPos::MAX_NN_POLICY_SIZE {
        policy[i as usize] = -1.0f32;
      }

      result.policy_optimism_used = nn_input_params.policy_optimism as f32;

      //Fix up the value as well. Note that the neural net gives us back the value from the perspective
      //of the player so we need to negate that to make it the white value.
      if self.model_version == 3 {
        const twoOverPi: f64 = 0.63661977236758134308;

        let mut win_prob: f64;
        let mut loss_prob: f64;
        let mut no_result_prob: f64;
        //Version 3 neural nets just pack the pre-arctanned scoreValue into the whiteScoreMean field
        // TODO(port): needs port::math::atanf: `atan(float)` is the float overload upstream;
        // approximated by the double atan of the widened float rounded back to float (equal
        // whenever glibc's correctly rounded atanf does not hit a double-rounding case).
        let score_value: f64 =
          (math::atan((result.white_score_mean * self.post_process_params.output_scale_multiplier) as f64) as f32) as f64 * twoOverPi;
        {
          let win_logits: f64 = (result.white_win_prob * self.post_process_params.output_scale_multiplier) as f64;
          let loss_logits: f64 = (result.white_loss_prob * self.post_process_params.output_scale_multiplier) as f64;
          let no_result_logits: f64 = (result.white_no_result_prob * self.post_process_params.output_scale_multiplier) as f64;

          //Softmax
          let max_logits: f64 = cpp::cpp_max(cpp::cpp_max(win_logits, loss_logits), no_result_logits);
          win_prob = math::exp(win_logits - max_logits);
          loss_prob = math::exp(loss_logits - max_logits);
          no_result_prob = math::exp(no_result_logits - max_logits);

          let prob_sum: f64 = win_prob + loss_prob + no_result_prob;
          win_prob /= prob_sum;
          loss_prob /= prob_sum;
          no_result_prob /= prob_sum;

          if !prob_sum.is_finite() || !score_value.is_finite() {
            log::error!(
              "Got nonfinite for nneval value\n{} {} {} {}",
              crate::port::fmt::cpp_ostream_f64(win_logits),
              crate::port::fmt::cpp_ostream_f64(loss_logits),
              crate::port::fmt::cpp_ostream_f64(no_result_logits),
              crate::port::fmt::cpp_ostream_f64(score_value)
            );
            return Err(KataError::StringError("Got nonfinite for nneval value".to_string()));
          }
        }

        if next_player == P_WHITE {
          result.white_win_prob = win_prob as f32;
          result.white_loss_prob = loss_prob as f32;
          result.white_no_result_prob = no_result_prob as f32;
          result.white_score_mean = ScoreValue::approx_white_score_of_score_value_smooth(score_value, 0.0, 2.0, board.sqrt_board_area()) as f32;
          result.white_score_mean_sq = result.white_score_mean * result.white_score_mean;
          result.white_lead = result.white_score_mean;
          result.var_time_left = -1.0;
          result.shortterm_winloss_error = -1.0;
          result.shortterm_score_error = -1.0;
        } else {
          result.white_win_prob = loss_prob as f32;
          result.white_loss_prob = win_prob as f32;
          result.white_no_result_prob = no_result_prob as f32;
          result.white_score_mean = -(ScoreValue::approx_white_score_of_score_value_smooth(score_value, 0.0, 2.0, board.sqrt_board_area()) as f32);
          result.white_score_mean_sq = result.white_score_mean * result.white_score_mean;
          result.white_lead = result.white_score_mean;
          result.var_time_left = -1.0;
          result.shortterm_winloss_error = -1.0;
          result.shortterm_score_error = -1.0;
        }
      } else if self.model_version >= 4 {
        let mut win_prob: f64;
        let mut loss_prob: f64;
        let mut no_result_prob: f64;
        let mut score_mean: f64;
        let mut score_mean_sq: f64;
        let mut lead: f64;
        let var_time_left: f64;
        let shortterm_winloss_error: f64;
        let shortterm_score_error: f64;
        {
          let pp = &self.post_process_params;
          let win_logits: f64 = (result.white_win_prob * pp.output_scale_multiplier) as f64;
          let loss_logits: f64 = (result.white_loss_prob * pp.output_scale_multiplier) as f64;
          let mut no_result_logits: f64 = (result.white_no_result_prob * pp.output_scale_multiplier) as f64;
          let score_mean_pre_scaled: f64 = (result.white_score_mean * pp.output_scale_multiplier) as f64;
          let score_stdev_pre_softplus: f64 = (result.white_score_mean_sq * pp.output_scale_multiplier) as f64;
          let lead_pre_scaled: f64 = (result.white_lead * pp.output_scale_multiplier) as f64;
          let var_time_left_pre_softplus: f64 = (result.var_time_left * pp.output_scale_multiplier) as f64;
          let shortterm_winloss_error_pre_softplus: f64 = (result.shortterm_winloss_error * pp.output_scale_multiplier) as f64;
          let shortterm_score_error_pre_softplus: f64 = (result.shortterm_score_error * pp.output_scale_multiplier) as f64;

          if history.rules.ko_rule != Rules::KO_SIMPLE && history.rules.scoring_rule != Rules::SCORING_TERRITORY {
            no_result_logits -= 100000.0;
          }

          //Softmax
          let max_logits: f64 = cpp::cpp_max(cpp::cpp_max(win_logits, loss_logits), no_result_logits);
          win_prob = math::exp(win_logits - max_logits);
          loss_prob = math::exp(loss_logits - max_logits);
          no_result_prob = math::exp(no_result_logits - max_logits);

          if history.rules.ko_rule != Rules::KO_SIMPLE && history.rules.scoring_rule != Rules::SCORING_TERRITORY {
            no_result_prob = 0.0;
          }

          let prob_sum: f64 = win_prob + loss_prob + no_result_prob;
          win_prob /= prob_sum;
          loss_prob /= prob_sum;
          no_result_prob /= prob_sum;

          score_mean = score_mean_pre_scaled * pp.score_mean_multiplier;
          let score_stdev: f64 = soft_plus(score_stdev_pre_softplus) * pp.score_stdev_multiplier;
          score_mean_sq = score_mean * score_mean + score_stdev * score_stdev;
          lead = lead_pre_scaled * pp.lead_multiplier;
          var_time_left = soft_plus(var_time_left_pre_softplus) * pp.variance_time_multiplier;

          //scoreMean and scoreMeanSq are still conditional on having a result, we need to make them unconditional now
          //noResult counts as 0 score for scorevalue purposes.
          score_mean = score_mean * (1.0 - no_result_prob);
          score_mean_sq = score_mean_sq * (1.0 - no_result_prob);
          lead = lead * (1.0 - no_result_prob);

          if self.model_version >= 14 {
            {
              let s: f64 = soft_plus(shortterm_winloss_error_pre_softplus * 0.5);
              shortterm_winloss_error = (s * s * pp.shortterm_value_error_multiplier).sqrt();
            }
            {
              let s: f64 = soft_plus(shortterm_score_error_pre_softplus * 0.5);
              shortterm_score_error = (s * s * pp.shortterm_score_error_multiplier).sqrt();
            }
          } else if self.model_version >= 10 {
            shortterm_winloss_error = (soft_plus(shortterm_winloss_error_pre_softplus) * pp.shortterm_value_error_multiplier).sqrt();
            shortterm_score_error = (soft_plus(shortterm_score_error_pre_softplus) * pp.shortterm_score_error_multiplier).sqrt();
          } else {
            shortterm_winloss_error = soft_plus(shortterm_winloss_error_pre_softplus);
            shortterm_score_error = soft_plus(shortterm_score_error_pre_softplus) * 10.0;
          }

          if !prob_sum.is_finite()
            || !score_mean.is_finite()
            || !score_mean_sq.is_finite()
            || !lead.is_finite()
            || !var_time_left.is_finite()
            || !shortterm_winloss_error.is_finite()
            || !shortterm_score_error.is_finite()
          {
            let f = crate::port::fmt::cpp_ostream_f64;
            log::error!(
              "Got nonfinite for nneval value\n{} {} {} {} {} {} {} {} {}",
              f(win_logits),
              f(loss_logits),
              f(no_result_logits),
              f(score_mean),
              f(score_mean_sq),
              f(lead),
              f(var_time_left),
              f(shortterm_winloss_error),
              f(shortterm_score_error)
            );
            return Err(KataError::StringError("Got nonfinite for nneval value".to_string()));
          }
        }

        if next_player == P_WHITE {
          result.white_win_prob = win_prob as f32;
          result.white_loss_prob = loss_prob as f32;
          result.white_no_result_prob = no_result_prob as f32;
          result.white_score_mean = score_mean as f32;
          result.white_score_mean_sq = score_mean_sq as f32;
          result.white_lead = lead as f32;
        } else {
          result.white_win_prob = loss_prob as f32;
          result.white_loss_prob = win_prob as f32;
          result.white_no_result_prob = no_result_prob as f32;
          result.white_score_mean = -(score_mean as f32);
          result.white_score_mean_sq = score_mean_sq as f32;
          result.white_lead = -(lead as f32);
        }

        if self.model_version >= 9 {
          result.var_time_left = var_time_left as f32;
          result.shortterm_winloss_error = shortterm_winloss_error as f32;
          result.shortterm_score_error = shortterm_score_error as f32;
        } else {
          result.var_time_left = -1.0;
          result.shortterm_winloss_error = -1.0;
          result.shortterm_score_error = -1.0;
        }
      } else {
        return Err(KataError::StringError("NNEval value postprocessing not implemented for model version".to_string()));
      }
    }

    //Postprocess ownermap
    if let Some(white_owner_map) = &mut result.white_owner_map {
      if self.model_version >= 3 {
        let output_scale_multiplier: f32 = self.post_process_params.output_scale_multiplier;
        for pos in 0..self.nn_x_len * self.nn_y_len {
          let y: i32 = pos / self.nn_x_len;
          let x: i32 = pos % self.nn_x_len;
          if y >= board.y_size || x >= board.x_size {
            white_owner_map[pos as usize] = 0.0f32;
          } else {
            //Similarly as mentioned above, the result we get back from the net is actually not from white's perspective,
            //but from the player to move, so we need to flip it to make it white at the same time as we tanh it.
            if next_player == P_WHITE {
              white_owner_map[pos as usize] = math::tanhf(white_owner_map[pos as usize] * output_scale_multiplier);
            } else {
              white_owner_map[pos as usize] = -math::tanhf(white_owner_map[pos as usize] * output_scale_multiplier);
            }
          }
        }
      } else {
        return Err(KataError::StringError("NNEval value postprocessing not implemented for model version".to_string()));
      }
    }

    //And record the nnHash in the result and put it into the table
    result.nn_hash = nn_hash;
    drop(result_slot);
    if let Some(nn_cache_table) = &self.nn_cache_table {
      nn_cache_table.set(buf.result.borrow().as_ref().expect("buf.result is set above"));
    }
    Ok(())
  }
}

impl Drop for NNEvaluator {
  /// upstream: NNEvaluator::~NNEvaluator (nneval.cpp)
  /// computeContext / loadedModel / nnCacheTable are freed by their Drops (in that field order).
  fn drop(&mut self) {
    self.kill_server_threads();
  }
}

/// upstream: tryAbbreviateStepString (nneval.cpp) static
fn try_abbreviate_step_string(input: &str, buf: &mut String) -> bool {
  let ib = input.as_bytes();
  let mut i: usize = 0;
  while i < ib.len() && !Global::is_digit(ib[i]) {
    i += 1;
  }
  if i > 1 {
    return false;
  }

  let prefix: String = input[..i].to_string();
  let mut number: i64 = 0;
  let suc: bool = Global::try_string_to_int64(&input[i..], &mut number);
  if !suc {
    return false;
  }

  if number >= 10000000000i64 {
    *buf = prefix.clone() + &(number / 1000000000i64).to_string() + "G";
  }
  if number >= 10000000 {
    *buf = prefix + &(number / 1000000).to_string() + "M";
  } else if number >= 10000 {
    *buf = prefix + &(number / 1000).to_string() + "K";
  } else {
    *buf = input.to_string();
  }
  true
}

/// upstream: serveEvals (nneval.cpp) static
/// PORT NOTE: the thread entry point becomes the spawned task. `loadedModel` comes from `nn_eval`.
/// Upstream lets an exception escape the thread (terminate); here it is a fatal error (panic).
/// The end of the task decrements `num_server_threads_alive` and notifies `server_threads_exited`
/// (stands in for join(), see killServerThreads / spawnServerThreads, D8).
async fn serve_evals(rand_seed_this_thread: String, nn_eval: Rc<NNEvaluator>, gpu_idx_for_this_thread: i32, server_thread_idx: i32) {
  let mut buf: NNServerBuf = NNServerBuf::new(&nn_eval, nn_eval.loaded_model.as_deref());
  let mut rand: Rand = Rand::new_seed(&rand_seed_this_thread);

  //Used to have a try catch around this but actually we're in big trouble if this raises an exception
  //and causes possibly the only nnEval thread to die, so actually go ahead and let the exception escape to
  //toplevel for easier debugging
  let alive_ref: Rc<NNEvaluator> = nn_eval.clone();
  let res = nn_eval.serve(&mut buf, &mut rand, gpu_idx_for_this_thread, server_thread_idx).await;
  drop(buf);
  alive_ref.num_server_threads_alive.set(alive_ref.num_server_threads_alive.get() - 1);
  alive_ref.server_threads_exited.notify_all();
  if let Err(e) = res {
    Global::fatal_error(&("NNEvaluator server thread: ".to_string() + e.what()));
  }
}

/// upstream: softPlus (nneval.cpp) static
fn soft_plus(x: f64) -> f64 {
  //Avoid blowup
  if x > 40.0 {
    x
  } else {
    math::ln(1.0 + math::exp(x))
  }
}

/// upstream: daggerPattern (nneval.cpp) static
const daggerPattern: [[i32; 8]; 9] = [
  [0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 2, 1, 0, 0, 0, 0],
  [0, 0, 2, 1, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0],
  [0, 2, 1, 0, 0, 0, 0, 0],
  [0, 3, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0],
];

/// upstream: daggerMatch (nneval.cpp) static
fn dagger_match(board: &Board, next_pla: Player, banned: &mut Loc, symmetry: i32) -> bool {
  for yi in 0..9 {
    for xi in 0..8 {
      let mut y: i32 = yi;
      let mut x: i32 = xi;
      if (symmetry & 0x1) != 0 {
        std::mem::swap(&mut x, &mut y);
      }
      if (symmetry & 0x2) != 0 {
        x = board.x_size - 1 - x;
      }
      if (symmetry & 0x4) != 0 {
        y = board.y_size - 1 - y;
      }
      let loc: Loc = Location::get_loc(x, y, board.x_size);
      let m: i32 = daggerPattern[yi as usize][xi as usize];
      if m == 0 && board.colors[loc as usize] != C_EMPTY {
        return false;
      }
      if m == 1 && board.colors[loc as usize] != next_pla {
        return false;
      }
      if m == 2 && board.colors[loc as usize] != get_opp(next_pla) {
        return false;
      }
      if m == 3 {
        *banned = loc;
      }
    }
  }
  true
}

//Uncomment this to lower the effective hash size down to one where we get true collisions
//#define SIMULATE_TRUE_HASH_COLLISIONS

impl NNCacheTable {
  /// upstream: NNCacheTable::NNCacheTable (nneval.cpp)
  /// PORT NOTE: the mutex pool is removed; its size argument is still validated and clamped as
  /// upstream (nothing derived from it remains).
  pub fn new(size_power_of_two: i32, mut mutex_pool_size_power_of_two: i32) -> Result<NNCacheTable, KataError> {
    if size_power_of_two < 0 || size_power_of_two > 63 {
      return Err(KataError::StringError("NNCacheTable: Invalid sizePowerOfTwo: ".to_string() + &Global::int_to_string(size_power_of_two)));
    }
    if mutex_pool_size_power_of_two < 0 || mutex_pool_size_power_of_two > 31 {
      return Err(KataError::StringError(
        "NNCacheTable: Invalid mutexPoolSizePowerOfTwo: ".to_string() + &Global::int_to_string(mutex_pool_size_power_of_two),
      ));
    }
    if mutex_pool_size_power_of_two > size_power_of_two {
      mutex_pool_size_power_of_two = size_power_of_two;
    }
    let _ = mutex_pool_size_power_of_two;

    let table_size: u64 = 1u64 << size_power_of_two;
    let table_mask: u64 = table_size - 1;
    let mut entries: Vec<Entry> = Vec::with_capacity(table_size as usize);
    for _ in 0..table_size {
      entries.push(Entry::new());
    }
    Ok(NNCacheTable { entries: RefCell::new(entries.into_boxed_slice()), table_size, table_mask })
  }

  /// upstream: NNCacheTable::get (nneval.cpp)
  //These are thread-safe. For get, ret will be set to nullptr upon a failure to find.
  pub fn get(&self, nn_hash: Hash128, ret: &mut Option<Rc<NNOutput>>) -> bool {
    //Free ret BEFORE locking, to avoid any expensive operations while locked.
    if ret.is_some() {
      *ret = None;
    }

    let idx: u64 = nn_hash.hash0 & self.table_mask;
    let entries = self.entries.borrow();
    let entry: &Entry = &entries[idx as usize];
    // PORT NOTE: lock removed (single-threaded)

    let mut found: bool = false;
    if let Some(p) = &entry.ptr {
      if p.nn_hash == nn_hash {
        *ret = Some(p.clone());
        found = true;
      }
    }
    found
  }

  /// upstream: NNCacheTable::set (nneval.cpp)
  pub fn set(&self, p: &Rc<NNOutput>) {
    //Immediately copy p right now, before locking, to avoid any expensive operations while locked.
    let mut buf: Option<Rc<NNOutput>> = Some(p.clone());

    let idx: u64 = p.nn_hash.hash0 & self.table_mask;
    {
      let mut entries = self.entries.borrow_mut();
      let entry: &mut Entry = &mut entries[idx as usize];
      // PORT NOTE: lock removed (single-threaded)
      //Perform a swap, to avoid any expensive free under the mutex.
      std::mem::swap(&mut entry.ptr, &mut buf);
    }

    //No longer locked, allow buf to fall out of scope now, will free whatever used to be present in the table.
    drop(buf);
  }

  /// upstream: NNCacheTable::clear (nneval.cpp)
  pub fn clear(&self) {
    let mut buf: Option<Rc<NNOutput>> = None;
    for idx in 0..self.table_size {
      {
        let mut entries = self.entries.borrow_mut();
        let entry: &mut Entry = &mut entries[idx as usize];
        std::mem::swap(&mut entry.ptr, &mut buf);
      }
      buf = None;
    }
    let _ = buf;
  }
}
