//! Shared helpers for the real-net search test transcriptions (testsearchv9.rs, testsearchv8.rs): test-local
//! copies of upstream cpp/tests/testsearchcommon.cpp (TestSearchCommon), a test-local CompactSgf subset, the
//! cout/Logger capture, the burn `startNNEval`, and the section comparator used against the Eigen reruns
//! (fixtures/upstream-v1.16.5/unit/testsearchv{8,9}/*.expected0.txt). Not a port of upstream code.
//!
//! Output capture. Upstream's Logger (logToStdout, logTime false) writes `": " + message + endl` to cout; the
//! expected texts contain those lines. The port logs through the `log` crate, so `CoutLogger` appends the same
//! text to the thread-local buffer the transcription's own `cout!` writes to. `delete nnEval` (which joins the
//! server thread upstream; it logs "GPU 0 finishing..." as it ends) is `kill_server_threads()` + running the pool
//! until stalled.
//!
//! Comparison (a real-net oracle, not byte-exact: burn's conv accumulation order differs from Eigen's, NN outputs
//! differ by ~1e-5, see search_smoke.rs). Both texts are split into sections (`split_sections`) and each section
//! is compared line by line (`compare_section`):
//!   - backend log lines are dropped on both sides first (`EXPECTED_BACKEND_LOG_PREFIXES`,
//!     `ACTUAL_BACKEND_LOG_PREFIXES`: Eigen and burn print different text when a compute handle is created);
//!   - every line is tokenized into text and numeric tokens; text tokens must be identical;
//!   - numeric tokens without '.'/exponent on both sides (visit counts N/PSV, root visits, NN rows/batches, move
//!     coordinates, hashes' digits...) must be identical;
//!   - decimal tokens may differ by at most max(`TOL_UNITS` units of the last printed digit of the coarser of the
//!     two, `TOL_REL` x max(1, |value|)), i.e. one unit in the last printed digit, or 1e-4 for the 6-significant-digit
//!     `cout << double` values.
//!
//! Each section is reported as `exact` (byte-identical after the log filter), `within tolerance` (max deviation in
//! last-digit units and absolute), or `DIVERGED` (first diverging line, both texts).

#![allow(dead_code, clippy::needless_late_init, clippy::default_constructed_unit_structs)]

use std::cell::Cell;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::future::Future;
use std::rc::Rc;
use std::sync::OnceLock;
use std::time::Duration;

// The net-agnostic helpers shared with nn_parity / search_smoke / search_tolerance (tests/common_nn/mod.rs).
#[path = "../common_nn/mod.rs"]
mod common_nn;
use common_nn::fake_clock;

use burn::prelude::*;
use futures::executor::LocalPool;

use katago_core::core::commontypes::enabled_t;
use katago_core::core::global::Global;
use katago_core::core::global::KataError;
use katago_core::core::timer::set_clock;
use katago_core::game::board::*;
use katago_core::game::boardhistory::BoardHistory;
use katago_core::game::rules::Rules;
use katago_core::neuralnet::nneval::NNEvaluator;
use katago_core::neuralnet::desc::ModelDesc;
use katago_core::neuralnet::nneval::NNResultBuf;
use katago_core::neuralnet::nninputs::NNOutput;
use katago_core::neuralnet::nninputs::NNPos;
use katago_core::neuralnet::nninputs::ScoreValue;
use katago_core::neuralnet::nninterface::ComputeContext;
use katago_core::neuralnet::nninterface::ComputeHandle;
use katago_core::neuralnet::nninterface::InputBuffers;
use katago_core::neuralnet::nninterface::LoadedModel;
use katago_core::neuralnet::nninterface::NeuralNet;
use katago_core::port::executor::LocalBoxFuture;
use katago_core::port::executor::Spawner;
use katago_core::port::fmt::cpp_ostream_f64;
use katago_core::port::node_arena::NodeIdx;
use katago_core::search::asyncbot::AsyncBot;
use katago_core::search::asyncbot::SearchBegunCallback;
use katago_core::search::reportedsearchvalues::ReportedSearchValues;
use katago_core::search::search::Search;
use katago_core::search::searchnode::SearchNode;
use katago_core::search::searchprint::PrintTreeOptions;
use katago_core::search::timecontrols::TimeControls;
use katago_core::strprintf;
use katago_core::testAssert;
use katago_nn_burn::Network;
use katago_nn_burn::backend::BurnNeuralNet;

pub type Net = katago_nn_burn::b6c96::Model;

//------------------------------------------------------------------------------------------------
// cout and the Logger

thread_local! {
  /// Everything upstream would have written to stdout: the test's own `cout` text and the Logger lines.
  pub static OUT: RefCell<String> = const { RefCell::new(String::new()) };
  /// `logger.setDisabled(true)` (runSearchTestsV8 disables it before its last `delete nnEval`).
  pub static LOGGER_DISABLED: Cell<bool> = const { Cell::new(false) };
  /// "message (at file:line)" of the last panic on this thread.
  pub static LAST_PANIC: RefCell<String> = const { RefCell::new(String::new()) };
  /// `static int seedCounter` of TestSearchCommon::getSearchRandSeed.
  static SEED_COUNTER: Cell<i32> = const { Cell::new(0) };
}

/// `cout`
pub struct Cout;

impl std::fmt::Write for Cout {
  fn write_str(&mut self, s: &str) -> std::fmt::Result {
    OUT.with(|o| o.borrow_mut().push_str(s));
    Ok(())
  }
}

macro_rules! cout {
  ($($t:tt)*) => {{
    use std::fmt::Write as _;
    write!($crate::common_search::Cout, $($t)*).unwrap();
  }};
}

/// `cout << ... << endl`
macro_rules! coutln {
  () => {{
    cout!("\n");
  }};
  ($($t:tt)*) => {{
    cout!($($t)*);
    cout!("\n");
  }};
}

/// `Logger logger(nullptr, logToStdout=true, logToStderr=false, logTime=false)`: `Logger::write(str)` =
/// `": " + str + endl` on cout.
struct CoutLogger;

impl log::Log for CoutLogger {
  fn enabled(&self, _metadata: &log::Metadata) -> bool {
    true
  }
  fn log(&self, record: &log::Record) {
    // Only the engine's own log lines are upstream Logger output (burn / cubecl may log too).
    let target = record.target();
    if !(target.starts_with("katago_core") || target.starts_with("katago_nn_burn")) {
      return;
    }
    if LOGGER_DISABLED.with(|d| d.get()) {
      return;
    }
    let msg = format!("{}", record.args());
    coutln!(": {}", msg);
  }
  fn flush(&self) {}
}

static LOGGER: CoutLogger = CoutLogger;

/// Program start (upstream main + runtests): Board::initHash, ScoreValue::initTables, the Logger and the clock.
pub fn init() {
  static ONCE: OnceLock<()> = OnceLock::new();
  ONCE.get_or_init(|| {
    log::set_logger(&LOGGER).expect("no other logger in this test binary");
    log::set_max_level(log::LevelFilter::Trace);
  });
  Board::init_hash();
  ScoreValue::init_tables();
  set_clock(fake_clock);
}

/// Takes everything written to `OUT` so far.
pub fn take_out() -> String {
  OUT.with(|o| std::mem::take(&mut *o.borrow_mut()))
}

//------------------------------------------------------------------------------------------------
// Executor

/// Drives `fut` on `pool`; panics when the pool stalls with `fut` pending and not woken (lost wakeup).
/// flex completes everything inside the pool, so no grace period (common_nn::run_watchdog with grace 0).
pub fn run_watchdog<F: Future>(pool: &mut LocalPool, label: &str, fut: F) -> F::Output {
  common_nn::run_watchdog(pool, label, Duration::ZERO, fut)
}

/// The executor the NN server tasks and the searches run on, and the burn backend.
pub struct Ctx {
  pub pool: LocalPool,
  pub spawner: Rc<dyn Spawner>,
  pub backend: Rc<dyn NeuralNet>,
  /// For evaluators smaller than the graph's fixed 19x19 (`PaddedNet`).
  pub padded_backend: Rc<dyn NeuralNet>,
}

impl Ctx {
  pub fn new() -> Ctx {
    let pool = LocalPool::new();
    let spawner: Rc<dyn Spawner> = Rc::new(pool.spawner());
    let device: Device = burn::backend::FlexDevice::default().into();
    let backend: Rc<dyn NeuralNet> = Rc::new(BurnNeuralNet::<Net>::new(device));
    let padded_backend: Rc<dyn NeuralNet> = Rc::new(PaddedNet { inner: backend.clone() });
    Ctx { pool, spawner, backend, padded_backend }
  }

  /// Runs `fut` to completion on this context's pool.
  pub fn run<F: Future>(&mut self, label: &str, fut: F) -> F::Output {
    run_watchdog(&mut self.pool, label, fut)
  }
}

/// The model file upstream's regen.sh passes (`tests/models/...bin.gz`); the burn backend gets the `.bpk`
/// weights of the same net (katago_nn_burn::b6c96::BPK_PATH) as `model_bytes`.
pub const MODEL_FILE: &str = "tests/models/g170-b6c96-s175395328-d26788732.bin.gz";

fn model_bytes() -> Vec<u8> {
  static BYTES: OnceLock<Vec<u8>> = OnceLock::new();
  BYTES.get_or_init(|| std::fs::read(katago_nn_burn::b6c96::BPK_PATH).expect("g170-b6c96 .bpk weights")).clone()
}

//------------------------------------------------------------------------------------------------
// Evaluators smaller than 19x19

/// Test-local backend adapter (no upstream counterpart): lets an NNEvaluator with nnXLen x nnYLen smaller than the
/// burn graph's fixed 19x19 (testsearchv8's 9x9 evaluators, `startNNEval(...,9,9,...)`) run, since
/// `BurnNeuralNet::create_compute_context` rejects nnXLen/nnYLen != posLen (backend.rs PORT NOTE). Each row is
/// padded into a 19x19 NCHW buffer (off-board 0, including the on-board mask channel 0: what NNEvaluator itself
/// writes for a smaller board in a 19x19 evaluator), evaluated by the wrapped burn backend, and the policy /
/// ownership outputs are cropped back to nnXLen x nnYLen. KataGo nets mask every spatial op with channel 0, so this
/// equals the exact-size evaluation Eigen does up to float accumulation order (the gtpForceMaxNNSize equivalence).
/// `requireExactNNLen` only lets a backend skip masking (Eigen ignores it), so the exact 9x9 evaluator is padded
/// the same way.
pub struct PaddedNet {
  inner: Rc<dyn NeuralNet>,
}

struct PaddedContext {
  inner: Box<dyn ComputeContext>,
  nn_x_len: i32,
  nn_y_len: i32,
}

impl ComputeContext for PaddedContext {
  fn as_any(&self) -> &dyn std::any::Any {
    self
  }
}

struct PaddedHandle {
  inner: Box<dyn ComputeHandle>,
  nn_x_len: i32,
  nn_y_len: i32,
}

impl ComputeHandle for PaddedHandle {
  fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
    self
  }
  fn as_any(&self) -> &dyn std::any::Any {
    self
  }
}

impl NeuralNet for PaddedNet {
  fn global_initialize(&self) {
    self.inner.global_initialize()
  }
  fn global_cleanup(&self) {
    self.inner.global_cleanup()
  }
  fn print_devices(&self) {
    self.inner.print_devices()
  }
  fn load_model_file(&self, file: &str, model_bytes: Vec<u8>, expected_sha256: &str) -> Result<Box<dyn LoadedModel>, KataError> {
    self.inner.load_model_file(file, model_bytes, expected_sha256)
  }
  fn get_model_desc<'a>(&self, loaded_model: &'a dyn LoadedModel) -> &'a ModelDesc {
    self.inner.get_model_desc(loaded_model)
  }
  fn create_compute_context(
    &self,
    gpu_idxs: &[i32],
    nn_x_len: i32,
    nn_y_len: i32,
    open_cl_tuner_file: &str,
    home_data_dir_override: &str,
    open_cl_re_tune_per_board_size: bool,
    use_fp16_mode: enabled_t,
    use_nhwc_mode: enabled_t,
    loaded_model: &dyn LoadedModel,
  ) -> Result<Box<dyn ComputeContext>, KataError> {
    let inner = self.inner.create_compute_context(
      gpu_idxs,
      Net::POS_LEN,
      Net::POS_LEN,
      open_cl_tuner_file,
      home_data_dir_override,
      open_cl_re_tune_per_board_size,
      use_fp16_mode,
      use_nhwc_mode,
      loaded_model,
    )?;
    Ok(Box::new(PaddedContext { inner, nn_x_len, nn_y_len }))
  }
  fn create_compute_handle(
    &self,
    context: &dyn ComputeContext,
    loaded_model: &dyn LoadedModel,
    max_batch_size: i32,
    require_exact_nn_len: bool,
    inputs_use_nhwc: bool,
    gpu_idx_for_this_thread: i32,
    server_thread_idx: i32,
  ) -> Result<Box<dyn ComputeHandle>, KataError> {
    let context = context.as_any().downcast_ref::<PaddedContext>().expect("PaddedContext");
    assert!(!inputs_use_nhwc, "PaddedNet pads NCHW rows");
    let inner =
      self.inner.create_compute_handle(&*context.inner, loaded_model, max_batch_size, require_exact_nn_len, inputs_use_nhwc, gpu_idx_for_this_thread, server_thread_idx)?;
    Ok(Box::new(PaddedHandle { inner, nn_x_len: context.nn_x_len, nn_y_len: context.nn_y_len }))
  }
  fn is_using_fp16(&self, compute_handle: &dyn ComputeHandle) -> bool {
    let h = compute_handle.as_any().downcast_ref::<PaddedHandle>().expect("PaddedHandle");
    self.inner.is_using_fp16(&*h.inner)
  }
  fn create_input_buffers(&self, loaded_model: &dyn LoadedModel, max_batch_size: i32, _nn_x_len: i32, _nn_y_len: i32) -> Box<dyn InputBuffers> {
    self.inner.create_input_buffers(loaded_model, max_batch_size, Net::POS_LEN, Net::POS_LEN)
  }
  fn get_output<'a>(
    &'a self,
    compute_handle: &'a mut dyn ComputeHandle,
    buffers: &'a mut dyn InputBuffers,
    num_batch_elts_filled: i32,
    input_bufs: &'a [Rc<NNResultBuf>],
    outputs: &'a mut [NNOutput],
  ) -> LocalBoxFuture<'a, Result<(), KataError>> {
    Box::pin(async move {
      let h = compute_handle.as_any_mut().downcast_mut::<PaddedHandle>().expect("PaddedHandle");
      let (xl, yl) = (h.nn_x_len as usize, h.nn_y_len as usize);
      let big = Net::POS_LEN as usize;
      let n = num_batch_elts_filled as usize;
      let mut padded: Vec<Rc<NNResultBuf>> = Vec::with_capacity(n);
      for i in 0..n {
        let src = &input_bufs[i];
        let spatial = src.row_spatial_buf.borrow();
        let num_channels = spatial.len() / (xl * yl);
        let mut row: Vec<f32> = vec![0.0; num_channels * big * big];
        for c in 0..num_channels {
          for y in 0..yl {
            for x in 0..xl {
              row[c * big * big + y * big + x] = spatial[c * xl * yl + y * xl + x];
            }
          }
        }
        let b = NNResultBuf::new();
        *b.row_spatial_buf.borrow_mut() = row;
        *b.row_global_buf.borrow_mut() = src.row_global_buf.borrow().clone();
        *b.row_meta_buf.borrow_mut() = src.row_meta_buf.borrow().clone();
        b.has_row_meta.set(src.has_row_meta.get());
        b.include_owner_map.set(src.include_owner_map.get());
        b.board_x_size_for_server.set(src.board_x_size_for_server.get());
        b.board_y_size_for_server.set(src.board_y_size_for_server.get());
        b.symmetry.set(src.symmetry.get());
        b.policy_optimism.set(src.policy_optimism.get());
        padded.push(Rc::new(b));
      }
      let mut big_outputs: Vec<NNOutput> = Vec::with_capacity(n);
      for i in 0..n {
        let mut o = NNOutput::new();
        o.nn_x_len = Net::POS_LEN;
        o.nn_y_len = Net::POS_LEN;
        if outputs[i].white_owner_map.is_some() {
          o.white_owner_map = Some(vec![0.0; big * big]);
        }
        big_outputs.push(o);
      }
      self.inner.get_output(&mut *h.inner, buffers, num_batch_elts_filled, &padded, &mut big_outputs).await?;
      for i in 0..n {
        let (o, bo) = (&mut outputs[i], &big_outputs[i]);
        o.white_win_prob = bo.white_win_prob;
        o.white_loss_prob = bo.white_loss_prob;
        o.white_no_result_prob = bo.white_no_result_prob;
        o.white_score_mean = bo.white_score_mean;
        o.white_score_mean_sq = bo.white_score_mean_sq;
        o.white_lead = bo.white_lead;
        o.var_time_left = bo.var_time_left;
        o.shortterm_winloss_error = bo.shortterm_winloss_error;
        o.shortterm_score_error = bo.shortterm_score_error;
        for y in 0..yl {
          for x in 0..xl {
            o.policy_probs[y * xl + x] = bo.policy_probs[y * big + x];
          }
        }
        o.policy_probs[xl * yl] = bo.policy_probs[big * big];
        if let Some(map) = o.white_owner_map.as_mut() {
          let bmap = bo.white_owner_map.as_ref().expect("ownership requested");
          for y in 0..yl {
            for x in 0..xl {
              map[y * xl + x] = bmap[y * big + x];
            }
          }
        }
      }
      Ok(())
    })
  }
}

//------------------------------------------------------------------------------------------------
// TestSearchCommon (testsearchcommon.cpp), test-local copies

/// upstream: TestSearchCommon::getSearchRandSeed (testsearchcommon.cpp) — test-local copy
pub fn get_search_rand_seed() -> String {
  let c = SEED_COUNTER.with(|s| {
    let c = s.get();
    s.set(c + 1);
    c
  });
  "testSearchSeed".to_string() + &Global::int_to_string(c)
}

/// Resets `getSearchRandSeed`'s static counter (a fresh process upstream).
pub fn reset_search_rand_seed() {
  SEED_COUNTER.with(|s| s.set(0));
}

/// upstream: TestSearchCommon::TestSearchOptions (testsearchcommon.h) — test-local copy
#[derive(Clone)]
pub struct TestSearchOptions {
  pub num_moves_in_a_row: i32,
  pub print_root_policy: bool,
  pub print_ownership: bool,
  pub print_ending_score_value_bonus: bool,
  pub print_play_selection_values: bool,
  pub print_root_values: bool,
  pub print_pruned_root_values: bool,
  pub print_sharp_score_and_error: bool,
  pub no_clear_bot: bool,
  pub no_clear_cache: bool,
  pub print_more: bool,
  pub print_more_more_more: bool,
  pub print_after_begun: bool,
  pub ignore_position: bool,
  pub print_post_order_node_count: bool,
  pub root_hint_loc: Loc,
}

impl TestSearchOptions {
  /// upstream: TestSearchCommon::TestSearchOptions::TestSearchOptions (testsearchcommon.cpp) — test-local copy
  pub fn new() -> TestSearchOptions {
    TestSearchOptions {
      num_moves_in_a_row: 1,
      print_root_policy: false,
      print_ownership: false,
      print_ending_score_value_bonus: false,
      print_play_selection_values: false,
      print_root_values: false,
      print_pruned_root_values: false,
      print_sharp_score_and_error: false,
      no_clear_bot: false,
      no_clear_cache: false,
      print_more: false,
      print_more_more_more: false,
      print_after_begun: false,
      ignore_position: false,
      print_post_order_node_count: false,
      root_hint_loc: Board::NULL_LOC,
    }
  }
}

/// `search->printTree(cout, search->rootNode, options, perspective)`
pub fn print_tree(search: &Search, options: &PrintTreeOptions, perspective: Player) {
  let root = search.get_root_node();
  search.print_tree(&mut Cout, root.as_deref(), options, perspective);
}

/// upstream: TestSearchCommon::printPolicyValueOwnership (testsearchcommon.cpp) — test-local copy
pub fn print_policy_value_ownership(board: &Board, buf: &NNResultBuf) {
  coutln!("{}", board);
  coutln!();
  buf.result.borrow().as_ref().expect("evaluated").debug_print(&mut Cout, board);
}

/// upstream: TestSearchCommon::printBasicStuffAfterSearch (testsearchcommon.cpp) — test-local copy
pub fn print_basic_stuff_after_search(board: &Board, hist: &BoardHistory, search: &Search, options: &PrintTreeOptions) {
  Board::print_board(&mut Cout, board, Board::NULL_LOC, Some(&hist.move_history));
  cout!("{} {}\n", hist.rules, hist.encore_phase);
  cout!("Root visits: {}\n", search.get_root_visits());
  cout!("New playouts: {}\n", search.last_search_num_playouts);
  coutln!("NN rows: {}", search.nn_evaluator.num_rows_processed());
  coutln!("NN batches: {}", search.nn_evaluator.num_batches_processed());
  coutln!("NN avg batch size: {}", cpp_ostream_f64(search.nn_evaluator.average_processed_batch_size()));
  cout!("PV: ");
  let root = search.get_root_node();
  search.print_pv(&mut Cout, root.as_deref(), 25);
  cout!("\n");
  cout!("Tree:\n");
  print_tree(search, options, P_WHITE);
}

/// upstream: TestSearchCommon::runBotOnPosition (testsearchcommon.cpp) — test-local copy
/// PORT NOTE: genMoveSynchronous is async, driven on `ctx`; onSearchBegun gets the Search (asyncbot.rs).
pub fn run_bot_on_position(ctx: &mut Ctx, bot: &mut AsyncBot, board: &Board, next_pla: Player, hist: &BoardHistory, opts: &TestSearchOptions) {
  let mut board: Board = board.clone();
  let mut next_pla: Player = next_pla;
  let mut hist: Box<BoardHistory> = Box::new(hist.clone());

  if !opts.ignore_position {
    bot.set_position(next_pla, &board, &hist).unwrap();
  }

  let mut options = PrintTreeOptions::new();
  options = options.max_depth(1);
  if opts.print_more_more_more {
    options = options.max_depth(20);
  } else if opts.print_more {
    options = options.min_visits_prop_to_expand(0.1).max_depth(2);
  }
  if opts.print_ownership {
    bot.set_always_include_owner_map(true);
  }

  if opts.root_hint_loc != Board::NULL_LOC {
    bot.set_root_hint_loc(opts.root_hint_loc);
  }

  for i in 0..opts.num_moves_in_a_row {
    let move_: Loc;
    if opts.print_after_begun {
      coutln!("Just after begun");
      let options2 = options.clone();
      let on_search_begun: SearchBegunCallback = Rc::new(move |search: &Search| {
        print_tree(search, &options2, P_WHITE);
      });
      move_ = ctx.run("genMoveSynchronous", bot.gen_move_synchronous_full(next_pla, &TimeControls::new(), 1.0, Some(on_search_begun))).unwrap();
    } else {
      move_ = ctx.run("genMoveSynchronous", bot.gen_move_synchronous(next_pla, &TimeControls::new())).unwrap();
    }
    let search: &Search = bot.get_search();

    print_basic_stuff_after_search(&board, &hist, search, &options);

    if opts.print_root_policy {
      search.print_root_policy_map(&mut Cout);
    }
    if opts.print_ownership {
      let ownership_and_stdev: (Vec<f64>, Vec<f64>) = search.get_average_and_standard_deviation_tree_ownership(None);
      let ownership: &Vec<f64> = &ownership_and_stdev.0;
      let ownership_stdev: &Vec<f64> = &ownership_and_stdev.1;
      for y in 0..board.y_size {
        for x in 0..board.x_size {
          let pos = NNPos::xy_to_pos(x, y, search.nn_x_len);
          cout!("{}", strprintf!("%6.1f ", ownership[pos as usize] * 100.0));
        }
        coutln!();
      }
      coutln!();
      for y in 0..board.y_size {
        for x in 0..board.x_size {
          let pos = NNPos::xy_to_pos(x, y, search.nn_x_len);
          cout!("{}", strprintf!("%6.1f ", ownership_stdev[pos as usize] * 100.0));
        }
        coutln!();
      }
      coutln!();
    }
    if opts.print_ending_score_value_bonus {
      search.print_root_ownership_map(&mut Cout, P_WHITE);
      search.print_root_ending_score_value_bonus(&mut Cout);
    }
    if opts.print_play_selection_values {
      coutln!("Play selection values");
      let scale_max_to_at_least = 10.0;
      let mut locs_buf: Vec<Loc> = Vec::new();
      let mut play_selection_values_buf: Vec<f64> = Vec::new();
      let success = search.get_play_selection_values(&mut locs_buf, &mut play_selection_values_buf, scale_max_to_at_least);
      testAssert!(success);
      for j in 0..locs_buf.len() {
        coutln!("{} {}", Location::to_string(locs_buf[j], &board), cpp_ostream_f64(play_selection_values_buf[j]));
      }
    }
    if opts.print_root_values {
      let mut values = ReportedSearchValues::new();
      let suc = search.get_root_values(&mut values);
      if !suc {
        coutln!("Unsuccessful getting root values");
      } else {
        coutln!("{}", values);
      }
    }
    if opts.print_pruned_root_values {
      let mut values = ReportedSearchValues::new();
      let suc = search.get_pruned_root_values(&mut values);
      if !suc {
        coutln!("Unsuccessful getting pruned root values");
      } else {
        coutln!("{}", values);
      }
    }
    if opts.print_sharp_score_and_error {
      let mut sharp_score = 0.0;
      let root = search.get_root_node();
      let suc = search.get_sharp_score(root.as_deref(), &mut sharp_score);
      testAssert!(suc);
      let errors: (f64, f64) = search.get_shallow_average_shortterm_wl_and_score_error(root.as_deref());
      coutln!("getSharpScore {}", cpp_ostream_f64(sharp_score));
      coutln!("avg WL error {}", cpp_ostream_f64(errors.0));
      coutln!("avg score error {}", cpp_ostream_f64(errors.1));
    }

    if opts.print_post_order_node_count {
      verify_tree_post_order(bot.get_search_stop_and_wait(), -1);
    }

    if i < opts.num_moves_in_a_row - 1 {
      bot.make_move(move_, next_pla).unwrap();
      hist.make_board_move_assume_legal(&mut board, move_, next_pla, None, false);
      coutln!("Just after move{}", Location::to_string(move_, &board));
      print_tree(bot.get_search(), &options, P_WHITE);
      next_pla = get_opp(next_pla);

      if opts.print_post_order_node_count {
        verify_tree_post_order(bot.get_search_stop_and_wait(), -1);
      }
    }
  }

  let search: &Search = bot.get_search();
  if !opts.no_clear_cache {
    search.nn_evaluator.clear_cache();
    search.nn_evaluator.clear_stats();
  }
  if !opts.no_clear_bot {
    bot.clear_search();
  }
}

/// upstream: TestSearchCommon::runBotOnSgf (testsearchcommon.cpp) — test-local copy
pub fn run_bot_on_sgf(ctx: &mut Ctx, bot: &mut AsyncBot, sgf_str: &str, default_rules: &Rules, turn_idx: i32, override_komi: f32, opts: &TestSearchOptions) {
  let sgf = CompactSgf::parse(sgf_str);

  let mut board = Board::new();
  let mut next_pla: Player = P_BLACK;
  let mut hist = Box::new(BoardHistory::new());
  let initial_rules = sgf.get_rules_or_fail_allow_unspecified(default_rules);
  sgf.setup_board_and_hist_assume_legal(&initial_rules, &mut board, &mut next_pla, &mut hist, turn_idx as i64);
  hist.set_komi(override_komi);
  run_bot_on_position(ctx, bot, &board, next_pla, &hist, opts);
}

/// upstream: TestSearchCommon::startNNEval (testsearchcommon.cpp) — test-local copy
/// PORT NOTE: `Logger&` dropped (log crate). The burn backend (NCHW, fp32) rejects inputsUseNHWC / useNHWC /
/// useFP16 = true, which upstream's regen.sh passes to the Eigen build (`true true false`, Eigen rejects NCHW);
/// the outputs do not depend on them, so callers pass `false` for all three. `spawnServerThreads()` is async and
/// driven to completion; the `sleep_for(0.1)` ordering hack is not needed (one OS thread, deterministic).
pub fn start_nn_eval(
  ctx: &mut Ctx,
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
  let open_cl_re_tune_per_board_size = false;
  let model_name = model_file;
  let open_cl_tuner_file = "";
  let home_data_dir_override = "";
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
    open_cl_tuner_file,
    home_data_dir_override,
    open_cl_re_tune_per_board_size,
    if use_fp16 { enabled_t::True } else { enabled_t::False },
    if use_nhwc { enabled_t::True } else { enabled_t::False },
    num_nn_server_threads_per_model,
    &gpu_idx_by_server_thread,
    &nn_rand_seed,
    nn_randomize,
    default_symmetry,
    if nn_x_len == Net::POS_LEN && nn_y_len == Net::POS_LEN { ctx.backend.clone() } else { ctx.padded_backend.clone() },
    ctx.spawner.clone(),
    model_bytes(),
  )
  .unwrap();
  let nn_eval = Rc::new(nn_eval);

  ctx.run("spawnServerThreads", nn_eval.spawn_server_threads()).unwrap();
  nn_eval
}

/// `nnEval->killServerThreads(); nnEval->spawnServerThreads();` (the "Reset random seeds for nnEval" idiom).
pub fn respawn_server_threads(ctx: &mut Ctx, nn_eval: &Rc<NNEvaluator>) {
  nn_eval.kill_server_threads();
  ctx.run("spawnServerThreads", nn_eval.spawn_server_threads()).unwrap();
}

/// `delete nnEval;` (~NNEvaluator -> killServerThreads, which joins the server thread: it logs
/// "GPU 0 finishing..." as it ends). Everything else holding the evaluator must be dropped first.
pub fn delete_nn_eval(ctx: &mut Ctx, nn_eval: Rc<NNEvaluator>) {
  nn_eval.kill_server_threads();
  ctx.pool.run_until_stalled();
  assert_eq!(Rc::strong_count(&nn_eval), 1, "the serve task ended and released the evaluator");
  drop(nn_eval);
}

/// upstream: TestSearchCommon::verifyTreePostOrder (testsearchcommon.cpp) — test-local copy
pub fn verify_tree_post_order(search: &Search, only_require_at_least: i32) {
  //Enumerate the tree and make sure every node is indeed hit exactly once in postorder.
  let nodes: Vec<NodeIdx> = search.enumerate_tree_post_order();
  let mut idx_of_node: std::collections::HashMap<NodeIdx, usize> = std::collections::HashMap::new();
  for i in 0..nodes.len() {
    idx_of_node.insert(nodes[i], i);
  }
  for i in 0..nodes.len() {
    let node: Rc<SearchNode> = search.node_arena.get(nodes[i]);
    let children = node.get_children();
    let children_capacity = children.get_capacity();
    for j in 0..children_capacity {
      let child: Option<NodeIdx> = children.index(j).get_if_allocated();
      let child = match child {
        None => break,
        Some(c) => c,
      };
      testAssert!(idx_of_node.contains_key(&child));
      testAssert!(idx_of_node[&child] < i);
    }
  }
  if only_require_at_least > 0 {
    if nodes.len() >= only_require_at_least as usize {
      coutln!("Post order okay: {}", "yes");
    } else {
      coutln!("Post order got too few nodes {} {}", nodes.len(), only_require_at_least);
      testAssert!(false);
    }
  } else {
    coutln!("Post order okay: {}", nodes.len());
  }
}

//------------------------------------------------------------------------------------------------
// Small helpers for the transcriptions

/// The k-th `R"%%(...)%%"` board diagram (`<slug>.input<k>.txt`) of a test file, parsed with `Board::parseBoard`.
pub fn parse_board_input(stem: &str, slug: &str, k: usize, x_size: i32, y_size: i32) -> Board {
  let s = fixture(stem, &format!("{}.input{}.txt", slug, k));
  Board::parse_board(x_size, y_size, &s, '\n').unwrap()
}

/// `fixtures/upstream-v1.16.5/unit/<stem>/<file>`
pub fn fixture(stem: &str, file: &str) -> String {
  let p = format!("{}/../../fixtures/upstream-v1.16.5/unit/{}/{}", env!("CARGO_MANIFEST_DIR"), stem, file);
  std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("missing fixture {}: {}", p, e))
}

/// `Location::ofString(s, board)`
pub fn loc_of(s: &str, board: &Board) -> Loc {
  Location::of_string(s, board).unwrap()
}

/// `hist.makeBoardMoveAssumeLegal(board, Location::ofString(s, board), pla, NULL)`
pub fn play(hist: &mut BoardHistory, board: &mut Board, s: &str, pla: Player) {
  let loc = loc_of(s, board);
  hist.make_board_move_assume_legal(board, loc, pla, None, false);
}

//------------------------------------------------------------------------------------------------
// Test-local CompactSgf subset

/// upstream: CompactSgf (sgf.cpp) — test-local subset (extends the one in katago-core/tests/testnninputs.rs).
/// Only what the SGF strings of the search tests use: one variation, unescaped values, root properties SZ
/// (square or "x:y"), KM, RU, PL, AB/AW/AE (points or rectangles), B/W moves with pass as `[]` or `[tt]`.
/// cpp/dataio/sgf is an [[ignored]] entry in port-manifest.toml (PORTING.md §13 test-local copies).
pub struct CompactSgf {
  root: BTreeMap<String, Vec<String>>,
  pub x_size: i32,
  pub y_size: i32,
  pub placements: Vec<Move>,
  pub moves: Vec<Move>,
}

/// upstream: parseSgfCoord (sgf.cpp) static — test-local copy
fn parse_sgf_coord(c: u8) -> i32 {
  if c.is_ascii_lowercase() {
    return c as i32 - 'a' as i32;
  }
  if c.is_ascii_uppercase() {
    return c as i32 - 'A' as i32 + 26;
  }
  -1
}

impl CompactSgf {
  /// upstream: CompactSgf::parse (sgf.cpp) — test-local subset
  pub fn parse(s: &str) -> CompactSgf {
    let b = s.as_bytes();
    let mut nodes: Vec<Vec<(String, Vec<String>)>> = Vec::new();
    let mut i = 0;
    assert_eq!(b[0], b'(');
    i += 1;
    while i < b.len() && b[i] != b')' {
      if b[i] == b';' {
        nodes.push(Vec::new());
        i += 1;
        continue;
      }
      if b[i].is_ascii_whitespace() {
        i += 1;
        continue;
      }
      assert!(b[i].is_ascii_uppercase(), "unexpected sgf byte at {}", i);
      let k0 = i;
      while b[i].is_ascii_uppercase() {
        i += 1;
      }
      let key = s[k0..i].to_string();
      let mut vals = Vec::new();
      while i < b.len() && b[i] == b'[' {
        let v0 = i + 1;
        let v1 = v0 + s[v0..].find(']').expect("unterminated value");
        vals.push(s[v0..v1].to_string());
        i = v1 + 1;
      }
      nodes.last_mut().expect("property before node").push((key, vals));
    }
    let root: BTreeMap<String, Vec<String>> = nodes[0].iter().cloned().collect();

    // upstream: Sgf::getXYSize
    let (x_size, y_size) = match root.get("SZ") {
      None => (19, 19),
      Some(v) => {
        let sz = &v[0];
        if let Some((a, c)) = sz.split_once(':') {
          (a.parse::<i32>().unwrap(), c.parse::<i32>().unwrap())
        } else {
          let n = sz.parse::<i32>().unwrap();
          (n, n)
        }
      }
    };

    // upstream: SgfNode::accumPlacements (AB, then AW, then AE)
    let mut placements = Vec::new();
    for (key, color) in [("AB", P_BLACK), ("AW", P_WHITE), ("AE", C_EMPTY)] {
      if let Some(elts) = root.get(key) {
        for e in elts {
          let e = e.as_bytes();
          let (x1, y1, x2, y2) = if e.len() == 5 && e[2] == b':' {
            (parse_sgf_coord(e[0]), parse_sgf_coord(e[1]), parse_sgf_coord(e[3]), parse_sgf_coord(e[4]))
          } else {
            assert_eq!(e.len(), 2);
            (parse_sgf_coord(e[0]), parse_sgf_coord(e[1]), parse_sgf_coord(e[0]), parse_sgf_coord(e[1]))
          };
          for x in x1..=x2 {
            for y in y1..=y2 {
              placements.push(Move::new(Location::get_loc(x, y, x_size), color));
            }
          }
        }
      }
    }

    // upstream: Sgf::getMoves / SgfNode::accumMoves (parseSgfLocOrPass)
    let mut moves = Vec::new();
    for node in &nodes {
      for (key, vals) in node {
        let pla = match key.as_str() {
          "B" => P_BLACK,
          "W" => P_WHITE,
          _ => continue,
        };
        for v in vals {
          let loc = if v.is_empty() || (v == "tt" && (x_size <= 19 || y_size <= 19)) {
            Board::PASS_LOC
          } else {
            let vb = v.as_bytes();
            assert_eq!(vb.len(), 2);
            let (x, y) = (parse_sgf_coord(vb[0]), parse_sgf_coord(vb[1]));
            assert!(x >= 0 && x < x_size && y >= 0 && y < y_size);
            Location::get_loc(x, y, x_size)
          };
          moves.push(Move::new(loc, pla));
        }
      }
    }
    CompactSgf { root, x_size, y_size, placements, moves }
  }

  /// upstream: SgfNode::getRulesFromRUTagOrFail (sgf.cpp) — test-local copy
  fn get_rules_from_ru_tag_or_fail(&self) -> Rules {
    let s = &self.root.get("RU").expect("SGF file does not specify rules")[0];
    let mut parsed = Rules::new();
    let suc = Rules::try_parse_rules(s, &mut parsed).unwrap();
    assert!(suc, "Could not parse rules in sgf: {}", s);
    parsed
  }

  /// upstream: SgfNode::getKomiOrFail (sgf.cpp) — test-local subset (no GoGoD / foxwq komi hacks: the test
  /// SGFs have plain int / half-int komis, asserted).
  fn get_komi_or_fail(&self) -> f32 {
    let v = &self.root.get("KM").expect("Sgf does not specify komi")[0];
    let mut komi: f32 = 0.0;
    assert!(Global::try_string_to_float(v, &mut komi), "Could not parse komi in sgf");
    assert!(Rules::komi_is_int_or_half_int(komi), "Komi in sgf is not integer or half-integer");
    komi
  }

  /// upstream: CompactSgf::getRulesOrFail (sgf.cpp) — test-local copy
  pub fn get_rules_or_fail(&self) -> Rules {
    let mut rules = self.get_rules_from_ru_tag_or_fail();
    rules.komi = self.get_komi_or_fail();
    rules
  }

  /// upstream: CompactSgf::getRulesOrFailAllowUnspecified (sgf.cpp) — test-local copy
  pub fn get_rules_or_fail_allow_unspecified(&self, default_rules: &Rules) -> Rules {
    let mut rules: Rules;
    if !self.root.contains_key("RU") {
      rules = *default_rules;
    } else {
      rules = self.get_rules_from_ru_tag_or_fail();
    }

    if self.root.contains_key("KM") {
      rules.komi = self.get_komi_or_fail();
    }
    rules
  }

  /// upstream: CompactSgf::setupInitialBoardAndHist (sgf.cpp) — test-local copy
  pub fn setup_initial_board_and_hist(&self, initial_rules: &Rules, board: &mut Board, next_pla: &mut Player, hist: &mut BoardHistory) {
    // getPLSpecifiedColor
    let pl_player: Color = match self.root.get("PL").map(|v| v[0].to_lowercase()) {
      Some(s) if s == "b" || s == "black" => C_BLACK,
      Some(s) if s == "w" || s == "white" => C_WHITE,
      _ => C_EMPTY,
    };
    if pl_player == P_BLACK || pl_player == P_WHITE {
      *next_pla = pl_player;
    } else {
      let mut has_black = false;
      let mut all_black = true;
      for i in 0..self.placements.len() {
        if self.placements[i].pla == P_BLACK {
          has_black = true;
        } else {
          all_black = false;
        }
      }
      if has_black && all_black {
        *next_pla = P_WHITE;
      } else {
        *next_pla = P_BLACK;
      }
    }

    // Override with the actual color of the move, if it exists
    if !self.moves.is_empty() {
      *next_pla = self.moves[0].pla;
    }

    *board = Board::new_x_y(self.x_size, self.y_size).unwrap();
    let suc = board.set_stones_fail_if_no_libs(&self.placements);
    assert!(suc, "setupInitialBoardAndHist: initial board position contains invalid stones or zero-liberty stones");
    *hist = BoardHistory::new_full(board, *next_pla, initial_rules, 0);
    if hist.initial_turn_number < board.num_stones_on_board() as i64 {
      hist.initial_turn_number = board.num_stones_on_board() as i64;
    }
  }

  /// upstream: CompactSgf::playMovesAssumeLegal (sgf.cpp) — test-local copy
  pub fn play_moves_assume_legal(&self, board: &mut Board, next_pla: &mut Player, hist: &mut BoardHistory, turn_idx: i64) {
    assert!(
      turn_idx >= 0 && turn_idx <= self.moves.len() as i64,
      "Attempting to set up position from SGF for invalid turn idx {}, valid values are 0 to {}",
      turn_idx,
      self.moves.len()
    );

    for i in 0..turn_idx as usize {
      hist.make_board_move_assume_legal(board, self.moves[i].loc, self.moves[i].pla, None, false);
      *next_pla = get_opp(self.moves[i].pla);
    }
  }

  /// upstream: CompactSgf::setupBoardAndHistAssumeLegal (sgf.cpp) — test-local copy
  pub fn setup_board_and_hist_assume_legal(&self, initial_rules: &Rules, board: &mut Board, next_pla: &mut Player, hist: &mut BoardHistory, turn_idx: i64) {
    self.setup_initial_board_and_hist(initial_rules, board, next_pla, hist);
    self.play_moves_assume_legal(board, next_pla, hist, turn_idx);
  }
}

//------------------------------------------------------------------------------------------------
// Sections

/// Backend log lines (upstream Logger output from inside NeuralNet::createComputeHandle) dropped before comparing:
/// Eigen prints ": Eigen (CPU) backend thread 0: Model version 8" / "...: Model name: g170-b6c96-...", burn prints
/// ": Burn backend thread 0: Model version 8" / "...: Model name: ...". Known backend differences, not test output.
pub const EXPECTED_BACKEND_LOG_PREFIXES: [&str; 1] = [": Eigen (CPU) backend thread "];
pub const ACTUAL_BACKEND_LOG_PREFIXES: [&str; 1] = [": Burn backend thread "];

/// One section of a test's stdout.
#[derive(Clone, Debug)]
pub struct Section {
  /// "#<index> <title>", unique.
  pub label: String,
  pub title: String,
  pub lines: Vec<String>,
}

fn is_header_with_title(line: &str) -> Option<String> {
  // "Title ======" (at least 5 trailing '='), with a non-empty title.
  if !line.ends_with("=====") {
    return None;
  }
  let t = line.trim_end_matches('=').trim_end();
  if t.is_empty() {
    return None;
  }
  Some(t.to_string())
}

/// Splits a test's stdout into sections. A new section starts at:
///   - a header trio: a line of exactly 67 '=', a title line, a line of 67 '=' (title = the middle line);
///   - a line "<title> =====..." (ends with >= 5 '=' after a non-empty title);
///   - a line equal to one of `extra_headers` (titled sub-blocks printed without '=' by the test code);
///   - the first of a run of NNEvaluator construction lines (": Initializing neural net buffer..."), titled
///     "<startNNEval>": the evaluator set-up between two test functions;
///
/// The final "Done" line plus the ": GPU ... finishing" lines just before it form "<teardown>"; lines before the
/// first header form "<preamble>".
pub fn split_sections(text: &str, extra_headers: &[&str]) -> Vec<Section> {
  let eq = "=".repeat(67);
  let lines: Vec<&str> = text.lines().collect();
  let mut teardown_start = lines.len();
  if lines.last() == Some(&"Done") {
    teardown_start = lines.len() - 1;
    while teardown_start > 0 && lines[teardown_start - 1].starts_with(": GPU ") {
      teardown_start -= 1;
    }
  }
  let mut out: Vec<Section> = vec![Section { label: String::new(), title: "<preamble>".to_string(), lines: Vec::new() }];
  let mut i = 0;
  while i < teardown_start {
    let l = lines[i];
    let new_title: Option<(String, usize)> = if i + 2 < teardown_start && l == eq && lines[i + 2] == eq {
      Some((lines[i + 1].to_string(), 3))
    } else if let Some(t) = is_header_with_title(l) {
      Some((t, 1))
    } else if extra_headers.contains(&l) {
      Some((l.to_string(), 1))
    } else if l.starts_with(": Initializing neural net buffer") && out.last().unwrap().title != "<startNNEval>" {
      Some(("<startNNEval>".to_string(), 1))
    } else {
      None
    };
    match new_title {
      Some((title, n)) => {
        out.push(Section { label: String::new(), title, lines: lines[i..i + n].iter().map(|s| s.to_string()).collect() });
        i += n;
      }
      None => {
        out.last_mut().unwrap().lines.push(l.to_string());
        i += 1;
      }
    }
  }
  if teardown_start < lines.len() {
    out.push(Section { label: String::new(), title: "<teardown>".to_string(), lines: lines[teardown_start..].iter().map(|s| s.to_string()).collect() });
  }
  if out[0].lines.is_empty() {
    out.remove(0);
  }
  for (k, s) in out.iter_mut().enumerate() {
    s.label = format!("#{} {}", k, s.title);
  }
  out
}

//------------------------------------------------------------------------------------------------
// Line comparison

/// Max deviation allowed for decimal tokens, in units of the last printed digit (of the coarser token): covers the
/// values printed with few decimals (`%.2f` utilities, `%6.1f` ownership, ...).
pub const TOL_UNITS: f64 = 1.0;
/// ... or this much relative to max(1, |value|): covers values printed by `cout << double` with 6 significant digits
/// (getSharpScore, play selection values, ReportedSearchValues), where the ~1e-5 NN difference between burn and Eigen
/// shows in the last digits. 1e-4 is the "values +-1e-4" search tier of CLAUDE.md / PORTING.md §13.
pub const TOL_REL: f64 = 1e-4;

#[derive(Debug, Clone, PartialEq)]
enum Tok {
  Text(String),
  Num(String),
}

fn tokenize(line: &str) -> Vec<Tok> {
  let b = line.as_bytes();
  let mut out: Vec<Tok> = Vec::new();
  let mut text = String::new();
  let mut i = 0;
  while i < b.len() {
    let c = b[i];
    let prev_alnum = i > 0 && (b[i - 1].is_ascii_alphanumeric() || b[i - 1] == b'.');
    let starts_num = c.is_ascii_digit() || ((c == b'-' || c == b'+') && i + 1 < b.len() && b[i + 1].is_ascii_digit() && !prev_alnum);
    if starts_num {
      let s = i;
      if c == b'-' || c == b'+' {
        i += 1;
      }
      while i < b.len() && b[i].is_ascii_digit() {
        i += 1;
      }
      if i + 1 < b.len() && b[i] == b'.' && b[i + 1].is_ascii_digit() {
        i += 1;
        while i < b.len() && b[i].is_ascii_digit() {
          i += 1;
        }
      }
      // exponent (cout's %g form: 1e-05, 2.5e+06)
      if i + 1 < b.len() && b[i] == b'e' {
        let mut j = i + 1;
        if j < b.len() && (b[j] == b'-' || b[j] == b'+') {
          j += 1;
        }
        if j < b.len() && b[j].is_ascii_digit() {
          while j < b.len() && b[j].is_ascii_digit() {
            j += 1;
          }
          i = j;
        }
      }
      if !text.is_empty() {
        out.push(Tok::Text(std::mem::take(&mut text)));
      }
      out.push(Tok::Num(line[s..i].to_string()));
    } else {
      text.push(c as char);
      i += 1;
    }
  }
  if !text.is_empty() {
    out.push(Tok::Text(text));
  }
  out
}

fn is_decimal(s: &str) -> bool {
  s.contains('.') || s.contains('e')
}

/// One unit of the last printed digit of a decimal token ("1.25" -> 0.01, "3e-05" -> 1e-05, "2.5e+06" -> 1e+05).
fn last_digit_unit(s: &str) -> f64 {
  let (mant, exp) = match s.find('e') {
    Some(k) => (&s[..k], s[k + 1..].parse::<i32>().unwrap()),
    None => (s, 0),
  };
  let decimals = match mant.find('.') {
    Some(k) => (mant.len() - k - 1) as i32,
    None => 0,
  };
  10f64.powi(exp - decimals)
}

/// Result of comparing one line.
enum LineCmp {
  Equal,
  /// Same text, decimals within tolerance: (max deviation in last-digit units, max absolute deviation).
  Within(f64, f64),
  /// Same text skeleton and integers, a decimal beyond tolerance: (units, abs).
  NumericBeyond(f64, f64),
  /// Different text or integers.
  Different,
}

fn compare_line(exp: &str, act: &str) -> LineCmp {
  if exp == act {
    return LineCmp::Equal;
  }
  let te = tokenize(exp);
  let ta = tokenize(act);
  if te.len() != ta.len() {
    return LineCmp::Different;
  }
  let mut max_units: f64 = 0.0;
  let mut max_abs: f64 = 0.0;
  let mut beyond = false;
  for (e, a) in te.iter().zip(ta.iter()) {
    match (e, a) {
      (Tok::Text(x), Tok::Text(y)) => {
        if x != y {
          return LineCmp::Different;
        }
      }
      (Tok::Num(x), Tok::Num(y)) => {
        if x == y {
          continue;
        }
        if !is_decimal(x) && !is_decimal(y) {
          return LineCmp::Different;
        }
        let (fx, fy) = (x.parse::<f64>().unwrap(), y.parse::<f64>().unwrap());
        let unit = last_digit_unit(x).max(last_digit_unit(y));
        let d = (fx - fy).abs();
        max_units = max_units.max(d / unit);
        max_abs = max_abs.max(d);
        let allowed = (TOL_UNITS * unit).max(TOL_REL * fx.abs().max(fy.abs()).max(1.0));
        if d > allowed * (1.0 + 1e-9) {
          beyond = true;
        }
      }
      _ => return LineCmp::Different,
    }
  }
  if !beyond { LineCmp::Within(max_units, max_abs) } else { LineCmp::NumericBeyond(max_units, max_abs) }
}

/// Per-section outcome.
#[derive(Debug, Clone)]
pub enum Outcome {
  Exact,
  /// (max deviation in last-digit units, max absolute deviation, number of lines that differ)
  Within(f64, f64, usize),
  /// First diverging line: (line index in the section, expected, actual, what)
  Diverged(usize, String, String, String),
  /// Not compared: reason.
  Pending(String),
}

fn filter_lines(lines: &[String], prefixes: &[&str]) -> Vec<String> {
  lines.iter().filter(|l| !prefixes.iter().any(|p| l.starts_with(p))).cloned().collect()
}

/// Compares one section line by line (after dropping backend log lines).
pub fn compare_section(expected: &Section, actual: &Section) -> Outcome {
  let e = filter_lines(&expected.lines, &EXPECTED_BACKEND_LOG_PREFIXES);
  let a = filter_lines(&actual.lines, &ACTUAL_BACKEND_LOG_PREFIXES);
  let mut max_units: f64 = 0.0;
  let mut max_abs: f64 = 0.0;
  let mut ndiff = 0;
  let n = e.len().max(a.len());
  for k in 0..n {
    let (el, al) = (e.get(k), a.get(k));
    match (el, al) {
      (Some(el), Some(al)) => match compare_line(el, al) {
        LineCmp::Equal => {}
        LineCmp::Within(u, d) => {
          max_units = max_units.max(u);
          max_abs = max_abs.max(d);
          ndiff += 1;
        }
        LineCmp::NumericBeyond(u, d) => {
          return Outcome::Diverged(k, el.clone(), al.clone(), format!("decimal beyond tolerance: {:.1} units, abs {:.3e}", u, d));
        }
        LineCmp::Different => return Outcome::Diverged(k, el.clone(), al.clone(), "text or integer differs".to_string()),
      },
      (Some(el), None) => return Outcome::Diverged(k, el.clone(), "<missing>".to_string(), "actual section is shorter".to_string()),
      (None, Some(al)) => return Outcome::Diverged(k, "<missing>".to_string(), al.clone(), "actual section is longer".to_string()),
      (None, None) => unreachable!(),
    }
  }
  if ndiff == 0 { Outcome::Exact } else { Outcome::Within(max_units, max_abs, ndiff) }
}

/// Compares only the lines of each section that start with one of `prefixes` (in order), for sections whose
/// other lines are thread-timing dependent upstream.
pub fn compare_section_summary_only(expected: &Section, actual: &Section, prefixes: &[&str]) -> Outcome {
  let pick = |s: &Section| -> Vec<String> { s.lines.iter().filter(|l| prefixes.iter().any(|p| l.starts_with(p))).cloned().collect() };
  let e = Section { label: expected.label.clone(), title: expected.title.clone(), lines: pick(expected) };
  let a = Section { label: actual.label.clone(), title: actual.title.clone(), lines: pick(actual) };
  compare_section(&e, &a)
}

/// How a section is compared.
pub enum Policy {
  /// Full line-by-line comparison (with the numeric tolerance).
  Full,
  /// Only the lines starting with these prefixes.
  SummaryOnly(&'static [&'static str]),
  /// Not compared: reason.
  Pending(&'static str),
}

/// A known divergence: a section label ("#k title") that may diverge, and why.
pub struct Allowed {
  pub label: &'static str,
  pub reason: &'static str,
}

/// Compares `actual` (split the same way) against `expected` sections. Returns the report and the number of
/// sections that diverged without being on `allowlist`. `policy_of(label)` gives each section's policy.
pub fn compare_all(
  name: &str,
  expected: &[Section],
  actual: &[Section],
  policy_of: &dyn Fn(&str) -> Policy,
  allowlist: &[Allowed],
  panic_msg: Option<&str>,
) -> (String, usize, usize) {
  let mut report = String::new();
  let mut num_bad = 0;
  let mut num_compared = 0;
  let (mut n_exact, mut n_within, mut n_div, mut n_allowed, mut n_pending) = (0, 0, 0, 0, 0);
  let mut used_allow: Vec<&str> = Vec::new();
  for (k, es) in expected.iter().enumerate() {
    let outcome: Outcome = match policy_of(&es.label) {
      Policy::Pending(reason) => Outcome::Pending(reason.to_string()),
      p => match actual.get(k) {
        None => Outcome::Diverged(0, es.lines.first().cloned().unwrap_or_default(), "<missing section>".to_string(), "actual output ended (panic?)".to_string()),
        Some(asec) if asec.title != es.title => {
          Outcome::Diverged(0, es.label.clone(), asec.label.clone(), "section titles differ (structure)".to_string())
        }
        Some(asec) => match p {
          Policy::Full => compare_section(es, asec),
          Policy::SummaryOnly(prefixes) => compare_section_summary_only(es, asec, prefixes),
          Policy::Pending(_) => unreachable!(),
        },
      },
    };
    let allowed = allowlist.iter().find(|a| a.label == es.label);
    if let Some(a) = allowed {
      used_allow.push(a.label);
    }
    match &outcome {
      Outcome::Exact => {
        n_exact += 1;
        num_compared += 1;
        let _ = writeln!(report, "  exact            {}{}", es.label, if allowed.is_some() { "   (allowlisted, but passes now)" } else { "" });
      }
      Outcome::Within(u, d, nd) => {
        n_within += 1;
        num_compared += 1;
        let _ = writeln!(
          report,
          "  within tol       {}   ({} lines differ, max {:.2} last-digit units, max abs {:.3e}){}",
          es.label,
          nd,
          u,
          d,
          if allowed.is_some() { "   (allowlisted, but passes now)" } else { "" }
        );
      }
      Outcome::Diverged(line, el, al, what) => {
        num_compared += 1;
        if let Some(a) = allowed {
          n_allowed += 1;
          let _ = writeln!(report, "  DIVERGED (allowed) {}   line {}: {}\n      reason: {}\n      eigen: {}\n      burn:  {}", es.label, line, what, a.reason, el, al);
        } else {
          n_div += 1;
          num_bad += 1;
          let _ = writeln!(report, "  DIVERGED         {}   line {}: {}\n      eigen: {}\n      burn:  {}", es.label, line, what, el, al);
        }
      }
      Outcome::Pending(reason) => {
        n_pending += 1;
        let _ = writeln!(report, "  pending          {}   ({})", es.label, reason);
      }
    }
  }
  if actual.len() > expected.len() {
    num_bad += 1;
    let _ = writeln!(report, "  EXTRA actual sections: {}", actual[expected.len()..].iter().map(|s| s.label.as_str()).collect::<Vec<_>>().join(", "));
  }
  for a in allowlist {
    if !used_allow.contains(&a.label) {
      let _ = writeln!(report, "  note: allowlist entry matches no section: {}", a.label);
    }
  }
  let head = format!(
    "{}: {} sections: {} exact, {} within tolerance, {} diverged (allowlisted), {} DIVERGED (not allowed), {} pending{}\n",
    name,
    expected.len(),
    n_exact,
    n_within,
    n_allowed,
    n_div,
    n_pending,
    panic_msg.map(|p| format!("\n  PANIC during the run: {}", p)).unwrap_or_default()
  );
  (head + &report, num_bad, num_compared)
}

//------------------------------------------------------------------------------------------------
// Running

#[cfg(not(target_arch = "wasm32"))]
pub fn install_panic_hook() {
  std::panic::set_hook(Box::new(|info| {
    let msg = if let Some(s) = info.payload().downcast_ref::<&str>() {
      s.to_string()
    } else if let Some(s) = info.payload().downcast_ref::<String>() {
      s.clone()
    } else {
      "<non-string panic payload>".to_string()
    };
    let at = info.location().map(|l| format!("{}:{}", l.file(), l.line())).unwrap_or_default();
    let s = format!("{} (at {})", msg, at);
    eprintln!("panic: {}", s);
    LAST_PANIC.with(|p| *p.borrow_mut() = s);
  }));
}

/// Runs `f` on a big-stack thread (Search ~40 KB, BoardHistory ~34 KB, recursive playout futures in debug),
/// catching a panic; returns the captured stdout and the panic message if any.
pub fn run_capturing(name: &str, f: fn()) -> (String, Option<String>) {
  std::thread::Builder::new()
    .name(name.to_string())
    .stack_size(512 << 20)
    .spawn(move || {
      install_panic_hook();
      init();
      take_out();
      let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
      let out = take_out();
      let p = match r {
        Ok(()) => None,
        Err(_) => Some(LAST_PANIC.with(|p| p.borrow().clone())),
      };
      (out, p)
    })
    .unwrap()
    .join()
    .expect("run_capturing thread")
}

/// Writes `text` to the system temp dir for inspection; returns the path.
pub fn dump_actual(name: &str, text: &str) -> String {
  let p = std::env::temp_dir().join(format!("katago-rs-{}.actual.txt", name));
  let _ = std::fs::write(&p, text);
  p.display().to_string()
}
