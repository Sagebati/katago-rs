//! Transcription (a test, not a port) of upstream cpp/tests/testsearchnonn.cpp `Tests::runNNLessSearchTests()`
//! @ v1.16.5, with the helpers it needs from cpp/tests/testsearchcommon.cpp as test-local copies.
//!
//! The test replays the same calls in the same order (same params, same seeds, same printing) through the
//! ported search with a `debugSkipNeuralNet` NNEvaluator, builds the whole stdout text, and compares it
//! byte-exact with fixtures/upstream-v1.16.5/unit/testsearchnonn/runnnlesssearchtests.expected0.txt (the
//! exact stdout slice from "Running neuralnetless search tests" to "Done"; backend independent). Both
//! texts are split into sections on the header trio (a line of 67 '=', a title line, another 67-'=' line)
//! and each section is compared with `common::compare_lines` and reported separately, so one failing
//! section does not hide the others. The board diagrams are the extracted `R"%%(...)%%"` blocks
//! (`runnnlesssearchtests.input<k>.txt`, k = order of appearance in the C++).
//!
//! Output interleaving. Upstream's Logger writes `": " + message + endl` to cout (logTime false) and the
//! expected text contains those lines (NNEvaluator's "Initializing neural net buffer...", the server
//! thread's "GPU 0 finishing..."). The port replaces Logger by the `log` crate, so this test installs a
//! `log::Log` that appends the same text to the same thread-local buffer as the test's own `cout`
//! writes (`Cout`). `delete nnEval` joins the server thread upstream (which logs as it ends); here it is
//! `kill_server_threads()` + `pool.run_until_stalled()` so the line lands in the same place.
//! Two `log::info!` calls of `SearchParams::print_params` stand for lines upstream writes straight to
//! `std::cout` (no ": " prefix); the logger below writes those two raw (see `PLAIN_COUT_LINES`).
//!
//! Every section is compared. "Board size distribution" runs a test-local copy of the GameInitializer
//! pieces it needs (program/play.cpp is [[ignored]]; see `GameInitializer` below). "SearchParams for
//! GTP/Analysis/Match/Benchmark/Other/Distributed" use the ported `Setup::loadSingleParams` (program/setup.rs).
//!
//! Timing dependence: none (no maxTime/time controls; the injected clock is a constant).
//! Not a port of upstream code.

#![allow(clippy::too_many_arguments, clippy::field_reassign_with_default, clippy::needless_range_loop, clippy::assign_op_pattern)]

mod common;

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::collections::HashMap;
use std::fmt::Write as _;
use std::rc::Rc;
use std::sync::OnceLock;

use common::Cases;
use common::compare_lines;
use common::fixture;
use common::init;
use common::run_or_stall_panic;
use futures::executor::LocalPool;
use katago_core::core::commontypes::enabled_t;
use katago_core::core::config_parser::ConfigParser;
use katago_core::core::global::KataError;
use katago_core::core::rand::Rand;
use katago_core::core::timer::set_clock;
use katago_core::game::board::*;
use katago_core::game::boardhistory::BoardHistory;
use katago_core::game::rules::Rules;
use katago_core::neuralnet::desc::ModelDesc;
use katago_core::neuralnet::nneval::NNEvaluator;
use katago_core::neuralnet::nneval::NNResultBuf;
use katago_core::neuralnet::nninputs::NNOutput;
use katago_core::neuralnet::nninputs::NNPos;
use katago_core::neuralnet::nninputs::ScoreValue;
use katago_core::neuralnet::nninterface::*;
use katago_core::port::executor::LocalBoxFuture;
use katago_core::port::executor::Spawner;
use katago_core::port::fmt::cpp_ostream_f64;
use katago_core::port::math::powf;
use katago_core::port::nlohmann;
use katago_core::port::node_arena::NodeIdx;
use katago_core::program::setup::Setup;
use katago_core::program::setup::setup_for_t;
use katago_core::search::reportedsearchvalues::ReportedSearchValues;
use katago_core::search::search::Search;
use katago_core::search::searchnode::SearchNode;
use katago_core::search::searchparams::SearchParams;
use katago_core::search::searchprint::PrintTreeOptions;
use katago_core::strprintf;
use katago_core::testAssert;

const F: &str = "testsearchnonn";

//------------------------------------------------------------------------------------------------
// cout and the Logger

thread_local! {
  /// Everything upstream would have written to stdout: the test's own `cout` text and the Logger lines.
  static OUT: RefCell<String> = const { RefCell::new(String::new()) };
  /// "message (at file:line)" of the last panic on this thread, for the per-section report.
  static LAST_PANIC: RefCell<String> = const { RefCell::new(String::new()) };
}

/// `cout`
struct Cout;

impl std::fmt::Write for Cout {
  fn write_str(&mut self, s: &str) -> std::fmt::Result {
    OUT.with(|o| o.borrow_mut().push_str(s));
    Ok(())
  }
}

macro_rules! cout {
  ($($t:tt)*) => {{
    write!(Cout, $($t)*).unwrap();
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

/// printParams writes these two lines to `std::cout` directly upstream (no Logger, so no ": ");
/// the port logs them with `log::info!`.
const PLAIN_COUT_LINES: [&str; 2] = ["avoidMYTDaggerHackPla: ", "playoutDoublingAdvantagePla: "];

/// `Logger logger(nullptr, logToStdout=false, logToStderr=false, logTime=false); logger.addOStream(cout);`:
/// `Logger::write(str)` = `": " + str + endl` on cout.
struct CoutLogger;

impl log::Log for CoutLogger {
  fn enabled(&self, _metadata: &log::Metadata) -> bool {
    true
  }
  fn log(&self, record: &log::Record) {
    let msg = format!("{}", record.args());
    if PLAIN_COUT_LINES.iter().any(|p| msg.starts_with(p)) {
      coutln!("{}", msg);
    } else {
      coutln!(": {}", msg);
    }
  }
  fn flush(&self) {}
}

static LOGGER: CoutLogger = CoutLogger;

/// Deterministic clock for `ClockTimer` (the search needs one installed; no time limit is set anywhere here).
fn fake_clock() -> f64 {
  1000.0
}

fn install_logger_and_clock() {
  static ONCE: OnceLock<()> = OnceLock::new();
  ONCE.get_or_init(|| {
    log::set_logger(&LOGGER).expect("no other logger in this test binary");
    log::set_max_level(log::LevelFilter::Trace);
  });
  set_clock(fake_clock);
}

//------------------------------------------------------------------------------------------------
// Backend that must never be reached (debugSkipNeuralNet), as in nneval_skipnn.rs.

struct NoBackend;

impl NeuralNet for NoBackend {
  fn global_initialize(&self) {}
  fn global_cleanup(&self) {}
  fn print_devices(&self) {}
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
    _: &str,
    _: bool,
    _: enabled_t,
    _: enabled_t,
    _: &dyn LoadedModel,
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

//------------------------------------------------------------------------------------------------
// The executor, and the test-local copies of the testsearchcommon.cpp helpers

/// One per section: the executor the NN server tasks and the search futures run on.
struct Ctx {
  pool: LocalPool,
  spawner: Rc<dyn Spawner>,
  backend: Rc<dyn NeuralNet>,
}

impl Ctx {
  fn new() -> Ctx {
    let pool = LocalPool::new();
    let spawner: Rc<dyn Spawner> = Rc::new(pool.spawner());
    Ctx { pool, spawner, backend: Rc::new(NoBackend) }
  }
}

/// upstream: TestSearchCommon::startNNEval (testsearchcommon.cpp) — test-local copy
/// PORT NOTE: `Logger&` dropped (log crate); `spawnServerThreads()` is async, driven to completion on
/// the pool; the `sleep_for(0.1)` ordering hack is not needed (single thread, deterministic).
fn start_nn_eval(
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
    ctx.backend.clone(),
    ctx.spawner.clone(),
    Vec::new(),
  )
  .unwrap();
  let nn_eval = Rc::new(nn_eval);

  run_or_stall_panic(&mut ctx.pool, "spawnServerThreads", nn_eval.spawn_server_threads()).unwrap();
  nn_eval
}

/// `delete nnEval;` (~NNEvaluator -> killServerThreads, which joins the server thread: it logs
/// "GPU 0 finishing..." as it ends). Everything else holding the evaluator must be dropped first.
fn delete_nn_eval(ctx: &mut Ctx, nn_eval: Rc<NNEvaluator>) {
  nn_eval.kill_server_threads();
  ctx.pool.run_until_stalled();
  assert_eq!(Rc::strong_count(&nn_eval), 1, "the serve task ended and released the evaluator");
  drop(nn_eval);
}

/// upstream: TestSearchCommon::verifyTreePostOrder (testsearchcommon.cpp) — test-local copy
fn verify_tree_post_order(search: &Search, only_require_at_least: i32) {
  //Enumerate the tree and make sure every node is indeed hit exactly once in postorder.
  let nodes: Vec<NodeIdx> = search.enumerate_tree_post_order();
  let mut idx_of_node: HashMap<NodeIdx, usize> = HashMap::new();
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

/// `std::sort` of libstdc++ (introsort: partition while more than 16 elements, then insertion sort),
/// which is NOT stable: sampleChosenMoves sorts counts that tie, and the expected order of the ties
/// is whatever libstdc++ produces. The heap-sort fallback (depth limit exhausted) is not replicated.
/// Test-local, no upstream counterpart.
fn cpp_std_sort<T: Clone>(v: &mut [T], comp: &dyn Fn(&T, &T) -> bool) {
  fn lg(n: usize) -> usize {
    (usize::BITS - 1 - n.leading_zeros()) as usize
  }
  fn move_median_to_first<T: Clone>(v: &mut [T], result: usize, a: usize, b: usize, c: usize, comp: &dyn Fn(&T, &T) -> bool) {
    if comp(&v[a], &v[b]) {
      if comp(&v[b], &v[c]) {
        v.swap(result, b);
      } else if comp(&v[a], &v[c]) {
        v.swap(result, c);
      } else {
        v.swap(result, a);
      }
    } else if comp(&v[a], &v[c]) {
      v.swap(result, a);
    } else if comp(&v[b], &v[c]) {
      v.swap(result, c);
    } else {
      v.swap(result, b);
    }
  }
  fn unguarded_partition<T: Clone>(v: &mut [T], mut first: usize, mut last: usize, pivot: usize, comp: &dyn Fn(&T, &T) -> bool) -> usize {
    loop {
      while comp(&v[first], &v[pivot]) {
        first += 1;
      }
      last -= 1;
      while comp(&v[pivot], &v[last]) {
        last -= 1;
      }
      if !(first < last) {
        return first;
      }
      v.swap(first, last);
      first += 1;
    }
  }
  fn introsort_loop<T: Clone>(v: &mut [T], first: usize, mut last: usize, mut depth_limit: usize, comp: &dyn Fn(&T, &T) -> bool) {
    while last - first > 16 {
      if depth_limit == 0 {
        panic!("cpp_std_sort: libstdc++'s heap-sort fallback is not replicated");
      }
      depth_limit -= 1;
      let mid = first + (last - first) / 2;
      move_median_to_first(v, first, first + 1, mid, last - 1, comp);
      let cut = unguarded_partition(v, first + 1, last, first, comp);
      introsort_loop(v, cut, last, depth_limit, comp);
      last = cut;
    }
  }
  fn unguarded_linear_insert<T: Clone>(v: &mut [T], last: usize, comp: &dyn Fn(&T, &T) -> bool) {
    let val = v[last].clone();
    let mut last = last;
    let mut next = last - 1;
    while comp(&val, &v[next]) {
      v[last] = v[next].clone();
      last = next;
      next -= 1;
    }
    v[last] = val;
  }
  fn insertion_sort<T: Clone>(v: &mut [T], first: usize, last: usize, comp: &dyn Fn(&T, &T) -> bool) {
    if first == last {
      return;
    }
    for i in first + 1..last {
      if comp(&v[i], &v[first]) {
        let val = v[i].clone();
        for k in (first..i).rev() {
          v[k + 1] = v[k].clone();
        }
        v[first] = val;
      } else {
        unguarded_linear_insert(v, i, comp);
      }
    }
  }
  let n = v.len();
  if n == 0 {
    return;
  }
  introsort_loop(v, 0, n, lg(n) * 2, comp);
  //__final_insertion_sort
  if n > 16 {
    insertion_sort(v, 0, 16, comp);
    for i in 16..n {
      unguarded_linear_insert(v, i, comp);
    }
  } else {
    insertion_sort(v, 0, n, comp);
  }
}

//------------------------------------------------------------------------------------------------
// Small helpers for the transcription

/// The k-th `R"%%(...)%%"` board diagram of testsearchnonn.cpp, in order of appearance, parsed with `Board::parseBoard`.
fn parse_board_input(k: usize, x_size: i32, y_size: i32) -> Board {
  let s = fixture(F, &format!("runnnlesssearchtests.input{}.txt", k));
  Board::parse_board(x_size, y_size, &s, '\n').unwrap()
}

fn root_node(search: &Search) -> Rc<SearchNode> {
  search.get_root_node().expect("search has a root node")
}

fn new_search(params: &SearchParams, nn_eval: &Rc<NNEvaluator>, seed: &str) -> Box<Search> {
  Box::new(Search::new(params, nn_eval.clone(), seed))
}

fn set_position(search: &mut Search, pla: Player, board: &Board, hist: &BoardHistory) {
  search.set_position(pla, board, hist).unwrap();
}

fn run_whole_search(ctx: &mut Ctx, search: &mut Search, pla: Player) {
  run_or_stall_panic(&mut ctx.pool, "runWholeSearch", search.run_whole_search(pla)).unwrap();
}

fn begin_search(ctx: &mut Ctx, search: &mut Search, pondering: bool) {
  run_or_stall_panic(&mut ctx.pool, "beginSearch", search.begin_search(pondering)).unwrap();
}

fn make_move(search: &mut Search, loc: Loc, pla: Player) {
  search.make_move(loc, pla).unwrap();
}

/// `search->printTree(cout, search->rootNode, options, perspective)`
fn print_tree(search: &Search, options: &PrintTreeOptions, perspective: Player) {
  let root = search.get_root_node();
  search.print_tree(&mut Cout, root.as_deref(), options, perspective);
}

/// `hist.makeBoardMoveAssumeLegal(board, Location::ofString(s, board), pla, NULL)`
fn play(hist: &mut BoardHistory, board: &mut Board, s: &str, pla: Player) {
  let loc = Location::of_string(s, board).unwrap();
  hist.make_board_move_assume_legal(board, loc, pla, None, false);
}

fn loc_of(s: &str, board: &Board) -> Loc {
  Location::of_string(s, board).unwrap()
}

/// The nlohmann::json getAnalysisJson call block repeated all over the C++.
fn analysis_json(
  search: &Search,
  perspective: Player,
  analysis_pv_len: i32,
  prevent_encore: bool,
  include_policy: bool,
  include_ownership: bool,
  include_ownership_stdev: bool,
  include_moves_ownership: bool,
  include_moves_ownership_stdev: bool,
  include_pv_visits: bool,
  include_no_result_value: bool,
) -> (bool, serde_json::Value) {
  let mut json = serde_json::Value::Null;
  let suc = search
    .get_analysis_json(
      perspective,
      analysis_pv_len,
      prevent_encore,
      include_policy,
      include_ownership,
      include_ownership_stdev,
      include_moves_ownership,
      include_moves_ownership_stdev,
      include_pv_visits,
      include_no_result_value,
      &mut json,
    )
    .unwrap();
  (suc, json)
}

/// `cout << json << endl`
fn cout_json(json: &serde_json::Value) {
  coutln!("{}", nlohmann::dump(json));
}

/// The common avoidMoveUntilByLoc setup of the three "with avoid moves" sections.
fn avoid_black_stones_setup(board: &mut Board) -> Vec<i32> {
  let mut avoid_move_until_by_loc: Vec<i32> = vec![0; Board::MAX_ARR_SIZE as usize];
  for y in 0..board.y_size {
    for x in 0..board.x_size {
      let loc = Location::get_loc(x, y, board.x_size);
      if board.colors[loc as usize] == C_BLACK {
        avoid_move_until_by_loc[loc as usize] = 0;
        board.set_stone(loc, C_EMPTY);
      } else {
        avoid_move_until_by_loc[loc as usize] = if y % 2 == 0 { 1 } else { 2 };
      }
    }
  }
  avoid_move_until_by_loc
}

//------------------------------------------------------------------------------------------------
// Test-local Setup::loadSingleParams

//------------------------------------------------------------------------------------------------
// The sections of Tests::runNNLessSearchTests(), in order. Each `fn` is the body of one `{ ... }` block
// (or of one header trio in the last block), after its three header couts, which the driver prints.

//Placeholder, doesn't actually do anything since we have debugSkipNeuralNet = true
const MODEL_FILE: &str = "/dev/null";

fn basic_search_and_chosen_move_randomization(ctx: &mut Ctx) {
  let nn_eval = start_nn_eval(ctx, MODEL_FILE, "", NNPos::MAX_BOARD_LEN, NNPos::MAX_BOARD_LEN, 0, true, false, false, true, false);
  let mut params = SearchParams::new();
  params.max_visits = 100;
  let mut search = new_search(&params, &nn_eval, "autoSearchRandSeed");
  let rules = Rules::get_tromp_taylorish();

  let board = parse_board_input(0, 9, 9);
  let next_pla = P_BLACK;
  let hist = BoardHistory::new_full(&board, next_pla, &rules, 0);

  set_position(&mut search, next_pla, &board, &hist);
  run_whole_search(ctx, &mut search, next_pla);

  let options = PrintTreeOptions::new().max_depth(1);
  print_tree(&search, &options, P_WHITE);

  let sample_chosen_moves = |search: &Search| {
    let mut move_locs_and_counts: BTreeMap<Loc, i32> = BTreeMap::new();
    for _ in 0..10000 {
      let loc = search.get_chosen_move_loc();
      *move_locs_and_counts.entry(loc).or_insert(0) += 1;
    }
    let mut move_locs_and_counts_sorted: Vec<(Loc, i32)> = move_locs_and_counts.iter().map(|(k, v)| (*k, *v)).collect();
    cpp_std_sort(&mut move_locs_and_counts_sorted, &|a: &(Loc, i32), b: &(Loc, i32)| a.1 > b.1);

    for i in 0..move_locs_and_counts_sorted.len() {
      coutln!("{} {}", Location::to_string(move_locs_and_counts_sorted[i].0, &board), move_locs_and_counts_sorted[i].1);
    }
  };

  coutln!("Chosen moves at temperature 0");
  sample_chosen_moves(&search);

  {
    coutln!("Chosen moves at temperature 1 but early temperature 0, when it's perfectly early");
    search.search_params.chosen_move_temperature = 1.0;
    search.search_params.chosen_move_temperature_early = 0.0;
    sample_chosen_moves(&search);
  }

  {
    coutln!("Chosen moves at temperature 1");
    search.search_params.chosen_move_temperature = 1.0;
    search.search_params.chosen_move_temperature_early = 1.0;
    sample_chosen_moves(&search);
  }

  {
    coutln!("Chosen moves at some intermediate temperature");
    //Ugly hack to artifically fill history. Breaks all sorts of invariants, but should work to
    //make the search htink there's some history to choose an intermediate temperature
    for _ in 0..16 {
      search.root_history.move_history.push(Move::new(Board::NULL_LOC, P_BLACK));
    }

    search.search_params.chosen_move_temperature = 1.0;
    search.search_params.chosen_move_temperature_early = 0.0;
    search.search_params.chosen_move_temperature_halflife = 16.0 * 19.0 / 9.0;
    sample_chosen_moves(&search);
  }

  drop(search);
  delete_nn_eval(ctx, nn_eval);
  coutln!();
}

fn preservation_of_search_tree_across_moves(ctx: &mut Ctx) {
  let nn_eval = start_nn_eval(ctx, MODEL_FILE, "", NNPos::MAX_BOARD_LEN, NNPos::MAX_BOARD_LEN, 0, true, false, false, true, false);
  let mut params = SearchParams::new();
  params.max_visits = 100;
  params.cpuct_exploration *= 2.0;
  let mut search = new_search(&params, &nn_eval, "autoSearchRandSeed");
  let rules = Rules::get_tromp_taylorish();

  let board = parse_board_input(1, 7, 7);
  let mut next_pla = P_BLACK;
  let hist = BoardHistory::new_full(&board, next_pla, &rules, 0);

  {
    //--------------------------------------
    coutln!("First perform a basic search.");

    set_position(&mut search, next_pla, &board, &hist);
    run_whole_search(ctx, &mut search, next_pla);

    let loc_to_descend: Loc;
    {
      let root = root_node(&search);
      let children = root.get_children();
      let children_capacity = children.get_capacity();
      testAssert!(children_capacity > 1);

      //In theory nothing requires this, but it would be kind of crazy if this were false
      testAssert!(children.iterate_and_count_children() > 1);
      testAssert!(children.index(1).get_if_allocated().is_some());

      loc_to_descend = children.index(1).get_move_loc();
    }

    let options = PrintTreeOptions::new().max_depth(1);
    coutln!("{}", search.root_board);
    print_tree(&search, &options, P_WHITE);
    print_tree(&search, &options.only_branch(&board, &Location::to_string(loc_to_descend, &board)).unwrap(), P_WHITE);

    coutln!();

    //--------------------------------------
    coutln!("Next, make a move, and with no search, print the tree.");

    make_move(&mut search, loc_to_descend, next_pla);
    next_pla = get_opp(next_pla);

    coutln!("{}", search.root_board);
    print_tree(&search, &options, P_WHITE);
    coutln!();

    //--------------------------------------
    coutln!("Then continue the search to complete 100 visits.");

    run_whole_search(ctx, &mut search, next_pla);
    print_tree(&search, &options, P_WHITE);
    coutln!();
  }

  drop(search);
  delete_nn_eval(ctx, nn_eval);

  coutln!();
}

fn has_suicide_root_moves(search: &Search) -> bool {
  let root = root_node(search);
  let children = root.get_children();
  let children_capacity = children.get_capacity();
  for i in 0..children_capacity {
    let child = children.index(i).get_if_allocated();
    if child.is_none() {
      break;
    }
    if search.root_board.is_suicide(children.index(i).get_move_loc(), search.root_pla) {
      return true;
    }
  }
  false
}

fn has_pass_alive_root_moves(search: &Search) -> bool {
  let root = root_node(search);
  let children = root.get_children();
  let children_capacity = children.get_capacity();
  for i in 0..children_capacity {
    let child = children.index(i).get_if_allocated();
    if child.is_none() {
      break;
    }
    if search.root_safe_area[children.index(i).get_move_loc() as usize] != C_EMPTY {
      return true;
    }
  }
  false
}

fn pruning_of_search_tree_across_moves_due_to_root_restrictions(ctx: &mut Ctx) {
  let mut board = parse_board_input(2, 7, 7);
  let mut next_pla = P_BLACK;
  let rules = Rules::get_tromp_taylorish();
  let mut hist = BoardHistory::new_full(&board, next_pla, &rules, 0);
  play(&mut hist, &mut board, "B5", next_pla);
  next_pla = get_opp(next_pla);
  play(&mut hist, &mut board, "pass", next_pla);
  next_pla = get_opp(next_pla);
  play(&mut hist, &mut board, "C6", next_pla);
  next_pla = get_opp(next_pla);
  play(&mut hist, &mut board, "pass", next_pla);
  next_pla = get_opp(next_pla);
  play(&mut hist, &mut board, "G7", next_pla);
  next_pla = get_opp(next_pla);
  play(&mut hist, &mut board, "pass", next_pla);
  next_pla = get_opp(next_pla);
  play(&mut hist, &mut board, "F3", next_pla);
  next_pla = get_opp(next_pla);
  play(&mut hist, &mut board, "pass", next_pla);
  next_pla = get_opp(next_pla);

  {
    coutln!("First with no pruning");
    let nn_eval = start_nn_eval(ctx, MODEL_FILE, "seed1", NNPos::MAX_BOARD_LEN, NNPos::MAX_BOARD_LEN, 0, true, false, false, true, false);
    let mut params = SearchParams::new();
    params.max_visits = 400;
    let mut search = new_search(&params, &nn_eval, "autoSearchRandSeed3");

    set_position(&mut search, next_pla, &board, &hist);
    run_whole_search(ctx, &mut search, next_pla);
    let options = PrintTreeOptions::new().max_depth(1);
    coutln!("{}", search.root_board);
    print_tree(&search, &options, P_WHITE);

    testAssert!(has_suicide_root_moves(&search));

    drop(search);
    delete_nn_eval(ctx, nn_eval);

    coutln!();
  }

  {
    coutln!("Next, with rootPruneUselessMoves");
    let nn_eval = start_nn_eval(ctx, MODEL_FILE, "seed1", NNPos::MAX_BOARD_LEN, NNPos::MAX_BOARD_LEN, 0, true, false, false, true, false);
    let mut params = SearchParams::new();
    params.max_visits = 400;
    params.root_prune_useless_moves = true;
    let mut search = new_search(&params, &nn_eval, "autoSearchRandSeed3");

    set_position(&mut search, next_pla, &board, &hist);
    run_whole_search(ctx, &mut search, next_pla);
    let options = PrintTreeOptions::new().max_depth(1);
    coutln!("{}", search.root_board);
    print_tree(&search, &options, P_WHITE);

    testAssert!(!has_suicide_root_moves(&search));

    drop(search);
    delete_nn_eval(ctx, nn_eval);

    coutln!();
  }

  coutln!("Progress the game, having black fill space while white passes...");
  play(&mut hist, &mut board, "A7", next_pla);
  next_pla = get_opp(next_pla);
  play(&mut hist, &mut board, "pass", next_pla);
  next_pla = get_opp(next_pla);
  play(&mut hist, &mut board, "E7", next_pla);
  next_pla = get_opp(next_pla);
  play(&mut hist, &mut board, "pass", next_pla);
  next_pla = get_opp(next_pla);
  play(&mut hist, &mut board, "F7", next_pla);
  next_pla = get_opp(next_pla);

  {
    coutln!("Searching on the opponent, the move before");
    let nn_eval = start_nn_eval(ctx, MODEL_FILE, "seed1b", NNPos::MAX_BOARD_LEN, NNPos::MAX_BOARD_LEN, 0, true, false, false, true, false);
    let mut params = SearchParams::new();
    params.max_visits = 400;
    params.root_prune_useless_moves = true;
    let mut search = new_search(&params, &nn_eval, "autoSearchRandSeed3");

    set_position(&mut search, next_pla, &board, &hist);
    run_whole_search(ctx, &mut search, next_pla);
    let options = PrintTreeOptions::new().max_depth(1);
    coutln!("{}", search.root_board);
    print_tree(&search, &options, P_WHITE);
    print_tree(&search, &options.only_branch(&board, "pass").unwrap(), P_WHITE);

    coutln!();

    coutln!("Now play forward the pass. The tree should still have useless suicides and also other moves in it");
    make_move(&mut search, Board::PASS_LOC, next_pla);
    testAssert!(has_suicide_root_moves(&search));
    testAssert!(has_pass_alive_root_moves(&search));

    coutln!("{}", search.root_board);
    print_tree(&search, &options, P_WHITE);

    coutln!();

    coutln!("But the moment we begin a search, it should no longer.");
    begin_search(ctx, &mut search, false);
    testAssert!(!has_suicide_root_moves(&search));
    testAssert!(!has_pass_alive_root_moves(&search));

    coutln!("{}", search.root_board);
    print_tree(&search, &options, P_WHITE);

    coutln!();

    coutln!("Continue searching a bit more");
    run_whole_search(ctx, &mut search, get_opp(next_pla));

    coutln!("{}", search.root_board);
    print_tree(&search, &options, P_WHITE);

    drop(search);
    delete_nn_eval(ctx, nn_eval);
    coutln!();
  }
}

fn search_tree_update_near_terminal_positions(ctx: &mut Ctx) {
  let board = parse_board_input(3, 7, 7);

  let next_pla = P_WHITE;
  let mut rules = Rules::get_tromp_taylorish();
  rules.multi_stone_suicide_legal = false;
  let hist = BoardHistory::new_full(&board, next_pla, &rules, 0);

  {
    coutln!("First with no pruning");
    let nn_eval = start_nn_eval(ctx, MODEL_FILE, "seed1", NNPos::MAX_BOARD_LEN, NNPos::MAX_BOARD_LEN, 0, true, false, false, true, false);
    let mut params = SearchParams::new();
    params.max_visits = 400;
    params.dynamic_score_utility_factor = 0.5;
    params.use_lcb_for_selection = true;

    let mut search = new_search(&params, &nn_eval, "autoSearchRandSeed3");

    set_position(&mut search, next_pla, &board, &hist);
    run_whole_search(ctx, &mut search, next_pla);
    let options = PrintTreeOptions::new().max_depth(1).print_sqs(true);
    coutln!("{}", search.root_board);
    print_tree(&search, &options, P_WHITE);

    coutln!("Begin search is idempotent?");
    begin_search(ctx, &mut search, false);
    print_tree(&search, &options, P_WHITE);
    make_move(&mut search, loc_of("B1", &board), next_pla);
    print_tree(&search, &options, P_WHITE);
    begin_search(ctx, &mut search, false);
    print_tree(&search, &options, P_WHITE);

    drop(search);
    delete_nn_eval(ctx, nn_eval);

    coutln!();
  }
}

fn root_symmetry_pruning_empty_board(ctx: &mut Ctx) {
  let nn_eval = start_nn_eval(ctx, MODEL_FILE, "", 9, 9, 0, true, false, false, true, false);
  let mut params = SearchParams::new();
  params.max_visits = 5000;
  params.cpuct_exploration *= 4.0;
  params.root_symmetry_pruning = true;
  let mut search = new_search(&params, &nn_eval, "autoSearchRandSeed");
  let rules = Rules::get_tromp_taylorish();

  let board = parse_board_input(4, 9, 9);
  let mut next_pla = P_BLACK;
  let hist = BoardHistory::new_full(&board, next_pla, &rules, 0);

  set_position(&mut search, next_pla, &board, &hist);
  run_whole_search(ctx, &mut search, next_pla);

  coutln!("{}", search.root_board);

  let options = PrintTreeOptions::new().max_depth(1);
  print_tree(&search, &options, P_WHITE);
  //Print the subtree under a move
  coutln!("H8 subtree: ");
  let loc_to_descend = loc_of("H8", &search.root_board);
  print_tree(
    &search,
    &options.only_branch(&search.root_board, &Location::to_string(loc_to_descend, &search.root_board)).unwrap(),
    P_WHITE,
  );

  //Enumerate the tree and make sure every node is indeed hit exactly once in postorder.
  verify_tree_post_order(&search, -1);

  //--------------------------------------
  coutln!("Next, make a move, and with no search, print the tree.");

  make_move(&mut search, loc_to_descend, next_pla);
  next_pla = get_opp(next_pla);
  let _ = next_pla;

  coutln!("{}", search.root_board);
  print_tree(&search, &options, P_WHITE);
  coutln!();

  //Enumerate the tree and make sure every node is indeed hit exactly once in postorder.
  verify_tree_post_order(&search, -1);

  //--------------------------------------
  coutln!("Begin search but make no additional playouts, print the tree.");
  let pondering = false;
  begin_search(ctx, &mut search, pondering);
  print_tree(&search, &options, P_WHITE);
  coutln!();

  //Enumerate the tree and make sure every node is indeed hit exactly once in postorder.
  verify_tree_post_order(&search, -1);

  drop(search);
  delete_nn_eval(ctx, nn_eval);
  coutln!();
}

fn root_symmetry_pruning_asymmetry_board(ctx: &mut Ctx) {
  let nn_eval = start_nn_eval(ctx, MODEL_FILE, "", 9, 9, 0, true, false, false, true, false);
  let mut params = SearchParams::new();
  params.max_visits = 2000;
  params.root_symmetry_pruning = true;
  let mut search = new_search(&params, &nn_eval, "autoSearchRandSeed");
  let rules = Rules::get_tromp_taylorish();

  let board = parse_board_input(5, 9, 9);
  let next_pla = P_BLACK;
  let hist = BoardHistory::new_full(&board, next_pla, &rules, 0);

  set_position(&mut search, next_pla, &board, &hist);
  run_whole_search(ctx, &mut search, next_pla);

  coutln!("{}", search.root_board);

  let options = PrintTreeOptions::new().max_depth(1);
  print_tree(&search, &options, P_WHITE);

  drop(search);
  delete_nn_eval(ctx, nn_eval);
  coutln!();
}

fn root_symmetry_pruning_diagonal_flips_empty_board(ctx: &mut Ctx) {
  let nn_eval = start_nn_eval(ctx, MODEL_FILE, "", 9, 9, 0, true, false, false, true, false);
  let mut params = SearchParams::new();
  params.max_visits = 5000;
  params.root_symmetry_pruning = true;
  let mut search = new_search(&params, &nn_eval, "autoSearchRandSeed");
  let rules = Rules::get_tromp_taylorish();

  let board = parse_board_input(6, 9, 9);
  let next_pla = P_BLACK;
  let hist = BoardHistory::new_full(&board, next_pla, &rules, 0);

  set_position(&mut search, next_pla, &board, &hist);
  search.set_root_symmetry_pruning_only(&[0, 3, 4, 7]);
  run_whole_search(ctx, &mut search, next_pla);

  coutln!("{}", search.root_board);

  let options = PrintTreeOptions::new().max_depth(1);
  print_tree(&search, &options, P_WHITE);

  drop(search);
  delete_nn_eval(ctx, nn_eval);
  coutln!();
}

fn root_symmetry_pruning_diagonal_flips_one_diagonal(ctx: &mut Ctx) {
  let nn_eval = start_nn_eval(ctx, MODEL_FILE, "", 9, 9, 0, true, false, false, true, false);
  let mut params = SearchParams::new();
  params.max_visits = 5000;
  params.root_symmetry_pruning = true;
  let mut search = new_search(&params, &nn_eval, "autoSearchRandSeed");
  let rules = Rules::get_tromp_taylorish();

  let board = parse_board_input(7, 9, 9);
  let next_pla = P_BLACK;
  let hist = BoardHistory::new_full(&board, next_pla, &rules, 0);

  set_position(&mut search, next_pla, &board, &hist);
  search.set_root_symmetry_pruning_only(&[0, 3, 4, 7]);
  run_whole_search(ctx, &mut search, next_pla);

  coutln!("{}", search.root_board);

  let options = PrintTreeOptions::new().max_depth(1);
  print_tree(&search, &options, P_WHITE);

  drop(search);
  delete_nn_eval(ctx, nn_eval);
  coutln!();
}

/// The body shared (verbatim in the C++) by the three "with avoid moves" sections.
fn avoid_moves_section(ctx: &mut Ctx, input_idx: usize, root_symmetry_pruning: bool) {
  let nn_eval = start_nn_eval(ctx, MODEL_FILE, "", 9, 9, 0, true, false, false, true, false);
  let mut params = SearchParams::new();
  params.max_visits = 5000;
  params.root_symmetry_pruning = root_symmetry_pruning;
  params.wide_root_noise = 0.05;
  let mut search = new_search(&params, &nn_eval, "autoSearchRandSeed");
  let rules = Rules::get_tromp_taylorish();

  let mut board = parse_board_input(input_idx, 9, 9);

  let avoid_move_until_by_loc = avoid_black_stones_setup(&mut board);

  let next_pla = P_BLACK;
  let hist = BoardHistory::new_full(&board, next_pla, &rules, 0);

  set_position(&mut search, next_pla, &board, &hist);
  search.set_root_symmetry_pruning_only(&[0, 3, 4, 7]);
  search.set_avoid_move_until_by_loc(&avoid_move_until_by_loc, &avoid_move_until_by_loc);
  run_whole_search(ctx, &mut search, next_pla);

  coutln!("{}", search.root_board);

  let options = PrintTreeOptions::new().max_depth(1);
  print_tree(&search, &options, P_WHITE);

  let perspective = P_WHITE;
  let analysis_pv_len = 2;
  let prevent_encore = true;
  let include_policy = false;
  let include_ownership = false;
  let include_ownership_stdev = false;
  let include_moves_ownership = false;
  let include_moves_ownership_stdev = false;
  let include_pv_visits = false;
  let include_no_result_value = false;
  let (suc, json) = analysis_json(
    &search,
    perspective,
    analysis_pv_len,
    prevent_encore,
    include_policy,
    include_ownership,
    include_ownership_stdev,
    include_moves_ownership,
    include_moves_ownership_stdev,
    include_pv_visits,
    include_no_result_value,
  );
  testAssert!(suc);
  cout_json(&json);

  drop(search);
  delete_nn_eval(ctx, nn_eval);
  coutln!();
}

fn root_symmetry_pruning_diagonal_flips_empty_board_avoid_moves(ctx: &mut Ctx) {
  avoid_moves_section(ctx, 8, true);
}

fn root_symmetry_pruning_diagonal_flips_one_diagonal_avoid_moves(ctx: &mut Ctx) {
  avoid_moves_section(ctx, 9, true);
}

fn root_pruning_no_symmetries_avoid_moves(ctx: &mut Ctx) {
  avoid_moves_section(ctx, 10, false);
}

fn non_square_board_search(ctx: &mut Ctx) {
  let nn_eval = start_nn_eval(ctx, MODEL_FILE, "", 7, 17, 0, true, false, false, true, false);
  let mut params = SearchParams::new();
  params.max_visits = 100;
  let mut search = new_search(&params, &nn_eval, "autoSearchRandSeed");
  let rules = Rules::get_tromp_taylorish();

  let board = parse_board_input(11, 7, 17);
  let next_pla = P_BLACK;
  let hist = BoardHistory::new_full(&board, next_pla, &rules, 0);

  set_position(&mut search, next_pla, &board, &hist);
  run_whole_search(ctx, &mut search, next_pla);

  coutln!("{}", search.root_board);

  let options = PrintTreeOptions::new().max_depth(1);
  print_tree(&search, &options, P_WHITE);

  drop(search);
  delete_nn_eval(ctx, nn_eval);
  coutln!();
}

fn visualize_dirichlet_noise(_ctx: &mut Ctx) {
  let mut params = SearchParams::new();
  params.root_noise_enabled = true;
  let mut rand = Rand::new_seed("noiseVisualize");

  let mut run = |x_size: i32, y_size: i32| {
    let board = Board::new_x_y(x_size, y_size).unwrap();
    let nn_x_len = 19;
    let nn_y_len = 19;
    let mut sum: f32 = 0.0;
    let mut counter: i32 = 0;

    let mut orig_policy_probs = [0.0f32; NNPos::MAX_NN_POLICY_SIZE as usize];
    let mut policy_probs = [0.0f32; NNPos::MAX_NN_POLICY_SIZE as usize];
    policy_probs.fill(-1.0f32);
    {
      for y in 0..board.y_size {
        for x in 0..board.x_size {
          let pos = NNPos::xy_to_pos(x, y, nn_x_len) as usize;
          policy_probs[pos] = powf(0.9, counter as f64) as f32;
          counter += 1;
          sum += policy_probs[pos];
        }
      }
      let pos = NNPos::loc_to_pos(Board::PASS_LOC, board.x_size, nn_x_len, nn_y_len) as usize;
      policy_probs[pos] = powf(0.9, counter as f64) as f32;
      counter += 1;
      let _ = counter;
      sum += policy_probs[pos];

      for i in 0..NNPos::MAX_NN_POLICY_SIZE as usize {
        if policy_probs[i] as f64 >= 0.0 {
          policy_probs[i] /= sum;
        }
      }
    }

    orig_policy_probs.copy_from_slice(&policy_probs);
    Search::add_dirichlet_noise(&params, &mut rand, NNPos::MAX_NN_POLICY_SIZE, &mut policy_probs[..]).unwrap();

    {
      for y in 0..board.y_size {
        for x in 0..board.x_size {
          let pos = NNPos::xy_to_pos(x, y, nn_x_len) as usize;
          cout!("{}", strprintf!("%+6.2f ", 100.0 * ((policy_probs[pos] - orig_policy_probs[pos]) as f64)));
        }
        coutln!();
      }
      let pos = NNPos::loc_to_pos(Board::PASS_LOC, board.x_size, nn_x_len, nn_y_len) as usize;
      cout!("{}", strprintf!("%+6.2f ", 100.0 * ((policy_probs[pos] - orig_policy_probs[pos]) as f64)));
      coutln!();
    }
  };

  run(19, 19);
  run(11, 7);
}

fn search_tolerates_moving_past_game_end(ctx: &mut Ctx) {
  let nn_eval = start_nn_eval(ctx, MODEL_FILE, "", 7, 7, 0, true, false, false, true, false);
  let mut params = SearchParams::new();
  params.max_visits = 200;
  let mut search = new_search(&params, &nn_eval, "autoSearchRandSeed");
  let mut search2 = new_search(&params, &nn_eval, "autoSearchRandSeed");
  let mut search3 = new_search(&params, &nn_eval, "autoSearchRandSeed");
  let rules = Rules::get_tromp_taylorish();
  let options = PrintTreeOptions::new().max_depth(1);

  let mut board = parse_board_input(12, 7, 7);
  let mut next_pla = P_WHITE;
  let mut hist = BoardHistory::new_full(&board, next_pla, &rules, 0);

  set_position(&mut search, next_pla, &board, &hist);
  set_position(&mut search2, next_pla, &board, &hist);
  set_position(&mut search3, next_pla, &board, &hist);

  make_move(&mut search, loc_of("C7", &board), next_pla);
  make_move(&mut search2, loc_of("C7", &board), next_pla);
  play(&mut hist, &mut board, "C7", next_pla);
  next_pla = get_opp(next_pla);
  set_position(&mut search3, next_pla, &board, &hist);

  make_move(&mut search, loc_of("pass", &board), next_pla);
  make_move(&mut search2, loc_of("pass", &board), next_pla);
  play(&mut hist, &mut board, "pass", next_pla);
  next_pla = get_opp(next_pla);
  set_position(&mut search3, next_pla, &board, &hist);
  board.check_consistency().unwrap();

  run_whole_search(ctx, &mut search2, next_pla);

  make_move(&mut search, loc_of("pass", &board), next_pla);
  make_move(&mut search2, loc_of("pass", &board), next_pla);
  board.check_consistency().unwrap();
  play(&mut hist, &mut board, "pass", next_pla);
  next_pla = get_opp(next_pla);
  set_position(&mut search3, next_pla, &board, &hist);

  testAssert!(hist.is_game_finished);

  run_whole_search(ctx, &mut search, next_pla);
  run_whole_search(ctx, &mut search2, next_pla);
  run_whole_search(ctx, &mut search3, next_pla);

  hist.print_debug_info(&mut Cout, &board);
  coutln!("Search made move after gameover");
  print_tree(&search, &options, P_WHITE);
  coutln!("Search made move (carrying tree over) after gameover");
  print_tree(&search2, &options, P_WHITE);
  coutln!("Position was set after gameover");
  print_tree(&search3, &options, P_WHITE);

  coutln!("Recapturing ko after two passes and supposed game over (violates superko)");
  make_move(&mut search, loc_of("D7", &board), next_pla);
  make_move(&mut search2, loc_of("D7", &board), next_pla);
  play(&mut hist, &mut board, "D7", next_pla);
  next_pla = get_opp(next_pla);
  set_position(&mut search3, next_pla, &board, &hist);

  run_whole_search(ctx, &mut search, next_pla);
  run_whole_search(ctx, &mut search2, next_pla);
  run_whole_search(ctx, &mut search3, next_pla);

  hist.print_debug_info(&mut Cout, &board);
  coutln!("Search made move");
  print_tree(&search, &options, P_WHITE);
  coutln!("Search made move (carrying tree over)");
  print_tree(&search2, &options, P_WHITE);
  coutln!("Position was set");
  print_tree(&search3, &options, P_WHITE);

  drop(search);
  drop(search2);
  drop(search3);
  delete_nn_eval(ctx, nn_eval);
  coutln!();
}

/// The "search, make the chosen move, search again, ..." sequence shared by the value bias and graph search sections.
fn search_make_move_search_again(ctx: &mut Ctx, search: &mut Search, next_pla: Player, options: &PrintTreeOptions, verify_post_order_after_second_search: bool) {
  run_whole_search(ctx, search, next_pla);
  coutln!("{}", search.root_board);
  print_tree(search, options, P_WHITE);

  coutln!("Making move");
  let loc = search.get_chosen_move_loc();
  make_move(search, loc, next_pla);
  coutln!("{}", search.root_board);
  print_tree(search, options, P_WHITE);

  coutln!("Searching again");
  run_whole_search(ctx, search, next_pla);
  coutln!("{}", search.root_board);
  print_tree(search, options, P_WHITE);
  if verify_post_order_after_second_search {
    verify_tree_post_order(search, -1);
  }

  coutln!("Making move");
  let loc = search.get_chosen_move_loc();
  make_move(search, loc, next_pla);
  coutln!("{}", search.root_board);
  print_tree(search, options, P_WHITE);

  coutln!("Searching again");
  run_whole_search(ctx, search, next_pla);
  coutln!("{}", search.root_board);
  print_tree(search, options, P_WHITE);

  coutln!("Making move");
  let loc = search.get_chosen_move_loc();
  make_move(search, loc, next_pla);
  coutln!("{}", search.root_board);
  print_tree(search, options, P_WHITE);
}

fn integrity_of_value_bias(ctx: &mut Ctx) {
  let nn_eval = start_nn_eval(ctx, MODEL_FILE, "", 7, 7, 0, true, false, false, true, false);
  let mut params = SearchParams::new();
  params.max_visits = 500;
  params.subtree_value_bias_factor = 0.5;
  params.chosen_move_temperature = 0.0;
  let mut search = new_search(&params, &nn_eval, "autoSearchRandSeed");
  let rules = Rules::get_tromp_taylorish();
  let board = parse_board_input(13, 7, 7);
  let next_pla = P_BLACK;
  let hist = BoardHistory::new_full(&board, next_pla, &rules, 0);

  let options = PrintTreeOptions::new().max_depth(1);

  set_position(&mut search, next_pla, &board, &hist);

  //(in the C++ the three rounds are written out; the first round has no post-order check)
  search_make_move_search_again(ctx, &mut search, next_pla, &options, false);

  drop(search);
  delete_nn_eval(ctx, nn_eval);
  coutln!();
}

fn value_bias_with_ko(ctx: &mut Ctx) {
  let nn_eval = start_nn_eval(ctx, MODEL_FILE, "seeeed", 14, 14, 0, true, false, false, true, false);
  let mut params = SearchParams::new();
  params.max_visits = 2000;
  params.subtree_value_bias_factor = 0.8;
  params.chosen_move_temperature = 0.0;
  let mut search = new_search(&params, &nn_eval, "seeeed");
  let rules = Rules::parse_rules("japanese").unwrap();
  let board = parse_board_input(14, 14, 14);
  let next_pla = P_BLACK;
  let hist = BoardHistory::new_full(&board, next_pla, &rules, 0);

  let options = PrintTreeOptions::new().max_depth(1);

  set_position(&mut search, next_pla, &board, &hist);

  search_make_move_search_again(ctx, &mut search, next_pla, &options, false);

  drop(search);
  delete_nn_eval(ctx, nn_eval);
  coutln!();
}

fn analysis_json_section(ctx: &mut Ctx) {
  let nn_eval = start_nn_eval(ctx, MODEL_FILE, "", 9, 9, 0, true, false, false, true, false);
  let mut params = SearchParams::new();
  params.max_visits = 10;
  params.subtree_value_bias_factor = 0.5;
  params.chosen_move_temperature = 0.0;
  let mut search = new_search(&params, &nn_eval, "autoSearchRandSeed");
  search.set_always_include_owner_map(true);
  let rules = Rules::get_tromp_taylorish();
  let board = parse_board_input(15, 7, 7);
  let next_pla = P_BLACK;
  let hist = BoardHistory::new_full(&board, next_pla, &rules, 0);

  let options = PrintTreeOptions::new().max_depth(1);

  set_position(&mut search, next_pla, &board, &hist);
  run_whole_search(ctx, &mut search, next_pla);
  print_tree(&search, &options, P_WHITE);
  let perspective = P_WHITE;
  let analysis_pv_len = 2;
  let prevent_encore = true;
  let include_policy = true;
  let include_ownership = true;
  let include_ownership_stdev = false;
  let include_moves_ownership = false;
  let include_moves_ownership_stdev = false;
  let include_pv_visits = true;
  let include_no_result_value = false;
  let (suc, json) = analysis_json(
    &search,
    perspective,
    analysis_pv_len,
    prevent_encore,
    include_policy,
    include_ownership,
    include_ownership_stdev,
    include_moves_ownership,
    include_moves_ownership_stdev,
    include_pv_visits,
    include_no_result_value,
  );
  testAssert!(suc);
  cout_json(&json);

  drop(search);
  delete_nn_eval(ctx, nn_eval);
}

fn analysis_json_moves_ownership(ctx: &mut Ctx, root_symmetry_pruning: bool, input_idx: usize) {
  let nn_eval = start_nn_eval(ctx, MODEL_FILE, "movesown", 9, 9, 0, true, false, false, true, false);
  let mut params = SearchParams::new();
  params.max_visits = 4;
  params.subtree_value_bias_factor = 0.5;
  params.chosen_move_temperature = 0.0;
  if root_symmetry_pruning {
    params.root_symmetry_pruning = true;
  }
  let mut search = new_search(&params, &nn_eval, "autoSearchRandSeed");
  search.set_always_include_owner_map(true);
  let rules = Rules::get_tromp_taylorish();
  let board = parse_board_input(input_idx, 7, 7);
  let next_pla = P_BLACK;
  let hist = BoardHistory::new_full(&board, next_pla, &rules, 0);

  let options = PrintTreeOptions::new().max_depth(1);

  set_position(&mut search, next_pla, &board, &hist);
  run_whole_search(ctx, &mut search, next_pla);
  print_tree(&search, &options, P_WHITE);
  let perspective = P_WHITE;
  let analysis_pv_len = 2;
  let prevent_encore = true;
  let include_policy = false;
  let include_ownership = true;
  let include_ownership_stdev = true;
  let include_moves_ownership = true;
  let include_moves_ownership_stdev = true;
  let include_pv_visits = false;
  let include_no_result_value = false;
  let (suc, json) = analysis_json(
    &search,
    perspective,
    analysis_pv_len,
    prevent_encore,
    include_policy,
    include_ownership,
    include_ownership_stdev,
    include_moves_ownership,
    include_moves_ownership_stdev,
    include_pv_visits,
    include_no_result_value,
  );
  testAssert!(suc);
  cout_json(&json);

  drop(search);
  delete_nn_eval(ctx, nn_eval);
}

fn analysis_json_with_moves_ownership_and_stdev(ctx: &mut Ctx) {
  analysis_json_moves_ownership(ctx, false, 16);
}

fn analysis_json_with_moves_ownership_and_stdev_and_symmetry(ctx: &mut Ctx) {
  analysis_json_moves_ownership(ctx, true, 17);
}

fn analysis_json_2(ctx: &mut Ctx) {
  let nn_eval = start_nn_eval(ctx, MODEL_FILE, "", 9, 9, 0, true, false, false, true, false);
  let mut params = SearchParams::new();
  params.max_visits = 10;
  params.subtree_value_bias_factor = 0.5;
  params.chosen_move_temperature = 0.0;
  let mut search = new_search(&params, &nn_eval, "autoSearchRandSeed");
  search.set_always_include_owner_map(false);
  let rules = Rules::get_tromp_taylorish();
  let board = parse_board_input(18, 9, 6);
  let next_pla = P_BLACK;
  let hist = BoardHistory::new_full(&board, next_pla, &rules, 0);

  let options = PrintTreeOptions::new().max_depth(1);

  set_position(&mut search, next_pla, &board, &hist);
  run_whole_search(ctx, &mut search, next_pla);
  print_tree(&search, &options, P_WHITE);
  let perspective = P_WHITE;
  let analysis_pv_len = 2;
  let prevent_encore = true;
  let include_policy = true;
  let include_ownership = false;
  let include_ownership_stdev = false;
  let include_moves_ownership = false;
  let include_moves_ownership_stdev = false;
  let include_pv_visits = false;
  let include_no_result_value = false;
  let (suc, json) = analysis_json(
    &search,
    perspective,
    analysis_pv_len,
    prevent_encore,
    include_policy,
    include_ownership,
    include_ownership_stdev,
    include_moves_ownership,
    include_moves_ownership_stdev,
    include_pv_visits,
    include_no_result_value,
  );
  testAssert!(suc);
  cout_json(&json);

  drop(search);
  delete_nn_eval(ctx, nn_eval);
}

fn search_results_at_0_1_2_visits_and_terminal_position(ctx: &mut Ctx) {
  let print_results = |search: &Search, allow_direct_policy_moves: bool| {
    let mut values = ReportedSearchValues::new();
    coutln!("getRootVisits {}", search.get_root_visits());
    let mut suc = search.get_root_values(&mut values);
    coutln!("getRootValues success: {}", suc as i32);
    if suc {
      coutln!("{} {} {}", values.visits, cpp_ostream_f64(values.weight), cpp_ostream_f64(values.win_loss_value));
    }

    suc = search.get_pruned_root_values(&mut values);
    coutln!("getPrunedRootValues success: {}", suc as i32);
    if suc {
      coutln!("{} {} {}", values.visits, cpp_ostream_f64(values.weight), cpp_ostream_f64(values.win_loss_value));
    }

    let node: Option<Rc<SearchNode>> = search.get_root_node();
    let node: Option<Rc<SearchNode>> = if node.is_some() { search.get_child_for_move(node.as_deref(), Board::PASS_LOC) } else { None };
    if node.is_some() {
      suc = search.get_node_values(node.as_deref(), &mut values);
      coutln!("getNodeValues for pass child success: {}", suc as i32);
      if suc {
        coutln!("{} {} {}", values.visits, cpp_ostream_f64(values.weight), cpp_ostream_f64(values.win_loss_value));
      }

      suc = search.get_pruned_node_values(node.as_deref(), &mut values);
      coutln!("getPrunedNodeValues for pass child success: {}", suc as i32);
      if suc {
        coutln!("{} {} {}", values.visits, cpp_ostream_f64(values.weight), cpp_ostream_f64(values.win_loss_value));
      }
    }

    let mut play_selection_values: Vec<f64> = Vec::new();
    let mut locs: Vec<Loc> = Vec::new(); // not used
    if allow_direct_policy_moves {
      suc = search.get_play_selection_values_ret_visit_counts(&mut locs, &mut play_selection_values, None, 1.0);
    } else {
      match search.get_root_node() {
        None => suc = false,
        Some(root) => {
          suc = search.get_play_selection_values_node(&root, &mut locs, &mut play_selection_values, None, 1.0, allow_direct_policy_moves);
        }
      }
    }
    coutln!("getPlaySelectionValues success: {}", suc as i32);
    if suc {
      for i in 0..play_selection_values.len() {
        coutln!("{} {}", Location::to_string(locs[i], search.get_root_board()), cpp_ostream_f64(play_selection_values[i]));
      }
    }
    let perspective = P_WHITE;
    let analysis_pv_len = 2;
    let prevent_encore = true;
    let include_policy = true;
    let include_ownership = false;
    let include_ownership_stdev = false;
    let include_moves_ownership = false;
    let include_moves_ownership_stdev = false;
    let include_pv_visits = true;
    let include_no_result_value = false;
    let (suc, json) = analysis_json(
      search,
      perspective,
      analysis_pv_len,
      prevent_encore,
      include_policy,
      include_ownership,
      include_ownership_stdev,
      include_moves_ownership,
      include_moves_ownership_stdev,
      include_pv_visits,
      include_no_result_value,
    );
    coutln!("getAnalysisJson success: {}", suc as i32);
    cout_json(&json);
  };

  let nn_eval = start_nn_eval(ctx, MODEL_FILE, "", 9, 9, 0, true, false, false, true, false);
  let rules = Rules::get_tromp_taylorish();
  let board = parse_board_input(19, 7, 7);
  let next_pla = P_BLACK;
  let hist = BoardHistory::new_full(&board, next_pla, &rules, 0);

  let _options = PrintTreeOptions::new().max_depth(1);

  {
    let mut params = SearchParams::new();
    params.max_visits = 1;
    let mut search = new_search(&params, &nn_eval, "autoSearchRandSeed");
    set_position(&mut search, next_pla, &board, &hist);
    begin_search(ctx, &mut search, false);

    coutln!("Testing 0 visits allowDirectPolicyMoves false...");
    let allow_direct_policy_moves = false;
    print_results(&search, allow_direct_policy_moves);
    drop(search);
  }
  {
    let mut params = SearchParams::new();
    params.max_visits = 1;
    let mut search = new_search(&params, &nn_eval, "autoSearchRandSeed");
    set_position(&mut search, next_pla, &board, &hist);
    begin_search(ctx, &mut search, false);

    coutln!("Testing 0 visits allowDirectPolicyMoves true...");
    let allow_direct_policy_moves = true;
    print_results(&search, allow_direct_policy_moves);
    drop(search);
  }
  {
    let mut params = SearchParams::new();
    params.max_visits = 1;
    let mut search = new_search(&params, &nn_eval, "autoSearchRandSeed");
    set_position(&mut search, next_pla, &board, &hist);
    run_whole_search(ctx, &mut search, next_pla);

    coutln!("Testing 1 visits allowDirectPolicyMoves false...");
    let allow_direct_policy_moves = false;
    print_results(&search, allow_direct_policy_moves);
    drop(search);
  }
  {
    let mut params = SearchParams::new();
    params.max_visits = 1;
    let mut search = new_search(&params, &nn_eval, "autoSearchRandSeed");
    set_position(&mut search, next_pla, &board, &hist);
    run_whole_search(ctx, &mut search, next_pla);

    coutln!("Testing 1 visits allowDirectPolicyMoves true...");
    let allow_direct_policy_moves = true;
    print_results(&search, allow_direct_policy_moves);
    drop(search);
  }
  {
    let mut params = SearchParams::new();
    params.max_visits = 2;
    let mut search = new_search(&params, &nn_eval, "autoSearchRandSeed");
    set_position(&mut search, next_pla, &board, &hist);
    run_whole_search(ctx, &mut search, next_pla);

    coutln!("Testing 2 visits allowDirectPolicyMoves false...");
    let allow_direct_policy_moves = false;
    print_results(&search, allow_direct_policy_moves);
    drop(search);
  }
  {
    let mut params = SearchParams::new();
    params.max_visits = 2;
    let mut search = new_search(&params, &nn_eval, "autoSearchRandSeed");
    set_position(&mut search, next_pla, &board, &hist);
    run_whole_search(ctx, &mut search, next_pla);

    coutln!("Testing 2 visits allowDirectPolicyMoves true...");
    let allow_direct_policy_moves = true;
    print_results(&search, allow_direct_policy_moves);
    drop(search);
  }
  {
    let mut params = SearchParams::new();
    params.max_visits = 2;
    let mut search = new_search(&params, &nn_eval, "autoSearchRandSeed");
    set_position(&mut search, next_pla, &board, &hist);
    make_move(&mut search, Board::PASS_LOC, P_BLACK);
    make_move(&mut search, Board::PASS_LOC, P_WHITE);
    run_whole_search(ctx, &mut search, next_pla);

    coutln!("Testing 2 visits terminal position allowDirectPolicyMoves false...");
    let allow_direct_policy_moves = false;
    print_results(&search, allow_direct_policy_moves);
    drop(search);
  }
  {
    let mut params = SearchParams::new();
    params.max_visits = 2;
    let mut search = new_search(&params, &nn_eval, "autoSearchRandSeed");
    set_position(&mut search, next_pla, &board, &hist);
    make_move(&mut search, Board::PASS_LOC, P_BLACK);
    make_move(&mut search, Board::PASS_LOC, P_WHITE);
    run_whole_search(ctx, &mut search, next_pla);

    coutln!("Testing 2 visits terminal position allowDirectPolicyMoves true...");
    let allow_direct_policy_moves = true;
    print_results(&search, allow_direct_policy_moves);
    drop(search);
  }
  {
    let mut params = SearchParams::new();
    params.max_visits = 1000;
    let mut search = new_search(&params, &nn_eval, "autoSearchRandSeed");
    set_position(&mut search, next_pla, &board, &hist);
    make_move(&mut search, Board::PASS_LOC, P_BLACK);
    search.set_root_hint_loc(Board::PASS_LOC);
    run_whole_search(ctx, &mut search, P_WHITE);

    coutln!("Testing 1000 visits just before terminal position allowDirectPolicyMoves false...");
    let allow_direct_policy_moves = false;
    print_results(&search, allow_direct_policy_moves);

    coutln!("Testing 1000 visits just before terminal position, then playing the pass and having tree reuse. allowDirectPolicyMoves false...");
    make_move(&mut search, Board::PASS_LOC, P_WHITE);
    print_results(&search, allow_direct_policy_moves);

    drop(search);
  }

  delete_nn_eval(ctx, nn_eval);
}

fn coherence_of_search_tree_recursive_walking(ctx: &mut Ctx) {
  let nn_eval = start_nn_eval(ctx, MODEL_FILE, "", NNPos::MAX_BOARD_LEN, NNPos::MAX_BOARD_LEN, 0, true, false, false, true, false);
  let mut params = SearchParams::new();
  params.max_visits = 1000;
  params.dynamic_score_utility_factor = 3.0;
  let mut search = new_search(&params, &nn_eval, "autoSearchRandSeed");
  let rules = Rules::get_tromp_taylorish();

  let board = parse_board_input(20, 7, 7);
  let mut next_pla = P_BLACK;
  let hist = BoardHistory::new_full(&board, next_pla, &rules, 0);

  {
    //--------------------------------------
    coutln!("First perform a basic search.");

    set_position(&mut search, next_pla, &board, &hist);
    run_whole_search(ctx, &mut search, next_pla);

    let loc_to_descend: Loc;
    {
      let root = root_node(&search);
      let children = root.get_children();
      let children_capacity = children.get_capacity();
      testAssert!(children_capacity > 1);

      //In theory nothing requires this, but it would be kind of crazy if this were false
      testAssert!(children.iterate_and_count_children() > 1);
      testAssert!(children.index(1).get_if_allocated().is_some());

      loc_to_descend = children.index(1).get_move_loc();
    }

    let options = PrintTreeOptions::new().max_depth(1);
    coutln!("{}", search.root_board);
    print_tree(&search, &options, P_WHITE);
    print_tree(&search, &options.only_branch(&board, &Location::to_string(loc_to_descend, &board)).unwrap(), P_WHITE);

    coutln!();

    //Enumerate the tree and make sure every node is indeed hit exactly once in postorder.
    verify_tree_post_order(&search, -1);

    //--------------------------------------
    coutln!("Next, make a move, and with no search, print the tree.");

    make_move(&mut search, loc_to_descend, next_pla);
    next_pla = get_opp(next_pla);
    let _ = next_pla;

    coutln!("{}", search.root_board);
    print_tree(&search, &options, P_WHITE);
    coutln!();

    //Enumerate the tree and make sure every node is indeed hit exactly once in postorder.
    verify_tree_post_order(&search, -1);

    //--------------------------------------
    coutln!("Begin search but make no additional playouts, print the tree.");
    let pondering = false;
    begin_search(ctx, &mut search, pondering);
    print_tree(&search, &options, P_WHITE);
    coutln!();

    //Enumerate the tree and make sure every node is indeed hit exactly once in postorder.
    verify_tree_post_order(&search, -1);
  }

  drop(search);
  delete_nn_eval(ctx, nn_eval);
  coutln!();
}

fn avoiding_of_all_or_almost_all_moves(ctx: &mut Ctx) {
  let mut params = SearchParams::new();
  params.max_visits = 100;
  let rules = Rules::get_tromp_taylorish();

  let board = parse_board_input(21, 9, 5);

  {
    let nn_eval = start_nn_eval(ctx, MODEL_FILE, "", 9, 5, 0, true, false, false, true, false);
    let mut search = new_search(&params, &nn_eval, "autoSearchRandSeed1234");
    coutln!("Avoid all but 2 moves for both players, including passing");
    let mut avoid_move_until_by_loc: Vec<i32> = vec![0; Board::MAX_ARR_SIZE as usize];
    for y in 0..board.y_size {
      for x in 0..board.x_size {
        let loc = Location::get_loc(x, y, board.x_size);
        if x == 0 || x == 1 {
          avoid_move_until_by_loc[loc as usize] = 0;
        } else {
          avoid_move_until_by_loc[loc as usize] = 3;
        }
      }
    }
    avoid_move_until_by_loc[Board::PASS_LOC as usize] = 3;

    let mut next_pla = P_WHITE;
    let hist = BoardHistory::new_full(&board, next_pla, &rules, 0);

    set_position(&mut search, next_pla, &board, &hist);
    search.set_avoid_move_until_by_loc(&avoid_move_until_by_loc, &avoid_move_until_by_loc);
    run_whole_search(ctx, &mut search, next_pla);

    coutln!("{}", search.root_board);
    let mut options = PrintTreeOptions::new().max_depth(4);
    print_tree(&search, &options, P_WHITE);

    let perspective = P_WHITE;
    let analysis_pv_len = 2;
    let prevent_encore = true;
    let (suc, json) = analysis_json(&search, perspective, analysis_pv_len, prevent_encore, false, false, false, false, false, false, false);
    testAssert!(suc);
    cout_json(&json);

    //--------------------------------------
    coutln!("Next, make a move, and with no search, print the tree.");

    let loc_to_descend = loc_of("B3", &search.root_board);
    make_move(&mut search, loc_to_descend, next_pla);
    next_pla = get_opp(next_pla);

    coutln!("{}", search.root_board);
    options = options.max_depth(1);
    print_tree(&search, &options, P_WHITE);
    coutln!();

    //--------------------------------------
    coutln!("Then continue the search to complete 100 visits.");

    run_whole_search(ctx, &mut search, next_pla);
    options = options.max_depth(1);
    print_tree(&search, &options, P_WHITE);
    coutln!();
    drop(search);
    delete_nn_eval(ctx, nn_eval);
  }

  {
    let nn_eval = start_nn_eval(ctx, MODEL_FILE, "", 9, 5, 0, true, false, false, true, false);
    let mut search = new_search(&params, &nn_eval, "autoSearchRandSeed1235");
    coutln!("Avoid all moves for both players, including passing");
    let mut avoid_move_until_by_loc: Vec<i32> = vec![0; Board::MAX_ARR_SIZE as usize];
    for y in 0..board.y_size {
      for x in 0..board.x_size {
        let loc = Location::get_loc(x, y, board.x_size);
        avoid_move_until_by_loc[loc as usize] = 3;
      }
    }
    avoid_move_until_by_loc[Board::PASS_LOC as usize] = 3;

    let next_pla = P_WHITE;
    let hist = BoardHistory::new_full(&board, next_pla, &rules, 0);

    set_position(&mut search, next_pla, &board, &hist);
    search.set_avoid_move_until_by_loc(&avoid_move_until_by_loc, &avoid_move_until_by_loc);
    run_whole_search(ctx, &mut search, next_pla);

    coutln!("{}", search.root_board);
    let options = PrintTreeOptions::new().max_depth(4);
    print_tree(&search, &options, P_WHITE);

    let perspective = P_WHITE;
    let analysis_pv_len = 2;
    let prevent_encore = true;
    let (suc, json) = analysis_json(&search, perspective, analysis_pv_len, prevent_encore, false, false, false, false, false, false, false);
    testAssert!(suc);
    cout_json(&json);
    drop(search);
    delete_nn_eval(ctx, nn_eval);
  }

  {
    let nn_eval = start_nn_eval(ctx, MODEL_FILE, "", 9, 5, 0, true, false, false, true, false);
    let mut search = new_search(&params, &nn_eval, "autoSearchRandSeed1236");
    coutln!("Avoid all moves for black, including passing");
    let mut avoid_move_until_by_loc: Vec<i32> = vec![0; Board::MAX_ARR_SIZE as usize];
    for y in 0..board.y_size {
      for x in 0..board.x_size {
        let loc = Location::get_loc(x, y, board.x_size);
        avoid_move_until_by_loc[loc as usize] = 10;
      }
    }
    avoid_move_until_by_loc[Board::PASS_LOC as usize] = 10;

    let next_pla = P_WHITE;
    let hist = BoardHistory::new_full(&board, next_pla, &rules, 0);

    set_position(&mut search, next_pla, &board, &hist);
    search.set_avoid_move_until_by_loc(&avoid_move_until_by_loc, &[]);
    run_whole_search(ctx, &mut search, next_pla);

    coutln!("{}", search.root_board);
    let options = PrintTreeOptions::new().max_depth(4);
    print_tree(&search, &options, P_WHITE);

    let perspective = P_WHITE;
    let analysis_pv_len = 2;
    let prevent_encore = true;
    let (suc, json) = analysis_json(&search, perspective, analysis_pv_len, prevent_encore, false, false, false, false, false, false, false);
    testAssert!(suc);
    cout_json(&json);
    drop(search);
    delete_nn_eval(ctx, nn_eval);
  }
  coutln!();
}

/// "Graph search, ..." sections: they differ in the diagram and the rules only.
fn graph_search_section(ctx: &mut Ctx, input_idx: usize, rules: Rules, next_pla: Player) {
  let nn_eval = start_nn_eval(ctx, MODEL_FILE, "", 7, 7, 0, true, false, false, true, false);
  let mut params = SearchParams::new();
  params.max_visits = 1000;
  params.subtree_value_bias_factor = 0.5;
  params.use_graph_search = true;
  params.chosen_move_temperature = 0.0;
  let mut search = new_search(&params, &nn_eval, "autoSearchRandSeed");
  let board = parse_board_input(input_idx, 7, 7);
  let hist = BoardHistory::new_full(&board, next_pla, &rules, 0);

  let options = PrintTreeOptions::new().max_depth(1);

  set_position(&mut search, next_pla, &board, &hist);

  //The C++ repeats "search / print, make move / print, search again / print (+ verify after the
  //second search), making move, searching again, making move" and a final verifyTreePostOrder.
  search_make_move_search_again(ctx, &mut search, next_pla, &options, true);
  verify_tree_post_order(&search, -1);

  drop(search);
  delete_nn_eval(ctx, nn_eval);
  coutln!();
}

fn graph_search_opening(ctx: &mut Ctx) {
  graph_search_section(ctx, 22, Rules::get_tromp_taylorish(), P_BLACK);
}

fn graph_search_7x7_big_fight(ctx: &mut Ctx) {
  graph_search_section(ctx, 23, Rules::get_tromp_taylorish(), P_WHITE);
}

fn graph_search_7x7_endgame_kos(ctx: &mut Ctx) {
  graph_search_section(ctx, 24, Rules::parse_rules("japanese").unwrap(), P_BLACK);
}

/// "FPU parent weight by visited policy ..." sections.
fn fpu_section(ctx: &mut Ctx, input_idx: usize, by_visited_policy: bool, pow: f64) {
  let nn_eval = start_nn_eval(ctx, MODEL_FILE, "seeed", 7, 7, 0, true, false, false, true, false);
  let mut params = SearchParams::for_tests_v1();
  params.max_visits = 1000;
  params.fpu_parent_weight_by_visited_policy = by_visited_policy;
  params.fpu_parent_weight_by_visited_policy_pow = pow;

  let mut search = new_search(&params, &nn_eval, "autoSearchRandSeed");
  let rules = Rules::parse_rules("japanese").unwrap();
  let board = parse_board_input(input_idx, 7, 7);
  let next_pla = P_BLACK;
  let hist = BoardHistory::new_full(&board, next_pla, &rules, 0);

  let options = PrintTreeOptions::new().max_depth(1);

  set_position(&mut search, next_pla, &board, &hist);

  run_whole_search(ctx, &mut search, next_pla);
  coutln!("{}", search.root_board);
  print_tree(&search, &options, P_WHITE);

  drop(search);
  delete_nn_eval(ctx, nn_eval);
  coutln!();
}

fn fpu_false(ctx: &mut Ctx) {
  fpu_section(ctx, 25, false, 1.0);
}

fn fpu_1_0(ctx: &mut Ctx) {
  fpu_section(ctx, 26, true, 1.0);
}

fn fpu_2_5(ctx: &mut Ctx) {
  fpu_section(ctx, 27, true, 2.5);
}

fn fpu_0_5(ctx: &mut Ctx) {
  fpu_section(ctx, 28, true, 0.5);
}

fn policy_optimism_with_tree_reuse_section(ctx: &mut Ctx, input_idx: usize, root_policy_optimism: f64, policy_optimism: f64) {
  let nn_eval = start_nn_eval(ctx, MODEL_FILE, "policyoptimismtreereuse", 7, 7, 0, true, false, false, true, false);
  let mut params = SearchParams::for_tests_v2();
  params.max_visits = 100;
  params.root_policy_optimism = root_policy_optimism;
  params.policy_optimism = policy_optimism;
  let mut params_low_visits = params.clone();
  params_low_visits.max_visits = 8;

  let mut search = new_search(&params, &nn_eval, "autoSearchRandSeeeeeed");
  let rules = Rules::parse_rules("japanese").unwrap();
  let board = parse_board_input(input_idx, 5, 5);
  let mut next_pla = P_BLACK;
  let hist = BoardHistory::new_full(&board, next_pla, &rules, 0);
  let options = PrintTreeOptions::new().max_depth(1);

  set_position(&mut search, next_pla, &board, &hist);

  let debug_print_root = |search: &Search| {
    root_node(search).get_nn_output().unwrap().debug_print(&mut Cout, &search.root_board);
  };
  let policy_optimism_used = |search: &Search| -> f64 { root_node(search).get_nn_output().unwrap().policy_optimism_used as f64 };

  coutln!("Root position");
  run_whole_search(ctx, &mut search, next_pla);
  coutln!("{}", search.root_board);
  print_tree(&search, &options, P_WHITE);
  debug_print_root(&search);
  testAssert!((policy_optimism_used(&search) - root_policy_optimism).abs() < 0.00001);

  coutln!("Make move and print");
  let loc = search.get_chosen_move_loc();
  make_move(&mut search, loc, next_pla);
  next_pla = get_opp(next_pla);
  print_tree(&search, &options, P_WHITE);
  debug_print_root(&search);
  testAssert!((policy_optimism_used(&search) - policy_optimism).abs() < 0.00001);

  coutln!("Do search again");
  run_whole_search(ctx, &mut search, next_pla);
  coutln!("{}", search.root_board);
  print_tree(&search, &options, P_WHITE);
  debug_print_root(&search);
  testAssert!((policy_optimism_used(&search) - root_policy_optimism).abs() < 0.00001);

  coutln!("Make move and print");
  let loc = search.get_chosen_move_loc();
  make_move(&mut search, loc, next_pla);
  next_pla = get_opp(next_pla);
  print_tree(&search, &options, P_WHITE);
  debug_print_root(&search);
  testAssert!((policy_optimism_used(&search) - policy_optimism).abs() < 0.00001);

  coutln!("Do search again but with very low visits so the search already meets max visits");
  search.set_params_no_clearing(&params_low_visits);
  run_whole_search(ctx, &mut search, next_pla);
  coutln!("{}", search.root_board);
  print_tree(&search, &options, P_WHITE);
  debug_print_root(&search);
  testAssert!((policy_optimism_used(&search) - root_policy_optimism).abs() < 0.00001);

  drop(search);
  delete_nn_eval(ctx, nn_eval);
  coutln!();
}

fn policy_optimism_with_tree_reuse(ctx: &mut Ctx) {
  policy_optimism_with_tree_reuse_section(ctx, 29, 0.43, 0.71);
}

fn policy_optimism_with_tree_reuse_0_root(ctx: &mut Ctx) {
  //The 0.0 / 1.0 expectations of the C++ are the same asserts with rootPolicyOptimism = 0.0, policyOptimism = 1.0.
  policy_optimism_with_tree_reuse_section(ctx, 30, 0.0, 1.0);
}

fn zero_node_search(ctx: &mut Ctx) {
  let nn_eval = start_nn_eval(ctx, MODEL_FILE, "zeronodesearch", 13, 13, 0, true, false, false, true, false);
  let params = SearchParams::for_tests_v2();
  let mut search = new_search(&params, &nn_eval, "autoSearchRandSeeeeeed");
  let rules = Rules::parse_rules("japanese").unwrap();
  let board = parse_board_input(31, 13, 6);
  let next_pla = P_BLACK;
  let hist = BoardHistory::new_full(&board, next_pla, &rules, 0);
  let options = PrintTreeOptions::new().max_depth(1);

  set_position(&mut search, next_pla, &board, &hist);

  let should_stop_early: &dyn Fn() -> bool = &|| true;
  run_or_stall_panic(&mut ctx.pool, "runWholeSearch", search.run_whole_search_should_stop_early(next_pla, Some(should_stop_early))).unwrap();
  coutln!("{}", search.root_board);
  print_tree(&search, &options, P_WHITE);
  testAssert!(root_node(&search).get_nn_output().is_none());
  coutln!("Chosen move: {}", Location::to_string(search.get_chosen_move_loc(), &board));

  drop(search);
  delete_nn_eval(ctx, nn_eval);
  coutln!();
}

fn chosen_move_probs(_ctx: &mut Ctx) {
  let mut rand = Rand::new();

  let mut test_relative_probs = |relative_probs: &[f64], temperature: f64, only_below_prob: f64| {
    let mut processed_rel_probs_buf: Vec<f64> = vec![0.0; Board::MAX_ARR_SIZE as usize];
    Search::choose_index_with_temperature(
      &mut rand,
      relative_probs,
      relative_probs.len() as i32,
      temperature,
      only_below_prob,
      Some(&mut processed_rel_probs_buf[..]),
    );
    coutln!("Temperature {} onlyBelowProb {}", cpp_ostream_f64(temperature), cpp_ostream_f64(only_below_prob));
    for i in 0..relative_probs.len() {
      coutln!("{}", cpp_ostream_f64(processed_rel_probs_buf[i]));
    }
  };

  {
    let relative_probs = [40.0, 20.0, 160.0, 80.0, 10.0, 10.0];
    let temperature = 1.0;
    let only_below_prob = 1.0;
    test_relative_probs(&relative_probs, temperature, only_below_prob);
  }
  {
    let relative_probs = [40.0, 20.0, 160.0, 80.0, 10.0, 10.0];
    let temperature = 1.0;
    let only_below_prob = 0.2;
    test_relative_probs(&relative_probs, temperature, only_below_prob);
  }
  {
    let relative_probs = [40.0, 20.0, 160.0, 80.0, 10.0, 10.0];
    let temperature = 0.5;
    let only_below_prob = 1.0;
    test_relative_probs(&relative_probs, temperature, only_below_prob);
  }
  {
    let relative_probs = [40.0, 20.0, 160.0, 80.0, 10.0, 10.0];
    let temperature = 0.5;
    let only_below_prob = 0.2;
    test_relative_probs(&relative_probs, temperature, only_below_prob);
  }
  {
    let relative_probs = [40.0, 20.0, 160.0, 80.0, 10.0, 10.0];
    let temperature = 2.0;
    let only_below_prob = 0.2;
    test_relative_probs(&relative_probs, temperature, only_below_prob);
  }
  {
    let relative_probs = [40.0, 20.0, 160.0, 80.0, 10.0, 10.0];
    let temperature = 100000.0;
    let only_below_prob = 0.2;
    test_relative_probs(&relative_probs, temperature, only_below_prob);
  }
  {
    let relative_probs = [40.0, 20.0, 160.0, 80.0, 10.0, 10.0];
    let temperature = 0.00001;
    let only_below_prob = 0.2;
    test_relative_probs(&relative_probs, temperature, only_below_prob);
  }
}

fn uninitialized_search_params(_ctx: &mut Ctx) {
  let params = SearchParams::new();
  params.print_params(&mut Cout);
  coutln!();
}

fn search_params_for_tests_v1(_ctx: &mut Ctx) {
  let params = SearchParams::for_tests_v1();
  params.print_params(&mut Cout);
  coutln!();
}

/// "SearchParams for <setup>" sections: `ConfigParser cfg; cfg.overrideKey("numSearchThreads","1");` once
/// before the first of them in the C++; same config here for each.
fn search_params_for(setup_for: setup_for_t) {
  let mut cfg = ConfigParser::new(false, true);
  cfg.override_key("numSearchThreads", "1");
  let params = Setup::load_single_params(&mut cfg, setup_for).expect("upstream: loadSingleParams succeeds for this config");
  params.print_params(&mut Cout);
  coutln!();
}

fn search_params_for_gtp(_ctx: &mut Ctx) {
  search_params_for(Setup::SETUP_FOR_GTP);
}

fn search_params_for_analysis(_ctx: &mut Ctx) {
  search_params_for(Setup::SETUP_FOR_ANALYSIS);
}

fn search_params_for_match(_ctx: &mut Ctx) {
  search_params_for(Setup::SETUP_FOR_MATCH);
}

fn search_params_for_benchmark(_ctx: &mut Ctx) {
  search_params_for(Setup::SETUP_FOR_BENCHMARK);
}

fn search_params_for_other(_ctx: &mut Ctx) {
  search_params_for(Setup::SETUP_FOR_OTHER);
}

fn search_params_for_distributed(_ctx: &mut Ctx) {
  search_params_for(Setup::SETUP_FOR_DISTRIBUTED);
}


//------------------------------------------------------------------------------------------------
// Board size distribution: test-local copy of the GameInitializer pieces it uses
//
// PORTING.md §13 "test-local copies": program/play.cpp (GameInitializer) and program/playsettings.cpp are
// [[ignored]] (selfplay / match machinery the analysis engine never runs), and a full port would need
// dataio/sgf (Sgf::PositionSample: start/hint positions, M8) and file IO. This copy is faithful for the
// path the test runs: GameInitializer(cfg, logger, randSeed) without startPosesFromSgfDir / hintPosesDir,
// createGame(..., initialPosition=NULL, PlaySettings(), ..., startPosSample=NULL), plus the PlayUtils komi
// helpers it calls (playutils.cpp; they take play.h's ExtraBlackAndKomi). Omitted branches panic.

/// upstream: ExtraBlackAndKomi (play.h) — test-local copy
#[allow(dead_code)] // fields kept as upstream; the test reads only some
#[derive(Clone, Copy, Debug)]
struct ExtraBlackAndKomi {
  extra_black: i32,
  komi_mean: f32,
  komi_stdev: f32,
  make_game_fair: bool,
  make_game_fair_for_empty_board: bool,
  allow_integer: bool,
  interp_zero: bool,
}

impl Default for ExtraBlackAndKomi {
  fn default() -> Self {
    ExtraBlackAndKomi {
      extra_black: 0,
      komi_mean: 7.5f32,
      komi_stdev: 7.5f32,
      make_game_fair: false,
      make_game_fair_for_empty_board: false,
      allow_integer: true,
      interp_zero: false,
    }
  }
}

/// upstream: OtherGameProperties (play.h) — test-local copy
#[allow(dead_code)] // fields kept as upstream; the test reads only some
#[derive(Clone, Copy, Debug)]
struct OtherGameProperties {
  is_sgf_pos: bool,
  is_hint_pos: bool,
  allow_policy_init: bool,
  is_fork: bool,
  is_hint_fork: bool,
  hint_turn: i32,
  hint_loc: Loc,
  training_weight: f64,
  playout_doubling_advantage: f64,
  playout_doubling_advantage_pla: Player,
}

impl Default for OtherGameProperties {
  fn default() -> Self {
    OtherGameProperties {
      is_sgf_pos: false,
      is_hint_pos: false,
      allow_policy_init: true,
      is_fork: false,
      is_hint_fork: false,
      hint_turn: -1,
      hint_loc: Board::NULL_LOC,
      training_weight: 1.0,
      playout_doubling_advantage: 0.0,
      playout_doubling_advantage_pla: C_EMPTY,
    }
  }
}

/// upstream: PlaySettings (playsettings.h) — test-local copy of the fields createGame reads, with
/// PlaySettings::PlaySettings()'s values.
struct PlaySettings {
  handicap_asymmetric_playout_prob: f64,
  normal_asymmetric_playout_prob: f64,
  max_asymmetric_ratio: f64,
  // PORT NOTE: left uninitialized by upstream's default constructor; read only when an asymmetric
  // playout probability is positive, which the defaults never are.
  min_asymmetric_compensate_komi_prob: f64,
}

impl PlaySettings {
  /// upstream: PlaySettings::PlaySettings (playsettings.cpp) — test-local copy
  fn new() -> Self {
    PlaySettings { handicap_asymmetric_playout_prob: 0.0, normal_asymmetric_playout_prob: 0.0, max_asymmetric_ratio: 2.0, min_asymmetric_compensate_komi_prob: 0.0 }
  }
}

/// upstream: getDefaultMaxExtraBlack (playutils.cpp) static — test-local copy
fn get_default_max_extra_black(sqrt_board_area: f64) -> i32 {
  if sqrt_board_area <= 10.00001 {
    return 0;
  }
  if sqrt_board_area <= 14.00001 {
    return 1;
  }
  if sqrt_board_area <= 16.00001 {
    return 2;
  }
  if sqrt_board_area <= 17.00001 {
    return 3;
  }
  if sqrt_board_area <= 18.00001 {
    return 4;
  }
  5
}

/// upstream: PlayUtils::chooseExtraBlackAndKomi (playutils.cpp) — test-local copy
fn choose_extra_black_and_komi(
  base: f32,
  stdev: f32,
  allow_integer_prob: f64,
  handicap_prob: f64,
  num_extra_black_fixed: i32,
  big_stdev_prob: f64,
  big_stdev: f32,
  bigger_stdev_prob: f64,
  bigger_stdev: f32,
  sqrt_board_area: f64,
  rand: &mut Rand,
) -> ExtraBlackAndKomi {
  let mut extra_black: i32 = 0;
  let komi: f32 = base;

  let mut stdev_to_use: f32 = 0.0f32;
  if stdev > 0.0f32 {
    stdev_to_use = stdev;
  }
  if big_stdev > 0.0f32 && rand.next_bool(big_stdev_prob) {
    stdev_to_use = big_stdev;
  }
  if bigger_stdev > 0.0f32 && bigger_stdev_prob > 0.0 && rand.next_bool(bigger_stdev_prob) {
    stdev_to_use = bigger_stdev;
  }
  //Adjust for board size, so that we don't give the same massive komis on smaller boards
  stdev_to_use = stdev_to_use * (sqrt_board_area / 19.0) as f32;

  //Add handicap stones
  let default_max_extra_black: i32 = get_default_max_extra_black(sqrt_board_area);
  if (num_extra_black_fixed > 0 || default_max_extra_black > 0) && rand.next_bool(handicap_prob) {
    if num_extra_black_fixed > 0 {
      extra_black = num_extra_black_fixed;
    } else {
      extra_black += 1 + rand.next_uint_n(default_max_extra_black as u32) as i32;
    }
  }

  let allow_integer: bool = rand.next_bool(allow_integer_prob);

  let mut ret: ExtraBlackAndKomi = ExtraBlackAndKomi::default();
  ret.extra_black = extra_black;
  ret.komi_mean = komi;
  ret.komi_stdev = stdev_to_use;
  //These are set later
  ret.make_game_fair = false;
  ret.make_game_fair_for_empty_board = false;
  ret.interp_zero = false;
  //This is recorded for application later, since other things may adjust the komi in between.
  ret.allow_integer = allow_integer;
  ret
}

/// upstream: roundKomiWithLinearProb (playutils.cpp) static — test-local copy
fn round_komi_with_linear_prob(mut komi: f32, rand: &mut Rand) -> f32 {
  //Discretize komi
  let lower: f32 = (komi * 2.0f32).floor() / 2.0f32;
  let upper: f32 = (komi * 2.0f32).ceil() / 2.0f32;

  if lower == upper {
    komi = lower;
  } else {
    assert!(upper > lower);
    if rand.next_double() < ((komi - lower) / (upper - lower)) as f64 {
      komi = upper;
    } else {
      komi = lower;
    }
  }
  komi
}

/// upstream: PlayUtils::roundAndClipKomi (playutils.cpp) — test-local copy
fn round_and_clip_komi(mut unrounded: f64, board: &Board) -> f32 {
  //Just in case, make sure komi is reasonable
  let range: f32 = NNPos::KOMI_CLIP_RADIUS + (board.x_size * board.y_size) as f32;
  if unrounded < -range as f64 {
    unrounded = -range as f64;
  }
  if unrounded > range as f64 {
    unrounded = range as f64;
  }
  (0.5 * (2.0 * unrounded).round()) as f32
}

/// upstream: PlayUtils::setKomiWithNoise (playutils.cpp) — test-local copy
fn set_komi_with_noise(extra_black_and_komi: &ExtraBlackAndKomi, hist: &mut BoardHistory, rand: &mut Rand) {
  let mut komi: f32 = extra_black_and_komi.komi_mean;
  if extra_black_and_komi.komi_stdev > 0.0 {
    komi += extra_black_and_komi.komi_stdev * rand.next_gaussian_truncated(3.0) as f32;
  }
  if extra_black_and_komi.interp_zero {
    komi = komi * rand.next_double() as f32;
  }
  komi = round_komi_with_linear_prob(komi, rand);
  komi = round_and_clip_komi(komi as f64, hist.get_recent_board(0));
  testAssert!(Rules::komi_is_int_or_half_int(komi));
  // PORT NOTE: `(int)komi` is in range: komi was clipped above.
  if !extra_black_and_komi.allow_integer && komi == (komi as i32) as f32 {
    komi += if rand.next_bool(0.5) { -0.5f32 } else { 0.5f32 };
  }
  hist.set_komi(komi);
}

/// upstream: GameInitializer (play.h) — test-local copy (the fields the copied functions use)
#[allow(dead_code)] // fields kept as upstream; the test reads only some
struct GameInitializer {
  rand: Rand,
  allowed_ko_rule_strs: Vec<String>,
  allowed_scoring_rule_strs: Vec<String>,
  allowed_tax_rule_strs: Vec<String>,
  allowed_multi_stone_suicide_legals: Vec<bool>,
  allowed_buttons: Vec<bool>,
  allowed_ko_rules: Vec<i32>,
  allowed_scoring_rules: Vec<i32>,
  allowed_tax_rules: Vec<i32>,
  allowed_b_sizes: Vec<(i32, i32)>,
  allowed_b_size_rel_probs: Vec<f64>,
  komi_mean: f32,
  komi_stdev: f32,
  komi_allow_integer_prob: f64,
  handicap_prob: f64,
  handicap_compensate_komi_prob: f64,
  fork_compensate_komi_prob: f64,
  sgf_compensate_komi_prob: f64,
  komi_big_stdev_prob: f64,
  komi_big_stdev: f32,
  komi_bigger_stdev_prob: f64,
  komi_bigger_stdev: f32,
  handicap_komi_interp_zero_prob: f64,
  sgf_komi_interp_zero_prob: f64,
  komi_auto: bool,
  num_extra_black_fixed: i32,
  no_result_stdev: f64,
  draw_rand_radius: f64,
  start_poses_prob: f64,
  hint_poses_prob: f64,
  min_board_x_size: i32,
  min_board_y_size: i32,
  max_board_x_size: i32,
  max_board_y_size: i32,
}

impl GameInitializer {
  /// upstream: GameInitializer::GameInitializer(ConfigParser&, Logger&, const string& randSeed) (play.cpp) — test-local copy
  fn new_rand_seed(cfg: &mut ConfigParser, rand_seed: &str) -> Result<GameInitializer, KataError> {
    let mut g = GameInitializer {
      rand: Rand::new_seed(rand_seed),
      allowed_ko_rule_strs: Vec::new(),
      allowed_scoring_rule_strs: Vec::new(),
      allowed_tax_rule_strs: Vec::new(),
      allowed_multi_stone_suicide_legals: Vec::new(),
      allowed_buttons: Vec::new(),
      allowed_ko_rules: Vec::new(),
      allowed_scoring_rules: Vec::new(),
      allowed_tax_rules: Vec::new(),
      allowed_b_sizes: Vec::new(),
      allowed_b_size_rel_probs: Vec::new(),
      komi_mean: 0.0,
      komi_stdev: 0.0,
      komi_allow_integer_prob: 0.0,
      handicap_prob: 0.0,
      handicap_compensate_komi_prob: 0.0,
      fork_compensate_komi_prob: 0.0,
      sgf_compensate_komi_prob: 0.0,
      komi_big_stdev_prob: 0.0,
      komi_big_stdev: 0.0,
      komi_bigger_stdev_prob: 0.0,
      komi_bigger_stdev: 0.0,
      handicap_komi_interp_zero_prob: 0.0,
      sgf_komi_interp_zero_prob: 0.0,
      komi_auto: false,
      num_extra_black_fixed: 0,
      no_result_stdev: 0.0,
      draw_rand_radius: 0.0,
      start_poses_prob: 0.0,
      hint_poses_prob: 0.0,
      min_board_x_size: 0,
      min_board_y_size: 0,
      max_board_x_size: 0,
      max_board_y_size: 0,
    };
    g.init_shared(cfg)?;
    Ok(g)
  }

  /// upstream: GameInitializer::initShared (play.cpp) — test-local copy (Logger& dropped)
  fn init_shared(&mut self, cfg: &mut ConfigParser) -> Result<(), KataError> {
    self.allowed_ko_rule_strs = cfg.get_strings_possibles("koRules", &Rules::ko_rule_strings())?;
    self.allowed_scoring_rule_strs = cfg.get_strings_possibles("scoringRules", &Rules::scoring_rule_strings())?;
    self.allowed_tax_rule_strs = cfg.get_strings_possibles("taxRules", &Rules::tax_rule_strings())?;
    self.allowed_multi_stone_suicide_legals = cfg.get_bools("multiStoneSuicideLegals")?;
    self.allowed_buttons = cfg.get_bools("hasButtons")?;

    for i in 0..self.allowed_ko_rule_strs.len() {
      self.allowed_ko_rules.push(Rules::parse_ko_rule(&self.allowed_ko_rule_strs[i])?);
    }
    for i in 0..self.allowed_scoring_rule_strs.len() {
      self.allowed_scoring_rules.push(Rules::parse_scoring_rule(&self.allowed_scoring_rule_strs[i])?);
    }
    for i in 0..self.allowed_tax_rule_strs.len() {
      self.allowed_tax_rules.push(Rules::parse_tax_rule(&self.allowed_tax_rule_strs[i])?);
    }

    if self.allowed_ko_rules.is_empty() {
      return Err(KataError::IOError("koRules must have at least one value in ".to_string() + &cfg.get_file_name()));
    }
    if self.allowed_scoring_rules.is_empty() {
      return Err(KataError::IOError("scoringRules must have at least one value in ".to_string() + &cfg.get_file_name()));
    }
    if self.allowed_tax_rules.is_empty() {
      return Err(KataError::IOError("taxRules must have at least one value in ".to_string() + &cfg.get_file_name()));
    }
    if self.allowed_multi_stone_suicide_legals.is_empty() {
      return Err(KataError::IOError("multiStoneSuicideLegals must have at least one value in ".to_string() + &cfg.get_file_name()));
    }
    if self.allowed_buttons.is_empty() {
      return Err(KataError::IOError("hasButtons must have at least one value in ".to_string() + &cfg.get_file_name()));
    }

    {
      let mut has_area_scoring: bool = false;
      for i in 0..self.allowed_scoring_rules.len() {
        if self.allowed_scoring_rules[i] == Rules::SCORING_AREA {
          has_area_scoring = true;
        }
      }
      let mut has_true_button: bool = false;
      for i in 0..self.allowed_buttons.len() {
        if self.allowed_buttons[i] {
          has_true_button = true;
        }
      }
      if !has_area_scoring && has_true_button {
        return Err(KataError::IOError("If scoringRules does not include AREA, hasButtons must be false in ".to_string() + &cfg.get_file_name()));
      }
    }

    if cfg.contains("bSizes") == cfg.contains("bSizesXY") {
      return Err(KataError::IOError("Must specify exactly one of bSizes or bSizesXY".to_string()));
    }

    if cfg.contains("bSizes") {
      let allowed_b_edges: Vec<i32> = cfg.get_ints("bSizes", 2, Board::MAX_LEN)?;
      let allowed_b_edge_rel_probs: Vec<f64> = cfg.get_doubles("bSizeRelProbs", 0.0, 1e100)?;
      let mut rel_prob_sum: f64 = 0.0;
      for p in allowed_b_edge_rel_probs.iter() {
        rel_prob_sum += *p;
      }
      if rel_prob_sum <= 1e-100 {
        return Err(KataError::IOError("bSizeRelProbs must sum to a positive value".to_string()));
      }
      let allow_rectangle_prob: f64 = if cfg.contains("allowRectangleProb") { cfg.get_double("allowRectangleProb", 0.0, 1.0)? } else { 0.0 };

      if allowed_b_edges.is_empty() {
        return Err(KataError::IOError("bSizes must have at least one value in ".to_string() + &cfg.get_file_name()));
      }
      if allowed_b_edges.len() != allowed_b_edge_rel_probs.len() {
        return Err(KataError::IOError("bSizes and bSizeRelProbs must have same number of values in ".to_string() + &cfg.get_file_name()));
      }

      self.allowed_b_sizes.clear();
      self.allowed_b_size_rel_probs.clear();
      for i in 0..allowed_b_edges.len() {
        for j in 0..allowed_b_edges.len() {
          let x: i32 = allowed_b_edges[i];
          let y: i32 = allowed_b_edges[j];
          if x == y {
            self.allowed_b_sizes.push((x, y));
            self.allowed_b_size_rel_probs.push(
              (1.0 - allow_rectangle_prob) * allowed_b_edge_rel_probs[i] / rel_prob_sum
                + allow_rectangle_prob * allowed_b_edge_rel_probs[i] * allowed_b_edge_rel_probs[j] / rel_prob_sum / rel_prob_sum,
            );
          } else {
            if allow_rectangle_prob > 0.0 {
              self.allowed_b_sizes.push((x, y));
              self.allowed_b_size_rel_probs
                .push(allow_rectangle_prob * allowed_b_edge_rel_probs[i] * allowed_b_edge_rel_probs[j] / rel_prob_sum / rel_prob_sum);
            }
          }
        }
      }
    } else if cfg.contains("bSizesXY") {
      if cfg.contains("allowRectangleProb") {
        return Err(KataError::IOError(
          "Cannot specify allowRectangleProb when specifying bSizesXY, please adjust the relative frequency of rectangles yourself".to_string(),
        ));
      }
      self.allowed_b_sizes = cfg.get_non_negative_int_dashed_pairs("bSizesXY", 2, Board::MAX_LEN)?;
      self.allowed_b_size_rel_probs = cfg.get_doubles("bSizeRelProbs", 0.0, 1e100)?;

      let mut rel_prob_sum: f64 = 0.0;
      for p in self.allowed_b_size_rel_probs.iter() {
        rel_prob_sum += *p;
      }
      if rel_prob_sum <= 1e-100 {
        return Err(KataError::IOError("bSizeRelProbs must sum to a positive value".to_string()));
      }
    }

    if !cfg.contains("komiMean") && !(cfg.contains("komiAuto") && cfg.get_bool("komiAuto")?) {
      return Err(KataError::IOError("Must specify either komiMean=<komi value> or komiAuto=True in config".to_string()));
    }
    if cfg.contains("komiMean") && (cfg.contains("komiAuto") && cfg.get_bool("komiAuto")?) {
      return Err(KataError::IOError("Must specify only one of komiMean=<komi value> or komiAuto=True in config".to_string()));
    }

    self.komi_mean = if cfg.contains("komiMean") { cfg.get_float("komiMean", Rules::MIN_USER_KOMI, Rules::MAX_USER_KOMI)? } else { 7.5f32 };
    self.komi_stdev = if cfg.contains("komiStdev") { cfg.get_float("komiStdev", 0.0f32, 60.0f32)? } else { 0.0f32 };
    self.handicap_prob = if cfg.contains("handicapProb") { cfg.get_double("handicapProb", 0.0, 1.0)? } else { 0.0 };
    self.handicap_compensate_komi_prob =
      if cfg.contains("handicapCompensateKomiProb") { cfg.get_double("handicapCompensateKomiProb", 0.0, 1.0)? } else { 0.0 };
    self.komi_big_stdev_prob = if cfg.contains("komiBigStdevProb") { cfg.get_double("komiBigStdevProb", 0.0, 1.0)? } else { 0.0 };
    self.komi_big_stdev = if cfg.contains("komiBigStdev") { cfg.get_float("komiBigStdev", 0.0f32, 60.0f32)? } else { 10.0f32 };
    self.komi_bigger_stdev_prob = if cfg.contains("komiBiggerStdevProb") { cfg.get_double("komiBiggerStdevProb", 0.0, 1.0)? } else { 0.0 };
    self.komi_bigger_stdev = if cfg.contains("komiBiggerStdev") { cfg.get_float("komiBiggerStdev", 0.0f32, 120.0f32)? } else { 30.0f32 };
    self.handicap_komi_interp_zero_prob =
      if cfg.contains("handicapKomiInterpZeroProb") { cfg.get_double("handicapKomiInterpZeroProb", 0.0, 1.0)? } else { 0.0 };
    self.sgf_komi_interp_zero_prob = if cfg.contains("sgfKomiInterpZeroProb") { cfg.get_double("sgfKomiInterpZeroProb", 0.0, 1.0)? } else { 0.0 };
    self.komi_auto = if cfg.contains("komiAuto") { cfg.get_bool("komiAuto")? } else { false };

    self.fork_compensate_komi_prob =
      if cfg.contains("forkCompensateKomiProb") { cfg.get_double("forkCompensateKomiProb", 0.0, 1.0)? } else { self.handicap_compensate_komi_prob };
    self.sgf_compensate_komi_prob =
      if cfg.contains("sgfCompensateKomiProb") { cfg.get_double("sgfCompensateKomiProb", 0.0, 1.0)? } else { self.fork_compensate_komi_prob };
    self.komi_allow_integer_prob = if cfg.contains("komiAllowIntegerProb") { cfg.get_double("komiAllowIntegerProb", 0.0, 1.0)? } else { 1.0 };

    // Omitted from this copy: the generateCumProbs lambda and the startPosesFromSgfDir / hintPosesDir
    // branches (SGF and file loading; Sgf::PositionSample is not ported). The test sets neither key.
    self.start_poses_prob = 0.0;
    if cfg.contains("startPosesFromSgfDir") {
      panic!("test-local GameInitializer: startPosesFromSgfDir is not supported (SGF loading)");
    }
    self.hint_poses_prob = 0.0;
    if cfg.contains("hintPosesDir") {
      panic!("test-local GameInitializer: hintPosesDir is not supported (file loading)");
    }

    if self.allowed_b_sizes.is_empty() {
      return Err(KataError::IOError("bSizes or bSizesXY must have at least one value in ".to_string() + &cfg.get_file_name()));
    }
    if self.allowed_b_sizes.len() != self.allowed_b_size_rel_probs.len() {
      return Err(KataError::IOError("bSizes or bSizesXY and bSizeRelProbs must have same number of values in ".to_string() + &cfg.get_file_name()));
    }

    self.min_board_x_size = self.allowed_b_sizes[0].0;
    self.min_board_y_size = self.allowed_b_sizes[0].1;
    self.max_board_x_size = self.allowed_b_sizes[0].0;
    self.max_board_y_size = self.allowed_b_sizes[0].1;
    for b_size in self.allowed_b_sizes.iter() {
      self.min_board_x_size = std::cmp::min(self.min_board_x_size, b_size.0);
      self.min_board_y_size = std::cmp::min(self.min_board_y_size, b_size.1);
      self.max_board_x_size = std::cmp::max(self.max_board_x_size, b_size.0);
      self.max_board_y_size = std::cmp::max(self.max_board_y_size, b_size.1);
    }
    //for(const Sgf::PositionSample& pos : hintPoses) ... (hintPoses is empty in this copy)

    self.no_result_stdev = if cfg.contains("noResultStdev") { cfg.get_double("noResultStdev", 0.0, 1.0)? } else { 0.0 };
    self.num_extra_black_fixed = if cfg.contains("numExtraBlackFixed") { cfg.get_int("numExtraBlackFixed", 1, 18)? } else { 0 };
    self.draw_rand_radius = if cfg.contains("drawRandRadius") { cfg.get_double("drawRandRadius", 0.0, 1.0)? } else { 0.0 };
    Ok(())
  }

  /// upstream: GameInitializer::createGame (play.cpp), the overload without SearchParams — test-local copy
  /// (initialPosition and startPosSample are NULL at the only caller, so they are not parameters here).
  fn create_game(
    &mut self,
    board: &mut Board,
    pla: &mut Player,
    hist: &mut BoardHistory,
    extra_black_and_komi: &mut ExtraBlackAndKomi,
    play_settings: &PlaySettings,
    other_game_props: &mut OtherGameProperties,
  ) -> Result<(), KataError> {
    //Multiple threads will be calling this, and we have some mutable state such as rand.
    self.create_game_shared_unsynchronized(board, pla, hist, extra_black_and_komi, play_settings, other_game_props);
    if self.no_result_stdev != 0.0 || self.draw_rand_radius != 0.0 {
      return Err(KataError::StringError(
        "GameInitializer::createGame called in a mode that doesn't support specifying noResultStdev or drawRandRadius".to_string(),
      ));
    }
    Ok(())
  }

  /// upstream: GameInitializer::createRulesUnsynchronized (play.cpp) — test-local copy
  fn create_rules_unsynchronized(&mut self) -> Rules {
    let mut rules: Rules = Rules::new();
    rules.ko_rule = self.allowed_ko_rules[self.rand.next_uint_n(self.allowed_ko_rules.len() as u32) as usize];
    rules.scoring_rule = self.allowed_scoring_rules[self.rand.next_uint_n(self.allowed_scoring_rules.len() as u32) as usize];
    rules.tax_rule = self.allowed_tax_rules[self.rand.next_uint_n(self.allowed_tax_rules.len() as u32) as usize];
    rules.multi_stone_suicide_legal =
      self.allowed_multi_stone_suicide_legals[self.rand.next_uint_n(self.allowed_multi_stone_suicide_legals.len() as u32) as usize];

    if rules.scoring_rule == Rules::SCORING_AREA {
      rules.has_button = self.allowed_buttons[self.rand.next_uint_n(self.allowed_buttons.len() as u32) as usize];
    } else {
      rules.has_button = false;
    }
    rules
  }

  /// upstream: GameInitializer::createGameSharedUnsynchronized (play.cpp) — test-local copy, for
  /// initialPosition == NULL and startPosSample == NULL with no start/hint positions loaded.
  fn create_game_shared_unsynchronized(
    &mut self,
    board: &mut Board,
    pla: &mut Player,
    hist: &mut BoardHistory,
    extra_black_and_komi: &mut ExtraBlackAndKomi,
    play_settings: &PlaySettings,
    other_game_props: &mut OtherGameProperties,
  ) {
    //if(initialPosition != NULL) {...}: initialPosition is NULL here.

    let mut make_game_fair_prob: f64;

    let b_size_idx: i32 = self.rand.next_uint_rel_probs_n(&self.allowed_b_size_rel_probs, self.allowed_b_size_rel_probs.len()) as i32;
    let rules: Rules = self.create_rules_unsynchronized();

    //posSample stays NULL: startPosSample is NULL, startPosesProb and hintPosesProb are 0 (no rand draw).
    assert!(self.start_poses_prob == 0.0 && self.hint_poses_prob == 0.0);

    {
      let x_size: i32 = self.allowed_b_sizes[b_size_idx as usize].0;
      let y_size: i32 = self.allowed_b_sizes[b_size_idx as usize].1;
      *board = Board::new_x_y(x_size, y_size).expect("upstream: allowed sizes are in [2, Board::MAX_LEN]");
      *pla = P_BLACK;
      hist.clear(board, *pla, &rules, 0);

      *extra_black_and_komi = choose_extra_black_and_komi(
        self.komi_mean,
        self.komi_stdev,
        self.komi_allow_integer_prob,
        self.handicap_prob,
        self.num_extra_black_fixed,
        self.komi_big_stdev_prob,
        self.komi_big_stdev,
        self.komi_bigger_stdev_prob,
        self.komi_bigger_stdev,
        ((board.x_size * board.y_size) as f64).sqrt(),
        &mut self.rand,
      );
      set_komi_with_noise(extra_black_and_komi, hist, &mut self.rand);

      other_game_props.is_sgf_pos = false;
      other_game_props.is_hint_pos = false;
      other_game_props.allow_policy_init = true; //Handicap and regular games do allow policy init
      other_game_props.is_fork = false;
      other_game_props.is_hint_fork = false;
      other_game_props.hint_loc = Board::NULL_LOC;
      other_game_props.hint_turn = -1;
      other_game_props.training_weight = 1.0;
      make_game_fair_prob = if extra_black_and_komi.extra_black > 0 { self.handicap_compensate_komi_prob } else { 0.0 };
      extra_black_and_komi.interp_zero = if self.handicap_komi_interp_zero_prob > 0.0 && extra_black_and_komi.extra_black > 0 {
        self.rand.next_bool(self.handicap_komi_interp_zero_prob)
      } else {
        false
      };
    }

    let asymmetric_prob: f64 =
      if extra_black_and_komi.extra_black > 0 { play_settings.handicap_asymmetric_playout_prob } else { play_settings.normal_asymmetric_playout_prob };
    if asymmetric_prob > 0.0 && self.rand.next_bool(asymmetric_prob) {
      testAssert!(play_settings.max_asymmetric_ratio >= 1.0);
      let max_num_doublings: f64 = katago_core::port::math::ln(play_settings.max_asymmetric_ratio) / katago_core::port::math::ln(2.0);
      let num_doublings: f64 = self.rand.next_double_n(max_num_doublings);
      if extra_black_and_komi.extra_black > 0 || self.rand.next_bool(0.5) {
        other_game_props.playout_doubling_advantage_pla = C_WHITE;
        other_game_props.playout_doubling_advantage = num_doublings;
      } else {
        other_game_props.playout_doubling_advantage_pla = C_BLACK;
        other_game_props.playout_doubling_advantage = num_doublings;
      }
      make_game_fair_prob = katago_core::port::cpp::cpp_max(make_game_fair_prob, play_settings.min_asymmetric_compensate_komi_prob);
    }

    if self.komi_auto {
      if make_game_fair_prob > 0.0 {
        extra_black_and_komi.make_game_fair = self.rand.next_bool(make_game_fair_prob);
      }
      extra_black_and_komi.make_game_fair_for_empty_board = !extra_black_and_komi.make_game_fair;
    } else {
      if make_game_fair_prob > 0.0 {
        extra_black_and_komi.make_game_fair = self.rand.next_bool(make_game_fair_prob);
      }
      extra_black_and_komi.make_game_fair_for_empty_board = false;
    }
  }
}

/// The "Board size distribution" block of Tests::runNNLessSearchTests, then the function's tail
/// (`NeuralNet::globalCleanup(); cout << "Done" << endl;`), whose line ends the expected section.
fn board_size_distribution(_ctx: &mut Ctx) {
  let mut cfg = ConfigParser::new(false, true);
  cfg.override_key("koRules", "SIMPLE");
  cfg.override_key("scoringRules", "AREA");
  cfg.override_key("taxRules", "SEKI");
  cfg.override_key("multiStoneSuicideLegals", "false");
  cfg.override_key("hasButtons", "false");
  cfg.override_key("bSizes", "2,4,6,8");
  cfg.override_key("bSizeRelProbs", "1,2,3,4");
  cfg.override_key("allowRectangleProb", "0.3");
  cfg.override_key("komiAuto", "true");
  let mut game_init = GameInitializer::new_rand_seed(&mut cfg, "board size distribution random seed").expect("upstream: valid config");

  let mut board_size_distribution: BTreeMap<(i32, i32), i32> = BTreeMap::new();
  // Boxed: BoardHistory is ~34 KB (wasm debug stack, PORTING.md §13).
  let mut hist: Box<BoardHistory> = Box::new(BoardHistory::new());
  for _i in 0..100000 {
    let mut board: Board = Board::new();
    let mut pla: Player = C_EMPTY;
    // PORT NOTE: upstream declares a fresh `BoardHistory hist;` per iteration; createGame clears it
    // (hist.clear) before any read, so one boxed history is reused.
    let mut extra_black_and_komi: ExtraBlackAndKomi = ExtraBlackAndKomi::default();
    let mut other_game_props: OtherGameProperties = OtherGameProperties::default();
    game_init
      .create_game(&mut board, &mut pla, &mut hist, &mut extra_black_and_komi, &PlaySettings::new(), &mut other_game_props)
      .expect("upstream: createGame succeeds");
    *board_size_distribution.entry((board.x_size, board.y_size)).or_insert(0) += 1;
  }
  let mut x = 2;
  while x <= 8 {
    let mut y = 2;
    while y <= 8 {
      coutln!("{}x{} {}", x, y, board_size_distribution.entry((x, y)).or_insert(0));
      y += 2;
    }
    x += 2;
  }

  //NeuralNet::globalCleanup();
  coutln!("Done");
}

//------------------------------------------------------------------------------------------------
// Driver

enum Kind {
  Run(fn(&mut Ctx)),
  /// Not transcribed: the reason. The expected section is not compared. (None left; kept for sections a
  /// later upstream adds.)
  #[allow(dead_code)]
  Pending(&'static str),
}

/// Every header trio of the C++ in order, with the section that produces it. The title is checked against
/// the expected text, so a typo or a reordering here fails the test instead of misaligning it.
const SECTIONS: [(&str, Kind); 43] = [
  ("Basic search with debugSkipNeuralNet and chosen move randomization", Kind::Run(basic_search_and_chosen_move_randomization)),
  ("Testing preservation of search tree across moves", Kind::Run(preservation_of_search_tree_across_moves)),
  ("Testing pruning of search tree across moves due to root restrictions", Kind::Run(pruning_of_search_tree_across_moves_due_to_root_restrictions)),
  ("Testing search tree update near terminal positions", Kind::Run(search_tree_update_near_terminal_positions)),
  ("Testing pruning of search tree at root node due to symmetries: empty board", Kind::Run(root_symmetry_pruning_empty_board)),
  ("Testing pruning of search tree at root node due to symmetries: asymmetry board", Kind::Run(root_symmetry_pruning_asymmetry_board)),
  (
    "Testing pruning of search tree at root node due to symmetries, allowing only diagonal flips: empty board",
    Kind::Run(root_symmetry_pruning_diagonal_flips_empty_board),
  ),
  (
    "Testing pruning of search tree at root node due to symmetries, allowing only diagonal flips: only one diagonal possible",
    Kind::Run(root_symmetry_pruning_diagonal_flips_one_diagonal),
  ),
  (
    "Testing pruning of search tree at root node due to symmetries, allowing only diagonal flips: empty board, with avoid moves",
    Kind::Run(root_symmetry_pruning_diagonal_flips_empty_board_avoid_moves),
  ),
  (
    "Testing pruning of search tree at root node due to symmetries, allowing only diagonal flips: only one diagonal possible, with avoid moves",
    Kind::Run(root_symmetry_pruning_diagonal_flips_one_diagonal_avoid_moves),
  ),
  ("Testing pruning of search tree at root node, no symmetries, with avoid moves", Kind::Run(root_pruning_no_symmetries_avoid_moves)),
  ("Non-square board search", Kind::Run(non_square_board_search)),
  ("Visualize dirichlet noise", Kind::Run(visualize_dirichlet_noise)),
  ("Search tolerates moving past game end", Kind::Run(search_tolerates_moving_past_game_end)),
  ("Integrity of value bias, mem safety and updates", Kind::Run(integrity_of_value_bias)),
  ("Value bias with ko", Kind::Run(value_bias_with_ko)),
  ("Analysis json", Kind::Run(analysis_json_section)),
  ("Analysis json with moves ownership and stdev", Kind::Run(analysis_json_with_moves_ownership_and_stdev)),
  ("Analysis json with moves ownership and stdev and symmetry", Kind::Run(analysis_json_with_moves_ownership_and_stdev_and_symmetry)),
  ("Analysis json 2", Kind::Run(analysis_json_2)),
  ("Search results at 0, 1, 2 visits, and at terminal position", Kind::Run(search_results_at_0_1_2_visits_and_terminal_position)),
  ("Testing coherence of search tree recursive walking", Kind::Run(coherence_of_search_tree_recursive_walking)),
  ("Testing avoiding of all or almost all moves", Kind::Run(avoiding_of_all_or_almost_all_moves)),
  ("Graph search, opening", Kind::Run(graph_search_opening)),
  ("Graph search, 7x7 big fight", Kind::Run(graph_search_7x7_big_fight)),
  ("Graph search, 7x7 endgame kos", Kind::Run(graph_search_7x7_endgame_kos)),
  ("FPU parent weight by visited policy false", Kind::Run(fpu_false)),
  ("FPU parent weight by visited policy 1.0", Kind::Run(fpu_1_0)),
  ("FPU parent weight by visited policy 2.5", Kind::Run(fpu_2_5)),
  ("FPU parent weight by visited policy 0.5", Kind::Run(fpu_0_5)),
  ("Policy optimism with tree reuse", Kind::Run(policy_optimism_with_tree_reuse)),
  ("Policy optimism with tree reuse, 0 root", Kind::Run(policy_optimism_with_tree_reuse_0_root)),
  ("Zero node search", Kind::Run(zero_node_search)),
  ("Chosen move probs", Kind::Run(chosen_move_probs)),
  ("Uninitialized search params", Kind::Run(uninitialized_search_params)),
  ("SearchParams forTestsV1", Kind::Run(search_params_for_tests_v1)),
  ("SearchParams for GTP", Kind::Run(search_params_for_gtp)),
  ("SearchParams for Analysis", Kind::Run(search_params_for_analysis)),
  ("SearchParams for Match", Kind::Run(search_params_for_match)),
  ("SearchParams for Benchmark", Kind::Run(search_params_for_benchmark)),
  ("SearchParams for Other", Kind::Run(search_params_for_other)),
  ("SearchParams for Distributed", Kind::Run(search_params_for_distributed)),
  // Its expected text also holds the final "Done" line of runNNLessSearchTests.
  ("Board size distribution", Kind::Run(board_size_distribution)),
];

/// Number of sections that are compared (the preamble "Running neuralnetless search tests" plus every `Kind::Run`).
const NUM_COMPARED: usize = 1 + 43;
const NUM_PENDING: usize = 0;

/// `(title, text)` for every section of the expected stdout: the preamble (title "") and one per header
/// trio (a line of 67 '=', a title, a line of 67 '='); a section runs up to the next trio.
fn split_sections(text: &str) -> Vec<(String, String)> {
  let eq = "=".repeat(67);
  let lines: Vec<&str> = text.split_inclusive('\n').collect();
  let is_eq = |l: &str| l.strip_suffix('\n').unwrap_or(l) == eq;
  let mut out: Vec<(String, String)> = vec![(String::new(), String::new())];
  let mut i = 0;
  while i < lines.len() {
    if i + 2 < lines.len() && is_eq(lines[i]) && is_eq(lines[i + 2]) {
      let title = lines[i + 1].strip_suffix('\n').unwrap_or(lines[i + 1]).to_string();
      out.push((title, String::new()));
      let cur = out.last_mut().unwrap();
      cur.1.push_str(lines[i]);
      cur.1.push_str(lines[i + 1]);
      cur.1.push_str(lines[i + 2]);
      i += 3;
    } else {
      out.last_mut().unwrap().1.push_str(lines[i]);
      i += 1;
    }
  }
  out
}

enum Status {
  Pass,
  Fail(String),
  Pending(&'static str),
}

/// Runs `f`, returning the panic message ("msg (at file:line)") instead of unwinding. On wasm32 panics abort,
/// so there is nothing to catch: the first failing section ends the test.
#[cfg(not(target_arch = "wasm32"))]
fn try_catch<R>(f: impl FnOnce() -> R) -> Result<R, String> {
  match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
    Ok(r) => Ok(r),
    Err(_) => Err(LAST_PANIC.with(|p| p.borrow().clone())),
  }
}

#[cfg(target_arch = "wasm32")]
fn try_catch<R>(f: impl FnOnce() -> R) -> Result<R, String> {
  Ok(f())
}

#[cfg(not(target_arch = "wasm32"))]
fn install_panic_hook() {
  std::panic::set_hook(Box::new(|info| {
    let msg = if let Some(s) = info.payload().downcast_ref::<&str>() {
      s.to_string()
    } else if let Some(s) = info.payload().downcast_ref::<String>() {
      s.clone()
    } else {
      "<non-string panic payload>".to_string()
    };
    let at = info.location().map(|l| format!("{}:{}", l.file(), l.line())).unwrap_or_default();
    let bt = std::backtrace::Backtrace::capture();
    let mut s = format!("{} (at {})", msg, at);
    if bt.status() == std::backtrace::BacktraceStatus::Captured {
      s.push_str(&format!("\n{}", bt));
    }
    LAST_PANIC.with(|p| *p.borrow_mut() = s);
  }));
}

#[cfg(target_arch = "wasm32")]
fn install_panic_hook() {}

/// Runs every section, returning (title, status) in order.
fn run_all(expected: &str) -> Vec<(String, Status)> {
  install_panic_hook();
  let expected_sections = split_sections(expected);
  assert_eq!(expected_sections.len(), 1 + SECTIONS.len(), "header trios in the expected text vs SECTIONS");
  let mut results: Vec<(String, Status)> = Vec::new();

  let compare = |title: &str, actual: &str, expected: &str, panic_msg: Option<String>| -> Status {
    let label = format!("section [{}]", title);
    let cmp = try_catch(|| compare_lines(&label, actual, expected));
    match (cmp, panic_msg) {
      (Ok(()), None) => Status::Pass,
      (Ok(()), Some(p)) => Status::Fail(format!("PANIC: {}", p)),
      (Err(m), None) => Status::Fail(m),
      (Err(m), Some(p)) => Status::Fail(format!("PANIC: {}\n  partial output vs expected: {}", p, m)),
    }
  };

  //Tests::runNNLessSearchTests: the preamble
  {
    let start = OUT.with(|o| o.borrow().len());
    coutln!("Running neuralnetless search tests");
    //NeuralNet::globalInitialize(); (no-op for the debugSkipNeuralNet backend)
    let end = OUT.with(|o| o.borrow().len());
    let actual = OUT.with(|o| o.borrow()[start..end].to_string());
    results.push(("<preamble>".to_string(), compare("<preamble>", &actual, &expected_sections[0].1, None)));
  }

  for (k, (title, kind)) in SECTIONS.iter().enumerate() {
    let (exp_title, exp_text) = &expected_sections[k + 1];
    assert_eq!(exp_title, title, "section {} title vs the expected text", k + 1);
    match kind {
      Kind::Pending(reason) => {
        results.push((title.to_string(), Status::Pending(reason)));
      }
      Kind::Run(f) => {
        let start = OUT.with(|o| o.borrow().len());
        coutln!("===================================================================");
        coutln!("{}", title);
        coutln!("===================================================================");
        let mut ctx = Ctx::new();
        let ran = try_catch(|| f(&mut ctx));
        let end = OUT.with(|o| o.borrow().len());
        let actual = OUT.with(|o| o.borrow()[start..end].to_string());
        let panic_msg = ran.err();
        drop(ctx);
        let status = compare(title, &actual, exp_text, panic_msg);
        results.push((title.to_string(), status));
      }
    }
  }
  //NeuralNet::globalCleanup(); cout << "Done" << endl;  -- printed by board_size_distribution, the last section
  results
}

fn run_nn_less_search_tests() {
  init();
  ScoreValue::init_tables(); //upstream's main(): "MUST BE CALLED AT PROGRAM START!"
  install_logger_and_clock();
  let mut cases = Cases::new("testsearchnonn::run_nn_less_search_tests");
  let expected = fixture(F, "runnnlesssearchtests.expected0.txt");
  assert!(expected.starts_with("Running neuralnetless search tests\n"));
  assert!(expected.ends_with("Done\n"));

  //Big stack: Search (~40 KB), BoardHistory (~34 KB) and the recursive playout futures, in a debug build.
  #[cfg(not(target_arch = "wasm32"))]
  let results = {
    let e = expected.clone();
    std::thread::Builder::new()
      .name("testsearchnonn".to_string())
      .stack_size(256 << 20)
      .spawn(move || run_all(&e))
      .unwrap()
      .join()
      .expect("run_all panicked outside the per-section catch")
  };
  //Back to the default hook (run_all installed the capturing one) so a failure below prints normally.
  #[cfg(not(target_arch = "wasm32"))]
  let _ = std::panic::take_hook();
  #[cfg(target_arch = "wasm32")]
  let results = run_all(&expected);

  let mut num_pass = 0;
  let mut num_fail = 0;
  let mut num_pending = 0;
  let mut summary = String::new();
  for (title, status) in results.iter() {
    match status {
      Status::Pass => {
        num_pass += 1;
        cases.ran();
        let _ = writeln!(summary, "pass    {}", title);
      }
      Status::Fail(msg) => {
        num_fail += 1;
        cases.ran();
        let _ = writeln!(summary, "FAIL    {}\n        {}", title, msg.replace('\n', "\n        "));
      }
      Status::Pending(reason) => {
        num_pending += 1;
        let _ = writeln!(summary, "pending: {} ({})", title, reason);
      }
    }
  }
  crate::report!("testsearchnonn sections: {} pass, {} FAIL, {} pending\n{}", num_pass, num_fail, num_pending, summary);

  assert_eq!(num_pass + num_fail, NUM_COMPARED, "compared sections");
  assert_eq!(num_pending, NUM_PENDING, "pending sections");
  assert!(num_fail == 0, "{} of {} compared sections differ from the upstream output:\n{}", num_fail, NUM_COMPARED, summary);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn nn_less_search_tests() {
  run_nn_less_search_tests();
}
