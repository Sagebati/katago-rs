//! Differential search oracle: the ported Search (search/*.rs) against the unmodified upstream Search.
//!
//! fixtures/upstream-v1.16.5/unit/oracle/search_nonn.cpp runs many deterministic searches
//! (numSearchThreads = 1) on an NNEvaluator with debugSkipNeuralNet = true and prints, per case, one
//! "CASE" line holding every input (board size, rules, komi, initial player, the move list as raw
//! locs, SearchParams overrides, search rand seed, maxVisits, nnXLen/nnYLen, and the op stream) followed
//! by what the ops printed (root visits, every root child, getRootValues/getPrunedRootValues/..., all
//! AnalysisData fields, printTree/printPV text, getAnalysisJson dumps, chosen moves, play selection
//! values, ownership, node table sizes and key hashes, crafted extreme root ownership (XO), time carried over by
//! tree reuse (ECT), the eval cache (EC/NE/ECC/ECN/ECS: dumps of the cache and of the nodes' entries, a second Search sharing it),
//! the pattern bonus table (PB/PBN/PBH and the nodes' patternBonusHash) and anti-mirror state (MI, HC for handicap) ...;
//! floats as IEEE-754 hex bits) and "END <n>".
//! This test rebuilds each case from its CASE line alone, executes the same op stream through the
//! port on a futures LocalPool and compares the text with `common::compare_lines`, per case: every
//! mismatching case is reported with its first differing line (the test fails at the end if any).
//! The op language and the generators are documented in search_nonn.cpp's header. For exploration only, the
//! environment variable SEARCH_ORACLE_FILE replays another output of that program (ORACLE_SEED, ORACLE_STRESS_ONLY).
//!
//! Second test (Rust only, no C++ counterpart): `multithread_virtual_threads` runs a subset of the cases
//! with numSearchThreads = 2, 4 and 16 virtual threads. Upstream is nondeterministic there, so it checks
//! determinism of the port (two fresh runs give byte-identical dumps), the invariants listed on that
//! test, and that nothing stalls (`common::run_or_stall_panic`).
//!
//! Timing dependence: none (no maxTime/time controls; the injected clock is a constant).
//! Not a port of upstream code.

#![allow(clippy::too_many_arguments, clippy::field_reassign_with_default, clippy::needless_range_loop, clippy::write_with_newline, clippy::manual_clamp)]

mod common;

use std::cell::Cell;
use std::cell::RefCell;
use std::collections::HashMap;
use std::collections::BTreeSet;
use std::collections::HashSet;
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
use katago_core::core::global::KataError;
use katago_core::core::hash::Hash128;
use katago_core::core::timer::set_clock;
use katago_core::game::board::*;
use katago_core::game::boardhistory::BoardHistory;
use katago_core::game::rules::Rules;
use katago_core::neuralnet::desc::ModelDesc;
use katago_core::neuralnet::nneval::NNEvaluator;
use katago_core::neuralnet::nneval::NNResultBuf;
use katago_core::neuralnet::nninputs::NNOutput;
use katago_core::neuralnet::nninputs::ScoreValue;
use katago_core::neuralnet::nninterface::*;
use katago_core::port::executor::LocalBoxFuture;
use katago_core::port::executor::Spawner;
use katago_core::port::node_arena::NodeIdx;
use katago_core::port::nlohmann;
use katago_core::search::analysisdata::AnalysisData;
use katago_core::search::evalcache::EvalCacheTable;
use katago_core::search::patternbonustable::PatternBonusTable;
use katago_core::search::reportedsearchvalues::ReportedSearchValues;
use katago_core::search::search::Search;
use katago_core::search::searchnode::SearchNode;
use katago_core::search::searchparams::SearchParams;
use katago_core::search::searchprint::PrintTreeOptions;

//------------------------------------------------------------------------------------------------
// cout, the Logger, the injected clock

thread_local! {
  /// Everything the case prints: the ops' text and the Search's Logger lines.
  static OUT: RefCell<String> = const { RefCell::new(String::new()) };
  /// "message (at file:line)" of the last panic on a capturing thread.
  static LAST_PANIC: RefCell<String> = const { RefCell::new(String::new()) };
  static CAPTURE: Cell<bool> = const { Cell::new(false) };
}

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

macro_rules! coutln {
  () => {{
    cout!("\n");
  }};
  ($($t:tt)*) => {{
    cout!($($t)*);
    cout!("\n");
  }};
}

/// The oracle's `Logger logger(nullptr, false, false, false); logger.addOStream(out)` for the Search only:
/// `Logger::write(str)` = `": " + str + endl`. The NNEvaluator got a NULL logger there, so records of the
/// port's neuralnet modules are dropped.
struct SearchLogger;

impl log::Log for SearchLogger {
  fn enabled(&self, _metadata: &log::Metadata) -> bool {
    true
  }
  fn log(&self, record: &log::Record) {
    if record.target().starts_with("katago_core::search") {
      coutln!(": {}", record.args());
    }
  }
  fn flush(&self) {}
}

static LOGGER: SearchLogger = SearchLogger;

fn install_logger_clock_and_hook() {
  static ONCE: OnceLock<()> = OnceLock::new();
  ONCE.get_or_init(|| {
    log::set_logger(&LOGGER).expect("no other logger in this test binary");
    log::set_max_level(log::LevelFilter::Trace);
    #[cfg(not(target_arch = "wasm32"))]
    {
      let prev = std::panic::take_hook();
      std::panic::set_hook(Box::new(move |info| {
        if CAPTURE.with(|c| c.get()) {
          let msg = if let Some(s) = info.payload().downcast_ref::<&str>() {
            s.to_string()
          } else if let Some(s) = info.payload().downcast_ref::<String>() {
            s.clone()
          } else {
            "<non-string panic payload>".to_string()
          };
          let at = info.location().map(|l| format!("{}:{}", l.file(), l.line())).unwrap_or_default();
          LAST_PANIC.with(|p| *p.borrow_mut() = format!("{} (at {})", msg, at));
        } else {
          prev(info);
        }
      }));
    }
  });
  set_clock(common::fixed_clock);
}

/// Runs `f`, returning the panic message instead of unwinding (only the capturing threads' panics are swallowed).
#[cfg(not(target_arch = "wasm32"))]
fn try_catch<R>(f: impl FnOnce() -> R) -> Result<R, String> {
  CAPTURE.with(|c| c.set(true));
  let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
  CAPTURE.with(|c| c.set(false));
  match r {
    Ok(r) => Ok(r),
    Err(_) => Err(LAST_PANIC.with(|p| p.borrow().clone())),
  }
}

#[cfg(target_arch = "wasm32")]
fn try_catch<R>(f: impl FnOnce() -> R) -> Result<R, String> {
  Ok(f())
}

//------------------------------------------------------------------------------------------------
// Backend that must never be reached (debugSkipNeuralNet), the executor, the evaluator

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

/// One per case: the executor the NN server task and the search futures run on.
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

/// upstream: TestSearchCommon::startNNEval (testsearchcommon.cpp) with the arguments the oracle uses:
/// model name "search_nonn", inputsUseNHWC, useNHWC = useFP16 = false, debugSkipNeuralNet,
/// requireExactNNLen = false, defaultSymmetry 0 (so no randomize), nnCache 2^16, batch 16.
fn start_nn_eval(ctx: &mut Ctx, seed: &str, nn_x_len: i32, nn_y_len: i32) -> Rc<NNEvaluator> {
  let gpu_idx_by_server_thread: Vec<i32> = vec![0];
  let nn_rand_seed = "runSearchTestsRandSeed".to_string() + seed;
  let nn_eval = NNEvaluator::new(
    "search_nonn",
    "search_nonn",
    "",
    16,
    nn_x_len,
    nn_y_len,
    false,
    true,
    16,
    12,
    true,
    "",
    "",
    false,
    enabled_t::False,
    enabled_t::False,
    1,
    &gpu_idx_by_server_thread,
    &nn_rand_seed,
    false,
    0,
    ctx.backend.clone(),
    ctx.spawner.clone(),
    Vec::new(),
  )
  .unwrap();
  let nn_eval = Rc::new(nn_eval);
  run_or_stall_panic(&mut ctx.pool, "spawnServerThreads", nn_eval.spawn_server_threads()).unwrap();
  nn_eval
}

/// `delete nnEval;` Everything else holding the evaluator must be dropped first.
fn delete_nn_eval(ctx: &mut Ctx, nn_eval: Rc<NNEvaluator>) {
  nn_eval.kill_server_threads();
  ctx.pool.run_until_stalled();
  assert_eq!(Rc::strong_count(&nn_eval), 1, "the serve task ended and released the evaluator");
  drop(nn_eval);
}

//------------------------------------------------------------------------------------------------
// Parsing a CASE line

struct Spec {
  line: String,
  idx: i32,
  kind: String,
  nn_x: i32,
  nn_y: i32,
  nn_seed: String,
  xs: i32,
  ys: i32,
  rules: Rules,
  initial_pla: Player,
  pla: Player,
  moves: Vec<(Loc, Player)>,
  search_seed: String,
  base: i32,
  max_visits: i64,
  params: Vec<(String, String)>,
  ops: Vec<String>,
}

fn parse_spec(line: &str) -> Spec {
  let toks: Vec<&str> = line.split(' ').collect();
  assert_eq!(toks[0], "CASE", "case line: {}", line);
  let mut kv: HashMap<&str, &str> = HashMap::new();
  for t in &toks[2..] {
    let (k, v) = t.split_once('=').unwrap_or_else(|| panic!("bad token {:?} in {}", t, line));
    kv.insert(k, v);
  }
  let get = |k: &str| -> &str { kv.get(k).unwrap_or_else(|| panic!("missing {} in {}", k, line)) };
  let int = |k: &str| -> i32 { get(k).parse().unwrap() };
  let (nn_x, nn_y) = get("nn").split_once('x').unwrap();
  let (xs, ys) = get("board").split_once('x').unwrap();
  let rules = Rules::new_full(
    int("ko"),
    int("score"),
    int("tax"),
    int("suicide") != 0,
    int("button") != 0,
    int("whb"),
    int("fpo") != 0,
    f32::from_bits(u32::from_str_radix(get("komi"), 16).unwrap()),
  );
  let moves: Vec<(Loc, Player)> = if get("moves") == "-" {
    Vec::new()
  } else {
    get("moves")
      .split(',')
      .map(|m| {
        let (l, p) = m.split_once('/').unwrap();
        (l.parse::<i32>().unwrap() as Loc, p.parse::<i32>().unwrap() as Player)
      })
      .collect()
  };
  let params: Vec<(String, String)> = if get("params") == "-" {
    Vec::new()
  } else {
    get("params")
      .split(',')
      .map(|p| {
        let (k, v) = p.split_once('=').unwrap();
        (k.to_string(), v.to_string())
      })
      .collect()
  };
  Spec {
    line: line.to_string(),
    idx: toks[1].parse().unwrap(),
    kind: get("kind").to_string(),
    nn_x: nn_x.parse().unwrap(),
    nn_y: nn_y.parse().unwrap(),
    nn_seed: get("nnseed").to_string(),
    xs: xs.parse().unwrap(),
    ys: ys.parse().unwrap(),
    rules,
    initial_pla: int("ipla") as Player,
    pla: int("pla") as Player,
    moves,
    search_seed: get("sseed").to_string(),
    base: int("base"),
    max_visits: get("maxvisits").parse().unwrap(),
    params,
    ops: get("ops").split(';').map(|s| s.to_string()).collect(),
  }
}

/// The fixture split into (CASE line, whole block including the END line).
fn split_cases(text: &str) -> Vec<(String, String)> {
  let mut out: Vec<(String, String)> = Vec::new();
  let mut cur: Option<(String, String)> = None;
  for l in text.split_inclusive('\n') {
    if l.starts_with("CASE ") {
      assert!(cur.is_none(), "CASE inside a case");
      cur = Some((l.trim_end_matches('\n').to_string(), String::new()));
    }
    let c = cur.as_mut().expect("text outside a case");
    c.1.push_str(l);
    if l.starts_with("END ") {
      out.push(cur.take().unwrap());
    }
  }
  assert!(cur.is_none(), "unterminated last case");
  out
}

/// The oracle's `setParam`: key=value text onto a SearchParams (strtod / strtoll / "1" for bools).
fn set_param(p: &mut SearchParams, k: &str, v: &str) -> bool {
  let d: f64 = v.parse().unwrap_or(0.0);
  let l: i64 = v.parse().unwrap_or(0);
  let b = v == "1";
  match k {
    "winLossUtilityFactor" => p.win_loss_utility_factor = d,
    "staticScoreUtilityFactor" => p.static_score_utility_factor = d,
    "dynamicScoreUtilityFactor" => p.dynamic_score_utility_factor = d,
    "dynamicScoreCenterZeroWeight" => p.dynamic_score_center_zero_weight = d,
    "dynamicScoreCenterScale" => p.dynamic_score_center_scale = d,
    "noResultUtilityForWhite" => p.no_result_utility_for_white = d,
    "drawEquivalentWinsForWhite" => p.draw_equivalent_wins_for_white = d,
    "cpuctExploration" => p.cpuct_exploration = d,
    "cpuctExplorationLog" => p.cpuct_exploration_log = d,
    "cpuctExplorationBase" => p.cpuct_exploration_base = d,
    "cpuctUtilityStdevPrior" => p.cpuct_utility_stdev_prior = d,
    "cpuctUtilityStdevPriorWeight" => p.cpuct_utility_stdev_prior_weight = d,
    "cpuctUtilityStdevScale" => p.cpuct_utility_stdev_scale = d,
    "fpuReductionMax" => p.fpu_reduction_max = d,
    "fpuLossProp" => p.fpu_loss_prop = d,
    "fpuParentWeightByVisitedPolicy" => p.fpu_parent_weight_by_visited_policy = b,
    "fpuParentWeightByVisitedPolicyPow" => p.fpu_parent_weight_by_visited_policy_pow = d,
    "fpuParentWeight" => p.fpu_parent_weight = d,
    "policyOptimism" => p.policy_optimism = d,
    "valueWeightExponent" => p.value_weight_exponent = d,
    "useNoisePruning" => p.use_noise_pruning = b,
    "noisePruneUtilityScale" => p.noise_prune_utility_scale = d,
    "noisePruningCap" => p.noise_pruning_cap = d,
    "useUncertainty" => p.use_uncertainty = b,
    "uncertaintyCoeff" => p.uncertainty_coeff = d,
    "uncertaintyExponent" => p.uncertainty_exponent = d,
    "uncertaintyMaxWeight" => p.uncertainty_max_weight = d,
    "useGraphSearch" => p.use_graph_search = b,
    "graphSearchRepBound" => p.graph_search_rep_bound = l as i32,
    "graphSearchCatchUpLeakProb" => p.graph_search_catch_up_leak_prob = d,
    "rootNoiseEnabled" => p.root_noise_enabled = b,
    "rootDirichletNoiseTotalConcentration" => p.root_dirichlet_noise_total_concentration = d,
    "rootDirichletNoiseWeight" => p.root_dirichlet_noise_weight = d,
    "rootPolicyTemperature" => p.root_policy_temperature = d,
    "rootPolicyTemperatureEarly" => p.root_policy_temperature_early = d,
    "rootFpuReductionMax" => p.root_fpu_reduction_max = d,
    "rootFpuLossProp" => p.root_fpu_loss_prop = d,
    "rootNumSymmetriesToSample" => p.root_num_symmetries_to_sample = l as i32,
    "rootSymmetryPruning" => p.root_symmetry_pruning = b,
    "rootDesiredPerChildVisitsCoeff" => p.root_desired_per_child_visits_coeff = d,
    "rootPolicyOptimism" => p.root_policy_optimism = d,
    "chosenMoveTemperature" => p.chosen_move_temperature = d,
    "chosenMoveTemperatureEarly" => p.chosen_move_temperature_early = d,
    "chosenMoveTemperatureHalflife" => p.chosen_move_temperature_halflife = d,
    "chosenMoveTemperatureOnlyBelowProb" => p.chosen_move_temperature_only_below_prob = d,
    "chosenMoveSubtract" => p.chosen_move_subtract = d,
    "chosenMovePrune" => p.chosen_move_prune = d,
    "useLcbForSelection" => p.use_lcb_for_selection = b,
    "lcbStdevs" => p.lcb_stdevs = d,
    "minVisitPropForLCB" => p.min_visit_prop_for_lcb = d,
    "useNonBuggyLcb" => p.use_non_buggy_lcb = b,
    "rootEndingBonusPoints" => p.root_ending_bonus_points = d,
    "rootPruneUselessMoves" => p.root_prune_useless_moves = b,
    "conservativePass" => p.conservative_pass = b,
    "fillDameBeforePass" => p.fill_dame_before_pass = b,
    "wideRootNoise" => p.wide_root_noise = d,
    "enablePassingHacks" => p.enable_passing_hacks = b,
    "enableMorePassingHacks" => p.enable_more_passing_hacks = b,
    "playoutDoublingAdvantage" => p.playout_doubling_advantage = d,
    "ignorePreRootHistory" => p.ignore_pre_root_history = b,
    "ignoreAllHistory" => p.ignore_all_history = b,
    "subtreeValueBiasFactor" => p.subtree_value_bias_factor = d,
    "subtreeValueBiasFreeProp" => p.subtree_value_bias_free_prop = d,
    "subtreeValueBiasWeightExponent" => p.subtree_value_bias_weight_exponent = d,
    "nodeTableShardsPowerOfTwo" => p.node_table_shards_power_of_two = l as i32,
    "maxVisits" => p.max_visits = l,
    "maxPlayouts" => p.max_playouts = l,
    "searchFactorAfterOnePass" => p.search_factor_after_one_pass = d,
    "searchFactorAfterTwoPass" => p.search_factor_after_two_pass = d,
    "futileVisitsThreshold" => p.futile_visits_threshold = d,
    "treeReuseCarryOverTimeFactor" => p.tree_reuse_carry_over_time_factor = d,
    "avoidMYTDaggerHackPla" => p.avoid_myt_dagger_hack_pla = l as Player,
    "playoutDoublingAdvantagePla" => p.playout_doubling_advantage_pla = l as Player,
    "useEvalCache" => p.use_eval_cache = b,
    "evalCacheMinVisits" => p.eval_cache_min_visits = l,
    "subtreeValueBiasTableNumShards" => p.subtree_value_bias_table_num_shards = l as i32,
    "antiMirror" => p.anti_mirror = b,
    "avoidRepeatedPatternUtility" => p.avoid_repeated_pattern_utility = d,
    "nnPolicyTemperature" => p.nn_policy_temperature = v.parse::<f32>().unwrap_or(0.0),
    _ => return false,
  }
  true
}

fn base_params(base: i32) -> SearchParams {
  match base {
    1 => SearchParams::for_tests_v1(),
    2 => SearchParams::for_tests_v2(),
    _ => SearchParams::new(),
  }
}

//------------------------------------------------------------------------------------------------
// Printing helpers (mirror search_nonn.cpp byte for byte)

fn fnv64(s: &str) -> u64 {
  let mut h: u64 = 0xcbf29ce484222325;
  for b in s.bytes() {
    h ^= b as u64;
    h = h.wrapping_mul(0x100000001b3);
  }
  h
}

/// upstream mirror: printNodeTableKeys (search_nonn.cpp): the node table's keys in table order (shard by shard,
/// each shard's map in Hash128 order), "<shard> <hash1> <hash0>" per entry, as a count and FNV-1a 64.
fn print_node_table_keys(search: &Search) {
  let mut text = String::new();
  let mut n: usize = 0;
  for (i, shard) in search.node_table.entries.borrow().iter().enumerate() {
    for k in shard.keys() {
      let _ = writeln!(text, "{} {:016X} {:016X}", i, k.hash1, k.hash0);
      n += 1;
    }
  }
  coutln!("ntkeys n={} fnv={:016X}", n, fnv64(&text));
}

/// upstream mirror: printEvalCache (search_nonn.cpp), op EC: "noec" if the search has no eval cache, else a count line
/// and the first EC_FULL entries in full; the count/FNV-1a 64 cover every shard in order, each shard's map in Hash128 order.
const EC_FULL: usize = 2;

fn print_eval_cache(search: &Search) {
  let ec = match &search.eval_cache {
    None => {
      coutln!("noec");
      return;
    }
    Some(ec) => ec,
  };
  let entries = ec.entries.borrow();
  let mut text = String::new();
  let mut head = String::new();
  let (mut n, mut nnull, mut nfe) = (0usize, 0usize, 0usize);
  for (i, shard) in entries.iter().enumerate() {
    for (k, v) in shard.iter() {
      let mut line = format!("{} {:016X} {:016X}", i, k.hash1, k.hash0);
      match v {
        None => {
          line.push_str(" null");
          nnull += 1;
        }
        Some(e) => {
          let _ = write!(
            line,
            " w={:08X} wl={:08X} nr={:08X} sm={:08X} lead={:08X} fe={}",
            e.cache_weight.to_bits(),
            e.avg_win_loss.to_bits(),
            e.avg_no_result.to_bits(),
            e.avg_score_mean.to_bits(),
            e.avg_lead.to_bits(),
            e.first_explore_evals.len()
          );
          for (loc, fe) in &e.first_explore_evals {
            let _ = write!(line, " {}:{:08X}:{:08X}:{:08X}", loc, fe.avg_win_loss.to_bits(), fe.avg_score_mean.to_bits(), fe.cache_weight.to_bits());
            nfe += 1;
          }
        }
      }
      line.push('\n');
      if n < EC_FULL {
        head.push_str(&line);
      }
      text.push_str(&line);
      n += 1;
    }
  }
  coutln!("ec shards={} n={} null={} fe={} fnv={:016X}", entries.len(), n, nnull, nfe, fnv64(&text));
  cout!("{}", head);
}

/// upstream mirror: printNodeExtras (search_nonn.cpp), op NE: per node of the tree (enumerateTreePostOrder order): whether it
/// holds an evalCacheEntry (and its cacheWeight), its patternBonusHash.
fn print_node_extras(search: &Search) {
  if search.root_node.is_none() {
    coutln!("noroot");
    return;
  }
  let nodes = search.enumerate_tree_post_order();
  let (mut with_entry, mut with_pattern) = (0, 0);
  let mut text = String::new();
  for idx in &nodes {
    let node: Rc<SearchNode> = search.node_arena.get(*idx);
    let entry = node.eval_cache_entry.borrow();
    let he = entry.is_some();
    let ph = node.pattern_bonus_hash.get();
    let hp = ph.hash0 != 0 || ph.hash1 != 0;
    with_entry += he as i32;
    with_pattern += hp as i32;
    let _ = writeln!(text, "{} {:016X} {:016X} {:08X}", he as i32, ph.hash1, ph.hash0, entry.as_ref().map_or(0u32, |e| e.cache_weight.to_bits()));
  }
  coutln!("ne nodes={} ece={} pbh={} fnv={:016X}", nodes.len(), with_entry, with_pattern, fnv64(&text));
}

/// upstream mirror: printPatternBonusTable (search_nonn.cpp): a count and FNV-1a 64 of "<shard> <hash1> <hash0> <utilityBonus bits>"
/// lines (shard by shard, each map in Hash128 order) and the sum of the bonuses in that order.
fn print_pattern_bonus_table(label: &str, t: &PatternBonusTable) {
  let mut text = String::new();
  let mut n = 0usize;
  let mut sum = 0.0f64;
  for (i, shard) in t.entries.iter().enumerate() {
    for (k, v) in shard.iter() {
      let _ = writeln!(text, "{} {:016X} {:016X} {:016X}", i, k.hash1, k.hash0, v.utility_bonus.to_bits());
      sum += v.utility_bonus;
      n += 1;
    }
  }
  coutln!("{} shards={} n={} sum={:016X} fnv={:016X}", label, t.entries.len(), n, sum.to_bits(), fnv64(&text));
}

/// upstream mirror: patternBonusHashOp (search_nonn.cpp), op PBH:<n>: the PatternBonusTable on the root position, no search involved.
fn pattern_bonus_hash_op(search: &Search, num_add: i32) {
  let board: &Board = search.get_root_board();
  let hist: &BoardHistory = search.get_root_hist();
  let mut locs: Vec<Loc> = vec![Board::NULL_LOC, Board::PASS_LOC];
  for y in 0..board.y_size {
    for x in 0..board.x_size {
      locs.push(Location::get_loc(x, y, board.x_size));
    }
  }
  let mut t = PatternBonusTable::new_num_shards(11);
  {
    let mut text = String::new();
    let mut nonzero = 0;
    for pla in [P_BLACK, P_WHITE] {
      for &loc in &locs {
        let h: Hash128 = t.get_hash(pla, loc, board);
        let _ = writeln!(text, "{} {} {:016X} {:016X}", pla, loc, h.hash1, h.hash0);
        if h.hash0 != 0 || h.hash1 != 0 {
          nonzero += 1;
        }
      }
    }
    coutln!("pbh hashes n={} nonzero={} fnv={:016X}", text.len(), nonzero, fnv64(&text));
  }
  let mut seen: BTreeSet<Hash128> = BTreeSet::new();
  let mut added = 0;
  'outer: for y in 0..board.y_size {
    for x in 0..board.x_size {
      if added >= num_add {
        break 'outer;
      }
      let loc = Location::get_loc(x, y, board.x_size);
      if board.colors[loc as usize] != C_EMPTY {
        continue;
      }
      let pla = if added % 2 == 0 { P_BLACK } else { P_WHITE };
      for flip in 0..2 {
        for sym in 0..8 {
          t.add_bonus(pla, loc, board, 0.25 * ((added + 1) as f64) * (if flip != 0 { -1.0 } else { 1.0 }), sym, flip != 0, &mut seen);
        }
      }
      added += 1;
    }
  }
  cout!("pbh added={} seen={} ", added, seen.len());
  print_pattern_bonus_table("table", &t);
  {
    let mut text = String::new();
    let mut nonzero = 0;
    for pla in [P_BLACK, P_WHITE] {
      for &loc in &locs {
        let e = t.get_pla_move_loc_board(pla, loc, board);
        let _ = writeln!(text, "{} {} {:016X}", pla, loc, e.utility_bonus.to_bits());
        if e.utility_bonus != 0.0 {
          nonzero += 1;
        }
      }
    }
    coutln!("pbh gets n={} nonzero={} fnv={:016X}", text.len(), nonzero, fnv64(&text));
  }
  let mut t2 = PatternBonusTable::new_num_shards(11);
  t2.add_bonus_for_game_moves(hist, 0.5);
  print_pattern_bonus_table("game", &t2);
  let mut t3 = PatternBonusTable::new_num_shards(7);
  t3.add_bonus_for_game_moves_only_pla(hist, -0.3, P_WHITE);
  print_pattern_bonus_table("gameW", &t3);
  let mut t4 = PatternBonusTable::new();
  t4.add_bonus_for_game_moves_only_pla(hist, 0.2, P_BLACK);
  print_pattern_bonus_table("gameB", &t4);
}

fn parse_persp(s: &str) -> Player {
  match s {
    "W" => P_WHITE,
    "B" => P_BLACK,
    _ => C_EMPTY,
  }
}

fn print_rv(label: &str, suc: bool, v: &ReportedSearchValues) {
  cout!("RV {} ok={}", label, suc as i32);
  if suc {
    cout!(
      " win={:016X} loss={:016X} nr={:016X} ssv={:016X} dsv={:016X} es={:016X} esd={:016X} lead={:016X} wl={:016X} util={:016X} weight={:016X} visits={}",
      v.win_value.to_bits(),
      v.loss_value.to_bits(),
      v.no_result_value.to_bits(),
      v.static_score_value.to_bits(),
      v.dynamic_score_value.to_bits(),
      v.expected_score.to_bits(),
      v.expected_score_stdev.to_bits(),
      v.lead.to_bits(),
      v.win_loss_value.to_bits(),
      v.utility.to_bits(),
      v.weight.to_bits(),
      v.visits
    );
  }
  coutln!();
}

/// The 4 lines of one AnalysisData entry.
fn ad_text(i: usize, d: &AnalysisData, board: &Board) -> String {
  let mut s = String::new();
  let _ = write!(
    s,
    "a {} {} loc={} visits={} psv={:016X} lcb={:016X} radius={:016X} util={:016X} resUtil={:016X} scoreUtil={:016X} wl={:016X} nr={:016X} prior={:016X} sm={:016X} ss={:016X} lead={:016X} ess={:016X} wf={:016X} ws={:016X} wsq={:016X} usq={:016X} smsq={:016X} cv={} cws={:016X} order={} symOf={} sym={} node={}\n",
    i,
    Location::to_string(d.move_, board),
    d.move_,
    d.num_visits,
    d.play_selection_value.to_bits(),
    d.lcb.to_bits(),
    d.radius.to_bits(),
    d.utility.to_bits(),
    d.result_utility.to_bits(),
    d.score_utility.to_bits(),
    d.win_loss_value.to_bits(),
    d.no_result_value.to_bits(),
    d.policy_prior.to_bits(),
    d.score_mean.to_bits(),
    d.score_stdev.to_bits(),
    d.lead.to_bits(),
    d.ess.to_bits(),
    d.weight_factor.to_bits(),
    d.weight_sum.to_bits(),
    d.weight_sq_sum.to_bits(),
    d.utility_sq_avg.to_bits(),
    d.score_mean_sq_avg.to_bits(),
    d.child_visits,
    d.child_weight_sum.to_bits(),
    d.order,
    d.is_symmetry_of,
    d.symmetry,
    d.node.is_some() as i32
  );
  s.push_str("pv");
  for l in &d.pv {
    let _ = write!(s, " {}", l);
  }
  s.push_str("\npvv");
  for v in &d.pv_visits {
    let _ = write!(s, " {}", v);
  }
  s.push_str("\npvev");
  for v in &d.pv_edge_visits {
    let _ = write!(s, " {}", v);
  }
  s.push('\n');
  s
}

const AD_FULL: usize = 2;

fn print_ad(buf: &[AnalysisData], board: &Board) {
  coutln!("AD n={}", buf.len());
  let mut rest = String::new();
  for i in 0..buf.len() {
    let t = ad_text(i, &buf[i], board);
    if i < AD_FULL {
      cout!("{}", t);
    } else {
      rest.push_str(&t);
    }
  }
  if buf.len() > AD_FULL {
    coutln!("adrest n={} fnv={:016X}", buf.len() - AD_FULL, fnv64(&rest));
  }
}

fn print_long(s: &str) {
  if s.len() <= 1500 {
    coutln!("{}", s);
    return;
  }
  coutln!("{}", &s[..800]);
  coutln!("...[len={} fnv={:016X}]", s.len(), fnv64(s));
}

//------------------------------------------------------------------------------------------------
// The op interpreter

struct Runner<'a> {
  spec: &'a Spec,
  ctx: &'a mut Ctx,
  search: Box<Search>,
  cur_params: SearchParams,
  owner_on: bool,
  /// Op HC (before P): setAssumeMultipleStartingBlackMovesAreHandicap / setOverrideNumHandicapStones
  hc_assume: bool,
  hc_override: i32,
  /// Op ECS: a second Search (same evaluator) sharing the first one's eval cache
  nn_eval: Rc<NNEvaluator>,
  search2: Option<Box<Search>>,
}

fn boxed_history(board: &Board, pla: Player, rules: &Rules) -> Box<BoardHistory> {
  Box::new(BoardHistory::new_full(board, pla, rules, 0))
}

impl<'a> Runner<'a> {
  fn exec(&mut self, op: &str) -> Result<(), KataError> {
    coutln!("> {}", op);
    let t: Vec<&str> = op.split(':').collect();
    let int = |k: usize| -> i32 { t[k].parse().unwrap_or(0) };
    match t[0] {
      "P" => {
        let spec = self.spec;
        let mut board = Board::new_x_y(spec.xs, spec.ys)?;
        let mut hist = boxed_history(&board, spec.initial_pla, &spec.rules);
        if self.hc_assume {
          hist.set_assume_multiple_starting_black_moves_are_handicap(true);
        }
        if self.hc_override >= 0 {
          hist.set_override_num_handicap_stones(self.hc_override);
        }
        for (loc, pla) in &spec.moves {
          hist.make_board_move_assume_legal(&mut board, *loc, *pla, None, false);
        }
        self.search.set_position(spec.pla, &board, &hist)?;
        Board::print_board(&mut Cout, &board, Board::NULL_LOC, Some(&hist.move_history));
        coutln!(
          "encore={} finished={} next={} whb={:08X} moves={}",
          hist.encore_phase,
          hist.is_game_finished as i32,
          hist.presumed_next_move_pla,
          hist.white_handicap_bonus_score.to_bits(),
          hist.move_history.len()
        );
      }
      "V" => {
        let spec = |s: &str| -> Vec<i32> {
          if s == "-" {
            return Vec::new();
          }
          let mut v = vec![0; Board::MAX_ARR_SIZE as usize];
          for e in s.split(',') {
            let (l, u) = e.split_once('/').unwrap();
            v[l.parse::<usize>().unwrap()] = u.parse().unwrap();
          }
          v
        };
        let b = spec(t[1]);
        let w = spec(t[2]);
        self.search.set_avoid_move_until_by_loc(&b, &w);
      }
      "AR" => self.search.set_avoid_move_until_rescale_root(t[1] == "1"),
      "H" => self.search.set_root_hint_loc(int(1) as Loc),
      "O" => {
        self.search.set_always_include_owner_map(true);
        self.owner_on = true;
      }
      "Y" => {
        let mut v: Vec<i32> = Vec::new();
        if t[1] != "-" {
          for e in t[1].split(',') {
            v.push(e.parse().unwrap());
          }
        }
        self.search.set_root_symmetry_pruning_only(&v);
      }
      "R" => {
        let pla = int(1) as Player;
        run_or_stall_panic(&mut self.ctx.pool, "runWholeSearch", self.search.run_whole_search(pla))?;
        coutln!("playouts={}", self.search.last_search_num_playouts);
      }
      "B" => {
        run_or_stall_panic(&mut self.ctx.pool, "beginSearch", self.search.begin_search(false))?;
      }
      "RS" => self.dump_root(),
      "PT" => {
        let root = self.search.get_root_node();
        if root.is_none() {
          coutln!("noroot");
          return Ok(());
        }
        let mut o = PrintTreeOptions::new().max_depth(int(1));
        let fl = int(3);
        if fl & 1 != 0 {
          o = o.print_sqs(true);
        }
        if fl & 2 != 0 {
          o = o.print_avg_shortterm_error(true);
        }
        if fl & 4 != 0 {
          o = o.min_visits_prop_to_expand(0.1);
        }
        if fl & 8 != 0 {
          o = o.max_children_to_show(5);
        }
        self.search.print_tree(&mut Cout, root.as_deref(), &o, parse_persp(t[2]));
      }
      "PV" => {
        let root = self.search.get_root_node();
        if root.is_none() {
          coutln!("noroot");
          return Ok(());
        }
        self.search.print_pv(&mut Cout, root.as_deref(), int(1));
        coutln!();
      }
      "AD" => {
        let mut buf: Vec<AnalysisData> = Vec::new();
        self.search.get_analysis_data(&mut buf, int(1), t[2] == "1", 10, t[3] == "1");
        print_ad(&buf, self.search.get_root_board());
      }
      "J" => {
        let bits: Vec<bool> = t[3].bytes().map(|c| c == b'1').collect();
        let mut json = serde_json::Value::Null;
        let suc = self.search.get_analysis_json(
          parse_persp(t[1]),
          int(2),
          bits[0],
          bits[1],
          bits[2],
          bits[3],
          bits[4],
          bits[5],
          bits[6],
          bits[7],
          &mut json,
        )?;
        coutln!("success={}", suc as i32);
        print_long(&nlohmann::dump(&json));
      }
      "HS" => self.hash_summary(&t),
      "CM" => {
        for _ in 0..int(1) {
          let l = self.search.get_chosen_move_loc();
          coutln!("cm {} {}", l, Location::to_string(l, self.search.get_root_board()));
        }
      }
      "PSV" => {
        let mut locs: Vec<Loc> = Vec::new();
        let mut vals: Vec<f64> = Vec::new();
        let mut visits: Vec<f64> = Vec::new();
        let suc = self.search.get_play_selection_values_ret_visit_counts(&mut locs, &mut vals, Some(&mut visits), 1.0);
        coutln!("success={} n={} nv={}", suc as i32, locs.len(), visits.len());
        if suc {
          let mut rest = String::new();
          for i in 0..locs.len() {
            let mut line = format!("s {} {} {:016X}", locs[i], Location::to_string(locs[i], self.search.get_root_board()), vals[i].to_bits());
            if i < visits.len() {
              let _ = writeln!(line, " {:016X}", visits[i].to_bits());
            } else {
              line.push_str(" -\n");
            }
            if i < 12 {
              cout!("{}", line);
            } else {
              rest.push_str(&line);
            }
          }
          if locs.len() > 12 {
            coutln!("srest n={} fnv={:016X}", locs.len() - 12, fnv64(&rest));
          }
        }
      }
      "OW" => {
        if self.search.root_node.is_none() || !self.owner_on {
          coutln!("noroot");
          return Ok(());
        }
        let avg = self.search.get_average_tree_ownership(None)?;
        cout!("ow n={}", avg.len());
        for x in &avg {
          cout!(" {:016X}", x.to_bits());
        }
        coutln!();
        let (first, second) = self.search.get_average_and_standard_deviation_tree_ownership(None);
        let mut avg_text = String::new();
        for x in &first {
          let _ = write!(avg_text, " {:016X}", x.to_bits());
        }
        cout!("ows n={} fnv={:016X}\nstd", first.len(), fnv64(&avg_text));
        for x in &second {
          cout!(" {:016X}", x.to_bits());
        }
        coutln!();
      }
      "PM" => {
        self.search.print_root_policy_map(&mut Cout);
        self.search.print_root_ownership_map(&mut Cout, P_WHITE);
        self.search.print_root_ending_score_value_bonus(&mut Cout);
      }
      "M" => {
        let ok = self.search.make_move_prevent_encore(int(1) as Loc, int(2) as Player, t[3] == "1")?;
        coutln!("made={} rootPla={}", ok as i32, self.search.get_root_pla());
      }
      "SPN" | "SP" => {
        let mut np = self.cur_params.clone();
        for kv in t[1].split(',') {
          let (k, v) = kv.split_once('=').unwrap();
          assert!(set_param(&mut np, k, v), "bad param {}", kv);
        }
        self.cur_params = np.clone();
        if t[0] == "SPN" {
          self.search.set_params_no_clearing(&np);
        } else {
          self.search.set_params(&np);
        }
      }
      "XO" => {
        // upstream mirror: op XO (search_nonn.cpp): the root's nnOutput becomes a copy whose whiteOwnerMap is
        // scaled by 6 and clamped to [-0.999, 0.999] (f32 math).
        let root: Option<Rc<SearchNode>> = self.search.get_root_node();
        let old: Option<Rc<NNOutput>> = root.as_ref().and_then(|r| r.get_nn_output());
        let (root, old) = match (root, old) {
          (Some(r), Some(o)) if o.white_owner_map.is_some() => (r, o),
          _ => {
            coutln!("noxo");
            return Ok(());
          }
        };
        let mut copy: NNOutput = (*old).clone();
        let n = (copy.nn_x_len * copy.nn_y_len) as usize;
        let mut num_extreme = 0;
        let map = copy.white_owner_map.as_mut().expect("checked above");
        for v in map.iter_mut().take(n) {
          let mut x: f32 = *v * 6.0f32;
          if x > 0.999f32 {
            x = 0.999f32;
          }
          if x < -0.999f32 {
            x = -0.999f32;
          }
          *v = x;
          if x >= 0.95f32 || x <= -0.95f32 {
            num_extreme += 1;
          }
        }
        *root.nn_output.borrow_mut() = Some(Rc::new(copy));
        coutln!("xo extreme={} of {}", num_extreme, n);
      }
      "ECT" => {
        if t[1] != "-" {
          self.search.effective_search_time_carried_over = f64::from_bits(u64::from_str_radix(t[1], 16).unwrap());
        }
        coutln!("ect {:016X}", self.search.effective_search_time_carried_over.to_bits());
      }
      "HC" => {
        self.hc_assume = t[1] == "1";
        self.hc_override = int(2);
        coutln!("hc assume={} override={}", self.hc_assume as i32, self.hc_override);
      }
      "EC" => print_eval_cache(&self.search),
      "NE" => print_node_extras(&self.search),
      "ECC" => {
        if let Some(ec) = &self.search.eval_cache {
          ec.clear();
        }
        coutln!("ecc {}", self.search.eval_cache.is_some() as i32);
      }
      "ECN" => {
        let shards = int(1);
        self.search.set_external_eval_cache(&Some(Rc::new(EvalCacheTable::new(shards))));
        coutln!("ecn {}", shards);
      }
      "ECS" => {
        // upstream mirror: op ECS (search_nonn.cpp)
        if self.search.eval_cache.is_none() {
          coutln!("noec");
          return Ok(());
        }
        self.search2 = None;
        let mut s2 = Box::new(Search::new(&self.cur_params, self.nn_eval.clone(), &format!("{}second", self.spec.search_seed)));
        s2.set_external_eval_cache(&self.search.eval_cache);
        s2.set_position(self.search.get_root_pla(), self.search.get_root_board(), self.search.get_root_hist())?;
        run_or_stall_panic(&mut self.ctx.pool, "runWholeSearch", s2.run_whole_search(int(1) as Player))?;
        coutln!("playouts2={}", s2.last_search_num_playouts);
        Self::dump_root_of(&s2);
        print_node_extras(&s2);
        self.search2 = Some(s2);
      }
      "PB" => {
        // upstream mirror: op PB (search_nonn.cpp): PB:<0 external|1 copy>:<shards, 0 = default>:<bonus>:<onlyPla>:<moves replayed, -1 = all>
        let spec = self.spec;
        let mut board = Board::new_x_y(spec.xs, spec.ys)?;
        let mut hist = boxed_history(&board, spec.initial_pla, &spec.rules);
        let n = int(5);
        for (i, (loc, pla)) in spec.moves.iter().enumerate() {
          if n >= 0 && i as i32 >= n {
            break;
          }
          hist.make_board_move_assume_legal(&mut board, *loc, *pla, None, false);
        }
        let shards = int(2);
        let mut table = if shards > 0 { PatternBonusTable::new_num_shards(shards) } else { PatternBonusTable::new() };
        table.add_bonus_for_game_moves_only_pla(&hist, t[3].parse::<f64>().unwrap_or(0.0), int(4) as Player);
        print_pattern_bonus_table("pb", &table);
        if t[1] == "0" {
          self.search.set_external_pattern_bonus_table(Some(Box::new(table)));
        } else {
          self.search.set_copy_of_external_pattern_bonus_table(&Some(Box::new(table)));
        }
      }
      "PBN" => {
        if t[1] == "0" {
          self.search.set_external_pattern_bonus_table(None);
        } else {
          self.search.set_copy_of_external_pattern_bonus_table(&None);
        }
        coutln!("pbn {}", t[1]);
      }
      "PBH" => pattern_bonus_hash_op(&self.search, int(1)),
      "MI" => coutln!(
        "mirror pla={} adv={:016X} err={:016X}",
        self.search.mirroring_pla,
        self.search.mirror_advantage.to_bits(),
        self.search.mirror_center_symmetry_error.to_bits()
      ),
      "CL" => self.search.clear_search(),
      "KOMI" => self.search.set_komi_if_new(f32::from_bits(u32::from_str_radix(t[1], 16).unwrap())),
      "PIN" => self.search.set_player_if_new(int(1) as Player)?,
      "PCH" => self.search.set_player_and_clear_history(int(1) as Player)?,
      other => panic!("unknown op {}", other),
    }
    Ok(())
  }

  fn print_hash(label: &str, text: &str) {
    coutln!("{} len={} fnv={:016X}", label, text.len(), fnv64(text));
  }

  /// upstream mirror: Runner::hashSummary (search_nonn.cpp)
  fn hash_summary(&mut self, t: &[&str]) {
    let search = &*self.search;
    coutln!("hs root visits={} lastPlayouts={}", search.get_root_visits(), search.last_search_num_playouts);
    let mut v = ReportedSearchValues::new();
    let mut suc = search.get_root_values(&mut v);
    print_rv("root", suc, &v);
    suc = search.get_pruned_root_values(&mut v);
    print_rv("pruned", suc, &v);
    let root: Option<Rc<SearchNode>> = search.get_root_node();
    let root = match root {
      None => {
        coutln!("noroot");
        return;
      }
      Some(r) => r,
    };
    let board = search.get_root_board();
    {
      let mut text = String::new();
      let children = root.get_children();
      let cap = children.get_capacity();
      let mut n = 0;
      for i in 0..cap {
        let cp = children.index(i);
        let child_idx = match cp.get_if_allocated() {
          None => break,
          Some(c) => c,
        };
        let child = search.node_arena.get(child_idx);
        let mv = cp.get_move_loc();
        let _ = writeln!(
          text,
          "c {} {} loc={} edge={} visits={} ws={:016X} util={:016X}",
          i,
          Location::to_string(mv, board),
          mv,
          cp.get_edge_visits(),
          child.stats.visits.get(),
          child.stats.weight_sum.get().to_bits(),
          child.stats.utility_avg.get().to_bits()
        );
        n += 1;
      }
      cout!("children={} ", n);
      Self::print_hash("childtext", &text);
    }
    {
      let mut tree = String::new();
      let o = PrintTreeOptions::new().max_depth(t[1].parse().unwrap_or(0));
      search.print_tree(&mut tree, Some(&*root), &o, parse_persp(t[2]));
      Self::print_hash("tree", &tree);
    }
    {
      let mut buf: Vec<AnalysisData> = Vec::new();
      search.get_analysis_data(&mut buf, 3, true, 10, true);
      let mut text = String::new();
      for i in 0..buf.len() {
        text.push_str(&ad_text(i, &buf[i], board));
      }
      cout!("ad n={} ", buf.len());
      Self::print_hash("adtext", &text);
    }
    {
      let bits: Vec<bool> = t[3].bytes().map(|c| c == b'1').collect();
      let mut json = serde_json::Value::Null;
      let jsuc = search
        .get_analysis_json(parse_persp(t[2]), 10, bits[0], bits[1], bits[2], bits[3], bits[4], bits[5], bits[6], bits[7], &mut json)
        .unwrap_or(false);
      cout!("json success={} ", jsuc as i32);
      Self::print_hash("jsontext", &nlohmann::dump(&json));
    }
    {
      let mut locs: Vec<Loc> = Vec::new();
      let mut vals: Vec<f64> = Vec::new();
      let mut visits: Vec<f64> = Vec::new();
      let psuc = search.get_play_selection_values_ret_visit_counts(&mut locs, &mut vals, Some(&mut visits), 1.0);
      let mut text = String::new();
      for i in 0..locs.len() {
        let vb = if i < visits.len() { visits[i].to_bits() } else { 0 };
        let _ = writeln!(text, "s {} {:016X} {:016X}", locs[i], vals[i].to_bits(), vb);
      }
      cout!("psv success={} n={} nv={} ", psuc as i32, locs.len(), visits.len());
      Self::print_hash("psvtext", &text);
    }
    let table_size: usize = search.node_table.entries.borrow().iter().map(|m| m.len()).sum();
    coutln!("nodetable={} postorder={}", table_size, search.enumerate_tree_post_order().len());
    print_node_table_keys(search);
  }

  fn dump_root(&mut self) {
    Self::dump_root_of(&self.search);
  }

  /// upstream mirror: Runner::dumpRootOf (search_nonn.cpp)
  fn dump_root_of(search: &Search) {
    coutln!("root visits={} lastPlayouts={}", search.get_root_visits(), search.last_search_num_playouts);
    let root: Option<Rc<SearchNode>> = search.get_root_node();
    if let Some(root) = &root {
      let children = root.get_children();
      let cap = children.get_capacity();
      let mut n: i32 = 0;
      let mut rest = String::new();
      const RS_FULL: i32 = 5;
      for i in 0..cap {
        let cp = children.index(i);
        let child_idx = match cp.get_if_allocated() {
          None => break,
          Some(c) => c,
        };
        let child = search.node_arena.get(child_idx);
        let mv = cp.get_move_loc();
        let line = format!(
          "c {} {} loc={} edge={} visits={} ws={:016X} util={:016X}\n",
          i,
          Location::to_string(mv, search.get_root_board()),
          mv,
          cp.get_edge_visits(),
          child.stats.visits.get(),
          child.stats.weight_sum.get().to_bits(),
          child.stats.utility_avg.get().to_bits()
        );
        if n < RS_FULL {
          cout!("{}", line);
        } else {
          rest.push_str(&line);
        }
        n += 1;
      }
      if n > RS_FULL {
        coutln!("crest n={} fnv={:016X}", n - RS_FULL, fnv64(&rest));
      }
      coutln!("children={}", n);
    }
    let mut v = ReportedSearchValues::new();
    let mut suc = search.get_root_values(&mut v);
    print_rv("root", suc, &v);
    suc = search.get_pruned_root_values(&mut v);
    print_rv("pruned", suc, &v);
    suc = search.get_root_raw_nn_values(&mut v);
    print_rv("raw", suc, &v);
    let (mut surprise, mut s_ent, mut p_ent) = (0.0f64, 0.0f64, 0.0f64);
    suc = search.get_policy_surprise_and_entropy(&mut surprise, &mut s_ent, &mut p_ent);
    coutln!("surprise ok={} {:016X} {:016X} {:016X}", suc as i32, surprise.to_bits(), s_ent.to_bits(), p_ent.to_bits());
    if root.is_some() {
      let mut sharp = 0.0f64;
      suc = search.get_sharp_score(root.as_deref(), &mut sharp);
      coutln!("sharp ok={} {:016X}", suc as i32, sharp.to_bits());
      let st = search.get_shallow_average_shortterm_wl_and_score_error(root.as_deref());
      coutln!("shallow {:016X} {:016X}", st.0.to_bits(), st.1.to_bits());
      let table_size: usize = search.node_table.entries.borrow().iter().map(|m| m.len()).sum();
      coutln!("nodetable={} postorder={}", table_size, search.enumerate_tree_post_order().len());
      print_node_table_keys(search);
    }
  }
}

//------------------------------------------------------------------------------------------------
// Running one case

/// Optional knobs for the Rust-only multi-thread test.
struct Variant<'v> {
  /// Replaces the spec's op stream.
  ops: Option<&'v [String]>,
  /// `numSearchThreads`
  num_threads: Option<i32>,
  /// key=value overrides applied after the spec's params (and before maxVisits).
  overrides: &'v [(&'v str, &'v str)],
  /// Called with the finished search before it is dropped.
  check: Option<&'v mut dyn FnMut(&Search)>,
}

fn no_variant<'v>() -> Variant<'v> {
  Variant { ops: None, num_threads: None, overrides: &[], check: None }
}

/// Runs the case's op stream; returns the text of the block: the CASE line, what the ops printed, "END <n>".
fn run_case(spec: &Spec, mut variant: Variant) -> String {
  OUT.with(|o| o.borrow_mut().clear());
  let mut ctx = Ctx::new();
  let nn_eval = start_nn_eval(&mut ctx, &spec.nn_seed, spec.nn_x, spec.nn_y);
  let mut params = base_params(spec.base);
  params.num_threads = 1;
  for (k, v) in &spec.params {
    assert!(set_param(&mut params, k, v), "bad param {}", k);
  }
  for (k, v) in variant.overrides {
    assert!(set_param(&mut params, k, v), "bad override {}", k);
  }
  params.max_visits = spec.max_visits;
  if let Some(t) = variant.num_threads {
    params.num_threads = t;
  }
  let search = Box::new(Search::new(&params, nn_eval.clone(), &spec.search_seed));
  let mut runner = Runner { spec, ctx: &mut ctx, search, cur_params: params, owner_on: false, hc_assume: false, hc_override: -1, nn_eval: nn_eval.clone(), search2: None };
  let ops: &[String] = variant.ops.unwrap_or(&spec.ops);
  for op in ops {
    if let Err(e) = runner.exec(op) {
      coutln!("EXC");
      eprintln!("case {} op {}: {:?}", spec.idx, op, e);
      break;
    }
  }
  if let Some(check) = variant.check.as_mut() {
    check(&runner.search);
  }
  let Runner { search, search2, nn_eval: runner_nn_eval, .. } = runner;
  drop(search2);
  drop(search);
  drop(runner_nn_eval);
  delete_nn_eval(&mut ctx, nn_eval);
  let out = OUT.with(|o| o.borrow().clone());
  format!("{}\n{}END {}\n", spec.line, out, spec.idx)
}

//------------------------------------------------------------------------------------------------
// The oracle test. Also on wasm32 (`just test-wasm`): `port::math::atan` is target-independent and glibc-identical on
// ScoreValue's table arguments (DECISIONS.md D13), so the bit-exact comparison holds there too. On wasm32 a panic
// cannot be caught (`try_catch`), so the first mismatching case aborts the test instead of being collected.

const NUM_CASES: usize = 536;

/// Cases of the multi-thread test taken from the fixture (spread over all generators).
const NUM_MT_CASES: usize = 36;

/// The fixture text. For exploration only, the environment variable SEARCH_ORACLE_FILE names another output of
/// search_nonn.cpp (e.g. `ORACLE_SEED=x ORACLE_STRESS_ONLY=2000 search_nonn`); then the case count is not checked.
fn fixture_text() -> (String, bool) {
  common::fixture_or_env("SEARCH_ORACLE_FILE", "oracle", "search_nonn.txt")
}

struct Mismatch {
  idx: i32,
  kind: String,
  line: String,
  msg: String,
}

/// `common::compare_lines` for one case, returning its failure message instead of unwinding.
fn compare_case(label: &str, actual: &str, expected: &str) -> Result<(), String> {
  try_catch(|| compare_lines(label, actual, expected))
}

fn run_oracle() -> (usize, Vec<Mismatch>) {
  init();
  ScoreValue::init_tables(); //upstream's main(): "MUST BE CALLED AT PROGRAM START!"
  install_logger_clock_and_hook();
  let (expected, explore) = fixture_text();
  let cases = split_cases(&expected);
  if !explore {
    assert_eq!(cases.len(), NUM_CASES, "number of cases in search_nonn.txt");
  }
  let mut mismatches: Vec<Mismatch> = Vec::new();
  let mut kinds: HashMap<String, usize> = HashMap::new();
  let mut num_ops = 0usize;
  for (line, block) in &cases {
    let spec = parse_spec(line);
    let prefix = spec.kind.split('-').next().unwrap_or("").to_string();
    *kinds.entry(prefix).or_insert(0) += 1;
    num_ops += spec.ops.len();
    let actual = try_catch(|| run_case(&spec, no_variant()));
    let status = match actual {
      Err(p) => Err(format!("PANIC: {}\n  partial output: {}", p, OUT.with(|o| o.borrow().chars().rev().take(300).collect::<Vec<_>>().into_iter().rev().collect::<String>()))),
      Ok(actual) => compare_case(&format!("case {} [{}]", spec.idx, spec.kind), &actual, block),
    };
    if let Err(msg) = status {
      mismatches.push(Mismatch { idx: spec.idx, kind: spec.kind.clone(), line: spec.line.clone(), msg });
    }
  }
  let mut kv: Vec<_> = kinds.into_iter().collect();
  kv.sort();
  crate::report!("search_oracle: {} cases ({} ops) by generator: {:?}", cases.len(), num_ops, kv);
  (cases.len(), mismatches)
}

fn run_oracle_report() {
  let mut cases = Cases::new("search_oracle::search_oracle");
  let (num_cases, mismatches) = run_oracle_on_big_stack();
  for _ in 0..num_cases - mismatches.len() {
    cases.ran();
  }
  crate::report!("search_oracle: {} cases match, {} MISMATCH", num_cases - mismatches.len(), mismatches.len());
  if !mismatches.is_empty() {
    let mut s = String::new();
    for m in &mismatches {
      let _ = writeln!(s, "--- case {} kind={}\n{}\n{}", m.idx, m.kind, m.msg.chars().take(1500).collect::<String>(), m.line.chars().take(1200).collect::<String>());
    }
    crate::report!("{}", s);
    panic!("{} of {} cases differ from the upstream output", mismatches.len(), num_cases);
  }
}

#[cfg(not(target_arch = "wasm32"))]
fn run_oracle_on_big_stack() -> (usize, Vec<Mismatch>) {
  //Big stack: Search (~40 KB), BoardHistory (~34 KB) and the recursive playout futures, in a debug build.
  std::thread::Builder::new()
    .name("search_oracle".to_string())
    .stack_size(256 << 20)
    .spawn(run_oracle)
    .unwrap()
    .join()
    .expect("run_oracle panicked outside the per-case catch")
}

#[cfg(target_arch = "wasm32")]
fn run_oracle_on_big_stack() -> (usize, Vec<Mismatch>) {
  //No threads on wasm32: the wasm stack size is a link-time setting (`just test-wasm`).
  run_oracle()
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn search_oracle() {
  run_oracle_report();
}

//------------------------------------------------------------------------------------------------
// Rust-only: numSearchThreads > 1 (virtual threads)

/// Enumerates the tree and checks it like `TestSearchCommon::verifyTreePostOrder`: every node is hit exactly
/// once, children before parents, the root last. Returns the nodes.
fn check_post_order(search: &Search, errs: &mut Vec<String>) -> Vec<NodeIdx> {
  let nodes: Vec<NodeIdx> = search.enumerate_tree_post_order();
  let mut idx_of_node: HashMap<NodeIdx, usize> = HashMap::new();
  for i in 0..nodes.len() {
    if idx_of_node.insert(nodes[i], i).is_some() {
      errs.push(format!("post order: node {} enumerated twice", i));
    }
  }
  for i in 0..nodes.len() {
    let node: Rc<SearchNode> = search.node_arena.get(nodes[i]);
    let children = node.get_children();
    for j in 0..children.get_capacity() {
      let child = match children.index(j).get_if_allocated() {
        None => break,
        Some(c) => c,
      };
      match idx_of_node.get(&child) {
        None => errs.push(format!("post order: child of node {} missing from the enumeration", i)),
        Some(&ci) => {
          if ci >= i {
            errs.push(format!("post order: child {} not before its parent {}", ci, i));
          }
        }
      }
    }
  }
  if nodes.last().copied() != search.root_node {
    errs.push("post order: the root is not last".to_string());
  }
  nodes
}

/// Invariants of a finished search from an empty tree (see `multithread_virtual_threads`).
/// `Mode::Tree`: tree-shaped search with no hacks/avoided moves, where every counted playout adds exactly one
/// root visit and one root edge visit (except the very first, which evaluates the root).
fn check_invariants(search: &Search, threads: i32, max_visits: i64, mode: Mode) -> Vec<String> {
  let strict_edges = mode == Mode::Tree;
  let counted_playouts = mode != Mode::Raw;
  let mut errs: Vec<String> = Vec::new();
  let root: Rc<SearchNode> = match search.get_root_node() {
    None => return vec!["no root node after the search".to_string()],
    Some(r) => r,
  };
  let root_visits = root.stats.visits.get();
  let playouts = search.last_search_num_playouts;
  //(1) The stop test `numPlayouts + numNonPlayoutVisits >= maxVisits` reads a thread-local snapshot of the shared
  //    counter, taken after that thread's own last finished playout. The thread that stops has a snapshot >= maxVisits,
  //    so at least maxVisits playouts finished; each of the other threads-1 can have one playout in flight (started
  //    from a stale snapshot) that still finishes afterwards: playouts <= maxVisits + threads - 1.
  if playouts < max_visits || playouts > max_visits + (threads as i64) - 1 {
    errs.push(format!("playouts {} not in [{}, {}]", playouts, max_visits, max_visits + threads as i64 - 1));
  }
  //(2) Fresh tree (numNonPlayoutVisits == 0): every counted playout updates the root's stats exactly once
  //    (updateStatsAfterPlayout / the first evaluation's leaf value), so root visits == finished playouts.
  //    (Raw mode runs the case's own params, with weightless playouts from the passing hacks: those count as playouts
  //    but add no root visit, so there only root visits <= playouts holds.)
  if (counted_playouts && root_visits != playouts) || root_visits > playouts {
    errs.push(format!("root visits {} vs lastSearchNumPlayouts {}", root_visits, playouts));
  }
  //(3) Σ root edge visits == root visits - 1: the first playout evaluates the root (a root visit, no edge); each other
  //    counted playout descends one root edge and does `addEdgeVisits(1)` + `updateStatsAfterPlayout(root)` (or the
  //    catch-up / cycle variants that do both), never one without the other.
  let children = root.get_children();
  let mut sum_edges: i64 = 0;
  let mut num_children = 0;
  for j in 0..children.get_capacity() {
    let cp = children.index(j);
    if cp.get_if_allocated().is_none() {
      break;
    }
    sum_edges += cp.get_edge_visits();
    num_children += 1;
  }
  if strict_edges && sum_edges != root_visits - 1 {
    errs.push(format!("sum of root edge visits {} != root visits {} - 1", sum_edges, root_visits));
  }
  if !strict_edges && sum_edges > root_visits - 1 {
    errs.push(format!("sum of root edge visits {} > root visits {} - 1", sum_edges, root_visits));
  }
  //(4) post order, virtual losses and node states.
  let nodes = check_post_order(search, &mut errs);
  for (i, &n) in nodes.iter().enumerate() {
    let node = search.node_arena.get(n);
    //Every playout, finished or abandoned, adds a virtual loss on its way down and takes it back on its way up.
    if node.virtual_losses.get() != 0 {
      errs.push(format!("node {} has virtual_losses {}", i, node.virtual_losses.get()));
    }
    //Every in-flight playout has completed: no node is left mid-transition.
    let st = node.state.get();
    if st == SearchNode::STATE_EVALUATING || st == SearchNode::STATE_GROWING1 || st == SearchNode::STATE_GROWING2 {
      errs.push(format!("node {} left in transient state {}", i, st));
    }
  }
  //(5) Node table: every node is inserted by allocateOrFindNode (tree search under a random hash, graph search under its
  //    graph hash) except the root, and the port has no await between allocation and linking the child pointer, so no
  //    allocated node is orphaned: the table holds exactly the reachable non-root nodes.
  let in_table: HashSet<NodeIdx> = search.node_table.entries.borrow().iter().flat_map(|m| m.values().copied()).collect();
  let table_size: usize = search.node_table.entries.borrow().iter().map(|m| m.len()).sum();
  if table_size != nodes.len() - 1 || in_table.len() != table_size {
    errs.push(format!("node table has {} entries ({} distinct nodes) but the tree has {} non-root nodes", table_size, in_table.len(), nodes.len() - 1));
  }
  for &n in &nodes {
    if Some(n) != search.root_node && !in_table.contains(&n) {
      errs.push("a reachable non-root node is not in the node table".to_string());
      break;
    }
  }
  //In a tree search every non-root node has exactly one parent edge, so #nodes == 1 + #child pointers.
  if strict_edges {
    let mut pointers = 0usize;
    for &n in &nodes {
      let node = search.node_arena.get(n);
      let ch = node.get_children();
      for j in 0..ch.get_capacity() {
        if ch.index(j).get_if_allocated().is_none() {
          break;
        }
        pointers += 1;
      }
    }
    if nodes.len() != 1 + pointers {
      errs.push(format!("tree search: {} nodes but {} child pointers", nodes.len(), pointers));
    }
  }
  let _ = num_children;
  errs
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Mode {
  /// useGraphSearch off, passing hacks off
  Tree,
  /// useGraphSearch on, passing hacks off
  Graph,
  /// the case's own params (hacks, graph search or not as the case says)
  Raw,
}

/// Wall-clock for the timing lines of the report only (`Instant` panics on wasm32-unknown-unknown).
struct Stopwatch(#[cfg(not(target_arch = "wasm32"))] std::time::Instant);

impl Stopwatch {
  fn new() -> Stopwatch {
    #[cfg(not(target_arch = "wasm32"))]
    {
      Stopwatch(std::time::Instant::now())
    }
    #[cfg(target_arch = "wasm32")]
    {
      Stopwatch()
    }
  }
  fn elapsed(&self) -> String {
    #[cfg(not(target_arch = "wasm32"))]
    {
      format!("{:?}", self.0.elapsed())
    }
    #[cfg(target_arch = "wasm32")]
    {
      "n/a".to_string()
    }
  }
}

fn run_multithread() -> String {
  init();
  ScoreValue::init_tables();
  install_logger_clock_and_hook();
  let started = Stopwatch::new();
  let expected = fixture("oracle", "search_nonn.txt");
  let all: Vec<Spec> = split_cases(&expected).iter().map(|(l, _)| parse_spec(l)).collect();
  //A subset: small boards and moderate visit counts, spread over the fixture (every kind of generator).
  let eligible: Vec<&Spec> = all.iter().filter(|s| s.xs * s.ys <= 100 && s.max_visits >= 20 && s.max_visits <= 300).collect();
  assert!(eligible.len() >= 100, "{} eligible cases", eligible.len());
  let step = eligible.len() / NUM_MT_CASES;
  let subset: Vec<&Spec> = (0..NUM_MT_CASES).map(|k| eligible[k * step]).collect();
  let mut summary = String::new();
  let mut runs = 0;
  let mut all_errs: Vec<String> = Vec::new();
  //Evidence that the virtual threads really interleave (otherwise the invariants would be vacuous):
  let mut differs_from_single = 0; // runs whose dump differs from the 1-thread run of the same case
  let mut overshoot = 0; // runs that finished more than maxVisits playouts (stale stop snapshots)
  //(case, [(threads, mode)]): the small cases in all modes; the wide ones (nodes with more than 64 children: the third
  //children array) once or twice.
  const ALL_MODES: [(i32, Mode); 6] = [(2, Mode::Tree), (4, Mode::Tree), (16, Mode::Tree), (4, Mode::Graph), (4, Mode::Raw), (16, Mode::Raw)];
  let mut jobs: Vec<(&Spec, Vec<(i32, Mode)>)> = subset.into_iter().map(|s| (s, ALL_MODES.to_vec())).collect();
  let wide: Vec<&Spec> = all.iter().filter(|s| s.kind.starts_with("wide-") && s.xs <= 15).collect();
  assert_eq!(wide.len(), 2, "wide cases");
  jobs.push((wide[0], vec![(16, Mode::Tree)]));
  jobs.push((wide[1], vec![(4, Mode::Graph), (16, Mode::Raw)]));
  for (spec, combos) in jobs {
    //Only the position and the search: P, a search for the root player, and a battery of dumps.
    let ops: Vec<String> = vec![
      "P".to_string(),
      format!("R:{}", spec.pla),
      "RS".to_string(),
      "PT:2:W:0".to_string(),
      "AD:0:1:1".to_string(),
      "J:W:5:01000010".to_string(),
      "CM:3".to_string(),
      "PSV".to_string(),
    ];
    //(threads, tree-shaped?) A tree search without hacks has strict counting invariants; the graph search run checks the rest.
    for (threads, mode) in combos {
      //The caps must be exactly maxVisits for the bounds below: no maxPlayouts, no search factor after passes.
      let mut overrides: Vec<(&str, &str)> = vec![("maxPlayouts", "1125899906842624"), ("searchFactorAfterOnePass", "1.0"), ("searchFactorAfterTwoPass", "1.0")];
      match mode {
        Mode::Tree => overrides.extend([("useGraphSearch", "0"), ("enablePassingHacks", "0"), ("enableMorePassingHacks", "0")]),
        Mode::Graph => overrides.extend([("useGraphSearch", "1"), ("enablePassingHacks", "0"), ("enableMorePassingHacks", "0")]),
        Mode::Raw => {}
      }
      let mut errs: Vec<String> = Vec::new();
      let mut playouts: i64 = 0;
      let mut check = |search: &Search| {
        errs = check_invariants(search, threads, spec.max_visits, mode);
        playouts = search.last_search_num_playouts;
      };
      let t0 = Stopwatch::new();
      let first = try_catch(|| run_case(spec, Variant { ops: Some(&ops), num_threads: Some(threads), overrides: &overrides, check: Some(&mut check) }));
      let first = match first {
        Ok(s) => s,
        Err(p) => {
          all_errs.push(format!("case {} threads {} {:?}: PANIC {}", spec.idx, threads, mode, p));
          continue;
        }
      };
      //Determinism: the same case in a fresh Search + NNEvaluator gives the same bytes.
      let second = try_catch(|| run_case(spec, Variant { ops: Some(&ops), num_threads: Some(threads), overrides: &overrides, check: None }));
      match second {
        Ok(second) => {
          if let Err(m) = try_catch(|| compare_lines(&format!("determinism case {} threads {}", spec.idx, threads), &second, &first)) {
            all_errs.push(m);
          }
        }
        Err(p) => all_errs.push(format!("case {} threads {} {:?} (second run): PANIC {}", spec.idx, threads, mode, p)),
      }
      let single = try_catch(|| run_case(spec, Variant { ops: Some(&ops), num_threads: Some(1), overrides: &overrides, check: None }));
      let differs = matches!(single, Ok(ref s) if *s != first);
      differs_from_single += differs as i32;
      overshoot += (playouts > spec.max_visits) as i32;
      for e in errs {
        all_errs.push(format!("case {} [{}] threads {} {:?} maxVisits {}: {}", spec.idx, spec.kind, threads, mode, spec.max_visits, e));
      }
      runs += 1;
      let _ = writeln!(
        summary,
        "case {:3} {:<24} {}x{} visits {:3} threads {:2} {:<5}: playouts {:3} differs-from-1-thread {} {}",
        spec.idx,
        spec.kind,
        spec.xs,
        spec.ys,
        spec.max_visits,
        threads,
        format!("{:?}", mode),
        playouts,
        differs as i32,
        t0.elapsed()
      );
    }
  }
  let _ = writeln!(
    summary,
    "{} of {} runs differ from the 1-thread run, {} overshoot maxVisits (threads still in flight when the stop test fires)",
    differs_from_single, runs, overshoot
  );
  if differs_from_single == 0 || overshoot == 0 {
    all_errs.push("the virtual threads never interleaved: the invariants above would be vacuous".to_string());
  }
  let _ = writeln!(summary, "{} (case, threads, mode) combinations, each run twice, {} in total", runs, started.elapsed());
  if !all_errs.is_empty() {
    summary.push_str(&format!("{} FAILURES:\n{}\n", all_errs.len(), all_errs.join("\n")));
  }
  summary
}

/// Rust-only (no C++ counterpart): numSearchThreads = 2, 4 and 16 virtual threads on 36 of the oracle cases (small
/// boards, 20..300 visits, spread over every generator) plus the two largest wide trees. Per case: tree mode
/// (useGraphSearch and the passing hacks off: the strict counting invariants apply) with 2, 4 and 16 threads, graph
/// mode (hacks off) with 4, and the case's own params ("raw": hacks, weightless playouts) with 4 and 16 threads. Upstream is
/// nondeterministic with real threads, so this checks
///  (a) determinism of the port: two fresh Search + NNEvaluator runs give byte-identical dumps (RS, printTree
///      depth 2, AD, analysis JSON, chosen moves, play selection values); and that the dump differs from the
///      1-thread run (the virtual threads really interleave);
///  (b) invariants, with the reasons in `check_invariants`: maxVisits <= playouts <= maxVisits + threads - 1;
///      root visits == playouts (tree/graph; `<=` in raw mode); sum of root edge visits == root visits - 1 (tree mode;
///      `<=` otherwise); every node's virtual_losses == 0 and no node in a transient state after the search;
///      enumerate_tree_post_order visits each node once with children before parents and the root last; the node table
///      holds exactly the reachable non-root nodes; in tree mode #nodes == 1 + #child pointers;
///  (c) no stall: every pool run goes through `common::run_or_stall_panic`.
#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn multithread_virtual_threads() {
  let summary = {
    #[cfg(not(target_arch = "wasm32"))]
    {
      std::thread::Builder::new()
        .name("search_oracle_mt".to_string())
        .stack_size(256 << 20)
        .spawn(run_multithread)
        .unwrap()
        .join()
        .expect("run_multithread panicked outside the per-run catch")
    }
    #[cfg(target_arch = "wasm32")]
    {
      run_multithread()
    }
  };
  crate::report!("search_oracle multithread:\n{}", summary);
  assert!(!summary.contains("FAILURES"), "multi-thread invariants violated:\n{}", summary);
}
