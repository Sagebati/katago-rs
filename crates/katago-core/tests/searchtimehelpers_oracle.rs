//! search/searchtimehelpers (Search::numVisitsNeededToBeNonFutile, computeUpperBoundVisitsLeftDueToTime,
//! recomputeSearchTimeLimit) against the unmodified upstream Search.
//!
//! Upstream's ClockTimer reads std::chrono::steady_clock, so a time-limited C++ search cannot be made
//! deterministic. The three functions only read the Search (root board / history, SearchParams,
//! effectiveSearchTimeCarriedOver, the root node's policy and play selection values) and their arguments,
//! so fixtures/upstream-v1.18.2/unit/oracle/searchtimehelpers.cpp builds 150 searches (5..19 wide boards,
//! varied rules, move counts and initial turn numbers; 0..200 NN-less visits on an NNEvaluator with
//! debugSkipNeuralNet so the root node exists) and calls the three functions directly on a grid of inputs
//! (TimeControls of every kind and state, timeUsed, searchFactor, rootVisits, carried-over time, the time
//! related SearchParams). Everything, floats as IEEE-754 bits, is on the fixture lines; this test rebuilds
//! each case from its CASE line, replays every `I` / `NF` / `UB` / `RT` line through the port and compares
//! the regenerated line with the fixture line (per case, every mismatch is reported with its inputs; the
//! test fails at the end if any). The line format is documented in searchtimehelpers.cpp's header.
//!
//! Second test (Rust only, no C++ counterpart): `time_limited_search_fake_clock` runs real time-limited
//! searches (maxTime, and TimeControls) with an injected deterministic clock that advances a fixed step per
//! read, and checks that they stop on time with >= 2 playouts and that the playout count reproduces.
//!
//! Access: the three functions are `pub` (PORT NOTE in searchtimehelpers.rs) and called through `access`.
//!
//! Not a port of upstream code.

#![allow(clippy::too_many_arguments, clippy::field_reassign_with_default)]

#[macro_use]
mod common;

use std::cell::Cell;
use std::collections::HashMap;
use std::rc::Rc;

use common::init;
use common::run_or_stall_panic;
use futures::executor::LocalPool;
use katago_core::neuralnet::nninterface::BatchPolicy;
use katago_core::core::config_parser::ConfigParser;
use katago_core::core::commontypes::enabled_t;
use katago_core::core::global::KataError;
use katago_core::core::timer::set_clock;
use katago_core::game::board::*;
use katago_core::game::boardhistory::BoardHistory;
use katago_core::game::boardhistory::BoardHistoryModes;
use katago_core::game::rules::Rules;
use katago_core::neuralnet::desc::ModelDesc;
use katago_core::neuralnet::nneval::NNEvaluator;
use katago_core::neuralnet::nneval::NNResultBuf;
use katago_core::neuralnet::nninputs::NNOutput;
use katago_core::neuralnet::nninputs::ScoreValue;
use katago_core::neuralnet::nninterface::*;
use katago_core::port::executor::LocalBoxFuture;
use katago_core::port::executor::Spawner;
use katago_core::search::search::Search;
use katago_core::search::searchparams::SearchParams;
use katago_core::search::timecontrols::TimeControls;

//------------------------------------------------------------------------------------------------
// The three functions under test

mod access {
  use super::*;
  pub fn num_visits_needed_to_be_non_futile(s: &Search, x: f64) -> f64 {
    s.num_visits_needed_to_be_non_futile(x)
  }
  pub fn compute_upper_bound_visits_left_due_to_time(s: &Search, root_visits: i64, time_used: f64, planned_time_limit: f64) -> f64 {
    s.compute_upper_bound_visits_left_due_to_time(root_visits, time_used, planned_time_limit)
  }
  pub fn recompute_search_time_limit(s: &Search, tc: &TimeControls, time_used: f64, search_factor: f64, root_visits: i64) -> Result<f64, KataError> {
    s.recompute_search_time_limit(tc, time_used, search_factor, root_visits)
  }
}

//------------------------------------------------------------------------------------------------
// Injected clock: advances a fixed step per read, per thread (each test resets it)

const CLOCK_STEP: f64 = 0.01;

thread_local! {
  static TICKS: Cell<u64> = const { Cell::new(0) };
}

fn fake_clock() -> f64 {
  TICKS.with(|t| {
    t.set(t.get() + 1);
    t.get() as f64 * CLOCK_STEP
  })
}

fn install_clock() {
  set_clock(fake_clock);
}

//------------------------------------------------------------------------------------------------
// Backend that must never be reached (debugSkipNeuralNet), the executor, the evaluator
// (copies of search_oracle.rs)

struct NoBackend;

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
    _: &[i32], _: i32, _: i32, _: &str, _: enabled_t, _: &dyn LoadedModel, _: &mut ConfigParser,
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

/// upstream: TestSearchCommon::startNNEval (testsearchcommon.cpp) with the arguments search_nonn.cpp uses.
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
    enabled_t::False,
    1,
    &gpu_idx_by_server_thread,
    &nn_rand_seed,
    false,
    0,
    &mut ConfigParser::new(false, true),
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

fn boxed_history(board: &Board, pla: Player, rules: &Rules) -> Box<BoardHistory> {
  Box::new(BoardHistory::new_full(board, pla, rules, 0, &BoardHistoryModes::new()))
}

fn base_params(base: i32) -> SearchParams {
  match base {
    1 => SearchParams::for_tests_v1(),
    2 => SearchParams::for_tests_v2(),
    _ => SearchParams::new(),
  }
}

//------------------------------------------------------------------------------------------------
// Parsing

fn hex_f64(s: &str) -> f64 {
  f64::from_bits(u64::from_str_radix(s, 16).unwrap_or_else(|_| panic!("bad hex double {:?}", s)))
}

/// The oracle's `setParam`: key=value text onto a SearchParams (doubles as hex bits, numThreads / maxVisits decimal).
fn set_param(p: &mut SearchParams, k: &str, v: &str) {
  let d = || hex_f64(v);
  match k {
    "lagBuffer" => p.lag_buffer = d(),
    "overallocateTimeFactor" => p.overallocate_time_factor = d(),
    "midgameTimeFactor" => p.midgame_time_factor = d(),
    "midgameTurnPeakTime" => p.midgame_turn_peak_time = d(),
    "endgameTurnTimeDecay" => p.endgame_turn_time_decay = d(),
    "obviousMovesTimeFactor" => p.obvious_moves_time_factor = d(),
    "obviousMovesPolicyEntropyTolerance" => p.obvious_moves_policy_entropy_tolerance = d(),
    "obviousMovesPolicySurpriseTolerance" => p.obvious_moves_policy_surprise_tolerance = d(),
    "futileVisitsThreshold" => p.futile_visits_threshold = d(),
    "chosenMoveTemperature" => p.chosen_move_temperature = d(),
    "chosenMoveTemperatureEarly" => p.chosen_move_temperature_early = d(),
    "chosenMoveTemperatureHalflife" => p.chosen_move_temperature_halflife = d(),
    "treeReuseCarryOverTimeFactor" => p.tree_reuse_carry_over_time_factor = d(),
    "cpuctExploration" => p.cpuct_exploration = d(),
    "rootPolicyTemperature" => p.root_policy_temperature = d(),
    "staticScoreUtilityFactor" => p.static_score_utility_factor = d(),
    "dynamicScoreUtilityFactor" => p.dynamic_score_utility_factor = d(),
    "numThreads" => p.num_threads = v.parse().unwrap(),
    "maxVisits" => p.max_visits = v.parse().unwrap(),
    _ => panic!("bad param {}", k),
  }
}

fn parse_params(s: &str) -> Vec<(String, String)> {
  if s == "-" {
    return Vec::new();
  }
  s.split(',')
    .map(|p| {
      let (k, v) = p.split_once('=').unwrap();
      (k.to_string(), v.to_string())
    })
    .collect()
}

struct Spec {
  idx: i32,
  nn_x: i32,
  nn_y: i32,
  nn_seed: String,
  xs: i32,
  ys: i32,
  rules: Rules,
  initial_pla: Player,
  initial_turn_number: i64,
  moves: Vec<(Loc, Player)>,
  pla: Player,
  search_seed: String,
  base: i32,
  visits: i64,
  run: char,
  sp: Vec<(String, String)>,
  tp: Vec<(String, String)>,
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
  Spec {
    idx: toks[1].parse().unwrap(),
    nn_x: nn_x.parse().unwrap(),
    nn_y: nn_y.parse().unwrap(),
    nn_seed: get("nnseed").to_string(),
    xs: xs.parse().unwrap(),
    ys: ys.parse().unwrap(),
    rules,
    initial_pla: int("ipla") as Player,
    initial_turn_number: get("iturn").parse().unwrap(),
    moves,
    pla: int("pla") as Player,
    search_seed: get("sseed").to_string(),
    base: int("base"),
    visits: get("visits").parse().unwrap(),
    run: get("run").chars().next().unwrap(),
    sp: parse_params(get("sp")),
    tp: parse_params(get("tp")),
  }
}

/// `tcText` of the oracle: the 12 fields of a TimeControls.
fn parse_tc(fields: &[&str]) -> TimeControls {
  assert_eq!(fields.len(), 12, "TC fields {:?}", fields);
  let mut tc = TimeControls::new();
  tc.original_main_time = hex_f64(fields[0]);
  tc.increment = hex_f64(fields[1]);
  tc.main_time_limit = hex_f64(fields[2]);
  tc.max_time_per_move = hex_f64(fields[3]);
  tc.original_num_periods = fields[4].parse().unwrap();
  tc.num_stones_per_period = fields[5].parse().unwrap();
  tc.per_period_time = hex_f64(fields[6]);
  tc.main_time_left = hex_f64(fields[7]);
  tc.in_overtime = fields[8] == "1";
  tc.num_periods_left_including_current = fields[9].parse().unwrap();
  tc.num_stones_left_in_period = fields[10].parse().unwrap();
  tc.time_left_in_period = hex_f64(fields[11]);
  tc
}

/// The fixture split into (CASE line, the lines after it up to END).
fn split_cases(text: &str) -> Vec<(String, Vec<String>)> {
  let mut out: Vec<(String, Vec<String>)> = Vec::new();
  let mut cur: Option<(String, Vec<String>)> = None;
  for l in text.lines() {
    if l.starts_with("CASE ") {
      assert!(cur.is_none(), "CASE inside a case");
      cur = Some((l.to_string(), Vec::new()));
    } else if l.starts_with("END ") {
      out.push(cur.take().expect("END outside a case"));
    } else {
      cur.as_mut().expect("text outside a case").1.push(l.to_string());
    }
  }
  assert!(cur.is_none(), "unterminated last case");
  out
}

//------------------------------------------------------------------------------------------------
// One case

fn bits(x: f64) -> String {
  format!("{:016X}", x.to_bits())
}

/// Replays the case; returns every mismatching line as (expected, actual, what).
fn run_case(spec: &Spec, lines: &[String], counts: &mut Counts) -> Vec<(String, String)> {
  let mut ctx = Ctx::new();
  let nn_eval = start_nn_eval(&mut ctx, &spec.nn_seed, spec.nn_x, spec.nn_y);
  let mut params = base_params(spec.base);
  for (k, v) in &spec.sp {
    set_param(&mut params, k, v);
  }
  params.num_threads = 1;
  params.max_visits = spec.visits;
  let mut search = Box::new(Search::new(&params, nn_eval.clone(), &spec.search_seed));
  {
    let mut board = Board::new_x_y(spec.xs, spec.ys).unwrap();
    let mut hist = boxed_history(&board, spec.initial_pla, &spec.rules);
    for (loc, pla) in &spec.moves {
      hist.make_board_move_assume_legal(&mut board, *loc, *pla, None, false);
    }
    hist.set_initial_turn_number(spec.initial_turn_number);
    search.set_position(spec.pla, &board, &hist).unwrap();
  }
  if spec.run == 'R' {
    run_or_stall_panic(&mut ctx.pool, "runWholeSearch", search.run_whole_search(spec.pla)).unwrap();
  } else {
    run_or_stall_panic(&mut ctx.pool, "beginSearch", search.begin_search(false)).unwrap();
  }
  for (k, v) in &spec.tp {
    set_param(&mut search.search_params, k, v);
  }

  let mut mismatches: Vec<(String, String)> = Vec::new();
  let mut tcs: Vec<TimeControls> = Vec::new();
  for expected in lines {
    let toks: Vec<&str> = expected.split(' ').collect();
    let kv = |name: &str| -> &str {
      toks.iter()
        .find_map(|t| t.strip_prefix(name).and_then(|r| r.strip_prefix('=')))
        .unwrap_or_else(|| panic!("missing {} in {}", name, expected))
    };
    let actual: String = match toks[0] {
      "I" => {
        let (mut su, mut se, mut pe) = (0.0, 0.0, 0.0);
        let ok = search.get_policy_surprise_and_entropy(&mut su, &mut se, &mut pe);
        let mut locs: Vec<Loc> = Vec::new();
        let mut psvals: Vec<f64> = Vec::new();
        let mut visit_counts: Vec<f64> = Vec::new();
        let sok = search.get_play_selection_values_ret_visit_counts(&mut locs, &mut psvals, Some(&mut visit_counts), 1.0);
        let mut maxv = 0.0f64;
        let mut best = 0usize;
        for i in 0..visit_counts.len() {
          if visit_counts[i] > maxv {
            maxv = visit_counts[i];
          }
          if psvals[i] > psvals[best] {
            best = i;
          }
        }
        counts.i += 1;
        if ok {
          counts.ps_ok += 1;
        }
        format!(
          "I turn={} stones={} rv={} ps={} {} {} {} psv={} n={} maxvisits={} best={}",
          search.root_history.get_current_turn_number(),
          search.root_board.num_stones_on_board(),
          search.get_root_visits(),
          ok as i32,
          bits(su),
          bits(se),
          bits(pe),
          sok as i32,
          psvals.len(),
          bits(maxv),
          best
        )
      }
      "TC" => {
        assert_eq!(toks[1].parse::<usize>().unwrap(), tcs.len(), "TC index in {}", expected);
        tcs.push(parse_tc(&toks[2..]));
        expected.clone() // inputs only
      }
      "NF" => {
        let x = hex_f64(kv("x"));
        let r = access::num_visits_needed_to_be_non_futile(&search, x);
        counts.nf += 1;
        format!("NF x={} r={}", bits(x), bits(r))
      }
      "UB" => {
        let car = hex_f64(kv("car"));
        let rv: i64 = kv("rv").parse().unwrap();
        let tu = hex_f64(kv("tu"));
        let pl = hex_f64(kv("pl"));
        search.effective_search_time_carried_over = car;
        let r = access::compute_upper_bound_visits_left_due_to_time(&search, rv, tu, pl);
        counts.ub += 1;
        format!("UB car={} rv={} tu={} pl={} r={}", bits(car), rv, bits(tu), bits(pl), bits(r))
      }
      "RT" => {
        let car = hex_f64(kv("car"));
        let ti: usize = kv("tc").parse().unwrap();
        let tu = hex_f64(kv("tu"));
        let sf = hex_f64(kv("sf"));
        let rv: i64 = kv("rv").parse().unwrap();
        search.effective_search_time_carried_over = car;
        let r = match access::recompute_search_time_limit(&search, &tcs[ti], tu, sf, rv) {
          Ok(r) => bits(r),
          Err(KataError::StringError(_)) => "EXC".to_string(),
          Err(e) => panic!("unexpected error {:?} in {}", e, expected),
        };
        counts.rt += 1;
        if r == "EXC" {
          counts.exc += 1;
        }
        format!("RT car={} tc={} tu={} sf={} rv={} r={}", bits(car), ti, bits(tu), bits(sf), rv, r)
      }
      other => panic!("unknown line kind {:?}: {}", other, expected),
    };
    if !line_matches(toks[0], &actual, expected) {
      mismatches.push((expected.clone(), actual));
    }
  }
  drop(search);
  delete_nn_eval(&mut ctx, nn_eval);
  mismatches
}

/// Exact equality. Only on wasm32, the `I` line's search-derived doubles (policy surprise / entropies, from a
/// 200-visit search whose utilities use `port::math::atan`/`tan`, which wrap the platform libm there: up to 1 ulp
/// from glibc, DECISIONS.md D13; search_oracle.rs is native-only for the same reason) may differ by up to 4 ulps.
/// The three functions under test are always compared exactly.
fn line_matches(kind: &str, actual: &str, expected: &str) -> bool {
  if actual == expected {
    return true;
  }
  if !cfg!(target_arch = "wasm32") || kind != "I" {
    return false;
  }
  let a: Vec<&str> = actual.split(' ').collect();
  let e: Vec<&str> = expected.split(' ').collect();
  if a.len() != e.len() {
    return false;
  }
  a.iter().zip(e.iter()).all(|(x, y)| {
    if x == y {
      return true;
    }
    // only the three 16-hex-digit doubles after "ps=<0|1>" (not maxvisits=, which is a visit count)
    match (u64::from_str_radix(x, 16), u64::from_str_radix(y, 16)) {
      (Ok(xb), Ok(yb)) if x.len() == 16 && y.len() == 16 => xb.abs_diff(yb) <= 4,
      _ => false,
    }
  })
}

#[derive(Default)]
struct Counts {
  cases: usize,
  i: usize,
  ps_ok: usize,
  nf: usize,
  ub: usize,
  rt: usize,
  exc: usize,
}

/// Number of cases in the fixture (searchtimehelpers.cpp: N).
const NUM_CASES: usize = 150;

fn fixture_text() -> (String, bool) {
  common::fixture_or_env("SEARCHTIMEHELPERS_ORACLE_FILE", "oracle", "searchtimehelpers.txt")
}

wasm_test! {
  fn searchtimehelpers_oracle() {
    init();
    ScoreValue::init_tables(); //upstream's main(): "MUST BE CALLED AT PROGRAM START!"
    install_clock();
    let (text, explore) = fixture_text();
    let cases = split_cases(&text);
    if !explore {
      assert_eq!(cases.len(), NUM_CASES, "number of cases in searchtimehelpers.txt");
    }
    let mut counts = Counts::default();
    let mut failures: Vec<String> = Vec::new();
    let mut num_mismatch_lines = 0usize;
    for (line, body) in &cases {
      let spec = parse_spec(line);
      let result = run_guarded(|| run_case(&spec, body, &mut counts));
      counts.cases += 1;
      match result {
        Err(p) => failures.push(format!("case {}: PANIC: {}\n  {}", spec.idx, p, line)),
        Ok(mm) => {
          num_mismatch_lines += mm.len();
          if !mm.is_empty() {
            let mut msg = format!("case {}: {} mismatching lines\n  {}", spec.idx, mm.len(), line);
            for (e, a) in mm.iter().take(6) {
              msg.push_str(&format!("\n  expected: {}\n    actual: {}", e, a));
            }
            failures.push(msg);
          }
        }
      }
    }
    crate::report!(
      "searchtimehelpers_oracle: {} cases, {} I lines ({} with policy surprise), {} NF, {} UB, {} RT lines ({} EXC): {} mismatching lines",
      counts.cases, counts.i, counts.ps_ok, counts.nf, counts.ub, counts.rt, counts.exc, num_mismatch_lines
    );
    assert!(counts.cases >= 1 && counts.rt >= 1, "no cases ran");
    if !failures.is_empty() {
      let shown: Vec<String> = failures.iter().take(8).cloned().collect();
      panic!("{} of {} cases mismatch:\n{}", failures.len(), counts.cases, shown.join("\n"));
    }
    // The generator must reach the branches (a fixture that never exercises them would pass vacuously).
    assert!(counts.ps_ok >= 100 && counts.exc >= 50, "fixture coverage: {} policy cases, {} EXC", counts.ps_ok, counts.exc);
  }
}

#[cfg(not(target_arch = "wasm32"))]
fn run_guarded<R>(f: impl FnOnce() -> R) -> Result<R, String> {
  let prev = std::panic::take_hook();
  std::panic::set_hook(Box::new(|_| {}));
  let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
  std::panic::set_hook(prev);
  r.map_err(|p| {
    if let Some(s) = p.downcast_ref::<&str>() {
      s.to_string()
    } else if let Some(s) = p.downcast_ref::<String>() {
      s.clone()
    } else {
      "<non-string panic payload>".to_string()
    }
  })
}

#[cfg(target_arch = "wasm32")]
fn run_guarded<R>(f: impl FnOnce() -> R) -> Result<R, String> {
  Ok(f())
}

//------------------------------------------------------------------------------------------------
// Rust-only: time-limited searches under an injected deterministic clock

/// One search on an empty 9x9 board (NN-less) with the time limits given; returns (playouts, clock reads).
fn timed_search(num_threads: i32, max_time: f64, tc: &TimeControls) -> (i64, u64) {
  TICKS.with(|t| t.set(0));
  let mut ctx = Ctx::new();
  let nn_eval = start_nn_eval(&mut ctx, "fake_clock", 9, 9);
  let mut params = SearchParams::for_tests_v1();
  params.num_threads = num_threads;
  params.max_visits = 1_000_000;
  params.max_playouts = 1_000_000;
  params.max_time = max_time;
  let mut search = Box::new(Search::new(&params, nn_eval.clone(), "fake_clock_search"));
  {
    let board = Board::new_x_y(9, 9).unwrap();
    let hist = boxed_history(&board, P_BLACK, &Rules::new());
    search.set_position(P_BLACK, &board, &hist).unwrap();
  }
  let sf = 1.0;
  run_or_stall_panic(&mut ctx.pool, "timed search", search.run_whole_search_full(None, None, false, tc, sf, None)).unwrap();
  let playouts = search.last_search_num_playouts;
  drop(search);
  delete_nn_eval(&mut ctx, nn_eval);
  (playouts, TICKS.with(|t| t.get()))
}

wasm_test! {
  fn time_limited_search_fake_clock() {
    init();
    ScoreValue::init_tables();
    install_clock();
    let unlimited = TimeControls::new();
    // (label, threads, maxTime, time controls)
    let cases: Vec<(&str, i32, f64, TimeControls)> = vec![
      ("maxTime 0.5s, 1 thread", 1, 0.5, unlimited),
      ("maxTime 0.5s, 4 virtual threads", 4, 0.5, unlimited),
      ("absolute time 12s", 1, 1.0e20, TimeControls::absolute_time(12.0)),
      ("fischer 20s+0.3s, 3 virtual threads", 3, 1.0e20, TimeControls::fischer_time(20.0, 0.3)),
      ("byo-yomi 5s main + 5x3s/1 stone", 1, 1.0e20, TimeControls::canadian_or_byo_yomi_time(5.0, 3.0, 5, 1)),
    ];
    let mut n = 0;
    for (label, threads, max_time, tc) in &cases {
      let (p1, ticks1) = timed_search(*threads, *max_time, tc);
      let (p2, ticks2) = timed_search(*threads, *max_time, tc);
      crate::report!("time_limited_search_fake_clock: {}: {} playouts, {} clock reads", label, p1, ticks1);
      assert!(p1 >= 2, "{}: {} playouts, the time limit must still allow at least 2", label, p1);
      assert!(p1 < 1_000_000, "{}: {} playouts, the search did not stop by time", label, p1);
      assert_eq!((p1, ticks1), (p2, ticks2), "{}: not reproducible across two runs", label);
      n += 1;
    }
    assert!(n >= 1);
    crate::report!("time_limited_search_fake_clock: {} searches, each run twice", n);
  }
}
