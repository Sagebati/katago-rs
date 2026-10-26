//! Differential oracle for the human-SL paths of the ported Search (M8): a main and a human-SL NNEvaluator
//! (both debugSkipNeuralNet, so `requiresSGFMetadata()` is false and every human-SL path is live) against the
//! unmodified upstream Search: fixtures/upstream-v1.18.2/unit/oracle/search_humansl.cpp (its header documents the
//! CASE line and the fixed op sequence: search, printTree, getPlaySelectionValues, getChosenMoveLoc, getAnalysisJson
//! with humanPrior / humanPolicy / rootInfo human*, then makeMove and a second search on the reused tree). Each case is
//! rebuilt from its CASE line and must print the same bytes (floats as bits or through upstream's printers).
//! Runs natively and on wasm32 (`just test-wasm`). Kept apart from tests/search_oracle.rs (search_nonn.cpp).
//! Not a port of upstream code.

#![allow(clippy::write_with_newline)]

#[macro_use]
mod common;

use std::cell::RefCell;
use std::fmt::Write as _;
use std::rc::Rc;

use common::Cases;
use common::compare_lines;
use common::fixture;
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
use katago_core::neuralnet::sgfmetadata::SGFMetadata;
use katago_core::port::executor::LocalBoxFuture;
use katago_core::port::executor::Spawner;
use katago_core::port::nlohmann;
use katago_core::search::search::Search;
use katago_core::search::searchparams::SearchParams;
use katago_core::search::searchprint::PrintTreeOptions;

thread_local! {
  static OUT: RefCell<String> = const { RefCell::new(String::new()) };
}

struct Cout;

impl std::fmt::Write for Cout {
  fn write_str(&mut self, s: &str) -> std::fmt::Result {
    OUT.with(|o| o.borrow_mut().push_str(s));
    Ok(())
  }
}

macro_rules! coutln {
  ($($t:tt)*) => {{
    write!(Cout, $($t)*).unwrap();
    Cout.write_str("\n").unwrap();
  }};
}

/// The oracle's Logger (`": " + message`) for the Search's records only (the evaluators got a NULL logger),
/// until the returned guard drops.
fn install_logger_and_clock() -> tracing::subscriber::DefaultGuard {
  set_clock(common::fixed_clock);
  common::capture_logs("katago_core::search", |msg| coutln!(": {}", msg))
}

/// debugSkipNeuralNet: the backend is never reached.
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

/// The oracle's `new NNEvaluator(name, name, "", NULL, 16, 19, 19, false, true, 16, 12, true, "", "", false, False, False,
/// 1, {0}, seed, false, 0)` + spawnServerThreads.
fn start_nn_eval(pool: &mut LocalPool, spawner: &Rc<dyn Spawner>, name: &str, seed: &str) -> Rc<NNEvaluator> {
  let nn_eval = Rc::new(
    NNEvaluator::new(
      name,
      name,
      "",
      16,
      19,
      19,
      false,
      true,
      16,
      12,
      true,
      "",
      enabled_t::False,
      1,
      &[0],
      seed,
      false,
      0,
      &mut ConfigParser::new(false, true),
      Rc::new(NoBackend),
      spawner.clone(),
      Vec::new(),
    )
    .unwrap(),
  );
  run_or_stall_panic(pool, "spawnServerThreads", nn_eval.spawn_server_threads()).unwrap();
  nn_eval
}

fn set_param(p: &mut SearchParams, k: &str, v: &str) -> bool {
  let d: f64 = v.parse().unwrap_or(0.0);
  let b = v == "1";
  match k {
    "humanSLCpuctExploration" => p.human_sl_cpuct_exploration = d,
    "humanSLCpuctPermanent" => p.human_sl_cpuct_permanent = d,
    "humanSLRootExploreProbWeightless" => p.human_sl_root_explore_prob_weightless = d,
    "humanSLRootExploreProbWeightful" => p.human_sl_root_explore_prob_weightful = d,
    "humanSLPlaExploreProbWeightless" => p.human_sl_pla_explore_prob_weightless = d,
    "humanSLPlaExploreProbWeightful" => p.human_sl_pla_explore_prob_weightful = d,
    "humanSLOppExploreProbWeightless" => p.human_sl_opp_explore_prob_weightless = d,
    "humanSLOppExploreProbWeightful" => p.human_sl_opp_explore_prob_weightful = d,
    "humanSLChosenMoveProp" => p.human_sl_chosen_move_prop = d,
    "humanSLChosenMoveIgnorePass" => p.human_sl_chosen_move_ignore_pass = b,
    "humanSLChosenMovePiklLambda" => p.human_sl_chosen_move_pikl_lambda = d,
    "chosenMoveTemperature" => p.chosen_move_temperature = d,
    "chosenMoveTemperatureEarly" => p.chosen_move_temperature_early = d,
    "rootPolicyTemperature" => p.root_policy_temperature = d,
    "wideRootNoise" => p.wide_root_noise = d,
    "cpuctExploration" => p.cpuct_exploration = d,
    "useNoisePruning" => p.use_noise_pruning = b,
    "enablePassingHacks" => p.enable_passing_hacks = b,
    "useLcbForSelection" => p.use_lcb_for_selection = b,
    "rootPruneUselessMoves" => p.root_prune_useless_moves = b,
    "useGraphSearch" => p.use_graph_search = b,
    "humanSLProfile" => p.human_sl_profile = SGFMetadata::get_profile(v).expect("profile"),
    _ => return false,
  }
  true
}

fn fnv64(s: &str) -> u64 {
  let mut h: u64 = 0xcbf29ce484222325;
  for b in s.bytes() {
    h ^= b as u64;
    h = h.wrapping_mul(0x100000001b3);
  }
  h
}

fn psv(search: &Search) {
  let mut locs: Vec<Loc> = Vec::new();
  let mut vals: Vec<f64> = Vec::new();
  let mut visits: Vec<f64> = Vec::new();
  let suc = search.get_play_selection_values_ret_visit_counts(&mut locs, &mut vals, Some(&mut visits), 1.0);
  coutln!("success={} n={} nv={}", suc as i32, locs.len(), visits.len());
  if suc {
    for i in 0..locs.len() {
      let mut line = format!("s {} {} {:016X}", locs[i], Location::to_string(locs[i], search.get_root_board()), vals[i].to_bits());
      if i < visits.len() {
        line += &format!(" {:016X}", visits[i].to_bits());
      } else {
        line += " -";
      }
      coutln!("{}", line);
    }
  }
}

fn search_and_report(pool: &mut LocalPool, search: &mut Search, pla: Player, full_json: bool) -> Result<Loc, KataError> {
  coutln!("> R {}", pla);
  run_or_stall_panic(pool, "runWholeSearch", search.run_whole_search(pla))?;
  coutln!("playouts={}", search.last_search_num_playouts);
  coutln!("> PT");
  let root = search.get_root_node();
  search.print_tree(&mut Cout, root.as_deref(), &PrintTreeOptions::new().max_depth(1), P_WHITE);
  coutln!("> PSV");
  psv(search);
  coutln!("> CM");
  let mut first: Loc = Board::NULL_LOC;
  for i in 0..3 {
    let l = search.get_chosen_move_loc();
    if i == 0 {
      first = l;
    }
    coutln!("cm {} {}", l, Location::to_string(l, search.get_root_board()));
  }
  coutln!("> J");
  let mut json = serde_json::Value::Null;
  let suc = search.get_analysis_json(P_WHITE, 3, true, true, false, false, false, false, true, false, &mut json)?;
  coutln!("success={}", suc as i32);
  let dumped = nlohmann::dump(&json);
  if full_json {
    coutln!("{}", dumped);
  } else {
    coutln!("jfnv {:016X} {}", fnv64(&dumped), dumped.len());
  }
  Ok(first)
}

/// The values of a CASE line's key=value tokens.
fn field<'a>(line: &'a str, key: &str) -> &'a str {
  line.split(' ').find_map(|t| t.strip_prefix(key).and_then(|r| r.strip_prefix('='))).unwrap_or_else(|| panic!("{key} in {line}"))
}

fn boxed_history(board: &Board, pla: Player, rules: &Rules) -> Box<BoardHistory> {
  Box::new(BoardHistory::new_full(board, pla, rules, 0, &BoardHistoryModes::new()))
}

fn run_case(line: &str) -> String {
  OUT.with(|o| o.borrow_mut().clear());
  let idx: i32 = line.split(' ').nth(1).unwrap().parse().unwrap();
  let (xs, ys) = field(line, "board").split_once('x').unwrap();
  let (xs, ys): (i32, i32) = (xs.parse().unwrap(), ys.parse().unwrap());
  let mut rules = Rules::parse_rules(field(line, "rules")).unwrap();
  rules.komi = f32::from_bits(u32::from_str_radix(field(line, "komi"), 16).unwrap());
  let pla: Player = field(line, "pla").parse().unwrap();
  let human = field(line, "human") == "1";
  let sseed = field(line, "sseed");
  let base: i32 = field(line, "base").parse().unwrap();
  let max_visits: i64 = field(line, "maxvisits").parse().unwrap();

  let mut board = Board::new_x_y(xs, ys).unwrap();
  let mut hist = boxed_history(&board, P_BLACK, &rules);
  let moves = field(line, "moves");
  if moves != "-" {
    for m in moves.split(',') {
      let (l, p) = m.split_once('/').unwrap();
      hist.make_board_move_assume_legal(&mut board, l.parse().unwrap(), p.parse().unwrap(), None, false);
    }
  }
  let mut params = if base == 2 { SearchParams::for_tests_v2() } else { SearchParams::new() };
  params.num_threads = 1;
  let ps = field(line, "params");
  if ps != "-" {
    for kv in ps.split(',') {
      let (k, v) = kv.split_once('=').unwrap();
      assert!(set_param(&mut params, k, v), "bad param {k}");
    }
  }
  params.max_visits = max_visits;

  let mut pool = LocalPool::new();
  let spawner: Rc<dyn Spawner> = Rc::new(pool.spawner());
  let nn_eval = start_nn_eval(&mut pool, &spawner, "search_humansl", &format!("runSearchTestsRandSeed{idx}"));
  let human_eval = if human { Some(start_nn_eval(&mut pool, &spawner, "search_humansl_human", &format!("humanSLRandSeed{idx}"))) } else { None };
  {
    let mut search = Box::new(Search::new_human_eval(&params, nn_eval.clone(), human_eval.clone(), sseed).unwrap());
    coutln!("> P");
    search.set_position(pla, &board, &hist).unwrap();
    let mv = search_and_report(&mut pool, &mut search, pla, idx < 24).unwrap();
    if mv != Board::NULL_LOC {
      coutln!("> M {} {}", mv, pla);
      let suc = search.make_move(mv, pla).unwrap();
      coutln!("suc={}", suc as i32);
      if suc && !search.get_root_hist().is_game_finished {
        search_and_report(&mut pool, &mut search, get_opp(pla), idx < 12).unwrap();
      }
    }
  }
  for e in [Some(nn_eval), human_eval].into_iter().flatten() {
    e.kill_server_threads();
    pool.run_until_stalled();
    assert_eq!(Rc::strong_count(&e), 1, "the serve task ended and released the evaluator");
  }
  let out = OUT.with(|o| o.borrow().clone());
  format!("{line}\n{out}END {idx}\n")
}

const NUM_CASES: usize = 120;

wasm_test! {
fn search_humansl_vs_upstream() {
  init();
  ScoreValue::init_tables();
  let _logs = install_logger_and_clock();
  let text = fixture("oracle", "search_humansl.txt");
  let mut cases = Cases::new("search_humansl_vs_upstream");
  let mut blocks: Vec<(String, String)> = Vec::new();
  let mut cur: Option<(String, String)> = None;
  for l in text.split_inclusive('\n') {
    if l.starts_with("CASE ") {
      assert!(cur.is_none());
      cur = Some((l.trim_end_matches('\n').to_string(), l.to_string()));
    } else {
      let c = cur.as_mut().expect("line outside a case");
      c.1.push_str(l);
      if l.starts_with("END ") {
        blocks.push(cur.take().unwrap());
      }
    }
  }
  assert_eq!(blocks.len(), NUM_CASES, "number of cases in search_humansl.txt");
  let mut num_human = 0;
  for (line, expected) in &blocks {
    let actual = run_case(line);
    compare_lines(line, &actual, expected);
    if field(line, "human") == "1" {
      num_human += 1;
    }
    cases.ran();
  }
  assert!(num_human > 0);
  assert!(text.contains("\"humanPolicy\"") && text.contains("\"humanPrior\""));
  crate::report!("search_humansl_vs_upstream: {} cases ({} with a human evaluator)", blocks.len(), num_human);
}
}
