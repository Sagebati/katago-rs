//! Crafted-input oracle for the ported selection / weighting / utility helpers of search/*.rs and
//! Global::roundDynamic, against fixtures/upstream-v1.18.2/unit/oracle/search_select_diff.cpp (the unmodified
//! upstream functions called directly; format in its header): computeWeightFromNNOutput,
//! getApproxScoreUtilityDerivative, getScoreUtility(Diff), chooseIndexWithTemperature,
//! downweightBadChildrenAndNormalizeWeight, pruneNoiseWeight (binding noisePruningCap),
//! getExploreSelectionValue(Inverse), shouldSuppressPass (crafted root ownership and child stats after a small search), roundDynamic (incl. infinite / NaN arguments: GCC's folding of the overflowing
//! `orderOfMagnitude - precision >= 0`) and getSelfUtilityLCBAndRadiusZeroVisits. Each line holds its inputs; this test
//! recomputes the outputs with the port and the whole text must match byte for byte (`common::compare_lines`).
//! Also on wasm32 (`just test-wasm`). Not a port of upstream code.

#[macro_use]
mod common;

use std::fmt::Write as _;
use std::rc::Rc;

use common::compare_lines;
use common::fixture;
use common::run_or_stall_panic;
use futures::executor::LocalPool;
use katago_core::neuralnet::nninterface::BatchPolicy;
use katago_core::core::config_parser::ConfigParser;
use katago_core::core::commontypes::enabled_t;
use katago_core::core::global::Global;
use katago_core::core::global::KataError;
use katago_core::core::rand::Rand;
use katago_core::game::board::*;
use katago_core::neuralnet::desc::ModelDesc;
use katago_core::neuralnet::nneval::NNEvaluator;
use katago_core::neuralnet::nneval::NNResultBuf;
use katago_core::game::boardhistory::BoardHistory;
use katago_core::game::boardhistory::BoardHistoryModes;
use katago_core::game::rules::Rules;
use katago_core::neuralnet::nninputs::NNOutput;
use katago_core::neuralnet::nninputs::NNPos;
use katago_core::neuralnet::nninputs::ScoreValue;
use katago_core::neuralnet::nninterface::*;
use katago_core::port::executor::LocalBoxFuture;
use katago_core::port::executor::Spawner;
use katago_core::search::search::Search;
use katago_core::search::searchnode::MoreNodeStats;
use katago_core::search::searchparams::SearchParams;

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
  fn create_compute_context(&self, _: &[i32], _: i32, _: i32, _: &str, _: enabled_t, _: &dyn LoadedModel, _: &mut ConfigParser) -> Result<Box<dyn ComputeContext>, KataError> {
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
  fn get_output<'a>(&'a self, _: &'a mut dyn ComputeHandle, _: &'a mut dyn InputBuffers, _: i32, _: &'a [Rc<NNResultBuf>], _: &'a mut [NNOutput]) -> LocalBoxFuture<'a, Result<(), KataError>> {
    panic!("debugSkipNeuralNet")
  }
}

fn hd(s: &str) -> f64 {
  f64::from_bits(u64::from_str_radix(s, 16).expect("16 hex digits"))
}
fn hf(s: &str) -> f32 {
  f32::from_bits(u32::from_str_radix(s, 16).expect("8 hex digits"))
}
fn b(x: f64) -> String {
  format!("{:016X}", x.to_bits())
}

/// A Search as the oracle builds it (SearchParams(), numThreads 1, seed "revseed"), boxed in a small frame.
fn boxed_search(nn_eval: Rc<NNEvaluator>) -> Box<Search> {
  let mut params = SearchParams::new();
  params.num_threads = 1;
  Box::new(Search::new(&params, nn_eval, "revseed"))
}

/// Recomputes the outputs of one fixture line ("<op> <inputs> = <outputs>") and returns the whole line.
fn run_line(s: &mut Search, line: &str) -> String {
  let (inputs, _) = line.split_once(" = ").unwrap_or_else(|| panic!("no ' = ' in {:?}", line));
  let t: Vec<&str> = inputs.split(' ').collect();
  let mut o = String::new();
  o.push_str(inputs);
  o.push_str(" =");
  match t[0] {
    "W" => {
      s.search_params.use_uncertainty = true;
      s.search_params.uncertainty_exponent = hd(t[1]);
      s.search_params.uncertainty_coeff = hd(t[2]);
      s.search_params.uncertainty_max_weight = hd(t[3]);
      s.search_params.win_loss_utility_factor = hd(t[4]);
      s.search_params.static_score_utility_factor = hd(t[5]);
      s.search_params.dynamic_score_utility_factor = hd(t[6]);
      s.search_params.dynamic_score_center_scale = hd(t[7]);
      s.recent_score_center = hd(t[8]);
      let mut nn = NNOutput::new();
      nn.white_score_mean = hf(t[9]);
      nn.shortterm_winloss_error = hf(t[10]);
      nn.shortterm_score_error = hf(t[11]);
      let w = s.compute_weight_from_nn_output(&nn);
      let der = s.get_approx_score_utility_derivative(nn.white_score_mean as f64);
      let _ = write!(o, " {} {}", b(w), b(der));
    }
    "S" => {
      s.search_params.static_score_utility_factor = hd(t[1]);
      s.search_params.dynamic_score_utility_factor = hd(t[2]);
      s.search_params.dynamic_score_center_scale = hd(t[3]);
      s.recent_score_center = hd(t[4]);
      let u = s.get_score_utility(hd(t[5]), hd(t[6]));
      let ud = s.get_score_utility_diff(hd(t[5]), hd(t[6]), hd(t[7]));
      let _ = write!(o, " {} {}", b(u), b(ud));
    }
    "T" => {
      let n: usize = t[2].parse().expect("n");
      let temp = hd(t[3]);
      let ob = hd(t[4]);
      let ps: Vec<f64> = (0..n).map(|i| hd(t[5 + i])).collect();
      let mut rand = Rand::new_seed(t[1]);
      let mut buf = vec![0.0f64; Board::MAX_ARR_SIZE as usize];
      let idx = Search::choose_index_with_temperature(&mut rand, &ps, n as i32, temp, ob, Some(&mut buf[..]));
      let _ = write!(o, " {}", idx);
      for x in buf.iter().take(n) {
        let _ = write!(o, " {}", b(*x));
      }
      let _ = write!(o, " next={}", rand.next_uint());
    }
    "D" => {
      let vwe = hd(t[1]);
      let n: usize = t[2].parse().expect("n");
      let (sub, prune, cur, des) = (hd(t[3]), hd(t[4]), hd(t[5]), hd(t[6]));
      let mut st: Vec<MoreNodeStats> = (0..n).map(|_| MoreNodeStats::new()).collect();
      for i in 0..n {
        st[i].stats.visits = t[7 + 3 * i].parse().expect("visits");
        st[i].weight_adjusted = hd(t[8 + 3 * i]);
        st[i].self_utility = hd(t[9 + 3 * i]);
      }
      s.search_params.value_weight_exponent = vwe;
      s.downweight_bad_children_and_normalize_weight(n as i32, cur, des, sub, prune, &mut st);
      for x in &st {
        let _ = write!(o, " {}", b(x.weight_adjusted));
      }
    }
    "N" => {
      let (sc, cap) = (hd(t[1]), hd(t[2]));
      let n: usize = t[3].parse().expect("n");
      let tot = hd(t[4]);
      let mut st: Vec<MoreNodeStats> = (0..n).map(|_| MoreNodeStats::new()).collect();
      let mut pol: Vec<f64> = vec![0.0; n];
      for i in 0..n {
        st[i].self_utility = hd(t[5 + 3 * i]);
        st[i].weight_adjusted = hd(t[6 + 3 * i]);
        pol[i] = hd(t[7 + 3 * i]);
      }
      s.search_params.noise_prune_utility_scale = sc;
      s.search_params.noise_pruning_cap = cap;
      let r = s.prune_noise_weight(&mut st, n as i32, tot, &pol);
      let _ = write!(o, " {}", b(r));
      for x in &st {
        let _ = write!(o, " {}", b(x.weight_adjusted));
      }
    }
    "X" => {
      let pla: Player = t[6].parse::<i32>().expect("pla") as Player;
      let inv = s.get_explore_selection_value_inverse(hd(t[1]), hd(t[2]), hd(t[3]), hd(t[4]), pla);
      let fwd = s.get_explore_selection_value(hd(t[2]), hd(t[3]), hd(t[5]), hd(t[4]), pla);
      let _ = write!(o, " {} {}", b(inv), b(fwd));
    }
    "R" => {
      let prec: i32 = t[2].parse().expect("prec");
      let _ = write!(o, " {}", b(Global::round_dynamic(hd(t[1]), prec)));
    }
    "Z" => {
      s.search_params.win_loss_utility_factor = hd(t[1]);
      s.search_params.static_score_utility_factor = hd(t[2]);
      s.search_params.dynamic_score_utility_factor = hd(t[3]);
      s.search_params.lcb_stdevs = hd(t[4]);
      let (mut lcb, mut radius) = (0.0f64, 0.0f64);
      s.get_self_utility_lcb_and_radius_zero_visits(&mut lcb, &mut radius);
      let _ = write!(o, " {} {}", b(lcb), b(radius));
    }
    other => panic!("unknown op {}", other),
  }
  o.push('\n');
  o
}

const QOWN: [f32; 13] = [-0.999, -0.97, -0.951, -0.95, -0.6, 0.0, 0.6, 0.95, 0.951, 0.955, 0.96, 0.97, 0.999];
const QVIS: [i64; 10] = [0, 1, 3, 10, 40, 120, 499, 500, 501, 900];
const QWGT: [f64; 10] = [0.0, 1e-11, 0.5, 2.0, 5.0, 12.0, 40.0, 150.0, 600.0, 3000.0];
const QUP: [f64; 5] = [-0.4, -0.1, 0.0, 0.2, 0.45];
const QSP: [f64; 3] = [-3.0, 0.0, 2.5];
const QDU: [f64; 10] = [-0.3, -0.2, -0.15, -0.1001, -0.1, -0.0999, -0.05, 0.0, 0.05, 0.2];
const QDS: [f64; 7] = [-0.6, -0.5, -0.4, 0.0, 0.4, 0.5, 0.6];

/// The QPOS search: 5x5 japanese rules, komi 6.5, pos 0 empty (black to move), pos 1 a black center stone.
fn q_search(pool: &mut LocalPool, nn_eval: Rc<NNEvaluator>, pos: i32) -> (Box<Search>, String) {
  let mut qp = SearchParams::new();
  qp.num_threads = 1;
  qp.max_visits = 300;
  qp.fill_dame_before_pass = true;
  qp.cpuct_exploration = 20.0;
  let mut q = Box::new(Search::new(&qp, nn_eval, &format!("qseed{}", pos)));
  let mut rules = Rules::parse_rules("japanese").expect("japanese");
  rules.komi = 6.5;
  let mut board = Board::new_x_y(5, 5).expect("5x5");
  let mut hist = Box::new(BoardHistory::new_full(&board, P_BLACK, &rules, 0, &BoardHistoryModes::new()));
  if pos == 1 {
    hist.make_board_move_assume_legal(&mut board, Location::get_loc(2, 2, 5), P_BLACK, None, false);
  }
  let pla: Player = hist.presumed_next_move_pla;
  q.set_position(pla, &board, &hist).expect("setPosition");
  run_or_stall_panic(pool, "q search", q.run_whole_search(pla)).expect("q search");
  let root = q.get_root_node().expect("root");
  let children = root.get_children();
  let (mut n, mut pass_idx) = (0, -1);
  let mut moves: Vec<String> = Vec::new();
  for i in 0..children.get_capacity() {
    if children.index(i).get_if_allocated().is_none() {
      break;
    }
    let mv: Loc = children.index(i).get_move_loc_relaxed();
    if mv == Board::PASS_LOC {
      pass_idx = i;
    }
    moves.push(mv.to_string());
    n += 1;
  }
  let line = format!("QPOS {} pla={} children={} pass={} moves={}\n", pos, pla, n, pass_idx, moves.join(","));
  drop(root);
  (q, line)
}

/// One Q line on the QPOS search (format in search_select_diff.cpp).
fn run_q(q: &Search, line: &str) -> String {
  let (inputs, _) = line.split_once(" = ").expect("' = '");
  let t: Vec<&str> = inputs.split(' ').collect();
  let root = q.get_root_node().expect("root");
  let mut copy: NNOutput = (*root.get_nn_output().expect("root nnOutput")).clone();
  let own: &[u8] = t[2].as_bytes();
  {
    let nn_x_len = copy.nn_x_len;
    let map = copy.white_owner_map.as_mut().expect("root ownership");
    for y in 0..5 {
      for x in 0..5 {
        map[NNPos::xy_to_pos(x, y, nn_x_len) as usize] = QOWN[(own[(y * 5 + x) as usize] - b'a') as usize];
      }
    }
  }
  *root.nn_output.borrow_mut() = Some(Rc::new(copy));
  let d = |s: &str, k: usize| -> usize { (s.as_bytes()[k] - b'0') as usize };
  let pc: &str = t[3];
  let children = root.get_children();
  let mut codes = t[4..].iter();
  for i in 0..children.get_capacity() {
    let cp = children.index(i);
    let child_idx = match cp.get_if_allocated() {
      None => break,
      Some(c) => c,
    };
    let child = q.node_arena.get(child_idx);
    if cp.get_move_loc_relaxed() == Board::PASS_LOC {
      child.stats.visits.set(QVIS[d(pc, 0)]);
      cp.set_edge_visits(QVIS[d(pc, 0)]);
      child.stats.weight_sum.set(QWGT[d(pc, 1)]);
      child.stats.utility_avg.set(QUP[d(pc, 2)]);
      child.stats.score_mean_avg.set(QSP[d(pc, 3)]);
      child.stats.lead_avg.set(QSP[d(pc, 4)]);
      continue;
    }
    let c: &str = codes.next().expect("a code per non-pass child");
    child.stats.visits.set(QVIS[d(c, 0)]);
    cp.set_edge_visits(QVIS[d(c, 1)]);
    child.stats.weight_sum.set(QWGT[d(c, 2)]);
    child.stats.utility_avg.set(QUP[d(pc, 2)] + QDU[d(c, 3)]);
    child.stats.score_mean_avg.set(QSP[d(pc, 3)] + QDS[d(c, 4)]);
    child.stats.lead_avg.set(QSP[d(pc, 4)] + QDS[d(c, 5)]);
  }
  assert!(codes.next().is_none(), "more codes than children");
  let sup: bool = q.should_suppress_pass(Some(&*root));
  format!("{} = {}\n", inputs, sup as i32)
}

wasm_test! {
fn search_select_diff() {
  common::init();
  ScoreValue::init_tables();
  katago_core::core::timer::set_clock(common::fixed_clock);
  let expected: String = fixture("oracle", "search_select_diff.txt");
  let mut pool = LocalPool::new();
  let spawner: Rc<dyn Spawner> = Rc::new(pool.spawner());
  let backend: Rc<dyn NeuralNet> = Rc::new(NoBackend);
  let nn_eval = Rc::new(
    NNEvaluator::new(
      "search_nonn",
      "search_nonn",
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
      "runSearchTestsRandSeedrev",
      false,
      0,
      &mut ConfigParser::new(false, true),
      backend,
      spawner,
      Vec::new(),
    )
    .expect("NNEvaluator::new"),
  );
  run_or_stall_panic(&mut pool, "spawnServerThreads", nn_eval.spawn_server_threads()).expect("spawnServerThreads");
  let mut s = boxed_search(nn_eval.clone());

  let mut actual = String::new();
  let mut lines = expected.lines();
  let header = lines.next().expect("header line");
  assert!(header.starts_with("sqrtBoardArea "), "{}", header);
  let _ = writeln!(actual, "sqrtBoardArea {} supports {}", b(s.root_board.sqrt_board_area()), nn_eval.supports_shortterm_error() as i32);
  let mut counts: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
  let mut q: Option<Box<Search>> = None;
  for line in lines {
    if let Some(rest) = line.strip_prefix("QPOS ") {
      let pos: i32 = rest.split(' ').next().expect("pos").parse().expect("pos");
      drop(q.take());
      let (qs, qline) = q_search(&mut pool, nn_eval.clone(), pos);
      actual.push_str(&qline);
      q = Some(qs);
    } else if line.starts_with("Q ") {
      actual.push_str(&run_q(q.as_deref().expect("QPOS first"), line));
    } else {
      actual.push_str(&run_line(&mut s, line));
    }
    *counts.entry(line.split(' ').next().unwrap_or("").to_string()).or_insert(0) += 1;
  }
  drop(q);
  let total: usize = counts.values().sum();
  crate::report!("search_select_diff: {} lines by op {:?}", total, counts);
  assert!(total >= 3000, "{} lines", total);
  compare_lines("search_select_diff", &actual, &expected);

  drop(s);
  nn_eval.kill_server_threads();
  pool.run_until_stalled();
}
}
