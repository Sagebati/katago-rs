//! Crafted-input oracle for the ported eval cache update and adjust paths (search/evalcache.rs, searchupdatehelpers.rs,
//! searchexplorehelpers.rs) against fixtures/upstream-v1.18.2/unit/oracle/search_evalcache_diff.cpp (the unmodified upstream
//! functions called directly on hand-set statistics in which lead differs from score mean; format in its header):
//! adjustEvalsFromCacheHelper, EvalCacheTable::update (incl. the root pass rule), addCurrentNNOutputAsLeafValue and
//! recomputeNodeStats with an eval cache entry, and the firstExploreEvals loop of selectBestChildToDescend. Each line holds
//! its inputs; this test sets the same state, recomputes the outputs with the port and the whole text must match byte for
//! byte (`common::compare_lines`). Also on wasm32 (`just test-wasm`). Not a port of upstream code.

// Test code keeps the shapes of upstream's tests and oracle programs (PORTING.md §11b).
#![allow(clippy::identity_op, clippy::too_many_arguments)]

#[macro_use]
mod common;

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::rc::Rc;

use common::compare_lines;
use common::fixture;
use common::run_or_stall_panic;
use futures::executor::LocalPool;
use katago_core::core::commontypes::enabled_t;
use katago_core::core::global::KataError;
use katago_core::core::hash::Hash128;
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
use katago_core::search::evalcache::EvalCacheEntry;
use katago_core::search::evalcache::EvalCacheTable;
use katago_core::search::evalcache::FirstExploreEval;
use katago_core::search::search::Search;
use katago_core::search::search::SearchThread;
use katago_core::search::searchnode::SearchNode;
use katago_core::search::searchparams::SearchParams;

/// debugSkipNeuralNet: the backend is never reached.
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
  fn create_compute_context(&self, _: &[i32], _: i32, _: i32, _: &str, _: &str, _: bool, _: enabled_t, _: enabled_t, _: &dyn LoadedModel) -> Result<Box<dyn ComputeContext>, KataError> {
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
fn hu(s: &str) -> u64 {
  u64::from_str_radix(s, 16).expect("16 hex digits")
}
fn b(x: f64) -> String {
  format!("{:016X}", x.to_bits())
}
fn bf(x: f32) -> String {
  format!("{:08X}", x.to_bits())
}
fn int(s: &str) -> i64 {
  s.parse().unwrap_or_else(|_| panic!("integer {:?}", s))
}

const SZ: i32 = 5;

/// upstream harness makeSearch: position `pos` on a 5x5 board, searched for `max_visits`.
fn make_search(pool: &mut LocalPool, nn_eval: Rc<NNEvaluator>, pos: i32, max_visits: i64, cpuct: f64, seed: &str) -> Box<Search> {
  let mut qp = SearchParams::new();
  qp.num_threads = 1;
  qp.max_visits = max_visits;
  qp.cpuct_exploration = cpuct;
  qp.use_eval_cache = true;
  qp.use_graph_search = true;
  qp.eval_cache_min_visits = 2;
  let mut q = Box::new(Search::new(&qp, nn_eval, seed));
  let mut rules = Rules::parse_rules("japanese").expect("japanese");
  rules.komi = 6.5;
  let mut board = Board::new_x_y(SZ, SZ).expect("5x5");
  let mut hist = Box::new(BoardHistory::new_full(&board, P_BLACK, &rules, 0, &BoardHistoryModes::new()));
  let mut play = |x: i32, y: i32, pla: Player| {
    hist.make_board_move_assume_legal(&mut board, Location::get_loc(x, y, SZ), pla, None, false);
  };
  if pos == 1 {
    play(2, 2, P_BLACK);
  }
  if pos == 2 {
    play(1, 1, P_BLACK);
    play(3, 3, P_WHITE);
    play(3, 1, P_BLACK);
    play(1, 3, P_WHITE);
  }
  let pla: Player = hist.presumed_next_move_pla;
  q.set_position(pla, &board, &hist).expect("setPosition");
  run_or_stall_panic(pool, "search", q.run_whole_search(pla)).expect("search");
  q
}

/// (number of allocated children, index of the pass child or -1)
fn kids_of(q: &Search, node: &SearchNode) -> (i32, i32) {
  let _ = q;
  let ch = node.get_children();
  let (mut n, mut pass_idx) = (0, -1);
  for i in 0..ch.get_capacity() {
    if ch.index(i).get_if_allocated().is_none() {
      break;
    }
    if ch.index(i).get_move_loc_relaxed() == Board::PASS_LOC {
      pass_idx = i;
    }
    n += 1;
  }
  (n, pass_idx)
}

/// The root child with the most edge visits that has children of its own (-1 if none).
fn best_child_with_kids(q: &Search, root: &SearchNode) -> i32 {
  let ch = root.get_children();
  let mut best = -1;
  let mut best_visits: i64 = -1;
  for i in 0..ch.get_capacity() {
    let c = match ch.index(i).get_if_allocated() {
      None => break,
      Some(c) => q.node_arena.get(c),
    };
    if c.get_nn_output().is_none() {
      continue;
    }
    if kids_of(q, &c).0 == 0 && best >= 0 {
      continue;
    }
    let ev = ch.index(i).get_edge_visits();
    if ev > best_visits {
      best_visits = ev;
      best = i;
    }
  }
  best
}

fn set_nn_values(node: &SearchNode, wp: f32, lp: f32, np: f32, sm: f32, smsq: f32, lead: f32) {
  let mut copy: NNOutput = (*node.get_nn_output().expect("nnOutput")).clone();
  copy.white_win_prob = wp;
  copy.white_loss_prob = lp;
  copy.white_no_result_prob = np;
  copy.white_score_mean = sm;
  copy.white_score_mean_sq = smsq;
  copy.white_lead = lead;
  *node.nn_output.borrow_mut() = Some(Rc::new(copy));
}

fn set_stats_all(n: &SearchNode, visits: i64, wl: f64, nr: f64, sm: f64, smsq: f64, lead: f64, util: f64, utilsq: f64, ws: f64, wsq: f64) {
  n.stats.visits.set(visits);
  n.stats.win_loss_value_avg.set(wl);
  n.stats.no_result_value_avg.set(nr);
  n.stats.score_mean_avg.set(sm);
  n.stats.score_mean_sq_avg.set(smsq);
  n.stats.lead_avg.set(lead);
  n.stats.utility_avg.set(util);
  n.stats.utility_sq_avg.set(utilsq);
  n.stats.weight_sum.set(ws);
  n.stats.weight_sq_sum.set(wsq);
}

fn stats_str(n: &SearchNode) -> String {
  format!(
    "{} {} {} {} {} {} {} {} {} {}\n",
    n.stats.visits.get(),
    b(n.stats.win_loss_value_avg.get()),
    b(n.stats.no_result_value_avg.get()),
    b(n.stats.score_mean_avg.get()),
    b(n.stats.score_mean_sq_avg.get()),
    b(n.stats.lead_avg.get()),
    b(n.stats.utility_avg.get()),
    b(n.stats.utility_sq_avg.get()),
    b(n.stats.weight_sum.get()),
    b(n.stats.weight_sq_sum.get())
  )
}

/// The Prm of the harness: 13 tokens.
struct Prm {
  min_visits: i64,
  wlf: f64,
  nrw: f64,
  ssf: f64,
  dsf: f64,
  dscale: f64,
  rsc: f64,
  e_wl: f32,
  e_nr: f32,
  e_sm: f32,
  e_lead: f32,
  e_w: f32,
}

fn parse_prm(t: &[&str]) -> Prm {
  Prm {
    min_visits: int(t[0]),
    wlf: hd(t[1]),
    nrw: hd(t[2]),
    ssf: hd(t[3]),
    dsf: hd(t[4]),
    dscale: hd(t[5]),
    rsc: hd(t[6]),
    e_wl: hf(t[7]),
    e_nr: hf(t[8]),
    e_sm: hf(t[9]),
    e_lead: hf(t[10]),
    e_w: hf(t[11]),
  }
}
const PRM_TOKENS: usize = 12;

fn set_prm(s: &mut Search, p: &Prm) {
  s.search_params.eval_cache_min_visits = p.min_visits;
  s.search_params.win_loss_utility_factor = p.wlf;
  s.search_params.no_result_utility_for_white = p.nrw;
  s.search_params.static_score_utility_factor = p.ssf;
  s.search_params.dynamic_score_utility_factor = p.dsf;
  s.search_params.dynamic_score_center_scale = p.dscale;
  s.recent_score_center = p.rsc;
}

fn make_entry(p: &Prm) -> Rc<EvalCacheEntry> {
  let mut e = EvalCacheEntry::new();
  e.avg_win_loss = p.e_wl;
  e.avg_no_result = p.e_nr;
  e.avg_score_mean = p.e_sm;
  e.avg_lead = p.e_lead;
  e.cache_weight = p.e_w;
  Rc::new(e)
}

fn entry_str(e: &Option<Rc<EvalCacheEntry>>) -> String {
  let e = match e {
    None => return "null".to_string(),
    Some(e) => e,
  };
  let mut o = format!("{} {} {} {} {} {}", bf(e.cache_weight), bf(e.avg_win_loss), bf(e.avg_no_result), bf(e.avg_score_mean), bf(e.avg_lead), e.first_explore_evals.len());
  for (loc, fe) in e.first_explore_evals.iter() {
    let _ = write!(o, " {} {} {} {}", loc, bf(fe.avg_win_loss), bf(fe.avg_score_mean), bf(fe.cache_weight));
  }
  o
}

fn apply_gate(s: &mut Search, gate: i64) {
  s.search_params.use_eval_cache = gate != 1;
  s.search_params.use_graph_search = gate != 2;
  s.mirroring_pla = if gate == 3 { P_BLACK } else { C_EMPTY };
}

fn run_a(s: &mut Search, t: &[&str]) -> String {
  let p = parse_prm(&t[1..]);
  set_prm(s, &p);
  let entry = make_entry(&p);
  let r = &t[1 + PRM_TOKENS..];
  let visits = int(r[0]);
  let (mut wl, mut nr, mut sm, mut smsq, mut lead) = (hd(r[1]), hd(r[2]), hd(r[3]), hd(r[4]), hd(r[5]));
  let has_util = r[6] == "1";
  let mut util = hd(r[7]);
  s.adjust_evals_from_cache_helper(&entry, visits, &mut wl, &mut nr, &mut sm, &mut smsq, &mut lead, if has_util { Some(&mut util) } else { None });
  let mut o = format!("{} {} {} {} {} ", b(wl), b(nr), b(sm), b(smsq), b(lead));
  if has_util {
    o.push_str(&b(util));
  } else {
    o.push('-');
  }
  o.push('\n');
  o
}

fn run_u(q: &Search, t: &[&str]) -> String {
  let root = q.get_root_node().expect("root");
  let ch = root.get_children();
  let (n, _) = kids_of(q, &root);
  let shards = int(t[2]) as i32;
  let steps = int(t[3]) as usize;
  let hashes = [Hash128::new(hu(t[5]), hu(t[4])), Hash128::new(hu(t[7]), hu(t[6]))];
  let table = EvalCacheTable::new(shards);
  let mut k = 8;
  let mut outs = String::new();
  for step in 0..steps {
    let which = int(t[k]) as usize;
    let min_v = int(t[k + 1]);
    let is_root = t[k + 2] == "1";
    let visits = int(t[k + 3]);
    let (wl, nr, sm, lead, util) = (hd(t[k + 4]), hd(t[k + 5]), hd(t[k + 6]), hd(t[k + 7]), hd(t[k + 8]));
    k += 9;
    set_stats_all(&root, visits, wl, nr, sm, sm * sm, lead, util, util * util, visits as f64, visits as f64);
    for i in 0..n {
      let cv = int(t[k]);
      let ce = int(t[k + 1]);
      let (cwl, csm, clead, cutil) = (hd(t[k + 2]), hd(t[k + 3]), hd(t[k + 4]), hd(t[k + 5]));
      k += 6;
      let c = q.node_arena.get(ch.index(i).get_if_allocated().expect("child"));
      c.stats.visits.set(cv);
      ch.index(i).set_edge_visits(ce);
      c.stats.win_loss_value_avg.set(cwl);
      c.stats.score_mean_avg.set(csm);
      c.stats.lead_avg.set(clead);
      c.stats.utility_avg.set(cutil);
    }
    table.update(&q.node_arena, hashes[which], &root, min_v, is_root);
    if step > 0 {
      outs.push('|');
    }
    outs.push_str(&entry_str(&table.find(hashes[which])));
  }
  assert_eq!(k, t.len() - 0, "tokens before ' = '");
  let sizes: Vec<String> = table.entries.borrow().iter().map(|m| m.len().to_string()).collect();
  format!(" = {} shards={}\n", outs, sizes.join(","))
}

fn nth_child(q: &Search, root: &Rc<SearchNode>, which: i64) -> Rc<SearchNode> {
  if which == 0 {
    return root.clone();
  }
  let xi = best_child_with_kids(q, root);
  let ch = root.get_children();
  q.node_arena.get(ch.index(if xi < 0 { 0 } else { xi }).get_if_allocated().expect("child"))
}

fn run_l(q: &mut Search, t: &[&str]) -> String {
  let root = q.get_root_node().expect("root");
  let which = int(t[2]);
  let gate = int(t[3]);
  let p = parse_prm(&t[4..]);
  set_prm(q, &p);
  apply_gate(q, gate);
  let node = nth_child(q, &root, if which == 0 { 0 } else { 1 });
  *node.eval_cache_entry.borrow_mut() = Some(make_entry(&p));
  let nn = &t[4 + PRM_TOKENS..];
  set_nn_values(&node, hf(nn[0]), hf(nn[1]), hf(nn[2]), hf(nn[3]), hf(nn[4]), hf(nn[5]));
  set_stats_all(&node, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
  q.add_current_nn_output_as_leaf_value(&node, true);
  let o = stats_str(&node);
  *node.eval_cache_entry.borrow_mut() = None;
  apply_gate(q, 0);
  o
}

fn run_c(q: &mut Search, t: &[&str]) -> String {
  let root = q.get_root_node().expect("root");
  let which = int(t[2]);
  let gate = int(t[3]);
  let node = nth_child(q, &root, if which == 0 { 0 } else { 1 });
  let p = parse_prm(&t[4..]);
  set_prm(q, &p);
  apply_gate(q, gate);
  *node.eval_cache_entry.borrow_mut() = Some(make_entry(&p));
  let nn = &t[4 + PRM_TOKENS..];
  set_nn_values(&node, hf(nn[0]), hf(nn[1]), hf(nn[2]), hf(nn[3]), hf(nn[4]), hf(nn[5]));
  let node_visits = int(nn[6]);
  set_stats_all(&node, node_visits, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
  let nch = node.get_children();
  let mut k = 7;
  for i in 0..nch.get_capacity() {
    let c = match nch.index(i).get_if_allocated() {
      None => break,
      Some(c) => q.node_arena.get(c),
    };
    let (cv, ce) = (int(nn[k]), int(nn[k + 1]));
    let v: Vec<f64> = (2..11).map(|j| hd(nn[k + j])).collect();
    // cw cwl cnr csm csmsq clead cutil cutilsq cwsq
    set_stats_all(&c, cv, v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[0], v[8]);
    nch.index(i).set_edge_visits(ce);
    k += 11;
  }
  assert_eq!(k, nn.len(), "child tokens");
  let mut thread = SearchThread::new_boxed(0, q);
  q.recompute_node_stats(&node, &mut thread, 1, which == 0);
  let o = stats_str(&node);
  *node.eval_cache_entry.borrow_mut() = None;
  apply_gate(q, 0);
  o
}

fn run_f(q: &mut Search, t: &[&str], xloc: Loc) -> String {
  let root = q.get_root_node().expect("root");
  let depth = int(t[2]);
  let gate = int(t[3]);
  let (wlf, ssf, dsf, dscale, rsc, cpuct) = (hd(t[4]), hd(t[5]), hd(t[6]), hd(t[7]), hd(t[8]), hd(t[9]));
  let sym_prune = t[10] == "1";
  let (avoid_spec, dup_spec, fe_spec) = (t[11], t[12], t[13]);
  let mut fes: BTreeMap<Loc, FirstExploreEval> = BTreeMap::new();
  if fe_spec != "-" {
    for item in fe_spec.split(',') {
      let f: Vec<&str> = item.split(':').collect();
      fes.insert(f[0].parse().expect("loc"), FirstExploreEval::new_full(hf(f[1]), hf(f[2]), hf(f[3])));
    }
  }
  q.search_params.win_loss_utility_factor = wlf;
  q.search_params.static_score_utility_factor = ssf;
  q.search_params.dynamic_score_utility_factor = dsf;
  q.search_params.dynamic_score_center_scale = dscale;
  q.recent_score_center = rsc;
  q.search_params.cpuct_exploration = cpuct;
  q.search_params.root_symmetry_pruning = sym_prune;
  apply_gate(q, gate);
  let mut avoid_vec: Vec<i32> = Vec::new();
  if avoid_spec != "-" {
    avoid_vec = vec![0; Board::MAX_ARR_SIZE as usize];
    for item in avoid_spec.split(',') {
      let (l, u) = item.split_once('/').expect("loc/until");
      avoid_vec[l.parse::<usize>().expect("loc")] = u.parse().expect("until");
    }
  }
  q.avoid_move_until_by_loc_black = avoid_vec.clone();
  q.avoid_move_until_by_loc_white = avoid_vec;
  for k in 0..Board::MAX_ARR_SIZE as usize {
    q.root_sym_dup_loc[k] = false;
  }
  if dup_spec != "-" {
    for l in dup_spec.split(',') {
      q.root_sym_dup_loc[l.parse::<usize>().expect("loc")] = true;
    }
  }
  let node: Rc<SearchNode> = nth_child(q, &root, depth);
  let mut o = String::new();
  for _call in 0..8 {
    let mut entry = EvalCacheEntry::new();
    entry.first_explore_evals = fes.clone();
    *node.eval_cache_entry.borrow_mut() = Some(Rc::new(entry));
    let mut thread = SearchThread::new_boxed(0, q);
    if depth == 1 {
      let pla = thread.pla;
      thread.history.make_board_move_assume_legal(&mut thread.board, xloc, pla, Some(&*q.root_ko_hash_table), false);
      thread.pla = get_opp(pla);
    }
    let (mut num_found, mut best_idx, mut best_loc, mut count_edge_visit) = (0i32, -1i32, Board::NULL_LOC, false);
    q.select_best_child_to_descend(&mut thread, &node, node.state.get(), &mut num_found, &mut best_idx, &mut best_loc, &mut count_edge_visit, depth == 0);
    let _ = write!(o, " {}/{}/{}/{}", num_found, best_idx, best_loc, count_edge_visit as i32);
    if best_idx == num_found && fes.contains_key(&best_loc) {
      fes.remove(&best_loc);
    } else {
      break;
    }
  }
  o.push('\n');
  *node.eval_cache_entry.borrow_mut() = None;
  apply_gate(q, 0);
  o
}

wasm_test! {
fn search_evalcache_diff() {
  common::init();
  ScoreValue::init_tables();
  katago_core::core::timer::set_clock(common::fixed_clock);
  let expected: String = fixture("oracle", "search_evalcache_diff.txt");
  let mut pool = LocalPool::new();
  let spawner: Rc<dyn Spawner> = Rc::new(pool.spawner());
  let backend: Rc<dyn NeuralNet> = Rc::new(NoBackend);
  let nn_eval = Rc::new(
    NNEvaluator::new(
      "search_nonn", "search_nonn", "", 16, 19, 19, false, true, 16, 12, true, "", "", false, enabled_t::False, enabled_t::False, 1, &[0],
      "runSearchTestsRandSeedrev", false, 0, backend, spawner, Vec::new(),
    )
    .expect("NNEvaluator::new"),
  );
  run_or_stall_panic(&mut pool, "spawnServerThreads", nn_eval.spawn_server_threads()).expect("spawnServerThreads");
  let mut params = SearchParams::new();
  params.num_threads = 1;
  let mut s = Box::new(Search::new(&params, nn_eval.clone(), "ecdiffseed"));

  let mut actual = String::new();
  let mut lines = expected.lines();
  let header = lines.next().expect("header line");
  assert!(header.starts_with("supports "), "{}", header);
  let _ = writeln!(actual, "supports {}", nn_eval.supports_shortterm_error() as i32);
  let mut counts: BTreeMap<String, usize> = BTreeMap::new();
  let mut q: Option<Box<Search>> = None;
  let mut xloc: Loc = Board::NULL_LOC;
  for line in lines {
    let (inputs, _) = match line.split_once(" = ") {
      Some((i, o)) => (i, o),
      None => (line, ""),
    };
    let t: Vec<&str> = inputs.split(' ').collect();
    match t[0] {
      "A" => {
        actual.push_str(inputs);
        actual.push_str(" = ");
        actual.push_str(&run_a(&mut s, &t));
      }
      "UPOS" | "FPOS" => {
        let pos = int(t[1]) as i32;
        drop(q.take());
        let is_f = t[0] == "FPOS";
        let qs = if is_f {
          make_search(&mut pool, nn_eval.clone(), pos, 7, 1.0, &format!("fcseed{}", pos))
        } else {
          make_search(&mut pool, nn_eval.clone(), pos, 300, 20.0, &format!("ecseed{}", pos))
        };
        let root = qs.get_root_node().expect("root");
        let (n, pass_idx) = kids_of(&qs, &root);
        if is_f {
          let xi = best_child_with_kids(&qs, &root);
          let ch = root.get_children();
          let xnode = qs.node_arena.get(ch.index(xi).get_if_allocated().expect("x"));
          xloc = ch.index(xi).get_move_loc_relaxed();
          let _ = writeln!(actual, "FPOS {} pla={} children={} x={} xchildren={}", pos, qs.root_pla, n, xloc, kids_of(&qs, &xnode).0);
        } else {
          let _ = writeln!(actual, "UPOS {} pla={} children={} pass={}", pos, qs.root_pla, n, pass_idx);
        }
        drop(root);
        q = Some(qs);
      }
      "U" => {
        actual.push_str(inputs);
        actual.push_str(&run_u(q.as_deref().expect("UPOS first"), &t));
      }
      "L" => {
        actual.push_str(inputs);
        actual.push_str(" = ");
        actual.push_str(&run_l(q.as_deref_mut().expect("UPOS first"), &t));
      }
      "C" => {
        actual.push_str(inputs);
        actual.push_str(" = ");
        actual.push_str(&run_c(q.as_deref_mut().expect("UPOS first"), &t));
      }
      "F" => {
        actual.push_str(inputs);
        actual.push_str(" =");
        actual.push_str(&run_f(q.as_deref_mut().expect("FPOS first"), &t, xloc));
      }
      other => panic!("unknown op {}", other),
    }
    *counts.entry(t[0].to_string()).or_insert(0) += 1;
  }
  drop(q);
  let total: usize = counts.values().sum();
  crate::report!("search_evalcache_diff: {} lines by op {:?}", total, counts);
  assert!(total >= 1800, "{} lines", total);
  compare_lines("search_evalcache_diff", &actual, &expected);

  drop(s);
  nn_eval.kill_server_threads();
  pool.run_until_stalled();
}
}
