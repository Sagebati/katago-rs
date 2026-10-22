//! Real-net search TOLERANCE study: the ported `Search` over the ported `NNEvaluator` and the burn `flex` backend
//! vs the unmodified upstream Search over the Eigen fp32 NNEvaluator on the same net, on 160 searches (g170-b6c96)
//! and 32 searches (g170e-b10c128) at 400-1000 visits over varied positions and SearchParams. Oracle:
//! fixtures/upstream-v1.16.5/search/oracle/search_tolerance.cpp -> search_tolerance_{b6c96,b10c128}.txt (each CASE
//! line holds everything needed to rebuild the search; the lines after it are the upstream result). Not a port of
//! upstream code. A larger sibling of search_smoke.rs (same helpers and fixture format).
//!
//! burn's conv accumulation order differs from Eigen's, so the NN outputs differ by ~1e-5 (nn_parity.rs) and those
//! differences can change which child the search picks, hence this is a tolerance comparison, not byte-exact. It is
//! a MEASUREMENT: it prints per net and per maxVisits bucket
//!   - the number of searches,
//!   - the rate of identical root visit vectors (move + edge visits of every root child, children order),
//!   - the rate of identical chosen moves (getChosenMoveLoc, chosenMoveTemperature 0),
//!   - the rate of identical root visit counts,
//!   - max and median over the searches of the per-search max |burn - eigen| of winLossValue (children and root
//!     values), utility, scoreMean, lead and policy prior (over the children both trees have),
//!   - for the searches whose visit vector differs, the max and median of the largest edge-visit difference of any
//!     child (a child missing on one side counts as 0 visits).
//! The test FAILS only on gross problems: a root visit count differs, the chosen move differs in more than 10% of the
//! searches, or a value difference exceeds the loose bounds below (those of search_smoke.rs). Run in release (debug
//! burn is very slow; the tests are ignored in debug):
//!   `cargo test --release -p katago-nn-burn --test search_tolerance -- --nocapture`
#![cfg(all(feature = "flex", not(target_arch = "wasm32")))]

use std::collections::BTreeMap;
use std::collections::HashMap;
use std::fmt::Write as _;
use std::future::Future;
use std::rc::Rc;
use std::time::Instant;

use burn::prelude::*;
use futures::executor::LocalPool;

use katago_core::core::commontypes::enabled_t;
use katago_core::core::timer::set_clock;
use katago_core::game::board::Board;
use katago_core::game::board::Loc;
use katago_core::game::board::Location;
use katago_core::game::boardhistory::BoardHistory;
use katago_core::game::rules::Rules;
use katago_core::neuralnet::nneval::NNEvaluator;
use katago_core::neuralnet::nninputs::ScoreValue;
use katago_core::port::executor::Spawner;
use katago_core::search::analysisdata::AnalysisData;
use katago_core::search::reportedsearchvalues::ReportedSearchValues;
use katago_core::search::search::Search;
use katago_core::search::searchparams::SearchParams;
use katago_nn_burn::Network;
use katago_nn_burn::backend::BurnNeuralNet;

/// Loose bounds on the max |burn - eigen| of a search (after the 1e-5-level NN differences went through the search;
/// children whose visit counts differ between the two trees are compared too).
const TOL_WINRATE_UTILITY: f64 = 1e-2;
const TOL_SCORE: f64 = 0.5;
const TOL_PRIOR: f64 = 1e-2;
/// Minimum fraction of searches with the same chosen move.
const MIN_SAME_TOP: f64 = 0.9;

//------------------------------------------------------------------------------------------------
// Driving futures (as nn_parity.rs: flex completes everything inside the pool)

fn run_watchdog<F: Future>(pool: &mut LocalPool, label: &str, fut: F) -> F::Output {
  use std::sync::Arc;
  use std::sync::atomic::AtomicBool;
  use std::sync::atomic::Ordering;
  use std::task::Context;
  use std::task::Poll;
  struct Flag(AtomicBool);
  impl futures::task::ArcWake for Flag {
    fn wake_by_ref(arc_self: &Arc<Self>) {
      arc_self.0.store(true, Ordering::SeqCst);
    }
  }
  let flag = Arc::new(Flag(AtomicBool::new(true)));
  let waker = futures::task::waker(flag.clone());
  let mut cx = Context::from_waker(&waker);
  let mut fut = std::pin::pin!(fut);
  loop {
    flag.0.store(false, Ordering::SeqCst);
    if let Poll::Ready(v) = fut.as_mut().poll(&mut cx) {
      return v;
    }
    pool.run_until_stalled();
    if !flag.0.load(Ordering::SeqCst) {
      panic!("{label}: the executor stalled with the future pending and not woken (lost wakeup)");
    }
  }
}

fn fake_clock() -> f64 {
  1000.0
}

//------------------------------------------------------------------------------------------------
// The fixture

#[derive(Clone, Debug)]
struct ChildLine {
  mv: String,
  loc: Loc,
  edge: i64,
  visits: i64,
  wl: f64,
  sm: f64,
  lead: f64,
  util: f64,
  prior: f64,
}

/// What one search reports (the oracle's output lines, or the same read off the port's Search).
#[derive(Clone, Debug)]
struct SearchResult {
  root_visits: i64,
  children: Vec<ChildLine>,
  /// win, loss, expectedScore, lead, winLossValue, utility
  rv: [f64; 6],
  chosen: Loc,
  chosen_str: String,
}

struct Case {
  idx: usize,
  name: String,
  xs: i32,
  ys: i32,
  rules: Rules,
  ipla: i8,
  pla: i8,
  moves: Vec<(Loc, i8)>,
  sseed: String,
  base: i32,
  max_visits: i64,
  params: Vec<(String, String)>,
  expected: SearchResult,
}

fn fixture_path(file: &str) -> String {
  format!("{}/../../fixtures/upstream-v1.16.5/search/oracle/{file}", env!("CARGO_MANIFEST_DIR"))
}

fn parse_cases(file: &str) -> Vec<Case> {
  let text = std::fs::read_to_string(fixture_path(file)).unwrap();
  let mut cases: Vec<Case> = Vec::new();
  let mut lines = text.lines().filter(|l| !l.is_empty() && !l.starts_with('#'));
  while let Some(line) = lines.next() {
    assert!(line.starts_with("CASE "), "expected a CASE line, got: {line}");
    let kv: HashMap<&str, &str> = line.split_whitespace().skip(2).filter_map(|t| t.split_once('=')).collect();
    let idx: usize = line.split_whitespace().nth(1).unwrap().parse().unwrap();
    assert_eq!(idx, cases.len());
    let int = |k: &str| -> i32 { kv[k].parse().unwrap() };
    let (xs, ys) = kv["board"].split_once('x').unwrap();
    let rules = Rules::new_full(
      int("ko"),
      int("score"),
      int("tax"),
      int("suicide") != 0,
      int("button") != 0,
      int("whb"),
      int("fpo") != 0,
      f32::from_bits(u32::from_str_radix(kv["komi"], 16).unwrap()),
    );
    let moves: Vec<(Loc, i8)> = if kv["moves"] == "-" {
      Vec::new()
    } else {
      kv["moves"].split(',').map(|m| m.split_once('/').map(|(l, p)| (l.parse().unwrap(), p.parse().unwrap())).unwrap()).collect()
    };
    let params: Vec<(String, String)> = if kv["params"] == "-" {
      Vec::new()
    } else {
      kv["params"].split(',').map(|p| p.split_once('=').map(|(k, v)| (k.to_string(), v.to_string())).unwrap()).collect()
    };

    // The result block.
    let mut res = SearchResult { root_visits: -1, children: Vec::new(), rv: [f64::NAN; 6], chosen: Board::NULL_LOC, chosen_str: String::new() };
    loop {
      let l = lines.next().expect("END line");
      if l.starts_with("END ") {
        assert_eq!(l[4..].parse::<usize>().unwrap(), idx);
        break;
      }
      let t: Vec<&str> = l.split_whitespace().collect();
      let field = |t: &[&str], key: &str| -> String { t.iter().find_map(|x| x.strip_prefix(key)).unwrap_or_else(|| panic!("{key} in {l}")).to_string() };
      let num = |t: &[&str], key: &str| -> f64 { field(t, key).parse().unwrap() };
      match t[0] {
        "root" => res.root_visits = field(&t, "visits=").parse().unwrap(),
        "c" => res.children.push(ChildLine {
          mv: t[2].to_string(),
          loc: field(&t, "loc=").parse().unwrap(),
          edge: field(&t, "edge=").parse().unwrap(),
          visits: field(&t, "visits=").parse().unwrap(),
          wl: num(&t, "wl="),
          sm: num(&t, "sm="),
          lead: num(&t, "lead="),
          util: num(&t, "util="),
          prior: num(&t, "prior="),
        }),
        _ if t[0].starts_with("children=") => assert_eq!(t[0]["children=".len()..].parse::<usize>().unwrap(), res.children.len()),
        "RV" => res.rv = [num(&t, "win="), num(&t, "loss="), num(&t, "es="), num(&t, "lead="), num(&t, "wl="), num(&t, "util=")],
        "chosen" => {
          res.chosen = field(&t, "loc=").parse().unwrap();
          res.chosen_str = t[2].to_string();
        }
        other => panic!("unknown line {other}: {l}"),
      }
    }
    cases.push(Case {
      idx,
      name: kv["name"].to_string(),
      xs: xs.parse().unwrap(),
      ys: ys.parse().unwrap(),
      rules,
      ipla: int("ipla") as i8,
      pla: int("pla") as i8,
      moves,
      sseed: kv["sseed"].to_string(),
      base: int("base"),
      max_visits: kv["maxvisits"].parse().unwrap(),
      params,
      expected: res,
    });
  }
  cases
}

//------------------------------------------------------------------------------------------------
// Running a search through the port

fn set_param(p: &mut SearchParams, k: &str, v: &str) {
  match k {
    "useGraphSearch" => p.use_graph_search = v == "1",
    "rootSymmetryPruning" => p.root_symmetry_pruning = v == "1",
    "wideRootNoise" => p.wide_root_noise = v.parse().unwrap(),
    "rootPolicyTemperature" => p.root_policy_temperature = v.parse().unwrap(),
    "rootPolicyTemperatureEarly" => p.root_policy_temperature_early = v.parse().unwrap(),
    "conservativePass" => p.conservative_pass = v == "1",
    "chosenMoveTemperature" => p.chosen_move_temperature = v.parse().unwrap(),
    "chosenMoveTemperatureEarly" => p.chosen_move_temperature_early = v.parse().unwrap(),
    other => panic!("unknown param {other}"),
  }
}

fn run_search(pool: &mut LocalPool, nn_eval: &Rc<NNEvaluator>, c: &Case) -> SearchResult {
  let mut board = Board::new_x_y(c.xs, c.ys).unwrap();
  let mut hist = Box::new(BoardHistory::new_full(&board, c.ipla, &c.rules, 0));
  for &(loc, pla) in &c.moves {
    assert!(hist.is_legal(&board, loc, pla), "case {}: illegal move {loc}", c.idx);
    hist.make_board_move_assume_legal(&mut board, loc, pla, None, false);
  }
  assert_eq!(hist.presumed_next_move_pla, c.pla, "case {}", c.idx);

  let mut params = if c.base == 1 { SearchParams::for_tests_v1() } else { SearchParams::new() };
  params.num_threads = 1;
  set_param(&mut params, "chosenMoveTemperature", "0");
  set_param(&mut params, "chosenMoveTemperatureEarly", "0");
  for (k, v) in &c.params {
    set_param(&mut params, k, v);
  }
  params.max_visits = c.max_visits;

  let mut search = Box::new(Search::new(&params, nn_eval.clone(), &c.sseed));
  search.set_position(c.pla, &board, &hist).unwrap();
  run_watchdog(pool, "runWholeSearch", search.run_whole_search(c.pla)).unwrap();

  let root = search.get_root_node().expect("root node");
  let mut ad: Vec<AnalysisData> = Vec::new();
  search.get_analysis_data(&mut ad, 0, false, 1, false);
  let by_move: HashMap<Loc, &AnalysisData> = ad.iter().map(|d| (d.move_, d)).collect();
  assert_eq!(by_move.len(), ad.len());
  let children = root.get_children();
  let mut out: Vec<ChildLine> = Vec::new();
  for i in 0..children.get_capacity() {
    let cp = children.index(i);
    let child_idx = match cp.get_if_allocated() {
      None => break,
      Some(ci) => ci,
    };
    let child = search.node_arena.get(child_idx);
    let mv = cp.get_move_loc();
    let d = by_move[&mv];
    assert_eq!(d.num_visits, cp.get_edge_visits());
    out.push(ChildLine {
      mv: Location::to_string(mv, search.get_root_board()),
      loc: mv,
      edge: cp.get_edge_visits(),
      visits: child.stats.visits.get(),
      wl: d.win_loss_value,
      sm: d.score_mean,
      lead: d.lead,
      util: d.utility,
      prior: d.policy_prior,
    });
  }
  assert_eq!(out.len(), ad.len());
  let mut v = ReportedSearchValues::new();
  assert!(search.get_root_values(&mut v));
  let chosen = search.get_chosen_move_loc();
  let result = SearchResult {
    root_visits: search.get_root_visits(),
    children: out,
    rv: [v.win_value, v.loss_value, v.expected_score, v.lead, v.win_loss_value, v.utility],
    chosen,
    chosen_str: Location::to_string(chosen, search.get_root_board()),
  };
  result
}

//------------------------------------------------------------------------------------------------
// Comparison

#[derive(Default, Clone, Debug)]
struct Diff {
  /// winLossValue (children and root values: win, loss, winLossValue)
  wl: f64,
  util: f64,
  /// scoreMean (children, root expectedScore)
  sm: f64,
  /// lead (children and root lead)
  lead: f64,
  prior: f64,
}

struct Outcome {
  idx: usize,
  name: String,
  base: i32,
  max_visits: i64,
  params: String,
  root_visits_equal: bool,
  same_visits: bool,
  same_top: bool,
  diff: Diff,
  /// Largest |edge visits burn - eigen| of any child (a child missing on one side counts as 0).
  max_visit_diff: i64,
  /// First differing child of the visit vector, if it differs.
  first_div: Option<String>,
  top_note: String,
  seconds: f64,
}

fn compare(c: &Case, got: &SearchResult, seconds: f64) -> Outcome {
  let want = &c.expected;
  let root_visits_equal = got.root_visits == want.root_visits;
  let same_visits = got.children.len() == want.children.len() && got.children.iter().zip(&want.children).all(|(g, w)| g.loc == w.loc && g.edge == w.edge);
  let first_div = if same_visits {
    None
  } else {
    let n = got.children.len().min(want.children.len());
    let k = (0..n).find(|&i| got.children[i].loc != want.children[i].loc || got.children[i].edge != want.children[i].edge);
    let show = |ch: Option<&ChildLine>| -> String { ch.map_or("<none>".to_string(), |ch| format!("{} edge {} (visits {})", ch.mv, ch.edge, ch.visits)) };
    let i = k.unwrap_or(n);
    Some(format!(
      "child {i}: eigen {} vs burn {} ({} vs {} children)",
      show(want.children.get(i)),
      show(got.children.get(i)),
      want.children.len(),
      got.children.len()
    ))
  };
  let mut d = Diff::default();
  let by_loc: HashMap<Loc, &ChildLine> = got.children.iter().map(|g| (g.loc, g)).collect();
  let mut max_visit_diff = 0i64;
  for w in &want.children {
    match by_loc.get(&w.loc) {
      Some(g) => {
        d.wl = d.wl.max((g.wl - w.wl).abs());
        d.util = d.util.max((g.util - w.util).abs());
        d.sm = d.sm.max((g.sm - w.sm).abs());
        d.lead = d.lead.max((g.lead - w.lead).abs());
        d.prior = d.prior.max((g.prior - w.prior).abs());
        max_visit_diff = max_visit_diff.max((g.edge - w.edge).abs());
      }
      None => max_visit_diff = max_visit_diff.max(w.edge),
    }
  }
  let want_locs: std::collections::HashSet<Loc> = want.children.iter().map(|w| w.loc).collect();
  for g in &got.children {
    if !want_locs.contains(&g.loc) {
      max_visit_diff = max_visit_diff.max(g.edge);
    }
  }
  // rv: win, loss, es, lead, wl, util
  d.wl = d.wl.max((got.rv[0] - want.rv[0]).abs()).max((got.rv[1] - want.rv[1]).abs()).max((got.rv[4] - want.rv[4]).abs());
  d.sm = d.sm.max((got.rv[2] - want.rv[2]).abs());
  d.lead = d.lead.max((got.rv[3] - want.rv[3]).abs());
  d.util = d.util.max((got.rv[5] - want.rv[5]).abs());
  let same_top = got.chosen == want.chosen;
  let top_note = if same_top { String::new() } else { format!("eigen {} vs burn {}", want.chosen_str, got.chosen_str) };
  let params = if c.params.is_empty() { "-".to_string() } else { c.params.iter().map(|(k, v)| format!("{k}={v}")).collect::<Vec<_>>().join(",") };
  Outcome {
    idx: c.idx,
    name: c.name.clone(),
    base: c.base,
    max_visits: c.max_visits,
    params,
    root_visits_equal,
    same_visits,
    same_top,
    diff: d,
    max_visit_diff,
    first_div,
    top_note,
    seconds,
  }
}

fn median_f(v: &mut [f64]) -> f64 {
  v.sort_by(|a, b| a.partial_cmp(b).unwrap());
  if v.is_empty() { 0.0 } else { v[v.len() / 2] }
}

fn summary(label: &str, outcomes: &[Outcome], total_seconds: f64) -> String {
  let mut s = String::new();
  let _ = writeln!(s, "search_tolerance: {label}, burn flex vs Eigen fp32, {} searches in {:.1} s", outcomes.len(), total_seconds);
  let _ = writeln!(s, "per search (diffs are max |burn - eigen| over the children both trees have + the root values; dvis = max child edge-visit difference):");
  let _ = writeln!(
    s,
    "  {:>3} {:<18} {:>4} {:>6} {:>4} {:>4} {:>5} {:>9} {:>9} {:>9} {:>9} {:>9} {:>6}  {}",
    "idx", "position", "base", "visits", "vec", "top", "dvis", "d_wl", "d_util", "d_sm", "d_lead", "d_prior", "secs", "params"
  );
  for o in outcomes {
    let _ = writeln!(
      s,
      "  {:>3} {:<18} {:>4} {:>6} {:>4} {:>4} {:>5} {:>9.2e} {:>9.2e} {:>9.2e} {:>9.2e} {:>9.2e} {:>6.1}  {}",
      o.idx,
      o.name,
      o.base,
      o.max_visits,
      if o.same_visits { "=" } else { "DIFF" },
      if o.same_top { "=" } else { "DIFF" },
      o.max_visit_diff,
      o.diff.wl,
      o.diff.util,
      o.diff.sm,
      o.diff.lead,
      o.diff.prior,
      o.seconds,
      o.params
    );
  }
  let mut by_visits: BTreeMap<i64, Vec<&Outcome>> = BTreeMap::new();
  for o in outcomes {
    by_visits.entry(o.max_visits).or_default().push(o);
  }
  let _ = writeln!(s, "{label} by maxVisits (value columns: max / median over the searches of the per-search max diff):");
  let _ = writeln!(
    s,
    "  {:>6} {:>3} {:>8} {:>8} {:>8} {:>19} {:>19} {:>19} {:>19} {:>19} {:>12}",
    "visits", "N", "same-vec", "same-top", "same-rv", "winLoss max/med", "utility max/med", "scoreMean max/med", "lead max/med", "prior max/med", "dvis(div) max/med"
  );
  let mut row = |name: String, group: &[&Outcome]| {
    let n = group.len();
    let sv = group.iter().filter(|o| o.same_visits).count();
    let st = group.iter().filter(|o| o.same_top).count();
    let sr = group.iter().filter(|o| o.root_visits_equal).count();
    let col = |f: &dyn Fn(&Outcome) -> f64| -> String {
      let mut v: Vec<f64> = group.iter().map(|o| f(o)).collect();
      let max = v.iter().cloned().fold(0.0, f64::max);
      format!("{:>9.2e}/{:>9.2e}", max, median_f(&mut v))
    };
    let mut dv: Vec<f64> = group.iter().filter(|o| !o.same_visits).map(|o| o.max_visit_diff as f64).collect();
    let dvmax = dv.iter().cloned().fold(0.0, f64::max);
    let dvs = if dv.is_empty() { "-".to_string() } else { format!("{:.0}/{:.0}", dvmax, median_f(&mut dv)) };
    let _ = writeln!(
      s,
      "  {:>6} {:>3} {:>8} {:>8} {:>8} {:>19} {:>19} {:>19} {:>19} {:>19} {:>12}",
      name,
      n,
      format!("{sv}/{n}"),
      format!("{st}/{n}"),
      format!("{sr}/{n}"),
      col(&|o| o.diff.wl),
      col(&|o| o.diff.util),
      col(&|o| o.diff.sm),
      col(&|o| o.diff.lead),
      col(&|o| o.diff.prior),
      dvs
    );
  };
  for (v, group) in &by_visits {
    row(v.to_string(), group);
  }
  let all: Vec<&Outcome> = outcomes.iter().collect();
  row("all".to_string(), &all);
  let n_div = outcomes.iter().filter(|o| !o.same_visits).count();
  let n_top = outcomes.iter().filter(|o| !o.same_top).count();
  let _ = writeln!(
    s,
    "{label}: N={} searches: identical root visit vectors {}/{} ({:.1}%), identical chosen moves {}/{} ({:.1}%), equal root visit counts {}/{}",
    outcomes.len(),
    outcomes.len() - n_div,
    outcomes.len(),
    100.0 * (outcomes.len() - n_div) as f64 / outcomes.len() as f64,
    outcomes.len() - n_top,
    outcomes.len(),
    100.0 * (outcomes.len() - n_top) as f64 / outcomes.len() as f64,
    outcomes.iter().filter(|o| o.root_visits_equal).count(),
    outcomes.len()
  );
  if n_div > 0 {
    let _ = writeln!(s, "{label}: searches whose root visit vector differs (first differing child, children order; first 12):");
    for o in outcomes.iter().filter(|o| !o.same_visits).take(12) {
      let _ = writeln!(s, "  case {} ({}, base {}, {} visits, {}): {}", o.idx, o.name, o.base, o.max_visits, o.params, o.first_div.as_deref().unwrap_or(""));
    }
  }
  if n_top > 0 {
    let _ = writeln!(s, "{label}: searches whose chosen move differs:");
    for o in outcomes.iter().filter(|o| !o.same_top) {
      let _ = writeln!(s, "  case {} ({}, base {}, {} visits, {}): {}", o.idx, o.name, o.base, o.max_visits, o.params, o.top_note);
    }
  }
  s
}

fn run<N: Network + 'static>(label: &str, fixture: &str, bpk: &str) -> Vec<Outcome> {
  ScoreValue::init_tables();
  Board::init_hash();
  set_clock(fake_clock);
  let cases = parse_cases(fixture);
  assert!(!cases.is_empty());
  eprintln!("search_tolerance {label}: {} cases", cases.len());

  let device: Device = burn::backend::FlexDevice::default().into();
  let mut pool = LocalPool::new();
  let spawner: Rc<dyn Spawner> = Rc::new(pool.spawner());
  let backend = Rc::new(BurnNeuralNet::<N>::new(device));
  // As the oracle: maxBatchSize 16, 19x19 net input, nnCacheSizePowerOfTwo 16 (mutex pool 12), defaultSymmetry 0, no
  // randomization. One evaluator (and cache) for all cases, like the oracle's.
  let nn_eval = Rc::new(
    NNEvaluator::new(
      "search_tolerance",
      N::NAME,
      "",
      16,
      19,
      19,
      false,
      false,
      16,
      12,
      false,
      "",
      "",
      false,
      enabled_t::False,
      enabled_t::Auto,
      1,
      &[-1],
      "searchEigenSeed",
      false,
      0,
      backend,
      spawner,
      std::fs::read(bpk).unwrap(),
    )
    .unwrap(),
  );
  run_watchdog(&mut pool, "spawnServerThreads", nn_eval.spawn_server_threads()).unwrap();

  let start = Instant::now();
  let mut outcomes = Vec::new();
  for c in &cases {
    let t0 = Instant::now();
    let got = run_search(&mut pool, &nn_eval, c);
    let o = compare(c, &got, t0.elapsed().as_secs_f64());
    eprintln!(
      "{label} case {:>3} {:<18} base {} {:>4} visits: vec {} top {} ({:.1} s)",
      c.idx,
      c.name,
      c.base,
      c.max_visits,
      if o.same_visits { "=" } else { "DIFF" },
      if o.same_top { "=" } else { "DIFF" },
      o.seconds
    );
    outcomes.push(o);
  }
  nn_eval.kill_server_threads();
  pool.run_until_stalled();
  eprintln!("{}", summary(label, &outcomes, start.elapsed().as_secs_f64()));
  outcomes
}

fn check(label: &str, outcomes: Vec<Outcome>) {
  let n = outcomes.len();
  for o in &outcomes {
    assert!(o.root_visits_equal, "{label} case {} ({}): root visit count differs", o.idx, o.name);
    assert!(o.diff.wl <= TOL_WINRATE_UTILITY, "{label} case {} ({}): winLossValue diff {:.3e} > {:.0e}", o.idx, o.name, o.diff.wl, TOL_WINRATE_UTILITY);
    assert!(o.diff.util <= TOL_WINRATE_UTILITY, "{label} case {} ({}): utility diff {:.3e} > {:.0e}", o.idx, o.name, o.diff.util, TOL_WINRATE_UTILITY);
    assert!(o.diff.sm <= TOL_SCORE, "{label} case {} ({}): scoreMean diff {:.3e} > {}", o.idx, o.name, o.diff.sm, TOL_SCORE);
    assert!(o.diff.lead <= TOL_SCORE, "{label} case {} ({}): lead diff {:.3e} > {}", o.idx, o.name, o.diff.lead, TOL_SCORE);
    assert!(o.diff.prior <= TOL_PRIOR, "{label} case {} ({}): prior diff {:.3e} > {:.0e}", o.idx, o.name, o.diff.prior, TOL_PRIOR);
  }
  let same_top = outcomes.iter().filter(|o| o.same_top).count();
  assert!(same_top as f64 >= MIN_SAME_TOP * n as f64, "{label}: chosen move equal in only {same_top}/{n} searches (need {:.0}%)", MIN_SAME_TOP * 100.0);
}

/// Big stack: Search (~40 KB), BoardHistory (~34 KB) and the recursive playout futures.
fn on_big_stack(label: &'static str, f: impl FnOnce() -> Vec<Outcome> + Send + 'static) -> Vec<Outcome> {
  std::thread::Builder::new().name(label.to_string()).stack_size(256 << 20).spawn(f).unwrap().join().unwrap_or_else(|_| panic!("{label} run panicked"))
}

#[cfg(feature = "net-b6c96")]
#[test]
#[cfg_attr(debug_assertions, ignore = "debug burn is very slow: run `cargo test --release -p katago-nn-burn --test search_tolerance -- --nocapture`")]
fn search_tolerance_b6c96() {
  let outcomes = on_big_stack("b6c96", || {
    run::<katago_nn_burn::b6c96::Model>("g170-b6c96", "search_tolerance_b6c96.txt", katago_nn_burn::b6c96::BPK_PATH)
  });
  check("g170-b6c96", outcomes);
}

#[cfg(feature = "net-b10c128")]
#[test]
#[cfg_attr(debug_assertions, ignore = "debug burn is very slow: run `cargo test --release -p katago-nn-burn --test search_tolerance -- --nocapture`")]
fn search_tolerance_b10c128() {
  let outcomes = on_big_stack("b10c128", || {
    run::<katago_nn_burn::b10c128::Model>("g170e-b10c128", "search_tolerance_b10c128.txt", katago_nn_burn::b10c128::BPK_PATH)
  });
  check("g170e-b10c128", outcomes);
}
