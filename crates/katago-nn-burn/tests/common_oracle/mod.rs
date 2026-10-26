//! Shared by search_smoke.rs and search_tolerance.rs: the oracle fixture format (fixtures/upstream-v1.18.2/search/
//! oracle/search_*.txt: each CASE line holds everything needed to rebuild a search, the lines after it are the
//! upstream Eigen result), running a case through the port's Search over the burn `flex` NNEvaluator, and comparing
//! the two results. Net-agnostic; each test binary keeps its own tolerances, assertions and report. Not a port of
//! upstream code.

#![allow(dead_code)]

use std::collections::HashMap;
use std::rc::Rc;
use std::time::Duration;

use burn::prelude::*;
use futures::executor::LocalPool;

use katago_core::core::commontypes::enabled_t;
use katago_core::game::board::Board;
use katago_core::game::board::Loc;
use katago_core::game::board::Location;
use katago_core::game::boardhistory::BoardHistory;
use katago_core::game::rules::Rules;
use katago_core::neuralnet::nneval::NNEvaluator;
use katago_core::port::executor::Spawner;
use katago_core::search::analysisdata::AnalysisData;
use katago_core::search::reportedsearchvalues::ReportedSearchValues;
use katago_core::search::search::Search;
use katago_core::search::searchparams::SearchParams;
use katago_nn_burn::Network;
use katago_nn_burn::backend::BurnNeuralNet;

use crate::common_nn::run_watchdog;

//------------------------------------------------------------------------------------------------
// The fixture

#[derive(Clone, Debug)]
pub struct ChildLine {
  pub mv: String,
  pub loc: Loc,
  pub edge: i64,
  pub visits: i64,
  pub wl: f64,
  pub sm: f64,
  pub lead: f64,
  pub util: f64,
  pub prior: f64,
}

/// What one search reports (the oracle's output lines, or the same read off the port's Search).
#[derive(Clone, Debug)]
pub struct SearchResult {
  pub root_visits: i64,
  pub children: Vec<ChildLine>,
  /// win, loss, expectedScore, lead, winLossValue, utility
  pub rv: [f64; 6],
  pub chosen: Loc,
  pub chosen_str: String,
}

pub struct Case {
  pub idx: usize,
  pub name: String,
  pub xs: i32,
  pub ys: i32,
  pub rules: Rules,
  pub ipla: i8,
  pub pla: i8,
  pub moves: Vec<(Loc, i8)>,
  pub sseed: String,
  pub base: i32,
  pub max_visits: i64,
  pub params: Vec<(String, String)>,
  pub expected: SearchResult,
}

fn fixture_path(file: &str) -> String {
  format!("{}/../../fixtures/upstream-v1.18.2/search/oracle/{file}", env!("CARGO_MANIFEST_DIR"))
}

pub fn parse_cases(file: &str) -> Vec<Case> {
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
      kv["moves"]
        .split(',')
        .map(|m| m.split_once('/').map(|(l, p)| (l.parse().unwrap(), p.parse().unwrap())).unwrap())
        .collect()
    };
    let params: Vec<(String, String)> = if kv["params"] == "-" {
      Vec::new()
    } else {
      kv["params"]
        .split(',')
        .map(|p| p.split_once('=').map(|(k, v)| (k.to_string(), v.to_string())).unwrap())
        .collect()
    };

    // The result block.
    let mut res = SearchResult {
      root_visits: -1,
      children: Vec::new(),
      rv: [f64::NAN; 6],
      chosen: Board::NULL_LOC,
      chosen_str: String::new(),
    };
    loop {
      let l = lines.next().expect("END line");
      if let Some(n) = l.strip_prefix("END ") {
        assert_eq!(n.parse::<usize>().unwrap(), idx);
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
        "RV" => {
          res.rv = [
            num(&t, "win="),
            num(&t, "loss="),
            num(&t, "es="),
            num(&t, "lead="),
            num(&t, "wl="),
            num(&t, "util="),
          ]
        }
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

/// The SearchParams keys every oracle fixture uses; panics on any other key (a binary with more keys wraps this).
pub fn set_param_base(p: &mut SearchParams, k: &str, v: &str) {
  match k {
    "useGraphSearch" => p.use_graph_search = v == "1",
    "rootSymmetryPruning" => p.root_symmetry_pruning = v == "1",
    "chosenMoveTemperature" => p.chosen_move_temperature = v.parse().unwrap(),
    "chosenMoveTemperatureEarly" => p.chosen_move_temperature_early = v.parse().unwrap(),
    other => panic!("unknown param {other}"),
  }
}

/// `set_param` applies one fixture `key=value` to the params (`set_param_base`, or a wrapper with more keys).
pub fn run_search(pool: &mut LocalPool, nn_eval: &Rc<NNEvaluator>, c: &Case, set_param: fn(&mut SearchParams, &str, &str)) -> SearchResult {
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
  run_watchdog(pool, "runWholeSearch", Duration::ZERO, search.run_whole_search(c.pla)).unwrap();

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

  SearchResult {
    root_visits: search.get_root_visits(),
    children: out,
    rv: [v.win_value, v.loss_value, v.expected_score, v.lead, v.win_loss_value, v.utility],
    chosen,
    chosen_str: Location::to_string(chosen, search.get_root_board()),
  }
}

//------------------------------------------------------------------------------------------------
// Comparison

#[derive(Default, Clone, Debug)]
pub struct Diff {
  /// winLossValue (children and root values: win, loss, winLossValue)
  pub wl: f64,
  pub util: f64,
  /// scoreMean (children, root expectedScore)
  pub sm: f64,
  /// lead (children and root lead)
  pub lead: f64,
  pub prior: f64,
}

pub struct Outcome {
  pub idx: usize,
  pub name: String,
  pub base: i32,
  pub max_visits: i64,
  pub params: String,
  pub root_visits_equal: bool,
  pub same_visits: bool,
  pub same_top: bool,
  pub diff: Diff,
  /// Largest |edge visits burn - eigen| of any child (a child missing on one side counts as 0).
  pub max_visit_diff: i64,
  /// First differing child of the visit vector, if it differs.
  pub first_div: Option<String>,
  pub top_note: String,
  pub seconds: f64,
}

/// Fills every `Diff` / `Outcome` field (a binary prints the ones it reports).
pub fn compare(c: &Case, got: &SearchResult, seconds: f64) -> Outcome {
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
  d.wl = d
    .wl
    .max((got.rv[0] - want.rv[0]).abs())
    .max((got.rv[1] - want.rv[1]).abs())
    .max((got.rv[4] - want.rv[4]).abs());
  d.sm = d.sm.max((got.rv[2] - want.rv[2]).abs());
  d.lead = d.lead.max((got.rv[3] - want.rv[3]).abs());
  d.util = d.util.max((got.rv[5] - want.rv[5]).abs());
  let same_top = got.chosen == want.chosen;
  let top_note = if same_top {
    String::new()
  } else {
    format!("eigen {} vs burn {}", want.chosen_str, got.chosen_str)
  };
  let params = if c.params.is_empty() {
    "-".to_string()
  } else {
    c.params.iter().map(|(k, v)| format!("{k}={v}")).collect::<Vec<_>>().join(",")
  };
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

pub fn median(v: &mut [f64]) -> f64 {
  v.sort_by(|a, b| a.partial_cmp(b).unwrap());
  if v.is_empty() { 0.0 } else { v[v.len() / 2] }
}

/// The evaluator of the oracle: maxBatchSize 16, 19x19 net input, nnCacheSizePowerOfTwo 16 (mutex pool 12),
/// defaultSymmetry 0, no randomization. One evaluator (and cache) for all cases, like the oracle's. Returns the pool it
/// runs on, with the server threads spawned.
pub fn new_nn_eval<N: Network + 'static>(name: &str, bpk: &str) -> (LocalPool, Rc<NNEvaluator>) {
  let device: Device = burn::backend::FlexDevice.into();
  let mut pool = LocalPool::new();
  let spawner: Rc<dyn Spawner> = Rc::new(pool.spawner());
  let backend = Rc::new(BurnNeuralNet::<N>::new(device));
  let nn_eval = Rc::new(
    NNEvaluator::new(
      name,
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
  run_watchdog(&mut pool, "spawnServerThreads", Duration::ZERO, nn_eval.spawn_server_threads()).unwrap();
  (pool, nn_eval)
}
