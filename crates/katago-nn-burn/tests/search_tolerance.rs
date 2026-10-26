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
//!
//! The test FAILS only on gross problems: a root visit count differs, the chosen move differs in more than 10% of the
//! searches, or a value difference exceeds the loose bounds below (those of search_smoke.rs). Run in release (debug
//! burn is very slow; the tests are ignored in debug):
//!   `cargo test --release -p katago-nn-burn --test search_tolerance -- --nocapture`
#![cfg(all(feature = "flex", not(target_arch = "wasm32")))]

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::time::Instant;

use katago_core::core::timer::set_clock;
use katago_core::game::board::Board;
use katago_core::neuralnet::nninputs::ScoreValue;
use katago_core::search::searchparams::SearchParams;
use katago_nn_burn::Network;

mod common_nn;
mod common_oracle;

use common_nn::fake_clock;
use common_oracle::*;

/// Loose bounds on the max |burn - eigen| of a search (after the 1e-5-level NN differences went through the search;
/// children whose visit counts differ between the two trees are compared too).
const TOL_WINRATE_UTILITY: f64 = 1e-2;
const TOL_SCORE: f64 = 0.5;
const TOL_PRIOR: f64 = 1e-2;
/// Minimum fraction of searches with the same chosen move.
const MIN_SAME_TOP: f64 = 0.9;

/// The SearchParams keys of the tolerance fixtures: those of `set_param_base` plus four more.
fn set_param(p: &mut SearchParams, k: &str, v: &str) {
  match k {
    "wideRootNoise" => p.wide_root_noise = v.parse().unwrap(),
    "rootPolicyTemperature" => p.root_policy_temperature = v.parse().unwrap(),
    "rootPolicyTemperatureEarly" => p.root_policy_temperature_early = v.parse().unwrap(),
    "conservativePass" => p.conservative_pass = v == "1",
    other => set_param_base(p, other, v),
  }
}

fn summary(label: &str, outcomes: &[Outcome], total_seconds: f64) -> String {
  let mut s = String::new();
  let _ = writeln!(
    s,
    "search_tolerance: {label}, burn flex vs Eigen fp32, {} searches in {:.1} s",
    outcomes.len(),
    total_seconds
  );
  let _ = writeln!(
    s,
    "per search (diffs are max |burn - eigen| over the children both trees have + the root values; dvis = max child edge-visit difference):"
  );
  let _ = writeln!(
    s,
    "  {:>3} {:<18} {:>4} {:>6} {:>4} {:>4} {:>5} {:>9} {:>9} {:>9} {:>9} {:>9} {:>6}  params",
    "idx", "position", "base", "visits", "vec", "top", "dvis", "d_wl", "d_util", "d_sm", "d_lead", "d_prior", "secs"
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
  let _ = writeln!(
    s,
    "{label} by maxVisits (value columns: max / median over the searches of the per-search max diff):"
  );
  let _ = writeln!(
    s,
    "  {:>6} {:>3} {:>8} {:>8} {:>8} {:>19} {:>19} {:>19} {:>19} {:>19} {:>12}",
    "visits",
    "N",
    "same-vec",
    "same-top",
    "same-rv",
    "winLoss max/med",
    "utility max/med",
    "scoreMean max/med",
    "lead max/med",
    "prior max/med",
    "dvis(div) max/med"
  );
  let mut row = |name: String, group: &[&Outcome]| {
    let n = group.len();
    let sv = group.iter().filter(|o| o.same_visits).count();
    let st = group.iter().filter(|o| o.same_top).count();
    let sr = group.iter().filter(|o| o.root_visits_equal).count();
    let col = |f: &dyn Fn(&Outcome) -> f64| -> String {
      let mut v: Vec<f64> = group.iter().map(|o| f(o)).collect();
      let max = v.iter().cloned().fold(0.0, f64::max);
      format!("{:>9.2e}/{:>9.2e}", max, median(&mut v))
    };
    let mut dv: Vec<f64> = group.iter().filter(|o| !o.same_visits).map(|o| o.max_visit_diff as f64).collect();
    let dvmax = dv.iter().cloned().fold(0.0, f64::max);
    let dvs = if dv.is_empty() {
      "-".to_string()
    } else {
      format!("{:.0}/{:.0}", dvmax, median(&mut dv))
    };
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
    let _ = writeln!(
      s,
      "{label}: searches whose root visit vector differs (first differing child, children order; first 12):"
    );
    for o in outcomes.iter().filter(|o| !o.same_visits).take(12) {
      let _ = writeln!(
        s,
        "  case {} ({}, base {}, {} visits, {}): {}",
        o.idx,
        o.name,
        o.base,
        o.max_visits,
        o.params,
        o.first_div.as_deref().unwrap_or("")
      );
    }
  }
  if n_top > 0 {
    let _ = writeln!(s, "{label}: searches whose chosen move differs:");
    for o in outcomes.iter().filter(|o| !o.same_top) {
      let _ = writeln!(
        s,
        "  case {} ({}, base {}, {} visits, {}): {}",
        o.idx, o.name, o.base, o.max_visits, o.params, o.top_note
      );
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

  let (mut pool, nn_eval) = new_nn_eval::<N>("search_tolerance", bpk);

  let start = Instant::now();
  let mut outcomes = Vec::new();
  for c in &cases {
    let t0 = Instant::now();
    let got = run_search(&mut pool, &nn_eval, c, set_param);
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
    assert!(
      o.diff.wl <= TOL_WINRATE_UTILITY,
      "{label} case {} ({}): winLossValue diff {:.3e} > {:.0e}",
      o.idx,
      o.name,
      o.diff.wl,
      TOL_WINRATE_UTILITY
    );
    assert!(
      o.diff.util <= TOL_WINRATE_UTILITY,
      "{label} case {} ({}): utility diff {:.3e} > {:.0e}",
      o.idx,
      o.name,
      o.diff.util,
      TOL_WINRATE_UTILITY
    );
    assert!(
      o.diff.sm <= TOL_SCORE,
      "{label} case {} ({}): scoreMean diff {:.3e} > {}",
      o.idx,
      o.name,
      o.diff.sm,
      TOL_SCORE
    );
    assert!(
      o.diff.lead <= TOL_SCORE,
      "{label} case {} ({}): lead diff {:.3e} > {}",
      o.idx,
      o.name,
      o.diff.lead,
      TOL_SCORE
    );
    assert!(
      o.diff.prior <= TOL_PRIOR,
      "{label} case {} ({}): prior diff {:.3e} > {:.0e}",
      o.idx,
      o.name,
      o.diff.prior,
      TOL_PRIOR
    );
  }
  let same_top = outcomes.iter().filter(|o| o.same_top).count();
  assert!(
    same_top as f64 >= MIN_SAME_TOP * n as f64,
    "{label}: chosen move equal in only {same_top}/{n} searches (need {:.0}%)",
    MIN_SAME_TOP * 100.0
  );
}

/// Big stack: Search (~40 KB), BoardHistory (~34 KB) and the recursive playout futures.
fn on_big_stack(label: &'static str, f: impl FnOnce() -> Vec<Outcome> + Send + 'static) -> Vec<Outcome> {
  std::thread::Builder::new()
    .name(label.to_string())
    .stack_size(256 << 20)
    .spawn(f)
    .unwrap()
    .join()
    .unwrap_or_else(|_| panic!("{label} run panicked"))
}

#[cfg(feature = "net-b6c96")]
#[test]
#[cfg_attr(
  debug_assertions,
  ignore = "debug burn is very slow: run `cargo test --release -p katago-nn-burn --test search_tolerance -- --nocapture`"
)]
fn search_tolerance_b6c96() {
  let outcomes = on_big_stack("b6c96", || {
    run::<katago_nn_burn::b6c96::Model>("g170-b6c96", "search_tolerance_b6c96.txt", katago_nn_burn::b6c96::BPK_PATH)
  });
  check("g170-b6c96", outcomes);
}

#[cfg(feature = "net-b10c128")]
#[test]
#[cfg_attr(
  debug_assertions,
  ignore = "debug burn is very slow: run `cargo test --release -p katago-nn-burn --test search_tolerance -- --nocapture`"
)]
fn search_tolerance_b10c128() {
  let outcomes = on_big_stack("b10c128", || {
    run::<katago_nn_burn::b10c128::Model>("g170e-b10c128", "search_tolerance_b10c128.txt", katago_nn_burn::b10c128::BPK_PATH)
  });
  check("g170e-b10c128", outcomes);
}
