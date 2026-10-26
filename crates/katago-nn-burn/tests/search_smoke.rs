//! Real-net search smoke test: the ported `Search` (katago_core) over the ported `NNEvaluator` and the burn
//! `flex` backend (net g170-b6c96) vs the unmodified upstream Search over the Eigen fp32 NNEvaluator on the same
//! net. Oracle: fixtures/upstream-v1.18.2/search/oracle/search_eigen.cpp -> search_eigen_b6c96.txt (each CASE line
//! holds everything needed to rebuild the search; the lines after it are the upstream result). Not a port of
//! upstream code.
//!
//! burn's conv accumulation order differs from Eigen's, so the NN outputs differ by ~1e-5 (nn_parity.rs) and those
//! differences can change which child the search picks, hence this is a tolerance comparison, not byte-exact like
//! the NN-less search_oracle. Per search it checks / reports:
//!   - the root visit count (must be equal),
//!   - whether the root visit vector (move + edge visits of every root child, children order) is identical,
//!   - whether the chosen move (getChosenMoveLoc, chosenMoveTemperature 0) is identical,
//!   - the max |burn - eigen| of winLossValue / utility / scoreMean / lead / policy prior over the children both
//!     sides have, plus the root values (getRootValues).
//!
//! The test FAILS only if a root visit count differs, the chosen move differs in more than 10% of the searches, or
//! a value difference exceeds the loose bounds below. It always prints the summary table: these numbers inform the
//! tolerance policy (docs/PORTING.md section 13). Run in release (debug burn is very slow):
//!   `just search-smoke` = `cargo test --release -p katago-nn-burn --test search_smoke -- --nocapture`.
#![cfg(all(feature = "flex", feature = "net-b6c96"))]

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::time::Instant;

use katago_core::core::timer::set_clock;
use katago_core::game::board::Board;
use katago_core::neuralnet::nninputs::ScoreValue;

mod common_nn;
mod common_oracle;

use common_nn::fake_clock;
use common_oracle::*;

type N = katago_nn_burn::b6c96::Model;

/// Loose bounds on the max |burn - eigen| of a search (after the 1e-5-level NN differences went through the search;
/// children whose visit counts differ between the two trees are compared too).
const TOL_WINRATE_UTILITY: f64 = 1e-2;
const TOL_SCORE: f64 = 0.5;
const TOL_PRIOR: f64 = 1e-2;
/// Minimum fraction of searches with the same chosen move.
const MIN_SAME_TOP: f64 = 0.9;

fn summary(outcomes: &[Outcome], total_seconds: f64) -> String {
  let mut s = String::new();
  let _ = writeln!(
    s,
    "search_smoke: g170-b6c96, burn flex vs Eigen fp32, {} searches in {:.1} s",
    outcomes.len(),
    total_seconds
  );
  let _ = writeln!(
    s,
    "per search (diffs are max |burn - eigen| over the children both trees have + the root values):"
  );
  let _ = writeln!(
    s,
    "  {:>3} {:<12} {:>4} {:>6} {:>4} {:>4} {:>9} {:>9} {:>9} {:>9} {:>6}",
    "idx", "position", "base", "visits", "vec", "top", "d_wl", "d_util", "d_score", "d_prior", "secs"
  );
  for o in outcomes {
    let _ = writeln!(
      s,
      "  {:>3} {:<12} {:>4} {:>6} {:>4} {:>4} {:>9.2e} {:>9.2e} {:>9.2e} {:>9.2e} {:>6.1}",
      o.idx,
      o.name,
      o.base,
      o.max_visits,
      if o.same_visits { "=" } else { "DIFF" },
      if o.same_top { "=" } else { "DIFF" },
      o.diff.wl,
      o.diff.util,
      o.diff.sm.max(o.diff.lead),
      o.diff.prior,
      o.seconds
    );
  }
  let mut by_visits: BTreeMap<i64, Vec<&Outcome>> = BTreeMap::new();
  for o in outcomes {
    by_visits.entry(o.max_visits).or_default().push(o);
  }
  let _ = writeln!(s, "by maxVisits (value columns: max / median over the searches of the per-search max diff):");
  let _ = writeln!(
    s,
    "  {:>6} {:>3} {:>7} {:>7} {:>19} {:>19} {:>19} {:>19}",
    "visits", "N", "same-vec", "same-top", "winLoss max/med", "utility max/med", "score max/med", "prior max/med"
  );
  let mut row = |label: String, group: &[&Outcome]| {
    let n = group.len();
    let sv = group.iter().filter(|o| o.same_visits).count();
    let st = group.iter().filter(|o| o.same_top).count();
    let col = |f: &dyn Fn(&Outcome) -> f64| -> String {
      let mut v: Vec<f64> = group.iter().map(|o| f(o)).collect();
      let max = v.iter().cloned().fold(0.0, f64::max);
      format!("{:>9.2e}/{:>9.2e}", max, median(&mut v))
    };
    let _ = writeln!(
      s,
      "  {:>6} {:>3} {:>7} {:>7} {:>19} {:>19} {:>19} {:>19}",
      label,
      n,
      format!("{sv}/{n}"),
      format!("{st}/{n}"),
      col(&|o| o.diff.wl),
      col(&|o| o.diff.util),
      col(&|o| o.diff.sm.max(o.diff.lead)),
      col(&|o| o.diff.prior)
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
    "N={} searches: identical root visit vectors {}/{}, identical chosen moves {}/{}, equal root visit counts {}/{}",
    outcomes.len(),
    outcomes.len() - n_div,
    outcomes.len(),
    outcomes.len() - n_top,
    outcomes.len(),
    outcomes.iter().filter(|o| o.root_visits_equal).count(),
    outcomes.len()
  );
  if n_div > 0 {
    let _ = writeln!(s, "searches whose root visit vector differs (first differing child, children order):");
    for o in outcomes.iter().filter(|o| !o.same_visits).take(8) {
      let _ = writeln!(
        s,
        "  case {} ({}, base {}, {} visits): {}",
        o.idx,
        o.name,
        o.base,
        o.max_visits,
        o.first_div.as_deref().unwrap_or("")
      );
    }
  }
  if n_top > 0 {
    let _ = writeln!(s, "searches whose chosen move differs:");
    for o in outcomes.iter().filter(|o| !o.same_top) {
      let _ = writeln!(s, "  case {} ({}, base {}, {} visits): {}", o.idx, o.name, o.base, o.max_visits, o.top_note);
    }
  }
  s
}

fn run() -> Vec<Outcome> {
  ScoreValue::init_tables();
  Board::init_hash();
  set_clock(fake_clock);
  let cases = parse_cases("search_eigen_b6c96.txt");
  assert!(!cases.is_empty());

  let (mut pool, nn_eval) = new_nn_eval::<N>("search_smoke", katago_nn_burn::b6c96::BPK_PATH);

  let start = Instant::now();
  let mut outcomes = Vec::new();
  for c in &cases {
    let t0 = Instant::now();
    let got = run_search(&mut pool, &nn_eval, c, set_param_base);
    let o = compare(c, &got, t0.elapsed().as_secs_f64());
    eprintln!(
      "case {:>2} {:<12} base {} {:>3} visits: vec {} top {} ({:.1} s)",
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
  eprintln!("{}", summary(&outcomes, start.elapsed().as_secs_f64()));
  outcomes
}

#[test]
fn search_smoke_b6c96() {
  //Big stack: Search (~40 KB), BoardHistory (~34 KB) and the recursive playout futures.
  let outcomes = std::thread::Builder::new()
    .name("search_smoke".to_string())
    .stack_size(256 << 20)
    .spawn(run)
    .unwrap()
    .join()
    .expect("search_smoke run panicked");
  let n = outcomes.len();
  for o in &outcomes {
    assert!(o.root_visits_equal, "case {} ({}): root visit count differs", o.idx, o.name);
    assert!(
      o.diff.wl <= TOL_WINRATE_UTILITY,
      "case {} ({}): winLossValue diff {:.3e} > {:.0e}",
      o.idx,
      o.name,
      o.diff.wl,
      TOL_WINRATE_UTILITY
    );
    assert!(
      o.diff.util <= TOL_WINRATE_UTILITY,
      "case {} ({}): utility diff {:.3e} > {:.0e}",
      o.idx,
      o.name,
      o.diff.util,
      TOL_WINRATE_UTILITY
    );
    assert!(
      o.diff.sm.max(o.diff.lead) <= TOL_SCORE,
      "case {} ({}): score diff {:.3e} > {}",
      o.idx,
      o.name,
      o.diff.sm.max(o.diff.lead),
      TOL_SCORE
    );
    assert!(
      o.diff.prior <= TOL_PRIOR,
      "case {} ({}): prior diff {:.3e} > {:.0e}",
      o.idx,
      o.name,
      o.diff.prior,
      TOL_PRIOR
    );
  }
  let same_top = outcomes.iter().filter(|o| o.same_top).count();
  assert!(
    same_top as f64 >= MIN_SAME_TOP * n as f64,
    "chosen move equal in only {same_top}/{n} searches (need {:.0}%)",
    MIN_SAME_TOP * 100.0
  );
}
