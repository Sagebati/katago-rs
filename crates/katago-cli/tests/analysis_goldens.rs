//! Analysis goldens (M3 search tier, DECISIONS.md D9): the query files of upstream cpp/tests/analysis/ run
//! through the ported `MainCmds::analysis` (katago_core::command::analysis) with the burn flex backend and
//! the same config and overrides as fixtures/upstream-v1.16.5/analysis/regen.sh (= runcmdtests.sh), compared
//! with our Eigen reruns fixtures/upstream-v1.16.5/analysis/*.stdout by the explicit comparator in
//! tests/common (structure byte-exact, floats within VALUE_TOL / SCORE_TOL / WEIGHT_REL_TOL).
//! Each test prints "exact" / "within tolerance (max deviations)" / "DIVERGED (first difference)".
//!
//! Pending (not run here):
//! - humansl_sidetomove.stdout: needs a human-SL net (M8; tools/bin2onnx rejects them).
//!
//! Debug runs take a few minutes (the two 1000-visit pvvisits goldens dominate); `just analysis-goldens`
//! runs them in release. Not a port of upstream code.

mod common;

use common::*;

const LOOP_OVERRIDES: &str = "maxVisits=100, maxPlayouts=10000, rootSymmetryPruning=false";

fn assert_ok(name: &str, c: &Comparison) {
  assert!(c.num_lines >= 1, "{name}: no lines compared");
  assert!(c.divergence.is_none(), "{name}: {}", c.status());
}

#[test]
fn comparator_catches_differences() {
  let gold = read_golden("basic.txt.stdout");
  assert_eq!(compare(&gold, &gold).status(), "exact (byte-identical)");
  // An integer (visits) changed by one: structural.
  let mut g2 = gold.clone();
  g2[0] = g2[0].replacen("\"visits\":9", "\"visits\":8", 1);
  assert_ne!(g2[0], gold[0]);
  assert!(compare(&gold, &g2).divergence.is_some());
  // A winrate changed by 2e-5 > VALUE_TOL.
  let mut g3 = gold.clone();
  g3[0] = g3[0].replacen("\"winrate\":0.61100084", "\"winrate\":0.61102084", 1);
  assert_ne!(g3[0], gold[0]);
  assert!(compare(&gold, &g3).divergence.is_some());
  // The same winrate changed by 2e-6: within tolerance.
  let mut g4 = gold.clone();
  g4[0] = g4[0].replacen("\"winrate\":0.61100084", "\"winrate\":0.61100284", 1);
  let c4 = compare(&gold, &g4);
  assert!(c4.divergence.is_none() && !c4.byte_identical, "{}", c4.status());
  // A float written as an integer: structural.
  let mut g5 = gold.clone();
  g5[0] = g5[0].replacen("\"edgeWeight\":9.0", "\"edgeWeight\":9", 1);
  assert!(compare(&gold, &g5).divergence.is_some());
  // A key order change: structural.
  let mut g6 = gold.clone();
  g6[0] = g6[0].replacen("\"id\":\"foo\",\"isDuringSearch\":false", "\"isDuringSearch\":false,\"id\":\"foo\"", 1);
  assert!(compare(&gold, &g6).divergence.is_some());
  // A missing line.
  assert!(compare(&gold, &gold[1..]).divergence.is_some());

  let line = |l: &str| vec![l.to_string()];
  let differs = |g: &str, o: &str| compare(&line(g), &line(o)).divergence.is_some();
  // A tiny policy value off by 5e-6 (twice itself): relative tolerance below 1e-2.
  assert!(differs(r#"{"policy":[2.5e-06]}"#, r#"{"policy":[7.5e-06]}"#));
  assert!(!differs(r#"{"prior":0.25}"#, r#"{"prior":0.250004}"#));
  assert!(!differs(r#"{"prior":2.5e-06}"#, r#"{"prior":2.5001e-06}"#));
  // The same value written in exponent form where nlohmann would not: format class.
  assert!(differs(r#"{"winrate":0.000123}"#, r#"{"winrate":1.23e-04}"#));
  // Straddling 1e-4 within tolerance is fine.
  assert!(!differs(r#"{"winrate":0.0001}"#, r#"{"winrate":9.9999e-05}"#));
  // An unrounded f32 (17 digits, exactly a float) where upstream wrote a rounded 9-digit double.
  #[allow(clippy::excessive_precision)]
  let raw_f32 = format!("{:?}", 0.455756152f32 as f64);
  assert!(differs(r#"{"winrate":0.455756152}"#, &format!(r#"{{"winrate":{raw_f32}}}"#)));
  // Score keys: PORTING.md §13 NN tier (5e-4).
  assert!(!differs(r#"{"scoreStdev":4.59838054}"#, r#"{"scoreStdev":4.59825107}"#));
  assert!(differs(r#"{"scoreLead":1.5}"#, r#"{"scoreLead":1.5006}"#));
}

#[test]
fn golden_clear_cache() {
  let c = check_golden("clear_cache.txt.stdout", "tests/analysis/clear_cache.txt", M6, LOOP_OVERRIDES);
  assert_ok("clear_cache", &c);
  assert!(c.byte_identical, "clear_cache must be byte-identical");
}

#[test]
fn golden_badoverride() {
  let c = check_golden("badoverride.txt.stdout", "tests/analysis/badoverride.txt", M6, LOOP_OVERRIDES);
  assert_ok("badoverride", &c);
  // The warning lines carry no floats: byte-exact.
  let gold = read_golden("badoverride.txt.stdout");
  let warnings: Vec<&String> = gold.iter().filter(|l| l.contains("\"warning\"")).collect();
  assert_eq!(warnings.len(), 3);
}

#[test]
fn golden_basic() {
  let c = check_golden("basic.txt.stdout", "tests/analysis/basic.txt", M6, LOOP_OVERRIDES);
  assert_ok("basic", &c);
}

#[test]
fn golden_basic_sidetomove() {
  let c = check_golden("basic_sidetomove.stdout", "tests/analysis/basic.txt", M6, &format!("{LOOP_OVERRIDES}, reportAnalysisWinratesAs=SIDETOMOVE"));
  assert_ok("basic_sidetomove", &c);
}

#[test]
fn golden_symmetry() {
  let c = check_golden("symmetry.txt.stdout", "tests/analysis/symmetry.txt", M6, LOOP_OVERRIDES);
  assert_ok("symmetry", &c);
}

#[test]
fn golden_symmetry_with_pruning() {
  let c = check_golden("symmetry_with_pruning.stdout", "tests/analysis/symmetry.txt", M6, "maxVisits=100, maxPlayouts=10000, rootSymmetryPruning=true");
  assert_ok("symmetry_with_pruning", &c);
}

/// The loop form (maxVisits=100, no eval cache): repeatability, C equals A.
#[test]
fn golden_evalcache_loop() {
  let c = check_golden("evalcache.txt.stdout", "tests/analysis/evalcache.txt", M6, LOOP_OVERRIDES);
  assert_ok("evalcache", &c);
}

/// The eval-cache form (regen.sh `eval_cache`, runcmdtests.sh's fifo test): b10c128, rootSymmetryPruning=false,
/// useGraphSearch=true, useEvalCache=true, evalCacheMinVisits=10, the queries' own maxVisits (50/100/50), and line N+1 is fed
/// only after response N. The cache changes the result: C differs from A, B reuses A's entries.
#[test]
fn golden_eval_cache() {
  let name = "eval_cache.stdout";
  let gold = read_golden(name);
  assert!(!gold.is_empty(), "{name}: empty golden");
  let t0 = std::time::Instant::now();
  let got = run_analysis_handshake("tests/analysis/evalcache.txt", M10, "rootSymmetryPruning=false, useGraphSearch=true, useEvalCache=true, evalCacheMinVisits=10");
  let c = compare(&gold, &got);
  eprintln!("analysis golden {name}: {} lines, {} ({:.1} s)", c.num_lines, c.status(), t0.elapsed().as_secs_f64());
  assert_ok("eval_cache", &c);
  // The cache must matter: without it, C equals A (golden_evalcache_loop).
  assert_eq!(c.num_lines, 3);
  let strip = |l: &str| l.replacen("\"id\":\"C\"", "\"id\":\"A\"", 1);
  assert_ne!(strip(&got[2]), got[0], "eval_cache: C equals A, the cache had no effect");
}

/// 1000 visits (the queries' own maxVisits), graph search on.
#[test]
fn golden_pvvisits() {
  let c = check_golden("pvvisits.txt.stdout", "tests/analysis/pvvisits.txt", M6, LOOP_OVERRIDES);
  assert_ok("pvvisits", &c);
}

/// 1000 visits, no maxVisits override, useGraphSearch=false.
#[test]
fn golden_pvvisits_nograph() {
  let c = check_golden("pvvisits_nograph.stdout", "tests/analysis/pvvisits.txt", M6, "rootSymmetryPruning=false, useGraphSearch=false");
  assert_ok("pvvisits_nograph", &c);
}
