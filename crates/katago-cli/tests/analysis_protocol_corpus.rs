//! The analysis-protocol corpora (fixtures/upstream-v1.18.2/analysis-protocol, see its regen.py) replayed
//! through the `katago-rs` binary and compared with the reference KataGo's outputs: every corpus line set
//! runs as its own process with the same argv / override-config / stdin bytes as upstream did, and the
//! test compares
//! - the exit status (an uncaught exception is SIGABRT, -6, as upstream's std::terminate);
//! - stdout: JSON lines by the comparator of tests/common (structure exact, floats by key tolerance and
//!   format class); `": "` log lines (logToStdout) and stderr: the lines matching KEEP (warnings, errors,
//!   std::terminate's message, TCLAP's PARSE ERROR text), exactly;
//! - for multi-line cases, whose output interleaving races upstream: the request-loop lines (errors,
//!   warnings, action echoes) in order, and the analysis lines (results, noResults) in order; a case with
//!   several recorded upstream outcomes passes if it matches any of them.
//!
//! Normalized on both sides, as documented Rust choices (tests/analysis_protocol.rs checks the Rust values):
//! `git_hash` (the reference binary was built with NO_GIT_REVISION) and query_models' `maxBatchSize`
//! (upstream's Eigen backend forces 2; burn takes upstream's non-Eigen branch: nnMaxBatchSize, 64 here).
//! Partial responses of reportDuringSearchEvery / firstReportDuringSearchAfter cases are dropped (their
//! number is timing: see port-manifest notes); the final response is compared.
//! Not a port of upstream code.

mod common;

use std::collections::BTreeMap;
use std::io::Write as _;
use std::os::unix::process::ExitStatusExt;
use std::process::Command;
use std::process::Stdio;
use std::sync::Mutex;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use common::*;
use serde_json::Value;

const COMMON_OVERRIDES: &str = "logDir=, logTimeStamp=false, numAnalysisThreads=1, numSearchThreadsPerAnalysisThread=1, nnRandomize=false, nnRandSeed=analysisTest, forDeterministicTesting=true, maxVisits=8, rootSymmetryPruning=false";

/// The stderr / stdout log lines compared (regen.py KEEP_STDERR).
const KEEP: &[&str] = &[
  "terminate called",
  "  what()",
  ": ERROR",
  "ERROR",
  ": WARNING",
  "WARNING",
  ": Warning",
  "Warning",
  "PARSE ERROR",
  "             ",
  "Brief USAGE",
  "For complete USAGE",
  "   analysis",
  "KataGo v",
  "Started, ready",
  "Error",
];

/// Cases where katago-rs knowingly differs from the reference binary, with the reason. The test fails if
/// one of them starts matching (remove it then).
const KNOWN_DIVERGENCES: &[(&str, &str)] = &[
  (
    "cfg_18",
    "nnMaxBatchSize=0: upstream's Eigen backend ignores nnMaxBatchSize; burn follows the non-Eigen branch (range 1..65536)",
  ),
  // ovr_24 (investigated 2026-10, benign): on the empty 9x9 board the 8-symmetry average makes the four
  // center-adjacent points symmetric, so their priors tie up to float noise. Reference (Eigen): E4 = F5 =
  // 0.102564141 exactly, D5 = E6 = 0.102564074; selectBestChildToDescend's "best new child" loop takes the first
  // strictly greater prior in movePos order, so F5 (row 4) wins over E4 (row 5). katago-rs (burn flex): E4 =
  // 0.102564186, F5 = 0.102564178 (one f32 ulp, ~8e-9 apart: the ~1e-7 burn-vs-Eigen NN noise of the NN tier),
  // so E4 wins. The second root child (E5) and every root/raw value agree within tolerance; F5 and E4 are
  // different (single-symmetry) child evaluations, hence the different first moveInfo (lcb -0.807 vs -0.773).
  // The search's tie-break and averageMultipleSymmetries are checked bit-exactly by tests/search_oracle.rs and
  // nneval_skipnn.rs (katago-core); not a bug.
  (
    "ovr_24",
    "rootNumSymmetriesToSample=8, 3 visits, empty 9x9: E4/F5 priors tie in the reference (equal floats, first in \
     movePos order wins: F5) but differ by one f32 ulp under burn (E4 wins): NN float noise picks the symmetric twin",
  ),
];

fn fixture_dir() -> String {
  format!("{ROOT}/fixtures/upstream-v1.18.2/analysis-protocol")
}

fn decode_line(v: &Value) -> Vec<u8> {
  match v {
    Value::String(s) => s.as_bytes().to_vec(),
    Value::Object(o) => {
      let h = o["hex"].as_str().expect("hex line");
      (0..h.len()).step_by(2).map(|i| u8::from_str_radix(&h[i..i + 2], 16).expect("hex")).collect()
    }
    _ => panic!("bad corpus line {v}"),
  }
}

struct Outcome {
  stdout: Vec<String>,
  rc: i32,
  stderr: Vec<String>,
}

fn keep(line: &str) -> bool {
  KEEP.iter().any(|p| line.starts_with(p))
}

fn run_case(case: &Value) -> Outcome {
  let extra = case["extra"].as_str().unwrap_or("");
  let ov = if extra.is_empty() {
    COMMON_OVERRIDES.to_string()
  } else {
    format!("{COMMON_OVERRIDES}, {extra}")
  };
  let mut argv: Vec<String> = vec![
    "analysis".into(),
    "-config".into(),
    CFG.into(),
    "-model".into(),
    M6.into(),
    "-override-config".into(),
    ov,
  ];
  if let Some(a) = case.get("args").and_then(Value::as_array) {
    argv.extend(a.iter().map(|x| x.as_str().expect("arg").to_string()));
  }
  // {"sleep": s} entries pause the writer (regen.py does the same)
  let mut chunks: Vec<(Vec<u8>, f64)> = vec![(Vec::new(), 0.0)];
  for l in case["lines"].as_array().expect("lines") {
    if let Some(s) = l.get("sleep").and_then(Value::as_f64) {
      chunks.push((Vec::new(), s));
    } else {
      let cur = &mut chunks.last_mut().unwrap().0;
      cur.extend(decode_line(l));
      cur.push(b'\n');
    }
  }
  // KATAGO_RS_BIN: replay against another build (e.g. an older one, to see what a fix changed).
  let bin = std::env::var("KATAGO_RS_BIN").unwrap_or_else(|_| env!("CARGO_BIN_EXE_katago-rs").to_string());
  let mut child = Command::new(bin)
    .current_dir(cpp_dir())
    .args(&argv)
    .stdin(Stdio::piped())
    .stdout(Stdio::piped())
    .stderr(Stdio::piped())
    .spawn()
    .expect("spawn katago-rs");
  let mut sin = child.stdin.take().unwrap();
  let writer = std::thread::spawn(move || {
    for (bytes, sleep_before) in chunks {
      std::thread::sleep(std::time::Duration::from_secs_f64(sleep_before));
      let _ = sin.write_all(&bytes);
      let _ = sin.flush();
    }
  });
  let out = child.wait_with_output().unwrap();
  writer.join().unwrap();
  let rc = out.status.code().unwrap_or_else(|| -out.status.signal().unwrap_or(0));
  let stdout = String::from_utf8_lossy(&out.stdout).lines().map(str::to_string).collect();
  let stderr = String::from_utf8_lossy(&out.stderr).lines().filter(|l| keep(l)).map(str::to_string).collect();
  Outcome { stdout, rc, stderr }
}

/// git_hash / maxBatchSize (module docs).
fn normalize(line: &str) -> String {
  let mut s = line.to_string();
  for key in ["\"git_hash\":\"", "\"maxBatchSize\":"] {
    let mut from = 0;
    while let Some(p) = s[from..].find(key) {
      let start = from + p + key.len();
      let end = start + s[start..].find(['"', ',', '}']).unwrap_or(s.len() - start);
      s.replace_range(start..end, "<normalized>");
      from = start;
    }
  }
  s
}

fn is_analysis_line(l: &str) -> bool {
  l.contains("\"isDuringSearch\":")
}

fn compare_variant(case: &Value, var: &Value, got: &Outcome) -> Result<(), String> {
  let gold_rc = var["rc"].as_i64().unwrap() as i32;
  let strs = |v: &Value| -> Vec<String> { v.as_array().unwrap().iter().map(|x| x.as_str().unwrap().to_string()).collect() };
  let gold_out = strs(&var["stdout"]);
  let gold_err = strs(&var["stderr"]);
  let lines = case["lines"].as_array().unwrap();
  let rdse = lines.iter().any(|l| {
    l.as_str()
      .is_some_and(|s| s.contains("reportDuringSearchEvery") || s.contains("firstReportDuringSearchAfter"))
  });
  let split = |out: &[String]| -> (Vec<String>, Vec<String>) {
    let json: Vec<String> = out
      .iter()
      .filter(|l| l.starts_with('{'))
      .filter(|l| !(rdse && l.contains("\"isDuringSearch\":true")))
      .map(|l| normalize(l))
      .collect();
    let log: Vec<String> = out.iter().filter(|l| !l.starts_with('{') && keep(l)).cloned().collect();
    (json, log)
  };
  let (mut gj, gl) = split(&gold_out);
  let (mut oj, ol) = split(&got.stdout);
  if got.rc != gold_rc {
    return Err(format!("exit status {} vs {} expected", got.rc, gold_rc));
  }
  if gl != ol {
    return Err(format!("stdout log lines:\n  expected: {gl:?}\n  got:      {ol:?}"));
  }
  if gold_err != got.stderr {
    return Err(format!("stderr lines:\n  expected: {gold_err:?}\n  got:      {:?}", got.stderr));
  }
  let groups: Vec<(Vec<String>, Vec<String>)> = if case.get("sort").and_then(Value::as_bool) == Some(true) {
    gj.sort();
    oj.sort();
    vec![(gj, oj)]
  } else if lines.len() > 1 {
    let part = |v: &[String], a: bool| -> Vec<String> { v.iter().filter(|l| is_analysis_line(l) == a).cloned().collect() };
    vec![(part(&gj, false), part(&oj, false)), (part(&gj, true), part(&oj, true))]
  } else {
    vec![(gj, oj)]
  };
  for (g, o) in groups {
    let c = compare(&g, &o);
    if let Some(d) = c.divergence {
      return Err(format!("stdout: {d}"));
    }
  }
  Ok(())
}

static ONE_CORPUS_AT_A_TIME: Mutex<()> = Mutex::new(());

fn run_corpus(name: &str) {
  let _guard = ONE_CORPUS_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
  let path = format!("{}/expected/{name}.jsonl", fixture_dir());
  let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
  let cases: Vec<Value> = text.lines().map(|l| serde_json::from_str(l).expect("fixture JSON")).collect();
  assert!(!cases.is_empty(), "{name}: no cases");
  let jobs: usize = std::env::var("KATAGO_CORPUS_JOBS")
    .ok()
    .and_then(|s| s.parse().ok())
    .unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).clamp(1, 8));
  let next = AtomicUsize::new(0);
  let results: Mutex<BTreeMap<usize, Result<(), String>>> = Mutex::new(BTreeMap::new());
  let t0 = std::time::Instant::now();
  std::thread::scope(|s| {
    for _ in 0..jobs {
      s.spawn(|| {
        loop {
          let i = next.fetch_add(1, Ordering::SeqCst);
          if i >= cases.len() {
            break;
          }
          let case = &cases[i];
          let got = run_case(case);
          let variants = case["variants"].as_array().expect("variants");
          let mut r: Result<(), String> = Err("no variant".to_string());
          for v in variants {
            r = compare_variant(case, v, &got);
            if r.is_ok() {
              break;
            }
          }
          results.lock().unwrap().insert(i, r);
        }
      });
    }
  });
  let results = results.into_inner().unwrap();
  let mut failures: Vec<String> = Vec::new();
  let mut known = 0;
  for (i, r) in &results {
    let case_name = cases[*i]["name"].as_str().unwrap();
    let known_reason = KNOWN_DIVERGENCES.iter().find(|(n, _)| *n == case_name).map(|(_, r)| *r);
    match (r, known_reason) {
      (Ok(()), None) => {}
      (Err(_), Some(_)) => known += 1,
      (Ok(()), Some(reason)) => failures.push(format!("{case_name}: listed as a known divergence ({reason}) but now matches: remove it")),
      (Err(e), None) => failures.push(format!("{case_name}: {e}")),
    }
  }
  eprintln!(
    "analysis-protocol corpus {name}: {} cases, {} match upstream, {} known divergences, {} failures ({:.1} s)",
    cases.len(),
    cases.len() - known - failures.len(),
    known,
    failures.len(),
    t0.elapsed().as_secs_f64()
  );
  assert!(failures.is_empty(), "{name}: {} failures:\n{}", failures.len(), failures.join("\n"));
}

#[test]
fn corpus_fields() {
  run_corpus("fields");
}

#[test]
fn corpus_bytes_cfg_args() {
  run_corpus("bytes_cfg_args");
}

#[test]
fn corpus_avoid_allow() {
  run_corpus("avoid_allow");
}

#[test]
fn corpus_fuzz() {
  run_corpus("fuzz");
}

#[test]
fn corpus_multiline() {
  run_corpus("multiline");
}

#[test]
fn corpus_order() {
  run_corpus("order");
}
