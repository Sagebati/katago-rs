//! Shared by the analysis golden tests: running a query file through the ported analysis engine
//! (library path, the same host pieces as the binary) and the explicit golden comparator.
//! Not a port of upstream code.
#![allow(dead_code)]

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

use futures::executor::LocalPool;
use katago_cli::BurnHost;
use katago_core::command::analysis::AnalysisArgs;
use katago_core::command::analysis::AnalysisHost;
use katago_core::port::executor::Spawner;

pub const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");

/// upstream/KataGo/cpp: the cwd of runcmdtests.sh / regen.sh (relative paths below are relative to it).
pub fn cpp_dir() -> String {
  format!("{ROOT}/upstream/KataGo/cpp")
}

pub fn golden_path(name: &str) -> String {
  format!("{ROOT}/fixtures/upstream-v1.16.5/analysis/{name}")
}

/// `COMMON` of fixtures/upstream-v1.16.5/analysis/regen.sh (= runcmdtests.sh), minus `logFile=...` (no log
/// files in tests).
pub const COMMON: &str = "logDir=, logTimeStamp=false, logAllRequests=true, logAllResponses=true, logSearchInfo=true, numAnalysisThreads=1, numSearchThreadsPerAnalysisThread=1, nnRandomize=false, nnRandSeed=analysisTest, forDeterministicTesting=true, cudaUseFP16 = false, trtUseFP16 = false, openclUseFP16 = false, cudaUseNHWC = false";
pub const CFG: &str = "configs/analysis_example.cfg";
pub const M6: &str = "tests/models/g170-b6c96-s175395328-d26788732.bin.gz";
pub const M10: &str = "tests/models/g170e-b10c128-s1141046784-d204142634.bin.gz";

/// Runs `katago analysis -config configs/analysis_example.cfg -model <model> -override-config "<COMMON>, <extra>"`
/// with the lines of `query_file` (relative to cpp/) on stdin, through `katago_core::command::analysis::analysis`
/// with the burn flex backend; returns the stdout lines. Runs on a 256 MB-stack thread.
pub fn run_analysis(query_file: &str, model: &str, extra: &str) -> Vec<String> {
  let lines: Vec<String> = std::fs::read_to_string(format!("{}/{query_file}", cpp_dir())).expect("query file").lines().map(str::to_string).collect();
  assert!(!lines.is_empty(), "{query_file}: no query lines");
  run_analysis_lines(lines, model, extra)
}

/// As run_analysis, with the input lines given (all available at once, like a file on stdin).
pub fn run_analysis_lines(lines: Vec<String>, model: &str, extra: &str) -> Vec<String> {
  let model = model.to_string();
  let extra = extra.to_string();
  std::thread::Builder::new()
    .stack_size(256 << 20)
    .spawn(move || run_analysis_inner(lines, &model, &extra))
    .unwrap()
    .join()
    .expect("analysis run panicked")
}

fn run_analysis_inner(lines: Vec<String>, model: &str, extra: &str) -> Vec<String> {
  katago_cli::install_clock_and_sleep();
  let cpp = cpp_dir();
  let override_config = format!("{COMMON}, {extra}");
  let cfg = katago_cli::load_config(&[format!("{cpp}/{CFG}")], std::slice::from_ref(&override_config)).expect("config");

  let mut pool = LocalPool::new();
  let spawner: Rc<dyn Spawner> = Rc::new(pool.spawner());
  let nn_host = BurnHost::new("flex", spawner).expect("flex backend");
  let mut input = futures::stream::iter(lines);
  let out: Rc<RefCell<Vec<String>>> = Rc::new(RefCell::new(Vec::new()));
  let out2 = out.clone();
  let host = AnalysisHost {
    args: AnalysisArgs {
      model_file: model.to_string(),
      human_model_file: String::new(),
      num_analysis_threads_cmdline_specified: false,
      num_analysis_threads_cmdline: 0,
      quit_without_waiting: false,
      override_configs: vec![override_config],
    },
    input: &mut input,
    write_stdout: Box::new(move |s: &str| out2.borrow_mut().push(s.to_string())),
    write_stderr: Box::new(|_s: &str| {}),
    nn_host: &nn_host,
  };
  let code = pool.run_until(katago_core::command::analysis::analysis(cfg, host)).unwrap_or_else(|e| panic!("analysis failed: {}", e.what()));
  pool.run_until_stalled();
  assert_eq!(code, 0);
  out.borrow().clone()
}

//------------------------------------------------------------------------------------------------
// The comparator

/// Tolerances of the golden comparator (burn flex vs the Eigen build; NN outputs differ by ~1e-5, see
/// katago-nn-burn tests/nn_parity.rs and search_smoke.rs). Everything that is not a JSON float must be
/// byte-identical: punctuation, key order, strings (ids, moves, pv entries, warnings), integers (visits,
/// edgeVisits, order, turnNumber, pvVisits...), booleans, null, and whether a number is written as an
/// integer or a float. Floats compare by key:
/// - "score" keys (any key containing `score`/`Score`/`lead`/`Lead`): |a - b| <= SCORE_TOL;
/// - weight keys (`weight`, `edgeWeight`, `playSelectionValue`, which scale with visits): |a - b| <=
///   WEIGHT_REL_TOL * max(1, |gold|);
/// - every other float (winrate, lcb, utility, utilityLcb, prior, policy, ownership, ownershipStdev, raw*,
///   noResultValue...): |a - b| <= VALUE_TOL.
pub const VALUE_TOL: f64 = 1e-5;
pub const SCORE_TOL: f64 = 1e-4;
pub const WEIGHT_REL_TOL: f64 = 1e-5;

#[derive(Clone, Debug, PartialEq)]
enum Tok {
  Str(String),
  Num(String),
  Punct(char),
  Lit(String),
}

fn tokenize(line: &str) -> Result<Vec<Tok>, String> {
  let b = line.as_bytes();
  let mut i = 0;
  let mut toks = Vec::new();
  while i < b.len() {
    let c = b[i];
    match c {
      b'"' => {
        let start = i;
        i += 1;
        while i < b.len() && b[i] != b'"' {
          if b[i] == b'\\' {
            i += 1;
          }
          i += 1;
        }
        i += 1;
        toks.push(Tok::Str(line[start..i.min(b.len())].to_string()));
      }
      b'{' | b'}' | b'[' | b']' | b':' | b',' => {
        toks.push(Tok::Punct(c as char));
        i += 1;
      }
      b'-' | b'0'..=b'9' => {
        let start = i;
        while i < b.len() && matches!(b[i], b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9') {
          i += 1;
        }
        toks.push(Tok::Num(line[start..i].to_string()));
      }
      b't' | b'f' | b'n' => {
        let start = i;
        while i < b.len() && b[i].is_ascii_alphabetic() {
          i += 1;
        }
        toks.push(Tok::Lit(line[start..i].to_string()));
      }
      _ => return Err(format!("unexpected byte {:?} at {i}", c as char)),
    }
  }
  Ok(toks)
}

fn is_float_literal(s: &str) -> bool {
  s.contains('.') || s.contains('e') || s.contains('E')
}

fn tol_class(key: &str) -> &'static str {
  if key.contains("score") || key.contains("Score") || key.contains("lead") || key.contains("Lead") {
    "score"
  } else if key == "weight" || key == "edgeWeight" || key == "playSelectionValue" {
    "weight"
  } else {
    "value"
  }
}

/// The outcome of comparing one output with its golden.
#[derive(Debug, Default)]
pub struct Comparison {
  pub num_lines: usize,
  pub byte_identical: bool,
  /// max |a - b| per float key
  pub max_dev: BTreeMap<String, f64>,
  /// the first violation, if any
  pub divergence: Option<String>,
}

impl Comparison {
  pub fn status(&self) -> String {
    if let Some(d) = &self.divergence {
      format!("DIVERGED: {d}")
    } else if self.byte_identical {
      "exact (byte-identical)".to_string()
    } else {
      let mut worst: Vec<(String, f64)> = self.max_dev.iter().map(|(k, v)| (k.clone(), *v)).collect();
      worst.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
      let s: Vec<String> = worst.iter().take(4).map(|(k, v)| format!("{k} {v:.2e}")).collect();
      format!("within tolerance (max dev: {})", s.join(", "))
    }
  }
}

/// Compares `got` with `gold` line by line under the policy above.
pub fn compare(gold: &[String], got: &[String]) -> Comparison {
  let mut c = Comparison { num_lines: gold.len(), byte_identical: gold == got, ..Default::default() };
  if gold.len() != got.len() {
    c.divergence = Some(format!("{} lines vs {} expected", got.len(), gold.len()));
    return c;
  }
  for (ln, (g, o)) in gold.iter().zip(got.iter()).enumerate() {
    if g == o {
      continue;
    }
    let (tg, to) = match (tokenize(g), tokenize(o)) {
      (Ok(a), Ok(b)) => (a, b),
      (e1, e2) => {
        c.divergence = Some(format!("line {}: not tokenizable ({:?} / {:?})", ln + 1, e1.err(), e2.err()));
        return c;
      }
    };
    if tg.len() != to.len() {
      c.divergence = Some(format!("line {}: {} tokens vs {} expected\n  expected: {g}\n  got:      {o}", ln + 1, to.len(), tg.len()));
      return c;
    }
    let mut key = String::new();
    for i in 0..tg.len() {
      if let (Tok::Str(s), Some(Tok::Punct(':'))) = (&tg[i], tg.get(i + 1)) {
        key = s.trim_matches('"').to_string();
      }
      let ok = match (&tg[i], &to[i]) {
        (Tok::Num(a), Tok::Num(b)) if a != b => {
          if is_float_literal(a) && is_float_literal(b) {
            let (x, y): (f64, f64) = (a.parse().unwrap(), b.parse().unwrap());
            let d = (x - y).abs();
            let e = c.max_dev.entry(key.clone()).or_insert(0.0);
            if d > *e {
              *e = d;
            }
            match tol_class(&key) {
              "score" => d <= SCORE_TOL,
              "weight" => d <= WEIGHT_REL_TOL * x.abs().max(1.0),
              _ => d <= VALUE_TOL,
            }
          } else {
            false
          }
        }
        (a, b) => a == b,
      };
      if !ok {
        c.divergence = Some(format!("line {}, key {:?}: {:?} vs expected {:?}\n  expected: {g}\n  got:      {o}", ln + 1, key, to[i], tg[i]));
        return c;
      }
    }
  }
  c
}

pub fn read_golden(name: &str) -> Vec<String> {
  let text = std::fs::read_to_string(golden_path(name)).unwrap_or_else(|e| panic!("{name}: {e}"));
  text.lines().map(str::to_string).collect()
}

/// Runs one golden and asserts it is exact or within tolerance; prints the outcome.
pub fn check_golden(name: &str, query_file: &str, model: &str, extra: &str) -> Comparison {
  let gold = read_golden(name);
  assert!(!gold.is_empty(), "{name}: empty golden");
  let t0 = std::time::Instant::now();
  let got = run_analysis(query_file, model, extra);
  let c = compare(&gold, &got);
  eprintln!("analysis golden {name}: {} lines, {} ({:.1} s)", c.num_lines, c.status(), t0.elapsed().as_secs_f64());
  c
}
