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
  format!("{ROOT}/fixtures/upstream-v1.18.2/analysis/{name}")
}

/// `COMMON` of fixtures/upstream-v1.18.2/analysis/regen.sh (= runcmdtests.sh), minus `logFile=...` (no log
/// files in tests).
pub const COMMON: &str = "logDir=, logTimeStamp=false, logAllRequests=true, logAllResponses=true, logSearchInfo=true, numAnalysisThreads=1, numSearchThreadsPerAnalysisThread=1, nnRandomize=false, nnRandSeed=analysisTest, forDeterministicTesting=true, cudaUseFP16 = false, trtUseFP16 = false, openclUseFP16 = false, cudaUseNHWC = false";
pub const CFG: &str = "configs/analysis_example.cfg";
pub const M6: &str = "tests/models/g170-b6c96-s175395328-d26788732.bin.gz";
pub const M10: &str = "tests/models/g170e-b10c128-s1141046784-d204142634.bin.gz";
/// The human-SL net as regen.sh passes it (`-human-model models/b18c384nbt-humanv0.bin.gz`).
pub const HUMAN: &str = "models/b18c384nbt-humanv0.bin.gz";

/// Runs `katago analysis -config configs/analysis_example.cfg -model <model> -override-config "<COMMON>, <extra>"`
/// with the lines of `query_file` (relative to cpp/) on stdin, through `katago_core::command::analysis::analysis`
/// with the burn flex backend; returns the stdout lines. Runs on a 256 MB-stack thread.
pub fn run_analysis(query_file: &str, model: &str, extra: &str) -> Vec<String> {
  let lines: Vec<String> = std::fs::read_to_string(format!("{}/{query_file}", cpp_dir()))
    .expect("query file")
    .lines()
    .map(str::to_string)
    .collect();
  assert!(!lines.is_empty(), "{query_file}: no query lines");
  run_analysis_lines(lines, model, extra)
}

/// As run_analysis, with the input lines given (all available at once, like a file on stdin).
pub fn run_analysis_lines(lines: Vec<String>, model: &str, extra: &str) -> Vec<String> {
  let (out, code) = run_analysis_bytes(lines.into_iter().map(String::into_bytes).collect(), model, extra);
  match code {
    Ok(c) => assert_eq!(c, 0),
    Err(e) => panic!("analysis failed: {}", e.what()),
  }
  out
}

/// As run_analysis_lines with byte lines; returns the stdout lines and `analysis`'s result (an Err is
/// the exception that terminates the process upstream).
pub fn run_analysis_bytes(lines: Vec<Vec<u8>>, model: &str, extra: &str) -> (Vec<String>, Result<i32, katago_core::core::global::KataError>) {
  run_analysis_bytes_human(lines, model, "", extra)
}

/// As run_analysis_bytes, with `-human-model <human_model>` ("" = none).
pub fn run_analysis_bytes_human(
  lines: Vec<Vec<u8>>,
  model: &str,
  human_model: &str,
  extra: &str,
) -> (Vec<String>, Result<i32, katago_core::core::global::KataError>) {
  let model = model.to_string();
  let human_model = human_model.to_string();
  let extra = extra.to_string();
  std::thread::Builder::new()
    .stack_size(256 << 20)
    .spawn(move || run_analysis_inner(lines, &model, &human_model, &extra, false))
    .unwrap()
    .join()
    .expect("analysis run panicked")
}

/// As run_analysis, but line N+1 of `query_file` is fed only after N responses were written: the fifo
/// handshake of runcmdtests.sh's eval-cache test (regen.sh `eval_cache`), so each request finds the
/// previous one's search done.
pub fn run_analysis_handshake(query_file: &str, model: &str, extra: &str) -> Vec<String> {
  let lines: Vec<Vec<u8>> = std::fs::read_to_string(format!("{}/{query_file}", cpp_dir()))
    .expect("query file")
    .lines()
    .map(|l| l.as_bytes().to_vec())
    .collect();
  assert!(!lines.is_empty(), "{query_file}: no query lines");
  let model = model.to_string();
  let extra = extra.to_string();
  let (out, code) = std::thread::Builder::new()
    .stack_size(256 << 20)
    .spawn(move || run_analysis_inner(lines, &model, "", &extra, true))
    .unwrap()
    .join()
    .expect("analysis run panicked");
  match code {
    Ok(c) => assert_eq!(c, 0),
    Err(e) => panic!("analysis failed: {}", e.what()),
  }
  out
}

fn run_analysis_inner(
  lines: Vec<Vec<u8>>,
  model: &str,
  human_model: &str,
  extra: &str,
  handshake: bool,
) -> (Vec<String>, Result<i32, katago_core::core::global::KataError>) {
  katago_cli::install_clock_and_sleep();
  let cpp = cpp_dir();
  let override_config = format!("{COMMON}, {extra}");
  let cfg = katago_cli::load_config(&[format!("{cpp}/{CFG}")], std::slice::from_ref(&override_config)).expect("config");

  let mut pool = LocalPool::new();
  let spawner: Rc<dyn Spawner> = Rc::new(pool.spawner());
  let nn_host = BurnHost::new("flex", spawner).expect("flex backend");
  let out: Rc<RefCell<Vec<String>>> = Rc::new(RefCell::new(Vec::new()));
  let out2 = out.clone();
  // Handshake mode: the stream yields line i (0-based) once i responses were written; the stdout sink
  // wakes it.
  let waker: Rc<RefCell<Option<std::task::Waker>>> = Rc::new(RefCell::new(None));
  let waker2 = waker.clone();
  let out3 = out.clone();
  let mut lines_iter = lines.into_iter().enumerate();
  let mut pending: Option<(usize, Vec<u8>)> = None;
  let mut input = futures::stream::poll_fn(move |cx| {
    if pending.is_none() {
      pending = lines_iter.next();
    }
    match &pending {
      None => std::task::Poll::Ready(None),
      Some((i, _)) => {
        if !handshake || out3.borrow().len() >= *i {
          std::task::Poll::Ready(pending.take().map(|(_, l)| l))
        } else {
          *waker.borrow_mut() = Some(cx.waker().clone());
          std::task::Poll::Pending
        }
      }
    }
  });
  let host = AnalysisHost {
    args: AnalysisArgs {
      model_file: model.to_string(),
      human_model_file: human_model.to_string(),
      num_analysis_threads_cmdline_specified: false,
      num_analysis_threads_cmdline: 0,
      quit_without_waiting: false,
      override_configs: vec![override_config],
    },
    input: &mut input,
    write_stdout: Box::new(move |s: &str| {
      out2.borrow_mut().push(s.to_string());
      if let Some(w) = waker2.borrow_mut().take() {
        w.wake();
      }
    }),
    write_stderr: Box::new(|_s: &str| {}),
    nn_host: &nn_host,
    // No log backend in tests: the Logger's file / directory setup is the CLI's (tests/cli_*.rs).
    init_logger: Box::new(|_settings| Ok(())),
  };
  let code = pool.run_until(katago_core::command::analysis::analysis(cfg, host));
  pool.run_until_stalled();
  let out = out.borrow().clone();
  (out, code)
}

//------------------------------------------------------------------------------------------------
// The comparator

/// Tolerances of the golden comparator (burn flex vs the Eigen build; NN outputs differ by ~1e-5, see
/// katago-nn-burn tests/nn_parity.rs and search_smoke.rs). Everything that is not a JSON float must be
/// byte-identical: punctuation, key order, strings (ids, moves, pv entries, warnings), integers (visits,
/// edgeVisits, order, turnNumber, pvVisits...), booleans, null, and whether a number is written as an
/// integer or a float. Floats must have the same format class (exponent form or not, and float-sized
/// (<= 9 significant digits, an f32 widened) vs double-sized digit counts: an f32 stored as f64 or the
/// reverse shows there) and compare by key:
/// - "score" keys (any key containing `score`/`Score`/`lead`/`Lead`): |a - b| <= SCORE_TOL, the NN tier's
///   score tolerance of PORTING.md §13 (≤5e-4 score lead/mean: the score heads are scaled by ~20, so their
///   accumulation-order noise is ~20× the value heads');
/// - weight keys (`weight`, `edgeWeight`, `playSelectionValue`, which scale with visits): |a - b| <=
///   WEIGHT_REL_TOL * max(1, |gold|);
/// - probability keys (`prior`, `policy` entries, `humanPrior`): |a - b| <= min(VALUE_TOL, POLICY_REL_TOL * |gold|)
///   (at least 1e-9): absolute 1e-5 above 1e-2, relative 1e-3 below, so a tiny prior cannot drift by
///   several times its own size;
/// - every other float (winrate, lcb, utility, utilityLcb, ownership, ownershipStdev, raw*, noResultValue...):
///   |a - b| <= VALUE_TOL.
pub const VALUE_TOL: f64 = 1e-5;
pub const SCORE_TOL: f64 = 5e-4;
pub const WEIGHT_REL_TOL: f64 = 1e-5;
pub const POLICY_REL_TOL: f64 = 1e-3;

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

/// (exponent form, significant digits of the mantissa)
fn float_format(s: &str) -> (bool, usize) {
  let (mant, exp) = match s.find(['e', 'E']) {
    Some(p) => (&s[..p], true),
    None => (s, false),
  };
  let digits: String = mant.chars().filter(|c| c.is_ascii_digit()).collect();
  let sig = digits.trim_start_matches('0').trim_end_matches('0');
  (exp, sig.len().max(1))
}

/// The format class of two float literals that compare within tolerance:
/// - exponent form: nlohmann writes `x.ye-05` below 1e-4 (and from 1e15): both the same, unless the two
///   values straddle such a boundary;
/// - storage: a JSON value that upstream stored as a float (an f32 widened to f64: exactly representable
///   as f32, and written with more than 4 significant digits, so not just a short decimal like 0.5) must
///   be one in the port too, and the reverse (an f32/f64 mix-up in the port shows up here).
fn same_float_class(a: &str, b: &str) -> bool {
  let (ea, da) = float_format(a);
  let (eb, db) = float_format(b);
  let (x, y): (f64, f64) = (a.parse().unwrap(), b.parse().unwrap());
  let exp_form = |v: f64| v != 0.0 && (v.abs() < 1e-4 || v.abs() >= 1e15);
  if ea != eb && exp_form(x) == exp_form(y) {
    return false;
  }
  let is_f32 = |v: f64, d: usize| d > 4 && (v as f32) as f64 == v;
  let (fa, fb) = (is_f32(x, da), is_f32(y, db));
  if fa != fb && (da > 4 && db > 4) {
    return false;
  }
  true
}

fn tol_class(key: &str) -> &'static str {
  if key.contains("score") || key.contains("Score") || key.contains("lead") || key.contains("Lead") {
    "score"
  } else if key == "weight" || key == "edgeWeight" || key == "playSelectionValue" {
    "weight"
  } else if key == "prior" || key == "policy" || key == "humanPrior" || key == "humanPolicy" {
    "policy"
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

/// Compares one line with its golden line; Err is the first violation.
pub fn compare_line(g: &str, o: &str, max_dev: &mut BTreeMap<String, f64>) -> Result<(), String> {
  if g == o {
    return Ok(());
  }
  let (tg, to) = match (tokenize(g), tokenize(o)) {
    (Ok(a), Ok(b)) => (a, b),
    (e1, e2) => return Err(format!("not tokenizable ({:?} / {:?})\n  expected: {g}\n  got:      {o}", e1.err(), e2.err())),
  };
  if tg.len() != to.len() {
    return Err(format!("{} tokens vs {} expected\n  expected: {g}\n  got:      {o}", to.len(), tg.len()));
  }
  let mut key = String::new();
  for i in 0..tg.len() {
    if let (Tok::Str(s), Some(Tok::Punct(':'))) = (&tg[i], tg.get(i + 1)) {
      key = s.trim_matches('"').to_string();
    }
    let ok = match (&tg[i], &to[i]) {
      (Tok::Num(a), Tok::Num(b)) if a != b => {
        if is_float_literal(a) && is_float_literal(b) && same_float_class(a, b) {
          let (x, y): (f64, f64) = (a.parse().unwrap(), b.parse().unwrap());
          let d = (x - y).abs();
          let e = max_dev.entry(key.clone()).or_insert(0.0);
          if d > *e {
            *e = d;
          }
          match tol_class(&key) {
            "score" => d <= SCORE_TOL,
            "weight" => d <= WEIGHT_REL_TOL * x.abs().max(1.0),
            "policy" => d <= (POLICY_REL_TOL * x.abs()).clamp(1e-9, VALUE_TOL),
            _ => d <= VALUE_TOL,
          }
        } else {
          false
        }
      }
      (a, b) => a == b,
    };
    if !ok {
      return Err(format!("key {:?}: {:?} vs expected {:?}\n  expected: {g}\n  got:      {o}", key, to[i], tg[i]));
    }
  }
  Ok(())
}

/// Compares `got` with `gold` line by line under the policy above.
pub fn compare(gold: &[String], got: &[String]) -> Comparison {
  let mut c = Comparison {
    num_lines: gold.len(),
    byte_identical: gold == got,
    ..Default::default()
  };
  if gold.len() != got.len() {
    c.divergence = Some(format!("{} lines vs {} expected", got.len(), gold.len()));
    return c;
  }
  for (ln, (g, o)) in gold.iter().zip(got.iter()).enumerate() {
    if let Err(e) = compare_line(g, o, &mut c.max_dev) {
      c.divergence = Some(format!("line {}, {e}", ln + 1));
      return c;
    }
  }
  c
}

pub fn read_golden(name: &str) -> Vec<String> {
  let text = std::fs::read_to_string(golden_path(name)).unwrap_or_else(|e| panic!("{name}: {e}"));
  text.lines().map(str::to_string).collect()
}

/// Runs one golden and asserts it is exact or within tolerance; prints the outcome.
/// check_golden with `-human-model <human_model>`.
pub fn check_golden_human(name: &str, query_file: &str, model: &str, human_model: &str, extra: &str) -> Comparison {
  let gold = read_golden(name);
  assert!(!gold.is_empty(), "{name}: empty golden");
  let t0 = std::time::Instant::now();
  let lines: Vec<Vec<u8>> = std::fs::read_to_string(format!("{}/{query_file}", cpp_dir()))
    .expect("query file")
    .lines()
    .map(|l| l.as_bytes().to_vec())
    .collect();
  assert!(!lines.is_empty(), "{query_file}: no query lines");
  let (got, code) = run_analysis_bytes_human(lines, model, human_model, extra);
  match code {
    Ok(c) => assert_eq!(c, 0),
    Err(e) => panic!("analysis failed: {}", e.what()),
  }
  // query_models' maxBatchSize is a Rust choice (the non-Eigen branch: nnMaxBatchSize; the Eigen reference
  // prints its own), as in tests/analysis_protocol_corpus.rs: compared after normalizing it on both sides.
  let gold: Vec<String> = gold.iter().map(|l| normalize_max_batch_size(l)).collect();
  let got: Vec<String> = got.iter().map(|l| normalize_max_batch_size(l)).collect();
  let c = compare(&gold, &got);
  eprintln!(
    "analysis golden {name}: {} lines, {} ({:.1} s)",
    c.num_lines,
    c.status(),
    t0.elapsed().as_secs_f64()
  );
  c
}

/// Replaces every `"maxBatchSize":<n>` value with 0.
pub fn normalize_max_batch_size(line: &str) -> String {
  let key = "\"maxBatchSize\":";
  let mut s = line.to_string();
  let mut from = 0;
  while let Some(p) = s[from..].find(key) {
    let start = from + p + key.len();
    let end = start + s[start..].find([',', '}']).unwrap_or(s.len() - start);
    s.replace_range(start..end, "0");
    from = start;
  }
  s
}

pub fn check_golden(name: &str, query_file: &str, model: &str, extra: &str) -> Comparison {
  let gold = read_golden(name);
  assert!(!gold.is_empty(), "{name}: empty golden");
  let t0 = std::time::Instant::now();
  let got = run_analysis(query_file, model, extra);
  let c = compare(&gold, &got);
  eprintln!(
    "analysis golden {name}: {} lines, {} ({:.1} s)",
    c.num_lines,
    c.status(),
    t0.elapsed().as_secs_f64()
  );
  c
}
