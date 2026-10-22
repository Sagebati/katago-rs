//! katago-cli end to end: the `katago-rs` binary run exactly like upstream's runcmdtests.sh / our regen.sh
//! (`katago-rs analysis -config configs/analysis_example.cfg -model tests/models/<b6c96>.bin.gz
//! -override-config "<COMMON>, maxVisits=100, ..." < tests/analysis/basic.txt`, cwd = upstream cpp/), stdout
//! compared with the Eigen golden by the comparator of tests/common. Not a port of upstream code.

mod common;

use std::io::Write as _;
use std::process::Command;
use std::process::Stdio;

use common::*;

fn run_binary(query_file: &str, extra: &str) -> (Vec<String>, String, i32) {
  let cpp = cpp_dir();
  let input = std::fs::read(format!("{cpp}/{query_file}")).unwrap();
  let mut child = Command::new(env!("CARGO_BIN_EXE_katago-rs"))
    .current_dir(&cpp)
    .args(["analysis", "-config", CFG, "-model", M6, "-override-config", &format!("{COMMON}, {extra}")])
    .stdin(Stdio::piped())
    .stdout(Stdio::piped())
    .stderr(Stdio::piped())
    .spawn()
    .expect("spawn katago-rs");
  child.stdin.take().unwrap().write_all(&input).unwrap();
  let out = child.wait_with_output().unwrap();
  let stdout = String::from_utf8(out.stdout).unwrap();
  (stdout.lines().map(str::to_string).collect(), String::from_utf8_lossy(&out.stderr).into_owned(), out.status.code().unwrap_or(-1))
}

#[test]
fn cli_basic_matches_golden() {
  let (got, stderr, code) = run_binary("tests/analysis/basic.txt", "maxVisits=100, maxPlayouts=10000, rootSymmetryPruning=false");
  assert_eq!(code, 0, "exit code; stderr:\n{stderr}");
  let gold = read_golden("basic.txt.stdout");
  let c = compare(&gold, &got);
  eprintln!("katago-rs basic.txt: {} lines, {}", c.num_lines, c.status());
  assert!(c.num_lines >= 1);
  assert!(c.divergence.is_none(), "{}\nstderr:\n{stderr}", c.status());
  // The log goes to stderr (logToStderr defaults to true for analysis), ending like upstream's.
  assert!(stderr.contains(": All cleaned up, quitting"), "stderr:\n{stderr}");
}

#[test]
fn cli_clear_cache_byte_exact() {
  let (got, stderr, code) = run_binary("tests/analysis/clear_cache.txt", "maxVisits=100, maxPlayouts=10000, rootSymmetryPruning=false");
  assert_eq!(code, 0, "stderr:\n{stderr}");
  assert_eq!(got, read_golden("clear_cache.txt.stdout"));
}

#[test]
fn cli_bad_arguments() {
  let out = Command::new(env!("CARGO_BIN_EXE_katago-rs")).args(["analysis", "-config"]).output().unwrap();
  assert_eq!(out.status.code(), Some(1));
  let out = Command::new(env!("CARGO_BIN_EXE_katago-rs"))
    .current_dir(cpp_dir())
    .args(["analysis", "-config", CFG, "-model", "tests/models/some-other-net.bin.gz", "-override-config", "logDir="])
    .stdin(Stdio::null())
    .output()
    .unwrap();
  assert_eq!(out.status.code(), Some(1));
  assert!(String::from_utf8_lossy(&out.stderr).contains("no runtime .bin.gz loader"));
}
