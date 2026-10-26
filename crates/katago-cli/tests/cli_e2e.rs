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
  (
    stdout.lines().map(str::to_string).collect(),
    String::from_utf8_lossy(&out.stderr).into_owned(),
    out.status.code().unwrap_or(-1),
  )
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

/// TCLAP's parse error (stderr, exit 1), as upstream's `katago analysis -config` prints it.
#[test]
fn cli_bad_arguments() {
  let out = Command::new(env!("CARGO_BIN_EXE_katago-rs")).args(["analysis", "-config"]).output().unwrap();
  assert_eq!(out.status.code(), Some(1));
  assert_eq!(
    String::from_utf8_lossy(&out.stderr),
    "PARSE ERROR: Argument: (-config)\n             Missing a value for this argument!\n\nBrief USAGE: \n   analysis  -config <FILE> ... [-model <FILE>] [-human-model <FILE>]\n             [...other flags...]\n\nFor complete USAGE and HELP type: \n   analysis -help\n\n"
  );
  // `-flag=value` and `--flag` are not TCLAP forms here.
  for bad in [["-config=configs/analysis_example.cfg"], ["--config"]] {
    let out = Command::new(env!("CARGO_BIN_EXE_katago-rs")).arg("analysis").args(bad).output().unwrap();
    assert_eq!(out.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&out.stderr).starts_with(&format!("PARSE ERROR: Argument: {}\n             Couldn't find match for argument\n", bad[0])));
  }
  // -help: upstream's usage text on stdout, exit 0.
  let out = Command::new(env!("CARGO_BIN_EXE_katago-rs")).args(["analysis", "-help"]).output().unwrap();
  assert_eq!(out.status.code(), Some(0));
  assert_eq!(String::from_utf8_lossy(&out.stdout), katago_cli::HELP_TEXT);
}

/// An exception escaping MainCmds::analysis: std::terminate's message on stderr and SIGABRT (exit 134), as
/// upstream's Linux build (fixtures/upstream-v1.16.5/analysis-protocol has the config / argv cases).
#[test]
fn cli_uncaught_exception_aborts() {
  use std::os::unix::process::ExitStatusExt;
  let out = Command::new(env!("CARGO_BIN_EXE_katago-rs"))
    .current_dir(cpp_dir())
    .args([
      "analysis",
      "-config",
      CFG,
      "-model",
      "tests/models/some-other-net.bin.gz",
      "-override-config",
      "logDir=",
    ])
    .stdin(Stdio::null())
    .output()
    .unwrap();
  assert_eq!(out.status.signal(), Some(6), "{:?}", out.status);
  let err = String::from_utf8_lossy(&out.stderr);
  assert!(
    err.ends_with("terminate called after throwing an instance of 'StringError'\n  what():  Error loading or parsing model file tests/models/some-other-net.bin.gz: Could not open file tests/models/some-other-net.bin.gz - does not exist or invalid permissions?\n"),
    "{err}"
  );
}

/// KataGoCommandLine::getModelFile: no -model -> default_model.bin.gz / .txt.gz in the executable's directory,
/// then ~/.katago; neither -> its StringError (uncaught). A default that is a symlink to a compiled network's
/// file is that network.
#[test]
fn cli_default_model_lookup() {
  use std::os::unix::process::ExitStatusExt;
  let home = std::env::temp_dir().join(format!("katago-rs-home-{}", std::process::id()));
  let _ = std::fs::remove_dir_all(&home);
  std::fs::create_dir_all(&home).unwrap();
  let run = || {
    let mut child = Command::new(env!("CARGO_BIN_EXE_katago-rs"))
      .current_dir(cpp_dir())
      .env("HOME", &home)
      .args(["analysis", "-config", CFG, "-override-config", "logDir=, logToStderr=false"])
      .stdin(Stdio::piped())
      .stdout(Stdio::piped())
      .stderr(Stdio::piped())
      .spawn()
      .unwrap();
    child.stdin.take().unwrap().write_all(b"{\"id\":\"v\",\"action\":\"query_version\"}\n").unwrap();
    child.wait_with_output().unwrap()
  };
  let out = run();
  assert_eq!(out.status.signal(), Some(6));
  let exe_dir = std::path::Path::new(env!("CARGO_BIN_EXE_katago-rs"))
    .parent()
    .unwrap()
    .to_string_lossy()
    .into_owned();
  assert_eq!(
    String::from_utf8_lossy(&out.stderr),
    format!(
      "terminate called after throwing an instance of 'StringError'\n  what():  -model MODELFILENAME.bin.gz was not specified to tell KataGo where to find the neural net model, and default was not found at {exe_dir}/default_model.bin.gz\n"
    )
  );
  std::fs::create_dir_all(home.join(".katago")).unwrap();
  std::os::unix::fs::symlink(format!("{}/{M6}", cpp_dir()), home.join(".katago/default_model.bin.gz")).unwrap();
  let out = run();
  assert_eq!(out.status.code(), Some(0), "{}", String::from_utf8_lossy(&out.stderr));
  assert!(String::from_utf8_lossy(&out.stdout).starts_with("{\"action\":\"query_version\""));
  let _ = std::fs::remove_dir_all(&home);
}

/// The Logger's outputs from the config (logger.cpp): logToStdout puts the log lines on stdout; logDir makes
/// `<dir>/<YYYYmmdd-HHMMSS>-<8 hex>.log` with local-time stamps (`%F %T%z: `); an unopenable logFile is a
/// warning on the log and stderr; an uncreatable logDir is "Error creating directory" (uncaught).
#[test]
fn cli_logger_outputs() {
  use std::os::unix::process::ExitStatusExt;
  let dir = std::env::temp_dir().join(format!("katago-rs-logs-{}", std::process::id()));
  let _ = std::fs::remove_dir_all(&dir);
  let run = |ov: &str, tz: &str| {
    let mut child = Command::new(env!("CARGO_BIN_EXE_katago-rs"))
      .current_dir(cpp_dir())
      .env("TZ", tz)
      .args(["analysis", "-config", CFG, "-model", M6, "-override-config", ov])
      .stdin(Stdio::piped())
      .stdout(Stdio::piped())
      .stderr(Stdio::piped())
      .spawn()
      .unwrap();
    child.stdin.take().unwrap().write_all(b"{\"id\":\"v\",\"action\":\"query_version\"}\n").unwrap();
    child.wait_with_output().unwrap()
  };
  // logToStdout
  let out = run("logDir=, logTimeStamp=false, logToStdout=true, logToStderr=false", "UTC");
  let stdout = String::from_utf8_lossy(&out.stdout);
  assert!(stdout.contains("\n: Analysis Engine starting...\n") || stdout.starts_with(": "), "{stdout}");
  assert!(
    stdout.contains("{\"action\":\"query_version\"") && stdout.ends_with(": All cleaned up, quitting\n"),
    "{stdout}"
  );
  // logDir, local time stamps (TZ=JST-9: +0900)
  let out = run(&format!("logDir={}", dir.display()), "JST-9");
  assert_eq!(out.status.code(), Some(0), "{:?} {}", out.status, String::from_utf8_lossy(&out.stderr));
  let files: Vec<_> = std::fs::read_dir(&dir)
    .unwrap()
    .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
    .collect();
  assert_eq!(files.len(), 1, "{files:?}");
  let f = &files[0];
  assert!(
    f.len() == "20261023-071500-0123ABCD.log".len() && f.ends_with(".log") && f.as_bytes()[8] == b'-' && f.as_bytes()[15] == b'-',
    "{f}"
  );
  let log = std::fs::read_to_string(dir.join(f)).unwrap();
  let first = log.lines().next().unwrap();
  // "2026-10-23 16:15:00+0900: Running with following config:"
  assert!(
    first.len() > 26 && &first[19..26] == "+0900: " && first.ends_with("Running with following config:"),
    "{first}"
  );
  let stderr = String::from_utf8_lossy(&out.stderr);
  assert!(stderr.lines().next().unwrap().ends_with("+0900: Running with following config:"), "{stderr}");
  let _ = std::fs::remove_dir_all(&dir);
  // logFile that cannot be opened
  let out = run("logDir=, logFile=/nonexistent-dir/x.log, logTimeStamp=false", "UTC");
  assert_eq!(out.status.code(), Some(0));
  let stderr = String::from_utf8_lossy(&out.stderr);
  assert!(
    stderr.starts_with(": WARNING: could not open file for logging: Could not write to file /nonexistent-dir/x.log - invalid path or permissions?\nWARNING: could not open file for logging: Could not write to file /nonexistent-dir/x.log - invalid path or permissions?\n: Running with following config:\n"),
    "{stderr}"
  );
  // logDir that cannot be created
  let out = run("logDir=/proc/nope", "UTC");
  assert_eq!(out.status.signal(), Some(6));
  assert_eq!(
    String::from_utf8_lossy(&out.stderr),
    "terminate called after throwing an instance of 'StringError'\n  what():  Error creating directory: /proc/nope\n"
  );
}
