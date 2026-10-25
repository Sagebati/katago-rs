//! Host side of the native `katago-rs analysis` drop-in (plan §1): everything `MainCmds::analysis`
//! gets from the process in upstream KataGo, for the ported library function
//! `katago_core::command::analysis::analysis`:
//! - argv: a TCLAP 1.2.5 emulation for the analysis subcommand's arguments (`-model`, `-config`...,
//!   `-override-config`..., `-human-model`, `-analysis-threads`, `-quit-without-waiting`, `--`, `-version`,
//!   `-h`/`-help`) with upstream's error texts (`PARSE ERROR: ...` + brief usage, exit 1), plus the
//!   katago-rs-only `-backend flex|wgpu` -> [`parse_args`];
//! - `KataGoCommandLine::getModelFile` (default model lookup) -> [`get_model_file`], `getConfig` (config
//!   files, `@include` through the file system, `-override-config` with `Setup::getMutexKeySets`) ->
//!   [`load_config`];
//! - the Logger's outputs (stdout / stderr / `logFile` / `logDir` / `logDirDated`, local-time stamps) ->
//!   [`install_logger`] + [`init_logger`] (AnalysisHost::init_logger);
//! - the clock and the timed wait (`core::timer::set_clock`, `port::executor::set_sleep`) -> [`install_clock_and_sleep`];
//! - an exception escaping `MainCmds::analysis` (no catch in upstream's main on Linux): libstdc++'s
//!   terminate message and `abort()` -> [`terminate_uncaught`];
//! - the NN backend: the burn networks compiled into katago-nn-burn -> [`BurnHost`].
//!
//! Model files: there is no runtime `.bin.gz` loader. `-model` names one of the networks compiled into
//! katago-nn-burn (`g170-b6c96`, `g170e-b10c128`, `kata1-b18c384nbt`, and the human-SL net
//! `b18c384nbt-humanv0` for `-human-model` or `-model`), matched on the file name (with or
//! without directories and `.bin.gz` / `.txt.gz` / `.onnx` / `.bpk`, and the full internal name such as
//! `g170-b6c96-s175395328-d26788732`; a symlink is also matched by its target's name). The weights are the
//! `.bpk` generated at build time (`katago_nn_burn::<net>::BPK_PATH`) unless `-model` is itself a `.bpk`
//! file, which is then read. Any other network is an error (upstream: an unloadable model file).
//! Not a port of upstream code (idiomatic Rust at the API edge, PORTING.md §1).

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::io::Write as _;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::OnceLock;
use std::task::Poll;
use std::task::Waker;
use std::time::Instant;

use burn::prelude::Device;
use katago_core::command::analysis::LoggerSettings;
use katago_core::core::config_parser::ConfigFileReader;
use katago_core::core::config_parser::ConfigParser;
use katago_core::core::global::Global;
use katago_core::core::global::KataError;
use katago_core::core::rand::Rand;
use katago_core::neuralnet::nninterface::NeuralNet;
use katago_core::port::executor::LocalBoxFuture;
use katago_core::port::executor::Spawner;
use katago_core::program::setup::NNEvaluatorHost;
use katago_core::program::setup::Setup;

//------------------------------------------------------------------------------------------------
// Command line (TCLAP 1.2.5 as KataGoCommandLine configures it: TCLAP_NAMESTARTSTRING "-", delimiter ' ')

/// The parsed command line of `katago-rs analysis`.
#[derive(Clone, Debug, Default)]
pub struct CliArgs {
  /// `-model` ("" when not given: see get_model_file)
  pub model_file: String,
  pub human_model_file: String,
  pub config_files: Vec<String>,
  pub override_configs: Vec<String>,
  /// `-analysis-threads` (isSet, value)
  pub num_analysis_threads: Option<i32>,
  pub quit_without_waiting: bool,
  /// `flex` (default), `wgpu` (needs the `wgpu` feature) or `rocm` (the `rocm` feature). katago-rs only.
  pub backend: String,
}

/// What parse_args decided.
#[derive(Clone, Debug)]
pub enum ParsedArgs {
  Run(CliArgs),
  /// `-h` / `-help`: print [`HELP_TEXT`] to stdout, exit 0.
  Help,
  /// `-version`: print [`version_text`] to stdout, exit 0.
  Version,
}

/// A TCLAP ArgException: `argId()` and `error()`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArgError {
  /// `_argId` ("undefined" when none)
  pub arg_id: String,
  pub error: String,
}

impl ArgError {
  fn new(error: &str, arg_id: &str) -> ArgError {
    ArgError { arg_id: arg_id.to_string(), error: error.to_string() }
  }
  /// `ArgException::argId()`
  pub fn arg_id_text(&self) -> String {
    if self.arg_id == "undefined" { " ".to_string() } else { "Argument: ".to_string() + &self.arg_id }
  }
  /// `StdOutput::failure` (stderr, then `exit(1)`).
  pub fn failure_text(&self) -> String {
    format!(
      "PARSE ERROR: {}\n             {}\n\nBrief USAGE: \n{}\nFor complete USAGE and HELP type: \n   analysis -help\n\n",
      self.arg_id_text(),
      self.error,
      BRIEF_USAGE
    )
  }
}

/// KataHelpOutput::_shortUsage for the analysis subcommand (progName "analysis").
pub const BRIEF_USAGE: &str = "   analysis  -config <FILE> ... [-model <FILE>] [-human-model <FILE>]\n             [...other flags...]\n";

/// KataHelpOutput::usage for the analysis subcommand (upstream `katago analysis -help`, stdout).
pub const HELP_TEXT: &str = concat!(
  "\n",
  "DESCRIPTION: \n",
  "\n",
  "   Run KataGo parallel JSON-based analysis engine.\n",
  "\n",
  "USAGE: \n",
  "\n",
  "   analysis  -config <FILE> ... [-model <FILE>] [-human-model <FILE>]\n",
  "             [...other flags...]\n",
  "\n",
  "\n",
  "Where: \n",
  "\n",
  "   -config <FILE>  (accepted multiple times)\n",
  "     (required)  Config file(s) to use, can be one or multiple files (see\n",
  "     analysis_example.cfg or configs/analysis_example.cfg).\n",
  "\n",
  "   -model <FILE>\n",
  "     Neural net model file. Defaults to: (dir containing katago.exe, or\n",
  "     else ~/.katago)/default_model.bin.gz\n",
  "\n",
  "   -human-model <FILE>\n",
  "     Human SL neural net model file\n",
  "\n",
  "   -override-config <KEYVALUEPAIRS>  (accepted multiple times)\n",
  "     Override config parameters. Format: \"key=value, key=value,...\"\n",
  "\n",
  "   -analysis-threads <THREADS>\n",
  "     Analyze up to this many positions in parallel. Equivalent to\n",
  "     numAnalysisThreads in the config.\n",
  "\n",
  "   -quit-without-waiting\n",
  "     When stdin is closed, quit quickly without waiting for queued tasks\n",
  "\n",
  "   --,  -ignore_rest\n",
  "     Ignores the rest of the labeled arguments following this flag.\n",
  "\n",
  "   -version\n",
  "     Displays version information and exits.\n",
  "\n",
  "   -h,  -help\n",
  "     Displays usage information and exits.\n",
  "\n",
  "\n",
  "\n",
);

/// `Version::getKataGoVersionFullInfo()` as TCLAP's StdOutput::version prints it.
/// PORT NOTE: no compile time is recorded; the backend line names burn and the `-backend` values this build
/// accepts (katago-ws reports them).
pub fn version_text() -> String {
  format!(
    "\nanalysis  version: {}\nGit revision: {}\nUsing burn backend (katago-rs: {})\n\n\n",
    katago_core::command::analysis::KATAGO_VERSION_FOR_HELP,
    katago_core::command::analysis::GIT_REVISION,
    compiled_backends().join(", ")
  )
}

/// The `-backend` values compiled into this build.
pub fn compiled_backends() -> Vec<&'static str> {
  let mut out = Vec::new();
  #[cfg(feature = "flex")]
  out.push("flex");
  #[cfg(feature = "wgpu")]
  out.push("wgpu");
  #[cfg(feature = "rocm")]
  out.push("rocm");
  out
}

/// `istringstream is(s); while(is.good()) { if(is.peek() != EOF) is >> int; else break; n++; }` (TCLAP
/// ExtractValue for `int`): Ok(value) or the ArgParseException text.
fn extract_int(s: &str) -> Result<i32, String> {
  let b = s.as_bytes();
  let is_space = |c: u8| matches!(c, b' ' | b'\t' | b'\n' | 0x0b | 0x0c | b'\r');
  let mut pos = 0usize;
  let mut values_read = 0;
  let mut value: i32 = 0;
  let mut fail = false;
  loop {
    // is.good(): neither eof nor fail
    if pos >= b.len() {
      break; // peek() == EOF (or eofbit already set)
    }
    // operator>>(int): skip whitespace, optional sign, digits
    while pos < b.len() && is_space(b[pos]) {
      pos += 1;
    }
    let start = pos;
    if pos < b.len() && (b[pos] == b'+' || b[pos] == b'-') {
      pos += 1;
    }
    let digits_start = pos;
    while pos < b.len() && b[pos].is_ascii_digit() {
      pos += 1;
    }
    values_read += 1;
    if pos == digits_start {
      fail = true;
      break;
    }
    match s[start..pos].parse::<i32>() {
      Ok(v) => value = v,
      Err(_) => {
        fail = true;
        break;
      }
    }
    if pos >= b.len() {
      break; // eofbit
    }
  }
  if fail {
    return Err(format!("Couldn't read argument value from string '{s}'"));
  }
  if values_read > 1 {
    return Err(format!("More than one valid value parsed from string '{s}'"));
  }
  Ok(value)
}

/// Parses the arguments after the subcommand like `KataGoCommandLine::parseArgs` (TCLAP CmdLine::parse):
/// one `-name value` (or a single argv element `-name value`) per value argument, `-config` and
/// `-override-config` repeatable, `--` ignores the rest, a lone `-` is accepted, anything else is
/// "Couldn't find match for argument". `-backend` is katago-rs's own (a value argument).
pub fn parse_args(args: &[String]) -> Result<ParsedArgs, ArgError> {
  let mut out = CliArgs { backend: "flex".to_string(), ..CliArgs::default() };
  let mut model_set = false;
  let mut human_set = false;
  let mut threads_set = false;
  let mut quit_set = false;
  let mut backend_set = false;
  let mut ignore_rest = false;
  let mut i = 0usize;
  let has_blanks = |s: &str| s.bytes().skip(1).any(|c| c == 7);
  while i < args.len() {
    let arg = args[i].as_str();
    let mut matched = false;
    if !ignore_rest {
      // Switches (TCLAP: no flag/value split): processed in TCLAP's list order (the last added first).
      let switch = match arg {
        "-quit-without-waiting" => Some(("quit-without-waiting", "(-quit-without-waiting)")),
        "--" | "-ignore_rest" => Some(("ignore_rest", "- (-ignore_rest)")),
        "-version" => Some(("version", "(-version)")),
        "-h" | "-help" => Some(("help", "-h (-help)")),
        _ => None,
      };
      if let Some((name, id)) = switch {
        matched = true;
        match name {
          "quit-without-waiting" => {
            if quit_set {
              return Err(ArgError::new("Argument already set!", id));
            }
            quit_set = true;
            out.quit_without_waiting = true;
          }
          "ignore_rest" => ignore_rest = true,
          "version" => return Ok(ParsedArgs::Version),
          _ => return Ok(ParsedArgs::Help),
        }
      } else if !has_blanks(arg) {
        // Value args: trimFlag splits "-name value" at the first ' ' (when past index 1).
        let (flag, inline_value) = match arg.find(' ') {
          Some(stop) if stop > 1 => (&arg[..stop], Some(arg[stop + 1..].to_string())),
          _ => (arg, None),
        };
        let name = match flag {
          "-analysis-threads" | "-override-config" | "-human-model" | "-model" | "-config" | "-backend" => Some(&flag[1..]),
          _ => None,
        };
        if let Some(name) = name {
          matched = true;
          let id = format!("(-{name})");
          let already = match name {
            "model" => model_set,
            "human-model" => human_set,
            "analysis-threads" => threads_set,
            "backend" => backend_set,
            _ => false, // MultiArg
          };
          if already {
            return Err(ArgError::new("Argument already set!", &id));
          }
          let value: String = match inline_value {
            Some(v) if !v.is_empty() => v,
            _ => {
              i += 1;
              match args.get(i) {
                Some(v) => v.clone(),
                None => return Err(ArgError::new("Missing a value for this argument!", &id)),
              }
            }
          };
          match name {
            "model" => {
              out.model_file = value;
              model_set = true;
            }
            "human-model" => {
              out.human_model_file = value;
              human_set = true;
            }
            "config" => out.config_files.push(value),
            "override-config" => out.override_configs.push(value),
            "analysis-threads" => {
              out.num_analysis_threads = Some(extract_int(&value).map_err(|e| ArgError::new(&e, &id))?);
              threads_set = true;
            }
            _ => {
              out.backend = value;
              backend_set = true;
            }
          }
        }
      }
      // _emptyCombined: "-" alone (or followed only by blank chars)
      if !matched && arg.starts_with('-') && arg.bytes().skip(1).all(|c| c == 7) {
        matched = true;
      }
    }
    if !matched && !ignore_rest {
      return Err(ArgError::new("Couldn't find match for argument", arg));
    }
    i += 1;
  }
  if out.config_files.is_empty() {
    return Err(ArgError::new("Required argument missing: config", "undefined"));
  }
  Ok(ParsedArgs::Run(out))
}

/// `HomeData::getDefaultFilesDirs()` (homedata.cpp, Unix): the executable's directory, then `$HOME/.katago`.
fn default_files_dirs() -> Vec<String> {
  let mut ret = Vec::new();
  if let Some(dir) = std::fs::read_link("/proc/self/exe").or_else(|_| std::env::current_exe()).ok().as_deref().and_then(std::path::Path::parent) {
    ret.push(dir.to_string_lossy().into_owned());
  }
  // PORT NOTE: without HOME upstream asks getpwuid_r; here the fallback is upstream's last resort "./.katago".
  match std::env::var("HOME") {
    Ok(home) => ret.push(home + "/.katago"),
    Err(_) => ret.push("./.katago".to_string()),
  }
  ret
}

/// `KataGoCommandLine::getModelFile` (commandline.cpp): `-model`, else the first existing default
/// (`default_model.bin.gz` / `.txt.gz` in each default dir), else its StringError.
pub fn get_model_file(cli: &CliArgs) -> Result<String, KataError> {
  if !cli.model_file.is_empty() {
    return Ok(cli.model_file.clone());
  }
  let mut paths: Vec<String> = Vec::new();
  for d in default_files_dirs() {
    paths.push(d.clone() + "/default_model.bin.gz");
    paths.push(d + "/default_model.txt.gz");
  }
  let path_for_err_msg = paths.first().cloned().unwrap_or_else(|| "(dir containing katago.exe, or else ~/.katago)/default_model.bin.gz".to_string());
  for path in &paths {
    if std::path::Path::new(path).exists() {
      return Ok(path.clone());
    }
  }
  Err(KataError::StringError(
    "-model MODELFILENAME.bin.gz was not specified to tell KataGo where to find the neural net model, and default was not found at ".to_string()
      + &path_for_err_msg,
  ))
}

//------------------------------------------------------------------------------------------------
// Uncaught exceptions

/// The C++ type name libstdc++'s verbose terminate handler prints for the exception behind `e`.
pub fn exception_type_name(e: &KataError) -> String {
  match e {
    KataError::StringError(_) => "StringError".to_string(),
    KataError::IOError(_) => "IOError".to_string(),
    KataError::ConfigParsingError(_) => "ConfigParsingError".to_string(),
    KataError::ValueError(_) => "ValueError".to_string(),
    KataError::CommandError(_) => "CommandError".to_string(),
    KataError::NlohmannJson(m) => {
      // "[json.exception.<class>.<id>] ..."
      let class = m.strip_prefix("[json.exception.").and_then(|r| r.split('.').next()).unwrap_or("exception");
      format!("nlohmann::detail::{class}")
    }
  }
}

/// An exception escaping `main` (upstream's main has no catch on Linux): `std::terminate` prints
/// "terminate called after throwing an instance of '<type>'\n  what():  <what>\n" and aborts (SIGABRT,
/// shell exit status 134).
pub fn terminate_uncaught(e: &KataError) -> ! {
  let _ = std::io::stdout().flush();
  let msg = format!("terminate called after throwing an instance of '{}'\n  what():  {}\n", exception_type_name(e), e.what());
  let _ = std::io::stderr().write_all(msg.as_bytes());
  std::process::abort()
}

//------------------------------------------------------------------------------------------------
// Config

/// The file reader for `ConfigParser` (`FileUtils::open` + read), with upstream's error text.
pub fn file_reader() -> ConfigFileReader {
  Rc::new(|path: &str| -> Result<String, KataError> {
    std::fs::read_to_string(path).map_err(|_| KataError::IOError(format!("Could not open file {path} - does not exist or invalid permissions?")))
  })
}

/// `KataGoCommandLine::getConfig` + `maybeApplyOverrideConfigArg` (cpp/command/commandline.cpp).
pub fn load_config(config_files: &[String], override_configs: &[String]) -> Result<ConfigParser, KataError> {
  let mut cfg = ConfigParser::new(false, true);
  cfg.set_include_reader(file_reader());
  cfg.initialize(&config_files[0])?;
  for override_file in &config_files[1..] {
    cfg.override_keys(override_file)?;
  }
  for override_config in override_configs {
    if !override_config.is_empty() {
      let newkvs: BTreeMap<String, String> = ConfigParser::parse_comma_separated(override_config)?;
      //HACK to avoid a common possible conflict - if we specify some of the rules options on one side, the other side should be erased.
      let mutex_key_sets = Setup::get_mutex_key_sets();
      cfg.override_keys_newkvs_mutex_key_sets(&newkvs, &mutex_key_sets);
    }
  }
  Ok(cfg)
}

//------------------------------------------------------------------------------------------------
// Clock, sleep

static START: OnceLock<Instant> = OnceLock::new();

fn clock() -> f64 {
  START.get_or_init(Instant::now).elapsed().as_secs_f64()
}

/// The shared timer behind [`sleep`]: one thread waits for the earliest deadline of a map keyed by
/// (deadline, id) and wakes that future. A dropped [`Sleep`] removes its entry, so abandoned waits (an
/// analysis callback loop dropped when its search ends, with a period up to 1e6 s) leave nothing behind.
struct TimerState {
  /// (deadline, id) -> (fired, waker)
  entries: BTreeMap<(Instant, u64), SleepSlot>,
  next_id: u64,
}

struct Timer {
  state: Mutex<TimerState>,
  changed: std::sync::Condvar,
}

static TIMER: OnceLock<&'static Timer> = OnceLock::new();
static TIMER_THREADS_STARTED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn timer() -> &'static Timer {
  TIMER.get_or_init(|| {
    let t: &'static Timer = Box::leak(Box::new(Timer { state: Mutex::new(TimerState { entries: BTreeMap::new(), next_id: 0 }), changed: std::sync::Condvar::new() }));
    TIMER_THREADS_STARTED.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    std::thread::Builder::new()
      .name("katago-rs-timer".to_string())
      .spawn(move || {
        let mut st = t.state.lock().unwrap();
        loop {
          let now = Instant::now();
          // fire everything due
          while let Some((&key, _)) = st.entries.iter().next() {
            if key.0 > now {
              break;
            }
            let slot = st.entries.remove(&key).expect("first key exists");
            let mut slot = slot.lock().unwrap();
            slot.0 = true;
            if let Some(w) = slot.1.take() {
              w.wake();
            }
          }
          st = match st.entries.keys().next().map(|k| k.0) {
            Some(deadline) => t.changed.wait_timeout(st, deadline.saturating_duration_since(Instant::now())).unwrap().0,
            None => t.changed.wait(st).unwrap(),
          };
        }
      })
      .expect("spawn the timer thread");
    t
  })
}

/// Number of timer threads ever started (tests: one per process, whatever the number of sleeps).
pub fn timer_threads_started() -> usize {
  TIMER_THREADS_STARTED.load(std::sync::atomic::Ordering::SeqCst)
}

/// Number of pending (not fired, not dropped) timed waits.
pub fn pending_sleeps() -> usize {
  timer().state.lock().unwrap().entries.len()
}

/// (fired, waker) of one pending [`sleep`].
type SleepSlot = Arc<Mutex<(bool, Option<Waker>)>>;

/// The future of [`sleep`].
struct Sleep {
  key: (Instant, u64),
  slot: SleepSlot,
}

impl std::future::Future for Sleep {
  type Output = ();
  fn poll(self: std::pin::Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> Poll<()> {
    let mut slot = self.slot.lock().unwrap();
    if slot.0 {
      Poll::Ready(())
    } else {
      slot.1 = Some(cx.waker().clone());
      Poll::Pending
    }
  }
}

impl Drop for Sleep {
  fn drop(&mut self) {
    let t = timer();
    let mut st = t.state.lock().unwrap();
    if st.entries.remove(&self.key).is_some() {
      t.changed.notify_one();
    }
  }
}

/// A timed wait for `port::executor::sleep` on the shared timer thread (LocalPool parks its thread while
/// idle, and a wake from another thread unparks it). Durations beyond ~31 years are capped.
fn sleep(seconds: f64) -> LocalBoxFuture<'static, ()> {
  let secs = if seconds.is_finite() && seconds > 0.0 { seconds.min(1e9) } else { 0.0 };
  let deadline = Instant::now() + std::time::Duration::from_secs_f64(secs);
  let t = timer();
  let slot = Arc::new(Mutex::new((false, None)));
  let key = {
    let mut st = t.state.lock().unwrap();
    let key = (deadline, st.next_id);
    st.next_id += 1;
    st.entries.insert(key, slot.clone());
    key
  };
  t.changed.notify_one();
  Box::pin(Sleep { key, slot })
}

/// Installs the wall clock (`core::timer::set_clock`) and the timed wait (`port::executor::set_sleep`).
pub fn install_clock_and_sleep() {
  START.get_or_init(Instant::now);
  katago_core::core::timer::set_clock(clock);
  katago_core::port::executor::set_sleep(sleep);
}

//------------------------------------------------------------------------------------------------
// Logger (core/logger.cpp is replaced by the log crate; this is its backend)

struct LoggerState {
  to_stdout: bool,
  to_stderr: bool,
  log_time: bool,
  files: Vec<std::fs::File>,
}

/// The log backend. Until init_logger runs (the Logger constructor) there is no Logger upstream, so
/// messages are dropped.
struct CliLogger {
  state: Mutex<Option<LoggerState>>,
}

static LOGGER: CliLogger = CliLogger { state: Mutex::new(None) };

/// `strftime` of the local time now (`DateTime::writeTimeToStream` / `getCompactDateTimeString`).
fn local_time_string(fmt: &str) -> String {
  let fmt_c = std::ffi::CString::new(fmt).expect("format has no NUL");
  // SAFETY: time(NULL), localtime_r into a zeroed tm, strftime into a 128-byte buffer with its length.
  unsafe {
    let t: libc::time_t = libc::time(std::ptr::null_mut());
    let mut tm: libc::tm = std::mem::zeroed();
    libc::localtime_r(&t, &mut tm);
    let mut buf = [0u8; 128];
    let n = libc::strftime(buf.as_mut_ptr() as *mut libc::c_char, buf.len(), fmt_c.as_ptr(), &tm);
    String::from_utf8_lossy(&buf[..n]).into_owned()
  }
}

/// `DateTime::timeFormat` (Unix): "%F %T%z: ".
const TIME_FORMAT: &str = "%F %T%z: ";

impl CliLogger {
  /// `Logger::write(str, true)` with the state locked.
  fn write_locked(state: &mut LoggerState, msg: &str) {
    let line = if state.log_time { local_time_string(TIME_FORMAT) + msg + "\n" } else { ": ".to_string() + msg + "\n" };
    if state.to_stdout {
      let stdout = std::io::stdout();
      let mut lock = stdout.lock();
      let _ = lock.write_all(line.as_bytes());
      let _ = lock.flush();
    }
    if state.to_stderr {
      let _ = std::io::stderr().write_all(line.as_bytes());
    }
    for f in state.files.iter_mut() {
      let _ = f.write_all(line.as_bytes());
      let _ = f.flush();
    }
  }
}

impl log::Log for CliLogger {
  fn enabled(&self, metadata: &log::Metadata) -> bool {
    // The GPU stack (burn, cubecl and its pliron IR, wgpu, naga) logs kernel IR at Info: one analysis run wrote a
    // 2.6M-line log file. Only its warnings and errors reach the engine's log.
    const GPU_STACK: [&str; 6] = ["burn", "cubecl", "cubek", "wgpu", "naga", "pliron"];
    if GPU_STACK.iter().any(|p| metadata.target().starts_with(p)) {
      return metadata.level() <= log::Level::Warn;
    }
    metadata.level() <= log::Level::Info
  }
  fn log(&self, record: &log::Record) {
    if !self.enabled(record.metadata()) {
      return;
    }
    let mut guard = self.state.lock().unwrap();
    if let Some(state) = guard.as_mut() {
      CliLogger::write_locked(state, &record.args().to_string());
    }
  }
  fn flush(&self) {}
}

/// Installs the log backend (once). It writes nothing until [`init_logger`].
pub fn install_logger() {
  if log::set_logger(&LOGGER).is_ok() {
    log::set_max_level(log::LevelFilter::Info);
  }
}

/// `MakeDir::make` (makedir.cpp, Unix): mkdir 0775, EEXIST is fine, any other failure is
/// StringError("Error creating directory: <path>").
fn make_dir(path: &str) -> Result<(), KataError> {
  use std::os::unix::fs::DirBuilderExt;
  match std::fs::DirBuilder::new().mode(0o775).create(path) {
    Ok(()) => Ok(()),
    Err(e) if e.raw_os_error() == Some(libc::EEXIST) => Ok(()),
    Err(_) => Err(KataError::StringError("Error creating directory: ".to_string() + path)),
  }
}

/// `Logger::addFile(file, false)` (logger.cpp).
fn add_file(state: &mut LoggerState, file: &str) {
  if file.is_empty() {
    return;
  }
  match std::fs::OpenOptions::new().create(true).append(true).open(file) {
    Ok(f) => state.files.push(f),
    Err(_) => {
      // FileUtils::open's IOError text
      let what = "Could not write to file ".to_string() + file + " - invalid path or permissions?";
      CliLogger::write_locked(state, &("WARNING: could not open file for logging: ".to_string() + &what));
      let _ = std::io::stderr().write_all(("WARNING: could not open file for logging: ".to_string() + &what + "\n").as_bytes());
    }
  }
}

/// AnalysisHost::init_logger: the outputs of the `Logger` constructor (logger.cpp) from the settings it
/// read: stdout / stderr / timestamps, then `logFile` (addFile) or `logDir` / `logDirDated` (MakeDir::make,
/// then addFile of `<dir>/[<date>/]<compact date>-<random hex>.log`). Installs the backend if needed.
pub fn init_logger(settings: &LoggerSettings) -> Result<(), KataError> {
  install_logger();
  let mut state = LoggerState { to_stdout: settings.log_to_stdout, to_stderr: settings.log_to_stderr, log_time: settings.log_time, files: Vec::new() };
  let compact = || local_time_string("%Y%m%d-%H%M%S");
  if let Some(f) = &settings.log_file {
    add_file(&mut state, f);
  } else if let Some(d) = &settings.log_dir {
    make_dir(d)?;
    let mut rand = Rand::new();
    let path = d.clone() + "/" + &compact() + "-" + &Global::uint32_to_hex_string(rand.next_uint()) + ".log";
    add_file(&mut state, &path);
  } else if let Some(d) = &settings.log_dir_dated {
    make_dir(d)?;
    let mut rand = Rand::new();
    make_dir(&(d.clone() + "/" + &compact()))?;
    let path = d.clone() + "/" + &compact() + "/" + &Global::uint32_to_hex_string(rand.next_uint()) + ".log";
    add_file(&mut state, &path);
  }
  *LOGGER.state.lock().unwrap() = Some(state);
  Ok(())
}

//------------------------------------------------------------------------------------------------
// The burn backend as Setup's NNEvaluatorHost

/// The networks of katago-nn-burn (compiled in or not: see the crate features).
pub use katago_nn_burn::NetId as Net;

fn resolve_net_by_name(model_file: &str) -> Option<Net> {
  let base = std::path::Path::new(model_file).file_name().and_then(|s| s.to_str()).unwrap_or(model_file);
  let stem = [".bin.gz", ".txt.gz", ".onnx", ".bpk", ".gz", ".bin"].iter().fold(base.to_string(), |s, ext| s.strip_suffix(ext).map(str::to_string).unwrap_or(s));
  let candidates: [(Net, &str); 4] = [
    (Net::B6c96, "g170-b6c96-s175395328-d26788732"),
    (Net::B10c128, "g170e-b10c128-s1141046784-d204142634"),
    (Net::B18c384nbt, "kata1-b18c384nbt"),
    (Net::B18c384nbtHumanv0, "b18c384nbt-humanv0"),
  ];
  for (net, full) in candidates {
    if stem == net.name() || stem == full || stem.starts_with(&(net.name().to_string() + "-")) && full.starts_with(&stem) {
      return Some(net);
    }
  }
  None
}

/// Maps a `-model` path to a compiled network (see the crate docs).
pub fn resolve_net(model_file: &str) -> Result<Net, KataError> {
  if let Some(net) = resolve_net_by_name(model_file) {
    return Ok(net);
  }
  if let Some(net) = std::fs::canonicalize(model_file).ok().and_then(|target| resolve_net_by_name(&target.to_string_lossy())) {
    return Ok(net);
  }
  if !std::path::Path::new(model_file).exists() {
    // upstream's text for a missing file (ModelDesc loading: FileUtils::open's IOError, wrapped)
    return Err(KataError::StringError(format!(
      "Error loading or parsing model file {model_file}: Could not open file {model_file} - does not exist or invalid permissions?"
    )));
  }
  Err(KataError::StringError(format!(
    "Error loading or parsing model file {model_file}: katago-rs has no runtime .bin.gz loader; -model must name a network compiled into katago-nn-burn (g170-b6c96, g170e-b10c128, kata1-b18c384nbt, b18c384nbt-humanv0)"
  )))
}

/// Setup's `NNEvaluatorHost` for the burn networks of katago-nn-burn.
pub struct BurnHost {
  pub device: Device,
  pub spawner: Rc<dyn Spawner>,
  /// Model files loaded so far (for tests and diagnostics).
  pub loaded: RefCell<Vec<String>>,
}

impl BurnHost {
  pub fn new(backend: &str, spawner: Rc<dyn Spawner>) -> Result<BurnHost, KataError> {
    let device = compiled_backends().contains(&backend).then(|| katago_nn_burn::device(backend)).flatten().ok_or_else(|| {
      KataError::StringError(format!(
        "unknown or not compiled-in backend '{backend}' (flex, wgpu with the wgpu feature, rocm with the rocm feature)"
      ))
    })?;
    Ok(BurnHost { device, spawner, loaded: RefCell::new(Vec::new()) })
  }
}

fn read_bpk(model_file: &str, compiled: &str) -> Result<Vec<u8>, KataError> {
  let path = if model_file.ends_with(".bpk") { model_file.to_string() } else { compiled.to_string() };
  std::fs::read(&path).map_err(|e| KataError::IOError(format!("Could not open file {path} - does not exist or invalid permissions? ({e})")))
}

impl NNEvaluatorHost for BurnHost {
  fn backend_prefix(&self) -> String {
    "burn".to_string()
  }
  fn global_initialize(&self) {}
  fn global_cleanup(&self) {}
  fn load_model(&self, _nn_model_name: &str, nn_model_file: &str) -> Result<(Rc<dyn NeuralNet>, Vec<u8>), KataError> {
    self.loaded.borrow_mut().push(nn_model_file.to_string());
    let net = if nn_model_file == "/dev/null" { Net::B6c96 } else { resolve_net(nn_model_file)? };
    let empty = nn_model_file == "/dev/null";
    match (net.neural_net(self.device.clone(), Vec::new()), net.bpk_path()) {
      (Some(neural_net), Some(compiled)) => Ok((neural_net, if empty { Vec::new() } else { read_bpk(nn_model_file, compiled)? })),
      _ => Err(KataError::StringError(format!("network {} is not compiled in (enable the katago-cli feature for it)", net.name()))),
    }
  }
  fn spawner(&self) -> Rc<dyn Spawner> {
    self.spawner.clone()
  }
}

//------------------------------------------------------------------------------------------------
// stdin / stdout

/// `getline(cin, line)` as a Stream of byte lines (without '\n'). A reader thread sends every complete
/// line of one read() as one batch, so a burst that the client wrote at once reaches the request loop
/// at once (it is "immediately available", as upstream's buffered getline sees it). A final line without
/// '\n' is returned too.
pub fn stdin_lines() -> impl futures::Stream<Item = Vec<u8>> + Unpin {
  use futures::StreamExt;
  let (tx, rx) = futures::channel::mpsc::unbounded::<Vec<Vec<u8>>>();
  std::thread::spawn(move || {
    use std::io::Read;
    let stdin = std::io::stdin();
    let mut lock = stdin.lock();
    let mut pending: Vec<u8> = Vec::new();
    let mut buf = vec![0u8; 1 << 16];
    loop {
      match lock.read(&mut buf) {
        Ok(0) => {
          if !pending.is_empty() {
            let _ = tx.unbounded_send(vec![std::mem::take(&mut pending)]);
          }
          break;
        }
        Ok(n) => {
          pending.extend_from_slice(&buf[..n]);
          let mut batch: Vec<Vec<u8>> = Vec::new();
          while let Some(p) = pending.iter().position(|&c| c == b'\n') {
            let rest = pending.split_off(p + 1);
            let mut line = std::mem::replace(&mut pending, rest);
            line.pop();
            batch.push(line);
          }
          if !batch.is_empty() && tx.unbounded_send(batch).is_err() {
            break;
          }
        }
        Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
        Err(_) => break,
      }
    }
  });
  rx.flat_map(futures::stream::iter)
}

/// `cout << message << endl`.
pub fn write_stdout_line(message: &str) {
  let stdout = std::io::stdout();
  let mut lock = stdout.lock();
  let _ = lock.write_all(message.as_bytes());
  let _ = lock.write_all(b"\n");
  let _ = lock.flush();
}

/// `cerr << text`.
pub fn write_stderr(text: &str) {
  let _ = std::io::stderr().write_all(text.as_bytes());
}

#[cfg(test)]
mod tests {
  use super::*;

  fn a(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| s.to_string()).collect()
  }

  fn num_os_threads() -> usize {
    std::fs::read_dir("/proc/self/task").map(|d| d.count()).unwrap_or(0)
  }

  /// Abandoned long waits (an analysis callback loop dropped when its search ends) leave no thread and
  /// no timer entry behind; short waits still fire.
  #[test]
  fn sleep_is_one_shared_timer_and_cancels_on_drop() {
    let short = sleep(0.01);
    let threads_before = num_os_threads();
    let pending_before = pending_sleeps();
    let abandoned: Vec<_> = (0..1000).map(|_| sleep(1e6)).collect();
    assert_eq!(pending_sleeps(), pending_before + 1000);
    assert!(num_os_threads() <= threads_before + 1, "{} -> {} threads", threads_before, num_os_threads());
    drop(abandoned);
    assert_eq!(pending_sleeps(), pending_before);
    let t0 = Instant::now();
    futures::executor::block_on(short);
    futures::executor::block_on(sleep(0.02));
    assert!(t0.elapsed().as_secs_f64() >= 0.015);
    assert_eq!(timer_threads_started(), 1);
  }

  #[test]
  fn tclap_int_extraction() {
    assert_eq!(extract_int("2"), Ok(2));
    assert_eq!(extract_int(" 2"), Ok(2));
    assert_eq!(extract_int("-1"), Ok(-1));
    assert_eq!(extract_int("+3"), Ok(3));
    assert_eq!(extract_int(""), Ok(0));
    assert!(extract_int("2 ").is_err());
    assert!(extract_int("2x").is_err());
    assert!(extract_int("99999999999").is_err());
    assert_eq!(extract_int("2 3"), Err("More than one valid value parsed from string '2 3'".to_string()));
  }

  #[test]
  fn tclap_forms() {
    assert!(matches!(parse_args(&a(&["-config", "c", "-model", "m"])), Ok(ParsedArgs::Run(_))));
    assert!(matches!(parse_args(&a(&["-config c", "-model=m"])), Err(e) if e.arg_id == "-model=m"));
    assert!(matches!(parse_args(&a(&["--config", "c"])), Err(e) if e.error == "Couldn't find match for argument"));
    assert!(matches!(parse_args(&a(&["-config", "c", "--", "-zz"])), Ok(ParsedArgs::Run(_))));
    assert!(matches!(parse_args(&a(&["-config", "c", "-"])), Ok(ParsedArgs::Run(_))));
    assert!(matches!(parse_args(&a(&["-model", "-config", "c"])), Err(e) if e.arg_id == "c"));
    assert!(matches!(parse_args(&a(&["-model", "x", "-model", "y"])), Err(e) if e.error == "Argument already set!" && e.arg_id == "(-model)"));
    assert!(parse_args(&a(&["-zz", "-h"])).is_err());
    assert!(matches!(parse_args(&a(&["-h", "-zz"])), Ok(ParsedArgs::Help)));
    assert!(matches!(parse_args(&a(&[])), Err(e) if e.arg_id_text() == " "));
  }
}
