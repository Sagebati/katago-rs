//! Host side of the native `katago-rs analysis` drop-in (plan §1): everything `MainCmds::analysis`
//! gets from the process in upstream KataGo, for the ported library function
//! `katago_core::command::analysis::analysis`:
//! - argv (TCLAP's `-model`, `-config`, `-override-config`, `-human-model`, `-analysis-threads`,
//!   `-quit-without-waiting`, plus `-backend flex|wgpu`) -> [`CliArgs`];
//! - `KataGoCommandLine::getConfig` (config files, `@include` through the file system, `-override-config`
//!   with `Setup::getMutexKeySets`) -> [`load_config`];
//! - the Logger's outputs (stderr, `logFile`, `logDir`, timestamps) -> [`install_logger`];
//! - the clock and the timed wait (`core::timer::set_clock`, `port::executor::set_sleep`) -> [`install_clock_and_sleep`];
//! - the NN backend: the burn networks compiled into katago-nn-burn -> [`BurnHost`].
//!
//! Model files: there is no runtime `.bin.gz` loader. `-model` names one of the networks compiled into
//! katago-nn-burn (`g170-b6c96`, `g170e-b10c128`, `kata1-b18c384nbt`), matched on the file name (with or
//! without directories and `.bin.gz` / `.txt.gz` / `.onnx` / `.bpk`, and the full internal name such as
//! `g170-b6c96-s175395328-d26788732`). The weights are the `.bpk` generated at build time
//! (`katago_nn_burn::<net>::BPK_PATH`) unless `-model` is itself a `.bpk` file, which is then read. Any other
//! network is an error. Not a port of upstream code (idiomatic Rust at the API edge, PORTING.md §1).

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
use katago_core::core::config_parser::ConfigFileReader;
use katago_core::core::config_parser::ConfigParser;
use katago_core::core::global::KataError;
use katago_core::neuralnet::nninterface::NeuralNet;
use katago_core::port::executor::LocalBoxFuture;
use katago_core::port::executor::Spawner;
use katago_core::program::setup::NNEvaluatorHost;
use katago_core::program::setup::Setup;

//------------------------------------------------------------------------------------------------
// Command line

/// The parsed command line of `katago-rs analysis`.
#[derive(Clone, Debug, Default)]
pub struct CliArgs {
  pub model_file: String,
  pub human_model_file: String,
  pub config_files: Vec<String>,
  pub override_configs: Vec<String>,
  pub num_analysis_threads: Option<i32>,
  pub quit_without_waiting: bool,
  /// `flex` (default) or `wgpu` (needs the `wgpu` feature).
  pub backend: String,
}

pub const USAGE: &str = "Usage: katago-rs analysis -model <net (.bin.gz name or .bpk)> -config <cfg> [-config <more.cfg>...] \
[-override-config \"key=value, ...\"] [-human-model <file>] [-analysis-threads N] [-quit-without-waiting] [-backend flex|wgpu]";

/// Parses the arguments after the subcommand. Accepts `-flag value`, `--flag value`, `-flag=value`.
pub fn parse_args(args: &[String]) -> Result<CliArgs, String> {
  let mut out = CliArgs { backend: "flex".to_string(), ..CliArgs::default() };
  let mut i = 0;
  while i < args.len() {
    let raw = args[i].as_str();
    let stripped = raw.strip_prefix("--").or_else(|| raw.strip_prefix('-')).ok_or_else(|| format!("unexpected argument '{raw}'"))?;
    let (name, inline_value) = match stripped.split_once('=') {
      Some((n, v)) => (n, Some(v.to_string())),
      None => (stripped, None),
    };
    let value = |i: &mut usize| -> Result<String, String> {
      if let Some(v) = &inline_value {
        return Ok(v.clone());
      }
      *i += 1;
      args.get(*i).cloned().ok_or_else(|| format!("missing value for -{name}"))
    };
    match name {
      "model" => out.model_file = value(&mut i)?,
      "human-model" => out.human_model_file = value(&mut i)?,
      "config" => out.config_files.push(value(&mut i)?),
      "override-config" => out.override_configs.push(value(&mut i)?),
      "analysis-threads" => {
        let v = value(&mut i)?;
        out.num_analysis_threads = Some(v.parse::<i32>().map_err(|_| format!("invalid value '{v}' for -analysis-threads"))?);
      }
      "quit-without-waiting" => out.quit_without_waiting = true,
      "backend" => out.backend = value(&mut i)?,
      _ => return Err(format!("unknown argument '{raw}'")),
    }
    i += 1;
  }
  if out.model_file.is_empty() {
    return Err("-model is required (there is no default model location)".to_string());
  }
  if out.config_files.is_empty() {
    return Err("-config is required (there is no default config location)".to_string());
  }
  Ok(out)
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

/// A timed wait for `port::executor::sleep`: a helper thread wakes the future (LocalPool parks its
/// thread while idle, and a wake from another thread unparks it).
fn sleep(seconds: f64) -> LocalBoxFuture<'static, ()> {
  struct State {
    done: bool,
    waker: Option<Waker>,
  }
  let state = Arc::new(Mutex::new(State { done: false, waker: None }));
  let thread_state = state.clone();
  let duration = std::time::Duration::from_secs_f64(if seconds.is_finite() && seconds > 0.0 { seconds } else { 0.0 });
  std::thread::spawn(move || {
    std::thread::sleep(duration);
    let mut s = thread_state.lock().unwrap();
    s.done = true;
    if let Some(w) = s.waker.take() {
      w.wake();
    }
  });
  Box::pin(futures::future::poll_fn(move |cx| {
    let mut s = state.lock().unwrap();
    if s.done {
      Poll::Ready(())
    } else {
      s.waker = Some(cx.waker().clone());
      Poll::Pending
    }
  }))
}

/// Installs the wall clock (`core::timer::set_clock`) and the timed wait (`port::executor::set_sleep`).
pub fn install_clock_and_sleep() {
  START.get_or_init(Instant::now);
  katago_core::core::timer::set_clock(clock);
  katago_core::port::executor::set_sleep(sleep);
}

//------------------------------------------------------------------------------------------------
// Logger (core::logger is replaced by the log crate; this is its backend)

struct CliLogger {
  to_stderr: bool,
  log_time: bool,
  file: Option<Mutex<std::fs::File>>,
}

fn timestamp() -> String {
  let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0) as i64;
  let days = secs.div_euclid(86400);
  let sod = secs.rem_euclid(86400);
  // civil-from-days (Howard Hinnant), UTC
  let z = days + 719468;
  let era = z.div_euclid(146097);
  let doe = z - era * 146097;
  let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
  let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
  let mp = (5 * doy + 2) / 153;
  let d = doy - (153 * mp + 2) / 5 + 1;
  let m = if mp < 10 { mp + 3 } else { mp - 9 };
  let y = yoe + era * 400 + if m <= 2 { 1 } else { 0 };
  format!("{:04}-{:02}-{:02} {:02}:{:02}:{:02}+0000", y, m, d, sod / 3600, (sod / 60) % 60, sod % 60)
}

impl log::Log for CliLogger {
  fn enabled(&self, metadata: &log::Metadata) -> bool {
    metadata.level() <= log::Level::Info
  }
  fn log(&self, record: &log::Record) {
    if !self.enabled(record.metadata()) {
      return;
    }
    let line = if self.log_time { format!("{}: {}\n", timestamp(), record.args()) } else { format!(": {}\n", record.args()) };
    if self.to_stderr {
      let _ = std::io::stderr().write_all(line.as_bytes());
    }
    if let Some(f) = &self.file {
      let _ = f.lock().unwrap().write_all(line.as_bytes());
    }
  }
  fn flush(&self) {}
}

/// Installs the log backend from the keys upstream's Logger reads (`logToStderr` default true for
/// analysis, `logTimeStamp` default true, `logFile` / `logDir` / `logDirDated`). Reads `cfg` without
/// marking keys used (analysis.rs reads them again, as upstream's Logger constructor does).
pub fn install_logger(cfg: &ConfigParser) {
  let contents = cfg.get_all_key_vals();
  let mut kv: BTreeMap<String, String> = BTreeMap::new();
  for line in contents.lines() {
    if let Some((k, v)) = line.split_once(" = ") {
      kv.insert(k.to_string(), v.to_string());
    }
  }
  let get_bool = |k: &str, d: bool| -> bool { kv.get(k).map(|v| matches!(v.to_lowercase().as_str(), "true" | "1" | "yes" | "t" | "y")).unwrap_or(d) };
  let to_stderr = get_bool("logToStderr", true);
  let log_time = get_bool("logTimeStamp", true);
  let mut path: Option<String> = None;
  if let Some(f) = kv.get("logFile") {
    if !f.is_empty() {
      path = Some(f.clone());
    }
  } else if let Some(d) = kv.get("logDir").or_else(|| kv.get("logDirDated")).filter(|d| !d.is_empty()) {
    let _ = std::fs::create_dir_all(d);
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
    path = Some(format!("{}/{}-{:08x}.log", d, timestamp().replace([' ', ':', '+'], "-"), (secs as u32)));
  }
  let file = path.and_then(|p| std::fs::OpenOptions::new().create(true).append(true).open(p).ok()).map(Mutex::new);
  let logger = CliLogger { to_stderr, log_time, file };
  if log::set_logger(Box::leak(Box::new(logger))).is_ok() {
    log::set_max_level(log::LevelFilter::Info);
  }
}

//------------------------------------------------------------------------------------------------
// The burn backend as Setup's NNEvaluatorHost

/// The networks compiled into katago-nn-burn.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Net {
  B6c96,
  B10c128,
  B18c384nbt,
}

impl Net {
  pub fn name(self) -> &'static str {
    match self {
      Net::B6c96 => "g170-b6c96",
      Net::B10c128 => "g170e-b10c128",
      Net::B18c384nbt => "kata1-b18c384nbt",
    }
  }
}

/// Maps a `-model` path to a compiled network (see the crate docs).
pub fn resolve_net(model_file: &str) -> Result<Net, KataError> {
  let base = std::path::Path::new(model_file).file_name().and_then(|s| s.to_str()).unwrap_or(model_file);
  let stem = [".bin.gz", ".txt.gz", ".onnx", ".bpk", ".gz", ".bin"].iter().fold(base.to_string(), |s, ext| s.strip_suffix(ext).map(str::to_string).unwrap_or(s));
  let candidates: [(Net, &str); 3] =
    [(Net::B6c96, "g170-b6c96-s175395328-d26788732"), (Net::B10c128, "g170e-b10c128-s1141046784-d204142634"), (Net::B18c384nbt, "kata1-b18c384nbt")];
  for (net, full) in candidates {
    if stem == net.name() || stem == full || stem.starts_with(&(net.name().to_string() + "-")) && full.starts_with(&stem) {
      return Ok(net);
    }
  }
  Err(KataError::StringError(format!(
    "Error loading or parsing model file {model_file}: katago-rs has no runtime .bin.gz loader; -model must name a network compiled into katago-nn-burn (g170-b6c96, g170e-b10c128, kata1-b18c384nbt)"
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
    let device: Device = match backend {
      #[cfg(feature = "flex")]
      "flex" => burn::backend::FlexDevice.into(),
      #[cfg(feature = "wgpu")]
      "wgpu" => burn::backend::WgpuDevice::default().into(),
      other => return Err(KataError::StringError(format!("unknown or not compiled-in backend '{other}' (flex, or wgpu with the wgpu feature)"))),
    };
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
    match net {
      #[cfg(feature = "net-b6c96")]
      Net::B6c96 => Ok((
        Rc::new(katago_nn_burn::backend::BurnNeuralNet::<katago_nn_burn::b6c96::Model>::new(self.device.clone())),
        if empty { Vec::new() } else { read_bpk(nn_model_file, katago_nn_burn::b6c96::BPK_PATH)? },
      )),
      #[cfg(feature = "net-b10c128")]
      Net::B10c128 => Ok((
        Rc::new(katago_nn_burn::backend::BurnNeuralNet::<katago_nn_burn::b10c128::Model>::new(self.device.clone())),
        read_bpk(nn_model_file, katago_nn_burn::b10c128::BPK_PATH)?,
      )),
      #[cfg(feature = "net-b18c384nbt")]
      Net::B18c384nbt => Ok((
        Rc::new(katago_nn_burn::backend::BurnNeuralNet::<katago_nn_burn::b18c384nbt::Model>::new(self.device.clone())),
        read_bpk(nn_model_file, katago_nn_burn::b18c384nbt::BPK_PATH)?,
      )),
      #[allow(unreachable_patterns)]
      other => Err(KataError::StringError(format!("network {} is not compiled in (enable the katago-cli feature for it)", other.name()))),
    }
  }
  fn spawner(&self) -> Rc<dyn Spawner> {
    self.spawner.clone()
  }
}

//------------------------------------------------------------------------------------------------
// stdin / stdout

/// `getline(cin, line)` as a Stream: a reader thread forwards each line (without '\n').
pub fn stdin_lines() -> futures::channel::mpsc::UnboundedReceiver<String> {
  let (tx, rx) = futures::channel::mpsc::unbounded::<String>();
  std::thread::spawn(move || {
    use std::io::BufRead;
    let stdin = std::io::stdin();
    let mut lock = stdin.lock();
    let mut buf: Vec<u8> = Vec::new();
    loop {
      buf.clear();
      match lock.read_until(b'\n', &mut buf) {
        Ok(0) => break,
        Ok(_) => {
          if buf.last() == Some(&b'\n') {
            buf.pop();
          }
          if tx.unbounded_send(String::from_utf8_lossy(&buf).into_owned()).is_err() {
            break;
          }
        }
        Err(_) => break,
      }
    }
  });
  rx
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
