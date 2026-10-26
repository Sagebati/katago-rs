//! The Logger's outputs (core/logger.cpp is replaced by the log crate; this is its backend).

use std::io::Write as _;
use std::sync::Mutex;

use katago_core::command::analysis::LoggerSettings;
use katago_core::core::global::Global;
use katago_core::core::global::KataError;
use katago_core::core::rand::Rand;

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
    let line = if state.log_time {
      local_time_string(TIME_FORMAT) + msg + "\n"
    } else {
      ": ".to_string() + msg + "\n"
    };
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
  let mut state = LoggerState {
    to_stdout: settings.log_to_stdout,
    to_stderr: settings.log_to_stderr,
    log_time: settings.log_time,
    files: Vec::new(),
  };
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
