//! The command line: a TCLAP 1.2.5 emulation of the analysis subcommand (KataGoCommandLine), and the default
//! model lookup.

use katago_core::core::global::KataError;

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
    ArgError {
      arg_id: arg_id.to_string(),
      error: error.to_string(),
    }
  }
  /// `ArgException::argId()`
  pub fn arg_id_text(&self) -> String {
    if self.arg_id == "undefined" {
      " ".to_string()
    } else {
      "Argument: ".to_string() + &self.arg_id
    }
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
  [
    ("flex", cfg!(feature = "flex")),
    ("wgpu", cfg!(feature = "wgpu")),
    ("rocm", cfg!(feature = "rocm")),
  ]
  .into_iter()
  .filter_map(|(name, on)| on.then_some(name))
  .collect()
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
  let mut out = CliArgs {
    backend: "flex".to_string(),
    ..CliArgs::default()
  };
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
  if let Some(dir) = std::fs::read_link("/proc/self/exe")
    .or_else(|_| std::env::current_exe())
    .ok()
    .as_deref()
    .and_then(std::path::Path::parent)
  {
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
  let path_for_err_msg = paths
    .first()
    .cloned()
    .unwrap_or_else(|| "(dir containing katago.exe, or else ~/.katago)/default_model.bin.gz".to_string());
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

#[cfg(test)]
mod tests {
  use super::*;

  fn a(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| s.to_string()).collect()
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
