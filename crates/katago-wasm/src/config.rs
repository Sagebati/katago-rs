//! The analysis config of the browser engine, built in memory (no files in the browser): upstream's
//! `cpp/configs/analysis_example.cfg` (compiled in), the browser defaults for the chosen backend, then the
//! caller's overrides, applied as `katago analysis -config <cfg> -override-config <...>` applies them
//! (`KataGoCommandLine::getConfig` + `maybeApplyOverrideConfigArg`: `Setup::getMutexKeySets`).
//! Not a port of upstream code.

use std::rc::Rc;

use katago_core::core::config_parser::ConfigParser;
use katago_core::core::global::KataError;

/// upstream `cpp/configs/analysis_example.cfg` (v1.16.5), the base config when the caller gives none.
pub const ANALYSIS_EXAMPLE_CFG: &str = include_str!("../../../upstream/KataGo/cpp/configs/analysis_example.cfg");

/// The NN backend the engine runs on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
  /// burn wgpu on WebGPU
  WebGpu,
  /// burn flex (wasm SIMD128, single-threaded)
  Cpu,
}

impl Backend {
  pub fn parse(s: &str) -> Option<Backend> {
    match s {
      "webgpu" => Some(Backend::WebGpu),
      "cpu" => Some(Backend::Cpu),
      _ => None,
    }
  }

  pub fn name(self) -> &'static str {
    match self {
      Backend::WebGpu => "webgpu",
      Backend::Cpu => "cpu",
    }
  }
}

/// Overrides of analysis_example.cfg for the browser, applied before the caller's.
/// - No log directory (`logDir=` removes the key; there are no files, log lines go to the host's log callback).
/// - NN cache 2^16 entries (upstream 2^23: ~3 KB per filled entry with ownership, so up to ~190 MB here).
/// - WebGPU: one forward pass per batch costs a ~20 ms dispatch + readback floor, so the search must batch:
///   16 search threads per analysis thread (upstream's value), 2 analysis threads, batches of up to 32.
/// - CPU: the wasm flex backend is single-threaded and gains nothing from batching (~11 evals/s at batch
///   1 or 8 for b10c128, docs/M0-report.md), so one search thread (no virtual-loss dilution) and one
///   analysis thread; a terminate then waits for at most one ~90 ms forward pass.
pub fn browser_defaults(backend: Backend) -> &'static str {
  match backend {
    Backend::WebGpu => {
      "logDir=, numAnalysisThreads=2, numSearchThreadsPerAnalysisThread=16, nnMaxBatchSize=32, nnCacheSizePowerOfTwo=16, nnMutexPoolSizePowerOfTwo=14"
    }
    Backend::Cpu => {
      "logDir=, numAnalysisThreads=1, numSearchThreadsPerAnalysisThread=1, nnMaxBatchSize=4, nnCacheSizePowerOfTwo=16, nnMutexPoolSizePowerOfTwo=12"
    }
  }
}

/// The config: `base` (analysis_example.cfg when None) then, only when `base` is None, the browser defaults,
/// then each of `overrides` ("key=value, key=value"). `@include` is an error (no files in the browser).
/// Returns the ConfigParser and the override strings applied (for AnalysisArgs::override_configs, which the
/// engine logs as `cmd.logOverrides` does).
pub fn build_config(base: Option<&str>, backend: Backend, overrides: &[String]) -> Result<(ConfigParser, Vec<String>), KataError> {
  let mut cfg = ConfigParser::new(false, true);
  cfg.set_include_reader(Rc::new(|path: &str| -> Result<String, KataError> {
    Err(KataError::IOError(format!(
      "Could not open file {path} - does not exist or invalid permissions? (no config files in the browser)"
    )))
  }));
  cfg.initialize_in(base.unwrap_or(ANALYSIS_EXAMPLE_CFG))?;
  let mut applied: Vec<String> = Vec::new();
  if base.is_none() {
    applied.push(browser_defaults(backend).to_string());
  }
  applied.extend(overrides.iter().filter(|s| !s.is_empty()).cloned());
  katago_core::command::commandline::maybe_apply_override_config_arg(&mut cfg, &applied)?;
  Ok((cfg, applied))
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn defaults_per_backend() {
    let (mut cfg, applied) = build_config(None, Backend::WebGpu, &[]).unwrap();
    assert_eq!(applied.len(), 1);
    assert_eq!(cfg.get_int("numSearchThreadsPerAnalysisThread", 1, 1024).unwrap(), 16);
    assert_eq!(cfg.get_int("nnMaxBatchSize", 1, 1024).unwrap(), 32);
    assert_eq!(cfg.get_int("nnCacheSizePowerOfTwo", 0, 48).unwrap(), 16);
    // an empty override value removes the key (ConfigParser::overrideKeys)
    assert!(!cfg.contains("logDir"));
    // analysis_example.cfg values that the browser keeps
    assert_eq!(cfg.get_int("maxVisits", 1, 1 << 30).unwrap(), 500);
    assert_eq!(cfg.get_string("reportAnalysisWinratesAs").unwrap(), "BLACK");

    let (mut cfg, _) = build_config(None, Backend::Cpu, &[]).unwrap();
    assert_eq!(cfg.get_int("numSearchThreadsPerAnalysisThread", 1, 1024).unwrap(), 1);
    assert_eq!(cfg.get_int("numAnalysisThreads", 1, 1024).unwrap(), 1);
  }

  #[test]
  fn caller_overrides_win() {
    let (mut cfg, applied) = build_config(None, Backend::Cpu, &["maxVisits=100, numSearchThreadsPerAnalysisThread=4".to_string()]).unwrap();
    assert_eq!(applied.len(), 2);
    assert_eq!(cfg.get_int("maxVisits", 1, 1 << 30).unwrap(), 100);
    assert_eq!(cfg.get_int("numSearchThreadsPerAnalysisThread", 1, 1024).unwrap(), 4);
  }

  #[test]
  fn explicit_base_skips_browser_defaults() {
    let (mut cfg, applied) = build_config(Some("maxVisits = 7\nlogDir = x\n"), Backend::WebGpu, &["logDir=".to_string()]).unwrap();
    assert_eq!(applied, vec!["logDir=".to_string()]);
    assert_eq!(cfg.get_int("maxVisits", 1, 100).unwrap(), 7);
    assert!(!cfg.contains("nnMaxBatchSize"));
    assert!(!cfg.contains("logDir"));
  }

  #[test]
  fn include_is_an_error() {
    let e = build_config(Some("@include other.cfg\n"), Backend::Cpu, &[]).err().expect("no files");
    assert!(e.what().contains("other.cfg"), "{}", e.what());
  }

  #[test]
  fn bad_override_is_an_error() {
    assert!(build_config(None, Backend::Cpu, &["maxVisits".to_string()]).is_err());
  }
}
