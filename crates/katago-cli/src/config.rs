//! `KataGoCommandLine::getConfig`: config files, `@include` through the file system, `-override-config`.

use std::rc::Rc;

use katago_core::core::config_parser::ConfigFileReader;
use katago_core::core::config_parser::ConfigParser;
use katago_core::core::global::KataError;

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
  katago_core::command::commandline::maybe_apply_override_config_arg(&mut cfg, override_configs)?;
  Ok(cfg)
}
