//! `KataGoCommandLine::getConfig`: config files, `@include` through the file system, `-override-config`.

use std::collections::BTreeMap;
use std::rc::Rc;

use katago_core::core::config_parser::ConfigFileReader;
use katago_core::core::config_parser::ConfigParser;
use katago_core::core::global::KataError;
use katago_core::program::setup::Setup;


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
