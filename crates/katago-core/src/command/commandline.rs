// Ported from KataGo cpp/command/commandline.cpp, cpp/command/commandline.h @ fd0723fdbc0e9d82cf269c9630af8c27c57c07c4
//
// Only KataGoCommandLine::maybeApplyOverrideConfigArg, which both hosts need: the rest of KataGoCommandLine
// (TCLAP's argv parsing, getModelFile's default lookup, getConfig's file reading, logOverrides) is the host's
// (katago-cli's args / config modules, katago-wasm's config) or analysis.rs's (logOverrides).

use std::collections::BTreeMap;

use crate::core::config_parser::ConfigParser;
use crate::core::global::KataError;
use crate::program::setup::Setup;

/// upstream: KataGoCommandLine::maybeApplyOverrideConfigArg (commandline.cpp)
/// PORT NOTE: a free fn over the `-override-config` values (upstream: `overrideConfigArg->getValue()`, none when the
/// command has no such argument: an empty slice).
pub fn maybe_apply_override_config_arg(cfg: &mut ConfigParser, override_configs: &[String]) -> Result<(), KataError> {
  for override_config in override_configs {
    if !override_config.is_empty() {
      let newkvs: BTreeMap<String, String> = ConfigParser::parse_comma_separated(override_config)?;
      //HACK to avoid a common possible conflict - if we specify some of the rules options on one side, the other side should be erased.
      let mutex_key_sets = Setup::get_mutex_key_sets();
      cfg.override_keys_newkvs_mutex_key_sets(&newkvs, &mutex_key_sets);
    }
  }
  Ok(())
}
