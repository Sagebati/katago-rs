// Ported from KataGo cpp/core/config_parser.cpp, cpp/core/config_parser.h @ fd0723fdbc0e9d82cf269c9630af8c27c57c07c4
//
// Divergences (port-manifest.toml notes, PORTING.md §13 "IO-skipped"):
// - core does no file IO. Every place where upstream opens an `ifstream` (`FileUtils::open` in
//   `initialize(fname)` and `processIncludedFile`) calls the host-injected reader instead
//   (`set_include_reader`, a `Fn(path) -> Result<contents, KataError>`); `std::istream&` parameters
//   are the whole stream contents as `&str`.
// - Not ported (no Rust fn, they only forward to `initialize(fname)` on a fresh object, where there is
//   no moment to inject the reader): `ConfigParser(const std::string& file, bool, bool)` and
//   `ConfigParser(const char* file, bool, bool)`. Use `new` + `set_include_reader` + `initialize`.
// - `usedKeysMutex` removed (single-threaded): `usedKeys` is a plain set.
// - `Logger*` removed: `warnUnusedKeys` logs through `tracing`.

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::rc::Rc;

use crate::core::commontypes::enabled_t;
use crate::core::global::Global;
use crate::core::global::KataError;
use crate::testAssert;

/* Parses simple configs like:

#This is a comment
foo = true
bar = 64
baz = yay

*/

/// PORT NOTE: the injected replacement for opening a file (see the header of this file). It receives
/// the path exactly as upstream would pass it to `FileUtils::open` and returns the file's contents,
/// or the error to throw (upstream: `IOError("Could not open file ... - does not exist or invalid
/// permissions?")`).
pub type ConfigFileReader = Rc<dyn Fn(&str) -> Result<String, KataError>>;

/// upstream: ConfigParser (config_parser.h)
/// PORT NOTE: copy assignment and the move operations are deleted upstream, so no Clone; the copy
/// constructor throws on an uninitialized source, so it is `new_copy -> Result`.
pub struct ConfigParser {
  pub(crate) initialized: bool,
  pub(crate) file_name: String,
  pub(crate) contents: String,
  pub(crate) key_values: BTreeMap<String, String>,

  // If true, overriding keys within the same file is possible
  pub(crate) keys_override_enabled: bool,
  // If true (default), overriding keys from included files is possible
  pub(crate) keys_override_from_includes: bool,

  // Current reading state variables
  // Current filename being processed (can differ from fileName in case of using @include directive)
  pub(crate) cur_line_num: i32,
  pub(crate) cur_filename: String,
  pub(crate) included_files: Vec<String>,

  // Internal stack for tracking the file path as we process recursive includes.
  pub(crate) base_dirs: Vec<String>,

  // Currently unused. Messages tracking what overrides occurred.
  pub(crate) log_messages: Vec<String>,

  // PORT NOTE: `mutable std::mutex usedKeysMutex` removed (single-threaded).
  pub(crate) used_keys: BTreeSet<String>,

  // PORT NOTE: added field, see the header of this file.
  pub(crate) include_reader: Option<ConfigFileReader>,
}

impl ConfigParser {
  /// upstream: ConfigParser::ConfigParser (config_parser.cpp)
  /// PORT NOTE: overloads (PORTING.md §1 rule 3): `ConfigParser(bool keysOverride = false, bool
  /// keysOverrideFromIncludes = true)` is callable with no arguments, so it has the fewest
  /// parameters and keeps `new`; the others append their parameter names. Defaults are passed explicitly.
  pub fn new(keys_override: bool, keys_override_from_includes_: bool) -> ConfigParser {
    ConfigParser {
      initialized: false,
      file_name: String::new(),
      contents: String::new(),
      key_values: BTreeMap::new(),
      keys_override_enabled: keys_override,
      keys_override_from_includes: keys_override_from_includes_,
      cur_line_num: 0,
      cur_filename: String::new(),
      included_files: Vec::new(),
      base_dirs: Vec::new(),
      log_messages: Vec::new(),
      used_keys: BTreeSet::new(),
      include_reader: None,
    }
  }

  // upstream: ConfigParser::ConfigParser(const std::string& file, bool, bool) (config_parser.cpp)
  // upstream: ConfigParser::ConfigParser(const char* file, bool, bool) (config_parser.cpp)
  // PORT NOTE: file-reading constructors, no Rust fn (see the header of this file).

  /// upstream: ConfigParser::ConfigParser (config_parser.cpp)
  /// PORT NOTE: `std::istream& in` is the stream's whole contents as `&str`. The constructor throws,
  /// so it returns `Result`. @include in `in` needs a reader: use `new` + `set_include_reader` +
  /// `initialize_in` instead.
  pub fn new_in(in_: &str, keys_override: bool, keys_override_from_includes_: bool) -> Result<ConfigParser, KataError> {
    let mut s = ConfigParser::new(keys_override, keys_override_from_includes_);
    s.initialize_in(in_)?;
    Ok(s)
  }

  /// upstream: ConfigParser::ConfigParser (config_parser.cpp)
  pub fn new_kvs(kvs: &BTreeMap<String, String>) -> Result<ConfigParser, KataError> {
    let mut s = ConfigParser::new(false, true);
    s.initialize_kvs(kvs)?;
    Ok(s)
  }

  /// upstream: ConfigParser::ConfigParser (config_parser.cpp)
  /// PORT NOTE: copy constructor; throws for an uninitialized source, so `new_copy -> Result` rather
  /// than `Clone`. The usedKeys lock is removed. The injected include reader is shared with the copy.
  pub fn new_copy(source: &ConfigParser) -> Result<ConfigParser, KataError> {
    if !source.initialized {
      return Err(KataError::StringError(String::from("Can only copy a ConfigParser which has been initialized.")));
    }
    let mut s = ConfigParser::new(false, true);
    s.initialized = source.initialized;
    s.file_name = source.file_name.clone();
    s.base_dirs = source.base_dirs.clone();
    s.contents = source.contents.clone();
    s.key_values = source.key_values.clone();
    s.keys_override_enabled = source.keys_override_enabled;
    s.keys_override_from_includes = source.keys_override_from_includes;
    s.used_keys = source.used_keys.clone();
    s.include_reader = source.include_reader.clone();
    Ok(s)
  }

  /// PORT NOTE: added (no upstream counterpart): installs the reader that stands in for every
  /// `FileUtils::open` of upstream. Call before `initialize` / `override_keys` with a file name, or
  /// before `initialize_in` when the text contains @include directives.
  pub fn set_include_reader(&mut self, reader: ConfigFileReader) {
    self.include_reader = Some(reader);
  }

  /// PORT NOTE: `FileUtils::open(in, path)` + reading the stream, through the injected reader.
  /// The error without a reader is `FileUtils::open`'s message.
  fn open_file(&self, path: &str) -> Result<String, KataError> {
    match &self.include_reader {
      Some(reader) => reader(path),
      None => Err(KataError::IOError(format!(
        "Could not open file {} - does not exist or invalid permissions?",
        path
      ))),
    }
  }

  /// upstream: ConfigParser::initialize (config_parser.cpp)
  /// PORT NOTE: overloads: `initialize(const std::string& file)` is declared first and keeps the base
  /// name; the file is read through the injected reader (see the header of this file).
  pub fn initialize(&mut self, fname: &str) -> Result<(), KataError> {
    if self.initialized {
      return Err(KataError::StringError(String::from("ConfigParser already initialized, cannot initialize again")));
    }
    let in_ = self.open_file(fname)?;
    self.file_name = fname.to_string();
    let base_dir = self.extract_base_dir(fname);
    if !base_dir.is_empty() {
      self.base_dirs.push(base_dir);
    }
    self.initialize_internal(&in_)?;
    self.initialized = true;
    Ok(())
  }

  /// upstream: ConfigParser::initialize (config_parser.cpp)
  /// PORT NOTE: `std::istream& in` is the stream's whole contents as `&str`.
  pub fn initialize_in(&mut self, in_: &str) -> Result<(), KataError> {
    if self.initialized {
      return Err(KataError::StringError(String::from("ConfigParser already initialized, cannot initialize again")));
    }
    self.initialize_internal(in_)?;
    self.initialized = true;
    Ok(())
  }

  /// upstream: ConfigParser::initialize (config_parser.cpp)
  pub fn initialize_kvs(&mut self, kvs: &BTreeMap<String, String>) -> Result<(), KataError> {
    if self.initialized {
      return Err(KataError::StringError(String::from("ConfigParser already initialized, cannot initialize again")));
    }
    self.key_values = kvs.clone();
    self.initialized = true;
    Ok(())
  }

  /// upstream: ConfigParser::initializeInternal (config_parser.cpp)
  pub(crate) fn initialize_internal(&mut self, in_: &str) -> Result<(), KataError> {
    self.key_values.clear();
    self.contents.clear();
    self.cur_filename = self.file_name.clone();
    self.read_stream_content(in_)
  }

  /// upstream: ConfigParser::processIncludedFile (config_parser.cpp)
  pub(crate) fn process_included_file(&mut self, fname: &str) -> Result<(), KataError> {
    if fname == self.file_name || self.included_files.iter().any(|f| f == fname) {
      return Err(KataError::ConfigParsingError(format!(
        "Circular or multiple inclusion of the same file: '{}'{}",
        fname,
        self.line_and_file_info()
      )));
    }
    self.included_files.push(fname.to_string());
    self.cur_filename = fname.to_string();

    let mut fpath = String::new();
    for p in self.base_dirs.iter() {
      fpath += p;
    }
    fpath += fname;

    let base_dir = self.extract_base_dir(fname);
    if !base_dir.is_empty() {
      if base_dir.as_bytes()[0] == b'\\' || base_dir.as_bytes()[0] == b'/' {
        return Err(KataError::ConfigParsingError(String::from("Absolute paths in the included files are not supported yet")));
      }
      self.base_dirs.push(base_dir.clone());
    }

    let in_ = self.open_file(&fpath)?;
    self.read_stream_content(&in_)?;

    if !base_dir.is_empty() {
      self.base_dirs.pop();
    }
    Ok(())
  }

  /// upstream: ConfigParser::parseKeyValue (config_parser.cpp)
  pub(crate) fn parse_key_value(&self, trimmed_line: &str, key: &mut String, value: &mut String) -> Result<bool, KataError> {
    // PORT NOTE: std::string is bytes; `trimmedLine[i]` is `tl[i]`. Every cut is at an ASCII byte, and
    // `value` is accumulated as bytes, so multi-byte characters stay whole.
    let tl: &[u8] = trimmed_line.as_bytes();
    // Parse trimmed line, taking into account comments and quoting.
    key.clear();
    value.clear();
    let mut value_buf: Vec<u8> = Vec::new();

    // Parse key
    let mut found_any_key = false;
    let mut i: usize = 0;
    while i < tl.len() {
      let c = tl[i];
      if Global::is_alpha(c) || Global::is_digit(c) || c == b'_' || c == b'-' {
        key.push(c as char);
        found_any_key = true;
        i += 1;
        continue;
      } else if c == b'#' {
        if found_any_key {
          return Err(KataError::ConfigParsingError(String::from("Could not parse key value pair") + &self.line_and_file_info()));
        }
        return Ok(false);
      } else if Global::is_whitespace(c) || c == b'=' {
        break;
      } else {
        return Err(KataError::ConfigParsingError(String::from("Could not parse key value pair") + &self.line_and_file_info()));
      }
    }
    // Skip whitespace after key
    while i < tl.len() {
      let c = tl[i];
      if Global::is_whitespace(c) {
        i += 1;
        continue;
      } else if c == b'#' {
        if found_any_key {
          return Err(KataError::ConfigParsingError(String::from("Could not parse key value pair") + &self.line_and_file_info()));
        }
        return Ok(false);
      } else if c == b'=' {
        break;
      } else {
        return Err(KataError::ConfigParsingError(String::from("Could not parse key value pair") + &self.line_and_file_info()));
      }
    }
    // Skip equals sign
    let mut found_equals = false;
    if i < tl.len() {
      testAssert!(tl[i] == b'=');
      found_equals = true;
      i += 1;
    }
    // Skip whitespace after equals sign
    while i < tl.len() {
      let c = tl[i];
      if Global::is_whitespace(c) {
        i += 1;
        continue;
      } else if c == b'#' {
        if found_any_key || found_equals {
          return Err(KataError::ConfigParsingError(String::from("Could not parse key value pair") + &self.line_and_file_info()));
        }
        return Ok(false);
      } else {
        break;
      }
    }

    // Maybe parse double quotes
    let mut is_double_quotes = false;
    if i < tl.len() && tl[i] == b'"' {
      is_double_quotes = true;
      i += 1;
    }

    // Parse value
    let mut found_any_value = false;
    while i < tl.len() {
      let c = tl[i];
      if is_double_quotes {
        if c == b'\\' {
          if i + 1 >= tl.len() {
            return Err(KataError::ConfigParsingError(String::from("Could not parse key value pair") + &self.line_and_file_info()));
          }
          i += 1;
          value_buf.push(tl[i]);
          found_any_value = true;
          i += 1;
          continue;
        } else if c == b'"' {
          break;
        } else {
          value_buf.push(c);
          found_any_value = true;
          i += 1;
          continue;
        }
      } else {
        if c == b'#' {
          break;
        } else {
          value_buf.push(c);
          found_any_value = true;
          i += 1;
          continue;
        }
      }
    }
    // TODO(port): std::string is bytes; `value` is cut only at ASCII bytes of valid UTF-8 input, so
    // this conversion never replaces anything (an escaped lone lead byte `\é` keeps its continuation
    // bytes from the next iterations).
    *value = String::from_utf8_lossy(&value_buf).into_owned();

    if is_double_quotes {
      // Consume the trailing double quote
      if i < tl.len() && tl[i] == b'"' {
        i += 1;
      } else {
        return Err(KataError::ConfigParsingError(String::from("Could not parse key value pair") + &self.line_and_file_info()));
      }
      // The rest of the line can only be whitespace followed by a comment
      let remainder: String = Global::trim(&trimmed_line[i..], Global::TRIM_DEFAULT_DELIM);
      if !remainder.is_empty() && remainder.as_bytes()[0] != b'#' {
        return Err(KataError::ConfigParsingError(String::from("Could not parse key value pair") + &self.line_and_file_info()));
      }
    } else {
      // We stopped at a pound sign, the remainder is just comment or nothing
      // Trim whitespace off of unquoted values
      *value = Global::trim(value, Global::TRIM_DEFAULT_DELIM);
    }

    if is_double_quotes && !(found_any_key && found_any_value) {
      return Err(KataError::ConfigParsingError(String::from("Could not parse key value pair") + &self.line_and_file_info()));
    }
    if found_equals && !(found_any_key && found_any_value) {
      return Err(KataError::ConfigParsingError(String::from("Could not parse key value pair") + &self.line_and_file_info()));
    }
    if found_any_key != found_any_value {
      return Err(KataError::ConfigParsingError(String::from("Could not parse key value pair") + &self.line_and_file_info()));
    }

    Ok(found_any_key)
  }

  /// upstream: ConfigParser::readStreamContent (config_parser.cpp)
  /// PORT NOTE: `std::istream& in` is the stream's contents as `&str`; `while(getline(in,line))` is
  /// `Global::split_delim(in, '\n')`, which has getline's behaviour (no final empty token).
  pub(crate) fn read_stream_content(&mut self, in_: &str) -> Result<(), KataError> {
    self.cur_line_num = 0;
    let lines: Vec<String> = Global::split_delim(in_, '\n');
    let mut content_stream = String::new();
    let mut cur_file_keys: BTreeSet<String> = BTreeSet::new();
    for line in lines {
      let mut line: String = line;
      content_stream += &line;
      content_stream += "\n";
      self.cur_line_num += 1;
      line = Global::trim(&line, Global::TRIM_DEFAULT_DELIM);
      if line.is_empty() || line.as_bytes()[0] == b'#' {
        continue;
      }

      if line.as_bytes()[0] == b'@' {
        let comment_pos: Option<usize> = line.find('#');
        if let Some(comment_pos) = comment_pos {
          line = line[..comment_pos].to_string();
        }

        if line.len() < 9 {
          return Err(KataError::ConfigParsingError(String::from("Unsupported @ directive") + &self.line_and_file_info()));
        }
        // find_first_of(" \t\v\f=")
        let pos0: Option<usize> = line.as_bytes().iter().position(|c| b" \t\x0b\x0c=".contains(c));
        let pos0: usize = match pos0 {
          None => {
            return Err(KataError::ConfigParsingError(
              String::from("@ directive without value (key-val separator is not found)") + &self.line_and_file_info(),
            ));
          }
          Some(p) => p,
        };

        let key: String = Global::trim(&line[..pos0], Global::TRIM_DEFAULT_DELIM);
        if key != "@include" {
          return Err(KataError::ConfigParsingError(format!("Unsupported @ directive '{}'{}", key, self.line_and_file_info())));
        }

        let mut value: String = line[pos0 + 1..].to_string();
        // find_first_not_of(" \t\v\f=")
        let pos1: Option<usize> = value.as_bytes().iter().position(|c| !b" \t\x0b\x0c=".contains(c));
        let pos1: usize = match pos1 {
          None => {
            return Err(KataError::ConfigParsingError(
              String::from("@ directive without value (value after key-val separator is not found)") + &self.line_and_file_info(),
            ));
          }
          Some(p) => p,
        };

        value = Global::trim(&value[pos1..], Global::TRIM_DEFAULT_DELIM);
        value = Global::trim(&value, "'"); // remove single quotes for filename
        value = Global::trim(&value, "\""); // remove double quotes for filename

        let line_num: i32 = self.cur_line_num;
        self.process_included_file(&value)?;
        self.cur_line_num = line_num;
        continue;
      }

      let mut key = String::new();
      let mut value = String::new();
      let found_key_value: bool = self.parse_key_value(&line, &mut key, &mut value)?;
      if !found_key_value {
        continue;
      }

      if cur_file_keys.contains(&key) {
        if !self.keys_override_enabled {
          return Err(KataError::ConfigParsingError(format!(
            "Key '{}' + was specified multiple times in {}, you probably didn't mean to do this, please delete one of them",
            key, self.cur_filename
          )));
        } else {
          self.log_messages
            .push(format!("Key '{}' + was overriden by new value '{}'{}", key, value, self.line_and_file_info()));
        }
      }
      if self.key_values.contains_key(&key) {
        if !self.keys_override_from_includes {
          return Err(KataError::ConfigParsingError(format!(
            "Key '{}' + was specified multiple times in {} or its included files, and key overriding is disabled",
            key, self.cur_filename
          )));
        } else {
          self.log_messages
            .push(format!("Key '{}' + was overriden by new value '{}'{}", key, value, self.line_and_file_info()));
        }
      }
      self.key_values.insert(key.clone(), value);
      cur_file_keys.insert(key);
    }
    self.contents += &content_stream;
    Ok(())
  }

  /// upstream: ConfigParser::lineAndFileInfo (config_parser.cpp)
  pub(crate) fn line_and_file_info(&self) -> String {
    String::from(", line ") + &Global::int_to_string(self.cur_line_num) + " in '" + &self.cur_filename + "'"
  }

  /// upstream: ConfigParser::extractBaseDir (config_parser.cpp)
  pub(crate) fn extract_base_dir(&mut self, fname: &str) -> String {
    // find_last_of("/\\")
    let slash: Option<usize> = fname.as_bytes().iter().rposition(|c| *c == b'/' || *c == b'\\');
    match slash {
      Some(slash) => fname[..slash + 1].to_string(),
      None => String::new(),
    }
  }

  // upstream: ConfigParser::~ConfigParser (config_parser.cpp)
  // PORT NOTE: empty destructor, omitted.

  /// upstream: ConfigParser::getFileName (config_parser.cpp)
  pub fn get_file_name(&self) -> String {
    self.file_name.clone()
  }

  /// upstream: ConfigParser::getContents (config_parser.cpp)
  pub fn get_contents(&self) -> String {
    self.contents.clone()
  }

  /// upstream: ConfigParser::getAllKeyVals (config_parser.cpp)
  pub fn get_all_key_vals(&self) -> String {
    let mut ost = String::new();
    for (k, v) in self.key_values.iter() {
      ost += &(k.clone() + " = " + v);
      ost += "\n";
    }
    ost
  }

  /// upstream: ConfigParser::unsetUsedKey (config_parser.cpp)
  pub fn unset_used_key(&mut self, key: &str) {
    self.used_keys.remove(key);
  }

  /// upstream: ConfigParser::applyAlias (config_parser.cpp)
  pub fn apply_alias(&mut self, map_this_key: &str, to_this_key: &str) -> Result<(), KataError> {
    if self.contains(map_this_key) && self.contains(to_this_key) {
      return Err(KataError::IOError(format!("Cannot specify both {} and {} in the same config", map_this_key, to_this_key)));
    }
    if self.contains(map_this_key) {
      // `keyValues[toThisKey] = keyValues[mapThisKey]`: mapThisKey exists (checked above)
      let v: String = self.key_values.entry(map_this_key.to_string()).or_default().clone();
      self.key_values.insert(to_this_key.to_string(), v);
      self.key_values.remove(map_this_key);
      if self.used_keys.contains(map_this_key) {
        self.used_keys.insert(to_this_key.to_string());
        self.used_keys.remove(map_this_key);
      }
    }
    Ok(())
  }

  /// upstream: ConfigParser::overrideKey (config_parser.cpp)
  pub fn override_key(&mut self, key: &str, value: &str) {
    // Assume zero-length values mean to delete a key
    if value.is_empty() {
      if self.key_values.contains_key(key) {
        self.key_values.remove(key);
      }
    } else {
      self.key_values.insert(key.to_string(), value.to_string());
    }
  }

  /// upstream: ConfigParser::overrideKeys (config_parser.cpp)
  /// PORT NOTE: overloads: `overrideKeys(const std::string& fname)` is declared first (tie on parameter
  /// count) and keeps the base name; the file is read through the injected reader.
  pub fn override_keys(&mut self, fname: &str) -> Result<(), KataError> {
    // It's a new config file, so baseDir is not relevant anymore
    self.base_dirs.clear();
    self.process_included_file(fname)
  }

  /// upstream: ConfigParser::overrideKeys (config_parser.cpp)
  pub fn override_keys_newkvs(&mut self, newkvs: &BTreeMap<String, String>) {
    for (first, second) in newkvs.iter() {
      // Assume zero-length values mean to delete a key
      if second.is_empty() {
        if self.key_values.contains_key(first) {
          self.key_values.remove(first);
        }
      } else {
        self.key_values.insert(first.clone(), second.clone());
      }
    }
    self.file_name += " and/or command-line and query overrides";
  }

  //mutexKeySets: For each pair of sets (A,B), if newkvs contains anything in A, erase every existing key that overlaps with B, and vice versa.
  /// upstream: ConfigParser::overrideKeys (config_parser.cpp)
  pub fn override_keys_newkvs_mutex_key_sets(
    &mut self,
    newkvs: &BTreeMap<String, String>,
    mutex_key_sets: &[(BTreeSet<String>, BTreeSet<String>)],
  ) {
    for i in 0..mutex_key_sets.len() {
      let a: &BTreeSet<String> = &mutex_key_sets[i].0;
      let b: &BTreeSet<String> = &mutex_key_sets[i].1;
      let mut has_a = false;
      for iter in a.iter() {
        if newkvs.contains_key(iter) {
          has_a = true;
          break;
        }
      }
      let mut has_b = false;
      for iter in b.iter() {
        if newkvs.contains_key(iter) {
          has_b = true;
          break;
        }
      }
      if has_a {
        for iter in b.iter() {
          self.key_values.remove(iter);
        }
      }
      if has_b {
        for iter in a.iter() {
          self.key_values.remove(iter);
        }
      }
    }

    self.override_keys_newkvs(newkvs);
  }

  /// upstream: ConfigParser::parseCommaSeparated (config_parser.cpp)
  pub fn parse_comma_separated(comma_separated_values: &str) -> Result<BTreeMap<String, String>, KataError> {
    let mut key_values: BTreeMap<String, String> = BTreeMap::new();
    let pieces: Vec<String> = Global::split_delim(comma_separated_values, ',');
    for i in 0..pieces.len() {
      let s: String = Global::trim(&pieces[i], Global::TRIM_DEFAULT_DELIM);
      if s.is_empty() {
        continue;
      }
      let pos: Option<usize> = s.find('=');
      let pos: usize = match pos {
        None => {
          return Err(KataError::ConfigParsingError(String::from("Could not parse kv pair, could not find '=' in:") + &s));
        }
        Some(p) => p,
      };

      let key: String = Global::trim(&s[..pos], Global::TRIM_DEFAULT_DELIM);
      let value: String = Global::trim(&s[pos + 1..], Global::TRIM_DEFAULT_DELIM);
      key_values.insert(key, value);
    }
    Ok(key_values)
  }

  /// upstream: ConfigParser::markKeyUsed (config_parser.cpp)
  pub fn mark_key_used(&mut self, key: &str) {
    self.used_keys.insert(key.to_string());
  }

  /// upstream: ConfigParser::markAllKeysUsedWithPrefix (config_parser.cpp)
  pub fn mark_all_keys_used_with_prefix(&mut self, prefix: &str) {
    for key in self.key_values.keys() {
      if Global::is_prefix(key, prefix) {
        self.used_keys.insert(key.clone());
      }
    }
  }

  /// upstream: ConfigParser::warnUnusedKeys (config_parser.cpp)
  /// PORT NOTE: `Logger* logger` removed (PORTING.md §3): `if(logger) logger->write(m)` is `tracing::info!`.
  pub fn warn_unused_keys(&self, out: &mut dyn std::fmt::Write) {
    let unused: Vec<String> = self.unused_keys();
    let mut messages: Vec<String> = Vec::new();
    if !unused.is_empty() {
      messages.push(String::from("--------------"));
      messages.push(
        String::from("WARNING: Config had unused keys! You may have a typo, an option you specified is being unused from ")
          + &self.file_name,
      );
    }
    for i in 0..unused.len() {
      messages.push(String::from("WARNING: Unused key '") + &unused[i] + "' in " + &self.file_name);
    }
    if !unused.is_empty() {
      messages.push(String::from("--------------"));
    }

    for i in 0..messages.len() {
      tracing::info!("{}", messages[i]);
    }
    for i in 0..messages.len() {
      writeln!(out, "{}", messages[i]).unwrap();
    }
  }

  /// upstream: ConfigParser::unusedKeys (config_parser.cpp)
  pub fn unused_keys(&self) -> Vec<String> {
    let mut unused: Vec<String> = Vec::new();
    for key in self.key_values.keys() {
      if !self.used_keys.contains(key) {
        unused.push(key.clone());
      }
    }
    unused
  }

  /// upstream: ConfigParser::contains (config_parser.cpp)
  pub fn contains(&self, key: &str) -> bool {
    self.key_values.contains_key(key)
  }

  /// upstream: ConfigParser::containsAny (config_parser.cpp)
  pub fn contains_any(&self, possible_keys: &[String]) -> bool {
    for key in possible_keys.iter() {
      if self.contains(key) {
        return true;
      }
    }
    false
  }

  /// upstream: ConfigParser::containsAnyKeyContaining (config_parser.cpp)
  pub fn contains_any_key_containing(&self, substring: &str) -> bool {
    for (k, _v) in self.key_values.iter() {
      if k.contains(substring) {
        return true;
      }
    }
    false
  }

  /// upstream: ConfigParser::firstFoundOrFail (config_parser.cpp)
  pub fn first_found_or_fail(&self, possible_keys: &[String]) -> Result<String, KataError> {
    for key in possible_keys.iter() {
      if self.contains(key) {
        return Ok(key.clone());
      }
    }
    let mut message = String::from("Could not find key");
    for key in possible_keys.iter() {
      message += &(String::from(" '") + key + "'");
    }
    Err(KataError::IOError(message + " in config file " + &self.file_name))
  }

  /// upstream: ConfigParser::firstFoundOrEmpty (config_parser.cpp)
  pub fn first_found_or_empty(&self, possible_keys: &[String]) -> String {
    for key in possible_keys.iter() {
      if self.contains(key) {
        return key.clone();
      }
    }
    String::new()
  }

  /// upstream: ConfigParser::getString (config_parser.cpp)
  pub fn get_string(&mut self, key: &str) -> Result<String, KataError> {
    let value: String = match self.key_values.get(key) {
      None => {
        return Err(KataError::IOError(format!("Could not find key '{}' in config file {}", key, self.file_name)));
      }
      Some(v) => v.clone(),
    };

    self.used_keys.insert(key.to_string());

    Ok(value)
  }

  /// upstream: ConfigParser::getString (config_parser.cpp)
  pub fn get_string_possibles(&mut self, key: &str, possibles: &BTreeSet<String>) -> Result<String, KataError> {
    let value: String = self.get_string(key)?;
    if !possibles.contains(&value) {
      return Err(KataError::IOError(format!(
        "Key '{}' must be one of ({}) in config file {}",
        key,
        Global::concat_set(possibles, "|"),
        self.file_name
      )));
    }
    Ok(value)
  }

  /// upstream: ConfigParser::getStrings (config_parser.cpp)
  pub fn get_strings(&mut self, key: &str) -> Result<Vec<String>, KataError> {
    Ok(Global::split_delim(&self.get_string(key)?, ','))
  }

  /// upstream: ConfigParser::getStringsNonEmptyTrim (config_parser.cpp)
  pub fn get_strings_non_empty_trim(&mut self, key: &str) -> Result<Vec<String>, KataError> {
    let raw: Vec<String> = Global::split_delim(&self.get_string(key)?, ',');
    let mut trimmed: Vec<String> = Vec::new();
    for i in 0..raw.len() {
      let s: String = Global::trim(&raw[i], Global::TRIM_DEFAULT_DELIM);
      if s.is_empty() {
        continue;
      }
      trimmed.push(s);
    }
    Ok(trimmed)
  }

  /// upstream: ConfigParser::getStrings (config_parser.cpp)
  pub fn get_strings_possibles(&mut self, key: &str, possibles: &BTreeSet<String>) -> Result<Vec<String>, KataError> {
    let values: Vec<String> = self.get_strings(key)?;
    for i in 0..values.len() {
      let value: &String = &values[i];
      if !possibles.contains(value) {
        return Err(KataError::IOError(format!(
          "Key '{}' must be one of ({}) in config file {}",
          key,
          Global::concat_set(possibles, "|"),
          self.file_name
        )));
      }
    }
    Ok(values)
  }

  /// upstream: ConfigParser::getBool (config_parser.cpp)
  pub fn get_bool(&mut self, key: &str) -> Result<bool, KataError> {
    let value: String = self.get_string(key)?;
    let mut x: bool = false;
    if !Global::try_string_to_bool(&value, &mut x) {
      return Err(KataError::IOError(format!(
        "Could not parse '{}' as bool for key '{}' in config file {}",
        value, key, self.file_name
      )));
    }
    Ok(x)
  }

  /// upstream: ConfigParser::getBools (config_parser.cpp)
  pub fn get_bools(&mut self, key: &str) -> Result<Vec<bool>, KataError> {
    let values: Vec<String> = self.get_strings(key)?;
    let mut ret: Vec<bool> = Vec::new();
    for i in 0..values.len() {
      let value: &String = &values[i];
      let mut x: bool = false;
      if !Global::try_string_to_bool(value, &mut x) {
        return Err(KataError::IOError(format!(
          "Could not parse '{}' as bool for key '{}' in config file {}",
          value, key, self.file_name
        )));
      }
      ret.push(x);
    }
    Ok(ret)
  }

  /// upstream: ConfigParser::getEnabled (config_parser.cpp)
  pub fn get_enabled(&mut self, key: &str) -> Result<enabled_t, KataError> {
    let value: String = Global::trim(&Global::to_lower(&self.get_string(key)?), Global::TRIM_DEFAULT_DELIM);
    let mut x: enabled_t = enabled_t::False;
    if !enabled_t::try_parse(&value, &mut x) {
      return Err(KataError::IOError(format!(
        "Could not parse '{}' as bool or auto for key '{}' in config file {}",
        value, key, self.file_name
      )));
    }
    Ok(x)
  }

  /// upstream: ConfigParser::getInt (config_parser.cpp)
  /// PORT NOTE: the defaults `std::numeric_limits<int>::lowest()` / `max()` are passed explicitly by
  /// callers (`i32::MIN`, `i32::MAX`), as for every getXxx with a range (PORTING.md §5).
  pub fn get_int(&mut self, key: &str, min: i32, max: i32) -> Result<i32, KataError> {
    testAssert!(min <= max);
    let value: String = self.get_string(key)?;
    let mut x: i32 = 0;
    if !Global::try_string_to_int(&value, &mut x) {
      return Err(KataError::IOError(format!(
        "Could not parse '{}' as int for key '{}' in config file {}",
        value, key, self.file_name
      )));
    }
    if x < min || x > max {
      return Err(KataError::IOError(format!(
        "Key '{}' must be in the range {} to {} in config file {}",
        key,
        Global::int_to_string(min),
        Global::int_to_string(max),
        self.file_name
      )));
    }
    Ok(x)
  }

  /// upstream: ConfigParser::getInts (config_parser.cpp)
  pub fn get_ints(&mut self, key: &str, min: i32, max: i32) -> Result<Vec<i32>, KataError> {
    let values: Vec<String> = self.get_strings(key)?;
    let mut ret: Vec<i32> = Vec::new();
    for i in 0..values.len() {
      let value: &String = &values[i];
      let mut x: i32 = 0;
      if !Global::try_string_to_int(value, &mut x) {
        return Err(KataError::IOError(format!(
          "Could not parse '{}' as int for key '{}' in config file {}",
          value, key, self.file_name
        )));
      }
      if x < min || x > max {
        return Err(KataError::IOError(format!(
          "Key '{}' must be in the range {} to {} in config file {}",
          key,
          Global::int_to_string(min),
          Global::int_to_string(max),
          self.file_name
        )));
      }
      ret.push(x);
    }
    Ok(ret)
  }

  /// upstream: ConfigParser::getNonNegativeIntDashedPairs (config_parser.cpp)
  pub fn get_non_negative_int_dashed_pairs(&mut self, key: &str, min: i32, max: i32) -> Result<Vec<(i32, i32)>, KataError> {
    let pair_strs: Vec<String> = self.get_strings(key)?;
    let mut ret: Vec<(i32, i32)> = Vec::new();
    for pair_str in pair_strs.iter() {
      if Global::trim(pair_str, Global::TRIM_DEFAULT_DELIM).is_empty() {
        continue;
      }
      let pieces: Vec<String> = Global::split_delim(&Global::trim(pair_str, Global::TRIM_DEFAULT_DELIM), '-');
      if pieces.len() != 2 {
        return Err(KataError::IOError(format!(
          "Could not parse '{}' as a pair of integers separated by a dash for key '{}' in config file {}",
          pair_str, key, self.file_name
        )));
      }

      let suc: bool;
      let mut p0: i32 = 0;
      let mut p1: i32 = 0;
      suc = Global::try_string_to_int(&pieces[0], &mut p0);
      if !suc {
        return Err(KataError::IOError(format!(
          "Could not parse '{}' as a pair of integers separated by a dash for key '{}' in config file {}",
          pair_str, key, self.file_name
        )));
      }
      let suc: bool = Global::try_string_to_int(&pieces[1], &mut p1);
      if !suc {
        return Err(KataError::IOError(format!(
          "Could not parse '{}' as a pair of integers separated by a dash for key '{}' in config file {}",
          pair_str, key, self.file_name
        )));
      }

      if p0 < min || p0 > max || p1 < min || p1 > max {
        return Err(KataError::IOError(format!(
          "Expected key '{}' to have all values range {} to {} in config file {}",
          key,
          Global::int_to_string(min),
          Global::int_to_string(max),
          self.file_name
        )));
      }

      ret.push((p0, p1));
    }
    Ok(ret)
  }

  /// upstream: ConfigParser::getInt64 (config_parser.cpp)
  pub fn get_int64(&mut self, key: &str, min: i64, max: i64) -> Result<i64, KataError> {
    testAssert!(min <= max);
    let value: String = self.get_string(key)?;
    let mut x: i64 = 0;
    if !Global::try_string_to_int64(&value, &mut x) {
      return Err(KataError::IOError(format!(
        "Could not parse '{}' as int64_t for key '{}' in config file {}",
        value, key, self.file_name
      )));
    }
    if x < min || x > max {
      return Err(KataError::IOError(format!(
        "Key '{}' must be in the range {} to {} in config file {}",
        key,
        Global::int64_to_string(min),
        Global::int64_to_string(max),
        self.file_name
      )));
    }
    Ok(x)
  }

  /// upstream: ConfigParser::getInt64s (config_parser.cpp)
  pub fn get_int64s(&mut self, key: &str, min: i64, max: i64) -> Result<Vec<i64>, KataError> {
    let values: Vec<String> = self.get_strings(key)?;
    let mut ret: Vec<i64> = Vec::new();
    for i in 0..values.len() {
      let value: &String = &values[i];
      let mut x: i64 = 0;
      if !Global::try_string_to_int64(value, &mut x) {
        return Err(KataError::IOError(format!(
          "Could not parse '{}' as int64_t for key '{}' in config file {}",
          value, key, self.file_name
        )));
      }
      if x < min || x > max {
        return Err(KataError::IOError(format!(
          "Key '{}' must be in the range {} to {} in config file {}",
          key,
          Global::int64_to_string(min),
          Global::int64_to_string(max),
          self.file_name
        )));
      }
      ret.push(x);
    }
    Ok(ret)
  }

  /// upstream: ConfigParser::getUInt64 (config_parser.cpp)
  /// PORT NOTE: the parse error says "int64_t" (not "uint64_t") upstream; kept verbatim.
  pub fn get_uint64(&mut self, key: &str, min: u64, max: u64) -> Result<u64, KataError> {
    testAssert!(min <= max);
    let value: String = self.get_string(key)?;
    let mut x: u64 = 0;
    if !Global::try_string_to_uint64(&value, &mut x) {
      return Err(KataError::IOError(format!(
        "Could not parse '{}' as int64_t for key '{}' in config file {}",
        value, key, self.file_name
      )));
    }
    if x < min || x > max {
      return Err(KataError::IOError(format!(
        "Key '{}' must be in the range {} to {} in config file {}",
        key,
        Global::uint64_to_string(min),
        Global::uint64_to_string(max),
        self.file_name
      )));
    }
    Ok(x)
  }

  /// upstream: ConfigParser::getUInt64s (config_parser.cpp)
  pub fn get_uint64s(&mut self, key: &str, min: u64, max: u64) -> Result<Vec<u64>, KataError> {
    let values: Vec<String> = self.get_strings(key)?;
    let mut ret: Vec<u64> = Vec::new();
    for i in 0..values.len() {
      let value: &String = &values[i];
      let mut x: u64 = 0;
      if !Global::try_string_to_uint64(value, &mut x) {
        return Err(KataError::IOError(format!(
          "Could not parse '{}' as uint64_t for key '{}' in config file {}",
          value, key, self.file_name
        )));
      }
      if x < min || x > max {
        return Err(KataError::IOError(format!(
          "Key '{}' must be in the range {} to {} in config file {}",
          key,
          Global::uint64_to_string(min),
          Global::uint64_to_string(max),
          self.file_name
        )));
      }
      ret.push(x);
    }
    Ok(ret)
  }

  /// upstream: ConfigParser::getFloat (config_parser.cpp)
  pub fn get_float(&mut self, key: &str, min: f32, max: f32) -> Result<f32, KataError> {
    testAssert!(min <= max);
    let value: String = self.get_string(key)?;
    let mut x: f32 = 0.0;
    if !Global::try_string_to_float(&value, &mut x) {
      return Err(KataError::IOError(format!(
        "Could not parse '{}' as float for key '{}' in config file {}",
        value, key, self.file_name
      )));
    }
    if x.is_nan() {
      return Err(KataError::IOError(format!("Key '{}' is nan in config file {}", key, self.file_name)));
    }
    if x < min || x > max {
      return Err(KataError::IOError(format!(
        "Key '{}' must be in the range {} to {} in config file {}",
        key,
        Global::float_to_string(min),
        Global::float_to_string(max),
        self.file_name
      )));
    }
    Ok(x)
  }

  /// upstream: ConfigParser::getFloats (config_parser.cpp)
  pub fn get_floats(&mut self, key: &str, min: f32, max: f32) -> Result<Vec<f32>, KataError> {
    let values: Vec<String> = self.get_strings(key)?;
    let mut ret: Vec<f32> = Vec::new();
    for i in 0..values.len() {
      let value: &String = &values[i];
      let mut x: f32 = 0.0;
      if !Global::try_string_to_float(value, &mut x) {
        return Err(KataError::IOError(format!(
          "Could not parse '{}' as float for key '{}' in config file {}",
          value, key, self.file_name
        )));
      }
      if x.is_nan() {
        return Err(KataError::IOError(format!("Key '{}' is nan in config file {}", key, self.file_name)));
      }
      if x < min || x > max {
        return Err(KataError::IOError(format!(
          "Key '{}' must be in the range {} to {} in config file {}",
          key,
          Global::float_to_string(min),
          Global::float_to_string(max),
          self.file_name
        )));
      }
      ret.push(x);
    }
    Ok(ret)
  }

  /// upstream: ConfigParser::getDouble (config_parser.cpp)
  pub fn get_double(&mut self, key: &str, min: f64, max: f64) -> Result<f64, KataError> {
    testAssert!(min <= max);
    let value: String = self.get_string(key)?;
    let mut x: f64 = 0.0;
    if !Global::try_string_to_double(&value, &mut x) {
      return Err(KataError::IOError(format!(
        "Could not parse '{}' as double for key '{}' in config file {}",
        value, key, self.file_name
      )));
    }
    if x.is_nan() {
      return Err(KataError::IOError(format!("Key '{}' is nan in config file {}", key, self.file_name)));
    }
    if x < min || x > max {
      return Err(KataError::IOError(format!(
        "Key '{}' must be in the range {} to {} in config file {}",
        key,
        Global::double_to_string(min),
        Global::double_to_string(max),
        self.file_name
      )));
    }
    Ok(x)
  }

  /// upstream: ConfigParser::getDoubles (config_parser.cpp)
  pub fn get_doubles(&mut self, key: &str, min: f64, max: f64) -> Result<Vec<f64>, KataError> {
    let values: Vec<String> = self.get_strings(key)?;
    let mut ret: Vec<f64> = Vec::new();
    for i in 0..values.len() {
      let value: &String = &values[i];
      let mut x: f64 = 0.0;
      if !Global::try_string_to_double(value, &mut x) {
        return Err(KataError::IOError(format!(
          "Could not parse '{}' as double for key '{}' in config file {}",
          value, key, self.file_name
        )));
      }
      if x.is_nan() {
        return Err(KataError::IOError(format!("Key '{}' is nan in config file {}", key, self.file_name)));
      }
      if x < min || x > max {
        return Err(KataError::IOError(format!(
          "Key '{}' must be in the range {} to {} in config file {}",
          key,
          Global::double_to_string(min),
          Global::double_to_string(max),
          self.file_name
        )));
      }
      ret.push(x);
    }
    Ok(ret)
  }
}
