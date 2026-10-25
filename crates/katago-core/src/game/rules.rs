// Ported from KataGo cpp/game/rules.cpp, cpp/game/rules.h @ ba938676d7f42d70950b3a535af2466fb642008c

use std::collections::BTreeSet;

use serde_json::Map;
use serde_json::Value;

use crate::core::global::Global;
use crate::core::global::KataError;
use crate::core::hash::Hash128;
use crate::port::cpp;
use crate::port::fmt::cpp_ostream_f64;
use crate::port::nlohmann;

/// upstream: Rules (rules.h)
#[derive(Clone, Copy, Debug)]
pub struct Rules {
  pub ko_rule: i32,

  pub scoring_rule: i32,

  pub tax_rule: i32,

  pub multi_stone_suicide_legal: bool,
  pub has_button: bool,

  pub white_handicap_bonus_rule: i32,

  //Mostly an informational value - doesn't affect the actual implemented rules, but GTP or Analysis may, at a
  //high level, use this info to adjust passing behavior - whether it's okay to pass without capturing dead stones.
  //Only relevant for area scoring.
  pub friendly_pass_ok: bool,

  pub komi: f32,
}

impl Rules {
  /// upstream: Rules::KO_SIMPLE, Rules::KO_POSITIONAL, Rules::KO_SITUATIONAL, Rules::KO_SPIGHT (rules.h)
  pub const KO_SIMPLE: i32 = 0;
  pub const KO_POSITIONAL: i32 = 1;
  pub const KO_SITUATIONAL: i32 = 2;
  pub const KO_SPIGHT: i32 = 3;

  /// upstream: Rules::SCORING_AREA, Rules::SCORING_TERRITORY (rules.h)
  pub const SCORING_AREA: i32 = 0;
  pub const SCORING_TERRITORY: i32 = 1;

  /// upstream: Rules::TAX_NONE, Rules::TAX_SEKI, Rules::TAX_ALL (rules.h)
  pub const TAX_NONE: i32 = 0;
  pub const TAX_SEKI: i32 = 1;
  pub const TAX_ALL: i32 = 2;

  /// upstream: Rules::WHB_ZERO, Rules::WHB_N, Rules::WHB_N_MINUS_ONE (rules.h)
  pub const WHB_ZERO: i32 = 0;
  pub const WHB_N: i32 = 1;
  pub const WHB_N_MINUS_ONE: i32 = 2;

  /// upstream: Rules::MIN_USER_KOMI, Rules::MAX_USER_KOMI (rules.h)
  //Min and max acceptable komi in various places involving user input validation
  pub const MIN_USER_KOMI: f32 = -150.0f32;
  pub const MAX_USER_KOMI: f32 = 150.0f32;

  /// upstream: Rules::Rules (rules.cpp)
  pub fn new() -> Rules {
    // PORT NOTE: the C++ default constructor assigns every field in its body; the struct
    // literal holds the same values in the same order.
    //Defaults if not set - closest match to TT rules
    Rules {
      ko_rule: Rules::KO_POSITIONAL,
      scoring_rule: Rules::SCORING_AREA,
      tax_rule: Rules::TAX_NONE,
      multi_stone_suicide_legal: true,
      has_button: false,
      white_handicap_bonus_rule: Rules::WHB_ZERO,
      friendly_pass_ok: false,
      komi: 7.5f32,
    }
  }

  /// upstream: Rules::Rules (rules.cpp)
  /// PORT NOTE: name from PORTING.md Appendix B (overrides overload rule 3).
  pub fn new_full(
    k_rule: i32,
    s_rule: i32,
    t_rule: i32,
    suic: bool,
    button: bool,
    whb_rule: i32,
    p_ok: bool,
    km: f32,
  ) -> Rules {
    Rules {
      ko_rule: k_rule,
      scoring_rule: s_rule,
      tax_rule: t_rule,
      multi_stone_suicide_legal: suic,
      has_button: button,
      white_handicap_bonus_rule: whb_rule,
      friendly_pass_ok: p_ok,
      komi: km,
    }
  }

  /// upstream: Rules::equalsIgnoringKomi (rules.cpp)
  pub fn equals_ignoring_komi(&self, other: &Rules) -> bool {
    self.ko_rule == other.ko_rule
      && self.scoring_rule == other.scoring_rule
      && self.tax_rule == other.tax_rule
      && self.multi_stone_suicide_legal == other.multi_stone_suicide_legal
      && self.has_button == other.has_button
      && self.white_handicap_bonus_rule == other.white_handicap_bonus_rule
      && self.friendly_pass_ok == other.friendly_pass_ok
  }

  /// upstream: Rules::gameResultWillBeInteger (rules.cpp)
  pub fn game_result_will_be_integer(&self) -> bool {
    // PORT NOTE: `(int)komi` is C++ UB for NaN/out-of-range komi; x86-64 cvttss2si gives
    // INT_MIN, which port::cpp::static_cast_i32_f32 reproduces (Rust `as` would saturate).
    let komi_is_integer: bool = (cpp::static_cast_i32_f32(self.komi) as f32) == self.komi;
    komi_is_integer != self.has_button
  }

  /// upstream: Rules::getTrompTaylorish (rules.cpp)
  pub fn get_tromp_taylorish() -> Rules {
    let mut rules = Rules::new();
    rules.ko_rule = Rules::KO_POSITIONAL;
    rules.scoring_rule = Rules::SCORING_AREA;
    rules.tax_rule = Rules::TAX_NONE;
    rules.multi_stone_suicide_legal = true;
    rules.has_button = false;
    rules.white_handicap_bonus_rule = Rules::WHB_ZERO;
    rules.friendly_pass_ok = false;
    rules.komi = 7.5f32;
    rules
  }

  /// upstream: Rules::getSimpleTerritory (rules.cpp)
  pub fn get_simple_territory() -> Rules {
    let mut rules = Rules::new();
    rules.ko_rule = Rules::KO_SIMPLE;
    rules.scoring_rule = Rules::SCORING_TERRITORY;
    rules.tax_rule = Rules::TAX_SEKI;
    rules.multi_stone_suicide_legal = false;
    rules.has_button = false;
    rules.white_handicap_bonus_rule = Rules::WHB_ZERO;
    rules.friendly_pass_ok = false;
    rules.komi = 7.5f32;
    rules
  }

  /// upstream: Rules::komiIsIntOrHalfInt (rules.cpp)
  pub fn komi_is_int_or_half_int(komi: f32) -> bool {
    // PORT NOTE: `(int)(komi * 2)` is C++ UB out of range; x86-64 cvttss2si gives INT_MIN,
    // which port::cpp::static_cast_i32_f32 reproduces (Rust `as` would saturate).
    komi.is_finite() && komi * 2.0f32 == cpp::static_cast_i32_f32(komi * 2.0f32) as f32
  }

  /// upstream: Rules::koRuleStrings (rules.cpp)
  pub fn ko_rule_strings() -> BTreeSet<String> {
    ["SIMPLE", "POSITIONAL", "SITUATIONAL", "SPIGHT"].iter().map(|s| s.to_string()).collect()
  }
  /// upstream: Rules::scoringRuleStrings (rules.cpp)
  pub fn scoring_rule_strings() -> BTreeSet<String> {
    ["AREA", "TERRITORY"].iter().map(|s| s.to_string()).collect()
  }
  /// upstream: Rules::taxRuleStrings (rules.cpp)
  pub fn tax_rule_strings() -> BTreeSet<String> {
    ["NONE", "SEKI", "ALL"].iter().map(|s| s.to_string()).collect()
  }
  /// upstream: Rules::whiteHandicapBonusRuleStrings (rules.cpp)
  pub fn white_handicap_bonus_rule_strings() -> BTreeSet<String> {
    ["0", "N", "N-1"].iter().map(|s| s.to_string()).collect()
  }

  /// upstream: Rules::parseKoRule (rules.cpp)
  pub fn parse_ko_rule(s: &str) -> Result<i32, KataError> {
    if s == "SIMPLE" {
      Ok(Rules::KO_SIMPLE)
    } else if s == "POSITIONAL" {
      Ok(Rules::KO_POSITIONAL)
    } else if s == "SITUATIONAL" {
      Ok(Rules::KO_SITUATIONAL)
    } else if s == "SPIGHT" {
      Ok(Rules::KO_SPIGHT)
    } else {
      Err(KataError::IOError(String::from("Rules::parseKoRule: Invalid ko rule: ") + s))
    }
  }
  /// upstream: Rules::parseScoringRule (rules.cpp)
  pub fn parse_scoring_rule(s: &str) -> Result<i32, KataError> {
    if s == "AREA" {
      Ok(Rules::SCORING_AREA)
    } else if s == "TERRITORY" {
      Ok(Rules::SCORING_TERRITORY)
    } else {
      Err(KataError::IOError(String::from("Rules::parseScoringRule: Invalid scoring rule: ") + s))
    }
  }
  /// upstream: Rules::parseTaxRule (rules.cpp)
  pub fn parse_tax_rule(s: &str) -> Result<i32, KataError> {
    if s == "NONE" {
      Ok(Rules::TAX_NONE)
    } else if s == "SEKI" {
      Ok(Rules::TAX_SEKI)
    } else if s == "ALL" {
      Ok(Rules::TAX_ALL)
    } else {
      Err(KataError::IOError(String::from("Rules::parseTaxRule: Invalid tax rule: ") + s))
    }
  }
  /// upstream: Rules::parseWhiteHandicapBonusRule (rules.cpp)
  pub fn parse_white_handicap_bonus_rule(s: &str) -> Result<i32, KataError> {
    if s == "0" {
      Ok(Rules::WHB_ZERO)
    } else if s == "N" {
      Ok(Rules::WHB_N)
    } else if s == "N-1" {
      Ok(Rules::WHB_N_MINUS_ONE)
    } else {
      Err(KataError::IOError(String::from("Rules::parseWhiteHandicapBonusRule: Invalid whiteHandicapBonus rule: ") + s))
    }
  }

  /// upstream: Rules::writeKoRule (rules.cpp)
  pub fn write_ko_rule(ko_rule: i32) -> String {
    if ko_rule == Rules::KO_SIMPLE {
      return String::from("SIMPLE");
    }
    if ko_rule == Rules::KO_POSITIONAL {
      return String::from("POSITIONAL");
    }
    if ko_rule == Rules::KO_SITUATIONAL {
      return String::from("SITUATIONAL");
    }
    if ko_rule == Rules::KO_SPIGHT {
      return String::from("SPIGHT");
    }
    String::from("UNKNOWN")
  }
  /// upstream: Rules::writeScoringRule (rules.cpp)
  pub fn write_scoring_rule(scoring_rule: i32) -> String {
    if scoring_rule == Rules::SCORING_AREA {
      return String::from("AREA");
    }
    if scoring_rule == Rules::SCORING_TERRITORY {
      return String::from("TERRITORY");
    }
    String::from("UNKNOWN")
  }
  /// upstream: Rules::writeTaxRule (rules.cpp)
  pub fn write_tax_rule(tax_rule: i32) -> String {
    if tax_rule == Rules::TAX_NONE {
      return String::from("NONE");
    }
    if tax_rule == Rules::TAX_SEKI {
      return String::from("SEKI");
    }
    if tax_rule == Rules::TAX_ALL {
      return String::from("ALL");
    }
    String::from("UNKNOWN")
  }
  /// upstream: Rules::writeWhiteHandicapBonusRule (rules.cpp)
  pub fn write_white_handicap_bonus_rule(white_handicap_bonus_rule: i32) -> String {
    if white_handicap_bonus_rule == Rules::WHB_ZERO {
      return String::from("0");
    }
    if white_handicap_bonus_rule == Rules::WHB_N {
      return String::from("N");
    }
    if white_handicap_bonus_rule == Rules::WHB_N_MINUS_ONE {
      return String::from("N-1");
    }
    String::from("UNKNOWN")
  }

  /// upstream: Rules::toStringNoKomi (rules.cpp)
  pub fn to_string_no_komi(&self) -> String {
    let mut out = String::new();
    out += "ko";
    out += &Rules::write_ko_rule(self.ko_rule);
    out += "score";
    out += &Rules::write_scoring_rule(self.scoring_rule);
    out += "tax";
    out += &Rules::write_tax_rule(self.tax_rule);
    out += "sui";
    out += &(self.multi_stone_suicide_legal as i32).to_string();
    if self.has_button {
      out += "button";
      out += &(self.has_button as i32).to_string();
    }
    if self.white_handicap_bonus_rule != Rules::WHB_ZERO {
      out += "whb";
      out += &Rules::write_white_handicap_bonus_rule(self.white_handicap_bonus_rule);
    }
    if self.friendly_pass_ok {
      out += "fpok";
      out += &(self.friendly_pass_ok as i32).to_string();
    }
    out
  }

  /// upstream: Rules::toString (rules.cpp)
  pub fn to_string(&self) -> String {
    format!("{}", self)
  }

  /// upstream: Rules::toJsonHelper (rules.cpp)
  //omitDefaults: Takes up a lot of string space to include stuff, so omit some less common things if matches tromp-taylor rules
  //which is the default for parsing and if not otherwise specified
  pub(crate) fn to_json_helper(&self, omit_komi: bool, omit_defaults: bool) -> Value {
    let mut ret: Map<String, Value> = Map::new();
    ret.insert(String::from("ko"), Value::from(Rules::write_ko_rule(self.ko_rule)));
    ret.insert(String::from("scoring"), Value::from(Rules::write_scoring_rule(self.scoring_rule)));
    ret.insert(String::from("tax"), Value::from(Rules::write_tax_rule(self.tax_rule)));
    ret.insert(String::from("suicide"), Value::from(self.multi_stone_suicide_legal));
    if !omit_defaults || self.has_button {
      ret.insert(String::from("hasButton"), Value::from(self.has_button));
    }
    if !omit_defaults || self.white_handicap_bonus_rule != Rules::WHB_ZERO {
      ret.insert(String::from("whiteHandicapBonus"), Value::from(Rules::write_white_handicap_bonus_rule(self.white_handicap_bonus_rule)));
    }
    if !omit_defaults || self.friendly_pass_ok != false {
      ret.insert(String::from("friendlyPassOk"), Value::from(self.friendly_pass_ok));
    }
    if !omit_komi {
      // PORT NOTE: nlohmann stores a float as a double (PORTING.md §9).
      ret.insert(String::from("komi"), Value::from(self.komi as f64));
    }
    Value::Object(ret)
  }

  /// upstream: Rules::toJson (rules.cpp)
  pub fn to_json(&self) -> Value {
    self.to_json_helper(false, false)
  }

  /// upstream: Rules::toJsonNoKomi (rules.cpp)
  pub fn to_json_no_komi(&self) -> Value {
    self.to_json_helper(true, false)
  }

  /// upstream: Rules::toJsonNoKomiMaybeOmitStuff (rules.cpp)
  pub fn to_json_no_komi_maybe_omit_stuff(&self) -> Value {
    self.to_json_helper(true, true)
  }

  /// upstream: Rules::toJsonString (rules.cpp)
  pub fn to_json_string(&self) -> String {
    nlohmann::dump(&self.to_json_helper(false, false))
  }

  /// upstream: Rules::toJsonStringNoKomi (rules.cpp)
  pub fn to_json_string_no_komi(&self) -> String {
    nlohmann::dump(&self.to_json_helper(true, false))
  }

  /// upstream: Rules::toJsonStringNoKomiMaybeOmitStuff (rules.cpp)
  pub fn to_json_string_no_komi_maybe_omit_stuff(&self) -> String {
    nlohmann::dump(&self.to_json_helper(true, true))
  }

  /// upstream: Rules::updateRules (rules.cpp)
  pub fn update_rules(k: &str, v: &str, old_rules: &Rules) -> Result<Rules, KataError> {
    let mut rules: Rules = *old_rules;
    let key: String = Global::trim(k, Global::TRIM_DEFAULT_DELIM);
    let value: String = Global::trim(&Global::to_upper(v), Global::TRIM_DEFAULT_DELIM);
    if key == "ko" {
      rules.ko_rule = Rules::parse_ko_rule(&value)?;
    } else if key == "score" {
      rules.scoring_rule = Rules::parse_scoring_rule(&value)?;
    } else if key == "scoring" {
      rules.scoring_rule = Rules::parse_scoring_rule(&value)?;
    } else if key == "tax" {
      rules.tax_rule = Rules::parse_tax_rule(&value)?;
    } else if key == "suicide" {
      rules.multi_stone_suicide_legal = Global::string_to_bool(&value)?;
    } else if key == "hasButton" {
      rules.has_button = Global::string_to_bool(&value)?;
    } else if key == "whiteHandicapBonus" {
      rules.white_handicap_bonus_rule = Rules::parse_white_handicap_bonus_rule(&value)?;
    } else if key == "friendlyPassOk" {
      rules.friendly_pass_ok = Global::string_to_bool(&value)?;
    } else {
      return Err(KataError::IOError(String::from("Unknown rules option: ") + &key));
    }
    Ok(rules)
  }

  /// upstream: Rules::parseRules (rules.cpp)
  pub fn parse_rules(s_orig: &str) -> Result<Rules, KataError> {
    parse_rules_helper(s_orig, true)
  }
  /// upstream: Rules::parseRulesWithoutKomi (rules.cpp)
  pub fn parse_rules_without_komi(s_orig: &str, komi: f32) -> Result<Rules, KataError> {
    let mut rules: Rules = parse_rules_helper(s_orig, false)?;
    rules.komi = komi;
    Ok(rules)
  }

  /// upstream: Rules::tryParseRules (rules.cpp)
  /// PORT NOTE: `catch(const StringError&)` only; any other error propagates, hence Result.
  pub fn try_parse_rules(s_orig: &str, buf: &mut Rules) -> Result<bool, KataError> {
    let rules: Rules;
    match parse_rules_helper(s_orig, true) {
      Ok(r) => rules = r,
      Err(e) if e.is_string_error() => return Ok(false),
      Err(e) => return Err(e),
    }
    *buf = rules;
    Ok(true)
  }
  /// upstream: Rules::tryParseRulesWithoutKomi (rules.cpp)
  /// PORT NOTE: `catch(const StringError&)` only; any other error propagates, hence Result.
  pub fn try_parse_rules_without_komi(s_orig: &str, buf: &mut Rules, komi: f32) -> Result<bool, KataError> {
    let mut rules: Rules;
    match parse_rules_helper(s_orig, false) {
      Ok(r) => rules = r,
      Err(e) if e.is_string_error() => return Ok(false),
      Err(e) => return Err(e),
    }
    rules.komi = komi;
    *buf = rules;
    Ok(true)
  }

  /// upstream: Rules::toStringNoKomiMaybeNice (rules.cpp)
  pub fn to_string_no_komi_maybe_nice(&self) -> String {
    // PORT NOTE: parseRulesHelper cannot throw for these constant inputs (PORTING.md §7).
    if self.equals_ignoring_komi(&parse_rules_helper("TrompTaylor", false).expect("upstream: cannot throw for constant input")) {
      return String::from("TrompTaylor");
    }
    if self.equals_ignoring_komi(&parse_rules_helper("Japanese", false).expect("upstream: cannot throw for constant input")) {
      return String::from("Japanese");
    }
    if self.equals_ignoring_komi(&parse_rules_helper("Chinese", false).expect("upstream: cannot throw for constant input")) {
      return String::from("Chinese");
    }
    if self.equals_ignoring_komi(&parse_rules_helper("Chinese-OGS", false).expect("upstream: cannot throw for constant input")) {
      return String::from("Chinese-OGS");
    }
    if self.equals_ignoring_komi(&parse_rules_helper("AGA", false).expect("upstream: cannot throw for constant input")) {
      return String::from("AGA");
    }
    if self.equals_ignoring_komi(&parse_rules_helper("StoneScoring", false).expect("upstream: cannot throw for constant input")) {
      return String::from("StoneScoring");
    }
    if self.equals_ignoring_komi(&parse_rules_helper("NewZealand", false).expect("upstream: cannot throw for constant input")) {
      return String::from("NewZealand");
    }
    self.to_string_no_komi()
  }

  /// upstream: Rules::ZOBRIST_KO_RULE_HASH (rules.cpp)
  pub const ZOBRIST_KO_RULE_HASH: [Hash128; 4] = [
    Hash128::new(0x3cc7e0bf846820f6, 0x1fb7fbde5fc6ba4e), //Based on sha256 hash of Rules::KO_SIMPLE
    Hash128::new(0xcc18f5d47188554a, 0x3a63152c23e4128d), //Based on sha256 hash of Rules::KO_POSITIONAL
    Hash128::new(0x3bc55e42b23b35bf, 0xc75fa1e615621dcd), //Based on sha256 hash of Rules::KO_SITUATIONAL
    Hash128::new(0x5b2096e48241d21b, 0x23cc18d4e85cd67f), //Based on sha256 hash of Rules::KO_SPIGHT
  ];

  /// upstream: Rules::ZOBRIST_SCORING_RULE_HASH (rules.cpp)
  pub const ZOBRIST_SCORING_RULE_HASH: [Hash128; 2] = [
    //Based on sha256 hash of Rules::SCORING_AREA, but also mixing none tax rule hash, to preserve legacy hashes
    Hash128::new(0x8b3ed7598f901494 ^ 0x72eeccc72c82a5e7, 0x1dfd47ac77bce5f8 ^ 0x0d1265e413623e2b),
    //Based on sha256 hash of Rules::SCORING_TERRITORY, but also mixing seki tax rule hash, to preserve legacy hashes
    Hash128::new(0x381345dc357ec982 ^ 0x125bfe48a41042d5, 0x03ba55c026026b56 ^ 0x061866b5f2b98a79),
  ];
  /// upstream: Rules::ZOBRIST_TAX_RULE_HASH (rules.cpp)
  pub const ZOBRIST_TAX_RULE_HASH: [Hash128; 3] = [
    Hash128::new(0x72eeccc72c82a5e7, 0x0d1265e413623e2b), //Based on sha256 hash of Rules::TAX_NONE
    Hash128::new(0x125bfe48a41042d5, 0x061866b5f2b98a79), //Based on sha256 hash of Rules::TAX_SEKI
    Hash128::new(0xa384ece9d8ee713c, 0xfdc9f3b5d1f3732b), //Based on sha256 hash of Rules::TAX_ALL
  ];

  /// upstream: Rules::ZOBRIST_MULTI_STONE_SUICIDE_HASH (rules.cpp)
  pub const ZOBRIST_MULTI_STONE_SUICIDE_HASH: Hash128 = //Based on sha256 hash of Rules::ZOBRIST_MULTI_STONE_SUICIDE_HASH
    Hash128::new(0xf9b475b3bbf35e37, 0xefa19d8b1e5b3e5a);

  /// upstream: Rules::ZOBRIST_BUTTON_HASH (rules.cpp)
  pub const ZOBRIST_BUTTON_HASH: Hash128 = //Based on sha256 hash of Rules::ZOBRIST_BUTTON_HASH
    Hash128::new(0xb8b914c9234ece84, 0x3d759cddebe29c14);

  /// upstream: Rules::ZOBRIST_FRIENDLY_PASS_OK_HASH (rules.cpp)
  pub const ZOBRIST_FRIENDLY_PASS_OK_HASH: Hash128 = //Based on sha256 hash of Rules::ZOBRIST_FRIENDLY_PASS_OK_HASH
    Hash128::new(0x0113655998ef0a25, 0x99c9d04ecd964874);
}

/// upstream: Rules::operator==, Rules::operator!= (rules.cpp)
/// PORT NOTE: `!=` is the negation of `==`; upstream's field-wise `!=` is equivalent (NaN komi
/// compares unequal both ways).
impl PartialEq for Rules {
  fn eq(&self, other: &Rules) -> bool {
    self.ko_rule == other.ko_rule
      && self.scoring_rule == other.scoring_rule
      && self.tax_rule == other.tax_rule
      && self.multi_stone_suicide_legal == other.multi_stone_suicide_legal
      && self.has_button == other.has_button
      && self.white_handicap_bonus_rule == other.white_handicap_bonus_rule
      && self.friendly_pass_ok == other.friendly_pass_ok
      && self.komi == other.komi
  }
}

/// upstream: operator<<(ostream&, const Rules&) (rules.cpp)
impl std::fmt::Display for Rules {
  fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(
      out,
      "ko{}score{}tax{}sui{}",
      Rules::write_ko_rule(self.ko_rule),
      Rules::write_scoring_rule(self.scoring_rule),
      Rules::write_tax_rule(self.tax_rule),
      self.multi_stone_suicide_legal as i32
    )?;
    if self.has_button {
      write!(out, "button{}", self.has_button as i32)?;
    }
    if self.white_handicap_bonus_rule != Rules::WHB_ZERO {
      write!(out, "whb{}", Rules::write_white_handicap_bonus_rule(self.white_handicap_bonus_rule))?;
    }
    if self.friendly_pass_ok {
      write!(out, "fpok{}", self.friendly_pass_ok as i32)?;
    }
    write!(out, "komi{}", cpp_ostream_f64(self.komi as f64))?;
    Ok(())
  }
}

/// upstream: parseRulesHelper (rules.cpp) static
fn parse_rules_helper(s_orig: &str, allow_komi: bool) -> Result<Rules, KataError> {
  let mut rules = Rules::new();
  let lowercased: String = Global::trim(&Global::to_lower(s_orig), Global::TRIM_DEFAULT_DELIM);
  if lowercased == "japanese" || lowercased == "korean" {
    rules.scoring_rule = Rules::SCORING_TERRITORY;
    rules.ko_rule = Rules::KO_SIMPLE;
    rules.tax_rule = Rules::TAX_SEKI;
    rules.multi_stone_suicide_legal = false;
    rules.has_button = false;
    rules.white_handicap_bonus_rule = Rules::WHB_ZERO;
    rules.friendly_pass_ok = false;
    rules.komi = 6.5f64 as f32;
  } else if lowercased == "chinese" {
    rules.scoring_rule = Rules::SCORING_AREA;
    rules.ko_rule = Rules::KO_SIMPLE;
    rules.tax_rule = Rules::TAX_NONE;
    rules.multi_stone_suicide_legal = false;
    rules.has_button = false;
    rules.white_handicap_bonus_rule = Rules::WHB_N;
    rules.friendly_pass_ok = true;
    rules.komi = 7.5f64 as f32;
  } else if lowercased == "chineseogs"
    || lowercased == "chinese_ogs"
    || lowercased == "chinese-ogs"
    || lowercased == "chinesekgs"
    || lowercased == "chinese_kgs"
    || lowercased == "chinese-kgs"
  {
    rules.scoring_rule = Rules::SCORING_AREA;
    rules.ko_rule = Rules::KO_POSITIONAL;
    rules.tax_rule = Rules::TAX_NONE;
    rules.multi_stone_suicide_legal = false;
    rules.has_button = false;
    rules.white_handicap_bonus_rule = Rules::WHB_N;
    rules.friendly_pass_ok = true;
    rules.komi = 7.5f64 as f32;
  } else if lowercased == "ancientarea"
    || lowercased == "ancient-area"
    || lowercased == "ancient_area"
    || lowercased == "ancient area"
    || lowercased == "stonescoring"
    || lowercased == "stone-scoring"
    || lowercased == "stone_scoring"
    || lowercased == "stone scoring"
  {
    rules.scoring_rule = Rules::SCORING_AREA;
    rules.ko_rule = Rules::KO_SIMPLE;
    rules.tax_rule = Rules::TAX_ALL;
    rules.multi_stone_suicide_legal = false;
    rules.has_button = false;
    rules.white_handicap_bonus_rule = Rules::WHB_ZERO;
    rules.friendly_pass_ok = true;
    rules.komi = 7.5f64 as f32;
  } else if lowercased == "ancientterritory"
    || lowercased == "ancient-territory"
    || lowercased == "ancient_territory"
    || lowercased == "ancient territory"
  {
    rules.scoring_rule = Rules::SCORING_TERRITORY;
    rules.ko_rule = Rules::KO_SIMPLE;
    rules.tax_rule = Rules::TAX_ALL;
    rules.multi_stone_suicide_legal = false;
    rules.has_button = false;
    rules.white_handicap_bonus_rule = Rules::WHB_ZERO;
    rules.friendly_pass_ok = false;
    rules.komi = 6.5f64 as f32;
  } else if lowercased == "agabutton" || lowercased == "aga-button" || lowercased == "aga_button" || lowercased == "aga button" {
    rules.scoring_rule = Rules::SCORING_AREA;
    rules.ko_rule = Rules::KO_SITUATIONAL;
    rules.tax_rule = Rules::TAX_NONE;
    rules.multi_stone_suicide_legal = false;
    rules.has_button = true;
    rules.white_handicap_bonus_rule = Rules::WHB_N_MINUS_ONE;
    rules.friendly_pass_ok = true;
    rules.komi = 7.0f64 as f32;
  } else if lowercased == "aga" || lowercased == "bga" || lowercased == "french" {
    rules.scoring_rule = Rules::SCORING_AREA;
    rules.ko_rule = Rules::KO_SITUATIONAL;
    rules.tax_rule = Rules::TAX_NONE;
    rules.multi_stone_suicide_legal = false;
    rules.has_button = false;
    rules.white_handicap_bonus_rule = Rules::WHB_N_MINUS_ONE;
    rules.friendly_pass_ok = true;
    rules.komi = 7.5f64 as f32;
  } else if lowercased == "nz"
    || lowercased == "newzealand"
    || lowercased == "new zealand"
    || lowercased == "new-zealand"
    || lowercased == "new_zealand"
  {
    rules.scoring_rule = Rules::SCORING_AREA;
    rules.ko_rule = Rules::KO_SITUATIONAL;
    rules.tax_rule = Rules::TAX_NONE;
    rules.multi_stone_suicide_legal = true;
    rules.has_button = false;
    rules.white_handicap_bonus_rule = Rules::WHB_ZERO;
    rules.friendly_pass_ok = true;
    rules.komi = 7.5f64 as f32;
  } else if lowercased == "tromp-taylor" || lowercased == "tromp_taylor" || lowercased == "tromp taylor" || lowercased == "tromptaylor" {
    rules.scoring_rule = Rules::SCORING_AREA;
    rules.ko_rule = Rules::KO_POSITIONAL;
    rules.tax_rule = Rules::TAX_NONE;
    rules.multi_stone_suicide_legal = true;
    rules.has_button = false;
    rules.white_handicap_bonus_rule = Rules::WHB_ZERO;
    rules.friendly_pass_ok = false;
    rules.komi = 7.5f64 as f32;
  } else if lowercased == "goe" || lowercased == "ing" {
    rules.scoring_rule = Rules::SCORING_AREA;
    rules.ko_rule = Rules::KO_POSITIONAL;
    rules.tax_rule = Rules::TAX_NONE;
    rules.multi_stone_suicide_legal = true;
    rules.has_button = false;
    rules.white_handicap_bonus_rule = Rules::WHB_ZERO;
    rules.friendly_pass_ok = true;
    rules.komi = 7.5f64 as f32;
  } else if !s_orig.is_empty() && s_orig.as_bytes()[0] == b'{' {
    //Default if not specified
    rules = Rules::get_tromp_taylorish();
    let mut komi_specified = false;
    let mut tax_specified = false;
    // PORT NOTE: the try body becomes a closure returning Result (PORTING.md §7).
    let try_result: Result<(), KataError> = (|| {
      let input: Value = nlohmann::parse(s_orig)?;
      #[expect(unused_variables, reason = "mirrors the unused `string s;` in the JSON branch of upstream parseRulesHelper (rules.cpp)")]
      let s: String = String::new();
      // PORT NOTE: nlohmann's object is a std::map: iteration is in sorted key order, same as
      // serde_json's Map without preserve_order. A non-object top level cannot start with '{'.
      // TODO(port): iterating a non-object json (array/scalar) visits its values with key()
      // throwing invalid_iterator; unreachable here because the string starts with '{'.
      let object = match &input {
        Value::Object(m) => m,
        _ => return Err(KataError::NlohmannJson(String::from("[json.exception.invalid_iterator.207] cannot use key() for non-object iterators"))),
      };
      for (key, value) in object.iter() {
        let key: &str = key.as_str();
        if key == "ko" {
          rules.ko_rule = Rules::parse_ko_rule(&nlohmann::get_string(value)?)?;
        } else if key == "score" {
          rules.scoring_rule = Rules::parse_scoring_rule(&nlohmann::get_string(value)?)?;
        } else if key == "scoring" {
          rules.scoring_rule = Rules::parse_scoring_rule(&nlohmann::get_string(value)?)?;
        } else if key == "tax" {
          rules.tax_rule = Rules::parse_tax_rule(&nlohmann::get_string(value)?)?;
          tax_specified = true;
        } else if key == "suicide" {
          rules.multi_stone_suicide_legal = nlohmann::get_bool(value)?;
        } else if key == "hasButton" {
          rules.has_button = nlohmann::get_bool(value)?;
        } else if key == "whiteHandicapBonus" {
          rules.white_handicap_bonus_rule = Rules::parse_white_handicap_bonus_rule(&nlohmann::get_string(value)?)?;
        } else if key == "friendlyPassOk" {
          rules.friendly_pass_ok = nlohmann::get_bool(value)?;
        } else if key == "komi" {
          if !allow_komi {
            return Err(KataError::IOError(String::from("Unknown rules option: ") + key));
          }
          rules.komi = nlohmann::get_f32(value)?;
          komi_specified = true;
          if rules.komi < Rules::MIN_USER_KOMI || rules.komi > Rules::MAX_USER_KOMI || !Rules::komi_is_int_or_half_int(rules.komi) {
            return Err(KataError::IOError(String::from("Komi value is not a half-integer or is too extreme")));
          }
        } else {
          return Err(KataError::IOError(String::from("Unknown rules option: ") + key));
        }
      }
      Ok(())
    })();
    match try_result {
      Ok(()) => {}
      Err(KataError::NlohmannJson(_)) => {
        return Err(KataError::IOError(String::from("Could not parse rules: ") + s_orig));
      }
      Err(e) => return Err(e),
    }
    if !tax_specified {
      rules.tax_rule = if rules.scoring_rule == Rules::SCORING_TERRITORY { Rules::TAX_SEKI } else { Rules::TAX_NONE };
    }
    if !komi_specified {
      //Drop default komi to 6.5 for territory rules, and to 7.0 for button
      if rules.scoring_rule == Rules::SCORING_TERRITORY {
        rules.komi = 6.5f32;
      } else if rules.has_button {
        rules.komi = 7.0f32;
      }
    }
  }
  //This is more of a legacy internal format, not recommended for users to provide
  else {
    let starts_with_and_strip = |str: &mut String, prefix: &str| -> bool {
      let matches: bool = str.len() >= prefix.len() && str.as_bytes()[..prefix.len()] == *prefix.as_bytes();
      if matches {
        *str = str[prefix.len()..].to_string();
      }
      *str = Global::trim(str, Global::TRIM_DEFAULT_DELIM);
      matches
    };

    //Default if not specified
    rules = Rules::get_tromp_taylorish();

    let mut s: String = s_orig.to_string();
    s = Global::trim(&s, Global::TRIM_DEFAULT_DELIM);

    //But don't allow the empty string
    if s.is_empty() {
      return Err(KataError::IOError(String::from("Could not parse rules: ") + s_orig));
    }

    let mut komi_specified = false;
    let mut tax_specified = false;
    loop {
      if s.is_empty() {
        break;
      }

      if starts_with_and_strip(&mut s, "komi") {
        if !allow_komi {
          return Err(KataError::IOError(String::from("Could not parse rules: ") + s_orig));
        }
        let mut end_idx: usize = 0;
        while end_idx < s.len() && !Global::is_alpha(s.as_bytes()[end_idx]) && !Global::is_whitespace(s.as_bytes()[end_idx]) {
          end_idx += 1;
        }
        let mut komi: f32 = 0.0;
        let suc = Global::try_string_to_float(&s[0..end_idx], &mut komi);
        if !suc {
          return Err(KataError::IOError(String::from("Could not parse rules: ") + s_orig));
        }
        if !komi.is_finite() || (komi as f64) > 1e5 || (komi as f64) < -1e5 {
          return Err(KataError::IOError(String::from("Could not parse rules: ") + s_orig));
        }
        rules.komi = komi;
        komi_specified = true;
        s = s[end_idx..].to_string();
        s = Global::trim(&s, Global::TRIM_DEFAULT_DELIM);
        continue;
      }
      if starts_with_and_strip(&mut s, "ko") {
        if starts_with_and_strip(&mut s, "SIMPLE") {
          rules.ko_rule = Rules::KO_SIMPLE;
        } else if starts_with_and_strip(&mut s, "POSITIONAL") {
          rules.ko_rule = Rules::KO_POSITIONAL;
        } else if starts_with_and_strip(&mut s, "SITUATIONAL") {
          rules.ko_rule = Rules::KO_SITUATIONAL;
        } else if starts_with_and_strip(&mut s, "SPIGHT") {
          rules.ko_rule = Rules::KO_SPIGHT;
        } else {
          return Err(KataError::IOError(String::from("Could not parse rules: ") + s_orig));
        }
        continue;
      }
      if starts_with_and_strip(&mut s, "scoring") {
        if starts_with_and_strip(&mut s, "AREA") {
          rules.scoring_rule = Rules::SCORING_AREA;
        } else if starts_with_and_strip(&mut s, "TERRITORY") {
          rules.scoring_rule = Rules::SCORING_TERRITORY;
        } else {
          return Err(KataError::IOError(String::from("Could not parse rules: ") + s_orig));
        }
        continue;
      }
      if starts_with_and_strip(&mut s, "score") {
        if starts_with_and_strip(&mut s, "AREA") {
          rules.scoring_rule = Rules::SCORING_AREA;
        } else if starts_with_and_strip(&mut s, "TERRITORY") {
          rules.scoring_rule = Rules::SCORING_TERRITORY;
        } else {
          return Err(KataError::IOError(String::from("Could not parse rules: ") + s_orig));
        }
        continue;
      }
      if starts_with_and_strip(&mut s, "tax") {
        if starts_with_and_strip(&mut s, "NONE") {
          rules.tax_rule = Rules::TAX_NONE;
          tax_specified = true;
        } else if starts_with_and_strip(&mut s, "SEKI") {
          rules.tax_rule = Rules::TAX_SEKI;
          tax_specified = true;
        } else if starts_with_and_strip(&mut s, "ALL") {
          rules.tax_rule = Rules::TAX_ALL;
          tax_specified = true;
        } else {
          return Err(KataError::IOError(String::from("Could not parse rules: ") + s_orig));
        }
        continue;
      }
      if starts_with_and_strip(&mut s, "sui") {
        if starts_with_and_strip(&mut s, "1") {
          rules.multi_stone_suicide_legal = true;
        } else if starts_with_and_strip(&mut s, "0") {
          rules.multi_stone_suicide_legal = false;
        } else {
          return Err(KataError::IOError(String::from("Could not parse rules: ") + s_orig));
        }
        continue;
      }
      if starts_with_and_strip(&mut s, "button") {
        if starts_with_and_strip(&mut s, "1") {
          rules.has_button = true;
        } else if starts_with_and_strip(&mut s, "0") {
          rules.has_button = false;
        } else {
          return Err(KataError::IOError(String::from("Could not parse rules: ") + s_orig));
        }
        continue;
      }
      if starts_with_and_strip(&mut s, "whb") {
        if starts_with_and_strip(&mut s, "0") {
          rules.white_handicap_bonus_rule = Rules::WHB_ZERO;
        } else if starts_with_and_strip(&mut s, "N-1") {
          rules.white_handicap_bonus_rule = Rules::WHB_N_MINUS_ONE;
        } else if starts_with_and_strip(&mut s, "N") {
          rules.white_handicap_bonus_rule = Rules::WHB_N;
        } else {
          return Err(KataError::IOError(String::from("Could not parse rules: ") + s_orig));
        }
        continue;
      }
      if starts_with_and_strip(&mut s, "fpok") {
        if starts_with_and_strip(&mut s, "1") {
          rules.friendly_pass_ok = true;
        } else if starts_with_and_strip(&mut s, "0") {
          rules.friendly_pass_ok = false;
        } else {
          return Err(KataError::IOError(String::from("Could not parse rules: ") + s_orig));
        }
        continue;
      }
      //Unknown rules format
      else {
        return Err(KataError::IOError(String::from("Could not parse rules: ") + s_orig));
      }
    }
    if !tax_specified {
      rules.tax_rule = if rules.scoring_rule == Rules::SCORING_TERRITORY { Rules::TAX_SEKI } else { Rules::TAX_NONE };
    }
    if !komi_specified {
      //Drop default komi to 6.5 for territory rules, and to 7.0 for button
      if rules.scoring_rule == Rules::SCORING_TERRITORY {
        rules.komi = 6.5f32;
      } else if rules.has_button {
        rules.komi = 7.0f32;
      }
    }
  }

  Ok(rules)
}
