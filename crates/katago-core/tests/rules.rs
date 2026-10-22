//! game/rules.
//! 1. Rules cases from upstream cpp/tests/testrules.cpp that need no BoardHistory:
//!    "Test some roundtripping of rules strings" and "Rules parsing bug".
//! 2. Strings and JSON against the real C++ (fixtures/upstream-v1.16.5/unit/oracle/rules.txt).
//!    Mirrors `rulesMode()` in oracle.cpp.

mod common;

use common::Cases;
use common::fixture;
use katago_core::core::test::TestCommon;
use katago_core::game::rules::Rules;
use katago_core::port::fmt::cpp_ostream_f64;
use katago_core::testAssert;

fn rules8(k: i32, s: i32, t: i32, suic: bool, button: bool, whb: i32, pok: bool, komi: f32) -> Rules {
  Rules::new_full(
    k, s, t, suic, button, whb, pok, komi,
  )
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn rules_roundtrip_and_parsing_bug() {
  let mut cases = Cases::new("rules::rules_roundtrip_and_parsing_bug");
  {
    //const char* name = "Test some roundtripping of rules strings";
    let rules: Vec<Rules> = vec![
      rules8(Rules::KO_SIMPLE, Rules::SCORING_AREA, Rules::TAX_NONE, false, false, Rules::WHB_ZERO, false, 1.0f32),
      rules8(Rules::KO_POSITIONAL, Rules::SCORING_AREA, Rules::TAX_NONE, false, true, Rules::WHB_ZERO, true, 3.0f32),
      rules8(Rules::KO_SITUATIONAL, Rules::SCORING_AREA, Rules::TAX_NONE, true, false, Rules::WHB_ZERO, false, 5.5f32),
      rules8(Rules::KO_SIMPLE, Rules::SCORING_AREA, Rules::TAX_SEKI, false, false, Rules::WHB_ZERO, false, 1.0f32),
      rules8(Rules::KO_POSITIONAL, Rules::SCORING_AREA, Rules::TAX_SEKI, true, false, Rules::WHB_N, true, 4.5f32),
      rules8(Rules::KO_SITUATIONAL, Rules::SCORING_TERRITORY, Rules::TAX_NONE, true, true, Rules::WHB_N, true, 6.5f32),
      rules8(Rules::KO_SIMPLE, Rules::SCORING_TERRITORY, Rules::TAX_SEKI, false, true, Rules::WHB_ZERO, false, 2.0f32),
      rules8(Rules::KO_SITUATIONAL, Rules::SCORING_TERRITORY, Rules::TAX_SEKI, true, true, Rules::WHB_ZERO, true, 5.5f32),
      rules8(Rules::KO_SIMPLE, Rules::SCORING_AREA, Rules::TAX_ALL, true, false, Rules::WHB_N_MINUS_ONE, false, 2.5f32),
      rules8(Rules::KO_POSITIONAL, Rules::SCORING_AREA, Rules::TAX_ALL, false, true, Rules::WHB_ZERO, true, 3.0f32),
      rules8(Rules::KO_POSITIONAL, Rules::SCORING_TERRITORY, Rules::TAX_ALL, false, true, Rules::WHB_N_MINUS_ONE, false, 4.0f32),
      rules8(Rules::KO_SITUATIONAL, Rules::SCORING_TERRITORY, Rules::TAX_ALL, true, false, Rules::WHB_N, false, 6.5f32),
    ];

    for i in 0..rules.len() {
      let mut suc: bool;

      let mut parsed = Rules::new();
      suc = Rules::try_parse_rules(&rules[i].to_string(), &mut parsed).unwrap();
      testAssert!(suc);
      testAssert!(rules[i] == parsed);

      let mut parsed2 = Rules::new();
      suc = Rules::try_parse_rules_without_komi(&rules[i].to_string_no_komi(), &mut parsed2, rules[i].komi).unwrap();
      testAssert!(suc);
      testAssert!(rules[i] == parsed2);

      let mut parsed3 = Rules::new();
      suc = Rules::try_parse_rules(&rules[i].to_json_string(), &mut parsed3).unwrap();
      testAssert!(suc);
      testAssert!(rules[i] == parsed3);

      let mut parsed4 = Rules::new();
      suc = Rules::try_parse_rules_without_komi(&rules[i].to_json_string_no_komi(), &mut parsed4, rules[i].komi).unwrap();
      testAssert!(suc);
      testAssert!(rules[i] == parsed4);

      let mut parsed5 = Rules::new();
      suc = Rules::try_parse_rules_without_komi(&rules[i].to_json_string_no_komi_maybe_omit_stuff(), &mut parsed5, rules[i].komi).unwrap();
      testAssert!(suc);
      testAssert!(rules[i] == parsed5);
      cases.ran();
    }
  }

  {
    let name = "Rules parsing bug";
    let mut out = String::new();
    let parsed = Rules::parse_rules("komi23taxALL").unwrap();
    out += &format!("{}\n", parsed);
    let expected = fixture("testrules", "rules-parsing-bug.expected0.txt");
    TestCommon::expect_string(name, &mut out, &expected);
    cases.ran();
  }
}

fn print_rules(out: &mut String, r: &Rules) {
  *out += &format!("  toString {}\n", r.to_string());
  *out += &format!("  toStringNoKomi {}\n", r.to_string_no_komi());
  *out += &format!("  toJsonString {}\n", r.to_json_string());
  *out += &format!("  toJsonStringNoKomi {}\n", r.to_json_string_no_komi());
  *out += &format!("  toJsonStringNoKomiMaybeOmitStuff {}\n", r.to_json_string_no_komi_maybe_omit_stuff());
  *out += &format!("  toStringNoKomiMaybeNice {}\n", r.to_string_no_komi_maybe_nice());
  *out += &format!("  gameResultWillBeInteger {}\n", r.game_result_will_be_integer() as i32);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn rules_vs_cpp() {
  let mut cases = Cases::new("rules::rules_vs_cpp");
  let expected = fixture("oracle", "rules.txt");
  let mut out = String::new();
  let inputs: Vec<&str> = vec![
    "japanese", "Korean", "chinese", " Chinese-OGS ", "chinese_kgs", "ancientarea", "stone scoring", "ancient territory",
    "aga-button", "AGA", "bga", "french", "nz", "new zealand", "tromp-taylor", "TrompTaylor", "goe", "ing", "", "   ", "unknown",
    "koSIMPLEscoreAREAtaxNONEsui0komi7.5", "komi23taxALL", "koPOSITIONALscoringTERRITORYsui1button1whbN-1fpok1komi-3",
    "kosimple", "komi1e9", "komi7.5.5", "komi 7 koSPIGHT", "whbN", "taxSEKIscoreAREA", "button1", "komi+2.5", "komi.5", "komi-0",
    "{}", "{\"ko\":\"SIMPLE\",\"scoring\":\"TERRITORY\"}", "{\"komi\":6}", "{\"komi\":true}", "{\"komi\":\"6\"}",
    "{\"komi\":7.25}", "{\"komi\":200}", "{\"komi\":-150}", "{\"suicide\":1}", "{\"ko\":3}", "{\"hasButton\":true}",
    "{\"whiteHandicapBonus\":\"N-1\",\"friendlyPassOk\":true,\"tax\":\"ALL\"}", "{\"foo\":1}", "{bad json",
    "{\"ko\":\"SIMPLE\"} trailing", "{\"ko\":\"SIMPLE\",\"ko\":\"SPIGHT\"}", "{\"score\":\"AREA\",\"ko\":\"BOGUS\"}",
    "{\"komi\":1e2}", "{\"komi\":0.5,\"scoring\":\"TERRITORY\"}",
  ];
  for s in &inputs {
    out += &format!("INPUT [{}]\n", s);
    match Rules::parse_rules(s) {
      Ok(r) => {
        out += " parseRules ok\n";
        print_rules(&mut out, &r);
      }
      Err(e) if e.is_string_error() => out += &format!(" parseRules ERR {}\n", e.what()),
      Err(e) => panic!("{}", e),
    }
    match Rules::parse_rules_without_komi(s, 6.0f32) {
      Ok(r) => out += &format!(" parseRulesWithoutKomi ok {}\n", r.to_string()),
      Err(e) if e.is_string_error() => out += &format!(" parseRulesWithoutKomi ERR {}\n", e.what()),
      Err(e) => panic!("{}", e),
    }
    let mut buf = Rules::new();
    let suc = Rules::try_parse_rules(s, &mut buf).unwrap();
    out += &format!(" tryParseRules {} {}\n", suc as i32, buf.to_string());
    cases.ran();
  }
  let updates: Vec<(&str, &str)> = vec![
    ("ko", "simple"),
    (" score ", "territory"),
    ("scoring", "AREA"),
    ("tax", "all"),
    ("suicide", "false"),
    ("suicide", "TRUE"),
    ("suicide", "yes"),
    ("hasButton", "true"),
    ("whiteHandicapBonus", "n-1"),
    ("friendlyPassOk", "true"),
    ("komi", "7"),
    ("ko", "bad"),
  ];
  let mut cur = Rules::get_tromp_taylorish();
  for (k, v) in updates {
    match Rules::update_rules(k, v, &cur) {
      Ok(r) => {
        cur = r;
        out += &format!("updateRules {}={} -> {}\n", k, v, cur.to_string());
      }
      Err(e) if e.is_string_error() => out += &format!("updateRules {}={} ERR {}\n", k, v, e.what()),
      Err(e) => panic!("{}", e),
    }
    cases.ran();
  }
  let komis: [f32; 10] = [0.0, 0.5, 7.5, -7.5, 0.25, 1e10, -150.0, 3.3, f32::INFINITY, f32::NAN];
  for k in komis {
    out += &format!("komiIsIntOrHalfInt {} {}\n", cpp_ostream_f64(k as f64), Rules::komi_is_int_or_half_int(k) as i32);
    cases.ran();
  }
  let rules: Vec<Rules> = vec![
    rules8(Rules::KO_SIMPLE, Rules::SCORING_AREA, Rules::TAX_NONE, false, false, Rules::WHB_ZERO, false, 1.0f32),
    rules8(Rules::KO_POSITIONAL, Rules::SCORING_AREA, Rules::TAX_NONE, false, true, Rules::WHB_ZERO, true, 3.0f32),
    rules8(Rules::KO_SITUATIONAL, Rules::SCORING_AREA, Rules::TAX_NONE, true, false, Rules::WHB_ZERO, false, 5.5f32),
    rules8(Rules::KO_SPIGHT, Rules::SCORING_TERRITORY, Rules::TAX_ALL, true, true, Rules::WHB_N_MINUS_ONE, true, -2.5f32),
    rules8(Rules::KO_SIMPLE, Rules::SCORING_AREA, Rules::TAX_SEKI, false, false, Rules::WHB_N, false, 0.1f32),
  ];
  for r in &rules {
    out += "RULES\n";
    print_rules(&mut out, r);
    cases.ran();
  }
  out += &format!("simpleTerritory {}\n", Rules::get_simple_territory().to_string());
  common::compare_lines("oracle/rules.txt", &out, &expected);
}

/// `(int)komi` / `(int)(komi * 2)` out of range: C++ UB, x86-64 cvttss2si gives INT_MIN.
/// Expected values from the real C++ (Rules::komiIsIntOrHalfInt / gameResultWillBeInteger built
/// from upstream rules.cpp with g++ 16.2.1 -O1 on x86-64): komi, intOrHalfInt,
/// willBeInteger(no button), willBeInteger(button).
#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn rules_int_cast_out_of_range_vs_cpp() {
  let cases_table: [(f32, bool, bool, bool); 9] = [
    (1073741824.0, false, true, false),
    (2147483648.0, false, false, true),
    (-2147483648.0, false, true, false),
    (3e9, false, false, true),
    (-3e9, false, false, true),
    (f32::NAN, false, false, true),
    (f32::INFINITY, false, false, true),
    (7.5, true, false, true),
    (-1073741824.0, true, true, false),
  ];
  let mut cases = Cases::new("rules_int_cast_out_of_range_vs_cpp");
  for (komi, int_or_half, will_be_int, will_be_int_button) in cases_table {
    let mut r = Rules::get_tromp_taylorish();
    r.komi = komi;
    assert_eq!(Rules::komi_is_int_or_half_int(komi), int_or_half, "komiIsIntOrHalfInt({})", komi);
    assert_eq!(r.game_result_will_be_integer(), will_be_int, "gameResultWillBeInteger({}, no button)", komi);
    r.has_button = true;
    assert_eq!(r.game_result_will_be_integer(), will_be_int_button, "gameResultWillBeInteger({}, button)", komi);
    cases.ran();
  }
}

/// `e.what()` is `message.c_str()`: a message built from input with a NUL stops there in C++.
/// Expected texts from the real C++ (Rules::parseRules / Hash128::ofString, review harness).
#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn kata_error_what_stops_at_nul_vs_cpp() {
  let mut cases = Cases::new("kata_error_what_stops_at_nul_vs_cpp");
  let e = Rules::parse_rules("Tromp\0Taylor").expect_err("invalid rules");
  assert_eq!(e.what(), "Could not parse rules: Tromp");
  assert_eq!(e.to_string(), "Could not parse rules: Tromp");
  cases.ran();
  let e = Rules::parse_rules("ko\0SIMPLE").expect_err("invalid rules");
  assert_eq!(e.what(), "Could not parse rules: ko");
  cases.ran();
  let e = katago_core::core::hash::Hash128::of_string("AB\0CD").expect_err("invalid hash");
  assert_eq!(e.what(), "Could not parse as Hash128: AB");
  cases.ran();
}
