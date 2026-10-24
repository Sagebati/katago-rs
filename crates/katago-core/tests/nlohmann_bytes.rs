//! port::nlohmann over raw bytes (a std::string holds any bytes): `parse_bytes` keeps the C++ bytes of
//! `e.what()`, and `dump_string_bytes` throws dump()'s type_error.316 for a string that is not UTF-8. The
//! indices and bytes are the ones the reference KataGo reports when analysis.cpp's reportError dumps such a
//! text (fixtures/upstream-v1.16.5/analysis-protocol/expected/bytes_cfg_args.jsonl: bytes_*, multibyte_*).
//! Not a port of upstream code.

use katago_core::core::global::KataError;
use katago_core::port::nlohmann;

fn type_error(s: &[u8]) -> String {
  match nlohmann::dump_string_bytes(s) {
    Err(KataError::NlohmannJson(m)) => m,
    other => panic!("{s:?}: expected type_error.316, got {other:?}"),
  }
}

/// reportError's text for an unparseable line: `e.what() + " - could not parse input line as json request: " + line`.
fn report_error_text(line: &[u8]) -> Vec<u8> {
  let e = nlohmann::parse_bytes(line).expect_err("does not parse");
  let mut s = e.what().to_vec();
  s.extend_from_slice(b" - could not parse input line as json request: ");
  s.extend_from_slice(line);
  s
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn dump_string_bytes_dfa() {
  assert_eq!(nlohmann::dump_string_bytes(b"a\"b\\\x01\n\xc3\xa9\xf0\x9f\x98\x80").ok(), Some(b"\"a\\\"b\\\\\\u0001\\n\xc3\xa9\xf0\x9f\x98\x80\"".to_vec()));
  // the same as dump() of the valid string
  let v = serde_json::Value::from("a\"b\\\u{1}\né😀");
  assert_eq!(nlohmann::dump(&v).into_bytes(), nlohmann::dump_string_bytes("a\"b\\\u{1}\né😀".as_bytes()).unwrap());
  assert_eq!(type_error(b"a\xffb"), "[json.exception.type_error.316] invalid UTF-8 byte at index 1: 0xFF");
  assert_eq!(type_error(b"\xc0\xaf"), "[json.exception.type_error.316] invalid UTF-8 byte at index 0: 0xC0"); // overlong
  assert_eq!(type_error(b"x\xed\xa0\x80"), "[json.exception.type_error.316] invalid UTF-8 byte at index 2: 0xA0"); // CESU surrogate
  assert_eq!(type_error(b"\xe2\x82\""), "[json.exception.type_error.316] invalid UTF-8 byte at index 2: 0x22"); // truncated
  assert_eq!(type_error(b"\xf5\x80"), "[json.exception.type_error.316] invalid UTF-8 byte at index 0: 0xF5");
  assert_eq!(type_error(b"a\xe2\x82"), "[json.exception.type_error.316] incomplete UTF-8 string; last byte: 0x82");
}

/// The upstream-observed failures: the first invalid byte of reportError's text, as the reference binary's
/// `ERROR: request loop loop thread failed: [json.exception.type_error.316] ...` line reports it.
#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn report_error_texts_like_upstream() {
  let cases: &[(&[u8], &str)] = &[
    (b"{\"id\":\"a\xffb\",\"action\":\"query_version\"}", "invalid UTF-8 byte at index 154: 0xFF"),
    (b"\xff{\"id\":\"a\",\"action\":\"query_version\"}", "invalid UTF-8 byte at index 130: 0xFF"),
    (b"{\"id\":\"a\xe2\x82\",\"action\":\"query_version\"}", "invalid UTF-8 byte at index 157: 0x22"),
    (b"{\"id\":\"\xc0\xaf\",\"action\":\"query_version\"}", "invalid UTF-8 byte at index 153: 0xC0"),
    (b"{\"id\":\"\xed\xa0\x80\",\"action\":\"query_version\"}", "invalid UTF-8 byte at index 154: 0xA0"),
    (b"\xc2\xa0{\"id\":\"a\",\"action\":\"query_version\"}", "invalid UTF-8 byte at index 131: 0x27"),
    (b"{\"id\":\xc3\xa9}", "invalid UTF-8 byte at index 136: 0x27"),
    (b"{\"id\":\"a\",\xc3\xa9:1}", "invalid UTF-8 byte at index 141: 0x27"),
    (b"{\"id\":\"a\"}\xc3\xa9", "invalid UTF-8 byte at index 136: 0x27"),
    (b"\xc3\xa9", "invalid UTF-8 byte at index 131: 0x27"),
    (b"\xf0\x9f\x98\x80", "invalid UTF-8 byte at index 131: 0x27"),
    (b"[1,\xc3\xa9]", "invalid UTF-8 byte at index 133: 0x27"),
    (b"{\"\xff\":1}", "invalid UTF-8 byte at index 158: 0xFF"),
    (b"\x80", "invalid UTF-8 byte at index 130: 0x80"),
  ];
  for (line, what) in cases {
    assert_eq!(type_error(&report_error_text(line)), format!("[json.exception.type_error.316] {what}"), "{line:?}");
  }
  // Valid UTF-8 throughout (upstream answers with an error line): the bytes are the C++ what().
  let t = report_error_text("{\"é\"}".as_bytes());
  assert_eq!(
    String::from_utf8(nlohmann::dump_string_bytes(&t).unwrap()).unwrap(),
    "\"[json.exception.parse_error.101] parse error at line 1, column 6: syntax error while parsing object separator - unexpected '}'; expected ':' - could not parse input line as json request: {\\\"é\\\"}\""
  );
  // A NUL outside a string ends the input: the line parses.
  assert!(nlohmann::parse_bytes(b"{\"id\":\"a\"}\x00\xff").is_ok());
  // parse (&str) is parse_bytes with the what() lossily converted.
  assert!(nlohmann::parse("{\"id\":é}").err().unwrap().what().ends_with("last read: '\"id\":\u{FFFD}'"));
  assert!(nlohmann::parse_bytes("{\"id\":é}".as_bytes()).err().unwrap().what().ends_with(b"last read: '\"id\":\xc3'"));
}
