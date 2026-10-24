//! Protocol checks of the ported analysis engine through the library (the same host pieces as the binary).
//! The texts are the literal strings of cpp/command/analysis.cpp; the whole protocol surface is also compared
//! with the reference binary by tests/analysis_protocol_corpus.rs. Here: two values that are Rust choices
//! (query_version's git_hash and query_models' maxBatchSize, normalized in the corpus test), the burst order
//! and terminate outcomes that upstream shows in most runs, the invalid-UTF-8 termination path, and
//! reportDuringSearchEvery (partial responses from AsyncBot's callback loop through the injected sleep).
//! Not a port of upstream code.

mod common;

use common::*;

fn q(s: &str) -> String {
  s.to_string()
}

#[test]
fn protocol_actions_and_errors() {
  let lines = vec![
    q(r#"{"id":"v","action":"query_version"}"#),
    q(r#"{"id":"m","action":"query_models"}"#),
    q(r#"{"id":"x","action":"nope"}"#),
    q(r#"not json"#),
    q(r#"[1,2]"#),
    q(r#"{"action":"query_version"}"#),
    q(r#"{"id":"t","action":"terminate"}"#),
    q(r#"{"id":"t2","action":"terminate_all","turnNumbers":"bad"}"#),
    q(r#"{"id":"b1","moves":[],"rules":"tromp-taylor","boardXSize":1,"boardYSize":19}"#),
    q(r#"{"id":"b2","moves":[],"rules":"tromp-taylor","boardYSize":19}"#),
    q(r#"{"id":"b3","moves":[["B","Z99"]],"rules":"tromp-taylor","boardXSize":9,"boardYSize":9}"#),
    q(r#"{"id":"b4","moves":[["Q","C3"]],"rules":"tromp-taylor","boardXSize":9,"boardYSize":9}"#),
    q(r#"{"id":"b5","moves":[],"rules":"nonsense","boardXSize":9,"boardYSize":9}"#),
    q(r#"{"id":"b6","moves":[],"rules":"tromp-taylor","komi":7.25,"boardXSize":9,"boardYSize":9}"#),
    q(r#"{"id":"b7","moves":[],"rules":"tromp-taylor","boardXSize":9,"boardYSize":9,"analyzeTurns":[1]}"#),
    q(r#"{"id":"b8","moves":[["B","C3"],["W","C3"]],"rules":"tromp-taylor","boardXSize":9,"boardYSize":9}"#),
    q(r#"{"id":"b9","moves":[],"rules":"tromp-taylor","boardXSize":9,"boardYSize":9,"overrideSettings":{"numBots":3}}"#),
    q(r#"{"id":"b10","moves":[],"rules":"tromp-taylor","boardXSize":9,"boardYSize":9,"maxVisits":0}"#),
    q(r#"{"id":"b11","moves":[],"rules":"tromp-taylor","boardXSize":9,"boardYSize":9,"analyzeTurns":[0],"priority":1,"priorities":[1]}"#),
    q(r#"{"id":"b12","moves":[],"rules":"tromp-taylor","boardXSize":9,"boardYSize":9,"allowMoves":[{"player":"B","moves":["C3"],"untilDepth":1}],"avoidMoves":[]}"#),
  ];
  let got = run_analysis_lines(lines, M6, "maxVisits=10");
  // Rust choices (not upstream-verified values): git_hash is [upstream].commit (GIT_REVISION; the reference
  // binary was built with NO_GIT_REVISION and prints "<omitted>"), and maxBatchSize is nnMaxBatchSize = 64 from
  // analysis_example.cfg, upstream's non-Eigen branch of Setup::initializeNNEvaluator (its Eigen build forces 2).
  let expected = vec![
    r#"{"action":"query_version","git_hash":"ba938676d7f42d70950b3a535af2466fb642008c","id":"v","version":"1.16.5"}"#,
    r#"{"action":"query_models","id":"m","models":[{"internalName":"g170-b6c96-s175395328-d26788732","maxBatchSize":64,"name":"tests/models/g170-b6c96-s175395328-d26788732.bin.gz","usesHumanSLProfile":false,"usingFP16":"auto","version":8}]}"#,
    r#"{"error":"'action' field must be 'query_version' or 'query_models' or 'clear_cache' or 'terminate' or 'terminate_all'"}"#,
    r#"{"error":"[json.exception.parse_error.101] parse error at line 1, column 2: syntax error while parsing value - invalid literal; last read: 'no' - could not parse input line as json request: not json"}"#,
    r#"{"error":"Request line was valid json but was not an object, ignoring: [1,2]"}"#,
    r#"{"error":"Request must have a string \"id\" field"}"#,
    r#"{"error":"Requests for a terminate action must have a string \"terminateId\" field","field":"terminateId","id":"t"}"#,
    r#"{"error":"If provided, must be an array of integers indicating turns to terminate","field":"turnNumbers","id":"t2"}"#,
    r#"{"error":"Must provide an integer from 2 to 19","field":"boardXSize","id":"b1"}"#,
    r#"{"error":"Must provide an integer from 2 to 19","field":"boardXSize","id":"b2"}"#,
    r#"{"error":"Could not parse board location: Z99","field":"moves","id":"b3"}"#,
    r#"{"error":"Could not parse player: Q","field":"moves","id":"b4"}"#,
    r#"{"error":"Could not parse rules: nonsense","field":"rules","id":"b5"}"#,
    r#"{"error":"Must be a integer or half-integer from -150.0 to 150.0","field":"komi","id":"b6"}"#,
    r#"{"error":"Invalid turn number: 1","field":"analyzeTurns","id":"b7"}"#,
    r#"{"error":"Illegal move 1: C3","field":"moves","id":"b8"}"#,
    r#"{"error":"Could not set settings: The config for this command cannot have numBots > 0","field":"overrideSettings","id":"b9"}"#,
    r#"{"error":"Must be an integer from 1 to 2^50","field":"maxVisits","id":"b10"}"#,
    r#"{"error":"Cannot specify both priority and priorities","field":"priority","id":"b11"}"#,
    r#"{"error":"Cannot specify both allowMoves and avoidMoves","field":"allowMoves","id":"b12"}"#,
  ];
  for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
    assert_eq!(g, e, "response {i}");
  }
  assert_eq!(got.len(), expected.len(), "{got:#?}");
  eprintln!("protocol_actions_and_errors: {} responses", got.len());
}

/// Upstream's dominant outcome (fixtures/upstream-v1.16.5/analysis-protocol order.jsonl term_queued_test, 6/10
/// runs of the Eigen build): the idle analysis thread pops "a" as soon as it is queued, the terminate finds it
/// popped but not yet searching (the AsyncBot's search thread is still waking up), so the action echo comes
/// first and the analysis loop then writes noResults for "a"; "c" is analyzed.
#[test]
fn terminate_popped_request_reports_no_results() {
  let lines = vec![
    q(r#"{"id":"a","moves":[],"rules":"tromp-taylor","boardXSize":9,"boardYSize":9,"analyzeTurns":[0]}"#),
    q(r#"{"id":"term","action":"terminate","terminateId":"a"}"#),
    q(r#"{"id":"c","moves":[],"rules":"tromp-taylor","boardXSize":9,"boardYSize":9,"priority":5}"#),
  ];
  let got = run_analysis_lines(lines, M6, "maxVisits=5");
  assert_eq!(got[0], r#"{"action":"terminate","id":"term","terminateId":"a"}"#);
  assert_eq!(got[1], r#"{"id":"a","isDuringSearch":false,"noResults":true,"turnNumber":0}"#);
  assert_eq!(got.len(), 3, "{got:#?}");
  assert!(got[2].starts_with(r#"{"id":"c","isDuringSearch":false,"moveInfos":"#), "{}", got[2]);
}

fn order_shape(lines: &[String]) -> String {
  lines
    .iter()
    .map(|l| {
      let v: serde_json::Value = serde_json::from_str(l).unwrap();
      let id = v["id"].as_str().unwrap_or("").to_string();
      if v.get("error").is_some() {
        format!("E({})", v["field"].as_str().unwrap_or(""))
      } else if v.get("noResults").is_some() {
        format!("N({id},t{})", v["turnNumber"])
      } else if v.get("action").is_some() {
        format!("A({})", v["action"].as_str().unwrap())
      } else {
        format!("R({id},t{})", v["turnNumber"])
      }
    })
    .collect::<Vec<_>>()
    .join(" ")
}

/// Burst ordering (all lines available at once): the outcomes upstream shows in most runs (order.jsonl and
/// the port-manifest notes: prio_burst 8/10, prio_neg_first 10/10, priorities_turns_after_busy 7/10,
/// multiturn_term_all 9/10, multiturn_termall_02 7/10, multiturn_b_termall_10 9/10).
#[test]
fn burst_order_matches_upstream_majority() {
  let qq = |id: &str, extra: &str| format!(r#"{{"id":"{id}","moves":[],"rules":"tromp-taylor","boardXSize":9,"boardYSize":9{extra}}}"#);
  let qa = r#"{"id":"a","moves":[["B","C3"],["W","G7"]],"rules":"tromp-taylor","boardXSize":9,"boardYSize":9,"analyzeTurns":[0,1,2],"maxVisits":5}"#.to_string();
  let cases: Vec<(Vec<String>, &str)> = vec![
    (
      vec![qq("a", r#","maxVisits":30"#), qq("b", r#","maxVisits":5,"priority":10"#), qq("c", r#","maxVisits":5,"priority":5"#), qq("d", r#","maxVisits":5"#)],
      "R(a,t0) R(b,t0) R(c,t0) R(d,t0)",
    ),
    (vec![qq("a", r#","maxVisits":5,"priority":-5"#), qq("b", r#","maxVisits":5,"priority":0"#)], "R(a,t0) R(b,t0)"),
    (
      vec![qq("z", r#","maxVisits":60"#), qa.replace(r#""maxVisits":5"#, r#""priorities":[1,3,2],"maxVisits":5"#)],
      "R(z,t0) R(a,t1) R(a,t2) R(a,t0)",
    ),
    (vec![qa.clone(), q(r#"{"id":"t","action":"terminate","terminateId":"a"}"#)], "N(a,t1) N(a,t2) A(terminate) N(a,t0)"),
    (vec![qa.clone(), q(r#"{"id":"t","action":"terminate_all","turnNumbers":[0,2]}"#)], "N(a,t2) A(terminate_all) N(a,t0) R(a,t1)"),
    (
      vec![qa.clone(), qq("b", r#","maxVisits":5"#), q(r#"{"id":"t","action":"terminate_all","turnNumbers":[1,0]}"#)],
      "N(a,t1) N(b,t0) A(terminate_all) N(a,t0) R(a,t2)",
    ),
  ];
  for (lines, expected) in cases {
    let got = run_analysis_lines(lines, M6, "maxVisits=5");
    assert_eq!(order_shape(&got), expected);
  }
}

/// A line whose parse error text is not valid UTF-8 (invalid input bytes, or a "last read" excerpt cut
/// inside a multi-byte character) makes reportError's dump() throw type_error.316 out of the request loop:
/// upstream logs it, keeps the other threads running for logThreadUncaught's 5 s (the queued query is
/// answered), then terminates. Texts from fixtures/upstream-v1.16.5/analysis-protocol (bytes_*).
#[test]
fn invalid_utf8_error_text_terminates_like_upstream() {
  let query = q(r#"{"id":"q","moves":[],"rules":"tromp-taylor","boardXSize":9,"boardYSize":9}"#).into_bytes();
  let version = q(r#"{"id":"v","action":"query_version"}"#).into_bytes();
  let (got, r) = run_analysis_bytes(vec![query, b"\xc0\xaf".to_vec(), version.clone()], M6, "maxVisits=8");
  match r {
    Err(katago_core::core::global::KataError::NlohmannJson(m)) => assert_eq!(m, "[json.exception.type_error.316] invalid UTF-8 byte at index 130: 0xC0"),
    other => panic!("{other:?}"),
  }
  assert_eq!(got.len(), 1, "{got:#?}");
  assert!(got[0].starts_with(r#"{"id":"q","isDuringSearch":false,"moveInfos":"#));

  let cases: Vec<(&[u8], &str)> = vec![
    (b"{\"id\":\"a\xffb\",\"action\":\"query_version\"}", "invalid UTF-8 byte at index 154: 0xFF"),
    (b"{\"id\":\"a\xe2\x82\",\"action\":\"query_version\"}", "invalid UTF-8 byte at index 157: 0x22"),
    (b"{\"id\":\"\xed\xa0\x80\",\"action\":\"query_version\"}", "invalid UTF-8 byte at index 154: 0xA0"),
    (b"\xc2\xa0{\"id\":\"a\",\"action\":\"query_version\"}", "invalid UTF-8 byte at index 131: 0x27"),
    (b"{\"id\":\"a\"} \xc3", "invalid UTF-8 byte at index 137: 0x27"),
    (b"{\"id\":\xc3\xa9}", "invalid UTF-8 byte at index 136: 0x27"),
  ];
  for (line, what) in cases {
    let (got, r) = run_analysis_bytes(vec![line.to_vec(), version.clone()], M6, "maxVisits=8");
    match r {
      Err(katago_core::core::global::KataError::NlohmannJson(m)) => assert_eq!(m, format!("[json.exception.type_error.316] {what}"), "{line:?}"),
      other => panic!("{line:?}: {other:?}"),
    }
    assert!(got.is_empty(), "{got:?}");
  }
  // Valid UTF-8 everywhere: an ordinary error line, and the session goes on.
  let got = run_analysis_lines(vec![q("{\"é\"}"), q(r#"{"id":"v","action":"query_version"}"#)], M6, "maxVisits=8");
  assert_eq!(got[0], "{\"error\":\"[json.exception.parse_error.101] parse error at line 1, column 6: syntax error while parsing object separator - unexpected '}'; expected ':' - could not parse input line as json request: {\\\"é\\\"}\"}");
  assert_eq!(got.len(), 2);
}

#[test]
fn report_during_search_every() {
  let lines = vec![q(
    r#"{"id":"r","moves":[["B","D4"]],"rules":"tromp-taylor","boardXSize":19,"boardYSize":19,"maxVisits":300,"reportDuringSearchEvery":0.02}"#,
  )];
  let got = run_analysis_lines(lines, M6, "maxVisits=300");
  let during = got.iter().filter(|l| l.contains(r#""isDuringSearch":true"#)).count();
  let last = got.last().expect("a final response");
  assert!(last.starts_with(r#"{"id":"r","isDuringSearch":false,"#), "{last}");
  assert_eq!(got.iter().filter(|l| l.contains(r#""isDuringSearch":false"#)).count(), 1);
  eprintln!("report_during_search_every: {during} partial responses before the final one");
  assert!(during >= 1, "no partial response during a 300-visit search with reportDuringSearchEvery 0.02 s");
}
