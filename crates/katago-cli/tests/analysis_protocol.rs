//! Rust-only protocol checks of the ported analysis engine (no upstream golden covers them): the texts are
//! the literal strings of cpp/command/analysis.cpp, so these are regression tests of the transcription,
//! plus behaviour checks of terminate (queued request -> noResults) and reportDuringSearchEvery (partial
//! responses from AsyncBot's callback loop, through the injected sleep). Not a port of upstream code.

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

#[test]
fn terminate_queued_request_reports_no_results() {
  let lines = vec![
    q(r#"{"id":"a","moves":[],"rules":"tromp-taylor","boardXSize":9,"boardYSize":9,"analyzeTurns":[0]}"#),
    q(r#"{"id":"term","action":"terminate","terminateId":"a"}"#),
    q(r#"{"id":"c","moves":[],"rules":"tromp-taylor","boardXSize":9,"boardYSize":9,"priority":5}"#),
  ];
  let got = run_analysis_lines(lines, M6, "maxVisits=5");
  // The request loop handles all three lines before the analysis loop runs (the input is all available):
  // "a" is still queued when terminated (noResults); "c" is analyzed.
  assert_eq!(got[0], r#"{"id":"a","isDuringSearch":false,"noResults":true,"turnNumber":0}"#);
  assert_eq!(got[1], r#"{"action":"terminate","id":"term","terminateId":"a"}"#);
  assert_eq!(got.len(), 3, "{got:#?}");
  assert!(got[2].starts_with(r#"{"id":"c","isDuringSearch":false,"moveInfos":"#), "{}", got[2]);
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
