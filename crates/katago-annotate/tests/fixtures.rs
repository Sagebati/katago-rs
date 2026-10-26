//! Parity with katago-ws: fixtures/katago-annotate holds games run through katago-ws's own KataGo client
//! (reference KataGo v1.16.5, Eigen, g170e-b10c128): the query lines katago-ws wrote, the response lines KataGo
//! answered and the payload katago-ws's annotator built from them (fixtures/katago-annotate/README.md). Here
//! the same SGF goes through katago-annotate with a replay of those responses: every query line and the
//! payload must be byte-identical. Native and wasm32 (`wasm-bindgen-test-runner`, node).

use std::cell::Cell;
use std::collections::BTreeMap;
use std::io::Read as _;

use katago_annotate::{
    AnalysisQuery, Analyzer, AnnotateOptions, AnnotateResult, Fed, Pass, TurnCollector,
    TurnResponse,
};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test as test;

struct Case {
    name: &'static str,
    sgf: &'static str,
    case: &'static str,
    queries: &'static str,
    responses_gz: &'static [u8],
    expected_gz: &'static [u8],
}

macro_rules! case {
    ($name:literal) => {
        Case {
            name: $name,
            sgf: include_str!(concat!(
                "../../../fixtures/katago-annotate/",
                $name,
                "/game.sgf"
            )),
            case: include_str!(concat!(
                "../../../fixtures/katago-annotate/",
                $name,
                "/case.json"
            )),
            queries: include_str!(concat!(
                "../../../fixtures/katago-annotate/",
                $name,
                "/queries.jsonl"
            )),
            responses_gz: include_bytes!(concat!(
                "../../../fixtures/katago-annotate/",
                $name,
                "/responses.jsonl.gz"
            )),
            expected_gz: include_bytes!(concat!(
                "../../../fixtures/katago-annotate/",
                $name,
                "/expected.json.gz"
            )),
        }
    };
}

fn cases() -> Vec<Case> {
    vec![
        case!("hand-9x9"),
        case!("hand-13x13-h2"),
        case!("ogs-60000000"),
        case!("ogs-60023172-100"),
        case!("ogs-60046344-twopass"),
    ]
}

fn gunzip(bytes: &[u8]) -> String {
    let mut out = String::new();
    flate2::read::GzDecoder::new(bytes)
        .read_to_string(&mut out)
        .unwrap();
    out
}

mod support;
use support::block_on;

/// The engine of a recorded case: checks each query against the recorded line (ids numbered as katago-ws's
/// KataGo client numbers them: q0, q1, ...) and answers with the recorded response lines of that id.
struct Replay {
    queries: Vec<String>,
    responses: BTreeMap<String, Vec<String>>,
    next: Cell<usize>,
    passes: std::cell::RefCell<Vec<Pass>>,
}

impl Analyzer for Replay {
    async fn analyze(
        &self,
        pass: Pass,
        mut query: AnalysisQuery,
    ) -> AnnotateResult<Vec<TurnResponse>> {
        let k = self.next.get();
        self.next.set(k + 1);
        self.passes.borrow_mut().push(pass);
        query.id = format!("q{k}");
        assert_eq!(
            serde_json::to_string(&query).unwrap(),
            self.queries[k],
            "query line {k}"
        );
        let lines = &self.responses[&query.id];
        let mut collector = TurnCollector::new(query.id.clone(), lines.len());
        for line in lines {
            let fed = collector.feed(line)?;
            assert!(
                matches!(fed, Fed::Turn),
                "not a turn of {}: {fed:?}",
                query.id
            );
        }
        Ok(collector.into_turns())
    }
}

fn replay(case: &Case) -> Replay {
    let mut responses: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for line in gunzip(case.responses_gz).lines() {
        let id = serde_json::from_str::<serde_json::Value>(line).unwrap()["id"]
            .as_str()
            .unwrap()
            .to_owned();
        responses.entry(id).or_default().push(line.to_owned());
    }
    Replay {
        queries: case.queries.lines().map(str::to_owned).collect(),
        responses,
        next: Cell::new(0),
        passes: Default::default(),
    }
}

/// Where two texts first differ, with some context.
fn first_difference(got: &str, expected: &str) -> String {
    let at = got
        .bytes()
        .zip(expected.bytes())
        .take_while(|(a, b)| a == b)
        .count();
    let window = |s: &str| {
        s.get(at.saturating_sub(80)..(at + 40).min(s.len()))
            .unwrap_or("")
            .to_owned()
    };
    format!(
        "byte {at} of {}/{}: got ...{}... expected ...{}...",
        got.len(),
        expected.len(),
        window(got),
        window(expected)
    )
}

fn options(case: &Case) -> (AnnotateOptions, Option<String>) {
    let json: serde_json::Value = serde_json::from_str(case.case).unwrap();
    (
        serde_json::from_value(json["options"].clone()).unwrap(),
        json["humanProfile"].as_str().map(str::to_owned),
    )
}

#[test]
fn payloads_are_byte_identical_to_katago_ws() {
    for case in cases() {
        let (opts, human_profile) = options(&case);
        let engine = replay(&case);
        let payload = block_on(katago_annotate::analyze(
            &engine,
            &opts,
            case.sgf,
            human_profile.as_deref(),
        ))
        .unwrap();
        let json = serde_json::to_string(&payload).unwrap();
        assert_eq!(
            engine.next.get(),
            engine.queries.len(),
            "{}: every recorded query was made",
            case.name
        );
        let expected = gunzip(case.expected_gz);
        assert!(
            json == expected,
            "{}: payload differs from katago-ws's: {}",
            case.name,
            first_difference(&json, &expected)
        );
        let passes = engine.passes.borrow();
        assert_eq!(
            *passes,
            if opts.two_pass() {
                vec![Pass::Screen, Pass::Refine]
            } else {
                vec![Pass::Full]
            },
            "{}",
            case.name
        );
    }
}

/// The browser names its engine and version; nothing else in the payload changes.
#[test]
fn the_engine_name_and_version_change_only_the_engine_object() {
    let case = &cases()[0];
    let (mut opts, human_profile) = options(case);
    opts.engine_name = "katago-rs-wasm".to_owned();
    opts.engine_version = Some("1.16.5".to_owned());
    let payload = block_on(katago_annotate::analyze(
        &replay(case),
        &opts,
        case.sgf,
        human_profile.as_deref(),
    ))
    .unwrap();
    // Through the text (to_value would widen the f32s differently from parsing them).
    let mut got: serde_json::Value =
        serde_json::from_str(&serde_json::to_string(&payload).unwrap()).unwrap();
    let mut expected: serde_json::Value = serde_json::from_str(&gunzip(case.expected_gz)).unwrap();
    let engine = got.as_object_mut().unwrap().remove("engine").unwrap();
    let mut expected_engine = expected.as_object_mut().unwrap().remove("engine").unwrap();
    assert_eq!(got, expected);
    assert_eq!(engine["name"], "katago-rs-wasm");
    assert_eq!(engine["version"], "1.16.5");
    expected_engine["name"] = "katago-rs-wasm".into();
    expected_engine["version"] = "1.16.5".into();
    assert_eq!(engine, expected_engine);
}

/// A game given as moves is the same game as its SGF, and gives the same payload.
#[test]
fn a_game_given_as_moves_gives_the_sgf_payload() {
    let case = &cases()[1];
    let (opts, human_profile) = options(case);
    let game =
        katago_annotate::sgf::parse(case.sgf, opts.default_board_size, opts.default_komi).unwrap();
    let input: katago_annotate::GameInput = serde_json::from_value(serde_json::json!({
        "boardSize": game.board_size, "komi": game.komi, "rules": game.rules, "result": game.result,
        "initialStones": game.initial_stones, "moves": game.moves,
    }))
    .unwrap();
    let from_moves = input.into_game(19, 7.5).unwrap();
    assert_eq!(from_moves, game);
    let payload = block_on(katago_annotate::analyze_game(
        &replay(case),
        &opts,
        &from_moves,
        human_profile.as_deref(),
    ))
    .unwrap();
    assert!(serde_json::to_string(&payload).unwrap() == gunzip(case.expected_gz));
}
