//! Replays recorded cases through katago-ws's annotator (unchanged) and through katago-annotate, and checks:
//! - both parse the SGF to the same game;
//! - katago-annotate builds byte-identical query lines (ids as katago-ws numbers them);
//! - katago-ws's flow (turns_to_refine) asks for the turns the recording asked for;
//! - both payloads (serde_json::to_string) are byte-identical to each other and to expected.json.
//!
//!   compare <case-dir>...

use std::cell::Cell;
use std::collections::BTreeMap;

use katago_annotate as ka;
use ws_parity::config::EngineConfig;
use ws_parity::engine::{annotate as ws_annotate, katago::TurnResponse as WsTurn, sgf as ws_sgf};

struct Replay<'a> {
    queries: &'a [String],
    by_id: &'a BTreeMap<String, Vec<String>>,
    next: Cell<usize>,
}

impl ka::Analyzer for Replay<'_> {
    async fn analyze(&self, _pass: ka::Pass, mut query: ka::AnalysisQuery) -> ka::AnnotateResult<Vec<ka::TurnResponse>> {
        let k = self.next.get();
        self.next.set(k + 1);
        query.id = format!("q{k}");
        let line = serde_json::to_string(&query).unwrap();
        assert_eq!(line, self.queries[k], "query {k} differs");
        let mut out = Vec::new();
        for l in &self.by_id[&query.id] {
            match ka::ResponseLine::parse(l) {
                ka::ResponseLine::Turn { id, turn } => {
                    assert_eq!(id, query.id);
                    out.push(*turn);
                }
                other => panic!("unexpected line {other:?}"),
            }
        }
        Ok(out)
    }
}

fn main() {
    let mut failures = 0;
    for dir in std::env::args().skip(1) {
        let dir = std::path::Path::new(&dir);
        let read = |f: &str| std::fs::read_to_string(dir.join(f)).unwrap();
        let case: serde_json::Value = serde_json::from_str(&read("case.json")).unwrap();
        let sgf_text = read("game.sgf");
        let queries: Vec<String> = read("queries.jsonl").lines().map(str::to_owned).collect();
        let responses: Vec<String> = read("responses.jsonl").lines().map(str::to_owned).collect();
        let expected = read("expected.json");
        let human_profile = case["humanProfile"].as_str();

        let mut by_id: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for l in &responses {
            let v: serde_json::Value = serde_json::from_str(l).unwrap();
            by_id.entry(v["id"].as_str().unwrap().to_owned()).or_default().push(l.clone());
        }

        // katago-ws (EngineConfig from the same options, snake_case).
        let o = &case["options"];
        let ws_cfg: EngineConfig = serde_json::from_value(serde_json::json!({
            "model": o["model"], "max_visits": o["maxVisits"], "analysis_pv_len": o["analysisPvLen"],
            "include_ownership": o["includeOwnership"], "include_ownership_stdev": o["includeOwnershipStdev"],
            "include_policy": o["includePolicy"], "human_model": o["humanModel"], "top_k": o["topK"],
            "default_komi": o["defaultKomi"], "default_board_size": o["defaultBoardSize"],
            "screen_visits": o["screenVisits"], "refine_winrate_delta": o["refineWinrateDelta"],
            "refine_score_delta": o["refineScoreDelta"],
        }))
        .unwrap();
        let ws_game = ws_sgf::parse(&sgf_text, ws_cfg.default_board_size, ws_cfg.default_komi).unwrap();
        let turns_of = |id: &str| -> Vec<WsTurn> { by_id[id].iter().map(|l| serde_json::from_str(l).unwrap()).collect() };
        let ws_human = human_profile.filter(|_| !ws_cfg.human_model.is_empty());
        let turns = if ws_cfg.two_pass() {
            let screened = turns_of("q0");
            let refine: Vec<usize> = ws_annotate::turns_to_refine(&ws_game, &screened, &ws_cfg).into_iter().collect();
            let asked: Vec<usize> = if queries.len() > 1 {
                serde_json::from_str::<serde_json::Value>(&queries[1]).unwrap()["analyzeTurns"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|t| t.as_u64().unwrap() as usize)
                    .collect()
            } else {
                Vec::new()
            };
            assert_eq!(refine, asked, "{}: refined turns", dir.display());
            let refined = if refine.is_empty() { Vec::new() } else { turns_of("q1") };
            ws_parity::merge_turns(screened, refined)
        } else {
            turns_of("q0")
        };
        let ws_payload = serde_json::to_string(&ws_annotate::assemble(&ws_game, turns, &ws_cfg, ws_human)).unwrap();

        // katago-annotate.
        let ka_cfg: ka::AnnotateOptions = serde_json::from_value(o.clone()).unwrap();
        let ka_game = ka::sgf::parse(&sgf_text, ka_cfg.default_board_size, ka_cfg.default_komi).unwrap();
        let same_game = ka_game.board_size == ws_game.board_size
            && ka_game.komi.to_bits() == ws_game.komi.to_bits()
            && serde_json::to_string(&ka_game.rules).unwrap() == serde_json::to_string(&ws_game.rules).unwrap()
            && ka_game.result == ws_game.result
            && serde_json::to_string(&ka_game.initial_stones).unwrap() == serde_json::to_string(&ws_game.initial_stones).unwrap()
            && serde_json::to_string(&ka_game.moves).unwrap() == serde_json::to_string(&ws_game.moves).unwrap();
        let replay = Replay { queries: &queries, by_id: &by_id, next: Cell::new(0) };
        let ka_payload =
            serde_json::to_string(&futures::executor::block_on(ka::analyze(&replay, &ka_cfg, &sgf_text, human_profile)).unwrap()).unwrap();
        let queries_used = replay.next.get();

        let ok = same_game && queries_used == queries.len() && ws_payload == expected && ka_payload == expected;
        println!(
            "{}: {} moves, {} queries, {} responses, payload {} bytes: game {} | katago-ws replay == expected: {} | katago-annotate == expected: {}",
            dir.display(),
            ws_game.moves.len(),
            queries.len(),
            responses.len(),
            expected.len(),
            if same_game { "same" } else { "DIFFERENT" },
            ws_payload == expected,
            ka_payload == expected,
        );
        if !ok {
            failures += 1;
        }
    }
    if failures > 0 {
        eprintln!("{failures} case(s) FAILED");
        std::process::exit(1);
    }
}
