//! Records one parity case through katago-ws's own KataGo client (engine::katago::KataGo, unchanged): the binary
//! is a tee wrapper around the reference KataGo, so the query lines katago-ws writes and the response lines
//! KataGo answers are saved verbatim. The game-level flow is katago-ws mod.rs `AnalysisEngine::analyze`
//! (copied below: mod.rs needs muxa), and the payload is katago-ws annotate::assemble serialized with
//! serde_json::to_string, as the worker does.
//!
//!   record <case.json> <out-dir>
//! case.json: {"sgf": path relative to the repository root, "humanProfile": null | "rank_5k",
//!             "engine": {katago-ws EngineConfig fields, snake_case}}
//! Env: KATAGO (default build/eigen/katago, `just katago-eigen`), KATAGO_MODELS (directory of the .bin.gz
//! nets, the working directory of KataGo: default ../katago-ws/bench/models), KATAGO_CONFIG (default upstream's
//! analysis_example.cfg, as katago-ws's image ships it).

use std::path::PathBuf;

use ws_parity::config::EngineConfig;
use ws_parity::engine::{annotate, katago::KataGo, sgf};

/// The katago-rs checkout.
const REPO: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");

fn path_var(var: &str, default: &str) -> PathBuf {
    let path = std::env::var(var).map(PathBuf::from).unwrap_or_else(|_| PathBuf::from(REPO).join(default));
    path.canonicalize().unwrap_or_else(|err| panic!("{var} = {}: {err}", path.display()))
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    let case: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&args[1]).unwrap()).unwrap();
    let out = PathBuf::from(&args[2]);
    std::fs::create_dir_all(&out).unwrap();
    let out = out.canonicalize().unwrap();
    let katago = path_var("KATAGO", "build/eigen/katago");
    let models = path_var("KATAGO_MODELS", "../katago-ws/bench/models");
    let config = path_var("KATAGO_CONFIG", "upstream/KataGo/cpp/configs/analysis_example.cfg");
    let logs = out.join("../.katago-logs");
    let sgf_text = std::fs::read_to_string(PathBuf::from(REPO).join(case["sgf"].as_str().unwrap())).unwrap();
    std::fs::write(out.join("game.sgf"), &sgf_text).unwrap();
    let human_profile = case["humanProfile"].as_str().map(str::to_owned);

    let wrapper = out.join("../.katago-tee.sh");
    std::fs::write(
        &wrapper,
        format!(
            "#!/bin/sh\nOUT=\"$KATAGO_TEE_OUT\"\ntee \"$OUT/queries.jsonl\" | '{}' \"$@\" | tee \"$OUT/responses.jsonl\"\n",
            katago.display()
        ),
    )
    .unwrap();
    std::process::Command::new("chmod").arg("+x").arg(&wrapper).status().unwrap();
    // SAFETY: single-threaded at this point (no runtime tasks spawned yet).
    unsafe { std::env::set_var("KATAGO_TEE_OUT", &out) };
    std::env::set_current_dir(&models).unwrap();

    let mut engine_json = serde_json::json!({
        "binary": wrapper.to_str().unwrap(),
        "config": config.to_str().unwrap(),
        "analysis_threads": 8,
        "search_threads_per_analysis": 1,
        "override_config": format!("logDir={}", logs.display()),
        "autotune": false,
        "request_timeout_secs": 7200,
    });
    for (k, v) in case["engine"].as_object().unwrap() {
        engine_json[k] = v.clone();
    }
    let cfg: EngineConfig = serde_json::from_value(engine_json).unwrap();

    let katago = KataGo::spawn(&cfg).await.unwrap();
    katago.wait_ready().await.unwrap();
    let t0 = std::time::Instant::now();

    // katago-ws src/engine/mod.rs AnalysisEngine::analyze (and ::pass), minus tracing and metrics.
    let game = sgf::parse(&sgf_text, cfg.default_board_size, cfg.default_komi).unwrap();
    let human_profile = human_profile.as_deref().filter(|_| !cfg.human_model.is_empty());
    let all_turns: Vec<usize> = (0..=game.moves.len()).collect();
    let turns = if cfg.two_pass() {
        let screened = katago.analyze(&game, &cfg, all_turns, cfg.screen_visits, 0, human_profile).await.unwrap();
        let refine = annotate::turns_to_refine(&game, &screened, &cfg);
        let refined = katago
            .analyze(&game, &cfg, refine.into_iter().collect(), cfg.max_visits, 1, human_profile)
            .await
            .unwrap();
        ws_parity::merge_turns(screened, refined)
    } else {
        katago.analyze(&game, &cfg, all_turns, cfg.max_visits, 0, human_profile).await.unwrap()
    };
    let analysis = annotate::assemble(&game, turns, &cfg, human_profile);
    let payload = serde_json::to_string(&analysis).unwrap();
    std::fs::write(out.join("expected.json"), &payload).unwrap();
    eprintln!("{}: {} moves in {:.1} s", out.display(), game.moves.len(), t0.elapsed().as_secs_f64());

    // The annotate options (katago-annotate's camelCase names) this case ran with.
    let options = serde_json::json!({
        "model": cfg.model,
        "maxVisits": cfg.max_visits,
        "analysisPvLen": cfg.analysis_pv_len,
        "includeOwnership": cfg.include_ownership,
        "includeOwnershipStdev": cfg.include_ownership_stdev,
        "includePolicy": cfg.include_policy,
        "humanModel": cfg.human_model,
        "topK": cfg.top_k,
        "defaultKomi": cfg.default_komi,
        "defaultBoardSize": cfg.default_board_size,
        "screenVisits": cfg.screen_visits,
        "refineWinrateDelta": cfg.refine_winrate_delta,
        "refineScoreDelta": cfg.refine_score_delta,
    });
    let mut case_out = serde_json::json!({ "options": options });
    if let Some(profile) = case["humanProfile"].as_str() {
        case_out["humanProfile"] = profile.into();
    }
    std::fs::write(out.join("case.json"), serde_json::to_string_pretty(&case_out).unwrap() + "\n").unwrap();
    katago.shutdown().await;
    tokio::time::sleep(std::time::Duration::from_millis(300)).await;
}
