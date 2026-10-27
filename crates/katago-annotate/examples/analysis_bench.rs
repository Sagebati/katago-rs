//! Analysis throughput of any KataGo analysis engine on stdin/stdout (upstream `katago analysis`, or katago-cli's
//! `katago-rs analysis`): positions per second at katago-ws's queries, with `--in-flight` games queued at once.
//!
//!   cargo run --release -p katago-annotate --example analysis_bench -- [options] -- <engine> <args...>
//!
//! Options: `--games DIR` (default fixtures/bench-games, katago-ws's 17-game `bench/sgf`: every `*.sgf`, cycled,
//! with the NN cache cleared each time the corpus starts over), `--in-flight N` (4, games queued
//! at once, katago-ws's `[worker].concurrency`), `--visits V` (100), `--warmup S` (60, seconds from the first
//! response before measuring: kernel compiles, autotune, batch shapes), `--secs S` (120 per window), `--windows N`
//! (2). Each game is one query of all its turns, katago-annotate's query as katago-ws sends it (ownership, stdev
//! and policy included). A window's positions/s is the turns answered in it over its length; the engine is
//! killed after the last window. README "Benchmarks", `just bench-analysis`.

use std::io::{BufRead as _, BufReader, Write as _};
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use katago_annotate::{AnalysisQuery, AnnotateOptions, ParsedGame, ResponseLine};

struct Options {
    games: String,
    in_flight: usize,
    visits: u32,
    warmup: f64,
    secs: f64,
    windows: usize,
}

fn parse_options(args: &[String]) -> Options {
    let mut o = Options {
        games: concat!(env!("CARGO_MANIFEST_DIR"), "/../../fixtures/bench-games").to_owned(),
        in_flight: 4,
        visits: 100,
        warmup: 60.0,
        secs: 120.0,
        windows: 2,
    };
    for pair in args.chunks(2) {
        let [flag, value] = pair else {
            panic!("{}: missing value", pair[0])
        };
        let bad = |e: &dyn std::fmt::Display| -> ! { panic!("{flag} {value}: {e}") };
        match flag.as_str() {
            "--games" => o.games.clone_from(value),
            "--in-flight" => o.in_flight = value.parse().unwrap_or_else(|e| bad(&e)),
            "--visits" => o.visits = value.parse().unwrap_or_else(|e| bad(&e)),
            "--warmup" => o.warmup = value.parse().unwrap_or_else(|e| bad(&e)),
            "--secs" => o.secs = value.parse().unwrap_or_else(|e| bad(&e)),
            "--windows" => o.windows = value.parse().unwrap_or_else(|e| bad(&e)),
            _ => panic!("unknown option {flag}"),
        }
    }
    o
}

fn load_games(dir: &str) -> Vec<(String, ParsedGame)> {
    let mut paths: Vec<_> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("{dir}: {e}"))
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "sgf"))
        .collect();
    paths.sort();
    assert!(!paths.is_empty(), "no .sgf in {dir}");
    paths
        .iter()
        .map(|p| {
            let sgf = std::fs::read_to_string(p).unwrap();
            let game = katago_annotate::sgf::parse(&sgf, 19, 7.5)
                .unwrap_or_else(|e| panic!("{}: {e}", p.display()));
            (p.file_name().unwrap().to_string_lossy().into_owned(), game)
        })
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let sep = args
        .iter()
        .position(|a| a == "--")
        .expect("usage: analysis_bench [options] -- <engine> <args...>");
    let o = parse_options(&args[..sep]);
    let games = load_games(&o.games);
    let annotate = AnnotateOptions {
        max_visits: o.visits,
        ..AnnotateOptions::default()
    };

    let mut child = Command::new(&args[sep + 1])
        .args(&args[sep + 2..])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap_or_else(|e| panic!("{}: {e}", args[sep + 1]));
    let mut stdin = child.stdin.take().unwrap();
    let stdout = BufReader::new(child.stdout.take().unwrap());
    let started = Instant::now();

    // The reader thread timestamps each line as it arrives; the main thread keeps the queue full.
    let (tx, rx) = mpsc::channel::<(Instant, String)>();
    std::thread::spawn(move || {
        for line in stdout.lines() {
            let Ok(line) = line else { break };
            if tx.send((Instant::now(), line)).is_err() {
                break;
            }
        }
    });

    // id -> turns still to come
    let mut pending: Vec<(String, usize)> = Vec::new();
    let mut next_game = 0usize;
    let mut send = |pending: &mut Vec<(String, usize)>| {
        // A game seen again would come out of the NN cache: empty it each time the corpus starts over.
        if next_game > 0 && next_game.is_multiple_of(games.len()) {
            writeln!(
                stdin,
                r#"{{"id":"clear{next_game}","action":"clear_cache"}}"#
            )
            .unwrap();
        }
        let (name, game) = &games[next_game % games.len()];
        let turns: Vec<usize> = (0..=game.moves.len()).collect();
        let mut query = AnalysisQuery::new(game, &annotate, turns, o.visits, 0, None);
        query.id = format!("g{next_game}-{name}");
        writeln!(stdin, "{}", serde_json::to_string(&query).unwrap()).unwrap();
        stdin.flush().unwrap();
        pending.push((query.id, game.moves.len() + 1));
        next_game += 1;
    };
    for _ in 0..o.in_flight {
        send(&mut pending);
    }

    let mut first: Option<Instant> = None;
    let mut window_start: Option<Instant> = None;
    let mut counted = 0usize;
    let mut rates = Vec::new();
    let mut games_done = 0usize;
    while rates.len() < o.windows {
        let (t, line) = rx
            .recv_timeout(Duration::from_secs(1200))
            .unwrap_or_else(|e| panic!("no output from the engine: {e}"));
        let id = match ResponseLine::parse(&line) {
            // not clear_cache's acknowledgement
            ResponseLine::Turn { id, .. } if !id.starts_with("clear") => id,
            ResponseLine::Error { id, error } => panic!("engine error for {id:?}: {error}"),
            ResponseLine::UnparsableResponse { id, error } if !id.starts_with("clear") => {
                panic!("{id}: {error}")
            }
            _ => continue,
        };
        let first = *first.get_or_insert_with(|| {
            eprintln!(
                "analysis_bench: first response {:.1} s after start",
                (t - started).as_secs_f64()
            );
            t
        });
        let slot = pending
            .iter()
            .position(|(p, _)| *p == id)
            .unwrap_or_else(|| panic!("unknown id {id}"));
        pending[slot].1 -= 1;
        if pending[slot].1 == 0 {
            pending.swap_remove(slot);
            games_done += 1;
            send(&mut pending);
        }
        let since_first = (t - first).as_secs_f64();
        if since_first < o.warmup {
            continue;
        }
        let start = *window_start.get_or_insert(t);
        let elapsed = (t - start).as_secs_f64();
        if elapsed >= o.secs {
            #[allow(clippy::cast_precision_loss, reason = "thousands of positions")]
            let rate = counted as f64 / elapsed;
            eprintln!(
                "analysis_bench: window {}: {counted} positions in {elapsed:.1} s = {rate:.2} positions/s",
                rates.len() + 1
            );
            rates.push(rate);
            window_start = Some(t);
            counted = 0;
        }
        counted += 1;
    }
    let _ = child.kill();
    let _ = child.wait();
    let shown: Vec<String> = rates.iter().map(|r| format!("{r:.2}")).collect();
    println!(
        "positions/s {} ({} in flight, {} visits, {games_done} games done)",
        shown.join(" / "),
        o.in_flight,
        o.visits
    );
}
