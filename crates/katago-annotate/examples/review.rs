//! Reviews a game natively with any KataGo analysis engine on stdin/stdout (upstream `katago`, or
//! katago-cli's `katago-rs`) and prints the payload v3 JSON, as katago-ws would build it.
//!
//!   cargo run -p katago-annotate --example review -- <game.sgf> <options.json | '{...}' | -> -- <engine> <args...>
//!
//! e.g. `... game.sgf '{"maxVisits": 50}' -- target/release/katago-rs analysis -config cfg -model g170-b6c96`.
//! The options are katago-annotate's `AnnotateOptions` (camelCase) plus `humanProfile`. Used to make the
//! native reference the browser review is compared with (packages/katago-web/test, docs/M6-report.md).

use std::cell::RefCell;
use std::io::{BufRead as _, BufReader, Write as _};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use katago_annotate::{AnalysisQuery, AnnotateError, AnnotateOptions, AnnotateResult, Analyzer, Fed, Pass, TurnCollector, TurnResponse};

#[path = "../tests/support/mod.rs"]
mod support;
use support::block_on;

struct Subprocess {
    stdin: RefCell<ChildStdin>,
    stdout: RefCell<BufReader<ChildStdout>>,
    next_id: RefCell<u32>,
}

impl Analyzer for Subprocess {
    async fn analyze(&self, pass: Pass, mut query: AnalysisQuery) -> AnnotateResult<Vec<TurnResponse>> {
        let id = {
            let mut n = self.next_id.borrow_mut();
            *n += 1;
            format!("q{}", *n - 1)
        };
        query.id = id.clone();
        let expected = query.analyze_turns.len();
        let io = |e: std::io::Error| AnnotateError::Inference(e.to_string());
        writeln!(self.stdin.borrow_mut(), "{}", serde_json::to_string(&query).unwrap()).map_err(io)?;
        self.stdin.borrow_mut().flush().map_err(io)?;
        let mut collector = TurnCollector::new(id, expected);
        let mut line = String::new();
        while !collector.is_done() {
            line.clear();
            if self.stdout.borrow_mut().read_line(&mut line).map_err(io)? == 0 {
                return Err(AnnotateError::Inference("the engine exited".to_owned()));
            }
            match collector.feed(line.trim_end())? {
                Fed::Turn => {
                    let turns = collector.turns();
                    eprintln!("{} {}/{expected} (turn {})", pass.as_str(), turns.len(), turns.last().unwrap().turn_number);
                }
                other => eprintln!("ignored: {other:?}"),
            }
        }
        Ok(collector.into_turns())
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let sep = args.iter().position(|a| a == "--").expect("usage: review <game.sgf> <options> -- <engine> <args...>");
    let sgf = std::fs::read_to_string(&args[0]).unwrap();
    let options_text = match args[1].as_str() {
        "-" => "{}".to_owned(),
        s if s.trim_start().starts_with('{') => s.to_owned(),
        path => std::fs::read_to_string(path).unwrap(),
    };
    let (options, human_profile) = AnnotateOptions::from_json_with_profile(serde_json::from_str(&options_text).unwrap()).unwrap();

    let mut child: Child = Command::new(&args[sep + 1])
        .args(&args[sep + 2..])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let engine = Subprocess {
        stdin: RefCell::new(child.stdin.take().unwrap()),
        stdout: RefCell::new(BufReader::new(child.stdout.take().unwrap())),
        next_id: RefCell::new(0),
    };
    let payload = block_on(katago_annotate::analyze(&engine, &options, &sgf, human_profile.as_deref())).unwrap();
    println!("{}", serde_json::to_string(&payload).unwrap());
    drop(engine);
    let _ = child.wait();
}
