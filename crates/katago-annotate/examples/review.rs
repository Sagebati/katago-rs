//! Reviews a game natively with any KataGo analysis engine on stdin/stdout (upstream `katago`, or
//! katago-cli's `katago-rs`) and prints the payload v3 JSON, as katago-ws would build it.
//!
//!   cargo run -p katago-annotate --example review -- <game.sgf> <options.json | '{...}' | -> -- <engine> <args...>
//!
//! e.g. `... game.sgf '{"maxVisits": 50}' -- target/release/katago-rs analysis -config cfg -model g170-b6c96`.
//! The options are katago-annotate's `AnnotateOptions` (camelCase) plus `humanProfile`. Used to make the
//! native reference the browser review is compared with (packages/katago-web/test, docs/M6-report.md).

use std::cell::RefCell;
use std::future::Future;
use std::io::{BufRead as _, BufReader, Write as _};
use std::pin::pin;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::task::{Context, Poll, Waker};

use katago_annotate::{AnalysisQuery, AnnotateError, AnnotateOptions, AnnotateResult, Analyzer, Pass, ResponseLine, TurnResponse};

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
        let mut got = Vec::with_capacity(expected);
        let mut line = String::new();
        while got.len() < expected {
            line.clear();
            if self.stdout.borrow_mut().read_line(&mut line).map_err(io)? == 0 {
                return Err(AnnotateError::Inference("the engine exited".to_owned()));
            }
            match ResponseLine::parse(line.trim_end()) {
                ResponseLine::Turn { id: got_id, turn } if got_id == id => {
                    got.push(*turn);
                    eprintln!("{} {}/{expected} (turn {})", pass.as_str(), got.len(), got.last().unwrap().turn_number);
                }
                ResponseLine::Error { error, .. } => return Err(AnnotateError::Inference(format!("katago: {error}"))),
                other => eprintln!("ignored: {other:?}"),
            }
        }
        Ok(got)
    }
}

fn block_on<F: Future>(fut: F) -> F::Output {
    let mut fut = pin!(fut);
    let mut cx = Context::from_waker(Waker::noop());
    loop {
        if let Poll::Ready(out) = fut.as_mut().poll(&mut cx) {
            return out;
        }
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
    let mut options: serde_json::Value = serde_json::from_str(&options_text).unwrap();
    let human_profile = options.as_object_mut().unwrap().remove("humanProfile").and_then(|v| v.as_str().map(str::to_owned));
    let options: AnnotateOptions = serde_json::from_value(options).unwrap();

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
