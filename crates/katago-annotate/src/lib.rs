//! katago-ws's game annotation, without its IO: an SGF (or a move list) becomes KataGo analysis queries,
//! and the engine's per-turn responses become the **payload v3** ([`GameAnalysis`]) that baduk_review
//! stores and renders.
//!
//! Copied from katago-ws @ `bdd674bb9431af2bac8e2e5dfd7d2eb00e5a06f3` with as few changes as possible (each
//! file names its source; every divergence is marked `KATAGO-ANNOTATE:`), so that a payload built here is
//! byte-identical to katago-ws's for the same engine output (`fixtures/katago-annotate`, docs/M6-report.md):
//! - [`sgf`]: `src/engine/sgf.rs` (SGF → moves, rules, komi through `sgf-parse`), plus [`sgf::GameInput`];
//! - [`katago`]: the protocol half of `src/engine/katago.rs` (the query, the response types, reading a line);
//! - [`annotate`]: `src/engine/annotate.rs` (losses, classifications, report, per-turn grids, refinement);
//! - [`review`]: the game-level flow of `src/engine/mod.rs` (one pass, or screen + refine), over an
//!   [`review::Analyzer`] the host implements (a subprocess, the in-browser engine, a replay);
//! - [`options`]: the analysis knobs of `src/config.rs`'s `EngineConfig`, with its defaults.
//!
//! No IO, no threads, no tokio: builds for wasm32-unknown-unknown, where `expf` is katago-core's port of
//! glibc's (the payload's `accuracy` would otherwise differ in the last bit). Not a port of upstream KataGo code.

pub mod annotate;
pub mod collect;
pub mod error;
pub mod katago;
mod math;
pub mod options;
pub mod review;
pub mod sgf;

pub use annotate::{GameAnalysis, PAYLOAD_VERSION, assemble, turns_to_refine};
pub use collect::{Fed, TurnCollector};
pub use error::{AnnotateError, AnnotateResult};
pub use katago::{AnalysisQuery, ResponseLine, TurnResponse};
pub use options::AnnotateOptions;
pub use review::{Analyzer, Pass, analyze, analyze_game};
pub use sgf::{GameInput, ParsedGame};
