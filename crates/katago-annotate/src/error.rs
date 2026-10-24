// katago-ws src/error.rs @ bdd674bb9431af2bac8e2e5dfd7d2eb00e5a06f3, reduced to the `AppError` variants the
// copied code returns (same messages), without the HTTP side (axum, aide, diesel).

//! The errors of SGF parsing and of a review.

use std::fmt;

/// Errors surfaced by the copied katago-ws code (its `AppError`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AnnotateError {
    /// SGF could not be parsed or replayed.
    Sgf(String),
    /// The request was well-formed but asked for something we refuse.
    BadRequest(String),
    /// The engine failed (katago-ws: "ONNX inference failed", the variant its
    /// KataGo client reports engine errors with).
    Inference(String),
}

impl fmt::Display for AnnotateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Sgf(msg) => write!(f, "invalid SGF: {msg}"),
            Self::BadRequest(msg) => write!(f, "bad request: {msg}"),
            Self::Inference(msg) => write!(f, "inference error: {msg}"),
        }
    }
}

impl std::error::Error for AnnotateError {}

/// Convenience alias (katago-ws's `AppResult`).
pub type AnnotateResult<T> = Result<T, AnnotateError>;
