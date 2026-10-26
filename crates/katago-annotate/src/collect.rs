//! One analysis query's turn responses, collected from the engine's output lines: the loop every host of
//! [`crate::Analyzer`] runs (katago-wasm's reviews, the `review` example, the parity tests). Not part of
//! katago-ws's code: katago-ws's KataGo client does the same in its own `analyze` (handle_line).

use crate::error::{AnnotateError, AnnotateResult};
use crate::katago::{ResponseLine, TurnResponse};

/// The final responses of query `id`, one per requested turn (the queries ask for no `reportDuringSearchEvery`, so
/// every turn line of the query is a final one), fed one output line at a time.
pub struct TurnCollector {
    id: String,
    expected: usize,
    turns: Vec<TurnResponse>,
}

/// What one line was to a [`TurnCollector`].
#[derive(Debug)]
pub enum Fed {
    /// A turn of this query, now the last of [`TurnCollector::turns`].
    Turn,
    /// A `{"warning": ...}` line.
    Warning(String),
    /// A line that is not a response this collector can read (the text says what it was).
    Unparsable(String),
    /// A turn of another query.
    OtherQuery(String),
    /// A line with no `id` and no error.
    NoId,
}

impl TurnCollector {
    /// A collector for query `id`, which asked for `expected` turns.
    #[must_use]
    pub fn new(id: impl Into<String>, expected: usize) -> Self {
        Self {
            id: id.into(),
            expected,
            turns: Vec::with_capacity(expected),
        }
    }

    /// Whether every requested turn has arrived.
    #[must_use]
    pub fn is_done(&self) -> bool {
        self.turns.len() >= self.expected
    }

    /// The turns requested.
    #[must_use]
    pub fn expected(&self) -> usize {
        self.expected
    }

    /// The turns collected so far, in arrival order.
    #[must_use]
    pub fn turns(&self) -> &[TurnResponse] {
        &self.turns
    }

    /// Reads one output line.
    ///
    /// # Errors
    ///
    /// The engine reported an error (`{"error": ...}`): the query failed.
    pub fn feed(&mut self, line: &str) -> AnnotateResult<Fed> {
        Ok(match ResponseLine::parse(line) {
            ResponseLine::Turn { id, turn } if id == self.id => {
                self.turns.push(*turn);
                Fed::Turn
            }
            ResponseLine::Turn { id, .. } => Fed::OtherQuery(id),
            ResponseLine::Error { error, .. } => {
                return Err(AnnotateError::Inference(format!("katago: {error}")));
            }
            ResponseLine::Warning { warning, .. } => Fed::Warning(warning),
            ResponseLine::Unparsable(e) => Fed::Unparsable(format!("unparsable line: {e}")),
            ResponseLine::UnparsableResponse { error, .. } => {
                Fed::Unparsable(format!("unparsable response: {error}"))
            }
            ResponseLine::NoId => Fed::NoId,
        })
    }

    /// The collected turns.
    #[must_use]
    pub fn into_turns(self) -> Vec<TurnResponse> {
        self.turns
    }
}
