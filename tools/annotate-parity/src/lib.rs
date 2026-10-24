//! katago-ws's engine modules, unchanged (`#[path]` into the sibling checkout ../katago-ws, recorded at bdd674bb), plus a
//! shim of `crate::error` (AppError without its HTTP side).

#[path = "../../../../katago-ws/src/config.rs"]
pub mod config;
pub mod error;
pub mod engine;

use std::collections::BTreeMap;

use engine::katago::TurnResponse;

/// katago-ws src/engine/mod.rs `merge_turns`, verbatim (mod.rs itself needs muxa).
pub fn merge_turns(screened: Vec<TurnResponse>, refined: Vec<TurnResponse>) -> Vec<TurnResponse> {
    let mut by_turn: BTreeMap<usize, TurnResponse> = screened
        .into_iter()
        .map(|turn| (turn.turn_number, turn))
        .collect();
    by_turn.extend(refined.into_iter().map(|turn| (turn.turn_number, turn)));
    by_turn.into_values().collect()
}
