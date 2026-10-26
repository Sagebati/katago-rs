// The game-level flow of katago-ws src/engine/mod.rs @ bdd674bb9431af2bac8e2e5dfd7d2eb00e5a06f3
// (`AnalysisEngine::analyze`, `AnalysisEngine::pass`, `Pass`, `merge_turns`; see docs/M6-report.md), with the
// engine behind the [`Analyzer`] trait instead of katago-ws's `KataGo` subprocess client. Dropped: the muxa
// plugins, auto-tune and tracing spans/metrics. Divergences are marked `KATAGO-ANNOTATE:`.

//! A whole game through the engine: SGF → queries → [`GameAnalysis`].

use std::collections::BTreeMap;
use std::future::Future;

use crate::annotate::{self, GameAnalysis};
use crate::error::AnnotateResult;
use crate::katago::{AnalysisQuery, TurnResponse};
use crate::options::AnnotateOptions;
use crate::sgf::{self, ParsedGame};

/// What runs the queries: katago-ws's `KataGo` client (a subprocess), the
/// browser's in-wasm engine, or a replay of recorded responses.
///
/// KATAGO-ANNOTATE: katago-ws calls `KataGo::analyze(game, cfg, turns,
/// visits, priority, human_profile)` directly; here the query is built by
/// [`AnalysisQuery::new`] and the host only runs it.
pub trait Analyzer {
    /// Run one query (its `id` empty: the host assigns one) and return its
    /// final [`TurnResponse`]s, one per requested turn, in any order.
    /// Never called with an empty `analyze_turns`.
    fn analyze(
        &self,
        pass: Pass,
        query: AnalysisQuery,
    ) -> impl Future<Output = AnnotateResult<Vec<TurnResponse>>>;
}

/// Analyze a full game and annotate every move.
///
/// One pass at `max_visits` over every turn, or — with
/// [`AnnotateOptions::two_pass`] — a cheap screening pass over every turn and
/// a full-depth pass over only the turns around moves that swing (see
/// [`annotate::turns_to_refine`]).
///
/// `human_profile` (`rank_5k`, …) adds the human SL network's policy to the
/// payload when this engine has a human model. On one without, the profile is
/// dropped with a warning rather than failing the job: KataGo would reject
/// every query that carried it.
///
/// # Errors
///
/// The SGF does not parse, or the engine fails a query.
pub async fn analyze<A: Analyzer>(
    engine: &A,
    cfg: &AnnotateOptions,
    sgf: &str,
    human_profile: Option<&str>,
) -> AnnotateResult<GameAnalysis> {
    let game = sgf::parse(sgf, cfg.default_board_size, cfg.default_komi)?;
    analyze_game(engine, cfg, &game, human_profile).await
}

/// [`analyze`] for a game already parsed (KATAGO-ANNOTATE: split from it, for
/// games given as moves rather than SGF).
///
/// # Errors
///
/// The engine fails a query.
pub async fn analyze_game<A: Analyzer>(
    engine: &A,
    cfg: &AnnotateOptions,
    game: &ParsedGame,
    human_profile: Option<&str>,
) -> AnnotateResult<GameAnalysis> {
    let human_profile = human_profile.filter(|profile| {
        let loaded = !cfg.human_model.is_empty();
        if !loaded {
            log::warn!(
                "job asked for a human profile ({profile}) but no human_model is loaded; \
                 analysing without human policy"
            );
        }
        loaded
    });
    let all_turns: Vec<usize> = (0..=game.moves.len()).collect();

    let turns = if cfg.two_pass() {
        let screened = pass(
            engine,
            cfg,
            Pass::Screen,
            game,
            all_turns,
            cfg.screen_visits,
            human_profile,
        )
        .await?;
        let refine = annotate::turns_to_refine(game, &screened, cfg);
        // Refinement goes ahead of other games' screening in KataGo's
        // queue, so a game isn't held back waiting for its few deep
        // searches behind everyone else's shallow ones.
        let refined = pass(
            engine,
            cfg,
            Pass::Refine,
            game,
            refine.into_iter().collect(),
            cfg.max_visits,
            human_profile,
        )
        .await?;
        merge_turns(screened, refined)
    } else {
        pass(
            engine,
            cfg,
            Pass::Full,
            game,
            all_turns,
            cfg.max_visits,
            human_profile,
        )
        .await?
    };
    Ok(annotate::assemble(game, turns, cfg, human_profile))
}

/// One KataGo query over `turns`.
///
/// KATAGO-ANNOTATE: `AnalysisEngine::pass` plus the front of
/// `KataGo::analyze` (no query for no turns); no tracing span or metric.
async fn pass<A: Analyzer>(
    engine: &A,
    cfg: &AnnotateOptions,
    pass: Pass,
    game: &ParsedGame,
    turns: Vec<usize>,
    visits: u32,
    human_profile: Option<&str>,
) -> AnnotateResult<Vec<TurnResponse>> {
    if turns.is_empty() {
        return Ok(Vec::new());
    }
    let query = AnalysisQuery::new(game, cfg, turns, visits, pass.priority(), human_profile);
    engine.analyze(pass, query).await
}

/// Which search a KataGo query is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pass {
    /// Single-pass analysis: every turn at `max_visits`.
    Full,
    /// Two-pass, first pass: every turn at `screen_visits`.
    Screen,
    /// Two-pass, second pass: the swinging turns at `max_visits`.
    Refine,
}

impl Pass {
    /// `full` | `screen` | `refine`.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::Screen => "screen",
            Self::Refine => "refine",
        }
    }

    /// KataGo queue priority: refinement finishes games already in flight.
    #[must_use]
    pub fn priority(self) -> i32 {
        match self {
            Self::Full | Self::Screen => 0,
            Self::Refine => 1,
        }
    }
}

/// Overlay the refined turns on the screened ones, by turn number.
#[must_use]
pub fn merge_turns(screened: Vec<TurnResponse>, refined: Vec<TurnResponse>) -> Vec<TurnResponse> {
    let mut by_turn: BTreeMap<usize, TurnResponse> = screened
        .into_iter()
        .map(|turn| (turn.turn_number, turn))
        .collect();
    by_turn.extend(refined.into_iter().map(|turn| (turn.turn_number, turn)));
    by_turn.into_values().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn turn(number: usize, visits: u32) -> TurnResponse {
        serde_json::from_value(serde_json::json!({
            "id": "q0", "turnNumber": number,
            "rootInfo": {"winrate": 0.5, "scoreLead": 0.0, "visits": visits},
            "moveInfos": [{"move": "D4", "winrate": 0.5, "visits": visits}]
        }))
        .unwrap()
    }

    #[test]
    fn refined_turns_replace_their_screened_versions_in_turn_order() {
        let screened = vec![turn(2, 16), turn(0, 16), turn(1, 16), turn(3, 16)];
        let refined = vec![turn(2, 400), turn(1, 400)];
        let merged = merge_turns(screened, refined);
        let shape: Vec<_> = merged
            .iter()
            .map(|turn| (turn.turn_number, turn.move_infos[0].visits))
            .collect();
        assert_eq!(shape, vec![(0, 16), (1, 400), (2, 400), (3, 16)]);
    }
}
