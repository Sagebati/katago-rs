// Copied from katago-ws src/engine/annotate.rs @ bdd674bb9431af2bac8e2e5dfd7d2eb00e5a06f3 (see docs/M6-report.md);
// the payload types are what baduk_review's libs/katago-sdk/src/models.rs @ 8bcb98e1 reads (payload v3).
// Divergences, each marked `KATAGO-ANNOTATE:`: `EngineConfig` -> `AnnotateOptions`; `EngineInfo::name` comes
// from the options (default `katago`) and an optional `version` follows it (left out when `None`, so a
// default payload is byte-identical to katago-ws's); `tracing::warn!` fields -> the message; `f32::exp` ->
// `crate::math::expf` (glibc's expf on wasm32 as natively).

//! Turn KataGo per-turn analysis into the annotation payload returned to clients.
//!
//! Loss for move *i* is the difference between two consecutive root
//! evaluations: what the position was worth to the mover before they played,
//! minus what it was worth to them after. Differencing consecutive positions
//! rather than comparing against a candidate list stays correct even when the
//! move played isn't among KataGo's reported candidates.
//!
//! The subtlety that makes this hard — and that this module got wrong until
//! `payload_version = 2` — is that the two evaluations must be *in the same
//! player's frame*. Turns `i` and `i+1` belong to opposite players, and KataGo
//! reports in whichever fixed frame `reportAnalysisWinratesAs` names, so
//! "rotate both into the mover's frame" and "negate the second one" are
//! different operations that coincide only when the engine reports side-to-move.
//! See [`crate::katago::Perspective`].

use std::collections::{BTreeSet, HashMap};

use serde::Serialize;

use crate::katago::{Eval, REPORT_PERSPECTIVE, RootInfo, TurnResponse};
// KATAGO-ANNOTATE: katago-ws's `EngineConfig`, reduced to the analysis knobs.
use crate::options::AnnotateOptions as EngineConfig;
use crate::sgf::{self, Color, ParsedGame, Rules};

/// The payload contract version.
///
/// Bumped when a field's *meaning* changes, not when one is added — every
/// consumer tolerates unknown fields.
///
/// - `1` — assumed KataGo reported from the side to move. Every per-move loss
///   in a v1 payload is the wrong number, not merely an imprecise one.
/// - `2` — perspective-correct, and carries per-turn ownership.
/// - `3` — adds per-turn `ownership_stdev`, `policy` and (with a human profile)
///   `human_policy` grids, and `prior`/`human_prior` on the played move and
///   `human_prior` on candidates. Every v2 field is unchanged, so a v2 reader
///   reads a v3 payload as it was.
pub const PAYLOAD_VERSION: u32 = 3;

/// How a played move is judged, by either lens (win-rate loss or score loss).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Classification {
    /// Negligible loss — essentially the best move.
    Good,
    /// A small slip.
    Inaccuracy,
    /// A clear error.
    Mistake,
    /// A large, game-affecting error.
    Blunder,
}

/// A candidate move KataGo suggested for a position.
#[derive(Serialize)]
pub struct Candidate {
    /// Move in GTP notation.
    pub mv: String,
    /// Win probability after this move (mover's perspective).
    pub winrate: f32,
    /// Score lead after this move.
    pub score_lead: f32,
    /// Raw policy prior.
    pub prior: f32,
    /// The human SL network's prior for this move, under the job's
    /// `human_profile`. Absent without one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub human_prior: Option<f32>,
    /// MCTS visits.
    pub visits: u32,
    /// Principal variation following this move.
    pub pv: Vec<String>,
}

/// Annotation for a single played move.
#[derive(Serialize)]
pub struct MoveAnnotation {
    /// 1-based move number.
    pub move_number: usize,
    /// Mover's colour.
    pub color: Color,
    /// The move played, GTP notation.
    pub mv: String,
    /// Mover's win-rate after the actual move.
    pub winrate: f32,
    /// Best win-rate achievable from this position.
    pub best_winrate: f32,
    /// `best_winrate − winrate`, clamped at 0.
    pub winrate_loss: f32,
    /// Classification by win-rate loss.
    pub winrate_classification: Classification,
    /// Expected score lead (mover's perspective) before the move — the best
    /// achievable with optimal play.
    pub score_lead: f32,
    /// Points given up by the move: `score_lead − resulting_lead`, clamped at 0.
    pub score_loss: f32,
    /// Classification by score loss (points). A more uniform yardstick than
    /// win-rate across game phases.
    pub score_classification: Classification,
    /// The network's raw policy for the move played, in `[0, 1]` — exact,
    /// from the full policy, so present even when the move isn't among
    /// `top_moves`. Absent when the worker ran without `includePolicy` and the
    /// move isn't a candidate either.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prior: Option<f32>,
    /// The human SL network's policy for the move played, under the job's
    /// `human_profile`. Absent without one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub human_prior: Option<f32>,
    /// Top candidate moves for this position.
    pub top_moves: Vec<Candidate>,
}

/// Move-quality breakdown for one player under a single lens (win-rate loss or
/// score loss): how many of their moves fell into each [`Classification`].
#[derive(Serialize, Default)]
pub struct ClassCounts {
    /// Moves classified as `good`.
    pub good: usize,
    /// Moves classified as `inaccuracy`.
    pub inaccuracy: usize,
    /// Moves classified as `mistake`.
    pub mistake: usize,
    /// Moves classified as `blunder`.
    pub blunder: usize,
}

impl ClassCounts {
    /// Tally one move's classification.
    fn record(&mut self, classification: Classification) {
        match classification {
            Classification::Good => self.good += 1,
            Classification::Inaccuracy => self.inaccuracy += 1,
            Classification::Mistake => self.mistake += 1,
            Classification::Blunder => self.blunder += 1,
        }
    }
}

/// AI-review-style aggregate metrics for one player across the whole game.
#[derive(Serialize)]
pub struct PlayerReport {
    /// Number of moves this player made.
    pub moves: usize,
    /// Overall accuracy in `[0, 100]` — the mean of each move's accuracy, where a
    /// single move's accuracy is mapped from its win-rate loss by [`move_accuracy`].
    /// A flawless game approaches 100.
    pub accuracy: f32,
    /// Mean win probability given up per move (`[0, 1]`).
    pub mean_winrate_loss: f32,
    /// Mean points given up per move.
    pub mean_score_loss: f32,
    /// Move-quality breakdown by win-rate loss.
    pub by_winrate: ClassCounts,
    /// Move-quality breakdown by score loss.
    pub by_score: ClassCounts,
    /// Coarse strength estimate derived from `mean_score_loss`. Present only for
    /// full-size (19×19) games with enough moves to be meaningful; a rough
    /// indicator, not a calibrated rank. Absent otherwise.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub estimated_rank: Option<String>,
}

/// Whole-game report: per-player aggregate metrics (accuracy, mean loss, the
/// mistake/blunder breakdown, and a rough strength estimate).
#[derive(Serialize)]
pub struct GameReport {
    /// Number of annotated moves (both players).
    pub num_moves: usize,
    /// Black's aggregate metrics.
    pub black: PlayerReport,
    /// White's aggregate metrics.
    pub white: PlayerReport,
}

/// What produced a payload.
///
/// Recorded so a metric from one engine build is never silently compared with
/// another's: the same game at 40 visits and at 500 gives different point
/// losses, and neither is wrong.
#[derive(Serialize)]
pub struct EngineInfo {
    /// Engine name — `katago` from katago-ws.
    /// KATAGO-ANNOTATE: from the options (`&'static str` upstream).
    pub name: String,
    /// KATAGO-ANNOTATE: the engine's version, when the producer gives one
    /// (katago-ws does not; the browser sends the KataGo version it follows).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    /// Neural net the analysis ran against.
    pub model: String,
    /// Visits spent per analysed position.
    pub max_visits: u32,
    /// Candidates reported per position.
    pub top_k: usize,
    /// The frame KataGo was told to report in, before normalisation. Stored so
    /// a future perspective bug is diagnosable from a payload alone.
    pub report_perspective: &'static str,
    /// Whether ownership maps were requested.
    pub ownership: bool,
    /// Whether `ownership_stdev` grids are present.
    pub ownership_stdev: bool,
    /// Whether `policy` grids are present.
    pub policy: bool,
    /// The human SL network behind `human_policy`, when there is one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub human_model: Option<String>,
    /// Visits of the screening pass, when two-pass analysis ran: positions
    /// around moves that looked quiet at this depth were not re-searched at
    /// `max_visits` (each candidate's `visits` shows the depth it got).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub screen_visits: Option<u32>,
}

/// Full analysis payload.
#[derive(Serialize)]
pub struct GameAnalysis {
    /// Board size.
    pub board_size: u8,
    /// Komi.
    pub komi: f32,
    /// KataGo rules used.
    pub rules: Rules,
    /// Raw SGF result, if any.
    pub result: Option<String>,
    /// Per-move annotations in play order.
    pub moves: Vec<MoveAnnotation>,
    /// Per-player aggregate report.
    pub report: GameReport,
    /// Which payload contract this was built under. See [`PAYLOAD_VERSION`].
    pub payload_version: u32,
    /// What produced it.
    pub engine: EngineInfo,
    /// Expected ownership of every intersection at each turn `0..=moves.len()`,
    /// row-major from the top-left, in hundredths and **always Black-positive**
    /// whatever frame the engine reported in.
    ///
    /// Indexed by *turn*, not by move: turn `i` is the position before move
    /// `i + 1`, so turn `i + 1` is that move's "after". Storing before and
    /// after per move would duplicate every array for nothing.
    ///
    /// Empty when the worker ran without `includeOwnership`, and — deliberately
    /// — all-or-nothing: a partial set would invite consumers to index into it
    /// and land on the wrong turn.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub ownership: Vec<Vec<i8>>,
    /// How unsettled each point's ownership is at each turn: the standard
    /// deviation of ownership across KataGo's search tree, in hundredths
    /// (`0..=100`). Same shape, turn indexing and all-or-nothing rule as
    /// `ownership`. A spread, not a side's view, so it needs no flip.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub ownership_stdev: Vec<Vec<u8>>,
    /// The network's raw policy at each turn, for the side to move there:
    /// `board_size²` points row-major from the top-left (KataGo's trailing
    /// pass entry is dropped — the played move's `prior` covers a pass),
    /// encoded `round(255 · sqrt(p))` so small probabilities keep some
    /// resolution. `0` is an illegal point or `p < ~4e-6`. Decode with
    /// `(v / 255)²`. Same turn indexing and all-or-nothing rule as `ownership`.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub policy: Vec<Vec<u8>>,
    /// The human SL network's policy at each turn under `human_profile`, in
    /// `policy`'s shape and encoding. Present only when the job gave a profile
    /// and the worker has a human model.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub human_policy: Vec<Vec<u8>>,
    /// The human SL profile `human_policy` and every `human_prior` were
    /// computed for (`rank_5k`, …). Present exactly when `human_policy` is.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub human_profile: Option<String>,
}

/// Running per-player accumulator used while walking the move list; finalized
/// into a [`PlayerReport`] once every move is counted.
#[derive(Default)]
struct PlayerAccum {
    moves: usize,
    winrate_loss_sum: f32,
    score_loss_sum: f32,
    accuracy_sum: f32,
    by_winrate: ClassCounts,
    by_score: ClassCounts,
}

impl PlayerAccum {
    /// Fold one of the player's moves into the running totals.
    fn record(
        &mut self,
        winrate_loss: f32,
        score_loss: f32,
        winrate_class: Classification,
        score_class: Classification,
    ) {
        self.moves += 1;
        self.winrate_loss_sum += winrate_loss;
        self.score_loss_sum += score_loss;
        self.accuracy_sum += move_accuracy(winrate_loss);
        self.by_winrate.record(winrate_class);
        self.by_score.record(score_class);
    }

    /// Turn the accumulated totals into the player's report.
    fn finish(self, board_size: u8) -> PlayerReport {
        // Guard the empty-player case (e.g. a game with no White moves): means
        // are 0, not a division by zero.
        let denom = self.moves.max(1) as f32;
        let mean_score_loss = self.score_loss_sum / denom;
        PlayerReport {
            moves: self.moves,
            accuracy: self.accuracy_sum / denom,
            mean_winrate_loss: self.winrate_loss_sum / denom,
            mean_score_loss,
            by_winrate: self.by_winrate,
            by_score: self.by_score,
            estimated_rank: estimate_rank(board_size, self.moves, mean_score_loss),
        }
    }
}

/// Per-move accuracy in `[0, 100]` from win-rate loss, via the logistic mapping
/// popularised by Lichess: `103.1668·e^(−0.04354·Δ) − 3.1669`, where `Δ` is the
/// win probability dropped, in **percentage points**. A best move scores ~100 and
/// accuracy decays smoothly as a move gives up more win probability.
fn move_accuracy(winrate_loss: f32) -> f32 {
    let drop = (winrate_loss * 100.0).max(0.0);
    // KATAGO-ANNOTATE: `(-0.04354 * drop).exp()` upstream; glibc's expf on wasm32 too (crate::math).
    (103.1668 * crate::math::expf(-0.04354 * drop) - 3.1669).clamp(0.0, 100.0)
}

/// Below this move count a strength estimate is too noisy to report.
const RANK_MIN_MOVES: usize = 20;

/// Very rough strength band from mean points lost per move — monotone (fewer
/// points lost ⇒ stronger). Emitted only for full 19×19 games with at least
/// [`RANK_MIN_MOVES`] moves; it's a coarse indicator, not a calibrated rank, and
/// the thresholds are heuristic (tune to taste).
fn estimate_rank(board_size: u8, moves: usize, mean_score_loss: f32) -> Option<String> {
    if board_size != 19 || moves < RANK_MIN_MOVES {
        return None;
    }
    let band = if mean_score_loss < 0.8 {
        "~7d+"
    } else if mean_score_loss < 1.2 {
        "~5d"
    } else if mean_score_loss < 1.7 {
        "~3d"
    } else if mean_score_loss < 2.2 {
        "~1d"
    } else if mean_score_loss < 2.8 {
        "~2k"
    } else if mean_score_loss < 3.5 {
        "~5k"
    } else if mean_score_loss < 4.5 {
        "~8k"
    } else if mean_score_loss < 6.0 {
        "~12k"
    } else {
        "~15k+"
    };
    Some(band.to_owned())
}

/// Classify a move by win-rate loss (win probability the move gave up).
fn classify_winrate(loss: f32) -> Classification {
    if loss < 0.03 {
        Classification::Good
    } else if loss < 0.07 {
        Classification::Inaccuracy
    } else if loss < 0.15 {
        Classification::Mistake
    } else {
        Classification::Blunder
    }
}

/// Classify a move by score loss (points the move gave up). Heuristic
/// thresholds in points — tune to taste. Points are a steadier yardstick than
/// win-rate across game phases: a win-rate near 0/1 hides real point losses in
/// a decided game, while an even game inflates tiny slips. (This is why OGS
/// offers a score-based review alongside the win-rate one.)
fn classify_score(points_lost: f32) -> Classification {
    if points_lost < 1.0 {
        Classification::Good
    } else if points_lost < 3.0 {
        Classification::Inaccuracy
    } else if points_lost < 6.0 {
        Classification::Mistake
    } else {
        Classification::Blunder
    }
}

/// Who is to move at `turn`.
///
/// Turn `i` is the position *before* move `i`, so the side to move is simply
/// the colour of move `i`. The one turn past the end of the move list belongs
/// to whoever would have played next. Derived from the move list rather than
/// assumed to alternate, because handicap games start with White and a game
/// record can contain consecutive moves by the same colour.
fn side_to_move_at(game: &ParsedGame, turn: usize) -> Color {
    game.moves.get(turn).map_or_else(
        || {
            game.moves
                .last()
                .map_or(Color::Black, |(color, _)| color.other())
        },
        |(color, _)| *color,
    )
}

/// The evaluations either side of a move by `color`, both in the mover's frame.
///
/// `before` is the mover's own turn; `after` is their opponent's — which is
/// why reading the second one as-is (or merely negating it) is wrong unless
/// the engine happens to report side-to-move.
fn move_evals(
    color: Color,
    before: Option<&RootInfo>,
    after: Option<&RootInfo>,
) -> (Option<Eval>, Option<Eval>) {
    (
        before.map(|root| root.as_seen_by(color, color)),
        after.map(|root| root.as_seen_by(color, color.other())),
    )
}

/// The turns a screening pass says deserve a full-depth search.
///
/// Both turns either side of every move whose screened evaluation swings by at
/// least `refine_winrate_delta` or `refine_score_delta` — in *either*
/// direction: an apparent gain is as much a sign that a shallow search misread
/// the position as an apparent loss. A move missing either evaluation is
/// refined too, rather than trusted.
pub fn turns_to_refine(
    game: &ParsedGame,
    screened: &[TurnResponse],
    cfg: &EngineConfig,
) -> BTreeSet<usize> {
    let roots: HashMap<usize, &RootInfo> = screened
        .iter()
        .filter_map(|turn| turn.root_info.as_ref().map(|root| (turn.turn_number, root)))
        .collect();

    let mut refine = BTreeSet::new();
    for (index, (color, _)) in game.moves.iter().enumerate() {
        let swings = match move_evals(
            *color,
            roots.get(&index).copied(),
            roots.get(&(index + 1)).copied(),
        ) {
            (Some(before), Some(after)) => {
                (before.winrate - after.winrate).abs() >= cfg.refine_winrate_delta
                    || (before.score_lead - after.score_lead).abs() >= cfg.refine_score_delta
            }
            _ => true,
        };
        if swings {
            refine.insert(index);
            refine.insert(index + 1);
        }
    }
    refine
}

/// Warn once if KataGo's `currentPlayer` disagrees with the move list.
///
/// A mismatch means the turn indices are skewed, which would attribute every
/// loss to the wrong player — silently, and with numbers that still look
/// plausible. Worth saying out loud rather than quietly correcting, because the
/// correction would depend on guessing the skew.
fn check_turn_alignment(game: &ParsedGame, by_turn: &HashMap<usize, TurnResponse>) {
    let mismatched = (0..=game.moves.len())
        .filter(|&turn| {
            by_turn
                .get(&turn)
                .and_then(|response| response.root_info.as_ref())
                .and_then(|root| root.current_player)
                .is_some_and(|reported| reported != side_to_move_at(game, turn))
        })
        .count();

    if mismatched > 0 {
        // KATAGO-ANNOTATE: the fields are in the message (the hosts print the message only).
        tracing::warn!(
            "KataGo's currentPlayer disagrees with the move list — per-move \
             losses may be attributed to the wrong player (mismatched={mismatched}, turns={})",
            game.moves.len() + 1
        );
    }
}

/// A per-turn grid read from every turn's response, encoded point by point.
///
/// Returns empty unless *every* turn carries a correctly sized array:
/// consumers index these by turn number, so a gap would silently shift a swing
/// onto the neighbouring move. All-or-nothing is the only safe contract.
/// `trailing` is how many entries KataGo appends past the board (the policy's
/// pass); they're checked for, then dropped.
fn per_turn_grid<T>(
    game: &ParsedGame,
    by_turn: &HashMap<usize, TurnResponse>,
    name: &str,
    field: impl Fn(&TurnResponse) -> Option<&Vec<f32>>,
    trailing: usize,
    encode: impl Fn(usize, f32) -> T,
) -> Vec<Vec<T>> {
    let points = usize::from(game.board_size) * usize::from(game.board_size);
    let expected = points + trailing;
    let mut per_turn = Vec::with_capacity(game.moves.len() + 1);

    for turn in 0..=game.moves.len() {
        let Some(raw) = by_turn.get(&turn).and_then(&field) else {
            return Vec::new();
        };
        if raw.len() != expected {
            // KATAGO-ANNOTATE: the fields are in the message (the hosts print the message only).
            tracing::warn!(
                "per-turn grid has the wrong length — dropping it for this game \
                 (turn={turn}, got={}, expected={expected}, grid={name})",
                raw.len()
            );
            return Vec::new();
        }
        per_turn.push(
            raw.iter()
                .take(points)
                .map(|&value| encode(turn, value))
                .collect(),
        );
    }

    per_turn
}

/// Ownership at every turn, normalised to Black-positive hundredths.
fn normalise_ownership(game: &ParsedGame, by_turn: &HashMap<usize, TurnResponse>) -> Vec<Vec<i8>> {
    per_turn_grid(
        game,
        by_turn,
        "ownership",
        |turn| turn.ownership.as_ref(),
        0,
        |turn, cell| {
            // The map is in the reported frame; flip it when that frame isn't
            // Black. An ownership array carries no `currentPlayer` of its own,
            // so the side to move comes from the move list.
            let reporter = REPORT_PERSPECTIVE.reporter(None, side_to_move_at(game, turn));
            let signed = if reporter == Color::Black {
                cell
            } else {
                -cell
            };
            // Hundredths keep every digit a classifier can use while making
            // the array an order of magnitude smaller than f32s.
            (signed * 100.0).round().clamp(-100.0, 100.0) as i8
        },
    )
}

/// Ownership spread at every turn, in hundredths.
fn ownership_stdev_grid(game: &ParsedGame, by_turn: &HashMap<usize, TurnResponse>) -> Vec<Vec<u8>> {
    per_turn_grid(
        game,
        by_turn,
        "ownership_stdev",
        |turn| turn.ownership_stdev.as_ref(),
        0,
        |_, stdev| (stdev * 100.0).round().clamp(0.0, 100.0) as u8,
    )
}

/// A policy probability as one byte: `round(255 · sqrt(p))`, KataGo's `-1`
/// (illegal) as `0`. The square root spreads the many tiny probabilities over
/// more of the range than a linear scale would.
fn encode_policy(prob: f32) -> u8 {
    if prob <= 0.0 {
        0
    } else {
        (255.0 * prob.sqrt()).round().clamp(0.0, 255.0) as u8
    }
}

/// A policy (raw or human) at every turn, encoded per [`encode_policy`].
fn policy_grid(
    game: &ParsedGame,
    by_turn: &HashMap<usize, TurnResponse>,
    name: &str,
    field: impl Fn(&TurnResponse) -> Option<&Vec<f32>>,
) -> Vec<Vec<u8>> {
    per_turn_grid(game, by_turn, name, field, 1, |_, prob| encode_policy(prob))
}

/// A move's probability in one turn's full policy array (pass included).
fn policy_at(policy: Option<&Vec<f32>>, mv: &str, board_size: u8) -> Option<f32> {
    let expected = usize::from(board_size) * usize::from(board_size) + 1;
    let policy = policy.filter(|policy| policy.len() == expected)?;
    sgf::policy_index(mv, board_size)
        .and_then(|index| policy.get(index))
        .map(|&prob| prob.max(0.0))
}

/// Assemble the analysis from the parsed game and KataGo's per-turn results.
///
/// `human_profile` is the profile the query carried, if any; it's recorded
/// only when the human policy actually came back.
pub fn assemble(
    game: &ParsedGame,
    turns: Vec<TurnResponse>,
    cfg: &EngineConfig,
    human_profile: Option<&str>,
) -> GameAnalysis {
    let by_turn: HashMap<usize, TurnResponse> = turns
        .into_iter()
        .map(|turn| (turn.turn_number, turn))
        .collect();

    check_turn_alignment(game, &by_turn);

    let mut moves = Vec::with_capacity(game.moves.len());
    let mut black = PlayerAccum::default();
    let mut white = PlayerAccum::default();

    for (index, (color, mv)) in game.moves.iter().enumerate() {
        // Both evaluations rotated into the mover's frame (see `move_evals`).
        let (before, after) = move_evals(
            *color,
            by_turn.get(&index).and_then(|turn| turn.root_info.as_ref()),
            by_turn
                .get(&(index + 1))
                .and_then(|turn| turn.root_info.as_ref()),
        );

        let best_winrate = before.map_or(0.5, |eval| eval.winrate);
        let winrate = after.map_or(best_winrate, |eval| eval.winrate);
        let winrate_loss = (best_winrate - winrate).max(0.0);
        let winrate_classification = classify_winrate(winrate_loss);

        let best_score = before.map_or(0.0, |eval| eval.score_lead);
        let resulting_score = after.map_or(best_score, |eval| eval.score_lead);
        let score_loss = (best_score - resulting_score).max(0.0);
        let score_classification = classify_score(score_loss);

        let turn = by_turn.get(&index);
        let top_moves: Vec<Candidate> = turn
            .map(|turn| {
                let mut infos: Vec<&_> = turn.move_infos.iter().collect();
                infos.sort_by_key(|info| info.order);
                infos
                    .into_iter()
                    .take(cfg.top_k)
                    .map(|info| {
                        // Candidates at this turn belong to the mover.
                        let eval = info.as_seen_by(*color, *color);
                        Candidate {
                            mv: info.mv.clone(),
                            winrate: eval.winrate,
                            score_lead: eval.score_lead,
                            prior: info.prior,
                            human_prior: info.human_prior,
                            visits: info.visits,
                            pv: info.pv.clone(),
                        }
                    })
                    .collect()
            })
            .unwrap_or_default();

        // The played move's priors come from the full arrays, which cover
        // every move; a candidate's own figures are the fallback when the
        // worker didn't return them.
        let played = top_moves.iter().find(|candidate| candidate.mv == *mv);
        let prior = policy_at(
            turn.and_then(|turn| turn.policy.as_ref()),
            mv,
            game.board_size,
        )
        .or_else(|| played.map(|candidate| candidate.prior));
        let human_prior = policy_at(
            turn.and_then(|turn| turn.human_policy.as_ref()),
            mv,
            game.board_size,
        )
        .or_else(|| played.and_then(|candidate| candidate.human_prior));

        let accum = match color {
            Color::Black => &mut black,
            Color::White => &mut white,
        };
        accum.record(
            winrate_loss,
            score_loss,
            winrate_classification,
            score_classification,
        );

        moves.push(MoveAnnotation {
            move_number: index + 1,
            color: *color,
            mv: mv.clone(),
            winrate,
            best_winrate,
            winrate_loss,
            winrate_classification,
            score_lead: best_score,
            score_loss,
            score_classification,
            prior,
            human_prior,
            top_moves,
        });
    }

    let ownership = normalise_ownership(game, &by_turn);
    let ownership_stdev = ownership_stdev_grid(game, &by_turn);
    let policy = policy_grid(game, &by_turn, "policy", |turn| turn.policy.as_ref());
    let human_policy = policy_grid(game, &by_turn, "human_policy", |turn| {
        turn.human_policy.as_ref()
    });
    let human_profile = human_profile
        .filter(|_| !human_policy.is_empty())
        .map(str::to_owned);

    GameAnalysis {
        board_size: game.board_size,
        komi: game.komi,
        rules: game.rules,
        result: game.result.clone(),
        moves,
        report: GameReport {
            num_moves: game.moves.len(),
            black: black.finish(game.board_size),
            white: white.finish(game.board_size),
        },
        payload_version: PAYLOAD_VERSION,
        engine: EngineInfo {
            // KATAGO-ANNOTATE: `name: "katago"` upstream; `version` is new.
            name: cfg.engine_name.clone(),
            version: cfg.engine_version.clone(),
            model: cfg.model.clone(),
            max_visits: cfg.max_visits,
            top_k: cfg.top_k,
            report_perspective: REPORT_PERSPECTIVE.as_katago_setting(),
            ownership: !ownership.is_empty(),
            ownership_stdev: !ownership_stdev.is_empty(),
            policy: !policy.is_empty(),
            human_model: human_profile.as_ref().map(|_| cfg.human_model.clone()),
            screen_visits: cfg.two_pass().then_some(cfg.screen_visits),
        },
        ownership,
        ownership_stdev,
        policy,
        human_policy,
        human_profile,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn game_one_move() -> ParsedGame {
        ParsedGame {
            board_size: 19,
            komi: 7.5,
            rules: Rules::Japanese,
            result: None,
            initial_stones: vec![],
            moves: vec![(Color::Black, "Q16".to_owned())],
        }
    }

    /// A two-move game on a 2×2 board — small enough that an ownership map is
    /// four readable numbers.
    fn tiny_game() -> ParsedGame {
        ParsedGame {
            board_size: 2,
            komi: 0.5,
            rules: Rules::Japanese,
            result: None,
            initial_stones: vec![],
            moves: vec![
                (Color::Black, "A2".to_owned()),
                (Color::White, "B1".to_owned()),
            ],
        }
    }

    fn cfg(top_k: usize) -> EngineConfig {
        EngineConfig {
            top_k,
            max_visits: 100,
            model: "test-model".to_owned(),
            human_model: "test-model-human".to_owned(),
            ..EngineConfig::default()
        }
    }

    /// Turn responses for [`tiny_game`], in side-to-move perspective, with an
    /// ownership map per turn. `owner` values are in `[-1, 1]`.
    fn tiny_turns(ownership: [Option<Vec<f32>>; 3]) -> Vec<TurnResponse> {
        let players = ["B", "W", "B"];
        ownership
            .into_iter()
            .enumerate()
            .map(|(turn, owner)| {
                serde_json::from_value(serde_json::json!({
                    "id": "q0",
                    "turnNumber": turn,
                    "rootInfo": {
                        "winrate": 0.5, "scoreLead": 0.0,
                        "currentPlayer": players[turn], "visits": 10
                    },
                    "moveInfos": [],
                    "ownership": owner,
                }))
                .expect("fixture matches the response types")
            })
            .collect()
    }

    #[test]
    fn computes_loss_from_consecutive_turns() {
        // Turn 0 (Black to move): best winrate 0.60, best score lead +2.0.
        // Turn 1 (White to move): White winrate 0.55 → Black's actual = 0.45;
        //   White score lead +1.0 → Black's resulting lead = -1.0.
        // winrate loss = 0.60 - 0.45 = 0.15 → blunder.
        // score loss   = 2.0 - (-1.0) = 3.0 → mistake.
        // (Same move, two lenses, different verdicts — the whole point.)
        let turns: Vec<TurnResponse> = vec![
            serde_json::from_value(serde_json::json!({
                "id":"q0","turnNumber":0,
                "rootInfo":{"winrate":0.60,"scoreLead":2.0,"currentPlayer":"B","visits":100},
                "moveInfos":[{"move":"D4","winrate":0.60,"scoreLead":2.0,"prior":0.2,"visits":80,"order":0,"pv":["D4","Q16"]}]
            })).unwrap(),
            serde_json::from_value(serde_json::json!({
                "id":"q0","turnNumber":1,
                "rootInfo":{"winrate":0.55,"scoreLead":1.0,"currentPlayer":"W","visits":100},
                "moveInfos":[]
            })).unwrap(),
        ];

        let analysis = assemble(&game_one_move(), turns, &cfg(5), None);
        assert_eq!(analysis.moves.len(), 1);
        let mv = &analysis.moves[0];
        assert_eq!(mv.color, Color::Black);
        assert_eq!(mv.mv, "Q16");
        assert!((mv.best_winrate - 0.60).abs() < 1e-5);
        assert!((mv.winrate - 0.45).abs() < 1e-5);
        assert!((mv.winrate_loss - 0.15).abs() < 1e-5);
        assert_eq!(mv.winrate_classification, Classification::Blunder);
        assert!((mv.score_loss - 3.0).abs() < 1e-5);
        assert_eq!(mv.score_classification, Classification::Mistake);
        // The move lands in Black's report under each lens.
        assert_eq!(analysis.report.num_moves, 1);
        assert_eq!(analysis.report.black.moves, 1);
        assert_eq!(analysis.report.white.moves, 0);
        assert_eq!(analysis.report.black.by_winrate.blunder, 1);
        assert_eq!(analysis.report.black.by_score.mistake, 1);
        assert_eq!(mv.top_moves.len(), 1);
        assert_eq!(mv.top_moves[0].mv, "D4");
    }

    /// [`tiny_game`] turns carrying every v3 field: stdev, the raw policy and
    /// the human policy (each with KataGo's trailing pass entry), and one
    /// candidate per turn with a human prior.
    fn tiny_turns_v3() -> Vec<TurnResponse> {
        let players = ["B", "W", "B"];
        (0..3)
            .map(|turn| {
                serde_json::from_value(serde_json::json!({
                    "id": "q0",
                    "turnNumber": turn,
                    "rootInfo": {"winrate": 0.5, "scoreLead": 0.0, "currentPlayer": players[turn]},
                    "moveInfos": [{
                        "move": "B2", "winrate": 0.5, "scoreLead": 0.0,
                        "prior": 0.25, "humanPrior": 0.5, "visits": 10, "order": 0
                    }],
                    "ownership": [1.0, 0.0, 0.0, -1.0],
                    "ownershipStdev": [0.0, 0.123, 0.5, 1.0],
                    // A2, B2, A1, B1, pass. A2 is illegal from turn 1 on.
                    "policy": if turn == 0 {
                        serde_json::json!([0.64, 0.25, 0.01, 0.0, 0.1])
                    } else {
                        serde_json::json!([-1.0, 0.25, 0.01, 0.64, 0.1])
                    },
                    "humanPolicy": [0.0, 0.5, 0.25, 0.04, 0.21],
                }))
                .unwrap()
            })
            .collect()
    }

    #[test]
    fn v3_grids_are_encoded_per_turn() {
        let analysis = assemble(&tiny_game(), tiny_turns_v3(), &cfg(5), Some("rank_5k"));
        assert_eq!(analysis.payload_version, 3);

        assert_eq!(analysis.ownership_stdev.len(), 3, "one grid per turn");
        assert_eq!(analysis.ownership_stdev[0], vec![0, 12, 50, 100]);

        // round(255·√p), the pass entry dropped, illegal (-1) as 0.
        assert_eq!(analysis.policy.len(), 3);
        assert_eq!(analysis.policy[0], vec![204, 128, 26, 0]);
        assert_eq!(analysis.policy[1], vec![0, 128, 26, 204]);
        assert_eq!(analysis.human_policy[2], vec![0, 180, 128, 51]);
        assert_eq!(analysis.human_profile.as_deref(), Some("rank_5k"));

        assert!(analysis.engine.ownership_stdev && analysis.engine.policy);
        assert_eq!(
            analysis.engine.human_model.as_deref(),
            Some("test-model-human")
        );
    }

    /// The played move's priors come from the full arrays, so a move outside
    /// the candidate list still gets them; the candidates keep their own.
    #[test]
    fn the_played_move_gets_its_priors_from_the_full_policy() {
        let analysis = assemble(&tiny_game(), tiny_turns_v3(), &cfg(5), Some("rank_5k"));
        // Move 1, Black A2 — not the candidate (B2): straight from the arrays.
        let first = &analysis.moves[0];
        assert_eq!(first.prior, Some(0.64));
        assert_eq!(first.human_prior, Some(0.0));
        // Move 2, White B1.
        assert_eq!(analysis.moves[1].prior, Some(0.64));
        assert_eq!(analysis.moves[1].human_prior, Some(0.04));
        assert_eq!(first.top_moves[0].human_prior, Some(0.5));

        // Without the arrays, a played move that is a candidate keeps the
        // candidate's figures, and one that isn't has none.
        let mut turns = tiny_turns_v3();
        for turn in &mut turns {
            turn.policy = None;
            turn.human_policy = None;
            turn.move_infos[0].mv = "A2".to_owned();
        }
        let analysis = assemble(&tiny_game(), turns, &cfg(5), Some("rank_5k"));
        assert_eq!(analysis.moves[0].prior, Some(0.25));
        assert_eq!(analysis.moves[0].human_prior, Some(0.5));
        assert_eq!(analysis.moves[1].prior, None);
        assert!(analysis.policy.is_empty() && analysis.human_policy.is_empty());
        assert_eq!(analysis.human_profile, None, "no human policy, no profile");
    }

    /// Like ownership, a grid with a gap or a wrong length is dropped whole —
    /// and one grid going missing leaves the others alone.
    #[test]
    fn each_v3_grid_is_all_or_nothing_on_its_own() {
        let mut turns = tiny_turns_v3();
        turns[1].ownership_stdev = None;
        turns[2].policy = Some(vec![0.5, 0.5, 0.0, 0.0]); // no pass entry
        let analysis = assemble(&tiny_game(), turns, &cfg(5), Some("rank_5k"));
        assert!(analysis.ownership_stdev.is_empty());
        assert!(!analysis.engine.ownership_stdev);
        assert!(analysis.policy.is_empty());
        assert_eq!(analysis.human_policy.len(), 3);
        assert_eq!(analysis.ownership.len(), 3);
    }

    /// Without a profile there are no human fields at all, not empty ones.
    #[test]
    fn no_profile_no_human_fields() {
        let mut turns = tiny_turns_v3();
        for turn in &mut turns {
            turn.human_policy = None;
            turn.move_infos[0].human_prior = None;
        }
        let analysis = assemble(&tiny_game(), turns, &cfg(5), None);
        let json = serde_json::to_value(&analysis).unwrap();
        assert!(json.get("human_policy").is_none());
        assert!(json.get("human_profile").is_none());
        assert!(json["engine"].get("human_model").is_none());
        assert!(json["moves"][0].get("human_prior").is_none());
        assert!(
            json["moves"][0]["top_moves"][0]
                .get("human_prior")
                .is_none()
        );
        assert!(json["moves"][0].get("prior").is_some());
    }

    /// The exact JSON names and shapes a v3 payload ships, and that a reader
    /// written for v2 — the shape `baduk_review`'s `katago-sdk` mirrors, with
    /// no `deny_unknown_fields` — still reads every field it knows from it.
    #[test]
    fn a_v2_reader_reads_a_v3_payload() {
        #[derive(serde::Deserialize)]
        struct CandidateV2 {
            mv: String,
            prior: f32,
            visits: u32,
            #[serde(default)]
            pv: Vec<String>,
        }
        #[derive(serde::Deserialize)]
        struct MoveV2 {
            move_number: usize,
            mv: String,
            score_loss: f32,
            #[serde(default)]
            top_moves: Vec<CandidateV2>,
        }
        #[derive(serde::Deserialize)]
        struct EngineV2 {
            name: String,
            max_visits: u32,
            #[serde(default)]
            ownership: bool,
        }
        #[derive(serde::Deserialize)]
        struct AnalysisV2 {
            board_size: u8,
            moves: Vec<MoveV2>,
            report: serde_json::Value,
            payload_version: u32,
            #[serde(default)]
            engine: Option<EngineV2>,
            #[serde(default)]
            ownership: Vec<Vec<i8>>,
        }

        let v3 = assemble(&tiny_game(), tiny_turns_v3(), &cfg(5), Some("rank_5k"));
        let json = serde_json::to_value(&v3).unwrap();

        // The v3 additions, by their wire names and shapes.
        let cells = 4;
        for grid in ["ownership_stdev", "policy", "human_policy"] {
            let turns = json[grid].as_array().unwrap();
            assert_eq!(turns.len(), 3, "{grid}: one per turn");
            assert!(
                turns
                    .iter()
                    .all(|turn| turn.as_array().unwrap().len() == cells)
            );
        }
        assert_eq!(json["human_profile"], "rank_5k");
        assert!(json["moves"][0]["prior"].is_number());
        assert!(json["moves"][0]["human_prior"].is_number());
        assert!(json["moves"][0]["top_moves"][0]["human_prior"].is_number());
        assert_eq!(json["engine"]["ownership_stdev"], true);
        assert_eq!(json["engine"]["policy"], true);

        let v2: AnalysisV2 = serde_json::from_value(json).expect("a v2 reader reads v3");
        assert_eq!(v2.payload_version, 3);
        assert_eq!(v2.board_size, 2);
        assert_eq!(v2.ownership, v3.ownership);
        assert_eq!(v2.moves.len(), 2);
        assert_eq!(v2.moves[1].move_number, 2);
        assert_eq!(v2.moves[1].mv, "B1");
        assert!(v2.moves[0].score_loss >= 0.0);
        assert_eq!(v2.moves[0].top_moves[0].mv, "B2");
        assert!((v2.moves[0].top_moves[0].prior - 0.25).abs() < 1e-6);
        assert_eq!(v2.moves[0].top_moves[0].visits, 10);
        assert!(v2.moves[0].top_moves[0].pv.is_empty());
        assert!(v2.report.get("black").is_some());
        let engine = v2.engine.unwrap();
        assert_eq!((engine.name.as_str(), engine.max_visits), ("katago", 100));
        assert!(engine.ownership);
    }

    #[test]
    fn move_accuracy_is_high_for_best_moves_and_low_for_blunders() {
        // No win-rate given up ⇒ ~100% accurate.
        assert!(move_accuracy(0.0) > 99.0);
        // A heavy 30-point win-prob drop ⇒ well below half.
        assert!(move_accuracy(0.30) < 30.0);
        // Monotone: giving up more is never more accurate.
        assert!(move_accuracy(0.05) > move_accuracy(0.20));
        // Always within bounds.
        assert!((0.0..=100.0).contains(&move_accuracy(1.0)));
    }

    #[test]
    fn rank_estimate_gated_by_board_size_and_move_count() {
        // Too few moves, or a non-19×19 board ⇒ no estimate.
        assert_eq!(estimate_rank(19, RANK_MIN_MOVES - 1, 1.0), None);
        assert_eq!(estimate_rank(9, 200, 1.0), None);
        // A strong, full-size game ⇒ a dan band; a sloppy one ⇒ a kyu band.
        assert_eq!(estimate_rank(19, 200, 0.9).as_deref(), Some("~5d"));
        assert_eq!(estimate_rank(19, 200, 5.0).as_deref(), Some("~12k"));
    }

    /// The 9×9 symptom from the README: `best_winrate < winrate` on every move,
    /// so the loss clamps to zero and a whole game scores as flawless. Whatever
    /// else changes, a mover's own best line can never be worse than what they
    /// actually got.
    #[test]
    fn a_move_never_beats_the_best_line_available_to_it() {
        let turns = tiny_turns([None, None, None]);
        let analysis = assemble(&tiny_game(), turns, &cfg(5), None);

        for mv in &analysis.moves {
            assert!(
                mv.best_winrate >= mv.winrate,
                "move {} claims a better result than the best line",
                mv.move_number
            );
            assert!(mv.winrate_loss >= 0.0 && mv.score_loss >= 0.0);
        }
    }

    /// The 19×19 symptom: `score_loss ≈ 2 × score_lead`, rising with the lead,
    /// because the two evaluations were differenced across opposite frames. A
    /// commanding, *stable* lead must produce no loss at all.
    #[test]
    fn a_steady_large_lead_is_not_read_as_a_huge_loss() {
        // Black is +12 before and after their move — they gave up nothing.
        let turns: Vec<TurnResponse> = vec![
            serde_json::from_value(serde_json::json!({
                "id":"q0","turnNumber":0,
                "rootInfo":{"winrate":0.95,"scoreLead":12.0,"currentPlayer":"B","visits":100},
                "moveInfos":[]
            }))
            .unwrap(),
            // Same position from White's side: they are 12 points down.
            serde_json::from_value(serde_json::json!({
                "id":"q0","turnNumber":1,
                "rootInfo":{"winrate":0.05,"scoreLead":-12.0,"currentPlayer":"W","visits":100},
                "moveInfos":[]
            }))
            .unwrap(),
        ];

        let analysis = assemble(&game_one_move(), turns, &cfg(5), None);
        let mv = &analysis.moves[0];

        assert!(
            mv.score_loss.abs() < 1e-5,
            "a steady +12 lead was read as a {}-point loss",
            mv.score_loss
        );
        assert!(mv.winrate_loss.abs() < 1e-5);
        assert_eq!(mv.score_classification, Classification::Good);
    }

    /// Ownership is reported in the engine's frame and stored Black-positive,
    /// so a map taken on White's turn comes back negated.
    #[test]
    fn ownership_is_normalised_to_black_positive_hundredths() {
        let turns = tiny_turns([
            // Turn 0 — Black to move, so already Black-positive.
            Some(vec![1.0, 0.5, -0.5, -1.0]),
            // Turn 1 — White to move, so these are White's numbers.
            Some(vec![1.0, 0.5, -0.5, -1.0]),
            Some(vec![0.0, 0.0, 0.0, 0.0]),
        ]);
        let analysis = assemble(&tiny_game(), turns, &cfg(5), None);

        assert_eq!(analysis.ownership.len(), 3, "one map per turn");
        assert_eq!(analysis.ownership[0], vec![100, 50, -50, -100]);
        assert_eq!(
            analysis.ownership[1],
            vec![-100, -50, 50, 100],
            "White's frame is flipped into Black's"
        );
        assert!(analysis.engine.ownership);
        assert_eq!(analysis.payload_version, PAYLOAD_VERSION);
    }

    /// Consumers index ownership by turn number, so one missing or mis-sized
    /// map must drop the whole set rather than shift every later swing onto the
    /// wrong move.
    #[test]
    fn a_gap_or_a_wrong_length_drops_ownership_entirely() {
        let missing = tiny_turns([
            Some(vec![1.0, 0.0, 0.0, 0.0]),
            None,
            Some(vec![0.0, 0.0, 0.0, 0.0]),
        ]);
        let analysis = assemble(&tiny_game(), missing, &cfg(5), None);
        assert!(analysis.ownership.is_empty());
        assert!(!analysis.engine.ownership);

        let short = tiny_turns([
            Some(vec![1.0, 0.0, 0.0, 0.0]),
            Some(vec![1.0, 0.0]),
            Some(vec![0.0, 0.0, 0.0, 0.0]),
        ]);
        let analysis = assemble(&tiny_game(), short, &cfg(5), None);
        assert!(analysis.ownership.is_empty());
    }

    /// Turn `i` is the position before move `i`, and the turn past the end
    /// belongs to whoever would play next. Derived from the move list so a
    /// handicap game (White first) and a record with consecutive same-colour
    /// moves both come out right.
    #[test]
    fn the_side_to_move_follows_the_move_list_not_an_assumed_alternation() {
        let game = tiny_game();
        assert_eq!(side_to_move_at(&game, 0), Color::Black);
        assert_eq!(side_to_move_at(&game, 1), Color::White);
        assert_eq!(
            side_to_move_at(&game, 2),
            Color::Black,
            "past the last move"
        );

        let handicap = ParsedGame {
            moves: vec![
                (Color::White, "D4".to_owned()),
                (Color::White, "Q16".to_owned()),
            ],
            ..tiny_game()
        };
        assert_eq!(side_to_move_at(&handicap, 0), Color::White);
        assert_eq!(side_to_move_at(&handicap, 1), Color::White);
        assert_eq!(side_to_move_at(&handicap, 2), Color::Black);
    }

    /// Provenance travels with the payload so two analyses of the same game at
    /// different visit counts are never silently compared.
    #[test]
    fn the_payload_records_what_produced_it() {
        let analysis = assemble(
            &game_one_move(),
            tiny_turns([None, None, None]),
            &cfg(7),
            None,
        );
        assert_eq!(analysis.engine.name, "katago");
        assert_eq!(analysis.engine.model, "test-model");
        assert_eq!(analysis.engine.max_visits, 100);
        assert_eq!(analysis.engine.top_k, 7);
        assert_eq!(analysis.engine.report_perspective, "SIDETOMOVE");
    }

    /// Screened turns for [`tiny_game`] (B, W, B to move), each `(winrate,
    /// score_lead)` from the side to move; `None` drops the turn's root.
    fn screened(evals: [Option<(f32, f32)>; 3]) -> Vec<TurnResponse> {
        let players = ["B", "W", "B"];
        evals
            .into_iter()
            .enumerate()
            .map(|(turn, eval)| {
                let root = eval.map(|(winrate, score)| {
                    serde_json::json!({
                        "winrate": winrate, "scoreLead": score,
                        "currentPlayer": players[turn], "visits": 16
                    })
                });
                serde_json::from_value(serde_json::json!({
                    "id": "q0", "turnNumber": turn, "rootInfo": root, "moveInfos": []
                }))
                .unwrap()
            })
            .collect()
    }

    #[test]
    fn quiet_moves_are_not_refined() {
        let turns = screened([Some((0.5, 0.0)), Some((0.51, -0.4)), Some((0.5, 0.2))]);
        assert!(turns_to_refine(&tiny_game(), &turns, &cfg(5)).is_empty());
    }

    #[test]
    fn a_swing_refines_both_sides_of_the_move() {
        // White's move (turn 1 → 2) hands Black a won game: refine turns 1 and 2.
        let turns = screened([Some((0.5, 0.0)), Some((0.5, 0.0)), Some((0.9, 5.0))]);
        let refine: Vec<_> = turns_to_refine(&tiny_game(), &turns, &cfg(5))
            .into_iter()
            .collect();
        assert_eq!(refine, vec![1, 2]);
    }

    #[test]
    fn a_score_swing_alone_triggers_refinement() {
        // Win rate barely moves (a decided game) but two points change hands.
        let turns = screened([Some((0.99, 20.0)), Some((0.01, -18.0)), Some((0.99, 18.0))]);
        let refine: Vec<_> = turns_to_refine(&tiny_game(), &turns, &cfg(5))
            .into_iter()
            .collect();
        assert_eq!(refine, vec![0, 1]);
    }

    #[test]
    fn an_apparent_gain_or_a_missing_eval_is_refined_too() {
        // Black's move "gains" 0.3 for Black — a misread, not a free lunch.
        let gain = screened([Some((0.5, 0.0)), Some((0.2, -3.0)), Some((0.8, 3.0))]);
        assert!(turns_to_refine(&tiny_game(), &gain, &cfg(5)).contains(&0));

        let missing = screened([Some((0.5, 0.0)), None, Some((0.5, 0.0))]);
        let refine: Vec<_> = turns_to_refine(&tiny_game(), &missing, &cfg(5))
            .into_iter()
            .collect();
        assert_eq!(refine, vec![0, 1, 2]);
    }

    #[test]
    fn the_payload_records_the_screening_depth_only_when_it_ran() {
        let turns = tiny_turns([None, None, None]);
        let single = assemble(&tiny_game(), turns.clone(), &cfg(5), None);
        assert_eq!(single.engine.screen_visits, None);

        let two_pass = EngineConfig {
            screen_visits: 16,
            ..cfg(5)
        };
        let analysis = assemble(&tiny_game(), turns, &two_pass, None);
        assert_eq!(analysis.engine.screen_visits, Some(16));
    }
}
