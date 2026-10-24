// Copied from katago-ws src/engine/katago.rs @ bdd674bb9431af2bac8e2e5dfd7d2eb00e5a06f3 (see docs/M6-report.md).
// Kept: the protocol types (Perspective, REPORT_PERSPECTIVE, Eval, RootInfo, MoveInfo, TurnResponse,
// AnalysisQuery, OverrideSettings), the query `KataGo::analyze` builds, `handle_line`'s reading of a response
// line, and the tests. Dropped: the subprocess actor (tokio: KataGo, Owner, Launcher, Process, ...), which is
// IO; the hosts (katago-ws, katago-wasm) supply their own. Divergences are marked `KATAGO-ANNOTATE:`.

//! KataGo's analysis protocol as katago-ws speaks it: the query it sends for a
//! game and the per-turn responses it reads back.
//!
//! The protocol is asynchronous and id-keyed: one query (a whole game, all
//! turns) yields one response line per analyzed turn, possibly interleaved with
//! other in-flight queries. Driving the engine is the host's job (see
//! [`crate::review::Analyzer`]); this module holds what every host shares.

use serde::{Deserialize, Serialize};
use serde_json::Value;

// KATAGO-ANNOTATE: `EngineConfig` -> `AnnotateOptions` (the analysis knobs only).
use crate::options::AnnotateOptions;
use crate::sgf::{Color, ColoredMove, ParsedGame, Rules};

/// Whose point of view KataGo's numbers are in.
///
/// The single most consequential setting in this pipeline, and not what you'd
/// assume. KataGo's docs: *"All values will be from the perspective of
/// `reportAnalysisWinratesAs` as specified in the analysis config file"* — and
/// the stock `analysis_example.cfg` this image ships sets it to `BLACK`.
///
/// Reading a fixed-perspective evaluation as though it alternated with the side
/// to move is what produced the two symptoms the tests below pin down: a
/// win-rate loss that clamps to zero on every move, and a score loss that comes
/// out at roughly twice the score lead.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(
    dead_code,
    reason = "the three variants model KataGo's three settings, not our choices — \
              only SIDETOMOVE is pinned today, but `reporter` handles all three so \
              that changing REPORT_PERSPECTIVE stays a one-line, correct edit"
)]
pub enum Perspective {
    /// Every value is Black's.
    Black,
    /// Every value is White's.
    White,
    /// Each value belongs to whoever is to move at that position.
    SideToMove,
}

impl Perspective {
    /// The `reportAnalysisWinratesAs` value that produces this perspective.
    #[must_use]
    pub fn as_katago_setting(self) -> &'static str {
        match self {
            Self::Black => "BLACK",
            Self::White => "WHITE",
            Self::SideToMove => "SIDETOMOVE",
        }
    }

    /// Whose evaluation a reading at this position is, given who is to move.
    ///
    /// `current_player` is KataGo's own claim about the position, preferred
    /// when present; pass `None` for a reading that carries none (an ownership
    /// map, say) to fall back on the caller's knowledge of the move list.
    #[must_use]
    pub fn reporter(self, current_player: Option<Color>, side_to_move: Color) -> Color {
        match self {
            Self::Black => Color::Black,
            Self::White => Color::White,
            // KataGo emits `currentPlayer`; the caller's value is only a
            // fallback for a build that stops doing so.
            Self::SideToMove => current_player.unwrap_or(side_to_move),
        }
    }
}

/// The perspective this service pins on every query, and therefore the frame
/// responses are interpreted in.
///
/// One constant drives both the `overrideSettings` we send and the rotation the
/// annotator applies, so the two cannot drift apart. That drift is precisely
/// how the original bug survived: the config file said one thing, the code
/// assumed another, and nothing connected them.
pub const REPORT_PERSPECTIVE: Perspective = Perspective::SideToMove;

/// A position evaluation in a known player's frame.
#[derive(Debug, Clone, Copy)]
pub struct Eval {
    /// Win probability for that player, in `[0, 1]`.
    pub winrate: f32,
    /// Expected score lead in points for that player.
    pub score_lead: f32,
}

impl Eval {
    /// The same evaluation seen from the other side. Win rate and score lead
    /// are both zero-sum.
    fn flipped(self) -> Self {
        Self {
            winrate: 1.0 - self.winrate,
            score_lead: -self.score_lead,
        }
    }
}

/// KataGo `rootInfo` for one analyzed turn (position evaluation).
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RootInfo {
    /// Win probability in the [`Perspective`] the engine was told to report in
    /// — **not** necessarily the side to move. Use [`RootInfo::as_seen_by`]
    /// rather than reading this directly.
    pub winrate: f32,
    /// Expected score lead (points), in the same perspective as `winrate`.
    #[serde(default)]
    pub score_lead: f32,
    /// Who is to move at this position. Optional so an engine build that stops
    /// emitting it degrades to the caller's fallback rather than failing a job.
    #[serde(default)]
    pub current_player: Option<Color>,
}

impl RootInfo {
    /// This evaluation rotated into `viewer`'s frame.
    #[must_use]
    pub fn as_seen_by(&self, viewer: Color, side_to_move: Color) -> Eval {
        let eval = Eval {
            winrate: self.winrate,
            score_lead: self.score_lead,
        };
        if REPORT_PERSPECTIVE.reporter(self.current_player, side_to_move) == viewer {
            eval
        } else {
            eval.flipped()
        }
    }
}

/// KataGo `moveInfos` entry — one candidate move.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MoveInfo {
    /// Move in GTP notation. (Not camelCase — KataGo's key is literally `move`.)
    #[serde(rename = "move")]
    pub mv: String,
    /// Win probability after this move, in the reported [`Perspective`] — the
    /// same caveat as [`RootInfo::winrate`]. Use [`MoveInfo::as_seen_by`].
    pub winrate: f32,
    /// Score lead after this move, in the same perspective as `winrate`.
    #[serde(default)]
    pub score_lead: f32,
    /// Raw policy prior.
    #[serde(default)]
    pub prior: f32,
    /// Visits to this move.
    #[serde(default)]
    pub visits: u32,
    /// The human SL network's prior for this move, under the query's
    /// `humanSLProfile`. Present only when a human model is loaded and a
    /// profile was given.
    #[serde(default)]
    pub human_prior: Option<f32>,
    /// Rank among candidates (0 = best).
    #[serde(default)]
    pub order: u32,
    /// Principal variation following this move.
    #[serde(default)]
    pub pv: Vec<String>,
}

impl MoveInfo {
    /// This candidate's evaluation rotated into `viewer`'s frame.
    ///
    /// Candidates for turn `i` belong to the player to move at turn `i`, which
    /// under a fixed-perspective report is not who the numbers are stated for.
    /// Rotating them matters beyond cosmetics: a consumer comparing a user's
    /// move against these candidates would otherwise rank them backwards.
    #[must_use]
    pub fn as_seen_by(&self, viewer: Color, side_to_move: Color) -> Eval {
        let eval = Eval {
            winrate: self.winrate,
            score_lead: self.score_lead,
        };
        if REPORT_PERSPECTIVE.reporter(Some(side_to_move), side_to_move) == viewer {
            eval
        } else {
            eval.flipped()
        }
    }
}

/// One analyzed turn of a game.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TurnResponse {
    /// Which turn this is (0 = before the first move).
    #[serde(default)]
    pub turn_number: usize,
    /// Position evaluation. (`Option` → absent `rootInfo` deserializes to `None`.)
    pub root_info: Option<RootInfo>,
    /// Candidate moves, best first.
    #[serde(default)]
    pub move_infos: Vec<MoveInfo>,
    /// Expected ownership of every intersection, present only when
    /// `includeOwnership` is on. Length `boardYSize * boardXSize`, row-major
    /// from the top-left (A19 → T1), values in `[-1, 1]` in the reported
    /// [`Perspective`] — so it needs the same rotation as everything else.
    #[serde(default)]
    pub ownership: Option<Vec<f32>>,
    /// Standard deviation of ownership across the search tree, present only
    /// when `includeOwnershipStdev` is on. Same layout as `ownership`, values
    /// in `[0, 1]` — a spread, so it has no perspective to rotate.
    #[serde(default)]
    pub ownership_stdev: Option<Vec<f32>>,
    /// The network's raw policy, present only when `includePolicy` is on.
    /// Length `boardYSize * boardXSize + 1`: the board row-major from the
    /// top-left, then pass; `-1` marks an illegal move. Always the side to
    /// move's — a policy has no other frame.
    #[serde(default)]
    pub policy: Option<Vec<f32>>,
    /// The human SL network's policy under the query's `humanSLProfile`, in
    /// `policy`'s layout. Present only with `includePolicy`, a human model and
    /// a profile.
    #[serde(default)]
    pub human_policy: Option<Vec<f32>>,
}

/// A KataGo analysis query — the request side of the JSON protocol, modelled as
/// a typed struct (like the [`TurnResponse`] response types) rather than an
/// ad-hoc `json!` blob, so field names and types are checked at compile time.
/// `id` is filled in by the host (katago-ws: its owner task) so the correlation
/// counter stays local.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AnalysisQuery {
    pub id: String,
    pub rules: Rules,
    pub komi: f32,
    pub board_x_size: u8,
    pub board_y_size: u8,
    pub initial_stones: Vec<ColoredMove>,
    pub moves: Vec<ColoredMove>,
    pub analyze_turns: Vec<usize>,
    pub max_visits: u32,
    // camelCase would give `analysisPvLen`; KataGo wants `PV` upper-cased.
    #[serde(rename = "analysisPVLen")]
    pub analysis_pv_len: u32,
    pub include_ownership: bool,
    pub include_ownership_stdev: bool,
    pub include_policy: bool,
    /// Scheduling priority within KataGo's queue (higher runs first).
    pub priority: i32,
    /// Per-query settings that override the analysis config file.
    pub override_settings: OverrideSettings,
}

/// Query-level overrides of `analysis.cfg`.
///
/// The perspective is pinned here rather than trusted from the config file: the
/// config this image ships says `BLACK`, the annotator wants a frame it chose,
/// and a query-level override is the one place the two cannot disagree.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct OverrideSettings {
    pub report_analysis_winrates_as: &'static str,
    /// Whose play the human SL network imitates (`rank_5k`, …). It belongs
    /// here: KataGo ignores it on the outer query object. Sent only with a
    /// human model loaded — KataGo rejects the whole query otherwise.
    #[serde(rename = "humanSLProfile", skip_serializing_if = "Option::is_none")]
    pub human_sl_profile: Option<String>,
}

impl OverrideSettings {
    pub fn new(human_profile: Option<&str>) -> Self {
        Self {
            report_analysis_winrates_as: REPORT_PERSPECTIVE.as_katago_setting(),
            human_sl_profile: human_profile.map(str::to_owned),
        }
    }
}

impl AnalysisQuery {
    /// The query for `turns` of a game at `max_visits` each, as
    /// `KataGo::analyze` builds it (`id` left empty for the host to assign).
    ///
    /// `priority` orders this query against others in KataGo's queue (higher
    /// first; turns of equal priority run in submission order).
    /// `human_profile` must only be given when a human model is loaded:
    /// KataGo fails the whole query otherwise.
    ///
    /// KATAGO-ANNOTATE: the first half of `KataGo::analyze` (katago.rs), the
    /// second half (submit, await the responses) being the host's.
    #[must_use]
    pub fn new(
        game: &ParsedGame,
        cfg: &AnnotateOptions,
        turns: Vec<usize>,
        max_visits: u32,
        priority: i32,
        human_profile: Option<&str>,
    ) -> Self {
        // `id` is assigned by the owner task (keeps the correlation counter local).
        AnalysisQuery {
            id: String::new(),
            rules: game.rules,
            komi: game.komi,
            board_x_size: game.board_size,
            board_y_size: game.board_size,
            initial_stones: game.initial_stones.clone(),
            moves: game.moves.clone(),
            analyze_turns: turns,
            max_visits,
            analysis_pv_len: cfg.analysis_pv_len,
            include_ownership: cfg.include_ownership,
            include_ownership_stdev: cfg.include_ownership_stdev,
            include_policy: cfg.include_policy,
            priority,
            override_settings: OverrideSettings::new(human_profile),
        }
    }
}

/// What one line of the engine's output is, read as `handle_line` reads it.
///
/// KATAGO-ANNOTATE: `handle_line` (katago.rs) split from its pending table:
/// the host routes a [`ResponseLine::Turn`] or [`ResponseLine::Error`] to the
/// query with that id.
#[derive(Debug)]
pub enum ResponseLine {
    /// Not JSON (`unparsable line`).
    Unparsable(String),
    /// `{"error": ...}`, for a query when it carries an `id`.
    Error { id: Option<String>, error: String },
    /// `{"warning": ...}`: logged and otherwise ignored.
    Warning { id: Option<String>, warning: String },
    /// A line with no `id` (and no error): ignored.
    NoId,
    /// An `id`-keyed line that does not read as a [`TurnResponse`] (`unparsable response`).
    UnparsableResponse { id: String, error: String },
    /// One analyzed turn.
    Turn { id: String, turn: Box<TurnResponse> },
}

impl ResponseLine {
    /// Read one response line (`handle_line`'s parsing, in its order).
    #[must_use]
    pub fn parse(line: &str) -> Self {
        let value: Value = match serde_json::from_str(line) {
            Ok(parsed) => parsed,
            Err(err) => return Self::Unparsable(err.to_string()),
        };

        let id = value
            .get("id")
            .and_then(|val| val.as_str())
            .map(str::to_owned);

        if let Some(err) = value.get("error").and_then(|val| val.as_str()) {
            return Self::Error {
                id,
                error: err.to_owned(),
            };
        }
        if let Some(warn) = value.get("warning").and_then(|val| val.as_str()) {
            return Self::Warning {
                id,
                warning: warn.to_owned(),
            };
        }

        let Some(id) = id else { return Self::NoId };
        match serde_json::from_value::<TurnResponse>(value) {
            Ok(turn) => Self::Turn {
                id,
                turn: Box::new(turn),
            },
            Err(err) => Self::UnparsableResponse {
                id,
                error: err.to_string(),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn root(winrate: f32, score_lead: f32, current_player: Option<Color>) -> RootInfo {
        RootInfo {
            winrate,
            score_lead,
            current_player,
        }
    }

    /// The test that would have caught the original bug.
    ///
    /// The annotator's arithmetic is only correct if KataGo actually reports in
    /// the frame the annotator assumes. Nothing in the config file guarantees
    /// that — the shipped `analysis_example.cfg` says `BLACK` — so the setting
    /// is pinned per query. If this assertion ever fails, every per-move loss
    /// this service reports is wrong.
    #[test]
    fn every_query_pins_the_reporting_perspective() {
        let query = AnalysisQuery {
            id: "q0".to_owned(),
            rules: Rules::Japanese,
            komi: 7.5,
            board_x_size: 19,
            board_y_size: 19,
            initial_stones: vec![],
            moves: vec![],
            analyze_turns: vec![0],
            max_visits: 100,
            analysis_pv_len: 10,
            include_ownership: true,
            include_ownership_stdev: true,
            include_policy: true,
            priority: 0,
            override_settings: OverrideSettings::new(None),
        };

        let json = serde_json::to_value(&query).expect("the query serialises");
        assert_eq!(
            json["overrideSettings"]["reportAnalysisWinratesAs"],
            "SIDETOMOVE"
        );
        // The pin and the rotation are driven by one constant; they cannot drift.
        assert_eq!(
            json["overrideSettings"]["reportAnalysisWinratesAs"],
            REPORT_PERSPECTIVE.as_katago_setting()
        );
    }

    /// The human profile goes inside `overrideSettings` — KataGo silently
    /// ignores it on the outer object — and is left out entirely without one,
    /// since a worker with no human model fails any query that carries it.
    #[test]
    fn the_human_profile_rides_in_override_settings_only() {
        let query = |profile| AnalysisQuery {
            id: "q0".to_owned(),
            rules: Rules::Japanese,
            komi: 6.5,
            board_x_size: 19,
            board_y_size: 19,
            initial_stones: vec![],
            moves: vec![],
            analyze_turns: vec![0],
            max_visits: 10,
            analysis_pv_len: 15,
            include_ownership: true,
            include_ownership_stdev: true,
            include_policy: true,
            priority: 0,
            override_settings: OverrideSettings::new(profile),
        };

        let with = serde_json::to_value(query(Some("rank_5k"))).unwrap();
        assert_eq!(with["overrideSettings"]["humanSLProfile"], "rank_5k");
        assert!(with.get("humanSLProfile").is_none());
        // KataGo's spellings, which camelCase alone would get wrong for PV.
        assert_eq!(with["includeOwnershipStdev"], true);
        assert_eq!(with["includePolicy"], true);
        assert_eq!(with["analysisPVLen"], 15);

        let without = serde_json::to_value(query(None)).unwrap();
        assert!(without["overrideSettings"].get("humanSLProfile").is_none());
    }

    /// The v3 fields as KataGo v1.16.5 sends them (shape checked against a
    /// real run: `policy`/`humanPolicy` are `size² + 1` with `-1` illegal,
    /// `ownershipStdev` is `size²`, `humanPrior` sits on each candidate).
    #[test]
    fn a_turn_response_parses_policy_stdev_and_human_fields() {
        let turn: TurnResponse = serde_json::from_value(serde_json::json!({
            "id": "q0", "turnNumber": 1, "isDuringSearch": false,
            "rootInfo": {"winrate": 0.5, "scoreLead": 0.0, "currentPlayer": "W"},
            "moveInfos": [{"move": "A1", "winrate": 0.5, "prior": 0.4, "humanPrior": 0.3, "order": 0}],
            "ownership": [0.1, 0.2, 0.3, 0.4],
            "ownershipStdev": [0.01, 0.02, 0.03, 0.04],
            "policy": [0.4, -1.0, 0.3, 0.2, 0.1],
            "humanPolicy": [0.3, -1.0, 0.3, 0.3, 0.1]
        }))
        .unwrap();
        assert_eq!(turn.ownership_stdev.as_ref().map(Vec::len), Some(4));
        assert_eq!(turn.policy.as_ref().map(Vec::len), Some(5));
        assert_eq!(turn.human_policy.as_ref().map(Vec::len), Some(5));
        assert_eq!(turn.move_infos[0].human_prior, Some(0.3));

        let bare: TurnResponse = serde_json::from_value(serde_json::json!({
            "id": "q0", "turnNumber": 0, "moveInfos": [{"move": "A1", "winrate": 0.5}]
        }))
        .unwrap();
        assert!(bare.policy.is_none() && bare.human_policy.is_none());
        assert!(bare.ownership_stdev.is_none());
        assert_eq!(bare.move_infos[0].human_prior, None);
    }

    /// Win rate and score lead are zero-sum, so the same position seen from the
    /// other side is `1 - winrate` and `-score_lead` — and seeing it from your
    /// own side must change nothing.
    #[test]
    fn an_evaluation_rotates_into_either_players_frame() {
        let info = root(0.70, 5.0, Some(Color::Black));

        let black = info.as_seen_by(Color::Black, Color::Black);
        assert!((black.winrate - 0.70).abs() < 1e-6);
        assert!((black.score_lead - 5.0).abs() < 1e-6);

        let white = info.as_seen_by(Color::White, Color::Black);
        assert!((white.winrate - 0.30).abs() < 1e-6);
        assert!((white.score_lead + 5.0).abs() < 1e-6);
    }

    /// `currentPlayer` is what makes the rotation independent of any assumption
    /// about alternation; the caller's value is used only when it is absent.
    #[test]
    fn the_engines_own_current_player_wins_over_the_callers_guess() {
        // The engine says White is to move; the caller wrongly guessed Black.
        let info = root(0.70, 5.0, Some(Color::White));
        let eval = info.as_seen_by(Color::White, Color::Black);
        assert!(
            (eval.winrate - 0.70).abs() < 1e-6,
            "the engine's claim should have identified White as the reporter"
        );

        // With no claim, the caller's knowledge of the move list is all there is.
        let blind = root(0.70, 5.0, None);
        let eval = blind.as_seen_by(Color::White, Color::White);
        assert!((eval.winrate - 0.70).abs() < 1e-6);
        let eval = blind.as_seen_by(Color::White, Color::Black);
        assert!((eval.winrate - 0.30).abs() < 1e-6);
    }

    /// Why the pin matters, shown in numbers.
    ///
    /// These are the README's 19×19 figures: Black leads by 12 points and gives
    /// up nothing. Read as Black-perspective data — which is what the engine
    /// sends by default — a blind `-score_lead` on the following turn turns a
    /// steady lead into a 24-point "loss", exactly the `2 × score_lead` pattern
    /// observed. Rotating with a known perspective gives zero.
    #[test]
    fn misreading_the_perspective_doubles_the_lead_into_a_loss() {
        let before = 12.0_f32;
        // Turn i+1 in BLACK perspective: still +12 for Black.
        let after_reported_as_black = 12.0_f32;

        let naive_loss = before - (-after_reported_as_black);
        assert!(
            (naive_loss - 24.0).abs() < 1e-6,
            "the old arithmetic is the README's 2 x score_lead symptom"
        );

        // The same reading, rotated knowing it is Black's: Black gave up nothing.
        let info = root(0.95, after_reported_as_black, Some(Color::Black));
        let correct = before - info.as_seen_by(Color::Black, Color::Black).score_lead;
        assert!((correct).abs() < 1e-6);
    }

    /// The rotation is correct under *any* of the three settings, not only the
    /// one pinned — so a deliberate change to [`REPORT_PERSPECTIVE`] stays
    /// correct, and the pin is a second line of defence rather than the only one.
    #[test]
    fn the_reporter_is_identified_under_every_setting() {
        // Black to move, and the engine says so.
        let claim = Some(Color::Black);
        assert_eq!(
            Perspective::Black.reporter(claim, Color::Black),
            Color::Black
        );
        assert_eq!(
            Perspective::White.reporter(claim, Color::Black),
            Color::White,
            "a WHITE-perspective report is White's whoever is to move"
        );
        assert_eq!(
            Perspective::SideToMove.reporter(claim, Color::Black),
            Color::Black
        );

        // With White to move, only SIDETOMOVE changes its answer.
        let claim = Some(Color::White);
        assert_eq!(
            Perspective::Black.reporter(claim, Color::White),
            Color::Black
        );
        assert_eq!(
            Perspective::SideToMove.reporter(claim, Color::White),
            Color::White
        );

        // Each maps to the config value that produces it.
        assert_eq!(Perspective::Black.as_katago_setting(), "BLACK");
        assert_eq!(Perspective::White.as_katago_setting(), "WHITE");
        assert_eq!(Perspective::SideToMove.as_katago_setting(), "SIDETOMOVE");
    }

    /// Ownership only appears when it was asked for, and an engine that adds
    /// fields must not break jobs already in flight.
    #[test]
    fn a_turn_response_parses_with_and_without_ownership() {
        let bare: TurnResponse = serde_json::from_value(serde_json::json!({
            "id": "q0", "turnNumber": 3,
            "rootInfo": {"winrate": 0.5, "scoreLead": 0.0, "currentPlayer": "B"},
            "moveInfos": [], "somethingNew": 42
        }))
        .expect("unknown fields are tolerated");
        assert!(bare.ownership.is_none());
        assert_eq!(
            bare.root_info.and_then(|root| root.current_player),
            Some(Color::Black)
        );

        let owned: TurnResponse = serde_json::from_value(serde_json::json!({
            "id": "q0", "turnNumber": 0,
            "rootInfo": {"winrate": 0.5, "scoreLead": 0.0, "currentPlayer": "W"},
            "moveInfos": [], "ownership": [0.5, -0.5, 1.0, -1.0]
        }))
        .expect("an ownership map parses");
        assert_eq!(
            owned.ownership.as_deref(),
            Some([0.5, -0.5, 1.0, -1.0].as_slice())
        );
    }
}
