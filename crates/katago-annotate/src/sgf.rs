// Copied from katago-ws src/engine/sgf.rs @ bdd674bb9431af2bac8e2e5dfd7d2eb00e5a06f3 (see docs/M6-report.md).
// Divergences (each marked `KATAGO-ANNOTATE:`): `AppError`/`AppResult` -> `AnnotateError`/`AnnotateResult`
// (same variants and messages); [`GameInput`] (a game given as JSON instead of an SGF) is new.

//! SGF → KataGo query data.
//!
//! KataGo's analysis engine takes a move list + metadata (not SGF), so we parse
//! the SGF with `sgf-parse` (typed Go properties) and emit GTP-style
//! coordinates it understands.

use serde::{Deserialize, Serialize};
use sgf_parse::go::{self, Move, Point, Prop};

// KATAGO-ANNOTATE: katago-ws's AppError, reduced to the variants this crate returns.
use crate::error::{AnnotateError as AppError, AnnotateResult as AppResult};

/// Stone colour. Serializes to KataGo's shorthand (`"B"` / `"W"`), which is also
/// how it appears in the analysis result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Color {
    /// Black.
    #[serde(rename = "B")]
    Black,
    /// White.
    #[serde(rename = "W")]
    White,
}

impl Color {
    /// The other player.
    ///
    /// Every per-move figure is a difference between two evaluations taken one
    /// turn apart, and those turns belong to opposite players — so flipping
    /// colour is the most common operation in the annotator.
    #[must_use]
    pub fn other(self) -> Self {
        match self {
            Self::Black => Self::White,
            Self::White => Self::Black,
        }
    }
}

/// Scoring ruleset for KataGo. Serializes to KataGo's shorthand
/// (`"japanese"` / `"chinese"`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Rules {
    /// Japanese (territory) scoring.
    Japanese,
    /// Chinese (area) scoring.
    Chinese,
}

/// A colour + location pair as KataGo expects it: `(Color::Black, "Q4")` /
/// `(Color::White, "pass")`.
pub type ColoredMove = (Color, String);

/// Parsed game ready to turn into a KataGo analysis query.
// KATAGO-ANNOTATE: derives added (the browser host clones and prints games).
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedGame {
    /// Board size (square; KataGo gets this as both X and Y size).
    pub board_size: u8,
    /// Komi.
    pub komi: f32,
    /// Ruleset KataGo should score with.
    pub rules: Rules,
    /// Raw SGF result string, if present.
    pub result: Option<String>,
    /// Pre-placed stones (handicap / setup), as GTP coordinates.
    pub initial_stones: Vec<ColoredMove>,
    /// Moves in play order, as GTP coordinates.
    pub moves: Vec<ColoredMove>,
}

/// Properties that decide the position KataGo replays. A malformed one of these
/// is fatal; a malformed anything-else is ignored like any unused property.
const GAME_PROPS: &[&str] = &["B", "W", "AB", "AW", "AE", "SZ", "KM"];

/// GTP column letters; `I` is skipped. Also caps the board size we accept.
const GTP_LETTERS: &[u8] = b"ABCDEFGHJKLMNOPQRSTUVWXYZ";

/// Convert a 0-indexed SGF point (origin top-left) to a GTP string (column
/// letter skipping `I`, row counted from the bottom). `board_size` is needed to
/// flip the row to bottom-origin.
fn to_gtp(point: Point, board_size: u8) -> AppResult<String> {
    if point.x >= board_size || point.y >= board_size {
        return Err(AppError::Sgf(format!(
            "point ({}, {}) is off the {board_size}x{board_size} board",
            point.x, point.y
        )));
    }
    let letter = GTP_LETTERS
        .get(usize::from(point.x))
        .map_or('?', |&letter| char::from(letter));
    let row = board_size - point.y;
    Ok(format!("{letter}{row}"))
}

/// GTP location of a move. `tt` is the pre-FF4 pass on boards up to 19x19.
fn move_to_gtp(mv: Move, board_size: u8) -> AppResult<String> {
    match mv {
        Move::Pass => Ok("pass".to_owned()),
        Move::Move(Point { x: 19, y: 19 }) if board_size <= 19 => Ok("pass".to_owned()),
        Move::Move(point) => to_gtp(point, board_size),
    }
}

/// Where a GTP move (as [`parse`] emits them) sits in KataGo's policy array:
/// the board row-major from the top-left (A19 → T1), then pass at
/// `board_size²`. `None` for anything off the board or unreadable.
pub fn policy_index(mv: &str, board_size: u8) -> Option<usize> {
    let size = usize::from(board_size);
    if mv.eq_ignore_ascii_case("pass") {
        return Some(size * size);
    }
    let (&letter, row) = mv.as_bytes().split_first()?;
    let col = GTP_LETTERS
        .iter()
        .position(|&gtp| gtp == letter.to_ascii_uppercase())?;
    let row: usize = std::str::from_utf8(row).ok()?.parse().ok()?;
    (col < size && (1..=size).contains(&row)).then(|| (size - row) * size + col)
}

/// Check that an SGF would parse into a replayable [`ParsedGame`], without
/// keeping the result. Used at submit time so a malformed game is rejected up
/// front instead of failing in the worker. Board size and komi defaults don't
/// affect validity, so any values do.
pub fn validate(sgf: &str) -> AppResult<()> {
    parse(sgf, 19, 7.5).map(drop)
}

/// Parse an SGF into a [`ParsedGame`], following the main variation.
///
/// Rejects anything we'd otherwise have to guess at, since a guess desyncs the
/// board and KataGo only notices (as an "illegal move") after the job took an
/// engine slot: game properties `sgf-parse` couldn't type (e.g. `B[ab][bb]`, a
/// move with several points), B and W in one node, setup stones after the first
/// move (KataGo's query has nowhere to put them), and non-square boards.
pub fn parse(sgf: &str, default_size: u8, default_komi: f32) -> AppResult<ParsedGame> {
    let trees = go::parse(sgf).map_err(|err| AppError::Sgf(err.to_string()))?;
    let root = trees
        .first()
        .ok_or_else(|| AppError::Sgf("no game tree".to_owned()))?;

    let mut size = None;
    let mut komi = None;
    let mut rules = Rules::Japanese;
    let mut result = None;
    let mut setup: Vec<(Color, Point)> = Vec::new();
    let mut moves: Vec<(Color, Move)> = Vec::new();

    for (index, node) in root.main_variation().enumerate() {
        let mut node_move = None;
        for prop in node.properties() {
            match prop {
                // Malformed metadata (e.g. `AP[GoGoD95]`, no `:version`) is
                // harmless; only reject what shapes the position.
                Prop::Invalid(identifier, values) if GAME_PROPS.contains(&identifier.as_str()) => {
                    return Err(AppError::Sgf(format!(
                        "malformed property {identifier}[{}] in node {index}",
                        values.join("][")
                    )));
                }
                Prop::SZ((width, height)) => {
                    if width != height {
                        return Err(AppError::Sgf(format!(
                            "non-square board {width}x{height} is not supported"
                        )));
                    }
                    size = Some(*width);
                }
                Prop::KM(value) => komi = Some(*value as f32),
                Prop::RU(rule) => {
                    // Japanese for anything KataGo doesn't model here.
                    rules = if rule.text.trim().eq_ignore_ascii_case("chinese") {
                        Rules::Chinese
                    } else {
                        Rules::Japanese
                    };
                }
                Prop::RE(outcome) => result = Some(outcome.text.clone()),
                Prop::AB(points) | Prop::AW(points) => {
                    if !moves.is_empty() {
                        return Err(AppError::Sgf(format!(
                            "setup stones after move {} are not supported",
                            moves.len()
                        )));
                    }
                    let color = if matches!(prop, Prop::AB(_)) {
                        Color::Black
                    } else {
                        Color::White
                    };
                    setup.extend(points.iter().map(|&point| (color, point)));
                }
                Prop::AE(points) if !points.is_empty() => {
                    return Err(AppError::Sgf(
                        "removing stones (AE) is not supported".to_owned(),
                    ));
                }
                Prop::B(mv) | Prop::W(mv) => {
                    if node_move.is_some() {
                        return Err(AppError::Sgf(format!(
                            "node {index} has more than one move"
                        )));
                    }
                    let color = if matches!(prop, Prop::B(_)) {
                        Color::Black
                    } else {
                        Color::White
                    };
                    node_move = Some((color, *mv));
                }
                _ => {}
            }
        }
        moves.extend(node_move);
    }

    let board_size = size.unwrap_or(default_size);
    if board_size == 0 || usize::from(board_size) > GTP_LETTERS.len() {
        return Err(AppError::Sgf(format!(
            "board size {board_size} is not supported"
        )));
    }
    // `AB`/`AW` are sets; sort so the KataGo query is deterministic.
    setup.sort_by_key(|&(color, point)| (color == Color::White, point.y, point.x));

    Ok(ParsedGame {
        board_size,
        komi: komi.unwrap_or(default_komi),
        rules,
        result,
        initial_stones: setup
            .into_iter()
            .map(|(color, point)| Ok((color, to_gtp(point, board_size)?)))
            .collect::<AppResult<_>>()?,
        moves: moves
            .into_iter()
            .map(|(color, mv)| Ok((color, move_to_gtp(mv, board_size)?)))
            .collect::<AppResult<_>>()?,
    })
}

/// KATAGO-ANNOTATE (new): a game given as moves rather than an SGF, as
/// `reviewGame({moves, rules, komi, ...})` takes it in the browser: KataGo's
/// own query fields (`boardXSize`/`boardYSize` or `boardSize`, `initialStones`,
/// `moves` as `[["B", "Q16"], ...]`), checked like [`parse`] checks an SGF.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct GameInput {
    /// Board size (square); `boardXSize`/`boardYSize` are accepted when equal.
    #[serde(default)]
    pub board_size: Option<u8>,
    #[serde(default)]
    pub board_x_size: Option<u8>,
    #[serde(default)]
    pub board_y_size: Option<u8>,
    #[serde(default)]
    pub komi: Option<f32>,
    /// `japanese` (the default, as for an SGF without `RU`) or `chinese`.
    #[serde(default)]
    pub rules: Option<Rules>,
    /// The game's result, copied into the payload.
    #[serde(default)]
    pub result: Option<String>,
    /// Pre-placed stones (handicap / setup), GTP coordinates.
    #[serde(default)]
    pub initial_stones: Vec<ColoredMove>,
    /// Moves in play order, GTP coordinates or `pass`.
    pub moves: Vec<ColoredMove>,
}

impl GameInput {
    /// The [`ParsedGame`] this describes, with `default_size` / `default_komi`
    /// for what it leaves out. Coordinates are normalised to [`parse`]'s form
    /// (`Q16`, `pass`).
    ///
    /// # Errors
    ///
    /// [`AppError::Sgf`] for anything [`parse`] would reject (board size, a
    /// point off the board, a pass among the initial stones).
    pub fn into_game(self, default_size: u8, default_komi: f32) -> AppResult<ParsedGame> {
        let size = match (self.board_size, self.board_x_size, self.board_y_size) {
            (size, None, None) => size,
            (None, Some(width), Some(height)) if width == height => Some(width),
            (Some(size), Some(width), Some(height)) if size == width && width == height => {
                Some(size)
            }
            (_, width, height) => {
                return Err(AppError::Sgf(format!(
                    "non-square or inconsistent board size (boardXSize {width:?}, boardYSize {height:?}) is not supported"
                )));
            }
        };
        let board_size = size.unwrap_or(default_size);
        if board_size == 0 || usize::from(board_size) > GTP_LETTERS.len() {
            return Err(AppError::Sgf(format!(
                "board size {board_size} is not supported"
            )));
        }
        let point = |what: &str, index: usize, (color, mv): ColoredMove, pass_ok: bool| {
            let at = policy_index(&mv, board_size).ok_or_else(|| {
                AppError::Sgf(format!(
                    "{what} {index} ({mv}) is not a point of the {board_size}x{board_size} board"
                ))
            })?;
            let size = usize::from(board_size);
            if at == size * size {
                return if pass_ok {
                    Ok((color, "pass".to_owned()))
                } else {
                    Err(AppError::Sgf(format!("{what} {index} is a pass")))
                };
            }
            let letter = char::from(GTP_LETTERS[at % size]);
            Ok((color, format!("{letter}{}", size - at / size)))
        };
        Ok(ParsedGame {
            board_size,
            komi: self.komi.unwrap_or(default_komi),
            rules: self.rules.unwrap_or(Rules::Japanese),
            result: self.result,
            initial_stones: self
                .initial_stones
                .into_iter()
                .enumerate()
                .map(|(index, stone)| point("initial stone", index, stone, false))
                .collect::<AppResult<_>>()?,
            moves: self
                .moves
                .into_iter()
                .enumerate()
                .map(|(index, mv)| point("move", index + 1, mv, true))
                .collect::<AppResult<_>>()?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The inverse of the row flip in [`to_gtp`], checked at the corners and
    /// across the skipped `I`.
    #[test]
    fn gtp_moves_map_onto_the_policy_array() {
        assert_eq!(policy_index("A19", 19), Some(0));
        assert_eq!(policy_index("T19", 19), Some(18));
        assert_eq!(policy_index("A1", 19), Some(342));
        assert_eq!(policy_index("T1", 19), Some(360));
        assert_eq!(policy_index("J19", 19), Some(8), "J is the ninth column");
        assert_eq!(policy_index("pass", 19), Some(361));
        assert_eq!(policy_index("E5", 9), Some(40), "tengen on 9x9");
        assert_eq!(policy_index("I5", 9), None);
        assert_eq!(policy_index("K5", 9), None, "off a 9x9 board");
        assert_eq!(policy_index("A0", 9), None);
        assert_eq!(policy_index("", 9), None);
    }

    #[test]
    fn parses_metadata_and_gtp_moves() {
        let sgf = "(;GM[1]SZ[9]KM[6.5]RU[Japanese];B[ee];W[gg];B[cc])";
        let game = parse(sgf, 19, 7.5).unwrap();
        assert_eq!(game.board_size, 9);
        assert!((game.komi - 6.5).abs() < f32::EPSILON);
        assert_eq!(game.rules, Rules::Japanese);
        assert_eq!(game.moves.len(), 3);
        // ee on 9x9: col 5 -> 'E', line 5 -> row 9+1-5 = 5  => "E5", Black.
        assert_eq!(game.moves[0], (Color::Black, "E5".to_owned()));
        // gg: col 7 -> 'G', line 7 -> row 3 => "G3", White.
        assert_eq!(game.moves[1], (Color::White, "G3".to_owned()));
    }

    #[test]
    fn applies_defaults_when_size_and_komi_absent() {
        let sgf = "(;GM[1];B[pd];W[dp])";
        let game = parse(sgf, 19, 7.5).unwrap();
        assert_eq!(game.board_size, 19);
        assert!((game.komi - 7.5).abs() < f32::EPSILON);
        // pd on 19x19: col 16 -> 'Q', line 4 -> row 16 => "Q16".
        assert_eq!(game.moves[0], (Color::Black, "Q16".to_owned()));
    }

    #[test]
    fn rejects_move_node_with_several_coordinates() {
        // Shape of OGS game 65078272: a B node used as a setup position.
        let sgf = "(;GM[1]SZ[19];B[ab][bb][hc]AW[ba][cb];W[gb];B[fb])";
        let err = parse(sgf, 19, 7.5).err().unwrap();
        assert!(matches!(err, AppError::Sgf(_)), "{err}");
        assert!(err.to_string().contains("B[ab][bb][hc]"), "{err}");
        assert!(validate(sgf).is_err());
    }

    #[test]
    fn root_setup_becomes_initial_stones() {
        let sgf = "(;GM[1]SZ[9]HA[2]AB[cc][gg];W[ee];B[tt])";
        let game = parse(sgf, 19, 7.5).unwrap();
        assert_eq!(
            game.initial_stones,
            [
                (Color::Black, "C7".to_owned()),
                (Color::Black, "G3".to_owned()),
            ]
        );
        assert_eq!(game.moves[0], (Color::White, "E5".to_owned()));
        // `tt` is a pass on boards up to 19x19.
        assert_eq!(game.moves[1], (Color::Black, "pass".to_owned()));
    }

    #[test]
    fn ignores_malformed_metadata() {
        // AP should be `name:version`; it doesn't affect the game.
        let game = parse("(;GM[1]AP[GoGoD95];B[pd])", 19, 7.5).unwrap();
        assert_eq!(game.moves.len(), 1);
    }

    #[test]
    fn rejects_setup_after_first_move() {
        assert!(parse("(;GM[1];B[pd];W[dp]AB[aa])", 19, 7.5).is_err());
    }

    #[test]
    fn rejects_off_board_move() {
        assert!(parse("(;GM[1]SZ[9];B[jj])", 19, 7.5).is_err());
    }

    #[test]
    fn rejects_garbage() {
        assert!(parse("not sgf", 19, 7.5).is_err());
    }

    // KATAGO-ANNOTATE (new): games given as moves.
    #[test]
    fn a_game_input_becomes_the_same_game_as_its_sgf() {
        let sgf = parse("(;GM[1]SZ[9]KM[6.5]HA[2]AB[cc][gg];W[ee];B[tt])", 19, 7.5).unwrap();
        let input: GameInput = serde_json::from_value(serde_json::json!({
            "boardXSize": 9, "boardYSize": 9, "komi": 6.5,
            "initialStones": [["B", "c7"], ["B", "G3"]],
            "moves": [["W", "E5"], ["B", "PASS"]]
        }))
        .unwrap();
        assert_eq!(input.into_game(19, 7.5).unwrap(), sgf);
    }

    #[test]
    fn a_game_input_is_checked_like_an_sgf() {
        let game = |value: serde_json::Value| {
            serde_json::from_value::<GameInput>(value)
                .map_err(|err| AppError::Sgf(err.to_string()))
                .and_then(|input| input.into_game(19, 7.5))
        };
        let defaults = game(serde_json::json!({"moves": [["B", "Q16"]]})).unwrap();
        assert_eq!((defaults.board_size, defaults.rules), (19, Rules::Japanese));
        assert!((defaults.komi - 7.5).abs() < f32::EPSILON);
        assert!(game(serde_json::json!({"boardSize": 9, "moves": [["B", "K5"]]})).is_err());
        assert!(game(serde_json::json!({"boardSize": 9, "moves": [["B", "I5"]]})).is_err());
        assert!(game(serde_json::json!({"boardXSize": 9, "boardYSize": 13, "moves": []})).is_err());
        assert!(game(serde_json::json!({"boardSize": 26, "moves": []})).is_err());
        assert!(game(serde_json::json!({"initialStones": [["B", "pass"]], "moves": []})).is_err());
        assert!(game(serde_json::json!({"rules": "aga", "moves": []})).is_err());
        assert!(game(serde_json::json!({"moves": [["X", "D4"]]})).is_err());
        assert!(game(serde_json::json!({"moves": [], "komy": 6.5})).is_err());
    }
}
