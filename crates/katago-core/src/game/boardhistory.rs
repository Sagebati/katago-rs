// Ported from KataGo cpp/game/boardhistory.cpp, cpp/game/boardhistory.h @ fd0723fdbc0e9d82cf269c9630af8c27c57c07c4

use crate::core::global::KataError;
use crate::core::hash::Hash;
use crate::core::hash::Hash128;
use crate::game::board::*;
use crate::game::rules::Rules;
use crate::port::cpp;
use crate::port::cpp::cmp_from_lt;
use crate::port::fmt::cpp_ostream_f64;
use crate::testAssert;

/// upstream: BoardHistory::EncoreKoCapture (boardhistory.h)
//Used to implement once-only rules for ko captures in encore
/// PORT NOTE: STRUCT_NAMED_TRIPLE(Hash128,posHashBeforeMove,Loc,moveLoc,Player,movePla,EncoreKoCapture)
/// -> plain struct + new (PORTING.md §3, OWNERSHIP.tsv).
#[derive(Clone, Copy, Debug, Default)]
pub struct EncoreKoCapture {
  pub pos_hash_before_move: Hash128,
  pub move_loc: Loc,
  pub move_pla: Player,
}

impl EncoreKoCapture {
  /// upstream: BoardHistory::EncoreKoCapture::EncoreKoCapture (boardhistory.h)
  pub fn new(pos_hash_before_move: Hash128, move_loc: Loc, move_pla: Player) -> EncoreKoCapture {
    EncoreKoCapture { pos_hash_before_move, move_loc, move_pla }
  }
}

/// upstream: BoardHistoryModes (boardhistory.h)
//Bundles the flags that control how certain computations are performed.
//These exist to support migrations of KataGo's rules implementation - each mode selects between a
//legacy behavior (false) and a new behavior (true), with neural nets declaring which behavior they
//were trained with.
//They are PRESERVED by BoardHistory::clear() - they apply to the whole session of usage of a history
//object rather than being per-game state.
//Use BoardHistoryModes() for plain all-legacy behavior, or an appropriately resolved value
//(e.g. from Search::resolveHistoryModes or a governing history/book/game).
#[derive(Clone, Copy, Debug)]
pub struct BoardHistoryModes {
  //If true, all computations of pass-alive area (for game-ending, final scoring, and nn features)
  //are performed as if multi-stone suicide were legal, regardless of rules.multiStoneSuicideLegal.
  //Actual move legality is NOT affected.
  pub always_compute_pass_alive_under_suicide_rules: bool,

  //If true, under territory scoring with TaxRule NONE, empty points adjacent to any chain in atari
  //(e.g. unfilled ko mouths in seki) do not count as territory, per rules version 3, avoiding pass
  //fights over kos in seki. If false, such points count as under rules version 2. Also applies to
  //nn featurization of the territory. No effect under any other scoring/tax rules combination.
  pub exclude_territory_adjacent_to_atari: bool,
}

impl BoardHistoryModes {
  /// upstream: BoardHistoryModes::BoardHistoryModes (boardhistory.cpp)
  pub fn new() -> BoardHistoryModes {
    BoardHistoryModes {
      always_compute_pass_alive_under_suicide_rules: false,
      exclude_territory_adjacent_to_atari: false,
    }
  }

  /// upstream: BoardHistoryModes::BoardHistoryModes (boardhistory.cpp)
  // PORT NOTE: (bool alwaysPassAliveSuicide, bool excludeTerritoryAdjAtari) -> new_full (PORTING.md Appendix B).
  pub fn new_full(always_pass_alive_suicide: bool, exclude_territory_adj_atari: bool) -> BoardHistoryModes {
    BoardHistoryModes {
      always_compute_pass_alive_under_suicide_rules: always_pass_alive_suicide,
      exclude_territory_adjacent_to_atari: exclude_territory_adj_atari,
    }
  }
}

/// upstream: BoardHistoryModes::operator==, BoardHistoryModes::operator!= (boardhistory.cpp)
/// PORT NOTE: `!=` is upstream's `!(*this == other)`, which is PartialEq's default `ne`.
impl PartialEq for BoardHistoryModes {
  fn eq(&self, other: &BoardHistoryModes) -> bool {
    self.always_compute_pass_alive_under_suicide_rules == other.always_compute_pass_alive_under_suicide_rules
      && self.exclude_territory_adjacent_to_atari == other.exclude_territory_adjacent_to_atari
  }
}

/// upstream: BoardHistory (boardhistory.h)
//A data structure enabling checking of move legality, including optionally superko,
//and implements scoring and support for various rulesets (see rules.h)
pub struct BoardHistory {
  pub rules: Rules,

  //Chronological history of moves
  pub move_history: Vec<Move>,
  //Chronological history of preventEncore, for ability to replay a board history
  pub prevent_encore_history: Vec<bool>,
  //Chronological history of hashes, including the latest board's hash.
  //Theses are the hashes that determine whether a board is the "same" or not given the rules
  //(e.g. they include the player if situational superko, and not if positional)
  //Cleared on a pass if passes clear ko bans
  pub ko_hash_history: Vec<Hash128>,
  //The index of the first turn for which we have a koHashHistory (since depending on rules, passes may clear it).
  //Index 0 = starting state, index 1 = state after move 0, index 2 = state after move 1, etc...
  pub first_turn_idx_with_ko_history: usize,

  //The board and player to move as of the very start, before moveHistory.
  pub initial_board: Board,
  pub initial_pla: Player,
  pub initial_encore_phase: i32,
  //The "turn number" as of the initial board. Does not affect any rules, but possibly uses may
  //care about this number, for cases where we set up a position from midgame.
  pub initial_turn_number: i64,
  //How we count handicap at the start of the game. Set manually by some close-to-user-level apps or subcommands
  pub assume_multiple_starting_black_moves_are_handicap: bool,
  pub white_has_moved: bool,
  pub override_num_handicap_stones: i32,

  //Modes controlling how derived computations are performed (see BoardHistoryModes).
  //Deliberately preserved by clear() - these are modes applying to the whole session of usage of
  //this history object rather than per-game state, set only via setModes.
  pub modes: BoardHistoryModes,

  pub recent_boards: [Board; BoardHistory::NUM_RECENT_BOARDS as usize],
  pub current_recent_board_idx: i32,
  pub presumed_next_move_pla: Player,

  //Did this board location ever have a stone there before, or was it ever played?
  //(Also includes locations of suicides)
  pub was_ever_occupied_or_played: [bool; Board::MAX_ARR_SIZE as usize],
  //Locations where the next player is not allowed to play due to superko
  pub super_ko_banned: [bool; Board::MAX_ARR_SIZE as usize],

  //Number of consecutive passes made that count for ending the game or phase
  pub consecutive_ending_passes: i32,
  //All ko hashes from which a player passed
  pub hashes_before_black_pass: Vec<Hash128>,
  pub hashes_before_white_pass: Vec<Hash128>,

  //Encore phase 0,1,2 for territory scoring
  pub encore_phase: i32,
  //How many turns of history do we have in the current main or encore phase?
  pub num_turns_this_phase: i32,
  //What's the longest suffix of the history we can include that still probably obey the rules
  //for this phase? Checks for if the game continued past passes that should have ended the game
  //but does not check self-capture rules or ko/superko violations
  //that were tolerated in a game record.
  pub num_approx_valid_turns_this_phase: i32,
  //Similar to numApproxValidTurnsThisPhase but resets to 0 upon a rules violation rather than game phase change.
  pub num_consec_valid_turns_this_game: i32,

  //Ko-recapture-block locations for territory scoring in encore
  pub ko_recap_blocked: [bool; Board::MAX_ARR_SIZE as usize],
  pub ko_recap_block_hash: Hash128, //Hash contribution from ko-recap-block locations in encore.

  //Used to implement once-only rules for ko captures in encore
  pub ko_captures_in_encore: Vec<EncoreKoCapture>,

  //State of the grid as of the start of encore phase 2 for territory scoring
  pub second_encore_start_colors: [Color; Board::MAX_ARR_SIZE as usize],

  //Amount that should be added to komi
  pub white_bonus_score: f32,
  pub white_handicap_bonus_score: f32,
  //Is there a button to take?
  pub has_button: bool,

  //Is the game been prolonged to stay in a given phase without proceeding to the next?
  pub is_past_normal_phase_end: bool,

  //Is the game supposed to be ended now?
  pub is_game_finished: bool,
  //Winner of the game if the game is supposed to have ended now, C_EMPTY if it is a draw or isNoResult.
  pub winner: Player,
  //Score difference of the game if the game is supposed to have ended now, does NOT take into account whiteKomiAdjustmentForDrawUtility
  //Always an integer or half-integer.
  pub final_white_minus_black_score: f32,
  //True if this game is supposed to be ended with a score
  pub is_scored: bool,
  //True if this game is supposed to be ended but there is no result
  pub is_no_result: bool,
  //True if this game is supposed to be ended but it was by resignation rather than an actual end position
  pub is_resignation: bool,
}

/// upstream: getKoHash (boardhistory.cpp) static
fn get_ko_hash(rules: &Rules, board: &Board, pla: Player, encore_phase: i32, ko_recap_block_hash: Hash128) -> Hash128 {
  if rules.ko_rule == Rules::KO_SITUATIONAL || rules.ko_rule == Rules::KO_SIMPLE || encore_phase > 0 {
    board.pos_hash ^ Board::statics().ZOBRIST_PLAYER_HASH[pla as usize] ^ ko_recap_block_hash
  } else {
    board.pos_hash ^ ko_recap_block_hash
  }
}
/// upstream: getKoHashAfterMoveNonEncore (boardhistory.cpp) static
fn get_ko_hash_after_move_non_encore(rules: &Rules, pos_hash_after_move: Hash128, pla: Player) -> Hash128 {
  if rules.ko_rule == Rules::KO_SITUATIONAL || rules.ko_rule == Rules::KO_SIMPLE {
    pos_hash_after_move ^ Board::statics().ZOBRIST_PLAYER_HASH[pla as usize]
  } else {
    pos_hash_after_move
  }
}
// static Hash128 getKoHashAfterMove(const Rules& rules, Hash128 posHashAfterMove, Player pla, int encorePhase, Hash128 koRecapBlockHashAfterMove) {
//   if(rules.koRule == Rules::KO_SITUATIONAL || rules.koRule == Rules::KO_SIMPLE || encorePhase > 0)
//     return posHashAfterMove ^ Board::ZOBRIST_PLAYER_HASH[pla] ^ koRecapBlockHashAfterMove;
//   else
//     return posHashAfterMove ^ koRecapBlockHashAfterMove;
// }

impl BoardHistory {
  /// upstream: BoardHistory::NUM_RECENT_BOARDS (boardhistory.h)
  pub const NUM_RECENT_BOARDS: i32 = 6;

  /// upstream: BoardHistory::BoardHistory (boardhistory.cpp)
  pub fn new() -> BoardHistory {
    let mut s = BoardHistory {
      rules: Rules::new(),
      move_history: Vec::new(),
      prevent_encore_history: Vec::new(),
      ko_hash_history: Vec::new(),
      first_turn_idx_with_ko_history: 0,
      initial_board: Board::new(),
      initial_pla: P_BLACK,
      initial_encore_phase: 0,
      initial_turn_number: 0,
      assume_multiple_starting_black_moves_are_handicap: false,
      white_has_moved: false,
      override_num_handicap_stones: -1,
      modes: BoardHistoryModes::new(),
      recent_boards: std::array::from_fn(|_| Board::new()),
      current_recent_board_idx: 0,
      presumed_next_move_pla: P_BLACK,
      was_ever_occupied_or_played: [false; Board::MAX_ARR_SIZE as usize],
      super_ko_banned: [false; Board::MAX_ARR_SIZE as usize],
      consecutive_ending_passes: 0,
      hashes_before_black_pass: Vec::new(),
      hashes_before_white_pass: Vec::new(),
      encore_phase: 0,
      num_turns_this_phase: 0,
      num_approx_valid_turns_this_phase: 0,
      num_consec_valid_turns_this_game: 0,
      ko_recap_blocked: [false; Board::MAX_ARR_SIZE as usize],
      ko_recap_block_hash: Hash128::default(),
      ko_captures_in_encore: Vec::new(),
      second_encore_start_colors: [C_EMPTY; Board::MAX_ARR_SIZE as usize],
      white_bonus_score: 0.0f32,
      white_handicap_bonus_score: 0.0f32,
      has_button: false,
      is_past_normal_phase_end: false,
      is_game_finished: false,
      winner: C_EMPTY,
      final_white_minus_black_score: 0.0f32,
      is_scored: false,
      is_no_result: false,
      is_resignation: false,
    };
    s.was_ever_occupied_or_played.fill(false);
    s.super_ko_banned.fill(false);
    s.ko_recap_blocked.fill(false);
    s.second_encore_start_colors.fill(C_EMPTY);
    s
  }

  /// upstream: BoardHistory::BoardHistory (boardhistory.cpp)
  //The modes argument is deliberately required, so that every construction site makes an explicit
  //choice of the BoardHistoryModes. Pass BoardHistoryModes() for plain all-legacy behavior,
  //or an appropriately resolved value (e.g. from Search::resolveHistoryModes or a governing history/book/game).
  pub fn new_full(board: &Board, pla: Player, r: &Rules, e_phase: i32, modes_: &BoardHistoryModes) -> BoardHistory {
    let mut s = BoardHistory {
      rules: *r,
      move_history: Vec::new(),
      prevent_encore_history: Vec::new(),
      ko_hash_history: Vec::new(),
      first_turn_idx_with_ko_history: 0,
      initial_board: Board::new(),
      // PORT NOTE: `initialPla()` value-initializes the int8_t to 0.
      initial_pla: 0,
      initial_encore_phase: 0,
      initial_turn_number: 0,
      assume_multiple_starting_black_moves_are_handicap: false,
      white_has_moved: false,
      override_num_handicap_stones: -1,
      modes: *modes_,
      recent_boards: std::array::from_fn(|_| Board::new()),
      current_recent_board_idx: 0,
      presumed_next_move_pla: pla,
      was_ever_occupied_or_played: [false; Board::MAX_ARR_SIZE as usize],
      super_ko_banned: [false; Board::MAX_ARR_SIZE as usize],
      consecutive_ending_passes: 0,
      hashes_before_black_pass: Vec::new(),
      hashes_before_white_pass: Vec::new(),
      encore_phase: 0,
      num_turns_this_phase: 0,
      num_approx_valid_turns_this_phase: 0,
      num_consec_valid_turns_this_game: 0,
      ko_recap_blocked: [false; Board::MAX_ARR_SIZE as usize],
      ko_recap_block_hash: Hash128::default(),
      ko_captures_in_encore: Vec::new(),
      second_encore_start_colors: [C_EMPTY; Board::MAX_ARR_SIZE as usize],
      white_bonus_score: 0.0f32,
      white_handicap_bonus_score: 0.0f32,
      has_button: false,
      is_past_normal_phase_end: false,
      is_game_finished: false,
      winner: C_EMPTY,
      final_white_minus_black_score: 0.0f32,
      is_scored: false,
      is_no_result: false,
      is_resignation: false,
    };
    s.was_ever_occupied_or_played.fill(false);
    s.super_ko_banned.fill(false);
    s.ko_recap_blocked.fill(false);
    s.second_encore_start_colors.fill(C_EMPTY);

    // PORT NOTE: upstream passes the member `rules` (already equal to r); passing r avoids the aliasing borrow.
    s.clear(board, pla, r, e_phase);
    s
  }

  // upstream: BoardHistory::~BoardHistory (boardhistory.cpp)
  // PORT NOTE: the destructor body is empty; no Drop impl.

  // upstream: BoardHistory::BoardHistory(BoardHistory&&), BoardHistory::operator=(BoardHistory&&) (boardhistory.cpp)
  // PORT NOTE: move constructor / move assignment are Rust moves.

  /// upstream: BoardHistory::clear (boardhistory.cpp)
  //Clears all history and status and bonus points, sets encore phase and rules
  pub fn clear(&mut self, board: &Board, pla: Player, r: &Rules, e_phase: i32) {
    self.rules = *r;
    self.move_history.clear();
    self.prevent_encore_history.clear();
    self.ko_hash_history.clear();
    self.first_turn_idx_with_ko_history = 0;

    self.initial_board = board.clone();
    self.initial_pla = pla;
    self.initial_encore_phase = e_phase;
    self.initial_turn_number = 0;
    self.assume_multiple_starting_black_moves_are_handicap = false;
    self.white_has_moved = false;
    self.override_num_handicap_stones = -1;
    //Deliberately does NOT reset modes - see boardhistory.h.

    //This makes it so that if we ask for recent boards with a lookback beyond what we have a history for,
    //we simply return copies of the starting board.
    for i in 0..BoardHistory::NUM_RECENT_BOARDS {
      self.recent_boards[i as usize] = board.clone();
    }
    self.current_recent_board_idx = 0;

    self.presumed_next_move_pla = pla;

    for y in 0..board.y_size {
      for x in 0..board.x_size {
        let loc: Loc = Location::get_loc(x, y, board.x_size);
        self.was_ever_occupied_or_played[loc as usize] = board.colors[loc as usize] != C_EMPTY;
      }
    }

    self.super_ko_banned.fill(false);
    self.consecutive_ending_passes = 0;
    self.hashes_before_black_pass.clear();
    self.hashes_before_white_pass.clear();
    self.num_turns_this_phase = 0;
    self.num_approx_valid_turns_this_phase = 0;
    self.num_consec_valid_turns_this_game = 0;
    self.ko_recap_blocked.fill(false);
    self.ko_recap_block_hash = Hash128::default();
    self.ko_captures_in_encore.clear();
    self.white_bonus_score = 0.0f32;
    self.white_handicap_bonus_score = 0.0f32;
    // PORT NOTE: reads the encorePhase member before it is set below, as upstream does.
    self.has_button = self.rules.has_button && self.encore_phase == 0;
    self.is_past_normal_phase_end = false;
    self.is_game_finished = false;
    self.winner = C_EMPTY;
    self.final_white_minus_black_score = 0.0f32;
    self.is_scored = false;
    self.is_no_result = false;
    self.is_resignation = false;

    //Handle encore phase
    self.encore_phase = e_phase;
    testAssert!(self.encore_phase >= 0 && self.encore_phase <= 2);
    if self.encore_phase > 0 {
      testAssert!(self.rules.scoring_rule == Rules::SCORING_TERRITORY);
    }
    //Update the few parameters that depend on encore
    if self.encore_phase == 2 {
      self.second_encore_start_colors.copy_from_slice(&board.colors);
    } else {
      self.second_encore_start_colors.fill(C_EMPTY);
    }

    //Push hash for the new board state
    self.ko_hash_history.push(get_ko_hash(&self.rules, board, pla, self.encore_phase, self.ko_recap_block_hash));

    if self.rules.scoring_rule == Rules::SCORING_TERRITORY {
      //Chill 1 point for every move played
      for y in 0..board.y_size {
        for x in 0..board.x_size {
          let loc: Loc = Location::get_loc(x, y, board.x_size);
          if board.colors[loc as usize] == P_BLACK {
            self.white_bonus_score += 1.0f32;
          } else if board.colors[loc as usize] == P_WHITE {
            self.white_bonus_score -= 1.0f32;
          }
        }
      }
      //If white actually played extra moves that got captured so we don't see them,
      //then chill for those too
      let net_white_captures: i32 = board.num_white_captures - board.num_black_captures;
      self.white_bonus_score -= net_white_captures as f32;
    }
    self.white_handicap_bonus_score = self.compute_white_handicap_bonus() as f32;
  }

  /// upstream: BoardHistory::copyToInitial (boardhistory.cpp)
  //Returns a copy of this board history rewound to the initial board, pla, etc, with other fields
  //(such as setInitialTurnNumber, setAssumeMultipleStartingBlackMovesAreHandicap) set identically.
  pub fn copy_to_initial(&self) -> BoardHistory {
    let mut hist = BoardHistory::new_full(&self.initial_board, self.initial_pla, &self.rules, self.initial_encore_phase, &self.modes);
    hist.set_initial_turn_number(self.initial_turn_number);
    hist.set_assume_multiple_starting_black_moves_are_handicap(self.assume_multiple_starting_black_moves_are_handicap);
    hist.set_override_num_handicap_stones(self.override_num_handicap_stones);
    hist
  }

  /// upstream: BoardHistory::setInitialTurnNumber (boardhistory.cpp)
  //Set the initial turn number. Affects nothing else.
  pub fn set_initial_turn_number(&mut self, n: i64) {
    self.initial_turn_number = n;
  }

  /// upstream: BoardHistory::setAssumeMultipleStartingBlackMovesAreHandicap (boardhistory.cpp)
  //Set assumeMultipleStartingBlackMovesAreHandicap and update bonus points accordingly
  pub fn set_assume_multiple_starting_black_moves_are_handicap(&mut self, b: bool) {
    self.assume_multiple_starting_black_moves_are_handicap = b;
    self.white_handicap_bonus_score = self.compute_white_handicap_bonus() as f32;
  }

  /// upstream: BoardHistory::setOverrideNumHandicapStones (boardhistory.cpp)
  //Set overrideNumHandicapStones and update bonus points accordingly
  pub fn set_override_num_handicap_stones(&mut self, n: i32) {
    self.override_num_handicap_stones = n;
    self.white_handicap_bonus_score = self.compute_white_handicap_bonus() as f32;
  }

  /// upstream: BoardHistory::setModes (boardhistory.cpp)
  //Set the BoardHistoryModes. Does not re-adjudicate a game that is already ended and scored.
  //Changes the result of getSituationRulesAndKoHash - callers responsible for invalidating
  //anything cached based on that hash (e.g. clearing search).
  pub fn set_modes(&mut self, modes_: &BoardHistoryModes) {
    self.modes = *modes_;
  }

  /// upstream: BoardHistory::suicideLegalForPassAlive (boardhistory.cpp)
  //The suicide legality that should be passed to Board::calculateArea and calculateIndependentLifeArea
  //for all pass-alive area computations, taking modes.alwaysComputePassAliveUnderSuicideRules into account.
  pub fn suicide_legal_for_pass_alive(&self) -> bool {
    self.rules.multi_stone_suicide_legal || self.modes.always_compute_pass_alive_under_suicide_rules
  }
}

/// upstream: numHandicapStonesOnBoardHelper (boardhistory.cpp) static
fn num_handicap_stones_on_board_helper(board: &Board, black_non_pass_turns_to_start: i32) -> i32 {
  let mut start_board_num_black_stones: i32 = 0;
  let mut start_board_num_white_stones: i32 = 0;
  for y in 0..board.y_size {
    for x in 0..board.x_size {
      let loc: Loc = Location::get_loc(x, y, board.x_size);
      if board.colors[loc as usize] == C_BLACK {
        start_board_num_black_stones += 1;
      } else if board.colors[loc as usize] == C_WHITE {
        start_board_num_white_stones += 1;
      }
    }
  }
  //If we set up in a nontrivial position, then consider it a non-handicap game.
  if start_board_num_white_stones != 0 {
    return 0;
  }
  //Add in additional counted stones
  let black_turn_advantage: i32 = start_board_num_black_stones + black_non_pass_turns_to_start;

  //If there was only one black move/stone to start, then it was a regular game
  if black_turn_advantage <= 1 {
    return 0;
  }
  black_turn_advantage
}

impl BoardHistory {
  /// upstream: BoardHistory::numHandicapStonesOnBoard (boardhistory.cpp)
  //Does not do anything like assumeMultipleStartingBlackMovesAreHandicap, computes based on board alone
  pub fn num_handicap_stones_on_board(b: &Board) -> i32 {
    num_handicap_stones_on_board_helper(b, 0)
  }

  /// upstream: BoardHistory::computeNumHandicapStones (boardhistory.cpp)
  //Takes into account assumeMultipleStartingBlackMovesAreHandicap and/or overrideNumHandicapStones
  pub fn compute_num_handicap_stones(&self) -> i32 {
    if self.override_num_handicap_stones >= 0 {
      return self.override_num_handicap_stones;
    }

    let mut black_non_pass_turns_to_start: i32 = 0;
    if self.assume_multiple_starting_black_moves_are_handicap {
      //Find the length of the initial sequence of black moves - treat a string of consecutive black
      //moves at the start of the game as "handicap"
      //This is necessary because when loading sgfs or on some servers, (particularly with free placement)
      //handicap is implemented by having black just make a bunch of moves in a row.
      //But if white makes multiple moves in a row after that, then the plays are probably not handicap, someone's setting
      //up a problem position by having black play all moves in a row then white play all moves in a row.
      for i in 0..self.move_history.len() {
        let move_loc: Loc = self.move_history[i].loc;
        let move_pla: Player = self.move_history[i].pla;
        if move_pla != P_BLACK {
          //Two white moves in a row? Re-set count and quit out.
          if i + 1 < self.move_history.len() && self.move_history[i + 1].pla != P_BLACK {
            black_non_pass_turns_to_start = 0;
            break;
          }
          //White move is a single isolated pass? Assume that it's still a handicap game, it's just that the black
          //moves are interleaved with white passes. Ignore it and continue.
          if move_loc == Board::PASS_LOC {
            continue;
          }
          //Otherwise quit out, we have a normal white move.
          break;
        }
        if move_loc != Board::PASS_LOC && move_loc != Board::NULL_LOC {
          black_non_pass_turns_to_start += 1;
        }
      }
    }
    num_handicap_stones_on_board_helper(&self.initial_board, black_non_pass_turns_to_start)
  }

  /// upstream: BoardHistory::computeWhiteHandicapBonus (boardhistory.cpp)
  pub fn compute_white_handicap_bonus(&self) -> i32 {
    if self.rules.white_handicap_bonus_rule == Rules::WHB_ZERO {
      0
    } else {
      let num_handicap_stones: i32 = self.compute_num_handicap_stones();
      if self.rules.white_handicap_bonus_rule == Rules::WHB_N {
        num_handicap_stones
      } else if self.rules.white_handicap_bonus_rule == Rules::WHB_N_MINUS_ONE {
        if num_handicap_stones > 1 { num_handicap_stones - 1 } else { 0 }
      } else {
        unreachable!()
      }
      // return 0; (after ASSERT_UNREACHABLE; dead in Rust)
    }
  }

  /// upstream: BoardHistory::printBasicInfo (boardhistory.cpp)
  pub fn print_basic_info(&self, out: &mut dyn std::fmt::Write, board: &Board) {
    Board::print_board(out, board, Board::NULL_LOC, Some(&self.move_history));
    writeln!(out, "Next player: {}", PlayerIO::player_to_string(self.presumed_next_move_pla)).unwrap();
    if self.encore_phase > 0 {
      writeln!(out, "Game phase: {}", self.encore_phase).unwrap();
    }
    writeln!(out, "Rules: {}", self.rules.to_json_string()).unwrap();
    if self.white_handicap_bonus_score != 0.0f32 {
      writeln!(out, "Handicap bonus score: {}", cpp_ostream_f64(self.white_handicap_bonus_score as f64)).unwrap();
    }
    writeln!(out, "B stones captured: {}", board.num_black_captures).unwrap();
    writeln!(out, "W stones captured: {}", board.num_white_captures).unwrap();
  }

  /// upstream: BoardHistory::printDebugInfo (boardhistory.cpp)
  pub fn print_debug_info(&self, out: &mut dyn std::fmt::Write, board: &Board) {
    writeln!(out, "{}", board).unwrap();
    writeln!(out, "Initial pla {}", PlayerIO::player_to_string(self.initial_pla)).unwrap();
    writeln!(out, "Encore phase {}", self.encore_phase).unwrap();
    writeln!(out, "Turns this phase {}", self.num_turns_this_phase).unwrap();
    writeln!(out, "Approx valid turns this phase {}", self.num_approx_valid_turns_this_phase).unwrap();
    writeln!(out, "Approx consec valid turns this game {}", self.num_consec_valid_turns_this_game).unwrap();
    writeln!(out, "Rules {}", self.rules).unwrap();
    writeln!(out, "Ko recap block hash {}", self.ko_recap_block_hash).unwrap();
    writeln!(out, "White bonus score {}", cpp_ostream_f64(self.white_bonus_score as f64)).unwrap();
    writeln!(out, "White handicap bonus score {}", cpp_ostream_f64(self.white_handicap_bonus_score as f64)).unwrap();
    writeln!(out, "Has button {}", self.has_button as i32).unwrap();
    writeln!(out, "Presumed next pla {}", PlayerIO::player_to_string(self.presumed_next_move_pla)).unwrap();
    writeln!(out, "Past normal phase end {}", self.is_past_normal_phase_end as i32).unwrap();
    writeln!(
      out,
      "Game result {} {} {} {} {} {}",
      self.is_game_finished as i32,
      PlayerIO::player_to_string(self.winner),
      cpp_ostream_f64(self.final_white_minus_black_score as f64),
      self.is_scored as i32,
      self.is_no_result as i32,
      self.is_resignation as i32
    )
    .unwrap();
    write!(out, "Last moves ").unwrap();
    for i in 0..self.move_history.len() {
      write!(out, "{} ", Location::to_string(self.move_history[i].loc, board)).unwrap();
    }
    writeln!(out).unwrap();
    debug_assert!(self.first_turn_idx_with_ko_history + self.ko_hash_history.len() == self.move_history.len() + 1);
  }

  /// upstream: BoardHistory::getRecentBoard (boardhistory.cpp)
  //Returns a reference a recent board state, where 0 is the current board, 1 is 1 move ago, etc.
  //Requires that numMovesAgo < NUM_RECENT_BOARDS
  pub fn get_recent_board(&self, num_moves_ago: i32) -> &Board {
    debug_assert!(num_moves_ago >= 0 && num_moves_ago < BoardHistory::NUM_RECENT_BOARDS);
    let idx: i32 = (self.current_recent_board_idx - num_moves_ago + BoardHistory::NUM_RECENT_BOARDS) % BoardHistory::NUM_RECENT_BOARDS;
    &self.recent_boards[idx as usize]
  }

  /// upstream: BoardHistory::setKomi (boardhistory.cpp)
  //Set only the komi field of the rules, does not clear history, does recompute game score if game is over.
  pub fn set_komi(&mut self, new_komi: f32) {
    let old_komi: f32 = self.rules.komi;
    self.rules.komi = new_komi;

    //Recompute the game result due to the new komi
    if self.is_game_finished && self.is_scored {
      self.set_final_score_and_winner(self.final_white_minus_black_score - old_komi + new_komi);
    }
  }

  /// upstream: BoardHistory::koHashOccursInHistory (boardhistory.cpp)
  //If rootKoHashTable is provided, will take advantage of rootKoHashTable rather than search within the first
  //rootKoHashTable->size() moves of koHashHistory.
  //ALSO counts the most recent ko hash!
  pub(crate) fn ko_hash_occurs_in_history(&self, ko_hash: Hash128, root_ko_hash_table: Option<&KoHashTable>) -> bool {
    let mut start: usize = 0;
    let ko_hash_history_size: usize = self.ko_hash_history.len();
    if let Some(root_ko_hash_table) = root_ko_hash_table {
      if self.first_turn_idx_with_ko_history == root_ko_hash_table.first_turn_idx_with_ko_history {
        let table_size: usize = root_ko_hash_table.size();
        debug_assert!(self.first_turn_idx_with_ko_history + self.ko_hash_history.len() == self.move_history.len() + 1);
        debug_assert!(table_size <= ko_hash_history_size);
        if root_ko_hash_table.contains_hash(ko_hash) {
          return true;
        }
        start = table_size;
      }
    }
    for i in start..ko_hash_history_size {
      if self.ko_hash_history[i] == ko_hash {
        return true;
      }
    }
    false
  }

  /// upstream: BoardHistory::numberOfKoHashOccurrencesInHistory (boardhistory.cpp)
  //If rootKoHashTable is provided, will take advantage of rootKoHashTable rather than search within the first
  //rootKoHashTable->size() moves of koHashHistory.
  //ALSO counts the most recent ko hash!
  pub fn number_of_ko_hash_occurrences_in_history(&self, ko_hash: Hash128, root_ko_hash_table: Option<&KoHashTable>) -> i32 {
    let mut count: i32 = 0;
    let mut start: usize = 0;
    let ko_hash_history_size: usize = self.ko_hash_history.len();
    if let Some(root_ko_hash_table) = root_ko_hash_table {
      if self.first_turn_idx_with_ko_history == root_ko_hash_table.first_turn_idx_with_ko_history {
        let table_size: usize = root_ko_hash_table.size();
        debug_assert!(self.first_turn_idx_with_ko_history + self.ko_hash_history.len() == self.move_history.len() + 1);
        debug_assert!(table_size <= ko_hash_history_size);
        count += root_ko_hash_table.number_of_occurrences_of_hash(ko_hash);
        start = table_size;
      }
    }
    for i in start..ko_hash_history_size {
      if self.ko_hash_history[i] == ko_hash {
        count += 1;
      }
    }
    count
  }

  /// upstream: BoardHistory::whiteKomiAdjustmentForDraws (boardhistory.cpp)
  pub fn white_komi_adjustment_for_draws(&self, draw_equivalent_wins_for_white: f64) -> f32 {
    //We fold the draw utility into the komi, for input into things like the neural net.
    //Basically we model it as if the final score were jittered by a uniform draw from [-0.5,0.5].
    //E.g. if komi from self perspective is 7 and a draw counts as 0.75 wins and 0.25 losses,
    //then komi input should be as if it was 7.25, which in a jigo game when jittered by 0.5 gives white 75% wins and 25% losses.
    let draw_adjustment: f32 =
      if self.rules.game_result_will_be_integer() { (draw_equivalent_wins_for_white - 0.5) as f32 } else { 0.0f32 };
    draw_adjustment
  }

  /// upstream: BoardHistory::currentSelfKomi (boardhistory.cpp)
  pub fn current_self_komi(&self, pla: Player, draw_equivalent_wins_for_white: f64) -> f32 {
    let white_komi_adjusted: f32 = self.white_bonus_score
      + self.white_handicap_bonus_score
      + self.rules.komi
      + self.white_komi_adjustment_for_draws(draw_equivalent_wins_for_white);

    if pla == P_WHITE {
      white_komi_adjusted
    } else if pla == P_BLACK {
      -white_komi_adjusted
    } else {
      unreachable!()
    }
  }

  /// upstream: BoardHistory::countAreaScoreWhiteMinusBlack (boardhistory.cpp)
  pub(crate) fn count_area_score_white_minus_black(&self, board: &Board, area: &mut [Color; Board::MAX_ARR_SIZE as usize]) -> i32 {
    let mut score: i32 = 0;
    if self.rules.tax_rule == Rules::TAX_NONE {
      let non_pass_alive_stones = true;
      let safe_big_territories = true;
      let unsafe_big_territories = true;
      board.calculate_area(area, non_pass_alive_stones, safe_big_territories, unsafe_big_territories, self.suicide_legal_for_pass_alive());
    } else if self.rules.tax_rule == Rules::TAX_SEKI || self.rules.tax_rule == Rules::TAX_ALL {
      let keep_territories = false;
      let keep_stones = true;
      let mut white_minus_black_independent_life_region_count: i32 = 0;
      board.calculate_independent_life_area(
        area,
        &mut white_minus_black_independent_life_region_count,
        keep_territories,
        keep_stones,
        self.modes.exclude_territory_adjacent_to_atari,
        self.suicide_legal_for_pass_alive(),
      );
      if self.rules.tax_rule == Rules::TAX_ALL {
        score -= 2 * white_minus_black_independent_life_region_count;
      }
    } else {
      unreachable!()
    }

    for y in 0..board.y_size {
      for x in 0..board.x_size {
        let loc: Loc = Location::get_loc(x, y, board.x_size);
        if area[loc as usize] == C_WHITE {
          score += 1;
        } else if area[loc as usize] == C_BLACK {
          score -= 1;
        }
      }
    }

    score
  }

  /// upstream: BoardHistory::countTerritoryAreaScoreWhiteMinusBlack (boardhistory.cpp)
  //ALSO makes area color the points that were not pass alive but were scored for a side.
  pub(crate) fn count_territory_area_score_white_minus_black(&self, board: &Board, area: &mut [Color; Board::MAX_ARR_SIZE as usize]) -> i32 {
    let mut score: i32 = 0;
    let keep_territories: bool;
    let keep_stones: bool;
    if self.rules.tax_rule == Rules::TAX_NONE {
      keep_territories = true;
      keep_stones = false;
    } else if self.rules.tax_rule == Rules::TAX_SEKI || self.rules.tax_rule == Rules::TAX_ALL {
      keep_territories = false;
      keep_stones = false;
    } else {
      unreachable!()
    }

    let mut white_minus_black_independent_life_region_count: i32 = 0;
    board.calculate_independent_life_area(
      area,
      &mut white_minus_black_independent_life_region_count,
      keep_territories,
      keep_stones,
      self.modes.exclude_territory_adjacent_to_atari,
      self.suicide_legal_for_pass_alive(),
    );

    for y in 0..board.y_size {
      for x in 0..board.x_size {
        let loc: Loc = Location::get_loc(x, y, board.x_size);
        if area[loc as usize] == C_WHITE {
          score += 1;
        } else if area[loc as usize] == C_BLACK {
          score -= 1;
        } else {
          //Checking encorePhase < 2 allows us to get the correct score if we directly end the game before the second
          //encore such that we never actually fill secondEncoreStartColors. This matters for premature termination
          //of the game like ending due to a move limit and such.
          if board.colors[loc as usize] == C_WHITE && (self.encore_phase < 2 || self.second_encore_start_colors[loc as usize] == C_WHITE) {
            score += 1;
            area[loc as usize] = C_WHITE;
          }
          if board.colors[loc as usize] == C_BLACK && (self.encore_phase < 2 || self.second_encore_start_colors[loc as usize] == C_BLACK) {
            score -= 1;
            area[loc as usize] = C_BLACK;
          }
        }
      }
    }
    if self.rules.tax_rule == Rules::TAX_ALL {
      score -= 2 * white_minus_black_independent_life_region_count;
    }
    score
  }

  /// upstream: BoardHistory::setFinalScoreAndWinner (boardhistory.cpp)
  pub(crate) fn set_final_score_and_winner(&mut self, score: f32) {
    self.final_white_minus_black_score = score;
    if self.final_white_minus_black_score > 0.0f32 {
      self.winner = C_WHITE;
    } else if self.final_white_minus_black_score < 0.0f32 {
      self.winner = C_BLACK;
    } else {
      self.winner = C_EMPTY;
    }
  }

  /// upstream: BoardHistory::getAreaNow (boardhistory.cpp)
  pub fn get_area_now(&self, board: &Board, area: &mut [Color; Board::MAX_ARR_SIZE as usize]) {
    if self.rules.scoring_rule == Rules::SCORING_AREA {
      self.count_area_score_white_minus_black(board, area);
    } else if self.rules.scoring_rule == Rules::SCORING_TERRITORY {
      self.count_territory_area_score_white_minus_black(board, area);
    } else {
      unreachable!()
    }
  }

  /// upstream: BoardHistory::endAndScoreGameNow (boardhistory.cpp)
  pub fn end_and_score_game_now_area(&mut self, board: &Board, area: &mut [Color; Board::MAX_ARR_SIZE as usize]) {
    let board_score: i32;
    if self.rules.scoring_rule == Rules::SCORING_AREA {
      board_score = self.count_area_score_white_minus_black(board, area);
    } else if self.rules.scoring_rule == Rules::SCORING_TERRITORY {
      board_score = self.count_territory_area_score_white_minus_black(board, area);
    } else {
      unreachable!()
    }

    if self.has_button {
      self.has_button = false;
      self.white_bonus_score += if self.presumed_next_move_pla == P_WHITE { 0.5f32 } else { -0.5f32 };
    }

    self.set_final_score_and_winner(board_score as f32 + self.white_bonus_score + self.white_handicap_bonus_score + self.rules.komi);
    self.is_scored = true;
    self.is_no_result = false;
    self.is_resignation = false;
    self.is_game_finished = true;
    self.is_past_normal_phase_end = false;
  }

  /// upstream: BoardHistory::endAndScoreGameNow (boardhistory.cpp)
  //Score the board as-is. If the game is already finished, and is NOT a no-result, then this should be idempotent.
  pub fn end_and_score_game_now(&mut self, board: &Board) {
    let mut area: [Color; Board::MAX_ARR_SIZE as usize] = [0; Board::MAX_ARR_SIZE as usize];
    self.end_and_score_game_now_area(board, &mut area);
  }

  /// upstream: BoardHistory::endGameIfAllPassAlive (boardhistory.cpp)
  //Slightly expensive, check if the entire game is all pass-alive-territory, and if so, declare the game finished
  pub fn end_game_if_all_pass_alive(&mut self, board: &Board) {
    let mut board_score: i32 = 0;
    let non_pass_alive_stones = false;
    let safe_big_territories = false;
    let unsafe_big_territories = false;
    let mut area: [Color; Board::MAX_ARR_SIZE as usize] = [0; Board::MAX_ARR_SIZE as usize];
    board.calculate_area(&mut area, non_pass_alive_stones, safe_big_territories, unsafe_big_territories, self.suicide_legal_for_pass_alive());

    for y in 0..board.y_size {
      for x in 0..board.x_size {
        let loc: Loc = Location::get_loc(x, y, board.x_size);
        if area[loc as usize] == C_WHITE {
          board_score += 1;
        } else if area[loc as usize] == C_BLACK {
          board_score -= 1;
        } else {
          return;
        }
      }
    }

    //In the case that we have a group tax, rescore normally to actually count the group tax
    if self.rules.tax_rule == Rules::TAX_ALL {
      self.end_and_score_game_now(board);
    } else {
      if self.has_button {
        self.has_button = false;
        self.white_bonus_score += if self.presumed_next_move_pla == P_WHITE { 0.5f32 } else { -0.5f32 };
      }
      self.set_final_score_and_winner(board_score as f32 + self.white_bonus_score + self.white_handicap_bonus_score + self.rules.komi);
      self.is_scored = true;
      self.is_no_result = false;
      self.is_resignation = false;
      self.is_game_finished = true;
      self.is_past_normal_phase_end = false;
    }
  }

  /// upstream: BoardHistory::setWinnerByResignation (boardhistory.cpp)
  pub fn set_winner_by_resignation(&mut self, pla: Player) {
    self.is_game_finished = true;
    self.is_past_normal_phase_end = false;
    self.is_scored = false;
    self.is_no_result = false;
    self.is_resignation = true;
    self.winner = pla;
    self.final_white_minus_black_score = 0.0f32;
  }

  /// upstream: BoardHistory::setKoRecapBlocked (boardhistory.cpp)
  pub(crate) fn set_ko_recap_blocked(&mut self, loc: Loc, b: bool) {
    if self.ko_recap_blocked[loc as usize] != b {
      self.ko_recap_blocked[loc as usize] = b;
      //We used to have per-color marks, so the zobrist was for both. Just combine them.
      self.ko_recap_block_hash ^=
        Board::statics().ZOBRIST_KO_MARK_HASH[loc as usize][C_BLACK as usize] ^ Board::statics().ZOBRIST_KO_MARK_HASH[loc as usize][C_WHITE as usize];
    }
  }

  /// upstream: BoardHistory::isLegal (boardhistory.cpp)
  //Check if a move on the board is legal, taking into account the full game state and superko
  pub fn is_legal(&self, board: &Board, move_loc: Loc, move_pla: Player) -> bool {
    if move_pla != self.presumed_next_move_pla {
      return false;
    }

    //Ko-moves in the encore that are recapture blocked are interpreted as pass-for-ko, so they are legal
    if self.encore_phase > 0 {
      if move_loc >= 0 && (move_loc as i32) < Board::MAX_ARR_SIZE && move_loc != Board::PASS_LOC {
        if board.colors[move_loc as usize] == get_opp(move_pla)
          && self.ko_recap_blocked[move_loc as usize]
          && board.get_chain_size(move_loc) == 1
          && board.get_num_liberties(move_loc) == 1
        {
          return true;
        }
        let ko_capture_loc: Loc = board.get_ko_capture_loc(move_loc, move_pla);
        if ko_capture_loc != Board::NULL_LOC
          && self.ko_recap_blocked[ko_capture_loc as usize]
          && board.colors[ko_capture_loc as usize] == get_opp(move_pla)
        {
          return true;
        }
      }
    } else {
      //Only check ko bans during normal play.
      //Ko mechanics in the encore are totally different, we ignore simple ko loc.
      if board.is_ko_banned(move_loc) {
        return false;
      }
    }
    if !board.is_legal_ignoring_ko(move_loc, move_pla, self.rules.multi_stone_suicide_legal) {
      return false;
    }
    if self.super_ko_banned[move_loc as usize] {
      return false;
    }

    true
  }

  /// upstream: BoardHistory::isPassForKo (boardhistory.cpp)
  //Check if the specified move is a pass-for-ko encore move.
  pub fn is_pass_for_ko(&self, board: &Board, move_loc: Loc, move_pla: Player) -> bool {
    if self.encore_phase > 0 && move_loc >= 0 && (move_loc as i32) < Board::MAX_ARR_SIZE && move_loc != Board::PASS_LOC {
      if board.colors[move_loc as usize] == get_opp(move_pla)
        && self.ko_recap_blocked[move_loc as usize]
        && board.get_chain_size(move_loc) == 1
        && board.get_num_liberties(move_loc) == 1
      {
        return true;
      }

      let ko_capture_loc: Loc = board.get_ko_capture_loc(move_loc, move_pla);
      if ko_capture_loc != Board::NULL_LOC
        && self.ko_recap_blocked[ko_capture_loc as usize]
        && board.colors[ko_capture_loc as usize] == get_opp(move_pla)
      {
        return true;
      }
    }
    false
  }

  /// upstream: BoardHistory::getCurrentTurnNumber (boardhistory.cpp)
  //Current turn number, based on initial turn number
  pub fn get_current_turn_number(&self) -> i64 {
    std::cmp::max(0i64, self.initial_turn_number + self.move_history.len() as i64)
  }

  /// upstream: BoardHistory::newConsecutiveEndingPassesAfterPass (boardhistory.cpp)
  //Return the number of consecutive game-ending passes there would be if a pass was made
  pub(crate) fn new_consecutive_ending_passes_after_pass(&self) -> i32 {
    let mut new_consecutive_ending_passes: i32 = self.consecutive_ending_passes;
    if self.encore_phase > 0 {
      new_consecutive_ending_passes += 1;
    } else {
      match self.rules.ko_rule {
        Rules::KO_SIMPLE | Rules::KO_POSITIONAL | Rules::KO_SITUATIONAL => {
          new_consecutive_ending_passes += 1;
        }
        Rules::KO_SPIGHT => {
          new_consecutive_ending_passes = 0;
        }
        _ => unreachable!(),
      }
    }
    new_consecutive_ending_passes
  }

  /// upstream: BoardHistory::phaseHasSpightlikeEndingAndPassHistoryClearing (boardhistory.cpp)
  //Returns true if the rules of the game specify that passes should clear history for the purposes
  //of ko rules checking and for no-result infinite cycles. Also implies that the phase will end
  //spightlight - i.e. upon a pass where the same player has passed in the same situation before.
  pub(crate) fn phase_has_spightlike_ending_and_pass_history_clearing(&self) -> bool {
    self.encore_phase > 0 || self.rules.ko_rule == Rules::KO_SIMPLE || self.rules.ko_rule == Rules::KO_SPIGHT
  }

  /// upstream: BoardHistory::wouldBeSpightlikeEndingPass (boardhistory.cpp)
  //Returns true if this move would be a pass that causes spight-style ending of the phase
  //(i.e. ending that ignores the number of consecutive passes)
  pub(crate) fn would_be_spightlike_ending_pass(&self, move_pla: Player, ko_hash_before_move: Hash128) -> bool {
    if self.phase_has_spightlike_ending_and_pass_history_clearing() {
      if move_pla == P_BLACK && self.hashes_before_black_pass.contains(&ko_hash_before_move) {
        return true;
      }
      if move_pla == P_WHITE && self.hashes_before_white_pass.contains(&ko_hash_before_move) {
        return true;
      }
    }
    false
  }

  /// upstream: BoardHistory::passWouldEndPhase (boardhistory.cpp)
  //Check if passing right now would end the current phase of play, or the entire game
  pub fn pass_would_end_phase(&self, board: &Board, move_pla: Player) -> bool {
    let ko_hash_before_move: Hash128 = get_ko_hash(&self.rules, board, move_pla, self.encore_phase, self.ko_recap_block_hash);
    if self.new_consecutive_ending_passes_after_pass() >= 2 || self.would_be_spightlike_ending_pass(move_pla, ko_hash_before_move) {
      return true;
    }
    false
  }

  /// upstream: BoardHistory::passWouldEndGame (boardhistory.cpp)
  pub fn pass_would_end_game(&self, board: &Board, move_pla: Player) -> bool {
    self.pass_would_end_phase(board, move_pla)
      && (self.rules.scoring_rule == Rules::SCORING_AREA
        || (self.rules.scoring_rule == Rules::SCORING_TERRITORY && self.encore_phase >= 2))
  }

  /// upstream: BoardHistory::shouldSuppressEndGameFromFriendlyPass (boardhistory.cpp)
  //If friendly pass is okay in area scoring rules, we require a "third" pass to end the game during search, unless it ends
  //via spightlike rule. This function returns true when passing would end the game, but we shouldn't accept it because it's
  //a second pass and not a third pass.
  pub fn should_suppress_end_game_from_friendly_pass(&self, board: &Board, move_pla: Player) -> bool {
    self.rules.friendly_pass_ok
      && self.rules.scoring_rule == Rules::SCORING_AREA
      && self.new_consecutive_ending_passes_after_pass() == 2
      && !self.would_be_spightlike_ending_pass(
        move_pla,
        get_ko_hash(&self.rules, board, move_pla, self.encore_phase, self.ko_recap_block_hash),
      )
  }

  /// upstream: BoardHistory::isFinalPhase (boardhistory.cpp)
  //Check if this is the final phase of the game, such that ending it moves to scoring.
  pub fn is_final_phase(&self) -> bool {
    self.rules.scoring_rule == Rules::SCORING_AREA
      || (self.rules.scoring_rule == Rules::SCORING_TERRITORY && self.encore_phase >= 2)
  }

  /// upstream: BoardHistory::isLegalTolerant (boardhistory.cpp)
  pub fn is_legal_tolerant(&self, board: &Board, move_loc: Loc, move_pla: Player) -> bool {
    // Allow either side to move during tolerant play, but still check that a player is specified
    if move_pla != P_BLACK && move_pla != P_WHITE {
      return false;
    }
    let multi_stone_suicide_legal = true; // Tolerate suicide regardless of rules
    if !self.is_pass_for_ko(board, move_loc, move_pla) && !board.is_legal_ignoring_ko(move_loc, move_pla, multi_stone_suicide_legal) {
      return false;
    }
    true
  }
  /// upstream: BoardHistory::makeBoardMoveTolerant (boardhistory.cpp)
  //Make a move with legality checking, but be mostly tolerant and allow moves that can still be handled but that may not technically
  //be legal. This is intended for reading moves from SGFs and such where maybe we're getting moves that were played in a different
  //ruleset than ours. Returns true if successful, false if was illegal even unter tolerant rules.
  pub fn make_board_move_tolerant(&mut self, board: &mut Board, move_loc: Loc, move_pla: Player) -> bool {
    // Allow either side to move during tolerant play, but still check that a player is specified
    if move_pla != P_BLACK && move_pla != P_WHITE {
      return false;
    }
    let multi_stone_suicide_legal = true; // Tolerate suicide regardless of rules
    if !self.is_pass_for_ko(board, move_loc, move_pla) && !board.is_legal_ignoring_ko(move_loc, move_pla, multi_stone_suicide_legal) {
      return false;
    }
    self.make_board_move_assume_legal(board, move_loc, move_pla, None, false);
    true
  }
  /// upstream: BoardHistory::makeBoardMoveTolerant (boardhistory.cpp)
  pub fn make_board_move_tolerant_prevent_encore(&mut self, board: &mut Board, move_loc: Loc, move_pla: Player, prevent_encore: bool) -> bool {
    // Allow either side to move during tolerant play, but still check that a player is specified
    if move_pla != P_BLACK && move_pla == self.presumed_next_move_pla && move_pla != P_WHITE {
      return false;
    }
    let multi_stone_suicide_legal = true; // Tolerate suicide regardless of rules
    if !self.is_pass_for_ko(board, move_loc, move_pla) && !board.is_legal_ignoring_ko(move_loc, move_pla, multi_stone_suicide_legal) {
      return false;
    }
    self.make_board_move_assume_legal(board, move_loc, move_pla, None, prevent_encore);
    true
  }

  /// upstream: BoardHistory::makeBoardMoveAssumeLegal (boardhistory.cpp)
  //For all of the below, rootKoHashTable is optional and if provided will slightly speedup superko searches
  //This function should behave gracefully so long as it is pseudolegal (board.isLegal, but also still ok if the move is on board.ko_loc)
  //even if the move violates superko or encore ko recapture prohibitions, or is past when the game is ended.
  //This allows for robustness when this code is being used for analysis or with external data sources.
  //preventEncore artifically prevents any move from entering or advancing the encore phase when using territory scoring.
  pub fn make_board_move_assume_legal(
    &mut self,
    board: &mut Board,
    move_loc: Loc,
    move_pla: Player,
    root_ko_hash_table: Option<&KoHashTable>,
    prevent_encore: bool,
  ) {
    let pos_hash_before_move: Hash128 = board.pos_hash;

    //Handle if somehow we're making a move after the game or phase was ended
    if self.is_game_finished || self.is_past_normal_phase_end {
      //Cap at 1 - do include the latest move that likely ended the game by itself since by itself
      //absent any history of passes it should be valid still.
      self.num_approx_valid_turns_this_phase = std::cmp::min(self.num_approx_valid_turns_this_phase, 1);
      self.num_consec_valid_turns_this_game = std::cmp::min(self.num_consec_valid_turns_this_game, 1);
    }

    let move_is_illegal: bool = !self.is_legal(board, move_loc, move_pla);

    //And if somehow we're making a move after the game was ended, just clear those values and continue.
    self.is_game_finished = false;
    self.is_past_normal_phase_end = false;
    self.winner = C_EMPTY;
    self.final_white_minus_black_score = 0.0f32;
    self.is_scored = false;
    self.is_no_result = false;
    self.is_resignation = false;

    //Update consecutiveEndingPasses and button
    let mut is_spightlike_ending_pass = false;
    if move_loc != Board::PASS_LOC {
      self.consecutive_ending_passes = 0;
    } else if self.has_button {
      debug_assert!(self.encore_phase == 0 && self.rules.has_button);
      self.has_button = false;
      self.white_bonus_score += if move_pla == P_WHITE { 0.5f32 } else { -0.5f32 };
      self.consecutive_ending_passes = 0;
      //Taking the button clears all ko hash histories (this is equivalent to not clearing them and treating buttonless
      //state as different than buttonful state)
      self.hashes_before_black_pass.clear();
      self.hashes_before_white_pass.clear();
      self.ko_hash_history.clear();
      //The first turn idx with history will be the one RESULTING from this move.
      self.first_turn_idx_with_ko_history = self.move_history.len() + 1;
    } else {
      //Passes clear ko history in the main phase with spight ko rules and in the encore
      //This lifts bans in spight ko rules and lifts 3-fold-repetition checking in the encore for no-resultifying infinite cycles
      //They also clear in simple ko rules for the purpose of no-resulting long cycles. Long cycles with passes do not no-result.
      if self.phase_has_spightlike_ending_and_pass_history_clearing() {
        self.ko_hash_history.clear();
        //The first turn idx with history will be the one RESULTING from this move.
        self.first_turn_idx_with_ko_history = self.move_history.len() + 1;
        //Does not clear hashesBeforeBlackPass or hashesBeforeWhitePass. Passes lift ko bans, but
        //still repeated positions after pass end the game or phase, which these arrays are used to check.
      }

      let ko_hash_before_this_move: Hash128 = get_ko_hash(&self.rules, board, move_pla, self.encore_phase, self.ko_recap_block_hash);
      self.consecutive_ending_passes = self.new_consecutive_ending_passes_after_pass();
      //Check if we have a game-ending pass BEFORE updating hashesBeforeBlackPass and hashesBeforeWhitePass
      is_spightlike_ending_pass = self.would_be_spightlike_ending_pass(move_pla, ko_hash_before_this_move);

      //Update hashesBeforeBlackPass and hashesBeforeWhitePass
      if move_pla == P_BLACK {
        self.hashes_before_black_pass.push(ko_hash_before_this_move);
      } else if move_pla == P_WHITE {
        self.hashes_before_white_pass.push(ko_hash_before_this_move);
      } else {
        unreachable!();
      }
    }

    //Handle pass-for-ko moves in the encore. Pass for ko lifts a ko recapture block and does nothing else.
    let mut was_pass_for_ko = false;
    if self.encore_phase > 0 && move_loc != Board::PASS_LOC {
      if board.colors[move_loc as usize] == get_opp(move_pla) && self.ko_recap_blocked[move_loc as usize] {
        self.set_ko_recap_blocked(move_loc, false);
        was_pass_for_ko = true;
        //Clear simple ko loc just in case
        //Since we aren't otherwise touching the board, from the board's perspective a player will be moving twice in a row.
        board.clear_simple_ko_loc();
      } else {
        let ko_capture_loc: Loc = board.get_ko_capture_loc(move_loc, move_pla);
        if ko_capture_loc != Board::NULL_LOC
          && self.ko_recap_blocked[ko_capture_loc as usize]
          && board.colors[ko_capture_loc as usize] == get_opp(move_pla)
        {
          self.set_ko_recap_blocked(ko_capture_loc, false);
          was_pass_for_ko = true;
          //Clear simple ko loc just in case
          //Since we aren't otherwise touching the board, from the board's perspective a player will be moving twice in a row.
          board.clear_simple_ko_loc();
        }
      }
    }
    //Otherwise handle regular moves
    if !was_pass_for_ko {
      board.play_move_assume_legal(move_loc, move_pla);

      if self.encore_phase > 0 {
        //Update ko recapture blocks and record that this was a ko capture
        if board.ko_loc != Board::NULL_LOC {
          self.set_ko_recap_blocked(move_loc, true);
          self.ko_captures_in_encore.push(EncoreKoCapture::new(pos_hash_before_move, move_loc, move_pla));
          //Clear simple ko loc now that we've absorbed the ko loc information into the korecap blocks
          //Once we have that, the simple ko loc plays no further role in game state or legality
          board.clear_simple_ko_loc();
        }
        //Unmark all ko recap blocks not on stones
        for y in 0..board.y_size {
          for x in 0..board.x_size {
            let loc: Loc = Location::get_loc(x, y, board.x_size);
            if board.colors[loc as usize] == C_EMPTY && self.ko_recap_blocked[loc as usize] {
              self.set_ko_recap_blocked(loc, false);
            }
          }
        }
      }
    }

    //Update recent boards
    self.current_recent_board_idx = (self.current_recent_board_idx + 1) % BoardHistory::NUM_RECENT_BOARDS;
    self.recent_boards[self.current_recent_board_idx as usize] = board.clone();

    let ko_hash_after_this_move: Hash128 = get_ko_hash(&self.rules, board, get_opp(move_pla), self.encore_phase, self.ko_recap_block_hash);
    self.ko_hash_history.push(ko_hash_after_this_move);
    self.move_history.push(Move::new(move_loc, move_pla));
    self.prevent_encore_history.push(prevent_encore);
    self.num_turns_this_phase += 1;
    self.num_approx_valid_turns_this_phase += 1;
    self.num_consec_valid_turns_this_game += 1;
    self.presumed_next_move_pla = get_opp(move_pla);

    if move_is_illegal {
      self.num_consec_valid_turns_this_game = 0;
    }

    if move_loc != Board::PASS_LOC {
      self.was_ever_occupied_or_played[move_loc as usize] = true;
    }

    //Mark all locations that are superko-illegal for the next player, by iterating and testing each point.
    let next_pla: Player = get_opp(move_pla);
    if self.encore_phase <= 0 && self.rules.ko_rule != Rules::KO_SIMPLE {
      debug_assert!(self.ko_recap_block_hash == Hash128::default());
      for y in 0..board.y_size {
        for x in 0..board.x_size {
          let loc: Loc = Location::get_loc(x, y, board.x_size);
          //Cannot be superko banned if it's not a pseudolegal move in the first place, or we would already ban the move under simple ko.
          if board.colors[loc as usize] != C_EMPTY {
            self.super_ko_banned[loc as usize] = false;
          }
          //Also cannot be superko banned if a stone was never there or played there before AND the move is not suicide, because that means
          //the move results in a new stone there and if no stone was ever there in the past the it must be a new position.
          else if !self.was_ever_occupied_or_played[loc as usize] && !board.is_suicide(loc, next_pla) {
            self.super_ko_banned[loc as usize] = false;
          } else if board.is_illegal_suicide(loc, next_pla, self.rules.multi_stone_suicide_legal) || loc == board.ko_loc {
            self.super_ko_banned[loc as usize] = false;
          } else {
            let pos_hash_after_move: Hash128 = board.get_pos_hash_after_move(loc, next_pla);
            let ko_hash_after_move: Hash128 = get_ko_hash_after_move_non_encore(&self.rules, pos_hash_after_move, get_opp(next_pla));
            self.super_ko_banned[loc as usize] = self.ko_hash_occurs_in_history(ko_hash_after_move, root_ko_hash_table);
          }
        }
      }
    } else if self.encore_phase > 0 {
      //During the encore, only one capture of each ko in a given position by a given player
      self.super_ko_banned.fill(false);
      for i in 0..self.ko_captures_in_encore.len() {
        let ekc: &EncoreKoCapture = &self.ko_captures_in_encore[i];
        if ekc.pos_hash_before_move == board.pos_hash && ekc.move_pla == next_pla {
          self.super_ko_banned[ekc.move_loc as usize] = true;
        }
      }
    }

    //Territory scoring - chill 1 point per move in main phase and first encore
    if self.rules.scoring_rule == Rules::SCORING_TERRITORY && self.encore_phase <= 1 && move_loc != Board::PASS_LOC && !was_pass_for_ko {
      if move_pla == P_BLACK {
        self.white_bonus_score += 1.0f32;
      } else if move_pla == P_WHITE {
        self.white_bonus_score -= 1.0f32;
      } else {
        unreachable!();
      }
    }

    //Handicap bonus score
    if move_pla == P_WHITE && move_loc != Board::PASS_LOC {
      self.white_has_moved = true;
    }
    if self.assume_multiple_starting_black_moves_are_handicap
      && !self.white_has_moved
      && move_pla == P_BLACK
      && self.rules.white_handicap_bonus_rule != Rules::WHB_ZERO
    {
      self.white_handicap_bonus_score = self.compute_white_handicap_bonus() as f32;
    }

    //Phase transitions and game end
    if self.consecutive_ending_passes >= 2 || is_spightlike_ending_pass {
      if self.rules.scoring_rule == Rules::SCORING_AREA {
        debug_assert!(self.encore_phase <= 0);
        self.end_and_score_game_now(board);
      } else if self.rules.scoring_rule == Rules::SCORING_TERRITORY {
        if self.encore_phase >= 2 {
          self.end_and_score_game_now(board);
        } else {
          if prevent_encore {
            self.is_past_normal_phase_end = true;
            //Cap at 1 - do include just the single pass here by itself since the single pass by itself
            //absent any history of passes before that should be valid still.
            self.num_approx_valid_turns_this_phase = std::cmp::min(self.num_approx_valid_turns_this_phase, 1);
            self.num_consec_valid_turns_this_game = std::cmp::min(self.num_consec_valid_turns_this_game, 1);
          } else {
            self.encore_phase += 1;
            self.num_turns_this_phase = 0;
            self.num_approx_valid_turns_this_phase = 0;
            if self.encore_phase == 2 {
              self.second_encore_start_colors.copy_from_slice(&board.colors);
            }

            self.super_ko_banned.fill(false);
            self.consecutive_ending_passes = 0;
            self.hashes_before_black_pass.clear();
            self.hashes_before_white_pass.clear();
            self.ko_recap_blocked.fill(false);
            self.ko_recap_block_hash = Hash128::default();
            self.ko_captures_in_encore.clear();

            self.ko_hash_history.clear();
            self.ko_hash_history.push(get_ko_hash(&self.rules, board, get_opp(move_pla), self.encore_phase, self.ko_recap_block_hash));
            //The first ko hash history is the one for the move we JUST appended to the move history earlier.
            self.first_turn_idx_with_ko_history = self.move_history.len();
          }
        }
      } else {
        unreachable!();
      }
    }

    //Break long cycles with no-result
    if move_loc != Board::PASS_LOC && (self.encore_phase > 0 || self.rules.ko_rule == Rules::KO_SIMPLE) {
      if self.number_of_ko_hash_occurrences_in_history(self.ko_hash_history[self.ko_hash_history.len() - 1], root_ko_hash_table) >= 3 {
        self.is_no_result = true;
        self.is_game_finished = true;
      }
    }
  }

  /// upstream: BoardHistory::hasBlackPassOrWhiteFirst (boardhistory.cpp)
  //Heuristically check if this history looks like an sgf variation where black passed to effectively
  //turn into white, or similar.
  pub fn has_black_pass_or_white_first(&self) -> bool {
    //First move was made by white this game, on an empty board.
    if self.initial_board.is_empty() && !self.move_history.is_empty() && self.move_history[0].pla == P_WHITE {
      return true;
    }
    //Black passed exactly once or white doublemoved
    let mut num_black_passes: i32 = 0;
    let mut num_white_passes: i32 = 0;
    let mut num_black_double_moves: i32 = 0;
    let mut num_white_double_moves: i32 = 0;
    for i in 0..self.move_history.len() {
      if self.move_history[i].loc == Board::PASS_LOC && self.move_history[i].pla == P_BLACK {
        num_black_passes += 1;
      }
      if self.move_history[i].loc == Board::PASS_LOC && self.move_history[i].pla == P_WHITE {
        num_white_passes += 1;
      }
      if i > 0 && self.move_history[i].pla == P_BLACK && self.move_history[i - 1].pla == P_BLACK {
        num_black_double_moves += 1;
      }
      if i > 0 && self.move_history[i].pla == P_WHITE && self.move_history[i - 1].pla == P_WHITE {
        num_white_double_moves += 1;
      }
    }
    if num_black_passes == 1 && num_white_passes == 0 && num_black_double_moves == 0 && num_white_double_moves == 0 {
      return true;
    }
    if num_black_passes == 0 && num_white_passes == 0 && num_black_double_moves == 0 && num_white_double_moves == 1 {
      return true;
    }

    false
  }

  /// upstream: BoardHistory::getSituationAndSimpleKoHash (boardhistory.cpp)
  //Compute a hash that takes into account the full situation and simple ko prohibition. Does NOT include rules or history.
  pub fn get_situation_and_simple_ko_hash(board: &Board, next_player: Player) -> Hash128 {
    //Note that board.pos_hash also incorporates the size of the board.
    let mut hash: Hash128 = board.pos_hash;
    hash ^= Board::statics().ZOBRIST_PLAYER_HASH[next_player as usize];
    if board.ko_loc != Board::NULL_LOC {
      hash ^= Board::statics().ZOBRIST_KO_LOC_HASH[board.ko_loc as usize];
    }
    hash
  }

  /// upstream: BoardHistory::getSituationAndSimpleKoAndPrevPosHash (boardhistory.cpp)
  //Compute a hash that takes into account the full situation, simple ko prohibition, and the previous turn's position. (Does NOT include rules).
  pub fn get_situation_and_simple_ko_and_prev_pos_hash(board: &Board, hist: &BoardHistory, next_player: Player) -> Hash128 {
    //Note that board.pos_hash also incorporates the size of the board.
    let mut hash: Hash128 = board.pos_hash;
    hash ^= Board::statics().ZOBRIST_PLAYER_HASH[next_player as usize];
    if board.ko_loc != Board::NULL_LOC {
      hash ^= Board::statics().ZOBRIST_KO_LOC_HASH[board.ko_loc as usize];
    }

    let mut mixed: Hash128 = Hash128::default();
    mixed.hash1 = Hash::rrmxmx(hash.hash0);
    mixed.hash0 = Hash::split_mix64(hash.hash1);
    if !hist.move_history.is_empty() {
      mixed ^= hist.get_recent_board(1).pos_hash;
    }
    mixed
  }

  /// upstream: BoardHistory::getSituationRulesAndKoHash (boardhistory.cpp)
  //Compute a hash that takes into account the full situation, the rules, discretized komi, and any immediate ko prohibitions.
  pub fn get_situation_rules_and_ko_hash(board: &Board, hist: &BoardHistory, next_player: Player, draw_equivalent_wins_for_white: f64) -> Hash128 {
    BoardHistory::get_situation_rules_and_ko_hash_modes(board, hist, next_player, draw_equivalent_wins_for_white, &hist.modes)
  }

  /// upstream: BoardHistory::getSituationRulesAndKoHash (boardhistory.cpp)
  //Same, but hashing as if hist.modes had the given value, for callers
  //(e.g. nn featurization with a per-query override) that use a different value than the history's own.
  pub fn get_situation_rules_and_ko_hash_modes(
    board: &Board, hist: &BoardHistory, next_player: Player, draw_equivalent_wins_for_white: f64,
    modes: &BoardHistoryModes
  ) -> Hash128 {
    let x_size: i32 = board.x_size;
    let y_size: i32 = board.y_size;
    let statics = Board::statics();

    //Note that board.pos_hash also incorporates the size of the board.
    let mut hash: Hash128 = board.pos_hash;
    hash ^= statics.ZOBRIST_PLAYER_HASH[next_player as usize];

    testAssert!(hist.encore_phase >= 0 && hist.encore_phase <= 2);
    hash ^= statics.ZOBRIST_ENCORE_HASH[hist.encore_phase as usize];

    if hist.encore_phase == 0 {
      if board.ko_loc != Board::NULL_LOC {
        hash ^= statics.ZOBRIST_KO_LOC_HASH[board.ko_loc as usize];
      }
      for y in 0..y_size {
        for x in 0..x_size {
          let loc: Loc = Location::get_loc(x, y, x_size);
          if hist.super_ko_banned[loc as usize] && loc != board.ko_loc {
            hash ^= statics.ZOBRIST_KO_LOC_HASH[loc as usize];
          }
        }
      }
    } else {
      for y in 0..y_size {
        for x in 0..x_size {
          let loc: Loc = Location::get_loc(x, y, x_size);
          if hist.super_ko_banned[loc as usize] {
            hash ^= statics.ZOBRIST_KO_LOC_HASH[loc as usize];
          }
          if hist.ko_recap_blocked[loc as usize] {
            hash ^= statics.ZOBRIST_KO_MARK_HASH[loc as usize][P_BLACK as usize] ^ statics.ZOBRIST_KO_MARK_HASH[loc as usize][P_WHITE as usize];
          }
        }
      }
      if hist.encore_phase == 2 {
        for y in 0..y_size {
          for x in 0..x_size {
            let loc: Loc = Location::get_loc(x, y, x_size);
            let c: Color = hist.second_encore_start_colors[loc as usize];
            if c != C_EMPTY {
              hash ^= statics.ZOBRIST_SECOND_ENCORE_START_HASH[loc as usize][c as usize];
            }
          }
        }
      }
    }

    let self_komi: f32 = hist.current_self_komi(next_player, draw_equivalent_wins_for_white);

    //Discretize the komi for the purpose of matching hash
    // PORT NOTE: C++ UB; x86-64 gives INT64_MIN. `(int64_t)(selfKomi*256.0f)` is out of range or NaN
    // when selfKomi is (setKomi does not validate, and a NaN/inf draw utility makes it NaN/inf);
    // Rust `as` would saturate, so port::cpp::static_cast_i64_f32 reproduces cvttss2si.
    let komi_discretized: i64 = cpp::static_cast_i64_f32(self_komi * 256.0f32);
    let komi_hash: u64 = Hash::murmur_mix(komi_discretized as u64);
    hash.hash0 ^= komi_hash;
    hash.hash1 ^= Hash::basic_l_cong(komi_hash);

    //Fold in the ko, scoring, and suicide rules
    hash ^= Rules::ZOBRIST_KO_RULE_HASH[hist.rules.ko_rule as usize];
    hash ^= Rules::ZOBRIST_SCORING_RULE_HASH[hist.rules.scoring_rule as usize];
    hash ^= Rules::ZOBRIST_TAX_RULE_HASH[hist.rules.tax_rule as usize];
    if hist.rules.multi_stone_suicide_legal {
      hash ^= Rules::ZOBRIST_MULTI_STONE_SUICIDE_HASH;
    }
    if hist.has_button {
      hash ^= Rules::ZOBRIST_BUTTON_HASH;
    }
    if hist.rules.friendly_pass_ok {
      hash ^= Rules::ZOBRIST_FRIENDLY_PASS_OK_HASH;
    }

    //Fold in whether pass-alive computations are being performed as if suicide were legal, when that
    //differs from what the suicide rule alone would give. When the rules already have suicide legal
    //the flag is a no-op, and we deliberately don't fold it then, so that caches can be shared.
    if modes.always_compute_pass_alive_under_suicide_rules && !hist.rules.multi_stone_suicide_legal {
      hash ^= Rules::ZOBRIST_PASS_ALIVE_UNDER_SUICIDE_HASH;
    }

    //Fold in whether territory scoring excludes points adjacent to atari, but only under the rules
    //where the flag has any effect (territory scoring with no seki tax), so that caches can be shared
    //between the modes under all other rules.
    if modes.exclude_territory_adjacent_to_atari && hist.rules.scoring_rule == Rules::SCORING_TERRITORY && hist.rules.tax_rule == Rules::TAX_NONE {
      hash ^= Rules::ZOBRIST_EXCLUDE_TERRITORY_ADJ_ATARI_HASH;
    }

    hash
  }
}

/// upstream: BoardHistory::BoardHistory(const BoardHistory&), BoardHistory::operator=(const BoardHistory&) (boardhistory.cpp)
/// PORT NOTE: both copy every field; `operator=`'s self-assignment check is implicit in Rust.
impl Clone for BoardHistory {
  fn clone(&self) -> BoardHistory {
    BoardHistory {
      rules: self.rules,
      move_history: self.move_history.clone(),
      prevent_encore_history: self.prevent_encore_history.clone(),
      ko_hash_history: self.ko_hash_history.clone(),
      first_turn_idx_with_ko_history: self.first_turn_idx_with_ko_history,
      initial_board: self.initial_board.clone(),
      initial_pla: self.initial_pla,
      initial_encore_phase: self.initial_encore_phase,
      initial_turn_number: self.initial_turn_number,
      assume_multiple_starting_black_moves_are_handicap: self.assume_multiple_starting_black_moves_are_handicap,
      white_has_moved: self.white_has_moved,
      override_num_handicap_stones: self.override_num_handicap_stones,
      modes: self.modes,
      recent_boards: self.recent_boards.clone(),
      current_recent_board_idx: self.current_recent_board_idx,
      presumed_next_move_pla: self.presumed_next_move_pla,
      was_ever_occupied_or_played: self.was_ever_occupied_or_played,
      super_ko_banned: self.super_ko_banned,
      consecutive_ending_passes: self.consecutive_ending_passes,
      hashes_before_black_pass: self.hashes_before_black_pass.clone(),
      hashes_before_white_pass: self.hashes_before_white_pass.clone(),
      encore_phase: self.encore_phase,
      num_turns_this_phase: self.num_turns_this_phase,
      num_approx_valid_turns_this_phase: self.num_approx_valid_turns_this_phase,
      num_consec_valid_turns_this_game: self.num_consec_valid_turns_this_game,
      ko_recap_blocked: self.ko_recap_blocked,
      ko_recap_block_hash: self.ko_recap_block_hash,
      ko_captures_in_encore: self.ko_captures_in_encore.clone(),
      second_encore_start_colors: self.second_encore_start_colors,
      white_bonus_score: self.white_bonus_score,
      white_handicap_bonus_score: self.white_handicap_bonus_score,
      has_button: self.has_button,
      is_past_normal_phase_end: self.is_past_normal_phase_end,
      is_game_finished: self.is_game_finished,
      winner: self.winner,
      final_white_minus_black_score: self.final_white_minus_black_score,
      is_scored: self.is_scored,
      is_no_result: self.is_no_result,
      is_resignation: self.is_resignation,
    }
  }
}

/// upstream: KoHashTable (boardhistory.h)
pub struct KoHashTable {
  pub idx_table: Box<[u32; KoHashTable::TABLE_SIZE as usize]>,
  pub ko_hash_history_sorted_by_low_bits: Vec<Hash128>,
  pub first_turn_idx_with_ko_history: usize,
}

impl KoHashTable {
  /// upstream: KoHashTable::TABLE_SIZE, KoHashTable::TABLE_MASK (boardhistory.h)
  pub const TABLE_SIZE: i32 = 1 << 10;
  pub const TABLE_MASK: u64 = (KoHashTable::TABLE_SIZE - 1) as u64;

  /// upstream: KoHashTable::KoHashTable (boardhistory.cpp)
  pub fn new() -> KoHashTable {
    let mut s = KoHashTable {
      idx_table: Box::new([0u32; KoHashTable::TABLE_SIZE as usize]),
      ko_hash_history_sorted_by_low_bits: Vec::new(),
      first_turn_idx_with_ko_history: 0,
    };
    s.idx_table.fill(0u32);
    s
  }

  // upstream: KoHashTable::~KoHashTable (boardhistory.cpp)
  // PORT NOTE: `delete[] idxTable` is the Box drop; no Drop impl. Copy ctor/assignment are deleted: no Clone.

  /// upstream: KoHashTable::size (boardhistory.cpp)
  pub fn size(&self) -> usize {
    self.ko_hash_history_sorted_by_low_bits.len()
  }

  /// upstream: KoHashTable::recompute (boardhistory.cpp)
  pub fn recompute(&mut self, history: &BoardHistory) -> Result<(), KataError> {
    self.ko_hash_history_sorted_by_low_bits = history.ko_hash_history.clone();
    self.first_turn_idx_with_ko_history = history.first_turn_idx_with_ko_history;

    let cmp_first_by_low_bits = |a: &Hash128, b: &Hash128| -> bool {
      if (a.hash0 & KoHashTable::TABLE_MASK) < (b.hash0 & KoHashTable::TABLE_MASK) {
        return true;
      }
      if (a.hash0 & KoHashTable::TABLE_MASK) > (b.hash0 & KoHashTable::TABLE_MASK) {
        return false;
      }
      a < b
    };

    // PORT NOTE: std::sort -> sort_unstable_by: elements that compare equal are identical hashes (PORTING.md §5).
    self.ko_hash_history_sorted_by_low_bits.sort_unstable_by(cmp_from_lt(cmp_first_by_low_bits));

    //Just in case, since we're using 32 bits for indices.
    if self.ko_hash_history_sorted_by_low_bits.len() > 1000000000 {
      return Err(KataError::StringError(String::from("Board history length longer than 1000000000, not supported")));
    }
    let size: u32 = self.ko_hash_history_sorted_by_low_bits.len() as u32;

    let mut idx: u32 = 0;
    for bits in 0..KoHashTable::TABLE_SIZE as u32 {
      while idx < size && ((self.ko_hash_history_sorted_by_low_bits[idx as usize].hash0 & KoHashTable::TABLE_MASK) < bits as u64) {
        idx += 1;
      }
      self.idx_table[bits as usize] = idx;
    }
    Ok(())
  }

  /// upstream: KoHashTable::containsHash (boardhistory.cpp)
  pub fn contains_hash(&self, hash: Hash128) -> bool {
    let bits: u32 = (hash.hash0 & KoHashTable::TABLE_MASK) as u32;
    let mut idx: usize = self.idx_table[bits as usize] as usize;
    let size: usize = self.ko_hash_history_sorted_by_low_bits.len();
    while idx < size && ((self.ko_hash_history_sorted_by_low_bits[idx].hash0 & KoHashTable::TABLE_MASK) == bits as u64) {
      if hash == self.ko_hash_history_sorted_by_low_bits[idx] {
        return true;
      }
      idx += 1;
    }
    false
  }

  /// upstream: KoHashTable::numberOfOccurrencesOfHash (boardhistory.cpp)
  pub fn number_of_occurrences_of_hash(&self, hash: Hash128) -> i32 {
    let bits: u32 = (hash.hash0 & KoHashTable::TABLE_MASK) as u32;
    let mut idx: usize = self.idx_table[bits as usize] as usize;
    let size: usize = self.ko_hash_history_sorted_by_low_bits.len();
    let mut count: i32 = 0;
    while idx < size && ((self.ko_hash_history_sorted_by_low_bits[idx].hash0 & KoHashTable::TABLE_MASK) == bits as u64) {
      if hash == self.ko_hash_history_sorted_by_low_bits[idx] {
        count += 1;
      }
      idx += 1;
    }
    count
  }
}
