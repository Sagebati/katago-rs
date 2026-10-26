// Ported from KataGo cpp/search/patternbonustable.cpp, cpp/search/patternbonustable.h @ fd0723fdbc0e9d82cf269c9630af8c27c57c07c4
//
// Not ported (file IO and SGF loading: core does no IO, dataio/sgf is not ported, M8; see the
// manifest notes): PatternBonusTable::avoidRepeatedSgfMoves and
// PatternBonusTable::avoidRepeatedPosMovesAndDeleteExcessFiles.

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::sync::OnceLock;

use crate::core::global;
use crate::core::hash::Hash128;
use crate::core::rand::Rand;
use crate::game::board::Board;
use crate::game::board::Loc;
use crate::game::board::Player;
use crate::game::board::C_EMPTY;
use crate::game::boardhistory::BoardHistory;
use crate::neuralnet::nninputs::SymmetryHelpers;
use crate::search::localpattern::LocalPatternHasher;

/// upstream: PatternBonusEntry (patternbonustable.h)
// All bonuses are bonuses to white's utility for the pattern occuring on the board.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PatternBonusEntry {
  pub utility_bonus: f64,
}

/// upstream: PatternBonusTable (patternbonustable.h)
pub struct PatternBonusTable {
  pub entries: Vec<BTreeMap<Hash128, PatternBonusEntry>>,
}

/// upstream: (file statics) initMutex, isInited, patternHasher, ZOBRIST_MOVE_LOCS (patternbonustable.cpp) static
/// PORT NOTE: the runtime-initialized file statics live in one struct behind a OnceLock
/// (PORTING.md §8, OWNERSHIP.tsv); `isInited` and `initMutex` are replaced by the OnceLock state.
pub struct PatternBonusTableStatics {
  pub patternHasher: LocalPatternHasher,
  pub ZOBRIST_MOVE_LOCS: [Hash128; Board::MAX_ARR_SIZE as usize],
}

static PATTERNBONUSTABLE_STATICS: OnceLock<PatternBonusTableStatics> = OnceLock::new();

/// upstream: initIfNeeded (patternbonustable.cpp) static
/// PORT NOTE: the body of the C++ `initIfNeeded` (RNG call order unchanged) is `compute_statics`;
/// `initIfNeeded` forces the OnceLock and returns the tables.
fn compute_statics() -> PatternBonusTableStatics {
  let mut patternHasher: LocalPatternHasher = LocalPatternHasher::new();
  let mut ZOBRIST_MOVE_LOCS: [Hash128; Board::MAX_ARR_SIZE as usize] = [Hash128::default(); Board::MAX_ARR_SIZE as usize];

  let mut rand: Rand = Rand::new_seed("PatternBonusTable ZOBRIST STUFF");
  patternHasher.init(9, 9, &mut rand);

  rand.init_seed("Reseed PatternBonusTable zobrist so that zobrists don't change when Board::MAX_ARR_SIZE changes");
  for i in 0..Board::MAX_ARR_SIZE {
    let h0: u64 = rand.next_uint64();
    let h1: u64 = rand.next_uint64();
    ZOBRIST_MOVE_LOCS[i as usize] = Hash128::new(h0, h1);
  }

  PatternBonusTableStatics {
    patternHasher,
    ZOBRIST_MOVE_LOCS,
  }
}

/// upstream: initIfNeeded (patternbonustable.cpp) static
fn initIfNeeded() -> &'static PatternBonusTableStatics {
  PATTERNBONUSTABLE_STATICS.get_or_init(compute_statics)
}

impl PatternBonusTable {
  /// upstream: PatternBonusTable::PatternBonusTable (patternbonustable.cpp)
  pub fn new() -> PatternBonusTable {
    initIfNeeded();
    let mut entries: Vec<BTreeMap<Hash128, PatternBonusEntry>> = Vec::new();
    entries.resize_with(1024, BTreeMap::new);
    PatternBonusTable { entries }
  }

  /// upstream: PatternBonusTable::PatternBonusTable (patternbonustable.cpp)
  pub fn new_num_shards(num_shards: i32) -> PatternBonusTable {
    initIfNeeded();
    let mut entries: Vec<BTreeMap<Hash128, PatternBonusEntry>> = Vec::new();
    entries.resize_with(num_shards as usize, BTreeMap::new);
    PatternBonusTable { entries }
  }

  /// upstream: PatternBonusTable::getHash (patternbonustable.cpp)
  // The board specified here is expected to be the board BEFORE the move is played.
  pub fn get_hash(&self, pla: Player, move_loc: Loc, board: &Board) -> Hash128 {
    let st: &PatternBonusTableStatics = initIfNeeded();
    //We don't want to over-trigger this on a ko that repeats the same pattern over and over
    //So we just disallow this on ko fight
    //Also no bonuses for passing.
    if move_loc == Board::NULL_LOC || move_loc == Board::PASS_LOC || board.would_be_ko_capture(move_loc, pla) {
      return Hash128::default();
    }

    let mut hash: Hash128 = st.patternHasher.get_hash(board, move_loc, pla);
    hash ^= st.ZOBRIST_MOVE_LOCS[move_loc as usize];
    hash ^= Board::statics().ZOBRIST_SIZE_X_HASH[board.x_size as usize];
    hash ^= Board::statics().ZOBRIST_SIZE_Y_HASH[board.y_size as usize];

    hash
  }

  /// upstream: PatternBonusTable::get (patternbonustable.cpp)
  pub fn get(&self, hash: Hash128) -> PatternBonusEntry {
    //Hash 0 indicates to not do anything. If anything legit collides with it, then it will do nothing
    //but this should be very rare.
    if hash == Hash128::default() {
      return PatternBonusEntry::default();
    }

    let sub_map_idx: u64 = hash.hash0 % self.entries.len() as u64;

    let sub_map: &BTreeMap<Hash128, PatternBonusEntry> = &self.entries[sub_map_idx as usize];
    match sub_map.get(&hash) {
      None => PatternBonusEntry::default(),
      Some(entry) => *entry,
    }
  }

  /// upstream: PatternBonusTable::get (patternbonustable.cpp)
  // The board specified here is expected to be the board BEFORE the move is played.
  pub fn get_pla_move_loc_board(&self, pla: Player, move_loc: Loc, board: &Board) -> PatternBonusEntry {
    let hash: Hash128 = self.get_hash(pla, move_loc, board);
    self.get(hash)
  }

  /// upstream: PatternBonusTable::addBonus (patternbonustable.cpp)
  // All bonuses are bonuses to white's utility for the pattern occuring on the board.
  // The board specified here is expected to be the board BEFORE the move is played.
  pub fn add_bonus(
    &mut self,
    pla: Player,
    move_loc: Loc,
    board: &Board,
    bonus: f64,
    symmetry: i32,
    flip_colors: bool,
    hashes_this_game: &mut BTreeSet<Hash128>,
  ) {
    let st: &PatternBonusTableStatics = initIfNeeded();
    //We don't want to over-trigger this on a ko that repeats the same pattern over and over
    //So we just disallow this on ko fight
    //Also no bonuses for passing.
    if move_loc == Board::NULL_LOC || move_loc == Board::PASS_LOC || board.would_be_ko_capture(move_loc, pla) {
      return;
    }

    let mut hash: Hash128 = st.patternHasher.get_hash_with_sym(board, move_loc, pla, symmetry, flip_colors);
    hash ^= st.ZOBRIST_MOVE_LOCS[SymmetryHelpers::get_sym_loc(move_loc, board, symmetry) as usize];
    if SymmetryHelpers::is_transpose(symmetry) {
      hash ^= Board::statics().ZOBRIST_SIZE_X_HASH[board.y_size as usize];
      hash ^= Board::statics().ZOBRIST_SIZE_Y_HASH[board.x_size as usize];
    } else {
      hash ^= Board::statics().ZOBRIST_SIZE_X_HASH[board.x_size as usize];
      hash ^= Board::statics().ZOBRIST_SIZE_Y_HASH[board.y_size as usize];
    }

    if global::contains_set(hashes_this_game, &hash) {
      return;
    }
    hashes_this_game.insert(hash);

    let sub_map_idx: u64 = hash.hash0 % self.entries.len() as u64;

    let sub_map: &mut BTreeMap<Hash128, PatternBonusEntry> = &mut self.entries[sub_map_idx as usize];
    // PORT NOTE: std::map::operator[] inserts a default entry when the key is missing (PORTING.md §5).
    sub_map.entry(hash).or_default().utility_bonus += bonus;
  }

  /// upstream: PatternBonusTable::addBonusForGameMoves (patternbonustable.cpp)
  pub fn add_bonus_for_game_moves(&mut self, game: &BoardHistory, bonus: f64) {
    self.add_bonus_for_game_moves_only_pla(game, bonus, C_EMPTY);
  }

  /// upstream: PatternBonusTable::addBonusForGameMoves (patternbonustable.cpp)
  pub fn add_bonus_for_game_moves_only_pla(&mut self, game: &BoardHistory, bonus: f64, only_pla: Player) {
    let mut hashes_this_game: BTreeSet<Hash128> = BTreeSet::new();
    let mut board: Board = game.initial_board.clone();
    //Replay under the same BoardHistoryModes as the history we're replaying.
    let mut hist: Box<BoardHistory> = Box::new(BoardHistory::new_full(&board, game.initial_pla, &game.rules, game.initial_encore_phase, &game.modes));
    for i in 0..game.move_history.len() {
      let pla: Player = game.move_history[i].pla;
      let loc: Loc = game.move_history[i].loc;
      //We first play the move to see if it's a move we can accept
      let suc: bool = hist.make_board_move_tolerant(&mut board, loc, pla);
      if !suc {
        break;
      }
      if only_pla == C_EMPTY || only_pla == pla {
        for flip_colors in 0..2 {
          for symmetry in 0..8 {
            //getRecentBoard(1) - the convention is to pattern match on the board BEFORE the move is played.
            //This is also more pricipled than convening on the board after since with different captures, moves
            //may have different effects even while leading to the same position.
            self.add_bonus(pla, loc, hist.get_recent_board(1), bonus, symmetry, flip_colors != 0, &mut hashes_this_game);
          }
        }
      }
    }
  }
}

/// upstream: PatternBonusTable::PatternBonusTable(const PatternBonusTable&) (patternbonustable.cpp)
impl Clone for PatternBonusTable {
  fn clone(&self) -> PatternBonusTable {
    initIfNeeded();
    PatternBonusTable {
      entries: self.entries.clone(),
    }
  }
}
