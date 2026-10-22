// Ported from KataGo cpp/search/subtreevaluebiastable.cpp, cpp/search/subtreevaluebiastable.h @ ba938676d7f42d70950b3a535af2466fb642008c

use std::cell::Cell;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;
use std::sync::OnceLock;

use crate::core::hash::Hash128;
use crate::core::rand::Rand;
use crate::game::board::Board;
use crate::game::board::Loc;
use crate::game::board::Player;
use crate::search::localpattern::LocalPatternHasher;

/// upstream: SubtreeValueBiasEntry (subtreevaluebiastable.h)
pub struct SubtreeValueBiasEntry {
  // PORT NOTE: guarded by entryLock upstream; single-threaded, so plain Cells (OWNERSHIP.tsv).
  pub delta_utility_sum: Cell<f64>,
  pub weight_sum: Cell<f64>,
  // PORT NOTE: `mutable std::atomic_flag entryLock` (spin lock) removed (single-threaded).
}

impl SubtreeValueBiasEntry {
  /// upstream: SubtreeValueBiasEntry (subtreevaluebiastable.h)
  /// PORT NOTE: the default member initializers (`= 0.0`) of the C++ aggregate.
  pub fn new() -> SubtreeValueBiasEntry {
    SubtreeValueBiasEntry {
      delta_utility_sum: Cell::new(0.0),
      weight_sum: Cell::new(0.0),
    }
  }
}

/// upstream: SubtreeValueBiasTable (subtreevaluebiastable.h)
pub struct SubtreeValueBiasTable {
  // PORT NOTE: entries is mutated by get() from the playout loop through &self, so it is a RefCell
  // (its upstream guard was the per-shard mutex from mutexPool, which is removed). The borrow scope
  // in get() equals the upstream lock scope and never spans an await.
  pub entries: RefCell<Vec<BTreeMap<Hash128, Rc<SubtreeValueBiasEntry>>>>,
  // PORT NOTE: `MutexPool* mutexPool` removed (single-threaded); the shard index is still computed.
}

/// upstream: (file statics) initMutex, isInited, patternHasher, ZOBRIST_MOVE_LOCS, ZOBRIST_KO_BAN (subtreevaluebiastable.cpp) static
/// PORT NOTE: the runtime-initialized file statics live in one struct behind a OnceLock
/// (PORTING.md §8, OWNERSHIP.tsv); `isInited` and `initMutex` are replaced by the OnceLock state.
pub struct SubtreeValueBiasTableStatics {
  pub patternHasher: LocalPatternHasher,
  pub ZOBRIST_MOVE_LOCS: [[Hash128; 2]; Board::MAX_ARR_SIZE as usize],
  pub ZOBRIST_KO_BAN: [Hash128; Board::MAX_ARR_SIZE as usize],
}

static SUBTREEVALUEBIASTABLE_STATICS: OnceLock<SubtreeValueBiasTableStatics> = OnceLock::new();

/// upstream: initIfNeeded (subtreevaluebiastable.cpp) static
/// PORT NOTE: the body of the C++ `initIfNeeded` (RNG call order unchanged) is `compute_statics`;
/// `initIfNeeded` forces the OnceLock and returns the tables.
fn compute_statics() -> SubtreeValueBiasTableStatics {
  let mut patternHasher: LocalPatternHasher = LocalPatternHasher::new();
  let mut ZOBRIST_MOVE_LOCS: [[Hash128; 2]; Board::MAX_ARR_SIZE as usize] = [[Hash128::default(); 2]; Board::MAX_ARR_SIZE as usize];
  let mut ZOBRIST_KO_BAN: [Hash128; Board::MAX_ARR_SIZE as usize] = [Hash128::default(); Board::MAX_ARR_SIZE as usize];

  let mut rand: Rand = Rand::new_seed("ValueBiasTable ZOBRIST STUFF");
  patternHasher.init(5, 5, &mut rand);

  for i in 0..Board::MAX_ARR_SIZE {
    for j in 0..2 {
      let h0: u64 = rand.next_uint64();
      let h1: u64 = rand.next_uint64();
      ZOBRIST_MOVE_LOCS[i as usize][j as usize] = Hash128::new(h0, h1);
    }
  }

  rand.init_seed("Reseed ValueBiasTable zobrist so that zobrists don't change when Board::MAX_ARR_SIZE changes");
  for i in 0..Board::MAX_ARR_SIZE {
    let h0: u64 = rand.next_uint64();
    let h1: u64 = rand.next_uint64();
    ZOBRIST_KO_BAN[i as usize] = Hash128::new(h0, h1);
  }

  SubtreeValueBiasTableStatics {
    patternHasher,
    ZOBRIST_MOVE_LOCS,
    ZOBRIST_KO_BAN,
  }
}

/// upstream: initIfNeeded (subtreevaluebiastable.cpp) static
fn initIfNeeded() -> &'static SubtreeValueBiasTableStatics {
  SUBTREEVALUEBIASTABLE_STATICS.get_or_init(compute_statics)
}

impl SubtreeValueBiasTable {
  /// upstream: SubtreeValueBiasTable::SubtreeValueBiasTable (subtreevaluebiastable.cpp)
  pub fn new(num_shards: i32) -> SubtreeValueBiasTable {
    initIfNeeded();
    // PORT NOTE: `mutexPool = new MutexPool(numShards)` removed (single-threaded).
    let mut entries: Vec<BTreeMap<Hash128, Rc<SubtreeValueBiasEntry>>> = Vec::new();
    entries.resize_with(num_shards as usize, BTreeMap::new);
    SubtreeValueBiasTable {
      entries: RefCell::new(entries),
    }
  }

  /// upstream: SubtreeValueBiasTable::clearUnusedSynchronous (subtreevaluebiastable.cpp)
  // ASSUMES there is no concurrent multithreading of this table or any of its entries,
  // and that all past mutations on this table or any of its entries are now visible to this thread.
  pub fn clear_unused_synchronous(&mut self) {
    let mut entries = self.entries.borrow_mut();
    for i in 0..entries.len() {
      let submap: &mut BTreeMap<Hash128, Rc<SubtreeValueBiasEntry>> = &mut entries[i];
      // PORT NOTE: the erase loop becomes retain; the visit order is the map order and erase has no
      // other side effect, so the result is identical.
      // Anything in this map NOT used by anyone else - clear
      submap.retain(|_, v| !(Rc::strong_count(v) <= 1));
    }
  }

  /// upstream: SubtreeValueBiasTable::get (subtreevaluebiastable.cpp)
  // The board specified here is expected to be the board BEFORE the move is played.
  pub fn get(&self, pla: Player, parent_prev_move_loc: Loc, prev_move_loc: Loc, prev_board: &Board) -> Rc<SubtreeValueBiasEntry> {
    let statics: &SubtreeValueBiasTableStatics = initIfNeeded();
    let mut hash: Hash128 = statics.ZOBRIST_MOVE_LOCS[parent_prev_move_loc as usize][0] ^ statics.ZOBRIST_MOVE_LOCS[prev_move_loc as usize][1];

    hash ^= statics.patternHasher.get_hash(prev_board, prev_move_loc, pla);
    if prev_board.ko_loc != Board::NULL_LOC {
      hash ^= statics.ZOBRIST_KO_BAN[prev_board.ko_loc as usize];
    }

    let mut entries = self.entries.borrow_mut();
    let sub_map_idx: u32 = (hash.hash0 % entries.len() as u64) as u32;

    // PORT NOTE: `mutexPool->getMutex(subMapIdx)` + `lock_guard` removed (single-threaded); the
    // RefCell borrow above spans the same scope.
    // PORT NOTE: `slot = entries[subMapIdx][hash]; if(slot == nullptr) slot = make_shared<..>()`
    // is `entry().or_insert_with` (operator[] inserts a null that is immediately replaced).
    let slot: &mut Rc<SubtreeValueBiasEntry> = entries[sub_map_idx as usize]
      .entry(hash)
      .or_insert_with(|| Rc::new(SubtreeValueBiasEntry::new()));
    slot.clone()
  }
}
