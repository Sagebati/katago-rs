// Ported from KataGo cpp/search/searchnodetable.cpp, cpp/search/searchnodetable.h @ fd0723fdbc0e9d82cf269c9630af8c27c57c07c4

use std::cell::RefCell;
use std::collections::BTreeMap;

use crate::core::hash::Hash128;
use crate::port::node_arena::NodeIdx;

/// upstream: SearchNodeTable (searchnodetable.h)
pub struct SearchNodeTable {
  /// PORT NOTE (OWNERSHIP.tsv): the nodes are owned by Search's NodeArena; the maps hold their
  /// indices. RefCell: allocateOrFindNode inserts during search through &self (its upstream guard
  /// was the shard mutex).
  pub entries: RefCell<Vec<BTreeMap<Hash128, NodeIdx>>>,
  /// PORT NOTE (OWNERSHIP.tsv): `MutexPool* mutexPool` removed (single-threaded); only its size
  /// is kept, because Search's constructor reads `nodeTable->mutexPool->getNumMutexes()`.
  pub mutex_pool_num_mutexes: u32,
  pub num_shards: u32,
}

impl SearchNodeTable {
  /// upstream: SearchNodeTable::SearchNodeTable (searchnodetable.cpp)
  pub fn new(num_shards_power_of_two: i32) -> SearchNodeTable {
    let num_shards: u32 = 1u32 << num_shards_power_of_two;
    // PORT NOTE: `mutexPool = new MutexPool(numShards)` -> its mutex count (single-threaded).
    let mutex_pool_num_mutexes: u32 = num_shards;
    let mut entries: Vec<BTreeMap<Hash128, NodeIdx>> = Vec::new();
    entries.resize_with(num_shards as usize, BTreeMap::new);
    SearchNodeTable { entries: RefCell::new(entries), mutex_pool_num_mutexes, num_shards }
  }

  /// upstream: SearchNodeTable::getIndex (searchnodetable.cpp)
  pub fn get_index(&self, hash: u64) -> u32 {
    let mutex_pool_mask: u32 = self.num_shards - 1; //Always a power of two
    (hash & mutex_pool_mask as u64) as u32
  }
}

// ~SearchNodeTable only deletes the mutex pool: omitted (PORTING.md §5).
