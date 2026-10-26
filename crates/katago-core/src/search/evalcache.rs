// Ported from KataGo cpp/search/evalcache.cpp, cpp/search/evalcache.h @ fd0723fdbc0e9d82cf269c9630af8c27c57c07c4

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

use crate::core::hash::Hash128;
use crate::game::board::Board;
use crate::game::board::Loc;
use crate::game::board::P_WHITE;
use crate::port::cpp::cpp_max;
use crate::port::node_arena::NodeArena;
use crate::port::node_arena::NodeIdx;
use crate::search::searchnode::ConstSearchNodeChildrenReference;
use crate::search::searchnode::SearchChildPointer;
use crate::search::searchnode::SearchNode;

/// upstream: FirstExploreEval (evalcache.h)
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FirstExploreEval {
  pub avg_win_loss: f32,
  pub avg_score_mean: f32,
  pub cache_weight: f32,
}

impl FirstExploreEval {
  /// upstream: FirstExploreEval::FirstExploreEval (evalcache.cpp)
  pub fn new() -> FirstExploreEval {
    FirstExploreEval {
      avg_win_loss: 0.0f32,
      avg_score_mean: 0.0f32,
      cache_weight: 0.0f32,
    }
  }

  /// upstream: FirstExploreEval::FirstExploreEval (evalcache.cpp)
  pub fn new_full(avg_wl: f32, avg_sm: f32, w: f32) -> FirstExploreEval {
    FirstExploreEval {
      avg_win_loss: avg_wl,
      avg_score_mean: avg_sm,
      cache_weight: w,
    }
  }
}

/// PORT NOTE: `std::map<Loc,FirstExploreEval>::operator[]` default-constructs the value.
impl Default for FirstExploreEval {
  fn default() -> FirstExploreEval {
    FirstExploreEval::new()
  }
}

/// upstream: EvalCacheEntry (evalcache.h)
#[derive(Clone, Debug)]
pub struct EvalCacheEntry {
  pub avg_win_loss: f32,
  pub avg_no_result: f32,
  pub avg_score_mean: f32,
  pub avg_lead: f32,
  pub cache_weight: f32,
  pub first_explore_evals: BTreeMap<Loc, FirstExploreEval>,
}

impl EvalCacheEntry {
  /// upstream: EvalCacheEntry::EvalCacheEntry (evalcache.cpp)
  pub fn new() -> EvalCacheEntry {
    EvalCacheEntry {
      avg_win_loss: 0.0f32,
      avg_no_result: 0.0f32,
      avg_score_mean: 0.0f32,
      avg_lead: 0.0f32,
      cache_weight: 0.0f32,
      first_explore_evals: BTreeMap::new(),
    }
  }
}

/// upstream: EvalCacheTable (evalcache.h)
pub struct EvalCacheTable {
  // PORT NOTE: find/update/clear run from the playout loop and tree walks through &self, so the
  // shards are a RefCell (their upstream guard was the per-shard mutex from mutexPool, which is
  // removed). Values are Option because update() reads them via `operator[]`, which inserts a null
  // (OWNERSHIP.tsv). Borrow scopes equal the upstream lock scopes and never span an await.
  pub entries: RefCell<Vec<BTreeMap<Hash128, Option<Rc<EvalCacheEntry>>>>>,
  // PORT NOTE: `MutexPool* mutexPool` removed (single-threaded); the shard index is still computed.
}

impl EvalCacheTable {
  /// upstream: EvalCacheTable::EvalCacheTable (evalcache.cpp)
  pub fn new(num_shards: i32) -> EvalCacheTable {
    // PORT NOTE: `mutexPool = new MutexPool(numShards);` removed (single-threaded).
    let mut entries: Vec<BTreeMap<Hash128, Option<Rc<EvalCacheEntry>>>> = Vec::new();
    entries.resize_with(num_shards as usize, BTreeMap::new);
    EvalCacheTable {
      entries: RefCell::new(entries),
    }
  }

  // These are all thread-safe, and EvalCacheEntry is only ever updated via replacement, not mutation
  // similar to nnevals.

  /// upstream: EvalCacheTable::find (evalcache.cpp)
  pub fn find(&self, graph_hash: Hash128) -> Option<Rc<EvalCacheEntry>> {
    let entries = self.entries.borrow();
    let sub_map_idx: u32 = (graph_hash.hash0 % entries.len() as u64) as u32;
    // PORT NOTE: lock removed (single-threaded).

    match entries[sub_map_idx as usize].get(&graph_hash) {
      None => None,
      Some(entry) => entry.clone(),
    }
  }

  /// upstream: EvalCacheTable::update (evalcache.cpp)
  /// PORT NOTE: gains `nodes` to dereference the node's children (PORTING.md §4, OWNERSHIP.tsv).
  pub fn update(&self, nodes: &NodeArena<SearchNode>, graph_hash: Hash128, node: &SearchNode, eval_cache_min_visits: i64, is_root_node: bool) {
    let sub_map_idx: u32 = (graph_hash.hash0 % self.entries.borrow().len() as u64) as u32;
    // PORT NOTE: lock removed (single-threaded).

    let old_entry: Option<Rc<EvalCacheEntry>>;

    {
      let mut entries = self.entries.borrow_mut();
      // PORT NOTE: std::map::operator[] inserts a null entry when the key is missing (PORTING.md §5).
      old_entry = entries[sub_map_idx as usize].entry(graph_hash).or_default().clone();
    }

    let new_cache_weight: f32 = node.stats.visits.get() /*acquire*/ as f32;
    if let Some(old) = &old_entry {
      if new_cache_weight < old.cache_weight * 0.75f32 {
        return;
      }
    }

    // Create a new entry, copying from old if it exists
    let mut new_entry: EvalCacheEntry = EvalCacheEntry::new();
    if let Some(old) = &old_entry {
      new_entry.cache_weight = old.cache_weight;
      new_entry.avg_win_loss = old.avg_win_loss;
      new_entry.avg_no_result = old.avg_no_result;
      new_entry.avg_score_mean = old.avg_score_mean;
      new_entry.avg_lead = old.avg_lead;
      new_entry.first_explore_evals = old.first_explore_evals.clone();
    }

    let children: ConstSearchNodeChildrenReference = node.get_children();
    let children_capacity: i32 = children.get_capacity();
    for i in 0..children_capacity {
      let child_idx: NodeIdx = match children.index(i).get_if_allocated() {
        None => break,
        Some(c) => c,
      };
      let child: Rc<SearchNode> = nodes.get(child_idx);

      let child_num_visits: i64 = child.stats.visits.get() /*acquire*/;
      let child_win_loss_avg: f32 = child.stats.win_loss_value_avg.get() /*acquire*/ as f32;
      let child_score_mean_avg: f32 = child.stats.score_mean_avg.get() /*acquire*/ as f32;
      let move_loc: Loc = children.index(i).get_move_loc_relaxed();
      if child_num_visits >= eval_cache_min_visits {
        let child_cache_weight: f32 = child_num_visits as f32;
        let eval: &mut FirstExploreEval = new_entry.first_explore_evals.entry(move_loc).or_default();
        if child_cache_weight >= eval.cache_weight {
          *eval = FirstExploreEval::new_full(child_win_loss_avg, child_score_mean_avg, child_cache_weight);
        }
      }
    }

    // Should we record this node's aggregate evals in the cache?
    // Or should we only leave it at recording initial first-play evals for exploring children?
    let mut should_record_evals: bool = true;
    if is_root_node {
      // On the root node, due to it handling passing differently than other nodes, we do NOT record it if passing
      // is near the highest utility move or is a decent fraction of the visits.
      let mut total_edge_visits: i64 = 0;
      let mut pass_edge_visits: i64 = 0;
      let mut max_self_utility: f64 = -1e50;
      let mut pass_self_utility: f64 = -1e50;
      for i in 0..children_capacity {
        let child_pointer: &SearchChildPointer = children.index(i);
        let child_idx: NodeIdx = match child_pointer.get_if_allocated() {
          None => break,
          Some(c) => c,
        };
        let child: Rc<SearchNode> = nodes.get(child_idx);
        let edge_visits: i64 = child_pointer.get_edge_visits();
        let child_utility: f64 = child.stats.utility_avg.get() /*acquire*/;
        let self_utility: f64 = if node.next_pla == P_WHITE { child_utility } else { -child_utility };
        total_edge_visits += edge_visits;
        max_self_utility = cpp_max(max_self_utility, self_utility);
        if child_pointer.get_move_loc_relaxed() == Board::PASS_LOC {
          pass_edge_visits += edge_visits;
          pass_self_utility = self_utility;
        }
      }
      if pass_edge_visits * 8 >= total_edge_visits || pass_self_utility + 0.05 >= max_self_utility {
        should_record_evals = false;
      }
    }

    if should_record_evals {
      new_entry.cache_weight = new_cache_weight;
      new_entry.avg_win_loss = node.stats.win_loss_value_avg.get() /*acquire*/ as f32;
      new_entry.avg_no_result = node.stats.no_result_value_avg.get() /*acquire*/ as f32;
      new_entry.avg_score_mean = node.stats.score_mean_avg.get() /*acquire*/ as f32;
      new_entry.avg_lead = node.stats.lead_avg.get() /*acquire*/ as f32;
    }

    // Replace the old entry with the new one
    // PORT NOTE: lock removed (single-threaded).
    self.entries.borrow_mut()[sub_map_idx as usize].insert(graph_hash, Some(Rc::new(new_entry)));
  }

  /// upstream: EvalCacheTable::clear (evalcache.cpp)
  pub fn clear(&self) {
    let mut entries = self.entries.borrow_mut();
    for i in 0..entries.len() {
      // PORT NOTE: lock removed (single-threaded).
      entries[i].clear();
    }
  }
}
