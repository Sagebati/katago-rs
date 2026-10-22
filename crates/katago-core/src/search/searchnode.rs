// Ported from KataGo cpp/search/searchnode.cpp, cpp/search/searchnode.h @ ba938676d7f42d70950b3a535af2466fb642008c
//
// PORT NOTE (PORTING.md §4, OWNERSHIP.tsv): nodes live in `port::node_arena::NodeArena` (a field of
// Search); every stored `SearchNode*` is a `NodeIdx` and `SearchNode` gains a first field `idx`.
// Atomics are `Cell`s (memory orders kept as comments), the `statsLock` spin lock is removed (its
// bodies kept), `std::atomic<std::shared_ptr<NNOutput>*>` is `RefCell<Option<Rc<NNOutput>>>`, and
// the children arrays are `OnceCell<Box<[SearchChildPointer]>>`, allocated once and never freed while
// the node lives: `collapseChildrenCapacity` (through `&self`) lowers the state and keeps the emptied
// array, which `tryExpandingChildrenCapacityAssumeFull` reuses (PORTING.md §4).

use std::cell::Cell;
use std::cell::OnceCell;
use std::cell::RefCell;
use std::rc::Rc;

use crate::core::hash::Hash128;
use crate::game::board::Board;
use crate::game::board::Loc;
use crate::game::board::Player;
use crate::neuralnet::nninputs::NNOutput;
use crate::neuralnet::nninputs::NNPos;
use crate::port::cpp;
use crate::port::node_arena::NodeIdx;
use crate::search::search::SearchThread;
use crate::search::subtreevaluebiastable::SubtreeValueBiasEntry;
use crate::testAssert;

/// upstream: SearchNodeState (searchnode.h)
pub type SearchNodeState = i32; // See SearchNode::STATE_*

/// upstream: EvalCacheEntry (evalcache.h)
/// TODO(port): deferred (evalcache.cpp): placeholder with no values (DECISIONS.md D7), so
/// `SearchNode::evalCacheEntry` is always None until the eval cache is ported.
pub enum EvalCacheEntry {}

/// upstream: NodeStatsAtomic (searchnode.h)
pub struct NodeStatsAtomic {
  pub visits: Cell<i64>,
  pub win_loss_value_avg: Cell<f64>,
  pub no_result_value_avg: Cell<f64>,
  pub score_mean_avg: Cell<f64>,
  pub score_mean_sq_avg: Cell<f64>,
  pub lead_avg: Cell<f64>,
  pub utility_avg: Cell<f64>,
  pub utility_sq_avg: Cell<f64>,
  pub weight_sum: Cell<f64>,
  pub weight_sq_sum: Cell<f64>,
}

impl NodeStatsAtomic {
  /// upstream: NodeStatsAtomic::NodeStatsAtomic (searchnode.cpp)
  pub fn new() -> NodeStatsAtomic {
    NodeStatsAtomic {
      visits: Cell::new(0),
      win_loss_value_avg: Cell::new(0.0),
      no_result_value_avg: Cell::new(0.0),
      score_mean_avg: Cell::new(0.0),
      score_mean_sq_avg: Cell::new(0.0),
      lead_avg: Cell::new(0.0),
      utility_avg: Cell::new(0.0),
      utility_sq_avg: Cell::new(0.0),
      weight_sum: Cell::new(0.0),
      weight_sq_sum: Cell::new(0.0),
    }
  }

  /// upstream: NodeStatsAtomic::NodeStatsAtomic(const NodeStatsAtomic&) (searchnode.cpp)
  pub fn new_copy(other: &NodeStatsAtomic) -> NodeStatsAtomic {
    NodeStatsAtomic {
      visits: Cell::new(other.visits.get() /*acquire*/),
      win_loss_value_avg: Cell::new(other.win_loss_value_avg.get() /*acquire*/),
      no_result_value_avg: Cell::new(other.no_result_value_avg.get() /*acquire*/),
      score_mean_avg: Cell::new(other.score_mean_avg.get() /*acquire*/),
      score_mean_sq_avg: Cell::new(other.score_mean_sq_avg.get() /*acquire*/),
      lead_avg: Cell::new(other.lead_avg.get() /*acquire*/),
      utility_avg: Cell::new(other.utility_avg.get() /*acquire*/),
      utility_sq_avg: Cell::new(other.utility_sq_avg.get() /*acquire*/),
      weight_sum: Cell::new(other.weight_sum.get() /*acquire*/),
      weight_sq_sum: Cell::new(other.weight_sq_sum.get() /*acquire*/),
    }
  }

  /// upstream: NodeStatsAtomic::getChildWeight (searchnode.h)
  pub fn get_child_weight(&self, edge_visits: i64) -> f64 {
    NodeStats::child_weight(edge_visits, self.visits.get() /*acquire*/, self.weight_sum.get() /*acquire*/)
  }
  /// upstream: NodeStatsAtomic::getChildWeight (searchnode.h)
  pub fn get_child_weight_child_visits(&self, edge_visits: i64, child_visits: i64) -> f64 {
    NodeStats::child_weight(edge_visits, child_visits, self.weight_sum.get() /*acquire*/)
  }
  /// upstream: NodeStatsAtomic::getChildWeightSq (searchnode.h)
  pub fn get_child_weight_sq(&self, edge_visits: i64) -> f64 {
    NodeStats::child_weight_sq(edge_visits, self.visits.get() /*acquire*/, self.weight_sq_sum.get() /*acquire*/)
  }
  /// upstream: NodeStatsAtomic::getChildWeightSq (searchnode.h)
  pub fn get_child_weight_sq_child_visits(&self, edge_visits: i64, child_visits: i64) -> f64 {
    NodeStats::child_weight_sq(edge_visits, child_visits, self.weight_sq_sum.get() /*acquire*/)
  }
}

/// upstream: NodeStats (searchnode.h)
#[derive(Clone, Copy, Debug, Default)]
pub struct NodeStats {
  pub visits: i64,
  pub win_loss_value_avg: f64,
  pub no_result_value_avg: f64,
  pub score_mean_avg: f64,
  pub score_mean_sq_avg: f64,
  pub lead_avg: f64,
  pub utility_avg: f64,
  pub utility_sq_avg: f64,
  pub weight_sum: f64,
  pub weight_sq_sum: f64,
}

impl NodeStats {
  /// upstream: NodeStats::NodeStats (searchnode.cpp)
  pub fn new() -> NodeStats {
    NodeStats {
      visits: 0,
      win_loss_value_avg: 0.0,
      no_result_value_avg: 0.0,
      score_mean_avg: 0.0,
      score_mean_sq_avg: 0.0,
      lead_avg: 0.0,
      utility_avg: 0.0,
      utility_sq_avg: 0.0,
      weight_sum: 0.0,
      weight_sq_sum: 0.0,
    }
  }

  /// upstream: NodeStats::NodeStats(const NodeStatsAtomic&) (searchnode.cpp)
  pub fn new_other(other: &NodeStatsAtomic) -> NodeStats {
    NodeStats {
      visits: other.visits.get() /*acquire*/,
      win_loss_value_avg: other.win_loss_value_avg.get() /*acquire*/,
      no_result_value_avg: other.no_result_value_avg.get() /*acquire*/,
      score_mean_avg: other.score_mean_avg.get() /*acquire*/,
      score_mean_sq_avg: other.score_mean_sq_avg.get() /*acquire*/,
      lead_avg: other.lead_avg.get() /*acquire*/,
      utility_avg: other.utility_avg.get() /*acquire*/,
      utility_sq_avg: other.utility_sq_avg.get() /*acquire*/,
      weight_sum: other.weight_sum.get() /*acquire*/,
      weight_sq_sum: other.weight_sq_sum.get() /*acquire*/,
    }
  }

  /// upstream: NodeStats::childWeight (searchnode.h)
  #[inline]
  pub fn child_weight(edge_visits: i64, child_visits: i64, raw_child_weight: f64) -> f64 {
    raw_child_weight * (edge_visits as f64 / std::cmp::max(child_visits, 1i64) as f64)
  }
  /// upstream: NodeStats::childWeightSq (searchnode.h)
  #[inline]
  pub fn child_weight_sq(edge_visits: i64, child_visits: i64, raw_child_weight_sq: f64) -> f64 {
    raw_child_weight_sq * (edge_visits as f64 / std::cmp::max(child_visits, 1i64) as f64)
  }
  /// upstream: NodeStats::getChildWeight (searchnode.h)
  pub fn get_child_weight(&self, edge_visits: i64) -> f64 {
    NodeStats::child_weight(edge_visits, self.visits, self.weight_sum)
  }
}

/// upstream: MoreNodeStats (searchnode.h)
#[derive(Clone, Copy, Debug)]
pub struct MoreNodeStats {
  pub stats: NodeStats,
  pub self_utility: f64,
  pub weight_adjusted: f64,
  pub prev_move_loc: Loc,
}

impl MoreNodeStats {
  /// upstream: MoreNodeStats::MoreNodeStats (searchnode.cpp)
  pub fn new() -> MoreNodeStats {
    MoreNodeStats { stats: NodeStats::new(), self_utility: 0.0, weight_adjusted: 0.0, prev_move_loc: Board::NULL_LOC }
  }
}

//----------------------------------------------------------------------------------------

/// upstream: SearchChildPointer (searchnode.h)
pub struct SearchChildPointer {
  data: Cell<Option<NodeIdx>>,
  edge_visits: Cell<i64>,
  move_loc: Cell<Loc>, // Generally this will be always guarded under release semantics of data or of the array itself.
}

impl SearchChildPointer {
  /// upstream: SearchChildPointer::SearchChildPointer (searchnode.cpp)
  pub fn new() -> SearchChildPointer {
    SearchChildPointer { data: Cell::new(None), edge_visits: Cell::new(0), move_loc: Cell::new(Board::NULL_LOC) }
  }

  /// upstream: SearchChildPointer::storeAll (searchnode.cpp)
  pub fn store_all(&self, other: &SearchChildPointer) {
    let d: Option<NodeIdx> = other.data.get() /*acquire*/;
    let e: i64 = other.edge_visits.get() /*acquire*/;
    let m: Loc = other.move_loc.get() /*acquire*/;
    self.move_loc.set(m) /*release*/;
    self.edge_visits.set(e) /*release*/;
    self.data.set(d) /*release*/;
  }

  /// upstream: SearchChildPointer::getIfAllocated (searchnode.h)
  #[inline]
  pub fn get_if_allocated(&self) -> Option<NodeIdx> {
    self.data.get() /*acquire*/
  }
  /// upstream: SearchChildPointer::getIfAllocatedRelaxed (searchnode.h)
  #[inline]
  pub fn get_if_allocated_relaxed(&self) -> Option<NodeIdx> {
    self.data.get() /*relaxed*/
  }
  /// upstream: SearchChildPointer::store (searchnode.h)
  /// PORT NOTE: takes `Option<NodeIdx>` because upstream also stores NULL (Search::beginSearch).
  #[inline]
  pub fn store(&self, node: Option<NodeIdx>) {
    self.data.set(node) /*release*/;
  }
  /// upstream: SearchChildPointer::storeRelaxed (searchnode.h)
  #[inline]
  pub fn store_relaxed(&self, node: Option<NodeIdx>) {
    self.data.set(node) /*relaxed*/;
  }
  /// upstream: SearchChildPointer::storeIfNull (searchnode.cpp)
  pub fn store_if_null(&self, node: NodeIdx) -> bool {
    let mut expected: Option<NodeIdx> = None;
    cpp::cell_cas(&self.data, &mut expected, Some(node)) /*acq_rel*/
  }

  /// upstream: SearchChildPointer::getEdgeVisits (searchnode.h)
  #[inline]
  pub fn get_edge_visits(&self) -> i64 {
    self.edge_visits.get() /*acquire*/
  }
  /// upstream: SearchChildPointer::getEdgeVisitsRelaxed (searchnode.h)
  #[inline]
  pub fn get_edge_visits_relaxed(&self) -> i64 {
    self.edge_visits.get() /*relaxed*/
  }
  /// upstream: SearchChildPointer::setEdgeVisits (searchnode.h)
  #[inline]
  pub fn set_edge_visits(&self, x: i64) {
    self.edge_visits.set(x) /*release*/;
  }
  /// upstream: SearchChildPointer::setEdgeVisitsRelaxed (searchnode.h)
  #[inline]
  pub fn set_edge_visits_relaxed(&self, x: i64) {
    self.edge_visits.set(x) /*relaxed*/;
  }
  /// upstream: SearchChildPointer::addEdgeVisits (searchnode.h)
  #[inline]
  pub fn add_edge_visits(&self, delta: i64) {
    cpp::cell_fetch_add(&self.edge_visits, delta) /*acq_rel*/;
  }
  /// upstream: SearchChildPointer::compexweakEdgeVisits (searchnode.cpp)
  /// PORT NOTE: compare_exchange_weak never fails spuriously here (port::cpp::cell_cas).
  pub fn compexweak_edge_visits(&self, expected: &mut i64, desired: i64) -> bool {
    cpp::cell_cas(&self.edge_visits, expected, desired) /*acq_rel*/
  }

  /// upstream: SearchChildPointer::getMoveLoc (searchnode.h)
  #[inline]
  pub fn get_move_loc(&self) -> Loc {
    self.move_loc.get() /*acquire*/
  }
  /// upstream: SearchChildPointer::getMoveLocRelaxed (searchnode.h)
  #[inline]
  pub fn get_move_loc_relaxed(&self) -> Loc {
    self.move_loc.get() /*relaxed*/
  }
  /// upstream: SearchChildPointer::setMoveLoc (searchnode.h)
  #[inline]
  pub fn set_move_loc(&self, loc: Loc) {
    self.move_loc.set(loc) /*release*/;
  }
  /// upstream: SearchChildPointer::setMoveLocRelaxed (searchnode.h)
  #[inline]
  pub fn set_move_loc_relaxed(&self, loc: Loc) {
    self.move_loc.set(loc) /*relaxed*/;
  }
}

/// upstream: namespace SearchChildrenSizes (searchnode.h)
pub struct SearchChildrenSizes;

impl SearchChildrenSizes {
  pub const SIZE0TOTAL: i32 = 8;
  pub const SIZE1TOTAL: i32 = 64;
  pub const SIZE2TOTAL: i32 = NNPos::MAX_NN_POLICY_SIZE;
  pub const SIZE0OVERFLOW: i32 = SearchChildrenSizes::SIZE0TOTAL;
  pub const SIZE1OVERFLOW: i32 = SearchChildrenSizes::SIZE1TOTAL - SearchChildrenSizes::SIZE0TOTAL;
  pub const SIZE2OVERFLOW: i32 = SearchChildrenSizes::SIZE2TOTAL - SearchChildrenSizes::SIZE1TOTAL;
}

/// upstream: SearchNodeChildrenReference (searchnode.h)
// Abstracts children{0,1,2} in SearchNode as a single concatenated array
#[derive(Clone, Copy)]
pub struct SearchNodeChildrenReference<'a> {
  pub snapshotted_state: SearchNodeState,
  pub node: &'a SearchNode,
}

/// upstream: ConstSearchNodeChildrenReference (searchnode.h)
/// PORT NOTE (OWNERSHIP.tsv): the same type as SearchNodeChildrenReference (Rust references carry
/// constness at the use site); its never-read `capacity` field is dropped.
pub type ConstSearchNodeChildrenReference<'a> = SearchNodeChildrenReference<'a>;

impl<'a> SearchNodeChildrenReference<'a> {
  /// upstream: SearchNodeChildrenReference::SearchNodeChildrenReference (searchnode.h)
  #[inline]
  pub fn new(snapshotted_state: SearchNodeState, node: &'a SearchNode) -> SearchNodeChildrenReference<'a> {
    SearchNodeChildrenReference { snapshotted_state, node }
  }

  /// upstream: SearchNodeChildrenReference::operator[] (searchnode.h)
  /// PORT NOTE: upstream indexes the raw arrays without checking that they exist; reading an
  /// array the snapshotted state does not cover would be a null dereference, here a panic.
  #[inline]
  pub fn index(&self, i: i32) -> &'a SearchChildPointer {
    let node: &'a SearchNode = self.node;
    if i < SearchChildrenSizes::SIZE0TOTAL {
      return &node.children0.get().expect("children0 allocated")[i as usize];
    }
    if i < SearchChildrenSizes::SIZE1TOTAL {
      return &node.children1.get().expect("children1 allocated")[(i - SearchChildrenSizes::SIZE0TOTAL) as usize];
    }
    &node.children2.get().expect("children2 allocated")[(i - SearchChildrenSizes::SIZE1TOTAL) as usize]
  }

  /// upstream: SearchNodeChildrenReference::getCapacity (searchnode.cpp)
  pub fn get_capacity(&self) -> i32 {
    get_children_capacity(self.snapshotted_state)
  }

  /// upstream: ConstSearchNodeChildrenReference::iterateAndCountChildren (searchnode.cpp)
  /// PORT NOTE: the non-const version forwards to the const one; both are this fn.
  pub fn iterate_and_count_children(&self) -> i32 {
    let arr: &[SearchChildPointer];
    let arr_capacity: i32;
    let offset: i32;

    if self.snapshotted_state < SearchNode::STATE_EXPANDED0 {
      return 0;
    } else if self.snapshotted_state < SearchNode::STATE_EXPANDED1 {
      arr = self.node.children0.get().expect("children0 allocated");
      arr_capacity = SearchChildrenSizes::SIZE0OVERFLOW;
      offset = 0;
    } else if self.snapshotted_state < SearchNode::STATE_EXPANDED2 {
      arr = self.node.children1.get().expect("children1 allocated");
      arr_capacity = SearchChildrenSizes::SIZE1OVERFLOW;
      offset = SearchChildrenSizes::SIZE0TOTAL;
    } else {
      arr = self.node.children2.get().expect("children2 allocated");
      arr_capacity = SearchChildrenSizes::SIZE2OVERFLOW;
      offset = SearchChildrenSizes::SIZE1TOTAL;
    }

    for i in 0..arr_capacity {
      if arr[i as usize].get_if_allocated().is_none() {
        return offset + i;
      }
    }
    offset + arr_capacity
  }
}

/// upstream: getChildrenCapacity (searchnode.cpp) static
fn get_children_capacity(state_value: SearchNodeState) -> i32 {
  if state_value < SearchNode::STATE_EXPANDED0 {
    0
  } else if state_value < SearchNode::STATE_EXPANDED1 {
    SearchChildrenSizes::SIZE0TOTAL
  } else if state_value < SearchNode::STATE_EXPANDED2 {
    SearchChildrenSizes::SIZE1TOTAL
  } else {
    SearchChildrenSizes::SIZE2TOTAL
  }
}

fn new_child_array(n: i32) -> Box<[SearchChildPointer]> {
  let mut v: Vec<SearchChildPointer> = Vec::with_capacity(n as usize);
  for _ in 0..n {
    v.push(SearchChildPointer::new());
  }
  v.into_boxed_slice()
}

/// `childrenN = new SearchChildPointer[n]` in tryExpandingChildrenCapacityAssumeFull.
/// PORT NOTE (PORTING.md §4): collapseChildrenCapacity cannot free an array through `&self` (and the
/// node may be shared), so it leaves the emptied array in its OnceCell; growing again reuses it, every
/// slot reset to a fresh SearchChildPointer, which is what the new array would hold.
fn install_child_array(cell: &OnceCell<Box<[SearchChildPointer]>>, n: i32) {
  match cell.get() {
    None => {
      let r = cell.set(new_child_array(n));
      debug_assert!(r.is_ok());
    }
    Some(kept) => {
      debug_assert!(kept.len() == n as usize);
      let fresh: SearchChildPointer = SearchChildPointer::new();
      for slot in kept.iter() {
        testAssert!(slot.get_if_allocated_relaxed().is_none());
        slot.store_all(&fresh);
      }
    }
  }
}

/// upstream: SearchNode (searchnode.h)
pub struct SearchNode {
  /// PORT NOTE (OWNERSHIP.tsv): new first field, the node's arena index (= its identity, what
  /// upstream compares pointers for). Set by NodeArena::alloc.
  pub idx: NodeIdx,

  //Locks------------------------------------------------------------------------------
  // PORT NOTE: `mutable std::atomic_flag statsLock` removed (single-threaded); the bodies it guarded are kept.

  //Constant during search--------------------------------------------------------------
  pub next_pla: Player,
  pub force_non_terminal: bool,
  pub pattern_bonus_hash: Cell<Hash128>,
  pub mutex_idx: u32, // For lookup into mutex pool

  //Mutable---------------------------------------------------------------------------
  //During search, only ever transitions forward.
  pub state: Cell<SearchNodeState>,

  //During search, will only ever transition from NULL -> non-NULL.
  //Guaranteed to be non-NULL once state >= STATE_EXPANDED0.
  //After this is non-NULL, might rarely change mid-search, but it is guaranteed that old values remain
  //valid to access for the duration of the search and will not be deallocated.
  pub nn_output: RefCell<Option<Rc<NNOutput>>>,
  pub human_output: RefCell<Option<Rc<NNOutput>>>,

  //Used to coordinate various multithreaded updates.
  //During search, for updating nnOutput when it needs recomputation at the root if it wasn't updated yet.
  //During various other events - for coordinating recursive updates of the tree or subtree value bias cleanup
  pub node_age: Cell<u32>,

  //During search, each will only ever transition from NULL -> non-NULL.
  //We get progressive resizing of children array simply overflowing on to successive later arrays.
  //Mutex pool guards insertion of children at a node. Reading of children is always fine.
  pub children0: OnceCell<Box<[SearchChildPointer]>>, //Guaranteed to be non-NULL once state >= STATE_EXPANDED0
  pub children1: OnceCell<Box<[SearchChildPointer]>>, //Guaranteed to be non-NULL once state >= STATE_EXPANDED1
  pub children2: OnceCell<Box<[SearchChildPointer]>>, //Guaranteed to be non-NULL once state >= STATE_EXPANDED2

  //Lightweight mutable---------------------------------------------------------------
  //Protected under statsLock for writing
  pub stats: NodeStatsAtomic,
  pub virtual_losses: Cell<i32>,

  //Protected under the entryLock in subtreeValueBiasTableEntry
  //Used only if subtreeValueBiasTableEntry is not nullptr.
  //During search, subtreeValueBiasTableEntry itself is set upon creation of the node and remains constant
  //thereafter, making it safe to access without synchronization.
  pub last_subtree_value_bias_delta_sum: Cell<f64>,
  pub last_subtree_value_bias_weight: Cell<f64>,
  pub subtree_value_bias_table_entry: RefCell<Option<Rc<SubtreeValueBiasEntry>>>,

  //Only valid if useGraphSearch is true.
  //Graph hash of this node.
  //Note that this is NOT a unique key for evaluations due to nodes varying by forceNonTerminal.
  pub graph_hash: Hash128,
  // TODO(port): deferred (evalcache.cpp): always None (EvalCacheEntry has no values, D7).
  pub eval_cache_entry: RefCell<Option<Rc<EvalCacheEntry>>>,

  pub dirty_counter: Cell<i32>,
}

impl SearchNode {
  /// upstream: SearchNode::STATE_* (searchnode.h)
  pub const STATE_UNEVALUATED: SearchNodeState = 0;
  pub const STATE_EVALUATING: SearchNodeState = 1;
  pub const STATE_EXPANDED0: SearchNodeState = 2;
  pub const STATE_GROWING1: SearchNodeState = 3;
  pub const STATE_EXPANDED1: SearchNodeState = 4;
  pub const STATE_GROWING2: SearchNodeState = 5;
  pub const STATE_EXPANDED2: SearchNodeState = 6;

  /// upstream: SearchNode::SearchNode(Player, bool, uint32_t, Hash128) (searchnode.cpp)
  //Makes a search node resulting from prevPla playing prevLoc
  /// PORT NOTE: takes the arena index first (OWNERSHIP.tsv).
  pub fn new(idx: NodeIdx, pla: Player, fnt: bool, m_idx: u32, gh: Hash128) -> SearchNode {
    SearchNode {
      idx,
      next_pla: pla,
      force_non_terminal: fnt,
      pattern_bonus_hash: Cell::new(Hash128::default()),
      mutex_idx: m_idx,
      state: Cell::new(SearchNode::STATE_UNEVALUATED),
      nn_output: RefCell::new(None),
      human_output: RefCell::new(None),
      node_age: Cell::new(0),
      children0: OnceCell::new(),
      children1: OnceCell::new(),
      children2: OnceCell::new(),
      stats: NodeStatsAtomic::new(),
      virtual_losses: Cell::new(0),
      last_subtree_value_bias_delta_sum: Cell::new(0.0),
      last_subtree_value_bias_weight: Cell::new(0.0),
      subtree_value_bias_table_entry: RefCell::new(None),
      graph_hash: gh,
      eval_cache_entry: RefCell::new(None),
      dirty_counter: Cell::new(0),
    }
  }

  /// upstream: SearchNode::SearchNode(const SearchNode&, bool, bool) (searchnode.cpp)
  /// PORT NOTE: takes the arena index first (OWNERSHIP.tsv). The copied children arrays hold the
  /// same child indices (the children are ARENA nodes, shared, never deep-copied).
  pub fn new_copy(idx: NodeIdx, other: &SearchNode, fnt: bool, copy_subtree_value_bias: bool) -> SearchNode {
    let s = SearchNode {
      idx,
      next_pla: other.next_pla,
      force_non_terminal: fnt,
      pattern_bonus_hash: Cell::new(other.pattern_bonus_hash.get()),
      mutex_idx: other.mutex_idx,
      state: Cell::new(other.state.get() /*acquire*/),
      nn_output: RefCell::new(None),
      human_output: RefCell::new(None),
      node_age: Cell::new(other.node_age.get() /*acquire*/),
      children0: OnceCell::new(),
      children1: OnceCell::new(),
      children2: OnceCell::new(),
      stats: NodeStatsAtomic::new_copy(&other.stats),
      virtual_losses: Cell::new(other.virtual_losses.get() /*acquire*/),
      last_subtree_value_bias_delta_sum: Cell::new(0.0),
      last_subtree_value_bias_weight: Cell::new(0.0),
      subtree_value_bias_table_entry: RefCell::new(None),
      graph_hash: other.graph_hash,
      eval_cache_entry: RefCell::new(other.eval_cache_entry.borrow().clone()),
      dirty_counter: Cell::new(other.dirty_counter.get() /*acquire*/),
    };
    {
      let other_val: Option<Rc<NNOutput>> = other.nn_output.borrow().clone() /*acquire*/;
      if let Some(other_val) = other_val {
        *s.nn_output.borrow_mut() = Some(other_val) /*release*/;
      }
    }
    {
      let other_val: Option<Rc<NNOutput>> = other.human_output.borrow().clone() /*acquire*/;
      if let Some(other_val) = other_val {
        *s.human_output.borrow_mut() = Some(other_val) /*release*/;
      }
    }
    if let Some(other_children0) = other.children0.get() {
      let children0 = new_child_array(SearchChildrenSizes::SIZE0OVERFLOW);
      for i in 0..SearchChildrenSizes::SIZE0OVERFLOW {
        children0[i as usize].store_all(&other_children0[i as usize]);
      }
      let _ = s.children0.set(children0);
    }
    if let Some(other_children1) = other.children1.get() {
      let children1 = new_child_array(SearchChildrenSizes::SIZE1OVERFLOW);
      for i in 0..SearchChildrenSizes::SIZE1OVERFLOW {
        children1[i as usize].store_all(&other_children1[i as usize]);
      }
      let _ = s.children1.set(children1);
    }
    if let Some(other_children2) = other.children2.get() {
      let children2 = new_child_array(SearchChildrenSizes::SIZE2OVERFLOW);
      for i in 0..SearchChildrenSizes::SIZE2OVERFLOW {
        children2[i as usize].store_all(&other_children2[i as usize]);
      }
      let _ = s.children2.set(children2);
    }
    if copy_subtree_value_bias {
      //Currently NOT implemented. If we ever want this, think very carefully about copying subtree value bias since
      //if we later delete this node we risk double-counting removal of the subtree value bias!
      unreachable!();
      //lastSubtreeValueBiasDeltaSum = other.lastSubtreeValueBiasDeltaSum;
      //lastSubtreeValueBiasWeight = other.lastSubtreeValueBiasWeight;
      //subtreeValueBiasTableEntry = other.subtreeValueBiasTableEntry;
    }
    s
  }

  /// upstream: SearchNode::getChildren (searchnode.cpp)
  //The array returned by these is guaranteed not to be deallocated during the lifetime of a search or even
  //any time up until a new operation is peformed (such as starting a new search, or making a move, or setting params).
  pub fn get_children(&self) -> SearchNodeChildrenReference<'_> {
    SearchNodeChildrenReference::new(self.state.get() /*acquire*/, self)
  }
  /// upstream: SearchNode::getChildren(SearchNodeState) (searchnode.cpp)
  pub fn get_children_state(&self, state_value: SearchNodeState) -> SearchNodeChildrenReference<'_> {
    SearchNodeChildrenReference::new(state_value, self)
  }

  /// upstream: SearchNode::maybeExpandChildrenCapacityForNewChild (searchnode.cpp)
  //Precondition: Assumes that we have actually checked the children array that stateValue suggests that
  //we should use, and that every slot in it is full up to numChildrenFullPlusOne-1, and
  //that we have found a new legal child to add.
  //Postcondition:
  //Returns true: node state, stateValue, children arrays are all updated if needed so that they are large enough.
  //Returns false: failure since another thread is handling it.
  //Thread-safe.
  pub fn maybe_expand_children_capacity_for_new_child(&self, state_value: &mut SearchNodeState, num_children_full_plus_one: i32) -> bool {
    let capacity: i32 = get_children_capacity(*state_value);
    if capacity < num_children_full_plus_one {
      debug_assert!(capacity == num_children_full_plus_one - 1);
      return self.try_expanding_children_capacity_assume_full(state_value);
    }
    true
  }

  /// upstream: SearchNode::initializeChildren (searchnode.cpp)
  pub fn initialize_children(&self) {
    debug_assert!(self.children0.get().is_none());
    let r = self.children0.set(new_child_array(SearchChildrenSizes::SIZE0OVERFLOW));
    debug_assert!(r.is_ok());
  }

  /// upstream: SearchNode::tryExpandingChildrenCapacityAssumeFull (searchnode.cpp)
  //Precondition: Assumes that we have actually checked the childen array that stateValue suggests that
  //we should use, and that every slot in it is full.
  fn try_expanding_children_capacity_assume_full(&self, state_value: &mut SearchNodeState) -> bool {
    if *state_value < SearchNode::STATE_EXPANDED1 {
      if *state_value == SearchNode::STATE_GROWING1 {
        return false;
      }
      debug_assert!(*state_value == SearchNode::STATE_EXPANDED0);
      let suc: bool = cpp::cell_cas(&self.state, state_value, SearchNode::STATE_GROWING1) /*acq_rel*/;
      if !suc {
        return false;
      }
      *state_value = SearchNode::STATE_GROWING1;

      install_child_array(&self.children1, SearchChildrenSizes::SIZE1OVERFLOW);
      self.state.set(SearchNode::STATE_EXPANDED1) /*release*/;
      *state_value = SearchNode::STATE_EXPANDED1;
    } else if *state_value < SearchNode::STATE_EXPANDED2 {
      if *state_value == SearchNode::STATE_GROWING2 {
        return false;
      }
      debug_assert!(*state_value == SearchNode::STATE_EXPANDED1);
      let suc: bool = cpp::cell_cas(&self.state, state_value, SearchNode::STATE_GROWING2) /*acq_rel*/;
      if !suc {
        return false;
      }
      *state_value = SearchNode::STATE_GROWING2;

      install_child_array(&self.children2, SearchChildrenSizes::SIZE2OVERFLOW);
      self.state.set(SearchNode::STATE_EXPANDED2) /*release*/;
      *state_value = SearchNode::STATE_EXPANDED2;
    } else {
      unreachable!();
    }
    true
  }

  /// upstream: SearchNode::collapseChildrenCapacity (searchnode.cpp)
  //If we pruned some of the child nodes, collapse down the arrays and the node state to fit.
  //This preserves the invariant that the level of expansion is only the minimum needed to hold those child nodes.
  //Not thread-safe.
  /// PORT NOTE (PORTING.md §4): `&self`; `delete[] childrenN; childrenN = NULL;` leaves the emptied array
  /// in its OnceCell (see install_child_array): only the state says which arrays are in use.
  pub fn collapse_children_capacity(&self, num_good_children: i32) {
    let mut state_value: i32 = self.state.get() /*acquire*/;
    if num_good_children <= SearchChildrenSizes::SIZE1TOTAL && state_value > SearchNode::STATE_EXPANDED1 {
      debug_assert!(state_value == SearchNode::STATE_EXPANDED2);
      debug_assert!(self.children2.get().is_some());
      {
        let children2 = self.children2.get().expect("children2 allocated");
        for i in 0..SearchChildrenSizes::SIZE2OVERFLOW {
          testAssert!(children2[i as usize].get_if_allocated_relaxed().is_none());
        }
      }
      // delete[] children2; children2 = NULL; (PORT NOTE: kept, see above)
      state_value = SearchNode::STATE_EXPANDED1;
      self.state.set(state_value) /*release*/;
    }
    if num_good_children <= SearchChildrenSizes::SIZE0TOTAL && state_value > SearchNode::STATE_EXPANDED0 {
      debug_assert!(state_value == SearchNode::STATE_EXPANDED1);
      debug_assert!(self.children1.get().is_some());
      {
        let children1 = self.children1.get().expect("children1 allocated");
        for i in 0..SearchChildrenSizes::SIZE1OVERFLOW {
          testAssert!(children1[i as usize].get_if_allocated_relaxed().is_none());
        }
      }
      // delete[] children1; children1 = NULL; (PORT NOTE: kept, see above)
      state_value = SearchNode::STATE_EXPANDED0;
      self.state.set(state_value) /*release*/;
    }
  }

  /// upstream: SearchNode::getNNOutput (searchnode.cpp)
  //The NNOutput returned by these is guaranteed not to be deallocated during the lifetime of a search or even
  //any time up until a new operation is peformed (such as starting a new search, or making a move, or setting params).
  /// PORT NOTE: returns a clone of the Rc (OWNERSHIP.tsv), which keeps it alive as long as needed.
  pub fn get_nn_output(&self) -> Option<Rc<NNOutput>> {
    self.nn_output.borrow().clone() /*acquire*/
  }

  /// upstream: SearchNode::getHumanOutput (searchnode.cpp)
  pub fn get_human_output(&self) -> Option<Rc<NNOutput>> {
    self.human_output.borrow().clone() /*acquire*/
  }

  /// upstream: SearchNode::storeNNOutput (searchnode.cpp)
  //Always replaces the current nnoutput, and stores the existing one in the thread for later deletion.
  //Returns true if there was NOT already an nnOutput
  pub fn store_nn_output(&self, new_nn_output: Rc<NNOutput>, thread: &mut SearchThread) -> bool {
    let to_clean_up: Option<Rc<NNOutput>> = self.nn_output.replace(Some(new_nn_output)) /*acq_rel*/;
    if let Some(to_clean_up) = to_clean_up {
      thread.old_nn_outputs_to_clean_up.push(to_clean_up);
      return false;
    }
    true
  }

  /// upstream: SearchNode::storeNNOutputIfNull (searchnode.cpp)
  //Only stores if there isn't an nnOutput already. Returns true if it was stored.
  /// PORT NOTE: on failure the passed Rc is dropped here (upstream: the caller deletes it).
  pub fn store_nn_output_if_null(&self, new_nn_output: Rc<NNOutput>) -> bool {
    let mut slot = self.nn_output.borrow_mut();
    if slot.is_none() {
      *slot = Some(new_nn_output) /*acq_rel*/;
      return true;
    }
    false
  }

  /// upstream: SearchNode::storeHumanOutput (searchnode.cpp)
  pub fn store_human_output(&self, new_human_output: Rc<NNOutput>, thread: &mut SearchThread) -> bool {
    let to_clean_up: Option<Rc<NNOutput>> = self.human_output.replace(Some(new_human_output)) /*acq_rel*/;
    if let Some(to_clean_up) = to_clean_up {
      thread.old_nn_outputs_to_clean_up.push(to_clean_up);
      return false;
    }
    true
  }

  /// upstream: SearchNode::storeHumanOutputIfNull (searchnode.cpp)
  /// PORT NOTE: on failure the passed Rc is dropped here (upstream: the caller deletes it).
  pub fn store_human_output_if_null(&self, new_human_output: Rc<NNOutput>) -> bool {
    let mut slot = self.human_output.borrow_mut();
    if slot.is_none() {
      *slot = Some(new_human_output) /*acq_rel*/;
      return true;
    }
    false
  }
}

// SearchNode::~SearchNode only frees the children arrays and the nnOutput holders ("Do NOT recursively
// delete children"): Drop of the fields does exactly that (the children arrays hold indices).
