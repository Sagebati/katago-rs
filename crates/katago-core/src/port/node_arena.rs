//! Search node storage (docs/PORTING.md §4, DECISIONS.md D11). Not a port of any upstream file.
//!
//! Upstream allocates every `SearchNode` with `new` and links them with raw `SearchNode*`
//! pointers (children arrays, the graph-search node table, `rootNode`, `graphPath`,
//! `AnalysisData::node`). The port stores the nodes here, as `Rc<T>` slots addressed by a
//! [`NodeIdx`]; a stored pointer becomes a `NodeIdx` and pointer identity becomes index
//! equality. Every method takes `&self`, because search code runs through `&Search` while the
//! virtual threads interleave (PORTING.md §11). Nodes are never mutated through the arena: every
//! field a search writes is a `Cell`/`RefCell`/`OnceCell` (OWNERSHIP.tsv), so a host may hold a
//! node's `Rc` (`Search::get_root_node`) across any Search operation.
//!
//! `free(idx)` is `delete node`: the slot is emptied (anyone still holding an `Rc` keeps the node
//! alive, as upstream's old-pointer guarantees within a search) and the slot may be reused by a
//! later `alloc`, like freed memory. A stale `NodeIdx` (one whose node was freed) is a dangling
//! pointer: `get` panics on an empty slot, and in debug builds every `NodeIdx` also carries the
//! slot's generation, so `get`/`free` of a stale index panic even after the slot was reused
//! (release builds would silently reach the new node there, like a reused address upstream).
//! Slot numbers are never observable in outputs (nothing iterates nodes in index order).

use std::cell::RefCell;
use std::rc::Rc;

/// A `SearchNode*` stored in a field or collection: a slot of the search's [`NodeArena`] (plus,
/// in debug builds, the slot's generation when the node was allocated, to catch stale indices).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeIdx {
  slot: u32,
  #[cfg(debug_assertions)]
  generation: u32,
}

impl NodeIdx {
  /// The slot number (debugging and tests; a later node may reuse it after a free).
  pub fn slot(self) -> u32 {
    self.slot
  }
}

struct Slot<T> {
  node: Option<Rc<T>>,
  /// Incremented by every `free` of the slot (debug builds only).
  #[cfg(debug_assertions)]
  generation: u32,
}

/// The owner of every node of one search (PORTING.md §4).
pub struct NodeArena<T> {
  slots: RefCell<Vec<Slot<T>>>,
  free_list: RefCell<Vec<u32>>,
}

impl<T> Default for NodeArena<T> {
  fn default() -> Self {
    NodeArena::new()
  }
}

impl<T> NodeArena<T> {
  pub fn new() -> NodeArena<T> {
    NodeArena { slots: RefCell::new(Vec::new()), free_list: RefCell::new(Vec::new()) }
  }

  #[cfg(debug_assertions)]
  fn make_idx(slot: u32, s: &Slot<T>) -> NodeIdx {
    NodeIdx { slot, generation: s.generation }
  }
  #[cfg(not(debug_assertions))]
  fn make_idx(slot: u32, _s: &Slot<T>) -> NodeIdx {
    NodeIdx { slot }
  }

  /// Debug builds: panics if `idx` was allocated before the slot's last free (a dangling pointer).
  #[cfg(debug_assertions)]
  fn check_generation(idx: NodeIdx, s: &Slot<T>, what: &str) {
    if s.generation != idx.generation {
      panic!("node arena: {} through a stale node index {:?} (its node was freed; the slot is at generation {})", what, idx, s.generation);
    }
  }
  #[cfg(not(debug_assertions))]
  fn check_generation(_idx: NodeIdx, _s: &Slot<T>, _what: &str) {}

  /// `new T(...)`: `f` receives the index the node will live at (SearchNode stores it as `idx`).
  pub fn alloc<F: FnOnce(NodeIdx) -> T>(&self, f: F) -> NodeIdx {
    let reused: Option<u32> = self.free_list.borrow_mut().pop();
    match reused {
      Some(i) => {
        let idx: NodeIdx = {
          let slots = self.slots.borrow();
          debug_assert!(slots[i as usize].node.is_none());
          Self::make_idx(i, &slots[i as usize])
        };
        let node: Rc<T> = Rc::new(f(idx));
        self.slots.borrow_mut()[i as usize].node = Some(node);
        idx
      }
      None => {
        let i: u32 = {
          let slots = self.slots.borrow();
          u32::try_from(slots.len()).expect("node arena: more than 2^32 nodes")
        };
        let slot: Slot<T> = Slot {
          node: None,
          #[cfg(debug_assertions)]
          generation: 0,
        };
        let idx: NodeIdx = Self::make_idx(i, &slot);
        let node: Rc<T> = Rc::new(f(idx));
        self.slots.borrow_mut().push(Slot { node: Some(node), ..slot });
        idx
      }
    }
  }

  /// Dereferences a stored pointer. Panics if the slot is empty (the node was freed: a
  /// use-after-free upstream) and, in debug builds, if the index is stale (its node was freed and
  /// the slot reused since); release builds return the slot's current node there.
  pub fn get(&self, idx: NodeIdx) -> Rc<T> {
    let slots = self.slots.borrow();
    match slots.get(idx.slot as usize) {
      Some(s) => match &s.node {
        Some(n) => {
          Self::check_generation(idx, s, "access");
          n.clone()
        }
        None => panic!("node arena: access to freed node {:?}", idx),
      },
      None => panic!("node arena: access to never allocated node {:?}", idx),
    }
  }

  /// Whether `idx` currently holds a node (debugging and tests). Debug builds: false for a stale index.
  pub fn is_allocated(&self, idx: NodeIdx) -> bool {
    let slots = self.slots.borrow();
    match slots.get(idx.slot as usize) {
      Some(s) => {
        #[cfg(debug_assertions)]
        if s.generation != idx.generation {
          return false;
        }
        s.node.is_some()
      }
      None => false,
    }
  }

  /// `delete node`: empties the slot. The node itself is dropped once no `Rc` to it remains.
  pub fn free(&self, idx: NodeIdx) {
    let old: Option<Rc<T>> = {
      let mut slots = self.slots.borrow_mut();
      match slots.get_mut(idx.slot as usize) {
        Some(s) if s.node.is_some() => {
          Self::check_generation(idx, s, "free");
          #[cfg(debug_assertions)]
          {
            s.generation = s.generation.wrapping_add(1);
          }
          s.node.take()
        }
        _ => panic!("node arena: double free of node {:?}", idx),
      }
    };
    self.free_list.borrow_mut().push(idx.slot);
    // Dropped outside the borrow: a node's Drop never touches the arena, but keep it safe anyway.
    drop(old);
  }

  /// Number of live nodes (tests and debugging).
  pub fn num_allocated(&self) -> usize {
    self.slots.borrow().len() - self.free_list.borrow().len()
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn alloc_get_free_reuse() {
    let arena: NodeArena<(NodeIdx, i32)> = NodeArena::new();
    let a = arena.alloc(|idx| (idx, 1));
    let b = arena.alloc(|idx| (idx, 2));
    assert_eq!(arena.get(a).0, a);
    assert_eq!(arena.get(b).1, 2);
    assert_eq!(arena.num_allocated(), 2);
    let held = arena.get(a);
    arena.free(a);
    assert!(!arena.is_allocated(a));
    // A held Rc keeps the freed node alive, like an old pointer within a search.
    assert_eq!(held.1, 1);
    let c = arena.alloc(|idx| (idx, 3));
    assert_eq!(c.slot(), a.slot());
    assert_eq!(arena.get(c).1, 3);
    assert_eq!(arena.get(c).0, c);
    assert_eq!(arena.num_allocated(), 2);
  }

  #[test]
  #[should_panic(expected = "freed node")]
  fn get_after_free_panics() {
    let arena: NodeArena<i32> = NodeArena::new();
    let a = arena.alloc(|_| 1);
    arena.free(a);
    arena.get(a);
  }

  /// A stale index whose slot was reused: debug builds panic instead of reaching the new node.
  #[cfg(debug_assertions)]
  #[test]
  #[should_panic(expected = "stale node index")]
  fn stale_index_after_reuse_panics_in_debug() {
    let arena: NodeArena<i32> = NodeArena::new();
    let a = arena.alloc(|_| 1);
    arena.free(a);
    let b = arena.alloc(|_| 2);
    assert_eq!(a.slot(), b.slot());
    assert_ne!(a, b);
    assert!(!arena.is_allocated(a));
    assert!(arena.is_allocated(b));
    arena.get(a);
  }

  #[cfg(debug_assertions)]
  #[test]
  #[should_panic(expected = "stale node index")]
  fn stale_free_after_reuse_panics_in_debug() {
    let arena: NodeArena<i32> = NodeArena::new();
    let a = arena.alloc(|_| 1);
    arena.free(a);
    let _b = arena.alloc(|_| 2);
    arena.free(a);
  }
}
