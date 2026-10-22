//! Search node storage (docs/PORTING.md §4, DECISIONS.md D11). Not a port of any upstream file.
//!
//! Upstream allocates every `SearchNode` with `new` and links them with raw `SearchNode*`
//! pointers (children arrays, the graph-search node table, `rootNode`, `graphPath`,
//! `AnalysisData::node`). The port stores the nodes here, as `Rc<T>` slots addressed by a
//! [`NodeIdx`]; a stored pointer becomes a `NodeIdx` and pointer identity becomes index
//! equality. Every method takes `&self`, because search code runs through `&Search` while the
//! virtual threads interleave (PORTING.md §11). `free(idx)` is `delete node`: the slot is
//! emptied (anyone still holding an `Rc` keeps the node alive, as upstream's old-pointer
//! guarantees within a search) and the index may be reused by a later `alloc`, like freed
//! memory. Indices are never observable in outputs (nothing iterates nodes in index order).

use std::cell::RefCell;
use std::rc::Rc;

/// A `SearchNode*` stored in a field or collection: an index into the search's [`NodeArena`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeIdx(pub u32);

/// The owner of every node of one search (PORTING.md §4).
pub struct NodeArena<T> {
  slots: RefCell<Vec<Option<Rc<T>>>>,
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

  /// `new T(...)`: `f` receives the index the node will live at (SearchNode stores it as `idx`).
  pub fn alloc<F: FnOnce(NodeIdx) -> T>(&self, f: F) -> NodeIdx {
    let reused: Option<u32> = self.free_list.borrow_mut().pop();
    match reused {
      Some(i) => {
        let node: Rc<T> = Rc::new(f(NodeIdx(i)));
        let mut slots = self.slots.borrow_mut();
        debug_assert!(slots[i as usize].is_none());
        slots[i as usize] = Some(node);
        NodeIdx(i)
      }
      None => {
        let i: u32 = {
          let slots = self.slots.borrow();
          u32::try_from(slots.len()).expect("node arena: more than 2^32 nodes")
        };
        let node: Rc<T> = Rc::new(f(NodeIdx(i)));
        self.slots.borrow_mut().push(Some(node));
        NodeIdx(i)
      }
    }
  }

  /// Dereferences a stored pointer. Panics if the node was freed (a use-after-free upstream).
  pub fn get(&self, idx: NodeIdx) -> Rc<T> {
    let slots = self.slots.borrow();
    match slots.get(idx.0 as usize) {
      Some(Some(n)) => n.clone(),
      _ => panic!("node arena: access to freed or never allocated node {:?}", idx),
    }
  }

  /// Whether `idx` currently holds a node (debugging and tests).
  pub fn is_allocated(&self, idx: NodeIdx) -> bool {
    let slots = self.slots.borrow();
    matches!(slots.get(idx.0 as usize), Some(Some(_)))
  }

  /// `delete node`: empties the slot. The node itself is dropped once no `Rc` to it remains.
  pub fn free(&self, idx: NodeIdx) {
    let old: Option<Rc<T>> = {
      let mut slots = self.slots.borrow_mut();
      match slots.get_mut(idx.0 as usize) {
        Some(slot) if slot.is_some() => slot.take(),
        _ => panic!("node arena: double free of node {:?}", idx),
      }
    };
    self.free_list.borrow_mut().push(idx.0);
    // Dropped outside the borrow: a node's Drop never touches the arena, but keep it safe anyway.
    drop(old);
  }

  /// Mutable access to one node, for the few upstream operations that are documented as not
  /// thread-safe and run outside the playout loop (`SearchNode::collapseChildrenCapacity`).
  /// Panics if anyone else holds an `Rc` to the node.
  pub fn with_mut<R, F: FnOnce(&mut T) -> R>(&self, idx: NodeIdx, f: F) -> R {
    let mut rc: Rc<T> = {
      let mut slots = self.slots.borrow_mut();
      match slots.get_mut(idx.0 as usize) {
        Some(slot) if slot.is_some() => slot.take().expect("checked is_some"),
        _ => panic!("node arena: with_mut on freed node {:?}", idx),
      }
    };
    let r: R = {
      let node: &mut T = Rc::get_mut(&mut rc).expect("node arena: with_mut on a node that is still shared (an Rc to it is alive)");
      f(node)
    };
    self.slots.borrow_mut()[idx.0 as usize] = Some(rc);
    r
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
    assert_eq!(c, a);
    assert_eq!(arena.get(c).1, 3);
    arena.with_mut(b, |n| n.1 = 20);
    assert_eq!(arena.get(b).1, 20);
  }

  #[test]
  #[should_panic(expected = "still shared")]
  fn with_mut_panics_when_shared() {
    let arena: NodeArena<i32> = NodeArena::new();
    let a = arena.alloc(|_| 1);
    let _held = arena.get(a);
    arena.with_mut(a, |n| *n = 2);
  }
}
