// Ported from KataGo cpp/search/searchmultithreadhelpers.cpp @ fd0723fdbc0e9d82cf269c9630af8c27c57c07c4
//
// PORT NOTE (PORTING.md §11): there is no task thread pool. numAdditionalThreadsToUseForTasks()
// returns 0, so every sharded or tree-walking task runs on thread index 0 only (and no PCG32 is
// drawn from nonSearchRand for additional threads); spawnThreadsIfNeeded / killThreads /
// respawnThreads are prescribed no-ops; threadTaskLoop (the task threads' body) has no Rust fn.
// The tree walks take &self (nodeAge marking goes through Cells) and their callbacks are
// `&mut dyn FnMut(&SearchNode, i32)`.

use std::collections::HashSet;
use std::rc::Rc;

use crate::core::rand_helpers::PCG32;
use crate::port::node_arena::NodeIdx;
use crate::search::search::Search;
use crate::search::searchnode::SearchNode;
use crate::search::searchnode::SearchNodeChildrenReference;

/// A tree-walk callback (`std::function<void(SearchNode*,int)>*`, NULL = mark only).
pub type NodeFn<'f> = Option<&'f mut dyn FnMut(&SearchNode, i32)>;

/// upstream: maybeAppendShuffledIntRange (searchmultithreadhelpers.cpp) static
fn maybe_append_shuffled_int_range(cap: i32, rand: &mut Option<&mut PCG32>, rand_buf: &mut Vec<i32>) {
  if let Some(rand) = rand.as_mut() {
    let rand_buf_start: usize = rand_buf.len();
    for i in 0..cap {
      rand_buf.push(i);
    }
    for i in 1..cap {
      let r: i32 = (rand.next_uint() % (i + 1) as u32) as i32;
      let tmp: i32 = rand_buf[rand_buf_start + i as usize];
      rand_buf[rand_buf_start + i as usize] = rand_buf[rand_buf_start + r as usize];
      rand_buf[rand_buf_start + r as usize] = tmp;
    }
  }
}

impl Search {
  /// upstream: Search::numAdditionalThreadsToUseForTasks (searchmultithreadhelpers.cpp)
  /// PORT NOTE (PORTING.md §11): returns 0 instead of `searchParams.numThreads-1`: no task threads.
  pub(crate) fn num_additional_threads_to_use_for_tasks(&self) -> i32 {
    0
  }

  /// upstream: Search::spawnThreadsIfNeeded (searchmultithreadhelpers.cpp)
  /// PORT NOTE (PORTING.md §11): prescribed no-op, there are no task threads to spawn.
  pub(crate) fn spawn_threads_if_needed(&self) {}

  /// upstream: Search::killThreads (searchmultithreadhelpers.cpp)
  /// PORT NOTE (PORTING.md §11): prescribed no-op, there are no task threads to kill.
  pub(crate) fn kill_threads(&self) {}

  /// upstream: Search::respawnThreads (searchmultithreadhelpers.cpp)
  //If the number of threads is reduced, this can free up some excess threads in the thread pool.
  //Calling this is never necessary, it may just reduce some resource use.
  /// PORT NOTE (PORTING.md §11): kills and respawns nothing.
  pub fn respawn_threads(&self) {
    self.kill_threads();
    self.spawn_threads_if_needed();
  }

  /// upstream: Search::performTaskWithThreads (searchmultithreadhelpers.cpp)
  /// PORT NOTE (PORTING.md §11): non-search tasks only (the search loop is a join_all in
  /// runWholeSearch). With numAdditionalThreadsToUseForTasks() == 0 the task always runs inline on
  /// thread index 0.
  pub(crate) fn perform_task_with_threads(&self, task: &mut dyn FnMut(i32), cap_threads: i32) {
    self.spawn_threads_if_needed();
    let num_additional_threads_to_use: i32 = std::cmp::min(cap_threads - 1, self.num_additional_threads_to_use_for_tasks());
    if num_additional_threads_to_use <= 0 {
      task(0);
    } else {
      unreachable!("numAdditionalThreadsToUseForTasks() is 0 in the port (PORTING.md §11)");
    }
  }

  /// upstream: Search::applyRecursivelyPostOrderMulithreaded (searchmultithreadhelpers.cpp)
  //Walk over all nodes and their children recursively and call f, children first.
  //Assumes that only other instances of this function are running - in particular, the tree
  //is not being mutated by something else. It's okay if f mutates nodes, so long as it only mutates
  //nodes that will no longer be iterated over (namely, only stuff at the node or within its subtree).
  //As a side effect, nodeAge == searchNodeAge will be true only for the nodes walked over.
  pub(crate) fn apply_recursively_post_order_mulithreaded(&self, nodes: &[NodeIdx], mut f: NodeFn<'_>) {
    //We invalidate all node ages so we can use them as a marker for what's done.
    self.search_node_age.set(self.search_node_age.get() + 1);

    //Simple cheap RNGs so that we can get the different threads into different parts of the tree and not clash.
    let num_additional_threads: i32 = self.num_additional_threads_to_use_for_tasks();
    let mut rands: Vec<Option<PCG32>> = Vec::new();
    rands.resize_with((num_additional_threads + 1) as usize, || None);
    for thread_idx in 1..num_additional_threads + 1 {
      rands[thread_idx as usize] = Some(PCG32::new(self.non_search_rand.borrow_mut().next_uint64()));
    }

    let num_children: i32 = nodes.len() as i32;
    let mut g = |thread_idx: i32| {
      debug_assert!(thread_idx >= 0 && (thread_idx as usize) < rands.len());
      let mut rand: Option<&mut PCG32> = rands[thread_idx as usize].as_mut();
      let mut node_buf: HashSet<NodeIdx> = HashSet::new();
      let mut rand_buf: Vec<i32> = Vec::new();

      let rand_buf_start: usize = rand_buf.len();
      maybe_append_shuffled_int_range(num_children, &mut rand, &mut rand_buf);
      for i in 0..num_children {
        let child_idx: i32 = if rand.is_some() { rand_buf[rand_buf_start + i as usize] } else { i };
        let node: Rc<SearchNode> = self.node_arena.get(nodes[child_idx as usize]);
        self.apply_recursively_post_order_mulithreaded_helper(&node, thread_idx, &mut rand, &mut node_buf, &mut rand_buf, &mut f);
      }
      rand_buf.truncate(rand_buf_start);
    };
    self.perform_task_with_threads(&mut g, 0x3fffFFFF);
    // PORT NOTE: `delete rands[threadIdx]` is the drop of the Vec.
  }

  /// upstream: Search::applyRecursivelyPostOrderMulithreadedHelper (searchmultithreadhelpers.cpp)
  pub(crate) fn apply_recursively_post_order_mulithreaded_helper(
    &self,
    node: &SearchNode,
    thread_idx: i32,
    rand: &mut Option<&mut PCG32>,
    node_buf: &mut HashSet<NodeIdx>,
    rand_buf: &mut Vec<i32>,
    f: &mut NodeFn<'_>,
  ) {
    //nodeAge == searchNodeAge means that the node is done.
    if node.node_age.get() /*acquire*/ == self.search_node_age.get() {
      return;
    }
    //Cycle! Just consider this node "done" and return.
    if node_buf.contains(&node.idx) {
      return;
    }

    //Recurse on all children
    let children: SearchNodeChildrenReference = node.get_children();
    let num_children: i32 = children.iterate_and_count_children();

    if num_children > 0 {
      let rand_buf_start: usize = rand_buf.len();
      maybe_append_shuffled_int_range(num_children, rand, rand_buf);

      node_buf.insert(node.idx);
      for i in 0..num_children {
        let child_idx: i32 = if rand.is_some() { rand_buf[rand_buf_start + i as usize] } else { i };
        let child: Rc<SearchNode> = self.node_arena.get(children.index(child_idx).get_if_allocated().expect("counted children are allocated"));
        self.apply_recursively_post_order_mulithreaded_helper(&child, thread_idx, rand, node_buf, rand_buf, f);
      }
      rand_buf.truncate(rand_buf_start);
      node_buf.remove(&node.idx);
    }

    //Now call postorder function, protected by lock
    // PORT NOTE: lock removed (single-threaded).
    //Make sure another node didn't get there first.
    if node.node_age.get() /*acquire*/ == self.search_node_age.get() {
      return;
    }
    if let Some(f) = f.as_mut() {
      f(node, thread_idx);
    }
    node.node_age.set(self.search_node_age.get()) /*release*/;
  }

  /// upstream: Search::applyRecursivelyAnyOrderMulithreaded (searchmultithreadhelpers.cpp)
  //Walk over all nodes and their children recursively and call f. No order guarantee, but does guarantee that f is called only once per node.
  //Assumes that only other instances of this function are running - in particular, the tree
  //is not being mutated by something else. It's okay if f mutates nodes.
  //As a side effect, nodeAge == searchNodeAge will be true only for the nodes walked over.
  pub(crate) fn apply_recursively_any_order_mulithreaded(&self, nodes: &[NodeIdx], mut f: NodeFn<'_>) {
    //We invalidate all node ages so we can use them as a marker for what's done.
    self.search_node_age.set(self.search_node_age.get() + 1);

    //Simple cheap RNGs so that we can get the different threads into different parts of the tree and not clash.
    let num_additional_threads: i32 = self.num_additional_threads_to_use_for_tasks();
    let mut rands: Vec<Option<PCG32>> = Vec::new();
    rands.resize_with((num_additional_threads + 1) as usize, || None);
    for thread_idx in 1..num_additional_threads + 1 {
      rands[thread_idx as usize] = Some(PCG32::new(self.non_search_rand.borrow_mut().next_uint64()));
    }

    let num_children: i32 = nodes.len() as i32;
    let mut g = |thread_idx: i32| {
      debug_assert!(thread_idx >= 0 && (thread_idx as usize) < rands.len());
      let mut rand: Option<&mut PCG32> = rands[thread_idx as usize].as_mut();
      let mut node_buf: HashSet<NodeIdx> = HashSet::new();
      let mut rand_buf: Vec<i32> = Vec::new();

      let rand_buf_start: usize = rand_buf.len();
      maybe_append_shuffled_int_range(num_children, &mut rand, &mut rand_buf);
      for i in 0..num_children {
        let child_idx: i32 = if rand.is_some() { rand_buf[rand_buf_start + i as usize] } else { i };
        let node: Rc<SearchNode> = self.node_arena.get(nodes[child_idx as usize]);
        self.apply_recursively_any_order_mulithreaded_helper(&node, thread_idx, &mut rand, &mut node_buf, &mut rand_buf, &mut f);
      }
      rand_buf.truncate(rand_buf_start);
    };
    self.perform_task_with_threads(&mut g, 0x3fffFFFF);
    // PORT NOTE: `delete rands[threadIdx]` is the drop of the Vec.
  }

  /// upstream: Search::applyRecursivelyAnyOrderMulithreadedHelper (searchmultithreadhelpers.cpp)
  pub(crate) fn apply_recursively_any_order_mulithreaded_helper(
    &self,
    node: &SearchNode,
    thread_idx: i32,
    rand: &mut Option<&mut PCG32>,
    node_buf: &mut HashSet<NodeIdx>,
    rand_buf: &mut Vec<i32>,
    f: &mut NodeFn<'_>,
  ) {
    //nodeAge == searchNodeAge means that the node is done.
    if node.node_age.get() /*acquire*/ == self.search_node_age.get() {
      return;
    }
    //Cycle! Just consider this node "done" and return.
    if node_buf.contains(&node.idx) {
      return;
    }

    //Recurse on all children
    let children: SearchNodeChildrenReference = node.get_children();
    let num_children: i32 = children.iterate_and_count_children();

    if num_children > 0 {
      let rand_buf_start: usize = rand_buf.len();
      maybe_append_shuffled_int_range(num_children, rand, rand_buf);

      node_buf.insert(node.idx);
      for i in 0..num_children {
        let child_idx: i32 = if rand.is_some() { rand_buf[rand_buf_start + i as usize] } else { i };
        let child: Rc<SearchNode> = self.node_arena.get(children.index(child_idx).get_if_allocated().expect("counted children are allocated"));
        self.apply_recursively_any_order_mulithreaded_helper(&child, thread_idx, rand, node_buf, rand_buf, f);
      }
      rand_buf.truncate(rand_buf_start);
      node_buf.remove(&node.idx);
    }

    //The thread that is first to update it wins and does the action.
    let old_age: u32 = node.node_age.replace(self.search_node_age.get()) /*acq_rel*/;
    if old_age == self.search_node_age.get() {
      return;
    }
    if let Some(f) = f.as_mut() {
      f(node, thread_idx);
    }
  }

  /// upstream: Search::enumerateTreePostOrder (searchmultithreadhelpers.cpp)
  //Mainly for testing
  pub fn enumerate_tree_post_order(&self) -> Vec<NodeIdx> {
    let root_node: NodeIdx = self.root_node.expect("enumerateTreePostOrder needs a root node");
    let size_counter: std::cell::Cell<i64> = std::cell::Cell::new(0);
    let mut f = |node: &SearchNode, thread_idx: i32| {
      let _ = node;
      let _ = thread_idx;
      crate::port::cpp::cell_fetch_add(&size_counter, 1i64) /*relaxed*/;
    };
    self.apply_recursively_post_order_mulithreaded(&[root_node], Some(&mut f));

    let size: i64 = size_counter.get() /*relaxed*/;
    let mut nodes: Vec<Option<NodeIdx>> = vec![None; size as usize];
    let index_counter: std::cell::Cell<i64> = std::cell::Cell::new(0);
    let mut g = |node: &SearchNode, thread_idx: i32| {
      let _ = thread_idx;
      let index: i64 = crate::port::cpp::cell_fetch_add(&index_counter, 1i64) /*relaxed*/;
      debug_assert!(index >= 0 && index < size);
      nodes[index as usize] = Some(node.idx);
    };
    self.apply_recursively_post_order_mulithreaded(&[root_node], Some(&mut g));
    debug_assert!(index_counter.get() /*relaxed*/ == size);
    nodes.into_iter().map(|n| n.expect("every slot filled")).collect()
  }
}
