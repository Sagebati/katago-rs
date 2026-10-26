//! Port-specific robustness of the ported Search (not a port of upstream code; no C++ counterpart):
//! what the Rust ownership model adds on top of upstream's raw pointers, and what the port does on
//! paths upstream treats as fatal.
//!
//! - A host holding the root's `Rc` (`Search::get_root_node`) across a search whose beginSearch
//!   filters root children and collapses the children arrays, then grows them again
//!   (collapseChildrenCapacity through `&self`, PORTING.md §4). Upstream: a held `SearchNode*` stays valid.
//! - `run_whole_search` failing mid-search (NaN NN outputs with many virtual threads): the tree is
//!   discarded (clearSearch) instead of keeping virtual losses and EVALUATING nodes, and the Search
//!   searches normally afterwards.
//! - `getAverageTreeOwnership` without a root and with alwaysIncludeOwnerMap off throws, like upstream.
//!
//! The NN is a small deterministic fake backend (features in, policy/value out), so the trees are not
//! compared with C++ here; tests/search_oracle.rs does that for the same paths (kind=collapse cases).

#![allow(clippy::needless_range_loop)]

mod common;

use std::cell::Cell;
use std::rc::Rc;

use common::run_or_stall_panic;
use futures::executor::LocalPool;
use katago_core::core::commontypes::enabled_t;
use katago_core::core::global::KataError;
use katago_core::game::board::*;
use katago_core::game::boardhistory::BoardHistory;
use katago_core::game::boardhistory::BoardHistoryModes;
use katago_core::game::rules::Rules;
use katago_core::neuralnet::desc::ModelDesc;
use katago_core::neuralnet::nneval::NNEvaluator;
use katago_core::neuralnet::nneval::NNResultBuf;
use katago_core::neuralnet::nninputs::NNOutput;
use katago_core::neuralnet::nninputs::ScoreValue;
use katago_core::neuralnet::nninterface::*;
use katago_core::port::executor::LocalBoxFuture;
use katago_core::port::executor::Spawner;
use katago_core::port::executor::yield_now;
use katago_core::search::search::Search;
use katago_core::search::searchnode::SearchNode;
use katago_core::search::searchparams::SearchParams;

//------------------------------------------------------------------------------------------------
// A deterministic fake backend (adapted from the M3 core review's stress harness)

struct FakeModel {
  desc: ModelDesc,
}
impl LoadedModel for FakeModel {
  fn as_any(&self) -> &dyn std::any::Any {
    self
  }
}
struct Unit;
impl ComputeContext for Unit {
  fn as_any(&self) -> &dyn std::any::Any {
    self
  }
}
impl InputBuffers for Unit {
  fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
    self
  }
}
struct Handle;
impl ComputeHandle for Handle {
  fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
    self
  }
  fn as_any(&self) -> &dyn std::any::Any {
    self
  }
}

struct FakeBackend {
  batches: Cell<usize>,
  /// Batches after this many return a NaN win probability ("Got nonfinite for nneval value").
  nan_after_batches: Cell<usize>,
}

impl NeuralNet for FakeBackend {
  fn global_initialize(&self) {}
  fn global_cleanup(&self) {}
  fn print_devices(&self) {}
  fn load_model_file(&self, _: &str, _: Vec<u8>, _: &str) -> Result<Box<dyn LoadedModel>, KataError> {
    let mut desc = ModelDesc::new();
    desc.name = "fake".to_string();
    desc.model_version = 8;
    desc.num_input_channels = 22;
    desc.num_input_global_channels = 19;
    desc.num_policy_channels = 1;
    desc.num_value_channels = 3;
    desc.num_score_value_channels = 4;
    desc.num_ownership_channels = 1;
    Ok(Box::new(FakeModel { desc }))
  }
  fn get_model_desc<'a>(&self, m: &'a dyn LoadedModel) -> &'a ModelDesc {
    &m.as_any().downcast_ref::<FakeModel>().expect("FakeModel").desc
  }
  fn create_compute_context(&self, _: &[i32], _: i32, _: i32, _: &str, _: &str, _: bool, _: enabled_t, _: enabled_t, _: &dyn LoadedModel) -> Result<Box<dyn ComputeContext>, KataError> {
    Ok(Box::new(Unit))
  }
  fn create_compute_handle(&self, _: &dyn ComputeContext, _: &dyn LoadedModel, _: i32, _: bool, _: bool, _: i32, _: i32) -> Result<Box<dyn ComputeHandle>, KataError> {
    Ok(Box::new(Handle))
  }
  fn is_using_fp16(&self, _: &dyn ComputeHandle) -> bool {
    false
  }
  fn create_input_buffers(&self, _: &dyn LoadedModel, _: i32, _: i32, _: i32) -> Box<dyn InputBuffers> {
    Box::new(Unit)
  }
  fn get_output<'a>(&'a self, _h: &'a mut dyn ComputeHandle, _: &'a mut dyn InputBuffers, n: i32, input_bufs: &'a [Rc<NNResultBuf>], outputs: &'a mut [NNOutput]) -> LocalBoxFuture<'a, Result<(), KataError>> {
    Box::pin(async move {
      let mut snaps: Vec<(Vec<f32>, Vec<f32>, i32)> = Vec::new();
      for row in 0..n as usize {
        snaps.push((input_bufs[row].row_spatial_buf.borrow().clone(), input_bufs[row].row_global_buf.borrow().clone(), input_bufs[row].symmetry.get()));
      }
      self.batches.set(self.batches.get() + 1);
      let nan: bool = self.batches.get() > self.nan_after_batches.get();
      // Let the other virtual threads run (and queue more rows) before this batch completes.
      yield_now().await;
      for row in 0..n as usize {
        let (spatial, global, sym) = &snaps[row];
        fill_fake(&mut outputs[row], spatial, global, *sym);
        if nan {
          outputs[row].white_win_prob = f32::NAN;
        }
      }
      Ok(())
    })
  }
}

/// Raw "logits" from the input features (19x19 NN layout, NCHW spatial input).
fn fill_fake(o: &mut NNOutput, spatial: &[f32], global: &[f32], sym: i32) {
  let stones: f32 = spatial[361..3 * 361].iter().sum();
  for i in 0..361 {
    o.policy_probs[i] = spatial[361 + i] * 0.3 - spatial[722 + i] * 0.2 + spatial[i] * 0.05 + sym as f32 * 0.01 + ((i * 7 + sym as usize) % 11) as f32 * 0.02;
  }
  o.policy_probs[361] = global[5] * 0.1 + sym as f32 * 0.003 - 1.0;
  o.white_win_prob = stones * 0.05 + sym as f32 * 0.001;
  o.white_loss_prob = global[5];
  o.white_no_result_prob = -2.0;
  o.white_score_mean = stones * 0.01;
  o.white_score_mean_sq = 0.5;
  o.white_lead = stones * 0.005;
  o.var_time_left = 1.0;
  o.shortterm_winloss_error = 0.0;
  o.shortterm_score_error = 0.0;
  if let Some(m) = &mut o.white_owner_map {
    for (i, x) in m.iter_mut().enumerate() {
      *x = spatial[361 + i] - spatial[2 * 361 + i];
    }
  }
}

fn init() {
  common::init();
  ScoreValue::init_tables();
  katago_core::core::timer::set_clock(common::fixed_clock);
}

fn make_eval(pool: &mut LocalPool, backend: Rc<FakeBackend>, max_batch: i32, num_server_threads: usize) -> Rc<NNEvaluator> {
  let spawner: Rc<dyn Spawner> = Rc::new(pool.spawner());
  let gpus: Vec<i32> = vec![-1; num_server_threads];
  let e = Rc::new(
    NNEvaluator::new(
      "fake", "fake.bin.gz", "", max_batch, 19, 19, false, false, 10, 0, false, "", "", false, enabled_t::False, enabled_t::False,
      gpus.len() as i32, &gpus, "robustness", true, 0, backend, spawner, Vec::new(),
    )
    .expect("NNEvaluator::new"),
  );
  run_or_stall_panic(pool, "spawnServerThreads", e.spawn_server_threads()).expect("spawnServerThreads");
  e
}

fn delete_eval(pool: &mut LocalPool, e: Rc<NNEvaluator>) {
  e.kill_server_threads();
  pool.run_until_stalled();
  assert_eq!(Rc::strong_count(&e), 1, "the serve tasks ended and nothing still holds the evaluator");
}

fn backend(nan_after_batches: usize) -> Rc<FakeBackend> {
  Rc::new(FakeBackend { batches: Cell::new(0), nan_after_batches: Cell::new(nan_after_batches) })
}

/// The tree invariants a finished search must satisfy (no virtual losses or transient states left,
/// children contiguous in their arrays, every child allocated, root visits = 1 + its edge visits when
/// nothing was filtered...). Returns the number of reachable nodes.
fn check_tree(search: &Search, tag: &str) -> usize {
  let nodes = search.enumerate_tree_post_order();
  for &n in &nodes {
    let node = search.node_arena.get(n);
    assert_eq!(node.virtual_losses.get(), 0, "{tag}: a node keeps virtual losses");
    let st = node.state.get();
    assert!(st != SearchNode::STATE_EVALUATING && st != SearchNode::STATE_GROWING1 && st != SearchNode::STATE_GROWING2, "{tag}: transient state {st}");
    let children = node.get_children();
    let mut seen_null = false;
    for j in 0..children.get_capacity() {
      match children.index(j).get_if_allocated() {
        None => seen_null = true,
        Some(c) => {
          assert!(!seen_null, "{tag}: child after an empty slot");
          assert!(search.node_arena.is_allocated(c), "{tag}: dangling child index");
        }
      }
    }
  }
  nodes.len()
}

fn num_root_children(root: &SearchNode) -> i32 {
  let children = root.get_children();
  let mut n: i32 = 0;
  while n < children.get_capacity() && children.index(n).get_if_allocated().is_some() {
    n += 1;
  }
  n
}

//------------------------------------------------------------------------------------------------

/// The core review's repro (review_held_root_rc_vs_begin_search_filtering), extended through the
/// regrowth of the collapsed arrays: a search on the empty 5x5 board without symmetry pruning (the
/// root gets all 26 children: STATE_EXPANDED1), then searches that reuse the tree with
/// rootSymmetryPruning (beginSearch filters the symmetric duplicates and collapses the root to
/// STATE_EXPANDED0) and again without it (the root grows past 8 children again), all while the host
/// holds the root's Rc. Before the fix, the first filtering search panicked in NodeArena::with_mut.
#[test]
fn held_root_rc_across_root_filtering_collapse_and_regrowth() {
  init();
  let mut pool = LocalPool::new();
  let e = make_eval(&mut pool, backend(usize::MAX), 4, 1);
  let mut params = SearchParams::new();
  params.num_threads = 1;
  params.max_visits = 400;
  let rules = Rules::new_full(Rules::KO_POSITIONAL, Rules::SCORING_AREA, Rules::TAX_NONE, true, false, 0, false, 7.5);
  let board = Board::new_x_y(5, 5).expect("5x5");
  let hist = BoardHistory::new_full(&board, P_BLACK, &rules, 0, &BoardHistoryModes::new());
  let mut search = Box::new(Search::new(&params, e.clone(), "held"));
  search.set_position(P_BLACK, &board, &hist).expect("setPosition");
  run_or_stall_panic(&mut pool, "search 1", search.run_whole_search(P_BLACK)).expect("search 1");
  check_tree(&search, "search 1");

  let held: Rc<SearchNode> = search.get_root_node().expect("root");
  assert_eq!(held.state.get(), SearchNode::STATE_EXPANDED1, "the root has more than 8 children");
  let children_before: i32 = num_root_children(&held);
  assert!(children_before > 8, "{children_before} children");

  // beginSearch filters the symmetric duplicates: the root collapses to the first children array.
  let mut p2 = search.search_params.clone();
  p2.root_symmetry_pruning = true;
  search.set_params_no_clearing(&p2);
  run_or_stall_panic(&mut pool, "search 2", search.run_whole_search(P_BLACK)).expect("search 2 (filtering the reused root)");
  assert_eq!(search.root_node, Some(held.idx), "same root node");
  assert_eq!(held.state.get(), SearchNode::STATE_EXPANDED0, "collapsed to children0");
  let children_filtered: i32 = num_root_children(&held);
  assert!(children_filtered <= 8, "{children_filtered} children after filtering");
  check_tree(&search, "search 2");
  let mut edge_sum: i64 = 0;
  {
    let children = held.get_children();
    for j in 0..children_filtered {
      edge_sum += children.index(j).get_edge_visits();
    }
  }
  assert_eq!(held.stats.visits.get(), edge_sum + 1, "root visits = 1 + edge visits");

  // Without pruning the root grows again, into the kept (emptied) children1 array.
  let mut p3 = search.search_params.clone();
  p3.root_symmetry_pruning = false;
  p3.max_visits = 1200;
  search.set_params_no_clearing(&p3);
  run_or_stall_panic(&mut pool, "search 3", search.run_whole_search(P_BLACK)).expect("search 3 (regrowth)");
  assert_eq!(search.root_node, Some(held.idx), "same root node");
  assert!(held.state.get() >= SearchNode::STATE_EXPANDED1, "grew past children0 again");
  let children_after: i32 = num_root_children(&held);
  assert!(children_after > 8, "{children_after} children after regrowth");
  let n = check_tree(&search, "search 3");
  assert_eq!(held.stats.visits.get(), 1200);
  crate::report!("held root: {children_before} -> {children_filtered} -> {children_after} root children, {n} nodes");

  drop(held);
  search.clear_search();
  assert_eq!(search.node_arena.num_allocated(), 0);
  drop(search);
  delete_eval(&mut pool, e);
}

/// NaN NN values mid-search with 16 virtual threads: runWholeSearch returns the evaluator's error
/// (upstream throws, which ends the process) and the port discards the tree (clearSearch), so no
/// virtual loss or EVALUATING node survives. Then a normal search runs on the same Search and
/// evaluator. Before the fix the root stayed, with virtual losses left on the in-flight paths.
#[test]
fn run_whole_search_error_discards_the_tree() {
  init();
  let mut total_errors: usize = 0;
  for (k, &(threads, nan_after, graph)) in [(16, 12usize, false), (16, 20, true), (4, 7, true), (64, 30, false)].iter().enumerate() {
    let mut pool = LocalPool::new();
    let be = backend(nan_after);
    let e = make_eval(&mut pool, be.clone(), 3, 2);
    let mut params = SearchParams::new();
    params.num_threads = threads;
    params.max_visits = 400;
    params.use_graph_search = graph;
    let rules = Rules::new_full(Rules::KO_POSITIONAL, Rules::SCORING_AREA, Rules::TAX_NONE, true, false, 0, false, 7.5);
    let board = Board::new_x_y(9, 9).expect("9x9");
    let hist = BoardHistory::new_full(&board, P_BLACK, &rules, 0, &BoardHistoryModes::new());
    let mut search = Box::new(Search::new(&params, e.clone(), &format!("err{k}")));
    search.set_position(P_BLACK, &board, &hist).expect("setPosition");

    let r = run_or_stall_panic(&mut pool, "failing search", search.run_whole_search(P_BLACK));
    match r {
      Err(err) => {
        assert!(format!("{:?}", err).contains("nonfinite"), "case {k}: unexpected error {:?}", err);
        total_errors += 1;
      }
      Ok(()) => panic!("case {k}: the NaN outputs did not surface as an error"),
    }
    let vloss_nodes: usize = match search.root_node {
      None => 0,
      Some(_) => search.enumerate_tree_post_order().iter().filter(|n| search.node_arena.get(**n).virtual_losses.get() != 0).count(),
    };
    assert_eq!(vloss_nodes, 0, "case {k}: nodes keep virtual losses after the failed search");
    assert!(search.root_node.is_none(), "case {k}: the tree of the failed search is discarded");
    assert_eq!(search.node_arena.num_allocated(), 0, "case {k}: every node freed");

    // The same Search searches normally afterwards.
    be.nan_after_batches.set(usize::MAX);
    run_or_stall_panic(&mut pool, "search after the error", search.run_whole_search(P_BLACK)).expect("search after the error");
    check_tree(&search, &format!("case {k} after the error"));
    assert!(search.get_root_visits() >= 400, "case {k}");
    drop(search);
    delete_eval(&mut pool, e);
  }
  crate::report!("run_whole_search_error_discards_the_tree: {total_errors} failing searches");
}

/// Upstream checks alwaysIncludeOwnerMap before it touches the (possibly NULL) root, so the call
/// throws instead of dereferencing it. Before the fix the port panicked on the missing root first.
#[test]
fn get_average_tree_ownership_without_root_or_owner_map_throws() {
  init();
  let mut pool = LocalPool::new();
  let e = make_eval(&mut pool, backend(usize::MAX), 4, 1);
  let params = SearchParams::new();
  let search = Box::new(Search::new(&params, e.clone(), "own"));
  assert!(search.get_root_node().is_none());
  match search.get_average_tree_ownership(None) {
    Err(KataError::StringError(m)) => assert_eq!(m, "Called Search::getAverageTreeOwnership when alwaysIncludeOwnerMap is false"),
    other => panic!("expected the StringError, got {:?}", other.map(|v| v.len())),
  }
  drop(search);
  delete_eval(&mut pool, e);
}
