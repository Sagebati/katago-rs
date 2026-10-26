//! The async parts of NNEvaluator (neuralnet/nneval.rs) and port::executor with a fake backend:
//! server-task lifecycle (killServerThreads -> spawnServerThreads, DECISIONS.md D8: the deferred
//! join), waitForNextNNEvalIfAny and its kill wakeup, the symmetry choice of the serve task
//! (doRandomize / defaultSymmetry / requested) and of averageMultipleSymmetries, a post-processing
//! error, re-pushing a buf whose evaluate was cancelled, and randomized many-virtual-thread stress
//! runs. Not a port of upstream code. The expected symmetry streams are `Rand(seed +
//! ":NNEvalServerThread:" + k)` draws as upstream's serve() makes them; the same choices against
//! the unmodified upstream NNEvaluator are checked in katago-nn-burn's nn_parity (`evals_rand.txt`,
//! Eigen oracle). Every pool run goes through `common::run_or_stall_panic` / `run_until_stalled`,
//! so a lost wakeup fails instead of hanging.
#![allow(clippy::all)]

mod common;

use std::cell::Cell;
use std::cell::RefCell;
use std::rc::Rc;

use common::run_or_stall_panic;
use futures::executor::LocalPool;
use futures::task::LocalSpawnExt;
use katago_core::core::commontypes::enabled_t;
use katago_core::core::global::KataError;
use katago_core::core::rand::Rand;
use katago_core::game::board::*;
use katago_core::game::boardhistory::BoardHistory;
use katago_core::game::boardhistory::BoardHistoryModes;
use katago_core::game::rules::Rules;
use katago_core::neuralnet::desc::ModelDesc;
use katago_core::neuralnet::nneval::NNEvaluator;
use katago_core::neuralnet::nneval::NNResultBuf;
use katago_core::neuralnet::nninputs::MiscNNInputParams;
use katago_core::neuralnet::nninputs::NNInputs;
use katago_core::neuralnet::nninputs::NNOutput;
use katago_core::neuralnet::nninterface::*;
use katago_core::port::executor::AsyncQueue;
use katago_core::port::executor::LocalBoxFuture;
use katago_core::port::executor::Spawner;
use katago_core::port::executor::WakerList;
use katago_core::port::executor::yield_now;

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
struct Handle {
  id: i32,
}
impl ComputeHandle for Handle {
  fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
    self
  }
  fn as_any(&self) -> &dyn std::any::Any {
    self
  }
}
impl InputBuffers for Unit {
  fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
    self
  }
}

/// Deterministic fake net (model version 8): every output is a function of that row's own input
/// features and symmetry, so misrouted rows and wrong symmetries are visible. `get_output` can be
/// held at a gate (`gate_open`), can yield a random number of times, and can emit a NaN value.
struct FakeBackend {
  handles_created: Cell<i32>,
  handle_ids_used: RefCell<Vec<i32>>,
  batch_sizes: RefCell<Vec<i32>>,
  symmetries_seen: RefCell<Vec<i32>>,
  rng: RefCell<Rand>,
  max_yields: u32,
  gate_open: Cell<bool>,
  gate: WakerList,
  nan_value: Cell<bool>,
}

impl FakeBackend {
  fn new(seed: &str, max_yields: u32) -> FakeBackend {
    FakeBackend {
      handles_created: Cell::new(0),
      handle_ids_used: RefCell::new(Vec::new()),
      batch_sizes: RefCell::new(Vec::new()),
      symmetries_seen: RefCell::new(Vec::new()),
      rng: RefCell::new(Rand::new_seed(seed)),
      max_yields,
      gate_open: Cell::new(true),
      gate: WakerList::new(),
      nan_value: Cell::new(false),
    }
  }
  fn open_gate(&self) {
    self.gate_open.set(true);
    self.gate.notify_all();
  }
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
    &m.as_any().downcast_ref::<FakeModel>().unwrap().desc
  }
  fn create_compute_context(
    &self,
    _: &[i32],
    _: i32,
    _: i32,
    _: &str,
    _: &str,
    _: bool,
    _: enabled_t,
    _: enabled_t,
    _: &dyn LoadedModel,
  ) -> Result<Box<dyn ComputeContext>, KataError> {
    Ok(Box::new(Unit))
  }
  fn create_compute_handle(&self, _: &dyn ComputeContext, _: &dyn LoadedModel, _: i32, _: bool, _: bool, _: i32, _: i32) -> Result<Box<dyn ComputeHandle>, KataError> {
    let id = self.handles_created.get();
    self.handles_created.set(id + 1);
    Ok(Box::new(Handle { id }))
  }
  fn is_using_fp16(&self, _: &dyn ComputeHandle) -> bool {
    false
  }
  fn create_input_buffers(&self, _: &dyn LoadedModel, _: i32, _: i32, _: i32) -> Box<dyn InputBuffers> {
    Box::new(Unit)
  }
  fn get_output<'a>(
    &'a self,
    h: &'a mut dyn ComputeHandle,
    _: &'a mut dyn InputBuffers,
    n: i32,
    input_bufs: &'a [Rc<NNResultBuf>],
    outputs: &'a mut [NNOutput],
  ) -> LocalBoxFuture<'a, Result<(), KataError>> {
    Box::pin(async move {
      let hid = h.as_any().downcast_ref::<Handle>().unwrap().id;
      self.handle_ids_used.borrow_mut().push(hid);
      self.batch_sizes.borrow_mut().push(n);
      // Snapshot the inputs before awaiting, as a real backend does (copy into its buffers).
      let mut snaps: Vec<(Vec<f32>, Vec<f32>, i32)> = Vec::new();
      for row in 0..n as usize {
        snaps.push((input_bufs[row].row_spatial_buf.borrow().clone(), input_bufs[row].row_global_buf.borrow().clone(), input_bufs[row].symmetry.get()));
        self.symmetries_seen.borrow_mut().push(input_bufs[row].symmetry.get());
      }
      self.gate.wait_until(|| self.gate_open.get()).await;
      let ny = if self.max_yields == 0 { 0 } else { self.rng.borrow_mut().next_uint_n(self.max_yields + 1) };
      for _ in 0..ny {
        yield_now().await;
      }
      for row in 0..n as usize {
        let (spatial, global, sym) = &snaps[row];
        fill_fake(&mut outputs[row], spatial, global, *sym);
        if self.nan_value.get() {
          outputs[row].white_win_prob = f32::NAN;
        }
      }
      Ok(())
    })
  }
}

fn fill_fake(o: &mut NNOutput, spatial: &[f32], global: &[f32], sym: i32) {
  let stones: f32 = spatial[361..3 * 361].iter().sum();
  for i in 0..361 {
    o.policy_probs[i] = spatial[361 + i] * 0.3 - spatial[722 + i] * 0.2 + spatial[i] * 0.05 + sym as f32 * 0.01 + ((i * 7 + sym as usize) % 11) as f32 * 0.02;
  }
  o.policy_probs[361] = global[5] * 0.1 + sym as f32 * 0.003;
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
      *x = spatial[361 + i] - spatial[2 * 361 + i] + sym as f32 * 0.0001;
    }
  }
}

fn make_eval(backend: Rc<FakeBackend>, spawner: Rc<dyn Spawner>, max_batch: i32, cache_pow: i32, gpus: &[i32], do_randomize: bool, default_symmetry: i32) -> Rc<NNEvaluator> {
  Rc::new(
    NNEvaluator::new(
      "fake",
      "fake.bin.gz",
      "",
      max_batch,
      19,
      19,
      false,
      false,
      cache_pow,
      0,
      false,
      "",
      "",
      false,
      enabled_t::False,
      enabled_t::False,
      gpus.len() as i32,
      gpus,
      "seed",
      do_randomize,
      default_symmetry,
      backend,
      spawner,
      Vec::new(),
    )
    .unwrap(),
  )
}

/// Number of serve tasks alive: each holds two `Rc<NNEvaluator>` (serve's `self` and serveEvals'
/// own), and the test holds one.
fn live_serve_tasks(e: &Rc<NNEvaluator>) -> usize {
  (Rc::strong_count(e) - 1) / 2
}

struct Pos {
  board: Board,
  hist: BoardHistory,
  next_pla: Player,
}

fn positions(n: usize) -> Vec<Pos> {
  let mut rand = Rand::new_seed("positions");
  let mut v = Vec::new();
  let sizes = [(19, 19), (9, 9), (13, 13), (19, 11), (7, 7)];
  for k in 0..n {
    let (xs, ys) = sizes[k % sizes.len()];
    let rules = Rules::new_full(Rules::KO_POSITIONAL, Rules::SCORING_AREA, Rules::TAX_NONE, true, false, 0, false, 7.5 - k as f32);
    let mut board = Board::new_x_y(xs, ys).unwrap();
    let mut hist = BoardHistory::new_full(&board, P_BLACK, &rules, 0, &BoardHistoryModes::new());
    let nmoves = 3 + rand.next_uint_n(25);
    for _ in 0..nmoves {
      let pla = hist.presumed_next_move_pla;
      let mut tries = 0;
      loop {
        let x = rand.next_uint_n(xs as u32) as i32;
        let y = rand.next_uint_n(ys as u32) as i32;
        let loc = Location::get_loc(x, y, xs);
        if hist.is_legal(&board, loc, pla) && board.colors[loc as usize] == C_EMPTY {
          hist.make_board_move_assume_legal(&mut board, loc, pla, None, false);
          break;
        }
        tries += 1;
        if tries > 100 {
          break;
        }
      }
    }
    let next_pla = hist.presumed_next_move_pla;
    v.push(Pos { board, hist, next_pla });
  }
  v
}

fn params_sym(sym: i32) -> MiscNNInputParams {
  let mut p = MiscNNInputParams::new();
  p.symmetry = sym;
  p
}

/// The symmetry draws upstream's serve thread number `k` makes: `rand.nextUInt(NUM_SYMMETRIES)`.
fn server_stream(k: i32, n: usize) -> Vec<i32> {
  let mut r = Rand::new_seed(&format!("seed:NNEvalServerThread:{k}"));
  (0..n).map(|_| r.next_uint_n(8) as i32).collect()
}

fn evaluate_n(pool: &mut LocalPool, e: &Rc<NNEvaluator>, p: &Pos, buf: &Rc<NNResultBuf>, sym: i32, n: usize) -> Vec<i32> {
  let mut syms = Vec::new();
  for _ in 0..n {
    run_or_stall_panic(pool, "evaluate", e.evaluate(&p.board, &p.hist, p.next_pla, &params_sym(sym), buf, true, true)).unwrap();
    syms.push(buf.symmetry.get());
  }
  syms
}

/// Upstream testsearchv8 does `nnEval->killServerThreads(); nnEval->spawnServerThreads();`.
/// Upstream joins in between, so afterwards exactly one server thread serves: the new one, whose
/// Rand is seeded "...:NNEvalServerThread:1". (Before D8's deferred join the old serve task never
/// saw the read-only queue and kept serving with thread 0's Rand: review M2 repro.)
#[test]
fn kill_then_respawn_leaves_no_zombie_server() {
  common::init();
  let mut pool = LocalPool::new();
  let spawner: Rc<dyn Spawner> = Rc::new(pool.spawner());
  let backend = Rc::new(FakeBackend::new("x", 0));
  let e = make_eval(backend.clone(), spawner, 4, -1, &[-1], true, 0);
  run_or_stall_panic(&mut pool, "spawn", e.spawn_server_threads()).unwrap();
  let ps = positions(1);
  let p = &ps[0];
  let buf = Rc::new(NNResultBuf::new());
  let mut got = evaluate_n(&mut pool, &e, p, &buf, NNInputs::SYMMETRY_NOTSPECIFIED, 3);
  assert_eq!(got, server_stream(0, 3));
  for k in 1..=2 {
    // kill + respawn back to back, as testsearchv8.cpp does.
    e.kill_server_threads();
    run_or_stall_panic(&mut pool, "respawn", e.spawn_server_threads()).unwrap();
    pool.run_until_stalled();
    assert_eq!(live_serve_tasks(&e), 1, "a zombie serve task survived kill+respawn {k}");
    let first = backend.handle_ids_used.borrow().len();
    got = evaluate_n(&mut pool, &e, p, &buf, NNInputs::SYMMETRY_NOTSPECIFIED, 12);
    assert_eq!(got, server_stream(k, 12), "respawn {k}: the symmetry stream is not the new server thread's Rand");
    assert!(backend.handle_ids_used.borrow()[first..].iter().all(|&h| h == k), "respawn {k}: evals served by an old server task");
  }
  e.kill_server_threads();
  pool.run_until_stalled();
  assert_eq!(Rc::strong_count(&e), 1, "serve task ended and released the evaluator");
  report!("kill_then_respawn_leaves_no_zombie_server: {} evals", backend.handle_ids_used.borrow().len());
}

/// killServerThreads while a batch is in flight: the old serve task finishes that batch (as
/// upstream's join waits for it), and the next spawnServerThreads does not reopen the queue until
/// that task has ended. The client still gets its result, from the old task.
#[test]
fn respawn_waits_for_the_in_flight_batch() {
  common::init();
  let mut pool = LocalPool::new();
  let spawner: Rc<dyn Spawner> = Rc::new(pool.spawner());
  let backend = Rc::new(FakeBackend::new("x", 0));
  let e = make_eval(backend.clone(), spawner, 4, -1, &[-1], true, 0);
  run_or_stall_panic(&mut pool, "spawn", e.spawn_server_threads()).unwrap();
  let ps = Rc::new(positions(1));
  let buf = Rc::new(NNResultBuf::new());
  backend.gate_open.set(false);
  let client_done = Rc::new(Cell::new(false));
  {
    let (e, ps, buf, done) = (e.clone(), ps.clone(), buf.clone(), client_done.clone());
    LocalSpawnExt::spawn_local(&pool.spawner(), async move {
        let p = &ps[0];
        e.evaluate(&p.board, &p.hist, p.next_pla, &params_sym(NNInputs::SYMMETRY_NOTSPECIFIED), &buf, true, true).await.unwrap();
        done.set(true);
      })
      .unwrap();
  }
  pool.run_until_stalled();
  assert!(!client_done.get());
  e.kill_server_threads();
  let spawned = Rc::new(Cell::new(false));
  {
    let (e, spawned) = (e.clone(), spawned.clone());
    LocalSpawnExt::spawn_local(&pool.spawner(), async move {
        e.spawn_server_threads().await.unwrap();
        spawned.set(true);
      })
      .unwrap();
  }
  pool.run_until_stalled();
  assert!(!spawned.get(), "spawnServerThreads returned while the old serve task was still alive");
  assert!(!client_done.get());
  backend.open_gate();
  pool.run_until_stalled();
  assert!(client_done.get() && spawned.get());
  assert_eq!(buf.symmetry.get(), server_stream(0, 1)[0]);
  assert_eq!(live_serve_tasks(&e), 1);
  let got = evaluate_n(&mut pool, &e, &ps[0], &buf, NNInputs::SYMMETRY_NOTSPECIFIED, 4);
  assert_eq!(got, server_stream(1, 4));
  assert_eq!(*backend.handle_ids_used.borrow(), vec![0, 1, 1, 1, 1]);
  e.kill_server_threads();
  pool.run_until_stalled();
  assert_eq!(Rc::strong_count(&e), 1);
}

/// waitForNextNNEvalIfAny: returns at once with nothing in flight; otherwise waits until a batch
/// completes.
#[test]
fn wait_for_next_nn_eval_waits_for_a_batch() {
  common::init();
  let mut pool = LocalPool::new();
  let spawner: Rc<dyn Spawner> = Rc::new(pool.spawner());
  let backend = Rc::new(FakeBackend::new("x", 0));
  let e = make_eval(backend.clone(), spawner, 4, -1, &[-1], false, 0);
  run_or_stall_panic(&mut pool, "spawn", e.spawn_server_threads()).unwrap();
  run_or_stall_panic(&mut pool, "idle wait", e.wait_for_next_nn_eval_if_any());
  let ps = Rc::new(positions(1));
  backend.gate_open.set(false);
  let client_done = Rc::new(Cell::new(false));
  let waiter_done = Rc::new(Cell::new(false));
  {
    let (e, ps, done) = (e.clone(), ps.clone(), client_done.clone());
    LocalSpawnExt::spawn_local(&pool.spawner(), async move {
        let buf = Rc::new(NNResultBuf::new());
        let p = &ps[0];
        e.evaluate(&p.board, &p.hist, p.next_pla, &params_sym(0), &buf, true, true).await.unwrap();
        done.set(true);
      })
      .unwrap();
  }
  pool.run_until_stalled();
  {
    let (e, done, client_done) = (e.clone(), waiter_done.clone(), client_done.clone());
    LocalSpawnExt::spawn_local(&pool.spawner(), async move {
        e.wait_for_next_nn_eval_if_any().await;
        // The batch it waited for has been delivered.
        assert!(client_done.get());
        done.set(true);
      })
      .unwrap();
  }
  pool.run_until_stalled();
  assert!(!waiter_done.get(), "waitForNextNNEvalIfAny returned while an eval was in flight");
  backend.open_gate();
  pool.run_until_stalled();
  assert!(client_done.get() && waiter_done.get());
  e.kill_server_threads();
  pool.run_until_stalled();
  assert_eq!(Rc::strong_count(&e), 1);
}

/// killServerThreads wakes waitForNextNNEvalIfAny (`isKilled`, `waitingForFinish.notify_all()`)
/// even though the batch it waited for has not completed; the counters still balance at the
/// deferred join (upstream's post-join testAsserts run in the next spawnServerThreads).
#[test]
fn kill_wakes_wait_for_next_nn_eval() {
  common::init();
  let mut pool = LocalPool::new();
  let spawner: Rc<dyn Spawner> = Rc::new(pool.spawner());
  let backend = Rc::new(FakeBackend::new("x", 0));
  let e = make_eval(backend.clone(), spawner, 4, -1, &[-1], false, 0);
  run_or_stall_panic(&mut pool, "spawn", e.spawn_server_threads()).unwrap();
  let ps = Rc::new(positions(1));
  backend.gate_open.set(false);
  let client_done = Rc::new(Cell::new(false));
  let waiter_done = Rc::new(Cell::new(false));
  {
    let (e, ps, done) = (e.clone(), ps.clone(), client_done.clone());
    LocalSpawnExt::spawn_local(&pool.spawner(), async move {
        let buf = Rc::new(NNResultBuf::new());
        let p = &ps[0];
        e.evaluate(&p.board, &p.hist, p.next_pla, &params_sym(0), &buf, true, true).await.unwrap();
        done.set(true);
      })
      .unwrap();
  }
  pool.run_until_stalled();
  {
    let (e, done) = (e.clone(), waiter_done.clone());
    LocalSpawnExt::spawn_local(&pool.spawner(), async move {
        e.wait_for_next_nn_eval_if_any().await;
        done.set(true);
      })
      .unwrap();
  }
  pool.run_until_stalled();
  assert!(!waiter_done.get());
  e.kill_server_threads();
  pool.run_until_stalled();
  assert!(waiter_done.get(), "killServerThreads did not wake waitForNextNNEvalIfAny");
  assert!(!client_done.get());
  backend.open_gate();
  pool.run_until_stalled();
  assert!(client_done.get());
  // The deferred join (with the post-join testAsserts) and a fresh server.
  run_or_stall_panic(&mut pool, "respawn", e.spawn_server_threads()).unwrap();
  let buf = Rc::new(NNResultBuf::new());
  evaluate_n(&mut pool, &e, &ps[0], &buf, 0, 1);
  e.kill_server_threads();
  pool.run_until_stalled();
  assert_eq!(Rc::strong_count(&e), 1);
}

/// The serve task's symmetry choice: a requested symmetry is used as is (no Rand draw);
/// SYMMETRY_NOTSPECIFIED draws `nextUInt(8)` when doRandomize, else takes defaultSymmetry; both
/// settings change at runtime (setDoRandomize / setDefaultSymmetry).
#[test]
fn serve_symmetry_choice() {
  common::init();
  let mut pool = LocalPool::new();
  let spawner: Rc<dyn Spawner> = Rc::new(pool.spawner());
  let backend = Rc::new(FakeBackend::new("x", 0));
  let e = make_eval(backend.clone(), spawner, 4, -1, &[-1], true, 5);
  run_or_stall_panic(&mut pool, "spawn", e.spawn_server_threads()).unwrap();
  let ps = positions(2);
  let buf = Rc::new(NNResultBuf::new());
  let mut want_stream = server_stream(0, 6).into_iter();
  let mut got: Vec<i32> = Vec::new();
  let mut want: Vec<i32> = Vec::new();
  for (sym, randomize, default_sym) in [(-1, true, 5), (3, true, 5), (-1, true, 5), (-1, false, 5), (-1, false, 2), (6, false, 2), (-1, true, 2), (-1, true, 7)] {
    e.set_do_randomize(randomize);
    e.set_default_symmetry(default_sym);
    got.extend(evaluate_n(&mut pool, &e, &ps[got.len() % 2], &buf, sym, 1));
    want.push(if sym >= 0 {
      sym
    } else if randomize {
      want_stream.next().unwrap()
    } else {
      default_sym
    });
  }
  assert_eq!(got, want);
  assert_eq!(*backend.symmetries_seen.borrow(), want, "the backend saw other symmetries than the bufs report");
  e.kill_server_threads();
  pool.run_until_stalled();
}

/// averageMultipleSymmetries: partial Fisher-Yates over the 8 symmetries with the caller's Rand
/// (`nextInt(i, 7)`), each evaluated with that explicit symmetry, then NNOutput's averaging ctor.
#[test]
fn average_multiple_symmetries_choice() {
  common::init();
  let mut pool = LocalPool::new();
  let spawner: Rc<dyn Spawner> = Rc::new(pool.spawner());
  let backend = Rc::new(FakeBackend::new("x", 0));
  let e = make_eval(backend.clone(), spawner, 4, -1, &[-1], true, 0);
  run_or_stall_panic(&mut pool, "spawn", e.spawn_server_threads()).unwrap();
  let ps = positions(3);
  let buf = Rc::new(NNResultBuf::new());
  let mut cases = 0;
  for (k, num) in [1, 2, 3, 5, 8, 8].into_iter().enumerate() {
    let p = &ps[k % 3];
    let seed = format!("avg{k}");
    backend.symmetries_seen.borrow_mut().clear();
    let mut rand = Rand::new_seed(&seed);
    let res = run_or_stall_panic(
      &mut pool,
      "averageMultipleSymmetries",
      e.average_multiple_symmetries(&p.board, &p.hist, p.next_pla, None, &MiscNNInputParams::new(), &buf, true, &mut rand, num),
    )
    .unwrap();
    let mut r = Rand::new_seed(&seed);
    let mut idx: Vec<i32> = (0..8).collect();
    let mut want = Vec::new();
    for i in 0..num {
      let j = r.next_int_a_b(i, 7);
      idx.swap(i as usize, j as usize);
      want.push(idx[i as usize]);
    }
    assert_eq!(*backend.symmetries_seen.borrow(), want, "case {k}");
    // The average of the per-symmetry results, recomputed one eval at a time.
    let mut ptrs = Vec::new();
    for &s in &want {
      run_or_stall_panic(&mut pool, "evaluate", e.evaluate(&p.board, &p.hist, p.next_pla, &params_sym(s), &buf, true, true)).unwrap();
      ptrs.push(buf.result.borrow().clone().unwrap());
    }
    let avg = NNOutput::new_others(&ptrs);
    assert_eq!(res.white_win_prob.to_bits(), avg.white_win_prob.to_bits());
    assert_eq!(res.policy_probs.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), avg.policy_probs.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
    cases += 1;
  }
  e.kill_server_threads();
  pool.run_until_stalled();
  report!("average_multiple_symmetries_choice: {cases} cases");
}

/// A post-processing error (here a NaN value logit: "Got nonfinite for nneval value") returns the
/// error and, as upstream (which post-processes `buf.result` in place), leaves the partially
/// processed result in `buf.result`: policy already normalized, values still raw.
#[test]
fn postprocess_error_leaves_result_in_buf() {
  common::init();
  let mut pool = LocalPool::new();
  let spawner: Rc<dyn Spawner> = Rc::new(pool.spawner());
  let backend = Rc::new(FakeBackend::new("x", 0));
  let e = make_eval(backend.clone(), spawner, 4, -1, &[-1], false, 0);
  run_or_stall_panic(&mut pool, "spawn", e.spawn_server_threads()).unwrap();
  let ps = positions(1);
  let p = &ps[0];
  let buf = Rc::new(NNResultBuf::new());
  backend.nan_value.set(true);
  let r = run_or_stall_panic(&mut pool, "evaluate", e.evaluate(&p.board, &p.hist, p.next_pla, &params_sym(0), &buf, true, true));
  match r {
    Err(KataError::StringError(s)) => assert_eq!(s, "Got nonfinite for nneval value"),
    other => panic!("expected the nonfinite-value error, got {:?}", other.map(|_| ())),
  }
  assert!(buf.has_result.get());
  let res = buf.result.borrow().clone().expect("upstream leaves the partially processed result in buf.result");
  assert!(res.white_win_prob.is_nan(), "values are not post-processed yet");
  let sum: f64 = res.policy_probs[..362].iter().filter(|&&x| x >= 0.0).map(|&x| x as f64).sum();
  assert!((sum - 1.0).abs() < 1e-5, "policy is post-processed (sum {sum})");
  e.kill_server_threads();
  pool.run_until_stalled();
}

/// Evaluate futures must not be cancelled (PORTING.md §11): a dropped evaluate leaves its buf in
/// the queue, and evaluating with that buf again would queue it twice. Debug builds catch that.
#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "already queued")]
fn cancelled_evaluate_then_reuse_buf_debug_asserts() {
  common::init();
  let pool = LocalPool::new();
  let spawner: Rc<dyn Spawner> = Rc::new(pool.spawner());
  let backend = Rc::new(FakeBackend::new("x", 0));
  let e = make_eval(backend, spawner, 4, -1, &[-1], false, 0);
  let mut pool = pool;
  run_or_stall_panic(&mut pool, "spawn", e.spawn_server_threads()).unwrap();
  let ps = positions(1);
  let p = &ps[0];
  let buf = Rc::new(NNResultBuf::new());
  let waker = futures::task::noop_waker();
  let mut cx = std::task::Context::from_waker(&waker);
  {
    let params = params_sym(0);
    let mut f = Box::pin(e.evaluate(&p.board, &p.hist, p.next_pla, &params, &buf, true, true));
    assert!(f.as_mut().poll(&mut cx).is_pending());
    // dropped here: cancelled after force_push
  }
  let params = params_sym(0);
  let mut f = Box::pin(e.evaluate(&p.board, &p.hist, p.next_pla, &params, &buf, true, true));
  let _ = f.as_mut().poll(&mut cx);
}

//------------------------------------------------------------------------------------------------
// Stress: many virtual threads, random yields in clients and backend, several server tasks,
// changing batch size, waitForNextNNEvalIfAny, with and without the cache.

fn fp(o: &NNOutput) -> (u64, u32, u32, Vec<u32>, Option<Vec<u32>>) {
  (
    o.nn_hash.hash0,
    o.white_win_prob.to_bits(),
    o.white_score_mean.to_bits(),
    o.policy_probs.iter().map(|x| x.to_bits()).collect(),
    o.white_owner_map.as_ref().map(|m| m.iter().map(|x| x.to_bits()).collect()),
  )
}

fn stress(seed: &str, num_clients: usize, evals_per_client: usize, max_batch: i32, gpus: &[i32], cache_pow: i32, backend_yields: u32) {
  common::init();
  let ps = positions(10);
  // Reference: one eval at a time, no cache.
  let mut pool = LocalPool::new();
  let spawner: Rc<dyn Spawner> = Rc::new(pool.spawner());
  let refb = Rc::new(FakeBackend::new("ref", 0));
  let refe = make_eval(refb, spawner.clone(), 1, -1, &[-1], false, 0);
  run_or_stall_panic(&mut pool, "spawn", refe.spawn_server_threads()).unwrap();
  let mut reference = vec![vec![None; 8]; ps.len()];
  for (pi, p) in ps.iter().enumerate() {
    for sym in 0..8 {
      let buf = Rc::new(NNResultBuf::new());
      run_or_stall_panic(&mut pool, "evaluate", refe.evaluate(&p.board, &p.hist, p.next_pla, &params_sym(sym), &buf, true, true)).unwrap();
      reference[pi][sym as usize] = Some(fp(&buf.result.borrow().clone().unwrap()));
    }
  }
  refe.kill_server_threads();
  pool.run_until_stalled();

  let backend = Rc::new(FakeBackend::new(seed, backend_yields));
  let e = make_eval(backend.clone(), spawner.clone(), max_batch, cache_pow, gpus, false, 0);
  run_or_stall_panic(&mut pool, "spawn", e.spawn_server_threads()).unwrap();
  let done = Rc::new(Cell::new(0usize));
  let failures: Rc<RefCell<Vec<String>>> = Rc::new(RefCell::new(Vec::new()));
  let ps = Rc::new(ps);
  let reference = Rc::new(reference);
  let sp = pool.spawner();
  for c in 0..num_clients {
    let (e, done, failures, ps, reference) = (e.clone(), done.clone(), failures.clone(), ps.clone(), reference.clone());
    let mut rand = Rand::new_seed(&format!("{seed}:client{c}"));
    LocalSpawnExt::spawn_local(&sp, async move {
      let buf = Rc::new(NNResultBuf::new());
      for k in 0..evals_per_client {
        for _ in 0..rand.next_uint_n(4) {
          yield_now().await;
        }
        if rand.next_uint_n(5) == 0 {
          e.wait_for_next_nn_eval_if_any().await;
        }
        if rand.next_uint_n(7) == 0 {
          let bs = 1 + rand.next_uint_n(max_batch as u32) as i32;
          e.set_current_batch_size(bs).unwrap();
        }
        let pi = rand.next_uint_n(ps.len() as u32) as usize;
        let sym = rand.next_uint_n(8) as i32;
        let include_owner = rand.next_uint_n(3) != 0;
        let skip_cache = rand.next_uint_n(3) == 0;
        let p = &ps[pi];
        e.evaluate(&p.board, &p.hist, p.next_pla, &params_sym(sym), &buf, skip_cache, include_owner).await.unwrap();
        let got = fp(&buf.result.borrow().clone().unwrap());
        let refs = &reference[pi];
        let ok = if cache_pow < 0 || skip_cache {
          let want = refs[sym as usize].as_ref().unwrap();
          got.0 == want.0 && got.1 == want.1 && got.2 == want.2 && got.3 == want.3 && (!include_owner || got.4 == want.4)
        } else {
          // From the cache any symmetry may come back; policy/values must match one of the refs.
          got.0 == refs[0].as_ref().unwrap().0
            && refs.iter().any(|r| {
              let r = r.as_ref().unwrap();
              got.1 == r.1 && got.2 == r.2 && got.3 == r.3
            })
            && (!include_owner || got.4.is_some())
        };
        if !ok {
          failures.borrow_mut().push(format!("client {c} eval {k}: pos {pi} sym {sym} owner {include_owner} skip {skip_cache}: wrong result"));
        }
      }
      done.set(done.get() + 1);
    })
    .unwrap();
  }
  pool.run_until_stalled();
  let finished = done.get();
  let batches = backend.batch_sizes.borrow().clone();
  for f in failures.borrow().iter().take(5) {
    report!("  {f}");
  }
  assert_eq!(finished, num_clients, "stress {seed}: some virtual threads never finished (the pool stalled)");
  assert!(failures.borrow().is_empty(), "stress {seed}: wrong results");
  assert!(batches.iter().any(|&b| b > 1), "stress {seed}: nothing was batched");
  e.kill_server_threads();
  pool.run_until_stalled();
  assert_eq!(Rc::strong_count(&e), 1);
}

#[test]
fn stress_nocache_1server() {
  for s in 0..2 {
    stress(&format!("a{s}"), 37, 30, 8, &[-1], -1, 3);
  }
}
#[test]
fn stress_nocache_2servers() {
  for s in 0..2 {
    stress(&format!("b{s}"), 41, 30, 5, &[-1, -1], -1, 4);
  }
}
#[test]
fn stress_cache_3servers() {
  for s in 0..2 {
    stress(&format!("c{s}"), 29, 30, 16, &[-1, 0, 1], 3, 2);
  }
}
#[test]
fn stress_cache_no_backend_yield() {
  for s in 0..2 {
    stress(&format!("d{s}"), 33, 30, 3, &[-1, -1], 5, 0);
  }
}

//------------------------------------------------------------------------------------------------
// port::executor

/// FIFO across interleaved pushes and pops; waitPopUpToN appends min(size, n) without waiting.
#[test]
fn queue_fifo_and_up_to_n() {
  let mut pool = LocalPool::new();
  let q: Rc<AsyncQueue<i32>> = Rc::new(AsyncQueue::new());
  let got: Rc<RefCell<Vec<Vec<i32>>>> = Rc::new(RefCell::new(Vec::new()));
  let mut rand = Rand::new_seed("q");
  {
    let (q, got) = (q.clone(), got.clone());
    LocalSpawnExt::spawn_local(&pool.spawner(), async move {
        let mut n = 1;
        loop {
          let mut buf = vec![-1];
          if !q.wait_pop_up_to_n(&mut buf, n).await {
            break;
          }
          assert_eq!(buf[0], -1, "appends, does not clear");
          buf.remove(0);
          assert!(buf.len() <= n && !buf.is_empty());
          got.borrow_mut().push(buf);
          n = n % 5 + 1;
        }
      })
      .unwrap();
  }
  let mut next = 0;
  for _ in 0..200 {
    let k = rand.next_uint_n(7);
    for _ in 0..k {
      assert!(q.force_push(next));
      next += 1;
    }
    if rand.next_uint_n(2) == 0 {
      pool.run_until_stalled();
    }
  }
  pool.run_until_stalled();
  q.set_read_only();
  pool.run_until_stalled();
  let flat: Vec<i32> = got.borrow().iter().flatten().copied().collect();
  assert_eq!(flat, (0..next).collect::<Vec<_>>(), "FIFO order / no loss / no duplication");
}

/// Read-only with elements left: drains them, then returns false; read-only on an empty queue
/// wakes every waiter with false.
#[test]
fn queue_read_only_drains_then_false() {
  let mut pool = LocalPool::new();
  let q: Rc<AsyncQueue<i32>> = Rc::new(AsyncQueue::new());
  for i in 0..5 {
    q.force_push(i);
  }
  q.set_read_only();
  let res: Rc<RefCell<Vec<(bool, Vec<i32>)>>> = Rc::new(RefCell::new(Vec::new()));
  for _ in 0..4 {
    let (q, res) = (q.clone(), res.clone());
    LocalSpawnExt::spawn_local(&pool.spawner(), async move {
        let mut buf = Vec::new();
        let r = q.wait_pop_up_to_n(&mut buf, 2).await;
        res.borrow_mut().push((r, buf));
      })
      .unwrap();
  }
  pool.run_until_stalled();
  let r = res.borrow().clone();
  assert_eq!(r, vec![(true, vec![0, 1]), (true, vec![2, 3]), (true, vec![4]), (false, vec![])]);
}

/// Several waiters on one queue: every push into an empty queue must wake a waiter (no lost
/// wakeup), even if the first woken one re-parks without popping.
#[test]
fn queue_many_waiters_no_lost_wakeup() {
  let mut pool = LocalPool::new();
  let q: Rc<AsyncQueue<i32>> = Rc::new(AsyncQueue::new());
  let got = Rc::new(Cell::new(0));
  for _ in 0..3 {
    let (q, got) = (q.clone(), got.clone());
    LocalSpawnExt::spawn_local(&pool.spawner(), async move {
        loop {
          let mut buf = Vec::new();
          if !q.wait_pop_up_to_n(&mut buf, 1).await {
            break;
          }
          got.set(got.get() + buf.len());
          yield_now().await;
        }
      })
      .unwrap();
  }
  pool.run_until_stalled();
  for round in 0..50 {
    for i in 0..(round % 4 + 1) {
      q.force_push(i);
    }
    pool.run_until_stalled();
  }
  let total: i32 = (0..50).map(|r| r % 4 + 1).sum();
  assert_eq!(got.get() as i32, total);
  assert_eq!(q.size(), 0);
  q.close();
  pool.run_until_stalled();
}

/// WakerList: waiters registered by different tasks, notify between condition flips.
#[test]
fn waker_list_stress() {
  let mut pool = LocalPool::new();
  let cv = Rc::new(WakerList::new());
  let counter = Rc::new(Cell::new(0));
  let done = Rc::new(Cell::new(0));
  for t in 0..20 {
    let (cv, counter, done) = (cv.clone(), counter.clone(), done.clone());
    LocalSpawnExt::spawn_local(&pool.spawner(), async move {
        for k in 0..10 {
          let target = t + k * 20;
          cv.wait_until(|| counter.get() >= target).await;
        }
        done.set(done.get() + 1);
      })
      .unwrap();
  }
  for _ in 0..400 {
    counter.set(counter.get() + 1);
    cv.notify_all();
    pool.run_until_stalled();
  }
  assert_eq!(done.get(), 20);
}

/// setReadOnly then unsetReadOnly before a parked waiter is polled: the waiter never observes
/// read-only. ThreadSafeQueue has the same property; upstream relies on join() in between, which is
/// why spawnServerThreads awaits the old serve tasks' end before unsetReadOnly (D8).
#[test]
fn queue_read_only_then_unset_is_missed() {
  let mut pool = LocalPool::new();
  let q: Rc<AsyncQueue<i32>> = Rc::new(AsyncQueue::new());
  let ended = Rc::new(Cell::new(false));
  {
    let (q, ended) = (q.clone(), ended.clone());
    LocalSpawnExt::spawn_local(&pool.spawner(), async move {
        let mut buf = Vec::new();
        while q.wait_pop_up_to_n(&mut buf, 8).await {
          buf.clear();
        }
        ended.set(true);
      })
      .unwrap();
  }
  pool.run_until_stalled();
  q.set_read_only();
  q.unset_read_only();
  pool.run_until_stalled();
  assert!(!ended.get());
  q.close();
  pool.run_until_stalled();
  assert!(ended.get());
}

/// The watchdog itself: a future nobody wakes fails instead of hanging.
#[test]
#[should_panic(expected = "lost wakeup")]
fn watchdog_catches_a_lost_wakeup() {
  let mut pool = LocalPool::new();
  let cv = WakerList::new();
  run_or_stall_panic(&mut pool, "never notified", cv.wait_until(|| false));
}
