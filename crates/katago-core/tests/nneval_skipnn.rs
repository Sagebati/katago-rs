//! NNEvaluator (neuralnet/nneval.rs) against the unmodified upstream NNEvaluator with
//! debugSkipNeuralNet = true: fixtures/upstream-v1.18.2/unit/oracle/nneval_skipnn.cpp prints an op
//! stream (E/P/V/A) and the results (R/D/S lines); this test replays every op through the port on a
//! futures LocalPool and must print the same bytes. This checks evaluate()'s post-processing
//! (expf/exp/ln/tanhf, temperature, passing hacks, dagger ban, masking), the NN cache incl. the
//! hadResultWithoutOwnerMap path, averageMultipleSymmetries, the serve task's Rand stream, and
//! killServerThreads -> spawnServerThreads on one evaluator (K: the new server thread's Rand; a
//! surviving old serve task would draw from the old stream, D8). Every pool run goes through
//! `common::run_or_stall_panic`, so a lost wakeup fails instead of hanging.
//! Not a port of upstream code. Also: a Rust-only batching test with a fake backend.

#[macro_use]
mod common;

use std::cell::Cell;
use std::cell::RefCell;
use std::rc::Rc;

use common::Cases;
use common::compare_lines;
use common::fixture;
use common::init;
use common::run_or_stall_panic;
use futures::executor::LocalPool;
use katago_core::neuralnet::nninterface::BatchPolicy;
use katago_core::core::config_parser::ConfigParser;
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
use katago_core::neuralnet::nninputs::NNPos;
use katago_core::neuralnet::nninterface::*;
use katago_core::port::executor::LocalBoxFuture;
use katago_core::port::executor::Spawner;
use katago_core::port::executor::yield_now;

fn fnv_floats(a: &[f32]) -> u64 {
  let mut h: u64 = 0xcbf29ce484222325;
  for x in a {
    let u = x.to_bits();
    for b in 0..4 {
      h ^= ((u >> (8 * b)) & 0xFF) as u64;
      h = h.wrapping_mul(0x100000001b3);
    }
  }
  h
}

fn result_lines(o: &NNOutput, policy_size: i32, full: bool, out: &mut String) {
  let owner_fnv = match &o.white_owner_map {
    Some(m) => fnv_floats(&m[..(o.nn_x_len * o.nn_y_len) as usize]),
    None => 0,
  };
  out.push_str(&format!(
    "R {:016X} {:016X} {:08X} {:08X} {:08X} {:08X} {:08X} {:08X} {:08X} {:08X} {:08X} {:08X} {} {} {:016X} {} {:016X}\n",
    o.nn_hash.hash0,
    o.nn_hash.hash1,
    o.white_win_prob.to_bits(),
    o.white_loss_prob.to_bits(),
    o.white_no_result_prob.to_bits(),
    o.white_score_mean.to_bits(),
    o.white_score_mean_sq.to_bits(),
    o.white_lead.to_bits(),
    o.var_time_left.to_bits(),
    o.shortterm_winloss_error.to_bits(),
    o.shortterm_score_error.to_bits(),
    o.policy_optimism_used.to_bits(),
    o.nn_x_len,
    o.nn_y_len,
    fnv_floats(&o.policy_probs[..]),
    if o.white_owner_map.is_some() { 1 } else { 0 },
    owner_fnv
  ));
  if full {
    out.push('D');
    for i in 0..policy_size {
      out.push_str(&format!(" {:08X}", o.policy_probs[i as usize].to_bits()));
    }
    out.push_str(" |");
    if let Some(m) = &o.white_owner_map {
      for i in 0..o.nn_x_len * o.nn_y_len {
        out.push_str(&format!(" {:08X}", m[i as usize].to_bits()));
      }
    }
    out.push('\n');
  }
}

/// A backend that must never be reached (debugSkipNeuralNet).
struct NoBackend;

impl NeuralNet for NoBackend {
  fn global_initialize(&self) {}
  fn global_cleanup(&self) {}
  fn print_devices(&self) {}
  fn get_runtime_backend_detail(&self, _: &mut ConfigParser) -> String {
    String::new()
  }
  fn get_batch_policy(&self, _: &mut ConfigParser) -> BatchPolicy {
    BatchPolicy::Dynamic
  }
  fn get_num_effective_devices(&self, _: &mut ConfigParser, gpu_idx_by_server_thread: &[i32]) -> i32 {
    let distinct_devices: std::collections::BTreeSet<i32> = gpu_idx_by_server_thread.iter().copied().collect();
    std::cmp::max(1, distinct_devices.len() as i32)
  }
  fn set_is_warmup(&self, _: &dyn ComputeHandle, _: bool) -> bool {
    false
  }
  fn load_model_file(&self, _: &str, _: Vec<u8>, _: &str) -> Result<Box<dyn LoadedModel>, KataError> {
    panic!("debugSkipNeuralNet: no model is loaded")
  }
  fn get_model_desc<'a>(&self, _: &'a dyn LoadedModel) -> &'a ModelDesc {
    panic!("debugSkipNeuralNet")
  }
  fn create_compute_context(
    &self,
    _: &[i32], _: i32, _: i32, _: &str, _: enabled_t, _: &dyn LoadedModel, _: &mut ConfigParser,
  ) -> Result<Box<dyn ComputeContext>, KataError> {
    panic!("debugSkipNeuralNet")
  }
  fn create_compute_handle(&self, _: &dyn ComputeContext, _: &dyn LoadedModel, _: i32, _: bool, _: bool, _: i32, _: i32) -> Result<Box<dyn ComputeHandle>, KataError> {
    panic!("debugSkipNeuralNet")
  }
  fn is_using_fp16(&self, _: &dyn ComputeHandle) -> bool {
    panic!("debugSkipNeuralNet")
  }
  fn create_input_buffers(&self, _: &dyn LoadedModel, _: i32, _: i32, _: i32) -> Box<dyn InputBuffers> {
    panic!("debugSkipNeuralNet")
  }
  fn get_output<'a>(
    &'a self,
    _: &'a mut dyn ComputeHandle,
    _: &'a mut dyn InputBuffers,
    _: i32,
    _: &'a [Rc<NNResultBuf>],
    _: &'a mut [NNOutput],
  ) -> LocalBoxFuture<'a, Result<(), KataError>> {
    panic!("debugSkipNeuralNet")
  }
}

fn hex_u32(s: &str) -> u32 {
  u32::from_str_radix(s, 16).unwrap()
}
fn hex_u64(s: &str) -> u64 {
  u64::from_str_radix(s, 16).unwrap()
}

struct Pos {
  board: Board,
  hist: BoardHistory,
  next_pla: Player,
}

fn parse_pos(f: &[&str]) -> Box<Pos> {
  let i = |k: usize| -> i32 { f[k].parse().unwrap() };
  let rules = Rules::new_full(i(3), i(4), i(5), i(6) != 0, i(7) != 0, i(8), i(9) != 0, f32::from_bits(hex_u32(f[10])));
  let mut board = Board::new_x_y(i(1), i(2)).unwrap();
  let mut hist = BoardHistory::new_full(&board, i(11) as Player, &rules, 0, &BoardHistoryModes::new());
  let n = i(12) as usize;
  for m in 0..n {
    let loc = i(13 + 2 * m) as Loc;
    let pla = i(14 + 2 * m) as Player;
    hist.make_board_move_assume_legal(&mut board, loc, pla, None, false);
  }
  let next_pla = hist.presumed_next_move_pla;
  Box::new(Pos { board, hist, next_pla })
}

/// The 11 MiscNNInputParams/flag fields of a V/A line: (params, skipCache, includeOwnerMap).
fn parse_params(f: &[&str]) -> (MiscNNInputParams, bool, bool) {
  let mut p = MiscNNInputParams::new();
  p.draw_equivalent_wins_for_white = f64::from_bits(hex_u64(f[1]));
  p.conservative_pass_and_is_root = f[2] == "1";
  p.enable_passing_hacks = f[3] == "1";
  p.playout_doubling_advantage = f64::from_bits(hex_u64(f[4]));
  p.nn_policy_temperature = f32::from_bits(hex_u32(f[5]));
  p.avoid_myt_dagger_hack = f[6] == "1";
  p.symmetry = f[7].parse().unwrap();
  p.policy_optimism = f64::from_bits(hex_u64(f[8]));
  p.max_history = f[9].parse().unwrap();
  (p, f[10] == "1", f[11] == "1")
}

wasm_test! {
fn nneval_skipnn_oracle() {
  // Separate frames: debug-build Board::initHash and the replay loop (with its Board/BoardHistory
  // temporaries) would not fit together in wasm's 1 MiB stack.
  init();
  run_oracle();
}
}

fn run_oracle() {
  let mut cases = Cases::new("nneval_skipnn::nneval_skipnn_oracle");
  let expected = fixture("oracle", "nneval_skipnn.txt");
  let mut pool = LocalPool::new();
  let spawner: Rc<dyn Spawner> = Rc::new(pool.spawner());
  let backend: Rc<dyn NeuralNet> = Rc::new(NoBackend);

  let mut out = String::new();
  let mut nn_eval: Option<Rc<NNEvaluator>> = None;
  let mut nn_len: i32 = 19;
  let mut pos: Option<Box<Pos>> = None;
  let mut evals_this_cfg = 0;
  let buf: Rc<NNResultBuf> = Rc::new(NNResultBuf::new());
  for line in expected.lines() {
    let f: Vec<&str> = line.split(' ').collect();
    match f[0] {
      "E" => {
        out.push_str(line);
        out.push('\n');
        nn_len = f[1].parse().unwrap();
        let e = NNEvaluator::new(
          "skipnn",
          "skipnn.bin.gz",
          "",
          8,
          nn_len,
          f[2].parse().unwrap(),
          false,
          false,
          f[3].parse().unwrap(),
          3,
          true,
          "",
          enabled_t::False,
          1,
          &[-1],
          f[4],
          f[5] == "1",
          0,
          &mut ConfigParser::new(false, true),
          backend.clone(),
          spawner.clone(),
          Vec::new(),
        )
        .unwrap();
        let e = Rc::new(e);
        run_or_stall_panic(&mut pool, "spawnServerThreads", e.spawn_server_threads()).unwrap();
        nn_eval = Some(e);
        evals_this_cfg = 0;
      }
      "P" => {
        out.push_str(line);
        out.push('\n');
        pos = Some(parse_pos(&f));
      }
      "V" | "W" | "A" => {
        out.push_str(line);
        out.push('\n');
        let e = nn_eval.as_ref().unwrap();
        let p = pos.as_ref().unwrap();
        let (params, skip_cache, include_owner_map) = parse_params(&f);
        let policy_size = NNPos::get_policy_size(nn_len, nn_len);
        if f[0] == "A" {
          let num_sym: i32 = f[12].parse().unwrap();
          let mut rand = Rand::new_seed(f[13]);
          let res = run_or_stall_panic(
            &mut pool,
            line,
            e.average_multiple_symmetries(&p.board, &p.hist, p.next_pla, None, &params, &buf, include_owner_map, &mut rand, num_sym),
          )
          .unwrap();
          result_lines(&res, policy_size, evals_this_cfg < 6, &mut out);
          evals_this_cfg += 1;
        } else {
          run_or_stall_panic(&mut pool, line, e.evaluate(&p.board, &p.hist, p.next_pla, &params, &buf, skip_cache, include_owner_map)).unwrap();
          let r = buf.result.borrow().clone().unwrap();
          let is_reeval = f[0] == "W";
          result_lines(&r, policy_size, !is_reeval && evals_this_cfg < 6, &mut out);
          if !is_reeval {
            evals_this_cfg += 1;
          }
        }
        cases.ran();
      }
      "S" => {
        let e = nn_eval.take().unwrap();
        out.push_str(&format!("S {} {}\n", e.num_rows_processed(), e.num_batches_processed()));
        e.kill_server_threads();
        pool.run_until_stalled();
        assert_eq!(Rc::strong_count(&e), 1, "serve task ended and released the evaluator");
        drop(e);
      }
      "K" => {
        out.push_str(line);
        out.push('\n');
        let e = nn_eval.as_ref().unwrap();
        e.kill_server_threads();
        run_or_stall_panic(&mut pool, "respawn", e.spawn_server_threads()).unwrap();
      }
      "R" | "D" => {}
      other => panic!("unknown op {}", other),
    }
  }
  compare_lines("nneval_skipnn.txt", &out, &expected);
}

//------------------------------------------------------------------------------------------------
// Batching with a fake backend (Rust-only).

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
impl ComputeHandle for Unit {
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

/// Deterministic "network": every output is a function of that row's own input features, so a
/// result routed to the wrong buf is detected.
struct FakeBackend {
  batch_sizes: RefCell<Vec<i32>>,
  calls: Cell<i32>,
}

impl NeuralNet for FakeBackend {
  fn global_initialize(&self) {}
  fn global_cleanup(&self) {}
  fn print_devices(&self) {}
  fn get_runtime_backend_detail(&self, _: &mut ConfigParser) -> String {
    String::new()
  }
  fn get_batch_policy(&self, _: &mut ConfigParser) -> BatchPolicy {
    BatchPolicy::Dynamic
  }
  fn get_num_effective_devices(&self, _: &mut ConfigParser, gpu_idx_by_server_thread: &[i32]) -> i32 {
    let distinct_devices: std::collections::BTreeSet<i32> = gpu_idx_by_server_thread.iter().copied().collect();
    std::cmp::max(1, distinct_devices.len() as i32)
  }
  fn set_is_warmup(&self, _: &dyn ComputeHandle, _: bool) -> bool {
    false
  }
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
    _: &[i32], _: i32, _: i32, _: &str, _: enabled_t, _: &dyn LoadedModel, _: &mut ConfigParser,
  ) -> Result<Box<dyn ComputeContext>, KataError> {
    Ok(Box::new(Unit))
  }
  fn create_compute_handle(&self, _: &dyn ComputeContext, _: &dyn LoadedModel, _: i32, _: bool, _: bool, _: i32, _: i32) -> Result<Box<dyn ComputeHandle>, KataError> {
    Ok(Box::new(Unit))
  }
  fn is_using_fp16(&self, _: &dyn ComputeHandle) -> bool {
    false
  }
  fn create_input_buffers(&self, _: &dyn LoadedModel, _: i32, _: i32, _: i32) -> Box<dyn InputBuffers> {
    Box::new(Unit)
  }
  fn get_output<'a>(
    &'a self,
    _: &'a mut dyn ComputeHandle,
    _: &'a mut dyn InputBuffers,
    n: i32,
    input_bufs: &'a [Rc<NNResultBuf>],
    outputs: &'a mut [NNOutput],
  ) -> LocalBoxFuture<'a, Result<(), KataError>> {
    Box::pin(async move {
      self.batch_sizes.borrow_mut().push(n);
      self.calls.set(self.calls.get() + 1);
      yield_now().await;
      for row in 0..n as usize {
        let spatial = input_bufs[row].row_spatial_buf.borrow();
        let global = input_bufs[row].row_global_buf.borrow();
        let o = &mut outputs[row];
        let stones: f32 = spatial[361..3 * 361].iter().sum();
        for i in 0..362 {
          o.policy_probs[i] = (i % 7) as f32 * 0.1 + stones * 0.01;
        }
        o.white_win_prob = stones * 0.05;
        o.white_loss_prob = global[5];
        o.white_no_result_prob = -2.0;
        o.white_score_mean = stones;
        o.white_score_mean_sq = 0.5;
        o.white_lead = stones * 0.5;
        o.var_time_left = 1.0;
        o.shortterm_winloss_error = 0.0;
        o.shortterm_score_error = 0.0;
        if let Some(m) = &mut o.white_owner_map {
          for (i, x) in m.iter_mut().enumerate() {
            *x = spatial[361 + i] - spatial[2 * 361 + i];
          }
        }
      }
      Ok(())
    })
  }
}

fn fake_eval(backend: Rc<FakeBackend>, spawner: Rc<dyn Spawner>, max_batch: i32) -> Rc<NNEvaluator> {
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
      -1,
      0,
      false,
      "",
      enabled_t::False,
      1,
      &[-1],
      "seed",
      false,
      0,
      &mut ConfigParser::new(false, true),
      backend,
      spawner,
      Vec::new(),
    )
    .unwrap(),
  )
}

fn positions() -> Vec<Pos> {
  let mut v = Vec::new();
  for k in 0..6 {
    let rules = Rules::new_full(Rules::KO_POSITIONAL, Rules::SCORING_AREA, Rules::TAX_NONE, true, false, 0, false, 7.5 - k as f32);
    let mut board = Board::new_x_y(19, 19).unwrap();
    let mut hist = BoardHistory::new_full(&board, P_BLACK, &rules, 0, &BoardHistoryModes::new());
    for m in 0..(2 * k + 1) {
      let loc = Location::get_loc(3 + m, 3 + k, 19);
      let pla = hist.presumed_next_move_pla;
      hist.make_board_move_assume_legal(&mut board, loc, pla, None, false);
    }
    let next_pla = hist.presumed_next_move_pla;
    v.push(Pos { board, hist, next_pla });
  }
  v
}

wasm_test! {
fn nneval_batching_fake_backend() {
  init();
  let mut cases = Cases::new("nneval_skipnn::nneval_batching_fake_backend");
  let positions = positions();
  let params = MiscNNInputParams::new();

  // Reference: one at a time.
  let mut pool = LocalPool::new();
  let spawner: Rc<dyn Spawner> = Rc::new(pool.spawner());
  let single_backend = Rc::new(FakeBackend { batch_sizes: RefCell::new(Vec::new()), calls: Cell::new(0) });
  let single = fake_eval(single_backend.clone(), spawner.clone(), 4);
  run_or_stall_panic(&mut pool, "spawn", single.spawn_server_threads()).unwrap();
  let mut reference: Vec<Rc<NNOutput>> = Vec::new();
  for p in &positions {
    let buf = Rc::new(NNResultBuf::new());
    run_or_stall_panic(&mut pool, "evaluate", single.evaluate(&p.board, &p.hist, p.next_pla, &params, &buf, true, true)).unwrap();
    reference.push(buf.result.borrow().clone().unwrap());
  }
  assert!(single_backend.batch_sizes.borrow().iter().all(|&n| n == 1));

  // Six virtual threads at once with maxBatchSize 4: batches of 4 then 2, same results.
  let batch_backend = Rc::new(FakeBackend { batch_sizes: RefCell::new(Vec::new()), calls: Cell::new(0) });
  let batched = fake_eval(batch_backend.clone(), spawner.clone(), 4);
  run_or_stall_panic(&mut pool, "spawn", batched.spawn_server_threads()).unwrap();
  let bufs: Vec<Rc<NNResultBuf>> = (0..positions.len()).map(|_| Rc::new(NNResultBuf::new())).collect();
  let futs = positions.iter().zip(bufs.iter()).map(|(p, buf)| batched.evaluate(&p.board, &p.hist, p.next_pla, &params, buf, true, true));
  let results = run_or_stall_panic(&mut pool, "batched evals", futures::future::join_all(futs));
  for r in results {
    r.unwrap();
  }
  assert_eq!(*batch_backend.batch_sizes.borrow(), vec![4, 2]);
  assert_eq!(batched.num_batches_processed(), 2);
  assert_eq!(batched.num_rows_processed(), 6);
  for (k, buf) in bufs.iter().enumerate() {
    let got = buf.result.borrow().clone().unwrap();
    let want = &reference[k];
    assert_eq!(got.nn_hash, want.nn_hash, "position {}", k);
    assert_eq!(got.white_win_prob.to_bits(), want.white_win_prob.to_bits(), "position {}", k);
    assert_eq!(got.white_score_mean.to_bits(), want.white_score_mean.to_bits(), "position {}", k);
    assert_eq!(fnv_floats(&got.policy_probs[..]), fnv_floats(&want.policy_probs[..]), "position {}", k);
    assert_eq!(fnv_floats(got.white_owner_map.as_ref().unwrap()), fnv_floats(want.white_owner_map.as_ref().unwrap()), "position {}", k);
    cases.ran();
  }
  // Distinct positions really gave distinct outputs (routing is observable).
  assert!(reference[0].white_win_prob != reference[5].white_win_prob);

  // waitForNextNNEvalIfAny returns immediately with nothing in flight.
  run_or_stall_panic(&mut pool, "waitForNextNNEvalIfAny", batched.wait_for_next_nn_eval_if_any());
  assert_eq!(batched.average_processed_batch_size(), 3.0);
  assert!(!batched.is_any_thread_using_fp16());
  let _ = NNInputs::SYMMETRY_ALL;

  single.kill_server_threads();
  batched.kill_server_threads();
  pool.run_until_stalled();
  assert_eq!(Rc::strong_count(&single), 1);
  assert_eq!(Rc::strong_count(&batched), 1);
}
}

