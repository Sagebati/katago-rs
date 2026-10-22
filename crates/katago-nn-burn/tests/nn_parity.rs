//! NN tier (M2): the ported `NNEvaluator` (katago-core) over the burn backend (`BurnNeuralNet`)
//! vs upstream's NNEvaluator over the Eigen fp32 backend, on the positions of
//! fixtures/upstream-v1.16.5/nn (oracle: fixtures/upstream-v1.16.5/nn/oracle/nneval_eigen.cpp,
//! cross-checked against `katago gtp` + `kata-raw-nn`). Not a port of upstream code.
//!
//! Every eval of a net is issued from 8 virtual threads at once, so the server batches them
//! (maxBatchSize 16). Checked per eval: nnHash bit-exact (NNInputs::getHash), policyOptimismUsed
//! exact, the set of illegal moves (policy -1) exact, off-board ownership exactly 0, and every
//! float field within the tolerances below.
//!
//! `nn_parity_b6c96_rand` replays `evals_rand.txt` (the oracle's `evalrand` mode) on one evaluator,
//! one eval at a time: the serve task's symmetry choice under doRandomize / defaultSymmetry (the
//! chosen symmetry must equal upstream's exactly), killServerThreads -> spawnServerThreads (the new
//! server's Rand stream), averageMultipleSymmetries, and passing hacks where the 95% pass cap binds.
//! Pools are driven by `run_watchdog`, so a lost wakeup fails instead of hanging.
#![cfg(feature = "flex")]

use std::cell::RefCell;
use std::future::Future;
use std::rc::Rc;
use std::time::Duration;

use burn::prelude::*;
use futures::executor::LocalPool;

use katago_core::core::commontypes::enabled_t;
use katago_core::core::hash::Hash128;
use katago_core::game::board::Board;
use katago_core::game::board::Location;
use katago_core::game::boardhistory::BoardHistory;
use katago_core::game::rules::Rules;
use katago_core::neuralnet::nneval::NNEvaluator;
use katago_core::neuralnet::nneval::NNResultBuf;
use katago_core::neuralnet::nninputs::MiscNNInputParams;
use katago_core::neuralnet::nninputs::NNOutput;
use katago_core::core::rand::Rand;
use katago_core::neuralnet::nninputs::ScoreValue;
use katago_core::port::executor::Spawner;
use katago_nn_burn::Network;
use katago_nn_burn::backend::BurnNeuralNet;

const WORDS_PER_RECORD: usize = 4 + 10 + 362 + 361;

/// Drives `fut` on `pool` like `LocalPool::run_until`, but never blocks forever: after each poll it
/// runs the other ready tasks (`run_until_stalled`); if `fut` is then pending and not woken, it
/// waits up to `grace` for a wakeup from outside the pool (a GPU readback callback) and panics
/// after that: a lost wakeup fails the test. flex completes everything inside the pool (grace 0).
fn run_watchdog<F: Future>(pool: &mut LocalPool, label: &str, grace: Duration, fut: F) -> F::Output {
  use std::sync::Arc;
  use std::sync::atomic::AtomicBool;
  use std::sync::atomic::Ordering;
  use std::task::Context;
  use std::task::Poll;
  struct Flag(AtomicBool);
  impl futures::task::ArcWake for Flag {
    fn wake_by_ref(arc_self: &Arc<Self>) {
      arc_self.0.store(true, Ordering::SeqCst);
    }
  }
  let flag = Arc::new(Flag(AtomicBool::new(true)));
  let waker = futures::task::waker(flag.clone());
  let mut cx = Context::from_waker(&waker);
  let mut fut = std::pin::pin!(fut);
  loop {
    flag.0.store(false, Ordering::SeqCst);
    if let Poll::Ready(v) = fut.as_mut().poll(&mut cx) {
      return v;
    }
    pool.run_until_stalled();
    let start = std::time::Instant::now();
    while !flag.0.load(Ordering::SeqCst) {
      if start.elapsed() >= grace {
        panic!("{label}: the executor stalled with the future pending and not woken for {grace:?} (lost wakeup)");
      }
      std::thread::sleep(Duration::from_millis(1));
      pool.run_until_stalled();
    }
  }
}

/// No outside wakeups on flex; wgpu readbacks may complete from the device's own thread.
fn grace_for(backend_name: &str) -> Duration {
  if backend_name == "flex" { Duration::ZERO } else { Duration::from_secs(120) }
}

fn fixture_path(name: &str) -> String {
  format!("{}/../../fixtures/upstream-v1.16.5/nn/{}", env!("CARGO_MANIFEST_DIR"), name)
}

fn hex_u64(s: &str) -> u64 {
  u64::from_str_radix(s, 16).unwrap()
}
fn hex_u32(s: &str) -> u32 {
  u32::from_str_radix(s, 16).unwrap()
}

/// A `pos` line of the shared position format (see the oracle's header).
struct Pos {
  x_size: i32,
  y_size: i32,
  rules: Rules,
  encore_phase: i32,
  initial_pla: i8,
  assume_multi_handicap: bool,
  setup: Vec<(i32, i32, i8)>,
  moves: Vec<(i32, i32, i8, bool)>,
}

fn parse_positions() -> Vec<Pos> {
  let text = std::fs::read_to_string(fixture_path("positions.txt")).unwrap();
  let mut out = Vec::new();
  for (i, line) in text.lines().filter(|l| !l.is_empty()).enumerate() {
    let t: Vec<&str> = line.split_whitespace().collect();
    assert_eq!(t[0], "pos");
    assert_eq!(t[1].parse::<usize>().unwrap(), i, "positions are numbered 0..");
    let n = |k: usize| -> i32 { t[k].parse().unwrap() };
    let rules = Rules::new_full(n(4), n(5), n(6), n(7) != 0, n(8) != 0, n(9), n(10) != 0, f32::from_bits(hex_u32(t[11])));
    let mut k = 15;
    let n_setup = n(k) as usize;
    k += 1;
    let mut setup = Vec::new();
    for _ in 0..n_setup {
      setup.push((n(k), n(k + 1), n(k + 2) as i8));
      k += 3;
    }
    let n_moves = n(k) as usize;
    k += 1;
    let mut moves = Vec::new();
    for _ in 0..n_moves {
      moves.push((n(k), n(k + 1), n(k + 2) as i8, n(k + 3) != 0));
      k += 4;
    }
    assert_eq!(k, t.len());
    out.push(Pos {
      x_size: n(2),
      y_size: n(3),
      rules,
      encore_phase: n(12),
      initial_pla: n(13) as i8,
      assume_multi_handicap: n(14) != 0,
      setup,
      moves,
    });
  }
  out
}

fn replay(p: &Pos) -> (Board, BoardHistory) {
  let mut board = Board::new_x_y(p.x_size, p.y_size).unwrap();
  for &(x, y, c) in &p.setup {
    assert!(board.set_stone(Location::get_loc(x, y, p.x_size), c));
  }
  let mut hist = BoardHistory::new_full(&board, p.initial_pla, &p.rules, p.encore_phase);
  hist.set_assume_multiple_starting_black_moves_are_handicap(p.assume_multi_handicap);
  for &(x, y, pla, prevent_encore) in &p.moves {
    let loc = if x == -1 { Board::PASS_LOC } else { Location::get_loc(x, y, p.x_size) };
    assert!(hist.is_legal(&board, loc, pla));
    hist.make_board_move_assume_legal(&mut board, loc, pla, None, prevent_encore);
  }
  (board, hist)
}

/// An `eval` line.
struct Eval {
  pos_id: usize,
  next_pla: i8,
  params: MiscNNInputParams,
}

fn parse_evals(file: &str) -> Vec<Eval> {
  let text = std::fs::read_to_string(fixture_path(file)).unwrap();
  let mut out = Vec::new();
  for (i, line) in text.lines().filter(|l| !l.is_empty()).enumerate() {
    let t: Vec<&str> = line.split_whitespace().collect();
    assert_eq!(t[0], "eval");
    assert_eq!(t[2].parse::<usize>().unwrap(), i);
    out.push(parse_eval_tokens(&t));
  }
  out
}

/// The 15 tokens of an `eval` line (the first may be any tag).
fn parse_eval_tokens(t: &[&str]) -> Eval {
  {
    let mut params = MiscNNInputParams::new();
    params.draw_equivalent_wins_for_white = f64::from_bits(hex_u64(t[4]));
    params.conservative_pass_and_is_root = t[5] == "1";
    params.enable_passing_hacks = t[6] == "1";
    params.playout_doubling_advantage = f64::from_bits(hex_u64(t[7]));
    params.nn_policy_temperature = f32::from_bits(hex_u32(t[8]));
    params.avoid_myt_dagger_hack = t[9] == "1";
    params.symmetry = t[10].parse().unwrap();
    params.policy_optimism = f64::from_bits(hex_u64(t[11]));
    params.max_history = t[12].parse().unwrap();
    assert_eq!((t[13], t[14]), ("19", "19"));
    Eval { pos_id: t[1].parse().unwrap(), next_pla: t[3].parse().unwrap(), params }
  }
}

fn read_records(file: &str) -> Vec<Vec<u32>> {
  let bytes = std::fs::read(fixture_path(file)).unwrap();
  assert_eq!(bytes.len() % (4 * WORDS_PER_RECORD), 0);
  let words: Vec<u32> = bytes.chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect();
  words.chunks_exact(WORDS_PER_RECORD).map(|c| c.to_vec()).collect()
}

const FIELDS: [&str; 11] = [
  "whiteWinProb",
  "whiteLossProb",
  "whiteNoResultProb",
  "whiteScoreMean",
  "whiteScoreMeanSq",
  "whiteLead",
  "varTimeLeft",
  "shorttermWinlossError",
  "shorttermScoreError",
  "policy",
  "ownership",
];

/// Max |burn - eigen| per field, and the max |eigen| per field (for relative reporting).
struct Diffs {
  max: [f64; 11],
  scale: [f64; 11],
}

impl Diffs {
  fn add(&mut self, field: usize, got: f32, want: f32) {
    let d = (got as f64 - want as f64).abs();
    assert!(d.is_finite(), "{}: got {} want {}", FIELDS[field], got, want);
    self.max[field] = self.max[field].max(d);
    self.scale[field] = self.scale[field].max((want as f64).abs());
  }
}

fn compare(eval_idx: usize, out: &NNOutput, rec: &[u32], diffs: &mut Diffs) {
  let want_hash = Hash128::new(rec[0] as u64 | ((rec[1] as u64) << 32), rec[2] as u64 | ((rec[3] as u64) << 32));
  assert!(out.nn_hash == want_hash, "eval {}: nnHash {} != oracle {}", eval_idx, out.nn_hash, want_hash);
  let f = |i: usize| f32::from_bits(rec[4 + i]);
  let got = [
    out.white_win_prob,
    out.white_loss_prob,
    out.white_no_result_prob,
    out.white_score_mean,
    out.white_score_mean_sq,
    out.white_lead,
    out.var_time_left,
    out.shortterm_winloss_error,
    out.shortterm_score_error,
  ];
  for (i, g) in got.iter().enumerate() {
    diffs.add(i, *g, f(i));
  }
  assert_eq!(out.policy_optimism_used.to_bits(), f(9).to_bits(), "eval {}: policyOptimismUsed", eval_idx);
  assert_eq!((out.nn_x_len, out.nn_y_len), (19, 19));
  for pos in 0..362 {
    let want = f32::from_bits(rec[14 + pos]);
    let g = out.policy_probs[pos];
    if want < 0.0 || g < 0.0 {
      assert!(want == -1.0 && g == -1.0, "eval {}: policy[{}] legality: got {} want {}", eval_idx, pos, g, want);
    } else {
      diffs.add(9, g, want);
    }
  }
  let own = out.white_owner_map.as_ref().expect("includeOwnerMap");
  for pos in 0..361 {
    let want = f32::from_bits(rec[14 + 362 + pos]);
    if want == 0.0 {
      assert!(own[pos] == 0.0 || (own[pos] as f64).abs() < 1e-4, "eval {}: ownership[{}] got {} want 0", eval_idx, pos, own[pos]);
    }
    diffs.add(10, own[pos], want);
  }
}

/// Runs every eval of `evals_file` through the ported NNEvaluator on the burn flex backend and
/// compares with `out_file`. `limit`: evaluate only the first `limit` evals.
fn run_net<N: Network + 'static>(backend_name: &str, bpk: &str, evals_file: &str, out_file: &str, limit: usize) -> Diffs {
  ScoreValue::init_tables();
  Board::init_hash();
  let positions = parse_positions();
  let mut evals = parse_evals(evals_file);
  let records = read_records(out_file);
  assert_eq!(evals.len(), records.len());
  evals.truncate(limit);
  assert!(!evals.is_empty());

  let device: Device = match backend_name {
    "flex" => burn::backend::FlexDevice::default().into(),
    #[cfg(feature = "wgpu")]
    "wgpu" => burn::backend::WgpuDevice::default().into(),
    other => panic!("backend {other} not compiled in"),
  };
  let mut pool = LocalPool::new();
  let spawner: Rc<dyn Spawner> = Rc::new(pool.spawner());
  let backend = Rc::new(BurnNeuralNet::<N>::new(device));
  let nn_eval = Rc::new(
    NNEvaluator::new(
      "parity",
      N::NAME,
      "",
      16,
      19,
      19,
      false,
      false,
      -1,
      0,
      false,
      "",
      "",
      false,
      enabled_t::False,
      enabled_t::Auto,
      1,
      &[-1],
      "paritySeed",
      false,
      0,
      backend,
      spawner,
      std::fs::read(bpk).unwrap(),
    )
    .unwrap(),
  );
  let grace = grace_for(backend_name);
  run_watchdog(&mut pool, "spawnServerThreads", grace, nn_eval.spawn_server_threads()).unwrap();

  const NUM_VIRTUAL_THREADS: usize = 8;
  let results: RefCell<Vec<Option<Rc<NNOutput>>>> = RefCell::new(vec![None; evals.len()]);
  let futs = (0..NUM_VIRTUAL_THREADS).map(|t| {
    let (nn_eval, evals, positions, results) = (&nn_eval, &evals, &positions, &results);
    async move {
      let buf = Rc::new(NNResultBuf::new());
      let mut i = t;
      while i < evals.len() {
        let e = &evals[i];
        let (board, hist) = replay(&positions[e.pos_id]);
        assert_eq!(e.next_pla, hist.presumed_next_move_pla);
        nn_eval.evaluate(&board, &hist, e.next_pla, &e.params, &buf, true, true).await.unwrap();
        let r = buf.result.borrow_mut().take().expect("result");
        results.borrow_mut()[i] = Some(r);
        i += NUM_VIRTUAL_THREADS;
      }
    }
  });
  run_watchdog(&mut pool, "evals", grace, futures::future::join_all(futs));
  nn_eval.kill_server_threads();
  pool.run_until_stalled();
  assert!(nn_eval.num_batches_processed() > 0);
  let avg_batch = nn_eval.average_processed_batch_size();
  assert!(avg_batch > 1.5, "evals were not batched (average batch {avg_batch})");

  let mut diffs = Diffs { max: [0.0; 11], scale: [0.0; 11] };
  for (i, r) in results.borrow().iter().enumerate() {
    compare(i, r.as_ref().unwrap(), &records[i], &mut diffs);
  }
  eprintln!("{} on {}: {} evals, average batch {:.2}, max |burn - eigen|:", N::NAME, backend_name, evals.len(), avg_batch);
  for (i, name) in FIELDS.iter().enumerate() {
    eprintln!("  {:<22} {:.3e}   (max |eigen| {:.3})", name, diffs.max[i], diffs.scale[i]);
  }
  diffs
}

/// Tolerances (PORTING.md §13: flex <= 1e-5). Probabilities and policy: 1e-5 absolute.
/// Ownership: 2e-5 absolute. It is tanh of a raw output that burn's conv (im2col/gemm order)
/// and Eigen's accumulate in different orders; over the 512 b6c96 evals the worst point is
/// 1.15e-5 (b10c128 8.2e-6, b18c384nbt 8.9e-6), with no systematic error (nnHash, legality and
/// off-board zeros are exact).
/// Score-like outputs are in points (scoreMeanSq in points^2): the raw score head agrees with
/// Eigen to ~1e-6 of a logit (docs/M0-report.md) and the post-processing multiplies by up to
/// scoreMeanMultiplier = 20 (shorttermScoreError: 30..150), so they get 1e-5 relative to the
/// largest magnitude in the fixture, floored at 1e-5.
fn check(diffs: &Diffs) {
  for (i, name) in FIELDS.iter().enumerate() {
    let tol = match i {
      0..=2 | 9 => 1e-5,
      10 => 2e-5,
      _ => (1e-5 * diffs.scale[i]).max(1e-5),
    };
    assert!(diffs.max[i] <= tol, "{}: max diff {:.3e} > tolerance {:.3e}", name, diffs.max[i], tol);
  }
}


#[cfg(feature = "net-b6c96")]
#[test]
fn nn_parity_b6c96_rand() {
  type N = katago_nn_burn::b6c96::Model;
  ScoreValue::init_tables();
  Board::init_hash();
  let positions = parse_positions();
  let text = std::fs::read_to_string(fixture_path("evals_rand.txt")).unwrap();
  let bytes = std::fs::read(fixture_path("g170-b6c96.rand.out.bin")).unwrap();
  assert_eq!(bytes.len() % (4 * (1 + WORDS_PER_RECORD)), 0);
  let words: Vec<u32> = bytes.chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap())).collect();
  let records: Vec<&[u32]> = words.chunks_exact(1 + WORDS_PER_RECORD).collect();

  let mut lines = text.lines().filter(|l| !l.is_empty());
  let cfg: Vec<&str> = lines.next().unwrap().split_whitespace().collect();
  assert_eq!(cfg[0], "config");
  let device: Device = burn::backend::FlexDevice::default().into();
  let mut pool = LocalPool::new();
  let spawner: Rc<dyn Spawner> = Rc::new(pool.spawner());
  let backend = Rc::new(BurnNeuralNet::<N>::new(device));
  let nn_eval = Rc::new(
    NNEvaluator::new(
      "oracle",
      N::NAME,
      "",
      16,
      19,
      19,
      false,
      false,
      -1,
      0,
      false,
      "",
      "",
      false,
      enabled_t::False,
      enabled_t::Auto,
      1,
      &[-1],
      cfg[1],
      cfg[2] == "1",
      cfg[3].parse().unwrap(),
      backend,
      spawner,
      std::fs::read(katago_nn_burn::bpk_path!("g170-b6c96")).unwrap(),
    )
    .unwrap(),
  );
  let grace = Duration::ZERO;
  run_watchdog(&mut pool, "spawnServerThreads", grace, nn_eval.spawn_server_threads()).unwrap();
  let buf = Rc::new(NNResultBuf::new());
  let mut diffs = Diffs { max: [0.0; 11], scale: [0.0; 11] };
  let (mut n, mut num_avg, mut num_respawn, mut num_pass_capped, mut num_drawn) = (0usize, 0, 0, 0, 0);
  for line in lines {
    let t: Vec<&str> = line.split_whitespace().collect();
    match t[0] {
      "setrandomize" => nn_eval.set_do_randomize(t[1] == "1"),
      "setdefaultsymmetry" => nn_eval.set_default_symmetry(t[1].parse().unwrap()),
      "respawn" => {
        nn_eval.kill_server_threads();
        run_watchdog(&mut pool, "respawn", grace, nn_eval.spawn_server_threads()).unwrap();
        num_respawn += 1;
      }
      "eval" | "avg" => {
        let e = parse_eval_tokens(&t[..15]);
        assert_eq!(t[2].parse::<usize>().unwrap(), n);
        let (board, hist) = replay(&positions[e.pos_id]);
        assert_eq!(e.next_pla, hist.presumed_next_move_pla);
        let out: Rc<NNOutput> = if t[0] == "eval" {
          assert_eq!(t.len(), 15);
          run_watchdog(&mut pool, line, grace, nn_eval.evaluate(&board, &hist, e.next_pla, &e.params, &buf, true, true)).unwrap();
          buf.result.borrow().clone().unwrap()
        } else {
          assert_eq!(t.len(), 17);
          let mut rand = Rand::new_seed(t[16]);
          num_avg += 1;
          run_watchdog(
            &mut pool,
            line,
            grace,
            nn_eval.average_multiple_symmetries(&board, &hist, e.next_pla, None, &e.params, &buf, true, &mut rand, t[15].parse().unwrap()),
          )
          .unwrap()
        };
        let rec = records[n];
        assert_eq!(buf.symmetry.get(), rec[0] as i32, "eval {n} ({line}): symmetry used");
        if e.params.symmetry < 0 {
          num_drawn += 1;
        }
        compare(n, &out, &rec[1..], &mut diffs);
        let want_pass = f32::from_bits(rec[1 + 14 + 361]);
        if e.params.enable_passing_hacks && (want_pass - 0.95).abs() < 1e-4 {
          // The cap binds: pass = 19 x (sum of the other moves), i.e. 0.95 after normalization.
          num_pass_capped += 1;
          assert!((out.policy_probs[361] - 0.95).abs() < 1e-5, "eval {n}: pass {} with the cap binding", out.policy_probs[361]);
        }
        n += 1;
      }
      other => panic!("unknown op {other}"),
    }
  }
  nn_eval.kill_server_threads();
  pool.run_until_stalled();
  assert_eq!(n, records.len());
  assert!(num_avg > 0 && num_respawn >= 2 && num_pass_capped > 0 && num_drawn > 0);
  eprintln!(
    "g170-b6c96 rand on flex: {n} evals ({num_drawn} with the symmetry picked by the server, {num_avg} averageMultipleSymmetries, {num_respawn} respawns, {num_pass_capped} with the pass cap binding), max |burn - eigen|:"
  );
  for (i, name) in FIELDS.iter().enumerate() {
    eprintln!("  {:<22} {:.3e}   (max |eigen| {:.3})", name, diffs.max[i], diffs.scale[i]);
  }
  check(&diffs);
}

#[cfg(feature = "net-b6c96")]
#[test]
fn nn_parity_b6c96() {
  let d = run_net::<katago_nn_burn::b6c96::Model>("flex", katago_nn_burn::bpk_path!("g170-b6c96"), "evals.txt", "g170-b6c96.out.bin", usize::MAX);
  check(&d);
}

#[cfg(feature = "net-b10c128")]
#[test]
fn nn_parity_b10c128() {
  let d = run_net::<katago_nn_burn::b10c128::Model>("flex", katago_nn_burn::bpk_path!("g170e-b10c128"), "evals.txt", "g170e-b10c128.out.bin", usize::MAX);
  check(&d);
}

#[cfg(feature = "net-b18c384nbt")]
#[test]
fn nn_parity_b18c384nbt() {
  let d = run_net::<katago_nn_burn::b18c384nbt::Model>("flex", katago_nn_burn::bpk_path!("kata1-b18c384nbt"), "evals_b18.txt", "kata1-b18c384nbt.out.bin", usize::MAX);
  check(&d);
}

/// wgpu (native Vulkan/Metal): PORTING.md §13 / CLAUDE.md: 2e-4 on logits, 1e-4 on win rate and
/// ownership. After post-processing that is 1e-4 absolute on probabilities, policy and
/// ownership, and 1e-4 relative (floor 1e-4) on the score-like outputs.
#[cfg(feature = "wgpu")]
fn check_wgpu(diffs: &Diffs) {
  for (i, name) in FIELDS.iter().enumerate() {
    let tol = match i {
      0..=2 | 9 | 10 => 1e-4,
      _ => (1e-4 * diffs.scale[i]).max(1e-4),
    };
    assert!(diffs.max[i] <= tol, "wgpu {}: max diff {:.3e} > tolerance {:.3e}", name, diffs.max[i], tol);
  }
}

#[cfg(all(feature = "wgpu", feature = "net-b10c128"))]
#[test]
fn nn_parity_b10c128_wgpu() {
  let d = run_net::<katago_nn_burn::b10c128::Model>("wgpu", katago_nn_burn::bpk_path!("g170e-b10c128"), "evals.txt", "g170e-b10c128.out.bin", usize::MAX);
  check_wgpu(&d);
}

#[cfg(all(feature = "wgpu", feature = "net-b18c384nbt"))]
#[test]
fn nn_parity_b18c384nbt_wgpu() {
  let d = run_net::<katago_nn_burn::b18c384nbt::Model>("wgpu", katago_nn_burn::bpk_path!("kata1-b18c384nbt"), "evals_b18.txt", "kata1-b18c384nbt.out.bin", usize::MAX);
  check_wgpu(&d);
}

/// The layouts the generated graph cannot take are rejected like eigenbackend rejects its own
/// (StringError at context / handle creation), not silently mis-evaluated.
#[cfg(feature = "net-b6c96")]
#[test]
fn backend_rejects_unsupported_layouts() {
  use katago_core::neuralnet::nninterface::NeuralNet;
  type N = katago_nn_burn::b6c96::Model;
  let device: Device = burn::backend::FlexDevice::default().into();
  let backend = BurnNeuralNet::<N>::new(device);
  let model = backend.load_model_file("b6", std::fs::read(katago_nn_burn::bpk_path!("g170-b6c96")).unwrap(), "").unwrap();
  let desc = backend.get_model_desc(&*model);
  assert_eq!((desc.model_version, desc.num_policy_channels, desc.num_score_value_channels), (8, 1, 4));
  assert!(backend.load_model_file("b6", Vec::new(), "abc").is_err());
  let ctx = |x: i32, y: i32, fp16: enabled_t, nhwc: enabled_t| backend.create_compute_context(&[-1], x, y, "", "", false, fp16, nhwc, &*model).is_ok();
  assert!(ctx(19, 19, enabled_t::False, enabled_t::Auto));
  assert!(ctx(19, 19, enabled_t::Auto, enabled_t::False));
  assert!(!ctx(9, 9, enabled_t::False, enabled_t::Auto));
  assert!(!ctx(19, 19, enabled_t::True, enabled_t::Auto));
  assert!(!ctx(19, 19, enabled_t::False, enabled_t::True));
  let context = backend.create_compute_context(&[-1], 19, 19, "", "", false, enabled_t::False, enabled_t::Auto, &*model).unwrap();
  assert!(backend.create_compute_handle(&*context, &*model, 4, false, true, -1, 0).is_err());
  let handle = backend.create_compute_handle(&*context, &*model, 4, false, false, -1, 0).unwrap();
  assert!(!backend.is_using_fp16(&*handle));
}
