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
//!
//! `nn_parity_b18c384nbt_humanv0` (feature `net-b18c384nbt-humanv0`): the human-SL net, whose eval lines
//! (`evals_human.txt`) carry a humanSLProfile: SGFMetadata::getProfile -> evaluate's sgfMeta, so the
//! nnHash includes SGFMetadata::getHash and the net gets fillMetadataRow's row as `meta_input`.
#![cfg(feature = "flex")]

use std::cell::RefCell;
use std::rc::Rc;
use std::time::Duration;

use burn::prelude::*;
use futures::executor::LocalPool;

use katago_core::core::commontypes::enabled_t;
use katago_core::core::hash::Hash128;
use katago_core::game::board::Board;
use katago_core::neuralnet::nneval::NNEvaluator;
use katago_core::neuralnet::nneval::NNResultBuf;
use katago_core::neuralnet::nninputs::MiscNNInputParams;
use katago_core::neuralnet::nninputs::NNOutput;
use katago_core::core::rand::Rand;
use katago_core::neuralnet::nninputs::ScoreValue;
use katago_core::neuralnet::sgfmetadata::SGFMetadata;
use katago_core::port::executor::Spawner;
use katago_nn_burn::Network;
use katago_nn_burn::backend::BurnNeuralNet;

mod common_nn;

use common_nn::hex_u32;
use common_nn::hex_u64;
use common_nn::parse_pos_tokens;
use common_nn::replay;
use common_nn::run_watchdog;
use common_nn::Pos;

const WORDS_PER_RECORD: usize = 4 + 10 + 362 + 361;

/// No outside wakeups on flex; wgpu readbacks may complete from the device's own thread.
fn grace_for(backend_name: &str) -> Duration {
  if backend_name == "flex" { Duration::ZERO } else { Duration::from_secs(120) }
}

fn fixture_path(name: &str) -> String {
  format!("{}/../../fixtures/upstream-v1.16.5/nn/{}", env!("CARGO_MANIFEST_DIR"), name)
}

fn parse_positions() -> Vec<Pos> {
  let text = std::fs::read_to_string(fixture_path("positions.txt")).unwrap();
  let mut out = Vec::new();
  for (i, line) in text.lines().filter(|l| !l.is_empty()).enumerate() {
    let t: Vec<&str> = line.split_whitespace().collect();
    assert_eq!(t[0], "pos");
    assert_eq!(t[1].parse::<usize>().unwrap(), i, "positions are numbered 0..");
    out.push(parse_pos_tokens(&t));
  }
  out
}

/// An `eval` line.
struct Eval {
  pos_id: usize,
  next_pla: i8,
  params: MiscNNInputParams,
  /// The optional trailing humanSLProfile token.
  profile: Option<String>,
}

fn parse_evals(file: &str) -> Vec<Eval> {
  let text = std::fs::read_to_string(fixture_path(file)).unwrap();
  let mut out = Vec::new();
  for (i, line) in text.lines().filter(|l| !l.is_empty()).enumerate() {
    let t: Vec<&str> = line.split_whitespace().collect();
    assert_eq!(t[0], "eval");
    assert_eq!(t[2].parse::<usize>().unwrap(), i);
    let mut e = parse_eval_tokens(&t);
    assert!(t.len() == 15 || t.len() == 16, "{line}");
    e.profile = t.get(15).map(|p| p.to_string());
    out.push(e);
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
    Eval { pos_id: t[1].parse().unwrap(), next_pla: t[3].parse().unwrap(), params, profile: None }
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
  run_net_with::<N>(backend_name, bpk, evals_file, out_file, limit, enabled_t::False)
}

/// [`run_net`] with the evaluator's `useFP16` (`enabled_t::True`: this network's 3x3 convs in f16).
fn run_net_with<N: Network + 'static>(backend_name: &str, bpk: &str, evals_file: &str, out_file: &str, limit: usize, use_fp16: enabled_t) -> Diffs {
  ScoreValue::init_tables();
  Board::init_hash();
  let positions = parse_positions();
  let mut evals = parse_evals(evals_file);
  let records = read_records(out_file);
  assert_eq!(evals.len(), records.len());
  evals.truncate(limit);
  assert!(!evals.is_empty());

  let device: Device = katago_nn_burn::device(backend_name).unwrap_or_else(|| panic!("backend {backend_name} not compiled in"));
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
      use_fp16,
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
        let meta: Option<SGFMetadata> = e.profile.as_ref().map(|p| SGFMetadata::get_profile(p).unwrap());
        nn_eval.evaluate_sgf_meta(&board, &hist, e.next_pla, meta.as_ref(), &e.params, &buf, true, true).await.unwrap();
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
      std::fs::read(katago_nn_burn::b6c96::BPK_PATH).unwrap(),
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
  let d = run_net::<katago_nn_burn::b6c96::Model>("flex", katago_nn_burn::b6c96::BPK_PATH, "evals.txt", "g170-b6c96.out.bin", usize::MAX);
  check(&d);
}

#[cfg(feature = "net-b10c128")]
#[test]
fn nn_parity_b10c128() {
  let d = run_net::<katago_nn_burn::b10c128::Model>("flex", katago_nn_burn::b10c128::BPK_PATH, "evals.txt", "g170e-b10c128.out.bin", usize::MAX);
  check(&d);
}

#[cfg(feature = "net-b18c384nbt")]
#[test]
fn nn_parity_b18c384nbt() {
  let d = run_net::<katago_nn_burn::b18c384nbt::Model>("flex", katago_nn_burn::b18c384nbt::BPK_PATH, "evals_b18.txt", "kata1-b18c384nbt.out.bin", usize::MAX);
  check(&d);
}

#[cfg(feature = "net-b18c384nbt-humanv0")]
#[test]
fn nn_parity_b18c384nbt_humanv0() {
  let d = run_net::<katago_nn_burn::b18c384nbt_humanv0::Model>(
    "flex",
    katago_nn_burn::b18c384nbt_humanv0::BPK_PATH,
    "evals_human.txt",
    "b18c384nbt-humanv0.out.bin",
    usize::MAX,
  );
  check(&d);
}

/// wgpu (native Vulkan/Metal): PORTING.md §13 / CLAUDE.md: 2e-4 on logits, 1e-4 on win rate and
/// ownership. After post-processing that is 1e-4 absolute on probabilities, policy and
/// ownership, and 1e-4 relative (floor 1e-4) on the score-like outputs.
#[cfg(feature = "wgpu-fixed-kernels")]
fn check_wgpu(diffs: &Diffs) {
  check_wgpu_with(diffs, 1e-4, 1e-4);
}

/// `abs` on probabilities, policy and ownership; `rel` of the largest magnitude (floor `abs`) on the score-like
/// outputs.
#[cfg(feature = "wgpu-fixed-kernels")]
fn check_wgpu_with(diffs: &Diffs, abs: f64, rel: f64) {
  for (i, name) in FIELDS.iter().enumerate() {
    let tol = match i {
      0..=2 | 9 | 10 => abs,
      _ => (rel * diffs.scale[i]).max(abs),
    };
    assert!(diffs.max[i] <= tol, "wgpu {}: max diff {:.3e} > tolerance {:.3e}", name, diffs.max[i], tol);
  }
}

/// The wgpu tests share the process-wide precision switch of wino.rs: one at a time.
#[cfg(feature = "wgpu-fixed-kernels")]
static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(feature = "wgpu-fixed-kernels")]
fn gpu_lock() -> std::sync::MutexGuard<'static, ()> {
  GPU.lock().unwrap_or_else(|e| e.into_inner())
}

/// wino.rs in f16 for the guard's lifetime.
#[cfg(feature = "wgpu-fixed-kernels")]
struct Half;

#[cfg(feature = "wgpu-fixed-kernels")]
impl Half {
  fn on() -> Self {
    katago_nn_burn::wino::set_half(true);
    Half
  }
}

#[cfg(feature = "wgpu-fixed-kernels")]
impl Drop for Half {
  fn drop(&mut self) {
    katago_nn_burn::wino::set_half(false);
  }
}

#[cfg(all(feature = "wgpu", feature = "net-b10c128"))]
#[test]
fn nn_parity_b10c128_wgpu() {
  let _gpu = gpu_lock();
  let d = run_net::<katago_nn_burn::b10c128::Model>("wgpu", katago_nn_burn::b10c128::BPK_PATH, "evals.txt", "g170e-b10c128.out.bin", usize::MAX);
  check_wgpu(&d);
}

#[cfg(all(feature = "wgpu", feature = "net-b18c384nbt"))]
#[test]
fn nn_parity_b18c384nbt_wgpu() {
  let _gpu = gpu_lock();
  let d = run_net::<katago_nn_burn::b18c384nbt::Model>("wgpu", katago_nn_burn::b18c384nbt::BPK_PATH, "evals_b18.txt", "kata1-b18c384nbt.out.bin", usize::MAX);
  check_wgpu(&d);
}

/// The precision is per network: an f16 evaluator (useFP16 = true) in the process leaves a later f32 one exactly as
/// it was before (it used to switch every network of the process to f16).
#[cfg(all(feature = "wgpu-fixed-kernels", feature = "net-b18c384nbt"))]
#[test]
fn nn_parity_b18c384nbt_precision_is_per_network_wgpu() {
  type N = katago_nn_burn::b18c384nbt::Model;
  let _gpu = gpu_lock();
  let run = |fp16| run_net_with::<N>("wgpu", katago_nn_burn::b18c384nbt::BPK_PATH, "evals_b18.txt", "kata1-b18c384nbt.out.bin", 24, fp16);
  let before = run(enabled_t::False);
  let half = run(enabled_t::True);
  let after = run(enabled_t::False);
  assert_eq!(before.max, after.max, "an f16 network changed a later f32 one");
  assert_ne!(before.max, half.max, "useFP16 = true did not run in f16");
  check_wgpu(&before);
  check_f16(&half);
}

#[cfg(all(feature = "wgpu", feature = "net-b18c384nbt-humanv0"))]
#[test]
fn nn_parity_b18c384nbt_humanv0_wgpu() {
  let _gpu = gpu_lock();
  let d = run_net::<katago_nn_burn::b18c384nbt_humanv0::Model>(
    "wgpu",
    katago_nn_burn::b18c384nbt_humanv0::BPK_PATH,
    "evals_human.txt",
    "b18c384nbt-humanv0.out.bin",
    usize::MAX,
  );
  check_wgpu(&d);
}

/// The 3x3 convs in f16 (wino.rs `set_half`, as the browser runs them on adapters with shader-f16).
#[cfg(all(feature = "wgpu", feature = "net-b18c384nbt"))]
#[test]
fn nn_parity_b18c384nbt_wgpu_f16() {
  let _gpu = gpu_lock();
  let _half = Half::on();
  let d = run_net::<katago_nn_burn::b18c384nbt::Model>("wgpu", katago_nn_burn::b18c384nbt::BPK_PATH, "evals_b18.txt", "kata1-b18c384nbt.out.bin", usize::MAX);
  check_f16(&d);
}

#[cfg(all(feature = "wgpu", feature = "net-b18c384nbt-humanv0"))]
#[test]
fn nn_parity_b18c384nbt_humanv0_wgpu_f16() {
  let _gpu = gpu_lock();
  let _half = Half::on();
  let d = run_net::<katago_nn_burn::b18c384nbt_humanv0::Model>(
    "wgpu",
    katago_nn_burn::b18c384nbt_humanv0::BPK_PATH,
    "evals_human.txt",
    "b18c384nbt-humanv0.out.bin",
    usize::MAX,
  );
  check_f16(&d);
}

/// f16: no worse than upstream's own FP16. `katago testgpuerror` (v1.16.5 OpenCL, FP16 storage + compute, RX 6800
/// XT, kata1-b18c384nbt, 2247 positions) gives these max errors vs its FP32: winrate 4.45%, lead 2.27, scoreMean
/// 2.28, stWLError 4.32c, stScoreError 0.612, ownership 9.23c, top policy 5.04%. They bound the matching fields
/// (policy per move by the top-policy delta); scoreMeanSq and varTimeLeft, which testgpuerror does not report, get
/// 5% of their largest magnitude. Measured here (b18, 72 evals): winrate 2.5e-3, lead 0.21, stScoreError 0.16,
/// policy 9.6e-3, ownership 1.3e-2.
#[cfg(feature = "wgpu-fixed-kernels")]
fn check_f16(diffs: &Diffs) {
  for (i, name) in FIELDS.iter().enumerate() {
    let tol = match i {
      0..=2 => 0.0445,
      3 => 2.28,
      5 => 2.27,
      7 => 0.0432,
      8 => 0.612,
      9 => 0.0504,
      10 => 0.0923,
      _ => 0.05 * diffs.scale[i],
    };
    assert!(diffs.max[i] <= tol, "wgpu f16 {}: max diff {:.3e} > tolerance {:.3e}", name, diffs.max[i], tol);
  }
}

/// ROCm (HIP): the same cubecl kernels as WebGPU, held to the same tiers.
#[cfg(all(feature = "rocm", feature = "net-b18c384nbt"))]
#[test]
fn nn_parity_b18c384nbt_rocm() {
  let _gpu = gpu_lock();
  let d = run_net::<katago_nn_burn::b18c384nbt::Model>("rocm", katago_nn_burn::b18c384nbt::BPK_PATH, "evals_b18.txt", "kata1-b18c384nbt.out.bin", usize::MAX);
  check_wgpu(&d);
}

#[cfg(all(feature = "rocm", feature = "net-b18c384nbt"))]
#[test]
fn nn_parity_b18c384nbt_rocm_f16() {
  let _gpu = gpu_lock();
  let _half = Half::on();
  let d = run_net::<katago_nn_burn::b18c384nbt::Model>("rocm", katago_nn_burn::b18c384nbt::BPK_PATH, "evals_b18.txt", "kata1-b18c384nbt.out.bin", usize::MAX);
  check_f16(&d);
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
  let model = backend.load_model_file("b6", std::fs::read(katago_nn_burn::b6c96::BPK_PATH).unwrap(), "").unwrap();
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
