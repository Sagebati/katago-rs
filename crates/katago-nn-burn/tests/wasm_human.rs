//! wasm32 only, opt-in (`just test-wasm-human`, node, release): the human-SL net b18c384nbt-humanv0 on burn flex inside
//! wasm, through the ported NNEvaluator with SGFMetadata profiles, against the upstream NNEvaluator + Eigen records of
//! fixtures/upstream-v1.18.2/nn (evals_human.txt / b18c384nbt-humanv0.out.bin, the same data as the native
//! nn_parity_b18c384nbt_humanv0): the first NUM_EVALS evals (the empty board under 8 profiles, then mixed positions,
//! symmetries and profiles), nnHash exact (incl. SGFMetadata::getHash), policyOptimismUsed exact, the illegal-move
//! set exact, floats within the flex tolerances of nn_parity.rs. It shows the human net runs in the browser's CPU
//! backend (wasm SIMD flex) and prints the time per evaluation. Not a port of upstream code.
#![cfg(all(target_arch = "wasm32", feature = "flex", feature = "net-b18c384nbt-humanv0"))]

use std::rc::Rc;

use burn::prelude::*;
use wasm_bindgen::JsValue;
use wasm_bindgen::prelude::wasm_bindgen;
use wasm_bindgen_test::wasm_bindgen_test;

use katago_core::core::commontypes::enabled_t;
use katago_core::core::config_parser::ConfigParser;
use katago_core::core::hash::Hash128;
use katago_core::game::board::Board;
use katago_core::neuralnet::nneval::NNEvaluator;
use katago_core::neuralnet::nneval::NNResultBuf;
use katago_core::neuralnet::nninputs::MiscNNInputParams;
use katago_core::neuralnet::nninputs::ScoreValue;
use katago_core::neuralnet::sgfmetadata::SGFMetadata;
use katago_core::port::executor::LocalBoxFuture;
use katago_core::port::executor::Spawner;
use katago_nn_burn::backend::BurnNeuralNet;

mod common_nn;

use common_nn::hex_u32;
use common_nn::hex_u64;
use common_nn::parse_pos_tokens;
use common_nn::replay;

const NUM_EVALS: usize = 12;
const WORDS_PER_RECORD: usize = 4 + 10 + 362 + 361;

#[wasm_bindgen(module = "fs")]
extern "C" {
  #[wasm_bindgen(js_name = readFileSync, catch)]
  fn read_file_sync(path: &str) -> Result<js_sys::Uint8Array, JsValue>;
}

fn read(path: &str) -> Vec<u8> {
  read_file_sync(path).unwrap_or_else(|e| panic!("{path}: {e:?}")).to_vec()
}

fn fixture(name: &str) -> Vec<u8> {
  read(&format!("{}/../../fixtures/upstream-v1.18.2/nn/{}", env!("CARGO_MANIFEST_DIR"), name))
}

#[wasm_bindgen]
extern "C" {
  #[wasm_bindgen(js_namespace = Date)]
  fn now() -> f64;
}

struct WasmSpawner;
impl Spawner for WasmSpawner {
  fn spawn_local(&self, f: LocalBoxFuture<'static, ()>) {
    wasm_bindgen_futures::spawn_local(f);
  }
}

#[wasm_bindgen_test]
async fn human_net_on_wasm_flex() {
  Board::init_hash();
  ScoreValue::init_tables();
  let positions: Vec<String> = String::from_utf8(fixture("positions.txt"))
    .unwrap()
    .lines()
    .filter(|l| !l.is_empty())
    .map(str::to_string)
    .collect();
  let evals: Vec<String> = String::from_utf8(fixture("evals_human.txt"))
    .unwrap()
    .lines()
    .filter(|l| !l.is_empty())
    .map(str::to_string)
    .collect();
  let words: Vec<u32> = fixture("b18c384nbt-humanv0.out.bin")
    .chunks_exact(4)
    .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
    .collect();
  assert_eq!(words.len(), evals.len() * WORDS_PER_RECORD);

  let bytes = read(katago_nn_burn::b18c384nbt_humanv0::BPK_PATH);
  let device: Device = burn::backend::FlexDevice::default().into();
  let backend = Rc::new(BurnNeuralNet::<katago_nn_burn::b18c384nbt_humanv0::Model>::new(device));
  let nn_eval = Rc::new(
    NNEvaluator::new(
      "wasm",
      "b18c384nbt-humanv0",
      "",
      1,
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
      "wasmSeed",
      false,
      0,
      &mut ConfigParser::new(false, true),
      backend,
      Rc::new(WasmSpawner),
      bytes,
    )
    .unwrap(),
  );
  nn_eval.spawn_server_threads().await.unwrap();
  assert!(nn_eval.requires_sgf_metadata());

  let mut max_prob = 0.0f64;
  let mut max_policy = 0.0f64;
  let mut max_own = 0.0f64;
  let t0 = now();
  for (i, line) in evals.iter().take(NUM_EVALS).enumerate() {
    let t: Vec<&str> = line.split_whitespace().collect();
    assert_eq!(t.len(), 16, "{line}");
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
    let meta = SGFMetadata::get_profile(t[15]).unwrap();
    let (board, hist) = replay(&parse_pos_tokens(
      &positions[t[1].parse::<usize>().unwrap()].split_whitespace().collect::<Vec<_>>(),
    ));
    let pla: i8 = t[3].parse().unwrap();
    let buf = Rc::new(NNResultBuf::new());
    nn_eval
      .evaluate_sgf_meta(&board, &hist, pla, Some(&meta), &params, &buf, true, true)
      .await
      .unwrap();
    let out = buf.result.borrow_mut().take().unwrap();
    let rec = &words[i * WORDS_PER_RECORD..(i + 1) * WORDS_PER_RECORD];
    let want_hash = Hash128::new(rec[0] as u64 | ((rec[1] as u64) << 32), rec[2] as u64 | ((rec[3] as u64) << 32));
    assert!(out.nn_hash == want_hash, "eval {i}: nnHash");
    let f = |k: usize| f32::from_bits(rec[4 + k]);
    for (k, g) in [out.white_win_prob, out.white_loss_prob, out.white_no_result_prob].iter().enumerate() {
      max_prob = max_prob.max((*g as f64 - f(k) as f64).abs());
    }
    assert_eq!(out.policy_optimism_used.to_bits(), f(9).to_bits());
    for pos in 0..362 {
      let want = f32::from_bits(rec[14 + pos]);
      let g = out.policy_probs[pos];
      if want < 0.0 || g < 0.0 {
        assert!(want == -1.0 && g == -1.0, "eval {i}: policy[{pos}] legality");
      } else {
        max_policy = max_policy.max((g as f64 - want as f64).abs());
      }
    }
    let own = out.white_owner_map.as_ref().unwrap();
    for pos in 0..361 {
      max_own = max_own.max((own[pos] as f64 - f32::from_bits(rec[14 + 362 + pos]) as f64).abs());
    }
  }
  let ms = (now() - t0) / NUM_EVALS as f64;
  nn_eval.kill_server_threads();
  wasm_bindgen_test::console_log!(
    "human_net_on_wasm_flex: {NUM_EVALS} evals ({ms:.0} ms each, batch 1), max |wasm - eigen|: probabilities {max_prob:.2e}, policy {max_policy:.2e}, ownership {max_own:.2e}"
  );
  assert!(max_prob <= 1e-5 && max_policy <= 1e-5 && max_own <= 2e-5);
}
