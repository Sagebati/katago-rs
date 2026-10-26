//! Batch-shape buckets (`BurnNeuralNet::with_batch_buckets`, port-only): a padded batch gives the same
//! outputs as the unpadded one (rows are independent; the padding rows' outputs are dropped), and the
//! bucket choice. burn flex, g170-b6c96. Not a port of upstream code.
#![cfg(all(not(target_arch = "wasm32"), feature = "flex", feature = "net-b6c96"))]

use std::rc::Rc;

use futures::executor::LocalPool;
use katago_core::core::commontypes::enabled_t;
use katago_core::game::board::Board;
use katago_core::game::board::Location;
use katago_core::game::board::P_BLACK;
use katago_core::game::boardhistory::BoardHistory;
use katago_core::game::boardhistory::BoardHistoryModes;
use katago_core::game::rules::Rules;
use katago_core::neuralnet::nneval::NNEvaluator;
use katago_core::neuralnet::nneval::NNResultBuf;
use katago_core::neuralnet::nninputs::MiscNNInputParams;
use katago_core::neuralnet::nninputs::NNOutput;
use katago_core::neuralnet::nninputs::ScoreValue;
use katago_core::port::executor::Spawner;
use katago_nn_burn::backend::BurnNeuralNet;
use katago_nn_burn::backend::padded_batch_size;

#[test]
fn bucket_choice() {
  let b = [1, 2, 4, 8, 16, 32];
  assert_eq!(padded_batch_size(&b, 1, 32), 1);
  assert_eq!(padded_batch_size(&b, 3, 32), 4);
  assert_eq!(padded_batch_size(&b, 17, 32), 32);
  // a bucket above the buffers' capacity is never used
  assert_eq!(padded_batch_size(&b, 17, 24), 17);
  assert_eq!(padded_batch_size(&[], 5, 32), 5);
}

/// Evaluates three positions one at a time (batches of 1) with `buckets`, returns their outputs.
fn eval_positions(buckets: Vec<i32>) -> Vec<NNOutput> {
  Board::init_hash();
  ScoreValue::init_tables();
  let mut pool = LocalPool::new();
  let spawner: Rc<dyn Spawner> = Rc::new(pool.spawner());
  let bytes = std::fs::read(katago_nn_burn::b6c96::BPK_PATH).unwrap();
  let device: burn::prelude::Device = burn::backend::FlexDevice.into();
  let backend = Rc::new(BurnNeuralNet::<katago_nn_burn::b6c96::Model>::new(device).with_batch_buckets(buckets));
  let nn_eval = Rc::new(
    NNEvaluator::new(
      "buckets",
      "g170-b6c96",
      "",
      8,
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
      "bucketSeed",
      false,
      0,
      backend,
      spawner,
      bytes,
    )
    .unwrap(),
  );
  let outputs = pool.run_until(async {
    nn_eval.spawn_server_threads().await.unwrap();
    let mut board = Board::new_x_y(19, 19).unwrap();
    let rules = Rules::new_full(Rules::KO_POSITIONAL, Rules::SCORING_AREA, Rules::TAX_NONE, true, false, 0, false, 7.5);
    let mut hist = BoardHistory::new_full(&board, P_BLACK, &rules, 0, &BoardHistoryModes::new());
    let mut outputs = Vec::new();
    for (x, y) in [(3, 3), (15, 15), (2, 15)] {
      let pla = hist.presumed_next_move_pla;
      hist.make_board_move_assume_legal(&mut board, Location::get_loc(x, y, 19), pla, None, false);
      let buf = Rc::new(NNResultBuf::new());
      let next = hist.presumed_next_move_pla;
      nn_eval
        .evaluate(&board, &hist, next, &MiscNNInputParams::new(), &buf, true, true)
        .await
        .unwrap();
      outputs.push((**buf.result.borrow().as_ref().unwrap()).clone());
    }
    outputs
  });
  nn_eval.kill_server_threads();
  pool.run_until_stalled();
  outputs
}

#[test]
fn padded_batches_give_the_same_outputs() {
  let plain = eval_positions(Vec::new());
  // batch 1 runs as 4 rows: three of stale (here: the previous positions' or zero) inputs
  let padded = eval_positions(vec![4, 8]);
  for (i, (a, b)) in plain.iter().zip(&padded).enumerate() {
    let max_policy = a
      .policy_probs
      .iter()
      .zip(b.policy_probs.iter())
      .map(|(x, y)| (x - y).abs())
      .fold(0.0f32, f32::max);
    let own_a = a.white_owner_map.as_ref().unwrap();
    let own_b = b.white_owner_map.as_ref().unwrap();
    let max_own = own_a.iter().zip(own_b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
    let d_win = (a.white_win_prob - b.white_win_prob).abs();
    let d_lead = (a.white_lead - b.white_lead).abs();
    // flex runs each row the same way whatever the batch size; allow accumulation-order noise only.
    assert!(
      max_policy <= 1e-6 && max_own <= 1e-6 && d_win <= 1e-6 && d_lead <= 1e-5,
      "position {i}: policy {max_policy:e}, ownership {max_own:e}, win {d_win:e}, lead {d_lead:e}"
    );
  }
}
