//! wasm32 only (`just test-wasm`, node): the burn CPU backend yields to the JS event loop inside
//! `getOutput` (PORTING.md §11), so a `setTimeout(_, 0)` callback queued before an evaluate runs
//! before that evaluate returns. With a microtask-only yield the whole evaluate (queue push, serve
//! task, forward pass, post-processing) would finish inside one macrotask and the callback would
//! run only afterwards. Not a port of upstream code.
#![cfg(all(target_arch = "wasm32", feature = "flex", feature = "net-b6c96"))]

use std::cell::Cell;
use std::rc::Rc;

use burn::prelude::*;
use wasm_bindgen::JsCast;
use wasm_bindgen::JsValue;
use wasm_bindgen::closure::Closure;
use wasm_bindgen::prelude::wasm_bindgen;
use wasm_bindgen_test::wasm_bindgen_test;

use katago_core::core::commontypes::enabled_t;
use katago_core::game::board::Board;
use katago_core::game::board::Location;
use katago_core::game::board::P_BLACK;
use katago_core::game::board::P_WHITE;
use katago_core::game::boardhistory::BoardHistory;
use katago_core::game::rules::Rules;
use katago_core::neuralnet::nneval::NNEvaluator;
use katago_core::neuralnet::nneval::NNResultBuf;
use katago_core::neuralnet::nninputs::MiscNNInputParams;
use katago_core::neuralnet::nninputs::ScoreValue;
use katago_core::port::executor::LocalBoxFuture;
use katago_core::port::executor::Spawner;
use katago_nn_burn::backend::BurnNeuralNet;

/// `globalThis.setTimeout(handler, 0)` (through Reflect: katago_nn_burn::host binds the same import).
fn set_timeout0(handler: &js_sys::Function) {
  let global = js_sys::global();
  let st: js_sys::Function = js_sys::Reflect::get(&global, &JsValue::from_str("setTimeout")).unwrap().dyn_into().unwrap();
  st.call2(&global, handler, &JsValue::from(0)).unwrap();
}

#[wasm_bindgen(module = "fs")]
extern "C" {
  #[wasm_bindgen(js_name = readFileSync, catch)]
  fn read_file_sync(path: &str) -> Result<js_sys::Uint8Array, JsValue>;
}

/// The browser host's Spawner: `wasm_bindgen_futures::spawn_local` (port::executor docs).
struct WasmSpawner;
impl Spawner for WasmSpawner {
  fn spawn_local(&self, f: LocalBoxFuture<'static, ()>) {
    wasm_bindgen_futures::spawn_local(f);
  }
}

#[wasm_bindgen_test]
async fn evaluate_yields_to_the_event_loop() {
  Board::init_hash();
  ScoreValue::init_tables();
  let bytes = read_file_sync(katago_nn_burn::b6c96::BPK_PATH)
    .expect("b6c96 .bpk (build with net-b6c96)")
    .to_vec();
  let device: Device = burn::backend::FlexDevice::default().into();
  let backend = Rc::new(BurnNeuralNet::<katago_nn_burn::b6c96::Model>::new(device));
  let nn_eval = Rc::new(
    NNEvaluator::new(
      "wasm",
      "g170-b6c96",
      "",
      4,
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
      "wasmSeed",
      false,
      0,
      backend,
      Rc::new(WasmSpawner),
      bytes,
    )
    .unwrap(),
  );
  nn_eval.spawn_server_threads().await.unwrap();

  let mut board = Board::new_x_y(19, 19).unwrap();
  let rules = Rules::new_full(Rules::KO_POSITIONAL, Rules::SCORING_AREA, Rules::TAX_NONE, true, false, 0, false, 7.5);
  let mut hist = BoardHistory::new_full(&board, P_BLACK, &rules, 0);
  for (x, y) in [(3, 3), (15, 15), (2, 15)] {
    let pla = hist.presumed_next_move_pla;
    hist.make_board_move_assume_legal(&mut board, Location::get_loc(x, y, 19), pla, None, false);
  }
  assert_eq!(hist.presumed_next_move_pla, P_WHITE);

  let mut batches_seen = 0;
  for k in 0..3 {
    let fired = Rc::new(Cell::new(false));
    let f2 = fired.clone();
    let cb = Closure::once_into_js(move || f2.set(true));
    set_timeout0(cb.unchecked_ref());
    assert!(!fired.get());
    let buf = Rc::new(NNResultBuf::new());
    nn_eval
      .evaluate(&board, &hist, P_WHITE, &MiscNNInputParams::new(), &buf, true, true)
      .await
      .unwrap();
    assert!(
      fired.get(),
      "eval {k}: a setTimeout(0) queued before evaluate did not run before it returned (getOutput did not yield to the event loop)"
    );
    assert!(buf.result.borrow().as_ref().unwrap().white_owner_map.is_some());
    batches_seen += 1;
  }
  assert_eq!(nn_eval.num_batches_processed(), batches_seen);
  nn_eval.kill_server_threads();
  wasm_bindgen_test::console_log!("evaluate_yields_to_the_event_loop: {batches_seen} evals");
}
