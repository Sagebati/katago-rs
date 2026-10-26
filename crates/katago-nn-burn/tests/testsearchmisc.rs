//! Transcription (a test, not a port) of upstream cpp/tests/testsearchmisc.cpp `Tests::runNNOnTinyBoard` @ v1.18.2,
//! with the TestSearchCommon helpers as test-local copies (common_search/mod.rs).
//!
//! It replays the same calls (a 5x5 position on a 6x6 evaluator, symmetry 0, skipCache, owner map) through the
//! ported NNEvaluator with the REAL network g170-b6c96 on the burn flex backend (6x6 through common_search's
//! `PaddedNet`: the burn graph is fixed at 19x19) and compares the captured stdout with our Eigen rerun of
//! upstream (fixtures/upstream-v1.18.2/unit/testsearchmisc/runnnontinyboardtest-g170-bin-sym0.expected0.txt =
//! stdout of `katago runnnontinyboardtest g170-b6c96.bin.gz true true 0 false`, see
//! fixtures/upstream-v1.18.2/unit/testsearchcommon/regen.sh). The board diagram is the extracted
//! `runnnontinyboard.input0.txt`.
//!
//! Comparison: common_search's section comparator (integers exact, decimals within one unit of the last printed
//! digit; the backend's own log lines are dropped). `inputsNHWC=true useNHWC=true` of the Eigen run are `false`
//! here (burn is NCHW only; outputs do not depend on them). `runNNSymmetries` (the other block of this file)
//! needs a model-version-5 net (g103, fillRowV4): not covered (tests/fixture_coverage.rs allowlist).
// A transcription of upstream: keeps upstream's line breaks (rustfmt.toml).
#![cfg_attr(rustfmt, rustfmt::skip)]
#![cfg(all(feature = "flex", feature = "net-b6c96"))]
#![allow(clippy::field_reassign_with_default, clippy::needless_range_loop, clippy::too_many_arguments, clippy::needless_bool)]

#[macro_use]
mod common_search;

use std::rc::Rc;

use common_search::*;
use katago_core::game::board::*;
use katago_core::game::boardhistory::BoardHistory;
use katago_core::game::boardhistory::BoardHistoryModes;
use katago_core::game::rules::Rules;
use katago_core::neuralnet::nneval::NNEvaluator;
use katago_core::neuralnet::nneval::NNResultBuf;
use katago_core::neuralnet::nninputs::MiscNNInputParams;

const STEM: &str = "testsearchmisc";

/// upstream: Tests::runNNOnTinyBoard (testsearchmisc.cpp), with (g170-b6c96, inputsNHWC, useNHWC, symmetry 0, useFP16=false)
fn run_nn_on_tiny_board() {
  let model_symmetry = 0;
  //TestCommon::overrideForBackends(inputsNHWC, useNHWC);
  //NeuralNet::globalInitialize();
  let mut ctx = Ctx::new();

  let board = parse_board_input(STEM, "runnnontinyboard", 0, 5, 5);

  let next_pla: Player = P_WHITE;
  let rules = Rules::get_tromp_taylorish();
  let hist = Box::new(BoardHistory::new_full(&board, next_pla, &rules, 0, &BoardHistoryModes::new_full(false, false)));

  //Logger logger(nullptr, logToStdout=true, logToStderr=false, logTime=false): the Logger's lines go to the
  //captured stdout through common_search's logger.

  let nn_eval: Rc<NNEvaluator> = start_nn_eval(&mut ctx, MODEL_FILE, "", 6, 6, model_symmetry, false, false, false, false, false);

  let nn_input_params = MiscNNInputParams::default();
  let buf: Rc<NNResultBuf> = Rc::new(NNResultBuf::new());
  let skip_cache = true;
  let include_owner_map = true;
  ctx.run("evaluate", nn_eval.evaluate(&board, &hist, next_pla, &nn_input_params, &buf, skip_cache, include_owner_map)).unwrap();

  print_policy_value_ownership(&board, &buf);
  coutln!();
  coutln!();

  drop(buf);
  delete_nn_eval(&mut ctx, nn_eval);
  //NeuralNet::globalCleanup();
}

#[test]
fn run_nn_on_tiny_board_g170_sym0() {
  let expected_text = fixture(STEM, "runnnontinyboardtest-g170-bin-sym0.expected0.txt");
  let (actual_text, panic_msg) = run_capturing("testsearchmisc-tinyboard", run_nn_on_tiny_board);
  assert!(panic_msg.is_none(), "the transcription panicked: {}", panic_msg.unwrap_or_default());
  if let Ok(dir) = std::env::var("KATAGO_SEARCH_TEST_DUMP") {
    let p = std::path::Path::new(&dir).join("testsearchmisc-tinyboard.actual.txt");
    std::fs::write(&p, &actual_text).unwrap();
    eprintln!("wrote {}", p.display());
  }
  let expected = split_sections(&expected_text, &[]);
  let actual = split_sections(&actual_text, &[]);
  assert!(!expected.is_empty());
  assert_eq!(expected.len(), actual.len(), "section count\n--- actual ---\n{}", actual_text);
  let mut num_lines = 0;
  for (e, a) in expected.iter().zip(actual.iter()) {
    assert_eq!(e.title, a.title);
    num_lines += e.lines.len();
    match compare_section(e, a) {
      Outcome::Exact => eprintln!("testsearchmisc {}: exact", e.label),
      Outcome::Within(u, d, n) => eprintln!("testsearchmisc {}: within tolerance ({} lines, max {:.1} units, abs {:.3e})", e.label, n, u, d),
      Outcome::Diverged(k, el, al, why) => panic!("testsearchmisc {}: line {}: {}\n  expected: {}\n  actual:   {}", e.label, k, why, el, al),
      Outcome::Pending(r) => panic!("testsearchmisc {}: unexpected pending outcome: {}", e.label, r),
    }
  }
  eprintln!("testsearchmisc runNNOnTinyBoard: {} sections, {} expected lines compared", expected.len(), num_lines);
}
