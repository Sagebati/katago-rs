//! AsyncBot (search/asyncbot.rs) with a debugSkipNeuralNet evaluator (no net: the search runs on the evaluator's
//! seeded fake logits, so everything here is deterministic): genMoveSynchronous* return the chosen move, the
//! analyze callback loop (`gen_move_synchronous_analyze*`, upstream's callbackLoop thread, here runWholeSearch's
//! concurrent task) does not change the search result, honours firstCallbackAfter / callbackPeriod (huge, negative),
//! stops the search through the stop handle (`stopWithoutWait`, also from inside a callback and from onSearchBegun,
//! the way analysis.cpp ends a request that was terminated between pop and search start), `setKilled`, and a callback
//! error ends the search cleanly (the other search threads stop, the bot and the evaluator are reusable).
//! Not a port of upstream code (upstream has no unit test for AsyncBot; analysis goldens exercise it end to end).
//! TODO: the terminate-after-pop path itself (analysis.rs `handle_request`/the analysis loop) needs the analysis
//! engine and is covered by katago-cli tests/analysis_protocol.rs; the stop-from-onSearchBegun test below is its
//! AsyncBot-level half.
mod common;

use std::cell::Cell;
use std::future::Future;
use std::cell::RefCell;
use std::rc::Rc;

use common::Cases;
use common::run_or_stall_panic;
use futures::executor::LocalPool;
use katago_core::core::commontypes::enabled_t;
use katago_core::core::global::KataError;
use katago_core::core::timer::set_clock;
use katago_core::game::board::*;
use katago_core::game::boardhistory::BoardHistory;
use katago_core::game::rules::Rules;
use katago_core::neuralnet::desc::ModelDesc;
use katago_core::neuralnet::nneval::NNEvaluator;
use katago_core::neuralnet::nneval::NNResultBuf;
use katago_core::neuralnet::nninputs::NNOutput;
use katago_core::neuralnet::nninterface::*;
use katago_core::port::executor::*;
use katago_core::port::nlohmann;
use katago_core::search::asyncbot::*;
use katago_core::search::search::Search;
use katago_core::search::searchparams::SearchParams;
use katago_core::search::timecontrols::TimeControls;

struct NoBackend;
impl NeuralNet for NoBackend {
  fn global_initialize(&self) {}
  fn global_cleanup(&self) {}
  fn print_devices(&self) {}
  fn load_model_file(&self, _: &str, _: Vec<u8>, _: &str) -> Result<Box<dyn LoadedModel>, KataError> { panic!() }
  fn get_model_desc<'a>(&self, _: &'a dyn LoadedModel) -> &'a ModelDesc { panic!() }
  fn create_compute_context(&self, _: &[i32], _: i32, _: i32, _: &str, _: &str, _: bool, _: enabled_t, _: enabled_t, _: &dyn LoadedModel) -> Result<Box<dyn ComputeContext>, KataError> { panic!() }
  fn create_compute_handle(&self, _: &dyn ComputeContext, _: &dyn LoadedModel, _: i32, _: bool, _: bool, _: i32, _: i32) -> Result<Box<dyn ComputeHandle>, KataError> { panic!() }
  fn is_using_fp16(&self, _: &dyn ComputeHandle) -> bool { panic!() }
  fn create_input_buffers(&self, _: &dyn LoadedModel, _: i32, _: i32, _: i32) -> Box<dyn InputBuffers> { panic!() }
  fn get_output<'a>(&'a self, _: &'a mut dyn ComputeHandle, _: &'a mut dyn InputBuffers, _: i32, _: &'a [Rc<NNResultBuf>], _: &'a mut [NNOutput]) -> LocalBoxFuture<'a, Result<(), KataError>> { panic!() }
}

fn fake_clock() -> f64 {
  1000.0
}

/// Deterministic "sleep": yields 1 + floor(seconds*100) times.
fn yield_sleep(seconds: f64) -> LocalBoxFuture<'static, ()> {
  let n = 1 + if seconds.is_finite() && seconds > 0.0 { (seconds * 100.0) as usize } else { 0 };
  SLEEP_CALLS.with(|c| c.set(c.get() + 1));
  Box::pin(async move {
    for _ in 0..n {
      yield_now().await;
    }
  })
}
thread_local! { static SLEEP_CALLS: Cell<usize> = const { Cell::new(0) }; }

fn init() {
  Board::init_hash();
  katago_core::neuralnet::nninputs::ScoreValue::init_tables();
  set_clock(fake_clock);
  set_sleep(yield_sleep);
}

struct Env {
  pool: LocalPool,
  nn_eval: Rc<NNEvaluator>,
}

fn make_env() -> Env {
  let mut pool = LocalPool::new();
  let spawner: Rc<dyn Spawner> = Rc::new(pool.spawner());
  let nn_eval = Rc::new(
    NNEvaluator::new(
      "m", "m", "", 16, 9, 9, false, true, 12, 4, true, "", "", false, enabled_t::False, enabled_t::False, 1, &[0], "asyncbotseed", false, 0,
      Rc::new(NoBackend), spawner, Vec::new(),
    )
    .unwrap(),
  );
  run_or_stall_panic(&mut pool, "spawn", nn_eval.spawn_server_threads()).unwrap();
  Env { pool, nn_eval }
}

fn params(threads: i32, visits: i64) -> SearchParams {
  let mut p = SearchParams::for_tests_v2();
  p.num_threads = threads;
  p.max_visits = visits;
  p
}

fn make_bot(env: &Env, p: &SearchParams) -> AsyncBot {
  let mut bot = AsyncBot::new(p, env.nn_eval.clone(), "asyncbotseed");
  let board = Board::new_x_y(9, 9).unwrap();
  let rules = Rules::get_tromp_taylorish();
  let hist = BoardHistory::new_full(&board, P_BLACK, &rules, 0);
  bot.set_position(P_BLACK, &board, &hist).unwrap();
  bot.set_always_include_owner_map(true);
  bot
}

fn final_json(search: &Search) -> String {
  let mut v = serde_json::Value::Null;
  let ok = search.get_analysis_json(P_WHITE, 15, false, true, true, true, true, true, true, true, &mut v).unwrap();
  assert!(ok);
  format!("visits={} move={} {}", search.get_root_visits(), search.get_chosen_move_loc(), nlohmann::dump(&v))
}

fn run_plain(threads: i32, visits: i64) -> String {
  let mut env = make_env();
  let p = params(threads, visits);
  let mut bot = make_bot(&env, &p);
  let loc = run_or_stall_panic(&mut env.pool, "genmove", bot.gen_move_synchronous(P_BLACK, &TimeControls::new())).unwrap();
  assert_eq!(loc, bot.get_search().get_chosen_move_loc(), "genMoveSynchronous returns the chosen move");
  assert!(loc != Board::NULL_LOC);
  let s = final_json(bot.get_search());
  format!("loc={} {}", loc, s)
}

fn run_with_callback(threads: i32, visits: i64, period: f64, first: f64, calls: Rc<Cell<usize>>) -> String {
  let mut env = make_env();
  let p = params(threads, visits);
  let mut bot = make_bot(&env, &p);
  let c2 = calls.clone();
  let cb: AnalyzeCallback = Rc::new(move |s: &Search| {
    c2.set(c2.get() + 1);
    let mut v = serde_json::Value::Null;
    let _ = s.get_analysis_json(P_WHITE, 15, false, true, true, true, true, true, true, true, &mut v)?;
    let _ = s.get_root_visits();
    Ok(())
  });
  let loc = run_or_stall_panic(&mut env.pool, "genmoveAnalyze", bot.gen_move_synchronous_analyze(P_BLACK, &TimeControls::new(), 1.0, period, first, cb)).unwrap();
  assert_eq!(loc, bot.get_search().get_chosen_move_loc(), "genMoveSynchronousAnalyze returns the chosen move");
  assert!(loc != Board::NULL_LOC);
  let s = final_json(bot.get_search());
  format!("loc={} {}", loc, s)
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn callback_does_not_change_search_result_1thread() {
  init();
  for visits in [1i64, 2, 5, 50, 300] {
    let a = run_plain(1, visits);
    let calls = Rc::new(Cell::new(0));
    let b = run_with_callback(1, visits, 0.0, 0.0, calls.clone());
    crate::report!("visits {visits}: {} callbacks", calls.get());
    assert_eq!(a, b, "callback changed the result at visits={visits}");
  }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn callback_multithread_invariants() {
  init();
  for threads in [2, 4, 8] {
    let calls = Rc::new(Cell::new(0));
    let b = run_with_callback(threads, 200, 0.0, 0.0, calls.clone());
    assert!(b.contains("visits=20"), "{}", &b[..40]);
    let a = run_plain(threads, 200);
    crate::report!("threads {threads}: {} callbacks; same as plain: {}", calls.get(), a == b);
  }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn callback_huge_periods() {
  init();
  let calls = Rc::new(Cell::new(0));
  run_with_callback(1, 50, 1.0e7, 0.0, calls.clone());
  assert_eq!(calls.get(), 1, "callbackPeriod >= 1e7 -> only the first callback");
  let calls = Rc::new(Cell::new(0));
  run_with_callback(1, 50, 0.0, 1.0e7, calls.clone());
  assert_eq!(calls.get(), 0, "firstCallbackAfter >= 1e7 -> no callbacks");
  let calls = Rc::new(Cell::new(0));
  run_with_callback(1, 50, 0.0, -1.0, calls.clone());
  assert_eq!(calls.get(), 0, "firstCallbackAfter < 0 -> upstream callbackLoop returns at once");
  let calls = Rc::new(Cell::new(0));
  run_with_callback(1, 50, -1.0, 0.0, calls.clone());
  assert_eq!(calls.get(), 1, "callbackPeriod < 0 -> one callback");
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn stop_from_callback_and_kill() {
  init();
  let mut env = make_env();
  let p = params(2, 20000);
  let mut bot = make_bot(&env, &p);
  let h = bot.get_stop_handle();
  let n = Rc::new(Cell::new(0));
  let n2 = n.clone();
  let cb: AnalyzeCallback = Rc::new(move |_s: &Search| {
    n2.set(n2.get() + 1);
    if n2.get() == 3 {
      h.stop_without_wait();
    }
    Ok(())
  });
  let loc = run_or_stall_panic(&mut env.pool, "stop", bot.gen_move_synchronous_analyze(P_BLACK, &TimeControls::new(), 1.0, 0.0, 0.0, cb)).unwrap();
  let v = bot.get_search().get_root_visits();
  crate::report!("stopped after {} callbacks, root visits {v}, loc {loc}", n.get());
  assert_eq!(n.get(), 3, "no callback after the one that stopped the search");
  assert!(v < 2000 && loc != Board::NULL_LOC, "stopWithoutWait from a callback must end the search early (visits {v} of 20000)");
  // A new search after a stop runs normally (shouldStopNow reset).
  bot.set_params_no_clearing(&params(2, 500));
  let loc2 = run_or_stall_panic(&mut env.pool, "after stop", bot.gen_move_synchronous(P_BLACK, &TimeControls::new())).unwrap();
  crate::report!("second search: {} visits {}", loc2, bot.get_search().get_root_visits());
  assert!(loc2 != Board::NULL_LOC);
  assert!(bot.get_search().get_root_visits() >= 500, "a new search after a stop runs to maxVisits (shouldStopNow is reset)");
  // Kill during search
  let h2 = bot.get_stop_handle();
  let k = Rc::new(Cell::new(0));
  let k2 = k.clone();
  let cb2: AnalyzeCallback = Rc::new(move |_s: &Search| {
    k2.set(k2.get() + 1);
    h2.set_killed();
    Ok(())
  });
  bot.clear_search();
  let loc3 = run_or_stall_panic(&mut env.pool, "kill", bot.gen_move_synchronous_analyze(P_BLACK, &TimeControls::new(), 1.0, 0.0, 0.0, cb2)).unwrap();
  crate::report!("killed: loc {loc3} visits {}", bot.get_search().get_root_visits());
  let loc4 = run_or_stall_panic(&mut env.pool, "after kill", bot.gen_move_synchronous(P_BLACK, &TimeControls::new())).unwrap();
  assert_eq!(loc4, Board::NULL_LOC, "killed bot does not search");
  drop(bot);
  env.nn_eval.kill_server_threads();
  env.pool.run_until_stalled();
  assert_eq!(Rc::strong_count(&env.nn_eval), 1);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn callback_error_ends_search_cleanly() {
  const POLL_BOUND: usize = 200; // 5 when the threads stop; thousands when they run to maxVisits
  init();
  for threads in [1, 3] {
    let mut env = make_env();
    let p = params(threads, 20000);
    let mut bot = make_bot(&env, &p);
    let n = Rc::new(Cell::new(0));
    let n2 = n.clone();
    let cb: AnalyzeCallback = Rc::new(move |_s: &Search| {
      n2.set(n2.get() + 1);
      if n2.get() == 2 {
        return Err(KataError::StringError("boom".to_string()));
      }
      Ok(())
    });
    // The tree is discarded when the search fails, so the visits cannot be read afterwards: count how often the
    // search future is polled instead (every playout waits for the evaluator, so this grows with the visits).
    let polls = Cell::new(0usize);
    let r = {
      let tc = TimeControls::new();
      let mut fut = Box::pin(bot.gen_move_synchronous_analyze(P_BLACK, &tc, 1.0, 0.0, 0.0, cb));
      let counted = std::future::poll_fn(|cx| {
        polls.set(polls.get() + 1);
        fut.as_mut().poll(cx)
      });
      run_or_stall_panic(&mut env.pool, "err", counted)
    };
    assert!(matches!(r, Err(KataError::StringError(ref m)) if m == "boom"));
    crate::report!("threads {threads}: error after {} callbacks, {} polls", n.get(), polls.get());
    assert_eq!(n.get(), 2, "no callback after the one that failed");
    assert!(polls.get() < POLL_BOUND, "the search threads must stop when the callback task fails ({} polls)", polls.get());
    // the bot is usable again and the evaluator is not wedged
    bot.set_params(&params(threads, 50));
    let loc = run_or_stall_panic(&mut env.pool, "after err", bot.gen_move_synchronous(P_BLACK, &TimeControls::new())).unwrap();
    assert!(loc != Board::NULL_LOC);
    drop(bot);
    env.nn_eval.kill_server_threads();
    env.pool.run_until_stalled();
    assert_eq!(Rc::strong_count(&env.nn_eval), 1, "serve task ended");
  }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn search_begun_called_once_before_callbacks() {
  init();
  let mut env = make_env();
  let p = params(1, 30);
  let mut bot = make_bot(&env, &p);
  let order: Rc<RefCell<Vec<String>>> = Rc::new(RefCell::new(Vec::new()));
  let o1 = order.clone();
  let begun: SearchBegunCallback = Rc::new(move |s: &Search| o1.borrow_mut().push(format!("begun {}", s.get_root_visits())));
  let o2 = order.clone();
  let cb: AnalyzeCallback = Rc::new(move |s: &Search| {
    o2.borrow_mut().push(format!("cb {}", s.get_root_visits()));
    Ok(())
  });
  run_or_stall_panic(&mut env.pool, "begun", bot.gen_move_synchronous_analyze_full(P_BLACK, &TimeControls::new(), 1.0, 0.0, 0.0, cb, Some(begun))).unwrap();
  let o = order.borrow();
  assert!(o[0].starts_with("begun"), "{o:?}");
  assert_eq!(o.iter().filter(|s| s.starts_with("begun")).count(), 1);
  crate::report!("{:?}", &o[..o.len().min(8)]);
}

/// The three non-analyze overloads return the chosen move, as the 4-argument one does; a killed bot returns NULL_LOC.
#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn gen_move_synchronous_overloads_return_the_chosen_move() {
  init();
  let mut cases = Cases::new("asyncbot::gen_move_synchronous_overloads_return_the_chosen_move");
  let mut env = make_env();
  let p = params(1, 60);
  let mut bot = make_bot(&env, &p);
  let tc = TimeControls::new();
  let a = run_or_stall_panic(&mut env.pool, "plain", bot.gen_move_synchronous(P_BLACK, &tc)).unwrap();
  assert!(a != Board::NULL_LOC);
  assert_eq!(a, bot.get_search().get_chosen_move_loc());
  assert_eq!(bot.get_search().get_root_visits(), 60);
  cases.ran();
  for name in ["search_factor", "full", "full_begun"] {
    bot.clear_search();
    let loc = match name {
      "search_factor" => run_or_stall_panic(&mut env.pool, name, bot.gen_move_synchronous_search_factor(P_BLACK, &tc, 1.0)),
      "full" => run_or_stall_panic(&mut env.pool, name, bot.gen_move_synchronous_full(P_BLACK, &tc, 1.0, None)),
      _ => {
        let begun: SearchBegunCallback = Rc::new(|_s: &Search| {});
        run_or_stall_panic(&mut env.pool, name, bot.gen_move_synchronous_full(P_BLACK, &tc, 1.0, Some(begun)))
      }
    }
    .unwrap();
    assert!(loc != Board::NULL_LOC, "{name}");
    assert_eq!(loc, bot.get_search().get_chosen_move_loc(), "{name}");
    cases.ran();
  }
  // searchFactor scales the visit budget.
  bot.clear_search();
  let loc = run_or_stall_panic(&mut env.pool, "half", bot.gen_move_synchronous_search_factor(P_BLACK, &tc, 0.5)).unwrap();
  assert!(loc != Board::NULL_LOC);
  assert_eq!(bot.get_search().get_root_visits(), 30, "searchFactor scales maxVisits");
  cases.ran();
  // The analyze overload takes it too.
  bot.clear_search();
  let cb: AnalyzeCallback = Rc::new(|_s: &Search| Ok(()));
  let loc = run_or_stall_panic(&mut env.pool, "analyze half", bot.gen_move_synchronous_analyze(P_BLACK, &tc, 0.5, 0.0, 0.0, cb)).unwrap();
  assert_eq!(loc, bot.get_search().get_chosen_move_loc());
  assert_eq!(bot.get_search().get_root_visits(), 30, "the analyze overload takes searchFactor too");
  cases.ran();
  // Killed: nothing is searched.
  bot.get_stop_handle().set_killed();
  let loc = run_or_stall_panic(&mut env.pool, "killed", bot.gen_move_synchronous(P_BLACK, &tc)).unwrap();
  assert_eq!(loc, Board::NULL_LOC);
  cases.ran();
  drop(bot);
  env.nn_eval.kill_server_threads();
  env.pool.run_until_stalled();
}

/// onSearchBegun is called exactly once per search, before any playout (root visits are still those of the reused
/// tree), by every overload that takes it.
#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn search_begun_is_called_once_per_search() {
  init();
  let mut cases = Cases::new("asyncbot::search_begun_is_called_once_per_search");
  let mut env = make_env();
  let p = params(1, 40);
  let mut bot = make_bot(&env, &p);
  let tc = TimeControls::new();
  let calls: Rc<RefCell<Vec<i64>>> = Rc::new(RefCell::new(Vec::new()));
  let mk = |calls: &Rc<RefCell<Vec<i64>>>| -> SearchBegunCallback {
    let calls = calls.clone();
    Rc::new(move |s: &Search| calls.borrow_mut().push(s.get_root_visits()))
  };
  // Plain overload.
  let loc = run_or_stall_panic(&mut env.pool, "plain", bot.gen_move_synchronous_full(P_BLACK, &tc, 1.0, Some(mk(&calls)))).unwrap();
  assert!(loc != Board::NULL_LOC);
  assert_eq!(*calls.borrow(), vec![0], "once, before the first visit");
  cases.ran();
  // Again on the same tree (not cleared): once more, with the visits of the reused tree.
  bot.set_params_no_clearing(&params(1, 80));
  run_or_stall_panic(&mut env.pool, "plain 2", bot.gen_move_synchronous_full(P_BLACK, &tc, 1.0, Some(mk(&calls)))).unwrap();
  assert_eq!(*calls.borrow(), vec![0, 40]);
  assert_eq!(bot.get_search().get_root_visits(), 80);
  cases.ran();
  // Analyze overload: once, before the first callback, and no matter how many callbacks follow.
  bot.clear_search();
  bot.set_params(&params(1, 1500));
  calls.borrow_mut().clear();
  let order: Rc<RefCell<Vec<&'static str>>> = Rc::new(RefCell::new(Vec::new()));
  let (o1, o2) = (order.clone(), order.clone());
  let c1 = calls.clone();
  let begun: SearchBegunCallback = Rc::new(move |s: &Search| {
    c1.borrow_mut().push(s.get_root_visits());
    o1.borrow_mut().push("begun");
  });
  let cb: AnalyzeCallback = Rc::new(move |_s: &Search| {
    o2.borrow_mut().push("cb");
    Ok(())
  });
  run_or_stall_panic(&mut env.pool, "analyze", bot.gen_move_synchronous_analyze_full(P_BLACK, &tc, 1.0, 0.0, 0.0, cb, Some(begun))).unwrap();
  assert_eq!(*calls.borrow(), vec![0]);
  {
    let o = order.borrow();
    assert_eq!(o[0], "begun");
    assert_eq!(o.iter().filter(|s| **s == "begun").count(), 1);
    assert!(o.len() >= 2, "callbacks ran: {o:?}");
  }
  cases.ran();
  // A bot that is killed never starts a search, so onSearchBegun is not called.
  bot.get_stop_handle().set_killed();
  calls.borrow_mut().clear();
  let loc = run_or_stall_panic(&mut env.pool, "killed", bot.gen_move_synchronous_full(P_BLACK, &tc, 1.0, Some(mk(&calls)))).unwrap();
  assert_eq!(loc, Board::NULL_LOC);
  assert!(calls.borrow().is_empty());
  cases.ran();
  drop(bot);
  env.nn_eval.kill_server_threads();
  env.pool.run_until_stalled();
}

/// What analysis.cpp does for a request terminated between the pop and the search start: its onSearchBegun callback
/// calls `bot->stopWithoutWait()`, and the search must end at once (with whatever root eval it has) instead of running
/// to maxVisits. Both genMove flavours; the bot is reusable afterwards.
#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn stop_from_search_begun_ends_the_search_early() {
  init();
  let mut cases = Cases::new("asyncbot::stop_from_search_begun_ends_the_search_early");
  for threads in [1, 3] {
    for analyze in [false, true] {
      let mut env = make_env();
      let p = params(threads, 20000);
      let mut bot = make_bot(&env, &p);
      let tc = TimeControls::new();
      let h = bot.get_stop_handle();
      let begun: SearchBegunCallback = Rc::new(move |_s: &Search| h.stop_without_wait());
      let loc = if analyze {
        let cb: AnalyzeCallback = Rc::new(|_s: &Search| Ok(()));
        run_or_stall_panic(&mut env.pool, "begun-stop analyze", bot.gen_move_synchronous_analyze_full(P_BLACK, &tc, 1.0, 0.0, 0.0, cb, Some(begun))).unwrap()
      } else {
        run_or_stall_panic(&mut env.pool, "begun-stop", bot.gen_move_synchronous_full(P_BLACK, &tc, 1.0, Some(begun))).unwrap()
      };
      let v = bot.get_search().get_root_visits();
      crate::report!("threads {threads} analyze {analyze}: stopped in onSearchBegun, visits {v}, loc {loc}");
      assert!(v < 200, "stopWithoutWait from onSearchBegun must end the search at once (visits {v} of 20000)");
      // Stopped before the first playout: no root children, so no chosen move (analysis.cpp writes a dummy response).
      assert_eq!(loc, bot.get_search().get_chosen_move_loc());
      // The next search is not stopped.
      bot.set_params_no_clearing(&params(threads, 300));
      let loc = run_or_stall_panic(&mut env.pool, "after", bot.gen_move_synchronous(P_BLACK, &tc)).unwrap();
      assert!(loc != Board::NULL_LOC);
      assert!(bot.get_search().get_root_visits() >= 300);
      drop(bot);
      env.nn_eval.kill_server_threads();
      env.pool.run_until_stalled();
      cases.ran();
    }
  }
}

/// A stop requested while nothing is running is forgotten when the next genMove starts (genMoveAsync resets
/// shouldStopNow), for both flavours.
#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn stop_before_the_search_is_forgotten() {
  init();
  let mut cases = Cases::new("asyncbot::stop_before_the_search_is_forgotten");
  let mut env = make_env();
  let p = params(1, 120);
  let mut bot = make_bot(&env, &p);
  let tc = TimeControls::new();
  bot.get_stop_handle().stop_without_wait();
  bot.stop_without_wait();
  run_or_stall_panic(&mut env.pool, "plain", bot.gen_move_synchronous(P_BLACK, &tc)).unwrap();
  assert_eq!(bot.get_search().get_root_visits(), 120);
  cases.ran();
  bot.clear_search();
  bot.get_stop_handle().stop_without_wait();
  let cb: AnalyzeCallback = Rc::new(|_s: &Search| Ok(()));
  run_or_stall_panic(&mut env.pool, "analyze", bot.gen_move_synchronous_analyze(P_BLACK, &tc, 1.0, 0.0, 0.0, cb)).unwrap();
  assert_eq!(bot.get_search().get_root_visits(), 120);
  cases.ran();
  drop(bot);
  env.nn_eval.kill_server_threads();
  env.pool.run_until_stalled();
}
