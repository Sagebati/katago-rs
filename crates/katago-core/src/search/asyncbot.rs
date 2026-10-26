// Ported from KataGo cpp/search/asyncbot.cpp, cpp/search/asyncbot.h @ fd0723fdbc0e9d82cf269c9630af8c27c57c07c4
//
// Divergences (port-manifest.toml notes, OWNERSHIP.tsv, PORTING.md §11):
// - No bot thread. Upstream's search thread waits on a condition variable for a job that genMoveAsync*
//   queues; here the "thread loop" body (internalSearchThreadLoop) is an async fn that runs the queued
//   job once, awaited by waitForSearchToEnd, so genMoveSynchronous* = queue the job + await it. A
//   search therefore only runs while its caller holds `&mut AsyncBot`, and the setters' stopAndWait
//   never has a running search to wait for. controlMutex and the condition variables are removed.
// - The fire-and-forget entry points (genMoveAsync*, genMoveAsyncAnalyze*, ponder, analyzeAsync) are not
//   public: without a thread nothing would run the job they queue. The two full genMove*Async overloads
//   stay as the private queueing halves of genMoveSynchronous*.
// - stopWithoutWait / setKilled can be called by other virtual threads while the owner awaits a search
//   (analysis.cpp's terminate): `shouldStopNow` and `isKilled` are shared Cells, reachable through
//   `AsyncBotStopHandle` (PORT NOTE).
// - Callbacks that upstream writes as `bot->getSearch()` get the Search as an argument (the bot is
//   mutably borrowed by the running search): `onSearchBegun` / the analyze callback are
//   `Rc<dyn Fn(&Search)>`. The analyze callback loop thread is a task that Search::runWholeSearch polls
//   next to its search threads (its `concurrent_task` parameter), waiting through the injected
//   `port::executor::sleep` (upstream: `condition_variable::wait_for`).
// - `Logger*` dropped (log crate).

use std::cell::Cell;
use std::rc::Rc;

use crate::core::global::KataError;
use crate::game::board::Board;
use crate::game::board::Loc;
use crate::game::board::Player;
use crate::game::boardhistory::BoardHistory;
use crate::neuralnet::nneval::NNEvaluator;
use crate::port::cpp;
use crate::port::executor::sleep;
use crate::port::executor::LocalBoxFuture;
use crate::search::reportedsearchvalues::ReportedSearchValues;
use crate::search::evalcache::EvalCacheTable;
use crate::search::patternbonustable::PatternBonusTable;
use crate::search::search::ConcurrentTask;
use crate::search::search::Search;
use crate::search::searchparams::SearchParams;
use crate::search::timecontrols::TimeControls;
use crate::testAssert;

/// The analyze callback (upstream `std::function<void(const Search* search)>`).
/// PORT NOTE: returns Result: an exception from the callback is uncaught in upstream's callback thread
/// (std::terminate); here it ends the search with that error.
pub type AnalyzeCallback = Rc<dyn Fn(&Search) -> Result<(), KataError>>;
/// `onSearchBegun` (upstream `std::function<void()>`). PORT NOTE: gets the Search (see the file header).
pub type SearchBegunCallback = Rc<dyn Fn(&Search)>;
/// `queuedOnMove` (upstream `std::function<void(Loc,int,Search*)>`).
pub type OnMoveCallback = Box<dyn FnMut(Loc, i32, &mut Search)>;

/// upstream: AsyncBot (asyncbot.h)
pub struct AsyncBot {
  search: Box<Search>,

  // PORT NOTE: controlMutex, threadWaitingToSearch, userWaitingForStop, searchThread removed (OWNERSHIP.tsv).
  is_running: bool,
  is_pondering: bool,
  /// PORT NOTE: shared with AsyncBotStopHandle (setKilled from another virtual thread).
  is_killed: Rc<Cell<bool>>,
  should_stop_now: Rc<Cell<bool>>,
  queued_search_id: i32,
  queued_on_move: Option<OnMoveCallback>,
  time_controls: TimeControls,
  search_factor: f64,
  analyze_callback_period: f64,
  analyze_first_callback_after: f64,
  analyze_callback: Option<AnalyzeCallback>,
  search_begun_callback: Option<SearchBegunCallback>,
}

/// What other virtual threads may call on an AsyncBot while its owner awaits a search
/// (`stopWithoutWait`, `setKilled`; analysis.cpp's terminate and shutdown).
/// PORT NOTE: not upstream; upstream calls these on the shared `AsyncBot*` from other threads.
#[derive(Clone)]
pub struct AsyncBotStopHandle {
  should_stop_now: Rc<Cell<bool>>,
  is_killed: Rc<Cell<bool>>,
}

impl AsyncBotStopHandle {
  /// upstream: AsyncBot::stopWithoutWait (asyncbot.cpp)
  pub fn stop_without_wait(&self) {
    self.should_stop_now.set(true);
  }

  /// upstream: AsyncBot::setKilled (asyncbot.cpp)
  pub fn set_killed(&self) {
    // PORT NOTE: lock removed; threadWaitingToSearch.notify_all() has no waiter (no bot thread).
    self.is_killed.set(true);
    self.should_stop_now.set(true);
  }
}

impl AsyncBot {
  /// upstream: AsyncBot::AsyncBot(const SearchParams&, NNEvaluator*, Logger*, const std::string&) (asyncbot.cpp)
  /// PORT NOTE: `Logger*` dropped; no search thread is started.
  pub fn new(params: &SearchParams, nn_eval: Rc<NNEvaluator>, rand_seed: &str) -> AsyncBot {
    let search: Box<Search> = Box::new(Search::new(params, nn_eval, rand_seed));
    AsyncBot::from_search(search)
  }

  /// upstream: AsyncBot::AsyncBot(const SearchParams&, NNEvaluator*, NNEvaluator*, Logger*, const std::string&) (asyncbot.cpp)
  /// PORT NOTE: `Logger*` dropped; no search thread is started; throws (Search's constructor) -> Result.
  pub fn new_human_eval(params: &SearchParams, nn_eval: Rc<NNEvaluator>, human_eval: Option<Rc<NNEvaluator>>, rand_seed: &str) -> Result<AsyncBot, KataError> {
    let search: Box<Search> = Box::new(Search::new_human_eval(params, nn_eval, human_eval, rand_seed)?);
    Ok(AsyncBot::from_search(search))
  }

  /// The shared member initializer list of both constructors.
  fn from_search(search: Box<Search>) -> AsyncBot {
    AsyncBot {
      search,
      is_running: false,
      is_pondering: false,
      is_killed: Rc::new(Cell::new(false)),
      should_stop_now: Rc::new(Cell::new(false)),
      queued_search_id: 0,
      queued_on_move: None,
      time_controls: TimeControls::new(),
      search_factor: 1.0,
      analyze_callback_period: -1.0,
      analyze_first_callback_after: -1.0,
      analyze_callback: None,
      search_begun_callback: None,
    }
  }

  // upstream: AsyncBot::~AsyncBot (asyncbot.cpp)
  // PORT NOTE: omitted: it stops and joins the search thread (none here) and deletes the search (Box).

  /// PORT NOTE: not upstream: a handle for stopWithoutWait / setKilled from other virtual threads.
  pub fn get_stop_handle(&self) -> AsyncBotStopHandle {
    AsyncBotStopHandle { should_stop_now: self.should_stop_now.clone(), is_killed: self.is_killed.clone() }
  }

  /// upstream: AsyncBot::getRootBoard (asyncbot.cpp)
  pub fn get_root_board(&self) -> &Board {
    &self.search.root_board
  }
  /// upstream: AsyncBot::getRootHist (asyncbot.cpp)
  pub fn get_root_hist(&self) -> &BoardHistory {
    &self.search.root_history
  }
  /// upstream: AsyncBot::getRootPla (asyncbot.cpp)
  pub fn get_root_pla(&self) -> Player {
    self.search.root_pla
  }
  /// upstream: AsyncBot::getPlayoutDoublingAdvantagePla (asyncbot.cpp)
  pub fn get_playout_doubling_advantage_pla(&self) -> Player {
    self.search.get_playout_doubling_advantage_pla()
  }

  /// upstream: AsyncBot::getSearchStopAndWait (asyncbot.cpp)
  //Get the search, after stopping and waiting to terminate any existing search
  //Note that one still should NOT mind any threading issues using this search object and other asyncBot calls at the same time.
  pub fn get_search_stop_and_wait(&mut self) -> &mut Search {
    self.stop_and_wait();
    &mut self.search
  }
  /// upstream: AsyncBot::getSearch (asyncbot.cpp)
  //Get the search directly. If the asyncbot is doing anything asynchronous, the search MAY STILL BE RUNNING!
  pub fn get_search(&self) -> &Search {
    &self.search
  }
  /// upstream: AsyncBot::getParams (asyncbot.cpp)
  pub fn get_params(&self) -> &SearchParams {
    &self.search.search_params
  }

  /// upstream: AsyncBot::setPosition (asyncbot.cpp)
  /// PORT NOTE: throws (Search::setPosition) -> Result.
  pub fn set_position(&mut self, pla: Player, board: &Board, history: &BoardHistory) -> Result<(), KataError> {
    self.stop_and_wait();
    self.search.set_position(pla, board, history)
  }
  /// upstream: AsyncBot::setPlayerAndClearHistory (asyncbot.cpp)
  pub fn set_player_and_clear_history(&mut self, pla: Player) -> Result<(), KataError> {
    self.stop_and_wait();
    self.search.set_player_and_clear_history(pla)
  }
  /// upstream: AsyncBot::setPlayerIfNew (asyncbot.cpp)
  pub fn set_player_if_new(&mut self, pla: Player) -> Result<(), KataError> {
    self.stop_and_wait();
    self.search.set_player_if_new(pla)
  }
  /// upstream: AsyncBot::setKomiIfNew (asyncbot.cpp)
  pub fn set_komi_if_new(&mut self, new_komi: f32) {
    self.stop_and_wait();
    self.search.set_komi_if_new(new_komi);
  }
  /// upstream: AsyncBot::setAvoidMoveUntilByLoc (asyncbot.cpp)
  pub fn set_avoid_move_until_by_loc(&mut self, b_vec: &[i32], w_vec: &[i32]) {
    self.stop_and_wait();
    self.search.set_avoid_move_until_by_loc(b_vec, w_vec);
  }
  /// upstream: AsyncBot::setAvoidMoveUntilRescaleRoot (asyncbot.cpp)
  pub fn set_avoid_move_until_rescale_root(&mut self, b: bool) {
    self.stop_and_wait();
    self.search.set_avoid_move_until_rescale_root(b);
  }
  /// upstream: AsyncBot::setRootHintLoc (asyncbot.cpp)
  pub fn set_root_hint_loc(&mut self, loc: Loc) {
    self.stop_and_wait();
    self.search.set_root_hint_loc(loc);
  }
  /// upstream: AsyncBot::setAlwaysIncludeOwnerMap (asyncbot.cpp)
  pub fn set_always_include_owner_map(&mut self, b: bool) {
    self.stop_and_wait();
    self.search.set_always_include_owner_map(b);
  }
  /// upstream: AsyncBot::setParams (asyncbot.cpp)
  pub fn set_params(&mut self, params: &SearchParams) {
    self.stop_and_wait();
    self.search.set_params(params);
  }
  /// upstream: AsyncBot::setParamsNoClearing (asyncbot.cpp)
  pub fn set_params_no_clearing(&mut self, params: &SearchParams) {
    self.stop_and_wait();
    self.search.set_params_no_clearing(params);
  }
  /// upstream: AsyncBot::setExternalPatternBonusTable (asyncbot.cpp)
  pub fn set_external_pattern_bonus_table(&mut self, table: Option<Box<PatternBonusTable>>) {
    self.stop_and_wait();
    self.search.set_external_pattern_bonus_table(table);
  }
  /// upstream: AsyncBot::setCopyOfExternalPatternBonusTable (asyncbot.cpp)
  pub fn set_copy_of_external_pattern_bonus_table(&mut self, table: &Option<Box<PatternBonusTable>>) {
    self.stop_and_wait();
    self.search.set_copy_of_external_pattern_bonus_table(table);
  }
  /// upstream: AsyncBot::setExternalEvalCache (asyncbot.cpp)
  pub fn set_external_eval_cache(&mut self, cache: &Option<Rc<EvalCacheTable>>) {
    self.stop_and_wait();
    self.search.set_external_eval_cache(cache);
  }
  /// upstream: AsyncBot::clearSearch (asyncbot.cpp)
  pub fn clear_search(&mut self) {
    self.stop_and_wait();
    self.search.clear_search();
  }
  /// upstream: AsyncBot::clearEvalCache (asyncbot.cpp)
  pub fn clear_eval_cache(&mut self) {
    self.stop_and_wait();
    if let Some(eval_cache) = &self.search.eval_cache {
      eval_cache.clear();
    }
  }

  /// upstream: AsyncBot::makeMove(Loc, Player) (asyncbot.cpp)
  //Updates position and preserves the relevant subtree of search
  //Will stop any ongoing search, waiting for a full stop.
  //If the move is not legal for the current player, returns false and does nothing, else returns true
  /// PORT NOTE: throws (Search::makeMove) -> Result.
  pub fn make_move(&mut self, move_loc: Loc, move_pla: Player) -> Result<bool, KataError> {
    self.stop_and_wait();
    self.search.make_move(move_loc, move_pla)
  }
  /// upstream: AsyncBot::makeMove(Loc, Player, bool) (asyncbot.cpp)
  pub fn make_move_prevent_encore(&mut self, move_loc: Loc, move_pla: Player, prevent_encore: bool) -> Result<bool, KataError> {
    self.stop_and_wait();
    self.search.make_move_prevent_encore(move_loc, move_pla, prevent_encore)
  }

  /// upstream: AsyncBot::isLegalTolerant (asyncbot.cpp)
  pub fn is_legal_tolerant(&self, move_loc: Loc, move_pla: Player) -> bool {
    self.search.is_legal_tolerant(move_loc, move_pla)
  }
  /// upstream: AsyncBot::isLegalStrict (asyncbot.cpp)
  pub fn is_legal_strict(&self, move_loc: Loc, move_pla: Player) -> bool {
    self.search.is_legal_strict(move_loc, move_pla)
  }

  /// upstream: AsyncBot::genMoveAsync(Player, int, const TimeControls&, double, const std::function<void(Loc,int,Search*)>&, const std::function<void()>&) (asyncbot.cpp)
  /// PORT NOTE: private and named per PORTING.md Appendix B: it only queues the job, which the next
  /// waitForSearchToEnd runs (no bot thread). The other genMoveAsync overloads are not ported.
  /// throws (setPlayerAndClearHistory) -> Result.
  fn gen_move_async_full(
    &mut self,
    move_pla: Player,
    search_id: i32,
    tc: &TimeControls,
    sf: f64,
    on_move: OnMoveCallback,
    on_search_begun: Option<SearchBegunCallback>,
  ) -> Result<(), KataError> {
    // PORT NOTE: lock removed.
    self.stop_and_wait_already_locked();
    testAssert!(!self.is_running);
    if self.is_killed.get() {
      return Ok(());
    }

    if move_pla != self.search.root_pla {
      self.search.set_player_and_clear_history(move_pla)?;
    }

    self.queued_search_id = search_id;
    self.queued_on_move = Some(on_move);
    self.is_running = true;
    self.is_pondering = false;
    self.should_stop_now.set(false);
    self.time_controls = *tc;
    self.search_factor = sf;
    self.analyze_callback_period = -1.0;
    self.analyze_first_callback_after = -1.0;
    self.analyze_callback = None;
    self.search_begun_callback = on_search_begun;
    // PORT NOTE: lock.unlock(); threadWaitingToSearch.notify_all(): the job runs in waitForSearchToEnd.
    Ok(())
  }

  /// upstream: AsyncBot::genMoveSynchronous(Player, const TimeControls&) (asyncbot.cpp)
  //Same as genMove, but waits directly for the move and returns it here.
  /// PORT NOTE: async (PORTING.md §11); throws -> Result.
  pub async fn gen_move_synchronous(&mut self, move_pla: Player, tc: &TimeControls) -> Result<Loc, KataError> {
    self.gen_move_synchronous_full(move_pla, tc, 1.0, None).await
  }

  /// upstream: AsyncBot::genMoveSynchronous(Player, const TimeControls&, double) (asyncbot.cpp)
  pub async fn gen_move_synchronous_search_factor(&mut self, move_pla: Player, tc: &TimeControls, sf: f64) -> Result<Loc, KataError> {
    self.gen_move_synchronous_full(move_pla, tc, sf, None).await
  }

  /// upstream: AsyncBot::genMoveSynchronous(Player, const TimeControls&, double, const std::function<void()>&) (asyncbot.cpp)
  /// PORT NOTE: name from PORTING.md Appendix B; `onSearchBegun` gets the Search (file header) and is
  /// None where upstream passes nullptr.
  pub async fn gen_move_synchronous_full(
    &mut self,
    move_pla: Player,
    tc: &TimeControls,
    sf: f64,
    on_search_begun: Option<SearchBegunCallback>,
  ) -> Result<Loc, KataError> {
    // PORT NOTE: the lambda writes the caller's local through a shared Cell (it is stored in the bot).
    let move_loc: Rc<Cell<Loc>> = Rc::new(Cell::new(Board::NULL_LOC));
    let move_loc_ref: Rc<Cell<Loc>> = move_loc.clone();
    let on_move: OnMoveCallback = Box::new(move |loc: Loc, search_id: i32, s: &mut Search| {
      testAssert!(search_id == 0);
      let _ = s;
      move_loc_ref.set(loc);
    });
    self.gen_move_async_full(move_pla, 0, tc, sf, on_move, on_search_begun)?;
    self.wait_for_search_to_end().await?;
    Ok(move_loc.get())
  }

  /// upstream: AsyncBot::genMoveAsyncAnalyze(Player, int, const TimeControls&, double, const std::function<void(Loc,int,Search*)>&, double, double, const std::function<void(const Search*)>&, const std::function<void()>&) (asyncbot.cpp)
  /// PORT NOTE: private and named per PORTING.md Appendix B: it only queues the job (see
  /// gen_move_async_full). The 8-parameter overload is not ported. throws -> Result.
  fn gen_move_async_analyze_full(
    &mut self,
    move_pla: Player,
    search_id: i32,
    tc: &TimeControls,
    sf: f64,
    on_move: OnMoveCallback,
    callback_period: f64,
    first_callback_after: f64,
    callback: AnalyzeCallback,
    on_search_begun: Option<SearchBegunCallback>,
  ) -> Result<(), KataError> {
    // PORT NOTE: lock removed.
    self.stop_and_wait_already_locked();
    testAssert!(!self.is_running);
    if self.is_killed.get() {
      return Ok(());
    }

    if move_pla != self.search.root_pla {
      self.search.set_player_and_clear_history(move_pla)?;
    }

    self.queued_search_id = search_id;
    self.queued_on_move = Some(on_move);
    self.is_running = true;
    self.is_pondering = false;
    self.should_stop_now.set(false);
    self.time_controls = *tc;
    self.search_factor = sf;
    self.analyze_callback_period = callback_period;
    self.analyze_first_callback_after = first_callback_after;
    self.analyze_callback = Some(callback);
    self.search_begun_callback = on_search_begun;
    // PORT NOTE: lock.unlock(); threadWaitingToSearch.notify_all(): the job runs in waitForSearchToEnd.
    Ok(())
  }

  /// upstream: AsyncBot::genMoveSynchronousAnalyze(Player, const TimeControls&, double, double, double, const std::function<void(const Search*)>&) (asyncbot.cpp)
  /// PORT NOTE: async (PORTING.md §11); throws -> Result.
  pub async fn gen_move_synchronous_analyze(
    &mut self,
    move_pla: Player,
    tc: &TimeControls,
    sf: f64,
    callback_period: f64,
    first_callback_after: f64,
    callback: AnalyzeCallback,
  ) -> Result<Loc, KataError> {
    self.gen_move_synchronous_analyze_full(move_pla, tc, sf, callback_period, first_callback_after, callback, None).await
  }

  /// upstream: AsyncBot::genMoveSynchronousAnalyze(Player, const TimeControls&, double, double, double, const std::function<void(const Search*)>&, const std::function<void()>&) (asyncbot.cpp)
  /// PORT NOTE: name from PORTING.md Appendix B; async; throws -> Result.
  pub async fn gen_move_synchronous_analyze_full(
    &mut self,
    move_pla: Player,
    tc: &TimeControls,
    sf: f64,
    callback_period: f64,
    first_callback_after: f64,
    callback: AnalyzeCallback,
    on_search_begun: Option<SearchBegunCallback>,
  ) -> Result<Loc, KataError> {
    // PORT NOTE: the lambda writes the caller's local through a shared Cell (it is stored in the bot).
    let move_loc: Rc<Cell<Loc>> = Rc::new(Cell::new(Board::NULL_LOC));
    let move_loc_ref: Rc<Cell<Loc>> = move_loc.clone();
    let on_move: OnMoveCallback = Box::new(move |loc: Loc, search_id: i32, s: &mut Search| {
      testAssert!(search_id == 0);
      let _ = s;
      move_loc_ref.set(loc);
    });
    self.gen_move_async_analyze_full(move_pla, 0, tc, sf, on_move, callback_period, first_callback_after, callback, on_search_begun)?;
    self.wait_for_search_to_end().await?;
    Ok(move_loc.get())
  }

  /// upstream: AsyncBot::waitForSearchToEnd (asyncbot.cpp)
  //Wait for the ongoing search to end, if any search is running.
  /// PORT NOTE: `while(isRunning) userWaitingForStop.wait(lock)` -> runs the queued job (the bot
  /// thread's loop body) to its end. Async; the job's error (upstream: logged by the dying bot thread)
  /// is returned.
  pub async fn wait_for_search_to_end(&mut self) -> Result<(), KataError> {
    if self.is_running {
      self.internal_search_thread_loop().await?;
    }
    Ok(())
  }

  /// upstream: AsyncBot::stopAndWait (asyncbot.cpp)
  //Signal an ongoing genMove or ponder to stop as soon as possible, and wait for the stop to happen.
  //Safe to call even if nothing is running.
  /// PORT NOTE: synchronous: with no bot thread, nothing can be running while the caller holds
  /// `&mut self` (a queued job is always awaited right after it is queued).
  pub fn stop_and_wait(&mut self) {
    self.should_stop_now.set(true);
    testAssert!(!self.is_running);
  }

  /// upstream: AsyncBot::stopAndWaitAlreadyLocked (asyncbot.cpp)
  /// PORT NOTE: lock removed; as stop_and_wait.
  fn stop_and_wait_already_locked(&mut self) {
    self.should_stop_now.set(true);
    testAssert!(!self.is_running);
  }

  /// upstream: AsyncBot::stopWithoutWait (asyncbot.cpp)
  //Same, but does NOT wait for the stop. Also safe to call even if nothing is running.
  //Does not lock anything, so even safe to call from inside callbacks from this API.
  pub fn stop_without_wait(&self) {
    self.should_stop_now.set(true);
  }

  /// upstream: AsyncBot::setKilled (asyncbot.cpp)
  //Call this to permanently kill this bot and prevent future search.
  pub fn set_killed(&self) {
    // PORT NOTE: lock removed; threadWaitingToSearch.notify_all() has no waiter (no bot thread).
    self.is_killed.set(true);
    self.should_stop_now.set(true);
  }

  /// upstream: AsyncBot::internalSearchThreadLoop (asyncbot.cpp)
  /// PORT NOTE: one pass of the bot thread's loop, for the job genMove*Async queued (no bot thread, see
  /// the file header). `while(!isRunning && !isKilled) wait` becomes returning when nothing is queued;
  /// the analyze `callbackLoop` thread becomes the concurrent task of runWholeSearch. An error is logged
  /// as upstream's searchThreadLoop does and returned (upstream's waiter would then block forever).
  async fn internal_search_thread_loop(&mut self) -> Result<(), KataError> {
    if !self.is_running && !self.is_killed.get() {
      return Ok(());
    }
    if self.is_killed.get() {
      self.is_running = false;
      self.is_pondering = false;
      // PORT NOTE: userWaitingForStop.notify_all(): the waiter is our caller.
      return Ok(());
    }

    let pondering: bool = self.is_pondering;
    let tc: TimeControls = self.time_controls;
    let mut callback_period: f64 = self.analyze_callback_period;
    let mut first_callback_after: f64 = self.analyze_first_callback_after;
    //Make local copies just in case, to simplify thread reasoning for the member fields
    let analyze_callback_local: Option<AnalyzeCallback> = self.analyze_callback.clone();
    let search_begun_callback_local: Option<SearchBegunCallback> = self.search_begun_callback.clone();
    // PORT NOTE: lock.unlock() removed.

    //Make sure we don't feed in absurdly large numbers, this seems to cause wait_for to hang.
    //For a long period, just don't do callbacks.
    if callback_period >= 10000000.0 {
      callback_period = -1.0;
    }
    if first_callback_after >= 10000000.0 {
      first_callback_after = -1.0;
      callback_period = -1.0;
    }

    //Handle search begun and analysis callback loops
    let using_callback_loop: bool = (first_callback_after >= 0.0 || callback_period >= 0.0) && analyze_callback_local.is_some();

    // PORT NOTE: isSearchBegun / callbackLoopWaitingForSearchBegun are gone: runWholeSearch starts the
    // callback task after searchBegun.
    let search_begun = |search: &Search| {
      if let Some(search_begun_callback_local) = &search_begun_callback_local {
        search_begun_callback_local(search);
      }
    };

    // PORT NOTE: `std::thread callbackLoopThread(callbackLoop)` -> the concurrent task of runWholeSearch;
    // callbackLoopShouldStop + notify + join -> runWholeSearch drops the task when its search threads end.
    let callback_loop = higher_ranked_task(|search: &Search| {
      let analyze_callback_local: AnalyzeCallback = analyze_callback_local.clone().expect("usingCallbackLoop implies a callback");
      Box::pin(callback_loop(search, callback_period, first_callback_after, analyze_callback_local))
    });
    let concurrent_task: Option<ConcurrentTask<'_>> =
      if using_callback_loop { Some(&callback_loop) } else { None };

    let should_stop_now: Rc<Cell<bool>> = self.should_stop_now.clone();
    let should_stop_early = move || -> bool { should_stop_now.get() /*acquire*/ };

    let result: Result<(), KataError> =
      self.search.run_whole_search_full(Some(&search_begun), Some(&should_stop_early), pondering, &tc, self.search_factor, concurrent_task).await;
    if let Err(e) = result {
      // upstream: searchThreadLoop's catch (the bot thread then exits).
      log::info!("{}", "ERROR: Async bot thread failed: ".to_string() + e.what());
      self.is_running = false;
      self.is_pondering = false;
      return Err(e);
    }
    let move_loc: Loc = self.search.get_chosen_move_loc();

    // PORT NOTE: lock.lock() removed.
    //Call queuedOnMove under the lock just in case
    if let Some(queued_on_move) = self.queued_on_move.as_mut() {
      queued_on_move(move_loc, self.queued_search_id, &mut self.search);
    }
    self.is_running = false;
    self.is_pondering = false;
    // PORT NOTE: userWaitingForStop.notify_all(): the waiter is our caller.
    Ok(())
  }
}

/// PORT NOTE: not upstream: makes Rust infer the closure's higher-ranked signature (the task borrows the
/// Search for as long as runWholeSearch lends it).
fn higher_ranked_task<F>(f: F) -> F
where
  F: for<'s> Fn(&'s Search) -> LocalBoxFuture<'s, Result<(), KataError>>,
{
  f
}

/// upstream: the `callbackLoop` lambda of AsyncBot::internalSearchThreadLoop (asyncbot.cpp)
/// PORT NOTE: runs as runWholeSearch's concurrent task (see internal_search_thread_loop): it starts after
/// searchBegun, so the `while(!isSearchBegun)` wait is gone, and it is dropped when the search threads end
/// instead of checking callbackLoopShouldStop. `callbackLoopWaiting.wait_for(periodToWait)` is the injected
/// `port::executor::sleep`. The lock around the callback is removed.
async fn callback_loop(search: &Search, callback_period: f64, first_callback_after: f64, analyze_callback_local: AnalyzeCallback) -> Result<(), KataError> {
  let mut period_to_wait: f64 = first_callback_after;
  let mut got_report_yet: bool = false;
  loop {
    if period_to_wait < 0.0 {
      return Ok(());
    }

    sleep(period_to_wait).await;

    // If the search hasn't yet produced any reportable root values because firstCallbackAfter
    // elapsed before the root NN eval finished, skip callback and re-wait.
    // Use a mild exponential backoff to avoid overly eager busy looping for tiny waits, just in case.
    if !got_report_yet {
      let mut unused_vals: ReportedSearchValues = ReportedSearchValues::new();
      if !search.get_root_values(&mut unused_vals) {
        period_to_wait = period_to_wait * 1.25 + 0.001;
        let cap: f64 = cpp::cpp_max(first_callback_after, if callback_period < 0.0 { 1.0 } else { callback_period });
        period_to_wait = cpp::cpp_min(period_to_wait, cap);
        continue;
      }
      got_report_yet = true;
      period_to_wait = callback_period;
    }
    analyze_callback_local(search)?;
  }
}
