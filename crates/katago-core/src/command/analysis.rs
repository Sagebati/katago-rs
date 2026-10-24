// Ported from KataGo cpp/command/analysis.cpp @ ba938676d7f42d70950b3a535af2466fb642008c
//
// `MainCmds::analysis`, the JSON-lines analysis engine, as a library function over injected host
// services (port-manifest.toml notes, PORTING.md §11):
// - No argv parsing (TCLAP, `KataGoCommandLine`: cpp/command/commandline.cpp is replaced by the host's
//   flags) and no file IO: the host builds the ConfigParser (`cmd.getConfig(cfg)`), hands over the
//   parsed arguments (`AnalysisArgs`), the input lines (`getline(cin, line)` -> an async Stream), the
//   response sink (`cout << message << endl`), stderr (`cerr`), and the NN backend (Setup's
//   `NNEvaluatorHost`, D6).
// - Threads -> virtual threads, joined inside `analysis`: the request loop (upstream's main thread),
//   one analysis loop per `numAnalysisThreads` (each owning its AsyncBot) and the write loop.
//   `ThreadSafeQueue<string*>` / `ThreadSafePriorityQueue` -> port::executor::AsyncQueue /
//   AsyncPriorityQueue; `openRequestsMutex` removed (RefCell). An `AnalyzeRequest` is shared
//   (`Rc`) between the request loop and an analysis loop; its `status` is a Cell (OWNERSHIP.tsv).
// - The lambdas of MainCmds::analysis become fns over `AnalysisContext`, the struct holding what they
//   capture (PORT NOTE: captures passed explicitly). The body of the request loop's `while(getline)` is
//   `handle_request_line` (`continue` -> `return Ok(())`).
// - Logger -> log crate (`logger.write(s)` -> `log::info!`); the Logger constructor's config reads are
//   kept (`logger_config_reads`) so used keys and `logToStderr` decisions are unchanged, and its
//   outputs (stdout / stderr / timestamps / logFile / logDir / logDirDated) are the host's
//   (`AnalysisHost::init_logger`, called where the constructor opens its file).
// - Input lines are bytes (`std::string` from getline): JSON parsing is `nlohmann::parse_bytes`, and an
//   unparseable line whose error text is not valid UTF-8 makes reportError's dump() throw
//   type_error.316 out of the request loop, as upstream.
// - `Version::` strings (main.cpp, [[ignored]]) are the constants below: [upstream].tag and commit.
// - Errors: an exception escaping the request loop or an analysis loop kills the process upstream
//   (logThreadUncaught logs it, sleeps 5 s while the other threads keep running, rethrows ->
//   std::terminate). Here the failing loop logs the same line and sleeps 5 s (port::executor::sleep),
//   then the process counts as terminated: nothing more is written, every loop stops, and `analysis`
//   returns the error without its normal closing log lines. The host ends the process (katago-cli:
//   std::terminate's message and abort()).
// - Virtual-thread scheduling (PORTING.md §11): threads interleave only at awaits, so where upstream's
//   output order is a race, the port picks the interleaving upstream shows in practice: after a line
//   enqueues requests the request loop yields (an idle analysis thread pops the first request as soon as
//   it is queued), and a popped request starts its search only once the request loop is waiting for
//   input (upstream's request thread handles the rest of a burst of lines while the AsyncBot's search
//   thread wakes up). See `request_loop` and `analysis_loop`.
// - Avoid-SGF pattern bonus tables: Setup::loadAvoidSgfPatternBonusTables reads their keys as upstream but
//   cannot load SGF files (IO/M8), so a fully configured set is an error.

use std::cell::Cell;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::future::Future;
use std::pin::Pin;
use std::rc::Rc;
use std::task::Poll;

use futures::Stream;
use futures::StreamExt;
use serde_json::Value;

use crate::core::config_parser::ConfigParser;
use crate::core::global::Global;
use crate::core::global::KataError;
use crate::core::rand::Rand;
use crate::game::board::get_opp;
use crate::game::board::Board;
use crate::game::board::Loc;
use crate::game::board::Location;
use crate::game::board::Move;
use crate::game::board::Player;
use crate::game::board::PlayerIO;
use crate::game::board::C_EMPTY;
use crate::game::board::P_BLACK;
use crate::game::board::P_WHITE;
use crate::game::boardhistory::BoardHistory;
use crate::game::rules::Rules;
use crate::neuralnet::nneval::NNEvaluator;
use crate::neuralnet::nninputs::NNPos;
use crate::neuralnet::nninputs::ScoreValue;
use crate::port::cpp;
use crate::port::executor::AsyncPriorityQueue;
use crate::port::executor::AsyncQueue;
use crate::port::executor::WakerList;
use crate::port::nlohmann;
use crate::program::playutils::PlayUtils;
use crate::program::setup::NNEvaluatorHost;
use crate::program::setup::Setup;
use crate::search::asyncbot::AnalyzeCallback;
use crate::search::asyncbot::AsyncBot;
use crate::search::asyncbot::AsyncBotStopHandle;
use crate::search::asyncbot::SearchBegunCallback;
use crate::search::evalcache::EvalCacheTable;
use crate::search::patternbonustable::PatternBonusTable;
use crate::search::search::Search;
use crate::search::searchparams::SearchParams;
use crate::search::timecontrols::TimeControls;
use crate::testAssert;

/// upstream: Version::getKataGoVersion (main.cpp)
/// PORT NOTE: main.cpp is [[ignored]]; the version is [upstream].tag (port-manifest.toml).
pub const KATAGO_VERSION: &str = "1.16.5";
/// upstream: Version::getKataGoVersionForHelp (main.cpp)
pub const KATAGO_VERSION_FOR_HELP: &str = "KataGo v1.16.5";
/// upstream: Version::getGitRevision (main.cpp)
/// PORT NOTE: the revision a release build of the tag reports: [upstream].commit.
pub const GIT_REVISION: &str = "ba938676d7f42d70950b3a535af2466fb642008c";

/// upstream: AnalyzeRequest (analysis.cpp)
pub struct AnalyzeRequest {
  pub internal_id: i64,
  pub id: String,
  pub turn_number: i32,
  pub priority: i64,

  pub board: Board,
  pub hist: BoardHistory,
  pub next_pla: Player,

  pub params: SearchParams,
  pub perspective: Player,
  pub analysis_pv_len: i32,
  pub include_ownership: bool,
  pub include_ownership_stdev: bool,
  pub include_moves_ownership: bool,
  pub include_moves_ownership_stdev: bool,
  pub include_policy: bool,
  pub include_pv_visits: bool,
  pub include_no_result_value: bool,

  pub report_during_search: bool,
  pub report_during_search_every: f64,
  pub first_report_during_search_after: f64,

  pub avoid_move_until_by_loc_black: Vec<i32>,
  pub avoid_move_until_by_loc_white: Vec<i32>,

  //Starts with STATUS_IN_QUEUE.
  //Thread that grabs it from queue it changes it to STATUS_POPPED
  //Once search is fully started thread sticks in its own thread index
  //At any point it may change to STATUS_TERMINATED.
  //If it ever gets to STATUS_POPPED or later, then the analysis thread is reponsible for writing the result, else the api thread is
  /// PORT NOTE: std::atomic<int> -> Cell (OWNERSHIP.tsv).
  pub status: Cell<i32>,
}

impl AnalyzeRequest {
  /// upstream: AnalyzeRequest::STATUS_IN_QUEUE, STATUS_POPPED, STATUS_TERMINATED (analysis.cpp)
  pub const STATUS_IN_QUEUE: i32 = -1;
  pub const STATUS_POPPED: i32 = -2;
  pub const STATUS_TERMINATED: i32 = -3;

  /// upstream: AnalyzeRequest's implicit default constructor (analysis.cpp)
  /// PORT NOTE: the uninitialized scalars are zero; Board / BoardHistory / SearchParams default-construct.
  pub fn new() -> AnalyzeRequest {
    AnalyzeRequest {
      internal_id: 0,
      id: String::new(),
      turn_number: 0,
      priority: 0,
      board: Board::new(),
      hist: BoardHistory::new(),
      next_pla: C_EMPTY,
      params: SearchParams::new(),
      perspective: C_EMPTY,
      analysis_pv_len: 0,
      include_ownership: false,
      include_ownership_stdev: false,
      include_moves_ownership: false,
      include_moves_ownership_stdev: false,
      include_policy: false,
      include_pv_visits: false,
      include_no_result_value: false,
      report_during_search: false,
      report_during_search_every: 0.0,
      first_report_during_search_after: 0.0,
      avoid_move_until_by_loc_black: Vec::new(),
      avoid_move_until_by_loc_white: Vec::new(),
      status: Cell::new(0),
    }
  }
}

/// The command-line arguments of `katago analysis`, as TCLAP would have parsed them.
/// PORT NOTE: not upstream (cpp/command/commandline.cpp is replaced by the host's flags).
#[derive(Clone, Debug, Default)]
pub struct AnalysisArgs {
  /// `-model` (cmd.getModelFile(); the host resolves defaults)
  pub model_file: String,
  /// `-human-model` ("" if not given)
  pub human_model_file: String,
  /// `-analysis-threads` given?
  pub num_analysis_threads_cmdline_specified: bool,
  /// `-analysis-threads` value
  pub num_analysis_threads_cmdline: i32,
  /// `-quit-without-waiting`
  pub quit_without_waiting: bool,
  /// every `-override-config` value (for cmd.logOverrides; the host already applied them to the cfg)
  pub override_configs: Vec<String>,
}

/// The settings the `Logger` constructor reads from the config (logger.cpp), handed to the host, which
/// implements the Logger's outputs.
/// PORT NOTE: not upstream (file header).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LoggerSettings {
  /// `logToStdout` (log lines also go to stdout)
  pub log_to_stdout: bool,
  /// `logToStderr`
  pub log_to_stderr: bool,
  /// `logTimeStamp` (`DateTime::timeFormat` prefix instead of ": ")
  pub log_time: bool,
  /// `addFile(cfg.getString("logFile"))`: open for append; on failure the Logger writes
  /// "WARNING: could not open file for logging: <IOError>" to the log and stderr and goes on.
  pub log_file: Option<String>,
  /// `MakeDir::make(logDir)` (StringError "Error creating directory: <dir>" on failure, uncaught),
  /// then `addFile(logDir + "/" + compact date-time + "-" + random hex + ".log")`.
  pub log_dir: Option<String>,
  /// As log_dir with an extra date-time subdirectory.
  pub log_dir_dated: Option<String>,
}

/// What MainCmds::analysis gets from the process and the build.
/// PORT NOTE: not upstream (file header).
pub struct AnalysisHost<'a> {
  pub args: AnalysisArgs,
  /// `getline(cin, line)`: the input lines as bytes (a std::string), without their '\n'; the stream
  /// ends at end of input. A line that is Ready without waiting counts as already available (part of a
  /// burst: see `request_loop`).
  pub input: &'a mut (dyn Stream<Item = Vec<u8>> + Unpin),
  /// `cout << message << endl`: one response line (the sink adds the newline and flushes).
  pub write_stdout: Box<dyn Fn(&str)>,
  /// `cerr << text`: `text` includes its newlines.
  pub write_stderr: Box<dyn Fn(&str)>,
  /// The NN backend, the model files and the executor (Setup::initializeNNEvaluator).
  pub nn_host: &'a dyn NNEvaluatorHost,
  /// The `Logger` constructor's outputs (see LoggerSettings), called once where it opens its file, before
  /// its config header is logged. An Err is the constructor's (uncaught) exception.
  pub init_logger: InitLoggerFn,
}

/// AnalysisHost::init_logger.
pub type InitLoggerFn = Box<dyn Fn(&LoggerSettings) -> Result<(), KataError>>;

/// The locals of MainCmds::analysis that its lambdas capture, shared by the virtual threads.
/// PORT NOTE: not upstream (file header).
pub struct AnalysisContext {
  cfg: ConfigParser,
  human_model_file: String,
  log_all_requests: bool,
  log_all_responses: bool,
  log_errors_and_warnings: bool,
  log_search_info: bool,
  warn_unused_fields: bool,
  default_params: SearchParams,
  default_perspective: Player,
  analysis_pv_len: i32,
  assume_multiple_starting_black_moves_are_handicap: bool,
  prevent_encore: bool,
  nn_eval: Rc<NNEvaluator>,
  human_eval: Option<Rc<NNEvaluator>>,
  expected_keys: BTreeSet<String>,
  to_write_queue: AsyncQueue<String>,
  write_stdout: Box<dyn Fn(&str)>,
  to_analyze_queue: AsyncPriorityQueue<(i64, i64), Rc<AnalyzeRequest>>,
  num_requests_so_far: Cell<i64>, // Used as tie breaker for requests with same priority
  internal_id_counter: Cell<i64>, // Counter for internalId on requests.
  //Open requests, keyed by internalId, mutexed by the mutex
  /// PORT NOTE: openRequestsMutex removed (RefCell); OWNERSHIP.tsv: BTreeMap<i64, Rc<AnalyzeRequest>>.
  open_requests: RefCell<BTreeMap<i64, Rc<AnalyzeRequest>>>,
  eval_cache: Option<Rc<EvalCacheTable>>,
  bots: Vec<AsyncBotStopHandle>,
  /// PORT NOTE: set when a loop's exception has terminated the process upstream (after
  /// logThreadUncaught's 5 s sleep): nothing more is written and every loop stops.
  loop_failed: Cell<bool>,
  loop_failed_notify: WakerList,
  /// PORT NOTE: scheduling (file header): true while the request loop waits for input that is not
  /// immediately available, and once it has ended; `request_loop_idle_notify` is notified when it
  /// becomes true.
  request_loop_waiting: Cell<bool>,
  request_loop_idle_notify: WakerList,
}

/// upstream: the `loadParams` lambda of MainCmds::analysis (analysis.cpp)
/// PORT NOTE: captures (`humanModelFile`) passed explicitly; throws -> Result.
fn load_params(human_model_file: &str, config: &mut ConfigParser, params: &mut SearchParams, perspective: &mut Player, default_perspective: Player) -> Result<(), KataError> {
  let has_human_model: bool = human_model_file != "";
  *params = Setup::load_single_params_has_human_model(config, Setup::SETUP_FOR_ANALYSIS, has_human_model)?;
  *perspective = Setup::parse_report_analysis_winrates(config, default_perspective)?;
  //Set a default for conservativePass that differs from matches or selfplay
  if !config.contains("conservativePass") {
    params.conservative_pass = true;
  }
  Ok(())
}

/// upstream: Logger::Logger(ConfigParser*, bool, bool, bool, bool) (logger.cpp), as MainCmds::analysis constructs it
/// (`Logger logger(&cfg, logToStdoutDefault, logToStderrDefault)`: logTimeDefault = true, logConfigContents = true)
/// PORT NOTE: the Logger is replaced by the log crate; the constructor's config reads are kept so the used
/// keys match upstream, its outputs (and addFile / MakeDir) are the host's `init_logger`, called where the
/// constructor sets up its file, and the config header is logged last, as upstream. Returns
/// `logger.isLoggingToStderr()`. Throws -> Result.
fn logger_config_reads(
  cfg: &mut ConfigParser,
  log_to_stdout_default: bool,
  log_to_stderr_default: bool,
  init_logger: &dyn Fn(&LoggerSettings) -> Result<(), KataError>,
) -> Result<bool, KataError> {
  let log_time_default: bool = true;
  let log_config_contents: bool = true;
  let mut settings: LoggerSettings =
    LoggerSettings { log_to_stdout: log_to_stdout_default, log_to_stderr: log_to_stderr_default, log_time: log_time_default, ..LoggerSettings::default() };
  let mut header: String = String::new();
  // Also avoid logging if cfg specifies it
  if log_config_contents && !(cfg.contains("logConfigContents") && !cfg.get_bool("logConfigContents")?) {
    header = "Running with following config:\n".to_string() + &cfg.get_all_key_vals();
  }
  if cfg.contains("logToStdout") {
    settings.log_to_stdout = cfg.get_bool("logToStdout")?;
  }

  if cfg.contains("logToStderr") {
    settings.log_to_stderr = cfg.get_bool("logToStderr")?;
  }

  if cfg.contains("logTimeStamp") {
    settings.log_time = cfg.get_bool("logTimeStamp")?;
  }

  if (cfg.contains("logFile") as i32) + (cfg.contains("logDir") as i32) + (cfg.contains("logDirDated") as i32) > 1 {
    return Err(KataError::StringError("Cannot specify more than one of logFile and logDir and logDirDated in config".to_string()));
  } else if cfg.contains("logFile") {
    settings.log_file = Some(cfg.get_string("logFile")?);
  } else if cfg.contains("logDir") {
    settings.log_dir = Some(cfg.get_string("logDir")?);
  } else if cfg.contains("logDirDated") {
    settings.log_dir_dated = Some(cfg.get_string("logDirDated")?);
  }
  // PORT NOTE: addFile / MakeDir::make, and the stream setup, by the host.
  init_logger(&settings)?;

  if !header.is_empty() {
    log::info!("{}", header);
  }
  Ok(settings.log_to_stderr)
}

impl AnalysisContext {
  /// upstream: the `pushToWrite` lambda of MainCmds::analysis (analysis.cpp)
  fn push_to_write(&self, s: String) {
    let suc: bool = self.to_write_queue.force_push(s);
    if !suc {
      // delete s: dropped by force_push
    }
  }

  /// upstream: the `reportError` lambda of MainCmds::analysis (analysis.cpp)
  fn report_error(&self, s: &str) {
    let mut ret: Value = Value::Null;
    ret["error"] = Value::from(s);
    self.push_to_write(nlohmann::dump(&ret));
    if self.log_errors_and_warnings {
      log::info!("{}", "Error: ".to_string() + &nlohmann::dump(&ret));
    }
  }

  /// upstream: the `reportError` lambda of MainCmds::analysis (analysis.cpp), for a `const string& s`
  /// holding any bytes
  /// PORT NOTE: `ret["error"] = s; ret.dump()` throws type_error.316 when `s` is not valid UTF-8
  /// (serde_json strings cannot hold it, so the dump of the string is checked first); throws -> Result.
  fn report_error_bytes(&self, s: &[u8]) -> Result<(), KataError> {
    nlohmann::dump_string_bytes(s)?;
    self.report_error(std::str::from_utf8(s).expect("dump_string_bytes accepted it: valid UTF-8"));
    Ok(())
  }

  /// upstream: the `reportErrorForId` lambda of MainCmds::analysis (analysis.cpp)
  fn report_error_for_id(&self, id: &str, field: &str, s: &str) {
    let mut ret: Value = Value::Null;
    ret["id"] = Value::from(id);
    ret["field"] = Value::from(field);
    ret["error"] = Value::from(s);
    self.push_to_write(nlohmann::dump(&ret));
    if self.log_errors_and_warnings {
      log::info!("{}", "Error: ".to_string() + &nlohmann::dump(&ret));
    }
  }

  /// upstream: the `reportWarningForId` lambda of MainCmds::analysis (analysis.cpp)
  fn report_warning_for_id(&self, id: &str, field: &str, s: &str) {
    let mut ret: Value = Value::Null;
    ret["id"] = Value::from(id);
    ret["field"] = Value::from(field);
    ret["warning"] = Value::from(s);
    self.push_to_write(nlohmann::dump(&ret));
    if self.log_errors_and_warnings {
      log::info!("{}", "Warning: ".to_string() + &nlohmann::dump(&ret));
    }
  }

  /// upstream: the `reportNoAnalysis` lambda of MainCmds::analysis (analysis.cpp)
  //Report analysis for which we don't actually have results. This is used when something is user-terminated before being actually
  //analyzed properly. Only used outside of search too
  fn report_no_analysis(&self, request: &AnalyzeRequest) {
    let mut ret: Value = Value::Null;
    ret["id"] = Value::from(request.id.as_str());
    ret["turnNumber"] = Value::from(request.turn_number);
    ret["isDuringSearch"] = Value::from(false);
    ret["noResults"] = Value::from(true);
    self.push_to_write(nlohmann::dump(&ret));
  }

  /// upstream: the `reportAnalysis` lambda of MainCmds::analysis (analysis.cpp)
  //Returns false if no analysis was reportable due to there being no root node or search results.
  /// PORT NOTE: throws (getAnalysisJson) -> Result.
  fn report_analysis(&self, request: &AnalyzeRequest, search: &Search, is_during_search: bool) -> Result<bool, KataError> {
    let mut ret: Value = Value::Null;
    ret["id"] = Value::from(request.id.as_str());
    ret["turnNumber"] = Value::from(request.turn_number);
    ret["isDuringSearch"] = Value::from(is_during_search);

    let success: bool = search.get_analysis_json(
      request.perspective,
      request.analysis_pv_len,
      self.prevent_encore,
      request.include_policy,
      request.include_ownership,
      request.include_ownership_stdev,
      request.include_moves_ownership,
      request.include_moves_ownership_stdev,
      request.include_pv_visits,
      request.include_no_result_value,
      &mut ret,
    )?;

    if success {
      self.push_to_write(nlohmann::dump(&ret));
    }
    Ok(success)
  }

  /// upstream: the `terminateRequest` lambda of MainCmds::analysis (analysis.cpp)
  fn terminate_request(&self, request: &AnalyzeRequest) {
    //Firstly, flag the request as terminated
    let prev_status: i32 = request.status.replace(AnalyzeRequest::STATUS_TERMINATED) /*acq_rel*/;
    //Already terminated? Nothing to do.
    if prev_status == AnalyzeRequest::STATUS_TERMINATED {
    }
    //No thread claimed it, so it's up to us to write the result
    else if prev_status == AnalyzeRequest::STATUS_IN_QUEUE {
      self.report_no_analysis(request);
    }
    //A thread popped it. That thread will notice that it's terminated once it tries to put its thread idx in, so we need not do anything.
    else if prev_status == AnalyzeRequest::STATUS_POPPED {
    }
    //A thread started searching it and put its thread idx in
    else {
      testAssert!(prev_status >= 0);
      //We've already set the above status to terminated so when the thread terminates due to our killing it below, it will see this.
      //Or else the thread has already done so, in which case it's already properly written a result, also fine.
      let thread_idx: i32 = prev_status;
      //Terminate it by thread index
      self.bots[thread_idx as usize].stop_without_wait();
    }
  }
}

/// upstream: the `writeLoop` lambda of MainCmds::analysis (analysis.cpp)
/// PORT NOTE: a virtual thread (PORTING.md §11); `cout << *message << endl` is the host's sink.
async fn write_loop(ctx: &AnalysisContext) {
  loop {
    let message: String = match ctx.to_write_queue.wait_pop().await {
      None => break,
      Some(m) => m,
    };
    if ctx.loop_failed.get() {
      // PORT NOTE: the process has terminated upstream (file header): nothing more is written.
      continue;
    }
    (ctx.write_stdout)(&message);
    if ctx.log_all_responses {
      log::info!("{}", "Response: ".to_string() + &message);
    }
    // delete message: dropped
  }
}

/// upstream: the `analysisLoop` lambda of MainCmds::analysis (analysis.cpp)
/// PORT NOTE: a virtual thread owning its bot (PORTING.md §11); throws -> Result.
async fn analysis_loop(ctx: &Rc<AnalysisContext>, bot: &mut AsyncBot, thread_idx: i32) -> Result<(), KataError> {
  loop {
    let analysis_item: Option<((i64, i64), Rc<AnalyzeRequest>)> = ctx.to_analyze_queue.wait_pop().await;
    let request: Rc<AnalyzeRequest> = match analysis_item {
      None => break,
      Some((_, r)) => r,
    };
    let mut expected: i32 = AnalyzeRequest::STATUS_IN_QUEUE;
    //If it's already terminated, then there's nothing for us to do
    if !cpp::cell_cas(&request.status, &mut expected, AnalyzeRequest::STATUS_POPPED) /*acq_rel*/ {
      testAssert!(expected == AnalyzeRequest::STATUS_TERMINATED);
    }
    //Else, the request is live and we marked it as popped
    else {
      bot.set_position(request.next_pla, &request.board, &request.hist)?;
      bot.set_always_include_owner_map(
        request.include_ownership || request.include_ownership_stdev || request.include_moves_ownership || request.include_moves_ownership_stdev,
      );
      bot.set_params(&request.params);
      bot.set_avoid_move_until_by_loc(&request.avoid_move_until_by_loc_black, &request.avoid_move_until_by_loc_white);

      let pla: Player = request.next_pla;
      let search_factor: f64 = 1.0;

      // PORT NOTE: scheduling (file header). genMoveSynchronous hands the search to the AsyncBot's own
      // thread; upstream's request thread processes the rest of a burst of input lines before that
      // thread has begun the search (so a terminate in the same burst finds this request popped, not
      // yet searching). The port's search begins once the request loop waits for input.
      ctx.request_loop_idle_notify.wait_until(|| ctx.request_loop_waiting.get()).await;

      //Handle termination between the time we pop and the search starts
      let on_search_begun: SearchBegunCallback = {
        let request: Rc<AnalyzeRequest> = request.clone();
        let bot_stop: AsyncBotStopHandle = bot.get_stop_handle();
        Rc::new(move |_search: &Search| {
          //Try to record that we're handling this request and indicate that the search is started by this thread
          let mut expected2: i32 = AnalyzeRequest::STATUS_POPPED;
          //If it was terminated, then stop our search
          if !cpp::cell_cas(&request.status, &mut expected2, thread_idx) /*acq_rel*/ {
            testAssert!(expected2 == AnalyzeRequest::STATUS_TERMINATED);
            bot_stop.stop_without_wait();
          }
        })
      };

      if request.report_during_search {
        let callback: AnalyzeCallback = {
          let request: Rc<AnalyzeRequest> = request.clone();
          let ctx: Rc<AnalysisContext> = ctx.clone();
          Rc::new(move |search: &Search| -> Result<(), KataError> {
            let is_during_search: bool = true;
            ctx.report_analysis(&request, search, is_during_search)?;
            Ok(())
          })
        };
        bot
          .gen_move_synchronous_analyze_full(
            pla,
            &TimeControls::new(),
            search_factor,
            request.report_during_search_every,
            request.first_report_during_search_after,
            callback,
            Some(on_search_begun),
          )
          .await?;
      } else {
        bot.gen_move_synchronous_full(pla, &TimeControls::new(), search_factor, Some(on_search_begun)).await?;
      }

      if ctx.log_search_info {
        let mut sout: String = String::new();
        PlayUtils::print_genmove_log(&mut sout, bot.get_search(), &ctx.nn_eval, Board::NULL_LOC, f64::NAN, request.perspective, false)?;
        log::info!("{}", sout);
      }

      {
        let is_during_search: bool = false;
        let search: &Search = bot.get_search();
        let analysis_written: bool = ctx.report_analysis(&request, search, is_during_search)?;
        //If the search didn't have any root or root neural net output, it must have been interrupted and we must be quitting imminently
        if !analysis_written {
          //If the reason we stopped was because we noticed a terminate, then we will write out a dummy response even if we didn't have
          //enough info to generate a real one, to fulfill a promise in the API docs that we always write something.
          if request.status.get() /*acquire*/ == AnalyzeRequest::STATUS_TERMINATED {
            ctx.report_no_analysis(&request);
          }
          //Otherwise, this case is only possible if we're just shutting down
          else {
            log::info!("Note: Search quitting due to no visits - this is normal and possible when shutting down but a bug under any other situation.");
          }
        }
      }
    }

    //Free up bot resources in case it's a while before we do more search
    bot.clear_search();

    //This request is no longer open
    {
      ctx.open_requests.borrow_mut().remove(&request.internal_id);
    }
    // delete request: the last Rc is dropped here (or by a callback the bot no longer holds).
  }
  Ok(())
}

/// upstream: Logger::logThreadUncaught (logger.cpp), after its `catch(const exception& e)`
/// PORT NOTE: logs, sleeps 5 s (the other loops keep running), then, where upstream's rethrow terminates
/// the process, flags the process as terminated (file header) and stops every loop.
async fn log_thread_uncaught_failed(ctx: &AnalysisContext, name: &str, e: &KataError) {
  log::info!("{}", "ERROR: ".to_string() + name + " loop thread failed: " + e.what());
  crate::port::executor::sleep(5.0).await;
  ctx.loop_failed.set(true);
  ctx.loop_failed_notify.notify_all();
  ctx.to_write_queue.set_read_only();
  ctx.to_analyze_queue.set_read_only();
  for i in 0..ctx.bots.len() {
    ctx.bots[i].stop_without_wait();
  }
  for i in 0..ctx.bots.len() {
    ctx.bots[i].set_killed();
  }
}

/// upstream: the `analysisLoopProtected` lambda of MainCmds::analysis (analysis.cpp), i.e. Logger::logThreadUncaught("analysis loop", ...)
/// PORT NOTE: on error see log_thread_uncaught_failed; the error is returned.
async fn analysis_loop_protected(ctx: &Rc<AnalysisContext>, mut bot: AsyncBot, thread_idx: i32) -> Result<(), KataError> {
  let r: Result<(), KataError> = analysis_loop(ctx, &mut bot, thread_idx).await;
  if let Err(e) = &r {
    if !ctx.loop_failed.get() {
      log_thread_uncaught_failed(ctx, "analysis loop", e).await;
    }
  }
  // delete bot (upstream: after all threads joined): dropped here.
  r
}

/// upstream: `getline(cin,line)` in the requestLoop lambda of MainCmds::analysis (analysis.cpp)
/// PORT NOTE: None at end of input, or once the process has terminated (see log_thread_uncaught_failed).
/// A line that is not immediately available marks the request loop as waiting (scheduling, file header).
async fn next_input_line(ctx: &AnalysisContext, input: &mut (dyn Stream<Item = Vec<u8>> + Unpin)) -> Option<Vec<u8>> {
  if ctx.loop_failed.get() {
    return None;
  }
  let mut next = input.next();
  if let Poll::Ready(line) = futures::future::poll_fn(|cx| Poll::Ready(Pin::new(&mut next).poll(cx))).await {
    return line;
  }
  ctx.request_loop_waiting.set(true);
  ctx.request_loop_idle_notify.notify_all();
  let failed = ctx.loop_failed_notify.wait_until(|| ctx.loop_failed.get());
  let line: Option<Vec<u8>> = match futures::future::select(next, failed).await {
    futures::future::Either::Left((line, _)) => line,
    futures::future::Either::Right(((), _)) => None,
  };
  if line.is_some() {
    ctx.request_loop_waiting.set(false);
  }
  line
}

/// upstream: Global::trim(line) (global.cpp) on the bytes of a std::string
/// PORT NOTE: Global::trim takes a &str; getline's line is bytes (they may be invalid UTF-8).
fn trim_bytes(s: &[u8]) -> &[u8] {
  let d: &[u8] = Global::TRIM_DEFAULT_DELIM.as_bytes();
  let p2 = match s.iter().rposition(|c| !d.contains(c)) {
    None => return &[],
    Some(p) => p,
  };
  let p1 = s.iter().position(|c| !d.contains(c)).unwrap_or(0);
  &s[p1..p2 + 1]
}

/// upstream: the `requestLoop` lambda of MainCmds::analysis (analysis.cpp)
/// PORT NOTE: async (getline -> the input stream); the loop body is handle_request_line. Throws -> Result.
/// Ends (Ok) once the process has terminated because of an analysis loop (that loop returns the error).
async fn request_loop(ctx: &Rc<AnalysisContext>, input: &mut (dyn Stream<Item = Vec<u8>> + Unpin)) -> Result<(), KataError> {
  while let Some(line) = next_input_line(ctx, input).await {
    let line: &[u8] = trim_bytes(&line);
    if line.len() == 0 {
      continue;
    }

    if ctx.log_all_requests {
      // PORT NOTE: the log crate takes text; invalid UTF-8 bytes are logged as U+FFFD.
      log::info!("{}", "Request: ".to_string() + &String::from_utf8_lossy(line));
    }

    let num_requests_before: i64 = ctx.num_requests_so_far.get();
    handle_request_line(ctx, line)?;
    // PORT NOTE: scheduling (file header): an idle analysis loop pops what this line queued before the
    // next line is handled.
    if ctx.num_requests_so_far.get() != num_requests_before {
      crate::port::executor::yield_now().await;
    }
  }
  Ok(())
}

/// nlohmann `json::find(key) != end()` on an object (and `end()` for any non-object).
fn json_contains(v: &Value, key: &str) -> bool {
  match v {
    Value::Object(m) => m.contains_key(key),
    _ => false,
  }
}

/// upstream: the `parseInteger` lambda of the requestLoop (analysis.cpp)
/// PORT NOTE: captures (rbase.id, reportErrorForId) passed explicitly (aliasing: `buf` is often a field of rbase).
fn parse_integer(ctx: &AnalysisContext, rbase_id: &str, dict: &Value, field: &str, buf: &mut i64, min: i64, max: i64, error_message: &str) -> bool {
  let r: Result<bool, KataError> = (|| {
    if !nlohmann::is_number_integer(&dict[field]) {
      ctx.report_error_for_id(rbase_id, field, error_message);
      return Ok(false);
    }
    let x: i64 = nlohmann::get_i64(&dict[field])?;
    if x < min || x > max {
      ctx.report_error_for_id(rbase_id, field, error_message);
      return Ok(false);
    }
    *buf = x;
    Ok(true)
  })();
  match r {
    Ok(b) => b,
    Err(KataError::NlohmannJson(_)) => {
      ctx.report_error_for_id(rbase_id, field, error_message);
      false
    }
    Err(_) => unreachable!("only nlohmann exceptions can be raised here"),
  }
}

/// upstream: the `parseDouble` lambda of the requestLoop (analysis.cpp)
/// PORT NOTE: captures passed explicitly (see parse_integer).
fn parse_double(ctx: &AnalysisContext, rbase_id: &str, dict: &Value, field: &str, buf: &mut f64, min: f64, max: f64, error_message: &str) -> bool {
  let r: Result<bool, KataError> = (|| {
    if !dict[field].is_number() {
      ctx.report_error_for_id(rbase_id, field, error_message);
      return Ok(false);
    }
    let x: f64 = nlohmann::get_f64(&dict[field])?;
    if !x.is_finite() || x < min || x > max {
      ctx.report_error_for_id(rbase_id, field, error_message);
      return Ok(false);
    }
    *buf = x;
    Ok(true)
  })();
  match r {
    Ok(b) => b,
    Err(KataError::NlohmannJson(_)) => {
      ctx.report_error_for_id(rbase_id, field, error_message);
      false
    }
    Err(_) => unreachable!("only nlohmann exceptions can be raised here"),
  }
}

/// upstream: the `parseBoolean` lambda of the requestLoop (analysis.cpp)
/// PORT NOTE: captures passed explicitly (see parse_integer).
fn parse_boolean(ctx: &AnalysisContext, rbase_id: &str, dict: &Value, field: &str, buf: &mut bool, error_message: &str) -> bool {
  let r: Result<bool, KataError> = (|| {
    if !dict[field].is_boolean() {
      ctx.report_error_for_id(rbase_id, field, error_message);
      return Ok(false);
    }
    *buf = nlohmann::get_bool(&dict[field])?;
    Ok(true)
  })();
  match r {
    Ok(b) => b,
    Err(KataError::NlohmannJson(_)) => {
      ctx.report_error_for_id(rbase_id, field, error_message);
      false
    }
    Err(_) => unreachable!("only nlohmann exceptions can be raised here"),
  }
}

/// upstream: the `parsePlayer` lambda of the requestLoop (analysis.cpp)
/// PORT NOTE: captures passed explicitly (see parse_integer).
fn parse_player(ctx: &AnalysisContext, rbase_id: &str, dict: &Value, field: &str, buf: &mut Player) -> bool {
  *buf = C_EMPTY;
  match nlohmann::get_string(&dict[field]) {
    Ok(s) => {
      PlayerIO::try_parse_player(&s, buf);
    }
    Err(_) => {}
  }
  if *buf != P_BLACK && *buf != P_WHITE {
    ctx.report_error_for_id(rbase_id, field, "Must be \"b\" or \"w\"");
    return false;
  }
  true
}

/// upstream: the `parseBoardLocs` lambda of the requestLoop (analysis.cpp)
/// PORT NOTE: captures (boardXSize, boardYSize by value; rbase, reportErrorForId) passed explicitly.
fn parse_board_locs(
  ctx: &AnalysisContext,
  rbase_id: &str,
  board_x_size: i32,
  board_y_size: i32,
  dict: &Value,
  field: &str,
  buf: &mut Vec<Loc>,
  allow_pass: bool,
) -> bool {
  buf.clear();
  if !dict[field].is_array() {
    ctx.report_error_for_id(rbase_id, field, "Must be an array of GTP board vertices");
    return false;
  }
  for elt in dict[field].as_array().expect("checked is_array") {
    let s: String = match nlohmann::get_string(elt) {
      Ok(s) => s,
      Err(_) => {
        ctx.report_error_for_id(rbase_id, field, "Must be an array of GTP board vertices");
        return false;
      }
    };

    let mut loc: Loc = 0;
    if !Location::try_of_string_x_size_y_size(&s, board_x_size, board_y_size, &mut loc)
      || (!allow_pass && loc == Board::PASS_LOC)
      || (loc == Board::NULL_LOC)
    {
      ctx.report_error_for_id(rbase_id, field, &("Could not parse board location: ".to_string() + &s));
      return false;
    }
    buf.push(loc);
  }
  true
}

/// upstream: the `parseBoardMoves` lambda of the requestLoop (analysis.cpp)
/// PORT NOTE: captures passed explicitly (see parse_board_locs).
fn parse_board_moves(
  ctx: &AnalysisContext,
  rbase_id: &str,
  board_x_size: i32,
  board_y_size: i32,
  dict: &Value,
  field: &str,
  buf: &mut Vec<Move>,
  allow_pass: bool,
) -> bool {
  buf.clear();
  if !dict[field].is_array() {
    ctx.report_error_for_id(rbase_id, field, "Must be an array of pairs of the form: [\"b\" or \"w\", GTP board vertex]");
    return false;
  }
  for elt in dict[field].as_array().expect("checked is_array") {
    if !elt.is_array() || elt.as_array().expect("checked is_array").len() != 2 {
      ctx.report_error_for_id(rbase_id, field, "Must be an array of pairs of the form: [\"b\" or \"w\", GTP board vertex]");
      return false;
    }

    let s0: String;
    let s1: String;
    match (|| -> Result<(String, String), KataError> { Ok((nlohmann::get_string(&elt[0])?, nlohmann::get_string(&elt[1])?)) })() {
      Ok((a, b)) => {
        s0 = a;
        s1 = b;
      }
      Err(_) => {
        ctx.report_error_for_id(rbase_id, field, "Must be an array of pairs of the form: [\"b\" or \"w\", GTP board vertex]");
        return false;
      }
    }

    let mut pla: Player = C_EMPTY;
    if !PlayerIO::try_parse_player(&s0, &mut pla) {
      ctx.report_error_for_id(rbase_id, field, &("Could not parse player: ".to_string() + &s0));
      return false;
    }

    let mut loc: Loc = 0;
    if !Location::try_of_string_x_size_y_size(&s1, board_x_size, board_y_size, &mut loc)
      || (!allow_pass && loc == Board::PASS_LOC)
      || (loc == Board::NULL_LOC)
    {
      ctx.report_error_for_id(rbase_id, field, &("Could not parse board location: ".to_string() + &s1));
      return false;
    }
    buf.push(Move::new(loc, pla));
  }
  true
}

/// upstream: the body of the requestLoop lambda's `while(getline(cin,line))` (analysis.cpp), after the
/// trim, the empty-line check and the "Request: " log line.
/// PORT NOTE: a fn of its own (`continue` -> `return Ok(())`); the request loop's reused `json input`
/// is a local. Throws (uncaught upstream) -> Result.
/// `line` is the trimmed input line's bytes.
pub fn handle_request_line(ctx: &Rc<AnalysisContext>, line: &[u8]) -> Result<(), KataError> {
  let mut input: Value = match nlohmann::parse_bytes(line) {
    Ok(v) => v,
    Err(e) => {
      // `e.what() + string(...)`: what() is a C string (it ends at a NUL); `line` is all its bytes.
      let what: &[u8] = e.what();
      let what: &[u8] = &what[..what.iter().position(|&c| c == 0).unwrap_or(what.len())];
      let mut s: Vec<u8> = what.to_vec();
      s.extend_from_slice(b" - could not parse input line as json request: ");
      s.extend_from_slice(line);
      ctx.report_error_bytes(&s)?;
      return Ok(());
    }
  };

  if !input.is_object() {
    ctx.report_error(&("Request line was valid json but was not an object, ignoring: ".to_string() + &nlohmann::dump(&input)));
    return Ok(());
  }

  if !json_contains(&input, "id") || !input["id"].is_string() {
    ctx.report_error("Request must have a string \"id\" field");
    return Ok(());
  }

  let mut rbase: AnalyzeRequest = AnalyzeRequest::new();
  rbase.id = nlohmann::get_string(&input["id"])?;

  //Special actions
  if json_contains(&input, "action") && input["action"].is_string() {
    let action: String = nlohmann::get_string(&input["action"])?;
    if action == "query_version" {
      input["version"] = Value::from(KATAGO_VERSION);
      input["git_hash"] = Value::from(GIT_REVISION);
      ctx.push_to_write(nlohmann::dump(&input));
    } else if action == "query_models" {
      input["models"] = Value::Array(Vec::new());
      {
        let nn_eval: &NNEvaluator = &ctx.nn_eval;
        let mut model_info: Value = Value::Null;
        model_info["name"] = Value::from(nn_eval.get_model_name());
        model_info["internalName"] = Value::from(nn_eval.get_internal_model_name());
        model_info["maxBatchSize"] = Value::from(nn_eval.get_max_batch_size());
        model_info["usesHumanSLProfile"] = Value::from(nn_eval.requires_sgf_metadata());
        model_info["version"] = Value::from(nn_eval.get_model_version());
        model_info["usingFP16"] = Value::from(nn_eval.get_using_fp16_mode().to_string());
        input["models"].as_array_mut().expect("just set to an array").push(model_info);
      }
      if let Some(human_eval) = &ctx.human_eval {
        let mut model_info: Value = Value::Null;
        model_info["name"] = Value::from(human_eval.get_model_name());
        model_info["internalName"] = Value::from(human_eval.get_internal_model_name());
        model_info["maxBatchSize"] = Value::from(human_eval.get_max_batch_size());
        model_info["usesHumanSLProfile"] = Value::from(human_eval.requires_sgf_metadata());
        model_info["version"] = Value::from(human_eval.get_model_version());
        model_info["usingFP16"] = Value::from(human_eval.get_using_fp16_mode().to_string());
        input["models"].as_array_mut().expect("just set to an array").push(model_info);
      }
      ctx.push_to_write(nlohmann::dump(&input));
    } else if action == "clear_cache" {
      //This should be thread-safe.
      ctx.nn_eval.clear_cache();
      if let Some(human_eval) = &ctx.human_eval {
        human_eval.clear_cache();
      }
      if let Some(eval_cache) = &ctx.eval_cache {
        eval_cache.clear();
      }
      ctx.push_to_write(nlohmann::dump(&input));
    } else if action == "terminate" {
      let mut terminate_id_found: bool = false;
      let mut terminate_id: String = String::new();
      if json_contains(&input, "terminateId") && input["terminateId"].is_string() {
        terminate_id = nlohmann::get_string(&input["terminateId"])?;
        terminate_id_found = true;
      }
      if !terminate_id_found {
        ctx.report_error_for_id(&rbase.id, "terminateId", "Requests for a terminate action must have a string \"terminateId\" field");
        return Ok(());
      }

      let mut has_turn_numbers: bool = false;
      let mut turn_numbers: Vec<i32> = Vec::new();
      if json_contains(&input, "turnNumbers") {
        match nlohmann::get_vector_i32(&input["turnNumbers"]) {
          Ok(v) => {
            turn_numbers = v;
            has_turn_numbers = true;
          }
          Err(KataError::NlohmannJson(_)) => {
            ctx.report_error_for_id(&rbase.id, "turnNumbers", "If provided, must be an array of integers indicating turns to terminate");
            return Ok(());
          }
          Err(e) => return Err(e),
        }
      }

      {
        // PORT NOTE: lock removed.
        let turn_numbers_set: BTreeSet<i32> = turn_numbers.iter().copied().collect();
        let open_requests = ctx.open_requests.borrow();
        for (_, request) in open_requests.iter() {
          if request.id == terminate_id && (!has_turn_numbers || turn_numbers_set.contains(&request.turn_number)) {
            ctx.terminate_request(request);
          }
        }
      }
      ctx.push_to_write(nlohmann::dump(&input));
    } else if action == "terminate_all" {
      let mut has_turn_numbers: bool = false;
      let mut turn_numbers: Vec<i32> = Vec::new();
      if json_contains(&input, "turnNumbers") {
        match nlohmann::get_vector_i32(&input["turnNumbers"]) {
          Ok(v) => {
            turn_numbers = v;
            has_turn_numbers = true;
          }
          Err(KataError::NlohmannJson(_)) => {
            ctx.report_error_for_id(&rbase.id, "turnNumbers", "If provided, must be an array of integers indicating turns to terminate");
            return Ok(());
          }
          Err(e) => return Err(e),
        }
      }

      {
        // PORT NOTE: lock removed.
        let turn_numbers_set: BTreeSet<i32> = turn_numbers.iter().copied().collect();
        let open_requests = ctx.open_requests.borrow();
        for (_, request) in open_requests.iter() {
          if !has_turn_numbers || turn_numbers_set.contains(&request.turn_number) {
            ctx.terminate_request(request);
          }
        }
      }
      ctx.push_to_write(nlohmann::dump(&input));
    } else {
      ctx.report_error("'action' field must be 'query_version' or 'query_models' or 'clear_cache' or 'terminate' or 'terminate_all'");
    }

    return Ok(());
  }

  //Defaults
  rbase.params = ctx.default_params.clone();
  rbase.perspective = ctx.default_perspective;
  rbase.analysis_pv_len = ctx.analysis_pv_len;
  rbase.include_ownership = false;
  rbase.include_ownership_stdev = false;
  rbase.include_moves_ownership = false;
  rbase.include_moves_ownership_stdev = false;
  rbase.include_policy = false;
  rbase.include_pv_visits = false;
  rbase.include_no_result_value = false;
  rbase.report_during_search = false;
  rbase.report_during_search_every = 1e30;
  rbase.first_report_during_search_after = 1e30;
  rbase.priority = 0;
  rbase.avoid_move_until_by_loc_black.clear();
  rbase.avoid_move_until_by_loc_white.clear();

  let rbase_id: String = rbase.id.clone();

  let board_x_size: i32;
  let board_y_size: i32;
  {
    let mut x_buf: i64 = 0;
    let mut y_buf: i64 = 0;
    // PORT NOTE: function-local `static const string` -> a local (PORTING.md §8).
    let board_size_error: String = "Must provide an integer from 2 to ".to_string() + &Global::int_to_string(Board::MAX_LEN);
    if !json_contains(&input, "boardXSize") {
      ctx.report_error_for_id(&rbase_id, "boardXSize", &board_size_error);
      return Ok(());
    }
    if !json_contains(&input, "boardYSize") {
      ctx.report_error_for_id(&rbase_id, "boardYSize", &board_size_error);
      return Ok(());
    }
    if !parse_integer(ctx, &rbase_id, &input, "boardXSize", &mut x_buf, 2, Board::MAX_LEN as i64, &board_size_error) {
      return Ok(());
    }
    if !parse_integer(ctx, &rbase_id, &input, "boardYSize", &mut y_buf, 2, Board::MAX_LEN as i64, &board_size_error) {
      return Ok(());
    }
    // PORT NOTE: in range because parse_integer checked 2..MAX_LEN.
    board_x_size = x_buf as i32;
    board_y_size = y_buf as i32;
  }

  let mut placements: Vec<Move> = Vec::new();
  if json_contains(&input, "initialStones") {
    if !parse_board_moves(ctx, &rbase_id, board_x_size, board_y_size, &input, "initialStones", &mut placements, false) {
      return Ok(());
    }
  }
  let mut move_history: Vec<Move> = Vec::new();
  if json_contains(&input, "moves") {
    if !parse_board_moves(ctx, &rbase_id, board_x_size, board_y_size, &input, "moves", &mut move_history, true) {
      return Ok(());
    }
  } else {
    ctx.report_error_for_id(&rbase_id, "moves", "Must specify an array of [player,location] pairs");
    return Ok(());
  }
  let mut initial_player: Player = C_EMPTY;
  if json_contains(&input, "initialPlayer") {
    let suc: bool = parse_player(ctx, &rbase_id, &input, "initialPlayer", &mut initial_player);
    if !suc {
      return Ok(());
    }
  }

  let mut should_analyze: Vec<bool> = vec![false; move_history.len() + 1];
  if json_contains(&input, "analyzeTurns") {
    let analyze_turns: Vec<i32> = match nlohmann::get_vector_i32(&input["analyzeTurns"]) {
      Ok(v) => v,
      Err(KataError::NlohmannJson(_)) => {
        ctx.report_error_for_id(&rbase_id, "analyzeTurns", "Must specify an array of integers indicating turns to analyze");
        return Ok(());
      }
      Err(e) => return Err(e),
    };

    let mut failed: bool = false;
    for i in 0..analyze_turns.len() {
      let turn_number: i32 = analyze_turns[i];
      if turn_number < 0 || (turn_number as usize) >= should_analyze.len() {
        ctx.report_error_for_id(&rbase_id, "analyzeTurns", &("Invalid turn number: ".to_string() + &Global::int_to_string(turn_number)));
        failed = true;
        break;
      }
      should_analyze[turn_number as usize] = true;
    }
    if failed {
      return Ok(());
    }
  } else {
    let n: usize = should_analyze.len();
    should_analyze[n - 1] = true;
  }

  let mut priorities: BTreeMap<i32, i64> = BTreeMap::new();
  if json_contains(&input, "priorities") {
    let priorities_vec: Vec<i64> = match nlohmann::get_vector_i64(&input["priorities"]) {
      Ok(v) => v,
      Err(KataError::NlohmannJson(_)) => {
        ctx.report_error_for_id(&rbase_id, "priorities", "Must specify an array of integers indicating priorities");
        return Ok(());
      }
      Err(e) => return Err(e),
    };
    if !json_contains(&input, "analyzeTurns") {
      ctx.report_error_for_id(&rbase_id, "priorities", "Can only specify when also specifying analyzeTurns");
      return Ok(());
    }
    let analyze_turns: Vec<i32> = nlohmann::get_vector_i32(&input["analyzeTurns"])?;
    if priorities_vec.len() != analyze_turns.len() {
      ctx.report_error_for_id(&rbase_id, "priorities", "Must be of matching length to analyzeTurns");
      return Ok(());
    }

    let mut failed: bool = false;
    for i in 0..priorities_vec.len() {
      let priority: i64 = priorities_vec[i];
      if priority < -0x3FFFffffFFFFffffi64 || priority > 0x3FFFffffFFFFffffi64 {
        ctx.report_error_for_id(&rbase_id, "priorities", &("Invalid priority: ".to_string() + &Global::int64_to_string(priority)));
        failed = true;
        break;
      }
      priorities.insert(analyze_turns[i], priority);
    }
    if failed {
      priorities.clear();
      return Ok(());
    }
  }

  let mut rules: Rules = Rules::new();
  if json_contains(&input, "rules") {
    if input["rules"].is_string() {
      let s: String = nlohmann::get_string(&input["rules"])?;
      if !Rules::try_parse_rules(&s, &mut rules)? {
        ctx.report_error_for_id(&rbase_id, "rules", &("Could not parse rules: ".to_string() + &s));
        return Ok(());
      }
    } else if input["rules"].is_object() {
      let s: String = nlohmann::dump(&input["rules"]);
      if !Rules::try_parse_rules(&s, &mut rules)? {
        ctx.report_error_for_id(&rbase_id, "rules", &("Could not parse rules: ".to_string() + &s));
        return Ok(());
      }
    } else {
      ctx.report_error_for_id(
        &rbase_id,
        "rules",
        "Must specify rules string, such as \"chinese\" or \"tromp-taylor\", or a JSON object with detailed rules parameters.",
      );
      return Ok(());
    }
  } else {
    ctx.report_error_for_id(
      &rbase_id,
      "rules",
      "Must specify rules string, such as \"chinese\" or \"tromp-taylor\", or a JSON object with detailed rules parameters.",
    );
    return Ok(());
  }

  if json_contains(&input, "komi") {
    let mut komi: f64 = 0.0;
    // static_assert(Rules::MIN_USER_KOMI == -150.0f, "");
    // static_assert(Rules::MAX_USER_KOMI == 150.0f, "");
    const _: () = assert!(Rules::MIN_USER_KOMI == -150.0f32 && Rules::MAX_USER_KOMI == 150.0f32);
    let msg: &str = "Must be a integer or half-integer from -150.0 to 150.0";
    let suc: bool = parse_double(ctx, &rbase_id, &input, "komi", &mut komi, Rules::MIN_USER_KOMI as f64, Rules::MAX_USER_KOMI as f64, msg);
    if !suc {
      return Ok(());
    }
    rules.komi = komi as f32;
    if !Rules::komi_is_int_or_half_int(rules.komi) {
      ctx.report_error_for_id(&rbase_id, "komi", msg);
      return Ok(());
    }
  }

  if json_contains(&input, "whiteHandicapBonus") {
    if !input["whiteHandicapBonus"].is_string() {
      ctx.report_error_for_id(&rbase_id, "whiteHandicapBonus", "Must be a string");
      return Ok(());
    }
    let s: String = nlohmann::get_string(&input["whiteHandicapBonus"])?;
    match Rules::parse_white_handicap_bonus_rule(&s) {
      Ok(white_handicap_bonus_rule) => {
        rules.white_handicap_bonus_rule = white_handicap_bonus_rule;
      }
      Err(err) if err.is_string_error() => {
        ctx.report_error_for_id(&rbase_id, "whiteHandicapBonus", err.what());
        return Ok(());
      }
      Err(err) => return Err(err),
    }
  }

  if json_contains(&input, "overrideSettings") {
    let settings: Value = input["overrideSettings"].clone();
    if !settings.is_object() {
      ctx.report_error_for_id(&rbase_id, "overrideSettings", "Must be an object");
      return Ok(());
    }
    let mut override_settings: BTreeMap<String, String> = BTreeMap::new();
    for (key, value) in settings.as_object().expect("checked is_object") {
      override_settings.insert(
        key.clone(),
        if value.is_string() { nlohmann::get_string(value)? } else { nlohmann::dump(value) }, // always convert to string
      );
    }

    // Reload settings to allow overrides
    if !override_settings.is_empty() {
      let r: Result<(), KataError> = (|| {
        let mut local_cfg: ConfigParser = ConfigParser::new_copy(&ctx.cfg)?;
        //Ignore any unused keys in the ORIGINAL config
        local_cfg.mark_all_keys_used_with_prefix("");
        local_cfg.override_keys_newkvs(&override_settings);
        load_params(&ctx.human_model_file, &mut local_cfg, &mut rbase.params, &mut rbase.perspective, ctx.default_perspective)?;
        SearchParams::fail_if_params_differ_on_unchangeable_parameter(&ctx.default_params, &rbase.params)?;
        //Soft failure on unused override keys newly present in the config
        let unused_keys: Vec<String> = local_cfg.unused_keys();
        if unused_keys.len() > 0 {
          ctx.report_warning_for_id(&rbase_id, "overrideSettings", &("Unknown config params: ".to_string() + &Global::concat(&unused_keys, ",")));
        }
        let mut out: String = String::new();
        if Setup::maybe_warn_human_sl_params(&rbase.params, Some(&ctx.nn_eval), ctx.human_eval.as_deref(), &mut out, false) {
          return Err(KataError::StringError(out));
        }
        Ok(())
      })();
      match r {
        Ok(()) => {}
        Err(exception) if exception.is_string_error() => {
          ctx.report_error_for_id(&rbase_id, "overrideSettings", &("Could not set settings: ".to_string() + exception.what()));
          return Ok(());
        }
        Err(exception) => return Err(exception),
      }
    }
  }

  if json_contains(&input, "maxVisits") {
    let suc: bool = parse_integer(ctx, &rbase_id, &input, "maxVisits", &mut rbase.params.max_visits, 1, 1i64 << 50, "Must be an integer from 1 to 2^50");
    if !suc {
      return Ok(());
    }
  }

  if json_contains(&input, "analysisPVLen") {
    let mut buf: i64 = 0;
    let suc: bool = parse_integer(ctx, &rbase_id, &input, "analysisPVLen", &mut buf, 1, 1000, "Must be an integer from 1 to 1000");
    if !suc {
      return Ok(());
    }
    // PORT NOTE: in range because parse_integer checked 1..1000.
    rbase.analysis_pv_len = buf as i32;
  }

  if json_contains(&input, "rootFpuReductionMax") {
    let suc: bool = parse_double(ctx, &rbase_id, &input, "rootFpuReductionMax", &mut rbase.params.root_fpu_reduction_max, 0.0, 2.0, "Must be a number from 0.0 to 2.0");
    if !suc {
      return Ok(());
    }
  }
  if json_contains(&input, "rootPolicyTemperature") {
    let suc: bool =
      parse_double(ctx, &rbase_id, &input, "rootPolicyTemperature", &mut rbase.params.root_policy_temperature, 0.01, 100.0, "Must be a number from 0.01 to 100.0");
    if !suc {
      return Ok(());
    }
    rbase.params.root_policy_temperature_early = rbase.params.root_policy_temperature;
  }
  if json_contains(&input, "includeMovesOwnership") {
    let suc: bool = parse_boolean(ctx, &rbase_id, &input, "includeMovesOwnership", &mut rbase.include_moves_ownership, "Must be a boolean");
    if !suc {
      return Ok(());
    }
  }
  if json_contains(&input, "includeMovesOwnershipStdev") {
    let suc: bool = parse_boolean(ctx, &rbase_id, &input, "includeMovesOwnershipStdev", &mut rbase.include_moves_ownership_stdev, "Must be a boolean");
    if !suc {
      return Ok(());
    }
  }
  if json_contains(&input, "includeOwnership") {
    let suc: bool = parse_boolean(ctx, &rbase_id, &input, "includeOwnership", &mut rbase.include_ownership, "Must be a boolean");
    if !suc {
      return Ok(());
    }
  }
  if json_contains(&input, "includeOwnershipStdev") {
    let suc: bool = parse_boolean(ctx, &rbase_id, &input, "includeOwnershipStdev", &mut rbase.include_ownership_stdev, "Must be a boolean");
    if !suc {
      return Ok(());
    }
  }
  if json_contains(&input, "includePolicy") {
    let suc: bool = parse_boolean(ctx, &rbase_id, &input, "includePolicy", &mut rbase.include_policy, "Must be a boolean");
    if !suc {
      return Ok(());
    }
  }
  if json_contains(&input, "includePVVisits") {
    let suc: bool = parse_boolean(ctx, &rbase_id, &input, "includePVVisits", &mut rbase.include_pv_visits, "Must be a boolean");
    if !suc {
      return Ok(());
    }
  }
  if json_contains(&input, "includeNoResultValue") {
    let suc: bool = parse_boolean(ctx, &rbase_id, &input, "includeNoResultValue", &mut rbase.include_no_result_value, "Must be a boolean");
    if !suc {
      return Ok(());
    }
  }
  if json_contains(&input, "reportDuringSearchEvery") {
    let suc: bool = parse_double(
      ctx,
      &rbase_id,
      &input,
      "reportDuringSearchEvery",
      &mut rbase.report_during_search_every,
      0.001,
      1000000.0,
      "Must be number of seconds from 0.001 to 1000000.0",
    );
    if !suc {
      return Ok(());
    }
    rbase.report_during_search = true;
    rbase.first_report_during_search_after = rbase.report_during_search_every;
  }
  if json_contains(&input, "firstReportDuringSearchAfter") {
    let suc: bool = parse_double(
      ctx,
      &rbase_id,
      &input,
      "firstReportDuringSearchAfter",
      &mut rbase.first_report_during_search_after,
      0.001,
      1000000.0,
      "Must be number of seconds from 0.001 to 1000000.0",
    );
    if !suc {
      return Ok(());
    }
    rbase.report_during_search = true;
  }
  if json_contains(&input, "priority") {
    if json_contains(&input, "priorities") {
      ctx.report_error_for_id(&rbase_id, "priority", "Cannot specify both priority and priorities");
      return Ok(());
    }
    let mut buf: i64 = 0;
    let suc: bool = parse_integer(ctx, &rbase_id, &input, "priority", &mut buf, -0x3FFFffffFFFFffffi64, 0x3FFFffffFFFFffffi64, "Must be a number between -2^62 and 2^62");
    if !suc {
      return Ok(());
    }
    rbase.priority = buf;
  }

  let has_allow_moves: bool = json_contains(&input, "allowMoves");
  let has_avoid_moves: bool = json_contains(&input, "avoidMoves");
  if has_allow_moves || has_avoid_moves {
    if has_allow_moves && has_avoid_moves {
      ctx.report_error_for_id(&rbase_id, "allowMoves", "Cannot specify both allowMoves and avoidMoves");
      return Ok(());
    }
    let field: &str = if has_allow_moves { "allowMoves" } else { "avoidMoves" };
    let avoid_params_list: &Value = &input[field];
    if !avoid_params_list.is_array() {
      ctx.report_error_for_id(&rbase_id, field, "Must be a list of dicts with subfields 'player', 'moves', 'untilDepth'");
      return Ok(());
    }
    let avoid_params_list: &Vec<Value> = avoid_params_list.as_array().expect("checked is_array");
    if has_allow_moves && avoid_params_list.len() > 1 {
      ctx.report_error_for_id(&rbase_id, field, "Currently allowMoves only allows one entry");
      return Ok(());
    }

    let mut failed: bool = false;
    for i in 0..avoid_params_list.len() {
      let avoid_params: &Value = &avoid_params_list[i];
      if !json_contains(avoid_params, "moves") || !json_contains(avoid_params, "untilDepth") || !json_contains(avoid_params, "player") {
        ctx.report_error_for_id(&rbase_id, field, "Must be a list of dicts with subfields 'player', 'moves', 'untilDepth'");
        failed = true;
        break;
      }

      let mut avoid_pla: Player = C_EMPTY;
      let mut parsed_locs: Vec<Loc> = Vec::new();
      let mut until_depth: i64 = 0;
      let mut suc: bool;
      suc = parse_player(ctx, &rbase_id, avoid_params, "player", &mut avoid_pla);
      if !suc {
        failed = true;
        break;
      }
      suc = parse_board_locs(ctx, &rbase_id, board_x_size, board_y_size, avoid_params, "moves", &mut parsed_locs, true);
      if !suc {
        failed = true;
        break;
      }
      suc = parse_integer(ctx, &rbase_id, avoid_params, "untilDepth", &mut until_depth, 1, 1000000000, "Must be a positive integer");
      if !suc {
        failed = true;
        break;
      }

      let avoid_move_until_by_loc: &mut Vec<i32> =
        if avoid_pla == P_BLACK { &mut rbase.avoid_move_until_by_loc_black } else { &mut rbase.avoid_move_until_by_loc_white };
      avoid_move_until_by_loc.resize(Board::MAX_ARR_SIZE as usize, 0);
      if has_allow_moves {
        // PORT NOTE: in range because parse_integer checked 1..1e9.
        avoid_move_until_by_loc.fill(until_depth as i32);
        for loc in &parsed_locs {
          avoid_move_until_by_loc[*loc as usize] = 0;
        }
      } else {
        for loc in &parsed_locs {
          avoid_move_until_by_loc[*loc as usize] = until_depth as i32;
        }
      }
    }
    if failed {
      return Ok(());
    }
  }

  let mut board: Board = Board::new_x_y(board_x_size, board_y_size).expect("upstream: cannot throw, sizes were checked to be 2..MAX_LEN");
  for i in 0..placements.len() {
    board.set_stone(placements[i].loc, placements[i].pla);
  }

  if initial_player == C_EMPTY {
    if move_history.len() > 0 {
      initial_player = move_history[0].pla;
    } else {
      initial_player = if BoardHistory::num_handicap_stones_on_board(&board) > 0 { P_WHITE } else { P_BLACK };
    }
  }

  let mut rules_were_supported: bool = false;
  let supported_rules: Rules = ctx.nn_eval.get_supported_rules(&rules, &mut rules_were_supported);
  if !rules_were_supported {
    let out: String = format!("Rules {} not supported by neural net, using {} instead", rules, supported_rules);
    ctx.report_warning_for_id(&rbase_id, "rules", &out);
    rules = supported_rules;
  }

  let mut next_pla: Player = initial_player;
  let mut hist: BoardHistory = BoardHistory::new_full(&board, next_pla, &rules, 0);
  hist.set_assume_multiple_starting_black_moves_are_handicap(ctx.assume_multiple_starting_black_moves_are_handicap);

  if ctx.warn_unused_fields {
    for (key, _) in input.as_object().expect("checked is_object") {
      if !ctx.expected_keys.contains(key) {
        ctx.report_warning_for_id(
          &rbase_id,
          key,
          "Unexpected or unused field, do you have a typo? (set warnUnusedFields=false in the config to disable this warning)",
        );
      }
    }
  }

  //Build and enqueue requests
  let mut new_requests: Vec<AnalyzeRequest> = Vec::new();
  let mut found_illegal_move: bool = false;
  let mut turn_number: i32 = 0;
  while (turn_number as usize) <= move_history.len() {
    if should_analyze[turn_number as usize] {
      let mut priority: i64 = rbase.priority;
      if priorities.len() > 0 {
        testAssert!(priorities.len() > new_requests.len());
        testAssert!(priorities.contains_key(&turn_number));
        priority = *priorities.entry(turn_number).or_default();
      }

      let mut new_request: AnalyzeRequest = AnalyzeRequest::new();
      new_request.internal_id = ctx.internal_id_counter.get();
      ctx.internal_id_counter.set(ctx.internal_id_counter.get() + 1);
      new_request.id = rbase.id.clone();
      new_request.turn_number = turn_number;
      new_request.board = board.clone();
      new_request.hist = hist.clone();
      new_request.next_pla = next_pla;
      new_request.params = rbase.params.clone();
      new_request.perspective = rbase.perspective;
      new_request.analysis_pv_len = rbase.analysis_pv_len;
      new_request.include_ownership = rbase.include_ownership;
      new_request.include_ownership_stdev = rbase.include_ownership_stdev;
      new_request.include_moves_ownership = rbase.include_moves_ownership;
      new_request.include_moves_ownership_stdev = rbase.include_moves_ownership_stdev;
      new_request.include_policy = rbase.include_policy;
      new_request.include_pv_visits = rbase.include_pv_visits;
      new_request.include_no_result_value = rbase.include_no_result_value;
      new_request.report_during_search = rbase.report_during_search;
      new_request.report_during_search_every = rbase.report_during_search_every;
      new_request.first_report_during_search_after = rbase.first_report_during_search_after;
      new_request.priority = priority;
      new_request.avoid_move_until_by_loc_black = rbase.avoid_move_until_by_loc_black.clone();
      new_request.avoid_move_until_by_loc_white = rbase.avoid_move_until_by_loc_white.clone();
      new_request.status.set(AnalyzeRequest::STATUS_IN_QUEUE) /*release*/;
      new_requests.push(new_request);
    }
    if (turn_number as usize) >= move_history.len() {
      break;
    }

    let move_pla: Player = move_history[turn_number as usize].pla;
    let move_loc: Loc = move_history[turn_number as usize].loc;
    if move_pla != next_pla {
      board.clear_simple_ko_loc();
      let encore_phase: i32 = hist.encore_phase;
      hist.clear(&board, move_pla, &rules, encore_phase);
      hist.set_assume_multiple_starting_black_moves_are_handicap(ctx.assume_multiple_starting_black_moves_are_handicap);
    }

    let suc: bool = hist.make_board_move_tolerant_prevent_encore(&mut board, move_loc, move_pla, ctx.prevent_encore);
    if !suc {
      ctx.report_error_for_id(
        &rbase_id,
        "moves",
        &("Illegal move ".to_string() + &Global::int_to_string(turn_number) + ": " + &Location::to_string(move_loc, &board)),
      );
      found_illegal_move = true;
      break;
    }
    next_pla = get_opp(move_pla);
    turn_number += 1;
  }

  if found_illegal_move {
    // delete newRequests[i]: dropped
    new_requests.clear();
    return Ok(());
  }

  let new_requests: Vec<Rc<AnalyzeRequest>> = new_requests.into_iter().map(Rc::new).collect();
  //Add all requests to open requests
  {
    // PORT NOTE: lock removed.
    let mut open_requests = ctx.open_requests.borrow_mut();
    for i in 0..new_requests.len() {
      open_requests.insert(new_requests[i].internal_id, new_requests[i].clone());
    }
  }
  //Push into queue for processing
  for i in 0..new_requests.len() {
    //Compare first by user-provided priority, and next breaks ties by preferring earlier requests.
    let priority_key: (i64, i64) = (new_requests[i].priority, -ctx.num_requests_so_far.get());
    let suc: bool = ctx.to_analyze_queue.force_push(priority_key, new_requests[i].clone());
    testAssert!(suc);
    ctx.num_requests_so_far.set(ctx.num_requests_so_far.get() + 1);
  }
  Ok(())
}

/// upstream: MainCmds::analysis (analysis.cpp)
/// PORT NOTE: a library fn over injected host services (see the file header): argv parsing and
/// `cmd.getConfig(cfg)` are the host's (`cfg` arrives initialized with the -config files and the
/// -override-config values applied), so the TCLAP `catch` (exit code 1) is the host's too. Async: the
/// request loop, the analysis loops and the write loop are joined virtual threads. Returns the exit code;
/// an exception that kills the process upstream is the Err (after every loop has stopped).
pub async fn analysis(mut cfg: ConfigParser, host: AnalysisHost<'_>) -> Result<i32, KataError> {
  Board::init_hash();
  ScoreValue::init_tables();
  let mut seed_rand: Rand = Rand::new();

  let model_file: String = host.args.model_file.clone();
  let human_model_file: String = host.args.human_model_file.clone();
  let num_analysis_threads_cmdline_specified: bool = host.args.num_analysis_threads_cmdline_specified;
  let num_analysis_threads_cmdline: i32 = host.args.num_analysis_threads_cmdline;
  let quit_without_waiting: bool = host.args.quit_without_waiting;
  let write_stderr: &dyn Fn(&str) = &*host.write_stderr;

  cfg.apply_alias("numSearchThreadsPerAnalysisThread", "numSearchThreads")?;

  if cfg.contains("numAnalysisThreads") && num_analysis_threads_cmdline_specified {
    return Err(KataError::StringError(
      "When specifying numAnalysisThreads in the config (".to_string()
        + &cfg.get_file_name()
        + "), it is redundant and disallowed to also specify it via -analysis-threads",
    ));
  }

  let num_analysis_threads: i32 =
    if num_analysis_threads_cmdline_specified { num_analysis_threads_cmdline } else { cfg.get_int("numAnalysisThreads", 1, 16384)? };
  if num_analysis_threads <= 0 || num_analysis_threads > 16384 {
    return Err(KataError::StringError("Invalid value for numAnalysisThreads: ".to_string() + &Global::int_to_string(num_analysis_threads)));
  }

  let for_deterministic_testing: bool = if cfg.contains("forDeterministicTesting") { cfg.get_bool("forDeterministicTesting")? } else { false };
  if for_deterministic_testing {
    seed_rand.init_seed("forDeterministicTesting");
  }

  let log_to_stdout_default: bool = false;
  let log_to_stderr_default: bool = true;
  // PORT NOTE: `Logger logger(&cfg, logToStdoutDefault, logToStderrDefault);` (file header).
  let log_to_stderr: bool = logger_config_reads(&mut cfg, log_to_stdout_default, log_to_stderr_default, &*host.init_logger)?;

  log::info!("Analysis Engine starting...");
  log::info!("{}", KATAGO_VERSION_FOR_HELP);
  if !log_to_stderr {
    write_stderr(&(KATAGO_VERSION_FOR_HELP.to_string() + "\n"));
  }

  let log_all_requests: bool = if cfg.contains("logAllRequests") { cfg.get_bool("logAllRequests")? } else { false };
  let log_all_responses: bool = if cfg.contains("logAllResponses") { cfg.get_bool("logAllResponses")? } else { false };
  let log_errors_and_warnings: bool = if cfg.contains("logErrorsAndWarnings") { cfg.get_bool("logErrorsAndWarnings")? } else { true };
  let log_search_info: bool = if cfg.contains("logSearchInfo") { cfg.get_bool("logSearchInfo")? } else { false };

  let warn_unused_fields: bool = if cfg.contains("warnUnusedFields") { cfg.get_bool("warnUnusedFields")? } else { true };

  let mut default_params: SearchParams = SearchParams::new();
  let mut default_perspective: Player = C_EMPTY;
  load_params(&human_model_file, &mut cfg, &mut default_params, &mut default_perspective, C_EMPTY)?;

  let pattern_bonus_table: Option<Box<PatternBonusTable>>;
  {
    let mut tables: Vec<Option<Box<PatternBonusTable>>> = Setup::load_avoid_sgf_pattern_bonus_tables(&mut cfg)?;
    testAssert!(tables.len() == 1);
    pattern_bonus_table = tables.remove(0);
  }

  let analysis_pv_len: i32 = if cfg.contains("analysisPVLen") { cfg.get_int("analysisPVLen", 1, 100)? } else { 15 };
  let assume_multiple_starting_black_moves_are_handicap: bool =
    if cfg.contains("assumeMultipleStartingBlackMovesAreHandicap") { cfg.get_bool("assumeMultipleStartingBlackMovesAreHandicap")? } else { true };
  let prevent_encore: bool = if cfg.contains("preventCleanupPhase") { cfg.get_bool("preventCleanupPhase")? } else { true };

  let nn_eval: Rc<NNEvaluator>;
  let mut human_eval: Option<Rc<NNEvaluator>> = None;
  {
    Setup::initialize_session(&mut cfg, host.nn_host);
    let expected_concurrent_evals: i32 = num_analysis_threads * default_params.num_threads;
    let default_require_exact_nn_len: bool = false;
    let default_max_batch_size: i32 = -1;
    let disable_fp16: bool = false;
    let expected_sha256: String = String::new();
    nn_eval = Setup::initialize_nn_evaluator(
      &model_file,
      &model_file,
      &expected_sha256,
      &mut cfg,
      &mut seed_rand,
      expected_concurrent_evals,
      NNPos::MAX_BOARD_LEN,
      NNPos::MAX_BOARD_LEN,
      default_max_batch_size,
      default_require_exact_nn_len,
      disable_fp16,
      Setup::SETUP_FOR_ANALYSIS,
      host.nn_host,
    )
    .await?;
    if human_model_file != "" {
      let h: Rc<NNEvaluator> = Setup::initialize_nn_evaluator(
        &human_model_file,
        &human_model_file,
        &expected_sha256,
        &mut cfg,
        &mut seed_rand,
        expected_concurrent_evals,
        NNPos::MAX_BOARD_LEN,
        NNPos::MAX_BOARD_LEN,
        default_max_batch_size,
        default_require_exact_nn_len,
        disable_fp16,
        Setup::SETUP_FOR_ANALYSIS,
        host.nn_host,
      )
      .await?;
      if !h.requires_sgf_metadata() {
        let mut warning: String = String::new();
        warning += "WARNING: Human model was not trained from SGF metadata to vary by rank! Did you pass the wrong model for -human-model?\n";
        log::info!("{}", warning);
        if !log_to_stderr {
          write_stderr(&(warning.clone() + "\n"));
        }
      }
      human_eval = Some(h);
    }
  }

  // PORT NOTE: the `#ifndef USE_EIGEN_BACKEND` block (burn is not Eigen).
  {
    let nn_max_batch_size_total: i32 = nn_eval.get_num_gpus() * nn_eval.get_max_batch_size();
    let num_threads_total: i32 = default_params.num_threads * num_analysis_threads;
    if nn_max_batch_size_total as f64 * 1.5 <= num_threads_total as f64 {
      log::info!(
        "{}",
        crate::strprintf!(
          "Note: nnMaxBatchSize * number of GPUs (%d) is smaller than numSearchThreads * numAnalysisThreads (%d)",
          nn_max_batch_size_total,
          num_threads_total
        )
      );
      log::info!("The number of simultaneous threads that might query the GPU could be larger than the batch size that the GPU will handle at once.");
      log::info!("It may improve performance to increase nnMaxBatchSize, unless you are constrained on GPU memory.");
    }
  }

  //Check for unused config keys
  {
    let mut cerr: String = String::new();
    cfg.warn_unused_keys(&mut cerr);
    write_stderr(&cerr);
  }
  {
    let mut cerr: String = String::new();
    Setup::maybe_warn_human_sl_params(&default_params, Some(&nn_eval), human_eval.as_deref(), &mut cerr, true);
    write_stderr(&cerr);
  }

  log::info!("{}", "Loaded config ".to_string() + &cfg.get_file_name());
  log::info!("{}", "Loaded model ".to_string() + &model_file);
  // upstream: cmd.logOverrides(logger) (commandline.cpp)
  for override_config in &host.args.override_configs {
    if override_config != "" {
      let newkvs: BTreeMap<String, String> = ConfigParser::parse_comma_separated(override_config)?;
      for (k, v) in &newkvs {
        log::info!("{}", "Config override: ".to_string() + k + " = " + v);
      }
    }
  }

  if human_model_file != "" && !cfg.contains("humanSLProfile") && human_eval.as_ref().expect("humanModelFile given").requires_sgf_metadata() {
    log::info!(
      "Warning: Provided -human-model but humanSLProfile is not yet set. The human SL model will only be used on queries that provide humanSLProfile in overrideSettings."
    );
    if !log_to_stderr {
      write_stderr(
        "Warning: Provided -human-model but humanSLProfile is not yet set. The human SL model will only be used on queries that provide humanSLProfile in overrideSettings.\n",
      );
    }
  }

  //Expected possible keys for queries
  let expected_keys: BTreeSet<String> = [
    "id",
    "action",
    "terminateId",
    "turnNumbers",
    "boardXSize",
    "boardYSize",
    "initialStones",
    "moves",
    "initialPlayer",
    "analyzeTurns",
    "priorities",
    "rules",
    "komi",
    "whiteHandicapBonus",
    "overrideSettings",
    "maxVisits",
    "analysisPVLen",
    "rootFpuReductionMax",
    "rootPolicyTemperature",
    "includeMovesOwnership",
    "includeMovesOwnershipStdev",
    "includeOwnership",
    "includeOwnershipStdev",
    "includePolicy",
    "includePVVisits",
    "includeNoResultValue",
    "reportDuringSearchEvery",
    "firstReportDuringSearchAfter",
    "priority",
    "allowMoves",
    "avoidMoves",
  ]
  .iter()
  .map(|s| s.to_string())
  .collect();

  // Common eval cache for all analysis threads
  let mut eval_cache: Option<Rc<EvalCacheTable>> = None;
  if default_params.use_eval_cache {
    eval_cache = Some(Rc::new(EvalCacheTable::new(default_params.subtree_value_bias_table_num_shards)));
  }

  let mut bots: Vec<AsyncBot> = Vec::new();
  for _thread_idx in 0..num_analysis_threads {
    // PORT NOTE: GCC arg order: the right operand of `+` is evaluated first (checked with the fixture compiler).
    let search_rand_seed_rhs: String = Global::uint64_to_hex_string(seed_rand.next_uint64());
    let search_rand_seed_lhs: String = Global::uint64_to_hex_string(seed_rand.next_uint64());
    let search_rand_seed: String = search_rand_seed_lhs + &search_rand_seed_rhs;
    let mut bot: AsyncBot = AsyncBot::new_human_eval(&default_params, nn_eval.clone(), human_eval.clone(), &search_rand_seed)?;
    bot.set_copy_of_external_pattern_bonus_table(&pattern_bonus_table);
    bot.set_external_eval_cache(&eval_cache);
    // PORT NOTE: `threads.emplace_back(analysisLoopProtected,bot,threadIdx)`: the loops start below.
    bots.push(bot);
  }

  let ctx: Rc<AnalysisContext> = Rc::new(AnalysisContext {
    cfg,
    human_model_file: human_model_file.clone(),
    log_all_requests,
    log_all_responses,
    log_errors_and_warnings,
    log_search_info,
    warn_unused_fields,
    default_params,
    default_perspective,
    analysis_pv_len,
    assume_multiple_starting_black_moves_are_handicap,
    prevent_encore,
    nn_eval: nn_eval.clone(),
    human_eval: human_eval.clone(),
    expected_keys,
    to_write_queue: AsyncQueue::new(),
    write_stdout: host.write_stdout,
    to_analyze_queue: AsyncPriorityQueue::new(),
    num_requests_so_far: Cell::new(0),
    internal_id_counter: Cell::new(0),
    open_requests: RefCell::new(BTreeMap::new()),
    eval_cache,
    bots: bots.iter().map(|b| b.get_stop_handle()).collect(),
    loop_failed: Cell::new(false),
    loop_failed_notify: WakerList::new(),
    request_loop_waiting: Cell::new(false),
    request_loop_idle_notify: WakerList::new(),
  });

  log::info!("{}", "Analyzing up to ".to_string() + &Global::int_to_string(num_analysis_threads) + " positions at a time in parallel");
  log::info!("Started, ready to begin handling requests");
  if !log_to_stderr {
    write_stderr("Started, ready to begin handling requests\n");
  }

  let input: &mut (dyn Stream<Item = Vec<u8>> + Unpin) = host.input;
  let shutdown = |ctx: &AnalysisContext, quit_without_waiting: bool| {
    if quit_without_waiting {
      //Making this readOnly will halt futher output that isn't already queued and signal the write loop thread to terminate.
      ctx.to_write_queue.set_read_only();
      //Making this readOnly should signal the analysis loop threads to terminate once they have nothing left.
      ctx.to_analyze_queue.set_read_only();
      //Interrupt any searches going on to help the analysis threads realize to terminate faster.
      for i in 0..ctx.bots.len() {
        ctx.bots[i].stop_without_wait();
      }
      for i in 0..ctx.bots.len() {
        ctx.bots[i].set_killed();
      }
    } else {
      //Making this readOnly should signal the analysis loop threads to terminate once they have nothing left.
      ctx.to_analyze_queue.set_read_only();
    }
  };

  //If request loop raises an exception, we need to log here BEFORE destructing main context, because in some cases
  //gameThreads[i].join() will abort without useful exception due to thread not being joinable,
  //hiding the real exception.
  let main_thread = async {
    let r: Result<(), KataError> = request_loop(&ctx, input).await;
    // PORT NOTE: scheduling (file header): no more input.
    ctx.request_loop_waiting.set(true);
    ctx.request_loop_idle_notify.notify_all();
    if let Err(e) = &r {
      // upstream: Logger::logThreadUncaught("request loop", &logger, requestLoop) (logger.cpp)
      if !ctx.loop_failed.get() {
        log_thread_uncaught_failed(&ctx, "request loop", e).await;
      }
    }
    // PORT NOTE: once the process has terminated, the loops are already stopping (quitWithoutWaiting).
    shutdown(&ctx, quit_without_waiting || ctx.loop_failed.get());
    r
  };
  let analysis_threads = async {
    let loops: Vec<_> =
      bots.into_iter().enumerate().map(|(thread_idx, bot)| Box::pin(analysis_loop_protected(&ctx, bot, thread_idx as i32))).collect();
    //Wait patiently for everything to finish
    let results: Vec<Result<(), KataError>> = futures::future::join_all(loops).await;
    //Signal the write loop thread to terminate
    ctx.to_write_queue.set_read_only();
    results
  };
  let (main_result, thread_results, ()) = futures::future::join3(main_thread, analysis_threads, write_loop(&ctx)).await;

  // delete bots: dropped with their loops.

  if ctx.loop_failed.get() {
    // PORT NOTE: the process has terminated upstream (file header): no closing log lines. The serve
    // tasks still end (PORTING.md §11 Lifecycle) so the host's executor can stop.
    nn_eval.kill_server_threads();
    if let Some(human_eval) = &human_eval {
      human_eval.kill_server_threads();
    }
    main_result?;
    for r in thread_results {
      r?;
    }
    unreachable!("loop_failed is set only by a failing loop, whose error is returned");
  }

  log::info!("{}", nn_eval.get_model_file_name());
  log::info!("{}", "NN rows: ".to_string() + &Global::uint64_to_string(nn_eval.num_rows_processed()));
  log::info!("{}", "NN batches: ".to_string() + &Global::uint64_to_string(nn_eval.num_batches_processed()));
  log::info!("{}", "NN avg batch size: ".to_string() + &Global::double_to_string(nn_eval.average_processed_batch_size()));
  if let Some(human_eval) = &human_eval {
    log::info!("{}", human_eval.get_model_file_name());
    log::info!("{}", "NN rows: ".to_string() + &Global::uint64_to_string(human_eval.num_rows_processed()));
    log::info!("{}", "NN batches: ".to_string() + &Global::uint64_to_string(human_eval.num_batches_processed()));
    log::info!("{}", "NN avg batch size: ".to_string() + &Global::double_to_string(human_eval.average_processed_batch_size()));
  }
  // delete nnEval; delete humanEval;
  // PORT NOTE (PORTING.md §11 Lifecycle): the serve tasks hold an Rc, so the evaluators are killed
  // explicitly here; `delete` joins the server threads (each logs "GPU ... finishing" as it ends) before
  // the next log line, so their end is awaited (D8's join).
  nn_eval.kill_server_threads();
  nn_eval.server_threads_exited.wait_until(|| nn_eval.num_server_threads_alive.get() == 0).await;
  if let Some(human_eval) = &human_eval {
    human_eval.kill_server_threads();
    human_eval.server_threads_exited.wait_until(|| human_eval.num_server_threads_alive.get() == 0).await;
  }
  host.nn_host.global_cleanup();
  ScoreValue::free_tables();

  main_result?;
  for r in thread_results {
    r?;
  }
  log::info!("All cleaned up, quitting");
  Ok(0)
}
