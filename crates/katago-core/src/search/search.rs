// Ported from KataGo cpp/search/search.cpp, cpp/search/search.h @ ba938676d7f42d70950b3a535af2466fb642008c
//
//-------------------------------------------------------------------------------------
//This file contains the main core logic of the search.
//-------------------------------------------------------------------------------------
//
// PORT NOTE (PORTING.md §4, §11, OWNERSHIP.tsv): search threads are virtual threads, i.e. futures
// joined in runWholeSearch; runSinglePlayout, playoutDescend, beginSearch, computeRootValues and
// the runWholeSearch* overloads are async. Methods reachable from the playout loop take `&self`
// and mutate only Cell/RefCell fields. Nodes live in `node_arena`; stored node pointers are
// NodeIdx. Every mutex/spin lock is removed (their bodies kept); `createMutexIdxForNode` still
// draws from the thread's Rand (D12). `Logger*` is replaced by the log crate (ctor param dropped).
// Deferred (DECISIONS.md D7, manifest [[ignored]] deferred): eval cache (evalcache.cpp), pattern
// bonus tables (patternbonustable.cpp), anti-mirror (searchmirror.cpp): their fields and call
// sites stay, their types have no values, so the fields are always None.

use std::cell::Cell;
use std::cell::RefCell;
use std::collections::BTreeSet;
use std::collections::HashSet;
use std::rc::Rc;

use crate::core::fancymath::FancyMath;
use crate::core::global::Global;
use crate::core::global::KataError;
use crate::core::hash::Hash128;
use crate::core::rand::Rand;
use crate::core::timer::ClockTimer;
use crate::game::board::get_opp;
use crate::game::board::Board;
use crate::game::board::Color;
use crate::game::board::Loc;
use crate::game::board::Location;
use crate::game::board::Player;
use crate::game::board::C_EMPTY;
use crate::game::board::P_BLACK;
use crate::game::boardhistory::BoardHistory;
use crate::game::boardhistory::KoHashTable;
use crate::game::graphhash::GraphHash;
use crate::game::rules::Rules;
use crate::neuralnet::nneval::NNEvaluator;
use crate::neuralnet::nneval::NNResultBuf;
use crate::neuralnet::nninputs::NNOutput;
use crate::neuralnet::nninputs::NNPos;
use crate::neuralnet::nninputs::ScoreValue;
use crate::neuralnet::nninputs::SymmetryHelpers;
use crate::port::cpp;
use crate::port::executor::yield_now;
use crate::port::executor::LocalBoxFuture;
use crate::port::node_arena::NodeArena;
use crate::port::node_arena::NodeIdx;
use crate::search::distributiontable::DistributionTable;
use crate::search::searchnode::MoreNodeStats;
use crate::search::searchnode::SearchChildPointer;
use crate::search::searchnode::SearchNode;
use crate::search::searchnode::SearchNodeChildrenReference;
use crate::search::searchnode::SearchNodeState;
use crate::search::searchnodetable::SearchNodeTable;
use crate::search::searchparams::SearchParams;
use crate::search::subtreevaluebiastable::SubtreeValueBiasEntry;
use crate::search::subtreevaluebiastable::SubtreeValueBiasTable;
use crate::search::timecontrols::TimeControls;
use crate::testAssert;

/// upstream: PatternBonusTable (patternbonustable.h)
/// TODO(port): deferred (patternbonustable.cpp): placeholder with no values (DECISIONS.md D7), so
/// `Search::patternBonusTable` / `externalPatternBonusTable` are always None until it is ported.
pub enum PatternBonusTable {}

/// upstream: EvalCacheTable (evalcache.h)
/// TODO(port): deferred (evalcache.cpp): placeholder with no values (DECISIONS.md D7), so
/// `Search::evalCache` is always None until it is ported.
pub enum EvalCacheTable {}

//-----------------------------------------------------------------------------------------

/// upstream: SearchThread (search.h)
//Per-thread state
pub struct SearchThread {
  pub thread_idx: i32,

  pub pla: Player,
  pub board: Board,
  pub history: BoardHistory,
  pub graph_hash: Hash128,
  //The path we trace down the graph as we do a playout
  pub graph_path: HashSet<NodeIdx>,

  //Tracks whether this thread did something that "should" be counted as a playout
  //for the purpose of playout limits
  pub should_count_playout: bool,

  pub rand: Rand,

  /// PORT NOTE (OWNERSHIP.tsv): inline upstream; Rc so the NN query queue can hold it while this
  /// virtual thread awaits.
  pub nn_result_buf: Rc<NNResultBuf>,
  pub stats_buf: Vec<MoreNodeStats>,

  pub upper_bound_visits_left: f64,

  //Occasionally we may need to swap out an NNOutput from a node mid-search.
  //However, to prevent access-after-delete races, the thread that swaps one out stores
  //it here instead of deleting it, so that pointers and accesses to it remain valid.
  pub old_nn_outputs_to_clean_up: Vec<Rc<NNOutput>>,

  //Just controls some debug output
  pub illegal_move_hashes: BTreeSet<Hash128>,
}

/// upstream: makeSeed (search.cpp) static
fn make_seed(search: &Search, thread_idx: i32) -> String {
  let mut ss: String = String::new();
  ss += &search.rand_seed;
  ss += "$searchThread$";
  ss += &thread_idx.to_string();
  ss += "$";
  ss += &search.root_board.pos_hash.to_string();
  ss += "$";
  ss += &search.root_history.move_history.len().to_string();
  ss += "$";
  ss += &search.num_searches_begun.to_string();
  ss
}

impl SearchThread {
  /// upstream: SearchThread::SearchThread (search.cpp)
  pub fn new(t_idx: i32, search: &Search) -> SearchThread {
    let mut s = SearchThread {
      thread_idx: t_idx,
      pla: search.root_pla,
      board: search.root_board.clone(),
      history: search.root_history.clone(),
      graph_hash: search.root_graph_hash,
      graph_path: HashSet::new(),
      should_count_playout: false,
      rand: Rand::new_seed(&make_seed(search, t_idx)),
      nn_result_buf: Rc::new(NNResultBuf::new()),
      stats_buf: Vec::new(),
      upper_bound_visits_left: 1e30,
      old_nn_outputs_to_clean_up: Vec::new(),
      illegal_move_hashes: BTreeSet::new(),
    };
    s.stats_buf.resize(NNPos::MAX_NN_POLICY_SIZE as usize, MoreNodeStats::new());
    s.graph_path.reserve(256);

    //Reserving even this many is almost certainly overkill but should guarantee that we never have hit allocation here.
    s.old_nn_outputs_to_clean_up.reserve(8);
    s
  }

  /// `new SearchThread(tIdx, search)`. PORT NOTE: a SearchThread holds a Board and a BoardHistory
  /// (~40 KB); building it in this small frame and boxing keeps it out of async state machines and
  /// off the 1 MiB debug wasm stack (PORTING.md §13 wasm stack).
  pub fn new_boxed(t_idx: i32, search: &Search) -> Box<SearchThread> {
    Box::new(SearchThread::new(t_idx, search))
  }
}

// SearchThread::~SearchThread deletes the old NNOutput holders: Drop of the Vec<Rc<NNOutput>>.

/// upstream: VALUE_WEIGHT_DEGREES_OF_FREEDOM (search.cpp) static
const VALUE_WEIGHT_DEGREES_OF_FREEDOM: f64 = 3.0;

/// upstream: FORCE_NON_TERMINAL_HASH (search.cpp) static
//Based on sha256 of "search.cpp FORCE_NON_TERMINAL_HASH"
const FORCE_NON_TERMINAL_HASH: Hash128 = Hash128::new(0xd4c31800cb8809e2u64, 0xf75f9d2083f2ffcau64);

/// upstream: Search (search.h)
pub struct Search {
  //================================================================================================================
  // Constant/immutable during search
  //================================================================================================================
  pub root_pla: Player,
  pub root_board: Board,
  pub root_history: BoardHistory,
  pub root_graph_hash: Hash128,
  pub root_hint_loc: Loc,

  //External user-specified moves that are illegal or that should be nontrivially searched, and the number of turns for which they should
  //be excluded. Empty if not active, else of length MAX_ARR_SIZE and nonzero anywhere a move should be banned, for the number of ply
  //of depth that it should be banned.
  pub avoid_move_until_by_loc_black: Vec<i32>,
  pub avoid_move_until_by_loc_white: Vec<i32>,
  pub avoid_move_until_rescale_root: bool, // When avoiding moves at the root, rescale the root policy to sum to 1.

  //If rootSymmetryPruning==true and the board is symmetric, mask all the equivalent copies of each move except one.
  pub root_sym_dup_loc: [bool; Board::MAX_ARR_SIZE as usize],
  //If rootSymmetryPruning==true, symmetries under which the root board and history are invariant, including some heuristics for ko and encore-related state.
  pub root_symmetries: Vec<i32>,
  pub root_prune_only_symmetries: Vec<i32>,

  //Strictly pass-alive areas in the root board position
  pub root_safe_area: Box<[Color; Board::MAX_ARR_SIZE as usize]>,
  //Used to center for dynamic scorevalue
  pub recent_score_center: f64,

  //If the opponent is mirroring, then the color of that opponent, for countering mirroring
  pub mirroring_pla: Player,
  pub mirror_advantage: f64, //Number of points the opponent wins by if mirror holds indefinitely.
  pub mirror_center_symmetry_error: f64,

  pub always_include_owner_map: bool,

  pub search_params: SearchParams,
  pub num_searches_begun: i64,
  /// PORT NOTE: Cell because the tree walks (applyRecursively*, searchmultithreadhelpers.rs) bump it
  /// through &self while their callbacks also borrow the Search (recursivelyRecomputeStats).
  pub search_node_age: Cell<u32>,
  pub pla_that_search_is_for: Player,
  pub pla_that_search_is_for_last_search: Player,
  pub last_search_num_playouts: i64,
  pub effective_search_time_carried_over: f64, //Effective search time carried over from previous moves due to ponder/tree reuse

  pub rand_seed: String,

  //Contains all koHashes of positions/situations up to and including the root
  pub root_ko_hash_table: Box<KoHashTable>,

  //Precomputed distribution for downweighting child values based on their values
  pub value_weight_distribution: Box<DistributionTable>,

  //Precomputed Fancymath::normToTApprox values, for a fixed Z
  pub norm_to_t_approx_z: f64,
  pub norm_to_t_approx_table: Vec<f64>,

  //Pattern bonuses are currently only looked up for shapes completed by the player who the search is for.
  //Implicitly these utility adjustments "assume" the opponent likes the negative of our adjustments.
  // TODO(port): deferred (patternbonustable.cpp): always None.
  pub pattern_bonus_table: Option<Box<PatternBonusTable>>,
  pub external_pattern_bonus_table: Option<Box<PatternBonusTable>>,

  // TODO(port): deferred (evalcache.cpp): always None.
  pub eval_cache: Option<Rc<EvalCacheTable>>,

  /// PORT NOTE: RefCell because applyRecursively* (which run through &self, see search_node_age)
  /// draw from it.
  pub non_search_rand: RefCell<Rand>, //only for use not in search, since rand isn't threadsafe

  //================================================================================================================
  // Externally owned values
  //================================================================================================================

  // PORT NOTE: `Logger* logger` removed (log crate).
  pub nn_evaluator: Rc<NNEvaluator>,
  pub human_evaluator: Option<Rc<NNEvaluator>>,
  pub nn_x_len: i32,
  pub nn_y_len: i32,
  pub policy_size: i32,

  //================================================================================================================
  // Mutated during search
  //================================================================================================================

  pub root_node: Option<NodeIdx>,
  /// PORT NOTE (OWNERSHIP.tsv): new; owns every SearchNode (the root and the node table's nodes).
  pub node_arena: NodeArena<SearchNode>,
  pub node_table: Box<SearchNodeTable>,
  /// PORT NOTE (OWNERSHIP.tsv): `MutexPool* mutexPool` -> its mutex count, which
  /// createMutexIdxForNode still uses.
  pub mutex_pool_num_mutexes: u32,
  pub subtree_value_bias_table: Option<Box<SubtreeValueBiasTable>>,

  //Thread pool
  // PORT NOTE: numThreadsSpawned, threads, threadTasks, threadTasksRemaining removed (PORTING.md §11).

  //Occasionally we may need to swap out an NNOutput from a node mid-search.
  //However, to prevent access-after-delete races, this vector collects them after a thread exits, and is cleaned up
  //very lazily only when a new search begins or the search is cleared.
  // PORT NOTE: oldNNOutputsToCleanUpMutex removed; RefCell because transferOldNNOutputs runs at the
  // end of each virtual thread through &self.
  pub old_nn_outputs_to_clean_up: RefCell<Vec<Rc<NNOutput>>>,
}

/// The by-reference captures of runWholeSearch's `searchLoop` lambda.
/// PORT NOTE (PORTING.md §11): the lambda becomes the async fn `search_loop`; the locals it shares
/// across threads are Cells borrowed by every virtual thread.
struct SearchLoopCaptures<'a> {
  timer: &'a ClockTimer,
  num_playouts_shared: &'a Cell<i64>,
  num_non_playout_visits: i64,
  tc_max_time: &'a Cell<f64>,
  upper_bound_visits_left_due_to_time: &'a Cell<f64>,
  tc: &'a TimeControls,
  has_max_time: bool,
  has_tc: bool,
  should_stop_now: &'a Cell<bool>,
  should_stop_early: Option<&'a dyn Fn() -> bool>,
  max_visits: i64,
  max_playouts: i64,
  max_time: f64,
  pondering: bool,
  search_factor: f64,
}

impl Search {
  /// upstream: Search::POLICY_ILLEGAL_SELECTION_VALUE etc. (search.h)
  pub(crate) const POLICY_ILLEGAL_SELECTION_VALUE: f64 = -1e50;
  pub(crate) const FUTILE_VISITS_PRUNE_VALUE: f64 = -1e40;
  // Only referenced by a commented-out line in searchexplorehelpers.cpp upstream.
  #[allow(dead_code)]
  pub(crate) const EVALUATING_SELECTION_VALUE_PENALTY: f64 = 1e20;

  /// upstream: Search::Search(const SearchParams&, NNEvaluator*, Logger*, const std::string&) (search.cpp)
  //Note - randSeed controls a few things in the search, but a lot of the randomness actually comes from
  //random symmetries of the neural net evaluations, see nneval.h
  /// PORT NOTE: `Logger*` dropped. Delegates to the humanEval overload with NULL, which cannot throw
  /// then (its only throws are the humanEval size check and KoHashTable::recompute's 1e9 history
  /// limit, unreachable for the empty root history).
  pub fn new(params: &SearchParams, nn_eval: Rc<NNEvaluator>, r_seed: &str) -> Search {
    Search::new_human_eval(params, nn_eval, None, r_seed).expect("upstream: cannot throw with humanEval NULL and an empty history")
  }

  /// upstream: Search::Search(const SearchParams&, NNEvaluator*, NNEvaluator*, Logger*, const std::string&) (search.cpp)
  /// PORT NOTE: `Logger*` dropped; throws -> Result.
  pub fn new_human_eval(params: &SearchParams, nn_eval: Rc<NNEvaluator>, human_eval: Option<Rc<NNEvaluator>>, r_seed: &str) -> Result<Search, KataError> {
    let node_table: Box<SearchNodeTable> = Box::new(SearchNodeTable::new(params.node_table_shards_power_of_two));
    let mut s = Search {
      root_pla: P_BLACK,
      root_board: Board::new(),
      root_history: BoardHistory::new(),
      root_graph_hash: Hash128::default(),
      root_hint_loc: Board::NULL_LOC,
      avoid_move_until_by_loc_black: Vec::new(),
      avoid_move_until_by_loc_white: Vec::new(),
      avoid_move_until_rescale_root: false,
      root_sym_dup_loc: [false; Board::MAX_ARR_SIZE as usize],
      root_symmetries: Vec::new(),
      root_prune_only_symmetries: Vec::new(),
      // PORT NOTE: `rootSafeArea(NULL)` then `new Color[MAX_ARR_SIZE]` in the body (uninitialized; zeroed here).
      root_safe_area: Box::new([C_EMPTY; Board::MAX_ARR_SIZE as usize]),
      recent_score_center: 0.0,
      mirroring_pla: C_EMPTY,
      mirror_advantage: 0.0,
      mirror_center_symmetry_error: 1e10,
      always_include_owner_map: false,
      search_params: params.clone(),
      num_searches_begun: 0,
      search_node_age: Cell::new(0),
      pla_that_search_is_for: C_EMPTY,
      pla_that_search_is_for_last_search: C_EMPTY,
      last_search_num_playouts: 0,
      effective_search_time_carried_over: 0.0,
      rand_seed: r_seed.to_string(),
      // PORT NOTE: `rootKoHashTable(NULL)` then `new KoHashTable()` in the body.
      root_ko_hash_table: Box::new(KoHashTable::new()),
      // PORT NOTE: `valueWeightDistribution(NULL)` then `new DistributionTable(...)` in the body.
      value_weight_distribution: Box::new(DistributionTable::new(
        &|z: f64| FancyMath::tdistpdf(z, VALUE_WEIGHT_DEGREES_OF_FREEDOM),
        &|z: f64| FancyMath::tdistcdf(z, VALUE_WEIGHT_DEGREES_OF_FREEDOM),
        -50.0,
        50.0,
        2000,
      )),
      norm_to_t_approx_z: 0.0,
      norm_to_t_approx_table: Vec::new(),
      pattern_bonus_table: None,
      external_pattern_bonus_table: None,
      eval_cache: None,
      non_search_rand: RefCell::new(Rand::new_seed(&(r_seed.to_string() + "$nonSearchRand"))),
      nn_evaluator: nn_eval,
      human_evaluator: human_eval,
      nn_x_len: 0,
      nn_y_len: 0,
      policy_size: 0,
      root_node: None,
      node_arena: NodeArena::new(),
      // PORT NOTE: `nodeTable(NULL)` then `new SearchNodeTable(...)` in the body.
      mutex_pool_num_mutexes: node_table.mutex_pool_num_mutexes,
      node_table,
      subtree_value_bias_table: None,
      old_nn_outputs_to_clean_up: RefCell::new(Vec::new()),
    };
    // PORT NOTE: `testAssert(logger != NULL)` dropped with the logger.
    s.nn_x_len = s.nn_evaluator.get_nnx_len();
    s.nn_y_len = s.nn_evaluator.get_nny_len();
    testAssert!(s.nn_x_len > 0 && s.nn_x_len <= NNPos::MAX_BOARD_LEN);
    testAssert!(s.nn_y_len > 0 && s.nn_y_len <= NNPos::MAX_BOARD_LEN);
    s.policy_size = NNPos::get_policy_size(s.nn_x_len, s.nn_y_len);

    if let Some(human_evaluator) = &s.human_evaluator {
      if human_evaluator.get_nnx_len() != s.nn_x_len || human_evaluator.get_nny_len() != s.nn_y_len {
        return Err(KataError::StringError("Search::init - humanEval has different nnXLen or nnYLen".to_string()));
      }
    }

    s.root_node = None;
    // PORT NOTE: `mutexPool = new MutexPool(nodeTable->mutexPool->getNumMutexes())` is the
    // mutex_pool_num_mutexes initializer above.

    s.root_history.clear(&s.root_board, s.root_pla, &Rules::new(), 0);
    s.root_ko_hash_table.recompute(&s.root_history)?;
    Ok(s)
  }

  // Search::~Search: clearSearch() and deletes owned tables, then killThreads(): only frees memory
  // (the arena and Boxes drop with the Search; killThreads is a no-op, PORTING.md §11). Omitted.

  /// upstream: Search::getRootBoard (search.cpp)
  pub fn get_root_board(&self) -> &Board {
    &self.root_board
  }
  /// upstream: Search::getRootHist (search.cpp)
  pub fn get_root_hist(&self) -> &BoardHistory {
    &self.root_history
  }
  /// upstream: Search::getRootPla (search.cpp)
  pub fn get_root_pla(&self) -> Player {
    self.root_pla
  }

  /// upstream: Search::getPlayoutDoublingAdvantagePla (search.cpp)
  pub fn get_playout_doubling_advantage_pla(&self) -> Player {
    if self.search_params.playout_doubling_advantage_pla == C_EMPTY {
      self.pla_that_search_is_for
    } else {
      self.search_params.playout_doubling_advantage_pla
    }
  }

  /// upstream: Search::getPos (search.h)
  //Get the NNPos corresponding to a loc, convenience method
  #[inline]
  pub fn get_pos(&self, move_loc: Loc) -> i32 {
    NNPos::loc_to_pos(move_loc, self.root_board.x_size, self.nn_x_len, self.nn_y_len)
  }

  /// upstream: Search::setPosition (search.cpp)
  //Clear all results of search and sets a new position or something else
  /// PORT NOTE: Result because KoHashTable::recompute can throw.
  pub fn set_position(&mut self, pla: Player, board: &Board, history: &BoardHistory) -> Result<(), KataError> {
    self.clear_search();
    self.root_pla = pla;
    self.pla_that_search_is_for = C_EMPTY;
    self.root_board = board.clone();
    self.root_history = history.clone();
    self.root_ko_hash_table.recompute(&self.root_history)?;
    self.avoid_move_until_by_loc_black.clear();
    self.avoid_move_until_by_loc_white.clear();
    Ok(())
  }

  /// upstream: Search::setPlayerAndClearHistory (search.cpp)
  /// PORT NOTE: Result because KoHashTable::recompute can throw.
  pub fn set_player_and_clear_history(&mut self, pla: Player) -> Result<(), KataError> {
    self.clear_search();
    self.root_pla = pla;
    self.pla_that_search_is_for = C_EMPTY;
    self.root_board.clear_simple_ko_loc();
    let rules: Rules = self.root_history.rules;
    //Preserve this value even when we get multiple moves in a row by some player
    let assume_multiple_starting_black_moves_are_handicap: bool = self.root_history.assume_multiple_starting_black_moves_are_handicap;
    let encore_phase: i32 = self.root_history.encore_phase;
    self.root_history.clear(&self.root_board, self.root_pla, &rules, encore_phase);
    self.root_history.set_assume_multiple_starting_black_moves_are_handicap(assume_multiple_starting_black_moves_are_handicap);

    self.root_ko_hash_table.recompute(&self.root_history)?;

    //If changing the player alone, don't clear these, leave the user's setting - the user may have tried
    //to adjust the player or will be calling runWholeSearchAndGetMove with a different player and will
    //still want avoid moves to apply.
    //avoidMoveUntilByLocBlack.clear();
    //avoidMoveUntilByLocWhite.clear();
    Ok(())
  }

  /// upstream: Search::setPlayerIfNew (search.cpp)
  pub fn set_player_if_new(&mut self, pla: Player) -> Result<(), KataError> {
    if pla != self.root_pla {
      self.set_player_and_clear_history(pla)?;
    }
    Ok(())
  }

  /// upstream: Search::setKomiIfNew (search.cpp)
  //Does not clear history, does clear search unless komi is equal.
  pub fn set_komi_if_new(&mut self, new_komi: f32) {
    if self.root_history.rules.komi != new_komi {
      self.clear_search();
      self.root_history.set_komi(new_komi);
    }
  }

  /// upstream: Search::setAvoidMoveUntilByLoc (search.cpp)
  pub fn set_avoid_move_until_by_loc(&mut self, b_vec: &[i32], w_vec: &[i32]) {
    if self.avoid_move_until_by_loc_black == b_vec && self.avoid_move_until_by_loc_white == w_vec {
      return;
    }
    self.clear_search();
    self.avoid_move_until_by_loc_black = b_vec.to_vec();
    self.avoid_move_until_by_loc_white = w_vec.to_vec();
  }

  /// upstream: Search::setAvoidMoveUntilRescaleRoot (search.cpp)
  pub fn set_avoid_move_until_rescale_root(&mut self, b: bool) {
    self.avoid_move_until_rescale_root = b;
  }

  /// upstream: Search::setRootHintLoc (search.cpp)
  pub fn set_root_hint_loc(&mut self, loc: Loc) {
    //When we positively change the hint loc, we clear the search to make absolutely sure
    //that the hintloc takes effect, and that all nnevals (including the root noise that adds the hintloc) has a chance to happen
    if loc != Board::NULL_LOC && self.root_hint_loc != loc {
      self.clear_search();
    }
    self.root_hint_loc = loc;
  }

  /// upstream: Search::setAlwaysIncludeOwnerMap (search.cpp)
  pub fn set_always_include_owner_map(&mut self, b: bool) {
    if !self.always_include_owner_map && b {
      self.clear_search();
    }
    self.always_include_owner_map = b;
  }

  /// upstream: Search::setRootSymmetryPruningOnly (search.cpp)
  pub fn set_root_symmetry_pruning_only(&mut self, v: &[i32]) {
    if self.root_prune_only_symmetries == v {
      return;
    }
    self.clear_search();
    self.root_prune_only_symmetries = v.to_vec();
  }

  /// upstream: Search::setParams (search.cpp)
  pub fn set_params(&mut self, params: &SearchParams) {
    self.clear_search();
    self.search_params = params.clone();
  }

  /// upstream: Search::setParamsNoClearing (search.cpp)
  //Does not clear search
  pub fn set_params_no_clearing(&mut self, params: &SearchParams) {
    self.search_params = params.clone();
  }

  /// upstream: Search::setExternalPatternBonusTable (search.cpp)
  /// TODO(port): deferred (patternbonustable.cpp): PatternBonusTable has no values, so `table` is
  /// always None. `table == externalPatternBonusTable` compares the unique_ptrs (both null here).
  pub fn set_external_pattern_bonus_table(&mut self, table: Option<Box<PatternBonusTable>>) {
    if table.is_none() && self.external_pattern_bonus_table.is_none() {
      return;
    }
    //Probably not actually needed so long as we do a fresh search to refresh and use the new table
    //but this makes behavior consistent with all the other setters.
    self.clear_search();
    self.external_pattern_bonus_table = table;
  }

  /// upstream: Search::setCopyOfExternalPatternBonusTable (search.cpp)
  /// TODO(port): deferred (patternbonustable.cpp): the copy constructor is not ported; a non-null
  /// table cannot exist yet.
  pub fn set_copy_of_external_pattern_bonus_table(&mut self, table: &Option<Box<PatternBonusTable>>) {
    let copy: Option<Box<PatternBonusTable>> = match table {
      None => None,
      Some(t) => match **t {},
    };
    self.set_external_pattern_bonus_table(copy);
  }

  /// upstream: Search::setExternalEvalCache (search.cpp)
  /// TODO(port): deferred (evalcache.cpp): EvalCacheTable has no values, so `cache` is always None.
  pub fn set_external_eval_cache(&mut self, cache: &Option<Rc<EvalCacheTable>>) {
    let same: bool = match (cache, &self.eval_cache) {
      (None, None) => true,
      (Some(a), Some(b)) => Rc::ptr_eq(a, b),
      _ => false,
    };
    if same {
      return;
    }
    self.clear_search();
    self.eval_cache = cache.clone();
  }

  /// upstream: Search::setNNEval (search.cpp)
  /// PORT NOTE: throws -> Result.
  pub fn set_nn_eval(&mut self, nn_eval: Rc<NNEvaluator>) -> Result<(), KataError> {
    self.clear_search();
    self.nn_evaluator = nn_eval;
    self.nn_x_len = self.nn_evaluator.get_nnx_len();
    self.nn_y_len = self.nn_evaluator.get_nny_len();
    testAssert!(self.nn_x_len > 0 && self.nn_x_len <= NNPos::MAX_BOARD_LEN);
    testAssert!(self.nn_y_len > 0 && self.nn_y_len <= NNPos::MAX_BOARD_LEN);
    self.policy_size = NNPos::get_policy_size(self.nn_x_len, self.nn_y_len);

    if let Some(human_evaluator) = &self.human_evaluator {
      if human_evaluator.get_nnx_len() != self.nn_x_len || human_evaluator.get_nny_len() != self.nn_y_len {
        return Err(KataError::StringError("Search::setNNEval - humanEval has different nnXLen or nnYLen".to_string()));
      }
    }
    Ok(())
  }

  /// upstream: Search::clearSearch (search.cpp)
  //Just directly clear search without changing anything
  pub fn clear_search(&mut self) {
    self.effective_search_time_carried_over = 0.0;
    if self.root_node.is_some() {
      self.delete_all_table_nodes_mulithreaded();
      //Root is not stored in node table
      if let Some(root_node) = self.root_node {
        self.node_arena.free(root_node);
        self.root_node = None;
      }
    }
    self.clear_old_nn_outputs();
    self.search_node_age.set(0);
  }

  /// upstream: Search::isLegalTolerant (search.cpp)
  //isLegalTolerant also specially handles players moving multiple times in a row.
  pub fn is_legal_tolerant(&self, move_loc: Loc, move_pla: Player) -> bool {
    self.root_history.is_legal_tolerant(&self.root_board, move_loc, move_pla)
  }

  /// upstream: Search::isLegalStrict (search.cpp)
  pub fn is_legal_strict(&self, move_loc: Loc, move_pla: Player) -> bool {
    move_pla == self.root_pla && self.root_history.is_legal(&self.root_board, move_loc, move_pla)
  }

  /// upstream: Search::makeMove(Loc, Player) (search.cpp)
  //Updates position and preserves the relevant subtree of search
  //If the move is not legal for the specified player, returns false and does nothing, else returns true
  //In the case where the player was not the expected one moving next, also clears history.
  /// PORT NOTE: Result because KoHashTable::recompute can throw.
  pub fn make_move(&mut self, move_loc: Loc, move_pla: Player) -> Result<bool, KataError> {
    self.make_move_prevent_encore(move_loc, move_pla, false)
  }

  /// upstream: Search::makeMove(Loc, Player, bool) (search.cpp)
  pub fn make_move_prevent_encore(&mut self, move_loc: Loc, move_pla: Player, prevent_encore: bool) -> Result<bool, KataError> {
    if !self.is_legal_tolerant(move_loc, move_pla) {
      return Ok(false);
    }

    if move_pla != self.root_pla {
      self.set_player_and_clear_history(move_pla)?;
    }

    //If the white handicap bonus changes due to the move, we will also need to recompute everything since this is
    //basically like a change to the komi.
    let old_white_handicap_bonus_score: f32 = self.root_history.white_handicap_bonus_score;

    //Compute these first so we can know if we need to set forceNonTerminal below.
    self.root_history.make_board_move_assume_legal(&mut self.root_board, move_loc, self.root_pla, Some(&*self.root_ko_hash_table), prevent_encore);
    self.root_pla = get_opp(self.root_pla);
    self.root_ko_hash_table.recompute(&self.root_history)?;

    if let Some(root_node_idx) = self.root_node {
      let mut child: Option<Rc<SearchNode>> = None;
      {
        let root_node: Rc<SearchNode> = self.node_arena.get(root_node_idx);
        let children: SearchNodeChildrenReference = root_node.get_children();
        let children_capacity: i32 = children.get_capacity();
        for i in 0..children_capacity {
          let child_candidate: Option<NodeIdx> = children.index(i).get_if_allocated();
          let child_candidate: NodeIdx = match child_candidate {
            None => break,
            Some(c) => c,
          };
          if children.index(i).get_move_loc_relaxed() == move_loc {
            child = Some(self.node_arena.get(child_candidate));
            break;
          }
        }
      }

      //Just in case, make sure the child has an nnOutput, otherwise no point keeping it.
      //This is a safeguard against any oddity involving node preservation into states that
      //were considered terminal.
      if let Some(c) = &child {
        let nn_output: Option<Rc<NNOutput>> = c.get_nn_output();
        if nn_output.is_none() {
          child = None;
        }
      }

      if let Some(child) = child {
        //Account for time carried over
        {
          let root_visits: i64 = self.node_arena.get(root_node_idx).stats.visits.get() /*acquire*/;
          let child_visits: i64 = child.stats.visits.get() /*acquire*/;
          let mut visit_proportion: f64 = child_visits as f64 / root_visits as f64;
          if visit_proportion > 1.0 {
            visit_proportion = 1.0;
          }
          self.effective_search_time_carried_over =
            self.effective_search_time_carried_over * visit_proportion * self.search_params.tree_reuse_carry_over_time_factor;
        }

        let old_root_node: NodeIdx = root_node_idx;

        //Okay, this is now our new root! Create a copy so as to keep the root out of the node table.
        let copy_subtree_value_bias: bool = false;
        let force_non_terminal: bool = self.root_history.is_game_finished; // Make sure the root isn't considered terminal if game would be finished.
        let new_root: NodeIdx = self.node_arena.alloc(|idx| SearchNode::new_copy(idx, &child, force_non_terminal, copy_subtree_value_bias));
        self.root_node = Some(new_root);
        drop(child);
        //Sweep over the new root marking it as good (calling NULL function), and then delete anything unmarked.
        //This will include the old copy of the child that we promoted to root.
        self.apply_recursively_any_order_mulithreaded(&[new_root], None);
        let old: bool = true;
        self.delete_all_old_or_all_new_table_nodes_and_subtree_value_bias_mulithreaded(old);
        //Old root is not stored in node table, delete it too.
        self.node_arena.free(old_root_node);
      } else {
        self.clear_search();
      }
    }

    //Explicitly clear avoid move arrays when we play a move - user needs to respecify them if they want them.
    self.avoid_move_until_by_loc_black.clear();
    self.avoid_move_until_by_loc_white.clear();

    //If we're newly inferring some moves as handicap that we weren't before, clear since score will be wrong.
    if self.root_history.white_handicap_bonus_score != old_white_handicap_bonus_score {
      self.clear_search();
    }

    //In the case that we are conservativePass and a pass would end the game, need to clear the search.
    //This is because deeper in the tree, such a node would have been explored as ending the game, but now that
    //it's a root pass, it needs to be treated as if it no longer ends the game.
    if self.search_params.conservative_pass && self.root_history.pass_would_end_game(&self.root_board, self.root_pla) {
      self.clear_search();
    }

    //In the case that we're preventing encore, and the phase would have ended, we also need to clear the search
    //since the search was conducted on the assumption that we're going into encore now.
    if prevent_encore && self.root_history.pass_would_end_phase(&self.root_board, self.root_pla) {
      self.clear_search();
    }

    Ok(true)
  }

  /// upstream: Search::runWholeSearchAndGetMove(Player) (search.cpp)
  //Run an entire search from start to finish
  /// PORT NOTE: async (PORTING.md §11); throws -> Result.
  pub async fn run_whole_search_and_get_move(&mut self, move_pla: Player) -> Result<Loc, KataError> {
    self.run_whole_search_and_get_move_pondering(move_pla, false).await
  }

  /// upstream: Search::runWholeSearchAndGetMove(Player, bool) (search.cpp)
  pub async fn run_whole_search_and_get_move_pondering(&mut self, move_pla: Player, pondering: bool) -> Result<Loc, KataError> {
    self.run_whole_search_pondering(move_pla, pondering).await?;
    Ok(self.get_chosen_move_loc())
  }

  /// upstream: Search::runWholeSearch(Player) (search.cpp)
  pub async fn run_whole_search(&mut self, move_pla: Player) -> Result<(), KataError> {
    self.run_whole_search_pondering(move_pla, false).await
  }

  /// upstream: Search::runWholeSearch(Player, std::function<bool()>*) (search.cpp)
  pub async fn run_whole_search_should_stop_early(&mut self, move_pla: Player, should_stop_early: Option<&dyn Fn() -> bool>) -> Result<(), KataError> {
    self.run_whole_search_pondering_should_stop_early(move_pla, false, should_stop_early).await
  }

  /// upstream: Search::runWholeSearch(Player, bool) (search.cpp)
  //Pondering indicates that we are searching "for" the last player that we did a non-ponder search for, and should use ponder search limits.
  pub async fn run_whole_search_pondering(&mut self, move_pla: Player, pondering: bool) -> Result<(), KataError> {
    if move_pla != self.root_pla {
      self.set_player_and_clear_history(move_pla)?;
    }
    let search_begun: Option<&dyn Fn()> = None;
    let should_stop_early: Option<&dyn Fn() -> bool> = None;
    self.run_whole_search_full(search_begun, should_stop_early, pondering, &TimeControls::new(), 1.0).await
  }

  /// upstream: Search::runWholeSearch(Player, bool, std::function<bool()>*) (search.cpp)
  pub async fn run_whole_search_pondering_should_stop_early(
    &mut self,
    move_pla: Player,
    pondering: bool,
    should_stop_early: Option<&dyn Fn() -> bool>,
  ) -> Result<(), KataError> {
    if move_pla != self.root_pla {
      self.set_player_and_clear_history(move_pla)?;
    }
    let search_begun: Option<&dyn Fn()> = None;
    self.run_whole_search_full(search_begun, should_stop_early, pondering, &TimeControls::new(), 1.0).await
  }

  /// upstream: Search::runWholeSearch(const std::function<void()>*, std::function<bool()>*, bool, const TimeControls&, double) (search.cpp)
  /// PORT NOTE: name from PORTING.md Appendix B. async (PORTING.md §11): `performTaskWithThreads(&searchLoop,
  /// capThreads)` becomes a join_all over the virtual threads upstream would use; the shared locals
  /// (numPlayoutsShared, shouldStopNow, tcMaxTime, upperBoundVisitsLeftDueToTime) are Cells.
  pub async fn run_whole_search_full(
    &mut self,
    search_begun: Option<&dyn Fn()>, //If not null, will be called once search has begun and tree inspection is safe
    should_stop_early: Option<&dyn Fn() -> bool>, //If not null and returns true, search will stop soon after
    pondering: bool,
    tc: &TimeControls,
    mut search_factor: f64,
  ) -> Result<(), KataError> {
    let timer: ClockTimer = ClockTimer::new();
    let num_playouts_shared: Cell<i64> = Cell::new(0);
    let should_stop_now: Cell<bool> = Cell::new(false);

    // PORT NOTE: the two `std::atomic_is_lock_free` warnings are dropped (Cells, single-threaded).

    //Do this first, just in case this causes us to clear things and have 0 effective time carried over
    self.begin_search(pondering).await?;
    if let Some(search_begun) = search_begun {
      search_begun();
    }
    let num_non_playout_visits: i64 = self.get_root_visits();

    //Compute caps on search
    let mut max_visits: i64 = if pondering { self.search_params.max_visits_pondering } else { self.search_params.max_visits };
    let mut max_playouts: i64 = if pondering { self.search_params.max_playouts_pondering } else { self.search_params.max_playouts };
    let mut max_time: f64 = if pondering { self.search_params.max_time_pondering } else { self.search_params.max_time };

    {
      //Possibly reduce computation time, for human friendliness
      let move_history_len: usize = self.root_history.move_history.len();
      if move_history_len >= 1 && self.root_history.move_history[move_history_len - 1].loc == Board::PASS_LOC {
        if move_history_len >= 3 && self.root_history.move_history[move_history_len - 3].loc == Board::PASS_LOC {
          search_factor *= self.search_params.search_factor_after_two_pass;
        } else {
          search_factor *= self.search_params.search_factor_after_one_pass;
        }
      }

      if search_factor != 1.0 {
        let cap: f64 = (1i64 << 62) as f64;
        // PORT NOTE: C++ UB for NaN; x86-64 gives INT64_MIN.
        max_visits = cpp::static_cast_i64_f64(cpp::cpp_min(cap, max_visits as f64 * search_factor).ceil());
        max_playouts = cpp::static_cast_i64_f64(cpp::cpp_min(cap, max_playouts as f64 * search_factor).ceil());
        max_time = max_time * search_factor;
      }
    }

    let mut cap_threads: i32 = 0x3fffFFFF;
    if self.search_params.min_playouts_per_thread > 0.0 {
      let num_new_playouts: i64 = std::cmp::min(max_visits - num_non_playout_visits, max_playouts);
      let cap: f64 = num_new_playouts as f64 / self.search_params.min_playouts_per_thread;
      if !cap.is_nan() && cap < 0x3fffFFFF as f64 {
        // PORT NOTE: C++ UB for a cap below INT_MIN; x86-64 gives INT_MIN.
        cap_threads = std::cmp::max(1, cpp::static_cast_i32_f64(cap.floor()));
      }
    }

    //Apply time controls. These two don't particularly need to be synchronized with each other so its fine to have two separate atomics.
    let tc_max_time: Cell<f64> = Cell::new(1e30);
    let upper_bound_visits_left_due_to_time: Cell<f64> = Cell::new(1e30);
    let has_max_time: bool = max_time < 1.0e12;
    let has_tc: bool = !pondering && !tc.is_effectively_unlimited_time();
    if !pondering && (has_tc || has_max_time) {
      let root_visits: i64 = num_playouts_shared.get() /*relaxed*/ + num_non_playout_visits;
      let time_used: f64 = timer.get_seconds();
      let mut tc_limit: f64 = 1e30;
      if has_tc {
        tc_limit = self.recompute_search_time_limit(tc, time_used, search_factor, root_visits)?;
        tc_max_time.set(tc_limit) /*release*/;
      }
      let upper_bound_visits: f64 = self.compute_upper_bound_visits_left_due_to_time(root_visits, time_used, cpp::cpp_min(tc_limit, max_time));
      upper_bound_visits_left_due_to_time.set(upper_bound_visits) /*release*/;
    }

    let actual_search_start_time: f64 = timer.get_seconds();
    {
      let this: &Search = &*self;
      let captures: SearchLoopCaptures = SearchLoopCaptures {
        timer: &timer,
        num_playouts_shared: &num_playouts_shared,
        num_non_playout_visits,
        tc_max_time: &tc_max_time,
        upper_bound_visits_left_due_to_time: &upper_bound_visits_left_due_to_time,
        tc,
        has_max_time,
        has_tc,
        should_stop_now: &should_stop_now,
        should_stop_early,
        max_visits,
        max_playouts,
        max_time,
        pondering,
        search_factor,
      };
      // PORT NOTE (PORTING.md §11): performTaskWithThreads(&searchLoop, capThreads) runs the task on
      // 1 + min(capThreads-1, numThreads-1) threads; here as many virtual threads, joined.
      let mut num_additional_threads_to_use: i32 = std::cmp::min(cap_threads - 1, this.search_params.num_threads - 1);
      if num_additional_threads_to_use < 0 {
        num_additional_threads_to_use = 0;
      }
      let mut loops: Vec<LocalBoxFuture<'_, Result<(), KataError>>> = Vec::new();
      for thread_idx in 0..(num_additional_threads_to_use + 1) {
        loops.push(Box::pin(this.search_loop(thread_idx, &captures)));
      }
      let results: Vec<Result<(), KataError>> = futures::future::join_all(loops).await;
      for r in results {
        r?;
      }
    }

    //If the search did not actually do anything, we need to still make sure to update the root node if it needs
    //such an update (since root params may differ from tree params).
    if let Some(root_node_idx) = self.root_node {
      let root_node: Rc<SearchNode> = self.node_arena.get(root_node_idx);
      if root_node.node_age.get() /*acquire*/ != self.search_node_age.get() {
        //Also check if the root node even got an nn eval or not. It might not be, if we quit the search instantly upon
        //start due to an explicit stop signal
        if root_node.get_nn_output().is_some() {
          let thread_idx: i32 = 0;
          let is_root: bool = true;
          let mut thread: Box<SearchThread> = SearchThread::new_boxed(thread_idx, self);
          self.maybe_recompute_existing_nn_output(&mut thread, &root_node, is_root).await?;
        }
      }
    }

    if self.search_params.use_eval_cache && self.search_params.use_graph_search && self.eval_cache.is_some() && self.root_node.is_some() && self.mirroring_pla == C_EMPTY {
      let root_node: Rc<SearchNode> = self.node_arena.get(self.root_node.expect("checked above"));
      self.recursively_record_eval_cache(&root_node);
    }

    //Relaxed load is fine since numPlayoutsShared should be synchronized already due to the joins
    self.last_search_num_playouts = num_playouts_shared.get() /*relaxed*/;
    self.effective_search_time_carried_over += timer.get_seconds() - actual_search_start_time;
    Ok(())
  }

  /// upstream: the `searchLoop` lambda in Search::runWholeSearch (search.cpp)
  /// PORT NOTE (PORTING.md §11): one virtual thread. `catch(...) { transferOldNNOutputs; delete
  /// stbuf; throw; }` is the error path below; on error this thread also sets shouldStopNow so the
  /// other virtual threads end their loops and the join completes (eval futures must not be dropped).
  async fn search_loop(&self, thread_idx: i32, c: &SearchLoopCaptures<'_>) -> Result<(), KataError> {
    let mut stbuf: Box<SearchThread> = SearchThread::new_boxed(thread_idx, self);

    let mut num_playouts: i64 = c.num_playouts_shared.get() /*relaxed*/;
    let result: Result<(), KataError> = async {
      let last_time_used_recomputing_tc_limit: f64 = 0.0;
      loop {
        let mut time_used: f64 = 0.0;
        if c.has_tc || c.has_max_time {
          time_used = c.timer.get_seconds();
        }

        let mut tc_max_time_limit: f64 = 0.0;
        if c.has_tc {
          tc_max_time_limit = c.tc_max_time.get() /*acquire*/;
        }

        let mut should_stop: bool = (num_playouts >= c.max_playouts) || (num_playouts + c.num_non_playout_visits >= c.max_visits);

        //Time limits cannot stop us from doing at least a little search so we have a non-null tree
        if c.has_max_time && num_playouts >= 2 && time_used >= c.max_time {
          should_stop = true;
        }
        if c.has_tc && num_playouts >= 2 && time_used >= tc_max_time_limit {
          should_stop = true;
        }
        if let Some(should_stop_early) = c.should_stop_early {
          if should_stop_early() {
            should_stop = true;
          }
        }

        //But an explicit stop signal can stop us from doing any search
        if should_stop || c.should_stop_now.get() /*relaxed*/ {
          c.should_stop_now.set(true) /*relaxed*/;
          break;
        }

        //Thread 0 alone is responsible for recomputing time limits every once in a while
        //Cap of 10 times per second.
        if !c.pondering && (c.has_tc || c.has_max_time) && thread_idx == 0 && time_used >= last_time_used_recomputing_tc_limit + 0.1 {
          let root_visits: i64 = num_playouts + c.num_non_playout_visits;
          let mut tc_limit: f64 = 1e30;
          if c.has_tc {
            tc_limit = self.recompute_search_time_limit(c.tc, time_used, c.search_factor, root_visits)?;
            c.tc_max_time.set(tc_limit) /*release*/;
          }
          let upper_bound_visits: f64 = self.compute_upper_bound_visits_left_due_to_time(root_visits, time_used, cpp::cpp_min(tc_limit, c.max_time));
          c.upper_bound_visits_left_due_to_time.set(upper_bound_visits) /*release*/;
        }

        let mut upper_bound_visits_left: f64 = 1e30;
        if c.has_tc {
          upper_bound_visits_left = c.upper_bound_visits_left_due_to_time.get() /*acquire*/;
        }
        upper_bound_visits_left = cpp::cpp_min(upper_bound_visits_left, c.max_playouts as f64 - num_playouts as f64);
        upper_bound_visits_left = cpp::cpp_min(upper_bound_visits_left, c.max_visits as f64 - num_playouts as f64 - c.num_non_playout_visits as f64);

        let finished_playout: bool = self.run_single_playout(&mut stbuf, upper_bound_visits_left).await?;
        if finished_playout {
          num_playouts = cpp::cell_fetch_add(c.num_playouts_shared, 1i64) /*relaxed*/;
          num_playouts += 1;
        } else {
          //In the case that we didn't finish a playout, give other threads a chance to run before we try again
          //so that it's more likely we become unstuck.
          yield_now().await;
        }
      }
      Ok(())
    }
    .await;

    if result.is_err() {
      c.should_stop_now.set(true);
    }
    self.transfer_old_nn_outputs(&mut stbuf);
    drop(stbuf);
    result
  }

  /// upstream: Search::beginSearch (search.cpp)
  //If we're being asked to search from a position where the game is over, this is fine. Just keep going, the boardhistory
  //should reasonably tolerate just continuing. We do NOT want to clear history because we could inadvertently make a move
  //that an external ruleset COULD think violated superko.
  /// PORT NOTE: async (computeRootValues awaits a root NN evaluation); throws -> Result.
  pub async fn begin_search(&mut self, pondering: bool) -> Result<(), KataError> {
    if self.root_board.x_size > self.nn_x_len || self.root_board.y_size > self.nn_y_len {
      return Err(KataError::StringError(
        "Search got from NNEval nnXLen = ".to_string()
          + &Global::int_to_string(self.nn_x_len)
          + " nnYLen = "
          + &Global::int_to_string(self.nn_y_len)
          + " but was asked to search board with larger x or y size",
      ));
    }

    self.root_board.check_consistency()?;

    self.num_searches_begun += 1;

    //Avoid any issues in principle from rolling over
    if self.search_node_age.get() > 0x3FFFFFFF {
      self.clear_search();
    }

    if !pondering {
      self.pla_that_search_is_for = self.root_pla;
    }
    //If we begin the game with a ponder, then assume that "we" are the opposing side until we see otherwise.
    if self.pla_that_search_is_for == C_EMPTY {
      self.pla_that_search_is_for = get_opp(self.root_pla);
    }

    if self.pla_that_search_is_for_last_search != self.pla_that_search_is_for {
      //In the case we are doing playoutDoublingAdvantage without a specific player (so, doing the root player)
      //and the player that the search is for changes, we need to clear the tree since we need new evals for the new way around
      if self.search_params.playout_doubling_advantage != 0.0 && self.search_params.playout_doubling_advantage_pla == C_EMPTY {
        self.clear_search();
      }
      //If we are doing pattern bonus and the player the search is for changes, clear the search. Recomputing the search tree
      //recursively *would* fix all our utilities, but the problem is the playout distribution will still be matching the
      //old probabilities without a lot of new search, so clearing ensures a better distribution.
      if self.search_params.avoid_repeated_pattern_utility != 0.0 || self.external_pattern_bonus_table.is_some() {
        self.clear_search();
      }
      //If we have a human SL net and the parameters are different for the different sides, clear the search.
      if self.human_evaluator.is_some() {
        if (self.search_params.human_sl_pla_explore_prob_weightless != self.search_params.human_sl_opp_explore_prob_weightless)
          || (self.search_params.human_sl_pla_explore_prob_weightful != self.search_params.human_sl_opp_explore_prob_weightful)
          || (self.search_params.human_sl_pla_explore_prob_weightless != self.search_params.human_sl_root_explore_prob_weightless)
          || (self.search_params.human_sl_pla_explore_prob_weightful != self.search_params.human_sl_root_explore_prob_weightful)
        {
          self.clear_search();
        }
      }
    }
    self.pla_that_search_is_for_last_search = self.pla_that_search_is_for;
    //cout << "BEGINSEARCH " << PlayerIO::playerToString(rootPla) << " " << PlayerIO::playerToString(plaThatSearchIsFor) << endl;

    self.clear_old_nn_outputs();
    self.compute_root_values().await?;

    //Prepare value bias table if we need it
    if self.search_params.subtree_value_bias_factor != 0.0
      && self.subtree_value_bias_table.is_none()
      && !(self.search_params.anti_mirror && self.mirroring_pla != C_EMPTY)
    {
      self.subtree_value_bias_table = Some(Box::new(SubtreeValueBiasTable::new(self.search_params.subtree_value_bias_table_num_shards)));
    }

    //Prepare eval cache if we need it
    if self.search_params.use_eval_cache && self.search_params.use_graph_search && self.eval_cache.is_none() && self.mirroring_pla == C_EMPTY {
      // TODO(port): deferred (evalcache.cpp): `evalCache = std::make_shared<EvalCacheTable>(searchParams.subtreeValueBiasTableNumShards);`
      // The eval cache is not ported, so the search runs as with useEvalCache=false.
      log::warn!("useEvalCache is not supported by this port yet (evalcache.cpp deferred); searching without the eval cache");
    }

    //Refresh pattern bonuses if needed
    if self.pattern_bonus_table.is_some() {
      self.pattern_bonus_table = None;
    }
    if self.search_params.avoid_repeated_pattern_utility != 0.0 || self.external_pattern_bonus_table.is_some() {
      // TODO(port): deferred (patternbonustable.cpp): `patternBonusTable = new PatternBonusTable(...)`,
      // `addBonusForGameMoves(rootHistory,bonus,plaThatSearchIsFor)` and clearing the root's
      // patternBonusHash. Pattern bonuses are not ported, so the search runs as with
      // avoidRepeatedPatternUtility=0.
      log::warn!("avoidRepeatedPatternUtility / pattern bonus tables are not supported by this port yet (patternbonustable.cpp deferred)");
      //Clear any pattern bonus on the root node itself
      if let Some(root_node_idx) = self.root_node {
        self.node_arena.get(root_node_idx).pattern_bonus_hash.set(Hash128::default());
      }
    }

    if self.search_params.root_symmetry_pruning {
      let avoid_move_until_by_loc: &Vec<i32> =
        if self.root_pla == P_BLACK { &self.avoid_move_until_by_loc_black } else { &self.avoid_move_until_by_loc_white };
      if !self.root_prune_only_symmetries.is_empty() {
        SymmetryHelpers::mark_duplicate_move_locs(
          &self.root_board,
          &self.root_history,
          Some(&self.root_prune_only_symmetries),
          avoid_move_until_by_loc,
          &mut self.root_sym_dup_loc,
          &mut self.root_symmetries,
        );
      } else {
        SymmetryHelpers::mark_duplicate_move_locs(
          &self.root_board,
          &self.root_history,
          None,
          avoid_move_until_by_loc,
          &mut self.root_sym_dup_loc,
          &mut self.root_symmetries,
        );
      }
    } else {
      //Just in case, don't leave the values undefined.
      self.root_sym_dup_loc.fill(false);
      self.root_symmetries.clear();
      self.root_symmetries.push(0);
    }

    let mut dummy_thread: Box<SearchThread> = SearchThread::new_boxed(-1, self);

    //If we're using graph search, we recompute the graph hash from scratch at the start of search.
    if self.search_params.use_graph_search {
      self.root_graph_hash = GraphHash::get_graph_hash_from_scratch(
        &self.root_history,
        self.root_pla,
        self.search_params.graph_search_rep_bound,
        self.search_params.draw_equivalent_wins_for_white,
      );
    } else {
      self.root_graph_hash = Hash128::default();
    }

    if self.root_node.is_none() {
      //Avoid storing the root node in the nodeTable, guarantee that it never is part of a cycle, allocate it directly.
      //Also force that it is non-terminal.
      let force_non_terminal: bool = self.root_history.is_game_finished; // Make sure the root isn't considered terminal if game would be finished.
      let mutex_idx: u32 = self.create_mutex_idx_for_node(&mut dummy_thread);
      let root_pla: Player = self.root_pla;
      let root_graph_hash: Hash128 = self.root_graph_hash;
      self.root_node = Some(self.node_arena.alloc(|idx| SearchNode::new(idx, root_pla, force_non_terminal, mutex_idx, root_graph_hash)));
      if self.search_params.use_eval_cache && self.search_params.use_graph_search && self.eval_cache.is_some() && self.mirroring_pla == C_EMPTY {
        // TODO(port): deferred (evalcache.cpp): `rootNode->evalCacheEntry = evalCache->find(rootNode->graphHash);`
        match **self.eval_cache.as_ref().expect("checked is_some") {}
      }
    } else {
      //If the root node has any existing children, then prune things down if there are moves that should not be allowed at the root.
      let root_node_idx: NodeIdx = self.root_node.expect("checked above");
      let mut any_filtered: bool = false;
      let mut num_good_children: i32 = 0;
      let children_capacity_before: i32;
      {
        let node: Rc<SearchNode> = self.node_arena.get(root_node_idx);
        let children: SearchNodeChildrenReference = node.get_children();
        let children_capacity: i32 = children.get_capacity();
        children_capacity_before = children_capacity;
        if children_capacity > 0 {
          //This filtering, by deleting children, doesn't conform to the normal invariants that hold during search.
          //However nothing else should be running at this time and the search hasn't actually started yet, so this is okay.
          //Also we can't be affecting the tree since the root node isn't in the table and can't be transposed to.
          let mut filtered_nodes: Vec<NodeIdx> = Vec::new();
          {
            let mut i: i32 = 0;
            while i < children_capacity {
              let child: Option<NodeIdx> = children.index(i).get_if_allocated();
              let edge_visits: i64 = children.index(i).get_edge_visits();
              let move_loc: Loc = children.index(i).get_move_loc();
              let child: NodeIdx = match child {
                None => break,
                Some(c) => c,
              };
              //Remove the child from its current spot
              children.index(i).store(None);
              children.index(i).set_edge_visits(0);
              children.index(i).set_move_loc(Board::NULL_LOC);
              //Maybe add it back. Specifically check for legality just in case weird graph interaction in the
              //tree gives wrong legality - ensure that once we are the root, we are strict on legality.
              if self.root_history.is_legal(&self.root_board, move_loc, self.root_pla) && self.is_allowed_root_move(move_loc) {
                children.index(num_good_children).store(Some(child));
                children.index(num_good_children).set_edge_visits(edge_visits);
                children.index(num_good_children).set_move_loc(move_loc);
                num_good_children += 1;
              } else {
                any_filtered = true;
                filtered_nodes.push(child);
              }
              i += 1;
            }
            while i < children_capacity {
              let child: Option<NodeIdx> = children.index(i).get_if_allocated();
              testAssert!(child.is_none());
              i += 1;
            }
          }
          let _ = filtered_nodes;
        }
      }

      if children_capacity_before > 0 && any_filtered {
        //Fix up the node state and child arrays.
        // PORT NOTE: collapseChildrenCapacity is "not thread-safe": it runs through NodeArena::with_mut,
        // which needs the root's Rc to be unique, so the borrows above end first (PORTING.md §4).
        self.node_arena.with_mut(root_node_idx, |n| n.collapse_children_capacity(num_good_children));
        let node: Rc<SearchNode> = self.node_arena.get(root_node_idx);
        let children: SearchNodeChildrenReference = node.get_children();
        let children_capacity: i32 = children.get_capacity();

        //Fix up the number of visits of the root node after doing this filtering
        let mut new_num_visits: i64 = 0;
        for i in 0..children_capacity {
          let child: Option<NodeIdx> = children.index(i).get_if_allocated();
          if child.is_none() {
            break;
          }
          let edge_visits: i64 = children.index(i).get_edge_visits();
          new_num_visits += edge_visits;
        }

        //For the node's own visit itself
        new_num_visits += 1;

        //Set the visits in place
        // PORT NOTE: statsLock spin lock removed (single-threaded).
        node.stats.visits.set(new_num_visits) /*release*/;

        //Update all other stats
        self.recompute_node_stats(&node, &mut dummy_thread, 0, true);
      }

      //Recursively update all stats in the tree if we have dynamic score values
      //And also to clear out lastResponseBiasDeltaSum and lastResponseBiasWeight
      if self.search_params.dynamic_score_utility_factor != 0.0 || self.search_params.subtree_value_bias_factor != 0.0 || self.pattern_bonus_table.is_some() {
        let node: Rc<SearchNode> = self.node_arena.get(root_node_idx);
        self.recursively_recompute_stats(&node);
        if any_filtered {
          //Recursive stats recomputation resulted in us marking all nodes we have. Anything filtered is old now, delete it.
          let old: bool = true;
          self.delete_all_old_or_all_new_table_nodes_and_subtree_value_bias_mulithreaded(old);
        }
      } else {
        if any_filtered {
          //Sweep over the entire child marking it as good (calling NULL function), and then delete anything unmarked.
          self.apply_recursively_any_order_mulithreaded(&[root_node_idx], None);
          let old: bool = true;
          self.delete_all_old_or_all_new_table_nodes_and_subtree_value_bias_mulithreaded(old);
        }
      }
    }

    //Clear unused stuff in value bias table since we may have pruned rootNode stuff
    if self.search_params.subtree_value_bias_factor != 0.0 {
      if let Some(subtree_value_bias_table) = self.subtree_value_bias_table.as_mut() {
        subtree_value_bias_table.clear_unused_synchronous();
      }
    }

    //Mark all nodes old for the purposes of updating old nnoutputs
    self.search_node_age.set(self.search_node_age.get() + 1);
    Ok(())
  }

  /// upstream: Search::createMutexIdxForNode (search.cpp)
  /// PORT NOTE (D12): the mutex pool is gone but the RNG draw that picked a mutex is kept.
  pub(crate) fn create_mutex_idx_for_node(&self, thread: &mut SearchThread) -> u32 {
    thread.rand.next_uint() & (self.mutex_pool_num_mutexes - 1)
  }

  /// upstream: Search::allocateOrFindNode (search.cpp)
  //Must be called AFTER making the bestChildMoveLoc in the thread board and hist.
  pub(crate) fn allocate_or_find_node(
    &self,
    thread: &mut SearchThread,
    next_pla: Player,
    best_child_move_loc: Loc,
    force_non_terminal: bool,
    graph_hash: Hash128,
  ) -> Rc<SearchNode> {
    //Hash to use as a unique id for this node in the table, for transposition detection.
    //If this collides, we will be sad, but it should be astronomically rare since our hash is 128 bits.
    let mut child_hash: Hash128;
    if self.search_params.use_graph_search {
      child_hash = graph_hash;
      if force_non_terminal {
        child_hash ^= FORCE_NON_TERMINAL_HASH;
      }
    } else {
      // PORT NOTE: GCC arg order (right to left): the second constructor argument is drawn first.
      let h1: u64 = thread.rand.next_uint64();
      let h0: u64 = thread.rand.next_uint64();
      child_hash = thread.board.pos_hash ^ Hash128::new(h0, h1);
    }

    let node_table_idx: u32 = self.node_table.get_index(child_hash.hash0);
    // PORT NOTE: the shard mutex lock is removed (single-threaded); the RefCell borrow below spans its scope.

    let child: Rc<SearchNode>;
    let mut entries = self.node_table.entries.borrow_mut();
    let node_map = &mut entries[node_table_idx as usize];

    loop {
      // PORT NOTE: `lower_bound` + key equality is a map lookup; `insert(insertLoc, ...)` with the hint is an insert.
      let existing: Option<NodeIdx> = node_map.get(&child_hash).copied();

      if let Some(existing) = existing {
        let existing_node: Rc<SearchNode> = self.node_arena.get(existing);
        //Attempt to transpose to invalid node - rerandomize hash and just store this node somewhere arbitrary.
        if existing_node.next_pla != next_pla {
          // PORT NOTE: GCC arg order (right to left).
          let h1: u64 = thread.rand.next_uint64();
          let h0: u64 = thread.rand.next_uint64();
          child_hash = thread.board.pos_hash ^ Hash128::new(h0, h1);
          continue;
        }
        child = existing_node;
      } else {
        let mutex_idx: u32 = self.create_mutex_idx_for_node(thread);
        let new_idx: NodeIdx = self.node_arena.alloc(|idx| SearchNode::new(idx, next_pla, force_non_terminal, mutex_idx, graph_hash));
        let new_child: Rc<SearchNode> = self.node_arena.get(new_idx);

        //Also perform subtree value bias and pattern bonus handling under the mutex. These parameters are no atomic, so
        //if the node is accessed concurrently by other nodes through the table, we need to make sure these parameters are fully
        //fully-formed before we make the node accessible to anyone.

        if self.search_params.subtree_value_bias_factor != 0.0 {
          if let Some(subtree_value_bias_table) = &self.subtree_value_bias_table {
            //TODO can we make subtree value bias not depend on prev move loc?
            if thread.history.move_history.len() >= 2 {
              let prev_move_loc: Loc = thread.history.move_history[thread.history.move_history.len() - 2].loc;
              //Avoid subtree value bias application when bestChildMoveLoc is a pass
              if prev_move_loc != Board::NULL_LOC && best_child_move_loc != Board::PASS_LOC {
                let entry: Rc<SubtreeValueBiasEntry> =
                  subtree_value_bias_table.get(get_opp(thread.pla), prev_move_loc, best_child_move_loc, thread.history.get_recent_board(1));
                *new_child.subtree_value_bias_table_entry.borrow_mut() = Some(entry);
              }
            }
          }
        }

        if self.search_params.use_eval_cache && self.search_params.use_graph_search && self.eval_cache.is_some() && self.mirroring_pla == C_EMPTY {
          // TODO(port): deferred (evalcache.cpp): `child->evalCacheEntry = evalCache->find(child->graphHash);`
          match **self.eval_cache.as_ref().expect("checked is_some") {}
        }

        if let Some(pattern_bonus_table) = &self.pattern_bonus_table {
          // TODO(port): deferred (patternbonustable.cpp):
          // `child->patternBonusHash = patternBonusTable->getHash(getOpp(thread.pla), bestChildMoveLoc, thread.history.getRecentBoard(1));`
          match **pattern_bonus_table {}
        }

        //Insert into map! Use insertLoc as hint.
        node_map.insert(child_hash, new_idx);
        child = new_child;
      }
      break;
    }
    child
  }

  /// upstream: Search::clearOldNNOutputs (search.cpp)
  pub(crate) fn clear_old_nn_outputs(&self) {
    let mut old = self.old_nn_outputs_to_clean_up.borrow_mut();
    old.clear();
  }

  /// upstream: Search::transferOldNNOutputs (search.cpp)
  pub(crate) fn transfer_old_nn_outputs(&self, thread: &mut SearchThread) {
    // PORT NOTE: oldNNOutputsToCleanUpMutex lock removed (single-threaded).
    let mut old = self.old_nn_outputs_to_clean_up.borrow_mut();
    for i in 0..thread.old_nn_outputs_to_clean_up.len() {
      old.push(thread.old_nn_outputs_to_clean_up[i].clone());
    }
    thread.old_nn_outputs_to_clean_up.clear();
  }

  /// upstream: Search::removeSubtreeValueBias (search.cpp)
  pub(crate) fn remove_subtree_value_bias(&self, node: &SearchNode) {
    let entry_opt: Option<Rc<SubtreeValueBiasEntry>> = node.subtree_value_bias_table_entry.borrow().clone();
    if let Some(entry) = entry_opt {
      let delta_utility_sum_to_subtract: f64 = node.last_subtree_value_bias_delta_sum.get() * self.search_params.subtree_value_bias_free_prop;
      let weight_sum_to_subtract: f64 = node.last_subtree_value_bias_weight.get() * self.search_params.subtree_value_bias_free_prop;

      // PORT NOTE: entryLock spin lock removed (single-threaded).
      entry.delta_utility_sum.set(entry.delta_utility_sum.get() - delta_utility_sum_to_subtract);
      entry.weight_sum.set(entry.weight_sum.get() - weight_sum_to_subtract);
      drop(entry);
      *node.subtree_value_bias_table_entry.borrow_mut() = None;
    }
  }

  /// upstream: Search::deleteAllOldOrAllNewTableNodesAndSubtreeValueBiasMulithreaded (search.cpp)
  //Delete ALL nodes where nodeAge < searchNodeAge if old is true, else all nodes where nodeAge >= searchNodeAge
  //Also clears subtreevaluebias for deleted nodes.
  pub(crate) fn delete_all_old_or_all_new_table_nodes_and_subtree_value_bias_mulithreaded(&self, old: bool) {
    let num_additional_threads: i32 = self.num_additional_threads_to_use_for_tasks();
    testAssert!(num_additional_threads >= 0);
    let mut g = |thread_idx: i32| {
      let entries_size: u64 = self.node_table.entries.borrow().len() as u64;
      let idx0: usize = (thread_idx as u64 * entries_size / (num_additional_threads + 1) as u64) as usize;
      let idx1: usize = ((thread_idx + 1) as u64 * entries_size / (num_additional_threads + 1) as u64) as usize;
      for i in idx0..idx1 {
        let mut entries = self.node_table.entries.borrow_mut();
        let node_map = &mut entries[i];
        // PORT NOTE: the `it = nodeMap.erase(it)` loop is retain: same visit order (map order) and
        // the same side effects per erased node (PORTING.md §3).
        node_map.retain(|_, node_idx| {
          let node: Rc<SearchNode> = self.node_arena.get(*node_idx);
          if old == (node.node_age.get() /*acquire*/ < self.search_node_age.get()) {
            self.remove_subtree_value_bias(&node);
            drop(node);
            self.node_arena.free(*node_idx);
            false
          } else {
            true
          }
        });
      }
    };
    self.perform_task_with_threads(&mut g, 0x3FFFffff);
  }

  /// upstream: Search::deleteAllTableNodesMulithreaded (search.cpp)
  //Delete ALL nodes. More efficient than deleteAllOldOrAllNewTableNodesAndSubtreeValueBiasMulithreaded if deleting everything.
  //Doesn't clear subtree value bias.
  pub(crate) fn delete_all_table_nodes_mulithreaded(&self) {
    let num_additional_threads: i32 = self.num_additional_threads_to_use_for_tasks();
    testAssert!(num_additional_threads >= 0);
    let mut g = |thread_idx: i32| {
      let entries_size: u64 = self.node_table.entries.borrow().len() as u64;
      let idx0: usize = (thread_idx as u64 * entries_size / (num_additional_threads + 1) as u64) as usize;
      let idx1: usize = ((thread_idx + 1) as u64 * entries_size / (num_additional_threads + 1) as u64) as usize;
      for i in idx0..idx1 {
        let mut entries = self.node_table.entries.borrow_mut();
        let node_map = &mut entries[i];
        for node_idx in node_map.values() {
          self.node_arena.free(*node_idx);
        }
        node_map.clear();
      }
    };
    self.perform_task_with_threads(&mut g, 0x3FFFffff);
  }

  /// upstream: Search::recursivelyRecomputeStats (search.cpp)
  //This function should NOT ever be called concurrently with any other threads modifying the search tree.
  //However, it does thread-safely modify things itself, so can safely in theory run concurrently with things
  //like ownership computation or analysis that simply read the tree.
  pub(crate) fn recursively_recompute_stats(&self, n: &SearchNode) {
    let num_additional_threads: i32 = self.num_additional_threads_to_use_for_tasks();
    let mut dummy_threads: Vec<Box<SearchThread>> = Vec::with_capacity((num_additional_threads + 1) as usize);
    for thread_idx in 0..num_additional_threads + 1 {
      dummy_threads.push(SearchThread::new_boxed(thread_idx, self));
    }

    let mut f = |node: &SearchNode, thread_idx: i32| {
      debug_assert!(thread_idx >= 0 && (thread_idx as usize) < dummy_threads.len());
      let thread: &mut SearchThread = &mut dummy_threads[thread_idx as usize];

      let mut found_any_children: bool = false;
      let children: SearchNodeChildrenReference = node.get_children();
      let children_capacity: i32 = children.get_capacity();
      let mut i: i32 = 0;
      while i < children_capacity {
        let child: Option<NodeIdx> = children.index(i).get_if_allocated();
        if child.is_none() {
          break;
        }
        found_any_children = true;
        i += 1;
      }
      while i < children_capacity {
        let child: Option<NodeIdx> = children.index(i).get_if_allocated();
        debug_assert!(child.is_none());
        let _ = child;
        i += 1;
      }

      //If this node has children, it MUST also have an nnOutput.
      if found_any_children {
        let nn_output: Option<Rc<NNOutput>> = node.get_nn_output();
        debug_assert!(nn_output.is_some());
        let _ = nn_output; //avoid warning when we have no asserts
      }

      //Also, something is wrong if we have virtual losses at this point
      let num_virtual_losses: i32 = node.virtual_losses.get() /*acquire*/;
      debug_assert!(num_virtual_losses == 0);
      let _ = num_virtual_losses;

      let is_root: bool = Some(node.idx) == self.root_node;

      //If the node has no children, then just update its utility directly
      //Again, this would be a little wrong if this function were running concurrently with anything else in the
      //case that new children were added in the meantime. Although maybe it would be okay.
      if !found_any_children {
        let num_visits: i64 = node.stats.visits.get() /*acquire*/;
        let weight_sum: f64 = node.stats.weight_sum.get() /*acquire*/;
        let win_loss_value_avg: f64 = node.stats.win_loss_value_avg.get() /*acquire*/;
        let no_result_value_avg: f64 = node.stats.no_result_value_avg.get() /*acquire*/;
        let score_mean_avg: f64 = node.stats.score_mean_avg.get() /*acquire*/;
        let score_mean_sq_avg: f64 = node.stats.score_mean_sq_avg.get() /*acquire*/;

        //It's possible that this node has 0 weight in the case where it's the root node
        //and has 0 visits because we began a search and then stopped it before any playouts happened.
        //In that case, there's not much to recompute.
        if weight_sum <= 0.0 {
          testAssert!(num_visits == 0);
          testAssert!(is_root);
        } else {
          let result_utility: f64 = self.get_result_utility(win_loss_value_avg, no_result_value_avg);
          let score_utility: f64 = self.get_score_utility(score_mean_avg, score_mean_sq_avg);
          let mut new_utility_avg: f64 = result_utility + score_utility;
          new_utility_avg += self.get_pattern_bonus(node.pattern_bonus_hash.get(), get_opp(node.next_pla));
          let new_utility_sq_avg: f64 = new_utility_avg * new_utility_avg;

          // PORT NOTE: statsLock spin lock removed (single-threaded).
          node.stats.utility_avg.set(new_utility_avg) /*release*/;
          node.stats.utility_sq_avg.set(new_utility_sq_avg) /*release*/;
        }
      } else {
        //Otherwise recompute it using the usual method
        self.recompute_node_stats(node, thread, 0, is_root);
      }
    };

    let nodes: Vec<NodeIdx> = vec![n.idx];
    self.apply_recursively_post_order_mulithreaded(&nodes, Some(&mut f));

    // PORT NOTE: `delete dummyThreads[threadIdx]` is the drop of the Vec<Box<SearchThread>>.
  }

  /// upstream: Search::recursivelyRecordEvalCache (search.cpp)
  /// TODO(port): deferred (evalcache.cpp): EvalCacheTable has no values, so this is only reached
  /// with `evalCache != nullptr`, which cannot happen yet.
  pub(crate) fn recursively_record_eval_cache(&self, n: &SearchNode) {
    let mut f = |node: &SearchNode, thread_idx: i32| {
      let _ = thread_idx;
      let num_visits: i64 = node.stats.visits.get() /*acquire*/;
      if num_visits >= self.search_params.eval_cache_min_visits && !node.force_non_terminal {
        let is_root_node: bool = Some(node.idx) == self.root_node;
        let _ = is_root_node;
        // TODO(port): deferred (evalcache.cpp): `evalCache->update(node->graphHash, node, searchParams.evalCacheMinVisits, isRootNode);`
        match **self.eval_cache.as_ref().expect("recursivelyRecordEvalCache needs an evalCache") {}
      }
    };
    let nodes: Vec<NodeIdx> = vec![n.idx];
    self.apply_recursively_post_order_mulithreaded(&nodes, Some(&mut f));
  }

  /// upstream: Search::computeRootValues (search.cpp)
  // Helper for begin search
  /// PORT NOTE: async (awaits computeRootNNEvaluation); throws -> Result.
  pub(crate) async fn compute_root_values(&mut self) -> Result<(), KataError> {
    //rootSafeArea is strictly pass-alive groups and strictly safe territory.
    let non_pass_alive_stones: bool = false;
    let safe_big_territories: bool = false;
    let unsafe_big_territories: bool = false;
    let is_multi_stone_suicide_legal: bool = self.root_history.rules.multi_stone_suicide_legal;
    self.root_board.calculate_area(
      &mut self.root_safe_area[..],
      non_pass_alive_stones,
      safe_big_territories,
      unsafe_big_territories,
      is_multi_stone_suicide_legal,
    );

    //Figure out how to set recentScoreCenter
    {
      let mut found_expected_score_from_tree: bool = false;
      let mut expected_score: f64 = 0.0;
      if let Some(root_node_idx) = self.root_node {
        let node: Rc<SearchNode> = self.node_arena.get(root_node_idx);
        let num_visits: i64 = node.stats.visits.get() /*acquire*/;
        let weight_sum: f64 = node.stats.weight_sum.get() /*acquire*/;
        let score_mean_avg: f64 = node.stats.score_mean_avg.get() /*acquire*/;
        if num_visits > 0 && weight_sum > 0.0 {
          found_expected_score_from_tree = true;
          expected_score = score_mean_avg;
        }
      }

      //Grab a neural net evaluation for the current position and use that as the center
      if !found_expected_score_from_tree {
        let nn_result_buf: Rc<NNResultBuf> = Rc::new(NNResultBuf::new());
        let include_owner_map: bool = true;
        self.compute_root_nn_evaluation(&nn_result_buf, include_owner_map).await?;
        expected_score = nn_result_buf.result.borrow().as_ref().expect("evaluate leaves a result in the buf").white_score_mean as f64;
      }

      self.recent_score_center = expected_score * (1.0 - self.search_params.dynamic_score_center_zero_weight);
      let cap: f64 = ((self.root_board.x_size * self.root_board.y_size) as f64).sqrt() * self.search_params.dynamic_score_center_scale;
      if self.recent_score_center > expected_score + cap {
        self.recent_score_center = expected_score + cap;
      }
      if self.recent_score_center < expected_score - cap {
        self.recent_score_center = expected_score - cap;
      }
    }

    let opponent_was_mirroring_pla: Player = self.mirroring_pla;
    //Update mirroringPla, mirrorAdvantage, mirrorCenterSymmetryError
    // TODO(port): deferred (searchmirror.cpp): `updateMirroring();`. Only its antiMirror=false effect
    // (the first three assignments) is reproduced; with antiMirror the opponent is never detected as
    // mirroring, so mirroringPla stays C_EMPTY and every anti-mirror path stays off.
    self.mirroring_pla = C_EMPTY;
    self.mirror_advantage = 0.0;
    self.mirror_center_symmetry_error = 1e10;
    if self.search_params.anti_mirror {
      log::warn!("antiMirror is not supported by this port yet (searchmirror.cpp deferred); ignoring it");
    }

    //Clear search if opponent mirror status changed, so that our tree adjusts appropriately
    if opponent_was_mirroring_pla != self.mirroring_pla {
      self.clear_search();
      self.subtree_value_bias_table = None;
    }
    Ok(())
  }

  /// upstream: Search::runSinglePlayout (search.cpp)
  /// PORT NOTE: async (PORTING.md §11); throws -> Result.
  pub async fn run_single_playout(&self, thread: &mut SearchThread, upper_bound_visits_left: f64) -> Result<bool, KataError> {
    //Store this value, used for futile-visit pruning this thread's root children selections.
    thread.upper_bound_visits_left = upper_bound_visits_left;

    //Prep this value, playoutDescend will set it to false if the playout shouldn't count
    thread.should_count_playout = true;

    let root_node: Rc<SearchNode> = self.node_arena.get(self.root_node.expect("runSinglePlayout needs a root node (beginSearch)"));
    let finished_playout: bool = self.playout_descend(thread, &root_node, true).await?;
    let _ = finished_playout;

    //Restore thread state back to the root state
    thread.pla = self.root_pla;
    thread.board = self.root_board.clone();
    thread.history = self.root_history.clone();
    thread.graph_hash = self.root_graph_hash;
    thread.graph_path.clear();

    Ok(thread.should_count_playout)
  }

  /// upstream: Search::playoutDescend (search.cpp)
  /// PORT NOTE: async and recursive, so it returns a boxed future (PORTING.md §11).
  /// PERF(port): one allocation per depth.
  pub(crate) fn playout_descend<'a>(
    &'a self,
    thread: &'a mut SearchThread,
    node: &'a SearchNode,
    is_root: bool,
  ) -> LocalBoxFuture<'a, Result<bool, KataError>> {
    Box::pin(async move {
      //Hit terminal node, finish
      //forceNonTerminal marks special nodes where we cannot end the game, and is set IF they would normally be finished.
      //This includes the root if the root would be game-ended, since if we are searching a position
      //we presumably want to actually explore deeper and get a result. Also it includes the node following a pass from the root in
      //the case where we are conservativePass and the game would be ended. For friendlyPassOk rules, it may include deeper nodes.
      //Note that we also carefully clear the search when a pass from the root would be terminal, so nodes should never need to switch
      //status after tree reuse in the latter case.
      if thread.history.is_game_finished && !node.force_non_terminal {
        //Avoid running "too fast", by making sure that a leaf evaluation takes roughly the same time as a genuine nn eval
        //This stops a thread from building a silly number of visits to distort MCTS statistics while other threads are stuck on the GPU.
        self.nn_evaluator.wait_for_next_nn_eval_if_any().await;
        if thread.history.is_no_result {
          let win_loss_value: f64 = 0.0;
          let no_result_value: f64 = 1.0;
          let score_mean: f64 = 0.0;
          let score_mean_sq: f64 = 0.0;
          let lead: f64 = 0.0;
          let weight: f64 =
            if self.search_params.use_uncertainty && self.nn_evaluator.supports_shortterm_error() { self.search_params.uncertainty_max_weight } else { 1.0 };
          self.add_leaf_value(node, win_loss_value, no_result_value, score_mean, score_mean_sq, lead, weight, true, false);
          return Ok(true);
        } else {
          let win_loss_value: f64 = 2.0 * ScoreValue::white_wins_of_winner(thread.history.winner, self.search_params.draw_equivalent_wins_for_white) - 1.0;
          let no_result_value: f64 = 0.0;
          let score_mean: f64 = ScoreValue::white_score_draw_adjust(
            thread.history.final_white_minus_black_score as f64,
            self.search_params.draw_equivalent_wins_for_white,
            &thread.history,
          );
          let score_mean_sq: f64 =
            ScoreValue::white_score_mean_sq_of_score_gridded(thread.history.final_white_minus_black_score as f64, self.search_params.draw_equivalent_wins_for_white);
          let lead: f64 = score_mean;
          let weight: f64 =
            if self.search_params.use_uncertainty && self.nn_evaluator.supports_shortterm_error() { self.search_params.uncertainty_max_weight } else { 1.0 };
          self.add_leaf_value(node, win_loss_value, no_result_value, score_mean, score_mean_sq, lead, weight, true, false);
          return Ok(true);
        }
      }

      let mut node_state: SearchNodeState = node.state.get() /*acquire*/;
      if node_state == SearchNode::STATE_UNEVALUATED {
        //Always attempt to set a new nnOutput. That way, if some GPU is slow and malfunctioning, we don't get blocked by it.
        {
          let suc: bool = self.init_node_nn_output(thread, node, is_root, false, false).await?;
          //Leave the node as unevaluated - only the thread that first actually set the nnOutput into the node
          //gets to update the state, to avoid races where we update the state while the node stats aren't updated yet.
          if !suc {
            thread.should_count_playout = false;
            return Ok(false);
          }
        }

        let suc: bool = cpp::cell_cas(&node.state, &mut node_state, SearchNode::STATE_EVALUATING) /*seq_cst*/;
        if !suc {
          //Presumably someone else got there first.
          //Just give up on this playout and try again from the start.
          thread.should_count_playout = false;
          return Ok(false);
        } else {
          //Perform the nn evaluation and finish!
          node.initialize_children();
          node.state.set(SearchNode::STATE_EXPANDED0) /*seq_cst*/;
          return Ok(true);
        }
      } else if node_state == SearchNode::STATE_EVALUATING {
        //Just give up on this playout and try again from the start.
        thread.should_count_playout = false;
        return Ok(false);
      }

      debug_assert!(node_state >= SearchNode::STATE_EXPANDED0);
      self.maybe_recompute_existing_nn_output(thread, node, is_root).await?;

      //Find the best child to descend down
      let mut num_children_found: i32 = 0;
      let mut best_child_idx: i32 = 0;
      let mut best_child_move_loc: Loc = Board::NULL_LOC;
      let mut count_edge_visit: bool = false;

      let child: Rc<SearchNode>;
      loop {
        self.select_best_child_to_descend(
          thread,
          node,
          node_state,
          &mut num_children_found,
          &mut best_child_idx,
          &mut best_child_move_loc,
          &mut count_edge_visit,
          is_root,
        );

        //The absurdly rare case that the move chosen is not legal
        //(this should only happen either on a bug or where the nnHash doesn't have full legality information or when there's an actual hash collision).
        //Regenerate the neural net call and continue
        //Could also be true if we have an illegal move due to graph search and we had a cycle and superko interaction, or a true collision
        //on an older path that results in bad transposition between positions that don't transpose.
        if best_child_idx >= 0 && !thread.history.is_legal(&thread.board, best_child_move_loc, thread.pla) {
          {
            let nn_output: Rc<NNOutput> = node.get_nn_output().expect("an expanded node has an nnOutput");
            let nn_hash: Hash128 = nn_output.nn_hash;
            //In case of a cycle or bad transposition, this will fire a lot, so limit it to once per thread per search.
            if !thread.illegal_move_hashes.contains(&nn_hash) {
              log::info!("{}", "WARNING: Chosen move not legal so about to regenerate nn output, nnhash=".to_string() + &nn_hash.to_string());
              let mut out: String = String::new();
              nn_output.debug_print(&mut out, &thread.board);
              self.debug_print_children_summary(&mut out, node, &nn_output);
              log::info!("{}", out);
            }
          }

          let is_re_init: bool = true;
          self.init_node_nn_output(thread, node, is_root, true, is_re_init).await?;

          {
            let nn_output: Rc<NNOutput> = node.get_nn_output().expect("an expanded node has an nnOutput");
            let nn_hash: Hash128 = nn_output.nn_hash;
            //In case of a cycle or bad transposition, this will fire a lot, so limit it to once per thread per search.
            if !thread.illegal_move_hashes.contains(&nn_hash) {
              thread.illegal_move_hashes.insert(nn_hash);
              log::info!("{}", "WARNING: Chosen move not legal so regenerated nn output, nnhash=".to_string() + &nn_hash.to_string());
              let mut out: String = String::new();
              thread.history.print_basic_info(&mut out, &thread.board);
              thread.history.print_debug_info(&mut out, &thread.board);
              {
                use std::fmt::Write;
                writeln!(out, "{}", best_child_idx).unwrap();
                writeln!(out, "{}", Location::to_string(best_child_move_loc, &thread.board)).unwrap();
              }
              nn_output.debug_print(&mut out, &thread.board);
              self.debug_print_children_summary(&mut out, node, &nn_output);
              log::info!("{}", out);
            }
          }

          //Give up on this playout now that we've forced the nn output to be consistent legality of this path.
          //Return TRUE though, so that the parent path we traversed increments its edge visits.
          //We want the search to continue as best it can, so we increment visits so search will still make progress
          //even if this keeps happening in some really bad transposition or something.
          return Ok(true);
        }

        if best_child_idx <= -1 {
          //This might happen if all moves have been forbidden. The node will just get stuck counting visits without expanding
          //and we won't do any search.
          self.add_current_nn_output_as_leaf_value(node, false);
          return Ok(true);
        }

        //Do we think we are searching a new child for the first time?
        if best_child_idx >= num_children_found {
          debug_assert!(best_child_idx == num_children_found);
          debug_assert!(best_child_idx < NNPos::MAX_NN_POLICY_SIZE);
          let suc: bool = node.maybe_expand_children_capacity_for_new_child(&mut node_state, num_children_found + 1);
          //Someone else is expanding. Loop again trying to select the best child to explore.
          if !suc {
            yield_now().await;
            node_state = node.state.get() /*acquire*/;
            continue;
          }

          let children: SearchNodeChildrenReference = node.get_children_state(node_state);
          let children_capacity: i32 = children.get_capacity();
          debug_assert!(children_capacity > best_child_idx);
          let _ = children_capacity;

          //We can only test this before we make the move, so do it now.
          let can_force_non_terminal_due_to_friendly_pass: bool =
            best_child_move_loc == Board::PASS_LOC && thread.history.should_suppress_end_game_from_friendly_pass(&thread.board, thread.pla);

          //Make the move! We need to make the move before we create the node so we can see the new state and get the right graphHash.
          thread.history.make_board_move_assume_legal(&mut thread.board, best_child_move_loc, thread.pla, Some(&*self.root_ko_hash_table), false);
          thread.pla = get_opp(thread.pla);
          if self.search_params.use_graph_search {
            thread.graph_hash = GraphHash::get_graph_hash(
              thread.graph_hash,
              &thread.history,
              thread.pla,
              self.search_params.graph_search_rep_bound,
              self.search_params.draw_equivalent_wins_for_white,
            );
          }

          //If conservative pass, passing from the root is always non-terminal
          //If friendly passing rules, we might also be non-terminal
          let force_non_terminal: bool = best_child_move_loc == Board::PASS_LOC
            && thread.history.is_game_finished
            && ((self.search_params.conservative_pass && (Some(node.idx) == self.root_node)) || can_force_non_terminal_due_to_friendly_pass);
          let next_pla: Player = thread.pla;
          let graph_hash: Hash128 = thread.graph_hash;
          let new_child: Rc<SearchNode> = self.allocate_or_find_node(thread, next_pla, best_child_move_loc, force_non_terminal, graph_hash);
          cpp::cell_fetch_add(&new_child.virtual_losses, 1) /*release*/;

          {
            //Lock mutex to store child and move loc in a synchronized way
            // PORT NOTE: lock removed (single-threaded).
            let existing_child: Option<NodeIdx> = children.index(best_child_idx).get_if_allocated();
            if existing_child.is_none() {
              //Set relaxed *first*, then release this value via storing the child. Anyone who load-acquires the child
              //is guaranteed by release semantics to see the move as well.
              let child_pointer: &SearchChildPointer = children.index(best_child_idx);
              child_pointer.set_move_loc_relaxed(best_child_move_loc);
              child_pointer.store(Some(new_child.idx));
            } else {
              //Someone got there ahead of us. We already made a move so we can't just loop again. Instead just fail this playout and try again.
              //Even if the node was newly allocated, no need to delete the node, it will get cleaned up next time we mark and sweep the node table later.
              //Clean up virtual losses in case the node is a transposition and is being used.
              cpp::cell_fetch_add(&new_child.virtual_losses, -1) /*release*/;
              thread.should_count_playout = false;
              return Ok(false);
            }
          }

          //If edge visits is too much smaller than the child's visits, we can avoid descending.
          //Instead just add edge visits and treat that as a visit.
          //If we're not counting edge visits, then we're deliberately trying to add child visits beyond edge visits, don't return early
          if count_edge_visit && self.maybe_catch_up_edge_visits(thread, node, &new_child, node_state, best_child_idx) {
            self.update_stats_after_playout(node, thread, is_root);
            cpp::cell_fetch_add(&new_child.virtual_losses, -1) /*release*/;
            return Ok(true);
          }
          child = new_child;
        }
        //Searching an existing child
        else {
          let children: SearchNodeChildrenReference = node.get_children_state(node_state);
          let existing_child: Rc<SearchNode> =
            self.node_arena.get(children.index(best_child_idx).get_if_allocated().expect("an existing child is allocated"));

          cpp::cell_fetch_add(&existing_child.virtual_losses, 1) /*release*/;

          //If edge visits is too much smaller than the child's visits, we can avoid descending.
          //Instead just add edge visits and treat that as a visit.
          //If we're not counting edge visits, then we're deliberately trying to add child visits beyond edge visits, don't return early
          if count_edge_visit && self.maybe_catch_up_edge_visits(thread, node, &existing_child, node_state, best_child_idx) {
            self.update_stats_after_playout(node, thread, is_root);
            cpp::cell_fetch_add(&existing_child.virtual_losses, -1) /*release*/;
            return Ok(true);
          }

          //Make the move!
          thread.history.make_board_move_assume_legal(&mut thread.board, best_child_move_loc, thread.pla, Some(&*self.root_ko_hash_table), false);
          thread.pla = get_opp(thread.pla);
          if self.search_params.use_graph_search {
            thread.graph_hash = GraphHash::get_graph_hash(
              thread.graph_hash,
              &thread.history,
              thread.pla,
              self.search_params.graph_search_rep_bound,
              self.search_params.draw_equivalent_wins_for_white,
            );
          }
          child = existing_child;
        }

        break;
      }

      //If somehow we find ourselves in a cycle, increment edge visits and terminate the playout.
      //Basically if the search likes a cycle... just reinforce playing around the cycle and hope we return something
      //reasonable in the end of the search.
      //Note that this means that child visits >= edge visits is NOT an invariant.
      {
        let inserted: bool = thread.graph_path.insert(child.idx);
        //No insertion, child was already there
        if !inserted {
          if count_edge_visit {
            let children: SearchNodeChildrenReference = node.get_children_state(node_state);
            children.index(best_child_idx).add_edge_visits(1);
            self.update_stats_after_playout(node, thread, is_root);
          }
          // Regardless of whether we count an edge visit or not here, we
          // leave thread.shouldCountPlayout as true so that if we repeatedly are stuck searching a cycle
          // we don't go forever, and eventually hit a visits/playouts limit.

          cpp::cell_fetch_add(&child.virtual_losses, -1) /*release*/;
          // If we didn't count an edge visit, none of the parents need to update either.
          return Ok(count_edge_visit);
        }
      }

      //Recurse!
      let mut should_update_child_ancestors: bool = self.playout_descend(thread, &child, false).await?;

      //Update this node stats
      should_update_child_ancestors = should_update_child_ancestors && count_edge_visit;
      if should_update_child_ancestors {
        node_state = node.state.get() /*acquire*/;
        let children: SearchNodeChildrenReference = node.get_children_state(node_state);
        children.index(best_child_idx).add_edge_visits(1);
        self.update_stats_after_playout(node, thread, is_root);
      }
      cpp::cell_fetch_add(&child.virtual_losses, -1) /*release*/;

      Ok(should_update_child_ancestors)
    })
  }

  /// upstream: Search::maybeCatchUpEdgeVisits (search.cpp)
  //If edge visits is too much smaller than the child's visits, we can avoid descending.
  //Instead just add edge visits and return immediately.
  pub(crate) fn maybe_catch_up_edge_visits(
    &self,
    thread: &mut SearchThread,
    node: &SearchNode,
    child: &SearchNode,
    node_state: SearchNodeState,
    best_child_idx: i32,
  ) -> bool {
    //Don't need to do this since we already are pretty recent as of finding the best child.
    //nodeState = node.state.load(std::memory_order_acquire);
    let children: SearchNodeChildrenReference = node.get_children_state(node_state);
    let child_pointer: &SearchChildPointer = children.index(best_child_idx);

    // int64_t maxNumToAdd = 1;
    // if(searchParams.graphSearchCatchUpProp > 0.0) {
    //   int64_t parentVisits = node.stats.visits.load(std::memory_order_acquire);
    //   //Truncate down
    //   maxNumToAdd = 1 + (int64_t)(searchParams.graphSearchCatchUpProp * parentVisits);
    // }
    let child_visits: i64 = child.stats.visits.get() /*acquire*/;
    let mut edge_visits: i64 = child_pointer.get_edge_visits();

    //If we want to leak through some of the time, then we keep searching the transposition node even if we'd be happy to stop here with
    //how many visits it has
    if self.search_params.graph_search_catch_up_leak_prob > 0.0
      && edge_visits < child_visits
      && thread.rand.next_bool(self.search_params.graph_search_catch_up_leak_prob)
    {
      return false;
    }

    //If the edge visits exceeds the child then we need to search the child more, but as long as that's not the case,
    //we can add more edge visits.
    const NUM_TO_ADD: i64 = 1;
    // int64_t numToAdd;
    loop {
      if edge_visits >= child_visits {
        return false;
      }
      // numToAdd = std::min((childVisits - edgeVisits + 3) / 4, maxNumToAdd);
      let desired: i64 = edge_visits + NUM_TO_ADD;
      if child_pointer.compexweak_edge_visits(&mut edge_visits, desired) {
        break;
      }
    }

    true
  }
}
