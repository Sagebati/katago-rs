// Ported from KataGo cpp/search/searchnnhelpers.cpp @ ba938676d7f42d70950b3a535af2466fb642008c
//
// PORT NOTE (PORTING.md §11): computeRootNNEvaluation, initNodeNNOutput and
// maybeRecomputeExistingNNOutput are async (they await NNEvaluator::evaluate, the await point of a
// playout future) and take &self. `std::shared_ptr<NNOutput>*` results are `Rc<NNOutput>`; the
// `delete`s of unused results are drops.

use std::rc::Rc;

use crate::core::global::KataError;
use crate::game::board::get_opp;
use crate::game::board::Board;
use crate::game::board::Location;
use crate::game::board::Player;
use crate::game::board::C_EMPTY;
use crate::neuralnet::nneval::NNResultBuf;
use crate::neuralnet::nninputs::MiscNNInputParams;
use crate::neuralnet::nninputs::NNOutput;
use crate::search::search::Search;
use crate::search::search::SearchThread;
use crate::search::searchnode::SearchNode;
use crate::testAssert;

impl Search {
  /// upstream: Search::computeRootNNEvaluation (searchnnhelpers.cpp)
  /// PORT NOTE: async; throws -> Result. The buf is an Rc (NNEvaluator queues it while awaiting).
  pub(crate) async fn compute_root_nn_evaluation(&self, nn_result_buf: &Rc<NNResultBuf>, include_owner_map: bool) -> Result<(), KataError> {
    let board: Box<Board> = Box::new(self.root_board.clone());
    let hist = &self.root_history;
    let pla: Player = self.root_pla;
    let skip_cache: bool = false;
    let is_root: bool = true;
    let mut nn_input_params: MiscNNInputParams = MiscNNInputParams::new();
    nn_input_params.draw_equivalent_wins_for_white = self.search_params.draw_equivalent_wins_for_white;
    nn_input_params.conservative_pass_and_is_root = self.search_params.conservative_pass && is_root;
    nn_input_params.enable_passing_hacks = self.search_params.enable_passing_hacks;
    nn_input_params.nn_policy_temperature = self.search_params.nn_policy_temperature;
    nn_input_params.avoid_myt_dagger_hack = self.search_params.avoid_myt_dagger_hack_pla == pla;
    nn_input_params.policy_optimism = self.search_params.root_policy_optimism;
    if self.search_params.playout_doubling_advantage != 0.0 {
      let playout_doubling_advantage_pla: Player = self.get_playout_doubling_advantage_pla();
      nn_input_params.playout_doubling_advantage = if get_opp(pla) == playout_doubling_advantage_pla {
        -self.search_params.playout_doubling_advantage
      } else {
        self.search_params.playout_doubling_advantage
      };
    }
    if self.search_params.ignore_pre_root_history || self.search_params.ignore_all_history {
      nn_input_params.max_history = 0;
    }
    self
      .nn_evaluator
      .evaluate_sgf_meta(&board, hist, pla, Some(&self.search_params.human_sl_profile), &nn_input_params, nn_result_buf, skip_cache, include_owner_map)
      .await
  }

  /// upstream: Search::needsHumanOutputAtRoot (searchnnhelpers.cpp)
  pub(crate) fn needs_human_output_at_root(&self) -> bool {
    match &self.human_evaluator {
      None => false,
      Some(human_evaluator) => self.search_params.human_sl_profile.initialized || !human_evaluator.requires_sgf_metadata(),
    }
  }

  /// upstream: Search::needsHumanOutputInTree (searchnnhelpers.cpp)
  pub(crate) fn needs_human_output_in_tree(&self) -> bool {
    self.needs_human_output_at_root()
      && (self.search_params.human_sl_pla_explore_prob_weightless > 0.0
        || self.search_params.human_sl_pla_explore_prob_weightful > 0.0
        || self.search_params.human_sl_opp_explore_prob_weightless > 0.0
        || self.search_params.human_sl_opp_explore_prob_weightful > 0.0)
  }

  /// upstream: Search::initNodeNNOutput (searchnnhelpers.cpp)
  //If isReInit is false, among any threads trying to store, the first one wins
  //If isReInit is true, we always replace, even for threads that come later.
  //Returns true if a nnOutput was set where there was none before.
  /// PORT NOTE: async; throws -> Result.
  pub(crate) async fn init_node_nn_output(
    &self,
    thread: &mut SearchThread,
    node: &SearchNode,
    is_root: bool,
    skip_cache: bool,
    is_re_init: bool,
  ) -> Result<bool, KataError> {
    let include_owner_map: bool = is_root || self.always_include_owner_map;
    let anti_mirror_difficult: bool = false;
    if self.search_params.anti_mirror
      && self.mirroring_pla != C_EMPTY
      && self.mirror_advantage >= -0.5
      && Location::get_center_loc(&thread.board) != Board::NULL_LOC
      && thread.board.colors[Location::get_center_loc(&thread.board) as usize] == get_opp(self.root_pla)
    {
      // TODO(port): deferred (searchmirror.cpp): the last conjunct
      // `isMirroringSinceSearchStart(thread.history,4)` (skip recent 4 ply to be a bit tolerant) and the
      // body `includeOwnerMap = true; antiMirrorDifficult = true;`. Unreachable while updateMirroring
      // is deferred: mirroringPla stays C_EMPTY.
      unreachable!("anti-mirror is deferred (searchmirror.cpp): mirroringPla is always C_EMPTY");
    }
    let mut nn_input_params: MiscNNInputParams = MiscNNInputParams::new();
    nn_input_params.draw_equivalent_wins_for_white = self.search_params.draw_equivalent_wins_for_white;
    nn_input_params.conservative_pass_and_is_root = self.search_params.conservative_pass && is_root;
    nn_input_params.enable_passing_hacks = self.search_params.enable_passing_hacks;
    nn_input_params.nn_policy_temperature = self.search_params.nn_policy_temperature;
    nn_input_params.avoid_myt_dagger_hack = self.search_params.avoid_myt_dagger_hack_pla == thread.pla;
    nn_input_params.policy_optimism = if is_root { self.search_params.root_policy_optimism } else { self.search_params.policy_optimism };
    if self.search_params.playout_doubling_advantage != 0.0 {
      let playout_doubling_advantage_pla: Player = self.get_playout_doubling_advantage_pla();
      nn_input_params.playout_doubling_advantage = if get_opp(thread.pla) == playout_doubling_advantage_pla {
        -self.search_params.playout_doubling_advantage
      } else {
        self.search_params.playout_doubling_advantage
      };
    }
    if self.search_params.ignore_all_history {
      nn_input_params.max_history = 0;
    } else if self.search_params.ignore_pre_root_history {
      nn_input_params.max_history =
        if is_root { 0 } else { std::cmp::max(0, thread.history.move_history.len() as i32 - self.root_history.move_history.len() as i32) };
    }

    let mut result: Option<Rc<NNOutput>>;
    let mut human_result: Option<Rc<NNOutput>> = None;
    let nn_result_buf: Rc<NNResultBuf> = thread.nn_result_buf.clone();
    if is_root && self.search_params.root_num_symmetries_to_sample > 1 {
      result = Some(
        self
          .nn_evaluator
          .average_multiple_symmetries(
            &thread.board,
            &thread.history,
            thread.pla,
            Some(&self.search_params.human_sl_profile),
            &nn_input_params,
            &nn_result_buf,
            include_owner_map,
            &mut thread.rand,
            self.search_params.root_num_symmetries_to_sample,
          )
          .await?,
      );
      if self.needs_human_output_in_tree() || (is_root && self.needs_human_output_at_root()) {
        let human_evaluator = self.human_evaluator.as_ref().expect("needsHumanOutput implies a humanEvaluator");
        human_result = Some(
          human_evaluator
            .average_multiple_symmetries(
              &thread.board,
              &thread.history,
              thread.pla,
              Some(&self.search_params.human_sl_profile),
              &nn_input_params,
              &nn_result_buf,
              include_owner_map,
              &mut thread.rand,
              self.search_params.root_num_symmetries_to_sample,
            )
            .await?,
        );
      }
    } else {
      self
        .nn_evaluator
        .evaluate_sgf_meta(
          &thread.board,
          &thread.history,
          thread.pla,
          Some(&self.search_params.human_sl_profile),
          &nn_input_params,
          &nn_result_buf,
          skip_cache,
          include_owner_map,
        )
        .await?;
      result = Some(nn_result_buf.result.borrow_mut().take().expect("evaluate leaves a result in the buf"));
      if self.needs_human_output_in_tree() || (is_root && self.needs_human_output_at_root()) {
        let human_evaluator = self.human_evaluator.as_ref().expect("needsHumanOutput implies a humanEvaluator");
        human_evaluator
          .evaluate_sgf_meta(
            &thread.board,
            &thread.history,
            thread.pla,
            Some(&self.search_params.human_sl_profile),
            &nn_input_params,
            &nn_result_buf,
            skip_cache,
            include_owner_map,
          )
          .await?;
        human_result = Some(nn_result_buf.result.borrow_mut().take().expect("evaluate leaves a result in the buf"));
      }
    }

    if anti_mirror_difficult {
      // Copy
      // TODO(port): deferred (searchmirror.cpp): `result = new shared_ptr(new NNOutput(**result)); hackNNOutputForMirror(*result);`
      // Unreachable (see above).
      unreachable!("anti-mirror is deferred (searchmirror.cpp)");
    }

    debug_assert!(result.as_ref().expect("set above").noised_policy_probs.is_none());
    let noised_result: Option<Rc<NNOutput>> = {
      let r: Rc<NNOutput> = result.clone().expect("set above");
      self.maybe_add_policy_noise_and_temp(thread, is_root, &r)?
    };
    if noised_result.is_some() {
      result = noised_result;
    }
    let result: Rc<NNOutput> = result.expect("set above");

    node.node_age.set(self.search_node_age.get()) /*release*/;
    //If this is a re-initialization of the nnOutput, we don't want to add any visits or anything.
    //Also don't bother updating any of the stats. Technically we should do so because winLossValueSum
    //and such will have changed potentially due to a new orientation of the neural net eval
    //slightly affecting the evals, but this is annoying to recompute from scratch, and on the next
    //visit updateStatsAfterPlayout should fix it all up anyways.
    if is_re_init {
      if let Some(human_result) = human_result {
        node.store_human_output(human_result, thread); // ignore the wasNullBefore from this one
      }
      let was_null_before: bool = node.store_nn_output(result, thread);
      Ok(was_null_before)
    } else {
      // Store human result first, so that the presence of the main result guarantees
      // that the human result exists in the case we have a human evaluator.
      if let Some(human_result) = human_result {
        let human_suc: bool = node.store_human_output_if_null(human_result);
        // PORT NOTE: `if(!humanSuc) delete humanResult;` is the drop inside store_human_output_if_null.
        let _ = human_suc;
      }
      let suc: bool = node.store_nn_output_if_null(result);
      // PORT NOTE: `if(!suc) delete result;` is the drop inside store_nn_output_if_null.
      if suc {
        self.add_current_nn_output_as_leaf_value(node, true);
      }
      Ok(suc)
    }
  }

  /// upstream: Search::maybeRecomputeExistingNNOutput (searchnnhelpers.cpp)
  //Assumes node already has an nnOutput
  // Returns true if any recomputation happened
  /// PORT NOTE: async; throws -> Result.
  pub(crate) async fn maybe_recompute_existing_nn_output(&self, thread: &mut SearchThread, node: &SearchNode, is_root: bool) -> Result<bool, KataError> {
    let mut recompute_happened: bool = false;
    //Right now only the root node currently ever needs to recompute, and only if it's old
    if is_root && node.node_age.get() /*acquire*/ != self.search_node_age.get() {
      //See if we're the lucky thread that gets to do the update!
      //Threads that pass by here later will NOT wait for us to be done before proceeding with search.
      //We accept this and tolerate that for a few iterations potentially we will be using the OLD policy - without noise,
      //or without root temperature, etc.
      //Or if we have none of those things, then we'll not end up updating anything except the age, which is okay too.
      let old_age: u32 = node.node_age.replace(self.search_node_age.get()) /*acq_rel*/;
      if old_age < self.search_node_age.get() {
        let nn_output: Option<Rc<NNOutput>> = node.get_nn_output();
        let human_output: Option<Rc<NNOutput>> = node.get_human_output();
        testAssert!(nn_output.is_some());
        let nn_output: Rc<NNOutput> = nn_output.expect("testAssert above");

        //Recompute if we have no ownership map, since we need it for getEndingWhiteScoreBonus
        //If conservative passing, then we may also need to recompute the root policy ignoring the history if a pass ends the game
        //If averaging a bunch of symmetries, then we need to recompute it too
        //If root needs different optimism, we need to recompute it.
        //If humanSL is missing, but we want it, we need to recompute.
        //Also do so when ignoring history pre root
        if nn_output.white_owner_map.is_none()
          || (self.search_params.conservative_pass && thread.history.pass_would_end_game(&thread.board, thread.pla))
          || self.search_params.root_num_symmetries_to_sample > 1
          || self.search_params.root_policy_optimism != self.search_params.policy_optimism
          || (self.search_params.ignore_pre_root_history && !self.search_params.ignore_all_history)
          || (human_output.is_none() && self.needs_human_output_at_root())
        {
          //We *can* use cached evaluations even though parameters are changing, because:
          //conservativePass is part of the nn hash
          //Symmetry averaging skips the cache on its own when it does symmetry sampling without replacement
          //The optimism is part of the nn hash
          //When pre-root history is ignored at the root, maxHistory is 0 and the nn cache distinguishes 0 from nonzero.
          let skip_cache: bool = false;
          self.init_node_nn_output(thread, node, is_root, skip_cache, true).await?;
          recompute_happened = true;
        }
        //We also need to recompute the root nn if we have root noise or temperature and that's missing.
        else {
          //We don't need to go all the way to the nnEvaluator, we just need to maybe add those transforms
          //to the existing policy.
          let result: Option<Rc<NNOutput>> = self.maybe_add_policy_noise_and_temp(thread, is_root, &nn_output)?;
          if let Some(result) = result {
            node.store_nn_output(result, thread);
            recompute_happened = true;
          }
        }
      }
    }
    Ok(recompute_happened)
  }
}
