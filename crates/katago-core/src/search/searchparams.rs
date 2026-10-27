// Ported from KataGo cpp/search/searchparams.cpp, cpp/search/searchparams.h @ fd0723fdbc0e9d82cf269c9630af8c27c57c07c4

use serde_json::Map;
use serde_json::Value;

use crate::core::commontypes::enabled_t;
use crate::core::global::KataError;
use crate::core::hash::Hash128;
use crate::game::board::Player;
use crate::game::board::PlayerIO;
use crate::game::board::C_EMPTY;
use crate::game::board::P_BLACK;
use crate::neuralnet::sgfmetadata::SGFMetadata;
use crate::port::fmt::cpp_ostream_f64;

/// upstream: SearchParams (searchparams.h)
#[derive(Clone, Debug)]
pub struct SearchParams {
  //Utility function parameters
  pub win_loss_utility_factor: f64, //Scaling for [-1,1] value for winning/losing
  pub static_score_utility_factor: f64, //Scaling for a [-1,1] "scoreValue" for having more/fewer points, centered at 0.
  pub dynamic_score_utility_factor: f64, //Scaling for a [-1,1] "scoreValue" for having more/fewer points, centered at recent estimated expected score.
  pub dynamic_score_center_zero_weight: f64, //Adjust dynamic score center this proportion of the way towards zero, capped at a reasonable amount.
  pub dynamic_score_center_scale: f64, //Adjust dynamic score scale. 1.0 indicates that score is cared about roughly up to board sizeish.
  pub no_result_utility_for_white: f64, //Utility of having a no-result game (simple ko rules or nonterminating territory encore)
  pub draw_equivalent_wins_for_white: f64, //Consider a draw to be this many wins and one minus this many losses.

  //Search tree exploration parameters
  pub cpuct_exploration: f64, //Constant factor on exploration, should also scale up linearly with magnitude of utility
  pub cpuct_exploration_log: f64, //Constant factor on log-scaling exploration, should also scale up linearly with magnitude of utility
  pub cpuct_exploration_base: f64, //Scale of number of visits at which log behavior starts having an effect

  pub cpuct_utility_stdev_prior: f64,
  pub cpuct_utility_stdev_prior_weight: f64,
  pub cpuct_utility_stdev_scale: f64,

  pub fpu_reduction_max: f64, //Max amount to reduce fpu value for unexplore children
  pub fpu_loss_prop: f64, //Scale fpu this proportion of the way towards assuming a move is a loss.

  pub fpu_parent_weight_by_visited_policy: bool, //For fpu, blend between parent average and parent nn value based on proportion of policy visited.
  pub fpu_parent_weight_by_visited_policy_pow: f64, //If fpuParentWeightByVisitedPolicy, what power to raise the proportion of policy visited for blending.
  pub fpu_parent_weight: f64, //For fpu, 0 = use parent average, 1 = use parent nn value, interpolates between.

  pub policy_optimism: f64, //Interpolate geometrically between raw policy and optimistic policy

  //Tree value aggregation parameters
  pub value_weight_exponent: f64, //Amount to apply a downweighting of children with very bad values relative to good ones
  pub use_noise_pruning: bool, //For computation of value, prune out weight that greatly exceeds what is justified by policy prior
  pub noise_prune_utility_scale: f64, //The scale of the utility difference at which useNoisePruning has effect
  pub noise_pruning_cap: f64, //Maximum amount of weight that noisePruning can remove

  //Uncertainty weighting
  pub use_uncertainty: bool, //Weight visits by uncertainty
  pub uncertainty_coeff: f64, //The amount of visits weight that an uncertainty of 1 utility is.
  pub uncertainty_exponent: f64, //Visits weight scales inversely with this power of the uncertainty
  pub uncertainty_max_weight: f64, //Add minimum uncertainty so that the most weight a node can have is this

  //Graph search
  pub use_graph_search: bool, //Enable graph search instead of tree search?
  pub graph_search_rep_bound: i32, //Rep bound to use for graph search transposition safety. Higher will reduce transpositions but be more safe.
  pub graph_search_catch_up_leak_prob: f64, //Chance to perform a visit to deepen a branch anyways despite being behind on visit count.
  //double graphSearchCatchUpProp; //When sufficiently far behind on visits on a transposition, catch up extra by adding up to this fraction of parents visits at once.

  //Root parameters
  pub root_noise_enabled: bool,
  pub root_dirichlet_noise_total_concentration: f64, //Same as alpha * board size, to match alphazero this might be 0.03 * 361, total number of balls in the urn
  pub root_dirichlet_noise_weight: f64, //Policy at root is this weight * noise + (1 - this weight) * nn policy

  pub root_policy_temperature: f64, //At the root node, scale policy probs by this power
  pub root_policy_temperature_early: f64, //At the root node, scale policy probs by this power, early in the game
  pub root_fpu_reduction_max: f64, //Same as fpuReductionMax, but at root
  pub root_fpu_loss_prop: f64, //Same as fpuLossProp, but at root
  pub root_num_symmetries_to_sample: i32, //For the root node, sample this many random symmetries (WITHOUT replacement) and average the results together.
  pub root_symmetry_pruning: bool, //For the root node, search only one copy of each symmetrically equivalent move.
  //We use the min of these two together, and also excess visits get pruned if the value turns out bad.
  pub root_desired_per_child_visits_coeff: f64, //Funnel sqrt(this * policy prob * total visits) down any given child that receives any visits at all at the root

  pub root_policy_optimism: f64, //Interpolate geometrically between raw policy and optimistic policy

  //Parameters for choosing the move to play
  pub chosen_move_temperature: f64, //Make move roughly proportional to visit count ** (1/chosenMoveTemperature)
  pub chosen_move_temperature_early: f64, //Temperature at start of game
  pub chosen_move_temperature_halflife: f64, //Halflife of decay from early temperature to temperature for the rest of the game, scales for board sizes other than 19.
  pub chosen_move_temperature_only_below_prob: f64, //chosenMoveTemperature only begins dampening moves that pre-temperature are less likely than this.
  pub chosen_move_subtract: f64, //Try to subtract this many visits from every move prior to applying temperature
  pub chosen_move_prune: f64, //Outright prune moves that have fewer than this many visits

  pub use_lcb_for_selection: bool, //Using LCB for move selection?
  pub lcb_stdevs: f64, //How many stdevs a move needs to be better than another for LCB selection
  pub min_visit_prop_for_lcb: f64, //Only use LCB override when a move has this proportion of visits as the top move
  pub use_non_buggy_lcb: bool, //LCB was very minorly buggy as of pre-v1.8. Set to true to fix.

  //Mild behavior hackery
  pub root_ending_bonus_points: f64, //Extra bonus (or penalty) to encourage good passing behavior at the end of the game.
  pub root_prune_useless_moves: bool, //Prune moves that are entirely useless moves that prolong the game.
  pub conservative_pass: bool, //Never assume one's own pass will end the game.
  pub fill_dame_before_pass: bool, //When territory scoring, heuristically discourage passing before filling the dame.
  pub avoid_myt_dagger_hack_pla: Player, //Hacky hack to avoid a particular pattern that gives some KG nets some trouble. Should become unnecessary in the future.
  pub wide_root_noise: f64, //Explore at the root more widely
  pub enable_passing_hacks: bool, //Enable some hacks that mitigate rare instances when passing messes up deeper searches.
  pub enable_more_passing_hacks: bool, //Always weightless search passing and non passing moves when a pass would end the phase after a few visits.
  //Whether to compute pass-alive areas as if multi-stone suicide were always legal regardless of the actual suicide
  //rule, for game end/scoring within the search tree and for nn input featurization.
  //Auto = do so if and only if the neural net declares that it expects this. See BoardHistoryModes.
  pub always_compute_pass_alive_under_suicide_rules: enabled_t,
  //Whether territory scoring with TaxRule NONE excludes empty points adjacent to chains in atari (rules version 3),
  //for game end/scoring within the search tree and for nn input featurization.
  //Auto = do so if and only if the neural net declares that it expects this. See BoardHistoryModes.
  pub exclude_territory_adjacent_to_atari: enabled_t,

  pub playout_doubling_advantage: f64, //Play as if we have this many doublings of playouts vs the opponent
  pub playout_doubling_advantage_pla: Player, //Negate playoutDoublingAdvantage when making a move for the opponent of this player. If empty, opponent of the root player.

  pub avoid_repeated_pattern_utility: f64, //Have the root player avoid repeating similar shapes, penalizing this much utility per instance.

  pub nn_policy_temperature: f32, //Scale neural net policy probabilities by this temperature, applies everywhere in the tree
  pub anti_mirror: bool, //Enable anti-mirroring logic

  //Ignore history prior to the root of the search. This is enforced strictly only for the root node of the
  //search. Deeper nodes may see history prior to the root of the search if searches were performed from earlier positions
  //and those gamestates were also reached by those earlier searches with the nn evals cached.
  //This is true even without tree reuse.
  pub ignore_pre_root_history: bool,
  pub ignore_all_history: bool, //Always ignore history entirely

  pub subtree_value_bias_factor: f64, //Dynamically adjust neural net utilties based on empirical stats about their errors in search
  pub subtree_value_bias_table_num_shards: i32, //Number of shards for subtreeValueBiasFactor for initial hash lookup and mutexing
  pub subtree_value_bias_free_prop: f64, //When a node is no longer part of the relevant search tree, only decay this proportion of the weight.
  pub subtree_value_bias_weight_exponent: f64, //When computing empiricial bias, weight subtree results by childvisits to this power.

  pub use_eval_cache: bool,
  pub eval_cache_min_visits: i64,

  //Threading-related
  pub node_table_shards_power_of_two: i32, //Controls number of shards of node table for graph search transposition lookup
  pub num_virtual_losses_per_thread: f64, //Number of virtual losses for one thread to add

  //Asyncbot
  pub num_threads: i32, //Number of threads
  pub min_playouts_per_thread: f64, //If the number of playouts to perform per thread is smaller than this, cap the number of threads used.
  pub max_visits: i64, //Max number of playouts from the root to think for, counting earlier playouts from tree reuse
  pub max_playouts: i64, //Max number of playouts from the root to think for, not counting earlier playouts from tree reuse
  pub max_time: f64, //Max number of seconds to think for

  //Same caps but when pondering
  pub max_visits_pondering: i64,
  pub max_playouts_pondering: i64,
  pub max_time_pondering: f64,

  //Amount of time to reserve for lag when using a time control
  pub lag_buffer: f64,

  //Human-friendliness
  pub search_factor_after_one_pass: f64, //Multiply playouts and visits and time by this much after a pass by the opponent
  pub search_factor_after_two_pass: f64, //Multiply playouts and visits and time by this after two passes by the opponent

  //Time control
  pub tree_reuse_carry_over_time_factor: f64, //Assume we gain this much "time" on the next move purely from % tree preserved * time spend on that tree.
  pub overallocate_time_factor: f64, //Prefer to think this factor longer than recommended by base level time control
  pub midgame_time_factor: f64, //Think this factor longer in the midgame, proportional to midgame weight
  pub midgame_turn_peak_time: f64, //The turn considered to have midgame weight 1.0, rising up from 0.0 in the opening, for 19x19
  pub endgame_turn_time_decay: f64, //The scale of exponential decay of midgame weight back to 1.0, for 19x19
  pub obvious_moves_time_factor: f64, //Think up to this factor longer on obvious moves, weighted by obviousness
  pub obvious_moves_policy_entropy_tolerance: f64, //What entropy does the policy need to be at most to be (1/e) obvious?
  pub obvious_moves_policy_surprise_tolerance: f64, //What logits of surprise does the search result need to be at most to be (1/e) obvious?

  pub futile_visits_threshold: f64, //If a move would not be able to match this proportion of the max visits move in the time or visit or playout cap remaining, prune it.

  //Human SL network
  pub human_sl_profile: SGFMetadata, //For any human SL model, make the net predict this rank / source / profile of player.
  pub human_sl_cpuct_exploration: f64, //Use this cpuct for -human-model. This divided by sqrt(visits) is the utility diff to majorly affect the posterior.
  pub human_sl_cpuct_permanent: f64, //Same, but multiplied by sqrt(visits). This is the utility diff to majorly affect the posterior.

  //Probability that a playout selects a move using the human SL net instead of the main net.
  //Weightless - search the move to better judge it but do NOT weight that playout for the parent's value.
  //Weightful - search the move and DO weight that playout for the parent's value. (Note: consider disabling useNoisePruning, which uses katago's policy).
  pub human_sl_root_explore_prob_weightless: f64,
  pub human_sl_root_explore_prob_weightful: f64,
  pub human_sl_pla_explore_prob_weightless: f64,
  pub human_sl_pla_explore_prob_weightful: f64,
  pub human_sl_opp_explore_prob_weightless: f64,
  pub human_sl_opp_explore_prob_weightful: f64,

  //These three are PRIOR to the normal chosenMoveTemperature.
  pub human_sl_chosen_move_prop: f64, //Proportion of final move selection probability using human SL policy
  pub human_sl_chosen_move_ignore_pass: bool, //If true, ignore human SL pass probability and use KataGo's passing logic
  pub human_sl_chosen_move_pikl_lambda: f64, //Shift the final move selection significantly in response to utility differences this large.
}

impl SearchParams {
  /// upstream: SearchParams::SearchParams (searchparams.cpp)
  //Default search params
  //The intent is that the are good default guesses for values of the parameters,
  //with deterministic behavior (no noise, no randomization) and no bounds (unbounded time and visits).
  //They are not necessarily the best parameters though, and have been kept mostly fixed over time even as things
  //have changed to preserve the behavior of tests.
  pub fn new() -> SearchParams {
    SearchParams {
      win_loss_utility_factor: 1.0,
      static_score_utility_factor: 0.3,
      dynamic_score_utility_factor: 0.0,
      dynamic_score_center_zero_weight: 0.0,
      dynamic_score_center_scale: 1.0,
      no_result_utility_for_white: 0.0,
      draw_equivalent_wins_for_white: 0.5,
      cpuct_exploration: 1.0,
      cpuct_exploration_log: 0.0,
      cpuct_exploration_base: 500.0,
      cpuct_utility_stdev_prior: 0.25,
      cpuct_utility_stdev_prior_weight: 1.0,
      cpuct_utility_stdev_scale: 0.0,
      fpu_reduction_max: 0.2,
      fpu_loss_prop: 0.0,
      fpu_parent_weight_by_visited_policy: false,
      fpu_parent_weight_by_visited_policy_pow: 1.0,
      fpu_parent_weight: 0.0,
      policy_optimism: 0.0,
      value_weight_exponent: 0.5,
      use_noise_pruning: false,
      noise_prune_utility_scale: 0.15,
      noise_pruning_cap: 1e50,
      use_uncertainty: false,
      uncertainty_coeff: 0.2,
      uncertainty_exponent: 1.0,
      uncertainty_max_weight: 8.0,
      use_graph_search: false,
      graph_search_rep_bound: 11,
      graph_search_catch_up_leak_prob: 0.0,
      //graphSearchCatchUpProp(0.0),
      root_noise_enabled: false,
      root_dirichlet_noise_total_concentration: 10.83,
      root_dirichlet_noise_weight: 0.25,
      root_policy_temperature: 1.0,
      root_policy_temperature_early: 1.0,
      root_fpu_reduction_max: 0.2,
      root_fpu_loss_prop: 0.0,
      root_num_symmetries_to_sample: 1,
      root_symmetry_pruning: false,
      root_desired_per_child_visits_coeff: 0.0,
      root_policy_optimism: 0.0,
      chosen_move_temperature: 0.0,
      chosen_move_temperature_early: 0.0,
      chosen_move_temperature_halflife: 19.0,
      chosen_move_temperature_only_below_prob: 1.0,
      chosen_move_subtract: 0.0,
      chosen_move_prune: 1.0,
      use_lcb_for_selection: false,
      lcb_stdevs: 4.0,
      min_visit_prop_for_lcb: 0.05,
      use_non_buggy_lcb: false,
      root_ending_bonus_points: 0.0,
      root_prune_useless_moves: false,
      conservative_pass: false,
      fill_dame_before_pass: false,
      avoid_myt_dagger_hack_pla: C_EMPTY,
      wide_root_noise: 0.0,
      enable_passing_hacks: false,
      enable_more_passing_hacks: false,
      always_compute_pass_alive_under_suicide_rules: enabled_t::Auto,
      exclude_territory_adjacent_to_atari: enabled_t::Auto,
      playout_doubling_advantage: 0.0,
      playout_doubling_advantage_pla: C_EMPTY,
      avoid_repeated_pattern_utility: 0.0,
      nn_policy_temperature: 1.0f32,
      anti_mirror: false,
      ignore_pre_root_history: false,
      ignore_all_history: false,
      subtree_value_bias_factor: 0.0,
      subtree_value_bias_table_num_shards: 65536,
      subtree_value_bias_free_prop: 0.8,
      subtree_value_bias_weight_exponent: 0.5,
      use_eval_cache: false,
      eval_cache_min_visits: 100,
      node_table_shards_power_of_two: 16,
      num_virtual_losses_per_thread: 3.0,
      num_threads: 1,
      min_playouts_per_thread: 0.0,
      max_visits: 1i64 << 50,
      max_playouts: 1i64 << 50,
      max_time: 1.0e20,
      max_visits_pondering: 1i64 << 50,
      max_playouts_pondering: 1i64 << 50,
      max_time_pondering: 1.0e20,
      lag_buffer: 0.0,
      search_factor_after_one_pass: 1.0,
      search_factor_after_two_pass: 1.0,
      tree_reuse_carry_over_time_factor: 0.0,
      overallocate_time_factor: 1.0,
      midgame_time_factor: 1.0,
      midgame_turn_peak_time: 130.0,
      endgame_turn_time_decay: 100.0,
      obvious_moves_time_factor: 1.0,
      obvious_moves_policy_entropy_tolerance: 0.30,
      obvious_moves_policy_surprise_tolerance: 0.15,
      futile_visits_threshold: 0.0,
      human_sl_profile: SGFMetadata::default(),
      human_sl_cpuct_exploration: 1.0,
      human_sl_cpuct_permanent: 0.0,
      human_sl_root_explore_prob_weightless: 0.0,
      human_sl_root_explore_prob_weightful: 0.0,
      human_sl_pla_explore_prob_weightless: 0.0,
      human_sl_pla_explore_prob_weightful: 0.0,
      human_sl_opp_explore_prob_weightless: 0.0,
      human_sl_opp_explore_prob_weightful: 0.0,
      human_sl_chosen_move_prop: 0.0,
      human_sl_chosen_move_ignore_pass: false,
      human_sl_chosen_move_pikl_lambda: 1000000000.0,
    }
  }
}

// ~SearchParams() is an empty destructor: omitted (PORTING.md §5).

/// upstream: SearchParams::operator==, SearchParams::operator!= (searchparams.cpp)
/// `!=` is the negation of this `==` (the C++ `operator!=` is `!(*this == other)`).
impl PartialEq for SearchParams {
  fn eq(&self, other: &SearchParams) -> bool {
    self.win_loss_utility_factor == other.win_loss_utility_factor &&
    self.static_score_utility_factor == other.static_score_utility_factor &&
    self.dynamic_score_utility_factor == other.dynamic_score_utility_factor &&
    self.dynamic_score_center_zero_weight == other.dynamic_score_center_zero_weight &&
    self.dynamic_score_center_scale == other.dynamic_score_center_scale &&
    self.no_result_utility_for_white == other.no_result_utility_for_white &&
    self.draw_equivalent_wins_for_white == other.draw_equivalent_wins_for_white &&

    self.cpuct_exploration == other.cpuct_exploration &&
    self.cpuct_exploration_log == other.cpuct_exploration_log &&
    self.cpuct_exploration_base == other.cpuct_exploration_base &&

    self.cpuct_utility_stdev_prior == other.cpuct_utility_stdev_prior &&
    self.cpuct_utility_stdev_prior_weight == other.cpuct_utility_stdev_prior_weight &&
    self.cpuct_utility_stdev_scale == other.cpuct_utility_stdev_scale &&

    self.fpu_reduction_max == other.fpu_reduction_max &&
    self.fpu_loss_prop == other.fpu_loss_prop &&

    self.fpu_parent_weight_by_visited_policy == other.fpu_parent_weight_by_visited_policy &&
    self.fpu_parent_weight_by_visited_policy_pow == other.fpu_parent_weight_by_visited_policy_pow &&
    self.fpu_parent_weight == other.fpu_parent_weight &&

    self.policy_optimism == other.policy_optimism &&

    self.value_weight_exponent == other.value_weight_exponent &&
    self.use_noise_pruning == other.use_noise_pruning &&
    self.noise_prune_utility_scale == other.noise_prune_utility_scale &&
    self.noise_pruning_cap == other.noise_pruning_cap &&

    self.use_uncertainty == other.use_uncertainty &&
    self.uncertainty_coeff == other.uncertainty_coeff &&
    self.uncertainty_exponent == other.uncertainty_exponent &&
    self.uncertainty_max_weight == other.uncertainty_max_weight &&

    self.use_graph_search == other.use_graph_search &&
    self.graph_search_rep_bound == other.graph_search_rep_bound &&
    self.graph_search_catch_up_leak_prob == other.graph_search_catch_up_leak_prob &&

    self.root_noise_enabled == other.root_noise_enabled &&
    self.root_dirichlet_noise_total_concentration == other.root_dirichlet_noise_total_concentration &&
    self.root_dirichlet_noise_weight == other.root_dirichlet_noise_weight &&

    self.root_policy_temperature == other.root_policy_temperature &&
    self.root_policy_temperature_early == other.root_policy_temperature_early &&
    self.root_fpu_reduction_max == other.root_fpu_reduction_max &&
    self.root_fpu_loss_prop == other.root_fpu_loss_prop &&
    self.root_num_symmetries_to_sample == other.root_num_symmetries_to_sample &&
    self.root_symmetry_pruning == other.root_symmetry_pruning &&
    self.root_desired_per_child_visits_coeff == other.root_desired_per_child_visits_coeff &&

    self.root_policy_optimism == other.root_policy_optimism &&

    self.chosen_move_temperature == other.chosen_move_temperature &&
    self.chosen_move_temperature_early == other.chosen_move_temperature_early &&
    self.chosen_move_temperature_halflife == other.chosen_move_temperature_halflife &&

    self.chosen_move_temperature_only_below_prob == other.chosen_move_temperature_only_below_prob &&
    self.chosen_move_subtract == other.chosen_move_subtract &&
    self.chosen_move_prune == other.chosen_move_prune &&

    self.use_lcb_for_selection == other.use_lcb_for_selection &&
    self.lcb_stdevs == other.lcb_stdevs &&
    self.min_visit_prop_for_lcb == other.min_visit_prop_for_lcb &&
    self.use_non_buggy_lcb == other.use_non_buggy_lcb &&

    self.root_ending_bonus_points == other.root_ending_bonus_points &&
    self.root_prune_useless_moves == other.root_prune_useless_moves &&
    self.conservative_pass == other.conservative_pass &&
    self.fill_dame_before_pass == other.fill_dame_before_pass &&
    self.avoid_myt_dagger_hack_pla == other.avoid_myt_dagger_hack_pla &&
    self.wide_root_noise == other.wide_root_noise &&
    self.enable_passing_hacks == other.enable_passing_hacks &&
    self.enable_more_passing_hacks == other.enable_more_passing_hacks &&
    self.always_compute_pass_alive_under_suicide_rules == other.always_compute_pass_alive_under_suicide_rules &&
    self.exclude_territory_adjacent_to_atari == other.exclude_territory_adjacent_to_atari &&

    self.playout_doubling_advantage == other.playout_doubling_advantage &&
    self.playout_doubling_advantage_pla == other.playout_doubling_advantage_pla &&

    self.avoid_repeated_pattern_utility == other.avoid_repeated_pattern_utility &&

    self.nn_policy_temperature == other.nn_policy_temperature &&
    self.anti_mirror == other.anti_mirror &&

    self.ignore_pre_root_history == other.ignore_pre_root_history &&
    self.ignore_all_history == other.ignore_all_history &&

    self.subtree_value_bias_factor == other.subtree_value_bias_factor &&
    self.subtree_value_bias_table_num_shards == other.subtree_value_bias_table_num_shards &&
    self.subtree_value_bias_free_prop == other.subtree_value_bias_free_prop &&
    self.subtree_value_bias_weight_exponent == other.subtree_value_bias_weight_exponent &&

    self.use_eval_cache == other.use_eval_cache &&
    self.eval_cache_min_visits == other.eval_cache_min_visits &&

    self.node_table_shards_power_of_two == other.node_table_shards_power_of_two &&
    self.num_virtual_losses_per_thread == other.num_virtual_losses_per_thread &&

    self.num_threads == other.num_threads &&
    self.min_playouts_per_thread == other.min_playouts_per_thread &&
    self.max_visits == other.max_visits &&
    self.max_playouts == other.max_playouts &&
    self.max_time == other.max_time &&

    self.max_visits_pondering == other.max_visits_pondering &&
    self.max_playouts_pondering == other.max_playouts_pondering &&
    self.max_time_pondering == other.max_time_pondering &&

    self.lag_buffer == other.lag_buffer &&

    self.search_factor_after_one_pass == other.search_factor_after_one_pass &&
    self.search_factor_after_two_pass == other.search_factor_after_two_pass &&

    self.tree_reuse_carry_over_time_factor == other.tree_reuse_carry_over_time_factor &&
    self.overallocate_time_factor == other.overallocate_time_factor &&
    self.midgame_time_factor == other.midgame_time_factor &&
    self.midgame_turn_peak_time == other.midgame_turn_peak_time &&
    self.endgame_turn_time_decay == other.endgame_turn_time_decay &&
    self.obvious_moves_time_factor == other.obvious_moves_time_factor &&
    self.obvious_moves_policy_entropy_tolerance == other.obvious_moves_policy_entropy_tolerance &&
    self.obvious_moves_policy_surprise_tolerance == other.obvious_moves_policy_surprise_tolerance &&

    self.futile_visits_threshold == other.futile_visits_threshold &&

    self.human_sl_profile == other.human_sl_profile &&
    self.human_sl_cpuct_exploration == other.human_sl_cpuct_exploration &&
    self.human_sl_cpuct_permanent == other.human_sl_cpuct_permanent &&
    self.human_sl_root_explore_prob_weightless == other.human_sl_root_explore_prob_weightless &&
    self.human_sl_root_explore_prob_weightful == other.human_sl_root_explore_prob_weightful &&
    self.human_sl_pla_explore_prob_weightless == other.human_sl_pla_explore_prob_weightless &&
    self.human_sl_pla_explore_prob_weightful == other.human_sl_pla_explore_prob_weightful &&
    self.human_sl_opp_explore_prob_weightless == other.human_sl_opp_explore_prob_weightless &&
    self.human_sl_opp_explore_prob_weightful == other.human_sl_opp_explore_prob_weightful &&

    self.human_sl_chosen_move_prop == other.human_sl_chosen_move_prop &&
    self.human_sl_chosen_move_ignore_pass == other.human_sl_chosen_move_ignore_pass &&
    self.human_sl_chosen_move_pikl_lambda == other.human_sl_chosen_move_pikl_lambda
  }
}

impl SearchParams {
  /// upstream: SearchParams::forTestsV1 (searchparams.cpp)
  //Params to use for testing, with some more recent values representative of more real use (as of Jan 2019)
  pub fn for_tests_v1() -> SearchParams {
    let mut params: SearchParams = SearchParams::new();
    params.static_score_utility_factor = 0.1;
    params.dynamic_score_utility_factor = 0.3;
    params.dynamic_score_center_zero_weight = 0.2;
    params.dynamic_score_center_scale = 0.75;
    params.cpuct_exploration = 0.9;
    params.cpuct_exploration_log = 0.4;
    params.root_fpu_reduction_max = 0.1;
    params.root_policy_temperature_early = 1.2;
    params.root_policy_temperature = 1.1;
    params.use_lcb_for_selection = true;
    params.lcb_stdevs = 5.0;
    params.min_visit_prop_for_lcb = 0.15;
    params.root_ending_bonus_points = 0.5;
    params.root_prune_useless_moves = true;
    params.conservative_pass = true;
    params.use_non_buggy_lcb = true;
    params
  }

  /// upstream: SearchParams::forTestsV2 (searchparams.cpp)
  //Params to use for testing, with some more recent values representative of more real use (as of Mar 2022)
  pub fn for_tests_v2() -> SearchParams {
    let mut params: SearchParams = SearchParams::new();
    params.static_score_utility_factor = 0.1;
    params.dynamic_score_utility_factor = 0.3;
    params.dynamic_score_center_zero_weight = 0.2;
    params.dynamic_score_center_scale = 0.75;
    params.cpuct_exploration = 0.9;
    params.cpuct_exploration_log = 0.4;
    params.root_fpu_reduction_max = 0.1;
    params.root_policy_temperature_early = 1.2;
    params.root_policy_temperature = 1.1;
    params.use_lcb_for_selection = true;
    params.lcb_stdevs = 5.0;
    params.min_visit_prop_for_lcb = 0.15;
    params.root_ending_bonus_points = 0.5;
    params.root_prune_useless_moves = true;
    params.conservative_pass = true;
    params.use_non_buggy_lcb = true;
    params.use_graph_search = true;
    params.fpu_parent_weight_by_visited_policy = true;
    params.value_weight_exponent = 0.25;
    params.use_noise_pruning = true;
    params.use_uncertainty = true;
    params.uncertainty_coeff = 0.25;
    params.uncertainty_exponent = 1.0;
    params.uncertainty_max_weight = 8.0;
    params.cpuct_utility_stdev_prior = 0.40;
    params.cpuct_utility_stdev_prior_weight = 2.0;
    params.cpuct_utility_stdev_scale = 0.85;
    params.fill_dame_before_pass = true;
    params.subtree_value_bias_factor = 0.45;
    params.subtree_value_bias_free_prop = 0.8;
    params.subtree_value_bias_weight_exponent = 0.85;
    params
  }

  /// upstream: SearchParams::basicDecentParams (searchparams.cpp)
  pub fn basic_decent_params() -> SearchParams {
    let mut params: SearchParams = SearchParams::new();
    params.static_score_utility_factor = 0.1;
    params.dynamic_score_utility_factor = 0.3;
    params.dynamic_score_center_zero_weight = 0.2;
    params.dynamic_score_center_scale = 0.75;
    params.cpuct_exploration = 1.0;
    params.cpuct_exploration_log = 0.45;
    params.root_fpu_reduction_max = 0.1;
    params.root_policy_temperature_early = 1.0;
    params.root_policy_temperature = 1.0;
    params.use_lcb_for_selection = true;
    params.lcb_stdevs = 5.0;
    params.min_visit_prop_for_lcb = 0.20;
    params.root_ending_bonus_points = 0.5;
    params.root_prune_useless_moves = true;
    params.conservative_pass = true;
    params.enable_passing_hacks = true;
    params.use_non_buggy_lcb = true;
    params.use_graph_search = true;
    params.fpu_parent_weight_by_visited_policy = true;
    params.value_weight_exponent = 0.25;
    params.use_noise_pruning = true;
    params.use_uncertainty = true;
    params.uncertainty_coeff = 0.25;
    params.uncertainty_exponent = 1.0;
    params.uncertainty_max_weight = 8.0;
    params.cpuct_utility_stdev_prior = 0.40;
    params.cpuct_utility_stdev_prior_weight = 2.0;
    params.cpuct_utility_stdev_scale = 0.85;
    params.fill_dame_before_pass = true;
    params.subtree_value_bias_factor = 0.45;
    params.subtree_value_bias_free_prop = 0.8;
    params.subtree_value_bias_weight_exponent = 0.85;
    params
  }

  /// upstream: SearchParams::failIfParamsDifferOnUnchangeableParameter (searchparams.cpp)
  pub fn fail_if_params_differ_on_unchangeable_parameter(initial: &SearchParams, dynamic: &SearchParams) -> Result<(), KataError> {
    if dynamic.node_table_shards_power_of_two != initial.node_table_shards_power_of_two {
      return Err(KataError::StringError(String::from("Cannot change nodeTableShardsPowerOfTwo after initialization")));
    }

    // Analysis engine shares eval cache across multiple analysis threads, so changing/overriding it is awkward.
    if dynamic.use_eval_cache != initial.use_eval_cache {
      return Err(KataError::StringError(String::from("Cannot change useEvalCache after initialization")));
    }
    if dynamic.eval_cache_min_visits != initial.eval_cache_min_visits {
      return Err(KataError::StringError(String::from("Cannot change evalCacheMinVisits after initialization")));
    }
    Ok(())
  }

  /// upstream: SearchParams::changeableParametersToJson (searchparams.cpp)
  pub fn changeable_parameters_to_json(&self) -> Value {
    let mut ret: Map<String, Value> = Map::new();
    ret.insert(String::from("winLossUtilityFactor"), Value::from(self.win_loss_utility_factor));
    ret.insert(String::from("staticScoreUtilityFactor"), Value::from(self.static_score_utility_factor));
    ret.insert(String::from("dynamicScoreUtilityFactor"), Value::from(self.dynamic_score_utility_factor));
    ret.insert(String::from("dynamicScoreCenterZeroWeight"), Value::from(self.dynamic_score_center_zero_weight));
    ret.insert(String::from("dynamicScoreCenterScale"), Value::from(self.dynamic_score_center_scale));
    ret.insert(String::from("noResultUtilityForWhite"), Value::from(self.no_result_utility_for_white));
    ret.insert(String::from("drawEquivalentWinsForWhite"), Value::from(self.draw_equivalent_wins_for_white));

    ret.insert(String::from("cpuctExploration"), Value::from(self.cpuct_exploration));
    ret.insert(String::from("cpuctExplorationLog"), Value::from(self.cpuct_exploration_log));
    ret.insert(String::from("cpuctExplorationBase"), Value::from(self.cpuct_exploration_base));

    ret.insert(String::from("cpuctUtilityStdevPrior"), Value::from(self.cpuct_utility_stdev_prior));
    ret.insert(String::from("cpuctUtilityStdevPriorWeight"), Value::from(self.cpuct_utility_stdev_prior_weight));
    ret.insert(String::from("cpuctUtilityStdevScale"), Value::from(self.cpuct_utility_stdev_scale));

    ret.insert(String::from("fpuReductionMax"), Value::from(self.fpu_reduction_max));
    ret.insert(String::from("fpuLossProp"), Value::from(self.fpu_loss_prop));

    ret.insert(String::from("fpuParentWeightByVisitedPolicy"), Value::from(self.fpu_parent_weight_by_visited_policy));
    ret.insert(String::from("fpuParentWeightByVisitedPolicyPow"), Value::from(self.fpu_parent_weight_by_visited_policy_pow));
    ret.insert(String::from("fpuParentWeight"), Value::from(self.fpu_parent_weight));

    ret.insert(String::from("policyOptimism"), Value::from(self.policy_optimism));

    ret.insert(String::from("valueWeightExponent"), Value::from(self.value_weight_exponent));
    ret.insert(String::from("useNoisePruning"), Value::from(self.use_noise_pruning));
    ret.insert(String::from("noisePruneUtilityScale"), Value::from(self.noise_prune_utility_scale));
    ret.insert(String::from("noisePruningCap"), Value::from(self.noise_pruning_cap));

    ret.insert(String::from("useUncertainty"), Value::from(self.use_uncertainty));
    ret.insert(String::from("uncertaintyCoeff"), Value::from(self.uncertainty_coeff));
    ret.insert(String::from("uncertaintyExponent"), Value::from(self.uncertainty_exponent));
    ret.insert(String::from("uncertaintyMaxWeight"), Value::from(self.uncertainty_max_weight));

    ret.insert(String::from("useGraphSearch"), Value::from(self.use_graph_search));
    ret.insert(String::from("graphSearchRepBound"), Value::from(self.graph_search_rep_bound));
    ret.insert(String::from("graphSearchCatchUpLeakProb"), Value::from(self.graph_search_catch_up_leak_prob));

    ret.insert(String::from("rootNoiseEnabled"), Value::from(self.root_noise_enabled));
    ret.insert(String::from("rootDirichletNoiseTotalConcentration"), Value::from(self.root_dirichlet_noise_total_concentration));
    ret.insert(String::from("rootDirichletNoiseWeight"), Value::from(self.root_dirichlet_noise_weight));

    ret.insert(String::from("rootPolicyTemperature"), Value::from(self.root_policy_temperature));
    ret.insert(String::from("rootPolicyTemperatureEarly"), Value::from(self.root_policy_temperature_early));
    ret.insert(String::from("rootFpuReductionMax"), Value::from(self.root_fpu_reduction_max));
    ret.insert(String::from("rootFpuLossProp"), Value::from(self.root_fpu_loss_prop));
    ret.insert(String::from("rootNumSymmetriesToSample"), Value::from(self.root_num_symmetries_to_sample));
    ret.insert(String::from("rootSymmetryPruning"), Value::from(self.root_symmetry_pruning));
    ret.insert(String::from("rootDesiredPerChildVisitsCoeff"), Value::from(self.root_desired_per_child_visits_coeff));

    ret.insert(String::from("rootPolicyOptimism"), Value::from(self.root_policy_optimism));

    ret.insert(String::from("chosenMoveTemperature"), Value::from(self.chosen_move_temperature));
    ret.insert(String::from("chosenMoveTemperatureEarly"), Value::from(self.chosen_move_temperature_early));
    ret.insert(String::from("chosenMoveTemperatureHalflife"), Value::from(self.chosen_move_temperature_halflife));
    ret.insert(String::from("chosenMoveTemperatureOnlyBelowProb"), Value::from(self.chosen_move_temperature_only_below_prob));

    ret.insert(String::from("chosenMoveSubtract"), Value::from(self.chosen_move_subtract));
    ret.insert(String::from("chosenMovePrune"), Value::from(self.chosen_move_prune));
    ret.insert(String::from("useLcbForSelection"), Value::from(self.use_lcb_for_selection));

    ret.insert(String::from("lcbStdevs"), Value::from(self.lcb_stdevs));
    ret.insert(String::from("minVisitPropForLCB"), Value::from(self.min_visit_prop_for_lcb));
    ret.insert(String::from("useNonBuggyLcb"), Value::from(self.use_non_buggy_lcb));
    ret.insert(String::from("rootEndingBonusPoints"), Value::from(self.root_ending_bonus_points));

    ret.insert(String::from("rootPruneUselessMoves"), Value::from(self.root_prune_useless_moves));
    ret.insert(String::from("conservativePass"), Value::from(self.conservative_pass));
    ret.insert(String::from("fillDameBeforePass"), Value::from(self.fill_dame_before_pass));
    // Unused
    // ret["avoidMYTDaggerHackPla"] = PlayerIO::playerToStringShort(avoidMYTDaggerHackPla);
    ret.insert(String::from("wideRootNoise"), Value::from(self.wide_root_noise));
    ret.insert(String::from("enablePassingHacks"), Value::from(self.enable_passing_hacks));
    ret.insert(String::from("enableMorePassingHacks"), Value::from(self.enable_more_passing_hacks));
    ret.insert(String::from("alwaysComputePassAliveUnderSuicideRules"), Value::from(self.always_compute_pass_alive_under_suicide_rules.to_string()));
    ret.insert(String::from("excludeTerritoryAdjacentToAtari"), Value::from(self.exclude_territory_adjacent_to_atari.to_string()));

    // Special handling in GTP
    ret.insert(String::from("playoutDoublingAdvantage"), Value::from(self.playout_doubling_advantage));
    ret.insert(String::from("playoutDoublingAdvantagePla"), Value::from(PlayerIO::player_to_string_short(self.playout_doubling_advantage_pla)));

    // Special handling in GTP
    // ret["avoidRepeatedPatternUtility"] = avoidRepeatedPatternUtility;

    ret.insert(String::from("nnPolicyTemperature"), Value::from(self.nn_policy_temperature as f64));
    // Special handling in GTP
    // ret["antiMirror"] = antiMirror;

    ret.insert(String::from("ignorePreRootHistory"), Value::from(self.ignore_pre_root_history));
    ret.insert(String::from("ignoreAllHistory"), Value::from(self.ignore_all_history));

    ret.insert(String::from("subtreeValueBiasFactor"), Value::from(self.subtree_value_bias_factor));
    ret.insert(String::from("subtreeValueBiasTableNumShards"), Value::from(self.subtree_value_bias_table_num_shards));
    ret.insert(String::from("subtreeValueBiasFreeProp"), Value::from(self.subtree_value_bias_free_prop));
    ret.insert(String::from("subtreeValueBiasWeightExponent"), Value::from(self.subtree_value_bias_weight_exponent));

    // ret["useEvalCache"] = useEvalCache;
    // ret["evalCacheMinVisits"] = evalCacheMinVisits;

    // ret["nodeTableShardsPowerOfTwo"] = nodeTableShardsPowerOfTwo;
    ret.insert(String::from("numVirtualLossesPerThread"), Value::from(self.num_virtual_losses_per_thread));

    ret.insert(String::from("numSearchThreads"), Value::from(self.num_threads)); // NOTE: different name since that's how setup.cpp loads it
    ret.insert(String::from("minPlayoutsPerThread"), Value::from(self.min_playouts_per_thread));
    ret.insert(String::from("maxVisits"), Value::from(self.max_visits));
    ret.insert(String::from("maxPlayouts"), Value::from(self.max_playouts));
    ret.insert(String::from("maxTime"), Value::from(self.max_time));

    ret.insert(String::from("maxVisitsPondering"), Value::from(self.max_visits_pondering));
    ret.insert(String::from("maxPlayoutsPondering"), Value::from(self.max_playouts_pondering));
    ret.insert(String::from("maxTimePondering"), Value::from(self.max_time_pondering));

    ret.insert(String::from("lagBuffer"), Value::from(self.lag_buffer));

    ret.insert(String::from("searchFactorAfterOnePass"), Value::from(self.search_factor_after_one_pass));
    ret.insert(String::from("searchFactorAfterTwoPass"), Value::from(self.search_factor_after_two_pass));

    ret.insert(String::from("treeReuseCarryOverTimeFactor"), Value::from(self.tree_reuse_carry_over_time_factor));
    ret.insert(String::from("overallocateTimeFactor"), Value::from(self.overallocate_time_factor));
    ret.insert(String::from("midgameTimeFactor"), Value::from(self.midgame_time_factor));
    ret.insert(String::from("midgameTurnPeakTime"), Value::from(self.midgame_turn_peak_time));
    ret.insert(String::from("endgameTurnTimeDecay"), Value::from(self.endgame_turn_time_decay));
    ret.insert(String::from("obviousMovesTimeFactor"), Value::from(self.obvious_moves_time_factor));
    ret.insert(String::from("obviousMovesPolicyEntropyTolerance"), Value::from(self.obvious_moves_policy_entropy_tolerance));
    ret.insert(String::from("obviousMovesPolicySurpriseTolerance"), Value::from(self.obvious_moves_policy_surprise_tolerance));

    ret.insert(String::from("futileVisitsThreshold"), Value::from(self.futile_visits_threshold));

    ret.insert(String::from("humanSLCpuctExploration"), Value::from(self.human_sl_cpuct_exploration));
    ret.insert(String::from("humanSLCpuctPermanent"), Value::from(self.human_sl_cpuct_permanent));

    ret.insert(String::from("humanSLRootExploreProbWeightless"), Value::from(self.human_sl_root_explore_prob_weightless));
    ret.insert(String::from("humanSLRootExploreProbWeightful"), Value::from(self.human_sl_root_explore_prob_weightful));
    ret.insert(String::from("humanSLPlaExploreProbWeightless"), Value::from(self.human_sl_pla_explore_prob_weightless));
    ret.insert(String::from("humanSLPlaExploreProbWeightful"), Value::from(self.human_sl_pla_explore_prob_weightful));
    ret.insert(String::from("humanSLOppExploreProbWeightless"), Value::from(self.human_sl_opp_explore_prob_weightless));
    ret.insert(String::from("humanSLOppExploreProbWeightful"), Value::from(self.human_sl_opp_explore_prob_weightful));

    ret.insert(String::from("humanSLChosenMoveProp"), Value::from(self.human_sl_chosen_move_prop));
    ret.insert(String::from("humanSLChosenMoveIgnorePass"), Value::from(self.human_sl_chosen_move_ignore_pass));
    ret.insert(String::from("humanSLChosenMovePiklLambda"), Value::from(self.human_sl_chosen_move_pikl_lambda));

    Value::Object(ret)
  }

  /// upstream: SearchParams::getHash (searchparams.cpp)
  // Hash of all parameters, such that params that compare unequal hash differently (with high probability).
  pub fn get_hash(&self) -> Hash128 {
    // Build on changeableParametersToJson, but fold in the parameters it deliberately omits so that this
    // hash reflects *all* parameters (any of which can affect cached search results). If a new parameter is
    // added to operator== it should also be reflected here - either it will already flow through
    // changeableParametersToJson, or it must be added to the supplement below.
    let mut ret: Value = self.changeable_parameters_to_json();
    let obj = ret.as_object_mut().expect("changeableParametersToJson returns an object");

    // Parameters omitted from changeableParametersToJson
    obj.insert(String::from("avoidMYTDaggerHackPla"), Value::from(self.avoid_myt_dagger_hack_pla as i32));
    obj.insert(String::from("avoidRepeatedPatternUtility"), Value::from(self.avoid_repeated_pattern_utility));
    obj.insert(String::from("antiMirror"), Value::from(self.anti_mirror));
    obj.insert(String::from("useEvalCache"), Value::from(self.use_eval_cache));
    obj.insert(String::from("evalCacheMinVisits"), Value::from(self.eval_cache_min_visits));
    obj.insert(String::from("nodeTableShardsPowerOfTwo"), Value::from(self.node_table_shards_power_of_two));

    // humanSLProfile's own getHash asserts on an uninitialized profile,
    // so only call it when initialized (the common no-human-model case leaves it uninitialized).
    obj.insert(String::from("humanSLProfileInitialized"), Value::from(self.human_sl_profile.initialized));
    if self.human_sl_profile.initialized {
      //Use a fixed player. We only need a stable value that distinguishes distinct profiles.
      let profile_hash: Hash128 = self.human_sl_profile.get_hash(P_BLACK);
      obj.insert(String::from("humanSLProfileHash0"), Value::from(profile_hash.hash0));
      obj.insert(String::from("humanSLProfileHash1"), Value::from(profile_hash.hash1));
    }

    let dumped: String = crate::port::nlohmann::dump(&ret);
    let hash: [u64; 4] = crate::port::digest::sha2_get256_u64(&dumped);
    Hash128::new(hash[0], hash[1])
  }

  /// upstream: SearchParams::printParams (searchparams.cpp)
  /// PORT NOTE: `PRINTPARAM(X)` (`out << #X << ": " << X << std::endl;`) is expanded in place:
  /// doubles/floats through cpp_ostream_f64 (default `ostream` precision), bools as `0`/`1`
  /// (no boolalpha), integers as is.
  /// PORT NOTE: upstream writes `avoidMYTDaggerHackPla` and `playoutDoublingAdvantagePla` to
  /// `std::cout`, not to `out` (apparent upstream bug, kept): `cout` becomes the `tracing` macros.
  pub fn print_params(&self, out: &mut dyn std::fmt::Write) {



    writeln!(out, "winLossUtilityFactor: {}", cpp_ostream_f64(self.win_loss_utility_factor)).unwrap();
    writeln!(out, "staticScoreUtilityFactor: {}", cpp_ostream_f64(self.static_score_utility_factor)).unwrap();
    writeln!(out, "dynamicScoreUtilityFactor: {}", cpp_ostream_f64(self.dynamic_score_utility_factor)).unwrap();
    writeln!(out, "dynamicScoreCenterZeroWeight: {}", cpp_ostream_f64(self.dynamic_score_center_zero_weight)).unwrap();
    writeln!(out, "dynamicScoreCenterScale: {}", cpp_ostream_f64(self.dynamic_score_center_scale)).unwrap();
    writeln!(out, "noResultUtilityForWhite: {}", cpp_ostream_f64(self.no_result_utility_for_white)).unwrap();
    writeln!(out, "drawEquivalentWinsForWhite: {}", cpp_ostream_f64(self.draw_equivalent_wins_for_white)).unwrap();

    writeln!(out, "cpuctExploration: {}", cpp_ostream_f64(self.cpuct_exploration)).unwrap();
    writeln!(out, "cpuctExplorationLog: {}", cpp_ostream_f64(self.cpuct_exploration_log)).unwrap();
    writeln!(out, "cpuctExplorationBase: {}", cpp_ostream_f64(self.cpuct_exploration_base)).unwrap();

    writeln!(out, "cpuctUtilityStdevPrior: {}", cpp_ostream_f64(self.cpuct_utility_stdev_prior)).unwrap();
    writeln!(out, "cpuctUtilityStdevPriorWeight: {}", cpp_ostream_f64(self.cpuct_utility_stdev_prior_weight)).unwrap();
    writeln!(out, "cpuctUtilityStdevScale: {}", cpp_ostream_f64(self.cpuct_utility_stdev_scale)).unwrap();

    writeln!(out, "fpuReductionMax: {}", cpp_ostream_f64(self.fpu_reduction_max)).unwrap();
    writeln!(out, "fpuLossProp: {}", cpp_ostream_f64(self.fpu_loss_prop)).unwrap();

    writeln!(out, "fpuParentWeightByVisitedPolicy: {}", self.fpu_parent_weight_by_visited_policy as i32).unwrap();
    writeln!(out, "fpuParentWeightByVisitedPolicyPow: {}", cpp_ostream_f64(self.fpu_parent_weight_by_visited_policy_pow)).unwrap();
    writeln!(out, "fpuParentWeight: {}", cpp_ostream_f64(self.fpu_parent_weight)).unwrap();

    writeln!(out, "policyOptimism: {}", cpp_ostream_f64(self.policy_optimism)).unwrap();

    writeln!(out, "valueWeightExponent: {}", cpp_ostream_f64(self.value_weight_exponent)).unwrap();
    writeln!(out, "useNoisePruning: {}", self.use_noise_pruning as i32).unwrap();
    writeln!(out, "noisePruneUtilityScale: {}", cpp_ostream_f64(self.noise_prune_utility_scale)).unwrap();
    writeln!(out, "noisePruningCap: {}", cpp_ostream_f64(self.noise_pruning_cap)).unwrap();


    writeln!(out, "useUncertainty: {}", self.use_uncertainty as i32).unwrap();
    writeln!(out, "uncertaintyCoeff: {}", cpp_ostream_f64(self.uncertainty_coeff)).unwrap();
    writeln!(out, "uncertaintyExponent: {}", cpp_ostream_f64(self.uncertainty_exponent)).unwrap();
    writeln!(out, "uncertaintyMaxWeight: {}", cpp_ostream_f64(self.uncertainty_max_weight)).unwrap();


    writeln!(out, "useGraphSearch: {}", self.use_graph_search as i32).unwrap();
    writeln!(out, "graphSearchRepBound: {}", self.graph_search_rep_bound).unwrap();
    writeln!(out, "graphSearchCatchUpLeakProb: {}", cpp_ostream_f64(self.graph_search_catch_up_leak_prob)).unwrap();



    writeln!(out, "rootNoiseEnabled: {}", self.root_noise_enabled as i32).unwrap();
    writeln!(out, "rootDirichletNoiseTotalConcentration: {}", cpp_ostream_f64(self.root_dirichlet_noise_total_concentration)).unwrap();
    writeln!(out, "rootDirichletNoiseWeight: {}", cpp_ostream_f64(self.root_dirichlet_noise_weight)).unwrap();

    writeln!(out, "rootPolicyTemperature: {}", cpp_ostream_f64(self.root_policy_temperature)).unwrap();
    writeln!(out, "rootPolicyTemperatureEarly: {}", cpp_ostream_f64(self.root_policy_temperature_early)).unwrap();
    writeln!(out, "rootFpuReductionMax: {}", cpp_ostream_f64(self.root_fpu_reduction_max)).unwrap();
    writeln!(out, "rootFpuLossProp: {}", cpp_ostream_f64(self.root_fpu_loss_prop)).unwrap();
    writeln!(out, "rootNumSymmetriesToSample: {}", self.root_num_symmetries_to_sample).unwrap();
    writeln!(out, "rootSymmetryPruning: {}", self.root_symmetry_pruning as i32).unwrap();

    writeln!(out, "rootDesiredPerChildVisitsCoeff: {}", cpp_ostream_f64(self.root_desired_per_child_visits_coeff)).unwrap();

    writeln!(out, "rootPolicyOptimism: {}", cpp_ostream_f64(self.root_policy_optimism)).unwrap();

    writeln!(out, "chosenMoveTemperature: {}", cpp_ostream_f64(self.chosen_move_temperature)).unwrap();
    writeln!(out, "chosenMoveTemperatureEarly: {}", cpp_ostream_f64(self.chosen_move_temperature_early)).unwrap();
    writeln!(out, "chosenMoveTemperatureHalflife: {}", cpp_ostream_f64(self.chosen_move_temperature_halflife)).unwrap();
    writeln!(out, "chosenMoveTemperatureOnlyBelowProb: {}", cpp_ostream_f64(self.chosen_move_temperature_only_below_prob)).unwrap();
    writeln!(out, "chosenMoveSubtract: {}", cpp_ostream_f64(self.chosen_move_subtract)).unwrap();
    writeln!(out, "chosenMovePrune: {}", cpp_ostream_f64(self.chosen_move_prune)).unwrap();

    writeln!(out, "useLcbForSelection: {}", self.use_lcb_for_selection as i32).unwrap();
    writeln!(out, "lcbStdevs: {}", cpp_ostream_f64(self.lcb_stdevs)).unwrap();
    writeln!(out, "minVisitPropForLCB: {}", cpp_ostream_f64(self.min_visit_prop_for_lcb)).unwrap();
    writeln!(out, "useNonBuggyLcb: {}", self.use_non_buggy_lcb as i32).unwrap();


    writeln!(out, "rootEndingBonusPoints: {}", cpp_ostream_f64(self.root_ending_bonus_points)).unwrap();
    writeln!(out, "rootPruneUselessMoves: {}", self.root_prune_useless_moves as i32).unwrap();
    writeln!(out, "conservativePass: {}", self.conservative_pass as i32).unwrap();
    writeln!(out, "fillDameBeforePass: {}", self.fill_dame_before_pass as i32).unwrap();
    tracing::info!("avoidMYTDaggerHackPla: {}", self.avoid_myt_dagger_hack_pla as i32);
    writeln!(out, "wideRootNoise: {}", cpp_ostream_f64(self.wide_root_noise)).unwrap();
    writeln!(out, "enablePassingHacks: {}", self.enable_passing_hacks as i32).unwrap();
    writeln!(out, "enableMorePassingHacks: {}", self.enable_more_passing_hacks as i32).unwrap();
    writeln!(out, "alwaysComputePassAliveUnderSuicideRules: {}", self.always_compute_pass_alive_under_suicide_rules.to_string()).unwrap();
    writeln!(out, "excludeTerritoryAdjacentToAtari: {}", self.exclude_territory_adjacent_to_atari.to_string()).unwrap();

    writeln!(out, "playoutDoublingAdvantage: {}", cpp_ostream_f64(self.playout_doubling_advantage)).unwrap();
    tracing::info!("playoutDoublingAdvantagePla: {}", self.playout_doubling_advantage_pla as i32);

    writeln!(out, "avoidRepeatedPatternUtility: {}", cpp_ostream_f64(self.avoid_repeated_pattern_utility)).unwrap();

    writeln!(out, "nnPolicyTemperature: {}", cpp_ostream_f64(self.nn_policy_temperature as f64)).unwrap();
    writeln!(out, "antiMirror: {}", self.anti_mirror as i32).unwrap();

    writeln!(out, "ignorePreRootHistory: {}", self.ignore_pre_root_history as i32).unwrap();
    writeln!(out, "ignoreAllHistory: {}", self.ignore_all_history as i32).unwrap();

    writeln!(out, "subtreeValueBiasFactor: {}", cpp_ostream_f64(self.subtree_value_bias_factor)).unwrap();
    writeln!(out, "subtreeValueBiasTableNumShards: {}", self.subtree_value_bias_table_num_shards).unwrap();
    writeln!(out, "subtreeValueBiasFreeProp: {}", cpp_ostream_f64(self.subtree_value_bias_free_prop)).unwrap();
    writeln!(out, "subtreeValueBiasWeightExponent: {}", cpp_ostream_f64(self.subtree_value_bias_weight_exponent)).unwrap();

    writeln!(out, "useEvalCache: {}", self.use_eval_cache as i32).unwrap();
    writeln!(out, "evalCacheMinVisits: {}", self.eval_cache_min_visits).unwrap();

    writeln!(out, "nodeTableShardsPowerOfTwo: {}", self.node_table_shards_power_of_two).unwrap();
    writeln!(out, "numVirtualLossesPerThread: {}", cpp_ostream_f64(self.num_virtual_losses_per_thread)).unwrap();


    writeln!(out, "numThreads: {}", self.num_threads).unwrap();
    writeln!(out, "minPlayoutsPerThread: {}", cpp_ostream_f64(self.min_playouts_per_thread)).unwrap();
    writeln!(out, "maxVisits: {}", self.max_visits).unwrap();
    writeln!(out, "maxPlayouts: {}", self.max_playouts).unwrap();
    writeln!(out, "maxTime: {}", cpp_ostream_f64(self.max_time)).unwrap();


    writeln!(out, "maxVisitsPondering: {}", self.max_visits_pondering).unwrap();
    writeln!(out, "maxPlayoutsPondering: {}", self.max_playouts_pondering).unwrap();
    writeln!(out, "maxTimePondering: {}", cpp_ostream_f64(self.max_time_pondering)).unwrap();


    writeln!(out, "lagBuffer: {}", cpp_ostream_f64(self.lag_buffer)).unwrap();


    writeln!(out, "searchFactorAfterOnePass: {}", cpp_ostream_f64(self.search_factor_after_one_pass)).unwrap();
    writeln!(out, "searchFactorAfterTwoPass: {}", cpp_ostream_f64(self.search_factor_after_two_pass)).unwrap();


    writeln!(out, "treeReuseCarryOverTimeFactor: {}", cpp_ostream_f64(self.tree_reuse_carry_over_time_factor)).unwrap();
    writeln!(out, "overallocateTimeFactor: {}", cpp_ostream_f64(self.overallocate_time_factor)).unwrap();
    writeln!(out, "midgameTimeFactor: {}", cpp_ostream_f64(self.midgame_time_factor)).unwrap();
    writeln!(out, "midgameTurnPeakTime: {}", cpp_ostream_f64(self.midgame_turn_peak_time)).unwrap();
    writeln!(out, "endgameTurnTimeDecay: {}", cpp_ostream_f64(self.endgame_turn_time_decay)).unwrap();
    writeln!(out, "obviousMovesTimeFactor: {}", cpp_ostream_f64(self.obvious_moves_time_factor)).unwrap();
    writeln!(out, "obviousMovesPolicyEntropyTolerance: {}", cpp_ostream_f64(self.obvious_moves_policy_entropy_tolerance)).unwrap();
    writeln!(out, "obviousMovesPolicySurpriseTolerance: {}", cpp_ostream_f64(self.obvious_moves_policy_surprise_tolerance)).unwrap();

    writeln!(out, "futileVisitsThreshold: {}", cpp_ostream_f64(self.futile_visits_threshold)).unwrap();


    writeln!(out, "humanSLCpuctExploration: {}", cpp_ostream_f64(self.human_sl_cpuct_exploration)).unwrap();
    writeln!(out, "humanSLCpuctPermanent: {}", cpp_ostream_f64(self.human_sl_cpuct_permanent)).unwrap();
    writeln!(out, "humanSLRootExploreProbWeightless: {}", cpp_ostream_f64(self.human_sl_root_explore_prob_weightless)).unwrap();
    writeln!(out, "humanSLRootExploreProbWeightful: {}", cpp_ostream_f64(self.human_sl_root_explore_prob_weightful)).unwrap();
    writeln!(out, "humanSLPlaExploreProbWeightless: {}", cpp_ostream_f64(self.human_sl_pla_explore_prob_weightless)).unwrap();
    writeln!(out, "humanSLPlaExploreProbWeightful: {}", cpp_ostream_f64(self.human_sl_pla_explore_prob_weightful)).unwrap();
    writeln!(out, "humanSLOppExploreProbWeightless: {}", cpp_ostream_f64(self.human_sl_opp_explore_prob_weightless)).unwrap();
    writeln!(out, "humanSLOppExploreProbWeightful: {}", cpp_ostream_f64(self.human_sl_opp_explore_prob_weightful)).unwrap();
    writeln!(out, "humanSLChosenMoveProp: {}", cpp_ostream_f64(self.human_sl_chosen_move_prop)).unwrap();
    writeln!(out, "humanSLChosenMoveIgnorePass: {}", self.human_sl_chosen_move_ignore_pass as i32).unwrap();
    writeln!(out, "humanSLChosenMovePiklLambda: {}", cpp_ostream_f64(self.human_sl_chosen_move_pikl_lambda)).unwrap();

  }
}
