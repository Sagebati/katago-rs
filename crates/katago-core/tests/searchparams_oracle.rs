//! search/searchparams against the unmodified upstream SearchParams, and against the real binary.
//! 1. oracle/searchparams.cpp (links the upstream objects): every field of the constructor defaults,
//!    forTestsV1/V2 and basicDecentParams as bit patterns, changeableParametersToJson().dump(),
//!    printParams output (stream and the two lines upstream writes to cout), operator==/!= over presets
//!    and over every single-field mutation, failIfParamsDifferOnUnchangeableParameter.
//! 2. oracle/searchparams_gtp_kata_get_params.txt: the reply of the real `build/eigen/katago gtp` to
//!    `kata-get-params`, against a SearchParams built the way program/setup.cpp's GTP path builds it
//!    (test-local subset, below). Not a port of upstream code.
//!
//! Fixture 2 was produced with (from the repo root, scratch dirs for logs and home data):
//!   build/eigen/katago gtp -config upstream/KataGo/cpp/configs/gtp_example.cfg \
//!     -model <g170-b6c96.bin.gz> -override-config "logDir=<scratch>,homeDataDir=<scratch>,numSearchThreads=2,maxVisits=500,nnCacheSizePowerOfTwo=16,nnMutexPoolSizePowerOfTwo=12"
//!   with `kata-get-params` then `quit` on stdin; the stdout line after "= " is the fixture.
#![allow(clippy::bool_comparison, clippy::needless_late_init, clippy::field_reassign_with_default)]

#[macro_use]
mod common;

use std::cell::RefCell;
use std::sync::OnceLock;

use common::Cases;
use common::compare_lines;
use common::fixture;
use katago_core::game::board::C_EMPTY;
use katago_core::port::nlohmann;
use katago_core::search::searchparams::SearchParams;

//Captures `log` records per thread, for the two printParams lines that upstream writes to std::cout.
thread_local! {
  static CAPTURED: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
}

struct CaptureLogger;

impl log::Log for CaptureLogger {
  fn enabled(&self, _metadata: &log::Metadata) -> bool {
    true
  }
  fn log(&self, record: &log::Record) {
    CAPTURED.with(|c| c.borrow_mut().push(format!("{}", record.args())));
  }
  fn flush(&self) {}
}

static LOGGER: CaptureLogger = CaptureLogger;

fn install_logger() {
  static ONCE: OnceLock<()> = OnceLock::new();
  ONCE.get_or_init(|| {
    log::set_logger(&LOGGER).expect("no other logger in this test binary");
    log::set_max_level(log::LevelFilter::Info);
  });
}

fn emit_lines(tag: &str, s: &str, out: &mut String) {
  let mut i = 0;
  while i < s.len() {
    let j = match s[i..].find('\n') {
      Some(k) => i + k,
      None => s.len(),
    };
    out.push_str(&format!("{} {}\n", tag, &s[i..j]));
    i = j + 1;
  }
}

/// `dump()` in oracle/searchparams.cpp
fn dump(name: &str, p: &SearchParams, out: &mut String) {
  out.push_str(&format!("S {}\n", name));
  out.push_str(&format!("F winLossUtilityFactor {:016X}\n", p.win_loss_utility_factor.to_bits()));
  out.push_str(&format!("F staticScoreUtilityFactor {:016X}\n", p.static_score_utility_factor.to_bits()));
  out.push_str(&format!("F dynamicScoreUtilityFactor {:016X}\n", p.dynamic_score_utility_factor.to_bits()));
  out.push_str(&format!("F dynamicScoreCenterZeroWeight {:016X}\n", p.dynamic_score_center_zero_weight.to_bits()));
  out.push_str(&format!("F dynamicScoreCenterScale {:016X}\n", p.dynamic_score_center_scale.to_bits()));
  out.push_str(&format!("F noResultUtilityForWhite {:016X}\n", p.no_result_utility_for_white.to_bits()));
  out.push_str(&format!("F drawEquivalentWinsForWhite {:016X}\n", p.draw_equivalent_wins_for_white.to_bits()));
  out.push_str(&format!("F cpuctExploration {:016X}\n", p.cpuct_exploration.to_bits()));
  out.push_str(&format!("F cpuctExplorationLog {:016X}\n", p.cpuct_exploration_log.to_bits()));
  out.push_str(&format!("F cpuctExplorationBase {:016X}\n", p.cpuct_exploration_base.to_bits()));
  out.push_str(&format!("F cpuctUtilityStdevPrior {:016X}\n", p.cpuct_utility_stdev_prior.to_bits()));
  out.push_str(&format!("F cpuctUtilityStdevPriorWeight {:016X}\n", p.cpuct_utility_stdev_prior_weight.to_bits()));
  out.push_str(&format!("F cpuctUtilityStdevScale {:016X}\n", p.cpuct_utility_stdev_scale.to_bits()));
  out.push_str(&format!("F fpuReductionMax {:016X}\n", p.fpu_reduction_max.to_bits()));
  out.push_str(&format!("F fpuLossProp {:016X}\n", p.fpu_loss_prop.to_bits()));
  out.push_str(&format!("F fpuParentWeightByVisitedPolicy {}\n", p.fpu_parent_weight_by_visited_policy as i32));
  out.push_str(&format!("F fpuParentWeightByVisitedPolicyPow {:016X}\n", p.fpu_parent_weight_by_visited_policy_pow.to_bits()));
  out.push_str(&format!("F fpuParentWeight {:016X}\n", p.fpu_parent_weight.to_bits()));
  out.push_str(&format!("F policyOptimism {:016X}\n", p.policy_optimism.to_bits()));
  out.push_str(&format!("F valueWeightExponent {:016X}\n", p.value_weight_exponent.to_bits()));
  out.push_str(&format!("F useNoisePruning {}\n", p.use_noise_pruning as i32));
  out.push_str(&format!("F noisePruneUtilityScale {:016X}\n", p.noise_prune_utility_scale.to_bits()));
  out.push_str(&format!("F noisePruningCap {:016X}\n", p.noise_pruning_cap.to_bits()));
  out.push_str(&format!("F useUncertainty {}\n", p.use_uncertainty as i32));
  out.push_str(&format!("F uncertaintyCoeff {:016X}\n", p.uncertainty_coeff.to_bits()));
  out.push_str(&format!("F uncertaintyExponent {:016X}\n", p.uncertainty_exponent.to_bits()));
  out.push_str(&format!("F uncertaintyMaxWeight {:016X}\n", p.uncertainty_max_weight.to_bits()));
  out.push_str(&format!("F useGraphSearch {}\n", p.use_graph_search as i32));
  out.push_str(&format!("F graphSearchRepBound {}\n", p.graph_search_rep_bound));
  out.push_str(&format!("F graphSearchCatchUpLeakProb {:016X}\n", p.graph_search_catch_up_leak_prob.to_bits()));
  out.push_str(&format!("F rootNoiseEnabled {}\n", p.root_noise_enabled as i32));
  out.push_str(&format!("F rootDirichletNoiseTotalConcentration {:016X}\n", p.root_dirichlet_noise_total_concentration.to_bits()));
  out.push_str(&format!("F rootDirichletNoiseWeight {:016X}\n", p.root_dirichlet_noise_weight.to_bits()));
  out.push_str(&format!("F rootPolicyTemperature {:016X}\n", p.root_policy_temperature.to_bits()));
  out.push_str(&format!("F rootPolicyTemperatureEarly {:016X}\n", p.root_policy_temperature_early.to_bits()));
  out.push_str(&format!("F rootFpuReductionMax {:016X}\n", p.root_fpu_reduction_max.to_bits()));
  out.push_str(&format!("F rootFpuLossProp {:016X}\n", p.root_fpu_loss_prop.to_bits()));
  out.push_str(&format!("F rootNumSymmetriesToSample {}\n", p.root_num_symmetries_to_sample));
  out.push_str(&format!("F rootSymmetryPruning {}\n", p.root_symmetry_pruning as i32));
  out.push_str(&format!("F rootDesiredPerChildVisitsCoeff {:016X}\n", p.root_desired_per_child_visits_coeff.to_bits()));
  out.push_str(&format!("F rootPolicyOptimism {:016X}\n", p.root_policy_optimism.to_bits()));
  out.push_str(&format!("F chosenMoveTemperature {:016X}\n", p.chosen_move_temperature.to_bits()));
  out.push_str(&format!("F chosenMoveTemperatureEarly {:016X}\n", p.chosen_move_temperature_early.to_bits()));
  out.push_str(&format!("F chosenMoveTemperatureHalflife {:016X}\n", p.chosen_move_temperature_halflife.to_bits()));
  out.push_str(&format!("F chosenMoveTemperatureOnlyBelowProb {:016X}\n", p.chosen_move_temperature_only_below_prob.to_bits()));
  out.push_str(&format!("F chosenMoveSubtract {:016X}\n", p.chosen_move_subtract.to_bits()));
  out.push_str(&format!("F chosenMovePrune {:016X}\n", p.chosen_move_prune.to_bits()));
  out.push_str(&format!("F useLcbForSelection {}\n", p.use_lcb_for_selection as i32));
  out.push_str(&format!("F lcbStdevs {:016X}\n", p.lcb_stdevs.to_bits()));
  out.push_str(&format!("F minVisitPropForLCB {:016X}\n", p.min_visit_prop_for_lcb.to_bits()));
  out.push_str(&format!("F useNonBuggyLcb {}\n", p.use_non_buggy_lcb as i32));
  out.push_str(&format!("F rootEndingBonusPoints {:016X}\n", p.root_ending_bonus_points.to_bits()));
  out.push_str(&format!("F rootPruneUselessMoves {}\n", p.root_prune_useless_moves as i32));
  out.push_str(&format!("F conservativePass {}\n", p.conservative_pass as i32));
  out.push_str(&format!("F fillDameBeforePass {}\n", p.fill_dame_before_pass as i32));
  out.push_str(&format!("F avoidMYTDaggerHackPla {}\n", p.avoid_myt_dagger_hack_pla as i32));
  out.push_str(&format!("F wideRootNoise {:016X}\n", p.wide_root_noise.to_bits()));
  out.push_str(&format!("F enablePassingHacks {}\n", p.enable_passing_hacks as i32));
  out.push_str(&format!("F enableMorePassingHacks {}\n", p.enable_more_passing_hacks as i32));
  out.push_str(&format!("F playoutDoublingAdvantage {:016X}\n", p.playout_doubling_advantage.to_bits()));
  out.push_str(&format!("F playoutDoublingAdvantagePla {}\n", p.playout_doubling_advantage_pla as i32));
  out.push_str(&format!("F avoidRepeatedPatternUtility {:016X}\n", p.avoid_repeated_pattern_utility.to_bits()));
  out.push_str(&format!("F nnPolicyTemperature {:08X}\n", p.nn_policy_temperature.to_bits()));
  out.push_str(&format!("F antiMirror {}\n", p.anti_mirror as i32));
  out.push_str(&format!("F ignorePreRootHistory {}\n", p.ignore_pre_root_history as i32));
  out.push_str(&format!("F ignoreAllHistory {}\n", p.ignore_all_history as i32));
  out.push_str(&format!("F subtreeValueBiasFactor {:016X}\n", p.subtree_value_bias_factor.to_bits()));
  out.push_str(&format!("F subtreeValueBiasTableNumShards {}\n", p.subtree_value_bias_table_num_shards));
  out.push_str(&format!("F subtreeValueBiasFreeProp {:016X}\n", p.subtree_value_bias_free_prop.to_bits()));
  out.push_str(&format!("F subtreeValueBiasWeightExponent {:016X}\n", p.subtree_value_bias_weight_exponent.to_bits()));
  out.push_str(&format!("F useEvalCache {}\n", p.use_eval_cache as i32));
  out.push_str(&format!("F evalCacheMinVisits {}\n", p.eval_cache_min_visits));
  out.push_str(&format!("F nodeTableShardsPowerOfTwo {}\n", p.node_table_shards_power_of_two));
  out.push_str(&format!("F numVirtualLossesPerThread {:016X}\n", p.num_virtual_losses_per_thread.to_bits()));
  out.push_str(&format!("F numThreads {}\n", p.num_threads));
  out.push_str(&format!("F minPlayoutsPerThread {:016X}\n", p.min_playouts_per_thread.to_bits()));
  out.push_str(&format!("F maxVisits {}\n", p.max_visits));
  out.push_str(&format!("F maxPlayouts {}\n", p.max_playouts));
  out.push_str(&format!("F maxTime {:016X}\n", p.max_time.to_bits()));
  out.push_str(&format!("F maxVisitsPondering {}\n", p.max_visits_pondering));
  out.push_str(&format!("F maxPlayoutsPondering {}\n", p.max_playouts_pondering));
  out.push_str(&format!("F maxTimePondering {:016X}\n", p.max_time_pondering.to_bits()));
  out.push_str(&format!("F lagBuffer {:016X}\n", p.lag_buffer.to_bits()));
  out.push_str(&format!("F searchFactorAfterOnePass {:016X}\n", p.search_factor_after_one_pass.to_bits()));
  out.push_str(&format!("F searchFactorAfterTwoPass {:016X}\n", p.search_factor_after_two_pass.to_bits()));
  out.push_str(&format!("F treeReuseCarryOverTimeFactor {:016X}\n", p.tree_reuse_carry_over_time_factor.to_bits()));
  out.push_str(&format!("F overallocateTimeFactor {:016X}\n", p.overallocate_time_factor.to_bits()));
  out.push_str(&format!("F midgameTimeFactor {:016X}\n", p.midgame_time_factor.to_bits()));
  out.push_str(&format!("F midgameTurnPeakTime {:016X}\n", p.midgame_turn_peak_time.to_bits()));
  out.push_str(&format!("F endgameTurnTimeDecay {:016X}\n", p.endgame_turn_time_decay.to_bits()));
  out.push_str(&format!("F obviousMovesTimeFactor {:016X}\n", p.obvious_moves_time_factor.to_bits()));
  out.push_str(&format!("F obviousMovesPolicyEntropyTolerance {:016X}\n", p.obvious_moves_policy_entropy_tolerance.to_bits()));
  out.push_str(&format!("F obviousMovesPolicySurpriseTolerance {:016X}\n", p.obvious_moves_policy_surprise_tolerance.to_bits()));
  out.push_str(&format!("F futileVisitsThreshold {:016X}\n", p.futile_visits_threshold.to_bits()));
  out.push_str(&format!("F humanSLProfile {}\n", p.human_sl_profile.initialized as i32));
  out.push_str(&format!("F humanSLCpuctExploration {:016X}\n", p.human_sl_cpuct_exploration.to_bits()));
  out.push_str(&format!("F humanSLCpuctPermanent {:016X}\n", p.human_sl_cpuct_permanent.to_bits()));
  out.push_str(&format!("F humanSLRootExploreProbWeightless {:016X}\n", p.human_sl_root_explore_prob_weightless.to_bits()));
  out.push_str(&format!("F humanSLRootExploreProbWeightful {:016X}\n", p.human_sl_root_explore_prob_weightful.to_bits()));
  out.push_str(&format!("F humanSLPlaExploreProbWeightless {:016X}\n", p.human_sl_pla_explore_prob_weightless.to_bits()));
  out.push_str(&format!("F humanSLPlaExploreProbWeightful {:016X}\n", p.human_sl_pla_explore_prob_weightful.to_bits()));
  out.push_str(&format!("F humanSLOppExploreProbWeightless {:016X}\n", p.human_sl_opp_explore_prob_weightless.to_bits()));
  out.push_str(&format!("F humanSLOppExploreProbWeightful {:016X}\n", p.human_sl_opp_explore_prob_weightful.to_bits()));
  out.push_str(&format!("F humanSLChosenMoveProp {:016X}\n", p.human_sl_chosen_move_prop.to_bits()));
  out.push_str(&format!("F humanSLChosenMoveIgnorePass {}\n", p.human_sl_chosen_move_ignore_pass as i32));
  out.push_str(&format!("F humanSLChosenMovePiklLambda {:016X}\n", p.human_sl_chosen_move_pikl_lambda.to_bits()));
  out.push_str(&format!("J {}\n", nlohmann::dump(&p.changeable_parameters_to_json())));
  {
    let mut out_stream = String::new();
    CAPTURED.with(|c| c.borrow_mut().clear());
    p.print_params(&mut out_stream);
    let cout_capture: String = CAPTURED.with(|c| c.borrow().iter().map(|l| format!("{}\n", l)).collect());
    emit_lines("PO", &out_stream, out);
    emit_lines("PC", &cout_capture, out);
  }
  { let mut q = p.clone(); q.win_loss_utility_factor = p.win_loss_utility_factor * 2.0 + 1.0; out.push_str(&format!("M winLossUtilityFactor {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.static_score_utility_factor = p.static_score_utility_factor * 2.0 + 1.0; out.push_str(&format!("M staticScoreUtilityFactor {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.dynamic_score_utility_factor = p.dynamic_score_utility_factor * 2.0 + 1.0; out.push_str(&format!("M dynamicScoreUtilityFactor {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.dynamic_score_center_zero_weight = p.dynamic_score_center_zero_weight * 2.0 + 1.0; out.push_str(&format!("M dynamicScoreCenterZeroWeight {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.dynamic_score_center_scale = p.dynamic_score_center_scale * 2.0 + 1.0; out.push_str(&format!("M dynamicScoreCenterScale {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.no_result_utility_for_white = p.no_result_utility_for_white * 2.0 + 1.0; out.push_str(&format!("M noResultUtilityForWhite {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.draw_equivalent_wins_for_white = p.draw_equivalent_wins_for_white * 2.0 + 1.0; out.push_str(&format!("M drawEquivalentWinsForWhite {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.cpuct_exploration = p.cpuct_exploration * 2.0 + 1.0; out.push_str(&format!("M cpuctExploration {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.cpuct_exploration_log = p.cpuct_exploration_log * 2.0 + 1.0; out.push_str(&format!("M cpuctExplorationLog {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.cpuct_exploration_base = p.cpuct_exploration_base * 2.0 + 1.0; out.push_str(&format!("M cpuctExplorationBase {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.cpuct_utility_stdev_prior = p.cpuct_utility_stdev_prior * 2.0 + 1.0; out.push_str(&format!("M cpuctUtilityStdevPrior {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.cpuct_utility_stdev_prior_weight = p.cpuct_utility_stdev_prior_weight * 2.0 + 1.0; out.push_str(&format!("M cpuctUtilityStdevPriorWeight {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.cpuct_utility_stdev_scale = p.cpuct_utility_stdev_scale * 2.0 + 1.0; out.push_str(&format!("M cpuctUtilityStdevScale {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.fpu_reduction_max = p.fpu_reduction_max * 2.0 + 1.0; out.push_str(&format!("M fpuReductionMax {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.fpu_loss_prop = p.fpu_loss_prop * 2.0 + 1.0; out.push_str(&format!("M fpuLossProp {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.fpu_parent_weight_by_visited_policy = !p.fpu_parent_weight_by_visited_policy; out.push_str(&format!("M fpuParentWeightByVisitedPolicy {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.fpu_parent_weight_by_visited_policy_pow = p.fpu_parent_weight_by_visited_policy_pow * 2.0 + 1.0; out.push_str(&format!("M fpuParentWeightByVisitedPolicyPow {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.fpu_parent_weight = p.fpu_parent_weight * 2.0 + 1.0; out.push_str(&format!("M fpuParentWeight {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.policy_optimism = p.policy_optimism * 2.0 + 1.0; out.push_str(&format!("M policyOptimism {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.value_weight_exponent = p.value_weight_exponent * 2.0 + 1.0; out.push_str(&format!("M valueWeightExponent {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.use_noise_pruning = !p.use_noise_pruning; out.push_str(&format!("M useNoisePruning {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.noise_prune_utility_scale = p.noise_prune_utility_scale * 2.0 + 1.0; out.push_str(&format!("M noisePruneUtilityScale {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.noise_pruning_cap = p.noise_pruning_cap * 2.0 + 1.0; out.push_str(&format!("M noisePruningCap {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.use_uncertainty = !p.use_uncertainty; out.push_str(&format!("M useUncertainty {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.uncertainty_coeff = p.uncertainty_coeff * 2.0 + 1.0; out.push_str(&format!("M uncertaintyCoeff {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.uncertainty_exponent = p.uncertainty_exponent * 2.0 + 1.0; out.push_str(&format!("M uncertaintyExponent {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.uncertainty_max_weight = p.uncertainty_max_weight * 2.0 + 1.0; out.push_str(&format!("M uncertaintyMaxWeight {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.use_graph_search = !p.use_graph_search; out.push_str(&format!("M useGraphSearch {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.graph_search_rep_bound = p.graph_search_rep_bound + 1; out.push_str(&format!("M graphSearchRepBound {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.graph_search_catch_up_leak_prob = p.graph_search_catch_up_leak_prob * 2.0 + 1.0; out.push_str(&format!("M graphSearchCatchUpLeakProb {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.root_noise_enabled = !p.root_noise_enabled; out.push_str(&format!("M rootNoiseEnabled {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.root_dirichlet_noise_total_concentration = p.root_dirichlet_noise_total_concentration * 2.0 + 1.0; out.push_str(&format!("M rootDirichletNoiseTotalConcentration {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.root_dirichlet_noise_weight = p.root_dirichlet_noise_weight * 2.0 + 1.0; out.push_str(&format!("M rootDirichletNoiseWeight {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.root_policy_temperature = p.root_policy_temperature * 2.0 + 1.0; out.push_str(&format!("M rootPolicyTemperature {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.root_policy_temperature_early = p.root_policy_temperature_early * 2.0 + 1.0; out.push_str(&format!("M rootPolicyTemperatureEarly {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.root_fpu_reduction_max = p.root_fpu_reduction_max * 2.0 + 1.0; out.push_str(&format!("M rootFpuReductionMax {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.root_fpu_loss_prop = p.root_fpu_loss_prop * 2.0 + 1.0; out.push_str(&format!("M rootFpuLossProp {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.root_num_symmetries_to_sample = p.root_num_symmetries_to_sample + 1; out.push_str(&format!("M rootNumSymmetriesToSample {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.root_symmetry_pruning = !p.root_symmetry_pruning; out.push_str(&format!("M rootSymmetryPruning {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.root_desired_per_child_visits_coeff = p.root_desired_per_child_visits_coeff * 2.0 + 1.0; out.push_str(&format!("M rootDesiredPerChildVisitsCoeff {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.root_policy_optimism = p.root_policy_optimism * 2.0 + 1.0; out.push_str(&format!("M rootPolicyOptimism {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.chosen_move_temperature = p.chosen_move_temperature * 2.0 + 1.0; out.push_str(&format!("M chosenMoveTemperature {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.chosen_move_temperature_early = p.chosen_move_temperature_early * 2.0 + 1.0; out.push_str(&format!("M chosenMoveTemperatureEarly {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.chosen_move_temperature_halflife = p.chosen_move_temperature_halflife * 2.0 + 1.0; out.push_str(&format!("M chosenMoveTemperatureHalflife {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.chosen_move_temperature_only_below_prob = p.chosen_move_temperature_only_below_prob * 2.0 + 1.0; out.push_str(&format!("M chosenMoveTemperatureOnlyBelowProb {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.chosen_move_subtract = p.chosen_move_subtract * 2.0 + 1.0; out.push_str(&format!("M chosenMoveSubtract {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.chosen_move_prune = p.chosen_move_prune * 2.0 + 1.0; out.push_str(&format!("M chosenMovePrune {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.use_lcb_for_selection = !p.use_lcb_for_selection; out.push_str(&format!("M useLcbForSelection {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.lcb_stdevs = p.lcb_stdevs * 2.0 + 1.0; out.push_str(&format!("M lcbStdevs {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.min_visit_prop_for_lcb = p.min_visit_prop_for_lcb * 2.0 + 1.0; out.push_str(&format!("M minVisitPropForLCB {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.use_non_buggy_lcb = !p.use_non_buggy_lcb; out.push_str(&format!("M useNonBuggyLcb {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.root_ending_bonus_points = p.root_ending_bonus_points * 2.0 + 1.0; out.push_str(&format!("M rootEndingBonusPoints {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.root_prune_useless_moves = !p.root_prune_useless_moves; out.push_str(&format!("M rootPruneUselessMoves {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.conservative_pass = !p.conservative_pass; out.push_str(&format!("M conservativePass {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.fill_dame_before_pass = !p.fill_dame_before_pass; out.push_str(&format!("M fillDameBeforePass {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.avoid_myt_dagger_hack_pla = p.avoid_myt_dagger_hack_pla ^ 1; out.push_str(&format!("M avoidMYTDaggerHackPla {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.wide_root_noise = p.wide_root_noise * 2.0 + 1.0; out.push_str(&format!("M wideRootNoise {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.enable_passing_hacks = !p.enable_passing_hacks; out.push_str(&format!("M enablePassingHacks {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.enable_more_passing_hacks = !p.enable_more_passing_hacks; out.push_str(&format!("M enableMorePassingHacks {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.playout_doubling_advantage = p.playout_doubling_advantage * 2.0 + 1.0; out.push_str(&format!("M playoutDoublingAdvantage {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.playout_doubling_advantage_pla = p.playout_doubling_advantage_pla ^ 1; out.push_str(&format!("M playoutDoublingAdvantagePla {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.avoid_repeated_pattern_utility = p.avoid_repeated_pattern_utility * 2.0 + 1.0; out.push_str(&format!("M avoidRepeatedPatternUtility {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.nn_policy_temperature = p.nn_policy_temperature * 2.0f32 + 1.0f32; out.push_str(&format!("M nnPolicyTemperature {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.anti_mirror = !p.anti_mirror; out.push_str(&format!("M antiMirror {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.ignore_pre_root_history = !p.ignore_pre_root_history; out.push_str(&format!("M ignorePreRootHistory {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.ignore_all_history = !p.ignore_all_history; out.push_str(&format!("M ignoreAllHistory {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.subtree_value_bias_factor = p.subtree_value_bias_factor * 2.0 + 1.0; out.push_str(&format!("M subtreeValueBiasFactor {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.subtree_value_bias_table_num_shards = p.subtree_value_bias_table_num_shards + 1; out.push_str(&format!("M subtreeValueBiasTableNumShards {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.subtree_value_bias_free_prop = p.subtree_value_bias_free_prop * 2.0 + 1.0; out.push_str(&format!("M subtreeValueBiasFreeProp {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.subtree_value_bias_weight_exponent = p.subtree_value_bias_weight_exponent * 2.0 + 1.0; out.push_str(&format!("M subtreeValueBiasWeightExponent {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.use_eval_cache = !p.use_eval_cache; out.push_str(&format!("M useEvalCache {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.eval_cache_min_visits = p.eval_cache_min_visits + 1; out.push_str(&format!("M evalCacheMinVisits {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.node_table_shards_power_of_two = p.node_table_shards_power_of_two + 1; out.push_str(&format!("M nodeTableShardsPowerOfTwo {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.num_virtual_losses_per_thread = p.num_virtual_losses_per_thread * 2.0 + 1.0; out.push_str(&format!("M numVirtualLossesPerThread {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.num_threads = p.num_threads + 1; out.push_str(&format!("M numThreads {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.min_playouts_per_thread = p.min_playouts_per_thread * 2.0 + 1.0; out.push_str(&format!("M minPlayoutsPerThread {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.max_visits = p.max_visits + 1; out.push_str(&format!("M maxVisits {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.max_playouts = p.max_playouts + 1; out.push_str(&format!("M maxPlayouts {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.max_time = p.max_time * 2.0 + 1.0; out.push_str(&format!("M maxTime {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.max_visits_pondering = p.max_visits_pondering + 1; out.push_str(&format!("M maxVisitsPondering {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.max_playouts_pondering = p.max_playouts_pondering + 1; out.push_str(&format!("M maxPlayoutsPondering {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.max_time_pondering = p.max_time_pondering * 2.0 + 1.0; out.push_str(&format!("M maxTimePondering {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.lag_buffer = p.lag_buffer * 2.0 + 1.0; out.push_str(&format!("M lagBuffer {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.search_factor_after_one_pass = p.search_factor_after_one_pass * 2.0 + 1.0; out.push_str(&format!("M searchFactorAfterOnePass {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.search_factor_after_two_pass = p.search_factor_after_two_pass * 2.0 + 1.0; out.push_str(&format!("M searchFactorAfterTwoPass {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.tree_reuse_carry_over_time_factor = p.tree_reuse_carry_over_time_factor * 2.0 + 1.0; out.push_str(&format!("M treeReuseCarryOverTimeFactor {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.overallocate_time_factor = p.overallocate_time_factor * 2.0 + 1.0; out.push_str(&format!("M overallocateTimeFactor {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.midgame_time_factor = p.midgame_time_factor * 2.0 + 1.0; out.push_str(&format!("M midgameTimeFactor {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.midgame_turn_peak_time = p.midgame_turn_peak_time * 2.0 + 1.0; out.push_str(&format!("M midgameTurnPeakTime {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.endgame_turn_time_decay = p.endgame_turn_time_decay * 2.0 + 1.0; out.push_str(&format!("M endgameTurnTimeDecay {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.obvious_moves_time_factor = p.obvious_moves_time_factor * 2.0 + 1.0; out.push_str(&format!("M obviousMovesTimeFactor {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.obvious_moves_policy_entropy_tolerance = p.obvious_moves_policy_entropy_tolerance * 2.0 + 1.0; out.push_str(&format!("M obviousMovesPolicyEntropyTolerance {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.obvious_moves_policy_surprise_tolerance = p.obvious_moves_policy_surprise_tolerance * 2.0 + 1.0; out.push_str(&format!("M obviousMovesPolicySurpriseTolerance {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.futile_visits_threshold = p.futile_visits_threshold * 2.0 + 1.0; out.push_str(&format!("M futileVisitsThreshold {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.human_sl_profile.initialized = !q.human_sl_profile.initialized; out.push_str(&format!("M humanSLProfile {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.human_sl_cpuct_exploration = p.human_sl_cpuct_exploration * 2.0 + 1.0; out.push_str(&format!("M humanSLCpuctExploration {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.human_sl_cpuct_permanent = p.human_sl_cpuct_permanent * 2.0 + 1.0; out.push_str(&format!("M humanSLCpuctPermanent {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.human_sl_root_explore_prob_weightless = p.human_sl_root_explore_prob_weightless * 2.0 + 1.0; out.push_str(&format!("M humanSLRootExploreProbWeightless {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.human_sl_root_explore_prob_weightful = p.human_sl_root_explore_prob_weightful * 2.0 + 1.0; out.push_str(&format!("M humanSLRootExploreProbWeightful {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.human_sl_pla_explore_prob_weightless = p.human_sl_pla_explore_prob_weightless * 2.0 + 1.0; out.push_str(&format!("M humanSLPlaExploreProbWeightless {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.human_sl_pla_explore_prob_weightful = p.human_sl_pla_explore_prob_weightful * 2.0 + 1.0; out.push_str(&format!("M humanSLPlaExploreProbWeightful {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.human_sl_opp_explore_prob_weightless = p.human_sl_opp_explore_prob_weightless * 2.0 + 1.0; out.push_str(&format!("M humanSLOppExploreProbWeightless {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.human_sl_opp_explore_prob_weightful = p.human_sl_opp_explore_prob_weightful * 2.0 + 1.0; out.push_str(&format!("M humanSLOppExploreProbWeightful {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.human_sl_chosen_move_prop = p.human_sl_chosen_move_prop * 2.0 + 1.0; out.push_str(&format!("M humanSLChosenMoveProp {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.human_sl_chosen_move_ignore_pass = !p.human_sl_chosen_move_ignore_pass; out.push_str(&format!("M humanSLChosenMoveIgnorePass {}\n", (*p == q) as i32)); }
  { let mut q = p.clone(); q.human_sl_chosen_move_pikl_lambda = p.human_sl_chosen_move_pikl_lambda * 2.0 + 1.0; out.push_str(&format!("M humanSLChosenMovePiklLambda {}\n", (*p == q) as i32)); }
}

fn unchangeable(name: &str, a: &SearchParams, b: &SearchParams, out: &mut String) {
  match SearchParams::fail_if_params_differ_on_unchangeable_parameter(a, b) {
    Ok(()) => out.push_str(&format!("U {} OK\n", name)),
    Err(e) => out.push_str(&format!("U {} ERR {}\n", name, e.what())),
  }
}

wasm_test! {
fn searchparams_vs_cpp() {
  install_logger();
  let mut cases = Cases::new("searchparams_oracle::searchparams_vs_cpp");
  let expected = fixture("oracle", "searchparams.txt");
  let mut out = String::new();

  let presets: Vec<(&str, SearchParams)> = vec![
    ("default", SearchParams::new()),
    ("forTestsV1", SearchParams::for_tests_v1()),
    ("forTestsV2", SearchParams::for_tests_v2()),
    ("basicDecentParams", SearchParams::basic_decent_params()),
  ];
  for (name, p) in presets.iter() {
    dump(name, p, &mut out);
    cases.ran();
  }
  for (a_name, a) in presets.iter() {
    for (b_name, b) in presets.iter() {
      out.push_str(&format!("E {} {} {} {}\n", a_name, b_name, (*a == *b) as i32, (*a != *b) as i32));
      cases.ran();
    }
  }

  let base = SearchParams::for_tests_v2();
  unchangeable("same", &base, &base, &mut out);
  {
    let mut d = base.clone();
    d.node_table_shards_power_of_two += 1;
    unchangeable("nodeTableShardsPowerOfTwo", &base, &d, &mut out);
  }
  {
    let mut d = base.clone();
    d.use_eval_cache = !d.use_eval_cache;
    unchangeable("useEvalCache", &base, &d, &mut out);
  }
  {
    let mut d = base.clone();
    d.eval_cache_min_visits += 1;
    unchangeable("evalCacheMinVisits", &base, &d, &mut out);
  }
  {
    let mut d = base.clone();
    d.max_visits += 1;
    d.num_threads += 3;
    unchangeable("changeableOnly", &base, &d, &mut out);
  }
  {
    let mut d = base.clone();
    d.node_table_shards_power_of_two += 1;
    d.use_eval_cache = !d.use_eval_cache;
    d.eval_cache_min_visits += 1;
    unchangeable("allThree", &base, &d, &mut out);
  }
  {
    let mut d = base.clone();
    d.use_eval_cache = !d.use_eval_cache;
    d.eval_cache_min_visits += 1;
    unchangeable("evalCacheBoth", &base, &d, &mut out);
  }
  cases.ran();

  compare_lines("searchparams", &out, &expected);
}
}

/// upstream: Setup::loadParams (setup.cpp) + the `loadParams` lambda of gtp.cpp, test-local subset:
/// SETUP_FOR_GTP, one bot, and a config that contains only maxVisits=500, maxTimePondering=60.0,
/// lagBuffer=1.0, numSearchThreads=2, searchFactorAfterOnePass=0.50 and searchFactorAfterTwoPass=0.25
/// (gtp_example.cfg plus the command-line overrides), so every other key takes its `else` branch.
/// Returns the genmove params. TODO(port): replace by Setup::load_params (program/setup.rs, M3 wave 2b)
/// and the ConfigParser; humanSLProfile (SGFMetadata::getProfile, M8) is left at its default.
fn gtp_genmove_params_for_example_cfg() -> SearchParams {
  let mut params = SearchParams::new();

  //maxPlayouts, maxTime: not in the config
  params.max_visits = 500;
  params.max_playouts_pondering = 1i64 << 50;
  params.max_visits_pondering = 1i64 << 50;
  params.max_time_pondering = 60.0;
  params.lag_buffer = 1.0;
  params.search_factor_after_one_pass = 0.50;
  params.search_factor_after_two_pass = 0.25;
  params.num_threads = 2;
  params.min_playouts_per_thread = 8.0; //(setupFor == SETUP_FOR_ANALYSIS || setupFor == SETUP_FOR_GTP) ? 8.0 : 0.0

  params.win_loss_utility_factor = 1.0;
  params.static_score_utility_factor = 0.1;
  params.dynamic_score_utility_factor = 0.3;
  params.no_result_utility_for_white = 0.0;
  params.draw_equivalent_wins_for_white = 0.5;

  params.dynamic_score_center_zero_weight = 0.20;
  params.dynamic_score_center_scale = 0.75;

  params.cpuct_exploration = 1.0;
  params.cpuct_exploration_log = 0.45;
  params.cpuct_exploration_base = 500.0;

  params.cpuct_utility_stdev_prior = 0.40;
  params.cpuct_utility_stdev_prior_weight = 2.0;
  params.cpuct_utility_stdev_scale = 0.85; //((setupFor != DISTRIBUTED && setupFor != OTHER) ? 0.85 : 0.0)

  params.fpu_reduction_max = 0.2;
  params.fpu_loss_prop = 0.0;
  params.fpu_parent_weight_by_visited_policy = true; //(setupFor != DISTRIBUTED)
  if params.fpu_parent_weight_by_visited_policy {
    params.fpu_parent_weight_by_visited_policy_pow = 2.0;
  } else {
    params.fpu_parent_weight = 0.0;
  }

  params.policy_optimism = 1.0;

  params.value_weight_exponent = 0.25;
  params.use_noise_pruning = true;
  params.noise_prune_utility_scale = 0.15;
  params.noise_pruning_cap = 1e50;

  params.use_uncertainty = true;
  params.uncertainty_coeff = 0.25;
  params.uncertainty_exponent = 1.0;
  params.uncertainty_max_weight = 8.0;

  params.use_graph_search = true;
  params.graph_search_rep_bound = 11;
  params.graph_search_catch_up_leak_prob = 0.0;

  params.root_noise_enabled = false;
  params.root_dirichlet_noise_total_concentration = 10.83;
  params.root_dirichlet_noise_weight = 0.25;

  params.root_policy_temperature = 1.0;
  params.root_policy_temperature_early = params.root_policy_temperature;
  params.root_fpu_reduction_max = if params.root_noise_enabled { 0.0 } else { 0.1 };
  params.root_fpu_loss_prop = params.fpu_loss_prop;
  params.root_num_symmetries_to_sample = 1;
  params.root_symmetry_pruning = true; //(setupFor == SETUP_FOR_ANALYSIS || setupFor == SETUP_FOR_GTP)

  params.root_desired_per_child_visits_coeff = 0.0;

  params.root_policy_optimism = katago_core::port::cpp::cpp_min(params.policy_optimism, 0.2);

  params.chosen_move_temperature = 0.1;
  params.chosen_move_temperature_early = 0.5;
  params.chosen_move_temperature_halflife = 19.0;
  params.chosen_move_temperature_only_below_prob = 1.0;
  params.chosen_move_subtract = 0.0;
  params.chosen_move_prune = 1.0;

  params.use_lcb_for_selection = true;
  params.lcb_stdevs = 5.0;
  params.min_visit_prop_for_lcb = 0.15;
  params.use_non_buggy_lcb = true;

  params.root_ending_bonus_points = 0.5;
  params.root_prune_useless_moves = true;
  params.conservative_pass = false;
  params.fill_dame_before_pass = false;
  //Controlled by GTP directly, not used in any other mode
  params.avoid_myt_dagger_hack_pla = C_EMPTY;
  params.wide_root_noise = 0.00; //(setupFor == SETUP_FOR_ANALYSIS ? DEFAULT_ANALYSIS_WIDE_ROOT_NOISE : 0.00)

  params.enable_passing_hacks = true;
  params.enable_more_passing_hacks = true;

  params.playout_doubling_advantage = 0.0;
  params.playout_doubling_advantage_pla = C_EMPTY;

  params.avoid_repeated_pattern_utility = 0.0;

  params.nn_policy_temperature = 1.0f32;

  params.anti_mirror = false;

  params.ignore_pre_root_history = false;
  params.ignore_all_history = false;

  params.subtree_value_bias_factor = 0.45;
  params.subtree_value_bias_free_prop = 0.8;
  params.subtree_value_bias_weight_exponent = 0.85;

  params.use_eval_cache = false;
  params.eval_cache_min_visits = 100;

  params.node_table_shards_power_of_two = 16;
  params.num_virtual_losses_per_thread = 1.0;

  params.tree_reuse_carry_over_time_factor = 0.0;
  params.overallocate_time_factor = 1.0;
  params.midgame_time_factor = 1.0;
  params.midgame_turn_peak_time = 130.0;
  params.endgame_turn_time_decay = 100.0;
  params.obvious_moves_time_factor = 1.0;
  params.obvious_moves_policy_entropy_tolerance = 0.30;
  params.obvious_moves_policy_surprise_tolerance = 0.15;
  params.futile_visits_threshold = 0.0;

  params.human_sl_cpuct_exploration = 1.0;
  params.human_sl_cpuct_permanent = 0.0;
  params.human_sl_root_explore_prob_weightless = 0.0;
  params.human_sl_root_explore_prob_weightful = 0.0;
  params.human_sl_pla_explore_prob_weightless = 0.0;
  params.human_sl_pla_explore_prob_weightful = 0.0;
  params.human_sl_opp_explore_prob_weightless = 0.0;
  params.human_sl_opp_explore_prob_weightful = 0.0;
  params.human_sl_chosen_move_prop = 0.0;
  params.human_sl_chosen_move_ignore_pass = false;
  params.human_sl_chosen_move_pikl_lambda = 1000000000.0;

  //The `loadParams` lambda of gtp.cpp: "Set a default for conservativePass that differs from matches or selfplay"
  params.conservative_pass = true;
  params.fill_dame_before_pass = true;
  params
}

wasm_test! {
fn searchparams_vs_real_katago_get_params() {
  let mut cases = Cases::new("searchparams_oracle::searchparams_vs_real_katago_get_params");
  let reply = fixture("oracle", "searchparams_gtp_kata_get_params.txt");
  //GTP success reply: "= <json>\n\n"
  let json_text = reply.strip_prefix("= ").expect("GTP success prefix").trim_end();
  let mut expected = nlohmann::parse(json_text).expect("fixture is JSON");
  let expected_obj = expected.as_object_mut().expect("params object");

  //Keys gtp.cpp adds around changeableParametersToJson() (kata-get-params handler)
  for k in [
    "allowResignation",
    "analysisIgnorePreRootHistory",
    "analysisWideRootNoise",
    "antiMirror",
    "delayMoveMax",
    "delayMoveScale",
    "genmoveAntiMirror",
    "humanSLProfile",
    "ponderingEnabled",
  ] {
    assert!(expected_obj.remove(k).is_some(), "fixture lacks {}", k);
    cases.ran();
  }

  let actual = gtp_genmove_params_for_example_cfg().changeable_parameters_to_json();
  let actual_obj = actual.as_object().expect("params object");
  let ek: Vec<&String> = expected_obj.keys().collect();
  let ak: Vec<&String> = actual_obj.keys().collect();
  assert_eq!(ak, ek, "key sets differ");
  for (k, v) in actual_obj.iter() {
    assert_eq!(nlohmann::dump(v), nlohmann::dump(&expected_obj[k]), "value of {}", k);
    cases.ran();
  }
  //byte-exact dump of the whole object
  compare_lines("kata-get-params", &nlohmann::dump(&actual), &nlohmann::dump(&serde_json::Value::Object(expected_obj.clone())));

  //The constructor default that setup.cpp never touches
  assert_eq!(SearchParams::new().subtree_value_bias_table_num_shards, 65536);
  assert_eq!(expected_obj["subtreeValueBiasTableNumShards"], 65536);
  cases.ran();
}
}
