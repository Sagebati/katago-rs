//! Setup (program/setup.rs) against the unmodified upstream program/setup.cpp: fixtures/upstream-v1.16.5/unit/oracle/
//! setup_oracle.cpp drives upstream's Setup on the config cases of setup_oracle_cases.txt (every upstream config,
//! mutated variants, fuzzed configs; made by setup_oracle_gen.py) and dumps what it produces. This test replays
//! every op through the port and must print the same bytes (see the oracle's header for the record format):
//! loadParams / loadSingleParams for all 6 setup_for_t x hasHumanModel x single (every SearchParams field,
//! floats as bit patterns), loadSingleRules, parseReportAnalysisWinrates, loadDefaultBoardXYSize, the unused keys
//! after each op, and initializeNNEvaluators with debugSkipNeuralNet ("/dev/null": the evaluators' settings, the
//! nnRandSeed stream, the log lines). Two backend prefixes: setup.cpp built as "dummybackend" (all ops) and as
//! "opencl" (an NCHW backend, initializeNNEvaluators ops only: the only ones that depend on the prefix).
//!
//! Differences the test accounts for (all port-side, none changes a result):
//! - `Setup::getBackendPrefixes` ends with "burn" (katago-nn-burn's backend; PORT NOTE in setup.rs). The oracle
//!   generator leaves "burn*" keys out of the cases, so the only place it shows is the PREFIXES line.
//! - Pending: none. (Until M8 the loadParams ops (P) of cases that set `humanSLProfile` were left out, because
//!   SGFMetadata::get_profile only knew the empty profile; the PENDING_* lists stay for future gaps.)
//!
//! Not a port of upstream code.

#![allow(clippy::unnecessary_cast)] // the dump casts every integer field to i64 like the oracle's (long long)

mod common;

use std::cell::RefCell;
use std::rc::Rc;

use common::Cases;
use common::compare_lines;
use common::fixture;
use futures::executor::LocalPool;
use katago_core::core::config_parser::ConfigParser;
use katago_core::core::global::KataError;
use katago_core::core::rand::Rand;
use katago_core::game::board::C_EMPTY;
use katago_core::game::board::P_BLACK;
use katago_core::game::board::P_WHITE;
use katago_core::game::board::Player;
use katago_core::neuralnet::nninterface::NeuralNet;
use katago_core::port::executor::Spawner;
use katago_core::program::setup::NNEvaluatorHost;
use katago_core::program::setup::Setup;
use katago_core::program::setup::setup_for_t;
use katago_core::search::searchparams::SearchParams;

/// Cases whose loadParams ops (P) are not compared yet, with the reason (none since M8 ported
/// SGFMetadata::get_profile: the configs/gtp_human* cases are compared).
/// A name is a prefix of the case name in setup_oracle_cases.txt ("#mutN" variants follow their config).
/// `pending_entries_match_cases` fails when an entry matches nothing.
const PENDING_M8_NAME_PREFIXES: &[&str] = &[];
/// Fuzzed cases (by number in setup_oracle_cases.txt) with the same reason.
const PENDING_M8_CASES: &[i32] = &[];

thread_local! {
  static LOGS: RefCell<Vec<String>> = const { RefCell::new(Vec::new()) };
}

/// Collects the log lines (`logger.write` -> `log::info!`) of the current thread.
struct Cap;
impl log::Log for Cap {
  fn enabled(&self, _: &log::Metadata) -> bool {
    true
  }
  fn log(&self, r: &log::Record) {
    LOGS.with(|l| l.borrow_mut().push(format!("{}", r.args())));
  }
  fn flush(&self) {}
}
static CAP: Cap = Cap;

/// load_model must not be reached: every NN op here has debugSkipNeuralNet ("/dev/null").
struct Host {
  prefix: String,
  spawner: Rc<dyn Spawner>,
}

impl NNEvaluatorHost for Host {
  fn backend_prefix(&self) -> String {
    self.prefix.clone()
  }
  fn global_initialize(&self) {}
  fn global_cleanup(&self) {}
  fn load_model(&self, nn_model_name: &str, nn_model_file: &str) -> Result<(Rc<dyn NeuralNet>, Vec<u8>), KataError> {
    panic!("load_model({nn_model_name}, {nn_model_file}): the oracle cases skip the net")
  }
  fn spawner(&self) -> Rc<dyn Spawner> {
    self.spawner.clone()
  }
}

fn err_line(e: &KataError) -> String {
  let t = match e {
    KataError::StringError(_) => "StringError",
    KataError::IOError(_) => "IOError",
    KataError::ConfigParsingError(_) => "ConfigParsingError",
    KataError::ValueError(_) => "ValueError",
    KataError::CommandError(_) => "CommandError",
    KataError::NlohmannJson(_) => "Exception",
  };
  format!("ERR {} {}", t, e.what())
}

fn sf_of(i: i32) -> setup_for_t {
  [
    setup_for_t::SETUP_FOR_GTP,
    setup_for_t::SETUP_FOR_BENCHMARK,
    setup_for_t::SETUP_FOR_MATCH,
    setup_for_t::SETUP_FOR_ANALYSIS,
    setup_for_t::SETUP_FOR_OTHER,
    setup_for_t::SETUP_FOR_DISTRIBUTED,
  ][i as usize]
}

fn param_lines(p: &SearchParams) -> Vec<String> {
  let mut o: Vec<String> = Vec::new();
  o.push(format!("F winLossUtilityFactor {:016X}", p.win_loss_utility_factor.to_bits()));
  o.push(format!("F staticScoreUtilityFactor {:016X}", p.static_score_utility_factor.to_bits()));
  o.push(format!("F dynamicScoreUtilityFactor {:016X}", p.dynamic_score_utility_factor.to_bits()));
  o.push(format!("F dynamicScoreCenterZeroWeight {:016X}", p.dynamic_score_center_zero_weight.to_bits()));
  o.push(format!("F dynamicScoreCenterScale {:016X}", p.dynamic_score_center_scale.to_bits()));
  o.push(format!("F noResultUtilityForWhite {:016X}", p.no_result_utility_for_white.to_bits()));
  o.push(format!("F drawEquivalentWinsForWhite {:016X}", p.draw_equivalent_wins_for_white.to_bits()));
  o.push(format!("F cpuctExploration {:016X}", p.cpuct_exploration.to_bits()));
  o.push(format!("F cpuctExplorationLog {:016X}", p.cpuct_exploration_log.to_bits()));
  o.push(format!("F cpuctExplorationBase {:016X}", p.cpuct_exploration_base.to_bits()));
  o.push(format!("F cpuctUtilityStdevPrior {:016X}", p.cpuct_utility_stdev_prior.to_bits()));
  o.push(format!("F cpuctUtilityStdevPriorWeight {:016X}", p.cpuct_utility_stdev_prior_weight.to_bits()));
  o.push(format!("F cpuctUtilityStdevScale {:016X}", p.cpuct_utility_stdev_scale.to_bits()));
  o.push(format!("F fpuReductionMax {:016X}", p.fpu_reduction_max.to_bits()));
  o.push(format!("F fpuLossProp {:016X}", p.fpu_loss_prop.to_bits()));
  o.push(format!("F fpuParentWeightByVisitedPolicy {}", p.fpu_parent_weight_by_visited_policy as i64));
  o.push(format!("F fpuParentWeightByVisitedPolicyPow {:016X}", p.fpu_parent_weight_by_visited_policy_pow.to_bits()));
  o.push(format!("F fpuParentWeight {:016X}", p.fpu_parent_weight.to_bits()));
  o.push(format!("F policyOptimism {:016X}", p.policy_optimism.to_bits()));
  o.push(format!("F valueWeightExponent {:016X}", p.value_weight_exponent.to_bits()));
  o.push(format!("F useNoisePruning {}", p.use_noise_pruning as i64));
  o.push(format!("F noisePruneUtilityScale {:016X}", p.noise_prune_utility_scale.to_bits()));
  o.push(format!("F noisePruningCap {:016X}", p.noise_pruning_cap.to_bits()));
  o.push(format!("F useUncertainty {}", p.use_uncertainty as i64));
  o.push(format!("F uncertaintyCoeff {:016X}", p.uncertainty_coeff.to_bits()));
  o.push(format!("F uncertaintyExponent {:016X}", p.uncertainty_exponent.to_bits()));
  o.push(format!("F uncertaintyMaxWeight {:016X}", p.uncertainty_max_weight.to_bits()));
  o.push(format!("F useGraphSearch {}", p.use_graph_search as i64));
  o.push(format!("F graphSearchRepBound {}", p.graph_search_rep_bound as i64));
  o.push(format!("F graphSearchCatchUpLeakProb {:016X}", p.graph_search_catch_up_leak_prob.to_bits()));
  o.push(format!("F rootNoiseEnabled {}", p.root_noise_enabled as i64));
  o.push(format!("F rootDirichletNoiseTotalConcentration {:016X}", p.root_dirichlet_noise_total_concentration.to_bits()));
  o.push(format!("F rootDirichletNoiseWeight {:016X}", p.root_dirichlet_noise_weight.to_bits()));
  o.push(format!("F rootPolicyTemperature {:016X}", p.root_policy_temperature.to_bits()));
  o.push(format!("F rootPolicyTemperatureEarly {:016X}", p.root_policy_temperature_early.to_bits()));
  o.push(format!("F rootFpuReductionMax {:016X}", p.root_fpu_reduction_max.to_bits()));
  o.push(format!("F rootFpuLossProp {:016X}", p.root_fpu_loss_prop.to_bits()));
  o.push(format!("F rootNumSymmetriesToSample {}", p.root_num_symmetries_to_sample as i64));
  o.push(format!("F rootSymmetryPruning {}", p.root_symmetry_pruning as i64));
  o.push(format!("F rootDesiredPerChildVisitsCoeff {:016X}", p.root_desired_per_child_visits_coeff.to_bits()));
  o.push(format!("F rootPolicyOptimism {:016X}", p.root_policy_optimism.to_bits()));
  o.push(format!("F chosenMoveTemperature {:016X}", p.chosen_move_temperature.to_bits()));
  o.push(format!("F chosenMoveTemperatureEarly {:016X}", p.chosen_move_temperature_early.to_bits()));
  o.push(format!("F chosenMoveTemperatureHalflife {:016X}", p.chosen_move_temperature_halflife.to_bits()));
  o.push(format!("F chosenMoveTemperatureOnlyBelowProb {:016X}", p.chosen_move_temperature_only_below_prob.to_bits()));
  o.push(format!("F chosenMoveSubtract {:016X}", p.chosen_move_subtract.to_bits()));
  o.push(format!("F chosenMovePrune {:016X}", p.chosen_move_prune.to_bits()));
  o.push(format!("F useLcbForSelection {}", p.use_lcb_for_selection as i64));
  o.push(format!("F lcbStdevs {:016X}", p.lcb_stdevs.to_bits()));
  o.push(format!("F minVisitPropForLCB {:016X}", p.min_visit_prop_for_lcb.to_bits()));
  o.push(format!("F useNonBuggyLcb {}", p.use_non_buggy_lcb as i64));
  o.push(format!("F rootEndingBonusPoints {:016X}", p.root_ending_bonus_points.to_bits()));
  o.push(format!("F rootPruneUselessMoves {}", p.root_prune_useless_moves as i64));
  o.push(format!("F conservativePass {}", p.conservative_pass as i64));
  o.push(format!("F fillDameBeforePass {}", p.fill_dame_before_pass as i64));
  o.push(format!("F avoidMYTDaggerHackPla {}", p.avoid_myt_dagger_hack_pla as i64));
  o.push(format!("F wideRootNoise {:016X}", p.wide_root_noise.to_bits()));
  o.push(format!("F enablePassingHacks {}", p.enable_passing_hacks as i64));
  o.push(format!("F enableMorePassingHacks {}", p.enable_more_passing_hacks as i64));
  o.push(format!("F playoutDoublingAdvantage {:016X}", p.playout_doubling_advantage.to_bits()));
  o.push(format!("F playoutDoublingAdvantagePla {}", p.playout_doubling_advantage_pla as i64));
  o.push(format!("F avoidRepeatedPatternUtility {:016X}", p.avoid_repeated_pattern_utility.to_bits()));
  o.push(format!("F nnPolicyTemperature {:08X}", p.nn_policy_temperature.to_bits()));
  o.push(format!("F antiMirror {}", p.anti_mirror as i64));
  o.push(format!("F ignorePreRootHistory {}", p.ignore_pre_root_history as i64));
  o.push(format!("F ignoreAllHistory {}", p.ignore_all_history as i64));
  o.push(format!("F subtreeValueBiasFactor {:016X}", p.subtree_value_bias_factor.to_bits()));
  o.push(format!("F subtreeValueBiasTableNumShards {}", p.subtree_value_bias_table_num_shards as i64));
  o.push(format!("F subtreeValueBiasFreeProp {:016X}", p.subtree_value_bias_free_prop.to_bits()));
  o.push(format!("F subtreeValueBiasWeightExponent {:016X}", p.subtree_value_bias_weight_exponent.to_bits()));
  o.push(format!("F useEvalCache {}", p.use_eval_cache as i64));
  o.push(format!("F evalCacheMinVisits {}", p.eval_cache_min_visits as i64));
  o.push(format!("F nodeTableShardsPowerOfTwo {}", p.node_table_shards_power_of_two as i64));
  o.push(format!("F numVirtualLossesPerThread {:016X}", p.num_virtual_losses_per_thread.to_bits()));
  o.push(format!("F numThreads {}", p.num_threads as i64));
  o.push(format!("F minPlayoutsPerThread {:016X}", p.min_playouts_per_thread.to_bits()));
  o.push(format!("F maxVisits {}", p.max_visits as i64));
  o.push(format!("F maxPlayouts {}", p.max_playouts as i64));
  o.push(format!("F maxTime {:016X}", p.max_time.to_bits()));
  o.push(format!("F maxVisitsPondering {}", p.max_visits_pondering as i64));
  o.push(format!("F maxPlayoutsPondering {}", p.max_playouts_pondering as i64));
  o.push(format!("F maxTimePondering {:016X}", p.max_time_pondering.to_bits()));
  o.push(format!("F lagBuffer {:016X}", p.lag_buffer.to_bits()));
  o.push(format!("F searchFactorAfterOnePass {:016X}", p.search_factor_after_one_pass.to_bits()));
  o.push(format!("F searchFactorAfterTwoPass {:016X}", p.search_factor_after_two_pass.to_bits()));
  o.push(format!("F treeReuseCarryOverTimeFactor {:016X}", p.tree_reuse_carry_over_time_factor.to_bits()));
  o.push(format!("F overallocateTimeFactor {:016X}", p.overallocate_time_factor.to_bits()));
  o.push(format!("F midgameTimeFactor {:016X}", p.midgame_time_factor.to_bits()));
  o.push(format!("F midgameTurnPeakTime {:016X}", p.midgame_turn_peak_time.to_bits()));
  o.push(format!("F endgameTurnTimeDecay {:016X}", p.endgame_turn_time_decay.to_bits()));
  o.push(format!("F obviousMovesTimeFactor {:016X}", p.obvious_moves_time_factor.to_bits()));
  o.push(format!("F obviousMovesPolicyEntropyTolerance {:016X}", p.obvious_moves_policy_entropy_tolerance.to_bits()));
  o.push(format!("F obviousMovesPolicySurpriseTolerance {:016X}", p.obvious_moves_policy_surprise_tolerance.to_bits()));
  o.push(format!("F futileVisitsThreshold {:016X}", p.futile_visits_threshold.to_bits()));
  o.push(format!("F humanSLProfile {}", p.human_sl_profile.initialized as i32));
  o.push(format!("F humanSLCpuctExploration {:016X}", p.human_sl_cpuct_exploration.to_bits()));
  o.push(format!("F humanSLCpuctPermanent {:016X}", p.human_sl_cpuct_permanent.to_bits()));
  o.push(format!("F humanSLRootExploreProbWeightless {:016X}", p.human_sl_root_explore_prob_weightless.to_bits()));
  o.push(format!("F humanSLRootExploreProbWeightful {:016X}", p.human_sl_root_explore_prob_weightful.to_bits()));
  o.push(format!("F humanSLPlaExploreProbWeightless {:016X}", p.human_sl_pla_explore_prob_weightless.to_bits()));
  o.push(format!("F humanSLPlaExploreProbWeightful {:016X}", p.human_sl_pla_explore_prob_weightful.to_bits()));
  o.push(format!("F humanSLOppExploreProbWeightless {:016X}", p.human_sl_opp_explore_prob_weightless.to_bits()));
  o.push(format!("F humanSLOppExploreProbWeightful {:016X}", p.human_sl_opp_explore_prob_weightful.to_bits()));
  o.push(format!("F humanSLChosenMoveProp {:016X}", p.human_sl_chosen_move_prop.to_bits()));
  o.push(format!("F humanSLChosenMoveIgnorePass {}", p.human_sl_chosen_move_ignore_pass as i64));
  o.push(format!("F humanSLChosenMovePiklLambda {:016X}", p.human_sl_chosen_move_pikl_lambda.to_bits()));
  o
}

/// The unused keys. The n-th distinct list of a case is printed the first time it appears ("U+") as the difference to
/// the previous distinct list of the case (the empty list for the first): " -key" for each key that left it (in the
/// previous list's order), then " +key" for each that joined (in the list's order); later as the list's number ("U n").
struct Unused {
  seen: Vec<Vec<String>>,
}

impl Unused {
  fn dump(&mut self, cfg: &ConfigParser, o: &mut String) {
    let u = cfg.unused_keys();
    if let Some(i) = self.seen.iter().position(|s| *s == u) {
      o.push_str(&format!("U {}\n", i));
      return;
    }
    let empty: Vec<String> = Vec::new();
    let prev = self.seen.last().unwrap_or(&empty);
    o.push_str("U+");
    for k in prev {
      if !u.contains(k) {
        o.push_str(&format!(" -{}", k));
      }
    }
    for k in &u {
      if !prev.contains(k) {
        o.push_str(&format!(" +{}", k));
      }
    }
    o.push('\n');
    self.seen.push(u);
  }
}

/// Only the fields that differ from the previous dump of the case (BASE for the first) are printed.
fn dump_params(p: &SearchParams, prev: &mut Vec<String>, o: &mut String) {
  let lines = param_lines(p);
  for (i, l) in lines.iter().enumerate() {
    if *l != prev[i] {
      o.push_str(l);
      o.push('\n');
    }
  }
  *prev = lines;
}

struct Case {
  n: i32,
  nn: i32,
  name: String,
  text: String,
}

fn parse_cases(text: &str) -> Vec<Case> {
  let mut cases = Vec::new();
  let mut lines = text.lines();
  while let Some(line) = lines.next() {
    if !line.starts_with("CASE ") {
      continue;
    }
    let f: Vec<&str> = line.split(' ').collect();
    let mut ctext = String::new();
    for l in lines.by_ref() {
      if l == "END" {
        break;
      }
      ctext.push_str(l);
      ctext.push('\n');
    }
    cases.push(Case { n: f[1].parse().unwrap(), nn: f[2].parse().unwrap(), name: f[3].to_string(), text: ctext });
  }
  cases
}

fn is_pending(c: &Case) -> bool {
  PENDING_M8_CASES.contains(&c.n) || PENDING_M8_NAME_PREFIXES.iter().any(|p| c.name.starts_with(p))
}

/// Removes the P ops of the pending cases from an oracle output (the P ops are the last of a case: the records
/// from the first "P ..." line up to the next case).
fn drop_pending(expected: &str, cases: &[Case]) -> String {
  let mut o = String::new();
  let mut pending_case = false;
  let mut in_p = false;
  for l in expected.split_inclusive('\n') {
    if let Some(rest) = l.strip_prefix("C ") {
      let n: i32 = rest.split(' ').next().unwrap().parse().unwrap();
      pending_case = cases.iter().any(|c| c.n == n && is_pending(c));
      in_p = false;
    } else if pending_case && l.starts_with("P ") {
      in_p = true;
    }
    if !in_p {
      o.push_str(l);
    }
  }
  o
}

/// The ops of setup_oracle.cpp, through the port. `nn_only` is the oracle's "nn" mode.
fn run_oracle(cases: &[Case], prefix: &str, nn_only: bool) -> String {
  common::init();
  let _ = log::set_logger(&CAP);
  log::set_max_level(log::LevelFilter::Info);
  let base_lines = param_lines(&SearchParams::new());
  let mut o = String::new();
  {
    let m = Setup::get_mutex_key_sets();
    o.push_str("MUTEX");
    for (a, b) in &m {
      o.push_str(" {");
      for s in a {
        o.push(' ');
        o.push_str(s);
      }
      o.push_str(" } {");
      for s in b {
        o.push(' ');
        o.push_str(s);
      }
      o.push_str(" }");
    }
    o.push('\n');
    o.push_str("PREFIXES");
    // PORT NOTE: the port's own "burn" prefix is not upstream's.
    let prefixes = Setup::get_backend_prefixes();
    assert_eq!(prefixes.last().map(|s| s.as_str()), Some("burn"));
    for s in &prefixes[..prefixes.len() - 1] {
      o.push(' ');
      o.push_str(s);
    }
    o.push('\n');
    o.push_str("BASE\n");
    for l in &base_lines {
      o.push_str(l);
      o.push('\n');
    }
    o.push_str("ENDBASE\n");
  }
  for c in cases {
    let (n, name, ctext) = (c.n, &c.name, &c.text);
    o.push_str(&format!("C {} {}\n", n, name));
    let mut unused = Unused { seen: Vec::new() };
    let mut seen_n: Vec<String> = Vec::new();
    let mut prev = base_lines.clone();
    let mk = |o: &mut String| -> Option<ConfigParser> {
      match ConfigParser::new_in(ctext, false, true) {
        Ok(c) => Some(c),
        Err(e) => {
          o.push_str(&format!("PARSE {}\n", err_line(&e)));
          None
        }
      }
    };
    if !nn_only && c.nn != 2 {
      for lk in 0..2 {
        let Some(mut cfg) = mk(&mut o) else { continue };
        o.push_str(&format!("R {}\n", lk));
        match Setup::load_single_rules(&mut cfg, lk == 1) {
          Ok(r) => o.push_str(&format!(
            "RULES {} {} {} {} {} {} {} {:08X} {}\n",
            r.ko_rule,
            r.scoring_rule,
            r.tax_rule,
            r.multi_stone_suicide_legal as i32,
            r.has_button as i32,
            r.white_handicap_bonus_rule,
            r.friendly_pass_ok as i32,
            r.komi.to_bits(),
            r.to_string()
          )),
          Err(e) => {
            o.push_str(&err_line(&e));
            o.push('\n');
          }
        }
        unused.dump(&cfg, &mut o);
      }
      for dp in 0..3 {
        let Some(mut cfg) = mk(&mut o) else { continue };
        let def: Player = if dp == 0 {
          P_BLACK
        } else if dp == 1 {
          P_WHITE
        } else {
          C_EMPTY
        };
        o.push_str(&format!("W {}\n", dp));
        match Setup::parse_report_analysis_winrates(&mut cfg, def) {
          Ok(p) => o.push_str(&format!("PLA {}\n", p as i32)),
          Err(e) => {
            o.push_str(&err_line(&e));
            o.push('\n');
          }
        }
        unused.dump(&cfg, &mut o);
      }
      if let Some(mut cfg) = mk(&mut o) {
        o.push_str("X\n");
        let (mut x, mut y) = (-7, -7);
        match Setup::load_default_board_xy_size(&mut cfg, &mut x, &mut y) {
          Ok(b) => o.push_str(&format!("XY {} {} {}\n", b as i32, x, y)),
          Err(e) => {
            o.push_str(&err_line(&e));
            o.push('\n');
          }
        }
        unused.dump(&cfg, &mut o);
      }
    }
    if c.nn >= 1 {
      for sf in 0..6 {
        for variant in 0..4 {
          let Some(mut cfg) = mk(&mut o) else { continue };
          let default_max_batch_size = if variant & 1 != 0 { 16 } else { -1 };
          let disable_fp16 = variant & 2 != 0;
          let n_models = if variant == 3 { 1 } else { 2 };
          o.push_str(&format!("N {} {}\n", sf, variant));
          let mut blk = String::new();
          LOGS.with(|l| l.borrow_mut().clear());
          let mut seed_rand = Rand::new_seed(&format!("setuporacle{}", n));
          let names: Vec<String> = (0..n_models).map(|i| format!("model{}", i)).collect();
          let files: Vec<String> = (0..n_models).map(|_| "/dev/null".to_string()).collect();
          let shas: Vec<String> = if variant == 2 { (0..n_models).map(|_| String::new()).collect() } else { Vec::new() };
          let mut pool = LocalPool::new();
          let host = Host { prefix: prefix.to_string(), spawner: Rc::new(pool.spawner()) };
          let r = common::run_or_stall_panic(
            &mut pool,
            "initialize_nn_evaluators",
            Setup::initialize_nn_evaluators(&names, &files, &shas, &mut cfg, &mut seed_rand, 4, 19, 19, default_max_batch_size, false, disable_fp16, sf_of(sf), &host),
          );
          match r {
            Ok(evals) => {
              for (i, e) in evals.iter().enumerate() {
                let cache = match &e.nn_cache_table {
                  None => -1i64,
                  Some(c) => c.table_size as i64,
                };
                blk.push_str(&format!(
                  "E {} name={} file={} maxBatch={} curBatch={} x={} y={} exact={} nhwc={} fp16={} nhwcMode={} threads={}",
                  i,
                  e.get_model_name(),
                  e.get_model_file_name(),
                  e.get_max_batch_size(),
                  e.get_current_batch_size(),
                  e.get_nnx_len(),
                  e.get_nny_len(),
                  e.require_exact_nn_len as i32,
                  e.inputs_use_nhwc as i32,
                  e.get_using_fp16_mode().to_string(),
                  e.get_using_nhwc_mode().to_string(),
                  e.num_threads.get()
                ));
                blk.push_str(" gpus=");
                for g in e.gpu_idx_by_server_thread.borrow().iter() {
                  blk.push_str(&format!("{},", g));
                }
                blk.push_str(&format!(
                  " seed={} skip={} rand={} sym={} cache={} internal={} ver={}\n",
                  e.rand_seed,
                  e.is_neural_net_less() as i32,
                  e.get_do_randomize() as i32,
                  e.get_default_symmetry(),
                  cache,
                  e.get_internal_model_name(),
                  e.get_model_version()
                ));
              }
              // upstream's `delete e` joins the server threads (D8: the join is the pool running them out).
              for e in &evals {
                e.kill_server_threads();
              }
              pool.run_until_stalled();
            }
            Err(e) => {
              blk.push_str(&err_line(&e));
              blk.push('\n');
            }
          }
          blk.push_str(&format!("SEEDNEXT {}\n", seed_rand.next_uint64()));
          LOGS.with(|l| {
            for s in l.borrow().iter() {
              // only the lines setup.cpp itself writes
              let a = s.find("nnRandSeed");
              let b = s.find("After dedups");
              let pos = match (a, b) {
                (Some(x), Some(y)) => Some(x.min(y)),
                (x, None) => x,
                (None, y) => y,
              };
              if let Some(p) = pos {
                blk.push_str(&format!("L {}\n", &s[p..]));
              }
            }
          });
          // The distinct result blocks of the case, as in the oracle ("NR+" and the lines the first time, later "NR n").
          if let Some(k) = seen_n.iter().position(|b| *b == blk) {
            o.push_str(&format!("NR {}\n", k));
          } else {
            o.push_str("NR+\n");
            o.push_str(&blk);
            seen_n.push(blk);
          }
          unused.dump(&cfg, &mut o);
        }
      }
    }
    if !nn_only && c.nn != 2 && !is_pending(c) {
      for sf in 0..6 {
        for hh in 0..2 {
          for single in 0..2 {
            let Some(mut cfg) = mk(&mut o) else { continue };
            o.push_str(&format!("P {} {} {}\n", sf, hh, single));
            let r = if single == 1 {
              Setup::load_single_params_has_human_model(&mut cfg, sf_of(sf), hh == 1).map(|p| vec![p])
            } else {
              Setup::load_params_has_human_model(&mut cfg, sf_of(sf), hh == 1)
            };
            match r {
              Ok(ps) => {
                o.push_str(&format!("NB {}\n", ps.len()));
                for (i, p) in ps.iter().enumerate() {
                  o.push_str(&format!("B {}\n", i));
                  dump_params(p, &mut prev, &mut o);
                }
              }
              Err(e) => {
                o.push_str(&err_line(&e));
                o.push('\n');
              }
            }
            unused.dump(&cfg, &mut o);
          }
        }
      }
    }
  }
  o
}

fn check(prefix: &str, expected_file: &str, nn_only: bool, label: &'static str) {
  let cases = parse_cases(&fixture("oracle", "setup_oracle_cases.txt"));
  let expected = fixture("oracle", expected_file);
  let npending = cases.iter().filter(|c| is_pending(c)).count();
  let actual = run_oracle(&cases, prefix, nn_only);
  compare_lines(label, &actual, &drop_pending(&expected, &cases));
  let mut count = Cases::new(label);
  for _ in &cases {
    count.ran();
  }
  crate::report!("{label}: P ops of {} pending (M8) cases left out", npending);
}

/// setup.cpp as the "dummybackend" backend (NHWC inputs), all ops.
#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn setup_oracle_dummybackend() {
  check("dummybackend", "setup_oracle_dummybackend.txt", false, "setup_oracle::dummybackend");
}

/// setup.cpp as the "opencl" backend (NCHW inputs, other keys), initializeNNEvaluators ops.
#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn setup_oracle_opencl() {
  check("opencl", "setup_oracle_opencl.txt", true, "setup_oracle::opencl");
}

/// Every pending entry names a case that exists (a stale entry would silently skip nothing).
#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn pending_entries_match_cases() {
  let cases = parse_cases(&fixture("oracle", "setup_oracle_cases.txt"));
  for p in PENDING_M8_NAME_PREFIXES {
    assert!(cases.iter().any(|c| c.name.starts_with(p)), "pending prefix {p} matches no case");
  }
  for n in PENDING_M8_CASES {
    assert!(cases.iter().any(|c| c.n == *n), "pending case {n} does not exist");
  }
}
