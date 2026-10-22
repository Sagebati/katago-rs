// Ported from KataGo cpp/search/searchprint.h, cpp/search/searchprint.cpp @ ba938676d7f42d70950b3a535af2466fb642008c
//
// searchprint.cpp only includes searchprint.h: everything is the header's inline code.
// PORT NOTE (PORTING.md Appendix B): the fields keep upstream's trailing underscore (`max_depth_`),
// so the builder methods can keep the plain names (`max_depth(..)`).

use crate::core::global::KataError;
use crate::game::board::Board;
use crate::game::board::Loc;
use crate::game::board::Location;

/// upstream: PrintTreeOptions (searchprint.h)
#[derive(Clone, Debug)]
pub struct PrintTreeOptions {
  pub max_depth_: i32,
  pub max_children_to_show_: i32,
  pub min_visits_to_show_: i64,
  pub min_visits_to_expand_: i64,
  pub min_visits_prop_to_show_: f64,
  pub min_visits_prop_to_expand_: f64,
  pub max_pv_depth_: i32,
  pub print_raw_nn_: bool,
  pub print_sqs_: bool,
  pub print_avg_shortterm_error_: bool,
  pub branch_: Vec<Loc>,
  pub also_branch_: bool,
}

impl PrintTreeOptions {
  /// upstream: PrintTreeOptions::PrintTreeOptions (searchprint.h)
  pub fn new() -> PrintTreeOptions {
    PrintTreeOptions {
      max_depth_: 1,
      max_children_to_show_: 100000,
      min_visits_to_show_: 0,
      min_visits_to_expand_: 1,
      min_visits_prop_to_show_: 0.0,
      min_visits_prop_to_expand_: 0.0,
      max_pv_depth_: 7,
      print_raw_nn_: false,
      print_sqs_: false,
      print_avg_shortterm_error_: false,
      branch_: Vec::new(),
      also_branch_: false,
    }
  }

  /// upstream: PrintTreeOptions::maxDepth (searchprint.h)
  pub fn max_depth(&self, d: i32) -> PrintTreeOptions {
    let mut other: PrintTreeOptions = self.clone();
    other.max_depth_ = d;
    other
  }
  /// upstream: PrintTreeOptions::maxChildrenToShow (searchprint.h)
  pub fn max_children_to_show(&self, c: i32) -> PrintTreeOptions {
    let mut other: PrintTreeOptions = self.clone();
    other.max_children_to_show_ = c;
    other
  }
  /// upstream: PrintTreeOptions::minVisitsToShow (searchprint.h)
  pub fn min_visits_to_show(&self, v: i64) -> PrintTreeOptions {
    let mut other: PrintTreeOptions = self.clone();
    other.min_visits_to_show_ = v;
    other
  }
  /// upstream: PrintTreeOptions::minVisitsToExpand (searchprint.h)
  pub fn min_visits_to_expand(&self, v: i64) -> PrintTreeOptions {
    let mut other: PrintTreeOptions = self.clone();
    other.min_visits_to_expand_ = v;
    other
  }
  /// upstream: PrintTreeOptions::minVisitsPropToShow (searchprint.h)
  pub fn min_visits_prop_to_show(&self, p: f64) -> PrintTreeOptions {
    let mut other: PrintTreeOptions = self.clone();
    other.min_visits_prop_to_show_ = p;
    other
  }
  /// upstream: PrintTreeOptions::minVisitsPropToExpand (searchprint.h)
  pub fn min_visits_prop_to_expand(&self, p: f64) -> PrintTreeOptions {
    let mut other: PrintTreeOptions = self.clone();
    other.min_visits_prop_to_expand_ = p;
    other
  }
  /// upstream: PrintTreeOptions::printSqs (searchprint.h)
  pub fn print_sqs(&self, b: bool) -> PrintTreeOptions {
    let mut other: PrintTreeOptions = self.clone();
    other.print_sqs_ = b;
    other
  }
  /// upstream: PrintTreeOptions::printAvgShorttermError (searchprint.h)
  pub fn print_avg_shortterm_error(&self, b: bool) -> PrintTreeOptions {
    let mut other: PrintTreeOptions = self.clone();
    other.print_avg_shortterm_error_ = b;
    other
  }
  /// upstream: PrintTreeOptions::onlyBranch (searchprint.h)
  /// PORT NOTE: Location::parseSequence throws -> Result.
  pub fn only_branch(&self, board: &Board, moves: &str) -> Result<PrintTreeOptions, KataError> {
    let mut other: PrintTreeOptions = self.clone();
    other.branch_ = Location::parse_sequence(moves, board)?;
    Ok(other)
  }
  /// upstream: PrintTreeOptions::alsoBranch (searchprint.h)
  /// PORT NOTE: Location::parseSequence throws -> Result.
  pub fn also_branch(&self, board: &Board, moves: &str) -> Result<PrintTreeOptions, KataError> {
    let mut other: PrintTreeOptions = self.clone();
    other.branch_ = Location::parse_sequence(moves, board)?;
    other.also_branch_ = true;
    Ok(other)
  }
}
