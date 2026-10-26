// Ported from KataGo cpp/search/reportedsearchvalues.cpp, cpp/search/reportedsearchvalues.h @ fd0723fdbc0e9d82cf269c9630af8c27c57c07c4

use crate::port::fmt::cpp_ostream_f64;

use crate::neuralnet::nninputs::ScoreValue;
use crate::search::search::Search;

/// upstream: ReportedSearchValues (reportedsearchvalues.h)
#[derive(Clone, Debug, Default)]
pub struct ReportedSearchValues {
  pub win_value: f64,
  pub loss_value: f64,
  pub no_result_value: f64,
  pub static_score_value: f64,
  pub dynamic_score_value: f64,
  pub expected_score: f64,
  pub expected_score_stdev: f64,
  pub lead: f64,
  pub win_loss_value: f64,
  pub utility: f64,
  pub weight: f64,
  pub visits: i64,
}

impl ReportedSearchValues {
  /// upstream: ReportedSearchValues::ReportedSearchValues (reportedsearchvalues.cpp)
  /// PORT NOTE: the C++ default constructor leaves every field uninitialized; Rust zero-initializes
  /// (PORTING.md §3, uninitialized locals/fields).
  pub fn new() -> ReportedSearchValues {
    ReportedSearchValues {
      win_value: 0.0,
      loss_value: 0.0,
      no_result_value: 0.0,
      static_score_value: 0.0,
      dynamic_score_value: 0.0,
      expected_score: 0.0,
      expected_score_stdev: 0.0,
      lead: 0.0,
      win_loss_value: 0.0,
      utility: 0.0,
      weight: 0.0,
      visits: 0,
    }
  }

  /// upstream: ReportedSearchValues::ReportedSearchValues(const Search&, double, ...) (reportedsearchvalues.cpp)
  /// PORT NOTE: name from PORTING.md Appendix B.
  pub fn new_search(
    search: &Search,
    win_loss_value_avg: f64,
    no_result_value_avg: f64,
    score_mean_avg: f64,
    score_mean_sq_avg: f64,
    lead_avg: f64,
    utility_avg: f64,
    total_weight: f64,
    total_visits: i64,
  ) -> ReportedSearchValues {
    // PORT NOTE: the C++ constructor assigns the fields in its body; start from zeros (see `new`).
    let mut s: ReportedSearchValues = ReportedSearchValues::new();
    s.win_loss_value = win_loss_value_avg;
    s.no_result_value = no_result_value_avg;
    let score_mean: f64 = score_mean_avg;
    let score_mean_sq: f64 = score_mean_sq_avg;
    let score_stdev: f64 = ScoreValue::get_score_stdev(score_mean, score_mean_sq);
    let sqrt_board_area: f64 = search.root_board.sqrt_board_area();
    s.static_score_value = ScoreValue::expected_white_score_value(score_mean, score_stdev, 0.0, 2.0, sqrt_board_area);
    s.dynamic_score_value = ScoreValue::expected_white_score_value(
      score_mean,
      score_stdev,
      search.recent_score_center,
      search.search_params.dynamic_score_center_scale,
      sqrt_board_area,
    );
    s.expected_score = score_mean;
    s.expected_score_stdev = score_stdev;
    s.lead = lead_avg;
    s.utility = utility_avg;

    //Clamp. Due to tiny floating point errors, these could be outside range.
    if s.win_loss_value < -1.0 {
      s.win_loss_value = -1.0;
    }
    if s.win_loss_value > 1.0 {
      s.win_loss_value = 1.0;
    }
    if s.no_result_value < 0.0 {
      s.no_result_value = 0.0;
    }
    if s.no_result_value > 1.0 - s.win_loss_value.abs() {
      s.no_result_value = 1.0 - s.win_loss_value.abs();
    }

    s.win_value = 0.5 * (s.win_loss_value + (1.0 - s.no_result_value));
    s.loss_value = 0.5 * (-s.win_loss_value + (1.0 - s.no_result_value));

    //Handle float imprecision
    if s.win_value < 0.0 {
      s.win_value = 0.0;
    }
    if s.win_value > 1.0 {
      s.win_value = 1.0;
    }
    if s.loss_value < 0.0 {
      s.loss_value = 0.0;
    }
    if s.loss_value > 1.0 {
      s.loss_value = 1.0;
    }

    s.weight = total_weight;
    s.visits = total_visits;
    s
  }
}

// ~ReportedSearchValues() is an empty destructor: omitted (PORTING.md §5).

/// upstream: operator<<(std::ostream&, const ReportedSearchValues&) (reportedsearchvalues.cpp)
impl std::fmt::Display for ReportedSearchValues {
  fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    // PORT NOTE: `ostream << double` is the default-precision (%g, 6) form: cpp_ostream_f64.
    write!(out, "winValue {}\n", cpp_ostream_f64(self.win_value))?;
    write!(out, "lossValue {}\n", cpp_ostream_f64(self.loss_value))?;
    write!(out, "noResultValue {}\n", cpp_ostream_f64(self.no_result_value))?;
    write!(out, "staticScoreValue {}\n", cpp_ostream_f64(self.static_score_value))?;
    write!(out, "dynamicScoreValue {}\n", cpp_ostream_f64(self.dynamic_score_value))?;
    write!(out, "expectedScore {}\n", cpp_ostream_f64(self.expected_score))?;
    write!(out, "expectedScoreStdev {}\n", cpp_ostream_f64(self.expected_score_stdev))?;
    write!(out, "lead {}\n", cpp_ostream_f64(self.lead))?;
    write!(out, "winLossValue {}\n", cpp_ostream_f64(self.win_loss_value))?;
    write!(out, "utility {}\n", cpp_ostream_f64(self.utility))?;
    write!(out, "weight {}\n", cpp_ostream_f64(self.weight))?;
    write!(out, "visits {}\n", self.visits)?;
    Ok(())
  }
}
