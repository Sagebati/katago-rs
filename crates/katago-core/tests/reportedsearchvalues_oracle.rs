//! search/reportedsearchvalues: `operator<<` against the unmodified upstream
//! (fixtures/upstream-v1.16.5/unit/oracle/reportedsearchvalues.txt, from oracle/reportedsearchvalues.cpp)
//! over special and generated field values. The constructor taking `const Search&` waits for
//! search::Search (TODO(port) in the source) and is not covered. Not a port of upstream code.
//! Keep in the same order as the oracle.

mod common;

use common::Cases;
use common::compare_lines;
use common::fixture;
use katago_core::search::reportedsearchvalues::ReportedSearchValues;

struct Splitmix {
  s: u64,
}

impl Splitmix {
  fn nxt(&mut self) -> u64 {
    self.s = self.s.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = self.s;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
  }
  fn gen_value(&mut self) -> f64 {
    const SCALES: [f64; 8] = [1.0, 1e-5, 1e6, 1e-9, 123456.789, 1e21, 100.0, 0.001];
    let u = (self.nxt() >> 11) as f64 / 9007199254740992.0;
    let s = SCALES[(self.nxt() % 8) as usize];
    let v = u * s;
    if (self.nxt() & 1) != 0 {
      -v
    } else {
      v
    }
  }
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn reportedsearchvalues_display_vs_cpp() {
  let mut cases = Cases::new("reportedsearchvalues_oracle::reportedsearchvalues_display_vs_cpp");
  let expected = fixture("oracle", "reportedsearchvalues.txt");
  let mut out = String::new();
  let mut r = Splitmix { s: 0x5265706f72746564 };

  let specials: [f64; 16] = [
    0.0,
    -0.0,
    1.0,
    -1.0,
    0.5,
    0.25,
    100000.0,
    999999.5,
    1234567.0,
    1e-5,
    0.0001,
    1e100,
    f64::INFINITY,
    f64::NEG_INFINITY,
    f64::NAN,
    5e-324,
  ];
  let visits_list: [i64; 7] = [0, 1, -1, 500, 1125899906842624, i64::MIN, i64::MAX];
  let mut idx = 0;
  for &sp in specials.iter() {
    let mut v = ReportedSearchValues::new();
    v.win_value = sp;
    v.loss_value = -sp;
    v.no_result_value = sp * 0.5;
    v.static_score_value = sp + 1.0;
    v.dynamic_score_value = sp;
    v.expected_score = sp * 3.0;
    v.expected_score_stdev = sp;
    v.lead = sp - 2.0;
    v.win_loss_value = sp;
    v.utility = sp * 0.1;
    v.weight = sp;
    v.visits = visits_list[idx % 7];
    out.push_str(&format!("---- {}\n", idx));
    idx += 1;
    out.push_str(&format!("{}", v));
    cases.ran();
  }
  for _k in 0..60 {
    let mut v = ReportedSearchValues::new();
    v.win_value = r.gen_value();
    v.loss_value = r.gen_value();
    v.no_result_value = r.gen_value();
    v.static_score_value = r.gen_value();
    v.dynamic_score_value = r.gen_value();
    v.expected_score = r.gen_value();
    v.expected_score_stdev = r.gen_value();
    v.lead = r.gen_value();
    v.win_loss_value = r.gen_value();
    v.utility = r.gen_value();
    v.weight = r.gen_value();
    v.visits = r.nxt() as i64;
    out.push_str(&format!("---- {}\n", idx));
    idx += 1;
    out.push_str(&format!("{}", v));
    cases.ran();
  }

  compare_lines("reportedsearchvalues", &out, &expected);
}
