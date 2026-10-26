//! Replays the input-version-7 parts of upstream cpp/tests/testnninputs.cpp @ v1.16.5
//! (Tests::runNNInputsV3V4Tests). That test prints to cout during `katago runoutputtests`, so the
//! oracle is upstream cpp/tests/results/runOutputTests.txt, cut into one fixture per "VERSION 7"
//! chunk (plus the fillRowV7-only "Passing hack condition" block) by
//! tools/extract_runoutputtests.py into fixtures/upstream-v1.18.2/unit/runoutputtests/
//! (nn-inputs-*.expected<k>.txt, passing-hack-*.expected0.txt). Boards from the raw blocks are in
//! fixtures/upstream-v1.18.2/unit/testnninputs/ (inputs only).
//!
//! Versions 3..6 are not replayed: fillRowV3..V6 are not ported (port-manifest.toml).
//! Each Rust test builds the exact text upstream wrote to cout from the "VERSION 7" line to the
//! next section/version boundary and compares it with TestCommon::expect (per-line trim, as every
//! other unit tier). The upstream `expect(name, actualNHWC, actualNCHW)` consistency checks run too.
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

// Test code keeps the shapes of upstream's tests and oracle programs (PORTING.md §11b).
#![allow(clippy::field_reassign_with_default, clippy::type_complexity)]

#[macro_use]
mod common;

use std::collections::BTreeMap;

use common::Cases;
use common::endl;
use common::fixture;
use common::init;
use katago_core::core::hash::Hash128;
use katago_core::core::test::TestCommon;
use katago_core::game::board::*;
use katago_core::game::boardhistory::BoardHistory;
use katago_core::game::rules::Rules;
use katago_core::neuralnet::nninputs::MiscNNInputParams;
use katago_core::neuralnet::nninputs::NNInputs;
use katago_core::neuralnet::nninputs::NNPos;
use katago_core::port::fmt::cpp_ostream_f64;
use katago_core::port::fmt::cpp_ostream_f64_prec;
use katago_core::strprintf;
use katago_core::testAssert;

const EXPECTED: &str = "runoutputtests";
const INPUTS: &str = "testnninputs";

const VERSION: i32 = 7;

/// `out << x` with `out << std::setprecision(5)` (the test's ostringstream), for floats.
fn p5(x: f32) -> String {
  cpp_ostream_f64_prec(x as f64, 5)
}

/// `cout << x` (default precision 6), for floats and doubles.
fn p6(x: f64) -> String {
  cpp_ostream_f64(x)
}

/// `cout << bool`
fn pb(b: bool) -> i32 {
  b as i32
}

fn expected(slug: &str, k: usize) -> String {
  fixture(EXPECTED, &format!("{}.expected{}.txt", slug, k))
}

/// upstream: printNNInputHWAndBoard (testnninputs.cpp) static, for inputsVersion 7.
fn print_nn_input_hw_and_board(
  out: &mut String,
  board: &Board,
  hist: &BoardHistory,
  nn_x_len: i32,
  nn_y_len: i32,
  inputs_use_nhwc: bool,
  row: &[f32],
  c: i32,
) {
  let num_features: i32 = NNInputs::NUM_FEATURES_SPATIAL_V7;

  *out += &format!("Channel: {}\n", c);

  for y in 0..nn_y_len {
    for x in 0..nn_x_len {
      let pos = NNPos::xy_to_pos(x, y, nn_x_len);
      if x > 0 {
        *out += " ";
      }
      if inputs_use_nhwc {
        *out += &p5(row[(pos * num_features + c) as usize]);
      } else {
        *out += &p5(row[(c * nn_x_len * nn_y_len + pos) as usize]);
      }
    }
    if y < board.y_size {
      *out += "  ";
      for x in 0..board.x_size {
        let loc = Location::get_loc(x, y, board.x_size);
        let s = PlayerIO::color_to_char(board.colors[loc as usize]);
        out.push(s);

        let mut hist_marked = false;
        // `for(int i = (int)size-5; i<hist.moveHistory.size(); i++)`: the comparison converts a
        // negative i to size_t, so with fewer than 5 moves the loop body never runs (PORTING §6.8).
        let size = hist.move_history.len() as i64;
        let mut i: i64 = size - 5;
        while (i as u64) < (size as u64) {
          if i >= 0 && hist.move_history[i as usize].loc == loc {
            *out += &format!("{}", i - (size - 5) + 1);
            hist_marked = true;
            break;
          }
          i += 1;
        }
        if x < board.x_size - 1 && !hist_marked {
          out.push(' ');
        }
      }
    }
    endl(out);
  }
  endl(out);
}

/// upstream: printNNInputGlobal (testnninputs.cpp) static, for inputsVersion 7.
fn print_nn_input_global(out: &mut String, row: &[f32], c: i32) {
  *out += &format!("Channel: {}: {}\n", c, p5(row[c as usize]));
}

/// upstream: finalScoreIfGameEndedNow (testnninputs.cpp) static
fn final_score_if_game_ended_now(base_hist: &BoardHistory, base_board: &Board) -> f64 {
  let mut pla: Player = P_BLACK;
  let mut board = base_board.clone();
  let mut hist = base_hist.clone();
  if !hist.move_history.is_empty() {
    pla = get_opp(hist.move_history[hist.move_history.len() - 1].pla);
  }
  while !hist.is_game_finished {
    hist.make_board_move_assume_legal(&mut board, Board::PASS_LOC, pla, None, false);
    pla = get_opp(pla);
  }

  let score: f64 = hist.final_white_minus_black_score as f64;
  hist.end_and_score_game_now(&board);
  testAssert!(hist.final_white_minus_black_score as f64 == score);
  score
}

/// The `fillRows` lambda of runNNInputsV3V4Tests for version 7.
fn fill_rows(
  hash: &mut Hash128,
  board: &Board,
  hist: &BoardHistory,
  next_pla: Player,
  nn_input_params: &MiscNNInputParams,
  nn_x_len: i32,
  nn_y_len: i32,
  inputs_use_nhwc: bool,
  row_bin: &mut [f32],
  row_global: &mut [f32],
) {
  *hash = NNInputs::get_hash(board, hist, next_pla, nn_input_params);
  NNInputs::fill_row_v7(board, hist, next_pla, nn_input_params, nn_x_len, nn_y_len, inputs_use_nhwc, row_bin, row_global);
}

/// The `allocateRows` lambda for version 7.
fn allocate_rows(nn_x_len: i32, nn_y_len: i32) -> (Vec<f32>, Vec<f32>) {
  (
    vec![0.0f32; (NNInputs::NUM_FEATURES_SPATIAL_V7 * nn_x_len * nn_y_len) as usize],
    vec![0.0f32; NNInputs::NUM_FEATURES_GLOBAL_V7 as usize],
  )
}

fn params(draw_equivalent_wins_for_white: f64) -> MiscNNInputParams {
  let mut p = MiscNNInputParams::default();
  p.draw_equivalent_wins_for_white = draw_equivalent_wins_for_white;
  p
}

// ---------------------------------------------------------------------------------------------
// Test-local SGF subset
// ---------------------------------------------------------------------------------------------

/// upstream: CompactSgf::parse / setupInitialBoardAndHist (sgf.cpp) — test-local subset.
/// Only what the SGF strings of testnninputs.cpp use: one variation, unescaped values, root
/// properties SZ (square or "x:y"), KM, RU, PL, AB/AW/AE (points or rectangles), and B/W moves with
/// pass as `[]` or `[tt]` (when xSize <= 19 || ySize <= 19). cpp/dataio/sgf is an [[ignored]]
/// entry in port-manifest.toml, so this lives here (PORTING.md §13 test-local copies).
struct CompactSgf {
  root: BTreeMap<String, Vec<String>>,
  x_size: i32,
  y_size: i32,
  placements: Vec<Move>,
  moves: Vec<Move>,
}

/// upstream: parseSgfCoord (sgf.cpp) static
fn parse_sgf_coord(c: u8) -> i32 {
  if c.is_ascii_lowercase() {
    return c as i32 - 'a' as i32;
  }
  if c.is_ascii_uppercase() {
    return c as i32 - 'A' as i32 + 26;
  }
  -1
}

impl CompactSgf {
  fn parse(s: &str) -> CompactSgf {
    // Nodes: Vec of property maps, in order (props within a node keep their order of appearance
    // per key; B/W are the only move properties).
    let b = s.as_bytes();
    let mut nodes: Vec<Vec<(String, Vec<String>)>> = Vec::new();
    let mut i = 0;
    assert_eq!(b[0], b'(');
    i += 1;
    while i < b.len() && b[i] != b')' {
      if b[i] == b';' {
        nodes.push(Vec::new());
        i += 1;
        continue;
      }
      if b[i].is_ascii_whitespace() {
        i += 1;
        continue;
      }
      assert!(b[i].is_ascii_uppercase(), "unexpected sgf byte at {}", i);
      let k0 = i;
      while b[i].is_ascii_uppercase() {
        i += 1;
      }
      let key = s[k0..i].to_string();
      let mut vals = Vec::new();
      while i < b.len() && b[i] == b'[' {
        let v0 = i + 1;
        let v1 = v0 + s[v0..].find(']').expect("unterminated value");
        vals.push(s[v0..v1].to_string());
        i = v1 + 1;
      }
      nodes.last_mut().expect("property before node").push((key, vals));
    }
    let root: BTreeMap<String, Vec<String>> = nodes[0].iter().cloned().collect();

    // upstream: Sgf::getXYSize
    let (x_size, y_size) = match root.get("SZ") {
      None => (19, 19),
      Some(v) => {
        let sz = &v[0];
        if let Some((a, c)) = sz.split_once(':') {
          (a.parse::<i32>().unwrap(), c.parse::<i32>().unwrap())
        } else {
          let n = sz.parse::<i32>().unwrap();
          (n, n)
        }
      }
    };

    // upstream: SgfNode::accumPlacements (AB, then AW, then AE)
    let mut placements = Vec::new();
    for (key, color) in [("AB", P_BLACK), ("AW", P_WHITE), ("AE", C_EMPTY)] {
      if let Some(elts) = root.get(key) {
        for e in elts {
          let e = e.as_bytes();
          let (x1, y1, x2, y2) = if e.len() == 5 && e[2] == b':' {
            (parse_sgf_coord(e[0]), parse_sgf_coord(e[1]), parse_sgf_coord(e[3]), parse_sgf_coord(e[4]))
          } else {
            assert_eq!(e.len(), 2);
            (parse_sgf_coord(e[0]), parse_sgf_coord(e[1]), parse_sgf_coord(e[0]), parse_sgf_coord(e[1]))
          };
          for x in x1..=x2 {
            for y in y1..=y2 {
              placements.push(Move::new(Location::get_loc(x, y, x_size), color));
            }
          }
        }
      }
    }

    // upstream: Sgf::getMoves / SgfNode::accumMoves (parseSgfLocOrPass)
    let mut moves = Vec::new();
    for node in &nodes {
      for (key, vals) in node {
        let pla = match key.as_str() {
          "B" => P_BLACK,
          "W" => P_WHITE,
          _ => continue,
        };
        for v in vals {
          let loc = if v.is_empty() || (v == "tt" && (x_size <= 19 || y_size <= 19)) {
            Board::PASS_LOC
          } else {
            let vb = v.as_bytes();
            assert_eq!(vb.len(), 2);
            let (x, y) = (parse_sgf_coord(vb[0]), parse_sgf_coord(vb[1]));
            assert!(x >= 0 && x < x_size && y >= 0 && y < y_size);
            Location::get_loc(x, y, x_size)
          };
          moves.push(Move::new(loc, pla));
        }
      }
    }
    CompactSgf { root, x_size, y_size, placements, moves }
  }

  /// upstream: CompactSgf::getRulesOrFailAllowUnspecified (sgf.cpp)
  fn get_rules_or_fail_allow_unspecified(&self, default_rules: &Rules) -> Rules {
    let mut rules: Rules;
    match self.root.get("RU") {
      None => rules = *default_rules,
      Some(v) => {
        // upstream: SgfNode::getRulesFromRUTagOrFail
        let mut parsed = Rules::new();
        let suc = Rules::try_parse_rules(&v[0], &mut parsed).unwrap();
        assert!(suc, "Could not parse rules in sgf: {}", v[0]);
        rules = parsed;
      }
    }
    if let Some(v) = self.root.get("KM") {
      // upstream: SgfNode::getKomiOrDefault (the strings here are int/half-int komis)
      let mut komi: f32 = 0.0;
      assert!(katago_core::core::global::Global::try_string_to_float(&v[0], &mut komi));
      assert!(Rules::komi_is_int_or_half_int(komi));
      rules.komi = komi;
    }
    rules
  }

  /// upstream: CompactSgf::setupInitialBoardAndHist (sgf.cpp)
  fn setup_initial_board_and_hist(&self, initial_rules: &Rules, board: &mut Board, next_pla: &mut Player, hist: &mut BoardHistory) {
    // getPLSpecifiedColor
    let pl_player: Color = match self.root.get("PL").map(|v| v[0].to_lowercase()) {
      Some(s) if s == "b" || s == "black" => C_BLACK,
      Some(s) if s == "w" || s == "white" => C_WHITE,
      _ => C_EMPTY,
    };
    if pl_player == P_BLACK || pl_player == P_WHITE {
      *next_pla = pl_player;
    } else {
      let mut has_black = false;
      let mut all_black = true;
      for i in 0..self.placements.len() {
        if self.placements[i].pla == P_BLACK {
          has_black = true;
        } else {
          all_black = false;
        }
      }
      if has_black && all_black {
        *next_pla = P_WHITE;
      } else {
        *next_pla = P_BLACK;
      }
    }

    // Override with the actual color of the move, if it exists
    if !self.moves.is_empty() {
      *next_pla = self.moves[0].pla;
    }

    *board = Board::new_x_y(self.x_size, self.y_size).unwrap();
    let suc = board.set_stones_fail_if_no_libs(&self.placements);
    assert!(suc, "setupInitialBoardAndHist: initial board position contains invalid stones or zero-liberty stones");
    *hist = BoardHistory::new_full(board, *next_pla, initial_rules, 0);
    if hist.initial_turn_number < board.num_stones_on_board() as i64 {
      hist.initial_turn_number = board.num_stones_on_board() as i64;
    }
  }
}

/// Board, nextPla, hist after `setupInitialBoardAndHist` and playing every SGF move.
fn setup_and_play(sgf_str: &str, default_rules: &Rules) -> (Board, Player, BoardHistory) {
  let sgf = CompactSgf::parse(sgf_str);
  let mut board = Board::new();
  let mut next_pla: Player = P_BLACK;
  let mut hist = BoardHistory::new();
  let initial_rules = sgf.get_rules_or_fail_allow_unspecified(default_rules);
  sgf.setup_initial_board_and_hist(&initial_rules, &mut board, &mut next_pla, &mut hist);
  let moves = &sgf.moves;
  for i in 0..moves.len() {
    hist.make_board_move_assume_legal(&mut board, moves[i].loc, moves[i].pla, None, false);
    next_pla = get_opp(moves[i].pla);
  }
  (board, next_pla, hist)
}

// ---------------------------------------------------------------------------------------------
// Sections
// ---------------------------------------------------------------------------------------------

/// Sections "Basic", "Ko", "7x7", "7x7 embedded in 9x9": an SGF, played out, then `run` for
/// NHWC and NCHW (which must agree), printing either all channels or only `only_channel`.
fn sgf_section(
  name: &str,
  slug: &str,
  sgf_str: &str,
  nn_x_len: i32,
  nn_y_len: i32,
  draw_equivalent_wins_for_white: f64,
  only_channel: Option<i32>,
) {
  let (board, next_pla, hist) = setup_and_play(sgf_str, &Rules::get_tromp_taylorish());
  let (mut row_bin, mut row_global) = allocate_rows(nn_x_len, nn_y_len);
  let mut run = |inputs_use_nhwc: bool| -> String {
    let mut out = String::new();
    let mut hash = Hash128::default();
    let nn_input_params = params(draw_equivalent_wins_for_white);
    fill_rows(&mut hash, &board, &hist, next_pla, &nn_input_params, nn_x_len, nn_y_len, inputs_use_nhwc, &mut row_bin, &mut row_global);
    out += &format!("{}\n", hash);
    match only_channel {
      None => {
        for c in 0..NNInputs::NUM_FEATURES_SPATIAL_V7 {
          print_nn_input_hw_and_board(&mut out, &board, &hist, nn_x_len, nn_y_len, inputs_use_nhwc, &row_bin, c);
        }
      }
      Some(c) => print_nn_input_hw_and_board(&mut out, &board, &hist, nn_x_len, nn_y_len, inputs_use_nhwc, &row_bin, c),
    }
    for c in 0..NNInputs::NUM_FEATURES_GLOBAL_V7 {
      print_nn_input_global(&mut out, &row_global, c);
    }
    out
  };
  let actual_nhwc = run(true);
  let actual_nchw = run(false);
  TestCommon::expect(name, &actual_nhwc, &actual_nchw);
  let cout = format!("VERSION {}\n{}\n", VERSION, actual_nhwc);
  TestCommon::expect(name, &cout, &expected(slug, 0));
}

wasm_test! {
fn nn_inputs_sgf_sections() {
  init();
  let mut cases = Cases::new("testnninputs::nn_inputs_sgf_sections");
  sgf_section(
    "NN Inputs V3V4V5V6 Basic",
    "nn-inputs-v3v4v5v6-basic",
    "(;FF[4]KM[7.5];B[pd];W[pq];B[dq];W[dd];B[qo];W[pl];B[qq];W[qr];B[pp];W[rq];B[oq];W[qp];B[pr];W[qq];B[oo];W[ro];B[qn];W[do];B[dl];W[gp];B[eo];W[en];B[fo];W[dp];B[eq];W[cq];B[cr];W[br];B[dn];W[bp];B[cn];W[ep];B[fp];W[fq];B[gq];W[fr];B[gr];W[er];B[hp];W[go];B[fn];W[ho];B[ip];W[io];B[jp];W[jo];B[lp];W[kp];B[kq];W[ko];B[lq];W[ir];B[hq];W[jq];B[jr];W[em];B[gm];W[el];B[hl];W[kl];B[ek];W[fk];B[ej];W[fl];B[fj];W[gk];B[ik];W[gj];B[jj];W[dm];B[lk];W[mm];B[nl];W[nm];B[om];W[ol];B[nk];W[ll];B[kk];W[jl];B[im];W[jk];B[ij];W[kj];B[mk];W[ki];B[ih];W[jh];B[ig];W[jg];B[if];W[oi];B[mi];W[mh];B[lh];W[li];B[nh];W[mj];B[ni];W[nj];B[oj];W[lj];B[ok];W[oh];B[ng];W[pj];B[ji];W[kh];B[jf];W[lg];B[cm];W[cl];B[dk];W[bl];B[bk];W[bn];B[ck];W[bm];B[cc];W[cd];B[dc];W[ec];B[eb];W[fb];B[fc];W[ed];B[gb];W[bc];B[cb];W[cg];B[be];W[bd];B[bg];W[bh];B[cf];W[df];B[ch];W[dg];B[bi];W[qd];B[qc];W[rc];B[rd];W[qe];B[re];W[rb];B[pc];W[qb];B[qf];W[ff];B[sc];W[pb];B[bo];W[ob];B[nc];W[nb];B[mb];W[mc];B[lb])",
    19,
    19,
    0.2,
    None,
  );
  cases.ran();
  sgf_section(
    "NN Inputs V3V4V5V6 Ko",
    "nn-inputs-v3v4v5v6-ko",
    "(;FF[4]KM[0.5];B[rj];W[ri];B[si];W[rh];B[sh];W[sg];B[rk];W[sk];B[sl];W[sj];B[eg];W[fg];B[ff];W[gf];B[fh];W[gh];B[gg];W[hg];B[si];W[fg];B[sh];W[sk];B[gg])",
    19,
    19,
    0.3,
    Some(6),
  );
  cases.ran();
  let sgf7 = "(;GM[1]FF[4]CA[UTF-8]ST[2]RU[Tromp-Taylor]SZ[7]HA[3]KM[-4.50]PW[White]PB[Black]AB[fb][bf][ff];W[ed];B[ee];W[de];B[dd];W[ef];B[df];W[fe];B[ce];W[dc];B[ee];W[eg];B[fd];W[de])";
  sgf_section("NN Inputs V3V4V5V6 7x7", "nn-inputs-v3v4v5v6-7x7", sgf7, 7, 7, 0.5, None);
  cases.ran();
  sgf_section("NN Inputs V3V4V5V6 7x7 embedded in 9x9", "nn-inputs-v3v4v5v6-7x7-embedded-in-9x9", sgf7, 9, 9, 0.8, None);
  cases.ran();
}
}

wasm_test! {
/// upstream: "NN Inputs V3V4V6 Area Komi"
fn nn_inputs_area_komi() {
  init();
  let mut cases = Cases::new("testnninputs::nn_inputs_area_komi");
  let name = "NN Inputs V3V4V6 Area Komi";
  let mut out = String::new();
  let board = Board::parse_board(7, 7, &fixture(INPUTS, "nn-inputs-v3v4v6-area-komi.input0.txt"), '\n').unwrap();

  let mut next_pla: Player = P_BLACK;
  let mut initial_rules = Rules::get_tromp_taylorish();
  initial_rules.komi = 2.0;
  let mut hist = BoardHistory::new_full(&board, next_pla, &initial_rules, 0);

  let nn_x_len = 7;
  let nn_y_len = 7;
  let draw_equivalent_wins_for_white = 0.3;
  let (mut row_bin, mut row_global) = allocate_rows(nn_x_len, nn_y_len);

  let inputs_use_nhwc = true;
  let mut hash = Hash128::default();
  let nn_input_params = params(draw_equivalent_wins_for_white);
  fill_rows(&mut hash, &board, &hist, next_pla, &nn_input_params, nn_x_len, nn_y_len, inputs_use_nhwc, &mut row_bin, &mut row_global);

  print_nn_input_hw_and_board(&mut out, &board, &hist, nn_x_len, nn_y_len, inputs_use_nhwc, &row_bin, 18);
  print_nn_input_hw_and_board(&mut out, &board, &hist, nn_x_len, nn_y_len, inputs_use_nhwc, &row_bin, 19);
  for c in 0..NNInputs::NUM_FEATURES_GLOBAL_V7 {
    print_nn_input_global(&mut out, &row_global, c);
  }

  next_pla = P_WHITE;
  hist.clear(&board, next_pla, &initial_rules, 0);
  fill_rows(&mut hash, &board, &hist, next_pla, &nn_input_params, nn_x_len, nn_y_len, inputs_use_nhwc, &mut row_bin, &mut row_global);
  for c in 0..NNInputs::NUM_FEATURES_GLOBAL_V7 {
    print_nn_input_global(&mut out, &row_global, c);
  }

  next_pla = P_BLACK;
  initial_rules.komi = 1.0;
  hist.clear(&board, next_pla, &initial_rules, 0);
  fill_rows(&mut hash, &board, &hist, next_pla, &nn_input_params, nn_x_len, nn_y_len, inputs_use_nhwc, &mut row_bin, &mut row_global);
  for c in 0..NNInputs::NUM_FEATURES_GLOBAL_V7 {
    print_nn_input_global(&mut out, &row_global, c);
  }

  let cout = format!("VERSION {}\n{}\n", VERSION, out);
  TestCommon::expect(name, &cout, &expected("nn-inputs-v3v4v6-area-komi", 0));
  cases.ran();
}
}

wasm_test! {
/// upstream: "NN Inputs V3V4V5V6 Rules" (board sizes 7 then 6: two VERSION 7 chunks)
fn nn_inputs_rules() {
  init();
  let mut cases = Cases::new("testnninputs::nn_inputs_rules");
  let name = "NN Inputs V3V4V5V6 Rules";
  let mut k = 0;
  let mut size = 7;
  while size >= 6 {
    let board = Board::new_x_y(size, size).unwrap();
    let next_pla: Player = P_BLACK;
    let mut rules: Vec<Rules> = Vec::new();
    for (tax, score) in [
      (Rules::TAX_NONE, Rules::SCORING_AREA),
      (Rules::TAX_SEKI, Rules::SCORING_AREA),
      (Rules::TAX_NONE, Rules::SCORING_TERRITORY),
      (Rules::TAX_SEKI, Rules::SCORING_TERRITORY),
      (Rules::TAX_ALL, Rules::SCORING_AREA),
      (Rules::TAX_ALL, Rules::SCORING_TERRITORY),
    ] {
      let mut komi = 1.0f32;
      for ko in [Rules::KO_SIMPLE, Rules::KO_POSITIONAL, Rules::KO_SITUATIONAL] {
        for (suic, button) in [(false, false), (true, false), (false, true), (true, true)] {
          rules.push(Rules::new_full(ko, score, tax, suic, button, Rules::WHB_ZERO, false, komi));
          komi += 0.5;
        }
      }
    }
    assert_eq!(rules.len(), 72);

    let nn_x_len = size;
    let nn_y_len = size;
    let draw_equivalent_wins_for_white = 0.47;

    let mut out = String::new();
    let (mut row_bin, mut row_global) = allocate_rows(nn_x_len, nn_y_len);
    let mut i = 0;
    while i < rules.len() {
      for c in 0..NNInputs::NUM_FEATURES_GLOBAL_V7 {
        for j in 0..24 {
          testAssert!(i + j < rules.len());
          let hist = BoardHistory::new_full(&board, next_pla, &rules[i + j], 0);
          let inputs_use_nhwc = true;
          let mut hash = Hash128::default();
          let nn_input_params = params(draw_equivalent_wins_for_white);
          fill_rows(&mut hash, &board, &hist, next_pla, &nn_input_params, nn_x_len, nn_y_len, inputs_use_nhwc, &mut row_bin, &mut row_global);
          out += &p5(row_global[c as usize]);
          out += " ";
        }
        endl(&mut out);
      }
      endl(&mut out);
      i += 24;
    }
    let cout = format!("VERSION {}\n{}\n", VERSION, out);
    TestCommon::expect(name, &cout, &expected("nn-inputs-v3v4v5v6-rules", k));
    k += 1;
    cases.ran();
    size -= 1;
  }
}
}

wasm_test! {
/// upstream: "NN Inputs V3V4V6 Ko Prohib and pass hist and whitebonus and encorestart"
fn nn_inputs_ko_prohib() {
  init();
  let mut cases = Cases::new("testnninputs::nn_inputs_ko_prohib");
  let name = "NN Inputs V3V4V6 Ko Prohib and pass hist and whitebonus and encorestart";
  //Immediately enters encore via b0 pass w1 pass. Through w19, sets up various ko shapes. Then starts ko captures. b26 pass b27 pass switches to second encore.
  let sgf_str = "(;GM[1]FF[4]SZ[6]KM[0.00];B[];W[];B[ab];W[bb];B[ba];W[ca];B[ec];W[ed];B[fd];W[fe];B[fb];W[dc];B[db];W[ae];B[ea];W[bf];B[be];W[ad];B[cf];W[dd];B[af];W[aa];B[];W[fc];B[bd];W[eb];B[];W[];B[ec];W[bf];B[ac];W[eb];B[af];W[eb])";
  let sgf = CompactSgf::parse(sgf_str);
  let moves = &sgf.moves;

  let mut out = String::new();
  let mut board = Board::new();
  let mut next_pla: Player = P_BLACK;
  let mut hist = BoardHistory::new();
  let mut initial_rules = Rules::new_full(Rules::KO_SIMPLE, Rules::SCORING_TERRITORY, Rules::TAX_SEKI, false, false, Rules::WHB_ZERO, false, 0.0f32);
  initial_rules = sgf.get_rules_or_fail_allow_unspecified(&initial_rules);
  sgf.setup_initial_board_and_hist(&initial_rules, &mut board, &mut next_pla, &mut hist);

  let nn_x_len = 6;
  let nn_y_len = 6;
  let draw_equivalent_wins_for_white = 0.0;
  let (mut row_bin, mut row_global) = allocate_rows(nn_x_len, nn_y_len);

  for i in 0..moves.len() {
    testAssert!(hist.is_legal(&board, moves[i].loc, moves[i].pla));
    hist.make_board_move_assume_legal(&mut board, moves[i].loc, moves[i].pla, None, false);
    next_pla = get_opp(moves[i].pla);

    if i == 24 || i == 25 || i == 26 || i == 30 || i == 31 || i == 32 || i == 33 || i == 34 {
      out += &format!("Move {}\n", i);
      for y in 0..board.y_size {
        for x in 0..board.x_size {
          out += &format!("{}", pb(hist.is_legal(&board, Location::get_loc(x, y, board.x_size), next_pla)));
        }
        out += " ";
      }
      endl(&mut out);
      let inputs_use_nhwc = true;
      let mut hash = Hash128::default();
      let nn_input_params = params(draw_equivalent_wins_for_white);
      fill_rows(&mut hash, &board, &hist, next_pla, &nn_input_params, nn_x_len, nn_y_len, inputs_use_nhwc, &mut row_bin, &mut row_global);
      out += "Pass Hist Channels: ";
      for c in 0..5 {
        out += &p5(row_global[c]);
        out += " ";
      }
      endl(&mut out);
      out += &format!("Selfkomi channel times 15: {}\n", p5(row_global[5] * 15.0f32));
      // version >= 6
      out += &format!("EncorePhase channel 12,13: {} {}\n", p5(row_global[12]), p5(row_global[13]));
      out += &format!("PassWouldEndPhase channel 14: {}\n", p5(row_global[14]));
      print_nn_input_hw_and_board(&mut out, &board, &hist, nn_x_len, nn_y_len, inputs_use_nhwc, &row_bin, 7);
      print_nn_input_hw_and_board(&mut out, &board, &hist, nn_x_len, nn_y_len, inputs_use_nhwc, &row_bin, 8);
      print_nn_input_hw_and_board(&mut out, &board, &hist, nn_x_len, nn_y_len, inputs_use_nhwc, &row_bin, 20);
      print_nn_input_hw_and_board(&mut out, &board, &hist, nn_x_len, nn_y_len, inputs_use_nhwc, &row_bin, 21);
      cases.ran();
    }
  }

  let cout = format!("VERSION {}\n{}\n", VERSION, out);
  TestCommon::expect(name, &cout, &expected("nn-inputs-v3v4v6-ko-prohib-and-pass-hist-and-whitebonus-and-encorestart", 0));
}
}

wasm_test! {
/// upstream: "NN Inputs V3V4V6 some other test positions"
fn nn_inputs_some_other_test_positions() {
  init();
  let mut cases = Cases::new("testnninputs::nn_inputs_some_other_test_positions");
  let name = "NN Inputs V3V4V6 some other test positions";
  let sgf_str = "(;FF[4]GM[1]SZ[13]PB[s75411712-d5152283-b8c128]PW[s78621440-d5365731-b8c128]HA[0]KM[7.5]RU[koPOSITIONALscoreAREAsui0]RE[B+11.5];B[ck];W[lb];B[ke];W[ld];B[jd];W[kc];B[jc];W[jb];B[ib];W[kk];B[ki];W[kh];B[ja];W[le];B[ic];W[kf];B[lj];W[li];B[kj];W[lk];B[jk];W[jl];B[ik];W[mj];B[kb];W[jj];B[ji];W[ij];B[ii];W[hj];B[lh];W[mi];B[kg];W[jg];B[jh];W[lg];B[hk];W[hi];B[mh];W[gk];B[mk];W[il];B[jf];W[lf];B[ig];W[cc];B[dc];W[cd];B[ed];W[kd];B[dj];W[el];B[eg];W[de];B[ee];W[ec];B[je];W[db];B[fc];W[eb];B[bj];W[fd];B[gc];W[cl];B[df];W[dd];B[cf];W[dl];B[gh];W[fk];B[la];W[hh];B[hg];W[fi];B[gg];W[mc];B[bk];W[fb];B[gb];W[ei];B[gi];W[fe];B[ef];W[ej];B[gj];W[hl];B[bh];W[mg];B[be];W[bd];B[ad];W[bb];B[ae];W[di];B[me];W[ci];B[bi];W[bl];B[ab];W[ba];B[ac];W[ml];B[ga];W[fa];B[al];W[bc];B[bf];W[mj];B[mi];W[mb];B[ge];W[mk];B[dk];W[md];B[ek];W[fj];B[jb];W[fh];B[ff];W[bm];B[ka];W[ce];B[ak];W[cj];B[ch];W[];B[id];W[fl];B[hc];W[am];B[ik];W[jk];B[ma];W[];B[mm];W[gl];B[aa];W[ca];B[dh];W[fg];B[];W[lm];B[bg];W[];B[hd];W[];B[ag];W[];B[hf];W[];B[gd];W[];B[ih];W[];B[li];W[];B[hb];W[];B[af];W[];B[ia];W[];B[kl];W[];B[])";
  let sgf = CompactSgf::parse(sgf_str);
  let mut out = String::new();
  let mut board = Board::new();
  let mut next_pla: Player = P_BLACK;
  let mut hist = BoardHistory::new();
  let mut initial_rules = Rules::new();
  initial_rules = sgf.get_rules_or_fail_allow_unspecified(&initial_rules);
  sgf.setup_initial_board_and_hist(&initial_rules, &mut board, &mut next_pla, &mut hist);
  let moves = &sgf.moves;

  let nn_x_len = 13;
  let nn_y_len = 13;
  let draw_equivalent_wins_for_white = 0.0;
  let (mut row_bin, mut row_global) = allocate_rows(nn_x_len, nn_y_len);

  for i in 0..moves.len() {
    testAssert!(hist.is_legal(&board, moves[i].loc, moves[i].pla));
    hist.make_board_move_assume_legal(&mut board, moves[i].loc, moves[i].pla, None, false);
    next_pla = get_opp(moves[i].pla);

    if i == 163 {
      out += &format!("Move {}\n", i);
      let inputs_use_nhwc = true;
      let mut hash = Hash128::default();
      let nn_input_params = params(draw_equivalent_wins_for_white);
      fill_rows(&mut hash, &board, &hist, next_pla, &nn_input_params, nn_x_len, nn_y_len, inputs_use_nhwc, &mut row_bin, &mut row_global);
      print_nn_input_hw_and_board(&mut out, &board, &hist, nn_x_len, nn_y_len, inputs_use_nhwc, &row_bin, 18);
      print_nn_input_hw_and_board(&mut out, &board, &hist, nn_x_len, nn_y_len, inputs_use_nhwc, &row_bin, 19);
      cases.ran();
    }
  }

  let cout = format!("VERSION {}\n{}\n", VERSION, out);
  TestCommon::expect(name, &cout, &expected("nn-inputs-v3v4v6-some-other-test-positions", 0));
}
}

wasm_test! {
/// upstream: "NN Inputs V6 Area Feature and Komi"
fn nn_inputs_area_feature_and_komi() {
  init();
  let mut cases = Cases::new("testnninputs::nn_inputs_area_feature_and_komi");
  let name = "NN Inputs V6 Area Feature and Komi";
  let mut cout = format!("VERSION {}\n", VERSION);

  let nn_x_len = 7;
  let nn_y_len = 7;
  let draw_equivalent_wins_for_white = 0.5;
  let (mut row_bin, mut row_global) = allocate_rows(nn_x_len, nn_y_len);

  for go_to_encore2 in 0..=1 {
    let scoring_rules: [i32; 4] = [Rules::SCORING_AREA, Rules::SCORING_AREA, Rules::SCORING_TERRITORY, Rules::SCORING_TERRITORY];
    let tax_rules: [i32; 4] = [Rules::TAX_NONE, Rules::TAX_SEKI, Rules::TAX_NONE, Rules::TAX_SEKI];
    for which_rules in 0..4 {
      let mut board = Board::parse_board(7, 7, &fixture(INPUTS, "nn-inputs-v6-area-feature-and-komi.input0.txt"), '\n').unwrap();
      let mut rules = Rules::new();
      rules.ko_rule = Rules::KO_POSITIONAL;
      rules.scoring_rule = scoring_rules[which_rules];
      rules.komi = 6.5f32;
      rules.multi_stone_suicide_legal = false;
      rules.tax_rule = tax_rules[which_rules];
      let mut hist = BoardHistory::new_full(&board, P_WHITE, &rules, 0);

      let mut run = |board: &Board, hist: &BoardHistory, inputs_use_nhwc: bool| -> String {
        let mut out = String::new();
        let next_pla: Player =
          if !hist.move_history.is_empty() { get_opp(hist.move_history[hist.move_history.len() - 1].pla) } else { hist.initial_pla };
        let mut hash = Hash128::default();
        let nn_input_params = params(draw_equivalent_wins_for_white);
        fill_rows(&mut hash, board, hist, next_pla, &nn_input_params, nn_x_len, nn_y_len, inputs_use_nhwc, &mut row_bin, &mut row_global);
        out += &format!("{}\n", hash);
        print_nn_input_global(&mut out, &row_global, 5);
        print_nn_input_hw_and_board(&mut out, board, hist, nn_x_len, nn_y_len, inputs_use_nhwc, &row_bin, 18);
        print_nn_input_hw_and_board(&mut out, board, hist, nn_x_len, nn_y_len, inputs_use_nhwc, &row_bin, 19);
        out
      };
      let print_scoring = |cout: &mut String, board: &Board, hist: &BoardHistory| {
        let b = board.clone();
        let mut h = hist.clone();
        let mut area: [Color; Board::MAX_ARR_SIZE as usize] = [0; Board::MAX_ARR_SIZE as usize];
        let mut scoring: [f32; Board::MAX_ARR_SIZE as usize] = [0.0; Board::MAX_ARR_SIZE as usize];
        h.end_and_score_game_now_area(&b, &mut area);
        NNInputs::fill_scoring(&b, &area, false, &mut scoring);
        for y in 0..board.y_size {
          for x in 0..board.x_size {
            let loc = Location::get_loc(x, y, board.x_size);
            *cout += &strprintf!("%4.0f", 100.0f32 * scoring[loc as usize]);
            *cout += " ";
          }
          endl(cout);
        }
        endl(cout);
      };
      let mut run_both = |cout: &mut String, board: &Board, hist: &BoardHistory| {
        let actual_nhwc = run(board, hist, true);
        let actual_nchw = run(board, hist, false);
        TestCommon::expect(name, &actual_nhwc, &actual_nchw);
        *cout += &format!("Rules: {}\n", hist.rules);
        *cout += &format!("Komi and Bonus: {} {}\n", p6(hist.rules.komi as f64), p6(hist.white_bonus_score as f64));
        *cout += &format!("Encore phase: {}\n", hist.encore_phase);
        for i in 0..hist.move_history.len() {
          *cout += &format!("{} ", Location::to_string(hist.move_history[i].loc, board));
        }
        *cout += &actual_nhwc;
        print_scoring(cout, board, hist);
      };

      cout += "=========================================== \n";
      cout += &format!("goToEncore2 = {}\n", go_to_encore2);

      run_both(&mut cout, &board, &hist);
      if go_to_encore2 != 0 {
        hist.make_board_move_assume_legal(&mut board, Board::PASS_LOC, P_WHITE, None, false);
        hist.make_board_move_assume_legal(&mut board, Board::PASS_LOC, P_BLACK, None, false);
        run_both(&mut cout, &board, &hist);
        hist.make_board_move_assume_legal(&mut board, Board::PASS_LOC, P_WHITE, None, false);
        hist.make_board_move_assume_legal(&mut board, Board::PASS_LOC, P_BLACK, None, false);
        run_both(&mut cout, &board, &hist);
      }
      for (x, y, pla) in [(6, 5, P_WHITE), (5, 6, P_BLACK), (0, 4, P_WHITE), (6, 0, P_BLACK), (1, 0, P_WHITE), (4, 5, P_BLACK), (5, 4, P_WHITE)] {
        let loc = Location::get_loc(x, y, board.x_size);
        hist.make_board_move_assume_legal(&mut board, loc, pla, None, false);
        run_both(&mut cout, &board, &hist);
      }
      endl(&mut cout);
      cases.ran();
    }
  }
  TestCommon::expect(name, &cout, &expected("nn-inputs-v6-area-feature-and-komi", 0));
}
}

wasm_test! {
/// upstream: "NN Inputs V6 Pass history"
fn nn_inputs_pass_history() {
  init();
  let mut cases = Cases::new("testnninputs::nn_inputs_pass_history");
  let name = "NN Inputs V6 Pass history";
  let mut cout = format!("VERSION {}\n", VERSION);

  let nn_x_len = 9;
  let nn_y_len = 1;
  let draw_equivalent_wins_for_white = 0.5;
  let (mut row_bin, mut row_global) = allocate_rows(nn_x_len, nn_y_len);

  let mut board = Board::new_x_y(9, 1).unwrap();
  let mut next_pla: Player = P_BLACK;
  let initial_rules = Rules::get_simple_territory();
  let mut hist = BoardHistory::new_full(&board, next_pla, &initial_rules, 0);

  let mut run = |board: &Board, hist: &BoardHistory, next_pla: Player, inputs_use_nhwc: bool| -> String {
    let mut out = String::new();
    let mut hash = Hash128::default();
    let nn_input_params = params(draw_equivalent_wins_for_white);
    fill_rows(&mut hash, board, hist, next_pla, &nn_input_params, nn_x_len, nn_y_len, inputs_use_nhwc, &mut row_bin, &mut row_global);
    print_nn_input_hw_and_board(&mut out, board, hist, nn_x_len, nn_y_len, inputs_use_nhwc, &row_bin, 9);
    print_nn_input_hw_and_board(&mut out, board, hist, nn_x_len, nn_y_len, inputs_use_nhwc, &row_bin, 10);
    print_nn_input_hw_and_board(&mut out, board, hist, nn_x_len, nn_y_len, inputs_use_nhwc, &row_bin, 11);
    print_nn_input_global(&mut out, &row_global, 0);
    print_nn_input_global(&mut out, &row_global, 1);
    print_nn_input_global(&mut out, &row_global, 2);
    out
  };

  let locs = Location::parse_sequence("pass A1 D1 F1 H1 E1 G1 B1 J1 F1 J1 G1 C1 E1 B1 pass pass H1 J1 E1 G1 pass pass A1 pass D1 C1 B1", &board).unwrap();
  for i in 0..locs.len() {
    let actual_nhwc = run(&board, &hist, next_pla, true);
    let actual_nchw = run(&board, &hist, next_pla, false);
    TestCommon::expect(name, &actual_nhwc, &actual_nchw);
    for k in 0..hist.move_history.len() {
      cout += &format!("{} ", Location::to_string(hist.move_history[k].loc, &board));
    }
    endl(&mut cout);
    cout += &actual_nhwc;
    cout += "-----\n";
    hist.make_board_move_assume_legal(&mut board, locs[i], next_pla, None, false);
    next_pla = get_opp(next_pla);
    cases.ran();
  }
  TestCommon::expect(name, &cout, &expected("nn-inputs-v6-pass-history", 0));
}
}

wasm_test! {
/// upstream: "NN Inputs SelfKomi Handicap White Bonus" (version 7 only upstream)
fn nn_inputs_self_komi_handicap_white_bonus() {
  init();
  let mut cases = Cases::new("testnninputs::nn_inputs_self_komi_handicap_white_bonus");
  let name = "NN Inputs SelfKomi Handicap White Bonus";
  let size = 7;
  let rules: [Rules; 5] = [
    Rules::new_full(Rules::KO_POSITIONAL, Rules::SCORING_AREA, Rules::TAX_NONE, false, false, Rules::WHB_ZERO, false, 3.0f32),
    Rules::new_full(Rules::KO_POSITIONAL, Rules::SCORING_AREA, Rules::TAX_NONE, true, false, Rules::WHB_N_MINUS_ONE, false, 3.0f32),
    Rules::new_full(Rules::KO_POSITIONAL, Rules::SCORING_AREA, Rules::TAX_NONE, false, true, Rules::WHB_N, false, 3.0f32),
    Rules::new_full(Rules::KO_POSITIONAL, Rules::SCORING_TERRITORY, Rules::TAX_NONE, true, false, Rules::WHB_N_MINUS_ONE, false, 3.0f32),
    Rules::new_full(Rules::KO_POSITIONAL, Rules::SCORING_TERRITORY, Rules::TAX_NONE, false, true, Rules::WHB_N, false, 3.0f32),
  ];
  let nn_x_len = size;
  let nn_y_len = size;
  let draw_equivalent_wins_for_white = 0.5;

  let mut cout = format!("VERSION {}\n", VERSION);
  let (mut row_bin, mut row_global) = allocate_rows(nn_x_len, nn_y_len);

  let mut test = |cout: &mut String, board: &Board, hist: &BoardHistory, next_pla: Player| {
    let inputs_use_nhwc = true;
    let mut hash = Hash128::default();
    let b = board.clone();
    let nn_input_params = params(draw_equivalent_wins_for_white);
    fill_rows(&mut hash, &b, hist, next_pla, &nn_input_params, nn_x_len, nn_y_len, inputs_use_nhwc, &mut row_bin, &mut row_global);
    for c in 0..NNInputs::NUM_FEATURES_GLOBAL_V7 as usize {
      *cout += &format!("{} ", p6(row_global[c] as f64));
    }
    endl(cout);
  };
  let final_score = |cout: &mut String, hist: &BoardHistory, board: &Board| {
    *cout += &format!("Final score: {}\n", p6(final_score_if_game_ended_now(hist, board)));
  };

  for i in 0..rules.len() {
    let mut board = Board::new_x_y(size, size).unwrap();
    let next_pla: Player = P_BLACK;
    let initial_rules = rules[i];
    let mut hist = BoardHistory::new_full(&board, next_pla, &initial_rules, 0);
    cout += "----------------------------------------\n";
    cout += "Black makes 3 moves in a row\n";
    if i >= 3 {
      cout += "Set assumeMultipleStartingBlackMovesAreHandicap\n";
      hist.set_assume_multiple_starting_black_moves_are_handicap(true);
    }
    for (x, y) in [(3, 3), (3, 2), (3, 1)] {
      let loc = Location::get_loc(x, y, board.x_size);
      hist.make_board_move_assume_legal(&mut board, loc, P_BLACK, None, false);
      hist.print_debug_info(&mut cout, &board);
      test(&mut cout, &board, &hist, next_pla);
      final_score(&mut cout, &hist, &board);
    }

    cout += "\nSet assumeMultipleStartingBlackMovesAreHandicap\n";
    hist.set_assume_multiple_starting_black_moves_are_handicap(true);
    hist.print_debug_info(&mut cout, &board);
    test(&mut cout, &board, &hist, next_pla);
    final_score(&mut cout, &hist, &board);

    cout += "\nOne more move\n";
    let loc = Location::get_loc(5, 1, board.x_size);
    hist.make_board_move_assume_legal(&mut board, loc, P_BLACK, None, false);
    hist.print_debug_info(&mut cout, &board);
    test(&mut cout, &board, &hist, next_pla);
    final_score(&mut cout, &hist, &board);

    cout += "\nReclear history\n";
    hist.clear(&board, next_pla, &initial_rules, 0);
    hist.print_debug_info(&mut cout, &board);
    test(&mut cout, &board, &hist, next_pla);
    final_score(&mut cout, &hist, &board);

    cout += "\nOne more move\n";
    let loc = Location::get_loc(2, 1, board.x_size);
    hist.make_board_move_assume_legal(&mut board, loc, P_BLACK, None, false);
    hist.print_debug_info(&mut cout, &board);
    test(&mut cout, &board, &hist, next_pla);
    final_score(&mut cout, &hist, &board);

    cout += "\nSet assumeMultipleStartingBlackMovesAreHandicap\n";
    hist.set_assume_multiple_starting_black_moves_are_handicap(true);
    hist.print_debug_info(&mut cout, &board);
    test(&mut cout, &board, &hist, next_pla);
    final_score(&mut cout, &hist, &board);

    cout += "\nUnset assumeMultipleStartingBlackMovesAreHandicap\n";
    hist.set_assume_multiple_starting_black_moves_are_handicap(false);
    hist.print_debug_info(&mut cout, &board);
    test(&mut cout, &board, &hist, next_pla);
    final_score(&mut cout, &hist, &board);

    cout += "\nPlay white move\n";
    let loc = Location::get_loc(4, 1, board.x_size);
    hist.make_board_move_assume_legal(&mut board, loc, P_WHITE, None, false);
    hist.print_debug_info(&mut cout, &board);
    test(&mut cout, &board, &hist, next_pla);
    final_score(&mut cout, &hist, &board);

    cout += "\nReclear history\n";
    hist.clear(&board, next_pla, &initial_rules, 0);
    hist.print_debug_info(&mut cout, &board);
    test(&mut cout, &board, &hist, next_pla);
    final_score(&mut cout, &hist, &board);
    cases.ran();
  }
  TestCommon::expect(name, &cout, &expected("nn-inputs-selfkomi-handicap-white-bonus", 0));
}
}

/// upstream: the `testScoring` lambda of the "Passing hack condition based on as-is scoring" block.
fn test_scoring(cout: &mut String, board: &Board, hist: &BoardHistory, expected_friendly_pass_suppress: bool) {
  hist.print_debug_info(cout, board);
  *cout += &format!("Finished or past normal end: {}\n", pb(hist.is_game_finished || hist.is_past_normal_phase_end));
  *cout += &format!("Pass would end phase: {}\n", pb(hist.pass_would_end_phase(board, hist.presumed_next_move_pla)));
  *cout += &format!("Pass would end game: {}\n", pb(hist.pass_would_end_game(board, hist.presumed_next_move_pla)));
  *cout += &format!("Pass should suppress: {}\n", pb(hist.should_suppress_end_game_from_friendly_pass(board, hist.presumed_next_move_pla)));

  let nn_x_len = 7;
  let nn_y_len = 7;
  let inputs_use_nhwc = false;
  let mut row_bin: Vec<f32> = vec![0.0; (NNInputs::NUM_FEATURES_SPATIAL_V7 * nn_x_len * nn_y_len) as usize];
  let mut row_global: Vec<f32> = vec![0.0; NNInputs::NUM_FEATURES_GLOBAL_V7 as usize];

  let mut nn_input_params = MiscNNInputParams::default();
  nn_input_params.draw_equivalent_wins_for_white = 0.5;

  // Currently should be true given that we only test game-end stuff, not illegal moves.
  testAssert!((hist.num_approx_valid_turns_this_phase > 0) == (hist.num_turns_this_phase > 0));

  let idx = |c: i32, y: i32, x: i32| (c * nn_x_len * nn_y_len + y * nn_x_len + x) as usize;

  {
    let mut hist_copy = hist.clone();
    nn_input_params.enable_passing_hacks = false;
    hist_copy.rules.friendly_pass_ok = false;
    NNInputs::fill_row_v7(board, &hist_copy, hist_copy.presumed_next_move_pla, &nn_input_params, nn_x_len, nn_y_len, inputs_use_nhwc, &mut row_bin, &mut row_global);
    let mut area: [Color; Board::MAX_ARR_SIZE as usize] = [0; Board::MAX_ARR_SIZE as usize];
    hist_copy.end_and_score_game_now_area(board, &mut area);
    *cout += &format!("SCORE NOW {}\n", p6(hist_copy.final_white_minus_black_score as f64));

    let mut history_input: f32 = 0.0;
    for y in 0..nn_y_len {
      for x in 0..nn_x_len {
        let loc = Location::get_loc(x, y, board.x_size);
        if hist.rules.scoring_rule == Rules::SCORING_TERRITORY && hist.encore_phase < 2 {
          testAssert!(row_bin[idx(18, y, x)] == 0.0f32);
          testAssert!(row_bin[idx(19, y, x)] == 0.0f32);
        } else {
          testAssert!((row_bin[idx(18, y, x)] == 1.0f32) == (area[loc as usize] == hist_copy.presumed_next_move_pla));
          testAssert!((row_bin[idx(19, y, x)] == 1.0f32) == (area[loc as usize] == get_opp(hist_copy.presumed_next_move_pla)));
        }
        history_input += row_bin[idx(9, y, x)];
      }
    }
    history_input += row_global[0];
    testAssert!((!hist.move_history.is_empty() && hist.num_approx_valid_turns_this_phase > 0) == (history_input > 0.0f32));
  }

  {
    let mut hist_copy = hist.clone();
    nn_input_params.enable_passing_hacks = false;
    hist_copy.rules.friendly_pass_ok = true;
    NNInputs::fill_row_v7(board, &hist_copy, hist_copy.presumed_next_move_pla, &nn_input_params, nn_x_len, nn_y_len, inputs_use_nhwc, &mut row_bin, &mut row_global);
    let mut area: [Color; Board::MAX_ARR_SIZE as usize] = [0; Board::MAX_ARR_SIZE as usize];
    hist_copy.end_and_score_game_now_area(board, &mut area);

    let mut history_input: f32 = 0.0;
    for y in 0..nn_y_len {
      for x in 0..nn_x_len {
        let loc = Location::get_loc(x, y, board.x_size);
        if hist.rules.scoring_rule == Rules::SCORING_TERRITORY && hist.encore_phase < 2 {
          testAssert!(row_bin[idx(18, y, x)] == 0.0f32);
          testAssert!(row_bin[idx(19, y, x)] == 0.0f32);
        } else {
          testAssert!((row_bin[idx(18, y, x)] == 1.0f32) == (area[loc as usize] == hist_copy.presumed_next_move_pla));
          testAssert!((row_bin[idx(19, y, x)] == 1.0f32) == (area[loc as usize] == get_opp(hist_copy.presumed_next_move_pla)));
        }
        history_input += row_bin[idx(9, y, x)];
      }
    }
    history_input += row_global[0];
    testAssert!(
      (!hist.move_history.is_empty() && hist.num_approx_valid_turns_this_phase > 0 && !expected_friendly_pass_suppress) == (history_input > 0.0f32)
    );
  }

  let mut area: [Color; Board::MAX_ARR_SIZE as usize] = [0; Board::MAX_ARR_SIZE as usize];
  let mut komi: f32 = -100.0;
  while komi <= 100.0f32 {
    let mut hist_copy = hist.clone();
    nn_input_params.enable_passing_hacks = true;
    hist_copy.rules.friendly_pass_ok = false;
    hist_copy.rules.komi = komi;
    NNInputs::fill_row_v7(board, &hist_copy, hist_copy.presumed_next_move_pla, &nn_input_params, nn_x_len, nn_y_len, inputs_use_nhwc, &mut row_bin, &mut row_global);
    hist_copy.end_and_score_game_now_area(board, &mut area);

    let mut history_input: f32 = 0.0;
    for y in 0..nn_y_len {
      for x in 0..nn_x_len {
        let loc = Location::get_loc(x, y, board.x_size);
        if hist.rules.scoring_rule == Rules::SCORING_TERRITORY && hist.encore_phase < 2 {
          testAssert!(row_bin[idx(18, y, x)] == 0.0f32);
          testAssert!(row_bin[idx(19, y, x)] == 0.0f32);
        } else {
          testAssert!((row_bin[idx(18, y, x)] == 1.0f32) == (area[loc as usize] == hist_copy.presumed_next_move_pla));
          testAssert!((row_bin[idx(19, y, x)] == 1.0f32) == (area[loc as usize] == get_opp(hist_copy.presumed_next_move_pla)));
        }
        history_input += row_bin[idx(9, y, x)];
      }
    }
    history_input += row_global[0];
    if hist_copy.final_white_minus_black_score.abs() <= 1.0f32 {
      *cout += &format!("Komi {} hasHistory {}\n", p6(komi as f64), pb(history_input > 0.0f32));
    }
    if hist.rules.scoring_rule == Rules::SCORING_TERRITORY && hist.encore_phase < 2 {
      testAssert!((!hist.move_history.is_empty() && hist.num_approx_valid_turns_this_phase > 0) == (history_input > 0.0f32));
    } else {
      testAssert!(
        (!hist.move_history.is_empty()
          && hist.num_approx_valid_turns_this_phase > 0
          && (!hist.pass_would_end_game(board, hist.presumed_next_move_pla) || hist_copy.winner == hist.presumed_next_move_pla))
          == (history_input > 0.0f32)
      );
    }
    komi += 0.5f32;
  }
}

wasm_test! {
/// upstream: "Passing hack condition based on as-is scoring" (calls fillRowV7 directly).
fn passing_hack_condition_based_on_as_is_scoring() {
  init();
  let mut cases = Cases::new("testnninputs::passing_hack_condition_based_on_as_is_scoring");
  let mut cout = String::new();
  cout += "Passing hack condition based on as-is scoring ==========================================================================\n";
  endl(&mut cout);

  // Raw block of testnninputs.cpp line 1176 (the extractor names it after the last `name`).
  let orig_board = Board::parse_board(7, 7, &fixture(INPUTS, "nn-inputs-selfkomi-handicap-white-bonus.input0.txt"), '\n').unwrap();

  // (rules string, taxRule override, move sequence, friendly-pass suppression per step)
  let area_moves: &[(&str, bool)] = &[("C7", false), ("pass", true), ("pass", false), ("pass", false)];
  let territory_moves: &[(&str, bool)] = &[
    ("C7", false),
    ("pass", false),
    ("pass", false),
    ("pass", false),
    ("pass", false),
    ("pass", false),
    ("E3", false),
    ("pass", false),
    ("G1", false),
    ("G3", false),
    ("G2", false),
    ("pass", false),
  ];
  let blocks: [(&str, Option<i32>, &[(&str, bool)]); 7] = [
    ("tromp-taylor", None, area_moves),
    ("chinese", None, area_moves),
    ("chinese", Some(Rules::TAX_SEKI), area_moves),
    ("stonescoring", None, area_moves),
    ("japanese", Some(Rules::TAX_NONE), territory_moves),
    ("japanese", None, territory_moves),
    ("japanese", Some(Rules::TAX_ALL), territory_moves),
  ];
  for (rules_str, tax, moves) in blocks.iter() {
    let mut rules = Rules::parse_rules(rules_str).unwrap();
    if let Some(t) = tax {
      rules.tax_rule = *t;
    }
    cout += "---------------------------------------------------\n";
    cout += &format!("{}\n", rules.to_string());
    let mut board = orig_board.clone();
    let mut hist = BoardHistory::new_full(&board, P_WHITE, &rules, 0);

    test_scoring(&mut cout, &board, &hist, false);
    // Moves alternate W, B, W, ... starting with white.
    let mut pla = P_WHITE;
    for (mv, suppress) in moves.iter() {
      let loc = Location::of_string(mv, &board).unwrap();
      hist.make_board_move_assume_legal(&mut board, loc, pla, None, false);
      test_scoring(&mut cout, &board, &hist, *suppress);
      pla = get_opp(pla);
    }
    cases.ran();
  }

  cout += "Ok\n";
  TestCommon::expect("Passing hack condition based on as-is scoring", &cout, &expected("passing-hack-condition-based-on-as-is-scoring", 0));
}
}

wasm_test! {
/// upstream: "NN Inputs history scoring past end of game or end of phase" (24 VERSION 7 chunks:
/// 4 rules x 2 SGFs x 3 modes; each chunk starts with its "rules" and "enablePassingHacks" lines).
fn nn_inputs_history_scoring_past_end() {
  init();
  let mut cases = Cases::new("testnninputs::nn_inputs_history_scoring_past_end");
  let name = "NN Inputs history scoring past end of game or end of phase";
  let sgf_strs: [&str; 2] = [
    "(;GM[1]FF[4]SZ[9]KM[7];B[ff];W[ee];B[dd];W[];B[];W[];B[cc];W[bb];B[];W[])",
    "(;GM[1]FF[4]SZ[9]KM[-7];B[ff];W[ee];B[dd];W[];B[];W[];B[cc];W[bb];B[];W[])",
  ];

  let mut k = 0;
  for which_rules in 0..=3 {
    let mut rules_to_use = Rules::parse_rules("tromp-taylor").unwrap();
    if which_rules == 1 {
      rules_to_use = Rules::parse_rules("japanese").unwrap();
      rules_to_use.friendly_pass_ok = false;
    }
    if which_rules == 2 {
      rules_to_use = Rules::parse_rules("tromp-taylor").unwrap();
      rules_to_use.has_button = true;
    }
    if which_rules == 3 {
      rules_to_use = Rules::parse_rules("tromp-taylor").unwrap();
      rules_to_use.friendly_pass_ok = true;
    }

    for sgf_str in sgf_strs.iter() {
      let sgf = CompactSgf::parse(sgf_str);
      let moves = &sgf.moves;

      for which_mode in 0..=2 {
        let mut enable_passing_hacks = false;
        let mut conservative_pass_and_is_root = false;
        if which_mode == 1 {
          enable_passing_hacks = true;
        }
        if which_mode == 2 {
          enable_passing_hacks = true;
          conservative_pass_and_is_root = true;
        }
        let mut cout = String::new();
        cout += &format!("rules {}\n", rules_to_use.to_string());
        cout += &format!("enablePassingHacks {} conservativePassAndIsRoot {}\n", pb(enable_passing_hacks), pb(conservative_pass_and_is_root));
        cout += &format!("VERSION {}\n", VERSION);
        let mut board = Board::new();
        let mut next_pla: Player = P_BLACK;
        let mut hist = BoardHistory::new();
        let rules = sgf.get_rules_or_fail_allow_unspecified(&rules_to_use);
        sgf.setup_initial_board_and_hist(&rules, &mut board, &mut next_pla, &mut hist);

        let nn_x_len = 9;
        let nn_y_len = 9;
        let draw_equivalent_wins_for_white = 0.5;
        let (mut row_bin, mut row_global) = allocate_rows(nn_x_len, nn_y_len);

        let mut out = String::new();
        for i in 0..moves.len() + 1 {
          let inputs_use_nhwc = false;
          let mut hash = Hash128::default();
          let mut nn_input_params = params(draw_equivalent_wins_for_white);
          nn_input_params.enable_passing_hacks = enable_passing_hacks;
          nn_input_params.conservative_pass_and_is_root = conservative_pass_and_is_root;
          fill_rows(&mut hash, &board, &hist, next_pla, &nn_input_params, nn_x_len, nn_y_len, inputs_use_nhwc, &mut row_bin, &mut row_global);

          let hist_loc_str = |lookback: i32| -> String {
            let mut num_poses_found = 0;
            let mut hist_pos = 0;
            for y in 0..board.y_size {
              for x in 0..board.x_size {
                if row_bin[(nn_x_len * nn_y_len * (9 + lookback - 1) + y * nn_x_len + x) as usize] > 0.0 {
                  num_poses_found += 1;
                  hist_pos += y * nn_x_len + x;
                }
              }
            }
            if row_global[(lookback - 1) as usize] > 0.0 {
              num_poses_found += 1;
              hist_pos += nn_y_len * nn_x_len;
            }
            testAssert!(num_poses_found <= 1);
            if num_poses_found == 1 {
              return Location::to_string(NNPos::pos_to_loc(hist_pos, board.x_size, board.y_size, nn_x_len, nn_y_len), &board);
            }
            Location::to_string(Board::NULL_LOC, &board)
          };

          out += &format!(
            "encorephase {} finished {} numTurnsThisPhase {} numApproxValidTurnsThisPhase {}\n",
            hist.encore_phase,
            pb(hist.is_game_finished),
            hist.num_turns_this_phase,
            hist.num_approx_valid_turns_this_phase
          );
          out += &format!("{}\n", hash);
          out += &format!(
            "History that net sees {} {} {} {} {}\n",
            hist_loc_str(1),
            hist_loc_str(2),
            hist_loc_str(3),
            hist_loc_str(4),
            hist_loc_str(5)
          );

          if i == moves.len() {
            break;
          }

          out += &format!("Move {} {}\n", i, Location::to_string(moves[i].loc, &board));
          testAssert!(hist.is_legal(&board, moves[i].loc, moves[i].pla));
          let prevent_encore = true;
          hist.make_board_move_assume_legal(&mut board, moves[i].loc, moves[i].pla, None, prevent_encore);
          next_pla = get_opp(moves[i].pla);
        }
        cout += &out;
        endl(&mut cout);
        TestCommon::expect(name, &cout, &expected("nn-inputs-history-scoring-past-end-of-game-or-end-of-phase", k));
        k += 1;
        cases.ran();
      }
    }
  }
  assert_eq!(k, 24);
}
}
