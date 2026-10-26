//! NN input features vs the real C++ (fixtures/upstream-v1.16.5/unit/oracle/nninputs.cpp, which also
//! documents the file formats). Replays every position of nninputs_positions.txt through the ported
//! Board/BoardHistory and checks, bit-exactly, against nninputs_expected.txt:
//!   res    NNInputs::getHash, fillRowV7 (NCHW and NHWC), rowGlobal, copyInputsWithSymmetry (both
//!          layouts), copyOutputsWithSymmetry
//!   posres fillScoring (groupTax false/true), markDuplicateMoveLocs, getSymBoard + getSymmetryDifferences
//!   dump   every nonzero rowBin entry of a subset of evals
//! over random games (incl. out-of-turn moves and odd x even boards), hand-built encore ko fights
//! (encore superKoBanned, koRecapBlocked, pass-for-ko) and SGF positions,
//! and the ScoreValue functions against nninputs_scorevalue.txt (bit-exact natively; on wasm32 the
//! atan/tan-based ones within 1e-12 relative, DECISIONS.md D13). Runs natively and on wasm32.
#![allow(clippy::needless_range_loop)]

// Test code keeps the shapes of upstream's tests and oracle programs (PORTING.md §11b).
#![allow(clippy::field_reassign_with_default)]

#[macro_use]
mod common;

use std::collections::HashMap;

use common::Cases;
use common::fixture;
use common::init;
use katago_core::game::board::*;
use katago_core::game::boardhistory::BoardHistory;
use katago_core::game::rules::Rules;
use katago_core::neuralnet::nninputs::*;

const FNV0: u64 = 0xCBF29CE484222325;

fn fnv(h: &mut u64, w: u32) {
  for k in 0..4 {
    *h ^= ((w >> (8 * k)) & 0xFF) as u64;
    *h = h.wrapping_mul(0x100000001B3);
  }
}
fn fnv64(h: &mut u64, w: u64) {
  fnv(h, w as u32);
  fnv(h, (w >> 32) as u32);
}
fn fnv_floats(a: &[f32]) -> u64 {
  let mut h = FNV0;
  for x in a {
    fnv(&mut h, x.to_bits());
  }
  h
}
fn fnv_bools(a: &[bool]) -> u64 {
  let mut h = FNV0;
  for x in a {
    fnv(&mut h, if *x { 1 } else { 0 });
  }
  h
}
fn x8(u: u32) -> String {
  format!("{:08X}", u)
}
fn x16(u: u64) -> String {
  format!("{:016X}", u)
}
fn hexf(s: &str) -> f32 {
  f32::from_bits(u32::from_str_radix(s, 16).expect("f32 hex"))
}
fn hexd(s: &str) -> f64 {
  f64::from_bits(u64::from_str_radix(s, 16).expect("f64 hex"))
}

struct MoveSpec {
  x: i32,
  y: i32,
  pla: i8,
  pe: bool,
}

struct PosSpec {
  xs: i32,
  ys: i32,
  rules: Rules,
  encore: i32,
  initial_pla: i8,
  amh: bool,
  setup: Vec<MoveSpec>,
  moves: Vec<MoveSpec>,
}

fn parse_pos(line: &str) -> (i32, PosSpec) {
  let t: Vec<&str> = line.split_whitespace().collect();
  assert_eq!(t[0], "pos");
  let i = |k: usize| -> i32 { t[k].parse().unwrap_or_else(|_| panic!("int token {} in {:?}", k, line)) };
  let id = i(1);
  let rules = Rules::new_full(i(4), i(5), i(6), i(7) != 0, i(8) != 0, i(9), i(10) != 0, hexf(t[11]));
  let mut k = 15;
  let ns = i(k) as usize;
  k += 1;
  let mut setup = Vec::new();
  for _ in 0..ns {
    setup.push(MoveSpec { x: i(k), y: i(k + 1), pla: i(k + 2) as i8, pe: false });
    k += 3;
  }
  let nm = i(k) as usize;
  k += 1;
  let mut moves = Vec::new();
  for _ in 0..nm {
    moves.push(MoveSpec { x: i(k), y: i(k + 1), pla: i(k + 2) as i8, pe: i(k + 3) != 0 });
    k += 4;
  }
  assert_eq!(k, t.len(), "trailing tokens in {:?}", line);
  (id, PosSpec { xs: i(2), ys: i(3), rules, encore: i(12), initial_pla: i(13) as i8, amh: i(14) != 0, setup, moves })
}

fn loc_of(m: &MoveSpec, xs: i32) -> Loc {
  if m.x == -1 { Board::PASS_LOC } else { Location::get_loc(m.x, m.y, xs) }
}

fn replay(p: &PosSpec) -> Box<(Board, BoardHistory)> {
  let mut board = Board::new_x_y(p.xs, p.ys).expect("board size");
  for m in &p.setup {
    board.set_stone(Location::get_loc(m.x, m.y, p.xs), m.pla);
  }
  let mut hist = BoardHistory::new_full(&board, p.initial_pla, &p.rules, p.encore);
  hist.set_assume_multiple_starting_black_moves_are_handicap(p.amh);
  for m in &p.moves {
    hist.make_board_move_assume_legal(&mut board, loc_of(m, p.xs), m.pla, None, m.pe);
  }
  Box::new((board, hist))
}

fn posres(id: i32, board: &Board, hist: &BoardHistory) -> String {
  let n = Board::MAX_ARR_SIZE as usize;
  let mut area: [Color; Board::MAX_ARR_SIZE as usize] = [0; Board::MAX_ARR_SIZE as usize];
  board.calculate_area(&mut area, true, true, true, hist.rules.multi_stone_suicide_legal);
  let mut scoring = vec![0.0f32; n];
  NNInputs::fill_scoring(board, &area, false, &mut scoring);
  let fs0 = fnv_floats(&scoring);
  NNInputs::fill_scoring(board, &area, true, &mut scoring);
  let fs1 = fnv_floats(&scoring);
  let mut is_sym_dup_loc = vec![false; n];
  let mut valid_syms: Vec<i32> = Vec::new();
  let avoid: Vec<i32> = Vec::new();
  SymmetryHelpers::mark_duplicate_move_locs(board, hist, None, &avoid, &mut is_sym_dup_loc, &mut valid_syms);
  let vs: Vec<String> = valid_syms.iter().map(|s| s.to_string()).collect();
  let mut s = format!("posres {} {} {} {} {}", id, x16(fs0), x16(fs1), vs.join(","), x16(fnv_bools(&is_sym_dup_loc)));
  for sym in [id % 8, (id * 3 + 1) % 8] {
    let sb = SymmetryHelpers::get_sym_board(board, sym).expect("getSymBoard of a legal board");
    let mut d = [0.0f64; SymmetryHelpers::NUM_SYMMETRIES as usize];
    SymmetryHelpers::get_symmetry_differences(board, &sb, 1e9, &mut d);
    let mut hd = FNV0;
    for x in d {
      fnv64(&mut hd, x.to_bits());
    }
    s += &format!(" {} {}", sym, x16(hd));
  }
  s
}

/// Returns (res line, optional dump body) for one eval line.
fn eval(line: &str, cur_id: i32, board: &Board, hist: &BoardHistory) -> (i32, i32, String, Vec<f32>, i32, i32) {
  let t: Vec<&str> = line.split_whitespace().collect();
  assert_eq!(t[0], "eval");
  assert_eq!(t.len(), 15, "eval tokens {:?}", line);
  let i = |k: usize| -> i32 { t[k].parse().unwrap() };
  let pid = i(1);
  assert_eq!(pid, cur_id);
  let eidx = i(2);
  let next_pla = i(3) as i8;
  let mut params = MiscNNInputParams::default();
  params.draw_equivalent_wins_for_white = hexd(t[4]);
  params.conservative_pass_and_is_root = i(5) != 0;
  params.enable_passing_hacks = i(6) != 0;
  params.playout_doubling_advantage = hexd(t[7]);
  params.nn_policy_temperature = hexf(t[8]);
  params.avoid_myt_dagger_hack = i(9) != 0;
  params.symmetry = i(10);
  params.policy_optimism = hexd(t[11]);
  params.max_history = i(12);
  let nn_x = i(13);
  let nn_y = i(14);
  let c = NNInputs::NUM_FEATURES_SPATIAL_V7;
  let g = NNInputs::NUM_FEATURES_GLOBAL_V7 as usize;
  let n = (c * nn_x * nn_y) as usize;
  let plane_len = (nn_x * nn_y) as usize;
  let mut bin_n = vec![0.0f32; n];
  let mut bin_h = vec![0.0f32; n];
  let mut glob = vec![0.0f32; g];
  let mut glob_h = vec![0.0f32; g];
  let h = NNInputs::get_hash(board, hist, next_pla, &params);
  NNInputs::fill_row_v7(board, hist, next_pla, &params, nn_x, nn_y, false, &mut bin_n, &mut glob);
  NNInputs::fill_row_v7(board, hist, next_pla, &params, nn_x, nn_y, true, &mut bin_h, &mut glob_h);
  assert!(glob.iter().zip(glob_h.iter()).all(|(a, b)| a.to_bits() == b.to_bits()), "rowGlobal NCHW vs NHWC");
  let s = if params.symmetry >= 0 { params.symmetry } else { 0 };
  let mut r = format!("res {} {} {} {} {} {}", pid, eidx, x16(h.hash0), x16(h.hash1), x16(fnv_floats(&bin_n)), x16(fnv_floats(&bin_h)));
  for v in &glob {
    r += " ";
    r += &if v.to_bits() == 0 { "0".to_string() } else { x8(v.to_bits()) };
  }
  let mut dst = vec![0.0f32; n];
  SymmetryHelpers::copy_inputs_with_symmetry(&bin_n, &mut dst, 1, nn_y, nn_x, c, false, s);
  r += &format!(" {}", x16(fnv_floats(&dst)));
  dst.fill(0.0);
  SymmetryHelpers::copy_inputs_with_symmetry(&bin_h, &mut dst, 1, nn_y, nn_x, c, true, s);
  r += &format!(" {}", x16(fnv_floats(&dst)));
  let plane: Vec<f32> = bin_n[..plane_len].to_vec();
  let mut plane_dst = vec![0.0f32; plane_len];
  SymmetryHelpers::copy_outputs_with_symmetry(&plane, &mut plane_dst, 1, nn_y, nn_x, s);
  r += &format!(" {}", x16(fnv_floats(&plane_dst)));
  (pid, eidx, r, bin_n, nn_x, nn_y)
}

fn dump_line(pid: i32, eidx: i32, bin: &[f32], nn_x: i32, nn_y: i32) -> String {
  let plane = (nn_x * nn_y) as usize;
  let mut cnt = 0;
  let mut d = String::new();
  for (k, v) in bin.iter().enumerate() {
    if v.to_bits() != 0 {
      cnt += 1;
      d += &format!(" {} {} {}", k / plane, k % plane, x8(v.to_bits()));
    }
  }
  format!("dump {} {} {}{}", pid, eidx, cnt, d)
}

wasm_test! {
fn nninputs_features_vs_cpp() {
  // wasm32 has a 1 MB stack: init (the 115 KB zobrist statics, copied around in debug builds) runs from
  // a small frame, and the replay keeps its Board/BoardHistory boxed.
  init();
  run_features();
}
}

#[inline(never)]
fn run_features() {
  let positions = fixture("oracle", "nninputs_positions.txt");
  let expected = fixture("oracle", "nninputs_expected.txt");
  // Key: "posres <id>", "res <id> <e>", "dump <id> <e>".
  let mut exp: HashMap<String, &str> = HashMap::new();
  for l in expected.lines() {
    let t: Vec<&str> = l.splitn(4, ' ').collect();
    let key = if t[0] == "posres" { format!("posres {}", t[1]) } else { format!("{} {} {}", t[0], t[1], t[2]) };
    assert!(exp.insert(key, l).is_none(), "duplicate expected line {:?}", l);
  }
  let num_dumps_expected = exp.keys().filter(|k| k.starts_with("dump ")).count();

  let mut pos_cases = Cases::new("nninputs_features_vs_cpp: positions (posres)");
  let mut eval_cases = Cases::new("nninputs_features_vs_cpp: evals (res)");
  let mut dump_cases = Cases::new("nninputs_features_vs_cpp: full dumps");
  let mut bad: Vec<String> = Vec::new();
  let mut num_bad = 0;
  let mut state: Option<(i32, Box<(Board, BoardHistory)>)> = None;
  let mut sizes = std::collections::BTreeSet::new();
  for line in positions.lines() {
    if line.starts_with("pos ") {
      let (id, p) = parse_pos(line);
      sizes.insert((p.xs, p.ys));
      let bh = replay(&p);
      let actual = posres(id, &bh.0, &bh.1);
      let want = exp.get(&format!("posres {}", id)).unwrap_or_else(|| panic!("no posres for {}", id));
      pos_cases.ran();
      if actual != *want {
        num_bad += 1;
        if bad.len() < 10 {
          bad.push(format!("rust {}\n cpp {}", actual, want));
        }
      }
      state = Some((id, bh));
      continue;
    }
    let (cur_id, bh) = state.as_ref().expect("eval before pos");
    let (pid, eidx, actual, bin, nn_x, nn_y) = eval(line, *cur_id, &bh.0, &bh.1);
    let want = exp.get(&format!("res {} {}", pid, eidx)).unwrap_or_else(|| panic!("no res for {} {}", pid, eidx));
    eval_cases.ran();
    if actual != *want {
      num_bad += 1;
      if bad.len() < 10 {
        bad.push(format!("{}\nrust {}\n cpp {}", line, actual, want));
      }
    }
    if let Some(want_dump) = exp.get(&format!("dump {} {}", pid, eidx)) {
      dump_cases.ran();
      let actual_dump = dump_line(pid, eidx, &bin, nn_x, nn_y);
      if actual_dump != *want_dump {
        num_bad += 1;
        if bad.len() < 10 {
          bad.push(format!("dump differs for {} {}", pid, eidx));
        }
      }
    }
  }
  assert!(num_bad == 0, "{} mismatches vs the C++ oracle; first ones:\n{}", num_bad, bad.join("\n"));
  assert!(sizes.len() >= 20, "board sizes covered: {:?}", sizes);
  // odd x even boards (xSize*ySize parity vs xSize^2, bArea)
  for s in [(9, 10), (5, 4), (7, 12), (3, 8), (19, 18)] {
    assert!(sizes.contains(&s), "board size {:?} not covered: {:?}", s, sizes);
  }
  report!("nninputs_features_vs_cpp: {} board sizes, {} full dumps expected", sizes.len(), num_dumps_expected);
}

/// Tolerance for the functions whose results go through atan/tan (platform libm on wasm, D13):
/// bit-exact natively, 1e-12 relative (or 1e-15 absolute; score values are in [-1,1]) on wasm32.
fn close(a: f64, b: f64) -> bool {
  if a.to_bits() == b.to_bits() {
    return true;
  }
  if cfg!(target_arch = "wasm32") {
    let d = (a - b).abs();
    // Absolute 1e-15 for the cancellation residues of expectedWhiteScoreValue at mean 0 (|value| ~1e-17).
    return d <= 1e-12 * a.abs().max(b.abs()) || d <= 1e-15;
  }
  false
}

wasm_test! {
fn scorevalue_vs_cpp() {
  init();
  ScoreValue::init_tables();
  run_scorevalue();
}
}

#[inline(never)]
fn run_scorevalue() {
  let text = fixture("oracle", "nninputs_scorevalue.txt");
  let mut hists: HashMap<i32, Box<BoardHistory>> = HashMap::new();
  let mut cases = Cases::new("scorevalue_vs_cpp");
  let mut inexact = 0;
  let mut num_bad = 0;
  let mut bad: Vec<String> = Vec::new();
  for line in text.lines() {
    let t: Vec<&str> = line.split_whitespace().collect();
    let d = |k: usize| hexd(t[k]);
    let i = |k: usize| -> i32 { t[k].parse().unwrap() };
    let (got, want, exact): (f64, f64, bool) = match t[0] {
      "hist" => {
        let b = Board::new_x_y(i(2), i(3)).unwrap();
        let rules = Rules::new_full(i(4), i(5), i(6), false, false, i(7), false, hexf(t[8]));
        hists.insert(i(1), Box::new(BoardHistory::new_full(&b, P_BLACK, &rules, 0)));
        continue;
      }
      "whiteWinsOfWinner" => (ScoreValue::white_wins_of_winner(i(1) as i8, d(2)), d(3), true),
      "whiteScoreDrawAdjust" => (ScoreValue::white_score_draw_adjust(d(2), d(3), &hists[&i(1)]), d(4), true),
      "whiteScoreValueOfScoreSmooth" => {
        (ScoreValue::white_score_value_of_score_smooth(d(2), d(3), d(4), d(5), d(6), &hists[&i(1)]), d(7), false)
      }
      "whiteScoreValueOfScoreSmoothNoDrawAdjust" => {
        (ScoreValue::white_score_value_of_score_smooth_no_draw_adjust(d(1), d(2), d(3), d(4)), d(5), false)
      }
      "whiteDScoreValueDScoreSmoothNoDrawAdjust" => {
        (ScoreValue::white_d_score_value_d_score_smooth_no_draw_adjust(d(1), d(2), d(3), d(4)), d(5), true)
      }
      "approxWhiteScoreOfScoreValueSmooth" => (ScoreValue::approx_white_score_of_score_value_smooth(d(1), d(2), d(3), d(4)), d(5), false),
      "whiteScoreMeanSqOfScoreGridded" => (ScoreValue::white_score_mean_sq_of_score_gridded(d(1), d(2)), d(3), true),
      "getScoreStdev" => (ScoreValue::get_score_stdev(d(1), d(2)), d(3), true),
      "expectedWhiteScoreValue" => (ScoreValue::expected_white_score_value(d(1), d(2), d(3), d(4), d(5)), d(6), false),
      other => panic!("unknown scorevalue line {:?}", other),
    };
    cases.ran();
    let ok = if exact { got.to_bits() == want.to_bits() } else { close(got, want) };
    if ok && got.to_bits() != want.to_bits() {
      inexact += 1;
    }
    if !ok {
      num_bad += 1;
      if bad.len() < 10 {
        bad.push(format!("{} -> rust {:016X} ({})", line, got.to_bits(), got));
      }
    }
  }
  assert!(num_bad == 0, "{} ScoreValue results differ from the C++; first ones:\n{}", num_bad, bad.join("\n"));
  report!("scorevalue_vs_cpp: {} results within tolerance but not bit-identical (0 expected natively)", inexact);
  if !cfg!(target_arch = "wasm32") {
    assert_eq!(inexact, 0);
  }
}
