//! Differential fuzz regression oracle for game/boardhistory against the real C++
//! (fixtures/upstream-v1.16.5/unit/oracle/boardhistory_fuzz.cpp). Mirrors boardhistory_fuzz.cpp line
//! for line: same splitmix stream, same calls in the same order, same output text.
//!   boardhistory_fuzz_hashes.txt  one FNV-1a 64 of each game's full trace (seed0 = 1)
//!   boardhistory_fuzz_trace.txt   full traces of 40 games (seed0 = 7), compared exactly
//!   boardhistory_fuzz_komiedge.txt  NaN/inf/out-of-range komi and draw values (getSituationRulesAndKoHash)
//! On a hash mismatch, set BHFUZZ_DUMP=<path> to write the Rust trace of the first differing game,
//! and diff it against `boardhistory_fuzz games <g+1> 1 | sed -n '/^G <g> /,$p'`.

mod common;

use common::Cases;
use common::compare_lines;
use common::fixture;
use common::init;
use katago_core::core::global::Global;
use katago_core::core::hash::Hash128;
use katago_core::game::board::*;
use katago_core::game::boardhistory::*;
use katago_core::game::graphhash::GraphHash;
use katago_core::game::rules::Rules;
use katago_core::port::fmt::cpp_ostream_f64;

struct R {
  s: u64,
}
impl R {
  fn nxt(&mut self) -> u64 {
    self.s = self.s.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = self.s;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
  }
  fn rnd(&mut self, n: i32) -> i32 {
    (self.nxt() % (n as u64)) as i32
  }
}

fn b(x: bool) -> &'static str {
  if x { "1" } else { "0" }
}
fn f(x: f32) -> String {
  cpp_ostream_f64(x as f64)
}
fn fx(x: f32) -> String {
  format!("{:08x}", x.to_bits())
}
fn ff(x: f32) -> String {
  f(x) + "/" + &fx(x)
}
fn hx(h: Hash128) -> String {
  h.to_string()
}
fn hs(h: Hash128) -> String {
  Global::uint64_to_hex_string(h.hash0 ^ h.hash1)
}

fn fold(b: &Board) -> u64 {
  let mut h: u64 = 1469598103934665603;
  let mut mix = |v: i64| {
    h ^= v as u64;
    h = h.wrapping_mul(1099511628211);
  };
  for loc in 0..Board::MAX_ARR_SIZE as usize {
    mix(b.colors[loc] as i64);
    if b.colors[loc] == C_BLACK || b.colors[loc] == C_WHITE {
      mix(b.chain_head[loc] as i64);
      mix(b.next_in_chain[loc] as i64);
      if b.chain_head[loc] as usize == loc {
        mix(b.chain_data[loc].owner as i64);
        mix(b.chain_data[loc].num_locs as i64);
        mix(b.chain_data[loc].num_liberties as i64);
      }
    }
  }
  mix(b.ko_loc as i64);
  mix(b.pos_hash.hash0 as i64);
  mix(b.pos_hash.hash1 as i64);
  mix(b.num_black_captures as i64);
  mix(b.num_white_captures as i64);
  h
}
fn st(b: &Board) -> String {
  format!(" F{} ko{}", fold(b) >> 1, b.ko_loc)
}

fn fold_arr(a: &[i8]) -> u64 {
  let mut h: u64 = 1469598103934665603;
  for v in a.iter().take(Board::MAX_ARR_SIZE as usize) {
    h ^= *v as i64 as u64;
    h = h.wrapping_mul(1099511628211);
  }
  h
}
fn fold_bool(a: &[bool]) -> u64 {
  let mut h: u64 = 1469598103934665603;
  for v in a.iter().take(Board::MAX_ARR_SIZE as usize) {
    h ^= if *v { 1 } else { 0 };
    h = h.wrapping_mul(1099511628211);
  }
  h
}

/// `xfold` in boardhistory_fuzz.cpp: folds of every BoardHistory vector and per-location array that
/// `dump` does not print in full, then the full board fold of every recent board and of initialBoard.
fn xfold(hist: &BoardHistory) -> String {
  let mut fh: u64 = 1469598103934665603;
  let mut mixh = |v: i64| {
    fh ^= v as u64;
    fh = fh.wrapping_mul(1099511628211);
  };
  for h in &hist.ko_hash_history {
    mixh(h.hash0 as i64);
    mixh(h.hash1 as i64);
  }
  mixh(-1);
  for h in &hist.hashes_before_black_pass {
    mixh(h.hash0 as i64);
    mixh(h.hash1 as i64);
  }
  mixh(-2);
  for h in &hist.hashes_before_white_pass {
    mixh(h.hash0 as i64);
    mixh(h.hash1 as i64);
  }
  mixh(-3);
  for e in &hist.ko_captures_in_encore {
    mixh(e.pos_hash_before_move.hash0 as i64);
    mixh(e.pos_hash_before_move.hash1 as i64);
    mixh(e.move_loc as i64);
    mixh(e.move_pla as i64);
  }
  mixh(-4);
  for i in 0..hist.move_history.len() {
    mixh(hist.move_history[i].loc as i64);
    mixh(hist.move_history[i].pla as i64);
    mixh(if hist.prevent_encore_history[i] { 1 } else { 0 });
  }
  mixh(-5);
  for loc in 0..Board::MAX_ARR_SIZE as usize {
    mixh(if hist.ko_recap_blocked[loc] { 1 } else { 0 });
    mixh(if hist.super_ko_banned[loc] { 1 } else { 0 });
  }
  mixh(hist.current_recent_board_idx as i64);
  let mut s = format!(" xf{}", (fh >> 1) as i64);
  for k in 0..BoardHistory::NUM_RECENT_BOARDS {
    s += &format!(",{}", (fold(hist.get_recent_board(k)) >> 1) as i64);
  }
  s += &format!(",{}", (fold(&hist.initial_board) >> 1) as i64);
  s
}

fn area_str(b: &Board, r: &[Color]) -> String {
  let mut s = String::new();
  for y in 0..b.y_size {
    for x in 0..b.x_size {
      s.push((b'0' as i32 + r[Location::get_loc(x, y, b.x_size) as usize] as i32) as u8 as char);
    }
  }
  s
}

const KOMIS: [f32; 18] = [0.0, 0.5, 6.5, 7.0, 7.5, -3.0, 1.5, 100.5, -150.0, 150.0, 2.0, -0.5, 0.25, 3.3, -7.1, 1e9, -1e9, 0.1];
const NUM_KOMIS: i32 = 18;
const DRAWS: [f64; 4] = [0.0, 0.5, 0.7, 1.0];
const DRAWS2: [f64; 6] = [0.25, 0.3, -1.0, 2.0, 0.1, 0.9999];

thread_local! {
  /// `static Hash128 GH, GH2, GH11` in boardhistory_fuzz.cpp: the graph hashes chained through a
  /// game's dumps (repBound 1..4, 0..22, and 11 = the search default).
  static GH: std::cell::Cell<Hash128> = const { std::cell::Cell::new(Hash128::new(0, 0)) };
  static GH2: std::cell::Cell<Hash128> = const { std::cell::Cell::new(Hash128::new(0, 0)) };
  static GH11: std::cell::Cell<Hash128> = const { std::cell::Cell::new(Hash128::new(0, 0)) };
}

/// `dump` in boardhistory_fuzz.cpp: one state line ("S") plus one legality line ("L").
fn dump(out: &mut String, hist: &BoardHistory, board: &Board, table: Option<&KoHashTable>, tag: &str) {
  let np: Player = hist.presumed_next_move_pla;
  let mut s = format!("S {}{}", tag, st(board));
  s += &format!(" np{}", np);
  s += &format!(" ep{} nt{} na{}", hist.encore_phase, hist.num_turns_this_phase, hist.num_approx_valid_turns_this_phase);
  s += &format!(" nc{} cp{}", hist.num_consec_valid_turns_this_game, hist.consecutive_ending_passes);
  s += &format!(" fin{} w{} sc{}", b(hist.is_game_finished), hist.winner, ff(hist.final_white_minus_black_score));
  s += &format!(" isS{} nr{} rs{}", b(hist.is_scored), b(hist.is_no_result), b(hist.is_resignation));
  s += &format!(" pe{} bt{}", b(hist.is_past_normal_phase_end), b(hist.has_button));
  s += &format!(" wb{} wh{} km{}", ff(hist.white_bonus_score), ff(hist.white_handicap_bonus_score), ff(hist.rules.komi));
  s += " k";
  for p in 1..=2 as Player {
    for d in DRAWS {
      s += &format!(",{}", fx(hist.current_self_komi(p, d)));
    }
  }
  s += " adj";
  for d in DRAWS {
    s += &format!(",{}", fx(hist.white_komi_adjustment_for_draws(d)));
  }
  let last = hist.ko_hash_history[hist.ko_hash_history.len() - 1];
  s += &format!(" kh{},{},{}", hx(last), hist.ko_hash_history.len(), hist.first_turn_idx_with_ko_history);
  s += &format!(" nko{}", hist.number_of_ko_hash_occurrences_in_history(last, None));
  s += &format!(",{}", hist.number_of_ko_hash_occurrences_in_history(last, table));
  s += &format!(" rk{}", hx(hist.ko_recap_block_hash));
  s += &format!(" hp{},{} ek{}", hist.hashes_before_black_pass.len(), hist.hashes_before_white_pass.len(), hist.ko_captures_in_encore.len());
  s += &format!(" mh{},{}", hist.move_history.len(), hist.prevent_encore_history.len());
  s += &format!(" h1{}", hx(BoardHistory::get_situation_rules_and_ko_hash(board, hist, P_BLACK, 0.5)));
  s += &format!(",{}", hx(BoardHistory::get_situation_rules_and_ko_hash(board, hist, P_WHITE, 0.5)));
  s += &format!(",{}", hs(BoardHistory::get_situation_rules_and_ko_hash(board, hist, np, 0.0)));
  s += &format!(",{}", hs(BoardHistory::get_situation_rules_and_ko_hash(board, hist, np, 1.0)));
  s += &format!(" h2{}", hx(BoardHistory::get_situation_and_simple_ko_hash(board, np)));
  s += &format!(" h3{}", hx(BoardHistory::get_situation_and_simple_ko_and_prev_pos_hash(board, hist, np)));
  s += " rb";
  for k in 0..BoardHistory::NUM_RECENT_BOARDS {
    s += &format!(",{}", hs(hist.get_recent_board(k).pos_hash));
  }
  s += &format!(
    " ib{},{},{},{}",
    hs(hist.initial_board.pos_hash),
    hist.initial_pla,
    hist.initial_encore_phase,
    hist.initial_turn_number
  );
  s += &format!(" se{}", (fold_arr(&hist.second_encore_start_colors) >> 1) as i64);
  s += &format!(" wo{}", (fold_bool(&hist.was_ever_occupied_or_played) >> 1) as i64);
  s += &format!(" bw{} tn{}", b(hist.has_black_pass_or_white_first()), hist.get_current_turn_number());
  s += &format!(
    " hc{},{},{}",
    hist.compute_num_handicap_stones(),
    hist.compute_white_handicap_bonus(),
    BoardHistory::num_handicap_stones_on_board(board)
  );
  s += &format!(
    ",{}{},{}",
    b(hist.assume_multiple_starting_black_moves_are_handicap),
    b(hist.white_has_moved),
    hist.override_num_handicap_stones
  );
  s += " pw";
  for p in 1..=2 as Player {
    s += b(hist.pass_would_end_phase(board, p));
    s += b(hist.pass_would_end_game(board, p));
    s += b(hist.should_suppress_end_game_from_friendly_pass(board, p));
  }
  s += b(hist.is_final_phase());
  // GraphHash (game/graphhash.cpp): state hashes, and a graph hash chained through every dump of the game.
  s += &format!(
    " gs{},{}",
    hx(GraphHash::get_state_hash(hist, np, 0.5)),
    hs(GraphHash::get_state_hash(hist, get_opp(np), 0.0))
  );
  let gh = GraphHash::get_graph_hash(GH.with(|c| c.get()), hist, np, 1 + (hist.move_history.len() % 4) as i32, 0.5);
  GH.with(|c| c.set(gh));
  s += &format!(" gh{}", hx(gh));
  let gh2 = GraphHash::get_graph_hash(GH2.with(|c| c.get()), hist, np, (hist.move_history.len() % 23) as i32, 0.5);
  GH2.with(|c| c.set(gh2));
  let gh11 = GraphHash::get_graph_hash(GH11.with(|c| c.get()), hist, np, 11, 0.3);
  GH11.with(|c| c.set(gh11));
  s += &format!(",{},{}", hx(gh2), hx(gh11));
  s += &xfold(hist);
  s += " k2";
  for p in 1..=2 as Player {
    for d in DRAWS2 {
      s += &format!(",{}{}", fx(hist.current_self_komi(p, d)), hs(BoardHistory::get_situation_rules_and_ko_hash(board, hist, p, d)));
    }
  }
  *out += &s;
  *out += "\n";

  let mut l = String::from("L");
  for p in 1..=2 as Player {
    let mut lg = String::new();
    let mut tl = String::new();
    let mut pk = String::new();
    let mut off_legal = 0;
    let mut off_tol = 0;
    for loc in 0..Board::MAX_ARR_SIZE {
      let a = hist.is_legal(board, loc as Loc, p);
      let t = hist.is_legal_tolerant(board, loc as Loc, p);
      if loc as Loc != Board::PASS_LOC && !board.is_on_board(loc as Loc) {
        if a {
          off_legal += 1;
        }
        if t {
          off_tol += 1;
        }
      }
    }
    for y in 0..board.y_size {
      for x in 0..board.x_size {
        let loc = Location::get_loc(x, y, board.x_size);
        lg += b(hist.is_legal(board, loc, p));
        tl += b(hist.is_legal_tolerant(board, loc, p));
        pk += b(hist.is_pass_for_ko(board, loc, p));
      }
    }
    lg += b(hist.is_legal(board, Board::PASS_LOC, p));
    tl += b(hist.is_legal_tolerant(board, Board::PASS_LOC, p));
    l += &format!(" p{}:{},{},{},{},{}", p, lg, tl, pk, off_legal, off_tol);
  }
  let mut kr = String::new();
  let mut sk = String::new();
  for y in 0..board.y_size {
    for x in 0..board.x_size {
      let loc = Location::get_loc(x, y, board.x_size);
      kr += b(hist.ko_recap_blocked[loc as usize]);
      sk += b(hist.super_ko_banned[loc as usize]);
    }
  }
  l += &format!(" kr{} sk{}", kr, sk);
  *out += &l;
  *out += "\n";
}

/// `runGame` in boardhistory_fuzz.cpp.
fn run_game(out: &mut String, g: i32, seed0: u64) {
  GH.with(|c| c.set(Hash128::default()));
  GH2.with(|c| c.set(Hash128::default()));
  GH11.with(|c| c.set(Hash128::default()));
  let sizes: [[i32; 2]; 24] = [
    [1, 1],
    [2, 2],
    [3, 3],
    [2, 3],
    [4, 1],
    [5, 5],
    [4, 4],
    [7, 7],
    [9, 9],
    [5, 7],
    [6, 4],
    [3, 5],
    [13, 13],
    [19, 19],
    [8, 3],
    [3, 2],
    [6, 6],
    [2, 1],
    [1, 2],
    [19, 1],
    [10, 10],
    [11, 11],
    [1, 7],
    [19, 2],
  ];
  let num_sizes = 24;
  let mut rng = R { s: seed0.wrapping_mul(1000003).wrapping_add(g as u64) };
  let si = rng.rnd(num_sizes);
  let xs = sizes[si as usize][0];
  let ys = sizes[si as usize][1];
  let mut board = Board::new_x_y(xs, ys).unwrap();
  let area = xs * ys;

  let mut rules = Rules::new();
  rules.ko_rule = rng.rnd(4);
  rules.scoring_rule = rng.rnd(2);
  rules.tax_rule = rng.rnd(3);
  rules.multi_stone_suicide_legal = rng.rnd(2) == 1;
  rules.white_handicap_bonus_rule = rng.rnd(3);
  rules.friendly_pass_ok = rng.rnd(2) == 1;
  let ki = rng.rnd(NUM_KOMIS);
  rules.komi = KOMIS[ki as usize];
  let mut e_phase = 0;
  if rules.scoring_rule == Rules::SCORING_TERRITORY {
    let e = rng.rnd(4);
    e_phase = if e <= 1 { 0 } else { e - 1 };
  }
  let bt = rng.rnd(4);
  rules.has_button = bt == 0 && e_phase == 0;
  let pl = rng.rnd(2);
  let pla: Player = if pl == 0 { P_BLACK } else { P_WHITE };

  // Non-empty initial board sometimes
  let setup = rng.rnd(10);
  if setup < 3 && area > 1 {
    let ns = rng.rnd(area / 2 + 1);
    for _k in 0..ns {
      let lx = rng.rnd(xs);
      let ly = rng.rnd(ys);
      let c = rng.rnd(3);
      // C++: `c == 0 ? C_BLACK : c == 1 ? C_WHITE : C_BLACK`
      let col: Color = if c == 1 { C_WHITE } else { C_BLACK };
      board.set_stone_fail_if_no_libs(Location::get_loc(lx, ly, xs), col);
    }
  }

  let mut hist = BoardHistory::new_full(&board, pla, &rules, e_phase);
  let o1 = rng.rnd(3);
  if o1 == 0 {
    let tn = rng.rnd(50);
    hist.set_initial_turn_number((tn - 5) as i64);
  }
  let o2 = rng.rnd(3);
  if o2 == 0 {
    hist.set_assume_multiple_starting_black_moves_are_handicap(true);
  }
  let o3 = rng.rnd(6);
  if o3 == 0 {
    let n = rng.rnd(5);
    hist.set_override_num_handicap_stones(n - 1);
  }
  let mut table_store = KoHashTable::new();
  let mut table_on = false;
  let use_table = rng.rnd(3);
  if use_table == 0 {
    table_store.recompute(&hist).unwrap();
    table_on = true;
  }
  let pass_prob = rng.rnd(25) + 2;
  let prevent_mode = rng.rnd(4);
  let ko_pref = rng.rnd(90);
  let mut max_steps = area * 3 + 10;
  if max_steps > 400 {
    max_steps = 400;
  }
  let mut resurrections = 0;
  let mut cleared = false;
  // Replaying moveHistory from copyToInitial() reproduces the game only if no out-of-band state change
  // (resignation, endAndScoreGameNow / endGameIfAllPassAlive taking the button, clear()) happened;
  // otherwise a replayed encore pass-for-ko can land on an occupied point in the main phase (C++ UB).
  let mut replayable = true;

  *out += &format!("G {} {} {} {} {} ", g, xs, ys, pla, e_phase);
  *out += &format!(
    "{} {} t{} pp{} pm{}{}\n",
    rules,
    rules.to_json_string(),
    b(table_on),
    pass_prob,
    prevent_mode,
    st(&board)
  );
  dump(out, &hist, &board, if table_on { Some(&table_store) } else { None }, "init");

  let mut last_was_pass = false;
  for n in 0..max_steps {
    if hist.is_game_finished || hist.is_past_normal_phase_end {
      let cont = rng.rnd(3);
      if cont == 0 || resurrections >= 8 {
        break;
      }
      resurrections += 1;
    }
    let mut mp: Player = hist.presumed_next_move_pla;
    let wrong = rng.rnd(25);
    if wrong == 0 {
      mp = get_opp(mp);
    }
    let mut pe = false;
    if prevent_mode == 0 {
      let x = rng.rnd(2);
      pe = x == 0;
    }
    let mut loc: Loc = Board::PASS_LOC;
    let r = rng.rnd(100);
    let eff_pass = pass_prob + if last_was_pass { 45 } else { 0 };
    if r >= eff_pass {
      let mut legal: Vec<Loc> = Vec::new();
      let mut ko_moves: Vec<Loc> = Vec::new();
      let mut illegal: Vec<Loc> = Vec::new();
      for y in 0..ys {
        for x in 0..xs {
          let l = Location::get_loc(x, y, xs);
          if hist.is_legal(&board, l, mp) {
            legal.push(l);
            if board.would_be_ko_capture(l, mp) || hist.is_pass_for_ko(&board, l, mp) {
              ko_moves.push(l);
            }
          } else if board.is_legal_ignoring_ko(l, mp, true) || hist.is_pass_for_ko(&board, l, mp) {
            illegal.push(l);
          }
        }
      }
      let r2 = rng.rnd(100);
      if (r2 < 6 || (wrong == 0 && r2 < 50)) && !illegal.is_empty() {
        loc = illegal[rng.rnd(illegal.len() as i32) as usize];
      } else if r2 < 6 + ko_pref && !ko_moves.is_empty() {
        loc = ko_moves[rng.rnd(ko_moves.len() as i32) as usize];
      } else if !legal.is_empty() {
        loc = legal[rng.rnd(legal.len() as i32) as usize];
      }
    }
    let kind = rng.rnd(10);
    let mut ms = format!("M {} {} {} k{} pe{}", n, mp, loc, kind, b(pe));
    if kind <= 6 {
      hist.make_board_move_assume_legal(&mut board, loc, mp, if table_on { Some(&table_store) } else { None }, pe);
    } else if kind <= 8 {
      let rl = rng.rnd(5);
      if rl == 0 {
        let z = rng.rnd(Board::MAX_ARR_SIZE);
        loc = z as Loc;
      }
      let ok = hist.make_board_move_tolerant(&mut board, loc, mp);
      ms += &format!(" tol{},{}", loc, b(ok));
    } else {
      let ok = hist.make_board_move_tolerant_prevent_encore(&mut board, loc, mp, pe);
      ms += &format!(" tolp{}", b(ok));
    }
    last_was_pass = loc == Board::PASS_LOC;
    *out += &ms;
    *out += "\n";
    dump(out, &hist, &board, if table_on { Some(&table_store) } else { None }, &n.to_string());

    let x1 = rng.rnd(40);
    if x1 == 0 {
      let k2 = rng.rnd(NUM_KOMIS);
      hist.set_komi(KOMIS[k2 as usize]);
      *out += &format!("KOMI {} {} {}\n", ff(KOMIS[k2 as usize]), ff(hist.final_white_minus_black_score), hist.winner);
    }
    let x2 = rng.rnd(200);
    if x2 == 0 {
      let w = rng.rnd(2);
      hist.set_winner_by_resignation(if w == 0 { P_BLACK } else { P_WHITE });
      replayable = false;
      dump(out, &hist, &board, if table_on { Some(&table_store) } else { None }, "resign");
    }
    let x3 = rng.rnd(80);
    if x3 == 0 {
      hist.end_and_score_game_now(&board);
      replayable = false;
      dump(out, &hist, &board, if table_on { Some(&table_store) } else { None }, "endscore");
    }
    let x4 = rng.rnd(60);
    if x4 == 0 {
      hist.end_game_if_all_pass_alive(&board);
      replayable = false;
      dump(out, &hist, &board, if table_on { Some(&table_store) } else { None }, "allpassalive");
    }
    let x5 = rng.rnd(12);
    if x5 == 0 {
      let mut ar = [0 as Color; Board::MAX_ARR_SIZE as usize];
      hist.get_area_now(&board, &mut ar);
      *out += &format!("AREA {}\n", area_str(&board, &ar));
      let mut copy = hist.clone();
      let mut ar2 = [0 as Color; Board::MAX_ARR_SIZE as usize];
      copy.end_and_score_game_now_area(&board, &mut ar2);
      *out += &format!(
        "ESC {} {} {} {} {}\n",
        ff(copy.final_white_minus_black_score),
        copy.winner,
        b(copy.has_button),
        ff(copy.white_bonus_score),
        area_str(&board, &ar2)
      );
    }
    let x6 = rng.rnd(25);
    if x6 == 0 {
      let mut o = String::new();
      hist.print_debug_info(&mut o, &board);
      hist.print_basic_info(&mut o, &board);
      *out += &o;
    }
    // Copy-assignment round trip: operator= onto a different history, then back onto hist.
    // PORT NOTE: `operator=` -> clone_from (BoardHistory implements Clone).
    let y1 = rng.rnd(20);
    if y1 == 0 {
      let mut tmp = hist.copy_to_initial();
      tmp.clone_from(&hist);
      hist.clone_from(&tmp);
      dump(out, &hist, &board, if table_on { Some(&table_store) } else { None }, "assign");
    }
    // getGraphHashFromScratch replays moveHistory from copyToInitial() (see `replayable`).
    let y2 = rng.rnd(10);
    if y2 == 0 && replayable && !hist.is_game_finished {
      let rb = rng.rnd(15);
      let gfs = GraphHash::get_graph_hash_from_scratch(&hist, hist.presumed_next_move_pla, rb, 0.5);
      *out += &format!("GFS {} {}\n", rb, hx(gfs));
    }
    let x7 = rng.rnd(15);
    if x7 == 0 && replayable {
      let mut h2 = hist.copy_to_initial();
      let mut b2 = h2.get_recent_board(0).clone();
      for i in 0..hist.move_history.len() {
        h2.make_board_move_assume_legal(&mut b2, hist.move_history[i].loc, hist.move_history[i].pla, None, hist.prevent_encore_history[i]);
      }
      *out += &format!("REPLAY eq{}\n", b(b2.pos_hash == board.pos_hash));
      dump(out, &h2, &b2, None, "replay");
    }
    if table_on {
      let x8 = rng.rnd(20);
      if x8 == 0 {
        table_store.recompute(&hist).unwrap();
        *out += &format!("RECOMPUTE {}\n", table_store.size());
      }
    }
    if !cleared {
      let x9 = rng.rnd(60);
      if x9 == 0 {
        cleared = true;
        replayable = false;
        let mut ne = 0;
        if hist.rules.scoring_rule == Rules::SCORING_TERRITORY && !hist.rules.has_button {
          let e = rng.rnd(3);
          ne = e;
        }
        // PORT NOTE: C++ passes hist.presumedNextMovePla / hist.rules by reference (aliasing); copy first.
        let p = hist.presumed_next_move_pla;
        let r = hist.rules;
        hist.clear(&board, p, &r, ne);
        table_on = false;
        dump(out, &hist, &board, None, &format!("clear{}", ne));
      }
    }
  }
  dump(out, &hist, &board, if table_on { Some(&table_store) } else { None }, "end");
}

/// `EDGE_KOMI_BITS` / `EDGE_DRAW_BITS` in boardhistory_fuzz.cpp.
const EDGE_KOMI_BITS: [u32; 16] = [
  0x40f00000, // 7.5
  0x40e00000, // 7.0
  0x7fc00000, // NaN
  0x7f800000, // +inf
  0xff800000, // -inf
  0x5bb1a2bc, // 1e17
  0xdbb1a2bc, // -1e17
  0x5c000000, // 2^57: *256 = 2^65, out of range
  0x5a000000, // 2^53: *256 = 2^61, in range
  0x5b000000, // 2^55: *256 = 2^63, first value out of range
  0x5affffff, // 2^55 - 2^31: *256 = 2^63 - 2^39, last value in range
  0xdb000000, // -2^55: *256 = -2^63, in range (INT64_MIN either way)
  0xdb000001, // -(2^55 + 2^32): *256 below -2^63, out of range
  0x5b0e1bca, // 4e16
  0x4e6e6b28, // 1e9
  0x3dcccccd, // 0.1
];
const EDGE_DRAW_BITS: [u64; 7] = [
  0x3fe0000000000000, // 0.5
  0x0000000000000000, // 0.0
  0x3fe6666666666666, // 0.7
  0x7ff8000000000000, // NaN
  0x7ff0000000000000, // +inf
  0xfff0000000000000, // -inf
  0x7e37e43c8800759c, // 1e300
];

/// `komiEdge` in boardhistory_fuzz.cpp.
fn komi_edge(out: &mut String) {
  let rules_list: [Rules; 2] = [Rules::get_tromp_taylorish(), Rules::get_simple_territory()];
  for ri in 0..2 {
    let mut board = Board::new_x_y(9, 9).unwrap();
    if ri == 1 {
      board.set_stone(Location::get_loc(2, 2, 9), C_BLACK);
      board.set_stone(Location::get_loc(6, 6, 9), C_WHITE);
    }
    let mut hist = BoardHistory::new_full(&board, P_BLACK, &rules_list[ri], 0);
    for kb in EDGE_KOMI_BITS {
      let k = f32::from_bits(kb);
      hist.set_komi(k);
      let mut s = format!("E r{} k{} int{}", ri, fx(k), b(hist.rules.game_result_will_be_integer()));
      for db in EDGE_DRAW_BITS {
        let d = f64::from_bits(db);
        for p in 1..=2 as Player {
          s += &format!(" {}", fx(hist.current_self_komi(p, d)));
          s += &format!(",{}", hx(BoardHistory::get_situation_rules_and_ko_hash(&board, &hist, p, d)));
          s += &format!(",{}", hx(GraphHash::get_state_hash(&hist, p, d)));
        }
      }
      *out += &s;
      *out += "\n";
    }
  }
}

fn fnv1a64(s: &str) -> u64 {
  let mut h: u64 = 1469598103934665603;
  for c in s.bytes() {
    h ^= c as u64;
    h = h.wrapping_mul(1099511628211);
  }
  h
}

/// `boardhistory_fuzz hashes <n> 1`: every game's trace must hash like the C++ one.
#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn boardhistory_fuzz_game_hashes_vs_cpp() {
  init();
  let mut cases = Cases::new("boardhistory_fuzz_oracle::boardhistory_fuzz_game_hashes_vs_cpp");
  let expected = fixture("oracle", "boardhistory_fuzz_hashes.txt");
  let mut failed: Vec<String> = Vec::new();
  let mut first_bad: Option<String> = None;
  for line in expected.lines() {
    let f: Vec<&str> = line.split(' ').collect();
    assert_eq!(f[0], "H", "bad fixture line {}", line);
    let g: i32 = f[1].parse().unwrap();
    let mut out = String::new();
    run_game(&mut out, g, 1);
    let actual = format!("H {} {} {:016x}", g, out.len(), fnv1a64(&out));
    if actual != line {
      failed.push(format!("game {}: expected [{}] got [{}]", g, line, actual));
      if first_bad.is_none() {
        first_bad = Some(out);
      }
    }
    cases.ran();
  }
  #[cfg(not(target_arch = "wasm32"))]
  if let (Some(trace), Ok(path)) = (&first_bad, std::env::var("BHFUZZ_DUMP")) {
    std::fs::write(&path, trace).unwrap();
  }
  #[cfg(target_arch = "wasm32")]
  let _ = &first_bad;
  assert!(
    failed.is_empty(),
    "{} games differ from C++ (set BHFUZZ_DUMP=<file> for the first Rust trace; C++: boardhistory_fuzz games <g+1> 1):\n{}",
    failed.len(),
    failed.join("\n")
  );
}

/// `boardhistory_fuzz games <n> 7`: full traces, compared line by line.
#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn boardhistory_fuzz_full_traces_vs_cpp() {
  init();
  let mut cases = Cases::new("boardhistory_fuzz_oracle::boardhistory_fuzz_full_traces_vs_cpp");
  let expected = fixture("oracle", "boardhistory_fuzz_trace.txt");
  let num_games = expected.lines().filter(|l| l.starts_with("G ")).count() as i32;
  let mut out = String::new();
  for g in 0..num_games {
    run_game(&mut out, g, 7);
    cases.ran();
  }
  compare_lines("boardhistory_fuzz_trace.txt", &out, &expected);
}

/// `boardhistory_fuzz komiedge`: NaN/inf/out-of-range komi and draw values. The `(int64_t)` cast in
/// getSituationRulesAndKoHash is UB there; x86-64 gives INT64_MIN (port::cpp::static_cast_i64_f32).
#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn boardhistory_komi_edge_vs_cpp() {
  init();
  let mut cases = Cases::new("boardhistory_fuzz_oracle::boardhistory_komi_edge_vs_cpp");
  let expected = fixture("oracle", "boardhistory_fuzz_komiedge.txt");
  let mut out = String::new();
  komi_edge(&mut out);
  for _ in out.lines() {
    cases.ran();
  }
  compare_lines("boardhistory_fuzz_komiedge.txt", &out, &expected);
}
