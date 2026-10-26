//! Differential fuzz regression oracle for game/board against the real C++
//! (fixtures/upstream-v1.18.2/unit/oracle/board_fuzz.cpp, adopted from the pilot board review).
//! Mirrors board_fuzz.cpp line for line: same splitmix stream, same queries, same output text.
//!   board_fuzz_hashes.txt  one FNV-1a 64 of each game's full trace (seed0 = 1)
//!   board_fuzz_trace.txt   full traces of a few games (seed0 = 7), compared line by line
//!   board_fuzz_loc.txt     Location::tryOfString on malformed / overflowing input

// Test code keeps the shapes of upstream's tests and oracle programs (PORTING.md §11b).
#![allow(clippy::needless_late_init)]

mod common;

use common::Cases;
use common::compare_lines;
use common::fixture;
use common::init;
use katago_core::core::global::Global;
use katago_core::game::board::*;

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
fn bi(x: bool) -> String {
  (x as i32).to_string()
}
fn cons(b: &Board) -> String {
  match b.check_consistency() {
    Ok(()) => "ok".to_string(),
    Err(e) => format!("ERR {}", e),
  }
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

fn analyze(out: &mut String, rng: &mut R, b: &mut Board, sui: bool, tag: i32) {
  *out += &format!("A {}{}\n", tag, st(b));
  let mut buf: Vec<Loc> = Vec::new();
  let mut buf2: Vec<Loc> = Vec::new();
  let mut area = [0 as Color; Board::MAX_ARR_SIZE as usize];
  for m in 0..16 {
    b.calculate_area(&mut area, m & 1 != 0, (m >> 1) & 1 != 0, (m >> 2) & 1 != 0, (m >> 3) & 1 != 0);
    *out += &format!("CA {} {}\n", m, area_str(b, &area));
  }
  // m < 8: excludeTerritoryAdjacentToAtari = false (the v1.16.5 behavior); m >= 8: the v1.18.2 rules-v3 mode.
  for m in 0..16 {
    let mut cnt: i32 = -999;
    b.calculate_independent_life_area(&mut area, &mut cnt, m & 1 != 0, (m >> 1) & 1 != 0, (m >> 3) & 1 != 0, (m >> 2) & 1 != 0);
    *out += &format!("IL {} {} {}\n", m, cnt, area_str(b, &area));
  }
  let mut pa = [0 as Color; Board::MAX_ARR_SIZE as usize];
  b.calculate_area(&mut pa, false, false, false, sui);
  for y in 0..b.y_size {
    for x in 0..b.x_size {
      let loc = Location::get_loc(x, y, b.x_size);
      let mut s = format!("L {}", loc);
      for bd in [0, 1, 2, 3, 5, 8, 13, 30] {
        s += if b.simple_repetition_bound_gt(loc, bd) { "1" } else { "0" };
      }
      if b.colors[loc as usize] == C_EMPTY {
        for p in 1..=2 as Player {
          s += &format!(" p{}:", p);
          s += &(bi(b.is_legal(loc, p, false)) + &bi(b.is_legal(loc, p, true)) + &bi(b.is_legal_ignoring_ko(loc, p, false)) + &bi(b.is_legal_ignoring_ko(loc, p, true)));
          s += &(bi(b.is_suicide(loc, p)) + &bi(b.is_illegal_suicide(loc, p, false)) + &bi(b.is_illegal_suicide(loc, p, true)));
          s += &(bi(b.is_simple_eye(loc, p)) + &bi(b.would_be_capture(loc, p)) + &bi(b.would_be_ko_capture(loc, p)) + "," + &b.get_ko_capture_loc(loc, p).to_string());
          s += &(String::from(",") + &bi(b.is_adjacent_to_pla(loc, p)) + &bi(b.is_adjacent_or_diagonal_to_pla(loc, p)) + &bi(b.is_non_pass_alive_self_connection(loc, p, &pa)));
          for mx in [1, 2, 3, 4, 6, 1000] {
            s += &format!(",{}", b.get_num_liberties_after_play(loc, p, mx));
          }
          let mut lo = -5;
          let mut hi = -5;
          b.get_bound_num_liberties_after_play(loc, p, &mut lo, &mut hi);
          s += &format!(",{},{}", lo, hi);
          let h = b.get_pos_hash_after_move(loc, p);
          s += &format!(",{}", Global::uint64_to_hex_string(h.hash0 ^ h.hash1));
          s += &format!(",{}", b.get_num_immediate_liberties(loc));
        }
        let cx = rng.rnd(b.x_size);
        let cy = rng.rnd(b.y_size);
        let ch = Location::get_loc(cx, cy, b.x_size);
        s += &format!(" adjch{}", bi(b.is_adjacent_to_chain(loc, ch)));
      } else {
        s += &format!(" libs{} sz{}", b.get_num_liberties(loc), b.get_chain_size(loc));
      }
      *out += &s;
      *out += "\n";
    }
  }
  for y in 0..b.y_size {
    for x in 0..b.x_size {
      let loc = Location::get_loc(x, y, b.x_size);
      if b.colors[loc as usize] == C_EMPTY {
        continue;
      }
      let r1 = b.search_is_ladder_captured(loc, true, &mut buf);
      let mut s = format!("LD {} {} b{}{}", loc, bi(r1), buf.len(), st(b));
      let r2 = b.search_is_ladder_captured(loc, false, &mut buf);
      s += &format!(" | {} b{}{}", bi(r2), buf.len(), st(b));
      let r3 = b.search_is_ladder_captured_attacker_first2_libs(loc, &mut buf, &mut buf2);
      s += &format!(" | {} b{} w", bi(r3), buf.len());
      for w in buf2.iter() {
        s += &format!("{},", w);
      }
      s += &st(b);
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

/// `runGame` in board_fuzz.cpp.
fn run_game(out: &mut String, g: i32, seed0: u64) {
  let sizes: [[i32; 2]; 15] =
    [[19, 19], [9, 9], [13, 7], [5, 17], [2, 2], [1, 1], [3, 3], [19, 1], [1, 19], [7, 7], [4, 6], [19, 13], [6, 6], [11, 11], [2, 9]];
  let num_sizes = 15;
  let mut rng = R { s: seed0.wrapping_mul(1000003).wrapping_add(g as u64) };
  let si = rng.rnd(num_sizes);
  let xs = sizes[si as usize][0];
  let ys = sizes[si as usize][1];
  let sui = rng.rnd(2) == 1;
  let mut b = Board::new_x_y(xs, ys).unwrap();
  *out += &format!("G {} {} {} {}\n", g, xs, ys, bi(sui));
  let area = xs * ys;
  let steps = rng.rnd(area * 2 + 2) + 1;
  let chase_prob = rng.rnd(90);
  let mut recs: Vec<MoveRecord> = Vec::new();
  let mut pla: Player = P_BLACK;
  let mut next_analysis = rng.rnd(area / 2 + 2);
  for n in 0..steps {
    let mut loc: Loc = Board::PASS_LOC;
    let r = rng.rnd(100);
    let mut found = false;
    if r >= 2 && r < 2 + chase_prob {
      // chase: liberties of chains with <= 2 libs
      let mut cands: Vec<Loc> = Vec::new();
      for y in 0..ys {
        for x in 0..xs {
          let l = Location::get_loc(x, y, xs);
          if b.colors[l as usize] != C_EMPTY {
            continue;
          }
          for i in 0..4 {
            let a = (l as i32 + b.adj_offsets[i] as i32) as Loc;
            if (b.colors[a as usize] == C_BLACK || b.colors[a as usize] == C_WHITE) && b.get_num_liberties(a) <= 2 {
              cands.push(l);
              break;
            }
          }
        }
      }
      if !cands.is_empty() {
        let l = cands[rng.rnd(cands.len() as i32) as usize];
        if b.is_legal(l, pla, sui) {
          loc = l;
          found = true;
        }
      }
    }
    if !found && r >= 2 && area > 0 {
      for _t in 0..60 {
        let lx = rng.rnd(xs);
        let ly = rng.rnd(ys);
        let l = Location::get_loc(lx, ly, xs);
        if b.is_legal(l, pla, sui) {
          loc = l;
          break;
        }
      }
    }
    recs.push(b.play_move_recorded(loc, pla));
    let rec = *recs.last().unwrap();
    *out += &format!(
      "M {} {} {} cd{} rk{}{} c{},{}\n",
      n,
      pla,
      loc,
      rec.cap_dirs,
      rec.ko_loc,
      st(&b),
      b.num_black_captures,
      b.num_white_captures
    );
    if n == next_analysis {
      analyze(out, &mut rng, &mut b, sui, n);
      next_analysis = n + 1 + rng.rnd(area / 2 + 2);
    }
    pla = if rng.rnd(10) < 9 { get_opp(pla) } else { pla };
  }
  analyze(out, &mut rng, &mut b, sui, -1);
  // undo some
  let nu = rng.rnd(recs.len() as i32 + 1);
  for _k in 0..nu {
    let rec = recs.pop().unwrap();
    b.undo(rec);
    *out += &format!("U{} c{},{}\n", st(&b), b.num_black_captures, b.num_white_captures);
  }
  if nu > 0 {
    analyze(out, &mut rng, &mut b, sui, -2);
  }
  // setStone fuzz
  if area > 0 {
    let ns = rng.rnd(40);
    for _k in 0..ns {
      let lx = rng.rnd(xs);
      let ly = rng.rnd(ys);
      let l = Location::get_loc(lx, ly, xs);
      let c = rng.rnd(3) as Color;
      let which = rng.rnd(2);
      let ok = if which == 0 { b.set_stone(l, c) } else { b.set_stone_fail_if_no_libs(l, c) };
      *out += &format!("S {} {} {} {}{}\n", which, l, c, bi(ok), st(&b));
    }
    analyze(out, &mut rng, &mut b, sui, -3);
  }
  // regenChainsFromColors on a copy, then setStonesTolerant (v1.18.2): random placements, including
  // off-board / wall / out-of-range locations and invalid colors (ignored), duplicates (last one wins),
  // and dense fills that leave zero-liberty groups to be removed simultaneously.
  {
    let mut rg = b.clone();
    rg.regen_chains_from_colors();
    *out += &format!("RG{} {}\n", st(&rg), cons(&rg));
    let nt = rng.rnd(4) + 1;
    for _k in 0..nt {
      let np = rng.rnd(area + 4);
      let mut pl: Vec<Move> = Vec::new();
      let mut ps = String::new();
      for _j in 0..np {
        let kind = rng.rnd(12);
        let l: Loc;
        if kind == 0 {
          let z = rng.rnd(Board::MAX_ARR_SIZE + 40);
          l = (z - 20) as Loc;
        } else {
          let lx = rng.rnd(xs);
          let ly = rng.rnd(ys);
          l = Location::get_loc(lx, ly, xs);
        }
        let cc = rng.rnd(8);
        let c: Color;
        if cc <= 2 {
          c = C_BLACK;
        } else if cc <= 5 {
          c = C_WHITE;
        } else if cc == 6 {
          c = C_EMPTY;
        } else {
          let bad = rng.rnd(3);
          c = if bad == 0 { -1 } else if bad == 1 { C_WALL } else { 4 };
        }
        pl.push(Move::new(l, c));
        ps += &format!("{}:{},", l, c);
      }
      let nr = b.set_stones_tolerant(&pl);
      *out += &format!("T {} {} {}{} {}\n", np, ps, nr, st(&b), cons(&b));
    }
    analyze(out, &mut rng, &mut b, sui, -4);
  }
}

/// `board_fuzz hashes <n> 1`: every game's trace must hash like the C++ one.
#[test]
fn board_fuzz_game_hashes_vs_cpp() {
  init();
  let mut cases = Cases::new("board_fuzz_oracle::board_fuzz_game_hashes_vs_cpp");
  let expected = fixture("oracle", "board_fuzz_hashes.txt");
  let mut failed: Vec<String> = Vec::new();
  for line in expected.lines() {
    let f: Vec<&str> = line.split(' ').collect();
    assert_eq!(f[0], "H", "bad fixture line {}", line);
    let g: i32 = f[1].parse().unwrap();
    let mut out = String::new();
    run_game(&mut out, g, 1);
    let actual = format!("H {} {} {:016x}", g, out.len(), fnv1a64(&out));
    if actual != line {
      failed.push(format!("game {}: expected [{}] got [{}] (rerun board_fuzz games for this seed to diff)", g, line, actual));
    }
    cases.ran();
  }
  assert!(failed.is_empty(), "{} games differ from C++:\n{}", failed.len(), failed.join("\n"));
}

/// `board_fuzz games <n> 7`: full traces, compared line by line.
#[test]
fn board_fuzz_full_traces_vs_cpp() {
  init();
  let mut cases = Cases::new("board_fuzz_oracle::board_fuzz_full_traces_vs_cpp");
  let expected = fixture("oracle", "board_fuzz_trace.txt");
  let num_games = expected.lines().filter(|l| l.starts_with("G ")).count() as i32;
  let mut out = String::new();
  for g in 0..num_games {
    run_game(&mut out, g, 7);
    cases.ran();
  }
  compare_lines("board_fuzz_trace.txt", &out, &expected);
}

/// `board_fuzz loc`: getLoc and `y = y_size - y` overflow on user input (C++ UB, wraps on x86-64).
/// Must not panic in debug builds and must give the C++ result.
#[test]
fn location_try_of_string_overflow_vs_cpp() {
  init();
  let mut cases = Cases::new("board_fuzz_oracle::location_try_of_string_overflow_vs_cpp");
  let expected = fixture("oracle", "board_fuzz_loc.txt");
  let mut out = String::new();
  for line in expected.lines() {
    // "tryOfString <xs> <ys> [<input>] -> <ok> <loc>"
    let open = line.find('[').unwrap();
    let close = line.rfind(']').unwrap();
    let dims: Vec<i32> = line[..open].split_whitespace().skip(1).map(|t| t.parse().unwrap()).collect();
    let input = &line[open + 1..close];
    let mut l: Loc = -7;
    let ok = Location::try_of_string_x_size_y_size(input, dims[0], dims[1], &mut l);
    out += &format!("tryOfString {} {} [{}] -> {} {}\n", dims[0], dims[1], input, bi(ok), l as i32);
    cases.ran();
  }
  compare_lines("board_fuzz_loc.txt", &out, &expected);
}
