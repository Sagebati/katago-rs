//! search/localpattern against the unmodified upstream LocalPatternHasher
//! (fixtures/upstream-v1.16.5/unit/oracle/localpattern.txt, from oracle/localpattern.cpp): init's RNG
//! stream and tables, getHash, and getHashWithSym over every symmetry and color flip, on random boards.
//! Not a port of upstream code. Keep in the same order as the oracle.

mod common;

use common::Cases;
use common::compare_lines;
use common::fixture;
use common::init;
use katago_core::core::hash::Hash128;
use katago_core::core::rand::Rand;
use katago_core::game::board::get_opp;
use katago_core::game::board::Board;
use katago_core::game::board::Loc;
use katago_core::game::board::Location;
use katago_core::game::board::Player;
use katago_core::game::board::P_BLACK;
use katago_core::game::board::P_WHITE;
use katago_core::neuralnet::nninputs::SymmetryHelpers;
use katago_core::search::localpattern::LocalPatternHasher;

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
  fn rnd(&mut self, n: i32) -> i32 {
    (self.nxt() % (n as u64)) as i32
  }
}

fn fnv_u64(h: &mut u64, v: u64) {
  for b in 0..8 {
    *h ^= (v >> (8 * b)) & 0xFF;
    *h = h.wrapping_mul(0x100000001b3);
  }
}

fn fnv_hash(h: &mut u64, x: &Hash128) {
  fnv_u64(h, x.hash0);
  fnv_u64(h, x.hash1);
}

const FNV0: u64 = 0xcbf29ce484222325;

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn localpattern_vs_cpp() {
  let mut cases = Cases::new("localpattern_oracle::localpattern_vs_cpp");
  init();
  let expected = fixture("oracle", "localpattern.txt");
  let mut out = String::new();
  let mut r = Splitmix { s: 0x4c6f63616c506174 };

  let hcfgs: [(i32, i32, &str); 6] = [
    (5, 5, "lp seed 1"),
    (3, 3, "lp seed 2"),
    (1, 1, "lp seed 3"),
    (7, 5, "lp seed 4"),
    (5, 7, "lp seed 5"),
    (9, 9, "lp seed 6"),
  ];
  let sizes: [(i32, i32); 6] = [(9, 9), (13, 13), (19, 19), (9, 13), (19, 9), (11, 15)];

  //Board snapshots
  let mut boards: Vec<Box<Board>> = Vec::new();
  for sz in sizes.iter() {
    let mut board = Board::new_x_y(sz.0, sz.1).unwrap();
    let mut pla: Player = P_BLACK;
    let area = sz.0 * sz.1;
    let snap_every = area / 3;
    for i in 1..=snap_every * 6 {
      let mut loc: Loc = Board::PASS_LOC;
      if r.rnd(40) != 0 {
        for _tries in 0..50 {
          let x = r.rnd(sz.0);
          let y = r.rnd(sz.1);
          let cand = Location::get_loc(x, y, sz.0);
          if board.is_legal(cand, pla, true) {
            loc = cand;
            break;
          }
        }
      }
      board.play_move_assume_legal(loc, pla);
      pla = get_opp(pla);
      if i % snap_every == 0 {
        boards.push(Box::new(board.clone()));
      }
    }
  }
  for (bi, b) in boards.iter().enumerate() {
    out.push_str(&format!("B {} {} {} {} {} {}\n", bi, b.x_size, b.y_size, b.num_stones_on_board(), b.ko_loc as i32, b.pos_hash.to_string()));
    cases.ran();
  }

  for (hi, hc) in hcfgs.iter().enumerate() {
    let mut rand = Rand::new_seed(hc.2);
    let mut hasher = LocalPatternHasher::new();
    hasher.init(hc.0, hc.1, &mut rand);
    let mut f0 = FNV0;
    let mut f1 = FNV0;
    let mut f2 = FNV0;
    for x in hasher.zobrist_local_pattern.iter() {
      fnv_hash(&mut f0, x);
    }
    for x in hasher.zobrist_pla.iter() {
      fnv_hash(&mut f1, x);
    }
    for x in hasher.zobrist_atari.iter() {
      fnv_hash(&mut f2, x);
    }
    out.push_str(&format!(
      "T {} {} {} {} {} {} {:016X} {:016X} {:016X} {} {}\n",
      hi,
      hasher.x_size,
      hasher.y_size,
      hasher.zobrist_local_pattern.len(),
      hasher.zobrist_pla.len(),
      hasher.zobrist_atari.len(),
      f0,
      f1,
      f2,
      hasher.zobrist_local_pattern.last().unwrap().to_string(),
      rand.get_num_calls()
    ));
    cases.ran();

    for (bi, b) in boards.iter().enumerate() {
      if b.x_size < hc.0 || b.y_size < hc.1 {
        continue;
      }
      let mut fa = FNV0;
      let mut fb = FNV0;
      let mut n: i32 = 0;
      let mut locs: Vec<Loc> = Vec::new();
      locs.push(Board::PASS_LOC);
      locs.push(Board::NULL_LOC);
      for y in 0..b.y_size {
        for x in 0..b.x_size {
          locs.push(Location::get_loc(x, y, b.x_size));
        }
      }
      for &loc in locs.iter() {
        for pla in [P_BLACK, P_WHITE] {
          let h = hasher.get_hash(b, loc, pla);
          fnv_hash(&mut fa, &h);
          n += 1;
          for sym in 0..SymmetryHelpers::NUM_SYMMETRIES {
            for flip in 0..2 {
              let hs = hasher.get_hash_with_sym(b, loc, pla, sym, flip != 0);
              fnv_hash(&mut fb, &hs);
              if n % 97 == 0 && sym == ((bi + hi) % 8) as i32 {
                out.push_str(&format!("H {} {} {} {} {} {} {} {}\n", hi, bi, loc as i32, pla as i32, h.to_string(), hs.to_string(), sym, flip));
              }
            }
          }
        }
      }
      out.push_str(&format!("C {} {} {} {:016X} {:016X}\n", hi, bi, n, fa, fb));
      cases.ran();
    }
  }

  compare_lines("localpattern", &out, &expected);
}
