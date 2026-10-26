//! search/subtreevaluebiastable against the unmodified upstream SubtreeValueBiasTable
//! (fixtures/upstream-v1.18.2/unit/oracle/subtreevaluebiastable.txt, from oracle/subtreevaluebiastable.cpp):
//! the file-static zobrist tables and LocalPatternHasher (observed through the keys get() stores), the
//! shard choice, shared entries, BTreeMap order (Hash128 order) and clearUnusedSynchronous, on random
//! boards and hand-made ko positions. Not a port of upstream code. Keep in the same order as the oracle.

#[macro_use]
mod common;

use std::rc::Rc;

use common::Cases;
use common::compare_lines;
use common::dbits;
use common::fixture;
use common::init;
use katago_core::game::board::get_opp;
use katago_core::game::board::Board;
use katago_core::game::board::Loc;
use katago_core::game::board::Location;
use katago_core::game::board::Player;
use katago_core::game::board::C_BLACK;
use katago_core::game::board::C_WHITE;
use katago_core::game::board::P_BLACK;
use katago_core::game::board::P_WHITE;
use katago_core::search::subtreevaluebiastable::SubtreeValueBiasEntry;
use katago_core::search::subtreevaluebiastable::SubtreeValueBiasTable;

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

struct PoolItem {
  board_idx: usize,
  pla: Player,
  parent_loc: Loc,
  prev_loc: Loc,
}

fn rand_loc(r: &mut Splitmix, b: &Board) -> Loc {
  let k = r.rnd(12);
  if k == 0 {
    return Board::PASS_LOC;
  }
  if k == 1 {
    return Board::NULL_LOC;
  }
  let x = r.rnd(b.x_size);
  let y = r.rnd(b.y_size);
  Location::get_loc(x, y, b.x_size)
}

/// `dumpEntries` in the oracle
fn dump_entries(shards: i32, step: i32, table: &SubtreeValueBiasTable, values: bool, out: &mut String) {
  let entries = table.entries.borrow();
  let mut total = 0;
  for s in 0..entries.len() {
    total += entries[s].len() as i32;
  }
  out.push_str(&format!("{} {} {} {}\n", if values { "N" } else { "A" }, shards, step, total));
  for s in 0..entries.len() {
    for (k, v) in entries[s].iter() {
      if values {
        out.push_str(&format!(
          "V {} {} {} {} {} {}\n",
          shards,
          s,
          k.to_string(),
          dbits(v.delta_utility_sum.get()),
          dbits(v.weight_sum.get()),
          Rc::strong_count(v)
        ));
      } else {
        out.push_str(&format!("K {} {} {} {}\n", shards, step, s, k.to_string()));
      }
    }
  }
}

wasm_test! {
fn subtreevaluebiastable_vs_cpp() {
  let mut cases = Cases::new("subtreevaluebiastable_oracle::subtreevaluebiastable_vs_cpp");
  init();
  let expected = fixture("oracle", "subtreevaluebiastable.txt");
  let mut out = String::new();
  let mut r = Splitmix { s: 0x5375627472656521 };

  let mut boards: Vec<Box<Board>> = Vec::new();
  {
    let sizes: [(i32, i32); 3] = [(9, 9), (19, 19), (13, 7)];
    for sz in sizes.iter() {
      let mut board = Board::new_x_y(sz.0, sz.1).unwrap();
      let mut pla: Player = P_BLACK;
      let area = sz.0 * sz.1;
      let snap_every = area / 3;
      for i in 1..=snap_every * 4 {
        let mut loc: Loc = Board::PASS_LOC;
        if r.rnd(40) != 0 {
          for _tries in 0..50 {
            let cx = r.rnd(sz.0);
            let cy = r.rnd(sz.1);
            let cand = Location::get_loc(cx, cy, sz.0);
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
    //Hand-made simple ko on 9x9, black to have just captured at (2,1); ko point (1,1), and the same shifted
    for shift in 0..2 {
      let mut b = Board::new_x_y(9, 9).unwrap();
      let ox = shift * 4;
      let oy = shift * 3;
      let l = |x: i32, y: i32| Location::get_loc(x + ox, y + oy, 9);
      b.set_stone(l(1, 0), C_BLACK);
      b.set_stone(l(0, 1), C_BLACK);
      b.set_stone(l(1, 2), C_BLACK);
      b.set_stone(l(1, 1), C_WHITE);
      b.set_stone(l(2, 0), C_WHITE);
      b.set_stone(l(3, 1), C_WHITE);
      b.set_stone(l(2, 2), C_WHITE);
      b.play_move_assume_legal(l(2, 1), P_BLACK);
      boards.push(Box::new(b));
    }
  }
  for (bi, b) in boards.iter().enumerate() {
    out.push_str(&format!("B {} {} {} {} {} {}\n", bi, b.x_size, b.y_size, b.num_stones_on_board(), b.ko_loc as i32, b.pos_hash.to_string()));
    cases.ran();
  }

  //Pool of get() arguments, with repeats drawn from it
  let mut pool: Vec<PoolItem> = Vec::new();
  for k in 0..90 {
    let board_idx: usize = if k < 10 { boards.len() - 1 - (k % 2) } else { r.rnd(boards.len() as i32) as usize };
    let b = &boards[board_idx];
    let pla: Player = if r.rnd(2) == 0 { P_BLACK } else { P_WHITE };
    let parent_loc = rand_loc(&mut r, b);
    let mut prev_loc = rand_loc(&mut r, b);
    if k < 10 {
      //ko boards: prevLoc is the ko capture, so ko_loc != NULL_LOC is part of the hash
      let last = board_idx == boards.len() - 1;
      prev_loc = Location::get_loc(2 + if last { 4 } else { 0 }, 1 + if last { 3 } else { 0 }, 9);
    }
    pool.push(PoolItem { board_idx, pla, parent_loc, prev_loc });
  }

  for shards in [1, 5, 64, 65536] {
    let mut table = SubtreeValueBiasTable::new(shards);
    let mut held: Vec<Option<Rc<SubtreeValueBiasEntry>>> = Vec::new();
    let mut seen: Vec<(i32, String)> = Vec::new();
    let num_calls = if shards == 65536 { 120 } else { 400 };
    for i in 0..num_calls {
      let pi = r.rnd(pool.len() as i32) as usize;
      let it = &pool[pi];
      let e: Rc<SubtreeValueBiasEntry> = table.get(it.pla, it.parent_loc, it.prev_loc, &boards[it.board_idx]);
      let mut found_shard: i32 = -1;
      let mut found_hash = String::new();
      {
        let entries = table.entries.borrow();
        let mut s = 0;
        while s < entries.len() && found_shard < 0 {
          for (k, v) in entries[s].iter() {
            if Rc::ptr_eq(v, &e) {
              found_shard = s as i32;
              found_hash = k.to_string();
              break;
            }
          }
          s += 1;
        }
      }
      let mut is_new = true;
      for sp in seen.iter() {
        if sp.0 == found_shard && sp.1 == found_hash {
          is_new = false;
        }
      }
      if is_new {
        seen.push((found_shard, found_hash.clone()));
      }
      out.push_str(&format!("G {} {} {} {} {} {} {}\n", shards, i, pi, found_shard, found_hash, is_new as i32, Rc::strong_count(&e)));
      e.delta_utility_sum.set(e.delta_utility_sum.get() + (i as f64 * 0.5 + 1.0));
      e.weight_sum.set(e.weight_sum.get() + (1.0 + 0.125 * ((i % 7) as f64)));
      if i % 3 == 0 {
        held.push(Some(e));
      }
      cases.ran();
    }
    dump_entries(shards, 0, &table, true, &mut out);
    table.clear_unused_synchronous();
    dump_entries(shards, 1, &table, false, &mut out);
    //Release every other held entry, clear again
    let mut k = 0;
    while k < held.len() {
      held[k] = None;
      k += 2;
    }
    table.clear_unused_synchronous();
    dump_entries(shards, 2, &table, false, &mut out);
    held.clear();
    table.clear_unused_synchronous();
    dump_entries(shards, 3, &table, false, &mut out);
    cases.ran();
  }

  compare_lines("subtreevaluebiastable", &out, &expected);
}
}
