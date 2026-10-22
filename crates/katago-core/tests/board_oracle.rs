//! game/board zobrist tables and position hashes against the real C++
//! (fixtures/upstream-v1.16.5/unit/oracle/board.txt). Mirrors `boardMode()` in oracle.cpp.

mod common;

use common::Cases;
use common::fixture;
use common::init;
use katago_core::core::global::Global;
use katago_core::core::hash::Hash;
use katago_core::core::hash::Hash128;
use katago_core::core::rand::Rand;
use katago_core::game::board::*;

fn fold_table(t: &[Hash128]) -> Hash128 {
  let mut acc = Hash128::default();
  for (i, x) in t.iter().enumerate() {
    acc ^= *x;
    acc = Hash128::new(Hash::murmur_mix(acc.hash0.wrapping_add(i as u64)), Hash::split_mix64(acc.hash1));
  }
  acc
}

#[test]
fn board_hashes_vs_cpp() {
  init();
  let mut cases = Cases::new("board_oracle::board_hashes_vs_cpp");
  let expected = fixture("oracle", "board.txt");
  let mut out = String::new();
  let st = Board::statics();
  let flat = |t: &[[Hash128; 4]]| -> Vec<Hash128> { t.iter().flat_map(|r| r.iter().copied()).collect() };
  out += &format!("fold ZOBRIST_PLAYER_HASH {}\n", fold_table(&st.ZOBRIST_PLAYER_HASH));
  out += &format!("fold ZOBRIST_ENCORE_HASH {}\n", fold_table(&st.ZOBRIST_ENCORE_HASH));
  out += &format!("fold ZOBRIST_SIZE_X_HASH {}\n", fold_table(&st.ZOBRIST_SIZE_X_HASH));
  out += &format!("fold ZOBRIST_SIZE_Y_HASH {}\n", fold_table(&st.ZOBRIST_SIZE_Y_HASH));
  out += &format!("fold ZOBRIST_BOARD_HASH {}\n", fold_table(&flat(&st.ZOBRIST_BOARD_HASH)));
  out += &format!("fold ZOBRIST_BOARD_HASH2 {}\n", fold_table(&flat(&st.ZOBRIST_BOARD_HASH2)));
  out += &format!("fold ZOBRIST_KO_LOC_HASH {}\n", fold_table(&st.ZOBRIST_KO_LOC_HASH));
  out += &format!("fold ZOBRIST_KO_MARK_HASH {}\n", fold_table(&flat(&st.ZOBRIST_KO_MARK_HASH)));
  out += &format!("fold ZOBRIST_SECOND_ENCORE_START_HASH {}\n", fold_table(&flat(&st.ZOBRIST_SECOND_ENCORE_START_HASH)));
  for i in 0..4 {
    out += &format!("ZOBRIST_PLAYER_HASH {} {}\n", i, st.ZOBRIST_PLAYER_HASH[i]);
  }
  let spots: [usize; 6] = [0, 1, 21, 22, 200, (Board::MAX_ARR_SIZE - 1) as usize];
  for i in spots {
    out += &format!(
      "spot {} B1 {} B2 {} KO {} H2_0 {} H2_3 {} SE1 {}\n",
      i,
      st.ZOBRIST_BOARD_HASH[i][1],
      st.ZOBRIST_BOARD_HASH[i][2],
      st.ZOBRIST_KO_LOC_HASH[i],
      st.ZOBRIST_BOARD_HASH2[i][0],
      st.ZOBRIST_BOARD_HASH2[i][3],
      st.ZOBRIST_SECOND_ENCORE_START_HASH[i][1]
    );
  }
  cases.ran();

  let sizes: [[i32; 2]; 7] = [[19, 19], [9, 9], [13, 7], [5, 17], [2, 2], [1, 1], [0, 0]];
  for s in 0..7 {
    let xs = sizes[s][0];
    let ys = sizes[s][1];
    let mut board = Board::new_x_y(xs, ys).unwrap();
    let suicide_legal = s % 2 == 0;
    let mut rand = Rand::new_seed(&(String::from("board oracle ") + &Global::int_to_string(xs) + "x" + &Global::int_to_string(ys)));
    out += &format!("BOARD {} {} suicide {} start {}\n", xs, ys, suicide_legal as i32, board.pos_hash);
    let mut records: Vec<MoveRecord> = Vec::new();
    let mut pla: Player = P_BLACK;
    let steps = if xs * ys < 10 { 40 } else { 300 };
    for n in 0..steps {
      let mut loc: Loc;
      if rand.next_bool(0.03) {
        loc = Board::PASS_LOC;
      } else {
        let mut tries = 0;
        loop {
          loc = rand.next_uint_n(Board::MAX_ARR_SIZE as u32) as Loc;
          tries += 1;
          if board.is_legal(loc, pla, suicide_legal) {
            break;
          }
          if tries > 2000 {
            loc = Board::PASS_LOC;
            break;
          }
        }
      }
      let predicted = board.get_pos_hash_after_move(loc, pla);
      let was_suicide = board.is_suicide(loc, pla);
      let was_capture = board.would_be_capture(loc, pla);
      let was_ko_capture = board.would_be_ko_capture(loc, pla);
      let libs_after = if loc == Board::PASS_LOC { -1 } else { board.get_num_liberties_after_play(loc, pla, 4) };
      records.push(board.play_move_recorded(loc, pla));
      out += &format!(
        "{} {} {} pred {} pos {} sit {} ko {} caps {} {} flags {}{}{} libs {} stones {}\n",
        n,
        PlayerIO::player_to_string_short(pla),
        Location::to_string(loc, &board),
        predicted,
        board.pos_hash,
        board.get_sit_hash_with_simple_ko(get_opp(pla)),
        Location::to_string(board.ko_loc, &board),
        board.num_black_captures,
        board.num_white_captures,
        was_suicide as i32,
        was_capture as i32,
        was_ko_capture as i32,
        libs_after,
        board.num_stones_on_board()
      );
      board.check_consistency().unwrap();
      pla = if rand.next_bool(0.9) { get_opp(pla) } else { pla };
    }
    Board::print_board(&mut out, &board, Board::NULL_LOC, None);
    let len = records.len() as i32;
    let mut n = len - 1;
    while n >= 0 && n >= len - 60 {
      board.undo(records[n as usize]);
      board.check_consistency().unwrap();
      out += &format!(
        "undo {} pos {} ko {} caps {} {}\n",
        n,
        board.pos_hash,
        Location::to_string(board.ko_loc, &board),
        board.num_black_captures,
        board.num_white_captures
      );
      n -= 1;
    }
    cases.ran();
  }
  common::compare_lines("oracle/board.txt", &out, &expected);
}
