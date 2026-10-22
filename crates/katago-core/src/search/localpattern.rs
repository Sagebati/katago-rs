// Ported from KataGo cpp/search/localpattern.cpp, cpp/search/localpattern.h @ ba938676d7f42d70950b3a535af2466fb642008c

use crate::core::hash::Hash128;
use crate::core::rand::Rand;
use crate::game::board::get_opp;
use crate::game::board::Board;
use crate::game::board::Loc;
use crate::game::board::Location;
use crate::game::board::Player;
use crate::game::board::NUM_BOARD_COLORS;
use crate::game::board::P_BLACK;
use crate::game::board::P_WHITE;
use crate::neuralnet::nninputs::SymmetryHelpers;
use crate::testAssert;

/// upstream: LocalPatternHasher (localpattern.h)
pub struct LocalPatternHasher {
  pub x_size: i32,
  pub y_size: i32,
  pub zobrist_local_pattern: Vec<Hash128>,
  pub zobrist_pla: Vec<Hash128>,
  pub zobrist_atari: Vec<Hash128>,
}

impl LocalPatternHasher {
  /// upstream: LocalPatternHasher::LocalPatternHasher (localpattern.cpp)
  pub fn new() -> LocalPatternHasher {
    LocalPatternHasher {
      x_size: 0,
      y_size: 0,
      zobrist_local_pattern: Vec::new(),
      zobrist_pla: Vec::new(),
      zobrist_atari: Vec::new(),
    }
  }

  /// upstream: LocalPatternHasher::init (localpattern.cpp)
  pub fn init(&mut self, x: i32, y: i32, rand: &mut Rand) {
    self.x_size = x;
    self.y_size = y;
    testAssert!(self.x_size > 0 && self.x_size % 2 == 1);
    testAssert!(self.y_size > 0 && self.y_size % 2 == 1);
    // PORT NOTE: `vector::resize` on a fresh (or smaller) vector value-initializes the new
    // elements; every element is overwritten below, so resizing with the default is equivalent.
    self.zobrist_local_pattern.resize((NUM_BOARD_COLORS * self.x_size * self.y_size) as usize, Hash128::default());
    self.zobrist_pla.resize(NUM_BOARD_COLORS as usize, Hash128::default());
    self.zobrist_atari.resize((self.x_size * self.y_size) as usize, Hash128::default());

    for i in 0..NUM_BOARD_COLORS {
      for dy in 0..self.y_size {
        for dx in 0..self.x_size {
          let h0: u64 = rand.next_uint64();
          let h1: u64 = rand.next_uint64();
          self.zobrist_local_pattern[(i * self.y_size * self.x_size + dy * self.x_size + dx) as usize] = Hash128::new(h0, h1);
        }
      }
    }
    for i in 0..NUM_BOARD_COLORS {
      let h0: u64 = rand.next_uint64();
      let h1: u64 = rand.next_uint64();
      self.zobrist_pla[i as usize] = Hash128::new(h0, h1);
    }
    for dy in 0..self.y_size {
      for dx in 0..self.x_size {
        let h0: u64 = rand.next_uint64();
        let h1: u64 = rand.next_uint64();
        self.zobrist_atari[(dy * self.x_size + dx) as usize] = Hash128::new(h0, h1);
      }
    }
  }

  /// upstream: LocalPatternHasher::getHash (localpattern.cpp)
  pub fn get_hash(&self, board: &Board, loc: Loc, pla: Player) -> Hash128 {
    let mut hash: Hash128 = self.zobrist_pla[pla as usize];

    if loc != Board::PASS_LOC && loc != Board::NULL_LOC {
      let dxi: i32 = board.adj_offsets[2] as i32;
      let dyi: i32 = board.adj_offsets[3] as i32;
      debug_assert!(dxi == 1);
      debug_assert!(dyi == board.x_size + 1);

      let x_radius: i32 = self.x_size / 2;
      let y_radius: i32 = self.y_size / 2;
      let x_center: i32 = self.x_size / 2;
      let y_center: i32 = self.y_size / 2;

      let x: i32 = Location::get_x(loc, board.x_size);
      let y: i32 = Location::get_y(loc, board.x_size);
      let mut dx_min: i32 = -x_radius;
      let mut dx_max: i32 = x_radius;
      let mut dy_min: i32 = -y_radius;
      let mut dy_max: i32 = y_radius;
      if x < x_radius {
        dx_min = -x;
      } else if x >= board.x_size - x_radius {
        dx_max = board.x_size - 1 - x;
      }
      if y < y_radius {
        dy_min = -y;
      } else if y >= board.y_size - y_radius {
        dy_max = board.y_size - 1 - y;
      }
      for dy in dy_min..=dy_max {
        for dx in dx_min..=dx_max {
          let loc2: Loc = (loc as i32 + dx * dxi + dy * dyi) as Loc;
          let y2: i32 = dy + y_center;
          let x2: i32 = dx + x_center;
          let xy2: i32 = y2 * self.x_size + x2;
          hash ^= self.zobrist_local_pattern[(board.colors[loc2 as usize] as i32 * self.x_size * self.y_size + xy2) as usize];
          if (board.colors[loc2 as usize] == P_BLACK || board.colors[loc2 as usize] == P_WHITE) && board.get_num_liberties(loc2) == 1 {
            hash ^= self.zobrist_atari[xy2 as usize];
          }
        }
      }
    }

    hash
  }

  /// upstream: LocalPatternHasher::getHashWithSym (localpattern.cpp)
  //Returns the hash that would occur if symmetry were applied to both board and loc.
  //So basically, the only thing that changes is the zobrist indexing.
  pub fn get_hash_with_sym(&self, board: &Board, loc: Loc, pla: Player, symmetry: i32, flip_colors: bool) -> Hash128 {
    let sym_pla: Player = if flip_colors { get_opp(pla) } else { pla };
    let mut hash: Hash128 = self.zobrist_pla[sym_pla as usize];

    if loc != Board::PASS_LOC && loc != Board::NULL_LOC {
      let dxi: i32 = board.adj_offsets[2] as i32;
      let dyi: i32 = board.adj_offsets[3] as i32;
      debug_assert!(dxi == 1);
      debug_assert!(dyi == board.x_size + 1);

      let x_radius: i32 = self.x_size / 2;
      let y_radius: i32 = self.y_size / 2;
      let x_center: i32 = self.x_size / 2;
      let y_center: i32 = self.y_size / 2;

      let transpose: bool = SymmetryHelpers::is_transpose(symmetry);
      let flip_x: bool = SymmetryHelpers::is_flip_x(symmetry);
      let flip_y: bool = SymmetryHelpers::is_flip_y(symmetry);

      let x: i32 = Location::get_x(loc, board.x_size);
      let y: i32 = Location::get_y(loc, board.x_size);
      let mut dx_min: i32 = -x_radius;
      let mut dx_max: i32 = x_radius;
      let mut dy_min: i32 = -y_radius;
      let mut dy_max: i32 = y_radius;
      if x < x_radius {
        dx_min = -x;
      } else if x >= board.x_size - x_radius {
        dx_max = board.x_size - 1 - x;
      }
      if y < y_radius {
        dy_min = -y;
      } else if y >= board.y_size - y_radius {
        dy_max = board.y_size - 1 - y;
      }
      for dy in dy_min..=dy_max {
        for dx in dx_min..=dx_max {
          let loc2: Loc = (loc as i32 + dx * dxi + dy * dyi) as Loc;
          let y2: i32 = dy + y_center;
          let x2: i32 = dx + x_center;

          let sym_xy2: i32;
          let mut sym_x2: i32 = if flip_x { self.x_size - x2 - 1 } else { x2 };
          let mut sym_y2: i32 = if flip_y { self.y_size - y2 - 1 } else { y2 };
          if transpose {
            std::mem::swap(&mut sym_x2, &mut sym_y2);
            sym_xy2 = sym_y2 * self.y_size + sym_x2;
          } else {
            sym_xy2 = sym_y2 * self.x_size + sym_x2;
          }

          let sym_color: i32;
          if board.colors[loc2 as usize] == P_BLACK || board.colors[loc2 as usize] == P_WHITE {
            sym_color = (if flip_colors { get_opp(board.colors[loc2 as usize]) } else { board.colors[loc2 as usize] }) as i32;
          } else {
            sym_color = board.colors[loc2 as usize] as i32;
          }

          hash ^= self.zobrist_local_pattern[(sym_color * self.x_size * self.y_size + sym_xy2) as usize];
          if (board.colors[loc2 as usize] == P_BLACK || board.colors[loc2 as usize] == P_WHITE) && board.get_num_liberties(loc2) == 1 {
            hash ^= self.zobrist_atari[sym_xy2 as usize];
          }
        }
      }
    }

    hash
  }
}
