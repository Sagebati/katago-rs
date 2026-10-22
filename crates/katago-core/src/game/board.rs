// Ported from KataGo cpp/game/board.cpp, cpp/game/board.h @ ba938676d7f42d70950b3a535af2466fb642008c

/*
 * board.h
 * Originally from an unreleased project back in 2010, modified since.
 * Authors: brettharrison (original), David Wu (original and later modifications).
 */

use std::collections::BTreeSet;
use std::sync::OnceLock;

use crate::core::global::Global;
use crate::core::global::KataError;
use crate::core::hash::Hash;
use crate::core::hash::Hash128;
use crate::core::rand::Rand;
use crate::strprintf;
use crate::testAssert;

/// upstream: COMPILE_MAX_BOARD_LEN (board.h)
pub const COMPILE_MAX_BOARD_LEN: i32 = 19;

//TYPES AND CONSTANTS-----------------------------------------------------------------

/// upstream: Player (board.h)
//Player
pub type Player = i8;
/// upstream: P_BLACK, P_WHITE (board.h)
pub const P_BLACK: Player = 1;
pub const P_WHITE: Player = 2;

/// upstream: Color (board.h)
//Color of a point on the board
pub type Color = i8;
/// upstream: C_EMPTY, C_BLACK, C_WHITE, C_WALL, NUM_BOARD_COLORS (board.h)
pub const C_EMPTY: Color = 0;
pub const C_BLACK: Color = 1;
pub const C_WHITE: Color = 2;
pub const C_WALL: Color = 3;
pub const NUM_BOARD_COLORS: i32 = 4;

/// upstream: getOpp (board.h)
#[inline]
pub fn get_opp(c: Color) -> Color {
  c ^ 3
}

/// upstream: namespace PlayerIO (board.h)
//Conversions for players and colors
pub struct PlayerIO;

/// upstream: Loc (board.h)
//Location of a point on the board
//(x,y) is represented as (x+1) + (y+1)*(x_size+1)
pub type Loc = i16;

/// upstream: namespace Location (board.h)
pub struct Location;

impl Location {
  /// upstream: Location::getLoc (board.h)
  #[inline]
  pub fn get_loc(x: i32, y: i32, x_size: i32) -> Loc {
    // PORT NOTE: C++ signed overflow is UB, reachable from user input (Location::tryOfString
    // "(x,y)"); wrap like x86-64 instead of panicking. Only the low 16 bits survive the (Loc)
    // cast, so this equals the C++ result for any int inputs.
    (x.wrapping_add(1).wrapping_add(y.wrapping_add(1).wrapping_mul(x_size.wrapping_add(1)))) as Loc
  }
  /// upstream: Location::getX (board.h)
  #[inline]
  pub fn get_x(loc: Loc, x_size: i32) -> i32 {
    (loc as i32 % (x_size + 1)) - 1
  }
  /// upstream: Location::getY (board.h)
  #[inline]
  pub fn get_y(loc: Loc, x_size: i32) -> i32 {
    (loc as i32 / (x_size + 1)) - 1
  }
}

/// upstream: Move (board.h)
//Simple structure for storing moves. Not used below, but this is a convenient place to define it.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Move {
  pub loc: Loc,
  pub pla: Player,
}

impl Move {
  /// upstream: Move::Move (board.h)
  pub fn new(loc: Loc, pla: Player) -> Move {
    Move { loc, pla }
  }
}

/// upstream: Board::ChainData (board.h)
//Tracks a chain/string/group of stones
#[derive(Clone, Copy, Debug, Default)]
pub struct ChainData {
  pub owner: Player,      //Owner of chain
  pub num_locs: i16,      //Number of stones in chain
  pub num_liberties: i16, //Number of liberties in chain
}

/// upstream: Board::MoveRecord (board.h)
//Move data passed back when moves are made to allow for undos
#[derive(Clone, Copy, Debug, Default)]
pub struct MoveRecord {
  pub pla: Player,
  pub loc: Loc,
  pub ko_loc: Loc,
  pub cap_dirs: u8, //First 4 bits indicate directions of capture, fifth bit indicates suicide
}

/// upstream: Board (board.h)
//Fast lightweight board designed for playouts and simulations, where speed is essential.
//Simple ko rule only.
//Does not enforce player turn order.
pub struct Board {
  //Data--------------------------------------------
  pub x_size: i32,                                            //Horizontal size of board
  pub y_size: i32,                                            //Vertical size of board
  pub colors: [Color; Board::MAX_ARR_SIZE as usize],          //Color of each location on the board.

  //Every chain of stones has one of its stones arbitrarily designated as the head.
  pub chain_data: [ChainData; Board::MAX_ARR_SIZE as usize], //For each head stone, the chaindata for the chain under that head. Undefined otherwise.
  pub chain_head: [Loc; Board::MAX_ARR_SIZE as usize],       //Where is the head of this chain? Undefined if EMPTY or WALL
  pub next_in_chain: [Loc; Board::MAX_ARR_SIZE as usize],    //Location of next stone in chain. Circular linked list. Undefined if EMPTY or WALL

  pub ko_loc: Loc, //A simple ko capture was made here, making it illegal to replay here next move

  /* PointList empty_list; //List of all empty locations on board */
  pub pos_hash: Hash128, //A zobrist hash of the current board position (does not include ko point or player to move)

  pub num_black_captures: i32, //Number of b stones captured, informational and used by board history when clearing pos
  pub num_white_captures: i32, //Number of w stones captured, informational and used by board history when clearing pos

  pub adj_offsets: [i16; 8], //Indices 0-3: Offsets to add for adjacent points. Indices 4-7: Offsets for diagonal points. 2 and 3 are +x and +y.
}

/// upstream: Board::IS_ZOBRIST_INITALIZED, Board::ZOBRIST_* (board.h)
/// PORT NOTE: the runtime-initialized statics live in one struct behind a OnceLock
/// (PORTING.md §8, OWNERSHIP.tsv). IS_ZOBRIST_INITALIZED is `BOARD_STATICS.get().is_some()`.
pub struct BoardStatics {
  pub ZOBRIST_SIZE_X_HASH: [Hash128; (Board::MAX_LEN + 1) as usize],
  pub ZOBRIST_SIZE_Y_HASH: [Hash128; (Board::MAX_LEN + 1) as usize],
  pub ZOBRIST_BOARD_HASH: [[Hash128; 4]; Board::MAX_ARR_SIZE as usize],
  pub ZOBRIST_BOARD_HASH2: [[Hash128; 4]; Board::MAX_ARR_SIZE as usize],
  pub ZOBRIST_PLAYER_HASH: [Hash128; 4],
  pub ZOBRIST_KO_LOC_HASH: [Hash128; Board::MAX_ARR_SIZE as usize],
  pub ZOBRIST_KO_MARK_HASH: [[Hash128; 4]; Board::MAX_ARR_SIZE as usize],
  pub ZOBRIST_ENCORE_HASH: [Hash128; 3],
  pub ZOBRIST_SECOND_ENCORE_START_HASH: [[Hash128; 4]; Board::MAX_ARR_SIZE as usize],
}

static BOARD_STATICS: OnceLock<BoardStatics> = OnceLock::new();

//STATIC VARS-----------------------------------------------------------------------------
impl Board {
  //Board parameters and Constants----------------------------------------

  /// upstream: Board::MAX_LEN, Board::DEFAULT_LEN, Board::MAX_PLAY_SIZE, Board::MAX_ARR_SIZE (board.h)
  pub const MAX_LEN: i32 = COMPILE_MAX_BOARD_LEN; //Maximum edge length allowed for the board
  pub const DEFAULT_LEN: i32 = if Board::MAX_LEN < 19 { Board::MAX_LEN } else { 19 }; //Default edge length for board if unspecified
  pub const MAX_PLAY_SIZE: i32 = Board::MAX_LEN * Board::MAX_LEN; //Maximum number of playable spaces
  pub const MAX_ARR_SIZE: i32 = (Board::MAX_LEN + 1) * (Board::MAX_LEN + 2) + 1; //Maximum size of arrays needed

  /// upstream: Board::NULL_LOC, Board::PASS_LOC (board.h)
  //Location used to indicate an invalid spot on the board.
  pub const NULL_LOC: Loc = 0;
  //Location used to indicate a pass move is desired.
  pub const PASS_LOC: Loc = 1;

  /// upstream: Board::ZOBRIST_PASS_ENDS_PHASE (board.cpp)
  pub const ZOBRIST_PASS_ENDS_PHASE: Hash128 = //Based on sha256 hash of Board::ZOBRIST_PASS_ENDS_PHASE
    Hash128::new(0x853E097C279EBF4E, 0xE3153DEF9E14A62C);
  /// upstream: Board::ZOBRIST_GAME_IS_OVER (board.cpp)
  pub const ZOBRIST_GAME_IS_OVER: Hash128 = //Based on sha256 hash of Board::ZOBRIST_GAME_IS_OVER
    Hash128::new(0xb6f9e465597a77ee, 0xf1d583d960a4ce7f);

  /// The zobrist tables (`Board::ZOBRIST_*` statics upstream).
  /// TODO(port): PORTING.md §8 does not say whether reading the statics before
  /// Board::initHash() should panic or initialize; this initializes (same values either way).
  #[inline]
  pub fn statics() -> &'static BoardStatics {
    BOARD_STATICS.get_or_init(compute_init_hash)
  }
}

//LOCATION--------------------------------------------------------------------------------
impl Location {
  /// upstream: Location::getAdjacentOffsets (board.cpp)
  pub fn get_adjacent_offsets(adj_offsets: &mut [i16; 8], x_size: i32) {
    adj_offsets[0] = (-(x_size + 1)) as i16;
    adj_offsets[1] = -1;
    adj_offsets[2] = 1;
    adj_offsets[3] = (x_size + 1) as i16;
    adj_offsets[4] = (-(x_size + 1) - 1) as i16;
    adj_offsets[5] = (-(x_size + 1) + 1) as i16;
    adj_offsets[6] = ((x_size + 1) - 1) as i16;
    adj_offsets[7] = ((x_size + 1) + 1) as i16;
  }

  /// upstream: Location::isAdjacent (board.cpp)
  pub fn is_adjacent(loc0: Loc, loc1: Loc, x_size: i32) -> bool {
    let (l0, l1) = (loc0 as i32, loc1 as i32);
    l0 == l1 - (x_size + 1) || l0 == l1 - 1 || l0 == l1 + 1 || l0 == l1 + (x_size + 1)
  }

  /// upstream: Location::getMirrorLoc (board.cpp)
  pub fn get_mirror_loc(loc: Loc, x_size: i32, y_size: i32) -> Loc {
    if loc == Board::NULL_LOC || loc == Board::PASS_LOC {
      return loc;
    }
    Location::get_loc(x_size - 1 - Location::get_x(loc, x_size), y_size - 1 - Location::get_y(loc, x_size), x_size)
  }

  /// upstream: Location::getCenterLoc (board.cpp)
  /// `getCenterLoc(int x_size, int y_size)` (overload rule 3: not the fewest-params overload).
  pub fn get_center_loc_x_size_y_size(x_size: i32, y_size: i32) -> Loc {
    if x_size % 2 == 0 || y_size % 2 == 0 {
      return Board::NULL_LOC;
    }
    Location::get_loc(x_size / 2, y_size / 2, x_size)
  }

  /// upstream: Location::getCenterLoc (board.cpp)
  pub fn get_center_loc(b: &Board) -> Loc {
    Location::get_center_loc_x_size_y_size(b.x_size, b.y_size)
  }

  /// upstream: Location::isCentral (board.cpp)
  pub fn is_central(loc: Loc, x_size: i32, y_size: i32) -> bool {
    let x = Location::get_x(loc, x_size);
    let y = Location::get_y(loc, x_size);
    x >= (x_size - 1) / 2 && x <= x_size / 2 && y >= (y_size - 1) / 2 && y <= y_size / 2
  }

  /// upstream: Location::isNearCentral (board.cpp)
  pub fn is_near_central(loc: Loc, x_size: i32, y_size: i32) -> bool {
    let x = Location::get_x(loc, x_size);
    let y = Location::get_y(loc, x_size);
    x >= (x_size - 1) / 2 - 1 && x <= x_size / 2 + 1 && y >= (y_size - 1) / 2 - 1 && y <= y_size / 2 + 1
  }
}

// PORT NOTE: FOREACHADJ / ADJ0..ADJ3 macros are expanded at each use (PORTING.md §5):
// FOREACHADJ(BLOCK) -> `for adjoffset in [-(self.x_size+1), -1, 1, self.x_size+1] { BLOCK }`.

//CONSTRUCTORS AND INITIALIZATION----------------------------------------------------------

impl Board {
  /// Storage before `init` runs (C++ leaves it uninitialized; Rust zero-initializes).
  fn new_uninit() -> Board {
    Board {
      x_size: 0,
      y_size: 0,
      colors: [0; Board::MAX_ARR_SIZE as usize],
      chain_data: [ChainData::default(); Board::MAX_ARR_SIZE as usize],
      chain_head: [0; Board::MAX_ARR_SIZE as usize],
      next_in_chain: [0; Board::MAX_ARR_SIZE as usize],
      ko_loc: 0,
      pos_hash: Hash128::default(),
      num_black_captures: 0,
      num_white_captures: 0,
      adj_offsets: [0; 8],
    }
  }

  /// upstream: Board::Board (board.cpp)
  //Create Board of size (DEFAULT_LEN,DEFAULT_LEN)
  pub fn new() -> Board {
    let mut b = Board::new_uninit();
    // PORT NOTE: init cannot throw for DEFAULT_LEN (PORTING.md §7).
    b.init(Board::DEFAULT_LEN, Board::DEFAULT_LEN).expect("upstream: cannot throw for constant input");
    b
  }

  /// upstream: Board::Board (board.cpp)
  //Create Board of size (x,y)
  pub fn new_x_y(x: i32, y: i32) -> Result<Board, KataError> {
    let mut b = Board::new_uninit();
    b.init(x, y)?;
    Ok(b)
  }

  /// upstream: Board::init (board.cpp)
  pub(crate) fn init(&mut self, x_s: i32, y_s: i32) -> Result<(), KataError> {
    // PORT NOTE: IS_ZOBRIST_INITALIZED -> BOARD_STATICS.get().is_some() (OWNERSHIP.tsv).
    testAssert!(BOARD_STATICS.get().is_some());
    if x_s < 0 || y_s < 0 || x_s > Board::MAX_LEN || y_s > Board::MAX_LEN {
      return Err(KataError::StringError(String::from("Board::init - invalid board size")));
    }

    self.x_size = x_s;
    self.y_size = y_s;

    for i in 0..Board::MAX_ARR_SIZE {
      self.colors[i as usize] = C_WALL;
    }

    for y in 0..self.y_size {
      for x in 0..self.x_size {
        let loc: Loc = ((x + 1) + (y + 1) * (self.x_size + 1)) as Loc;
        self.colors[loc as usize] = C_EMPTY;
        // empty_list.add(loc);
      }
    }

    self.ko_loc = Board::NULL_LOC;
    self.pos_hash = Board::statics().ZOBRIST_SIZE_X_HASH[self.x_size as usize] ^ Board::statics().ZOBRIST_SIZE_Y_HASH[self.y_size as usize];
    self.num_black_captures = 0;
    self.num_white_captures = 0;

    Location::get_adjacent_offsets(&mut self.adj_offsets, self.x_size);
    Ok(())
  }

  /// upstream: Board::initHash (board.cpp)
  //Initialize the zobrist hash.
  //MUST BE CALLED AT PROGRAM START!
  /// PORT NOTE: forces the OnceLock; the body is compute_init_hash (PORTING.md §8).
  pub fn init_hash() {
    BOARD_STATICS.get_or_init(compute_init_hash);
  }
}

/// upstream: Board::Board(const Board&) (board.cpp)
impl Clone for Board {
  fn clone(&self) -> Board {
    Board {
      x_size: self.x_size,
      y_size: self.y_size,

      colors: self.colors,
      chain_data: self.chain_data,
      chain_head: self.chain_head,
      next_in_chain: self.next_in_chain,

      ko_loc: self.ko_loc,
      // empty_list = other.empty_list;
      pos_hash: self.pos_hash,
      num_black_captures: self.num_black_captures,
      num_white_captures: self.num_white_captures,

      adj_offsets: self.adj_offsets,
    }
  }
}

/// upstream: Board::initHash (board.cpp)
/// The body of Board::initHash, with the RNG call order unchanged (PORTING.md §8).
fn compute_init_hash() -> BoardStatics {
  let mut st = BoardStatics {
    ZOBRIST_SIZE_X_HASH: [Hash128::default(); (Board::MAX_LEN + 1) as usize],
    ZOBRIST_SIZE_Y_HASH: [Hash128::default(); (Board::MAX_LEN + 1) as usize],
    ZOBRIST_BOARD_HASH: [[Hash128::default(); 4]; Board::MAX_ARR_SIZE as usize],
    ZOBRIST_BOARD_HASH2: [[Hash128::default(); 4]; Board::MAX_ARR_SIZE as usize],
    ZOBRIST_PLAYER_HASH: [Hash128::default(); 4],
    ZOBRIST_KO_LOC_HASH: [Hash128::default(); Board::MAX_ARR_SIZE as usize],
    ZOBRIST_KO_MARK_HASH: [[Hash128::default(); 4]; Board::MAX_ARR_SIZE as usize],
    ZOBRIST_ENCORE_HASH: [Hash128::default(); 3],
    ZOBRIST_SECOND_ENCORE_START_HASH: [[Hash128::default(); 4]; Board::MAX_ARR_SIZE as usize],
  };

  let mut rand = Rand::new_seed("Board::initHash()");

  // PORT NOTE: the `[&rand]` lambda takes rand as a parameter, because the enclosing fn
  // re-seeds rand between uses.
  let next_hash = |rand: &mut Rand| -> Hash128 {
    let h0: u64 = rand.next_uint64();
    let h1: u64 = rand.next_uint64();
    Hash128::new(h0, h1)
  };

  for i in 0..4 {
    st.ZOBRIST_PLAYER_HASH[i] = next_hash(&mut rand);
  }
  for i in 0..3 {
    st.ZOBRIST_ENCORE_HASH[i] = next_hash(&mut rand);
  }

  //Do this second so that the player and encore hashes are not
  //afffected by the size of the board we compile with.
  for i in 0..Board::MAX_ARR_SIZE as usize {
    for j in 0..4 as Color {
      if j == C_EMPTY || j == C_WALL {
        st.ZOBRIST_BOARD_HASH[i][j as usize] = Hash128::default();
      } else {
        st.ZOBRIST_BOARD_HASH[i][j as usize] = next_hash(&mut rand);
      }

      if j == C_EMPTY || j == C_WALL {
        st.ZOBRIST_KO_MARK_HASH[i][j as usize] = Hash128::default();
      } else {
        st.ZOBRIST_KO_MARK_HASH[i][j as usize] = next_hash(&mut rand);
      }
    }
    st.ZOBRIST_KO_LOC_HASH[i] = next_hash(&mut rand);
  }

  //Reseed the random number generator so that these hashes are also
  //not affected by the size of the board we compile with
  rand.init_seed("Board::initHash() for ZOBRIST_SECOND_ENCORE_START hashes");
  for i in 0..Board::MAX_ARR_SIZE as usize {
    for j in 0..4 as Color {
      if j == C_EMPTY || j == C_WALL {
        st.ZOBRIST_SECOND_ENCORE_START_HASH[i][j as usize] = Hash128::default();
      } else {
        st.ZOBRIST_SECOND_ENCORE_START_HASH[i][j as usize] = next_hash(&mut rand);
      }
    }
  }

  //Reseed the random number generator so that these size hashes are also
  //not affected by the size of the board we compile with
  rand.init_seed("Board::initHash() for ZOBRIST_SIZE hashes");
  for i in 0..(Board::MAX_LEN + 1) as usize {
    st.ZOBRIST_SIZE_X_HASH[i] = next_hash(&mut rand);
    st.ZOBRIST_SIZE_Y_HASH[i] = next_hash(&mut rand);
  }

  //Reseed and compute one more set of zobrist hashes, mixed a bit differently
  rand.init_seed("Board::initHash() for second set of ZOBRIST hashes");
  for i in 0..Board::MAX_ARR_SIZE as usize {
    for j in 0..4usize {
      st.ZOBRIST_BOARD_HASH2[i][j] = next_hash(&mut rand);
      st.ZOBRIST_BOARD_HASH2[i][j].hash0 = Hash::murmur_mix(st.ZOBRIST_BOARD_HASH2[i][j].hash0);
      st.ZOBRIST_BOARD_HASH2[i][j].hash1 = Hash::split_mix64(st.ZOBRIST_BOARD_HASH2[i][j].hash1);
    }
  }

  // PORT NOTE: IS_ZOBRIST_INITALIZED = true is the OnceLock being set by the caller.
  st
}

impl Board {
  /// upstream: Board::getSitHashWithSimpleKo (board.cpp)
  //Get a hash that combines the position of the board with simple ko prohibition and a player to move.
  pub fn get_sit_hash_with_simple_ko(&self, pla: Player) -> Hash128 {
    let mut h: Hash128 = self.pos_hash;
    if self.ko_loc != Board::NULL_LOC {
      h = h ^ Board::statics().ZOBRIST_KO_LOC_HASH[self.ko_loc as usize];
    }
    h ^= Board::statics().ZOBRIST_PLAYER_HASH[pla as usize];
    h
  }

  /// upstream: Board::clearSimpleKoLoc (board.cpp)
  //Lift any simple ko ban recorded on thie board due to an immediate prior ko capture.
  pub fn clear_simple_ko_loc(&mut self) {
    self.ko_loc = Board::NULL_LOC;
  }
  /// upstream: Board::setSimpleKoLoc (board.cpp)
  //Directly set that there is a simple ko prohibition on this location. Note that this is not necessarily safe
  //when also using a BoardHistory, since the BoardHistory may not know about this change, or the game could be in cleanup phase, etc.
  pub fn set_simple_ko_loc(&mut self, loc: Loc) {
    self.ko_loc = loc;
  }

  /// upstream: Board::sqrtBoardArea (board.cpp)
  pub fn sqrt_board_area(&self) -> f64 {
    if self.x_size == self.y_size {
      self.x_size as f64
    } else {
      ((self.x_size * self.y_size) as f64).sqrt()
    }
  }

  /// upstream: Board::getChainSize (board.cpp)
  //Gets the number of stones of the chain at loc. Precondition: location must be black or white.
  pub fn get_chain_size(&self, loc: Loc) -> i32 {
    self.chain_data[self.chain_head[loc as usize] as usize].num_locs as i32
  }

  /// upstream: Board::getNumLiberties (board.cpp)
  //Gets the number of liberties of the chain at loc. Assertion: location must be black or white.
  pub fn get_num_liberties(&self, loc: Loc) -> i32 {
    self.chain_data[self.chain_head[loc as usize] as usize].num_liberties as i32
  }

  /// upstream: Board::isSuicide (board.cpp)
  //Check if moving here would be a self-capture
  pub fn is_suicide(&self, loc: Loc, pla: Player) -> bool {
    if loc == Board::PASS_LOC {
      return false;
    }

    let opp: Player = get_opp(pla);
    for adjoffset in [-(self.x_size + 1), -1, 1, self.x_size + 1] {
      let adj: Loc = (loc as i32 + adjoffset) as Loc;

      if self.colors[adj as usize] == C_EMPTY {
        return false;
      } else if self.colors[adj as usize] == pla {
        if self.get_num_liberties(adj) > 1 {
          return false;
        }
      } else if self.colors[adj as usize] == opp {
        if self.get_num_liberties(adj) == 1 {
          return false;
        }
      }
    }

    true
  }

  /// upstream: Board::isIllegalSuicide (board.cpp)
  //Check if moving here is would be an illegal self-capture
  pub fn is_illegal_suicide(&self, loc: Loc, pla: Player, is_multi_stone_suicide_legal: bool) -> bool {
    let opp: Player = get_opp(pla);
    for adjoffset in [-(self.x_size + 1), -1, 1, self.x_size + 1] {
      let adj: Loc = (loc as i32 + adjoffset) as Loc;

      if self.colors[adj as usize] == C_EMPTY {
        return false;
      } else if self.colors[adj as usize] == pla {
        if is_multi_stone_suicide_legal || self.get_num_liberties(adj) > 1 {
          return false;
        }
      } else if self.colors[adj as usize] == opp {
        if self.get_num_liberties(adj) == 1 {
          return false;
        }
      }
    }

    true
  }

  /// upstream: Board::getBoundNumLibertiesAfterPlay (board.cpp)
  //Returns a fast lower bound on the number of liberties a new stone placed here would have
  pub fn get_bound_num_liberties_after_play(&self, loc: Loc, pla: Player, lower_bound: &mut i32, upper_bound: &mut i32) {
    let opp: Player = get_opp(pla);

    let mut num_immediate_libs: i32 = 0; //empty spaces adjacent
    let mut num_caps: i32 = 0; //number of adjacent directions in which we will capture
    let mut potential_libs_from_caps: i32 = 0; //Total number of stones we're capturing (possibly with multiplicity)
    let mut num_connection_libs: i32 = 0; //Sum over friendly groups connected to of their libs-1
    let mut max_connection_libs: i32 = 0; //Max over friendly groups connected to of their libs-1

    for i in 0..4 {
      let adj: Loc = (loc as i32 + self.adj_offsets[i] as i32) as Loc;
      if self.colors[adj as usize] == C_EMPTY {
        num_immediate_libs += 1;
      } else if self.colors[adj as usize] == opp {
        let libs: i32 = self.chain_data[self.chain_head[adj as usize] as usize].num_liberties as i32;
        if libs == 1 {
          num_caps += 1;
          potential_libs_from_caps += self.chain_data[self.chain_head[adj as usize] as usize].num_locs as i32;
        }
      } else if self.colors[adj as usize] == pla {
        let libs: i32 = self.chain_data[self.chain_head[adj as usize] as usize].num_liberties as i32;
        let conn_libs: i32 = libs - 1;
        num_connection_libs += conn_libs;
        if conn_libs > max_connection_libs {
          max_connection_libs = conn_libs;
        }
      }
    }

    *lower_bound = num_caps + (if max_connection_libs > num_immediate_libs { max_connection_libs } else { num_immediate_libs });
    *upper_bound = num_immediate_libs + potential_libs_from_caps + num_connection_libs;
  }

  /// upstream: Board::getNumLibertiesAfterPlay (board.cpp)
  //Returns the number of liberties a new stone placed here would have, or max if it would be >= max.
  pub fn get_num_liberties_after_play(&self, loc: Loc, pla: Player, max: i32) -> i32 {
    let opp: Player = get_opp(pla);

    let mut num_libs: i32 = 0;
    let mut libs: [Loc; Board::MAX_PLAY_SIZE as usize] = [0; Board::MAX_PLAY_SIZE as usize];
    let mut num_captured_groups: i32 = 0;
    let mut captured_group_heads: [Loc; 4] = [0; 4];

    //First, count immediate liberties and groups that would be captured
    for i in 0..4 {
      let adj: Loc = (loc as i32 + self.adj_offsets[i] as i32) as Loc;
      if self.colors[adj as usize] == C_EMPTY {
        libs[num_libs as usize] = adj;
        num_libs += 1;
        if num_libs >= max {
          return max;
        }
      } else if self.colors[adj as usize] == opp && self.get_num_liberties(adj) == 1 {
        libs[num_libs as usize] = adj;
        num_libs += 1;
        if num_libs >= max {
          return max;
        }

        let head: Loc = self.chain_head[adj as usize];
        let mut already_found = false;
        for j in 0..num_captured_groups {
          if captured_group_heads[j as usize] == head {
            already_found = true;
            break;
          }
        }
        if !already_found {
          captured_group_heads[num_captured_groups as usize] = head;
          num_captured_groups += 1;
        }
      }
    }

    let would_be_empty = |lc: Loc| -> bool {
      if self.colors[lc as usize] == C_EMPTY {
        return true;
      }
      if self.colors[lc as usize] == opp {
        for i in 0..num_captured_groups {
          if captured_group_heads[i as usize] == self.chain_head[lc as usize] {
            return true;
          }
        }
      }
      false
    };

    //Next, walk through all stones of all surrounding groups we would connect with and count liberties, avoiding overlap.
    let mut num_connecting_groups: i32 = 0;
    let mut connecting_group_heads: [Loc; 4] = [0; 4];
    for i in 0..4 {
      let adj: Loc = (loc as i32 + self.adj_offsets[i] as i32) as Loc;
      if self.colors[adj as usize] == pla {
        let head: Loc = self.chain_head[adj as usize];
        let mut already_found = false;
        for j in 0..num_connecting_groups {
          if connecting_group_heads[j as usize] == head {
            already_found = true;
            break;
          }
        }
        if !already_found {
          connecting_group_heads[num_connecting_groups as usize] = head;
          num_connecting_groups += 1;

          let mut cur: Loc = adj;
          loop {
            for k in 0..4 {
              let possible_lib: Loc = (cur as i32 + self.adj_offsets[k] as i32) as Loc;
              if possible_lib != loc && would_be_empty(possible_lib) {
                let mut already_counted = false;
                for l in 0..num_libs {
                  if libs[l as usize] == possible_lib {
                    already_counted = true;
                    break;
                  }
                }
                if !already_counted {
                  libs[num_libs as usize] = possible_lib;
                  num_libs += 1;
                  if num_libs >= max {
                    return max;
                  }
                }
              }
            }

            cur = self.next_in_chain[cur as usize];
            if !(cur != adj) {
              break;
            }
          }
        }
      }
    }
    num_libs
  }

  /// upstream: Board::isKoBanned (board.cpp)
  //Check if moving here is illegal due to simple ko
  pub fn is_ko_banned(&self, loc: Loc) -> bool {
    loc == self.ko_loc
  }

  /// upstream: Board::isOnBoard (board.cpp)
  //Check if this location is on the board
  pub fn is_on_board(&self, loc: Loc) -> bool {
    loc >= 0 && (loc as i32) < Board::MAX_ARR_SIZE && self.colors[loc as usize] != C_WALL
  }

  /// upstream: Board::isLegal (board.cpp)
  //Check if moving here is illegal.
  pub fn is_legal(&self, loc: Loc, pla: Player, is_multi_stone_suicide_legal: bool) -> bool {
    if pla != P_BLACK && pla != P_WHITE {
      return false;
    }
    loc == Board::PASS_LOC
      || (loc >= 0
        && (loc as i32) < Board::MAX_ARR_SIZE
        && (self.colors[loc as usize] == C_EMPTY)
        && !self.is_ko_banned(loc)
        && !self.is_illegal_suicide(loc, pla, is_multi_stone_suicide_legal))
  }

  /// upstream: Board::isLegalIgnoringKo (board.cpp)
  //Check if moving here is illegal, ignoring simple ko
  pub fn is_legal_ignoring_ko(&self, loc: Loc, pla: Player, is_multi_stone_suicide_legal: bool) -> bool {
    if pla != P_BLACK && pla != P_WHITE {
      return false;
    }
    loc == Board::PASS_LOC
      || (loc >= 0
        && (loc as i32) < Board::MAX_ARR_SIZE
        && (self.colors[loc as usize] == C_EMPTY)
        && !self.is_illegal_suicide(loc, pla, is_multi_stone_suicide_legal))
  }

  /// upstream: Board::isSimpleEye (board.cpp)
  //Check if this location contains a simple eye for the specified player.
  pub fn is_simple_eye(&self, loc: Loc, pla: Player) -> bool {
    if self.colors[loc as usize] != C_EMPTY {
      return false;
    }

    let mut against_wall = false;

    //Check that surounding points are owned
    for i in 0..4 {
      let adj: Loc = (loc as i32 + self.adj_offsets[i] as i32) as Loc;
      if self.colors[adj as usize] == C_WALL {
        against_wall = true;
      } else if self.colors[adj as usize] != pla {
        return false;
      }
    }

    //Check that opponent does not own too many diagonal points
    let opp: Player = get_opp(pla);
    let mut num_opp_corners: i32 = 0;
    for i in 4..8 {
      let corner: Loc = (loc as i32 + self.adj_offsets[i] as i32) as Loc;
      if self.colors[corner as usize] == opp {
        num_opp_corners += 1;
      }
    }

    if num_opp_corners >= 2 || (against_wall && num_opp_corners >= 1) {
      return false;
    }

    true
  }

  /// upstream: Board::wouldBeCapture (board.cpp)
  //Check if a move at this location would be a capture of an opponent group.
  pub fn would_be_capture(&self, loc: Loc, pla: Player) -> bool {
    if self.colors[loc as usize] != C_EMPTY {
      return false;
    }
    let opp: Player = get_opp(pla);
    for adjoffset in [-(self.x_size + 1), -1, 1, self.x_size + 1] {
      let adj: Loc = (loc as i32 + adjoffset) as Loc;
      if self.colors[adj as usize] == opp {
        if self.get_num_liberties(adj) == 1 {
          return true;
        }
      }
    }

    false
  }

  /// upstream: Board::wouldBeKoCapture (board.cpp)
  //Check if a move at this location would be a capture in a simple ko mouth.
  pub fn would_be_ko_capture(&self, loc: Loc, pla: Player) -> bool {
    if self.colors[loc as usize] != C_EMPTY {
      return false;
    }
    //Check that surounding points are are all opponent owned and exactly one of them is capturable
    let opp: Player = get_opp(pla);
    let mut opp_capturable_loc: Loc = Board::NULL_LOC;
    for i in 0..4 {
      let adj: Loc = (loc as i32 + self.adj_offsets[i] as i32) as Loc;
      if self.colors[adj as usize] != C_WALL && self.colors[adj as usize] != opp {
        return false;
      }
      if self.colors[adj as usize] == opp && self.get_num_liberties(adj) == 1 {
        if opp_capturable_loc != Board::NULL_LOC {
          return false;
        }
        opp_capturable_loc = adj;
      }
    }
    if opp_capturable_loc == Board::NULL_LOC {
      return false;
    }

    //Check that the capturable loc has exactly one stone
    if self.chain_data[self.chain_head[opp_capturable_loc as usize] as usize].num_locs != 1 {
      return false;
    }
    true
  }

  /// upstream: Board::getKoCaptureLoc (board.cpp)
  pub fn get_ko_capture_loc(&self, loc: Loc, pla: Player) -> Loc {
    if self.colors[loc as usize] != C_EMPTY {
      return Board::NULL_LOC;
    }
    //Check that surounding points are are all opponent owned and exactly one of them is capturable
    let opp: Player = get_opp(pla);
    let mut opp_capturable_loc: Loc = Board::NULL_LOC;
    for i in 0..4 {
      let adj: Loc = (loc as i32 + self.adj_offsets[i] as i32) as Loc;
      if self.colors[adj as usize] != C_WALL && self.colors[adj as usize] != opp {
        return Board::NULL_LOC;
      }
      if self.colors[adj as usize] == opp && self.get_num_liberties(adj) == 1 {
        if opp_capturable_loc != Board::NULL_LOC {
          return Board::NULL_LOC;
        }
        opp_capturable_loc = adj;
      }
    }
    if opp_capturable_loc == Board::NULL_LOC {
      return Board::NULL_LOC;
    }

    //Check that the capturable loc has exactly one stone
    if self.chain_data[self.chain_head[opp_capturable_loc as usize] as usize].num_locs != 1 {
      return Board::NULL_LOC;
    }
    opp_capturable_loc
  }

  /// upstream: Board::isAdjacentToPla (board.cpp)
  //Check if this location is adjacent to stones of the specified color
  pub fn is_adjacent_to_pla(&self, loc: Loc, pla: Player) -> bool {
    for adjoffset in [-(self.x_size + 1), -1, 1, self.x_size + 1] {
      let adj: Loc = (loc as i32 + adjoffset) as Loc;
      if self.colors[adj as usize] == pla {
        return true;
      }
    }
    false
  }

  /// upstream: Board::isAdjacentOrDiagonalToPla (board.cpp)
  pub fn is_adjacent_or_diagonal_to_pla(&self, loc: Loc, pla: Player) -> bool {
    for i in 0..8 {
      let adj: Loc = (loc as i32 + self.adj_offsets[i] as i32) as Loc;
      if self.colors[adj as usize] == pla {
        return true;
      }
    }
    false
  }

  /// upstream: Board::isAdjacentToChain (board.cpp)
  //Check if this location is adjacent a given chain.
  pub fn is_adjacent_to_chain(&self, loc: Loc, chain: Loc) -> bool {
    if self.colors[chain as usize] == C_EMPTY {
      return false;
    }
    for adjoffset in [-(self.x_size + 1), -1, 1, self.x_size + 1] {
      let adj: Loc = (loc as i32 + adjoffset) as Loc;
      if self.colors[adj as usize] == self.colors[chain as usize] && self.chain_head[adj as usize] == self.chain_head[chain as usize] {
        return true;
      }
    }
    false
  }

  /// upstream: Board::isNonPassAliveSelfConnection (board.cpp)
  //Does this connect two pla distinct groups that are not both pass-alive and not within opponent pass-alive area either?
  pub fn is_non_pass_alive_self_connection(&self, loc: Loc, pla: Player, pass_alive_area: &[Color]) -> bool {
    if self.colors[loc as usize] != C_EMPTY || pass_alive_area[loc as usize] == pla {
      return false;
    }

    let mut non_pass_alive_adj_head: Loc = Board::NULL_LOC;
    for i in 0..4 {
      let adj: Loc = (loc as i32 + self.adj_offsets[i] as i32) as Loc;
      if self.colors[adj as usize] == pla && pass_alive_area[adj as usize] == C_EMPTY {
        non_pass_alive_adj_head = self.chain_head[adj as usize];
        break;
      }
    }

    if non_pass_alive_adj_head == Board::NULL_LOC {
      return false;
    }

    for i in 0..4 {
      let adj: Loc = (loc as i32 + self.adj_offsets[i] as i32) as Loc;
      if self.colors[adj as usize] == pla && self.chain_head[adj as usize] != non_pass_alive_adj_head {
        return true;
      }
    }

    false
  }

  /// upstream: Board::isEmpty (board.cpp)
  //Is this board empty?
  pub fn is_empty(&self) -> bool {
    for y in 0..self.y_size {
      for x in 0..self.x_size {
        let loc: Loc = Location::get_loc(x, y, self.x_size);
        if self.colors[loc as usize] != C_EMPTY {
          return false;
        }
      }
    }
    true
  }

  /// upstream: Board::numStonesOnBoard (board.cpp)
  //Count the number of stones on the board
  pub fn num_stones_on_board(&self) -> i32 {
    let mut num: i32 = 0;
    for y in 0..self.y_size {
      for x in 0..self.x_size {
        let loc: Loc = Location::get_loc(x, y, self.x_size);
        if self.colors[loc as usize] == C_BLACK || self.colors[loc as usize] == C_WHITE {
          num += 1;
        }
      }
    }
    num
  }

  /// upstream: Board::numPlaStonesOnBoard (board.cpp)
  pub fn num_pla_stones_on_board(&self, pla: Player) -> i32 {
    let mut num: i32 = 0;
    for y in 0..self.y_size {
      for x in 0..self.x_size {
        let loc: Loc = Location::get_loc(x, y, self.x_size);
        if self.colors[loc as usize] == pla {
          num += 1;
        }
      }
    }
    num
  }

  /// upstream: Board::setStone (board.cpp)
  //Sets the specified stone if possible, including overwriting existing stones.
  //Resolves any captures and/or suicides that result from setting that stone, including deletions of the stone itself.
  //Returns false if location or color were out of range.
  pub fn set_stone(&mut self, loc: Loc, color: Color) -> bool {
    if loc < 0 || (loc as i32) >= Board::MAX_ARR_SIZE || self.colors[loc as usize] == C_WALL {
      return false;
    }
    if color != C_BLACK && color != C_WHITE && color != C_EMPTY {
      return false;
    }

    if self.colors[loc as usize] == color {
    } else if self.colors[loc as usize] == C_EMPTY {
      if !self.is_illegal_suicide(loc, color, true) {
        self.play_move_assume_legal(loc, color);
      }
    } else if color == C_EMPTY {
      self.remove_single_stone(loc);
    } else {
      self.remove_single_stone(loc);
      if !self.is_suicide(loc, color) {
        self.play_move_assume_legal(loc, color);
      }
    }

    self.ko_loc = Board::NULL_LOC;
    true
  }

  /// upstream: Board::setStoneFailIfNoLibs (board.cpp)
  //Sets the specified stone, including overwriting existing stones, but only if doing so will
  //not result in any captures or zero liberty groups.
  //Returns false if location or color were out of range, or if would cause a zero liberty group.
  //In case of failure, will restore the position, but may result in chain ids or ordering in the board changing.
  pub fn set_stone_fail_if_no_libs(&mut self, loc: Loc, color: Color) -> bool {
    if loc < 0 || (loc as i32) >= Board::MAX_ARR_SIZE || self.colors[loc as usize] == C_WALL {
      return false;
    }
    if color != C_BLACK && color != C_WHITE && color != C_EMPTY {
      return false;
    }

    let old_ko_loc: Loc = self.ko_loc;
    if self.colors[loc as usize] == color {
    } else if self.colors[loc as usize] == C_EMPTY {
      if self.is_suicide(loc, color) || self.would_be_capture(loc, color) {
        return false;
      }
      self.play_move_assume_legal(loc, color);
    } else if color == C_EMPTY {
      self.remove_single_stone(loc);
    } else {
      debug_assert!(self.colors[loc as usize] == get_opp(color));
      self.remove_single_stone(loc);
      if self.is_suicide(loc, color) || self.would_be_capture(loc, color) {
        self.play_move_assume_legal(loc, get_opp(color));
        self.ko_loc = old_ko_loc;
        return false;
      }
      self.play_move_assume_legal(loc, color);
    }

    self.ko_loc = Board::NULL_LOC;
    true
  }

  /// upstream: Board::setStonesFailIfNoLibs (board.cpp)
  //Same, but sets multiple stones, and only requires that the final configuration contain no zero-liberty groups.
  //If it does contain a zero liberty group, fails and returns false and leaves the board in an arbitrarily changed but valid state.
  //Also returns false if any location is specified more than once.
  pub fn set_stones_fail_if_no_libs(&mut self, placements: &[Move]) -> bool {
    let mut locs: BTreeSet<Loc> = BTreeSet::new();
    for placement in placements {
      if locs.contains(&placement.loc) {
        return false;
      }
      locs.insert(placement.loc);
    }
    //First empty out all locations that we plan to set.
    //This guarantees avoiding any intermediate liberty issues.
    for placement in placements {
      let suc = self.set_stone_fail_if_no_libs(placement.loc, C_EMPTY);
      if !suc {
        return false;
      }
    }
    //Now set all the stones we wanted.
    for placement in placements {
      let suc = self.set_stone_fail_if_no_libs(placement.loc, placement.pla);
      if !suc {
        return false;
      }
    }
    true
  }

  /// upstream: Board::playMove (board.cpp)
  //Attempts to play the specified move. Returns true if successful, returns false if the move was illegal.
  pub fn play_move(&mut self, loc: Loc, pla: Player, is_multi_stone_suicide_legal: bool) -> bool {
    if self.is_legal(loc, pla, is_multi_stone_suicide_legal) {
      self.play_move_assume_legal(loc, pla);
      return true;
    }
    false
  }

  /// upstream: Board::playMoveRecorded (board.cpp)
  //Plays the specified move, assuming it is legal, and returns a MoveRecord for the move
  pub fn play_move_recorded(&mut self, loc: Loc, pla: Player) -> MoveRecord {
    let mut record = MoveRecord::default();
    record.loc = loc;
    record.pla = pla;
    record.ko_loc = self.ko_loc;
    record.cap_dirs = 0;

    if loc != Board::PASS_LOC {
      let opp: Player = get_opp(pla);

      {
        let adj: i32 = loc as i32 + (-(self.x_size + 1));
        if self.colors[adj as usize] == opp && self.get_num_liberties(adj as Loc) == 1 {
          record.cap_dirs |= 1u8 << 0;
        }
      }
      {
        let adj: i32 = loc as i32 + (-1);
        if self.colors[adj as usize] == opp && self.get_num_liberties(adj as Loc) == 1 {
          record.cap_dirs |= 1u8 << 1;
        }
      }
      {
        let adj: i32 = loc as i32 + 1;
        if self.colors[adj as usize] == opp && self.get_num_liberties(adj as Loc) == 1 {
          record.cap_dirs |= 1u8 << 2;
        }
      }
      {
        let adj: i32 = loc as i32 + (self.x_size + 1);
        if self.colors[adj as usize] == opp && self.get_num_liberties(adj as Loc) == 1 {
          record.cap_dirs |= 1u8 << 3;
        }
      }

      if record.cap_dirs == 0 && self.is_suicide(loc, pla) {
        record.cap_dirs = 0x10;
      }
    }

    self.play_move_assume_legal(loc, pla);
    record
  }

  /// upstream: Board::undo (board.cpp)
  //Undo the move given by record. Moves MUST be undone in the order they were made.
  //Undos will NOT typically restore the precise representation in the board to the way it was. The heads of chains
  //might change, the order of the circular lists might change, etc.
  pub fn undo(&mut self, record: MoveRecord) {
    self.ko_loc = record.ko_loc;

    let loc: Loc = record.loc;
    if loc == Board::PASS_LOC {
      return;
    }

    //Re-fill stones in all captured directions
    for i in 0..4 {
      let adj: i32 = loc as i32 + self.adj_offsets[i] as i32;
      if (record.cap_dirs as i32 & (1 << i)) != 0 {
        if self.colors[adj as usize] == C_EMPTY {
          self.add_chain(adj as Loc, get_opp(record.pla));

          let num_uncaptured: i32 = self.chain_data[self.chain_head[adj as usize] as usize].num_locs as i32;
          if record.pla == P_BLACK {
            self.num_white_captures -= num_uncaptured;
          } else {
            self.num_black_captures -= num_uncaptured;
          }
        }
      }
    }
    //Re-fill suicided stones
    if record.cap_dirs == 0x10 {
      debug_assert!(self.colors[loc as usize] == C_EMPTY);
      self.add_chain(loc, record.pla);
      let num_uncaptured: i32 = self.chain_data[self.chain_head[loc as usize] as usize].num_locs as i32;
      if record.pla == P_BLACK {
        self.num_black_captures -= num_uncaptured;
      } else {
        self.num_white_captures -= num_uncaptured;
      }
    }

    //Delete the stone played here.
    self.pos_hash ^= Board::statics().ZOBRIST_BOARD_HASH[loc as usize][self.colors[loc as usize] as usize];
    self.colors[loc as usize] = C_EMPTY;
    // empty_list.add(loc);

    //Uneat opp liberties
    self.change_surrounding_liberties(loc, get_opp(record.pla), 1);

    //If this was not a single stone, we may need to recompute the chain from scratch
    if self.chain_data[self.chain_head[loc as usize] as usize].num_locs > 1 {
      let mut num_neighbors: i32 = 0;
      for adjoffset in [-(self.x_size + 1), -1, 1, self.x_size + 1] {
        let adj: i32 = loc as i32 + adjoffset;
        if self.colors[adj as usize] == record.pla {
          num_neighbors += 1;
        }
      }

      //If the move had exactly one neighbor, we know its undoing didn't disconnect the group,
      //so don't need to rebuild the whole chain.
      if num_neighbors <= 1 {
        //If the undone move was the location of the head, we need to move the head.
        let mut head: Loc = self.chain_head[loc as usize];
        if head == loc {
          let new_head: Loc = self.next_in_chain[loc as usize];
          //Run through the whole chain and make their heads point to the new head
          let mut cur: Loc = loc;
          loop {
            self.chain_head[cur as usize] = new_head;
            cur = self.next_in_chain[cur as usize];
            if !(cur != loc) {
              break;
            }
          }

          //Move over the head data
          self.chain_data[new_head as usize] = self.chain_data[head as usize];
          head = new_head;
        }

        //Extract this move out of the circlar list of stones. Unfortunately we don't have a prev pointer, so we need to walk the loop.
        {
          //Starting at the head is likely to need to walk less since whenever we merge a single stone into an existing group
          //we put it right after the old head.
          let mut cur: Loc = head;
          while self.next_in_chain[cur as usize] != loc {
            cur = self.next_in_chain[cur as usize];
          }
          //Advance the pointer to put loc out of the loop
          self.next_in_chain[cur as usize] = self.next_in_chain[loc as usize];
        }

        //Lastly, fix up liberties. Removing this stone removed all liberties next to this stone
        //that weren't already liberties of the group.
        let mut liberty_delta: i32 = 0;
        for adjoffset in [-(self.x_size + 1), -1, 1, self.x_size + 1] {
          let adj: i32 = loc as i32 + adjoffset;
          if self.colors[adj as usize] == C_EMPTY && !self.is_liberty_of(adj as Loc, head) {
            liberty_delta -= 1;
          }
        }
        //Removing this stone itself added a liberty to the group though.
        liberty_delta += 1;
        self.chain_data[head as usize].num_liberties = (self.chain_data[head as usize].num_liberties as i32 + liberty_delta) as i16;
        //And update the count of stones in the group
        self.chain_data[head as usize].num_locs -= 1;
      }
      //More than one neighbor. Removing this stone potentially disconnects the group into two, so we just do a complete rebuild
      //of the resulting group(s).
      else {
        //Run through the whole chain and make their heads point to nothing
        let mut cur: Loc = loc;
        loop {
          self.chain_head[cur as usize] = Board::NULL_LOC;
          cur = self.next_in_chain[cur as usize];
          if !(cur != loc) {
            break;
          }
        }

        //Rebuild each chain adjacent now
        for i in 0..4 {
          let adj: i32 = loc as i32 + self.adj_offsets[i] as i32;
          if self.colors[adj as usize] == record.pla && self.chain_head[adj as usize] == Board::NULL_LOC {
            self.rebuild_chain(adj as Loc, record.pla);
          }
        }
      }
    }
  }

  /// upstream: Board::getPosHashAfterMove (board.cpp)
  //Get what the position hash would be if we were to play this move and resolve captures and suicides.
  //Assumes the move is on an empty location.
  pub fn get_pos_hash_after_move(&self, loc: Loc, pla: Player) -> Hash128 {
    if loc == Board::PASS_LOC {
      return self.pos_hash;
    }
    debug_assert!(loc != Board::NULL_LOC);

    let st = Board::statics();
    let mut hash: Hash128 = self.pos_hash;
    hash ^= st.ZOBRIST_BOARD_HASH[loc as usize][pla as usize];

    let opp: Player = get_opp(pla);

    //Count immediate liberties and groups that would be captured
    let mut would_be_suicide = true;
    let mut num_captured_groups: i32 = 0;
    let mut captured_group_heads: [Loc; 4] = [0; 4];
    for i in 0..4 {
      let adj: Loc = (loc as i32 + self.adj_offsets[i] as i32) as Loc;
      if self.colors[adj as usize] == C_EMPTY {
        would_be_suicide = false;
      } else if self.colors[adj as usize] == pla && self.get_num_liberties(adj) > 1 {
        would_be_suicide = false;
      } else if self.colors[adj as usize] == opp {
        //Capture!
        if self.get_num_liberties(adj) == 1 {
          //Make sure we haven't already counted it
          let head: Loc = self.chain_head[adj as usize];
          let mut already_found = false;
          for j in 0..num_captured_groups {
            if captured_group_heads[j as usize] == head {
              already_found = true;
              break;
            }
          }
          if !already_found {
            captured_group_heads[num_captured_groups as usize] = head;
            num_captured_groups += 1;
            would_be_suicide = false;

            //Now iterate through the group to update the hash
            let mut cur: Loc = adj;
            loop {
              hash ^= st.ZOBRIST_BOARD_HASH[cur as usize][opp as usize];
              cur = self.next_in_chain[cur as usize];
              if !(cur != adj) {
                break;
              }
            }
          }
        }
      }
    }

    //Update hash for suicidal moves
    if would_be_suicide {
      debug_assert!(num_captured_groups == 0);

      for i in 0..4 {
        let adj: Loc = (loc as i32 + self.adj_offsets[i] as i32) as Loc;
        //Suicide capture!
        if self.colors[adj as usize] == pla && self.get_num_liberties(adj) == 1 {
          //Make sure we haven't already counted it
          let head: Loc = self.chain_head[adj as usize];
          let mut already_found = false;
          for j in 0..num_captured_groups {
            if captured_group_heads[j as usize] == head {
              already_found = true;
              break;
            }
          }
          if !already_found {
            captured_group_heads[num_captured_groups as usize] = head;
            num_captured_groups += 1;

            //Now iterate through the group to update the hash
            let mut cur: Loc = adj;
            loop {
              hash ^= st.ZOBRIST_BOARD_HASH[cur as usize][pla as usize];
              cur = self.next_in_chain[cur as usize];
              if !(cur != adj) {
                break;
              }
            }
          }
        }
      }

      //Don't forget the stone we'd place would also die
      hash ^= st.ZOBRIST_BOARD_HASH[loc as usize][pla as usize];
    }

    hash
  }

  /// upstream: Board::playMoveAssumeLegal (board.cpp)
  //Plays the specified move, assuming it is legal.
  pub fn play_move_assume_legal(&mut self, loc: Loc, pla: Player) {
    //Pass?
    if loc == Board::PASS_LOC {
      self.ko_loc = Board::NULL_LOC;
      return;
    }

    let opp: Player = get_opp(pla);

    //Add the new stone as an independent group
    self.colors[loc as usize] = pla;
    self.pos_hash ^= Board::statics().ZOBRIST_BOARD_HASH[loc as usize][pla as usize];
    self.chain_data[loc as usize].owner = pla;
    self.chain_data[loc as usize].num_locs = 1;
    self.chain_data[loc as usize].num_liberties = self.get_num_immediate_liberties(loc) as i16;
    self.chain_head[loc as usize] = loc;
    self.next_in_chain[loc as usize] = loc;
    // empty_list.remove(loc);

    //Merge with surrounding friendly chains and capture any necessary opp chains
    let mut num_captured: i32 = 0; //Number of stones captured
    let mut possible_ko_loc: Loc = Board::NULL_LOC; //What location a ko ban might become possible in
    let mut num_opps_seen: i32 = 0; //How many opp chains we have seen so far
    let mut opp_heads_seen: [Loc; 4] = [0; 4]; //Heads of the opp chains seen so far

    for i in 0..4 {
      let adj: i32 = loc as i32 + self.adj_offsets[i] as i32;

      //Friendly chain!
      if self.colors[adj as usize] == pla {
        //Already merged?
        if self.chain_head[adj as usize] == self.chain_head[loc as usize] {
          continue;
        }

        //Otherwise, eat one liberty and merge them
        self.chain_data[self.chain_head[adj as usize] as usize].num_liberties -= 1;
        self.merge_chains(adj as Loc, loc);
      }
      //Opp chain!
      else if self.colors[adj as usize] == opp {
        let opp_head: Loc = self.chain_head[adj as usize];

        //Have we seen it already?
        let mut seen = false;
        for j in 0..num_opps_seen {
          if opp_heads_seen[j as usize] == opp_head {
            seen = true;
            break;
          }
        }

        if seen {
          continue;
        }

        //Not already seen! Eat one liberty from it and mark it as seen
        self.chain_data[opp_head as usize].num_liberties -= 1;
        opp_heads_seen[num_opps_seen as usize] = opp_head;
        num_opps_seen += 1;

        //Kill it?
        if self.get_num_liberties(adj as Loc) == 0 {
          num_captured += self.remove_chain(adj as Loc);
          possible_ko_loc = adj as Loc;
        }
      }
    }

    //We have a ko if 1 stone was captured and the capturing move is one isolated stone
    //And the capturing move itself now only has one liberty
    if num_captured == 1
      && self.chain_data[self.chain_head[loc as usize] as usize].num_locs == 1
      && self.chain_data[self.chain_head[loc as usize] as usize].num_liberties == 1
    {
      self.ko_loc = possible_ko_loc;
    } else {
      self.ko_loc = Board::NULL_LOC;
    }

    if pla == P_BLACK {
      self.num_white_captures += num_captured;
    } else {
      self.num_black_captures += num_captured;
    }

    //Handle suicide
    if self.get_num_liberties(loc) == 0 {
      let num_suicided: i32 = self.chain_data[self.chain_head[loc as usize] as usize].num_locs as i32;
      self.remove_chain(loc);

      if pla == P_BLACK {
        self.num_black_captures += num_suicided;
      } else {
        self.num_white_captures += num_suicided;
      }
    }
  }

  /// upstream: Board::getNumImmediateLiberties (board.cpp)
  //Gets the number of empty spaces directly adjacent to this location
  pub fn get_num_immediate_liberties(&self, loc: Loc) -> i32 {
    let mut num_libs: i32 = 0;
    if self.colors[(loc as i32 + (-(self.x_size + 1))) as usize] == C_EMPTY {
      num_libs += 1;
    }
    if self.colors[(loc as i32 + (-1)) as usize] == C_EMPTY {
      num_libs += 1;
    }
    if self.colors[(loc as i32 + 1) as usize] == C_EMPTY {
      num_libs += 1;
    }
    if self.colors[(loc as i32 + (self.x_size + 1)) as usize] == C_EMPTY {
      num_libs += 1;
    }

    num_libs
  }

  /// upstream: Board::countHeuristicConnectionLibertiesX2 (board.cpp)
  pub(crate) fn count_heuristic_connection_liberties_x2(&self, loc: Loc, pla: Player) -> i32 {
    let mut num_libs_x2: i32 = 0;
    for i in 0..4 {
      let adj: Loc = (loc as i32 + self.adj_offsets[i] as i32) as Loc;
      if self.colors[adj as usize] == pla {
        let libs: i32 = self.chain_data[self.chain_head[adj as usize] as usize].num_liberties as i32;
        if libs > 1 {
          num_libs_x2 += libs * 2 - 3;
        }
      }
    }
    num_libs_x2
  }

  /// upstream: Board::isLibertyOf (board.cpp)
  //Loc is a liberty of head's chain if loc is empty and adjacent to a stone of head.
  //Assumes loc is empty
  pub(crate) fn is_liberty_of(&self, loc: Loc, head: Loc) -> bool {
    let mut adj: Loc;
    adj = (loc as i32 + (-(self.x_size + 1))) as Loc;
    if self.colors[adj as usize] == self.colors[head as usize] && self.chain_head[adj as usize] == head {
      return true;
    }
    adj = (loc as i32 + (-1)) as Loc;
    if self.colors[adj as usize] == self.colors[head as usize] && self.chain_head[adj as usize] == head {
      return true;
    }
    adj = (loc as i32 + 1) as Loc;
    if self.colors[adj as usize] == self.colors[head as usize] && self.chain_head[adj as usize] == head {
      return true;
    }
    adj = (loc as i32 + (self.x_size + 1)) as Loc;
    if self.colors[adj as usize] == self.colors[head as usize] && self.chain_head[adj as usize] == head {
      return true;
    }

    false
  }

  /// upstream: Board::mergeChains (board.cpp)
  pub(crate) fn merge_chains(&mut self, loc1: Loc, loc2: Loc) {
    //Find heads
    let mut head1: Loc = self.chain_head[loc1 as usize];
    let mut head2: Loc = self.chain_head[loc2 as usize];

    debug_assert!(head1 != head2);
    debug_assert!(self.chain_data[head1 as usize].owner == self.chain_data[head2 as usize].owner);

    //Make sure head2 is the smaller chain.
    if self.chain_data[head1 as usize].num_locs < self.chain_data[head2 as usize].num_locs {
      let temp: Loc = head1;
      head1 = head2;
      head2 = temp;
    }

    //Iterate through each stone of head2's chain to make it a member of head1's chain.
    //Count new liberties for head1 as we go.
    self.chain_data[head1 as usize].num_locs += self.chain_data[head2 as usize].num_locs;
    let mut num_new_liberties: i32 = 0;
    let mut loc: Loc = head2;
    loop {
      //Any adjacent liberty is a new liberty for head1 if it is not adjacent to a stone of head1
      for adjoffset in [-(self.x_size + 1), -1, 1, self.x_size + 1] {
        let adj: Loc = (loc as i32 + adjoffset) as Loc;
        if self.colors[adj as usize] == C_EMPTY && !self.is_liberty_of(adj, head1) {
          num_new_liberties += 1;
        }
      }

      //Now, add this stone to head1.
      self.chain_head[loc as usize] = head1;

      //If are not back around, we are done.
      if self.next_in_chain[loc as usize] != head2 {
        loc = self.next_in_chain[loc as usize];
      } else {
        break;
      }
    }

    //Add in the liberties
    self.chain_data[head1 as usize].num_liberties = (self.chain_data[head1 as usize].num_liberties as i32 + num_new_liberties) as i16;

    //We link up (head1 -> next1 -> ... -> last1 -> head1) and (head2 -> next2 -> ... -> last2 -> head2)
    //as: head1 -> head2 -> next2 -> ... -> last2 -> next1 -> ... -> last1 -> head1
    //loc is now last_2
    self.next_in_chain[loc as usize] = self.next_in_chain[head1 as usize];
    self.next_in_chain[head1 as usize] = head2;
  }

  /// upstream: Board::removeChain (board.cpp)
  //Returns number of stones captured
  pub(crate) fn remove_chain(&mut self, loc: Loc) -> i32 {
    let mut num_stones_removed: i32 = 0; //Num stones removed
    let opp: Player = get_opp(self.colors[loc as usize]);

    //Walk around the chain...
    let mut cur: Loc = loc;
    loop {
      //Empty out this location
      self.pos_hash ^= Board::statics().ZOBRIST_BOARD_HASH[cur as usize][self.colors[cur as usize] as usize];
      self.colors[cur as usize] = C_EMPTY;
      num_stones_removed += 1;
      // empty_list.add(cur);

      //For each distinct opp chain around, add a liberty to it.
      self.change_surrounding_liberties(cur, opp, 1);

      cur = self.next_in_chain[cur as usize];

      if !(cur != loc) {
        break;
      }
    }

    num_stones_removed
  }

  /// upstream: Board::removeSingleStone (board.cpp)
  //Remove a single stone, even a stone part of a larger group.
  pub(crate) fn remove_single_stone(&mut self, loc: Loc) {
    let pla: Player = self.colors[loc as usize];

    //Save the entire chain's stone locations
    let num_locs: i32 = self.chain_data[self.chain_head[loc as usize] as usize].num_locs as i32;
    let mut locs: [i32; Board::MAX_PLAY_SIZE as usize] = [0; Board::MAX_PLAY_SIZE as usize];
    let mut idx: i32 = 0;
    let mut cur: Loc = loc;
    loop {
      locs[idx as usize] = cur as i32;
      idx += 1;
      cur = self.next_in_chain[cur as usize];
      if !(cur != loc) {
        break;
      }
    }
    debug_assert!(idx == num_locs);

    //Delete the entire chain
    self.remove_chain(loc);

    //Then add all the other stones back one by one.
    for i in 0..num_locs {
      if locs[i as usize] != loc as i32 {
        self.play_move_assume_legal(locs[i as usize] as Loc, pla);
      }
    }
  }

  /// upstream: Board::addChain (board.cpp)
  //Add a chain of the given player to the given region of empty space, floodfilling it.
  //Assumes that this region does not border any chains of the desired color already
  pub(crate) fn add_chain(&mut self, loc: Loc, pla: Player) {
    self.chain_data[loc as usize].num_liberties = 0;
    self.chain_data[loc as usize].num_locs = 0;
    self.chain_data[loc as usize].owner = pla;

    //Add a chain with links front -> ... -> loc -> loc with all head pointers towards loc
    let front: Loc = self.add_chain_helper(loc, loc, loc, pla);

    //Now, we make loc point to front, and that completes the circle!
    self.next_in_chain[loc as usize] = front;
  }

  /// upstream: Board::addChainHelper (board.cpp)
  //Floodfill a chain of the given color into this region of empty spaces
  //Make the specified loc the head for all the chains and updates the chainData of head with the number of stones.
  //Does NOT connect the stones into a circular list. Rather, it produces an linear linked list with the tail pointing
  //to tailTarget, and returns the head of the list. The tail is guaranteed to be loc.
  pub(crate) fn add_chain_helper(&mut self, head: Loc, tail_target: Loc, loc: Loc, pla: Player) -> Loc {
    //Add stone here
    self.colors[loc as usize] = pla;
    self.pos_hash ^= Board::statics().ZOBRIST_BOARD_HASH[loc as usize][pla as usize];
    self.chain_head[loc as usize] = head;
    self.chain_data[head as usize].num_locs += 1;
    self.next_in_chain[loc as usize] = tail_target;
    // empty_list.remove(loc);

    //Eat opp liberties
    self.change_surrounding_liberties(loc, get_opp(pla), -1);

    //Recursively add stones around us.
    let mut next_tail_target: Loc = loc;
    for i in 0..4 {
      let adj: Loc = (loc as i32 + self.adj_offsets[i] as i32) as Loc;
      if self.colors[adj as usize] == C_EMPTY {
        next_tail_target = self.add_chain_helper(head, next_tail_target, adj, pla);
      }
    }
    next_tail_target
  }

  /// upstream: Board::rebuildChain (board.cpp)
  //Floods through a chain of the specified player already on the board
  //rebuilding its links and counting its liberties as we go.
  //Requires that all their heads point towards
  //some invalid location, such as NULL_LOC or a location not of color.
  //The head of the chain will be loc.
  pub(crate) fn rebuild_chain(&mut self, loc: Loc, pla: Player) {
    self.chain_data[loc as usize].num_liberties = 0;
    self.chain_data[loc as usize].num_locs = 0;
    self.chain_data[loc as usize].owner = pla;

    //Rebuild chain with links front -> ... -> loc -> loc with all head pointers towards loc
    let front: Loc = self.rebuild_chain_helper(loc, loc, loc, pla);

    //Now, we make loc point to front, and that completes the circle!
    self.next_in_chain[loc as usize] = front;
  }

  /// upstream: Board::rebuildChainHelper (board.cpp)
  //Does same thing as addChain, but floods through a chain of the specified color already on the board
  //rebuilding its links and also counts its liberties as we go. Requires that all their heads point towards
  //some invalid location, such as NULL_LOC or a location not of color.
  pub(crate) fn rebuild_chain_helper(&mut self, head: Loc, tail_target: Loc, loc: Loc, pla: Player) -> Loc {
    let adj0: Loc = (loc as i32 + (-(self.x_size + 1))) as Loc;
    let adj1: Loc = (loc as i32 + (-1)) as Loc;
    let adj2: Loc = (loc as i32 + 1) as Loc;
    let adj3: Loc = (loc as i32 + (self.x_size + 1)) as Loc;

    //Count new liberties
    let mut num_head_liberties_to_add: i32 = 0;
    if self.colors[adj0 as usize] == C_EMPTY && !self.is_liberty_of(adj0, head) {
      num_head_liberties_to_add += 1;
    }
    if self.colors[adj1 as usize] == C_EMPTY && !self.is_liberty_of(adj1, head) {
      num_head_liberties_to_add += 1;
    }
    if self.colors[adj2 as usize] == C_EMPTY && !self.is_liberty_of(adj2, head) {
      num_head_liberties_to_add += 1;
    }
    if self.colors[adj3 as usize] == C_EMPTY && !self.is_liberty_of(adj3, head) {
      num_head_liberties_to_add += 1;
    }
    self.chain_data[head as usize].num_liberties = (self.chain_data[head as usize].num_liberties as i32 + num_head_liberties_to_add) as i16;

    //Add stone here to the chain by setting its head
    self.chain_head[loc as usize] = head;
    self.chain_data[head as usize].num_locs += 1;
    self.next_in_chain[loc as usize] = tail_target;

    //Recursively add stones around us.
    let mut next_tail_target: Loc = loc;
    if self.colors[adj0 as usize] == pla && self.chain_head[adj0 as usize] != head {
      next_tail_target = self.rebuild_chain_helper(head, next_tail_target, adj0, pla);
    }
    if self.colors[adj1 as usize] == pla && self.chain_head[adj1 as usize] != head {
      next_tail_target = self.rebuild_chain_helper(head, next_tail_target, adj1, pla);
    }
    if self.colors[adj2 as usize] == pla && self.chain_head[adj2 as usize] != head {
      next_tail_target = self.rebuild_chain_helper(head, next_tail_target, adj2, pla);
    }
    if self.colors[adj3 as usize] == pla && self.chain_head[adj3 as usize] != head {
      next_tail_target = self.rebuild_chain_helper(head, next_tail_target, adj3, pla);
    }
    next_tail_target
  }

  /// upstream: Board::changeSurroundingLiberties (board.cpp)
  //Apply the specified delta to the liberties of all adjacent groups of the specified color
  pub(crate) fn change_surrounding_liberties(&mut self, loc: Loc, pla: Player, delta: i32) {
    let adj0: Loc = (loc as i32 + (-(self.x_size + 1))) as Loc;
    let adj1: Loc = (loc as i32 + (-1)) as Loc;
    let adj2: Loc = (loc as i32 + 1) as Loc;
    let adj3: Loc = (loc as i32 + (self.x_size + 1)) as Loc;
    let (a0, a1, a2, a3) = (adj0 as usize, adj1 as usize, adj2 as usize, adj3 as usize);

    if self.colors[a0] == pla {
      let h = self.chain_head[a0] as usize;
      self.chain_data[h].num_liberties = (self.chain_data[h].num_liberties as i32 + delta) as i16;
    }
    if self.colors[a1] == pla && !(self.colors[a0] == pla && self.chain_head[a0] == self.chain_head[a1]) {
      let h = self.chain_head[a1] as usize;
      self.chain_data[h].num_liberties = (self.chain_data[h].num_liberties as i32 + delta) as i16;
    }
    if self.colors[a2] == pla
      && !(self.colors[a0] == pla && self.chain_head[a0] == self.chain_head[a2])
      && !(self.colors[a1] == pla && self.chain_head[a1] == self.chain_head[a2])
    {
      let h = self.chain_head[a2] as usize;
      self.chain_data[h].num_liberties = (self.chain_data[h].num_liberties as i32 + delta) as i16;
    }
    if self.colors[a3] == pla
      && !(self.colors[a0] == pla && self.chain_head[a0] == self.chain_head[a3])
      && !(self.colors[a1] == pla && self.chain_head[a1] == self.chain_head[a3])
      && !(self.colors[a2] == pla && self.chain_head[a2] == self.chain_head[a3])
    {
      let h = self.chain_head[a3] as usize;
      self.chain_data[h].num_liberties = (self.chain_data[h].num_liberties as i32 + delta) as i16;
    }
  }
}

// Board::PointList (board.h, board.cpp) is commented out upstream: not ported.

impl Location {
  /// upstream: Location::distance (board.cpp)
  pub fn distance(loc0: Loc, loc1: Loc, x_size: i32) -> i32 {
    let dx: i32 = Location::get_x(loc1, x_size) - Location::get_x(loc0, x_size);
    let dy: i32 = (loc1 as i32 - loc0 as i32 - dx) / (x_size + 1);
    (if dx >= 0 { dx } else { -dx }) + (if dy >= 0 { dy } else { -dy })
  }

  /// upstream: Location::euclideanDistanceSquared (board.cpp)
  pub fn euclidean_distance_squared(loc0: Loc, loc1: Loc, x_size: i32) -> i32 {
    let dx: i32 = Location::get_x(loc1, x_size) - Location::get_x(loc0, x_size);
    let dy: i32 = (loc1 as i32 - loc0 as i32 - dx) / (x_size + 1);
    dx * dx + dy * dy
  }
}

//TACTICAL STUFF--------------------------------------------------------------------

impl Board {
  /// upstream: Board::findLiberties (board.cpp)
  //Helper, find liberties of group at loc. Fills in buf, returns the number of liberties.
  //bufStart is where to start checking to avoid duplicates. bufIdx is where to start actually writing.
  pub(crate) fn find_liberties(&self, loc: Loc, buf: &mut Vec<Loc>, buf_start: i32, buf_idx: i32) -> i32 {
    let mut num_found: i32 = 0;
    let mut cur: Loc = loc;
    loop {
      for i in 0..4 {
        let lib: Loc = (cur as i32 + self.adj_offsets[i] as i32) as Loc;
        if self.colors[lib as usize] == C_EMPTY {
          //Check for dups
          let mut found_dup = false;
          let mut j = buf_start;
          while j < buf_idx + num_found {
            if buf[j as usize] == lib {
              found_dup = true;
              break;
            }
            j += 1;
          }
          if !found_dup {
            if ((buf_idx + num_found) as usize) >= buf.len() {
              let new_len = buf.len() * 3 / 2 + 64;
              buf.resize(new_len, 0);
            }
            buf[(buf_idx + num_found) as usize] = lib;
            num_found += 1;
          }
        }
      }

      cur = self.next_in_chain[cur as usize];
      if !(cur != loc) {
        break;
      }
    }

    num_found
  }

  /// upstream: Board::findLibertyGainingCaptures (board.cpp)
  //Helper, find captures that gain liberties for the group at loc. Fills in result, returns the number of captures.
  //bufStart is where to start checking to avoid duplicates. bufIdx is where to start actually writing.
  pub(crate) fn find_liberty_gaining_captures(&self, loc: Loc, buf: &mut Vec<Loc>, buf_start: i32, buf_idx: i32) -> i32 {
    let opp: Player = get_opp(self.colors[loc as usize]);

    //For performance, avoid checking for captures on any chain twice
    //int arrSize = x_size*y_size;
    let mut chain_heads_checked: [Loc; Board::MAX_PLAY_SIZE as usize] = [0; Board::MAX_PLAY_SIZE as usize];
    let mut num_chain_heads_checked: i32 = 0;

    let mut num_found: i32 = 0;
    let mut cur: Loc = loc;
    loop {
      for i in 0..4 {
        let adj: Loc = (cur as i32 + self.adj_offsets[i] as i32) as Loc;
        if self.colors[adj as usize] == opp {
          let head: Loc = self.chain_head[adj as usize];
          if self.chain_data[head as usize].num_liberties == 1 {
            let mut already_checked = false;
            for j in 0..num_chain_heads_checked {
              if chain_heads_checked[j as usize] == head {
                already_checked = true;
                break;
              }
            }
            if !already_checked {
              //Capturing moves are precisely the liberties of the groups around us with 1 liberty.
              num_found += self.find_liberties(adj, buf, buf_start, buf_idx + num_found);
              chain_heads_checked[num_chain_heads_checked as usize] = head;
              num_chain_heads_checked += 1;
            }
          }
        }
      }

      cur = self.next_in_chain[cur as usize];
      if !(cur != loc) {
        break;
      }
    }

    num_found
  }

  /// upstream: Board::hasLibertyGainingCaptures (board.cpp)
  //Helper, does the group at loc have at least one opponent group adjacent to it in atari?
  pub(crate) fn has_liberty_gaining_captures(&self, loc: Loc) -> bool {
    let opp: Player = get_opp(self.colors[loc as usize]);
    let mut cur: Loc = loc;
    loop {
      for adjoffset in [-(self.x_size + 1), -1, 1, self.x_size + 1] {
        let adj: Loc = (cur as i32 + adjoffset) as Loc;
        if self.colors[adj as usize] == opp {
          let head: Loc = self.chain_head[adj as usize];
          if self.chain_data[head as usize].num_liberties == 1 {
            return true;
          }
        }
      }
      cur = self.next_in_chain[cur as usize];
      if !(cur != loc) {
        break;
      }
    }

    false
  }

  /// upstream: Board::searchIsLadderCapturedAttackerFirst2Libs (board.cpp)
  pub fn search_is_ladder_captured_attacker_first2_libs(&mut self, loc: Loc, buf: &mut Vec<Loc>, working_moves: &mut Vec<Loc>) -> bool {
    if loc < 0 || (loc as i32) >= Board::MAX_ARR_SIZE {
      return false;
    }
    if self.colors[loc as usize] != C_BLACK && self.colors[loc as usize] != C_WHITE {
      return false;
    }
    if self.chain_data[self.chain_head[loc as usize] as usize].num_liberties != 2 {
      return false;
    }

    //Make it so that pla is always the defender
    let pla: Player = self.colors[loc as usize];
    let opp: Player = get_opp(pla);

    let num_libs: i32 = self.find_liberties(loc, buf, 0, 0);
    debug_assert!(num_libs == 2);
    let _ = num_libs; //Avoid warning when asserts are off

    let move0: Loc = buf[0];
    let move1: Loc = buf[1];
    let mut move0_works = false;
    let mut move1_works = false;

    //Suicide never relevant for ladders
    //Attacker: A suicide move cannot reduce the defender's liberties
    //Defender: A suicide move cannot gain liberties
    let is_multi_stone_suicide_legal = false;
    if self.is_legal(move0, opp, is_multi_stone_suicide_legal) {
      let record: MoveRecord = self.play_move_recorded(move0, opp);
      move0_works = self.search_is_ladder_captured(loc, true, buf);
      self.undo(record);
    }
    if self.is_legal(move1, opp, is_multi_stone_suicide_legal) {
      let record: MoveRecord = self.play_move_recorded(move1, opp);
      move1_works = self.search_is_ladder_captured(loc, true, buf);
      self.undo(record);
    }

    if move0_works || move1_works {
      working_moves.clear();
      if move0_works {
        working_moves.push(move0);
      }
      if move1_works {
        working_moves.push(move1);
      }
      return true;
    }
    false
  }

  /// upstream: Board::searchIsLadderCaptured (board.cpp)
  //Check if the given stone is in unescapable atari or can be put into unescapable atari.
  //WILL perform a mutable search - may alter the linked lists or heads, etc.
  pub fn search_is_ladder_captured(&mut self, loc: Loc, defender_first: bool, buf: &mut Vec<Loc>) -> bool {
    if loc < 0 || (loc as i32) >= Board::MAX_ARR_SIZE {
      return false;
    }
    if self.colors[loc as usize] != C_BLACK && self.colors[loc as usize] != C_WHITE {
      return false;
    }

    if self.chain_data[self.chain_head[loc as usize] as usize].num_liberties > 2
      || (defender_first && self.chain_data[self.chain_head[loc as usize] as usize].num_liberties > 1)
    {
      return false;
    }

    //Make it so that pla is always the defender
    let pla: Player = self.colors[loc as usize];
    let opp: Player = get_opp(pla);

    //Clear the ko loc for the defender at the root node - assume all kos work for the defender
    let ko_loc_saved: Loc = self.ko_loc;
    if defender_first {
      self.ko_loc = Board::NULL_LOC;
    }

    //Stack for the search. These point to lists of possible moves to search at each level of the stack, indices refer to indices in [buf].
    let stack_size: i32 = self.x_size * self.y_size * 3 / 2 + 1; //A bit bigger due to paranoia about recaptures making the sequence longer.
    const arrSize: usize = (Board::MAX_PLAY_SIZE * 3 / 2 + 1) as usize;
    // PERF(port): zero-init where C++ leaves uninitialised (these four arrays, MAX_PLAY_SIZE*3/2+1 entries each,
    // on every call; `records` is MoveRecord::default()).
    let mut move_list_starts: [i32; arrSize] = [0; arrSize]; //Buf idx of start of list
    let mut move_list_lens: [i32; arrSize] = [0; arrSize]; //Len of list
    let mut move_list_cur: [i32; arrSize] = [0; arrSize]; //Current move list idx searched, equal to -1 if list has not been generated.
    let mut records: [MoveRecord; arrSize] = [MoveRecord::default(); arrSize]; //Records so that we can undo moves as we search back up.
    let mut stack_idx: i32 = 0;
    let mut search_node_count: i32 = 0;
    const MAX_LADDER_SEARCH_NODE_BUDGET: i32 = 25000;

    move_list_cur[0] = -1;
    move_list_starts[0] = 0;
    move_list_lens[0] = 0;
    let mut return_value = false;
    let mut returned_from_deeper = false;
    // bool print = true;

    loop {
      // if(print) cout << ": " << stackIdx << " " << moveListCur[stackIdx] << " " << moveListStarts[stackIdx] << " " << moveListLens[stackIdx] << " " << returnValue << " " << returnedFromDeeper << endl;

      //Returned from the root - so that's the answer
      if stack_idx <= -1 {
        debug_assert!(stack_idx == -1);
        self.ko_loc = ko_loc_saved;
        return return_value;
      }

      //If we hit the stack limit, just consider it a failed ladder.
      if stack_idx >= stack_size - 1 {
        return_value = true;
        returned_from_deeper = true;
        stack_idx -= 1;
        continue;
      }
      //If we hit a total node count limit, then just assume it doesn't work.
      if search_node_count >= MAX_LADDER_SEARCH_NODE_BUDGET {
        stack_idx -= 1;
        while stack_idx >= 0 {
          self.undo(records[stack_idx as usize]);
          stack_idx -= 1;
        }
        return false;
      }

      let is_defender: bool = (defender_first && (stack_idx % 2) == 0) || (!defender_first && (stack_idx % 2) == 1);

      //We just entered this level?
      if move_list_cur[stack_idx as usize] == -1 {
        let libs: i32 = self.chain_data[self.chain_head[loc as usize] as usize].num_liberties as i32;

        //Base cases.
        //If we are the attacker and the group has only 1 liberty, we already win.
        if !is_defender && libs <= 1 {
          return_value = true;
          returned_from_deeper = true;
          stack_idx -= 1;
          continue;
        }
        //If we are the attacker and the group has 3 liberties, we already lose.
        if !is_defender && libs >= 3 {
          return_value = false;
          returned_from_deeper = true;
          stack_idx -= 1;
          continue;
        }
        //If we are the defender and the group has 2 liberties, we already win.
        if is_defender && libs >= 2 {
          return_value = false;
          returned_from_deeper = true;
          stack_idx -= 1;
          continue;
        }
        //If we are the defender and the attacker left a simple ko point, assume we already win
        //because we don't want to say yes on ladders that depend on kos
        //This should also hopefully prevent any possible infinite loops - I don't know of any infinite loop
        //that would come up in a continuous atari sequence that doesn't ever leave a simple ko point.
        if is_defender && self.ko_loc != Board::NULL_LOC {
          return_value = false;
          returned_from_deeper = true;
          stack_idx -= 1;
          continue;
        }

        //Otherwise we need to keep searching.
        //Generate the move list. Attacker and defender generate moves on the group's liberties, but only the defender
        //generates moves on surrounding capturable opposing groups.
        let start: i32 = move_list_starts[stack_idx as usize];
        let mut move_list_len: i32 = 0;
        if is_defender {
          move_list_len = self.find_liberty_gaining_captures(loc, buf, start, start);
          move_list_len += self.find_liberties(loc, buf, start, start + move_list_len);

          let mut lower_bound_libs: i32 = 0;
          let mut upper_bound_libs: i32 = 0;
          //List is always nonempty, and the last element always is the lone liberty of the defender group
          self.get_bound_num_liberties_after_play(buf[(start + move_list_len - 1) as usize], pla, &mut lower_bound_libs, &mut upper_bound_libs);
          //Defender immediately wins if there are provably enough libs
          if lower_bound_libs >= 3 {
            return_value = false;
            returned_from_deeper = true;
            stack_idx -= 1;
            continue;
          }
          //Attacker immediately wins if defender has not enough libs and there are no alternatives
          if move_list_len == 1 && upper_bound_libs <= 1 {
            return_value = true;
            returned_from_deeper = true;
            stack_idx -= 1;
            continue;
          }
        } else {
          move_list_len += self.find_liberties(loc, buf, start, start);
          // if(moveListLen != 2) {
          //   cout << *this << endl;
          //   cout << stackIdx << endl;
          //   for(int i = 0; i<stackIdx; i++) {
          //     cout << moveListCur[stackIdx] << " " << moveListStarts[stackIdx] << " " << moveListLens[stackIdx] << " "
          //          << Location::toString(buf[moveListStarts[stackIdx] + moveListCur[stackIdx]],*this) << endl;
          //   }
          //   cout << "===" << endl;
          //   checkConsistency();
          // }
          debug_assert!(move_list_len == 2);

          let mut libs0: i32 = self.get_num_immediate_liberties(buf[start as usize]);
          let mut libs1: i32 = self.get_num_immediate_liberties(buf[(start + 1) as usize]);

          //If we are the attacker and we're in a double-ko death situation, then assume we win.
          //Both defender liberties must be ko mouths, connecting either ko mouth must not increase the defender's
          //liberties, and none of the attacker's surrounding stones can currently be in atari.
          //This is not complete - there are situations where the defender's connections increase liberties, or where
          //the attacker has stones in atari, but where the defender is still in inescapable atari even if they have
          //a large finite number of ko threats. But it's better than nothing.
          if libs0 == 0
            && libs1 == 0
            && self.would_be_ko_capture(buf[start as usize], opp)
            && self.would_be_ko_capture(buf[(start + 1) as usize], opp)
          {
            if self.get_num_liberties_after_play(buf[start as usize], pla, 3) <= 2
              && self.get_num_liberties_after_play(buf[(start + 1) as usize], pla, 3) <= 2
            {
              if !self.has_liberty_gaining_captures(loc) {
                return_value = true;
                returned_from_deeper = true;
                stack_idx -= 1;
                continue;
              }
            }
          }

          //Early quitouts if the liberties are not adjacent
          //(so that filling one doesn't fill an immediate liberty of the other)
          if !Location::is_adjacent(buf[start as usize], buf[(start + 1) as usize], self.x_size) {
            //We lose automatically if both escapes get the defender too many libs
            if libs0 >= 3 && libs1 >= 3 {
              return_value = false;
              returned_from_deeper = true;
              stack_idx -= 1;
              continue;
            }
            //Move 1 is not possible, so shrink the list
            else if libs0 >= 3 {
              move_list_len = 1;
            }
            //Move 0 is not possible, so swap and shrink the list
            else if libs1 >= 3 {
              buf[start as usize] = buf[(start + 1) as usize];
              move_list_len = 1;
            }
          }
          //Order the two moves based on a simple heuristic - for each neighboring group with any liberties
          //count that the opponent could connect to, count liberties - 1.5.
          if move_list_len > 1 {
            libs0 = libs0 * 2 + self.count_heuristic_connection_liberties_x2(buf[start as usize], pla);
            libs1 = libs1 * 2 + self.count_heuristic_connection_liberties_x2(buf[(start + 1) as usize], pla);
            if libs1 > libs0 {
              let tmp: i32 = buf[start as usize] as i32;
              buf[start as usize] = buf[(start + 1) as usize];
              buf[(start + 1) as usize] = tmp as Loc;
            }
          }
        }
        move_list_lens[stack_idx as usize] = move_list_len;

        //And indicate to begin search on the first move generated.
        move_list_cur[stack_idx as usize] = 0;
      }
      //Else, we returned from a deeper level (or the same level, via illegal move)
      else {
        debug_assert!(move_list_cur[stack_idx as usize] >= 0);
        debug_assert!(move_list_cur[stack_idx as usize] < move_list_lens[stack_idx as usize]);
        //If we returned from deeper we need to undo the move we made
        if returned_from_deeper {
          self.undo(records[stack_idx as usize]);
        }

        //Defender has a move that is not ladder captured?
        if is_defender && !return_value {
          //Return! (returnValue is still false, as desired)
          returned_from_deeper = true;
          stack_idx -= 1;
          continue;
        }
        //Attacker has a move that does ladder capture?
        if !is_defender && return_value {
          //Return! (returnValue is still true, as desired)
          returned_from_deeper = true;
          stack_idx -= 1;
          continue;
        }

        //Move on to the next move to search
        move_list_cur[stack_idx as usize] += 1;
      }

      //If there is no next move to search, then we lose.
      if move_list_cur[stack_idx as usize] >= move_list_lens[stack_idx as usize] {
        //For a defender, that means a ladder capture.
        //For an attacker, that means no ladder capture found.
        return_value = is_defender;
        returned_from_deeper = true;
        stack_idx -= 1;
        continue;
      }

      //Otherwise we do have an next move to search. Grab it.
      let mv: Loc = buf[(move_list_starts[stack_idx as usize] + move_list_cur[stack_idx as usize]) as usize];
      let p: Player = if is_defender { pla } else { opp };

      // if(print) cout << "play " << Location::getX(move,x_size) << " " << Location::getY(move,x_size) << " " << p << endl;

      //Illegal move - treat it the same as a failed move, but don't return up a level so that we
      //loop again and just try the next move.
      let is_multi_stone_suicide_legal = false;
      if !self.is_legal(mv, p, is_multi_stone_suicide_legal) {
        return_value = is_defender;
        returned_from_deeper = false;
        // if(print) cout << "illegal " << endl;
        continue;
      }

      //Play and record the move!
      records[stack_idx as usize] = self.play_move_recorded(mv, p);
      search_node_count += 1;

      //And recurse to the next level
      stack_idx += 1;
      move_list_cur[stack_idx as usize] = -1;
      move_list_starts[stack_idx as usize] = move_list_starts[(stack_idx - 1) as usize] + move_list_lens[(stack_idx - 1) as usize];
      move_list_lens[stack_idx as usize] = 0;
    }
  }

  /// upstream: Board::calculateArea (board.cpp)
  //If a point is a pass-alive stone or pass-alive territory for a color, mark it that color.
  //If nonPassAliveStones, also marks non-pass-alive stones that are not part of the opposing pass-alive territory.
  //If safeBigTerritories, also marks for each pla empty regions bordered by pla stones and no opp stones, where all pla stones are pass-alive.
  //If unsafeBigTerritories, also marks for each pla empty regions bordered by pla stones and no opp stones, regardless.
  //All other points are marked as C_EMPTY.
  //[result] must be a buffer of size MAX_ARR_SIZE and will get filled with the result
  pub fn calculate_area(
    &self,
    result: &mut [Color],
    non_pass_alive_stones: bool,
    safe_big_territories: bool,
    unsafe_big_territories: bool,
    is_multi_stone_suicide_legal: bool,
  ) {
    result[..Board::MAX_ARR_SIZE as usize].fill(C_EMPTY);
    self.calculate_area_for_pla(P_BLACK, safe_big_territories, unsafe_big_territories, is_multi_stone_suicide_legal, result);
    self.calculate_area_for_pla(P_WHITE, safe_big_territories, unsafe_big_territories, is_multi_stone_suicide_legal, result);

    //TODO can we merge this in to calculate area for pla?
    if non_pass_alive_stones {
      for y in 0..self.y_size {
        for x in 0..self.x_size {
          let loc: Loc = Location::get_loc(x, y, self.x_size);
          if result[loc as usize] == C_EMPTY {
            result[loc as usize] = self.colors[loc as usize];
          }
        }
      }
    }
  }

  /// upstream: Board::calculateIndependentLifeArea (board.cpp)
  //Calculates the area (including non pass alive stones, safe and unsafe big territories)
  //However, strips out any "seki" regions.
  //Seki regions are that are adjacent to any remaining empty regions.
  //If keepTerritories, then keeps the surrounded territories in seki regions, only strips points for stones.
  //If keepStones, then keeps the stones, only strips points for surrounded territories.
  //whiteMinusBlackIndependentLifeRegionCount - multiply this by two for a group tax.
  pub fn calculate_independent_life_area(
    &self,
    result: &mut [Color],
    white_minus_black_independent_life_region_count: &mut i32,
    keep_territories: bool,
    keep_stones: bool,
    is_multi_stone_suicide_legal: bool,
  ) {
    //First, just compute basic area.
    let mut basic_area: [Color; Board::MAX_ARR_SIZE as usize] = [0; Board::MAX_ARR_SIZE as usize];
    result[..Board::MAX_ARR_SIZE as usize].fill(C_EMPTY);
    basic_area.fill(C_EMPTY);
    self.calculate_area_for_pla(P_BLACK, true, true, is_multi_stone_suicide_legal, &mut basic_area);
    self.calculate_area_for_pla(P_WHITE, true, true, is_multi_stone_suicide_legal, &mut basic_area);

    //TODO can we merge this in to calculate area for pla?
    for y in 0..self.y_size {
      for x in 0..self.x_size {
        let loc: Loc = Location::get_loc(x, y, self.x_size);
        if basic_area[loc as usize] == C_EMPTY {
          basic_area[loc as usize] = self.colors[loc as usize];
        }
      }
    }

    self.calculate_independent_life_area_helper(&basic_area, result, white_minus_black_independent_life_region_count);

    if keep_territories {
      for y in 0..self.y_size {
        for x in 0..self.x_size {
          let loc: Loc = Location::get_loc(x, y, self.x_size);
          if basic_area[loc as usize] != C_EMPTY && basic_area[loc as usize] != self.colors[loc as usize] {
            result[loc as usize] = basic_area[loc as usize];
          }
        }
      }
    }
    if keep_stones {
      for y in 0..self.y_size {
        for x in 0..self.x_size {
          let loc: Loc = Location::get_loc(x, y, self.x_size);
          if basic_area[loc as usize] != C_EMPTY && basic_area[loc as usize] == self.colors[loc as usize] {
            result[loc as usize] = basic_area[loc as usize];
          }
        }
      }
    }
  }

  /// upstream: Board::calculateAreaForPla (board.cpp)
  //This marks pass-alive stones, pass-alive territory always.
  //If safeBigTerritories, marks empty regions bordered by pla stones and no opp stones, where all pla stones are pass-alive.
  //If unsafeBigTerritories, marks empty regions bordered by pla stones and no opp stones, but ONLY on locations in result that are C_EMPTY.
  //The reason for this is to avoid overwriting the opponent's pass-alive territory in situations like this:
  // .ox.x.x
  // oxxxxxx
  // xx.....
  //The top left corner is black's pass-alive territory. It's also an empty region bordered only by white, but we should not mark
  //it as white's unsafeBigTerritory because it's already marked as black's pass alive territory.
  pub(crate) fn calculate_area_for_pla(
    &self,
    pla: Player,
    safe_big_territories: bool,
    unsafe_big_territories: bool,
    is_multi_stone_suicide_legal: bool,
    result: &mut [Color],
  ) {
    let opp: Color = get_opp(pla);

    //https://senseis.xmp.net/?BensonsAlgorithm
    //https://zhuanlan.zhihu.com/p/110998764
    //First compute all empty-or-opp regions

    //For each loc, if it's empty or opp, the index of the region
    // PERF(port): zero-init where C++ leaves uninitialised (this array and the MAX_ARR_SIZE /
    // maxRegions arrays below; the first three are then filled again as upstream does).
    let mut region_idx_by_loc: [i16; Board::MAX_ARR_SIZE as usize] = [0; Board::MAX_ARR_SIZE as usize];
    //For each loc, if it's empty or opp, the next empty or opp belonging to the same region
    let mut next_empty_or_opp: [Loc; Board::MAX_ARR_SIZE as usize] = [0; Board::MAX_ARR_SIZE as usize];
    //Does this border a pla group that has been marked as not pass alive?
    let mut borders_non_pass_alive_pla_by_head: [bool; Board::MAX_ARR_SIZE as usize] = [false; Board::MAX_ARR_SIZE as usize];

    //Set to initial values. Faster than std::fill in O2 optimization by compiler, might be similar when use O3.
    //This code assumes twos complement. Memset only sets bytes, so we are relying on the concatenation of two -1 bytes
    //being the same as (int16_t)(-1). Technically C++ doesn't mandate twos-complement, but everything in practice uses it.
    region_idx_by_loc.fill(-1);
    //This memset is only safe because Board::NULL_LOC is 0. If it were a nonzero value, setting adjacent bytes to that
    //value is NOT equivalent to setting each Loc (2 bytes) to that value.
    const _: () = assert!(Board::NULL_LOC == 0, "Memset in Board::calculateAreaForPla relies on Board::NULL_LOC == 0");
    next_empty_or_opp.fill(Board::NULL_LOC);
    borders_non_pass_alive_pla_by_head.fill(false);

    //A list for each region head, indicating which pla group heads the region is vital for.
    //A region is vital for a pla group if all its spaces are adjacent to that pla group.
    //All lists are concatenated together, the most we can have is bounded by (MAX_LEN * MAX_LEN+1) / 2
    //independent regions, each one vital for at most 4 pla groups, add some extra just in case.
    const maxRegions: usize = ((Board::MAX_LEN * Board::MAX_LEN + 1) / 2 + 1) as usize;
    const vitalForPlaHeadsListsMaxLen: usize = maxRegions * 4;
    let mut vital_for_pla_heads_lists: [Loc; vitalForPlaHeadsListsMaxLen] = [0; vitalForPlaHeadsListsMaxLen];
    let mut vital_for_pla_heads_lists_total: i32 = 0;

    //A list of region heads
    let mut num_regions: i32 = 0;
    let mut region_heads: [Loc; maxRegions] = [0; maxRegions];
    //Start indices and list lengths in vitalForPlaHeadsLists
    let mut vital_start: [u16; maxRegions] = [0; maxRegions];
    let mut vital_len: [u16; maxRegions] = [0; maxRegions];
    //For each region, are there 0, 1, or 2+ spaces of that region not bordering any pla?
    let mut num_internal_spaces_max2: [u8; maxRegions] = [0; maxRegions];
    let mut contains_opp: [bool; maxRegions] = [false; maxRegions];

    //Breadth-first-search trace maximal non-pla regions of the board and record their properties and join them into a
    //linked list through nextEmptyOrOpp.
    //Takes as input the location serving as the head, the tip node of the linked list so far, the next loc, and the
    //numeric index of the region
    //Returns the loc serving as the current tip node ("tailTarget") of the linked list.

    let mut build_region_queue: [Loc; Board::MAX_ARR_SIZE as usize] = [0; Board::MAX_ARR_SIZE as usize];

    // PORT NOTE: the `[&]` lambda takes the arrays it captures by reference as parameters,
    // because the enclosing loop also writes them between calls (Rust forbids the aliasing).
    let build_region = |initial_loc: Loc,
                        region_idx: i32,
                        region_idx_by_loc: &mut [i16; Board::MAX_ARR_SIZE as usize],
                        vital_for_pla_heads_lists: &mut [Loc; vitalForPlaHeadsListsMaxLen],
                        vital_start: &[u16; maxRegions],
                        vital_len: &mut [u16; maxRegions],
                        num_internal_spaces_max2: &mut [u8; maxRegions],
                        contains_opp: &mut [bool; maxRegions],
                        build_region_queue: &mut [Loc; Board::MAX_ARR_SIZE as usize],
                        next_empty_or_opp: &mut [Loc; Board::MAX_ARR_SIZE as usize]|
     -> Loc {
      //Commented out - we don't check if this region is already built or not, but carefully rely on the caller to ensure it is not.
      //if(regionIdxByLoc[initialLoc] != -1)
      //  return tailTarget;

      //This code use a queue to build regions. We use the initial provided location as the beginning of the tail of the linkedlist
      let mut tail_target: Loc = initial_loc;

      let ri = region_idx as usize;
      let mut is_vlen_non_zero: bool = vital_len[ri] > 0;
      let mut build_region_queue_head: i32 = 0;
      let mut build_region_queue_tail: i32 = 1;
      build_region_queue[0] = initial_loc;
      region_idx_by_loc[initial_loc as usize] = region_idx as i16;

      while build_region_queue_head != build_region_queue_tail {
        //Pop next location off queue
        let loc: Loc = build_region_queue[build_region_queue_head as usize];
        build_region_queue_head += 1;

        //First, filter out any pla heads it turns out we're not vital for because we're not adjacent to them
        //In the case where suicide is disallowed, we only do this filtering on intersections that are actually empty
        {
          if is_vlen_non_zero && (is_multi_stone_suicide_legal || self.colors[loc as usize] == C_EMPTY) {
            let v_start: u16 = vital_start[ri];
            let old_v_len: u16 = vital_len[ri];
            let mut new_v_len: u16 = 0;
            for i in 0..old_v_len {
              if self.is_adjacent_to_pla_head(pla, loc, vital_for_pla_heads_lists[(v_start as i32 + i as i32) as usize]) {
                vital_for_pla_heads_lists[(v_start as i32 + new_v_len as i32) as usize] = vital_for_pla_heads_lists[(v_start as i32 + i as i32) as usize];
                new_v_len += 1;
              }
            }
            vital_len[ri] = new_v_len;
            is_vlen_non_zero = new_v_len > 0;
          }
        }

        //Determine if this point is internal, unless we already have many internal points
        if num_internal_spaces_max2[ri] < 2 && !self.is_adjacent_to_pla(loc, pla) {
          num_internal_spaces_max2[ri] += 1;
        }

        if self.colors[loc as usize] == opp {
          contains_opp[ri] = true;
        }

        next_empty_or_opp[loc as usize] = tail_target;
        tail_target = loc;

        //Push adjacent locations on to queue.
        for adjoffset in [-(self.x_size + 1), -1, 1, self.x_size + 1] {
          let adj: Loc = (loc as i32 + adjoffset) as Loc;
          if (self.colors[adj as usize] == C_EMPTY || self.colors[adj as usize] == opp) && region_idx_by_loc[adj as usize] == -1 {
            build_region_queue[build_region_queue_tail as usize] = adj;
            build_region_queue_tail += 1;
            region_idx_by_loc[adj as usize] = region_idx as i16;
          }
        }
      }

      debug_assert!(build_region_queue_tail < Board::MAX_ARR_SIZE);
      tail_target
    };

    let mut at_least_one_pla = false;
    for y in 0..self.y_size {
      for x in 0..self.x_size {
        let loc: Loc = Location::get_loc(x, y, self.x_size);
        if region_idx_by_loc[loc as usize] != -1 {
          continue;
        }
        if self.colors[loc as usize] != C_EMPTY {
          at_least_one_pla |= self.colors[loc as usize] == pla;
          continue;
        }
        let region_idx: i16 = num_regions as i16;
        num_regions += 1;
        debug_assert!(num_regions as usize <= maxRegions);
        let ri = region_idx as usize;

        //Initialize region metadata
        region_heads[ri] = loc; //Use loc itself as the head
        vital_start[ri] = vital_for_pla_heads_lists_total as u16;
        vital_len[ri] = 0;
        num_internal_spaces_max2[ri] = 0;
        contains_opp[ri] = false;

        //Fill in all adjacent pla heads as vital, which will get filtered during buildRegion
        {
          let v_start: u16 = vital_start[ri];
          debug_assert!(v_start as usize + 4 <= vitalForPlaHeadsListsMaxLen);
          let mut initial_v_len: u16 = 0;
          for i in 0..4 {
            let adj: Loc = (loc as i32 + self.adj_offsets[i] as i32) as Loc;
            if self.colors[adj as usize] == pla {
              let pla_head: Loc = self.chain_head[adj as usize];
              let mut already_present = false;
              for j in 0..initial_v_len as i32 {
                if vital_for_pla_heads_lists[(v_start as i32 + j) as usize] == pla_head {
                  already_present = true;
                  break;
                }
              }
              if !already_present {
                vital_for_pla_heads_lists[(v_start as i32 + initial_v_len as i32) as usize] = pla_head;
                initial_v_len += 1;
              }
            }
          }
          vital_len[ri] = initial_v_len;
        }
        let tail_target: Loc = build_region(
          loc,
          region_idx as i32,
          &mut region_idx_by_loc,
          &mut vital_for_pla_heads_lists,
          &vital_start,
          &mut vital_len,
          &mut num_internal_spaces_max2,
          &mut contains_opp,
          &mut build_region_queue,
          &mut next_empty_or_opp,
        ); //buildRegion uses loc itself as the head
        next_empty_or_opp[loc as usize] = tail_target; //Close the circular linked list - loc points to buildRegion's tailTarget

        vital_for_pla_heads_lists_total += vital_len[ri] as i32;

        // for(int k = 0; k<vitalLen[regionIdx]; k++)
        //   cout << Location::toString(head,x_size) << "is vital for" << Location::toString(vitalForPlaHeadsLists[vitalStart[regionIdx]+k],x_size) << endl;
      }
    }

    //Also accumulate all player heads
    let mut num_pla_heads: i32 = 0;
    let mut all_pla_heads: [Loc; Board::MAX_PLAY_SIZE as usize] = [0; Board::MAX_PLAY_SIZE as usize];
    for loc in 0..Board::MAX_ARR_SIZE as Loc {
      if self.colors[loc as usize] == pla && self.chain_head[loc as usize] == loc {
        all_pla_heads[num_pla_heads as usize] = loc;
        num_pla_heads += 1;
      }
    }

    let mut pla_has_been_killed: [bool; Board::MAX_PLAY_SIZE as usize] = [false; Board::MAX_PLAY_SIZE as usize];
    pla_has_been_killed[..num_pla_heads as usize].fill(false);

    //Zero out vital liberties by head
    let mut vital_count_by_pla_head: [u16; Board::MAX_ARR_SIZE as usize] = [0; Board::MAX_ARR_SIZE as usize];
    for i in 0..num_pla_heads {
      vital_count_by_pla_head[all_pla_heads[i as usize] as usize] = 0;
    }

    //Walk all regions and accumulate a vital liberty to each pla it is vital for.
    for i in 0..num_regions {
      let v_start: i32 = vital_start[i as usize] as i32;
      let v_len: i32 = vital_len[i as usize] as i32;
      for j in 0..v_len {
        let pla_head: Loc = vital_for_pla_heads_lists[(v_start + j) as usize];
        vital_count_by_pla_head[pla_head as usize] += 1;
      }
    }

    //Now, we can begin the benson iteration
    loop {
      //Walk all player heads and kill them if they haven't accumulated at least 2 vital liberties
      let mut killed_anything = false;
      for i in 0..num_pla_heads {
        //Already killed - skip
        if pla_has_been_killed[i as usize] {
          continue;
        }

        let pla_head: Loc = all_pla_heads[i as usize];
        if vital_count_by_pla_head[pla_head as usize] < 2 {
          pla_has_been_killed[i as usize] = true;
          killed_anything = true;
          //Walk the pla chain to update bordering regions
          let mut cur: Loc = pla_head;
          loop {
            for j in 0..4 {
              let adj: Loc = (cur as i32 + self.adj_offsets[j] as i32) as Loc;
              let region_idx: i16 = region_idx_by_loc[adj as usize];
              //Mark regions as no longer vital
              if region_idx >= 0
                && !borders_non_pass_alive_pla_by_head[region_heads[region_idx as usize] as usize]
                && (self.colors[adj as usize] == C_EMPTY || self.colors[adj as usize] == opp)
              {
                borders_non_pass_alive_pla_by_head[region_heads[region_idx as usize] as usize] = true;
                //Decrement vitality for all pla chains that it was vital for.
                let v_start: i32 = vital_start[region_idx as usize] as i32;
                let v_len: i32 = vital_len[region_idx as usize] as i32;
                for k in 0..v_len {
                  let pla_h: Loc = vital_for_pla_heads_lists[(v_start + k) as usize];
                  vital_count_by_pla_head[pla_h as usize] -= 1;
                }
              }
            }
            cur = self.next_in_chain[cur as usize];
            if !(cur != pla_head) {
              break;
            }
          }
        }
      }
      if !killed_anything {
        break;
      }
    }

    //Debug - Make sure nothing overflowed
    // for(int i = 0; i<numPlaHeads; i++)
    //  assert(vitalCountByPlaHead[allPlaHeads[i]] >= 0 && vitalCountByPlaHead[allPlaHeads[i]] < 1000);

    //Mark result with pass-alive groups
    for i in 0..num_pla_heads {
      if !pla_has_been_killed[i as usize] {
        let pla_head: Loc = all_pla_heads[i as usize];
        let mut cur: Loc = pla_head;
        loop {
          result[cur as usize] = pla;
          cur = self.next_in_chain[cur as usize];
          if !(cur != pla_head) {
            break;
          }
        }
      }
    }

    //Mark result with territory
    for i in 0..num_regions {
      let head: Loc = region_heads[i as usize];

      //Mark pass alive territory and pass-alive stones and large empty regions bordered only own pass-alive stones unconditionally
      //These should be mutually exclusive with these same regions but for the opponent, so this is safe.
      //We need to mark unconditionally since we WILL sometimes overwrite points of the opponent's color marked earlier, in the
      //case that the opponent was marking unsafeBigTerritories and marked an empty spot surrounded by a pass-dead group.
      let mut should_mark: bool =
        num_internal_spaces_max2[i as usize] <= 1 && !borders_non_pass_alive_pla_by_head[head as usize] && at_least_one_pla; //internal space can be 2
      should_mark = should_mark
        || (safe_big_territories && !contains_opp[i as usize] && !borders_non_pass_alive_pla_by_head[head as usize] && at_least_one_pla);
      if should_mark {
        let mut cur: Loc = head;
        loop {
          result[cur as usize] = pla;
          cur = next_empty_or_opp[cur as usize];
          if !(cur != head) {
            break;
          }
        }
      } else {
        //Mark unsafeBigTerritories only if the opponent didn't already claim the very stones we're using to surround it as
        //pass-dead and therefore the whole thing as pass-alive-territory.
        let should_mark_if_empty: bool = unsafe_big_territories && !contains_opp[i as usize] && at_least_one_pla;
        if should_mark_if_empty {
          let mut cur: Loc = head;
          loop {
            if result[cur as usize] == C_EMPTY {
              result[cur as usize] = pla;
            }
            cur = next_empty_or_opp[cur as usize];
            if !(cur != head) {
              break;
            }
          }
        }
      }
    }
  }

  /// upstream: Board::calculateIndependentLifeAreaHelper (board.cpp)
  pub(crate) fn calculate_independent_life_area_helper(
    &self,
    basic_area: &[Color],
    result: &mut [Color],
    white_minus_black_independent_life_region_count: &mut i32,
  ) {
    let mut queue: [Loc; Board::MAX_ARR_SIZE as usize] = [0; Board::MAX_ARR_SIZE as usize];
    *white_minus_black_independent_life_region_count = 0;

    //Iterate through all the regions that players own via area scoring and mark
    //all the ones that are touching dame OR that contain an atari stone
    let mut is_seki: [bool; Board::MAX_ARR_SIZE as usize] = [false; Board::MAX_ARR_SIZE as usize];
    for i in 0..Board::MAX_ARR_SIZE {
      is_seki[i as usize] = false;
    }

    let mut queue_head: i32 = 0;
    let mut queue_tail: i32 = 0;

    let adj0: i32 = -(self.x_size + 1);
    let adj1: i32 = -1;
    let adj2: i32 = 1;
    let adj3: i32 = self.x_size + 1;

    for y in 0..self.y_size {
      for x in 0..self.x_size {
        let loc: Loc = Location::get_loc(x, y, self.x_size);
        let l: i32 = loc as i32;
        if basic_area[loc as usize] != C_EMPTY && !is_seki[loc as usize] {
          if
          //Stone of player owning the area is in atari? Treat as seki.
          (self.colors[loc as usize] == basic_area[loc as usize] && self.get_num_liberties(loc) == 1)
            ||
          //Touches dame? Treat as seki
          ((self.colors[(l + adj0) as usize] == C_EMPTY && basic_area[(l + adj0) as usize] == C_EMPTY)
            || (self.colors[(l + adj1) as usize] == C_EMPTY && basic_area[(l + adj1) as usize] == C_EMPTY)
            || (self.colors[(l + adj2) as usize] == C_EMPTY && basic_area[(l + adj2) as usize] == C_EMPTY)
            || (self.colors[(l + adj3) as usize] == C_EMPTY && basic_area[(l + adj3) as usize] == C_EMPTY))
          {
            let pla: Player = basic_area[loc as usize];
            is_seki[loc as usize] = true;
            queue[queue_tail as usize] = loc;
            queue_tail += 1;
            while queue_head != queue_tail {
              //Pop next location off queue
              let next_loc: Loc = queue[queue_head as usize];
              queue_head += 1;

              //Look all around it, floodfill
              for adjoffset in [-(self.x_size + 1), -1, 1, self.x_size + 1] {
                let adj: Loc = (next_loc as i32 + adjoffset) as Loc;
                if basic_area[adj as usize] == pla && !is_seki[adj as usize] {
                  is_seki[adj as usize] = true;
                  queue[queue_tail as usize] = adj;
                  queue_tail += 1;
                }
              }
            }
          }
        }
      }
    }

    queue_head = 0;
    queue_tail = 0;

    //Now, walk through and copy all non-seki-touching basic areas into the result counting
    //how many there are.
    for y in 0..self.y_size {
      for x in 0..self.x_size {
        let loc: Loc = Location::get_loc(x, y, self.x_size);
        if basic_area[loc as usize] != C_EMPTY && !is_seki[loc as usize] && result[loc as usize] != basic_area[loc as usize] {
          let pla: Player = basic_area[loc as usize];
          *white_minus_black_independent_life_region_count += if pla == P_WHITE { 1 } else { -1 };
          result[loc as usize] = basic_area[loc as usize];
          queue[queue_tail as usize] = loc;
          queue_tail += 1;
          while queue_head != queue_tail {
            //Pop next location off queue
            let next_loc: Loc = queue[queue_head as usize];
            queue_head += 1;

            //Look all around it, floodfill
            for adjoffset in [-(self.x_size + 1), -1, 1, self.x_size + 1] {
              let adj: Loc = (next_loc as i32 + adjoffset) as Loc;
              if basic_area[adj as usize] == pla && result[adj as usize] != basic_area[adj as usize] {
                result[adj as usize] = basic_area[adj as usize];
                queue[queue_tail as usize] = adj;
                queue_tail += 1;
              }
            }
          }
        }
      }
    }
  }

  /// upstream: Board::checkConsistency (board.cpp)
  //Run some basic sanity checks on the board state, throws an exception if not consistent, for testing/debugging
  pub fn check_consistency(&self) -> Result<(), KataError> {
    let err_label: String = String::from("Board::checkConsistency(): ");

    let mut chain_loc_checked: [bool; Board::MAX_ARR_SIZE as usize] = [false; Board::MAX_ARR_SIZE as usize];
    for i in 0..Board::MAX_ARR_SIZE {
      chain_loc_checked[i as usize] = false;
    }

    let mut buf: Vec<Loc> = Vec::new();
    // PORT NOTE: the `[&buf,errLabel,&chainLocChecked,this]` lambda takes buf and chainLocChecked
    // as parameters, because the enclosing loop also reads chainLocChecked between calls.
    let check_chain_consistency = |loc: Loc, buf: &mut Vec<Loc>, chain_loc_checked: &mut [bool; Board::MAX_ARR_SIZE as usize]| -> Result<(), KataError> {
      let pla: Player = self.colors[loc as usize];
      let head: Loc = self.chain_head[loc as usize];
      let mut cur: Loc = loc;
      let mut stone_count: i32 = 0;
      let mut pseudo_libs: i32 = 0;
      let mut found_chain_head = false;
      loop {
        chain_loc_checked[cur as usize] = true;

        if self.colors[cur as usize] != pla {
          return Err(KataError::StringError(err_label.clone() + "Chain is not all the same color"));
        }
        if self.chain_head[cur as usize] != head {
          return Err(KataError::StringError(err_label.clone() + "Chain does not all have the same head"));
        }

        stone_count += 1;
        pseudo_libs += self.get_num_immediate_liberties(cur);
        if cur == head {
          found_chain_head = true;
        }

        if stone_count > Board::MAX_PLAY_SIZE {
          return Err(KataError::StringError(err_label.clone() + "Chain exceeds size of board - broken circular list?"));
        }
        cur = self.next_in_chain[cur as usize];

        if cur < 0 || (cur as i32) >= Board::MAX_ARR_SIZE {
          return Err(KataError::StringError(err_label.clone() + "Chain location is outside of board bounds, data corruption?"));
        }

        if !(cur != loc) {
          break;
        }
      }

      if !found_chain_head {
        return Err(KataError::StringError(err_label.clone() + "Chain loop does not contain head"));
      }

      let data: &ChainData = &self.chain_data[head as usize];
      if data.owner != pla {
        return Err(KataError::StringError(err_label.clone() + "Chain data owner does not match stones"));
      }
      if data.num_locs as i32 != stone_count {
        return Err(KataError::StringError(err_label.clone() + "Chain data num_locs does not match actual stone count"));
      }
      if data.num_liberties as i32 > pseudo_libs {
        return Err(KataError::StringError(err_label.clone() + "Chain data liberties exceeds pseudoliberties"));
      }
      if data.num_liberties <= 0 {
        return Err(KataError::StringError(err_label.clone() + "Chain data liberties is nonpositive"));
      }

      let num_found_libs: i32 = self.find_liberties(loc, buf, 0, 0);
      if num_found_libs != data.num_liberties as i32 {
        return Err(KataError::StringError(err_label.clone() + "FindLiberties found a different number of libs"));
      }
      Ok(())
    };

    let st = Board::statics();
    let mut tmp_pos_hash: Hash128 = st.ZOBRIST_SIZE_X_HASH[self.x_size as usize] ^ st.ZOBRIST_SIZE_Y_HASH[self.y_size as usize];
    // int emptyCount = 0;
    for loc in 0..Board::MAX_ARR_SIZE as Loc {
      let x: i32 = Location::get_x(loc, self.x_size);
      let y: i32 = Location::get_y(loc, self.x_size);
      if x < 0 || x >= self.x_size || y < 0 || y >= self.y_size {
        if self.colors[loc as usize] != C_WALL {
          return Err(KataError::StringError(err_label.clone() + "Non-WALL value outside of board legal area"));
        }
      } else {
        if self.colors[loc as usize] == C_BLACK || self.colors[loc as usize] == C_WHITE {
          if !chain_loc_checked[loc as usize] {
            check_chain_consistency(loc, &mut buf, &mut chain_loc_checked)?;
          }
          // if(empty_list.contains(loc))
          //   throw StringError(errLabel + "Empty list contains filled location");

          tmp_pos_hash ^= st.ZOBRIST_BOARD_HASH[loc as usize][self.colors[loc as usize] as usize];
          tmp_pos_hash ^= st.ZOBRIST_BOARD_HASH[loc as usize][C_EMPTY as usize];
        } else if self.colors[loc as usize] == C_EMPTY {
          // if(!empty_list.contains(loc))
          //   throw StringError(errLabel + "Empty list doesn't contain empty location");
          // emptyCount += 1;
        } else {
          return Err(KataError::StringError(err_label.clone() + "Non-(black,white,empty) value within board legal area"));
        }
      }
    }

    if self.pos_hash != tmp_pos_hash {
      return Err(KataError::StringError(err_label.clone() + "Pos hash does not match expected"));
    }

    // if(empty_list.size_ != emptyCount)
    //   throw StringError(errLabel + "Empty list size is not the number of empty points");
    // for(int i = 0; i<emptyCount; i++) {
    //   Loc loc = empty_list.list_[i];
    //   int x = Location::getX(loc,x_size);
    //   int y = Location::getY(loc,x_size);
    //   if(x < 0 || x >= x_size || y < 0 || y >= y_size)
    //     throw StringError(errLabel + "Invalid empty list loc");
    //   if(empty_list.indices_[loc] != i)
    //     throw StringError(errLabel + "Empty list index for loc in index i is not i");
    // }

    if self.ko_loc != Board::NULL_LOC {
      let x: i32 = Location::get_x(self.ko_loc, self.x_size);
      let y: i32 = Location::get_y(self.ko_loc, self.x_size);
      if x < 0 || x >= self.x_size || y < 0 || y >= self.y_size {
        return Err(KataError::StringError(err_label.clone() + "Invalid simple ko loc"));
      }
      if self.get_num_immediate_liberties(self.ko_loc) != 0 {
        return Err(KataError::StringError(err_label.clone() + "Simple ko loc has immediate liberties"));
      }
    }

    let mut tmp_adj_offsets: [i16; 8] = [0; 8];
    Location::get_adjacent_offsets(&mut tmp_adj_offsets, self.x_size);
    for i in 0..8 {
      if tmp_adj_offsets[i] != self.adj_offsets[i] {
        return Err(KataError::StringError(err_label.clone() + "Corrupted adj_offsets array"));
      }
    }
    Ok(())
  }

  /// upstream: Board::isEqualForTesting (board.cpp)
  //For the moment, only used in testing since it does extra consistency checks.
  //If we need a version to be used in "prod", we could make an efficient version maybe as operator==.
  pub fn is_equal_for_testing(&self, other: &Board, check_num_captures: bool, check_simple_ko: bool) -> Result<bool, KataError> {
    self.check_consistency()?;
    other.check_consistency()?;
    if self.x_size != other.x_size {
      return Ok(false);
    }
    if self.y_size != other.y_size {
      return Ok(false);
    }
    if check_simple_ko && self.ko_loc != other.ko_loc {
      return Ok(false);
    }
    if check_num_captures && self.num_black_captures != other.num_black_captures {
      return Ok(false);
    }
    if check_num_captures && self.num_white_captures != other.num_white_captures {
      return Ok(false);
    }
    if self.pos_hash != other.pos_hash {
      return Ok(false);
    }
    for i in 0..Board::MAX_ARR_SIZE as usize {
      if self.colors[i] != other.colors[i] {
        return Ok(false);
      }
    }
    //We don't require that the chain linked lists are in the same order.
    //Consistency check ensures that all the linked lists are consistent with colors array, which we checked.
    Ok(true)
  }
}

//IO FUNCS------------------------------------------------------------------------------------------

impl PlayerIO {
  /// upstream: PlayerIO::colorToChar (board.cpp)
  pub fn color_to_char(c: Color) -> char {
    match c {
      C_BLACK => 'X',
      C_WHITE => 'O',
      C_EMPTY => '.',
      _ => '#',
    }
  }

  /// upstream: PlayerIO::playerToString (board.cpp)
  pub fn player_to_string(c: Color) -> String {
    match c {
      C_BLACK => String::from("Black"),
      C_WHITE => String::from("White"),
      C_EMPTY => String::from("Empty"),
      _ => String::from("Wall"),
    }
  }

  /// upstream: PlayerIO::playerToStringShort (board.cpp)
  pub fn player_to_string_short(c: Color) -> String {
    match c {
      C_BLACK => String::from("B"),
      C_WHITE => String::from("W"),
      C_EMPTY => String::from("E"),
      _ => String::from(""),
    }
  }

  /// upstream: PlayerIO::tryParsePlayer (board.cpp)
  pub fn try_parse_player(s: &str, pla: &mut Player) -> bool {
    let str: String = Global::to_lower(s);
    if str == "black" || str == "b" {
      *pla = P_BLACK;
      return true;
    } else if str == "white" || str == "w" {
      *pla = P_WHITE;
      return true;
    }
    false
  }

  /// upstream: PlayerIO::parsePlayer (board.cpp)
  pub fn parse_player(s: &str) -> Result<Player, KataError> {
    let mut pla: Player = C_EMPTY;
    let suc = PlayerIO::try_parse_player(s, &mut pla);
    if !suc {
      return Err(KataError::StringError(String::from("Could not parse player: ") + s));
    }
    Ok(pla)
  }
}

impl Location {
  /// upstream: Location::toStringMach (board.cpp)
  /// `toStringMach(Loc loc, int x_size)` (declared first, keeps the base name).
  pub fn to_string_mach(loc: Loc, x_size: i32) -> String {
    if loc == Board::PASS_LOC {
      return String::from("pass");
    }
    if loc == Board::NULL_LOC {
      return String::from("null");
    }
    strprintf!("(%d,%d)", Location::get_x(loc, x_size), Location::get_y(loc, x_size))
  }

  /// upstream: Location::toString (board.cpp)
  /// `toString(Loc loc, int x_size, int y_size)` (overload rule 3).
  pub fn to_string_x_size_y_size(loc: Loc, x_size: i32, y_size: i32) -> String {
    if x_size > 25 * 25 {
      return Location::to_string_mach(loc, x_size);
    }
    if loc == Board::PASS_LOC {
      return String::from("pass");
    }
    if loc == Board::NULL_LOC {
      return String::from("null");
    }
    let x_char: &[u8] = b"ABCDEFGHJKLMNOPQRSTUVWXYZ";
    let x: i32 = Location::get_x(loc, x_size);
    let y: i32 = Location::get_y(loc, x_size);
    if x >= x_size || x < 0 || y < 0 || y >= y_size {
      return Location::to_string_mach(loc, x_size);
    }

    if x <= 24 {
      strprintf!("%c%d", x_char[x as usize], y_size - y)
    } else {
      strprintf!("%c%c%d", x_char[(x / 25 - 1) as usize], x_char[(x % 25) as usize], y_size - y)
    }
  }

  /// upstream: Location::toString (board.cpp)
  pub fn to_string(loc: Loc, b: &Board) -> String {
    Location::to_string_x_size_y_size(loc, b.x_size, b.y_size)
  }

  /// upstream: Location::toStringMach (board.cpp)
  /// `toStringMach(Loc loc, const Board& b)` (overload rule 3: same param count, declared second).
  pub fn to_string_mach_b(loc: Loc, b: &Board) -> String {
    Location::to_string_mach(loc, b.x_size)
  }
}

/// upstream: tryParseLetterCoordinate (board.cpp) static
fn try_parse_letter_coordinate(c: u8, x: &mut i32) -> bool {
  if c >= b'A' && c <= b'H' {
    *x = (c - b'A') as i32;
  } else if c >= b'a' && c <= b'h' {
    *x = (c - b'a') as i32;
  } else if c >= b'J' && c <= b'Z' {
    *x = c as i32 - b'A' as i32 - 1;
  } else if c >= b'j' && c <= b'z' {
    *x = c as i32 - b'a' as i32 - 1;
  } else {
    return false;
  }
  true
}

impl Location {
  /// upstream: Location::tryOfString (board.cpp)
  /// `tryOfString(const std::string& str, int x_size, int y_size, Loc& result)` (overload rule 3).
  pub fn try_of_string_x_size_y_size(str: &str, x_size: i32, y_size: i32, result: &mut Loc) -> bool {
    let mut s: String = Global::trim(str, Global::TRIM_DEFAULT_DELIM);
    if s.len() < 2 {
      return false;
    }
    if Global::is_equal_case_insensitive(&s, "pass") || Global::is_equal_case_insensitive(&s, "pss") {
      *result = Board::PASS_LOC;
      return true;
    }
    if s.as_bytes()[0] == b'(' {
      if s.as_bytes()[s.len() - 1] != b')' {
        return false;
      }
      s = s[1..1 + (s.len() - 2)].to_string();
      let pieces: Vec<String> = Global::split_delim(&s, ',');
      if pieces.len() != 2 {
        return false;
      }
      let mut x: i32 = 0;
      let mut y: i32 = 0;
      let suc_x = Global::try_string_to_int(&pieces[0], &mut x);
      let suc_y = Global::try_string_to_int(&pieces[1], &mut y);
      if !suc_x || !suc_y {
        return false;
      }
      *result = Location::get_loc(x, y, x_size);
      true
    } else {
      let mut x: i32 = 0;
      if !try_parse_letter_coordinate(s.as_bytes()[0], &mut x) {
        return false;
      }

      //Extended format
      let s1 = s.as_bytes()[1];
      if (s1 >= b'A' && s1 <= b'Z') || (s1 >= b'a' && s1 <= b'z') {
        let mut x1: i32 = 0;
        if !try_parse_letter_coordinate(s1, &mut x1) {
          return false;
        }
        x = (x + 1) * 25 + x1;
        s = s[2..2 + (s.len() - 2)].to_string();
      } else {
        s = s[1..1 + (s.len() - 1)].to_string();
      }

      let mut y: i32 = 0;
      let suc_y = Global::try_string_to_int(&s, &mut y);
      if !suc_y {
        return false;
      }
      // PORT NOTE: C++ signed overflow is UB, reachable from user input ("A-2147483648");
      // wrap like x86-64 instead of panicking.
      y = y_size.wrapping_sub(y);
      if x < 0 || y < 0 || x >= x_size || y >= y_size {
        return false;
      }
      *result = Location::get_loc(x, y, x_size);
      true
    }
  }

  /// upstream: Location::tryOfStringAllowNull (board.cpp)
  //Same, but will parse "null" as Board::NULL_LOC
  pub fn try_of_string_allow_null_x_size_y_size(str: &str, x_size: i32, y_size: i32, result: &mut Loc) -> bool {
    if str == "null" {
      *result = Board::NULL_LOC;
      return true;
    }
    Location::try_of_string_x_size_y_size(str, x_size, y_size, result)
  }

  /// upstream: Location::tryOfString (board.cpp)
  pub fn try_of_string(str: &str, b: &Board, result: &mut Loc) -> bool {
    Location::try_of_string_x_size_y_size(str, b.x_size, b.y_size, result)
  }

  /// upstream: Location::tryOfStringAllowNull (board.cpp)
  pub fn try_of_string_allow_null(str: &str, b: &Board, result: &mut Loc) -> bool {
    Location::try_of_string_allow_null_x_size_y_size(str, b.x_size, b.y_size, result)
  }

  /// upstream: Location::ofString (board.cpp)
  pub fn of_string_x_size_y_size(str: &str, x_size: i32, y_size: i32) -> Result<Loc, KataError> {
    let mut result: Loc = 0;
    if Location::try_of_string_x_size_y_size(str, x_size, y_size, &mut result) {
      return Ok(result);
    }
    Err(KataError::StringError(String::from("Could not parse board location: ") + str))
  }

  /// upstream: Location::ofStringAllowNull (board.cpp)
  pub fn of_string_allow_null_x_size_y_size(str: &str, x_size: i32, y_size: i32) -> Result<Loc, KataError> {
    let mut result: Loc = 0;
    if Location::try_of_string_allow_null_x_size_y_size(str, x_size, y_size, &mut result) {
      return Ok(result);
    }
    Err(KataError::StringError(String::from("Could not parse board location: ") + str))
  }

  /// upstream: Location::ofString (board.cpp)
  pub fn of_string(str: &str, b: &Board) -> Result<Loc, KataError> {
    Location::of_string_x_size_y_size(str, b.x_size, b.y_size)
  }

  /// upstream: Location::ofStringAllowNull (board.cpp)
  pub fn of_string_allow_null(str: &str, b: &Board) -> Result<Loc, KataError> {
    Location::of_string_allow_null_x_size_y_size(str, b.x_size, b.y_size)
  }

  /// upstream: Location::parseSequence (board.cpp)
  pub fn parse_sequence(str: &str, board: &Board) -> Result<Vec<Loc>, KataError> {
    let pieces: Vec<String> = Global::split_delim(&Global::trim(str, Global::TRIM_DEFAULT_DELIM), ' ');
    let mut locs: Vec<Loc> = Vec::new();
    for i in 0..pieces.len() {
      let piece: String = Global::trim(&pieces[i], Global::TRIM_DEFAULT_DELIM);
      if piece.is_empty() {
        continue;
      }
      locs.push(Location::of_string(&piece, board)?);
    }
    Ok(locs)
  }
}

impl Board {
  /// upstream: Board::printBoard (board.cpp)
  pub fn print_board(out: &mut dyn std::fmt::Write, board: &Board, mark_loc: Loc, hist: Option<&[Move]>) {
    if let Some(hist) = hist {
      write!(out, "MoveNum: {} ", hist.len()).unwrap();
    }
    write!(out, "HASH: {}\n", board.pos_hash).unwrap();
    let show_coords: bool = board.x_size <= 50 && board.y_size <= 50;
    if show_coords {
      let x_char: &[u8] = b"ABCDEFGHJKLMNOPQRSTUVWXYZ";
      write!(out, "  ").unwrap();
      for x in 0..board.x_size {
        if x <= 24 {
          write!(out, " ").unwrap();
          write!(out, "{}", x_char[x as usize] as char).unwrap();
        } else {
          write!(out, "A{}", x_char[(x - 25) as usize] as char).unwrap();
        }
      }
      write!(out, "\n").unwrap();
    }

    for y in 0..board.y_size {
      if show_coords {
        let buf: String = strprintf!("%2d", board.y_size - y);
        write!(out, "{} ", buf).unwrap();
      }
      for x in 0..board.x_size {
        let loc: Loc = Location::get_loc(x, y, board.x_size);
        let s: char = PlayerIO::color_to_char(board.colors[loc as usize]);
        if board.colors[loc as usize] == C_EMPTY && mark_loc == loc {
          write!(out, "@").unwrap();
        } else {
          write!(out, "{}", s).unwrap();
        }

        let mut hist_marked = false;
        if let Some(hist) = hist {
          let start: usize = if hist.len() >= 3 { hist.len() - 3 } else { 0 };
          let mut i: usize = 0;
          while start + i < hist.len() {
            if hist[start + i].loc == loc {
              write!(out, "{}", 1 + i).unwrap();
              hist_marked = true;
              break;
            }
            i += 1;
          }
        }

        if x < board.x_size - 1 && !hist_marked {
          write!(out, " ").unwrap();
        }
      }
      write!(out, "\n").unwrap();
    }
    write!(out, "\n").unwrap();
  }

  /// upstream: Board::toStringSimple (board.cpp)
  pub fn to_string_simple(board: &Board, line_delimiter: char) -> String {
    let mut s = String::new();
    for y in 0..board.y_size {
      for x in 0..board.x_size {
        let loc: Loc = Location::get_loc(x, y, board.x_size);
        s.push(PlayerIO::color_to_char(board.colors[loc as usize]));
      }
      s.push(line_delimiter);
    }
    s
  }

  /// upstream: Board::parseBoard (board.cpp)
  pub fn parse_board(x_size: i32, y_size: i32, s: &str, line_delimiter: char) -> Result<Board, KataError> {
    let mut board = Board::new_x_y(x_size, y_size)?;
    let mut lines: Vec<String> = Global::split_delim(&Global::trim(s, Global::TRIM_DEFAULT_DELIM), line_delimiter);

    //Throw away coordinate labels line if it exists
    if lines.len() == (y_size + 1) as usize && Global::is_prefix(&lines[0], "A") {
      lines.remove(0);
    }

    if lines.len() != y_size as usize {
      return Err(KataError::StringError(String::from(
        "Board::parseBoard - string has different number of board rows than ySize",
      )));
    }

    for y in 0..y_size {
      let mut line: String = Global::trim(&lines[y as usize], Global::TRIM_DEFAULT_DELIM);
      //Throw away coordinates if they exist
      let mut first_non_digit_idx: usize = 0;
      while first_non_digit_idx < line.len() && Global::is_digit(line.as_bytes()[first_non_digit_idx]) {
        first_non_digit_idx += 1;
      }
      line.replace_range(0..first_non_digit_idx, "");
      line = Global::trim(&line, Global::TRIM_DEFAULT_DELIM);

      if line.len() != x_size as usize && line.len() != (2 * x_size - 1) as usize {
        return Err(KataError::StringError(String::from("Board::parseBoard - line length not compatible with xSize")));
      }

      for x in 0..x_size {
        let c: u8 = if line.len() == x_size as usize { line.as_bytes()[x as usize] } else { line.as_bytes()[(x * 2) as usize] };

        let loc: Loc = Location::get_loc(x, y, board.x_size);
        if c == b'.' || c == b' ' || c == b'*' || c == b',' || c == b'`' {
          continue;
        } else if c == b'o' || c == b'O' {
          let suc = board.set_stone_fail_if_no_libs(loc, P_WHITE);
          if !suc {
            return Err(KataError::StringError(
              String::from("Board::parseBoard - zero-liberty group near ") + &Location::to_string(loc, &board),
            ));
          }
        } else if c == b'x' || c == b'X' {
          let suc = board.set_stone_fail_if_no_libs(loc, P_BLACK);
          if !suc {
            return Err(KataError::StringError(
              String::from("Board::parseBoard - zero-liberty group near ") + &Location::to_string(loc, &board),
            ));
          }
        } else {
          // TODO(port): `string + c` appends the raw byte; a non-ASCII byte is pushed as a Latin-1 char here.
          return Err(KataError::StringError(String::from("Board::parseBoard - could not parse board character: ") + &(c as char).to_string()));
        }
      }
    }
    Ok(board)
  }

  // Board::toJson / Board::ofJson (board.cpp) are not ported (port-manifest.toml notes: only used by book/).

  /// upstream: Board::isAdjacentToPlaHead (board.cpp)
  pub(crate) fn is_adjacent_to_pla_head(&self, pla: Player, loc: Loc, pla_head: Loc) -> bool {
    for adjoffset in [-(self.x_size + 1), -1, 1, self.x_size + 1] {
      let adj: Loc = (loc as i32 + adjoffset) as Loc;
      if self.colors[adj as usize] == pla && self.chain_head[adj as usize] == pla_head {
        return true;
      }
    }
    false
  }

  /// upstream: Board::countEmptyHelper (board.cpp)
  //Count empty spaces in the connected empty region containing initialLoc (which must be empty)
  //not counting where emptyCounted == true, incrementing count each time and marking emptyCounted.
  //Return early and true if count > bound. Return false otherwise.
  pub(crate) fn count_empty_helper(&self, empty_counted: &mut [bool], initial_loc: Loc, count: &mut i32, bound: i32) -> bool {
    if empty_counted[initial_loc as usize] {
      return false;
    }
    *count += 1;
    empty_counted[initial_loc as usize] = true;
    if *count > bound {
      return true;
    }

    let mut num_left: i32 = 0;
    let mut num_expanded: i32 = 0;
    let mut to_expand: [i32; Board::MAX_ARR_SIZE as usize] = [0; Board::MAX_ARR_SIZE as usize];
    to_expand[num_left as usize] = initial_loc as i32;
    num_left += 1;
    while num_expanded < num_left {
      let loc: Loc = to_expand[num_expanded as usize] as Loc;
      num_expanded += 1;
      for adjoffset in [-(self.x_size + 1), -1, 1, self.x_size + 1] {
        let adj: Loc = (loc as i32 + adjoffset) as Loc;
        if self.colors[adj as usize] == C_EMPTY && !empty_counted[adj as usize] {
          *count += 1;
          empty_counted[adj as usize] = true;
          if *count > bound {
            return true;
          }
          to_expand[num_left as usize] = adj as i32;
          num_left += 1;
        }
      }
    }
    false
  }

  /// upstream: Board::simpleRepetitionBoundGt (board.cpp)
  //Returns true if, for a move just played at loc, the sum of the number of stones in loc's group and the sizes of the empty regions it touches
  //are greater than bound. See also https://senseis.xmp.net/?Cycle for some interesting test cases for thinking about this bound.
  //Returns false for passes.
  pub fn simple_repetition_bound_gt(&self, loc: Loc, bound: i32) -> bool {
    if loc == Board::NULL_LOC || loc == Board::PASS_LOC {
      return false;
    }

    let mut count: i32 = 0;

    if self.colors[loc as usize] != C_EMPTY {
      count += self.chain_data[self.chain_head[loc as usize] as usize].num_locs as i32;
      //Fast quitout
      if count + self.chain_data[self.chain_head[loc as usize] as usize].num_liberties as i32 > bound {
        return true;
      }
    }

    let mut empty_counted: [bool; Board::MAX_ARR_SIZE as usize] = [false; Board::MAX_ARR_SIZE as usize];
    empty_counted.fill(false);

    //Suicide - return just the count of the empty spaces in the region
    if self.colors[loc as usize] == C_EMPTY {
      return self.count_empty_helper(&mut empty_counted, loc, &mut count, bound);
    } else {
      let mut cur: Loc = loc;
      loop {
        for i in 0..4 {
          let lib: Loc = (cur as i32 + self.adj_offsets[i] as i32) as Loc;
          if self.colors[lib as usize] == C_EMPTY {
            if self.count_empty_helper(&mut empty_counted, lib, &mut count, bound) {
              return true;
            }
          }
        }
        cur = self.next_in_chain[cur as usize];
        if !(cur != loc) {
          break;
        }
      }
    }

    false
  }
}

/// upstream: operator<<(ostream&, const Board&) (board.cpp)
impl std::fmt::Display for Board {
  fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    let mut s = String::new();
    Board::print_board(&mut s, self, Board::NULL_LOC, None);
    out.write_str(&s)
  }
}
