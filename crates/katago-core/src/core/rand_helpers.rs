// Ported from KataGo cpp/core/rand_helpers.h @ ba938676d7f42d70950b3a535af2466fb642008c
//
// cpp/core/rand_helpers.cpp only includes the header ("ensure it compiles on its own").
// The out-of-line members of XorShift1024Mult and PCG32 (constructors, init, test) are defined
// in rand.cpp, so they live in core/rand.rs (PORTING.md §1: each .cpp gets its own impl block).

//-------------------------------------------------------------------------------------

/// upstream: XorShift1024Mult (rand_helpers.h)
//xorshift1024* from http://xoroshiro.di.unimi.it/
//Not all values should be zero
//Period = 2^1024 - 1
//Not threadsafe
#[derive(Clone, Debug)]
pub struct XorShift1024Mult {
  pub(crate) a: [u64; XorShift1024Mult::XORMULT_LEN as usize],
  pub(crate) a_idx: u64,
}

impl XorShift1024Mult {
  pub const XORMULT_LEN: i32 = 16;
  pub const XORMULT_MASK: i32 = XorShift1024Mult::XORMULT_LEN - 1;

  /// upstream: XorShift1024Mult::nextUInt (rand_helpers.h)
  #[inline]
  pub fn next_uint(&mut self) -> u32 {
    let mut a0: u64 = self.a[self.a_idx as usize];
    // PORT NOTE: `a[a_idx = (a_idx + 1) & XORMULT_MASK]`: assign first, then index (PORTING.md §5).
    self.a_idx = (self.a_idx + 1) & (XorShift1024Mult::XORMULT_MASK as u64);
    let mut a1: u64 = self.a[self.a_idx as usize];
    a1 ^= a1 << 31; // a
    a1 ^= a1 >> 11; // b
    a0 ^= a0 >> 30; // c
    self.a[self.a_idx as usize] = a0 ^ a1;
    let a_result: u64 = self.a[self.a_idx as usize].wrapping_mul(1181783497276652981i64 as u64);

    (a_result >> 32) as u32
  }
}

//-------------------------------------------------------------------------------------

/// upstream: PCG32 (rand_helpers.h)
//PCG Generator from http://www.pcg-random.org/
//Period = 2^64
//Not threadsafe
#[derive(Clone, Debug)]
pub struct PCG32 {
  pub(crate) s: u64,
}

impl PCG32 {
  /// upstream: PCG32::nextUInt (rand_helpers.h)
  #[inline]
  pub fn next_uint(&mut self) -> u32 {
    self.s = self.s.wrapping_mul(6364136223846793005u64).wrapping_add(1442695040888963407u64);
    let x: u32 = (((self.s >> 18) ^ self.s) >> 27) as u32;
    let rot: i32 = (self.s >> 59) as i32;
    if rot == 0 { x } else { (x >> rot) | (x << (32 - rot)) }
  }
}
