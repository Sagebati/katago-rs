// Ported from KataGo cpp/core/hash.cpp, cpp/core/hash.h @ ba938676d7f42d70950b3a535af2466fb642008c

/*
 * hash.h
 * Author: David Wu
 */

use crate::core::global::Global;
use crate::core::global::KataError;

/// upstream: namespace Hash (hash.h)
pub struct Hash;

impl Hash {
  //BITS-----------------------------------

  /// upstream: Hash::highBits (hash.cpp)
  //Splitting of uint64s
  pub fn high_bits(x: u64) -> u32 {
    ((x >> 32) & 0xFFFFFFFF) as u32
  }

  /// upstream: Hash::lowBits (hash.cpp)
  pub fn low_bits(x: u64) -> u32 {
    (x & 0xFFFFFFFF) as u32
  }

  /// upstream: Hash::combine (hash.cpp)
  pub fn combine(hi: u32, lo: u32) -> u64 {
    ((hi as u64) << 32) | (lo as u64)
  }

  /// upstream: Hash::basicLCong (hash.cpp)
  //A simple 64 bit linear congruential
  pub fn basic_l_cong(x: u64) -> u64 {
    2862933555777941757u64.wrapping_mul(x).wrapping_add(3037000493u64)
  }

  /// upstream: Hash::basicLCong2 (hash.cpp)
  pub fn basic_l_cong2(x: u64) -> u64 {
    6364136223846793005u64.wrapping_mul(x).wrapping_add(1442695040888963407u64)
  }

  /// upstream: Hash::murmurMix (hash.cpp)
  //MurmurHash3 finalization - good avalanche properties
  //Reversible, but maps 0 -> 0
  pub fn murmur_mix(mut x: u64) -> u64 {
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51afd7ed558ccd);
    x ^= x >> 33;
    x = x.wrapping_mul(0xc4ceb9fe1a85ec53);
    x ^= x >> 33;
    x
  }

  /// upstream: Hash::splitMix64 (hash.cpp)
  //Splitmix64 mixing step
  pub fn split_mix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
  }

  /// upstream: Hash::rrmxmx (hash.cpp)
  //algorithm from Pelle Evensen https://mostlymangling.blogspot.com/
  pub fn rrmxmx(mut x: u64) -> u64 {
    x ^= rotate_right(x, 49) ^ rotate_right(x, 24);
    x = x.wrapping_mul(0x9fb21c651e98df25);
    x ^= x >> 28;
    x = x.wrapping_mul(0x9fb21c651e98df25);
    x ^ (x >> 28)
  }

  /// upstream: Hash::nasam (hash.cpp)
  //algorithm from Pelle Evensen https://mostlymangling.blogspot.com/
  pub fn nasam(mut x: u64) -> u64 {
    x ^= rotate_right(x, 25) ^ rotate_right(x, 47);
    x = x.wrapping_mul(0x9e6c63d0676a9a99);
    x ^= (x >> 23) ^ (x >> 51);
    x = x.wrapping_mul(0x9e6d62d06f6a9a9b);
    x ^= (x >> 23) ^ (x >> 51);
    x
  }

  /// upstream: Hash::jenkinsMixSingle (hash.cpp)
  //Robert Jenkins' 96 bit Mix Function
  pub fn jenkins_mix_single(mut a: u32, mut b: u32, mut c: u32) -> u32 {
    a = a.wrapping_sub(b); a = a.wrapping_sub(c); a ^= c >> 13;
    b = b.wrapping_sub(c); b = b.wrapping_sub(a); b ^= a << 8;
    c = c.wrapping_sub(a); c = c.wrapping_sub(b); c ^= b >> 13;
    a = a.wrapping_sub(b); a = a.wrapping_sub(c); a ^= c >> 12;
    b = b.wrapping_sub(c); b = b.wrapping_sub(a); b ^= a << 16;
    c = c.wrapping_sub(a); c = c.wrapping_sub(b); c ^= b >> 5;
    a = a.wrapping_sub(b); a = a.wrapping_sub(c); a ^= c >> 3;
    b = b.wrapping_sub(c); b = b.wrapping_sub(a); b ^= a << 10;
    c = c.wrapping_sub(a); c = c.wrapping_sub(b); c ^= b >> 15;
    c
  }

  /// upstream: Hash::jenkinsMix (hash.cpp)
  pub fn jenkins_mix(a: &mut u32, b: &mut u32, c: &mut u32) {
    *a = a.wrapping_sub(*b); *a = a.wrapping_sub(*c); *a ^= *c >> 13;
    *b = b.wrapping_sub(*c); *b = b.wrapping_sub(*a); *b ^= *a << 8;
    *c = c.wrapping_sub(*a); *c = c.wrapping_sub(*b); *c ^= *b >> 13;
    *a = a.wrapping_sub(*b); *a = a.wrapping_sub(*c); *a ^= *c >> 12;
    *b = b.wrapping_sub(*c); *b = b.wrapping_sub(*a); *b ^= *a << 16;
    *c = c.wrapping_sub(*a); *c = c.wrapping_sub(*b); *c ^= *b >> 5;
    *a = a.wrapping_sub(*b); *a = a.wrapping_sub(*c); *a ^= *c >> 3;
    *b = b.wrapping_sub(*c); *b = b.wrapping_sub(*a); *b ^= *a << 10;
    *c = c.wrapping_sub(*a); *c = c.wrapping_sub(*b); *c ^= *b >> 15;
  }

  /// upstream: Hash::simpleHash (hash.cpp)
  /// `simpleHash(const char* str)`.
  /// PORT NOTE: a C string ends at the first NUL; a `&str` may contain NULs, so stop there too.
  pub fn simple_hash(str: &str) -> u64 {
    let mut m1: u64 = 123456789;
    let mut m2: u64 = 314159265;
    let mut m3: u64 = 958473711;
    for &byte in str.as_bytes() {
      if byte == b'\0' {
        break;
      }
      let c = byte as i8; // x86-64 char is signed
      m1 = m1.wrapping_mul(31).wrapping_add(c as u64);
      m2 = m2.wrapping_mul(317).wrapping_add(c as u64);
      m3 = m3.wrapping_mul(1609).wrapping_add(c as u64);
    }
    let lo = Hash::jenkins_mix_single(Hash::low_bits(m1), Hash::high_bits(m2), Hash::low_bits(m3));
    let hi = Hash::jenkins_mix_single(Hash::high_bits(m1), Hash::low_bits(m2), Hash::high_bits(m3));
    Hash::combine(hi, lo)
  }

  /// upstream: Hash::simpleHash (hash.cpp)
  /// `simpleHash(const int* input, int len)`.
  pub fn simple_hash_input_len(input: &[i32], len: i32) -> u64 {
    let mut m1: u64 = 123456789;
    let mut m2: u64 = 314159265;
    let mut m3: u64 = 958473711;
    let mut m4: u32 = 0xCAFEBABE;
    for i in 0..len {
      let c: i32 = input[i as usize];
      m1 = m1.wrapping_mul(31).wrapping_add(c as u64);
      m2 = m2.wrapping_mul(317).wrapping_add(c as u64);
      m3 = m3.wrapping_mul(1609).wrapping_add(c as u64);
      m4 = m4.wrapping_add(c as u32);
      m4 = m4.wrapping_add(m4 << 10);
      m4 ^= m4 >> 6;
    }
    m4 = m4.wrapping_add(m4 << 3);
    m4 ^= m4 >> 11;
    m4 = m4.wrapping_add(m4 << 15);

    let lo = Hash::jenkins_mix_single(Hash::low_bits(m1), Hash::high_bits(m2), Hash::low_bits(m3));
    let hi = Hash::jenkins_mix_single(Hash::high_bits(m1), Hash::low_bits(m2), Hash::high_bits(m3));
    Hash::combine(hi ^ m4, lo.wrapping_add(m4))
  }

  /// upstream: Hash::seededHashFloat (hash.cpp)
  //Return value in [0,1) deterministically based on seeded hash. Note that 1.0 might
  //be possible depending on how float math is configured
  pub fn seeded_hash_float(str: &str, seed: &str) -> f64 {
    let s: String = String::from(seed) + "|" + str;
    // PORT NOTE: SHA2::get256(const char*, uint64_t[4]) -> sha2 crate, big-endian words (PORTING.md §10).
    let hash: [u64; 4] = crate::port::digest::sha2_get256_u64(&s);
    let bits: u64 = hash[0] & ((1u64 << 53) - 1);
    let x: f64 = bits as f64 / (1u64 << 53) as f64;
    x
  }
}

/// upstream: rotateRight (hash.cpp) static
fn rotate_right(x: u64, r: i32) -> u64 {
  (x >> r) | (x << (64 - r))
}

//Hash128------------------------------------------------------------------------

/// upstream: Hash128 (hash.h)
//Hash is "little endian" in the sense that if you printed out hash1, then hash0, you would
//get the standard 128 bit hash string as would be output by something like MD5, as well as
//for comparison.
#[derive(Clone, Copy, Default, PartialEq, Eq, Hash, Debug)]
pub struct Hash128 {
  pub hash0: u64,
  pub hash1: u64,
}

impl Hash128 {
  /// upstream: Hash128::Hash128 (hash.h)
  /// PORT NOTE: `Hash128()` is `Hash128::default()`; `Hash128(h0, h1)` is `new` (PORTING.md §3).
  pub const fn new(h0: u64, h1: u64) -> Hash128 {
    Hash128 { hash0: h0, hash1: h1 }
  }

  /// upstream: Hash128::toString (hash.cpp)
  pub fn to_string(&self) -> String {
    Global::uint64_to_hex_string(self.hash1) + &Global::uint64_to_hex_string(self.hash0)
  }

  /// upstream: Hash128::ofString (hash.cpp)
  pub fn of_string(s: &str) -> Result<Hash128, KataError> {
    if s.len() != 32 {
      return Err(KataError::IOError(String::from("Could not parse as Hash128: ") + s));
    }
    for &c in s.as_bytes() {
      if !(c >= b'0' && c <= b'9') && !(c >= b'A' && c <= b'F') && !(c >= b'a' && c <= b'f') {
        return Err(KataError::IOError(String::from("Could not parse as Hash128: ") + s));
      }
    }
    let h1: u64 = Global::hex_string_to_uint64(&s[0..16])?;
    let h0: u64 = Global::hex_string_to_uint64(&s[16..32])?;
    Ok(Hash128::new(h0, h1))
  }
}

/// upstream: Hash128::operator< , operator>, operator<=, operator>= (hash.h)
/// Compares hash1 first, then hash0 (PORTING.md §3: not the derived order).
impl PartialOrd for Hash128 {
  fn partial_cmp(&self, other: &Hash128) -> Option<std::cmp::Ordering> {
    Some(self.cmp(other))
  }
}

impl Ord for Hash128 {
  fn cmp(&self, other: &Hash128) -> std::cmp::Ordering {
    if self.hash1 < other.hash1 {
      return std::cmp::Ordering::Less;
    }
    if self.hash1 > other.hash1 {
      return std::cmp::Ordering::Greater;
    }
    self.hash0.cmp(&other.hash0)
  }
}

/// upstream: Hash128::operator^ (hash.h)
impl std::ops::BitXor for Hash128 {
  type Output = Hash128;
  fn bitxor(self, other: Hash128) -> Hash128 {
    Hash128::new(self.hash0 ^ other.hash0, self.hash1 ^ other.hash1)
  }
}
/// upstream: Hash128::operator| (hash.h)
impl std::ops::BitOr for Hash128 {
  type Output = Hash128;
  fn bitor(self, other: Hash128) -> Hash128 {
    Hash128::new(self.hash0 | other.hash0, self.hash1 | other.hash1)
  }
}
/// upstream: Hash128::operator& (hash.h)
impl std::ops::BitAnd for Hash128 {
  type Output = Hash128;
  fn bitand(self, other: Hash128) -> Hash128 {
    Hash128::new(self.hash0 & other.hash0, self.hash1 & other.hash1)
  }
}
/// upstream: Hash128::operator^= (hash.h)
impl std::ops::BitXorAssign for Hash128 {
  fn bitxor_assign(&mut self, other: Hash128) {
    self.hash0 ^= other.hash0;
    self.hash1 ^= other.hash1;
  }
}
/// upstream: Hash128::operator|= (hash.h)
impl std::ops::BitOrAssign for Hash128 {
  fn bitor_assign(&mut self, other: Hash128) {
    self.hash0 |= other.hash0;
    self.hash1 |= other.hash1;
  }
}
/// upstream: Hash128::operator&= (hash.h)
impl std::ops::BitAndAssign for Hash128 {
  fn bitand_assign(&mut self, other: Hash128) {
    self.hash0 &= other.hash0;
    self.hash1 &= other.hash1;
  }
}

/// upstream: operator<<(ostream&, const Hash128) (hash.cpp)
impl std::fmt::Display for Hash128 {
  fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(out, "{}{}", Global::uint64_to_hex_string(self.hash1), Global::uint64_to_hex_string(self.hash0))
  }
}
