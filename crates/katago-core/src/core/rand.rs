// Ported from KataGo cpp/core/rand.cpp, cpp/core/rand.h @ fd0723fdbc0e9d82cf269c9630af8c27c57c07c4

/*
 * rand.h
 * Author: David Wu
 *
 * Basic class for random number generation.
 * Individual rand objects are not threadsafe but the seeding logic should ensure that different
 * instances created simultaneously in different threads or on different machines should all be distinct.
 * Note: Signed integer functions might not be portable to other architectures.
 *
 * On a single core basic cloud machine, roughly generates about 500 million uint32_t/second and
 * can be initialized roughly 100k/second, and 250k/second for fixed seed.
 *
 *
 * Combines:
 * PCG32 (period 2^64)
 * XorShift1024Mult (period 2^1024-1)
 */

use crate::core::bsearch::BSearch;
use crate::core::global::Global;
use crate::core::global::KataError;
use crate::core::rand_helpers::PCG32;
use crate::core::rand_helpers::XorShift1024Mult;
use crate::port::digest;
// PORT NOTE: log/pow go through port::math (glibc-identical on every target incl. wasm, PORTING.md §6).
use crate::port::math;
use crate::testAssert;

/// upstream: Rand (rand.h)
/// PORT NOTE: copy constructor and assignment are deleted upstream, so no Clone.
pub struct Rand {
  pub(crate) xorm: XorShift1024Mult,
  pub(crate) pcg32: PCG32,

  pub(crate) has_gaussian: bool,
  pub(crate) stored_gaussian: f64,

  pub(crate) init_seed: String,
  pub(crate) num_calls: u64,
}

impl Rand {
  //MISC-------------------------------------------------

  /// upstream: Rand::getSeed (rand.h)
  #[inline]
  pub fn get_seed(&self) -> String {
    self.init_seed.clone()
  }

  /// upstream: Rand::getNumCalls (rand.h)
  #[inline]
  pub fn get_num_calls(&self) -> u64 {
    self.num_calls
  }

  //UNSIGNED INTEGER-------------------------------------

  /// upstream: Rand::nextUInt (rand.h)
  //Returns a random integer in [0,2^32)
  #[inline]
  pub fn next_uint(&mut self) -> u32 {
    self.pcg32.next_uint().wrapping_add(self.xorm.next_uint())
  }

  /// upstream: Rand::nextUInt (rand.h)
  //Returns a random integer in [0,n)
  #[inline]
  pub fn next_uint_n(&mut self, n: u32) -> u32 {
    debug_assert!(n > 0);
    let mut bits: u32;
    let mut val: u32;
    loop {
      bits = self.next_uint();
      val = bits % n;
      if !(bits.wrapping_sub(val).wrapping_add(n - 1) < bits.wrapping_sub(val)) {
        break;
      }
    } //If adding (n-1) overflows, no good.
    val
  }

  /// upstream: Rand::nextInt (rand.h)
  //Returns a random integer in [-2^31,2^31)
  #[inline]
  pub fn next_int(&mut self) -> i32 {
    self.next_uint() as i32
  }

  /// upstream: Rand::nextInt (rand.h)
  //Returns a random integer in [a,b]
  #[inline]
  pub fn next_int_a_b(&mut self, a: i32, b: i32) -> i32 {
    debug_assert!(b >= a);
    let max: u32 = (b as u32).wrapping_sub(a as u32).wrapping_add(1u32);
    if max == 0 {
      self.next_uint() as i32
    } else {
      self.next_uint_n(max).wrapping_add(a as u32) as i32
    }
  }

  //64-BIT INTEGER--------------------------------------

  /// upstream: Rand::nextUInt64 (rand.h)
  //Returns a random integer in [0,2^64)
  #[inline]
  pub fn next_uint64(&mut self) -> u64 {
    let lower: u64 = self.next_uint() as u64;
    let upper: u64 = (self.next_uint() as u64) << 32;
    lower | upper
  }

  /// upstream: Rand::nextUInt64 (rand.h)
  //Returns a random integer in [0,n)
  #[inline]
  pub fn next_uint64_n(&mut self, n: u64) -> u64 {
    debug_assert!(n > 0);
    let mut bits: u64;
    let mut val: u64;
    loop {
      bits = self.next_uint64();
      val = bits % n;
      if !(bits.wrapping_sub(val).wrapping_add(n - 1) < bits.wrapping_sub(val)) {
        break;
      }
    } //If adding (n-1) overflows, no good.
    val
  }

  /// upstream: Rand::nextUInt (rand.h)
  //Returns a random integer according to the given frequency distribution
  #[inline]
  pub fn next_uint_freq_n(&mut self, freq: &[i32], n: usize) -> u32 {
    debug_assert!(n > 0);
    debug_assert!(n < 0xFFFFFFFF);
    let mut sum: i64 = 0;
    let mut i: u32 = 0;
    while (i as usize) < n {
      debug_assert!(freq[i as usize] >= 0);
      sum += freq[i as usize] as i64;
      i += 1;
    }

    let mut r: i64 = self.next_uint64_n(sum as u64) as i64;
    debug_assert!(r >= 0);
    let mut i: u32 = 0;
    while (i as usize) < n {
      r -= freq[i as usize] as i64;
      if r < 0 {
        return i;
      }
      i += 1;
    }
    //Should not get to here
    unreachable!()
  }

  /// upstream: Rand::nextUInt (rand.h)
  //Returns a random integer according to the given unnormalized probability distribution
  #[inline]
  pub fn next_uint_rel_probs_n(&mut self, rel_probs: &[f64], n: usize) -> u32 {
    debug_assert!(n > 0);
    debug_assert!(n < 0xFFFFFFFF);
    let mut sum: f64 = 0.0;
    let mut i: u32 = 0;
    while (i as usize) < n {
      debug_assert!(rel_probs[i as usize] >= 0.0);
      sum += rel_probs[i as usize];
      i += 1;
    }

    let d: f64 = self.next_double_n(sum);
    sum = 0.0;
    let mut i: u32 = 0;
    while (i as usize) < n {
      sum += rel_probs[i as usize];
      if sum > d {
        return i;
      }
      i += 1;
    }
    (n - 1) as u32
  }

  //DOUBLE----------------------------------------------

  /// upstream: Rand::nextDouble (rand.h)
  //Returns a random double in [0,1)
  #[inline]
  pub fn next_double(&mut self) -> f64 {
    let mut x: f64;
    loop {
      let bits: u64 = self.next_uint64() & ((1u64 << 53) - 1);
      x = bits as f64 / (1u64 << 53) as f64;
      //Catch loss of precision of long --> double conversion
      if !(!(x >= 0.0 && x < 1.0)) {
        break;
      }
    }

    x
  }

  /// upstream: Rand::nextDouble (rand.h)
  //Returns a random double in [0,n). Note: Rarely, it may be possible for n to occur.
  #[inline]
  pub fn next_double_n(&mut self, n: f64) -> f64 {
    debug_assert!(n >= 0.0);
    self.next_double() * n
  }

  /// upstream: Rand::nextDouble (rand.h)
  //Returns a random double in [a,b). Note: Rarely, it may be possible for b to occur.
  #[inline]
  pub fn next_double_a_b(&mut self, a: f64, b: f64) -> f64 {
    debug_assert!(b >= a);
    a + self.next_double_n(b - a)
  }

  //BOOL------------------------------------------------

  /// upstream: Rand::nextBool (rand.h)
  //Returns true with probability prob
  #[inline]
  pub fn next_bool(&mut self, prob: f64) -> bool {
    self.next_double() < prob
  }

  /// upstream: Rand::nextGaussian (rand.h)
  //Returns a normally distributed double with mean 0 stdev 1
  #[inline]
  pub fn next_gaussian(&mut self) -> f64 {
    if self.has_gaussian {
      self.has_gaussian = false;
      self.stored_gaussian
    } else {
      let mut v1: f64;
      let mut v2: f64;
      let mut s: f64;
      loop {
        v1 = self.next_double() * 2.0 - 1.0;
        v2 = self.next_double() * 2.0 - 1.0;
        s = v1 * v1 + v2 * v2;
        if !(s >= 1.0 || s == 0.0) {
          break;
        }
      }

      let multiplier: f64 = (-2.0 * math::ln(s) / s).sqrt();
      self.stored_gaussian = v2 * multiplier;
      self.has_gaussian = true;
      v1 * multiplier
    }
  }

  /// upstream: Rand::nextGaussianTruncated (rand.h)
  //Truncated refers to the probability distribution, not the sample
  //So on falling outside the range, we redraw, rather than capping.
  #[inline]
  pub fn next_gaussian_truncated(&mut self, bound: f64) -> f64 {
    debug_assert!(bound >= 0.1);
    let mut d: f64 = self.next_gaussian();
    while d < -bound || d > bound {
      d = self.next_gaussian();
    }
    d
  }

  /// upstream: Rand::nextExponential (rand.h)
  //Returns an exponentially distributed double with mean 1.
  #[inline]
  pub fn next_exponential(&mut self) -> f64 {
    let mut r: f64 = 0.0;
    while r <= 1e-17 {
      r = self.next_double();
    }
    -math::ln(r)
  }

  /// upstream: Rand::nextLogistic (rand.h)
  //Returns a logistically distributed double with mean 0 and scale 1 (cdf = 1/(1+exp(-x)))
  #[inline]
  pub fn next_logistic(&mut self) -> f64 {
    let mut num: f64 = 0.0;
    let mut denom: f64 = 0.0;
    while num <= 0.0 || denom <= 0.0 {
      num = self.next_double();
      denom = 1.0 - num;
    }
    math::ln(num / denom)
  }

  //OTHER------------------------------------------------

  /// upstream: Rand::shuffle (rand.h)
  pub fn shuffle<T>(&mut self, vec: &mut Vec<T>) {
    let mut i: usize = 1;
    while i < vec.len() {
      // PORT NOTE: size_t fed to the RNG stays 64-bit (PORTING.md §3).
      let r: usize = self.next_uint64_n((i as u64) + 1) as usize;
      vec.swap(i, r);
      i += 1;
    }
  }
}

// ----------------------------------------------------------------------------------------
// rand.cpp
// ----------------------------------------------------------------------------------------

impl XorShift1024Mult {
  /// upstream: XorShift1024Mult::XorShift1024Mult (rand.cpp)
  pub fn new(init_a: &[u64]) -> XorShift1024Mult {
    let mut x = XorShift1024Mult { a: [0u64; XorShift1024Mult::XORMULT_LEN as usize], a_idx: 0 };
    x.init(init_a);
    x
  }

  /// upstream: XorShift1024Mult::init (rand.cpp)
  pub fn init(&mut self, init_a: &[u64]) {
    self.a_idx = 0;
    for i in 0..XorShift1024Mult::XORMULT_LEN {
      self.a[i as usize] = init_a[i as usize];
    }
  }

  /// upstream: XorShift1024Mult::test (rand.cpp)
  pub fn test() {
    let init_a: [u64; XorShift1024Mult::XORMULT_LEN as usize] = [
      15148282349006049087,
      3601266951833665894,
      16929445066801446424,
      13475938501103070154,
      15713138009143754412,
      4148159782736716337,
      16035594834001032141,
      5555591070439871209,
      4101130512537511022,
      12821547636792886909,
      9050874162294428797,
      6187760405891629771,
      10053646276519763308,
      2219782655280501359,
      3719698449347562208,
      5421263376768154227,
    ];

    let mut xorm = XorShift1024Mult::new(&init_a);

    let expected: [u32; 32] = [
      0x749746d1, 0x9242ca14, 0x98db98a1, 0x1348e491, 0xde60e668, 0x77e37a69, 0xeb51a9d3, 0xd44b4727,
      0x341895b0, 0xc7b1b3f4, 0xe7ef0529, 0x8e72ea7e, 0x5855da19, 0xfffcd2b2, 0xa684e430, 0xb76a7e0d,
      0x5af3820e, 0x320b0699, 0xdbb85ee0, 0xc1dcd25c, 0x4b395e3e, 0x4007756f, 0x76a0c667, 0xaa6041f6,
      0x756f94bb, 0x39527d1b, 0x6e1232ef, 0xb3027668, 0x776ea832, 0x35a0ed1b, 0x1f2f0268, 0xadd59669,
    ];

    for i in 0..32 {
      let r: u32 = xorm.next_uint();
      if r != expected[i] {
        log::error!("{}", i);
        log::error!("{}", Global::uint32_to_hex_string(r));
        log::error!("{}", Global::uint32_to_hex_string(expected[i]));
        Global::fatal_error("XorShift1024Mult generated unexpected values");
      }
    }
  }
}

//-----------------------------------------------------------------------------

impl PCG32 {
  /// upstream: PCG32::PCG32 (rand.cpp)
  pub fn new(state: u64) -> PCG32 {
    let mut p = PCG32 { s: 0 };
    p.init(state);
    p
  }

  /// upstream: PCG32::init (rand.cpp)
  pub fn init(&mut self, state: u64) {
    self.s = state;
  }

  /// upstream: PCG32::test (rand.cpp)
  pub fn test() {
    let mut pcg = PCG32::new(123);

    let expected: [u32; 16] = [
      0xb3766cbd, 0x65fdd305, 0x2a3b9b9c, 0x09a2dee9, 0x1a86aabc, 0x36a98234, 0x82e6e2b4, 0x10c077e5,
      0x29755fc7, 0xf7fa7b5c, 0x1cb7ae7d, 0xcce0e3d9, 0x065ec08b, 0x505d1cdb, 0x8b778f3c, 0xdb72f217,
    ];

    for i in 0..16 {
      let r: u32 = pcg.next_uint();
      if r != expected[i] {
        log::error!("{}", i);
        log::error!("{}", Global::uint32_to_hex_string(r));
        log::error!("{}", Global::uint32_to_hex_string(expected[i]));
        Global::fatal_error("PCG32 generated unexpected values");
      }
    }
  }
}

//---------------------------------------------------------------------------------

/// upstream: zeros (rand.cpp) static
const zeros: [u64; XorShift1024Mult::XORMULT_LEN as usize] = [0u64; XorShift1024Mult::XORMULT_LEN as usize];

impl Rand {
  /// Field values of the `:xorm(zeros),pcg32(0ULL)` init list. Dummy values, overridden by init.
  fn new_uninit() -> Rand {
    Rand {
      xorm: XorShift1024Mult::new(&zeros),
      pcg32: PCG32::new(0u64),
      has_gaussian: false,
      stored_gaussian: 0.0,
      init_seed: String::new(),
      num_calls: 0,
    }
  }

  /// upstream: Rand::Rand (rand.cpp)
  //Initializes according to system time and some other unique junk, tries
  //to make sure multiple invocations will be different
  pub fn new() -> Rand {
    let mut r = Rand::new_uninit(); //Dummy values, overridden by init
    r.init();
    r
  }

  /// upstream: Rand::Rand (rand.cpp)
  //Intializes according to the provided seed
  /// PORT NOTE: the `const char*` and `const std::string&` overloads collapse into one `&str` fn.
  pub fn new_seed(seed: &str) -> Rand {
    let mut r = Rand::new_uninit(); //Dummy values, overridden by init
    r.init_seed(seed);
    r
  }

  /// upstream: Rand::Rand (rand.cpp)
  pub fn new_seed_u64(seed: u64) -> Rand {
    let mut r = Rand::new_uninit(); //Dummy values, overridden by init
    r.init_seed_u64(seed);
    r
  }

  /// upstream: Rand::init (rand.cpp)
  //Reinitialize according to system time and some other unique junk
  /// PORT NOTE: upstream mixes an atomic call counter, time(), clock(), the precision system
  /// time, pid, hostname, thread id and stack/heap addresses into the entropy string. None of
  /// those exist (or are allowed) on wasm32-unknown-unknown, so the string is 32 bytes from
  /// `getrandom` instead (port-manifest.toml notes). Never used where outputs are compared.
  pub fn init(&mut self) {
    //Assemble entropy sources
    let mut entropy = [0u8; 32];
    if let Err(e) = getrandom::fill(&mut entropy) {
      Global::fatal_error(&format!("Rand::init: getrandom failed: {}", e));
    }
    let mut s = String::new();
    for b in entropy.iter() {
      s += &Global::uint32_to_hex_string(*b as u32)[6..8];
    }

    let hash: String = digest::sha2_get256_hex(&s);
    testAssert!(hash.len() == 64);
    let hashed: String = hash;
    self.init_seed(&hashed);
  }

  /// upstream: Rand::init (rand.cpp)
  pub fn init_seed_u64(&mut self, seed: u64) {
    self.init_seed(&Global::uint64_to_hex_string(seed));
  }

  /// upstream: Rand::init (rand.cpp)
  //Reinitialize according to the provided seed
  /// PORT NOTE: the `const char*` overload (which forwards to this one) collapses into it.
  pub fn init_seed(&mut self, seed: &str) {
    self.init_seed = seed.to_string();

    let mut s = String::new();
    {
      let hash: [u32; 4] = digest::md5_get(seed.as_bytes());
      s += "|";
      s += &hash[0].to_string();
      s += "|";
      s += seed;
    }

    let mut counter: i32 = 0;
    let mut next_hash_idx: i32 = 4;
    let mut hash: [u64; 4] = [0u64; 4];
    let mut get_nonzero = || -> u64 {
      let mut next_value: u64;
      loop {
        if next_hash_idx >= 4 {
          let tmp: String = counter.to_string() + &s;
          //cout << tmp << endl;
          counter += 37;
          hash = digest::sha2_get256_u64(&tmp);
          next_hash_idx = 0;
        }
        next_value = hash[next_hash_idx as usize];
        next_hash_idx += 1;
        if !(next_value == 0) {
          break;
        }
      }
      next_value
    };

    let mut init_a: [u64; XorShift1024Mult::XORMULT_LEN as usize] = [0u64; XorShift1024Mult::XORMULT_LEN as usize];
    for i in 0..XorShift1024Mult::XORMULT_LEN {
      init_a[i as usize] = get_nonzero();
    }

    self.xorm.init(&init_a);
    let v = get_nonzero();
    self.pcg32.init(v);

    self.has_gaussian = false;
    self.stored_gaussian = 0.0;
    self.num_calls = 0;
  }

  /// upstream: Rand::nextIndexCumulative (rand.cpp)
  //Returns a random integer according to the given unnormalized cumulative probability distribution
  pub fn next_index_cumulative(&mut self, cum_rel_probs: &[f64], n: usize) -> usize {
    testAssert!(n > 0);
    testAssert!(n < 0xFFFFFFFF);
    let sum: f64 = cum_rel_probs[n - 1];
    let d: f64 = self.next_double_n(sum);
    let r: usize = BSearch::find_first_gt(cum_rel_probs, d, 0, n);
    if r == n {
      return n - 1;
    }
    r
  }

  /// upstream: Rand::nextGamma (rand.cpp)
  //Marsaglia and Tsang's algorithm
  //Returns a gamma distributed double with shape a and scale 1
  pub fn next_gamma(&mut self, a: f64) -> Result<f64, KataError> {
    if !(a > 0.0) {
      return Err(KataError::StringError(String::from("Rand::nextGamma: invalid value for a: ") + &Global::double_to_string(a)));
    }

    if a <= 1.0 {
      let r: f64 = self.next_gamma(a + 1.0)?;
      let inva: f64 = 1.0 / a;
      //Technically in C++, pow(0,0) could be implementation-dependent or result in an error
      //so we explicitly force the desired behavior
      let scale: f64 = if inva == 0.0 { 1.0 } else { math::powf(self.next_double(), inva) };
      return Ok(r * scale);
    }

    let d: f64 = a - 1.0 / 3.0;
    let c: f64 = (1.0 / 3.0) / d.sqrt();

    loop {
      let x: f64 = self.next_gaussian();
      let vtmp: f64 = 1.0 + c * x;
      if vtmp <= 0.0 {
        continue;
      }
      let v: f64 = vtmp * vtmp * vtmp;
      let u: f64 = self.next_double();
      let xx: f64 = x * x;
      if u < 1.0 - 0.0331 * xx * xx {
        return Ok(d * v);
      }
      if u == 0.0 || math::ln(u) < 0.5 * xx + d * (1.0 - v + math::ln(v)) {
        return Ok(d * v);
      }
    }

    //Numeric analysis notes:
    // d >= 2/3
    // c:  0 <= 1/sqrt(9d) <= 0.4 ish
    // vtmp > 0  vtmp < some large number since gaussian can't return infinity
    // u [0,1)
    // v > 0  < some large number
    // xx >= 0 < some large number
  }

  /// upstream: Rand::fillShuffledUIntRange (rand.cpp)
  // Fills buf[0] through buf[n-1] with a random permutation of the integers from 0 to n-1 inclusive.
  pub fn fill_shuffled_uint_range(&mut self, n: usize, buf: &mut [u32]) -> Result<(), KataError> {
    if (n as u64) >= 0xFFFFffffu64 {
      return Err(KataError::StringError(
        String::from("Rand::fillShuffledUIntRange: out of range value for n: ") + &Global::uint64_to_string(n as u64),
      ));
    }
    for i in 0..n {
      buf[i] = i as u32;
    }
    for i in 1..n {
      let r: usize = self.next_uint_n((i + 1) as u32) as usize;
      buf.swap(i, r);
    }
    Ok(())
  }
}
