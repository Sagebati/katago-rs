// Ported from KataGo cpp/external/nlohmann_json/json.hpp (nlohmann/json 3.8.0, MIT) @ fd0723fdbc0e9d82cf269c9630af8c27c57c07c4
//
// 1. The compact `basic_json::dump()` path (indent = -1, ensure_ascii = false,
//    error_handler = strict) and the grisu2 `dtoa_impl` it uses for doubles.
// 2. `basic_json::parse` / `sax_parse` for JSON text: detail::lexer, detail::parser and
//    detail::json_sax_dom_parser, building a serde_json::Value.
// 3. `get<T>` (from_json) for the value types upstream reads.
// PORT NOTE: `dump(&Value)` is the only serializer entry point (docs/PORTING.md appendix), so the
// pretty-print branches (indent >= 0), ensure_ascii and the binary value type are not ported.
// PORT NOTE: serde_json strings are always valid UTF-8, so dump_escaped's UTF8_REJECT and
// incomplete-sequence paths (type_error 316) are unreachable from `dump`. They are ported over
// raw bytes in `dump_escaped_bytes` (`check_dump_string`): a C++ string_t is bytes, and a
// std::string built from invalid input makes upstream's dump() throw.

use serde_json::Map;
use serde_json::Number;
use serde_json::Value;

use crate::core::global::KataError;

/// upstream: nlohmann::detail::dtoa_impl (json.hpp)
mod dtoa_impl {
  /// upstream: nlohmann::detail::dtoa_impl::diyfp (json.hpp)
  #[derive(Clone, Copy, Debug)]
  pub struct diyfp {
    // f * 2^e
    pub f: u64,
    pub e: i32,
  }

  impl diyfp {
    #[expect(dead_code, reason = "mirrors diyfp::kPrecision (json.hpp); nothing in the ported dtoa path reads it")]
    pub const kPrecision: i32 = 64; // = q

    pub const fn new(f_: u64, e_: i32) -> diyfp {
      diyfp { f: f_, e: e_ }
    }

    /// upstream: nlohmann::detail::dtoa_impl::diyfp::sub (json.hpp)
    // returns x - y
    // pre: x.e == y.e and x.f >= y.f
    pub fn sub(x: &diyfp, y: &diyfp) -> diyfp {
      debug_assert!(x.e == y.e);
      debug_assert!(x.f >= y.f);

      diyfp::new(x.f - y.f, x.e)
    }

    /// upstream: nlohmann::detail::dtoa_impl::diyfp::mul (json.hpp)
    // returns x * y
    // The result is rounded. (Only the upper q bits are returned.)
    pub fn mul(x: &diyfp, y: &diyfp) -> diyfp {
      // Computes:
      //  f = round((x.f * y.f) / 2^q)
      //  e = x.e + y.e + q

      // Emulate the 64-bit * 64-bit multiplication:
      //
      // p = u * v
      //   = (u_lo + 2^32 u_hi) (v_lo + 2^32 v_hi)
      //   = (u_lo v_lo         ) + 2^32 ((u_lo v_hi         ) + (u_hi v_lo         )) + 2^64 (u_hi v_hi         )
      //   = (p0                ) + 2^32 ((p1                ) + (p2                )) + 2^64 (p3                )
      //   = (p0_lo + 2^32 p0_hi) + 2^32 ((p1_lo + 2^32 p1_hi) + (p2_lo + 2^32 p2_hi)) + 2^64 (p3                )
      //   = (p0_lo             ) + 2^32 (p0_hi + p1_lo + p2_lo                      ) + 2^64 (p1_hi + p2_hi + p3)
      //   = (p0_lo             ) + 2^32 (Q                                          ) + 2^64 (H                 )
      //   = (p0_lo             ) + 2^32 (Q_lo + 2^32 Q_hi                           ) + 2^64 (H                 )
      //
      // (Since Q might be larger than 2^32 - 1)
      //
      //   = (p0_lo + 2^32 Q_lo) + 2^64 (Q_hi + H)
      //
      // (Q_hi + H does not overflow a 64-bit int)
      //
      //   = p_lo + 2^64 p_hi

      let u_lo: u64 = x.f & 0xFFFFFFFF;
      let u_hi: u64 = x.f >> 32;
      let v_lo: u64 = y.f & 0xFFFFFFFF;
      let v_hi: u64 = y.f >> 32;

      let p0: u64 = u_lo * v_lo;
      let p1: u64 = u_lo * v_hi;
      let p2: u64 = u_hi * v_lo;
      let p3: u64 = u_hi * v_hi;

      let p0_hi: u64 = p0 >> 32;
      let p1_lo: u64 = p1 & 0xFFFFFFFF;
      let p1_hi: u64 = p1 >> 32;
      let p2_lo: u64 = p2 & 0xFFFFFFFF;
      let p2_hi: u64 = p2 >> 32;

      let mut Q: u64 = p0_hi + p1_lo + p2_lo;

      // The full product might now be computed as
      //
      // p_hi = p3 + p2_hi + p1_hi + (Q >> 32)
      // p_lo = p0_lo + (Q << 32)
      //
      // But in this particular case here, the full p_lo is not required.
      // Effectively we only need to add the highest bit in p_lo to p_hi (and
      // Q_hi + 1 does not overflow).

      Q += 1u64 << (64 - 32 - 1); // round, ties up

      let h: u64 = p3 + p2_hi + p1_hi + (Q >> 32);

      diyfp::new(h, x.e + y.e + 64)
    }

    /// upstream: nlohmann::detail::dtoa_impl::diyfp::normalize (json.hpp)
    // normalize x such that the significand is >= 2^(q-1)
    // pre: x.f != 0
    pub fn normalize(mut x: diyfp) -> diyfp {
      debug_assert!(x.f != 0);

      while (x.f >> 63) == 0 {
        x.f <<= 1;
        x.e -= 1;
      }

      x
    }

    /// upstream: nlohmann::detail::dtoa_impl::diyfp::normalize_to (json.hpp)
    // normalize x such that the result has the exponent E
    // pre: e >= x.e and the upper e - x.e bits of x.f must be zero.
    pub fn normalize_to(x: &diyfp, target_exponent: i32) -> diyfp {
      let delta: i32 = x.e - target_exponent;

      debug_assert!(delta >= 0);
      debug_assert!(((x.f << delta) >> delta) == x.f);

      diyfp::new(x.f << delta, target_exponent)
    }
  }

  /// upstream: nlohmann::detail::dtoa_impl::boundaries (json.hpp)
  pub struct boundaries {
    pub w: diyfp,
    pub minus: diyfp,
    pub plus: diyfp,
  }

  /// upstream: nlohmann::detail::dtoa_impl::compute_boundaries (json.hpp)
  // Compute the (normalized) diyfp representing the input number 'value' and its
  // boundaries.
  // pre: value must be finite and positive
  // PORT NOTE: the template is only instantiated with FloatType = double (number_float_t).
  pub fn compute_boundaries(value: f64) -> boundaries {
    debug_assert!(value.is_finite());
    debug_assert!(value > 0.0);

    // Convert the IEEE representation into a diyfp.
    //
    // If v is denormal:
    //      value = 0.F * 2^(1 - bias) = (          F) * 2^(1 - bias - (p-1))
    // If v is normalized:
    //      value = 1.F * 2^(E - bias) = (2^(p-1) + F) * 2^(E - bias - (p-1))

    let kPrecision: i32 = 53; // = p (includes the hidden bit)
    let kBias: i32 = 1024 - 1 + (kPrecision - 1);
    let kMinExp: i32 = 1 - kBias;
    let kHiddenBit: u64 = 1u64 << (kPrecision - 1); // = 2^(p-1)

    let bits: u64 = value.to_bits();
    let E: u64 = bits >> (kPrecision - 1);
    let F: u64 = bits & (kHiddenBit - 1);

    let is_denormal = E == 0;
    let v = if is_denormal { diyfp::new(F, kMinExp) } else { diyfp::new(F + kHiddenBit, E as i32 - kBias) };

    // Compute the boundaries m- and m+ of the floating-point value
    // v = f * 2^e.
    //
    // Determine v- and v+, the floating-point predecessor and successor if v,
    // respectively.
    //
    //      v- = v - 2^e        if f != 2^(p-1) or e == e_min                (A)
    //         = v - 2^(e-1)    if f == 2^(p-1) and e > e_min                (B)
    //
    //      v+ = v + 2^e
    //
    // Let m- = (v- + v) / 2 and m+ = (v + v+) / 2. All real numbers _strictly_
    // between m- and m+ round to v, regardless of how the input rounding
    // algorithm breaks ties.
    //
    //      ---+-------------+-------------+-------------+-------------+---  (A)
    //         v-            m-            v             m+            v+
    //
    //      -----------------+------+------+-------------+-------------+---  (B)
    //                       v-     m-     v             m+            v+

    let lower_boundary_is_closer = F == 0 && E > 1;
    let m_plus = diyfp::new(2 * v.f + 1, v.e - 1);
    let m_minus = if lower_boundary_is_closer {
      diyfp::new(4 * v.f - 1, v.e - 2) // (B)
    } else {
      diyfp::new(2 * v.f - 1, v.e - 1) // (A)
    };

    // Determine the normalized w+ = m+.
    let w_plus = diyfp::normalize(m_plus);

    // Determine w- = m- such that e_(w-) = e_(w+).
    let w_minus = diyfp::normalize_to(&m_minus, w_plus.e);

    boundaries { w: diyfp::normalize(v), minus: w_minus, plus: w_plus }
  }

  // Given normalized diyfp w, Grisu needs to find a (normalized) cached
  // power-of-ten c, such that the exponent of the product c * w = f * 2^e lies
  // within a certain range [alpha, gamma] (Definition 3.2 from [1])
  // (See json.hpp for the full derivation; (alpha,gamma)=(-60,-32).)

  pub const kAlpha: i32 = -60;
  pub const kGamma: i32 = -32;

  /// upstream: nlohmann::detail::dtoa_impl::cached_power (json.hpp)
  #[derive(Clone, Copy)]
  pub struct cached_power {
    // c = f * 2^e ~= 10^k
    pub f: u64,
    pub e: i32,
    pub k: i32,
  }

  const fn cp(f: u64, e: i32, k: i32) -> cached_power {
    cached_power { f, e, k }
  }

  /// upstream: nlohmann::detail::dtoa_impl::get_cached_power_for_binary_exponent (json.hpp)
  // For a normalized diyfp w = f * 2^e, this function returns a (normalized) cached
  // power-of-ten c = f_c * 2^e_c, such that the exponent of the product w * c
  // satisfies (Definition 3.2 from [1])
  //
  //      alpha <= e_c + e + q <= gamma.
  pub fn get_cached_power_for_binary_exponent(e: i32) -> cached_power {
    // NB:
    // Actually this function returns c, such that -60 <= e_c + e + 64 <= -34.

    let kCachedPowersMinDecExp: i32 = -300;
    let kCachedPowersDecStep: i32 = 8;

    const kCachedPowers: [cached_power; 79] = [
      cp(0xAB70FE17C79AC6CA, -1060, -300),
      cp(0xFF77B1FCBEBCDC4F, -1034, -292),
      cp(0xBE5691EF416BD60C, -1007, -284),
      cp(0x8DD01FAD907FFC3C, -980, -276),
      cp(0xD3515C2831559A83, -954, -268),
      cp(0x9D71AC8FADA6C9B5, -927, -260),
      cp(0xEA9C227723EE8BCB, -901, -252),
      cp(0xAECC49914078536D, -874, -244),
      cp(0x823C12795DB6CE57, -847, -236),
      cp(0xC21094364DFB5637, -821, -228),
      cp(0x9096EA6F3848984F, -794, -220),
      cp(0xD77485CB25823AC7, -768, -212),
      cp(0xA086CFCD97BF97F4, -741, -204),
      cp(0xEF340A98172AACE5, -715, -196),
      cp(0xB23867FB2A35B28E, -688, -188),
      cp(0x84C8D4DFD2C63F3B, -661, -180),
      cp(0xC5DD44271AD3CDBA, -635, -172),
      cp(0x936B9FCEBB25C996, -608, -164),
      cp(0xDBAC6C247D62A584, -582, -156),
      cp(0xA3AB66580D5FDAF6, -555, -148),
      cp(0xF3E2F893DEC3F126, -529, -140),
      cp(0xB5B5ADA8AAFF80B8, -502, -132),
      cp(0x87625F056C7C4A8B, -475, -124),
      cp(0xC9BCFF6034C13053, -449, -116),
      cp(0x964E858C91BA2655, -422, -108),
      cp(0xDFF9772470297EBD, -396, -100),
      cp(0xA6DFBD9FB8E5B88F, -369, -92),
      cp(0xF8A95FCF88747D94, -343, -84),
      cp(0xB94470938FA89BCF, -316, -76),
      cp(0x8A08F0F8BF0F156B, -289, -68),
      cp(0xCDB02555653131B6, -263, -60),
      cp(0x993FE2C6D07B7FAC, -236, -52),
      cp(0xE45C10C42A2B3B06, -210, -44),
      cp(0xAA242499697392D3, -183, -36),
      cp(0xFD87B5F28300CA0E, -157, -28),
      cp(0xBCE5086492111AEB, -130, -20),
      cp(0x8CBCCC096F5088CC, -103, -12),
      cp(0xD1B71758E219652C, -77, -4),
      cp(0x9C40000000000000, -50, 4),
      cp(0xE8D4A51000000000, -24, 12),
      cp(0xAD78EBC5AC620000, 3, 20),
      cp(0x813F3978F8940984, 30, 28),
      cp(0xC097CE7BC90715B3, 56, 36),
      cp(0x8F7E32CE7BEA5C70, 83, 44),
      cp(0xD5D238A4ABE98068, 109, 52),
      cp(0x9F4F2726179A2245, 136, 60),
      cp(0xED63A231D4C4FB27, 162, 68),
      cp(0xB0DE65388CC8ADA8, 189, 76),
      cp(0x83C7088E1AAB65DB, 216, 84),
      cp(0xC45D1DF942711D9A, 242, 92),
      cp(0x924D692CA61BE758, 269, 100),
      cp(0xDA01EE641A708DEA, 295, 108),
      cp(0xA26DA3999AEF774A, 322, 116),
      cp(0xF209787BB47D6B85, 348, 124),
      cp(0xB454E4A179DD1877, 375, 132),
      cp(0x865B86925B9BC5C2, 402, 140),
      cp(0xC83553C5C8965D3D, 428, 148),
      cp(0x952AB45CFA97A0B3, 455, 156),
      cp(0xDE469FBD99A05FE3, 481, 164),
      cp(0xA59BC234DB398C25, 508, 172),
      cp(0xF6C69A72A3989F5C, 534, 180),
      cp(0xB7DCBF5354E9BECE, 561, 188),
      cp(0x88FCF317F22241E2, 588, 196),
      cp(0xCC20CE9BD35C78A5, 614, 204),
      cp(0x98165AF37B2153DF, 641, 212),
      cp(0xE2A0B5DC971F303A, 667, 220),
      cp(0xA8D9D1535CE3B396, 694, 228),
      cp(0xFB9B7CD9A4A7443C, 720, 236),
      cp(0xBB764C4CA7A44410, 747, 244),
      cp(0x8BAB8EEFB6409C1A, 774, 252),
      cp(0xD01FEF10A657842C, 800, 260),
      cp(0x9B10A4E5E9913129, 827, 268),
      cp(0xE7109BFBA19C0C9D, 853, 276),
      cp(0xAC2820D9623BF429, 880, 284),
      cp(0x80444B5E7AA7CF85, 907, 292),
      cp(0xBF21E44003ACDD2D, 933, 300),
      cp(0x8E679C2F5E44FF8F, 960, 308),
      cp(0xD433179D9C8CB841, 986, 316),
      cp(0x9E19DB92B4E31BA9, 1013, 324),
    ];

    // This computation gives exactly the same results for k as
    //      k = ceil((kAlpha - e - 1) * 0.30102999566398114)
    // for |e| <= 1500, but doesn't require floating-point operations.
    // NB: log_10(2) ~= 78913 / 2^18
    debug_assert!(e >= -1500);
    debug_assert!(e <= 1500);
    let f: i32 = kAlpha - e - 1;
    let k: i32 = (f * 78913) / (1 << 18) + (f > 0) as i32;

    let index: i32 = (-kCachedPowersMinDecExp + k + (kCachedPowersDecStep - 1)) / kCachedPowersDecStep;
    debug_assert!(index >= 0);
    debug_assert!((index as usize) < kCachedPowers.len());

    let cached = kCachedPowers[index as usize];
    debug_assert!(kAlpha <= cached.e + e + 64);
    debug_assert!(kGamma >= cached.e + e + 64);

    cached
  }

  /// upstream: nlohmann::detail::dtoa_impl::find_largest_pow10 (json.hpp)
  // For n != 0, returns k, such that pow10 := 10^(k-1) <= n < 10^k.
  // For n == 0, returns 1 and sets pow10 := 1.
  pub fn find_largest_pow10(n: u32, pow10: &mut u32) -> i32 {
    // LCOV_EXCL_START
    if n >= 1000000000 {
      *pow10 = 1000000000;
      10
    }
    // LCOV_EXCL_STOP
    else if n >= 100000000 {
      *pow10 = 100000000;
      9
    } else if n >= 10000000 {
      *pow10 = 10000000;
      8
    } else if n >= 1000000 {
      *pow10 = 1000000;
      7
    } else if n >= 100000 {
      *pow10 = 100000;
      6
    } else if n >= 10000 {
      *pow10 = 10000;
      5
    } else if n >= 1000 {
      *pow10 = 1000;
      4
    } else if n >= 100 {
      *pow10 = 100;
      3
    } else if n >= 10 {
      *pow10 = 10;
      2
    } else {
      *pow10 = 1;
      1
    }
  }

  /// upstream: nlohmann::detail::dtoa_impl::grisu2_round (json.hpp)
  pub fn grisu2_round(buf: &mut [u8], len: i32, dist: u64, delta: u64, mut rest: u64, ten_k: u64) {
    debug_assert!(len >= 1);
    debug_assert!(dist <= delta);
    debug_assert!(rest <= delta);
    debug_assert!(ten_k > 0);

    //               <--------------------------- delta ---->
    //                                  <---- dist --------->
    // --------------[------------------+-------------------]--------------
    //               M-                 w                   M+
    //
    //                                  ten_k
    //                                <------>
    //                                       <---- rest ---->
    // --------------[------------------+----+--------------]--------------
    //                                  w    V
    //                                       = buf * 10^k
    //
    // ten_k represents a unit-in-the-last-place in the decimal representation
    // stored in buf.
    // Decrement buf by ten_k while this takes buf closer to w.

    // The tests are written in this order to avoid overflow in unsigned
    // integer arithmetic.

    while rest < dist && delta - rest >= ten_k && (rest + ten_k < dist || dist - rest > rest + ten_k - dist) {
      debug_assert!(buf[(len - 1) as usize] != b'0');
      buf[(len - 1) as usize] -= 1;
      rest += ten_k;
    }
  }

  /// upstream: nlohmann::detail::dtoa_impl::grisu2_digit_gen (json.hpp)
  // Generates V = buffer * 10^decimal_exponent, such that M- <= V <= M+.
  // M- and M+ must be normalized and share the same exponent -60 <= e <= -32.
  pub fn grisu2_digit_gen(buffer: &mut [u8], length: &mut i32, decimal_exponent: &mut i32, M_minus: diyfp, w: diyfp, M_plus: diyfp) {
    // Generates the digits (and the exponent) of a decimal floating-point
    // number V = buffer * 10^decimal_exponent in the range [M-, M+]. The diyfp's
    // w, M- and M+ share the same exponent e, which satisfies alpha <= e <= gamma.
    //
    //               <--------------------------- delta ---->
    //                                  <---- dist --------->
    // --------------[------------------+-------------------]--------------
    //               M-                 w                   M+
    //
    // Grisu2 generates the digits of M+ from left to right and stops as soon as
    // V is in [M-,M+].

    debug_assert!(M_plus.e >= kAlpha);
    debug_assert!(M_plus.e <= kGamma);

    let mut delta: u64 = diyfp::sub(&M_plus, &M_minus).f; // (significand of (M+ - M-), implicit exponent is e)
    let mut dist: u64 = diyfp::sub(&M_plus, &w).f; // (significand of (M+ - w ), implicit exponent is e)

    // Split M+ = f * 2^e into two parts p1 and p2 (note: e < 0):
    //
    //      M+ = f * 2^e
    //         = ((f div 2^-e) * 2^-e + (f mod 2^-e)) * 2^e
    //         = ((p1        ) * 2^-e + (p2        )) * 2^e
    //         = p1 + p2 * 2^e

    let one = diyfp::new(1u64 << -M_plus.e, M_plus.e);

    let mut p1: u32 = (M_plus.f >> -one.e) as u32; // p1 = f div 2^-e (Since -e >= 32, p1 fits into a 32-bit int.)
    let mut p2: u64 = M_plus.f & (one.f - 1); // p2 = f mod 2^-e

    // 1)
    //
    // Generate the digits of the integral part p1 = d[n-1]...d[1]d[0]

    debug_assert!(p1 > 0);

    let mut pow10: u32 = 0;
    let k: i32 = find_largest_pow10(p1, &mut pow10);

    //      10^(k-1) <= p1 < 10^k, pow10 = 10^(k-1)
    //
    //      p1 = (p1 div 10^(k-1)) * 10^(k-1) + (p1 mod 10^(k-1))
    //         = (d[k-1]         ) * 10^(k-1) + (p1 mod 10^(k-1))
    //
    //      M+ = p1                                             + p2 * 2^e
    //         = d[k-1] * 10^(k-1) + (p1 mod 10^(k-1))          + p2 * 2^e
    //         = d[k-1] * 10^(k-1) + ((p1 mod 10^(k-1)) * 2^-e + p2) * 2^e
    //         = d[k-1] * 10^(k-1) + (                         rest) * 2^e
    //
    // Now generate the digits d[n] of p1 from left to right (n = k-1,...,0)
    //
    //      p1 = d[k-1]...d[n] * 10^n + d[n-1]...d[0]
    //
    // but stop as soon as
    //
    //      rest * 2^e = (d[n-1]...d[0] * 2^-e + p2) * 2^e <= delta * 2^e

    let mut n: i32 = k;
    while n > 0 {
      // Invariants:
      //      M+ = buffer * 10^n + (p1 + p2 * 2^e)    (buffer = 0 for n = k)
      //      pow10 = 10^(n-1) <= p1 < 10^n
      //
      let d: u32 = p1 / pow10; // d = p1 div 10^(n-1)
      let r: u32 = p1 % pow10; // r = p1 mod 10^(n-1)
      //
      //      M+ = buffer * 10^n + (d * 10^(n-1) + r) + p2 * 2^e
      //         = (buffer * 10 + d) * 10^(n-1) + (r + p2 * 2^e)
      //
      debug_assert!(d <= 9);
      buffer[*length as usize] = b'0' + d as u8; // buffer := buffer * 10 + d
      *length += 1;
      //
      //      M+ = buffer * 10^(n-1) + (r + p2 * 2^e)
      //
      p1 = r;
      n -= 1;
      //
      //      M+ = buffer * 10^n + (p1 + p2 * 2^e)
      //      pow10 = 10^n
      //

      // Now check if enough digits have been generated.
      // Compute
      //
      //      p1 + p2 * 2^e = (p1 * 2^-e + p2) * 2^e = rest * 2^e
      //
      // Note:
      // Since rest and delta share the same exponent e, it suffices to
      // compare the significands.
      let rest: u64 = ((p1 as u64) << -one.e) + p2;
      if rest <= delta {
        // V = buffer * 10^n, with M- <= V <= M+.

        *decimal_exponent += n;

        // We may now just stop. But instead look if the buffer could be
        // decremented to bring V closer to w.
        //
        // pow10 = 10^n is now 1 ulp in the decimal representation V.
        // The rounding procedure works with diyfp's with an implicit
        // exponent of e.
        //
        //      10^n = (10^n * 2^-e) * 2^e = ulp * 2^e
        //
        let ten_n: u64 = (pow10 as u64) << -one.e;
        grisu2_round(buffer, *length, dist, delta, rest, ten_n);

        return;
      }

      pow10 /= 10;
      //
      //      pow10 = 10^(n-1) <= p1 < 10^n
      // Invariants restored.
    }

    // 2)
    //
    // The digits of the integral part have been generated:
    //
    //      M+ = d[k-1]...d[1]d[0] + p2 * 2^e
    //         = buffer            + p2 * 2^e
    //
    // Now generate the digits of the fractional part p2 * 2^e.
    //
    // Note:
    // No decimal point is generated: the exponent is adjusted instead.
    //
    // p2 actually represents the fraction
    //
    //      p2 * 2^e
    //          = p2 / 2^-e
    //          = d[-1] / 10^1 + d[-2] / 10^2 + ...
    //
    // Now generate the digits d[-m] of p1 from left to right (m = 1,2,...)
    //
    //      p2 * 2^e = d[-1]d[-2]...d[-m] * 10^-m
    //                      + 10^-m * (d[-m-1] / 10^1 + d[-m-2] / 10^2 + ...)
    //
    // using
    //
    //      10^m * p2 = ((10^m * p2) div 2^-e) * 2^-e + ((10^m * p2) mod 2^-e)
    //                = (                   d) * 2^-e + (                   r)
    //
    // or
    //      10^m * p2 * 2^e = d + r * 2^e
    //
    // i.e.
    //
    //      M+ = buffer + p2 * 2^e
    //         = buffer + 10^-m * (d + r * 2^e)
    //         = (buffer * 10^m + d) * 10^-m + 10^-m * r * 2^e
    //
    // and stop as soon as 10^-m * r * 2^e <= delta * 2^e

    debug_assert!(p2 > delta);

    let mut m: i32 = 0;
    loop {
      // Invariant:
      //      M+ = buffer * 10^-m + 10^-m * (d[-m-1] / 10 + d[-m-2] / 10^2 + ...) * 2^e
      //         = buffer * 10^-m + 10^-m * (p2                                 ) * 2^e
      //         = buffer * 10^-m + 10^-m * (1/10 * (10 * p2)                   ) * 2^e
      //         = buffer * 10^-m + 10^-m * (1/10 * ((10*p2 div 2^-e) * 2^-e + (10*p2 mod 2^-e)) * 2^e
      //
      debug_assert!(p2 <= u64::MAX / 10);
      p2 *= 10;
      let d: u64 = p2 >> -one.e; // d = (10 * p2) div 2^-e
      let r: u64 = p2 & (one.f - 1); // r = (10 * p2) mod 2^-e
      //
      //      M+ = buffer * 10^-m + 10^-m * (1/10 * (d * 2^-e + r) * 2^e
      //         = buffer * 10^-m + 10^-m * (1/10 * (d + r * 2^e))
      //         = (buffer * 10 + d) * 10^(-m-1) + 10^(-m-1) * r * 2^e
      //
      debug_assert!(d <= 9);
      buffer[*length as usize] = b'0' + d as u8; // buffer := buffer * 10 + d
      *length += 1;
      //
      //      M+ = buffer * 10^(-m-1) + 10^(-m-1) * r * 2^e
      //
      p2 = r;
      m += 1;
      //
      //      M+ = buffer * 10^-m + 10^-m * p2 * 2^e
      // Invariant restored.

      // Check if enough digits have been generated.
      //
      //      10^-m * p2 * 2^e <= delta * 2^e
      //              p2 * 2^e <= 10^m * delta * 2^e
      //                    p2 <= 10^m * delta
      delta *= 10;
      dist *= 10;
      if p2 <= delta {
        break;
      }
    }

    // V = buffer * 10^-m, with M- <= V <= M+.

    *decimal_exponent -= m;

    // 1 ulp in the decimal representation is now 10^-m.
    // Since delta and dist are now scaled by 10^m, we need to do the
    // same with ulp in order to keep the units in sync.
    //
    //      10^m * 10^-m = 1 = 2^-e * 2^e = ten_m * 2^e
    //
    let ten_m: u64 = one.f;
    grisu2_round(buffer, *length, dist, delta, p2, ten_m);

    // By construction this algorithm generates the shortest possible decimal
    // number (Loitsch, Theorem 6.2) which rounds back to w.
    // For an input number of precision p, at least
    //
    //      N = 1 + ceil(p * log_10(2))
    //
    // decimal digits are sufficient to identify all binary floating-point
    // numbers (Matula, "In-and-Out conversions").
    // This implies that the algorithm does not produce more than N decimal
    // digits.
    //
    //      N = 17 for p = 53 (IEEE double precision)
    //      N = 9  for p = 24 (IEEE single precision)
  }

  /// upstream: nlohmann::detail::dtoa_impl::grisu2 (json.hpp)
  // v = buf * 10^decimal_exponent
  // len is the length of the buffer (number of decimal digits)
  // The buffer must be large enough, i.e. >= max_digits10.
  pub fn grisu2_diyfp(buf: &mut [u8], len: &mut i32, decimal_exponent: &mut i32, m_minus: diyfp, v: diyfp, m_plus: diyfp) {
    debug_assert!(m_plus.e == m_minus.e);
    debug_assert!(m_plus.e == v.e);

    //  --------(-----------------------+-----------------------)--------    (A)
    //          m-                      v                       m+
    //
    //  --------------------(-----------+-----------------------)--------    (B)
    //                      m-          v                       m+
    //
    // First scale v (and m- and m+) such that the exponent is in the range
    // [alpha, gamma].

    let cached = get_cached_power_for_binary_exponent(m_plus.e);

    let c_minus_k = diyfp::new(cached.f, cached.e); // = c ~= 10^-k

    // The exponent of the products is = v.e + c_minus_k.e + q and is in the range [alpha,gamma]
    let w = diyfp::mul(&v, &c_minus_k);
    let w_minus = diyfp::mul(&m_minus, &c_minus_k);
    let w_plus = diyfp::mul(&m_plus, &c_minus_k);

    //  ----(---+---)---------------(---+---)---------------(---+---)----
    //          w-                      w                       w+
    //          = c*m-                  = c*v                   = c*m+
    //
    // diyfp::mul rounds its result and c_minus_k is approximated too. w, w- and
    // w+ are now off by a small amount.
    // In fact:
    //
    //      w - v * 10^k < 1 ulp
    //
    // To account for this inaccuracy, add resp. subtract 1 ulp.
    //
    //  --------+---[---------------(---+---)---------------]---+--------
    //          w-  M-                  w                   M+  w+
    //
    // Now any number in [M-, M+] (bounds included) will round to w when input,
    // regardless of how the input rounding algorithm breaks ties.
    //
    // And digit_gen generates the shortest possible such number in [M-, M+].
    // Note that this does not mean that Grisu2 always generates the shortest
    // possible number in the interval (m-, m+).
    let M_minus = diyfp::new(w_minus.f + 1, w_minus.e);
    let M_plus = diyfp::new(w_plus.f - 1, w_plus.e);

    *decimal_exponent = -cached.k; // = -(-k) = k

    grisu2_digit_gen(buf, len, decimal_exponent, M_minus, w, M_plus);
  }

  /// upstream: nlohmann::detail::dtoa_impl::grisu2 (json.hpp)
  // v = buf * 10^decimal_exponent
  // len is the length of the buffer (number of decimal digits)
  // The buffer must be large enough, i.e. >= max_digits10.
  // PORT NOTE: overload renamed (grisu2 diyfp overload is grisu2_diyfp); FloatType = double.
  pub fn grisu2(buf: &mut [u8], len: &mut i32, decimal_exponent: &mut i32, value: f64) {
    debug_assert!(value.is_finite());
    debug_assert!(value > 0.0);

    // If the neighbors (and boundaries) of 'value' are always computed for double-precision
    // numbers, all float's can be recovered using strtod (and strtof). However, the resulting
    // decimal representations are not exactly "short".
    //
    // The documentation for 'std::to_chars' (https://en.cppreference.com/w/cpp/utility/to_chars)
    // says "value is converted to a string as if by std::sprintf in the default ("C") locale"
    // and since sprintf promotes float's to double's, I think this is exactly what 'std::to_chars'
    // does.
    // On the other hand, the documentation for 'std::to_chars' requires that "parsing the
    // representation using the corresponding std::from_chars function recovers value exactly". That
    // indicates that single precision floating-point numbers should be recovered using
    // 'std::strtof'.
    //
    // NB: If the neighbors are computed for single-precision numbers, there is a single float
    //     (7.0385307e-26f) which can't be recovered using strtod. The resulting double precision
    //     value is off by 1 ulp.
    let w = compute_boundaries(value);

    grisu2_diyfp(buf, len, decimal_exponent, w.minus, w.w, w.plus);
  }

  /// upstream: nlohmann::detail::dtoa_impl::append_exponent (json.hpp)
  // appends a decimal representation of e to buf
  // return: a pointer to the element following the exponent.
  // pre: -1000 < e < 1000
  // PORT NOTE: pointers become an index into `buf`.
  pub fn append_exponent(buf: &mut [u8], mut pos: usize, mut e: i32) -> usize {
    debug_assert!(e > -1000);
    debug_assert!(e < 1000);

    if e < 0 {
      e = -e;
      buf[pos] = b'-';
      pos += 1;
    } else {
      buf[pos] = b'+';
      pos += 1;
    }

    let mut k: u32 = e as u32;
    if k < 10 {
      // Always print at least two digits in the exponent.
      // This is for compatibility with printf("%g").
      buf[pos] = b'0';
      pos += 1;
      buf[pos] = b'0' + k as u8;
      pos += 1;
    } else if k < 100 {
      buf[pos] = b'0' + (k / 10) as u8;
      pos += 1;
      k %= 10;
      buf[pos] = b'0' + k as u8;
      pos += 1;
    } else {
      buf[pos] = b'0' + (k / 100) as u8;
      pos += 1;
      k %= 100;
      buf[pos] = b'0' + (k / 10) as u8;
      pos += 1;
      k %= 10;
      buf[pos] = b'0' + k as u8;
      pos += 1;
    }

    pos
  }

  /// upstream: nlohmann::detail::dtoa_impl::format_buffer (json.hpp)
  // prettify v = buf * 10^decimal_exponent
  //
  // If v is in the range [10^min_exp, 10^max_exp) it will be printed in fixed-point
  // notation. Otherwise it will be printed in exponential notation.
  //
  // pre: min_exp < 0
  // pre: max_exp > 0
  // PORT NOTE: returns the end index instead of a pointer.
  pub fn format_buffer(buf: &mut [u8], len: i32, decimal_exponent: i32, min_exp: i32, max_exp: i32) -> usize {
    debug_assert!(min_exp < 0);
    debug_assert!(max_exp > 0);

    let k: i32 = len;
    let n: i32 = len + decimal_exponent;

    // v = buf * 10^(n-k)
    // k is the length of the buffer (number of decimal digits)
    // n is the position of the decimal point relative to the start of the buffer.

    if k <= n && n <= max_exp {
      // digits[000]
      // len <= max_exp + 2

      buf[k as usize..n as usize].fill(b'0');
      // Make it look like a floating-point number (#362, #378)
      buf[n as usize] = b'.';
      buf[n as usize + 1] = b'0';
      return n as usize + 2;
    }

    if 0 < n && n <= max_exp {
      // dig.its
      // len <= max_digits10 + 1

      debug_assert!(k > n);

      buf.copy_within(n as usize..k as usize, n as usize + 1);
      buf[n as usize] = b'.';
      return k as usize + 1;
    }

    if min_exp < n && n <= 0 {
      // 0.[000]digits
      // len <= 2 + (-min_exp - 1) + max_digits10

      buf.copy_within(0..k as usize, 2 + (-n) as usize);
      buf[0] = b'0';
      buf[1] = b'.';
      buf[2..2 + (-n) as usize].fill(b'0');
      return 2 + (-n) as usize + k as usize;
    }

    let pos: usize;
    if k == 1 {
      // dE+123
      // len <= 1 + 5

      pos = 1;
    } else {
      // d.igitsE+123
      // len <= max_digits10 + 1 + 5

      buf.copy_within(1..k as usize, 2);
      buf[1] = b'.';
      pos = 1 + k as usize;
    }

    buf[pos] = b'e';
    append_exponent(buf, pos + 1, n - 1)
  }
}

/// upstream: nlohmann::detail::to_chars (json.hpp)
// generates a decimal representation of the floating-point number value in [first, last).
// The format of the resulting decimal representation is similar to printf's %g
// format. Returns an iterator pointing past-the-end of the decimal representation.
// note: The input number must be finite, i.e. NaN's and Inf's are not supported.
// PORT NOTE: writes into a String instead of a char range; FloatType = double.
pub fn to_chars(out: &mut String, mut value: f64) {
  debug_assert!(value.is_finite());

  // Use signbit(value) instead of (value < 0) since signbit works for -0.
  if value.is_sign_negative() {
    value = -value;
    out.push('-');
  }

  if value == 0.0 {
    // +-0
    // Always print 0.0 for +-0 (#362, #378)
    out.push_str("0.0");
    return;
  }

  // Compute v = buffer * 10^decimal_exponent.
  // The decimal digits are stored in the buffer, which needs to be interpreted
  // as an unsigned decimal integer.
  // len is the length of the buffer, i.e. the number of decimal digits.
  let mut buf = [0u8; 64];
  let mut len: i32 = 0;
  let mut decimal_exponent: i32 = 0;
  dtoa_impl::grisu2(&mut buf, &mut len, &mut decimal_exponent, value);

  debug_assert!(len <= 17);

  // Format the buffer like printf("%.*g", prec, value)
  let kMinExp: i32 = -4;
  // Use digits10 here to increase compatibility with version 2.
  let kMaxExp: i32 = 15; // std::numeric_limits<double>::digits10

  let end = dtoa_impl::format_buffer(&mut buf, len, decimal_exponent, kMinExp, kMaxExp);
  out.push_str(std::str::from_utf8(&buf[..end]).expect("dtoa output is ASCII"));
}

/// upstream: nlohmann::detail::serializer::dump_escaped (json.hpp)
// PORT NOTE: ensure_ascii = false and error_handler = strict only (the defaults of dump()).
fn dump_escaped(o: &mut String, s: &str) {
  for ch in s.chars() {
    let codepoint = ch as u32;
    match codepoint {
      0x08 => o.push_str("\\b"),  // backspace
      0x09 => o.push_str("\\t"),  // horizontal tab
      0x0A => o.push_str("\\n"),  // newline
      0x0C => o.push_str("\\f"),  // formfeed
      0x0D => o.push_str("\\r"),  // carriage return
      0x22 => o.push_str("\\\""), // quotation mark
      0x5C => o.push_str("\\\\"), // reverse solidus
      _ => {
        // escape control characters (0x00..0x1F)
        if codepoint <= 0x1F {
          o.push_str(&crate::port::fmt::printf("\\u%04x", &[crate::port::fmt::PrintfArg::U32(codepoint)]));
        } else {
          o.push(ch);
        }
      }
    }
  }
}

/// upstream: nlohmann::detail::serializer::UTF8_ACCEPT, UTF8_REJECT (json.hpp)
const UTF8_ACCEPT: u8 = 0;
const UTF8_REJECT: u8 = 1;

/// upstream: nlohmann::detail::serializer::decode (json.hpp)
/// check whether a string is UTF-8 encoded
///
/// The function checks each byte of a string whether it is UTF-8 encoded. The
/// result of the check is stored in the @a state parameter. The function must
/// be called initially with state 0 (accept). State 1 means the string must
/// be rejected, because the current byte is not allowed. If the string is
/// completely processed, but the state is non-zero, the string ended
/// prematurely; that is, the last byte indicated more bytes should have
/// followed.
///
/// @copyright Copyright (c) 2008-2009 Bjoern Hoehrmann <bjoern@hoehrmann.de>
/// @sa http://bjoern.hoehrmann.de/utf-8/decoder/dfa/
fn decode(state: &mut u8, codep: &mut u32, byte: u8) -> u8 {
  #[rustfmt::skip]
  const UTF8D: [u8; 400] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 00..1F
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 20..3F
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 40..5F
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // 60..7F
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, // 80..9F
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, // A0..BF
    8, 8, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, // C0..DF
    0xA, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x4, 0x3, 0x3, // E0..EF
    0xB, 0x6, 0x6, 0x6, 0x5, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, 0x8, // F0..FF
    0x0, 0x1, 0x2, 0x3, 0x5, 0x8, 0x7, 0x1, 0x1, 0x1, 0x4, 0x6, 0x1, 0x1, 0x1, 0x1, // s0..s0
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, // s1..s2
    1, 2, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, // s3..s4
    1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 3, 1, 1, 1, 1, 1, 1, // s5..s6
    1, 3, 1, 1, 1, 1, 1, 3, 1, 3, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, // s7..s8
  ];

  let type_: u8 = UTF8D[byte as usize];

  *codep = if *state != UTF8_ACCEPT { ((byte as u32) & 0x3f) | (*codep << 6) } else { (0xFFu32 >> type_) & (byte as u32) };

  let index: usize = 256 + (*state as usize) * 16 + (type_ as usize);
  debug_assert!(index < 400);
  *state = UTF8D[index];
  *state
}

/// upstream: nlohmann::detail::type_error::create (json.hpp), as `KataError::NlohmannJson(e.what())`
fn type_error_create(id_: i32, what_arg: &str) -> KataError {
  KataError::NlohmannJson(Exception::name("type_error", id_) + what_arg)
}

/// upstream: nlohmann::detail::serializer::dump_escaped (json.hpp)
/// dump escaped string
///
/// Escape a string by replacing certain special characters by a sequence of an
/// escape character (backslash) and another character and other control
/// characters by a sequence of "\u" followed by a four-digit hex
/// representation. The escaped string is written to output stream @a o.
///
/// @param[in] s  the string to escape
/// @param[in] ensure_ascii  whether to escape non-ASCII characters with
///                          \uXXXX sequences
///
/// @complexity Linear in the length of string @a s.
/// PORT NOTE: over raw bytes (a C++ string_t holds any bytes), with ensure_ascii = false and
/// error_handler = strict (the defaults of dump()); the string_buffer chunking is folded into
/// `o.extend`. The throw is `Err(KataError::NlohmannJson(type_error.316))`.
fn dump_escaped_bytes(o: &mut Vec<u8>, s: &[u8]) -> Result<(), KataError> {
  let mut codepoint: u32 = 0;
  let mut state: u8 = UTF8_ACCEPT;
  // the bytes of the code point being decoded (string_buffer between accepts)
  let mut pending: Vec<u8> = Vec::new();

  for i in 0..s.len() {
    let byte: u8 = s[i];

    match decode(&mut state, &mut codepoint, byte) {
      UTF8_ACCEPT => {
        // decode found a new code point
        match codepoint {
          0x08 => o.extend_from_slice(b"\\b"),  // backspace
          0x09 => o.extend_from_slice(b"\\t"),  // horizontal tab
          0x0A => o.extend_from_slice(b"\\n"),  // newline
          0x0C => o.extend_from_slice(b"\\f"),  // formfeed
          0x0D => o.extend_from_slice(b"\\r"),  // carriage return
          0x22 => o.extend_from_slice(b"\\\""), // quotation mark
          0x5C => o.extend_from_slice(b"\\\\"), // reverse solidus
          _ => {
            // escape control characters (0x00..0x1F) or, if
            // ensure_ascii parameter is used, non-ASCII characters
            if codepoint <= 0x1F {
              o.extend_from_slice(crate::port::fmt::printf("\\u%04x", &[crate::port::fmt::PrintfArg::U32(codepoint)]).as_bytes());
            } else {
              // copy byte to buffer (all previous bytes
              // been copied have in default case above)
              o.extend_from_slice(&pending);
              o.push(byte);
            }
          }
        }
        pending.clear();
      }

      UTF8_REJECT => {
        // decode found invalid UTF-8 byte
        // error_handler_t::strict
        let sn: String = crate::port::fmt::printf("%.2X", &[crate::port::fmt::PrintfArg::U32(byte as u32)]);
        return Err(type_error_create(316, &("invalid UTF-8 byte at index ".to_string() + &i.to_string() + ": 0x" + &sn)));
      }

      _ => {
        // decode found yet incomplete multi-byte code point
        // code point will not be escaped - copy byte to buffer
        pending.push(byte);
      }
    }
  }

  // we finished processing the string
  if state == UTF8_ACCEPT {
    // write buffer: already written
    Ok(())
  } else {
    // we finish reading, but do not accept: string was incomplete
    // error_handler_t::strict
    let sn: String = crate::port::fmt::printf("%.2X", &[crate::port::fmt::PrintfArg::U32(*s.last().expect("non-accept state implies a byte") as u32)]);
    Err(type_error_create(316, &("incomplete UTF-8 string; last byte: 0x".to_string() + &sn)))
  }
}

/// `json j = s; j.dump()` for a string value whose C++ bytes are `s` (std::string, any bytes):
/// Ok(the dump) for valid UTF-8, else the type_error.316 that dump() throws.
/// PORT NOTE: serde_json strings cannot hold invalid UTF-8; callers that build a JSON string from
/// raw input bytes (analysis.cpp's reportError of an unparseable line) check it here first.
pub fn dump_string_bytes(s: &[u8]) -> Result<Vec<u8>, KataError> {
  let mut o: Vec<u8> = vec![b'"'];
  dump_escaped_bytes(&mut o, s)?;
  o.push(b'"');
  Ok(o)
}

/// upstream: nlohmann::detail::serializer::dump (json.hpp)
// PORT NOTE: pretty_print = false (compact) branches only.
fn dump_impl(o: &mut String, val: &Value) {
  match val {
    Value::Object(object) => {
      if object.is_empty() {
        o.push_str("{}");
        return;
      }
      o.push('{');
      let size = object.len();
      for (cnt, (key, v)) in object.iter().enumerate() {
        o.push('"');
        dump_escaped(o, key);
        o.push_str("\":");
        dump_impl(o, v);
        if cnt + 1 < size {
          o.push(',');
        }
      }
      o.push('}');
    }
    Value::Array(array) => {
      if array.is_empty() {
        o.push_str("[]");
        return;
      }
      o.push('[');
      let size = array.len();
      for (cnt, v) in array.iter().enumerate() {
        dump_impl(o, v);
        if cnt + 1 < size {
          o.push(',');
        }
      }
      o.push(']');
    }
    Value::String(s) => {
      o.push('"');
      dump_escaped(o, s);
      o.push('"');
    }
    Value::Bool(b) => {
      if *b {
        o.push_str("true");
      } else {
        o.push_str("false");
      }
    }
    Value::Number(num) => {
      // number_integer / number_unsigned -> dump_integer (plain decimal); number_float -> dump_float.
      if let Some(u) = num.as_u64() {
        o.push_str(&u.to_string());
      } else if let Some(i) = num.as_i64() {
        o.push_str(&i.to_string());
      } else {
        let x = num.as_f64().expect("serde_json Number is u64, i64 or f64");
        dump_float(o, x);
      }
    }
    Value::Null => o.push_str("null"),
  }
}

/// upstream: nlohmann::detail::serializer::dump_float (json.hpp)
fn dump_float(o: &mut String, x: f64) {
  // NaN / inf
  if !x.is_finite() {
    o.push_str("null");
    return;
  }
  to_chars(o, x);
}

/// upstream: nlohmann::basic_json::dump (json.hpp)
/// `json.dump()` with default arguments: compact, no ensure_ascii.
pub fn dump(v: &Value) -> String {
  let mut s = String::new();
  dump_impl(&mut s, v);
  s
}

// ---------------------------------------------------------------------------------------
// basic_json::get<T>() for the value types upstream reads (from_json in json.hpp).
// Errors are KataError::NlohmannJson (nlohmann::detail::type_error), which is NOT a StringError.
// ---------------------------------------------------------------------------------------

fn type_name(v: &Value) -> &'static str {
  match v {
    Value::Null => "null",
    Value::Bool(_) => "boolean",
    Value::Number(_) => "number",
    Value::String(_) => "string",
    Value::Array(_) => "array",
    Value::Object(_) => "object",
  }
}

/// upstream: nlohmann::detail::from_json(const BasicJsonType&, string_t&) (json.hpp)
/// `j.get<std::string>()`: only a string; anything else is type_error 302.
pub fn get_string(v: &Value) -> Result<String, KataError> {
  match v {
    Value::String(s) => Ok(s.clone()),
    _ => Err(KataError::NlohmannJson(format!("[json.exception.type_error.302] type must be string, but is {}", type_name(v)))),
  }
}

/// upstream: nlohmann::detail::from_json(const BasicJsonType&, boolean_t&) (json.hpp)
/// `j.get<bool>()`: only a boolean; anything else is type_error 302.
pub fn get_bool(v: &Value) -> Result<bool, KataError> {
  match v {
    Value::Bool(b) => Ok(*b),
    _ => Err(KataError::NlohmannJson(format!("[json.exception.type_error.302] type must be boolean, but is {}", type_name(v)))),
  }
}

/// upstream: nlohmann::detail::get_arithmetic_value / from_json(const BasicJsonType&, ArithmeticType&) (json.hpp)
/// `j.get<float>()`: number_unsigned, number_integer, number_float or boolean, `static_cast<float>`.
pub fn get_f32(v: &Value) -> Result<f32, KataError> {
  match v {
    Value::Number(n) => {
      if let Some(u) = n.as_u64() {
        Ok(u as f32)
      } else if let Some(i) = n.as_i64() {
        Ok(i as f32)
      } else {
        Ok(n.as_f64().expect("serde_json Number is u64, i64 or f64") as f32)
      }
    }
    Value::Bool(b) => Ok(if *b { 1.0 } else { 0.0 }),
    _ => Err(KataError::NlohmannJson(format!("[json.exception.type_error.302] type must be number, but is {}", type_name(v)))),
  }
}

/// `static_cast<int>(double)` as compiled for x86-64 (cvttsd2si).
/// PORT NOTE: C++ UB; x86-64 cvttsd2si gives INT_MIN for NaN and out-of-range values, while Rust
/// `as` saturates (NaN -> 0). In range it truncates toward zero like `as`.
pub fn static_cast_i32_f64(x: f64) -> i32 {
  crate::port::cpp::static_cast_i32_f64(x)
}

/// upstream: nlohmann::detail::from_json(const BasicJsonType&, ArithmeticType&) (json.hpp)
/// `j.get<int>()`: like get_f32, `static_cast<int>`.
/// PORT NOTE: `static_cast<int>` of a u64/i64 keeps the low 32 bits (implementation-defined,
/// = Rust `as`); of a double see static_cast_i32_f64.
pub fn get_i32(v: &Value) -> Result<i32, KataError> {
  match v {
    Value::Number(n) => {
      if let Some(u) = n.as_u64() {
        Ok(u as i32)
      } else if let Some(i) = n.as_i64() {
        Ok(i as i32)
      } else {
        Ok(static_cast_i32_f64(n.as_f64().expect("serde_json Number is u64, i64 or f64")))
      }
    }
    Value::Bool(b) => Ok(*b as i32),
    _ => Err(KataError::NlohmannJson(format!("[json.exception.type_error.302] type must be number, but is {}", type_name(v)))),
  }
}

/// upstream: nlohmann::detail::from_json(const BasicJsonType&, number_integer_t&) -> get_arithmetic_value (json.hpp)
/// `j.get<int64_t>()`: number_unsigned, number_integer or number_float (NOT boolean, unlike get<int>),
/// `static_cast<int64_t>`; anything else is type_error 302.
/// PORT NOTE: `static_cast<int64_t>` of a u64 above i64::MAX wraps (= Rust `as`); of a double out of
/// range or NaN it is C++ UB, x86-64 cvttsd2si gives INT64_MIN (port::cpp::static_cast_i64_f64).
pub fn get_i64(v: &Value) -> Result<i64, KataError> {
  match v {
    Value::Number(n) => {
      if let Some(u) = n.as_u64() {
        Ok(u as i64)
      } else if let Some(i) = n.as_i64() {
        Ok(i)
      } else {
        Ok(crate::port::cpp::static_cast_i64_f64(n.as_f64().expect("serde_json Number is u64, i64 or f64")))
      }
    }
    _ => Err(KataError::NlohmannJson(format!("[json.exception.type_error.302] type must be number, but is {}", type_name(v)))),
  }
}

/// upstream: nlohmann::detail::from_json(const BasicJsonType&, number_float_t&) -> get_arithmetic_value (json.hpp)
/// `j.get<double>()`: number_unsigned, number_integer or number_float (NOT boolean); anything else is
/// type_error 302. Integers convert with `static_cast<double>` (round to nearest, = Rust `as`).
pub fn get_f64(v: &Value) -> Result<f64, KataError> {
  match v {
    Value::Number(n) => {
      if let Some(u) = n.as_u64() {
        Ok(u as f64)
      } else if let Some(i) = n.as_i64() {
        Ok(i as f64)
      } else {
        Ok(n.as_f64().expect("serde_json Number is u64, i64 or f64"))
      }
    }
    _ => Err(KataError::NlohmannJson(format!("[json.exception.type_error.302] type must be number, but is {}", type_name(v)))),
  }
}

/// upstream: nlohmann::detail::from_json(const BasicJsonType&, ConstructibleArrayType&) (json.hpp), for `std::vector<int>`
/// `j.get<std::vector<int>>()`: an array (else type_error 302 "type must be array"), each element `get<int>()`
/// (so booleans and floats convert, see get_i32).
pub fn get_vector_i32(v: &Value) -> Result<Vec<i32>, KataError> {
  match v {
    Value::Array(a) => {
      let mut ret: Vec<i32> = Vec::with_capacity(a.len());
      for elt in a {
        ret.push(get_i32(elt)?);
      }
      Ok(ret)
    }
    _ => Err(KataError::NlohmannJson(format!("[json.exception.type_error.302] type must be array, but is {}", type_name(v)))),
  }
}

/// upstream: nlohmann::detail::from_json(const BasicJsonType&, ConstructibleArrayType&) (json.hpp), for `std::vector<int64_t>`
/// `j.get<std::vector<int64_t>>()`: an array, each element `get<int64_t>()` (see get_i64).
pub fn get_vector_i64(v: &Value) -> Result<Vec<i64>, KataError> {
  match v {
    Value::Array(a) => {
      let mut ret: Vec<i64> = Vec::with_capacity(a.len());
      for elt in a {
        ret.push(get_i64(elt)?);
      }
      Ok(ret)
    }
    _ => Err(KataError::NlohmannJson(format!("[json.exception.type_error.302] type must be array, but is {}", type_name(v)))),
  }
}

/// nlohmann `is_number_integer()`: number_integer or number_unsigned (PORTING.md §9).
pub fn is_number_integer(v: &Value) -> bool {
  v.is_i64() || v.is_u64()
}

// ---------------------------------------------------------------------------------------
// json::parse: detail::lexer, detail::parser and detail::json_sax_dom_parser (json.hpp).
// PORT NOTE: only the JSON text format over a contiguous byte input (`input_adapter(std::string)`
// = iterator_input_adapter), without parser callbacks (KataGo never passes one), with
// allow_exceptions = true. Errors are KataError::NlohmannJson(e.what()).
// PORT NOTE: `parse` takes a `&str` (always valid UTF-8); `parse_bytes` takes any bytes (as a
// std::string holds them), and the lexer checks UTF-8 byte ranges exactly like upstream.
// PORT NOTE: the C locale is assumed (KataGo never calls setlocale), so
// get_decimal_point() is '.'.
// ---------------------------------------------------------------------------------------

/// upstream: nlohmann::detail::exception, parse_error, out_of_range (json.hpp)
/// PORT NOTE: one struct for the exception classes the parser can raise. `what()` returns the
/// C++ bytes: a token string can end inside a multi-byte UTF-8 character (e.g. "invalid
/// literal; last read: '\xef'"), which a Rust String cannot hold.
pub struct Exception {
  /// the id of the exception
  pub id: i32,
  m: Vec<u8>,
}

impl Exception {
  /// upstream: nlohmann::detail::exception::what (json.hpp)
  /// returns the explanatory string
  pub fn what(&self) -> &[u8] {
    &self.m
  }

  /// upstream: nlohmann::detail::exception::name (json.hpp)
  fn name(ename: &str, id_: i32) -> String {
    "[json.exception.".to_string() + ename + "." + &id_.to_string() + "] "
  }

  /// upstream: nlohmann::detail::parse_error::create(int, const position_t&, const std::string&) (json.hpp)
  fn parse_error_create(id_: i32, pos: &position_t, what_arg: &[u8]) -> Exception {
    let mut w: Vec<u8> = (Exception::name("parse_error", id_) + "parse error" + &Exception::position_string(pos) + ": ").into_bytes();
    w.extend_from_slice(what_arg);
    Exception { id: id_, m: w }
  }

  /// upstream: nlohmann::detail::parse_error::position_string (json.hpp)
  fn position_string(pos: &position_t) -> String {
    " at line ".to_string() + &(pos.lines_read + 1).to_string() + ", column " + &pos.chars_read_current_line.to_string()
  }

  /// upstream: nlohmann::detail::out_of_range::create (json.hpp)
  fn out_of_range_create(id_: i32, what_arg: &[u8]) -> Exception {
    let mut w: Vec<u8> = Exception::name("out_of_range", id_).into_bytes();
    w.extend_from_slice(what_arg);
    Exception { id: id_, m: w }
  }

  /// `catch(nlohmann::detail::exception& e)` → KataError::NlohmannJson(e.what()).
  /// PORT NOTE: KataError holds a String; a what() that is not valid UTF-8 (a token string cut
  /// inside a multi-byte character, or raw invalid input bytes) gets U+FFFD there. Callers that
  /// need the C++ bytes use `parse_bytes`, which returns the Exception itself.
  pub fn into_kata_error(self) -> KataError {
    KataError::NlohmannJson(String::from_utf8_lossy(&self.m).into_owned())
  }
}

/// upstream: nlohmann::detail::position_t (json.hpp)
/// struct to capture the start position of the current token
#[derive(Clone, Copy, Default)]
pub struct position_t {
  /// the total number of characters read
  pub chars_read_total: usize,
  /// the number of characters read in the current line
  pub chars_read_current_line: usize,
  /// the number of lines read
  pub lines_read: usize,
}

/// upstream: nlohmann::detail::lexer_base::token_type (json.hpp)
/// token types for the parser
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(i32)]
enum token_type {
  uninitialized,    // indicating the scanner is uninitialized
  literal_true,     // the `true` literal
  literal_false,    // the `false` literal
  literal_null,     // the `null` literal
  value_string,     // a string -- use get_string() for actual value
  value_unsigned,   // an unsigned integer -- use get_number_unsigned() for actual value
  value_integer,    // a signed integer -- use get_number_integer() for actual value
  value_float,      // an floating point number -- use get_number_float() for actual value
  begin_array,      // the character for array begin `[`
  begin_object,     // the character for object begin `{`
  end_array,        // the character for array end `]`
  end_object,       // the character for object end `}`
  name_separator,   // the name separator `:`
  value_separator,  // the value separator `,`
  parse_error,      // indicating a parse error
  end_of_input,     // indicating the end of the input buffer
  literal_or_value, // a literal or the begin of a value (only for diagnostics)
}

/// upstream: nlohmann::detail::lexer_base::token_type_name (json.hpp)
/// return name of values of type token_type (only used for errors)
fn token_type_name(t: token_type) -> &'static str {
  match t {
    token_type::uninitialized => "<uninitialized>",
    token_type::literal_true => "true literal",
    token_type::literal_false => "false literal",
    token_type::literal_null => "null literal",
    token_type::value_string => "string literal",
    token_type::value_unsigned | token_type::value_integer | token_type::value_float => "number literal",
    token_type::begin_array => "'['",
    token_type::begin_object => "'{'",
    token_type::end_array => "']'",
    token_type::end_object => "'}'",
    token_type::name_separator => "':'",
    token_type::value_separator => "','",
    token_type::parse_error => "<parse error>",
    token_type::end_of_input => "end of input",
    token_type::literal_or_value => "'[', '{', or a literal",
  }
}

/// `std::char_traits<char>::eof()`
const EOF: i32 = -1;

/// upstream: nlohmann::detail::iterator_input_adapter (json.hpp)
/// PORT NOTE: over the bytes of a `&str` (input_adapter(std::string) in upstream).
struct iterator_input_adapter<'a> {
  current: usize,
  input: &'a [u8],
}

impl<'a> iterator_input_adapter<'a> {
  /// upstream: nlohmann::detail::iterator_input_adapter::get_character (json.hpp)
  fn get_character(&mut self) -> i32 {
    if self.current != self.input.len() {
      let result = self.input[self.current] as i32;
      self.current += 1;
      result
    } else {
      EOF
    }
  }
}

/// upstream: nlohmann::detail::lexer (json.hpp)
/// lexical analysis
///
/// This class organizes the lexical analysis during JSON deserialization.
struct lexer<'a> {
  /// input adapter
  ia: iterator_input_adapter<'a>,
  /// the current character
  current: i32,
  /// whether the next get() call should just return current
  next_unget: bool,
  /// the start position of the current token
  position: position_t,
  /// raw input token string (for error messages)
  token_string: Vec<u8>,
  /// buffer for variable-length tokens (numbers, strings)
  token_buffer: Vec<u8>,
  /// a description of occurred lexer errors
  error_message: &'static str,
  // number values
  value_integer: i64,
  value_unsigned: u64,
  value_float: f64,
  /// the decimal point
  decimal_point_char: i32,
}

impl<'a> lexer<'a> {
  /// upstream: nlohmann::detail::lexer::lexer (json.hpp)
  fn new(adapter: iterator_input_adapter<'a>) -> lexer<'a> {
    lexer {
      ia: adapter,
      current: EOF,
      next_unget: false,
      position: position_t::default(),
      token_string: Vec::new(),
      token_buffer: Vec::new(),
      error_message: "",
      value_integer: 0,
      value_unsigned: 0,
      value_float: 0.0,
      decimal_point_char: lexer::get_decimal_point() as i32,
    }
  }

  /////////////////////
  // locales
  /////////////////////

  /// upstream: nlohmann::detail::lexer::get_decimal_point (json.hpp)
  /// return the locale-dependent decimal point
  /// PORT NOTE: `localeconv()->decimal_point` in the C locale.
  fn get_decimal_point() -> u8 {
    b'.'
  }

  /////////////////////
  // scan functions
  /////////////////////

  /// upstream: nlohmann::detail::lexer::get_codepoint (json.hpp)
  /// get codepoint from 4 hex characters following `\u`
  ///
  /// @return codepoint (0x0000..0xFFFF) or -1 in case of an error (e.g. EOF or
  ///         non-hex character)
  fn get_codepoint(&mut self) -> i32 {
    // this function only makes sense after reading `\u`
    debug_assert!(self.current == b'u' as i32);
    let mut codepoint: i32 = 0;

    let factors = [12u32, 8u32, 4u32, 0u32];
    for factor in factors {
      self.get();

      if self.current >= b'0' as i32 && self.current <= b'9' as i32 {
        codepoint += (((self.current as u32).wrapping_sub(0x30u32)) << factor) as i32;
      } else if self.current >= b'A' as i32 && self.current <= b'F' as i32 {
        codepoint += (((self.current as u32).wrapping_sub(0x37u32)) << factor) as i32;
      } else if self.current >= b'a' as i32 && self.current <= b'f' as i32 {
        codepoint += (((self.current as u32).wrapping_sub(0x57u32)) << factor) as i32;
      } else {
        return -1;
      }
    }

    debug_assert!(0x0000 <= codepoint && codepoint <= 0xFFFF);
    codepoint
  }

  /// upstream: nlohmann::detail::lexer::next_byte_in_range (json.hpp)
  /// check if the next byte(s) are inside a given range
  ///
  /// Adds the current byte and, for each passed range, reads a new byte and
  /// checks if it is inside the range. If a violation was detected, set up an
  /// error message and return false. Otherwise, return true.
  fn next_byte_in_range(&mut self, ranges: &[i32]) -> bool {
    debug_assert!(ranges.len() == 2 || ranges.len() == 4 || ranges.len() == 6);
    self.add(self.current);

    let mut range = 0;
    while range < ranges.len() {
      self.get();
      if ranges[range] <= self.current && self.current <= ranges[range + 1] {
        self.add(self.current);
      } else {
        self.error_message = "invalid string: ill-formed UTF-8 byte";
        return false;
      }
      range += 2;
    }

    true
  }

  /// upstream: nlohmann::detail::lexer::scan_string (json.hpp)
  /// scan a string literal
  ///
  /// This function scans a string according to Sect. 7 of RFC 7159. While
  /// scanning, bytes are escaped and copied into buffer token_buffer. Then the
  /// function returns successfully, token_buffer is *not* null-terminated (as it
  /// may contain \0 bytes), and token_buffer.size() is the number of bytes in the
  /// string.
  ///
  /// @return token_type::value_string if string could be successfully scanned,
  ///         token_type::parse_error otherwise
  ///
  /// @note In case of errors, variable error_message contains a textual
  ///       description.
  fn scan_string(&mut self) -> token_type {
    // reset token_buffer (ignore opening quote)
    self.reset();

    // we entered the function by reading an open quote
    debug_assert!(self.current == b'"' as i32);

    loop {
      // get next character
      match self.get() {
        // end of file while parsing string
        EOF => {
          self.error_message = "invalid string: missing closing quote";
          return token_type::parse_error;
        }

        // closing quote
        0x22 => {
          return token_type::value_string;
        }

        // escapes
        0x5C => {
          match self.get() {
            // quotation mark
            0x22 => self.add(b'"' as i32),
            // reverse solidus
            0x5C => self.add(b'\\' as i32),
            // solidus
            0x2F => self.add(b'/' as i32),
            // backspace
            0x62 => self.add(0x08),
            // form feed
            0x66 => self.add(0x0C),
            // line feed
            0x6E => self.add(b'\n' as i32),
            // carriage return
            0x72 => self.add(b'\r' as i32),
            // tab
            0x74 => self.add(b'\t' as i32),

            // unicode escapes
            0x75 => {
              let codepoint1: i32 = self.get_codepoint();
              let mut codepoint: i32 = codepoint1; // start with codepoint1

              if codepoint1 == -1 {
                self.error_message = "invalid string: '\\u' must be followed by 4 hex digits";
                return token_type::parse_error;
              }

              // check if code point is a high surrogate
              if 0xD800 <= codepoint1 && codepoint1 <= 0xDBFF {
                // expect next \uxxxx entry
                if self.get() == b'\\' as i32 && self.get() == b'u' as i32 {
                  let codepoint2: i32 = self.get_codepoint();

                  if codepoint2 == -1 {
                    self.error_message = "invalid string: '\\u' must be followed by 4 hex digits";
                    return token_type::parse_error;
                  }

                  // check if codepoint2 is a low surrogate
                  if 0xDC00 <= codepoint2 && codepoint2 <= 0xDFFF {
                    // overwrite codepoint
                    codepoint = (
                      // high surrogate occupies the most significant 22 bits
                      ((codepoint1 as u32) << 10u32)
                      // low surrogate occupies the least significant 15 bits
                      .wrapping_add(codepoint2 as u32)
                      // there is still the 0xD800, 0xDC00 and 0x10000 noise
                      // in the result so we have to subtract with:
                      // (0xD800 << 10) + DC00 - 0x10000 = 0x35FDC00
                      .wrapping_sub(0x35FDC00u32)
                    ) as i32;
                  } else {
                    self.error_message = "invalid string: surrogate U+D800..U+DBFF must be followed by U+DC00..U+DFFF";
                    return token_type::parse_error;
                  }
                } else {
                  self.error_message = "invalid string: surrogate U+D800..U+DBFF must be followed by U+DC00..U+DFFF";
                  return token_type::parse_error;
                }
              } else {
                if 0xDC00 <= codepoint1 && codepoint1 <= 0xDFFF {
                  self.error_message = "invalid string: surrogate U+DC00..U+DFFF must follow U+D800..U+DBFF";
                  return token_type::parse_error;
                }
              }

              // result of the above calculation yields a proper codepoint
              debug_assert!(0x00 <= codepoint && codepoint <= 0x10FFFF);

              // translate codepoint into bytes
              let cp = codepoint as u32;
              if codepoint < 0x80 {
                // 1-byte characters: 0xxxxxxx (ASCII)
                self.add(codepoint);
              } else if codepoint <= 0x7FF {
                // 2-byte characters: 110xxxxx 10xxxxxx
                self.add((0xC0u32 | (cp >> 6u32)) as i32);
                self.add((0x80u32 | (cp & 0x3Fu32)) as i32);
              } else if codepoint <= 0xFFFF {
                // 3-byte characters: 1110xxxx 10xxxxxx 10xxxxxx
                self.add((0xE0u32 | (cp >> 12u32)) as i32);
                self.add((0x80u32 | ((cp >> 6u32) & 0x3Fu32)) as i32);
                self.add((0x80u32 | (cp & 0x3Fu32)) as i32);
              } else {
                // 4-byte characters: 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
                self.add((0xF0u32 | (cp >> 18u32)) as i32);
                self.add((0x80u32 | ((cp >> 12u32) & 0x3Fu32)) as i32);
                self.add((0x80u32 | ((cp >> 6u32) & 0x3Fu32)) as i32);
                self.add((0x80u32 | (cp & 0x3Fu32)) as i32);
              }
            }

            // other characters after escape
            _ => {
              self.error_message = "invalid string: forbidden character after backslash";
              return token_type::parse_error;
            }
          }
        }

        // invalid control characters
        0x00 => {
          self.error_message = "invalid string: control character U+0000 (NUL) must be escaped to \\u0000";
          return token_type::parse_error;
        }
        0x01 => {
          self.error_message = "invalid string: control character U+0001 (SOH) must be escaped to \\u0001";
          return token_type::parse_error;
        }
        0x02 => {
          self.error_message = "invalid string: control character U+0002 (STX) must be escaped to \\u0002";
          return token_type::parse_error;
        }
        0x03 => {
          self.error_message = "invalid string: control character U+0003 (ETX) must be escaped to \\u0003";
          return token_type::parse_error;
        }
        0x04 => {
          self.error_message = "invalid string: control character U+0004 (EOT) must be escaped to \\u0004";
          return token_type::parse_error;
        }
        0x05 => {
          self.error_message = "invalid string: control character U+0005 (ENQ) must be escaped to \\u0005";
          return token_type::parse_error;
        }
        0x06 => {
          self.error_message = "invalid string: control character U+0006 (ACK) must be escaped to \\u0006";
          return token_type::parse_error;
        }
        0x07 => {
          self.error_message = "invalid string: control character U+0007 (BEL) must be escaped to \\u0007";
          return token_type::parse_error;
        }
        0x08 => {
          self.error_message = "invalid string: control character U+0008 (BS) must be escaped to \\u0008 or \\b";
          return token_type::parse_error;
        }
        0x09 => {
          self.error_message = "invalid string: control character U+0009 (HT) must be escaped to \\u0009 or \\t";
          return token_type::parse_error;
        }
        0x0A => {
          self.error_message = "invalid string: control character U+000A (LF) must be escaped to \\u000A or \\n";
          return token_type::parse_error;
        }
        0x0B => {
          self.error_message = "invalid string: control character U+000B (VT) must be escaped to \\u000B";
          return token_type::parse_error;
        }
        0x0C => {
          self.error_message = "invalid string: control character U+000C (FF) must be escaped to \\u000C or \\f";
          return token_type::parse_error;
        }
        0x0D => {
          self.error_message = "invalid string: control character U+000D (CR) must be escaped to \\u000D or \\r";
          return token_type::parse_error;
        }
        0x0E => {
          self.error_message = "invalid string: control character U+000E (SO) must be escaped to \\u000E";
          return token_type::parse_error;
        }
        0x0F => {
          self.error_message = "invalid string: control character U+000F (SI) must be escaped to \\u000F";
          return token_type::parse_error;
        }
        0x10 => {
          self.error_message = "invalid string: control character U+0010 (DLE) must be escaped to \\u0010";
          return token_type::parse_error;
        }
        0x11 => {
          self.error_message = "invalid string: control character U+0011 (DC1) must be escaped to \\u0011";
          return token_type::parse_error;
        }
        0x12 => {
          self.error_message = "invalid string: control character U+0012 (DC2) must be escaped to \\u0012";
          return token_type::parse_error;
        }
        0x13 => {
          self.error_message = "invalid string: control character U+0013 (DC3) must be escaped to \\u0013";
          return token_type::parse_error;
        }
        0x14 => {
          self.error_message = "invalid string: control character U+0014 (DC4) must be escaped to \\u0014";
          return token_type::parse_error;
        }
        0x15 => {
          self.error_message = "invalid string: control character U+0015 (NAK) must be escaped to \\u0015";
          return token_type::parse_error;
        }
        0x16 => {
          self.error_message = "invalid string: control character U+0016 (SYN) must be escaped to \\u0016";
          return token_type::parse_error;
        }
        0x17 => {
          self.error_message = "invalid string: control character U+0017 (ETB) must be escaped to \\u0017";
          return token_type::parse_error;
        }
        0x18 => {
          self.error_message = "invalid string: control character U+0018 (CAN) must be escaped to \\u0018";
          return token_type::parse_error;
        }
        0x19 => {
          self.error_message = "invalid string: control character U+0019 (EM) must be escaped to \\u0019";
          return token_type::parse_error;
        }
        0x1A => {
          self.error_message = "invalid string: control character U+001A (SUB) must be escaped to \\u001A";
          return token_type::parse_error;
        }
        0x1B => {
          self.error_message = "invalid string: control character U+001B (ESC) must be escaped to \\u001B";
          return token_type::parse_error;
        }
        0x1C => {
          self.error_message = "invalid string: control character U+001C (FS) must be escaped to \\u001C";
          return token_type::parse_error;
        }
        0x1D => {
          self.error_message = "invalid string: control character U+001D (GS) must be escaped to \\u001D";
          return token_type::parse_error;
        }
        0x1E => {
          self.error_message = "invalid string: control character U+001E (RS) must be escaped to \\u001E";
          return token_type::parse_error;
        }
        0x1F => {
          self.error_message = "invalid string: control character U+001F (US) must be escaped to \\u001F";
          return token_type::parse_error;
        }

        // U+0020..U+007F (except U+0022 (quote) and U+005C (backspace))
        0x20..=0x21 | 0x23..=0x5B | 0x5D..=0x7F => {
          self.add(self.current);
        }

        // U+0080..U+07FF: bytes C2..DF 80..BF
        0xC2..=0xDF => {
          if !self.next_byte_in_range(&[0x80, 0xBF]) {
            return token_type::parse_error;
          }
        }

        // U+0800..U+0FFF: bytes E0 A0..BF 80..BF
        0xE0 => {
          if !(self.next_byte_in_range(&[0xA0, 0xBF, 0x80, 0xBF])) {
            return token_type::parse_error;
          }
        }

        // U+1000..U+CFFF: bytes E1..EC 80..BF 80..BF
        // U+E000..U+FFFF: bytes EE..EF 80..BF 80..BF
        0xE1..=0xEC | 0xEE..=0xEF => {
          if !(self.next_byte_in_range(&[0x80, 0xBF, 0x80, 0xBF])) {
            return token_type::parse_error;
          }
        }

        // U+D000..U+D7FF: bytes ED 80..9F 80..BF
        0xED => {
          if !(self.next_byte_in_range(&[0x80, 0x9F, 0x80, 0xBF])) {
            return token_type::parse_error;
          }
        }

        // U+10000..U+3FFFF F0 90..BF 80..BF 80..BF
        0xF0 => {
          if !(self.next_byte_in_range(&[0x90, 0xBF, 0x80, 0xBF, 0x80, 0xBF])) {
            return token_type::parse_error;
          }
        }

        // U+40000..U+FFFFF F1..F3 80..BF 80..BF 80..BF
        0xF1..=0xF3 => {
          if !(self.next_byte_in_range(&[0x80, 0xBF, 0x80, 0xBF, 0x80, 0xBF])) {
            return token_type::parse_error;
          }
        }

        // U+100000..U+10FFFF F4 80..8F 80..BF 80..BF
        0xF4 => {
          if !(self.next_byte_in_range(&[0x80, 0x8F, 0x80, 0xBF, 0x80, 0xBF])) {
            return token_type::parse_error;
          }
        }

        // remaining bytes (80..C1 and F5..FF) are ill-formed
        _ => {
          self.error_message = "invalid string: ill-formed UTF-8 byte";
          return token_type::parse_error;
        }
      }
    }
  }

  /// upstream: nlohmann::detail::lexer::strtof(double&, const char*, char**) (json.hpp)
  /// PORT NOTE: `std::strtod` → Rust's `f64::from_str`. Both are correctly rounded
  /// (round-to-nearest-even, glibc and core::num::dec2flt), overflow gives ±inf and underflow
  /// ±0/subnormal in both. Only RFC 7159 number tokens reach this (scan_number's DFA), so hex,
  /// inf and nan spellings never do. The `endptr` check upstream is an assert.
  fn strtof(str_: &[u8]) -> f64 {
    let s = ::core::str::from_utf8(str_).expect("upstream: scan_number only stores ASCII digits, '-', '+', '.', 'e', 'E'");
    s.parse::<f64>().expect("upstream: scan_number checked the number format before (assert(endptr == ...))")
  }

  /// upstream: nlohmann::detail::lexer::scan_number (json.hpp)
  /// scan a number literal
  ///
  /// This function scans a string according to Sect. 6 of RFC 7159.
  ///
  /// The function is realized with a deterministic finite state machine derived
  /// from the grammar described in RFC 7159. Starting in state "init", the
  /// input is read and used to determined the next state. Only state "done"
  /// accepts the number. State "error" is a trap state to model errors. In the
  /// table below, "anything" means any character but the ones listed before.
  ///
  /// state    | 0        | 1-9      | e E      | +       | -       | .        | anything
  /// ---------|----------|----------|----------|---------|---------|----------|-----------
  /// init     | zero     | any1     | [error]  | [error] | minus   | [error]  | [error]
  /// minus    | zero     | any1     | [error]  | [error] | [error] | [error]  | [error]
  /// zero     | done     | done     | exponent | done    | done    | decimal1 | done
  /// any1     | any1     | any1     | exponent | done    | done    | decimal1 | done
  /// decimal1 | decimal2 | decimal2 | [error]  | [error] | [error] | [error]  | [error]
  /// decimal2 | decimal2 | decimal2 | exponent | done    | done    | done     | done
  /// exponent | any2     | any2     | [error]  | sign    | sign    | [error]  | [error]
  /// sign     | any2     | any2     | [error]  | [error] | [error] | [error]  | [error]
  /// any2     | any2     | any2     | done     | done    | done    | done     | done
  ///
  /// The state machine is realized with one label per state (prefixed with
  /// "scan_number_") and `goto` statements between them. The state machine
  /// contains cycles, but any cycle can be left when EOF is read. Therefore,
  /// the function is guaranteed to terminate.
  ///
  /// During scanning, the read bytes are stored in token_buffer. This string is
  /// then converted to a signed integer, an unsigned integer, or a
  /// floating-point number.
  ///
  /// @return token_type::value_unsigned, token_type::value_integer, or
  ///         token_type::value_float if number could be successfully scanned,
  ///         token_type::parse_error otherwise
  ///
  /// @note The scanner is independent of the current locale. Internally, the
  ///       locale's decimal point is used instead of `.` to work with the
  ///       locale-dependent converters.
  /// PORT NOTE: the `goto` labels become a `state` enum driven by a loop; every transition,
  /// add() and error message is unchanged.
  fn scan_number(&mut self) -> token_type {
    #[derive(Clone, Copy)]
    enum State {
      minus,
      zero,
      any1,
      decimal1,
      decimal2,
      exponent,
      sign,
      any2,
      done,
    }

    // reset token_buffer to store the number's bytes
    self.reset();

    // the type of the parsed number; initially set to unsigned; will be
    // changed if minus sign, decimal point or exponent is read
    let mut number_type = token_type::value_unsigned;

    // state (init): we just found out we need to scan a number
    let mut state = match self.current {
      0x2D => {
        self.add(self.current);
        State::minus
      }

      0x30 => {
        self.add(self.current);
        State::zero
      }

      0x31..=0x39 => {
        self.add(self.current);
        State::any1
      }

      // all other characters are rejected outside scan_number()
      _ => unreachable!(), // LCOV_EXCL_LINE
    };

    loop {
      state = match state {
        State::minus => {
          // state: we just parsed a leading minus sign
          number_type = token_type::value_integer;
          match self.get() {
            0x30 => {
              self.add(self.current);
              State::zero
            }

            0x31..=0x39 => {
              self.add(self.current);
              State::any1
            }

            _ => {
              self.error_message = "invalid number; expected digit after '-'";
              return token_type::parse_error;
            }
          }
        }

        State::zero => {
          // state: we just parse a zero (maybe with a leading minus sign)
          match self.get() {
            0x2E => {
              self.add(self.decimal_point_char);
              State::decimal1
            }

            0x65 | 0x45 => {
              self.add(self.current);
              State::exponent
            }

            _ => State::done,
          }
        }

        State::any1 => {
          // state: we just parsed a number 0-9 (maybe with a leading minus sign)
          match self.get() {
            0x30..=0x39 => {
              self.add(self.current);
              State::any1
            }

            0x2E => {
              self.add(self.decimal_point_char);
              State::decimal1
            }

            0x65 | 0x45 => {
              self.add(self.current);
              State::exponent
            }

            _ => State::done,
          }
        }

        State::decimal1 => {
          // state: we just parsed a decimal point
          number_type = token_type::value_float;
          match self.get() {
            0x30..=0x39 => {
              self.add(self.current);
              State::decimal2
            }

            _ => {
              self.error_message = "invalid number; expected digit after '.'";
              return token_type::parse_error;
            }
          }
        }

        State::decimal2 => {
          // we just parsed at least one number after a decimal point
          match self.get() {
            0x30..=0x39 => {
              self.add(self.current);
              State::decimal2
            }

            0x65 | 0x45 => {
              self.add(self.current);
              State::exponent
            }

            _ => State::done,
          }
        }

        State::exponent => {
          // we just parsed an exponent
          number_type = token_type::value_float;
          match self.get() {
            0x2B | 0x2D => {
              self.add(self.current);
              State::sign
            }

            0x30..=0x39 => {
              self.add(self.current);
              State::any2
            }

            _ => {
              self.error_message = "invalid number; expected '+', '-', or digit after exponent";
              return token_type::parse_error;
            }
          }
        }

        State::sign => {
          // we just parsed an exponent sign
          match self.get() {
            0x30..=0x39 => {
              self.add(self.current);
              State::any2
            }

            _ => {
              self.error_message = "invalid number; expected digit after exponent sign";
              return token_type::parse_error;
            }
          }
        }

        State::any2 => {
          // we just parsed a number after the exponent or exponent sign
          match self.get() {
            0x30..=0x39 => {
              self.add(self.current);
              State::any2
            }

            _ => State::done,
          }
        }

        State::done => break,
      };
    }

    // scan_number_done:
    // unget the character after the number (we only read it to know that
    // we are done scanning a number)
    self.unget();

    // try to parse integers first and fall back to floats
    // PORT NOTE: `std::strtoull`/`std::strtoll` with `errno` → `u64`/`i64::from_str`. The token is
    // only digits (with a leading '-' for value_integer), so the only error is ERANGE (overflow),
    // which is exactly when from_str fails.
    if number_type == token_type::value_unsigned {
      let x: Result<u64, _> = ::core::str::from_utf8(&self.token_buffer).expect("ASCII digits").parse::<u64>();

      if let Ok(x) = x {
        self.value_unsigned = x;
        if self.value_unsigned == x {
          return token_type::value_unsigned;
        }
      }
    } else if number_type == token_type::value_integer {
      let x: Result<i64, _> = ::core::str::from_utf8(&self.token_buffer).expect("ASCII digits").parse::<i64>();

      if let Ok(x) = x {
        self.value_integer = x;
        if self.value_integer == x {
          return token_type::value_integer;
        }
      }
    }

    // this code is reached if we parse a floating-point number or if an
    // integer conversion above failed
    self.value_float = lexer::strtof(&self.token_buffer);

    token_type::value_float
  }

  /// upstream: nlohmann::detail::lexer::scan_literal (json.hpp)
  /// @param[in] literal_text  the literal text to expect
  /// @param[in] length        the length of the passed literal text
  /// @param[in] return_type   the token type to return on success
  fn scan_literal(&mut self, literal_text: &[u8], length: usize, return_type: token_type) -> token_type {
    debug_assert!(self.current == literal_text[0] as i32);
    for i in 1..length {
      // PORT NOTE: `to_char_type(get()) != literal_text[i]`; EOF (-1) never equals an ASCII byte.
      if self.get() != literal_text[i] as i32 {
        self.error_message = "invalid literal";
        return token_type::parse_error;
      }
    }
    return_type
  }

  /////////////////////
  // input management
  /////////////////////

  /// upstream: nlohmann::detail::lexer::reset (json.hpp)
  /// reset token_buffer; current character is beginning of token
  fn reset(&mut self) {
    self.token_buffer.clear();
    self.token_string.clear();
    self.token_string.push(self.current as u8);
  }

  /// upstream: nlohmann::detail::lexer::get (json.hpp)
  /// get next character from the input
  ///
  /// This function provides the interface to the used input adapter. It does
  /// not throw in case the input reached EOF, but returns a
  /// `std::char_traits<char>::eof()` in that case.  Stores the scanned characters
  /// for use in error messages.
  ///
  /// @return character read from the input
  fn get(&mut self) -> i32 {
    self.position.chars_read_total += 1;
    self.position.chars_read_current_line += 1;

    if self.next_unget {
      // just reset the next_unget variable and work with current
      self.next_unget = false;
    } else {
      self.current = self.ia.get_character();
    }

    if self.current != EOF {
      self.token_string.push(self.current as u8);
    }

    if self.current == b'\n' as i32 {
      self.position.lines_read += 1;
      self.position.chars_read_current_line = 0;
    }

    self.current
  }

  /// upstream: nlohmann::detail::lexer::unget (json.hpp)
  /// unget current character (read it again on next get)
  ///
  /// We implement unget by setting variable next_unget to true. The input is not
  /// changed - we just simulate ungetting by modifying chars_read_total,
  /// chars_read_current_line, and token_string. The next call to get() will
  /// behave as if the unget character is read again.
  fn unget(&mut self) {
    self.next_unget = true;

    self.position.chars_read_total -= 1;

    // in case we "unget" a newline, we have to also decrement the lines_read
    if self.position.chars_read_current_line == 0 {
      if self.position.lines_read > 0 {
        self.position.lines_read -= 1;
      }
    } else {
      self.position.chars_read_current_line -= 1;
    }

    if self.current != EOF {
      debug_assert!(!self.token_string.is_empty());
      self.token_string.pop();
    }
  }

  /// upstream: nlohmann::detail::lexer::add (json.hpp)
  /// add a character to token_buffer
  fn add(&mut self, c: i32) {
    self.token_buffer.push(c as u8);
  }

  /////////////////////
  // value getters
  /////////////////////

  /// upstream: nlohmann::detail::lexer::get_number_integer (json.hpp)
  /// return integer value
  fn get_number_integer(&self) -> i64 {
    self.value_integer
  }

  /// upstream: nlohmann::detail::lexer::get_number_unsigned (json.hpp)
  /// return unsigned integer value
  fn get_number_unsigned(&self) -> u64 {
    self.value_unsigned
  }

  /// upstream: nlohmann::detail::lexer::get_number_float (json.hpp)
  /// return floating-point value
  fn get_number_float(&self) -> f64 {
    self.value_float
  }

  /// upstream: nlohmann::detail::lexer::get_string (json.hpp)
  /// return current string value (implicitly resets the token; useful only once)
  fn get_string(&mut self) -> &mut Vec<u8> {
    &mut self.token_buffer
  }

  /////////////////////
  // diagnostics
  /////////////////////

  /// upstream: nlohmann::detail::lexer::get_position (json.hpp)
  /// return position of last read token
  fn get_position(&self) -> position_t {
    self.position
  }

  /// upstream: nlohmann::detail::lexer::get_token_string (json.hpp)
  /// return the last read token (for errors only).  Will never contain EOF
  /// (an arbitrary value that is not a valid char value, often -1), because
  /// 255 may legitimately occur.  May contain NUL, which should be escaped.
  fn get_token_string(&self) -> Vec<u8> {
    // escape control characters
    let mut result: Vec<u8> = Vec::new();
    for &c in &self.token_string {
      if c <= 0x1F {
        // escape control characters
        result.extend_from_slice(crate::strprintf!("<U+%.4X>", c as i32).as_bytes());
      } else {
        // add character as is
        result.push(c);
      }
    }

    result
  }

  /// upstream: nlohmann::detail::lexer::get_error_message (json.hpp)
  /// return syntax error message
  fn get_error_message(&self) -> &'static str {
    self.error_message
  }

  /////////////////////
  // actual scanner
  /////////////////////

  /// upstream: nlohmann::detail::lexer::skip_bom (json.hpp)
  /// skip the UTF-8 byte order mark
  /// @return true iff there is no BOM or the correct BOM has been skipped
  fn skip_bom(&mut self) -> bool {
    if self.get() == 0xEF {
      // check if we completely parse the BOM
      return self.get() == 0xBB && self.get() == 0xBF;
    }

    // the first character is not the beginning of the BOM; unget it to
    // process is later
    self.unget();
    true
  }

  /// upstream: nlohmann::detail::lexer::scan (json.hpp)
  fn scan(&mut self) -> token_type {
    // initially, skip the BOM
    if self.position.chars_read_total == 0 && !self.skip_bom() {
      self.error_message = "invalid BOM; must be 0xEF 0xBB 0xBF if given";
      return token_type::parse_error;
    }

    // read next character and ignore whitespace
    loop {
      self.get();
      if !(self.current == b' ' as i32 || self.current == b'\t' as i32 || self.current == b'\n' as i32 || self.current == b'\r' as i32) {
        break;
      }
    }

    match self.current {
      // structural characters
      0x5B => token_type::begin_array,
      0x5D => token_type::end_array,
      0x7B => token_type::begin_object,
      0x7D => token_type::end_object,
      0x3A => token_type::name_separator,
      0x2C => token_type::value_separator,

      // literals
      0x74 => {
        let true_literal: [u8; 4] = *b"true";
        self.scan_literal(&true_literal, true_literal.len(), token_type::literal_true)
      }
      0x66 => {
        let false_literal: [u8; 5] = *b"false";
        self.scan_literal(&false_literal, false_literal.len(), token_type::literal_false)
      }
      0x6E => {
        let null_literal: [u8; 4] = *b"null";
        self.scan_literal(&null_literal, null_literal.len(), token_type::literal_null)
      }

      // string
      0x22 => self.scan_string(),

      // number
      0x2D | 0x30..=0x39 => self.scan_number(),

      // end of input (the null byte is needed when parsing from
      // string literals)
      0x00 | EOF => token_type::end_of_input,

      // error
      _ => {
        self.error_message = "invalid literal";
        token_type::parse_error
      }
    }
  }
}

/// upstream: nlohmann::json_sax (json.hpp)
/// SAX interface the parser reports to (the methods `sax_parse_internal` calls).
/// PORT NOTE: `binary` is omitted: only the binary formats (CBOR, MessagePack…) call it.
/// `parse_error` returns `Err` where the C++ SAX throws.
pub trait Sax {
  /// a null value was read
  fn null(&mut self) -> bool;
  /// a boolean value was read
  fn boolean(&mut self, val: bool) -> bool;
  /// an integer number was read
  fn number_integer(&mut self, val: i64) -> bool;
  /// an unsigned integer number was read
  fn number_unsigned(&mut self, val: u64) -> bool;
  /// an floating-point number was read (`s` is the raw token)
  fn number_float(&mut self, val: f64, s: &[u8]) -> bool;
  /// a string was read
  fn string(&mut self, val: &mut Vec<u8>) -> bool;
  /// the beginning of an object was read (`len` is `usize::MAX` = `std::size_t(-1)` if unknown)
  fn start_object(&mut self, len: usize) -> bool;
  /// an object key was read
  fn key(&mut self, val: &mut Vec<u8>) -> bool;
  /// the end of an object was read
  fn end_object(&mut self) -> bool;
  /// the beginning of an array was read
  fn start_array(&mut self, len: usize) -> bool;
  /// the end of an array was read
  fn end_array(&mut self) -> bool;
  /// a parse error occurred (`position` = chars_read_total, `last_token` = get_token_string())
  fn parse_error(&mut self, position: usize, last_token: &[u8], ex: &Exception) -> Result<bool, KataError>;
}

/// upstream: nlohmann::detail::json_sax_dom_parser (json.hpp)
/// SAX implementation to create a JSON value from SAX events
/// PORT NOTE: upstream keeps raw pointers into the tree being built (`ref_stack`,
/// `object_element`) and inserts each container into its parent when it starts. Rust can't hold
/// those aliases, so `ref_stack` owns the open containers (an object with its pending key) and a
/// container is inserted into its parent when it ends. The finished value is the same: an
/// error throws and the partial tree is discarded either way, and a repeated key is
/// overwritten (`object->operator[](key) = value`, last one wins).
struct json_sax_dom_parser<'a> {
  root: &'a mut Value,
  ref_stack: Vec<DomRef>,
  errored: bool,
  allow_exceptions: bool,
  /// PORT NOTE: the exception parse_error throws, kept with its C++ bytes (see `parse_bytes`).
  thrown: Option<Exception>,
}

/// An open container on json_sax_dom_parser's `ref_stack`.
enum DomRef {
  Array(Vec<Value>),
  /// the object and its `object_element` (the key whose value comes next)
  Object(Map<String, Value>, Option<String>),
}

/// The DOM's string_t is a Rust String; the lexer's buffer is always valid UTF-8 (scan_string
/// rejects ill-formed UTF-8 bytes exactly like upstream, and `\u` escapes are checked for
/// surrogates), so this cannot fail.
fn dom_string(val: &[u8]) -> String {
  String::from_utf8(val.to_vec()).expect("lexer token_buffer is valid UTF-8: scan_string checks UTF-8 and \\u escapes are surrogate-checked")
}

impl<'a> json_sax_dom_parser<'a> {
  /// upstream: nlohmann::detail::json_sax_dom_parser::json_sax_dom_parser (json.hpp)
  /// @param[in, out] r  reference to a JSON value that is manipulated while
  ///                    parsing
  /// @param[in] allow_exceptions_  whether parse errors yield exceptions
  fn new(r: &'a mut Value, allow_exceptions_: bool) -> json_sax_dom_parser<'a> {
    json_sax_dom_parser { root: r, ref_stack: Vec::new(), errored: false, allow_exceptions: allow_exceptions_, thrown: None }
  }

  /// upstream: nlohmann::detail::json_sax_dom_parser::is_errored (json.hpp)
  fn is_errored(&self) -> bool {
    self.errored
  }

  /// upstream: nlohmann::detail::json_sax_dom_parser::handle_value (json.hpp)
  /// @invariant If the ref stack is empty, then the passed value will be the new
  ///            root.
  /// @invariant If the ref stack contains a value, then it is an array or an
  ///            object to which we can add elements
  fn handle_value(&mut self, v: Value) {
    match self.ref_stack.last_mut() {
      None => {
        *self.root = v;
      }
      Some(DomRef::Array(a)) => {
        a.push(v);
      }
      Some(DomRef::Object(o, object_element)) => {
        let k = object_element.take().expect("upstream: assert(object_element)");
        o.insert(k, v);
      }
    }
  }
}

impl<'a> Sax for json_sax_dom_parser<'a> {
  fn null(&mut self) -> bool {
    self.handle_value(Value::Null);
    true
  }

  fn boolean(&mut self, val: bool) -> bool {
    self.handle_value(Value::Bool(val));
    true
  }

  fn number_integer(&mut self, val: i64) -> bool {
    self.handle_value(Value::Number(Number::from(val)));
    true
  }

  fn number_unsigned(&mut self, val: u64) -> bool {
    self.handle_value(Value::Number(Number::from(val)));
    true
  }

  fn number_float(&mut self, val: f64, _s: &[u8]) -> bool {
    self.handle_value(Value::Number(Number::from_f64(val).expect("upstream: sax_parse_internal rejects non-finite floats (out_of_range.406)")));
    true
  }

  fn string(&mut self, val: &mut Vec<u8>) -> bool {
    self.handle_value(Value::String(dom_string(val)));
    true
  }

  fn start_object(&mut self, len: usize) -> bool {
    self.ref_stack.push(DomRef::Object(Map::new(), None));

    // PORT NOTE: `len > max_size()` (out_of_range.408) cannot happen: the JSON parser always
    // passes `std::size_t(-1)`.
    debug_assert!(len == usize::MAX);
    true
  }

  fn key(&mut self, val: &mut Vec<u8>) -> bool {
    // PORT NOTE: upstream `object->operator[](val)` inserts a null here; the value replaces it
    // before the tree can be observed, so the key is held until handle_value.
    match self.ref_stack.last_mut() {
      Some(DomRef::Object(_, object_element)) => *object_element = Some(dom_string(val)),
      _ => unreachable!("upstream: key() is only called inside an object"),
    }
    true
  }

  fn end_object(&mut self) -> bool {
    match self.ref_stack.pop() {
      Some(DomRef::Object(o, _)) => self.handle_value(Value::Object(o)),
      _ => unreachable!("upstream: end_object() closes an object"),
    }
    true
  }

  fn start_array(&mut self, len: usize) -> bool {
    self.ref_stack.push(DomRef::Array(Vec::new()));

    // PORT NOTE: see start_object (out_of_range.408 unreachable).
    debug_assert!(len == usize::MAX);
    true
  }

  fn end_array(&mut self) -> bool {
    match self.ref_stack.pop() {
      Some(DomRef::Array(a)) => self.handle_value(Value::Array(a)),
      _ => unreachable!("upstream: end_array() closes an array"),
    }
    true
  }

  fn parse_error(&mut self, _position: usize, _last_token: &[u8], ex: &Exception) -> Result<bool, KataError> {
    self.errored = true;
    if self.allow_exceptions {
      // PORT NOTE: the C++ rethrows by id class (1 parse_error, 4 out_of_range); all are
      // KataError::NlohmannJson. The exception is also kept with its bytes (`parse_bytes`).
      self.thrown = Some(Exception { id: ex.id, m: ex.m.clone() });
      return Err(Exception { id: ex.id, m: ex.m.clone() }.into_kata_error());
    }
    Ok(false)
  }
}

/// upstream: nlohmann::detail::parser (json.hpp)
/// syntax analysis
///
/// This class implements a recursive descent parser.
/// PORT NOTE: no parser callback (KataGo never passes one); allow_exceptions is always true.
struct parser<'a> {
  /// the type of the last read token
  last_token: token_type,
  /// the lexer
  m_lexer: lexer<'a>,
  /// whether to throw exceptions in case of errors
  allow_exceptions: bool,
}

impl<'a> parser<'a> {
  /// upstream: nlohmann::detail::parser::parser (json.hpp)
  /// a parser reading from an input adapter
  fn new(adapter: iterator_input_adapter<'a>, allow_exceptions_: bool) -> parser<'a> {
    let mut p = parser { last_token: token_type::uninitialized, m_lexer: lexer::new(adapter), allow_exceptions: allow_exceptions_ };
    // read first token
    p.get_token();
    p
  }

  /// upstream: nlohmann::detail::parser::parse (json.hpp)
  /// public parser interface
  ///
  /// @param[in] strict      whether to expect the last token to be EOF
  /// @param[in,out] result  parsed JSON value
  ///
  /// @throw parse_error.101 in case of an unexpected token
  /// @throw parse_error.102 if to_unicode fails or surrogate error
  /// @throw parse_error.103 if to_unicode fails
  /// PORT NOTE: only the `callback == nullptr` branch. The throw is `Err(Exception)` (C++ bytes).
  fn parse(&mut self, strict: bool, result: &mut Value) -> Result<(), Exception> {
    let mut sdp = json_sax_dom_parser::new(result, self.allow_exceptions);
    // PORT NOTE: every Err of the DOM parser comes from its parse_error, which kept the exception.
    let thrown = |sdp: &mut json_sax_dom_parser, e: KataError| -> Exception {
      sdp.thrown.take().unwrap_or_else(|| Exception { id: -1, m: e.what().as_bytes().to_vec() })
    };
    if let Err(e) = self.sax_parse_internal(&mut sdp) {
      return Err(thrown(&mut sdp, e));
    }

    // in strict mode, input must be completely read
    if strict && (self.get_token() != token_type::end_of_input) {
      let ex = Exception::parse_error_create(101, &self.m_lexer.get_position(), &self.exception_message(token_type::end_of_input, "value"));
      if let Err(e) = sdp.parse_error(self.m_lexer.get_position().chars_read_total, &self.m_lexer.get_token_string(), &ex) {
        return Err(thrown(&mut sdp, e));
      }
    }

    // in case of an error, return discarded value
    if sdp.is_errored() {
      // PORT NOTE: `result = value_t::discarded` has no serde_json equivalent; with
      // allow_exceptions = true parse_error already returned Err, so this is unreachable.
      unreachable!("allow_exceptions is always true: parse_error returned Err");
    }
    Ok(())
  }

  /// upstream: nlohmann::detail::parser::sax_parse (json.hpp)
  fn sax_parse<S: Sax>(&mut self, sax: &mut S, strict: bool) -> Result<bool, KataError> {
    let result = self.sax_parse_internal(sax)?;

    // strict mode: next byte must be EOF
    if result && strict && (self.get_token() != token_type::end_of_input) {
      let ex = Exception::parse_error_create(101, &self.m_lexer.get_position(), &self.exception_message(token_type::end_of_input, "value"));
      return sax.parse_error(self.m_lexer.get_position().chars_read_total, &self.m_lexer.get_token_string(), &ex);
    }

    Ok(result)
  }

  /// `return sax->parse_error(m_lexer.get_position(), m_lexer.get_token_string(), parse_error::create(101, m_lexer.get_position(), exception_message(expected, context)));`
  fn sax_parse_error_101<S: Sax>(&mut self, sax: &mut S, expected: token_type, context: &str) -> Result<bool, KataError> {
    let ex = Exception::parse_error_create(101, &self.m_lexer.get_position(), &self.exception_message(expected, context));
    sax.parse_error(self.m_lexer.get_position().chars_read_total, &self.m_lexer.get_token_string(), &ex)
  }

  /// upstream: nlohmann::detail::parser::sax_parse_internal (json.hpp)
  /// PORT NOTE: iterative like upstream (explicit `states` stack), so there is no nesting limit.
  /// TODO(port): serde_json's Value drops recursively, so a parsed value nested ~10^5 deep can
  /// overflow the stack when dropped (upstream's basic_json destructor recurses too).
  fn sax_parse_internal<S: Sax>(&mut self, sax: &mut S) -> Result<bool, KataError> {
    // stack to remember the hierarchy of structured values we are parsing
    // true = array; false = object
    let mut states: Vec<bool> = Vec::new();
    // value to avoid a goto (see comment where set to true)
    let mut skip_to_state_evaluation = false;

    loop {
      if !skip_to_state_evaluation {
        // invariant: get_token() was called before each iteration
        match self.last_token {
          token_type::begin_object => {
            if !sax.start_object(usize::MAX) {
              return Ok(false);
            }

            // closing } -> we are done
            if self.get_token() == token_type::end_object {
              if !sax.end_object() {
                return Ok(false);
              }
            } else {
              // parse key
              if self.last_token != token_type::value_string {
                return self.sax_parse_error_101(sax, token_type::value_string, "object key");
              }
              if !sax.key(self.m_lexer.get_string()) {
                return Ok(false);
              }

              // parse separator (:)
              if self.get_token() != token_type::name_separator {
                return self.sax_parse_error_101(sax, token_type::name_separator, "object separator");
              }

              // remember we are now inside an object
              states.push(false);

              // parse values
              self.get_token();
              continue;
            }
          }

          token_type::begin_array => {
            if !sax.start_array(usize::MAX) {
              return Ok(false);
            }

            // closing ] -> we are done
            if self.get_token() == token_type::end_array {
              if !sax.end_array() {
                return Ok(false);
              }
            } else {
              // remember we are now inside an array
              states.push(true);

              // parse values (no need to call get_token)
              continue;
            }
          }

          token_type::value_float => {
            let res = self.m_lexer.get_number_float();

            if !res.is_finite() {
              let tok = self.m_lexer.get_token_string();
              let mut what: Vec<u8> = b"number overflow parsing '".to_vec();
              what.extend_from_slice(&tok);
              what.extend_from_slice(b"'");
              let ex = Exception::out_of_range_create(406, &what);
              return sax.parse_error(self.m_lexer.get_position().chars_read_total, &tok, &ex);
            }

            let s: Vec<u8> = self.m_lexer.get_string().clone();
            if !sax.number_float(res, &s) {
              return Ok(false);
            }
          }

          token_type::literal_false => {
            if !sax.boolean(false) {
              return Ok(false);
            }
          }

          token_type::literal_null => {
            if !sax.null() {
              return Ok(false);
            }
          }

          token_type::literal_true => {
            if !sax.boolean(true) {
              return Ok(false);
            }
          }

          token_type::value_integer => {
            if !sax.number_integer(self.m_lexer.get_number_integer()) {
              return Ok(false);
            }
          }

          token_type::value_string => {
            if !sax.string(self.m_lexer.get_string()) {
              return Ok(false);
            }
          }

          token_type::value_unsigned => {
            if !sax.number_unsigned(self.m_lexer.get_number_unsigned()) {
              return Ok(false);
            }
          }

          token_type::parse_error => {
            // using "uninitialized" to avoid "expected" message
            return self.sax_parse_error_101(sax, token_type::uninitialized, "value");
          }

          _ => {
            // the last token was unexpected
            return self.sax_parse_error_101(sax, token_type::literal_or_value, "value");
          }
        }
      } else {
        skip_to_state_evaluation = false;
      }

      // we reached this line after we successfully parsed a value
      if states.is_empty() {
        // empty stack: we reached the end of the hierarchy: done
        return Ok(true);
      }

      if *states.last().expect("not empty") {
        // array
        // comma -> next value
        if self.get_token() == token_type::value_separator {
          // parse a new value
          self.get_token();
          continue;
        }

        // closing ]
        if self.last_token == token_type::end_array {
          if !sax.end_array() {
            return Ok(false);
          }

          // We are done with this array. Before we can parse a
          // new value, we need to evaluate the new state first.
          // By setting skip_to_state_evaluation to false, we
          // are effectively jumping to the beginning of this if.
          debug_assert!(!states.is_empty());
          states.pop();
          skip_to_state_evaluation = true;
          continue;
        }

        return self.sax_parse_error_101(sax, token_type::end_array, "array");
      } else {
        // object
        // comma -> next value
        if self.get_token() == token_type::value_separator {
          // parse key
          if self.get_token() != token_type::value_string {
            return self.sax_parse_error_101(sax, token_type::value_string, "object key");
          }

          if !sax.key(self.m_lexer.get_string()) {
            return Ok(false);
          }

          // parse separator (:)
          if self.get_token() != token_type::name_separator {
            return self.sax_parse_error_101(sax, token_type::name_separator, "object separator");
          }

          // parse values
          self.get_token();
          continue;
        }

        // closing }
        if self.last_token == token_type::end_object {
          if !sax.end_object() {
            return Ok(false);
          }

          // We are done with this object. Before we can parse a
          // new value, we need to evaluate the new state first.
          // By setting skip_to_state_evaluation to false, we
          // are effectively jumping to the beginning of this if.
          debug_assert!(!states.is_empty());
          states.pop();
          skip_to_state_evaluation = true;
          continue;
        }

        return self.sax_parse_error_101(sax, token_type::end_object, "object");
      }
    }
  }

  /// upstream: nlohmann::detail::parser::get_token (json.hpp)
  /// get next token from lexer
  fn get_token(&mut self) -> token_type {
    self.last_token = self.m_lexer.scan();
    self.last_token
  }

  /// upstream: nlohmann::detail::parser::exception_message (json.hpp)
  fn exception_message(&self, expected: token_type, context: &str) -> Vec<u8> {
    let mut error_msg: Vec<u8> = b"syntax error ".to_vec();
    if !context.is_empty() {
      error_msg.extend_from_slice(("while parsing ".to_string() + context + " ").as_bytes());
    }

    error_msg.extend_from_slice(b"- ");

    if self.last_token == token_type::parse_error {
      error_msg.extend_from_slice(self.m_lexer.get_error_message().as_bytes());
      error_msg.extend_from_slice(b"; last read: '");
      error_msg.extend_from_slice(&self.m_lexer.get_token_string());
      error_msg.extend_from_slice(b"'");
    } else {
      error_msg.extend_from_slice(("unexpected ".to_string() + token_type_name(self.last_token)).as_bytes());
    }

    if expected != token_type::uninitialized {
      error_msg.extend_from_slice(("; expected ".to_string() + token_type_name(expected)).as_bytes());
    }

    error_msg
  }
}

/// upstream: nlohmann::basic_json::parse(InputType&&, const parser_callback_t, const bool) (json.hpp)
/// `json::parse(s)`: deserialize from a compatible input (strict, no callback, exceptions on).
/// Errors are KataError::NlohmannJson(e.what()): parse_error.101 and out_of_range.406.
/// Numbers: no '-' and fits u64 → u64; '-' and fits i64 → i64 (so `-0` is integer 0 and dumps
/// "0"); otherwise f64 (strtod-exact). A NUL byte outside a string ends the input, as upstream.
/// PORT NOTE: nlohmann tells number_integer from number_unsigned; serde_json stores every
/// non-negative integer as u64, so the only value whose class changes is integer `-0`
/// (`is_number_unsigned()` false upstream, `is_u64()` true here). `sax_parse` keeps the classes.
pub fn parse(s: &str) -> Result<Value, KataError> {
  parse_bytes(s.as_bytes()).map_err(Exception::into_kata_error)
}

/// upstream: nlohmann::basic_json::parse(InputType&&, const parser_callback_t, const bool) (json.hpp)
/// `json::parse(s)` over the raw bytes of a std::string (any bytes, as `getline` returns them):
/// as `parse`, but the throw is the Exception itself, whose `what()` keeps the C++ bytes (a
/// "last read" token string may hold invalid UTF-8 or end inside a multi-byte character).
/// Ill-formed UTF-8 inside a string is a parse_error.101 ("invalid string: ill-formed UTF-8
/// byte"), so the Value's strings are always valid UTF-8.
pub fn parse_bytes(s: &[u8]) -> Result<Value, Exception> {
  let mut result = Value::Null;
  parser::new(iterator_input_adapter { current: 0, input: s }, true).parse(true, &mut result)?;
  Ok(result)
}

/// upstream: nlohmann::basic_json::sax_parse(InputType&&, SAX*, input_format_t, const bool) (json.hpp)
/// JSON text only (`format = input_format_t::json`).
pub fn sax_parse<S: Sax>(s: &str, sax: &mut S, strict: bool) -> Result<bool, KataError> {
  parser::new(iterator_input_adapter { current: 0, input: s.as_bytes() }, true).sax_parse(sax, strict)
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn floats_like_nlohmann() {
    let f = |x: f64| dump(&serde_json::json!(x));
    assert_eq!(f(9.0), "9.0");
    assert_eq!(f(7.5), "7.5");
    assert_eq!(f(-0.0), "-0.0");
    assert_eq!(f(0.65662659), "0.65662659");
    assert_eq!(f(4.12549165e-05), "4.12549165e-05");
    assert_eq!(f(0.1f32 as f64), "0.10000000149011612");
    assert_eq!(f(1e16), "1e+16");
    assert_eq!(f(123456789012345.0), "123456789012345.0");
    assert_eq!(f(1e-5), "1e-05");
    assert_eq!(f(0.0001), "0.0001");
  }

  #[test]
  fn get_arithmetic_and_vectors() {
    let v = |t: &str| parse(t).ok().unwrap();
    assert_eq!(get_i64(&v("5")).ok(), Some(5));
    assert_eq!(get_i64(&v("-5")).ok(), Some(-5));
    assert_eq!(get_i64(&v("18446744073709551615")).ok(), Some(-1)); // static_cast<int64_t>(u64) wraps
    assert_eq!(get_i64(&v("2.9")).ok(), Some(2));
    assert_eq!(get_i64(&v("1e300")).ok(), Some(i64::MIN)); // x86-64 cvttsd2si
    assert!(get_i64(&v("true")).is_err()); // number_integer_t: no boolean
    assert_eq!(get_f64(&v("3")).ok(), Some(3.0));
    assert!(get_f64(&v("false")).is_err());
    assert!(get_f64(&v("\"x\"")).is_err());
    assert_eq!(get_vector_i32(&v("[1, true, 2.5, -3]")).ok(), Some(vec![1, 1, 2, -3])); // get<int> accepts booleans
    assert!(get_vector_i32(&v("{}")).is_err());
    assert!(get_vector_i32(&v("[1, \"a\"]")).is_err());
    assert_eq!(get_vector_i64(&v("[1, -2]")).ok(), Some(vec![1, -2]));
    assert!(get_vector_i64(&v("[true]")).is_err());
    assert!(is_number_integer(&v("3")) && is_number_integer(&v("-3")) && !is_number_integer(&v("3.0")));
  }
}
