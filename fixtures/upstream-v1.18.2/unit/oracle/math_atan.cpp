// Oracle for crates/katago-core/src/port/math/atan.rs (DECISIONS.md D13, option b):
//   - `cr_atan` (CORE-MATH, correctly rounded) against MPFR's correctly rounded atan on a fixed
//     pseudo-random input set (2^20 inputs);
//   - `atan` (cr_atan + GLIBC_ATAN_OVERRIDES) against glibc's atan on all 50,531 ScoreValue table
//     arguments x = (i/10.0 - 0.0)/(1.0*19.0), i in [-25265, 25265] (nninputs.cpp initTables);
//   - the table arguments where glibc is not correctly rounded (= the override table);
//   - special inputs (zeros, subnormals, infinities, NaN, every branch boundary of cr_atan,
//     CORE-MATH's hard cases, the override inputs and their neighbours) with both results.
// tests/math_atan.rs regenerates the same inputs and compares. Not upstream code.
// Build and run (from the repo root; -ffp-contract=off so the input expressions round like Rust):
//   g++ -std=c++17 -O2 -ffp-contract=off -fno-builtin \
//     -o /tmp/math_atan_oracle fixtures/upstream-v1.18.2/unit/oracle/math_atan.cpp -lmpfr -lgmp -lm
//   /tmp/math_atan_oracle > fixtures/upstream-v1.18.2/unit/oracle/math_atan.txt
// MUST run on an x86-64 CPU with FMA: glibc's atan resolves to __atan_fma there, the variant the
// KataGo fixtures were produced with (its non-FMA variant misrounds a different set of table
// arguments). Generated with glibc 2.44, MPFR 4.2.2, g++ 16.2.1, Intel i7-8700K.
// Output:
//   C <chunk> <hash>          FNV-1a-style fold of MPFR atan's result bits over 65536 inputs
//   T <chunk> <hash>          the same fold of glibc atan over table arguments 4096*chunk ..
//                             (the last chunk is shorter), in order of increasing i
//   D <x bits> <glibc bits> <mpfr bits>   table arguments where glibc != MPFR, sorted by x bits
//   S <x bits> <glibc bits> <mpfr bits>   special inputs, one per line
// NaN results are folded/printed as 7FF8000000000000 (payloads differ between x86 and wasm).
#include <mpfr.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <tuple>
#include <vector>

static uint64_t S;
static uint64_t sm() {
  uint64_t z = (S += 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}
static double u() { return (double)(sm() >> 11) * 0x1p-53; }
static double fb(uint64_t b) { double d; memcpy(&d, &b, 8); return d; }
static uint64_t bits(double d) { if(std::isnan(d)) return 0x7FF8000000000000ULL; uint64_t b; memcpy(&b, &d, 8); return b; }
static uint64_t mix(uint64_t h, uint64_t v) { return (h ^ v) * 1099511628211ULL; }

static const int CHUNK = 65536;
static const int CHUNKS = 16;
static const int TCHUNK = 4096;

// Correctly rounded binary64 atan (round to nearest), subnormals included.
static mpfr_t A, R;
static double mpfrAtan(double x) {
  if(std::isnan(x)) return x;
  mpfr_set_d(A, x, MPFR_RNDN);
  int t = mpfr_atan(R, A, MPFR_RNDN);
  t = mpfr_subnormalize(R, t, MPFR_RNDN);
  return mpfr_get_d(R, MPFR_RNDN);
}

// Input generator, mirrored in tests/math_atan.rs. i is the input index. Every sm()/u() call
// is its own statement so the draw order is the same in C++ and Rust.
static double genAtan(long i) {
  switch(i % 8) {
  case 0: return fb(sm()); // uniform bits
  case 1: return u() * 16.0 - 8.0;
  case 2: { // log-uniform 2^-60 .. 2^60, random sign and mantissa
    uint64_t e = sm() % 120;
    uint64_t m = sm();
    return fb((m & 0x800FFFFFFFFFFFFFULL) | ((e + 1023 - 60) << 52));
  }
  case 3: { // exponent in [-30, 20]
    uint64_t v = sm();
    return fb((v & 0x800FFFFFFFFFFFFFULL) | (((v % 51) + 1023 - 30) << 52));
  }
  case 4: { // ScoreValue-like: (score - center) / (scale * sqrtBoardArea)
    double k = (double)(int64_t)(sm() % 200001) - 100000.0;
    double s = u();
    return (k / 10.0) / ((s * 2.0 + 0.1) * 19.0);
  }
  case 5: { // around the small-|x| fast-path and 2^-27 boundaries
    uint64_t v = sm();
    return fb((v & 0x800FFFFFFFFFFFFFULL) | (((v % 21) + 1023 - 28) << 52));
  }
  case 6: { // large |x|: around 0x1.2ded8e34a9035p+7 and 0x1.d02967c31cdb5p+53
    uint64_t v = sm();
    return fb((v & 0x800FFFFFFFFFFFFFULL) | (((v % 50) + 1023 + 6) << 52));
  }
  default: return u() * 2.0 - 1.0;
  }
}

static double tableArg(int i) {
  double mean = (double)i / 10.0;
  return (mean - 0.0) / (1.0 * 19.0);
}

static void special(double x) {
  printf("S %016llX %016llX %016llX\n", (unsigned long long)bits(x), (unsigned long long)bits(atan(x)),
         (unsigned long long)bits(mpfrAtan(x)));
}

int main() {
  mpfr_set_emin(-1073);
  mpfr_set_emax(1024);
  mpfr_init2(A, 53);
  mpfr_init2(R, 53);

  S = 21;
  for(int c = 0; c < CHUNKS; c++) {
    uint64_t h = 1469598103934665603ULL;
    for(int k = 0; k < CHUNK; k++)
      h = mix(h, bits(mpfrAtan(genAtan((long)c * CHUNK + k))));
    printf("C %d %016llX\n", c, (unsigned long long)h);
  }

  std::vector<std::tuple<uint64_t, uint64_t, uint64_t>> dev;
  const int lo = -25265, hi = 25265;
  for(int c = 0; lo + c * TCHUNK <= hi; c++) {
    uint64_t h = 1469598103934665603ULL;
    for(int i = lo + c * TCHUNK; i < lo + (c + 1) * TCHUNK && i <= hi; i++) {
      double x = tableArg(i);
      double g = atan(x);
      double m = mpfrAtan(x);
      h = mix(h, bits(g));
      if(bits(g) != bits(m))
        dev.emplace_back(bits(x), bits(g), bits(m));
    }
    printf("T %d %016llX\n", c, (unsigned long long)h);
  }
  std::sort(dev.begin(), dev.end());
  for(auto& [x, g, m] : dev)
    printf("D %016llX %016llX %016llX\n", (unsigned long long)x, (unsigned long long)g, (unsigned long long)m);

  std::vector<uint64_t> xs = {
    0x0000000000000000ULL, 0x0000000000000001ULL, 0x0000000000000002ULL, 0x000FFFFFFFFFFFFFULL,
    0x0010000000000000ULL, 0x3FF0000000000000ULL, 0x7FEFFFFFFFFFFFFFULL, 0x7FF0000000000000ULL,
    0x7FF8000000000000ULL, 0x7FF0000000000001ULL,
  };
  // cr_atan's branch boundaries (|x| < 0x1.b21c475e6362ap-8, < 0x1p-27, > 0x1.2ded8e34a9035p+7,
  // >= 0x1.d02967c31cdb5p+53), 1, 2^-54, each +- 3 ulp
  for(uint64_t b : {0x3F7B21C475E6362AULL, 0x3E40000000000000ULL, 0x4062DED8E34A9035ULL, 0x434D02967C31CDB5ULL,
                    0x3FF0000000000000ULL, 0x3C90000000000000ULL})
    for(int d = -3; d <= 3; d++) xs.push_back(b + d);
  // CORE-MATH's hard cases (as_atan_refine2's db[] table), +- 1 ulp
  for(double h : {0x1.0dc89a3b5501p-7, 0x1.e3fb41d2d226p-8, 0x1.7ba49f739829fp-1, 0x1.a933fe176b375p-3,
                  0x1.bb04a79820063p-8, 0x1.cd30a9499618bp-8, 0x1.f44aa37b8e66bp-7, 0x1.fd2ac95e57ef9p-8,
                  0x1.6419079bbf601p-6, 0x1.d768804487b07p-3})
    for(int d = -1; d <= 1; d++) xs.push_back(bits(h) + d);
  // the table arguments where glibc misrounds, +- 1 ulp
  for(auto& [x, g, m] : dev)
    if(!(x >> 63))
      for(int d = -1; d <= 1; d++) xs.push_back(x + d);
  // tan(pi/256*j)-ish range-reduction points: j/32 for the integer j in [1, 4096]
  for(int j = 1; j <= 4096; j += 37) xs.push_back(bits((double)j / 32.0));
  for(uint64_t x : xs) {
    special(fb(x));
    special(fb(x ^ 0x8000000000000000ULL));
  }
  mpfr_clear(A);
  mpfr_clear(R);
  return 0;
}
