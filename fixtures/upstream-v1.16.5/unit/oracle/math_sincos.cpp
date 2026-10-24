// Oracle for crates/katago-core/src/port/math/sincos.rs: CORE-MATH's correctly rounded
// `cr_sincos` (the C file itself, compiled and linked in) on a fixed pseudo-random input set plus
// special inputs, each result checked against MPFR's correctly rounded mpfr_sin / mpfr_cos
// (53 bits, MPFR_RNDN, subnormalized): the generator aborts if CORE-MATH and MPFR ever disagree.
// tests/math_sincos.rs replays the file line by line. Not upstream code; no upstream sources
// needed. glibc's own sin/cos are not used (not correctly rounded, see port/math.rs).
// Build and run (from the repo root; CORE_MATH is a checkout of
// https://gitlab.inria.fr/core-math/core-math at commit
// 0ce0cbc15935e24c656280e5a46c34e89f301b7a). -ffp-contract=off and -fno-builtin so that the
// generator's own arithmetic rounds the way a plain C evaluation does (as Rust does); without
// -mfma, CORE-MATH's __builtin_fma becomes a call to libm's fma, which is exact:
//   gcc -O2 -ffp-contract=off -fno-builtin -c "$CORE_MATH/src/binary64/sincos/sincos.c" \
//     -o /tmp/cr_sincos.o
//   g++ -std=c++17 -O2 -ffp-contract=off -fno-builtin -o /tmp/math_sincos_oracle \
//     fixtures/upstream-v1.16.5/unit/oracle/math_sincos.cpp /tmp/cr_sincos.o -lmpfr -lgmp -lm
//   /tmp/math_sincos_oracle "$CORE_MATH/src/binary64/sincos/sincos.wc" \
//     > fixtures/upstream-v1.16.5/unit/oracle/math_sincos.txt
// Generated with glibc 2.44, MPFR 4.2.2, gcc/g++ 16.2.1 (the results are correctly rounded, so
// neither the CPU nor the compiler matters).
// Inputs, in this order:
//   - special inputs: +-0, +-inf, NaN, the smallest/largest subnormal and normal numbers, DBL_MAX,
//     the tiny-path threshold 0x1.6a09e667f3bccp-27 and its neighbours, 2^-26, 2^-27,
//     0x1.7137449123ef6p-26, the reduce_fast branch point 0x1.921fb54442d17p+2 and its
//     neighbours, 2^52 and 2^1023 with neighbours, the exceptions of sin_accurate/cos_accurate
//     and their neighbours, the comments' extreme inputs, each with both signs;
//   - every 40th line of CORE-MATH's list of hard-to-round sin/cos inputs (sincos.wc, the
//     argument), both signs alternately: these take the accurate paths;
//   - then deterministic splitmix64 (seeded with 11) draws up to N_TOTAL lines, input i from
//     distribution i % 16:
//       0: random bit patterns (whole range, both signs, inf/NaN); 1: subnormals (exponent field
//       0); 2: |x| < 2^-26 (random mantissa, exponent in [-1022, -27], the tiny path and just
//       above it); 3: the 2^-26 / 2^-27 / 0x1.6a09e667f3bccp-27 / 0x1.7137449123ef6p-26
//       thresholds +- up to 4096 ulps; 4: uniform [-4, 4]; 5: uniform [-1e3, 1e3]; 6: uniform
//       [-1e6, 1e6]; 7: large |x| (random mantissa, exponent in [20, 996], up to ~1e300);
//       8: the double nearest k*pi/2 (MPFR), k uniform in [-2^20, 2^20] or random up to 2^45,
//       +- up to 4 ulps; 9..15: sgfmetadata.cpp fillMetadataRow's argument
//       (d / period) * 6.283185307179586476925 with d an integer uniform in [-80000, 80000] and
//       period = 7.0 * factor^i (computed as upstream does: period *= factor, from 7.0, with
//       factor = pow(80000, 1.0/31)), i uniform in [0, 31].
// Output (one line per input; NaN inputs and results are printed as 7FF8000000000000, payloads
// differ between x86 and wasm):
//   S <x bits> <sin(x) bits> <cos(x) bits>
#include <mpfr.h>

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

extern "C" void cr_sincos(double x, double* s, double* c);

static uint64_t S;
static uint64_t sm() {
  uint64_t z = (S += 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}
static double u() { return (double)(sm() >> 11) * 0x1p-53; }
static double fb(uint64_t b) { double d; memcpy(&d, &b, 8); return d; }
static uint64_t rawbits(double d) { uint64_t b; memcpy(&b, &d, 8); return b; }
static uint64_t bits(double d) { if(std::isnan(d)) return 0x7FF8000000000000ULL; return rawbits(d); }
static double sgn(double v) { return (sm() & 1) ? -v : v; }

static const long N_TOTAL = 20480;

static mpfr_t MX, MY;
static double mpfrEval(int (*f)(mpfr_ptr, mpfr_srcptr, mpfr_rnd_t), double x) {
  mpfr_set_d(MX, x, MPFR_RNDN);
  int t = f(MY, MX, MPFR_RNDN);
  t = mpfr_subnormalize(MY, t, MPFR_RNDN);
  return mpfr_get_d(MY, MPFR_RNDN);
}

static long lines = 0;
static void emit(double x) {
  if(std::isnan(x)) x = fb(0x7FF8000000000000ULL);
  double s, c;
  cr_sincos(x, &s, &c);
  double ms = mpfrEval(mpfr_sin, x), mc = mpfrEval(mpfr_cos, x);
  if(bits(s) != bits(ms) || bits(c) != bits(mc)) {
    fprintf(stderr, "CORE-MATH != MPFR at x=%a (%016llX): sin %a vs %a, cos %a vs %a\n", x,
            (unsigned long long)bits(x), s, ms, c, mc);
    abort();
  }
  printf("S %016llX %016llX %016llX\n", (unsigned long long)bits(x), (unsigned long long)bits(s),
         (unsigned long long)bits(c));
  lines++;
}

// the double nearest k*pi/2
static double nearestMultipleOfHalfPi(int64_t k) {
  mpfr_t p;
  mpfr_init2(p, 300);
  mpfr_const_pi(p, MPFR_RNDN);
  mpfr_mul_si(p, p, (long)k, MPFR_RNDN);
  mpfr_div_2ui(p, p, 1, MPFR_RNDN);
  double d = mpfr_get_d(p, MPFR_RNDN);
  mpfr_clear(p);
  return d;
}

static double PERIODS[32];

static double gen(long i) {
  switch(i % 16) {
  case 0: return fb(sm());
  case 1: return fb((sm() & 0x800FFFFFFFFFFFFFULL));
  case 2: { uint64_t e = 1 + sm() % 996; uint64_t m = sm(); return fb((m & 0x800FFFFFFFFFFFFFULL) | (e << 52)); }
  case 3: {
    static const uint64_t T[4] = {0x3E50000000000000ULL, 0x3E40000000000000ULL, 0x3E46A09E667F3BCCULL,
                                  0x3E57137449123EF6ULL};
    uint64_t b = T[sm() % 4];
    int64_t k = (int64_t)(sm() % 8193) - 4096;
    return sgn(fb(b + k));
  }
  case 4: return u() * 8.0 - 4.0;
  case 5: return u() * 2e3 - 1e3;
  case 6: return u() * 2e6 - 1e6;
  case 7: { uint64_t e = 1023 + 20 + sm() % 977; uint64_t m = sm(); return fb((m & 0x800FFFFFFFFFFFFFULL) | (e << 52)); }
  case 8: {
    int64_t k;
    if(sm() & 1) k = (int64_t)(sm() % ((1ULL << 21) + 1)) - (1LL << 20);
    else { k = (int64_t)(sm() % (1ULL << (1 + sm() % 45))); if(sm() & 1) k = -k; }
    double d = nearestMultipleOfHalfPi(k);
    int64_t j = (int64_t)(sm() % 9) - 4;
    if(d == 0.0) return fb((uint64_t)(j < 0 ? -j : j)) * (j < 0 ? -1.0 : 1.0);
    return fb(rawbits(d) + j);
  }
  default: {
    double daysDifference = (double)((int64_t)(sm() % 160001) - 80000);
    int p = (int)(sm() % 32);
    double numRevolutions = daysDifference / PERIODS[p];
    return numRevolutions * 6.283185307179586476925;
  }
  }
}

int main(int argc, char** argv) {
  if(argc != 2) {
    fprintf(stderr, "usage: %s <core-math>/src/binary64/sincos/sincos.wc\n", argv[0]);
    return 2;
  }
  mpfr_set_emin(-1073);
  mpfr_set_emax(1024);
  mpfr_init2(MX, 53);
  mpfr_init2(MY, 53);
  {
    double period = 7.0;
    static const double factor = pow(80000, 1.0 / (32 - 1));
    for(int i = 0; i < 32; i++) {
      PERIODS[i] = period;
      period *= factor;
    }
  }

  // special inputs
  static const uint64_t SPECIAL[] = {
    0x0000000000000000ULL, 0x7FF0000000000000ULL, 0x7FF8000000000000ULL, 0x0000000000000001ULL,
    0x000FFFFFFFFFFFFFULL, 0x0010000000000000ULL, 0x7FEFFFFFFFFFFFFFULL, 0x3FF0000000000000ULL,
    0x3E50000000000000ULL, 0x3E40000000000000ULL, 0x3E57137449123EF6ULL,
    0x4330000000000000ULL, 0x7FE0000000000000ULL,
  };
  static const uint64_t SPECIAL_NB[] = {
    0x3E46A09E667F3BCCULL,  // 0x1.6a09e667f3bccp-27, last tiny-path input
    0x401921FB54442D17ULL,  // 0x1.921fb54442d17p+2, reduce_fast branch
    0x400921FB54442D18ULL,  // pi
    0x3FF921FB54442D18ULL,  // pi/2
    0x3FE921FB54442D18ULL,  // pi/4
  };
  static const double EXC[] = {
    0x1.e0000000001c2p-20, 0x1.6ac5b262ca1ffp+849, 0x1.8000000000009p-23, 0x1.8000000000024p-22,
    0x1.800000000009p-21, 0x1.20000000000f3p-20, 0x1.800000000024p-20, 0x1.6ac5b262ca1ffp+851,
    0x1.61a3db8c8d129p+1023,
  };
  for(uint64_t b : SPECIAL) {
    emit(fb(b));
    emit(-fb(b));
  }
  for(uint64_t b : SPECIAL_NB) {
    for(int64_t j = -2; j <= 2; j++) {
      emit(fb(b + j));
      emit(-fb(b + j));
    }
  }
  for(double x : EXC) {
    for(int64_t j = -1; j <= 1; j++) {
      emit(fb(rawbits(x) + j));
      emit(-fb(rawbits(x) + j));
    }
  }

  // hard-to-round cases
  FILE* f = fopen(argv[1], "r");
  if(f == NULL) {
    perror(argv[1]);
    return 1;
  }
  char buf[256];
  long n = 0;
  while(fgets(buf, sizeof buf, f)) {
    if(!(buf[0] == '0' || buf[0] == '-' || buf[0] == '+'))
      continue;
    double x = strtod(buf, NULL);
    if(n % 40 == 0)
      emit((n / 40) % 2 ? -x : x);
    n++;
  }
  fclose(f);

  S = 11;
  for(long i = 0; lines < N_TOTAL; i++)
    emit(gen(i));

  mpfr_clear(MX);
  mpfr_clear(MY);
  mpfr_free_cache();
  return 0;
}
