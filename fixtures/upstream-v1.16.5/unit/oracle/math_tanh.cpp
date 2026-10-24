// Oracle for crates/katago-core/src/port/math/{tanh,atanh}.rs: glibc 2.44's `tanh` and `atanh`
// (CORE-MATH's correctly rounded cr_tanh / cr_atanh) on a fixed pseudo-random input set plus
// special inputs. tests/math_tanh.rs replays the file line by line. Not upstream code; no
// upstream sources needed.
// Build and run (from the repo root; -ffp-contract=off and -fno-builtin so that the generator's
// own arithmetic rounds the way a plain C evaluation does and the libm functions are really called):
//   g++ -std=c++17 -O2 -ffp-contract=off -fno-builtin \
//     -o /tmp/math_tanh_oracle fixtures/upstream-v1.16.5/unit/oracle/math_tanh.cpp -lm
//   /tmp/math_tanh_oracle > fixtures/upstream-v1.16.5/unit/oracle/math_tanh.txt
// Generated with glibc 2.44, g++ 16.2.1, Intel i7-8700K (both functions are correctly rounded, so
// the CPU does not matter).
// Generator (deterministic, splitmix64 seeded with 7): TANH_N inputs for tanh and ATANH_N for
// atanh, input i drawn from distribution i % 16:
//   tanh:  random bit patterns (whole exponent range, subnormals, both signs, inf/NaN); uniform
//          [-25, 25]; uniform [-4, 4]; random mantissa with exponent in [-60, 3]; |x| < 5e-9;
//          0.5 +- 0.005; +-(19.06 +- 0.1) (saturation); |x| < 0.25 (polynomial branch);
//          +-(3.683 +- 0.01) (branch change); the caller's form tanh(atanh(w) + s*o) with w
//          uniform in (-1, 1), s in {0.3, 0.7}, o uniform in [-1, 1] (searchmirror.cpp-like);
//          +-[0, 3); subnormals/tiny (random mantissa, exponent field 0); exponent in [-40, -21];
//          the saturation threshold 0x1.30fc1931f09cap+4 +- up to 200 ulps; +-[0, 40);
//          0.25 +- up to 100 ulps.
//   atanh: random bit patterns; uniform (-1, 1); +-(1 - k*2^-53), k < 100000; random mantissa
//          with exponent in [-60, 3]; |x| < 5e-9; |x| < 0.25; 0.25 +- 0.005; exponent in
//          [-30, -1]; +-(1 - u*1e-3); +-(1 - u*2^-k), k in [1, 52]; +-u; subnormals/tiny;
//          tanh(uniform [-20, 20]); +-(1 - k ulp), k < 100; 0.25 + k ulp and 0.25 - k ulp, k < 1000.
// Output (one line per case; NaN inputs and results are printed as 7FF8000000000000, payloads
// differ between x86 and wasm):
//   T <x bits> <tanh(x) bits>
//   A <x bits> <atanh(x) bits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>

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
static double ulpup(double x, uint64_t k) { uint64_t b; memcpy(&b, &x, 8); return fb(b + k); }
static double ulpdn(double x, uint64_t k) { uint64_t b; memcpy(&b, &x, 8); return fb(b - k); }
// one draw per statement below, in a fixed order (the order matters: this is a stream)
static double sgn(double v) { return (sm() & 1) ? -v : v; }

static const int TANH_N = 24576;
static const int ATANH_N = 16384;

static double genTanh(long i) {
  switch(i % 16) {
  case 0: return fb(sm());
  case 1: return u() * 50.0 - 25.0;
  case 2: return u() * 8.0 - 4.0;
  case 3: { uint64_t e = sm() % 64; uint64_t m = sm(); return fb((m & 0x800FFFFFFFFFFFFFULL) | ((e + 1023 - 60) << 52)); }
  case 4: return (u() - 0.5) * 1e-8;
  case 5: return 0.5 + (u() - 0.5) * 0.01;
  case 6: { double v = 19.06 + (u() - 0.5) * 0.2; return sgn(v); }
  case 7: return (u() - 0.5) * 0.5;
  case 8: { double v = 3.683 + (u() - 0.5) * 0.02; return sgn(v); }
  case 9: {
    double w = u() * 2.0 - 1.0;
    double s = (sm() & 1) ? 0.3 : 0.7;
    double o = u() * 2.0 - 1.0;
    return atanh(w) + s * o;
  }
  case 10: { double v = u() * 3.0; return sgn(v); }
  case 11: return fb(sm() & 0x800FFFFFFFFFFFFFULL);
  case 12: { uint64_t e = sm() % 20; uint64_t m = sm(); return fb((m & 0x800FFFFFFFFFFFFFULL) | ((e + 1023 - 40) << 52)); }
  case 13: { double t = fb(0x40330fc1931f09caULL); uint64_t k = sm() % 200; return (sm() & 1) ? ulpup(t, k) : ulpdn(t, k); }
  case 14: { double v = u() * 40.0; return sgn(v); }
  default: { uint64_t k = sm() % 100; double v = (sm() & 1) ? ulpup(0.25, k) : ulpdn(0.25, k); return sgn(v); }
  }
}

static double genAtanh(long i) {
  switch(i % 16) {
  case 0: return fb(sm());
  case 1: return u() * 2.0 - 1.0;
  case 2: { uint64_t k = sm() % 100000; double v = 1.0 - (double)k * 0x1p-53; return sgn(v); }
  case 3: { uint64_t e = sm() % 64; uint64_t m = sm(); return fb((m & 0x800FFFFFFFFFFFFFULL) | ((e + 1023 - 60) << 52)); }
  case 4: return (u() - 0.5) * 1e-8;
  case 5: return (u() - 0.5) * 0.5;
  case 6: { double v = 0.25 + (u() - 0.5) * 0.01; return sgn(v); }
  case 7: { uint64_t e = sm() % 30; uint64_t m = sm(); return fb((m & 0x800FFFFFFFFFFFFFULL) | ((e + 1023 - 30) << 52)); }
  case 8: { double v = 1.0 - u() * 1e-3; return sgn(v); }
  case 9: { int k = (int)(sm() % 52) + 1; double v = 1.0 - u() * ldexp(1.0, -k); return sgn(v); }
  case 10: { double v = u(); return sgn(v); }
  case 11: return fb(sm() & 0x800FFFFFFFFFFFFFULL);
  case 12: { double a = u() * 40.0 - 20.0; return tanh(a); }
  case 13: { uint64_t k = sm() % 100; double v = ulpdn(1.0, k); return sgn(v); }
  case 14: { uint64_t k = sm() % 1000; double v = ulpup(0.25, k); return sgn(v); }
  default: { uint64_t k = sm() % 1000; double v = ulpdn(0.25, k); return sgn(v); }
  }
}

// Special inputs for both functions (each is also run negated): zeros, +-1, infinities, NaN,
// subnormals, the branch boundaries of cr_tanh / cr_atanh with their neighbours, the hard cases
// of CORE-MATH's databases (as_tanh_database / as_atanh_database / as_atanh_refine) and the
// failure cases named in their comments.
static const double SPECIAL[] = {
  0.0, 1.0, 0.5, 0.25, 2.0, 3.0, 19.0, 20.0, 100.0, 710.0, 1e300, 0x1p-1022, 0x1p-1074, 0x1.8p-1074,
  0x1.fffffffffffffp-1023, 0x1p-27, 0x1p-26, 0x1p-24, 0x1p-3, 0x1.00cp-3, 0x1.015891c9eaef8p-3,
  0x1.ap-3, 0x1p-2, 0x1.fffffffffffffp-3,
  0x1.d12ed0af1a27fp-27, 0x1.d12ed0af1a27ep-27, 0x1.d12ed0af1a280p-27,
  0x1.30fc1931f09cap+4, 0x1.30fc1931f09c9p+4, 0x1.30fc1931f09cbp+4,
  0x1.d76c8b4395810p+1, 0x1.d76c8b439580fp+1, 0x1.d76c8b4395811p+1,
  0x1.fffffffffffffp-1, 0x1.ffffffffffffep-1, 0x1.ep-1, 0x1.8p-1, 0x1.0000000000001p+0,
  0x1.ac343b179fec4p-3, 0x1.00764a988bf73p-2, 0x1.17d1e8a63711fp-2, 0x1.291c601a05276p-2,
  0x1.36f33d51c264dp-2, 0x1.43eaea23649c3p-2, 0x1.d88d7550b2826p-2, 0x1.e611aa58ab608p-2,
  0x1.01efe7ac8c15dp-1, 0x1.1005ec0bccabbp-1, 0x1.33dfeb0fa4bfep-1, 0x1.49f24ac5cac35p-1,
  0x1.2dbb7b1c91363p-2, 0x1.c493dc899e4a5p-2, 0x1.110e96a6c2d96p-2, 0x1.dc3fe1b524821p-2,
  0x1.64279e7c7064bp-1, 0x1.cc7092205cfafp-10, 0x1.2f67d96be6eafp-9, 0x1.ebf0bfefa727dp-1,
  0x1.27a0e7f47f0fap-4, 0x1.a0112a16e9318p+1, 0x1.a0bd10af4ac2bp+1, 0x1.09cc2de69e78cp+2,
  0x1.f676484c0703bp-3, 0x1.a612499c53078p-3, 0x1.36f33d51c264cp-2,
};
static const int NSPECIAL = sizeof(SPECIAL) / sizeof(SPECIAL[0]);

static void emit(double x) {
  printf("T %016llX %016llX\n", (unsigned long long)bits(x), (unsigned long long)bits(tanh(x)));
}
static void emitA(double x) {
  printf("A %016llX %016llX\n", (unsigned long long)bits(x), (unsigned long long)bits(atanh(x)));
}

int main() {
  S = 7;
  for(long i = 0; i < TANH_N; i++) emit(genTanh(i));
  for(long i = 0; i < ATANH_N; i++) emitA(genAtanh(i));
  const double INF = std::numeric_limits<double>::infinity();
  const double NaN = std::numeric_limits<double>::quiet_NaN();
  const double MAXD = std::numeric_limits<double>::max();
  const double EX[] = {INF, NaN, MAXD, 0x1.fffffffffffffp+1023 / 2};
  for(int a = 0; a < NSPECIAL; a++) {
    double x = SPECIAL[a];
    // each special input with both signs, its two neighbours, and for the db-like values every
    // neighbour up to 2 ulps away
    for(int k = -2; k <= 2; k++) {
      double v = k >= 0 ? ulpup(x, (uint64_t)k) : ulpdn(x, (uint64_t)(-k));
      if(std::isnan(v)) continue;
      emit(v); emit(-v); emitA(v); emitA(-v);
    }
  }
  emit(-0.0); emitA(-0.0);
  for(double e : EX) { emit(e); emit(-e); emitA(e); emitA(-e); }
  return 0;
}
