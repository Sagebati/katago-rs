// Oracle for crates/katago-core/src/port/math.rs `lgamma`/`lgamma_r` and `expm1`: glibc's
// lgamma_r (CORE-MATH, correctly rounded since glibc 2.43) and expm1 (fdlibm, `__expm1_fma` on
// an FMA CPU) on a fixed pseudo-random input set. tests/math_lgamma.rs regenerates the same
// inputs and compares. Not upstream code; no upstream sources needed.
// Build and run (from the repo root; -ffp-contract=off so the input expressions round like Rust):
//   g++ -std=c++17 -O2 -ffp-contract=off -fno-builtin \
//     -o /tmp/math_lgamma_oracle fixtures/upstream-v1.16.5/unit/oracle/math_lgamma.cpp -lm
//   /tmp/math_lgamma_oracle > fixtures/upstream-v1.16.5/unit/oracle/math_lgamma.txt
// MUST run on an x86-64 CPU with FMA (expm1 resolves to __expm1_fma there, the variant the
// KataGo fixtures were produced with; lgamma is correctly rounded, so the CPU does not matter
// for it). Generated with glibc 2.44, g++ 16.2.1, Intel i7-8700K.
// Output:
//   C <fn> <chunk> <hash>   FNV-1a-style fold of the result bit patterns of 65536 inputs
//                            (for lgamma, each result is followed by its signgam)
//   S lgamma <x bits> <result bits> <signgam>   special inputs, one per line
//   S expm1 <x bits> <result bits>
// NaN results are folded/printed as 7FF8000000000000 (payloads differ between x86 and wasm).
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <limits>

static uint64_t S;
static uint64_t sm() {
  uint64_t z = (S += 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}
static double u() { return (double)(sm() >> 11) * 0x1p-53; }
static double rb() { uint64_t b = sm(); double d; memcpy(&d, &b, 8); return d; }
static double fb(uint64_t b) { double d; memcpy(&d, &b, 8); return d; }
// 2^e for -1022 <= e <= 1023, exactly.
static double p2(int e) { return fb((uint64_t)(1023 + e) << 52); }

static uint64_t bits(double d) { if(std::isnan(d)) return 0x7FF8000000000000ULL; uint64_t b; memcpy(&b, &d, 8); return b; }
static uint64_t mix(uint64_t h, uint64_t v) { return (h ^ v) * 1099511628211ULL; }

static const int CHUNK = 65536;
static const int CHUNKS = 16;

// Input generators, mirrored in tests/math_lgamma.rs. i is the input index. Every sm()/u() call
// is its own statement so the draw order is the same in C++ and Rust.
static double genLgamma(long i) {
  switch(i % 12) {
  case 0: return rb();
  case 1: return u() * 400.0 - 200.0;
  case 2: return u() * 24.0 - 12.0;
  case 3: { // near the poles 0, -1, ..., -299
    double k = (double)(sm() % 300);
    int e = (int)(sm() % 60) + 1;
    double v = u();
    return -k + (v - 0.5) * p2(-e);
  }
  case 4: { // near the roots 1 and 2
    double k = 1.0 + (double)(sm() % 2);
    int e = (int)(sm() % 60) + 1;
    double v = u();
    return k + (v - 0.5) * p2(-e);
  }
  case 5: { // any exponent, either sign
    uint64_t e = sm() % 2047;
    uint64_t m = sm() & 0x000fffffffffffffULL;
    uint64_t s = sm() & (1ULL << 63);
    return fb(s | (e << 52) | m);
  }
  case 6: return fb(sm() & 0x800fffffffffffffULL); // subnormals and zeros
  case 7: return u() * 16.0 - 10.5;                // negative roots of lgamma
  case 8: { // integers and half-integers, some nudged
    double x = (double)(sm() % 400) * 0.5 - 100.0;
    uint64_t c = sm() & 1;
    double v = u();
    return c == 0 ? x : x + (v - 0.5) * 1e-12;
  }
  case 9: return (u() - 0.5) * 2.0;
  case 10: { // huge, up to the overflow threshold 0x1.006df1bfac84ep+1015 and beyond
    int e = (int)(sm() % 64) + 960;
    uint64_t c = sm() & 1;
    double v = u();
    return (c == 0 ? 1.0 : -1.0) * (1.0 + v) * p2(e);
  }
  default: return -(u() * 11.0);
  }
}
static double genExpm1(long i) {
  switch(i % 8) {
  case 0: return rb();
  case 1: return u() * 80.0 - 60.0;
  case 2: return u() * 2.0 - 1.0;
  case 3: return (u() - 0.5) * 1e-3;
  case 4: return u() * 1440.0 - 720.0;
  case 5: return (u() - 0.5) * 2.08; // around 0.5*ln2 and 1.5*ln2
  case 6: {
    int e = (int)(sm() % 70);
    double v = u();
    return (v - 0.5) * p2(-e);
  }
  default: return u() * 50.0;
  }
}

static const uint64_t SPECIAL_LGAMMA_BITS[] = {
  0x7f5754d9278b51a6ULL, 0x7f5754d9278b51a7ULL, 0xff5754d9278b51a6ULL, // special-cased in glibc
  0x7f5754d9278b51a5ULL, 0x7f5754d9278b51a8ULL, 0x7fe0000000000000ULL,
  0x0000000000000001ULL, 0x8000000000000001ULL, 0x000fffffffffffffULL, 0x0010000000000000ULL,
};
static const double SPECIAL_LGAMMA[] = {
  0.0, -0.0, 1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, 0.5, -0.5, 1.5, -1.5, 2.5, -2.5, 0.25,
  0.03125, -0.03125, 0x1p-75, -0x1p-75, 0x1p-100, -0x1p-100, 0x1.fffffffffffffp-2,
  8.29541, 8.29540, 8.295410156, 0x1p52, -0x1p52, 0x1.fffffffffffffp51, -0x1.fffffffffffffp51,
  -4503599627370495.5, -0x1p52 + 0.5, 0x1p100, 0x1p120, 0x1p1014, 0x1.754cp+1014,
  0x1.006df1bfac84ep+1015, 0x1.006df1bfac84dp+1015, -0x1.006df1bfac84ep+1015, 1e300, -1e300,
  std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity(),
  std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::max(),
  -std::numeric_limits<double>::max(), std::numeric_limits<double>::min(),
  // roots of lgamma on the negative axis (to double precision)
  -2.4570247382208006, -2.7476826467274127, -3.1435808883499800, -3.9552942848585511,
  -4.0391158464646568, -4.9991351855215996, -5.0008967117703208, -5.9998551072945867,
  -6.0001477878263450, -6.9999801776710604, -7.0000201733457158, -7.9999975456761645,
  -8.0000024625102657, -8.9999997272630760, -9.0000002730663648, -9.9999999725597830,
  -10.0000000274377196, -10.4,
  // the hard cases in as_lgamma_database
  -0x1.1b649eb4316fbp+3, -0x1.808d3e2f56b4fp+2, -0x1.d02b2008d5bf5p+1, -0x1.a123d403647d7p+1,
  -0x1.90404f46978b6p+1, -0x1.75692939b3defp+1, -0x1.64cc652d46a2dp+1, -0x1.593ae533139c1p+1,
  -0x1.549cde4f1fd16p+1, -0x1.53712a3a51156p+1, -0x1.51333e5a494f7p+1, -0x1.504f0b5b9d1f9p+1,
  -0x1.4c213b7fa02dcp+1, -0x1.75824f4f0c7c8p+0, -0x1.667cb87bcfa0fp+0, -0x1.51668b6af122dp+0,
  -0x1.50b41410187c1p+0, -0x1.307eb80d114afp-2, -0x1.26923ac1f7c1p-2,
  // values fancymath.cpp uses: (v+1)/2, v/2, a, b, a+b of its tests
  0.5, 1.5, 1.7, 2.2, 6.15, 6.65, 5e4, 5.0, 50005.0, 5e9, 50.0, 5000000050.0, 5e14, 500.0,
  500000000000500.0, 1.0, 2.0, 3.0, 4.0, 2.0,
};
static const double SPECIAL_EXPM1[] = {
  0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 2.0, 10.0, -10.0, 20.0, 37.0, 38.0, 39.0, 40.0, 709.0,
  709.782712893384, 709.7827128933840, 0x1.62e42fefa39efp+9, 0x1.62e42fefa39f0p+9, 710.0, 1e300,
  -1e300, -38.816242111356935, -38.81624211135694, -39.0, -745.0, -1000.0, 0x1p-54, -0x1p-54,
  0x1.ffffffffffffffp-55, 0x1p-53, 0x1p-1022, 0x1p-1074, -0x1p-1074, 0.34657359027997264,
  0.3465735902799727, 1.0397207708399179, 1.039720770839918, -0.34657359027997264,
  -1.0397207708399179, 0.25, -0.25, -0.2500000001, 13.5, 14.0, 39.5, 40.5, 1e-10, -1e-10,
  std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity(),
  std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::max(),
  -std::numeric_limits<double>::max(),
  // values fancymath.cpp's normToTApprox computes for its tests
  4.0 * (2.0 - 1.5) / 1.0, 4.0 * 2.5 / 9.0, 4.0 * 6.5 / 49.0, 4.0 * 14.5 / 225.0,
  4.0 * 9998.5 / (9999.0 * 9999.0), 16.0 * 0.5 / 1.0, 64.0 * 0.5 / 1.0,
};

static void printLgamma(double x) {
  int sg = 0;
  double r = lgamma_r(x, &sg);
  printf("S lgamma %016llX %016llX %d\n", (unsigned long long)bits(x), (unsigned long long)bits(r), sg);
}

int main() {
  S = 11;
  for(int c = 0; c < CHUNKS; c++) {
    uint64_t h = 1469598103934665603ULL;
    for(int k = 0; k < CHUNK; k++) {
      long i = (long)c * CHUNK + k;
      double x = genLgamma(i);
      int sg = 0;
      double r = lgamma_r(x, &sg);
      h = mix(h, bits(r));
      h = mix(h, (uint64_t)(int64_t)sg);
    }
    printf("C lgamma %d %016llX\n", c, (unsigned long long)h);
  }
  S = 12;
  for(int c = 0; c < CHUNKS; c++) {
    uint64_t h = 1469598103934665603ULL;
    for(int k = 0; k < CHUNK; k++) { long i = (long)c * CHUNK + k; double x = genExpm1(i); h = mix(h, bits(expm1(x))); }
    printf("C expm1 %d %016llX\n", c, (unsigned long long)h);
  }
  for(uint64_t b : SPECIAL_LGAMMA_BITS) printLgamma(fb(b));
  for(double x : SPECIAL_LGAMMA) {
    printLgamma(x);
    printLgamma(std::nextafter(x, -INFINITY));
    printLgamma(std::nextafter(x, INFINITY));
  }
  for(double x : SPECIAL_EXPM1) {
    for(double y : {std::nextafter(x, -INFINITY), x, std::nextafter(x, INFINITY)})
      printf("S expm1 %016llX %016llX\n", (unsigned long long)bits(y), (unsigned long long)bits(expm1(y)));
  }
  return 0;
}
