// Oracle for crates/katago-core/src/core/fancymath.rs: every FancyMath function of the
// unmodified upstream KataGo v1.16.5 (cpp/core/fancymath.cpp, the object file of the reference
// Eigen build) on fixed and pseudo-random inputs. tests/fancymath.rs regenerates the same inputs
// and compares bit for bit.
// Build and run (from the repo root, after `just katago-eigen`; -ffp-contract=off so the input
// expressions here round like Rust; fancymath.cpp.o is upstream's own build, -O3 without FMA):
//   U=upstream/KataGo/cpp; O=build/eigen/CMakeFiles/katago.dir/core
//   g++ -std=c++17 -O2 -ffp-contract=off -I$U -o /tmp/fancymath_oracle \
//     fixtures/upstream-v1.16.5/unit/oracle/fancymath.cpp $O/fancymath.cpp.o $O/global.cpp.o
//   /tmp/fancymath_oracle > fixtures/upstream-v1.16.5/unit/oracle/fancymath.txt
// MUST run on an x86-64 CPU with FMA (glibc's exp/log/pow/expm1 then use their FMA variants,
// the ones the KataGo fixtures were produced with). Generated with glibc 2.44, g++ 16.2.1,
// Intel i7-8700K.
// Output:
//   T <i> <z bits> <tdistpdf bits> <tdistcdf bits>   the DistributionTable search.cpp builds:
//        pdf/cdf of StudentT(VALUE_WEIGHT_DEGREES_OF_FREEDOM = 3.0) at z = -50 + i*100/1999,
//        i = 1..1998 (entries 0 and 1999 are constants)
//   C <fn> <chunk> <hash>   FNV-1a-style fold of the result bit patterns of 4096 random inputs
//   S <fn> <arg bits>... <result bits>   fixed inputs, one per line
//   S cf <case> <tolerance bits> <maxTerms> <result bits>   evaluateContinuedFraction
// NaN results are folded/printed as 7FF8000000000000.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <vector>
#include "core/fancymath.h"

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

static const int CHUNK = 4096;
static const int CHUNKS = 16;

// Input generators, mirrored in tests/fancymath.rs. Every sm()/u() call is its own statement so
// the draw order is the same in C++ and Rust.
static double genAB() {
  switch(sm() % 4) {
  case 0: return 0.5 + u() * 20.0;
  case 1: { int e = (int)(sm() % 26) - 10; double v = u(); return (1.0 + v) * p2(e); }
  case 2: return (double)(sm() % 40 + 1) * 0.5;
  default: return u() * 3.0;
  }
}
static double genX01() {
  switch(sm() % 5) {
  case 0: return u();
  case 1: { double v = u(); double w = u(); return v * w; }
  case 2: return 1.0 - u() * 1e-3;
  case 3: return (double)(sm() % 11) / 10.0;
  default: return u() * 1.2 - 0.1;
  }
}
static double genTX() {
  switch(sm() % 4) {
  case 0: return u() * 120.0 - 60.0;
  case 1: return u() * 10.0 - 5.0;
  case 2: { uint64_t s = sm() & 1; int e = (int)(sm() % 40) - 20; double v = u(); return (s == 0 ? 1.0 : -1.0) * (1.0 + v) * p2(e); }
  default: return rb();
  }
}
static double genV() {
  switch(sm() % 6) {
  case 0: return 3.0;
  case 1: return (double)(sm() % 30 + 1);
  case 2: { int e = (int)(sm() % 24) - 8; double v = u(); return (1.0 + v) * p2(e); }
  case 3: return u() * 20.0;
  case 4: return u() * 3.0;
  default: return -u();
  }
}
static double genProb() {
  switch(sm() % 3) {
  case 0: return u();
  case 1: return (double)(sm() % 5) / 4.0;
  default: return u() * 1e-3;
  }
}
static double genEps() {
  switch(sm() % 4) {
  case 0: return u() * 0.5;
  case 1: return 0.0;
  case 2: return 1e-3;
  default: return 0.25;
  }
}

template<typename F>
static void chunks(const char* name, uint64_t seed, F f) {
  S = seed;
  for(int c = 0; c < CHUNKS; c++) {
    uint64_t h = 1469598103934665603ULL;
    for(int k = 0; k < CHUNK; k++) h = mix(h, bits(f()));
    printf("C %s %d %016llX\n", name, c, (unsigned long long)h);
  }
}

static void s1(const char* fn, double a, double r) {
  printf("S %s %016llX %016llX\n", fn, (unsigned long long)bits(a), (unsigned long long)bits(r));
}
static void s2(const char* fn, double a, double b, double r) {
  printf("S %s %016llX %016llX %016llX\n", fn, (unsigned long long)bits(a), (unsigned long long)bits(b), (unsigned long long)bits(r));
}
static void s3(const char* fn, double a, double b, double c, double r) {
  printf("S %s %016llX %016llX %016llX %016llX\n", fn, (unsigned long long)bits(a), (unsigned long long)bits(b), (unsigned long long)bits(c), (unsigned long long)bits(r));
}

int main() {
  // search.cpp: valueWeightDistribution = new DistributionTable(tdistpdf(z,3.0), tdistcdf(z,3.0), -50.0, 50.0, 2000)
  {
    const double VALUE_WEIGHT_DEGREES_OF_FREEDOM = 3.0;
    double minZ = -50.0, maxZ = 50.0;
    int size = 2000;
    for(int i = 1; i < size - 1; i++) {
      double z = minZ + i * (maxZ - minZ) / (double)(size - 1);
      double pdf = FancyMath::tdistpdf(z, VALUE_WEIGHT_DEGREES_OF_FREEDOM);
      double cdf = FancyMath::tdistcdf(z, VALUE_WEIGHT_DEGREES_OF_FREEDOM);
      printf("T %d %016llX %016llX %016llX\n", i, (unsigned long long)bits(z), (unsigned long long)bits(pdf), (unsigned long long)bits(cdf));
    }
  }

  chunks("beta", 21, [] { double a = genAB(); double b = genAB(); return FancyMath::beta(a, b); });
  chunks("logbeta", 22, [] { double a = genAB(); double b = genAB(); return FancyMath::logbeta(a, b); });
  chunks("incompleteBeta", 23, [] { double x = genX01(); double a = genAB(); double b = genAB(); return FancyMath::incompleteBeta(x, a, b); });
  chunks("regularizedIncompleteBeta", 24, [] { double x = genX01(); double a = genAB(); double b = genAB(); return FancyMath::regularizedIncompleteBeta(x, a, b); });
  chunks("betapdf", 25, [] { double x = genX01(); double a = genAB(); double b = genAB(); return FancyMath::betapdf(x, a, b); });
  chunks("betacdf", 26, [] { double x = genX01(); double a = genAB(); double b = genAB(); return FancyMath::betacdf(x, a, b); });
  chunks("tdistpdf", 27, [] { double x = genTX(); double v = genV(); return FancyMath::tdistpdf(x, v); });
  chunks("tdistcdf", 28, [] { double x = genTX(); double v = genV(); return FancyMath::tdistcdf(x, v); });
  chunks("tdistpdf3", 29, [] { double x = u() * 120.0 - 60.0; return FancyMath::tdistpdf(x, 3.0); });
  chunks("tdistcdf3", 30, [] { double x = u() * 120.0 - 60.0; return FancyMath::tdistcdf(x, 3.0); });
  chunks("normToTApprox", 31, [] { double z = u() * 20.0 - 10.0; double n = genV(); return FancyMath::normToTApprox(z, n); });
  chunks("binaryCrossEntropy", 32, [] { double p = genProb(); double t = genProb(); double e = genEps(); return FancyMath::binaryCrossEntropy(p, t, e); });

  const double inf = std::numeric_limits<double>::infinity();
  const double nan = std::numeric_limits<double>::quiet_NaN();
  {
    std::vector<double> ab = {0.5, 1.0, 2.0, 1e-300, 1e300, 0.0, -0.5, -1.0, nan, inf, 3.4, 0.25};
    for(double a : ab) for(double b : ab) {
      s2("beta", a, b, FancyMath::beta(a, b));
      s2("logbeta", a, b, FancyMath::logbeta(a, b));
    }
    std::vector<double> xs = {0.0, 1.0, 0.5, -0.1, 1.1, nan, 0.25, 1.0 - 1e-4};
    std::vector<double> abs2 = {0.5, 1.0, 2.0, 0.0, -1.0, nan, 5e4, 5.0};
    for(double x : xs) for(double a : abs2) for(double b : abs2) {
      s3("incompleteBeta", x, a, b, FancyMath::incompleteBeta(x, a, b));
      s3("regularizedIncompleteBeta", x, a, b, FancyMath::regularizedIncompleteBeta(x, a, b));
      s3("betapdf", x, a, b, FancyMath::betapdf(x, a, b));
      s3("betacdf", x, a, b, FancyMath::betacdf(x, a, b));
    }
  }
  {
    std::vector<double> xs = {0.0, -0.0, 1.0, -1.0, 6.0, -6.0, 50.0, -50.0, inf, -inf, nan, 1e300, -1e300};
    std::vector<double> vs = {3.0, 1.0, 2.0, 3.4, 12.3, 0.0, -1.0, nan, inf, 1e300};
    for(double x : xs) for(double v : vs) {
      s2("tdistpdf", x, v, FancyMath::tdistpdf(x, v));
      s2("tdistcdf", x, v, FancyMath::tdistcdf(x, v));
    }
  }
  {
    std::vector<double> zs = {0.0, 2.0, 4.0, 8.0, -2.0, inf, nan, 1e-300};
    std::vector<double> ns = {2.0, 4.0, 8.0, 16.0, 10000.0, 1.5, 1.0, 0.0, 0.5, 3.0, inf};
    for(double z : zs) for(double n : ns) s2("normToTApprox", z, n, FancyMath::normToTApprox(z, n));
  }
  {
    std::vector<double> ps = {0.0, 0.5, 1.0, 1.0 / 6.0, 1.0 / std::exp(1.0), nan, 0.999};
    std::vector<double> ts = {0.0, 0.7, 0.8, 1.0};
    std::vector<double> es = {0.0, 0.25, 0.5, 0.001};
    for(double p : ps) for(double t : ts) for(double e : es) s3("binaryCrossEntropy", p, t, e, FancyMath::binaryCrossEntropy(p, t, e));
  }
  {
    // The continued fractions of FancyMath::runTests, and an alternating one, at several
    // tolerances and term limits.
    std::vector<std::function<double(int)>> numers = {
      [](int n) noexcept { (void)n; return 1.0; },
      [](int n) noexcept { (void)n; return 1.0; },
      [](int n) noexcept { (void)n; return 1.0; },
      [](int n) noexcept { return (n*2-1)*(n*2-1); },
      [](int n) noexcept { return n == 1 ? 4 : ((n-1)*(n-1)*4-1); },
      [](int n) noexcept { return n == 1 ? 2 : n*(n-1); },
      [](int n) noexcept { return n % 2 == 0 ? 0.0 : -1.0; },
    };
    std::vector<std::function<double(int)>> denoms = {
      [](int n) noexcept { (void)n; return 1.0; },
      [](int n) noexcept { return n == 0 ? 1.0 : 2.0; },
      [](int n) noexcept { return n == 0 ? 2.0 : n%3 == 2 ? (double)((n+1)/3*2) : 1.0; },
      [](int n) noexcept { return n == 0 ? 3.0 : 6.0; },
      [](int n) noexcept { return n == 0 ? 2.0 : n == 1 ? 3.0 : 4.0; },
      [](int n) noexcept { return n == 0 ? 2.0 : 1.0; },
      [](int n) noexcept { return n == 0 ? 0.0 : 1.0; },
    };
    std::vector<double> tols = {1e-15, 1e-10, 0.0, 0.5};
    std::vector<int> maxTerms = {0, 1, 2, 3, 10, 1000, 10000};
    for(size_t c = 0; c < numers.size(); c++)
      for(double tol : tols)
        for(int mt : maxTerms) {
          double r = FancyMath::evaluateContinuedFraction(numers[c], denoms[c], tol, mt);
          printf("S cf %d %016llX %d %016llX\n", (int)c, (unsigned long long)bits(tol), mt, (unsigned long long)bits(r));
        }
  }
  return 0;
}
