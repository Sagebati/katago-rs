// Oracle for search/distributiontable (DistributionTable: table construction, getPdfCdf/getPdf/getCdf).
// Links against the unmodified upstream sources of KataGo v1.16.5. The Rust mirror is
// crates/katago-core/tests/distributiontable.rs: keep the two in the same order.
// The pdf/cdf functions are not FancyMath's t-distribution (core/fancymath is ported separately): they use
// only + - * / sqrt (exact everywhere) and, for the last table, exp (glibc; port::math::exp matches it, D14).
// Doubles are printed as 16 hex digits of their bits.
//
// Build (from the repo root, output outside the repo; upstream untouched; asserts on):
//   U=upstream/KataGo/cpp; g++ -std=c++17 -O1 -I$U -o /tmp/distributiontable_oracle \
//     fixtures/upstream-v1.16.5/unit/oracle/distributiontable.cpp \
//     $U/core/global.cpp $U/search/distributiontable.cpp -lpthread
//   /tmp/distributiontable_oracle > fixtures/upstream-v1.16.5/unit/oracle/distributiontable.txt
// Generated with g++ (GCC) 16.2.1 / glibc 2.44 on x86_64 Linux, FMA CPU.
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <string>
#include "core/global.h"
#include "search/distributiontable.h"
using namespace std;

static string bits(double x) {
  uint64_t u;
  memcpy(&u, &x, sizeof(u));
  char buf[17];
  snprintf(buf, sizeof(buf), "%016llX", (unsigned long long)u);
  return string(buf);
}

static double ratPdf(double z) { return 1.0 / (1.0 + z * z) / 3.14; }
static double ratCdf(double z) { return 0.5 + z / (2.0 * (1.0 + (z < 0 ? -z : z))); }
static double sqrtPdf(double z) { return 0.5 / ((1.0 + z * z) * sqrt(1.0 + z * z)); }
static double sqrtCdf(double z) { return 0.5 * (1.0 + z / sqrt(1.0 + z * z)); }
static double expPdf(double z) { return exp(-0.5 * z * z) / 2.5066282746310002; }
static double expCdf(double z) { return 0.5 * (1.0 + z / sqrt(1.0 + z * z)) * exp(-0.001 * z * z); }

static void dump(const string& name, const function<double(double)>& pdf, const function<double(double)>& cdf, double minZ, double maxZ, int size) {
  DistributionTable t(pdf, cdf, minZ, maxZ, size);
  cout << "table " << name << " size " << t.size << " minZ " << bits(t.minZ) << " maxZ " << bits(t.maxZ) << "\n";
  uint64_t fnv = 1469598103934665603ULL;
  for(int i = 0; i < size; i++) {
    for(double v : {t.pdfTable[i], t.cdfTable[i]}) {
      uint64_t u;
      memcpy(&u, &v, sizeof(u));
      for(int b = 0; b < 8; b++) {
        fnv ^= (u >> (8 * b)) & 0xFF;
        fnv *= 1099511628211ULL;
      }
    }
  }
  cout << "tablehash " << Global::uint64ToHexString(fnv) << "\n";
  if(size <= 200) {
    for(int i = 0; i < size; i++)
      cout << "entry " << i << " " << bits(t.pdfTable[i]) << " " << bits(t.cdfTable[i]) << "\n";
  }
  else {
    for(int i : {0, 1, 2, size / 4, size / 2, size / 2 + 1, size - 3, size - 2, size - 1})
      cout << "entry " << i << " " << bits(t.pdfTable[i]) << " " << bits(t.cdfTable[i]) << "\n";
  }

  double range = maxZ - minZ;
  auto query = [&](double z) {
    double p, c;
    t.getPdfCdf(z, p, c);
    cout << "q " << bits(z) << " " << bits(p) << " " << bits(c) << " " << bits(t.getPdf(z)) << " " << bits(t.getCdf(z)) << "\n";
  };
  // a grid reaching well outside [minZ,maxZ]
  const int K = 400;
  double lo = minZ - 0.1 * range;
  double hi = maxZ + 0.1 * range;
  for(int k = 0; k <= K; k++) {
    double z = lo + k * (hi - lo) / K;
    query(z);
  }
  // the table nodes and their neighbourhoods (where idx changes)
  int step = size <= 101 ? 1 : 7;
  for(int i = 0; i < size; i += step) {
    double z = minZ + i * (maxZ - minZ) / (double)(size - 1);
    query(z);
    query(z + 1e-9 * range);
    query(z - 1e-9 * range);
    query(z + 0.5 * range / (size - 1));
  }
  // exact boundaries
  query(minZ);
  query(maxZ);
  query(0.0);
  query(-0.0);
  query(1e6);  // d < 2^31 for every table here; larger z makes (int)d overflow: UB (INT_MIN, then an out-of-bounds read), not exercised
  query(-1e300);
  query(minZ - 1e-300);
}

int main() {
  dump("rational 2000", ratPdf, ratCdf, -50.0, 50.0, 2000);
  dump("rational 101", ratPdf, ratCdf, -5.0, 5.0, 101);
  dump("sqrt 2", sqrtPdf, sqrtCdf, 0.0, 1.0, 2);
  dump("sqrt 3", sqrtPdf, sqrtCdf, 0.0, 1.0, 3);
  dump("sqrt 7", sqrtPdf, sqrtCdf, -1.0, 1.0, 7);
  dump("rational 1000 asym", ratPdf, ratCdf, -3.5, 12.25, 1000);
  dump("exp 2000", expPdf, expCdf, -50.0, 50.0, 2000);
  dump("exp 64", expPdf, expCdf, -8.0, 8.0, 64);
  return 0;
}
