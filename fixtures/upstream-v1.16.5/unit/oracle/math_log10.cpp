// Oracle for port::math::log10 (crates/katago-core/src/port/math.rs): glibc's log10 on a fixed
// pseudo-random input set plus every power of ten and its two neighbours. tests/math_log10.rs
// regenerates the same inputs and compares. Not upstream code; no upstream sources needed.
// Build and run (from the repo root):
//   g++ -std=c++17 -O2 -ffp-contract=off -fno-builtin \
//     -o /tmp/math_log10 fixtures/upstream-v1.16.5/unit/oracle/math_log10.cpp -lm
//   /tmp/math_log10 > fixtures/upstream-v1.16.5/unit/oracle/math_log10.txt
// Run on an x86-64 CPU with FMA (log10 calls __ieee754_log, which resolves to __log_fma there).
// Generated with glibc 2.44, g++ 16.2.1.
// Output:
//   C <chunk> <hash>   FNV-1a-style fold of the result bit patterns of 65536 inputs
//   S <x bits> <result bits>   special inputs, one per line
// NaN results are printed as 7FF8000000000000.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

static uint64_t S;
static uint64_t sm() {
  uint64_t z = (S += 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}
static double u() { return (double)(sm() >> 11) * 0x1p-53; }
static double rb() { uint64_t b = sm(); double d; memcpy(&d, &b, 8); return d; }
static uint64_t bits(double d) { if(std::isnan(d)) return 0x7FF8000000000000ULL; uint64_t b; memcpy(&b, &d, 8); return b; }
static double fromBits(uint64_t b) { double d; memcpy(&d, &b, 8); return d; }
static uint64_t mix(uint64_t h, uint64_t v) { return (h ^ v) * 1099511628211ULL; }

// Input generator, mirrored in tests/math_log10.rs.
static double gen(long i) {
  switch(i % 5) {
  case 0: return rb();
  case 1: return u() * 1e3;
  case 2: return u() * 2.0;
  case 3: return fromBits(sm() >> 12); // subnormal or tiny
  default: return (u() - 0.5) * 1e-3 + 1.0;
  }
}

int main() {
  S = 0x4c4f473130ULL; // "LOG10"
  const int CHUNK = 65536;
  const int CHUNKS = 16;
  for(int c = 0; c < CHUNKS; c++) {
    uint64_t h = 1469598103934665603ULL;
    for(int j = 0; j < CHUNK; j++) {
      double x = gen((long)c * CHUNK + j);
      h = mix(h, bits(log10(x)));
    }
    printf("C %d %016llX\n", c, (unsigned long long)h);
  }
  // Powers of ten (as the decimal literal strtod gives, i.e. the nearest double) and neighbours.
  for(int e = -323; e <= 308; e++) {
    char buf[32];
    snprintf(buf, sizeof(buf), "1e%d", e);
    double x = strtod(buf, NULL);
    double xs[3] = { nextafter(x, 0.0), x, nextafter(x, INFINITY) };
    for(double v : xs)
      printf("S %016llX %016llX\n", (unsigned long long)bits(v), (unsigned long long)bits(log10(v)));
  }
  double specials[] = { 0.0, -0.0, -1.0, 1.0, INFINITY, -INFINITY, NAN, 5e-324, 2.2250738585072014e-308, 1.7976931348623157e308 };
  for(double v : specials)
    printf("S %016llX %016llX\n", (unsigned long long)bits(v), (unsigned long long)bits(log10(v)));
  return 0;
}
