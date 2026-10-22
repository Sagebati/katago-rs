// Oracle for port::math::{atan, tan} (DECISIONS.md D13): glibc results on inputs in the ranges
// ScoreValue uses (nninputs.cpp: atan of a scaled score, tan of a value in (-pi/2, pi/2)).
// Those two delegate to the platform libm, so the test allows 1 ulp, not bit-exactness.
// Not upstream code. Build and run (from the repo root):
//   g++ -std=c++17 -O2 -ffp-contract=off -fno-builtin \
//     -o /tmp/math_ulp_oracle fixtures/upstream-v1.16.5/unit/oracle/math_ulp.cpp -lm
//   /tmp/math_ulp_oracle > fixtures/upstream-v1.16.5/unit/oracle/math_ulp.txt
// Output: <fn> <x bits> <result bits>, hex, one per line.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

static uint64_t state = 0x9E3779B97F4A7C15ULL;
static uint64_t splitmix() {
  uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}
static double unit() { return (double)(splitmix() >> 11) * 0x1.0p-53; }
static uint64_t bits(double d) { uint64_t u; memcpy(&u, &d, 8); return u; }

int main() {
  for(int i = 0; i < 2048; i++) {
    // Mostly |x| < 4 (typical scores over scale*sqrt(area)), some up to 1e3, plus tiny values.
    double x = i % 8 == 0 ? (unit() - 0.5) * 2000.0 : i % 8 == 1 ? (unit() - 0.5) * 1e-6 : (unit() - 0.5) * 8.0;
    printf("atan %016llX %016llX\n", (unsigned long long)bits(x), (unsigned long long)bits(atan(x)));
  }
  for(int i = 0; i < 2048; i++) {
    double x = (unit() - 0.5) * 3.14159;
    printf("tan %016llX %016llX\n", (unsigned long long)bits(x), (unsigned long long)bits(tan(x)));
  }
  return 0;
}
