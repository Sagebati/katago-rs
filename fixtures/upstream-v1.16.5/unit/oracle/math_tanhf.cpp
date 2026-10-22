// Oracle for crates/katago-core/src/port/math.rs `tanhf`: glibc's tanhf (CORE-MATH, correctly
// rounded since glibc 2.41) on a fixed pseudo-random input set. tests/math_tanhf.rs regenerates
// the same inputs and compares. Not upstream code; no upstream sources needed.
// (port::math::tanhf was also checked exhaustively against glibc 2.44 on all 2^32 inputs; this
// fixture exists so that the check also runs on wasm32 under `just test-wasm`.)
// Build and run (from the repo root; -ffp-contract=off so the input expressions round like Rust):
//   g++ -std=c++17 -O2 -ffp-contract=off -fno-builtin \
//     -o /tmp/math_tanhf_oracle fixtures/upstream-v1.16.5/unit/oracle/math_tanhf.cpp -lm
//   /tmp/math_tanhf_oracle > fixtures/upstream-v1.16.5/unit/oracle/math_tanhf.txt
// Generated with glibc 2.44, g++ 16.2.1, Intel i7-8700K (tanhf is correctly rounded, so the CPU
// and FMA variant do not matter).
// Output:
//   C tanhf <chunk> <hash>   FNV-1a-style fold of the result bit patterns of 65536 inputs
//   S tanhf <x bits> <result bits>   special inputs, one per line
// NaN results are folded/printed as 7FF8000000000000 (payloads differ between x86 and wasm).
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
static float rbf() { uint32_t b = (uint32_t)sm(); float f; memcpy(&f, &b, 4); return f; }

static uint64_t bitsf(float f) { if(std::isnan(f)) return 0x7FF8000000000000ULL; uint32_t b; memcpy(&b, &f, 4); return b; }
static uint64_t mix(uint64_t h, uint64_t v) { return (h ^ v) * 1099511628211ULL; }

static const int CHUNK = 65536;
static const int CHUNKS = 16;

// Input generator, mirrored in tests/math_tanhf.rs. i is the input index.
static float genTanhf(long i) {
  switch(i % 6) {
  case 0: return rbf();
  case 1: return (float)(u() * 40.0 - 20.0);
  case 2: return (float)(u() * 4.0 - 2.0);
  case 3: return (float)((u() - 0.5) * 0x1p-9);
  case 4: return (float)(8.0 + u() * 3.0);
  default: return (float)(-(u() * 0.75));
  }
}

static const float SPECIAL[] = {
  0.0f, -0.0f, 1.0f, -1.0f, 0.5f, -0.5f, 2.0f, 3.0f, 9.0f, 9.010913f, 9.0109135f, 10.0f,
  -10.0f, 20.0f, 88.0f, 100.0f, 0x1p-13f, -0x1p-13f, 0x1.fffffep-14f, 0x1.000002p-13f, 0x1p-12f,
  0x1p-24f, 0x1p-126f, 0x1p-149f, -0x1p-149f, 0x1.fffffcp-127f,
  std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(),
  std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
  std::numeric_limits<float>::quiet_NaN(), 0.25f, 0.625f, 0.6931472f, 1.5f, 4.0f, 7.5f,
};
static const int NSPECIAL = sizeof(SPECIAL) / sizeof(SPECIAL[0]);

int main() {
  S = 5;
  for(int c = 0; c < CHUNKS; c++) {
    uint64_t h = 1469598103934665603ULL;
    for(int k = 0; k < CHUNK; k++) { long i = (long)c * CHUNK + k; float x = genTanhf(i); h = mix(h, bitsf(tanhf(x))); }
    printf("C tanhf %d %016llX\n", c, (unsigned long long)h);
  }
  for(int a = 0; a < NSPECIAL; a++) {
    float x = SPECIAL[a];
    printf("S tanhf %016llX %016llX\n", (unsigned long long)bitsf(x), (unsigned long long)bitsf(tanhf(x)));
  }
  return 0;
}
