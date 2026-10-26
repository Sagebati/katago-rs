// Oracle for crates/katago-core/src/port/math.rs: glibc's exp, log, pow, expf on a fixed
// pseudo-random input set. tests/math.rs regenerates the same inputs and compares.
// Not upstream code; no upstream sources needed.
// Build and run (from the repo root; -ffp-contract=off so the input expressions round like Rust):
//   g++ -std=c++17 -O2 -ffp-contract=off -fno-builtin \
//     -o /tmp/math_oracle fixtures/upstream-v1.18.2/unit/oracle/math.cpp -lm
//   /tmp/math_oracle > fixtures/upstream-v1.18.2/unit/oracle/math.txt
// MUST run on an x86-64 CPU with FMA (glibc then resolves exp/log/pow/expf to __exp_fma etc.,
// the variant the KataGo fixtures were produced with). Generated with glibc 2.44, g++ 16.2.1,
// Intel i7-8700K.
// Output:
//   C <fn> <chunk> <hash>   FNV-1a-style fold of the result bit patterns of 65536 inputs
//   S <fn> <x bits> [<y bits>] <result bits>   special inputs, one per line
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
static double rb() { uint64_t b = sm(); double d; memcpy(&d, &b, 8); return d; }
static float rbf() { uint32_t b = (uint32_t)sm(); float f; memcpy(&f, &b, 4); return f; }

static uint64_t bits(double d) { if(std::isnan(d)) return 0x7FF8000000000000ULL; uint64_t b; memcpy(&b, &d, 8); return b; }
static uint64_t bitsf(float f) { if(std::isnan(f)) return 0x7FF8000000000000ULL; uint32_t b; memcpy(&b, &f, 4); return b; }
static uint64_t mix(uint64_t h, uint64_t v) { return (h ^ v) * 1099511628211ULL; }

static const int CHUNK = 65536;
static const int CHUNKS = 16;

// Input generators, mirrored in tests/math.rs. i is the input index.
static double genExp(long i) {
  switch(i % 6) {
  case 0: return rb();
  case 1: return u() * 1500.0 - 750.0;
  case 2: return u() * 2.0 - 1.0;
  case 3: return (u() - 0.5) * 1e-10;
  case 4: return 700.0 + u() * 20.0;
  default: return -745.0 + u() * 40.0;
  }
}
static double genLog(long i) {
  switch(i % 6) {
  case 0: return rb();
  case 1: return u() * 1e3;
  case 2: return 0.9 + u() * 0.2;
  case 3: return u() * 1e-300;
  case 4: return u();
  default: return u() * 1e-310;
  }
}
static void genPow(long i, double& x, double& y) {
  switch(i % 7) {
  case 0: x = rb(); y = rb(); return;
  case 1: x = u() * 10.0; y = u() * 20.0 - 10.0; return;
  case 2: x = u(); y = 1.0 / (0.03 + u() * 3.0); return;
  case 3: x = -(u() * 10.0); y = (double)((int64_t)(u() * 40.0) - 20); return;
  case 4: x = 1.0 + (u() - 0.5) * 1e-6; y = u() * 1e9; return;
  case 5: x = u() * 1e-310; y = u() * 2.0; return;
  default: x = 1.0 + u(); y = (u() * 2.0 - 1.0) * 1500.0; return;
  }
}
static float genExpf(long i) {
  switch(i % 4) {
  case 0: return rbf();
  case 1: return (float)(u() * 220.0 - 115.0);
  case 2: return (float)(u() * 2.0 - 1.0);
  default: return (float)(-103.97 + u() * 0.7);
  }
}

static const double SPECIAL[] = {
  0.0, -0.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 3.0, -3.0, 10.0, 0.1,
  std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity(),
  std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::denorm_min(),
  -std::numeric_limits<double>::denorm_min(), std::numeric_limits<double>::min(),
  std::numeric_limits<double>::max(), -std::numeric_limits<double>::max(),
  709.782712893384, 709.7827128933841, -708.3964185322641, -745.1332191019411,
  -745.1332191019412, -744.44007192138122, 1e-300, -1e-300, 0x1p-54, 0x1p-55, 512.0, 1024.0,
  -1024.0, 0x1p-65, 0x1p-66, 0x1p63, -0x1p63, 0x1p70, 1e300, -1e300, 0.9999999999999999,
  1.0000000000000002, 0.9375, 1.064697265625, 2.718281828459045, 0.6931471805599453,
};
static const int NSPECIAL = sizeof(SPECIAL) / sizeof(SPECIAL[0]);

int main() {
  S = 1;
  for(int c = 0; c < CHUNKS; c++) {
    uint64_t h = 1469598103934665603ULL;
    for(int k = 0; k < CHUNK; k++) { long i = (long)c * CHUNK + k; double x = genExp(i); h = mix(h, bits(exp(x))); }
    printf("C exp %d %016llX\n", c, (unsigned long long)h);
  }
  S = 2;
  for(int c = 0; c < CHUNKS; c++) {
    uint64_t h = 1469598103934665603ULL;
    for(int k = 0; k < CHUNK; k++) { long i = (long)c * CHUNK + k; double x = genLog(i); h = mix(h, bits(log(x))); }
    printf("C ln %d %016llX\n", c, (unsigned long long)h);
  }
  S = 3;
  for(int c = 0; c < CHUNKS; c++) {
    uint64_t h = 1469598103934665603ULL;
    for(int k = 0; k < CHUNK; k++) { long i = (long)c * CHUNK + k; double x, y; genPow(i, x, y); h = mix(h, bits(pow(x, y))); }
    printf("C powf %d %016llX\n", c, (unsigned long long)h);
  }
  S = 4;
  for(int c = 0; c < CHUNKS; c++) {
    uint64_t h = 1469598103934665603ULL;
    for(int k = 0; k < CHUNK; k++) { long i = (long)c * CHUNK + k; float x = genExpf(i); h = mix(h, bitsf(expf(x))); }
    printf("C expf %d %016llX\n", c, (unsigned long long)h);
  }
  for(int a = 0; a < NSPECIAL; a++) {
    double x = SPECIAL[a];
    printf("S exp %016llX %016llX\n", (unsigned long long)bits(x), (unsigned long long)bits(exp(x)));
    printf("S ln %016llX %016llX\n", (unsigned long long)bits(x), (unsigned long long)bits(log(x)));
    float xf = (float)x;
    printf("S expf %016llX %016llX\n", (unsigned long long)bitsf(xf), (unsigned long long)bitsf(expf(xf)));
    for(int b = 0; b < NSPECIAL; b++) {
      double y = SPECIAL[b];
      printf("S powf %016llX %016llX %016llX\n", (unsigned long long)bits(x), (unsigned long long)bits(y), (unsigned long long)bits(pow(x, y)));
    }
  }
  return 0;
}
