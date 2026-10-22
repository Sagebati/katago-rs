// Oracle for crates/katago-core/src/search/reportedsearchvalues.rs: the unmodified upstream
// operator<<(ostream&, ReportedSearchValues) (default-precision double formatting) over generated field
// values. The constructor taking `const Search&` is not covered (the Rust one waits for search::Search).
// Links the unmodified upstream sources via the Eigen build's object files.
// The Rust mirror is crates/katago-core/tests/reportedsearchvalues_oracle.rs: keep the two in the same order.
// Build (from the repo root, after `just katago-eigen`; output outside the repo):
//   O=build/eigen/CMakeFiles/katago.dir; g++ -std=c++17 -O2 -Iupstream/KataGo/cpp -o /tmp/reportedsearchvalues \
//     fixtures/upstream-v1.16.5/unit/oracle/reportedsearchvalues.cpp \
//     $(find $O/core $O/game $O/neuralnet $O/search $O/dataio $O/program -name '*.o') -lz -lzip -lpthread
//   /tmp/reportedsearchvalues > fixtures/upstream-v1.16.5/unit/oracle/reportedsearchvalues.txt
// Output: "---- <i>" then the exact text of `cout << values`, per case.
// Generated with g++ (GCC) 16.2.1 / glibc 2.44 on x86_64 (FMA) Linux.
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <iostream>
#include <limits>
#include "core/global.h"
#include "search/reportedsearchvalues.h"
using namespace std;

static uint64_t S;
static uint64_t nxt() { uint64_t z = (S += 0x9E3779B97F4A7C15ULL); z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL; z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL; return z ^ (z >> 31); }

static double gen() {
  static const double scales[8] = {1.0, 1e-5, 1e6, 1e-9, 123456.789, 1e21, 100.0, 0.001};
  double u = (double)(nxt() >> 11) / 9007199254740992.0;
  double s = scales[nxt() % 8];
  double v = u * s;
  return (nxt() & 1) ? -v : v;
}

int main() {
  S = 0x5265706f72746564ULL;
  const double specials[] = {0.0, -0.0, 1.0, -1.0, 0.5, 0.25, 100000.0, 999999.5, 1234567.0, 1e-5, 0.0001, 1e100,
    numeric_limits<double>::infinity(), -numeric_limits<double>::infinity(), numeric_limits<double>::quiet_NaN(), 5e-324};
  const int64_t visitsList[] = {0, 1, -1, 500, 1125899906842624LL, -9223372036854775807LL - 1, 9223372036854775807LL};
  int idx = 0;
  for(double sp : specials) {
    ReportedSearchValues v;
    v.winValue = sp; v.lossValue = -sp; v.noResultValue = sp * 0.5; v.staticScoreValue = sp + 1.0; v.dynamicScoreValue = sp;
    v.expectedScore = sp * 3.0; v.expectedScoreStdev = sp; v.lead = sp - 2.0; v.winLossValue = sp; v.utility = sp * 0.1; v.weight = sp;
    v.visits = visitsList[idx % 7];
    printf("---- %d\n", idx++);
    cout << v;
    cout.flush();
  }
  for(int k = 0; k < 60; k++) {
    ReportedSearchValues v;
    v.winValue = gen(); v.lossValue = gen(); v.noResultValue = gen(); v.staticScoreValue = gen(); v.dynamicScoreValue = gen();
    v.expectedScore = gen(); v.expectedScoreStdev = gen(); v.lead = gen(); v.winLossValue = gen(); v.utility = gen(); v.weight = gen();
    v.visits = (int64_t)nxt();
    printf("---- %d\n", idx++);
    cout << v;
    cout.flush();
  }
  return 0;
}
