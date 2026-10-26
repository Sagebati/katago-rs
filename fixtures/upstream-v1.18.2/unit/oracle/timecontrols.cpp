// Oracle for search/timecontrols (all four time-control kinds, getTime, roundUpTimeLimitIfNeeded,
// toDebugString, the factories and their errors). Links against the unmodified upstream sources of
// KataGo v1.16.5. The Rust mirror is crates/katago-core/tests/timecontrols.rs.
//
// Three outputs, written into the directory given as argv[1]:
//   timecontrols_testtime.txt  stdout of the UNMODIFIED upstream Tests::runTimeControlsTests() (tests/testtime.cpp):
//                              53 time controls x 4 boards. The Rust test replays the same time controls.
//   timecontrols_cases.txt     the generated sweep inputs (one line per case: board x y stones, lagBuffer,
//                              and the 12 TimeControls fields as exact hex bits), so the Rust side does not
//                              have to reproduce the generator's Rand stream.
//   timecontrols.txt           expected results: the fixed sequence (factories, errors, unlimited checks)
//                              followed by one result line per sweep case (min/rec/max, the rounded limits,
//                              isEffectivelyUnlimitedTime, both toDebugString forms, or the thrown what()).
// Doubles are printed as 16 hex digits of their bits ("nan" for any NaN) so comparison is exact.
//
// Build (from the repo root, output outside the repo; upstream untouched; asserts on):
//   U=upstream/KataGo/cpp; g++ -std=c++17 -O1 -I$U -o /tmp/timecontrols_oracle \
//     fixtures/upstream-v1.18.2/unit/oracle/timecontrols.cpp \
//     $U/core/global.cpp $U/core/hash.cpp $U/core/rand.cpp $U/core/sha2.cpp $U/core/md5.cpp \
//     $U/core/timer.cpp $U/core/bsearch.cpp $U/core/test.cpp $U/game/board.cpp $U/game/rules.cpp \
//     $U/game/boardhistory.cpp $U/game/graphhash.cpp $U/search/timecontrols.cpp \
//     $U/tests/testtime.cpp $U/tests/testcommon.cpp -lpthread
// Run:
//   /tmp/timecontrols_oracle fixtures/upstream-v1.18.2/unit/oracle
// Generated with g++ (GCC) 16.2.1 / glibc 2.44 on x86_64 Linux. getTime calls exp(-1.0): run on x86-64 with FMA (D14).
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "core/global.h"
#include "core/rand.h"
#include "core/test.h"
#include "game/board.h"
#include "game/boardhistory.h"
#include "game/rules.h"
#include "search/timecontrols.h"
#include "tests/tests.h"
using namespace std;

static string bits(double x) {
  if(std::isnan(x))
    return "nan";
  uint64_t u;
  memcpy(&u, &x, sizeof(u));
  char buf[17];
  snprintf(buf, sizeof(buf), "%016llX", (unsigned long long)u);
  return string(buf);
}


static string fields(const TimeControls& t) {
  ostringstream o;
  o << bits(t.originalMainTime) << " " << bits(t.increment) << " " << bits(t.mainTimeLimit) << " " << bits(t.maxTimePerMove)
    << " " << t.originalNumPeriods << " " << t.numStonesPerPeriod << " " << bits(t.perPeriodTime) << " " << bits(t.mainTimeLeft)
    << " " << (t.inOvertime ? 1 : 0) << " " << t.numPeriodsLeftIncludingCurrent << " " << t.numStonesLeftInPeriod << " "
    << bits(t.timeLeftInPeriod);
  return o.str();
}

static Board makeBoard(int xs, int ys, int n) {
  Board b(xs, ys);
  int placed = 0;
  for(int y = 0; y < ys; y++) {
    for(int x = 0; x < xs; x++) {
      if(placed >= n)
        break;
      Loc loc = Location::getLoc(x, y, xs);
      Color c = ((x * 7 + y * 3) % 5 < 2) ? C_BLACK : C_WHITE;
      b.setStone(loc, c);
      placed++;
    }
  }
  return b;
}

static string q(const string& s) { return "[" + s + "]"; }

// One result line for (timeControls, board, lagBuffer). Shared by the fixed sequence and the sweep.
static string evaluate(const TimeControls& tc, const Board& board, const BoardHistory& hist, double lagBuffer) {
  ostringstream o;
  o << "stones " << board.numStonesOnBoard() << " unlimited " << (tc.isEffectivelyUnlimitedTime() ? 1 : 0);
  double minTime = -7.0, rec = -7.0, maxTime = -7.0;
  double lim = 7.5;
  try {
    tc.getTime(board, hist, lagBuffer, minTime, rec, maxTime);
    o << " getTime " << bits(minTime) << " " << bits(rec) << " " << bits(maxTime);
    lim = rec;
  } catch(const StringError& e) {
    o << " getTime throws " << q(e.what());
  }
  double rrec0 = tc.roundUpTimeLimitIfNeeded(lagBuffer, 0, lim);
  double rreclimit = tc.roundUpTimeLimitIfNeeded(lagBuffer, lim - 0.000001, lim);
  double rreclimit2 = tc.roundUpTimeLimitIfNeeded(lagBuffer, rreclimit - 0.000001, rreclimit);
  double rhalf = tc.roundUpTimeLimitIfNeeded(lagBuffer, lim * 0.5, lim);
  double rover = tc.roundUpTimeLimitIfNeeded(lagBuffer, lim + 3.0, lim);
  double rmain = tc.roundUpTimeLimitIfNeeded(lagBuffer, tc.mainTimeLeft * 0.9, lim * 3.0 + 1.0);
  o << " round " << bits(rrec0) << " " << bits(rreclimit) << " " << bits(rreclimit2) << " " << bits(rhalf) << " " << bits(rover)
    << " " << bits(rmain);
  o << " debug " << q(tc.toDebugString());
  try {
    string full = tc.toDebugString(board, hist, lagBuffer);
    o << " debugfull " << q(full);
  } catch(const StringError& e) {
    o << " debugfull throws " << q(e.what());
  }
  return o.str();
}

// ---- sweep generation (each random draw is its own statement: no GCC argument-order dependence) ----

static double pickTime(Rand& rand) {
  static const double nice[] = {0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0, 61.0, 300.0, 600.0, 3600.0, 1e19, 1e20, 1e21, 1e30, -1.0, -5.0};
  uint32_t k = rand.nextUInt(3);
  if(k == 0) {
    uint32_t i = rand.nextUInt(sizeof(nice) / sizeof(nice[0]));
    return nice[i];
  }
  else if(k == 1) {
    return rand.nextDouble(0.0, 100.0);
  }
  else {
    return rand.nextDouble(-10.0, 4000.0);
  }
}

static double pickLag(Rand& rand) {
  static const double nice[] = {0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0};
  uint32_t k = rand.nextUInt(4);
  if(k == 0) {
    return rand.nextDouble(0.0, 12.0);
  }
  uint32_t i = rand.nextUInt(sizeof(nice) / sizeof(nice[0]));
  return nice[i];
}

static TimeControls genTimeControls(Rand& rand, int kind) {
  TimeControls tc;
  if(kind == 0) {
    return tc;
  }
  else if(kind == 1) {
    double mainTime = pickTime(rand);
    tc = TimeControls::absoluteTime(mainTime);
    uint32_t k = rand.nextUInt(3);
    if(k == 1)
      tc.mainTimeLeft = pickTime(rand);
    else if(k == 2)
      tc.mainTimeLeft = mainTime * rand.nextDouble();
    return tc;
  }
  else if(kind == 2 || kind == 3) {
    double mainTime = pickTime(rand);
    double inc = pickTime(rand);
    if(kind == 2) {
      tc = TimeControls::fischerTime(mainTime, inc);
    }
    else {
      double extra = pickTime(rand);
      double limit = mainTime + (extra < 0 ? -extra : extra);
      double maxPer = pickTime(rand);
      tc = TimeControls::fischerCappedTime(mainTime, inc, limit, maxPer);
    }
    uint32_t k = rand.nextUInt(3);
    if(k == 1)
      tc.mainTimeLeft = pickTime(rand);
    else if(k == 2)
      tc.mainTimeLeft = mainTime * rand.nextDouble();
    return tc;
  }
  else if(kind == 4 || kind == 5) {
    double mainTime = pickTime(rand);
    double per = rand.nextDouble(1.0, 60.0);
    int numPeriods = 1 + (int)rand.nextUInt(12);
    int stones = 1 + (int)rand.nextUInt(5);
    tc = TimeControls::canadianOrByoYomiTime(mainTime, per, numPeriods, stones);
    uint32_t k = rand.nextUInt(4);
    if(k == 0) {
      // still in main time, possibly part of it used
      tc.mainTimeLeft = mainTime * rand.nextDouble();
    }
    else if(k == 1) {
      // in overtime, consistent state
      tc.mainTimeLeft = 0.0;
      tc.inOvertime = true;
      tc.numPeriodsLeftIncludingCurrent = 1 + (int)rand.nextUInt(numPeriods);
      tc.numStonesLeftInPeriod = 1 + (int)rand.nextUInt(stones);
      tc.timeLeftInPeriod = rand.nextDouble(-0.2 * per, per);
    }
    else if(k == 2) {
      // negative main time, not yet moved into overtime
      tc.mainTimeLeft = -rand.nextDouble(0.0, 2.5 * per);
    }
    else {
      tc.mainTimeLeft = pickTime(rand);
    }
    if(kind == 5) {
      // possibly inconsistent: perturb one thing
      uint32_t j = rand.nextUInt(6);
      if(j == 0) tc.numPeriodsLeftIncludingCurrent = (int)rand.nextUInt(numPeriods + 3);
      else if(j == 1) tc.numStonesLeftInPeriod = (int)rand.nextUInt(stones + 2) - 1;
      else if(j == 2) tc.inOvertime = !tc.inOvertime;
      else if(j == 3) tc.mainTimeLimit = pickTime(rand);
      else if(j == 4) tc.numStonesPerPeriod = (int)rand.nextUInt(3) - 1;
      else tc.timeLeftInPeriod = pickTime(rand);
    }
    return tc;
  }
  else if(kind == 6) {
    // maxTimePerMove / mainTimeLimit variants of fischer
    double mainTime = rand.nextDouble(10.0, 1000.0);
    double inc = rand.nextDouble(0.0, 30.0);
    double limit = mainTime + rand.nextDouble(0.0, 20.0);
    double maxPer = rand.nextDouble(1.0, 60.0);
    tc = TimeControls::fischerCappedTime(mainTime, inc, limit, maxPer);
    tc.mainTimeLeft = mainTime * rand.nextDouble();
    return tc;
  }
  else {
    // garbage fields, including inconsistent combinations
    tc.originalMainTime = pickTime(rand);
    tc.increment = pickTime(rand);
    tc.mainTimeLimit = pickTime(rand);
    tc.maxTimePerMove = pickTime(rand);
    tc.originalNumPeriods = (int)rand.nextUInt(8) - 1;
    tc.numStonesPerPeriod = (int)rand.nextUInt(6) - 1;
    tc.perPeriodTime = pickTime(rand);
    tc.mainTimeLeft = pickTime(rand);
    tc.inOvertime = rand.nextBool(0.5);
    tc.numPeriodsLeftIncludingCurrent = (int)rand.nextUInt(8) - 1;
    tc.numStonesLeftInPeriod = (int)rand.nextUInt(6) - 1;
    tc.timeLeftInPeriod = pickTime(rand);
    return tc;
  }
}

int main(int argc, char** argv) {
  if(argc < 2) {
    cerr << "usage: timecontrols_oracle <outdir>" << endl;
    return 1;
  }
  string dir = argv[1];
  Board::initHash();

  // 1) upstream's own test, unmodified, with stdout redirected into the fixture
  {
    ofstream f(dir + "/timecontrols_testtime.txt", ios::binary);
    streambuf* old = cout.rdbuf(f.rdbuf());
    Tests::runTimeControlsTests();
    cout.flush();
    cout.rdbuf(old);
  }

  ostringstream res;
  ostringstream cases;
  Rules rules;

  // 2) fixed sequence
  {
    Board b9 = makeBoard(9, 9, 0);
    BoardHistory h9(b9, P_BLACK, rules, 0);
    Board b19 = makeBoard(19, 19, 150);
    BoardHistory h19(b19, P_BLACK, rules, 0);

    res << "constants " << bits(TimeControls::UNLIMITED_TIME_THRESHOLD) << " " << bits(TimeControls::MAX_USER_INPUT_TIME) << " "
        << bits(TimeControls::UNLIMITED_TIME_DEFAULT) << " " << bits(TimeControls::UNLIMITED_TIME_DEFAULT_LARGE) << "\n";

    vector<pair<string, TimeControls>> named;
    named.push_back(make_pair(string("default"), TimeControls()));
    named.push_back(make_pair(string("absolute 600"), TimeControls::absoluteTime(600.0)));
    named.push_back(make_pair(string("absolute -3"), TimeControls::absoluteTime(-3.0)));
    named.push_back(make_pair(string("absolute 1e21"), TimeControls::absoluteTime(1e21)));
    named.push_back(make_pair(string("fischer 600 10"), TimeControls::fischerTime(600.0, 10.0)));
    named.push_back(make_pair(string("fischer 0 5"), TimeControls::fischerTime(0.0, 5.0)));
    named.push_back(make_pair(string("fischer 1e30 5"), TimeControls::fischerTime(1e30, 5.0)));
    named.push_back(make_pair(string("capped 600 10 700 30"), TimeControls::fischerCappedTime(600.0, 10.0, 700.0, 30.0)));
    named.push_back(make_pair(string("capped 600 10 600 1e30"), TimeControls::fischerCappedTime(600.0, 10.0, 600.0, 1e30)));
    named.push_back(make_pair(string("capped 600 0 1e25 1e25"), TimeControls::fischerCappedTime(600.0, 0.0, 1e25, 1e25)));
    named.push_back(make_pair(string("byo 600 30 5 1"), TimeControls::canadianOrByoYomiTime(600.0, 30.0, 5, 1)));
    named.push_back(make_pair(string("byo 0 30 1 1"), TimeControls::canadianOrByoYomiTime(0.0, 30.0, 1, 1)));
    named.push_back(make_pair(string("canadian 3600 300 1 25"), TimeControls::canadianOrByoYomiTime(3600.0, 300.0, 1, 25)));
    named.push_back(make_pair(string("byo 1e30 30 3 1"), TimeControls::canadianOrByoYomiTime(1e30, 30.0, 3, 1)));
    {
      TimeControls t = TimeControls::canadianOrByoYomiTime(10.0, 30.0, 5, 1);
      t.mainTimeLeft = 0.0; t.inOvertime = true; t.numPeriodsLeftIncludingCurrent = 3; t.numStonesLeftInPeriod = 1; t.timeLeftInPeriod = 1e30;
      named.push_back(make_pair(string("byo overtime unlimited period"), t));
    }
    {
      TimeControls t;
      t.inOvertime = true; t.timeLeftInPeriod = 1e21; t.mainTimeLeft = 5.0;
      named.push_back(make_pair(string("unlimited via overtime period"), t));
    }
    {
      TimeControls t;
      t.maxTimePerMove = 10.0;
      named.push_back(make_pair(string("default with maxTimePerMove"), t));
    }
    {
      TimeControls t;
      t.mainTimeLeft = 1e20;
      named.push_back(make_pair(string("mainTimeLeft exactly threshold"), t));
    }
    {
      TimeControls t;
      t.mainTimeLeft = 1.0000001e20; t.maxTimePerMove = 1.0000001e20;
      named.push_back(make_pair(string("just over threshold"), t));
    }
    for(size_t i = 0; i < named.size(); i++) {
      res << "named " << q(named[i].first) << " fields " << fields(named[i].second) << "\n";
      res << "named " << q(named[i].first) << " b9 lag1 " << evaluate(named[i].second, b9, h9, 1.0) << "\n";
      res << "named " << q(named[i].first) << " b19 lag0 " << evaluate(named[i].second, b19, h19, 0.0) << "\n";
      res << "named " << q(named[i].first) << " b19 lag7 " << evaluate(named[i].second, b19, h19, 7.0) << "\n";
    }

    // fischerCappedTime error
    {
      double args[][4] = {{600.0, 10.0, 599.0, 30.0}, {600.0, 10.0, 600.0, 30.0}, {0.0, 0.0, -1.0, 0.0}, {5.0, 1.0, 1e40, 1e40}};
      for(auto& a : args) {
        try {
          TimeControls t = TimeControls::fischerCappedTime(a[0], a[1], a[2], a[3]);
          res << "capped " << bits(a[0]) << " " << bits(a[1]) << " " << bits(a[2]) << " " << bits(a[3]) << " ok " << fields(t) << "\n";
        } catch(const StringError& e) {
          res << "capped " << bits(a[0]) << " " << bits(a[1]) << " " << bits(a[2]) << " " << bits(a[3]) << " throws " << q(e.what()) << "\n";
        }
      }
    }
  }

  // 3) sweep
  {
    Rand rand("timecontrols oracle sweep");
    const int N = 4000;
    for(int i = 0; i < N; i++) {
      uint32_t kindDraw = rand.nextUInt(10);
      int kind = (int)kindDraw;
      if(kind >= 7) kind = 7;
      TimeControls tc = genTimeControls(rand, kind);
      uint32_t xsDraw = rand.nextUInt(18);
      uint32_t ysDraw = rand.nextUInt(18);
      int xs = 2 + (int)xsDraw;
      int ys = 2 + (int)ysDraw;
      uint32_t nDraw = rand.nextUInt(xs * ys + 1);
      int n = (int)nDraw;
      double lag = pickLag(rand);

      cases << xs << " " << ys << " " << n << " " << bits(lag) << " " << fields(tc) << "\n";

      Board board = makeBoard(xs, ys, n);
      BoardHistory hist(board, P_BLACK, rules, 0);
      res << "case " << i << " " << evaluate(tc, board, hist, lag) << "\n";
    }
  }

  {
    ofstream f(dir + "/timecontrols_cases.txt", ios::binary);
    f << cases.str();
  }
  {
    ofstream f(dir + "/timecontrols.txt", ios::binary);
    f << res.str();
  }
  return 0;
}
