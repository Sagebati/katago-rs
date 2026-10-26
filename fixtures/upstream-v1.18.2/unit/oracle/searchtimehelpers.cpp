// Oracle for crates/katago-core/src/search/searchtimehelpers.rs: the three time-management helpers of the
// unmodified upstream Search (cpp/search/searchtimehelpers.cpp), called directly with deterministic inputs:
//   Search::numVisitsNeededToBeNonFutile(double)
//   Search::computeUpperBoundVisitsLeftDueToTime(int64_t rootVisits, double timeUsed, double plannedTimeLimit)
//   Search::recomputeSearchTimeLimit(const TimeControls&, double timeUsed, double searchFactor, int64_t rootVisits)
// A time-limited upstream *search* cannot be made deterministic (ClockTimer reads std::chrono::steady_clock and
// upstream is read-only), but these functions only read the Search (rootBoard, rootHistory, searchParams,
// effectiveSearchTimeCarriedOver, the root node's policy / play selection values) and their arguments, so every
// input is fixed here and every result is checked bit for bit against the port.
// Not upstream code. The Rust mirror is crates/katago-core/tests/searchtimehelpers_oracle.rs: it parses every
// line below (all inputs are on the lines), rebuilds the case and prints the same bytes.
//
// The three functions are private members of Search. They are reached WITHOUT touching the upstream headers
// (no `#define private public`): explicit template instantiation ignores access checking (the "Rob" idiom below,
// standard C++). searchParams, rootBoard, rootHistory and effectiveSearchTimeCarriedOver are public members.
//
// Build (from the repo root, after `just katago-eigen`; links the Eigen build's objects of the unmodified
// upstream sources (library dirs only); the binary goes outside the repo):
//   O=build/eigen/CMakeFiles/katago.dir; g++ -std=c++17 -O2 -Iupstream/KataGo/cpp -o /tmp/searchtimehelpers \
//     fixtures/upstream-v1.18.2/unit/oracle/searchtimehelpers.cpp \
//     $(find $O/core $O/game $O/neuralnet $O/search $O/dataio $O/program -name '*.o') -lz -lzip -lpthread
//   /tmp/searchtimehelpers > fixtures/upstream-v1.18.2/unit/oracle/searchtimehelpers.txt
// Generated with g++ (GCC) 16.2.1 / glibc 2.44 on x86_64 (FMA) Linux. Optional: ORACLE_SEED=<str> (another seed,
// exploration only; the Rust test replays another file when SEARCHTIMEHELPERS_ORACLE_FILE names it).
//
// Output. One block per case: a "CASE" line, the lines below, "END <n>".
//   CASE <n> nn=<x>x<y> nnseed=<str> board=<x>x<y> ko= score= tax= suicide= button= whb= fpo= komi=<f32 bits, 8 hex>
//     ipla=<1 B|2 W> iturn=<BoardHistory::initialTurnNumber, decimal> moves=<loc/pla,...|-> (raw Locs, each made with
//     makeBoardMoveAssumeLegal on a BoardHistory(board, ipla, rules, 0), then initialTurnNumber is set) pla=<root
//     player given to setPosition> sseed=<Search rand seed> base=<0 SearchParams()|1 forTestsV1|2 forTestsV2>
//     visits=<maxVisits of the search> run=<R runWholeSearch|B beginSearch only> sp=<k=v,...|-> tp=<k=v,...|->
//   Search = Search(base + sp (applied in order) + numThreads 1 + maxVisits=visits, nnEval, ...), setPosition, the
//   search, THEN tp (direct assignments to search->searchParams, so the search itself ignores them; numThreads
//   is only read by computeUpperBoundVisitsLeftDueToTime). Param values: doubles as 16 hex digits of the IEEE
//   bits, ints (numThreads, maxVisits) decimal.
//   I turn=<getCurrentTurnNumber> stones=<numStonesOnBoard> rv=<getRootVisits> ps=<getPolicySurpriseAndEntropy ok>
//     <surprise> <searchEntropy> <policyEntropy> (16 hex digits each) psv=<getPlaySelectionValues(.., &visitCounts, 1.0)
//     ok> n=<size> maxvisits=<bits of the largest visit count> best=<index of the best play selection value>
//   TC <i> <12 fields>        time control i of the case: originalMainTime increment mainTimeLimit maxTimePerMove
//                             originalNumPeriods numStonesPerPeriod perPeriodTime mainTimeLeft inOvertime
//                             numPeriodsLeftIncludingCurrent numStonesLeftInPeriod timeLeftInPeriod (doubles as
//                             16 hex digits, ints decimal), built with the upstream factories (absoluteTime,
//                             fischerTime, fischerCappedTime, canadianOrByoYomiTime, default = unlimited) and then
//                             moved into other states (main time used, overtime, periods and stones left); a few
//                             are inconsistent on purpose (getTime throws)
//   NF x=<bits> r=<bits>      numVisitsNeededToBeNonFutile(x)
//   UB car=<bits> rv=<n> tu=<bits> pl=<bits> r=<bits>
//                             effectiveSearchTimeCarriedOver = car, computeUpperBoundVisitsLeftDueToTime(rv, tu, pl)
//   RT car=<bits> tc=<i> tu=<bits> sf=<bits> rv=<n> r=<bits|EXC>
//                             effectiveSearchTimeCarriedOver = car, recomputeSearchTimeLimit(TC i, tu, sf, rv);
//                             EXC when a StringError was thrown (inconsistent time control)
// The searches have 0..200 visits on 5..19 wide boards, so the root node, its policy and visit counts exist
// where the obvious-moves and futile-visits branches read them. The futile branch (the best move is out of
// reach in the remaining time) is hit by choosing timeUsed as a fraction of the recommended time.
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include "core/global.h"
#include "core/rand.h"
#include "core/logger.h"
#include "game/board.h"
#include "game/rules.h"
#include "game/boardhistory.h"
#include "neuralnet/nninputs.h"
#include "neuralnet/nneval.h"
#include "search/search.h"
#include "search/searchnode.h"
#include "search/searchparams.h"
#include "search/timecontrols.h"
using namespace std;

static uint64_t dbits(double d) { uint64_t u; memcpy(&u, &d, 8); return u; }
static uint32_t fbits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
#define DB(x) ((unsigned long long)dbits(x))

//-------------------------------------------------------------------------------------------------
// Access to the private members (explicit instantiation ignores access control)

template <typename Tag, typename Tag::type M>
struct Rob {
  friend typename Tag::type robGet(Tag) { return M; }
};
struct NFTag { typedef double (Search::*type)(double); friend type robGet(NFTag); };
struct UBTag { typedef double (Search::*type)(int64_t, double, double); friend type robGet(UBTag); };
struct RTTag { typedef double (Search::*type)(const TimeControls&, double, double, int64_t); friend type robGet(RTTag); };
template struct Rob<NFTag, &Search::numVisitsNeededToBeNonFutile>;
template struct Rob<UBTag, &Search::computeUpperBoundVisitsLeftDueToTime>;
template struct Rob<RTTag, &Search::recomputeSearchTimeLimit>;

static double callNF(Search* s, double x) { return (s->*robGet(NFTag()))(x); }
static double callUB(Search* s, int64_t rv, double tu, double pl) { return (s->*robGet(UBTag()))(rv, tu, pl); }
static double callRT(Search* s, const TimeControls& tc, double tu, double sf, int64_t rv) { return (s->*robGet(RTTag()))(tc, tu, sf, rv); }

//-------------------------------------------------------------------------------------------------

static string S;  // output of the current case

static void pf(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
static void pf(const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  va_list ap2;
  va_copy(ap2, ap);
  int n = vsnprintf(NULL, 0, fmt, ap);
  va_end(ap);
  string s((size_t)n + 1, '\0');
  vsnprintf(&s[0], (size_t)n + 1, fmt, ap2);
  va_end(ap2);
  s.resize((size_t)n);
  S += s;
}

static Rand gen(getenv("ORACLE_SEED") ? getenv("ORACLE_SEED") : "searchtimehelpers_oracle_20261022");
static int rnd(int n) { return (int)gen.nextUInt((uint32_t)n); }
static bool prob(double p) { return gen.nextBool(p); }
template<typename T, size_t N> static const T& pick(const T (&a)[N]) { return a[rnd((int)N)]; }
template<typename T> static T pick(const vector<T>& a) { return a[rnd((int)a.size())]; }

// SearchParams by key: 'D' double (hex bits), 'I' int, 'L' int64. Keys the case lines use.
static bool setParam(SearchParams& p, const string& k, double d, long long l) {
#define D(name) if(k == #name) { p.name = d; return true; }
  D(lagBuffer) D(overallocateTimeFactor) D(midgameTimeFactor) D(midgameTurnPeakTime) D(endgameTurnTimeDecay)
  D(obviousMovesTimeFactor) D(obviousMovesPolicyEntropyTolerance) D(obviousMovesPolicySurpriseTolerance)
  D(futileVisitsThreshold) D(chosenMoveTemperature) D(chosenMoveTemperatureEarly) D(chosenMoveTemperatureHalflife)
  D(treeReuseCarryOverTimeFactor) D(cpuctExploration) D(rootPolicyTemperature) D(staticScoreUtilityFactor)
  D(dynamicScoreUtilityFactor)
#undef D
  if(k == "numThreads") { p.numThreads = (int)l; return true; }
  if(k == "maxVisits") { p.maxVisits = (int64_t)l; return true; }
  return false;
}
static bool isIntParam(const string& k) { return k == "numThreads" || k == "maxVisits"; }

struct PV {  // one param assignment
  string key;
  double d = 0;
  long long l = 0;
};
static string pvText(const vector<PV>& v) {
  if(v.empty()) return "-";
  string s;
  char buf[64];
  for(size_t i = 0; i < v.size(); i++) {
    if(isIntParam(v[i].key)) snprintf(buf, sizeof buf, "%lld", v[i].l);
    else snprintf(buf, sizeof buf, "%016llX", DB(v[i].d));
    s += (i ? "," : "") + v[i].key + "=" + buf;
  }
  return s;
}
static void applyPV(SearchParams& p, const vector<PV>& v) {
  for(const PV& e : v)
    if(!setParam(p, e.key, e.d, e.l)) { fprintf(stderr, "bad param %s\n", e.key.c_str()); exit(1); }
}
static void addD(vector<PV>& v, const char* k, double d) { PV e; e.key = k; e.d = d; v.push_back(e); }
static void addI(vector<PV>& v, const char* k, long long l) { PV e; e.key = k; e.l = l; v.push_back(e); }

//-------------------------------------------------------------------------------------------------
// Time controls

static void tcText(string& s, const TimeControls& t) {
  char buf[512];
  snprintf(buf, sizeof buf, "%016llX %016llX %016llX %016llX %d %d %016llX %016llX %d %d %d %016llX",
           DB(t.originalMainTime), DB(t.increment), DB(t.mainTimeLimit), DB(t.maxTimePerMove), t.originalNumPeriods,
           t.numStonesPerPeriod, DB(t.perPeriodTime), DB(t.mainTimeLeft), (int)t.inOvertime, t.numPeriodsLeftIncludingCurrent,
           t.numStonesLeftInPeriod, DB(t.timeLeftInPeriod));
  s += buf;
}

static TimeControls genTC() {
  static const double mains[] = {0.0, 0.3, 2.0, 10.0, 60.0, 300.0, 1800.0, 7200.0};
  static const double incs[] = {0.01, 0.5, 2.0, 10.0};
  static const double pers[] = {1.0, 5.0, 10.0, 30.0, 60.0};
  static const int periods[] = {1, 2, 3, 5, 8};
  static const int stones[] = {1, 1, 5, 10, 25};
  TimeControls tc;
  int kind = rnd(10);
  double main = pick(mains);
  if(kind == 0) {
    tc = TimeControls();  // unlimited
  }
  else if(kind == 1) {
    tc = TimeControls::absoluteTime(1e30);
  }
  else if(kind == 2 || kind == 3) {  // absolute
    tc = TimeControls::absoluteTime(main);
    if(prob(0.6)) tc.mainTimeLeft = main * pick(mains) / 300.0;
    if(prob(0.1)) tc.mainTimeLeft = -pick(incs);
  }
  else if(kind == 4 || kind == 5) {  // fischer
    double inc = pick(incs);
    tc = TimeControls::fischerTime(main, inc);
    if(prob(0.6)) tc.mainTimeLeft = main * (0.1 + 1.4 * gen.nextDouble());
    if(prob(0.1)) tc.mainTimeLeft = inc * 0.5;
  }
  else if(kind == 6) {  // fischer capped
    double inc = pick(incs);
    double limit = main * (1.0 + 2.0 * gen.nextDouble());
    double maxPer = prob(0.5) ? pick(pers) : TimeControls::UNLIMITED_TIME_DEFAULT_LARGE;
    tc = TimeControls::fischerCappedTime(main, inc, limit, maxPer);
    if(prob(0.5)) tc.mainTimeLeft = main * (0.2 + 1.6 * gen.nextDouble());
  }
  else {  // canadian / byo-yomi
    int np = pick(periods);
    int ns = pick(stones);
    double per = pick(pers);
    tc = TimeControls::canadianOrByoYomiTime(main, per, np, ns);
    int st = rnd(5);
    if(st == 0) {
      // still in main time, untouched
    }
    else if(st == 1) {
      tc.mainTimeLeft = main * gen.nextDouble();
    }
    else if(st == 2) {
      tc.mainTimeLeft = -per * gen.nextDouble() * 1.5;  // negative main time: slips into byo-yomi
    }
    else {  // in overtime
      tc.mainTimeLeft = 0.0;
      tc.inOvertime = true;
      tc.numPeriodsLeftIncludingCurrent = 1 + rnd(np);
      tc.numStonesLeftInPeriod = 1 + rnd(ns);
      tc.timeLeftInPeriod = per * (st == 3 ? gen.nextDouble() : 1.0);
      if(prob(0.1)) tc.timeLeftInPeriod = -per * 0.5;
    }
    if(prob(0.04)) tc.numStonesPerPeriod = 0;  // inconsistent: getTime throws
  }
  if(prob(0.03)) tc.inOvertime = !tc.inOvertime;  // possibly inconsistent: getTime throws
  return tc;
}

//-------------------------------------------------------------------------------------------------
// Case

struct Case {
  int idx = 0;
  int nnX = 19, nnY = 19;
  string nnSeed;
  int xs = 0, ys = 0;
  Rules rules;
  Player ipla = P_BLACK;
  int64_t iturn = 0;
  vector<pair<Loc, Player>> moves;
  Player pla = P_BLACK;
  string sseed;
  int base = 0;
  int64_t visits = 0;
  char run = 'R';
  vector<PV> sp, tp;
};

static SearchParams baseParams(int base) {
  if(base == 1) return SearchParams::forTestsV1();
  if(base == 2) return SearchParams::forTestsV2();
  return SearchParams();
}

static string caseLine(const Case& c) {
  char buf[512];
  snprintf(buf, sizeof buf, "CASE %d nn=%dx%d nnseed=%s board=%dx%d ko=%d score=%d tax=%d suicide=%d button=%d whb=%d fpo=%d komi=%08X ipla=%d iturn=%lld moves=",
           c.idx, c.nnX, c.nnY, c.nnSeed.c_str(), c.xs, c.ys, c.rules.koRule, c.rules.scoringRule, c.rules.taxRule,
           (int)c.rules.multiStoneSuicideLegal, (int)c.rules.hasButton, c.rules.whiteHandicapBonusRule, (int)c.rules.friendlyPassOk,
           fbits(c.rules.komi), (int)c.ipla, (long long)c.iturn);
  string s = buf;
  if(c.moves.empty()) s += "-";
  for(size_t i = 0; i < c.moves.size(); i++) {
    snprintf(buf, sizeof buf, "%s%d/%d", i ? "," : "", (int)c.moves[i].first, (int)c.moves[i].second);
    s += buf;
  }
  snprintf(buf, sizeof buf, " pla=%d sseed=%s base=%d visits=%lld run=%c sp=", (int)c.pla, c.sseed.c_str(), c.base, (long long)c.visits, c.run);
  s += buf;
  s += pvText(c.sp) + " tp=" + pvText(c.tp);
  return s;
}

// v1.18.2 NNEvaluator takes a ConfigParser (consulted by the backend only; unused with debugSkipNeuralNet): empty, as testsearchcommon.cpp does
static ConfigParser nnEvalCfg;

static NNEvaluator* newNNEval(const Case& c) {
  NNEvaluator* e = new NNEvaluator(
    "search_nonn", "search_nonn", "", NULL, 16, c.nnX, c.nnY, false, true, 16, 12, true, "",
    enabled_t::False, 1, {0}, "runSearchTestsRandSeed" + c.nnSeed, false, 0,
    nnEvalCfg
  );
  e->spawnServerThreads();
  return e;
}

static int numCases = 0, numNF = 0, numUB = 0, numRT = 0, numExc = 0, numFutileStops = 0, numPSok = 0, numUB1e30 = 0;

static void evaluate(Case& c, Search* search) {
  // Search-derived info
  {
    double su = 0, se = 0, pe = 0;
    bool ok = search->getPolicySurpriseAndEntropy(su, se, pe);
    vector<Loc> locs;
    vector<double> psvals, visitCounts;
    bool sok = search->getPlaySelectionValues(locs, psvals, &visitCounts, 1.0);
    double maxv = 0;
    int best = 0;
    for(size_t i = 0; i < visitCounts.size(); i++) {
      if(visitCounts[i] > maxv) maxv = visitCounts[i];
      if(psvals[i] > psvals[best]) best = (int)i;
    }
    pf("I turn=%lld stones=%d rv=%lld ps=%d %016llX %016llX %016llX psv=%d n=%d maxvisits=%016llX best=%d\n",
       (long long)search->rootHistory.getCurrentTurnNumber(), search->rootBoard.numStonesOnBoard(), (long long)search->getRootVisits(),
       (int)ok, DB(su), DB(se), DB(pe), (int)sok, (int)psvals.size(), DB(maxv), best);
    if(ok) numPSok++;
  }
  int64_t actualRV = search->getRootVisits();

  // Time controls of the case
  vector<TimeControls> tcs;
  int ntc = 6 + rnd(4);
  for(int i = 0; i < ntc; i++) {
    tcs.push_back(genTC());
    string s;
    tcText(s, tcs.back());
    pf("TC %d %s\n", i, s.c_str());
  }

  static const double nfx[] = {0.0, 1.0, 7.0, 100.0, 12345.5, 1e9, 3.0, 55.0};
  for(int i = 0; i < 4; i++) {
    double x = pick(nfx);
    double r = callNF(search, x);
    pf("NF x=%016llX r=%016llX\n", DB(x), DB(r));
    numNF++;
  }

  static const int64_t rvs[] = {0, 1, 2, 3, 50, 3000, 10000000};
  static const double cars[] = {0.0, 0.0, 0.02, 0.4, 5.0, 1000.0};
  static const double tus[] = {0.0, 0.05, 0.1, 0.1000001, 1.0, 10.5, 300.0};
  static const double pls[] = {0.0, 0.5, 1.0, 20.0, 1e30};
  for(int i = 0; i < 8; i++) {
    int64_t rv = prob(0.3) ? actualRV : pick(rvs);
    double car = pick(cars);
    double tu = pick(tus);
    double pl = prob(0.5) ? tu * (1.0 + 4.0 * gen.nextDouble()) : pick(pls);
    search->effectiveSearchTimeCarriedOver = car;
    double r = callUB(search, rv, tu, pl);
    pf("UB car=%016llX rv=%lld tu=%016llX pl=%016llX r=%016llX\n", DB(car), (long long)rv, DB(tu), DB(pl), DB(r));
    numUB++;
    if(r == 1e30) numUB1e30++;
  }

  static const double fracs[] = {0.0, 0.1, 0.5, 0.9, 0.97, 0.99, 1.0, 1.0, 1.3, 3.0};
  static const double sfs[] = {1.0, 1.0, 1.0, 0.5, 0.1, 1.5, 0.0};
  for(int i = 0; i < 26; i++) {
    int ti = rnd(ntc);
    const TimeControls& tc = tcs[ti];
    double car = pick(cars);
    if(prob(0.3)) car = 0.0;
    double tu;
    double minT = 0, recT = 0, maxT = 0;
    bool haveRec = false;
    try {
      tc.getTime(search->rootBoard, search->rootHistory, search->searchParams.lagBuffer, minT, recT, maxT);
      haveRec = true;
    }
    catch(const StringError&) {}
    if(haveRec && recT < 1e15 && prob(0.75)) tu = recT * pick(fracs) * (prob(0.5) ? 1.0 : search->searchParams.overallocateTimeFactor);
    else tu = pick(tus);
    double sf = pick(sfs);
    int64_t rv = prob(0.5) ? actualRV : (prob(0.5) ? actualRV * (1 + rnd(4)) : pick(rvs));
    search->effectiveSearchTimeCarriedOver = car;
    char out[64];
    try {
      double r = callRT(search, tc, tu, sf, rv);
      snprintf(out, sizeof out, "%016llX", DB(r));
      if(search->searchParams.futileVisitsThreshold > 0 && sf == 1.0 && fabs(r - tu * (1.0 - 1e-10)) <= 1e-12 * fabs(tu) && tu > 0.001) numFutileStops++;
    }
    catch(const StringError&) {
      snprintf(out, sizeof out, "EXC");
      numExc++;
    }
    pf("RT car=%016llX tc=%d tu=%016llX sf=%016llX rv=%lld r=%s\n", DB(car), ti, DB(tu), DB(sf), (long long)rv, out);
    numRT++;
  }
}

static void runCase(Case& c) {
  S.clear();
  ostringstream logSink;
  Logger logger(nullptr, false, false, false, false);
  logger.addOStream(logSink);
  NNEvaluator* nnEval = newNNEval(c);
  SearchParams params = baseParams(c.base);
  applyPV(params, c.sp);
  params.numThreads = 1;
  params.maxVisits = c.visits;
  Search* search = new Search(params, nnEval, &logger, c.sseed);
  {
    Board board(c.xs, c.ys);
    BoardHistory hist(board, c.ipla, c.rules, 0, BoardHistoryModes());
    for(const auto& m : c.moves)
      hist.makeBoardMoveAssumeLegal(board, m.first, m.second, NULL);
    hist.setInitialTurnNumber(c.iturn);
    search->setPosition(c.pla, board, hist);
  }
  if(c.run == 'R') search->runWholeSearch(c.pla);
  else search->beginSearch(false);
  applyPV(search->searchParams, c.tp);
  evaluate(c, search);
  delete search;
  nnEval->killServerThreads();
  delete nnEval;
  printf("%s\n%sEND %d\n", caseLine(c).c_str(), S.c_str(), c.idx);
  numCases++;
}

//-------------------------------------------------------------------------------------------------
// Generators

static Rules makeRules() {
  static const char* names[] = {"tromp-taylor", "chinese", "japanese", "korean", "aga", "nz", "stonescoring", "agabutton", "ancientterritory", "goe"};
  Rules r = Rules::parseRules(pick(names));
  if(prob(0.4)) {
    r.koRule = rnd(3);
    r.multiStoneSuicideLegal = rnd(2) == 0;
    r.whiteHandicapBonusRule = rnd(3);
    r.friendlyPassOk = rnd(2) == 0;
  }
  static const float komis[] = {7.5f, 6.5f, 0.0f, -3.5f, 0.5f, 5.0f, 1.0f, 7.0f, 6.0f, -1.5f, 9.5f, 2.0f};
  r.komi = pick(komis);
  return r;
}

static void pickParams(Case& c) {
  // Search-affecting (before the search)
  if(prob(0.3)) addD(c.sp, "cpuctExploration", pick(vector<double>{0.5, 1.0, 2.0, 4.0}));
  if(prob(0.3)) addD(c.sp, "rootPolicyTemperature", pick(vector<double>{0.5, 1.0, 1.5, 3.0}));
  if(prob(0.2)) addD(c.sp, "staticScoreUtilityFactor", pick(vector<double>{0.0, 0.1, 0.5}));
  if(prob(0.2)) addD(c.sp, "dynamicScoreUtilityFactor", pick(vector<double>{0.0, 0.1, 0.5}));

  // Time-related (after the search)
  static const double lag[] = {0.0, 0.1, 0.25, 1.0, 5.0};
  static const double over[] = {1.0, 1.0, 1.25, 0.8, 2.0};
  static const double mid[] = {1.0, 1.0, 1.5, 0.7, 2.5, 0.2};
  static const double peak[] = {130.0, 130.0, 50.0, 10.0, 0.0, 200.0, 300.0};
  static const double decay[] = {100.0, 100.0, 20.0, 5.0, 300.0};
  static const double obvf[] = {1.0, 1.0, 0.5, 0.2, 0.05, 0.0, 0.9};
  static const double tolE[] = {0.3, 0.3, 1.0, 3.0, 10.0, 50.0, 0.05};
  static const double tolS[] = {0.15, 0.15, 1.0, 3.0, 10.0, 50.0, 0.02};
  static const double fut[] = {0.0, 0.0, 0.1, 0.4, 0.8, 1.0, 2.0, 0.25};
  static const double temp[] = {0.0, 0.0, 0.1, 0.5, 1.0, 3.0, 0.0005};
  static const double tempE[] = {0.0, 0.8, 2.0, 0.1};
  static const double half[] = {19.0, 19.0, 5.0, 50.0, 0.001, 200.0};
  static const int thr[] = {1, 2, 4, 16, 64};
  if(prob(0.7)) addD(c.tp, "lagBuffer", pick(lag));
  if(prob(0.5)) addD(c.tp, "overallocateTimeFactor", pick(over));
  if(prob(0.6)) addD(c.tp, "midgameTimeFactor", pick(mid));
  if(prob(0.5)) addD(c.tp, "midgameTurnPeakTime", pick(peak));
  if(prob(0.5)) addD(c.tp, "endgameTurnTimeDecay", pick(decay));
  if(prob(0.7)) addD(c.tp, "obviousMovesTimeFactor", pick(obvf));
  if(prob(0.6)) addD(c.tp, "obviousMovesPolicyEntropyTolerance", pick(tolE));
  if(prob(0.6)) addD(c.tp, "obviousMovesPolicySurpriseTolerance", pick(tolS));
  if(prob(0.8)) addD(c.tp, "futileVisitsThreshold", pick(fut));
  if(prob(0.5)) addD(c.tp, "chosenMoveTemperature", pick(temp));
  if(prob(0.4)) addD(c.tp, "chosenMoveTemperatureEarly", pick(tempE));
  if(prob(0.4)) addD(c.tp, "chosenMoveTemperatureHalflife", pick(half));
  if(prob(0.2)) addD(c.tp, "treeReuseCarryOverTimeFactor", pick(vector<double>{0.0, 0.5, 1.0}));
  if(prob(0.7)) addI(c.tp, "numThreads", pick(thr));
}

int main() {
  Board::initHash();
  ScoreValue::initTables();
  const int N = 150;
  for(int i = 0; i < N; i++) {
    Case c;
    c.idx = i;
    static const int sizes[] = {5, 7, 9, 9, 13, 13, 19, 19};
    c.xs = pick(sizes);
    c.ys = prob(0.15) ? pick(sizes) : c.xs;
    c.nnX = prob(0.5) ? 19 : max(c.xs, c.ys);
    c.nnY = c.nnX;
    c.nnSeed = "tm" + to_string(i);
    c.sseed = "tmsearch" + to_string(i);
    c.rules = makeRules();
    c.ipla = prob(0.2) ? P_WHITE : P_BLACK;
    c.base = rnd(3);
    static const int64_t visitChoices[] = {0, 1, 2, 5, 10, 30, 60, 100, 200};
    c.visits = pick(visitChoices);
    c.run = c.visits == 0 ? 'B' : 'R';
    if(c.visits == 0) c.visits = 1;
    {
      // Random legal game, no passes: 0..~area moves
      Board board(c.xs, c.ys);
      BoardHistory hist(board, c.ipla, c.rules, 0, BoardHistoryModes());
      int area = c.xs * c.ys;
      int nm = prob(0.15) ? 0 : (prob(0.5) ? rnd(area / 3 + 1) : rnd(area + 1));
      for(int m = 0; m < nm; m++) {
        Player pla = hist.presumedNextMovePla;
        for(int tries = 0; tries < 40; tries++) {
          Loc l = Location::getLoc(rnd(c.xs), rnd(c.ys), c.xs);
          if(hist.isLegal(board, l, pla)) {
            hist.makeBoardMoveAssumeLegal(board, l, pla, NULL);
            c.moves.push_back(make_pair(l, pla));
            break;
          }
        }
      }
      c.pla = hist.presumedNextMovePla;
    }
    static const int64_t iturns[] = {0, 0, 0, 0, 0, 12, 100, 250, -7, 400};
    c.iturn = pick(iturns);
    pickParams(c);
    runCase(c);
  }
  fprintf(stderr, "cases=%d NF=%d UB=%d (1e30: %d) RT=%d (EXC %d, futile stops %d) cases with policy surprise ok=%d\n",
          numCases, numNF, numUB, numUB1e30, numRT, numExc, numFutileStops, numPSok);
  return 0;
}
