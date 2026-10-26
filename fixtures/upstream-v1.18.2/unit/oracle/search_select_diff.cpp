// Crafted-input oracle for the selection / weighting / utility helpers of crates/katago-core/src/search/*.rs and
// Global::roundDynamic: the unmodified upstream functions called directly on inputs no NN-less search produces
// (extreme score means and errors, exact ties and zero weights, binding noise-pruning caps, temperatures down to
// 1e-5, onlyBelowProb cut-offs, infinite / NaN / denormal / power-of-ten roundDynamic arguments ...).
// Not upstream code. From the M3 select review's scratch harness (diff.cpp + gen_inputs.py); the inputs are generated
// here from a fixed Rand seed instead of an input file. The Rust mirror is crates/katago-core/tests/search_select_diff.rs:
// it parses the inputs of every line, calls the ported functions and must print the same bytes.
//
// The Search functions are private members: reached with `#define private public` around the upstream headers
// (after every standard header, so only KataGo's classes are affected). The Search is built as in search_nonn.cpp
// (debugSkipNeuralNet evaluator, 19x19, SearchParams(), numThreads 1); no search is run. Each op sets the
// searchParams fields (and recentScoreCenter) it lists, so lines are independent.
//
// Build (from the repo root, after `just katago-eigen`; links the Eigen build's objects of the unmodified upstream
// sources (library dirs only), so roundDynamic is the fixture compiler's code; the binary goes outside the repo):
//   O=build/eigen/CMakeFiles/katago.dir; g++ -std=c++17 -O2 -Iupstream/KataGo/cpp -o /tmp/search_select_diff \
//     fixtures/upstream-v1.18.2/unit/oracle/search_select_diff.cpp \
//     $(find $O/core $O/game $O/neuralnet $O/search $O/dataio $O/program -name '*.o') -lz -lzip -lpthread
//   /tmp/search_select_diff > fixtures/upstream-v1.18.2/unit/oracle/search_select_diff.txt
// Generated with g++ (GCC) 16.2.1 / glibc 2.44 on x86_64 (FMA) Linux.
//
// Output: a header line "sqrtBoardArea <bits> supports <0|1>", then one line per call: "<op> <inputs> = <outputs>".
// Doubles are 16 hex digits of the IEEE bits, floats 8; n and counts decimal.
//   W ue coeff maxw wlf ssf dsf dscale rsc sm(f32) stwl(f32) sts(f32) = weight deriv
//       useUncertainty=1, uncertaintyExponent, uncertaintyCoeff, uncertaintyMaxWeight, winLossUtilityFactor,
//       staticScoreUtilityFactor, dynamicScoreUtilityFactor, dynamicScoreCenterScale, recentScoreCenter; an NNOutput()
//       with whiteScoreMean, shorttermWinlossError, shorttermScoreError: computeWeightFromNNOutput and
//       getApproxScoreUtilityDerivative(whiteScoreMean)
//   S ssf dsf dscale rsc mean meanSq delta = getScoreUtility(mean, meanSq) getScoreUtilityDiff(mean, meanSq, delta)
//   T seed n temp onlyBelow p_0..p_{n-1} = index buf_0..buf_{n-1} next=<rand.nextUInt() after>
//       chooseIndexWithTemperature(Rand(seed), p, n, temp, onlyBelow, buf)
//   D vwe n subtract prune current desired (visits weight selfUtility)*n = weightAdjusted*n
//       valueWeightExponent; downweightBadChildrenAndNormalizeWeight(n, current, desired, subtract, prune, stats)
//   N scale cap n total (selfUtility weight policy)*n = result weightAdjusted*n
//       noisePruneUtilityScale, noisePruningCap; pruneNoiseWeight(stats, n, total, policy)
//   X esv scaling prior utility childWeight pla = inverse forward
//       getExploreSelectionValueInverse(esv, scaling, prior, utility, pla), getExploreSelectionValue(scaling, prior,
//       childWeight, utility, pla)
//   R x prec = Global::roundDynamic(x, prec)
//   Z wlf ssf dsf lcbStdevs = lcb radius
//       winLossUtilityFactor, staticScoreUtilityFactor, dynamicScoreUtilityFactor, lcbStdevs;
//       getSelfUtilityLCBAndRadiusZeroVisits
//   QPOS <pos> pla=<1|2> children=<n> pass=<index of the pass child or -1> moves=<loc,...>
//       position <pos> for the Q lines: a second Search (SearchParams() + fillDameBeforePass, numThreads 1,
//       maxVisits 300, cpuctExploration 20 so that every root move gets visits, seed "qseed<pos>") on 5x5 japanese
//       rules (komi 6.5), pos 0 the empty board (black to move),
//       pos 1 a black stone in the center (white to move); after runWholeSearch, the root's children in array order
//   Q <pos> <own> <pass> <child>... = <0|1>
//       shouldSuppressPass(root) after crafting: the root's nnOutput replaced by a copy whose whiteOwnerMap holds
//       QOWN[own[i]] at the i-th point (row-major, one letter a.. per point, 25 letters); the pass child's stats set
//       from <pass> = 5 digits: visits=edgeVisits=QVIS[d0], weightSum=QWGT[d1], utilityAvg=QUP[d2],
//       scoreMeanAvg=QSP[d3], leadAvg=QSP[d4]; every other child (array order) from 6 digits: visits=QVIS[d0],
//       edgeVisits=QVIS[d1], weightSum=QWGT[d2], utilityAvg=passUtility+QDU[d3], scoreMeanAvg=passScoreMean+QDS[d4],
//       leadAvg=passLead+QDS[d5] (tables below; the sums in double)
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>
#include <sstream>
#include <iostream>
#include <fstream>
#include <map>
#include <set>
#include <unordered_set>
#include <memory>
#include <atomic>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <functional>
#include <algorithm>
#define private public
#define protected public
#include "core/global.h"
#include "core/logger.h"
#include "core/rand.h"
#include "game/board.h"
#include "neuralnet/nninputs.h"
#include "neuralnet/nneval.h"
#include "search/search.h"
#include "search/searchnode.h"
#include "game/boardhistory.h"
#include "game/rules.h"
#undef private
#undef protected
using namespace std;

static unsigned long long B(double d) { uint64_t u; memcpy(&u, &d, 8); return (unsigned long long)u; }
static unsigned F(float f) { uint32_t u; memcpy(&u, &f, 4); return (unsigned)u; }

static Rand gen("search_select_diff_20261022");
static double uni(double a, double b) { return a + (b - a) * gen.nextDouble(); }
template<typename T, size_t N> static T pick(const T (&a)[N]) { return a[gen.nextUInt((uint32_t)N)]; }

int main() {
  Board::initHash();
  ScoreValue::initTables();
  NNEvaluator* e = new NNEvaluator(
    "search_nonn", "search_nonn", "", NULL, 16, 19, 19, false, true, 16, 12, true, "", "", false,
    enabled_t::False, enabled_t::False, 1, {0}, "runSearchTestsRandSeedrev", false, 0
  );
  e->spawnServerThreads();
  Logger logger(nullptr, false, false, false, false);
  SearchParams params;
  params.numThreads = 1;
  Search* s = new Search(params, e, &logger, "revseed");
  printf("sqrtBoardArea %016llX supports %d\n", B(s->rootBoard.sqrtBoardArea()), (int)e->supportsShorttermError());

  // W: computeWeightFromNNOutput + getApproxScoreUtilityDerivative
  for(int i = 0; i < 300; i++) {
    static const double ues[] = {1.0, 0.5, 2.0, 0.73, 1.3, 0.25};
    static const double coeffs[] = {0.25, 0.2, 0.05, 0.5};
    static const double maxws[] = {8.0, 2.0, 20.0};
    static const double wlfs[] = {1.0, 0.5, 2.0};
    static const double ssfs[] = {0.0, 0.1, 0.02, 0.5};
    static const double dsfs[] = {0.0, 0.3, 0.05, 1.0};
    static const double dscales[] = {0.75, 1.0, 0.5};
    double ue = pick(ues), coeff = pick(coeffs), maxw = pick(maxws), wlf = pick(wlfs), ssf = pick(ssfs), dsf = pick(dsfs), dscale = pick(dscales);
    int r = (int)gen.nextUInt(3);
    double rsc = r == 0 ? 0.0 : (r == 1 ? uni(-20, 20) : uni(-3, 3));
    r = (int)gen.nextUInt(4);
    float sm = (float)(r == 0 ? uni(-60, 60) : (r == 1 ? uni(-3, 3) : (r == 2 ? 0.0 : gen.nextGaussian() * 0.2)));
    r = (int)gen.nextUInt(3);
    float stwl = (float)(r == 0 ? 0.0 : (r == 1 ? uni(0, 1.2) : uni(0, 0.05)));
    r = (int)gen.nextUInt(3);
    float sts = (float)(r == 0 ? 0.0 : (r == 1 ? uni(0, 30) : uni(0, 2)));
    s->searchParams.useUncertainty = true;
    s->searchParams.uncertaintyExponent = ue;
    s->searchParams.uncertaintyCoeff = coeff;
    s->searchParams.uncertaintyMaxWeight = maxw;
    s->searchParams.winLossUtilityFactor = wlf;
    s->searchParams.staticScoreUtilityFactor = ssf;
    s->searchParams.dynamicScoreUtilityFactor = dsf;
    s->searchParams.dynamicScoreCenterScale = dscale;
    s->recentScoreCenter = rsc;
    NNOutput o;
    o.whiteScoreMean = sm;
    o.shorttermWinlossError = stwl;
    o.shorttermScoreError = sts;
    double w = s->computeWeightFromNNOutput(&o);
    double der = s->getApproxScoreUtilityDerivative((double)o.whiteScoreMean);
    printf("W %016llX %016llX %016llX %016llX %016llX %016llX %016llX %016llX %08X %08X %08X = %016llX %016llX\n",
           B(ue), B(coeff), B(maxw), B(wlf), B(ssf), B(dsf), B(dscale), B(rsc), F(sm), F(stwl), F(sts), B(w), B(der));
  }

  // S: getScoreUtility / getScoreUtilityDiff
  for(int i = 0; i < 300; i++) {
    static const double ssfs[] = {0.0, 0.1, 0.02, 0.5};
    static const double dsfs[] = {0.0, 0.3, 0.05, 1.0};
    static const double dscales[] = {0.75, 1.0, 0.5};
    double ssf = pick(ssfs), dsf = pick(dsfs), dscale = pick(dscales);
    double rsc = gen.nextBool(0.5) ? 0.0 : uni(-20, 20);
    int r = (int)gen.nextUInt(3);
    double m = r == 0 ? uni(-80, 80) : (r == 1 ? gen.nextGaussian() * 0.3 : uni(-5, 5));
    r = (int)gen.nextUInt(4);
    double msq = m * m + (r == 0 ? 0.0 : (r == 1 ? uni(0, 400) : (r == 2 ? uni(0, 1) : -uni(0, 1))));
    r = (int)gen.nextUInt(4);
    double dl = r == 0 ? 0.0 : (r == 1 ? uni(-2, 2) : (r == 2 ? 0.5 : -1.0 / 3.0));
    s->searchParams.staticScoreUtilityFactor = ssf;
    s->searchParams.dynamicScoreUtilityFactor = dsf;
    s->searchParams.dynamicScoreCenterScale = dscale;
    s->recentScoreCenter = rsc;
    double u = s->getScoreUtility(m, msq);
    double ud = s->getScoreUtilityDiff(m, msq, dl);
    printf("S %016llX %016llX %016llX %016llX %016llX %016llX %016llX = %016llX %016llX\n",
           B(ssf), B(dsf), B(dscale), B(rsc), B(m), B(msq), B(dl), B(u), B(ud));
  }

  // T: chooseIndexWithTemperature
  for(int i = 0; i < 200; i++) {
    int n = 1 + (int)gen.nextUInt(16);
    vector<double> ps(n);
    for(int j = 0; j < n; j++) {
      int r = (int)gen.nextUInt(6);
      ps[j] = r == 0 ? 0.0 : (r == 1 ? uni(0, 100) : (r == 2 ? uni(0, 1) : (r == 3 ? (double)gen.nextUInt(50) : (r == 4 ? -uni(0, 1) : uni(0, 1e-5)))));
    }
    double mx = 0.0;
    for(double p : ps) mx = std::max(mx, p);
    if(mx <= 0) ps[gen.nextUInt((uint32_t)n)] = uni(0.1, 10);
    static const double temps[] = {0.0, 1e-5, 0.3, 1.0, 2.0, 0.05};
    static const double obs[] = {1.0, 0.5, 0.1, 0.01, 1e-60, 0.0, 2.0};
    static const double temps2[] = {0.3, 1.0, 2.0, 0.05};
    double temp = pick(temps);
    double ob = pick(obs);
    if(ob < 1.0 && temp < 0.01) temp = pick(temps2);
    string seed = "rev" + to_string(i);
    Rand rand(seed);
    double buf[Board::MAX_ARR_SIZE];
    uint32_t idx = Search::chooseIndexWithTemperature(rand, ps.data(), n, temp, ob, buf);
    printf("T %s %d %016llX %016llX", seed.c_str(), n, B(temp), B(ob));
    for(int j = 0; j < n; j++) printf(" %016llX", B(ps[j]));
    printf(" = %u", idx);
    for(int j = 0; j < n; j++) printf(" %016llX", B(buf[j]));
    printf(" next=%u\n", rand.nextUInt());
  }

  // D: downweightBadChildrenAndNormalizeWeight
  for(int i = 0; i < 200; i++) {
    static const double vwes[] = {0.25, 0.5, 0.7, 1.0, 0.0};
    double vwe = pick(vwes);
    int n = 1 + (int)gen.nextUInt(16);
    vector<int64_t> vis(n);
    vector<double> w(n), su(n);
    for(int j = 0; j < n; j++) {
      int r = (int)gen.nextUInt(4);
      vis[j] = r == 0 ? 0 : (r == 1 ? 1 : (r == 2 ? 2 : 1 + (int64_t)gen.nextUInt(199)));
      r = (int)gen.nextUInt(5);
      w[j] = r == 0 ? uni(0.01, 50) : (r == 1 ? uni(0.5, 3) : (r == 2 ? (double)(1 + gen.nextUInt(99)) : (r == 3 ? 1e-9 : uni(0, 1000))));
      r = (int)gen.nextUInt(3);
      su[j] = r == 0 ? uni(-1.2, 1.2) : (r == 1 ? gen.nextGaussian() * 0.05 : 0.0);
    }
    bool anyVisited = false;
    for(int j = 0; j < n; j++) anyVisited = anyVisited || vis[j] > 0;
    if(!anyVisited) vis[0] = 5;
    double cur = 0.0;
    for(int j = 0; j < n; j++) cur += w[j];
    static const double subs[] = {0.0, 0.0, 0.5, 2.0};
    static const double prunes[] = {0.0, 0.0, 1.0, 3.0};
    double sub = pick(subs), prune = pick(prunes);
    if(sub > 0 || prune > 0) {
      double maxVisitedWeight = 0.0;
      for(int j = 0; j < n; j++) if(vis[j] > 0) maxVisitedWeight = std::max(maxVisitedWeight, w[j]);
      if(maxVisitedWeight <= std::max(sub, prune) + 0.01) { sub = 0.0; prune = 0.0; }
    }
    int r = (int)gen.nextUInt(3);
    double desired = r == 0 ? cur : (r == 1 ? cur * 1.3 : (cur > 1 ? cur - 0.5 : cur));
    vector<MoreNodeStats> st(n);
    for(int j = 0; j < n; j++) {
      st[j].stats.visits = vis[j];
      st[j].weightAdjusted = w[j];
      st[j].selfUtility = su[j];
    }
    s->searchParams.valueWeightExponent = vwe;
    s->downweightBadChildrenAndNormalizeWeight(n, cur, desired, sub, prune, st);
    printf("D %016llX %d %016llX %016llX %016llX %016llX", B(vwe), n, B(sub), B(prune), B(cur), B(desired));
    for(int j = 0; j < n; j++) printf(" %lld %016llX %016llX", (long long)vis[j], B(w[j]), B(su[j]));
    printf(" =");
    for(int j = 0; j < n; j++) printf(" %016llX", B(st[j].weightAdjusted));
    printf("\n");
  }

  // N: pruneNoiseWeight
  for(int i = 0; i < 200; i++) {
    static const double scs[] = {0.15, 0.05, 0.5};
    static const double caps[] = {1e50, 0.5, 2.0};
    double sc = pick(scs), cap = pick(caps);
    int n = 1 + (int)gen.nextUInt(16);
    vector<double> su(n), w(n), pol(n);
    for(int j = 0; j < n; j++) {
      su[j] = gen.nextBool(0.5) ? uni(-1.2, 1.2) : gen.nextGaussian() * 0.05;
      int r = (int)gen.nextUInt(3);
      w[j] = r == 0 ? uni(0.01, 50) : (r == 1 ? (double)(1 + gen.nextUInt(99)) : uni(0, 3));
      r = (int)gen.nextUInt(3);
      pol[j] = std::max(1e-30, r == 0 ? uni(0, 0.5) : (r == 1 ? uni(0, 0.01) : 1e-30));
    }
    double tot = 0.0;
    for(int j = 0; j < n; j++) tot += w[j];
    if(gen.nextBool(0.5)) tot = 1e-6;
    vector<MoreNodeStats> st(n);
    for(int j = 0; j < n; j++) {
      st[j].selfUtility = su[j];
      st[j].weightAdjusted = w[j];
    }
    s->searchParams.noisePruneUtilityScale = sc;
    s->searchParams.noisePruningCap = cap;
    double res = s->pruneNoiseWeight(st, n, tot, pol.data());
    printf("N %016llX %016llX %d %016llX", B(sc), B(cap), n, B(tot));
    for(int j = 0; j < n; j++) printf(" %016llX %016llX %016llX", B(su[j]), B(w[j]), B(pol[j]));
    printf(" = %016llX", B(res));
    for(int j = 0; j < n; j++) printf(" %016llX", B(st[j].weightAdjusted));
    printf("\n");
  }

  // X: getExploreSelectionValue(+Inverse)
  for(int i = 0; i < 300; i++) {
    double esv = gen.nextBool(0.5) ? uni(-2, 5) : uni(0, 1);
    double sc = gen.nextBool(0.5) ? uni(0, 50) : uni(0, 2);
    int r = (int)gen.nextUInt(4);
    double pr = r == 0 ? uni(0, 1) : (r == 1 ? -1.0 : (r == 2 ? 0.0 : uni(0, 0.01)));
    double ut = uni(-1.5, 1.5);
    double cw = gen.nextBool(0.5) ? 0.0 : uni(0, 100);
    int pla = gen.nextBool(0.5) ? 1 : 2;
    double inv = s->getExploreSelectionValueInverse(esv, sc, pr, ut, (Player)pla);
    double fwd = s->getExploreSelectionValue(sc, pr, cw, ut, (Player)pla);
    printf("X %016llX %016llX %016llX %016llX %016llX %d = %016llX %016llX\n", B(esv), B(sc), B(pr), B(ut), B(cw), pla, B(inv), B(fwd));
  }

  // R: Global::roundDynamic, on special values (infinities, NaN, zeros, the 1e-60 cut-off, denormals, huge values),
  // every power of ten in 1e-30..1e30 and its neighbours, and random values.
  {
    vector<double> vals = {0.0, -0.0, 1e-61, 1e-60, -1e-60, 1e-59, INFINITY, -INFINITY, NAN, 1e300, -1e300, 5e-324, 2.2250738585072014e-308};
    for(int k = -30; k <= 30; k++) {
      double p = strtod(("1e" + to_string(k)).c_str(), NULL);
      vals.push_back(p);
      vals.push_back(-p);
      vals.push_back(nextafter(p, 0.0));
      vals.push_back(nextafter(p, INFINITY));
      vals.push_back(p * 0.5);
      vals.push_back(p * 9.999999999);
    }
    for(int i = 0; i < 100; i++) {
      int r = (int)gen.nextUInt(5);
      double v;
      if(r == 0) v = uni(-1, 1);
      else if(r == 1) v = uni(-1000, 1000);
      else if(r == 2) v = gen.nextGaussian() * pow(10.0, (double)((int)gen.nextUInt(24) - 12));
      else if(r == 3) v = uni(0.49999, 0.50001);
      else v = ldexp(gen.nextDouble(), (int)gen.nextUInt(160) - 80);
      vals.push_back(v);
    }
    static const int precs[] = {8, 4, 0, 12, 1};
    for(double v : vals)
      for(int prec : precs)
        printf("R %016llX %d = %016llX\n", B(v), prec, B(Global::roundDynamic(v, prec)));
  }

  // Z: getSelfUtilityLCBAndRadiusZeroVisits
  for(int i = 0; i < 40; i++) {
    static const double wlfs[] = {1.0, 0.5, 2.0};
    static const double ssfs[] = {0.0, 0.1, 0.02, 0.5};
    static const double dsfs[] = {0.0, 0.3, 0.05, 1.0};
    static const double stdevs[] = {5.0, 4.0, 2.0, 6.0};
    double wlf = pick(wlfs), ssf = pick(ssfs), dsf = pick(dsfs), lcbStdevs = pick(stdevs);
    s->searchParams.winLossUtilityFactor = wlf;
    s->searchParams.staticScoreUtilityFactor = ssf;
    s->searchParams.dynamicScoreUtilityFactor = dsf;
    s->searchParams.lcbStdevs = lcbStdevs;
    double lcb = 0.0, radius = 0.0;
    s->getSelfUtilityLCBAndRadiusZeroVisits(lcb, radius);
    printf("Z %016llX %016llX %016llX %016llX = %016llX %016llX\n", B(wlf), B(ssf), B(dsf), B(lcbStdevs), B(lcb), B(radius));
  }

  // Q: shouldSuppressPass on crafted ownership and child stats (the per-move loop: opponent-owned points, adjacency to
  // owned points, the visit threshold and the utility / score / lead margins against pass)
  {
    static const float QOWN[] = {-0.999f, -0.97f, -0.951f, -0.95f, -0.6f, 0.0f, 0.6f, 0.95f, 0.951f, 0.955f, 0.96f, 0.97f, 0.999f};
    static const int64_t QVIS[] = {0, 1, 3, 10, 40, 120, 499, 500, 501, 900};
    static const double QWGT[] = {0.0, 1e-11, 0.5, 2.0, 5.0, 12.0, 40.0, 150.0, 600.0, 3000.0};
    static const double QUP[] = {-0.4, -0.1, 0.0, 0.2, 0.45};
    static const double QSP[] = {-3.0, 0.0, 2.5};
    static const double QDU[] = {-0.3, -0.2, -0.15, -0.1001, -0.1, -0.0999, -0.05, 0.0, 0.05, 0.2};
    static const double QDS[] = {-0.6, -0.5, -0.4, 0.0, 0.4, 0.5, 0.6};
    const int numOwn = (int)(sizeof(QOWN) / sizeof(QOWN[0]));
    for(int pos = 0; pos < 2; pos++) {
      SearchParams qp;
      qp.numThreads = 1;
      qp.maxVisits = 300;
      qp.fillDameBeforePass = true;
      qp.cpuctExploration = 20.0;  // explore every root move, pass included
      Search* q = new Search(qp, e, &logger, "qseed" + to_string(pos));
      Rules rules = Rules::parseRules("japanese");
      rules.komi = 6.5f;
      Board board(5, 5);
      BoardHistory hist(board, P_BLACK, rules, 0);
      if(pos == 1) hist.makeBoardMoveAssumeLegal(board, Location::getLoc(2, 2, 5), P_BLACK, NULL);
      Player pla = hist.presumedNextMovePla;
      q->setPosition(pla, board, hist);
      q->runWholeSearch(pla);
      SearchNode* root = q->rootNode;
      SearchNodeChildrenReference children = root->getChildren();
      int cap = children.getCapacity();
      int n = 0, passIdx = -1;
      string movesText;
      for(int i = 0; i < cap; i++) {
        if(children[i].getIfAllocated() == NULL) break;
        Loc mv = children[i].getMoveLocRelaxed();
        if(mv == Board::PASS_LOC) passIdx = i;
        movesText += (i ? "," : "") + to_string((int)mv);
        n++;
      }
      printf("QPOS %d pla=%d children=%d pass=%d moves=%s\n", pos, (int)pla, n, passIdx, movesText.c_str());
      NNOutput* base = root->getNNOutput();
      for(int k = 0; k < 150 && passIdx >= 0; k++) {
        string own, passCode, childCodes;
        NNOutput* copy = new NNOutput(*base);
        for(int y = 0; y < 5; y++)
          for(int x = 0; x < 5; x++) {
            // mostly extreme values, so that both owned and opponent-owned points occur next to each other
            int r = gen.nextBool(0.75) ? (gen.nextBool(0.5) ? (int)gen.nextUInt(4) : 7 + (int)gen.nextUInt(6)) : (int)gen.nextUInt((uint32_t)numOwn);
            own += (char)('a' + r);
            copy->whiteOwnerMap[NNPos::xyToPos(x, y, copy->nnXLen)] = QOWN[r];
          }
        root->nnOutput.store(new std::shared_ptr<NNOutput>(copy), std::memory_order_release);
        int pv = 1 + (int)gen.nextUInt(5), pw = (int)gen.nextUInt(10), pu = (int)gen.nextUInt(5), ps = (int)gen.nextUInt(3), pl = (int)gen.nextUInt(3);
        if(gen.nextBool(0.1)) pv = 0;
        passCode = to_string(pv) + to_string(pw) + to_string(pu) + to_string(ps) + to_string(pl);
        SearchNode* passNode = children[passIdx].getIfAllocated();
        passNode->stats.visits.store(QVIS[pv]);
        children[passIdx].setEdgeVisits(QVIS[pv]);
        passNode->stats.weightSum.store(QWGT[pw]);
        passNode->stats.utilityAvg.store(QUP[pu]);
        passNode->stats.scoreMeanAvg.store(QSP[ps]);
        passNode->stats.leadAvg.store(QSP[pl]);
        // Sparse lines: only a few children carry weight (the others are skipped as weightless), with utilities
        // around the pass utility - 0.1 margin, so that a single child decides.
        bool sparse = gen.nextBool(0.5);
        for(int i = 0; i < n; i++) {
          if(i == passIdx) continue;
          int cv = (int)gen.nextUInt(10), ce = gen.nextBool(0.7) ? cv : (int)gen.nextUInt(10), cw = (int)gen.nextUInt(10);
          int du = (int)gen.nextUInt(10);
          if(sparse) {
            if(gen.nextBool(0.85)) cw = 0;
            else du = 2 + (int)gen.nextUInt(4);
          }
          int ds = gen.nextBool(0.6) ? 3 + (int)gen.nextUInt(4) : (int)gen.nextUInt(7);
          int dl = gen.nextBool(0.6) ? 3 + (int)gen.nextUInt(4) : (int)gen.nextUInt(7);
          if(pla == P_BLACK) { ds = 6 - ds; dl = 6 - dl; }
          char code[8];
          snprintf(code, sizeof code, " %d%d%d%d%d%d", cv, ce, cw, du, ds, dl);
          childCodes += code;
          SearchNode* child = children[i].getIfAllocated();
          child->stats.visits.store(QVIS[cv]);
          children[i].setEdgeVisits(QVIS[ce]);
          child->stats.weightSum.store(QWGT[cw]);
          child->stats.utilityAvg.store(QUP[pu] + QDU[du]);
          child->stats.scoreMeanAvg.store(QSP[ps] + QDS[ds]);
          child->stats.leadAvg.store(QSP[pl] + QDS[dl]);
        }
        bool sup = q->shouldSuppressPass(root);
        printf("Q %d %s %s%s = %d\n", pos, own.c_str(), passCode.c_str(), childCodes.c_str(), (int)sup);
      }
      delete q;
    }
  }

  delete s;
  e->killServerThreads();
  delete e;
  ScoreValue::freeTables();
  return 0;
}
