// Crafted-input oracle for the eval cache's update and adjust paths of crates/katago-core/src/search/*.rs: the unmodified
// upstream functions called directly on hand-set node and child statistics in which the lead differs from the score mean
// (a NN-less search cannot tell them apart: debugSkipNeuralNet makes whiteLead = whiteScoreMean), with distinct win-loss,
// no-result, score mean, lead and utility values. Covers EvalCacheTable::update (old-entry copy, the 0.75 replacement
// threshold, the firstExploreEvals merge at the min-visits and >= weight boundaries, the root pass rule at pass*8 against
// the total edge visits and the 0.05 utility margin, shard choice by hash0), Search::adjustEvalsFromCacheHelper (the
// cacheWeight squish, the cacheFrac, each average adjusted separately, scoreMeanSq's clamp, the utility), its two callers
// addCurrentNNOutputAsLeafValue and recomputeNodeStats (with a crafted NNOutput and crafted child stats; the root and the
// useEvalCache / useGraphSearch / mirroringPla gates), and the firstExploreEvals loop of selectBestChildToDescend (each
// eval-cache move not yet tried is a candidate with its cache utility in place of the FPU; skipped when illegal, already
// tried, a masked root move or avoided; the candidates are peeled one by one to compare their order).
// Not upstream code. Same method as search_select_diff.cpp (its header explains `#define private public`). The Rust
// mirror is crates/katago-core/tests/search_evalcache_diff.rs: it parses the inputs of every line, sets the same state,
// calls the ported functions and must print the same bytes.
//
// Build (from the repo root, after `just katago-eigen`; links the Eigen build's objects of the unmodified upstream
// sources (library dirs only); the binary goes outside the repo):
//   O=build/eigen/CMakeFiles/katago.dir; g++ -std=c++17 -O2 -Iupstream/KataGo/cpp -o /path/outside/repo/search_evalcache_diff \
//     fixtures/upstream-v1.18.2/unit/oracle/search_evalcache_diff.cpp \
//     $(find $O/core $O/game $O/neuralnet $O/search $O/dataio $O/program -name '*.o') -lz -lzip -lpthread
//   /path/outside/repo/search_evalcache_diff > fixtures/upstream-v1.18.2/unit/oracle/search_evalcache_diff.txt
// Generated with g++ (GCC) 16.2.1 / glibc 2.44 on x86_64 (FMA) Linux.
//
// Output: a header line "supports <0|1>", then position lines and call lines. Doubles are 16 hex digits of the IEEE
// bits, floats 8; counts, visits and locs decimal (raw Loc ints).
//   A <minVisits> <wlf> <nrw> <ssf> <dsf> <dscale> <rsc> <eWL f32> <eNR f32> <eSM f32> <eLead f32> <eWeight f32> <visits>
//     <wl> <nr> <sm> <smsq> <lead> <hasUtil 0|1> <util> = <wl> <nr> <sm> <smsq> <lead> <util|->
//       evalCacheMinVisits, winLossUtilityFactor, noResultUtilityForWhite, staticScoreUtilityFactor, dynamicScoreUtilityFactor,
//       dynamicScoreCenterScale, recentScoreCenter; an EvalCacheEntry (avgWinLoss, avgNoResult, avgScoreMean, avgLead,
//       cacheWeight); thisNodeVisits and the averages; adjustEvalsFromCacheHelper (utility pointer NULL if hasUtil = 0);
//       the adjusted averages (utility "-" if none).
//   UPOS <pos> pla=<root player> children=<n> pass=<index of the pass child or -1>
//       the search of position <pos> for the U, L and C lines: SearchParams() + useEvalCache, useGraphSearch,
//       evalCacheMinVisits 2, numThreads 1, maxVisits 300, cpuctExploration 20 (every root move, pass included, gets
//       visits), seed "ecseed<pos>"; 5x5 japanese rules (komi 6.5); pos 0 the empty board (black to move), pos 1 a black
//       stone in the center (white to move), pos 2 two black and two white stones (black to move). After runWholeSearch the
//       root's children in array order; the lines below set statistics on them (visits, edge visits, win-loss, score mean,
//       lead, utility, ...), and on the root, as listed.
//   U <pos> <shards> <steps> <hashA hash1> <hashA hash0> <hashB hash1> <hashB hash0> (<step>)*steps = (<out>|)*steps shards=<size,...>
//       step = <which hash 0 A|1 B> <minVisits> <isRoot> <root visits> <wl> <nr> <sm> <lead> <util> then, per root child in array
//       order, <visits> <edge visits> <wl> <sm> <lead> <util>. A fresh EvalCacheTable(<shards>); per step the root's and its
//       children's statistics are set, then EvalCacheTable::update(hash, root, minVisits, isRoot) and find(hash): out = "null",
//       or <cacheWeight> <wl> <nr> <sm> <lead> <number of firstExploreEvals> then per eval <loc> <wl> <sm> <weight> (floats);
//       shards = the size of every shard map at the end.
//   L <pos> <which node> <gate> <minVisits> <wlf> <nrw> <ssf> <dsf> <dscale> <rsc> <eWL> <eNR> <eSM> <eLead> <eWeight> <nn: win loss
//     noresult scoremean scoremeansq lead (f32)> = <stats>
//       addCurrentNNOutputAsLeafValue(node, true) on the root (which = 0) or the root's most visited child (1), whose NNOutput is
//       replaced by a copy with the six values, whose evalCacheEntry is the entry given, whose stats are zeroed first. gate: 0 all on,
//       1 useEvalCache off, 2 useGraphSearch off, 3 mirroringPla set. stats = visits then wl nr sm smsq lead util utilsq weightsum
//       weightsqsum (doubles).
//   C <pos> <which node> <gate> <minVisits> <wlf> <nrw> <ssf> <dsf> <dscale> <rsc> <eWL> <eNR> <eSM> <eLead> <eWeight> <nn: ...>
//     <node visits> (<child>)*k = <stats>
//       recomputeNodeStats(node, thread, 1, isRoot = which == 0) as L but on the node's k children (array order), each
//       <visits> <edge visits> <weightsum> <wl> <nr> <sm> <smsq> <lead> <util> <utilsq> <weightsqsum>; the node's stats before
//       are all zero except visits (<node visits>).
//   FPOS <pos> pla=<root player> children=<n> x=<loc of the most visited root child> xchildren=<n>
//       the search for the F lines: position <pos> (as UPOS), maxVisits 7, cpuctExploration 1.0 (default), seed "fcseed<pos>":
//       few root children, so that most moves are untried
//   F <pos> <depth 0 root|1 the root child x> <gate> <wlf> <ssf> <dsf> <dscale> <rsc> <cpuct> <rootSymPruning 0|1> <avoid> <dup> <fe> =
//     (<numChildrenFound>/<bestChildIdx>/<bestChildMoveLoc>/<countEdgeVisit>)*
//       the node's evalCacheEntry holds firstExploreEvals <fe> = "loc:wl:sm:weight,..." (floats), avoid = "loc/untilDepth,..." (both
//       players' avoidMoveUntilByLoc vectors, "-" none), dup = "loc,..." (rootSymDupLoc set; only the root is affected), gate as in L.
//       selectBestChildToDescend is called (with the thread moved to x's position at depth 1); while its answer is one of the
//       firstExploreEvals (bestChildIdx = numChildrenFound), that eval is erased and it is called again (at most 8 calls)
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
#include "search/evalcache.h"
#include "game/boardhistory.h"
#include "game/rules.h"
#undef private
#undef protected
using namespace std;

static unsigned long long B(double d) { uint64_t u; memcpy(&u, &d, 8); return (unsigned long long)u; }
static unsigned F(float f) { uint32_t u; memcpy(&u, &f, 4); return (unsigned)u; }

static Rand gen("search_evalcache_diff_20261024");
static double uni(double a, double b) { return a + (b - a) * gen.nextDouble(); }
static int rnd(int n) { return (int)gen.nextUInt((uint32_t)n); }
static bool prob(double p) { return gen.nextBool(p); }
template<typename T, size_t N> static T pick(const T (&a)[N]) { return a[gen.nextUInt((uint32_t)N)]; }

static const int SZ = 5;

static Search* makeSearch(NNEvaluator* e, Logger* logger, int pos, int maxVisits, double cpuct, const string& seed) {
  SearchParams qp;
  qp.numThreads = 1;
  qp.maxVisits = maxVisits;
  qp.cpuctExploration = cpuct;
  qp.useEvalCache = true;
  qp.useGraphSearch = true;
  qp.evalCacheMinVisits = 2;
  Search* q = new Search(qp, e, logger, seed);
  Rules rules = Rules::parseRules("japanese");
  rules.komi = 6.5f;
  Board board(SZ, SZ);
  BoardHistory hist(board, P_BLACK, rules, 0);
  auto play = [&](int x, int y, Player pla) { hist.makeBoardMoveAssumeLegal(board, Location::getLoc(x, y, SZ), pla, NULL); };
  if(pos == 1) play(2, 2, P_BLACK);
  if(pos == 2) { play(1, 1, P_BLACK); play(3, 3, P_WHITE); play(3, 1, P_BLACK); play(1, 3, P_WHITE); }
  Player pla = hist.presumedNextMovePla;
  q->setPosition(pla, board, hist);
  q->runWholeSearch(pla);
  return q;
}

struct Kids {
  int n = 0, passIdx = -1;
};
static Kids kidsOf(SearchNode* node) {
  Kids k;
  SearchNodeChildrenReference ch = node->getChildren();
  for(int i = 0; i < ch.getCapacity(); i++) {
    if(ch[i].getIfAllocated() == NULL) break;
    if(ch[i].getMoveLocRelaxed() == Board::PASS_LOC) k.passIdx = i;
    k.n++;
  }
  return k;
}
// the root child with the most edge visits that has children of its own (-1 if none)
static int bestChildWithKids(SearchNode* root) {
  SearchNodeChildrenReference ch = root->getChildren();
  int best = -1;
  int64_t bestVisits = -1;
  for(int i = 0; i < ch.getCapacity(); i++) {
    SearchNode* c = ch[i].getIfAllocated();
    if(c == NULL) break;
    if(c->getNNOutput() == NULL) continue;
    if(kidsOf(c).n == 0 && best >= 0) continue;
    int64_t ev = ch[i].getEdgeVisits();
    if(ev > bestVisits) { bestVisits = ev; best = i; }
  }
  return best;
}

static void setNNValues(SearchNode* node, float wp, float lp, float np, float sm, float smsq, float lead) {
  NNOutput* copy = new NNOutput(*node->getNNOutput());
  copy->whiteWinProb = wp;
  copy->whiteLossProb = lp;
  copy->whiteNoResultProb = np;
  copy->whiteScoreMean = sm;
  copy->whiteScoreMeanSq = smsq;
  copy->whiteLead = lead;
  node->nnOutput.store(new std::shared_ptr<NNOutput>(copy), std::memory_order_release);
}
static void setStatsAll(SearchNode* n, int64_t visits, double wl, double nr, double sm, double smsq, double lead, double util, double utilsq, double ws, double wsq) {
  n->stats.visits.store(visits);
  n->stats.winLossValueAvg.store(wl);
  n->stats.noResultValueAvg.store(nr);
  n->stats.scoreMeanAvg.store(sm);
  n->stats.scoreMeanSqAvg.store(smsq);
  n->stats.leadAvg.store(lead);
  n->stats.utilityAvg.store(util);
  n->stats.utilitySqAvg.store(utilsq);
  n->stats.weightSum.store(ws);
  n->stats.weightSqSum.store(wsq);
}
static void printStats(const SearchNode* n) {
  printf("%lld %016llX %016llX %016llX %016llX %016llX %016llX %016llX %016llX %016llX\n", (long long)n->stats.visits.load(),
         B(n->stats.winLossValueAvg.load()), B(n->stats.noResultValueAvg.load()), B(n->stats.scoreMeanAvg.load()), B(n->stats.scoreMeanSqAvg.load()),
         B(n->stats.leadAvg.load()), B(n->stats.utilityAvg.load()), B(n->stats.utilitySqAvg.load()), B(n->stats.weightSum.load()), B(n->stats.weightSqSum.load()));
}

// Params shared by the A, L and C lines.
struct Prm {
  int64_t minVisits;
  double wlf, nrw, ssf, dsf, dscale, rsc;
  float eWL, eNR, eSM, eLead, eW;
};
static Prm genPrm() {
  static const int64_t mins[] = {1, 2, 3, 5, 10, 100};
  static const double wlfs[] = {1.0, 0.5, 2.0};
  static const double nrws[] = {0.0, 0.0, 0.5, -0.5};
  static const double ssfs[] = {0.0, 0.1, 0.02, 0.5};
  static const double dsfs[] = {0.0, 0.3, 0.05, 1.0};
  static const double dscales[] = {0.75, 1.0, 0.5};
  Prm p;
  p.minVisits = pick(mins);
  p.wlf = pick(wlfs); p.nrw = pick(nrws); p.ssf = pick(ssfs); p.dsf = pick(dsfs); p.dscale = pick(dscales);
  p.rsc = prob(0.5) ? 0.0 : uni(-10, 10);
  p.eWL = (float)uni(-1, 1);
  p.eNR = prob(0.5) ? 0.0f : (float)uni(0, 0.3);
  p.eSM = (float)(prob(0.8) ? uni(-40, 40) : uni(-3, 3));
  p.eLead = (float)(prob(0.8) ? uni(-40, 40) : uni(-3, 3));  // independent of the score mean
  int r = rnd(8);
  double m = (double)p.minVisits;
  p.eW = (float)(r == 0 ? m : (r == 1 ? m + 1 : (r == 2 ? m * 4 : (r == 3 ? 0.5 : (r == 4 ? 1 + rnd(30) : (r == 5 ? m * 2.5 : (r == 6 ? uni(0.1, 1000) : 1e5)))))));
  return p;
}
static void setPrm(Search* s, const Prm& p) {
  s->searchParams.evalCacheMinVisits = p.minVisits;
  s->searchParams.winLossUtilityFactor = p.wlf;
  s->searchParams.noResultUtilityForWhite = p.nrw;
  s->searchParams.staticScoreUtilityFactor = p.ssf;
  s->searchParams.dynamicScoreUtilityFactor = p.dsf;
  s->searchParams.dynamicScoreCenterScale = p.dscale;
  s->recentScoreCenter = p.rsc;
}
static shared_ptr<EvalCacheEntry> makeEntry(const Prm& p) {
  auto e = make_shared<EvalCacheEntry>();
  e->avgWinLoss = p.eWL; e->avgNoResult = p.eNR; e->avgScoreMean = p.eSM; e->avgLead = p.eLead; e->cacheWeight = p.eW;
  return e;
}
static void printPrm(const Prm& p) {
  printf("%lld %016llX %016llX %016llX %016llX %016llX %016llX %08X %08X %08X %08X %08X", (long long)p.minVisits, B(p.wlf), B(p.nrw), B(p.ssf), B(p.dsf), B(p.dscale), B(p.rsc),
         F(p.eWL), F(p.eNR), F(p.eSM), F(p.eLead), F(p.eW));
}

static string entryStr(const shared_ptr<EvalCacheEntry>& e) {
  if(e == nullptr) return "null";
  char b[128];
  snprintf(b, sizeof b, "%08X %08X %08X %08X %08X %d", F(e->cacheWeight), F(e->avgWinLoss), F(e->avgNoResult), F(e->avgScoreMean), F(e->avgLead), (int)e->firstExploreEvals.size());
  string o = b;
  for(const auto& kv : e->firstExploreEvals) {
    snprintf(b, sizeof b, " %d %08X %08X %08X", (int)kv.first, F(kv.second.avgWinLoss), F(kv.second.avgScoreMean), F(kv.second.cacheWeight));
    o += b;
  }
  return o;
}

static void applyGate(Search* s, int gate) {
  s->searchParams.useEvalCache = gate != 1;
  s->searchParams.useGraphSearch = gate != 2;
  s->mirroringPla = gate == 3 ? P_BLACK : C_EMPTY;
}

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
  Search* s = new Search(params, e, &logger, "ecdiffseed");
  printf("supports %d\n", (int)e->supportsShorttermError());

  // A: adjustEvalsFromCacheHelper
  for(int i = 0; i < 700; i++) {
    Prm p = genPrm();
    setPrm(s, p);
    auto entry = makeEntry(p);
    static const int64_t vis[] = {1, 1, 2, 3, 5, 8, 17, 40, 100, 1000, 20000};
    int64_t visits = pick(vis);
    if(prob(0.15)) visits = (int64_t)p.eW;  // thisNodeVisits equal to the cache weight
    double wl = uni(-1, 1), nr = prob(0.5) ? 0.0 : uni(0, 0.3);
    double sm = prob(0.8) ? uni(-40, 40) : uni(-3, 3), lead = prob(0.8) ? uni(-40, 40) : uni(-3, 3);
    int r = rnd(5);
    double smsq = r == 0 ? 0.0 : (r == 1 ? sm * sm : (r == 2 ? sm * sm + uni(0, 300) : (r == 3 ? sm * sm - uni(0, 50) : uni(0, 1))));
    bool hasUtil = prob(0.8);
    double util = uni(-1.2, 1.2);
    printf("A ");
    printPrm(p);
    printf(" %lld %016llX %016llX %016llX %016llX %016llX %d %016llX = ", (long long)visits, B(wl), B(nr), B(sm), B(smsq), B(lead), (int)hasUtil, B(util));
    s->adjustEvalsFromCacheHelper(entry, visits, wl, nr, sm, smsq, lead, hasUtil ? &util : NULL);
    printf("%016llX %016llX %016llX %016llX %016llX ", B(wl), B(nr), B(sm), B(smsq), B(lead));
    if(hasUtil) printf("%016llX\n", B(util)); else printf("-\n");
  }

  // U, L and C: on the 300-visit search of each position
  for(int pos = 0; pos < 3; pos++) {
    Search* q = makeSearch(e, &logger, pos, 300, 20.0, "ecseed" + to_string(pos));
    SearchNode* root = q->rootNode;
    Kids rk = kidsOf(root);
    printf("UPOS %d pla=%d children=%d pass=%d\n", pos, (int)q->rootPla, rk.n, rk.passIdx);
    SearchNodeChildrenReference ch = root->getChildren();
    const bool rootWhite = root->nextPla == P_WHITE;

    // U: EvalCacheTable::update
    for(int line = 0; line < 130; line++) {
      static const int shardsChoices[] = {1, 2, 3, 7, 16};
      int shards = pick(shardsChoices);
      int steps = 1 + rnd(4);
      Hash128 hashes[2] = {Hash128(gen.nextUInt64(), gen.nextUInt64()), Hash128(gen.nextUInt64(), gen.nextUInt64())};
      EvalCacheTable table(shards);
      printf("U %d %d %d %016llX %016llX %016llX %016llX", pos, shards, steps, (unsigned long long)hashes[0].hash1, (unsigned long long)hashes[0].hash0,
             (unsigned long long)hashes[1].hash1, (unsigned long long)hashes[1].hash0);
      string outs;
      for(int step = 0; step < steps; step++) {
        int which = prob(0.7) ? 0 : 1;
        shared_ptr<EvalCacheEntry> prev = table.find(hashes[which]);
        static const int64_t mins[] = {1, 2, 3, 5};
        int64_t minV = pick(mins);
        bool isRoot = prob(0.6);
        // the root's visits: around 0.75 * the previous cache weight (the replacement threshold), or free
        int64_t visits;
        int r = rnd(6);
        float pw = prev ? prev->cacheWeight : 0.0f;
        if(prev && r < 4) {
          int64_t base = (int64_t)(pw * 0.75f);
          int64_t cands[] = {base, base - 1, base + 1, (int64_t)pw, (int64_t)pw + 5, 2 * (int64_t)pw};
          visits = max<int64_t>(1, cands[rnd(6)]);
        }
        else {
          static const int64_t vs[] = {2, 3, 4, 8, 12, 16, 20, 40, 100, 333};
          visits = pick(vs);
        }
        double wl = uni(-1, 1), nr = prob(0.5) ? 0.0 : uni(0, 0.3);
        double sm = prob(0.8) ? uni(-40, 40) : uni(-3, 3), lead = prob(0.8) ? uni(-40, 40) : uni(-3, 3);
        double util = uni(-1, 1);
        printf(" %d %lld %d %lld %016llX %016llX %016llX %016llX %016llX", which, (long long)minV, (int)isRoot, (long long)visits, B(wl), B(nr), B(sm), B(lead), B(util));
        setStatsAll(root, visits, wl, nr, sm, sm * sm, lead, util, util * util, (double)visits, (double)visits);
        // the children
        vector<int64_t> edges(rk.n);
        for(int i = 0; i < rk.n; i++) edges[i] = rnd(3) == 0 ? 0 : 1 + rnd(30);
        vector<double> selfUt(rk.n);
        for(int i = 0; i < rk.n; i++) selfUt[i] = uni(-1, 1);
        if(isRoot && rk.passIdx >= 0 && rk.n >= 2 && prob(0.75)) {
          // designed: the pass edge visits P and the total T = 8P-1, 8P, 8P+1, 9P-1, 9P, 9P+1, 10P, 4P (pass*8 >= total skips)
          static const int64_t Ps[] = {1, 2, 3, 5, 8, 13, 40};
          int64_t P = pick(Ps);
          int64_t Ts[] = {8 * P - 1, 8 * P, 8 * P + 1, 9 * P - 1, 9 * P, 9 * P + 1, 10 * P, 4 * P};
          int64_t T = max<int64_t>(P, Ts[rnd(8)]);
          int64_t rest = T - P;
          vector<int> others;
          for(int i = 0; i < rk.n; i++) if(i != rk.passIdx) others.push_back(i);
          for(int i : others) edges[i] = 0;
          // distribute `rest` over the others, one visit at a time with some structure
          for(int64_t k = 0; k < rest; k++) edges[others[rnd((int)others.size())]]++;
          edges[rk.passIdx] = P;
          // pass utility relative to the best other child
          double m = -1e50;
          for(int i : others) m = max(m, selfUt[i]);
          static const double deltas[] = {-0.045, -0.0499, -0.0501, -0.055, -0.3, 0.02, 0.0, -0.0401, -0.0399, -0.049, -0.06, 0.5};
          selfUt[rk.passIdx] = m + pick(deltas);
        }
        for(int i = 0; i < rk.n; i++) {
          SearchNode* c = ch[i].getIfAllocated();
          int64_t cv;
          int rr = rnd(7);
          int64_t fw = 0;
          if(prev) {
            auto it = prev->firstExploreEvals.find(ch[i].getMoveLocRelaxed());
            if(it != prev->firstExploreEvals.end()) fw = (int64_t)it->second.cacheWeight;
          }
          if(rr == 0) cv = 0;
          else if(rr == 1) cv = minV - 1;
          else if(rr == 2) cv = minV;
          else if(rr == 3) cv = minV + 1;
          else if(rr == 4 && fw > 0) cv = fw + (rnd(3) - 1);
          else cv = 1 + rnd(60);
          cv = max<int64_t>(0, cv);
          double cwl = uni(-1, 1), csm = prob(0.8) ? uni(-40, 40) : uni(-3, 3), clead = prob(0.8) ? uni(-40, 40) : uni(-3, 3);
          double cutil = rootWhite ? selfUt[i] : -selfUt[i];
          printf(" %lld %lld %016llX %016llX %016llX %016llX", (long long)cv, (long long)edges[i], B(cwl), B(csm), B(clead), B(cutil));
          c->stats.visits.store(cv);
          ch[i].setEdgeVisits(edges[i]);
          c->stats.winLossValueAvg.store(cwl);
          c->stats.scoreMeanAvg.store(csm);
          c->stats.leadAvg.store(clead);
          c->stats.utilityAvg.store(cutil);
        }
        table.update(hashes[which], root, minV, isRoot);
        outs += (step ? "|" : "");
        outs += entryStr(table.find(hashes[which]));
      }
      printf(" = %s shards=", outs.c_str());
      for(size_t k = 0; k < table.entries.size(); k++) printf("%s%d", k ? "," : "", (int)table.entries[k].size());
      printf("\n");
    }

    // L: addCurrentNNOutputAsLeafValue
    for(int line = 0; line < 70; line++) {
      int which = rnd(3) == 0 ? 0 : 1;
      int gate = prob(0.7) ? 0 : 1 + rnd(3);
      Prm p = genPrm();
      setPrm(q, p);
      applyGate(q, gate);
      SearchNode* node = which == 0 ? root : ch[bestChildWithKids(root) < 0 ? 0 : bestChildWithKids(root)].getIfAllocated();
      node->evalCacheEntry = makeEntry(p);
      float wp = (float)uni(0, 1), lp = (float)uni(0, 1 - wp), np = prob(0.3) ? (float)(1 - wp - lp) : 0.0f;
      float sm = (float)(prob(0.8) ? uni(-40, 40) : uni(-3, 3)), lead = (float)(prob(0.8) ? uni(-40, 40) : uni(-3, 3));
      float smsq = (float)(sm * sm + uni(0, 100));
      setNNValues(node, wp, lp, np, sm, smsq, lead);
      setStatsAll(node, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
      printf("L %d %d %d ", pos, which, gate);
      printPrm(p);
      printf(" %08X %08X %08X %08X %08X %08X = ", F(wp), F(lp), F(np), F(sm), F(smsq), F(lead));
      q->addCurrentNNOutputAsLeafValue(*node, true);
      printStats(node);
      node->evalCacheEntry = nullptr;
      applyGate(q, 0);
    }

    // C: recomputeNodeStats
    for(int line = 0; line < 50; line++) {
      int which = rnd(4) == 0 ? 0 : 1;
      int gate = prob(0.7) ? 0 : 1 + rnd(3);
      int xi = bestChildWithKids(root);
      SearchNode* node = which == 0 ? root : ch[xi < 0 ? 0 : xi].getIfAllocated();
      Prm p = genPrm();
      setPrm(q, p);
      applyGate(q, gate);
      node->evalCacheEntry = makeEntry(p);
      float wp = (float)uni(0, 1), lp = (float)uni(0, 1 - wp), np = prob(0.3) ? (float)(1 - wp - lp) : 0.0f;
      float sm = (float)(prob(0.8) ? uni(-40, 40) : uni(-3, 3)), lead = (float)(prob(0.8) ? uni(-40, 40) : uni(-3, 3));
      float smsq = (float)(sm * sm + uni(0, 100));
      setNNValues(node, wp, lp, np, sm, smsq, lead);
      int64_t nodeVisits = 1 + rnd(20);
      setStatsAll(node, nodeVisits, 0, 0, 0, 0, 0, 0, 0, 0, 0);
      printf("C %d %d %d ", pos, which, gate);
      printPrm(p);
      printf(" %08X %08X %08X %08X %08X %08X %lld", F(wp), F(lp), F(np), F(sm), F(smsq), F(lead), (long long)nodeVisits);
      SearchNodeChildrenReference nch = node->getChildren();
      for(int i = 0; i < nch.getCapacity(); i++) {
        SearchNode* c = nch[i].getIfAllocated();
        if(c == NULL) break;
        int64_t cv = rnd(6) == 0 ? 0 : 1 + rnd(30);
        int64_t ce = rnd(6) == 0 ? 0 : 1 + rnd((int)max<int64_t>(1, cv + 2));
        double cw = rnd(8) == 0 ? 0.0 : (double)max<int64_t>(1, cv) * uni(0.5, 2.0);
        double cwl = uni(-1, 1), cnr = prob(0.5) ? 0.0 : uni(0, 0.3), csm = prob(0.8) ? uni(-40, 40) : uni(-3, 3), clead = prob(0.8) ? uni(-40, 40) : uni(-3, 3);
        double csmsq = csm * csm + uni(0, 50), cutil = uni(-1, 1), cutilsq = cutil * cutil + uni(0, 0.2), cwsq = cw * cw * uni(0.3, 1.0);
        printf(" %lld %lld %016llX %016llX %016llX %016llX %016llX %016llX %016llX %016llX %016llX", (long long)cv, (long long)ce, B(cw), B(cwl), B(cnr), B(csm), B(csmsq), B(clead),
               B(cutil), B(cutilsq), B(cwsq));
        setStatsAll(c, cv, cwl, cnr, csm, csmsq, clead, cutil, cutilsq, cw, cwsq);
        nch[i].setEdgeVisits(ce);
      }
      printf(" = ");
      SearchThread thread(0, *q);
      q->recomputeNodeStats(*node, thread, 1, which == 0);
      printStats(node);
      node->evalCacheEntry = nullptr;
      applyGate(q, 0);
    }
    delete q;
  }

  // F: the firstExploreEvals loop of selectBestChildToDescend
  for(int pos = 0; pos < 3; pos++) {
    Search* q = makeSearch(e, &logger, pos, 7, 1.0, "fcseed" + to_string(pos));
    SearchNode* root = q->rootNode;
    SearchNodeChildrenReference ch = root->getChildren();
    int xi = bestChildWithKids(root);
    SearchNode* xnode = ch[xi].getIfAllocated();
    Loc xloc = ch[xi].getMoveLocRelaxed();
    printf("FPOS %d pla=%d children=%d x=%d xchildren=%d\n", pos, (int)q->rootPla, kidsOf(root).n, (int)xloc, kidsOf(xnode).n);
    for(int line = 0; line < 140; line++) {
      int depth = line % 2;
      int gate = prob(0.85) ? 0 : 1 + rnd(3);
      static const double wlfs[] = {1.0, 0.5, 2.0};
      static const double ssfs[] = {0.0, 0.1, 0.02, 0.5};
      static const double dsfs[] = {0.0, 0.3, 0.05, 1.0};
      static const double dscales[] = {0.75, 1.0, 0.5};
      static const double cpucts[] = {1.0, 0.5, 4.0, 0.2, 10.0};
      double wlf = pick(wlfs), ssf = pick(ssfs), dsf = pick(dsfs), dscale = pick(dscales), rsc = prob(0.5) ? 0.0 : uni(-5, 5), cpuct = pick(cpucts);
      int symPrune = prob(0.5) ? 1 : 0;
      // candidate locs: every point and pass
      vector<Loc> all;
      for(int y = 0; y < SZ; y++) for(int x = 0; x < SZ; x++) all.push_back(Location::getLoc(x, y, SZ));
      all.push_back(Board::PASS_LOC);
      int nfe = 1 + rnd(9);
      std::map<Loc, FirstExploreEval> fes;
      for(int k = 0; k < nfe; k++) {
        Loc l = all[rnd((int)all.size())];
        float wl = (float)(prob(0.2) ? (prob(0.5) ? 1.0 : -1.0) : uni(-1, 1));
        float sm = (float)(prob(0.2) ? 0.0 : (prob(0.7) ? uni(-15, 15) : uni(-2, 2)));
        float w = (float)(1 + rnd(50));
        fes[l] = FirstExploreEval(wl, sm, w);
      }
      string avoidSpec, dupSpec, feSpec;
      vector<int> avoidVec;
      if(prob(0.4)) {
        avoidVec.assign(Board::MAX_ARR_SIZE, 0);
        int na = 1 + rnd(3);
        for(int k = 0; k < na; k++) {
          Loc l = prob(0.7) && !fes.empty() ? std::next(fes.begin(), rnd((int)fes.size()))->first : all[rnd((int)all.size())];
          int until = 1 + rnd(3);
          avoidVec[l] = until;
        }
        for(size_t l = 0; l < avoidVec.size(); l++) if(avoidVec[l] > 0) avoidSpec += (avoidSpec.empty() ? "" : ",") + to_string(l) + "/" + to_string(avoidVec[l]);
      }
      if(avoidSpec.empty()) avoidSpec = "-";
      vector<Loc> dups;
      if(prob(0.4)) {
        int nd = 1 + rnd(3);
        for(int k = 0; k < nd; k++) {
          Loc l = prob(0.7) && !fes.empty() ? std::next(fes.begin(), rnd((int)fes.size()))->first : all[rnd((int)all.size())];
          if(l != Board::PASS_LOC && find(dups.begin(), dups.end(), l) == dups.end()) dups.push_back(l);
        }
        sort(dups.begin(), dups.end());
      }
      for(Loc l : dups) dupSpec += (dupSpec.empty() ? "" : ",") + to_string((int)l);
      if(dupSpec.empty()) dupSpec = "-";
      for(const auto& kv : fes)
        feSpec += (feSpec.empty() ? "" : ",") + to_string((int)kv.first) + ":" + [&] { char b[64]; snprintf(b, sizeof b, "%08X:%08X:%08X", F(kv.second.avgWinLoss), F(kv.second.avgScoreMean), F(kv.second.cacheWeight)); return string(b); }();

      q->searchParams.winLossUtilityFactor = wlf;
      q->searchParams.staticScoreUtilityFactor = ssf;
      q->searchParams.dynamicScoreUtilityFactor = dsf;
      q->searchParams.dynamicScoreCenterScale = dscale;
      q->recentScoreCenter = rsc;
      q->searchParams.cpuctExploration = cpuct;
      q->searchParams.rootSymmetryPruning = symPrune != 0;
      applyGate(q, gate);
      q->avoidMoveUntilByLocBlack = avoidVec;
      q->avoidMoveUntilByLocWhite = avoidVec;
      for(int k = 0; k < Board::MAX_ARR_SIZE; k++) q->rootSymDupLoc[k] = false;
      for(Loc l : dups) q->rootSymDupLoc[l] = true;

      SearchNode* node = depth == 0 ? root : xnode;
      printf("F %d %d %d %016llX %016llX %016llX %016llX %016llX %016llX %d %s %s %s =", pos, depth, gate, B(wlf), B(ssf), B(dsf), B(dscale), B(rsc), B(cpuct), symPrune,
             avoidSpec.c_str(), dupSpec.c_str(), feSpec.c_str());
      for(int call = 0; call < 8; call++) {
        auto entry = make_shared<EvalCacheEntry>();
        entry->firstExploreEvals = fes;
        node->evalCacheEntry = entry;
        SearchThread thread(0, *q);
        if(depth == 1) {
          thread.history.makeBoardMoveAssumeLegal(thread.board, xloc, thread.pla, q->rootKoHashTable);
          thread.pla = getOpp(thread.pla);
        }
        int numFound = 0, bestIdx = -1;
        Loc bestLoc = Board::NULL_LOC;
        bool countEdgeVisit = false;
        q->selectBestChildToDescend(thread, *node, node->state.load(std::memory_order_acquire), numFound, bestIdx, bestLoc, countEdgeVisit, depth == 0);
        printf(" %d/%d/%d/%d", numFound, bestIdx, (int)bestLoc, (int)countEdgeVisit);
        if(bestIdx == numFound && fes.count(bestLoc)) fes.erase(bestLoc);
        else break;
      }
      printf("\n");
      node->evalCacheEntry = nullptr;
      applyGate(q, 0);
    }
    delete q;
  }

  delete s;
  e->killServerThreads();
  delete e;
  ScoreValue::freeTables();
  return 0;
}
