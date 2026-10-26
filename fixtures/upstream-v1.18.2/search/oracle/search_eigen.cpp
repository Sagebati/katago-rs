// Real-net search smoke oracle: the UNMODIFIED upstream Search (numSearchThreads = 1) over the upstream
// NNEvaluator with the Eigen fp32 CPU backend, on a real network (g170-b6c96-s175395328-d26788732, the net
// katago-nn-burn's `g170-b6c96` was converted from), on a few dozen positions x SearchParams x maxVisits.
// Not upstream code. Unlike search_nonn.cpp (bit-exact, no net) this is a TOLERANCE comparison: the
// consumer crates/katago-nn-burn/tests/search_smoke.rs runs the same cases through the ported Search over
// the burn flex backend, whose conv accumulation order differs from Eigen's, so small float differences
// propagate into the tree. The numbers it reports (identical root visit vectors? identical top move? max
// value diffs by maxVisits) inform the search tolerance policy (docs/PORTING.md section 13). Floats are
// therefore printed as decimal %.9g, not hex.
//
// Settings: nnXLen = nnYLen = 19 (every board size runs on the 19x19 net input, masked), maxBatchSize 16,
// nnCacheSizePowerOfTwo 16 (mutex pool 12), defaultSymmetry 0, doRandomize false (deterministic NN),
// requireExactNNLen false, inputsUseNHWC true, fp16 False, NHWC Auto, 1 server thread. One evaluator for
// all cases (a cache hit returns the stored output unchanged, so sharing changes no result and saves time;
// the Rust side shares one too). SearchParams: base SearchParams() (base=0) or forTestsV1() (base=1), plus
// the per-case overrides, with numThreads = 1 and chosenMoveTemperature = chosenMoveTemperatureEarly = 0
// (the top move is deterministic) and maxVisits as listed.
//
// Build (from the repo root, after `just katago-eigen`; links the Release objects of the Eigen build of the
// unmodified upstream sources, library dirs only; the binary goes outside the repo):
//   B=build/eigen; U=upstream/KataGo/cpp; O=$B/CMakeFiles/katago.dir
//   g++ -std=gnu++17 -O2 -mfpmath=sse -DNO_GIT_REVISION -DUSE_EIGEN_BACKEND -I$U -I$U/external -I$B \
//     -isystem $U/external/filesystem-1.5.8/include -isystem build/deps/eigen \
//     -o /tmp/search_eigen fixtures/upstream-v1.18.2/search/oracle/search_eigen.cpp \
//     $(find $O/core $O/game $O/neuralnet $O/search $O/dataio $O/program -name '*.o') -lz -lzip -lpthread
// Generate (the positions file supplies the three real-game positions; the header lines of the output record
// the compiler, glibc and CPU of the run):
//   /tmp/search_eigen upstream/KataGo/cpp/tests/models/g170-b6c96-s175395328-d26788732.bin.gz \
//     fixtures/upstream-v1.18.2/nn/positions.txt > fixtures/upstream-v1.18.2/search/oracle/search_eigen_b6c96.txt
// Generated with g++ (GCC) 16.2.1 / glibc 2.44 on x86-64 (Intel i7-8700K, FMA), Eigen 3.4.0.
//
// Output. Lines starting with '#' are comments. One block per search:
//   CASE <n> name=<label> board=<x>x<y> ko=<0 simple|1 positional|2 situational> score=<0 area|1 territory>
//     tax=<0 none|1 seki|2 all> suicide=<0|1> button=<0|1> whb=<0..2> fpo=<friendlyPassOk 0|1> komi=<f32 bits, 8 hex>
//     ipla=<initial player 1 B|2 W> pla=<root player> moves=<loc/pla,...|-> (raw Loc ints: Board(x,y),
//     BoardHistory(board,ipla,rules,0), makeBoardMoveAssumeLegal(board,loc,pla,NULL) for each) sseed=<Search rand
//     seed> base=<0 SearchParams()|1 forTestsV1> maxvisits=<n> params=<key=value,...|->
//   root visits=<n>
//   c <i> <move> loc=<loc> edge=<edge visits> visits=<child node visits> wl=<winLossValue> sm=<scoreMean>
//     lead=<lead> util=<utility> prior=<policy prior>      one per root child, children-array order
//   children=<n>
//   RV win=<winValue> loss=<lossValue> es=<expectedScore> lead=<lead> wl=<winLossValue> util=<utility>   (getRootValues)
//   chosen loc=<loc> <move>                                                       (getChosenMoveLoc)
//   END <n>
// Values are white's perspective (AnalysisData / ReportedSearchValues convention). Params keys: the SearchParams
// field names handled by setParam below (useGraphSearch, rootSymmetryPruning, ...).
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include <gnu/libc-version.h>
#include "core/global.h"
#include "core/logger.h"
#include "core/test.h"
#include "game/board.h"
#include "game/boardhistory.h"
#include "game/rules.h"
#include "neuralnet/nneval.h"
#include "neuralnet/nninputs.h"
#include "search/analysisdata.h"
#include "search/reportedsearchvalues.h"
#include "search/search.h"
#include "search/searchnode.h"
#include "search/searchparams.h"
using namespace std;

static uint32_t fbits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }

//-------------------------------------------------------------------------------------------------
// Deterministic generator for the random games

static uint64_t S = 0x5EA2C4B1D3F00D17ULL;
static uint64_t nxt() {
  uint64_t z = (S += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}
static int rnd(int n) { return (int)(nxt() % (uint64_t)n); }

//-------------------------------------------------------------------------------------------------
// SearchParams by key=value text (the keys this oracle uses)

static bool setParam(SearchParams& p, const string& k, const string& v) {
  double d = strtod(v.c_str(), NULL);
  bool b = v == "1";
  if(k == "useGraphSearch") { p.useGraphSearch = b; return true; }
  if(k == "rootSymmetryPruning") { p.rootSymmetryPruning = b; return true; }
  if(k == "chosenMoveTemperature") { p.chosenMoveTemperature = d; return true; }
  if(k == "chosenMoveTemperatureEarly") { p.chosenMoveTemperatureEarly = d; return true; }
  return false;
}

//-------------------------------------------------------------------------------------------------
// Positions

struct Position {
  string name;
  int xs, ys;
  Rules rules;
  vector<pair<Loc, Player>> moves;
};

struct PosBuilder {
  Position p;
  Board board;
  BoardHistory hist;
  PosBuilder(const string& name, int xs, int ys, const string& rulesName, float komi) {
    p.name = name; p.xs = xs; p.ys = ys;
    p.rules = Rules::parseRules(rulesName);
    p.rules.komi = komi;
    board = Board(xs, ys);
    hist = BoardHistory(board, P_BLACK, p.rules, 0);
  }
  void play(int x, int y, Player pla) {
    Loc loc = Location::getLoc(x, y, p.xs);
    testAssert(hist.isLegal(board, loc, pla));
    hist.makeBoardMoveAssumeLegal(board, loc, pla, NULL);
    p.moves.push_back(make_pair(loc, pla));
  }
  void playLoc(Loc loc, Player pla) {
    testAssert(hist.isLegal(board, loc, pla));
    hist.makeBoardMoveAssumeLegal(board, loc, pla, NULL);
    p.moves.push_back(make_pair(loc, pla));
  }
  // Random legal moves, never filling one's own simple eye, no passes; alternating colors.
  void randomMoves(int numMoves) {
    for(int i = 0; i < numMoves; i++) {
      Player pla = hist.presumedNextMovePla;
      vector<Loc> legal;
      for(int y = 0; y < p.ys; y++)
        for(int x = 0; x < p.xs; x++) {
          Loc loc = Location::getLoc(x, y, p.xs);
          if(board.colors[loc] == C_EMPTY && hist.isLegal(board, loc, pla) && !board.isSimpleEye(loc, pla))
            legal.push_back(loc);
        }
      testAssert(legal.size() > 0);
      playLoc(legal[rnd((int)legal.size())], pla);
    }
    testAssert(!hist.isGameFinished);
  }
  Position done() { return p; }
};

// The real-game positions of fixtures/upstream-v1.18.2/nn/positions.txt (ids): no setup stones, black first, no
// encore. Format: see nn/oracle/nneval_eigen.cpp.
static Position loadRealPosition(const string& positionsFile, int id, const string& name) {
  ifstream in(positionsFile);
  testAssert(in.good());
  string line;
  for(int i = 0; i <= id; i++)
    testAssert((bool)getline(in, line));
  istringstream ls(line);
  string tag, komiHex;
  int pid, xs, ys, ko, sc, tax, suic, button, whb, fpok, enc, ipla, amh;
  ls >> tag >> pid >> xs >> ys >> ko >> sc >> tax >> suic >> button >> whb >> fpok >> komiHex >> enc >> ipla >> amh;
  testAssert(tag == "pos" && pid == id && enc == 0 && ipla == P_BLACK && amh == 0);
  size_t nSetup; ls >> nSetup;
  testAssert(nSetup == 0);
  size_t nMoves; ls >> nMoves;
  uint32_t kb = (uint32_t)stoul(komiHex, nullptr, 16);
  float komi; memcpy(&komi, &kb, 4);
  Position p;
  p.name = name; p.xs = xs; p.ys = ys;
  p.rules = Rules(ko, sc, tax, suic != 0, button != 0, whb, fpok != 0, komi);
  Board board(xs, ys);
  BoardHistory hist(board, P_BLACK, p.rules, 0);
  for(size_t i = 0; i < nMoves; i++) {
    int x, y, pla, pe;
    ls >> x >> y >> pla >> pe;
    testAssert(!ls.fail() && pe == 0);
    Loc loc = x == -1 ? Board::PASS_LOC : Location::getLoc(x, y, xs);
    testAssert(hist.isLegal(board, loc, (Player)pla));
    hist.makeBoardMoveAssumeLegal(board, loc, (Player)pla, NULL);
    p.moves.push_back(make_pair(loc, (Player)pla));
  }
  testAssert(!hist.isGameFinished);
  return p;
}

//-------------------------------------------------------------------------------------------------
// Cases

struct CaseSpec {
  int pos;
  int base;
  int64_t maxVisits;
  vector<pair<string, string>> params;
};

static string caseLine(int idx, const Position& p, const CaseSpec& c, Player pla, const string& sseed) {
  char buf[512];
  snprintf(buf, sizeof buf,
           "CASE %d name=%s board=%dx%d ko=%d score=%d tax=%d suicide=%d button=%d whb=%d fpo=%d komi=%08X ipla=%d pla=%d moves=",
           idx, p.name.c_str(), p.xs, p.ys, p.rules.koRule, p.rules.scoringRule, p.rules.taxRule, (int)p.rules.multiStoneSuicideLegal,
           (int)p.rules.hasButton, p.rules.whiteHandicapBonusRule, (int)p.rules.friendlyPassOk, fbits(p.rules.komi), (int)P_BLACK, (int)pla);
  string s = buf;
  if(p.moves.empty()) s += "-";
  for(size_t i = 0; i < p.moves.size(); i++) {
    snprintf(buf, sizeof buf, "%s%d/%d", i ? "," : "", (int)p.moves[i].first, (int)p.moves[i].second);
    s += buf;
  }
  snprintf(buf, sizeof buf, " sseed=%s base=%d maxvisits=%lld params=", sseed.c_str(), c.base, (long long)c.maxVisits);
  s += buf;
  if(c.params.empty()) s += "-";
  for(size_t i = 0; i < c.params.size(); i++)
    s += (i ? "," : "") + c.params[i].first + "=" + c.params[i].second;
  return s;
}

static void runSearch(int idx, const Position& p, const CaseSpec& c, NNEvaluator* nnEval, Logger* logger) {
  Board board(p.xs, p.ys);
  BoardHistory hist(board, P_BLACK, p.rules, 0);
  for(const auto& m : p.moves)
    hist.makeBoardMoveAssumeLegal(board, m.first, m.second, NULL);
  Player pla = hist.presumedNextMovePla;
  string sseed = "smoke" + to_string(idx);

  SearchParams params = c.base == 1 ? SearchParams::forTestsV1() : SearchParams();
  params.numThreads = 1;
  testAssert(setParam(params, "chosenMoveTemperature", "0") && setParam(params, "chosenMoveTemperatureEarly", "0"));
  for(const auto& kv : c.params)
    testAssert(setParam(params, kv.first, kv.second));
  params.maxVisits = c.maxVisits;

  Search* search = new Search(params, nnEval, logger, sseed);
  search->setPosition(pla, board, hist);
  search->runWholeSearch(pla);

  printf("%s\n", caseLine(idx, p, c, pla, sseed).c_str());
  printf("root visits=%lld\n", (long long)search->getRootVisits());
  const SearchNode* root = search->getRootNode();
  testAssert(root != NULL);
  vector<AnalysisData> ad;
  search->getAnalysisData(ad, 0, false, 1, false);
  map<Loc, const AnalysisData*> byMove;
  for(const AnalysisData& d : ad)
    byMove[d.move] = &d;
  testAssert(byMove.size() == ad.size());
  ConstSearchNodeChildrenReference children = root->getChildren();
  int cap = children.getCapacity();
  int n = 0;
  for(int i = 0; i < cap; i++) {
    const SearchNode* child = children[i].getIfAllocated();
    if(child == NULL)
      break;
    Loc mv = children[i].getMoveLoc();
    auto it = byMove.find(mv);
    testAssert(it != byMove.end());
    const AnalysisData& d = *it->second;
    testAssert(d.numVisits == children[i].getEdgeVisits());
    printf("c %d %s loc=%d edge=%lld visits=%lld wl=%.9g sm=%.9g lead=%.9g util=%.9g prior=%.9g\n", i,
           Location::toString(mv, search->getRootBoard()).c_str(), (int)mv, (long long)children[i].getEdgeVisits(),
           (long long)child->stats.visits.load(), d.winLossValue, d.scoreMean, d.lead, d.utility, d.policyPrior);
    n++;
  }
  testAssert(n == (int)ad.size());
  printf("children=%d\n", n);
  ReportedSearchValues v;
  bool suc = search->getRootValues(v);
  testAssert(suc);
  printf("RV win=%.9g loss=%.9g es=%.9g lead=%.9g wl=%.9g util=%.9g\n", v.winValue, v.lossValue, v.expectedScore, v.lead, v.winLossValue, v.utility);
  Loc chosen = search->getChosenMoveLoc();
  printf("chosen loc=%d %s\n", (int)chosen, Location::toString(chosen, search->getRootBoard()).c_str());
  printf("END %d\n", idx);
  delete search;
}

int main(int argc, char** argv) {
  if(argc != 3) {
    fprintf(stderr, "usage: %s <model.bin.gz> <nn/positions.txt>\n", argv[0]);
    return 1;
  }
  string modelFile = argv[1], positionsFile = argv[2];
  ScoreValue::initTables();  // upstream's main(): "MUST BE CALLED AT PROGRAM START!"
  Board::initHash();

  // CPU model for the header
  string cpu = "unknown";
  {
    ifstream ci("/proc/cpuinfo");
    string l;
    while(getline(ci, l))
      if(l.rfind("model name", 0) == 0) { cpu = l.substr(l.find(':') + 2); break; }
  }

  vector<Position> pos;
  {
    pos.push_back(PosBuilder("empty19", 19, 19, "tromp-taylor", 7.5f).done());                     // 0
    pos.push_back(PosBuilder("empty9", 9, 9, "chinese", 7.5f).done());                              // 1
    pos.push_back(PosBuilder("empty13", 13, 13, "japanese", 6.5f).done());                          // 2
    {  // 3: a corner fight on a 9x9 board, black to move: white's three corner stones are in atari (black captures at the
       // corner point), black's pair on the edge has two liberties
      PosBuilder b("fight9", 9, 9, "japanese", 6.5f);
      b.play(2, 0, P_BLACK); b.play(1, 0, P_WHITE); b.play(2, 1, P_BLACK); b.play(1, 1, P_WHITE);
      b.play(1, 2, P_BLACK); b.play(0, 1, P_WHITE); b.play(0, 2, P_BLACK); b.play(3, 0, P_WHITE);
      pos.push_back(b.done());
    }
    { PosBuilder b("rnd9a-m14", 9, 9, "chinese", 7.5f); b.randomMoves(14); pos.push_back(b.done()); }              // 4
    { PosBuilder b("rnd9b-m36", 9, 9, "tromp-taylor", 7.5f); b.randomMoves(36); pos.push_back(b.done()); }          // 5
    { PosBuilder b("rnd13a-m24", 13, 13, "japanese", 6.5f); b.randomMoves(24); pos.push_back(b.done()); }           // 6
    { PosBuilder b("rnd13b-m70", 13, 13, "chinese", 7.5f); b.randomMoves(70); pos.push_back(b.done()); }            // 7
    { PosBuilder b("rnd19a-m12", 19, 19, "chinese", 7.5f); b.randomMoves(12); pos.push_back(b.done()); }            // 8
    { PosBuilder b("rnd19b-m50", 19, 19, "japanese", 6.5f); b.randomMoves(50); pos.push_back(b.done()); }           // 9
    { PosBuilder b("rnd19c-m110", 19, 19, "tromp-taylor", 7.5f); b.randomMoves(110); pos.push_back(b.done()); }     // 10
    pos.push_back(loadRealPosition(positionsFile, 3, "game19-m20"));                                // 11
    pos.push_back(loadRealPosition(positionsFile, 4, "game19-m60"));                                // 12
    pos.push_back(loadRealPosition(positionsFile, 5, "game19-m120"));                               // 13
  }

  const vector<pair<string, string>> NOP;
  const vector<pair<string, string>> NOGRAPH = {{"useGraphSearch", "0"}};
  const vector<pair<string, string>> SYMPRUNE = {{"rootSymmetryPruning", "1"}};
  vector<CaseSpec> cases = {
    // pos base visits params
    {0, 0, 1, NOP},        {0, 0, 50, NOP},       {0, 1, 200, NOP},      {0, 1, 400, NOP},
    {1, 0, 10, NOP},       {1, 0, 100, NOP},      {1, 1, 400, NOP},      {1, 0, 100, SYMPRUNE},
    {2, 0, 50, SYMPRUNE},  {2, 1, 100, NOP},
    {3, 0, 50, NOP},       {3, 1, 200, NOP},      {3, 0, 400, NOP},
    {4, 0, 100, NOP},      {4, 1, 50, NOP},       {4, 1, 200, NOGRAPH},
    {5, 0, 10, NOP},       {5, 0, 200, NOP},      {5, 1, 100, NOP},      {5, 1, 400, NOP},
    {6, 0, 50, NOP},       {6, 1, 100, NOP},      {6, 0, 1, NOP},
    {7, 0, 100, NOP},      {7, 1, 200, NOP},      {7, 0, 400, NOGRAPH},
    {8, 0, 10, NOP},       {8, 0, 100, NOP},      {8, 1, 400, NOP},
    {9, 0, 50, NOP},       {9, 1, 200, NOP},      {9, 0, 400, NOP},
    {10, 0, 100, NOP},     {10, 1, 100, NOP},     {10, 0, 10, NOP},
    {11, 0, 200, NOP},     {12, 1, 200, NOP},     {13, 0, 100, NOP},     {13, 1, 400, NOP},
  };

  printf("# search_eigen: unmodified upstream Search (1 thread) over NNEvaluator + Eigen fp32 on %s\n", modelFile.substr(modelFile.rfind('/') + 1).c_str());
  printf("# generated with g++ %s / glibc %s on %s\n", __VERSION__, gnu_get_libc_version(), cpu.c_str());
  printf("# cases: %d (board sizes 9/13/19, nnXLen=nnYLen=19, maxBatchSize 16, nnCacheSizePowerOfTwo 16, defaultSymmetry 0, no randomize)\n", (int)cases.size());

  Logger logger(nullptr, false, false, false, false);
  logger.addOStream(cerr);
  vector<int> gpuIdxs = {-1};
  NNEvaluator* nnEval = new NNEvaluator(
    "search_eigen", modelFile, "", &logger, 16, 19, 19, false, true, 16, 12, false, "", "", false,
    enabled_t::False, enabled_t::Auto, 1, gpuIdxs, "searchEigenSeed", false, 0
  );
  nnEval->spawnServerThreads();
  for(size_t i = 0; i < cases.size(); i++) {
    runSearch((int)i, pos[cases[i].pos], cases[i], nnEval, &logger);
    fflush(stdout);
  }
  nnEval->killServerThreads();
  delete nnEval;
  return 0;
}
