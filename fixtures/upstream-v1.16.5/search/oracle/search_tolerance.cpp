// Real-net search TOLERANCE study oracle: the UNMODIFIED upstream Search (numSearchThreads = 1) over the upstream
// NNEvaluator with the Eigen fp32 CPU backend, on a real network, on many positions x SearchParams x
// maxVisits in 400..1000. Not upstream code. A larger sibling of search_eigen.cpp (the smoke oracle; same output
// format, same evaluator settings, same setParam semantics plus a few more keys): the consumer
// crates/katago-nn-burn/tests/search_tolerance.rs runs the same cases through the ported Search over the burn flex
// backend and prints statistics (identical root visit vectors, identical top move, value diffs) per net and
// maxVisits bucket. It is a measurement: it informs the search tolerance policy (docs/PORTING.md section 13).
// Floats are printed as decimal %.7g (float precision; search_eigen prints %.9g).
//
// Settings: as search_eigen.cpp (nnXLen = nnYLen = 19, maxBatchSize 16, nnCacheSizePowerOfTwo 16, defaultSymmetry 0,
// doRandomize false, 1 server thread, one evaluator for all cases), numThreads = 1, chosenMoveTemperature =
// chosenMoveTemperatureEarly = 0, maxVisits one of 400/600/800/1000.
//
// Positions (a fixed list, the same for every run): empty boards 7/9/13/19; every qualifying real-game position of
// fixtures/upstream-v1.16.5/nn/positions.txt (no setup stones, black first, no encore, game not finished; others
// are skipped); random legal positions on 7/9/13/19 boards built with a fixed-seed splitmix generator (never
// filling one's own simple eye, no passes). Cases: <numCases> specs from the same generator (seed = argv[4]): a
// position (a shuffled cycle through the list, so each position is used before any repeats), a base
// (SearchParams() or forTestsV1(), 50/50), maxVisits and a param variant (no change x3, useGraphSearch=0,
// rootSymmetryPruning=1, wideRootNoise=0.05, rootPolicyTemperature(Early)=1.4, conservativePass=1, and two
// combinations).
//
// Build (from the repo root; needs build/eigen from `just katago-eigen`; links the Release objects of the Eigen
// build of the unmodified upstream sources, library dirs only; the binary goes outside the repo):
//   B=build/eigen; U=upstream/KataGo/cpp; O=$B/CMakeFiles/katago.dir
//   g++ -std=gnu++17 -O2 -mfpmath=sse -DNO_GIT_REVISION -DUSE_EIGEN_BACKEND -I$U -I$U/external -I$B \
//     -isystem $U/external/filesystem-1.5.8/include -isystem build/deps/eigen \
//     -o /tmp/search_tolerance fixtures/upstream-v1.16.5/search/oracle/search_tolerance.cpp \
//     $(find $O/core $O/game $O/neuralnet $O/search $O/dataio $O/program -name '*.o') -lz -lzip -lpthread
// Generate (<model> <positions> <numCases> <caseSeed>; the header lines of the output record the compiler, glibc and
// CPU of the run):
//   /tmp/search_tolerance upstream/KataGo/cpp/tests/models/g170-b6c96-s175395328-d26788732.bin.gz \
//     fixtures/upstream-v1.16.5/nn/positions.txt 160 1 > fixtures/upstream-v1.16.5/search/oracle/search_tolerance_b6c96.txt
//   /tmp/search_tolerance upstream/KataGo/cpp/tests/models/g170e-b10c128-s1141046784-d204142634.bin.gz \
//     fixtures/upstream-v1.16.5/nn/positions.txt 32 2 > fixtures/upstream-v1.16.5/search/oracle/search_tolerance_b10c128.txt
// Generated with g++ (GCC) 16.2.1 / glibc 2.44 on x86-64 (Intel i7-8700K, FMA), Eigen 3.4.0; the two runs in parallel took
// 18 min 25 s (b6c96, 160 searches) and 13 min 17 s (b10c128, 32 searches), ~7 s and ~25 s per search single-threaded.
//
// Output: as search_eigen.cpp. Lines starting with '#' are comments. One block per search:
//   CASE <n> name=<label> board=<x>x<y> ko=<0 simple|1 positional|2 situational> score=<0 area|1 territory>
//     tax=<0 none|1 seki|2 all> suicide=<0|1> button=<0|1> whb=<0..2> fpo=<friendlyPassOk 0|1> komi=<f32 bits, 8 hex>
//     ipla=<initial player 1 B|2 W> pla=<root player> moves=<loc/pla,...|-> (raw Loc ints: Board(x,y),
//     BoardHistory(board,ipla,rules,0), makeBoardMoveAssumeLegal(board,loc,pla,NULL) for each) sseed=<Search rand
//     seed> base=<0 SearchParams()|1 forTestsV1> maxvisits=<n> params=<key=value,...|->
//   root visits=<n>
//   c <i> <move> loc=<loc> edge=<edge visits> visits=<child node visits> wl=<winLossValue> sm=<scoreMean>
//     lead=<lead> util=<utility> prior=<policy prior>      one per allocated root child, children-array order
//   children=<n>
//   RV win=<winValue> loss=<lossValue> es=<expectedScore> lead=<lead> wl=<winLossValue> util=<utility>   (getRootValues)
//   chosen loc=<loc> <move>                                                       (getChosenMoveLoc)
//   END <n>
// Values are white's perspective (AnalysisData / ReportedSearchValues convention). Params keys: the SearchParams
// field names handled by setParam below.
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
  if(k == "wideRootNoise") { p.wideRootNoise = d; return true; }
  if(k == "rootPolicyTemperature") { p.rootPolicyTemperature = d; return true; }
  if(k == "rootPolicyTemperatureEarly") { p.rootPolicyTemperatureEarly = d; return true; }
  if(k == "conservativePass") { p.conservativePass = b; return true; }
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

// The real-game positions of fixtures/upstream-v1.16.5/nn/positions.txt (ids): format: see nn/oracle/nneval_eigen.cpp.
// Returns false for the ones this oracle cannot replay as "black first, no setup stones, no encore, unfinished".
static bool loadRealPosition(const vector<string>& lines, int id, const string& name, Position& p) {
  istringstream ls(lines[id]);
  string tag, komiHex;
  int pid, xs, ys, ko, sc, tax, suic, button, whb, fpok, enc, ipla, amh;
  ls >> tag >> pid >> xs >> ys >> ko >> sc >> tax >> suic >> button >> whb >> fpok >> komiHex >> enc >> ipla >> amh;
  testAssert(tag == "pos" && pid == id);
  if(!(enc == 0 && ipla == P_BLACK && amh == 0))
    return false;
  size_t nSetup; ls >> nSetup;
  if(nSetup != 0)
    return false;
  size_t nMoves; ls >> nMoves;
  uint32_t kb = (uint32_t)stoul(komiHex, nullptr, 16);
  float komi; memcpy(&komi, &kb, 4);
  p = Position();
  p.name = name; p.xs = xs; p.ys = ys;
  p.rules = Rules(ko, sc, tax, suic != 0, button != 0, whb, fpok != 0, komi);
  Board board(xs, ys);
  BoardHistory hist(board, P_BLACK, p.rules, 0);
  for(size_t i = 0; i < nMoves; i++) {
    int x, y, pla, pe;
    ls >> x >> y >> pla >> pe;
    if(ls.fail() || pe != 0)
      return false;
    Loc loc = x == -1 ? Board::PASS_LOC : Location::getLoc(x, y, xs);
    if(!hist.isLegal(board, loc, (Player)pla))
      return false;
    hist.makeBoardMoveAssumeLegal(board, loc, (Player)pla, NULL);
    p.moves.push_back(make_pair(loc, (Player)pla));
  }
  if(hist.isGameFinished)
    return false;
  return true;
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
  string sseed = "tol" + to_string(idx);

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
    printf("c %d %s loc=%d edge=%lld visits=%lld wl=%.7g sm=%.7g lead=%.7g util=%.7g prior=%.7g\n", i,
           Location::toString(mv, search->getRootBoard()).c_str(), (int)mv, (long long)children[i].getEdgeVisits(),
           (long long)child->stats.visits.load(), d.winLossValue, d.scoreMean, d.lead, d.utility, d.policyPrior);
    n++;
  }
  testAssert(n == (int)ad.size());
  printf("children=%d\n", n);
  ReportedSearchValues v;
  bool suc = search->getRootValues(v);
  testAssert(suc);
  printf("RV win=%.7g loss=%.7g es=%.7g lead=%.7g wl=%.7g util=%.7g\n", v.winValue, v.lossValue, v.expectedScore, v.lead, v.winLossValue, v.utility);
  Loc chosen = search->getChosenMoveLoc();
  printf("chosen loc=%d %s\n", (int)chosen, Location::toString(chosen, search->getRootBoard()).c_str());
  printf("END %d\n", idx);
  delete search;
}

int main(int argc, char** argv) {
  if(argc != 5) {
    fprintf(stderr, "usage: %s <model.bin.gz> <nn/positions.txt> <numCases> <caseSeed>\n", argv[0]);
    return 1;
  }
  string modelFile = argv[1], positionsFile = argv[2];
  int numCases = atoi(argv[3]);
  uint64_t caseSeed = strtoull(argv[4], NULL, 10);
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
    pos.push_back(PosBuilder("empty19", 19, 19, "tromp-taylor", 7.5f).done());
    pos.push_back(PosBuilder("empty13", 13, 13, "japanese", 6.5f).done());
    pos.push_back(PosBuilder("empty9", 9, 9, "chinese", 7.5f).done());
    pos.push_back(PosBuilder("empty7", 7, 7, "aga", 7.0f).done());
    // Real games.
    {
      ifstream in(positionsFile);
      testAssert(in.good());
      vector<string> lines;
      string line;
      while(getline(in, line))
        if(!line.empty())
          lines.push_back(line);
      for(int id = 0; id < (int)lines.size(); id++) {
        Position p;
        if(loadRealPosition(lines, id, "", p)) {
          p.name = "game" + to_string(p.xs) + "-p" + to_string(id) + "-m" + to_string(p.moves.size());
          pos.push_back(p);
        }
      }
    }
    // Random legal positions: (size, rules, komi, number of moves)
    struct R { int size; const char* rules; float komi; int moves; };
    const R randoms[] = {
      {7, "chinese", 7.5f, 8},        {7, "japanese", 6.5f, 20},      {7, "tromp-taylor", 7.5f, 12},
      {9, "chinese", 7.5f, 14},       {9, "tromp-taylor", 7.5f, 36},  {9, "japanese", 6.5f, 24},
      {9, "aga", 7.5f, 44},           {9, "chinese-ogs", 7.5f, 6},
      {13, "japanese", 6.5f, 24},     {13, "chinese", 7.5f, 70},      {13, "tromp-taylor", 7.5f, 40},
      {13, "aga", 7.5f, 100},         {13, "stone-scoring", 7.5f, 55},
      {19, "chinese", 7.5f, 12},      {19, "japanese", 6.5f, 50},     {19, "tromp-taylor", 7.5f, 110},
      {19, "aga", 7.5f, 30},          {19, "chinese-ogs", 7.5f, 80},  {19, "japanese", 0.5f, 160},
      {19, "stone-scoring", 7.5f, 200},
    };
    int k = 0;
    for(const R& r : randoms) {
      string name = "rnd" + to_string(r.size) + "-" + to_string(k++) + "-m" + to_string(r.moves);
      PosBuilder b(name, r.size, r.size, r.rules, r.komi);
      b.randomMoves(r.moves);
      pos.push_back(b.done());
    }
  }

  const vector<pair<string, string>> NOP;
  const vector<vector<pair<string, string>>> variants = {
    NOP, NOP, NOP,
    {{"useGraphSearch", "0"}},
    {{"rootSymmetryPruning", "1"}},
    {{"wideRootNoise", "0.05"}},
    {{"rootPolicyTemperature", "1.4"}, {"rootPolicyTemperatureEarly", "1.4"}},
    {{"conservativePass", "1"}},
    {{"useGraphSearch", "0"}, {"rootSymmetryPruning", "1"}},
    {{"wideRootNoise", "0.05"}, {"conservativePass", "1"}},
  };
  const int64_t visitChoices[] = {400, 600, 800, 1000};

  // Case specs: a fresh generator seeded by caseSeed (the position generator above has run already).
  S = 0xC0FFEE1234567ULL ^ (caseSeed * 0x9E3779B97F4A7C15ULL);
  vector<CaseSpec> cases;
  vector<int> order;
  for(int i = 0; i < numCases; i++) {
    if(order.empty()) {
      for(int j = 0; j < (int)pos.size(); j++)
        order.push_back(j);
      for(int j = (int)order.size() - 1; j > 0; j--)
        swap(order[j], order[rnd(j + 1)]);
    }
    CaseSpec c;
    c.pos = order.back();
    order.pop_back();
    c.base = rnd(2);
    c.maxVisits = visitChoices[rnd(4)];
    c.params = variants[rnd((int)variants.size())];
    cases.push_back(c);
  }

  printf("# search_tolerance: unmodified upstream Search (1 thread) over NNEvaluator + Eigen fp32 on %s\n", modelFile.substr(modelFile.rfind('/') + 1).c_str());
  printf("# generated with g++ %s / glibc %s on %s\n", __VERSION__, gnu_get_libc_version(), cpu.c_str());
  printf("# cases: %d from %d positions, seed %llu (board sizes 7/9/13/19, nnXLen=nnYLen=19, maxBatchSize 16, nnCacheSizePowerOfTwo 16, defaultSymmetry 0, no randomize)\n", (int)cases.size(), (int)pos.size(), (unsigned long long)caseSeed);

  Logger logger(nullptr, false, false, false, false);
  logger.addOStream(cerr);
  vector<int> gpuIdxs = {-1};
  NNEvaluator* nnEval = new NNEvaluator(
    "search_tolerance", modelFile, "", &logger, 16, 19, 19, false, true, 16, 12, false, "", "", false,
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
