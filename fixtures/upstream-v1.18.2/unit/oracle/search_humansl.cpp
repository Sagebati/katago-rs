// Oracle for the human-SL paths of crates/katago-core/src/search/*.rs (M8): deterministic searches by the
// unmodified upstream Search (numSearchThreads = 1) with a main NNEvaluator AND a human-SL evaluator
// (Search(params, nnEval, humanEval, logger, seed)), both with debugSkipNeuralNet = true (their server threads
// draw Gaussian "logits" from their own Rand; built like search_nonn.cpp's: nnCache 2^16, maxBatchSize 16,
// defaultSymmetry 0). A debugSkipNeuralNet evaluator has no meta channels, so requiresSGFMetadata() is false and
// every human-SL path of the search is live whatever humanSLProfile is: the human output at the root and in the
// tree (needsHumanOutputAtRoot / InTree, initNodeNNOutput, maybeRecomputeExistingNNOutput after tree reuse), the
// human-SL exploration (humanSLRoot/Pla/OppExploreProbWeightless/Weightful, humanSLCpuctExploration/Permanent:
// searchexplorehelpers.cpp), the human chosen-move mix (humanSLChosenMoveProp, humanSLChosenMoveIgnorePass,
// humanSLChosenMovePiklLambda: getPlaySelectionValues), and getAnalysisJson's humanPrior / humanPolicy /
// rootInfo human* keys. Floats as IEEE-754 bit patterns or as upstream's printers print them.
// Not upstream code. Kept apart from search_nonn.cpp (whose sections other work appends to). The Rust mirror is
// crates/katago-core/tests/search_humansl.rs: it rebuilds every case from its CASE line, runs the same fixed op
// sequence through the ported Search and must print the same bytes.
//
// Build (from the repo root, after `just katago-eigen`; links the Eigen build's objects of the unmodified upstream
// sources (library dirs only); the binary goes outside the repo):
//   O=build/eigen/CMakeFiles/katago.dir; g++ -std=c++17 -O2 -Iupstream/KataGo/cpp -o /tmp/search_humansl \
//     fixtures/upstream-v1.18.2/unit/oracle/search_humansl.cpp \
//     $(find $O/core $O/game $O/neuralnet $O/search $O/dataio $O/program -name '*.o') -lz -lzip -lpthread
//   /tmp/search_humansl > fixtures/upstream-v1.18.2/unit/oracle/search_humansl.txt
// Generated with g++ (GCC) 16.2.1 / glibc 2.44 on x86_64 (FMA) Linux.
//
// Output. One block per case: the CASE line, the lines the ops print, "END <n>". The CASE line holds every input:
//   CASE <n> board=<x>x<y> rules=<name> komi=<f32 bits> moves=<loc/pla,...|-> (raw Loc ints; Board(x,y),
//     BoardHistory(board,P_BLACK,rules,0), makeBoardMoveAssumeLegal(board,loc,pla,NULL)) pla=<root player>
//     human=<0|1: a human evaluator> sseed=<Search seed> base=<0 SearchParams()|2 forTestsV2> maxvisits=<n>
//     params=<key=value,...|-> (applied in order after the base; numThreads 1; humanSLProfile=<name> goes through
//     SGFMetadata::getProfile)
// The ops (fixed): setPosition(pla); runWholeSearch(pla) ("playouts=<n>"); printTree(depth 1, P_WHITE); PSV
// (getPlaySelectionValues with visit counts, as search_nonn's PSV); "cm" getChosenMoveLoc() x3; J (getAnalysisJson(
// P_WHITE, 3, preventEncore, policy, pvVisits): success and dump(), for later cases only "jfnv <FNV-1a 64> <length>" of
// the dump); then makeMove(the first chosen move) and the
// same again for the other player (tree reuse), each op line prefixed by "> <name>". The search's Logger writes
// ": <message>" lines into the same buffer.
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <algorithm>
#include <sstream>
#include <string>
#include <vector>
#include "core/global.h"
#include "core/logger.h"
#include "core/rand.h"
#include "game/board.h"
#include "game/rules.h"
#include "game/boardhistory.h"
#include "neuralnet/nneval.h"
#include "neuralnet/sgfmetadata.h"
#include "search/search.h"
#include "search/searchnode.h"
#include "search/searchparams.h"
#include "search/searchprint.h"
using namespace std;

static uint64_t dbits(double d) { uint64_t u; memcpy(&u, &d, 8); return u; }
static uint32_t fbits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
#define DB(x) ((unsigned long long)dbits(x))

static ostringstream O;

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
  O << s;
}

static bool setParam(SearchParams& p, const string& k, const string& v) {
  const char* s = v.c_str();
  double d = strtod(s, NULL);
  bool b = v == "1";
#define D(name) if(k == #name) { p.name = d; return true; }
#define Bo(name) if(k == #name) { p.name = b; return true; }
  D(humanSLCpuctExploration) D(humanSLCpuctPermanent)
  D(humanSLRootExploreProbWeightless) D(humanSLRootExploreProbWeightful)
  D(humanSLPlaExploreProbWeightless) D(humanSLPlaExploreProbWeightful)
  D(humanSLOppExploreProbWeightless) D(humanSLOppExploreProbWeightful)
  D(humanSLChosenMoveProp) Bo(humanSLChosenMoveIgnorePass) D(humanSLChosenMovePiklLambda)
  D(chosenMoveTemperature) D(chosenMoveTemperatureEarly) D(rootPolicyTemperature) D(wideRootNoise) D(cpuctExploration)
  Bo(useNoisePruning) Bo(enablePassingHacks) Bo(useLcbForSelection) Bo(rootPruneUselessMoves) Bo(useGraphSearch)
#undef D
#undef Bo
  if(k == "humanSLProfile") { p.humanSLProfile = SGFMetadata::getProfile(v); return true; }
  return false;
}

static Rand gen("search_humansl_oracle_20261024");
static int rnd(int n) { return (int)gen.nextUInt((uint32_t)n); }
static bool prob(double p) { return gen.nextBool(p); }
template<typename T, size_t N> static const T& pick(const T (&a)[N]) { return a[rnd((int)N)]; }

static void psv(Search* search) {
  vector<Loc> locs;
  vector<double> vals, visits;
  bool suc = search->getPlaySelectionValues(locs, vals, &visits, 1.0);
  pf("success=%d n=%d nv=%d\n", (int)suc, (int)locs.size(), (int)visits.size());
  if(suc) {
    for(size_t i = 0; i < locs.size(); i++) {
      pf("s %d %s %016llX", (int)locs[i], Location::toString(locs[i], search->getRootBoard()).c_str(), DB(vals[i]));
      if(i < visits.size()) pf(" %016llX\n", DB(visits[i]));
      else pf(" -\n");
    }
  }
}

static uint64_t fnv64(const string& s) {
  uint64_t h = 0xcbf29ce484222325ULL;
  for(unsigned char c : s) { h ^= c; h *= 0x100000001b3ULL; }
  return h;
}

// The depth of the tree after a search. The Rust test also runs on wasm32 with a 1 MB stack, where deep trees overflow
// the recursive playout of a debug build: weightless human-SL exploration (humanSL*ExploreProbWeightless near 1) follows
// the human policy for hundreds of plies. Cases whose trees get deeper than MAX_TREE_DEPTH are redrawn (with
// ORACLE_DEPTH=1 every search's depth is printed to stderr).
static const int MAX_TREE_DEPTH = 100;
static int maxDepthThisCase = 0;
static int treeDepth(const SearchNode* n, int depthSoFar) {
  if(depthSoFar > 2000) return depthSoFar; // graph search cycles
  int best = 0;
  ConstSearchNodeChildrenReference children = n->getChildren();
  int cap = children.getCapacity();
  for(int i = 0; i < cap; i++) {
    const SearchNode* c = children[i].getIfAllocated();
    if(c == NULL) break;
    best = std::max(best, 1 + treeDepth(c, depthSoFar + 1));
  }
  return best;
}

static Loc searchAndReport(Search* search, Player pla, bool fullJson) {
  pf("> R %d\n", (int)pla);
  search->runWholeSearch(pla);
  int depth = treeDepth(search->rootNode, 0);
  maxDepthThisCase = std::max(maxDepthThisCase, depth);
  if(getenv("ORACLE_DEPTH")) fprintf(stderr, "depth %d\n", depth);
  pf("playouts=%lld\n", (long long)search->lastSearchNumPlayouts);
  pf("> PT\n");
  search->printTree(O, search->rootNode, PrintTreeOptions().maxDepth(1), P_WHITE);
  pf("> PSV\n");
  psv(search);
  pf("> CM\n");
  Loc first = Board::NULL_LOC;
  for(int i = 0; i < 3; i++) {
    Loc l = search->getChosenMoveLoc();
    if(i == 0) first = l;
    pf("cm %d %s\n", (int)l, Location::toString(l, search->getRootBoard()).c_str());
  }
  pf("> J\n");
  nlohmann::json json;
  bool suc = search->getAnalysisJson(P_WHITE, 3, true, true, false, false, false, false, true, false, json);
  pf("success=%d\n", (int)suc);
  string dumped = json.dump();
  if(fullJson)
    O << dumped << "\n";
  else
    pf("jfnv %016llX %d\n", (unsigned long long)fnv64(dumped), (int)dumped.size());
  return first;
}

int main() {
  Board::initHash();
  ScoreValue::initTables();
  static const int sizes[][2] = {{7, 7}, {9, 9}, {9, 7}, {11, 11}, {13, 13}, {19, 19}};
  static const char* ruleNames[] = {"tromp-taylor", "chinese", "japanese", "korean", "aga"};
  static const float komis[] = {7.5f, 6.5f, 0.5f, -3.5f, 5.0f};
  static const char* profiles[] = {"", "rank_5k", "preaz_20k", "proyear_1900", "rank_9d_1k"};
  static const char* probs[] = {"0", "0", "0.3", "0.8", "1"};
  static const char* cpucts[] = {"0.5", "1", "2"};
  static const char* perms[] = {"0", "0.2", "1"};
  static const char* chosenProps[] = {"0", "0.5", "1", "0.25"};
  static const char* lambdas[] = {"100000000", "0.1", "0.5", "0.02"};
  static const int64_t visitsList[] = {12, 30, 60, 120, 250};
  const int numCases = 120;
  for(int idx = 0; idx < numCases; idx++) {
    const int* sz = sizes[rnd(6)];
    int xs = sz[0], ys = sz[1];
    string ruleName = pick(ruleNames);
    Rules rules = Rules::parseRules(ruleName);
    rules.komi = pick(komis);
    Board board(xs, ys);
    BoardHistory hist(board, P_BLACK, rules, 0);
    vector<pair<Loc, Player>> moves;
    int numMoves = rnd(xs * ys * 2 / 3 + 1);
    for(int m = 0; m < numMoves && !hist.isGameFinished; m++) {
      Player pla = hist.presumedNextMovePla;
      Loc loc = Board::PASS_LOC;
      if(rnd(100) >= 4) {
        for(int tries = 0; tries < 40; tries++) {
          Loc l = Location::getLoc(rnd(xs), rnd(ys), xs);
          if(hist.isLegal(board, l, pla)) { loc = l; break; }
        }
      }
      if(!hist.isLegal(board, loc, pla)) break;
      hist.makeBoardMoveAssumeLegal(board, loc, pla, NULL);
      moves.push_back(make_pair(loc, pla));
    }
    if(hist.isGameFinished) {
      // Keep the root playable: drop the moves that ended the game.
      idx--;
      continue;
    }
    Player pla = hist.presumedNextMovePla;
    bool human = idx % 8 != 7;
    int base = prob(0.7) ? 2 : 0;
    int64_t maxVisits = pick(visitsList);
    vector<pair<string, string>> params;
    auto add = [&](const string& k, const string& v) { params.push_back(make_pair(k, v)); };
    if(prob(0.6)) add("humanSLProfile", pick(profiles));
    if(prob(0.5)) add("humanSLCpuctExploration", pick(cpucts));
    if(prob(0.5)) add("humanSLCpuctPermanent", pick(perms));
    // Root / player / opponent explore probabilities: often all equal (search.cpp only warns when they differ).
    if(prob(0.5)) {
      string wl = pick(probs), wf = pick(probs);
      add("humanSLRootExploreProbWeightless", wl); add("humanSLRootExploreProbWeightful", wf);
      add("humanSLPlaExploreProbWeightless", wl); add("humanSLPlaExploreProbWeightful", wf);
      add("humanSLOppExploreProbWeightless", wl); add("humanSLOppExploreProbWeightful", wf);
    }
    else {
      if(prob(0.6)) add("humanSLRootExploreProbWeightless", pick(probs));
      if(prob(0.6)) add("humanSLRootExploreProbWeightful", pick(probs));
      if(prob(0.4)) add("humanSLPlaExploreProbWeightless", pick(probs));
      if(prob(0.4)) add("humanSLPlaExploreProbWeightful", pick(probs));
      if(prob(0.4)) add("humanSLOppExploreProbWeightless", pick(probs));
      if(prob(0.4)) add("humanSLOppExploreProbWeightful", pick(probs));
    }
    if(prob(0.7)) add("humanSLChosenMoveProp", pick(chosenProps));
    if(prob(0.4)) add("humanSLChosenMoveIgnorePass", prob(0.5) ? "1" : "0");
    if(prob(0.5)) add("humanSLChosenMovePiklLambda", pick(lambdas));
    if(prob(0.4)) add("chosenMoveTemperature", prob(0.5) ? "0.5" : "1");
    if(prob(0.2)) add("chosenMoveTemperatureEarly", "0.8");
    if(prob(0.2)) add("rootPolicyTemperature", "1.3");
    if(prob(0.2)) add("wideRootNoise", "0.04");
    if(prob(0.2)) add("useNoisePruning", "1");
    if(prob(0.2)) add("enablePassingHacks", "1");
    if(prob(0.2)) add("useLcbForSelection", prob(0.5) ? "1" : "0");
    if(prob(0.15)) add("useGraphSearch", "0");
    string sseed = "humansl" + to_string(idx);

    O.str("");
    O.clear();
    Logger logger(nullptr, false, false, false, false);
    logger.addOStream(O);
    NNEvaluator* nnEval = new NNEvaluator(
      "search_humansl", "search_humansl", "", NULL, 16, 19, 19, false, true, 16, 12, true, "", "", false,
      enabled_t::False, enabled_t::False, 1, {0}, "runSearchTestsRandSeed" + to_string(idx), false, 0
    );
    nnEval->spawnServerThreads();
    NNEvaluator* humanEval = NULL;
    if(human) {
      humanEval = new NNEvaluator(
        "search_humansl_human", "search_humansl_human", "", NULL, 16, 19, 19, false, true, 16, 12, true, "", "", false,
        enabled_t::False, enabled_t::False, 1, {0}, "humanSLRandSeed" + to_string(idx), false, 0
      );
      humanEval->spawnServerThreads();
    }
    SearchParams sp = base == 2 ? SearchParams::forTestsV2() : SearchParams();
    sp.numThreads = 1;
    for(const auto& kv : params)
      if(!setParam(sp, kv.first, kv.second)) { fprintf(stderr, "bad param %s\n", kv.first.c_str()); return 1; }
    sp.maxVisits = maxVisits;

    string movesStr;
    for(size_t i = 0; i < moves.size(); i++)
      movesStr += (i ? "," : "") + to_string((int)moves[i].first) + "/" + to_string((int)moves[i].second);
    if(movesStr.empty()) movesStr = "-";
    string paramsStr;
    for(size_t i = 0; i < params.size(); i++)
      paramsStr += (i ? "," : "") + params[i].first + "=" + params[i].second;
    if(paramsStr.empty()) paramsStr = "-";
    string caseLine = Global::strprintf(
      "CASE %d board=%dx%d rules=%s komi=%08X moves=%s pla=%d human=%d sseed=%s base=%d maxvisits=%lld params=%s",
      idx, xs, ys, ruleName.c_str(), fbits(rules.komi), movesStr.c_str(), (int)pla, (int)human, sseed.c_str(), base,
      (long long)maxVisits, paramsStr.c_str()
    );

    maxDepthThisCase = 0;
    Search* search = new Search(sp, nnEval, humanEval, &logger, sseed);
    pf("> P\n");
    search->setPosition(pla, board, hist);
    Loc mv = searchAndReport(search, pla, idx < 24);
    if(mv != Board::NULL_LOC) {
      pf("> M %d %d\n", (int)mv, (int)pla);
      bool suc = search->makeMove(mv, pla);
      pf("suc=%d\n", (int)suc);
      if(suc && !search->getRootHist().isGameFinished)
        searchAndReport(search, getOpp(pla), idx < 12);
    }
    delete search;
    nnEval->killServerThreads();
    delete nnEval;
    if(humanEval != NULL) {
      humanEval->killServerThreads();
      delete humanEval;
    }
    if(maxDepthThisCase > MAX_TREE_DEPTH) {
      if(getenv("ORACLE_DEPTH")) fprintf(stderr, "case %d redrawn (depth %d)\n", idx, maxDepthThisCase);
      idx--;
      continue;
    }
    printf("%s\n%sEND %d\n", caseLine.c_str(), O.str().c_str(), idx);
  }
  return 0;
}
