// Oracle for crates/katago-core/src/search/*.rs: many deterministic searches run by the unmodified
// upstream Search (numSearchThreads = 1) on an NNEvaluator with debugSkipNeuralNet = true (its server
// thread draws Gaussian "logits" from its Rand; built as TestSearchCommon::startNNEval does: nnCache
// 2^16, maxBatchSize 16, defaultSymmetry 0, no randomize), so every result of the search (tree shape,
// visit counts, stats, selection values, analysis data/JSON, printed trees, chosen moves, ...) is checked
// bit for bit against the port without a net. Float results are printed as IEEE-754 bit patterns.
// Not upstream code. The Rust mirror is crates/katago-core/tests/search_oracle.rs: it parses every
// "CASE" line below (all inputs of the case are on that line), rebuilds the case, executes the op stream
// through the ported Search on a futures LocalPool and must print the same bytes.
//
// Build (from the repo root, after `just katago-eigen`; links the Eigen build's objects of the
// unmodified upstream sources (library dirs only); the binary goes outside the repo):
//   O=build/eigen/CMakeFiles/katago.dir; g++ -std=c++17 -O2 -Iupstream/KataGo/cpp -o /tmp/search_nonn \
//     fixtures/upstream-v1.16.5/unit/oracle/search_nonn.cpp \
//     $(find $O/core $O/game $O/neuralnet $O/search $O/dataio $O/program -name '*.o') -lz -lzip -lpthread
//   /tmp/search_nonn > fixtures/upstream-v1.16.5/unit/oracle/search_nonn.txt
// Generated with g++ (GCC) 16.2.1 / glibc 2.44 on x86_64 (FMA) Linux.
//
// Output. One block per case: a "CASE" line followed by the lines the ops print, ended by "END <n>".
// The CASE line holds every input (space separated key=value tokens, no spaces inside values):
//   CASE <n> kind=<generator name> nn=<nnXLen>x<nnYLen> nnseed=<str> board=<x>x<y> ko=<0 simple|1 positional|
//     2 situational> score=<0 area|1 territory> tax=<0 none|1 seki|2 all> suicide=<0|1> button=<0|1> whb=<0..2>
//     fpo=<friendlyPassOk 0|1> komi=<f32 bits, 8 hex digits> ipla=<initial player 1 B|2 W> pla=<root player given
//     to setPosition> moves=<loc/pla,...|-> (raw Loc ints; Board(x,y), BoardHistory(board,ipla,rules,0) and
//     makeBoardMoveAssumeLegal(board,loc,pla,NULL) for each) sseed=<Search rand seed> base=<0 SearchParams()|
//     1 forTestsV1|2 forTestsV2> maxvisits=<n> params=<key=value,...|-> ops=<op;op;...>
//   The initial SearchParams are base + params (applied in order) + maxVisits; numThreads is 1. Values are
//   parsed with strtod/strtol (the Rust side uses the same decimal text), bools are 0/1.
// Ops (':' separates the arguments; <pla> 1 B, 2 W; <persp> W, B or N (= P_EMPTY, the root player's)):
//   P                          setPosition(pla, board, hist) from the CASE line; prints the board, encore phase, ...
//   V:<bspec>:<wspec>          setAvoidMoveUntilByLoc; spec = "-" (empty vector) or "loc/until,..." on a
//                              MAX_ARR_SIZE vector of zeros
//   AR:<0|1>  H:<loc>  O  Y:<sym,...|->      setAvoidMoveUntilRescaleRoot, setRootHintLoc,
//                              setAlwaysIncludeOwnerMap(true), setRootSymmetryPruningOnly
//   R:<pla>   B                runWholeSearch(pla) (prints lastSearchNumPlayouts) / beginSearch(false) only
//   RS                         root summary: root visits, every root child in children-array order, getRootValues,
//                              getPrunedRootValues, getRootRawNNValues, policy surprise/entropy, sharp score,
//                              shallow short-term errors, node table size, enumerateTreePostOrder().size(), and
//                              "ntkeys": count + FNV-1a 64 of the node table's keys in table order (shard by shard, each
//                              shard's map in Hash128 order; "<shard> <hash1> <hash0>" lines), also at the end of HS
//   PT:<depth>:<persp>:<flags> printTree (flags bits: 1 printSqs, 2 printAvgShorttermError, 4 minVisitsPropToExpand(0.1),
//                              8 maxChildrenToShow(5))
//   PV:<depth>                 printPV(root)
//   HS:<depth>:<persp>:<8 bits>  compact summary for stress cases: root visits, getRootValues/getPrunedRootValues in full, and
//                              FNV-1a 64 + length of: all root child lines, printTree(depth), getAnalysisData(3,true,10,true)
//                              (all entries as AD prints them), getAnalysisJson (bits as in J) and getPlaySelectionValues
//   AD:<min>:<w>:<dup>         getAnalysisData(buf, min, includeWeightFactors, 10, duplicateForSymmetries)
//   J:<persp>:<pvLen>:<8 bits> getAnalysisJson (bits: preventEncore, policy, ownership, ownershipStdev, movesOwnership,
//                              movesOwnershipStdev, pvVisits, noResultValue); prints success and .dump()
//   CM:<n>                     getChosenMoveLoc() n times (nonSearchRand)
//   PSV                        getPlaySelectionValues (with visit counts)
//   OW                         getAverageTreeOwnership / getAverageAndStandardDeviationTreeOwnership of the root
//   PM                         printRootPolicyMap, printRootOwnershipMap(W), printRootEndingScoreValueBonus
//   M:<loc>:<pla>:<prevent>    makeMove(loc, pla, preventEncore)
//   SPN:<k=v,...>  SP:<k=v,...>  setParamsNoClearing / setParams with the current params + overrides
//   CL  KOMI:<f32 bits>  PIN:<pla>  PCH:<pla>   clearSearch, setKomiIfNew, setPlayerIfNew, setPlayerAndClearHistory
//   XO                         the root's nnOutput replaced by a copy whose whiteOwnerMap is scaled by 6 and clamped to
//                              [-0.999, 0.999] (float math); prints the number of entries beyond +-0.95 ("noxo" if none)
//   ECT:<f64 bits|->           sets effectiveSearchTimeCarriedOver (unless "-") and prints it. After a search it holds
//                              real elapsed time upstream, so the generators set it before every print they compare
// A thrown exception prints "EXC" and ends the case (the message goes to stderr only). Inputs that trip an upstream
// testAssert (a fatal exit, not an exception) are avoided by the generators: a root player other than the history's
// presumed next one (nneval.cpp), getChosenMoveLoc after beginSearch alone, chosenMoveTemperatureOnlyBelowProb.
// Cases (341 + 41), by the generator named in "kind": random (85: random legal games, 5..13 odd/even/non-square, every named
// ruleset plus random ko/suicide/whb/friendlyPassOk, out-of-turn moves), endgame (28: games ended by pass pairs: encore
// phases 1/2 and finished games), ko (27: hand-built kos, plain and in encore phases), passy (6: rootPruneUselessMoves
// applies), sym (15: empty boards / center stone with root symmetry pruning), big/big19 (10: 11..13 and 19 wide boards
// with captures), lowvisits (18: 1 and 2 visits, beginSearch only), avoid (18: avoided root moves, up to all moves
// and pass), stress (130: compact HS reports, up to 3 follow-up searches: makeMove/tree reuse, setParams*, clearSearch,
// setPlayerAndClearHistory, setKomiIfNew), wide (4: 900..1500 visits, nodes with more than 64 children).
// Appended later (M3 wave-2a fixes; cases 0..340 kept their inputs and outputs, plus the ntkeys lines): widereuse (3:
// makeMove into a child with more than 64 children), ownext (24: XO, ending score bonus / pass suppression / noise
// pruning cap), collapse (6: root children filtered by rootSymmetryPruning in a reused tree, then regrown), timecarry
// (8: ECT around makeMove with treeReuseCarryOverTimeFactor). 382 cases in all.
// For exploration only (not the fixture): ORACLE_SEED=<str> ORACLE_STRESS_ONLY=<n> prints n stress cases for another
// seed; the Rust test replays such a file when SEARCH_ORACLE_FILE names it; ORACLE_TRACE=1 echoes the ops to stderr.
// The search's Logger writes into the same buffer (": <message>" lines, as testsearchnonn's does).
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <algorithm>
#include <cstdlib>
#include <functional>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include "core/global.h"
#include "core/hash.h"
#include "core/logger.h"
#include "core/rand.h"
#include "game/board.h"
#include "game/rules.h"
#include "game/boardhistory.h"
#include "neuralnet/nninputs.h"
#include "neuralnet/nneval.h"
#include "search/analysisdata.h"
#include "search/reportedsearchvalues.h"
#include "search/search.h"
#include "search/searchnode.h"
#include "search/searchnodetable.h"
#include "search/searchparams.h"
#include "search/searchprint.h"
using namespace std;

static uint64_t dbits(double d) { uint64_t u; memcpy(&u, &d, 8); return u; }
static uint32_t fbits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static float fromBits(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
#define DB(x) ((unsigned long long)dbits(x))

//-------------------------------------------------------------------------------------------------
// Output buffer of the current case (the Logger writes into it too)

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

static vector<string> splitOn(const string& s, char d) {
  vector<string> r;
  string cur;
  for(char ch : s) {
    if(ch == d) { r.push_back(cur); cur.clear(); }
    else cur.push_back(ch);
  }
  r.push_back(cur);
  return r;
}

//-------------------------------------------------------------------------------------------------
// SearchParams by key=value text

static bool setParam(SearchParams& p, const string& k, const string& v) {
  const char* s = v.c_str();
  double d = strtod(s, NULL);
  long long l = strtoll(s, NULL, 10);
  bool b = v == "1";
#define D(name) if(k == #name) { p.name = d; return true; }
#define I(name) if(k == #name) { p.name = (int)l; return true; }
#define L(name) if(k == #name) { p.name = (int64_t)l; return true; }
#define Bo(name) if(k == #name) { p.name = b; return true; }
  D(winLossUtilityFactor) D(staticScoreUtilityFactor) D(dynamicScoreUtilityFactor) D(dynamicScoreCenterZeroWeight)
  D(dynamicScoreCenterScale) D(noResultUtilityForWhite) D(drawEquivalentWinsForWhite)
  D(cpuctExploration) D(cpuctExplorationLog) D(cpuctExplorationBase)
  D(cpuctUtilityStdevPrior) D(cpuctUtilityStdevPriorWeight) D(cpuctUtilityStdevScale)
  D(fpuReductionMax) D(fpuLossProp) Bo(fpuParentWeightByVisitedPolicy) D(fpuParentWeightByVisitedPolicyPow) D(fpuParentWeight)
  D(policyOptimism) D(valueWeightExponent) Bo(ignoreAllHistory) Bo(useNoisePruning) D(noisePruneUtilityScale) D(noisePruningCap)
  Bo(useUncertainty) D(uncertaintyCoeff) D(uncertaintyExponent) D(uncertaintyMaxWeight)
  Bo(useGraphSearch) I(graphSearchRepBound) D(graphSearchCatchUpLeakProb)
  Bo(rootNoiseEnabled) D(rootDirichletNoiseTotalConcentration) D(rootDirichletNoiseWeight)
  D(rootPolicyTemperature) D(rootPolicyTemperatureEarly) D(rootFpuReductionMax) D(rootFpuLossProp)
  I(rootNumSymmetriesToSample) Bo(rootSymmetryPruning) D(rootDesiredPerChildVisitsCoeff) D(rootPolicyOptimism)
  D(chosenMoveTemperature) D(chosenMoveTemperatureEarly) D(chosenMoveTemperatureHalflife)
  D(chosenMoveTemperatureOnlyBelowProb) D(chosenMoveSubtract) D(chosenMovePrune)
  Bo(useLcbForSelection) D(lcbStdevs) D(minVisitPropForLCB) Bo(useNonBuggyLcb)
  D(rootEndingBonusPoints) Bo(rootPruneUselessMoves) Bo(conservativePass) Bo(fillDameBeforePass)
  D(wideRootNoise) Bo(enablePassingHacks) Bo(enableMorePassingHacks)
  D(playoutDoublingAdvantage) Bo(ignorePreRootHistory)
  D(subtreeValueBiasFactor) D(subtreeValueBiasFreeProp) D(subtreeValueBiasWeightExponent)
  I(nodeTableShardsPowerOfTwo) L(maxVisits) L(maxPlayouts) D(futileVisitsThreshold) D(searchFactorAfterOnePass) D(searchFactorAfterTwoPass)
  D(treeReuseCarryOverTimeFactor)
#undef D
#undef I
#undef L
#undef Bo
  if(k == "avoidMYTDaggerHackPla") { p.avoidMYTDaggerHackPla = (Player)l; return true; }
  if(k == "playoutDoublingAdvantagePla") { p.playoutDoublingAdvantagePla = (Player)l; return true; }
  if(k == "nnPolicyTemperature") { p.nnPolicyTemperature = strtof(s, NULL); return true; }
  return false;
}

static SearchParams baseParams(int base) {
  if(base == 1) return SearchParams::forTestsV1();
  if(base == 2) return SearchParams::forTestsV2();
  return SearchParams();
}

//-------------------------------------------------------------------------------------------------
// A case

struct Case {
  int idx = 0;
  string kind;
  int nnX = 0, nnY = 0;
  string nnSeed;
  int xs = 0, ys = 0;
  Rules rules;
  Player initialPla = P_BLACK;
  Player pla = P_BLACK;
  vector<pair<Loc, Player>> moves;
  string sseed;
  int base = 0;
  int64_t maxVisits = 100;
  vector<pair<string, string>> params;
  vector<string> ops;
};

static string caseLine(const Case& c) {
  string s;
  char buf[256];
  snprintf(buf, sizeof buf, "CASE %d kind=%s nn=%dx%d nnseed=%s board=%dx%d ko=%d score=%d tax=%d suicide=%d button=%d whb=%d fpo=%d komi=%08X ipla=%d pla=%d moves=",
           c.idx, c.kind.c_str(), c.nnX, c.nnY, c.nnSeed.c_str(), c.xs, c.ys, c.rules.koRule, c.rules.scoringRule, c.rules.taxRule,
           (int)c.rules.multiStoneSuicideLegal, (int)c.rules.hasButton, c.rules.whiteHandicapBonusRule, (int)c.rules.friendlyPassOk,
           fbits(c.rules.komi), (int)c.initialPla, (int)c.pla);
  s = buf;
  if(c.moves.empty()) s += "-";
  for(size_t i = 0; i < c.moves.size(); i++) {
    snprintf(buf, sizeof buf, "%s%d/%d", i ? "," : "", (int)c.moves[i].first, (int)c.moves[i].second);
    s += buf;
  }
  snprintf(buf, sizeof buf, " sseed=%s base=%d maxvisits=%lld params=", c.sseed.c_str(), c.base, (long long)c.maxVisits);
  s += buf;
  if(c.params.empty()) s += "-";
  for(size_t i = 0; i < c.params.size(); i++)
    s += (i ? "," : "") + c.params[i].first + "=" + c.params[i].second;
  s += " ops=";
  for(size_t i = 0; i < c.ops.size(); i++)
    s += (i ? ";" : "") + c.ops[i];
  return s;
}

//-------------------------------------------------------------------------------------------------
// Printing helpers

static void printRV(const char* label, bool suc, const ReportedSearchValues& v) {
  pf("RV %s ok=%d", label, (int)suc);
  if(suc)
    pf(" win=%016llX loss=%016llX nr=%016llX ssv=%016llX dsv=%016llX es=%016llX esd=%016llX lead=%016llX wl=%016llX util=%016llX weight=%016llX visits=%lld",
       DB(v.winValue), DB(v.lossValue), DB(v.noResultValue), DB(v.staticScoreValue), DB(v.dynamicScoreValue), DB(v.expectedScore),
       DB(v.expectedScoreStdev), DB(v.lead), DB(v.winLossValue), DB(v.utility), DB(v.weight), (long long)v.visits);
  pf("\n");
}

static uint64_t fnv64(const string& s) {
  uint64_t h = 0xcbf29ce484222325ULL;
  for(unsigned char ch : s) { h ^= ch; h *= 0x100000001b3ULL; }
  return h;
}

// The 4 lines of one AnalysisData entry.
static string adText(size_t i, const AnalysisData& d, const Board& board) {
  string s;
  char buf[2048];
  snprintf(buf, sizeof buf, "a %d %s loc=%d visits=%lld psv=%016llX lcb=%016llX radius=%016llX util=%016llX resUtil=%016llX scoreUtil=%016llX wl=%016llX nr=%016llX prior=%016llX sm=%016llX ss=%016llX lead=%016llX ess=%016llX wf=%016llX ws=%016llX wsq=%016llX usq=%016llX smsq=%016llX cv=%lld cws=%016llX order=%d symOf=%d sym=%d node=%d\n",
       (int)i, Location::toString(d.move, board).c_str(), (int)d.move, (long long)d.numVisits, DB(d.playSelectionValue), DB(d.lcb), DB(d.radius),
       DB(d.utility), DB(d.resultUtility), DB(d.scoreUtility), DB(d.winLossValue), DB(d.noResultValue), DB(d.policyPrior), DB(d.scoreMean),
       DB(d.scoreStdev), DB(d.lead), DB(d.ess), DB(d.weightFactor), DB(d.weightSum), DB(d.weightSqSum), DB(d.utilitySqAvg), DB(d.scoreMeanSqAvg),
       (long long)d.childVisits, DB(d.childWeightSum), d.order, (int)d.isSymmetryOf, d.symmetry, (int)(d.node != NULL));
  s = buf;
  s += "pv";
  for(Loc l : d.pv) s += " " + to_string((int)l);
  s += "\npvv";
  for(int64_t v : d.pvVisits) s += " " + to_string((long long)v);
  s += "\npvev";
  for(int64_t v : d.pvEdgeVisits) s += " " + to_string((long long)v);
  s += "\n";
  return s;
}

// The first AD_FULL entries in full (the best moves), the remaining ones as one FNV-1a 64 over their text.
static const size_t AD_FULL = 2;
static void printAD(const vector<AnalysisData>& buf, const Board& board) {
  pf("AD n=%d\n", (int)buf.size());
  string rest;
  for(size_t i = 0; i < buf.size(); i++) {
    string t = adText(i, buf[i], board);
    if(i < AD_FULL) O << t;
    else rest += t;
  }
  if(buf.size() > AD_FULL) pf("adrest n=%d fnv=%016llX\n", (int)(buf.size() - AD_FULL), (unsigned long long)fnv64(rest));
}

// Long text (JSON): the first 800 characters, then its length and FNV-1a 64.
static void printLong(const string& s) {
  if(s.size() <= 1500) { O << s << "\n"; return; }
  O << s.substr(0, 800) << "\n";
  pf("...[len=%d fnv=%016llX]\n", (int)s.size(), (unsigned long long)fnv64(s));
}

// The node table's keys in table order (shard by shard, each shard's std::map in Hash128 order), one
// "<shard> <hash1> <hash0>" line each, as a count and FNV-1a 64: checks the child hashes (Hash128(rand,rand)
// argument order in tree search, the graph hash otherwise) and the shard each node went to.
static void printNodeTableKeys(const Search* search) {
  string text;
  size_t n = 0;
  for(size_t i = 0; i < search->nodeTable->entries.size(); i++) {
    for(const auto& kv : search->nodeTable->entries[i]) {
      char line[64];
      snprintf(line, sizeof line, "%d %016llX %016llX\n", (int)i, (unsigned long long)kv.first.hash1, (unsigned long long)kv.first.hash0);
      text += line;
      n++;
    }
  }
  pf("ntkeys n=%d fnv=%016llX\n", (int)n, (unsigned long long)fnv64(text));
}

static Player parsePersp(const string& s) {
  if(s == "W") return P_WHITE;
  if(s == "B") return P_BLACK;
  return C_EMPTY;
}

//-------------------------------------------------------------------------------------------------
// The op interpreter

struct Runner {
  const Case& c;
  Search* search;
  SearchParams curParams;
  bool ownerOn = false;

  Runner(const Case& cc, Search* s, const SearchParams& p) : c(cc), search(s), curParams(p) {}

  void exec(const string& op) {
    pf("> %s\n", op.c_str());
    vector<string> t = splitOn(op, ':');
    const string& name = t[0];
    if(name == "P") {
      Board board(c.xs, c.ys);
      BoardHistory hist(board, c.initialPla, c.rules, 0);
      for(const auto& m : c.moves)
        hist.makeBoardMoveAssumeLegal(board, m.first, m.second, NULL);
      search->setPosition(c.pla, board, hist);
      Board::printBoard(O, board, Board::NULL_LOC, &hist.moveHistory);
      pf("encore=%d finished=%d next=%d whb=%08X moves=%d\n", hist.encorePhase, (int)hist.isGameFinished, (int)hist.presumedNextMovePla,
         fbits(hist.whiteHandicapBonusScore), (int)hist.moveHistory.size());
    }
    else if(name == "V") {
      auto spec = [&](const string& s) {
        vector<int> v;
        if(s == "-") return v;
        v.assign(Board::MAX_ARR_SIZE, 0);
        for(const string& e : splitOn(s, ',')) {
          vector<string> p = splitOn(e, '/');
          v[atoi(p[0].c_str())] = atoi(p[1].c_str());
        }
        return v;
      };
      search->setAvoidMoveUntilByLoc(spec(t[1]), spec(t[2]));
    }
    else if(name == "AR") search->setAvoidMoveUntilRescaleRoot(t[1] == "1");
    else if(name == "H") search->setRootHintLoc((Loc)atoi(t[1].c_str()));
    else if(name == "O") { search->setAlwaysIncludeOwnerMap(true); ownerOn = true; }
    else if(name == "Y") {
      vector<int> v;
      if(t[1] != "-")
        for(const string& e : splitOn(t[1], ',')) v.push_back(atoi(e.c_str()));
      search->setRootSymmetryPruningOnly(v);
    }
    else if(name == "R") {
      search->runWholeSearch((Player)atoi(t[1].c_str()));
      pf("playouts=%lld\n", (long long)search->lastSearchNumPlayouts);
    }
    else if(name == "B") search->beginSearch(false);
    else if(name == "RS") dumpRoot();
    else if(name == "PT") {
      if(search->rootNode == NULL) { pf("noroot\n"); return; }
      PrintTreeOptions o = PrintTreeOptions().maxDepth(atoi(t[1].c_str()));
      int fl = atoi(t[3].c_str());
      if(fl & 1) o = o.printSqs(true);
      if(fl & 2) o = o.printAvgShorttermError(true);
      if(fl & 4) o = o.minVisitsPropToExpand(0.1);
      if(fl & 8) o = o.maxChildrenToShow(5);
      search->printTree(O, search->rootNode, o, parsePersp(t[2]));
    }
    else if(name == "PV") {
      if(search->rootNode == NULL) { pf("noroot\n"); return; }
      search->printPV(O, search->rootNode, atoi(t[1].c_str()));
      O << "\n";
    }
    else if(name == "AD") {
      vector<AnalysisData> buf;
      search->getAnalysisData(buf, atoi(t[1].c_str()), t[2] == "1", 10, t[3] == "1");
      printAD(buf, search->getRootBoard());
    }
    else if(name == "J") {
      const string& bits = t[3];
      nlohmann::json json;
      bool suc = search->getAnalysisJson(parsePersp(t[1]), atoi(t[2].c_str()), bits[0] == '1', bits[1] == '1', bits[2] == '1', bits[3] == '1',
                                         bits[4] == '1', bits[5] == '1', bits[6] == '1', bits[7] == '1', json);
      pf("success=%d\n", (int)suc);
      printLong(json.dump());
    }
    else if(name == "HS") hashSummary(t);
    else if(name == "CM") {
      int n = atoi(t[1].c_str());
      for(int i = 0; i < n; i++) {
        Loc l = search->getChosenMoveLoc();
        pf("cm %d %s\n", (int)l, Location::toString(l, search->getRootBoard()).c_str());
      }
    }
    else if(name == "PSV") {
      vector<Loc> locs;
      vector<double> vals, visits;
      bool suc = search->getPlaySelectionValues(locs, vals, &visits, 1.0);
      pf("success=%d n=%d nv=%d\n", (int)suc, (int)locs.size(), (int)visits.size());
      if(suc)
        {
          string rest;
          for(size_t i = 0; i < locs.size(); i++) {
            char line[256];
            int len = snprintf(line, sizeof line, "s %d %s %016llX", (int)locs[i], Location::toString(locs[i], search->getRootBoard()).c_str(), DB(vals[i]));
            if(i < visits.size()) snprintf(line + len, sizeof line - len, " %016llX\n", DB(visits[i]));
            else snprintf(line + len, sizeof line - len, " -\n");
            if(i < 12) O << line;
            else rest += line;
          }
          if(locs.size() > 12) pf("srest n=%d fnv=%016llX\n", (int)locs.size() - 12, (unsigned long long)fnv64(rest));
        }
    }
    else if(name == "OW") {
      if(search->rootNode == NULL || !ownerOn) { pf("noroot\n"); return; }
      vector<double> avg = search->getAverageTreeOwnership(NULL);
      pf("ow n=%d", (int)avg.size());
      for(double x : avg) pf(" %016llX", DB(x));
      pf("\n");
      pair<vector<double>, vector<double>> as = search->getAverageAndStandardDeviationTreeOwnership(NULL);
      string avgText;
      for(double x : as.first) { char b[24]; snprintf(b, sizeof b, " %016llX", DB(x)); avgText += b; }
      pf("ows n=%d fnv=%016llX\nstd", (int)as.first.size(), (unsigned long long)fnv64(avgText));
      for(double x : as.second) pf(" %016llX", DB(x));
      pf("\n");
    }
    else if(name == "PM") {
      search->printRootPolicyMap(O);
      search->printRootOwnershipMap(O, P_WHITE);
      search->printRootEndingScoreValueBonus(O);
    }
    else if(name == "M") {
      bool ok = search->makeMove((Loc)atoi(t[1].c_str()), (Player)atoi(t[2].c_str()), t[3] == "1");
      pf("made=%d rootPla=%d\n", (int)ok, (int)search->getRootPla());
    }
    else if(name == "SPN" || name == "SP") {
      SearchParams np = curParams;
      for(const string& kv : splitOn(t[1], ',')) {
        size_t eq = kv.find('=');
        if(!setParam(np, kv.substr(0, eq), kv.substr(eq + 1))) { fprintf(stderr, "bad param %s\n", kv.c_str()); exit(1); }
      }
      curParams = np;
      if(name == "SPN") search->setParamsNoClearing(np);
      else search->setParams(np);
    }
    else if(name == "XO") {
      // The root's nnOutput becomes a copy whose whiteOwnerMap is scaled by 6 and clamped to [-0.999, 0.999] (float
      // math), so getEndingWhiteScoreBonus / shouldSuppressPass see owned spots beyond +-0.95. The old output is
      // leaked (storeNNOutput would park it in a SearchThread until the search ends).
      SearchNode* root = search->rootNode;
      NNOutput* old = root == NULL ? NULL : root->getNNOutput();
      if(old == NULL || old->whiteOwnerMap == NULL) { pf("noxo\n"); return; }
      NNOutput* copy = new NNOutput(*old);
      int n = copy->nnXLen * copy->nnYLen;
      int numExtreme = 0;
      for(int i = 0; i < n; i++) {
        float v = copy->whiteOwnerMap[i] * 6.0f;
        if(v > 0.999f) v = 0.999f;
        if(v < -0.999f) v = -0.999f;
        copy->whiteOwnerMap[i] = v;
        if(v >= 0.95f || v <= -0.95f) numExtreme++;
      }
      root->nnOutput.store(new std::shared_ptr<NNOutput>(copy), std::memory_order_release);
      pf("xo extreme=%d of %d\n", numExtreme, n);
    }
    else if(name == "ECT") {
      if(t[1] != "-") {
        uint64_t u = strtoull(t[1].c_str(), NULL, 16);
        double d;
        memcpy(&d, &u, 8);
        search->effectiveSearchTimeCarriedOver = d;
      }
      pf("ect %016llX\n", DB(search->effectiveSearchTimeCarriedOver));
    }
    else if(name == "CL") search->clearSearch();
    else if(name == "KOMI") search->setKomiIfNew(fromBits((uint32_t)strtoul(t[1].c_str(), NULL, 16)));
    else if(name == "PIN") search->setPlayerIfNew((Player)atoi(t[1].c_str()));
    else if(name == "PCH") search->setPlayerAndClearHistory((Player)atoi(t[1].c_str()));
    else { fprintf(stderr, "unknown op %s\n", op.c_str()); exit(1); }
  }

  static void pfHash(const char* label, const string& text) {
    pf("%s len=%d fnv=%016llX\n", label, (int)text.size(), (unsigned long long)fnv64(text));
  }

  void hashSummary(const vector<string>& t) {
    pf("hs root visits=%lld lastPlayouts=%lld\n", (long long)search->getRootVisits(), (long long)search->lastSearchNumPlayouts);
    ReportedSearchValues v;
    bool suc = search->getRootValues(v);
    printRV("root", suc, v);
    suc = search->getPrunedRootValues(v);
    printRV("pruned", suc, v);
    const SearchNode* root = search->getRootNode();
    if(root == NULL) { pf("noroot\n"); return; }
    const Board& board = search->getRootBoard();
    {
      string text;
      ConstSearchNodeChildrenReference children = root->getChildren();
      int cap = children.getCapacity();
      int n = 0;
      for(int i = 0; i < cap; i++) {
        const SearchNode* child = children[i].getIfAllocated();
        if(child == NULL) break;
        Loc mv = children[i].getMoveLoc();
        char line[512];
        snprintf(line, sizeof line, "c %d %s loc=%d edge=%lld visits=%lld ws=%016llX util=%016llX\n", i, Location::toString(mv, board).c_str(), (int)mv,
                 (long long)children[i].getEdgeVisits(), (long long)child->stats.visits.load(), DB(child->stats.weightSum.load()),
                 DB(child->stats.utilityAvg.load()));
        text += line;
        n++;
      }
      pf("children=%d ", n);
      pfHash("childtext", text);
    }
    {
      ostringstream tree;
      search->printTree(tree, root, PrintTreeOptions().maxDepth(atoi(t[1].c_str())), parsePersp(t[2]));
      pfHash("tree", tree.str());
    }
    {
      vector<AnalysisData> buf;
      search->getAnalysisData(buf, 3, true, 10, true);
      string text;
      for(size_t i = 0; i < buf.size(); i++) text += adText(i, buf[i], board);
      pf("ad n=%d ", (int)buf.size());
      pfHash("adtext", text);
    }
    {
      const string& bits = t[3];
      nlohmann::json json;
      bool jsuc = search->getAnalysisJson(parsePersp(t[2]), 10, bits[0] == '1', bits[1] == '1', bits[2] == '1', bits[3] == '1',
                                          bits[4] == '1', bits[5] == '1', bits[6] == '1', bits[7] == '1', json);
      pf("json success=%d ", (int)jsuc);
      pfHash("jsontext", json.dump());
    }
    {
      vector<Loc> locs;
      vector<double> vals, visits;
      bool psuc = search->getPlaySelectionValues(locs, vals, &visits, 1.0);
      string text;
      for(size_t i = 0; i < locs.size(); i++) {
        char line[256];
        snprintf(line, sizeof line, "s %d %016llX %016llX\n", (int)locs[i], DB(vals[i]), i < visits.size() ? DB(visits[i]) : 0ULL);
        text += line;
      }
      pf("psv success=%d n=%d nv=%d ", (int)psuc, (int)locs.size(), (int)visits.size());
      pfHash("psvtext", text);
    }
    size_t tableSize = 0;
    for(size_t i = 0; i < search->nodeTable->entries.size(); i++) tableSize += search->nodeTable->entries[i].size();
    pf("nodetable=%d postorder=%d\n", (int)tableSize, (int)search->enumerateTreePostOrder().size());
    printNodeTableKeys(search);
  }

  void dumpRoot() {
    pf("root visits=%lld lastPlayouts=%lld\n", (long long)search->getRootVisits(), (long long)search->lastSearchNumPlayouts);
    const SearchNode* root = search->getRootNode();
    if(root != NULL) {
      ConstSearchNodeChildrenReference children = root->getChildren();
      int cap = children.getCapacity();
      int n = 0;
      string rest;
      const int RS_FULL = 5;  // children listed in full; the others are hashed (FNV-1a 64 of their lines)
      for(int i = 0; i < cap; i++) {
        const SearchNode* child = children[i].getIfAllocated();
        if(child == NULL) break;
        Loc mv = children[i].getMoveLoc();
        char line[512];
        snprintf(line, sizeof line, "c %d %s loc=%d edge=%lld visits=%lld ws=%016llX util=%016llX\n", i, Location::toString(mv, search->getRootBoard()).c_str(), (int)mv,
                 (long long)children[i].getEdgeVisits(), (long long)child->stats.visits.load(), DB(child->stats.weightSum.load()),
                 DB(child->stats.utilityAvg.load()));
        if(n < RS_FULL) O << line;
        else rest += line;
        n++;
      }
      if(n > RS_FULL) pf("crest n=%d fnv=%016llX\n", n - RS_FULL, (unsigned long long)fnv64(rest));
      pf("children=%d\n", n);
    }
    ReportedSearchValues v;
    bool suc = search->getRootValues(v);
    printRV("root", suc, v);
    suc = search->getPrunedRootValues(v);
    printRV("pruned", suc, v);
    suc = search->getRootRawNNValues(v);
    printRV("raw", suc, v);
    double surprise = 0, sEnt = 0, pEnt = 0;
    suc = search->getPolicySurpriseAndEntropy(surprise, sEnt, pEnt);
    pf("surprise ok=%d %016llX %016llX %016llX\n", (int)suc, DB(surprise), DB(sEnt), DB(pEnt));
    if(root != NULL) {
      double sharp = 0;
      suc = search->getSharpScore(root, sharp);
      pf("sharp ok=%d %016llX\n", (int)suc, DB(sharp));
      pair<double, double> st = search->getShallowAverageShorttermWLAndScoreError(root);
      pf("shallow %016llX %016llX\n", DB(st.first), DB(st.second));
      size_t tableSize = 0;
      for(size_t i = 0; i < search->nodeTable->entries.size(); i++) tableSize += search->nodeTable->entries[i].size();
      pf("nodetable=%d postorder=%d\n", (int)tableSize, (int)search->enumerateTreePostOrder().size());
      printNodeTableKeys(search);
    }
  }
};

//-------------------------------------------------------------------------------------------------
// Running a case

static NNEvaluator* newNNEval(const Case& c) {
  NNEvaluator* e = new NNEvaluator(
    "search_nonn", "search_nonn", "", NULL, 16, c.nnX, c.nnY, false, true, 16, 12, true, "", "", false,
    enabled_t::False, enabled_t::False, 1, {0}, "runSearchTestsRandSeed" + c.nnSeed, false, 0
  );
  e->spawnServerThreads();
  return e;
}

struct Gen {
  Case& c;
  Runner* r;
  bool failed = false;
  void run(const string& op) {
    if(failed) return;
    c.ops.push_back(op);
    if(getenv("ORACLE_TRACE")) fprintf(stderr, "case %d: %s\n", c.idx, op.c_str());
    try {
      r->exec(op);
    }
    catch(const std::exception& e) {
      fprintf(stderr, "case %d op %s: exception: %s\n", c.idx, op.c_str(), e.what());
      pf("EXC\n");
      failed = true;
    }
  }
};

static int numCases = 0;
static map<string, int> featureCounts;

static void runCase(Case& c, const function<void(Gen&)>& script) {
  O.str("");
  O.clear();
  Logger logger(nullptr, false, false, false, false);
  logger.addOStream(O);
  NNEvaluator* nnEval = newNNEval(c);
  SearchParams params = baseParams(c.base);
  params.numThreads = 1;
  for(const auto& kv : c.params)
    if(!setParam(params, kv.first, kv.second)) { fprintf(stderr, "bad param %s\n", kv.first.c_str()); exit(1); }
  params.maxVisits = c.maxVisits;
  Search* search = new Search(params, nnEval, &logger, c.sseed);
  Runner runner(c, search, params);
  Gen g{c, &runner};
  if(getenv("ORACLE_TRACE")) fprintf(stderr, "case %d: %s\n", c.idx, caseLine(c).c_str());
  script(g);
  delete search;
  nnEval->killServerThreads();
  delete nnEval;
  string line = caseLine(c);
  string out = O.str();
  printf("%s\n%sEND %d\n", line.c_str(), out.c_str(), c.idx);
  numCases++;
  for(const auto& kv : c.params) featureCounts[kv.first]++;
  featureCounts["base" + to_string(c.base)]++;
  featureCounts["kind:" + c.kind]++;
}

//-------------------------------------------------------------------------------------------------
// Generators

// ORACLE_SEED / ORACLE_STRESS_ONLY=<n> (environment) are for exploration only: another seed, and n stress cases instead of the fixture.
static Rand gen(getenv("ORACLE_SEED") ? getenv("ORACLE_SEED") : "search_nonn_oracle_20261022");
static int rnd(int n) { return (int)gen.nextUInt((uint32_t)n); }
static bool prob(double p) { return gen.nextBool(p); }
template<typename T, size_t N> static const T& pick(const T (&a)[N]) { return a[rnd((int)N)]; }

static Rules makeRules(string& name) {
  static const char* names[] = {"tromp-taylor", "chinese", "japanese", "korean", "aga", "nz", "stonescoring", "agabutton", "ancientterritory", "goe"};
  name = pick(names);
  Rules r = Rules::parseRules(name);
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

struct Pos {
  vector<pair<Loc, Player>> moves;
  Board board;
  BoardHistory hist;
  Pos(int xs, int ys, const Rules& rules, Player ipla) : board(xs, ys), hist(board, ipla, rules, 0) {}
  bool play(Loc loc, Player pla) {
    if(hist.isGameFinished) return false;
    if(!hist.isLegal(board, loc, pla)) return false;
    hist.makeBoardMoveAssumeLegal(board, loc, pla, NULL);
    moves.push_back(make_pair(loc, pla));
    return true;
  }
  Player nextPla() const { return hist.presumedNextMovePla; }
};

// Random legal moves (passes with probability passPct percent, out-of-turn moves with outOfTurnPct).
static void randomMoves(Pos& p, int xs, int ys, int numMoves, int passPct, int outOfTurnPct) {
  for(int m = 0; m < numMoves && !p.hist.isGameFinished; m++) {
    Player pla = p.nextPla();
    if(outOfTurnPct > 0 && rnd(100) < outOfTurnPct) pla = getOpp(pla);
    Loc loc = Board::PASS_LOC;
    if(rnd(100) >= passPct) {
      for(int tries = 0; tries < 40; tries++) {
        Loc l = Location::getLoc(rnd(xs), rnd(ys), xs);
        if(p.hist.isLegal(p.board, l, pla)) { loc = l; break; }
      }
    }
    p.play(loc, pla);
  }
}

static void passPair(Pos& p) {
  p.play(Board::PASS_LOC, p.nextPla());
  p.play(Board::PASS_LOC, p.nextPla());
}

// Hand-built ko (see the diagram in koMoves) with the shape's origin at (ox,oy).
static void koMoves(Pos& p, int xs, int ys, int ox, int oy, int stage, int encoreRounds, int tailPassPairs) {
  struct M { int x, y, pla; };  // y < 0: pass
  const int B = P_BLACK, W = P_WHITE;
  // . B W .      B(1,0) W(2,0) B(0,1) W(3,1) B(1,2) W(2,2) pass W(1,1): black to move; B(2,1) then captures one stone:
  // B W . W      a ko at (1,1) (white may not retake at once)
  // . B W .
  vector<M> script = {
    {1, 0, B}, {2, 0, W}, {0, 1, B}, {3, 1, W}, {1, 2, B}, {2, 2, W}, {0, -1, B}, {1, 1, W},
  };
  for(const M& m : script) {
    Loc loc = m.y < 0 ? Board::PASS_LOC : Location::getLoc(ox + m.x, oy + m.y, xs);
    p.play(loc, m.pla);
  }
  for(int i = 0; i < encoreRounds; i++) passPair(p);
  Loc elsewhere[6] = {
    Location::getLoc(xs - 1, ys - 1, xs), Location::getLoc(0, ys - 1, xs), Location::getLoc(xs - 1, ys - 2, xs),
    Location::getLoc(0, ys - 2, xs), Location::getLoc(xs - 1, ys - 3, xs), Location::getLoc(0, ys - 3, xs),
  };
  Loc koPoint = Location::getLoc(ox + 2, oy + 1, xs);
  Loc koPoint2 = Location::getLoc(ox + 1, oy + 1, xs);
  // stage: 0 = just the shape (black to move) ... 1 = capture, 2.. = threats and recaptures
  if(stage >= 1) p.play(koPoint, B);                                  // black captures
  if(stage >= 2) p.play(elsewhere[0], W);                             // white's ko threat
  if(stage >= 3) p.play(elsewhere[1], B);                             // black answers
  if(stage >= 4) p.play(koPoint2, W);                                 // white retakes
  if(stage >= 5) p.play(elsewhere[2], B);                             // black's ko threat
  if(stage >= 6) p.play(elsewhere[3], W);                             // white answers
  if(stage >= 7) p.play(koPoint, B);                                  // black retakes
  if(stage >= 8) p.play(elsewhere[4], W);
  if(stage >= 9) p.play(elsewhere[5], B);
  if(stage >= 10) p.play(koPoint2, W);
  for(int i = 0; i < tailPassPairs; i++) passPair(p);
}

// White's pass-alive group (two single-point eyes) in the top-left corner, then black passes 4 times
// with white playing in between: a root where rootPruneUselessMoves applies (white to move).
static void passyMoves(Pos& p, int xs, int ys) {
  int ws[8][2] = {{0, 1}, {1, 1}, {2, 1}, {3, 1}, {4, 1}, {1, 0}, {3, 0}, {4, 0}};
  for(auto& w : ws) p.play(Location::getLoc(w[0], w[1], xs), P_WHITE);
  p.play(Board::PASS_LOC, P_BLACK);
  p.play(Location::getLoc(0, ys - 1, xs), P_WHITE);
  p.play(Board::PASS_LOC, P_BLACK);
  p.play(Location::getLoc(1, ys - 1, xs), P_WHITE);
  p.play(Board::PASS_LOC, P_BLACK);
  p.play(Location::getLoc(2, ys - 1, xs), P_WHITE);
  p.play(Board::PASS_LOC, P_BLACK);
}

//-------------------------------------------------------------------------------------------------
// Param features and op scripts

static void addParams(Case& c, int ys, bool small) {
  auto add = [&](const string& k, const string& v) { c.params.push_back(make_pair(k, v)); };
  auto bit = [&]() { return string(rnd(2) ? "1" : "0"); };
  int r = rnd(10);
  c.base = r < 4 ? 0 : (r < 7 ? 1 : 2);
  (void)ys;
  (void)small;
  // nodeTableShardsPowerOfTwo: small (the default 2^16 shards is costly), the default occasionally
  static const char* shards[] = {"0", "1", "2", "3", "4"};
  if(rnd(25) != 0) add("nodeTableShardsPowerOfTwo", pick(shards));
  if(prob(0.5)) add("useGraphSearch", bit());
  if(prob(0.35)) add("rootSymmetryPruning", bit());
  if(prob(0.25)) add("wideRootNoise", prob(0.5) ? "0.05" : "0");
  if(prob(0.3)) {
    add("useLcbForSelection", "1");
    add("useNonBuggyLcb", bit());
  }
  if(prob(0.3)) {
    static const char* f[] = {"0", "0.35", "0.45", "0.8"};
    static const char* e[] = {"0.5", "0.85", "0.25", "1.0"};
    add("subtreeValueBiasFactor", pick(f));
    if(prob(0.6)) add("subtreeValueBiasWeightExponent", pick(e));
    if(prob(0.3)) add("subtreeValueBiasFreeProp", prob(0.5) ? "0.5" : "1.0");
  }
  if(prob(0.25)) add("enablePassingHacks", bit());
  if(prob(0.25)) add("enableMorePassingHacks", bit());
  if(prob(0.25)) add("conservativePass", bit());
  if(prob(0.25)) add("fillDameBeforePass", bit());
  if(prob(0.3)) {
    static const char* v[] = {"0.5", "-0.5", "1.0", "0"};
    add("rootEndingBonusPoints", pick(v));
  }
  if(prob(0.25)) add("rootPruneUselessMoves", bit());
  if(prob(0.3)) {
    static const char* v[] = {"0", "0.3", "1.0", "0.05"};
    add("dynamicScoreUtilityFactor", pick(v));
  }
  if(prob(0.3)) {
    static const char* v[] = {"0", "0.1", "0.5", "0.02"};
    add("staticScoreUtilityFactor", pick(v));
  }
  if(prob(0.2)) {
    add("rootNoiseEnabled", "1");
    if(prob(0.5)) add("rootDirichletNoiseWeight", prob(0.5) ? "0.5" : "0.1");
  }
  if(prob(0.25)) {
    static const char* v[] = {"1.0", "1.3", "0.8", "2.0"};
    add("rootPolicyTemperature", pick(v));
    if(prob(0.6)) add("rootPolicyTemperatureEarly", pick(v));
  }
  if(prob(0.3)) {
    static const char* v[] = {"0", "0.3", "1.0", "2.0"};
    add("chosenMoveTemperature", pick(v));
    if(prob(0.6)) add("chosenMoveTemperatureEarly", pick(v));
    if(prob(0.5)) {
      static const char* h[] = {"19", "5", "0.5", "50"};
      add("chosenMoveTemperatureHalflife", pick(h));
    }
  }
  if(prob(0.2)) {
    static const char* v[] = {"0", "1.5", "3", "0.5"};
    add("chosenMoveSubtract", pick(v));
  }
  if(prob(0.2)) {
    static const char* v[] = {"0", "1", "2", "5"};
    add("chosenMovePrune", pick(v));
  }
  if(prob(0.3)) {
    static const char* v[] = {"0", "0.25", "0.5", "1.0"};
    add("valueWeightExponent", pick(v));
  }
  if(prob(0.2)) add("useNoisePruning", bit());
  if(prob(0.2)) add("useUncertainty", bit());
  if(prob(0.25)) {
    static const char* v[] = {"0", "0.5", "1.0", "0.2"};
    add("policyOptimism", pick(v));
    if(prob(0.5)) add("rootPolicyOptimism", pick(v));
  }
  if(prob(0.25)) {
    add("fpuParentWeightByVisitedPolicy", bit());
    if(prob(0.4)) add("fpuParentWeightByVisitedPolicyPow", prob(0.5) ? "2.0" : "0.5");
  }
  if(prob(0.25)) {
    add("cpuctUtilityStdevPrior", prob(0.5) ? "0.4" : "0.2");
    add("cpuctUtilityStdevPriorWeight", prob(0.5) ? "2.0" : "5.0");
    add("cpuctUtilityStdevScale", prob(0.5) ? "0.85" : "1.0");
  }
  if(prob(0.2)) {
    static const char* v[] = {"0", "0.05", "0.2", "1.0"};
    add("graphSearchCatchUpLeakProb", pick(v));
  }
  if(prob(0.2)) {
    static const char* v[] = {"0.5", "0.3", "0.7", "0"};
    add("drawEquivalentWinsForWhite", pick(v));
  }
  if(prob(0.2)) {
    static const char* v[] = {"0", "1.0", "-0.5", "2.0"};
    add("playoutDoublingAdvantage", pick(v));
    if(prob(0.5)) add("playoutDoublingAdvantagePla", to_string(rnd(3)));
  }
  if(prob(0.2)) add("ignorePreRootHistory", bit());
  if(prob(0.08)) add("ignoreAllHistory", bit());
  if(prob(0.25)) {
    static const char* v[] = {"1", "2", "4"};
    add("rootNumSymmetriesToSample", pick(v));
  }
  if(prob(0.15)) add("avoidMYTDaggerHackPla", to_string(rnd(3)));
  if(prob(0.15)) add("noResultUtilityForWhite", prob(0.5) ? "0.3" : "-0.4");
  if(prob(0.15)) add("cpuctExploration", prob(0.5) ? "1.5" : "0.7");
  if(prob(0.1)) add("nnPolicyTemperature", prob(0.5) ? "1.2" : "0.8");
  // Less common knobs: three of them at a time in a third of the cases.
  if(prob(0.33)) {
    struct Knob { const char* key; const char* vals[3]; };
    static const Knob knobs[] = {
      {"winLossUtilityFactor", {"1.0", "0.5", "2.0"}}, {"dynamicScoreCenterZeroWeight", {"0.2", "0.0", "0.5"}},
      {"dynamicScoreCenterScale", {"0.75", "1.0", "0.5"}}, {"cpuctExplorationLog", {"0.4", "0.0", "0.8"}},
      {"cpuctExplorationBase", {"500", "100", "2000"}}, {"fpuReductionMax", {"0.2", "0.0", "0.5"}}, {"fpuLossProp", {"0.0", "0.3", "1.0"}},
      {"rootFpuReductionMax", {"0.1", "0.0", "0.4"}}, {"rootFpuLossProp", {"0.0", "0.3", "1.0"}}, {"fpuParentWeight", {"0.0", "0.5", "1.0"}},
      {"noisePruneUtilityScale", {"0.15", "0.05", "0.5"}}, {"noisePruningCap", {"1e50", "0.5", "2.0"}},
      {"uncertaintyCoeff", {"0.2", "0.5", "0.05"}}, {"uncertaintyExponent", {"1.0", "0.5", "2.0"}}, {"uncertaintyMaxWeight", {"8.0", "2.0", "20.0"}},
      {"rootDesiredPerChildVisitsCoeff", {"0.0", "2.0", "10.0"}}, {"rootDirichletNoiseTotalConcentration", {"10.83", "3.0", "30.0"}},
      {"graphSearchRepBound", {"11", "3", "30"}}, {"lcbStdevs", {"4.0", "2.0", "6.0"}}, {"minVisitPropForLCB", {"0.05", "0.3", "0.0"}},
      {"searchFactorAfterOnePass", {"0.5", "0.2", "1.0"}}, {"searchFactorAfterTwoPass", {"0.25", "0.1", "1.0"}},
      {"subtreeValueBiasFreeProp", {"0.8", "0.0", "1.0"}}, {"futileVisitsThreshold", {"0.4", "0.0", "0.8"}},
    };
    for(int k = 0; k < 3; k++) {
      const Knob& kn = knobs[rnd((int)(sizeof(knobs) / sizeof(knobs[0])))];
      add(kn.key, kn.vals[rnd(3)]);
    }
  }
  if(prob(0.1)) add("maxPlayouts", to_string(5 + rnd(150)));
}

static string perspChar() {
  static const char* p[] = {"W", "B", "N", "W"};
  return pick(p);
}

// The standard battery of ops after a search.
static void report(Gen& g, bool full, bool deep, bool ownerOn, int xs, int ys, bool noChoose = false) {
  if(full || prob(0.6)) g.run("RS");
  int flags = prob(0.3) ? rnd(16) : 0;
  if(full || prob(0.5)) g.run("PT:" + to_string(deep ? 2 : 1) + ":" + perspChar() + ":" + to_string(flags));
  if(!full) {
    // lite: one analysis JSON
    string bits = string(prob(0.5) ? "1" : "0") + (prob(0.4) ? "1" : "0") + "000" + "0" + (prob(0.5) ? "1" : "0") + (prob(0.3) ? "1" : "0");
    g.run("J:" + perspChar() + ":" + to_string(1 + rnd(8)) + ":" + bits);
    return;
  }
  g.run("PV:" + to_string(1 + rnd(12)));
  g.run("AD:" + to_string(rnd(4)) + ":" + (prob(0.5) ? "1" : "0") + ":" + (prob(0.5) ? "1" : "0"));
  if(prob(0.25)) g.run("AD:" + to_string(rnd(4)) + ":" + (prob(0.5) ? "1" : "0") + ":" + (prob(0.5) ? "1" : "0"));
  bool smallBoard = xs * ys <= 49;
  string bits = string(prob(0.5) ? "1" : "0") + (prob(0.25) ? "1" : "0");
  if(ownerOn) bits += string(prob(0.5) ? "1" : "0") + (prob(0.5) ? "1" : "0") + (smallBoard && prob(0.3) ? "1" : "0") + (smallBoard && prob(0.3) ? "1" : "0");
  else bits += "0000";
  bits += string(prob(0.5) ? "1" : "0") + (prob(0.3) ? "1" : "0");
  g.run("J:" + perspChar() + ":" + to_string(1 + rnd(10)) + ":" + bits);
  if(!noChoose) g.run("CM:" + to_string(1 + rnd(4)));
  g.run("PSV");
  if(ownerOn) {
    if(prob(0.5)) g.run("OW");
    if(prob(0.5)) g.run("PM");
  }
}

// The best root child by edge visits (first on ties), or NULL_LOC.
static Loc bestRootChild(Gen& g) {
  const Search* s = g.r->search;
  if(s->rootNode == NULL) return Board::NULL_LOC;
  ConstSearchNodeChildrenReference children = s->rootNode->getChildren();
  int cap = children.getCapacity();
  Loc best = Board::NULL_LOC;
  int64_t bestVisits = -1;
  for(int i = 0; i < cap; i++) {
    if(children[i].getIfAllocated() == NULL) break;
    int64_t v = children[i].getEdgeVisits();
    if(v > bestVisits) { bestVisits = v; best = children[i].getMoveLoc(); }
  }
  return best;
}

// A random legal move for the search's root player (pass if none is found).
static Loc randomLegalRootMove(Gen& g, Player pla) {
  const Search* s = g.r->search;
  const Board& b = s->getRootBoard();
  if(prob(0.05)) return Location::getLoc(rnd(b.x_size), rnd(b.y_size), b.x_size);  // possibly illegal: makeMove must refuse it
  for(int tries = 0; tries < 40; tries++) {
    Loc l = Location::getLoc(rnd(b.x_size), rnd(b.y_size), b.x_size);
    if(s->isLegalTolerant(l, pla)) return l;
  }
  return Board::PASS_LOC;
}

enum Flow { FLOW_BASIC, FLOW_REUSE, FLOW_PARAMS, FLOW_ZERO, FLOW_AVOID, FLOW_LOWVISITS };

static string avoidSpec(const Case& c, const Board& b, int mode, Player forPla) {
  // mode 0: random subset of empty points, until 1..3; 1: all but a few (and pass); 2: everything incl. pass; 3: only pass avoided
  string s;
  auto addE = [&](Loc l, int until) {
    if(!s.empty()) s += ",";
    s += to_string((int)l) + "/" + to_string(until);
  };
  (void)c;
  (void)forPla;
  if(mode == 3) { addE(Board::PASS_LOC, 1 + rnd(3)); return s; }
  int keep = mode == 1 ? 2 + rnd(3) : 0;
  int kept = 0;
  for(int y = 0; y < b.y_size; y++)
    for(int x = 0; x < b.x_size; x++) {
      Loc l = Location::getLoc(x, y, b.x_size);
      if(mode == 0) {
        if(rnd(3) == 0) addE(l, 1 + rnd(3));
      }
      else if(mode == 1) {
        if(kept < keep && rnd(b.x_size * b.y_size) < keep + 1) { kept++; continue; }
        addE(l, 3);
      }
      else addE(l, 3);
    }
  if(mode == 1 || mode == 2) addE(Board::PASS_LOC, 3);
  if(s.empty()) s = "-";
  return s;
}

static void script(Gen& g, Flow flow, bool ownerOn) {
  Case& c = g.c;
  Player rootPla = c.pla;
  string rp = to_string((int)rootPla);
  g.run("P");
  if(ownerOn) g.run("O");
  if(prob(0.08)) g.run("H:" + to_string((int)Location::getLoc(rnd(c.xs), rnd(c.ys), c.xs)));
  if(prob(0.1)) {
    static const char* sy[] = {"0", "0,1", "0,2,3", "0,4,5,6,7", "-"};
    g.run(string("Y:") + pick(sy));
  }
  if(flow == FLOW_AVOID) {
    Board b(c.xs, c.ys);
    int mode = rnd(4);
    string bs = prob(0.2) ? "-" : avoidSpec(c, b, mode, P_BLACK);
    string ws = prob(0.2) ? "-" : avoidSpec(c, b, mode, P_WHITE);
    if(prob(0.5)) ws = bs;
    g.run("V:" + bs + ":" + ws);
    if(prob(0.3)) g.run("AR:1");
  }
  bool deep = c.maxVisits <= 30;
  if(flow == FLOW_ZERO) {
    g.run("B");
    report(g, true, deep, ownerOn, c.xs, c.ys, true);
    if(prob(0.5)) {
      g.run("R:" + rp);
      report(g, false, deep, ownerOn, c.xs, c.ys);
    }
    return;
  }
  g.run("R:" + (prob(0.08) ? to_string((int)getOpp(rootPla)) : rp));
  report(g, true, deep, ownerOn, c.xs, c.ys);
  if(flow == FLOW_BASIC || flow == FLOW_LOWVISITS || flow == FLOW_AVOID) {
    if(flow == FLOW_AVOID && prob(0.6)) {
      // make a move and continue the search (tree reuse under root restrictions)
      Loc mv = prob(0.7) ? bestRootChild(g) : randomLegalRootMove(g, g.r->search->getRootPla());
      if(mv == Board::NULL_LOC) mv = Board::PASS_LOC;
      Player mp = g.r->search->getRootPla();
      g.run("M:" + to_string((int)mv) + ":" + to_string((int)mp) + ":0");
      g.run("R:" + to_string((int)g.r->search->getRootPla()));
      report(g, false, deep, ownerOn, c.xs, c.ys);
    }
    return;
  }
  if(flow == FLOW_REUSE) {
    int rounds = 1 + rnd(3);
    for(int i = 0; i < rounds; i++) {
      Player mp = g.r->search->getRootPla();
      Loc mv;
      int mode = rnd(10);
      if(mode < 6) mv = bestRootChild(g);
      else if(mode < 9) mv = randomLegalRootMove(g, mp);
      else mv = Board::PASS_LOC;
      if(mv == Board::NULL_LOC) mv = Board::PASS_LOC;
      if(prob(0.07)) mp = getOpp(mp);
      g.run("M:" + to_string((int)mv) + ":" + to_string((int)mp) + ":" + (prob(0.15) ? "1" : "0"));
      if(prob(0.2))
        report(g, false, deep, ownerOn, c.xs, c.ys);  // tree right after the move, before searching
      if(prob(0.6)) g.run("SPN:maxVisits=" + to_string((long long)(g.c.maxVisits * (2 + rnd(2)))));
      g.run("R:" + to_string((int)g.r->search->getRootPla()));
      report(g, i == rounds - 1, deep, ownerOn, c.xs, c.ys);
    }
    return;
  }
  if(flow == FLOW_PARAMS) {
    int steps = 1 + rnd(3);
    for(int i = 0; i < steps; i++) {
      int kind = rnd(5);
      if(kind <= 1) {
        // params that can change mid-tree without clearing
        string kv = "maxVisits=" + to_string((long long)(c.maxVisits * (1 + rnd(3))));
        static const char* extra[] = {"cpuctExploration=1.4", "chosenMoveTemperature=0.5", "useLcbForSelection=1", "wideRootNoise=0.05", "lcbStdevs=3",
                                      "chosenMovePrune=2", "chosenMoveSubtract=1", "cpuctExplorationLog=0.2"};
        if(prob(0.7)) kv += string(",") + pick(extra);
        g.run("SPN:" + kv);
      }
      else if(kind == 2) {
        // a full parameter change (clears the search)
        string kv = "maxVisits=" + to_string((long long)(c.maxVisits));
        static const char* extra[] = {"useGraphSearch=1", "useGraphSearch=0", "rootSymmetryPruning=1", "subtreeValueBiasFactor=0.35", "useUncertainty=1",
                                      "valueWeightExponent=0.25", "conservativePass=1", "enablePassingHacks=1", "dynamicScoreUtilityFactor=0.3",
                                      "rootNumSymmetriesToSample=2", "fpuParentWeightByVisitedPolicy=1"};
        kv += string(",") + pick(extra);
        g.run("SP:" + kv);
      }
      else if(kind == 3) g.run("CL");
      else {
        // komi change (clears the search unless equal)
        static const float komis[] = {7.5f, 6.5f, 0.0f, 5.5f};
        char buf[32];
        snprintf(buf, sizeof buf, "%08X", fbits(pick(komis)));
        g.run(string("KOMI:") + buf);
      }
      if(prob(0.85)) {
        g.run("R:" + to_string((int)g.r->search->getRootPla()));
        report(g, prob(0.5), deep, ownerOn, c.xs, c.ys);
      }
      else
        g.run("RS");
    }
  }
}


// Stress cases: more searches and params than the detailed cases, reported compactly (HS: hashes).
static string hsOp(bool ownerOn, int xs, int ys) {
  string bits = string(prob(0.5) ? "1" : "0") + (prob(0.3) ? "1" : "0");
  bool smallBoard = xs * ys <= 49;
  if(ownerOn) bits += string(prob(0.5) ? "1" : "0") + (prob(0.5) ? "1" : "0") + (smallBoard && prob(0.3) ? "1" : "0") + (smallBoard && prob(0.3) ? "1" : "0");
  else bits += "0000";
  bits += string(prob(0.5) ? "1" : "0") + (prob(0.3) ? "1" : "0");
  return "HS:" + to_string(1 + rnd(3)) + ":" + perspChar() + ":" + bits;
}

static void stressScript(Gen& g, bool ownerOn) {
  Case& c = g.c;
  g.run("P");
  if(ownerOn) g.run("O");
  if(prob(0.08)) g.run("H:" + to_string((int)Location::getLoc(rnd(c.xs), rnd(c.ys), c.xs)));
  if(prob(0.1)) {
    static const char* sy[] = {"0", "0,1", "0,2,3", "0,4,5,6,7", "-"};
    g.run(string("Y:") + pick(sy));
  }
  if(prob(0.15)) {
    Board b(c.xs, c.ys);
    int mode = rnd(3);
    g.run("V:" + avoidSpec(c, b, mode, P_BLACK) + ":" + avoidSpec(c, b, mode, P_WHITE));
  }
  g.run("R:" + to_string((int)c.pla));
  g.run(hsOp(ownerOn, c.xs, c.ys));
  g.run("CM:" + to_string(1 + rnd(3)));
  int rounds = 1 + rnd(3);
  for(int i = 0; i < rounds; i++) {
    Player rp = g.r->search->getRootPla();
    int kind = rnd(10);
    if(kind <= 3) {
      Loc mv = kind <= 1 ? bestRootChild(g) : randomLegalRootMove(g, rp);
      if(mv == Board::NULL_LOC) mv = Board::PASS_LOC;
      g.run("M:" + to_string((int)mv) + ":" + to_string((int)(prob(0.05) ? getOpp(rp) : rp)) + ":" + (prob(0.2) ? "1" : "0"));
    }
    else if(kind == 4 || kind == 5) {
      string kv = "maxVisits=" + to_string((long long)(c.maxVisits * (1 + rnd(2))));
      static const char* extra[] = {"cpuctExploration=1.4", "chosenMoveTemperature=0.5", "useLcbForSelection=1", "wideRootNoise=0.05", "lcbStdevs=3",
                                    "chosenMovePrune=2", "cpuctExplorationLog=0.2", "fpuReductionMax=0.3", "rootFpuLossProp=0.2"};
      if(prob(0.7)) kv += string(",") + pick(extra);
      g.run("SPN:" + kv);
    }
    else if(kind == 6) {
      static const char* extra[] = {"useGraphSearch=1", "useGraphSearch=0", "rootSymmetryPruning=1", "subtreeValueBiasFactor=0.35", "useUncertainty=1",
                                    "valueWeightExponent=0.25", "conservativePass=1", "enablePassingHacks=1", "dynamicScoreUtilityFactor=0.3",
                                    "rootNumSymmetriesToSample=2", "fpuParentWeightByVisitedPolicy=1", "enableMorePassingHacks=1", "rootNoiseEnabled=1",
                                    "useNoisePruning=1", "policyOptimism=0.5", "playoutDoublingAdvantage=1.0", "wideRootNoise=0.05"};
      g.run(string("SP:maxVisits=") + to_string((long long)c.maxVisits) + "," + pick(extra));
    }
    else if(kind == 7) g.run("CL");
    else if(kind == 8) g.run("PCH:" + to_string((int)getOpp(rp)));
    else {
      static const float komis[] = {7.5f, 6.5f, 0.0f, 5.5f, -2.0f};
      char buf[32];
      snprintf(buf, sizeof buf, "%08X", fbits(pick(komis)));
      g.run(string("KOMI:") + buf);
    }
    g.run("R:" + to_string((int)g.r->search->getRootPla()));
    g.run(hsOp(ownerOn, c.xs, c.ys));
    if(prob(0.4)) g.run("CM:" + to_string(1 + rnd(3)));
  }
}

// Wide trees: enough visits on big boards for nodes to have more than 64 children (the third children array,
// STATE_GROWING2 / STATE_EXPANDED2).
static void wideScript(Gen& g, bool ownerOn) {
  Case& c = g.c;
  g.run("P");
  if(ownerOn) g.run("O");
  g.run("R:" + to_string((int)c.pla));
  g.run(hsOp(ownerOn, c.xs, c.ys));
  g.run("CM:2");
  g.run("SPN:maxVisits=" + to_string((long long)(c.maxVisits * 3 / 2)));
  g.run("R:" + to_string((int)c.pla));
  g.run(hsOp(ownerOn, c.xs, c.ys));
  Loc mv = bestRootChild(g);
  if(mv == Board::NULL_LOC) mv = Board::PASS_LOC;
  g.run("M:" + to_string((int)mv) + ":" + to_string((int)c.pla) + ":0");
  g.run("R:" + to_string((int)g.r->search->getRootPla()));
  g.run("RS");
  g.run(hsOp(ownerOn, c.xs, c.ys));
}

//-------------------------------------------------------------------------------------------------

static int64_t pickVisits(int xs, int ys, int cap) {
  int r = rnd(100);
  int64_t v;
  if(r < 5) v = 1 + rnd(2);
  else if(r < 40) v = 3 + rnd(28);
  else if(r < 80) v = 31 + rnd(70);
  else if(r < 96) v = 101 + rnd(150);
  else v = 251 + rnd(150);
  if(xs * ys > 100) v = std::min<int64_t>(v, 150);
  return std::min<int64_t>(v, cap);
}

static void finishCase(Case& c, const Pos& p, Flow flow, bool ownerOn, bool outOfTurnRoot, bool stress = false, bool wide = false) {
  c.moves = p.moves;
  c.pla = p.nextPla();
  (void)outOfTurnRoot;  // setPosition with a player other than the history's presumed next one trips an nneval assert
  if(c.pla != P_BLACK && c.pla != P_WHITE) c.pla = P_BLACK;
  c.idx = numCases;
  c.sseed = "oracleSearch" + to_string(c.idx);
  c.nnSeed = "oracle" + to_string(c.idx);
  runCase(c, [&](Gen& g) { if(wide) wideScript(g, ownerOn); else if(stress) stressScript(g, ownerOn); else script(g, flow, ownerOn); });
}

static void setSizes(Case& c, int xs, int ys) {
  c.xs = xs;
  c.ys = ys;
  c.nnX = xs;
  c.nnY = ys;
  if(prob(0.25)) c.nnX = std::min(19, xs + 1 + rnd(3));
  if(prob(0.25)) c.nnY = std::min(19, ys + 1 + rnd(3));
  if(prob(0.1)) c.nnX = c.nnY = std::min(19, std::max(xs, ys) + rnd(2));

  if(prob(0.05)) c.nnX = c.nnY = 19;}

static Flow pickFlow() {
  int r = rnd(100);
  if(r < 40) return FLOW_BASIC;
  if(r < 70) return FLOW_REUSE;
  return FLOW_PARAMS;
}

int main() {
  Board::initHash();
  ScoreValue::initTables();
  const int stressOnly = getenv("ORACLE_STRESS_ONLY") ? atoi(getenv("ORACLE_STRESS_ONLY")) : -1;
  const bool detailed = stressOnly < 0;

  // 1. Random legal games (odd/even/non-square sizes 5..13, every rules family), varied params, all flows.
  for(int i = 0; detailed && i < 85; i++) {
    Case c;
    string rname;
    c.rules = makeRules(rname);
    int xs = 5 + rnd(9);
    int ys = prob(0.3) ? 5 + rnd(9) : xs;
    c.kind = "random-" + rname;
    setSizes(c, xs, ys);
    c.initialPla = prob(0.12) ? P_WHITE : P_BLACK;
    Pos p(xs, ys, c.rules, c.initialPla);
    int area = xs * ys;
    int nm = prob(0.15) ? 0 : (prob(0.5) ? rnd(area / 2 + 1) : rnd(area * 2 + 1));
    randomMoves(p, xs, ys, nm, prob(0.3) ? 15 : 3, prob(0.15) ? 8 : 0);
    addParams(c, ys, area <= 49);
    c.maxVisits = pickVisits(xs, ys, 400);
    bool ownerOn = prob(0.2);
    finishCase(c, p, pickFlow(), ownerOn, prob(0.06));
  }

  // 2. Territory-scoring (and area) games that end in pass pairs: encore phases 1 and 2, finished games.
  for(int i = 0; detailed && i < 28; i++) {
    Case c;
    string rname;
    c.rules = makeRules(rname);
    if(i % 3 != 2) {
      Rules t = Rules::parseRules(i % 2 ? "japanese" : "korean");
      t.koRule = rnd(3);
      t.komi = c.rules.komi;
      if(prob(0.5)) t.multiStoneSuicideLegal = true;
      c.rules = t;
      rname = i % 2 ? "japanese" : "korean";
    }
    int xs = 5 + rnd(6);
    int ys = prob(0.3) ? 5 + rnd(6) : xs;
    c.kind = "endgame-" + rname;
    setSizes(c, xs, ys);
    c.initialPla = prob(0.15) ? P_WHITE : P_BLACK;
    Pos p(xs, ys, c.rules, c.initialPla);
    int area = xs * ys;
    int rounds = 1 + rnd(3);  // territory: 1 -> encore phase 1, 2 -> phase 2, 3 -> finished
    for(int r = 0; r < rounds; r++) {
      randomMoves(p, xs, ys, rnd(area) + (r == 0 ? area / 3 : 0), 6, 0);
      passPair(p);
    }
    if(prob(0.4)) randomMoves(p, xs, ys, 1 + rnd(4), 20, 0);  // moves (or passes) inside the final phase / past the end
    addParams(c, ys, area <= 49);
    c.maxVisits = pickVisits(xs, ys, 200);
    finishCase(c, p, pickFlow(), prob(0.15), false);
  }

  // 3. Hand-built kos (simple/positional/situational), plain and in encore phases.
  for(int i = 0; detailed && i < 27; i++) {
    Case c;
    string rname;
    c.rules = makeRules(rname);
    if(i % 3 != 0) {
      Rules t = Rules::parseRules(i % 2 ? "japanese" : "korean");
      t.komi = c.rules.komi;
      c.rules = t;
      rname = "territory";
    }
    c.rules.koRule = i % 3;
    int ox = rnd(3), oy = rnd(3);
    int xs = 5 + ox + rnd(3);
    int ys = prob(0.3) ? 5 + oy + rnd(3) : std::max(xs, 5 + oy);
    if(ys < 5 + oy) ys = 5 + oy;
    c.kind = "ko-" + rname;
    setSizes(c, xs, ys);
    c.initialPla = P_BLACK;
    Pos p(xs, ys, c.rules, c.initialPla);
    int stage = rnd(11);
    int encoreRounds = (i % 3 != 0 && prob(0.5)) ? 1 : 0;
    int tail = prob(0.3) ? 1 + rnd(2) : 0;
    koMoves(p, xs, ys, ox, oy, stage, encoreRounds, tail);
    addParams(c, ys, true);
    c.maxVisits = pickVisits(xs, ys, 250);
    finishCase(c, p, pickFlow(), prob(0.15), false);
  }

  // 4. Passy root: rootPruneUselessMoves applies.
  for(int i = 0; detailed && i < 6; i++) {
    Case c;
    string rname;
    c.rules = makeRules(rname);
    int xs = 5 + rnd(5);
    int ys = 5 + rnd(4);
    c.kind = "passy-" + rname;
    setSizes(c, xs, ys);
    c.initialPla = P_BLACK;
    Pos p(xs, ys, c.rules, c.initialPla);
    passyMoves(p, xs, ys);
    addParams(c, ys, true);
    c.params.push_back(make_pair("rootPruneUselessMoves", "1"));
    c.maxVisits = pickVisits(xs, ys, 150);
    finishCase(c, p, pickFlow(), false, false);
  }

  // 5. Symmetric positions: empty boards and a center stone, with root symmetry pruning.
  for(int i = 0; detailed && i < 15; i++) {
    Case c;
    string rname;
    c.rules = makeRules(rname);
    int xs = 5 + rnd(7);
    int ys = prob(0.3) ? 5 + rnd(7) : xs;
    if(i % 3 == 2) { xs |= 1; ys = xs; }
    c.kind = "sym-" + rname;
    setSizes(c, xs, ys);
    c.initialPla = P_BLACK;
    Pos p(xs, ys, c.rules, c.initialPla);
    if(i % 3 == 2) p.play(Location::getLoc(xs / 2, ys / 2, xs), P_BLACK);
    addParams(c, ys, xs * ys <= 49);
    c.params.push_back(make_pair("rootSymmetryPruning", "1"));
    if(prob(0.5)) c.params.push_back(make_pair("useGraphSearch", prob(0.5) ? "1" : "0"));
    c.maxVisits = pickVisits(xs, ys, 150);
    finishCase(c, p, pickFlow(), prob(0.2), false);
  }

  // 6. Big boards with captures.
  for(int i = 0; detailed && i < 10; i++) {
    Case c;
    string rname;
    c.rules = makeRules(rname);
    bool huge = i >= 8;
    int xs = huge ? 19 : 11 + rnd(3);
    int ys = huge ? 19 : (prob(0.3) ? 11 + rnd(3) : xs);
    c.kind = string(huge ? "big19-" : "big-") + rname;
    setSizes(c, xs, ys);
    c.initialPla = P_BLACK;
    Pos p(xs, ys, c.rules, c.initialPla);
    randomMoves(p, xs, ys, (huge ? 120 : 140) + rnd(120), 2, 0);
    addParams(c, ys, false);
    c.maxVisits = huge ? 10 + rnd(30) : 20 + rnd(120);
    finishCase(c, p, i % 2 ? FLOW_REUSE : FLOW_BASIC, prob(0.2), false);
  }

  // 7. 0, 1 and 2 visits, plus beginSearch without any playout.
  for(int i = 0; detailed && i < 18; i++) {
    Case c;
    string rname;
    c.rules = makeRules(rname);
    int xs = 5 + rnd(9);
    int ys = prob(0.3) ? 5 + rnd(9) : xs;
    c.kind = "lowvisits-" + rname;
    setSizes(c, xs, ys);
    c.initialPla = P_BLACK;
    Pos p(xs, ys, c.rules, c.initialPla);
    randomMoves(p, xs, ys, rnd(xs * ys), 3, 0);
    if(prob(0.2)) { passPair(p); }
    addParams(c, ys, xs * ys <= 49);
    c.maxVisits = i % 4 == 3 ? 1 : (i % 4 == 2 ? 2 : (i % 4 == 1 ? 1 : 2));
    finishCase(c, p, (i % 2 == 0) ? FLOW_ZERO : (i % 3 == 0 ? FLOW_REUSE : FLOW_BASIC), prob(0.3), false);
  }

  // 8. Avoided moves at the root: random subsets, all but a few, everything, only pass.
  for(int i = 0; detailed && i < 18; i++) {
    Case c;
    string rname;
    c.rules = makeRules(rname);
    int xs = 5 + rnd(6);
    int ys = prob(0.3) ? 5 + rnd(6) : xs;
    c.kind = "avoid-" + rname;
    setSizes(c, xs, ys);
    c.initialPla = P_BLACK;
    Pos p(xs, ys, c.rules, c.initialPla);
    randomMoves(p, xs, ys, rnd(xs * ys / 2 + 1), 3, 0);
    addParams(c, ys, true);
    c.maxVisits = pickVisits(xs, ys, 120);
    finishCase(c, p, FLOW_AVOID, prob(0.2), false);
  }

  // 9. Stress: many more random games / rules / parameter combinations, higher visit counts, compact reports.
  for(int i = 0; i < (detailed ? 130 : stressOnly); i++) {
    Case c;
    string rname;
    c.rules = makeRules(rname);
    int sel = rnd(10);
    int xs = 5 + rnd(9);
    int ys = prob(0.3) ? 5 + rnd(9) : xs;
    c.initialPla = prob(0.1) ? P_WHITE : P_BLACK;
    if(sel < 6) {
      c.kind = "stress-random-" + rname;
    }
    else if(sel < 8) {
      c.kind = "stress-endgame-" + rname;
      Rules t = Rules::parseRules(prob(0.5) ? "japanese" : "korean");
      t.koRule = rnd(3);
      t.komi = c.rules.komi;
      if(prob(0.3)) t = c.rules;
      c.rules = t;
    }
    else if(sel < 9) {
      c.kind = "stress-ko-" + rname;
      c.rules.koRule = rnd(3);
      c.initialPla = P_BLACK;
      xs = 5 + rnd(5);
      ys = prob(0.3) ? 5 + rnd(5) : xs;
    }
    else {
      c.kind = "stress-sym-" + rname;
      c.initialPla = P_BLACK;
      if(prob(0.5)) { xs = 5 + rnd(6); ys = prob(0.3) ? 5 + rnd(6) : xs; }
      if(prob(0.5)) { xs |= 1; ys = xs; }
    }
    setSizes(c, xs, ys);
    Pos p(xs, ys, c.rules, c.initialPla);
    int area = xs * ys;
    if(sel < 6) randomMoves(p, xs, ys, prob(0.1) ? 0 : rnd(area * 2), prob(0.3) ? 12 : 3, prob(0.1) ? 8 : 0);
    else if(sel < 8) {
      int rounds = 1 + rnd(3);
      for(int r = 0; r < rounds; r++) {
        randomMoves(p, xs, ys, rnd(area) + (r == 0 ? area / 3 : 0), 6, 0);
        passPair(p);
      }
      if(prob(0.3)) randomMoves(p, xs, ys, 1 + rnd(4), 20, 0);
    }
    else if(sel < 9) koMoves(p, xs, ys, 0, 0, rnd(11), (c.rules.scoringRule == Rules::SCORING_TERRITORY && prob(0.5)) ? 1 : 0, prob(0.3) ? 1 : 0);
    else if(xs & 1) { if(prob(0.5)) p.play(Location::getLoc(xs / 2, ys / 2, xs), P_BLACK); }
    addParams(c, ys, area <= 49);
    if(sel >= 9 || prob(0.1)) c.params.push_back(make_pair("rootSymmetryPruning", "1"));
    c.maxVisits = 80 + rnd(area > 100 ? 120 : 320);
    finishCase(c, p, FLOW_BASIC, prob(0.15), false, true);
  }

  // 10. Wide trees (more than 64 children at a node).
  for(int i = 0; detailed && i < 4; i++) {
    Case c;
    string rname;
    c.rules = makeRules(rname);
    static const int sizes[4] = {19, 19, 15, 13};
    static const int visits[4] = {900, 1400, 1000, 1500};
    int xs = sizes[i], ys = sizes[i];
    c.kind = "wide-" + rname;
    c.initialPla = P_BLACK;
    setSizes(c, xs, ys);
    Pos p(xs, ys, c.rules, c.initialPla);
    randomMoves(p, xs, ys, rnd(60), 3, 0);
    addParams(c, ys, false);
    c.maxVisits = visits[i];
    finishCase(c, p, FLOW_BASIC, i == 3, false, true, true);
  }

  // Sections 11-14 were added after the first 341 cases (M3 wave-2a fixes); they only append, so cases 0..340 kept
  // their inputs (the generator's Rand draws for them are unchanged).

  // 11. Tree reuse into nodes with more than 64 children: makeMove copies a child in STATE_EXPANDED2 (children2)
  // into the new root (from the M3 core review's probe/search_nonn_widereuse.cpp). The root player may only play a
  // few moves (avoidMoveUntilByLoc, depth 3), so the visits concentrate on few root children, whose own children
  // (the opponent is unrestricted) spread past 64.
  for(int i = 0; detailed && i < 3; i++) {
    Case c;
    string rname;
    c.rules = makeRules(rname);
    int xs = (i % 3 == 0) ? 19 : ((i % 3 == 1) ? 13 : 11);
    c.kind = "widereuse-" + rname;
    setSizes(c, xs, xs);
    c.initialPla = P_BLACK;
    Pos p(xs, xs, c.rules, c.initialPla);
    randomMoves(p, xs, xs, rnd(30), 3, 0);
    addParams(c, xs, false);
    c.params.push_back(make_pair("cpuctExploration", i % 2 ? "6.0" : "3.0"));
    c.maxVisits = 1200 + rnd(600);
    c.moves = p.moves;
    c.pla = p.nextPla();
    if(c.pla != P_BLACK && c.pla != P_WHITE) c.pla = P_BLACK;
    c.idx = numCases;
    c.sseed = "oracleSearch" + to_string(c.idx);
    c.nnSeed = "oracle" + to_string(c.idx);
    runCase(c, [&](Gen& g) {
      g.run("P");
      string spec = avoidSpec(c, Board(c.xs, c.ys), 1, c.pla);
      g.run(c.pla == P_BLACK ? "V:" + spec + ":-" : "V:-:" + spec);
      g.run("R:" + to_string((int)c.pla));
      g.run("RS");
      for(int k = 0; k < 2; k++) {
        Player rp = g.r->search->getRootPla();
        Loc mv = bestRootChild(g);
        if(mv == Board::NULL_LOC) mv = Board::PASS_LOC;
        g.run("M:" + to_string((int)mv) + ":" + to_string((int)rp) + ":0");
        g.run("RS");
        g.run("R:" + to_string((int)g.r->search->getRootPla()));
        g.run("RS");
        g.run(hsOp(false, c.xs, c.ys));
      }
    });
  }

  // 12. Extreme root ownership (op XO): getEndingWhiteScoreBonus (area and territory branches, button, a fresh ko,
  // encore phase 1) and shouldSuppressPass (fillDameBeforePass, territory scoring, main phase) on owned spots beyond
  // +-0.95, which the NN-less evaluator (tanh of Gaussian * 0.2) practically never produces; half of them with noise
  // pruning forced on and a binding noisePruningCap. Root params that recompute the root output from the NN cache
  // (and would drop the crafted ownership) are forced off.
  for(int i = 0; detailed && i < 24; i++) {
    Case c;
    string rname;
    c.rules = makeRules(rname);
    bool territory = i % 2 == 1;
    if(territory) {
      Rules t = Rules::parseRules(prob(0.5) ? "japanese" : "korean");
      t.komi = c.rules.komi;
      c.rules = t;
      rname = "territory";
    }
    else if(i % 4 == 0) {
      bool button = prob(0.6);
      Rules t = Rules::parseRules(button ? "agabutton" : "chinese");
      t.komi = c.rules.komi;
      c.rules = t;
      rname = button ? "agabutton" : "chinese";
    }
    int xs = 5 + rnd(5);
    int ys = prob(0.3) ? 5 + rnd(5) : xs;
    c.kind = "ownext-" + rname;
    setSizes(c, xs, ys);
    c.initialPla = P_BLACK;
    Pos p(xs, ys, c.rules, c.initialPla);
    int mode = rnd(4);
    if(mode == 0) koMoves(p, xs, ys, 0, 0, 1, 0, 0);  // black just captured: a ko at the root (white to move)
    else randomMoves(p, xs, ys, xs * ys / 3 + rnd(xs * ys / 2), 4, 0);
    if(territory && mode == 3) passPair(p);  // encore phase 1: no pass suppression, territory branch of the bonus
    addParams(c, ys, xs * ys <= 49);
    static const char* bonus[] = {"0.5", "1.0", "2.0", "-0.5"};
    c.params.push_back(make_pair("rootEndingBonusPoints", pick(bonus)));
    if(prob(0.75)) c.params.push_back(make_pair("fillDameBeforePass", "1"));
    if(prob(0.5)) {
      static const char* caps[] = {"0.5", "2.0", "0.05"};
      c.params.push_back(make_pair("useNoisePruning", "1"));
      c.params.push_back(make_pair("noisePruningCap", pick(caps)));
      if(prob(0.5)) c.params.push_back(make_pair("noisePruneUtilityScale", prob(0.5) ? "0.05" : "0.15"));
    }
    c.params.push_back(make_pair("rootNumSymmetriesToSample", "1"));
    c.params.push_back(make_pair("policyOptimism", "0"));
    c.params.push_back(make_pair("rootPolicyOptimism", "0"));
    c.params.push_back(make_pair("ignorePreRootHistory", "0"));
    c.params.push_back(make_pair("conservativePass", "0"));
    c.maxVisits = 30 + rnd(150);
    bool ownerOn = prob(0.4);
    c.moves = p.moves;
    c.pla = p.nextPla();
    if(c.pla != P_BLACK && c.pla != P_WHITE) c.pla = P_BLACK;
    c.idx = numCases;
    c.sseed = "oracleSearch" + to_string(c.idx);
    c.nnSeed = "oracle" + to_string(c.idx);
    runCase(c, [&](Gen& g) {
      string rp = to_string((int)c.pla);
      g.run("P");
      if(ownerOn) g.run("O");
      g.run("R:" + rp);
      g.run("XO");
      g.run("PM");
      g.run("PSV");
      g.run("AD:" + to_string(rnd(3)) + ":1:0");
      g.run("CM:2");
      g.run("SPN:maxVisits=" + to_string((long long)(c.maxVisits * (2 + rnd(2)))));
      g.run("R:" + rp);
      report(g, true, c.maxVisits <= 30, ownerOn, c.xs, c.ys);
      if(prob(0.5)) {
        Loc mv = bestRootChild(g);
        if(mv == Board::NULL_LOC) mv = Board::PASS_LOC;
        g.run("M:" + to_string((int)mv) + ":" + rp + ":0");
        g.run("R:" + to_string((int)g.r->search->getRootPla()));
        g.run("XO");
        g.run("PM");
        g.run(hsOp(ownerOn, c.xs, c.ys));
      }
    });
  }

  // 13. Root children filtered out of a reused tree, then grown again: rootSymmetryPruning switched on by
  // setParamsNoClearing on a symmetric root that already has more than 8 children (beginSearch's filtering and
  // collapseChildrenCapacity), then off again with more visits (the root regrows past 8 and 64 children).
  for(int i = 0; detailed && i < 6; i++) {
    Case c;
    string rname;
    c.rules = makeRules(rname);
    int xs = 5 + 2 * rnd(3);
    c.kind = "collapse-" + rname;
    setSizes(c, xs, xs);
    c.initialPla = P_BLACK;
    Pos p(xs, xs, c.rules, c.initialPla);
    Loc center = Location::getLoc(xs / 2, xs / 2, xs);
    if(i % 3 == 1) p.play(center, P_BLACK);
    bool viaMove = i % 3 == 2;
    addParams(c, xs, xs * xs <= 49);
    c.params.push_back(make_pair("rootSymmetryPruning", "0"));
    c.maxVisits = 150 + rnd(200);
    c.moves = p.moves;
    c.pla = p.nextPla();
    if(c.pla != P_BLACK && c.pla != P_WHITE) c.pla = P_BLACK;
    c.idx = numCases;
    c.sseed = "oracleSearch" + to_string(c.idx);
    c.nnSeed = "oracle" + to_string(c.idx);
    runCase(c, [&](Gen& g) {
      g.run("P");
      g.run("R:" + to_string((int)c.pla));
      g.run("RS");
      if(viaMove) {
        g.run("M:" + to_string((int)center) + ":" + to_string((int)c.pla) + ":0");
        g.run("RS");
      }
      string rp = to_string((int)g.r->search->getRootPla());
      g.run("SPN:rootSymmetryPruning=1");
      g.run("R:" + rp);
      g.run("RS");
      g.run(hsOp(false, c.xs, c.ys));
      g.run("SPN:rootSymmetryPruning=0,maxVisits=" + to_string((long long)(c.maxVisits * 4)));
      g.run("R:" + rp);
      g.run("RS");
      g.run(hsOp(false, c.xs, c.ys));
    });
  }

  // 14. Time carried over by tree reuse (op ECT): makeMove scales effectiveSearchTimeCarriedOver by the new root's
  // share of the visits and treeReuseCarryOverTimeFactor (0 without reuse); clearSearch and friends reset it. ECT
  // sets it before every printed makeMove: after a search it holds real elapsed time, which is never printed.
  for(int i = 0; detailed && i < 8; i++) {
    Case c;
    string rname;
    c.rules = makeRules(rname);
    int xs = 7 + rnd(5);
    int ys = prob(0.3) ? 7 + rnd(5) : xs;
    c.kind = "timecarry-" + rname;
    setSizes(c, xs, ys);
    c.initialPla = P_BLACK;
    Pos p(xs, ys, c.rules, c.initialPla);
    randomMoves(p, xs, ys, rnd(xs * ys / 2), 3, 0);
    addParams(c, ys, xs * ys <= 49);
    static const char* factors[] = {"1.0", "0.5", "0.75", "0.3"};
    c.params.push_back(make_pair("treeReuseCarryOverTimeFactor", pick(factors)));
    c.maxVisits = 40 + rnd(120);
    c.moves = p.moves;
    c.pla = p.nextPla();
    if(c.pla != P_BLACK && c.pla != P_WHITE) c.pla = P_BLACK;
    c.idx = numCases;
    c.sseed = "oracleSearch" + to_string(c.idx);
    c.nnSeed = "oracle" + to_string(c.idx);
    runCase(c, [&](Gen& g) {
      g.run("P");
      g.run("R:" + to_string((int)c.pla));
      int rounds = 2 + rnd(2);
      for(int k = 0; k < rounds; k++) {
        char buf[32];
        snprintf(buf, sizeof buf, "%016llX", DB(0.25 + rnd(4000) / 100.0));
        g.run(string("ECT:") + buf);
        Player mp = g.r->search->getRootPla();
        int kind = rnd(10);
        if(kind < 6) {
          Loc mv = kind < 4 ? bestRootChild(g) : randomLegalRootMove(g, mp);
          if(mv == Board::NULL_LOC) mv = Board::PASS_LOC;
          g.run("M:" + to_string((int)mv) + ":" + to_string((int)mp) + ":0");
        }
        else if(kind == 6) g.run("M:" + to_string((int)Board::PASS_LOC) + ":" + to_string((int)mp) + ":0");
        else if(kind == 7) g.run("CL");
        else if(kind == 8) g.run("PCH:" + to_string((int)getOpp(mp)));
        else g.run("SP:maxVisits=" + to_string((long long)c.maxVisits) + ",cpuctExploration=1.4");
        g.run("ECT:-");
        g.run("R:" + to_string((int)g.r->search->getRootPla()));
        if(prob(0.5)) g.run("RS");
      }
    });
  }

  fprintf(stderr, "cases: %d\n", numCases);
  for(const auto& kv : featureCounts) fprintf(stderr, "  %-45s %d\n", kv.first.c_str(), kv.second);
  ScoreValue::freeTables();
  return 0;
}
