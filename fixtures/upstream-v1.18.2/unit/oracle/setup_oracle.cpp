// Oracle for crates/katago-core/src/program/setup.rs: the unmodified upstream cpp/program/setup.cpp driven on
// config cases (fixtures/.../oracle/setup_oracle_cases.txt, made by setup_oracle_gen.py), dumping everything
// it produces: loadParams / loadSingleParams (every setup_for_t x hasHumanModel x single), loadSingleRules,
// parseReportAnalysisWinrates, loadDefaultBoardXYSize, getMutexKeySets, getBackendPrefixes, and for the cases
// flagged nn=1 initializeNNEvaluators (debugSkipNeuralNet via "/dev/null": the evaluators' settings, the
// seed stream, the log lines setup.cpp writes) plus, after every op, the unused keys. Not upstream code.
// The Rust mirror is crates/katago-core/tests/setup_oracle.rs and must print the same bytes.
//
// setup.cpp is compiled here WITHOUT USE_EIGEN_BACKEND, so the `#ifndef USE_EIGEN_BACKEND` branches (the
// ones the port has) run; backendPrefix is "dummybackend", or "opencl" with -DUSE_OPENCL_BACKEND (the NCHW
// backends differ in inputsUseNHWC's default and in which keys are read).
// Build (from the repo root, after `just katago-eigen`; links the Eigen build's objects of the unmodified
// upstream sources, library dirs only, except program/setup.cpp.o which is recompiled; output outside the repo):
//   O=build/eigen/CMakeFiles/katago.dir; UP=upstream/KataGo/cpp
//   OBJS=$(find $O/core $O/game $O/neuralnet $O/search $O/dataio $O/program -name '*.o' | grep -v 'program/setup.cpp.o')
//   g++ -std=c++17 -O2 -DNDEBUG -I$UP -o /tmp/setup_oracle_dummy \
//     fixtures/upstream-v1.18.2/unit/oracle/setup_oracle.cpp $UP/program/setup.cpp $OBJS -lz -lzip -lpthread
//   g++ -std=c++17 -O2 -DNDEBUG -DUSE_OPENCL_BACKEND -I$UP -o /tmp/setup_oracle_opencl \
//     fixtures/upstream-v1.18.2/unit/oracle/setup_oracle.cpp $UP/program/setup.cpp $OBJS -lz -lzip -lpthread
//   /tmp/setup_oracle_dummy  fixtures/upstream-v1.18.2/unit/oracle/setup_oracle_cases.txt > fixtures/upstream-v1.18.2/unit/oracle/setup_oracle_dummybackend.txt
//   /tmp/setup_oracle_opencl fixtures/upstream-v1.18.2/unit/oracle/setup_oracle_cases.txt nn > fixtures/upstream-v1.18.2/unit/oracle/setup_oracle_opencl.txt
// (the second run is "nn" mode: only the initializeNNEvaluators ops, the only ones that depend on the backend)
//
// Output (one record per line; doubles are IEEE bit patterns in uppercase hex):
//   MUTEX {..} {..}...   getMutexKeySets
//   PREFIXES a b c       getBackendPrefixes (the port appends "burn", katago-nn-burn's backend; the test drops it)
//   BASE / F.. / ENDBASE every field of a default-constructed SearchParams
//   C <n> <name>         a case (cases flagged nn=2 only run the N ops). Then, per op: (R loadSingleRules, W parseReportAnalysisWinrates, X loadDefaultBoardXYSize,
//                        N initializeNNEvaluators, then P loadParams... last: the pending cases drop them), a header line, the result
//                        (NB/B/F, RULES, PLA, XY, E.../SEEDNEXT/L, or ERR <type> <what>) and the U line.
//   NR+ <lines> / NR <n> the result of an N op (E.../ERR, SEEDNEXT and L lines; one "NR+" line, then the lines, up to
//                        the U line): the n-th distinct block of the case is printed the first time, later as its number
//   F <field> <bits>     only the fields that differ from the previous dump of the case (BASE for the first)
//   U+ <diff> / U <n>    the unused keys after the op: the n-th distinct list of the case (0-based) is printed the
//                        first time as "U+" and a diff to the previous distinct list (" -key"s that left, then
//                        " +key"s that joined), later as its number
//
#include <cstdarg>
#include <cstdio>
#include <algorithm>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <typeinfo>
#include "core/global.h"
#include "core/config_parser.h"
#include "core/logger.h"
#include "core/rand.h"
#include "neuralnet/nneval.h"
#include "program/setup.h"
#include "search/searchparams.h"
using namespace std;

static unsigned long long dbits(double d) { uint64_t u; memcpy(&u, &d, 8); return (unsigned long long)u; }
static unsigned fbits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }

template<typename Tag, typename Tag::type M> struct Rob { friend typename Tag::type get(Tag) { return M; } };
#define ROB(NAME, CLS, TYPE, MEMBER) \
  struct NAME { typedef TYPE CLS::* type; friend type get(NAME); }; \
  template struct Rob<NAME, &CLS::MEMBER>;
ROB(R_requireExact, NNEvaluator, const bool, requireExactNNLen)
ROB(R_inputsNHWC, NNEvaluator, const bool, inputsUseNHWC)
ROB(R_randSeed, NNEvaluator, const std::string, randSeed)
ROB(R_skip, NNEvaluator, const bool, debugSkipNeuralNet)
ROB(R_cache, NNEvaluator, NNCacheTable*, nnCacheTable)
ROB(R_gpuIdx, NNEvaluator, std::vector<int>, gpuIdxByServerThread)
ROB(R_numThreads, NNEvaluator, int, numThreads)
ROB(R_tableSize, NNCacheTable, uint64_t, tableSize)

static string errLine(const std::exception& e) {
  const char* t = "Exception";
  if(dynamic_cast<const ConfigParsingError*>(&e)) t = "ConfigParsingError";
  else if(dynamic_cast<const IOError*>(&e)) t = "IOError";
  else if(dynamic_cast<const ValueError*>(&e)) t = "ValueError";
  else if(dynamic_cast<const CommandError*>(&e)) t = "CommandError";
  else if(dynamic_cast<const StringError*>(&e)) t = "StringError";
  return string("ERR ") + t + " " + e.what();
}

static void paramLines(const SearchParams& p, vector<string>& out) {
  char buf[256];
#define D(f) { snprintf(buf, sizeof buf, "F " #f " %016llX", dbits(p.f)); out.push_back(buf); }
#define I(f) { snprintf(buf, sizeof buf, "F " #f " %lld", (long long)p.f); out.push_back(buf); }
  D(winLossUtilityFactor) D(staticScoreUtilityFactor) D(dynamicScoreUtilityFactor) D(dynamicScoreCenterZeroWeight)
  D(dynamicScoreCenterScale) D(noResultUtilityForWhite) D(drawEquivalentWinsForWhite) D(cpuctExploration)
  D(cpuctExplorationLog) D(cpuctExplorationBase) D(cpuctUtilityStdevPrior) D(cpuctUtilityStdevPriorWeight)
  D(cpuctUtilityStdevScale) D(fpuReductionMax) D(fpuLossProp) I(fpuParentWeightByVisitedPolicy)
  D(fpuParentWeightByVisitedPolicyPow) D(fpuParentWeight) D(policyOptimism) D(valueWeightExponent) I(useNoisePruning)
  D(noisePruneUtilityScale) D(noisePruningCap) I(useUncertainty) D(uncertaintyCoeff) D(uncertaintyExponent)
  D(uncertaintyMaxWeight) I(useGraphSearch) I(graphSearchRepBound) D(graphSearchCatchUpLeakProb) I(rootNoiseEnabled)
  D(rootDirichletNoiseTotalConcentration) D(rootDirichletNoiseWeight) D(rootPolicyTemperature) D(rootPolicyTemperatureEarly)
  D(rootFpuReductionMax) D(rootFpuLossProp) I(rootNumSymmetriesToSample) I(rootSymmetryPruning)
  D(rootDesiredPerChildVisitsCoeff) D(rootPolicyOptimism) D(chosenMoveTemperature) D(chosenMoveTemperatureEarly)
  D(chosenMoveTemperatureHalflife) D(chosenMoveTemperatureOnlyBelowProb) D(chosenMoveSubtract) D(chosenMovePrune)
  I(useLcbForSelection) D(lcbStdevs) D(minVisitPropForLCB) I(useNonBuggyLcb) D(rootEndingBonusPoints)
  I(rootPruneUselessMoves) I(conservativePass) I(fillDameBeforePass) I(avoidMYTDaggerHackPla) D(wideRootNoise)
  I(enablePassingHacks) I(enableMorePassingHacks) D(playoutDoublingAdvantage) I(playoutDoublingAdvantagePla)
  D(avoidRepeatedPatternUtility)
  snprintf(buf, sizeof buf, "F nnPolicyTemperature %08X", fbits(p.nnPolicyTemperature)); out.push_back(buf);
  I(antiMirror) I(ignorePreRootHistory) I(ignoreAllHistory) D(subtreeValueBiasFactor) I(subtreeValueBiasTableNumShards)
  D(subtreeValueBiasFreeProp) D(subtreeValueBiasWeightExponent) I(useEvalCache) I(evalCacheMinVisits)
  I(nodeTableShardsPowerOfTwo) D(numVirtualLossesPerThread) I(numThreads) D(minPlayoutsPerThread) I(maxVisits)
  I(maxPlayouts) D(maxTime) I(maxVisitsPondering) I(maxPlayoutsPondering) D(maxTimePondering) D(lagBuffer)
  D(searchFactorAfterOnePass) D(searchFactorAfterTwoPass) D(treeReuseCarryOverTimeFactor) D(overallocateTimeFactor)
  D(midgameTimeFactor) D(midgameTurnPeakTime) D(endgameTurnTimeDecay) D(obviousMovesTimeFactor)
  D(obviousMovesPolicyEntropyTolerance) D(obviousMovesPolicySurpriseTolerance) D(futileVisitsThreshold)
  snprintf(buf, sizeof buf, "F humanSLProfile %d", (int)p.humanSLProfile.initialized); out.push_back(buf);
  D(humanSLCpuctExploration) D(humanSLCpuctPermanent) D(humanSLRootExploreProbWeightless) D(humanSLRootExploreProbWeightful)
  D(humanSLPlaExploreProbWeightless) D(humanSLPlaExploreProbWeightful) D(humanSLOppExploreProbWeightless)
  D(humanSLOppExploreProbWeightful) D(humanSLChosenMoveProp) I(humanSLChosenMoveIgnorePass) D(humanSLChosenMovePiklLambda)
#undef D
#undef I
}

static vector<string> baseLines;
static vector<string> prevLines;

//Only the fields that differ from the previous dump of the case (from BASE, the default-constructed
//SearchParams, for the first) are printed.
static void dumpParams(const SearchParams& p) {
  vector<string> lines;
  paramLines(p, lines);
  for(size_t i = 0; i < lines.size(); i++)
    if(lines[i] != prevLines[i])
      printf("%s\n", lines[i].c_str());
  prevLines = lines;
}

//The unused keys. The n-th distinct list of a case is printed the first time it appears ("U+") as the difference
//to the previous distinct list of the case (the empty list for the first): " -key" for each key that left it
//(in the previous list's order), then " +key" for each that joined (in the list's order); later appearances as
//the list's number ("U n").
static vector<vector<string>> seenUnused;
static void unused(const ConfigParser& cfg) {
  vector<string> u = cfg.unusedKeys();
  for(size_t i = 0; i < seenUnused.size(); i++) {
    if(seenUnused[i] == u) { printf("U %d\n", (int)i); return; }
  }
  vector<string> prev = seenUnused.empty() ? vector<string>() : seenUnused.back();
  printf("U+");
  for(const string& k: prev)
    if(std::find(u.begin(), u.end(), k) == u.end()) printf(" -%s", k.c_str());
  for(const string& k: u)
    if(std::find(prev.begin(), prev.end(), k) == prev.end()) printf(" +%s", k.c_str());
  printf("\n");
  seenUnused.push_back(u);
}

//Appends printf-style text to s.
static void bprintf(string& s, const char* fmt, ...) {
  char buf[4096];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(buf, sizeof buf, fmt, ap);
  va_end(ap);
  s += buf;
}
//The results of an initializeNNEvaluators op (E/ERR, SEEDNEXT and L lines) are mostly the same for many ops of a
//case: the n-th distinct result block of the case is printed the first time ("NR+" and its lines), later as "NR n".
static vector<string> seenN;

int main(int argc, const char* argv[]) {
  Board::initHash();
  if(argc < 2) { fprintf(stderr, "usage: setup_oracle cases.txt\n"); return 1; }
  ifstream in(argv[1]);
  string line;
  bool nnOnly = argc >= 3 && string(argv[2]) == "nn";
  paramLines(SearchParams(), baseLines);
  {
    auto m = Setup::getMutexKeySets();
    printf("MUTEX");
    for(auto& pr: m) { printf(" {"); for(auto& s: pr.first) printf(" %s", s.c_str()); printf(" } {"); for(auto& s: pr.second) printf(" %s", s.c_str()); printf(" }"); }
    printf("\n");
    printf("PREFIXES");
    for(auto& s: Setup::getBackendPrefixes()) printf(" %s", s.c_str());
    printf("\n");
    printf("BASE\n");
    for(const string& l: baseLines) printf("%s\n", l.c_str());
    printf("ENDBASE\n");
  }
  while(getline(in, line)) {
    if(line.rfind("CASE ", 0) != 0) continue;
    istringstream hs(line);
    string tag; int n; int nn; string name;
    hs >> tag >> n >> nn >> name;
    string text;
    while(getline(in, line) && line != "END") text += line + "\n";
    printf("C %d %s\n", n, name.c_str());
    seenUnused.clear();
    seenN.clear();
    prevLines = baseLines;
    auto mk = [&](ConfigParser*& cfg) -> bool {
      try { istringstream is(text); cfg = new ConfigParser(is); return true; }
      catch(const std::exception& e) { printf("PARSE %s\n", errLine(e).c_str()); return false; }
    };
    if(!nnOnly && nn != 2) for(int lk = 0; lk < 2; lk++) {
      ConfigParser* cfg = NULL;
      if(!mk(cfg)) continue;
      printf("R %d\n", lk);
      try {
        Rules r = Setup::loadSingleRules(*cfg, (bool)lk);
        printf("RULES %d %d %d %d %d %d %d %08X %s\n", r.koRule, r.scoringRule, r.taxRule, (int)r.multiStoneSuicideLegal, (int)r.hasButton,
               r.whiteHandicapBonusRule, (int)r.friendlyPassOk, fbits(r.komi), r.toString().c_str());
      } catch(const std::exception& e) { printf("%s\n", errLine(e).c_str()); }
      unused(*cfg);
      delete cfg;
    }
    if(!nnOnly && nn != 2) for(int dp = 0; dp < 3; dp++) {
      ConfigParser* cfg = NULL;
      if(!mk(cfg)) continue;
      Player def = dp == 0 ? P_BLACK : dp == 1 ? P_WHITE : C_EMPTY;
      printf("W %d\n", dp);
      try { printf("PLA %d\n", (int)Setup::parseReportAnalysisWinrates(*cfg, def)); }
      catch(const std::exception& e) { printf("%s\n", errLine(e).c_str()); }
      unused(*cfg);
      delete cfg;
    }
    if(!nnOnly && nn != 2) {
      ConfigParser* cfg = NULL;
      if(mk(cfg)) {
        printf("X\n");
        Logger logger(nullptr, false, false, false, false);
        try {
          int x = -7, y = -7;
          bool b = Setup::loadDefaultBoardXYSize(*cfg, logger, x, y);
          printf("XY %d %d %d\n", (int)b, x, y);
        } catch(const std::exception& e) { printf("%s\n", errLine(e).c_str()); }
        unused(*cfg);
        delete cfg;
      }
    }
    if(nn) {
      for(int sf = 0; sf < 6; sf++) for(int variant = 0; variant < 4; variant++) {
        ConfigParser* cfg = NULL;
        if(!mk(cfg)) continue;
        // v1.18.2: the int defaultMaxBatchSize (-1 / 16) became a MaxBatchSizeRequest; 16 -> explicitSize(16), -1 -> fromConcurrency()
        Setup::MaxBatchSizeRequest defaultMaxBatchSize = (variant & 1) ? Setup::MaxBatchSizeRequest::explicitSize(16) : Setup::MaxBatchSizeRequest::fromConcurrency();
        bool disableFP16 = (variant & 2) != 0;
        int nModels = (variant == 3) ? 1 : 2;
        printf("N %d %d\n", sf, variant);
        string blk;
        Logger logger(nullptr, false, false, false, false);
        ostringstream logOut;
        logger.addOStream(logOut, false);
        Rand seedRand(string("setuporacle") + Global::intToString(n));
        vector<string> names, files, shas;
        for(int i = 0; i < nModels; i++) { names.push_back("model" + Global::intToString(i)); files.push_back("/dev/null"); }
        if(variant == 2) for(int i = 0; i < nModels; i++) shas.push_back("");
        try {
          vector<NNEvaluator*> evals = Setup::initializeNNEvaluators(
            names, files, shas, *cfg, logger, seedRand, 4, 19, 19, defaultMaxBatchSize, false, disableFP16, (Setup::setup_for_t)sf);
          for(size_t i = 0; i < evals.size(); i++) {
            NNEvaluator* e = evals[i];
            NNCacheTable* c = e->*get(R_cache());
            bprintf(blk, "E %d name=%s file=%s maxBatch=%d curBatch=%d x=%d y=%d exact=%d nhwc=%d fp16=%s threads=%d",
                   (int)i, e->getModelName().c_str(), e->getModelFileName().c_str(), e->getMaxBatchSize(), e->getMaxRowsToSendPerBatch(),
                   e->getNNXLen(), e->getNNYLen(), (int)(e->*get(R_requireExact())), (int)(e->*get(R_inputsNHWC())),
                   e->getUsingFP16Mode().toString().c_str(), e->*get(R_numThreads()));
            bprintf(blk, " gpus=");
            for(int g: e->*get(R_gpuIdx())) bprintf(blk, "%d,", g);
            bprintf(blk, " seed=%s skip=%d rand=%d sym=%d cache=%lld internal=%s ver=%d\n",
                   (e->*get(R_randSeed())).c_str(), (int)(e->*get(R_skip())), (int)e->getDoRandomize(), e->getDefaultSymmetry(),
                   c == NULL ? -1LL : (long long)(c->*get(R_tableSize())),
                   e->getInternalModelName().c_str(), e->getModelVersion());
          }
          for(NNEvaluator* e: evals) delete e;
        } catch(const std::exception& e) { bprintf(blk, "%s\n", errLine(e).c_str()); }
        bprintf(blk, "SEEDNEXT %llu\n", (unsigned long long)seedRand.nextUInt64());
        {
          istringstream ls(logOut.str());
          string l;
          while(getline(ls, l)) {
            // only the lines setup.cpp itself writes
            size_t a = l.find("nnRandSeed"); size_t b = l.find("After dedups");
            size_t pos = std::min(a, b);
            if(pos != string::npos) bprintf(blk, "L %s\n", l.substr(pos).c_str());
          }
        }
        {
          size_t k = 0;
          while(k < seenN.size() && seenN[k] != blk) k++;
          if(k < seenN.size()) printf("NR %d\n", (int)k);
          else { seenN.push_back(blk); printf("NR+\n%s", blk.c_str()); }
        }
        unused(*cfg);
        delete cfg;
      }
    }
    if(!nnOnly && nn != 2) for(int sf = 0; sf < 6; sf++) for(int hh = 0; hh < 2; hh++) for(int single = 0; single < 2; single++) {
      ConfigParser* cfg = NULL;
      if(!mk(cfg)) continue;
      printf("P %d %d %d\n", sf, hh, single);
      try {
        vector<SearchParams> ps;
        if(single) { ps.push_back(Setup::loadSingleParams(*cfg, (Setup::setup_for_t)sf, (bool)hh)); }
        else ps = Setup::loadParams(*cfg, (Setup::setup_for_t)sf, (bool)hh);
        printf("NB %d\n", (int)ps.size());
        for(size_t i = 0; i < ps.size(); i++) { printf("B %d\n", (int)i); dumpParams(ps[i]); }
      } catch(const std::exception& e) { printf("%s\n", errLine(e).c_str()); }
      unused(*cfg);
      delete cfg;
    }
    fflush(stdout);
  }
  return 0;
}
