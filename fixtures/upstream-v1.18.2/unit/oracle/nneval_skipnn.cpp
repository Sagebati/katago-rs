// Oracle for crates/katago-core/src/neuralnet/nneval.rs: the unmodified upstream NNEvaluator with
// debugSkipNeuralNet = true (its server thread draws Gaussian "logits" from its Rand), so evaluate()'s
// post-processing (policy temperature, passing hacks, MYT dagger ban, legality masking, value
// softmax, softplus/sqrt score fields, tanhf ownership), the NN cache (hits, the
// hadResultWithoutOwnerMap path) and averageMultipleSymmetries are checked bit for bit without a net.
// Not upstream code. The Rust mirror is crates/katago-core/tests/nneval_skipnn.rs: it replays the
// op stream below through the ported NNEvaluator and must print the same R/D lines.
// Build (from the repo root, after `just katago-eigen`; links the Eigen build's objects of the
// unmodified upstream sources (library dirs only); output outside the repo):
//   O=build/eigen/CMakeFiles/katago.dir; g++ -std=c++17 -O2 -Iupstream/KataGo/cpp -o /tmp/nneval_skipnn \
//     fixtures/upstream-v1.18.2/unit/oracle/nneval_skipnn.cpp \
//     $(find $O/core $O/game $O/neuralnet $O/search $O/dataio $O/program -name '*.o') -lz -lzip -lpthread
//   /tmp/nneval_skipnn > fixtures/upstream-v1.18.2/unit/oracle/nneval_skipnn.txt
// Output, one op per line (floats are IEEE bit patterns in uppercase hex):
//   E <nnXLen> <nnYLen> <nnCacheSizePowerOfTwo> <randSeed> <doRandomize>   new evaluator (old one killed)
//   P <xSize> <ySize> <ko> <score> <tax> <suicide> <button> <whb> <friendlyPassOk> <komi f32> <initialPla>
//     <numMoves> {<loc> <pla>}*   position: Board(x,y); BoardHistory(board,initialPla,rules,0);
//     makeBoardMoveAssumeLegal(board,loc,pla,NULL) for each move; nextPla = hist.presumedNextMovePla
//   V <drawEquiv f64> <conservativePass> <passingHacks> <playoutDoubling f64> <policyTemp f32> <dagger>
//     <symmetry> <policyOptimism f64> <maxHistory> <skipCache> <includeOwnerMap>     evaluate()
//   W <same 11 fields as V>   a re-evaluate() of the same position right after a V, from the cache
//     (with includeOwnerMap flipped: false->true takes the hadResultWithoutOwnerMap path); no D line
//   A <same 11 fields as V> <numSymmetriesToSample> <randSeed>   averageMultipleSymmetries(..., Rand(seed), n)
//   R <hash0> <hash1> <whiteWin> <whiteLoss> <noResult> <scoreMean> <scoreMeanSq> <lead> <varTimeLeft>
//     <stWinloss> <stScore> <optimismUsed> <nnXLen> <nnYLen> <policyFnv> <ownerFlag> <ownerFnv>
//     result of the preceding V/W/A; *Fnv = FNV-1a 64 over the f32 bit patterns (policy over
//     MAX_NN_POLICY_SIZE, owner map over nnXLen*nnYLen)
//   D <policy[0..policySize]...> | <owner...>   full arrays for the first evals of each evaluator
//   S <numRowsProcessed> <numBatchesProcessed>   stats before each evaluator is killed
//   K   killServerThreads(); spawnServerThreads(); on the same evaluator (as testsearchv8.cpp does):
//     the new server thread's Rand is "<randSeed>:NNEvalServerThread:<k>", k = threads ever spawned
// Generated with g++ (GCC) 16.2.1 / glibc 2.44 on x86_64 (FMA) Linux.
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include "core/global.h"
#include "core/hash.h"
#include "core/rand.h"
#include "game/board.h"
#include "game/rules.h"
#include "game/boardhistory.h"
#include "neuralnet/nninputs.h"
#include "neuralnet/nneval.h"
using namespace std;

static uint64_t S;
static uint64_t nxt() { uint64_t z = (S += 0x9E3779B97F4A7C15ULL); z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL; z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL; return z ^ (z >> 31); }
static int rnd(int n) { return (int)(nxt() % (uint64_t)n); }

static uint32_t fbits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static uint64_t dbits(double d) { uint64_t u; memcpy(&u, &d, 8); return u; }
static uint64_t fnvFloats(const float* a, int n) {
  uint64_t h = 0xcbf29ce484222325ULL;
  for(int i = 0; i < n; i++) {
    uint32_t u = fbits(a[i]);
    for(int b = 0; b < 4; b++) { h ^= (u >> (8 * b)) & 0xFF; h *= 0x100000001b3ULL; }
  }
  return h;
}

struct Pos {
  Board board;
  BoardHistory hist;
  Player nextPla;
};

static Pos makeGame(int xs, int ys, int maxMoves, string& line) {
  Rules rules;
  rules.koRule = rnd(4);
  rules.scoringRule = rnd(2);
  rules.taxRule = rnd(3);
  rules.multiStoneSuicideLegal = rnd(2) == 0;
  rules.hasButton = rules.scoringRule == Rules::SCORING_AREA && rnd(4) == 0;
  rules.whiteHandicapBonusRule = rnd(3);
  rules.friendlyPassOk = rnd(2) == 0;
  static const float komis[] = {7.5f, 6.5f, 0.0f, -3.5f, 0.5f, 5.0f, 1.0f, -12.0f};
  rules.komi = komis[rnd(8)];
  Player initialPla = rnd(5) == 0 ? P_WHITE : P_BLACK;
  Board board(xs, ys);
  BoardHistory hist(board, initialPla, rules, 0, BoardHistoryModes());
  vector<pair<Loc,Player>> moves;
  int numMoves = rnd(maxMoves + 1);
  for(int m = 0; m < numMoves && !hist.isGameFinished; m++) {
    Player pla = hist.presumedNextMovePla;
    Loc loc = Board::PASS_LOC;
    int passRoll = rnd(100);
    if(passRoll >= (m > numMoves - 6 ? 50 : 4)) {
      for(int tries = 0; tries < 40; tries++) {
        int x = rnd(xs);
        int y = rnd(ys);
        Loc l = Location::getLoc(x, y, xs);
        if(hist.isLegal(board, l, pla)) { loc = l; break; }
      }
    }
    if(!hist.isLegal(board, loc, pla))
      continue;
    hist.makeBoardMoveAssumeLegal(board, loc, pla, NULL);
    moves.push_back(make_pair(loc, pla));
  }
  char buf[256];
  snprintf(buf, sizeof(buf), "P %d %d %d %d %d %d %d %d %d %08X %d %d",
           xs, ys, rules.koRule, rules.scoringRule, rules.taxRule, (int)rules.multiStoneSuicideLegal,
           (int)rules.hasButton, rules.whiteHandicapBonusRule, (int)rules.friendlyPassOk, fbits(rules.komi),
           (int)initialPla, (int)moves.size());
  line = buf;
  for(auto& mv : moves) { snprintf(buf, sizeof(buf), " %d %d", (int)mv.first, (int)mv.second); line += buf; }
  return Pos{board, hist, hist.presumedNextMovePla};
}

// Dagger pattern of nneval.cpp in the top-left corner of a 19x19, black (pla) to move.
static Pos makeDagger(string& line) {
  Rules rules = Rules::getTrompTaylorish();
  rules.komi = 7.5f;
  Board board(19, 19);
  BoardHistory hist(board, P_BLACK, rules, 0, BoardHistoryModes());
  int xy[][2] = {{3,2},{2,2},{3,3},{2,3},{2,5},{1,5},{15,15},{16,3}};
  vector<pair<Loc,Player>> moves;
  for(int i = 0; i < 8; i++) {
    Loc loc = Location::getLoc(xy[i][0], xy[i][1], 19);
    Player pla = hist.presumedNextMovePla;
    testAssert(hist.isLegal(board, loc, pla));
    hist.makeBoardMoveAssumeLegal(board, loc, pla, NULL);
    moves.push_back(make_pair(loc, pla));
  }
  char buf[256];
  snprintf(buf, sizeof(buf), "P 19 19 %d %d %d %d %d %d %d %08X %d %d",
           rules.koRule, rules.scoringRule, rules.taxRule, (int)rules.multiStoneSuicideLegal, (int)rules.hasButton,
           rules.whiteHandicapBonusRule, (int)rules.friendlyPassOk, fbits(rules.komi), (int)P_BLACK, (int)moves.size());
  line = buf;
  for(auto& mv : moves) { snprintf(buf, sizeof(buf), " %d %d", (int)mv.first, (int)mv.second); line += buf; }
  return Pos{board, hist, hist.presumedNextMovePla};
}

static void printResult(const NNOutput& o, int policySize, bool full) {
  printf("R %016llX %016llX %08X %08X %08X %08X %08X %08X %08X %08X %08X %08X %d %d %016llX %d %016llX\n",
         (unsigned long long)o.nnHash.hash0, (unsigned long long)o.nnHash.hash1,
         fbits(o.whiteWinProb), fbits(o.whiteLossProb), fbits(o.whiteNoResultProb), fbits(o.whiteScoreMean),
         fbits(o.whiteScoreMeanSq), fbits(o.whiteLead), fbits(o.varTimeLeft), fbits(o.shorttermWinlossError),
         fbits(o.shorttermScoreError), fbits(o.policyOptimismUsed), o.nnXLen, o.nnYLen,
         (unsigned long long)fnvFloats(o.policyProbs, NNPos::MAX_NN_POLICY_SIZE),
         o.whiteOwnerMap != NULL ? 1 : 0,
         (unsigned long long)(o.whiteOwnerMap != NULL ? fnvFloats(o.whiteOwnerMap, o.nnXLen * o.nnYLen) : 0));
  if(full) {
    printf("D");
    for(int i = 0; i < policySize; i++) printf(" %08X", fbits(o.policyProbs[i]));
    printf(" |");
    if(o.whiteOwnerMap != NULL)
      for(int i = 0; i < o.nnXLen * o.nnYLen; i++) printf(" %08X", fbits(o.whiteOwnerMap[i]));
    printf("\n");
  }
}

struct Params {
  MiscNNInputParams p;
  bool skipCache;
  bool includeOwnerMap;
};

static Params randomParams() {
  Params r;
  static const double draws[] = {0.5, 0.2, 1.0, 0.0, 0.35};
  r.p.drawEquivalentWinsForWhite = draws[rnd(5)];
  r.p.conservativePassAndIsRoot = rnd(4) == 0;
  r.p.enablePassingHacks = rnd(3) == 0;
  static const double pdas[] = {0.0, 0.0, 0.0, 1.5, -1.0, 0.37};
  r.p.playoutDoublingAdvantage = pdas[rnd(6)];
  static const float temps[] = {1.0f, 1.0f, 1.2f, 0.73f, 2.5f, 0.1f};
  r.p.nnPolicyTemperature = temps[rnd(6)];
  r.p.avoidMYTDaggerHack = rnd(3) == 0;
  r.p.symmetry = rnd(3) == 0 ? NNInputs::SYMMETRY_NOTSPECIFIED : rnd(8);
  static const double opts[] = {0.0, 0.0, 0.5, 1.0, 0.25};
  r.p.policyOptimism = opts[rnd(5)];
  static const int hists[] = {1000, 1000, 0, 2, 5};
  r.p.maxHistory = hists[rnd(5)];
  r.skipCache = rnd(3) == 0;
  r.includeOwnerMap = rnd(3) != 0;
  return r;
}

static void printParams(char kind, const Params& r) {
  printf("%c %016llX %d %d %016llX %08X %d %d %016llX %d %d %d", kind,
         (unsigned long long)dbits(r.p.drawEquivalentWinsForWhite), (int)r.p.conservativePassAndIsRoot,
         (int)r.p.enablePassingHacks, (unsigned long long)dbits(r.p.playoutDoublingAdvantage),
         fbits(r.p.nnPolicyTemperature), (int)r.p.avoidMYTDaggerHack, r.p.symmetry,
         (unsigned long long)dbits(r.p.policyOptimism), r.p.maxHistory, (int)r.skipCache, (int)r.includeOwnerMap);
}

// v1.18.2 NNEvaluator takes a ConfigParser (consulted by the backend only; unused with debugSkipNeuralNet): empty, as testsearchcommon.cpp does
static ConfigParser nnEvalCfg;

static NNEvaluator* newEval(int nnLen, int cachePow, const string& seed, bool doRandomize) {
  printf("E %d %d %d %s %d\n", nnLen, nnLen, cachePow, seed.c_str(), (int)doRandomize);
  NNEvaluator* e = new NNEvaluator(
    "skipnn", "skipnn.bin.gz", "", NULL, 8, nnLen, nnLen, false, false, cachePow, 3, true,
    "", enabled_t::False, 1, {-1}, seed, doRandomize, 0,
    nnEvalCfg
  );
  e->spawnServerThreads();
  return e;
}

static void killEval(NNEvaluator* e) {
  printf("S %llu %llu\n", (unsigned long long)e->numRowsProcessed(), (unsigned long long)e->numBatchesProcessed());
  e->killServerThreads();
  delete e;
}

int main() {
  Board::initHash();
  ScoreValue::initTables();
  S = 20261022;

  struct Cfg { int nnLen; int cachePow; const char* seed; bool doRandomize; int numPos; int maxLen; int killEvery; };
  Cfg cfgs[] = {
    {19, 10, "skipnn-a", false, 60, 19, 0},
    {19, -1, "skipnn-b", true, 50, 19, 0},
    {13, 6, "skipnn-c", false, 50, 13, 0},
    {9, 12, "skipnn-d", false, 40, 9, 0},
    // kill -> respawn every 4 positions (no cache: every eval draws from the current server's Rand)
    {9, -1, "skipnn-e", false, 12, 9, 4},
  };
  for(const Cfg& cfg : cfgs) {
    NNEvaluator* nnEval = newEval(cfg.nnLen, cfg.cachePow, cfg.seed, cfg.doRandomize);
    NNResultBuf buf;
    int numEvalsThisCfg = 0;
    for(int pi = 0; pi < cfg.numPos; pi++) {
      if(cfg.killEvery > 0 && pi > 0 && pi % cfg.killEvery == 0) {
        printf("K\n");
        nnEval->killServerThreads();
        nnEval->spawnServerThreads();
      }
      string line;
      Pos pos = (pi == 0 && cfg.nnLen == 19) ? makeDagger(line) : [&]() {
        int xs = cfg.maxLen - rnd(cfg.maxLen >= 13 ? 8 : 4);
        int ys = rnd(3) == 0 ? cfg.maxLen - rnd(cfg.maxLen >= 13 ? 8 : 4) : xs;
        int maxMoves = rnd(3) == 0 ? xs * ys * 2 : xs * ys / 2;
        return makeGame(xs, ys, maxMoves, line);
      }();
      printf("%s\n", line.c_str());
      int numEvals = 1 + rnd(4);
      for(int k = 0; k < numEvals; k++) {
        Params r = randomParams();
        if(pi == 0 && cfg.nnLen == 19 && k == 0) {
          r.p.avoidMYTDaggerHack = true;
          r.p.enablePassingHacks = false;
        }
        int kind = rnd(8);
        if(kind == 0) {
          int numSym = 1 + rnd(8);
          string seed = "avg" + Global::intToString(rnd(1000));
          printParams('A', r);
          printf(" %d %s\n", numSym, seed.c_str());
          Rand rand(seed);
          std::shared_ptr<NNOutput>* res = nnEval->averageMultipleSymmetries(
            pos.board, pos.hist, pos.nextPla, NULL, r.p, buf, r.includeOwnerMap, rand, numSym);
          printResult(**res, NNPos::getPolicySize(cfg.nnLen, cfg.nnLen), numEvalsThisCfg < 6);
          delete res;
        }
        else {
          printParams('V', r);
          printf("\n");
          nnEval->evaluate(pos.board, pos.hist, pos.nextPla, r.p, buf, r.skipCache, r.includeOwnerMap);
          printResult(*buf.result, NNPos::getPolicySize(cfg.nnLen, cfg.nnLen), numEvalsThisCfg < 6);
          // Same position again from the cache, then with the owner map (hadResultWithoutOwnerMap path).
          if(kind == 1 && !r.skipCache) {
            r.includeOwnerMap = !r.includeOwnerMap;
            printParams('W', r);
            printf("\n");
            nnEval->evaluate(pos.board, pos.hist, pos.nextPla, r.p, buf, false, r.includeOwnerMap);
            printResult(*buf.result, NNPos::getPolicySize(cfg.nnLen, cfg.nnLen), false);
          }
        }
        numEvalsThisCfg++;
      }
    }
    killEval(nnEval);
  }
  ScoreValue::freeTables();
  return 0;
}
