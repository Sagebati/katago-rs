// Oracle for NNOutput in crates/katago-core/src/neuralnet/nninputs.rs: NNOutput::debugPrint (incl.
// the (int)round(x*1000) casts on NaN, infinities and out-of-int-range values) and the averaging
// constructor NNOutput(const vector<shared_ptr<NNOutput>>&) (owner-map averaging over the outputs
// that have one, the legality-mismatch "take the first" fallback for hash collisions, and
// policyOptimismUsed averaging). Not upstream code; links the unmodified upstream sources. The Rust
// mirror is crates/katago-core/tests/nnoutput_oracle.rs: it parses the o lines, replays every op and
// must print the same bytes.
// Every random draw is its own statement (no two rnd() calls in one expression: GCC argument order).
//
// Build (from the repo root, output outside the repo; upstream untouched; asserts on):
//   U=upstream/KataGo/cpp; g++ -std=c++17 -O2 -I$U -o /tmp/nnoutput_oracle \
//     fixtures/upstream-v1.16.5/unit/oracle/nnoutput.cpp \
//     $U/core/global.cpp $U/core/hash.cpp $U/core/rand.cpp $U/core/sha2.cpp $U/core/md5.cpp \
//     $U/core/timer.cpp $U/core/bsearch.cpp $U/game/board.cpp $U/game/rules.cpp \
//     $U/game/boardhistory.cpp $U/game/graphhash.cpp $U/neuralnet/nninputs.cpp -lpthread
//   /tmp/nnoutput_oracle > fixtures/upstream-v1.16.5/unit/oracle/nnoutput.txt
// Generated with g++ (GCC) 16.2.1 / glibc 2.44 on x86_64 Linux.
//
// Output (floats as uppercase hex bit patterns, f32: 8 digits):
//   o <nnXLen> <nnYLen> <hash0> <hash1> <whiteWinProb> <whiteLossProb> <whiteNoResultProb>
//     <whiteScoreMean> <whiteScoreMeanSq> <whiteLead> <varTimeLeft> <shorttermWinlossError>
//     <shorttermScoreError> <policyOptimismUsed> <n> <policyProbs[0..n)> <owner01> [<whiteOwnerMap[0..nnXLen*nnYLen)>]
//     an NNOutput (n = nnXLen*nnYLen+1; policyProbs[n..MAX_NN_POLICY_SIZE) are -1)
//   debugPrint <xSize> <ySize>   then one "|<line>" per output line of the preceding o's
//     debugPrint(out, Board(xSize,ySize))
//   avg <k>   then "r ..." (same fields as o, plus <noised01>): NNOutput(others) of the preceding k o's
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include "core/global.h"
#include "core/hash.h"
#include "core/test.h"
#include "game/board.h"
#include "neuralnet/nninputs.h"
using namespace std;

static uint64_t S;
static uint64_t nxt() { uint64_t z = (S += 0x9E3779B97F4A7C15ULL); z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL; z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL; return z ^ (z >> 31); }
static int rnd(int n) { return (int)(nxt() % (uint64_t)n); }
static float unif() { return (float)((double)(nxt() >> 11) * 0x1.0p-53); }

static uint32_t fbits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }

static void printOut(const char* tag, const NNOutput& o, bool withNoised) {
  printf("%s %d %d %016llX %016llX", tag, o.nnXLen, o.nnYLen, (unsigned long long)o.nnHash.hash0, (unsigned long long)o.nnHash.hash1);
  const float vals[10] = {o.whiteWinProb, o.whiteLossProb, o.whiteNoResultProb, o.whiteScoreMean, o.whiteScoreMeanSq,
                          o.whiteLead, o.varTimeLeft, o.shorttermWinlossError, o.shorttermScoreError, o.policyOptimismUsed};
  for(int i = 0; i < 10; i++) printf(" %08X", fbits(vals[i]));
  int n = o.nnXLen * o.nnYLen + 1;
  printf(" %d", n);
  for(int i = 0; i < n; i++) printf(" %08X", fbits(o.policyProbs[i]));
  for(int i = n; i < NNPos::MAX_NN_POLICY_SIZE; i++) testAssert(fbits(o.policyProbs[i]) == fbits(-1.0f));
  printf(" %d", o.whiteOwnerMap != NULL ? 1 : 0);
  if(o.whiteOwnerMap != NULL)
    for(int i = 0; i < o.nnXLen * o.nnYLen; i++) printf(" %08X", fbits(o.whiteOwnerMap[i]));
  if(withNoised) printf(" %d", o.noisedPolicyProbs != NULL ? 1 : 0);
  printf("\n");
}

// Values that stress debugPrint's "%.2f"-style printing and (int)round(x*1000).
static float special() {
  static const float SP[] = {
    std::numeric_limits<float>::quiet_NaN(), -std::numeric_limits<float>::quiet_NaN(),
    std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
    1e30f, -1e30f, 3e9f, -3e9f, 2147483.647f, -2147483.648f, 2147483.0f, 2147484.0f, 0.0005f, -0.0005f, 0.0015f, 0.0025f,
    -0.0f, 0.0f, 1e-45f, 0.9995f, -0.9995f, 0.12345f, 99.995f, 1.005f};
  int k = rnd((int)(sizeof(SP) / sizeof(SP[0])));
  return SP[k];
}

static float val(int specialPct, float scale) {
  int r = rnd(100);
  if(r < specialPct) return special();
  float u = unif();
  return (u * 2.0f - 1.0f) * scale;
}

static shared_ptr<NNOutput> randOut(int nnX, int nnY, Hash128 hash, const vector<bool>& legal, int specialPct, int ownerPct, float optimism) {
  shared_ptr<NNOutput> o = make_shared<NNOutput>();
  o->nnHash = hash;
  o->nnXLen = nnX;
  o->nnYLen = nnY;
  o->whiteWinProb = val(specialPct, 1.0f);
  o->whiteLossProb = val(specialPct, 1.0f);
  o->whiteNoResultProb = val(specialPct, 0.1f);
  o->whiteScoreMean = val(specialPct, 60.0f);
  o->whiteScoreMeanSq = val(specialPct, 3000.0f);
  o->whiteLead = val(specialPct, 60.0f);
  o->varTimeLeft = val(specialPct, 200.0f);
  o->shorttermWinlossError = val(specialPct, 0.5f);
  o->shorttermScoreError = val(specialPct, 20.0f);
  o->policyOptimismUsed = optimism;
  int n = nnX * nnY + 1;
  for(int i = 0; i < NNPos::MAX_NN_POLICY_SIZE; i++) o->policyProbs[i] = -1.0f;
  for(int i = 0; i < n; i++) {
    if(!legal[i]) { o->policyProbs[i] = -1.0f; continue; }
    int r = rnd(100);
    if(r < specialPct) o->policyProbs[i] = special();
    else o->policyProbs[i] = unif() * 0.3f;
  }
  int ro = rnd(100);
  if(ro < ownerPct) {
    o->whiteOwnerMap = new float[nnX * nnY];
    for(int i = 0; i < nnX * nnY; i++) o->whiteOwnerMap[i] = val(specialPct, 1.0f);
  }
  return o;
}

static Hash128 randHash() {
  uint64_t a = nxt();
  uint64_t b = nxt();
  return Hash128(a, b);
}

static vector<bool> randLegal(int n, int illegalPct) {
  vector<bool> legal(n);
  for(int i = 0; i < n; i++) {
    int r = rnd(100);
    legal[i] = !(r < illegalPct);
  }
  legal[n - 1] = true;
  return legal;
}

int main() {
  Board::initHash();
  S = 0x00A1E5EEDULL;

  // debugPrint
  for(int c = 0; c < 40; c++) {
    int nnX = 2 + rnd(6);
    int nnY = 2 + rnd(6);
    int xs = nnX;
    int ys = nnY;
    if(rnd(3) == 0) xs = 1 + rnd(nnX);
    if(rnd(3) == 0) ys = 1 + rnd(nnY);
    if(xs < 2 && ys < 2) xs = 2;
    int specialPct = c < 10 ? 0 : (c < 25 ? 20 : 60);
    vector<bool> legal = randLegal(nnX * nnY + 1, 20);
    float opt = val(specialPct, 1.0f);
    shared_ptr<NNOutput> o = randOut(nnX, nnY, randHash(), legal, specialPct, 70, opt);
    printOut("o", *o, false);
    printf("debugPrint %d %d\n", xs, ys);
    Board board(xs, ys);
    ostringstream out;
    o->debugPrint(out, board);
    istringstream in(out.str());
    string line;
    while(getline(in, line)) printf("|%s\n", line.c_str());
  }

  // Averaging constructor
  for(int c = 0; c < 80; c++) {
    int nnX = 2 + rnd(5);
    int nnY = 2 + rnd(5);
    int len = 1 + rnd(8);
    int kind = c % 4;
    Hash128 hash = randHash();
    int n = nnX * nnY + 1;
    vector<bool> legal = randLegal(n, 25);
    static const float OPTS[] = {0.0f, 0.5f, 1.0f, 0.25f, 0.75f};
    int ko = rnd(5);
    float opt0 = OPTS[ko];
    int ownerPct = kind == 3 ? 50 : (rnd(2) == 0 ? 100 : 0);
    vector<shared_ptr<NNOutput>> others;
    for(int i = 0; i < len; i++) {
      vector<bool> l = legal;
      // kind 2: a legality mismatch (a "hash collision") in some later output
      if(kind == 2 && i > 0) {
        int r = rnd(3);
        if(r == 0) {
          int p = rnd(n - 1);
          l[p] = !l[p];
        }
      }
      float opt = opt0;
      // kind 1: the optimism differs between the outputs
      if(kind == 1) {
        int k = rnd(5);
        opt = OPTS[k];
      }
      others.push_back(randOut(nnX, nnY, hash, l, 0, ownerPct, opt));
      printOut("o", *others.back(), false);
    }
    printf("avg %d\n", len);
    NNOutput avg(others);
    printOut("r", avg, true);
  }
  return 0;
}
