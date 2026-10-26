// Oracle for crates/katago-core/src/search/localpattern.rs: the unmodified upstream LocalPatternHasher
// (init's RNG stream and tables, getHash, getHashWithSym over every symmetry and color flip) on random
// boards. Links the unmodified upstream sources via the Eigen build's object files.
// The Rust mirror is crates/katago-core/tests/localpattern_oracle.rs: keep the two in the same order.
// Build (from the repo root, after `just katago-eigen`; output outside the repo):
//   O=build/eigen/CMakeFiles/katago.dir; g++ -std=c++17 -O2 -Iupstream/KataGo/cpp -o /tmp/localpattern \
//     fixtures/upstream-v1.18.2/unit/oracle/localpattern.cpp \
//     $(find $O/core $O/game $O/neuralnet $O/search $O/dataio $O/program -name '*.o') -lz -lzip -lpthread
//   /tmp/localpattern > fixtures/upstream-v1.18.2/unit/oracle/localpattern.txt
// Output:
//   T <hcfg> <xSize> <ySize> <nLocalPattern> <nPla> <nAtari> <fnvLocalPattern> <fnvPla> <fnvAtari> <last local pattern> <rand calls>
//   B <board idx> <x> <y> <stones> <ko loc> <pos hash>                     board snapshot
//   C <hcfg> <board idx> <n> <fnv of getHash over all locs and both players> <fnv of getHashWithSym over all
//     locs, both players, symmetries 0..7, flipColors false/true>
//   H <hcfg> <board idx> <loc> <pla> <getHash> <getHashWithSym(sym=hcfg-board-derived, flip)>   a sample of single values
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
#include "neuralnet/nninputs.h"
#include "search/localpattern.h"
using namespace std;

static uint64_t S;
static uint64_t nxt() { uint64_t z = (S += 0x9E3779B97F4A7C15ULL); z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL; z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL; return z ^ (z >> 31); }
static int rnd(int n) { return (int)(nxt() % (uint64_t)n); }

static void fnvU64(uint64_t& h, uint64_t v) {
  for(int b = 0; b < 8; b++) { h ^= (v >> (8 * b)) & 0xFF; h *= 0x100000001b3ULL; }
}
static void fnvHash(uint64_t& h, const Hash128& x) { fnvU64(h, x.hash0); fnvU64(h, x.hash1); }
static const uint64_t FNV0 = 0xcbf29ce484222325ULL;

int main() {
  Board::initHash();
  S = 0x4c6f63616c506174ULL;

  struct HCfg { int px, py; const char* seed; };
  vector<HCfg> hcfgs = {
    {5,5,"lp seed 1"}, {3,3,"lp seed 2"}, {1,1,"lp seed 3"}, {7,5,"lp seed 4"}, {5,7,"lp seed 5"}, {9,9,"lp seed 6"},
  };
  vector<pair<int,int>> sizes = {{9,9},{13,13},{19,19},{9,13},{19,9},{11,15}};

  //Board snapshots
  vector<Board> boards;
  for(auto& sz : sizes) {
    Board board(sz.first, sz.second);
    Player pla = P_BLACK;
    int area = sz.first * sz.second;
    int snapEvery = area / 3;
    for(int i = 1; i <= snapEvery * 6; i++) {
      Loc loc = Board::PASS_LOC;
      if(rnd(40) != 0) {
        for(int tries = 0; tries < 50; tries++) {
          int cx = rnd(sz.first);
          int cy = rnd(sz.second);
          Loc cand = Location::getLoc(cx, cy, sz.first);
          if(board.isLegal(cand, pla, true)) { loc = cand; break; }
        }
      }
      board.playMoveAssumeLegal(loc, pla);
      pla = getOpp(pla);
      if(i % snapEvery == 0)
        boards.push_back(board);
    }
  }
  for(size_t bi = 0; bi < boards.size(); bi++) {
    const Board& b = boards[bi];
    printf("B %d %d %d %d %d %s\n", (int)bi, b.x_size, b.y_size, b.numStonesOnBoard(), (int)b.ko_loc, b.pos_hash.toString().c_str());
  }

  for(size_t hi = 0; hi < hcfgs.size(); hi++) {
    const HCfg& hc = hcfgs[hi];
    Rand rand(hc.seed);
    LocalPatternHasher hasher;
    hasher.init(hc.px, hc.py, rand);
    uint64_t f0 = FNV0, f1 = FNV0, f2 = FNV0;
    for(const Hash128& x : hasher.zobristLocalPattern) fnvHash(f0, x);
    for(const Hash128& x : hasher.zobristPla) fnvHash(f1, x);
    for(const Hash128& x : hasher.zobristAtari) fnvHash(f2, x);
    printf("T %d %d %d %d %d %d %016llX %016llX %016llX %s %llu\n", (int)hi, hasher.xSize, hasher.ySize,
           (int)hasher.zobristLocalPattern.size(), (int)hasher.zobristPla.size(), (int)hasher.zobristAtari.size(),
           (unsigned long long)f0, (unsigned long long)f1, (unsigned long long)f2,
           hasher.zobristLocalPattern.back().toString().c_str(), (unsigned long long)rand.getNumCalls());

    for(size_t bi = 0; bi < boards.size(); bi++) {
      const Board& b = boards[bi];
      if(b.x_size < hc.px || b.y_size < hc.py)
        continue;
      uint64_t fa = FNV0, fb = FNV0;
      int n = 0;
      vector<Loc> locs;
      locs.push_back(Board::PASS_LOC);
      locs.push_back(Board::NULL_LOC);
      for(int y = 0; y < b.y_size; y++)
        for(int x = 0; x < b.x_size; x++)
          locs.push_back(Location::getLoc(x, y, b.x_size));
      for(Loc loc : locs) {
        for(Player pla : {P_BLACK, P_WHITE}) {
          Hash128 h = hasher.getHash(b, loc, pla);
          fnvHash(fa, h);
          n++;
          for(int sym = 0; sym < SymmetryHelpers::NUM_SYMMETRIES; sym++) {
            for(int flip = 0; flip < 2; flip++) {
              Hash128 hs = hasher.getHashWithSym(b, loc, pla, sym, flip != 0);
              fnvHash(fb, hs);
              if(n % 97 == 0 && sym == (int)((bi + hi) % 8))
                printf("H %d %d %d %d %s %s %d %d\n", (int)hi, (int)bi, (int)loc, (int)pla, h.toString().c_str(), hs.toString().c_str(), sym, flip);
            }
          }
        }
      }
      printf("C %d %d %d %016llX %016llX\n", (int)hi, (int)bi, n, (unsigned long long)fa, (unsigned long long)fb);
    }
  }
  return 0;
}
