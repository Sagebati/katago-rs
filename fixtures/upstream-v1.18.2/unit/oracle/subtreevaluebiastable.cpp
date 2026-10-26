// Oracle for crates/katago-core/src/search/subtreevaluebiastable.rs: the unmodified upstream
// SubtreeValueBiasTable (its file-static zobrist tables and LocalPatternHasher are observed through the
// keys that get() stores in the public `entries` shards; shard choice; shared entries; map order;
// clearUnusedSynchronous) on random boards and on hand-made ko positions. Links the unmodified upstream
// sources via the Eigen build's object files.
// The Rust mirror is crates/katago-core/tests/subtreevaluebiastable_oracle.rs: keep the two in the same order.
// Build (from the repo root, after `just katago-eigen`; output outside the repo):
//   O=build/eigen/CMakeFiles/katago.dir; g++ -std=c++17 -O2 -Iupstream/KataGo/cpp -o /tmp/subtreevaluebiastable \
//     fixtures/upstream-v1.18.2/unit/oracle/subtreevaluebiastable.cpp \
//     $(find $O/core $O/game $O/neuralnet $O/search $O/dataio $O/program -name '*.o') -lz -lzip -lpthread
//   /tmp/subtreevaluebiastable > fixtures/upstream-v1.18.2/unit/oracle/subtreevaluebiastable.txt
// Output:
//   B <idx> <x> <y> <stones> <ko loc> <pos hash>                       board
//   G <shards> <i> <pool idx> <shard> <hash> <new> <use_count>          one get(); entry then gets +=, and every 3rd is held
//   N <shards> <total entries>                                          after the loop
//   V <shards> <shard> <hash> <deltaUtilitySum bits> <weightSum bits> <use_count>   all entries, shard then map order
//   A <shards> <step> <total entries>                                   after clearUnusedSynchronous()
//   K <shards> <step> <shard> <hash>                                    remaining keys, shard then map order
// Generated with g++ (GCC) 16.2.1 / glibc 2.44 on x86_64 (FMA) Linux.
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include "core/global.h"
#include "core/hash.h"
#include "core/rand.h"
#include "game/board.h"
#include "search/subtreevaluebiastable.h"
using namespace std;

static uint64_t S;
static uint64_t nxt() { uint64_t z = (S += 0x9E3779B97F4A7C15ULL); z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL; z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL; return z ^ (z >> 31); }
static int rnd(int n) { return (int)(nxt() % (uint64_t)n); }
static unsigned long long dbits(double d) { uint64_t u; memcpy(&u, &d, 8); return (unsigned long long)u; }

struct PoolItem { int boardIdx; Player pla; Loc parentLoc; Loc prevLoc; };

static void dumpEntries(const char* tag, int shards, int step, SubtreeValueBiasTable& table, bool values) {
  int total = 0;
  for(size_t s = 0; s < table.entries.size(); s++)
    total += (int)table.entries[s].size();
  printf("%s %d %d %d\n", values ? "N" : "A", shards, step, total);
  for(size_t s = 0; s < table.entries.size(); s++) {
    for(auto& kv : table.entries[s]) {
      if(values)
        printf("V %d %d %s %016llX %016llX %d\n", shards, (int)s, kv.first.toString().c_str(), dbits(kv.second->deltaUtilitySum), dbits(kv.second->weightSum), (int)kv.second.use_count());
      else
        printf("K %d %d %d %s\n", shards, step, (int)s, kv.first.toString().c_str());
    }
  }
  (void)tag;
}

int main() {
  Board::initHash();
  S = 0x5375627472656521ULL;

  vector<Board> boards;
  {
    vector<pair<int,int>> sizes = {{9,9},{19,19},{13,7}};
    for(auto& sz : sizes) {
      Board board(sz.first, sz.second);
      Player pla = P_BLACK;
      int area = sz.first * sz.second;
      int snapEvery = area / 3;
      for(int i = 1; i <= snapEvery * 4; i++) {
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
    //Hand-made simple ko on 9x9, black to have just captured at (2,1); ko point (1,1), and the same shifted
    for(int shift = 0; shift < 2; shift++) {
      Board b(9, 9);
      int ox = shift * 4, oy = shift * 3;
      auto L = [&](int x, int y) { return Location::getLoc(x + ox, y + oy, 9); };
      b.setStone(L(1,0), C_BLACK); b.setStone(L(0,1), C_BLACK); b.setStone(L(1,2), C_BLACK);
      b.setStone(L(1,1), C_WHITE);
      b.setStone(L(2,0), C_WHITE); b.setStone(L(3,1), C_WHITE); b.setStone(L(2,2), C_WHITE);
      b.playMoveAssumeLegal(L(2,1), P_BLACK);
      boards.push_back(b);
    }
  }
  for(size_t bi = 0; bi < boards.size(); bi++) {
    const Board& b = boards[bi];
    printf("B %d %d %d %d %d %s\n", (int)bi, b.x_size, b.y_size, b.numStonesOnBoard(), (int)b.ko_loc, b.pos_hash.toString().c_str());
  }

  //Pool of get() arguments, with repeats drawn from it
  vector<PoolItem> pool;
  for(int k = 0; k < 90; k++) {
    PoolItem it;
    it.boardIdx = (k < 10) ? (int)boards.size() - 1 - (k % 2) : rnd((int)boards.size());
    const Board& b = boards[it.boardIdx];
    it.pla = rnd(2) == 0 ? P_BLACK : P_WHITE;
    auto randLoc = [&]() -> Loc {
      int r = rnd(12);
      if(r == 0) return Board::PASS_LOC;
      if(r == 1) return Board::NULL_LOC;
      int x = rnd(b.x_size);
      int y = rnd(b.y_size);
      return Location::getLoc(x, y, b.x_size);
    };
    it.parentLoc = randLoc();
    it.prevLoc = randLoc();
    if(k < 10) {
      //ko boards: prevLoc is the ko capture, so ko_loc != NULL_LOC is part of the hash
      it.prevLoc = Location::getLoc(2 + (it.boardIdx == (int)boards.size() - 1 ? 4 : 0), 1 + (it.boardIdx == (int)boards.size() - 1 ? 3 : 0), 9);
    }
    pool.push_back(it);
  }

  for(int shards : {1, 5, 64, 65536}) {
    SubtreeValueBiasTable table(shards);
    vector<shared_ptr<SubtreeValueBiasEntry>> held;
    vector<pair<int,string>> seen;
    int numCalls = shards == 65536 ? 120 : 400;
    for(int i = 0; i < numCalls; i++) {
      int pi = rnd((int)pool.size());
      const PoolItem& it = pool[pi];
      shared_ptr<SubtreeValueBiasEntry> e = table.get(it.pla, it.parentLoc, it.prevLoc, boards[it.boardIdx]);
      int foundShard = -1;
      string foundHash;
      for(size_t s = 0; s < table.entries.size() && foundShard < 0; s++) {
        for(auto& kv : table.entries[s]) {
          if(kv.second.get() == e.get()) { foundShard = (int)s; foundHash = kv.first.toString(); break; }
        }
      }
      bool isNew = true;
      for(auto& sp : seen) if(sp.first == foundShard && sp.second == foundHash) isNew = false;
      if(isNew) seen.push_back(make_pair(foundShard, foundHash));
      printf("G %d %d %d %d %s %d %d\n", shards, i, pi, foundShard, foundHash.c_str(), (int)isNew, (int)e.use_count());
      e->deltaUtilitySum += i * 0.5 + 1.0;
      e->weightSum += 1.0 + 0.125 * (i % 7);
      if(i % 3 == 0)
        held.push_back(e);
    }
    dumpEntries("N", shards, 0, table, true);
    table.clearUnusedSynchronous();
    dumpEntries("A", shards, 1, table, false);
    //Release every other held entry, clear again
    for(size_t k = 0; k < held.size(); k += 2)
      held[k].reset();
    table.clearUnusedSynchronous();
    dumpEntries("A", shards, 2, table, false);
    held.clear();
    table.clearUnusedSynchronous();
    dumpEntries("A", shards, 3, table, false);
  }
  return 0;
}
