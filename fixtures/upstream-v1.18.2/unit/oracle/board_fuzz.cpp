// Differential fuzz oracle for game/board (Board, Location), adopted from the pilot board
// review. Links against the unmodified upstream sources of KataGo v1.18.2.
// The Rust mirror is crates/katago-core/tests/board_fuzz_oracle.rs: keep the two line for line.
// Usage: board_fuzz games <numGames> <seed0>   full trace of every game (stdout)
//        board_fuzz hashes <numGames> <seed0>  one line per game: "H <g> <bytes> <fnv1a64 hex>"
//        board_fuzz loc                        Location::tryOfString on malformed/overflowing input
// Build (from the repo root, output outside the repo; upstream untouched; asserts on):
//   U=upstream/KataGo/cpp; g++ -std=c++17 -O2 -I$U -o /tmp/board_fuzz \
//     fixtures/upstream-v1.18.2/unit/oracle/board_fuzz.cpp \
//     $U/core/global.cpp $U/core/hash.cpp $U/core/rand.cpp $U/core/sha2.cpp $U/core/md5.cpp \
//     $U/core/timer.cpp $U/core/bsearch.cpp $U/game/board.cpp -lpthread
//   cd fixtures/upstream-v1.18.2/unit/oracle
//   /tmp/board_fuzz hashes 800 1 > board_fuzz_hashes.txt
//   /tmp/board_fuzz games 10 7 > board_fuzz_trace.txt
//   /tmp/board_fuzz loc > board_fuzz_loc.txt
// Generated with g++ 16.2.1 / glibc on x86_64 Linux.
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include "core/global.h"
#include "core/hash.h"
#include "game/board.h"
using namespace std;

static uint64_t S;
static uint64_t nxt() { uint64_t z = (S += 0x9E3779B97F4A7C15ULL); z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL; z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL; return z ^ (z >> 31); }
static int rnd(int n) { return (int)(nxt() % (uint64_t)n); }

static string out;
static void P(const string& s) { out += s; }
static string I(long long x) { return to_string(x); }

static uint64_t fold(const Board& b) {
  uint64_t h = 1469598103934665603ULL;
  auto mix = [&](long long v) { h ^= (uint64_t)v; h *= 1099511628211ULL; };
  for(int loc = 0; loc < Board::MAX_ARR_SIZE; loc++) {
    mix(b.colors[loc]);
    if(b.colors[loc] == C_BLACK || b.colors[loc] == C_WHITE) {
      mix(b.chain_head[loc]); mix(b.next_in_chain[loc]);
      if(b.chain_head[loc] == loc) { mix(b.chain_data[loc].owner); mix(b.chain_data[loc].num_locs); mix(b.chain_data[loc].num_liberties); }
    }
  }
  mix(b.ko_loc); mix((long long)b.pos_hash.hash0); mix((long long)b.pos_hash.hash1); mix(b.numBlackCaptures); mix(b.numWhiteCaptures);
  return h;
}
static string st(const Board& b) { return " F" + I((long long)(fold(b) >> 1)) + " ko" + I(b.ko_loc); }

static string cons(const Board& b) {
  try { b.checkConsistency(); return "ok"; }
  catch(const StringError& e) { return string("ERR ") + e.what(); }
}

static string areaStr(const Board& b, const Color* r) {
  string s;
  for(int y = 0; y < b.y_size; y++) for(int x = 0; x < b.x_size; x++) s += (char)('0' + r[Location::getLoc(x,y,b.x_size)]);
  return s;
}

static void analyze(Board& b, bool sui, int tag) {
  P("A " + I(tag) + st(b) + "\n");
  vector<Loc> buf, buf2;
  Color area[Board::MAX_ARR_SIZE];
  for(int m = 0; m < 16; m++) {
    b.calculateArea(area, m&1, (m>>1)&1, (m>>2)&1, (m>>3)&1);
    P("CA " + I(m) + " " + areaStr(b, area) + "\n");
  }
  // m < 8: excludeTerritoryAdjacentToAtari = false (the v1.16.5 behavior); m >= 8: the v1.18.2 rules-v3 mode.
  for(int m = 0; m < 16; m++) {
    int cnt = -999;
    b.calculateIndependentLifeArea(area, cnt, m&1, (m>>1)&1, (m>>3)&1, (m>>2)&1);
    P("IL " + I(m) + " " + I(cnt) + " " + areaStr(b, area) + "\n");
  }
  Color pa[Board::MAX_ARR_SIZE];
  b.calculateArea(pa, false, false, false, sui);
  for(int y = 0; y < b.y_size; y++) for(int x = 0; x < b.x_size; x++) {
    Loc loc = Location::getLoc(x,y,b.x_size);
    string s = "L " + I(loc);
    for(int bd : {0,1,2,3,5,8,13,30}) s += (b.simpleRepetitionBoundGt(loc,bd) ? "1" : "0");
    if(b.colors[loc] == C_EMPTY) {
      for(Player p = 1; p <= 2; p++) {
        s += " p" + I(p) + ":";
        s += I(b.isLegal(loc,p,false)) + I(b.isLegal(loc,p,true)) + I(b.isLegalIgnoringKo(loc,p,false)) + I(b.isLegalIgnoringKo(loc,p,true));
        s += I(b.isSuicide(loc,p)) + I(b.isIllegalSuicide(loc,p,false)) + I(b.isIllegalSuicide(loc,p,true));
        s += I(b.isSimpleEye(loc,p)) + I(b.wouldBeCapture(loc,p)) + I(b.wouldBeKoCapture(loc,p)) + "," + I(b.getKoCaptureLoc(loc,p));
        s += "," + I(b.isAdjacentToPla(loc,p)) + I(b.isAdjacentOrDiagonalToPla(loc,p)) + I(b.isNonPassAliveSelfConnection(loc,p,pa));
        for(int mx : {1,2,3,4,6,1000}) s += "," + I(b.getNumLibertiesAfterPlay(loc,p,mx));
        int lo = -5, hi = -5; b.getBoundNumLibertiesAfterPlay(loc,p,lo,hi);
        s += "," + I(lo) + "," + I(hi);
        Hash128 h = b.getPosHashAfterMove(loc,p);
        s += "," + Global::uint64ToHexString(h.hash0 ^ h.hash1);
        s += "," + I(b.getNumImmediateLiberties(loc));
      }
      int cx = rnd(b.x_size); int cy = rnd(b.y_size); Loc ch = Location::getLoc(cx, cy, b.x_size);
      s += " adjch" + I(b.isAdjacentToChain(loc,ch));
    }
    else {
      s += " libs" + I(b.getNumLiberties(loc)) + " sz" + I(b.getChainSize(loc));
    }
    P(s + "\n");
  }
  // Ladders, run in place (mutates linked lists)
  for(int y = 0; y < b.y_size; y++) for(int x = 0; x < b.x_size; x++) {
    Loc loc = Location::getLoc(x,y,b.x_size);
    if(b.colors[loc] == C_EMPTY) continue;
    bool r1 = b.searchIsLadderCaptured(loc,true,buf);
    string s = "LD " + I(loc) + " " + I(r1) + " b" + I(buf.size()) + st(b);
    bool r2 = b.searchIsLadderCaptured(loc,false,buf);
    s += " | " + I(r2) + " b" + I(buf.size()) + st(b);
    bool r3 = b.searchIsLadderCapturedAttackerFirst2Libs(loc,buf,buf2);
    s += " | " + I(r3) + " b" + I(buf.size()) + " w";
    for(Loc w : buf2) s += I(w) + ",";
    s += st(b);
    P(s + "\n");
  }
}


static uint64_t fnv1a64(const string& s) {
  uint64_t h = 1469598103934665603ULL;
  for(unsigned char c : s) { h ^= (uint64_t)c; h *= 1099511628211ULL; }
  return h;
}

static void runGame(int g, uint64_t seed0) {
  int sizes[][2] = {{19,19},{9,9},{13,7},{5,17},{2,2},{1,1},{3,3},{19,1},{1,19},{7,7},{4,6},{19,13},{6,6},{11,11},{2,9}};
  int numSizes = 15;
  {
    S = seed0 * 1000003ULL + g;
    int si = rnd(numSizes);
    int xs = sizes[si][0], ys = sizes[si][1];
    bool sui = rnd(2) == 1;
    Board b(xs,ys);
    P("G " + I(g) + " " + I(xs) + " " + I(ys) + " " + I(sui) + "\n");
    int area = xs*ys;
    int steps = rnd(area*2 + 2) + 1;
    int chaseProb = rnd(90);
    vector<Board::MoveRecord> recs;
    Player pla = P_BLACK;
    int nextAnalysis = rnd(area/2 + 2);
    for(int n = 0; n < steps; n++) {
      Loc loc = Board::PASS_LOC;
      int r = rnd(100);
      bool found = false;
      if(r >= 2 && r < 2 + chaseProb) {
        // chase: liberties of chains with <= 2 libs
        vector<Loc> cands;
        for(int y = 0; y < ys; y++) for(int x = 0; x < xs; x++) {
          Loc l = Location::getLoc(x,y,xs);
          if(b.colors[l] != C_EMPTY) continue;
          for(int i = 0; i < 4; i++) {
            Loc a = l + b.adj_offsets[i];
            if((b.colors[a] == C_BLACK || b.colors[a] == C_WHITE) && b.getNumLiberties(a) <= 2) { cands.push_back(l); break; }
          }
        }
        if(cands.size() > 0) {
          Loc l = cands[rnd((int)cands.size())];
          if(b.isLegal(l,pla,sui)) { loc = l; found = true; }
        }
      }
      if(!found && r >= 2 && area > 0) {
        for(int t = 0; t < 60; t++) {
          int lx = rnd(xs); int ly = rnd(ys); Loc l = Location::getLoc(lx, ly, xs);
          if(b.isLegal(l,pla,sui)) { loc = l; found = true; break; }
        }
      }
      recs.push_back(b.playMoveRecorded(loc,pla));
      P("M " + I(n) + " " + I(pla) + " " + I(loc) + " cd" + I(recs.back().capDirs) + " rk" + I(recs.back().ko_loc) + st(b) + " c" + I(b.numBlackCaptures) + "," + I(b.numWhiteCaptures) + "\n");
      if(n == nextAnalysis) {
        analyze(b, sui, n);
        nextAnalysis = n + 1 + rnd(area/2 + 2);
      }
      pla = rnd(10) < 9 ? getOpp(pla) : pla;
    }
    analyze(b, sui, -1);
    // undo some
    int nu = rnd((int)recs.size() + 1);
    for(int k = 0; k < nu; k++) {
      b.undo(recs.back()); recs.pop_back();
      P("U" + st(b) + " c" + I(b.numBlackCaptures) + "," + I(b.numWhiteCaptures) + "\n");
    }
    if(nu > 0) analyze(b, sui, -2);
    // setStone fuzz
    if(area > 0) {
      int ns = rnd(40);
      for(int k = 0; k < ns; k++) {
        int lx = rnd(xs); int ly = rnd(ys); Loc l = Location::getLoc(lx, ly, xs);
        Color c = (Color)rnd(3);
        bool ok;
        int which = rnd(2);
        if(which == 0) ok = b.setStone(l,c);
        else ok = b.setStoneFailIfNoLibs(l,c);
        P("S " + I(which) + " " + I(l) + " " + I(c) + " " + I(ok) + st(b) + "\n");
      }
      analyze(b, sui, -3);
    }
    // regenChainsFromColors on a copy, then setStonesTolerant (v1.18.2): random placements, including
    // off-board / wall / out-of-range locations and invalid colors (ignored), duplicates (last one wins),
    // and dense fills that leave zero-liberty groups to be removed simultaneously.
    {
      Board rg = b;
      rg.regenChainsFromColors();
      P("RG" + st(rg) + " " + cons(rg) + "\n");
      int nt = rnd(4) + 1;
      for(int k = 0; k < nt; k++) {
        int np = rnd(area + 4);
        vector<Move> pl;
        string ps;
        for(int j = 0; j < np; j++) {
          int kind = rnd(12);
          Loc l;
          if(kind == 0) {
            int z = rnd(Board::MAX_ARR_SIZE + 40);
            l = (Loc)(z - 20);
          }
          else {
            int lx = rnd(xs); int ly = rnd(ys); l = Location::getLoc(lx, ly, xs);
          }
          int cc = rnd(8);
          Color c;
          if(cc <= 2) c = C_BLACK;
          else if(cc <= 5) c = C_WHITE;
          else if(cc == 6) c = C_EMPTY;
          else {
            int bad = rnd(3);
            c = bad == 0 ? (Color)-1 : bad == 1 ? C_WALL : (Color)4;
          }
          pl.push_back(Move(l, c));
          ps += I(l) + ":" + I(c) + ",";
        }
        int nr = b.setStonesTolerant(pl);
        P("T " + I(np) + " " + ps + " " + I(nr) + st(b) + " " + cons(b) + "\n");
      }
      analyze(b, sui, -4);
    }
  }
}

static void locMode() {
  const char* inputs[] = {
    "A1", "T19", "t19", "a19", "J9", "I9", "Z1", "AA1", "zz25", "pass", "PSS", "Pass", "null", "", "A", "A0", "A20",
    "(0,0)", "(18,18)", "(3,4)", "(-1,0)", "(-2,0)", "(19,0)", "(1000000,1000000)", "(1,2", "(1,2,3)", "( 5 , 6 )",
    // C++ signed overflow (UB) in getLoc / `y = y_size - y`: the pilot review's panicking inputs and neighbours.
    "(2147483647,0)", "(0,2147483647)", "A-2147483648",
    "(-2147483648,0)", "(0,-2147483648)", "(2147483647,2147483647)", "(-2147483648,-2147483648)",
    "A2147483647", "A-2147483647", "ZZ-2147483648", "(2147483646,1)", "(65535,0)", "(0,65535)",
  };
  int sizes[][2] = {{19,19},{9,13},{1,1},{25,25}};
  for(auto& sz : sizes) {
    for(const char* in : inputs) {
      Loc l = -7;
      bool ok = Location::tryOfString(string(in), sz[0], sz[1], l);
      cout << "tryOfString " << sz[0] << " " << sz[1] << " [" << in << "] -> " << ok << " " << (int)l << "\n";
    }
  }
}

int main(int argc, char** argv) {
  Board::initHash();
  string mode = argc > 1 ? string(argv[1]) : string("");
  if(mode == "loc") {
    locMode();
    return 0;
  }
  if(mode != "games" && mode != "hashes") {
    cerr << "usage: board_fuzz games|hashes <numGames> <seed0> | loc" << endl;
    return 1;
  }
  int numGames = atoi(argv[2]);
  uint64_t seed0 = strtoull(argv[3],NULL,10);
  for(int g = 0; g < numGames; g++) {
    out.clear();
    runGame(g, seed0);
    if(mode == "games")
      fwrite(out.data(), 1, out.size(), stdout);
    else
      printf("H %d %zu %016llx\n", g, out.size(), (unsigned long long)fnv1a64(out));
  }
  return 0;
}
