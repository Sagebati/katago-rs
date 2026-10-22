// Differential fuzz oracle for game/boardhistory (BoardHistory, KoHashTable, GraphHash). Links against the
// unmodified upstream sources of KataGo v1.16.5.
// The Rust mirror is crates/katago-core/tests/boardhistory_fuzz_oracle.rs: keep the two line for line.
// Every random draw is its own statement (no two rnd() calls in one expression: GCC argument order).
// Usage: boardhistory_fuzz games <numGames> <seed0>   full trace of every game (stdout)
//        boardhistory_fuzz hashes <numGames> <seed0>  one line per game: "H <g> <bytes> <fnv1a64 hex>"
//        boardhistory_fuzz komiedge                   komi/draw edge cases of getSituationRulesAndKoHash
// Build (from the repo root, output outside the repo; upstream untouched; asserts on):
//   U=upstream/KataGo/cpp; g++ -std=c++17 -O2 -I$U -o /tmp/boardhistory_fuzz \
//     fixtures/upstream-v1.16.5/unit/oracle/boardhistory_fuzz.cpp \
//     $U/core/global.cpp $U/core/hash.cpp $U/core/rand.cpp $U/core/sha2.cpp $U/core/md5.cpp \
//     $U/core/timer.cpp $U/core/bsearch.cpp $U/game/board.cpp $U/game/rules.cpp \
//     $U/game/boardhistory.cpp $U/game/graphhash.cpp -lpthread
//   (rules.cpp includes ../external/nlohmann_json/json.hpp relative to itself, so -I$U suffices.)
//   cd fixtures/upstream-v1.16.5/unit/oracle
//   /tmp/boardhistory_fuzz hashes 3000 1 > boardhistory_fuzz_hashes.txt
//   /tmp/boardhistory_fuzz games 40 7 > boardhistory_fuzz_trace.txt
//   /tmp/boardhistory_fuzz komiedge > boardhistory_fuzz_komiedge.txt
// Generated with g++ (GCC) 16.2.1 20260810 / glibc on x86_64 Linux.
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
#include "game/rules.h"
#include "game/boardhistory.h"
#include "game/graphhash.h"
using namespace std;

static uint64_t S;
static uint64_t nxt() { uint64_t z = (S += 0x9E3779B97F4A7C15ULL); z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL; z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL; return z ^ (z >> 31); }
static int rnd(int n) { return (int)(nxt() % (uint64_t)n); }

static string out;
static void P(const string& s) { out += s; }
static string I(long long x) { return to_string(x); }
static string B(bool b) { return b ? "1" : "0"; }
static string F(float f) { ostringstream ss; ss << f; return ss.str(); }
static string FX(float f) { uint32_t u; memcpy(&u, &f, 4); char buf[16]; snprintf(buf, sizeof(buf), "%08x", u); return string(buf); }
static string FF(float f) { return F(f) + "/" + FX(f); }
static string HX(Hash128 h) { return h.toString(); }
static string HS(Hash128 h) { return Global::uint64ToHexString(h.hash0 ^ h.hash1); }

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

static uint64_t foldArr(const int8_t* a) {
  uint64_t h = 1469598103934665603ULL;
  for(int loc = 0; loc < Board::MAX_ARR_SIZE; loc++) { h ^= (uint64_t)(long long)a[loc]; h *= 1099511628211ULL; }
  return h;
}
static uint64_t foldBool(const bool* a) {
  uint64_t h = 1469598103934665603ULL;
  for(int loc = 0; loc < Board::MAX_ARR_SIZE; loc++) { h ^= (uint64_t)(a[loc] ? 1 : 0); h *= 1099511628211ULL; }
  return h;
}

// Folds of every BoardHistory vector and per-location array that `dump` does not print in full:
// koHashHistory, hashesBeforeBlackPass/WhitePass, koCapturesInEncore (hash, loc, pla), moveHistory with
// preventEncoreHistory, koRecapBlocked/superKoBanned over all MAX_ARR_SIZE locations (off-board ones too),
// currentRecentBoardIdx; then the full board fold of every recent board and of initialBoard.
static uint64_t FH = 0;
static void mixh(long long v) { FH ^= (uint64_t)v; FH *= 1099511628211ULL; }
static string xfold(const BoardHistory& hist) {
  FH = 1469598103934665603ULL;
  for(size_t i = 0; i < hist.koHashHistory.size(); i++) { mixh((long long)hist.koHashHistory[i].hash0); mixh((long long)hist.koHashHistory[i].hash1); }
  mixh(-1);
  for(size_t i = 0; i < hist.hashesBeforeBlackPass.size(); i++) { mixh((long long)hist.hashesBeforeBlackPass[i].hash0); mixh((long long)hist.hashesBeforeBlackPass[i].hash1); }
  mixh(-2);
  for(size_t i = 0; i < hist.hashesBeforeWhitePass.size(); i++) { mixh((long long)hist.hashesBeforeWhitePass[i].hash0); mixh((long long)hist.hashesBeforeWhitePass[i].hash1); }
  mixh(-3);
  for(size_t i = 0; i < hist.koCapturesInEncore.size(); i++) {
    mixh((long long)hist.koCapturesInEncore[i].posHashBeforeMove.hash0);
    mixh((long long)hist.koCapturesInEncore[i].posHashBeforeMove.hash1);
    mixh(hist.koCapturesInEncore[i].moveLoc);
    mixh(hist.koCapturesInEncore[i].movePla);
  }
  mixh(-4);
  for(size_t i = 0; i < hist.moveHistory.size(); i++) { mixh(hist.moveHistory[i].loc); mixh(hist.moveHistory[i].pla); mixh(hist.preventEncoreHistory[i] ? 1 : 0); }
  mixh(-5);
  for(int loc = 0; loc < Board::MAX_ARR_SIZE; loc++) { mixh(hist.koRecapBlocked[loc] ? 1 : 0); mixh(hist.superKoBanned[loc] ? 1 : 0); }
  mixh(hist.currentRecentBoardIdx);
  string s = " xf" + I((long long)(FH >> 1));
  for(int k = 0; k < BoardHistory::NUM_RECENT_BOARDS; k++)
    s += "," + I((long long)(fold(hist.getRecentBoard(k)) >> 1));
  s += "," + I((long long)(fold(hist.initialBoard) >> 1));
  return s;
}

static string areaStr(const Board& b, const Color* r) {
  string s;
  for(int y = 0; y < b.y_size; y++) for(int x = 0; x < b.x_size; x++) s += (char)('0' + r[Location::getLoc(x,y,b.x_size)]);
  return s;
}

static const float KOMIS[] = {0.0f, 0.5f, 6.5f, 7.0f, 7.5f, -3.0f, 1.5f, 100.5f, -150.0f, 150.0f, 2.0f, -0.5f, 0.25f, 3.3f, -7.1f, 1e9f, -1e9f, 0.1f};
static const int NUM_KOMIS = 18;
static const double DRAWS[] = {0.0, 0.5, 0.7, 1.0};
static const double DRAWS2[] = {0.25, 0.3, -1.0, 2.0, 0.1, 0.9999};

// One state line ("S") plus one legality line ("L").
// The graph hashes chained through a game's dumps (reset by runGame): GH at repBound 1..4, GH2 at
// repBound 0..22, GH11 at repBound 11 (the search default) with another draw utility.
static Hash128 GH;
static Hash128 GH2;
static Hash128 GH11;

static void dump(const BoardHistory& hist, const Board& board, const KoHashTable* table, const string& tag) {
  Player np = hist.presumedNextMovePla;
  string s = "S " + tag + st(board);
  s += " np" + I(np);
  s += " ep" + I(hist.encorePhase) + " nt" + I(hist.numTurnsThisPhase) + " na" + I(hist.numApproxValidTurnsThisPhase);
  s += " nc" + I(hist.numConsecValidTurnsThisGame) + " cp" + I(hist.consecutiveEndingPasses);
  s += " fin" + B(hist.isGameFinished) + " w" + I(hist.winner) + " sc" + FF(hist.finalWhiteMinusBlackScore);
  s += " isS" + B(hist.isScored) + " nr" + B(hist.isNoResult) + " rs" + B(hist.isResignation);
  s += " pe" + B(hist.isPastNormalPhaseEnd) + " bt" + B(hist.hasButton);
  s += " wb" + FF(hist.whiteBonusScore) + " wh" + FF(hist.whiteHandicapBonusScore) + " km" + FF(hist.rules.komi);
  s += " k";
  for(Player p = 1; p <= 2; p++)
    for(int d = 0; d < 4; d++)
      s += "," + FX(hist.currentSelfKomi(p, DRAWS[d]));
  s += " adj";
  for(int d = 0; d < 4; d++)
    s += "," + FX(hist.whiteKomiAdjustmentForDraws(DRAWS[d]));
  s += " kh" + HX(hist.koHashHistory[hist.koHashHistory.size()-1]) + "," + I(hist.koHashHistory.size()) + "," + I(hist.firstTurnIdxWithKoHistory);
  s += " nko" + I(hist.numberOfKoHashOccurrencesInHistory(hist.koHashHistory[hist.koHashHistory.size()-1], NULL));
  s += "," + I(hist.numberOfKoHashOccurrencesInHistory(hist.koHashHistory[hist.koHashHistory.size()-1], table));
  s += " rk" + HX(hist.koRecapBlockHash);
  s += " hp" + I(hist.hashesBeforeBlackPass.size()) + "," + I(hist.hashesBeforeWhitePass.size()) + " ek" + I(hist.koCapturesInEncore.size());
  s += " mh" + I(hist.moveHistory.size()) + "," + I(hist.preventEncoreHistory.size());
  s += " h1" + HX(BoardHistory::getSituationRulesAndKoHash(board, hist, P_BLACK, 0.5));
  s += "," + HX(BoardHistory::getSituationRulesAndKoHash(board, hist, P_WHITE, 0.5));
  s += "," + HS(BoardHistory::getSituationRulesAndKoHash(board, hist, np, 0.0));
  s += "," + HS(BoardHistory::getSituationRulesAndKoHash(board, hist, np, 1.0));
  s += " h2" + HX(BoardHistory::getSituationAndSimpleKoHash(board, np));
  s += " h3" + HX(BoardHistory::getSituationAndSimpleKoAndPrevPosHash(board, hist, np));
  s += " rb";
  for(int k = 0; k < BoardHistory::NUM_RECENT_BOARDS; k++)
    s += "," + HS(hist.getRecentBoard(k).pos_hash);
  s += " ib" + HS(hist.initialBoard.pos_hash) + "," + I(hist.initialPla) + "," + I(hist.initialEncorePhase) + "," + I(hist.initialTurnNumber);
  s += " se" + I((long long)(foldArr(hist.secondEncoreStartColors) >> 1));
  s += " wo" + I((long long)(foldBool(hist.wasEverOccupiedOrPlayed) >> 1));
  s += " bw" + B(hist.hasBlackPassOrWhiteFirst()) + " tn" + I(hist.getCurrentTurnNumber());
  s += " hc" + I(hist.computeNumHandicapStones()) + "," + I(hist.computeWhiteHandicapBonus()) + "," + I(BoardHistory::numHandicapStonesOnBoard(board));
  s += "," + B(hist.assumeMultipleStartingBlackMovesAreHandicap) + B(hist.whiteHasMoved) + "," + I(hist.overrideNumHandicapStones);
  s += " pw";
  for(Player p = 1; p <= 2; p++)
    s += B(hist.passWouldEndPhase(board, p)) + B(hist.passWouldEndGame(board, p)) + B(hist.shouldSuppressEndGameFromFriendlyPass(board, p));
  s += B(hist.isFinalPhase());
  // GraphHash (game/graphhash.cpp): state hashes, and a graph hash chained through every dump of the game.
  s += " gs" + HX(GraphHash::getStateHash(hist, np, 0.5)) + "," + HS(GraphHash::getStateHash(hist, getOpp(np), 0.0));
  GH = GraphHash::getGraphHash(GH, hist, np, 1 + (int)(hist.moveHistory.size() % 4), 0.5);
  s += " gh" + HX(GH);
  GH2 = GraphHash::getGraphHash(GH2, hist, np, (int)(hist.moveHistory.size() % 23), 0.5);
  GH11 = GraphHash::getGraphHash(GH11, hist, np, 11, 0.3);
  s += "," + HX(GH2) + "," + HX(GH11);
  s += xfold(hist);
  s += " k2";
  for(Player p = 1; p <= 2; p++)
    for(int d = 0; d < 6; d++)
      s += "," + FX(hist.currentSelfKomi(p, DRAWS2[d])) + HS(BoardHistory::getSituationRulesAndKoHash(board, hist, p, DRAWS2[d]));
  P(s + "\n");

  string l = "L";
  for(Player p = 1; p <= 2; p++) {
    string lg, tl, pk;
    int offLegal = 0;
    int offTol = 0;
    for(int loc = 0; loc < Board::MAX_ARR_SIZE; loc++) {
      bool a = hist.isLegal(board, (Loc)loc, p);
      bool t = hist.isLegalTolerant(board, (Loc)loc, p);
      if(loc != Board::PASS_LOC && !board.isOnBoard((Loc)loc)) {
        if(a) offLegal++;
        if(t) offTol++;
      }
    }
    for(int y = 0; y < board.y_size; y++) for(int x = 0; x < board.x_size; x++) {
      Loc loc = Location::getLoc(x,y,board.x_size);
      lg += B(hist.isLegal(board, loc, p));
      tl += B(hist.isLegalTolerant(board, loc, p));
      pk += B(hist.isPassForKo(board, loc, p));
    }
    lg += B(hist.isLegal(board, Board::PASS_LOC, p));
    tl += B(hist.isLegalTolerant(board, Board::PASS_LOC, p));
    l += " p" + I(p) + ":" + lg + "," + tl + "," + pk + "," + I(offLegal) + "," + I(offTol);
  }
  string kr, sk;
  for(int y = 0; y < board.y_size; y++) for(int x = 0; x < board.x_size; x++) {
    Loc loc = Location::getLoc(x,y,board.x_size);
    kr += B(hist.koRecapBlocked[loc]);
    sk += B(hist.superKoBanned[loc]);
  }
  l += " kr" + kr + " sk" + sk;
  P(l + "\n");
}

static void runGame(int g, uint64_t seed0) {
  GH = Hash128();
  GH2 = Hash128();
  GH11 = Hash128();
  int sizes[][2] = {{1,1},{2,2},{3,3},{2,3},{4,1},{5,5},{4,4},{7,7},{9,9},{5,7},{6,4},{3,5},{13,13},{19,19},{8,3},{3,2},{6,6},{2,1},{1,2},{19,1},{10,10},{11,11},{1,7},{19,2}};
  int numSizes = 24;
  S = seed0 * 1000003ULL + g;
  int si = rnd(numSizes);
  int xs = sizes[si][0], ys = sizes[si][1];
  Board board(xs,ys);
  int area = xs*ys;

  Rules rules;
  rules.koRule = rnd(4);
  rules.scoringRule = rnd(2);
  rules.taxRule = rnd(3);
  rules.multiStoneSuicideLegal = rnd(2) == 1;
  rules.whiteHandicapBonusRule = rnd(3);
  rules.friendlyPassOk = rnd(2) == 1;
  int ki = rnd(NUM_KOMIS);
  rules.komi = KOMIS[ki];
  int ePhase = 0;
  if(rules.scoringRule == Rules::SCORING_TERRITORY) {
    int e = rnd(4);
    ePhase = e <= 1 ? 0 : e - 1;
  }
  // Upstream's clear() reads the OLD encorePhase member for hasButton, and a pass with the button in the
  // encore trips assert(encorePhase == 0): only allow the button when the game starts in the main phase.
  int bt = rnd(4);
  rules.hasButton = bt == 0 && ePhase == 0;
  int pl = rnd(2);
  Player pla = pl == 0 ? P_BLACK : P_WHITE;

  // Non-empty initial board sometimes
  int setup = rnd(10);
  if(setup < 3 && area > 1) {
    int ns = rnd(area / 2 + 1);
    for(int k = 0; k < ns; k++) {
      int lx = rnd(xs);
      int ly = rnd(ys);
      int c = rnd(3);
      Color col = c == 0 ? C_BLACK : c == 1 ? C_WHITE : C_BLACK;
      board.setStoneFailIfNoLibs(Location::getLoc(lx,ly,xs), col);
    }
  }

  BoardHistory hist(board, pla, rules, ePhase);
  int o1 = rnd(3);
  if(o1 == 0) {
    int tn = rnd(50);
    hist.setInitialTurnNumber(tn - 5);
  }
  int o2 = rnd(3);
  if(o2 == 0)
    hist.setAssumeMultipleStartingBlackMovesAreHandicap(true);
  int o3 = rnd(6);
  if(o3 == 0) {
    int n = rnd(5);
    hist.setOverrideNumHandicapStones(n - 1);
  }
  KoHashTable tableStore;
  KoHashTable* table = NULL;
  int useTable = rnd(3);
  if(useTable == 0) {
    tableStore.recompute(hist);
    table = &tableStore;
  }
  int passProb = rnd(25) + 2;
  int preventMode = rnd(4);
  int koPref = rnd(90);
  int maxSteps = area * 3 + 10;
  if(maxSteps > 400) maxSteps = 400;
  int resurrections = 0;
  bool cleared = false;
  // Replaying moveHistory from copyToInitial() reproduces the game only if no out-of-band state change
  // (resignation, endAndScoreGameNow / endGameIfAllPassAlive taking the button, clear()) happened;
  // otherwise a replayed encore pass-for-ko can land on an occupied point in the main phase (C++ UB).
  bool replayable = true;

  P("G " + I(g) + " " + I(xs) + " " + I(ys) + " " + I(pla) + " " + I(ePhase) + " ");
  ostringstream rs; rs << rules;
  P(rs.str() + " " + rules.toJsonString() + " t" + I(table != NULL) + " pp" + I(passProb) + " pm" + I(preventMode) + st(board) + "\n");
  dump(hist, board, table, "init");

  bool lastWasPass = false;
  for(int n = 0; n < maxSteps; n++) {
    if(hist.isGameFinished || hist.isPastNormalPhaseEnd) {
      int cont = rnd(3);
      if(cont == 0 || resurrections >= 8)
        break;
      resurrections++;
    }
    Player mp = hist.presumedNextMovePla;
    int wrong = rnd(25);
    if(wrong == 0)
      mp = getOpp(mp);
    bool pe = false;
    if(preventMode == 0) {
      int x = rnd(2);
      pe = x == 0;
    }
    Loc loc = Board::PASS_LOC;
    int r = rnd(100);
    int effPass = passProb + (lastWasPass ? 45 : 0);
    if(r >= effPass) {
      vector<Loc> legal, koMoves, illegal;
      for(int y = 0; y < ys; y++) for(int x = 0; x < xs; x++) {
        Loc l = Location::getLoc(x,y,xs);
        if(hist.isLegal(board, l, mp)) {
          legal.push_back(l);
          if(board.wouldBeKoCapture(l, mp) || hist.isPassForKo(board, l, mp))
            koMoves.push_back(l);
        }
        else if(board.isLegalIgnoringKo(l, mp, true) || hist.isPassForKo(board, l, mp))
          illegal.push_back(l);
      }
      int r2 = rnd(100);
      if((r2 < 6 || (wrong == 0 && r2 < 50)) && illegal.size() > 0)
        loc = illegal[rnd((int)illegal.size())];
      else if(r2 < 6 + koPref && koMoves.size() > 0)
        loc = koMoves[rnd((int)koMoves.size())];
      else if(legal.size() > 0)
        loc = legal[rnd((int)legal.size())];
    }
    int kind = rnd(10);
    string ms = "M " + I(n) + " " + I(mp) + " " + I(loc) + " k" + I(kind) + " pe" + B(pe);
    if(kind <= 6) {
      hist.makeBoardMoveAssumeLegal(board, loc, mp, table, pe);
    }
    else if(kind <= 8) {
      int rl = rnd(5);
      if(rl == 0) {
        int z = rnd(Board::MAX_ARR_SIZE);
        loc = (Loc)z;
      }
      bool ok = hist.makeBoardMoveTolerant(board, loc, mp);
      ms += " tol" + I(loc) + "," + B(ok);
    }
    else {
      bool ok = hist.makeBoardMoveTolerant(board, loc, mp, pe);
      ms += " tolp" + B(ok);
    }
    lastWasPass = loc == Board::PASS_LOC;
    P(ms + "\n");
    dump(hist, board, table, I(n));

    int x1 = rnd(40);
    if(x1 == 0) {
      int k2 = rnd(NUM_KOMIS);
      hist.setKomi(KOMIS[k2]);
      P("KOMI " + FF(KOMIS[k2]) + " " + FF(hist.finalWhiteMinusBlackScore) + " " + I(hist.winner) + "\n");
    }
    int x2 = rnd(200);
    if(x2 == 0) {
      int w = rnd(2);
      hist.setWinnerByResignation(w == 0 ? P_BLACK : P_WHITE);
      replayable = false;
      dump(hist, board, table, "resign");
    }
    int x3 = rnd(80);
    if(x3 == 0) {
      hist.endAndScoreGameNow(board);
      replayable = false;
      dump(hist, board, table, "endscore");
    }
    int x4 = rnd(60);
    if(x4 == 0) {
      hist.endGameIfAllPassAlive(board);
      replayable = false;
      dump(hist, board, table, "allpassalive");
    }
    int x5 = rnd(12);
    if(x5 == 0) {
      Color ar[Board::MAX_ARR_SIZE];
      hist.getAreaNow(board, ar);
      P("AREA " + areaStr(board, ar) + "\n");
      BoardHistory copy(hist);
      Color ar2[Board::MAX_ARR_SIZE];
      copy.endAndScoreGameNow(board, ar2);
      P("ESC " + FF(copy.finalWhiteMinusBlackScore) + " " + I(copy.winner) + " " + B(copy.hasButton) + " " + FF(copy.whiteBonusScore) + " " + areaStr(board, ar2) + "\n");
    }
    int x6 = rnd(25);
    if(x6 == 0) {
      ostringstream o;
      hist.printDebugInfo(o, board);
      hist.printBasicInfo(o, board);
      P(o.str());
    }
    // Copy-assignment round trip: operator= onto a different history, then back onto hist.
    int y1 = rnd(20);
    if(y1 == 0) {
      BoardHistory tmp = hist.copyToInitial();
      tmp = hist;
      hist = tmp;
      dump(hist, board, table, "assign");
    }
    // getGraphHashFromScratch replays moveHistory from copyToInitial() (see `replayable`).
    int y2 = rnd(10);
    if(y2 == 0 && replayable && !hist.isGameFinished) {
      int rb = rnd(15);
      Hash128 gfs = GraphHash::getGraphHashFromScratch(hist, hist.presumedNextMovePla, rb, 0.5);
      P("GFS " + I(rb) + " " + HX(gfs) + "\n");
    }
    int x7 = rnd(15);
    if(x7 == 0 && replayable) {
      BoardHistory h2 = hist.copyToInitial();
      Board b2 = h2.getRecentBoard(0);
      for(size_t i = 0; i < hist.moveHistory.size(); i++)
        h2.makeBoardMoveAssumeLegal(b2, hist.moveHistory[i].loc, hist.moveHistory[i].pla, NULL, hist.preventEncoreHistory[i]);
      P("REPLAY eq" + B(b2.pos_hash == board.pos_hash) + "\n");
      dump(h2, b2, NULL, "replay");
    }
    if(table != NULL) {
      int x8 = rnd(20);
      if(x8 == 0) {
        tableStore.recompute(hist);
        P("RECOMPUTE " + I(tableStore.size()) + "\n");
      }
    }
    if(!cleared) {
      int x9 = rnd(60);
      if(x9 == 0) {
        cleared = true;
        replayable = false;
        int ne = 0;
        if(hist.rules.scoringRule == Rules::SCORING_TERRITORY && !hist.rules.hasButton) {
          int e = rnd(3);
          ne = e;
        }
        hist.clear(board, hist.presumedNextMovePla, hist.rules, ne);
        table = NULL;
        dump(hist, board, table, "clear" + I(ne));
      }
    }
  }
  dump(hist, board, table, "end");
}

// Komi edge cases for getSituationRulesAndKoHash's `(int64_t)(selfKomi*256.0f)` (C++ UB out of range / NaN;
// x86-64 cvttss2si gives INT64_MIN) and for draw-utility values that make selfKomi NaN/inf. Floats are
// given as bit patterns so both sides use the exact same values.
static const uint32_t EDGE_KOMI_BITS[] = {
  0x40f00000u, // 7.5
  0x40e00000u, // 7.0
  0x7fc00000u, // NaN
  0x7f800000u, // +inf
  0xff800000u, // -inf
  0x5bb1a2bcu, // 1e17
  0xdbb1a2bcu, // -1e17
  0x5c000000u, // 2^57: *256 = 2^65, out of range
  0x5a000000u, // 2^53: *256 = 2^61, in range
  0x5b000000u, // 2^55: *256 = 2^63, first value out of range
  0x5affffffu, // 2^55 - 2^31: *256 = 2^63 - 2^39, last value in range
  0xdb000000u, // -2^55: *256 = -2^63, in range (INT64_MIN either way)
  0xdb000001u, // -(2^55 + 2^32): *256 below -2^63, out of range
  0x5b0e1bcau, // 4e16
  0x4e6e6b28u, // 1e9
  0x3dcccccdu, // 0.1
};
static const uint64_t EDGE_DRAW_BITS[] = {
  0x3fe0000000000000ull, // 0.5
  0x0000000000000000ull, // 0.0
  0x3fe6666666666666ull, // 0.7
  0x7ff8000000000000ull, // NaN
  0x7ff0000000000000ull, // +inf
  0xfff0000000000000ull, // -inf
  0x7e37e43c8800759cull, // 1e300
};

static void komiEdge() {
  Rules rulesList[2] = { Rules::getTrompTaylorish(), Rules::getSimpleTerritory() };
  for(int ri = 0; ri < 2; ri++) {
    Board board(9,9);
    if(ri == 1) {
      board.setStone(Location::getLoc(2,2,9), C_BLACK);
      board.setStone(Location::getLoc(6,6,9), C_WHITE);
    }
    BoardHistory hist(board, P_BLACK, rulesList[ri], 0);
    for(size_t ki = 0; ki < sizeof(EDGE_KOMI_BITS)/sizeof(EDGE_KOMI_BITS[0]); ki++) {
      float k; memcpy(&k, &EDGE_KOMI_BITS[ki], 4);
      hist.setKomi(k);
      string s = "E r" + I(ri) + " k" + FX(k) + " int" + B(hist.rules.gameResultWillBeInteger());
      for(size_t di = 0; di < sizeof(EDGE_DRAW_BITS)/sizeof(EDGE_DRAW_BITS[0]); di++) {
        double d; memcpy(&d, &EDGE_DRAW_BITS[di], 8);
        for(Player p = 1; p <= 2; p++) {
          s += " " + FX(hist.currentSelfKomi(p, d));
          s += "," + HX(BoardHistory::getSituationRulesAndKoHash(board, hist, p, d));
          s += "," + HX(GraphHash::getStateHash(hist, p, d));
        }
      }
      P(s + "\n");
    }
  }
}

static uint64_t fnv1a64(const string& s) {
  uint64_t h = 1469598103934665603ULL;
  for(unsigned char c : s) { h ^= (uint64_t)c; h *= 1099511628211ULL; }
  return h;
}

int main(int argc, char** argv) {
  Board::initHash();
  string mode = argc > 1 ? string(argv[1]) : string("");
  if(mode == "komiedge") {
    komiEdge();
    fwrite(out.data(), 1, out.size(), stdout);
    return 0;
  }
  if(mode != "games" && mode != "hashes") {
    cerr << "usage: boardhistory_fuzz games|hashes <numGames> <seed0>" << endl;
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
