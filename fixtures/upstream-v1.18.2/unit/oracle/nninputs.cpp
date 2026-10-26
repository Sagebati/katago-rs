// Features oracle for neuralnet/nninputs (M2: NN input features must be bit-exact). Links against the
// unmodified upstream sources of KataGo v1.18.2. The Rust mirror is
// crates/katago-core/tests/nninputs_oracle.rs (it only reads the files below; it does not regenerate).
// Every random draw is its own statement (no two rnd() calls in one expression: GCC argument order).
//
// Build (from the repo root, output outside the repo; upstream untouched; asserts on):
//   U=upstream/KataGo/cpp; g++ -std=c++17 -O2 -I$U -o /tmp/nninputs_oracle \
//     fixtures/upstream-v1.18.2/unit/oracle/nninputs.cpp \
//     $U/core/global.cpp $U/core/hash.cpp $U/core/rand.cpp $U/core/sha2.cpp $U/core/md5.cpp \
//     $U/core/timer.cpp $U/core/bsearch.cpp $U/game/board.cpp $U/game/rules.cpp \
//     $U/game/boardhistory.cpp $U/game/graphhash.cpp $U/neuralnet/nninputs.cpp -lpthread
// Run (writes the three files into the given directory; the SGFs are mainline-only OGS games):
//   /tmp/nninputs_oracle fixtures/upstream-v1.18.2/unit/oracle \
//     /home/sam/src/web/katago-ws/bench/sgf/*.sgf
// MUST run on x86-64 with FMA for nninputs_scorevalue.txt (glibc exp/atan/tan; D14).
// Generated with g++ (GCC) 16.2.1 / glibc 2.44 on x86_64 Linux (Intel i7-8700K).
//
// Two phases: (1) generate positions (random legal games on many board sizes and rules, plus
// positions sampled from SGF games) and their eval parameters, as text lines; (2) parse those text
// lines back, replay them, and compute the expected results. So the positions file is exactly what
// produced the expected file.
//
// ---- nninputs_positions.txt (shared POSITION FORMAT, also used by the NN tier) ----
// Players/colors: 1 = black, 2 = white. Floats are uppercase hex bit patterns (f32: 8 digits, f64: 16).
//   pos <id> <xSize> <ySize> <koRule> <scoringRule> <taxRule> <suicide01> <button01> <whbRule>
//       <friendlyPassOk01> <komiF32Hex8> <encorePhase> <initialPla> <assumeMultiHandicap01>
//       <nSetup> {<x> <y> <color>}*nSetup <nMoves> {<x> <y> <pla> <preventEncore01>}*nMoves
//   Replay: Board board(xSize,ySize); board.setStone(Location::getLoc(x,y,xSize), color) for each setup
//   stone; Rules rules(koRule,scoringRule,taxRule,suicide,button,whbRule,friendlyPassOk,komi);
//   BoardHistory hist(board, initialPla, rules, encorePhase);
//   hist.setAssumeMultipleStartingBlackMovesAreHandicap(flag); for each move: loc = x == -1 ? PASS_LOC :
//   Location::getLoc(x,y,xSize); hist.makeBoardMoveAssumeLegal(board, loc, pla, NULL, preventEncore).
//   Every move is hist.isLegal, except the initial consecutive black handicap moves of
//   assumeMultiHandicap games and the out-of-turn moves of the random games (main phase only, 1 in 5
//   turns: the presumed player's opponent plays a board-legal, non-superko-banned move, so the
//   history has consecutive moves by one player).
//   Positions: random games (27 board sizes incl. odd x even ones, random rules, encore 1/2 starts), the
//   hand-built encore ko fights of encoreKoFight (encore superKoBanned via koCapturesInEncore, ko
//   recapture blocks, pass-for-ko, a stone-symmetric board whose recapture blocks are not), SGF games.
//   eval <posId> <evalIdx> <nextPla> <drawEquivF64Hex16> <conservativePassAndIsRoot01>
//       <enablePassingHacks01> <playoutDoublingAdvantageF64Hex16> <nnPolicyTemperatureF32Hex8>
//       <avoidMYTDaggerHack01> <symmetry> <policyOptimismF64Hex16> <maxHistory> <nnXLen> <nnYLen>
//   (the MiscNNInputParams fields of the eval; symmetry only feeds copy*WithSymmetry here.)
//
// ---- nninputs_expected.txt ----
// FNV = FNV-1a 64 over bytes: h = 0xCBF29CE484222325; for each byte b: h ^= b; h *= 0x100000001B3.
// A float is folded as the 4 little-endian bytes of its bit pattern, a bool as the 4 bytes of 0/1, a
// double as the 8 little-endian bytes of its bit pattern.
//   res <posId> <evalIdx> <hash0> <hash1> <fnvNCHW> <fnvNHWC> <g0> .. <g18> <fnvSymInNCHW> <fnvSymInNHWC> <fnvSymOutMask>
//     hash = NNInputs::getHash; fnvNCHW / fnvNHWC = FNV of the 22*nnXLen*nnYLen rowBin floats filled
//     by fillRowV7 with useNHWC false / true; g<i> = rowGlobal[i] f32 bits, or "0" when the bits are 0;
//     fnvSymInNCHW = FNV of dst after copyInputsWithSymmetry(rowBinNCHW, dst, 1, nnYLen, nnXLen, 22, false,
//     sym) (dst zeroed first), fnvSymInNHWC the same from rowBinNHWC with useNHWC true, fnvSymOutMask =
//     FNV of dst after copyOutputsWithSymmetry(channel 0 plane of rowBinNCHW, dst, 1, nnYLen, nnXLen, sym);
//     sym = the eval's symmetry if >= 0, else 0.
//   posres <posId> <fnvScoring> <fnvScoringGroupTax> <validSyms> <fnvSymDup> <s1> <fnvDiffs1> <s2> <fnvDiffs2>
//     fnvScoring* = FNV of the MAX_ARR_SIZE floats of NNInputs::fillScoring(board, area, groupTax, scoring)
//     where area = board.calculateArea(area, true, true, true, rules.multiStoneSuicideLegal);
//     validSyms (comma list) and fnvSymDup (FNV of MAX_ARR_SIZE bools) from
//     SymmetryHelpers::markDuplicateMoveLocs(board, hist, NULL, {}, isSymDupLoc, validSymmetries);
//     fnvDiffs = FNV of the 8 doubles of getSymmetryDifferences(board, getSymBoard(board, s), 1e9, d), for
//     s1 = id%8 and s2 = (id*3+1)%8.
//   dump <posId> <evalIdx> <n> {<c> <pos> <bitsHex8>}*n   every nonzero rowBin entry of the NCHW fill
//     (index c*nnXLen*nnYLen + pos), for a subset of evals.
//
// ---- nninputs_scorevalue.txt ----  (after ScoreValue::initTables(); all values f64 hex bits)
//   <fn> <arg>... <result>, one call per line; fn names are the upstream ScoreValue function names.
//   whiteScoreValueOfScoreSmooth and whiteScoreDrawAdjust take the hist of "hist <k>" lines:
//   hist <k> <xSize> <ySize> <koRule> <scoringRule> <taxRule> <whbRule> <komiF32Hex8>
//   (BoardHistory(Board(xSize,ySize), P_BLACK, Rules(ko,score,tax,false,false,whb,false,komi), 0)).
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <vector>
#include <string>
#include "core/global.h"
#include "core/hash.h"
#include "core/test.h"
#include "game/board.h"
#include "game/rules.h"
#include "game/boardhistory.h"
#include "neuralnet/nninputs.h"
using namespace std;

static uint64_t S;
static uint64_t nxt() { uint64_t z = (S += 0x9E3779B97F4A7C15ULL); z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL; z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL; return z ^ (z >> 31); }
static int rnd(int n) { return (int)(nxt() % (uint64_t)n); }

static string I(long long x) { return to_string(x); }
static string X8(uint32_t u) { char b[16]; snprintf(b, sizeof(b), "%08X", u); return b; }
static string X16(uint64_t u) { char b[24]; snprintf(b, sizeof(b), "%016llX", (unsigned long long)u); return b; }
static uint32_t fbits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static uint64_t dbits(double d) { uint64_t u; memcpy(&u, &d, 8); return u; }
static float bitsf(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static double bitsd(uint64_t u) { double d; memcpy(&d, &u, 8); return d; }
static string FX(float f) { return X8(fbits(f)); }
static string DX(double d) { return X16(dbits(d)); }

static const uint64_t FNV0 = 0xCBF29CE484222325ULL;
static void fnv(uint64_t& h, uint32_t w) { for(int k = 0; k < 4; k++) { h ^= (uint64_t)((w >> (8 * k)) & 0xFF); h *= 0x100000001B3ULL; } }
static void fnv64(uint64_t& h, uint64_t w) { fnv(h, (uint32_t)w); fnv(h, (uint32_t)(w >> 32)); }
static uint64_t fnvFloats(const float* a, size_t n) { uint64_t h = FNV0; for(size_t i = 0; i < n; i++) fnv(h, fbits(a[i])); return h; }
static uint64_t fnvBools(const bool* a, size_t n) { uint64_t h = FNV0; for(size_t i = 0; i < n; i++) fnv(h, a[i] ? 1u : 0u); return h; }

//------------------------------------------------------------------------------------------------
// Phase 1: generation

struct MoveSpec { int x, y, pla, pe; };
struct PosSpec {
  int xs, ys, ko, sc, tax, suic, button, whb, fpok;
  float komi;
  int encore, initialPla, amh;
  vector<MoveSpec> setup; // x y color (pe unused)
  vector<MoveSpec> moves;
};

static vector<string> posLines;
static vector<string> evalLines;

static string posLine(int id, const PosSpec& p) {
  string s = "pos " + I(id) + " " + I(p.xs) + " " + I(p.ys) + " " + I(p.ko) + " " + I(p.sc) + " " + I(p.tax) + " " + I(p.suic) + " " +
    I(p.button) + " " + I(p.whb) + " " + I(p.fpok) + " " + FX(p.komi) + " " + I(p.encore) + " " + I(p.initialPla) + " " + I(p.amh);
  s += " " + I(p.setup.size());
  for(const MoveSpec& m : p.setup) s += " " + I(m.x) + " " + I(m.y) + " " + I(m.pla);
  s += " " + I(p.moves.size());
  for(const MoveSpec& m : p.moves) s += " " + I(m.x) + " " + I(m.y) + " " + I(m.pla) + " " + I(m.pe);
  return s;
}

static const double DRAWS[] = {0.5, 0.2, 0.0, 1.0, 0.37};
static const double PDAS[] = {0.0, 1.5, -2.25, 0.3};
static const float TEMPS[] = {1.0f, 1.25f, 0.7f};
static const double OPTS[] = {0.0, 0.5, 1.0};
static const int MAXHISTS[] = {1000, 0, 1, 3};

// Adds 1-4 evals for a position (generated on the live board/hist so nextPla can depend on it).
static void addEvals(int id, const PosSpec& p, Player presumed) {
  int numEvals = 1 + rnd(3);
  for(int e = 0; e < numEvals; e++) {
    int nextPla = presumed;
    int r = rnd(5);
    if(r == 0) nextPla = getOpp(presumed);
    // Mostly defaults, with each parameter varied independently some of the time.
    double draw = 0.5;
    if(rnd(3) == 0) draw = DRAWS[rnd(5)];
    int cpr = 0;
    if(rnd(4) == 0) cpr = 1;
    int ph = 0;
    if(rnd(4) == 0) ph = 1;
    double pda = 0.0;
    if(rnd(4) == 0) pda = PDAS[rnd(4)];
    float temp = 1.0f;
    if(rnd(4) == 0) temp = TEMPS[rnd(3)];
    int dagger = 0;
    if(rnd(5) == 0) dagger = 1;
    int sym = rnd(9) - 1;
    double opt = 0.0;
    if(rnd(4) == 0) opt = OPTS[rnd(3)];
    int maxHist = 1000;
    if(rnd(4) == 0) maxHist = MAXHISTS[rnd(4)];
    int nnX = p.xs;
    int nnY = p.ys;
    int rs = rnd(3);
    if(rs == 1) { nnX = 19; nnY = 19; }
    else if(rs == 2) {
      int ax = rnd(19 - p.xs + 1);
      int ay = rnd(19 - p.ys + 1);
      nnX = p.xs + ax;
      nnY = p.ys + ay;
    }
    string s = "eval " + I(id) + " " + I(e) + " " + I(nextPla) + " " + DX(draw) + " " + I(cpr) + " " + I(ph) + " " + DX(pda) + " " +
      FX(temp) + " " + I(dagger) + " " + I(sym) + " " + DX(opt) + " " + I(maxHist) + " " + I(nnX) + " " + I(nnY);
    evalLines.push_back(s);
  }
}

static int nextPosId = 0;
static void emit(const PosSpec& p, Player presumed) {
  int id = nextPosId++;
  posLines.push_back(posLine(id, p));
  addEvals(id, p, presumed);
}

struct RulesPreset { int ko, sc, tax, suic, button, whb, fpok; };
static const RulesPreset PRESETS[] = {
  {Rules::KO_SIMPLE, Rules::SCORING_TERRITORY, Rules::TAX_SEKI, 0, 0, Rules::WHB_ZERO, 1},          // japanese
  {Rules::KO_SIMPLE, Rules::SCORING_AREA, Rules::TAX_NONE, 0, 0, Rules::WHB_N, 1},                  // chinese
  {Rules::KO_POSITIONAL, Rules::SCORING_AREA, Rules::TAX_NONE, 1, 0, Rules::WHB_ZERO, 0},           // tromp-taylor
  {Rules::KO_SITUATIONAL, Rules::SCORING_AREA, Rules::TAX_NONE, 0, 0, Rules::WHB_N_MINUS_ONE, 0},   // aga
  {Rules::KO_SITUATIONAL, Rules::SCORING_AREA, Rules::TAX_NONE, 0, 1, Rules::WHB_N_MINUS_ONE, 0},   // aga button
  {Rules::KO_SPIGHT, Rules::SCORING_TERRITORY, Rules::TAX_SEKI, 0, 0, Rules::WHB_ZERO, 0},          // spight-ish
  {Rules::KO_SIMPLE, Rules::SCORING_TERRITORY, Rules::TAX_ALL, 0, 0, Rules::WHB_ZERO, 0},           // territory tax all
  {Rules::KO_POSITIONAL, Rules::SCORING_AREA, Rules::TAX_ALL, 1, 0, Rules::WHB_N, 0},               // stone scoring
  {Rules::KO_SIMPLE, Rules::SCORING_AREA, Rules::TAX_SEKI, 0, 0, Rules::WHB_ZERO, 1},               // area seki tax
  {Rules::KO_SITUATIONAL, Rules::SCORING_TERRITORY, Rules::TAX_NONE, 1, 0, Rules::WHB_N, 0},        // territory no tax
};
static const int NUM_PRESETS = 10;
static const float KOMIS[] = {7.5f, 6.5f, 0.0f, 0.5f, 7.0f, -3.0f, 5.5f, 2.0f, 1.5f, 9.5f, 100.0f, -100.0f, 0.25f, -7.5f, 3.0f, 1000.0f};
static const int NUM_KOMIS = 16;
static const int SIZES[][2] = {
  {9,9},{9,9},{9,9},{13,13},{13,13},{19,19},{19,19},{5,5},{7,7},{11,11},{9,13},{19,7},{13,9},{6,6},{3,3},{2,2},{4,7},{19,19},{8,10},{17,17},{15,15},{2,5},{9,10},{5,4},{7,12},{3,8},{19,18}
};
static const int NUM_SIZES = 27;

static void pickRules(PosSpec& p) {
  if(rnd(5) < 2) {
    const RulesPreset& r = PRESETS[rnd(NUM_PRESETS)];
    p.ko = r.ko; p.sc = r.sc; p.tax = r.tax; p.suic = r.suic; p.button = r.button; p.whb = r.whb; p.fpok = r.fpok;
  }
  else {
    p.ko = rnd(4);
    p.sc = rnd(2);
    p.tax = rnd(3);
    p.suic = rnd(2);
    p.button = rnd(4) == 0 ? 1 : 0;
    p.whb = rnd(3);
    p.fpok = rnd(2);
  }
  p.komi = KOMIS[rnd(NUM_KOMIS)];
}

static Loc locOf(const MoveSpec& m, int xs) { return m.x == -1 ? Board::PASS_LOC : Location::getLoc(m.x, m.y, xs); }

static Rules rulesOf(const PosSpec& p) {
  return Rules(p.ko, p.sc, p.tax, p.suic != 0, p.button != 0, p.whb, p.fpok != 0, p.komi);
}

// A random legal game; emits snapshots along it. local = prefer moves near the last few moves (fights,
// ataris, ladders).
static void randomGame(int g) {
  PosSpec p;
  int si = rnd(NUM_SIZES);
  p.xs = SIZES[si][0];
  p.ys = SIZES[si][1];
  pickRules(p);
  p.encore = 0;
  if(p.sc == Rules::SCORING_TERRITORY) {
    int e = rnd(12);
    if(e < 4) p.encore = 1;
    else if(e < 8) p.encore = 2;
  }
  // BoardHistory::clear reads the old encorePhase member for hasButton, so a button with a nonzero
  // starting encore phase trips an assert in makeBoardMoveAssumeLegal: avoid that combination.
  if(p.encore > 0) p.button = 0;
  p.initialPla = P_BLACK;
  if(rnd(6) == 0) p.initialPla = P_WHITE;
  p.amh = 0;

  Board board(p.xs, p.ys);
  // Setup stones (handicap-like): only ones that do not capture or leave no liberties.
  int setupKind = rnd(5);
  if(setupKind == 0 || setupKind == 1) {
    int n = 1 + rnd(setupKind == 0 ? 4 : 9);
    for(int i = 0; i < n; i++) {
      int x = rnd(p.xs);
      int y = rnd(p.ys);
      int c = P_BLACK;
      if(setupKind == 1 && rnd(3) == 0) c = P_WHITE;
      Loc loc = Location::getLoc(x, y, p.xs);
      if(board.colors[loc] != C_EMPTY) continue;
      Board copy(board);
      if(!copy.setStoneFailIfNoLibs(loc, (Color)c)) continue;
      board.setStone(loc, (Color)c);
      p.setup.push_back(MoveSpec{x, y, c, 0});
    }
    if(setupKind == 0 && p.setup.size() > 1) p.initialPla = P_WHITE;
  }
  BoardHistory hist(board, (Player)p.initialPla, rulesOf(p), p.encore, BoardHistoryModes());
  if(setupKind == 2 && p.initialPla == P_BLACK) {
    p.amh = 1;
    hist.setAssumeMultipleStartingBlackMovesAreHandicap(true);
    int n = 1 + rnd(4);
    for(int i = 0; i < n; i++) {
      int x = rnd(p.xs);
      int y = rnd(p.ys);
      Loc loc = Location::getLoc(x, y, p.xs);
      if(!board.isLegal(loc, P_BLACK, hist.rules.multiStoneSuicideLegal) || board.colors[loc] != C_EMPTY) continue;
      MoveSpec m{x, y, P_BLACK, 0};
      hist.makeBoardMoveAssumeLegal(board, loc, P_BLACK, NULL, false);
      p.moves.push_back(m);
    }
  }

  bool local = rnd(2) == 0;
  int passPermille = 10 + rnd(60);
  int preventEncorePermille = 0;
  if(rnd(4) == 0) preventEncorePermille = 300;
  int area = p.xs * p.ys;
  int maxMoves = area * (1 + rnd(3)) + 10;
  if(maxMoves > 200) maxMoves = 200;
  int snapDiv = 2 + rnd(3);
  int snapEvery = maxMoves / snapDiv;
  if(snapEvery < 3) snapEvery = 3;
  vector<Loc> legal;
  for(int t = 0; t < maxMoves; t++) {
    if(hist.isGameFinished) break;
    Player pla = hist.presumedNextMovePla;
    // Out-of-turn moves (main phase only): board-legal, not superko-banned, but not pla's presumed turn,
    // so the history has consecutive moves by the same player (the pla checks of fillRowV7's history).
    if(rnd(5) == 0) pla = getOpp(pla);
    legal.clear();
    vector<Loc> near;
    for(int y = 0; y < p.ys; y++) for(int x = 0; x < p.xs; x++) {
      Loc loc = Location::getLoc(x, y, p.xs);
      bool okL = pla == hist.presumedNextMovePla ? hist.isLegal(board, loc, pla) :
        (hist.encorePhase == 0 && board.colors[loc] == C_EMPTY && board.isLegal(loc, pla, hist.rules.multiStoneSuicideLegal) && !hist.superKoBanned[loc]);
      if(!okL) continue;
      legal.push_back(loc);
      if(local && board.colors[loc] == C_EMPTY) {
        bool adj = false;
        for(int i = 0; i < 8; i++) {
          Loc a = loc + board.adj_offsets[i];
          if(board.colors[a] == P_BLACK || board.colors[a] == P_WHITE) adj = true;
        }
        if(adj) near.push_back(loc);
      }
    }
    // Late in a long game, pass more.
    int pp = passPermille;
    if(t > area / 2) pp += 100;
    Loc loc = Board::PASS_LOC;
    int r = rnd(1000);
    if(legal.size() > 0 && r >= pp) {
      if(near.size() > 0 && rnd(10) < 7) loc = near[rnd((int)near.size())];
      else loc = legal[rnd((int)legal.size())];
    }
    int pe = 0;
    if(loc == Board::PASS_LOC && preventEncorePermille > 0 && rnd(1000) < preventEncorePermille) pe = 1;
    if(loc == Board::PASS_LOC && pla != hist.presumedNextMovePla) pla = hist.presumedNextMovePla;
    if(pla == hist.presumedNextMovePla && !hist.isLegal(board, loc, pla)) {
      pla = hist.presumedNextMovePla;
      loc = Board::PASS_LOC;
      if(!hist.isLegal(board, loc, pla)) { cerr << "illegal pass?" << endl; exit(1); }
    }
    MoveSpec m;
    m.x = loc == Board::PASS_LOC ? -1 : Location::getX(loc, p.xs);
    m.y = loc == Board::PASS_LOC ? -1 : Location::getY(loc, p.xs);
    m.pla = pla;
    m.pe = pe;
    hist.makeBoardMoveAssumeLegal(board, loc, pla, NULL, pe != 0);
    p.moves.push_back(m);
    if((t + 1) % snapEvery == 0 || hist.isGameFinished || (hist.encorePhase > 0 && rnd(12) == 0) || (board.ko_loc != Board::NULL_LOC && rnd(2) == 0) ||
       hist.isPastNormalPhaseEnd)
      emit(p, hist.presumedNextMovePla);
  }
  (void)g;
  emit(p, hist.presumedNextMovePla);
}

// Hand-built encore ko fight (territory scoring): a simple ko on top, and below it the mirror image (in y)
// of the same shape after the capture. Snapshots after each move:
//   1. capture: koRecapBlocked on the capturing stone; the stones are symmetric under the y flip but the
//      ko recapture blocks are not (markDuplicateMoveLocs's koRecapBlocked check, encore 1);
//   2. the opponent's pass-for-ko lifts the block; 3. a pass; 4. the recapture: the position before the
//      first capture repeats with the capturer to move, so koCapturesInEncore makes the capture
//      superKoBanned (feature 6 in the encore) and the recapturing stone is koRecapBlocked (feature 7);
//   5. the capturer's pass-for-ko; 6. a pass.
// swap exchanges the colors (white captures first); extraCols adds empty columns on the right.
static void encoreKoFight(int encore, bool swap, int extraCols) {
  PosSpec p;
  p.xs = 4 + extraCols;
  p.ys = 7;
  const RulesPreset& r = PRESETS[0];
  p.ko = r.ko; p.sc = r.sc; p.tax = r.tax; p.suic = r.suic; p.button = r.button; p.whb = r.whb; p.fpok = r.fpok;
  p.komi = 6.5f;
  p.encore = encore;
  int C = swap ? P_WHITE : P_BLACK;
  int O = swap ? P_BLACK : P_WHITE;
  p.initialPla = C;
  p.amh = 0;
  // top: before the capture; bottom (rows 4..6): the y-mirror of the top after the capture
  const int st[][3] = {
    {1,0,1},{2,0,2},{0,1,1},{1,1,2},{3,1,2},{1,2,1},{2,2,2},
    {1,6,1},{2,6,2},{0,5,1},{2,5,1},{3,5,2},{1,4,1},{2,4,2},
  };
  Board board(p.xs, p.ys);
  for(const auto& q : st) {
    int c = q[2] == 1 ? C : O;
    Loc loc = Location::getLoc(q[0], q[1], p.xs);
    testAssert(board.setStone(loc, (Color)c));
    p.setup.push_back(MoveSpec{q[0], q[1], c, 0});
  }
  BoardHistory hist(board, (Player)p.initialPla, rulesOf(p), p.encore, BoardHistoryModes());
  const int mv[][3] = {{2,1,1},{1,1,2},{-1,-1,1},{1,1,2},{2,1,1},{-1,-1,2}};
  for(const auto& q : mv) {
    int pla = q[2] == 1 ? C : O;
    MoveSpec m{q[0], q[1], pla, 0};
    Loc loc = locOf(m, p.xs);
    testAssert(hist.presumedNextMovePla == pla);
    testAssert(hist.isLegal(board, loc, (Player)pla));
    hist.makeBoardMoveAssumeLegal(board, loc, (Player)pla, NULL, false);
    p.moves.push_back(m);
    emit(p, hist.presumedNextMovePla);
  }
  testAssert(hist.encorePhase == encore);
  testAssert(hist.koRecapBlocked[Location::getLoc(1, 1, p.xs)] == false);
}

// Minimal SGF mainline reader: root properties SZ, KM, HA, AB, AW, RU; then B[]/W[] in textual order
// up to the first ')' (OGS writes the mainline as the first child at every node).
struct SgfGame { int size = 19; float komi = 7.5f; int ha = 0; string ru; vector<pair<int,int>> ab, aw; vector<MoveSpec> moves; };
static bool parseSgf(const string& text, SgfGame& g) {
  size_t i = text.find('(');
  if(i == string::npos) return false;
  i++;
  string prop;
  bool rootDone = false;
  int nodes = 0;
  while(i < text.size()) {
    char c = text[i];
    if(c == ')') break;
    if(c == ';') { nodes++; if(nodes > 1) rootDone = true; i++; prop.clear(); continue; }
    if(c >= 'A' && c <= 'Z') { prop += c; i++; continue; }
    if(c == '[') {
      size_t j = i + 1;
      string val;
      while(j < text.size() && text[j] != ']') {
        if(text[j] == '\\' && j + 1 < text.size()) { val += text[j+1]; j += 2; continue; }
        val += text[j];
        j++;
      }
      i = j + 1;
      if(!rootDone) {
        if(prop == "SZ") g.size = atoi(val.c_str());
        else if(prop == "KM") g.komi = (float)atof(val.c_str());
        else if(prop == "HA") g.ha = atoi(val.c_str());
        else if(prop == "RU") g.ru = val;
        else if(prop == "AB" && val.size() == 2) g.ab.push_back({val[0]-'a', val[1]-'a'});
        else if(prop == "AW" && val.size() == 2) g.aw.push_back({val[0]-'a', val[1]-'a'});
      }
      if(prop == "B" || prop == "W") {
        MoveSpec m;
        m.pla = prop == "B" ? P_BLACK : P_WHITE;
        m.pe = 0;
        if(val.size() != 2 || (val == "tt" && g.size <= 19)) { m.x = -1; m.y = -1; }
        else { m.x = val[0]-'a'; m.y = val[1]-'a'; }
        g.moves.push_back(m);
      }
      // A following [..] keeps the same property name (AB[aa][bb]).
      continue;
    }
    if(c == '(') prop.clear();
    i++;
  }
  return g.size >= 2 && g.size <= 19;
}

static void sgfPositions(const string& path, int gameIdx) {
  ifstream in(path);
  if(!in) { cerr << "cannot read " << path << endl; exit(1); }
  stringstream ss; ss << in.rdbuf();
  SgfGame g;
  if(!parseSgf(ss.str(), g)) { cerr << "skip " << path << endl; return; }
  PosSpec p;
  p.xs = g.size; p.ys = g.size;
  string ru = Global::toLower(g.ru);
  const RulesPreset* r = NULL;
  if(ru.find("japanese") != string::npos || ru.find("korean") != string::npos) r = &PRESETS[0];
  else if(ru.find("chinese") != string::npos) r = &PRESETS[1];
  else if(ru.find("aga") != string::npos) r = &PRESETS[3];
  else r = &PRESETS[(gameIdx * 7) % NUM_PRESETS];
  // Alternate the rules across games so each SGF exercises territory and area paths.
  if(gameIdx % 3 == 1) r = &PRESETS[0];
  if(gameIdx % 3 == 2) r = &PRESETS[1];
  p.ko = r->ko; p.sc = r->sc; p.tax = r->tax; p.suic = r->suic; p.button = r->button; p.whb = r->whb; p.fpok = r->fpok;
  p.komi = g.komi;
  p.encore = 0;
  p.amh = 0;
  Board board(p.xs, p.ys);
  for(auto& s : g.ab) { if(s.first < 0 || s.first >= p.xs || s.second < 0 || s.second >= p.ys) continue; board.setStone(Location::getLoc(s.first, s.second, p.xs), P_BLACK); p.setup.push_back(MoveSpec{s.first, s.second, P_BLACK, 0}); }
  for(auto& s : g.aw) { if(s.first < 0 || s.first >= p.xs || s.second < 0 || s.second >= p.ys) continue; board.setStone(Location::getLoc(s.first, s.second, p.xs), P_WHITE); p.setup.push_back(MoveSpec{s.first, s.second, P_WHITE, 0}); }
  p.initialPla = g.moves.size() > 0 ? g.moves[0].pla : P_BLACK;
  BoardHistory hist(board, (Player)p.initialPla, rulesOf(p), 0, BoardHistoryModes());
  int n = (int)g.moves.size();
  int numSnaps = 8 + rnd(6);
  vector<int> snaps;
  for(int k = 0; k < numSnaps; k++) snaps.push_back(rnd(n + 1));
  for(int t = 0; t <= n; t++) {
    for(int s : snaps) if(s == t) { emit(p, hist.presumedNextMovePla); break; }
    if(t == n || hist.isGameFinished) break;
    MoveSpec m = g.moves[t];
    if(m.x != -1 && (m.x < 0 || m.x >= p.xs || m.y < 0 || m.y >= p.ys)) continue;
    Loc loc = locOf(m, p.xs);
    if(!hist.isLegal(board, loc, (Player)m.pla)) continue;
    hist.makeBoardMoveAssumeLegal(board, loc, (Player)m.pla, NULL, false);
    p.moves.push_back(m);
  }
}

//------------------------------------------------------------------------------------------------
// Phase 2: replay from text and compute the expected results.

struct Replayed { PosSpec p; };

static PosSpec parsePos(const string& line, int& id) {
  istringstream in(line);
  string tag;
  in >> tag;
  testAssert(tag == "pos");
  PosSpec p;
  string komiHex;
  in >> id >> p.xs >> p.ys >> p.ko >> p.sc >> p.tax >> p.suic >> p.button >> p.whb >> p.fpok >> komiHex >> p.encore >> p.initialPla >> p.amh;
  p.komi = bitsf((uint32_t)stoul(komiHex, NULL, 16));
  size_t ns; in >> ns;
  for(size_t i = 0; i < ns; i++) { MoveSpec m; in >> m.x >> m.y >> m.pla; m.pe = 0; p.setup.push_back(m); }
  size_t nm; in >> nm;
  for(size_t i = 0; i < nm; i++) { MoveSpec m; in >> m.x >> m.y >> m.pla >> m.pe; p.moves.push_back(m); }
  testAssert(!in.fail());
  return p;
}

static void replay(const PosSpec& p, Board& board, BoardHistory& hist) {
  board = Board(p.xs, p.ys);
  for(const MoveSpec& m : p.setup) board.setStone(Location::getLoc(m.x, m.y, p.xs), (Color)m.pla);
  hist = BoardHistory(board, (Player)p.initialPla, rulesOf(p), p.encore, BoardHistoryModes());
  hist.setAssumeMultipleStartingBlackMovesAreHandicap(p.amh != 0);
  for(const MoveSpec& m : p.moves)
    hist.makeBoardMoveAssumeLegal(board, locOf(m, p.xs), (Player)m.pla, NULL, m.pe != 0);
}

static int64_t stats[32];

int main(int argc, char** argv) {
  if(argc < 2) { cerr << "usage: nninputs_oracle <outdir> [sgf files...]" << endl; return 1; }
  string outDir = argv[1];
  Board::initHash();
  ScoreValue::initTables();

  // Phase 1
  S = 0x5EED0001ULL;
  for(int g = 0; g < 450; g++)
    randomGame(g);
  int numRandomPos = nextPosId;
  for(int encore = 1; encore <= 2; encore++)
    for(int swap = 0; swap < 2; swap++)
      for(int extraCols = 0; extraCols < 3; extraCols += 2)
        encoreKoFight(encore, swap != 0, extraCols);
  int numKoPos = nextPosId - numRandomPos;
  for(int a = 2; a < argc; a++)
    sgfPositions(argv[a], a - 2);
  cerr << "positions: random " << numRandomPos << ", encore ko " << numKoPos << ", sgf " << (nextPosId - numRandomPos - numKoPos) << ", evals " << evalLines.size() << endl;

  {
    ofstream out(outDir + "/nninputs_positions.txt");
    size_t e = 0;
    for(size_t i = 0; i < posLines.size(); i++) {
      out << posLines[i] << "\n";
      while(e < evalLines.size()) {
        istringstream in(evalLines[e]);
        string tag; int pid; in >> tag >> pid;
        if(pid != (int)i) break;
        out << evalLines[e] << "\n";
        e++;
      }
    }
    testAssert(e == evalLines.size());
  }

  // Phase 2: re-read what was written.
  ifstream posIn(outDir + "/nninputs_positions.txt");
  ofstream out(outDir + "/nninputs_expected.txt");
  string line;
  Board board;
  BoardHistory hist;
  int curId = -1;
  PosSpec cur;
  int numEvals = 0;
  int numDumps = 0;
  while(getline(posIn, line)) {
    if(line.compare(0, 4, "pos ") == 0) {
      cur = parsePos(line, curId);
      replay(cur, board, hist);
      stats[0]++;
      if(hist.encorePhase == 1) stats[1]++;
      if(hist.encorePhase == 2) stats[2]++;
      if(hist.isGameFinished) stats[3]++;
      if(hist.isPastNormalPhaseEnd) stats[4]++;
      if(board.ko_loc != Board::NULL_LOC) stats[5]++;
      // Per-position results
      Color area[Board::MAX_ARR_SIZE];
      board.calculateArea(area, true, true, true, hist.rules.multiStoneSuicideLegal);
      float scoring[Board::MAX_ARR_SIZE];
      NNInputs::fillScoring(board, area, false, scoring);
      uint64_t fs0 = fnvFloats(scoring, Board::MAX_ARR_SIZE);
      NNInputs::fillScoring(board, area, true, scoring);
      uint64_t fs1 = fnvFloats(scoring, Board::MAX_ARR_SIZE);
      bool isSymDupLoc[Board::MAX_ARR_SIZE];
      vector<int> validSyms;
      vector<int> avoid;
      SymmetryHelpers::markDuplicateMoveLocs(board, hist, NULL, avoid, isSymDupLoc, validSyms);
      string vs;
      for(size_t i = 0; i < validSyms.size(); i++) vs += (i > 0 ? "," : "") + I(validSyms[i]);
      if(validSyms.size() > 1) stats[6]++;
      string s = "posres " + I(curId) + " " + X16(fs0) + " " + X16(fs1) + " " + vs + " " + X16(fnvBools(isSymDupLoc, Board::MAX_ARR_SIZE));
      int ss[2] = {curId % 8, (curId * 3 + 1) % 8};
      for(int k = 0; k < 2; k++) {
        Board sb = SymmetryHelpers::getSymBoard(board, ss[k]);
        double d[SymmetryHelpers::NUM_SYMMETRIES];
        SymmetryHelpers::getSymmetryDifferences(board, sb, 1e9, d);
        uint64_t hd = FNV0;
        for(int j = 0; j < SymmetryHelpers::NUM_SYMMETRIES; j++) fnv64(hd, dbits(d[j]));
        s += " " + I(ss[k]) + " " + X16(hd);
      }
      out << s << "\n";
      continue;
    }
    istringstream in(line);
    string tag; int pid, eidx, nextPla, cpr, ph, dagger, sym, maxHist, nnX, nnY;
    string drawH, pdaH, tempH, optH;
    in >> tag >> pid >> eidx >> nextPla >> drawH >> cpr >> ph >> pdaH >> tempH >> dagger >> sym >> optH >> maxHist >> nnX >> nnY;
    testAssert(tag == "eval" && !in.fail() && pid == curId);
    MiscNNInputParams params;
    params.drawEquivalentWinsForWhite = bitsd(stoull(drawH, NULL, 16));
    params.conservativePassAndIsRoot = cpr != 0;
    params.enablePassingHacks = ph != 0;
    params.playoutDoublingAdvantage = bitsd(stoull(pdaH, NULL, 16));
    params.nnPolicyTemperature = bitsf((uint32_t)stoul(tempH, NULL, 16));
    params.avoidMYTDaggerHack = dagger != 0;
    params.symmetry = sym;
    params.policyOptimism = bitsd(stoull(optH, NULL, 16));
    params.maxHistory = maxHist;
    const int C = NNInputs::NUM_FEATURES_SPATIAL_V7;
    const int G = NNInputs::NUM_FEATURES_GLOBAL_V7;
    size_t n = (size_t)C * nnX * nnY;
    vector<float> binN(n), binH(n), dst(n), plane(nnX * nnY), planeDst(nnX * nnY);
    float glob[G], globH[G];
    Hash128 h = NNInputs::getHash(board, hist, (Player)nextPla, params);
    NNInputs::fillRowV7(board, hist, (Player)nextPla, params, nnX, nnY, false, binN.data(), glob);
    NNInputs::fillRowV7(board, hist, (Player)nextPla, params, nnX, nnY, true, binH.data(), globH);
    testAssert(memcmp(glob, globH, sizeof(glob)) == 0);
    int s = sym >= 0 ? sym : 0;
    string r = "res " + I(pid) + " " + I(eidx) + " " + X16(h.hash0) + " " + X16(h.hash1) + " " + X16(fnvFloats(binN.data(), n)) + " " + X16(fnvFloats(binH.data(), n));
    for(int i = 0; i < G; i++) r += " " + (fbits(glob[i]) == 0 ? string("0") : FX(glob[i]));
    std::fill(dst.begin(), dst.end(), 0.0f);
    SymmetryHelpers::copyInputsWithSymmetry(binN.data(), dst.data(), 1, nnY, nnX, C, false, s);
    r += " " + X16(fnvFloats(dst.data(), n));
    std::fill(dst.begin(), dst.end(), 0.0f);
    SymmetryHelpers::copyInputsWithSymmetry(binH.data(), dst.data(), 1, nnY, nnX, C, true, s);
    r += " " + X16(fnvFloats(dst.data(), n));
    std::copy(binN.begin(), binN.begin() + nnX * nnY, plane.begin());
    std::fill(planeDst.begin(), planeDst.end(), 0.0f);
    SymmetryHelpers::copyOutputsWithSymmetry(plane.data(), planeDst.data(), 1, nnY, nnX, s);
    r += " " + X16(fnvFloats(planeDst.data(), nnX * nnY));
    out << r << "\n";
    numEvals++;
    // Coverage stats: channels in use.
    for(int c = 0; c < C; c++) {
      bool any = false;
      for(int pos = 0; pos < nnX * nnY; pos++) if(binN[c * nnX * nnY + pos] != 0.0f) any = true;
      if(any && c >= 6) stats[8 + c - 6]++;
    }
    for(int i = 0; i < G; i++) if(glob[i] != 0.0f) stats[24 + (i < 5 ? 0 : i < 14 ? 1 : i < 18 ? 2 : 3)]++;
    // Full dumps for a spread subset.
    if(numEvals % 71 == 3) {
      numDumps++;
      string d;
      int cnt = 0;
      for(size_t k = 0; k < n; k++) if(fbits(binN[k]) != 0) { cnt++; d += " " + I(k / (nnX * nnY)) + " " + I(k % (nnX * nnY)) + " " + FX(binN[k]); }
      out << "dump " << pid << " " << eidx << " " << cnt << d << "\n";
    }
  }
  cerr << "evals " << numEvals << ", dumps " << numDumps << ", positions " << stats[0] << ": encore1 " << stats[1] << " encore2 " << stats[2]
       << " finished " << stats[3] << " pastNormalEnd " << stats[4] << " koLoc " << stats[5] << " symmetric " << stats[6] << endl;
  cerr << "evals with nonzero channel 6..21:";
  for(int c = 6; c < 22; c++) cerr << " " << c << ":" << stats[8 + c - 6];
  cerr << endl << "global nonzero: hist " << stats[24] << " rules/komi " << stats[25] << " 14-17 " << stats[26] << " 18 " << stats[27] << endl;

  // ScoreValue
  ofstream sv(outDir + "/nninputs_scorevalue.txt");
  for(int w = 0; w <= 2; w++)
    for(double d : {0.5, 0.0, 1.0, 0.37})
      sv << "whiteWinsOfWinner " << w << " " << DX(d) << " " << DX(ScoreValue::whiteWinsOfWinner((Player)w, d)) << "\n";
  struct HistSpec { int xs, ys, ko, sc, tax, whb; float komi; };
  HistSpec hs[] = {
    {19,19,Rules::KO_SIMPLE,Rules::SCORING_AREA,Rules::TAX_NONE,Rules::WHB_N,7.5f},
    {19,19,Rules::KO_SIMPLE,Rules::SCORING_TERRITORY,Rules::TAX_SEKI,Rules::WHB_ZERO,7.0f},
    {9,9,Rules::KO_POSITIONAL,Rules::SCORING_AREA,Rules::TAX_NONE,Rules::WHB_ZERO,0.0f},
    {13,7,Rules::KO_SITUATIONAL,Rules::SCORING_AREA,Rules::TAX_ALL,Rules::WHB_N_MINUS_ONE,-3.0f},
  };
  for(int k = 0; k < 4; k++) {
    const HistSpec& q = hs[k];
    sv << "hist " << k << " " << q.xs << " " << q.ys << " " << q.ko << " " << q.sc << " " << q.tax << " " << q.whb << " " << FX(q.komi) << "\n";
    Board b(q.xs, q.ys);
    BoardHistory h(b, P_BLACK, Rules(q.ko, q.sc, q.tax, false, false, q.whb, false, q.komi), 0, BoardHistoryModes());
    for(double score : {-30.5, -7.0, 0.0, 0.5, 3.0, 12.5, 200.0}) {
      for(double draw : {0.5, 0.0, 1.0}) {
        sv << "whiteScoreDrawAdjust " << k << " " << DX(score) << " " << DX(draw) << " " << DX(ScoreValue::whiteScoreDrawAdjust(score, draw, h)) << "\n";
        for(double center : {0.0, 4.5}) for(double scale : {1.0, 0.5}) {
          double sqrtA = b.sqrtBoardArea();
          sv << "whiteScoreValueOfScoreSmooth " << k << " " << DX(score) << " " << DX(center) << " " << DX(scale) << " " << DX(draw) << " " << DX(sqrtA)
             << " " << DX(ScoreValue::whiteScoreValueOfScoreSmooth(score, center, scale, draw, sqrtA, h)) << "\n";
        }
      }
    }
  }
  for(double score : {-400.0, -50.25, -3.0, -0.5, 0.0, 1e-9, 2.5, 19.0, 361.0, 1e6})
    for(double center : {0.0, -2.0, 7.5})
      for(double scale : {1.0, 0.25, 3.0})
        for(double sqrtA : {19.0, 9.0, sqrt(63.0)}) {
          sv << "whiteScoreValueOfScoreSmoothNoDrawAdjust " << DX(score) << " " << DX(center) << " " << DX(scale) << " " << DX(sqrtA)
             << " " << DX(ScoreValue::whiteScoreValueOfScoreSmoothNoDrawAdjust(score, center, scale, sqrtA)) << "\n";
          sv << "whiteDScoreValueDScoreSmoothNoDrawAdjust " << DX(score) << " " << DX(center) << " " << DX(scale) << " " << DX(sqrtA)
             << " " << DX(ScoreValue::whiteDScoreValueDScoreSmoothNoDrawAdjust(score, center, scale, sqrtA)) << "\n";
        }
  for(double v : {-1.0, -0.999999, -0.9, -0.5, -0.1, 0.0, 1e-7, 0.3, 0.75, 0.99, 0.9999999, 1.0})
    for(double center : {0.0, 3.5})
      for(double scale : {1.0, 2.0})
        for(double sqrtA : {19.0, 13.0}) {
          sv << "approxWhiteScoreOfScoreValueSmooth " << DX(v) << " " << DX(center) << " " << DX(scale) << " " << DX(sqrtA)
             << " " << DX(ScoreValue::approxWhiteScoreOfScoreValueSmooth(v, center, scale, sqrtA)) << "\n";
        }
  for(double score : {-7.0, -6.5, 0.0, 0.5, 3.0, 12.5, 100.0})
    for(double draw : {0.5, 0.0, 1.0, 0.2})
      sv << "whiteScoreMeanSqOfScoreGridded " << DX(score) << " " << DX(draw) << " " << DX(ScoreValue::whiteScoreMeanSqOfScoreGridded(score, draw)) << "\n";
  for(double m : {-3.0, 0.0, 2.5, 10.0})
    for(double sq : {0.0, 4.0, 9.0, 110.0, 1000.0})
      sv << "getScoreStdev " << DX(m) << " " << DX(sq) << " " << DX(ScoreValue::getScoreStdev(m, sq)) << "\n";
  // expectedWhiteScoreValue over a grid incl. clamping (mean beyond the table, stdev beyond the table).
  S = 0x5C0E5EEDULL;
  int numExp = 0;
  for(double mean : {-1000.0, -400.0, -60.3, -12.0, -0.5, 0.0, 0.25, 3.7, 7.5, 33.3, 380.0, 450.0, 2000.0})
    for(double stdev : {0.0, 0.3, 1.0, 5.5, 20.0, 60.0, 300.0, 420.0, 1000.0})
      for(double center : {0.0, 6.5, -2.0})
        for(double scale : {1.0, 0.5})
          for(double sqrtA : {19.0, 9.0, sqrt(91.0)}) {
            sv << "expectedWhiteScoreValue " << DX(mean) << " " << DX(stdev) << " " << DX(center) << " " << DX(scale) << " " << DX(sqrtA)
               << " " << DX(ScoreValue::expectedWhiteScoreValue(mean, stdev, center, scale, sqrtA)) << "\n";
            numExp++;
          }
  // Dense random sample in the typical range.
  for(int i = 0; i < 1500; i++) {
    double mean = ((double)(nxt() >> 11) * 0x1.0p-53 - 0.5) * 120.0;
    double stdev = (double)(nxt() >> 11) * 0x1.0p-53 * 40.0;
    double sqrtA = 5.0 + (double)rnd(15);
    sv << "expectedWhiteScoreValue " << DX(mean) << " " << DX(stdev) << " " << DX(0.0) << " " << DX(1.0) << " " << DX(sqrtA)
       << " " << DX(ScoreValue::expectedWhiteScoreValue(mean, stdev, 0.0, 1.0, sqrtA)) << "\n";
    numExp++;
  }
  // Means whose (int)round(meanScaled) + svTableMeanRadius overflows int (x86-64 wraps it negative, so the
  // index clamps to 0), or whose cast is out of range / NaN (INT_MIN). sqrtA 19, scale 1: scaleFactor 1.
  for(double mean : {2147483640.0, 2147483647.0, 2147483226.6, 2147483000.0, -2147483640.0, 1e300, -1e300, std::numeric_limits<double>::quiet_NaN()})
    for(double stdev : {0.0, 5.5, 1000.0})
      for(double sqrtA : {19.0, 9.0}) {
        sv << "expectedWhiteScoreValue " << DX(mean) << " " << DX(stdev) << " " << DX(0.0) << " " << DX(1.0) << " " << DX(sqrtA)
           << " " << DX(ScoreValue::expectedWhiteScoreValue(mean, stdev, 0.0, 1.0, sqrtA)) << "\n";
        numExp++;
      }
  cerr << "scorevalue: expectedWhiteScoreValue " << numExp << endl;
  return 0;
}
