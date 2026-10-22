// NN-tier oracle (M2): runs the UNMODIFIED upstream NNEvaluator + Eigen backend (fp32 CPU) of
// KataGo v1.16.5 on fixed positions and records every NNOutput field bit-exactly.
// The Rust consumer is crates/katago-nn-burn/tests/nn_parity.rs (ported NNEvaluator + burn backend).
//
// Modes:
//   nneval_eigen positions <outdir>
//       Deterministically generates the positions and eval requests: writes
//       <outdir>/positions.txt (pos lines), <outdir>/evals.txt (eval lines for the v8 nets) and
//       <outdir>/evals_b18.txt (eval lines for the v14 net, incl. policyOptimism > 0).
//   nneval_eigen eval <model.bin.gz> <positions.txt> <evals.txt> <out.bin>
//       Replays every pos line, evaluates every eval line in file order and writes one record per
//       eval line to <out.bin> (little-endian 32-bit words):
//         4 words nnHash (hash0 low, hash0 high, hash1 low, hash1 high)
//         10 f32: whiteWinProb whiteLossProb whiteNoResultProb whiteScoreMean whiteScoreMeanSq
//                 whiteLead varTimeLeft shorttermWinlossError shorttermScoreError policyOptimismUsed
//         362 f32: policyProbs[0..362) (19x19 policy size, pass last)
//         361 f32: whiteOwnerMap[0..361)
//       = 737 words per record.
//   nneval_eigen evalrand <model.bin.gz> <positions.txt> <evals_rand.txt> <out.bin>
//       The serve thread's symmetry choice and NNEvaluator's lifecycle, on one evaluator made by the
//       first line `config <randSeed> <doRandomize01> <defaultSymmetry>` (else as `eval`'s), then in
//       file order:
//         eval ...            evaluate (symmetry -1 = SYMMETRY_NOTSPECIFIED: the serve thread picks)
//         avg <eval fields> <numSymmetries> <randSeed>
//                             averageMultipleSymmetries(..., Rand(randSeed), numSymmetries)
//         setrandomize <01>   setDoRandomize      setdefaultsymmetry <s>   setDefaultSymmetry
//         respawn             killServerThreads(); spawnServerThreads(); (as testsearchv8.cpp does)
//       and writes one record per eval/avg line: 1 word buf.symmetry after the call (the symmetry
//       the serve thread used; for avg the last sampled one), then the 737 words of `eval`.
//   nneval_eigen gtp <positions.txt> <posId>
//       Prints the GTP commands (boardsize/komi/kata-set-rules/play) that set up that position, to
//       cross-check the oracle against `katago gtp` + `kata-raw-nn <sym>`.
//
// SHARED POSITION FORMAT (also fixtures/upstream-v1.16.5/unit/oracle/nninputs.cpp; colors 1 black
// 2 white; floats as uppercase hex bits):
//   pos <id> <xSize> <ySize> <koRule> <scoringRule> <taxRule> <suicide01> <button01> <whbRule>
//       <friendlyPassOk01> <komiF32Hex8> <encorePhase> <initialPla> <assumeMultiHandicap01>
//       <nSetup> {<x> <y> <color>}*nSetup <nMoves> {<x> <y> <pla> <preventEncore01>}*nMoves
//   Replay: Board(xSize,ySize); setStone(getLoc(x,y,xSize),color) for each setup stone;
//   Rules(ko,scoring,tax,suicide,button,whb,fpok,komi); BoardHistory(board,initialPla,rules,encorePhase);
//   setAssumeMultipleStartingBlackMovesAreHandicap(flag); each move: loc = (x == -1 ? PASS_LOC :
//   getLoc(x,y,xSize)), testAssert(hist.isLegal(board,loc,pla)), makeBoardMoveAssumeLegal(board,loc,pla,NULL,preventEncore).
//   eval <posId> <evalIdx> <nextPla> <drawEquivF64Hex16> <conservativePassAndIsRoot01>
//        <enablePassingHacks01> <playoutDoublingAdvantageF64Hex16> <nnPolicyTemperatureF32Hex8>
//        <avoidMYTDaggerHack01> <symmetry> <policyOptimismF64Hex16> <maxHistory> <nnXLen> <nnYLen>
//
// NNEvaluator settings: nnXLen = nnYLen = 19, requireExactNNLen false, inputsUseNHWC true (the
// Eigen backend requires it), nnCacheSizePowerOfTwo -1 (no cache) and skipCache, fp16 False,
// NHWC Auto, 1 server thread (gpuIdx -1), doRandomize false, defaultSymmetry 0, includeOwnerMap.
//
// Build (from the repo root, after `just katago-eigen`; links the Release objects of the Eigen
// reference build, i.e. the exact code of build/eigen/katago; upstream untouched):
//   B=build/eigen; U=upstream/KataGo/cpp; O=$B/CMakeFiles/katago.dir
//   g++ -std=gnu++17 -O2 -mfpmath=sse -DNO_GIT_REVISION -DUSE_EIGEN_BACKEND -I$U -I$U/external -I$B \
//     -isystem $U/external/filesystem-1.5.8/include -isystem build/deps/eigen \
//     -o /tmp/nneval_eigen fixtures/upstream-v1.16.5/nn/oracle/nneval_eigen.cpp \
//     $O/core/*.o $O/game/*.o $O/neuralnet/*.o $O/search/mutexpool.cpp.o -lz -lzip -lpthread
// Cross-checked against `build/eigen/katago gtp` + `kata-raw-nn <sym>` (gtpForceMaxNNSize=true,
// rootPolicyOptimism=0; setup from `nneval_eigen gtp`) on positions 7/31/40 (b6c96), 0/26/44
// (b10c128), 0/2/4/26/30 (b18c384nbt), all symmetries: every field agrees to GTP's printed
// precision (probabilities 5e-7, scores 5e-4, ownership 5e-8).
// Generate (from the repo root):
//   /tmp/nneval_eigen positions fixtures/upstream-v1.16.5/nn
//   M=/home/sam/src/web/katago-ws/bench/models; N=fixtures/upstream-v1.16.5/nn
//   /tmp/nneval_eigen eval $M/g170-b6c96.bin.gz $N/positions.txt $N/evals.txt $N/g170-b6c96.out.bin
//   /tmp/nneval_eigen eval $M/g170e-b10c128.bin.gz $N/positions.txt $N/evals.txt $N/g170e-b10c128.out.bin
//   /tmp/nneval_eigen eval $M/kata1-b18c384nbt.bin.gz $N/positions.txt $N/evals_b18.txt $N/kata1-b18c384nbt.out.bin
//   /tmp/nneval_eigen evalrand $M/g170-b6c96.bin.gz $N/positions.txt $N/evals_rand.txt $N/g170-b6c96.rand.out.bin
// Generated with g++ (GCC) 16.2.1 / glibc 2.44 on x86-64 (Intel i7-8700K, FMA), Eigen 3.4.0.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "core/global.h"
#include "core/test.h"
#include "game/board.h"
#include "game/boardhistory.h"
#include "game/rules.h"
#include "neuralnet/nneval.h"
#include "neuralnet/nninputs.h"
#include "neuralnet/nninterface.h"

using namespace std;

static uint64_t S = 0x2545F4914F6CDD1DULL;
static uint64_t nxt() {
  uint64_t z = (S += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}
static int rnd(int n) { return (int)(nxt() % (uint64_t)n); }

static string f32hex(float f) { uint32_t u; memcpy(&u, &f, 4); return Global::strprintf("%08X", u); }
static string f64hex(double d) { uint64_t u; memcpy(&u, &d, 8); return Global::strprintf("%016llX", (unsigned long long)u); }
static float hexf32(const string& s) { uint32_t u = (uint32_t)stoul(s, nullptr, 16); float f; memcpy(&f, &u, 4); return f; }
static double hexf64(const string& s) { uint64_t u = stoull(s, nullptr, 16); double d; memcpy(&d, &u, 8); return d; }

struct MoveSpec { int x; int y; int pla; int pe; };
struct PosSpec {
  int id;
  int xs, ys;
  Rules rules;
  int encorePhase;
  int initialPla;
  int amh;
  vector<MoveSpec> setup;
  vector<MoveSpec> moves;
};

static string posLine(const PosSpec& p) {
  ostringstream o;
  o << "pos " << p.id << " " << p.xs << " " << p.ys << " " << p.rules.koRule << " " << p.rules.scoringRule << " "
    << p.rules.taxRule << " " << (int)p.rules.multiStoneSuicideLegal << " " << (int)p.rules.hasButton << " "
    << p.rules.whiteHandicapBonusRule << " " << (int)p.rules.friendlyPassOk << " " << f32hex(p.rules.komi) << " "
    << p.encorePhase << " " << p.initialPla << " " << p.amh << " " << p.setup.size();
  for(const MoveSpec& m: p.setup)
    o << " " << m.x << " " << m.y << " " << m.pla;
  o << " " << p.moves.size();
  for(const MoveSpec& m: p.moves)
    o << " " << m.x << " " << m.y << " " << m.pla << " " << m.pe;
  return o.str();
}

static void replay(const PosSpec& p, Board& board, BoardHistory& hist) {
  board = Board(p.xs, p.ys);
  for(const MoveSpec& m: p.setup) {
    bool suc = board.setStone(Location::getLoc(m.x, m.y, p.xs), (Color)m.pla);
    testAssert(suc);
  }
  hist = BoardHistory(board, (Player)p.initialPla, p.rules, p.encorePhase);
  hist.setAssumeMultipleStartingBlackMovesAreHandicap(p.amh != 0);
  for(const MoveSpec& m: p.moves) {
    Loc loc = m.x == -1 ? Board::PASS_LOC : Location::getLoc(m.x, m.y, p.xs);
    testAssert(hist.isLegal(board, loc, (Player)m.pla));
    hist.makeBoardMoveAssumeLegal(board, loc, (Player)m.pla, NULL, m.pe != 0);
  }
}

static PosSpec parsePos(const string& line) {
  istringstream in(line);
  string tag; in >> tag; testAssert(tag == "pos");
  PosSpec p;
  int ko, sc, tax, suic, button, whb, fpok; string komi;
  in >> p.id >> p.xs >> p.ys >> ko >> sc >> tax >> suic >> button >> whb >> fpok >> komi >> p.encorePhase >> p.initialPla >> p.amh;
  p.rules = Rules(ko, sc, tax, suic != 0, button != 0, whb, fpok != 0, hexf32(komi));
  size_t n; in >> n;
  for(size_t i = 0; i < n; i++) { MoveSpec m; m.pe = 0; in >> m.x >> m.y >> m.pla; p.setup.push_back(m); }
  in >> n;
  for(size_t i = 0; i < n; i++) { MoveSpec m; in >> m.x >> m.y >> m.pla >> m.pe; p.moves.push_back(m); }
  testAssert(!in.fail());
  return p;
}

// ---------------------------------------------------------------------------------------------
// Position generation

// Plays one random legal move (never fills its own simple eye) or a pass with probability passPct/100.
static MoveSpec randomMove(const Board& board, const BoardHistory& hist, Player pla, int passPct) {
  vector<Loc> legal;
  for(int y = 0; y < board.y_size; y++)
    for(int x = 0; x < board.x_size; x++) {
      Loc loc = Location::getLoc(x, y, board.x_size);
      if(board.colors[loc] == C_EMPTY && hist.isLegal(board, loc, pla) && !board.isSimpleEye(loc, pla))
        legal.push_back(loc);
    }
  int r = rnd(100);
  if(legal.size() == 0 || r < passPct)
    return MoveSpec{-1, -1, pla, 0};
  Loc loc = legal[rnd((int)legal.size())];
  return MoveSpec{Location::getX(loc, board.x_size), Location::getY(loc, board.x_size), pla, 0};
}

static void push(PosSpec& p, Board& board, BoardHistory& hist, MoveSpec m) {
  Loc loc = m.x == -1 ? Board::PASS_LOC : Location::getLoc(m.x, m.y, board.x_size);
  testAssert(hist.isLegal(board, loc, (Player)m.pla));
  hist.makeBoardMoveAssumeLegal(board, loc, (Player)m.pla, NULL, m.pe != 0);
  p.moves.push_back(m);
}

static PosSpec start(int id, int xs, int ys, const Rules& rules, Board& board, BoardHistory& hist) {
  PosSpec p;
  p.id = id; p.xs = xs; p.ys = ys; p.rules = rules; p.encorePhase = 0; p.initialPla = P_BLACK; p.amh = 0;
  board = Board(xs, ys);
  hist = BoardHistory(board, P_BLACK, rules, 0);
  return p;
}

static PosSpec randomGame(int id, int xs, int ys, const Rules& rules, int numMoves, int passPct) {
  Board board; BoardHistory hist;
  PosSpec p = start(id, xs, ys, rules, board, hist);
  for(int i = 0; i < numMoves && !hist.isGameFinished; i++) {
    Player pla = hist.presumedNextMovePla;
    push(p, board, hist, randomMove(board, hist, pla, passPct));
  }
  return p;
}

// Random game until a simple ko is on the board (after at least minMoves moves).
static PosSpec koGame(int id, int xs, int ys, const Rules& rules, int minMoves) {
  for(int attempt = 0; attempt < 1000; attempt++) {
    Board board; BoardHistory hist;
    PosSpec p = start(id, xs, ys, rules, board, hist);
    for(int i = 0; i < xs * ys * 3 && !hist.isGameFinished; i++) {
      Player pla = hist.presumedNextMovePla;
      push(p, board, hist, randomMove(board, hist, pla, 0));
      if(i >= minMoves && board.ko_loc != Board::NULL_LOC)
        return p;
    }
  }
  testAssert(false);
  return PosSpec();
}

static Rules rulesNamed(const string& name, float komi) {
  Rules r = Rules::parseRules(name);
  r.komi = komi;
  return r;
}

static vector<PosSpec> sgfPositions(int& nextId) {
  const char* files[] = {
    "ogs-60000000.sgf", "ogs-60001931.sgf", "ogs-60023172.sgf", "ogs-60025103.sgf", "ogs-60046344.sgf", "ogs-60054068.sgf",
  };
  const int cuts[] = {20, 60, 120, 100000};
  vector<PosSpec> out;
  for(const char* f: files) {
    string path = string("/home/sam/src/web/katago-ws/bench/sgf-quick/") + f;
    ifstream in(path);
    testAssert(in.good());
    string text((istreambuf_iterator<char>(in)), istreambuf_iterator<char>());
    auto prop = [&](const string& key) -> string {
      size_t i = text.find("\n" + key + "[");
      if(i == string::npos) i = text.find(";" + key + "[");
      if(i == string::npos) return "";
      size_t s = text.find('[', i) + 1;
      return text.substr(s, text.find(']', s) - s);
    };
    string ru = Global::toLower(prop("RU"));
    float komi = Global::stringToFloat(prop("KM"));
    Rules rules = rulesNamed(ru == "japanese" ? "japanese" : "chinese", komi);
    vector<MoveSpec> setup;
    size_t ab = text.find("AB[");
    if(ab != string::npos) {
      size_t i = ab + 2;
      while(i < text.size() && text[i] == '[') {
        setup.push_back(MoveSpec{text[i + 1] - 'a', text[i + 2] - 'a', P_BLACK, 0});
        i += 4;
      }
    }
    vector<MoveSpec> moves;
    for(size_t i = 0; i + 3 < text.size(); i++) {
      if(text[i] == ';' && (text[i + 1] == 'B' || text[i + 1] == 'W') && text[i + 2] == '[') {
        int pla = text[i + 1] == 'B' ? P_BLACK : P_WHITE;
        if(text[i + 3] == ']')
          moves.push_back(MoveSpec{-1, -1, pla, 0});
        else
          moves.push_back(MoveSpec{text[i + 3] - 'a', text[i + 4] - 'a', pla, 0});
      }
    }
    for(int cut: cuts) {
      PosSpec p;
      p.id = nextId++; p.xs = 19; p.ys = 19; p.rules = rules; p.encorePhase = 0;
      p.initialPla = setup.size() > 0 ? P_WHITE : P_BLACK;
      p.amh = 0;
      p.setup = setup;
      Board board; BoardHistory hist;
      replay(p, board, hist);
      for(int i = 0; i < (int)moves.size() && i < cut && !hist.isGameFinished; i++) {
        Loc loc = moves[i].x == -1 ? Board::PASS_LOC : Location::getLoc(moves[i].x, moves[i].y, 19);
        if(!hist.isLegal(board, loc, (Player)moves[i].pla))
          break;
        push(p, board, hist, moves[i]);
      }
      out.push_back(p);
    }
  }
  return out;
}

static vector<PosSpec> allPositions() {
  vector<PosSpec> ps;
  int id = 0;
  Board board; BoardHistory hist;
  // Empty boards
  ps.push_back(start(id++, 19, 19, rulesNamed("japanese", 6.5f), board, hist));
  ps.push_back(start(id++, 19, 19, rulesNamed("chinese", 7.5f), board, hist));
  ps.push_back(start(id++, 19, 19, rulesNamed("tromp-taylor", 7.5f), board, hist));
  // Real games at several move numbers (handicap game as AB setup + white first)
  for(const PosSpec& p: sgfPositions(id))
    ps.push_back(p);
  // Random 19x19 games: after a pass, encore phases (territory scoring), a ko
  {
    PosSpec p = randomGame(id++, 19, 19, rulesNamed("chinese", 7.5f), 40, 0);
    Board b; BoardHistory h; replay(p, b, h);
    push(p, b, h, MoveSpec{-1, -1, h.presumedNextMovePla, 0});
    ps.push_back(p);
  }
  {
    PosSpec p = randomGame(id++, 19, 19, rulesNamed("japanese", 6.5f), 60, 0);
    Board b; BoardHistory h; replay(p, b, h);
    push(p, b, h, MoveSpec{-1, -1, h.presumedNextMovePla, 0});
    push(p, b, h, MoveSpec{-1, -1, h.presumedNextMovePla, 0});
    testAssert(h.encorePhase == 1);
    ps.push_back(p);
    PosSpec p2 = p; p2.id = id++;
    for(int i = 0; i < 4; i++) push(p2, b, h, randomMove(b, h, h.presumedNextMovePla, 0));
    push(p2, b, h, MoveSpec{-1, -1, h.presumedNextMovePla, 0});
    push(p2, b, h, MoveSpec{-1, -1, h.presumedNextMovePla, 0});
    for(int i = 0; i < 3 && !h.isGameFinished; i++) push(p2, b, h, randomMove(b, h, h.presumedNextMovePla, 0));
    ps.push_back(p2);
  }
  ps.push_back(koGame(id++, 19, 19, rulesNamed("japanese", 6.5f), 30));
  // preventEncore passes: past normal phase end
  {
    PosSpec p = randomGame(id++, 19, 19, rulesNamed("japanese", 6.5f), 50, 0);
    Board b; BoardHistory h; replay(p, b, h);
    push(p, b, h, MoveSpec{-1, -1, h.presumedNextMovePla, 1});
    push(p, b, h, MoveSpec{-1, -1, h.presumedNextMovePla, 1});
    ps.push_back(p);
  }
  // 9x9
  ps.push_back(koGame(id++, 9, 9, rulesNamed("japanese", 6.5f), 10));
  ps.push_back(randomGame(id++, 9, 9, rulesNamed("chinese", 7.5f), 30, 0));
  ps.push_back(randomGame(id++, 9, 9, rulesNamed("aga-button", 7.0f), 25, 0));
  ps.push_back(randomGame(id++, 9, 9, rulesNamed("tromp-taylor", 0.0f), 45, 5));
  ps.push_back(randomGame(id++, 9, 9, rulesNamed("stone-scoring", 7.5f), 35, 0));
  ps.push_back(randomGame(id++, 9, 9, rulesNamed("aga", 7.5f), 70, 10));
  // 13x13
  ps.push_back(randomGame(id++, 13, 13, rulesNamed("japanese", 6.5f), 50, 0));
  ps.push_back(randomGame(id++, 13, 13, rulesNamed("chinese", 13.0f), 80, 0));
  ps.push_back(randomGame(id++, 13, 13, rulesNamed("new-zealand", -3.5f), 30, 0));
  // 7x7, non-square
  ps.push_back(randomGame(id++, 7, 7, rulesNamed("chinese", 9.0f), 15, 0));
  ps.push_back(koGame(id++, 7, 7, rulesNamed("japanese", 0.5f), 5));
  ps.push_back(randomGame(id++, 9, 13, rulesNamed("chinese", 7.5f), 40, 0));
  ps.push_back(randomGame(id++, 13, 9, rulesNamed("japanese", 6.5f), 30, 0));
  ps.push_back(randomGame(id++, 19, 11, rulesNamed("korean", 0.5f), 60, 0));
  // Big random middlegames
  ps.push_back(randomGame(id++, 19, 19, rulesNamed("japanese", 6.5f), 150, 0));
  ps.push_back(randomGame(id++, 19, 19, rulesNamed("chinese", 7.5f), 250, 0));
  return ps;
}

struct EvalSpec {
  int posId, evalIdx, nextPla;
  MiscNNInputParams params;
};

static string evalLine(const EvalSpec& e) {
  const MiscNNInputParams& q = e.params;
  ostringstream o;
  o << "eval " << e.posId << " " << e.evalIdx << " " << e.nextPla << " " << f64hex(q.drawEquivalentWinsForWhite) << " "
    << (int)q.conservativePassAndIsRoot << " " << (int)q.enablePassingHacks << " " << f64hex(q.playoutDoublingAdvantage) << " "
    << f32hex(q.nnPolicyTemperature) << " " << (int)q.avoidMYTDaggerHack << " " << q.symmetry << " " << f64hex(q.policyOptimism)
    << " " << q.maxHistory << " 19 19";
  return o.str();
}

static EvalSpec parseEval(const string& line) {
  istringstream in(line);
  string tag; in >> tag; testAssert(tag == "eval");
  EvalSpec e;
  string dr, pda, temp, opt; int cpr, ph, dag, nnx, nny;
  in >> e.posId >> e.evalIdx >> e.nextPla >> dr >> cpr >> ph >> pda >> temp >> dag >> e.params.symmetry >> opt >> e.params.maxHistory >> nnx >> nny;
  testAssert(!in.fail());
  testAssert(nnx == 19 && nny == 19);
  e.params.drawEquivalentWinsForWhite = hexf64(dr);
  e.params.conservativePassAndIsRoot = cpr != 0;
  e.params.enablePassingHacks = ph != 0;
  e.params.playoutDoublingAdvantage = hexf64(pda);
  e.params.nnPolicyTemperature = hexf32(temp);
  e.params.avoidMYTDaggerHack = dag != 0;
  e.params.policyOptimism = hexf64(opt);
  return e;
}

// evals_rand.txt (no rnd() draws, so the files above do not change): symmetry -1 under
// doRandomize, explicit symmetries, defaultSymmetry, kill -> respawn (new server Rand),
// averageMultipleSymmetries, and passing hacks on game-end positions where the 95% pass cap binds.
// Random game played until both players pass (randomMove never fills its own simple eye, so it
// ends when only eyes are left), minus the final pass: the opponent just passed, every point is
// settled and the net's pass logit dominates.
static PosSpec endGame(int id, int xs, int ys, const Rules& rules) {
  PosSpec full = randomGame(id, xs, ys, rules, 100000, 0);
  Board board; BoardHistory hist;
  replay(full, board, hist);
  testAssert(hist.isGameFinished);
  testAssert(full.moves.size() >= 2 && full.moves.back().x == -1);
  full.moves.pop_back();
  return full;
}

static void genEvalsRand(const vector<PosSpec>& ps, const string& path) {
  ofstream ev(path);
  ev << "config oracleRandSeed 1 5\n";
  int evalIdx = 0;
  auto add = [&](const string& kind, int posId, MiscNNInputParams params, const string& extra) {
    Board board; BoardHistory hist;
    replay(ps[posId], board, hist);
    EvalSpec e{posId, evalIdx++, hist.presumedNextMovePla, params};
    string line = evalLine(e);
    ev << kind << line.substr(4) << extra << "\n";
  };
  auto sym = [](int s) { MiscNNInputParams p; p.symmetry = s; return p; };
  const int asym[] = {3, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24, 25, 27, 28, 29, 32, 33, 38, 39, 43, 44, 45, 46};
  const int numAsym = sizeof(asym) / sizeof(asym[0]);
  // The serve thread's Rand (":NNEvalServerThread:0") picks; explicit symmetries draw nothing.
  for(int k = 0; k < 10; k++)
    add("eval", asym[k], sym(k == 4 ? 6 : k == 7 ? 1 : -1), "");
  // doRandomize off: defaultSymmetry (5, then 2, then 7)
  ev << "setrandomize 0\n";
  add("eval", asym[10], sym(-1), "");
  add("eval", asym[11], sym(-1), "");
  ev << "setdefaultsymmetry 2\n";
  add("eval", asym[12], sym(-1), "");
  add("eval", asym[13], sym(4), "");
  ev << "setdefaultsymmetry 7\n";
  add("eval", asym[14], sym(-1), "");
  ev << "setrandomize 1\n";
  add("eval", asym[15], sym(-1), "");
  // kill -> respawn: ":NNEvalServerThread:1", then ":2"
  ev << "respawn\n";
  for(int k = 0; k < 8; k++)
    add("eval", asym[(16 + k) % numAsym], sym(-1), "");
  const int avgN[] = {1, 2, 3, 5, 8};
  for(int k = 0; k < 5; k++)
    add("avg", asym[(3 * k + 1) % numAsym], sym(-1), " " + Global::intToString(avgN[k]) + " avgSeed" + Global::intToString(k));
  ev << "respawn\n";
  for(int k = 0; k < 4; k++)
    add("eval", asym[(5 * k + 2) % numAsym], sym(-1), "");
  // Passing hacks on finished-game positions (endGame, appended to positions.txt; the pass logit
  // dominates and the cap of 19x the other moves' sum binds), explicit symmetry 0 and drawn, at
  // temperatures 1 and 0.3.
  vector<int> ends;
  for(const PosSpec& p: ps)
    if(p.id >= 48) ends.push_back(p.id);
  testAssert(ends.size() == 5);
  for(size_t k = 0; k < ends.size(); k++) {
    for(int t = 0; t < 2; t++) {
      MiscNNInputParams p = sym(t == 0 ? 0 : -1);
      p.enablePassingHacks = true;
      p.nnPolicyTemperature = t == 0 ? 1.0f : 0.3f;
      add("eval", ends[k], p, "");
    }
  }
  cout << evalIdx << " rand evals" << endl;
}

static int genPositions(const string& outdir) {
  vector<PosSpec> ps = allPositions();
  ofstream pos(outdir + "/positions.txt");
  ofstream ev(outdir + "/evals.txt");
  ofstream ev18(outdir + "/evals_b18.txt");
  int evalIdx = 0;
  int evalIdx18 = 0;
  for(const PosSpec& p: ps) {
    pos << posLine(p) << "\n";
    Board board; BoardHistory hist;
    replay(p, board, hist);
    int nextPla = hist.presumedNextMovePla;
    for(int sym = 0; sym < 8; sym++) {
      EvalSpec e{p.id, evalIdx++, nextPla, MiscNNInputParams()};
      e.params.symmetry = sym;
      ev << evalLine(e) << "\n";
    }
    // Every third position: the other MiscNNInputParams knobs, one at a time, at symmetry 3.
    if(p.id % 3 == 0) {
      for(int k = 0; k < 8; k++) {
        EvalSpec e{p.id, evalIdx++, nextPla, MiscNNInputParams()};
        e.params.symmetry = 3;
        if(k == 0) e.params.conservativePassAndIsRoot = true;
        if(k == 1) e.params.enablePassingHacks = true;
        if(k == 2) e.params.playoutDoublingAdvantage = 1.5;
        if(k == 3) e.params.nnPolicyTemperature = 0.75f;
        if(k == 4) e.params.avoidMYTDaggerHack = true;
        if(k == 5) e.params.drawEquivalentWinsForWhite = 0.2;
        if(k == 6) e.params.maxHistory = 0;
        if(k == 7) { e.params.playoutDoublingAdvantage = -0.625; e.params.nnPolicyTemperature = 1.3f; }
        ev << evalLine(e) << "\n";
      }
    }
    // b18 (v14, optimistic policy channel): every second position, 3 symmetry/optimism combos.
    if(p.id % 2 == 0) {
      const int syms[3] = {0, 6, 3};
      const double opts[3] = {0.0, 0.5, 1.0};
      for(int k = 0; k < 3; k++) {
        EvalSpec e{p.id, evalIdx18++, nextPla, MiscNNInputParams()};
        e.params.symmetry = syms[(k + p.id / 2) % 3];
        e.params.policyOptimism = opts[k];
        if(p.id % 6 == 0 && k == 1) e.params.enablePassingHacks = true;
        ev18 << evalLine(e) << "\n";
      }
    }
  }
  cout << ps.size() << " positions, " << evalIdx << " evals, " << evalIdx18 << " b18 evals" << endl;
  // Finished-game positions for evals_rand.txt only (appended, so evals.txt / evals_b18.txt and
  // their .out.bin records are unchanged).
  vector<PosSpec> all = ps;
  int id = (int)ps.size();
  all.push_back(endGame(id++, 7, 7, rulesNamed("chinese", 9.0f)));
  all.push_back(endGame(id++, 9, 9, rulesNamed("chinese", 7.5f)));
  all.push_back(endGame(id++, 9, 9, rulesNamed("tromp-taylor", 7.5f)));
  all.push_back(endGame(id++, 13, 13, rulesNamed("chinese", 7.5f)));
  all.push_back(endGame(id++, 19, 19, rulesNamed("chinese", 7.5f)));
  for(size_t i = ps.size(); i < all.size(); i++)
    pos << posLine(all[i]) << "\n";
  genEvalsRand(all, outdir + "/evals_rand.txt");
  return 0;
}

static vector<string> readLines(const string& path) {
  ifstream in(path);
  testAssert(in.good());
  vector<string> lines;
  string line;
  while(getline(in, line))
    if(line.size() > 0)
      lines.push_back(line);
  return lines;
}

static void putU32(vector<uint32_t>& out, uint32_t u) { out.push_back(u); }
static void putF32(vector<uint32_t>& out, float f) { uint32_t u; memcpy(&u, &f, 4); out.push_back(u); }

static int runEval(const string& modelFile, const string& posFile, const string& evalFile, const string& outFile) {
  vector<PosSpec> ps;
  for(const string& l: readLines(posFile)) ps.push_back(parsePos(l));
  vector<EvalSpec> es;
  for(const string& l: readLines(evalFile)) es.push_back(parseEval(l));

  vector<int> gpuIdxs = {-1};
  NNEvaluator* nnEval = new NNEvaluator(
    "oracle", modelFile, "", NULL,
    16, 19, 19, false, true, -1, 0, false, "", "", false,
    enabled_t::False, enabled_t::Auto, 1, gpuIdxs, "oracleSeed", false, 0
  );
  nnEval->spawnServerThreads();

  vector<uint32_t> out;
  for(const EvalSpec& e: es) {
    const PosSpec& p = ps[e.posId];
    testAssert(p.id == e.posId);
    Board board; BoardHistory hist;
    replay(p, board, hist);
    testAssert(e.nextPla == hist.presumedNextMovePla);
    NNResultBuf buf;
    nnEval->evaluate(board, hist, (Player)e.nextPla, e.params, buf, true, true);
    const NNOutput& o = *buf.result;
    putU32(out, (uint32_t)o.nnHash.hash0); putU32(out, (uint32_t)(o.nnHash.hash0 >> 32));
    putU32(out, (uint32_t)o.nnHash.hash1); putU32(out, (uint32_t)(o.nnHash.hash1 >> 32));
    putF32(out, o.whiteWinProb); putF32(out, o.whiteLossProb); putF32(out, o.whiteNoResultProb);
    putF32(out, o.whiteScoreMean); putF32(out, o.whiteScoreMeanSq); putF32(out, o.whiteLead);
    putF32(out, o.varTimeLeft); putF32(out, o.shorttermWinlossError); putF32(out, o.shorttermScoreError);
    putF32(out, o.policyOptimismUsed);
    testAssert(o.nnXLen == 19 && o.nnYLen == 19);
    for(int i = 0; i < 362; i++) putF32(out, o.policyProbs[i]);
    testAssert(o.whiteOwnerMap != NULL);
    for(int i = 0; i < 361; i++) putF32(out, o.whiteOwnerMap[i]);
  }
  nnEval->killServerThreads();
  delete nnEval;

  FILE* f = fopen(outFile.c_str(), "wb");
  testAssert(f != NULL);
  fwrite(out.data(), 4, out.size(), f);
  fclose(f);
  cout << es.size() << " evals -> " << outFile << endl;
  return 0;
}

static void putRecord(vector<uint32_t>& out, const NNOutput& o) {
  putU32(out, (uint32_t)o.nnHash.hash0); putU32(out, (uint32_t)(o.nnHash.hash0 >> 32));
  putU32(out, (uint32_t)o.nnHash.hash1); putU32(out, (uint32_t)(o.nnHash.hash1 >> 32));
  putF32(out, o.whiteWinProb); putF32(out, o.whiteLossProb); putF32(out, o.whiteNoResultProb);
  putF32(out, o.whiteScoreMean); putF32(out, o.whiteScoreMeanSq); putF32(out, o.whiteLead);
  putF32(out, o.varTimeLeft); putF32(out, o.shorttermWinlossError); putF32(out, o.shorttermScoreError);
  putF32(out, o.policyOptimismUsed);
  testAssert(o.nnXLen == 19 && o.nnYLen == 19);
  for(int i = 0; i < 362; i++) putF32(out, o.policyProbs[i]);
  testAssert(o.whiteOwnerMap != NULL);
  for(int i = 0; i < 361; i++) putF32(out, o.whiteOwnerMap[i]);
}

static int runEvalRand(const string& modelFile, const string& posFile, const string& evalFile, const string& outFile) {
  vector<PosSpec> ps;
  for(const string& l: readLines(posFile)) ps.push_back(parsePos(l));
  vector<string> lines = readLines(evalFile);
  istringstream cfg(lines[0]);
  string tag, seed; int doRandomize, defaultSymmetry;
  cfg >> tag >> seed >> doRandomize >> defaultSymmetry;
  testAssert(tag == "config" && !cfg.fail());

  vector<int> gpuIdxs = {-1};
  NNEvaluator* nnEval = new NNEvaluator(
    "oracle", modelFile, "", NULL,
    16, 19, 19, false, true, -1, 0, false, "", "", false,
    enabled_t::False, enabled_t::Auto, 1, gpuIdxs, seed, doRandomize != 0, defaultSymmetry
  );
  nnEval->spawnServerThreads();

  vector<uint32_t> out;
  int numEvals = 0;
  int numPassCapped = 0;
  NNResultBuf buf;
  for(size_t li = 1; li < lines.size(); li++) {
    istringstream in(lines[li]);
    string op; in >> op;
    if(op == "setrandomize") { int b; in >> b; nnEval->setDoRandomize(b != 0); continue; }
    if(op == "setdefaultsymmetry") { int s; in >> s; nnEval->setDefaultSymmetry(s); continue; }
    if(op == "respawn") { nnEval->killServerThreads(); nnEval->spawnServerThreads(); continue; }
    testAssert(op == "eval" || op == "avg");
    string rest; getline(in, rest);
    istringstream rin(rest);
    vector<string> toks; string t;
    while(rin >> t) toks.push_back(t);
    string evalPart = "eval";
    for(size_t i = 0; i < 14; i++) evalPart += " " + toks[i];
    EvalSpec e = parseEval(evalPart);
    const PosSpec& p = ps[e.posId];
    Board board; BoardHistory hist;
    replay(p, board, hist);
    testAssert(e.nextPla == hist.presumedNextMovePla);
    if(op == "eval") {
      testAssert(toks.size() == 14);
      nnEval->evaluate(board, hist, (Player)e.nextPla, e.params, buf, true, true);
      putU32(out, (uint32_t)buf.symmetry);
      putRecord(out, *buf.result);
      const NNOutput& o = *buf.result;
      if(e.params.enablePassingHacks) {
        int passPos = NNPos::locToPos(Board::PASS_LOC, board.x_size, 19, 19);
        double others = 0;
        for(int i = 0; i < passPos; i++) if(o.policyProbs[i] > 0) others += o.policyProbs[i];
        if(o.policyProbs[passPos] > 0.9499 && o.policyProbs[passPos] < 0.9501 && others > 0.0499) numPassCapped++;
      }
    }
    else {
      testAssert(toks.size() == 16);
      int numSym = Global::stringToInt(toks[14]);
      Rand rand(toks[15]);
      std::shared_ptr<NNOutput>* res = nnEval->averageMultipleSymmetries(board, hist, (Player)e.nextPla, NULL, e.params, buf, true, rand, numSym);
      putU32(out, (uint32_t)buf.symmetry);
      putRecord(out, **res);
      delete res;
    }
    numEvals++;
  }
  nnEval->killServerThreads();
  delete nnEval;

  FILE* f = fopen(outFile.c_str(), "wb");
  testAssert(f != NULL);
  fwrite(out.data(), 4, out.size(), f);
  fclose(f);
  cout << numEvals << " evals (" << numPassCapped << " with the pass cap binding) -> " << outFile << endl;
  return 0;
}

static int gtpSetup(const string& posFile, int posId) {
  vector<PosSpec> ps;
  for(const string& l: readLines(posFile)) ps.push_back(parsePos(l));
  const PosSpec& p = ps[posId];
  testAssert(p.encorePhase == 0 && !p.amh);
  if(p.xs == p.ys) cout << "boardsize " << p.xs << "\n";
  else cout << "rectangular_boardsize " << p.xs << " " << p.ys << "\n";
  cout << "kata-set-rules " << p.rules.toJsonStringNoKomi() << "\n";
  cout << "komi " << p.rules.komi << "\n";
  if(p.setup.size() > 0) {
    cout << "set_free_handicap";
    for(const MoveSpec& m: p.setup) cout << " " << Location::toString(Location::getLoc(m.x, m.y, p.xs), p.xs, p.ys);
    cout << "\n";
  }
  for(const MoveSpec& m: p.moves) {
    Loc loc = m.x == -1 ? Board::PASS_LOC : Location::getLoc(m.x, m.y, p.xs);
    cout << "play " << (m.pla == P_BLACK ? "B" : "W") << " " << Location::toString(loc, p.xs, p.ys) << "\n";
  }
  return 0;
}

int main(int argc, char** argv) {
  Board::initHash();
  ScoreValue::initTables();
  vector<string> args(argv + 1, argv + argc);
  if(args.size() == 2 && args[0] == "positions")
    return genPositions(args[1]);
  if(args.size() == 5 && args[0] == "eval")
    return runEval(args[1], args[2], args[3], args[4]);
  if(args.size() == 5 && args[0] == "evalrand")
    return runEvalRand(args[1], args[2], args[3], args[4]);
  if(args.size() == 3 && args[0] == "gtp")
    return gtpSetup(args[1], Global::stringToInt(args[2]));
  cerr << "usage: see header" << endl;
  return 1;
}
