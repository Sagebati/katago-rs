// RNG-stream oracle for testboardbasic's runBoardReplayTest (tests/testboardbasic.cpp). The upstream test
// only asserts replay invariants, so its expected output says nothing about which moves were drawn. This
// program runs the same loop (verbatim, with a copy of PlayUtils::chooseRandomLegalMove so playutils.cpp's
// dependencies are not needed) and prints fingerprints of the Rand stream and of the chosen moves, which
// crates/katago-core/tests/testboardbasic.rs (board_replay_test) asserts.
// The loop calls board.isAdjacentToPla(moveLoc, ..) with moveLoc == PASS_LOC, which reads colors[-9]: an
// out-of-bounds read (UB, PORTING.md §6b). The real upstream function is called here; next to it, the
// variant that checks only adjacent indices >= 0 (what the Rust test does) is evaluated and any
// disagreement is counted ("mismatch", must be 0 for the Rust test to be a faithful replay).
// Output (one line): "reps <n> moves <n> passadj <calls> <true> mismatch <n> movefnv <hex> final <hex>"
//   moves    number of moveLoc selections (one per inner step)
//   passadj  isAdjacentToPla(PASS_LOC, ..) calls and how many returned true
//   movefnv  FNV-1a 64 over every selected moveLoc (as int64), in order
//   final    rand.nextUInt64() after the 6000 reps (any extra/missing draw changes it)
// Build (from the repo root, output outside the repo; upstream untouched; asserts on):
//   U=upstream/KataGo/cpp; g++ -std=c++17 -O2 -I$U -o /tmp/boardreplay_rng \
//     fixtures/upstream-v1.16.5/unit/oracle/boardreplay_rng.cpp \
//     $U/core/global.cpp $U/core/hash.cpp $U/core/rand.cpp $U/core/sha2.cpp $U/core/md5.cpp \
//     $U/core/timer.cpp $U/core/bsearch.cpp $U/game/board.cpp $U/game/rules.cpp \
//     $U/game/boardhistory.cpp $U/game/graphhash.cpp -lpthread
//   /tmp/boardreplay_rng > fixtures/upstream-v1.16.5/unit/oracle/boardreplay_rng.txt
// Generated with g++ (GCC) 16.2.1 20260810 / glibc on x86_64 Linux.
#include <cstdio>
#include <cstdint>
#include "core/global.h"
#include "core/rand.h"
#include "core/test.h"
#include "game/board.h"
#include "game/rules.h"
#include "game/boardhistory.h"
#include "game/graphhash.h"
using namespace std;

// PlayUtils::chooseRandomLegalMove (program/playutils.cpp), verbatim.
static Loc chooseRandomLegalMove(const Board& board, const BoardHistory& hist, Player pla, Rand& gameRand, Loc banMove) {
  int numLegalMoves = 0;
  Loc locs[Board::MAX_ARR_SIZE];
  testAssert(pla == hist.presumedNextMovePla);
  for(Loc loc = 0; loc < Board::MAX_ARR_SIZE; loc++) {
    if(hist.isLegal(board,loc,pla) && loc != banMove) {
      locs[numLegalMoves] = loc;
      numLegalMoves += 1;
    }
  }
  if(numLegalMoves > 0) {
    int n = gameRand.nextUInt(numLegalMoves);
    return locs[n];
  }
  return Board::NULL_LOC;
}

static long long numMoves = 0;
static long long passAdjCalls = 0;
static long long passAdjTrue = 0;
static long long mismatches = 0;
static uint64_t moveFnv = 1469598103934665603ULL;

static void noteMove(Loc loc) {
  numMoves++;
  moveFnv ^= (uint64_t)(long long)loc;
  moveFnv *= 1099511628211ULL;
}

// board.isAdjacentToPla(loc,pla) as upstream calls it, plus the in-bounds-only variant for comparison.
static bool isAdjacentToPlaInstrumented(const Board& board, Loc loc, Player pla) {
  bool upstream = board.isAdjacentToPla(loc,pla);
  bool inBounds = false;
  for(int i = 0; i < 4; i++) {
    int adj = (int)loc + board.adj_offsets[i];
    if(adj >= 0 && board.colors[adj] == pla)
      inBounds = true;
  }
  if(loc == Board::PASS_LOC || loc == Board::NULL_LOC) {
    passAdjCalls++;
    if(upstream)
      passAdjTrue++;
  }
  if(upstream != inBounds)
    mismatches++;
  return upstream;
}

int main() {
  Board::initHash();
  Rand rand("runBoardReplayTest");

  Board base = Board::parseBoard(9,5,R"%%(
.xo.o.ooo
xo.oooooo
.xooxxxo.
oxxxxx.xo
.o.xx.xo.
)%%");

  int reps = 0;
  for(int rep = 0; rep<6000; rep++) {
    Board board(base);
    Player pla = rand.nextBool(0.5) ? P_BLACK : P_WHITE;
    int initialEncorePhase = rand.nextBool(0.9) ? 0 : rand.nextBool(0.5) ? 1 : 2;
    Rules rules = rand.nextBool(0.8) ? Rules::parseRules("japanese") : Rules::parseRules("chinese");
    if(rules.scoringRule == Rules::SCORING_AREA)
      initialEncorePhase = 0;
    if(rand.nextBool(0.1))
      rules.koRule = rand.nextBool(0.5) ? Rules::KO_SITUATIONAL : Rules::KO_POSITIONAL;
    if(rand.nextBool(0.2))
      rules.taxRule = rand.nextBool(0.5) ? Rules::TAX_SEKI : rand.nextBool(0.5) ? Rules::TAX_NONE : Rules::TAX_ALL;
    BoardHistory hist(board,pla,rules,initialEncorePhase);
    hist.setInitialTurnNumber(rand.nextInt(0,40));
    hist.setAssumeMultipleStartingBlackMovesAreHandicap(rand.nextBool(0.5));

    double drawEquivalentWinsForWhite = 0.7;
    int repBound = rand.nextInt(1,5);

    Hash128 graphHash = GraphHash::getGraphHashFromScratch(hist, pla, repBound, drawEquivalentWinsForWhite);

    bool anyKomiSet = false;
    double passProb = rand.nextDouble(0.05,0.80);
    int numSteps = rand.nextInt(6,15);
    for(int i = 0; i<numSteps; i++) {
      BoardHistory tmpHist(board,pla,rules,hist.encorePhase);
      Loc moveLoc;
      if(rand.nextBool(passProb))
        moveLoc = Board::PASS_LOC;
      else {
        moveLoc = chooseRandomLegalMove(board,tmpHist,pla,rand,Board::NULL_LOC);
        //Resample to increase likelihood of capture moves.
        if(!board.wouldBeCapture(moveLoc,pla))
          moveLoc = chooseRandomLegalMove(board,tmpHist,pla,rand,Board::NULL_LOC);
        //Resample to decrease likelihood of moves not adjacent to opponent
        if(!isAdjacentToPlaInstrumented(board,moveLoc,getOpp(pla)))
          moveLoc = chooseRandomLegalMove(board,tmpHist,pla,rand,Board::NULL_LOC);
      }
      noteMove(moveLoc);
      bool preventEncore = rand.nextBool(0.5);
      if(rand.nextBool(0.5)) {
        bool suc = hist.makeBoardMoveTolerant(board, moveLoc, pla, preventEncore);
        testAssert(suc);
      }
      else {
        if(hist.isLegal(board,moveLoc,pla))
          hist.makeBoardMoveAssumeLegal(board, moveLoc, pla, NULL, preventEncore);
        else
          continue;
      }
      pla = getOpp(pla);
      if(rand.nextBool(0.1))
        pla = getOpp(pla);
      if(rand.nextBool(0.025)) {
        anyKomiSet = true;
        hist.setKomi(hist.rules.komi + (float)(rand.nextBool(0.5) ? -0.5 : +0.5));
      }

      graphHash = GraphHash::getGraphHash(graphHash, hist, pla, repBound, drawEquivalentWinsForWhite);
    }

    BoardHistory histCopy = hist.copyToInitial();
    Board boardCopy = histCopy.getRecentBoard(0);
    for(int i = 0; i<hist.moveHistory.size(); i++) {
      histCopy.makeBoardMoveAssumeLegal(boardCopy, hist.moveHistory[i].loc, hist.moveHistory[i].pla, NULL, hist.preventEncoreHistory[i]);
    }
    if(rand.nextBool(0.05))
      histCopy = hist;

    testAssert(boardCopy.isEqualForTesting(board));
    testAssert(boardCopy.isEqualForTesting(histCopy.getRecentBoard(0)));
    testAssert(histCopy.getRecentBoard(0).isEqualForTesting(hist.getRecentBoard(0)));
    testAssert(BoardHistory::getSituationRulesAndKoHash(boardCopy,histCopy,pla,drawEquivalentWinsForWhite) == hist.getSituationRulesAndKoHash(board,hist,pla,drawEquivalentWinsForWhite));
    testAssert(histCopy.currentSelfKomi(P_BLACK, drawEquivalentWinsForWhite) == hist.currentSelfKomi(P_BLACK, drawEquivalentWinsForWhite));
    testAssert(histCopy.currentSelfKomi(P_WHITE, drawEquivalentWinsForWhite) == hist.currentSelfKomi(P_WHITE, drawEquivalentWinsForWhite));
    testAssert(histCopy.initialTurnNumber == hist.initialTurnNumber);
    testAssert(histCopy.presumedNextMovePla == hist.presumedNextMovePla);
    testAssert(histCopy.assumeMultipleStartingBlackMovesAreHandicap == hist.assumeMultipleStartingBlackMovesAreHandicap);
    testAssert(GraphHash::getStateHash(histCopy, pla, drawEquivalentWinsForWhite) == GraphHash::getStateHash(hist, pla, drawEquivalentWinsForWhite));
    testAssert(GraphHash::getGraphHashFromScratch(histCopy, pla, repBound, drawEquivalentWinsForWhite) == GraphHash::getGraphHashFromScratch(hist, pla, repBound, drawEquivalentWinsForWhite));
    testAssert(anyKomiSet || graphHash == GraphHash::getGraphHashFromScratch(hist, pla, repBound, drawEquivalentWinsForWhite));
    reps++;
  }

  uint64_t fin = rand.nextUInt64();
  printf("reps %d moves %lld passadj %lld %lld mismatch %lld movefnv %s final %s\n",
         reps, numMoves, passAdjCalls, passAdjTrue, mismatches,
         Global::uint64ToHexString(moveFnv).c_str(), Global::uint64ToHexString(fin).c_str());
  return 0;
}
