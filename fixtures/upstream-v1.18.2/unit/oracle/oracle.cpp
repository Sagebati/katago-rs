// Oracle generator for the katago-rs pilot (core/hash, core/rand, game/rules, game/board,
// port::fmt, port::nlohmann). Links against the unmodified upstream sources of KataGo v1.18.2.
// Usage: oracle <mode>   with mode in: rand hash board rules printf nlohmann
// Build (from the repo root, output outside the repo; upstream untouched; asserts on):
//   U=upstream/KataGo/cpp; g++ -std=c++17 -O1 -g -I$U -o /tmp/oracle oracle.cpp \
//     $U/core/global.cpp $U/core/hash.cpp $U/core/rand.cpp $U/core/sha2.cpp $U/core/md5.cpp \
//     $U/core/timer.cpp $U/core/bsearch.cpp $U/game/board.cpp $U/game/rules.cpp -lpthread
//   for m in rand hash board rules; do /tmp/oracle $m > $m.txt; done
//   /tmp/oracle printf > printf_glibc.txt; /tmp/oracle nlohmann > nlohmann_dump.txt
// Generated with g++ 16.2.1 / glibc on x86_64 Linux.
#include <cstdio>
#include <cstring>
#include <climits>
#include <cmath>
#include <iostream>
#include <sstream>
#include <vector>
#include <string>

#include "core/global.h"
#include "core/hash.h"
#include "core/rand.h"
#include "game/board.h"
#include "game/rules.h"
#include "external/nlohmann_json/json.hpp"

using namespace std;

static string dbits(double x) {
  uint64_t b; memcpy(&b, &x, 8);
  return Global::uint64ToHexString(b);
}

static void randMode() {
  vector<string> strSeeds = {"abc", "", "Board::initHash()", "seed with spaces and \xc3\xbcnicode", string(100, 'x'), "runBoardUndoTests"};
  vector<uint64_t> u64Seeds = {0ULL, 1ULL, 0xDEADBEEFCAFEBABEULL};
  int total = (int)strSeeds.size() + (int)u64Seeds.size();
  for(int si = 0; si < total; si++) {
    Rand* randp;
    if(si < (int)strSeeds.size()) {
      cout << "seedstr " << strSeeds[si] << "\n";
      randp = new Rand(strSeeds[si]);
    }
    else {
      cout << "seedu64 " << Global::uint64ToHexString(u64Seeds[si - strSeeds.size()]) << "\n";
      randp = new Rand(u64Seeds[si - strSeeds.size()]);
    }
    Rand& rand = *randp;
    cout << "getSeed " << rand.getSeed() << "\n";
    for(int i = 0; i<8; i++) cout << "nextUInt " << rand.nextUInt() << "\n";
    for(int i = 0; i<8; i++) cout << "nextUInt(27) " << rand.nextUInt(27) << "\n";
    for(int i = 0; i<2; i++) cout << "nextUInt(1) " << rand.nextUInt(1) << "\n";
    for(int i = 0; i<4; i++) cout << "nextUInt(0xFFFFFFFF) " << rand.nextUInt(0xFFFFFFFFu) << "\n";
    for(int i = 0; i<4; i++) cout << "nextInt " << rand.nextInt() << "\n";
    for(int i = 0; i<4; i++) cout << "nextInt(-8,8) " << rand.nextInt(-8,8) << "\n";
    for(int i = 0; i<2; i++) cout << "nextInt(INT_MIN,INT_MAX) " << rand.nextInt(INT_MIN,INT_MAX) << "\n";
    for(int i = 0; i<4; i++) cout << "nextUInt64 " << Global::uint64ToHexString(rand.nextUInt64()) << "\n";
    for(int i = 0; i<4; i++) cout << "nextUInt64(1000000007) " << rand.nextUInt64(1000000007ULL) << "\n";
    for(int i = 0; i<2; i++) cout << "nextUInt64(2^63+5) " << rand.nextUInt64((1ULL << 63) + 5) << "\n";
    for(int i = 0; i<4; i++) cout << "nextDouble " << dbits(rand.nextDouble()) << "\n";
    for(int i = 0; i<2; i++) cout << "nextDouble(12) " << dbits(rand.nextDouble(12)) << "\n";
    for(int i = 0; i<2; i++) cout << "nextDouble(-100,100) " << dbits(rand.nextDouble(-100,100)) << "\n";
    for(int i = 0; i<4; i++) cout << "nextBool(0.3) " << rand.nextBool(0.3) << "\n";
    for(int i = 0; i<5; i++) cout << "nextGaussian " << dbits(rand.nextGaussian()) << "\n";
    for(int i = 0; i<3; i++) cout << "nextGaussianTruncated(1.5) " << dbits(rand.nextGaussianTruncated(1.5)) << "\n";
    for(int i = 0; i<3; i++) cout << "nextExponential " << dbits(rand.nextExponential()) << "\n";
    for(int i = 0; i<3; i++) cout << "nextLogistic " << dbits(rand.nextLogistic()) << "\n";
    for(int i = 0; i<3; i++) cout << "nextGamma(0.5) " << dbits(rand.nextGamma(0.5)) << "\n";
    for(int i = 0; i<3; i++) cout << "nextGamma(1) " << dbits(rand.nextGamma(1.0)) << "\n";
    for(int i = 0; i<3; i++) cout << "nextGamma(4.5) " << dbits(rand.nextGamma(4.5)) << "\n";
    {
      int freq[5] = {1,0,3,7,2};
      for(int i = 0; i<4; i++) cout << "nextUInt(freq) " << rand.nextUInt(freq,5) << "\n";
      double relProbs[4] = {0.5,1.5,0.0,2.0};
      for(int i = 0; i<4; i++) cout << "nextUInt(relProbs) " << rand.nextUInt(relProbs,4) << "\n";
      double cum[4] = {1.0,3.0,3.5,7.0};
      for(int i = 0; i<4; i++) cout << "nextIndexCumulative " << rand.nextIndexCumulative(cum,4) << "\n";
    }
    {
      uint32_t buf[12];
      rand.fillShuffledUIntRange(12, buf);
      cout << "fillShuffledUIntRange(12)";
      for(int i = 0; i<12; i++) cout << " " << buf[i];
      cout << "\n";
      vector<int> v = {0,1,2,3,4,5,6,7,8,9};
      rand.shuffle(v);
      cout << "shuffle";
      for(int x: v) cout << " " << x;
      cout << "\n";
    }
    cout << "nextUInt " << rand.nextUInt() << "\n";
    delete randp;
  }
}

static void hashMode() {
  vector<uint64_t> xs = {0ULL, 1ULL, 12345ULL, 0x0123456789abcdefULL, 0xfefefefefefefefeULL, 0xFFFFFFFFFFFFFFFFULL};
  for(uint64_t x: xs) {
    cout << "x " << Global::uint64ToHexString(x)
         << " highBits " << Hash::highBits(x) << " lowBits " << Hash::lowBits(x)
         << " basicLCong " << Global::uint64ToHexString(Hash::basicLCong(x))
         << " basicLCong2 " << Global::uint64ToHexString(Hash::basicLCong2(x))
         << " murmurMix " << Global::uint64ToHexString(Hash::murmurMix(x))
         << " splitMix64 " << Global::uint64ToHexString(Hash::splitMix64(x))
         << " rrmxmx " << Global::uint64ToHexString(Hash::rrmxmx(x))
         << " nasam " << Global::uint64ToHexString(Hash::nasam(x)) << "\n";
  }
  cout << "combine " << Global::uint64ToHexString(Hash::combine(0xDEADBEEFu, 0x12345678u)) << "\n";
  uint32_t triples[4][3] = {{0,0,0},{1,2,3},{0xFFFFFFFFu,0x80000000u,7},{123456789u,987654321u,555u}};
  for(int i = 0; i<4; i++) {
    uint32_t a = triples[i][0], b = triples[i][1], c = triples[i][2];
    cout << "jenkinsMixSingle " << Hash::jenkinsMixSingle(a,b,c);
    Hash::jenkinsMix(a,b,c);
    cout << " jenkinsMix " << a << " " << b << " " << c << "\n";
  }
  vector<string> strs = {"", "a", "hello", "Board::initHash()", "\xc3\xa9t\xc3\xa9 \xe2\x82\xac", "The quick brown fox jumps over the lazy dog."};
  for(const string& s: strs) {
    cout << "simpleHash \"" << s << "\" " << Global::uint64ToHexString(Hash::simpleHash(s.c_str())) << "\n";
  }
  vector<vector<int>> ivs = {{}, {0}, {1,2,3}, {-1,-2,-3,2147483647,-2147483647-1}, {42,0,-7,100000,3}};
  for(const vector<int>& v: ivs) {
    cout << "simpleHashInts";
    for(int x: v) cout << " " << x;
    cout << " -> " << Global::uint64ToHexString(Hash::simpleHash(v.data(), (int)v.size())) << "\n";
  }
  vector<pair<string,string>> sh = {{"",""},{"abc","seed"},{"writeTrainingData","writeTrainingDataKeepFrac"},{"x","\xc3\xbc"}};
  for(auto& p: sh) {
    cout << "seededHashFloat " << p.first << "|" << p.second << " " << dbits(Hash::seededHashFloat(p.first,p.second)) << "\n";
  }
  Hash128 h1(0x0123456789abcdefULL, 0xfedcba9876543210ULL);
  Hash128 h2(0x1111111111111111ULL, 0x0000000000000001ULL);
  Hash128 h3(0x0000000000000002ULL, 0x0000000000000001ULL);
  cout << "h1 " << h1 << " toString " << h1.toString() << "\n";
  cout << "xor " << (h1 ^ h2) << " or " << (h1 | h2) << " and " << (h1 & h2) << "\n";
  cout << "lt " << (h2 < h3) << (h3 < h2) << (h1 < h2) << (h2 < h1) << " le " << (h2 <= h2) << " gt " << (h2 > h3) << " ge " << (h3 >= h2) << "\n";
  vector<string> hs = {"FEDCBA98765432100123456789ABCDEF", "fedcba98765432100123456789abcdef", "00000000000000000000000000000000", "FEDCBA9876543210", "FEDCBA98765432100123456789ABCDEG", " EDCBA98765432100123456789ABCDEF"};
  for(const string& s: hs) {
    try {
      Hash128 h = Hash128::ofString(s);
      cout << "ofString " << s << " -> " << h << "\n";
    }
    catch(const StringError& e) {
      cout << "ofString " << s << " -> ERR " << e.what() << "\n";
    }
  }
}

static Hash128 foldTable(const Hash128* t, int n) {
  Hash128 acc;
  for(int i = 0; i<n; i++) {
    acc ^= t[i];
    acc = Hash128(Hash::murmurMix(acc.hash0 + (uint64_t)i), Hash::splitMix64(acc.hash1));
  }
  return acc;
}

static void boardMode() {
  Board::initHash();
  cout << "fold ZOBRIST_PLAYER_HASH " << foldTable(Board::ZOBRIST_PLAYER_HASH, 4) << "\n";
  cout << "fold ZOBRIST_ENCORE_HASH " << foldTable(Board::ZOBRIST_ENCORE_HASH, 3) << "\n";
  cout << "fold ZOBRIST_SIZE_X_HASH " << foldTable(Board::ZOBRIST_SIZE_X_HASH, Board::MAX_LEN+1) << "\n";
  cout << "fold ZOBRIST_SIZE_Y_HASH " << foldTable(Board::ZOBRIST_SIZE_Y_HASH, Board::MAX_LEN+1) << "\n";
  cout << "fold ZOBRIST_BOARD_HASH " << foldTable(&Board::ZOBRIST_BOARD_HASH[0][0], Board::MAX_ARR_SIZE*4) << "\n";
  cout << "fold ZOBRIST_BOARD_HASH2 " << foldTable(&Board::ZOBRIST_BOARD_HASH2[0][0], Board::MAX_ARR_SIZE*4) << "\n";
  cout << "fold ZOBRIST_KO_LOC_HASH " << foldTable(Board::ZOBRIST_KO_LOC_HASH, Board::MAX_ARR_SIZE) << "\n";
  cout << "fold ZOBRIST_KO_MARK_HASH " << foldTable(&Board::ZOBRIST_KO_MARK_HASH[0][0], Board::MAX_ARR_SIZE*4) << "\n";
  cout << "fold ZOBRIST_SECOND_ENCORE_START_HASH " << foldTable(&Board::ZOBRIST_SECOND_ENCORE_START_HASH[0][0], Board::MAX_ARR_SIZE*4) << "\n";
  for(int i = 0; i<4; i++) cout << "ZOBRIST_PLAYER_HASH " << i << " " << Board::ZOBRIST_PLAYER_HASH[i] << "\n";
  int spots[6] = {0, 1, 21, 22, 200, Board::MAX_ARR_SIZE-1};
  for(int k = 0; k<6; k++) {
    int i = spots[k];
    cout << "spot " << i
         << " B1 " << Board::ZOBRIST_BOARD_HASH[i][1] << " B2 " << Board::ZOBRIST_BOARD_HASH[i][2]
         << " KO " << Board::ZOBRIST_KO_LOC_HASH[i]
         << " H2_0 " << Board::ZOBRIST_BOARD_HASH2[i][0] << " H2_3 " << Board::ZOBRIST_BOARD_HASH2[i][3]
         << " SE1 " << Board::ZOBRIST_SECOND_ENCORE_START_HASH[i][1] << "\n";
  }

  int sizes[7][2] = {{19,19},{9,9},{13,7},{5,17},{2,2},{1,1},{0,0}};
  for(int s = 0; s<7; s++) {
    int xs = sizes[s][0], ys = sizes[s][1];
    Board board(xs,ys);
    bool suicideLegal = (s % 2 == 0);
    Rand rand("board oracle " + Global::intToString(xs) + "x" + Global::intToString(ys));
    cout << "BOARD " << xs << " " << ys << " suicide " << suicideLegal << " start " << board.pos_hash << "\n";
    vector<Board::MoveRecord> records;
    Player pla = P_BLACK;
    int steps = (xs*ys < 10) ? 40 : 300;
    for(int n = 0; n<steps; n++) {
      Loc loc;
      if(rand.nextBool(0.03))
        loc = Board::PASS_LOC;
      else {
        int tries = 0;
        while(true) {
          loc = (Loc)rand.nextUInt(Board::MAX_ARR_SIZE);
          tries++;
          if(board.isLegal(loc,pla,suicideLegal))
            break;
          if(tries > 2000) { loc = Board::PASS_LOC; break; }
        }
      }
      Hash128 predicted = board.getPosHashAfterMove(loc,pla);
      bool wasSuicide = board.isSuicide(loc,pla);
      bool wasCapture = board.wouldBeCapture(loc,pla);
      bool wasKoCapture = board.wouldBeKoCapture(loc,pla);
      int libsAfter = loc == Board::PASS_LOC ? -1 : board.getNumLibertiesAfterPlay(loc,pla,4);
      records.push_back(board.playMoveRecorded(loc,pla));
      cout << n << " " << PlayerIO::playerToStringShort(pla) << " " << Location::toString(loc,board)
           << " pred " << predicted << " pos " << board.pos_hash
           << " sit " << board.getSitHashWithSimpleKo(getOpp(pla))
           << " ko " << Location::toString(board.ko_loc,board)
           << " caps " << board.numBlackCaptures << " " << board.numWhiteCaptures
           << " flags " << wasSuicide << wasCapture << wasKoCapture << " libs " << libsAfter
           << " stones " << board.numStonesOnBoard() << "\n";
      board.checkConsistency();
      pla = rand.nextBool(0.9) ? getOpp(pla) : pla;
    }
    Board::printBoard(cout, board, Board::NULL_LOC, NULL);
    for(int n = (int)records.size()-1; n >= 0 && n >= (int)records.size()-60; n--) {
      board.undo(records[n]);
      board.checkConsistency();
      cout << "undo " << n << " pos " << board.pos_hash << " ko " << Location::toString(board.ko_loc,board)
           << " caps " << board.numBlackCaptures << " " << board.numWhiteCaptures << "\n";
    }
  }
}

static void printRules(const Rules& r) {
  cout << "  toString " << r.toString() << "\n";
  cout << "  toStringNoKomi " << r.toStringNoKomi() << "\n";
  cout << "  toJsonString " << r.toJsonString() << "\n";
  cout << "  toJsonStringNoKomi " << r.toJsonStringNoKomi() << "\n";
  cout << "  toJsonStringNoKomiMaybeOmitStuff " << r.toJsonStringNoKomiMaybeOmitStuff() << "\n";
  cout << "  toStringNoKomiMaybeNice " << r.toStringNoKomiMaybeNice() << "\n";
  cout << "  gameResultWillBeInteger " << r.gameResultWillBeInteger() << "\n";
}

static void rulesMode() {
  vector<string> inputs = {
    "japanese","Korean","chinese"," Chinese-OGS ","chinese_kgs","ancientarea","stone scoring","ancient territory",
    "aga-button","AGA","bga","french","nz","new zealand","tromp-taylor","TrompTaylor","goe","ing","","   ","unknown",
    "koSIMPLEscoreAREAtaxNONEsui0komi7.5","komi23taxALL","koPOSITIONALscoringTERRITORYsui1button1whbN-1fpok1komi-3",
    "kosimple","komi1e9","komi7.5.5","komi 7 koSPIGHT","whbN","taxSEKIscoreAREA","button1","komi+2.5","komi.5","komi-0",
    "{}","{\"ko\":\"SIMPLE\",\"scoring\":\"TERRITORY\"}","{\"komi\":6}","{\"komi\":true}","{\"komi\":\"6\"}",
    "{\"komi\":7.25}","{\"komi\":200}","{\"komi\":-150}","{\"suicide\":1}","{\"ko\":3}","{\"hasButton\":true}",
    "{\"whiteHandicapBonus\":\"N-1\",\"friendlyPassOk\":true,\"tax\":\"ALL\"}","{\"foo\":1}","{bad json",
    "{\"ko\":\"SIMPLE\"} trailing","{\"ko\":\"SIMPLE\",\"ko\":\"SPIGHT\"}","{\"score\":\"AREA\",\"ko\":\"BOGUS\"}",
    "{\"komi\":1e2}","{\"komi\":0.5,\"scoring\":\"TERRITORY\"}",
  };
  for(const string& s: inputs) {
    cout << "INPUT [" << s << "]\n";
    try {
      Rules r = Rules::parseRules(s);
      cout << " parseRules ok\n";
      printRules(r);
    }
    catch(const StringError& e) {
      cout << " parseRules ERR " << e.what() << "\n";
    }
    try {
      Rules r = Rules::parseRulesWithoutKomi(s, 6.0f);
      cout << " parseRulesWithoutKomi ok " << r.toString() << "\n";
    }
    catch(const StringError& e) {
      cout << " parseRulesWithoutKomi ERR " << e.what() << "\n";
    }
    Rules buf;
    bool suc = Rules::tryParseRules(s, buf);
    cout << " tryParseRules " << suc << " " << buf.toString() << "\n";
  }
  vector<pair<string,string>> updates = {{"ko","simple"},{" score ","territory"},{"scoring","AREA"},{"tax","all"},{"suicide","false"},{"suicide","TRUE"},{"suicide","yes"},{"hasButton","true"},{"whiteHandicapBonus","n-1"},{"friendlyPassOk","true"},{"komi","7"},{"ko","bad"}};
  Rules cur = Rules::getTrompTaylorish();
  for(auto& p: updates) {
    try {
      cur = Rules::updateRules(p.first, p.second, cur);
      cout << "updateRules " << p.first << "=" << p.second << " -> " << cur.toString() << "\n";
    }
    catch(const StringError& e) {
      cout << "updateRules " << p.first << "=" << p.second << " ERR " << e.what() << "\n";
    }
  }
  vector<float> komis = {0.0f, 0.5f, 7.5f, -7.5f, 0.25f, 1e10f, -150.0f, 3.3f, INFINITY, NAN};
  for(float k: komis) {
    cout << "komiIsIntOrHalfInt " << k << " " << Rules::komiIsIntOrHalfInt(k) << "\n";
  }
  vector<Rules> rules = {
    Rules(Rules::KO_SIMPLE, Rules::SCORING_AREA, Rules::TAX_NONE, false, false, Rules::WHB_ZERO, false, 1.0f),
    Rules(Rules::KO_POSITIONAL, Rules::SCORING_AREA, Rules::TAX_NONE, false, true, Rules::WHB_ZERO, true, 3.0f),
    Rules(Rules::KO_SITUATIONAL, Rules::SCORING_AREA, Rules::TAX_NONE, true, false, Rules::WHB_ZERO, false, 5.5f),
    Rules(Rules::KO_SPIGHT, Rules::SCORING_TERRITORY, Rules::TAX_ALL, true, true, Rules::WHB_N_MINUS_ONE, true, -2.5f),
    Rules(Rules::KO_SIMPLE, Rules::SCORING_AREA, Rules::TAX_SEKI, false, false, Rules::WHB_N, false, 0.1f),
  };
  for(const Rules& r: rules) {
    cout << "RULES\n";
    printRules(r);
  }
  cout << "simpleTerritory " << Rules::getSimpleTerritory().toString() << "\n";
}

// printf table: kind \t fmt \t arg \t output
static void printfMode() {
  vector<string> ifmts = {"%d","%3d","%-5d|","%05d","%+d","% d","%.3d","%2d","%i","%10.4d|","%-+6d|"};
  vector<int> ivals = {0,1,-1,7,42,-42,999,1000,2147483647,-2147483647-1,123456789};
  for(const string& f: ifmts) for(int v: ivals) {
    char buf[256]; snprintf(buf, sizeof(buf), f.c_str(), v);
    cout << "i32\t" << f << "\t" << v << "\t" << buf << "\n";
  }
  vector<string> ufmts = {"%u","%x","%X","%08X","%#x","%o","%5u|"};
  vector<unsigned> uvals = {0u,1u,255u,4294967295u,3735928559u};
  for(const string& f: ufmts) for(unsigned v: uvals) {
    char buf[256]; snprintf(buf, sizeof(buf), f.c_str(), v);
    cout << "u32\t" << f << "\t" << v << "\t" << buf << "\n";
  }
  vector<string> lfmts = {"%lld","%20lld|","%-20lld|"};
  vector<long long> lvals = {0LL, -1LL, 9223372036854775807LL, -9223372036854775807LL-1, 1234567890123LL};
  for(const string& f: lfmts) for(long long v: lvals) {
    char buf[256]; snprintf(buf, sizeof(buf), f.c_str(), v);
    cout << "i64\t" << f << "\t" << v << "\t" << buf << "\n";
  }
  vector<string> ffmts = {"%f","%.0f","%.1f","%.2f","%.3f","%e","%.3e","%.0e","%g","%.3g","%.10g","%.17g","%#g","%+g","%10.3f|","%-10.2f|","%010.3f","%4.0f ","%5.2E","%G","%.1g","%.0g","% .4f","%#.0f","%#.0e","%.20f","%.30e","%#.3g","%#.1g","%#.2g"};
  vector<double> fvals = {0.0, -0.0, 0.125, 0.5, 1.5, 2.5, -2.5, 3.5, 0.05, 0.15, 0.25, 0.35, 1e-5, 0.0001, 0.00012345, 123456789.0, 1e15, 1e16, 1e21, 1e-300,
    4.9406564584124654e-324, 2.2250738585072014e-308, 1.7976931348623157e308, 0.1, 1.0/3.0, 2.0/3.0, 999999.5, 9.9999995, 99.95, 100.0, 1e100,
    -1234.5678, 5e-324*3, 0.000099999, 314.159, 7.5, -150.0, 6.5, 0.333f, (double)0.1f, 99999.95, 9.9999995, 0.000999999, 0.0000999995, 9.5, 99.5, 999.95, 0.99999995, 9999995.0, INFINITY, -INFINITY, NAN, -NAN};
  for(const string& f: ffmts) for(double v: fvals) {
    char buf[512]; snprintf(buf, sizeof(buf), f.c_str(), v);
    cout << "f64\t" << f << "\t" << dbits(v) << "\t" << buf << "\n";
  }
  vector<string> sfmts = {"%s","%5s|","%-5s|","%.2s|"};
  vector<string> svals = {"", "ab", "hello world"};
  for(const string& f: sfmts) for(const string& v: svals) {
    char buf[256]; snprintf(buf, sizeof(buf), f.c_str(), v.c_str());
    cout << "str\t" << f << "\t" << v << "\t" << buf << "\n";
  }
  vector<string> cfmts = {"%c","%3c|","%-3c|","%c%c"};
  for(const string& f: cfmts) {
    char buf[256];
    if(f == "%c%c") snprintf(buf, sizeof(buf), f.c_str(), 'A', 'z');
    else snprintf(buf, sizeof(buf), f.c_str(), 'Q');
    cout << "char\t" << f << "\t" << "Q" << "\t" << buf << "\n";
  }
  // ostream << double (default precision) and precision(17)
  for(double v: fvals) {
    ostringstream o1; o1 << v;
    ostringstream o2; o2.precision(17); o2 << v;
    cout << "ostream\t" << dbits(v) << "\t" << o1.str() << "\t" << o2.str() << "\n";
  }
  vector<float> flvals = {0.1f, 7.5f, 1.0f/3.0f, 1e-7f, 16777217.0f, -0.0f, 3.4028235e38f};
  for(float v: flvals) {
    ostringstream o1; o1 << v;
    cout << "ostreamf32\t" << dbits((double)v) << "\t" << o1.str() << "\t" << Global::floatToString(v) << "\n";
  }
}

static uint64_t sm(uint64_t& state) {
  state += 0x9e3779b97f4a7c15ULL;
  uint64_t z = state;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

static void nlohmannMode() {
  vector<double> vals = {0.0, -0.0, 1.0, -1.0, 9.0, 7.5, 0.1, 0.2, 0.3, 1.0/3.0, 0.65662659, 4.12549165e-05, 1e-5, 1e-4, 1e15, 1e16, 1e17,
    123456789012345.0, 1234567890123456.0, 5e-324, 1.7976931348623157e308, 2.2250738585072014e-308, (double)0.1f, (double)7.5f, 100.0, 1e21, 1e22, 0.001, 299792458.0};
  uint64_t state = 12345;
  for(int i = 0; i<3000; i++) {
    uint64_t bits = sm(state);
    double d; memcpy(&d, &bits, 8);
    if(!std::isfinite(d)) continue;
    vals.push_back(d);
  }
  state = 999;
  for(int i = 0; i<2000; i++) {
    // "Human" scale values: random decimal with few digits, and float-widened values
    double d = (double)(int64_t)(sm(state) % 2000001) / 1000.0 - 1000.0;
    vals.push_back(d);
    float f = (float)((double)(sm(state) % 1000000) / 7919.0);
    vals.push_back((double)f);
    vals.push_back(ldexp((double)(sm(state) % 1000000), (int)(sm(state) % 200) - 100));
  }
  for(double d: vals) {
    nlohmann::json j = d;
    cout << dbits(d) << "\t" << j.dump() << "\n";
  }
  nlohmann::json obj;
  obj["b"] = 1;
  obj["a"] = "x\"y\\z\n\t\x01\x1f end \xc3\xa9";
  obj["c"] = nlohmann::json::array({1, -2, 3.5, true, false, nullptr, "s"});
  obj["d"] = nlohmann::json::object();
  obj["e"] = nlohmann::json::array();
  obj["f"] = (uint64_t)18446744073709551615ULL;
  obj["g"] = (int64_t)(-9223372036854775807LL-1);
  obj["h"] = NAN;
  obj["Z"] = 0.5f;
  cout << "OBJECT\t" << obj.dump() << "\n";
}

int main(int argc, char** argv) {
  if(argc < 2) { cerr << "usage: oracle <mode>" << endl; return 1; }
  string mode = argv[1];
  if(mode == "rand") randMode();
  else if(mode == "hash") hashMode();
  else if(mode == "board") boardMode();
  else if(mode == "rules") rulesMode();
  else if(mode == "printf") printfMode();
  else if(mode == "nlohmann") nlohmannMode();
  else { cerr << "bad mode" << endl; return 1; }
  return 0;
}
