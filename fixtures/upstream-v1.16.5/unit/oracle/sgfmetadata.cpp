// Oracle for crates/katago-core/src/neuralnet/sgfmetadata.rs (and core/datetime.rs's SimpleDate as it
// uses it): SGFMetadata::getProfile on every human-SL profile name (proyear_*, rank_*, preaz_*, the
// two-rank forms) plus malformed names, getHash for both players, fillMetadataRow for both players and
// several board areas, and operator== / operator!= on mutated pairs; the same on random metadata.
// Not upstream code; links the unmodified upstream sources. The Rust mirror is
// crates/katago-core/tests/sgfmetadata_oracle.rs (natively and on wasm32: the row uses log/pow/sin/cos).
// Every random draw is its own statement (GCC argument order).
//
// Build (from the repo root, output outside the repo; upstream untouched; asserts on):
//   U=upstream/KataGo/cpp; g++ -std=c++17 -O2 -I$U -o /tmp/sgfmetadata_oracle \
//     fixtures/upstream-v1.16.5/unit/oracle/sgfmetadata.cpp $U/neuralnet/sgfmetadata.cpp \
//     $U/core/datetime.cpp $U/core/global.cpp $U/core/hash.cpp $U/core/rand.cpp $U/core/sha2.cpp \
//     $U/core/md5.cpp $U/core/test.cpp $U/core/timer.cpp $U/core/bsearch.cpp -lpthread
//   /tmp/sgfmetadata_oracle > fixtures/upstream-v1.16.5/unit/oracle/sgfmetadata.txt
// Generated with g++ (GCC) 16.2.1 / glibc 2.44 on x86_64 Linux (Intel i7-8700K, FMA).
//
// Output (floats as uppercase hex bit patterns; names as hex bytes, "-" for the empty string):
//   P <nameHex> ERR <what()>                       getProfile threw (StringError)
//   P <nameHex> OK <id>                            getProfile returned metadata <id>, then its M line
//   M <id> <initialized> <invB> <invW> <bUnranked> <wUnranked> <bRankUnknown> <wRankUnknown> <bHuman>
//     <wHuman> <unrated> <ratednessUnknown> <tcUnknown> <tcNone> <tcAbsolute> <tcSimple> <tcByoYomi>
//     <tcCanadian> <tcFischer> <mainTimeF64> <periodTimeF64> <byoYomiPeriods> <canadianMoves>
//     <year> <month> <day> <source>
//     a metadata (for random ones: the input; for profiles: getProfile's result)
//   H <id> <pla> <hash0> <hash1>                   getHash(pla) (only for metadata getHash accepts)
//   R <id> <pla> <boardArea> <digest>              fillMetadataRow(&m, row, pla, boardArea): 64-bit
//     FNV-1a over the 192 floats' bit patterns (little-endian bytes), as 16 hex digits
//   F <id> <pla> <boardArea> <192 x f32>           the same row in full (a subset, first board area only)
//   Board areas: 361, 81, 169, 119, 1, 625 (the first numAreas of them, per emit call: 6 for single
//   profiles, 1 for the two-rank ones, 3 for random metadata)
//   E <id> <id2> <eq01> <neq01>                    operator== / operator!= of metadata id and id2
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "core/datetime.h"
#include "core/global.h"
#include "core/hash.h"
#include "neuralnet/sgfmetadata.h"

using namespace std;

static uint64_t S = 0x6A09E667F3BCC908ULL;
static uint64_t nxt() {
  uint64_t z = (S += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}
static int rnd(int n) { return (int)(nxt() % (uint64_t)n); }
static double unif() { return (double)(nxt() >> 11) * 0x1p-53; }

static string f32hex(float f) { uint32_t u; memcpy(&u, &f, 4); return Global::strprintf("%08X", u); }
static string f64hex(double d) { uint64_t u; memcpy(&u, &d, 8); return Global::strprintf("%016llX", (unsigned long long)u); }
static string namehex(const string& s) {
  if(s.empty()) return "-";
  string r;
  for(unsigned char c: s) r += Global::strprintf("%02X", (unsigned)c);
  return r;
}

static int nextId = 0;

static void printM(int id, const SGFMetadata& m) {
  printf("M %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %s %s %d %d %d %d %d %d\n",
    id, (int)m.initialized, m.inverseBRank, m.inverseWRank, (int)m.bIsUnranked, (int)m.wIsUnranked,
    (int)m.bRankIsUnknown, (int)m.wRankIsUnknown, (int)m.bIsHuman, (int)m.wIsHuman,
    (int)m.gameIsUnrated, (int)m.gameRatednessIsUnknown,
    (int)m.tcIsUnknown, (int)m.tcIsNone, (int)m.tcIsAbsolute, (int)m.tcIsSimple, (int)m.tcIsByoYomi,
    (int)m.tcIsCanadian, (int)m.tcIsFischer,
    f64hex(m.mainTimeSeconds).c_str(), f64hex(m.periodTimeSeconds).c_str(), m.byoYomiPeriods, m.canadianMoves,
    m.gameDate.year, m.gameDate.month, m.gameDate.day, m.source);
}

static bool hashOk(const SGFMetadata& m) {
  return m.initialized && m.inverseBRank >= 0 && m.inverseBRank < 128 && m.inverseWRank >= 0 && m.inverseWRank < 128 &&
    m.source >= 0 && m.source < 128 && m.mainTimeSeconds >= 0 && m.periodTimeSeconds >= 0 && m.byoYomiPeriods >= 0 &&
    m.canadianMoves >= 0 && std::isfinite(m.mainTimeSeconds) && std::isfinite(m.periodTimeSeconds);
}
static bool fillOk(const SGFMetadata& m) {
  int n = (int)m.tcIsUnknown + (int)m.tcIsNone + (int)m.tcIsAbsolute + (int)m.tcIsSimple + (int)m.tcIsByoYomi +
    (int)m.tcIsCanadian + (int)m.tcIsFischer;
  return m.initialized && n == 1 && m.source >= 0 && m.source < 16;
}

static void printH(int id, const SGFMetadata& m) {
  for(Player pla: {P_BLACK, P_WHITE}) {
    Hash128 h = m.getHash(pla);
    printf("H %d %d %016llX %016llX\n", id, (int)pla, (unsigned long long)h.hash0, (unsigned long long)h.hash1);
  }
}

static void printR(int id, const SGFMetadata& m, Player pla, int boardArea, bool full) {
  float row[SGFMetadata::METADATA_INPUT_NUM_CHANNELS];
  for(int i = 0; i < SGFMetadata::METADATA_INPUT_NUM_CHANNELS; i++)
    row[i] = -7.0f; // garbage: fillMetadataRow must overwrite every entry
  SGFMetadata::fillMetadataRow(&m, row, pla, boardArea);
  uint64_t h = 0xCBF29CE484222325ULL;
  for(int i = 0; i < SGFMetadata::METADATA_INPUT_NUM_CHANNELS; i++) {
    uint32_t u;
    memcpy(&u, &row[i], 4);
    for(int b = 0; b < 4; b++) {
      h ^= (u >> (8 * b)) & 0xFF;
      h *= 0x100000001B3ULL;
    }
  }
  printf("R %d %d %d %016llX\n", id, (int)pla, boardArea, (unsigned long long)h);
  if(full) {
    printf("F %d %d %d", id, (int)pla, boardArea);
    for(int i = 0; i < SGFMetadata::METADATA_INPUT_NUM_CHANNELS; i++)
      printf(" %s", f32hex(row[i]).c_str());
    printf("\n");
  }
}

static const int AREAS[] = {361, 81, 169, 119, 1, 625};

static int emit(const SGFMetadata& m, bool full, int numAreas) {
  int id = nextId++;
  printM(id, m);
  if(hashOk(m))
    printH(id, m);
  if(fillOk(m)) {
    for(Player pla: {P_BLACK, P_WHITE}) {
      for(int a = 0; a < numAreas; a++)
        printR(id, m, pla, AREAS[a], full && a == 0);
    }
  }
  return id;
}

static void profile(const string& name, bool full, int numAreas) {
  try {
    SGFMetadata m = SGFMetadata::getProfile(name);
    printf("P %s OK %d\n", namehex(name).c_str(), nextId);
    emit(m, full, numAreas);
  }
  catch(const StringError& e) {
    printf("P %s ERR %s\n", namehex(name).c_str(), e.what());
  }
}

static SGFMetadata randomMeta(bool validForFill) {
  SGFMetadata m;
  m.initialized = true;
  int r = rnd(10);
  if(r == 0) { m.inverseBRank = rnd(128); } else { m.inverseBRank = rnd(40); }
  r = rnd(10);
  if(r == 0) { m.inverseWRank = rnd(128); } else { m.inverseWRank = rnd(40); }
  m.bIsUnranked = rnd(5) == 0;
  m.wIsUnranked = rnd(5) == 0;
  m.bRankIsUnknown = rnd(5) == 0;
  m.wRankIsUnknown = rnd(5) == 0;
  m.bIsHuman = rnd(3) != 0;
  m.wIsHuman = rnd(3) != 0;
  m.gameIsUnrated = rnd(2) == 0;
  m.gameRatednessIsUnknown = rnd(3) == 0;
  if(validForFill) {
    int tc = rnd(7);
    m.tcIsUnknown = tc == 0; m.tcIsNone = tc == 1; m.tcIsAbsolute = tc == 2; m.tcIsSimple = tc == 3;
    m.tcIsByoYomi = tc == 4; m.tcIsCanadian = tc == 5; m.tcIsFischer = tc == 6;
  }
  else {
    m.tcIsUnknown = rnd(4) == 0; m.tcIsNone = rnd(4) == 0; m.tcIsAbsolute = rnd(4) == 0; m.tcIsSimple = rnd(4) == 0;
    m.tcIsByoYomi = rnd(4) == 0; m.tcIsCanadian = rnd(4) == 0; m.tcIsFischer = rnd(4) == 0;
  }
  r = rnd(6);
  if(r == 0) m.mainTimeSeconds = 0.0;
  else if(r == 1) m.mainTimeSeconds = unif() * 1e7; // above the 3-day cap
  else if(r == 2) m.mainTimeSeconds = (double)rnd(4000);
  else m.mainTimeSeconds = unif() * 7200.0;
  r = rnd(6);
  if(r == 0) m.periodTimeSeconds = 0.0;
  else if(r == 1) m.periodTimeSeconds = unif() * 1e6; // above the 1-day cap
  else if(r == 2) m.periodTimeSeconds = (double)rnd(120);
  else m.periodTimeSeconds = unif() * 60.0;
  r = rnd(4);
  if(r == 0) m.byoYomiPeriods = rnd(200); else m.byoYomiPeriods = rnd(10);
  r = rnd(4);
  if(r == 0) m.canadianMoves = rnd(200); else m.canadianMoves = rnd(30);
  int days = 0;
  r = rnd(4);
  if(r == 0) days = rnd(160001) - 80000; // 1751..2189
  else if(r == 1) days = rnd(2000001) - 1000000; // years -768..4707
  else days = rnd(30000) + 10000; // 1997..2079
  m.gameDate = SimpleDate() + days;
  if(validForFill) m.source = rnd(16); else m.source = rnd(128);
  return m;
}

int main() {
  vector<string> rankStrs;
  for(int d = 9; d >= 1; d--) rankStrs.push_back(Global::intToString(d) + "d");
  for(int k = 1; k <= 20; k++) rankStrs.push_back(Global::intToString(k) + "k");

  // Every profile name getProfile knows, plus neighbours that it rejects.
  profile("", true, 6);
  profile("_", false, 6);
  profile("\"\"", false, 6);
  for(int y = 1795; y <= 2026; y++) profile("proyear_" + Global::intToString(y), y % 37 == 0 || y >= 2019, 6);
  for(const string& prefix: {string("rank_"), string("preaz_")}) {
    for(size_t i = 0; i < rankStrs.size(); i++)
      profile(prefix + rankStrs[i], true, 6);
    for(size_t i = 0; i < rankStrs.size(); i++)
      for(size_t j = 0; j < rankStrs.size(); j++)
        profile(prefix + rankStrs[i] + "_" + rankStrs[j], (i * 31 + j) % 97 == 0, 1);
  }
  const char* bad[] = {
    "rank_", "preaz_", "rank_0d", "rank_10d", "rank_21k", "rank_0k", "rank_9D", "Rank_9d", "RANK_9d", "rank_9d_",
    "rank__9d", "rank_9d__1k", "rank_9d_1k_2k", "rank_9d_21k", "preaz_ 9d", "rank_9d ", " rank_9d", "rank-9d",
    "proyear_", "proyear_abc", "proyear_+2000", "proyear_02000", "proyear_2000.0", "proyear_ 2000", "proyear_2000 ",
    "proyear_-2000", "proyear_1e3", "proyear_0x7d0", "proyear_99999999999", "proyear_2000_", "proyear_1799",
    "proyear_2024", "foo", "pro", "rank", "9d", "preaz_rank_9d", "rank_preaz_9d", "\"", "\"\"\"", "__", "rank_1d_1d_",
    "rank_k", "rank_d", "rank_1", "proyear_2021x", "preaz_proyear_2000",
  };
  for(const char* b: bad) profile(b, false, 2);

  // Random metadata: valid for fillMetadataRow and getHash, then only for getHash (any tc flags, source < 128).
  int firstRandom = nextId;
  for(int i = 0; i < 1000; i++) {
    SGFMetadata m = randomMeta(true);
    emit(m, i < 40, 3);
  }
  for(int i = 0; i < 300; i++) {
    SGFMetadata m = randomMeta(false);
    emit(m, false, 1);
  }
  // A few uninitialized / out-of-range ones: no H/R lines (getHash and fillMetadataRow are fatal there).
  {
    SGFMetadata m;
    emit(m, false, 1);
    m = randomMeta(true);
    m.initialized = false;
    emit(m, false, 1);
  }

  // operator== / !=: a random metadata against copies that differ in one field (and identical copies).
  for(int i = 0; i < 300; i++) {
    SGFMetadata a = randomMeta(rnd(2) == 0);
    SGFMetadata b = a;
    int which = rnd(26);
    switch(which) {
    case 0: b.initialized = !b.initialized; break;
    case 1: b.inverseBRank += 1; break;
    case 2: b.inverseWRank += 1; break;
    case 3: b.bIsUnranked = !b.bIsUnranked; break;
    case 4: b.wIsUnranked = !b.wIsUnranked; break;
    case 5: b.bRankIsUnknown = !b.bRankIsUnknown; break;
    case 6: b.wRankIsUnknown = !b.wRankIsUnknown; break;
    case 7: b.bIsHuman = !b.bIsHuman; break;
    case 8: b.wIsHuman = !b.wIsHuman; break;
    case 9: b.gameIsUnrated = !b.gameIsUnrated; break;
    case 10: b.gameRatednessIsUnknown = !b.gameRatednessIsUnknown; break;
    case 11: b.tcIsUnknown = !b.tcIsUnknown; break;
    case 12: b.tcIsNone = !b.tcIsNone; break;
    case 13: b.tcIsAbsolute = !b.tcIsAbsolute; break;
    case 14: b.tcIsSimple = !b.tcIsSimple; break;
    case 15: b.tcIsByoYomi = !b.tcIsByoYomi; break;
    case 16: b.tcIsCanadian = !b.tcIsCanadian; break;
    case 17: b.tcIsFischer = !b.tcIsFischer; break;
    case 18: b.mainTimeSeconds = nextafter(b.mainTimeSeconds, 1e300); break;
    case 19: b.periodTimeSeconds = nextafter(b.periodTimeSeconds, 1e300); break;
    case 20: b.byoYomiPeriods += 1; break;
    case 21: b.canadianMoves += 1; break;
    case 22: b.gameDate += 1; break;
    case 23: b.gameDate.year += 1; b.gameDate.month = 1; break;
    case 24: b.source += 1; break;
    default: break; // identical
    }
    int ida = nextId++;
    printM(ida, a);
    int idb = nextId++;
    printM(idb, b);
    printf("E %d %d %d %d\n", ida, idb, (int)(a == b), (int)(a != b));
  }
  (void)firstRandom;
  return 0;
}
