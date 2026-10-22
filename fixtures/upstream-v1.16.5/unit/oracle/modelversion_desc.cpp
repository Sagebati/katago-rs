// Oracle for crates/katago-core/src/neuralnet/{modelversion,desc}.rs: NNModelVersion's functions
// over model versions -2..20 and meta encoder versions -1..3 (values or exception texts), the
// header constants, ModelPostProcessParams() defaults, and ModelDesc::getSupportedRules over every
// rules combination for model versions 3..16. Links the unmodified upstream sources (via the Eigen
// build's object files).
// Build (from the repo root, after `just katago-eigen`; output outside the repo):
//   O=build/eigen/CMakeFiles/katago.dir; g++ -std=c++17 -O2 -Iupstream/KataGo/cpp -o /tmp/modelversion_desc \
//     fixtures/upstream-v1.16.5/unit/oracle/modelversion_desc.cpp \
//     $(find $O/core $O/game $O/neuralnet $O/search $O/dataio $O/program -name '*.o') -lz -lzip -lpthread
//   /tmp/modelversion_desc > fixtures/upstream-v1.16.5/unit/oracle/modelversion_desc.txt
// Output lines:
//   C <name> <value>                                       constants
//   V <fn> <arg> <result>  |  V <fn> <arg> ERR <what()>    NNModelVersion functions
//   P <field> <f64 or f32 bits hex>                        ModelPostProcessParams() defaults
//   R <modelVersion> <ko> <score> <tax> <suic> <button> <whb> <fpok> -> <ko> <score> <tax> <suic> <button> <whb> <fpok> <komi bits> <supported>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <string>
#include <functional>
#include "core/global.h"
#include "neuralnet/desc.h"
#include "neuralnet/modelversion.h"
#include "neuralnet/nninputs.h"
using namespace std;

static unsigned long long dbits(double d) { uint64_t u; memcpy(&u, &d, 8); return (unsigned long long)u; }
static unsigned fbits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }

static void v(const char* name, int arg, const std::function<int(int)>& f) {
  try {
    int r = f(arg);
    printf("V %s %d %d\n", name, arg, r);
  }
  catch(const StringError& e) {
    printf("V %s %d ERR %s\n", name, arg, e.what());
  }
}

int main() {
  printf("C latestModelVersionImplemented %d\n", NNModelVersion::latestModelVersionImplemented);
  printf("C latestInputsVersionImplemented %d\n", NNModelVersion::latestInputsVersionImplemented);
  printf("C defaultModelVersion %d\n", NNModelVersion::defaultModelVersion);
  printf("C oldestModelVersionImplemented %d\n", NNModelVersion::oldestModelVersionImplemented);
  printf("C oldestInputsVersionImplemented %d\n", NNModelVersion::oldestInputsVersionImplemented);
  printf("C ACTIVATION_IDENTITY %d\n", ACTIVATION_IDENTITY);
  printf("C ACTIVATION_RELU %d\n", ACTIVATION_RELU);
  printf("C ACTIVATION_MISH %d\n", ACTIVATION_MISH);
  printf("C ACTIVATION_MISH_SCALE8 %d\n", ACTIVATION_MISH_SCALE8);
  for(int mv = -2; mv <= 20; mv++) {
    v("getInputsVersion", mv, NNModelVersion::getInputsVersion);
    v("getNumSpatialFeatures", mv, NNModelVersion::getNumSpatialFeatures);
    v("getNumGlobalFeatures", mv, NNModelVersion::getNumGlobalFeatures);
  }
  for(int me = -1; me <= 3; me++)
    v("getNumInputMetaChannels", me, NNModelVersion::getNumInputMetaChannels);

  ModelPostProcessParams pp;
  printf("P tdScoreMultiplier %016llX\n", dbits(pp.tdScoreMultiplier));
  printf("P scoreMeanMultiplier %016llX\n", dbits(pp.scoreMeanMultiplier));
  printf("P scoreStdevMultiplier %016llX\n", dbits(pp.scoreStdevMultiplier));
  printf("P leadMultiplier %016llX\n", dbits(pp.leadMultiplier));
  printf("P varianceTimeMultiplier %016llX\n", dbits(pp.varianceTimeMultiplier));
  printf("P shorttermValueErrorMultiplier %016llX\n", dbits(pp.shorttermValueErrorMultiplier));
  printf("P shorttermScoreErrorMultiplier %016llX\n", dbits(pp.shorttermScoreErrorMultiplier));
  printf("P outputScaleMultiplier %08X\n", fbits(pp.outputScaleMultiplier));

  for(int mv = 3; mv <= 16; mv++) {
    ModelDesc desc;
    desc.modelVersion = mv;
    for(int ko = 0; ko < 4; ko++)
    for(int sc = 0; sc < 2; sc++)
    for(int tax = 0; tax < 3; tax++)
    for(int suic = 0; suic < 2; suic++)
    for(int button = 0; button < 2; button++)
    for(int whb = 0; whb < 3; whb++)
    for(int fpok = 0; fpok < 2; fpok++) {
      Rules rules(ko, sc, tax, suic != 0, button != 0, whb, fpok != 0, 6.5f);
      bool supported = true;
      Rules r = desc.getSupportedRules(rules, supported);
      printf("R %d %d %d %d %d %d %d %d -> %d %d %d %d %d %d %d %08X %d\n",
             mv, ko, sc, tax, suic, button, whb, fpok,
             r.koRule, r.scoringRule, r.taxRule, (int)r.multiStoneSuicideLegal, (int)r.hasButton,
             r.whiteHandicapBonusRule, (int)r.friendlyPassOk, fbits(r.komi), (int)supported);
    }
  }
  return 0;
}
