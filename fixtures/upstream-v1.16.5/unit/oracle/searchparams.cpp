// Oracle for crates/katago-core/src/search/searchparams.rs: the unmodified upstream SearchParams
// (constructor defaults, forTestsV1/V2, basicDecentParams, operator==/!=, changeableParametersToJson,
// printParams, failIfParamsDifferOnUnchangeableParameter). Links the unmodified upstream sources via the
// Eigen build's object files. The Rust mirror is crates/katago-core/tests/searchparams_oracle.rs.
// Build (from the repo root, after `just katago-eigen`; output outside the repo):
//   O=build/eigen/CMakeFiles/katago.dir; g++ -std=c++17 -O2 -Iupstream/KataGo/cpp -o /tmp/searchparams \
//     fixtures/upstream-v1.16.5/unit/oracle/searchparams.cpp \
//     $(find $O/core $O/game $O/neuralnet $O/search $O/dataio $O/program -name '*.o') -lz -lzip -lpthread
//   /tmp/searchparams > fixtures/upstream-v1.16.5/unit/oracle/searchparams.txt
// Output, per preset P in {default, forTestsV1, forTestsV2, basicDecentParams}:
//   S <P>                       section start
//   F <field> <value>           every field (doubles as IEEE bits, floats as bits, Player/bool/int as int)
//   J <dump()>                  changeableParametersToJson().dump()
//   PO <line>                   printParams(out) lines written to the stream argument
//   PC <line>                   lines printParams wrote to std::cout instead (upstream writes two there)
//   M <field> <0|1>             (params == copy with that one field changed): must be 0 for every field
// then, once:
//   E <a> <b> <==> <!=>         presets compared pairwise
//   U <case> OK | ERR <what>    failIfParamsDifferOnUnchangeableParameter
// Generated with g++ (GCC) 16.2.1 / glibc 2.44 on x86_64 (FMA) Linux.
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "core/global.h"
#include "search/searchparams.h"
using namespace std;

static unsigned long long dbits(double d) { uint64_t u; memcpy(&u, &d, 8); return (unsigned long long)u; }
static unsigned fbits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }

static void emitLines(const char* tag, const string& s) {
  size_t i = 0;
  while(i < s.size()) {
    size_t j = s.find('\n', i);
    if(j == string::npos) j = s.size();
    printf("%s %s\n", tag, s.substr(i, j - i).c_str());
    i = j + 1;
  }
}

static void dump(const char* name, const SearchParams& p) {
  printf("S %s\n", name);
  printf("F winLossUtilityFactor %016llX\n", dbits(p.winLossUtilityFactor));
  printf("F staticScoreUtilityFactor %016llX\n", dbits(p.staticScoreUtilityFactor));
  printf("F dynamicScoreUtilityFactor %016llX\n", dbits(p.dynamicScoreUtilityFactor));
  printf("F dynamicScoreCenterZeroWeight %016llX\n", dbits(p.dynamicScoreCenterZeroWeight));
  printf("F dynamicScoreCenterScale %016llX\n", dbits(p.dynamicScoreCenterScale));
  printf("F noResultUtilityForWhite %016llX\n", dbits(p.noResultUtilityForWhite));
  printf("F drawEquivalentWinsForWhite %016llX\n", dbits(p.drawEquivalentWinsForWhite));
  printf("F cpuctExploration %016llX\n", dbits(p.cpuctExploration));
  printf("F cpuctExplorationLog %016llX\n", dbits(p.cpuctExplorationLog));
  printf("F cpuctExplorationBase %016llX\n", dbits(p.cpuctExplorationBase));
  printf("F cpuctUtilityStdevPrior %016llX\n", dbits(p.cpuctUtilityStdevPrior));
  printf("F cpuctUtilityStdevPriorWeight %016llX\n", dbits(p.cpuctUtilityStdevPriorWeight));
  printf("F cpuctUtilityStdevScale %016llX\n", dbits(p.cpuctUtilityStdevScale));
  printf("F fpuReductionMax %016llX\n", dbits(p.fpuReductionMax));
  printf("F fpuLossProp %016llX\n", dbits(p.fpuLossProp));
  printf("F fpuParentWeightByVisitedPolicy %d\n", (int)p.fpuParentWeightByVisitedPolicy);
  printf("F fpuParentWeightByVisitedPolicyPow %016llX\n", dbits(p.fpuParentWeightByVisitedPolicyPow));
  printf("F fpuParentWeight %016llX\n", dbits(p.fpuParentWeight));
  printf("F policyOptimism %016llX\n", dbits(p.policyOptimism));
  printf("F valueWeightExponent %016llX\n", dbits(p.valueWeightExponent));
  printf("F useNoisePruning %d\n", (int)p.useNoisePruning);
  printf("F noisePruneUtilityScale %016llX\n", dbits(p.noisePruneUtilityScale));
  printf("F noisePruningCap %016llX\n", dbits(p.noisePruningCap));
  printf("F useUncertainty %d\n", (int)p.useUncertainty);
  printf("F uncertaintyCoeff %016llX\n", dbits(p.uncertaintyCoeff));
  printf("F uncertaintyExponent %016llX\n", dbits(p.uncertaintyExponent));
  printf("F uncertaintyMaxWeight %016llX\n", dbits(p.uncertaintyMaxWeight));
  printf("F useGraphSearch %d\n", (int)p.useGraphSearch);
  printf("F graphSearchRepBound %d\n", (int)p.graphSearchRepBound);
  printf("F graphSearchCatchUpLeakProb %016llX\n", dbits(p.graphSearchCatchUpLeakProb));
  printf("F rootNoiseEnabled %d\n", (int)p.rootNoiseEnabled);
  printf("F rootDirichletNoiseTotalConcentration %016llX\n", dbits(p.rootDirichletNoiseTotalConcentration));
  printf("F rootDirichletNoiseWeight %016llX\n", dbits(p.rootDirichletNoiseWeight));
  printf("F rootPolicyTemperature %016llX\n", dbits(p.rootPolicyTemperature));
  printf("F rootPolicyTemperatureEarly %016llX\n", dbits(p.rootPolicyTemperatureEarly));
  printf("F rootFpuReductionMax %016llX\n", dbits(p.rootFpuReductionMax));
  printf("F rootFpuLossProp %016llX\n", dbits(p.rootFpuLossProp));
  printf("F rootNumSymmetriesToSample %d\n", (int)p.rootNumSymmetriesToSample);
  printf("F rootSymmetryPruning %d\n", (int)p.rootSymmetryPruning);
  printf("F rootDesiredPerChildVisitsCoeff %016llX\n", dbits(p.rootDesiredPerChildVisitsCoeff));
  printf("F rootPolicyOptimism %016llX\n", dbits(p.rootPolicyOptimism));
  printf("F chosenMoveTemperature %016llX\n", dbits(p.chosenMoveTemperature));
  printf("F chosenMoveTemperatureEarly %016llX\n", dbits(p.chosenMoveTemperatureEarly));
  printf("F chosenMoveTemperatureHalflife %016llX\n", dbits(p.chosenMoveTemperatureHalflife));
  printf("F chosenMoveTemperatureOnlyBelowProb %016llX\n", dbits(p.chosenMoveTemperatureOnlyBelowProb));
  printf("F chosenMoveSubtract %016llX\n", dbits(p.chosenMoveSubtract));
  printf("F chosenMovePrune %016llX\n", dbits(p.chosenMovePrune));
  printf("F useLcbForSelection %d\n", (int)p.useLcbForSelection);
  printf("F lcbStdevs %016llX\n", dbits(p.lcbStdevs));
  printf("F minVisitPropForLCB %016llX\n", dbits(p.minVisitPropForLCB));
  printf("F useNonBuggyLcb %d\n", (int)p.useNonBuggyLcb);
  printf("F rootEndingBonusPoints %016llX\n", dbits(p.rootEndingBonusPoints));
  printf("F rootPruneUselessMoves %d\n", (int)p.rootPruneUselessMoves);
  printf("F conservativePass %d\n", (int)p.conservativePass);
  printf("F fillDameBeforePass %d\n", (int)p.fillDameBeforePass);
  printf("F avoidMYTDaggerHackPla %d\n", (int)p.avoidMYTDaggerHackPla);
  printf("F wideRootNoise %016llX\n", dbits(p.wideRootNoise));
  printf("F enablePassingHacks %d\n", (int)p.enablePassingHacks);
  printf("F enableMorePassingHacks %d\n", (int)p.enableMorePassingHacks);
  printf("F playoutDoublingAdvantage %016llX\n", dbits(p.playoutDoublingAdvantage));
  printf("F playoutDoublingAdvantagePla %d\n", (int)p.playoutDoublingAdvantagePla);
  printf("F avoidRepeatedPatternUtility %016llX\n", dbits(p.avoidRepeatedPatternUtility));
  printf("F nnPolicyTemperature %08X\n", fbits(p.nnPolicyTemperature));
  printf("F antiMirror %d\n", (int)p.antiMirror);
  printf("F ignorePreRootHistory %d\n", (int)p.ignorePreRootHistory);
  printf("F ignoreAllHistory %d\n", (int)p.ignoreAllHistory);
  printf("F subtreeValueBiasFactor %016llX\n", dbits(p.subtreeValueBiasFactor));
  printf("F subtreeValueBiasTableNumShards %d\n", (int)p.subtreeValueBiasTableNumShards);
  printf("F subtreeValueBiasFreeProp %016llX\n", dbits(p.subtreeValueBiasFreeProp));
  printf("F subtreeValueBiasWeightExponent %016llX\n", dbits(p.subtreeValueBiasWeightExponent));
  printf("F useEvalCache %d\n", (int)p.useEvalCache);
  printf("F evalCacheMinVisits %lld\n", (long long)p.evalCacheMinVisits);
  printf("F nodeTableShardsPowerOfTwo %d\n", (int)p.nodeTableShardsPowerOfTwo);
  printf("F numVirtualLossesPerThread %016llX\n", dbits(p.numVirtualLossesPerThread));
  printf("F numThreads %d\n", (int)p.numThreads);
  printf("F minPlayoutsPerThread %016llX\n", dbits(p.minPlayoutsPerThread));
  printf("F maxVisits %lld\n", (long long)p.maxVisits);
  printf("F maxPlayouts %lld\n", (long long)p.maxPlayouts);
  printf("F maxTime %016llX\n", dbits(p.maxTime));
  printf("F maxVisitsPondering %lld\n", (long long)p.maxVisitsPondering);
  printf("F maxPlayoutsPondering %lld\n", (long long)p.maxPlayoutsPondering);
  printf("F maxTimePondering %016llX\n", dbits(p.maxTimePondering));
  printf("F lagBuffer %016llX\n", dbits(p.lagBuffer));
  printf("F searchFactorAfterOnePass %016llX\n", dbits(p.searchFactorAfterOnePass));
  printf("F searchFactorAfterTwoPass %016llX\n", dbits(p.searchFactorAfterTwoPass));
  printf("F treeReuseCarryOverTimeFactor %016llX\n", dbits(p.treeReuseCarryOverTimeFactor));
  printf("F overallocateTimeFactor %016llX\n", dbits(p.overallocateTimeFactor));
  printf("F midgameTimeFactor %016llX\n", dbits(p.midgameTimeFactor));
  printf("F midgameTurnPeakTime %016llX\n", dbits(p.midgameTurnPeakTime));
  printf("F endgameTurnTimeDecay %016llX\n", dbits(p.endgameTurnTimeDecay));
  printf("F obviousMovesTimeFactor %016llX\n", dbits(p.obviousMovesTimeFactor));
  printf("F obviousMovesPolicyEntropyTolerance %016llX\n", dbits(p.obviousMovesPolicyEntropyTolerance));
  printf("F obviousMovesPolicySurpriseTolerance %016llX\n", dbits(p.obviousMovesPolicySurpriseTolerance));
  printf("F futileVisitsThreshold %016llX\n", dbits(p.futileVisitsThreshold));
  printf("F humanSLProfile %d\n", (int)p.humanSLProfile.initialized);
  printf("F humanSLCpuctExploration %016llX\n", dbits(p.humanSLCpuctExploration));
  printf("F humanSLCpuctPermanent %016llX\n", dbits(p.humanSLCpuctPermanent));
  printf("F humanSLRootExploreProbWeightless %016llX\n", dbits(p.humanSLRootExploreProbWeightless));
  printf("F humanSLRootExploreProbWeightful %016llX\n", dbits(p.humanSLRootExploreProbWeightful));
  printf("F humanSLPlaExploreProbWeightless %016llX\n", dbits(p.humanSLPlaExploreProbWeightless));
  printf("F humanSLPlaExploreProbWeightful %016llX\n", dbits(p.humanSLPlaExploreProbWeightful));
  printf("F humanSLOppExploreProbWeightless %016llX\n", dbits(p.humanSLOppExploreProbWeightless));
  printf("F humanSLOppExploreProbWeightful %016llX\n", dbits(p.humanSLOppExploreProbWeightful));
  printf("F humanSLChosenMoveProp %016llX\n", dbits(p.humanSLChosenMoveProp));
  printf("F humanSLChosenMoveIgnorePass %d\n", (int)p.humanSLChosenMoveIgnorePass);
  printf("F humanSLChosenMovePiklLambda %016llX\n", dbits(p.humanSLChosenMovePiklLambda));
  printf("J %s\n", p.changeableParametersToJson().dump().c_str());
  {
    ostringstream outStream;
    ostringstream coutCapture;
    streambuf* old = cout.rdbuf(coutCapture.rdbuf());
    p.printParams(outStream);
    cout.rdbuf(old);
    emitLines("PO", outStream.str());
    emitLines("PC", coutCapture.str());
  }
  { SearchParams q = p; q.winLossUtilityFactor = p.winLossUtilityFactor * 2.0 + 1.0; printf("M winLossUtilityFactor %d\n", (int)(p == q)); }
  { SearchParams q = p; q.staticScoreUtilityFactor = p.staticScoreUtilityFactor * 2.0 + 1.0; printf("M staticScoreUtilityFactor %d\n", (int)(p == q)); }
  { SearchParams q = p; q.dynamicScoreUtilityFactor = p.dynamicScoreUtilityFactor * 2.0 + 1.0; printf("M dynamicScoreUtilityFactor %d\n", (int)(p == q)); }
  { SearchParams q = p; q.dynamicScoreCenterZeroWeight = p.dynamicScoreCenterZeroWeight * 2.0 + 1.0; printf("M dynamicScoreCenterZeroWeight %d\n", (int)(p == q)); }
  { SearchParams q = p; q.dynamicScoreCenterScale = p.dynamicScoreCenterScale * 2.0 + 1.0; printf("M dynamicScoreCenterScale %d\n", (int)(p == q)); }
  { SearchParams q = p; q.noResultUtilityForWhite = p.noResultUtilityForWhite * 2.0 + 1.0; printf("M noResultUtilityForWhite %d\n", (int)(p == q)); }
  { SearchParams q = p; q.drawEquivalentWinsForWhite = p.drawEquivalentWinsForWhite * 2.0 + 1.0; printf("M drawEquivalentWinsForWhite %d\n", (int)(p == q)); }
  { SearchParams q = p; q.cpuctExploration = p.cpuctExploration * 2.0 + 1.0; printf("M cpuctExploration %d\n", (int)(p == q)); }
  { SearchParams q = p; q.cpuctExplorationLog = p.cpuctExplorationLog * 2.0 + 1.0; printf("M cpuctExplorationLog %d\n", (int)(p == q)); }
  { SearchParams q = p; q.cpuctExplorationBase = p.cpuctExplorationBase * 2.0 + 1.0; printf("M cpuctExplorationBase %d\n", (int)(p == q)); }
  { SearchParams q = p; q.cpuctUtilityStdevPrior = p.cpuctUtilityStdevPrior * 2.0 + 1.0; printf("M cpuctUtilityStdevPrior %d\n", (int)(p == q)); }
  { SearchParams q = p; q.cpuctUtilityStdevPriorWeight = p.cpuctUtilityStdevPriorWeight * 2.0 + 1.0; printf("M cpuctUtilityStdevPriorWeight %d\n", (int)(p == q)); }
  { SearchParams q = p; q.cpuctUtilityStdevScale = p.cpuctUtilityStdevScale * 2.0 + 1.0; printf("M cpuctUtilityStdevScale %d\n", (int)(p == q)); }
  { SearchParams q = p; q.fpuReductionMax = p.fpuReductionMax * 2.0 + 1.0; printf("M fpuReductionMax %d\n", (int)(p == q)); }
  { SearchParams q = p; q.fpuLossProp = p.fpuLossProp * 2.0 + 1.0; printf("M fpuLossProp %d\n", (int)(p == q)); }
  { SearchParams q = p; q.fpuParentWeightByVisitedPolicy = !p.fpuParentWeightByVisitedPolicy; printf("M fpuParentWeightByVisitedPolicy %d\n", (int)(p == q)); }
  { SearchParams q = p; q.fpuParentWeightByVisitedPolicyPow = p.fpuParentWeightByVisitedPolicyPow * 2.0 + 1.0; printf("M fpuParentWeightByVisitedPolicyPow %d\n", (int)(p == q)); }
  { SearchParams q = p; q.fpuParentWeight = p.fpuParentWeight * 2.0 + 1.0; printf("M fpuParentWeight %d\n", (int)(p == q)); }
  { SearchParams q = p; q.policyOptimism = p.policyOptimism * 2.0 + 1.0; printf("M policyOptimism %d\n", (int)(p == q)); }
  { SearchParams q = p; q.valueWeightExponent = p.valueWeightExponent * 2.0 + 1.0; printf("M valueWeightExponent %d\n", (int)(p == q)); }
  { SearchParams q = p; q.useNoisePruning = !p.useNoisePruning; printf("M useNoisePruning %d\n", (int)(p == q)); }
  { SearchParams q = p; q.noisePruneUtilityScale = p.noisePruneUtilityScale * 2.0 + 1.0; printf("M noisePruneUtilityScale %d\n", (int)(p == q)); }
  { SearchParams q = p; q.noisePruningCap = p.noisePruningCap * 2.0 + 1.0; printf("M noisePruningCap %d\n", (int)(p == q)); }
  { SearchParams q = p; q.useUncertainty = !p.useUncertainty; printf("M useUncertainty %d\n", (int)(p == q)); }
  { SearchParams q = p; q.uncertaintyCoeff = p.uncertaintyCoeff * 2.0 + 1.0; printf("M uncertaintyCoeff %d\n", (int)(p == q)); }
  { SearchParams q = p; q.uncertaintyExponent = p.uncertaintyExponent * 2.0 + 1.0; printf("M uncertaintyExponent %d\n", (int)(p == q)); }
  { SearchParams q = p; q.uncertaintyMaxWeight = p.uncertaintyMaxWeight * 2.0 + 1.0; printf("M uncertaintyMaxWeight %d\n", (int)(p == q)); }
  { SearchParams q = p; q.useGraphSearch = !p.useGraphSearch; printf("M useGraphSearch %d\n", (int)(p == q)); }
  { SearchParams q = p; q.graphSearchRepBound = p.graphSearchRepBound + 1; printf("M graphSearchRepBound %d\n", (int)(p == q)); }
  { SearchParams q = p; q.graphSearchCatchUpLeakProb = p.graphSearchCatchUpLeakProb * 2.0 + 1.0; printf("M graphSearchCatchUpLeakProb %d\n", (int)(p == q)); }
  { SearchParams q = p; q.rootNoiseEnabled = !p.rootNoiseEnabled; printf("M rootNoiseEnabled %d\n", (int)(p == q)); }
  { SearchParams q = p; q.rootDirichletNoiseTotalConcentration = p.rootDirichletNoiseTotalConcentration * 2.0 + 1.0; printf("M rootDirichletNoiseTotalConcentration %d\n", (int)(p == q)); }
  { SearchParams q = p; q.rootDirichletNoiseWeight = p.rootDirichletNoiseWeight * 2.0 + 1.0; printf("M rootDirichletNoiseWeight %d\n", (int)(p == q)); }
  { SearchParams q = p; q.rootPolicyTemperature = p.rootPolicyTemperature * 2.0 + 1.0; printf("M rootPolicyTemperature %d\n", (int)(p == q)); }
  { SearchParams q = p; q.rootPolicyTemperatureEarly = p.rootPolicyTemperatureEarly * 2.0 + 1.0; printf("M rootPolicyTemperatureEarly %d\n", (int)(p == q)); }
  { SearchParams q = p; q.rootFpuReductionMax = p.rootFpuReductionMax * 2.0 + 1.0; printf("M rootFpuReductionMax %d\n", (int)(p == q)); }
  { SearchParams q = p; q.rootFpuLossProp = p.rootFpuLossProp * 2.0 + 1.0; printf("M rootFpuLossProp %d\n", (int)(p == q)); }
  { SearchParams q = p; q.rootNumSymmetriesToSample = p.rootNumSymmetriesToSample + 1; printf("M rootNumSymmetriesToSample %d\n", (int)(p == q)); }
  { SearchParams q = p; q.rootSymmetryPruning = !p.rootSymmetryPruning; printf("M rootSymmetryPruning %d\n", (int)(p == q)); }
  { SearchParams q = p; q.rootDesiredPerChildVisitsCoeff = p.rootDesiredPerChildVisitsCoeff * 2.0 + 1.0; printf("M rootDesiredPerChildVisitsCoeff %d\n", (int)(p == q)); }
  { SearchParams q = p; q.rootPolicyOptimism = p.rootPolicyOptimism * 2.0 + 1.0; printf("M rootPolicyOptimism %d\n", (int)(p == q)); }
  { SearchParams q = p; q.chosenMoveTemperature = p.chosenMoveTemperature * 2.0 + 1.0; printf("M chosenMoveTemperature %d\n", (int)(p == q)); }
  { SearchParams q = p; q.chosenMoveTemperatureEarly = p.chosenMoveTemperatureEarly * 2.0 + 1.0; printf("M chosenMoveTemperatureEarly %d\n", (int)(p == q)); }
  { SearchParams q = p; q.chosenMoveTemperatureHalflife = p.chosenMoveTemperatureHalflife * 2.0 + 1.0; printf("M chosenMoveTemperatureHalflife %d\n", (int)(p == q)); }
  { SearchParams q = p; q.chosenMoveTemperatureOnlyBelowProb = p.chosenMoveTemperatureOnlyBelowProb * 2.0 + 1.0; printf("M chosenMoveTemperatureOnlyBelowProb %d\n", (int)(p == q)); }
  { SearchParams q = p; q.chosenMoveSubtract = p.chosenMoveSubtract * 2.0 + 1.0; printf("M chosenMoveSubtract %d\n", (int)(p == q)); }
  { SearchParams q = p; q.chosenMovePrune = p.chosenMovePrune * 2.0 + 1.0; printf("M chosenMovePrune %d\n", (int)(p == q)); }
  { SearchParams q = p; q.useLcbForSelection = !p.useLcbForSelection; printf("M useLcbForSelection %d\n", (int)(p == q)); }
  { SearchParams q = p; q.lcbStdevs = p.lcbStdevs * 2.0 + 1.0; printf("M lcbStdevs %d\n", (int)(p == q)); }
  { SearchParams q = p; q.minVisitPropForLCB = p.minVisitPropForLCB * 2.0 + 1.0; printf("M minVisitPropForLCB %d\n", (int)(p == q)); }
  { SearchParams q = p; q.useNonBuggyLcb = !p.useNonBuggyLcb; printf("M useNonBuggyLcb %d\n", (int)(p == q)); }
  { SearchParams q = p; q.rootEndingBonusPoints = p.rootEndingBonusPoints * 2.0 + 1.0; printf("M rootEndingBonusPoints %d\n", (int)(p == q)); }
  { SearchParams q = p; q.rootPruneUselessMoves = !p.rootPruneUselessMoves; printf("M rootPruneUselessMoves %d\n", (int)(p == q)); }
  { SearchParams q = p; q.conservativePass = !p.conservativePass; printf("M conservativePass %d\n", (int)(p == q)); }
  { SearchParams q = p; q.fillDameBeforePass = !p.fillDameBeforePass; printf("M fillDameBeforePass %d\n", (int)(p == q)); }
  { SearchParams q = p; q.avoidMYTDaggerHackPla = (Player)(p.avoidMYTDaggerHackPla ^ 1); printf("M avoidMYTDaggerHackPla %d\n", (int)(p == q)); }
  { SearchParams q = p; q.wideRootNoise = p.wideRootNoise * 2.0 + 1.0; printf("M wideRootNoise %d\n", (int)(p == q)); }
  { SearchParams q = p; q.enablePassingHacks = !p.enablePassingHacks; printf("M enablePassingHacks %d\n", (int)(p == q)); }
  { SearchParams q = p; q.enableMorePassingHacks = !p.enableMorePassingHacks; printf("M enableMorePassingHacks %d\n", (int)(p == q)); }
  { SearchParams q = p; q.playoutDoublingAdvantage = p.playoutDoublingAdvantage * 2.0 + 1.0; printf("M playoutDoublingAdvantage %d\n", (int)(p == q)); }
  { SearchParams q = p; q.playoutDoublingAdvantagePla = (Player)(p.playoutDoublingAdvantagePla ^ 1); printf("M playoutDoublingAdvantagePla %d\n", (int)(p == q)); }
  { SearchParams q = p; q.avoidRepeatedPatternUtility = p.avoidRepeatedPatternUtility * 2.0 + 1.0; printf("M avoidRepeatedPatternUtility %d\n", (int)(p == q)); }
  { SearchParams q = p; q.nnPolicyTemperature = p.nnPolicyTemperature * 2.0f + 1.0f; printf("M nnPolicyTemperature %d\n", (int)(p == q)); }
  { SearchParams q = p; q.antiMirror = !p.antiMirror; printf("M antiMirror %d\n", (int)(p == q)); }
  { SearchParams q = p; q.ignorePreRootHistory = !p.ignorePreRootHistory; printf("M ignorePreRootHistory %d\n", (int)(p == q)); }
  { SearchParams q = p; q.ignoreAllHistory = !p.ignoreAllHistory; printf("M ignoreAllHistory %d\n", (int)(p == q)); }
  { SearchParams q = p; q.subtreeValueBiasFactor = p.subtreeValueBiasFactor * 2.0 + 1.0; printf("M subtreeValueBiasFactor %d\n", (int)(p == q)); }
  { SearchParams q = p; q.subtreeValueBiasTableNumShards = p.subtreeValueBiasTableNumShards + 1; printf("M subtreeValueBiasTableNumShards %d\n", (int)(p == q)); }
  { SearchParams q = p; q.subtreeValueBiasFreeProp = p.subtreeValueBiasFreeProp * 2.0 + 1.0; printf("M subtreeValueBiasFreeProp %d\n", (int)(p == q)); }
  { SearchParams q = p; q.subtreeValueBiasWeightExponent = p.subtreeValueBiasWeightExponent * 2.0 + 1.0; printf("M subtreeValueBiasWeightExponent %d\n", (int)(p == q)); }
  { SearchParams q = p; q.useEvalCache = !p.useEvalCache; printf("M useEvalCache %d\n", (int)(p == q)); }
  { SearchParams q = p; q.evalCacheMinVisits = p.evalCacheMinVisits + 1; printf("M evalCacheMinVisits %d\n", (int)(p == q)); }
  { SearchParams q = p; q.nodeTableShardsPowerOfTwo = p.nodeTableShardsPowerOfTwo + 1; printf("M nodeTableShardsPowerOfTwo %d\n", (int)(p == q)); }
  { SearchParams q = p; q.numVirtualLossesPerThread = p.numVirtualLossesPerThread * 2.0 + 1.0; printf("M numVirtualLossesPerThread %d\n", (int)(p == q)); }
  { SearchParams q = p; q.numThreads = p.numThreads + 1; printf("M numThreads %d\n", (int)(p == q)); }
  { SearchParams q = p; q.minPlayoutsPerThread = p.minPlayoutsPerThread * 2.0 + 1.0; printf("M minPlayoutsPerThread %d\n", (int)(p == q)); }
  { SearchParams q = p; q.maxVisits = p.maxVisits + 1; printf("M maxVisits %d\n", (int)(p == q)); }
  { SearchParams q = p; q.maxPlayouts = p.maxPlayouts + 1; printf("M maxPlayouts %d\n", (int)(p == q)); }
  { SearchParams q = p; q.maxTime = p.maxTime * 2.0 + 1.0; printf("M maxTime %d\n", (int)(p == q)); }
  { SearchParams q = p; q.maxVisitsPondering = p.maxVisitsPondering + 1; printf("M maxVisitsPondering %d\n", (int)(p == q)); }
  { SearchParams q = p; q.maxPlayoutsPondering = p.maxPlayoutsPondering + 1; printf("M maxPlayoutsPondering %d\n", (int)(p == q)); }
  { SearchParams q = p; q.maxTimePondering = p.maxTimePondering * 2.0 + 1.0; printf("M maxTimePondering %d\n", (int)(p == q)); }
  { SearchParams q = p; q.lagBuffer = p.lagBuffer * 2.0 + 1.0; printf("M lagBuffer %d\n", (int)(p == q)); }
  { SearchParams q = p; q.searchFactorAfterOnePass = p.searchFactorAfterOnePass * 2.0 + 1.0; printf("M searchFactorAfterOnePass %d\n", (int)(p == q)); }
  { SearchParams q = p; q.searchFactorAfterTwoPass = p.searchFactorAfterTwoPass * 2.0 + 1.0; printf("M searchFactorAfterTwoPass %d\n", (int)(p == q)); }
  { SearchParams q = p; q.treeReuseCarryOverTimeFactor = p.treeReuseCarryOverTimeFactor * 2.0 + 1.0; printf("M treeReuseCarryOverTimeFactor %d\n", (int)(p == q)); }
  { SearchParams q = p; q.overallocateTimeFactor = p.overallocateTimeFactor * 2.0 + 1.0; printf("M overallocateTimeFactor %d\n", (int)(p == q)); }
  { SearchParams q = p; q.midgameTimeFactor = p.midgameTimeFactor * 2.0 + 1.0; printf("M midgameTimeFactor %d\n", (int)(p == q)); }
  { SearchParams q = p; q.midgameTurnPeakTime = p.midgameTurnPeakTime * 2.0 + 1.0; printf("M midgameTurnPeakTime %d\n", (int)(p == q)); }
  { SearchParams q = p; q.endgameTurnTimeDecay = p.endgameTurnTimeDecay * 2.0 + 1.0; printf("M endgameTurnTimeDecay %d\n", (int)(p == q)); }
  { SearchParams q = p; q.obviousMovesTimeFactor = p.obviousMovesTimeFactor * 2.0 + 1.0; printf("M obviousMovesTimeFactor %d\n", (int)(p == q)); }
  { SearchParams q = p; q.obviousMovesPolicyEntropyTolerance = p.obviousMovesPolicyEntropyTolerance * 2.0 + 1.0; printf("M obviousMovesPolicyEntropyTolerance %d\n", (int)(p == q)); }
  { SearchParams q = p; q.obviousMovesPolicySurpriseTolerance = p.obviousMovesPolicySurpriseTolerance * 2.0 + 1.0; printf("M obviousMovesPolicySurpriseTolerance %d\n", (int)(p == q)); }
  { SearchParams q = p; q.futileVisitsThreshold = p.futileVisitsThreshold * 2.0 + 1.0; printf("M futileVisitsThreshold %d\n", (int)(p == q)); }
  { SearchParams q = p; q.humanSLProfile.initialized = !q.humanSLProfile.initialized; printf("M humanSLProfile %d\n", (int)(p == q)); }
  { SearchParams q = p; q.humanSLCpuctExploration = p.humanSLCpuctExploration * 2.0 + 1.0; printf("M humanSLCpuctExploration %d\n", (int)(p == q)); }
  { SearchParams q = p; q.humanSLCpuctPermanent = p.humanSLCpuctPermanent * 2.0 + 1.0; printf("M humanSLCpuctPermanent %d\n", (int)(p == q)); }
  { SearchParams q = p; q.humanSLRootExploreProbWeightless = p.humanSLRootExploreProbWeightless * 2.0 + 1.0; printf("M humanSLRootExploreProbWeightless %d\n", (int)(p == q)); }
  { SearchParams q = p; q.humanSLRootExploreProbWeightful = p.humanSLRootExploreProbWeightful * 2.0 + 1.0; printf("M humanSLRootExploreProbWeightful %d\n", (int)(p == q)); }
  { SearchParams q = p; q.humanSLPlaExploreProbWeightless = p.humanSLPlaExploreProbWeightless * 2.0 + 1.0; printf("M humanSLPlaExploreProbWeightless %d\n", (int)(p == q)); }
  { SearchParams q = p; q.humanSLPlaExploreProbWeightful = p.humanSLPlaExploreProbWeightful * 2.0 + 1.0; printf("M humanSLPlaExploreProbWeightful %d\n", (int)(p == q)); }
  { SearchParams q = p; q.humanSLOppExploreProbWeightless = p.humanSLOppExploreProbWeightless * 2.0 + 1.0; printf("M humanSLOppExploreProbWeightless %d\n", (int)(p == q)); }
  { SearchParams q = p; q.humanSLOppExploreProbWeightful = p.humanSLOppExploreProbWeightful * 2.0 + 1.0; printf("M humanSLOppExploreProbWeightful %d\n", (int)(p == q)); }
  { SearchParams q = p; q.humanSLChosenMoveProp = p.humanSLChosenMoveProp * 2.0 + 1.0; printf("M humanSLChosenMoveProp %d\n", (int)(p == q)); }
  { SearchParams q = p; q.humanSLChosenMoveIgnorePass = !p.humanSLChosenMoveIgnorePass; printf("M humanSLChosenMoveIgnorePass %d\n", (int)(p == q)); }
  { SearchParams q = p; q.humanSLChosenMovePiklLambda = p.humanSLChosenMovePiklLambda * 2.0 + 1.0; printf("M humanSLChosenMovePiklLambda %d\n", (int)(p == q)); }
}

static void unchangeable(const char* name, const SearchParams& a, const SearchParams& b) {
  try {
    SearchParams::failIfParamsDifferOnUnchangeableParameter(a, b);
    printf("U %s OK\n", name);
  }
  catch(const StringError& e) {
    printf("U %s ERR %s\n", name, e.what());
  }
}

int main() {
  vector<pair<string,SearchParams>> presets;
  presets.push_back(make_pair(string("default"), SearchParams()));
  presets.push_back(make_pair(string("forTestsV1"), SearchParams::forTestsV1()));
  presets.push_back(make_pair(string("forTestsV2"), SearchParams::forTestsV2()));
  presets.push_back(make_pair(string("basicDecentParams"), SearchParams::basicDecentParams()));
  for(auto& p : presets)
    dump(p.first.c_str(), p.second);
  for(auto& a : presets)
    for(auto& b : presets)
      printf("E %s %s %d %d\n", a.first.c_str(), b.first.c_str(), (int)(a.second == b.second), (int)(a.second != b.second));

  SearchParams base = SearchParams::forTestsV2();
  unchangeable("same", base, base);
  { SearchParams d = base; d.nodeTableShardsPowerOfTwo += 1; unchangeable("nodeTableShardsPowerOfTwo", base, d); }
  { SearchParams d = base; d.useEvalCache = !d.useEvalCache; unchangeable("useEvalCache", base, d); }
  { SearchParams d = base; d.evalCacheMinVisits += 1; unchangeable("evalCacheMinVisits", base, d); }
  { SearchParams d = base; d.maxVisits += 1; d.numThreads += 3; unchangeable("changeableOnly", base, d); }
  { SearchParams d = base; d.nodeTableShardsPowerOfTwo += 1; d.useEvalCache = !d.useEvalCache; d.evalCacheMinVisits += 1; unchangeable("allThree", base, d); }
  { SearchParams d = base; d.useEvalCache = !d.useEvalCache; d.evalCacheMinVisits += 1; unchangeable("evalCacheBoth", base, d); }
  return 0;
}
