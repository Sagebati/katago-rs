#!/bin/bash
# Regenerates the b18c384nbt analysis goldens (M7): the same deterministic invocations as regen.sh (upstream's
# runcmdtests.sh loop: maxVisits=100, rootSymmetryPruning=false) with kata1-b18c384nbt in place of g170-b6c96, on
# the Eigen (CPU, fp32) reference build. Upstream has no analysis golden for this net; these are our own.
# packages/katago-web's `just web-test-b18` replays them in the browser.
#
#   usage:  fixtures/upstream-v1.16.5/analysis/regen_b18.sh [outdir]
#   env:  KATAGO=build/eigen/katago (just katago-eigen)   LOGDIR=<scratch dir for logs>
#         B18_MODEL=<path to kata1-b18c384nbt.bin.gz>
#
# Output: b18_<name>.stdout (stdout only, as regen.sh).
set -eu
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
KATAGO=$ROOT/${KATAGO:-build/eigen/katago}
OUT=$(realpath -m "${1:-$ROOT/fixtures/upstream-v1.16.5/analysis}")
LOGDIR=${LOGDIR:-/tmp/katago-rs-analysis-regen-b18}
B18_MODEL=${B18_MODEL:-/home/sam/src/web/katago-ws/bench/models/kata1-b18c384nbt.bin.gz}
mkdir -p "$OUT" "$LOGDIR"
LOGDIR=$(realpath "$LOGDIR")
WORK=$LOGDIR/cwd; rm -rf "$WORK"; mkdir -p "$WORK/models"
ln -s "$ROOT/upstream/KataGo/cpp/configs" "$WORK/configs"
ln -s "$ROOT/upstream/KataGo/cpp/tests" "$WORK/tests"
ln -s "$B18_MODEL" "$WORK/models/kata1-b18c384nbt.bin.gz"
cd "$WORK"
CFG=configs/analysis_example.cfg
MODEL=models/kata1-b18c384nbt.bin.gz
COMMON="logDir=, logTimeStamp=false, logAllRequests=true, logAllResponses=true, logSearchInfo=true, numAnalysisThreads=1, numSearchThreadsPerAnalysisThread=1, nnRandomize=false, nnRandSeed=analysisTest, forDeterministicTesting=true, cudaUseFP16 = false, trtUseFP16 = false, openclUseFP16 = false, cudaUseNHWC = false"
LOOP="maxVisits=100, maxPlayouts=10000, rootSymmetryPruning=false"

run() {
  local name=$1 input=$2 extra=$3
  echo "b18_$name" >&2
  "$KATAGO" analysis -config "$CFG" -model "$MODEL" -override-config "logFile=$LOGDIR/b18_$name.log, $COMMON, $extra" \
    < "tests/analysis/$input" > "$OUT/b18_$name.stdout" 2> "$LOGDIR/b18_$name.stderr"
}
run basic basic.txt "$LOOP"
run symmetry symmetry.txt "$LOOP"
run symmetry_with_pruning symmetry.txt "maxVisits=100, maxPlayouts=10000, rootSymmetryPruning=true"
# not pvvisits.txt: two 1000-visit searches (about 10 minutes on the Eigen b18 here, minutes more in the browser at batch 1)
