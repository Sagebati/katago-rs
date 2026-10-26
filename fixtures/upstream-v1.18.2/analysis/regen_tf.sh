#!/bin/bash
# Regenerates the transformer-net analysis goldens: the same deterministic invocations as regen.sh (upstream's
# runcmdtests.sh loop: maxVisits=100, rootSymmetryPruning=false) with b10c384h6nbttflrs (a model v15 transformer
# net, katago-nn-burn feature `net-b10c384h6nbttflrs`) in place of g170-b6c96, on the Eigen (CPU, fp32) reference
# build. Upstream has no analysis golden for a transformer net; these are our own. basic.txt includes 7x7 boards
# (the attention masked to a smaller board inside the 19x19 graph). crates/katago-cli/tests/analysis_goldens.rs
# replays them.
#
#   usage:  fixtures/upstream-v1.18.2/analysis/regen_tf.sh [outdir]
#   env:  KATAGO=build/eigen/katago (just katago-eigen)   LOGDIR=<scratch dir for logs>
#         TF_MODEL=<path to b10c384h6nbttflrs.bin.gz> (default models/nets/)
#
# Output: b10c384h6nbttflrs_<name>.stdout (stdout only, as regen.sh).
set -eu
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
KATAGO=$ROOT/${KATAGO:-build/eigen/katago}
OUT=$(realpath -m "${1:-$ROOT/fixtures/upstream-v1.18.2/analysis}")
LOGDIR=${LOGDIR:-/tmp/katago-rs-analysis-regen-tf}
TF_MODEL=$(realpath "${TF_MODEL:-$ROOT/models/nets/b10c384h6nbttflrs.bin.gz}")
mkdir -p "$OUT" "$LOGDIR"
LOGDIR=$(realpath "$LOGDIR")
WORK=$LOGDIR/cwd; rm -rf "$WORK"; mkdir -p "$WORK/models"
ln -s "$ROOT/upstream/KataGo/cpp/configs" "$WORK/configs"
ln -s "$ROOT/upstream/KataGo/cpp/tests" "$WORK/tests"
ln -s "$TF_MODEL" "$WORK/models/b10c384h6nbttflrs.bin.gz"
cd "$WORK"
CFG=configs/analysis_example.cfg
MODEL=models/b10c384h6nbttflrs.bin.gz
COMMON="logDir=, logTimeStamp=false, logAllRequests=true, logAllResponses=true, logSearchInfo=true, numAnalysisThreads=1, numSearchThreadsPerAnalysisThread=1, nnRandomize=false, nnRandSeed=analysisTest, forDeterministicTesting=true, cudaUseFP16 = false, trtUseFP16 = false, openclUseFP16 = false, cudaUseNHWC = false"
LOOP="maxVisits=100, maxPlayouts=10000, rootSymmetryPruning=false"

run() {
  local name=$1 input=$2 extra=$3
  echo "b10c384h6nbttflrs_$name" >&2
  "$KATAGO" analysis -config "$CFG" -model "$MODEL" -override-config "logFile=$LOGDIR/tf_$name.log, $COMMON, $extra" \
    < "tests/analysis/$input" > "$OUT/b10c384h6nbttflrs_$name.stdout" 2> "$LOGDIR/tf_$name.stderr"
}
run basic basic.txt "$LOOP"
run symmetry symmetry.txt "$LOOP"
