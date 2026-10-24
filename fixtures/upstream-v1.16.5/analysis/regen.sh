#!/bin/bash
# Regenerates the Eigen (CPU, fp32) analysis goldens: the primary M3 oracle (docs/DECISIONS.md D9).
# Mirrors the `katago analysis` invocations of upstream/KataGo/cpp/runcmdtests.sh (v1.16.5) literally: the
# commands run from a scratch directory whose configs/ and tests/ are symlinks to upstream/KataGo/cpp/{configs,tests}
# (and models/b18c384nbt-humanv0.bin.gz -> $HUMAN_MODEL), so the command lines, the model names echoed by
# `query_models` and every override are byte-for-byte upstream's. Only logFile moves to the scratch directory.
#
#   usage (from anywhere):  fixtures/upstream-v1.16.5/analysis/regen.sh [outdir]
#   env:  KATAGO=build/eigen/katago (just katago-eigen)   LOGDIR=<scratch dir for logs>
#         HUMAN_MODEL=<path to b18c384nbt-humanv0.bin.gz>  (only for humansl_sidetomove; skipped if missing)
#
# Output names follow upstream's tests/results/analysis/ so `diff` against upstream is direct.
# Only stdout is stored (upstream's stderr = the log, which has GPU names, paths, timings).
set -eu
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
KATAGO=$ROOT/${KATAGO:-build/eigen/katago}
OUT=$(realpath -m "${1:-$ROOT/fixtures/upstream-v1.16.5/analysis}")
LOGDIR=${LOGDIR:-/tmp/katago-rs-analysis-regen}
HUMAN_MODEL=${HUMAN_MODEL:-/home/sam/src/web/baduk_review/backend/docker/models/b18c384nbt-humanv0.bin.gz}
mkdir -p "$OUT" "$LOGDIR"
LOGDIR=$(realpath "$LOGDIR")
# cwd with upstream's relative layout
WORK=$LOGDIR/cwd; rm -rf "$WORK"; mkdir -p "$WORK/models"
ln -s "$ROOT/upstream/KataGo/cpp/configs" "$WORK/configs"
ln -s "$ROOT/upstream/KataGo/cpp/tests" "$WORK/tests"
[ -f "$HUMAN_MODEL" ] && ln -s "$HUMAN_MODEL" "$WORK/models/b18c384nbt-humanv0.bin.gz"
cd "$WORK"
CPP=.
CFG=configs/analysis_example.cfg
M6=tests/models/g170-b6c96-s175395328-d26788732.bin.gz
M10=tests/models/g170e-b10c128-s1141046784-d204142634.bin.gz

COMMON="logDir=, logTimeStamp=false, logAllRequests=true, logAllResponses=true, logSearchInfo=true, numAnalysisThreads=1, numSearchThreadsPerAnalysisThread=1, nnRandomize=false, nnRandSeed=analysisTest, forDeterministicTesting=true, cudaUseFP16 = false, trtUseFP16 = false, openclUseFP16 = false, cudaUseNHWC = false"

# run <name> <input file> <model> <override-config suffix> [extra katago args]
run() {
  local name=$1 input=$2 model=$3 extra=$4; shift 4
  echo "$name" >&2
  "$KATAGO" analysis -config "$CFG" -model "$model" "$@" -override-config "logFile=$LOGDIR/$name.log, $COMMON, $extra" \
    < "$input" > "$OUT/$name.stdout" 2> "$LOGDIR/$name.stderr"
}

# The `for CMDFILE in tests/analysis/*.txt` loop: maxVisits=100, rootSymmetryPruning=false.
for f in badoverride basic clear_cache evalcache pvvisits symmetry; do
  run "$f.txt" "tests/analysis/$f.txt" "$M6" "maxVisits=100, maxPlayouts=10000, rootSymmetryPruning=false"
done
run basic_sidetomove "tests/analysis/basic.txt" "$M6" "maxVisits=100, maxPlayouts=10000, rootSymmetryPruning=false, reportAnalysisWinratesAs=SIDETOMOVE"
run symmetry_with_pruning "tests/analysis/symmetry.txt" "$M6" "maxVisits=100, maxPlayouts=10000, rootSymmetryPruning=true"
# NOTE: this one has no maxVisits override (analysis_example.cfg's maxVisits=500 applies, as upstream).
run pvvisits_nograph "tests/analysis/pvvisits.txt" "$M6" "rootSymmetryPruning=false, useGraphSearch=false"
if [ -f "$HUMAN_MODEL" ]; then
  run humansl_sidetomove "tests/analysis/humansl.txt.noauto" "$M6" "maxVisits=100, maxPlayouts=10000, rootSymmetryPruning=false, reportAnalysisWinratesAs=SIDETOMOVE, humanSLProfile = preaz_18k" -human-model models/b18c384nbt-humanv0.bin.gz
else
  echo "skip humansl_sidetomove: HUMAN_MODEL=$HUMAN_MODEL not found" >&2
fi

# eval_cache: upstream runs this through fifos (the harness feeds line N+1 only after the response
# to line N arrived, so the eval cache is populated by the previous request) and throws stdout away.
# We do the same, but keep the responses. rootSymmetryPruning=false is runcmdtests.sh:71's (it is in the command line there,
# not in the loop's overrides).
(
  T=$LOGDIR/evalcachecmdtest
  rm -rf "$T"; mkdir -p "$T"; mkfifo "$T/input" "$T/feedback"
  echo eval_cache >&2
  "$KATAGO" analysis -config "$CFG" -model "$M10" -override-config "logFile=$LOGDIR/eval_cache.txt.log, $COMMON, rootSymmetryPruning=false, useGraphSearch=true, useEvalCache=true, evalCacheMinVisits=10" \
    < "$T/input" 2> "$LOGDIR/eval_cache.stderr" | tee "$OUT/eval_cache.stdout" > "$T/feedback" &
  { head -1 "tests/analysis/evalcache.txt"; tail -n +2 "tests/analysis/evalcache.txt" | while read -r L; do read -r _ < "$T/feedback"; echo "$L"; done; } > "$T/input"
  wait
  rm -r "$T"
)
