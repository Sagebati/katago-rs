#!/bin/bash
# Regenerates every fixtures/upstream-v1.16.5/unit/testsearch*/ stem (M3 search test oracles).
#
#   usage (from anywhere):  fixtures/upstream-v1.16.5/unit/testsearchcommon/regen.sh
#   env:   KATAGO=build/eigen/katago (just katago-eigen)   SCRATCH=<scratch dir>   JOBS=<parallel runs, default 6>
#          UNIT=<output dir, default fixtures/upstream-v1.16.5/unit>
#          EXTRA_MODELS=1  also run the V9 tests with kata1-b18c384nbt / b40c256 if found (very slow on CPU; off by default)
#
# What it writes, per stem (<stem>/INDEX.tsv has the same 4 columns as tools/extract_raw_blocks.py):
#   1. role=input rows: the R"%%(...)%%" board diagrams, via tools/extract_raw_blocks.py (testsearchnonn, testsearchv3,
#      testsearchv8, testsearchv9, testsearchmisc). testsearch.cpp / testsearchcommon.cpp have no raw blocks.
#      NONE of these files has an R"%%(...)%%" *expected* block: their expected output is the stdout of the
#      `katago run*tests` subcommands, kept upstream in cpp/tests/results/*.txt.
#   2. role=expected rows, NN-less (testsearchnonn/runnnlesssearchtests.expected0.txt): the slice of
#      `katago runoutputtests` stdout from "Running neuralnetless search tests" through its "Done" line
#      (INDEX line = first line of that slice in upstream's results/runOutputTests.txt). debugSkipNeuralNet, so
#      backend independent: byte-identical to upstream's own runOutputTests.txt slice (checked at the end).
#   3. role=expected rows, with a net: stdout of the upstream test subcommand run with OUR Eigen CPU fp32 build
#      (the primary oracle, DECISIONS.md D9). Upstream's cpp/tests/results/runSearchTests*.txt were made on CUDA
#      (RTX A5000/V100) and differ in float last digits and, through them, visit distributions.
#      Lines starting with ": " are backend log lines (GPU/model info), not part of the test output.
#      The Eigen backend rejects NHWC=false and FP16, so the INPUTSNHWC and CUDANHWC args are `true true`.
set -eu
cd "$(dirname "$0")/../../../.."
ROOT=$PWD
KATAGO=$ROOT/${KATAGO:-build/eigen/katago}
UNIT=${UNIT:-$ROOT/fixtures/upstream-v1.16.5/unit}
TESTS=$ROOT/upstream/KataGo/cpp/tests
M=$TESTS/models
SCRATCH=${SCRATCH:-/tmp/katago-rs-search-regen}
JOBS=${JOBS:-6}
G103=$M/g103-b6c96-s103408384-d26419149.txt.gz
RUN4=$M/run4-s67105280-d24430742-b6c96.txt.gz
GRUN2=$M/grun2-b6c96-s128700160-d49811312.txt.gz
# (The g170 .txt.gz gives byte-identical test output to the .bin.gz, minus the "Model name" log line: only .bin.gz is run.)
G170B=$M/g170-b6c96-s175395328-d26788732.bin.gz

# cwd for the runs: runoutputtests reads tests/data and tests/models relative to cwd and may write; keep it out of upstream/.
rm -rf "$SCRATCH/cwd" "$SCRATCH/out"; mkdir -p "$SCRATCH/cwd/tests/results" "$SCRATCH/out"
for d in "$TESTS"/*; do b=$(basename "$d"); case $b in results|models) ;; *) cp -r "$d" "$SCRATCH/cwd/tests/";; esac; done
ln -s "$M" "$SCRATCH/cwd/tests/models"
cd "$SCRATCH/cwd"

# ---- 1. inputs -------------------------------------------------------------------------------------------------------
cd "$ROOT"
for f in testsearchnonn testsearchv3 testsearchv8 testsearchv9 testsearchmisc; do
  python3 -I tools/extract_raw_blocks.py "$UNIT" "$TESTS/$f.cpp"
done
mkdir -p "$UNIT/testsearch"
cd "$SCRATCH/cwd"

# ---- 3. Eigen reruns (parallel) --------------------------------------------------------------------------------------
# job <stem> <slug> <test label for INDEX> <katago args...>      -> $SCRATCH/out/<stem>/<slug>.expected0.txt
job() {
  local stem=$1 slug=$2 label=$3; shift 3
  mkdir -p "$SCRATCH/out/$stem"
  echo "run $stem/$slug: katago $*" >&2
  "$KATAGO" "$@" > "$SCRATCH/out/$stem/$slug.expected0.txt" 2> "$SCRATCH/out/$stem/$slug.stderr"
  printf '%s\t%s\n' "$slug.expected0.txt" "$label" > "$SCRATCH/out/$stem/$slug.row"
}
bg() { while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n; done; "$@" & }
run_all() {
  bg job testsearch runsearchtests-g103-sym0 "runSearchTests g103 symmetry 0 (Eigen rerun)" runsearchtests "$G103" true true 0 false
  bg job testsearch runsearchtests-g103-sym1 "runSearchTests g103 symmetry 1 (Eigen rerun)" runsearchtests "$G103" true true 1 false
  bg job testsearch runsearchtests-run4-sym0 "runSearchTests run4 symmetry 0 (Eigen rerun)" runsearchtests "$RUN4" true true 0 false
  bg job testsearch runsearchtests-run4-sym1 "runSearchTests run4 symmetry 1 (Eigen rerun)" runsearchtests "$RUN4" true true 1 false
  bg job testsearchv3 runsearchtestsv3-grun2-sym0 "runSearchTestsV3 grun2 symmetry 0 (Eigen rerun)" runsearchtestsv3 "$GRUN2" true true 0 false
  bg job testsearchv3 runsearchtestsv3-grun2-sym1 "runSearchTestsV3 grun2 symmetry 1 (Eigen rerun)" runsearchtestsv3 "$GRUN2" true true 1 false
  bg job testsearchv3 runsearchtestsv3-grun2-sym5 "runSearchTestsV3 grun2 symmetry 5 (Eigen rerun)" runsearchtestsv3 "$GRUN2" true true 5 false
  bg job testsearchv8 runsearchtestsv8-g170-bin "runSearchTestsV8 g170-b6c96 .bin.gz (Eigen rerun)" runsearchtestsv8 "$G170B" true true false
  bg job testsearchv9 runsearchtestsv9-g170-bin "runSearchTestsV9 g170-b6c96 .bin.gz (Eigen rerun)" runsearchtestsv9 "$G170B" true true false
  bg job testsearchmisc runnnontinyboardtest-g170-bin-sym0 "runNNOnTinyBoard g170-b6c96 .bin.gz symmetry 0 (Eigen rerun)" runnnontinyboardtest "$G170B" true true 0 false
  bg job testsearchmisc runnnsymmetriestest-g103 "runNNSymmetries g103 (Eigen rerun)" runnnsymmetriestest "$G103" true true false
  bg job testsearchnonn runoutputtests-full "(whole runoutputtests, sliced below)" runoutputtests
  wait
}
run_all
if [ "${EXTRA_MODELS:-0}" = 1 ]; then
  B18=${B18:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)/../katago-ws/bench/models/kata1-b18c384nbt.bin.gz}
  B40=${B40:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)/../katago-ws/bench/models/kata1-b40c256.bin.gz}
  bg job testsearchv9 runsearchtestsv9-b18nbt "runSearchTestsV9 kata1-b18c384nbt (Eigen rerun)" runsearchtestsv9 "$B18" true true false
  bg job testsearchv9 runsearchtestsv9-b40 "runSearchTestsV9 kata1-b40c256 (Eigen rerun)" runsearchtestsv9 "$B40" true true false
  wait
fi

# ---- 2. NN-less slice ------------------------------------------------------------------------------------------------
cd "$SCRATCH/out"
python3 -I - "$SCRATCH/out/testsearchnonn/runoutputtests-full.expected0.txt" "$SCRATCH/out/testsearchnonn/runnnlesssearchtests.expected0.txt" <<'PY'
import sys
lines = open(sys.argv[1], encoding='utf-8', newline='').read().split('\n')
a = next(i for i, l in enumerate(lines) if l == 'Running neuralnetless search tests')
b = next(i for i in range(a, len(lines)) if lines[i] == 'Done')
open(sys.argv[2], 'w', encoding='utf-8', newline='').write('\n'.join(lines[a:b + 1]) + '\n')
print('runNNLessSearchTests slice: output lines %d..%d' % (a + 1, b + 1))
PY
rm "$SCRATCH/out/testsearchnonn/runoutputtests-full.expected0.txt" "$SCRATCH/out/testsearchnonn/runoutputtests-full.row"
printf '%s\t%s\n' runnnlesssearchtests.expected0.txt "runNNLessSearchTests (katago runoutputtests, identical to upstream)" > "$SCRATCH/out/testsearchnonn/runnnlesssearchtests.row"
# sanity: identical to upstream's own file (generated on CUDA box, but no NN involved)
python3 -I - "$SCRATCH/out/testsearchnonn/runnnlesssearchtests.expected0.txt" "$TESTS/results/runOutputTests.txt" <<'PY'
import sys
mine = open(sys.argv[1], encoding='utf-8', newline='').read()
up = open(sys.argv[2], encoding='utf-8', newline='').read()
assert mine in up, 'NN-less search slice differs from upstream runOutputTests.txt'
print('NN-less slice is a verbatim substring of upstream results/runOutputTests.txt at line', up[:up.index(mine)].count('\n') + 1)
PY

# ---- install + INDEX rows --------------------------------------------------------------------------------------------
for dir in "$SCRATCH"/out/*/; do
  stem=$(basename "$dir")
  mkdir -p "$UNIT/$stem"
  [ -f "$UNIT/$stem/INDEX.tsv" ] || printf 'file\tline\ttest\trole\n' > "$UNIT/$stem/INDEX.tsv"
  # drop expected rows from an earlier run, then re-add
  grep -v '	expected$' "$UNIT/$stem/INDEX.tsv" > "$SCRATCH/idx.tmp" || true
  for row in $(ls "$dir"*.row | sort); do
    file=$(cut -f1 "$row"); label=$(cut -f2 "$row")
    cp "$dir$file" "$UNIT/$stem/$file"
    line=1; [ "$file" = runnnlesssearchtests.expected0.txt ] && line=$(python3 -I -c "
up=open('$TESTS/results/runOutputTests.txt',encoding='utf-8',newline='').read().split('\n'); print(up.index('Running neuralnetless search tests')+1)")
    printf '%s\t%s\t%s\texpected\n' "$file" "$line" "$label" >> "$SCRATCH/idx.tmp"
  done
  cp "$SCRATCH/idx.tmp" "$UNIT/$stem/INDEX.tsv"
done
echo done >&2
