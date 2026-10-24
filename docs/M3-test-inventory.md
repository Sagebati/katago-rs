# M3 test inventory (search + analysis protocol)

Scope: every upstream search test (`cpp/tests/testsearch*.cpp`) and every analysis golden
(`cpp/tests/analysis/*.txt`), with what each one needs and which oracle to use. Generated while
preparing the oracles; the fixtures named here are checked in and never hand-edited.
Regenerate with `fixtures/upstream-v1.16.5/analysis/regen.sh` and
`fixtures/upstream-v1.16.5/unit/testsearchcommon/regen.sh` (both run the Eigen build,
`just katago-eigen`; the second takes about 20 minutes, V8 dominates).

## 1. Findings that change the plan

1. **No `testsearch*.cpp` file has an `R"%%(...)%%"` expected block.** All 72 raw blocks are
   *inputs* (board diagrams for `Board::parseBoard`): nonn 32, v3 7, v8 21, v9 10, misc 2;
   `testsearch.cpp` and `testsearchcommon.cpp` have none. Their expected output is the stdout of
   `katago run*tests` subcommands, which upstream keeps in `cpp/tests/results/*.txt`. So
   `extract_raw_blocks.py` yields inputs only; the *expected* side is the new fixtures in section 3.
   The `fixture_coverage.rs` allowlist therefore gained 12 entries for the `expected` rows added
   to INDEX.tsv (reason text starts "M3 pending: ...").
2. **The NN-less search tests are a backend-independent, byte-exact oracle.** `Tests::runNNLessSearchTests`
   (testsearchnonn.cpp) runs with `debugSkipNeuralNet`, so there is no network and no CUDA/Eigen
   difference. Our Eigen rerun of `katago runoutputtests` is byte-identical to upstream's
   `results/runOutputTests.txt` (23682 of 23683 lines; the one missing line, "All task parsing tests
   passed!", is printed by `runTaskParsingTests` only in a `BUILD_DISTRIBUTED` build). The 3725-line slice is
   `unit/testsearchnonn/runnnlesssearchtests.expected0.txt`. NNEvaluator's `debugSkipNeuralNet`
   path is already ported (`oracle/nneval_skipnn`), so this is the first M3 target.
3. **Upstream does not re-check any of these on a CPU backend.** `.github/workflows/build.yml` only runs
   `./katago runtests` (no models, none of the search tests). The `results/*.txt` files are refreshed by hand
   on GPU boxes (headers: NVIDIA RTX A5000, a few old fp16 ones Tesla V100; commits "Update tests", "Update
   tests for cuda 12.1"), in many variants (`iNHWC`/`cNHWC`/`FP16`/symmetry `s1`). They differ from an fp32 CPU run in
   float last digits and, through the search, sometimes in visit distributions (section 5). Hence D9: the primary
   oracle is our Eigen rerun, upstream's file is only secondary.
4. **Only v8+ nets are reachable.** `runsearchtests` (v1) uses `g103` (model v5, input version 4) and
   `run4` (model v3, input version 3); `runsearchtestsv3` uses `grun2` (model v3). The port has `fillRowV7`
   only (v8..v16 nets), `tools/bin2onnx` does not verify v3-v5 nets, and `NNEvaluator` returns Err for
   `inputsVersion != 7`. Those tests are parked (priority 6) unless fillRowV3/V4 are ported. The usable nets
   are the two v8 ones: `g170-b6c96` (every V8/V9 test, all analysis goldens but one) and `g170e-b10c128`
   (`eval_cache`).
5. **Eigen quirks to know when rerunning:** the Eigen backend refuses `useNHWC=false` and FP16, so the upstream
   `INPUTSNHWC CUDANHWC` arguments are passed as `true true` (outputs do not depend on them). Log lines that start
   with `: ` ("Eigen (CPU) backend thread 0: Model version ...") are backend log output, not test output: filter
   them before comparing. A model's text (`.txt.gz`) and binary (`.bin.gz`) forms give identical test output.
6. **Several tests depend on things the port deliberately lacks** (details in section 4): `CompactSgf`
   (`dataio/sgf` is `[[ignored]]`; `tests/testnninputs.rs` already carries a test-local subset), `GameInitializer`
   (`program/play.cpp` is `[[ignored]]`), `Setup::loadSingleParams` for GTP/MATCH/BENCHMARK/OTHER/DISTRIBUTED
   (only the analysis branch is ported), the human-SL net (`bin2onnx` rejected `metaEncoderVersion > 0` until M8; resolved, section 7).

## 2. Analysis goldens (`fixtures/upstream-v1.16.5/analysis/`)

Invocation (all from `runcmdtests.sh`, rerun literally by `regen.sh` from a cwd that mirrors `cpp/`):
`katago analysis -config configs/analysis_example.cfg -model <model> -override-config "<COMMON>, <row overrides>"`
with COMMON = `logDir=, logTimeStamp=false, logAllRequests=true, logAllResponses=true, logSearchInfo=true,
numAnalysisThreads=1, numSearchThreadsPerAnalysisThread=1, nnRandomize=false, nnRandSeed=analysisTest,
forDeterministicTesting=true, cudaUseFP16 = false, trtUseFP16 = false, openclUseFP16 = false, cudaUseNHWC = false`.
The config file adds `reportAnalysisWinratesAs = BLACK`, `nnMaxBatchSize = 64`, `nnCacheSizePowerOfTwo = 23`,
`nnMutexPoolSizePowerOfTwo = 17`. Models: b6c96 = `tests/models/g170-b6c96-s175395328-d26788732.bin.gz` (v8, small),
b10c128 = `tests/models/g170e-b10c128-s1141046784-d204142634.bin.gz` (v8, larger, 11 MB).

Determinism: every file was generated twice (concurrently, so also independent of load) and is byte-identical
run to run.

Upstream columns compare our Eigen stdout with `cpp/tests/results/analysis/<same name>`; "same shape" means the same
responses in the same order with the same moves, `order`, `visits`, `edgeVisits` and `pv` lists and the same JSON key
order and format. Order in this table is the suggested porting order.

| # | golden (stdout) | queries (`tests/analysis/`) | model | overrides beyond COMMON | what it exercises | vs upstream (CUDA) |
|---|---|---|---|---|---|---|
| 1 | `clear_cache.txt.stdout` (36 B) | `clear_cache.txt`: one `{"action":"clear_cache"}` | none needed | maxVisits=100, rootSymmetryPruning=false | protocol: `clear_cache` action echo (`Fix clear_cache crash` commit) | byte-identical |
| 2 | `badoverride.txt.stdout` (11 KB) | `badoverride.txt`: unknown `overrideSettings` key, unknown/ misspelled fields | b6c96 | maxVisits=100 | warning responses (`Unknown config params`, `Unexpected or unused field`), field order of the warnings; 2 searches of the empty 19x19 board | warnings identical; the 2 search responses differ by float digits only (scoreStdev up to 1.3e-3, winrate 3e-5) |
| 3 | `basic.txt.stdout` (9.7 KB) | `basic.txt`: 5 queries: analyzeTurns, initialStones, 7x7 with ownership + policy, ownershipStdev, 72-move 9x9 japanese with `includeNoResultValue` | b6c96 | maxVisits=100 | JSON writer (key order, float formatting), `analyzeTurns`, ownership/policy/stdev arrays, rules parsing, `rootInfo`, `moveInfos` | same shape; floats differ: winrate <= 1.1e-4, utility 4.6e-4, prior/policy 6.7e-4, ownership 1.5e-4, scoreLead 3.3e-3, scoreStdev 9.6e-3 |
| 4 | `basic_sidetomove.stdout` (9.7 KB) | `basic.txt` | b6c96 | + reportAnalysisWinratesAs=SIDETOMOVE | perspective flip of winrate/lead/utility | same as #3 |
| 5 | `symmetry.txt.stdout` (21 KB) | `symmetry.txt`: 7 queries (empty, diagonal, asymmetric, 13x7 / 7x13 rectangles with `includeMovesOwnership[Stdev]`) | b6c96 | maxVisits=100, rootSymmetryPruning=false | `moveInfos` ordering ties across symmetric moves, rectangular boards, per-move ownership | same shape; winrate <= 9.5e-5, scoreLead 3.6e-3, ownership 2.4e-4 |
| 6 | `symmetry_with_pruning.stdout` (52 KB) | `symmetry.txt` | b6c96 | rootSymmetryPruning=true | `rootSymmetryPruning` (isSymmetry dedup, `symHash`, per-move ownership averaging over symmetries) | same shape; ownership 4.6e-4, weights 2.9e-3 |
| 7 | `pvvisits.txt.stdout` (7.3 KB) | `pvvisits.txt`: 2 queries, 7x7, 1000 visits | b6c96 | maxVisits=100 (the queries' own 1000 wins), graph search on (default) | `includePVVisits`, `pvEdgeVisits`, graph search (transpositions) | **structurally different**: both responses have different visit distributions (edgeVisits/pvEdgeVisits off by up to 14, one 10-move pv vs a 9-move one, one pv move different). A 1000-visit search amplifies float noise |
| 8 | `pvvisits_nograph.stdout` (7.9 KB) | `pvvisits.txt` | b6c96 | no maxVisits on the command line (the queries say 1000), useGraphSearch=false | same without graph search (tree edge visits == visits) | **structurally different**: one response lists 10 moves vs 9, move order differs in places, edgeVisits off by up to 11 |
| 9 | `evalcache.txt.stdout` (4.7 KB) | `evalcache.txt`: A/B/C (25/27/25 moves, 50/100/50 visits) | b6c96 | maxVisits=100 (loop form, no eval cache) | repeatability: C must equal A | n/a (upstream has no golden for it; the loop run is unchecked) |
| 10 | `eval_cache.stdout` (8 KB) | `evalcache.txt` through fifos (line N+1 only after response N) | **b10c128** | useGraphSearch=true, useEvalCache=true, evalCacheMinVisits=10 | `useEvalCache`: C is *not* equal to A (root visits 50, top move R19 with 8 visits vs S13 with 42), B reuses A's cache | n/a (upstream throws stdout away and keeps only `eval_cache.txt.log`); fed by the fifo handshake in `regen.sh` |
| 11 | `humansl_sidetomove.stdout` (17 KB) | `humansl.txt.noauto`: 5 searches + `query_models` + a 91-stone position with `humanSLRootExploreProbWeightless/CpuctPermanent` | b6c96 + human net `b18c384nbt-humanv0` (model v15, outside this repo: `backend/docker/models`) | + SIDETOMOVE, humanSLProfile = preaz_18k | human-SL head, `humanScoreMean/Stdev`, `query_models` | same shape; humanScoreMean <= 2.8e-2, scoreLead 5.8e-3. `query_models` line differs by backend: `maxBatchSize` 2 vs 64, `usingFP16` "auto" vs "false". Passes since M8 (`golden_humansl_sidetomove`, maxBatchSize normalized) |

Per-golden notes for the test writer:
- Compare line by line as JSON with `serde_json` `float_roundtrip` (the `just check-features` rule), but also assert the *key order and number formatting* of the raw line (the goldens have alphabetical keys and floats with 9 significant digits, e.g. `0.656578463`, as `nlohmann::json::dump` writes them).
- Numeric tolerance, measured for Eigen vs CUDA (a reasonable first guess for Rust-vs-Eigen): winrate/lcb/utility 5e-4, prior/policy 1e-3, ownership 5e-4, scoreMean/lead/selfplay 6e-3, scoreStdev 1e-2 (the CLAUDE.md rule "values +-1e-4" is stricter than what two correct backends produce at 100 visits; the Rust-vs-Eigen gap should be smaller than CUDA-vs-Eigen but not zero, flex differs in accumulation order).
- At <= 100 visits all six search goldens that exist upstream (#2-#6, #11) agree with ours on moves, order, visits and pv. Expect Rust-vs-Eigen to agree there too, and to diverge at 500-1000 visits (#7, #8): compare those on root visits, the top move, and the first few pv moves, or use a tolerance for visit counts.
- `query_models` (#11, line 5) echoes the `-model` path as given: tests must pass the same relative names (`tests/models/...`) or mask `name`, `maxBatchSize`, `usingFP16`.

## 3. Search-test fixtures (`fixtures/upstream-v1.16.5/unit/testsearch*/`)

Each stem has `INDEX.tsv` (same four columns as `extract_raw_blocks.py`): `role=input` rows are the 72 board
blocks (appendix A), `role=expected` rows are the files below. For Eigen reruns the INDEX `line` is 1 (no source
line); for the NN-less slice it is the first line of the slice in upstream's `results/runOutputTests.txt`.

| expected fixture | command (`katago ...`, cwd = scratch copy of `cpp/`) | net | size | upstream counterpart | lines byte-identical to it / identical up to decimals |
|---|---|---|---|---|---|
| `testsearchnonn/runnnlesssearchtests.expected0.txt` | `runoutputtests`, sliced from "Running neuralnetless search tests" to "Done" | none | 267 KB (3725 lines) | `results/runOutputTests.txt` lines 17841..21564 | **100% / 100%** (verbatim) |
| `testsearchv9/runsearchtestsv9-g170-bin.expected0.txt` | `runsearchtestsv9 g170-b6c96.bin.gz true true false` | b6c96 v8 | 559 KB | `runSearchTestsV9.txt` | 67% / 96% |
| `testsearchv8/runsearchtestsv8-g170-bin.expected0.txt` | `runsearchtestsv8 g170-b6c96.bin.gz true true false` | b6c96 v8 | 500 KB | `runSearchTestsV8Bin.txt` (`V8.txt` is the `.txt.gz` run, same output) | 65% / 92% |
| `testsearchmisc/runnnontinyboardtest-g170-bin-sym0.expected0.txt` | `runnnontinyboardtest g170-b6c96.bin.gz true true 0 false` | b6c96 v8 | 0.8 KB | `runNNOnTinyBoardTest.txt` | 64% / 78% (36 lines, floats printed to many digits) |
| `testsearchmisc/runnnsymmetriestest-g103.expected0.txt` | `runnnsymmetriestest g103-b6c96...txt.gz true true false` | g103 v5 | 15 KB | `runNNSymmetriesTest.txt` | 82% / 86% |
| `testsearch/runsearchtests-g103-sym0.expected0.txt`, `-sym1` | `runsearchtests g103-b6c96...txt.gz true true <sym> false` | g103 v5 | 85 KB, 84 KB | `runSearchTests-iNHWC-cNHWC.txt`, `runSearchTests-s1.txt` | 70% / 94%, 68% / 93% |
| `testsearch/runsearchtests-run4-sym0.expected0.txt`, `-sym1` | same with `run4-s67105280-d24430742-b6c96.txt.gz` | run4 v3 | 82 KB, 81 KB | `runSearchTests-r4-iNHWC-cNHWC.txt`, `-r4-s1.txt` | 71% / 95%, 72% / 96% |
| `testsearchv3/runsearchtestsv3-grun2-sym0.expected0.txt`, `-sym1`, `-sym5` | `runsearchtestsv3 grun2-b6c96...txt.gz true true <sym> false` | grun2 v3 | 101 KB, 97 KB, 101 KB | `runSearchTestsV3-g2-iNHWC-cNHWC.txt`, `-g2-s1.txt`, `-g2-s5.txt` | 61% / 88%, 68% / 95%, 64% / 96% |

(The two percentages are `difflib` matches over non-`: ` lines: first byte-identical, then after replacing every
decimal number by `#`. What remains in the second figure is moves, visit counts, policy-visit columns, NN-row counts
and line-count shifts, i.e. where the search tree itself went a different way.)

Upstream result files with no Eigen counterpart here, and why: the many `iNCHW/iNHWC/cNCHW/cNHWC/FP16` variants (backend
layout switches, Eigen has one path; FP16 does not exist on CPU), `runSearchTestsV9B18NBT*.txt`
(`b18c384nbt-uec-20221121b`, not in the repo; the local `kata1-b18c384nbt` is a different net and a V9 run costs
hours on CPU; `EXTRA_MODELS=1` in `regen.sh` would run it), `runSearchTestsV9B40.txt` (b40c256 likewise),
`runNNOnTinyBoardTestB28.txt` (b28c512), `runNNBatchingTest*`, `runNNCanaryTests`, `runTinyNNTests` (GPU-sanity tests).
`runSelfplayInitTests`, `runSelfplayInitStatTests` (`g170` bin), `runSekiTrainWriteTests` (`g103`) belong to
`testtrainingwrite.cpp` (selfplay/training-data writer, M4+; arguments in `main.cpp`: one `MODEL_FILE`), not in this wave.
`runownershiptests` has no upstream result file.

Rerun cost on this machine (12 cores, 6 jobs in parallel): g103/run4 `runsearchtests` ~1 min each, V3 ~1.5 min,
V9 ~6 min, V8 ~20 min (the 40-thread, 16000-visit sections), `runoutputtests` 12 s.
Determinism: a second complete run is compared in section 7.

## 4. Search tests in suggested porting order (wave 2)

"Net" is what the test must load; "params" are the settings the source sets. All are `numSearchThreads=1` unless the
row says otherwise; `SearchParams::forTestsV1/V2` are already ported and oracle-tested.
Blocks are listed in appendix A; counts below are input boards.

| prio | test (source) | blocks | net / NN mode | search params | upstream functions and modules exercised | expected output / oracle | notes and blockers |
|---|---|---|---|---|---|---|---|
| **0** | `runNNLessSearchTests` (testsearchnonn.cpp, 43 sections) | 32 | none: `startNNEval(..., debugSkipNeuralNet=true)` (already ported), symmetry 0, nnRandomize off, various `nnXLen/nnYLen` (19, 9, 7x17, 7, 14, 13, 9x5) | `maxVisits` 4..5000; defaults, `forTestsV1`, `cpuctExploration *=2/4`, `rootSymmetryPruning`, `wideRootNoise 0.05`, `rootPruneUselessMoves`, `dynamicScoreUtilityFactor`, `useLcbForSelection`, `subtreeValueBiasFactor 0.5/0.8`, `useGraphSearch`, `fpuParentWeightByVisitedPolicy[Pow]`, `rootPolicyOptimism/policyOptimism`, `rootNoiseEnabled`, `chosenMoveTemperature[Early|Halflife]`; seeds "autoSearchRandSeed", "seed1", "movesown", ...; 10000 `getChosenMoveLoc` draws | `Search` (setPosition, runWholeSearch, makeMove, beginSearch, printTree, printPV, getChosenMoveLoc, setAvoidMoveUntilByLoc, setRootSymmetryPruningOnly, setRootHintLoc, setAlwaysIncludeOwnerMap, setParamsNoClearing, getPlaySelectionValues, getRootValues/PrunedRootValues/NodeValues, getAnalysisJson, chooseIndexWithTemperature, addDirichletNoise), searchnode(table), searchexplorehelpers, searchpuct, searchupdatehelpers, searchnnhelpers, searchhelpers, searchresults, searchprint, subtreevaluebiastable, distributiontable, `Setup::loadSingleParams`, `SearchParams::printParams`, Rand (exact), `Board`/`BoardHistory`/`Rules` (done) | `testsearchnonn/runnnlesssearchtests.expected0.txt`, byte-exact (CPU/GPU independent) | Split the file on the 67-`=` header trios (section list = the 43 titles; `Done` ends it). Sections 37, 39, 40, 41, 42 (SearchParams printing for GTP/Match/Benchmark/Other/Distributed) need non-analysis `Setup::loadSingleParams` branches; section 43 "Board size distribution" needs `GameInitializer` (`program/play.cpp` is `[[ignored]]`, 100000 random games): park those or port the pieces. Natural split inside the test: A basic / tree reuse / pruning (1-4), B root symmetry pruning (5-11), C non-square, noise, past game end, value bias (12-16), D `getAnalysisJson`, search results, tree walking, avoid moves (17-23), E graph search (24-26), F FPU / optimism / zero node (27-33), G chosen-move probabilities (34), H SearchParams printing (35-42), I board sizes (43) |
| 1 | analysis goldens #1-#8 (section 2) | - | b6c96 v8 via katago-nn-burn (`models/onnx/g170-b6c96.onnx`) | maxVisits 5..1000, threads 1/1, `nnRandomize=false`, `rootSymmetryPruning` on/off, `reportAnalysisWinratesAs`, `useGraphSearch` on/off | `analysis` command loop (`cpp/command/analysis.cpp`), `Setup::loadSingleParams(SETUP_FOR_ANALYSIS)` (ported), `Search` incl. `getAnalysisJson`/ `AnalysisData`, `nlohmann::json::dump` port, rules/komi/board parsing, warnings for bad fields/overrides, `NNEvaluator` with the real net | `analysis/*.stdout` (Eigen primary; upstream secondary, section 2) | do `clear_cache`, `badoverride`, `basic` first (small, 1-2 s each on CPU); #7/#8 last (1000 visits, structurally fragile) |
| 1 | analysis golden #10 `eval_cache` | - | **b10c128** v8 | `useEvalCache=true`, `evalCacheMinVisits=10`, `useGraphSearch=true` | eval cache in `Search`/`NNEvaluator` path, request pipelining (the fifo handshake means one request in flight) | `analysis/eval_cache.stdout` | the Rust test must feed query N+1 only after response N |
| 2 | `runNNOnTinyBoard` (testsearchmisc.cpp) | 1 (5x5) | b6c96 v8, `nnXLen=nnYLen=6`, sym 0 | no search; `Rules::TAX/WHB/SCORING/KO` loop | `NNEvaluator::evaluate` + `NNOutput::debugPrint` | `testsearchmisc/runnnontinyboardtest-g170-bin-sym0.expected0.txt` (Eigen; 0.8 KB) | cheap end-to-end smoke test for the NN path; M2-level, listed because it lives in this file |
| 3 | `runV9Positions` (testsearchv9.cpp, 8 sections) | 10 | b6c96 v8, `nnXLen=nnYLen=19`, default symmetry 4, no randomization | `forTestsV2`, maxVisits 5..600 (mostly 100), `rootFpuReductionMax`, `conservativePass`, `enableMorePassingHacks` (8x), `enablePassingHacks`, `useNonBuggyLcb`, `minVisitPropForLCB/lcbStdevs`, `playoutDoublingAdvantage`, `rootSymmetryPruning`; prints play selection values + sharp score/error | `AsyncBot` (setParams, genMoveSynchronous), `Search`, `PlayUtils::getFullSymmetryNNOutput`, spight-rules ladder feature, `NNEvaluator::evaluate`, `TestSearchOptions` printers (`printRootPolicyMap`, `printRootOwnershipMap`, `getSharpScore`, `getShallowAverageShorttermWLAndScoreError`) | `testsearchv9/runsearchtestsv9-g170-bin.expected0.txt` (Eigen, 559 KB); secondary `runSearchTestsV9.txt` | first section (and the shared `runBotOnSgf`) needs a `CompactSgf` subset (test-local copy in `tests/testnninputs.rs`); the b18nbt / b40 upstream variants are not reproducible here |
| 4 | `runV8Tests*` (testsearchv8.cpp, ~30 sections in 7 functions) | 21 | b6c96 v8, several evaluators: 19x19, 9x9 (+exact `requireExactNNLen`), symmetry 1, 2, 5 and **-1 (nnRandomize with `Rand` seeded "runSearchTestsRandSeedv8seed")** | `forTestsV1`, maxVisits 1..16000, `maxPlayouts[Pondering]`, `rootNoiseEnabled`, `valueWeightExponent`, `playoutDoublingAdvantage[Pla]`, `chosenMovePrune/Subtract`, `antiMirror`, `futileVisitsThreshold`, `subtreeValueBias*`, `rootNumSymmetriesToSample`, `useNoisePruning`, `fillDameBeforePass`, `avoidRepeatedPatternUtility`, `useGraphSearch`, `numThreads` 1/4/30/40 | everything above plus pondering / `setRootHintLoc`, `genMoveSynchronous`, `PlayUtils::maybeFriendlyPass/maybeCleanupBeforePass`, local pattern bonus tables (`localpattern.rs`), `searchmultithreadhelpers` | `testsearchv8/runsearchtestsv8-g170-bin.expected0.txt` (Eigen, 500 KB); secondary `runSearchTestsV8Bin.txt` | the 2 "Multithreaded tree updating" / "Multithreaded (graph) search" sections print thread-timing-dependent values (upstream itself skips them: "exact values are nondeterministic") and only booleans (`near 0.05`, `numRowsProcessed as expected`): compare those lines, not the trees; `CompactSgf` subset needed for 7 sections; the randomized-evaluator sections only reproduce if the NN request order and `Rand` use match exactly. Test with the exact evaluators first (sym 1, 2, 5) |
| 5 | `runNNSymmetries` (testsearchmisc.cpp) | 1 (9x13) | g103 v5 (inputs v4) | no search | `NNEvaluator` all 8 symmetries, `setDoRandomize/setDefaultSymmetry` | `runnnsymmetriestest-g103.expected0.txt` | blocked on fillRowV4 + a v5 net; the v7 equivalent is `runNNSymmetryTests` in `runoutputtests` (done) |
| 6 | `runBasicPositions` (testsearch.cpp, 4 games, 22 searches) | 0 (SGF strings) | g103 v5 / run4 v3 | maxVisits 200..400, `rootNoiseEnabled`, `rootPolicyTemperature[Early]`, `rootDesiredPerChildVisitsCoeff`, `drawEquivalentWinsForWhite`, `maxPlayouts` | `AsyncBot`, `Search`, `CompactSgf`, `runBotOnSgf` | `testsearch/runsearchtests-{g103,run4}-sym{0,1}.expected0.txt` | blocked on fillRowV3/V4 and the v3/v5 nets; revive only if those are ported. `CompactSgf` needed anyway |
| 6 | `runOwnershipAndMisc` (testsearchv3.cpp, GAME 5..16) | 7 | grun2 v3 (inputs v3), 19x19 and 11x11 evaluators, extra evaluator for policy temperature | maxVisits 1..500, `rootNoiseEnabled`, `rootPolicyTemperature`, `dynamicScoreUtilityFactor`, `staticScoreUtilityFactor`, `conservativePass`, `useLcbForSelection`, `fpuReductionMax` | ownership map printing, root ending bonus (`printEndingScoreValueBonus`), territory/Japanese/Chinese rules endgames | `testsearchv3/runsearchtestsv3-grun2-sym{0,1,5}.expected0.txt` | same blockers as the row above |

## 5. How far upstream's CUDA results are from the Eigen reruns

- **NN-less**: identical (no NN).
- **Analysis**: section 2. At 100 visits moves, order, visits and pv are identical in all six goldens that have an
  upstream file; float fields differ in the 4th-5th digit (largest: scoreStdev 1e-2, scoreLead 4e-3). At 1000 visits
  (#7, #8) the trees differ: this is what an fp32 CUDA (TF32-capable A5000) vs CPU gap does to a long search.
- **run*tests**: roughly two thirds of the lines are byte-identical, ~92-96% are identical once decimals are ignored.
  The rest are the same story: a different policy rounding changes an `N`/`PSV` column or the order of two nearly tied
  moves and the subtree below it. This is why the Rust search tests must be compared against the Eigen files
  (the same fp32 CPU algorithm up to accumulation order), and why even then a handful of lines can legitimately
  differ: report mismatches by section, and look at the first diverging line before the rest.
- The `: ` log lines and the three NN counters (`NN rows`, `NN batches`, `NN avg batch size`) are part of the printed
  output (`printBasicStuffAfterSearch`). They are deterministic for `numSearchThreads=1`, they depend on the NN cache
  (`nnCacheSizePowerOfTwo=16` in `startNNEval`) and `maxBatchSize=16`.

## 6. Dependencies not yet in the port that block or limit the above

| missing | blocks | note |
|---|---|---|
| `CompactSgf` / `dataio/sgf` (`[[ignored]]`) | V9 section 1, V8 x7, V3, testsearch | `tests/testnninputs.rs` already has a test-local subset; reuse it (tag `test-local copy until M3`) |
| `fillRowV3..V6`, v3-v7 models | testsearch (v1), V3, `runNNSymmetries` | only fillRowV7 and v8+ nets |
| ~~`GameInitializer` (`program/play.cpp`, `[[ignored]]`)~~ | resolved: NN-less "Board size distribution" runs a faithful test-local copy (tests/testsearchnonn.rs, PORTING.md §13) | 100000 `createGame` draws, byte-exact |
| `Setup::loadSingleParams` non-analysis branches | NN-less "SearchParams for GTP/Match/Benchmark/Other/Distributed" | manifest: only analysis ported |
| ~~human-SL net / `SGFMetadata` (M8)~~ | resolved (M8): analysis #11 passes | `bin2onnx` converts `metaEncoderVersion` 1 nets, katago-nn-burn imports `b18c384nbt-humanv0` |
| avoid-SGF-pattern bonus loading (`PatternBonusTable::avoidRepeatedSgfMoves`, SGF files, M8) | none of the search tests (V8's pattern sections use the in-memory `avoidRepeatedPatternUtility`, now ported in `patternbonustable.rs`); `tests/gtp/avoidcorners` in runcmdtests only | `Setup::loadAvoidSgfPatternBonusTables` reads the keys as upstream (protocol case cfg_11 matches) |

## 7. Determinism and regeneration checks

- Analysis goldens: two concurrent complete runs byte-identical.
- Search fixtures: the complete `testsearchcommon/regen.sh` was run twice into separate scratch directories (first run 21 min, second 19 min, 6 jobs in parallel); all 12 expected fixtures are byte-identical between the runs, including the V8 sections with 30/40 search threads and the randomized-symmetry evaluators (with 1 NN server thread the Eigen outputs are reproducible even where upstream calls them nondeterministic). The NN-less slice is also a verbatim substring of upstream's file.
- Sizes: `fixtures/upstream-v1.16.5/analysis/` 176 KB (11 `.stdout` + `regen.sh`); `unit/testsearch*/` 2.3 MB (72 input blocks, 12 expected files totalling 1.9 MB incl. V8 500 KB and V9 559 KB, INDEX files, `testsearchcommon/regen.sh`). Total 2.4 MB.
- `tools/extract_raw_blocks.py`: one backward-compatible fix. The function-name regex did not accept digits, so
  every `runV8Tests*`, `runV9Positions` block was filed under `toplevel`. With the fix the already extracted stems
  (testrules, testscore, testboardarea, testboardbasic, testsymmetries, testnninputs, rand) regenerate
  byte-identically (checked), the output format is unchanged.
- Test run: `cargo test -p katago-core --test fixture_coverage` passes (1 test, 13 allowlist entries incl. the old one).

## Appendix A. Every extracted input block

The `board` column is the `Board::parseBoard` size. The section title is the nearest heading printed above the block
in the source; it is exact for the NN-less file (the 32 blocks map, in order, onto sections 1-33 of
`runnnlesssearchtests.expected0.txt` except "Visualize dirichlet noise", which has no board block) and approximate elsewhere.

#### testsearchnonn.cpp (32 blocks)

| block fixture | src line | board | nearest section title above the block |
|---|---|---|---|
| `runnnlesssearchtests.input0.txt` | 46 | 9x9 | Basic search with debugSkipNeuralNet and chosen move randomization |
| `runnnlesssearchtests.input1.txt` | 130 | 7x7 | Testing preservation of search tree across moves |
| `runnnlesssearchtests.input2.txt` | 197 | 7x7 | Testing pruning of search tree across moves due to root restrictions |
| `runnnlesssearchtests.input3.txt` | 368 | 7x7 | Testing search tree update near terminal positions |
| `runnnlesssearchtests.input4.txt` | 431 | 9x9 | Testing pruning of search tree at root node due to symmetries: empty board |
| `runnnlesssearchtests.input5.txt` | 502 | 9x9 | Testing pruning of search tree at root node due to symmetries: asymmetry board |
| `runnnlesssearchtests.input6.txt` | 543 | 9x9 | Testing pruning of search tree at root node due to symmetries, allowing only diagonal flips: empty board |
| `runnnlesssearchtests.input7.txt` | 585 | 9x9 | Testing pruning of search tree at root node due to symmetries, allowing only diagonal flips: only one diagonal possible |
| `runnnlesssearchtests.input8.txt` | 628 | 9x9 | Testing pruning of search tree at root node due to symmetries, allowing only diagonal flips: empty board, with avoid moves |
| `runnnlesssearchtests.input9.txt` | 706 | 9x9 | Testing pruning of search tree at root node due to symmetries, allowing only diagonal flips: only one diagonal possible, with avoid moves |
| `runnnlesssearchtests.input10.txt` | 784 | 9x9 | Testing pruning of search tree at root node, no symmetries, with avoid moves |
| `runnnlesssearchtests.input11.txt` | 860 | 7x17 | Non-square board search |
| `runnnlesssearchtests.input12.txt` | 971 | 7x7 | Search tolerates moving past game end |
| `runnnlesssearchtests.input13.txt` | 1061 | 7x7 | Integrity of value bias, mem safety and updates |
| `runnnlesssearchtests.input14.txt` | 1125 | 14x14 | Value bias with ko |
| `runnnlesssearchtests.input15.txt` | 1197 | 7x7 | Analysis json |
| `runnnlesssearchtests.input16.txt` | 1252 | 7x7 | Analysis json with moves ownership and stdev |
| `runnnlesssearchtests.input17.txt` | 1308 | 7x7 | Analysis json with moves ownership and stdev and symmetry |
| `runnnlesssearchtests.input18.txt` | 1363 | 9x6 | Analysis json 2 |
| `runnnlesssearchtests.input19.txt` | 1476 | 7x7 | Search results at 0, 1, 2 visits, and at terminal position |
| `runnnlesssearchtests.input20.txt` | 1628 | 7x7 | Testing coherence of search tree recursive walking |
| `runnnlesssearchtests.input21.txt` | 1707 | 9x5 | Testing avoiding of all or almost all moves |
| `runnnlesssearchtests.input22.txt` | 1899 | 7x7 | Graph search, opening |
| `runnnlesssearchtests.input23.txt` | 1966 | 7x7 | Graph search, 7x7 big fight |
| `runnnlesssearchtests.input24.txt` | 2033 | 7x7 | Graph search, 7x7 endgame kos |
| `runnnlesssearchtests.input25.txt` | 2100 | 7x7 | FPU parent weight by visited policy false |
| `runnnlesssearchtests.input26.txt` | 2139 | 7x7 | FPU parent weight by visited policy 1.0 |
| `runnnlesssearchtests.input27.txt` | 2178 | 7x7 | FPU parent weight by visited policy 2.5 |
| `runnnlesssearchtests.input28.txt` | 2217 | 7x7 | FPU parent weight by visited policy 0.5 |
| `runnnlesssearchtests.input29.txt` | 2258 | 5x5 | Policy optimism with tree reuse |
| `runnnlesssearchtests.input30.txt` | 2328 | 5x5 | Policy optimism with tree reuse, 0 root |
| `runnnlesssearchtests.input31.txt` | 2392 | 13x6 | Zero node search |

#### testsearchv3.cpp (7 blocks)

| block fixture | src line | board | nearest section title above the block |
|---|---|---|---|
| `runownershipandmisc.input0.txt` | 245 | 7x11 | GAME 11 |
| `runownershipandmisc.input1.txt` | 260 | 11x7 | GAME 11 |
| `runownershipandmisc.input2.txt` | 311 | 9x9 | GAME 12 |
| `runownershipandmisc.input3.txt` | 347 | 9x9 | GAME 13 |
| `runownershipandmisc.input4.txt` | 404 | 9x9 | GAME 14 |
| `runownershipandmisc.input5.txt` | 448 | 9x7 | GAME 15 |
| `runownershipandmisc.input6.txt` | 488 | 9x7 | GAME 16 |

#### testsearchv8.cpp (21 blocks)

| block fixture | src line | board | nearest section title above the block |
|---|---|---|---|
| `runv8tests.input0.txt` | 200 | 13x13 | Testing PDA + pondering, p200 v400 |
| `runv8tests.input1.txt` | 463 | 19x19 | Testing PDA + pondering, p200 v400 |
| `runmorev8tests.input0.txt` | 664 | 11x11 | TEST VALUE BIAS |
| `runmorev8tests.input1.txt` | 843 | 11x11 | TEST Ending bonus points |
| `runmorev8tests.input2.txt` | 891 | 9x9 | TEST futileVisitsThreshold |
| `runmorev8tests.input3.txt` | 931 | 9x9 | TEST futileVisitsThreshold with playouts |
| `runmorev8tests.input4.txt` | 974 | 19x19 | TEST Hintlocs |
| `runmorev8tests.input5.txt` | 1130 | 19x19 | Mix pruning, subtracting, dirichlet noise, value weight exponent |
| `runmorev8tests.input6.txt` | 1272 | 9x10 | Fill dame before pass |
| `runmorev8tests.input7.txt` | 1352 | 9x9 | Conservative pass |
| `runmorev8tests.input8.txt` | 1413 | 7x7 | Basic graph search 7x7 fight |
| `runmorev8tests.input9.txt` | 1455 | 9x9 | Friendly pass |
| `runmorev8tests.input10.txt` | 1490 | 9x9 | White to play |
| `runmorev8tests.input11.txt` | 1528 | 19x19 | Multithreaded tree updating |
| `runmorev8tests.input12.txt` | 1654 | 19x19 | Pattern bonus |
| `runmorev8tests.input13.txt` | 1744 | 19x19 | Pattern bonus does not care about ko |
| `runmorev8tests.input14.txt` | 1801 | 19x19 | Pattern bonus does not multi-count shapes |
| `runmorev8tests2.input0.txt` | 1876 | 7x9 | TEST ownership endgame |
| `runmorev8testsrandomizednnevals.input0.txt` | 1905 | 15x15 | TEST sampled symmetries |
| `runv8searchmultithreadtest.input0.txt` | 1998 | 19x19 | Multithreaded search test |
| `runv8searchmultithreadtest.input1.txt` | 2090 | 19x19 | Multithreaded graph search test |

#### testsearchv9.cpp (10 blocks)

| block fixture | src line | board | nearest section title above the block |
|---|---|---|---|
| `runv9positions.input0.txt` | 75 | 13x13 | Pruned root values test |
| `runv9positions.input1.txt` | 126 | 9x9 | Conservative passing with pass as hint loc |
| `runv9positions.input2.txt` | 250 | 7x7 | Phase change truncating ladder history in spight rules |
| `runv9positions.input3.txt` | 314 | 9x9 | Phase change truncating ladder history in spight rules |
| `runv9positions.input4.txt` | 368 | 8x8 | Passing details |
| `runv9positions.input5.txt` | 436 | 8x8 | Passing details |
| `runv9positions.input6.txt` | 498 | 8x5 | Passing details |
| `runv9positions.input7.txt` | 580 | 9x9 | Very low visit search discretization |
| `runv9positions.input8.txt` | 614 | 9x9 | More passing hack |
| `runv9positions.input9.txt` | 684 | 9x9 | More passing hack 2 |

#### testsearchmisc.cpp (2 blocks)

| block fixture | src line | board | nearest section title above the block |
|---|---|---|---|
| `runnnontinyboard.input0.txt` | 22 | 5x5 | runNNOnTinyBoard |
| `runnnsymmetries.input0.txt` | 58 | 9x13 | runNNSymmetries |

## 8. Status after the M3 tail (2026-10-24)

- Eval cache, pattern bonus table and anti-mirror (formerly deferred, D7) are ported: `search/evalcache.rs`,
  `search/patternbonustable.rs` (minus its SGF/file loaders), `search/searchmirror.rs` (+ `port::math::tanh`/`atanh`).
- Analysis golden #10 `eval_cache` passes (katago-cli `golden_eval_cache`: the fifo handshake is a stream that yields
  line N+1 after response N; within tolerance, C differs from A as upstream). All analysis goldens are compared,
  `humansl_sidetomove` since M8 (b6c96 + `-human-model b18c384nbt-humanv0`, within tolerance; max deviation
  humanScoreMean 5.5e-5).
- `runNNLessSearchTests`: 44/44 sections byte-exact ("Board size distribution" through a test-local GameInitializer).
- `runNNOnTinyBoard`: katago-nn-burn `tests/testsearchmisc.rs`, exact (39 lines). Only `runNNSymmetries` (g103 v5)
  stays on the fixture_coverage allowlist from testsearchmisc.
- V8: the antiMirror sections (2) are within tolerance; the pattern-bonus sections (3) are compared and allowlisted
  with measured reasons (a 1-visit near-tie shift in one tree-reuse search, a drift already present in the baseline
  search without pattern bonus, and the cumulative NN row count). No V8 section is pending.
- NN-less differential oracle (`oracle/search_nonn.cpp`): 480 cases (+32 evalcache, +40 antimirror, +26 patternbonus),
  bit-exact native and wasm32.
- Human-SL search paths (M8): `oracle/search_humansl.cpp` (a main and a human-SL debugSkipNeuralNet evaluator), 120 cases,
  bit-exact native and wasm32 (`tests/search_humansl.rs`); `oracle/sgfmetadata.cpp` and `DateTime::runTests` likewise.
- Analysis-protocol corpus: 1,398 cases, 1,396 match (cfg_10 / ovr_20, the human-SL profiles, match since M8); known
  divergences cfg_18 (nnMaxBatchSize under Eigen), ovr_24 (an exact E4/F5 prior tie in the reference that burn's NN noise breaks
  the other way; investigated, benign).
