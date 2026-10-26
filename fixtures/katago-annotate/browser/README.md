# Native references of the browser review (M6)

`hand-9x9` (../hand-9x9/game.sgf) reviewed natively with katago-annotate's `review` example, the network
g170-b6c96, upstream's `analysis_example.cfg` and a deterministic search:

    DET="logDir=, numAnalysisThreads=1, numSearchThreadsPerAnalysisThread=1, nnMaxBatchSize=8, nnRandomize=false, nnRandSeed=review, forDeterministicTesting=true"
    OPTS='{"maxVisits":50,"topK":5,"analysisPvLen":5,"includeOwnershipStdev":false,"includePolicy":false,"humanProfile":"rank_5k","model":"g170-b6c96","engineName":"katago-rs-wasm","engineVersion":"1.16.5"}'
    cargo run --release -p katago-annotate --example review -- fixtures/katago-annotate/hand-9x9/game.sgf "$OPTS" -- \
      target/release/katago-rs analysis -config upstream/KataGo/cpp/configs/analysis_example.cfg -model g170-b6c96 -override-config "$DET"

- `hand-9x9.b6c96-deterministic.json`: katago-cli (`katago-rs`, burn flex); identical across runs.
- `hand-9x9.b6c96-deterministic.eigen.json`: the same with the reference KataGo (`build/eigen/katago`, `-model
  upstream/KataGo/cpp/tests/models/g170-b6c96-s175395328-d26788732.bin.gz`). Against the katago-rs one: same moves,
  candidates, visits and PVs; max deviation 1.1e-6 (win rates), 2.3e-5 (scores), 1.5e-6 (priors), grids equal.

- `hand-9x9.b6c96-human-deterministic.json`: the same with the human SL network beside it (`-human-model` the
  b18c384nbt-humanv0 `.bpk`, options `"includePolicy":true,"humanModel":"b18c384nbt-humanv0"`): the payload has
  `rank_5k`'s human policy; identical across runs. `just web-test-human` compares the browser's (`humanModel`).

`just web-test --only review` reviews the same game in the browser with the same config and overrides and
compares (packages/katago-web/test/review.mjs).
