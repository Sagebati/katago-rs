# M6 report: full-game review in the browser (engine side)

Date: 2026-10-24. Upstream: KataGo v1.16.5 (`ba938676`). Copied from katago-ws `bdd674bb` (`src/engine/annotate.rs`,
`katago.rs`, `sgf.rs`, `mod.rs`, `config.rs`); payload v3 as baduk_review `8bcb98e1` reads it
(`libs/katago-sdk/src/models.rs`). Machine as M4 (i7-8700K, RX 6800 XT on RADV), Chromium 156 headless through
Playwright 1.62.

M6a is the engine half of plan §9 M6: katago-ws's annotation as a wasm-safe crate, and a review API in the browser
that returns the same payload a server analysis stores. The upload endpoint (`POST /api/games/{id}/analysis/client`,
`source = 'client'`) and the rendering are baduk_review's (M6b); this report ends with what that endpoint should check.

## Verdict

| Goal | Result |
|---|---|
| katago-annotate: katago-ws's annotation, wasm-safe | `crates/katago-annotate`: annotate.rs + the protocol half of katago.rs + sgf.rs + mod.rs's flow + EngineConfig's knobs, copied from katago-ws `bdd674bb`, every divergence marked; no IO, builds and tests on wasm32 |
| Byte-identical payloads | 5 recorded games (522 positions; one pass and two-pass, 9x9 / 13x13 / 19x19, handicap, passes, options off): katago-ws's annotator and katago-annotate give the same payload bytes and the same query lines, natively and on wasm32 (after giving wasm glibc's `expf`) |
| `reviewGame` in the browser | `KataGoClient.reviewGame(sgf \| moves, options, onProgress)` → payload v3 with `engine = {name: "katago-rs-wasm", version: "1.16.5", model}`; progress per turn; cancel (35-75 ms to reject); one pass or two-pass |
| Playwright (WebGPU + CPU) | a deterministic 9x9 review matches the native katago-rs run (same moves, candidates, visits, PVs; floats ≤ 3e-7 / scores ≤ 6e-6, grids equal) and upstream's Eigen KataGo (≤ 1.3e-6 / 2.5e-5); valid against the upload contract; the key paths of katago-ws's payloads |
| Time per turn, defaults (b10c128, 100 visits) | WebGPU 0.4-0.5 s (132-move game in 50-70 s, 42 s two-pass); wasm CPU ~14 s (machine loaded by other builds: see "Timings") |

## katago-annotate

`crates/katago-annotate` (no IO, no threads, no tokio; builds for wasm32-unknown-unknown):

| Module | From katago-ws | What |
|---|---|---|
| `sgf` | `src/engine/sgf.rs` | SGF → `ParsedGame` (moves, setup stones, rules, komi, result) through `sgf-parse` 4.2.8, `policy_index`; new: `GameInput` (a game as KataGo query fields) |
| `katago` | `src/engine/katago.rs`, protocol half | `AnalysisQuery` (+ `AnalysisQuery::new`, the front of `KataGo::analyze`), `OverrideSettings`, `TurnResponse` / `RootInfo` / `MoveInfo`, `Perspective`, `REPORT_PERSPECTIVE`, `ResponseLine::parse` (`handle_line`'s reading of a line) |
| `annotate` | `src/engine/annotate.rs` | the payload types, `assemble`, `turns_to_refine`, classifications, report, per-turn grids |
| `review` | `src/engine/mod.rs` | `analyze` / `analyze_game` (one pass at `maxVisits`, or screen + refine + `merge_turns`), `Pass`, over an `Analyzer` trait the host implements |
| `options` | `src/config.rs` | `AnnotateOptions`: the `EngineConfig` fields the code above reads, same defaults (`maxVisits` 100, `analysisPVLen` 15, `topK` 10, ownership / stdev / policy on, komi 7.5, size 19, `screenVisits` 0, refine deltas 0.03 / 1.0), camelCase JSON |
| `error` | `src/error.rs` | `AnnotateError`: the `AppError` variants used (`Sgf`, `BadRequest`, `Inference`), same messages |

Each file names its source and commit; every change is marked `KATAGO-ANNOTATE:`. The bodies are otherwise
unchanged (the copied unit tests run too: 38, plus 3 new). Divergences:
- `EngineConfig` → `AnnotateOptions` (imported under the old name in `annotate.rs`, so the body is untouched);
- `EngineInfo.name` comes from the options (default `"katago"`), and an optional `version` follows it, left out
  when `None`: a default payload is byte-identical to katago-ws's, the browser's says
  `{"name": "katago-rs-wasm", "version": "1.16.5", "model": "<network>", ...}`. baduk_review's reader already has
  `version` (`Option<String>`);
- `tracing::warn!` → `log::warn!` (the structured fields go into the message);
- `move_accuracy` calls `crate::math::expf`: `f32::exp` natively (glibc, as katago-ws), katago-core's port of glibc's
  `__expf_fma` on wasm32. Without it the report's `accuracy` differed in the last bit in the browser (below);
- `GameInput` and `analyze_game` (a game given as moves) are new; the actor half of `katago.rs` (subprocess, tokio)
  is not copied: hosts implement `Analyzer::analyze(pass, query) -> Vec<TurnResponse>`.

### Later: one copy

katago-annotate is a copy, so for now katago-ws and katago-annotate can drift; the parity tool (below) is what
notices. Once katago-rs is published (crates.io or a git dependency), the plan is:
1. katago-ws depends on `katago-annotate` and deletes `annotate.rs`, the protocol types of `katago.rs`, `sgf.rs` and
   the flow of `mod.rs`; its `KataGo` actor implements `Analyzer` (its `analyze` becomes `run(query)`), `EngineConfig`
   embeds or converts to `AnnotateOptions`, and `AppError` gets `From<AnnotateError>`. The `tracing` spans/metrics
   move into its `Analyzer` impl. Payloads stay byte-identical (the fixtures here are its regression test).
2. baduk_review's `katago-sdk` keeps its own *reader* types (tolerant: `default`, no `deny_unknown_fields`, the v3
   map validation in `models::maps`): they are a different contract (reading every payload version) from the writer's.
   It can use katago-annotate's `PAYLOAD_VERSION` and, for the client-upload check, `GameInput` / `sgf::parse` to
   replay the stored SGF, and the validation listed under "Uploads".
3. katago-annotate stops being a copy: the `KATAGO-ANNOTATE` markers become its own history, and katago-ws changes to
   the annotation land here first.

## Parity with katago-ws

`tools/annotate-parity` (a separate cargo workspace, also kept on disk in the scratch area) compiles katago-ws's
`config.rs`, `engine/{annotate,katago,sgf}.rs` **unchanged** (`#[path]` into the sibling checkout; `error.rs` is a
shim of `AppError` without axum) next to katago-annotate:
- `record`: runs a game through katago-ws's own `KataGo` client (its actor writes the queries, numbers them `q0`,
  `q1`, reads the responses), with `binary` a `tee` wrapper around the reference KataGo (Eigen, `just katago-eigen`),
  so the query lines and KataGo's output are saved verbatim; the flow is mod.rs's `analyze` (copied: mod.rs needs
  muxa) and the payload katago-ws's `assemble` + `serde_json::to_string`, as its worker stores it;
- `compare`: replays a recording through katago-ws's annotator and through katago-annotate (an `Analyzer` that checks
  each query line byte for byte and answers with the recorded lines), and checks both payloads are byte-identical to
  the recorded one, the parsed games equal, and the refined turns those the recording asked for.

`fixtures/katago-annotate` (2.9 MB, gzipped engine output; README there): 5 games, 522 analysed positions with
g170e-b10c128 at katago-ws's defaults or variations (one pass and two-pass; 19x19 Chinese / Japanese / 2-stone
handicap with `AB` setup; 13x13 handicap two-pass without ownership; 9x9 ending in passes, top 5, PV 5, no stdev /
policy, a human profile without a human model). Results:

| Check | Result |
|---|---|
| `regen.sh` (record + compare, 5 cases) and `--check-only` | katago-ws replay == recorded payload, katago-annotate == recorded payload, byte for byte, all 5; same parsed games; same refined turns |
| `cargo test -p katago-annotate` (native) | 44 passed: the 41 copied/unit tests + the replay (query lines and payloads byte-identical), engine name/version only change `engine`, a `GameInput` gives the SGF's payload |
| the same with serde_json `float_roundtrip` (as every katago-rs build has) | passed: no parsed f32 differs |
| `--target wasm32-unknown-unknown` (node, `wasm-bindgen-test-runner`) | 3 passed (the replay, byte-identical) |

The wasm run found one real difference: Rust's `f32::exp` on wasm32 is compiler-builtins' libm (FreeBSD-derived) and
differs from glibc's `expf` in the last ulp for some inputs, so `report.black.accuracy` came out `62.400883` instead
of `62.40088` in the 9x9 game (two of its twelve Black moves). The first recording of the fixtures happened to have
no such input, so the wasm replay passed by luck before the re-recording. `crate::math::expf` (katago-core's
glibc port on wasm32) makes it byte-identical. It is the only transcendental in the annotation (`sqrt`, `round`,
`clamp` are exact). What the payload reproduces is glibc on an x86-64 CPU with FMA (`__expf_fma`), like katago-core:
a katago-ws worker on a CPU without FMA could differ in that last bit itself.

## Browser API

`katago-wasm`: `Engine.reviewGame(reviewId, game, options, onProgress) -> Promise<string>` and
`Engine.cancelReview(reviewId)` (`src/review.rs`). The review runs katago-annotate's `analyze` with the engine as the
`Analyzer`: each query (id `__kgr_<review>_<n>`) goes into the engine's input like any other line; a router takes its
response lines out of the output before the worker's line callback (so they never reach `KataGoClient.analyze` runs)
and reads them as katago-ws does (`ResponseLine`: error → reject, warning → log, one final per turn). The options'
`model` is the network, `humanModel` empty, the engine name and version set by the engine (a caller cannot set them).

`katago-web`:

```ts
const kata = new KataGoClient({ model, backend: "auto" });
const run = kata.reviewGame(sgfText /* or {moves, rules, komi, boardSize, initialStones} */,
                            { maxVisits: 100, screenVisits: 0, topK: 10, signal },
                            (p) => bar(p.analysed, p.pass, p.done, p.total));   // after each analysed turn
const analysis: GameAnalysis = await run.result;   // payload v3; run.json is the exact JSON text to upload
run.cancel();                                      // or signal.abort(): rejects with KataGoError {cancelled: true}
```

- Options (`ReviewOptions`, katago-ws's names and defaults): `maxVisits`, `screenVisits` / `refineWinrateDelta` /
  `refineScoreDelta` (two-pass), `topK`, `analysisPvLen`, `includeOwnership`, `includeOwnershipStdev`,
  `includePolicy`, `defaultBoardSize`, `defaultKomi`, `humanProfile` (dropped with a warning: no human SL net yet, M8),
  `signal`. Unknown options are an error.
- Progress: `{pass: "full" | "screen" | "refine", passes, turnNumber, done, total, analysed}` per analysed turn
  (turns finish out of order with several analysis threads).
- Cancel: the query in flight is terminated (the engine stops within one NN batch, docs/M4-report.md), its partial
  results are discarded, no further pass starts. Before `ready`, a review is held by the client (and goes to the CPU
  worker if WebGPU fails before `ready`).
- A review shares the engine with `analyze` queries (KataGo's queue interleaves them; the review's refine pass has
  priority 1, like katago-ws). Errors: an SGF katago-ws would reject (`invalid SGF: ...`), bad options, an engine
  error (`inference error: katago: ...`).
- TS types of the payload: `src/review.ts` (`GameAnalysis`, `MoveAnnotation`, `Candidate`, `EngineInfo`, ...).

### Tests (`just web-test --only review`)

`test/run.mjs` group `review` (in the default set), in a persistent Chromium profile (`--profile`, default
`target/web-test-profile`) so that later runs take the WebGPU kernels from the kernel cache:

| Check | WebGPU | CPU |
|---|---|---|
| deterministic 9x9 (`hand-9x9`, g170-b6c96, `analysis_example.cfg` + 1 thread, `forDeterministicTesting`) vs the native katago-rs run (`fixtures/katago-annotate/browser`) | same structure; max dev 1.8e-7 win rate, 6e-6 score, 7.2e-7 prior, 1e-5 accuracy, grids 0 | same; 3e-7, 6e-6, 6e-7, accuracy equal |
| the same vs upstream KataGo (Eigen) | 1.3e-6, 2.5e-5, 1.7e-6, grids 0 | 1.1e-6, 2.3e-5, 2.1e-6, grids 0 |
| `validatePayload` (the upload contract below), engine `{katago-rs-wasm, 1.16.5, g170-b6c96}` | pass | pass |
| key paths vs katago-ws's payload of the same game and options | identical except `.engine.version` | same |
| one progress event per turn, the engine answers `analyze` afterwards | pass | pass |
| two-pass (screen 10, refine at 60), the 13x13 handicap game given as moves, no ownership | 21 screened + 14-16 refined turns (runs differ); valid; katago-ws's key paths | |
| cancel after 10 turns of a 133-turn review | rejects `review cancelled` 35-75 ms after `cancel()`; 10 progress events; the engine still answers | |

(The tolerances are the M4 tiers: flex 1e-5 win rate / 5e-4 score, WebGPU 1e-4 / 2e-3; grids ±1, policy bytes ±2.)
The native and the Eigen references agree to 1.1e-6 themselves, so the browser review gives the server's payload
up to float noise when the search is the same.

### Timings

Default browser config (WebGPU: 2 analysis threads x 16 search threads, batch 32; CPU: 1 x 1), katago-ws's default
options (100 visits, one pass, top 10, ownership + stdev + policy), g170e-b10c128, NN cache cleared before each
review. The machine ran two other agents' builds (load average 11-20 on 12 threads), so these are slower than
M4's quiet numbers (WebGPU 240-290 evals/s, CPU ~10.5) and vary between runs.

| Review | Backend | Wall clock | Per analysed turn | NN |
|---|---|---|---|---|
| first 10 moves of ogs-60000000 (11 turns) | WebGPU | 5.6-6.7 s | 0.50-0.61 s | 208 evals/s, 11.6 rows / batch, 51 ms / batch |
| same | wasm CPU | 154-170 s | 14.0-15.4 s (7.8-22 s) | 6 evals/s, 1 row, 169 ms / batch (M4 quiet: 95 ms) |
| the whole game (132 moves, 133 turns) | WebGPU | 70 s (109-129 s at load 15-20) | 0.52 s | 189 evals/s, 10.4 rows, 54 ms |
| same, `numAnalysisThreads=8, numSearchThreadsPerAnalysisThread=4` | WebGPU | 55 s | 0.42 s | 206 evals/s, 17.1 rows |
| same, 32 x 1 (katago-ws's throughput shape) | WebGPU | 49 s | 0.37 s | 217 evals/s, 27.2 rows |
| two-pass (screen 16; 96-97 turns refined), 8 x 4 / 32 x 1 | WebGPU | 42 s | 0.18 s (236 turns) | 214-229 evals/s |

- WebGPU is NN-bound in the default config (the NN busy 99% of the time, batches ~10 of 32 rows: M4's "batches
  are smaller than the thread count"). More positions in parallel fill the batches and help ~25-30%; the engine
  config is per client (`overrides` / `analysisThreads`), so a page that reviews whole games could start a client
  with `numAnalysisThreads=8` or more. Two-pass (as katago-ws can) cuts a full review by another ~15-20%.
- The wasm CPU is ~25x slower: ~14 s per turn at 100 visits, ~30 min for a 130-move game. Reviews on the CPU need
  far fewer visits (e.g. `maxVisits: 10-20`, or `screenVisits` only) and are mostly useful for short games.
- Two runs of the one-pass full game had single turns taking 6-8 s (max gap 7.6 s) under the heaviest load; the
  same review later had none above 2.1 s.

## Uploads: what baduk_review should check

A client payload is produced by code the server does not run, on a machine it does not control, so it is a claim,
not a measurement (plan §10 risk 8). The payload is self-consistent by construction when honest; the checks below
catch broken or tampered uploads cheaply, and the policy decides what a valid but untrusted payload may feed.

Reject unless:
1. **Shape**: parses as katago-sdk's `GameAnalysis`; `payload_version == 3`; `engine` present with
   `name == "katago-rs-wasm"` (an upload claiming `"katago"` is not a client payload), `report_perspective ==
   "SIDETOMOVE"`, `max_visits` within the allowed range (e.g. 1..=10000), `top_k <= 50`; no `human_policy` /
   `human_profile` / `engine.human_model` (no human SL model in the browser until M8); no `screen_visits`
   unless < `max_visits`.
2. **The game**: replay the stored SGF with katago-sdk's / katago-annotate's `sgf::parse` (the server's own copy of
   the game, never the client's): `board_size`, `komi`, `rules`, `result`, `moves.len()` and every
   `(move_number = i + 1, color, mv)` equal the parsed game's.
3. **Ranges**: win rates, priors, losses in `[0, 1]` and finite; scores finite (|x| ≤ board points + komi);
   `winrate_loss == max(best_winrate − winrate, 0)` (to 1e-6) and `score_loss ≥ 0`; classifications follow the
   thresholds (win rate 0.03 / 0.07 / 0.15, points 1 / 3 / 6); `top_moves.len() <= top_k`, candidate visits
   `<= max_visits` (a refined turn of a two-pass review: `max_visits`, a screened one `screen_visits`), candidate
   moves and PVs legal GTP points or `pass` on this board.
4. **Report**: recompute it from `moves` (counts per class, means, accuracy) and require equality (the server can
   simply overwrite it with its own).
5. **Grids**: each of `ownership` / `ownership_stdev` / `policy` absent or exactly `moves.len() + 1` rows of
   `board_size²` integers in `[-100, 100]` / `[0, 100]` / `[0, 255]`, present exactly when the matching `engine` flag
   is true (katago-sdk's `maps` readers already enforce the cell ranges); policy 0 on occupied points.
6. **Size**: a sane byte cap (a 300-move 19x19 payload with every grid is ~1.2 MB of JSON).

`packages/katago-web/test/review.mjs` (`validatePayload`) implements 1-5 minus the replay and is what the browser
tests run on every payload.

Then: store with `source = 'client'` and the engine object as sent; keep client payloads out of shared baselines
and anything aggregated across users (plan §10 risk 8), or spot-check: re-analyse a sample of turns on katago-ws at
the payload's `max_visits` and compare the top move and win rate within a loose tolerance (WebGPU and server engines
are not bit-identical; different searches differ more than that, so this catches fabrication, not drift).

## Tests run

- `cargo test -p katago-annotate`: 44 passed (38 copied from katago-ws, 3 new unit, 3 replay), also with `-F float-roundtrip`;
  `--target wasm32-unknown-unknown` (`just annotate-parity`, and in `just test-wasm`): the 3 replay tests passed.
- `tools/annotate-parity/regen.sh` (record all 5 cases, compare) and `--check-only`: 5/5 byte-identical (run from a
  copy with katago-ws as its sibling; the worktree has none).
- `cargo test -p katago-wasm` (native: 7 passed), `cargo check --workspace --tests`, clippy on katago-annotate
  (lib, tests, examples) and on katago-wasm for wasm32: no warnings in the changed crates.
- `just web-test-client`: 7 passed (2 new: reviews held until `ready` and sent to the worker that gets ready, progress,
  payload; cancel before / after posting, errors, AbortSignal, engine failure).
- `just web-test`: 60/60 checks (the 45 of M4 and the 15 `review` checks above).
- `just web-test --only review-timing [--timing first10,full,twopass] [--review-overrides ...]`: the timings above
  (report only).

## Gaps and follow-ups

- **Upload endpoint and rendering (M6b)**: baduk_review's `POST /api/games/{id}/analysis/client` with the checks in
  "Uploads", and `source = 'client'` in the UI. `run.json` is the text to send.
- **No human SL policy** in the browser (M8): `humanProfile` is dropped with a warning, as katago-ws does without a
  human model; the payload then has no human fields.
- **Throughput**: the browser engine's default config is tuned for live analysis (2 x 16 threads); full reviews run
  25-30% faster with more positions in parallel (Timings). A per-client choice today; a review-specific engine
  config, or letting `reviewGame` raise the analysis threads, is open.
- **CPU reviews** take ~14 s per turn at 100 visits on b10c128: usable with low visits or on small boards only.
- **Exact parity** holds for the annotation, not the search: the same game reviewed in the browser and on a
  katago-ws worker gives different payloads unless both run the same deterministic search (multi-threaded searches
  are not reproducible, on either side).
- katago-ws's two `tracing::warn!` became `log::warn!`; a later katago-ws depending on katago-annotate gets them via
  its `log` bridge (or the warnings move into its `Analyzer` impl). `tools/annotate-parity` needs `../katago-ws`
  (relative `#[path]`s): run it from a checkout next to katago-ws.
