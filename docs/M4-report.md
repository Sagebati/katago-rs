# M4 report: the analysis engine in a Web Worker (katago-wasm, katago-web)

Date: 2026-10-24. Upstream: KataGo v1.16.5 (`ba938676`). burn 0.22.1, wasm-bindgen 0.2.129.
Machine: as M0 (i7-8700K, Radeon RX 6800 XT on RADV), Chromium 156 headless through Playwright 1.62.

## Verdict

The three exit criteria of plan §9 M4 hold in Chromium (`just web-test`: 42/42 checks):

| Criterion | Result |
|---|---|
| Playwright parity, WebGPU and no-GPU | the 9 upstream analysis goldens that run natively (basic, basic_sidetomove, badoverride, clear_cache, symmetry, symmetry_with_pruning, evalcache loop, pvvisits, pvvisits_nograph; g170-b6c96) match our Eigen reruns on WebGPU, on the wasm CPU, and through `backend: "auto"` in a browser without WebGPU. Same structure (moves, order, visits, keys) and floats within the native comparator's tolerances, on WebGPU too |
| Second load does no network | the reload reads the weights from the Cache API: 0 `.bpk` requests seen by the server or issued by the browser (the server sends `no-store`, so it is not the HTTP cache) |
| Terminate within one batch | after a `terminate`, the engine completes only the evaluations its searches had already requested (≤ one batch's worth: 16 rows on WebGPU, 1 on the CPU), then reports. Wall clock from `cancel()` to the final response: 5-73 ms on WebGPU (batches ~36 ms), 15-97 ms on the CPU (batches ~95 ms) |

## What was built

- `crates/katago-wasm` (cdylib, wasm-bindgen): `Engine` runs `katago_core::command::analysis::analysis`
  (upstream's `MainCmds::analysis`) on the worker's event loop. The host pieces mirror `katago-cli`:
  input lines from `Engine.sendLines` (one call = one burst, all immediately available, as a block of
  lines written to stdin), a callback per response line, log and stderr to a callback, the clock from
  `performance.now()`, timed waits from `setTimeout` (cleared when dropped), `spawn_local` as the
  executor, the burn backend on WebGPU (wgpu) or the wasm CPU (flex). Config: upstream's
  `analysis_example.cfg` compiled in, then browser defaults, then the caller's `-override-config`
  strings (`src/config.rs`, with native unit tests); a caller-supplied config text replaces the first two.
  `sha256Hex` (crate `sha2`), `compiledNetworks()`, NN counters (`Engine.stats()`).
- `packages/katago-web` (npm, TypeScript, framework-agnostic): `worker.ts` (module worker: wasm init,
  weights from the Cache API or the network with progress, sha256 check, backend choice, the `subgroups`
  shim, warm-up, readiness through a `query_version`), `client.ts` (`KataGoClient`), `protocol.ts`
  (hand-written types of upstream's JSON), `messages.ts` (worker messages), `index.ts`
  (`fetchModelSpec` for `models.json`).
- `crates/katago-nn-burn`: an opt-in batch-shape padding (`BurnNeuralNet::with_batch_buckets`, port-only,
  off by default, tests in `tests/batch_buckets.rs`); see "Cold start" below.
- `just web-build` (cargo → wasm-bindgen `--target web` → wasm-opt when installed → tsc → `dist/`, plus
  `dist/models/<net>.<sha12>.bpk` and `models.json` for the compiled-in nets) and `just web-test`
  (`packages/katago-web/test/run.mjs`: static server + Playwright + system Chromium; `test/compare.mjs`
  is the Rust golden comparator ported to JS, with its self-check).

## API

```ts
import { KataGoClient, fetchModelSpec } from "katago-web";

const model = await fetchModelSpec("/models/models.json", "g170e-b10c128"); // {network, url, sha256, size}
const kata = new KataGoClient({ model, backend: "auto" });                   // "auto" | "webgpu" | "cpu"
const info = await kata.ready;  // {backend: "webgpu" | "cpu", adapter, version, modelSource: "network" | "cache",
                                //  fallbackReason?, configText, timings}

const run = kata.analyze({ moves: [["B", "Q16"]], rules: "japanese", komi: 6.5, boardXSize: 19, boardYSize: 19,
                           maxVisits: 400, reportDuringSearchEvery: 0.1, includeOwnership: true });
for await (const r of run) {     // TurnResponse: partials (isDuringSearch) then one final per analyzed turn
  draw(r.moveInfos, r.ownership);
}
run.cancel();                    // terminate (also on `break` out of the loop); kata.cancel() = terminate_all
await kata.queryVersion();       // {version: "1.16.5", git_hash}; also queryModels(), clearCache(), stats()
kata.dispose();
```

Options: `config` (full config text), `overrides` (string, strings or object, as `-override-config`),
`analysisThreads`, `warmupVisits`, `cacheModels`, `worker` / `workerUrl` / `wasmUrl` (bundlers), `onLog`,
`onProgress`. Ids: the client sends its own unique ids and gives responses back with the caller's `id`;
`{"error"}` responses reject the run (`KataGoError`), warnings go to `run.warnings`. Low level:
`sendRaw(lines)` / `onLine(cb)` (the protocol verbatim, used by the golden tests), `closeInput()`.

Browser defaults over `analysis_example.cfg` (`config.rs`): no `logDir`; NN cache 2^16 entries;
WebGPU: 2 analysis threads × 16 search threads, `nnMaxBatchSize = 32`; CPU: 1 × 1, batch 4 (the wasm
CPU gains nothing from batching, and one search thread has no virtual-loss dilution).

Backend choice (`auto`): WebGPU when `navigator.gpu.requestAdapter()` gives an adapter, else the CPU
(`fallbackReason` says why). If the WebGPU engine fails before it is ready (a kernel that does not
compile traps the wasm: seen on Chromium's SwiftShader adapter), the client restarts the worker on the CPU.

## Sizes

| File | Raw | gzip -9 | brotli -q 11 |
|---|---|---|---|
| `katago_wasm_bg.wasm` (b6c96 + b10c128 compiled in, cargo profile `web`) | 18.2 MB | 3.6 MB | 2.2 MB |
| same with the `release` profile (opt-level 3 everywhere) | 25.7 MB | 5.1 MB | 3.0 MB |
| `katago_wasm.js` (wasm-bindgen glue) | 65 KB | 12 KB | |
| `client.js` + `worker.js` + `index.js` | 27 KB | 8.6 KB | |
| weights: `g170e-b10c128.bpk` / `g170-b6c96.bpk` (fetched, cached) | 12.0 / 4.1 MB | | |

The `web` profile (workspace Cargo.toml) is opt-level "s" except burn's CPU kernels (burn-flex, gemm,
macerator, pulp, burn-backend at 3): 29% smaller, with the same WebGPU and CPU speed in the tests.
wasm-opt is not installed here, so the wasm is not post-optimised (`just web-build` runs it when present).

## Timings (b10c128, Chromium 156 headless)

Model load (`ready` timings; cold = empty Cache API, warm = page reload):

| | wasm init | weights | sha256 | engine start (model build) | warm-up | total |
|---|---|---|---|---|---|---|
| WebGPU, cold | 106 ms | 82 ms (network, localhost) | 117 ms | 251 ms (39 ms) | 8.4 s | 9.0 s |
| WebGPU, warm | 103 ms | 9 ms (Cache API) | 117 ms | 250 ms (37 ms) | 8.4 s | 9.0 s |
| CPU, warm | 105 ms | 12 ms | 130 ms | 218 ms (17 ms) | none | 0.47 s |

The WebGPU warm-up is the kernel compiles and autotune of the two batch shapes (16 rows: ~7 s, of
which most is the first-ever batch; 32 rows: ~1.2 s); see "Cold start" below. It is the same on a
reload: nothing of it persists in the browser.

100-visit searches (default browser config, fresh positions, NN cache cleared before each):

| Backend | visits/s | NN evals/s | batches | avg rows per batch |
|---|---|---|---|---|
| WebGPU (RX 6800 XT) | 232-291 (≈0.35-0.48 s per 100 visits) | 232-291 | 11-12 | 9-11 (padded to 16) |
| wasm CPU | 10-11 (≈9.5 s per 100 visits) | 10-11 | 100 | 1 |

CPU numbers from a quiet machine; under the load of a concurrent build/test run they fell to 5-6/s.
For comparison M0's bare forward passes: WebGPU 592 evals/s at batch 16, wasm CPU 10.8.

Terminate (5 trials per backend, cancel at a random point after a partial report): the engine completed
16 evaluations (1 batch, 2 smaller ones once) on WebGPU and 1 on the CPU after the terminate arrived;
wall clock from `cancel()` to the final response 5-73 ms (WebGPU, avg batch 35-38 ms) and 15-97 ms
(CPU, avg batch 94-98 ms). Partials with `reportDuringSearchEvery = 0.25` arrived every 251-273 ms
(WebGPU); with 0.5 every 579-600 ms on the CPU (the report waits for the batch in progress).

## Cold start on WebGPU: batch-shape buckets

Each new batch size costs wgpu a kernel compile + autotune: in Chrome 1.2 s for 16 or 32 rows, 3-6 s
for 1-8 rows, on top of ~6 s that the very first batch spends compiling everything. A search makes
batches of every size from 1 to 32, so without help the first queries stalled for seconds at random
(seen: a 100-visit query taking 2.6 s instead of 0.35 s). Now, on WebGPU only:
- `katago-nn-burn` pads a batch to the smallest of a few bucket sizes (rows past the real ones hold stale
  inputs; their outputs are dropped). katago-wasm uses `{nnMaxBatchSize/2, nnMaxBatchSize}` = {16, 32}:
  a 1-row batch costs 27 ms instead of 21 ms, far cheaper than compiling its shape. The goldens on WebGPU
  give the same values with and without padding; natively padded flex outputs equal unpadded ones (test).
- Before its first real batch the engine runs one batch per bucket (`StatsNeuralNet`); the worker
  triggers it with a 1-visit search before `ready`.

The cost moves to start-up (~8.5 s on this machine, every page load: neither the compiled kernels nor
the autotune results persist in the browser). Throughput for 100 visits went from ~350 to ~250-340
visits/s (padding), with no stalls.

## Tests (`just web-test`)

`packages/katago-web/test/run.mjs` serves `dist/`, the test page, upstream's `analysis_example.cfg` and
query files, and checks (all pass):
- goldens: the 9 analysis goldens (g170-b6c96, regen.sh's overrides, each in a fresh engine, the query
  file as one burst then end of input) on WebGPU and on the CPU: structure exact, floats within the native
  comparator's tolerances (WebGPU is held to a wider "webgpu" tier but also passes the flex tier; max
  deviations ≤ 7.9e-5, except 5.6e-4 on the weight keys of pvvisits_nograph, which scale with visits and
  have a relative tolerance);
- no-GPU: a browser with WebGPU disabled, `backend: "auto"` → CPU with a reason, basic goldens pass;
  a WebGPU engine failing before `ready` (simulated by `test/page/failing-worker.js`) → restart on the CPU;
- cache: first load 1 `.bpk` request, reload 0 (server log and Playwright's request log), source "cache";
- partials (`reportDuringSearchEvery`, ids, ownership), `queryVersion` / `queryModels`, an error
  response rejecting the run, terminate (above), throughput (report only).

Native: `cargo test -p katago-nn-burn -p katago-wasm` (debug: 20 passed, 4 ignored as before),
`cargo check --workspace --tests`, clippy on katago-wasm for wasm32 and natively (no new warnings).

## Gaps and follow-ups

- **WebGPU start-up is ~8.5 s** on every load (kernel compiles + autotune, not persisted). Options:
  fewer autotune candidates or a fixed matmul strategy for WebGPU, persisting autotune results (cubecl
  has no browser cache), starting the CPU engine at once and switching to WebGPU once warm.
- **Batches are smaller than the thread count** (~10 of 16 rows): the NN server takes what is queued when
  it is polled, so the evaluations of one search split into pipelined cohorts. It also means a terminate
  can wait for two small batches (seen once in 5). A host-side "wait one more microtask before popping"
  or a server that waits for all virtual threads would fill batches; not changed here (nneval is ported).
- **sha256 with `sha2` costs ~120 ms** for b10c128's 12 MB (measured 330 ms with the release profile under
  load); b18c384nbt's 105 MB would take ~1 s. WebCrypto `subtle.digest` would be ~10× faster.
- `device.lost` → re-init (plan §7) is not handled: a lost device surfaces as a fatal error.
- No wasm-opt (not installed); no Firefox / Safari / iGPU / Apple runs; the 18 MB wasm itself is cached
  only by the HTTP cache (the no-network check covers the weights).
- Cache API only (no OPFS fallback); without a secure context nothing is cached.
- `.bpk` files differ in bytes between builds (burn's metadata), so `models.json`'s hashes change with
  each `just web-build` and caches refill; the weights themselves are the same.
- One network per engine: no human-SL model (M8). b18c384nbt compiles in with the `net-b18c384nbt`
  feature (`just web-build --features net-b18c384nbt`) but was not run in the browser.
- Bundler integration (Vite: worker and wasm URLs) is M5's; the client accepts `worker` / `workerUrl` /
  `wasmUrl` for it.
