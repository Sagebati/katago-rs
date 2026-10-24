# M4 report: the analysis engine in a Web Worker (katago-wasm, katago-web)

Date: 2026-10-24. Upstream: KataGo v1.16.5 (`ba938676`). burn 0.22.1, wasm-bindgen 0.2.129.
Machine: as M0 (i7-8700K, Radeon RX 6800 XT on RADV), Chromium 156 headless through Playwright 1.62.

## Verdict

The three exit criteria of plan §9 M4 hold in Chromium (`just web-test`: 45/45 checks):

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
- WebGPU start-up (follow-up, see "Start-up on WebGPU" below): no burn autotune in the browser build
  (katago-nn-burn feature `wgpu-fixed-kernels`), the wasm module linked as a reactor (`__wasm_call_ctors`
  called once in `start`), compiled WGSL kernels persisted per build and adapter in the Cache API
  (`kernelCacheImport` / `kernelCacheExport`, backed by `third_party/cubecl-wgpu`, a patched cubecl-wgpu
  0.11.2), WebCrypto for the weights' sha256, and the weights / adapter probe / kernel cache read in parallel.
- `just web-build` (cargo → wasm-bindgen `--target web` → wasm-opt when installed → build id → tsc → `dist/`, plus
  `dist/models/<net>.<sha12>.bpk` and `models.json` for the compiled-in nets) and `just web-test`
  (`packages/katago-web/test/run.mjs`: static server + Playwright + system Chromium; `test/compare.mjs`
  is the Rust golden comparator ported to JS, with its self-check).

## API

```ts
import { KataGoClient, fetchModelSpec } from "katago-web";

const model = await fetchModelSpec("/models/models.json", "g170e-b10c128"); // {network, url, sha256, size}
const kata = new KataGoClient({ model, backend: "auto" });                   // "auto" | "webgpu" | "cpu"
const info = await kata.ready;  // {backend: "webgpu" | "cpu", adapter, version, modelSource: "network" | "cache",
                                //  fallbackReason?, configText, timings, kernelCache?: {loaded, hits, compiled}}

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
`analysisThreads`, `warmupVisits`, `cacheModels`, `cacheKernels`, `worker` / `workerUrl` / `wasmUrl` (bundlers), `onLog`,
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
| `katago_wasm_bg.wasm` (b6c96 + b10c128 compiled in, cargo profile `web`) | 17.1 MB | 3.4 MB | 2.1 MB |
| same with the `release` profile (opt-level 3 everywhere) | 25.7 MB | 5.1 MB | 3.0 MB |
| `katago_wasm.js` (wasm-bindgen glue) | 65 KB | 12 KB | |
| `client.js` + `worker.js` + `index.js` | 27 KB | 8.6 KB | |
| weights: `g170e-b10c128.bpk` / `g170-b6c96.bpk` (fetched, cached) | 12.0 / 4.1 MB | | |

The `web` profile (workspace Cargo.toml) is opt-level "s" except burn's CPU kernels (burn-flex, gemm,
macerator, pulp, burn-backend at 3): 29% smaller, with the same WebGPU and CPU speed in the tests.
wasm-opt is not installed here, so the wasm is not post-optimised (`just web-build` runs it when present).
(18.2 MB before the start-up work: dropping burn's autotune removed its candidate kernels.)

## Timings (b10c128, Chromium 156 headless)

Model load (`ready` timings; cold = empty Cache API, warm = page reload in the same browser profile):

| | wasm init | weights | sha256 | engine start (model build) | warm-up | total |
|---|---|---|---|---|---|---|
| WebGPU, cold | 96-110 ms | 22-29 ms (network, localhost) | 29-50 ms | 235-260 ms (38 ms) | 1.3-1.6 s | 1.8-2.0 s |
| WebGPU, warm | 92-112 ms | 12-21 ms (Cache API) | 28-46 ms | 235-290 ms (39 ms) | 0.43-0.48 s | **0.84-0.94 s** |
| WebGPU, warm, incognito-like context | 90-109 ms | 12 ms | 29-41 ms | 248-256 ms | 0.80-0.92 s | 1.2-1.3 s |
| CPU, cold (g170-b6c96, 4 MB, browser without WebGPU) | 99 ms | 10 ms | 11 ms | 207 ms (11 ms) | none | 0.35 s |
| *before (M4): CPU, warm (b10c128)* | *105 ms* | *12 ms* | *130 ms* | *218 ms (17 ms)* | *none* | *0.47 s* |
| *before (M4): WebGPU, cold or warm* | *103-106 ms* | *9-82 ms* | *117 ms* | *250 ms (37 ms)* | *8.4 s* | *9.0 s* |

The warm-up is the batch-shape warm-up below plus a 1-visit search. Cold, it is cubecl compiling its 45
kernels from IR to WGSL in wasm (~1 s of CPU) and Chrome compiling the pipelines; warm, every kernel comes
from the kernel cache and the pipelines from Chrome's GPU cache (in a persistent profile; an incognito-like
Playwright context keeps no GPU disk cache, so the first batch waits ~0.5 s for the driver: the 1.2-1.3 s
row, which is what `just web-test`'s cache check measures). Ranges over 3-6 loads on a machine with other
builds running (load average 4-12). Details, history and what did not work: "Start-up on WebGPU" below.

100-visit searches (default browser config, fresh positions, NN cache cleared before each):

| Backend | visits/s | NN evals/s | batches | avg rows per batch |
|---|---|---|---|---|
| WebGPU (RX 6800 XT) | 240-292 (≈0.34-0.46 s per 100 visits; the page's first query 185-243) | 240-292 | 10-11 | 10-11 (padded to 16) |
| wasm CPU | 10-11 (≈9.5 s per 100 visits) | 10-11 | 100 | 1 |

CPU numbers from a quiet machine; under the load of a concurrent build/test run they fell to 5-6/s.
For comparison M0's bare forward passes: WebGPU 592 evals/s at batch 16, wasm CPU 10.8. The WebGPU row is
after the start-up work (before: 232-291 visits/s); longer searches (400 visits, default config) run at
370-455 visits/s, NN 36-42 ms per batch. The first query after `ready` takes 0.45-0.65 s for 100 visits
(later ones 0.32-0.40 s): the wasm search code still runs in V8's baseline tier.

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

The cost moves to start-up: ~8.5 s on every page load at first (neither compiled kernels nor autotune
results persisted in the browser), 0.43-0.48 s on a warm load now (next section). Throughput for 100 visits
went from ~350 to ~250-340 visits/s (padding), with no stalls.

## Start-up on WebGPU

The 8.4 s warm-up, measured with a probe worker wrapping `worker.js` (counts and timestamps of WebGPU
calls) and Chrome tracing (V8 CPU profile of the worker, a build with the wasm name section):

- **burn/cubecl autotune: ~7 s.** 82 kernels compiled (2.6 MB of WGSL, most of them autotune candidates),
  5787 queue submits and 1610 buffer mappings before `ready`: every matmul / reduce / fused-matmul key is
  benchmarked with 10 samples per candidate. On wasm cubecl keeps tuning results in memory only (its
  persistent cache needs the `persistence` feature: an embedded database on OPFS), so this repeats on every
  load, and while a key is being tuned the fallback kernel runs (outputs depended on timing).
- **cubecl compiling kernels in wasm: ~1-1.2 s** (kernel expansion, the pliron optimizer passes, WGSL
  printing; one 67-71 KB matmul kernel takes 100-170 ms). The browser caches only its own half (Dawn's
  pipeline cache: `DawnWebGPUCache` in the profile, 7-16 MB): WGSL is regenerated on every load.
- **Every JS → wasm call ran all static constructors.** cubecl-core pulls in pliron, which registers
  ~1000 items with `inventory` (`.init_array` constructors). wasm-ld links a module that has
  constructors but never references `__wasm_call_ctors` as a *command* and wraps every export in a call to
  them: `__wbindgen_malloc`, `__externref_table_alloc`, every closure trampoline (each WebGPU promise,
  bind group, buffer mapping). Under the profiler this was most of the worker's time; unprofiled it is
  ~5% of the warm-up and 8-10% of each NN batch.
- The weights' sha256 in wasm: ~120 ms. The GPU side (Dawn + RADV compiling pipelines) overlaps the
  CPU work except for ~150 ms cold.

Changes (all in the browser build; native builds keep autotune):

| Step | WebGPU `ready`, cold / warm | warm-up warm |
|---|---|---|
| M4 | 9.2 s / 9.1 s | 8.5 s |
| no autotune (`wgpu-fixed-kernels`: burn's default kernel per op, 45 kernels, 49 submits before `ready`) | 2.05 s / 1.83 s | 1.3 s |
| + `__wasm_call_ctors` called once in `start` (reactor) | ~2.0 s / 1.78-1.89 s | 1.1-1.3 s |
| + kernel cache, WebCrypto sha256, weights / adapter / kernel cache in parallel | 1.8-2.0 s / **0.84-0.94 s** | 0.43-0.48 s |

- **Kernel cache** (`third_party/cubecl-wgpu`, changes marked `PATCH(katago-rs)`, `src/compute/wgsl_cache.rs`):
  when enabled (katago-wasm enables it for WebGPU), `WgpuServer::pipeline` records each WGSL kernel it
  compiles (entry point, source, buffer visibilities and IO attributes, shared memory size) and, before
  compiling, looks the kernel up by the key cubecl uses for its own persistent SPIR-V store (device
  properties hash, `KernelId::stable_hash`), confirmed by an exact match of `KernelId::stable_format`.
  The worker imports the blob (`kernelCacheImport`, ~0.6 MB, FNV-checked) from the Cache API before creating
  the engine and saves it (`kernelCacheExport`, 6-8 ms) before posting `ready` when kernels were compiled,
  then every 30 s if more were. The Cache API key holds the build id (`dist/wasm/build-id.js`, the first 16
  hex digits of the wasm's sha256, written by `just web-build`) and the adapter; blobs of other builds are
  deleted. A hit skips cubecl's expansion, optimization and printing; Chrome still creates the pipeline
  (from its own cache). Same WGSL → same outputs: `just web-test` checks a deterministic search
  (1 thread, `forDeterministicTesting`) is byte-identical with compiled and with cached kernels.
- **Fidelity.** Outputs are now deterministic for a given build and GPU (fixed kernels); before, the
  autotune winner (and the fallback kernel used while tuning) depended on timing. The goldens on WebGPU
  give the same maximum deviations as before (≤ 7.9e-5; 5.6e-4 on pvvisits_nograph's weights).

Throughput cost of dropping autotune: within run-to-run noise in the default config (400-visit searches:
fixed 370-455 visits/s, autotune 387-440, alternated runs on a loaded machine); with 1 × 16 search threads
(full 16-row batches) the NN batch took 36-38 ms fixed vs 31-32 ms autotuned (~15%), with 1 × 32 threads
45-50 ms vs 45 ms. The difference is the fused matmul: without autotune burn-cubecl-fusion tries its default
selector (accelerated `Cmma`, not available on WebGPU) and falls back to an unfused matmul plus elementwise
kernels; autotune picks the double-buffered unit matmul with fused read/write for the 3x3 convs.

Tried and not kept:
- opt-level 3 for the kernel-compile path (pliron, cubecl-opt/-ir/-core/-wgpu, burn-fusion, ...): warm-up
  −0.1-0.15 s before the kernel cache, wasm +23% (20.9 MB); pointless once kernels are cached. Same for
  `katago-core` at opt-level 3: no measurable change in engine start or visits/s.
- Pinning `DoubleUnit` as the fused-matmul fallback (a burn-cubecl-fusion patch): no difference outside the
  noise in the default config, and a choice tuned on one GPU; not worth a second vendored crate.
- Autotune with the kernel cache: candidates come from the cache, but the benchmarks still run on every
  load (cold 7.9-10.9 s).
- cubecl's own `persistence` feature: on wasm it stores into an embedded database (turso) on OPFS behind a
  Web Lock (one tab at a time), and still keeps no WGSL; not tried.

Remaining options: persist autotune results (seed cubecl-runtime's tuner, a second patch; the first load
would still tune or need shipped per-GPU defaults); `createComputePipelineAsync` for pipelines from the
kernel cache (parallel driver compiles; ~0.5 s in contexts without Chrome's GPU disk cache, ~0.15 s cold);
skip the 32-row warm-up batch (66 ms warm); ~140 ms of engine start between the WebGPU device and the
analysis engine's first log not profiled further; im2col without 9 slices + a concat per 3x3 conv (each
NN batch dispatches ~500 kernels, ~217 of them `slice`), which bounds both batch time and CPU per batch.

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
  the reload compiles no WebGPU kernel (all 45 from the kernel cache; ready times reported);
- kernel cache: an engine that compiles its kernels saves them and the next one loads them, and a
  deterministic search gives byte-identical final responses with both;
- partials (`reportDuringSearchEvery`, ids, ownership), `queryVersion` / `queryModels`, an error
  response rejecting the run, terminate (above), throughput (report only).

Native: `cargo test -p katago-nn-burn -p katago-wasm` (debug: 21 passed, 4 ignored; the new one is
`katago-wasm/tests/kernel_cache.rs`, the kernel cache's blob format and damaged blobs),
`cargo check --workspace --tests`, clippy on katago-wasm for wasm32 and natively (no new warnings).

## Gaps and follow-ups

- **WebGPU start-up**: ~0.9 s warm, ~1.9 s cold (was 9 s); remaining options in "Start-up on WebGPU".
  `third_party/cubecl-wgpu` must be re-applied on a cubecl bump (or upstreamed: a WGSL store next to
  cubecl's SPIR-V one); the M0 spike crate (`katago-web-bench`) still has the constructor wrappers.
- **Batches are smaller than the thread count** (~10 of 16 rows): the NN server takes what is queued when
  it is polled, so the evaluations of one search split into pipelined cohorts. It also means a terminate
  can wait for two small batches (seen once in 5). A host-side "wait one more microtask before popping"
  or a server that waits for all virtual threads would fill batches; not changed here (nneval is ported).
- The weights' sha256 now uses WebCrypto (~30 ms for b10c128; the wasm `sha2` path, ~120 ms, remains for
  insecure contexts, where nothing is cached anyway).
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
