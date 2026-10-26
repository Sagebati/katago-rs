# M7 report: kata1-b18c384nbt in the browser

Date: 2026-10-24. Upstream: KataGo v1.16.5 (`ba938676`). burn 0.22.1, wasm-bindgen 0.2.129.
Machine: as M0/M4 (i7-8700K, Radeon RX 6800 XT on RADV), Chromium 156 headless through Playwright 1.62.
The machine was shared with other agents' builds during these runs (load average 5-15), so throughput
numbers on the CPU side are noisy; the GPU was the bottleneck throughout (`gpu_busy_percent` 99%) and
those numbers vary by about 10-30% between runs. Ranges are given where they matter.

## Verdict

b18c384nbt runs in the browser on WebGPU, passes its parity checks, and is usable for review-style
analysis (100 visits in ~2 s). It is not a drop-in replacement for b10c128 as the live-analysis default:
it is ~4-5x slower per visit, costs a 105 MB download, ~0.4 GiB of wasm memory and ~0.5 GiB of VRAM, and
is unusable on the wasm CPU.

| Question | Result |
|---|---|
| Ship it in the same wasm? | Yes. It adds 0.2 MB to the 17.1 MB wasm (+1.2% raw, +34 KB gzip, +17 KB brotli). The net is chosen at run time by `network`; its weights are fetched only when asked for. `net-b18c384nbt` is now a default feature of `katago-wasm`; `just web-build` writes `kata1-b18c384nbt.<sha12>.bpk` and its `models.json` entry |
| Parity | New goldens from the Eigen reference (`fixtures/upstream-v1.18.2/analysis/b18_*.stdout`, upstream has none for this net) match on WebGPU: structure exact, floats within the native comparator's tolerances, max deviation 3.6e-5 (also on a device with WebGPU's default buffer limits). Native `nn_parity` b18 on flex and wgpu vs Eigen: policy 2-3e-6, ownership 1e-5, score 1e-4 |
| Speed (WebGPU) | After a fix found here (below): 93 evals/s at batch 16 and 100 at 32 natively without autotune; in the browser 48-71 visits/s for 100-visit searches (1.6-2.3 s) and 57-100 visits/s for 400 visits (4-7 s). Before the fix: 11-14 and 13-21 visits/s |
| Start-up | Persistent profile: 1.76-1.78 s warm, 3.0 s cold (b10c128: 0.94-0.96 s / 1.83 s). Fresh incognito-like context: 2.2-2.3 s warm, 2.9-3.4 s cold. Weights from the Cache API: 145 ms read + 235 ms sha256 (WebCrypto) |
| Limits | The adapter's limits (4 GiB-4 for storage binding and buffer) are passed on to the device. Biggest buffer: 2.5 MB per batch row (80 MB at 32). Works with WebGPU's default limits (128 MiB / 256 MiB) up to 53 rows; 64 rows fail there |
| CPU | ~0.8-0.9 evals/s in the wasm (100 visits ≈ 2 min). The client now refuses b18 on the CPU unless `allowSlowCpu`, and can swap in a smaller net (`cpuModel`) |

## What changed

- `crates/katago-nn-burn/src/conv.rs`: **1x1 convolutions are lowered to a matmul on GPUs too** (they used
  burn's direct conv kernel). See "The 1x1 conv problem"; this is what made b18 usable without autotune.
- `crates/katago-wasm/Cargo.toml`: `net-b18c384nbt` joins the default features.
- `justfile`: `web-build` takes the `.bpk` files from the out dir of the katago-nn-burn build it just ran (cargo's
  JSON messages) instead of `ls -t` over `target/`, which also found nets of earlier builds with other features;
  new `web-test-b18`.
- `packages/katago-web`:
  - `cpuModel` (a `ModelSpec` of a smaller net to run when the engine ends up on the CPU: `auto` without WebGPU,
    a WebGPU engine that failed before it was ready, or `backend: "cpu"`), `allowSlowCpu`;
    `ReadyInfo.requestedNetwork` (set when `cpuModel` replaced the asked-for net), `adapterLimits`, `wasmMemoryBytes`.
  - A GPU-only network (`kata1-b18c384nbt`) on the CPU without `cpuModel` or `allowSlowCpu` is refused by the worker
    with an error that names both options, before any weights are fetched. The refusal is final: the client does
    not restart on the CPU after it (`fatal` messages carry `retry: false`). With `backend: "auto"` and a net that
    has a `cpuModel` or is GPU-only, the worker probes the adapter before it starts the download, so a 105 MB
    fetch never happens just to be refused.
  - `fetchModelSpec(manifest, network)` works for the new entry unchanged.
- Tests: `just web-test-b18` (21 checks, ~2 min), 3 more unit tests in `just web-test-client`
  (`cpuModel`/`allowSlowCpu` reach the worker, a refusal does not restart, a CPU-restart failure names the WebGPU one).
  `fixtures/upstream-v1.18.2/analysis/regen_b18.sh` regenerates the goldens.
- `examples/bench` and `examples/parity` accept the browser's feature set (`-F wgpu-fixed-kernels`, no autotune);
  `examples/convbench` takes a channel count (`convbench wgpu 16 384`) and times the im2col stages and the
  global-pooling reduction.

## The 1x1 conv problem (and why b18 first looked unusable)

The first browser runs of b18 gave 11-14 visits/s (a 16-row batch took ~730 ms, GPU 99% busy), 5x below
M0's native numbers (122 evals/s at batch 16, with autotune). The same slowness reproduced natively with the
browser's feature set (`wgpu-fixed-kernels`: burn's default kernel per op, no autotune): 22.6 evals/s at batch 16.
cubecl's profiler (`CUBECL_DEBUG_LOG=<file> CUBECL_DEBUG_OPTION=profile-medium`, summed per kernel over 15
forward passes, b18, batch 16):

| Kernel | Share of GPU time |
|---|---|
| `DirectConv2dKernel` (546 calls) | **78.5%** |
| matmul (`MatmulEntry`, 1246) | 10.3% |
| `CopyKernelPack` | 3.8% |
| mask where / fill | 3.5% |
| everything else | < 4% |

b18c384nbt is a nested-bottleneck net: 41 of its 119 convolutions are 1x1 (36 of them 384↔192, in the blocks), and `conv2d_same` left 1x1
kernels to burn's own conv, which is its direct kernel, ~10x slower than the equivalent matmul at these shapes
(convbench, 384 channels, batch 16: 1x1 conv 21.9 ms vs matmul 2.5 ms). With autotune, burn picks a better conv
candidate (so M0's measurement did not see it); without autotune, which the browser build needs (M4: tuning cost ~8 s
per page load), the direct kernel runs. b10c128 has only a few 1x1 convs (heads), so M4 never noticed.

Fix: `conv2d_same` lowers 1x1 convs to the same permute + matmul as the larger kernels (no padding or window copies).
CPU backends are unchanged. Native, no autotune, no fusion, batch 1 / 16 / 32, evals/s:

| | before | after | autotuned (M0) |
|---|---|---|---|
| b18c384nbt | 12 / 22 / 24 | 12 / **93 / 101** | 30 / 122 / 131 |
| b10c128 | | 66 / 572 / 824 | |

b10c128 is unchanged (it was not slowed by 1x1 convs); the browser's b10 throughput is unchanged within noise
(alternating old and new builds: 100-visit searches 175-285 vs 200-297 visits/s, 400-visit 236-414 vs 234-416).
Parity after the change: `just nn-parity -F wgpu` 7/7 (b18 wgpu vs Eigen: policy 2.8e-6, ownership 1.1e-5, score mean 8e-5;
b10 wgpu 6e-6 / 9e-6 / 1.7e-4), `parity wgpu` against onnxruntime 2.1e-5 (policy) for both nets.

Left on the table: the matmul kernel burn picks without autotune (`SimpleUnit`, max tile). Forcing cubek's
`DoubleUnit` with the min tile size (an experiment with a patched `cubek-matmul`, not kept: it would be a third
vendored crate and the choice was only measured on this GPU) gives b18 124 / 129 evals/s at batch 16 / 32 (+34% / +28%) and
b10c128 818 / 943 (+43% / +15%), the same as autotune. Other unit strategies (SimpleUnit min tile, DoubleUnit max
tile, `Gemm`) were between -40% and +10%. A candidate follow-up, together with M4's "fixed kernel" caveats.

## Sizes

`katago_wasm_bg.wasm` after wasm-bindgen (cargo profile `web`, no wasm-opt):

| Build | Raw | gzip -9 | brotli -q 11 |
|---|---|---|---|
| b6c96 + b10c128 (before) | 17,107,018 | 3,416,083 | 2,101,447 |
| b6c96 + b10c128 + b18c384nbt (now the default) | 17,317,684 | 3,450,343 | 2,119,054 |
| b10c128 only / b18c384nbt only (side builds, same pipeline up to a few KB) | 17.08 MB / 17.22 MB | 3.41 / 3.44 MB | 2.10 / 2.11 MB |

The generated model code is small (parametrized by tensor shapes, weights are not embedded), so a separate build
variant would save 0.2 MB at most. Not worth it; `--no-default-features -F net-b10c128` still builds a b10-only wasm.

Weights: `kata1-b18c384nbt.bpk` 105,343,168 bytes (g170e-b10c128: 11,988,352; g170-b6c96: 4,120,704). It does not
compress (fp32): gzip -6 98.0 MB, brotli q5 97.9 MB (-7%), so serving it precompressed buys nothing; plan on
~98-105 MB over the wire (8-9 s at 100 Mbit/s, 1 s at 1 Gbit/s). `onProgress` reports the download.

## Timings (Chromium 156, localhost)

`ready` timings, WebGPU, ms:

| | wasm init | weights | sha256 | engine start (model build) | warm-up (bucket batches) | total |
|---|---|---|---|---|---|---|
| b18, persistent profile, cold | 131 | 141 (network) | 319 | 337 (134) | 1854 (1687) | **2994** |
| b18, persistent profile, warm | 109-119 | 145-153 (Cache API) | 233-237 | 348-351 (138) | 916-921 (748-757) | **1759-1783** |
| b18, fresh context, cold / warm | 104-118 / 95-110 | 145-177 / 69-75 | 234-246 / 238-251 | 352-470 / 356-409 | 1915-2201 / 1400-1499 | 2905-3386 / 2184-2342 |
| *b10, persistent profile, cold / warm* | 108 / 110-118 | 22 / 21-23 | 32 / 29-50 | 274 / 261-275 (39) | 1335 / 490-511 | *1831 / 939-958* |

- Warm = the kernel cache (47 kernels, all hits) and Chrome's GPU shader cache are in the profile. b18's warm-up is
  longer because the bucket warm-up really runs a 16-row and a 32-row b18 batch (~160 ms and ~320 ms); a "fresh
  context" has no GPU disk cache, which costs ~0.5 s of driver compiles per run.
- Cache API: the 105 MB weights are stored once, content-addressed (`usage` 106,179,109 bytes after the first load; the
  quota in this Chromium profile is 10.8 GB). A second load makes 0 requests (server and browser side). A cache write
  that fails (quota, private windows) logs a warning and the engine runs without a cache (existing M4 behaviour);
  not exercised with a full quota here.
- Memory after `ready`: wasm linear memory 424-445 MiB for b18 (b10c128: 68 MiB). The heap does not shrink, so this is
  the peak: the weights pass through it (a 105 MB copy from the JS buffer plus burn's tensor loading). VRAM: +523 and
  +566 MiB over idle (amdgpu `mem_info_vram_used`, two runs; the other-process baseline moves by hundreds of MiB, so
  ±100 MiB). Follow-up: load the tensors without the intermediate copies.
- GPU kernels compiled: 47 for b18 (42 for b10c128 in these runs, 45 in M4's count).

## Throughput and terminate (b18, WebGPU, default browser config 2 x 16 search threads, batches up to 32)

100-visit searches on fresh positions (NN cache cleared; the last run of `just web-test-b18`):

| | visits/s | evals/s | batches | avg rows |
|---|---|---|---|---|
| 100 visits | 48-71 (1.6-2.3 s per search) | same | 10-14 | 8.5-10.9 (padded to 16) |
| 400 visits | 57-100 (4.1-7.2 s) | 56-98 | 26-46 | 8.8-15.4 |

(b10c128, same page, same day: 144-297 visits/s at 100 visits, 234-416 at 400.) A batch of 15 rows takes ~155 ms
(padded to the 16 bucket; about half the time of a full 32-row batch, ~318 ms), so b18 evals/s are bounded by ~100-125 and by how
full the search can keep its batches: the single-query case fills ~10 of 16 rows (M4's "batches are smaller than the
thread count"). Config experiments on the same positions (400 visits, 3 positions, visits/s): 2 x 16 threads (default)
78 / 101 / 60; 1 x 32, batch 32: 103 / 78 / 84; 1 x 64, batch 64: 92 / 73 / 79; 1 x 16, batch 16: 97 / 67 / 66. Within the
noise between runs, so the defaults stay; a 1 x 32 config is slightly better for one query at a time and would be
a per-net default to consider.

Terminate (5 trials, cancel at a random point after 3 partial reports, partials every 500 ms): the engine completed
16 evaluations (one batch) after the terminate in all 5; wall clock from `cancel()` to the final response 10-155 ms
(average batch 155 ms). Partial reports with `reportDuringSearchEvery = 0.5` arrived every 500.4-500.7 ms.

## Parity

Upstream has no analysis golden for b18c384nbt (its goldens use g170-b6c96; the human-SL ones need M8's second
network). So the reference is our own Eigen run (`regen_b18.sh`: `basic.txt`, `symmetry.txt` with and without
`rootSymmetryPruning`, same deterministic invocations as `regen.sh`, 100 visits, 7x7 and 19x19 boards, ownership,
policy, PV; `pvvisits.txt` omitted: two 1000-visit searches take ~10 minutes on the Eigen build and minutes more in the browser at
batch 1). On WebGPU in the browser:

| Golden | Result | Max deviation |
|---|---|---|
| b18_basic | within tolerance (structure exact) | 3.59e-5 (weight, edgeWeight) |
| b18_symmetry | within tolerance | 2.15e-5 |
| b18_symmetry_with_pruning | within tolerance | 2.15e-5 |
| b18_basic on a device capped to WebGPU's default buffer limits | within tolerance | 3.59e-5 |

They pass the comparator's `webgpu` tier (1e-4 on values); `b18_basic` is slightly outside the stricter `flex` tier
(1e-5 absolute) on one key (`rawVarTimeLeft`, 2.1e-5 on a value of 9.5), the other two pass both tiers. Native: `just nn-parity -F wgpu` on the 72 b18 evals of
`fixtures/upstream-v1.18.2/nn/evals_b18.txt` (flex and wgpu vs Eigen), both under the usual bounds.

## GPU limits

- Adapter limits here: `maxStorageBufferBindingSize` = `maxBufferSize` = 4,294,967,292; 16 storage buffers per stage;
  workgroup count 65535. `ReadyInfo.adapterLimits` reports the first two. cubecl-wgpu requests the adapter's limits
  for the device (`required_limits: adapter.limits()`), so the engine's device gets them, not WebGPU's defaults
  (134,217,728 and 268,435,456).
- b18's tensors: the largest weight is 1.3 MB (192x192x3x3), the activations are 0.55 MB per batch row, and the
  biggest buffer is the im2col matrix of the 62 3x3 192-channel convs: 2.5 MB per row, 40 MB at 16 rows, 80 MB at 32,
  160 MB at 64. Nothing in the default config (batches of at most 32) exceeds even the default limits.
- Checked on a device capped to the default limits (`test/page/lowlimits-worker.js` caps `requiredLimits`; the log
  confirms `maxStorageBufferBindingSize 134217728`): the default config and the `b18_basic` golden work; a config with
  `nnMaxBatchSize=64` (160 MB buffers) fails with `readback failed` (a fatal engine error; the wasm traps). So on
  adapters that expose only the defaults, `nnMaxBatchSize` must stay <= 53 for b18 (<= 32 is the default). A
  guard in the worker would need such an adapter to test; none here.
- Subgroups: the `subgroups` device-feature shim (M0) is unchanged and still needed.

## CPU fallback

The wasm CPU runs b18 at 0.76-0.91 evals/s (4-10 visits with one search thread: 4413-13145 ms for 4-10 evals; `ready`
in 0.7-0.9 s, weights network or cache; b10c128: ~6-11 evals/s): a 100-visit query takes about two minutes and a terminate waits ~1 s
for the batch in flight. Not usable. `KataGoClient` therefore:

- `backend: "auto"` / `"cpu"` with b18 and no `cpuModel`: rejects `ready` with "kata1-b18c384nbt needs WebGPU: ... Give
  a smaller network as cpuModel (e.g. g170e-b10c128), or set allowSlowCpu", fetching nothing;
- with `cpuModel`: loads that net instead (it must be in the wasm; `ready.network` is the loaded one,
  `ready.requestedNetwork` the asked-for one, `fallbackReason` says why WebGPU was not used);
- `allowSlowCpu: true`: runs it anyway (tests do this for 4 visits).

## When to use b18 vs b10 in the app

- **Live analysis while the user plays or browses** (partials every 0.1-0.5 s, 100-400 visits): b10c128. 240-290
  visits/s, 0.9 s warm start, 12 MB, works on the CPU fallback (slowly). b18 gives ~50-70 visits/s here: a 100-visit
  answer takes ~2 s, a terminate ~0.15 s.
- **Game review / "deep" analysis / the stronger second opinion**: b18c384nbt, when WebGPU is present and the user
  opts in: the first load is 105 MB (show `onProgress`) and then it is cached; a 250-move game at 100 visits per
  position is ~8-10 minutes, at 400 visits ~30 minutes. Hand the client both models:
  `new KataGoClient({ model: b18, cpuModel: b10, backend: "auto" })`, so browsers without WebGPU get b10.
- Do not make b18 the default for first-time visitors (download size, 3 s cold start, ~0.45 GiB wasm memory +
  0.5 GiB VRAM). It also needs a GPU in the RX 6800 XT class to be pleasant: the numbers here are for a discrete
  GPU; an integrated GPU or a phone will be several times slower (none tested). Gate on a measured batch time
  (`kata.stats()`) rather than on the adapter name.
- One network per engine (M8's human-SL model is a second one); switching between b10 and b18 means a new client.

## Tests

- `just web-test` (b6/b10 suite): 45/45 pass after the conv change and the client changes (goldens on WebGPU and CPU,
  caches, terminate, fallbacks).
- `just web-test-b18`: 21/21: weights fetched once then from the Cache API (no request on reload), no kernel
  compiled on the second load, adapter limits >= the biggest buffer, 4 goldens on WebGPU (3 + 1 on a default-limit device),
  partials, 100 / 400-visit searches (printed), 5 terminate trials (<= 155 ms, one batch), CPU guard (3 refusals, `cpuModel`
  swap without a b18 download, `allowSlowCpu` run). Needs the b18 weights in the build (default now) and the Eigen goldens.
- `just web-test-client`: 8/8. `just nn-parity -F wgpu`: 7/7. `cargo test -p katago-wasm`: 7 passed (6 unit, 1 kernel cache).

## Gaps and follow-ups

- Kernel choice without autotune: pinning cubek's `DoubleUnit` min-tile matmul would add 15-45% on both nets (above).
- Wasm memory: 424 MiB peak for b18 from the weights' trip through wasm; streaming them in or freeing the
  intermediate copy would cut it.
- `nnMaxBatchSize` is not clamped for adapters with default limits (> 53 rows fail for b18).
- Search fills ~10 of 16 batch rows for single queries (M4); with b18's batch time proportional to the padded size
  this costs up to ~35%.
- No wasm-opt, no Firefox / Safari / iGPU / mobile runs, `device.lost` not handled (all as M4).
- The b18 goldens exclude `pvvisits` (cost) and anything with the human-SL net.
