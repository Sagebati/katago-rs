# katago-rs

KataGo's analysis engine ported to Rust, compiled to WebAssembly to run in the browser
(WebGPU, with a wasm CPU fallback). The port mirrors upstream's `cpp/` file by file so that
following upstream releases stays cheap.

Status:
- **M0** spike ([docs/M0-report.md](docs/M0-report.md)) and **M0.5** pilot: porting guide
  ([docs/PORTING.md](docs/PORTING.md)), ownership table, upstream manifest.
- **M1** game rules (`board`, `boardhistory`, `rules`, `graphhash`), **M2** NN inputs and the
  NNEvaluator on burn, **M3** search and the analysis engine protocol: ported and checked against
  upstream oracles, natively and on wasm32. Search with the NN skipped is bit-exact with upstream;
  burn-backed searches give the same visit counts as upstream's Eigen build.
- `katago-cli`: native `katago-rs analysis -backend flex|wgpu|rocm`, speaking upstream's analysis protocol on
  stdin/stdout. It is [katago-ws](https://github.com/Sagebati/katago-ws)'s engine, the server side of the same
  analysis (its images build it from this repository).
- **M4** the engine in the browser ([docs/M4-report.md](docs/M4-report.md)): `katago-wasm` in a Web
  Worker, on WebGPU or the wasm CPU, and the `katago-web` npm package (`KataGoClient`); upstream's
  analysis goldens pass in Chromium on both backends.
- **M6a** full-game review in the browser ([docs/M6-report.md](docs/M6-report.md)): `katago-annotate`
  (katago-ws's annotation, wasm-safe, byte-identical payloads on recorded games) and
  `KataGoClient.reviewGame`, which returns the payload v3 that baduk_review stores for a server analysis.
- **M7** kata1-b18c384nbt in the browser ([docs/M7-report.md](docs/M7-report.md)): same wasm (+0.2 MB), 105 MB
  weights fetched on demand, WebGPU only (`cpuModel` falls back to b10c128 without WebGPU). Checked against
  Eigen-generated goldens in Chromium (`just web-test-b18`).
- **M8** human-SL nets: b18c384nbt-humanv0 (`-human-model`, `humanSLProfile`), the SGF metadata inputs and the
  human-SL search, checked against Eigen natively (`analysis-goldens`) and in wasm (`just test-wasm-human`).
- **M9** WebGPU kernels at native KataGo speed ([docs/M9-report.md](docs/M9-report.md)): Winograd 3x3 convs, an NCHW
  1x1 GEMM, f16 (default with `shader-f16`) and fused batch-norm/mish prologues in cubecl. b18c384nbt forward passes
  at 2.5x native KataGo OpenCL FP16 on an RX 6800 XT (1137 vs 457 evals/s at batch 32); a 400-visit b18 search in
  Chromium runs at 461-550 visits/s. Current numbers: [Benchmarks](#benchmarks).
- Transformer nets (model v15/v17, attention + RMSNorm blocks): b10c384h6nbttflrs, b10c512h8nbt3tflrs-fson-silu-rsnh and
  the main-run kata1-tf3-b11c768 (opt-in `net-*` features of katago-nn-burn, katago-cli and, for the two smaller ones,
  katago-wasm), checked against Eigen v1.18.2 on flex, wgpu and rocm (`just nn-parity -F net-...`) and through the
  analysis engine (`just analysis-goldens -F net-b10c384h6nbttflrs`), and b10c384h6nbttflrs in Chromium
  (`just web-test-tf`). cubecl kernels for their blocks (token GEMM, RMSNorm, fused attention with RoPE): 1.7-2.7x
  native KataGo v1.18.2 OpenCL's forward passes at batches 8-64 on an RX 6800 XT, and 2x its positions/s analysing
  games with tf3 ([Benchmarks](#benchmarks)).
- Not ported: loading the SGF files of the avoid-repeated-moves pattern tables (`avoidSgfPatternDirs`, file IO);
  the keys are read and an error is raised if one is fully configured. Decisions: [docs/DECISIONS.md](docs/DECISIONS.md).

## Layout

| Path | What |
|---|---|
| `upstream/KataGo` | upstream KataGo (submodule, pinned to a release tag) |
| `tools/bin2onnx` | KataGo `.bin.gz` → ONNX converter and its verifier against real KataGo |
| `crates/katago-core` | the port, mirroring upstream `cpp/` (core, game, neuralnet, search, program, command) |
| `crates/katago-nn-burn` | networks imported with burn-onnx (`NetId`); the GPU convs (cubecl kernels in `wino/`) and transformer blocks (`tfk/`); the NN backend |
| `crates/katago-cli` | native `katago-rs analysis` binary (analysis protocol over stdin/stdout) |
| `crates/katago-annotate` | katago-ws's game annotation (SGF → queries → payload v3), IO-free and wasm-safe |
| `tools/annotate-parity` | a separate cargo workspace replaying katago-ws's own annotator against katago-annotate (needs `../katago-ws`) |
| `crates/katago-wasm` | browser host: wasm-bindgen `Engine` running the analysis engine in a Web Worker (WebGPU / wasm CPU) |
| `packages/katago-web` | npm package: module worker, `KataGoClient` (`analyze`, `reviewGame`), protocol and payload types; Playwright tests (`just web-build`, `just web-test`) |
| `fixtures/` | reference data generated from upstream KataGo |

## Benchmarks

Measured on one machine, 2026-10-27, at this README's commit: Intel i7-8700K (6 cores, 12 threads), AMD Radeon RX
6800 XT (RDNA2), Linux 7.2, Mesa 26.2.5 (RADV for Vulkan and WebGPU, rusticl for OpenCL), AMD's opencl-amd 10.1.0
(HIP 7.16 for ROCm, and AMD's OpenCL), Chromium 156 headless. "KataGo" is upstream KataGo **v1.18.2** (the submodule,
`just katago-opencl`), OpenCL with `useFP16 = auto`, on either driver: Mesa's rusticl or AMD's OpenCL. The milestone
reports ([M0](docs/M0-report.md), [M4](docs/M4-report.md), [M7](docs/M7-report.md), [M9](docs/M9-report.md)) have the
history and the profiles. Nothing here was measured on NVIDIA, Intel or Apple GPUs, or in Firefox or Safari.

Everything native: `just bench-all [nets]` (about half an hour per net, logs in target/bench/). Everything in the
browser: `just web-build -F net-b10c384h6nbttflrs,net-b10c512h8nbt3tflrs-fson-silu-rsnh,net-tf3-b11c768`, then
`just bench-web-all`. The pieces have their own recipes, below.

### Forward passes (evals/s)

Network evaluations per second at a fixed batch, each batch's outputs read back. katago-rs: `just bench <wgpu|rocm>
<net> <f32|f16>`, random inputs, without burn's autotune (the browser's kernel set); f16 runs the 3x3 convolutions,
and the transformer nets' token linears from 6 boards per batch on, in f16. KataGo: its `benchmarknn`, `just
bench-upstream-nn <net.bin.gz> <libRusticlOpenCL.so.1|libamdocl64.so>`. Browser (WebGPU in a Web Worker): the engine
evaluating every turn of two games under their 8 symmetries at 1 visit, with the batch size forced; evals/s over the
time the NN spends on its batches (`just bench-web-all`).

| Net | Engine | Precision | batch 1 | 8 | 16 | 32 | 64 |
|---|---|---|---:|---:|---:|---:|---:|
| b10c128 | KataGo, Mesa rusticl | auto | 164 | 1254 | 2014 | 2334 | 3412 |
| | KataGo, AMD OpenCL | auto | 1216 | 5050 | 5478 | 6372 | 6718 |
| | katago-rs, Vulkan | f32 | 402 | 2464 | 4177 | 5235 | 6098 |
| | katago-rs, Vulkan | f16 | 426 | 2497 | 4609 | 6383 | **7174** |
| | katago-rs, ROCm | f32 | 524 | 2617 | 3730 | 4653 | 5171 |
| | katago-rs, ROCm | f16 | 643 | 2986 | 4514 | 5814 | 6530 |
| | katago-rs, browser WebGPU | f32 | 128 | 811 | 1592 | 2498 | 3199 |
| | katago-rs, browser WebGPU | f16 | 118 | 795 | 1504 | 2729 | 3497 |
| | katago-rs, flex (CPU, 1 thread) | f32 | 30 | 33 | | | |
| b18c384nbt | KataGo, Mesa rusticl | auto | 51 | 242 | 342 | 511 | 597 |
| | KataGo, AMD OpenCL | auto | 157 | 367 | 421 | 449 | 465 |
| | katago-rs, Vulkan | f32 | 159 | 591 | 772 | 863 | 845 |
| | katago-rs, Vulkan | f16 | 178 | 715 | 938 | 1062 | **1075** |
| | katago-rs, ROCm | f32 | 129 | 509 | 642 | 725 | 754 |
| | katago-rs, ROCm | f16 | 147 | 633 | 812 | 926 | 967 |
| | katago-rs, browser WebGPU | f32 | 60 | 383 | 570 | 708 | 796 |
| | katago-rs, browser WebGPU | f16 | 54 | 384 | 647 | 819 | 972 |
| | katago-rs, flex (CPU, 1 thread) | f32 | 2.2 | 2.5 | | | |
| b10c384h6nbttflrs | KataGo, Mesa rusticl | auto | 45 | 213 | 288 | 395 | 444 |
| | KataGo, AMD OpenCL | auto | 92 | 160 | 170 | 174 | 178 |
| | katago-rs, Vulkan | f32 | 129 | 580 | 674 | 706 | 738 |
| | katago-rs, Vulkan | f16 | 129 | 578 | 725 | 758 | **773** |
| | katago-rs, ROCm | f32 | 128 | 507 | 548 | 595 | 631 |
| | katago-rs, ROCm | f16 | 127 | 525 | 580 | 613 | 668 |
| | katago-rs, browser WebGPU | f32 | 46 | 285 | 483 | 586 | 680 |
| | katago-rs, browser WebGPU | f16 | 45 | 293 | 515 | 624 | 705 |
| b10c512h8nbt3tflrs-fson-silu-rsnh | KataGo, Mesa rusticl | auto | 29 | 129 | 153 | 179 | 192 |
| | KataGo, AMD OpenCL | auto | 45 | 65 | 67 | 68 | 68 |
| | katago-rs, Vulkan | f32 | 81 | 251 | 298 | 318 | 326 |
| | katago-rs, Vulkan | f16 | 81 | 285 | 318 | 348 | **360** |
| | katago-rs, ROCm | f32 | 83 | 236 | 256 | 277 | 284 |
| | katago-rs, ROCm | f16 | 83 | 269 | 285 | 304 | 313 |
| | katago-rs, browser WebGPU | f32 | 34 | 191 | 258 | 297 | 318 |
| | katago-rs, browser WebGPU | f16 | 35 | 214 | 294 | 327 | 354 |
| kata1-tf3-b11c768 | KataGo, Mesa rusticl | auto | 22 | 74 | 86 | 92 | 95 |
| | KataGo, AMD OpenCL | auto | 24 | 31 | 32 | 32 | 32 |
| | katago-rs, Vulkan | f32 | 54 | 143 | 150 | 159 | 157 |
| | katago-rs, Vulkan | f16 | 54 | 167 | 174 | **185** | 182 |
| | katago-rs, ROCm | f32 | 55 | 120 | 131 | 136 | 136 |
| | katago-rs, ROCm | f16 | 55 | 146 | 159 | 166 | 163 |
| | katago-rs, browser WebGPU | f32 | 32 | 127 | 143 | 151 | 151 |
| | katago-rs, browser WebGPU | f16 | 32 | 149 | 170 | 183 | 183 |

- b10c128: AMD's OpenCL is KataGo's faster driver, and at small batches it is ahead of katago-rs (1216 vs 643 at
  batch 1); from batch 32 on they are even, Vulkan 7174 against 6718 at 64.
- b18c384nbt: katago-rs on Vulkan is 1.8-2.2x the better KataGo driver from batch 8 on (1062 vs 511 at 32).
- The transformer nets: Mesa's rusticl is KataGo's faster driver from batch 8 on (AMD's caps out early), and
  katago-rs is 1.7-2.7x it (b10c384h6nbttflrs 2.7x at batch 8, 1.7x at 64; tf3 1.9-2.3x).
- Vulkan is the faster katago-rs backend from batch 16 on; ROCm wins b10c128 at batches 1 and 8.
- In the browser, WebGPU reaches about half of native Vulkan on b10c128 at batch 64 (its per-dispatch overhead
  shows on a small net), 90% on b18c384nbt and 91-100% on the transformer nets.
- Upstream's Eigen CPU build does 164 evals/s on b10c128 with 12 threads, 5x flex on one (M0).

Earlier versions of this table had higher rocm numbers: the bench dropped each readback unread, and on rocm burn's
`into_data` returns before the GPU is done, so the batches queued up and the bench timed the queue. It now reads
the values, as the engine does (which was never affected).

### Game analysis (positions/s)

Positions analysed per second over whole games: every turn of each game at 100 visits, with ownership and policy,
the query [katago-ws](https://github.com/Sagebati/katago-ws) sends for a game review. The 17 games of
fixtures/bench-games (katago-ws's bench set) are cycled, with the NN cache cleared each time they start over.

- Native (the KataGo and katago-rs columns): `just bench-analysis <wgpu|rocm|rusticl|amdocl> <net.bin.gz> <servers>
  <threads>`, katago-annotate's `analysis_bench` driving the `analysis` command of either engine (`katago-rs
  analysis`, or KataGo's) over stdin/stdout. 4 games in flight (katago-ws's worker concurrency); after a warm-up, two
  windows, both shown. Settings are katago-ws's, as NN server threads x batch, then analysis threads with one search
  thread each: katago-rs Vulkan 4 x 64, 256; ROCm 8 x 64, 256; KataGo 4 x 64 and 128 analysis threads (256 for
  b10c128). katago-rs in f16, built as katago-ws's images build it (Vulkan with burn's autotune, ROCm without).
  60 s of warm-up and 120 s windows, except KataGo on AMD's OpenCL for b10c512h8nbt3tflrs and tf3: 240 s and 240 s
  (its first response there takes up to two minutes).
- Browser: `KataGoClient.reviewGame` in Chromium, one game after another for 2 minutes, f16. The browser columns are
  from the bench that measured whole-game reviews (`benchPositions`), which has since been replaced by the
  per-position timing below, so `just bench-web-all` no longer reproduces them. "2 x 16" was the default then
  (2 analysis threads x 16 search threads, batch 32) and is now 4 analysis threads; on the same 692 positions that
  went from 14.1 to 19.9 positions/s (average batch 10.5 to 20.5). "64 x 1" overrides the engine to 64 analysis
  threads of one search thread and batch 64 (`numAnalysisThreads=64, numSearchThreadsPerAnalysisThread=1,
  nnMaxBatchSize=64`), which fills the batches (47-51 rows on average against 11-12).

| Net | KataGo, Mesa rusticl | KataGo, AMD OpenCL | katago-rs, Vulkan | katago-rs, ROCm | browser, 2 x 16 | browser, 64 x 1 |
|---|---:|---:|---:|---:|---:|---:|
| b10c128 | 44.4 / 43.9 | **101.7 / 100.7** | 75.1 / 72.5 | 69.4 / 68.0 | 16.0 | 27.7 |
| b18c384nbt | 6.36 / 6.71 | 5.65 / 5.48 | **14.2 / 14.2** | 12.2 / 11.6 | 5.80 | 8.79 |
| b10c384h6nbttflrs | 5.17 / 5.06 | 2.26 / 2.17 | **10.1 / 9.74** | 8.21 / 8.06 | 4.34 | 6.22 |
| b10c512h8nbt3tflrs-fson-silu-rsnh | 2.38 / 2.19 | 0.94 / 0.77 | **4.38 / 4.73** | 3.43 / 3.81 | 2.21 | 3.24 |
| kata1-tf3-b11c768 | 1.25 / 1.17 | 0.40 / 0.48 | **2.56 / 2.27** | 2.25 / 2.02 | 1.16 | 1.82 |

- The GPU-bound nets (b18c384nbt and the transformers): katago-rs on Vulkan analyses about 2x as many positions per
  second as KataGo on its better driver (b18c384nbt 2.2x, b10c384h6nbttflrs 1.9x, b10c512h8nbt3tflrs 2.0x, tf3 2.0x),
  and ROCm 1.6-1.8x.
- b10c128 is bound by the search, not the GPU: KataGo on AMD's OpenCL is ahead, katago-rs reaching 0.73x on Vulkan.
  katago-rs runs every search on one OS thread (the wasm design), where KataGo spreads its 256 analysis threads over
  the CPU's 12.
- The browser runs one review at a time where the native runs keep 4 games going; with 64 x 1 it still analyses
  tf3 games faster than native KataGo (1.82 against 1.25 positions/s), and reaches 62-75% of native katago-rs on
  Vulkan with the GPU-bound nets.
- The analysis quality matches upstream's run-to-run noise (docs/M9-report.md).

### One position in the browser (ms)

The time to analyse one position to 200 visits, the setting a drill's check uses on WebGPU, with 8-move lines:
`KataGoClient.analyze` on the middle move of each of the 17 games of fixtures/bench-games (the first is a warm-up),
b10c128, f16, default config (4 analysis threads), Chromium, WebGPU. Run with
`just bench-web-all --bench-nets g170e-b10c128 --bench-precisions f16 --bench-batches 32 --bench-visits 192,200,208,224`.

| Visits asked | Visits reached | Mean | Median | Min-max | Average batch |
|---:|---:|---:|---:|---:|---:|
| 192 | 206-207 | 176 ms | 170 ms | 146-214 ms | 13.3 |
| 200 | 214-215 | 190 ms | 182 ms | 162-245 ms | 13.1 |
| 208 | 223 | 192 ms | 186 ms | 164-255 ms | 13.3 |
| 224 | 239 | 195 ms | 188 ms | 167-266 ms | 14.1 |

A repeat at 200 visits gave 211 ms mean (198 ms median), so treat about 190-210 ms as the figure.

- One position fills the NN batches only to about 13-14 rows, not 32: a query has 16 search threads and each holds at
  most one leaf waiting on the network. `nnMaxBatchSize` therefore only matters when several queries run together
  (reviews, or drills in parallel).
- A search overshoots its visit cap by about 15 (the leaves in flight). Time follows the number of network rounds, so
  200 and 208 visits cost about the same.
- One query uses one analysis thread, so the 4-thread default makes no difference to a single position (not measured
  at 2 threads). The native engines are not measured on this bench.

### One search in the browser (visits/s)

Visits per second of one search at a time on fresh positions (NN cache cleared), Chromium, WebGPU, 2026-10-25/27:
`just web-build && just bench-web`, and `just web-build -F net-b10c384h6nbttflrs && just web-test-tf`. One search
fills batches of only 7-15 rows; many searches at once (a game review, problem generation) get close to the forward
passes above.

| Net | Backend | 100 visits | 400 visits |
|---|---|---:|---:|
| b10c128 | WebGPU (`precision: auto`, f16 here) | 783-961 | |
| b10c128 | wasm CPU | 11 | |
| b18c384nbt | WebGPU f32 | 326-442 | 399-647 |
| b18c384nbt | WebGPU f16 | 422-529 | 474-672 |
| b10c384h6nbttflrs | WebGPU f32 | 351-375 | 391-561 |
| b10c384h6nbttflrs | WebGPU f16 | 390-405 | 388-606 |

## Setup

```
git submodule update --init
python3 -m venv .venv && .venv/bin/pip install numpy onnx onnxruntime sgfmill
just katago-eigen
just convert /path/to/g170e-b10c128.bin.gz g170e-b10c128
just parity flex
```

## License

KataGo's licence ([LICENSE](LICENSE), MIT, © David J Wu and contributors): the port derives from
[KataGo](https://github.com/lightvector/KataGo). Networks are licensed separately by their publishers.
