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
  (`just web-test-tf`). cubecl kernels for their blocks (token GEMM, RMSNorm, fused attention with RoPE): 2-3.5x
  native KataGo v1.18.2 OpenCL's forward passes on an RX 6800 XT ([Benchmarks](#transformer-nets)).
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

Measured 2026-10-26 on one machine: Intel i7-8700K (6 cores, 12 threads), AMD Radeon RX 6800 XT (RDNA2), Linux 7.2,
Mesa 26.2.5 (RADV for Vulkan/WebGPU), ROCm 7.2.3 (HIP, and AMD's OpenCL for upstream), Chromium 156. katago-rs at
this README's commit, against upstream KataGo v1.16.5 (`ba938676`). Every row has a command; the milestone reports
([M0](docs/M0-report.md), [M4](docs/M4-report.md), [M7](docs/M7-report.md), [M9](docs/M9-report.md)) have the
history and the profiles. Nothing here was measured on NVIDIA, Intel or Apple GPUs, or in Firefox or Safari.

### Forward passes

Network evaluations per second, random inputs, one readback of all outputs per batch (as the engine reads them).
`just bench <backend> <net> <precision>`; wgpu and rocm run the cubecl kernels without burn's autotune (the browser's
kernel set, and katago-ws's images). f16 runs the 3x3 convolutions in f16 (`useFP16` / katago-web `precision`).

| Net | Backend | Precision | batch 1 | 8 | 16 | 32 | 64 |
|---|---|---|---:|---:|---:|---:|---:|
| b10c128 | flex (CPU, 1 thread) | f32 | 30 | 33 | | | |
| | wgpu (Vulkan) | f32 | 680 | 3685 | 5756 | 6671 | 7116 |
| | wgpu (Vulkan) | f16 | 608 | 4169 | 6552 | 8046 | **8582** |
| | rocm (HIP) | f32 | 654 | 3408 | 4671 | 5641 | 6859 |
| | rocm (HIP) | f16 | 638 | 4101 | 5633 | 6849 | 8315 |
| b18c384nbt | flex (CPU, 1 thread) | f32 | 2.2 | 2.5 | | | |
| | wgpu (Vulkan) | f32 | 193 | 658 | 835 | 918 | 893 |
| | wgpu (Vulkan) | f16 | 200 | 801 | 1013 | 1139 | 1152 |
| | rocm (HIP) | f32 | 154 | 630 | 720 | 912 | 1160 |
| | rocm (HIP) | f16 | 181 | 756 | 888 | 1104 | **1364** |

Upstream KataGo for comparison, `just bench-upstream <net>` (KataGo's own `benchmark`, 800 visits; its nnEvals/s at
the average batch its search threads produce, so a search is running around them). OpenCL FP16 on AMD's OpenCL
device (`CONFIG` with `openclDeviceToUse = 0`; Mesa's rusticl device, which KataGo picks by default, is slower at
small batches):

| Net | Engine | avg batch 4 | 8 | 16 | 32 | 49 |
|---|---|---:|---:|---:|---:|---:|
| b10c128 | KataGo OpenCL FP16 | 2725 | 3476 | 3774 | 4120 | 4962 |
| b18c384nbt | KataGo OpenCL FP16 | 295 | 356 | 401 | 441 | 449 |
| b10c128 | KataGo Eigen (CPU, 12 threads) | 164 at batch 1 | | | | |

So on this GPU katago-rs's b18c384nbt forward pass is about 2.5x upstream's at batch 32 (1104-1139 vs 441), and
b10c128's 1.7-2x (6849-8046 vs 4120). On the CPU, flex is single-threaded: about upstream's Eigen per thread
(M0), and 5x behind Eigen on 12 threads.

### Searches in the browser

Chromium 156 headless, WebGPU through Vulkan, visits per second of one search at a time on fresh positions (NN cache
cleared), `just web-build && just bench-web`:

| Net | Backend | 100 visits | 400 visits |
|---|---|---:|---:|
| b10c128 | WebGPU (`precision: auto`, f16 here) | 783-961 | |
| b10c128 | wasm CPU | 11 | |
| b18c384nbt | WebGPU f32 | 326-442 | 399-647 |
| b18c384nbt | WebGPU f16 | 422-529 | 474-672 |

One search fills batches of only 7-15 rows; many searches at once (game review, problem generation) reach the
forward-pass numbers above.

### As a server (katago-ws)

[katago-ws](https://github.com/Sagebati/katago-ws)'s quick bench: 6 games analysed at every turn, 100 visits, 4 games
in flight, one engine process, two runs each (`scripts/bench_setup.sh`, then `BENCH_PRESET=quick scripts/bench.sh`
there). Positions per second:

| Net | Engine | Settings (NN servers x batch, analysis threads) | positions/s |
|---|---|---|---:|
| b18c384nbt | katago-rs, wgpu (Vulkan) f16 | 4 x 64, 256 | **15.6 / 15.5** |
| | katago-rs, rocm f16 | 8 x 64, 256 | 13.4 / 13.2 |
| | KataGo OpenCL FP16, Mesa rusticl (2026-10-25) | 4 x 64, 128 | 7.1 / 7.1 |
| | KataGo OpenCL FP16, AMD OpenCL | 4 x 64, 128 | 6.3 / 6.3 |
| b10c128 | katago-rs, wgpu (Vulkan) f16 | 4 x 64, 256 | **83.5 / 81.7** |
| | katago-rs, rocm f16 | 8 x 64, 256 | 66.6 / 65.6 |
| | KataGo OpenCL FP16, AMD OpenCL | 4 x 64, 256 | 80.6 / 76.9 |
| | KataGo OpenCL FP16, Mesa rusticl (2026-10-25) | 4 x 64, 256 | 63.0 / 71.6 |

The analysis quality matches upstream's run-to-run noise (docs/M9-report.md). b18c384nbt is GPU-bound and katago-rs
is about 2x upstream's throughput. b10c128 leans on the search side instead (katago-rs runs every search on one OS
thread, the wasm design), and the GPU's per-batch overhead matters more: on Vulkan, one katago-rs process matches
upstream's best, and on ROCm it reaches about 0.85x. katago-ws's `BENCH_BACKEND=upstream` runs the KataGo rows.

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
