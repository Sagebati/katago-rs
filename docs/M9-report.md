# M9 report: WebGPU kernels at native KataGo speed

Date: 2026-10-25. Upstream: KataGo v1.16.5 (`ba938676`). burn 0.22.1, cubecl 0.11.2.
Machine: as M0/M4/M7 (i7-8700K, Radeon RX 6800 XT on RADV, ROCm 7.2.3 installed), Chromium headless through
Playwright. Idle machine for the benchmarks below unless noted.

## Verdict

The goal was katago-rs in the browser at native KataGo speed, since problem and solution generation will run
client-side. It is there and past it: on the same GPU, b18c384nbt's forward pass with the browser's kernel set
is **2.5x native KataGo's OpenCL FP16 backend** at every batch size measured, and a single 400-visit search in
Chromium runs at native KataGo's single-search speed.

| | batch 8 | 16 | 32 | 64 |
|---|---|---|---|---|
| KataGo 1.16.5 OpenCL, FP16 storage + compute (`katago benchmark`, avg batch 7.9 / 15.9 / 33.0) | 322 | 391 | 457 | |
| katago-rs before (M7 browser kernel set: `wgpu-fixed-kernels` + fusion) | 71 | 113 | 123 | 143 |
| katago-rs now, f32 | 668 | 834 | 917 | 888 |
| katago-rs now, f16 (default on adapters with `shader-f16`) | **796** | **1031** | **1137** | **1125** |

b18c384nbt evals/s, forward passes with readback (`cargo run --release -p katago-nn-burn --example bench
-F wgpu-fixed-kernels,net-b18c384nbt -- wgpu b18 ...`, `WINO_HALF=1` for f16). Batch 1: 11.2 ms -> 5.9 ms.

In the browser (`just web-test-b18`, one search at a time, so batches of only 8-15 rows):

| b18c384nbt, Chromium, WebGPU | 100 visits | 400 visits |
|---|---|---|
| M7 | 48-71 visits/s | 57-100 visits/s |
| now, f32 | 286-340 | 381-478 |
| now, f16 | 301-357 | 461-550 |
| (native KataGo, one search, 16-32 threads, 800 visits) | | 376-430 |

b10c128 in the browser: 641-695 visits/s for 100-visit searches (M7: 144-297). Many searches at once (problem
generation, game review) fill the batches and get the forward-pass numbers of the first table.

## What changed

All in `crates/katago-nn-burn` unless noted; the kernels are cubecl, reached through a burn backend extension
(`#[backend_extension(Cube, Fusion)]`, `src/wino.rs`), so the generated models and the `.bpk` files are unchanged.

1. **Winograd F(4x4, 3x3) convolutions** (`wino.rs`), following upstream's OpenCL backend: the input tile
   transform (`winogradTransformNCHW` / `winogradBNActTransformNCHW`), one GEMM per tile point (36, upstream's
   `doBatchedXGemm_KM_KN_NM` layout), the untransform (`winogradUntransformNCHW`), and the filter transform of
   `ConvLayer` (`transform3x3_6`) done on the device (`wino_filter_3x3`, no readback: wasm cannot read a tensor
   synchronously). 3.6x fewer multiplies than direct 3x3 on 19x19 and no im2col copies: 7-12x faster per layer
   than im2col + matmul without autotune.
2. **NCHW 1x1 GEMM** for every 1x1 conv with a 16-multiple input (all of b18's, including the heads): no NHWC
   permutes, bit-identical to the matmul path.
3. **f16** (`set_half`, katago-web `precision`): the transforms store f16, the GEMM is a `vec4<f16>` kernel
   (packed FMAs on RDNA) that sums each 16-channel step in f16 and accumulates the steps in f32, and writes f32.
   Scalar f16 is no faster than f32 here; the packed math is the +20%.
4. **Fused prologues**: build.rs `fuse_bn_act` rewrites each generated batch norm -> mish -> mask chain that only
   feeds convs (94 in b18c384nbt) into `conv2d_bn_act`, which runs the chain inside the Winograd transform or
   the 1x1 GEMM's input loads, as upstream's `bnActTransform`. Halves the per-forward time at batch 1 (the
   elementwise kernels were most of the launches).
5. **Global pooling reduces** (build.rs `SpatialReduce`): ReduceSum/ReduceMax over the board as one workgroup
   per row (burn's reduce cost ~90 us per call at batch 1 without autotune; now ~3 us).
6. katago-wasm `setHalfPrecision`; katago-web `EngineInitOptions.precision` (`auto` default: f16 when the
   adapter has `shader-f16`) and `ReadyInfo.precision`. `wgpu-fixed-kernels` now implies `fusion`: the
   extension's Fusion routing must exist whenever burn's fusion is on.

## Accuracy

f32 is unchanged in kind: `nn_parity` on wgpu against the Eigen reference stays within the WebGPU tier
(b18: policy 2.6e-6, ownership 2.2e-5, lead 1.6e-4; the 1x1 and prologue paths are bit-identical to what
they replace), and the browser goldens pass on WebGPU, including with default buffer limits.

f16 is held to upstream's own FP16. `katago testgpuerror` (OpenCL FP16 vs FP32, this GPU, b18c384nbt, 2247
positions) gives maxima of winrate 4.45%, lead 2.27, scoreMean 2.28, short-term winloss error 4.32c, short-term
score error 0.612, ownership 9.23c, top policy 5.04%; the new `nn_parity_*_wgpu_f16` tests use those bounds.
Measured (b18, 72 evals vs Eigen): winrate 2.2e-3, lead 0.20, short-term score error 0.16, policy 8.6e-3,
ownership 1.0e-2, that is 3-18x inside upstream's FP16 error. In Chromium, f16 vs f32 on 12 positions: winrate
5.8e-4, lead 6.2e-3, ownership 6.4e-3, policy 5.7e-3. Summing the whole K in f16, as upstream does, was ~3x
worse (winrate 7.3e-3, ownership 6.6e-2) and only 1-2% faster.

The browser test page pins `precision: "f32"` for its Eigen comparisons (goldens, reviews) and checks f16
separately against f32.

## Where the time goes now

Profile of one f16 forward at batch 32 before the prologue fusion (cubecl `profile-medium`): 1x1 GEMMs 38%,
Winograd GEMM 22% (~60% of FP32 peak in f32), elementwise 21%, transforms 15%. At batch 1 the GPU is busy
6.2 ms of an 11.2 ms forward (444 kernels); the rest is launch overhead, which is larger in wasm. The prologue
fusion took out ~95 of those kernels. Left on the table:

- The residual adds (`add(x, conv(...))`) could go into the untransform / 1x1 store (the kernel has the
  `add_residual` option; build.rs would need a second rewrite).
- Small batches: the 1x1 GEMM has too few workgroups at batch 1-4 (64x64 tiles over 361 points); smaller tiles
  or split-K for small batches.
- The ReLU nets (b6c96, b10c128) get no prologue fusion yet (the rewrite matches mish only).
- Per-GPU tuning of the tile sizes (upstream's `opencltuner.cpp`), cached with the WGSL kernels.

## Tests run

- `cargo test --release -p katago-nn-burn --test nn_parity -F wgpu`: 11/11 (flex, wgpu f32, wgpu f16).
- `examples/wino_bench`: per-layer outputs and timings vs im2col, device filter transform vs the host reference
  (7e-9).
- `just web-test-b18`: 25/25 (goldens, cache, kernels, limits, throughput f32 and f16, f16 vs f32 bound,
  terminate, CPU guard). `just web-test`: 60/60.

## Review

An independent review of the branch found no result-corrupting defect in the shipped shapes, and four issues,
all fixed:

- The untransform and the reduce launched 1-D grids that pass WebGPU's 65535 workgroups per dimension at large
  batches (b18's input conv at N >= 437): both are 2-D now (x within a batch row, y over rows). b18 at batch 512
  runs (615 evals/s f32, 659 f16: slower per eval than at 32-64, not investigated; the browser uses <= 64).
- The filter cache was keyed by ParamId alone, which burn-onnx fixes per net at build time: a reload of the same
  net after `set_half` changed would have reused the other precision's filters. Keyed by (ParamId, precision,
  device) now.
- The browser's f16-vs-f32 check skipped negative ownership (the policy's illegal-move filter applied to
  ownership too).
- `auto` precision: an f16 WebGPU engine that fails to build is retried in f32 before falling back to the CPU;
  `is_using_fp16` reports the f16 convs.

## Native ROCm (HIP)

`-F rocm` (`bench rocm ...`, `nn_parity -F wgpu,rocm -- rocm`) runs the same cubecl kernels on burn's ROCm backend
(ROCm 7.2.3), native only:

| b18c384nbt evals/s | batch 1 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|
| burn ROCm + fusion before this work | | 133 | 174 | 232 | 347 |
| ROCm now, f32 | 155 | 630 | 726 | 910 | 1155 |
| ROCm now, f16 | 159 | 758 | 899 | 1104 | **1376** |

About WebGPU's numbers up to batch 32 and ahead at 64 (WebGPU f16 1125 there). Parity on ROCm: f32 policy 3.8e-6,
ownership 9.8e-6; f16 winrate 1.7e-3, lead 0.13, ownership 1.3e-2 (inside upstream's FP16 error).
