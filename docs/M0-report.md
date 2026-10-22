# M0 spike report: KataGo b10c128 through burn, native and in the browser

Date: 2026-10-22. Upstream: KataGo v1.16.5 (`ba938676`). burn 0.22.1, burn-onnx 0.22.0.
Machine: Intel i7-8700K (6C/12T), AMD Radeon RX 6800 XT (RDNA2, RADV), Arch Linux, Chromium 155.

## Verdict: go (Rust route), with two upstream issues to file

Every risk M0 was meant to retire is retired for b10c128:

| Risk | Result |
|---|---|
| Can we get KataGo nets into ONNX? | Yes, with our own converter (`tools/bin2onnx`), verified against KataGo's fp32 Eigen backend on b6c96, b10c128 **and b18c384nbt** |
| Does burn-onnx import the graph unmodified? | Yes, first try (203 nodes, 13 op types) |
| Are burn's outputs right? | Yes on every backend: native CPU, native Vulkan, browser WebGPU, browser wasm CPU |
| Is WebGPU fast enough for live analysis? | Yes on a discrete GPU: 590–830 evals/s at batch 16–64 in Chrome |
| Is the CPU fallback usable? | Only for low visit counts: 11 evals/s single-threaded in wasm |

## 1. Converter: `.bin.gz` → ONNX

`kaya-go/katago-onnx` was not usable: it is **AGPL-3.0** and only converts PyTorch `.ckpt`
checkpoints, which the g170 nets don't have. `tools/bin2onnx/bin2onnx.py` (MIT) parses the
`.bin.gz` format with one function per `*Desc` in `desc.cpp` (same `modelVersion` gates) and
builds the graph following `eigenbackend.cpp`. Opset 17, plain ops only:
`Add Concat Conv Div MatMul Mul ReduceMax ReduceSum Relu Reshape Slice Sqrt Sub`
(+ `Softplus Tanh` for Mish in v14).

`tools/bin2onnx/verify.py` compares onnxruntime + `nneval.cpp` post-processing against GTP
`kata-raw-nn 0` from the Eigen build (`just katago-eigen`) on 13 positions × {japanese, chinese}:
empty, ko capture, after pass, six game positions, 9x9 and 13x13 through the board mask.

| net | policy | win/loss/noresult | ownership | lead / scoreMean |
|---|---|---|---|---|
| g170e-b10c128 (v8) | 2.0e-6 | 1.8e-6 | 4.0e-6 | 4.7e-4 / 4.9e-4 |
| g170-b6c96 (v8) | 2.2e-6 | 1.4e-6 | 2.9e-6 | 4.7e-4 / 4.6e-4 |
| kata1-b18c384nbt (v14) | 2.3e-6 | 2.0e-6 | 3.6e-6 | 4.8e-4 / 5.0e-4 |

Score differences are GTP's 3-decimal printing. A negative control (friendly-pass emulation off)
fails loudly, so the test does catch input mistakes; the small-board cases caught a real
max-pooling mask bug during development.

## 2. burn import and raw-output parity

`crates/katago-nn-burn`: `build.rs` runs `burn_onnx::ModelGen` with `LoadStrategy::Bytes`
(weights fetched at runtime, not embedded). Raw outputs vs onnxruntime on the same 26 inputs
(`just parity <backend>`, browser page `crates/katago-web-bench`), logits up to |23.65|:

| backend | policy | value | score | ownership |
|---|---|---|---|---|
| flex (native CPU) | 1.9e-5 | 4.3e-6 | 1.2e-6 | 1.7e-6 |
| wgpu (native Vulkan) | 2.5e-5 | 3.8e-6 | 1.0e-6 | 2.4e-6 |
| WebGPU (Chrome) | 1.4e-5 | 5.7e-6 | 1.9e-6 | 2.4e-6 |
| flex (wasm, Chrome) | 1.5e-5 | 5.2e-6 | 1.4e-6 | 1.4e-6 |

## 3. Throughput (b10c128, 19x19, NN evals/s)

| Engine | batch 1 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|
| KataGo OpenCL fp16 native (benchmark, 1 → 12 search threads) | 274 | | | | 1310 at avg batch 5.8 |
| KataGo Eigen native, SSE (1 / 6 / 12 threads) | 31.5 | | | | 126 / 150 |
| burn flex native, 1 thread | 30 | 32 | | 33 | |
| burn wgpu native, burn's conv | 110 | 172 | 146 | 154 | 137 |
| burn wgpu native, **im2col conv** | 138 | 712 | 836 | 967 | 944 |
| **burn WebGPU in Chrome** (im2col + fusion) | 46 | 364 | 592 | 740 | 827 |
| **burn flex in wasm** (SIMD128, 1 thread) | 10.8 | 11.2 | | | |

- First run at a new batch shape costs 1–3 s (shader compile + autotune): warm up the shapes
  the search uses at engine start.
- Batch 1 in the browser has a ~20 ms floor (dispatch + `mapAsync` round trip), so the search
  must batch (planned "virtual threads", plan §6). At batch 16, 100 visits ≈ 0.17 s.
- wasm CPU is ~2.8× below native per core, as expected for SSE-class code on SIMD128. With
  wasm threads (COOP/COEP; the app is compatible) it could reach the Emscripten/Eigen
  estimate (~45–55 evals/s on 6 cores); single-threaded it is the same as Eigen-in-wasm would be.

### b18c384nbt (model v14, M7's net) — checked early

The converter output (verified above) also imports unmodified through burn-onnx (968 nodes,
Mish as `x·tanh(softplus(x))`); `just parity <backend> b18` passes on flex (1.3e-5 policy,
1.9e-5 score) and native wgpu (3.0e-5 / 2.7e-5). Native throughput: wgpu 30 / 109 / 122 / 131
evals/s at batch 1 / 8 / 16 / 32 (~7.4× slower than b10, ~9× the FLOPs); flex 2.2 evals/s.
Expect roughly 100 evals/s in the browser on this GPU: fine for full-game review, ~1 s per
100 visits for live analysis; CPU fallback is not viable for b18.

## 4. Problems found and fixed

1. **burn's direct conv kernel is slow on RDNA2** (no matrix cores): a 3x3 128→128 conv runs at
   ~400 GFLOP/s while the equivalent matmul runs at ~7 TFLOP/s. Fixed in our code with a
   burn-onnx `OpOverride` (`SameConv` in `build.rs`) that routes convs to
   `katago_nn_burn::conv::conv2d_same`: im2col + one matmul on GPU, burn's conv on CPU.
   Weights layout and `.bpk` are unchanged. ~7× on the GPU.
2. **WebGPU shaders fail in Chrome: `extension 'subgroups' is not allowed`.** wgpu 30's
   WebGPU backend never maps/requests the `subgroups` device feature, but cubecl-wgpu's WGSL
   matmul reads the subgroup-size builtin and always emits `enable subgroups;`. Worked around
   with a 10-line JS shim adding `subgroups` to `requestDevice` when the adapter has it
   (`www/index.html`). **To file upstream** (cubecl and/or wgpu).
3. **burn-flex ran without its `simd` feature** → no SIMD128 gemm in wasm (3.1 evals/s). Cause:
   our workspace uses `burn` with `default-features = false`, which also drops
   `burn-dispatch`'s `burn-flex?/default` (where `simd` lives). Fixed by depending on
   `burn-flex` with `simd` directly (3.5×). Not an upstream bug, but easy to miss.
4. `+relaxed-simd` (wasm FMA) doesn't compile: macerator 0.5.1 trait mismatch. **To file upstream.**
   Drafts for both upstream issues: [upstream-issues.md](upstream-issues.md).
5. ndarray is deprecated in burn 0.22: use `flex` for CPU everywhere.
6. The Claude desktop app's built-in browser exposes `navigator.gpu` but returns no adapter on
   Linux; browser numbers above come from Playwright + system Chromium
   (`--enable-unsafe-webgpu --enable-features=Vulkan`).

## 5. Open items carried forward

- Wasm size: 26 MB raw / 4.5 MB gzip (unoptimised: no `wasm-opt`, `opt-level=3`). Try
  `opt-level="s"` + `wasm-opt -Oz` in M4.
- More GPU speed if needed: keep activations NHWC to drop the permutes around each im2col,
  f16 (`shader-f16` is available), Winograd for 3x3.
- Measure on an integrated GPU and an Apple M-series before M5's UI budget is fixed.
- b18c384nbt in the browser (native done above; 105 MB `.bpk` download and GPU buffers).

## Reproduce

```
just katago-eigen                     # reference KataGo (CPU fp32)
just convert <net.bin.gz> g170e-b10c128 && just verify <net.bin.gz> g170e-b10c128
just parity flex && just parity wgpu && just parity wgpu b18
just bench wgpu && just bench flex 1 8
just web-bench                        # then serve crates/katago-web-bench/www
```
