# Inference speed-ups, 2026-11

Date: 2026-11-27. burn 0.22.1, cubecl 0.11.2 (cubecl-wgpu vendored), from master `7ebdc9c`.
Machine: i7-8700K, Radeon RX 6800 XT (RDNA2), Linux 7.3, Mesa RADV (Vulkan), HIP 7.16 (opencl-amd 10.1.0).

## Verdict

Three kernel changes and one routing change raise b18c384nbt's forward pass by **+20-24% in f16 on Vulkan**
(1110 -> 1337 evals/s at batch 32, 1108 -> 1377 at 64) and +18-21% on ROCm, +8-13% in f32. b10c128 gains on ROCm
(f16 +12-17% from batch 16 on); the transformer net b10c384h6nbttflrs +3-4% in f16 from batch 8 on. Outputs stay inside
the existing parity tiers: f32 within the WebGPU tier of the Eigen reference, f16 within upstream's own FP16 error.

No dependency upgrade is part of it: burn 0.22.1 is still the latest stable release (0.23 is pre-release only), and its
changes are mostly to the matmul and conv autotune that katago-rs does not use (its convs, reduces and linears are its
own cubecl kernels).

## Throughput

`just bench <wgpu|rocm> <net> <f32|f16>` (wgpu without autotune, the browser's kernel set), evals/s with one readback
per batch, batch 1 / 8 / 16 / 32 / 64, master vs this branch, binaries run interleaved; the mean of two clean
repetitions each (batch 1 varies by up to 15% between runs of one binary: the GPU's clocks are on `auto`).

| Net | Backend | Precision | master | branch |
|---|---|---|---|---|
| b18c384nbt | Vulkan | f16 | 153 756 992 1110 1108 | 160 **841 1152 1337 1377** |
| | Vulkan | f32 | 154 619 803 886 858 | 155 667 867 973 954 |
| | ROCm | f16 | 148 626 806 923 967 | 158 709 945 1095 1171 |
| | ROCm | f32 | 127 506 640 723 753 | 143 528 690 780 827 |
| b10c128 | Vulkan | f16 | 388 2276 3606 5749 7138 | 432 2453 4090 6499 8867 |
| | Vulkan | f32 | 395 2183 4150 5270 5957 | 405 2129 4187 5238 7097 |
| | ROCm | f16 | 623 3063 4513 5620 6482 | 657 3118 4799 6414 7496 |
| | ROCm | f32 | 531 2617 3761 4649 5163 | 540 2678 4012 5133 5836 |
| b10c384h6nbttflrs | Vulkan | f16 | 119 573 723 758 768 | 110 603 748 780 796 |
| | Vulkan | f32 | 121 568 659 696 730 | 114 563 664 702 736 |
| | ROCm | f16 | 115 514 572 611 660 | 125 535 591 642 686 |
| | ROCm | f32 | 123 504 543 589 627 | 120 502 543 589 627 |

b10c128 on Vulkan is the noisiest (master's f16 at batch 64 measured 5000 to 7400 across runs): a small net whose
batches finish before the clocks settle. Its gains there are real but smaller than the table's spread.

## Where the time went

Profile of one b18c384nbt f16 forward at batch 32 on Vulkan (`CUBECL_DEBUG_OPTION=profile-medium`), with the f16 1x1
GEMM already in: Winograd GEMM 25.5%, 1x1 GEMM 21.8%, Winograd input transform 21.6%, untransform 16.2%, burn's
elementwise kernels 7.0%, burn's matmul (the nets' small linears) 3.9%, burn's binary ops (the residual adds) 3.8%.

## What changed (kept)

1. **1x1 convolutions in f16** (`conv1x1_gemm_v`): with `useFP16` / `precision: "f16"`, the 1x1 convs (all of
   b18c384nbt's bottleneck and head convs whose output width is a multiple of 4) run a packed-f16 GEMM, `wino_gemm_v`'s
   scheme: the shared tiles as `vec4<f16>` (A along the board points, B along the output channels), each 16-channel
   step summed in f16 and added into f32 accumulators, the input and output in f32 and the batch norm + activation
   prologue applied in f32. Before, only the 3x3 convs ran in f16 and the 1x1 GEMM was 38% of the f16 forward
   (docs/M9-report.md). b18 f16 +5-10%; f32 unchanged.
2. **Winograd input transform through shared memory** (`wino_transform`): each cube's 64 tiles lie on at most four
   boards of one channel; the cube loads those into shared memory with coalesced reads and applies the prologue once
   per point, then each thread reads its 6x6 window from there. Before, every thread read its window straight from
   global memory (lanes 4 points apart, so nearly a cache line each) and activated each point once per window that
   covers it (2.25x on 19x19). +5-13% at batches 16-64 in both precisions and on both backends; f32 results are bit-identical.
3. **mish in closed form**: `tanh(log1p(e^a)) = n / (n + 2)` with `n = e^a (e^a + 2)`, one exp and one division
   instead of exp, log1p and tanh, in the three kernels that fuse the prologue. +1.5-2%.
4. **The conv nets' linears through `tf::linear`** (build.rs `route_linears`, so far only on the transformer nets):
   the global-pooling and value-head linears are tiny GEMMs that burn's matmul runs at ~90 us each without autotune.
   +2-3.5% on b18; b10c128 f16 at batch 8 +12% on Vulkan. One loss: ROCm b10c128 f32 at batch 1, -8% (526 vs 573),
   a configuration nothing ships.

## Tried and dropped

- **Residual adds in the conv epilogue** (44 in b18c384nbt, 10 in b10c128; a build.rs rewrite of `r.add(conv(...))`):
  in the Winograd untransform it was **-5%** (the residual is read in the untransform's 4x4-tile pattern, scattered,
  where burn's separate add reads it coalesced), in the 1x1 GEMM's store neutral. Not worth the code.
- **A burn or cubecl upgrade**: burn 0.22.1 is the latest stable; 0.23.0-pre.2 pins cubecl 0.12.0-pre.2 (a breaking
  `CubeType` derive change, a new tile layer behind its matmul and conv autotune, ndarray removed). cubecl 0.11.3/0.11.4
  are semver patches (an idle server thread that parks, a write-overwrite fix), but the vendored cubecl-wgpu would have
  to be re-vendored and re-patched; not attempted for a change that does not touch these kernels.
- **INT8**: cubecl 0.11 has no packed int8 dot product (`dot4I8Packed`, `VK_KHR_shader_integer_dot_product`); RDNA2
  has no matrix cores. The GEMMs (~47% of the forward) could at best double, the memory-bound transforms not at all,
  and the Winograd domain does not quantize well. A 1x1-only int8 path with per-channel scales is a separate study.

## Accuracy

`nn_parity` on Vulkan and ROCm (Eigen reference, fixtures/upstream-v1.18.2/nn): 12/12 pass on both master and this
branch (`cargo nextest run --release -p katago-nn-burn --test nn_parity -F wgpu,rocm,net-b10c384h6nbttflrs -E
'test(/wgpu|rocm/)'`). Max |burn - eigen|, master -> branch:

| test | winrate | lead | stScoreError | policy | ownership |
|---|---|---|---|---|---|
| b18c384nbt wgpu f32 | 1.37e-6 -> 1.49e-6 | 1.11e-4 -> 1.87e-4 | 3.15e-5 -> 5.05e-5 | 4.77e-6 -> 3.76e-6 | 5.16e-6 -> 8.52e-6 |
| b18c384nbt wgpu f16 | 1.90e-3 -> 2.25e-3 | 0.324 -> 0.285 | 0.188 -> 0.134 | 7.06e-3 -> 5.76e-3 | 2.22e-2 -> 1.86e-2 |
| b18c384nbt rocm f32 | 1.67e-6 -> 1.73e-6 | 2.37e-4 -> 2.63e-4 | 9.01e-5 -> 2.37e-4 | 4.77e-6 -> 5.72e-6 | 1.10e-5 -> 3.17e-5 |
| b18c384nbt rocm f16 | 2.03e-3 -> 2.40e-3 | 0.241 -> 0.361 | 0.073 -> 0.283 | 9.14e-3 -> 6.78e-3 | 1.29e-2 -> 4.61e-2 |
| b18c384nbt-humanv0 wgpu f16 | 1.28e-3 -> 1.04e-3 | 0.291 -> 0.121 | 2.54e-2 -> 3.21e-2 | 2.68e-3 -> 2.21e-3 | 6.89e-3 -> 6.31e-3 |
| b10c128 wgpu f32 | 2.62e-6 -> 2.62e-6 | 1.64e-4 -> 1.64e-4 | 0 -> 0 | 1.07e-5 -> 1.07e-5 | 1.92e-5 -> 1.92e-5 |
| b10c384h6nbttflrs wgpu f16 | 3.31e-3 -> 2.76e-3 | 9.04e-2 -> 9.81e-2 | 1.76e-2 -> 1.06e-2 | 8.80e-3 -> 8.36e-3 | 1.17e-2 -> 1.05e-2 |
| b10c384h6nbttflrs rocm f16 | 3.83e-3 -> 2.11e-3 | 9.17e-2 -> 7.90e-2 | 1.04e-2 -> 1.07e-2 | 9.26e-3 -> 7.14e-3 | 1.33e-2 -> 1.11e-2 |

The f16 bounds (upstream's `testgpuerror`, FP16 vs FP32 on this GPU) are winrate 4.45e-2, lead 2.27, stScoreError
0.612, policy 5.04e-2, ownership 9.23e-2. Everything is inside them; the largest move is ROCm b18 f16, whose ownership
error grew 3.6x to half its bound (Vulkan's shrank); the f32 rows move by rounding (the closed-form mish, the linears'
summation order) and stay in the WebGPU tier.

## Tests run

- `nn_parity` GPU tests as above: 12/12, master and branch.
- `cargo check -p katago-wasm --target wasm32-unknown-unknown`; `cargo fmt --all --check`.
- Browser (Chromium headless, WebGPU; the WGSL path, which native Vulkan does not exercise: cubecl-wgpu compiles to
  SPIR-V there): `just web-test --only goldens,throughput --backends webgpu` 9/9 (b10c128 100-visit searches
  800-980 visits/s); `just web-test-b18` 23/25: f16 vs f32 within upstream's FP16 error (winrate 8.4e-4, lead
  1.5e-2, ownership 6.2e-3, policy 5.0e-3), 400-visit f16 searches 481-645 visits/s (M9: 461-550). The two failures
  are one position (rect2, line 5) of `b18_symmetry` and `b18_symmetry_with_pruning`: an ownership of 0.234375 vs the
  golden's 0.234373, 2e-6 apart (tolerance 1e-4), reported as DIVERGED because test/compare.mjs `sameFloatClass` takes
  a 6-digit value that is exactly an f32 (0.234375 = 15/64) for a different number format. A comparator false
  positive; the harness is unchanged here.
