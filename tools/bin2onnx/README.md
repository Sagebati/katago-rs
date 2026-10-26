# bin2onnx

Converts a KataGo network (`.bin.gz`, also `.txt.gz` / `.bin` / `.txt`) to an ONNX
graph (opset 17) that computes the **raw** network outputs, exactly like KataGo's
Eigen backend before `NNEvaluator` post-processing.

The parser follows `cpp/neuralnet/desc.cpp` (one function per `*Desc`, same
`modelVersion` gates) and the graph follows `cpp/neuralnet/eigenbackend.cpp`.
Verified numerically against KataGo on v8 (`g170*`) and v14 (`kata1-b18c384nbt`:
nested bottleneck blocks, Mish, 2-channel policy, 6 score channels). The v3–v5 test
models in `upstream/KataGo/cpp/tests/models` parse and convert but are not
numerically verified; the v15/v16 code paths (pass-head MLP, 4-channel policy) are
written from desc.cpp/eigenbackend.cpp; v15's pass-head MLP is verified through the human-SL
net `b18c384nbt-humanv0` (v15, `metaEncoderVersion` 1: the SGF metadata encoder, a third
input `meta_input`), the 4-channel policy is untested.

Model version 17 (KataGo v1.17/v1.18) and transformer nets are supported and verified against
the v1.18.2 Eigen build: transformer attention / SwiGLU FFN blocks (also inside nested-bottleneck
blocks), learnable and fixed 2D RoPE, grouped-query attention, RMSNorm trunk tips (spatial and
per-position), SiLU. See [Transformer nets](#transformer-nets-model-version-17).

## Usage

```sh
python3 -m venv .venv && .venv/bin/pip install numpy onnx onnxruntime sgfmill
.venv/bin/python -I tools/bin2onnx/bin2onnx.py path/to/net.bin.gz -o models/onnx/net.onnx
```

Options: `--pos-len N` (fixed spatial size, default 19), `--no-check`.

## Graph contract

| name | shape | meaning |
|---|---|---|
| `bin_input` | `[N, C_in, 19, 19]` | spatial features (V7: 22 ch), NCHW; channel 0 = on-board mask |
| `global_input` | `[N, G_in]` | global features (V7: 19) |
| `meta_input` | `[N, 192]` | only for human-SL nets (`metaEncoderVersion > 0`): the row of `SGFMetadata::fillMetadataRow`. The encoder (`mul1, bias1, act1, mul2, bias2, act2, mul3`, desc.cpp `SGFMetadataEncoderDesc`) is added to the trunk as a per-channel bias after the global one (eigenbackend `Trunk::apply`) |
| `policy` | `[N, P, 362]` | policy logits, index `y*19+x`, last = pass. P = 1 (v<12), 2 (v12–15: normal, optimistic), 4 (v16) |
| `value` | `[N, 3]` | win / loss / no-result logits, player-to-move perspective |
| `scorevalue` | `[N, S]` | raw score outputs (v8: scoreMean, scoreStdev pre-softplus, lead, varTimeLeft pre-softplus; v9+: + 2 shortterm error outputs) |
| `ownership` | `[N, 1, 19, 19]` | ownership, pre-tanh, player-to-move perspective |

`N` is dynamic. Pooling is mask-aware (sums divided by the mask sum, max pooling pushes
off-board points to -1e4), so a smaller board can be evaluated by zero-padding the
inputs inside the 19x19 grid, as KataGo does with `gtpForceMaxNNSize`.

To get KataGo's numbers, apply `nneval.cpp` post-processing: combine policy channels
with `policyOptimism` (`p + (pOpt - p) * opt`), mask illegal moves, softmax; softmax the
value; multiply score outputs by the multipliers in the ONNX `metadata_props`
(`scoreMeanMultiplier`, `leadMultiplier`, ... — header values for v13+, desc.cpp
defaults otherwise, flagged by `postProcessParamsFromHeader`); `tanh` the ownership;
flip signs to white's perspective when black is to move. `verify.py` has a reference
implementation (`postprocess`).

Metadata keys: `name`, `modelVersion`, `posLen`, `numInputChannels`,
`numInputGlobalChannels`, `numInputMetaChannels`, `metaEncoderVersion` (written since M8;
older exports lack it and are all version 0), `numPolicyChannels`,
`numValueChannels`, `numScoreValueChannels`, `numOwnershipChannels`,
`trunkNumBlocks`, `trunkNumChannels`, the post-process multipliers, and `outputs`.

Keys added for model version 17, written only for nets that use v17-era features
(`modelVersion >= 17`, transformer blocks, an RMSNorm trunk tip, or a nonzero flag) so that older
exports stay byte-identical; **absent means 0**:

| key | values | meaning (desc.cpp `ModelDesc`) |
|---|---|---|
| `preferPassAliveUnderSuicideRules` | 0/1 | featurize pass-alive area (`calculateArea` / `calculateNonDameTouchingArea`) as if multi-stone suicide were legal, whatever the rules say (the suicide global feature and move legality keep the real rule). KataGo's search sets `BoardHistoryModes.alwaysComputePassAliveUnderSuicideRules` from it |
| `preferExcludeTerritoryAdjacentToAtari` | 0/1 | rules version 3: territory scoring with `TAX_NONE` (adjudication and the encore-phase territory feature) excludes empty points adjacent to the owner's chains in atari (`BoardHistoryModes.excludeTerritoryAdjacentToAtari`) |
| `trunkNormKind` | 0/1 | 0 = BatchNorm trunk tip, 1 = RMSNorm trunk tip (informational, already in the graph) |
| `hasAnyTransformerBlocks` | 0/1 | informational |

All four released v17 nets and the test nets have both prefer flags = 0.

ONNX ops used: `Conv Add Mul Sub Div Sqrt Relu MatMul Reshape Concat Slice
ReduceSum ReduceMax` (+ `Softplus Tanh` for Mish nets; + `Exp` for SiLU nets; + `Transpose
Softmax` for transformer nets). All reductions are over a single axis; all shape/axes inputs are
constant initializers. No `Gather`, `Where`, `Pow`, `Sigmoid` or `Erf`.

## Transformer nets (model version 17)

What v17 changed in the file format (desc.cpp):

- model header (v15+ slots, now meaningful): after `metaEncoderVersion`, two 0/1 flags
  `preferPassAliveUnderSuicideRules`, `preferExcludeTerritoryAdjacentToAtari`, then 5 slots that
  must be 0;
- trunk header (v15+ slots): first slot is `trunkNormKind` (0 BatchNorm, 1 RMSNorm), then 5 slots
  that must be 0. With `trunkNormKind` 1 the trunk tip is an `RMSNormLayerDesc` (`name C eps
  spatial cgroupSize(=0) gamma[C] beta[C]`) instead of a BatchNorm; the tip activation follows;
- block kinds `transformer_attention_block` (`numHeads numKVHeads qHeadDim vHeadDim useRope
  learnableRope`, a `TransformerRMSNormDesc` (`name C eps weight[C]`), q/k/v/out MatMuls, then
  either `name numKVHeads numPairs 2 freqs[...]` (learnable RoPE) or `name theta` (fixed RoPE)) and
  `transformer_ffn_block` (`C ffnC useSwiGLU`, RMSNorm, linear1, [linearGate], linear2), allowed in
  the trunk and inside nested-bottleneck blocks. Transformer blocks are not version-gated:
  `b10c384h6nbttflrs` is a v15 file;
- policy head: `policyOutChannels` (2 or 4) is stored in the file, then 3 zero slots; value head:
  3 zero slots (v17 dropped the v16 Q-value channels unless the file asks for 4);
- `ACTIVATION_SILU` (`x / (1 + exp(-x))`).

Graph (eigenbackend `TransformerAttentionBlock` / `TransformerFFNBlock` / `RMSNormLayer`):

- A run of consecutive transformer blocks works on tokens `[N, S, C]` (`S = 361`, `s = y*19+x`,
  eigenbackend's NHWC layout): `Reshape [N,C,S]` + `Transpose` in, `Transpose` + `Reshape` out.
- RMSNorm (pre-norm of each block): `Mul(x,x) ReduceSum(axis C) Div(C) Add(eps) Sqrt Div(1,.)
  Mul(x,rms) Mul(weight) Mul(mask)`.
- Attention: RMSNorm; `MatMul` q/k/v; `Reshape [N,S,h,d]` + `Transpose [N,h,S,d]`; RoPE;
  (GQA) `Reshape` Q to `[N,kvh,g*S,d]`; `Transpose` K; `MatMul` scores; `Mul(1/sqrt(qHeadDim))`;
  `Add(attn bias)`; `Softmax(axis 3)`; `MatMul` V; (GQA) `Reshape [N,h,S,vd]`; `Transpose`
  `Reshape [N,S,h*vd]`; `MatMul` out; `Mul(mask)`; residual `Add`.
- FFN: RMSNorm; `MatMul` linear1 and gate; SiLU(linear1) as `Mul(-1) Exp Add(1) Div`; `Mul` gate;
  `MatMul` linear2; `Mul(mask)`; residual `Add`. Non-SwiGLU FFNs are rejected (eigenbackend and
  upstream's ONNX exporter do not support them either).
- Masking: eigenbackend skips off-board keys in the softmax (their weight is exactly 0) and zeroes
  off-board query rows. Here the scores get `attn_mask_bias = mask*3e4 - 3e4` (`[N,1,1,S]`: 0 on
  board, -3e4 off board, so `exp` underflows to 0); off-board query rows compute finite garbage that
  the `Mul(mask)` before the residual add discards, as eigenbackend's `trunk += out * mask` does.
- RoPE: the cos/sin tables of `computeRopeCosSin(19, 19)` are constants (`[1,h,S,P]` per query
  head and `[1,kvh,S,P]` per kv head for learnable RoPE, `[1,1,S,P]` for fixed). Positions are
  coordinates in the 19x19 grid, as eigenbackend with `gtpForceMaxNNSize` (smaller boards sit in
  the top-left corner). The pair rotation `(x0*cos - x1*sin, x0*sin + x1*cos)` uses two `Slice`s:
  the q/k projection columns are permuted per head from interleaved pairs `(2p, 2p+1)` to
  `[all x0 | all x1]` (attention scores are invariant to a common permutation of q and k).
- GQA: query head `h` uses kv head `h / (numHeads/numKVHeads)` (= eigenbackend's
  `h*numKVHeads/numHeads` for RoPE); viewing Q as `[N, kvh, g*S, d]` makes plain batched MatMuls
  share K/V without copies.
- Trunk tip RMSNorm: per-position (`sum over C`) or spatial (`Mul(mask)` then one sum over
  `C*S`, divided by `onboard*C`); `x*rms*gamma + beta`, activation, `Mul(mask)`.

Name pieces of the test/release nets: `b<N>c<C>` blocks/channels, `h<H>` heads, `nbt` nested
bottleneck (`nbt3`: 3 attention+FFN pairs per bottleneck), `tf` transformer, `rs` fixed RoPE,
`lrs` learnable RoPE, `fson` SwiGLU FFN, `silu` SiLU in conv/heads, `rsnh` spatial RMSNorm trunk
tip, `cnorm` per-position RMSNorm trunk tip, `bnh` BatchNorm trunk tip, `kv3` 3 kv heads (GQA),
`qk32 v16` head dims.

## Verification

`verify.py` compares against a real KataGo binary via GTP `kata-raw-nn 0`, on
positions x {japanese, chinese} rules (19x19 openings and SGF positions, a
capture + ko, a pass, plus 9x9/13x13 boards inside the 19x19 graph). Features come
from upstream KataGo's MIT python code (`upstream/KataGo/python/katago/game`).

```sh
.venv/bin/python -I tools/bin2onnx/verify.py \
  --model ../katago-ws/bench/models/g170e-b10c128.bin.gz \
  --onnx models/onnx/g170e-b10c128.onnx
```

For a human-SL net every position is evaluated once per `--human-sl-profiles` entry
(default `preaz_18k,rank_9d,rank_1k_15k,proyear_1850,proyear_2022`): KataGo runs with
`humanSLProfile=<profile>`, the ONNX side gets the metadata row from upstream's
`python/katago/game/sgfmetadata.py` (profile fields mirrored from `SGFMetadata::getProfile`).
b18c384nbt-humanv0 (2026-10-24): 5 profiles x 26 positions, max diff policy 1.8e-6,
win/loss 1.3e-6, ownership 3.1e-6, lead 5.1e-4 (GTP print precision).

Transformer nets against the v1.18.2 Eigen build (2026-10-26, 26 positions each, max abs diff;
probability print precision 1e-6, score 1e-3):

| net | policy | wl/nr | ownership | lead | scoreMean | result |
|---|---|---|---|---|---|---|
| b7c96h3tfrs-test5-cnorm | 4.2e-6 | 3.3e-6 | 5.8e-6 | 5.1e-4 | 5.6e-4 | PASS |
| b7c96h6kv3qk32v16tflrs-fson-bnh | 1.2e-6 | 8.8e-7 | 3.1e-6 | 4.6e-4 | 5.2e-4 | PASS |
| b4c256h4nbttflrs-fson-silu-rsnh | 8.1e-6 | 2.5e-6 | 6.5e-6 | 7.8e-4 | 1.6e-3 | PASS with `--tol-score-rel 5e-5` |
| b10c384h6nbttflrs (v1.17.0 release) | 4.7e-6 | 3.6e-6 | 9.1e-6 | 1.9e-3 | 2.0e-3 | PASS with `--tol-score-rel 5e-5` |
| b10c512h8nbt3tflrs-fson-silu-rsnh (release) | 6.1e-6 | 4.8e-6 | 1.3e-5 | 2.0e-3 | 1.6e-3 | PASS with `--tol-score-rel 5e-5` |
| b11c768h12nbt3tflrs-fson-silu (release) | 2.6e-6 | 6.2e-7 | 2.6e-6 | 5.0e-4 | 4.9e-4 | PASS |
| kata1-tf3-b11c768-s11003M-d5973M-7gres | 1.5e-6 | 1.4e-6 | 2.5e-6 | 4.6e-4 | 4.9e-4 | PASS |

The three nets with a spatial RMSNorm trunk tip (`rsnh`, and b10c384h6nbttflrs) miss the absolute
1e-3 score tolerance only on positions with |score| > 100 (ogs-60054068). That is
eigenbackend's own float32 error: its spatial RMSNorm sums `C*361` squares sequentially in
float, e.g. -4.1e-5 relative on b10c384 (rms +2.0e-5), which moves a -102.76 score by 2e-3. The
ONNX graph agrees with a float64 evaluation of the same graph to 7e-5 there.

By default it uses `build/eigen/katago` (Eigen fp32 build of the submodule) if present.
For an OpenCL build pass `--katago ... --extra-overrides openclUseFP16=false`
(and `--home-data-dir` with existing tuning). Tolerances: `--tol-prob 1e-4`,
`--tol-score 1e-3` (opt-in `--tol-score-rel R` also accepts lead/scoreMean diffs up to
`R*|reference|`). The summary prints max abs and max relative (`|d|/max(|ref|, floor)`) errors per
head. For nets with `preferPassAliveUnderSuicideRules` / `preferExcludeTerritoryAdjacentToAtari`
the features follow those modes, as KataGo's search does. Exit code 0 = pass.
