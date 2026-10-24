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

ONNX ops used: `Conv Add Mul Sub Div Sqrt Relu MatMul Reshape Concat Slice
ReduceSum ReduceMax` (+ `Softplus Tanh` for Mish nets). All reductions are over a
single axis; all shape/axes inputs are constant initializers.

## Verification

`verify.py` compares against a real KataGo binary via GTP `kata-raw-nn 0`, on
positions x {japanese, chinese} rules (19x19 openings and SGF positions, a
capture + ko, a pass, plus 9x9/13x13 boards inside the 19x19 graph). Features come
from upstream KataGo's MIT python code (`upstream/KataGo/python/katago/game`).

```sh
.venv/bin/python -I tools/bin2onnx/verify.py \
  --model /home/sam/src/web/katago-ws/bench/models/g170e-b10c128.bin.gz \
  --onnx models/onnx/g170e-b10c128.onnx
```

For a human-SL net every position is evaluated once per `--human-sl-profiles` entry
(default `preaz_18k,rank_9d,rank_1k_15k,proyear_1850,proyear_2022`): KataGo runs with
`humanSLProfile=<profile>`, the ONNX side gets the metadata row from upstream's
`python/katago/game/sgfmetadata.py` (profile fields mirrored from `SGFMetadata::getProfile`).
b18c384nbt-humanv0 (2026-10-24): 5 profiles x 26 positions, max diff policy 1.8e-6,
win/loss 1.3e-6, ownership 3.1e-6, lead 5.1e-4 (GTP print precision).

By default it uses `build/eigen/katago` (Eigen fp32 build of the submodule) if present.
For an OpenCL build pass `--katago ... --extra-overrides openclUseFP16=false`
(and `--home-data-dir` with existing tuning). Tolerances: `--tol-prob 1e-4`,
`--tol-score 1e-3`. Exit code 0 = pass.
