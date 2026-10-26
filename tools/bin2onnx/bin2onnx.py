#!/usr/bin/env python3
# Copyright (c) 2026 katago-rs contributors. MIT License.
"""Convert a KataGo neural net (.bin.gz / .txt.gz / .bin / .txt) to ONNX.

The parser mirrors upstream KataGo cpp/neuralnet/desc.cpp (one function per
*Desc, with the same modelVersion gates), and the graph mirrors the forward
pass of cpp/neuralnet/eigenbackend.cpp. Outputs are the RAW network outputs,
before NNEvaluator post-processing (no softmax, no tanh, no score scaling).

Graph I/O (NCHW, batch dimension "N" is dynamic):
  inputs:
    bin_input     float32 [N, numInputChannels, pos_len, pos_len]
    global_input  float32 [N, numInputGlobalChannels]
    meta_input    float32 [N, numInputMetaChannels]   (only for metaEncoderVersion > 0, i.e.
                  human-SL nets: SGFMetadata::fillMetadataRow's row; the SGF metadata encoder's
                  output is added to the trunk after the global input, as eigenbackend's Trunk)
  outputs:
    policy        float32 [N, numPolicyChannels, pos_len*pos_len + 1]   (last = pass)
    value         float32 [N, numValueChannels]        (win/loss/noresult logits)
    scorevalue    float32 [N, numScoreValueChannels]   (pre-scaling misc values)
    ownership     float32 [N, numOwnershipChannels, pos_len, pos_len]  (pre-tanh)

Channel 0 of bin_input is the on-board mask; all global pooling is mask-aware.

Model version 17 (KataGo v1.17+) and transformer blocks (which can also appear in v15 files):
transformer attention / SwiGLU FFN blocks (also nested inside nested-bottleneck blocks) run on
tokens [N, S=H*W, C]; attention ignores off-board keys through an additive softmax bias; RoPE
tables are constants for the fixed pos_len x pos_len grid (positions are grid coordinates, as in
eigenbackend with gtpForceMaxNNSize); GQA by viewing query heads grouped per kv head. RMSNorm
trunk tips (TRUNK_NORM_KIND_RMSNORM, spatial or per-position) and ACTIVATION_SILU are supported.
"""

import argparse
import gzip
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OPSET = 17

ACTIVATION_IDENTITY = "ACTIVATION_IDENTITY"
ACTIVATION_RELU = "ACTIVATION_RELU"
ACTIVATION_MISH = "ACTIVATION_MISH"
ACTIVATION_SILU = "ACTIVATION_SILU"  # model version 17

LATEST_MODEL_VERSION_IMPLEMENTED = 17
OLDEST_MODEL_VERSION_IMPLEMENTED = 3

TRUNK_NORM_KIND_STANDARD = 0  # BatchNorm trunk tip
TRUNK_NORM_KIND_RMSNORM = 1   # RMSNormLayerDesc trunk tip (spatial or per-position)

# Attention softmax bias for off-board keys: 0 on board, -ATTN_MASK_BIG off board (eigenbackend skips
# off-board keys; exp(score - BIG - rowmax) underflows to exactly 0 in float32). Representable in fp16.
ATTN_MASK_BIG = 3.0e4


# ---------------------------------------------------------------------------
# Token stream over the uncompressed model file (mirrors istream >> semantics)
# ---------------------------------------------------------------------------

class ModelReader:
    WS = b" \t\n\r\v\f"

    def __init__(self, data: bytes, binary_floats: bool):
        self.data = data
        self.pos = 0
        self.binary_floats = binary_floats

    def _skip_ws(self):
        d, p, n = self.data, self.pos, len(self.data)
        while p < n and d[p] in self.WS:
            p += 1
        self.pos = p

    def token(self) -> str:
        self._skip_ws()
        d, start, n = self.data, self.pos, len(self.data)
        p = start
        while p < n and d[p] not in self.WS:
            p += 1
        if p == start:
            raise ValueError("Unexpected end of model file")
        self.pos = p
        return d[start:p].decode("ascii")

    def int(self) -> int:
        return int(self.token())

    def float(self) -> float:
        return float(np.float32(self.token()))

    def bool(self) -> bool:
        v = self.int()
        if v not in (0, 1):
            raise ValueError(f"Expected bool 0/1 but got {v}")
        return bool(v)

    def floats(self, n: int, name: str) -> np.ndarray:
        if not self.binary_floats:
            arr = np.array([np.float32(self.token()) for _ in range(n)], dtype=np.float32)
        else:
            # "@BIN@" followed by n little-endian float32s
            d = self.data
            num_before = 0
            while True:
                if self.pos >= len(d):
                    raise ValueError(f"{name}: could not find binary float block")
                c = d[self.pos]
                self.pos += 1
                if c == ord("@"):
                    break
                num_before += 1
                if num_before > 100:
                    raise ValueError(f"{name}: could not read float weights (no @BIN@ header)")
            if d[self.pos:self.pos + 4] != b"BIN@":
                raise ValueError(f"{name}: did not find expected header for binary float block")
            self.pos += 4
            nbytes = 4 * n
            if self.pos + nbytes > len(d):
                raise ValueError(f"{name}: did not find the expected number of floats in binary float block")
            arr = np.frombuffer(d, dtype="<f4", count=n, offset=self.pos).astype(np.float32)
            self.pos += nbytes
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name}: Nan or infinite neural net weight or parameter")
        return arr


# ---------------------------------------------------------------------------
# Descriptors (mirror desc.h / desc.cpp)
# ---------------------------------------------------------------------------

@dataclass
class ConvLayerDesc:
    name: str
    conv_y: int
    conv_x: int
    in_channels: int
    out_channels: int
    dilation_y: int
    dilation_x: int
    weights: np.ndarray  # [oc, ic, y, x] (ONNX OIHW)


def parse_conv(r: ModelReader) -> ConvLayerDesc:
    name = r.token()
    cy, cx, ic, oc, dy, dx = (r.int() for _ in range(6))
    if cx <= 0 or cy <= 0:
        raise ValueError(f"{name}: convolution filter sizes must be positive")
    if ic <= 0 or oc <= 0:
        raise ValueError(f"{name}: number of in and out channels must be positive")
    if dx <= 0 or dy <= 0:
        raise ValueError(f"{name}: dilation factors must be positive")
    if cx % 2 != 1 or cy % 2 != 1:
        raise ValueError(f"{name}: convolution filter sizes must be odd")
    # Model file order is y,x,ic,oc
    w = r.floats(cy * cx * ic * oc, name).reshape(cy, cx, ic, oc).transpose(3, 2, 0, 1)
    return ConvLayerDesc(name, cy, cx, ic, oc, dy, dx, np.ascontiguousarray(w))


@dataclass
class BatchNormLayerDesc:
    name: str
    num_channels: int
    epsilon: float
    has_scale: bool
    has_bias: bool
    mean: np.ndarray
    variance: np.ndarray
    scale: np.ndarray
    bias: np.ndarray
    merged_scale: np.ndarray = field(init=False)
    merged_bias: np.ndarray = field(init=False)

    def __post_init__(self):
        # computeMerged(), done in float32 like the C++ code
        eps = np.float32(self.epsilon)
        self.merged_scale = (self.scale / np.sqrt(self.variance + eps)).astype(np.float32)
        self.merged_bias = (self.bias - self.merged_scale * self.mean).astype(np.float32)


def parse_bn(r: ModelReader) -> BatchNormLayerDesc:
    name = r.token()
    c = r.int()
    eps = r.float()
    has_scale = r.bool()
    has_bias = r.bool()
    if c < 1:
        raise ValueError(f"{name}: numChannels < 1")
    if not (eps > 0) or not np.isfinite(eps):
        raise ValueError(f"{name}: epsilon is not positive and finite")
    mean = r.floats(c, name)
    var = r.floats(c, name)
    scale = r.floats(c, name) if has_scale else np.ones(c, np.float32)
    bias = r.floats(c, name) if has_bias else np.zeros(c, np.float32)
    return BatchNormLayerDesc(name, c, eps, has_scale, has_bias, mean, var, scale, bias)


@dataclass
class ActivationLayerDesc:
    name: str
    activation: str


def parse_activation(r: ModelReader, model_version: int) -> ActivationLayerDesc:
    name = r.token()
    if model_version >= 11:
        kind = r.token()
        if kind not in (ACTIVATION_IDENTITY, ACTIVATION_RELU, ACTIVATION_MISH, ACTIVATION_SILU):
            raise ValueError(f"{name}: unknown activation {kind}")
        return ActivationLayerDesc(name, kind)
    return ActivationLayerDesc(name, ACTIVATION_RELU)


@dataclass
class MatMulLayerDesc:
    name: str
    in_channels: int
    out_channels: int
    weights: np.ndarray  # [ic, oc]


def parse_matmul(r: ModelReader) -> MatMulLayerDesc:
    name = r.token()
    ic = r.int()
    oc = r.int()
    if ic <= 0 or oc <= 0:
        raise ValueError(f"{name}: number of in and out channels must be positive")
    # Model file order is ic,oc
    w = r.floats(ic * oc, name).reshape(ic, oc)
    return MatMulLayerDesc(name, ic, oc, w)


@dataclass
class MatBiasLayerDesc:
    name: str
    num_channels: int
    weights: np.ndarray


def parse_matbias(r: ModelReader) -> MatBiasLayerDesc:
    name = r.token()
    c = r.int()
    if c <= 0:
        raise ValueError(f"{name}: number of channels must be positive")
    return MatBiasLayerDesc(name, c, r.floats(c, name))


def _check(cond: bool, msg: str):
    if not cond:
        raise ValueError(msg)


@dataclass
class ResidualBlockDesc:
    name: str
    pre_bn: BatchNormLayerDesc
    pre_activation: ActivationLayerDesc
    regular_conv: ConvLayerDesc
    mid_bn: BatchNormLayerDesc
    mid_activation: ActivationLayerDesc
    final_conv: ConvLayerDesc

    @property
    def in_channels(self):
        return self.pre_bn.num_channels

    @property
    def out_channels(self):
        return self.final_conv.out_channels


def parse_residual_block(r: ModelReader, model_version: int) -> ResidualBlockDesc:
    name = r.token()
    b = ResidualBlockDesc(
        name,
        parse_bn(r), parse_activation(r, model_version),
        parse_conv(r),
        parse_bn(r), parse_activation(r, model_version),
        parse_conv(r),
    )
    _check(b.pre_bn.num_channels == b.regular_conv.in_channels, f"{name}: preBN.numChannels != regularConv.inChannels")
    _check(b.mid_bn.num_channels == b.regular_conv.out_channels, f"{name}: midBN.numChannels != regularConv.outChannels")
    _check(b.mid_bn.num_channels == b.final_conv.in_channels, f"{name}: midBN.numChannels != finalConv.inChannels")
    return b


@dataclass
class GlobalPoolingResidualBlockDesc:
    name: str
    model_version: int
    pre_bn: BatchNormLayerDesc
    pre_activation: ActivationLayerDesc
    regular_conv: ConvLayerDesc
    gpool_conv: ConvLayerDesc
    gpool_bn: BatchNormLayerDesc
    gpool_activation: ActivationLayerDesc
    gpool_to_bias_mul: MatMulLayerDesc
    mid_bn: BatchNormLayerDesc
    mid_activation: ActivationLayerDesc
    final_conv: ConvLayerDesc

    @property
    def in_channels(self):
        return self.pre_bn.num_channels

    @property
    def out_channels(self):
        return self.final_conv.out_channels


def parse_gpool_block(r: ModelReader, model_version: int) -> GlobalPoolingResidualBlockDesc:
    name = r.token()
    pre_bn = parse_bn(r)
    pre_act = parse_activation(r, model_version)
    regular_conv = parse_conv(r)
    gpool_conv = parse_conv(r)
    gpool_bn = parse_bn(r)
    gpool_act = parse_activation(r, model_version)
    gpool_to_bias_mul = parse_matmul(r)
    mid_bn = parse_bn(r)
    mid_act = parse_activation(r, model_version)
    final_conv = parse_conv(r)
    b = GlobalPoolingResidualBlockDesc(
        name, model_version, pre_bn, pre_act, regular_conv, gpool_conv, gpool_bn, gpool_act,
        gpool_to_bias_mul, mid_bn, mid_act, final_conv)
    _check(pre_bn.num_channels == regular_conv.in_channels, f"{name}: preBN.numChannels != regularConv.inChannels")
    _check(pre_bn.num_channels == gpool_conv.in_channels, f"{name}: preBN.numChannels != gpoolConv.inChannels")
    _check(gpool_bn.num_channels == gpool_conv.out_channels, f"{name}: gpoolBN.numChannels != gpoolConv.outChannels")
    _check(gpool_bn.num_channels * 3 == gpool_to_bias_mul.in_channels, f"{name}: gpoolBN.numChannels*3 != gpoolToBiasMul.inChannels")
    _check(mid_bn.num_channels == regular_conv.out_channels, f"{name}: midBN.numChannels != regularConv.outChannels")
    _check(mid_bn.num_channels == gpool_to_bias_mul.out_channels, f"{name}: midBN.numChannels != gpoolToBiasMul.outChannels")
    _check(mid_bn.num_channels == final_conv.in_channels, f"{name}: midBN.numChannels != finalConv.inChannels")
    return b


@dataclass
class NestedBottleneckResidualBlockDesc:
    name: str
    num_blocks: int
    pre_bn: BatchNormLayerDesc
    pre_activation: ActivationLayerDesc
    pre_conv: ConvLayerDesc
    blocks: list
    post_bn: BatchNormLayerDesc
    post_activation: ActivationLayerDesc
    post_conv: ConvLayerDesc

    @property
    def in_channels(self):
        return self.pre_bn.num_channels

    @property
    def out_channels(self):
        return self.post_conv.out_channels


def parse_nested_bottleneck_block(r: ModelReader, model_version: int) -> NestedBottleneckResidualBlockDesc:
    name = r.token()
    num_blocks = r.int()
    _check(num_blocks >= 1, f"{name}: nested bottleneck res block num blocks must be positive")
    pre_bn = parse_bn(r)
    pre_act = parse_activation(r, model_version)
    pre_conv = parse_conv(r)
    blocks = parse_residual_block_stack(r, model_version, name, num_blocks, pre_conv.out_channels)
    post_bn = parse_bn(r)
    post_act = parse_activation(r, model_version)
    post_conv = parse_conv(r)
    _check(pre_bn.num_channels == pre_conv.in_channels, f"{name}: preBN.numChannels != preConv.inChannels")
    _check(post_bn.num_channels == pre_conv.out_channels, f"{name}: postBN.numChannels != preConv.outChannels")
    _check(post_bn.num_channels == post_conv.in_channels, f"{name}: postBN.numChannels != postConv.inChannels")
    return NestedBottleneckResidualBlockDesc(
        name, num_blocks, pre_bn, pre_act, pre_conv, blocks, post_bn, post_act, post_conv)


@dataclass
class RMSNormLayerDesc:
    """Trunk-tip RMSNorm (model version 17, TRUNK_NORM_KIND_RMSNORM): gamma/beta, spatial or per-position."""
    name: str
    num_channels: int
    epsilon: float
    spatial: bool
    cgroup_size: int
    gamma: np.ndarray
    beta: np.ndarray


def parse_rmsnorm(r: ModelReader) -> RMSNormLayerDesc:
    name = r.token()
    c = r.int()
    eps = r.float()
    spatial = r.int() != 0
    cgroup_size = r.int()
    _check(eps > 0 and eps <= 1.0, f"{name}: rmsnorm epsilon ({eps}) is not positive or is too large")
    _check(c >= 1, f"{name}: rmsnorm numChannels ({c}) < 1")
    _check(cgroup_size == 0, f"{name}: rmsnorm cgroupSize ({cgroup_size}) != 0, grouped spatial RMSNorm is not supported")
    gamma = r.floats(c, name)
    beta = r.floats(c, name)
    return RMSNormLayerDesc(name, c, eps, spatial, cgroup_size, gamma, beta)


@dataclass
class TransformerRMSNormDesc:
    """Weight-only RMSNorm over channels per position, used inside transformer blocks."""
    name: str
    num_channels: int
    epsilon: float
    weight: np.ndarray


def parse_transformer_rmsnorm(r: ModelReader) -> TransformerRMSNormDesc:
    name = r.token()
    c = r.int()
    eps = r.float()
    _check(c >= 1, f"{name}: transformer rmsnorm numChannels ({c}) < 1")
    _check(eps > 0 and eps <= 1.0, f"{name}: transformer rmsnorm epsilon ({eps}) is not positive or is too large")
    return TransformerRMSNormDesc(name, c, eps, r.floats(c, name))


@dataclass
class TransformerAttentionDesc:
    name: str
    num_heads: int
    num_kv_heads: int
    q_head_dim: int
    v_head_dim: int
    use_rope: bool
    learnable_rope: bool
    pre_ln: TransformerRMSNormDesc
    q_proj: MatMulLayerDesc
    k_proj: MatMulLayerDesc
    v_proj: MatMulLayerDesc
    out_proj: MatMulLayerDesc
    rope_freqs: Optional[np.ndarray]  # learnable RoPE: [numKVHeads, numPairs, 2] (freqX, freqY)
    rope_theta: float                 # fixed RoPE

    @property
    def in_channels(self):
        return self.q_proj.in_channels

    @property
    def out_channels(self):
        return self.out_proj.out_channels

    def rope_cos_sin(self, nn_x_len: int, nn_y_len: int) -> Tuple[np.ndarray, np.ndarray]:
        """TransformerAttentionDesc::computeRopeCosSin (desc.cpp), float32 like the C++ code.

        Returns (cos, sin) tables shaped [numKVHeads, numPairs, nnY*nnX] for learnable RoPE,
        [numPairs, nnY*nnX] for fixed RoPE; position index xy = y*nnXLen + x.
        cosf/sinf/powf are evaluated in float64 and rounded to float32 (= correctly rounded float results).
        """
        f32 = np.float32
        num_pairs = self.q_head_dim // 2
        ys, xs = np.meshgrid(np.arange(nn_y_len), np.arange(nn_x_len), indexing="ij")
        xf = xs.reshape(-1).astype(np.float32)
        yf = ys.reshape(-1).astype(np.float32)

        def cos_sin(angle):  # angle float32
            a = angle.astype(np.float64)
            return np.cos(a).astype(np.float32), np.sin(a).astype(np.float32)

        if self.learnable_rope:
            freqs = self.rope_freqs.reshape(self.num_kv_heads, num_pairs, 2)
            fx = freqs[:, :, 0:1].astype(np.float32)
            fy = freqs[:, :, 1:2].astype(np.float32)
            # angle = (float)x * freqX + (float)y * freqY, all in float32
            angle = (xf[None, None, :] * fx).astype(np.float32) + (yf[None, None, :] * fy).astype(np.float32)
            return cos_sin(angle.astype(np.float32))
        pairs_per_dim = num_pairs // 2
        dim_half = self.q_head_dim // 2
        cos = np.zeros((num_pairs, nn_x_len * nn_y_len), np.float32)
        sin = np.zeros_like(cos)
        for p in range(num_pairs):
            p_adj = p if p < pairs_per_dim else p - pairs_per_dim
            expo = f32(f32(2 * p_adj) / f32(dim_half))
            powv = f32(np.power(np.float64(f32(self.rope_theta)), np.float64(expo)))
            freq = f32(f32(1.0) / powv)
            pos = yf if p < pairs_per_dim else xf  # first numPairs/2 pairs: height (y), the rest: width (x)
            cos[p], sin[p] = cos_sin((pos * freq).astype(np.float32))
        return cos, sin


def parse_transformer_attention(r: ModelReader) -> TransformerAttentionDesc:
    name = r.token()
    num_heads, num_kv_heads, q_head_dim, v_head_dim = (r.int() for _ in range(4))
    use_rope = r.int() != 0
    learnable_rope = r.int() != 0
    _check(num_heads >= 1 and num_kv_heads >= 1, f"{name}: transformer attention numHeads and numKVHeads must be positive")
    _check(num_heads % num_kv_heads == 0, f"{name}: numHeads must be divisible by numKVHeads")
    _check(q_head_dim >= 1 and v_head_dim >= 1, f"{name}: head dims must be positive")
    _check(not use_rope or q_head_dim % 2 == 0, f"{name}: qHeadDim ({q_head_dim}) must be even when RoPE is used")
    pre_ln = parse_transformer_rmsnorm(r)
    q_proj = parse_matmul(r)
    k_proj = parse_matmul(r)
    v_proj = parse_matmul(r)
    out_proj = parse_matmul(r)
    _check(q_proj.out_channels == num_heads * q_head_dim, f"{name}: qProj.outChannels != numHeads*qHeadDim")
    _check(k_proj.out_channels == num_kv_heads * q_head_dim, f"{name}: kProj.outChannels != numKVHeads*qHeadDim")
    _check(v_proj.out_channels == num_kv_heads * v_head_dim, f"{name}: vProj.outChannels != numKVHeads*vHeadDim")
    _check(out_proj.in_channels == num_heads * v_head_dim, f"{name}: outProj.inChannels != numHeads*vHeadDim")
    rope_freqs = None
    rope_theta = 0.0
    if use_rope:
        if learnable_rope:
            _freqs_name = r.token()
            rope_num_kv_heads, rope_num_pairs, rope_dim2 = r.int(), r.int(), r.int()
            _check(rope_num_kv_heads == num_kv_heads, f"{name}: ropeNumKVHeads ({rope_num_kv_heads}) != numKVHeads ({num_kv_heads})")
            _check(rope_num_pairs == q_head_dim // 2, f"{name}: ropeNumPairs ({rope_num_pairs}) != qHeadDim/2")
            _check(rope_dim2 == 2, f"{name}: rope freq dim2 must be 2")
            rope_freqs = r.floats(rope_num_kv_heads * rope_num_pairs * 2, name).reshape(rope_num_kv_heads, rope_num_pairs, 2)
        else:
            _theta_name = r.token()
            rope_theta = r.float()
            _check(rope_theta > 0 and np.isfinite(rope_theta), f"{name}: rope theta must be positive and finite")
    return TransformerAttentionDesc(name, num_heads, num_kv_heads, q_head_dim, v_head_dim, use_rope, learnable_rope,
                                    pre_ln, q_proj, k_proj, v_proj, out_proj, rope_freqs, rope_theta)


@dataclass
class TransformerFFNDesc:
    name: str
    num_channels: int
    ffn_channels: int
    use_swiglu: bool
    pre_ln: TransformerRMSNormDesc
    linear1: MatMulLayerDesc
    linear_gate: Optional[MatMulLayerDesc]
    linear2: MatMulLayerDesc

    @property
    def in_channels(self):
        return self.num_channels

    @property
    def out_channels(self):
        return self.linear2.out_channels


def parse_transformer_ffn(r: ModelReader) -> TransformerFFNDesc:
    name = r.token()
    c = r.int()
    ffn_c = r.int()
    use_swiglu = r.int() != 0
    _check(c >= 1 and ffn_c >= 1, f"{name}: transformer ffn channels must be positive")
    pre_ln = parse_transformer_rmsnorm(r)
    linear1 = parse_matmul(r)
    linear_gate = parse_matmul(r) if use_swiglu else None
    linear2 = parse_matmul(r)
    _check(linear1.in_channels == c, f"{name}: linear1.inChannels != numChannels")
    _check(linear1.out_channels == ffn_c, f"{name}: linear1.outChannels != ffnChannels")
    if use_swiglu:
        _check(linear_gate.in_channels == c, f"{name}: linearGate.inChannels != numChannels")
        _check(linear_gate.out_channels == ffn_c, f"{name}: linearGate.outChannels != ffnChannels")
    _check(linear2.in_channels == ffn_c, f"{name}: linear2.inChannels != ffnChannels")
    _check(linear2.out_channels == c, f"{name}: linear2.outChannels != numChannels")
    return TransformerFFNDesc(name, c, ffn_c, use_swiglu, pre_ln, linear1, linear_gate, linear2)


BlockDesc = Union[ResidualBlockDesc, GlobalPoolingResidualBlockDesc, NestedBottleneckResidualBlockDesc,
                  TransformerAttentionDesc, TransformerFFNDesc]
TRANSFORMER_BLOCKS = (TransformerAttentionDesc, TransformerFFNDesc)


def blocks_contain_transformer(blocks: List[BlockDesc]) -> bool:
    """blocksContainTransformerRecursive (desc.cpp)."""
    for b in blocks:
        if isinstance(b, TRANSFORMER_BLOCKS):
            return True
        if isinstance(b, NestedBottleneckResidualBlockDesc) and blocks_contain_transformer(b.blocks):
            return True
    return False


def parse_residual_block_stack(r: ModelReader, model_version: int, name: str,
                               num_blocks: int, trunk_num_channels: int) -> List[BlockDesc]:
    blocks: List[BlockDesc] = []
    for _ in range(num_blocks):
        kind = r.token()
        if kind == "ordinary_block":
            b = parse_residual_block(r, model_version)
        elif kind == "gpool_block":
            b = parse_gpool_block(r, model_version)
        elif kind == "nested_bottleneck_block":
            b = parse_nested_bottleneck_block(r, model_version)
        elif kind == "transformer_attention_block":
            b = parse_transformer_attention(r)
        elif kind == "transformer_ffn_block":
            b = parse_transformer_ffn(r)
        else:
            raise ValueError(f"{name}: found unknown block kind {kind}")
        _check(b.in_channels == trunk_num_channels,
               f"{name}: {b.name} input channels {b.in_channels} != trunk channels {trunk_num_channels}")
        _check(b.out_channels == trunk_num_channels,
               f"{name}: {b.name} output channels {b.out_channels} != trunk channels {trunk_num_channels}")
        blocks.append(b)
    return blocks


@dataclass
class SGFMetadataEncoderDesc:
    name: str
    meta_encoder_version: int
    num_input_meta_channels: int
    mul1: MatMulLayerDesc
    bias1: MatBiasLayerDesc
    act1: ActivationLayerDesc
    mul2: MatMulLayerDesc
    bias2: MatBiasLayerDesc
    act2: ActivationLayerDesc
    mul3: MatMulLayerDesc


def num_input_meta_channels_for(meta_encoder_version: int) -> int:
    """NNModelVersion::getNumInputMetaChannels (modelversion.cpp)."""
    if meta_encoder_version == 0:
        return 0
    if meta_encoder_version == 1:
        return 192
    raise ValueError(f"Unknown meta encoder version {meta_encoder_version}")


def parse_sgf_metadata_encoder(r: ModelReader, model_version: int, meta_encoder_version: int) -> SGFMetadataEncoderDesc:
    """SGFMetadataEncoderDesc::SGFMetadataEncoderDesc(istream&, modelVersion, metaEncVersion, binaryFloats)."""
    name = r.token()
    num_input_meta_channels = r.int()
    expected = num_input_meta_channels_for(meta_encoder_version)
    _check(num_input_meta_channels == expected,
           f"{name}: number of in channels ({num_input_meta_channels}) did not match expected ({expected})")
    mul1 = parse_matmul(r)
    bias1 = parse_matbias(r)
    act1 = parse_activation(r, model_version)
    mul2 = parse_matmul(r)
    bias2 = parse_matbias(r)
    act2 = parse_activation(r, model_version)
    mul3 = parse_matmul(r)
    _check(mul1.out_channels == bias1.num_channels,
           f"{name}: mul1.outChannels ({mul1.out_channels}) != bias1.numChannels ({bias1.num_channels})")
    _check(mul2.in_channels == mul1.out_channels,
           f"{name}: mul2.inChannels ({mul2.in_channels}) != mul1.outChannels ({mul1.out_channels})")
    _check(mul2.out_channels == bias2.num_channels,
           f"{name}: mul2.outChannels ({mul2.out_channels}) != bias2.numChannels ({bias2.num_channels})")
    _check(mul2.out_channels == mul3.in_channels,
           f"{name}: mul2.outChannels ({mul2.out_channels}) != mul3.inChannels ({mul3.in_channels})")
    return SGFMetadataEncoderDesc(name, meta_encoder_version, num_input_meta_channels,
                                  mul1, bias1, act1, mul2, bias2, act2, mul3)


@dataclass
class TrunkDesc:
    name: str
    model_version: int
    num_blocks: int
    trunk_num_channels: int
    mid_num_channels: int
    regular_num_channels: int
    gpool_num_channels: int
    meta_encoder_version: int
    initial_conv: ConvLayerDesc
    initial_matmul: MatMulLayerDesc
    sgf_metadata_encoder: Optional[SGFMetadataEncoderDesc]
    blocks: List[BlockDesc]
    trunk_tip_bn: Optional[BatchNormLayerDesc]          # TRUNK_NORM_KIND_STANDARD
    trunk_tip_activation: ActivationLayerDesc
    trunk_norm_kind: int = 0                            # TRUNK_NORM_KIND_* (model version >= 15 header)
    trunk_tip_rmsnorm: Optional[RMSNormLayerDesc] = None  # TRUNK_NORM_KIND_RMSNORM


def parse_trunk(r: ModelReader, model_version: int, meta_encoder_version: int) -> TrunkDesc:
    name = r.token()
    num_blocks = r.int()
    trunk_c = r.int()
    mid_c = r.int()
    regular_c = r.int()
    _dilated_c = r.int()  # unused
    gpool_c = r.int()
    trunk_norm_kind = TRUNK_NORM_KIND_STANDARD
    if model_version >= 15:
        trunk_norm_kind = r.int()
        for opt in "BCDEF":
            v = r.int()
            _check(v == 0, f"{name}: unknown/unsupported trunk option {opt}: {v}")
        _check(trunk_norm_kind in (TRUNK_NORM_KIND_STANDARD, TRUNK_NORM_KIND_RMSNORM),
               f"{name}: unknown or unsupported trunk norm kind: {trunk_norm_kind}")
    _check(num_blocks >= 1, f"{name}: trunk num blocks must be positive")
    _check(min(trunk_c, mid_c, regular_c, gpool_c) > 0, f"{name}: channel counts must be positive")

    initial_conv = parse_conv(r)
    _check(initial_conv.out_channels == trunk_c, f"{initial_conv.name}: outChannels != trunkNumChannels")
    initial_matmul = parse_matmul(r)
    _check(initial_matmul.out_channels == trunk_c, f"{initial_matmul.name}: outChannels != trunkNumChannels")

    sgf_metadata_encoder = None
    if meta_encoder_version > 0:
        sgf_metadata_encoder = parse_sgf_metadata_encoder(r, model_version, meta_encoder_version)
        num_input_meta_channels = num_input_meta_channels_for(meta_encoder_version)
        _check(num_input_meta_channels == sgf_metadata_encoder.mul1.in_channels,
               f"{name}: {sgf_metadata_encoder.name} sgfMetadataEncoder.mul1.inChannels "
               f"({sgf_metadata_encoder.mul1.in_channels}) != numInputMetaChannels ({num_input_meta_channels})")
        _check(sgf_metadata_encoder.mul3.out_channels == trunk_c,
               f"{name}: {sgf_metadata_encoder.name} sgfMetadataEncoder.mul3.outChannels "
               f"({sgf_metadata_encoder.mul3.out_channels}) != trunkNumChannels ({trunk_c})")

    blocks = parse_residual_block_stack(r, model_version, name, num_blocks, trunk_c)
    tip_bn = tip_rms = None
    if trunk_norm_kind == TRUNK_NORM_KIND_STANDARD:
        tip_bn = parse_bn(r)
        _check(tip_bn.num_channels == trunk_c, f"{name}: trunkTipBN.numChannels != trunkNumChannels")
    else:
        tip_rms = parse_rmsnorm(r)
        _check(tip_rms.num_channels == trunk_c, f"{name}: trunkTipRMSNorm.numChannels != trunkNumChannels")
    tip_act = parse_activation(r, model_version)
    return TrunkDesc(name, model_version, num_blocks, trunk_c, mid_c, regular_c, gpool_c,
                     meta_encoder_version, initial_conv, initial_matmul, sgf_metadata_encoder, blocks, tip_bn, tip_act,
                     trunk_norm_kind, tip_rms)


@dataclass
class PolicyHeadDesc:
    name: str
    model_version: int
    policy_out_channels: int
    p1_conv: ConvLayerDesc
    g1_conv: ConvLayerDesc
    g1_bn: BatchNormLayerDesc
    g1_activation: ActivationLayerDesc
    gpool_to_bias_mul: MatMulLayerDesc
    p1_bn: BatchNormLayerDesc
    p1_activation: ActivationLayerDesc
    p2_conv: ConvLayerDesc
    gpool_to_pass_mul: MatMulLayerDesc
    gpool_to_pass_bias: Optional[MatBiasLayerDesc]
    pass_activation: Optional[ActivationLayerDesc]
    gpool_to_pass_mul2: Optional[MatMulLayerDesc]


def parse_policy_head(r: ModelReader, model_version: int) -> PolicyHeadDesc:
    name = r.token()
    if model_version >= 17:
        # v17: 2 (optimistic policy) or 4 (+ q winloss / q score) channels, stored in the file
        policy_out_channels = r.int()
        _check(policy_out_channels in (2, 4), f"{name}: policy head got invalid policyOutChannels {policy_out_channels}")
        for opt in "ABC":
            v = r.int()
            _check(v == 0, f"{name}: unknown/unsupported policy option {opt}: {v}")
    elif model_version == 16:
        policy_out_channels = 4
    elif model_version >= 12:
        policy_out_channels = 2
    else:
        policy_out_channels = 1
    p1_conv = parse_conv(r)
    g1_conv = parse_conv(r)
    g1_bn = parse_bn(r)
    g1_act = parse_activation(r, model_version)
    gpool_to_bias_mul = parse_matmul(r)
    p1_bn = parse_bn(r)
    p1_act = parse_activation(r, model_version)
    p2_conv = parse_conv(r)
    gpool_to_pass_mul = parse_matmul(r)
    if model_version >= 15:
        gpool_to_pass_bias = parse_matbias(r)
        pass_act = parse_activation(r, model_version)
        gpool_to_pass_mul2 = parse_matmul(r)
    else:
        gpool_to_pass_bias = pass_act = gpool_to_pass_mul2 = None

    _check(p1_conv.out_channels == p1_bn.num_channels, f"{name}: p1Conv.outChannels != p1BN.numChannels")
    _check(g1_conv.out_channels == g1_bn.num_channels, f"{name}: g1Conv.outChannels != g1BN.numChannels")
    _check(gpool_to_bias_mul.in_channels == g1_bn.num_channels * 3, f"{name}: gpoolToBiasMul.inChannels != g1BN.numChannels*3")
    _check(gpool_to_bias_mul.out_channels == p1_bn.num_channels, f"{name}: gpoolToBiasMul.outChannels != p1BN.numChannels")
    _check(p2_conv.in_channels == p1_bn.num_channels, f"{name}: p2Conv.inChannels != p1BN.numChannels")
    _check(gpool_to_pass_mul.in_channels == g1_bn.num_channels * 3, f"{name}: gpoolToPassMul.inChannels != g1BN.numChannels*3")
    _check(p2_conv.out_channels == policy_out_channels, f"{name}: p2Conv.outChannels != policyOutChannels")
    if model_version >= 15:
        _check(gpool_to_pass_mul.out_channels == gpool_to_pass_bias.num_channels, f"{name}: gpoolToPassMul/Bias mismatch")
        _check(gpool_to_pass_mul.out_channels == gpool_to_pass_mul2.in_channels, f"{name}: gpoolToPassMul/Mul2 mismatch")
        _check(gpool_to_pass_mul.out_channels == p1_conv.out_channels, f"{name}: gpoolToPassMul.outChannels != p1Conv.outChannels")
        _check(gpool_to_pass_mul2.out_channels == policy_out_channels, f"{name}: gpoolToPassMul2.outChannels != policyOutChannels")
    else:
        _check(gpool_to_pass_mul.out_channels == policy_out_channels, f"{name}: gpoolToPassMul.outChannels != policyOutChannels")

    return PolicyHeadDesc(name, model_version, policy_out_channels, p1_conv, g1_conv, g1_bn, g1_act,
                          gpool_to_bias_mul, p1_bn, p1_act, p2_conv, gpool_to_pass_mul,
                          gpool_to_pass_bias, pass_act, gpool_to_pass_mul2)


@dataclass
class ValueHeadDesc:
    name: str
    model_version: int
    v1_conv: ConvLayerDesc
    v1_bn: BatchNormLayerDesc
    v1_activation: ActivationLayerDesc
    v2_mul: MatMulLayerDesc
    v2_bias: MatBiasLayerDesc
    v2_activation: ActivationLayerDesc
    v3_mul: MatMulLayerDesc
    v3_bias: MatBiasLayerDesc
    sv3_mul: MatMulLayerDesc
    sv3_bias: MatBiasLayerDesc
    v_ownership_conv: ConvLayerDesc


def parse_value_head(r: ModelReader, model_version: int) -> ValueHeadDesc:
    name = r.token()
    if model_version >= 17:
        for opt in "ABC":
            v = r.int()
            _check(v == 0, f"{name}: unknown/unsupported value option {opt}: {v}")
    v = ValueHeadDesc(
        name, model_version,
        parse_conv(r), parse_bn(r), parse_activation(r, model_version),
        parse_matmul(r), parse_matbias(r), parse_activation(r, model_version),
        parse_matmul(r), parse_matbias(r),
        parse_matmul(r), parse_matbias(r),
        parse_conv(r),
    )
    _check(v.v1_conv.out_channels == v.v1_bn.num_channels, f"{name}: v1Conv.outChannels != v1BN.numChannels")
    _check(v.v2_mul.in_channels == v.v1_bn.num_channels * 3, f"{name}: v2Mul.inChannels != v1BN.numChannels*3")
    _check(v.v2_mul.out_channels == v.v2_bias.num_channels, f"{name}: v2Mul.outChannels != v2Bias.numChannels")
    _check(v.v2_mul.out_channels == v.v3_mul.in_channels, f"{name}: v2Mul.outChannels != v3Mul.inChannels")
    _check(v.v3_mul.out_channels == 3 and v.v3_bias.num_channels == 3, f"{name}: v3 must have 3 outputs")
    _check(v.sv3_mul.in_channels == v.v2_mul.out_channels, f"{name}: sv3Mul.inChannels != v2Mul.outChannels")
    if model_version >= 9:
        expected_sv = 6
    elif model_version >= 8:
        expected_sv = 4
    elif model_version >= 4:
        expected_sv = 2
    else:
        expected_sv = 1
    _check(v.sv3_mul.out_channels == expected_sv and v.sv3_bias.num_channels == expected_sv,
           f"{name}: sv3 must have {expected_sv} outputs for model version {model_version}")
    _check(v.v_ownership_conv.in_channels == v.v1_conv.out_channels, f"{name}: vOwnershipConv.inChannels != v1Conv.outChannels")
    _check(v.v_ownership_conv.out_channels == 1, f"{name}: vOwnershipConv.outChannels != 1")
    return v


@dataclass
class ModelPostProcessParams:
    # Defaults from desc.cpp; read from the header for modelVersion >= 13.
    td_score_multiplier: float = 20.0
    score_mean_multiplier: float = 20.0
    score_stdev_multiplier: float = 20.0
    lead_multiplier: float = 20.0
    variance_time_multiplier: float = 40.0
    shortterm_value_error_multiplier: float = 0.25
    shortterm_score_error_multiplier: float = 30.0
    output_scale_multiplier: float = 1.0
    from_header: bool = False


@dataclass
class ModelDesc:
    name: str
    model_version: int
    num_input_channels: int
    num_input_global_channels: int
    num_input_meta_channels: int
    meta_encoder_version: int
    post_process_params: ModelPostProcessParams
    trunk: TrunkDesc
    policy_head: PolicyHeadDesc
    value_head: ValueHeadDesc
    # Model header options (desc.cpp ModelDesc, read for modelVersion >= 15; always 0 before v17 in practice)
    prefer_pass_alive_under_suicide_rules: bool = False
    prefer_exclude_territory_adjacent_to_atari: bool = False

    @property
    def has_any_transformer_blocks(self):
        return blocks_contain_transformer(self.trunk.blocks)

    @property
    def num_policy_channels(self):
        return self.policy_head.policy_out_channels

    @property
    def num_value_channels(self):
        return self.value_head.v3_mul.out_channels

    @property
    def num_score_value_channels(self):
        return self.value_head.sv3_mul.out_channels

    @property
    def num_ownership_channels(self):
        return self.value_head.v_ownership_conv.out_channels


def check_name_valid(name: str):
    """ModelDesc::checkNameValid (desc.cpp)."""
    _check(0 < len(name) <= 96, f"Model name must be 1..96 chars: {name!r}")
    _check(all(c.isascii() and (c.isalnum() or c in "_-") for c in name),
           f"Model name must contain only alphanumeric characters, underscores, and hyphens: {name}")


def parse_model(r: ModelReader) -> ModelDesc:
    name = r.token()
    model_version = r.int()
    check_name_valid(name)
    _check(model_version >= OLDEST_MODEL_VERSION_IMPLEMENTED,
           f"{name}: model version {model_version} is too old")
    _check(model_version <= LATEST_MODEL_VERSION_IMPLEMENTED,
           f"{name}: model version {model_version} is newer than supported ({LATEST_MODEL_VERSION_IMPLEMENTED})")
    num_input_channels = r.int()
    _check(num_input_channels > 0, f"{name}: numInputChannels must be positive")
    num_input_global_channels = r.int()
    _check(num_input_global_channels > 0, f"{name}: numInputGlobalChannels must be positive")

    pp = ModelPostProcessParams()
    if model_version >= 13:
        pp.td_score_multiplier = r.float()
        pp.score_mean_multiplier = r.float()
        pp.score_stdev_multiplier = r.float()
        pp.lead_multiplier = r.float()
        pp.variance_time_multiplier = r.float()
        pp.shortterm_value_error_multiplier = r.float()
        pp.shortterm_score_error_multiplier = r.float()
        pp.from_header = True
        for k, val in vars(pp).items():
            if k != "from_header":
                _check(val > 0, f"{name}: post-process param {k} must be positive")

    meta_encoder_version = 0
    num_input_meta_channels = 0
    prefer_pass_alive = prefer_exclude_atari = False
    if model_version >= 15:
        meta_encoder_version = r.int()
        _check(0 <= meta_encoder_version <= 1, f"{name}: unsupported metaEncoderVersion {meta_encoder_version}")
        num_input_meta_channels = num_input_meta_channels_for(meta_encoder_version)
        flags = []
        for flag in ("preferPassAliveUnderSuicideRules", "preferExcludeTerritoryAdjacentToAtari"):
            v = r.int()
            _check(v in (0, 1), f"{name}: model {flag} unexpected value: {v}")
            flags.append(bool(v))
        prefer_pass_alive, prefer_exclude_atari = flags
        for opt in "DEFGH":
            v = r.int()
            _check(v == 0, f"{name}: unknown/unsupported model option {opt}: {v}")

    trunk = parse_trunk(r, model_version, meta_encoder_version)
    policy_head = parse_policy_head(r, model_version)
    value_head = parse_value_head(r, model_version)

    _check(num_input_channels == trunk.initial_conv.in_channels, f"{name}: numInputChannels != initialConv.inChannels")
    _check(num_input_global_channels == trunk.initial_matmul.in_channels, f"{name}: numInputGlobalChannels != initialMatMul.inChannels")
    _check(trunk.trunk_num_channels == policy_head.p1_conv.in_channels, f"{name}: trunk channels != p1Conv.inChannels")
    _check(trunk.trunk_num_channels == policy_head.g1_conv.in_channels, f"{name}: trunk channels != g1Conv.inChannels")
    _check(trunk.trunk_num_channels == value_head.v1_conv.in_channels, f"{name}: trunk channels != v1Conv.inChannels")

    # Make sure we consumed the whole file (only trailing whitespace allowed)
    r._skip_ws()
    _check(r.pos == len(r.data), f"{name}: {len(r.data) - r.pos} unparsed trailing bytes in model file")

    return ModelDesc(name, model_version, num_input_channels, num_input_global_channels,
                     num_input_meta_channels, meta_encoder_version, pp, trunk, policy_head, value_head,
                     prefer_pass_alive, prefer_exclude_atari)


def load_model_desc(path: str) -> ModelDesc:
    lower = path.lower()
    if lower.endswith(".gz"):
        with gzip.open(path, "rb") as f:
            data = f.read()
    else:
        with open(path, "rb") as f:
            data = f.read()
    binary_floats = not (lower.endswith(".txt") or lower.endswith(".txt.gz"))
    return parse_model(ModelReader(data, binary_floats))


# ---------------------------------------------------------------------------
# ONNX graph construction (mirrors eigenbackend.cpp forward pass)
# ---------------------------------------------------------------------------

class GraphBuilder:
    def __init__(self, pos_len: int, max_pool_off_board: float):
        self.pos_len = pos_len
        self.max_pool_off_board = max_pool_off_board
        self.nodes = []
        self.initializers = []
        self._names = set()
        self._consts = {}
        self._counter = 0

    def fresh(self, base: str) -> str:
        base = base.replace("/", "_").replace(" ", "_")
        name = base
        while name in self._names:
            self._counter += 1
            name = f"{base}_{self._counter}"
        self._names.add(name)
        return name

    def init(self, base: str, arr: np.ndarray) -> str:
        name = self.fresh(base)
        self.initializers.append(numpy_helper.from_array(np.ascontiguousarray(arr), name))
        return name

    def const(self, value, dtype=np.float32) -> str:
        arr = np.array(value, dtype=dtype)
        key = (arr.dtype.str, arr.shape, arr.tobytes())
        if key not in self._consts:
            self._consts[key] = self.init("const", arr)
        return self._consts[key]

    def node(self, op: str, inputs: List[str], base: str, output: Optional[str] = None, **attrs) -> str:
        # `output` forces an exact (pre-reserved) tensor name, used for graph outputs.
        out = output if output is not None else self.fresh(base)
        self.nodes.append(helper.make_node(op, inputs, [out], name=out, **attrs))
        return out

    def rename_output(self, tensor: str, new: str) -> str:
        """Rename the (not yet consumed) output of the node producing `tensor` to a reserved graph output name."""
        for n in self.nodes:
            assert tensor not in n.input, f"{tensor} already consumed"
        for n in self.nodes:
            if tensor in n.output:
                n.output[list(n.output).index(tensor)] = new
                return new
        raise KeyError(tensor)

    # --- layers ---

    def conv(self, x: str, d: ConvLayerDesc) -> str:
        w = self.init(d.name + ".w", d.weights)
        pad_y = (d.conv_y // 2) * d.dilation_y
        pad_x = (d.conv_x // 2) * d.dilation_x
        return self.node(
            "Conv", [x, w], d.name,
            kernel_shape=[d.conv_y, d.conv_x],
            pads=[pad_y, pad_x, pad_y, pad_x],
            dilations=[d.dilation_y, d.dilation_x],
            strides=[1, 1],
        )

    def activation(self, x: str, d: ActivationLayerDesc) -> str:
        if d.activation == ACTIVATION_IDENTITY:
            return x
        if d.activation == ACTIVATION_RELU:
            return self.node("Relu", [x], d.name)
        if d.activation == ACTIVATION_MISH:
            # mish(x) = x * tanh(softplus(x)); ONNX Mish only exists from opset 18
            sp = self.node("Softplus", [x], d.name + ".softplus")
            th = self.node("Tanh", [sp], d.name + ".tanh")
            return self.node("Mul", [x, th], d.name)
        if d.activation == ACTIVATION_SILU:
            return self.silu(x, d.name)
        raise ValueError(f"Unsupported activation {d.activation}")

    def silu(self, x: str, base: str) -> str:
        """eigenbackend's SiLU, written as it is there: x / (exp(-x) + 1) (not x * sigmoid(x))."""
        neg = self.node("Mul", [x, self.const(-1.0)], base + ".neg")
        e = self.node("Exp", [neg], base + ".exp")
        den = self.node("Add", [e, self.const(1.0)], base + ".den")
        return self.node("Div", [x, den], base)

    def bn_act_mask(self, x: str, bn: BatchNormLayerDesc, act: ActivationLayerDesc, mask: str) -> str:
        """BatchNormLayer::apply in eigenbackend: act(x*mergedScale + mergedBias), zeroed off-board."""
        c = bn.num_channels
        s = self.init(bn.name + ".scale", bn.merged_scale.reshape(1, c, 1, 1))
        b = self.init(bn.name + ".bias", bn.merged_bias.reshape(1, c, 1, 1))
        y = self.node("Mul", [x, s], bn.name + ".mul")
        y = self.node("Add", [y, b], bn.name + ".add")
        y = self.activation(y, act)
        return self.node("Mul", [y, mask], bn.name + ".masked")

    def matmul(self, x: str, d: MatMulLayerDesc) -> str:
        w = self.init(d.name + ".w", d.weights)
        return self.node("MatMul", [x, w], d.name)

    def matbias(self, x: str, d: MatBiasLayerDesc) -> str:
        b = self.init(d.name + ".b", d.weights.reshape(1, d.num_channels))
        return self.node("Add", [x, b], d.name)

    def nc_to_nc11(self, x: str, c: int, base: str) -> str:
        shape = self.const([-1, c, 1, 1], np.int64)
        return self.node("Reshape", [x, shape], base)

    def flatten_hw(self, x: str, c: int, base: str) -> str:
        """[N,C,H,W] -> [N,C,H*W] so that all reductions are over a single axis."""
        shape = self.const([-1, c, self.pos_len * self.pos_len], np.int64)
        return self.node("Reshape", [x, shape], base)

    def reduce_sum_last(self, x: str, base: str) -> str:
        axes = self.const([2], np.int64)
        return self.node("ReduceSum", [x, axes], base, keepdims=0)

    # --- pooling (poolRowsGPool / poolRowsValueHead) ---

    def setup_mask(self, bin_input: str):
        starts = self.const([0], np.int64)
        ends = self.const([1], np.int64)
        axes = self.const([1], np.int64)
        self.mask = self.node("Slice", [bin_input, starts, ends, axes], "mask")  # [N,1,H,W]
        mask_flat = self.flatten_hw(self.mask, 1, "mask_flat")                  # [N,1,HW]
        self.mask_sum = self.reduce_sum_last(mask_flat, "mask_sum")             # [N,1]
        sqrt_div = self.node("Sqrt", [self.mask_sum], "mask_sum_sqrt")
        centered = self.node("Sub", [sqrt_div, self.const(14.0)], "mask_sum_sqrt_minus14")
        self.pool_scale1 = self.node("Mul", [centered, self.const(0.1)], "pool_scale1")  # (sqrt-14)*0.1
        sq = self.node("Mul", [centered, centered], "pool_sq")
        sq = self.node("Mul", [sq, self.const(0.01)], "pool_sq_scaled")
        self.pool_scale2 = self.node("Sub", [sq, self.const(0.1)], "pool_scale2")  # (sqrt-14)^2*0.01 - 0.1
        # 0 on board, large negative off board; added before max pooling:
        # mask*BIG - BIG  ==  (1-mask)*(-BIG)   (written tensor-op-scalar only)
        big = self.const(-self.max_pool_off_board)
        scaled = self.node("Mul", [mask_flat, big], "mask_scaled")
        self.max_pool_bias = self.node("Sub", [scaled, big], "mask_maxpool_bias")  # [N,1,HW]

    def gpool(self, x: str, c: int, base: str) -> str:
        """[N,C,H,W] (already masked) -> [N,3C]: mean, mean*(sqrt(area)-14)/10, max over board."""
        x = self.flatten_hw(x, c, base + ".flat")
        s = self.reduce_sum_last(x, base + ".sum")
        mean = self.node("Div", [s, self.mask_sum], base + ".mean")
        mean_scaled = self.node("Mul", [mean, self.pool_scale1], base + ".mean_scaled")
        shifted = self.node("Add", [x, self.max_pool_bias], base + ".shifted")
        mx = self.node("ReduceMax", [shifted], base + ".max", axes=[2], keepdims=0)
        return self.node("Concat", [mean, mean_scaled, mx], base + ".concat", axis=1)

    def value_pool(self, x: str, c: int, base: str) -> str:
        """[N,C,H,W] (already masked) -> [N,3C]: mean, mean*(sqrt-14)/10, mean*((sqrt-14)^2/100 - 0.1)."""
        x = self.flatten_hw(x, c, base + ".flat")
        s = self.reduce_sum_last(x, base + ".sum")
        mean = self.node("Div", [s, self.mask_sum], base + ".mean")
        m1 = self.node("Mul", [mean, self.pool_scale1], base + ".mean_scaled1")
        m2 = self.node("Mul", [mean, self.pool_scale2], base + ".mean_scaled2")
        return self.node("Concat", [mean, m1, m2], base + ".concat", axis=1)

    # --- blocks ---

    def residual_block(self, trunk: str, d: ResidualBlockDesc) -> str:
        t = self.bn_act_mask(trunk, d.pre_bn, d.pre_activation, self.mask)
        t = self.conv(t, d.regular_conv)
        t = self.bn_act_mask(t, d.mid_bn, d.mid_activation, self.mask)
        t = self.conv(t, d.final_conv)
        return self.node("Add", [trunk, t], d.name + ".residual")

    def gpool_block(self, trunk: str, d: GlobalPoolingResidualBlockDesc) -> str:
        t = self.bn_act_mask(trunk, d.pre_bn, d.pre_activation, self.mask)
        regular = self.conv(t, d.regular_conv)
        g = self.conv(t, d.gpool_conv)
        g = self.bn_act_mask(g, d.gpool_bn, d.gpool_activation, self.mask)
        pooled = self.gpool(g, d.gpool_bn.num_channels, d.name + ".gpool")
        bias = self.matmul(pooled, d.gpool_to_bias_mul)
        bias = self.nc_to_nc11(bias, d.gpool_to_bias_mul.out_channels, d.name + ".gpool_bias")
        regular = self.node("Add", [regular, bias], d.name + ".add_gpool_bias")
        regular = self.bn_act_mask(regular, d.mid_bn, d.mid_activation, self.mask)
        regular = self.conv(regular, d.final_conv)
        return self.node("Add", [trunk, regular], d.name + ".residual")

    def nested_bottleneck_block(self, trunk: str, d: NestedBottleneckResidualBlockDesc) -> str:
        t = self.bn_act_mask(trunk, d.pre_bn, d.pre_activation, self.mask)
        t = self.conv(t, d.pre_conv)
        t = self.block_stack(t, d.blocks)
        t = self.bn_act_mask(t, d.post_bn, d.post_activation, self.mask)
        t = self.conv(t, d.post_conv)
        return self.node("Add", [trunk, t], d.name + ".residual")

    def block_stack(self, x: str, blocks: List[BlockDesc]) -> str:
        # Conv blocks run on NCHW [N,C,H,W]; a run of consecutive transformer blocks runs on tokens
        # [N,S,C] (S = H*W, s = y*W + x, i.e. eigenbackend's NHWC memory layout), with one
        # NCHW <-> tokens conversion at each end of the run.
        tokens = False
        c = None
        for b in blocks:
            is_tf = isinstance(b, TRANSFORMER_BLOCKS)
            if is_tf and not tokens:
                c = b.in_channels
                x = self.nchw_to_tokens(x, c, b.name)
                tokens = True
            elif not is_tf and tokens:
                x = self.tokens_to_nchw(x, c, b.name)
                tokens = False
            if isinstance(b, ResidualBlockDesc):
                x = self.residual_block(x, b)
            elif isinstance(b, GlobalPoolingResidualBlockDesc):
                x = self.gpool_block(x, b)
            elif isinstance(b, NestedBottleneckResidualBlockDesc):
                x = self.nested_bottleneck_block(x, b)
            elif isinstance(b, TransformerAttentionDesc):
                x = self.transformer_attention_block(x, b)
            elif isinstance(b, TransformerFFNDesc):
                x = self.transformer_ffn_block(x, b)
            else:
                raise AssertionError(type(b))
        if tokens:
            x = self.tokens_to_nchw(x, c, blocks[-1].name + ".end")
        return x

    # --- transformer blocks (token layout [N,S,C]) ---

    def nchw_to_tokens(self, x: str, c: int, base: str) -> str:
        x = self.flatten_hw(x, c, base + ".to_tokens.flat")                         # [N,C,S]
        return self.node("Transpose", [x], base + ".to_tokens", perm=[0, 2, 1])     # [N,S,C]

    def tokens_to_nchw(self, x: str, c: int, base: str) -> str:
        x = self.node("Transpose", [x], base + ".to_nchw.t", perm=[0, 2, 1])        # [N,C,S]
        shape = self.const([-1, c, self.pos_len, self.pos_len], np.int64)
        return self.node("Reshape", [x, shape], base + ".to_nchw")                   # [N,C,H,W]

    def setup_token_masks(self):
        """Created on first use (transformer nets only): the mask in token layout [N,S,1] and the
        additive attention bias [N,1,1,S], 0 on board and -ATTN_MASK_BIG off board."""
        if getattr(self, "mask_tok", None) is not None:
            return
        s = self.pos_len * self.pos_len
        self.mask_tok = self.node("Reshape", [self.mask, self.const([-1, s, 1], np.int64)], "mask_tokens")
        m = self.node("Reshape", [self.mask, self.const([-1, 1, 1, s], np.int64)], "mask_keys")
        big = self.const(ATTN_MASK_BIG)
        scaled = self.node("Mul", [m, big], "mask_keys_scaled")
        self.attn_bias = self.node("Sub", [scaled, big], "attn_mask_bias")  # mask*BIG - BIG

    def matmul_tokens(self, x: str, d: MatMulLayerDesc, weights: Optional[np.ndarray] = None) -> str:
        """[N,S,in] @ [in,out] -> [N,S,out] (eigenbackend MatMulLayer applied to every position)."""
        w = self.init(d.name + ".w", d.weights if weights is None else weights)
        return self.node("MatMul", [x, w], d.name)

    def transformer_rmsnorm(self, x: str, d: TransformerRMSNormDesc) -> str:
        """TransformerRMSNormLayer: x * (1/sqrt(sum(x^2)/C + eps)) * weight per position, 0 off board."""
        sq = self.node("Mul", [x, x], d.name + ".sq")
        ss = self.node("ReduceSum", [sq, self.const([2], np.int64)], d.name + ".sumsq", keepdims=1)  # [N,S,1]
        ms = self.node("Div", [ss, self.const(float(d.num_channels))], d.name + ".meansq")
        den = self.node("Add", [ms, self.const(d.epsilon)], d.name + ".add_eps")
        den = self.node("Sqrt", [den], d.name + ".sqrt")
        rms = self.node("Div", [self.const(1.0), den], d.name + ".rms")
        y = self.node("Mul", [x, rms], d.name + ".normed")
        y = self.node("Mul", [y, self.init(d.name + ".weight", d.weight)], d.name + ".scaled")
        return self.node("Mul", [y, self.mask_tok], d.name + ".masked")

    def rope(self, x: str, cos: str, sin: str, half: int, head_dim: int, base: str) -> str:
        """Rotate pairs (x0, x1) -> (x0*cos - x1*sin, x0*sin + x1*cos), eigenbackend's applyRoPE.

        The projection weights are permuted so that each head's channels are [x0 of every pair...,
        x1 of every pair...] instead of interleaved (2p, 2p+1): the halves are two Slices on the last
        axis. Q.K dot products are invariant to that (same permutation for Q and K)."""
        axes = self.const([3], np.int64)
        x0 = self.node("Slice", [x, self.const([0], np.int64), self.const([half], np.int64), axes], base + ".x0")
        x1 = self.node("Slice", [x, self.const([half], np.int64), self.const([head_dim], np.int64), axes], base + ".x1")
        a = self.node("Mul", [x0, cos], base + ".x0cos")
        b = self.node("Mul", [x1, sin], base + ".x1sin")
        r0 = self.node("Sub", [a, b], base + ".r0")
        c = self.node("Mul", [x0, sin], base + ".x0sin")
        d = self.node("Mul", [x1, cos], base + ".x1cos")
        r1 = self.node("Add", [c, d], base + ".r1")
        return self.node("Concat", [r0, r1], base, axis=3)

    def transformer_attention_block(self, x: str, d: TransformerAttentionDesc) -> str:
        """TransformerAttentionBlock::apply. x: tokens [N,S,C]."""
        self.setup_token_masks()
        n = d.name
        s = self.pos_len * self.pos_len
        nh, nkv, hd, vhd = d.num_heads, d.num_kv_heads, d.q_head_dim, d.v_head_dim
        g = nh // nkv
        xn = self.transformer_rmsnorm(x, d.pre_ln)

        wq, wk = d.q_proj.weights, d.k_proj.weights
        if d.use_rope:
            # per head: interleaved pair channels (2p, 2p+1) -> [2p for all p] + [2p+1 for all p]
            half = hd // 2
            perm_head = np.concatenate([np.arange(0, hd, 2), np.arange(1, hd, 2)])
            wq = wq[:, (np.arange(nh)[:, None] * hd + perm_head[None, :]).reshape(-1)]
            wk = wk[:, (np.arange(nkv)[:, None] * hd + perm_head[None, :]).reshape(-1)]
        q = self.matmul_tokens(xn, d.q_proj, wq)  # [N,S,nh*hd]
        k = self.matmul_tokens(xn, d.k_proj, wk)  # [N,S,nkv*hd]
        v = self.matmul_tokens(xn, d.v_proj)      # [N,S,nkv*vhd]

        def heads(t, nheads, dim, base):  # [N,S,heads*dim] -> [N,heads,S,dim]
            t = self.node("Reshape", [t, self.const([-1, s, nheads, dim], np.int64)], base + ".split")
            return self.node("Transpose", [t], base + ".heads", perm=[0, 2, 1, 3])

        qh = heads(q, nh, hd, n + ".q")
        kh = heads(k, nkv, hd, n + ".k")
        vh = heads(v, nkv, vhd, n + ".v")

        if d.use_rope:
            cos, sin = d.rope_cos_sin(self.pos_len, self.pos_len)  # [nkv,P,S] (learnable) or [P,S] (fixed)
            if d.learnable_rope:
                cos_k = cos.transpose(0, 2, 1)[None]                 # [1,nkv,S,P]
                sin_k = sin.transpose(0, 2, 1)[None]
                kv_of_head = np.arange(nh) * nkv // nh               # eigenbackend: kvh = h*numKVHeads/numHeads
                cos_q, sin_q = cos_k[:, kv_of_head], sin_k[:, kv_of_head]  # [1,nh,S,P]
            else:
                cos_k = cos_q = cos.T[None, None]                    # [1,1,S,P]
                sin_k = sin_q = sin.T[None, None]
            cq = self.init(n + ".rope_cos_q", np.ascontiguousarray(cos_q))
            sq = self.init(n + ".rope_sin_q", np.ascontiguousarray(sin_q))
            if d.learnable_rope and nkv != nh:
                ck = self.init(n + ".rope_cos_k", np.ascontiguousarray(cos_k))
                sk = self.init(n + ".rope_sin_k", np.ascontiguousarray(sin_k))
            else:
                ck, sk = cq, sq
            qh = self.rope(qh, cq, sq, half, hd, n + ".q_rope")
            kh = self.rope(kh, ck, sk, half, hd, n + ".k_rope")

        if g > 1:
            # GQA: query heads h = kvh*g + j share kv head kvh -> view Q as [N,nkv,g*S,hd]
            qh = self.node("Reshape", [qh, self.const([-1, nkv, g * s, hd], np.int64)], n + ".q_gqa")
        kt = self.node("Transpose", [kh], n + ".k_t", perm=[0, 1, 3, 2])           # [N,nkv,hd,S]
        scores = self.node("MatMul", [qh, kt], n + ".scores")                      # [N,nkv,g*S,S]
        scale = np.float32(1.0) / np.sqrt(np.float32(hd))  # 1.0f / sqrtf((float)qHeadDim)
        scores = self.node("Mul", [scores, self.const(scale)], n + ".scores_scaled")
        scores = self.node("Add", [scores, self.attn_bias], n + ".scores_masked")  # off-board keys -> -BIG
        probs = self.node("Softmax", [scores], n + ".probs", axis=3)
        o = self.node("MatMul", [probs, vh], n + ".attn")                          # [N,nkv,g*S,vhd]
        if g > 1:
            o = self.node("Reshape", [o, self.const([-1, nh, s, vhd], np.int64)], n + ".attn_heads")
        o = self.node("Transpose", [o], n + ".attn_t", perm=[0, 2, 1, 3])         # [N,S,nh,vhd]
        o = self.node("Reshape", [o, self.const([-1, s, nh * vhd], np.int64)], n + ".attn_merged")
        o = self.matmul_tokens(o, d.out_proj)                                      # [N,S,C]
        o = self.node("Mul", [o, self.mask_tok], n + ".out_masked")
        return self.node("Add", [x, o], n + ".residual")

    def transformer_ffn_block(self, x: str, d: TransformerFFNDesc) -> str:
        """TransformerFFNBlock::apply (SwiGLU). x: tokens [N,S,C]."""
        if not d.use_swiglu:
            # Like eigenbackend (and upstream's ONNX exporter): the plain FFN has no defined activation.
            raise ValueError(f"{d.name}: non-SwiGLU transformer FFN is not supported (not supported by the Eigen backend either)")
        self.setup_token_masks()
        n = d.name
        xn = self.transformer_rmsnorm(x, d.pre_ln)
        a = self.matmul_tokens(xn, d.linear1)
        gate = self.matmul_tokens(xn, d.linear_gate)
        h = self.node("Mul", [self.silu(a, n + ".silu"), gate], n + ".swiglu")  # SiLU(linear1) * gate
        o = self.matmul_tokens(h, d.linear2)
        o = self.node("Mul", [o, self.mask_tok], n + ".out_masked")
        return self.node("Add", [x, o], n + ".residual")

    def trunk_tip_rmsnorm(self, x: str, d: RMSNormLayerDesc, act: ActivationLayerDesc) -> str:
        """RMSNormLayer::apply (trunk tip, NCHW): act(x * rms * gamma + beta), 0 off board.
        Per-position (spatial=0): rms over channels. Spatial: one rms over channels x on-board positions."""
        c = d.num_channels
        n = d.name
        sq = self.node("Mul", [x, x], n + ".sq")
        if not d.spatial:
            ss = self.node("ReduceSum", [sq, self.const([1], np.int64)], n + ".sumsq", keepdims=1)  # [N,1,H,W]
            ms = self.node("Div", [ss, self.const(float(c))], n + ".meansq")
        else:
            sq = self.node("Mul", [sq, self.mask], n + ".sq_onboard")
            flat = self.node("Reshape", [sq, self.const([-1, c * self.pos_len * self.pos_len], np.int64)], n + ".flat")
            ss = self.node("ReduceSum", [flat, self.const([1], np.int64)], n + ".sumsq", keepdims=1)  # [N,1]
            count = self.node("Mul", [self.mask_sum, self.const(float(c))], n + ".count")         # onboard * C
            ms = self.node("Div", [ss, count], n + ".meansq")
            ms = self.nc_to_nc11(ms, 1, n + ".meansq_nc11")                                       # [N,1,1,1]
        den = self.node("Add", [ms, self.const(d.epsilon)], n + ".add_eps")
        den = self.node("Sqrt", [den], n + ".sqrt")
        rms = self.node("Div", [self.const(1.0), den], n + ".rms")
        y = self.node("Mul", [x, rms], n + ".normed")
        y = self.node("Mul", [y, self.init(n + ".gamma", d.gamma.reshape(1, c, 1, 1))], n + ".scaled")
        y = self.node("Add", [y, self.init(n + ".beta", d.beta.reshape(1, c, 1, 1))], n + ".shifted")
        y = self.activation(y, act)
        return self.node("Mul", [y, self.mask], n + ".masked")

    # --- trunk and heads ---

    def sgf_metadata_encoder(self, meta_input: str, d: SGFMetadataEncoderDesc) -> str:
        """SGFMetadataEncoder::apply in eigenbackend: mul1, bias1, act1, mul2, bias2, act2, mul3."""
        m = self.matmul(meta_input, d.mul1)
        m = self.matbias(m, d.bias1)
        m = self.activation(m, d.act1)
        m = self.matmul(m, d.mul2)
        m = self.matbias(m, d.bias2)
        m = self.activation(m, d.act2)
        return self.matmul(m, d.mul3)

    def trunk(self, bin_input: str, global_input: str, meta_input: Optional[str], d: TrunkDesc) -> str:
        x = self.conv(bin_input, d.initial_conv)
        g = self.matmul(global_input, d.initial_matmul)
        g = self.nc_to_nc11(g, d.trunk_num_channels, d.initial_matmul.name + ".nc11")
        x = self.node("Add", [x, g], "trunk_input")
        if d.sgf_metadata_encoder is not None:
            # Trunk::apply: a second addNCBiasInplace with the encoder output after the global one.
            assert meta_input is not None
            m = self.sgf_metadata_encoder(meta_input, d.sgf_metadata_encoder)
            m = self.nc_to_nc11(m, d.trunk_num_channels, d.sgf_metadata_encoder.name + ".nc11")
            x = self.node("Add", [x, m], "trunk_input_meta")
        else:
            assert meta_input is None
        x = self.block_stack(x, d.blocks)
        if d.trunk_norm_kind == TRUNK_NORM_KIND_RMSNORM:
            return self.trunk_tip_rmsnorm(x, d.trunk_tip_rmsnorm, d.trunk_tip_activation)
        return self.bn_act_mask(x, d.trunk_tip_bn, d.trunk_tip_activation, self.mask)

    def policy_head(self, trunk: str, d: PolicyHeadDesc) -> str:
        p1 = self.conv(trunk, d.p1_conv)
        g1 = self.conv(trunk, d.g1_conv)
        g1 = self.bn_act_mask(g1, d.g1_bn, d.g1_activation, self.mask)
        g1_pooled = self.gpool(g1, d.g1_bn.num_channels, d.name + ".gpool")
        bias = self.matmul(g1_pooled, d.gpool_to_bias_mul)
        bias = self.nc_to_nc11(bias, d.gpool_to_bias_mul.out_channels, d.name + ".gpool_bias")
        p1 = self.node("Add", [p1, bias], d.name + ".add_gpool_bias")
        p1 = self.bn_act_mask(p1, d.p1_bn, d.p1_activation, self.mask)
        p2 = self.conv(p1, d.p2_conv)  # [N,P,H,W]
        if d.model_version >= 15:
            pas = self.matmul(g1_pooled, d.gpool_to_pass_mul)
            pas = self.matbias(pas, d.gpool_to_pass_bias)
            pas = self.activation(pas, d.pass_activation)
            pas = self.matmul(pas, d.gpool_to_pass_mul2)
        else:
            pas = self.matmul(g1_pooled, d.gpool_to_pass_mul)  # [N,P]
        p = d.policy_out_channels
        area = self.pos_len * self.pos_len
        spatial = self.node("Reshape", [p2, self.const([-1, p, area], np.int64)], d.name + ".spatial_flat")
        pas = self.node("Reshape", [pas, self.const([-1, p, 1], np.int64)], d.name + ".pass_flat")
        return self.node("Concat", [spatial, pas], "policy", output="policy", axis=2)

    def value_head(self, trunk: str, d: ValueHeadDesc) -> Tuple[str, str, str]:
        v1 = self.conv(trunk, d.v1_conv)
        v1 = self.bn_act_mask(v1, d.v1_bn, d.v1_activation, self.mask)
        pooled = self.value_pool(v1, d.v1_bn.num_channels, d.name + ".pool")
        v2 = self.matmul(pooled, d.v2_mul)
        v2 = self.matbias(v2, d.v2_bias)
        v2 = self.activation(v2, d.v2_activation)
        value = self.matbias(self.matmul(v2, d.v3_mul), d.v3_bias)
        score = self.matbias(self.matmul(v2, d.sv3_mul), d.sv3_bias)
        ownership = self.conv(v1, d.v_ownership_conv)
        value = self.rename_output(value, "value")
        score = self.rename_output(score, "scorevalue")
        ownership = self.rename_output(ownership, "ownership")
        return value, score, ownership


def build_onnx(desc: ModelDesc, pos_len: int = 19, max_pool_off_board: float = -1.0e4) -> onnx.ModelProto:
    gb = GraphBuilder(pos_len, max_pool_off_board)
    gb._names.update(["bin_input", "global_input", "meta_input", "policy", "value", "scorevalue", "ownership"])
    gb.setup_mask("bin_input")
    meta_input = "meta_input" if desc.num_input_meta_channels > 0 else None
    trunk = gb.trunk("bin_input", "global_input", meta_input, desc.trunk)
    policy = gb.policy_head(trunk, desc.policy_head)
    value, score, ownership = gb.value_head(trunk, desc.value_head)
    assert (policy, value, score, ownership) == ("policy", "value", "scorevalue", "ownership")

    f = TensorProto.FLOAT
    inputs = [
        helper.make_tensor_value_info("bin_input", f, ["N", desc.num_input_channels, pos_len, pos_len]),
        helper.make_tensor_value_info("global_input", f, ["N", desc.num_input_global_channels]),
    ]
    if meta_input is not None:
        inputs.append(helper.make_tensor_value_info("meta_input", f, ["N", desc.num_input_meta_channels]))
    outputs = [
        helper.make_tensor_value_info("policy", f, ["N", desc.num_policy_channels, pos_len * pos_len + 1]),
        helper.make_tensor_value_info("value", f, ["N", desc.num_value_channels]),
        helper.make_tensor_value_info("scorevalue", f, ["N", desc.num_score_value_channels]),
        helper.make_tensor_value_info("ownership", f, ["N", desc.num_ownership_channels, pos_len, pos_len]),
    ]
    graph = helper.make_graph(gb.nodes, desc.name.replace("-", "_"), inputs, outputs, gb.initializers)
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", OPSET)],
        producer_name="katago-rs bin2onnx",
        producer_version="0.1",
    )
    model.ir_version = 8  # ONNX IR version matching opset 17

    pp = desc.post_process_params
    t = desc.trunk
    meta = {
        "name": desc.name,
        "modelVersion": desc.model_version,
        "posLen": pos_len,
        "numInputChannels": desc.num_input_channels,
        "numInputGlobalChannels": desc.num_input_global_channels,
        "numInputMetaChannels": desc.num_input_meta_channels,
        "metaEncoderVersion": desc.meta_encoder_version,
        "numPolicyChannels": desc.num_policy_channels,
        "numValueChannels": desc.num_value_channels,
        "numScoreValueChannels": desc.num_score_value_channels,
        "numOwnershipChannels": desc.num_ownership_channels,
        "trunkNumBlocks": t.num_blocks,
        "trunkNumChannels": t.trunk_num_channels,
        "postProcessParamsFromHeader": int(pp.from_header),
        "tdScoreMultiplier": pp.td_score_multiplier,
        "scoreMeanMultiplier": pp.score_mean_multiplier,
        "scoreStdevMultiplier": pp.score_stdev_multiplier,
        "leadMultiplier": pp.lead_multiplier,
        "varianceTimeMultiplier": pp.variance_time_multiplier,
        "shorttermValueErrorMultiplier": pp.shortterm_value_error_multiplier,
        "shorttermScoreErrorMultiplier": pp.shortterm_score_error_multiplier,
        "outputScaleMultiplier": pp.output_scale_multiplier,
        "outputs": "policy=raw logits [N,P,HW+1] (last=pass); value=raw win/loss/noresult logits; "
                   "scorevalue=raw misc value outputs; ownership=pre-tanh, player-to-move perspective",
    }
    # Keys added with model version 17 support. Written only for nets that use v17-era features, so
    # that older nets' exports stay byte-identical; absent means 0.
    t_has_tf = desc.has_any_transformer_blocks
    if (desc.model_version >= 17 or t_has_tf or t.trunk_norm_kind != TRUNK_NORM_KIND_STANDARD
            or desc.prefer_pass_alive_under_suicide_rules or desc.prefer_exclude_territory_adjacent_to_atari):
        meta.update({
            "preferPassAliveUnderSuicideRules": int(desc.prefer_pass_alive_under_suicide_rules),
            "preferExcludeTerritoryAdjacentToAtari": int(desc.prefer_exclude_territory_adjacent_to_atari),
            "trunkNormKind": t.trunk_norm_kind,
            "hasAnyTransformerBlocks": int(t_has_tf),
        })
    for k, v in meta.items():
        entry = model.metadata_props.add()
        entry.key = k
        entry.value = repr(v) if isinstance(v, float) else str(v)
    return model


def main(argv=None):
    ap = argparse.ArgumentParser(description="Convert a KataGo .bin.gz/.txt.gz model to ONNX (opset 17).")
    ap.add_argument("model", help="Path to KataGo model (.bin.gz, .txt.gz, .bin, .txt)")
    ap.add_argument("-o", "--output", help="Output .onnx path (default: <model name>.onnx next to input)")
    ap.add_argument("--pos-len", type=int, default=19, help="Fixed spatial size of the graph (default 19)")
    ap.add_argument("--no-check", action="store_true", help="Skip onnx.checker validation")
    args = ap.parse_args(argv)

    desc = load_model_desc(args.model)
    model = build_onnx(desc, pos_len=args.pos_len)
    if not args.no_check:
        onnx.checker.check_model(model, full_check=True)
    out = args.output
    if out is None:
        base = os.path.basename(args.model)
        for ext in (".bin.gz", ".txt.gz", ".gz", ".bin", ".txt"):
            if base.endswith(ext):
                base = base[: -len(ext)]
                break
        out = os.path.join(os.path.dirname(os.path.abspath(args.model)), base + ".onnx")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    onnx.save(model, out)
    ops = sorted({n.op_type for n in model.graph.node})
    print(f"Wrote {out}")
    print(f"  model {desc.name} version {desc.model_version}, "
          f"{desc.trunk.num_blocks} blocks x {desc.trunk.trunk_num_channels} channels")
    print(f"  nodes {len(model.graph.node)}, initializers {len(model.graph.initializer)}")
    print(f"  ops: {' '.join(ops)}")


if __name__ == "__main__":
    main()
