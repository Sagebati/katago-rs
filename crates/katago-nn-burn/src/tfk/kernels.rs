//! The cubecl kernels of [`super`]: the token-major GEMM of the transformer linears, the per-token RMSNorm, and the
//! fused attention over the board.

use burn_cubecl::cubecl;
use burn_cubecl::cubecl::prelude::*;

/// GEMM block of [`linear_gemm`]: `LIN_M` rows x `LIN_N` output features per cube (16 inputs per shared-memory step),
/// `LIN_T` x `LIN_T` threads each computing 4 x 4 outputs.
pub(crate) const LIN_M: usize = 64;
pub(crate) const LIN_N: usize = 64;
pub(crate) const LIN_T: usize = 16;

/// `out[m, n] = sum_k x[m, k] * w[k, n] (+ bias[n])` for row-major `x` `[m_size, k_size]`, `w` `[k_size, n_size]` and
/// `out` `[m_size, n_size]`: burn's `Linear` (weight `[d_input, d_output]`) on the `[N * S, C]` tokens. Same blocking
/// as the NCHW 1x1 GEMM (`conv1x1_gemm`), with the A tile read along k (the token-major layout), the next K step
/// prefetched into registers during the current one, and every edge bounds-checked.
#[cube(launch)]
pub(crate) fn linear_gemm<F: Float>(x: &[F], w: &[F], bias: &[F], out: &mut [F], m_size: u32, n_size: u32, k_size: u32, #[comptime] has_bias: bool) {
  let m0 = CUBE_POS_X as usize * 64;
  let n0 = CUBE_POS_Y as usize * 64;
  let tx = UNIT_POS_X as usize;
  let ty = UNIT_POS_Y as usize;
  let tid = ty * 16 + tx;
  let m_size = m_size as usize;
  let n_size = n_size as usize;
  let k_size = k_size as usize;

  // A is stored k-major with rows of 65: the transposing stores of 16 consecutive k would hit one bank with 64.
  let mut a_s = Shared::<[F]>::new_slice(16usize * 65usize);
  let mut b_s = Shared::<[F]>::new_slice(16usize * 64usize);
  let mut acc = Array::<F>::new(16usize);
  #[unroll]
  for i in 0..16usize {
    acc[i] = F::new(0.0f32);
  }
  let mut a_reg = Array::<F>::new(4usize);
  let mut b_reg = Array::<F>::new(4usize);
  // The next K step's tiles, loaded from global memory while the current one is multiplied.
  let mut a_next = Array::<F>::new(4usize);
  let mut b_next = Array::<F>::new(4usize);
  // This thread's elements of the tiles: A 64 rows x 16 k read along k, B 16 k x 64 n read along n.
  let mm = tid / 16;
  let ka = tid % 16;
  let kb = tid / 64;
  let nn = tid % 64;

  #[unroll]
  for r in 0..4usize {
    let mut v = F::new(0.0f32);
    if m0 + mm + 16 * r < m_size && ka < k_size {
      v = x[(m0 + mm + 16 * r) * k_size + ka];
    }
    a_next[r] = v;
    let mut u = F::new(0.0f32);
    if n0 + nn < n_size && kb + 4 * r < k_size {
      u = w[(kb + 4 * r) * n_size + n0 + nn];
    }
    b_next[r] = u;
  }

  for k0 in range_stepped(0usize, k_size, 16usize) {
    #[unroll]
    for r in 0..4usize {
      a_s[ka * 65 + mm + 16 * r] = a_next[r];
      b_s[(kb + 4 * r) * 64 + nn] = b_next[r];
    }
    sync_cube();
    let k1 = k0 + 16;
    if k1 < k_size {
      #[unroll]
      for r in 0..4usize {
        let mut v = F::new(0.0f32);
        if m0 + mm + 16 * r < m_size && k1 + ka < k_size {
          v = x[(m0 + mm + 16 * r) * k_size + k1 + ka];
        }
        a_next[r] = v;
        let mut u = F::new(0.0f32);
        if n0 + nn < n_size && k1 + kb + 4 * r < k_size {
          u = w[(k1 + kb + 4 * r) * n_size + n0 + nn];
        }
        b_next[r] = u;
      }
    }
    #[unroll]
    for kk in 0..16usize {
      #[unroll]
      for i in 0..4usize {
        a_reg[i] = a_s[kk * 65 + ty + 16 * i];
        b_reg[i] = b_s[kk * 64 + tx + 16 * i];
      }
      #[unroll]
      for j in 0..4usize {
        #[unroll]
        for i in 0..4usize {
          acc[j * 4 + i] += a_reg[j] * b_reg[i];
        }
      }
    }
    sync_cube();
  }

  #[unroll]
  for j in 0..4usize {
    #[unroll]
    for i in 0..4usize {
      let m = m0 + ty + 16 * j;
      let n = n0 + tx + 16 * i;
      if m < m_size && n < n_size {
        let mut v = acc[j * 4 + i];
        if comptime![has_bias] {
          v += bias[n];
        }
        out[m * n_size + n] = v;
      }
    }
  }
}

/// [`linear_gemm`] with the weight and output read and written as `N`-wide vectors (`N` = 4) along the output
/// features, `n_size` a multiple of 4: each thread computes 4 rows x one vector of outputs per K step, the products of
/// each 16-input step summed in `S` and added into f32 accumulators (`wino_gemm_v`'s scheme). With `S` f16 (the f16
/// weight) the inner products use packed f16 math; `x`, `bias` and `out` stay in `F`.
#[cube(launch)]
pub(crate) fn linear_gemm_v<F: Float, S: Float, N: Size>(
  x: &[F],
  w: &[Vector<S, N>],
  bias: &[Vector<F, N>],
  out: &mut [Vector<F, N>],
  m_size: u32,
  n_size: u32,
  k_size: u32,
  #[comptime] has_bias: bool,
) {
  let m0 = CUBE_POS_X as usize * 64;
  let nv0 = CUBE_POS_Y as usize * 16;
  let tx = UNIT_POS_X as usize;
  let ty = UNIT_POS_Y as usize;
  let tid = ty * 16 + tx;
  let m_size = m_size as usize;
  let n_v = n_size as usize / 4;
  let k_size = k_size as usize;

  let mut a_s = Shared::<[S]>::new_slice(16usize * 65usize);
  let mut b_s = Shared::<[Vector<S, N>]>::new_slice(16usize * 16usize);
  let mut acc = Array::<Vector<f32, N>>::new(4usize);
  let mut part = Array::<Vector<S, N>>::new(4usize);
  #[unroll]
  for i in 0..4usize {
    acc[i] = Vector::new(0.0f32);
  }

  for k0 in range_stepped(0usize, k_size, 16usize) {
    #[unroll]
    for r in 0..4usize {
      let idx = tid + 256 * r;
      let mm = idx / 16;
      let ka = idx % 16;
      let mut v = F::new(0.0f32);
      if m0 + mm < m_size && k0 + ka < k_size {
        v = x[(m0 + mm) * k_size + k0 + ka];
      }
      a_s[ka * 65 + mm] = S::cast_from(v);
    }
    let kb = tid / 16;
    let nv = tid % 16;
    let mut u = Vector::new(S::new(0.0f32));
    if nv0 + nv < n_v && k0 + kb < k_size {
      u = w[(k0 + kb) * n_v + nv0 + nv];
    }
    b_s[tid] = u;
    sync_cube();
    #[unroll]
    for i in 0..4usize {
      part[i] = Vector::new(S::new(0.0f32));
    }
    #[unroll]
    for kk in 0..16usize {
      let bv = b_s[kk * 16 + tx];
      #[unroll]
      for i in 0..4usize {
        part[i] += Vector::new(a_s[kk * 65 + ty * 4 + i]) * bv;
      }
    }
    #[unroll]
    for i in 0..4usize {
      acc[i] += Vector::cast_from(part[i]);
    }
    sync_cube();
  }

  #[unroll]
  for i in 0..4usize {
    let m = m0 + ty * 4 + i;
    if m < m_size && nv0 + tx < n_v {
      let mut v: Vector<F, N> = Vector::cast_from(acc[i]);
      if comptime![has_bias] {
        v += bias[nv0 + tx];
      }
      out[m * n_v + nv0 + tx] = v;
    }
  }
}

/// The transformer blocks' RMSNorm of each token, as the generated graph computes it (bin2onnx): with `ss` the sum of
/// squares of the row, `out = x * (1 / sqrt(ss / divisor + eps)) * gamma * mask[row]`, over a contiguous `[rows, c]`
/// `x`. `divisor` and `eps` are 1-element buffers (the graph's constants, on the device). One 64-thread workgroup per
/// row: strided partial sums, then a shared-memory tree.
#[cube(launch)]
pub(crate) fn rms_norm_kernel<F: Float>(x: &[F], gamma: &[F], mask: &[F], divisor: &[F], eps: &[F], out: &mut [F], c: u32, rows_x: u32) {
  // x over the first 65535 rows' remainder, y over blocks of them (65535 workgroups per grid dimension).
  let row = (CUBE_POS_Y * rows_x + CUBE_POS_X) as usize;
  let tid = UNIT_POS_X as usize;
  let c = c as usize;
  let base = row * c;
  let mut acc = F::new(0.0f32);
  for i in range_stepped(tid, c, 64usize) {
    let v = x[base + i];
    acc += v * v;
  }
  let mut s = Shared::<[F]>::new_slice(64usize);
  s[tid] = acc;
  sync_cube();
  #[unroll]
  for step in 0..6u32 {
    let half = 32usize >> step as usize;
    if tid < half {
      s[tid] = s[tid] + s[tid + half];
    }
    sync_cube();
  }
  let inv = F::new(1.0f32) / F::sqrt(s[0] / divisor[0] + eps[0]);
  let m = mask[row];
  for i in range_stepped(tid, c, 64usize) {
    out[base + i] = x[base + i] * inv * gamma[i] * m;
  }
}

/// Queries per workgroup of [`attention_kernel`] (one thread each).
pub(crate) const ATT_Q: usize = 64;

/// Softmax attention over the board with an additive per-key bias (the off-board mask): one thread per query row,
/// online softmax over blocks of 32 keys staged in shared memory. `q` is a contiguous `[N, H, S, dq]`, `k` and `v`
/// `[N, Hkv, S, dq]` and `[N, Hkv, S, dv]`, `groups = H / Hkv` query heads sharing a key/value head (GQA), `bias` the
/// `[N, S]` additive key bias. The output is a contiguous `[N, H, S, dv]`; a query row with every key at -inf gets 0.
#[cube(launch)]
pub(crate) fn attention_kernel<F: Float>(
  q: &[F],
  k: &[F],
  v: &[F],
  bias: &[F],
  out: &mut [F],
  seq: u32,
  heads: u32,
  groups: u32,
  scale: f32,
  #[comptime] dq: usize,
  #[comptime] dv: usize,
) {
  let qi = CUBE_POS_X as usize * 64 + UNIT_POS_X as usize;
  let h = CUBE_POS_Y as usize;
  let n = CUBE_POS_Z as usize;
  let tid = UNIT_POS_X as usize;
  let seq = seq as usize;
  let kvh = h / groups as usize;
  let in_range = qi < seq;
  let qrow = if in_range { qi } else { 0usize };

  let mut qv = Array::<F>::new(dq);
  let q_base = ((n * heads as usize + h) * seq + qrow) * dq;
  #[unroll]
  for d in 0..dq {
    qv[d] = q[q_base + d] * F::cast_from(scale);
  }
  let mut acc = Array::<F>::new(dv);
  #[unroll]
  for d in 0..dv {
    acc[d] = F::new(0.0f32);
  }
  let mut m_run = F::new(-3.0e38f32);
  let mut l_run = F::new(0.0f32);

  let mut k_s = Shared::<[F]>::new_slice(32usize * dq);
  let mut v_s = Shared::<[F]>::new_slice(32usize * dv);
  let mut b_s = Shared::<[F]>::new_slice(32usize);
  let mut sc = Array::<F>::new(32usize);

  let kv_heads = heads as usize / groups as usize;
  let k_base = (n * kv_heads + kvh) * seq * dq;
  let v_base = (n * kv_heads + kvh) * seq * dv;
  for j0 in range_stepped(0usize, seq, 32usize) {
    // Stage 32 keys and values (zeros, and a -inf-like bias, past the end).
    for e in range_stepped(tid, 32usize * dq, 64usize) {
      let j = e / dq;
      let d = e % dq;
      let mut val = F::new(0.0f32);
      if j0 + j < seq {
        val = k[k_base + (j0 + j) * dq + d];
      }
      k_s[e] = val;
    }
    for e in range_stepped(tid, 32usize * dv, 64usize) {
      let j = e / dv;
      let d = e % dv;
      let mut val = F::new(0.0f32);
      if j0 + j < seq {
        val = v[v_base + (j0 + j) * dv + d];
      }
      v_s[e] = val;
    }
    if tid < 32usize {
      let mut bv = F::new(-3.0e38f32);
      if j0 + tid < seq {
        bv = bias[n * seq + j0 + tid];
      }
      b_s[tid] = bv;
    }
    sync_cube();

    let mut m_blk = m_run;
    #[unroll]
    for j in 0..32usize {
      let mut s = b_s[j];
      #[unroll]
      for d in 0..dq {
        s += qv[d] * k_s[j * dq + d];
      }
      sc[j] = s;
      m_blk = F::max(m_blk, s);
    }
    let alpha = F::exp(m_run - m_blk);
    l_run *= alpha;
    #[unroll]
    for d in 0..dv {
      acc[d] *= alpha;
    }
    #[unroll]
    for j in 0..32usize {
      let p = F::exp(sc[j] - m_blk);
      l_run += p;
      #[unroll]
      for d in 0..dv {
        acc[d] += p * v_s[j * dv + d];
      }
    }
    m_run = m_blk;
    sync_cube();
  }

  if in_range {
    // A row whose keys are all masked out (-inf bias) sums to 0: its output is 0 (burn-onnx's `hidden_rows`).
    let mut inv = F::new(0.0f32);
    if l_run > F::new(0.0f32) {
      inv = F::new(1.0f32) / l_run;
    }
    let o_base = ((n * heads as usize + h) * seq + qi) * dv;
    #[unroll]
    for d in 0..dv {
      out[o_base + d] = acc[d] * inv;
    }
  }
}

/// [`attention_kernel`] on the token layout, with the rotary embedding applied as `q` and `k` are loaded: `q` is the
/// query projection `[N, S, H * dq]`, `k`, `v` `[N, S, Hkv * dq]`, `[N, S, Hkv * dv]`, the output `[N, S, H * dv]` (no
/// head permutes on either side). With `rope`, channel `p < dq / 2` of a head pairs with `p + dq / 2` (bin2onnx's
/// permuted projections): `(x0 cos - x1 sin, x0 sin + x1 cos)`, `cos`/`sin` `[Hc, S, dq / 2]` with `Hc` 1 (fixed RoPE)
/// or the q (resp. kv) heads (learnable RoPE).
#[cube(launch)]
pub(crate) fn attention_tok_kernel<F: Float>(
  q: &[F],
  k: &[F],
  v: &[F],
  cos_q: &[F],
  sin_q: &[F],
  cos_k: &[F],
  sin_k: &[F],
  bias: &[F],
  out: &mut [F],
  seq: u32,
  heads: u32,
  kv_heads: u32,
  q_tables: u32,
  k_tables: u32,
  scale: f32,
  #[comptime] dq: usize,
  #[comptime] dv: usize,
  #[comptime] rope: bool,
) {
  let qi = CUBE_POS_X as usize * 64 + UNIT_POS_X as usize;
  let h = CUBE_POS_Y as usize;
  let n = CUBE_POS_Z as usize;
  let tid = UNIT_POS_X as usize;
  let seq = seq as usize;
  let heads = heads as usize;
  let kv_heads = kv_heads as usize;
  let kvh = h * kv_heads / heads;
  let half = dq / 2;
  let in_range = qi < seq;
  let qrow = if in_range { qi } else { 0usize };

  let mut qv = Array::<F>::new(dq);
  let q_base = (n * seq + qrow) * heads * dq + h * dq;
  if comptime![rope] {
    let qt = if q_tables as usize == 1 { 0usize } else { h };
    let t_base = (qt * seq + qrow) * half;
    #[unroll]
    for p in 0..half {
      let x0 = q[q_base + p];
      let x1 = q[q_base + half + p];
      let c = cos_q[t_base + p];
      let s = sin_q[t_base + p];
      qv[p] = (x0 * c - x1 * s) * F::cast_from(scale);
      qv[half + p] = (x0 * s + x1 * c) * F::cast_from(scale);
    }
  } else {
    #[unroll]
    for d in 0..dq {
      qv[d] = q[q_base + d] * F::cast_from(scale);
    }
  }
  let mut acc = Array::<F>::new(dv);
  #[unroll]
  for d in 0..dv {
    acc[d] = F::new(0.0f32);
  }
  let mut m_run = F::new(-3.0e38f32);
  let mut l_run = F::new(0.0f32);

  let mut k_s = Shared::<[F]>::new_slice(32usize * dq);
  let mut v_s = Shared::<[F]>::new_slice(32usize * dv);
  let mut b_s = Shared::<[F]>::new_slice(32usize);
  let mut sc = Array::<F>::new(32usize);

  let kt = if k_tables as usize == 1 { 0usize } else { kvh };
  for j0 in range_stepped(0usize, seq, 32usize) {
    for e in range_stepped(tid, 32usize * dq, 64usize) {
      let j = e / dq;
      let d = e % dq;
      let mut val = F::new(0.0f32);
      if j0 + j < seq {
        let k_base = (n * seq + j0 + j) * kv_heads * dq + kvh * dq;
        if comptime![rope] {
          let p = if d < half { d } else { d - half };
          let x0 = k[k_base + p];
          let x1 = k[k_base + half + p];
          let t = (kt * seq + j0 + j) * half + p;
          let c = cos_k[t];
          let s = sin_k[t];
          val = if d < half { x0 * c - x1 * s } else { x0 * s + x1 * c };
        } else {
          val = k[k_base + d];
        }
      }
      k_s[e] = val;
    }
    for e in range_stepped(tid, 32usize * dv, 64usize) {
      let j = e / dv;
      let d = e % dv;
      let mut val = F::new(0.0f32);
      if j0 + j < seq {
        val = v[(n * seq + j0 + j) * kv_heads * dv + kvh * dv + d];
      }
      v_s[e] = val;
    }
    if tid < 32usize {
      let mut bv = F::new(-3.0e38f32);
      if j0 + tid < seq {
        bv = bias[n * seq + j0 + tid];
      }
      b_s[tid] = bv;
    }
    sync_cube();

    let mut m_blk = m_run;
    #[unroll]
    for j in 0..32usize {
      let mut s = b_s[j];
      #[unroll]
      for d in 0..dq {
        s += qv[d] * k_s[j * dq + d];
      }
      sc[j] = s;
      m_blk = F::max(m_blk, s);
    }
    let alpha = F::exp(m_run - m_blk);
    l_run *= alpha;
    #[unroll]
    for d in 0..dv {
      acc[d] *= alpha;
    }
    #[unroll]
    for j in 0..32usize {
      let p = F::exp(sc[j] - m_blk);
      l_run += p;
      #[unroll]
      for d in 0..dv {
        acc[d] += p * v_s[j * dv + d];
      }
    }
    m_run = m_blk;
    sync_cube();
  }

  if in_range {
    let mut inv = F::new(0.0f32);
    if l_run > F::new(0.0f32) {
      inv = F::new(1.0f32) / l_run;
    }
    let o_base = (n * seq + qi) * heads * dv + h * dv;
    #[unroll]
    for d in 0..dv {
      out[o_base + d] = acc[d] * inv;
    }
  }
}
