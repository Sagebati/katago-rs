//! The cubecl kernels of [`super`]: Winograd F(4x4, 3x3) transforms and GEMMs, the NCHW 1x1 GEMM, the board
//! reduces, and the host reference of the filter transform.

use burn_cubecl::cubecl;
use burn_cubecl::cubecl::prelude::*;

/// Input tile side (F(4x4, 3x3)).
#[doc(hidden)]
pub const IN_TILE: usize = 6;
/// Output tile side.
#[doc(hidden)]
pub const OUT_TILE: usize = 4;
/// Tile points: one GEMM each.
#[doc(hidden)]
pub const TILE_POINTS: usize = IN_TILE * IN_TILE;

/// GEMM block: `GEMM_M` tiles x `GEMM_N` output channels per cube, `GEMM_K` input channels per shared-memory step.
#[doc(hidden)]
pub const GEMM_M: usize = 64;
#[doc(hidden)]
pub const GEMM_N: usize = 64;
#[doc(hidden)]
pub const GEMM_K: usize = 16;
/// Threads per cube side (16 x 16), each computing `GEMM_M / GEMM_T` x `GEMM_N / GEMM_T` outputs.
pub(crate) const GEMM_T: usize = 16;

pub(crate) fn round_up(x: usize, mult: usize) -> usize {
  x.div_ceil(mult) * mult
}

#[doc(hidden)]
/// Host reference of [`ConvOps::wino_filter_3x3`] (examples/wino_bench.rs checks one against the other).
/// Transforms a `[oc, ic, 3, 3]` filter into upstream's `[36, icPad, ocPad]` layout (`ConvLayer`'s
/// `transform3x3_6`, rows then columns), zero in the padding.
/// upstream: ConvLayer::ConvLayer (openclbackend.cpp), the 3x3 / inTileXSize == 6 branch
pub fn transform_filter_3x3(weights: &[f32], oc: usize, ic: usize) -> Vec<f32> {
  assert_eq!(weights.len(), oc * ic * 9);
  let ic_pad = round_up(ic, GEMM_K);
  let oc_pad = round_up(oc, GEMM_N);
  let transform = |z0: f32, z1: f32, z2: f32| -> [f32; 6] {
    [
      0.25f32 * z0,
      ((1.0 / 6.0) * (-z0 - z1 - z2) as f64) as f32,
      ((1.0 / 6.0) * (-z0 + z1 - z2) as f64) as f32,
      ((1.0 / 24.0) * (z0 as f64 + 2.0 * z1 as f64 + 4.0 * z2 as f64)) as f32,
      ((1.0 / 24.0) * (z0 as f64 - 2.0 * z1 as f64 + 4.0 * z2 as f64)) as f32,
      1.0f32 * z2,
    ]
  };
  let mut out = vec![0f32; TILE_POINTS * ic_pad * oc_pad];
  for o in 0..oc {
    for i in 0..ic {
      let g = &weights[(o * ic + i) * 9..(o * ic + i) * 9 + 9];
      // Rows: each of the 3 filter rows becomes 6 values.
      let mut rows = [[0f32; 6]; 3];
      for y in 0..3 {
        rows[y] = transform(g[y * 3], g[y * 3 + 1], g[y * 3 + 2]);
      }
      // Columns.
      for x in 0..6 {
        let col = transform(rows[0][x], rows[1][x], rows[2][x]);
        for (y, v) in col.iter().enumerate() {
          out[((y * IN_TILE + x) * ic_pad + i) * oc_pad + o] = *v;
        }
      }
    }
  }
  out
}

/// One thread per (tile, input channel), in `F`, storing in `S` (f16 halves the transformed buffer): load the 6x6
/// window (zero outside the board), apply B^T d B, write the
/// 36 values. With `bn_act`, the window first goes through `mish(x * scale + bias) * mask` as upstream's
/// `bnActTransform` (ACTIVATION 2), or `relu(x * scale + bias) * mask` with `relu` (ACTIVATION 1); without
/// `has_scale`, `x + bias` (a scale folded into the previous conv). Positions outside the board stay 0, as the padding
/// of the activated tensor.
/// upstream: winogradTransformNCHW / winogradBNActTransformNCHW (openclkernels.cpp), CONV 3, OUTTILE 4
#[cube(launch)]
pub(crate) fn wino_transform<F: Float, S: Float>(
  input: &[F],
  scale: &[F],
  bias: &[F],
  mask: &[F],
  transformed: &mut [S],
  n_size: u32,
  x_size: u32,
  y_size: u32,
  tiles_x: u32,
  tiles_y: u32,
  ic_size: u32,
  ic_pad: u32,
  nt_pad: u32,
  #[comptime] bn_act: bool,
  #[comptime] relu: bool,
  #[comptime] has_scale: bool,
) {
  let ntxty = ABSOLUTE_POS_X;
  let ic = ABSOLUTE_POS_Y;
  if ntxty >= nt_pad || ic >= ic_pad {
    terminate!();
  }
  let tile_x = ntxty % tiles_x;
  let rest = ntxty / tiles_x;
  let tile_y = rest % tiles_y;
  let n = rest / tiles_y;
  let xy_size = x_size * y_size;
  let live = n < n_size && ic < ic_size;
  let nic = (n * ic_size + ic) as usize;

  let mut w = Array::<F>::new(36usize);
  #[unroll]
  for sub_y in 0..6u32 {
    #[unroll]
    for sub_x in 0..6u32 {
      // y, x may be -1: compare in i32.
      let y = (tile_y * 4 + sub_y) as i32 - 1;
      let x = (tile_x * 4 + sub_x) as i32 - 1;
      let mut value = F::new(0.0f32);
      if live && y >= 0 && y < y_size as i32 && x >= 0 && x < x_size as i32 {
        let xy = (y as u32 * x_size + x as u32) as usize;
        let v = input[nic * xy_size as usize + xy];
        if comptime![bn_act] {
          let mut a = v + bias[ic as usize];
          if comptime![has_scale] {
            a = v * scale[ic as usize] + bias[ic as usize];
          }
          if comptime![relu] {
            value = F::max(a, F::new(0.0f32)) * mask[(n * xy_size) as usize + xy];
          } else {
            // burn's softplus: identity above 20 (exclusive).
            let sp = if a > F::new(20.0f32) { a } else { F::log1p(F::exp(a)) };
            value = a * F::tanh(sp) * mask[(n * xy_size) as usize + xy];
          }
        } else {
          value = v;
        }
      }
      w[(sub_y * 6 + sub_x) as usize] = value;
    }
  }

  // Rows (B^T applied along x).
  #[unroll]
  for sub_y in 0..6usize {
    let z0 = w[sub_y * 6];
    let z1 = w[sub_y * 6 + 1];
    let z2 = w[sub_y * 6 + 2];
    let z3 = w[sub_y * 6 + 3];
    let z4 = w[sub_y * 6 + 4];
    let z5 = w[sub_y * 6 + 5];
    w[sub_y * 6] = F::new(4.0f32) * z0 - F::new(5.0f32) * z2 + z4;
    w[sub_y * 6 + 1] = -F::new(4.0f32) * z1 - F::new(4.0f32) * z2 + z3 + z4;
    w[sub_y * 6 + 2] = F::new(4.0f32) * z1 - F::new(4.0f32) * z2 - z3 + z4;
    w[sub_y * 6 + 3] = -F::new(2.0f32) * z1 - z2 + F::new(2.0f32) * z3 + z4;
    w[sub_y * 6 + 4] = F::new(2.0f32) * z1 - z2 - F::new(2.0f32) * z3 + z4;
    w[sub_y * 6 + 5] = F::new(4.0f32) * z1 - F::new(5.0f32) * z3 + z5;
  }
  // Columns (along y).
  #[unroll]
  for sub_x in 0..6usize {
    let z0 = w[sub_x];
    let z1 = w[6 + sub_x];
    let z2 = w[12 + sub_x];
    let z3 = w[18 + sub_x];
    let z4 = w[24 + sub_x];
    let z5 = w[30 + sub_x];
    w[sub_x] = F::new(4.0f32) * z0 - F::new(5.0f32) * z2 + z4;
    w[6 + sub_x] = -F::new(4.0f32) * z1 - F::new(4.0f32) * z2 + z3 + z4;
    w[12 + sub_x] = F::new(4.0f32) * z1 - F::new(4.0f32) * z2 - z3 + z4;
    w[18 + sub_x] = -F::new(2.0f32) * z1 - z2 + F::new(2.0f32) * z3 + z4;
    w[24 + sub_x] = F::new(2.0f32) * z1 - z2 - F::new(2.0f32) * z3 + z4;
    w[30 + sub_x] = F::new(4.0f32) * z1 - F::new(5.0f32) * z3 + z5;
  }

  #[unroll]
  for p in 0..36usize {
    transformed[(p * ic_pad as usize + ic as usize) * nt_pad as usize + ntxty as usize] = S::cast_from(w[p]);
  }
}

/// Batched GEMM over the 36 tile points: `c[p][n][m] = sum_k a[p][k][m] * b[p][k][n]` (upstream's
/// `doBatchedXGemm_KM_KN_NM`). Every dimension is padded to its block size, so there are no bounds checks.
#[cube(launch)]
pub(crate) fn wino_gemm<F: Float>(a: &[F], b: &[F], c: &mut [F], m_pad: u32, n_pad: u32, k_pad: u32) {
  let p = CUBE_POS_Z as usize;
  let m0 = CUBE_POS_X as usize * 64;
  let n0 = CUBE_POS_Y as usize * 64;
  let tx = UNIT_POS_X as usize;
  let ty = UNIT_POS_Y as usize;
  let tid = ty * 16 + tx;
  let m_pad = m_pad as usize;
  let n_pad = n_pad as usize;
  let k_pad = k_pad as usize;

  let mut a_s = Shared::<[F]>::new_slice(16usize * 64usize);
  let mut b_s = Shared::<[F]>::new_slice(16usize * 64usize);
  let mut acc = Array::<F>::new(16usize);
  #[unroll]
  for i in 0..16usize {
    acc[i] = F::new(0.0f32);
  }
  let mut a_reg = Array::<F>::new(4usize);
  let mut b_reg = Array::<F>::new(4usize);

  let a_base = p * k_pad * m_pad;
  let b_base = p * k_pad * n_pad;

  for k0 in range_stepped(0usize, k_pad, 16usize) {
    #[unroll]
    for r in 0..4usize {
      let idx = tid + 256 * r;
      let kk = idx / 64;
      let mm = idx % 64;
      a_s[idx] = a[a_base + (k0 + kk) * m_pad + m0 + mm];
      b_s[idx] = b[b_base + (k0 + kk) * n_pad + n0 + mm];
    }
    sync_cube();
    #[unroll]
    for kk in 0..16usize {
      #[unroll]
      for i in 0..4usize {
        a_reg[i] = a_s[kk * 64 + tx + 16 * i];
        b_reg[i] = b_s[kk * 64 + ty + 16 * i];
      }
      #[unroll]
      for j in 0..4usize {
        #[unroll]
        for i in 0..4usize {
          acc[j * 4 + i] += a_reg[i] * b_reg[j];
        }
      }
    }
    sync_cube();
  }

  let c_base = p * n_pad * m_pad;
  #[unroll]
  for j in 0..4usize {
    #[unroll]
    for i in 0..4usize {
      c[c_base + (n0 + ty + 16 * j) * m_pad + m0 + tx + 16 * i] = acc[j * 4 + i];
    }
  }
}

/// [`wino_gemm`] on `N`-wide vectors (`N` = 4): each thread computes 4 tiles (one vector) x 4 output channels from
/// one vector load of each operand per K step, so packed f16 math is used when `S` is f16 (RDNA's `v_pk_fma_f16`).
/// Each step of `GEMM_K` input channels is summed in `S`, then added into f32 accumulators: f16 sums over 16 terms
/// instead of the whole K (192), for one conversion per 16 FMAs. The result is written as `O`.
#[cube(launch)]
pub(crate) fn wino_gemm_v<S: Float, O: Float, N: Size>(a: &[Vector<S, N>], b: &[Vector<S, N>], c: &mut [Vector<O, N>], m_pad: u32, n_pad: u32, k_pad: u32) {
  let p = CUBE_POS_Z as usize;
  let mv0 = CUBE_POS_X as usize * 16;
  let n0 = CUBE_POS_Y as usize * 64;
  let tx = UNIT_POS_X as usize;
  let ty = UNIT_POS_Y as usize;
  let tid = ty * 16 + tx;
  // In vectors.
  let m_v = m_pad as usize / 4;
  let n_v = n_pad as usize / 4;
  let k_pad = k_pad as usize;

  let mut a_s = Shared::<[Vector<S, N>]>::new_slice(16usize * 16usize);
  let mut b_s = Shared::<[Vector<S, N>]>::new_slice(16usize * 16usize);
  let mut acc = Array::<Vector<f32, N>>::new(4usize);
  let mut part = Array::<Vector<S, N>>::new(4usize);
  #[unroll]
  for j in 0..4usize {
    acc[j] = Vector::new(0.0f32);
  }

  let a_base = p * k_pad * m_v;
  let b_base = p * k_pad * n_v;
  let kk = tid / 16;
  let lane = tid % 16;

  for k0 in range_stepped(0usize, k_pad, 16usize) {
    a_s[tid] = a[a_base + (k0 + kk) * m_v + mv0 + lane];
    b_s[tid] = b[b_base + (k0 + kk) * n_v + n0 / 4 + lane];
    sync_cube();
    #[unroll]
    for j in 0..4usize {
      part[j] = Vector::new(S::new(0.0f32));
    }
    #[unroll]
    for k in 0..16usize {
      let av = a_s[k * 16 + tx];
      let bv = b_s[k * 16 + ty];
      #[unroll]
      for j in 0..4usize {
        part[j] += av * Vector::new(bv.extract(j));
      }
    }
    #[unroll]
    for j in 0..4usize {
      acc[j] += Vector::cast_from(part[j]);
    }
    sync_cube();
  }

  let c_base = p * n_pad as usize * m_v;
  #[unroll]
  for j in 0..4usize {
    c[c_base + (n0 + ty * 4 + j) * m_v + mv0 + tx] = Vector::cast_from(acc[j]);
  }
}

/// One thread per (n, oc, tile): read the 36 GEMM results, apply A^T m A, write the 4x4 output tile (cropped to
/// the board). Upstream adds a residual in a separate `addPointWise`, as the generated graph does.
/// upstream: winogradUntransformNCHW (openclkernels.cpp), CONV 3, OUTTILE 4
#[cube(launch)]
pub(crate) fn wino_untransform<S: Float, F: Float>(
  transformed: &[S],
  output: &mut [F],
  n_size: u32,
  x_size: u32,
  y_size: u32,
  tiles_x: u32,
  tiles_y: u32,
  oc_size: u32,
  oc_pad: u32,
  nt_pad: u32,
) {
  // x over (oc, tile) of one batch row, y over the rows: WebGPU caps each grid dimension at 65535 workgroups.
  let tiles = tiles_x * tiles_y;
  let within = CUBE_POS_X * 64 + UNIT_POS_X;
  if within >= oc_size * tiles || CUBE_POS_Y >= n_size {
    terminate!();
  }
  let t = CUBE_POS_Y * oc_size * tiles + within;
  let tile = t % tiles;
  let noc = t / tiles;
  let n = noc / oc_size;
  let oc = noc % oc_size;
  let tile_x = tile % tiles_x;
  let tile_y = tile / tiles_x;
  let ntile = (n * tiles + tile) as usize;

  let mut w = Array::<F>::new(36usize);
  #[unroll]
  for p in 0..36usize {
    w[p] = F::cast_from(transformed[(p * oc_pad as usize + oc as usize) * nt_pad as usize + ntile]);
  }
  #[unroll]
  for sub_y in 0..6usize {
    let z0 = w[sub_y * 6];
    let z1 = w[sub_y * 6 + 1];
    let z2 = w[sub_y * 6 + 2];
    let z3 = w[sub_y * 6 + 3];
    let z4 = w[sub_y * 6 + 4];
    let z5 = w[sub_y * 6 + 5];
    w[sub_y * 6] = z0 + z1 + z2 + z3 + z4;
    w[sub_y * 6 + 1] = (z1 - z2) + F::new(2.0f32) * (z3 - z4);
    w[sub_y * 6 + 2] = (z1 + z2) + F::new(4.0f32) * (z3 + z4);
    w[sub_y * 6 + 3] = (z1 - z2) + F::new(8.0f32) * (z3 - z4) + z5;
  }
  #[unroll]
  for sub_x in 0..4usize {
    let z0 = w[sub_x];
    let z1 = w[6 + sub_x];
    let z2 = w[12 + sub_x];
    let z3 = w[18 + sub_x];
    let z4 = w[24 + sub_x];
    let z5 = w[30 + sub_x];
    w[sub_x] = z0 + z1 + z2 + z3 + z4;
    w[6 + sub_x] = (z1 - z2) + F::new(2.0f32) * (z3 - z4);
    w[12 + sub_x] = (z1 + z2) + F::new(4.0f32) * (z3 + z4);
    w[18 + sub_x] = (z1 - z2) + F::new(8.0f32) * (z3 - z4) + z5;
  }

  let base = (noc * y_size * x_size) as usize;
  #[unroll]
  for sub_y in 0..4u32 {
    #[unroll]
    for sub_x in 0..4u32 {
      let y = tile_y * 4 + sub_y;
      let x = tile_x * 4 + sub_x;
      if y < y_size && x < x_size {
        let idx = base + (y * x_size + x) as usize;
        output[idx] = w[(sub_y * 6 + sub_x) as usize];
      }
    }
  }
}

/// The filter transform of [`transform_filter_3x3`] on the device (no readback, which wasm cannot do
/// synchronously): one thread per (oc, ic) of the padded filter.
#[cube(launch)]
pub(crate) fn wino_filter<F: Float, S: Float>(weights: &[F], out: &mut [S], oc_size: u32, ic_size: u32, oc_pad: u32, ic_pad: u32) {
  let o = ABSOLUTE_POS_X;
  let i = ABSOLUTE_POS_Y;
  if o >= oc_pad || i >= ic_pad {
    terminate!();
  }
  let live = o < oc_size && i < ic_size;
  let mut g = Array::<F>::new(9usize);
  #[unroll]
  for k in 0..9usize {
    let mut v = F::new(0.0f32);
    if live {
      v = weights[((o * ic_size + i) * 9) as usize + k];
    }
    g[k] = v;
  }
  // Rows: 3 rows of 6.
  let mut r = Array::<F>::new(18usize);
  #[unroll]
  for y in 0..3usize {
    let z0 = g[y * 3];
    let z1 = g[y * 3 + 1];
    let z2 = g[y * 3 + 2];
    r[y * 6] = F::new(0.25f32) * z0;
    r[y * 6 + 1] = (-z0 - z1 - z2) / F::new(6.0f32);
    r[y * 6 + 2] = (-z0 + z1 - z2) / F::new(6.0f32);
    r[y * 6 + 3] = (z0 + F::new(2.0f32) * z1 + F::new(4.0f32) * z2) / F::new(24.0f32);
    r[y * 6 + 4] = (z0 - F::new(2.0f32) * z1 + F::new(4.0f32) * z2) / F::new(24.0f32);
    r[y * 6 + 5] = z2;
  }
  // Columns.
  #[unroll]
  for x in 0..6usize {
    let z0 = r[x];
    let z1 = r[6 + x];
    let z2 = r[12 + x];
    let base = i as usize * oc_pad as usize + o as usize;
    let stride = ic_pad as usize * oc_pad as usize;
    out[x * stride + base] = S::cast_from(F::new(0.25f32) * z0);
    out[(6 + x) * stride + base] = S::cast_from((-z0 - z1 - z2) / F::new(6.0f32));
    out[(12 + x) * stride + base] = S::cast_from((-z0 + z1 - z2) / F::new(6.0f32));
    out[(18 + x) * stride + base] = S::cast_from((z0 + F::new(2.0f32) * z1 + F::new(4.0f32) * z2) / F::new(24.0f32));
    out[(24 + x) * stride + base] = S::cast_from((z0 - F::new(2.0f32) * z1 + F::new(4.0f32) * z2) / F::new(24.0f32));
    out[(30 + x) * stride + base] = S::cast_from(z2);
  }
}

/// 1x1 convolution in NCHW as one GEMM per batch row: `out[b][n][m] = sum_k x[b][k][m] * wt[k][n]`, with
/// `m` the board points and `n` the output channels bounds-checked, `k` the input channels a multiple of `GEMM_K`. Upstream's `doStridedBatchedXGemmDirect_KM_KN_NM` with the filter stride 0.
#[cube(launch)]
pub(crate) fn conv1x1_gemm<F: Float>(
  x: &[F],
  wt: &[F],
  scale: &[F],
  bias: &[F],
  mask: &[F],
  out: &mut [F],
  m_size: u32,
  n_size: u32,
  k_size: u32,
  #[comptime] bn_act: bool,
  #[comptime] relu: bool,
  #[comptime] has_scale: bool,
) {
  let bt = CUBE_POS_Z as usize;
  let m0 = CUBE_POS_X as usize * 64;
  let n0 = CUBE_POS_Y as usize * 64;
  let tx = UNIT_POS_X as usize;
  let ty = UNIT_POS_Y as usize;
  let tid = ty * 16 + tx;
  let m_size = m_size as usize;
  let n_size = n_size as usize;
  let k_size = k_size as usize;

  let mut a_s = Shared::<[F]>::new_slice(16usize * 64usize);
  let mut b_s = Shared::<[F]>::new_slice(16usize * 64usize);
  let mut acc = Array::<F>::new(16usize);
  #[unroll]
  for i in 0..16usize {
    acc[i] = F::new(0.0f32);
  }
  let mut a_reg = Array::<F>::new(4usize);
  let mut b_reg = Array::<F>::new(4usize);

  let a_base = bt * k_size * m_size;

  for k0 in range_stepped(0usize, k_size, 16usize) {
    #[unroll]
    for r in 0..4usize {
      let idx = tid + 256 * r;
      let kk = idx / 64;
      let mm = idx % 64;
      let mut v = F::new(0.0f32);
      if m0 + mm < m_size {
        v = x[a_base + (k0 + kk) * m_size + m0 + mm];
        if comptime![bn_act] {
          let mut a = v + bias[k0 + kk];
          if comptime![has_scale] {
            a = v * scale[k0 + kk] + bias[k0 + kk];
          }
          if comptime![relu] {
            v = F::max(a, F::new(0.0f32)) * mask[bt * m_size + m0 + mm];
          } else {
            let sp = if a > F::new(20.0f32) { a } else { F::log1p(F::exp(a)) };
            v = a * F::tanh(sp) * mask[bt * m_size + m0 + mm];
          }
        }
      }
      a_s[idx] = v;
      let mut w = F::new(0.0f32);
      if n0 + mm < n_size {
        w = wt[(k0 + kk) * n_size + n0 + mm];
      }
      b_s[idx] = w;
    }
    sync_cube();
    #[unroll]
    for kk in 0..16usize {
      #[unroll]
      for i in 0..4usize {
        a_reg[i] = a_s[kk * 64 + tx + 16 * i];
        b_reg[i] = b_s[kk * 64 + ty + 16 * i];
      }
      #[unroll]
      for j in 0..4usize {
        #[unroll]
        for i in 0..4usize {
          acc[j * 4 + i] += a_reg[i] * b_reg[j];
        }
      }
    }
    sync_cube();
  }

  let c_base = bt * n_size * m_size;
  #[unroll]
  for j in 0..4usize {
    #[unroll]
    for i in 0..4usize {
      let m = m0 + tx + 16 * i;
      let n = n0 + ty + 16 * j;
      if m < m_size && n < n_size {
        out[c_base + n * m_size + m] = acc[j * 4 + i];
      }
    }
  }
}

/// Sum (or max, with `max`) over the last axis of a contiguous `[rows, len]` tensor: one 64-thread workgroup per
/// row, strided partial results, then a shared-memory tree.
#[cube(launch)]
pub(crate) fn reduce_last_kernel<F: Float>(x: &[F], out: &mut [F], len: u32, channels: u32, #[comptime] max: bool) {
  // x over channels, y over batch rows (65535 workgroups per grid dimension).
  let row = (CUBE_POS_Y * channels + CUBE_POS_X) as usize;
  let tid = UNIT_POS_X as usize;
  let len = len as usize;
  let base = row * len;
  let mut acc = F::new(0.0f32);
  if comptime![max] {
    acc = F::min_value();
  }
  for i in range_stepped(tid, len, 64usize) {
    let v = x[base + i];
    if comptime![max] {
      acc = F::max(acc, v);
    } else {
      acc += v;
    }
  }
  let mut s = Shared::<[F]>::new_slice(64usize);
  s[tid] = acc;
  sync_cube();
  #[unroll]
  for step in 0..6u32 {
    let half = 32usize >> step as usize;
    if tid < half {
      if comptime![max] {
        s[tid] = F::max(s[tid], s[tid + half]);
      } else {
        s[tid] = s[tid] + s[tid + half];
      }
    }
    sync_cube();
  }
  if tid == 0 {
    out[row] = s[0];
  }
}

/// Output channels per thread of [`conv_direct_kernel`]: each input value it loads feeds that many accumulators.
pub(crate) const DIRECT_OC: usize = 16;

/// Stride-1 "same" convolution of any odd kernel size, computed directly: one thread per (row, board point) and
/// `DIRECT_OC` output channels, summing over input channels then kernel rows and columns (zero outside the board).
/// For the convs Winograd and the 1x1 GEMM do not take: the input conv of the older nets (5x5 over 22 features),
/// small enough that this beats im2col + matmul by far.
#[cube(launch)]
pub(crate) fn conv_direct_kernel<F: Float>(
  x: &[F],
  w: &[F],
  out: &mut [F],
  ic_size: u32,
  oc_size: u32,
  y_size: u32,
  x_size: u32,
  #[comptime] kh: u32,
  #[comptime] kw: u32,
) {
  // x over the board points of a row, y over output-channel groups, z over rows.
  let points = y_size * x_size;
  let p = ABSOLUTE_POS_X;
  if p >= points {
    terminate!();
  }
  let oc0 = CUBE_POS_Y * 16;
  let n = CUBE_POS_Z;
  let py = (p / x_size) as i32;
  let px = (p % x_size) as i32;
  let mut acc = Array::<F>::new(16usize);
  #[unroll]
  for j in 0..16usize {
    acc[j] = F::new(0.0f32);
  }
  let ph = (kh / 2) as i32;
  let pw = (kw / 2) as i32;
  for ic in 0..ic_size {
    let in_base = ((n * ic_size + ic) * points) as usize;
    #[unroll]
    for ky in 0..kh {
      let iy = py + ky as i32 - ph;
      #[unroll]
      for kx in 0..kw {
        let ix = px + kx as i32 - pw;
        if iy >= 0 && iy < y_size as i32 && ix >= 0 && ix < x_size as i32 {
          let v = x[in_base + (iy as u32 * x_size + ix as u32) as usize];
          #[unroll]
          for j in 0..16u32 {
            let oc = oc0 + j;
            if oc < oc_size {
              acc[j as usize] += v * w[(((oc * ic_size + ic) * kh + ky) * kw + kx) as usize];
            }
          }
        }
      }
    }
  }
  #[unroll]
  for j in 0..16u32 {
    let oc = oc0 + j;
    if oc < oc_size {
      out[((n * oc_size + oc) * points + p) as usize] = acc[j as usize];
    }
  }
}
