//! Winograd F(4x4, 3x3) convolution on cubecl, following KataGo's OpenCL backend
//! (upstream cpp/neuralnet/openclkernels.cpp `winogradTransformNCHW`, `winogradBNActTransformNCHW`,
//! `winogradUntransformNCHW`, and the 3x3 branch of `ConvLayer` in openclbackend.cpp): the input is cut into
//! 6x6 tiles (offset -1, stride 4), each tile transformed (B^T d B), one GEMM per tile point multiplies the
//! transformed inputs by the transformed filters (G g G^T, done once on the host), and each 6x6 result is
//! untransformed (A^T m A) into a 4x4 output tile.
//!
//! Layouts are upstream's: transformed inputs `[36, icPad, ntPad]`, filters `[36, icPad, ocPad]`, GEMM output
//! `[36, ocPad, ntPad]`, where `nt = N * tilesY * tilesX` is padded to `GEMM_M`, `ic` to `GEMM_K` and `oc` to
//! `GEMM_N`. The transforms are upstream's active ones (points 0, ±1, ±2), not the commented-out low-error set.

use std::cell::RefCell;
use std::collections::HashMap;

use burn::backend::{Backend, Dispatch, backend_extension, tensor::FloatTensor};
use burn::module::ParamId;
use burn::nn::conv::Conv2d;
use burn::tensor::{DType, Shape, Tensor, TensorData};
use burn_cubecl::cubecl;
use burn_cubecl::cubecl::prelude::*;
use burn_cubecl::kernel::into_contiguous;
use burn_cubecl::ops::numeric::empty_device_contiguous_dtype;
use burn_cubecl::tensor::CubeTensor;
use burn_cubecl::CubeBackend;

/// Input tile side (F(4x4, 3x3)).
pub const IN_TILE: usize = 6;
/// Output tile side.
pub const OUT_TILE: usize = 4;
/// Tile points: one GEMM each.
pub const TILE_POINTS: usize = IN_TILE * IN_TILE;

/// GEMM block: `GEMM_M` tiles x `GEMM_N` output channels per cube, `GEMM_K` input channels per shared-memory step.
pub const GEMM_M: usize = 64;
pub const GEMM_N: usize = 64;
pub const GEMM_K: usize = 16;
/// Threads per cube side (16 x 16), each computing `GEMM_M / GEMM_T` x `GEMM_N / GEMM_T` outputs.
const GEMM_T: usize = 16;

pub fn round_up(x: usize, mult: usize) -> usize {
    x.div_ceil(mult) * mult
}

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

/// One thread per (tile, input channel): load the 6x6 window (zero outside the board), apply B^T d B, write the
/// 36 values. With `bn_act`, the window first goes through `mish(x * scale + bias) * mask` as upstream's
/// `bnActTransform` (ACTIVATION 2); positions outside the board stay 0, as the padding of the activated tensor.
/// upstream: winogradTransformNCHW / winogradBNActTransformNCHW (openclkernels.cpp), CONV 3, OUTTILE 4
#[cube(launch)]
fn wino_transform<F: Float>(
    input: &[F],
    scale: &[F],
    bias: &[F],
    mask: &[F],
    transformed: &mut [F],
    n_size: u32,
    x_size: u32,
    y_size: u32,
    tiles_x: u32,
    tiles_y: u32,
    ic_size: u32,
    ic_pad: u32,
    nt_pad: u32,
    #[comptime] bn_act: bool,
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
                    let a = v * scale[ic as usize] + bias[ic as usize];
                    let sp = if a < F::new(20.0f32) { F::log1p(F::exp(a)) } else { a };
                    value = a * F::tanh(sp) * mask[(n * xy_size) as usize + xy];
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
        transformed[(p * ic_pad as usize + ic as usize) * nt_pad as usize + ntxty as usize] = w[p];
    }
}

/// Batched GEMM over the 36 tile points: `c[p][n][m] = sum_k a[p][k][m] * b[p][k][n]` (upstream's
/// `doBatchedXGemm_KM_KN_NM`). Every dimension is padded to its block size, so there are no bounds checks.
#[cube(launch)]
fn wino_gemm<F: Float>(a: &[F], b: &[F], c: &mut [F], m_pad: u32, n_pad: u32, k_pad: u32) {
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

/// One thread per (n, oc, tile): read the 36 GEMM results, apply A^T m A, write the 4x4 output tile (cropped to
/// the board). With `residual`, the residual tensor is added (upstream adds it in a separate `addPointWise`).
/// upstream: winogradUntransformNCHW (openclkernels.cpp), CONV 3, OUTTILE 4
#[cube(launch)]
fn wino_untransform<F: Float>(
    transformed: &[F],
    residual: &[F],
    output: &mut [F],
    n_size: u32,
    x_size: u32,
    y_size: u32,
    tiles_x: u32,
    tiles_y: u32,
    oc_size: u32,
    oc_pad: u32,
    nt_pad: u32,
    #[comptime] add_residual: bool,
) {
    let t = ABSOLUTE_POS_X;
    let tiles = tiles_x * tiles_y;
    if t >= n_size * oc_size * tiles {
        terminate!();
    }
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
        w[p] = transformed[(p * oc_pad as usize + oc as usize) * nt_pad as usize + ntile];
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
                let mut v = w[(sub_y * 6 + sub_x) as usize];
                if comptime![add_residual] {
                    v += residual[idx];
                }
                output[idx] = v;
            }
        }
    }
}

/// 1x1 convolution in NCHW as one GEMM per batch row: `out[b][n][m] = sum_k x[b][k][m] * wt[k][n]`, with
/// `m` the board points (not padded: bounds-checked), `k` input and `n` output channels (`k % GEMM_K == 0`,
/// `n % GEMM_N == 0`). Upstream's `doStridedBatchedXGemmDirect_KM_KN_NM` with the filter stride 0.
#[cube(launch)]
fn conv1x1_gemm<F: Float>(x: &[F], wt: &[F], out: &mut [F], m_size: u32, n_size: u32, k_size: u32) {
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
            }
            a_s[idx] = v;
            b_s[idx] = wt[(k0 + kk) * n_size + n0 + mm];
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
            if m < m_size {
                out[c_base + (n0 + ty + 16 * j) * m_size + m] = acc[j * 4 + i];
            }
        }
    }
}

/// Output shape of [`ConvOps::wino_conv3x3`] and [`ConvOps::conv1x1`]: `[N, oc, H, W]` for an input `[N, ic, H, W]`.
pub fn wino_out_shape(x: &Shape, oc: &usize) -> Shape {
    Shape::new([x[0], *oc, x[2], x[3]])
}

#[backend_extension(Cube, Fusion)]
pub trait ConvOps: Backend {
    /// Stride-1 "same" 3x3 convolution, no bias. `filter` is [`transform_filter_3x3`]'s output for `oc` output
    /// channels.
    #[fusion(dtype = x, shape = wino_out_shape(x, oc))]
    fn wino_conv3x3(x: FloatTensor<Self>, filter: FloatTensor<Self>, oc: usize) -> FloatTensor<Self>;

    /// 1x1 convolution, no bias. `wt` is the `[ic, oc]` transposed weight; `ic % GEMM_K == 0`,
    /// `oc % GEMM_N == 0`.
    #[fusion(dtype = x, shape = wino_out_shape(x, oc))]
    fn conv1x1(x: FloatTensor<Self>, wt: FloatTensor<Self>, oc: usize) -> FloatTensor<Self>;
}

fn contiguous(t: CubeTensor) -> CubeTensor {
    if t.is_contiguous() { t } else { into_contiguous(t) }
}

impl ConvOps for CubeBackend {
    fn conv1x1(x: CubeTensor, wt: CubeTensor, oc: usize) -> CubeTensor {
        let x = contiguous(x);
        let wt = contiguous(wt);
        let dims = x.meta.shape().to_vec();
        let (n, ic, h, w) = (dims[0], dims[1], dims[2], dims[3]);
        assert!(ic % GEMM_K == 0 && oc % GEMM_N == 0, "conv1x1: ic {ic} / oc {oc} not padded");
        assert_eq!(wt.meta.shape().num_elements(), ic * oc, "conv1x1 weight size");
        let dtype = x.dtype;
        let output =
            empty_device_contiguous_dtype(x.client.clone(), x.device.clone(), Shape::new([n, oc, h, w]), dtype);
        let m = h * w;
        let (x_len, w_len, o_len) = (x.meta.shape().num_elements(), ic * oc, n * oc * m);
        macro_rules! launch {
            ($f:ty) => {
                // SAFETY: contiguous buffers of the lengths given; the kernel bounds-checks m and needs ic, oc
                // padded, as asserted above.
                unsafe {
                    conv1x1_gemm::launch::<$f>(
                        &x.client,
                        CubeCount::Static(m.div_ceil(64) as u32, (oc / GEMM_N) as u32, n as u32),
                        CubeDim::new_2d(GEMM_T as u32, GEMM_T as u32),
                        BufferArg::from_raw_parts(x.handle.clone(), x_len),
                        BufferArg::from_raw_parts(wt.handle.clone(), w_len),
                        BufferArg::from_raw_parts(output.handle.clone(), o_len),
                        m as u32,
                        oc as u32,
                        ic as u32,
                    )
                }
            };
        }
        match dtype {
            DType::F32 => launch!(f32),
            DType::F16 => launch!(half::f16),
            other => panic!("conv1x1: unsupported dtype {other:?}"),
        }
        output
    }

    fn wino_conv3x3(x: CubeTensor, filter: CubeTensor, oc: usize) -> CubeTensor {
        let x = contiguous(x);
        let filter = contiguous(filter);
        let dims = x.meta.shape().to_vec();
        let (n, ic, h, w) = (dims[0], dims[1], dims[2], dims[3]);
        let dtype = x.dtype;
        let client = x.client.clone();
        let device = x.device.clone();

        let tiles_x = w.div_ceil(OUT_TILE);
        let tiles_y = h.div_ceil(OUT_TILE);
        let nt = n * tiles_x * tiles_y;
        let nt_pad = round_up(nt, GEMM_M);
        let ic_pad = round_up(ic, GEMM_K);
        let oc_pad = round_up(oc, GEMM_N);
        assert_eq!(filter.meta.shape().num_elements(), TILE_POINTS * ic_pad * oc_pad, "wino filter size");

        let transformed = empty_device_contiguous_dtype(
            client.clone(),
            device.clone(),
            Shape::new([TILE_POINTS * ic_pad * nt_pad]),
            dtype,
        );
        let gemm_out = empty_device_contiguous_dtype(
            client.clone(),
            device.clone(),
            Shape::new([TILE_POINTS * oc_pad * nt_pad]),
            dtype,
        );
        let output = empty_device_contiguous_dtype(client.clone(), device.clone(), Shape::new([n, oc, h, w]), dtype);

        let len = |t: &CubeTensor| t.meta.shape().num_elements();
        macro_rules! launch_all {
            ($f:ty) => {{
                let x_len = len(&x);
                let t_len = len(&transformed);
                let g_len = len(&gemm_out);
                let f_len = len(&filter);
                let o_len = len(&output);
                // SAFETY: every buffer is contiguous and as long as the length given; the kernels index within
                // the padded sizes they are given.
                unsafe {
                    wino_transform::launch::<$f>(
                        &client,
                        CubeCount::Static((nt_pad / 64) as u32, ic_pad as u32, 1),
                        CubeDim::new_2d(64, 1),
                        BufferArg::from_raw_parts(x.handle.clone(), x_len),
                        BufferArg::from_raw_parts(x.handle.clone(), 1),
                        BufferArg::from_raw_parts(x.handle.clone(), 1),
                        BufferArg::from_raw_parts(x.handle.clone(), 1),
                        BufferArg::from_raw_parts(transformed.handle.clone(), t_len),
                        n as u32,
                        w as u32,
                        h as u32,
                        tiles_x as u32,
                        tiles_y as u32,
                        ic as u32,
                        ic_pad as u32,
                        nt_pad as u32,
                        false,
                    );
                    wino_gemm::launch::<$f>(
                        &client,
                        CubeCount::Static((nt_pad / GEMM_M) as u32, (oc_pad / GEMM_N) as u32, TILE_POINTS as u32),
                        CubeDim::new_2d(GEMM_T as u32, GEMM_T as u32),
                        BufferArg::from_raw_parts(transformed.handle.clone(), t_len),
                        BufferArg::from_raw_parts(filter.handle.clone(), f_len),
                        BufferArg::from_raw_parts(gemm_out.handle.clone(), g_len),
                        nt_pad as u32,
                        oc_pad as u32,
                        ic_pad as u32,
                    );
                    let threads = n * oc * tiles_x * tiles_y;
                    wino_untransform::launch::<$f>(
                        &client,
                        CubeCount::Static(threads.div_ceil(64) as u32, 1, 1),
                        CubeDim::new_1d(64),
                        BufferArg::from_raw_parts(gemm_out.handle.clone(), g_len),
                        BufferArg::from_raw_parts(gemm_out.handle.clone(), 1),
                        BufferArg::from_raw_parts(output.handle.clone(), o_len),
                        n as u32,
                        w as u32,
                        h as u32,
                        tiles_x as u32,
                        tiles_y as u32,
                        oc as u32,
                        oc_pad as u32,
                        nt_pad as u32,
                        false,
                    );
                }
            }};
        }
        match dtype {
            DType::F32 => launch_all!(f32),
            DType::F16 => launch_all!(half::f16),
            other => panic!("wino_conv3x3: unsupported dtype {other:?}"),
        }
        output
    }
}

thread_local! {
    /// Transformed filters by weight parameter: computed on a conv's first call (one weight readback).
    static FILTERS: RefCell<HashMap<ParamId, Tensor<1>>> = RefCell::new(HashMap::new());
}

/// The cached GPU-side form of `conv`'s weight (`make` gets the weight as `[oc, ic, kh, kw]` f32).
fn cached_filter(conv: &Conv2d, x: &Tensor<4>, len: usize, make: impl FnOnce(&[f32]) -> Vec<f32>) -> Tensor<1> {
    FILTERS.with(|filters| {
        filters
            .borrow_mut()
            .entry(conv.weight.id)
            .or_insert_with(|| {
                let weights: Vec<f32> = conv.weight.val().into_data().convert::<f32>().try_into_vec::<f32>().unwrap();
                let host = make(&weights);
                Tensor::<1>::from_data(TensorData::new(host, [len]).convert_dtype(x.dtype()), &x.device())
            })
            .clone()
    })
}

/// Whether [`conv1x1`] takes this 1x1 conv (the trunk's; the small head convs stay on the matmul).
pub fn conv1x1_fits(oc: usize, ic: usize) -> bool {
    ic % GEMM_K == 0 && oc % GEMM_N == 0
}

/// `conv2d_same` for a 1x1 kernel with [`conv1x1_fits`] on a cubecl device.
pub fn conv1x1(conv: &Conv2d, x: Tensor<4>) -> Tensor<4> {
    let [oc, ic, _, _] = conv.weight.dims();
    let wt = cached_filter(conv, &x, ic * oc, |w| {
        let mut t = vec![0f32; ic * oc];
        for o in 0..oc {
            for i in 0..ic {
                t[i * oc + o] = w[o * ic + i];
            }
        }
        t
    });
    Tensor::<4>::from_dispatch(Dispatch::conv1x1(x.into_dispatch(), wt.into_dispatch(), oc))
}

/// `conv2d_same` for a 3x3 kernel on a cubecl device.
pub fn conv3x3(conv: &Conv2d, x: Tensor<4>) -> Tensor<4> {
    let [oc, ic, _, _] = conv.weight.dims();
    let len = TILE_POINTS * round_up(ic, GEMM_K) * round_up(oc, GEMM_N);
    let filter = cached_filter(conv, &x, len, |w| transform_filter_3x3(w, oc, ic));
    Tensor::<4>::from_dispatch(Dispatch::wino_conv3x3(x.into_dispatch(), filter.into_dispatch(), oc))
}
