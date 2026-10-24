// Ported from KataGo cpp/neuralnet/eigenbackend.cpp (NeuralNet:: glue only), following cpp/neuralnet/nninterface.h @ ba938676d7f42d70950b3a535af2466fb642008c
//
// The `NeuralNet` trait of katago-core (nninterface.rs, DECISIONS.md D6) implemented over a
// burn-onnx-generated network. Mirrors eigenbackend.cpp's LoadedModel / ComputeContext /
// ComputeHandle / InputBuffers structs and its NeuralNet::getOutput input/output layout and
// symmetry handling; the forward pass itself (eigenbackend's `Model::apply`) is the generated
// network (tools/bin2onnx + burn-onnx), not ported code.
//
// Divergences from eigenbackend.cpp (each also marked PORT NOTE below):
// - The generated graph is NCHW with a fixed 19x19 (posLen) spatial size; Eigen is NHWC with
//   nnXLen x nnYLen. So createComputeContext rejects nnXLen/nnYLen != posLen and fp16/NHWC modes,
//   and createComputeHandle rejects inputsUseNHWC = true (Eigen rejects false). Smaller boards
//   are evaluated through the on-board mask (input channel 0), as with gtpForceMaxNNSize.
// - The ONNX policy output is [N, P, HW+1] with the pass logit last in each channel, where Eigen
//   has a separate `policyPass` [N, P] buffer; channel 0 = normal, 1 = optimistic (v12+).
// - The ModelDesc comes from the ONNX metadata (build.rs), not from parsing the `.bin.gz`; the
//   model "file" is the burn `.bpk` weights, so expectedSha256 cannot be checked.
// - getOutput is async: one readback of all outputs (`into_data_async`), then a yield so other
//   virtual threads run, and on wasm32 also the host's event loop (PORTING.md §11). The per-row
//   `symmetry` / `policyOptimism` are read from the input bufs before that await.
// - Logger output goes to the `log` crate.
// - Optional batch-shape buckets (`BurnNeuralNet::with_batch_buckets`, port-only, off by default): a
//   batch is padded to the smallest bucket that holds it, so a GPU backend sees a few fixed shapes
//   (each new shape costs a shader compile + autotune on wgpu, 1-3 s in the browser).

// Lints the fidelity rules trigger (PORTING.md §11b), for this ported file.
#![allow(clippy::needless_bool, clippy::needless_range_loop)]

use std::any::Any;
use std::marker::PhantomData;
use std::rc::Rc;

use burn::prelude::*;

use katago_core::core::commontypes::enabled_t;
use katago_core::core::global::Global;
use katago_core::core::global::KataError;
use katago_core::neuralnet::desc::ModelDesc;
use katago_core::neuralnet::modelversion::NNModelVersion;
use katago_core::neuralnet::nneval::NNResultBuf;
use katago_core::neuralnet::nninputs::NNOutput;
use katago_core::neuralnet::nninputs::NNPos;
use katago_core::neuralnet::nninputs::SymmetryHelpers;
use katago_core::neuralnet::nninterface;
use katago_core::neuralnet::sgfmetadata::SGFMetadata;
use katago_core::port::executor::LocalBoxFuture;
#[cfg(not(target_arch = "wasm32"))]
use katago_core::port::executor::yield_now;
use katago_core::testAssert;

use crate::Network;

/// The burn backend: `NeuralNet::*` for the generated network `N`, on a device the host created
/// (device initialization is async on the web, so it is the host's job).
pub struct BurnNeuralNet<N: Network> {
  device: Device,
  batch_buckets: Vec<i32>,
  _net: PhantomData<N>,
}

impl<N: Network> BurnNeuralNet<N> {
  pub fn new(device: Device) -> BurnNeuralNet<N> {
    BurnNeuralNet { device, batch_buckets: Vec::new(), _net: PhantomData }
  }

  /// PORT NOTE: port-only. Pads every batch of `n` rows to the smallest bucket `>= n` (and within the
  /// input buffers' max batch size; larger batches run unpadded). The padding rows hold stale inputs and
  /// their outputs are dropped. For GPU backends, where each new input shape compiles and autotunes
  /// kernels: e.g. powers of two up to nnMaxBatchSize, warmed up once at start.
  pub fn with_batch_buckets(mut self, mut buckets: Vec<i32>) -> BurnNeuralNet<N> {
    buckets.retain(|&b| b > 0);
    buckets.sort_unstable();
    buckets.dedup();
    self.batch_buckets = buckets;
    self
  }
}

/// The number of rows a batch of `n` runs with: the smallest bucket in `[n, max_batch_size]`, else `n`.
pub fn padded_batch_size(buckets: &[i32], n: i32, max_batch_size: i32) -> i32 {
  buckets.iter().copied().filter(|&b| b >= n && b <= max_batch_size).min().unwrap_or(n)
}

// LoadedModel / ModelDesc ---------------------------------------------------------------------------------------------

/// upstream: LoadedModel (eigenbackend.cpp)
/// PORT NOTE: holds the `.bpk` weights next to the ModelDesc; the burn model is built per
/// ComputeHandle, as Eigen builds its `Model` per handle (cached in the context there).
pub struct BurnLoadedModel {
  pub model_desc: ModelDesc,
  pub bpk: Vec<u8>,
}

impl nninterface::LoadedModel for BurnLoadedModel {
  fn as_any(&self) -> &dyn Any {
    self
  }
}

/// upstream: ComputeContext (eigenbackend.cpp)
/// PORT NOTE: the cached-models map (shared Eigen weights between handles) is not needed: each
/// handle owns its burn module.
pub struct BurnComputeContext {
  pub nn_x_len: i32,
  pub nn_y_len: i32,
  pub device: Device,
}

impl nninterface::ComputeContext for BurnComputeContext {
  fn as_any(&self) -> &dyn Any {
    self
  }
}

/// upstream: ComputeHandle (eigenbackend.cpp)
/// PORT NOTE: `context` pointer -> the copied context values; `model` is the generated network.
pub struct BurnComputeHandle<N: Network> {
  pub nn_x_len: i32,
  pub nn_y_len: i32,
  pub inputs_use_nhwc: bool,
  pub device: Device,
  pub model: N,
  /// PORT NOTE: port-only, see BurnNeuralNet::with_batch_buckets.
  pub batch_buckets: Vec<i32>,
  pub model_version: i32,
  pub num_input_channels: i32,
  pub num_policy_channels: i32,
  pub num_value_channels: i32,
  pub num_score_value_channels: i32,
  pub num_ownership_channels: i32,
}

impl<N: Network + 'static> nninterface::ComputeHandle for BurnComputeHandle<N> {
  fn as_any_mut(&mut self) -> &mut dyn Any {
    self
  }
  fn as_any(&self) -> &dyn Any {
    self
  }
}

/// upstream: InputBuffers (eigenbackend.cpp)
/// PORT NOTE: `spatialInput` is NCHW here (Eigen: NHWC).
pub struct BurnInputBuffers {
  pub max_batch_size: i32,

  pub single_input_elts: usize,
  pub single_input_global_elts: usize,
  pub single_input_meta_elts: usize,

  pub single_policy_pass_result_elts: usize,
  pub single_policy_result_elts: usize,
  pub single_value_result_elts: usize,
  pub single_score_value_result_elts: usize,
  pub single_ownership_result_elts: usize,

  pub spatial_input: Vec<f32>,
  pub global_input: Vec<f32>,
  pub meta_input: Vec<f32>,
}

impl BurnInputBuffers {
  /// upstream: InputBuffers::InputBuffers (eigenbackend.cpp)
  pub fn new(loaded_model: &BurnLoadedModel, max_batch_sz: i32, nn_x_len: i32, nn_y_len: i32) -> BurnInputBuffers {
    let m: &ModelDesc = &loaded_model.model_desc;

    let max_batch_size = max_batch_sz;
    let single_input_elts = (m.num_input_channels * nn_x_len * nn_y_len) as usize;
    let single_input_global_elts = m.num_input_global_channels as usize;
    let single_input_meta_elts = m.num_input_meta_channels as usize;

    let single_policy_pass_result_elts = m.num_policy_channels as usize;
    let single_policy_result_elts = (m.num_policy_channels * nn_x_len * nn_y_len) as usize;
    let single_value_result_elts = m.num_value_channels as usize;
    let single_score_value_result_elts = m.num_score_value_channels as usize;
    let single_ownership_result_elts = (m.num_ownership_channels * nn_x_len * nn_y_len) as usize;

    testAssert!(NNModelVersion::get_num_spatial_features(m.model_version) == Ok(m.num_input_channels));
    testAssert!(NNModelVersion::get_num_global_features(m.model_version) == Ok(m.num_input_global_channels));
    if m.num_input_meta_channels > 0 {
      testAssert!(SGFMetadata::METADATA_INPUT_NUM_CHANNELS == m.num_input_meta_channels);
    }

    let spatial_input = vec![0.0f32; (m.num_input_channels * nn_x_len * nn_y_len * max_batch_size) as usize];
    let global_input = vec![0.0f32; (m.num_input_global_channels * max_batch_size) as usize];
    let meta_input = if m.num_input_meta_channels > 0 {
      vec![0.0f32; (m.num_input_meta_channels * max_batch_size) as usize]
    } else {
      vec![0.0f32; 1]
    };
    BurnInputBuffers {
      max_batch_size,
      single_input_elts,
      single_input_global_elts,
      single_input_meta_elts,
      single_policy_pass_result_elts,
      single_policy_result_elts,
      single_value_result_elts,
      single_score_value_result_elts,
      single_ownership_result_elts,
      spatial_input,
      global_input,
      meta_input,
    }
  }
}

impl nninterface::InputBuffers for BurnInputBuffers {
  fn as_any_mut(&mut self) -> &mut dyn Any {
    self
  }
}

fn downcast_model(loaded_model: &dyn nninterface::LoadedModel) -> &BurnLoadedModel {
  loaded_model.as_any().downcast_ref::<BurnLoadedModel>().expect("LoadedModel created by BurnNeuralNet")
}

impl<N: Network + 'static> nninterface::NeuralNet for BurnNeuralNet<N> {
  /// upstream: NeuralNet::globalInitialize (eigenbackend.cpp)
  fn global_initialize(&self) {
    // no-op for cpu
  }

  /// upstream: NeuralNet::globalCleanup (eigenbackend.cpp)
  fn global_cleanup(&self) {
    // no-op for cpu
  }

  /// upstream: NeuralNet::printDevices (eigenbackend.cpp)
  fn print_devices(&self) {}

  /// upstream: NeuralNet::loadModelFile (eigenbackend.cpp)
  /// PORT NOTE: `model_bytes` are the network's `.bpk` weights; the ModelDesc is N's (build.rs,
  /// ONNX metadata). The `.bin.gz` sha256 cannot be checked against `.bpk` bytes.
  fn load_model_file(&self, file: &str, model_bytes: Vec<u8>, expected_sha256: &str) -> Result<Box<dyn nninterface::LoadedModel>, KataError> {
    if !expected_sha256.is_empty() {
      // TODO(port): expectedSha256 is the sha of the .bin.gz; the burn backend only sees .bpk weights.
      return Err(KataError::StringError(
        "Error loading or parsing model file ".to_string() + file + ": expectedSha256 is not supported by the burn backend",
      ));
    }
    let model_desc = N::model_desc();
    if model_desc.model_version > NNModelVersion::latestModelVersionImplemented {
      return Err(KataError::StringError(
        "This neural net requires a newer KataGo version. Model version: ".to_string() + &Global::int_to_string(model_desc.model_version),
      ));
    }
    Ok(Box::new(BurnLoadedModel { model_desc, bpk: model_bytes }))
  }

  /// upstream: NeuralNet::getModelDesc (eigenbackend.cpp)
  fn get_model_desc<'a>(&self, loaded_model: &'a dyn nninterface::LoadedModel) -> &'a ModelDesc {
    &downcast_model(loaded_model).model_desc
  }

  /// upstream: NeuralNet::createComputeContext (eigenbackend.cpp)
  fn create_compute_context(
    &self,
    gpu_idxs: &[i32],
    nn_x_len: i32,
    nn_y_len: i32,
    open_cl_tuner_file: &str,
    home_data_dir_override: &str,
    open_cl_re_tune_per_board_size: bool,
    use_fp16_mode: enabled_t,
    use_nhwc_mode: enabled_t,
    loaded_model: &dyn nninterface::LoadedModel,
  ) -> Result<Box<dyn nninterface::ComputeContext>, KataError> {
    let _ = gpu_idxs;
    let _ = open_cl_tuner_file;
    let _ = home_data_dir_override;
    let _ = open_cl_re_tune_per_board_size;
    let _ = loaded_model;

    let use_fp16: bool = if use_fp16_mode == enabled_t::True { true } else { false };
    // PORT NOTE: the generated graph is NCHW: Auto means NCHW here (Eigen: Auto means NHWC).
    let use_nhwc: bool = if use_nhwc_mode == enabled_t::True { true } else { false };

    if use_fp16 {
      return Err(KataError::StringError("Burn backend: useFP16 = true not supported".to_string()));
    }
    if use_nhwc {
      return Err(KataError::StringError("Burn backend: useNHWC = true not supported".to_string()));
    }
    // PORT NOTE: the generated graph has a fixed spatial size (bin2onnx --pos-len).
    if nn_x_len != N::POS_LEN || nn_y_len != N::POS_LEN {
      return Err(KataError::StringError(
        "Burn backend: nnXLen and nnYLen must equal the network's posLen ".to_string() + &Global::int_to_string(N::POS_LEN),
      ));
    }

    Ok(Box::new(BurnComputeContext { nn_x_len, nn_y_len, device: self.device.clone() }))
  }

  /// upstream: NeuralNet::createComputeHandle (eigenbackend.cpp)
  fn create_compute_handle(
    &self,
    context: &dyn nninterface::ComputeContext,
    loaded_model: &dyn nninterface::LoadedModel,
    max_batch_size: i32,
    require_exact_nn_len: bool,
    inputs_use_nhwc: bool,
    gpu_idx_for_this_thread: i32,
    server_thread_idx: i32,
  ) -> Result<Box<dyn nninterface::ComputeHandle>, KataError> {
    let context = context.as_any().downcast_ref::<BurnComputeContext>().expect("ComputeContext created by BurnNeuralNet");
    let loaded_model = downcast_model(loaded_model);
    log::info!(
      "{}",
      "Burn backend thread ".to_string() + &Global::int_to_string(server_thread_idx) + ": Model version " + &Global::int_to_string(loaded_model.model_desc.model_version)
    );
    log::info!("{}", "Burn backend thread ".to_string() + &Global::int_to_string(server_thread_idx) + ": Model name: " + &loaded_model.model_desc.name);

    let _ = require_exact_nn_len; //We don't bother with mask optimizations if we know exact sizes right now.
    let _ = gpu_idx_for_this_thread; //Doesn't matter
    let _ = max_batch_size;

    // PORT NOTE: NCHW only (Eigen: "inputsUseNHWC = false unsupported").
    if inputs_use_nhwc {
      return Err(KataError::StringError("Burn backend: inputsUseNHWC = true unsupported".to_string()));
    }
    let m = &loaded_model.model_desc;
    Ok(Box::new(BurnComputeHandle::<N> {
      nn_x_len: context.nn_x_len,
      nn_y_len: context.nn_y_len,
      inputs_use_nhwc,
      device: context.device.clone(),
      model: N::from_bpk(loaded_model.bpk.clone(), &context.device),
      batch_buckets: self.batch_buckets.clone(),
      model_version: m.model_version,
      num_input_channels: m.num_input_channels,
      num_policy_channels: m.num_policy_channels,
      num_value_channels: m.num_value_channels,
      num_score_value_channels: m.num_score_value_channels,
      num_ownership_channels: m.num_ownership_channels,
    }))
  }

  /// upstream: NeuralNet::isUsingFP16 (eigenbackend.cpp)
  fn is_using_fp16(&self, compute_handle: &dyn nninterface::ComputeHandle) -> bool {
    let _ = compute_handle;
    false
  }

  /// upstream: NeuralNet::createInputBuffers (eigenbackend.cpp)
  fn create_input_buffers(&self, loaded_model: &dyn nninterface::LoadedModel, max_batch_size: i32, nn_x_len: i32, nn_y_len: i32) -> Box<dyn nninterface::InputBuffers> {
    Box::new(BurnInputBuffers::new(downcast_model(loaded_model), max_batch_size, nn_x_len, nn_y_len))
  }

  /// upstream: NeuralNet::getOutput (eigenbackend.cpp)
  fn get_output<'a>(
    &'a self,
    compute_handle: &'a mut dyn nninterface::ComputeHandle,
    buffers: &'a mut dyn nninterface::InputBuffers,
    num_batch_elts_filled: i32,
    input_bufs: &'a [Rc<NNResultBuf>],
    outputs: &'a mut [NNOutput],
  ) -> LocalBoxFuture<'a, Result<(), KataError>> {
    Box::pin(async move {
      let compute_handle = compute_handle.as_any_mut().downcast_mut::<BurnComputeHandle<N>>().expect("ComputeHandle created by BurnNeuralNet");
      let input_buffers = buffers.as_any_mut().downcast_mut::<BurnInputBuffers>().expect("InputBuffers created by BurnNeuralNet");
      get_output_impl(compute_handle, input_buffers, num_batch_elts_filled, input_bufs, outputs).await
    })
  }
}

/// upstream: NeuralNet::getOutput (eigenbackend.cpp)
async fn get_output_impl<N: Network>(
  compute_handle: &mut BurnComputeHandle<N>,
  input_buffers: &mut BurnInputBuffers,
  num_batch_elts_filled: i32,
  input_bufs: &[Rc<NNResultBuf>],
  outputs: &mut [NNOutput],
) -> Result<(), KataError> {
  debug_assert!(num_batch_elts_filled <= input_buffers.max_batch_size);
  debug_assert!(num_batch_elts_filled > 0);
  let batch_size: i32 = num_batch_elts_filled;
  let nn_x_len: i32 = compute_handle.nn_x_len;
  let nn_y_len: i32 = compute_handle.nn_y_len;
  let model_version: i32 = compute_handle.model_version;

  let num_spatial_features: i32 = NNModelVersion::get_num_spatial_features(model_version)?;
  let num_global_features: i32 = NNModelVersion::get_num_global_features(model_version)?;
  let num_meta_features: i32 = input_buffers.single_input_meta_elts as i32;
  debug_assert!(num_spatial_features == compute_handle.num_input_channels);
  debug_assert!((num_spatial_features * nn_x_len * nn_y_len) as usize == input_buffers.single_input_elts);
  debug_assert!(num_global_features as usize == input_buffers.single_input_global_elts);
  let num_policy_channels: i32 = compute_handle.num_policy_channels;

  // PORT NOTE: upstream reads `inputBufs[row]->symmetry` / `->policyOptimism` in the output loop
  // below; here that loop runs after an await, so both are read now, while the input is copied
  // (same values: nothing writes them while the eval is queued, PORTING.md §11).
  let mut symmetry_by_row: Vec<i32> = Vec::with_capacity(batch_size as usize);
  let mut policy_optimism_by_row: Vec<f32> = Vec::with_capacity(batch_size as usize);
  for n_idx in 0..batch_size as usize {
    symmetry_by_row.push(input_bufs[n_idx].symmetry.get());
    policy_optimism_by_row.push(input_bufs[n_idx].policy_optimism.get() as f32);
  }

  for n_idx in 0..batch_size as usize {
    let single_input_elts = input_buffers.single_input_elts;
    let single_input_global_elts = input_buffers.single_input_global_elts;
    let single_input_meta_elts = input_buffers.single_input_meta_elts;
    let row_spatial_input: &mut [f32] = &mut input_buffers.spatial_input[single_input_elts * n_idx..single_input_elts * (n_idx + 1)];
    let row_global_input: &mut [f32] = &mut input_buffers.global_input[single_input_global_elts * n_idx..single_input_global_elts * (n_idx + 1)];
    let row_meta_input: &mut [f32] = &mut input_buffers.meta_input[single_input_meta_elts * n_idx..];

    let row_global = input_bufs[n_idx].row_global_buf.borrow();
    let row_spatial = input_bufs[n_idx].row_spatial_buf.borrow();
    let row_meta = input_bufs[n_idx].row_meta_buf.borrow();
    let has_row_meta: bool = input_bufs[n_idx].has_row_meta.get();
    row_global_input[..num_global_features as usize].copy_from_slice(&row_global[..num_global_features as usize]);
    if num_meta_features > 0 {
      testAssert!(!row_meta.is_empty());
      testAssert!(has_row_meta);
      row_meta_input[..num_meta_features as usize].copy_from_slice(&row_meta[..num_meta_features as usize]);
    } else {
      testAssert!(!has_row_meta);
    }
    SymmetryHelpers::copy_inputs_with_symmetry(
      &row_spatial[..],
      row_spatial_input,
      1,
      nn_y_len,
      nn_x_len,
      num_spatial_features,
      compute_handle.inputs_use_nhwc,
      symmetry_by_row[n_idx],
    );
  }

  if num_meta_features > 0 {
    // TODO(port): human-SL nets (SGF metadata encoder input) are not supported by tools/bin2onnx,
    // so no generated network takes a meta input.
    return Err(KataError::StringError("Burn backend: networks with SGF metadata inputs are not supported".to_string()));
  }

  // PORT NOTE: Eigen maps the buffers as tensors and calls Model::apply; here the batch is copied
  // into burn tensors (NCHW) and the generated network runs, padded to a batch bucket if configured
  // (rows n..n_run of the input buffers hold stale inputs; their outputs are not read).
  let n = batch_size as usize;
  let n_run = padded_batch_size(&compute_handle.batch_buckets, batch_size, input_buffers.max_batch_size) as usize;
  let hw = (nn_x_len * nn_y_len) as usize;
  let device = &compute_handle.device;
  let input = Tensor::<4>::from_data(
    TensorData::new(
      input_buffers.spatial_input[..input_buffers.single_input_elts * n_run].to_vec(),
      [n_run, num_spatial_features as usize, nn_y_len as usize, nn_x_len as usize],
    ),
    device,
  );
  let input_global = Tensor::<2>::from_data(
    TensorData::new(input_buffers.global_input[..input_buffers.single_input_global_elts * n_run].to_vec(), [n_run, num_global_features as usize]),
    device,
  );
  let raw = compute_handle.model.forward(input, input_global);

  // PORT NOTE: one readback for all four outputs (one GPU->CPU round trip per batch).
  let policy_row_len = num_policy_channels as usize * (hw + 1);
  let num_value_channels = compute_handle.num_value_channels as usize;
  let num_score_value_channels = compute_handle.num_score_value_channels as usize;
  let ownership_row_len = compute_handle.num_ownership_channels as usize * hw;
  let flat = Tensor::cat(
    vec![
      raw.policy.reshape([n_run, policy_row_len]),
      raw.value.reshape([n_run, num_value_channels]),
      raw.score.reshape([n_run, num_score_value_channels]),
      raw.ownership.reshape([n_run, ownership_row_len]),
    ],
    1,
  );
  let data = flat
    .into_data_async()
    .await
    .map_err(|e| KataError::StringError(format!("Burn backend: readback failed: {e:?}")))?;
  let data: Vec<f32> = data.try_into_vec::<f32>().map_err(|e| KataError::StringError(format!("Burn backend: readback failed: {e:?}")))?;
  // PORT NOTE: the CPU backend computes synchronously; yield so the other virtual threads run
  // before the results are delivered (PORTING.md §11). On wasm32 the yield is a macrotask (a
  // resolved-by-setTimeout Promise), so the host's event loop (timers, worker messages, UI) runs
  // between batches; natively the LocalPool has no outside events and a plain yield is enough.
  #[cfg(not(target_arch = "wasm32"))]
  yield_now().await;
  #[cfg(target_arch = "wasm32")]
  crate::host::yield_to_event_loop().await;
  let row_len = policy_row_len + num_value_channels + num_score_value_channels + ownership_row_len;
  testAssert!(data.len() == row_len * n_run);

  debug_assert!(input_buffers.single_policy_pass_result_elts == num_policy_channels as usize);
  debug_assert!(input_buffers.single_policy_result_elts == (num_policy_channels as usize) * hw);

  debug_assert!(outputs.len() == n);

  let mut policy_probs_tmp = [0.0f32; NNPos::MAX_NN_POLICY_SIZE as usize];

  for row in 0..n {
    let output: &mut NNOutput = &mut outputs[row];
    debug_assert!(output.nn_x_len == nn_x_len);
    debug_assert!(output.nn_y_len == nn_y_len);
    let policy_optimism: f32 = policy_optimism_by_row[row];
    let symmetry: i32 = symmetry_by_row[row];

    let row_data: &[f32] = &data[row * row_len..(row + 1) * row_len];
    // PORT NOTE: channel c of the policy is row_data[c*(hw+1) .. c*(hw+1)+hw], its pass logit at
    // c*(hw+1)+hw (Eigen: policySrcBuf[i*numPolicyChannels+c] and policyPassSrcBuf[c]).
    let policy_src_buf: &[f32] = &row_data[..policy_row_len];
    let policy_pass_src_buf = |c: usize| -> f32 { policy_src_buf[c * (hw + 1) + hw] };
    let value_data: &[f32] = &row_data[policy_row_len..policy_row_len + num_value_channels];
    let score_value_data: &[f32] = &row_data[policy_row_len + num_value_channels..policy_row_len + num_value_channels + num_score_value_channels];
    let ownership_src_buf: &[f32] = &row_data[policy_row_len + num_value_channels + num_score_value_channels..];

    // These are in logits, the client does the postprocessing to turn them into
    // policy probabilities and white game outcome probabilities
    // Also we don't fill in the nnHash here either
    // Handle version >= 12 policy optimism
    if num_policy_channels == 2 || (num_policy_channels == 4 && model_version >= 16) {
      for i in 0..hw {
        let p: f32 = policy_src_buf[i];
        let p_opt: f32 = policy_src_buf[(hw + 1) + i];
        policy_probs_tmp[i] = p + (p_opt - p) * policy_optimism;
      }
      SymmetryHelpers::copy_outputs_with_symmetry(&policy_probs_tmp, &mut output.policy_probs, 1, nn_y_len, nn_x_len, symmetry);
      output.policy_probs[hw] = policy_pass_src_buf(0) + (policy_pass_src_buf(1) - policy_pass_src_buf(0)) * policy_optimism;
    } else {
      debug_assert!(num_policy_channels == 1);
      SymmetryHelpers::copy_outputs_with_symmetry(&policy_src_buf[..hw], &mut output.policy_probs, 1, nn_y_len, nn_x_len, symmetry);
      output.policy_probs[input_buffers.single_policy_result_elts] = policy_pass_src_buf(0);
    }

    let num_value_channels = compute_handle.num_value_channels;
    debug_assert!(num_value_channels == 3);
    output.white_win_prob = value_data[0];
    output.white_loss_prob = value_data[1];
    output.white_no_result_prob = value_data[2];

    //As above, these are NOT actually from white's perspective, but rather the player to move.
    //As usual the client does the postprocessing.
    if let Some(white_owner_map) = output.white_owner_map.as_mut() {
      debug_assert!(compute_handle.num_ownership_channels == 1);
      SymmetryHelpers::copy_outputs_with_symmetry(&ownership_src_buf[..hw], white_owner_map, 1, nn_y_len, nn_x_len, symmetry);
    }

    if model_version >= 9 {
      let num_score_value_channels = compute_handle.num_score_value_channels;
      debug_assert!(num_score_value_channels == 6);
      output.white_score_mean = score_value_data[0];
      output.white_score_mean_sq = score_value_data[1];
      output.white_lead = score_value_data[2];
      output.var_time_left = score_value_data[3];
      output.shortterm_winloss_error = score_value_data[4];
      output.shortterm_score_error = score_value_data[5];
    } else if model_version >= 8 {
      let num_score_value_channels = compute_handle.num_score_value_channels;
      debug_assert!(num_score_value_channels == 4);
      output.white_score_mean = score_value_data[0];
      output.white_score_mean_sq = score_value_data[1];
      output.white_lead = score_value_data[2];
      output.var_time_left = score_value_data[3];
      output.shortterm_winloss_error = 0.0;
      output.shortterm_score_error = 0.0;
    } else if model_version >= 4 {
      let num_score_value_channels = compute_handle.num_score_value_channels;
      debug_assert!(num_score_value_channels == 2);
      output.white_score_mean = score_value_data[0];
      output.white_score_mean_sq = score_value_data[1];
      output.white_lead = output.white_score_mean;
      output.var_time_left = 0.0;
      output.shortterm_winloss_error = 0.0;
      output.shortterm_score_error = 0.0;
    } else if model_version >= 3 {
      let num_score_value_channels = compute_handle.num_score_value_channels;
      debug_assert!(num_score_value_channels == 1);
      output.white_score_mean = score_value_data[0];
      //Version 3 neural nets don't have any second moment output, implicitly already folding it in, so we just use the mean squared
      output.white_score_mean_sq = output.white_score_mean * output.white_score_mean;
      output.white_lead = output.white_score_mean;
      output.var_time_left = 0.0;
      output.shortterm_winloss_error = 0.0;
      output.shortterm_score_error = 0.0;
    } else {
      unreachable!();
    }
  }
  Ok(())
}
