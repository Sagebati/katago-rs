//! The wasm32 engine: see the crate docs (lib.rs).

use std::cell::Cell;
use std::cell::RefCell;
use std::future::Future;
use std::pin::Pin;
use std::rc::Rc;
use std::task::Context;
use std::task::Poll;

use burn::prelude::Device;
use futures::StreamExt;
use futures::channel::mpsc;
use js_sys::Function;
use js_sys::Promise;
use js_sys::Reflect;
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;

use katago_core::command::analysis::AnalysisArgs;
use katago_core::command::analysis::AnalysisHost;
use katago_core::command::analysis::LoggerSettings;
use katago_core::core::commontypes::enabled_t;
use katago_core::core::global::KataError;
use katago_core::neuralnet::desc::ModelDesc;
use katago_core::neuralnet::nneval::NNResultBuf;
use katago_core::neuralnet::nninputs::NNOutput;
use katago_core::neuralnet::nninterface;
use katago_core::neuralnet::nninterface::NeuralNet;
use katago_core::port::executor::LocalBoxFuture;
use katago_core::port::executor::Spawner;
use katago_core::program::setup::NNEvaluatorHost;

use crate::config::Backend;
use crate::review;
use crate::sha256_hex;

unsafe extern "C" {
  /// The static constructors, synthesized by wasm-ld (`inventory` registrations of pliron, pulled in by
  /// cubecl-core).
  fn __wasm_call_ctors();
}

/// Runs the static constructors once, at instantiation (wasm-bindgen calls this from `__wbindgen_start`).
///
/// Referencing `__wasm_call_ctors` from code is also what makes wasm-ld link the module as a reactor:
/// without a reference it treats a module that has constructors as a command and wraps every export
/// (`__wbindgen_malloc`, `__externref_table_alloc`, closure trampolines...) in a call to all of them, so
/// every JS -> wasm call re-ran cubecl's `inventory` registrations (wgpu's WebGPU backend crosses into
/// wasm for each bind group, promise and buffer mapping): measured 8-10% of each NN batch and ~5% of the
/// warm-up; see docs/M4-report.md, "Start-up on WebGPU".
#[wasm_bindgen(start)]
fn start() {
  // SAFETY: called once, first thing after instantiation (before any other export runs Rust code that
  // reads the registries), as the reactor model expects; the constructors only register statics.
  unsafe { __wasm_call_ctors() };
  console_error_panic_hook::set_once();
}

//------------------------------------------------------------------------------------------------
// Clock, timed waits, executor

#[wasm_bindgen]
extern "C" {
  #[wasm_bindgen(js_name = setTimeout)]
  fn set_timeout_ms(handler: &Function, timeout_ms: f64) -> JsValue;
  #[wasm_bindgen(js_name = clearTimeout)]
  fn clear_timeout(handle: &JsValue);
}

thread_local! {
  static PERFORMANCE: web_sys::Performance = Reflect::get(&js_sys::global(), &JsValue::from_str("performance"))
    .expect("globalThis.performance")
    .unchecked_into::<web_sys::Performance>();
}

fn now_ms() -> f64 {
  PERFORMANCE.with(|p| p.now())
}

/// `core::timer::set_clock`: seconds from `performance.now()` (monotonic, like steady_clock).
fn clock() -> f64 {
  now_ms() / 1000.0
}

/// The future of [`sleep`]: a `setTimeout`-resolved Promise; the timer is cleared if the wait is dropped
/// first (AsyncBot's callback loop drops its wait when the search ends, with periods up to 1e6 s).
struct Sleep {
  fut: JsFuture,
  handle: JsValue,
  done: bool,
}

impl Future for Sleep {
  type Output = ();
  fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
    match Pin::new(&mut self.fut).poll(cx) {
      Poll::Ready(_) => {
        self.done = true;
        Poll::Ready(())
      }
      Poll::Pending => Poll::Pending,
    }
  }
}

impl Drop for Sleep {
  fn drop(&mut self) {
    if !self.done {
      clear_timeout(&self.handle);
    }
  }
}

/// `port::executor::set_sleep`: a timed wait on the worker's event loop. `setTimeout` takes at most
/// 2^31-1 ms (~24.8 days; larger delays fire at once), so longer waits are capped there.
fn sleep(seconds: f64) -> LocalBoxFuture<'static, ()> {
  let ms = if seconds.is_finite() && seconds > 0.0 { (seconds * 1000.0).min(2147483647.0) } else { 0.0 };
  let mut handle = JsValue::UNDEFINED;
  let promise = Promise::new(&mut |resolve: Function, _reject: Function| {
    handle = set_timeout_ms(&resolve, ms);
  });
  Box::pin(Sleep { fut: JsFuture::from(promise), handle, done: false })
}

/// The browser's local executor: the worker's microtask queue.
struct WasmSpawner;

impl Spawner for WasmSpawner {
  fn spawn_local(&self, f: LocalBoxFuture<'static, ()>) {
    wasm_bindgen_futures::spawn_local(f);
  }
}

//------------------------------------------------------------------------------------------------
// Logging (core/logger.cpp is the log crate; this is its browser backend)

thread_local! {
  /// `(kind, text)` callback of the current engine: kind is "error" | "warn" | "info" for log lines and
  /// "stderr" for `cerr` output.
  static LOG_FN: RefCell<Option<Function>> = const { RefCell::new(None) };
}

struct WebLogger;

static WEB_LOGGER: WebLogger = WebLogger;

impl log::Log for WebLogger {
  /// The engine's own lines (katago_core, katago_nn_burn, katago_wasm) at info; dependencies (wgpu,
  /// cubecl, burn) only from warn.
  fn enabled(&self, metadata: &log::Metadata) -> bool {
    if metadata.target().starts_with("katago") { metadata.level() <= log::Level::Info } else { metadata.level() <= log::Level::Warn }
  }
  fn log(&self, record: &log::Record) {
    if !self.enabled(record.metadata()) {
      return;
    }
    let kind = match record.level() {
      log::Level::Error => "error",
      log::Level::Warn => "warn",
      _ => "info",
    };
    emit_log(kind, &record.args().to_string());
  }
  fn flush(&self) {}
}

fn emit_log(kind: &str, text: &str) {
  LOG_FN.with_borrow(|f| match f {
    Some(f) => {
      let _ = f.call2(&JsValue::NULL, &JsValue::from_str(kind), &JsValue::from_str(text));
    }
    None => web_sys::console::log_2(&JsValue::from_str(kind), &JsValue::from_str(text)),
  });
}

/// Clock, timed wait and log backend: once per wasm instance.
fn install_globals() {
  katago_core::core::timer::set_clock(clock);
  katago_core::port::executor::set_sleep(sleep);
  if log::set_logger(&WEB_LOGGER).is_ok() {
    log::set_max_level(log::LevelFilter::Info);
  }
}

//------------------------------------------------------------------------------------------------
// NN backend

/// The networks compiled into katago-nn-burn (crate features).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Net {
  #[cfg(feature = "net-b6c96")]
  B6c96,
  #[cfg(feature = "net-b10c128")]
  B10c128,
  #[cfg(feature = "net-b18c384nbt")]
  B18c384nbt,
}

impl Net {
  fn parse(name: &str) -> Option<Net> {
    match name {
      #[cfg(feature = "net-b6c96")]
      "g170-b6c96" => Some(Net::B6c96),
      #[cfg(feature = "net-b10c128")]
      "g170e-b10c128" => Some(Net::B10c128),
      #[cfg(feature = "net-b18c384nbt")]
      "kata1-b18c384nbt" => Some(Net::B18c384nbt),
      _ => None,
    }
  }

  fn backend(self, device: Device, batch_buckets: Vec<i32>) -> Rc<dyn NeuralNet> {
    use katago_nn_burn::backend::BurnNeuralNet;
    match self {
      #[cfg(feature = "net-b6c96")]
      Net::B6c96 => Rc::new(BurnNeuralNet::<katago_nn_burn::b6c96::Model>::new(device).with_batch_buckets(batch_buckets)),
      #[cfg(feature = "net-b10c128")]
      Net::B10c128 => Rc::new(BurnNeuralNet::<katago_nn_burn::b10c128::Model>::new(device).with_batch_buckets(batch_buckets)),
      #[cfg(feature = "net-b18c384nbt")]
      Net::B18c384nbt => Rc::new(BurnNeuralNet::<katago_nn_burn::b18c384nbt::Model>::new(device).with_batch_buckets(batch_buckets)),
    }
  }
}

thread_local! {
  /// cubecl registers one runtime per device: initialise WebGPU once per worker.
  static WEBGPU_DEVICE: RefCell<Option<Device>> = const { RefCell::new(None) };
}

async fn make_device(backend: Backend) -> Result<Device, JsError> {
  match backend {
    Backend::WebGpu => {
      if let Some(d) = WEBGPU_DEVICE.with_borrow(|d| d.clone()) {
        return Ok(d);
      }
      // Before the first kernel: compiled WGSL is recorded, and looked up in what kernelCacheImport loaded.
      cubecl_wgpu::wgsl_cache::enable();
      let d = Device::wgpu_options().init_async().await.map_err(|e| JsError::new(&format!("WebGPU init failed: {e:?}")))?;
      WEBGPU_DEVICE.set(Some(d.clone()));
      Ok(d)
    }
    Backend::Cpu => Ok(Device::flex()),
  }
}

/// NN timings, read by [`Engine::stats`].
#[derive(Default)]
struct NnStats {
  batches: Cell<u64>,
  rows: Cell<u64>,
  /// sum of the getOutput durations (ms): forward pass, readback and the event-loop yield after it
  busy_ms: Cell<f64>,
  last_batch_ms: Cell<f64>,
  max_batch_ms: Cell<f64>,
  last_batch_rows: Cell<u64>,
  max_batch_rows: Cell<u64>,
  /// createComputeHandle (the burn module built from the `.bpk`, weights uploaded to the device)
  model_build_ms: Cell<f64>,
  /// the batch-shape warm-up before the first batch (WebGPU)
  warmup_ms: Cell<f64>,
  /// warm-up batches run
  warmup_batches: Cell<u64>,
  /// `batches` when the last input lines arrived (sendLines): batches completed since then = the
  /// batches the engine still ran after, e.g., a terminate
  batches_at_last_input: Cell<u64>,
  rows_at_last_input: Cell<u64>,
}

/// The backend with timing around getOutput and createComputeHandle; everything else is delegated.
/// With `warmup_buckets`, the first getOutput first runs one batch of every bucket size (up to the
/// input buffers' max batch size), so each GPU input shape is compiled and autotuned before the search
/// relies on its timing.
struct StatsNeuralNet {
  inner: Rc<dyn NeuralNet>,
  stats: Rc<NnStats>,
  warmup_buckets: Vec<i32>,
  warmed_up: Cell<bool>,
  max_batch_size: Cell<i32>,
}

impl NeuralNet for StatsNeuralNet {
  fn global_initialize(&self) {
    self.inner.global_initialize()
  }
  fn global_cleanup(&self) {
    self.inner.global_cleanup()
  }
  fn print_devices(&self) {
    self.inner.print_devices()
  }
  fn load_model_file(&self, file: &str, model_bytes: Vec<u8>, expected_sha256: &str) -> Result<Box<dyn nninterface::LoadedModel>, KataError> {
    self.inner.load_model_file(file, model_bytes, expected_sha256)
  }
  fn get_model_desc<'a>(&self, loaded_model: &'a dyn nninterface::LoadedModel) -> &'a ModelDesc {
    self.inner.get_model_desc(loaded_model)
  }
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
    self.inner.create_compute_context(
      gpu_idxs,
      nn_x_len,
      nn_y_len,
      open_cl_tuner_file,
      home_data_dir_override,
      open_cl_re_tune_per_board_size,
      use_fp16_mode,
      use_nhwc_mode,
      loaded_model,
    )
  }
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
    let t0 = now_ms();
    let r = self.inner.create_compute_handle(
      context,
      loaded_model,
      max_batch_size,
      require_exact_nn_len,
      inputs_use_nhwc,
      gpu_idx_for_this_thread,
      server_thread_idx,
    );
    self.stats.model_build_ms.set(self.stats.model_build_ms.get() + (now_ms() - t0));
    r
  }
  fn is_using_fp16(&self, compute_handle: &dyn nninterface::ComputeHandle) -> bool {
    self.inner.is_using_fp16(compute_handle)
  }
  fn create_input_buffers(&self, loaded_model: &dyn nninterface::LoadedModel, max_batch_size: i32, nn_x_len: i32, nn_y_len: i32) -> Box<dyn nninterface::InputBuffers> {
    self.max_batch_size.set(max_batch_size);
    self.inner.create_input_buffers(loaded_model, max_batch_size, nn_x_len, nn_y_len)
  }
  fn get_output<'a>(
    &'a self,
    compute_handle: &'a mut dyn nninterface::ComputeHandle,
    buffers: &'a mut dyn nninterface::InputBuffers,
    num_batch_elts_filled: i32,
    input_bufs: &'a [Rc<NNResultBuf>],
    outputs: &'a mut [NNOutput],
  ) -> LocalBoxFuture<'a, Result<(), KataError>> {
    Box::pin(async move {
      if !self.warmed_up.get() && num_batch_elts_filled > 0 {
        self.warmed_up.set(true);
        let t0 = now_ms();
        // Rows of this batch's first input, repeated: the backend only reads input_bufs, and the
        // warm-up outputs are thrown away.
        for &b in self.warmup_buckets.iter().filter(|&&b| b <= self.max_batch_size.get()) {
          let bufs: Vec<Rc<NNResultBuf>> = (0..b).map(|_| input_bufs[0].clone()).collect();
          let mut outs: Vec<NNOutput> = (0..b).map(|_| outputs[0].clone()).collect();
          let tb = now_ms();
          self.inner.get_output(&mut *compute_handle, &mut *buffers, b, &bufs, &mut outs).await?;
          self.stats.warmup_batches.set(self.stats.warmup_batches.get() + 1);
          log::info!("katago-wasm: warm-up batch of {b}: {:.0} ms", now_ms() - tb);
        }
        self.stats.warmup_ms.set(now_ms() - t0);
      }
      let t0 = now_ms();
      let r = self.inner.get_output(compute_handle, buffers, num_batch_elts_filled, input_bufs, outputs).await;
      let dt = now_ms() - t0;
      let s = &self.stats;
      let rows = num_batch_elts_filled.max(0) as u64;
      s.batches.set(s.batches.get() + 1);
      s.rows.set(s.rows.get() + rows);
      s.busy_ms.set(s.busy_ms.get() + dt);
      s.last_batch_ms.set(dt);
      s.last_batch_rows.set(rows);
      if dt > s.max_batch_ms.get() {
        s.max_batch_ms.set(dt);
      }
      if rows > s.max_batch_rows.get() {
        s.max_batch_rows.set(rows);
      }
      r
    })
  }
}

/// Setup's NNEvaluatorHost: one network with its weights handed over by the worker.
struct WebNNHost {
  device: Device,
  net: Net,
  bpk: RefCell<Option<Vec<u8>>>,
  stats: Rc<NnStats>,
  /// Batch-shape buckets (WebGPU), also warmed up before the first batch; empty on the CPU.
  batch_buckets: Vec<i32>,
}

/// WebGPU batch buckets: half the max batch size and the max (16 and 32 by default). Measured in Chrome
/// (b10c128, RX 6800 XT): a batch of 1 / 8 / 16 / 32 runs in ~21 / 22 / 27 / 43 ms, but warming up a
/// shape costs ~1.2 s for 16 or 32 and 3-6 s for 1-8 (beyond the ~7 s the very first batch spends
/// compiling every kernel), so padding small batches is far cheaper than compiling their shapes.
fn webgpu_batch_buckets(max_batch_size: i32) -> Vec<i32> {
  let max = max_batch_size.max(1);
  if max >= 4 { vec![(max + 1) / 2, max] } else { vec![max] }
}

impl NNEvaluatorHost for WebNNHost {
  fn backend_prefix(&self) -> String {
    "burn".to_string()
  }
  fn global_initialize(&self) {}
  fn global_cleanup(&self) {}
  fn load_model(&self, _nn_model_name: &str, nn_model_file: &str) -> Result<(Rc<dyn NeuralNet>, Vec<u8>), KataError> {
    let bytes = self.bpk.borrow_mut().take().ok_or_else(|| {
      KataError::StringError(format!(
        "Error loading or parsing model file {nn_model_file}: katago-wasm loads one network per engine (no human-SL model, no second evaluator)"
      ))
    })?;
    let inner = self.net.backend(self.device.clone(), self.batch_buckets.clone());
    Ok((
      Rc::new(StatsNeuralNet {
        inner,
        stats: self.stats.clone(),
        warmup_buckets: self.batch_buckets.clone(),
        warmed_up: Cell::new(false),
        max_batch_size: Cell::new(0),
      }),
      bytes,
    ))
  }
  fn spawner(&self) -> Rc<dyn Spawner> {
    Rc::new(WasmSpawner)
  }
}

//------------------------------------------------------------------------------------------------
// Compiled-kernel cache

/// Loads compiled WebGPU kernels saved by [`kernel_cache_export`] in an earlier page load of the same build
/// (the worker keys them by build and adapter). Returns how many kernels the blob held. Call before
/// [`Engine::create`]: each kernel found there skips cubecl's expansion, optimization and WGSL printing, the
/// bulk of the WebGPU warm-up (docs/M4-report.md, "Start-up on WebGPU").
#[wasm_bindgen(js_name = kernelCacheImport)]
pub fn kernel_cache_import(bytes: &[u8]) -> Result<u32, JsError> {
  cubecl_wgpu::wgsl_cache::import(bytes).map(|n| n as u32).map_err(|e| JsError::new(&e))
}

/// Every compiled WebGPU kernel of this instance (imported or compiled here) as one blob for
/// [`kernel_cache_import`]; resets `added` in [`kernel_cache_stats`].
#[wasm_bindgen(js_name = kernelCacheExport)]
pub fn kernel_cache_export() -> Vec<u8> {
  cubecl_wgpu::wgsl_cache::export()
}

/// `{entries, added, hits, misses}`: kernels held, kernels compiled since the last export, and the lookups
/// that found / did not find a kernel (a miss compiles it).
#[wasm_bindgen(js_name = kernelCacheStats)]
pub fn kernel_cache_stats() -> JsValue {
  let (entries, added, hits, misses) = cubecl_wgpu::wgsl_cache::stats();
  let o = js_sys::Object::new();
  for (k, v) in [("entries", entries as f64), ("added", added as f64), ("hits", hits as f64), ("misses", misses as f64)] {
    let _ = Reflect::set(&o, &JsValue::from_str(k), &JsValue::from_f64(v));
  }
  o.into()
}

//------------------------------------------------------------------------------------------------
// Options

fn get(options: &JsValue, key: &str) -> Result<Option<JsValue>, JsError> {
  if options.is_undefined() || options.is_null() {
    return Ok(None);
  }
  let v = Reflect::get(options, &JsValue::from_str(key)).map_err(|_| JsError::new(&format!("cannot read option {key}")))?;
  Ok(if v.is_undefined() || v.is_null() { None } else { Some(v) })
}

fn get_string(options: &JsValue, key: &str) -> Result<Option<String>, JsError> {
  match get(options, key)? {
    None => Ok(None),
    Some(v) => v.as_string().map(Some).ok_or_else(|| JsError::new(&format!("option {key} must be a string"))),
  }
}

/// `overrides`: a "key=value, key=value" string or an array of them (applied in order).
fn get_overrides(options: &JsValue) -> Result<Vec<String>, JsError> {
  match get(options, "overrides")? {
    None => Ok(Vec::new()),
    Some(v) => {
      if let Some(s) = v.as_string() {
        return Ok(vec![s]);
      }
      if js_sys::Array::is_array(&v) {
        let arr: js_sys::Array = v.unchecked_into();
        return arr.iter().map(|x| x.as_string().ok_or_else(|| JsError::new("option overrides: array entries must be strings"))).collect();
      }
      Err(JsError::new("option overrides must be a string or an array of strings"))
    }
  }
}

fn deferred() -> (Promise, Function, Function) {
  let mut resolve: Option<Function> = None;
  let mut reject: Option<Function> = None;
  let promise = Promise::new(&mut |res: Function, rej: Function| {
    resolve = Some(res);
    reject = Some(rej);
  });
  (promise, resolve.expect("Promise executor runs synchronously"), reject.expect("Promise executor runs synchronously"))
}

//------------------------------------------------------------------------------------------------
// Engine

/// The analysis engine (`katago analysis`) running on the worker's event loop.
#[wasm_bindgen]
pub struct Engine {
  input: RefCell<Option<mpsc::UnboundedSender<Vec<Vec<u8>>>>>,
  /// Response lines of the review queries (taken out of the output before `on_line`).
  router: review::Router,
  /// Reviews in flight, by the caller's review id.
  reviews: Rc<RefCell<std::collections::HashMap<String, Rc<review::ReviewState>>>>,
  backend: Backend,
  network: String,
  config_text: String,
  stats: Rc<NnStats>,
  finished: Promise,
}

#[wasm_bindgen]
impl Engine {
  /// Creates the device and starts the analysis engine. `options` (a plain object):
  /// - `backend`: "webgpu" | "cpu" (required; the worker resolves "auto");
  /// - `network`: one of [`crate::compiled_networks`] (required); `model` are its `.bpk` weights;
  /// - `expectedSha256`: if given, `model` must hash to it (lowercase hex) or creation fails;
  /// - `modelFile`: the `-model` name the engine reports (default: the network name);
  /// - `config`: a full config text replacing analysis_example.cfg (then the browser defaults are not applied);
  /// - `overrides`: "key=value, ..." string(s), as `-override-config`;
  /// - `analysisThreads`: as `-analysis-threads`;
  /// - `quitWithoutWaiting`: as `-quit-without-waiting` (what [`Engine::close_input`] does with open queries).
  ///
  /// `on_line(line)` receives every response line; `on_log(kind, text)` the log ("error" | "warn" | "info")
  /// and stderr ("stderr"). The model is loaded asynchronously after this returns: the first response
  /// (e.g. to a `query_version` action) arrives once the engine is ready.
  pub async fn create(options: JsValue, model: Vec<u8>, on_line: Function, on_log: Function) -> Result<Engine, JsError> {
    install_globals();
    let backend_name = get_string(&options, "backend")?.ok_or_else(|| JsError::new("option backend is required (webgpu | cpu)"))?;
    let backend = Backend::parse(&backend_name).ok_or_else(|| JsError::new(&format!("unknown backend {backend_name} (webgpu | cpu)")))?;
    let network = get_string(&options, "network")?.ok_or_else(|| JsError::new("option network is required"))?;
    let net = Net::parse(&network)
      .ok_or_else(|| JsError::new(&format!("network {network} is not compiled in (this build has: {})", crate::compiled_networks().join(", "))))?;
    if let Some(expected) = get_string(&options, "expectedSha256")? {
      let got = sha256_hex(&model);
      if got != expected.to_ascii_lowercase() {
        return Err(JsError::new(&format!("{network} weights: sha256 {got} does not match the expected {expected}")));
      }
    }
    let model_file = get_string(&options, "modelFile")?.unwrap_or_else(|| network.clone());
    let base = get_string(&options, "config")?;
    let overrides = get_overrides(&options)?;
    let analysis_threads: Option<i32> = match get(&options, "analysisThreads")? {
      None => None,
      Some(v) => Some(v.as_f64().ok_or_else(|| JsError::new("option analysisThreads must be a number"))? as i32),
    };
    let quit_without_waiting = get(&options, "quitWithoutWaiting")?.map(|v| v.is_truthy()).unwrap_or(false);

    let (cfg, applied) = crate::config::build_config(base.as_deref(), backend, &overrides).map_err(|e| JsError::new(e.what()))?;
    let config_text = cfg.get_all_key_vals();
    let batch_buckets = match backend {
      Backend::WebGpu => {
        // nnMaxBatchSize as Setup reads it (from a copy: reading marks the key used; the analysis engine
        // reports a bad value itself). Without it Setup derives one from the thread counts: powers of two
        // up to 256 then (the backend never pads past the input buffers' capacity).
        let max_batch_size = match katago_core::core::config_parser::ConfigParser::new_copy(&cfg) {
          Ok(mut c) if c.contains("nnMaxBatchSize") => c.get_int("nnMaxBatchSize", 1, 65536).unwrap_or(256),
          _ => 256,
        };
        webgpu_batch_buckets(max_batch_size)
      }
      Backend::Cpu => Vec::new(),
    };
    let device = make_device(backend).await?;

    LOG_FN.set(Some(on_log.clone()));
    let (tx, rx) = mpsc::unbounded::<Vec<Vec<u8>>>();
    let stats = Rc::new(NnStats::default());
    let (finished, resolve, reject) = deferred();
    let nn_host = WebNNHost { device, net, bpk: RefCell::new(Some(model)), stats: stats.clone(), batch_buckets };
    let router = review::Router::default();
    let output_router = router.clone();
    let args = AnalysisArgs {
      model_file,
      human_model_file: String::new(),
      num_analysis_threads_cmdline_specified: analysis_threads.is_some(),
      num_analysis_threads_cmdline: analysis_threads.unwrap_or(0),
      quit_without_waiting,
      override_configs: applied,
    };
    wasm_bindgen_futures::spawn_local(async move {
      let mut input = rx.flat_map(futures::stream::iter);
      let on_stderr = on_log;
      let host = AnalysisHost {
        args,
        input: &mut input,
        write_stdout: Box::new(move |line: &str| {
          if !output_router.route(line) {
            let _ = on_line.call1(&JsValue::NULL, &JsValue::from_str(line));
          }
        }),
        write_stderr: Box::new(move |text: &str| {
          // `cerr << s` with an empty s (no unused config keys...) writes nothing
          if !text.is_empty() {
            let _ = on_stderr.call2(&JsValue::NULL, &JsValue::from_str("stderr"), &JsValue::from_str(text));
          }
        }),
        nn_host: &nn_host,
        // No files in the browser: every log line goes to on_log whatever logFile / logDir / logToStderr say.
        init_logger: Box::new(|_settings: &LoggerSettings| Ok(())),
      };
      match katago_core::command::analysis::analysis(cfg, host).await {
        Ok(code) => {
          let _ = resolve.call1(&JsValue::NULL, &JsValue::from(code));
        }
        Err(e) => {
          let _ = reject.call1(&JsValue::NULL, &js_sys::Error::new(e.what()).into());
        }
      }
    });
    Ok(Engine { input: RefCell::new(Some(tx)), router, reviews: Default::default(), backend, network, config_text, stats, finished })
  }

  /// "webgpu" | "cpu"
  #[wasm_bindgen(getter)]
  pub fn backend(&self) -> String {
    self.backend.name().to_string()
  }

  #[wasm_bindgen(getter)]
  pub fn network(&self) -> String {
    self.network.clone()
  }

  /// The effective config (`ConfigParser::getAllKeyVals`) before the engine read it.
  #[wasm_bindgen(getter, js_name = configText)]
  pub fn config_text(&self) -> String {
    self.config_text.clone()
  }

  /// Input lines (each one JSON query or action, without its newline), all available at once: as a block
  /// of lines written to the engine's stdin in one write.
  #[wasm_bindgen(js_name = sendLines)]
  pub fn send_lines(&self, lines: Vec<String>) -> Result<(), JsError> {
    let input = self.input.borrow();
    let tx = input.as_ref().ok_or_else(|| JsError::new("input is closed"))?;
    self.stats.batches_at_last_input.set(self.stats.batches.get());
    self.stats.rows_at_last_input.set(self.stats.rows.get());
    tx.unbounded_send(lines.into_iter().map(String::into_bytes).collect()).map_err(|_| JsError::new("the engine has exited"))
  }

  /// Reviews a whole game as katago-ws does (crate katago-annotate): one analysis query over every turn, or
  /// with `screenVisits` a screening pass then a full-depth pass around the moves that swing, and resolves
  /// with the payload v3 JSON text (`GameAnalysis`, engine `{name: "katago-rs-wasm", version, model}`).
  /// - `review_id`: the caller's id for [`Engine::cancel_review`] (unique among the reviews in flight);
  /// - `game`: SGF text, or `{moves, rules?, komi?, boardSize? | boardXSize + boardYSize, initialStones?, result?}`;
  /// - `options`: katago-ws's analysis settings in camelCase (`maxVisits` 100, `screenVisits` 0 = one pass,
  ///   `refineWinrateDelta` 0.03, `refineScoreDelta` 1, `topK` 10, `analysisPvLen` 15, `includeOwnership`,
  ///   `includeOwnershipStdev`, `includePolicy` (all true), `defaultKomi` 7.5, `defaultBoardSize` 19) and
  ///   `humanProfile` (dropped with a warning: no human SL model in the browser);
  /// - `on_progress({pass, passes, turnNumber, done, total, analysed})` after each analysed turn.
  ///
  /// Rejects on a bad game or options, an engine error, or [`Engine::cancel_review`] ("review cancelled").
  #[wasm_bindgen(js_name = reviewGame)]
  pub fn review_game(&self, review_id: String, game: JsValue, options: JsValue, on_progress: Function) -> Result<Promise, JsError> {
    let (game, opts, human_profile) = review::read_args(&game, &options, &self.network)?;
    let input = self.input.borrow().clone().ok_or_else(|| JsError::new("input is closed"))?;
    if self.reviews.borrow().contains_key(&review_id) {
      return Err(JsError::new(&format!("review {review_id} is already running")));
    }
    let state = Rc::new(review::ReviewState::default());
    self.reviews.borrow_mut().insert(review_id.clone(), state.clone());
    let reviews = self.reviews.clone();
    let cancel_state = state.clone();
    let analyzer = review::WasmAnalyzer {
      input,
      router: self.router.clone(),
      review_id: review_id.clone(),
      state,
      on_progress,
      seq: Cell::new(0),
      analysed: Cell::new(0),
      two_pass: opts.two_pass(),
    };
    Ok(wasm_bindgen_futures::future_to_promise(async move {
      let result = match game {
        review::ReviewInput::Sgf(sgf) => katago_annotate::analyze(&analyzer, &opts, &sgf, human_profile.as_deref()).await,
        review::ReviewInput::Moves(input) => match input.into_game(opts.default_board_size, opts.default_komi) {
          Ok(game) => katago_annotate::analyze_game(&analyzer, &opts, &game, human_profile.as_deref()).await,
          Err(e) => Err(e),
        },
      };
      reviews.borrow_mut().remove(&review_id);
      match result.and_then(|payload| serde_json::to_string(&payload).map_err(|e| katago_annotate::AnnotateError::Inference(e.to_string()))) {
        Ok(json) => Ok(JsValue::from_str(&json)),
        Err(_) if cancel_state.cancelled.get() => Err(js_sys::Error::new(review::CANCELLED).into()),
        Err(e) => Err(js_sys::Error::new(&e.to_string()).into()),
      }
    }))
  }

  /// Cancels a review started by [`Engine::review_game`] (no-op when it is not running): its query in flight is
  /// terminated, no further pass starts, and its promise rejects with "review cancelled".
  #[wasm_bindgen(js_name = cancelReview)]
  pub fn cancel_review(&self, review_id: &str) {
    let state = self.reviews.borrow().get(review_id).cloned();
    if let Some(state) = state {
      review::cancel(&state, review_id, self.input.borrow().as_ref());
    }
  }

  /// End of input (stdin closed): the engine finishes the queued queries (unless quitWithoutWaiting) and
  /// exits; [`Engine::finished`] then resolves.
  #[wasm_bindgen(js_name = closeInput)]
  pub fn close_input(&self) {
    self.input.borrow_mut().take();
  }

  /// Resolves with the exit code when the engine exits; rejects with the exception that ended it.
  pub fn finished(&self) -> Promise {
    self.finished.clone()
  }

  /// NN counters: `{batches, rows, busyMs, lastBatchMs, maxBatchMs, lastBatchRows, maxBatchRows, modelBuildMs,
  /// warmupMs, warmupBatches, batchesSinceLastInput, rowsSinceLastInput}` (the warm-up batches are not counted
  /// in batches / rows / busyMs; the *SinceLastInput counters cover the batches completed since the last sendLines).
  pub fn stats(&self) -> JsValue {
    let o = js_sys::Object::new();
    let s = &self.stats;
    let set = |k: &str, v: f64| {
      let _ = Reflect::set(&o, &JsValue::from_str(k), &JsValue::from_f64(v));
    };
    set("batches", s.batches.get() as f64);
    set("rows", s.rows.get() as f64);
    set("busyMs", s.busy_ms.get());
    set("lastBatchMs", s.last_batch_ms.get());
    set("maxBatchMs", s.max_batch_ms.get());
    set("lastBatchRows", s.last_batch_rows.get() as f64);
    set("maxBatchRows", s.max_batch_rows.get() as f64);
    set("modelBuildMs", s.model_build_ms.get());
    set("warmupMs", s.warmup_ms.get());
    set("warmupBatches", s.warmup_batches.get() as f64);
    set("batchesSinceLastInput", (s.batches.get() - s.batches_at_last_input.get()) as f64);
    set("rowsSinceLastInput", (s.rows.get() - s.rows_at_last_input.get()) as f64);
    o.into()
  }
}
