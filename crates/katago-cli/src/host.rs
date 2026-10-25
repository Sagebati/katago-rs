//! The burn backend as Setup's NNEvaluatorHost, and `-model` to a compiled-in network.

use std::cell::RefCell;
use std::rc::Rc;

use burn::prelude::Device;
use katago_core::core::global::KataError;
use katago_core::neuralnet::nninterface::NeuralNet;
use katago_core::port::executor::Spawner;
use katago_core::program::setup::NNEvaluatorHost;

use crate::*;

/// The networks of katago-nn-burn (compiled in or not: see the crate features).
pub use katago_nn_burn::NetId as Net;

fn resolve_net_by_name(model_file: &str) -> Option<Net> {
  let base = std::path::Path::new(model_file).file_name().and_then(|s| s.to_str()).unwrap_or(model_file);
  let stem = [".bin.gz", ".txt.gz", ".onnx", ".bpk", ".gz", ".bin"].iter().fold(base.to_string(), |s, ext| s.strip_suffix(ext).map(str::to_string).unwrap_or(s));
  let candidates: [(Net, &str); 4] = [
    (Net::B6c96, "g170-b6c96-s175395328-d26788732"),
    (Net::B10c128, "g170e-b10c128-s1141046784-d204142634"),
    (Net::B18c384nbt, "kata1-b18c384nbt"),
    (Net::B18c384nbtHumanv0, "b18c384nbt-humanv0"),
  ];
  for (net, full) in candidates {
    if stem == net.name() || stem == full || stem.starts_with(&(net.name().to_string() + "-")) && full.starts_with(&stem) {
      return Some(net);
    }
  }
  None
}

/// Maps a `-model` path to a compiled network (see the crate docs).
pub fn resolve_net(model_file: &str) -> Result<Net, KataError> {
  if let Some(net) = resolve_net_by_name(model_file) {
    return Ok(net);
  }
  if let Some(net) = std::fs::canonicalize(model_file).ok().and_then(|target| resolve_net_by_name(&target.to_string_lossy())) {
    return Ok(net);
  }
  if !std::path::Path::new(model_file).exists() {
    // upstream's text for a missing file (ModelDesc loading: FileUtils::open's IOError, wrapped)
    return Err(KataError::StringError(format!(
      "Error loading or parsing model file {model_file}: Could not open file {model_file} - does not exist or invalid permissions?"
    )));
  }
  Err(KataError::StringError(format!(
    "Error loading or parsing model file {model_file}: katago-rs has no runtime .bin.gz loader; -model must name a network compiled into katago-nn-burn (g170-b6c96, g170e-b10c128, kata1-b18c384nbt, b18c384nbt-humanv0)"
  )))
}

/// Setup's `NNEvaluatorHost` for the burn networks of katago-nn-burn.
pub struct BurnHost {
  pub device: Device,
  pub spawner: Rc<dyn Spawner>,
  /// Model files loaded so far (for tests and diagnostics).
  pub loaded: RefCell<Vec<String>>,
}

impl BurnHost {
  pub fn new(backend: &str, spawner: Rc<dyn Spawner>) -> Result<BurnHost, KataError> {
    let device = compiled_backends().contains(&backend).then(|| katago_nn_burn::device(backend)).flatten().ok_or_else(|| {
      KataError::StringError(format!(
        "unknown or not compiled-in backend '{backend}' (flex, wgpu with the wgpu feature, rocm with the rocm feature)"
      ))
    })?;
    Ok(BurnHost { device, spawner, loaded: RefCell::new(Vec::new()) })
  }
}

fn read_bpk(model_file: &str, compiled: &str) -> Result<Vec<u8>, KataError> {
  let path = if model_file.ends_with(".bpk") { model_file.to_string() } else { compiled.to_string() };
  std::fs::read(&path).map_err(|e| KataError::IOError(format!("Could not open file {path} - does not exist or invalid permissions? ({e})")))
}

impl NNEvaluatorHost for BurnHost {
  fn backend_prefix(&self) -> String {
    "burn".to_string()
  }
  fn global_initialize(&self) {}
  fn global_cleanup(&self) {}
  fn load_model(&self, _nn_model_name: &str, nn_model_file: &str) -> Result<(Rc<dyn NeuralNet>, Vec<u8>), KataError> {
    self.loaded.borrow_mut().push(nn_model_file.to_string());
    let net = if nn_model_file == "/dev/null" { Net::B6c96 } else { resolve_net(nn_model_file)? };
    let empty = nn_model_file == "/dev/null";
    match (net.neural_net(self.device.clone(), Vec::new()), net.bpk_path()) {
      (Some(neural_net), Some(compiled)) => Ok((neural_net, if empty { Vec::new() } else { read_bpk(nn_model_file, compiled)? })),
      _ => Err(KataError::StringError(format!("network {} is not compiled in (enable the katago-cli feature for it)", net.name()))),
    }
  }
  fn spawner(&self) -> Rc<dyn Spawner> {
    self.spawner.clone()
  }
}
