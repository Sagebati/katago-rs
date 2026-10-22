//! Mirrors `upstream/KataGo/cpp/neuralnet/` (the backend-independent part; backends are
//! replaced by tools/bin2onnx + burn in crates/katago-nn-burn).

pub mod modelversion;
pub mod desc;
pub mod sgfmetadata;
pub mod nninputs;
pub mod nninterface;
pub mod nneval;
