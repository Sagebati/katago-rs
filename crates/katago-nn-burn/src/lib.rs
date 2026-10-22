//! KataGo networks imported through burn-onnx.

pub mod conv;

use burn::prelude::*;

/// Raw network outputs, as KataGo's backends produce them before `NNEvaluator`
/// post-processing: policy logits `[N, P, HW+1]` (last = pass), value logits
/// `[N, 3]`, misc score values `[N, S]`, pre-tanh ownership `[N, 1, H, W]`.
pub struct RawOutputs {
    pub policy: Tensor<3>,
    pub value: Tensor<2>,
    pub score: Tensor<2>,
    pub ownership: Tensor<4>,
}

/// A KataGo network taking v7 inputs (`[N, 22, 19, 19]` spatial, `[N, 19]` global).
pub trait Network {
    /// File name stem of the network, also the name of its ONNX/`.bpk` files.
    const NAME: &'static str;
    fn from_bpk(bytes: Vec<u8>, device: &Device) -> Self;
    fn forward(&self, bin_input: Tensor<4>, global_input: Tensor<2>) -> RawOutputs;
}

macro_rules! network {
    ($feature:literal, $module:ident, $name:literal) => {
        #[cfg(feature = $feature)]
        pub mod $module {
            include!(concat!(env!("OUT_DIR"), "/model/", $name, ".rs"));

            impl super::Network for Model {
                const NAME: &'static str = $name;

                fn from_bpk(bytes: Vec<u8>, device: &Device) -> Self {
                    Model::from_bytes(Bytes::from_bytes_vec(bytes), device)
                }

                fn forward(&self, bin_input: Tensor<4>, global_input: Tensor<2>) -> super::RawOutputs {
                    let (policy, value, score, ownership) = Model::forward(self, bin_input, global_input);
                    super::RawOutputs { policy, value, score, ownership }
                }
            }
        }
    };
}

network!("net-b10c128", b10c128, "g170e-b10c128");
network!("net-b18c384nbt", b18c384nbt, "kata1-b18c384nbt");

/// Path of a network's generated `.bpk` weights (native builds and tests).
#[macro_export]
macro_rules! bpk_path {
    ($name:literal) => {
        concat!(env!("OUT_DIR"), "/model/", $name, ".bpk")
    };
}
