//! KataGo networks imported through burn-onnx.

pub mod backend;
pub mod conv;
#[cfg(feature = "wgpu-fixed-kernels")]
pub mod wino;
#[cfg(target_arch = "wasm32")]
pub mod host;

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

/// A KataGo network taking v7 inputs (`[N, 22, 19, 19]` spatial, `[N, 19]` global), plus for human-SL nets
/// (metaEncoderVersion > 0) the SGF metadata rows (`[N, 192]`, SGFMetadata::fillMetadataRow).
pub trait Network {
    /// File name stem of the network, also the name of its ONNX/`.bpk` files.
    const NAME: &'static str;
    /// Fixed spatial size of the graph (bin2onnx `--pos-len`, 19).
    const POS_LEN: i32;
    /// Width of the `meta_input` (0: the network takes none).
    const NUM_INPUT_META_CHANNELS: i32;
    /// KataGo's `ModelDesc` for this network, from the ONNX `metadata_props` (build.rs).
    fn model_desc() -> katago_core::neuralnet::desc::ModelDesc;
    fn from_bpk(bytes: Vec<u8>, device: &Device) -> Self;
    /// `meta_input` is `Some` exactly when `NUM_INPUT_META_CHANNELS > 0` (panics otherwise).
    fn forward(&self, bin_input: Tensor<4>, global_input: Tensor<2>, meta_input: Option<Tensor<2>>) -> RawOutputs;
}

macro_rules! network {
    ($feature:literal, $module:ident, $name:literal) => {
        network!($feature, $module, $name, |model, bin_input, global_input, meta_input| {
            assert!(meta_input.is_none(), "{}: the network takes no meta_input", $name);
            Model::forward(model, bin_input, global_input)
        });
    };
    ($feature:literal, $module:ident, $name:literal, meta) => {
        network!($feature, $module, $name, |model, bin_input, global_input, meta_input| {
            let meta_input = meta_input.unwrap_or_else(|| panic!("{}: the network needs a meta_input", $name));
            Model::forward(model, bin_input, global_input, meta_input)
        });
    };
    ($feature:literal, $module:ident, $name:literal, |$m:ident, $b:ident, $g:ident, $mi:ident| $call:expr) => {
        #[cfg(feature = $feature)]
        pub mod $module {
            include!(concat!(env!("OUT_DIR"), "/model/", $name, ".rs"));
            include!(concat!(env!("OUT_DIR"), "/model/", $name, ".desc.rs"));

            /// Path of this network's generated `.bpk` weights in katago-nn-burn's build directory (native
            /// hosts read it at runtime; usable from other crates, unlike `bpk_path!`).
            pub const BPK_PATH: &str = concat!(env!("OUT_DIR"), "/model/", $name, ".bpk");

            impl super::Network for Model {
                const NAME: &'static str = $name;
                const POS_LEN: i32 = POS_LEN;
                const NUM_INPUT_META_CHANNELS: i32 = NUM_INPUT_META_CHANNELS;

                fn model_desc() -> katago_core::neuralnet::desc::ModelDesc {
                    model_desc()
                }

                fn from_bpk(bytes: Vec<u8>, device: &Device) -> Self {
                    Model::from_bytes(Bytes::from_bytes_vec(bytes), device)
                }

                fn forward(&self, $b: Tensor<4>, $g: Tensor<2>, $mi: Option<Tensor<2>>) -> super::RawOutputs {
                    let $m = self;
                    let (policy, value, score, ownership) = $call;
                    super::RawOutputs { policy, value, score, ownership }
                }
            }
        }
    };
}

network!("net-b6c96", b6c96, "g170-b6c96");
network!("net-b10c128", b10c128, "g170e-b10c128");
network!("net-b18c384nbt", b18c384nbt, "kata1-b18c384nbt");
network!("net-b18c384nbt-humanv0", b18c384nbt_humanv0, "b18c384nbt-humanv0", meta);

/// Path of a network's generated `.bpk` weights (native builds and tests).
#[macro_export]
macro_rules! bpk_path {
    ($name:literal) => {
        concat!(env!("OUT_DIR"), "/model/", $name, ".bpk")
    };
}
