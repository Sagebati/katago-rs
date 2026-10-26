//! KataGo networks imported through burn-onnx.

pub mod backend;
pub mod conv;
#[cfg(feature = "gpu-kernels")]
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
            /// hosts and tests read it at runtime).
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

/// The networks katago-nn-burn knows, compiled in or not (one `net-*` feature each).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NetId {
    B6c96,
    B10c128,
    B18c384nbt,
    /// The human-SL net (SGF metadata input): `-human-model`, or `-model` with a humanSLProfile.
    B18c384nbtHumanv0,
}

impl NetId {
    /// Every network, smallest first.
    pub const ALL: [NetId; 4] = [NetId::B6c96, NetId::B10c128, NetId::B18c384nbt, NetId::B18c384nbtHumanv0];

    /// The network's name, also the stem of its ONNX and `.bpk` files.
    pub fn name(self) -> &'static str {
        match self {
            NetId::B6c96 => "g170-b6c96",
            NetId::B10c128 => "g170e-b10c128",
            NetId::B18c384nbt => "kata1-b18c384nbt",
            NetId::B18c384nbtHumanv0 => "b18c384nbt-humanv0",
        }
    }

    /// The network named `name` ([`NetId::name`]), compiled in or not.
    pub fn from_name(name: &str) -> Option<NetId> {
        NetId::ALL.into_iter().find(|net| net.name() == name)
    }

    /// Whether this build has the network (its `net-*` feature).
    pub fn is_compiled(self) -> bool {
        self.bpk_path().is_some()
    }

    /// The networks compiled into this build, smallest first.
    pub fn compiled() -> impl Iterator<Item = NetId> {
        NetId::ALL.into_iter().filter(|net| net.is_compiled())
    }

    /// Path of the network's generated `.bpk` weights in katago-nn-burn's build directory, when compiled in.
    pub fn bpk_path(self) -> Option<&'static str> {
        match self {
            #[cfg(feature = "net-b6c96")]
            NetId::B6c96 => Some(b6c96::BPK_PATH),
            #[cfg(feature = "net-b10c128")]
            NetId::B10c128 => Some(b10c128::BPK_PATH),
            #[cfg(feature = "net-b18c384nbt")]
            NetId::B18c384nbt => Some(b18c384nbt::BPK_PATH),
            #[cfg(feature = "net-b18c384nbt-humanv0")]
            NetId::B18c384nbtHumanv0 => Some(b18c384nbt_humanv0::BPK_PATH),
            #[allow(unreachable_patterns)]
            _ => None,
        }
    }

    /// The burn backend of katago-core's `NeuralNet` trait for this network on `device`, with
    /// [`backend::BurnNeuralNet::with_batch_buckets`]; `None` when the network is not compiled in.
    pub fn neural_net(
        self,
        device: Device,
        batch_buckets: Vec<i32>,
    ) -> Option<std::rc::Rc<dyn katago_core::neuralnet::nninterface::NeuralNet>> {
        #[allow(unused_imports)]
        use backend::BurnNeuralNet;
        let _ = (&device, &batch_buckets);
        match self {
            #[cfg(feature = "net-b6c96")]
            NetId::B6c96 => Some(std::rc::Rc::new(BurnNeuralNet::<b6c96::Model>::new(device).with_batch_buckets(batch_buckets))),
            #[cfg(feature = "net-b10c128")]
            NetId::B10c128 => {
                Some(std::rc::Rc::new(BurnNeuralNet::<b10c128::Model>::new(device).with_batch_buckets(batch_buckets)))
            }
            #[cfg(feature = "net-b18c384nbt")]
            NetId::B18c384nbt => {
                Some(std::rc::Rc::new(BurnNeuralNet::<b18c384nbt::Model>::new(device).with_batch_buckets(batch_buckets)))
            }
            #[cfg(feature = "net-b18c384nbt-humanv0")]
            NetId::B18c384nbtHumanv0 => Some(std::rc::Rc::new(
                BurnNeuralNet::<b18c384nbt_humanv0::Model>::new(device).with_batch_buckets(batch_buckets),
            )),
            #[allow(unreachable_patterns)]
            _ => None,
        }
    }
}

/// Whether `device` is burn flex, the CPU backend (the GPU kernels of wino.rs never run there).
pub fn is_cpu(device: &Device) -> bool {
    #[cfg(feature = "flex")]
    if matches!(device.as_dispatch(), burn::backend::DispatchDevice::Flex(_)) {
        return true;
    }
    let _ = device;
    false
}

/// The device of a native backend by name (`flex`, `wgpu`, `rocm`), or `None` when it is not compiled in. The
/// browser creates its WebGPU device asynchronously itself (katago-wasm).
#[cfg(not(target_arch = "wasm32"))]
pub fn device(backend: &str) -> Option<Device> {
    match backend {
        #[cfg(feature = "flex")]
        "flex" => Some(burn::backend::FlexDevice.into()),
        #[cfg(feature = "wgpu-fixed")]
        "wgpu" => Some(burn::backend::WgpuDevice::default().into()),
        #[cfg(feature = "rocm")]
        "rocm" => Some(burn::backend::RocmDevice::default().into()),
        _ => None,
    }
}
