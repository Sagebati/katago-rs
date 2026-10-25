//! Generates Burn code for each enabled network from its ONNX export
//! (`tools/bin2onnx` converts KataGo's `.bin.gz` into `models/onnx/<net>.onnx`), plus
//! `<net>.desc.rs`: the network's `ModelDesc` built from the ONNX `metadata_props` that bin2onnx
//! writes (the chosen ModelDesc source, see crates/katago-core/src/neuralnet/desc.rs).

use burn_onnx::ext::proc_macro2::{Ident, Span, TokenStream};
use burn_onnx::ext::quote::quote;
use burn_onnx::ext::{
    CodegenContext, Field, Imports, Node, NodeType, OpOverride, PackTensor, ProcessError,
    arg_to_ident, create_deferred_tensor,
};
use burn_onnx::{LoadStrategy, ModelGen};

const NETS: &[(&str, &str)] = &[
    ("NET_B6C96", "g170-b6c96"),
    ("NET_B10C128", "g170e-b10c128"),
    ("NET_B18C384NBT", "kata1-b18c384nbt"),
    // Human-SL net (model v15, metaEncoderVersion 1: a third input `meta_input`, SGFMetadata rows).
    ("NET_B18C384NBT_HUMANV0", "b18c384nbt-humanv0"),
];

fn main() {
    let models = concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/onnx");
    for (feature, net) in NETS {
        if std::env::var_os(format!("CARGO_FEATURE_{feature}")).is_none() {
            continue;
        }
        let onnx = format!("{models}/{net}.onnx");
        println!("cargo:rerun-if-changed={onnx}");
        ModelGen::new()
            .input(&onnx)
            .out_dir("model/")
            // Weights are fetched at runtime (browser) or read from disk, never embedded.
            .load_strategy(LoadStrategy::Bytes)
            .register_op_override(SameConv)
            .register_op_override(SpatialReduce(NodeType::ReduceSum, "spatial_sum"))
            .register_op_override(SpatialReduce(NodeType::ReduceMax, "spatial_max"))
            .run_from_script();
        fuse_bn_act(net);
        write_model_desc(&onnx, net);
    }
}

/// Reads `ModelProto.metadata_props` (field 14, repeated `StringStringEntryProto {1: key, 2: value}`)
/// with a minimal protobuf walk and writes `$OUT_DIR/model/<net>.desc.rs`, a `model_desc()` fn.
fn write_model_desc(onnx: &str, net: &str) {
    let bytes = std::fs::read(onnx).unwrap_or_else(|e| panic!("{onnx}: {e}"));
    let mut meta: Vec<(String, String)> = Vec::new();
    for (field, value) in proto_fields(&bytes) {
        if field == 14 {
            let mut key = String::new();
            let mut val = String::new();
            for (f, v) in proto_fields(value) {
                match f {
                    1 => key = String::from_utf8(v.to_vec()).unwrap(),
                    2 => val = String::from_utf8(v.to_vec()).unwrap(),
                    _ => {}
                }
            }
            meta.push((key, val));
        }
    }
    let get = |k: &str| -> &str {
        meta.iter().find(|(key, _)| key == k).map(|(_, v)| v.as_str()).unwrap_or_else(|| panic!("{onnx}: metadata_props lacks {k} (re-run `just convert`)"))
    };
    let int = |k: &str| -> i32 { get(k).parse().unwrap_or_else(|e| panic!("{onnx}: {k}: {e}")) };
    let f64bits = |k: &str| -> u64 { get(k).parse::<f64>().unwrap_or_else(|e| panic!("{onnx}: {k}: {e}")).to_bits() };
    let f32bits = |k: &str| -> u32 { get(k).parse::<f32>().unwrap_or_else(|e| panic!("{onnx}: {k}: {e}")).to_bits() };
    let num_input_meta_channels = int("numInputMetaChannels");
    // Exports made before bin2onnx wrote metaEncoderVersion are all metaEncoderVersion 0.
    let meta_encoder_version: i32 = match meta.iter().find(|(key, _)| key == "metaEncoderVersion") {
        Some((_, v)) => v.parse().unwrap_or_else(|e| panic!("{onnx}: metaEncoderVersion: {e}")),
        None => {
            assert_eq!(num_input_meta_channels, 0, "{onnx}: numInputMetaChannels > 0 without metaEncoderVersion (re-run `just convert`)");
            0
        }
    };
    let code = format!(
        "/// The network's `ModelDesc`, from the ONNX `metadata_props` written by tools/bin2onnx (desc.cpp-faithful parser).\n\
         pub fn model_desc() -> katago_core::neuralnet::desc::ModelDesc {{\n\
         \x20   katago_core::neuralnet::desc::ModelDesc {{\n\
         \x20       name: {name:?}.to_string(),\n\
         \x20       sha256: String::new(),\n\
         \x20       model_version: {mv},\n\
         \x20       num_input_channels: {nic},\n\
         \x20       num_input_global_channels: {nigc},\n\
         \x20       num_input_meta_channels: {nimc},\n\
         \x20       num_policy_channels: {npc},\n\
         \x20       num_value_channels: {nvc},\n\
         \x20       num_score_value_channels: {nsvc},\n\
         \x20       num_ownership_channels: {noc},\n\
         \x20       meta_encoder_version: {mev},\n\
         \x20       post_process_params: katago_core::neuralnet::desc::ModelPostProcessParams {{\n\
         \x20           td_score_multiplier: f64::from_bits({td:#x}),\n\
         \x20           score_mean_multiplier: f64::from_bits({sm:#x}),\n\
         \x20           score_stdev_multiplier: f64::from_bits({ss:#x}),\n\
         \x20           lead_multiplier: f64::from_bits({lm:#x}),\n\
         \x20           variance_time_multiplier: f64::from_bits({vt:#x}),\n\
         \x20           shortterm_value_error_multiplier: f64::from_bits({sv:#x}),\n\
         \x20           shortterm_score_error_multiplier: f64::from_bits({se:#x}),\n\
         \x20           output_scale_multiplier: f32::from_bits({os:#x}),\n\
         \x20       }},\n\
         \x20   }}\n\
         }}\n\
         /// Fixed spatial size of the generated graph (bin2onnx `--pos-len`).\n\
         pub const POS_LEN: i32 = {pl};\n\
         /// Width of the `meta_input` graph input (0: the graph has none).\n\
         pub const NUM_INPUT_META_CHANNELS: i32 = {nimc};\n",
        name = get("name"),
        mv = int("modelVersion"),
        nic = int("numInputChannels"),
        nigc = int("numInputGlobalChannels"),
        nimc = num_input_meta_channels,
        mev = meta_encoder_version,
        npc = int("numPolicyChannels"),
        nvc = int("numValueChannels"),
        nsvc = int("numScoreValueChannels"),
        noc = int("numOwnershipChannels"),
        td = f64bits("tdScoreMultiplier"),
        sm = f64bits("scoreMeanMultiplier"),
        ss = f64bits("scoreStdevMultiplier"),
        lm = f64bits("leadMultiplier"),
        vt = f64bits("varianceTimeMultiplier"),
        sv = f64bits("shorttermValueErrorMultiplier"),
        se = f64bits("shorttermScoreErrorMultiplier"),
        os = f32bits("outputScaleMultiplier"),
        pl = int("posLen"),
    );
    let out = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("model");
    std::fs::create_dir_all(&out).unwrap();
    std::fs::write(out.join(format!("{net}.desc.rs")), code).unwrap();
}

/// Top-level fields of a protobuf message: (field number, payload) for length-delimited fields;
/// varint/fixed fields are skipped.
fn proto_fields(mut b: &[u8]) -> Vec<(u64, &[u8])> {
    fn varint(b: &mut &[u8]) -> u64 {
        let mut x = 0u64;
        let mut shift = 0;
        loop {
            let byte = b[0];
            *b = &b[1..];
            x |= ((byte & 0x7f) as u64) << shift;
            if byte & 0x80 == 0 {
                return x;
            }
            shift += 7;
        }
    }
    let mut out = Vec::new();
    while !b.is_empty() {
        let tag = varint(&mut b);
        match tag & 7 {
            0 => {
                varint(&mut b);
            }
            1 => b = &b[8..],
            2 => {
                let len = varint(&mut b) as usize;
                out.push((tag >> 3, &b[..len]));
                b = &b[len..];
            }
            5 => b = &b[4..],
            w => panic!("unsupported protobuf wire type {w}"),
        }
    }
    out
}

/// Routes every Conv through `crate::conv::conv2d_same`, which on GPUs runs 3x3 kernels as Winograd convolutions
/// and 1x1 kernels as an NCHW GEMM (wino/), and any other size as im2col + matmul; burn's own conv on the CPU.
/// Weights keep the built-in `Conv2d` field and `<name>.weight` path, so the generated `.bpk` is unchanged.
struct SameConv;

/// The (kernel, padding) of a stride-1, undilated, ungrouped, bias-free conv
/// with "same" padding: the only kind KataGo nets contain.
fn same_conv(node: &Node) -> Result<([usize; 2], [usize; 2]), ProcessError> {
    let Node::Conv2d(conv) = node else {
        return Err(ProcessError::Custom("expected a Conv2d node".into()));
    };
    let c = &conv.config;
    let (top, left, bottom, right) = c.padding.as_tuple();
    let [kh, kw] = c.kernel_size;
    let same = top == kh / 2 && bottom == kh / 2 && left == kw / 2 && right == kw / 2;
    let biased = conv.inputs.get(2).is_some_and(|b| !b.is_optional());
    if c.stride != [1, 1] || c.dilation != [1, 1] || c.groups != 1 || !same || biased || !conv.inputs[1].is_static() {
        return Err(ProcessError::Custom(format!("{}: unsupported conv {c:?}", conv.name)));
    }
    Ok((c.kernel_size, [kh / 2, kw / 2]))
}

impl OpOverride for SameConv {
    fn target(&self) -> NodeType {
        NodeType::Conv2d
    }

    fn field(&self, node: &Node) -> Result<Option<Field>, ProcessError> {
        let ([kh, kw], [ph, pw]) = same_conv(node)?;
        let shape = node.inputs()[1].ty.static_shape_known().expect("static conv weight");
        let (oc, ic) = (shape[0], shape[1]);
        let name = Ident::new(node.name(), Span::call_site());
        Ok(Some(Field::new(
            node.name().to_string(),
            quote! { Conv2d },
            quote! {
                let #name = Conv2dConfig::new([#ic, #oc], [#kh, #kw])
                    .with_padding(PaddingConfig2d::Explicit(#ph, #pw, #ph, #pw))
                    .with_bias(false)
                    .init(device);
            },
        )))
    }

    fn collect_tensors(&self, node: &Node, field_name: &str) -> Result<Vec<PackTensor>, ProcessError> {
        let weight = create_deferred_tensor(&node.inputs()[1], &format!("{field_name}.weight"));
        Ok(weight.into_iter().collect())
    }

    fn forward(&self, node: &Node, ctx: &mut CodegenContext<'_, '_>) -> Result<TokenStream, ProcessError> {
        same_conv(node)?;
        let input = ctx.arg(&node.inputs()[0]);
        let output = arg_to_ident(&node.outputs()[0]);
        let field = Ident::new(node.name(), Span::call_site());
        Ok(quote! {
            let #output = crate::conv::conv2d_same(&self.#field, #input);
        })
    }

    fn register_imports(&self, imports: &mut Imports<'_>) {
        imports.register("burn::nn::PaddingConfig2d");
        imports.register("burn::nn::conv::Conv2d");
        imports.register("burn::nn::conv::Conv2dConfig");
    }
}

/// Routes ReduceSum / ReduceMax through `crate::conv::spatial_sum` / `spatial_max`: KataGo nets only reduce the
/// last axis of a rank-3 `[N, C, H*W]` tensor without keepdims (global pooling and the mask's area), which on GPUs
/// is one workgroup per row instead of burn's generic reduce (~90 us per call at batch 1 without autotune).
struct SpatialReduce(NodeType, &'static str);

impl OpOverride for SpatialReduce {
    fn target(&self) -> NodeType {
        self.0.clone()
    }

    fn forward(&self, node: &Node, ctx: &mut CodegenContext<'_, '_>) -> Result<TokenStream, ProcessError> {
        // The axes as their Debug form: onnx-ir's ReduceAxes is not re-exported by burn-onnx. Any other form is
        // rejected below, loudly.
        let (axes, keepdims) = match node {
            Node::ReduceSum(n) => (format!("{:?}", n.config.axes), n.config.keepdims),
            Node::ReduceMax(n) => (format!("{:?}", n.config.axes), n.config.keepdims),
            _ => return Err(ProcessError::Custom("expected a ReduceSum or ReduceMax node".into())),
        };
        let rank = match &node.inputs()[0].ty {
            burn_onnx::ext::ArgType::Tensor(t) => t.rank,
            other => return Err(ProcessError::Custom(format!("{}: reduce of {other:?}", node.name()))),
        };
        if axes != "Static([2])" || keepdims || rank != 3 {
            return Err(ProcessError::Custom(format!(
                "{}: unsupported reduce (axes {axes}, keepdims {keepdims}, rank {rank})",
                node.name()
            )));
        }
        let input = ctx.arg(&node.inputs()[0]);
        let output = arg_to_ident(&node.outputs()[0]);
        let f = Ident::new(self.1, Span::call_site());
        Ok(quote! {
            let #output = crate::conv::#f(#input);
        })
    }
}

/// Rewrites, in the generated `$OUT_DIR/model/<net>.rs`, every batch norm + mish + mask chain that only feeds convs
///
/// ```text
/// let c1 = self.S.val(); let m = X[.clone()].mul(c1); let c2 = self.B.val(); let a = m.add(c2);
/// let sp = softplus(a.clone(), 1.0); let t = sp.tanh(); let m1 = a.mul(t); let m2 = m1.mul(MASK.clone());
/// ... crate::conv::conv2d_same(&self.CONV, m2[.clone()]) ...
/// ```
///
/// into `crate::conv::conv2d_bn_act(&self.CONV, X, self.S.val(), self.B.val(), MASK.clone())`, which on GPUs runs the
/// chain inside the conv's input loads. A chain is rewritten only when none of its intermediates is used
/// elsewhere and every use of its result is such a conv; anything else is left as generated.
fn fuse_bn_act(net: &str) {
    let path = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("model").join(format!("{net}.rs"));
    let code = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let chain = regex::Regex::new(concat!(
        r"let (\w+) = self\.(\w+)\.val\(\);\s*",
        r"let (\w+) = (\w+)(\.clone\(\))?\.mul\((\w+)\);\s*",
        r"let (\w+) = self\.(\w+)\.val\(\);\s*",
        r"let (\w+) = (\w+)\.add\((\w+)\);\s*",
        r"let (\w+) = burn::tensor::activation::softplus\(\s*(\w+)\.clone\(\),\s*1\.0,?\s*\);\s*",
        r"let (\w+) = (\w+)\.tanh\(\);\s*",
        r"let (\w+) = (\w+)\.mul\((\w+)\);\s*",
        r"let (\w+) = (\w+)\.mul\((\w+)\.clone\(\)\);",
    ))
    .unwrap();
    let mut words: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for w in regex::Regex::new(r"\b\w+\b").unwrap().find_iter(&code) {
        *words.entry(w.as_str()).or_default() += 1;
    }
    let count = |name: &str| words.get(name).copied().unwrap_or(0);
    let mut edits: Vec<(usize, usize, String)> = Vec::new();
    let mut fused = 0;
    for m in chain.captures_iter(&code) {
        let g = |i: usize| m.get(i).map_or("", |x| x.as_str());
        let (c1, scale, mul, x, x_clone, c2, bias, add, sp, th, m1, out, mask) =
            (g(1), g(2), g(3), g(4), g(5), g(7), g(8), g(9), g(12), g(14), g(16), g(19), g(21));
        let linked = g(6) == c1 && g(10) == mul && g(11) == c2 && g(13) == add && g(15) == sp && g(17) == add && g(18) == th && g(20) == m1;
        // Each intermediate appears exactly as often as the chain itself uses it.
        let private = [(c1, 2), (mul, 2), (c2, 2), (add, 3), (sp, 2), (th, 2), (m1, 2)].iter().all(|(n, k)| count(n) == *k);
        if !linked || !private {
            continue;
        }
        let whole = m.get(0).unwrap();
        let rest = &code[whole.end()..];
        let uses = regex::Regex::new(&format!(r"\b{out}\b")).unwrap();
        let conv_use = regex::Regex::new(&format!(r"crate::conv::conv2d_same\(&self\.(\w+), {out}(\.clone\(\))?\)")).unwrap();
        let convs: Vec<_> = conv_use.captures_iter(rest).collect();
        if convs.is_empty() || convs.len() != uses.find_iter(rest).count() {
            continue;
        }
        edits.push((whole.start(), whole.end(), String::new()));
        for (i, c) in convs.iter().enumerate() {
            // X is cloned unless the chain consumed it and this is its last use.
            let x_arg = if x_clone.is_empty() && i + 1 == convs.len() { x.to_string() } else { format!("{x}.clone()") };
            let span = c.get(0).unwrap();
            edits.push((
                whole.end() + span.start(),
                whole.end() + span.end(),
                format!("crate::conv::conv2d_bn_act(&self.{}, {x_arg}, self.{scale}.val(), self.{bias}.val(), {mask}.clone())", &c[1]),
            ));
        }
        fused += 1;
    }
    // A mish net (softplus in its activations) with nothing fused means the generated code no longer has the shape
    // the chain regex expects (a burn-onnx update): fail rather than silently lose the fused GPU path.
    if fused == 0 && code.contains("burn::tensor::activation::softplus(") {
        panic!(
            "build.rs fuse_bn_act: {net} has mish activations but no batch norm + mish + mask chain matched; \
             burn-onnx's generated code changed shape, update the regex in fuse_bn_act"
        );
    }
    edits.sort_by_key(|e| std::cmp::Reverse(e.0));
    let mut out = code.clone();
    for (start, end, text) in edits {
        out.replace_range(start..end, &text);
    }
    out.insert_str(0, &format!("// build.rs fuse_bn_act: {fused} batch norm + mish + mask chains fused into their convs.\n"));
    std::fs::write(&path, out).unwrap();
}
