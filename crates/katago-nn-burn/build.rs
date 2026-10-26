//! Generates Burn code for each enabled network from its ONNX export
//! (`tools/bin2onnx` converts KataGo's `.bin.gz` into `models/onnx/<net>.onnx`), plus
//! `<net>.desc.rs`: the network's `ModelDesc` built from the ONNX `metadata_props` that bin2onnx
//! writes (the chosen ModelDesc source, see crates/katago-core/src/neuralnet/desc.rs).

use burn_onnx::ext::proc_macro2::{Ident, Span, TokenStream};
use burn_onnx::ext::quote::quote;
use burn_onnx::ext::{CodegenContext, Field, Imports, Node, NodeType, OpOverride, PackTensor, ProcessError, arg_to_ident, create_deferred_tensor};
use burn_onnx::{LoadStrategy, ModelGen};

const NETS: &[(&str, &str)] = &[
  ("NET_B6C96", "g170-b6c96"),
  ("NET_B10C128", "g170e-b10c128"),
  ("NET_B18C384NBT", "kata1-b18c384nbt"),
  // Human-SL net (model v15, metaEncoderVersion 1: a third input `meta_input`, SGFMetadata rows).
  ("NET_B18C384NBT_HUMANV0", "b18c384nbt-humanv0"),
  // Transformer nets (hasAnyTransformerBlocks: attention + RMSNorm blocks between nested bottleneck blocks).
  // b10c384h6nbttflrs is model v15, the other two model v17.
  ("NET_B10C384H6NBTTFLRS", "b10c384h6nbttflrs"),
  ("NET_B10C512H8NBT3TFLRS_FSON_SILU_RSNH", "b10c512h8nbt3tflrs-fson-silu-rsnh"),
  ("NET_TF3_B11C768", "kata1-tf3-b11c768"),
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
    fuse_tf(net);
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
    meta
      .iter()
      .find(|(key, _)| key == k)
      .map(|(_, v)| v.as_str())
      .unwrap_or_else(|| panic!("{onnx}: metadata_props lacks {k} (re-run `just convert`)"))
  };
  let int = |k: &str| -> i32 { get(k).parse().unwrap_or_else(|e| panic!("{onnx}: {k}: {e}")) };
  let f64bits = |k: &str| -> u64 { get(k).parse::<f64>().unwrap_or_else(|e| panic!("{onnx}: {k}: {e}")).to_bits() };
  let f32bits = |k: &str| -> u32 { get(k).parse::<f32>().unwrap_or_else(|e| panic!("{onnx}: {k}: {e}")).to_bits() };
  let num_input_meta_channels = int("numInputMetaChannels");
  // Exports made before bin2onnx wrote metaEncoderVersion are all metaEncoderVersion 0.
  let meta_encoder_version: i32 = match meta.iter().find(|(key, _)| key == "metaEncoderVersion") {
    Some((_, v)) => v.parse().unwrap_or_else(|e| panic!("{onnx}: metaEncoderVersion: {e}")),
    None => {
      assert_eq!(
        num_input_meta_channels, 0,
        "{onnx}: numInputMetaChannels > 0 without metaEncoderVersion (re-run `just convert`)"
      );
      0
    }
  };
  // Keys added with model version 17 support (tools/bin2onnx/README.md): bin2onnx writes the prefer* flags only for
  // nets that use v17-era features, and the arch summary (numParameters, trunkSpatialConvDepth, has*Blocks) for every
  // net, but exports made before lack them. As upstream's ONNX loader (onnxmodelbuilder.cpp, getBoolOr /
  // getDoubleOr / getInt64Or), an absent key reads as its default: the prefer* flags false (what desc.cpp reads for
  // nets before the v15 header slots), the arch summary 0 / false. The arch summary is always `present`: an ONNX
  // export has no layer structure to walk.
  let opt = |k: &str| -> Option<&str> { meta.iter().find(|(key, _)| key == k).map(|(_, v)| v.as_str()) };
  let bool_or = |k: &str, default: bool| -> bool {
    match opt(k) {
      None => default,
      Some("0") | Some("false") => false,
      Some("1") | Some("true") => true,
      Some(v) => panic!("{onnx}: {k}: not a boolean: {v}"),
    }
  };
  let prefer_pass_alive = bool_or("preferPassAliveUnderSuicideRules", false);
  let prefer_exclude_atari = bool_or("preferExcludeTerritoryAdjacentToAtari", false);
  let trunk_depth_bits: u64 = opt("trunkSpatialConvDepth")
    .map(|v| v.parse::<f64>().unwrap_or_else(|e| panic!("{onnx}: trunkSpatialConvDepth: {e}")))
    .unwrap_or(0.0)
    .to_bits();
  let num_parameters: i64 = opt("numParameters")
    .map(|v| v.parse::<i64>().unwrap_or_else(|e| panic!("{onnx}: numParameters: {e}")))
    .unwrap_or(0);
  let has_transformer = bool_or("hasAnyTransformerBlocks", false);
  let has_nbt = bool_or("hasAnyNestedBottleneckBlocks", false);
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
         \x20       prefer_pass_alive_under_suicide_rules: {ppa},\n\
         \x20       prefer_exclude_territory_adjacent_to_atari: {pea},\n\
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
         \x20       arch_summary: katago_core::neuralnet::desc::ArchSummary {{\n\
         \x20           present: true,\n\
         \x20           trunk_spatial_conv_depth: f64::from_bits({tdepth:#x}),\n\
         \x20           num_parameters: {nparams},\n\
         \x20           has_any_transformer_blocks: {htf},\n\
         \x20           has_any_nested_bottleneck_blocks: {hnbt},\n\
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
    ppa = prefer_pass_alive,
    pea = prefer_exclude_atari,
    tdepth = trunk_depth_bits,
    nparams = num_parameters,
    htf = has_transformer,
    hnbt = has_nbt,
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
      node.name(),
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

/// Routes ReduceSum / ReduceMax through `crate::conv::spatial_sum` / `spatial_max`: the convolutional nets only
/// reduce the last axis of a rank-3 `[N, C, H*W]` tensor without keepdims (global pooling and the mask's area),
/// which on GPUs is one workgroup per row instead of burn's generic reduce (~90 us per call at batch 1 without
/// autotune). The transformer nets also have keepdims reduces (their RMSNorms, over the channels of `[N, H*W, C]`
/// or over all of `[N, C*H*W]`): those get the code burn-onnx's built-in would generate (an override replaces the
/// built-in for every node of its type).
struct SpatialReduce(NodeType, &'static str);

impl OpOverride for SpatialReduce {
  fn target(&self) -> NodeType {
    self.0.clone()
  }

  fn forward(&self, node: &Node, ctx: &mut CodegenContext<'_, '_>) -> Result<TokenStream, ProcessError> {
    let (axes, keepdims) = match node {
      Node::ReduceSum(n) => (&n.config.axes, n.config.keepdims),
      Node::ReduceMax(n) => (&n.config.axes, n.config.keepdims),
      _ => return Err(ProcessError::Custom("expected a ReduceSum or ReduceMax node".into())),
    };
    let rank = match &node.inputs()[0].ty {
      burn_onnx::ext::ArgType::Tensor(t) => t.rank,
      other => return Err(ProcessError::Custom(format!("{}: reduce of {other:?}", node.name()))),
    };
    let last_axis = matches!(axes, onnx_ir::node::reduce::ReduceAxes::Static(dims) if dims[..] == [2]);
    let input = ctx.arg(&node.inputs()[0]);
    let output = arg_to_ident(&node.outputs()[0]);
    if last_axis && !keepdims && rank == 3 {
      let f = Ident::new(self.1, Span::call_site());
      return Ok(quote! {
          let #output = crate::conv::#f(#input);
      });
    }
    // Anything else: burn-onnx's built-in code for one static axis (node/reduce.rs `forward_reduce`).
    let axis = match axes {
      onnx_ir::node::reduce::ReduceAxes::Static(dims) if dims.len() == 1 && dims[0] < rank => dims[0],
      _ => {
        return Err(ProcessError::Custom(format!(
          "{}: unsupported reduce (axes {axes:?}, keepdims {keepdims}, rank {rank})",
          node.name()
        )));
      }
    };
    let reduced = match node {
      Node::ReduceSum(_) => quote! { #input.sum_dim(#axis) },
      _ => quote! { #input.max_dim(#axis) },
    };
    if keepdims {
      Ok(quote! {
          let #output = #reduced;
      })
    } else {
      let out_rank = rank - 1;
      Ok(quote! {
          let #output = #reduced.squeeze_dims::<#out_rank>(&[#axis]);
      })
    }
  }
}

/// Rewrites, in the generated `$OUT_DIR/model/<net>.rs`, every batch norm + activation + mask chain that only feeds
/// convs. The mish chain (b18c384nbt and the human-SL net)
///
/// ```text
/// let c1 = self.S.val(); let m = X[.clone()].mul(c1); let c2 = self.B.val(); let a = m.add(c2);
/// let sp = softplus(a.clone(), 1.0); let t = sp.tanh(); let m1 = a.mul(t); let m2 = m1.mul(MASK.clone());
/// ... crate::conv::conv2d_same(&self.CONV, m2[.clone()]) ...
/// ```
///
/// becomes `crate::conv::conv2d_bn_act(&self.CONV, X, self.S.val(), self.B.val(), MASK.clone())`, and the relu chain
/// of the older nets (b6c96, b10c128), the same with `let r = relu(a); let m2 = r.mul(MASK.clone());`, becomes
/// `crate::conv::conv2d_bn_relu(...)`. On GPUs these run the chain inside the conv's input loads. A chain is
/// rewritten only when none of its intermediates is used elsewhere and every use of its result is such a conv;
/// anything else is left as generated.
fn fuse_bn_act(net: &str) {
  let path = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap())
    .join("model")
    .join(format!("{net}.rs"));
  let code = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
  let mish = regex::Regex::new(concat!(
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
  let relu = regex::Regex::new(concat!(
    r"let (\w+) = self\.(\w+)\.val\(\);\s*",
    r"let (\w+) = (\w+)(\.clone\(\))?\.mul\((\w+)\);\s*",
    r"let (\w+) = self\.(\w+)\.val\(\);\s*",
    r"let (\w+) = (\w+)\.add\((\w+)\);\s*",
    r"let (\w+) = burn::tensor::activation::relu\((\w+)\);\s*",
    r"let (\w+) = (\w+)\.mul\((\w+)(?:\.clone\(\))?\);",
  ))
  .unwrap();
  let relu_loaded = regex::Regex::new(concat!(
    r"let (\w+) = (\w+)(\.clone\(\))?\.mul\((\w+)\);\s*",
    r"let (\w+) = (\w+)\.add\((\w+)\);\s*",
    r"let (\w+) = burn::tensor::activation::relu\((\w+)\);\s*",
    r"let (\w+) = (\w+)\.mul\((\w+)(?:\.clone\(\))?\);",
  ))
  .unwrap();
  // Bias only (the scale folded into the previous conv): `let c = self.B.val(); let a = X[.clone()].add(c); relu, mask`.
  let bias_relu = regex::Regex::new(concat!(
    r"let (\w+) = self\.(\w+)\.val\(\);\s*",
    r"let (\w+) = (\w+)(\.clone\(\))?\.add\((\w+)\);\s*",
    r"let (\w+) = burn::tensor::activation::relu\((\w+)\);\s*",
    r"let (\w+) = (\w+)\.mul\((\w+)(?:\.clone\(\))?\);",
  ))
  .unwrap();
  // Bias only, with the bias loaded into a variable earlier (b6c96's generated code).
  let bias_relu_loaded = regex::Regex::new(concat!(
    r"let (\w+) = (\w+)(\.clone\(\))?\.add\((\w+)\);\s*",
    r"let (\w+) = burn::tensor::activation::relu\((\w+)\);\s*",
    r"let (\w+) = (\w+)\.mul\((\w+)(?:\.clone\(\))?\);",
  ))
  .unwrap();
  let mut words: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
  for w in regex::Regex::new(r"\b\w+\b").unwrap().find_iter(&code) {
    *words.entry(w.as_str()).or_default() += 1;
  }
  let count = |name: &str| words.get(name).copied().unwrap_or(0);
  let mut edits: Vec<(usize, usize, String)> = Vec::new();

  // One matched chain: its span, its input X (and whether the chain cloned it), S, B, MASK and its result. Rewrites
  // the convs that consume the result, if they are its only uses.
  struct Chain<'a> {
    start: usize,
    end: usize,
    x: &'a str,
    x_cloned: bool,
    /// The scale and bias arguments (`self.S.val()`, or a variable the generated code loaded earlier); no scale for
    /// a bias-only chain.
    scale: Option<String>,
    bias: String,
    mask: &'a str,
    out: &'a str,
  }
  // The chains already rewritten: the patterns overlap (a hoisted-constants or bias-only match can be the tail of
  // a full chain), so a chain inside one of these spans is skipped.
  let mut taken: Vec<(usize, usize)> = Vec::new();
  let mut rewrite = |c: Chain<'_>, func: &str| -> bool {
    if taken.iter().any(|&(start, end)| c.start < end && start < c.end) {
      return false;
    }
    let rest = &code[c.end..];
    let uses = regex::Regex::new(&format!(r"\b{}\b", c.out)).unwrap();
    let conv_use = regex::Regex::new(&format!(r"crate::conv::conv2d_same\(&self\.(\w+), {}(\.clone\(\))?\)", c.out)).unwrap();
    let convs: Vec<_> = conv_use.captures_iter(rest).collect();
    if convs.is_empty() || convs.len() != uses.find_iter(rest).count() {
      return false;
    }
    taken.push((c.start, c.end));
    edits.push((c.start, c.end, String::new()));
    for (i, conv) in convs.iter().enumerate() {
      // X is cloned unless the chain consumed it and this is its last use.
      let x_arg = if !c.x_cloned && i + 1 == convs.len() {
        c.x.to_string()
      } else {
        format!("{}.clone()", c.x)
      };
      let span = conv.get(0).unwrap();
      edits.push((
        c.end + span.start(),
        c.end + span.end(),
        match &c.scale {
          Some(scale) => format!("crate::conv::{func}(&self.{}, {x_arg}, {scale}, {}, {}.clone())", &conv[1], c.bias, c.mask),
          None => format!("crate::conv::{func}(&self.{}, {x_arg}, {}, {}.clone())", &conv[1], c.bias, c.mask),
        },
      ));
    }
    true
  };

  let mut fused_mish = 0;
  for m in mish.captures_iter(&code) {
    let g = |i: usize| m.get(i).map_or("", |x| x.as_str());
    let (c1, scale, mul, x, x_clone, c2, bias, add, sp, th, m1, out, mask) =
      (g(1), g(2), g(3), g(4), g(5), g(7), g(8), g(9), g(12), g(14), g(16), g(19), g(21));
    let linked = g(6) == c1 && g(10) == mul && g(11) == c2 && g(13) == add && g(15) == sp && g(17) == add && g(18) == th && g(20) == m1;
    // Each intermediate appears exactly as often as the chain itself uses it.
    let private = [(c1, 2), (mul, 2), (c2, 2), (add, 3), (sp, 2), (th, 2), (m1, 2)]
      .iter()
      .all(|(n, k)| count(n) == *k);
    let whole = m.get(0).unwrap();
    let (scale, bias) = (Some(format!("self.{scale}.val()")), format!("self.{bias}.val()"));
    let chain = Chain {
      start: whole.start(),
      end: whole.end(),
      x,
      x_cloned: !x_clone.is_empty(),
      scale,
      bias,
      mask,
      out,
    };
    if linked && private && rewrite(chain, "conv2d_bn_act") {
      fused_mish += 1;
    }
  }
  let mut fused_relu = 0;
  for m in relu.captures_iter(&code) {
    let g = |i: usize| m.get(i).map_or("", |x| x.as_str());
    let (c1, scale, mul, x, x_clone, c2, bias, add, r, out, mask) = (g(1), g(2), g(3), g(4), g(5), g(7), g(8), g(9), g(12), g(14), g(16));
    let linked = g(6) == c1 && g(10) == mul && g(11) == c2 && g(13) == add && g(15) == r;
    let private = [(c1, 2), (mul, 2), (c2, 2), (add, 2), (r, 2)].iter().all(|(n, k)| count(n) == *k);
    let whole = m.get(0).unwrap();
    let (scale, bias) = (Some(format!("self.{scale}.val()")), format!("self.{bias}.val()"));
    let chain = Chain {
      start: whole.start(),
      end: whole.end(),
      x,
      x_cloned: !x_clone.is_empty(),
      scale,
      bias,
      mask,
      out,
    };
    if linked && private && rewrite(chain, "conv2d_bn_relu") {
      fused_relu += 1;
    }
  }
  // The same relu chain where the scale and bias were loaded into variables earlier (b6c96's generated code).
  for m in relu_loaded.captures_iter(&code) {
    let g = |i: usize| m.get(i).map_or("", |x| x.as_str());
    let (mul, x, x_clone, scale, add, bias, r, out, mask) = (g(1), g(2), g(3), g(4), g(5), g(7), g(8), g(10), g(12));
    let linked = g(6) == mul && g(9) == add && g(11) == r;
    let private = [(mul, 2), (add, 2), (r, 2)].iter().all(|(n, k)| count(n) == *k);
    let whole = m.get(0).unwrap();
    let (scale, bias) = (Some(format!("{scale}.clone()")), format!("{bias}.clone()"));
    let chain = Chain {
      start: whole.start(),
      end: whole.end(),
      x,
      x_cloned: !x_clone.is_empty(),
      scale,
      bias,
      mask,
      out,
    };
    if linked && private && rewrite(chain, "conv2d_bn_relu") {
      fused_relu += 1;
    }
  }
  for m in bias_relu.captures_iter(&code) {
    let g = |i: usize| m.get(i).map_or("", |x| x.as_str());
    let (c, bias, add, x, x_clone, r, out, mask) = (g(1), g(2), g(3), g(4), g(5), g(7), g(9), g(11));
    let linked = g(6) == c && g(8) == add && g(10) == r;
    let private = [(c, 2), (add, 2), (r, 2)].iter().all(|(n, k)| count(n) == *k);
    let whole = m.get(0).unwrap();
    let chain = Chain {
      start: whole.start(),
      end: whole.end(),
      x,
      x_cloned: !x_clone.is_empty(),
      scale: None,
      bias: format!("self.{bias}.val()"),
      mask,
      out,
    };
    if linked && private && rewrite(chain, "conv2d_bias_relu") {
      fused_relu += 1;
    }
  }
  for m in bias_relu_loaded.captures_iter(&code) {
    let g = |i: usize| m.get(i).map_or("", |x| x.as_str());
    let (add, x, x_clone, bias, r, out, mask) = (g(1), g(2), g(3), g(4), g(5), g(7), g(9));
    let linked = g(6) == add && g(8) == r;
    let private = [(add, 2), (r, 2)].iter().all(|(n, k)| count(n) == *k);
    let whole = m.get(0).unwrap();
    let chain = Chain {
      start: whole.start(),
      end: whole.end(),
      x,
      x_cloned: !x_clone.is_empty(),
      scale: None,
      bias: format!("{bias}.clone()"),
      mask,
      out,
    };
    if linked && private && rewrite(chain, "conv2d_bias_relu") {
      fused_relu += 1;
    }
  }
  // A net with mish (softplus) or relu activations and nothing fused means the generated code no longer has the shape
  // the chain regexes expect (a burn-onnx update): fail rather than silently lose the fused GPU path.
  for (fused, op, act) in [(fused_mish, "softplus", "mish"), (fused_relu, "relu", "relu")] {
    if fused == 0 && code.contains(&format!("burn::tensor::activation::{op}(")) {
      panic!(
        "build.rs fuse_bn_act: {net} has {act} activations but no batch norm + {act} + mask chain matched; \
         burn-onnx's generated code changed shape, update the regex in fuse_bn_act"
      );
    }
  }
  edits.sort_by_key(|e| std::cmp::Reverse(e.0));
  let mut out = code.clone();
  for (start, end, text) in edits {
    out.replace_range(start..end, &text);
  }
  out.insert_str(
    0,
    &format!("// build.rs fuse_bn_act: {fused_mish} batch norm + mish + mask and {fused_relu} batch norm + relu + mask chains fused into their convs.\n"),
  );
  std::fs::write(&path, out).unwrap();
}

/// Routes the transformer blocks of the generated code to `crate::tf` (no-op on a convolutional net): every
/// `Linear` field's forward, burn's attention call, and each per-token RMSNorm chain (bin2onnx
/// `transformer_rmsnorm`: square, sum over the channels, / C, + eps, sqrt, 1 / that, times x, gamma and the token
/// mask), which becomes one `crate::tf::rms_norm` when its intermediates have no other use.
fn fuse_tf(net: &str) {
  let path = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap())
    .join("model")
    .join(format!("{net}.rs"));
  let code = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
  if !code.contains("burn::tensor::module::attention(") {
    return;
  }
  // A constant the chain loads on its first use (kept: later chains use it too).
  const LOAD: &str = r"(?:let \w+ = self\.\w+\.val\(\);\s*)?";
  let rms = regex::Regex::new(
    &[
      r"let (\w+) = (\w+)\.clone\(\)\s*\.mul\((\w+)\.clone\(\)\);\s*",
      r"let (\w+) = (\w+)\.sum_dim\(2usize\);\s*",
      LOAD,
      r"let (\w+) = (\w+)\s*\.div\(\((\w+(?:\.clone\(\))?)\)\.unsqueeze_dims\(&\[0isize, 1isize\]\)\);\s*",
      LOAD,
      r"let (\w+) = (\w+)\s*\.add\(\((\w+(?:\.clone\(\))?)\)\.unsqueeze_dims\(&\[0isize, 1isize\]\)\);\s*",
      r"let (\w+) = (\w+)\.sqrt\(\);\s*",
      LOAD,
      r"let (\w+) = \(\w+(?:\.clone\(\))?\)\s*\.unsqueeze_dims\(&\[0isize, 1isize\]\)\s*\.div\((\w+)\);\s*",
      r"let (\w+) = (\w+)(\.clone\(\))?\s*\.mul\((\w+)\);\s*",
      LOAD,
      r"let (\w+) = (\w+)\s*\.mul\(\((\w+(?:\.clone\(\))?)\)\.unsqueeze_dims\(&\[0isize, 1isize\]\)\);\s*",
      r"let (\w+) = (\w+)\s*\.mul\((\w+(?:\.clone\(\))?)\);",
    ]
    .concat(),
  )
  .unwrap();
  let loads = regex::Regex::new(r"let \w+ = self\.\w+\.val\(\);").unwrap();
  let mut words: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
  for w in regex::Regex::new(r"\b\w+\b").unwrap().find_iter(&code) {
    *words.entry(w.as_str()).or_default() += 1;
  }
  let count = |name: &str| words.get(name).copied().unwrap_or(0);
  let mut edits: Vec<(usize, usize, String)> = Vec::new();
  let mut fused_rms = 0;
  for m in rms.captures_iter(&code) {
    let g = |i: usize| m.get(i).map_or("", |x| x.as_str());
    let (sq, x, ss, ms, divisor, den, eps, sqrt, inv, normed, x_last, x_clone, scaled, gamma, out, mask) = (
      g(1),
      g(2),
      g(4),
      g(6),
      g(8),
      g(9),
      g(11),
      g(12),
      g(14),
      g(16),
      g(17),
      g(18),
      g(20),
      g(22),
      g(23),
      g(25),
    );
    let linked = g(3) == x
      && g(5) == sq
      && g(7) == ss
      && g(10) == ms
      && g(13) == den
      && g(15) == sqrt
      && x_last == x
      && g(19) == inv
      && g(21) == normed
      && g(24) == scaled;
    let private = [sq, ss, ms, den, sqrt, inv, normed, scaled].iter().all(|n| count(n) == 2);
    if !(linked && private) {
      continue;
    }
    let whole = m.get(0).unwrap();
    let kept: String = loads.find_iter(whole.as_str()).map(|l| format!("{}\n", l.as_str())).collect();
    edits.push((
      whole.start(),
      whole.end(),
      format!("{kept}let {out} = crate::tf::rms_norm({x}{x_clone}, {divisor}, {eps}, {gamma}, {mask});"),
    ));
    fused_rms += 1;
  }
  let mut out = code.clone();
  edits.sort_by_key(|e| std::cmp::Reverse(e.0));
  for (start, end, text) in edits {
    out.replace_range(start..end, &text);
  }
  // Linear fields (`name: Linear,` in the module struct) and the attention call.
  let fields: std::collections::BTreeSet<String> = regex::Regex::new(r"(?m)^\s*(?:pub )?(\w+): Linear,")
    .unwrap()
    .captures_iter(&out)
    .map(|c| c[1].to_string())
    .collect();
  let mut linears = 0;
  for f in &fields {
    let call = regex::Regex::new(&format!(r"self\.{f}\s*\.forward\(")).unwrap();
    linears += call.find_iter(&out).count();
    out = call.replace_all(&out, format!("crate::tf::linear(&self.{f}, ").as_str()).into_owned();
  }
  let attentions = out.matches("burn::tensor::module::attention(").count();
  out = out.replace("burn::tensor::module::attention(", "crate::tf::attention(");
  if fused_rms == 0 {
    panic!(
      "build.rs fuse_tf: {net} has attention but no transformer RMSNorm chain matched; burn-onnx's generated code \
       changed shape, update the regex in fuse_tf"
    );
  }
  out.insert_str(
    0,
    &format!("// build.rs fuse_tf: {fused_rms} RMSNorms, {linears} linears and {attentions} attentions routed to crate::tf.\n"),
  );
  std::fs::write(&path, out).unwrap();
}
