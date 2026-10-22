//! Generates Burn code for each enabled network from its ONNX export
//! (`tools/bin2onnx` converts KataGo's `.bin.gz` into `models/onnx/<net>.onnx`).

use burn_onnx::ext::proc_macro2::{Ident, Span, TokenStream};
use burn_onnx::ext::quote::quote;
use burn_onnx::ext::{
    CodegenContext, Field, Imports, Node, NodeType, OpOverride, PackTensor, ProcessError,
    arg_to_ident, create_deferred_tensor,
};
use burn_onnx::{LoadStrategy, ModelGen};

const NETS: &[(&str, &str)] = &[
    ("NET_B10C128", "g170e-b10c128"),
    ("NET_B18C384NBT", "kata1-b18c384nbt"),
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
            .run_from_script();
    }
}

/// Routes every Conv through `crate::conv::conv2d_same`, which lowers spatial
/// kernels to im2col + matmul: burn's direct conv kernel is ~7x slower than
/// matmul on GPUs without matrix cores. Weights keep the built-in `Conv2d`
/// field and `<name>.weight` path, so the generated `.bpk` is unchanged.
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
