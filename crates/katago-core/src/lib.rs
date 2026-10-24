//! katago-core: the 1:1 port of KataGo's `cpp/` tree (upstream v1.16.5).
//!
//! Module layout mirrors `upstream/KataGo/cpp/` (see port-manifest.toml); `port/` holds the
//! helpers that have no upstream file (docs/PORTING.md appendix).
// Lints the fidelity rules trigger (PORTING.md §11b): add to this list, never sprinkle
// per-item allows for these.
#![allow(non_snake_case, non_upper_case_globals, non_camel_case_types)]
#![allow(clippy::collapsible_if, clippy::collapsible_else_if, clippy::needless_range_loop,
         clippy::manual_range_contains, clippy::new_without_default, clippy::too_many_arguments,
         clippy::inherent_to_string_shadow_display, clippy::manual_swap, clippy::nonminimal_bool,
         clippy::needless_return, clippy::comparison_chain, clippy::if_same_then_else,
         clippy::field_reassign_with_default)]
// Added in M2 (upstream literals such as nninputs.cpp's twoOverPi trip the deny-by-default
// approx_constant; the rest are upstream statement shapes kept verbatim).
#![allow(clippy::approx_constant, clippy::excessive_precision, clippy::assign_op_pattern, clippy::bool_comparison,
         clippy::needless_late_init, clippy::int_plus_one, clippy::manual_clamp, clippy::neg_cmp_op_on_partial_ord,
         clippy::inherent_to_string, clippy::wrong_self_convention,
         clippy::write_with_newline)]
// Added in M3 wave 2a: upstream literals (0x3fffFFFF), the mirrored module path search::search,
// and statement shapes kept verbatim (is_some() checks before expect, `mh[last_idx - 0]`,
// `vector<T>&` params as &mut Vec<T> per PORTING.md §3, the TSV-prescribed
// SearchNodeChildrenReference::index, `&*box` reborrows).
#![allow(clippy::mixed_case_hex_literals, clippy::module_inception, clippy::unnecessary_unwrap,
         clippy::should_implement_trait, clippy::ptr_arg, clippy::identity_op, clippy::borrow_deref_ref)]
// Added in M3 wave 2b: upstream's `size() > 0`, `== ""`, `cond ? true : false`, try/catch-as-match, map
// iteration over (key, value) and push-after-construct shapes kept verbatim (setup.cpp, analysis.cpp).
#![allow(clippy::len_zero, clippy::comparison_to_empty, clippy::needless_bool, clippy::single_match,
         clippy::for_kv_map, clippy::vec_init_then_push)]
#![deny(clippy::await_holding_refcell_ref)]
// PORTING.md §6 item 3: transcendentals only through port::math (list in clippy.toml).
#![deny(clippy::disallowed_methods)]

pub mod port;
pub mod core;
pub mod game;
pub mod neuralnet;
pub mod search;
pub mod program;
pub mod command;
