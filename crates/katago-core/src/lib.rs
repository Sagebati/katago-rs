//! katago-core: the 1:1 port of KataGo's `cpp/` tree (upstream v1.18.2).
//!
//! Module layout mirrors `upstream/KataGo/cpp/` (see port-manifest.toml); `port/` holds the
//! helpers that have no upstream file (docs/PORTING.md appendix).
// Lints the fidelity rules trigger (PORTING.md §11b): add to this list, never sprinkle
// per-item allows for these. `expect` so that a lint that stops firing is reported.
// Upstream's identifiers (nninputs.cpp's NUM_FEATURES_SPATIAL_V7, camelCase members, ...).
#![expect(non_snake_case, non_upper_case_globals, non_camel_case_types)]
// Upstream's signatures: long argument lists, `vector<T>&` params as &mut Vec<T> (PORTING.md §3),
// the mirrored module path search::search, the TSV-prescribed SearchNodeChildrenReference::index,
// and methods named like std traits' (toString, ...).
#![expect(clippy::too_many_arguments, clippy::ptr_arg, clippy::new_without_default,
          clippy::should_implement_trait, clippy::inherent_to_string,
          clippy::inherent_to_string_shadow_display, clippy::wrong_self_convention,
          clippy::module_inception)]
// Upstream's numeric literals (twoOverPi trips the deny-by-default approx_constant, 0x3fffFFFF)
// and arithmetic shapes kept verbatim (`mh[last_idx - 0]`, `x = x + y`, `a <= b - 1`,
// rotates as `(x >> r) | (x << (64 - r))` in hash.cpp and rand_helpers.cpp).
#![expect(clippy::approx_constant, clippy::excessive_precision, clippy::mixed_case_hex_literals,
          clippy::identity_op, clippy::assign_op_pattern, clippy::int_plus_one,
          clippy::manual_clamp, clippy::neg_cmp_op_on_partial_ord, clippy::manual_range_contains,
          clippy::manual_swap, clippy::manual_rotate)]
// Upstream's control flow and statement shapes kept verbatim: nested ifs, index loops,
// `size() > 0`, `== ""`, `== false`, `cond ? true : false`, try/catch-as-match, is_some() checks
// before expect, `&*box` reborrows, push-after-construct, default-then-assign, map iteration over
// (key, value), a printf("...\n") written as write!, case-insensitive compares spelled out
// byte by byte (global.cpp).
#![expect(clippy::collapsible_if, clippy::collapsible_else_if, clippy::needless_range_loop,
          clippy::nonminimal_bool, clippy::needless_return, clippy::comparison_chain,
          clippy::if_same_then_else, clippy::field_reassign_with_default,
          clippy::bool_comparison, clippy::needless_late_init, clippy::write_with_newline,
          clippy::unnecessary_unwrap, clippy::len_zero,
          clippy::comparison_to_empty, clippy::needless_bool, clippy::single_match,
          clippy::for_kv_map, clippy::vec_init_then_push, clippy::manual_ignore_case_cmp)]
// port/cpp.rs keeps its unit tests next to the helpers they cover, followed by more helpers.
#![cfg_attr(test, expect(clippy::items_after_test_module))]
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
