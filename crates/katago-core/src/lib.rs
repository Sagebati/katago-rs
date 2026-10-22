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
         clippy::needless_return, clippy::comparison_chain)]
#![deny(clippy::await_holding_refcell_ref)]
// PORTING.md §6 item 3: transcendentals only through port::math (list in clippy.toml).
#![deny(clippy::disallowed_methods)]

pub mod port;
pub mod core;
pub mod game;
