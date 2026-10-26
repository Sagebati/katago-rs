//! Host side of the native `katago-rs analysis` drop-in (plan §1): everything `MainCmds::analysis`
//! gets from the process in upstream KataGo, for the ported library function
//! `katago_core::command::analysis::analysis`:
//! - argv: a TCLAP 1.2.5 emulation for the analysis subcommand's arguments (`-model`, `-config`...,
//!   `-override-config`..., `-human-model`, `-analysis-threads`, `-quit-without-waiting`, `--`, `-version`,
//!   `-h`/`-help`) with upstream's error texts (`PARSE ERROR: ...` + brief usage, exit 1), plus the
//!   katago-rs-only `-backend flex|wgpu|rocm` (those compiled in, see the features) -> [`parse_args`] ([`args`]);
//! - `KataGoCommandLine::getModelFile` (default model lookup) -> [`get_model_file`], `getConfig` (config
//!   files, `@include` through the file system, `-override-config` with `Setup::getMutexKeySets`) ->
//!   [`load_config`] ([`config`]);
//! - the Logger's outputs (stdout / stderr / `logFile` / `logDir` / `logDirDated`, local-time stamps) ->
//!   [`install_logger`] + [`init_logger`] (AnalysisHost::init_logger) ([`logger`]);
//! - the clock and the timed wait (`core::timer::set_clock`, `port::executor::set_sleep`) -> [`install_clock_and_sleep`] ([`timer`]);
//! - an exception escaping `MainCmds::analysis` (no catch in upstream's main on Linux): libstdc++'s
//!   terminate message and `abort()` -> [`terminate_uncaught`] ([`terminate`]);
//! - the NN backend: the burn networks compiled into katago-nn-burn -> [`BurnHost`] ([`host`]);
//! - stdin lines and stdout / stderr writes -> [`stdin_lines`], [`write_stdout_line`], [`write_stderr`] ([`stdio`]).
//!
//! Model files: there is no runtime `.bin.gz` loader. `-model` names one of the networks compiled into
//! katago-nn-burn (`g170-b6c96`, `g170e-b10c128`, `kata1-b18c384nbt`, and the human-SL net
//! `b18c384nbt-humanv0` for `-human-model` or `-model`), matched on the file name (with or
//! without directories and `.bin.gz` / `.txt.gz` / `.onnx` / `.bpk`, and the full internal name such as
//! `g170-b6c96-s175395328-d26788732`; a symlink is also matched by its target's name). The weights are the
//! `.bpk` generated at build time (`katago_nn_burn::<net>::BPK_PATH`) unless `-model` is itself a `.bpk`
//! file, which is then read. Any other network is an error (upstream: an unloadable model file).
//! Not a port of upstream code (idiomatic Rust at the API edge, PORTING.md §1).

pub mod args;
pub mod config;
pub mod host;
pub mod logger;
pub mod stdio;
pub mod terminate;
pub mod timer;

pub use args::*;
pub use config::*;
pub use host::*;
pub use logger::*;
pub use stdio::*;
pub use terminate::*;
pub use timer::*;
