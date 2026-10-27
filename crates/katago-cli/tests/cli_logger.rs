//! katago-cli's log subscriber in-process: which `tracing` events and which `log` records of the dependencies (burn,
//! cubecl, wgpu... reach it through LogTracer) are written, and that nothing is before `init_logger`. The byte format of
//! the lines is tests/cli_e2e.rs's and the analysis-protocol corpora. Not a port of upstream code.

use katago_core::command::analysis::LoggerSettings;

#[test]
fn log_filter_and_bridge() {
  let dir = std::env::temp_dir().join(format!("katago-rs-cli-logger-{}", std::process::id()));
  std::fs::create_dir_all(&dir).unwrap();
  let file = dir.join("test.log");
  let _ = std::fs::remove_file(&file);

  // Until init_logger runs there is no Logger upstream: dropped.
  katago_cli::install_logger();
  tracing::info!("before init");
  log::warn!("before init (log)");

  katago_cli::init_logger(&LoggerSettings {
    log_file: Some(file.display().to_string()),
    ..Default::default()
  })
  .unwrap();
  tracing::info!("engine info {}", 1);
  tracing::debug!("engine debug");
  tracing::warn!("engine warn");
  tracing::error!("engine error");
  // The GPU stack: warnings and errors only.
  log::info!(target: "wgpu_core::device", "wgpu info");
  log::warn!(target: "wgpu_core::device", "wgpu warn");
  log::info!(target: "cubecl_runtime::tune", "cubecl info");
  log::error!(target: "burn_cubecl::kernel", "burn error");
  // Any other dependency: info and up.
  log::info!(target: "some_dep", "dep info");
  log::debug!(target: "some_dep", "dep debug");

  let got = std::fs::read_to_string(&file).unwrap();
  let _ = std::fs::remove_dir_all(&dir);
  assert_eq!(got, ": engine info 1\n: engine warn\n: engine error\n: wgpu warn\n: burn error\n: dep info\n");
}
