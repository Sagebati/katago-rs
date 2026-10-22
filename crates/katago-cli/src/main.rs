//! `katago-rs analysis -model <net> -config <cfg> [-override-config ...]`: the ported KataGo analysis
//! engine (katago_core::command::analysis) over stdin/stdout with the burn backend, so katago-ws can point
//! its `binary` at it. See the crate docs (lib.rs) for the model-file limitation. Native only.

use std::rc::Rc;

use futures::executor::LocalPool;
use katago_cli::BurnHost;
use katago_cli::USAGE;
use katago_core::command::analysis::AnalysisArgs;
use katago_core::command::analysis::AnalysisHost;
use katago_core::port::executor::Spawner;

fn run(args: Vec<String>) -> i32 {
  if args.len() < 2 || args[1] != "analysis" {
    eprintln!("{USAGE}");
    return 1;
  }
  let cli = match katago_cli::parse_args(&args[2..]) {
    Ok(c) => c,
    Err(e) => {
      // upstream: `cerr << "Error: " << e.error() << " for argument " << e.argId() << endl; return 1;`
      eprintln!("Error: {e}");
      eprintln!("{USAGE}");
      return 1;
    }
  };

  katago_cli::install_clock_and_sleep();
  let cfg = match katago_cli::load_config(&cli.config_files, &cli.override_configs) {
    Ok(c) => c,
    Err(e) => {
      eprintln!("Uncaught exception: {}", e.what());
      return 1;
    }
  };
  katago_cli::install_logger(&cfg);

  let mut pool = LocalPool::new();
  let spawner: Rc<dyn Spawner> = Rc::new(pool.spawner());
  let nn_host = match BurnHost::new(&cli.backend, spawner) {
    Ok(h) => h,
    Err(e) => {
      eprintln!("Uncaught exception: {}", e.what());
      return 1;
    }
  };
  let mut input = katago_cli::stdin_lines();
  let host = AnalysisHost {
    args: AnalysisArgs {
      model_file: cli.model_file.clone(),
      human_model_file: cli.human_model_file.clone(),
      num_analysis_threads_cmdline_specified: cli.num_analysis_threads.is_some(),
      num_analysis_threads_cmdline: cli.num_analysis_threads.unwrap_or(0),
      quit_without_waiting: cli.quit_without_waiting,
      override_configs: cli.override_configs.clone(),
    },
    input: &mut input,
    write_stdout: Box::new(katago_cli::write_stdout_line),
    write_stderr: Box::new(katago_cli::write_stderr),
    nn_host: &nn_host,
  };
  let result = pool.run_until(katago_core::command::analysis::analysis(cfg, host));
  // The NN server tasks end on their next poll after kill_server_threads (PORTING.md §11).
  pool.run_until_stalled();
  match result {
    Ok(code) => code,
    Err(e) => {
      eprintln!("Uncaught exception: {}", e.what());
      1
    }
  }
}

fn main() {
  let args: Vec<String> = std::env::args().collect();
  // Big stack: Search (~40 KB), BoardHistory (~34 KB) per bot and the recursive playout futures.
  let code = std::thread::Builder::new()
    .name("katago-rs".to_string())
    .stack_size(256 << 20)
    .spawn(move || run(args))
    .expect("spawn the main thread")
    .join()
    .unwrap_or(1);
  // The stdin reader thread may still be blocked in read(); exit without joining it.
  std::process::exit(code);
}
