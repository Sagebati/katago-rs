//! `katago-rs analysis -model <net> -config <cfg> [-override-config ...]`: the ported KataGo analysis
//! engine (katago_core::command::analysis) over stdin/stdout with the burn backend, so katago-ws can point
//! its `binary` at it. See the crate docs (lib.rs) for the model-file limitation. Native only.
//!
//! The flow follows upstream's main.cpp (`main` -> `handleSubcommand`) and the start of
//! `MainCmds::analysis` (TCLAP parsing, `cmd.getModelFile()`, `cmd.getConfig(cfg)`); an exception escaping
//! them is std::terminate's message and abort(), as upstream's Linux build (no catch in main).

use std::rc::Rc;

use futures::executor::LocalPool;
use katago_cli::BurnHost;
use katago_cli::ParsedArgs;
use katago_core::command::analysis::AnalysisArgs;
use katago_core::command::analysis::AnalysisHost;
use katago_core::core::global::KataError;
use katago_core::port::executor::Spawner;

/// main.cpp's printHelp (stdout), for the subcommands katago-rs has.
fn print_help(args: &[String]) {
  let prog = args.first().map(String::as_str).unwrap_or("./katago");
  println!("Usage: {prog} SUBCOMMAND ");
  println!();
  println!("---Subcommands (katago-rs)------------------");
  println!();
  println!("analysis : Runs an engine designed to analyze entire games in parallel.");
  println!("version : Print version and exit.");
  println!();
}

/// `MainCmds::analysis` up to the library call: Err is an exception escaping it (uncaught upstream).
fn analysis_main(sub_args: &[String]) -> Result<i32, KataError> {
  let cli = match katago_cli::parse_args(&sub_args[1..]) {
    Ok(ParsedArgs::Run(c)) => c,
    Ok(ParsedArgs::Help) => {
      print!("{}", katago_cli::HELP_TEXT);
      return Ok(0);
    }
    Ok(ParsedArgs::Version) => {
      print!("{}", katago_cli::version_text());
      return Ok(0);
    }
    Err(e) => {
      // TCLAP StdOutput::failure, then exit(1) inside CmdLine::parse.
      katago_cli::write_stderr(&e.failure_text());
      return Ok(1);
    }
  };
  let model_file = katago_cli::get_model_file(&cli)?;
  katago_cli::install_clock_and_sleep();
  let cfg = katago_cli::load_config(&cli.config_files, &cli.override_configs)?;
  katago_cli::install_logger();

  let mut pool = LocalPool::new();
  let spawner: Rc<dyn Spawner> = Rc::new(pool.spawner());
  let nn_host = BurnHost::new(&cli.backend, spawner)?;
  let mut input = katago_cli::stdin_lines();
  let host = AnalysisHost {
    args: AnalysisArgs {
      model_file,
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
    init_logger: Box::new(katago_cli::init_logger),
  };
  let result = pool.run_until(katago_core::command::analysis::analysis(cfg, host));
  // The NN server tasks end on their next poll after kill_server_threads (PORTING.md §11). After an
  // exception the process terminates right away, as upstream (no "GPU ... finishing" log line).
  if result.is_ok() {
    pool.run_until_stalled();
  }
  result
}

/// main.cpp `main` + `handleSubcommand`.
fn run(args: Vec<String>) -> i32 {
  if args.len() < 2 {
    print_help(&args);
    return 0;
  }
  let cmd_arg = args[1].as_str();
  if matches!(cmd_arg, "-h" | "--h" | "-help" | "--help" | "help") {
    print_help(&args);
    return 0;
  }
  match cmd_arg {
    // subArgs = args[1..]: TCLAP's progName is "analysis".
    "analysis" => match analysis_main(&args[1..]) {
      Ok(code) => code,
      Err(e) => katago_cli::terminate_uncaught(&e),
    },
    "version" => {
      // Version::getKataGoVersionFullInfo
      print!("{}", katago_cli::version_text().trim_start_matches("\nanalysis  version: ").trim_end_matches('\n'));
      println!();
      0
    }
    other => {
      println!("Unknown subcommand: {other}");
      print_help(&args);
      1
    }
  }
}

fn main() {
  let args: Vec<String> = std::env::args().collect();
  // Global::fatalError: "\nFATAL ERROR:\n<s>" to stdout, then quick_exit(EXIT_FAILURE).
  let default_hook = std::panic::take_hook();
  std::panic::set_hook(Box::new(move |info| {
    let msg: Option<&str> = info.payload().downcast_ref::<String>().map(String::as_str).or_else(|| info.payload().downcast_ref::<&str>().copied());
    if let Some(m) = msg.filter(|m| m.starts_with("\nFATAL ERROR:\n")) {
      katago_cli::write_stdout_line(m);
      std::process::exit(1);
    }
    default_hook(info);
  }));
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
