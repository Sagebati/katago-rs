//! An exception escaping `MainCmds::analysis`: libstdc++'s terminate message and `abort()`.

use std::io::Write as _;

use katago_core::core::global::KataError;

/// The C++ type name libstdc++'s verbose terminate handler prints for the exception behind `e`.
pub fn exception_type_name(e: &KataError) -> String {
  match e {
    KataError::StringError(_) => "StringError".to_string(),
    KataError::IOError(_) => "IOError".to_string(),
    KataError::ConfigParsingError(_) => "ConfigParsingError".to_string(),
    KataError::ValueError(_) => "ValueError".to_string(),
    KataError::CommandError(_) => "CommandError".to_string(),
    KataError::NlohmannJson(m) => {
      // "[json.exception.<class>.<id>] ..."
      let class = m.strip_prefix("[json.exception.").and_then(|r| r.split('.').next()).unwrap_or("exception");
      format!("nlohmann::detail::{class}")
    }
  }
}

/// An exception escaping `main` (upstream's main has no catch on Linux): `std::terminate` prints
/// "terminate called after throwing an instance of '<type>'\n  what():  <what>\n" and aborts (SIGABRT,
/// shell exit status 134).
pub fn terminate_uncaught(e: &KataError) -> ! {
  let _ = std::io::stdout().flush();
  let msg = format!(
    "terminate called after throwing an instance of '{}'\n  what():  {}\n",
    exception_type_name(e),
    e.what()
  );
  let _ = std::io::stderr().write_all(msg.as_bytes());
  std::process::abort()
}
