//! stdin lines and stdout / stderr writes.

use std::io::Write as _;



/// `getline(cin, line)` as a Stream of byte lines (without '\n'). A reader thread sends every complete
/// line of one read() as one batch, so a burst that the client wrote at once reaches the request loop
/// at once (it is "immediately available", as upstream's buffered getline sees it). A final line without
/// '\n' is returned too.
pub fn stdin_lines() -> impl futures::Stream<Item = Vec<u8>> + Unpin {
  use futures::StreamExt;
  let (tx, rx) = futures::channel::mpsc::unbounded::<Vec<Vec<u8>>>();
  std::thread::spawn(move || {
    use std::io::Read;
    let stdin = std::io::stdin();
    let mut lock = stdin.lock();
    let mut pending: Vec<u8> = Vec::new();
    let mut buf = vec![0u8; 1 << 16];
    loop {
      match lock.read(&mut buf) {
        Ok(0) => {
          if !pending.is_empty() {
            let _ = tx.unbounded_send(vec![std::mem::take(&mut pending)]);
          }
          break;
        }
        Ok(n) => {
          pending.extend_from_slice(&buf[..n]);
          let mut batch: Vec<Vec<u8>> = Vec::new();
          while let Some(p) = pending.iter().position(|&c| c == b'\n') {
            let rest = pending.split_off(p + 1);
            let mut line = std::mem::replace(&mut pending, rest);
            line.pop();
            batch.push(line);
          }
          if !batch.is_empty() && tx.unbounded_send(batch).is_err() {
            break;
          }
        }
        Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
        Err(_) => break,
      }
    }
  });
  rx.flat_map(futures::stream::iter)
}

/// `cout << message << endl`.
pub fn write_stdout_line(message: &str) {
  let stdout = std::io::stdout();
  let mut lock = stdout.lock();
  let _ = lock.write_all(message.as_bytes());
  let _ = lock.write_all(b"\n");
  let _ = lock.flush();
}

/// `cerr << text`.
pub fn write_stderr(text: &str) {
  let _ = std::io::stderr().write_all(text.as_bytes());
}
