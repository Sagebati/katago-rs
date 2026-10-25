//! The clock and the timed wait the engine runs on (`core::timer::set_clock`, `port::executor::set_sleep`).

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::OnceLock;
use std::task::Poll;
use std::task::Waker;
use std::time::Instant;

use katago_core::port::executor::LocalBoxFuture;


static START: OnceLock<Instant> = OnceLock::new();

fn clock() -> f64 {
  START.get_or_init(Instant::now).elapsed().as_secs_f64()
}

/// The shared timer behind [`sleep`]: one thread waits for the earliest deadline of a map keyed by
/// (deadline, id) and wakes that future. A dropped [`Sleep`] removes its entry, so abandoned waits (an
/// analysis callback loop dropped when its search ends, with a period up to 1e6 s) leave nothing behind.
struct TimerState {
  /// (deadline, id) -> (fired, waker)
  entries: BTreeMap<(Instant, u64), SleepSlot>,
  next_id: u64,
}

struct Timer {
  state: Mutex<TimerState>,
  changed: std::sync::Condvar,
}

static TIMER: OnceLock<&'static Timer> = OnceLock::new();
static TIMER_THREADS_STARTED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn timer() -> &'static Timer {
  TIMER.get_or_init(|| {
    let t: &'static Timer = Box::leak(Box::new(Timer { state: Mutex::new(TimerState { entries: BTreeMap::new(), next_id: 0 }), changed: std::sync::Condvar::new() }));
    TIMER_THREADS_STARTED.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    std::thread::Builder::new()
      .name("katago-rs-timer".to_string())
      .spawn(move || {
        let mut st = t.state.lock().unwrap();
        loop {
          let now = Instant::now();
          // fire everything due
          while let Some((&key, _)) = st.entries.iter().next() {
            if key.0 > now {
              break;
            }
            let slot = st.entries.remove(&key).expect("first key exists");
            let mut slot = slot.lock().unwrap();
            slot.0 = true;
            if let Some(w) = slot.1.take() {
              w.wake();
            }
          }
          st = match st.entries.keys().next().map(|k| k.0) {
            Some(deadline) => t.changed.wait_timeout(st, deadline.saturating_duration_since(Instant::now())).unwrap().0,
            None => t.changed.wait(st).unwrap(),
          };
        }
      })
      .expect("spawn the timer thread");
    t
  })
}

/// Number of timer threads ever started (tests: one per process, whatever the number of sleeps).
pub fn timer_threads_started() -> usize {
  TIMER_THREADS_STARTED.load(std::sync::atomic::Ordering::SeqCst)
}

/// Number of pending (not fired, not dropped) timed waits.
pub fn pending_sleeps() -> usize {
  timer().state.lock().unwrap().entries.len()
}

/// (fired, waker) of one pending [`sleep`].
type SleepSlot = Arc<Mutex<(bool, Option<Waker>)>>;

/// The future of [`sleep`].
struct Sleep {
  key: (Instant, u64),
  slot: SleepSlot,
}

impl std::future::Future for Sleep {
  type Output = ();
  fn poll(self: std::pin::Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> Poll<()> {
    let mut slot = self.slot.lock().unwrap();
    if slot.0 {
      Poll::Ready(())
    } else {
      slot.1 = Some(cx.waker().clone());
      Poll::Pending
    }
  }
}

impl Drop for Sleep {
  fn drop(&mut self) {
    let t = timer();
    let mut st = t.state.lock().unwrap();
    if st.entries.remove(&self.key).is_some() {
      t.changed.notify_one();
    }
  }
}

/// A timed wait for `port::executor::sleep` on the shared timer thread (LocalPool parks its thread while
/// idle, and a wake from another thread unparks it). Durations beyond ~31 years are capped.
fn sleep(seconds: f64) -> LocalBoxFuture<'static, ()> {
  let secs = if seconds.is_finite() && seconds > 0.0 { seconds.min(1e9) } else { 0.0 };
  let deadline = Instant::now() + std::time::Duration::from_secs_f64(secs);
  let t = timer();
  let slot = Arc::new(Mutex::new((false, None)));
  let key = {
    let mut st = t.state.lock().unwrap();
    let key = (deadline, st.next_id);
    st.next_id += 1;
    st.entries.insert(key, slot.clone());
    key
  };
  t.changed.notify_one();
  Box::pin(Sleep { key, slot })
}

/// Installs the wall clock (`core::timer::set_clock`) and the timed wait (`port::executor::set_sleep`).
pub fn install_clock_and_sleep() {
  START.get_or_init(Instant::now);
  katago_core::core::timer::set_clock(clock);
  katago_core::port::executor::set_sleep(sleep);
}

#[cfg(test)]
mod tests {
  use super::*;

  fn num_os_threads() -> usize {
    std::fs::read_dir("/proc/self/task").map(|d| d.count()).unwrap_or(0)
  }

  /// Abandoned long waits (an analysis callback loop dropped when its search ends) leave no thread and
  /// no timer entry behind; short waits still fire.
  #[test]
  fn sleep_is_one_shared_timer_and_cancels_on_drop() {
    let short = sleep(0.01);
    let threads_before = num_os_threads();
    let pending_before = pending_sleeps();
    let abandoned: Vec<_> = (0..1000).map(|_| sleep(1e6)).collect();
    assert_eq!(pending_sleeps(), pending_before + 1000);
    assert!(num_os_threads() <= threads_before + 1, "{} -> {} threads", threads_before, num_os_threads());
    drop(abandoned);
    assert_eq!(pending_sleeps(), pending_before);
    let t0 = Instant::now();
    futures::executor::block_on(short);
    futures::executor::block_on(sleep(0.02));
    assert!(t0.elapsed().as_secs_f64() >= 0.015);
    assert_eq!(timer_threads_started(), 1);
  }
}
