//! Single-threaded async replacements for upstream's threads, condition variables and
//! `ThreadSafeQueue` (docs/PORTING.md §11, Appendix). Not a port of any upstream file, except that
//! [`AsyncQueue`] keeps the API and semantics of `ThreadSafeContainer` / `ThreadSafeQueue`
//! (cpp/core/threadsafequeue.h) with every blocking wait turned into an `.await`.
//!
//! Everything here is `!Send`: one OS thread, many futures ("virtual threads") on the host's
//! local executor. A waiter re-checks its condition every time it is polled, so spurious
//! wakeups are harmless, exactly like `while(!cond) cv.wait(lock)`.

use std::cell::Cell;
use std::cell::RefCell;
use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;
use std::task::Context;
use std::task::Poll;
use std::task::Waker;

pub use futures::future::LocalBoxFuture;

/// `std::this_thread::yield()`: returns `Pending` once (after waking itself), so every other
/// ready future on the executor runs before this one continues.
pub fn yield_now() -> YieldNow {
  YieldNow { yielded: false }
}

/// Future returned by [`yield_now`].
pub struct YieldNow {
  yielded: bool,
}

impl Future for YieldNow {
  type Output = ();
  fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
    if self.yielded {
      return Poll::Ready(());
    }
    self.yielded = true;
    cx.waker().wake_by_ref();
    Poll::Pending
  }
}

/// `std::condition_variable` without its mutex: `cv.wait_until(|| cond).await` is
/// `while(!cond) cv.wait(lock);`, and `notify_all()` wakes every waiter.
/// (There is no `notify_one`: upstream's code on the ported paths only uses `notify_all`.)
#[derive(Default)]
pub struct WakerList {
  wakers: RefCell<Vec<Waker>>,
}

impl WakerList {
  pub fn new() -> WakerList {
    WakerList { wakers: RefCell::new(Vec::new()) }
  }

  /// Resolves once `cond()` is true. `cond` is evaluated at the first poll and after every wakeup.
  pub fn wait_until<F: FnMut() -> bool>(&self, cond: F) -> WaitUntil<'_, F> {
    WaitUntil { list: self, cond }
  }

  /// Wakes every future currently waiting in `wait_until`.
  pub fn notify_all(&self) {
    let wakers: Vec<Waker> = std::mem::take(&mut *self.wakers.borrow_mut());
    for w in wakers {
      w.wake();
    }
  }

  /// Number of registered waiters (tests and debugging).
  pub fn num_waiting(&self) -> usize {
    self.wakers.borrow().len()
  }

  fn register(&self, waker: &Waker) {
    let mut wakers = self.wakers.borrow_mut();
    if !wakers.iter().any(|w| w.will_wake(waker)) {
      wakers.push(waker.clone());
    }
  }
}

/// Future returned by [`WakerList::wait_until`].
pub struct WaitUntil<'a, F: FnMut() -> bool> {
  list: &'a WakerList,
  cond: F,
}

impl<'a, F: FnMut() -> bool + Unpin> Future for WaitUntil<'a, F> {
  type Output = ();
  fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
    if (self.cond)() {
      return Poll::Ready(());
    }
    self.list.register(cx.waker());
    Poll::Pending
  }
}

/// `ThreadSafeQueue<T>` (threadsafequeue.h): a FIFO with the same names and semantics
/// (`close`, `setReadOnly`, `unsetReadOnly`, `forcePush`, `tryPop`, `waitPopUpToN`, ...).
/// `waitPopUpToN` is async: it waits until the queue is non-empty, closed or read-only, then pops
/// up to `n` elements, i.e. everything queued when the server is polled (PORTING.md §11).
/// PORT NOTE: the `maxSize` bound (`waitPush`, `notFullCondVar`) is not ported: the only queue on
/// the ported paths (`NNEvaluator::queryQueue`) uses the default unbounded size and `forcePush`.
pub struct AsyncQueue<T> {
  closed: Cell<bool>,
  read_only: Cell<bool>,
  elts: RefCell<VecDeque<T>>,
  not_empty: WakerList,
}

impl<T> Default for AsyncQueue<T> {
  fn default() -> Self {
    AsyncQueue::new()
  }
}

impl<T> AsyncQueue<T> {
  pub fn new() -> AsyncQueue<T> {
    AsyncQueue { closed: Cell::new(false), read_only: Cell::new(false), elts: RefCell::new(VecDeque::new()), not_empty: WakerList::new() }
  }

  /// `reserve(n)`
  pub fn reserve(&self, n: usize) {
    self.elts.borrow_mut().reserve(n);
  }

  /// `size()`
  pub fn size(&self) -> usize {
    self.elts.borrow().len()
  }

  /// Whether any queued element satisfies `f` (not in ThreadSafeQueue; for debug assertions, e.g.
  /// NNEvaluator::evaluate checks that a buf is not queued twice). O(size).
  pub fn any<F: FnMut(&T) -> bool>(&self, f: F) -> bool {
    self.elts.borrow().iter().any(f)
  }

  /// `isClosed()`
  pub fn is_closed(&self) -> bool {
    self.closed.get()
  }

  /// `isReadOnly()`
  pub fn is_read_only(&self) -> bool {
    self.read_only.get()
  }

  /// `close()`: drops queued elements; every waiter unblocks and further reads/writes fail.
  pub fn close(&self) {
    self.closed.set(true);
    let dropped: VecDeque<T> = std::mem::take(&mut *self.elts.borrow_mut());
    drop(dropped);
    self.not_empty.notify_all();
  }

  /// `setReadOnly()`: waiters unblock, writes fail, reads drain what is left without blocking.
  pub fn set_read_only(&self) {
    self.read_only.set(true);
    self.not_empty.notify_all();
  }

  /// `unsetReadOnly()`
  pub fn unset_read_only(&self) {
    self.read_only.set(false);
  }

  /// `forcePush(elt)`: false if closed or read-only.
  pub fn force_push(&self, elt: T) -> bool {
    if self.closed.get() || self.read_only.get() {
      return false;
    }
    let size = {
      let mut elts = self.elts.borrow_mut();
      elts.push_back(elt);
      elts.len()
    };
    if size == 1 {
      self.not_empty.notify_all();
    }
    true
  }

  /// `tryPop(buf)`: false if closed or empty.
  pub fn try_pop(&self, buf: &mut Option<T>) -> bool {
    if self.closed.get() {
      return false;
    }
    let mut elts = self.elts.borrow_mut();
    if elts.is_empty() {
      return false;
    }
    *buf = elts.pop_front();
    true
  }

  /// `waitPopUpToN(buf, n)`: false if nothing was popped (closed, or empty and read-only).
  pub async fn wait_pop_up_to_n(&self, buf: &mut Vec<T>, n: usize) -> bool {
    self.not_empty.wait_until(|| self.closed.get() || self.read_only.get() || !self.elts.borrow().is_empty()).await;
    if self.closed.get() {
      return false;
    }
    let mut elts = self.elts.borrow_mut();
    let size = elts.len();
    if size == 0 {
      return false;
    }
    let num_to_pop = std::cmp::min(size, n);
    for _ in 0..num_to_pop {
      buf.push(elts.pop_front().expect("size checked above"));
    }
    true
  }
}

/// The host's local executor, injected where upstream spawns a `std::thread`
/// (`NNEvaluator::spawnServerThreads`). Natively `futures::executor::LocalSpawner`; in the
/// browser `wasm_bindgen_futures::spawn_local`.
pub trait Spawner {
  fn spawn_local(&self, f: LocalBoxFuture<'static, ()>);
}

impl Spawner for futures::executor::LocalSpawner {
  fn spawn_local(&self, f: LocalBoxFuture<'static, ()>) {
    futures::task::LocalSpawnExt::spawn_local(self, f).expect("LocalPool is alive while its spawner is used");
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use futures::executor::LocalPool;
  use std::rc::Rc;

  #[test]
  fn yield_now_lets_others_run() {
    let mut pool = LocalPool::new();
    let log: Rc<RefCell<Vec<&'static str>>> = Rc::new(RefCell::new(Vec::new()));
    let (l1, l2) = (log.clone(), log.clone());
    let sp = pool.spawner();
    Spawner::spawn_local(
      &sp,
      Box::pin(async move {
        l1.borrow_mut().push("a1");
        yield_now().await;
        l1.borrow_mut().push("a2");
      }),
    );
    Spawner::spawn_local(
      &sp,
      Box::pin(async move {
        l2.borrow_mut().push("b1");
      }),
    );
    pool.run();
    assert_eq!(*log.borrow(), vec!["a1", "b1", "a2"]);
  }

  #[test]
  fn waker_list_wait_until_and_notify_all() {
    let mut pool = LocalPool::new();
    let flag = Rc::new(Cell::new(false));
    let cv = Rc::new(WakerList::new());
    let done = Rc::new(Cell::new(0));
    let sp = pool.spawner();
    for _ in 0..3 {
      let (f, c, d) = (flag.clone(), cv.clone(), done.clone());
      Spawner::spawn_local(
        &sp,
        Box::pin(async move {
          c.wait_until(|| f.get()).await;
          d.set(d.get() + 1);
        }),
      );
    }
    pool.run_until_stalled();
    assert_eq!(done.get(), 0);
    assert_eq!(cv.num_waiting(), 3);
    // A notify without the condition becoming true re-parks everybody (spurious wakeup).
    cv.notify_all();
    pool.run_until_stalled();
    assert_eq!(done.get(), 0);
    flag.set(true);
    cv.notify_all();
    pool.run_until_stalled();
    assert_eq!(done.get(), 3);
  }

  #[test]
  fn async_queue_semantics() {
    let mut pool = LocalPool::new();
    let q: Rc<AsyncQueue<i32>> = Rc::new(AsyncQueue::new());
    let got: Rc<RefCell<Vec<Vec<i32>>>> = Rc::new(RefCell::new(Vec::new()));
    let sp = pool.spawner();
    {
      let (q, got) = (q.clone(), got.clone());
      Spawner::spawn_local(
        &sp,
        Box::pin(async move {
          loop {
            let mut buf = Vec::new();
            if !q.wait_pop_up_to_n(&mut buf, 2).await {
              break;
            }
            got.borrow_mut().push(buf);
          }
        }),
      );
    }
    pool.run_until_stalled();
    assert!(got.borrow().is_empty());
    assert!(q.force_push(1));
    assert!(q.force_push(2));
    assert!(q.force_push(3));
    pool.run_until_stalled();
    assert_eq!(*got.borrow(), vec![vec![1, 2], vec![3]]);
    // Read-only: writes fail, the waiter drains what is left and then gets false.
    q.set_read_only();
    assert!(!q.force_push(4));
    pool.run_until_stalled();
    assert_eq!(got.borrow().len(), 2);
    q.unset_read_only();
    assert!(q.force_push(5));
    let mut one = None;
    assert!(q.try_pop(&mut one));
    assert_eq!(one, Some(5));
    assert!(!q.try_pop(&mut one));
    q.close();
    assert!(!q.force_push(6));
    assert!(q.is_closed());
  }

  #[test]
  fn async_queue_close_wakes_waiter() {
    let mut pool = LocalPool::new();
    let q: Rc<AsyncQueue<i32>> = Rc::new(AsyncQueue::new());
    let result = Rc::new(Cell::new(None));
    let sp = pool.spawner();
    {
      let (q, r) = (q.clone(), result.clone());
      Spawner::spawn_local(
        &sp,
        Box::pin(async move {
          let mut buf = Vec::new();
          r.set(Some(q.wait_pop_up_to_n(&mut buf, 8).await));
        }),
      );
    }
    pool.run_until_stalled();
    assert_eq!(result.get(), None);
    q.close();
    pool.run_until_stalled();
    assert_eq!(result.get(), Some(false));
  }
}
