//! port/executor.rs: the single-threaded stand-ins for std::condition_variable (`WakerList`) and
//! upstream's ThreadSafeQueue / ThreadSafePriorityQueue (`AsyncQueue`, `AsyncPriorityQueue`). The
//! interesting failures are lost wakeups, so every test registers its waiters BEFORE the event
//! (push, close, setReadOnly) and then checks that the pool, run until stalled, delivered it: a
//! waiter that is still parked with work (or a close) pending is a lost wakeup. FIFO tie-breaks of
//! the priority queue (equal keys pop in insertion order, std::priority_queue over unique keys in
//! analysis.cpp) and tryPop semantics are checked too.
//! Not a port of upstream code (upstream's queues are not unit-tested).

mod common;

use std::cell::Cell;
use std::cell::RefCell;
use std::rc::Rc;

use common::Cases;
use futures::executor::LocalPool;
use katago_core::port::executor::AsyncPriorityQueue;
use katago_core::port::executor::AsyncQueue;
use katago_core::port::executor::Spawner;
use katago_core::port::executor::WakerList;
use katago_core::port::executor::yield_now;

type Log = Rc<RefCell<Vec<(usize, i64)>>>;

/// `n` tasks looping `while let Some(x) = q.wait_pop().await { log.push((task, x)) }`; the number that ended is counted.
fn spawn_queue_consumers(pool: &LocalPool, q: &Rc<AsyncQueue<i64>>, n: usize, log: &Log, ended: &Rc<Cell<usize>>) {
  let sp = pool.spawner();
  for w in 0..n {
    let (q, log, ended) = (q.clone(), log.clone(), ended.clone());
    Spawner::spawn_local(
      &sp,
      Box::pin(async move {
        while let Some(x) = q.wait_pop().await {
          log.borrow_mut().push((w, x));
          yield_now().await;
        }
        ended.set(ended.get() + 1);
      }),
    );
  }
}

fn spawn_pq_consumers(pool: &LocalPool, q: &Rc<AsyncPriorityQueue<i64, i64>>, n: usize, log: &Log, ended: &Rc<Cell<usize>>) {
  let sp = pool.spawner();
  for w in 0..n {
    let (q, log, ended) = (q.clone(), log.clone(), ended.clone());
    Spawner::spawn_local(
      &sp,
      Box::pin(async move {
        while let Some((_, x)) = q.wait_pop().await {
          log.borrow_mut().push((w, x));
          yield_now().await;
        }
        ended.set(ended.get() + 1);
      }),
    );
  }
}

fn values(log: &Log) -> Vec<i64> {
  let mut v: Vec<i64> = log.borrow().iter().map(|(_, x)| *x).collect();
  v.sort();
  v
}

/// Waiters parked on an empty queue; pushes (one at a time, then bursts) must reach them all, with nothing left over.
#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn async_queue_waiters_registered_before_push_are_woken() {
  let mut cases = Cases::new("executor::async_queue_waiters_registered_before_push_are_woken");
  let mut pool = LocalPool::new();
  let q: Rc<AsyncQueue<i64>> = Rc::new(AsyncQueue::new());
  let (log, ended): (Log, Rc<Cell<usize>>) = (Rc::new(RefCell::new(Vec::new())), Rc::new(Cell::new(0)));
  spawn_queue_consumers(&pool, &q, 3, &log, &ended);
  pool.run_until_stalled();
  assert!(log.borrow().is_empty());
  // One push at a time: each is consumed even though all three waiters were parked before it.
  let mut expect: Vec<i64> = Vec::new();
  for round in 0..40i64 {
    let n = round % 5 + 1;
    for j in 0..n {
      assert!(q.force_push(round * 10 + j));
      expect.push(round * 10 + j);
    }
    pool.run_until_stalled();
    assert_eq!(q.size(), 0, "round {round}: elements left with idle waiters (lost wakeup)");
    assert_eq!(values(&log), expect, "round {round}");
    cases.ran();
  }
  // FIFO overall within a single consumer's view.
  for w in 0..3 {
    let mine: Vec<i64> = log.borrow().iter().filter(|(t, _)| *t == w).map(|(_, x)| *x).collect();
    assert!(mine.windows(2).all(|p| p[0] < p[1]), "task {w} saw {mine:?} out of order");
  }
  q.close();
  pool.run_until_stalled();
  assert_eq!(ended.get(), 3);
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn priority_queue_waiters_registered_before_push_are_woken() {
  let mut cases = Cases::new("executor::priority_queue_waiters_registered_before_push_are_woken");
  let mut pool = LocalPool::new();
  let q: Rc<AsyncPriorityQueue<i64, i64>> = Rc::new(AsyncPriorityQueue::new());
  let (log, ended): (Log, Rc<Cell<usize>>) = (Rc::new(RefCell::new(Vec::new())), Rc::new(Cell::new(0)));
  spawn_pq_consumers(&pool, &q, 3, &log, &ended);
  pool.run_until_stalled();
  assert!(log.borrow().is_empty());
  let mut expect: Vec<i64> = Vec::new();
  for round in 0..40i64 {
    let n = round % 5 + 1;
    for j in 0..n {
      assert!(q.force_push(round * 10 + j, round * 10 + j));
      expect.push(round * 10 + j);
    }
    pool.run_until_stalled();
    assert_eq!(q.size(), 0, "round {round}: elements left with idle waiters (lost wakeup)");
    assert_eq!(values(&log), expect, "round {round}");
    cases.ran();
  }
  q.close();
  pool.run_until_stalled();
  assert_eq!(ended.get(), 3);
}

/// close() with waiters registered before it: every waiter wakes and gets None; queued elements are dropped;
/// writes fail afterwards.
#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn close_wakes_waiters_registered_before_it() {
  let mut cases = Cases::new("executor::close_wakes_waiters_registered_before_it");
  {
    let mut pool = LocalPool::new();
    let q: Rc<AsyncQueue<i64>> = Rc::new(AsyncQueue::new());
    let (log, ended): (Log, Rc<Cell<usize>>) = (Rc::new(RefCell::new(Vec::new())), Rc::new(Cell::new(0)));
    spawn_queue_consumers(&pool, &q, 4, &log, &ended);
    pool.run_until_stalled();
    assert_eq!(ended.get(), 0);
    q.close();
    pool.run_until_stalled();
    assert_eq!(ended.get(), 4, "AsyncQueue::close left a waiter parked (lost wakeup)");
    assert!(log.borrow().is_empty());
    assert!(q.is_closed());
    assert!(!q.force_push(1));
    cases.ran();
  }
  {
    let mut pool = LocalPool::new();
    let q: Rc<AsyncPriorityQueue<i64, i64>> = Rc::new(AsyncPriorityQueue::new());
    let (log, ended): (Log, Rc<Cell<usize>>) = (Rc::new(RefCell::new(Vec::new())), Rc::new(Cell::new(0)));
    spawn_pq_consumers(&pool, &q, 4, &log, &ended);
    pool.run_until_stalled();
    assert_eq!(ended.get(), 0);
    q.close();
    pool.run_until_stalled();
    assert_eq!(ended.get(), 4, "AsyncPriorityQueue::close left a waiter parked (lost wakeup)");
    assert!(log.borrow().is_empty());
    assert!(q.is_closed());
    assert!(!q.force_push(1, 1));
    cases.ran();
  }
  // wait_pop_up_to_n parked before close returns false.
  {
    let mut pool = LocalPool::new();
    let q: Rc<AsyncQueue<i64>> = Rc::new(AsyncQueue::new());
    let res: Rc<Cell<Option<bool>>> = Rc::new(Cell::new(None));
    let (q2, r2) = (q.clone(), res.clone());
    Spawner::spawn_local(
      &pool.spawner(),
      Box::pin(async move {
        let mut buf = Vec::new();
        r2.set(Some(q2.wait_pop_up_to_n(&mut buf, 8).await));
      }),
    );
    pool.run_until_stalled();
    assert_eq!(res.get(), None);
    q.close();
    pool.run_until_stalled();
    assert_eq!(res.get(), Some(false));
    cases.ran();
  }
}

/// setReadOnly() with waiters registered before it: they wake and see an empty read-only queue (None), leftover
/// elements are still drained first, and after unsetReadOnly a new waiter blocks again and is woken by a push.
#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn set_read_only_wakes_waiters_registered_before_it() {
  let mut cases = Cases::new("executor::set_read_only_wakes_waiters_registered_before_it");
  {
    let mut pool = LocalPool::new();
    let q: Rc<AsyncQueue<i64>> = Rc::new(AsyncQueue::new());
    let (log, ended): (Log, Rc<Cell<usize>>) = (Rc::new(RefCell::new(Vec::new())), Rc::new(Cell::new(0)));
    spawn_queue_consumers(&pool, &q, 3, &log, &ended);
    pool.run_until_stalled();
    q.set_read_only();
    pool.run_until_stalled();
    assert_eq!(ended.get(), 3, "AsyncQueue::set_read_only left a waiter parked (lost wakeup)");
    assert!(!q.force_push(1));
    q.unset_read_only();
    // New waiters park again and are served by a later push.
    let (log2, ended2): (Log, Rc<Cell<usize>>) = (Rc::new(RefCell::new(Vec::new())), Rc::new(Cell::new(0)));
    spawn_queue_consumers(&pool, &q, 2, &log2, &ended2);
    pool.run_until_stalled();
    assert_eq!(ended2.get(), 0);
    assert!(q.force_push(7));
    pool.run_until_stalled();
    assert_eq!(values(&log2), vec![7]);
    assert_eq!(ended2.get(), 0);
    q.set_read_only();
    pool.run_until_stalled();
    assert_eq!(ended2.get(), 2);
    cases.ran();
  }
  {
    let mut pool = LocalPool::new();
    let q: Rc<AsyncPriorityQueue<i64, i64>> = Rc::new(AsyncPriorityQueue::new());
    let (log, ended): (Log, Rc<Cell<usize>>) = (Rc::new(RefCell::new(Vec::new())), Rc::new(Cell::new(0)));
    spawn_pq_consumers(&pool, &q, 3, &log, &ended);
    pool.run_until_stalled();
    q.set_read_only();
    pool.run_until_stalled();
    assert_eq!(ended.get(), 3, "AsyncPriorityQueue::set_read_only left a waiter parked (lost wakeup)");
    assert!(!q.force_push(1, 1));
    q.unset_read_only();
    let (log2, ended2): (Log, Rc<Cell<usize>>) = (Rc::new(RefCell::new(Vec::new())), Rc::new(Cell::new(0)));
    spawn_pq_consumers(&pool, &q, 2, &log2, &ended2);
    pool.run_until_stalled();
    assert_eq!(ended2.get(), 0);
    assert!(q.force_push(7, 7));
    pool.run_until_stalled();
    assert_eq!(values(&log2), vec![7]);
    q.set_read_only();
    pool.run_until_stalled();
    assert_eq!(ended2.get(), 2);
    cases.ran();
  }
  // Elements queued before setReadOnly are drained by waiters that were woken by it; only then None.
  {
    let mut pool = LocalPool::new();
    let q: Rc<AsyncQueue<i64>> = Rc::new(AsyncQueue::new());
    assert!(q.force_push(1));
    assert!(q.force_push(2));
    q.set_read_only();
    let (log, ended): (Log, Rc<Cell<usize>>) = (Rc::new(RefCell::new(Vec::new())), Rc::new(Cell::new(0)));
    spawn_queue_consumers(&pool, &q, 2, &log, &ended);
    pool.run_until_stalled();
    assert_eq!(values(&log), vec![1, 2]);
    assert_eq!(ended.get(), 2);
    cases.ran();
  }
}

/// The batch server: waiters in wait_pop_up_to_n parked before a push get everything queued when they run, at most n.
#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn wait_pop_up_to_n_waiter_registered_before_push() {
  let mut cases = Cases::new("executor::wait_pop_up_to_n_waiter_registered_before_push");
  let mut pool = LocalPool::new();
  let q: Rc<AsyncQueue<i64>> = Rc::new(AsyncQueue::new());
  let batches: Rc<RefCell<Vec<Vec<i64>>>> = Rc::new(RefCell::new(Vec::new()));
  {
    let (q, batches) = (q.clone(), batches.clone());
    Spawner::spawn_local(
      &pool.spawner(),
      Box::pin(async move {
        loop {
          let mut buf = Vec::new();
          if !q.wait_pop_up_to_n(&mut buf, 3).await {
            break;
          }
          batches.borrow_mut().push(buf);
        }
      }),
    );
  }
  pool.run_until_stalled();
  for i in 0..7 {
    assert!(q.force_push(i));
  }
  pool.run_until_stalled();
  assert_eq!(*batches.borrow(), vec![vec![0, 1, 2], vec![3, 4, 5], vec![6]]);
  assert!(q.force_push(7));
  pool.run_until_stalled();
  assert_eq!(batches.borrow().last().unwrap(), &vec![7]);
  q.set_read_only();
  pool.run_until_stalled();
  assert_eq!(batches.borrow().len(), 4);
  cases.ran();
}

/// Equal keys pop in insertion order (FIFO tie-break), by tryPop and by waitPop, also across later pushes; the
/// greatest key still wins over earlier insertions.
#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn priority_queue_equal_keys_are_fifo() {
  let mut cases = Cases::new("executor::priority_queue_equal_keys_are_fifo");
  let q: AsyncPriorityQueue<i32, i32> = AsyncPriorityQueue::new();
  for v in 0..50 {
    assert!(q.force_push(1, v));
  }
  assert!(q.force_push(0, 1000));
  assert!(q.force_push(2, 2000));
  assert_eq!(q.try_pop(), Some((2, 2000)));
  for v in 0..25 {
    assert_eq!(q.try_pop(), Some((1, v)));
  }
  // More equal keys pushed after some were popped queue up behind the older ones.
  for v in 50..60 {
    assert!(q.force_push(1, v));
  }
  for v in 25..60 {
    assert_eq!(q.try_pop(), Some((1, v)));
  }
  assert_eq!(q.try_pop(), Some((0, 1000)));
  assert_eq!(q.try_pop(), None);
  cases.ran();
  // Several keys interleaved: pops come out by key descending, FIFO within each key.
  let q: AsyncPriorityQueue<(i64, i64), usize> = AsyncPriorityQueue::new();
  let mut model: Vec<((i64, i64), usize)> = Vec::new();
  for i in 0..300usize {
    let key = ((i as i64 * 7) % 4, (i as i64 * 3) % 2);
    assert!(q.force_push(key, i));
    model.push((key, i));
  }
  // stable sort by key descending keeps insertion order within a key
  model.sort_by_key(|a| std::cmp::Reverse(a.0));
  for (k, v) in model {
    assert_eq!(q.try_pop(), Some((k, v)));
  }
  assert_eq!(q.size(), 0);
  cases.ran();
  // Through waitPop on a woken waiter, too.
  let mut pool = LocalPool::new();
  let q: Rc<AsyncPriorityQueue<i64, i64>> = Rc::new(AsyncPriorityQueue::new());
  let (log, ended): (Log, Rc<Cell<usize>>) = (Rc::new(RefCell::new(Vec::new())), Rc::new(Cell::new(0)));
  spawn_pq_consumers(&pool, &q, 1, &log, &ended);
  pool.run_until_stalled();
  for v in 0..20 {
    assert!(q.force_push(5, v));
  }
  pool.run_until_stalled();
  let got: Vec<i64> = log.borrow().iter().map(|(_, x)| *x).collect();
  assert_eq!(got, (0..20).collect::<Vec<i64>>());
  cases.ran();
}

/// tryPop: false/None on an empty queue without blocking, FIFO / greatest-first on a non-empty one, false/None after
/// close (even if elements were queued: close drops them), and it still drains a read-only queue.
#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn try_pop_semantics() {
  let mut cases = Cases::new("executor::try_pop_semantics");
  let q: AsyncQueue<i32> = AsyncQueue::new();
  let mut buf = None;
  assert!(!q.try_pop(&mut buf));
  assert_eq!(buf, None);
  for v in [3, 1, 2] {
    assert!(q.force_push(v));
  }
  assert!(q.try_pop(&mut buf));
  assert_eq!(buf, Some(3));
  q.set_read_only();
  assert!(q.try_pop(&mut buf));
  assert_eq!(buf, Some(1));
  assert!(q.try_pop(&mut buf));
  assert_eq!(buf, Some(2));
  assert!(!q.try_pop(&mut buf));
  assert_eq!(buf, Some(2), "a failed tryPop leaves buf alone");
  q.unset_read_only();
  assert!(q.force_push(9));
  q.close();
  assert_eq!(q.size(), 0, "close drops the elements");
  assert!(!q.try_pop(&mut buf));
  cases.ran();

  let q: AsyncPriorityQueue<i32, i32> = AsyncPriorityQueue::new();
  assert_eq!(q.try_pop(), None);
  assert!(q.force_push(1, 10));
  assert!(q.force_push(5, 50));
  assert!(q.force_push(3, 30));
  assert_eq!(q.try_pop(), Some((5, 50)));
  q.set_read_only();
  assert_eq!(q.try_pop(), Some((3, 30)));
  assert_eq!(q.try_pop(), Some((1, 10)));
  assert_eq!(q.try_pop(), None);
  q.unset_read_only();
  assert!(q.force_push(2, 20));
  q.close();
  assert_eq!(q.size(), 0);
  assert_eq!(q.try_pop(), None, "closed: no pop");
  cases.ran();
}

/// The model check of the reviewer's: random interleavings of pushes and pops against std's max-heap order.
#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn priority_queue_matches_std_priority_queue_semantics() {
  // std::priority_queue<pair<pair<i64,i64>, V>>: greatest pair first. Keys unique as in analysis.cpp.
  let mut cases = Cases::new("executor::priority_queue_matches_std_priority_queue_semantics");
  let q: AsyncPriorityQueue<(i64, i64), i32> = AsyncPriorityQueue::new();
  let mut rng = 12345u64;
  let mut model: Vec<((i64, i64), i32)> = Vec::new();
  for i in 0..5000 {
    rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    if (rng >> 60) < 10 {
      let pr = ((rng >> 20) % 7) as i64 - 3;
      let key = (pr, -(i as i64));
      assert!(q.force_push(key, i));
      model.push((key, i));
    } else {
      model.sort();
      let exp = model.pop();
      assert_eq!(q.try_pop(), exp);
      cases.ran();
    }
  }
}

/// WakerList: a condition that is already true never parks; notify_all without waiters is harmless; a waiter whose
/// future was dropped (cancelled) does not break later notifications; polling one waiter twice registers it once.
#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn waker_list_edge_cases() {
  let mut cases = Cases::new("executor::waker_list_edge_cases");
  let mut pool = LocalPool::new();
  let cv = Rc::new(WakerList::new());
  cv.notify_all();
  assert_eq!(cv.num_waiting(), 0);
  // Condition already true: ready at once, nothing registered.
  let done = Rc::new(Cell::new(false));
  {
    let (cv, done) = (cv.clone(), done.clone());
    Spawner::spawn_local(
      &pool.spawner(),
      Box::pin(async move {
        cv.wait_until(|| true).await;
        done.set(true);
      }),
    );
  }
  pool.run_until_stalled();
  assert!(done.get());
  assert_eq!(cv.num_waiting(), 0);
  cases.ran();
  // A cancelled waiter: the future is dropped while parked (select-like), then a real one is woken.
  let flag = Rc::new(Cell::new(false));
  {
    let (cv, flag) = (cv.clone(), flag.clone());
    // The select: whichever of the two resolves first ends the task; the other is dropped while parked.
    let (cv2, flag2) = (cv.clone(), flag.clone());
    Spawner::spawn_local(
      &pool.spawner(),
      Box::pin(async move {
        let a = cv.wait_until(|| flag.get());
        let b = std::future::ready(());
        futures::pin_mut!(a);
        match futures::future::select(a, b).await {
          futures::future::Either::Left(_) => {}
          futures::future::Either::Right(_) => {}
        }
        // `a` (polled once, parked, then dropped) left a waker behind; wake it spuriously and wait again properly.
        cv2.notify_all();
        cv2.wait_until(|| flag2.get()).await;
        flag2.set(false);
      }),
    );
  }
  pool.run_until_stalled();
  assert_eq!(cv.num_waiting(), 1, "the live waiter is registered");
  flag.set(true);
  cv.notify_all();
  pool.run_until_stalled();
  assert!(!flag.get(), "the waiter ran after the notification");
  assert_eq!(cv.num_waiting(), 0);
  cases.ran();
  // One task polled twice (spurious wakeups) is registered once.
  let go = Rc::new(Cell::new(false));
  let fin = Rc::new(Cell::new(false));
  {
    let (cv, go, fin) = (cv.clone(), go.clone(), fin.clone());
    Spawner::spawn_local(
      &pool.spawner(),
      Box::pin(async move {
        cv.wait_until(|| go.get()).await;
        fin.set(true);
      }),
    );
  }
  pool.run_until_stalled();
  for _ in 0..5 {
    cv.notify_all();
    pool.run_until_stalled();
    assert_eq!(cv.num_waiting(), 1);
  }
  go.set(true);
  cv.notify_all();
  pool.run_until_stalled();
  assert!(fin.get());
  assert_eq!(cv.num_waiting(), 0);
  cases.ran();
}
