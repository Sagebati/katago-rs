//! Shared by the tests and the `review` example (`#[path]`): an executor for futures that never wait.

use std::future::Future;
use std::pin::pin;
use std::task::{Context, Poll, Waker};

/// Polls `fut` to completion with a no-op waker: for futures that are always ready or only wait on the
/// thread itself (a synchronous engine, a replay).
pub fn block_on<F: Future>(fut: F) -> F::Output {
    let mut fut = pin!(fut);
    let mut cx = Context::from_waker(Waker::noop());
    loop {
        if let Poll::Ready(out) = fut.as_mut().poll(&mut cx) {
            return out;
        }
    }
}
