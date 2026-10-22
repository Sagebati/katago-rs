//! wasm32 host glue for the backend (no upstream counterpart).

use wasm_bindgen::JsCast;
use wasm_bindgen::JsValue;
use wasm_bindgen::prelude::wasm_bindgen;

#[wasm_bindgen]
extern "C" {
  /// The global `setTimeout` (window, worker and node all have it).
  #[wasm_bindgen(js_name = setTimeout)]
  fn set_timeout(handler: &js_sys::Function, timeout: i32) -> JsValue;
}

/// Yields to the host's event loop: resolves from a `setTimeout(_, 0)` macrotask, so every task the
/// host already queued (timers, worker `message` events, input) runs first. PORTING.md §11: a CPU
/// backend must do this in its `get_output`, since a microtask yield (`port::executor::yield_now`
/// under `wasm_bindgen_futures`) never lets the event loop turn.
/// PERF(port): browsers clamp nested `setTimeout(0)` to 4 ms after 5 levels; per batch that is
/// small next to a CPU forward pass. A `MessageChannel` yield avoids the clamp but its ordering
/// against timers is not specified.
pub async fn yield_to_event_loop() {
  let promise = js_sys::Promise::new(&mut |resolve: js_sys::Function, _reject: js_sys::Function| {
    set_timeout(resolve.unchecked_ref(), 0);
  });
  wasm_bindgen_futures::JsFuture::from(promise).await.expect("a setTimeout-resolved Promise does not reject");
}
