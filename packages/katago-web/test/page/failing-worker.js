// Test double for the CPU restart of KataGoClient: a worker whose WebGPU engine "fails" before it is
// ready (as a kernel that does not compile traps the wasm), and that runs the real worker on the CPU.
let started = false;
self.addEventListener("message", async (e) => {
  if (e.data?.type !== "init" || started) return;
  if (e.data.options.backend !== "cpu") {
    self.postMessage({ type: "fatal", message: "simulated WebGPU failure (RuntimeError: unreachable)" });
    return;
  }
  started = true;
  await import("/pkg/worker.js");
  self.dispatchEvent(new MessageEvent("message", { data: e.data }));
});
