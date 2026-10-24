// Unit tests of KataGoClient against a fake worker: no wasm, no browser (`just web-test-client`, after
// `just web-build`). The engine side is scripted by posting FromWorker messages from the fake.

import { test } from "node:test";
import assert from "node:assert/strict";
import { KataGoClient } from "../dist/client.js";

class FakeWorker {
  constructor(index) {
    this.index = index;
    this.sent = [];
    this.terminated = false;
    this.onmessage = null;
    this.onerror = null;
  }
  postMessage(m) {
    this.sent.push(m);
  }
  terminate() {
    this.terminated = true;
  }
  /** A message from the engine. */
  emit(data) {
    this.onmessage?.({ data });
  }
  get backend() {
    return this.sent.find((m) => m.type === "init")?.options.backend;
  }
}

function setup(backend = "auto") {
  const workers = [];
  const client = new KataGoClient({
    model: { network: "g170e-b10c128", url: "http://x/model.bpk", sha256: "00" },
    backend,
    worker: () => {
      const w = new FakeWorker(workers.length);
      workers.push(w);
      return w;
    },
    onLog: () => {},
  });
  return { client, workers };
}

const READY = { backend: "cpu", network: "g170e-b10c128", version: "1.16.5", gitHash: "x", modelSource: "cache", configText: "", timings: {} };

test("a WebGPU failure before ready restarts on the CPU", async () => {
  const { client, workers } = setup();
  assert.equal(workers.length, 1);
  assert.equal(workers[0].backend, "auto");
  workers[0].emit({ type: "fatal", message: "simulated WebGPU failure" });
  assert.equal(workers.length, 2);
  assert.ok(workers[0].terminated);
  assert.equal(workers[1].backend, "cpu");
  workers[1].emit({ type: "ready", info: READY });
  const info = await client.ready;
  assert.equal(info.backend, "cpu");
  assert.match(info.fallbackReason, /simulated WebGPU failure/);
  client.dispose();
  assert.ok(workers[1].terminated);
});

test("dispose before ready creates no new worker", async () => {
  const { client, workers } = setup();
  client.dispose();
  assert.equal(workers.length, 1, "no CPU worker started by the dispose");
  assert.ok(workers[0].terminated);
  await assert.rejects(client.ready, /disposed/);
  // A late failure report of the terminated worker does not start one either.
  workers[0].emit({ type: "fatal", message: "late" });
  workers[0].onerror?.({ message: "late", preventDefault() {} });
  assert.equal(workers.length, 1);
});

test("dispose after fallback terminates the CPU worker", async () => {
  const { client, workers } = setup();
  workers[0].emit({ type: "fatal", message: "simulated WebGPU failure" });
  assert.equal(workers.length, 2);
  // The CPU worker is still starting up.
  const run = client.analyze({ id: "q", moves: [], rules: "japanese", komi: 6.5, boardXSize: 19, boardYSize: 19 });
  client.dispose();
  assert.ok(workers[1].terminated, "the CPU worker is terminated");
  assert.equal(workers.length, 2, "and no third one is started");
  await assert.rejects(client.ready, /disposed/);
  await assert.rejects(run.done, /disposed/);
  // The queued query never reached the terminated worker.
  assert.ok(!workers[1].sent.some((m) => m.type === "lines"));
});

test("dispose after ready fails pending queries and starts nothing", async () => {
  const { client, workers } = setup();
  workers[0].emit({ type: "ready", info: { ...READY, backend: "webgpu" } });
  await client.ready;
  const run = client.analyze({ id: "q", moves: [], rules: "japanese", komi: 6.5, boardXSize: 19, boardYSize: 19 });
  client.dispose();
  assert.ok(workers[0].terminated);
  assert.equal(workers.length, 1);
  await assert.rejects(run.done, /disposed/);
});

test("an engine failure after ready is fatal, with no restart", async () => {
  const { client, workers } = setup();
  workers[0].emit({ type: "ready", info: { ...READY, backend: "webgpu" } });
  await client.ready;
  const run = client.analyze({ id: "q", moves: [], rules: "japanese", komi: 6.5, boardXSize: 19, boardYSize: 19 });
  workers[0].emit({ type: "fatal", message: "device lost" });
  assert.equal(workers.length, 1);
  const err = await run.done.then(() => null, (e) => e);
  assert.equal(err?.name, "KataGoError");
  assert.equal(err.response, undefined, "an engine failure carries no error response");
  assert.match(err.message, /device lost/);
});
