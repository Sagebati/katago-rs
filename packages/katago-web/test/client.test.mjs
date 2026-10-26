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

function setup(backend = "auto", extra = {}) {
  const workers = [];
  const client = new KataGoClient({
    model: { network: "g170e-b10c128", url: "http://x/model.bpk", sha256: "00" },
    ...extra,
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

const READY = { backend: "cpu", network: "g170e-b10c128", version: "1.18.2", gitHash: "x", modelSource: "cache", configText: "", timings: {} };

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

test("reviewGame: queued until ready (on the worker that gets ready), progress, payload", async () => {
  const { client, workers } = setup();
  const progress = [];
  const run = client.reviewGame("(;SZ[9];B[ee])", { maxVisits: 10 }, (p) => progress.push(p));
  assert.ok(!workers[0].sent.some((m) => m.type === "review"), "nothing posted before ready");
  workers[0].emit({ type: "fatal", message: "simulated WebGPU failure" });
  workers[1].emit({ type: "ready", info: READY });
  await client.ready;
  const posted = workers[1].sent.find((m) => m.type === "review");
  assert.deepEqual({ ...posted, requestId: 0 }, { type: "review", requestId: 0, game: "(;SZ[9];B[ee])", options: { maxVisits: 10 } });
  const p = { pass: "full", passes: 1, turnNumber: 0, done: 1, total: 2, analysed: 1 };
  workers[1].emit({ type: "reviewProgress", requestId: posted.requestId, progress: p });
  workers[1].emit({ type: "reviewDone", requestId: posted.requestId, payload: '{"payload_version":3,"moves":[]}' });
  assert.deepEqual(await run.result, { payload_version: 3, moves: [] });
  assert.equal(await run.json, '{"payload_version":3,"moves":[]}');
  assert.deepEqual(progress, [p]);
  client.dispose();
});

test("reviewGame: cancel before and after it is posted; errors", async () => {
  const { client, workers } = setup();
  const early = client.reviewGame("(;)");
  early.cancel();
  const e1 = await early.result.then(() => null, (e) => e);
  assert.ok(e1?.cancelled, "cancelled before ready");
  workers[0].emit({ type: "ready", info: READY });
  await client.ready;
  assert.ok(!workers[0].sent.some((m) => m.type === "review"), "the cancelled review is never posted");

  const run = client.reviewGame("(;)");
  const { requestId } = workers[0].sent.find((m) => m.type === "review");
  run.cancel();
  assert.deepEqual(workers[0].sent.at(-1), { type: "reviewCancel", requestId });
  workers[0].emit({ type: "reviewError", requestId, message: "review cancelled", cancelled: true });
  const e2 = await run.result.then(() => null, (e) => e);
  assert.equal(e2?.name, "KataGoError");
  assert.ok(e2.cancelled);

  const bad = client.reviewGame("not sgf");
  const id3 = workers[0].sent.at(-1).requestId;
  workers[0].emit({ type: "reviewError", requestId: id3, message: "invalid SGF: ...", cancelled: false });
  const e3 = await bad.result.then(() => null, (e) => e);
  assert.ok(!e3.cancelled);
  assert.match(e3.message, /invalid SGF/);

  const ac = new AbortController();
  const aborted = client.reviewGame("(;)", { signal: ac.signal });
  const id4 = workers[0].sent.at(-1).requestId;
  assert.ok(!("signal" in workers[0].sent.at(-1).options), "the signal stays on the main thread");
  ac.abort();
  assert.deepEqual(workers[0].sent.at(-1), { type: "reviewCancel", requestId: id4 });

  const pending = client.reviewGame("(;)");
  workers[0].emit({ type: "fatal", message: "device lost" });
  await assert.rejects(pending.result, /device lost/);
  void aborted.result.catch(() => {});
});

test("cpuModel and allowSlowCpu reach the worker, on the first start and on the CPU restart", () => {
  const cpuModel = { network: "g170e-b10c128", url: "http://x/b10.bpk", sha256: "11" };
  const { client, workers } = setup("auto", { model: { network: "kata1-b18c384nbt", url: "http://x/b18.bpk", sha256: "22" }, cpuModel, allowSlowCpu: true });
  const init = (w) => w.sent.find((m) => m.type === "init").options;
  assert.equal(init(workers[0]).model.network, "kata1-b18c384nbt");
  assert.deepEqual(init(workers[0]).cpuModel, cpuModel);
  assert.equal(init(workers[0]).allowSlowCpu, true);
  workers[0].emit({ type: "fatal", message: "simulated WebGPU failure" });
  assert.equal(workers.length, 2);
  assert.deepEqual(init(workers[1]).cpuModel, cpuModel);
  client.dispose();
});

test("a refusal (retry: false) is fatal at once, with no CPU restart", async () => {
  const { client, workers } = setup();
  workers[0].emit({ type: "fatal", message: "kata1-b18c384nbt needs WebGPU", retry: false });
  assert.equal(workers.length, 1);
  await assert.rejects(client.ready, /needs WebGPU/);
});

test("a failure of the CPU restart names the WebGPU failure", async () => {
  const { client, workers } = setup();
  workers[0].emit({ type: "fatal", message: "kernel did not compile" });
  workers[1].emit({ type: "fatal", message: "kata1-b18c384nbt needs WebGPU", retry: false });
  await assert.rejects(client.ready, (e) => /needs WebGPU/.test(e.message) && /kernel did not compile/.test(e.message));
});
