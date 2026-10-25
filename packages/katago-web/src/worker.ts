// The engine worker (a module Worker): loads the wasm, fetches / caches / verifies the network weights,
// picks the backend, runs the analysis engine and relays its JSON lines. Speaks messages.ts with
// KataGoClient; the analysis protocol itself passes through verbatim ("lines" in, "line" out).

import init, { Engine, compiledNetworks, kernelCacheExport, kernelCacheImport, kernelCacheStats, setHalfPrecision, sha256Hex } from "./wasm/katago_wasm.js";
import { BUILD_ID } from "./wasm/build-id.js";
import type { Backend, EngineInitOptions, EngineStats, FromWorker, ModelSpec, ReadyInfo, ToWorker } from "./messages.js";
import type { ReviewProgress } from "./review.js";

interface WorkerScope {
  postMessage(message: FromWorker): void;
  addEventListener(type: "message", listener: (e: MessageEvent<ToWorker>) => void): void;
  addEventListener(type: "error", listener: (e: ErrorEvent) => void): void;
  addEventListener(type: "unhandledrejection", listener: (e: PromiseRejectionEvent) => void): void;
}
const scope = globalThis as unknown as WorkerScope;
const post = (m: FromWorker) => scope.postMessage(m);

// Workaround (docs/M0-report.md §4.2): wgpu's WebGPU backend never requests the "subgroups" device feature,
// but cubecl's WGSL matmul always emits `enable subgroups;`, which Chrome rejects on a device without it.
// Request it whenever the adapter has it. Remove once fixed upstream (docs/upstream-issues.md).
function installSubgroupsShim(): void {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const GPUAdapterCtor = (globalThis as any).GPUAdapter;
  if (!GPUAdapterCtor) return;
  const requestDevice = GPUAdapterCtor.prototype.requestDevice;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  GPUAdapterCtor.prototype.requestDevice = function (this: any, desc: { requiredFeatures?: string[] } = {}) {
    const required = new Set(desc.requiredFeatures ?? []);
    if (this.features.has("subgroups")) required.add("subgroups");
    return requestDevice.call(this, { ...desc, requiredFeatures: [...required] });
  };
}
installSubgroupsShim();

const CACHE_NAME = "katago-web-models-v1";
const KERNEL_CACHE_NAME = "katago-web-kernels-v1";
const INTERNAL = "__kgw_";

/** Networks that are not usable on the wasm CPU (docs/M7-report.md: b18c384nbt and the b18 human-SL net run ~1 eval/s there, 100 visits
 * take about two minutes): refused on the CPU unless `allowSlowCpu` or replaced by `cpuModel`. */
const GPU_ONLY_NETWORKS = new Set(["kata1-b18c384nbt", "b18c384nbt-humanv0"]);

/** The network to run on the CPU: `cpuModel` when given, else `model` if it is practical there (or `allowSlowCpu`). */
class CpuRefused extends Error {}

function cpuSpec(options: EngineInitOptions): ModelSpec {
  if (options.cpuModel) return options.cpuModel;
  const net = options.model.network;
  if (GPU_ONLY_NETWORKS.has(net) && !options.allowSlowCpu) {
    throw new CpuRefused(
      `${net} needs WebGPU: on the CPU it runs at ~1 eval/s (100 visits take about two minutes). ` +
        "Give a smaller network as cpuModel (e.g. g170e-b10c128), or set allowSlowCpu to run it anyway.",
    );
  }
  return options.model;
}

let engine: Engine | null = null;
let ready = false;
let pendingLines: string[] = [];
/** Reviews asked for before the engine was ready (KataGoClient holds its own until `ready`; this covers other callers). */
let pendingReviews: Extract<ToWorker, { type: "review" }>[] = [];
const REVIEW_CANCELLED = "review cancelled";
/** Handler of the worker's own queries (ids starting with INTERNAL), set while one is in flight. */
let internalHandler: ((response: Record<string, unknown>) => void) | null = null;

function onEngineLine(line: string): void {
  if (line.includes(`"${INTERNAL}`)) {
    let obj: Record<string, unknown> | null = null;
    try {
      obj = JSON.parse(line) as Record<string, unknown>;
    } catch {
      obj = null;
    }
    if (obj && typeof obj.id === "string" && obj.id.startsWith(INTERNAL)) {
      internalHandler?.(obj);
      return;
    }
  }
  post({ type: "line", line });
}

function onEngineLog(kind: string, text: string): void {
  post({ type: "log", kind, text });
}

/** Sends one internal query and resolves with every response line until `isDone` says so. */
function internalQuery(query: Record<string, unknown>, isDone: (r: Record<string, unknown>) => boolean): Promise<Record<string, unknown>[]> {
  return new Promise((resolve, reject) => {
    const got: Record<string, unknown>[] = [];
    internalHandler = (r) => {
      got.push(r);
      if (typeof r.error === "string") {
        internalHandler = null;
        reject(new Error(`engine: ${r.error}${r.field ? ` (field ${String(r.field)})` : ""}`));
      } else if (isDone(r)) {
        internalHandler = null;
        resolve(got);
      }
    };
    engine!.sendLines([JSON.stringify(query)]);
  });
}

/** The WebGPU adapter ("vendor/architecture/description"), or why there is none. */
async function webgpuAdapter(): Promise<{ adapter?: string; limits?: ReadyInfo["adapterLimits"]; f16?: boolean; unavailable?: string }> {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const gpu = (navigator as any).gpu;
  if (!gpu) return { unavailable: "navigator.gpu is missing (no WebGPU in this browser or context)" };
  try {
    const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) return { unavailable: "navigator.gpu.requestAdapter() returned no adapter" };
    const info = adapter.info ?? {};
    return {
      adapter: [info.vendor, info.architecture, info.description].filter((x: string) => x).join("/") || "unknown",
      limits: { maxStorageBufferBindingSize: Number(adapter.limits?.maxStorageBufferBindingSize), maxBufferSize: Number(adapter.limits?.maxBufferSize) },
      f16: Boolean(adapter.features?.has("shader-f16")),
    };
  } catch (e) {
    return { unavailable: `navigator.gpu.requestAdapter() failed: ${String(e)}` };
  }
}

async function readBody(res: Response, total: number | undefined): Promise<Uint8Array> {
  if (!res.body) return new Uint8Array(await res.arrayBuffer());
  const reader = res.body.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  let lastPost = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    const now = performance.now();
    if (now - lastPost > 100) {
      lastPost = now;
      post({ type: "progress", phase: "model", loaded, total });
    }
  }
  const out = new Uint8Array(loaded);
  let off = 0;
  for (const c of chunks) {
    out.set(c, off);
    off += c.length;
  }
  post({ type: "progress", phase: "model", loaded, total: total ?? loaded });
  return out;
}

/** Lowercase hex SHA-256: WebCrypto when available (b10c128's 12 MB: ~30 ms, the wasm `sha256Hex` ~120 ms),
 * else the wasm one (WebCrypto needs a secure context). */
async function sha256(bytes: Uint8Array): Promise<string> {
  const subtle = globalThis.crypto?.subtle;
  if (subtle) {
    try {
      const digest = new Uint8Array(await subtle.digest("SHA-256", bytes as Uint8Array<ArrayBuffer>));
      return Array.from(digest, (b) => b.toString(16).padStart(2, "0")).join("");
    } catch {
      // fall through to the wasm implementation
    }
  }
  return sha256Hex(bytes);
}

/** The weights: from the Cache API when present and intact, else fetched, verified and cached. */
async function loadModel(options: EngineInitOptions, spec: ModelSpec): Promise<{ bytes: Uint8Array; source: "network" | "cache"; fetchMs: number; verifyMs: number }> {
  const sha = spec.sha256.toLowerCase();
  // Content-addressed key (same weights, any URL); the URL itself is only a key, never fetched.
  const key = new URL(`/__katago-web__/models/${sha}.bpk`, self.location.origin).href;
  let cache: Cache | null = null;
  if (options.cacheModels !== false && typeof caches !== "undefined") {
    try {
      cache = await caches.open(CACHE_NAME);
    } catch {
      cache = null; // e.g. opaque origins
    }
  }
  let fetchMs = 0;
  let verifyMs = 0;
  if (cache) {
    const t0 = performance.now();
    const hit = await cache.match(key);
    if (hit) {
      const bytes = new Uint8Array(await hit.arrayBuffer());
      fetchMs = performance.now() - t0;
      const t1 = performance.now();
      const got = await sha256(bytes);
      verifyMs = performance.now() - t1;
      if (got === sha) return { bytes, source: "cache", fetchMs, verifyMs };
      onEngineLog("warn", `cached ${spec.network} weights are corrupt (sha256 ${got}), fetching them again`);
      await cache.delete(key);
    }
  }
  const t0 = performance.now();
  const res = await fetch(new URL(spec.url, self.location.href).href);
  if (!res.ok) throw new Error(`fetching ${spec.url}: HTTP ${res.status}`);
  const lengthHeader = res.headers.get("content-length");
  const bytes = await readBody(res, spec.size ?? (lengthHeader ? Number(lengthHeader) : undefined));
  fetchMs += performance.now() - t0;
  const t1 = performance.now();
  const got = await sha256(bytes);
  verifyMs += performance.now() - t1;
  if (got !== sha) throw new Error(`${spec.network} weights from ${spec.url}: sha256 ${got} does not match the expected ${sha}`);
  if (cache) {
    try {
      await cache.put(key, new Response(bytes as Uint8Array<ArrayBuffer>, { headers: { "content-type": "application/octet-stream" } }));
    } catch (e) {
      onEngineLog("warn", `could not cache the ${spec.network} weights: ${String(e)}`);
    }
  }
  return { bytes, source: "network", fetchMs, verifyMs };
}

// Compiled WebGPU kernels (WGSL) persisted across page loads: cubecl compiles every kernel from its IR on each
// load (~1 s of wasm CPU for b10c128, most of the warm-up), and the browser only caches the driver's side of it.
// The blob (kernelCacheExport) is only valid for the wasm build that wrote it and is keyed by build and adapter;
// entries of other builds are deleted when a new blob is written.
const kernelCacheKey = (adapter: string) =>
  new URL(`/__katago-web__/kernels/${BUILD_ID}/${encodeURIComponent(adapter)}.bin`, self.location.origin).href;

async function openKernelCache(options: EngineInitOptions): Promise<Cache | null> {
  if (options.cacheKernels === false || typeof caches === "undefined") return null;
  try {
    return await caches.open(KERNEL_CACHE_NAME);
  } catch {
    return null;
  }
}

/** Imports the kernels a previous load of this build saved for this adapter; returns how many. */
async function loadKernels(options: EngineInitOptions, adapter: string): Promise<number> {
  const cache = await openKernelCache(options);
  if (!cache) return 0;
  try {
    const hit = await cache.match(kernelCacheKey(adapter));
    if (!hit) return 0;
    return kernelCacheImport(new Uint8Array(await hit.arrayBuffer()));
  } catch (e) {
    onEngineLog("warn", `could not load the cached WebGPU kernels: ${String(e)}`);
    return 0;
  }
}

/** Saves the kernels when some were compiled since the last save; drops the blobs of other builds. */
async function saveKernels(options: EngineInitOptions, adapter: string): Promise<void> {
  if ((kernelCacheStats() as { added: number }).added === 0) return;
  const cache = await openKernelCache(options);
  if (!cache) return;
  try {
    const key = kernelCacheKey(adapter);
    await cache.put(key, new Response(kernelCacheExport() as Uint8Array<ArrayBuffer>, { headers: { "content-type": "application/octet-stream" } }));
    const current = new URL(`/__katago-web__/kernels/${BUILD_ID}/`, self.location.origin).href;
    for (const req of await cache.keys()) if (!req.url.startsWith(current)) await cache.delete(req);
  } catch (e) {
    onEngineLog("warn", `could not cache the WebGPU kernels: ${String(e)}`);
  }
}

async function createEngine(backend: Backend, options: EngineInitOptions, network: string, bytes: Uint8Array, half: boolean): Promise<Engine> {
  setHalfPrecision(backend === "webgpu" && half);
  return Engine.create(
    {
      backend,
      network,
      modelFile: options.modelFile,
      config: options.config,
      overrides: options.overrides,
      analysisThreads: options.analysisThreads,
    },
    bytes,
    onEngineLine,
    onEngineLog,
  );
}

async function start(options: EngineInitOptions): Promise<void> {
  const tStart = performance.now();
  post({ type: "progress", phase: "wasm" });
  const wasm = await init(options.wasmUrl ? { module_or_path: new URL(options.wasmUrl, self.location.href) } : undefined);
  const wasmInitMs = performance.now() - tStart;
  const networks = compiledNetworks();
  for (const m of [options.model, options.cpuModel]) {
    if (m && !networks.includes(m.network)) throw new Error(`network ${m.network} is not compiled into this build (${networks.join(", ")})`);
  }

  // The weights, the adapter probe and the cached kernels (which need the adapter) in parallel.
  const choice = options.backend ?? "auto";
  let backend: Backend = choice === "cpu" ? "cpu" : "webgpu";
  let fallbackReason: string | undefined;
  let adapter: string | undefined;
  let adapterLimits: ReadyInfo["adapterLimits"];
  let adapterF16 = false;
  let kernelsLoaded = 0;
  let kernelCacheLoadMs = 0;
  const probing = (async () => {
    if (backend !== "webgpu") return;
    const probe = await webgpuAdapter();
    adapter = probe.adapter;
    adapterLimits = probe.limits;
    adapterF16 = probe.f16 ?? false;
    if (probe.unavailable && choice === "auto") {
      backend = "cpu";
      fallbackReason = probe.unavailable;
    }
    if (backend === "webgpu" && adapter) {
      const t0 = performance.now();
      kernelsLoaded = await loadKernels(options, adapter);
      kernelCacheLoadMs = performance.now() - t0;
    }
  })();
  // Which weights depend on the backend only when the net has a CPU replacement or is GPU-only: then (with
  // `auto`) the 105 MB of b18 must not be fetched before the probe says there is a GPU.
  const backendMatters = options.cpuModel !== undefined || GPU_ONLY_NETWORKS.has(options.model.network);
  let spec = options.model;
  let loading: Promise<Awaited<ReturnType<typeof loadModel>>>;
  if (choice === "cpu") {
    spec = cpuSpec(options);
    loading = loadModel(options, spec);
  } else if (choice === "auto" && backendMatters) {
    await probing;
    if (backend === "cpu") spec = cpuSpec(options);
    loading = loadModel(options, spec);
  } else {
    loading = loadModel(options, spec);
  }
  const [model] = await Promise.all([loading, probing]);
  post({ type: "progress", phase: "engine" });
  const tEngine = performance.now();
  let half = (options.precision ?? "auto") !== "f32" && adapterF16;
  try {
    try {
      engine = await createEngine(backend, options, spec.network, model.bytes, half);
    } catch (e) {
      // An adapter that advertises shader-f16 but fails with it (driver bugs): WebGPU in f32 before the CPU.
      if (!half || backend !== "webgpu" || options.precision === "f16") throw e;
      onEngineLog("warn", `WebGPU engine failed in f16, retrying in f32: ${String((e as Error)?.message ?? e)}`);
      half = false;
      engine = await createEngine(backend, options, spec.network, model.bytes, false);
    }
  } catch (e) {
    if (choice !== "auto" || backend === "cpu") throw e;
    fallbackReason = `WebGPU engine failed: ${String((e as Error)?.message ?? e)}`;
    backend = "cpu";
    let cpu: ModelSpec;
    try {
      cpu = cpuSpec(options);
    } catch (err) {
      throw new CpuRefused(`${String((err as Error).message)} (${fallbackReason})`);
    }
    const cpuModel = cpu === spec ? model : await loadModel(options, cpu);
    spec = cpu;
    model.bytes = cpuModel.bytes;
    model.source = cpuModel.source;
    engine = await createEngine(backend, options, spec.network, model.bytes, false);
  }
  const configText = engine.configText;
  engine.finished().then(
    (code: number) => post({ type: "exit", code }),
    (e: unknown) => post({ type: "fatal", message: `engine exited: ${String((e as Error)?.message ?? e)}` }),
  );
  // The first response arrives once the evaluator is up (model built, weights on the device).
  const [version] = await internalQuery({ id: `${INTERNAL}version`, action: "query_version" }, () => true);
  const engineStartMs = performance.now() - tEngine;
  const modelBuildMs = (engine.stats() as EngineStats).modelBuildMs;

  // One search on WebGPU: before its first batch the engine runs one batch of every batch-shape bucket
  // (katago-wasm StatsNeuralNet), so no kernel compiles while a real search waits on it.
  const warmupVisits = options.warmupVisits ?? (backend === "webgpu" ? 1 : 0);
  let warmupMs = 0;
  if (warmupVisits > 0) {
    post({ type: "progress", phase: "warmup" });
    const t0 = performance.now();
    await internalQuery(
      { id: `${INTERNAL}warmup`, moves: [], rules: "tromp-taylor", komi: 7.5, boardXSize: 19, boardYSize: 19, maxVisits: warmupVisits },
      (r) => r.isDuringSearch === false,
    );
    warmupMs = performance.now() - t0;
  }

  const info: ReadyInfo = {
    backend,
    adapter: backend === "webgpu" ? adapter : undefined,
    adapterLimits: backend === "webgpu" ? adapterLimits : undefined,
    precision: backend === "webgpu" ? (half ? "f16" : "f32") : undefined,
    network: spec.network,
    requestedNetwork: spec.network !== options.model.network ? options.model.network : undefined,
    version: String(version.version),
    gitHash: String(version.git_hash),
    modelSource: model.source,
    // The wasm heap never shrinks: this is the peak so far (the weights pass through it once on their way to the device).
    wasmMemoryBytes: wasm.memory.buffer.byteLength,
    fallbackReason,
    configText,
    timings: {
      wasmInitMs,
      modelFetchMs: model.fetchMs,
      modelVerifyMs: model.verifyMs,
      engineStartMs,
      modelBuildMs,
      warmupMs,
      bucketWarmupMs: (engine.stats() as EngineStats).warmupMs,
      kernelCacheLoadMs,
      totalMs: performance.now() - tStart,
    },
  };
  if (backend === "webgpu" && adapter) {
    const k = kernelCacheStats() as { hits: number; misses: number };
    info.kernelCache = { loaded: kernelsLoaded, hits: k.hits, compiled: k.misses };
    // Saved before `ready` (a few ms, only when kernels were compiled), so a page closed right after still
    // leaves them for the next load; kernels compiled later (rarely: a shape the warm-up did not cover) are
    // saved periodically.
    const key = adapter;
    const t0 = performance.now();
    await saveKernels(options, key);
    info.timings.kernelCacheSaveMs = performance.now() - t0;
    setInterval(() => void saveKernels(options, key), 30_000);
  }
  info.timings.totalMs = performance.now() - tStart;
  ready = true;
  post({ type: "ready", info });
  if (pendingLines.length > 0) {
    const lines = pendingLines;
    pendingLines = [];
    engine.sendLines(lines);
  }
  const reviews = pendingReviews;
  pendingReviews = [];
  for (const r of reviews) startReview(r);
}

/** A full-game review (katago-ws's flow and payload, in the wasm: Engine.reviewGame). */
function startReview(m: Extract<ToWorker, { type: "review" }>): void {
  const id = String(m.requestId);
  let promise: Promise<string>;
  try {
    promise = engine!.reviewGame(id, m.game, m.options, (progress: unknown) =>
      post({ type: "reviewProgress", requestId: m.requestId, progress: progress as ReviewProgress }),
    );
  } catch (e) {
    post({ type: "reviewError", requestId: m.requestId, message: String((e as Error)?.message ?? e), cancelled: false });
    return;
  }
  promise.then(
    (payload) => post({ type: "reviewDone", requestId: m.requestId, payload }),
    (e: unknown) => {
      const message = String((e as Error)?.message ?? e);
      post({ type: "reviewError", requestId: m.requestId, message, cancelled: message === REVIEW_CANCELLED });
    },
  );
}

const zeroStats: EngineStats = { batches: 0, rows: 0, busyMs: 0, lastBatchMs: 0, maxBatchMs: 0, lastBatchRows: 0, maxBatchRows: 0, modelBuildMs: 0, warmupMs: 0, warmupBatches: 0, batchesSinceLastInput: 0, rowsSinceLastInput: 0 };

scope.addEventListener("message", (e: MessageEvent<ToWorker>) => {
  const m = e.data;
  switch (m.type) {
    case "init":
      start(m.options).catch((err: unknown) => post({ type: "fatal", message: String((err as Error)?.message ?? err), retry: !(err instanceof CpuRefused) }));
      break;
    case "lines":
      if (ready && engine) engine.sendLines(m.lines);
      else pendingLines.push(...m.lines);
      break;
    case "stats":
      post({ type: "stats", requestId: m.requestId, stats: engine ? (engine.stats() as EngineStats) : zeroStats });
      break;
    case "review":
      if (ready && engine) startReview(m);
      else pendingReviews.push(m);
      break;
    case "reviewCancel": {
      const queued = pendingReviews.findIndex((r) => r.requestId === m.requestId);
      if (queued >= 0) {
        pendingReviews.splice(queued, 1);
        post({ type: "reviewError", requestId: m.requestId, message: REVIEW_CANCELLED, cancelled: true });
      } else engine?.cancelReview(String(m.requestId));
      break;
    }
    case "close":
      engine?.closeInput();
      break;
  }
});

// A panic in the wasm (console_error_panic_hook logs it) traps; the instance is unusable afterwards.
scope.addEventListener("error", (e: ErrorEvent) => {
  e.preventDefault(); // reported here; not again through the Worker object's error event
  post({ type: "fatal", message: String(e.message ?? e) });
});
scope.addEventListener("unhandledrejection", (e: PromiseRejectionEvent) => post({ type: "fatal", message: String((e.reason as Error)?.message ?? e.reason) }));
