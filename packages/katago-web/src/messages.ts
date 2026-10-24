// Messages between KataGoClient (main thread) and the engine worker (worker.ts).

import type { GameInput, ReviewOptions, ReviewProgress } from "./review.js";

export type BackendChoice = "auto" | "webgpu" | "cpu";
export type Backend = "webgpu" | "cpu";

/** The `.bpk` weights of a network compiled into the wasm (see `models.json` written by `just web-build`). */
export interface ModelSpec {
  /** "g170e-b10c128" | "g170-b6c96" | "kata1-b18c384nbt" | "b18c384nbt-humanv0" (must be compiled in; the human-SL net needs humanSLProfile). */
  network: string;
  /** Where to fetch the `.bpk` (resolved against the page URL). */
  url: string;
  /** Lowercase hex SHA-256 of the `.bpk`; the bytes are checked against it, and it keys the cache. */
  sha256: string;
  /** Size in bytes (progress reporting only). */
  size?: number;
}

export interface EngineInitOptions {
  model: ModelSpec;
  backend?: BackendChoice;
  /** A full config text replacing upstream's analysis_example.cfg (the browser defaults are then not applied). */
  config?: string;
  /** `-override-config` strings ("key=value, key=value"), applied in order after the browser defaults. */
  overrides?: string[];
  /** As `-analysis-threads`. */
  analysisThreads?: number;
  /** The `-model` name the engine reports (default: the network name). */
  modelFile?: string;
  /** Visits of a warm-up search on an empty 19x19 board before `ready`. On WebGPU its first NN batch runs
   * one batch per batch-shape bucket first (compiling and autotuning every kernel the search will use).
   * Default: 1 on WebGPU, 0 on the CPU; 0 skips it (the warm-up then happens in the first query). */
  warmupVisits?: number;
  /** Cache the weights with the Cache API (default true). */
  cacheModels?: boolean;
  /** Cache the compiled WebGPU kernels with the Cache API, per build and adapter (default true): later page
   * loads skip compiling them, most of the WebGPU warm-up. */
  cacheKernels?: boolean;
  /** URL of `katago_wasm_bg.wasm` (default: next to the worker's wasm-bindgen glue). */
  wasmUrl?: string;
}

export interface ReadyInfo {
  backend: Backend;
  /** The WebGPU adapter (`GPUAdapterInfo` vendor/architecture/description) when the backend is webgpu. */
  adapter?: string;
  network: string;
  /** KataGo version the port follows and its git hash (query_version). */
  version: string;
  gitHash: string;
  /** Where the weights came from. */
  modelSource: "network" | "cache";
  /** Why WebGPU was not used, when `backend: "auto"` fell back to the CPU. */
  fallbackReason?: string;
  /** The effective config. */
  configText: string;
  /** WebGPU: kernels imported from the kernel cache, and the kernel lookups up to `ready` that found one (hits) or
   * had to compile it (compiled). */
  kernelCache?: { loaded: number; hits: number; compiled: number };
  timings: {
    /** wasm fetch + compile + instantiate */
    wasmInitMs: number;
    /** weights: network or cache read */
    modelFetchMs: number;
    /** SHA-256 of the weights */
    modelVerifyMs: number;
    /** device init + engine start + model upload, up to the first response */
    engineStartMs: number;
    /** the burn module built from the weights (inside engineStartMs) */
    modelBuildMs: number;
    /** the warm-up search */
    warmupMs: number;
    /** WebGPU: one batch per batch-shape bucket (kernel compiles, or loads from the kernel cache), inside warmupMs */
    bucketWarmupMs: number;
    /** WebGPU: reading the cached kernels (in parallel with the weights) */
    kernelCacheLoadMs: number;
    /** WebGPU: saving the kernels compiled by this load (0 when all came from the cache) */
    kernelCacheSaveMs?: number;
    totalMs: number;
  };
}

export interface EngineStats {
  batches: number;
  rows: number;
  busyMs: number;
  lastBatchMs: number;
  maxBatchMs: number;
  lastBatchRows: number;
  maxBatchRows: number;
  modelBuildMs: number;
  /** The batch-shape warm-up (WebGPU), before the first batch; not counted in batches / rows / busyMs. */
  warmupMs: number;
  warmupBatches: number;
  /** NN batches completed since the engine last received input lines (e.g. a terminate). */
  batchesSinceLastInput: number;
  /** NN evaluations (batch rows) completed since the engine last received input lines. */
  rowsSinceLastInput: number;
}

export type ToWorker =
  | { type: "init"; options: EngineInitOptions }
  | { type: "lines"; lines: string[] }
  | { type: "stats"; requestId: number }
  | { type: "review"; requestId: number; game: string | GameInput; options: Omit<ReviewOptions, "signal"> }
  | { type: "reviewCancel"; requestId: number }
  | { type: "close" };

export type FromWorker =
  | { type: "progress"; phase: "wasm" | "model" | "engine" | "warmup"; loaded?: number; total?: number }
  | { type: "ready"; info: ReadyInfo }
  | { type: "line"; line: string }
  | { type: "log"; kind: string; text: string }
  | { type: "stats"; requestId: number; stats: EngineStats }
  | { type: "reviewProgress"; requestId: number; progress: ReviewProgress }
  /** `payload`: the GameAnalysis JSON text. */
  | { type: "reviewDone"; requestId: number; payload: string }
  | { type: "reviewError"; requestId: number; message: string; cancelled: boolean }
  | { type: "exit"; code: number }
  | { type: "fatal"; message: string };
