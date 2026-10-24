// katago-web: KataGo's analysis engine (ported to Rust, compiled to wasm) in a Web Worker, on WebGPU or
// the wasm CPU. See docs/M4-report.md in the katago-rs repository.

export { AnalysisRun, KataGoClient, KataGoError } from "./client.js";
export type { KataGoClientOptions } from "./client.js";
export type { Backend, BackendChoice, EngineInitOptions, EngineStats, ModelSpec, ReadyInfo } from "./messages.js";
export type * from "./protocol.js";

/** `models.json` next to the built package (`just web-build`): the weights of the compiled-in networks. */
export interface ModelManifest {
  models: { network: string; file: string; sha256: string; size: number }[];
}

/** Reads a `models.json` and returns the ModelSpec of `network`, with its URL resolved against the manifest's. */
export async function fetchModelSpec(manifestUrl: string | URL, network: string): Promise<import("./messages.js").ModelSpec> {
  const url = new URL(manifestUrl, globalThis.location?.href);
  const res = await fetch(url);
  if (!res.ok) throw new Error(`fetching ${url}: HTTP ${res.status}`);
  const manifest = (await res.json()) as ModelManifest;
  const m = manifest.models.find((x) => x.network === network);
  if (!m) throw new Error(`${url} has no network ${network} (has: ${manifest.models.map((x) => x.network).join(", ")})`);
  return { network: m.network, url: new URL(m.file, url).href, sha256: m.sha256, size: m.size };
}
