# katago-rs

KataGo's analysis engine ported to Rust, compiled to WebAssembly to run in the browser
(WebGPU, with a wasm CPU fallback). The port mirrors upstream's `cpp/` file by file so that
following upstream releases stays cheap.

Status:
- **M0** spike ([docs/M0-report.md](docs/M0-report.md)) and **M0.5** pilot: porting guide
  ([docs/PORTING.md](docs/PORTING.md)), ownership table, upstream manifest.
- **M1** game rules (`board`, `boardhistory`, `rules`, `graphhash`), **M2** NN inputs and the
  NNEvaluator on burn, **M3** search and the analysis engine protocol: ported and checked against
  upstream oracles, natively and on wasm32. Search with the NN skipped is bit-exact with upstream;
  burn-backed searches give the same visit counts as upstream's Eigen build.
- `katago-cli`: native `katago-rs analysis` speaking upstream's analysis protocol on stdin/stdout.
- **M4** the engine in the browser ([docs/M4-report.md](docs/M4-report.md)): `katago-wasm` in a Web
  Worker, on WebGPU or the wasm CPU, and the `katago-web` npm package (`KataGoClient`); upstream's
  analysis goldens pass in Chromium on both backends.
- **M7** kata1-b18c384nbt in the browser ([docs/M7-report.md](docs/M7-report.md)): same wasm (+0.2 MB), 105 MB
  weights fetched on demand, WebGPU only (~2 s per 100 visits on an RX 6800 XT; `cpuModel` falls back to b10c128
  without WebGPU). Checked against Eigen-generated goldens in Chromium (`just web-test-b18`).
- Deferred: eval cache, pattern bonus, anti-mirror (D7); human-SL nets (M8). Decisions:
  [docs/DECISIONS.md](docs/DECISIONS.md).

## Layout

| Path | What |
|---|---|
| `upstream/KataGo` | upstream KataGo (submodule, pinned to a release tag) |
| `tools/bin2onnx` | KataGo `.bin.gz` → ONNX converter and its verifier against real KataGo |
| `crates/katago-core` | the port, mirroring upstream `cpp/` (core, game, neuralnet, search, program, command) |
| `crates/katago-nn-burn` | networks imported with burn-onnx; GPU conv lowering; the NN backend |
| `crates/katago-cli` | native `katago-rs analysis` binary (analysis protocol over stdin/stdout) |
| `crates/katago-wasm` | browser host: wasm-bindgen `Engine` running the analysis engine in a Web Worker (WebGPU / wasm CPU) |
| `packages/katago-web` | npm package: module worker, `KataGoClient`, analysis protocol types; Playwright tests (`just web-build`, `just web-test`) |
| `crates/katago-web-bench` | browser spike: parity + throughput on WebGPU and wasm CPU |
| `fixtures/` | reference data generated from upstream KataGo |

## Setup

```
git submodule update --init
python3 -m venv .venv && .venv/bin/pip install numpy onnx onnxruntime sgfmill
just katago-eigen
just convert /path/to/g170e-b10c128.bin.gz g170e-b10c128
just parity flex
```

## License

MIT. Ported code derives from [KataGo](https://github.com/lightvector/KataGo) (MIT,
© David J Wu and contributors); see `upstream/KataGo/LICENSE`. Networks are licensed
separately by their publishers.
