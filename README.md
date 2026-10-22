# katago-rs

KataGo's analysis engine ported to Rust, compiled to WebAssembly to run in the browser
(WebGPU, with a wasm CPU fallback). The port mirrors upstream's `cpp/` file by file so that
following upstream releases stays cheap.

Status: **M0 spike done** ([docs/M0-report.md](docs/M0-report.md)); **M0.5 pilot done**: porting
guide ([docs/PORTING.md](docs/PORTING.md)), ownership table, upstream manifest, and a pilot port of
`core/hash`, `core/rand`, `game/rules`, `game/board` (`crates/katago-core`) that matches upstream
bit for bit natively and on wasm32. Open decisions: [docs/DECISIONS.md](docs/DECISIONS.md) (D4, D13).

## Layout

| Path | What |
|---|---|
| `upstream/KataGo` | upstream KataGo (submodule, pinned to a release tag) |
| `tools/bin2onnx` | KataGo `.bin.gz` → ONNX converter and its verifier against real KataGo |
| `crates/katago-core` | the port, mirroring `cpp/` (pilot: hash, rand, rules, board) |
| `crates/katago-nn-burn` | networks imported with burn-onnx; GPU conv lowering |
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
