#!/usr/bin/env bash
# The native rows of README "Benchmarks", one net after another (`just bench-all [nets]`; default all five):
#   evals/s     katago-rs forward passes (`just bench`: wgpu and rocm, f32 and f16) and upstream's (`just
#               bench-upstream-nn`: Mesa's rusticl and AMD's OpenCL), batches 1, 8, 16, 32, 64;
#   positions/s `just bench-analysis` with katago-rs on wgpu and rocm (FP16) and upstream on both OpenCL drivers.
# Logs to target/bench/<net>.log. STEPS (default forward,upstream-nn,analysis) picks the parts.
#
# Needs upstream's OpenCL build (`just katago-opencl`), ROCm (the .envrc's CUBECL_ROCM_DEVICE_LIB_PATH) and the
# nets' .bin.gz in NETS (default models/nets) under their release names (below). katago-rs's analysis runs use two
# CLI builds with every net, as katago-ws's images build them: wgpu (with burn's autotune) in target/release, rocm
# in target/bench-rocm.
set -euo pipefail
cd "$(dirname "$0")/.."

nets=("$@")
[ ${#nets[@]} -gt 0 ] || nets=(b10 b18 b10c384tf b10c512tf tf3)
steps=${STEPS:-forward,upstream-nn,analysis}
dir=${NETS:-models/nets}
mkdir -p target/bench

file() {
  case $1 in
    b10) echo "$dir/g170e-b10c128.bin.gz" ;;
    b18) echo "$dir/kata1-b18c384nbt.bin.gz" ;;
    b10c384tf) echo "$dir/b10c384h6nbttflrs.bin.gz" ;;
    b10c512tf) echo "$dir/b10c512h8nbt3tflrs-fson-silu-rsnh.bin.gz" ;;
    tf3) echo "$dir/kata1-tf3-b11c768-s11003M-d5973M-7gres.bin.gz" ;;
    *) echo "bench-all: unknown net $1 (b10 | b18 | b10c384tf | b10c512tf | tf3)" >&2; exit 2 ;;
  esac
}

if [[ $steps == *analysis* ]]; then
  features="net-b10c128,net-b18c384nbt,net-b10c384h6nbttflrs,net-b10c512h8nbt3tflrs-fson-silu-rsnh,net-tf3-b11c768"
  cargo build -q --release -p katago-cli --no-default-features -F "flex,wgpu,$features"
  CARGO_TARGET_DIR=target/bench-rocm cargo build -q --release -p katago-cli --no-default-features -F "flex,rocm,$features"
fi

for net in "${nets[@]}"; do
  bin=$(file "$net")
  [ -f "$bin" ] || { echo "bench-all: $bin missing" >&2; exit 1; }
  log=target/bench/$net.log
  : > "$log"
  say() { echo "== $net: $*" | tee -a "$log"; }
  if [[ $steps == *forward* ]]; then
    for backend in wgpu rocm; do
      for precision in f32 f16; do
        say "katago-rs forward passes, $backend $precision"
        just bench "$backend" "$net" "$precision" 2>&1 | grep -E "batch|evals" | tee -a "$log"
      done
    done
  fi
  if [[ $steps == *upstream-nn* ]]; then
    for icd in libRusticlOpenCL.so.1 libamdocl64.so; do
      say "upstream forward passes, $icd"
      just bench-upstream-nn "$bin" "$icd" | tee -a "$log"
    done
  fi
  if [[ $steps == *analysis* ]]; then
    # NN server threads x batch 64, analysis threads: katago-ws's settings (README "Benchmarks").
    up_threads=128
    [ "$net" = b10 ] && up_threads=256
    say "katago-rs positions/s, wgpu f16, 4 x 64, 256"
    KATAGO_RS=target/release/katago-rs just bench-analysis wgpu "$bin" 4 256 2>&1 | grep -E "analysis_bench|positions/s" | tee -a "$log"
    say "katago-rs positions/s, rocm f16, 8 x 64, 256"
    KATAGO_RS=target/bench-rocm/release/katago-rs just bench-analysis rocm "$bin" 8 256 2>&1 | grep -E "analysis_bench|positions/s" | tee -a "$log"
    for engine in rusticl amdocl; do
      say "upstream positions/s, $engine, 4 x 64, $up_threads"
      just bench-analysis "$engine" "$bin" 4 "$up_threads" 2>&1 | grep -E "analysis_bench|positions/s" | tee -a "$log"
    done
  fi
done
