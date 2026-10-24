#!/usr/bin/env bash
# Re-records fixtures/katago-annotate through katago-ws's own KataGo client (reference KataGo, Eigen; ~5 min with
# 8 analysis threads), checks katago-ws's annotator and katago-annotate agree byte for byte on every case, then
# writes the fixtures (responses and payloads gzipped). Needs the katago-ws checkout next to this repository
# (../katago-ws: its src/ is compiled unchanged, its bench/ has the SGFs and nets) and `just katago-eigen`.
#   tools/annotate-parity/regen.sh [case...]      (default: every cases/*.json)
#   tools/annotate-parity/regen.sh --check-only   (compare the checked-in fixtures, no engine run)
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
target="$repo/target/annotate-parity"
cargo build --release --manifest-path "$here/Cargo.toml" --target-dir "$target/build"
bin="$target/build/release"
if [ "${1:-}" = "--check-only" ]; then
    work="$target/check"; rm -rf "$work"; mkdir -p "$work"
    for d in "$repo"/fixtures/katago-annotate/*/; do
        name="$(basename "$d")"; mkdir -p "$work/$name"
        cp "$d"/game.sgf "$d"/case.json "$d"/queries.jsonl "$work/$name/"
        gunzip -c "$d/responses.jsonl.gz" > "$work/$name/responses.jsonl"
        gunzip -c "$d/expected.json.gz" > "$work/$name/expected.json"
    done
    exec "$bin/compare" "$work"/*/
fi
cases=("$@")
[ ${#cases[@]} -gt 0 ] || cases=($(cd "$here/cases" && ls *.json | sed 's/\.json$//'))
out="$target/out"
for name in "${cases[@]}"; do
    rm -rf "${out:?}/$name"
    "$bin/record" "$here/cases/$name.json" "$out/$name"
done
"$bin/compare" "${cases[@]/#/$out/}"
for name in "${cases[@]}"; do
    f="$repo/fixtures/katago-annotate/$name"; mkdir -p "$f"
    cp "$out/$name"/game.sgf "$out/$name"/case.json "$out/$name"/queries.jsonl "$f/"
    gzip -9 -n -c "$out/$name/responses.jsonl" > "$f/responses.jsonl.gz"
    gzip -9 -n -c "$out/$name/expected.json" > "$f/expected.json.gz"
done
du -sh "$repo/fixtures/katago-annotate"
