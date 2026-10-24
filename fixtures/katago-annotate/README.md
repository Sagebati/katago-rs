# katago-annotate parity fixtures (M6)

Games run through **katago-ws's own KataGo client** (`src/engine/katago.rs` @ `bdd674bb`, compiled unchanged by
`tools/annotate-parity`) against the reference KataGo v1.16.5 (Eigen, `just katago-eigen`) with
`g170e-b10c128.bin.gz`, upstream's `analysis_example.cfg`, 8 analysis threads x 1 search thread. Per case:

| File | What |
|---|---|
| `game.sgf` | the input |
| `case.json` | the options (katago-annotate's `AnnotateOptions`, camelCase) and the human profile asked for |
| `queries.jsonl` | the query lines katago-ws wrote to KataGo's stdin, verbatim (ids `q0`, `q1`) |
| `responses.jsonl.gz` | KataGo's stdout, verbatim (gzip) |
| `expected.json.gz` | katago-ws's `annotate::assemble` payload, `serde_json::to_string` (gzip) |

| Case | Game | Options | Turns |
|---|---|---|---|
| `ogs-60000000` | 19x19, Chinese, komi 7.5, 132 moves | katago-ws defaults (100 visits, one pass) | 133 |
| `ogs-60046344-twopass` | 19x19, Japanese, komi 6.5, 138 moves | screen 16 visits, refine at 100 | 139 + 89 |
| `ogs-60023172-100` | 19x19, Chinese, 2 handicap stones (AB), komi 0.5; the first 100 moves of the game | defaults | 101 |
| `hand-9x9` | 9x9, Japanese, komi 7, 23 moves ending in two passes | 50 visits, top 5, PV 5, no stdev / policy, `rank_5k` without a human model (dropped) | 24 |
| `hand-13x13-h2` | 13x13, Chinese, 2 handicap stones, komi 0.5, 20 moves | 60 visits, screen 10, refine win-rate delta 0.05, no ownership | 21 + 15 |

`ogs-*` are from katago-ws's `bench/sgf-quick`. Regenerate (new engine output, so new payloads) with
`just annotate-parity-regen`; `just annotate-parity-regen --check-only` re-checks katago-ws's annotator against
these files, `just annotate-parity` katago-annotate (native and wasm32).
