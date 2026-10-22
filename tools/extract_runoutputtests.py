#!/usr/bin/env python3
"""Extract expected outputs from upstream's cpp/tests/results/runOutputTests.txt (the cout of
`katago runoutputtests`) into fixtures/<tag>/unit/<stem>/<slug>.expected<k>.txt plus an INDEX.tsv
(same layout as tools/extract_raw_blocks.py, so tests/fixture_coverage.rs checks that every
chunk is consumed), all in the stem directory `runoutputtests/`:
  - nn-inputs-*.expected<k>.txt, passing-hack-*.expected0.txt: the input-version-7 parts of
    Tests::runNNInputsV3V4Tests (testnninputs.cpp; see below);
  - score-and-utility-tests.expected0.txt: Tests::runScoreTests (testscore.cpp), from its
    "Running score and utility tests" line to the last line before the output of
    Tests::runNNSymmetryTests;
  - nn-symmetry-tests.expected0.txt: Tests::runNNSymmetryTests (testnn.cpp), from its first
    "Symmetry 3-1-3-3" line to the last line before "SgfFileTests ok".

runOutputTests.txt is the cout of `katago runoutputtests`; its first part (up to the line
"Running neuralnetless search tests") comes from cpp/tests/testnninputs.cpp, which prints every
test for input versions 3..7. The port only has fillRowV7, so only these chunks are extracted:
  - every "VERSION 7" line and the lines after it, up to the next "VERSION " line, section header
    (65 dashes), "rules " line, SGF line ("(;..."), "Passing hack condition" line or the end of
    the NN-inputs part;
    in "NN Inputs history scoring past end of game or end of phase" the chunk also starts with the
    two lines printed before "VERSION 7" ("rules ...", "enablePassingHacks ...");
  - the whole "Passing hack condition based on as-is scoring" block (fillRowV7 only), through "Ok".
The test name is the section name (the line between two 65-dash lines). Never edit the outputs
by hand.

Usage: extract_runoutputtests.py <runOutputTests.txt> <outdir (fixtures/upstream-v1.16.5/unit)>"""
import os, re, sys

DASHES = '-' * 65
HISTORY_SECTION = 'NN Inputs history scoring past end of game or end of phase'
PASSING_HACK = 'Passing hack condition based on as-is scoring'


def slug(s):
    s = re.sub(r'[^A-Za-z0-9]+', '-', s).strip('-').lower()
    return s or 'unnamed'


def main(src, outdir):
    lines = open(src, encoding='utf-8').read().split('\n')
    end = next(i for i, l in enumerate(lines) if l.startswith('Running neuralnetless search tests'))
    lines = lines[:end]
    stem = 'runoutputtests'
    os.makedirs(os.path.join(outdir, stem), exist_ok=True)

    def is_boundary(l):
        return (l.startswith('VERSION ') or l == DASHES or l.startswith('rules ') or l.startswith(PASSING_HACK)
                or l.startswith('(;'))

    section = None
    counters = {}
    written = []
    i = 0
    while i < len(lines):
        l = lines[i]
        if l == DASHES and i + 2 < len(lines) and lines[i + 2] == DASHES:
            section = lines[i + 1]
            i += 3
            continue
        start = None
        test = section
        if l == 'VERSION 7':
            start = i - 2 if section == HISTORY_SECTION else i
            j = i + 1
            while j < len(lines) and not is_boundary(lines[j]):
                j += 1
        elif l.startswith(PASSING_HACK):
            start = i
            test = PASSING_HACK
            j = i + 1
            while lines[j] != 'Ok':
                j += 1
            j += 1
        if start is None:
            i += 1
            continue
        key = slug(test)
        k = counters.get(key, 0)
        counters[key] = k + 1
        fn = os.path.join(outdir, stem, f'{key}.expected{k}.txt')
        with open(fn, 'w', encoding='utf-8', newline='') as f:
            f.write('\n'.join(lines[start:j]) + '\n')
        written.append((os.path.basename(fn), start + 1, test))
        i = j

    # runScoreTests and runNNSymmetryTests, later in the file.
    all_lines = open(src, encoding='utf-8').read().split('\n')
    score_start = all_lines.index('Running score and utility tests')
    sym_start = next(k for k, l in enumerate(all_lines) if l.startswith('Symmetry 3-1-3-3 useNHWC 0 0'))
    sym_end = all_lines.index('SgfFileTests ok')
    assert score_start < sym_start < sym_end
    for name, a, b in [('Score and utility tests', score_start, sym_start), ('NN symmetry tests', sym_start, sym_end)]:
        fn = f'{slug(name)}.expected0.txt'
        with open(os.path.join(outdir, stem, fn), 'w', encoding='utf-8', newline='') as f:
            f.write('\n'.join(all_lines[a:b]) + '\n')
        written.append((fn, a + 1, name))
    write_index(outdir, stem, written)
    print(f'{src}: {stem}: {len(written)} chunks')


def write_index(outdir, stem, written):
    with open(os.path.join(outdir, stem, 'INDEX.tsv'), 'w') as f:
        f.write('file\tline\ttest\trole\n')
        for fn, lineno, test in written:
            f.write(f'{fn}\t{lineno}\t{test}\texpected\n')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
