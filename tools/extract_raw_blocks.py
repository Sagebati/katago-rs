#!/usr/bin/env python3
"""Extract every R"%%(...)%%" raw string block from an upstream C++ test file into
fixtures/<tag>/unit/<stem>/<test-slug>.<role><k>.txt  (role = input | expected).

The test name is the last `const char* name = "..."` seen in the enclosing function, or the
function name if none. role is `expected` when the block is assigned to `expected`.
Pilot stand-in for the planned `xtask extract-tests`. Never edit the outputs by hand."""
import os, re, sys

def slug(s):
    s = re.sub(r'[^A-Za-z0-9]+', '-', s).strip('-').lower()
    return s or 'unnamed'

def main(src, outdir):
    text = open(src, encoding='utf-8').read()
    stem = os.path.splitext(os.path.basename(src))[0]
    os.makedirs(os.path.join(outdir, stem), exist_ok=True)
    func_re = re.compile(r'^(?:static\s+)?void\s+([A-Za-z0-9_:]+)\s*\(', re.M)
    name_re = re.compile(r'const char\*\s*name\s*=\s*"([^"]*)"')
    # Walk tokens in order: function starts, name decls, raw blocks.
    events = []
    for m in func_re.finditer(text): events.append((m.start(), 'func', m.group(1).split('::')[-1]))
    for m in name_re.finditer(text): events.append((m.start(), 'name', m.group(1)))
    pos = 0
    while True:
        i = text.find('R"%%(', pos)
        if i < 0: break
        j = text.find(')%%"', i)
        events.append((i, 'block', (i, j)))
        pos = j + 4
    events.sort(key=lambda e: e[0])
    cur_func, cur_name = None, None
    counters = {}
    written = []
    for at, kind, val in events:
        if kind == 'func':
            cur_func, cur_name = val, None
        elif kind == 'name':
            cur_name = val
        else:
            i, j = val
            line_start = text.rfind('\n', 0, i) + 1
            prefix = text[line_start:i]
            role = 'expected' if re.search(r'\bexpected\w*\s*=\s*$', prefix) else 'input'
            test = cur_name if cur_name is not None else (cur_func or 'toplevel')
            key = (slug(test), role)
            k = counters.get(key, 0); counters[key] = k + 1
            content = text[i + len('R"%%('):j]
            fn = os.path.join(outdir, stem, f'{key[0]}.{role}{k}.txt')
            with open(fn, 'w', encoding='utf-8', newline='') as f: f.write(content)
            lineno = text.count('\n', 0, i) + 1
            written.append((fn, lineno, test, role))
    with open(os.path.join(outdir, stem, 'INDEX.tsv'), 'w') as f:
        f.write('file\tline\ttest\trole\n')
        for fn, lineno, test, role in written:
            f.write(f'{os.path.basename(fn)}\t{lineno}\t{test}\t{role}\n')
    print(f'{src}: {len(written)} blocks')

if __name__ == '__main__':
    outdir = sys.argv[1]
    for src in sys.argv[2:]: main(src, outdir)
