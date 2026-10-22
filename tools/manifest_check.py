#!/usr/bin/env python3
"""Check port-manifest.toml against the upstream submodule.

Invariant (see the header of port-manifest.toml): every file under upstream/KataGo/cpp/
is owned exactly once, either by one [[module]] `upstream` list or by one [[ignored]]
entry. Also checks that listed paths exist, ignored globs match something, `secondary`
files are owned by a module, fields are well formed, and [upstream].commit is the
submodule HEAD.

Usage: python3 -I tools/manifest_check.py [--manifest PATH] [--upstream DIR]
Stdlib only (Python >= 3.11 for tomllib). Exits 1 and lists every problem on failure.
"""

import argparse
import re
import subprocess
import sys
import tomllib
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
STATUSES = {"planned", "draft", "compiles", "partial", "done"}
PHASES = {"M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8"}
KINDS = {"backend", "external", "replaced", "not-needed", "deferred", "fixture"}
SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def glob_to_regex(pattern: str) -> re.Pattern:
    """`**` spans path segments, `*` and `?` stay inside one segment."""
    out = []
    i = 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif pattern[i] == "*":
            out.append("[^/]*")
            i += 1
        elif pattern[i] == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(pattern[i]))
            i += 1
    return re.compile("^" + "".join(out) + "$")


def submodule_head(upstream_dir: Path) -> str | None:
    try:
        res = subprocess.run(
            ["git", "-C", str(upstream_dir), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return res.stdout.strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", type=Path, default=REPO / "port-manifest.toml")
    ap.add_argument("--upstream", type=Path, default=REPO / "upstream" / "KataGo")
    args = ap.parse_args()

    errors: list[str] = []
    with open(args.manifest, "rb") as f:
        manifest = tomllib.load(f)

    cpp = args.upstream / "cpp"
    if not cpp.is_dir():
        print(f"error: {cpp} not found (git submodule update --init?)", file=sys.stderr)
        return 2
    files = sorted(p.relative_to(args.upstream).as_posix() for p in cpp.rglob("*") if p.is_file())
    file_set = set(files)

    # [upstream]
    up = manifest.get("upstream", {})
    for key in ("repo", "tag", "commit"):
        if key not in up:
            errors.append(f"[upstream] missing `{key}`")
    commit = up.get("commit", "")
    if commit and not SHA_RE.match(commit):
        errors.append(f"[upstream].commit is not a full sha: {commit!r}")
    head = submodule_head(args.upstream)
    if head is None:
        print("warning: could not read submodule HEAD; skipping commit check", file=sys.stderr)
    elif commit and head != commit:
        errors.append(f"[upstream].commit {commit} != submodule HEAD {head}")

    owners: dict[str, list[str]] = defaultdict(list)  # file -> owning entries
    module_owned: set[str] = set()

    # [[module]]
    rust_paths: set[str] = set()
    for i, mod in enumerate(manifest.get("module", [])):
        name = mod.get("rust", f"<module #{i}>")
        label = f"module {name}"
        for key in ("rust", "upstream", "synced", "status", "phase"):
            if key not in mod:
                errors.append(f"{label}: missing `{key}`")
        if name in rust_paths:
            errors.append(f"{label}: duplicate `rust` path")
        rust_paths.add(name)
        if mod.get("status") not in STATUSES:
            errors.append(f"{label}: status {mod.get('status')!r} not in {sorted(STATUSES)}")
        if mod.get("phase") not in PHASES:
            errors.append(f"{label}: phase {mod.get('phase')!r} not in {sorted(PHASES)}")
        if "synced" in mod and not SHA_RE.match(mod["synced"]):
            errors.append(f"{label}: synced is not a full sha: {mod['synced']!r}")
        seen: set[str] = set()
        for path in mod.get("upstream", []):
            if any(c in path for c in "*?["):
                errors.append(f"{label}: globs not allowed in `upstream`: {path}")
                continue
            if path in seen:
                errors.append(f"{label}: {path} listed twice")
            seen.add(path)
            if path not in file_set:
                errors.append(f"{label}: {path} does not exist upstream")
                continue
            owners[path].append(label)
            module_owned.add(path)

    # [[ignored]]
    for i, ign in enumerate(manifest.get("ignored", [])):
        label = f"ignored #{i + 1} ({ign.get('paths', ['?'])[0]})"
        if not ign.get("reason"):
            errors.append(f"{label}: missing `reason`")
        if ign.get("kind") not in KINDS:
            errors.append(f"{label}: kind {ign.get('kind')!r} not in {sorted(KINDS)}")
        matched: set[str] = set()
        for pat in ign.get("paths", []):
            rx = glob_to_regex(pat)
            hits = [f for f in files if rx.match(f)]
            if not hits:
                errors.append(f"{label}: pattern matches no file: {pat}")
            matched.update(hits)
        for path in matched:
            owners[path].append(label)

    # secondary (checked after all owners are known)
    for mod in manifest.get("module", []):
        label = f"module {mod.get('rust')}"
        for path in mod.get("secondary", []):
            if path not in file_set:
                errors.append(f"{label}: secondary {path} does not exist upstream")
            elif path not in module_owned:
                errors.append(f"{label}: secondary {path} is not owned by any module's `upstream`")
            elif path in mod.get("upstream", []):
                errors.append(f"{label}: {path} is both upstream and secondary")

    unmapped = [f for f in files if not owners.get(f)]
    duplicated = {f: o for f, o in owners.items() if len(o) > 1}
    for f in unmapped:
        errors.append(f"unmapped: {f}")
    for f, o in sorted(duplicated.items()):
        errors.append(f"owned more than once: {f} <- " + "; ".join(o))

    n_mod = len(module_owned)
    n_ign = len(file_set) - n_mod - len(unmapped)
    if errors:
        for e in errors:
            print(e, file=sys.stderr)
        print(f"manifest check FAILED: {len(errors)} problem(s); "
              f"{len(files)} files, {len(unmapped)} unmapped, {len(duplicated)} duplicated",
              file=sys.stderr)
        return 1
    print(f"manifest ok: {len(files)} upstream files = {n_mod} ported by "
          f"{len(manifest.get('module', []))} modules + {n_ign} ignored "
          f"({len(manifest.get('ignored', []))} entries)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
