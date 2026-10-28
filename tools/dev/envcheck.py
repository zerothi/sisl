# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""Check that the sisl which runs is the sisl of this source tree.

The source tree is the git top-level of the current directory.
Exits 1 if a check fails. Rules are in techs/dev-11-code-tools.md#envcheck.

Usage:
    python tools/dev/envcheck.py           # import path, PYTHONPATH, stale builds, test variables
    python tools/dev/envcheck.py PATH...   # also: the module that runs for each source file
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

PACKAGES = ("sisl", "sisl_toolbox")
THREADS = (
    "SISL_NUM_PROCS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
)
CIMPORT = re.compile(
    r"^\s*(?:from\s+(\S+)\s+cimport\s+([\w \t,]*)|cimport\s+([\w.]+))", re.M
)


def module_name(src: Path, path: Path) -> str:
    parts = list(path.resolve().relative_to(src).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def module_path(src: Path, name: str, suffix: str) -> Path:
    return src.joinpath(*name.split(".")).with_suffix(suffix)


def pxd_deps(src: Path, pyx: Path) -> set[Path]:
    """The .pxd files that `pyx` uses, also through other .pxd files."""
    todo, deps = [pyx], {pyx.with_suffix(".pxd")} - {pyx}
    while todo:
        f = todo.pop()
        package = module_name(src, f).rsplit(".", 1)[0]
        for m in CIMPORT.finditer(f.read_text()):
            base = m.group(1) or m.group(3)
            if base.startswith("."):
                rest = base.lstrip(".")
                parts = package.split(".")
                parts = parts[: len(parts) - (len(base) - len(rest)) + 1]
                base = ".".join(parts + ([rest] if rest else []))
            names = [base] + [
                f"{base}.{n.strip()}"
                for n in (m.group(2) or "").split(",")
                if n.strip()
            ]
            for name in names:
                pxd = module_path(src, name, ".pxd")
                if pxd.is_file() and pxd not in deps:
                    deps.add(pxd)
                    todo.append(pxd)
    return {d for d in deps if d.is_file()}


def origin(name: str) -> str | None:
    try:
        spec = importlib.util.find_spec(name)
    except Exception as e:  # a parent package can fail to import
        return f"not importable: {type(e).__name__}: {e}"
    return spec.origin if spec else None


def check_import(src: Path) -> list[str]:
    out = []
    for pkg in PACKAGES:
        try:
            dirs = [Path(d).resolve() for d in importlib.import_module(pkg).__path__]
        except ImportError as e:
            out.append(f"FAIL  {pkg}: not importable: {e}")
            continue
        ok = any(d.is_relative_to(src) for d in dirs)
        where = ", ".join(map(str, dirs))
        out.append(
            f"{'ok  ' if ok else 'FAIL'}  {pkg} from {where}"
            + ("" if ok else f" (not below {src})")
        )
    for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep):
        e = Path(entry).resolve() if entry else None
        if e and e != src and any((e / pkg).is_dir() for pkg in PACKAGES):
            out.append(f"FAIL  PYTHONPATH has a sisl: {entry}")
    return out


def check_stale(src: Path, root: Path) -> list[str]:
    pyxs = subprocess.run(
        ["git", "ls-files", "src/*.pyx"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    stale, missing = [], []
    for pyx in map(root.joinpath, pyxs):
        name = module_name(src, pyx)
        so = origin(name)
        if not so or not so.endswith((".so", ".pyd")):
            missing.append(name)
            continue
        newer = [
            d
            for d in {pyx} | pxd_deps(src, pyx)
            if d.stat().st_mtime > Path(so).stat().st_mtime
        ]
        if newer:
            stale.append(f"{name} (newer: {', '.join(d.name for d in sorted(newer))})")
    out = [f"FAIL  stale build: {s}" for s in stale]
    out += [f"FAIL  not built: {m}" for m in missing]
    if not out:
        out.append(
            f"ok    {len(pyxs)} compiled modules are newer than their .pyx and .pxd"
        )
    return out


def check_variables() -> list[str]:
    files = os.environ.get("SISL_FILES_TESTS")
    if not files:
        out = ["note  SISL_FILES_TESTS is not set: IO tests xfail, they do not pass"]
    elif not Path(files).is_dir():
        out = [f"FAIL  SISL_FILES_TESTS is not a directory: {files}"]
    else:
        out = [f"ok    SISL_FILES_TESTS={files}"]
    threads = " ".join(f"{v}={os.environ.get(v, '-')}" for v in THREADS)
    return out + [f"note  {threads}"]


def check_paths(src: Path, paths: list[Path]) -> list[str]:
    out = []
    for path in paths:
        stem = path.stem.lstrip("_")
        companions = [
            p
            for p in sorted(path.resolve().parent.glob(f"*{stem}.*"))
            if p.stem.lstrip("_") == stem
            and p.suffix in (".py", ".pyx")
            and p != path.resolve()
        ]
        for p in [path.resolve(), *companions]:
            name = module_name(src, p)
            loaded = (
                "loaded by 'import sisl'"
                if name in sys.modules
                else "not loaded by 'import sisl'"
            )
            out.append(
                f"{'path ' if p == path.resolve() else 'twin '} {p.relative_to(src)}: {name} -> {origin(name)} ({loaded})"
            )
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("paths", nargs="*", type=Path, help="source files below src/")
    args = p.parse_args(argv)

    root = Path(
        subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )
    src = root / "src"
    out = [f"note  python {sys.executable}"]
    out += check_import(src)
    if out[1].startswith("FAIL"):
        out.append("skip  stale builds: the compiled modules are of a different tree")
    else:
        out += check_stale(src, root)
    out += check_variables()
    out += check_paths(src, args.paths)
    print("\n".join(out))
    return int(any(line.startswith("FAIL") for line in out))


if __name__ == "__main__":
    sys.exit(main())
