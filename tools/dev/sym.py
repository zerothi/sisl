# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""Print the outline of a source file, or the source of named symbols.

Works on .py, .pyx and .pxd. Rules are in techs/dev-11-code-tools.md#sym.

Usage:
    python tools/dev/sym.py FILE                  # outline: line ranges of classes and functions
    python tools/dev/sym.py FILE NAME...          # source of each symbol (Class.method or method)
    python tools/dev/sym.py FILE NAME... --sig    # signature and doc-string only
    python tools/dev/sym.py FILE NAME --outline   # outline of the members of NAME
    python tools/dev/sym.py FILE NAME... --full   # source, also when it is longer than MAX_LINES
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
from dataclasses import dataclass
from pathlib import Path

MAX_LINES = 300
HEAD = re.compile(
    r"^(\s*)(?:async\s+)?(?:(?:def|class)\s+(\w+)"
    r"|(?:cdef|cpdef)\s+(?:class\s+)?(?:\([^)]*\)\s*)?[^=(]*?(\w+)\s*[(:])"
)


@dataclass
class Symbol:
    qual: str
    kind: str
    start: int  # first line, with decorators (0-based)
    head_end: int  # last line of the signature
    end: int  # last line of the body
    depth: int
    decorators: str


def scan(line: str, quote: str | None = None) -> tuple[str, str | None]:
    """The code of `line` without strings and comments, and the open triple quote at its end."""
    code, i = [], 0
    while i < len(line):
        if quote:
            j = line.find(quote, i)
            while j > 0 and (len(line[:j]) - len(line[:j].rstrip("\\"))) % 2:
                j = line.find(quote, j + 1)
            if j < 0:
                break
            i, quote = j + 3, None
        elif line[i] == "#":
            break
        elif line.startswith(('"""', "'''"), i):
            i, quote = i + 3, line[i : i + 3]
        elif line[i] in "\"'":
            j = i + 1
            while j < len(line) and line[j] != line[i]:
                j += 2 if line[j] == "\\" else 1
            i = j + 1
        else:
            code.append(line[i])
            i += 1
    return "".join(code), quote


def in_string(lines: list[str]) -> list[bool]:
    """For each line: True if it starts inside a triple-quoted string."""
    out, quote = [], None
    for line in lines:
        out.append(quote is not None)
        quote = scan(line, quote)[1]
    return out


def indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def code(line: str) -> str:
    return scan(line)[0].strip()


def head_end(lines: list[str], i: int) -> int:
    depth = 0
    for j in range(i, len(lines)):
        c = code(lines[j])
        depth += sum(c.count(b) for b in "([{") - sum(c.count(b) for b in ")]}")
        if depth <= 0:
            return j
    return len(lines) - 1


def decorator_start(lines: list[str], instr: list[bool], i: int, ind: int) -> int:
    start = i
    j = i - 1
    while j >= 0:
        line = lines[j]
        if instr[j] or line.lstrip().startswith("#"):
            pass
        elif indent(line) == ind and line.lstrip().startswith("@"):
            start = j
        elif not line.strip() or indent(line) < ind:
            break
        elif indent(line) == ind and not line.lstrip().startswith(")"):
            break
        j -= 1
    return start


def parse(lines: list[str]) -> list[Symbol]:
    instr = in_string(lines)
    symbols, stack = [], []
    for i, line in enumerate(lines):
        m = HEAD.match(line)
        if instr[i] or not m:
            continue
        ind = len(m.group(1))
        name = m.group(2) or m.group(3)
        kind = line[ind : m.start(2) if m.group(2) else m.start(3)].strip()
        h = head_end(lines, i)
        end = h
        if code(lines[h]).endswith(":"):
            for j in range(h + 1, len(lines)):
                c = lines[j].strip()
                if instr[j] or (c and not c.startswith("#") and indent(lines[j]) > ind):
                    end = j
                elif c and not c.startswith("#"):
                    break
        start = decorator_start(lines, instr, i, ind)
        decorators = [d for d in lines[start:i] if not d.lstrip().startswith("#")]
        decorators = " ".join(" ".join(decorators).split())
        decorators = re.sub(r"\(\s", "(", re.sub(r",?\s\)", ")", decorators))
        decorators = re.sub(r"@cython\.\w+(\([^)]*\))?\s*", "", decorators).strip()
        while stack and stack[-1].end < i:
            stack.pop()
        qual = f"{stack[-1].qual}.{name}" if stack else name
        sym = Symbol(qual, kind, start, h, end, len(stack), decorators)
        symbols.append(sym)
        stack.append(sym)
    return symbols


def docstring_end(lines: list[str], sym: Symbol) -> int:
    for j in range(sym.head_end + 1, sym.end + 1):
        s = lines[j].strip()
        if not s:
            continue
        m = re.match(r"[rRuU]?(\"\"\"|''')", s)
        if not m:
            break
        if s.count(m.group(1)) >= 2:
            return j
        for k in range(j + 1, sym.end + 1):
            if m.group(1) in lines[k]:
                return k
        break
    return sym.head_end


def outline(symbols: list[Symbol], base: int = 0) -> str:
    out = []
    for s in symbols:
        rng = f"{s.start + 1}-{s.end + 1}"
        deco = f"  {s.decorators}" if s.decorators else ""
        out.append(
            f"{rng:>11}  {'  ' * (s.depth - base)}{s.kind} {s.qual.rsplit('.', 1)[-1]}{deco}"
        )
    return "\n".join(out)


def find(symbols: list[Symbol], name: str) -> list[Symbol]:
    return [s for s in symbols if s.qual == name or s.qual.endswith(f".{name}")]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("file", type=Path)
    p.add_argument("names", nargs="*")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--sig", action="store_true", help="signature and doc-string only")
    g.add_argument(
        "--outline", action="store_true", help="outline of the members of NAME"
    )
    g.add_argument("--full", action="store_true", help=f"no limit of {MAX_LINES} lines")
    args = p.parse_args(argv)
    if not args.names and (args.sig or args.outline or args.full):
        p.error("--sig, --outline and --full need a NAME")

    lines = args.file.read_text().splitlines()
    symbols = parse(lines)
    if not args.names:
        print(outline(symbols))
        return 0

    status = 0
    for name in args.names:
        found = find(symbols, name)
        if not found:
            close = difflib.get_close_matches(name, [s.qual for s in symbols], n=5)
            print(
                f"{args.file}: no symbol {name!r}. Close: {', '.join(close) or '-'}",
                file=sys.stderr,
            )
            status = 1
        for s in found:
            members = [m for m in symbols if m.start > s.start and m.end <= s.end]
            n = s.end - s.start + 1
            if args.outline or (
                not args.sig and not args.full and n > MAX_LINES and members
            ):
                note = (
                    ""
                    if args.outline
                    else f" ({n} lines; outline only, use --full for the source)"
                )
                print(
                    f"# {args.file}:{s.start + 1}-{s.end + 1} {s.kind} {s.qual}{note}"
                )
                print(outline(members, s.depth + 1))
                continue
            last = docstring_end(lines, s) if args.sig else s.end
            print(f"# {args.file}:{s.start + 1}-{s.end + 1} {s.kind} {s.qual}")
            print("\n".join(lines[s.start : last + 1]))
    return status


if __name__ == "__main__":
    sys.exit(main())
