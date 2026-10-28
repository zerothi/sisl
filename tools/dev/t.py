# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""Run pytest with one thread and print a short report.

The arguments go to pytest without change. Exits with the exit code of pytest.
Rules are in techs/dev-11-code-tools.md#t.

Usage:
    python tools/dev/t.py src/sisl/_core/tests/test_geometry.py -k rotate
    python tools/dev/t.py -m "not slow" src/sisl/physics
"""

from __future__ import annotations

import collections
import os
import re
import subprocess
import sys

THREADS = (
    "SISL_NUM_PROCS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
)
PYTEST = ("-q", "-p", "no:cacheprovider", "--tb=short")
SECTION = re.compile(r"^=+ (.+?) =+$")
BLOCK = re.compile(r"^_+ (.+?) _+$")
FRAME = re.compile(r"^(\S+):\d+: ")
MAX_BLOCKS = 10
MAX_E = 20
MAX_CAPTURED = 5


def sections(lines: list[str]) -> list[tuple[str, list[str]]]:
    out = [("", [])]
    for line in lines:
        m = SECTION.match(line)
        if m:
            out.append((m.group(1), []))
        else:
            out[-1][1].append(line)
    return out


def failure(lines: list[str]) -> list[str]:
    """Keep frames of this tree, and limit the error and captured lines."""
    out, keep, n_e, n_cap = [], True, 0, -1
    for line in lines:
        if line.startswith("E "):
            n_e += 1
            if n_e <= MAX_E:
                out.append(line)
            elif n_e == MAX_E + 1:
                out.append("E   ... (more lines)")
            continue
        m = FRAME.match(line)
        if m:
            keep, n_cap = not m.group(1).startswith(("/", "..")), -1
        elif line.startswith("---") and "Captured" in line:
            keep, n_cap = True, 0
        elif n_cap >= 0:
            n_cap += 1
            keep = n_cap <= MAX_CAPTURED
        if keep:
            out.append(line)
    return out


def summary(lines: list[str]) -> list[str]:
    """Group the XFAIL lines by reason."""
    out, xfail = [], collections.Counter()
    for line in lines:
        if line.startswith("XFAIL "):
            xfail[line.partition(" - ")[2] or "(no reason)"] += 1
        else:
            out.append(line[:300])
    out += [f"XFAIL [{n}] {reason}" for reason, n in xfail.items()]
    return out


def report(text: str, durations: bool) -> list[str]:
    lines = text.rstrip().splitlines()
    result = lines.pop() if lines else ""
    out = []
    for title, lines in sections(lines):
        if title in ("FAILURES", "ERRORS"):
            out.append(f"== {title}")
            blocks = [[]]
            for line in lines:
                if BLOCK.match(line):
                    blocks.append([])
                blocks[-1].append(line)
            for block in blocks[1 : MAX_BLOCKS + 1]:
                out += failure(block)
            if len(blocks) > MAX_BLOCKS + 1:
                out.append(f"... {len(blocks) - 1 - MAX_BLOCKS} more; see the summary")
        elif title == "short test summary info":
            out.append(f"== {title}")
            out += summary(lines)
        elif title.startswith("slowest"):
            if durations:
                out.append(f"== {title}")
                out += lines
        elif title not in ("", "warnings summary"):
            out.append(f"== {title}")
            out += lines
    return out + [result]


def main(argv: list[str]) -> int:
    env = os.environ | dict.fromkeys(THREADS, "1")
    run = subprocess.run(
        [sys.executable, "-m", "pytest", *PYTEST, *argv],
        env=env,
        capture_output=True,
        text=True,
    )
    text = run.stdout + run.stderr
    if run.returncode in (0, 1, 5):
        text = "\n".join(report(text, any(a.startswith("--durations") for a in argv)))
    print(text.strip())
    return run.returncode


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
