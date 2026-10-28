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
from pathlib import Path

from envcheck import THREADS, git_root

PYTEST = ("-q", "-p", "no:cacheprovider", "--tb=short")
SECTION = re.compile(r"^=+ (.+?) =+$")
BLOCK = re.compile(r"^_+ (.+?) _+$")
FRAME = re.compile(r"^(\S+):\d+: ")
PROGRESS = re.compile(r"^[.FEsxX]+\s*(\[\s*\d+%\])?$")
MAX_BLOCKS = 10
MAX_E = 20
MAX_CAPTURED = 5
MAX_LINE = 300
MAX_STDERR = 20


def sections(lines: list[str]) -> list[tuple[str, list[str]]]:
    out = [("", [])]
    for line in lines:
        m = SECTION.match(line)
        if m:
            out.append((m.group(1), []))
        else:
            out[-1][1].append(line)
    return out


def failure(lines: list[str], root: Path) -> list[str]:
    """Keep frames of this tree, and limit the error and captured lines."""
    out, keep, n_e, n_cap = [], True, 0, -1
    removed = collections.Counter()
    for line in lines:
        if line.startswith("E "):
            n_e += 1
            if n_e <= MAX_E:
                out.append(line)
            removed["E lines"] += n_e > MAX_E
            continue
        m = FRAME.match(line)
        if m:
            keep, n_cap = Path(m.group(1)).resolve().is_relative_to(root), -1
            removed["frames outside the tree"] += not keep
        elif line.startswith("---") and "Captured" in line:
            keep, n_cap = True, 0
        elif n_cap >= 0:
            n_cap += 1
            keep = n_cap <= MAX_CAPTURED
            removed["captured lines"] += not keep
        if keep:
            out.append(line)
    if +removed:
        out.append(f"(removed: {', '.join(f'{n} {k}' for k, n in (+removed).items())})")
    return out


def summary(lines: list[str]) -> list[str]:
    """Group the XFAIL lines by reason."""
    out, xfail = [], collections.Counter()
    for line in lines:
        if line.startswith("XFAIL "):
            xfail[line.partition(" - ")[2] or "(no reason)"] += 1
        else:
            out.append(line if len(line) <= MAX_LINE else line[:MAX_LINE] + " ...")
    out += [f"XFAIL [{n}] {reason}" for reason, n in xfail.items()]
    return out


def report(stdout: str, root: Path, durations: bool) -> list[str]:
    lines = stdout.rstrip().splitlines()
    result = lines.pop() if lines else ""
    out, removed = [], []
    for title, lines in sections(lines):
        if title == "":
            # test output (-s), collected tests (--co) and headers (-v)
            out += [line for line in lines if line.strip() and not PROGRESS.match(line)]
        elif title in ("FAILURES", "ERRORS"):
            out.append(f"== {title}")
            blocks = [[]]
            for line in lines:
                if BLOCK.match(line):
                    blocks.append([])
                blocks[-1].append(line)
            for block in blocks[1 : MAX_BLOCKS + 1]:
                out += failure(block, root)
            if len(blocks) > MAX_BLOCKS + 1:
                out.append(f"... {len(blocks) - 1 - MAX_BLOCKS} more; see the summary")
        elif title == "short test summary info":
            out.append(f"== {title}")
            out += summary(lines)
        elif title == "warnings summary" or (
            title.startswith("slowest") and not durations
        ):
            removed.append(title)
        else:
            out.append(f"== {title}")
            out += lines
    if removed:
        out.append(f"(removed: {', '.join(removed)}; run pytest for the full output)")
    return out + [result]


def main(argv: list[str]) -> int:
    env = os.environ | dict.fromkeys(THREADS, "1")
    run = subprocess.run(
        [sys.executable, "-m", "pytest", *PYTEST, *argv],
        env=env,
        capture_output=True,
        text=True,
    )
    if run.returncode not in (0, 1, 5):
        print((run.stdout + run.stderr).strip())
        return run.returncode
    durations = any(a.startswith("--durations") for a in argv)
    print("\n".join(report(run.stdout, git_root() or Path.cwd(), durations)))
    err = run.stderr.strip().splitlines()
    if err:
        cut = (
            f" (last {MAX_STDERR} of {len(err)} lines)" if len(err) > MAX_STDERR else ""
        )
        print(f"== stderr{cut}")
        print("\n".join(err[-MAX_STDERR:]))
    return run.returncode


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
