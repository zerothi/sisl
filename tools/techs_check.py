# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""Point at the `techs/` chapters that own changed files.

Warn only: always exits 0. Rules are in techs/README.md#ownership.

Usage:
    python tools/techs_check.py          # staged changes (pre-commit)
    python tools/techs_check.py --all    # coverage report for all tracked files
"""

from __future__ import annotations

import re
import subprocess
import sys
from fnmatch import fnmatchcase
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TECHS = ROOT / "techs"
OWNS = re.compile(r"^Owns:(.*)$", re.M)
SKIP = ("*/tests/*", "*/__pycache__/*")


def git(*args: str) -> list[str]:
    out = subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
    )
    return [line for line in out.stdout.splitlines() if line]


def read_owners() -> dict[str, list[str]]:
    owners = {}
    for chapter in sorted(TECHS.glob("*.md")):
        m = OWNS.search(chapter.read_text())
        if m:
            owners[f"techs/{chapter.name}"] = re.findall(r"`([^`]+)`", m.group(1))
    return owners


def specificity(pattern: str) -> int:
    return len(re.split(r"[*?\[]", pattern, maxsplit=1)[0])


def owner_of(path: str, owners: dict[str, list[str]]) -> list[str]:
    best, chapters = -1, []
    for chapter, patterns in owners.items():
        for pattern in patterns:
            if not fnmatchcase(path, pattern):
                continue
            s = specificity(pattern)
            if s > best:
                best, chapters = s, [chapter]
            elif s == best and chapter not in chapters:
                chapters.append(chapter)
    return chapters


def checked(path: str) -> bool:
    return not any(fnmatchcase(path, s) for s in SKIP)


def main(argv: list[str]) -> int:
    owners = read_owners()
    tracked = git("ls-files")

    for chapter, patterns in owners.items():
        for pattern in patterns:
            if not any(fnmatchcase(f, pattern) for f in tracked):
                print(f"techs: {chapter}: `{pattern}` matches no tracked file")

    if "--all" in argv:
        unowned = [
            f
            for f in tracked
            if f.startswith("src/") and checked(f) and not owner_of(f, owners)
        ]
        print(f"techs: {len(unowned)} tracked files under src/ have no chapter:")
        for f in unowned:
            print(f"  {f}")
        return 0

    changed = git("diff", "--cached", "--name-only", "--diff-filter=ACMRD")
    touched = {f for f in changed if f.startswith("techs/")}
    review: dict[str, list[str]] = {}
    unowned = []
    for f in changed:
        if f.startswith("techs/") or not checked(f):
            continue
        chapters = owner_of(f, owners)
        if not chapters and f.startswith("src/"):
            unowned.append(f)
        elif chapters and not touched.intersection(chapters):
            review.setdefault(" or ".join(chapters), []).append(f)

    if review:
        print("techs: check that these chapters are still correct:")
        for chapter, files in review.items():
            print(f"  {chapter}")
            for f in files:
                print(f"    <- {f}")
    if unowned:
        print("techs: no chapter owns (add an `Owns:` pattern if it has decisions):")
        for f in unowned:
            print(f"  {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
