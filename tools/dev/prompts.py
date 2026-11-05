# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""Print the user prompts and the sources of commits, from oldest to newest.

A REV with ``..`` is a range, other REVs are one commit. The default is main..HEAD.
Rules are in techs/dev-11-code-tools.md#prompts.

Usage:
    python tools/dev/prompts.py                 # main..HEAD
    python tools/dev/prompts.py HEAD~2
    python tools/dev/prompts.py main..dev-prompts abc1234
"""

from __future__ import annotations

import re
import subprocess
import sys

SOURCE = re.compile(r"^([A-Z]+)(?:\[(\d+)\])?:\s*(\S.*)$")
TRAILER = re.compile(r"^[A-Za-z][\w-]*: ")


def commits(revs: list[str]) -> list[tuple[str, str, str]]:
    """(short hash, subject, body) of each commit, oldest first. Ranges skip merges."""
    out = []
    for rev in revs:
        walk = ["--no-merges", "--reverse"] if ".." in rev else ["-1"]
        log = subprocess.run(
            [
                "git",
                "log",
                *walk,
                "--format=%h%x00%s%x00%b%x1e",
                "--end-of-options",
                rev,
                "--",
            ],
            capture_output=True,
            text=True,
        )
        if log.returncode != 0:
            sys.exit(log.stderr.strip())
        for rec in log.stdout.split("\x1e"):
            if rec.strip():
                out.append(tuple(rec.lstrip("\n").split("\x00", 2)))
    return list(dict.fromkeys(out))


def parse(body: str) -> list[tuple[str, list[tuple[str, int | None, str, list[str]]]]]:
    """Prompts after the ``Prompt:`` line, each with its sources (kind, score, ref, detail lines)."""
    lines = body.rstrip().splitlines()
    start = next((i for i, l in enumerate(lines) if l.rstrip() == "Prompt:"), None)
    if start is None:
        return []
    blank = max((i for i, l in enumerate(lines) if not l.strip()), default=start)
    if blank > start and all(
        TRAILER.match(l) and not SOURCE.match(l) for l in lines[blank + 1 :]
    ):
        lines = lines[:blank]
    prompts = []
    quote = None
    for line in lines[start + 1 :]:
        if line.startswith(">"):
            if quote is None:
                quote = []
                prompts.append((quote, []))
            quote.append(line[2:] if line.startswith("> ") else line[1:])
            continue
        quote = None
        if m := SOURCE.match(line):
            if prompts:
                score = None if m[2] is None else int(m[2])
                prompts[-1][1].append((m[1], score, m[3], []))
        elif line[:1].isspace() and line.strip() and prompts and prompts[-1][1]:
            prompts[-1][1][-1][3].append(line)
    return [("\n".join(q), s) for q, s in prompts]


def main(revs: list[str]) -> None:
    found = commits(revs or ["main..HEAD"])
    prompts: dict[str, list[str]] = {}  # prompt -> short hashes
    sources: dict[tuple[str, str], tuple[list[str], set[int | None], set[int]]] = {}
    missing = []
    for sha, subject, body in found:
        parsed = parse(body)
        if not parsed:
            missing.append(f"{sha} {subject}")
        for text, srcs in parsed:
            hashes = prompts.setdefault(text, [])
            hashes.append(sha)
            p = list(prompts).index(text) + 1
            for kind, score, ref, detail in srcs:
                lines, scores, ps = sources.setdefault((kind, ref), ([], set(), set()))
                lines.extend(d for d in detail if d not in lines)
                scores.add(score)
                ps.add(p)

    print(f"commits: {len(found)}, with prompts: {len(found) - len(missing)}")
    print("\n1. Prompts")
    for i, (text, hashes) in enumerate(prompts.items(), 1):
        print(f"\nP{i}  {' '.join(hashes)}")
        print("\n".join(f"> {q}".rstrip() for q in text.splitlines()))
    print("\n2. Sources")
    for (kind, ref), (detail, scores, ps) in sources.items():
        known = sorted(scores - {None}, reverse=True)
        score = "/".join([str(s) for s in known] + ["?"] * (None in scores))
        print(f"{kind}[{score}]: {ref}  ({' '.join(f'P{p}' for p in sorted(ps))})")
        if detail:
            print("\n".join(detail))
    if missing:
        print("\nNo Prompt:")
        print("\n".join(missing))


if __name__ == "__main__":
    main(sys.argv[1:])
