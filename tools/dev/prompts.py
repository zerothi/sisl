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

SOURCE = re.compile(r"^([A-Z]+)\[(\d+)\]:\s*(.*)$")
TRAILER = re.compile(r"^[A-Za-z][\w-]*: ")


def commits(revs: list[str]) -> list[tuple[str, str, str]]:
    """(short hash, subject, body) of the non-merge commits, oldest first."""
    out = []
    for rev in revs:
        walk = ["--reverse"] if ".." in rev else ["-1"]
        log = subprocess.run(
            [
                "git",
                "log",
                "--no-merges",
                *walk,
                "--format=%h%x00%s%x00%b%x1e",
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


def parse(body: str) -> list[tuple[str, list[tuple[str, int, str, list[str]]]]]:
    """Prompts after the ``Prompt:`` line, each with its sources (kind, score, ref, detail lines)."""
    lines = body.splitlines()
    if "Prompt:" not in lines:
        return []
    prompts = []
    quote = None
    for line in lines[lines.index("Prompt:") + 1 :]:
        if line.startswith(">"):
            if quote is None:
                quote = []
                prompts.append((quote, []))
            quote.append(line[2:] if line.startswith("> ") else line[1:])
            continue
        quote = None
        if m := SOURCE.match(line):
            if prompts:
                prompts[-1][1].append((m[1], int(m[2]), m[3], []))
        elif line[:1].isspace() and line.strip() and prompts and prompts[-1][1]:
            prompts[-1][1][-1][3].append(line)
        elif TRAILER.match(line):
            break
    return [("\n".join(q), s) for q, s in prompts]


def main(revs: list[str]) -> None:
    found = commits(revs or ["main..HEAD"])
    prompts: dict[str, list[str]] = {}  # prompt -> short hashes
    sources: dict[tuple[str, str], tuple[list[str], set[int], set[int]]] = {}
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
        score = "/".join(map(str, sorted(scores, reverse=True)))
        print(f"{kind}[{score}]: {ref}  ({' '.join(f'P{p}' for p in sorted(ps))})")
        if detail:
            print("\n".join(detail))
    if missing:
        print("\nNo Prompt:")
        print("\n".join(missing))


if __name__ == "__main__":
    main(sys.argv[1:])
