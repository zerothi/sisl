#!/usr/bin/env python3
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# -*- encoding:utf-8 -*-
"""
Script to generate contributor and pull request lists

This script generates contributor and pull request lists for release
changelogs using Github v3 protocol. Use requires an authentication token in
order to have sufficient bandwidth, you can get one following the directions at
`<https://help.github.com/articles/creating-an-access-token-for-command-line-use/>_
Don't add any scope, as the default is read access to public information. The
token may be stored in an environment variable as you only get one chance to
see it.

Usage::

    $ ./tools/changelog.py <token> <revision range>

The output is utf8 rst.

Dependencies
------------

- gitpython
- pygithub

Some code was copied from scipy `tools/gh_list.py` and `tools/authors.py`.

Examples
--------

From the bash command line with $GITHUB token::

    $ ./tools/changelog.py $GITHUB v1.13.0..v1.14.0 > 1.14.0-changelog.rst

"""
import os
import sys
import re
from git import Repo
from github import Github

if sys.version_info[:2] < (3, 6):
    raise RuntimeError("Python version must be >= 3.6")

this_repo = Repo(os.path.join(os.path.dirname(__file__), ".."))

author_msg = """
A total of %d people contributed to this release.  People with a "+" by their
names contributed a patch for the first time.
"""

pull_request_msg = """
A total of %d pull requests were merged for this release.
"""


def get_authors(revision_range):
    pat = "^.*\\t(.*)$"
    lst_release, cur_release = [r.strip() for r in revision_range.split("..")]

    # authors, in current release and previous to current release.
    cur = set(re.findall(pat, this_repo.git.shortlog("-s", revision_range), re.M))
    pre = set(re.findall(pat, this_repo.git.shortlog("-s", lst_release), re.M))

    # Homu is the author of auto merges, clean him out.
    cur.discard("Homu")
    pre.discard("Homu")

    # Append '+' to new authors.
    authors = [s + " +" for s in cur - pre] + [s for s in cur & pre]
    authors.sort()
    return authors


def get_pull_requests(repo, revision_range):
    prnums = []

    # From regular merges
    merges = this_repo.git.log("--oneline", "--merges", revision_range)
    issues = re.findall("Merge pull request \\#(\\d*)", merges)
    prnums.extend(int(s) for s in issues)

    # From Homu merges (Auto merges)
    issues = re.findall("Auto merge of \\#(\\d*)", merges)
    prnums.extend(int(s) for s in issues)

    # From fast forward squash-merges
    commits = this_repo.git.log(
        "--oneline", "--no-merges", "--first-parent", revision_range
    )
    issues = re.findall("^.*\\(\\#(\\d+)\\)$", commits, re.M)
    prnums.extend(int(s) for s in issues)

    # get PR data from github repo
    prnums.sort()
    prs = [repo.get_pull(n) for n in prnums]
    return prs


def main(token, revision_range):
    lst_release, cur_release = [r.strip() for r in revision_range.split("..")]

    github = Github(token)
    github_repo = github.get_repo("zerothi/sisl")

    # document authors
    authors = get_authors(revision_range)
    heading = "Contributors"
    print()
    print(heading)
    print("=" * len(heading))
    print(author_msg % len(authors))

    for s in authors:
        print("* " + s)

    # document pull requests
    pull_requests = get_pull_requests(github_repo, revision_range)
    heading = "Pull requests merged"
    pull_msg = "* #{0}: {2}"

    print()
    print(heading)
    print("=" * len(heading))
    print(pull_request_msg % len(pull_requests))

    for pull in pull_requests:
        title = re.sub("\\s+", " ", pull.title.strip())
        if len(title) > 60:
            remainder = re.sub("\\s.*$", "...", title[60:])
            if len(remainder) > 20:
                title = title[:80] + "..."
            else:
                title = title[:60] + remainder
        print(pull_msg.format(pull.number, pull.html_url, title))

    # Also add the CHANGELOG.md information
    versions = []
    if lst_release.startswith("v"):
        versions.append(lst_release[1:])
    else:
        versions.append(lst_release)
    if cur_release.startswith("v"):
        versions.append(cur_release[1:])
    elif cur_release == "HEAD":
        versions.append("Unreleased")
    else:
        versions.append(cur_release)

    print_out = False
    out = []
    for line in open("../CHANGELOG.md", 'r'):
        if f"[{versions[1]}]" in line:
            print_out = True
            continue
        elif f"[{versions[0]}]" in line:
            break

        if print_out:
            out.append(line)

    if len(out) > 0:
        # new-line
        print()
        print(''.join(out).strip())


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Generate author/pr lists for release")
    parser.add_argument("token", help="github access token")
    parser.add_argument("revision_range", help="<revision>..<revision>")
    args = parser.parse_args()
    main(args.token, args.revision_range)
