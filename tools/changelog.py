#!/usr/bin/env python3
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# -*- encoding:utf-8 -*-
from __future__ import annotations

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

The output is utf8 rst or md.

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
import datetime
import os
import re
import sys
from typing import Literal

from git import Repo
from github import Github

if sys.version_info[:2] < (3, 6):
    raise RuntimeError("Python version must be >= 3.6")

DISCARD_AUTHORS = {"Homu", "lgtm-com[bot]", "dependabot[bot]", "dependabot-preview"}

this_repo = Repo(os.path.join(os.path.dirname(__file__), ".."))

author_msg = """
A total of %d people contributed to this release. People with a "+" by their
names contributed a patch for the first time.
"""

pull_request_msg = """
A total of %d pull requests were merged for this release.
"""

FormatType = Literal["md", "rst"]


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def get_authors(revision_range: str):
    lst_release, _ = [r.strip() for r in revision_range.split("..")]

    def rev_authors(rev: str) -> set[str]:
        # authors, in current release and previous to current release.
        authors_pat = "^.*\\t(.*)$"
        grp1 = "--group=author"
        grp2 = "--group=trailer:co-authored-by"
        logs = this_repo.git.shortlog("-s", grp1, grp2, rev)
        authors = set(re.findall(authors_pat, logs, re.M)) - DISCARD_AUTHORS
        return authors

    authors_cur = rev_authors(revision_range)
    authors_pre = rev_authors(lst_release)

    # Append '+' to new authors.
    authors_new = [s + " +" for s in authors_cur - authors_pre]
    authors_old = list(authors_cur & authors_pre)
    authors = authors_new + authors_old
    authors.sort()
    return authors


def get_commit_date(repo, rev: str) -> datetime.datetime:
    """Retrieve the object that defines the revision"""
    return datetime.datetime.fromtimestamp(repo.commit(rev).committed_date)


def get_pull_requests(repo, revision_range: str):
    prnums = []

    # From regular merges
    merges = this_repo.git.log("--oneline", "--merges", revision_range)
    issues = re.findall(r"Merge pull request \#(\d*)", merges)
    prnums.extend(int(s) for s in issues)

    # From Homu merges (Auto merges)
    issues = re.findall(r"Auto merge of \#(\d*)", merges)
    prnums.extend(int(s) for s in issues)

    # From fast forward squash-merges
    commits = this_repo.git.log(
        "--oneline", "--no-merges", "--first-parent", revision_range
    )
    issues = re.findall(r"^.*(\#|gh-|gh-\#)\((\d+)\)$", commits, re.M)
    prnums.extend(int(s) for s in issues)

    # get PR data from github repo
    prnums = set(prnums)
    prnums = sorted(list(prnums))
    if 1 in prnums:
        # there is a problem in the repo about referencing the first
        # pr (which is actually an issue). So we just let it go.
        del prnums[prnums.index(1)]

    prs = [repo.get_pull(n) for n in prnums]

    return prs


def date2format(date) -> str:
    """Convert the date to the output format we require"""
    date = date.strftime("%d of %B %Y")
    if date[0] == "0":
        date = date[1:]
    return date


def heading(heading: str, lvl: int, format: FormatType) -> str:
    """Convert to proper heading"""
    if format == "rst":
        heading = heading.strip()
        n = len(heading)
        return f"{heading}\n{' =-^'[lvl]*n}"
    return f"{'#'*lvl} {heading}"


def main(token: str, revision_range: str, format: FormatType = "md") -> None:
    prior_rel, current_rel = [r.strip() for r in revision_range.split("..")]
    if not current_rel:
        # default to HEAD as eo-commits
        current_rel = "HEAD"

    prior_rel_date = get_commit_date(this_repo, prior_rel)
    current_rel_date = get_commit_date(this_repo, current_rel)

    prior_version = prior_rel
    if prior_rel.startswith("v"):
        prior_rel = prior_rel[1:]

    current_version = current_rel
    if current_rel.startswith("v"):
        current_rel = current_rel[1:]
        current_version = current_rel
    elif current_rel == "HEAD":
        current_version = "TBD"

    github = Github(token)
    github_repo = github.get_repo("zerothi/sisl")

    # document authors
    print()
    print(heading("Contributors", 1, format))

    authors = get_authors(revision_range)
    print(author_msg % len(authors))

    for author in authors:
        print("* " + author)

    # document pull requests
    pull_requests = get_pull_requests(github_repo, revision_range)
    if format == "rst":
        pull_msg = "* :pull:`{0}`"
    else:
        pull_msg = "* [#{0}]({1}): {2}"

    # split into actual and maintenance PR's
    code_pull_requests = filter(
        lambda pr: pr.user.login not in DISCARD_AUTHORS, pull_requests
    )
    code_pull_requests = list(code_pull_requests)

    maint_pull_requests = filter(
        lambda pr: pr.user.login in DISCARD_AUTHORS, pull_requests
    )
    maint_pull_requests = list(maint_pull_requests)

    def sanitize_whitespace(string: str) -> str:
        return re.sub(r"\s+", " ", string.strip())

    def sanitize_backtick(string: str) -> str:
        # Courtesy of numpy! See numpy/tools/changelog.py
        # substitute any single backtick not adjacent to a backtick
        # for a double backtick
        string = re.sub(
            "(?P<pre>(?:^|(?<=[^`])))`(?P<post>(?=[^`]|$))",
            r"\g<pre>``\g<post>",
            string,
        )
        return string

    def shorten(string: str, max_len: int = 80) -> str:
        remainder = re.sub(r"\s.*$", "...", string[max_len - 20 :])
        if len(string) > max_len:
            remainder = re.sub(r"\s.*$", "...", string[max_len - 20 :])
            if len(remainder) > 20:
                string = string[:max_len] + "..."
            else:
                string = string[: max_len - 20] + remainder

            # check if there is a cut in a code-block
            nticks = 4
            # if we will change to two backticks, then this only needs
            # changing
            if abundance_ticks := string.count("`") % nticks != 0:
                string = string[:-3] + "`" * (nticks - abundance_ticks) + "..."

        return string

    def sanitize_title(string):
        string = sanitize_whitespace(string)
        string = sanitize_backtick(string)
        return shorten(string)

    if code_pull_requests:
        print()
        print(heading("Pull requests merged", 1, format))
        print(pull_request_msg % len(code_pull_requests))

        for pull in code_pull_requests:
            title = sanitize_title(pull.title.strip())
            print(pull_msg.format(pull.number, pull.html_url, title))

    if maint_pull_requests:
        print()
        print(heading("Maintenance pull requests merged", 2, format))
        print()
        for pull in maint_pull_requests:
            title = sanitize_title(pull.title.strip())
            print(pull_msg.format(pull.number, pull.html_url, title))


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Generate author/pr lists for release")
    parser.add_argument(
        "--format", choices=("md", "rst"), help="which format to write out in"
    )
    parser.add_argument("token", help="github access token")
    parser.add_argument("revision_range", help="<revision>..<revision>")
    args = parser.parse_args()
    main(args.token, args.revision_range, format=args.format)
