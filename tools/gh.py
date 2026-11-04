# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""Keep a local, compact text archive of the GitHub issues, pull requests and discussions.

Reads the public GitHub API; credentials are optional. Rules are in techs/dev-12-gh-archive.md.

Usage:
    python tools/gh.py sync            # all issues and pull requests (resumes at a rate limit)
    python tools/gh.py issue 123       # print the conversation of issue 123
    python tools/gh.py pr 456          # print the conversation of pull request 456
    python tools/gh.py discussion 7    # needs GITHUB_TOKEN or GH_TOKEN
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

API = "https://api.github.com"
TOKEN = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
# timeline events kept besides comments: name -> text
EVENTS = {
    "closed": "closed",
    "reopened": "reopened",
    "merged": "merged",
    "referenced": "referenced by commit",
    "cross-referenced": "mentioned in",
}


class RateLimited(Exception):
    pass


def now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def request(url: str, data: dict | None = None):
    """Return (json, next page url) of a GET, or of a POST of `data`."""
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "sisl-gh-archive",
    }
    if TOKEN:
        headers["Authorization"] = f"Bearer {TOKEN}"
    body = None if data is None else json.dumps(data).encode()
    req = urllib.request.Request(url, body, headers)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            m = re.search(r'<([^>]+)>; rel="next"', r.headers.get("Link", ""))
            return json.load(r), m and m.group(1)
    except urllib.error.HTTPError as e:
        if e.code in (403, 429) and e.headers.get("X-RateLimit-Remaining") == "0":
            reset = int(e.headers.get("X-RateLimit-Reset", 0))
            raise RateLimited(
                datetime.datetime.fromtimestamp(reset).strftime("%H:%M")
            ) from None
        raise SystemExit(f"{e.code} {e.reason}: {url}") from None


def pages(url: str | None):
    while url:
        data, url = request(url)
        yield from data


def date(s: str | None) -> str:
    return s[:16].replace("T", " ") if s else ""


def clean(text: str | None) -> str:
    text = re.sub(r"<!--.*?-->", "", text or "", flags=re.S).replace("\r", "")
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def user(u: dict | None) -> tuple[str, bool]:
    u = u or {}
    return u.get("login", "ghost"), u.get("type") == "Bot"


class Archive:
    def __init__(self, root: Path, repo: str):
        self.root = root
        self.repo = repo
        self.raw = root / "raw"
        self.raw.mkdir(parents=True, exist_ok=True)

    def url(self, path: str) -> str:
        return f"{API}/repos/{self.repo}/{path}"

    # storage
    def write(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(text)
        tmp.replace(path)

    def load(self, key: str) -> dict | None:
        p = self.raw / f"{key}.json"
        return json.loads(p.read_text()) if p.exists() else None

    def save(self, key: str, d: dict) -> None:
        self.write(self.raw / f"{key}.json", json.dumps(d, separators=(",", ":")))
        if key.startswith("d"):
            self.write(self.root / "discussions" / f"{d['n']}.md", render_discussion(d))
        else:
            self.write(self.md(d), render(d))

    def md(self, d: dict) -> Path:
        return (
            self.root / ("pulls" if d["kind"] == "pull" else "issues") / f"{d['n']}.md"
        )

    def index(self) -> None:
        rows = []
        for p in self.raw.glob("*.json"):
            d = json.loads(p.read_text())
            state = "merged" if d.get("merged") else d["state"]
            rows.append(
                (
                    d["kind"],
                    d["n"],
                    state,
                    date(d["updated"])[:10],
                    d["author"],
                    d["title"],
                )
            )
        rows.sort(key=lambda r: (r[0], -r[1]))
        self.write(
            self.root / "index.tsv",
            "".join("\t".join(map(str, r)) + "\n" for r in rows),
        )

    # merging of API objects into raw items
    def item(self, d: dict) -> dict:
        n = d["number"]
        old = self.load(str(n)) or {
            "comments": {},
            "review_comments": {},
            "events": None,
        }
        author, _ = user(d["user"])
        old.update(
            n=n,
            kind="pull" if "pull_request" in d else "issue",
            title=d["title"],
            state=d["state"],
            reason=d.get("state_reason"),
            author=author,
            created=d["created_at"],
            updated=d["updated_at"],
            closed=d.get("closed_at"),
            merged=(d.get("pull_request") or {}).get("merged_at"),
            draft=d.get("draft", False),
            labels=[lab["name"] for lab in d.get("labels", [])],
            body=clean(d.get("body")),
        )
        return old

    def merge(self, kind: str, d: dict, items: dict) -> None:
        if kind == "issues":
            items[d["number"]] = self.item(d)
            return
        key = "comments" if kind == "comments" else "review_comments"
        url = d["issue_url"] if kind == "comments" else d["pull_request_url"]
        n = int(url.rsplit("/", 1)[1])
        if n not in items:
            # comment of an item that the item pass has not stored: skip, a later sync adds it
            if (it := self.load(str(n))) is None:
                return
            items[n] = it
        items[n][key][str(d["id"])] = (
            comment if kind == "comments" else review_comment
        )(d)

    # commands
    def sync(self) -> int:
        """Sync all issues, pull requests and their comments; return the exit code."""
        state_file = self.root / "state.json"
        state = json.loads(state_file.read_text()) if state_file.exists() else {}
        status = 0
        for kind, path in (
            ("issues", "issues?state=all"),
            ("comments", "issues/comments?"),
            ("review_comments", "pulls/comments?"),
        ):
            st = state.setdefault(kind, {})
            url = st.get("next")
            if not url:
                st["started"] = now()
                url = self.url(f"{path}&sort=updated&direction=asc&per_page=100")
                if st.get("since"):
                    url += f"&since={st['since']}"
            try:
                while url:
                    data, url = request(url)
                    items = {}
                    for d in data:
                        self.merge(kind, d, items)
                    for n, d in items.items():
                        self.save(str(n), d)
                    st["next"] = url
                    self.write(state_file, json.dumps(state, indent=1))
                st["since"] = st.pop("started")
                st.pop("next", None)
                self.write(state_file, json.dumps(state, indent=1))
            except RateLimited as e:
                print(f"rate limit reached; run sync again after {e}", file=sys.stderr)
                status = 3
                break
        if TOKEN and not status:
            status = self.sync_discussions(state, state_file)
        elif not TOKEN:
            print(
                "discussions skipped: they need GITHUB_TOKEN or GH_TOKEN",
                file=sys.stderr,
            )
        self.index()
        return status

    def detail(self, n: int, kind: str) -> dict:
        """Fetch issue or pull request `n` with its timeline and review comments."""
        d = self.item(request(self.url(f"issues/{n}"))[0])
        if d["kind"] != kind:
            raise SystemExit(
                f"#{n} is a {d['kind']}: use `{'pr' if d['kind'] == 'pull' else 'issue'} {n}`"
            )
        d["comments"], d["events"] = {}, []
        for e in pages(self.url(f"issues/{n}/timeline?per_page=100")):
            ev = e.get("event")
            if ev == "commented":
                d["comments"][str(e["id"])] = comment(e)
            elif ev == "committed":
                d["events"].append(
                    {
                        "date": e["committer"]["date"],
                        "event": "commit",
                        "user": e["author"]["name"],
                        "sha": e["sha"][:9],
                        "text": e["message"].split("\n", 1)[0],
                    }
                )
            elif ev == "reviewed":
                body = clean(e.get("body"))
                if body or e["state"] != "commented":
                    d["events"].append(
                        {
                            "date": e["submitted_at"],
                            "event": f"review {e['state']}",
                            "user": user(e["user"])[0],
                            "text": body,
                        }
                    )
            elif ev in EVENTS:
                src = (e.get("source") or {}).get("issue") or {}
                d["events"].append(
                    {
                        "date": e["created_at"],
                        "event": EVENTS[ev],
                        "user": user(e.get("actor"))[0],
                        "sha": (e.get("commit_id") or "")[:9],
                        "text": f"#{src['number']} {src['title']}" if src else "",
                    }
                )
        if kind == "pull":
            d["review_comments"] = {
                str(c["id"]): review_comment(c)
                for c in pages(self.url(f"pulls/{n}/comments?per_page=100"))
            }
        d["detail_at"] = now()
        self.save(str(n), d)
        self.index()
        return d

    def show(self, n: int, kind: str, refresh: bool) -> str:
        d = self.load(str(n))
        stale = d is None or refresh or d.get("detail_at", "") < d["updated"]
        if stale:
            try:
                d = self.detail(n, kind)
            except RateLimited as e:
                if d is None:
                    raise SystemExit(f"rate limit reached; try again after {e}")
                print(
                    f"rate limit reached; showing stored copy (until {e})",
                    file=sys.stderr,
                )
        if d["kind"] != kind:
            raise SystemExit(
                f"#{n} is a {d['kind']}: use `{'pr' if d['kind'] == 'pull' else 'issue'} {n}`"
            )
        return self.md(d).read_text()

    # discussions (GraphQL needs a token)
    def graphql(self, query: str, **variables) -> dict:
        owner, name = self.repo.split("/")
        data, _ = request(
            f"{API}/graphql",
            {"query": query, "variables": dict(owner=owner, name=name, **variables)},
        )
        if "errors" in data:
            raise SystemExit(data["errors"][0]["message"])
        return data["data"]["repository"]

    def discussion(self, n: int) -> dict:
        q = """query($owner:String!,$name:String!,$n:Int!){repository(owner:$owner,name:$name){
discussion(number:$n){number title body createdAt updatedAt closed category{name} author{login}
answerChosenAt comments(first:100){nodes{author{login} body createdAt isAnswer
replies(first:100){nodes{author{login} body createdAt}}}}}}}"""
        g = self.graphql(q, n=n)["discussion"]
        d = {
            "n": n,
            "kind": "discussion",
            "title": g["title"],
            "state": "closed" if g["closed"] else "open",
            "author": (g["author"] or {}).get("login", "ghost"),
            "category": g["category"]["name"],
            "created": g["createdAt"],
            "updated": g["updatedAt"],
            "answered": g["answerChosenAt"],
            "body": clean(g["body"]),
            "comments": [
                {
                    "user": (c["author"] or {}).get("login", "ghost"),
                    "created": c["createdAt"],
                    "answer": c["isAnswer"],
                    "body": clean(c["body"]),
                    "replies": [
                        {
                            "user": (r["author"] or {}).get("login", "ghost"),
                            "created": r["createdAt"],
                            "body": clean(r["body"]),
                        }
                        for r in c["replies"]["nodes"]
                    ],
                }
                for c in g["comments"]["nodes"]
            ],
        }
        self.save(f"d{n}", d)
        return d

    def sync_discussions(self, state: dict, state_file: Path) -> int:
        q = """query($owner:String!,$name:String!,$after:String){repository(owner:$owner,name:$name){
discussions(first:100,after:$after,orderBy:{field:UPDATED_AT,direction:DESC}){
pageInfo{hasNextPage endCursor} nodes{number updatedAt}}}}"""
        st = state.setdefault("discussions", {})
        started, after = now(), None
        try:
            while True:
                g = self.graphql(q, after=after)["discussions"]
                for node in g["nodes"]:
                    if node["updatedAt"] < st.get("since", ""):
                        break
                    self.discussion(node["number"])
                else:
                    if g["pageInfo"]["hasNextPage"]:
                        after = g["pageInfo"]["endCursor"]
                        continue
                break
        except RateLimited as e:
            print(f"rate limit reached; run sync again after {e}", file=sys.stderr)
            return 3
        st["since"] = started
        self.write(state_file, json.dumps(state, indent=1))
        return 0

    def show_discussion(self, n: int, refresh: bool) -> str:
        if refresh or self.load(f"d{n}") is None:
            if not TOKEN:
                raise SystemExit("discussions need GITHUB_TOKEN or GH_TOKEN")
            self.discussion(n)
        return (self.root / "discussions" / f"{n}.md").read_text()


def comment(d: dict) -> dict:
    login, bot = user(d["user"] if "user" in d else d.get("actor"))
    return {
        "user": login,
        "bot": bot,
        "created": d["created_at"],
        "body": clean(d["body"]),
    }


def review_comment(d: dict) -> dict:
    c = comment(d)
    c.update(
        path=d["path"],
        line=d.get("line") or d.get("original_line"),
        outdated=d.get("position") is None,
        commit=d["original_commit_id"][:9],
        reply_to=d.get("in_reply_to_id"),
        hunk="\n".join(d.get("diff_hunk", "").splitlines()[-3:]),
    )
    return c


def entry(head: str, c: dict) -> str:
    if c.get("bot"):
        # bot reports (coverage, CI) are long and rarely needed: first line only
        return f"{head}\n{c['body'].split(chr(10), 1)[0].lstrip('# ')[:100]}"
    return f"{head}\n{c['body']}" if c["body"] else head


def render(d: dict) -> str:
    """Render an issue or pull request as one chronological text."""
    kind = "PR" if d["kind"] == "pull" else "Issue"
    state = "merged " + date(d["merged"]) if d.get("merged") else d["state"]
    if d["state"] == "closed" and not d.get("merged"):
        state += f" {date(d['closed'])} ({d['reason'] or 'closed'})"
    meta = [f"@{d['author']}", f"created {date(d['created'])}", state]
    if d.get("draft"):
        meta.append("draft")
    if d["labels"]:
        meta.append("labels: " + ", ".join(d["labels"]))
    events = d["events"] or []
    commits = [e for e in events if e["event"] == "commit"]
    if commits:
        meta.append(f"last commit {date(commits[-1]['date'])} {commits[-1]['sha']}")
    top = f"# {kind} #{d['n']}: {d['title']}\n{' | '.join(meta)}\n\n{d['body']}"
    out = []

    for c in d["comments"].values():
        out.append((c["created"], entry(f"## {date(c['created'])} @{c['user']}", c)))
    for e in events:
        if e["event"] == "closed" and d.get("merged"):
            continue
        line = f"{date(e['date'])} {e['event']}"
        line += f" {e['sha']}" if e.get("sha") else ""
        line += f" @{e['user']}"
        line += f": {e['text']}" if e["text"] else ""
        if e["event"].startswith("review"):
            out.append((e["date"], "## " + line))
        else:
            out.append((e["date"], "- " + line.split("\n", 1)[0]))

    # review comments: one block per thread, at the date of its first comment
    rcs = sorted(d["review_comments"].values(), key=lambda c: c["created"])
    by_id = {int(k): c for k, c in d["review_comments"].items()}
    threads = {}
    for c in rcs:
        root = c
        while root.get("reply_to") in by_id:
            root = by_id[root["reply_to"]]
        threads.setdefault(id(root), []).append(c)
    for thread in threads.values():
        root = thread[0]
        head = f"## {date(root['created'])} @{root['user']} on {root['path']}:{root['line']} @{root['commit']}"
        if root["outdated"]:
            later = [e for e in commits if e["date"] > root["created"]]
            head += (
                f" [outdated: changed by {later[0]['sha']} {date(later[0]['date'])}]"
                if later
                else " [outdated]"
            )
        hunk = "\n".join("> " + s for s in root["hunk"].splitlines())
        text = [entry(f"{head}\n{hunk}", root)]
        text += [
            entry(f"### {date(c['created'])} @{c['user']} reply", c) for c in thread[1:]
        ]
        out.append((root["created"], "\n".join(text)))

    out.sort(key=lambda t: t[0])
    if d.get("events") is None:
        out.append(
            (
                "",
                f"(no timeline: run `python tools/gh.py {'pr' if d['kind'] == 'pull' else 'issue'} {d['n']}`)",
            )
        )
    return "\n\n".join([top] + [t for _, t in out]) + "\n"


def render_discussion(d: dict) -> str:
    state = d["state"] + (f", answered {date(d['answered'])}" if d["answered"] else "")
    out = [
        f"# Discussion #{d['n']}: {d['title']}\n"
        f"@{d['author']} | created {date(d['created'])} | {d['category']} | {state}\n\n{d['body']}"
    ]
    for c in d["comments"]:
        mark = " [answer]" if c["answer"] else ""
        out.append(f"## {date(c['created'])} @{c['user']}{mark}\n{c['body']}")
        out += [
            f"### {date(r['created'])} @{r['user']} reply\n{r['body']}"
            for r in c["replies"]
        ]
    return "\n\n".join(out) + "\n"


def main() -> int:
    common = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent,
    ).stdout.strip()
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--repo", default="zerothi/sisl")
    p.add_argument(
        "--out", type=Path, default=Path(common).parent / "gh", help="archive folder"
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sync")
    for cmd in ("issue", "pr", "discussion"):
        s = sub.add_parser(cmd)
        s.add_argument("number", type=int)
        s.add_argument("--refresh", action="store_true", help="fetch again")
    args = p.parse_args()

    a = Archive(args.out, args.repo)
    if args.cmd == "sync":
        return a.sync()
    if args.cmd == "discussion":
        print(a.show_discussion(args.number, args.refresh), end="")
    else:
        print(
            a.show(args.number, "pull" if args.cmd == "pr" else "issue", args.refresh),
            end="",
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
