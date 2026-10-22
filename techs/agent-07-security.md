# agent-07 · Security

This chapter is for agents only. Human developers can skip it.
It gives the security rules for agents, when they develop code and when they review it.
Read [agent · Welcome](agent-welcome.md) and [agent-01 · Harness](agent-01-harness.md) first.

Follow [dev-10 · Security](dev-10-security.md). This chapter adds the rules for agents.

## No URL lookups

Do not add code below `src/` that reads from a URL. See [dev-10 · No URL lookups](dev-10-security.md#no-url-lookups).
Only a human can make this change.
If a task needs a URL lookup, stop and tell the maintainer.

## Code that runs automatically

Ask the maintainer before you change a file in
[dev-10 · Code that runs automatically](dev-10-security.md#code-that-runs-automatically).
See [agent-01 · Ask first](agent-01-harness.md#ask-first).

## Check your own commits

Before you report your work, look for the
[dev-10 · Signs of a trojan horse](dev-10-security.md#signs-of-a-trojan-horse) in your own commits.
Tell in the report why your code needs each sign that it has.
