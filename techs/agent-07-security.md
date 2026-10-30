# agent-07 · Security

For agents, when they develop code and when they review it.
Extends [dev-10 · Security](dev-10-security.md).

## No URL lookups

Do not add code below `src/` that reads from a URL ([dev-10 · No URL lookups](dev-10-security.md#no-url-lookups)).
Only a human can make this change. If a task needs a URL lookup, stop and tell the maintainer.

## Code that runs automatically

Ask the maintainer before you change a file in
[dev-10 · Code that runs automatically](dev-10-security.md#code-that-runs-automatically).

## Check your own commits

Before you report your work, look for the
[dev-10 · Signs of a trojan horse](dev-10-security.md#signs-of-a-trojan-horse) in your own commits.
Tell in the report why your code needs each sign that it has.
