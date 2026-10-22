# dev-10 · Security

This chapter gives the rules that keep the code and the history of sisl safe.
They apply when you develop code and when you review it.

sisl ships wheels that users install on computational resources.
Code in sisl runs with the rights of these users. Thus the code and each commit must be safe.

## No URL lookups

Do not add code below `src/` that reads from a URL.
This rule applies to all files below `src/`, also to the tests there.
Examples are `urllib`, `requests`, `httpx`, `pooch`, or a call to `curl` or `wget`.

Only the maintainer can approve such a change.

- sisl runs on computational resources. Many compute nodes have no internet access.
- A result must not change when a remote file changes.
- Data from a URL is not trusted.

## Do not run the content of a file

sisl reads files from many sources. A file that sisl reads is data, not code.

- Do not use `eval`, `exec` or `compile` on text from a file.
- Do not add `pickle` or `marshal` loads for a file format.
- Parse the values that the format defines. Raise a clear error for all other content.

## Dependencies and their sources

- Do not add a required dependency without the approval of the maintainer.
  See [dev-04 · Dependencies](dev-04-conventions.md#dependencies).
- Do not add a package index, a dependency from a git URL, or a dependency from a local path.
- Pin the version of each pre-commit hook with `rev`, as the other hooks do.

## Code that runs automatically

Some files run without a user call. A change to them needs the approval of the maintainer:

- `.github/workflows/`: the CI runs them with access to the repository.
- `.pre-commit-config.yaml` and the scripts in `tools/` that its hooks run.
- Entry points and build hooks in `pyproject.toml`.

A change to these files can run on the computer of each developer, or in the CI.
See [release-03 · CI](release-03-ci.md).

## No secrets and no local data

Do not commit:

- Tokens, keys, passwords or other credentials.
- Absolute paths, user names or host names of your computer, for example `/home/<user>/...`.
- Output that contains these, for example a log or a profile.

## Signs of a trojan horse

A trojan horse is code that does something different from what it tells.
Look for these signs in each commit that you make or review.
[dev-09 · No trojan horses](dev-09-review.md#no-trojan-horses) tells how to examine each commit of a branch.


- **Network access.** A URL, a socket, or a call to `curl` or `wget`.
  See [No URL lookups](#no-url-lookups).
- **Dynamic code.** `eval`, `exec`, `compile`, `__import__`, `importlib` with a computed name,
  `pickle` or `marshal` loads, `ctypes`, `subprocess`, `os.system`.
- **Hidden content.** Base64, hex or compressed strings that are decoded at run time.
  Hidden or bidirectional Unicode characters. Names that look like other names.
- **Binary files.** A binary file that is not test data, or test data that is not necessary.
- **Code that runs automatically.** Changes to `conftest.py`, `__init__.py` imports, `.pth` files,
  entry points in `pyproject.toml`, `CMakeLists.txt`, `.pre-commit-config.yaml`, `tools/`,
  `.github/workflows/` or `.gitattributes`.
- **Changes to the tree.** New submodules, symbolic links, and files that become executable.
- **Code that hides itself.** A commit that adds code and a later commit that removes it.
  A merge commit with changes that no parent has.
- **Changes to the sources of dependencies.** A new package index, a new repository in a hook,
  or a version that is not pinned where other versions are pinned.

A sign is not always an attack. The code can need it.
Then the commit message tells why.
