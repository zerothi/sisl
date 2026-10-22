# dev-04 · Conventions

Owns: `.pre-commit-config.yaml`, `.LICENSE_header`, `tools/add_header.sh`, `changes/README.rst`, `src/sisl/typing/*`, `src/sisl/messages.py`

## License header

Each source file starts with the MPL-2.0 header.
The text is in `.LICENSE_header`. The script `add_header.sh` adds it.

## Typing

- Each module starts with `from __future__ import annotations`.
- Type aliases are in `src/sisl/typing`. Use them. Do not define the same alias again.

## Dependencies

sisl ships wheels to computational resources. Each required dependency makes the install harder.

- Do not add a required dependency without the approval of the maintainer.
- Import an optional dependency in a `try`/`except ImportError` block.
  Fail with a clear message only when the user calls the feature that needs it.
  Examples: `pathos` in `physics/_brillouinzone_apply.py`, the `ase` conversions in `_core/geometry.py`.

## Style

pre-commit enforces the style. Install it one time:

```bash
pre-commit install
```

Do not format code by hand. pre-commit formats it when you commit.

## Changelog

A pull request that changes what users see must have a news fragment.
towncrier assembles the fragments into the release notes.

- Put the fragment in `changes/`. Put toolbox fragments in `changes/toolbox/`.
- Name it `<PR-number>.<type>.rst`.
- Types: `feat`, `fix`, `change`, `highlight`, `doc`, `dev`.
- See `changes/README.rst` for the format.

## Public and private names

- A name that starts with `_` is private. You can change it without a deprecation.
- A public name needs a deprecation before you remove or change it.
  Use `deprecate`, `deprecate_argument` or `deprecation` from `src/sisl/messages.py`.
  Give `from_version` and `remove_version`.
  When the name goes away is in [release-04 · Deprecations](release-04-deprecations.md).
