# 15 · Public documentation

Owns: `docs/*`

`docs/` is the public guide of sisl. Sphinx makes it from reStructuredText (RST), notebooks and doc-strings.
The readers are end-users (master and PhD students) and agents.
This chapter tells how to write in `docs/`. For the decision *when* to write there, see [README · Two kinds of documentation](README.md#two-kinds-of-documentation).

## Purpose

- Each page has one clear purpose. State it in the first paragraph.
- Show explicit usage. A short example that runs is better than a long description.
- Keep pages lean. Do not explain internal decisions. Put them in `techs/`.
- Document only public API. Do not show names that start with `_`.

## Structure

The toctrees in `docs/index.rst` set the sections. Put a new page in the section of its reader:

| Section | Folder or file | Content |
|---|---|---|
| Getting started | `introduction.rst`, `quickstart/`, `cite.rst` | Installation, overview, first steps. |
| User guide | `tutorials/`, `visualization/`, `scripts/` | Tutorials and how-to pages. Command line scripts. |
| Advanced usage | `api/`, `environment.rst`, `distribution.rst`, `toolbox/` | API reference, environment variables, toolbox. |
| Development | `dev/` | How to contribute, build and test. |
| Extras | `conventions.rst`, `release.rst`, `other.rst`, `references.rst` | Units, notation, release notes, references. |

- API pages in `api/` list the public names with `autosummary` and `:toctree: generated/`.
  Do not edit `api/generated/`. Sphinx writes it.
- Release notes come from fragments in `changes/`. See [13 · Conventions](13-conventions.md#changelog).
- Add a new page to a toctree. Else Sphinx warns and the reader cannot find the page.

## Writing style

`docs/` does not use ASD-STE100. Use clear, correct English for a scientific reader:

- Use short sentences and the active voice.
- Use one term for one concept. Use the names from the code (`Geometry`, `nsc`, `Hamiltonian`).
- Write sisl in lower case, also at the start of a sentence.
- Use the units and the mathematical notation of `docs/conventions.rst`:
  lengths in Ångström, energies in eV, bold vectors and matrices, `I, J` for atoms and `i, j` for orbitals.
- Add a reference to the literature with `sphinxcontrib.bibtex` and `docs/references.bib`.

## Doc-strings

Doc-strings are part of `docs/`. Sphinx reads them for the API reference.
They stay in the source files. See [02 · Documentation is in the file](02-concepts.md#documentation-is-in-the-file).

- Use the numpy style. `docs/conf.py` sets `napoleon_numpy_docstring = True`.
- Write the first line as a short summary.
- Do not write the types in the `Parameters` section. Write them as type hints.
  `autodoc_typehints = "description"` adds them to the description.
- Use the aliases from `sisl.typing` in the type hints. `docs/conf.py` links them.
- The numpydoc checks are on, except for `Examples` (`EX01`), `See Also` (`SA01`) and `Extended Summary` (`ES01`).
  Thus these sections are optional. Add `Examples` for each main entry point.

## RST markup

- Use these heading characters, in this order: `=` (page title), `-` (section), `~` (subsection).
  Do not use overlines.
  Exception: the pages in `api/` and `release.rst` use a `*` overline for the title, then `=` and `-`.
  The release notes and `changelog/` files come from templates. Do not change their headings.
- Start each page with a label (`.. _tutorial-01:`). Link to it with `:ref:`.
- Link to the API with roles: `` :class:`~sisl.Geometry` ``, `` :meth:`~sisl.Geometry.tile` ``, `` :func:`~sisl.geom.graphene` ``.
  The `~` shows only the last part of the name.
- Use double backticks for literal code: ``` ``nsc`` ```.
- Single backticks are a Python cross-reference. `docs/conf.py` sets `default_role = "py:obj"`.
  Thus `` `sisl.Geometry` `` is a link, and `` `~sisl.Geometry` `` shows only `Geometry`.
  A short name (`` `Geometry` ``) is a link only on pages that set `.. module::` or `.. currentmodule::`.
  A name that Sphinx cannot find shows as italic text, without a warning.
- Use single backticks for parameter names in doc-strings (numpydoc convention). They show as italic text.
- Before `default_role` was set, all single backticks showed as italic text and made no links.
  The change added no Sphinx warnings.
- Link to GitHub with `` :issue:`42` ``, `` :pull:`42` `` and `` :discussion:`42` ``. Link to a DOI with `` :doi:`...` ``.
- Use `.. code:: bash` for shell commands.
- Use `.. tab::` (`sphinx_inline_tabs`) for alternatives, for example SSH and HTTPS.
- Use the `sphinx_design` grid cards only on landing pages.
- Use `.. math::` and `:math:` for formulas.

## Examples

- Write examples as doctests (`>>>`). Show the output after the call.
- Use small systems, for example `sisl.geom.graphene()`. An example must run in seconds.
- pytest does not run doctests, and the CI does not run them.
  Run each new or changed example by hand.
- A notebook in `tutorials/` or `visualization/` runs when Sphinx builds.
  It must run without external files that are not in the `files` submodule.

## Plotly figures in notebooks

`PLOTLY_RENDERER` selects the renderer (default `png` in `docs/conf.py`).
The `png` renderer uses kaleido. Kaleido uses a headless chrome.

- Without a kaleido server, each figure starts a new chrome (approximately 2 s).
- Each new chrome loads MathJax from a CDN. A dropped connection stops the page load for minutes.
- The kaleido `timeout` is for the render only. It is not for the page load.

Thus, `docs/conf.py` starts one kaleido server in each notebook kernel
(`nbsphinx_execute_arguments`, `IPKernelApp.exec_lines`).
A server in the Sphinx process is not used: the kernels are other processes.

- `.readthedocs-pre_install.sh` keeps MathJax 3.2.2 in `<venv>/share/kaleido/tex-svg.min.js`.
- Give MathJax as `(file_uri, "utf-8")`. Without the charset, LaTeX text is not shown, and no error occurs.
- If the file is missing, MathJax is off (`mathjax=False`). LaTeX text then shows as raw `$...$`.
  Do not use the CDN: it can stop the build.

The `notebook_connected` renderer does not use chrome.
It links plotly.js from the CDN. Each figure keeps only its data (approximately 10-40 KiB).
Do not use the `notebook` renderer: it puts a copy of plotly.js (approximately 4.7 MB) in each page.

## Build and check

The steps are in `docs/dev/index.rst` (Contribute to the docs). In short:

```bash
pip install -e . --group docs
git submodule update --init --single-branch
export _SISL_DOC_SKIP=notebook   # optional: skip notebooks for a faster build
make -C docs
```

- Build the pages that you changed. Open them in `docs/build/html/`.
- Correct all Sphinx warnings that come from your change.
- Check the links and the rendered math.
