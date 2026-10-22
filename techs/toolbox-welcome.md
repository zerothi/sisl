# toolbox · Welcome

Owns: `src/sisl_toolbox/*`

Paths are relative to `src/sisl_toolbox/`.

## Namespace package

`sisl_toolbox` has no `__init__.py`. It is a PEP 420 namespace package.

- `pyproject.toml` lists it in `wheel.packages`. Thus the wheel contains it.
- Other distributions can add subpackages `sisl_toolbox.<name>`.
- Import a toolbox by its full name, for example `import sisl_toolbox.btd`.
  `sisl_toolbox` does no imports at the top level (see `README`).
- A toolbox is less stable than `sisl`. It can depend on more packages.

## CLI

The console script is `stoolbox` (`cli/_cli.py:stoolbox_cli`).

- A toolbox registers a sub-command with `register_toolbox_cli(setup)`.
  `setup` adds an argparse sub-parser and sets `runner=`.
- Registration happens at import. `cli/_cli_imports.py` imports the toolboxes that have a CLI.
  Add your toolbox there.
- Current commands: `atom-plot` (`siesta/atom`), `ts-fft` (`transiesta/poisson`).

## Tests and changelog

- Mark btd tests with `pytestmark = [pytest.mark.btd]`.
- Put toolbox news fragments in `changes/toolbox/`.

## Subpackages

| Subpackage | Purpose | Chapter |
|---|---|---|
| `btd` | Block-tridiagonal Green functions for N-electrode NEGF. | [toolbox-10 · BTD](toolbox-10-btd.md) |
| `negf` | `NEGF`, `Electrode`, `ChemicalPotential` from fdf/files. Work in progress (not in git). | — |
| `siesta.atom` | `AtomInput` for ATOM pseudopotential input. `atom-plot` CLI. | — |
| `siesta.minimizer` | Basis and pseudopotential optimization with `scipy.optimize`. YAML configuration. | — |
| `transiesta.poisson` | FFT Poisson correction for TranSiesta (`solve_poisson`, `ts-fft`). Uses pyamg. | — |
| `transiesta.bulk` | Stub. Not in git. | — |
| `models` | Tight-binding model registry (`BaseModel`, `ModelDispatcher`). `GrapheneHamiltonian` with published parameter sets. | — |
