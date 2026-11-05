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
| `models` | Tight-binding model registry (`BaseModel`, `ModelDispatcher`). `GrapheneHamiltonian` with published parameter sets. `SlaterKosterHamiltonian` (`slater_koster.H`) with Slater-Koster references. | [Slater-Koster models](#slater-koster-models) |

## Slater-Koster models

`models/_slater_koster.py` has `SlaterKosterHamiltonian`. Its references (`slater_koster.H.ref.<name>`)
return a `Hamiltonian` made with `sisl.SlaterKoster`.
One class holds all references, also for different elements ([overview-01 · Few strong classes](overview-01-concepts.md#few-strong-classes)).

- A user `geometry=` gets the atoms of the model by atomic number. An atomic number that the model
  does not have raises `ValueError`. A missing element would otherwise give a Hamiltonian without its rows.
- `_hamiltonian` copies the lattice of the user geometry, and gives `nnzpr` from the neighbor count of the geometry (`_neighbor_pairs`).
  A row of `SparseCSR` that grows past `nnzpr` is slow: 63 s against 2.2 s for 16000 Si atoms (Vogl1983).
  The maintainer decided to keep the growth of `SparseCSR`, and to give a larger `nnzpr`.

### Vogl1983

- The table is Table 1 of the paper (cubic parameters). The conversion to two-center integrals uses
  the 4 neighbors along `(±1, ±1, ±1)/√3`: `V_ssσ = V(s,s)/4`, `V_ppσ = (V(x,x) + 2 V(x,y))/4`,
  `V_ppπ = (V(x,x) - V(x,y))/4`, `V_spσ = √3/4 V(sa,pc)`.
- Table A has `⟨p_a|H|s_c⟩ = -V(pa,sc) g1`. Thus `V(sc,pa)` is the s of the cation at the origin and the p of the anion along `+z`:
  the key `(cation, anion), (s, p)`. The same holds for `s*`.
- The `s*` orbital is the `(n+1)s` shell, with `n` the period of the element.
- The `s*`-`s` and `s*`-`s*` integrals are 0 in the paper (note 15). They are given as 0, thus `SlaterKoster` does not warn.
- The range is the middle between the first and the second neighbor of the bond length of Table 2.
- ZnTe `E(s,c)` is `-0.9350`. The scan reads `0.9350`, but the sum rule `E(s,a) + E(s,c) = Γ1v + Γ1c`
  (eq. 10) and `V(s,s)` of Table 1 (eq. 9) both give `-0.9350`. All other rows obey both rules.
- The test builds the Bloch matrix of Table A from the cubic parameters, without `SlaterKoster`, and compares the eigenvalues.
- Table 2 is the fit target, not the result of the model. GaAs hits `X1c`, `X3c`. Si does not (`X1c` is 1.63 eV, not 1.13 eV):
  the Si conduction minimum is 1.171 eV at 0.731 (2π/a) on Δ, the same as Klimeck et al. 2000.
  Klimeck gives 2.160 eV at L; the model here gives 2.496 eV. This difference is not explained.

Sources:
DOI[10]: 10.1016/0022-3697(83)90064-1
URL[8]: https://alevi.usc.edu/files/2018/05/Vogl1983-10syajv.pdf
URL[6]: https://engineering.purdue.edu/gekcogrp/publications/pubs_src/J_2000_1.pdf

### Xu1992

- The paper is behind a paywall. The values come from two copies that cite it and agree:
  the XTANT-3 input files (all values) and Amara et al. 2009 (all values except the tail).
  The tail coefficients are only in XTANT-3. They continue the scaling function with its value and slope at `r1`
  (to 1e-8) and are 0 at `rm`. `test_xu1992` checks this.
- All four integrals use one scaling function: `V(r) = V(r0) s(r)`, `s(r0) = 1`. `SlaterKoster` sets `V = 0` above `rm = 2.6` Ang.
- Only the Hamiltonian. The repulsive energy of the paper is for total energies, not for `H`.
- The sign of `V_spσ` is not stated in the copies. For one element it has no effect:
  a sign change of all p orbitals changes the sign of all s-p elements and keeps the eigenvalues.
- No band energies of the paper are available. The default diamond gives a valence width of 23.7 eV and a gap of 6.4 eV.

Sources:
DOI[10]: 10.1088/0953-8984/4/28/006
URL[9]: https://raw.githubusercontent.com/N-Medvedev/XTANT-3/main/INPUT_DATA/C/Pettifor/C_C_TB_Hamiltonian_parameters.txt
URL[7]: https://arxiv.org/pdf/0810.1008
URL[5]: https://raw.githubusercontent.com/N-Medvedev/XTANT-3/main/Source_files/TB_Pettifor.f90
