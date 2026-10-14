# 34 · Geometry factories

Owns: `src/sisl/geom/*`

Paths are relative to `src/sisl/geom/`.

The `sisl.geom` package makes standard geometries.
It also holds the `NeighborFinder` and the atom categories.

## Factories

| Group | Functions | File |
|---|---|---|
| Bulk | `sc`, `bcc`, `fcc`, `hcp`, `rocksalt` | `basic.py` |
| Bulk | `diamond` | `special.py` |
| 2-D | `honeycomb`, `graphene`, `hexagonal`, `goldene`, flakes, `triangulene` | `flat.py` |
| 2-D | `bilayer` | `bilayer.py` |
| 1-D | `nanoribbon`, `agnr`, `zgnr`, `cgnr`, `heteroribbon` | `nanoribbon.py` |
| 1-D | `nanotube` | `nanotube.py` |
| Surfaces | `fcc_slab`, `bcc_slab`, `rocksalt_slab` | `surfaces.py` |

## Conventions

All factories follow these rules. A new factory must follow them too.

- Lengths are in Å.
- Bulk factories take the lattice constant `alat` (`hcp`: `a` and `coa`).
- 2-D, 1-D and flake factories take `bond`: the nearest-neighbour distance. It is not the lattice constant.
- A generic factory requires `atoms` (`honeycomb`, `nanoribbon`).
  A material factory (`graphene`, `agnr`) has a default `bond`.
  `atoms=None` gives `Atom(Z=6, R=bond * 1.01)`. Thus the neighbours are inside `maxR`.
- `atoms` is `AtomsLike`: a name, a `Z`, an `Atom` or a list.
- `orthogonal=False` gives the primitive cell. `True` gives the orthogonal cell.
- `vacuum=20.` is added along the directions that are not periodic.
- The function returns a new `Geometry`.
- Use `@set_module("sisl.geom")` on each public function.

### Supercell and boundary conditions

Each factory ends with `geometry_define_nsc(geometry, periodic)` (`_common.py`):

- A periodic direction gets `nsc = 3` and `Periodic`.
- Other directions get `nsc = 1` and `Unknown`.
- If all atoms have `maxR > 0`, `find_nsc` makes `nsc` as small as possible.

| Kind | `periodic` |
|---|---|
| Bulk | `(T, T, T)` |
| 2-D | `(T, T, F)` |
| Ribbon | `(T, F, F)` |
| Tube | `(F, F, T)` |
| Flake | `(F, F, F)` |

## Composite geometries

`_composite.py` builds a geometry from sections.

- `CompositeGeometrySection` is an abstract dataclass with `build_section(previous)` and `add_section(geom, addition)`.
- `composite_geometry(sections, section_cls, **kwargs)` accepts each section as a `Geometry`, a scalar, a tuple, a `dict` or a section object.
- Each section gets the previous section object. Thus it can check the junction.
- The wrapper function has the section class as an attribute (`heteroribbon.section`).

To add a composite builder:

1. Subclass `CompositeGeometrySection` as a `@dataclass`.
2. Implement `build_section` and `add_section`.
3. Write a function that calls `composite_geometry`. Set `function.section = YourSection`.

See `heteroribbon` in `nanoribbon.py`.

## NeighborFinder

`_neighbors/_finder.py` finds neighbours with cell lists.

- The bins are along the lattice vectors.
  The bin thickness perpendicular to each face is at least `2 * maxR`.
  Thus each atom searches only 8 bins (2 for each axis).
- If `R` is larger than the cell, the geometry is tiled with odd `nsc`.
- Atoms outside the cell (fractional coordinates outside `[0, 1)`) are rejected.
- `_neighbors/_operations.py` is Cython pure-Python mode. It is compiled by `_neighbors/CMakeLists.txt`.
- The results are `(n, 5)` `int64` arrays `[I, J, isc_x, isc_y, isc_z]`.
  The buffer grows by `grow_factor`. The start size is from `memory=("200MB", 1.5)`.

| Method | Result |
|---|---|
| `find_neighbors()` | `FullNeighborList` (or `PartialNeighborList` for some atoms) |
| `find_unique_pairs()` | `UniqueNeighborList` (`I < J`) |
| `find_close(xyz)` | `CoordsNeighborList` |

## Atom categories

`_category/` holds the atom categories. See [30 · Categories](30-geometry.md#categories) for the system.

| File | Categories |
|---|---|
| `_kind.py` | `AtomZ`, `AtomIndex`, `AtomSeq`, `AtomTag`, `AtomOdd`, `AtomEven` |
| `_coord.py` | `AtomFracSite`, `AtomXYZ` (and generated `Atomx`, `Atomfx`, ... with `<` and `>`) |
| `_neighbors.py` | `AtomNeighbors(min, max, neighbor=, R=)` |

`AtomNeighbors` counts the neighbors in `(lo, hi]` of each atom.
The default `R` is `(0.01, maxR)`. Thus the atom itself is not counted.
A single radius gives `lo = -1`, and the atom itself is counted.

`base._neighbor_pairs` gives all pairs in one `NeighborFinder` search.
The result is equal to `Geometry.close` for each atom:

- The search uses a copy with all atoms in the unit cell and `pbc` in all directions.
  `NeighborFinder` does not accept atoms outside the unit cell.
- The `isc` of the pairs goes back to the supercells of the input geometry.
  Pairs outside `nsc` are removed, as in `close`.
  Thus `close` and the categories ignore `pbc`.
- The `cache` of one `mask` call keeps the pairs of the largest `R`.
  A smaller `R` filters these pairs.

`test_geom_category_mask.py` compares the masks with `close` for each atom.
It also makes sure that a composite does one search and does not call `close`.

## Saturation

`_saturate.py` holds `saturate(geometry, rules)` and `Saturate`.

- `rules` is a sequence of `(selector, saturation)` pairs, not a `dict`.
  The categories define `__eq__`, thus they are not hashable.
- The first rule that selects an atom is used.
  All rules select on the input geometry, with one `cache`.
  Thus the rules share one neighbor search, and the added atoms do not change a selection.
- The added atoms are at the end, in the order of the rules.
  Thus the indices of the input atoms do not change.
- `saturate` is a function in `sisl.geom`, not a `Geometry` method.
  `Geometry.apply` has a different meaning.
  We can register it with `register_sisl_dispatch` when the API is stable.
- `Saturate` uses the `R` of `AtomNeighbors` to find the bonds.
- The default `coordination` is the most frequent number of bonds of the same `Z`.
  For a tie, it is the largest number.
- The default `bond` is the sum of `PeriodicTable.radius` (`"calc"`).
  This is approximate (C-H 1.20 Ang, not 1.09 Ang). The doc-string tells the user.
- The directions of the missing bonds (`_bond_directions`):
  - One missing bond: the negative sum of the present bonds.
  - Coordination 4, two bonds: the tetrahedral pair in the plane of the bisector.
  - One bond: the new bonds are in the plane of a bond of the neighbor.
    They are planar for coordination 3, and staggered for coordination 4.
  - Other cases raise `ValueError`.
- An atom without bonds gets a warning, and no new atoms.
- The loop over the selected atoms is in Python.
  Usually only edge atoms are selected, thus the loop is short.

## Add a factory

1. Write the function in the file of its group. Follow the [conventions](#conventions).
2. End it with `geometry_define_nsc(g, periodic)`.
3. Add the name to `__all__`. Import a new module in `geom/__init__.py`.
4. Add it to `docs/api/geom/building.rst`.
5. Add tests in `geom/tests/test_geom.py` (`pytest.mark.geom`).
   Use parameters for `orthogonal` and `vacuum`. Check `is_right_handed` and `has_vacuum`.
