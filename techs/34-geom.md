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

`AtomNeighbors` uses `Geometry.close` with `R = (0.01, maxR)`. Thus the atom itself is not counted.
It does not use `NeighborFinder`.

## Add a factory

1. Write the function in the file of its group. Follow the [conventions](#conventions).
2. End it with `geometry_define_nsc(g, periodic)`.
3. Add the name to `__all__`. Import a new module in `geom/__init__.py`.
4. Add it to `docs/api/geom/building.rst`.
5. Add tests in `geom/tests/test_geom.py` (`pytest.mark.geom`).
   Use parameters for `orthogonal` and `vacuum`. Check `is_right_handed` and `has_vacuum`.
