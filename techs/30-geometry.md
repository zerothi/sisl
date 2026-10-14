# 30 · Geometry

Owns: `src/sisl/_core/lattice.py`, `src/sisl/_core/geometry.py`, `src/sisl/_core/atom.py`, `src/sisl/_core/_ufuncs_lattice.py`, `src/sisl/_core/_ufuncs_geometry.py`, `src/sisl/_core/_ufuncs_atom.py`, `src/sisl/_dispatcher.py`, `src/sisl/_dispatch_class.py`, `src/sisl/_ufuncs.py`, `src/sisl/_namedindex.py`, `src/sisl/_category.py`, `src/sisl/_core/_lattice.pyx`, `src/sisl/_core/periodictable.py`, `src/sisl/_core/quaternion.py`

Paths are relative to `src/sisl/`.

## Classes

| Class | File | Content |
|---|---|---|
| `Lattice` | `_core/lattice.py` | Cell, origin, `nsc`, `sc_off`, `isc_off`, boundary conditions. |
| `LatticeChild` | `_core/lattice.py` | Mixin. Sends cell, volume, `n_s` and similar attributes to `self.lattice`. |
| `Geometry` | `_core/geometry.py` | `xyz` (shape `(na, 3)`), `Atoms`, `NamedIndex`, `Lattice`. |
| `Atom` | `_core/atom.py` | `Z`, mass, tag and a list of `Orbital`. |
| `Atoms` | `_core/atom.py` | Unique species and one species index for each atom. |
| `Orbital` | `_core/orbital.py` | Range `R` and an optional radial function. See [31 · Orbitals](31-orbitals.md). |
| `Grid` | `_core/grid.py` | Values on a grid. It has an optional `Geometry`. See [32 · Grid](32-grid.md). |

- `Lattice` and `Orbital` use `__slots__`. You cannot add attributes at run time.
- `SuperCell` is a deprecated name for `Lattice`.
- `set_lattice` always uses `Lattice.new(...)`.
- Without a lattice, `Geometry` estimates one from the nearest neighbour distances.
- An `Atom` without orbitals gets `[Orbital(-1.0)]`.
- `Atoms._species` is `int16`. Thus the maximum is 32767 unique species.
- `Atoms` finds equal species with `Atom.__eq__` (Z, orbitals, mass, tag).
- After a change of species or orbitals, call `Atoms._update_orbitals()`. It updates `_firsto`.
- `Orbital(R=None)` gives `R = -0.9999`. A negative `R` means "no range",
  or `{"contains": -R}` when a radial function exists.
- `Grid.set_geometry(also_lattice=True)` replaces the lattice of the grid with the lattice of the geometry.
  The boundary conditions of a grid are on its `Lattice`.

## Supercells

- Each `nsc[i]` must be odd. `0` changes to `1`.
- `sc_off[0]` is always `(0, 0, 0)`. This is the primary cell.
  Code that uses `isc == 0` for the unit cell depends on this.
- The other offsets are in the order z (outer), y, x (inner).
  Each axis goes over `range(-(n // 2), n // 2 + 1)`.
- `isc_off` has the shape `nsc`. You index it with the signed offsets.
  Negative offsets work because numpy wraps negative indices.
- `sc_index` accepts `None` for one component. `None` means "all".
- A supercell atom index is `isc * na + ia`. `asc2uc` is `% na`. `a2isc` uses `// na`.
  Orbitals use `no` in the same way.
- The Cartesian offset of a supercell is `isc @ cell`.
- To change `nsc` of a sparse matrix, use `set_nsc` on the matrix.
  See [20 · Supercell columns](20-sparse.md#supercell-columns).

## Boundary conditions

`Lattice._bc` has the shape `(3, 2)`: one value for each side of each axis.
The values are `BoundaryCondition`: `UNKNOWN`, `PERIODIC`, `DIRICHLET`, `NEUMANN`, `OPEN`.

- An axis is periodic on both sides or on no side. Other combinations raise `ValueError`.
- A non-periodic axis with `nsc > 1` gives a warning, not an error.
- `pbc` reads only the lower side.
- `pbc=False` on a periodic axis gives `UNKNOWN`. `False` does not identify one boundary condition.

## Conversion: `new` and `to`

`Lattice`, `Geometry`, `Grid` and `Atom` declare dispatchers in the class statement
(`_dispatcher.py`, `_dispatch_class.py`).

- `Cls.new(obj)` makes a sisl object from `obj`.
- `obj.to(target)` converts a sisl object to `target`.
- The key is a type or a string (for example `"ase"`).
  A type that is not registered falls back to its MRO. The result is then cached.
- `when_subclassing="copy"` gives a subclass its own copy of the registry.
  A registration on a subclass does not change the parent.
  `"keep"` shares the registry with the parent (`Atom`). `"new"` starts empty.

To add a conversion:

1. Subclass the base dispatch class of the module, for example `GeometryNewDispatch` or `GeometryToDispatch`.
2. Implement `dispatch()`. Use `self._get_class()` in `new`. Use `self._get_object()` in `to`.
3. Register the class with a type key and with a string key.
4. Put optional dependencies in `try`/`except ImportError`.

See the examples at the end of `_core/geometry.py`.

Note: when `new` makes a different class, it copies `xyz`, atoms and lattice. It does not copy the names.

## Methods: `register_sisl_dispatch`

Most operations on the core classes (`tile`, `repeat`, `rotate`, ...)
are functions in `_core/_ufuncs_<class>.py`.

```python
@register_sisl_dispatch(Geometry, module="sisl")
def tile(geometry: Geometry, reps: int, axis: int) -> Geometry:
    ...
```

- The decorator makes one `singledispatch` function for each name.
- It adds the function as a method on the class. It records the name in `cls._funcs`.
- It warns when it replaces an attribute that the class defines.
- `expose_registered_methods("sisl")` makes the function available as `sisl.tile`.
- For a type that is not registered, the function tries `cls.new(obj)`.
  Then it converts the result back with `.to(type(obj))`, unless `ret_sisl=True`.

Use this pattern for new operations on a core class.
Do not add a subclass for it. See [02 · Few strong classes](02-concepts.md#few-strong-classes).

The functions in `geom/` (`graphene`, `fcc`, ...) are factories.
They are not methods. See [34 · Geometry factories](34-geom.md).

## Named indices

`NamedIndex` (`_namedindex.py`) maps names to index arrays.

- A boolean mask changes to indices.
- A duplicate name raises an error.
- `index()` returns `None` for an unknown name. It returns other input without change.
  Thus `geom["name"]` and `geom[indices]` use the same code.

## Categories

`_category.py` holds the base classes. The atom categories are in `geom/_category/`.

- `AtomCategory` is a family class: `class AtomCategory(Category, family="Atom")`.
  A family class has no keys.
- A sub-class gets its keys from the class arguments `key=` and `aliases=`.
  Without `key=`, the key is the class name without the family prefix.
  `AtomNeighbors` gets `Neighbors`.
- `AtomCategory(Z=6)` works. `CategoryMeta.__call__` calls `cls.kw(**kwargs)`
  for a family class called with keywords only.
- `CategoryMeta` does not catch `TypeError`. An error in `__init__` is shown to the user.
  Before, it changed to a wrong "no match" error, and `AtomZ()` returned `None`.
- `kw` finds the sub-class with `is_class`: first case-sensitive, then case-insensitive.
  `Z=` selects `AtomZ`. `z=` selects the coordinate category.
  We keep the case-insensitive fallback because users write `xyz=` for `AtomXYZ`.
- `is_class` compares with the keys. A sub-class can override it.
- `kw` sends `True` as no arguments (`odd=True`).
  It sends a `dict` as keyword arguments, and other values as one argument.
- An unknown keyword gives the close matches (`difflib`) and the list of keys.
- `AtomZ` changes symbols and `Atom` objects to atomic numbers.
  An unknown symbol raises `ValueError`. Before, `AtomZ("C")` silently matched no atoms.
- Many keywords combine with `&`.
- "Not in the category" is `NullCategory`. It is equal to `None`.
- `&`, `|`, `^` and `~` make composite categories.
- Geometry indexing accepts a category or a `dict`.
- A `Shape` used as an index selects only atoms in the unit cell. See [33 · Shapes](33-shape.md).
