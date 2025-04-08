# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from collections.abc import Callable
from functools import reduce
from numbers import Integral
from typing import Any, Literal, Optional, Protocol, Union

import numpy as np
import numpy.typing as npt

import sisl._array as _a
from sisl._ufuncs import register_sisl_dispatch
from sisl.messages import deprecate_argument, warn
from sisl.typing import (
    AnyAxes,
    AtomsIndex,
    CellAxis,
    Coord,
    CoordOrScalar,
    GeometryLike,
    LatticeOrGeometryLike,
    SileLike,
)
from sisl.utils import direction
from sisl.utils.mathematics import fnorm

from .geometry import Geometry
from .lattice import Lattice
from .quaternion import Quaternion

# Nothing gets exposed here
__all__ = []


@register_sisl_dispatch(Geometry, module="sisl")
def copy(geometry: Geometry) -> Geometry:
    """Create a new object with the same content (a copy)."""
    g = geometry.__class__(
        np.copy(geometry.xyz),
        atoms=geometry.atoms.copy(),
        lattice=geometry.lattice.copy(),
    )
    g._names = geometry.names.copy()
    return g


@register_sisl_dispatch(Geometry, module="sisl")
def write(geometry: Geometry, sile: SileLike, *args, **kwargs) -> None:
    """Writes geometry to the `sile` using `sile.write_geometry`

    Parameters
    ----------
    sile :
        a `Sile` object which will be used to write the geometry
        if it is a string it will create a new sile using `get_sile`
    *args, **kwargs:
        Any other args will be passed directly to the
        underlying routine

    See Also
    --------
    Geometry.read : reads a `Geometry` from a given `Sile`/file
    write : generic sisl function dispatcher
    """
    # This only works because, they *must*
    # have been imported previously
    from sisl.io import BaseSile, get_sile

    if isinstance(sile, BaseSile):
        sile.write_geometry(geometry, *args, **kwargs)
    else:
        with get_sile(sile, mode="w") as fh:
            fh.write_geometry(geometry, *args, **kwargs)


class ApplyFunc(Protocol):
    def __call__(self, data: npt.ArrayLike, axis: int) -> Any:
        pass


@register_sisl_dispatch(Geometry, module="sisl")
def apply(
    geometry: Geometry,
    data: npt.ArrayLike,
    func: Union[ApplyFunc, str],
    mapper: Union[Callable[[int], int], str],
    axis: int = 0,
    segments: Union[
        Literal["atoms", "orbitals", "all"], Union[Iterator[int], Iterator[List[int]]]
    ] = "atoms",
) -> np.ndarray:
    r"""Apply a function `func` to the data along axis `axis` using the method specified

    This can be useful for applying conversions from orbital data to atomic data through
    sums or other functions.

    The data may be of any shape but it is expected the function can handle arguments as
    ``func(data, axis=axis)``.

    Parameters
    ----------
    data :
        the data to be converted
    func :
        a callable function that transforms the data in some way.
        If a `str`, will use ``getattr(numpy, func)``.
    mapper :
        a function transforming the `segments` into some other segments that
        is present in `data`.
        It can accept anything the `segments` returns.
        If a `str`, it will be equivalent to ``getattr(geometry, mapper)``
    axis :
        axis selector for `data` along which `func` will be applied
    segments :
        which segments the `mapper` will recieve, if atoms, each atom
        index will be passed to the ``mapper(ia)``.
        If ``'all'``, it will be ``range(data.shape[axis])``.

    Examples
    --------
    Convert orbital data into summed atomic data

    >>> g = sisl.geom.diamond(atoms=sisl.Atom(6, R=(1, 2)))
    >>> orbital_data = np.random.rand(10, g.no, 3)
    >>> atomic_data = g.apply(orbital_data, np.sum, mapper=partial(g.a2o, all=True), axis=1)

    The same can be accomplished by passing an explicit segment iterator,
    note that ``iter(g) == range(g.na)``

    >>> atomic_data = g.apply(orbital_data, np.sum, mapper=partial(g.a2o, all=True), axis=1,
    ...                       segments=g)

    To only take out every 2nd orbital:

    >>> alternate_data = g.apply(orbital_data, np.sum, mapper=lambda idx: idx[::2], axis=1,
    ...                          segments="all")

    """
    if isinstance(segments, str):
        segment = segments.lower().rstrip("s")
        if segment == "atom":
            segments = range(geometry.na)
        elif segment in ("orbital", "none"):
            segments = range(geometry.no)
        elif segment == "all":
            segments = range(data.shape[axis])
        else:
            raise ValueError(
                f"{geometry.__class__}.apply got wrong argument 'segments'={segments}"
            )

    if isinstance(func, str):
        func = getattr(np, func)

    if isinstance(mapper, str):
        # an internal mapper
        mapper = getattr(geometry, mapper)

    take = np.take
    atleast_1d = np.atleast_1d
    new_data = [
        func(take(data, atleast_1d(mapper(segment)), axis), axis=axis)
        for segment in segments
    ]

    new_data = np.stack(new_data, axis=axis)
    if new_data.ndim != data.ndim:
        new_data = np.expand_dims(new_data, axis=axis)
    return new_data


@register_sisl_dispatch(Geometry, module="sisl")
def sort(
    geometry: Geometry, **kwargs
) -> Union[Geometry, tuple[Geometry, list[list[int]]]]:
    r"""Sort atoms in a nested fashion according to various criteria

    There are many ways to sort a `Geometry`:

    * by Cartesian coordinates, `axes`/`axis`
    * by lattice vectors, `lattice`
    * by user defined vectors, `vector`
    * by grouping atoms, `group`
    * by a user defined function, `func`
    * by a user defined function using internal sorting algorithm, `func_sort`
    * a combination of the above in arbitrary order

    Additionally one may sort ascending or descending.

    This method allows nested sorting based on keyword arguments.

    Parameters
    ----------
    atoms : AtomsIndex, optional
       only perform sorting algorithm for subset of atoms. This is *NOT* a positional dependent
       argument. All sorting algorithms will *only* be performed on these atoms.
       Default, all atoms will be sorted.
    ret_atoms : bool, optional
       return a list of list for the groups of atoms that have been sorted.
    axis, axes : int or tuple of int, optional
       sort coordinates according to Cartesian coordinates, if a tuple of
       ints is passed it will be equivalent to ``sort(axes=axes) == sort(axis0=axes[0], axis1=axes[1])``.
       This behaves differently than `numpy.lexsort`!
    lattice : int or tuple of int, optional
       sort coordinates according to lattice vectors, if a tuple of
       ints is passed it will be equivalent to ``sort(lattice0=lattice[0], lattice1=lattice[1])``.
       Note that before sorting we multiply the fractional coordinates by the length of the
       lattice vector. This ensures that `atol` is meaningful for both `axes` and `lattice` since
       they will be on the same order of magnitude.
       This behaves differently than `numpy.lexsort`!
    vector : Coord, optional
       sort along a user defined vector, similar to `lattice` but with a user defined
       direction. Note that `lattice` sorting and `vector` sorting are *only* equivalent
       when the lattice vector is orthogonal to the other lattice vectors.
    group : {'Z', 'symbol', 'tag', 'species'} or (list of list), optional
       group together a set of atoms by various means.
       `group` may be one of the listed strings.
       For ``'Z'`` atoms will be grouped in atomic number
       For ``'symbol'`` atoms will be grouped by their atomic symbol.
       For ``'tag'`` atoms will be grouped by their atomic tag.
       For ``'species'`` atoms will be sorted according to their species index.
       If a tuple/list is passed the first item is described. All subsequent items are a
       list of groups, where each group comprises elements that should be sorted on an
       equal footing. If one of the groups is None, that group will be replaced with all
       non-mentioned elements. See examples.
    func : callable or list-like of callable, optional
       pass a sorting function which should have an interface like ``func(geometry, atoms, **kwargs)``.
       The first argument is the geometry to sort. The 2nd argument is a list of indices in
       the current group of sorted atoms. And ``**kwargs`` are any optional arguments
       currently collected, i.e. `ascend`, `atol` etc.
       The function should return either a list of atoms, or a list of list of atoms (in which
       case the atomic indices have been split into several groups that will be sorted individually
       for subsequent sorting methods).
       In either case the returned indices must never hold any other indices but the ones passed
       as ``atoms``.
       If a list/tuple of functions, they will be processed in that order.
    func_sort : callable or list-like of callable, optional
       pass a function returning a 1D array corresponding to all atoms in the geometry.
       The interface should simply be: ``func(geometry)``.
       Those values will be passed down to the internal sorting algorithm.
       To be compatible with `atol` the returned values from `func_sort` should
       be on the scale of coordinates (in Ang).
    ascend, descend : bool, optional
        control ascending or descending sorting for all subsequent sorting methods.
        Default ``ascend=True``.
    atol : float, optional
        absolute tolerance when sorting numerical arrays for subsequent sorting methods.
        When a selection of sorted coordinates are grouped via `atol`, we ensure such
        a group does not alter its indices. I.e. the group is *always* ascending indices.
        Note this may have unwanted side-effects if `atol` is very large compared
        to the difference between atomic coordinates.
        Default ``1e-9``.

    Notes
    -----
    The order of arguments is also the sorting order. ``sort(axes=0, lattice=0)`` is different
    from ``sort(lattice=0, axes=0)``

    All arguments may be suffixed with integers. This allows multiple keyword arguments
    to control sorting algorithms
    in different order. It also allows changing of sorting settings between different calls.
    Note that the integers have no relevance to the order of execution!
    See examples.

    Returns
    -------
    geometry : Geometry
        sorted geometry
    index : list of list of indices
        indices that would sort the original structure to the output, only returned if `ret_atoms` is True

    Examples
    --------
    >>> geom = sisl.geom.bilayer(top_atoms=sisl.Atom[5, 7], bottom_atoms=sisl.Atom(6))
    >>> geom = geom.tile(5, 0).repeat(7, 1)

    Sort according to :math:`x` coordinate

    >>> geom.sort(axes=0)

    Sort according to :math:`z`, then :math:`x` for each group created from first sort

    >>> geom.sort(axes=(2, 0))

    Sort according to :math:`z`, then first lattice vector

    >>> geom.sort(axes=2, lattice=0)

    Sort according to :math:`z` (ascending), then first lattice vector (descending)

    >>> geom.sort(axes=2, ascend=False, lattice=0)

    Sort according to :math:`z` (descending), then first lattice vector (ascending)
    Note how integer suffixes has no importance.

    >>> geom.sort(ascend1=False, axes=2, ascend0=True, lattice=0)

    Sort only atoms ``range(1, 5)`` first by :math:`z`, then by first lattice vector

    >>> geom.sort(axes=2, lattice=0, atoms=np.arange(1, 5))

    Sort two groups of atoms ``[range(1, 5), range(5, 10)]`` (individually) by :math:`z` coordinate

    >>> geom.sort(axes=2, atoms=[np.arange(1, 5), np.arange(5, 10)])

    The returned sorting indices may be used for manual sorting. Note
    however, that this requires one to perform a sorting for all atoms.
    In such a case the following sortings are equal.

    >>> geom0, atoms0 = geom.sort(axes=2, lattice=0, ret_atoms=True)
    >>> _, atoms1 = geom.sort(axes=2, ret_atoms=True)
    >>> geom1, atoms1 = geom.sort(lattice=0, atoms=atoms1, ret_atoms=True)
    >>> geom2 = geom.sub(np.concatenate(atoms0))
    >>> geom3 = geom.sub(np.concatenate(atoms1))
    >>> assert geom0 == geom1
    >>> assert geom0 == geom2
    >>> assert geom0 == geom3

    Default sorting is equivalent to ``axes=(0, 1, 2)``

    >>> assert geom.sort() == geom.sort(axes=(0, 1, 2))

    Sort along a user defined vector ``[2.2, 1., 0.]``

    >>> geom.sort(vector=[2.2, 1., 0.])

    Integer specification has no influence on the order of operations.
    It is *always* the keyword argument order that determines the operation.

    >>> assert geom.sort(axis2=1, axis0=0, axis1=2) == geom.sort(axes=(1, 0, 2))

    Sort by atomic numbers

    >>> geom.sort(group='Z') # 5, 6, 7

    One may group several elements together on an equal footing (``None`` means all non-mentioned elements)
    The order of the groups are important (the first two are *not* equal, the last three *are* equal)

    >>> geom.sort(group=('symbol', 'C'), axes=2) # C will be sorted along z
    >>> geom.sort(axes=1, atoms='C', axes1=2) # all along y, then C sorted along z
    >>> geom.sort(group=('symbol', 'C', None)) # C, [B, N]
    >>> geom.sort(group=('symbol', None, 'C')) # [B, N], C
    >>> geom.sort(group=('symbol', ['N', 'B'], 'C')) # [B, N], C (B and N unaltered order)
    >>> geom.sort(group=('symbol', ['B', 'N'], 'C')) # [B, N], C (B and N unaltered order)

    A group based sorting can use *anything* that can be fetched from the `Atom` object,
    sort first according to mass, then for all with the same mass, sort according to atomic
    tag:

    >>> geom.sort(group0='mass', group1='tag')

    One can also manually specify to only sort sub-groups via atomic indices, the
    following will keep ``[0, 1, 2]`` and ``[3, 4, 5]`` in their respective relative
    position, but each block of atoms will be sorted along the 2nd lattice vector:

    >>> geom.sort(group=([0, 1, 2], [3, 4, 5]), axes=1)

    A too high `atol` may have unexpected side-effects. This is because of the way
    the sorting algorithm splits the sections for nested sorting.
    So for coordinates with a continuous displacement the sorting may break and group
    a large range into 1 group. Consider the following array to be split in groups
    while sorting.

    An example would be a linear chain with a middle point with a much shorter distance.

    >>> x = np.arange(5) * 0.1
    >>> x[3:] -= 0.095
    y = z = np.zeros(5)
    geom = si.Geometry(np.stack((x, y, z), axes=1))
    >>> geom.xyz[:, 0]
    [0.    0.1   0.2   0.205 0.305]

    In this case a high tolerance (``atol>0.005``) would group atoms 2 and 3
    together

    >>> geom.sort(atol=0.01, axes=0, ret_atoms=True)[1]
    [[0], [1], [2, 3], [4]]

    However, a very low tolerance will not find these two as atoms close
    to each other.

    >>> geom.sort(atol=0.001, axes=0, ret_atoms=True)[1]
    [[0], [1], [2], [3], [4]]
    """

    # We need a way to easily handle nested lists
    # This small class handles lists and allows appending nested lists
    # while flattening them.
    class NestedList:
        __slots__ = ("_idx",)

        def __init__(geometry, idx=None, sort=False):
            geometry._idx = []
            if not idx is None:
                geometry.append(idx, sort)

        def append(geometry, idx, sort=False):
            if isinstance(idx, (tuple, list, np.ndarray)):
                if isinstance(idx[0], (tuple, list, np.ndarray)):
                    for ix in idx:
                        geometry.append(ix, sort)
                    return
            elif isinstance(idx, NestedList):
                idx = idx.tolist()
            if len(idx) > 0:
                if sort:
                    geometry._idx.append(np.sort(idx))
                else:
                    geometry._idx.append(np.asarray(idx))

        def __iter__(geometry):
            yield from geometry._idx

        def __len__(geometry):
            return len(geometry._idx)

        def ravel(geometry):
            if len(geometry) == 0:
                return np.array([], dtype=np.int64)
            return np.concatenate([i for i in geometry]).ravel()

        def tolist(geometry):
            return geometry._idx

        def __str__(geometry):
            if len(geometry) == 0:
                return f"{geometry.__class__.__name__}{{empty}}"
            out = ",\n ".join(map(lambda x: str(x.tolist()), geometry))
            return f"{geometry.__class__.__name__}{{\n {out}}}"

    def _sort(val, atoms, **kwargs):
        """We do not sort according to lexsort"""
        if len(val) <= 1:
            # no values to sort
            return atoms

        # control ascend vs descending
        ascend = kwargs["ascend"]
        atol = kwargs["atol"]

        new_atoms = NestedList()
        for atom in atoms:
            if len(atom) <= 1:
                # no need for complexity
                new_atoms.append(atom)
                continue

            # Sort values
            jdx = atom[np.argsort(val[atom])]
            if ascend:
                d = np.diff(val[jdx]) > atol
            else:
                jdx = jdx[::-1]
                d = np.diff(val[jdx]) < -atol
            new_atoms.append(np.split(jdx, d.nonzero()[0] + 1), sort=True)
        return new_atoms

    # Functions allowed by external users
    funcs = dict()

    def _axes(axes, atoms, **kwargs):
        """Cartesian coordinate sort"""
        if isinstance(axes, int):
            axes = (axes,)
        for axis in axes:
            atoms = _sort(geometry.xyz[:, axis], atoms, **kwargs)
        return atoms

    funcs["axis"] = _axes
    funcs["axes"] = _axes

    def _lattice(lattice, atoms, **kwargs):
        """
        We scale the fractional coordinates with the lattice vector length.
        This ensures `atol` has a meaningful size for very large structures.
        """
        if isinstance(lattice, int):
            lattice = (lattice,)
        fxyz = geometry.fxyz
        for ax in lattice:
            atoms = _sort(fxyz[:, ax] * geometry.lattice.length[ax], atoms, **kwargs)
        return atoms

    funcs["lattice"] = _lattice

    def _vector(vector, atoms, **kwargs):
        """
        Calculate fraction of positions along a vector and sort along it.
        We first normalize the vector to ensure that `atol` is meaningful
        for very large structures (ensures scale is on the order of Ang).

        A vector projection will only be equivalent to lattice projection
        when a lattice vector is orthogonal to the other lattice vectors.
        """
        # Ensure we are using a copied data array
        vector = _a.asarrayd(vector).copy()
        # normalize
        vector /= fnorm(vector)
        # Perform a . b^ == scalar projection
        return _sort(geometry.xyz.dot(vector), atoms, **kwargs)

    funcs["vector"] = _vector

    def _funcs(funcs, atoms, **kwargs):
        """
        User defined function (tuple/list of function)
        """

        def _func(func, atoms, kwargs):
            nl = NestedList()
            for atom in atoms:
                # TODO add check that
                #  res = func(...) in a
                # A user *may* remove an atom from the sorting here (but
                # that negates all sorting of that atom)
                nl.append(func(geometry, atom, **kwargs))
            return nl

        if callable(funcs):
            funcs = [funcs]
        for func in funcs:
            atoms = _func(func, atoms, kwargs)
        return atoms

    funcs["func"] = _funcs

    def _func_sort(funcs, atoms, **kwargs):
        """
        User defined function, but using internal sorting
        """
        if callable(funcs):
            funcs = [funcs]
        for func in funcs:
            atoms = _sort(func(geometry), atoms, **kwargs)
        return atoms

    funcs["func_sort"] = _func_sort

    def _group_vals(vals, groups, atoms, **kwargs):
        """
        `vals` should be of size ``len(geometry)`` and be parseable
        by numpy
        """
        nl = NestedList()

        # Create unique list of values
        uniq_vals = np.unique(vals)
        if len(groups) == 0:
            # fake the groups argument
            groups = [[i] for i in uniq_vals]
        else:
            # Check if one of the elements of group is None
            # In this case we replace it with the missing rest
            # of the missing unique items
            try:
                none_idx = groups.index(None)

                # we have a None (ensure we use a list, tuples are
                # immutable)
                groups = list(groups)
                groups[none_idx] = []

                uniq_groups = np.unique(np.concatenate(groups))
                # add a new group that is in uniq_vals, but not in uniq_groups
                rest = uniq_vals[np.isin(uniq_vals, uniq_groups, invert=True)]
                groups[none_idx] = rest
            except ValueError:
                # there is no None in the list
                pass

        for at in atoms:
            # reduce search
            at_vals = vals[at]
            # loop group values
            for group in groups:
                # retain original indexing
                nl.append(at[np.isin(at_vals, group)])
        return nl

    def _group(method_group, atoms, **kwargs):
        """
        Group based sorting is based on a named identification.

        group: str or tuple of (str, list of lists)

        symbol: order by symbol (most cases same as Z)
        Z: order by atomic number
        tag: order by atom tag (should be the same as species)
        specie/species: order by specie (in order of contained in the Geometry)
        """
        # Create new list
        nl = NestedList()

        if isinstance(method_group, str):
            method = method_group
            groups = []
        elif isinstance(method_group[0], str):
            method, *in_groups = method_group

            # Ensure all groups are lists
            groups = []
            NoneType = type(None)
            for group in in_groups:
                if isinstance(group, (tuple, list, np.ndarray, NoneType)):
                    groups.append(group)
                else:
                    groups.append([group])
        else:
            # a special case where group is a list of lists
            # i.e. [[0, 1, 2], [3, 4, 5]]
            for idx in method_group:
                idx = geometry._sanitize_atoms(idx)
                for at in atoms:
                    nl.append(at[np.isin(at, idx)])
            return nl

        # See if the attribute exists for the atoms
        if method.lower().startswith("specie"):
            # this one has two spelling options!
            method = "species"

        # now get them through `getattr`
        if hasattr(geometry.atoms[0], method):
            vals = [getattr(a, method) for a in geometry.atoms]

        elif hasattr(geometry.atoms[0], method.lower()):
            method = method.lower()
            vals = [getattr(a, method) for a in geometry.atoms]

        else:
            raise ValueError(
                f"{geometry.__class__.__name__}.sort group only supports attributes that can be fetched from Atom objects, some are [Z, species, tag, symbol, mass, ...] and more"
            )

        return _group_vals(np.array(vals), groups, atoms, **kwargs)

    funcs["group"] = _group
    funcs["groups"] = _group

    def stripint(s: str) -> str:
        """Remove integers from end of string -> Allow multiple arguments"""
        if s[-1] in "0123456789":
            return stripint(s[:-1])
        return s

    # Now perform cumulative sort function
    # Our point is that we would like to allow users to do consecutive sorting
    # based on different keys

    # We also allow specific keys for specific methods
    func_kw = dict()
    func_kw["ascend"] = True
    func_kw["atol"] = 1e-9

    def update_flag(kw, arg, val) -> bool:
        if arg in ("ascending", "ascend"):
            kw["ascend"] = val
            return True
        elif arg in ("descending", "descend"):
            kw["ascend"] = not val
            return True
        elif arg == "atol":
            kw["atol"] = val
            return True
        return False

    # Default to all atoms
    atoms = NestedList(geometry._sanitize_atoms(kwargs.pop("atoms", None)))
    ret_atoms = kwargs.pop("ret_atoms", False)

    # In case the user just did geometry.sort, it will default to sort x, y, z
    if len(kwargs) == 0:
        kwargs["axes"] = (0, 1, 2)

    for key_int, method in kwargs.items():
        key = stripint(key_int)
        if update_flag(func_kw, key, method):
            continue
        if key not in funcs:
            raise ValueError(
                f"{geometry.__class__.__name__}.sort unrecognized keyword '{key}' ('{key_int}')"
            )
        # call sorting algorithm and retrieve new grouped sorting
        atoms = funcs[key](method, atoms, **func_kw)

    # convert to direct list
    atoms_flat = atoms.ravel()

    # Ensure that all atoms are present
    # This is necessary so we don't remove any atoms.
    # Currently, the non-sorted atoms *stay* in-place.
    if len(atoms_flat) != len(geometry):
        all_atoms = _a.arangei(len(geometry))
        all_atoms[np.sort(atoms_flat)] = atoms_flat[:]
        atoms_flat = all_atoms

    if ret_atoms:
        return geometry.sub(atoms_flat), atoms.tolist()
    return geometry.sub(atoms_flat)


@register_sisl_dispatch(Geometry, module="sisl")
def swap(geometry: Geometry, atoms1: AtomsIndex, atoms2: AtomsIndex) -> Geometry:
    """Swap a set of atoms in the geometry and return a new one

    This can be used to reorder elements of a geometry.

    Parameters
    ----------
    atoms1 :
         the first list of atomic coordinates
    atoms2 :
         the second list of atomic coordinates
    """
    atoms1 = geometry._sanitize_atoms(atoms1)
    atoms2 = geometry._sanitize_atoms(atoms2)
    xyz = np.copy(geometry.xyz)
    xyz[atoms1, :] = geometry.xyz[atoms2, :]
    xyz[atoms2, :] = geometry.xyz[atoms1, :]
    return geometry.__class__(
        xyz,
        atoms=geometry.atoms.swap(atoms1, atoms2),
        lattice=geometry.lattice.copy(),
    )


@register_sisl_dispatch(Geometry, module="sisl")
def insert(geometry: Geometry, atom: AtomsIndex, other: GeometryLike) -> Geometry:
    """Inserts other atoms right before index

    We insert the `geometry` `Geometry` before `atom`.
    Note that this will not change the unit cell.

    Parameters
    ----------
    atom :
       the atomic index at which the other geometry is inserted
    other :
       the other geometry to be inserted

    See Also
    --------
    Geometry.add : add geometries
    Geometry.append : appending geometries
    Geometry.prepend : prepending geometries
    Geometry.attach : attach a geometry
    """
    atom = geometry._sanitize_atoms(atom)
    if atom.size > 1:
        raise ValueError(
            f"{geometry.__class__.__name__}.insert requires only 1 atomic index for insertion."
        )
    other = geometry.new(other)
    xyz = np.insert(geometry.xyz, atom, other.xyz, axis=0)
    atoms = geometry.atoms.insert(atom, other.atoms)
    return geometry.__class__(xyz, atoms, lattice=geometry.lattice.copy())


@register_sisl_dispatch(Geometry, module="sisl")
def tile(geometry: Geometry, reps: int, axis: CellAxis) -> Geometry:
    """Tile the geometry to create a bigger one

    The atomic indices are retained for the base structure.

    Tiling and repeating a geometry will result in the same geometry.
    The *only* difference between the two is the final ordering of the atoms.

    Parameters
    ----------
    reps :
       number of tiles (repetitions)
    axis :
       direction of tiling, 0, 1, 2 according to the cell-direction

    Examples
    --------
    >>> geom = Geometry([[0, 0, 0], [0.5, 0, 0]], lattice=1.)
    >>> g = geom.tile(2,axis=0)
    >>> print(g.xyz) # doctest: +NORMALIZE_WHITESPACE
    [[0.   0.   0. ]
     [0.5  0.   0. ]
     [1.   0.   0. ]
     [1.5  0.   0. ]]
    >>> g = tile(geom, 2,0).tile(2,axis=1)
    >>> print(g.xyz) # doctest: +NORMALIZE_WHITESPACE
    [[0.   0.   0. ]
     [0.5  0.   0. ]
     [1.   0.   0. ]
     [1.5  0.   0. ]
     [0.   1.   0. ]
     [0.5  1.   0. ]
     [1.   1.   0. ]
     [1.5  1.   0. ]]

    In functional form:

    >>> tile(geom, 2, axis=0)

    See Also
    --------
    Geometry.repeat : equivalent but different ordering of final structure
    Geometry.untile : opposite method of this
    """
    if reps < 1:
        raise ValueError(
            f"{geometry.__class__.__name__}.tile requires a repetition above 0"
        )
    axis = direction(axis)

    lattice = geometry.lattice.tile(reps, axis)

    # Our first repetition *must* be with
    # the former coordinate
    xyz = np.tile(geometry.xyz, (reps, 1))
    # We may use broadcasting rules instead of repeating stuff
    xyz.shape = (reps, geometry.na, 3)
    nr = _a.arangei(reps)
    nr.shape = (reps, 1, 1)
    # Correct the unit-cell offsets
    xyz += nr * geometry.cell[axis, :]
    xyz.shape = (-1, 3)

    # Create the geometry and return it (note the smaller atoms array
    # will also expand via tiling)
    return geometry.__class__(xyz, atoms=geometry.atoms.tile(reps), lattice=lattice)


@register_sisl_dispatch(Geometry, module="sisl")
def untile(
    geometry: Geometry,
    reps: int,
    axis: CellAxis,
    segment: int = 0,
    rtol: float = 1e-4,
    atol: float = 1e-4,
) -> Geometry:
    """A subset of atoms from the geometry by cutting the geometry into `reps` parts along the direction `axis`.

    This will effectively change the unit-cell in the `axis` as-well
    as removing ``geometry.na/reps`` atoms.
    It requires that ``geometry.na % reps == 0``.

    REMARK: You need to ensure that all atoms within the first
    cut out region are within the primary unit-cell.

    Doing ``geom.untile(2, 1).tile(2, 1)``, could for symmetric setups,
    be equivalent to a no-op operation. A ``UserWarning`` will be issued
    if this is not the case.

    This method is the reverse of `tile`.

    Parameters
    ----------
    reps :
        number of times the structure will be cut (untiled)
    axis :
        the axis that will be cut
    segment :
        returns the i'th segment of the untiled structure
        Currently the atomic coordinates are not translated,
        this may change in the future.
    rtol :
        directly passed to `numpy.allclose`
    atol :
        directly passed to `numpy.allclose`

    Examples
    --------
    >>> g = sisl.geom.graphene()
    >>> gxyz = g.tile(4, 0).tile(3, 1).tile(2, 2)
    >>> G = gxyz.untile(2, 2).untile(3, 1).untile(4, 0)
    >>> np.allclose(g.xyz, G.xyz)
    True

    In functional form:

    >>> untile(geom, 2, axis=0)

    See Also
    --------
    Geometry.tile : opposite method of this
    Geometry.repeat : equivalent geometry as `tile` but different ordering of final structure
    """
    if geometry.na % reps != 0:
        raise ValueError(
            f"{geometry.__class__.__name__}.untile "
            f"cannot be cut into {reps} different "
            "pieces. Please check your geometry and input."
        )
    axis = direction(axis)

    # Truncate to the correct segments
    lseg = segment % reps
    # Cut down cell
    lattice = geometry.lattice.untile(reps, axis)
    # List of atoms
    n = geometry.na // reps
    off = n * lseg
    new = geometry.sub(_a.arangei(off, off + n))
    new.set_lattice(lattice)
    if not np.allclose(new.tile(reps, axis).xyz, geometry.xyz, rtol=rtol, atol=atol):
        warn(
            "The cut structure cannot be re-created by tiling\n"
            "The tolerance between the coordinates can be altered using rtol, atol"
        )
    return new


@register_sisl_dispatch(Geometry, module="sisl")
def repeat(geometry: Geometry, reps: int, axis: CellAxis) -> Geometry:
    """Create a repeated geometry

    The atomic indices are *NOT* retained from the base structure.

    The expansion of the atoms are basically performed using this
    algorithm:

    >>> ja = 0
    >>> for ia in range(geometry.na):
    ...     for id,r in args:
    ...        for i in range(r):
    ...           ja = ia + cell[id,:] * i

    For geometries with a single atom this routine returns the same as
    `tile`.

    Tiling and repeating a geometry will result in the same geometry.
    The *only* difference between the two is the final ordering of the atoms.

    Parameters
    ----------
    reps :
       number of repetitions
    axis :
       direction of repetition, 0, 1, 2 according to the cell-direction

    Examples
    --------
    >>> geom = Geometry([[0, 0, 0], [0.5, 0, 0]], lattice=1)
    >>> g = geom.repeat(2,axis=0)
    >>> print(g.xyz) # doctest: +NORMALIZE_WHITESPACE
    [[0.   0.   0. ]
     [1.   0.   0. ]
     [0.5  0.   0. ]
     [1.5  0.   0. ]]
    >>> g = geom.repeat(2,0).repeat(2,1)
    >>> print(g.xyz) # doctest: +NORMALIZE_WHITESPACE
    [[0.   0.   0. ]
     [0.   1.   0. ]
     [1.   0.   0. ]
     [1.   1.   0. ]
     [0.5  0.   0. ]
     [0.5  1.   0. ]
     [1.5  0.   0. ]
     [1.5  1.   0. ]]

    In functional form:

    >>> repeat(geom, 2, axis=0)

    See Also
    --------
    Geometry.tile : equivalent geometry as `repeat` but different ordering of final structure
    Geometry.unrepeat : opposite method of this
    """
    if reps < 1:
        raise ValueError(
            f"{geometry.__class__.__name__}.repeat requires a repetition above 0"
        )
    axis = direction(axis)

    lattice = geometry.lattice.repeat(reps, axis)

    # Our first repetition *must* be with
    # the former coordinate
    xyz = np.repeat(geometry.xyz, reps, axis=0)
    # We may use broadcasting rules instead of repeating stuff
    xyz.shape = (geometry.na, reps, 3)
    nr = _a.arangei(reps)
    nr.shape = (1, reps)
    for i in range(3):
        # Correct the unit-cell offsets along `i`
        xyz[:, :, i] += nr * geometry.cell[axis, i]
    xyz.shape = (-1, 3)

    # Create the geometry and return it
    return geometry.__class__(xyz, atoms=geometry.atoms.repeat(reps), lattice=lattice)


@register_sisl_dispatch(Geometry, module="sisl")
def unrepeat(
    geometry: Geometry, reps: int, axis: CellAxis, *args, **kwargs
) -> Geometry:
    """Unrepeats the geometry similarly as `untile`

    This is the opposite of `Geometry.repeat`.

    Please see `Geometry.untile` for argument details.
    This algorithm first re-arranges atoms as though they had been tiled,
    then subsequently calls `untile`.

    See Also
    --------
    Geometry.repeat : opposite method of this
    """
    atoms = np.arange(geometry.na).reshape(-1, reps).T.ravel()
    return geometry.sub(atoms).untile(reps, axis, *args, **kwargs)


@register_sisl_dispatch(Geometry, module="sisl")
def translate(
    geometry: Geometry,
    v: CoordOrScalar,
    atoms: AtomsIndex = None,
) -> Geometry:
    """Translates the geometry by `v`

    `move` is a shorthand for this function.

    One can translate a subset of the atoms by supplying `atoms`.

    Returns a copy of the structure translated by `v`.

    Parameters
    ----------
    v :
         the value or vector to displace all atomic coordinates
         It should just be broad-castable with the geometry's coordinates.
    atoms :
         only displace the given atomic indices, if not specified, all
         atoms will be displaced
    """
    g = geometry.copy()
    if atoms is None:
        g.xyz += v
        np.asarray(v, g.xyz.dtype)
    else:
        g.xyz[geometry._sanitize_atoms(atoms).ravel(), :] += v
    return g


@register_sisl_dispatch(Geometry, module="sisl")
def move(geometry: Geometry, *args, **kwargs) -> Geometry:
    """See `translate` for details"""
    return translate(geometry, *args, **kwargs)


@register_sisl_dispatch(Geometry, module="sisl")
def sub(geometry: Geometry, atoms: AtomsIndex) -> Geometry:
    """Create a new `Geometry` with a subset of this `Geometry`

    Indices passed *MUST* be unique.

    Negative indices are wrapped and thus works.

    Parameters
    ----------
    atoms :
        indices/boolean of all atoms to be removed

    See Also
    --------
    Lattice.fit : update the supercell according to a reference supercell
    Geometry.remove : the negative of this routine, i.e. remove a subset of atoms
    """
    atoms = geometry.asc2uc(atoms)
    return geometry.__class__(
        geometry.xyz[atoms, :].copy(),
        atoms=geometry.atoms.sub(atoms),
        lattice=geometry.lattice.copy(),
    )


@register_sisl_dispatch(Geometry, module="sisl")
def remove(geometry: Geometry, atoms: AtomsIndex) -> Geometry:
    """Remove atoms from the geometry.

    Indices passed *MUST* be unique.

    Negative indices are wrapped and thus works.

    Parameters
    ----------
    atoms :
        indices/boolean of all atoms to be removed

    See Also
    --------
    Geometry.sub : the negative of this routine, i.e. retain a subset of atoms
    """
    atoms = geometry.asc2uc(atoms)
    if atoms.size == 0:
        return geometry.copy()
    atoms = np.delete(_a.arangei(geometry.na), atoms)
    return geometry.sub(atoms)


@register_sisl_dispatch(Geometry, module="sisl")
def rotate(
    geometry: Geometry,
    angle: float,
    v: Union[str, int, Coord],
    origin: Union[int, Coord] = (0, 0, 0),
    atoms: AtomsIndex = None,
    rad: bool = False,
    what: Optional[Literal["xyz", "abc", "abc+xyz", "x", "a", ...]] = None,
) -> Geometry:
    r"""Rotate geometry around vector and return a new geometry

    Per default will the entire geometry be rotated, such that everything
    is aligned as before rotation.

    However, by supplying ``what = 'abc|xyz'`` one can designate which
    part of the geometry that will be rotated.

    Parameters
    ----------
    angle :
         the angle in degrees to rotate the geometry. Set the ``rad``
         argument to use radians.
    v     :
         the normal vector to the rotated plane, i.e.
         ``[1, 0, 0]`` will rotate around the :math:`yz` plane.
         If a str it refers to the Cartesian direction (xyz), or the
         lattice vectors (abc). Providing several is the combined direction.
    origin :
         the origin of rotation. Anything but ``[0, 0, 0]`` is equivalent
         to a ``geometry.translate(-origin).rotate(...).translate(origin)``.
         If this is an `int` it corresponds to the atomic index.
    atoms :
         only rotate the given atomic indices, if not specified, all
         atoms will be rotated.
    rad :
         if ``True`` the angle is provided in radians (rather than degrees)
    what :
        which coordinate subject should be rotated,
        if any of ``abc`` is in this string the corresponding cell vector will be rotated
        if any of ``xyz`` is in this string the corresponding coordinates will be rotated
        If `atoms` is None, this defaults to "abc+xyz", otherwise it defaults
        to "xyz". See Examples.

    Examples
    --------
    rotate coordinates around the :math:`x`-axis
    >>> geom_x45 = geom.rotate(45, [1, 0, 0])

    rotate around the ``(1, 1, 0)`` direction but project the rotation onto the :math:`x`
    axis
    >>> geom_xy_x = geom.rotate(45, "xy", what='x')

    See Also
    --------
    Quaternion : class to rotate
    Lattice.rotate : rotation for a Lattice object
    """
    if isinstance(origin, Integral):
        origin = geometry.axyz(origin)
    origin = _a.asarray(origin)

    if atoms is None:
        if what is None:
            what = "abc+xyz"
        # important to not add a new dimension to xyz
        atoms = slice(None)
    else:
        if what is None:
            what = "xyz"
        # Only rotate the unique values
        atoms = geometry.asc2uc(atoms, unique=True)

    if isinstance(v, Integral):
        v = direction(v, abc=geometry.cell, xyz=np.diag([1, 1, 1]))
    elif isinstance(v, str):
        v = reduce(
            lambda a, b: a + direction(b, abc=geometry.cell, xyz=np.diag([1, 1, 1])),
            v,
            0,
        )

    # Ensure the normal vector is normalized... (flatten == copy)
    vn = _a.asarrayd(v).flatten()
    vn /= fnorm(vn)

    # Rotate by direct call
    lattice = geometry.lattice.rotate(angle, vn, rad=rad, what=what)

    # Copy
    xyz = np.copy(geometry.xyz)

    idx = []
    for i, d in enumerate("xyz"):
        if d in what:
            idx.append(i)

    if idx:
        # Prepare quaternion...
        q = Quaternion(angle, vn, rad=rad)
        q /= q.norm()
        # subtract and add origin, before and after rotation
        rotated = q.rotate(xyz[atoms] - origin) + origin
        # get which coordinates to rotate
        for i in idx:
            xyz[atoms, i] = rotated[:, i]

    return geometry.__class__(xyz, atoms=geometry.atoms.copy(), lattice=lattice)


@register_sisl_dispatch(Geometry, module="sisl")
def swapaxes(
    geometry: Geometry,
    axes1: AnyAxes,
    axes2: AnyAxes,
    what: Literal["abc", "xyz", "abc+xyz"] = "abc",
) -> Geometry:
    """Swap the axes components by either lattice vectors (only cell), or Cartesian coordinates

    See `Lattice.swapaxes` for details.

    Parameters
    ----------
    axes1 :
       the old axis indices (or labels if `str`)
       A string will translate each character as a specific
       axis index.
       Lattice vectors are denoted by ``abc`` while the
       Cartesian coordinates are denote by ``xyz``.
       If `str`, then `what` is not used.
    axes2 :
       the new axis indices, same as `axes1`
       old axis indices (or labels)
    what :
       what to swap, lattice vectors (abc) or Cartesian components (xyz),
       or both.
       Neglected for integer axes arguments.

    See Also
    --------
    Lattice.swapaxes

    Examples
    --------

    Only swap lattice vectors

    >>> g_ba = g.swapaxes(0, 1)
    >>> assert np.allclose(g.xyz, g_ba.xyz)

    Only swap Cartesian coordinates

    >>> g_ba = g.swapaxes(0, 1, "xyz")
    >>> assert np.allclose(g.xyz[:, [1, 0, 2]], g_ba.xyz)

    Consecutive swappings (what will be neglected if provided):

    1. abc, xyz -> bac, xyz
    2. bac, xyz -> bca, xyz
    3. bac, xyz -> bca, zyx

    >>> g_s = g.swapaxes("abx", "bcz")
    >>> assert np.allclose(g.xyz[:, [2, 1, 0]], g_s.xyz)
    >>> assert np.allclose(g.cell[[1, 2, 0]][:, [2, 1, 0]], g_s.cell)
    """
    # swap supercell
    # We do not need to check argument types etc,
    # Lattice.swapaxes will do this for us
    lattice = geometry.lattice.swapaxes(axes1, axes2, what)

    if isinstance(axes1, int) and isinstance(axes2, int):
        if "xyz" in what:
            axes1 = "xyz"[axes1]
            axes2 = "xyz"[axes2]
        else:
            axes1 = ""
            axes2 = ""

    # only thing we are going to swap is the coordinates
    idx = [0, 1, 2]
    for a, b in zip(axes1, axes2):
        aidx = "xyzabc".index(a)
        bidx = "xyzabc".index(b)
        if aidx < 3:
            idx[aidx], idx[bidx] = idx[bidx], idx[aidx]

    return geometry.__class__(
        geometry.xyz[:, idx].copy(), atoms=geometry.atoms.copy(), lattice=lattice
    )


@register_sisl_dispatch(Geometry, module="sisl")
def center(
    geometry: Geometry,
    atoms: AtomsIndex = None,
    what: Literal[
        "COP|xyz|position",
        "mm:xyz",
        "mm:lattice|mm:cell",
        "COM|mass",
        "COMM:pbc|mass:pbc",
        "COU|lattice|cell",
    ] = "xyz",
) -> np.ndarray:
    """Returns the center of the geometry

    By specifying `what` one can control whether it should be:

    * ``COP|xyz|position``: Center of coordinates (default)
    * ``mm:xyz``: Center of minimum+maximum of Cartesian coordinates
    * ``mm:lattice|mm:cell``: Center of minimum+maximum of lattice vectors Cartesian coordinates
    * ``COM|mass``: Center of mass
    * ``COM:pbc|mass:pbc``: Center of mass using periodicity, if the point 0, 0, 0 is returned it
        may likely be because of a completely periodic system with no true center of mass
    * ``COU|lattice|cell``: Center of lattice vectors

    Parameters
    ----------
    atoms :
        list of atomic indices to find center of
    what :
        determine which center to calculate
    """
    what = what.lower()
    if what in ("cou", "lattice", "cell"):
        return geometry.lattice.center()

    if atoms is None:
        g = geometry
    else:
        g = geometry.sub(atoms)

    if what in ("com:pbc", "mass:pbc"):
        mass = g.mass
        sum_mass = mass.sum()
        # the periodic center of mass is determined by transfering all
        # coordinates onto a circle -> fxyz * 2pi
        # Then we mass average the circle angles for each of the fractional
        # coordinates, and transform back into the cartesian coordinate system
        theta = g.fxyz * (2 * np.pi)
        # construct angles
        avg_cos = (mass @ np.cos(theta)) / sum_mass
        avg_sin = (mass @ np.sin(theta)) / sum_mass
        avg_theta = np.arctan2(-avg_sin, -avg_cos) / (2 * np.pi) + 0.5
        return avg_theta @ g.lattice.cell

    if what in ("com", "mass", "cop:mass"):
        mass = g.mass
        return mass @ g.xyz / mass.sum()

    if what in ("mm:xyz", "mm(xyz)"):
        return (g.xyz.max(0) + g.xyz.min(0)) / 2

    if what in ("mm:lattice", "mm:cell"):
        return (g.cell.max(0) + g.cell.min(0)) / 2

    if what in ("cop", "xyz", "position"):
        return np.mean(g.xyz, axis=0)

    raise ValueError(
        f"{geometry.__class__.__name__}.center could not understand option 'what' got {what}"
    )


@register_sisl_dispatch(Geometry, module="sisl")
def append(
    geometry: Geometry,
    other: LatticeOrGeometryLike,
    axis: CellAxis,
    offset: Union[Literal["none", "min"], Coord] = "none",
) -> Geometry:
    """Appends two structures along `axis`

    This will automatically add the ``geometry.cell[axis]`` to all atomic
    coordiates in the `other` structure before appending.

    The basic algorithm is this:

    >>> oxa = other.xyz + geometry.cell[axis][None,:]
    >>> geometry.xyz = np.append(geometry.xyz,oxa)
    >>> geometry.cell[axis] += other.cell[axis]

    NOTE: The cell appended is only in the axis that
    is appended, which means that the other cell directions
    need not conform.

    Parameters
    ----------
    other :
        Other geometry class which needs to be appended
        If a `Lattice` only the super cell will be extended
    axis :
        Cell direction to which the `other` geometry should be
        appended.
    offset :
        By default appending two structures will simply use the coordinates,
        as is.
        With 'min', the routine will shift both the structures along the cell
        axis of `geometry` such that they coincide at the first atom, lastly one
        may use a specified offset to manually select how `other` is displaced.
        NOTE: That ``geometry.cell[axis]`` will be added to `offset` if `other` is
        a geometry.

    See Also
    --------
    Geometry.add : add geometries
    Geometry.prepend : prepending geometries
    Geometry.attach : attach a geometry
    Geometry.insert : insert a geometry
    """
    axis = direction(axis)
    if isinstance(other, Lattice):
        # Only extend the supercell.
        xyz = np.copy(geometry.xyz)
        atoms = geometry.atoms.copy()
        lattice = geometry.lattice.append(other, axis)
        names = geometry._names.copy()
        if isinstance(offset, str):
            if offset == "none":
                offset = [0, 0, 0]
            else:
                raise ValueError(
                    f"{geometry.__class__.__name__}.append requires offset to be (3,) for supercell input"
                )
        xyz += _a.asarray(offset)

    else:
        # sanitize output
        other = geometry.new(other)
        if isinstance(offset, str):
            offset = offset.lower()
            if offset == "none":
                offset = geometry.cell[axis, :]
            elif offset == "min":
                # We want to align at the minimum position along the `axis`
                min_f = geometry.fxyz[:, axis].min()
                min_other_f = np.dot(other.xyz, geometry.icell.T)[:, axis].min()
                offset = geometry.cell[axis, :] * (1 + min_f - min_other_f)
            else:
                raise ValueError(
                    f"{geometry.__class__.__name__}.append requires align keyword to be one of [none, min, (3,)]"
                )
        else:
            offset = geometry.cell[axis, :] + _a.asarray(offset)

        xyz = np.append(geometry.xyz, offset + other.xyz, axis=0)
        atoms = geometry.atoms.append(other.atoms)
        lattice = geometry.lattice.append(other.lattice, axis)
        names = geometry._names.merge(other._names, offset=len(geometry))

    return geometry.__class__(xyz, atoms=atoms, lattice=lattice, names=names)


@register_sisl_dispatch(Geometry, module="sisl")
def prepend(
    geometry: Geometry,
    other: LatticeOrGeometryLike,
    axis: CellAxis,
    offset: Union[Literal["none", "min"], Coord] = "none",
) -> Geometry:
    """Prepend two structures along `axis`

    This will automatically add the ``geometry.cell[axis,:]`` to all atomic
    coordinates in the `other` structure before appending.

    The basic algorithm is this:

    >>> oxa = other.xyz
    >>> geometry.xyz = np.append(oxa, geometry.xyz + other.cell[axis,:][None,:])
    >>> geometry.cell[axis,:] += other.cell[axis,:]

    NOTE: The cell prepended is only in the axis that
    is prependend, which means that the other cell directions
    need not conform.

    Parameters
    ----------
    other :
        Other geometry class which needs to be prepended
        If a `Lattice` only the super cell will be extended
    axis :
        Cell direction to which the `other` geometry should be
        prepended
    offset :
        By default appending two structures will simply use the coordinates,
        as is.
        With 'min', the routine will shift both the structures along the cell
        axis of `other` such that they coincide at the first atom, lastly one
        may use a specified offset to manually select how `geometry` is displaced.
        NOTE: That `other.cell[axis, :]` will be added to `offset` if `other` is
        a geometry.

    See Also
    --------
    Geometry.add : add geometries
    Geometry.append : appending geometries
    Geometry.attach : attach a geometry
    Geometry.insert : insert a geometry
    """
    axis = direction(axis)
    if isinstance(other, Lattice):
        # Only extend the supercell.
        xyz = np.copy(geometry.xyz)
        atoms = geometry.atoms.copy()
        lattice = geometry.lattice.prepend(other, axis)
        names = geometry._names.copy()
        if isinstance(offset, str):
            if offset == "none":
                offset = [0, 0, 0]
            else:
                raise ValueError(
                    f"{geometry.__class__.__name__}.prepend requires offset to be (3,) for supercell input"
                )
        xyz += _a.arrayd(offset)

    else:
        # sanitize output
        other = geometry.new(other)
        if isinstance(offset, str):
            offset = offset.lower()
            if offset == "none":
                offset = other.cell[axis, :]
            elif offset == "min":
                # We want to align at the minimum position along the `axis`
                min_f = other.fxyz[:, axis].min()
                min_other_f = np.dot(geometry.xyz, other.icell.T)[:, axis].min()
                offset = other.cell[axis, :] * (1 + min_f - min_other_f)
            else:
                raise ValueError(
                    f"{geometry.__class__.__name__}.prepend requires align keyword to be one of [none, min, (3,)]"
                )
        else:
            offset = other.cell[axis, :] + _a.asarray(offset)

        xyz = np.append(other.xyz, offset + geometry.xyz, axis=0)
        atoms = geometry.atoms.prepend(other.atoms)
        lattice = geometry.lattice.prepend(other.lattice, axis)
        names = other._names.merge(geometry._names, offset=len(other))

    return geometry.__class__(xyz, atoms=atoms, lattice=lattice, names=names)


@register_sisl_dispatch(Geometry, module="sisl")
def add(
    geometry: Geometry,
    other: LatticeOrGeometryLike,
    offset: Coord = (0, 0, 0),
) -> Geometry:
    """Merge two geometries (or a Geometry and Lattice) by adding the two atoms together

    If `other` is a Geometry only the atoms gets added, to also add the supercell vectors
    simply do ``geom.add(other).add(other.lattice)``.

    Parameters
    ----------
    other :
        Other geometry class which is added
    offset :
        offset in geometry of `other` when adding the atoms.
        Otherwise it is the offset of the `geometry` atoms.

    See Also
    --------
    Geometry.append : appending geometries
    Geometry.prepend : prepending geometries
    Geometry.attach : attach a geometry
    Geometry.insert : insert a geometry

    Examples
    --------
    >>> first = Geometry(...)
    >>> second = Geometry(...)
    >>> lattice = Lattice(...)
    >>> added = first.add(second, offset=(0, 0, 2))
    >>> assert np.allclose(added.xyz[:len(first)], first.xyz)
    >>> assert np.allclose(added.xyz[len(first):] - [0, 0, 2], second.xyz)

    """
    if isinstance(other, Lattice):
        xyz = geometry.xyz.copy() + _a.arrayd(offset)
        lattice = geometry.lattice + other
        atoms = geometry.atoms.copy()
        names = geometry._names.copy()
    else:
        other = geometry.new(other)
        xyz = np.append(geometry.xyz, other.xyz + _a.arrayd(offset), axis=0)
        lattice = geometry.lattice.copy()
        atoms = geometry.atoms.add(other.atoms)
        names = geometry._names.merge(other._names, offset=len(geometry))
    return geometry.__class__(xyz, atoms=atoms, lattice=lattice, names=names)


@register_sisl_dispatch(Geometry, module="sisl")
@deprecate_argument(
    "scale_atoms",
    "scale_basis",
    "argument scale_atoms has been deprecated in favor of scale_basis, please update your code.",
    "0.15",
    "0.17",
)
def scale(
    geometry: Geometry,
    scale: CoordOrScalar,
    what: Literal["abc", "xyz"] = "abc",
    scale_basis: bool = True,
) -> Geometry:
    """Scale coordinates and unit-cell to get a new geometry with proper scaling

    Parameters
    ----------
    scale :
       the scale factor for the new geometry (lattice vectors, coordinates
       and the atomic radii are scaled).
    what :

       ``abc``
         Is applied on the corresponding lattice vector and the fractional coordinates.

       ``xyz``
         Is applied *only* to the atomic coordinates.

       If three different scale factors are provided, each will correspond to the
       Cartesian direction/lattice vector.
    scale_basis :
       if true, the atoms basis-sets will be also be scaled.
       The scaling of the basis-sets will be done based on the largest
       scaling factor.
    """
    # Ensure we are dealing with a numpy array
    scale = np.asarray(scale)

    # Scale the supercell
    lattice = geometry.lattice.scale(scale, what=what)

    what = what.lower()
    if what == "xyz":
        # It is faster to rescale coordinates by simply multiplying them by the scale
        xyz = geometry.xyz * scale
        max_scale = scale.max()

    elif what == "abc":
        # Scale the coordinates by keeping fractional coordinates the same
        xyz = geometry.fxyz @ lattice.cell

        if scale_basis:
            # To rescale atoms, we need to know the span of each cartesian coordinate before and
            # after the scaling, and scale the atoms according to the coordinate that has
            # been scaled by the largest factor.
            prev_verts = geometry.lattice.vertices().reshape(8, 3)
            prev_span = prev_verts.max(axis=0) - prev_verts.min(axis=0)
            scaled_verts = lattice.vertices().reshape(8, 3)
            scaled_span = scaled_verts.max(axis=0) - scaled_verts.min(axis=0)
            max_scale = (scaled_span / prev_span).max()
    else:
        raise ValueError(
            f"{geometry.__class__.__name__}.scale got wrong what argument, must be one of abc|xyz"
        )

    if scale_basis:
        # Atoms are rescaled to the maximum scale factor
        atoms = geometry.atoms.scale(max_scale)
    else:
        atoms = geometry.atoms.copy()

    return geometry.__class__(xyz, atoms=atoms, lattice=lattice)
