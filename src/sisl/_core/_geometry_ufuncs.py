# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from functools import reduce
from numbers import Integral
from typing import TYPE_CHECKING, Optional, Union

import numpy as np

import sisl._array as _a
from sisl._ufuncs import register_sisl_dispatch
from sisl.messages import deprecate_argument, warn
from sisl.utils import direction
from sisl.utils.mathematics import fnorm

from .geometry import Geometry
from .lattice import Lattice
from .quaternion import Quaternion

AtomsArgument = "AtomsArgument"
LatticeOrGeometryLike = "LatticeOrGeometryLike"
Coord = "Coord"
CoordOrScalar = "CoordOrScalar"
if TYPE_CHECKING:
    from sisl.typing import AtomsArgument, Coord, CoordOrScalar, LatticeOrGeometryLike

# Nothing gets exposed here
__all__ = []


@register_sisl_dispatch(module="sisl")
def copy(geometry: Geometry) -> Geometry:
    """Create a new object with the same content (a copy)."""
    g = geometry.__class__(
        np.copy(geometry.xyz),
        atoms=geometry.atoms.copy(),
        lattice=geometry.lattice.copy(),
    )
    g._names = geometry.names.copy()
    return g


@register_sisl_dispatch(module="sisl")
def tile(geometry: Geometry, reps: int, axis: int) -> Geometry:
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

    See Also
    --------
    repeat : equivalent but different ordering of final structure
    untile : opposite method of this
    """
    if reps < 1:
        raise ValueError(
            f"{geometry.__class__.__name__}.tile requires a repetition above 0"
        )

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


@register_sisl_dispatch(module="sisl")
def untile(
    geometry: Geometry,
    reps: int,
    axis: int,
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

    See Also
    --------
    tile : opposite method of this
    """
    if geometry.na % reps != 0:
        raise ValueError(
            f"{geometry.__class__.__name__}.untile "
            f"cannot be cut into {reps} different "
            "pieces. Please check your geometry and input."
        )
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


@register_sisl_dispatch(module="sisl")
def repeat(geometry: Geometry, reps: int, axis: int) -> Geometry:
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

    See Also
    --------
    tile : equivalent but different ordering of final structure
    """
    if reps < 1:
        raise ValueError(
            f"{geometry.__class__.__name__}.repeat requires a repetition above 0"
        )

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


@register_sisl_dispatch(module="sisl")
def unrepeat(geometry: Geometry, reps: int, axis: int, *args, **kwargs) -> Geometry:
    """Unrepeats the geometry similarly as `untile`

    This is the opposite of `repeat`.

    Please see `untile` for argument details.
    This algorithm first re-arranges atoms as though they had been tiled,
    then subsequently calls `untile`.

    See Also
    --------
    repeat : opposite method of this
    """
    atoms = np.arange(geometry.na).reshape(-1, reps).T.ravel()
    return geometry.sub(atoms).untile(reps, axis, *args, **kwargs)


@register_sisl_dispatch(module="sisl")
def translate(
    geometry: Geometry,
    v: CoordOrScalar,
    atoms: Optional[AtomsArgument] = None,
    cell: bool = False,
) -> Geometry:
    """Translates the geometry by `v`

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
    cell :
         If True the supercell also gets enlarged by the vector
    """
    g = geometry.copy()
    if atoms is None:
        g.xyz += np.asarray(v, g.xyz.dtype)
    else:
        g.xyz[geometry._sanitize_atoms(atoms).ravel(), :] += np.asarray(v, g.xyz.dtype)
    if cell:
        g.set_lattice(g.lattice.translate(v))
    return g


# simple copy...
Geometry.move = Geometry.translate


@register_sisl_dispatch(module="sisl")
def sub(geometry: Geometry, atoms: AtomsArgument) -> Geometry:
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
    remove : the negative of this routine, i.e. remove a subset of atoms
    """
    atoms = geometry.sc2uc(atoms)
    return geometry.__class__(
        geometry.xyz[atoms, :].copy(),
        atoms=geometry.atoms.sub(atoms),
        lattice=geometry.lattice.copy(),
    )


@register_sisl_dispatch(module="sisl")
def remove(geometry: Geometry, atoms: AtomsArgument) -> Geometry:
    """Remove atoms from the geometry.

    Indices passed *MUST* be unique.

    Negative indices are wrapped and thus works.

    Parameters
    ----------
    atoms :
        indices/boolean of all atoms to be removed

    See Also
    --------
    sub : the negative of this routine, i.e. retain a subset of atoms
    """
    atoms = geometry.sc2uc(atoms)
    if atoms.size == 0:
        return geometry.copy()
    atoms = np.delete(_a.arangei(geometry.na), atoms)
    return geometry.sub(atoms)


@register_sisl_dispatch(module="sisl")
@deprecate_argument(
    "only",
    "what",
    "argument only has been deprecated in favor of what, please update your code.",
    "0.14.0",
)
def rotate(
    geometry: Geometry,
    angle: float,
    v: Union[str, int, Coord],
    origin: Optional[Union[int, Coord]] = None,
    atoms: Optional[AtomsArgument] = None,
    rad: bool = False,
    what: Optional[str] = None,
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
         to a `geometry.move(-origin).rotate(...).move(origin)`.
         If this is an `int` it corresponds to the atomic index.
    atoms :
         only rotate the given atomic indices, if not specified, all
         atoms will be rotated.
    rad :
         if ``True`` the angle is provided in radians (rather than degrees)
    what : {'xyz', 'abc', 'abc+xyz', <or combinations of "xyzabc">}
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
    Lattice.rotate : rotation passed to the contained supercell
    """
    if origin is None:
        origin = [0.0, 0.0, 0.0]
    elif isinstance(origin, Integral):
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
        atoms = geometry.sc2uc(atoms, unique=True)

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


@register_sisl_dispatch(module="sisl")
def swapaxes(
    geometry: Geometry,
    axes_a: Union[int, str],
    axes_b: Union[int, str],
    what: str = "abc",
) -> Geometry:
    """Swap the axes components by either lattice vectors (only cell), or Cartesian coordinates

    See `Lattice.swapaxes` for details.

    Parameters
    ----------
    axes_a :
       the old axis indices (or labels if `str`)
       A string will translate each character as a specific
       axis index.
       Lattice vectors are denoted by ``abc`` while the
       Cartesian coordinates are denote by ``xyz``.
       If `str`, then `what` is not used.
    axes_b :
       the new axis indices, same as `axes_a`
       old axis indices (or labels)
    what : {'abc', 'xyz', 'abc+xyz'}
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
    lattice = geometry.lattice.swapaxes(axes_a, axes_b, what)

    if isinstance(axes_a, int) and isinstance(axes_b, int):
        if "xyz" in what:
            axes_a = "xyz"[axes_a]
            axes_b = "xyz"[axes_b]
        else:
            axes_a = ""
            axes_b = ""

    # only thing we are going to swap is the coordinates
    idx = [0, 1, 2]
    for a, b in zip(axes_a, axes_b):
        aidx = "xyzabc".index(a)
        bidx = "xyzabc".index(b)
        if aidx < 3:
            idx[aidx], idx[bidx] = idx[bidx], idx[aidx]

    return geometry.__class__(
        geometry.xyz[:, idx].copy(), atoms=geometry.atoms.copy(), lattice=lattice
    )


@register_sisl_dispatch(module="sisl")
def center(
    geometry: Geometry, atoms: Optional[AtomsArgument] = None, what: str = "xyz"
) -> np.ndarray:
    """Returns the center of the geometry

    By specifying `what` one can control whether it should be:

    * ``cop|xyz|position``: Center of coordinates (default)
    * ``mm:xyz`` or ``mm(xyz)``: Center of minimum/maximum of coordinates
    * ``com|mass``: Center of mass
    * ``com:pbc|mass:pbc``: Center of mass using periodicity, if the point 0, 0, 0 is returned it
        may likely be because of a completely periodic system with no true center of mass
    * ``cou|cell``: Center of cell

    Parameters
    ----------
    atoms :
        list of atomic indices to find center of
    what : {'xyz', 'mm:xyz', 'mass', 'mass:pbc', 'cell'}
        determine which center to calculate
    """
    what = what.lower()
    if what in ("cou", "cell", "lattice"):
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

    if what in ("com", "mass"):
        mass = g.mass
        return mass @ g.xyz / mass.sum()

    if what in ("mm:xyz", "mm(xyz)"):
        return (g.xyz.min(0) + g.xyz.max(0)) / 2

    if what in ("cop", "xyz", "position"):
        return np.mean(g.xyz, axis=0)

    raise ValueError(
        f"{geometry.__class__.__name__}.center could not understand option 'what' got {what}"
    )


@register_sisl_dispatch(module="sisl")
def append(
    geometry: Geometry,
    other: LatticeOrGeometryLike,
    axis: int,
    offset: Union[str, Coord] = "none",
) -> Geometry:
    """Appends two structures along `axis`

    This will automatically add the ``geometry.cell[axis,:]`` to all atomic
    coordiates in the `other` structure before appending.

    The basic algorithm is this:

    >>> oxa = other.xyz + geometry.cell[axis,:][None,:]
    >>> geometry.xyz = np.append(geometry.xyz,oxa)
    >>> geometry.cell[axis,:] += other.cell[axis,:]

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
    offset : {'none', 'min', (3,)}
        By default appending two structures will simply use the coordinates,
        as is.
        With 'min', the routine will shift both the structures along the cell
        axis of `geometry` such that they coincide at the first atom, lastly one
        may use a specified offset to manually select how `other` is displaced.
        NOTE: That `geometry.cell[axis, :]` will be added to `offset` if `other` is
        a geometry.

    See Also
    --------
    add : add geometries
    prepend : prending geometries
    attach : attach a geometry
    insert : insert a geometry
    """
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


@register_sisl_dispatch(module="sisl")
def prepend(
    geometry: Geometry,
    other: LatticeOrGeometryLike,
    axis: int,
    offset: Union[str, Coord] = "none",
) -> Geometry:
    """Prepend two structures along `axis`

    This will automatically add the ``geometry.cell[axis,:]`` to all atomic
    coordiates in the `other` structure before appending.

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
    offset : {'none', 'min', (3,)}
        By default appending two structures will simply use the coordinates,
        as is.
        With 'min', the routine will shift both the structures along the cell
        axis of `other` such that they coincide at the first atom, lastly one
        may use a specified offset to manually select how `geometry` is displaced.
        NOTE: That `other.cell[axis, :]` will be added to `offset` if `other` is
        a geometry.

    See Also
    --------
    add : add geometries
    append : appending geometries
    attach : attach a geometry
    insert : insert a geometry
    """
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


@register_sisl_dispatch(module="sisl")
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
    append : appending geometries
    prepend : prending geometries
    attach : attach a geometry
    insert : insert a geometry

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
