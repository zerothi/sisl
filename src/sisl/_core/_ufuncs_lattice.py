# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import math
from functools import reduce
from numbers import Integral
from typing import Optional, Union

import numpy as np

import sisl._array as _a
from sisl._ufuncs import register_sisl_dispatch
from sisl.messages import deprecate_argument
from sisl.typing import AnyAxes, CellAxes, CellAxis, Coord, CoordOrScalar, SileLike
from sisl.utils import direction
from sisl.utils.mathematics import fnorm
from sisl.utils.misc import direction

from .lattice import Lattice
from .quaternion import Quaternion

# Nothing gets exposed here
__all__ = []


@register_sisl_dispatch(Lattice, module="sisl")
def copy(lattice: Lattice, cell: Optional[np.ndarray] = None, **kwargs) -> Lattice:
    """A deep copy of the object

    Parameters
    ----------
    cell :
       the new cell parameters
    """
    d = dict()
    for key in ("nsc", "boundary_condition", "origin"):
        if key in kwargs:
            d[key] = kwargs.pop(key)
        else:
            d[key] = getattr(lattice, key).copy()
    if cell is None:
        d["cell"] = lattice.cell.copy()
    else:
        d["cell"] = np.array(cell)
    assert len(kwargs) == 0, f"Unknown arguments passed to Lattice.copy {kwargs.keys()}"

    copy = lattice.__class__(**d)
    # Ensure that the correct super-cell information gets carried through
    if np.allclose(copy.nsc, lattice.nsc) and not np.allclose(
        copy.sc_off, lattice.sc_off
    ):
        copy.sc_off = lattice.sc_off
    return copy


@register_sisl_dispatch(Lattice, module="sisl")
def write(lattice: Lattice, sile: SileLike, *args, **kwargs) -> None:
    """Writes latticey to the `sile` using `sile.write_lattice`

    Parameters
    ----------
    sile :
        a `Sile` object which will be used to write the lattice
        if it is a string it will create a new sile using `get_sile`
    *args, **kwargs:
        Any other args will be passed directly to the
        underlying routine

    See Also
    --------
    Lattice.read : reads a `Lattice` from a given `Sile`/file
    """
    # This only works because, they *must*
    # have been imported previously
    from sisl.io import BaseSile, get_sile

    if isinstance(sile, BaseSile):
        sile.write_lattice(lattice, *args, **kwargs)
    else:
        with get_sile(sile, mode="w") as fh:
            fh.write_lattice(lattice, *args, **kwargs)


@register_sisl_dispatch(Lattice, module="sisl")
def swapaxes(
    lattice: Lattice,
    axes1: AnyAxes,
    axes2: AnyAxes,
    what: Literal["abc", "xyz", "abc+xyz"] = "abc",
) -> Lattice:
    r"""Swaps axes `axes1` and `axes2`

    Swapaxes is a versatile method for changing the order
    of axes elements, either lattice vector order, or Cartesian
    coordinate orders.

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
    what :
       which elements to swap, lattice vectors (``abc``), or
       Cartesian coordinates (``xyz``), or both.
       This argument is only used if the axes arguments are
       ints.

    Examples
    --------

    Swap the first two axes

    >>> sc_ba = sc.swapaxes(0, 1)
    >>> assert np.allclose(sc_ba.cell[(1, 0, 2)], sc.cell)

    Swap the Cartesian coordinates of the lattice vectors

    >>> sc_yx = sc.swapaxes(0, 1, what="xyz")
    >>> assert np.allclose(sc_ba.cell[:, (1, 0, 2)], sc.cell)

    Consecutive swapping:
    1. abc -> bac
    2. bac -> bca

    >>> sc_bca = sc.swapaxes("ab", "bc")
    >>> assert np.allclose(sc_ba.cell[:, (1, 0, 2)], sc.cell)
    """
    if isinstance(axes1, int) and isinstance(axes2, int):
        idx = [0, 1, 2]
        idx[axes1], idx[axes2] = idx[axes2], idx[axes1]

        if "abc" in what or "cell" in what:
            if "xyz" in what:
                axes1 = "abc"[axes1] + "xyz"[axes1]
                axes2 = "abc"[axes2] + "xyz"[axes2]
            else:
                axes1 = "abc"[axes1]
                axes2 = "abc"[axes2]
        elif "xyz" in what:
            axes1 = "xyz"[axes1]
            axes2 = "xyz"[axes2]
        else:
            raise ValueError(
                f"{lattice.__class__.__name__}.swapaxes could not understand 'what' "
                "must contain abc and/or xyz."
            )
    elif (not isinstance(axes1, str)) or (not isinstance(axes2, str)):
        raise ValueError(
            f"{lattice.__class__.__name__}.swapaxes axes arguments must be either all int or all str, not a mix."
        )

    cell = lattice.cell
    nsc = lattice.nsc
    origin = lattice.origin
    bc = lattice.boundary_condition

    if len(axes1) != len(axes2):
        raise ValueError(
            f"{lattice.__class__.__name__}.swapaxes expects axes1 and axes2 to have the same lengeth {len(axes1)}, {len(axes2)}."
        )

    for a, b in zip(axes1, axes2):
        idx = [0, 1, 2]

        aidx = "abcxyz".index(a)
        bidx = "abcxyz".index(b)

        if aidx // 3 != bidx // 3:
            raise ValueError(
                f"{lattice.__class__.__name__}.swapaxes expects axes1 and axes2 to belong to the same category, do not mix lattice vector swaps with Cartesian coordinates."
            )

        if aidx < 3:
            idx[aidx], idx[bidx] = idx[bidx], idx[aidx]
            # we are dealing with lattice vectors
            cell = cell[idx]
            nsc = nsc[idx]
            bc = bc[idx]

        else:
            aidx -= 3
            bidx -= 3
            idx[aidx], idx[bidx] = idx[bidx], idx[aidx]

            # we are dealing with cartesian coordinates
            cell = cell[:, idx]
            origin = origin[idx]
            bc = bc[idx]

    return lattice.copy(
        cell.copy(), nsc=nsc.copy(), origin=origin.copy(), boundary_condition=bc
    )


@register_sisl_dispatch(Lattice, module="sisl")
def rotate(
    lattice: Lattice,
    angle: float,
    v: Union[str, int, Coord],
    rad: bool = False,
    what: Literal["abc", "a", ...] = "abc",
) -> Lattice:
    """Rotates the supercell, in-place by the angle around the vector

    One can control which cell vectors are rotated by designating them
    individually with ``only='[abc]'``.

    Parameters
    ----------
    angle :
         the angle of which the geometry should be rotated
    v     :
         the vector around the rotation is going to happen
         ``[1, 0, 0]`` will rotate in the ``yz`` plane
    rad :
         Whether the angle is in radians (True) or in degrees (False)
    what :
         only rotate the designated cell vectors.
    """
    if isinstance(v, Integral):
        v = direction(v, abc=lattice.cell, xyz=np.diag([1, 1, 1]))
    elif isinstance(v, str):
        v = reduce(
            lambda a, b: a + direction(b, abc=lattice.cell, xyz=np.diag([1, 1, 1])),
            v,
            0,
        )
    # flatten => copy
    vn = _a.asarrayd(v).flatten()
    vn /= fnorm(vn)
    q = Quaternion(angle, vn, rad=rad)
    q /= q.norm()  # normalize the quaternion
    cell = np.copy(lattice.cell)
    idx = []
    for i, d in enumerate("abc"):
        if d in what:
            idx.append(i)
    if idx:
        cell[idx] = q.rotate(lattice.cell[idx])
    return lattice.copy(cell)


@register_sisl_dispatch(Lattice, module="sisl")
def add(lattice: Lattice, other: LatticeLike) -> Lattice:
    """Add two supercell lattice vectors to each other

    Parameters
    ----------
    other :
       the lattice vectors of the other supercell to add
    """
    if not isinstance(other, Lattice):
        other = Lattice.new(other)
    cell = lattice.cell + other.cell
    origin = lattice.origin + other.origin
    nsc = np.where(lattice.nsc > other.nsc, lattice.nsc, other.nsc)
    return lattice.copy(cell, nsc=nsc, origin=origin)


@register_sisl_dispatch(Lattice, module="sisl")
def tile(lattice: Lattice, reps: int, axis: CellAxis) -> Lattice:
    """Extend the unit-cell `reps` times along the `axis` lattice vector

    Notes
    -----
    This is *exactly* equivalent to the `repeat` routine.

    Parameters
    ----------
    reps :
        number of times the unit-cell is repeated along the specified lattice vector
    axis :
        the lattice vector along which the repetition is performed
    """
    axis = direction(axis)
    cell = np.copy(lattice.cell)
    nsc = np.copy(lattice.nsc)
    cell[axis] *= reps
    # Only reduce the size if it is larger than 5
    if nsc[axis] > 3 and reps > 1:
        # This is number of connections for the primary cell
        h_nsc = nsc[axis] // 2
        # The new number of supercells will then be
        nsc[axis] = max(1, int(math.ceil(h_nsc / reps))) * 2 + 1
    return lattice.copy(cell, nsc=nsc)


@register_sisl_dispatch(Lattice, module="sisl")
def repeat(lattice: Lattice, reps: int, axis: CellAxis) -> Lattice:
    """Extend the unit-cell `reps` times along the `axis` lattice vector

    Notes
    -----
    This is *exactly* equivalent to the `tile` routine.

    Parameters
    ----------
    reps :
        number of times the unit-cell is repeated along the specified lattice vector
    axis :
        the lattice vector along which the repetition is performed
    """
    return lattice.tile(reps, axis)


@register_sisl_dispatch(Lattice, module="sisl")
def untile(lattice: Lattice, reps: int, axis: CellAxis) -> Lattice:
    """Reverses a `Lattice.tile` and returns the segmented version

    Notes
    -----
    Untiling will not correctly re-calculate nsc since it has no
    knowledge of connections.

    See Also
    --------
    Lattice.tile : opposite of this method
    """
    axis = direction(axis)
    cell = np.copy(lattice.cell)
    cell[axis] /= reps
    return lattice.copy(cell)


Lattice.unrepeat = untile


@register_sisl_dispatch(Lattice, module="sisl")
def append(lattice: Lattice, other, axis: CellAxis) -> Lattice:
    """Appends other `Lattice` to this grid along axis"""
    axis = direction(axis)
    cell = np.copy(lattice.cell)
    cell[axis] += other.cell[axis]
    # TODO fix nsc here
    return lattice.copy(cell)


@register_sisl_dispatch(Lattice, module="sisl")
def prepend(lattice: Lattice, other, axis: CellAxis) -> Lattice:
    """Prepends other `Lattice` to this grid along axis

    For a `Lattice` object this is equivalent to `append`.
    """
    return other.append(lattice, axis)


@register_sisl_dispatch(Lattice, module="sisl")
def center(lattice: Lattice, axes: CellAxes = (0, 1, 2)) -> np.ndarray:
    """Returns center of the `Lattice`, possibly with respect to axes"""
    if axes is None:
        return lattice.cell.sum(0) * 0.5
    if isinstance(axes, Integral):
        axes = [axes]
    axes = list(map(direction, axes))
    return lattice.cell[axes].sum(0) * 0.5


@register_sisl_dispatch(Lattice, module="sisl")
def scale(
    lattice: Lattice, scale: CoordOrScalar, what: Literal["abc", "xyz"] = "abc"
) -> Lattice:
    """Scale lattice vectors

    Does not scale `origin`.

    Parameters
    ----------
    scale :
       the scale factor for the new lattice vectors.
    what:
       If three different scale factors are provided, whether each scaling factor
       is to be applied on the corresponding lattice vector ("abc") or on the
       corresponding cartesian coordinate ("xyz").
    """
    what = what.lower()
    if what == "abc":
        return lattice.copy((lattice.cell.T * scale).T)
    if what == "xyz":
        return lattice.copy(lattice.cell * scale)
    raise ValueError(
        f"{lattice.__class__.__name__}.scale argument what='{what}' is not in ['abc', 'xyz']."
    )
