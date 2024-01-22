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
from sisl.typing import Coord, SileLike
from sisl.utils import direction
from sisl.utils.mathematics import fnorm

from .lattice import Lattice
from .quaternion import Quaternion

# Nothing gets exposed here
__all__ = []


@register_sisl_dispatch(Lattice, module="sisl")
def copy(lattice: Lattice, cell=None, origin: Optional[Coord] = None) -> Lattice:
    """A deepcopy of the object

    Parameters
    ----------
    cell : array_like
       the new cell parameters
    origin : array_like
       the new origin
    """
    d = dict()
    d["nsc"] = lattice.nsc.copy()
    d["boundary_condition"] = lattice.boundary_condition.copy()
    if origin is None:
        d["origin"] = lattice.origin.copy()
    else:
        d["origin"] = origin
    if cell is None:
        d["cell"] = lattice.cell.copy()
    else:
        d["cell"] = np.array(cell)

    copy = lattice.__class__(**d)
    # Ensure that the correct super-cell information gets carried through
    if not np.allclose(copy.sc_off, lattice.sc_off):
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
    read : reads a `Lattice` from a given `Sile`/file
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
    axes_a: Union[int, str],
    axes_b: Union[int, str],
    what: str = "abc",
) -> Lattice:
    r"""Swaps axes `axes_a` and `axes_b`

    Swapaxes is a versatile method for changing the order
    of axes elements, either lattice vector order, or Cartesian
    coordinate orders.

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
    what : {"abc", "xyz", "abc+xyz"}
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
    if isinstance(axes_a, int) and isinstance(axes_b, int):
        idx = [0, 1, 2]
        idx[axes_a], idx[axes_b] = idx[axes_b], idx[axes_a]

        if "abc" in what or "cell" in what:
            if "xyz" in what:
                axes_a = "abc"[axes_a] + "xyz"[axes_a]
                axes_b = "abc"[axes_b] + "xyz"[axes_b]
            else:
                axes_a = "abc"[axes_a]
                axes_b = "abc"[axes_b]
        elif "xyz" in what:
            axes_a = "xyz"[axes_a]
            axes_b = "xyz"[axes_b]
        else:
            raise ValueError(
                f"{lattice.__class__.__name__}.swapaxes could not understand 'what' "
                "must contain abc and/or xyz."
            )
    elif (not isinstance(axes_a, str)) or (not isinstance(axes_b, str)):
        raise ValueError(
            f"{lattice.__class__.__name__}.swapaxes axes arguments must be either all int or all str, not a mix."
        )

    cell = lattice.cell
    nsc = lattice.nsc
    origin = lattice.origin
    bc = lattice.boundary_condition

    if len(axes_a) != len(axes_b):
        raise ValueError(
            f"{lattice.__class__.__name__}.swapaxes expects axes_a and axes_b to have the same lengeth {len(axes_a)}, {len(axes_b)}."
        )

    for a, b in zip(axes_a, axes_b):
        idx = [0, 1, 2]

        aidx = "abcxyz".index(a)
        bidx = "abcxyz".index(b)

        if aidx // 3 != bidx // 3:
            raise ValueError(
                f"{lattice.__class__.__name__}.swapaxes expects axes_a and axes_b to belong to the same category, do not mix lattice vector swaps with Cartesian coordinates."
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

    return lattice.__class__(
        cell.copy(), nsc=nsc.copy(), origin=origin.copy(), boundary_condition=bc
    )


@register_sisl_dispatch(Lattice, module="sisl")
@deprecate_argument(
    "only",
    "what",
    "argument only has been deprecated in favor of what, please update your code.",
    "0.14.0",
)
def rotate(
    lattice: Lattice,
    angle: float,
    v: Union[str, int, Coord],
    rad: bool = False,
    what: str = "abc",
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
    what : combination of ``"abc"``, str, optional
         only rotate the designated cell vectors.
    rad : bool, optional
         Whether the angle is in radians (True) or in degrees (False)
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
def add(lattice: Lattice, other) -> Lattice:
    """Add two supercell lattice vectors to each other

    Parameters
    ----------
    other : Lattice, array_like
       the lattice vectors of the other supercell to add
    """
    if not isinstance(other, Lattice):
        other = Lattice(other)
    cell = lattice.cell + other.cell
    origin = lattice.origin + other.origin
    nsc = np.where(lattice.nsc > other.nsc, lattice.nsc, other.nsc)
    return lattice.__class__(cell, nsc=nsc, origin=origin)


@register_sisl_dispatch(Lattice, module="sisl")
def tile(lattice: Lattice, reps: int, axis: int) -> Lattice:
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
    cell = np.copy(lattice.cell)
    nsc = np.copy(lattice.nsc)
    origin = np.copy(lattice.origin)
    cell[axis] *= reps
    # Only reduce the size if it is larger than 5
    if nsc[axis] > 3 and reps > 1:
        # This is number of connections for the primary cell
        h_nsc = nsc[axis] // 2
        # The new number of supercells will then be
        nsc[axis] = max(1, int(math.ceil(h_nsc / reps))) * 2 + 1
    return lattice.__class__(cell, nsc=nsc, origin=origin)


@register_sisl_dispatch(Lattice, module="sisl")
def repeat(lattice: Lattice, reps: int, axis: int) -> Lattice:
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
def untile(lattice: Lattice, reps: int, axis: int) -> Lattice:
    """Reverses a `Lattice.tile` and returns the segmented version

    Notes
    -----
    Untiling will not correctly re-calculate nsc since it has no
    knowledge of connections.

    See Also
    --------
    tile : opposite of this method
    """
    cell = np.copy(lattice.cell)
    cell[axis] /= reps
    return lattice.copy(cell)


Lattice.unrepeat = untile


@register_sisl_dispatch(Lattice, module="sisl")
def append(lattice: Lattice, other, axis: int) -> Lattice:
    """Appends other `Lattice` to this grid along axis"""
    cell = np.copy(lattice.cell)
    cell[axis] += other.cell[axis]
    # TODO fix nsc here
    return lattice.copy(cell)


@register_sisl_dispatch(Lattice, module="sisl")
def prepend(lattice: Lattice, other, axis: int) -> Lattice:
    """Prepends other `Lattice` to this grid along axis

    For a `Lattice` object this is equivalent to `append`.
    """
    return other.append(lattice, axis)


@register_sisl_dispatch(Lattice, module="sisl")
def center(lattice: Lattice, axis: Optional[int] = None) -> np.ndarray:
    """Returns center of the `Lattice`, possibly with respect to an axis"""
    if axis is None:
        return lattice.cell.sum(0) * 0.5
    return lattice.cell[axis] * 0.5
