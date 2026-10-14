# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from collections.abc import Sequence
from typing import Callable, Optional, Union

import numpy as np

from sisl._category import Category
from sisl._core import Atom, AtomCategory, Geometry, PeriodicTable
from sisl._internal import set_module
from sisl.messages import warn
from sisl.typing import AtomsIndex

from ._category._neighbors import AtomNeighbors
from ._category.base import _neighbor_pairs

__all__ = ["Saturate", "saturate"]


def _perpendicular(u: np.ndarray, ref: Optional[np.ndarray]) -> np.ndarray:
    """Unit vector perpendicular to the unit vector `u`, in the plane of `u` and `ref`"""
    if ref is not None:
        p = ref - ref.dot(u) * u
        norm = np.linalg.norm(p)
        if norm > 1e-6:
            return p / norm
    e = np.zeros(3)
    e[np.argmin(np.fabs(u))] = 1.0
    p = e - e.dot(u) * u
    return p / np.linalg.norm(p)


def _bond_directions(
    u: np.ndarray, coordination: int, ref: Optional[np.ndarray]
) -> np.ndarray:
    """Unit vectors of the missing bonds for the ideal geometry of `coordination`

    Parameters
    ----------
    u :
       unit vectors of the present bonds, shape ``(n, 3)``
    coordination :
       the number of bonds after saturation
    ref :
       a direction which defines the plane of new bonds when only one bond is present
    """
    n = len(u)
    if n == coordination - 1:
        s = -u.sum(0)
        norm = np.linalg.norm(s)
        if norm < 1e-6:
            return _perpendicular(u[0], ref).reshape(1, 3)
        return (s / norm).reshape(1, 3)

    if coordination == 3 and n == 1:
        p = _perpendicular(u[0], ref)
        s = np.sqrt(3) / 2
        return np.array([-0.5 * u[0] + s * p, -0.5 * u[0] - s * p])

    if coordination == 4 and n == 2:
        b = -(u[0] + u[1])
        norm = np.linalg.norm(b)
        if norm < 1e-6:
            b = _perpendicular(u[0], ref)
        else:
            b = b / norm
        p = np.cross(u[0], b)
        p /= np.linalg.norm(p)
        half = np.arccos(-1 / 3) / 2
        return np.array(
            [
                np.cos(half) * b + np.sin(half) * p,
                np.cos(half) * b - np.sin(half) * p,
            ]
        )

    if coordination == 4 and n == 1:
        p = _perpendicular(u[0], ref)
        q = np.cross(u[0], p)
        # staggered with respect to `ref`
        phi = np.array([1, 3, 5]) * np.pi / 3
        st = np.sqrt(8 / 9)
        return (
            -u[0] / 3
            + st * np.cos(phi).reshape(-1, 1) * p
            + st * np.sin(phi).reshape(-1, 1) * q
        )

    raise ValueError(
        f"Saturate cannot place {coordination - n} bonds on an atom with {n} bonds "
        f"and coordination {coordination}. Only coordinations 2, 3 and 4 are supported."
    )


@set_module("sisl.geom")
class Saturate:
    r"""Add an atom for each missing bond of an atom

    The missing bonds get the directions of the ideal geometry: linear (2 bonds),
    trigonal planar (3 bonds) or tetrahedral (4 bonds).
    When an atom has only one bond, the plane of the new bonds follows
    a bond of the neighbor (planar for 3 bonds, staggered for 4 bonds).

    Parameters
    ----------
    atom :
       the atom to add for each missing bond
    bond :
       the bond length between the saturated atom and the added atom [Ang].
       Defaults to the sum of the radii (`PeriodicTable.radius`).
       This is an approximate value (e.g. 1.20 Ang for C-H), relax the structure
       or specify the bond length.
    coordination :
       number of bonds of a saturated atom.
       Defaults to the most frequent number of bonds of the atoms
       with the same atomic number in the geometry.
    R :
       the bond range, see `AtomNeighbors`.
       Defaults to ``(0.01, atom.maxR())`` of the saturated atom.

    See Also
    --------
    saturate : apply saturations to a geometry
    """

    def __init__(
        self,
        atom: Union[Atom, str, int],
        bond: Optional[float] = None,
        coordination: Optional[int] = None,
        R: Optional[Union[float, tuple[float, float], Callable]] = None,
    ):
        if not isinstance(atom, Atom):
            atom = Atom(atom)
        self.atom = atom
        self.bond = bond
        self.coordination = coordination
        self._neighbors = AtomNeighbors(R=R)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.atom.symbol}, bond={self.bond}, "
            f"coordination={self.coordination})"
        )

    def _bonds(self, geometry: Geometry, cache: dict):
        """Bonds ``(I, J, rij)`` sorted by ``I`` without the atom itself"""
        lo, hi = self._neighbors._R_window(geometry)
        I, J, isc, rij, d = _neighbor_pairs(geometry, hi.max(initial=-1.0), cache)
        keep = (lo[I] < d) & (d <= hi[I]) & ~((I == J) & ~isc.any(1))
        idx = np.flatnonzero(keep)
        idx = idx[np.argsort(I[idx], kind="stable")]
        return I[idx], J[idx], rij[idx]

    def _coordination(self, geometry: Geometry, n: np.ndarray, atoms: np.ndarray):
        """Target coordination of each atom in `atoms`"""
        if self.coordination is not None:
            return np.full(len(atoms), self.coordination)
        Z = geometry.atoms.Z
        target = {}
        for z in np.unique(Z[atoms]):
            count = np.bincount(n[Z == z])
            # most frequent, the largest in case of a tie
            target[z] = np.flatnonzero(count == count.max())[-1]
        return np.array([target[z] for z in Z[atoms]])

    def __call__(self, geometry: Geometry, atoms: np.ndarray, cache: dict):
        """Atoms and coordinates to add for the saturation of `atoms`

        Returns
        -------
        list of Atom
        numpy.ndarray
           coordinates of the added atoms, shape ``(len(list), 3)``
        """
        atoms = np.asarray(atoms).ravel()
        if len(atoms) == 0:
            return [], np.empty((0, 3))

        _, hi = self._neighbors._R_window(geometry)
        if np.any(hi[atoms] <= 0):
            raise ValueError(
                f"{self!r} found atoms without a bond range (no orbitals), "
                "specify the R argument."
            )

        I, J, rij = self._bonds(geometry, cache)
        n = np.bincount(I, minlength=geometry.na)
        ptr = np.concatenate(([0], np.cumsum(n)))
        coordination = self._coordination(geometry, n, atoms)

        pt = PeriodicTable()
        xyz = []
        for ia, c in zip(atoms, coordination):
            if n[ia] >= c:
                continue
            if n[ia] == 0:
                warn(f"{self!r} skips atom {ia}, it has no bonds.")
                continue
            r = rij[ptr[ia] : ptr[ia + 1]]
            u = r / np.linalg.norm(r, axis=1).reshape(-1, 1)

            ref = None
            if n[ia] == 1:
                # a bond of the neighbor, other than the bond back to `ia`
                ja = J[ptr[ia]]
                rj = rij[ptr[ja] : ptr[ja + 1]]
                rj = rj[np.linalg.norm(rj + r[0], axis=1) > 1e-6]
                if len(rj) > 0:
                    ref = rj[0]

            bond = self.bond
            if bond is None:
                bond = pt.radius(geometry.atoms.Z[ia]) + pt.radius(self.atom.Z)
            directions = _bond_directions(u, c, ref)
            xyz.append(geometry.xyz[ia] + bond * directions)

        if len(xyz) == 0:
            return [], np.empty((0, 3))
        xyz = np.concatenate(xyz)
        return [self.atom] * len(xyz), xyz


SelectorLike = Union[Category, dict, AtomsIndex]
SaturateLike = Union[Saturate, Atom, str, int]


def _selector_mask(geometry: Geometry, selector: SelectorLike, cache: dict):
    if isinstance(selector, dict):
        selector = AtomCategory(**selector)
    if isinstance(selector, Category):
        return selector._mask(geometry, cache=cache)
    mask = np.zeros(geometry.na, dtype=bool)
    mask[geometry._sanitize_atoms(selector).ravel()] = True
    return mask


@set_module("sisl.geom")
def saturate(
    geometry: Geometry,
    rules: Sequence[tuple[SelectorLike, SaturateLike]],
) -> Geometry:
    r"""Saturate missing bonds of atoms with other atoms

    Each rule is a pair ``(selector, saturation)``.
    An atom gets the saturation of the first rule that selects it.
    All rules use the input geometry, thus the added atoms do not change
    the selection of other rules.

    Parameters
    ----------
    geometry :
       the geometry to saturate
    rules :
       pairs of ``(selector, saturation)``.
       A selector is a category, a `dict` for `AtomCategory`, or atomic indices.
       A saturation is a `Saturate`, or an atom (``"H"`` equals ``Saturate("H")``).

    Returns
    -------
    Geometry
       the input geometry with the added atoms at the end, in the order of the rules

    Examples
    --------
    Saturate the edges of a zigzag nanoribbon with hydrogen:

    >>> ribbon = sisl.geom.zgnr(4)
    >>> edge = AtomNeighbors(max=2, R=1.5)
    >>> saturated = saturate(ribbon, [(edge, Saturate("H", bond=1.09, R=1.5))])
    >>> saturated.na - ribbon.na
    2

    The first rule that selects an atom is used:

    >>> rules = [
    ...     (edge & AtomOdd(), Saturate("F", bond=1.35, R=1.5)),
    ...     (edge, Saturate("H", bond=1.09, R=1.5)),
    ... ]
    >>> saturate(ribbon, rules).atoms.Z[ribbon.na:].tolist()
    [9, 1]

    See Also
    --------
    Saturate : the bond geometry and the defaults
    """
    cache = {}
    taken = np.zeros(geometry.na, dtype=bool)
    atoms = []
    xyz = []
    for selector, saturation in rules:
        if not isinstance(saturation, Saturate):
            saturation = Saturate(saturation)
        mask = _selector_mask(geometry, selector, cache) & ~taken
        taken |= mask
        add_atoms, add_xyz = saturation(geometry, np.flatnonzero(mask), cache)
        atoms.extend(add_atoms)
        xyz.append(add_xyz)

    if len(atoms) == 0:
        return geometry.copy()
    return geometry.add(
        Geometry(np.concatenate(xyz), atoms=atoms, lattice=geometry.lattice.copy())
    )
