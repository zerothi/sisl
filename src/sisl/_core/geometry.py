# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# To check for integers
from __future__ import annotations

import logging
import warnings
from collections import OrderedDict
from collections.abc import Iterator, Sequence
from functools import singledispatchmethod
from itertools import product
from math import acos
from numbers import Integral, Real
from pathlib import Path
from typing import Optional, Union

import numpy as np
import numpy.typing as npt
from numpy import (
    bool_,
    ceil,
    concatenate,
    diff,
    dot,
    floor,
    int32,
    ndarray,
    sqrt,
    square,
    tile,
    unique,
)
from scipy.sparse import csr_matrix

import sisl._array as _a
from sisl._category import Category, GenericCategory
from sisl._dispatch_class import _Dispatchs
from sisl._dispatcher import AbstractDispatch, ClassDispatcher, TypeDispatcher
from sisl._help import has_module, isndarray
from sisl._indices import (
    indices_gt_le,
    indices_in_sphere_with_dist,
    indices_le,
    list_index_le,
)
from sisl._internal import set_module
from sisl._lib._argparse import SislHelpFormatter
from sisl._math_small import cross3, is_ascending, xyz_to_spherical_cos_phi
from sisl._namedindex import NamedIndex
from sisl.messages import SislError, deprecate_argument, deprecation, info, warn
from sisl.shape import Cube, Shape, Sphere
from sisl.typing import (
    AtomsIndex,
    AtomsLike,
    CellAxes,
    LatticeLike,
    OrbitalsIndex,
    SileLike,
)
from sisl.utils import (
    angle,
    cmd,
    default_ArgumentParser,
    default_namespace,
    direction,
    listify,
    lstranges,
    str_spec,
    strmap,
)
from sisl.utils.mathematics import fnorm

from .atom import Atom, Atoms
from .lattice import Lattice, LatticeChild
from .orbital import Orbital

__all__ = ["Geometry", "sgeom", "AtomCategory"]

_log = logging.getLogger(__name__)


# It needs to be here otherwise we can't use it in these routines
# Note how we are overwriting the module
@set_module("sisl.geom")
class AtomCategory(Category):
    __slots__ = ()

    @classmethod
    def is_class(cls, name, case=True) -> bool:
        # Strip off `Atom`
        cls_name = cls.__name__[4:]
        if case:
            return cls_name == name
        return cls_name.lower() == name.lower()


@set_module("sisl")
class Geometry(
    LatticeChild,
    _Dispatchs,
    dispatchs=[
        ClassDispatcher("new", obj_getattr="error", instance_dispatcher=TypeDispatcher),
        ClassDispatcher("to", obj_getattr="error", type_dispatcher=None),
    ],
    when_subclassing="copy",
):
    """Holds atomic information, coordinates, species, lattice vectors

    The `Geometry` class holds information regarding atomic coordinates,
    the atomic species, the corresponding lattice-vectors.

    It enables the interaction and conversion of atomic structures via
    simple routine methods.

    All lengths are assumed to be in units of Angstrom, however, as
    long as units are kept same the exact units are irrespective.

    .. code:: python

       >>> square = Geometry([[0.5, 0.5, 0.5]], Atom(1),
       ...                   lattice=Lattice([1, 1, 10], nsc=[3, 3, 1]))
       >>> print(square)
       Geometry{na: 1, no: 1,
        Atoms{species: 1,
         Atom{H, Z: 1, mass(au): 1.00794, maxR: -1.00000,
          Orbital{R: -1.00000, q0: 0.0}
         }: 1,
        },
        maxR: -1.00000,
        Lattice{volume: 1.0000e+01, nsc: [3 3 1]}
       }


    Parameters
    ----------
    xyz :
        atomic coordinates
        ``xyz[i, :]`` is the atomic coordinate of the i'th atom.
    atoms :
        atomic species retrieved from the `PeriodicTable`
    lattice :
        the unit-cell describing the atoms in a periodic
        super-cell

    Examples
    --------

    An atomic cubic lattice of Hydrogen atoms

    >>> xyz = [[0, 0, 0],
    ...        [1, 1, 1]]
    >>> sc = Lattice([2,2,2])
    >>> g = Geometry(xyz, Atom('H'), sc)

    The following estimates the lattice vectors from the
    atomic coordinates, although possible, it is not recommended
    to be used.

    >>> xyz = [[0, 0, 0],
    ...        [1, 1, 1]]
    >>> g = Geometry(xyz, Atom('H'))

    Conversion of geometries to other projects instances can be done via
    sisl's dispatch functionality

    >>> g.to.ase()
    Atoms(...)

    converts to an ASE `ase.Atoms` object.

    See Also
    --------
    Atoms : contained atoms ``self.atoms``
    Atom : contained atoms are each an object of this
    """

    @deprecate_argument(
        "sc",
        "lattice",
        "argument sc has been deprecated in favor of lattice, please update your code.",
        "0.15",
        "0.17",
    )
    def __init__(
        self,
        xyz: npt.ArrayLike,
        atoms: Optional[AtomsLike] = None,
        lattice: Optional[LatticeLike] = None,
        names=None,
    ):
        # Create the geometry coordinate, be aware that we do not copy!
        self.xyz = _a.asarrayd(xyz, order="C").reshape(-1, 3)

        # Default value
        if atoms is None:
            atoms = Atom("H")

        # Create the local Atoms object
        self._atoms = Atoms(atoms, na=self.na)

        # Assign a group specifier
        if isinstance(names, NamedIndex):
            self._names = names.copy()
        else:
            self._names = NamedIndex(names)

        self._init_lattice(lattice)

    def _init_lattice(self, lattice: Optional[LatticeLike]) -> None:
        """Initializes the supercell by *calculating* the size if not supplied

        If the supercell has not been passed we estimate the unit cell size
        by calculating the bond-length in each direction for a square
        Cartesian coordinate system.
        """
        # We still need the *default* super cell for
        # estimating the supercell
        self.set_lattice(lattice)

        if lattice is not None:
            return

        # First create an initial guess for the supercell
        # It HAS to be VERY large to not interact
        closest = self.close(0, R=(0.0, 0.4, 5.0))[2]
        if len(closest) < 1:
            # We could not find any atoms very close,
            # hence we simply return and now it becomes
            # the users responsibility

            # We create a molecule box with +10 A in each direction
            m, M = np.amin(self.xyz, axis=0), np.amax(self.xyz, axis=0) + 10.0
            self.set_lattice(M - m)
            return

        sc_cart = _a.zerosd([3])
        cart = _a.zerosd([3])
        for i in range(3):
            # Initialize cartesian direction
            cart[i] = 1.0

            # Get longest distance between atoms
            max_dist = np.amax(self.xyz[:, i]) - np.amin(self.xyz[:, i])

            dist = self.xyz[closest, :] - self.xyz[0, :][None, :]
            # Project onto the direction
            dd = np.abs(dot(dist, cart))

            # Remove all below .4
            tmp_idx = (dd >= 0.4).nonzero()[0]
            if len(tmp_idx) > 0:
                # We have a success
                # Add the bond-distance in the Cartesian direction
                # to the maximum distance in the same direction
                sc_cart[i] = max_dist + np.amin(dd[tmp_idx])
            else:
                # Default to LARGE array so as no
                # interaction occurs (it may be 2D)
                sc_cart[i] = max(10.0, max_dist)
            cart[i] = 0.0

        # Re-set the supercell to the newly found one
        self.set_lattice(sc_cart)

    @property
    def atoms(self) -> Atoms:
        """Atoms associated with the geometry"""
        return self._atoms

    @property
    def names(self):
        """The named index specifier"""
        return self._names

    @property
    def q0(self) -> float:
        """Total initial charge in this geometry (sum of q0 off all atoms)"""
        return self.atoms.q0.sum()

    @property
    def mass(self) -> ndarray:
        """The mass of all atoms as an array"""
        return self.atoms.mass

    def maxR(self, all: bool = False) -> float:
        """Maximum orbital range of the atoms"""
        return self.atoms.maxR(all)

    @property
    def na(self) -> int:
        """Number of atoms in geometry"""
        return self.xyz.shape[0]

    @property
    def na_s(self) -> int:
        """Number of supercell atoms"""
        return self.na * self.n_s

    def __len__(self) -> int:
        """Number of atoms in geometry in unit cell"""
        return self.na

    @property
    def no(self) -> int:
        """Number of orbitals in unit cell"""
        return self.atoms.no

    @property
    def no_s(self) -> int:
        """Number of supercell orbitals"""
        return self.no * self.n_s

    @property
    def firsto(self) -> npt.NDArray[np.int32]:
        """The first orbital on the corresponding atom"""
        return self.atoms.firsto

    @property
    def lasto(self) -> npt.NDArray[np.int32]:
        """The last orbital on the corresponding atom"""
        return self.atoms.lasto

    @property
    def orbitals(self) -> list[Orbital]:
        """List of orbitals per atom"""
        return self.atoms.orbitals

    ## End size of geometry

    @property
    def fxyz(self) -> npt.NDArray[np.float64]:
        """Returns geometry coordinates in fractional coordinates"""
        return dot(self.xyz, self.icell.T)

    def __setitem__(self, atoms, value):
        """Specify geometry coordinates"""
        if isinstance(atoms, str):
            self.names.add_name(atoms, value)
        elif isinstance(value, str):
            self.names.add_name(value, atoms)

    @singledispatchmethod
    def __getitem__(self, atoms) -> ndarray:
        """Geometry coordinates (allows supercell indices)"""
        return self.axyz(atoms)

    @__getitem__.register
    def _(self, atoms: slice) -> ndarray:
        if atoms.stop is None:
            atoms = atoms.indices(self.na)
        else:
            atoms = atoms.indices(self.na_s)
        return self.axyz(_a.arangei(*atoms))

    @__getitem__.register
    def _(self, atoms: tuple) -> ndarray:
        return self[atoms[0]][..., atoms[1]]

    @singledispatchmethod
    def _sanitize_atoms(self, atoms) -> ndarray:
        """Converts an `atoms` to index under given inputs

        `atoms` may be one of the following:

        - boolean array -> nonzero()[0]
        - name -> self._names[name]
        - `Atom` -> self.atoms.index(atom)
        - range/list/ndarray -> ndarray
        - `...` -> ndarray
        """
        if atoms is None:
            return np.arange(self.na)
        elif atoms is Ellipsis:
            return np.arange(self.na)
        atoms = _a.asarray(atoms)
        if atoms.size == 0:
            return _a.asarrayl([])
        if atoms.dtype == bool_:
            return atoms.nonzero()[0]
        return atoms

    @_sanitize_atoms.register
    def _(self, atoms: ndarray) -> ndarray:
        if atoms.dtype == bool_:
            return np.flatnonzero(atoms)
        return atoms

    @_sanitize_atoms.register
    def _(self, atoms: slice) -> ndarray:
        atoms = atoms.indices(self.na)
        return np.arange(*atoms)

    @_sanitize_atoms.register
    def _(self, atoms: str) -> ndarray:
        return self.names[atoms]

    @_sanitize_atoms.register
    def _(self, atoms: Atom) -> ndarray:
        return self.atoms.index(atoms)

    @_sanitize_atoms.register(AtomCategory)
    @_sanitize_atoms.register(GenericCategory)
    def _(
        self,
        atoms_: Union[AtomCategory, GenericCategory],
        atoms: AtomsIndex = None,
    ) -> ndarray:
        # First do categorization
        cat = atoms_.categorize(self, atoms)

        def m(cat):
            for ia, c in enumerate(cat):
                if c == None:
                    # we are using NullCategory == None
                    pass
                else:
                    yield ia

        return _a.fromiterl(m(cat))

    @_sanitize_atoms.register
    def _(self, atoms_: dict, atoms: AtomsIndex = None) -> ndarray:
        # First do categorization
        return self._sanitize_atoms(AtomCategory.kw(**atoms_), atoms)

    @_sanitize_atoms.register
    def _(self, atoms: Shape) -> ndarray:
        # This is perhaps a bit weird since a shape could
        # extend into the supercell.
        # Since the others only does this for unit-cell atoms
        # then it seems natural to also do that here...
        return atoms.within_index(self.xyz)

    @_sanitize_atoms.register
    def _(self, atoms: bool) -> ndarray:
        if atoms:
            return np.arange(self.na)
        return np.array([], np.int64)

    @singledispatchmethod
    def _sanitize_orbs(self, orbitals) -> ndarray:
        """Converts an `orbital` to index under given inputs

        `orbital` may be one of the following:

        - boolean array -> nonzero()[0]
        - dict -> {atom: sub_orbital}
        """
        if orbitals is None:
            return np.arange(self.no)
        elif orbitals is Ellipsis:
            return np.arange(self.no)
        orbitals = _a.asarray(orbitals)
        if orbitals.size == 0:
            return _a.asarrayl([])
        elif orbitals.dtype == np.bool_:
            return orbitals.nonzero()[0]
        return orbitals

    @_sanitize_orbs.register
    def _(self, orbitals: ndarray) -> ndarray:
        if orbitals.dtype == bool_:
            return np.flatnonzero(orbitals)
        return orbitals

    @_sanitize_orbs.register
    def _(self, orbitals: slice) -> ndarray:
        orbitals = orbitals.indices(self.no)
        return np.arange(*orbitals)

    @_sanitize_orbs.register
    def _(self, orbitals: str) -> ndarray:
        atoms = self._sanitize_atoms(orbitals)
        return self.a2o(atoms, all=True)

    @_sanitize_orbs.register
    def _(self, orbitals: Atom) -> ndarray:
        atoms = self._sanitize_atoms(orbitals)
        return self.a2o(atoms, all=True)

    @_sanitize_orbs.register
    def _(self, orbitals: AtomCategory) -> ndarray:
        atoms = self._sanitize_atoms(orbitals)
        return self.a2o(atoms, all=True)

    @_sanitize_orbs.register
    def _(self, orbitals: Shape) -> ndarray:
        atoms = self._sanitize_atoms(orbitals)
        return self.a2o(atoms, all=True)

    @_sanitize_orbs.register
    def _(self, orbitals: dict) -> ndarray:
        """A dict has atoms as keys"""

        def conv(atom, orbs):
            atom = self._sanitize_atoms(atom)
            return np.add.outer(self.firsto[atom], orbs).ravel()

        return np.concatenate(
            tuple(conv(atom, orbs) for atom, orbs in orbitals.items())
        )

    @_sanitize_orbs.register
    def _(self, orbitals: bool) -> ndarray:
        if orbitals:
            return np.arange(self.no)
        return np.array([], dtype=np.int64)

    def as_primary(
        self, na_primary: int, axes: Sequence[int] = (0, 1, 2), ret_super: bool = False
    ) -> Union[Geometry, tuple[Geometry, Lattice]]:
        """Reduce the geometry to the primary unit-cell comprising `na_primary` atoms

        This will basically try and find the tiling/repetitions required for the geometry to only have
        `na_primary` atoms in the unit cell.

        Parameters
        ----------
        na_primary :
           number of atoms in the primary unit cell
        axes :
           only search the given directions for supercells, default to all directions
        ret_super :
           also return the number of supercells used in each direction

        Returns
        -------
        Geometry
             the primary unit cell
        Lattice
             the tiled supercell numbers used to find the primary unit cell (only if `ret_super` is true)

        Raises
        ------
        SislError
             If the algorithm fails.
        """
        na = len(self)
        if na % na_primary != 0:
            raise ValueError(
                f"{self.__class__.__name__}.as_primary requires the number of atoms to be divisable by the "
                "total number of atoms."
            )

        axes = _a.arrayi(axes)

        n_supercells = len(self) // na_primary
        if n_supercells == 1:
            # Return a copy of self
            if ret_super:
                return self.copy(), self.nsc.copy()
            return self.copy()

        # Now figure out the repetitions along each direction
        fxyz = self.fxyz
        # Move to 0
        fxyz -= fxyz.min(0)
        # Shift a little bit in to account for inaccuracies.
        fxyz += (0.5 - (fxyz.max(0) - fxyz.min(0)) / 2) * 0.01

        # Default guess to 1 along all directions
        supercell = _a.onesi(3)

        n_bin = n_supercells
        while n_bin > 1:
            # Create bins
            bins = np.linspace(0, 1, n_bin + 1)

            # Loop directions where we need to check
            for axis in axes:
                if supercell[axis] != 1:
                    continue

                # A histogram should yield an equal splitting for each bins
                # if the geometry is a n_bin repetition along the i'th direction.
                # Hence if diff == 0 for all elements we have a match.
                diff_bin = np.diff(np.histogram(fxyz[:, axis], bins)[0])

                if diff_bin.sum() == 0:
                    supercell[axis] = n_bin
                    if np.prod(supercell) > n_supercells:
                        # For geometries with more than 1 atom in the primary unit cell
                        # we can get false positives (each layer can be split again)
                        # We will search again the max-value supercell
                        i_max = supercell.argmax()
                        n_bin = supercell[i_max]
                        supercell[i_max] = 1

            # Quick escape if hit the correct number of supercells
            if np.prod(supercell) == n_supercells:
                break

            n_bin -= 1

        # Check that the number of supercells match
        if np.prod(supercell) != n_supercells:
            raise SislError(
                f"{self.__class__.__name__}.as_primary could not determine the optimal supercell."
            )

        # Cut down the supercell (TODO this does not correct the number of supercell connections!)
        lattice = self.lattice.copy()
        for i in range(3):
            lattice = lattice.untile(supercell[i], i)

        # Now we need to find the atoms that are in the primary cell
        # We do this by finding all coordinates within the primary unit-cell
        fxyz = dot(self.xyz, lattice.icell.T)
        # Move to 0 and shift in 0.05 Ang in each direction
        fxyz -= fxyz.min(0)

        # Find minimal distance in each direction
        sc_idx = (supercell > 1).nonzero()[0]
        min_fxyz = _a.zerosd(3)
        for i in sc_idx:
            s_fxyz = np.sort(fxyz[:, i])
            min_fxyz[i] = s_fxyz[(s_fxyz < 1e-4).nonzero()[0][-1] + 1]
        fxyz += min_fxyz * 0.05

        # Find all fractional indices that are below 1
        ind = np.logical_and.reduce(fxyz < 1.0, axis=1).nonzero()[0]

        geom = self.sub(ind)
        geom.set_lattice(lattice)
        if ret_super:
            return geom, supercell
        return geom

    def as_supercell(self) -> Geometry:
        """Create a new geometry equivalent to ``self * self.nsc``, where the indices are ordered as the supercells

        Returns
        -------
        `Geometry`
            the supercell expanded and reordered Geometry
        """
        # Get total number of atoms
        na = len(self)
        # create the big supercell geometry in the simplest (linear) way
        sc = self * self.nsc

        # remove nsc, this supercell should hold all information
        sc.set_nsc([1, 1, 1])

        # get off-set for first atom
        # this is used to correct the indices created after having shifted
        # everything
        f0 = self.fxyz[0]

        # translate the supercell such that the 0, 0, 0 (primary cell)
        # is located at the origin.
        sc = sc.translate(-(self.nsc // 2) @ self.cell)

        # Calculate the translation table such that the ordering in `sc` can
        # be made to look like the `self` supercell indices
        isc_sc = np.rint(sc.xyz[::na] @ self.icell.T - f0).astype(np.int32)
        isc_self = self.a2isc(np.arange(self.n_s) * na)

        def new_sub(isc):
            return (abs(isc_sc - isc).sum(1) == 0).nonzero()[0][0]

        # Create the translation table for the indices
        translate = np.array([new_sub(isc) for isc in isc_self])
        # make sure all atoms are present
        translate = np.repeat(translate * na, na).reshape(-1, na) + np.arange(na)

        # re-arrange the atoms and return
        return sc.sub(translate.ravel())

    def reorder(self) -> None:
        """Reorders atoms according to first occurence in the geometry

        The atoms gets reordered according to their placement in the geometry.
        For instance, if the first atom is the 2nd species in the geometry. Then
        this routine will swap the 2nd and 1st species in the `self.atoms` object.

        Notes
        -----
        This is an in-place operation.
        """
        self._atoms = self.atoms.reorder(inplace=True)

    def reduce(self) -> None:
        """Remove all atoms not currently used in the ``self.atoms`` object

        Notes
        -----
        This is an in-place operation.
        """
        self._atoms = self.atoms.reduce(inplace=True)

    def rij(self, ia: AtomsIndex, ja: AtomsIndex) -> ndarray:
        r"""Distance between atom `ia` and `ja`, atoms can be in super-cell indices

        Returns the distance between two atoms:

        .. math::
            r^{IJ} = |\mathbf r^J - \mathbf r^I|

        Parameters
        ----------
        ia :
           atomic index of first atom
        ja :
           atomic indices
        """
        R = self.Rij(ia, ja)

        if len(R.shape) == 1:
            return (R[0] ** 2.0 + R[1] ** 2 + R[2] ** 2) ** 0.5

        return fnorm(R)

    def Rij(self, ia: AtomsIndex, ja: AtomsIndex) -> ndarray:
        r"""Vector between atom `ia` and `ja`, atoms can be in super-cell indices

        Returns the vector between two atoms:

        .. math::
            \mathbf r^{IJ} = \mathbf r^J - \mathbf r^I

        Parameters
        ----------
        ia :
           atomic index of first atom
        ja :
           atomic indices
        """
        xi = self.axyz(ia)
        xj = self.axyz(ja)

        if isinstance(ja, Integral):
            return xj[:] - xi[:]
        elif np.allclose(xi.shape, xj.shape):
            return xj - xi

        return xj - xi[None, :]

    def orij(self, orbitals1: OrbitalsIndex, orbitals2: OrbitalsIndex) -> ndarray:
        r"""Distance between orbital `orbitals1` and `orbitals2`, orbitals can be in super-cell indices

        Returns the distance between two orbitals:

        .. math::
            r^{ij} = |\mathbf r^j - \mathbf r^i|

        Parameters
        ----------
        orbitals1 :
           orbital index of first orbital
        orbitals2 :
           orbital indices
        """
        return self.rij(self.o2a(orbitals1), self.o2a(orbitals2))

    def oRij(self, orbitals1: OrbitalsIndex, orbitals2: OrbitalsIndex) -> ndarray:
        r"""Vector between orbital `orbitals1` and `orbitals2`, orbitals can be in super-cell indices

        Returns the vector between two orbitals:

        .. math::
            \mathbf r^{ij} = \mathbf r^j - \mathbf r^i

        Parameters
        ----------
        orbitals1 :
           orbital index of first orbital
        orbitals2 :
           orbital indices
        """
        return self.Rij(self.o2a(orbitals1), self.o2a(orbitals2))

    @staticmethod
    def read(sile: SileLike, *args, **kwargs) -> Geometry:
        """Reads geometry from the `Sile` using `Sile.read_geometry`

        Parameters
        ----------
        sile :
            a `Sile` object which will be used to read the geometry
            if it is a string it will create a new sile using `get_sile`.

        See Also
        --------
        write : writes a `Geometry` to a given `Sile`/file
        """
        # This only works because, they *must*
        # have been imported previously
        from sisl.io import BaseSile, get_sile

        if isinstance(sile, BaseSile):
            return sile.read_geometry(*args, **kwargs)
        else:
            with get_sile(sile, mode="r") as fh:
                return fh.read_geometry(*args, **kwargs)

    def __str__(self) -> str:
        """str of the object"""
        s = f"{self.__class__.__name__}{{na: {self.na}, no: {self.no},\n "
        s += str(self.atoms).replace("\n", "\n ")
        if len(self.names) > 0:
            s += ",\n " + str(self.names).replace("\n", "\n ")
        return (
            s
            + ",\n maxR: {0:.5f},\n {1}\n}}".format(
                self.maxR(), str(self.lattice).replace("\n", "\n ")
            )
        ).strip()

    def __repr__(self) -> str:
        """A simple, short string representation."""
        return f"<{self.__module__}.{self.__class__.__name__} na={self.na}, no={self.no}, nsc={self.nsc}>"

    def iter(self) -> Iterator[int]:
        """An iterator over all atomic indices

        This iterator is the same as:

        >>> for ia in range(len(self)):
        ...    <do something>

        or equivalently

        >>> for ia in self:
        ...    <do something>

        See Also
        --------
        iter_species : iterate across indices and atomic species
        iter_orbitals : iterate across atomic indices and orbital indices
        """
        yield from range(len(self))

    __iter__ = iter

    def iter_species(self, atoms: AtomsIndex = None) -> Iterator[int, Atom, int]:
        """Iterator over all atoms (or a subset) and species as a tuple in this geometry

        >>> for ia, a, idx_species in self.iter_species():
        ...     isinstance(ia, int) == True
        ...     isinstance(a, Atom) == True
        ...     isinstance(idx_species, int) == True

        with ``ia`` being the atomic index, ``a`` the `Atom` object, ``idx_species``
        is the index of the specie

        Parameters
        ----------
        atoms :
           only loop on the given atoms, default to all atoms

        See Also
        --------
        iter : iterate over atomic indices
        iter_orbitals : iterate across atomic indices and orbital indices
        """
        if atoms is None:
            for ia in self:
                yield ia, self.atoms[ia], self.atoms.species[ia]
        else:
            for ia in self._sanitize_atoms(atoms).ravel():
                yield ia, self.atoms[ia], self.atoms.species[ia]

    def iter_orbitals(
        self, atoms: AtomsIndex = None, local: bool = True
    ) -> Iterator[int, int]:
        r"""Returns an iterator over all atoms and their associated orbitals

        >>> for ia, io in self.iter_orbitals():

        with ``ia`` being the atomic index, ``io`` the associated orbital index on atom ``ia``.
        Note that ``io`` will start from ``0``.

        Parameters
        ----------
        atoms :
           only loop on the given atoms, default to all atoms
        local :
           whether the orbital index is the global index, or the local index relative to
           the atom it resides on.

        Yields
        ------
        ia
           atomic index
        io
           orbital index

        See Also
        --------
        iter : iterate over atomic indices
        iter_species : iterate across indices and atomic species
        """
        if atoms is None:
            if local:
                for ia, IO in enumerate(zip(self.firsto, self.lasto + 1)):
                    for io in range(IO[1] - IO[0]):
                        yield ia, io
            else:
                for ia, IO in enumerate(zip(self.firsto, self.lasto + 1)):
                    for io in range(IO[0], IO[1]):
                        yield ia, io
        else:
            atoms = self._sanitize_atoms(atoms).ravel()
            if local:
                for ia, io1, io2 in zip(
                    atoms, self.firsto[atoms], self.lasto[atoms] + 1
                ):
                    for io in range(io2 - io1):
                        yield ia, io
            else:
                for ia, io1, io2 in zip(
                    atoms, self.firsto[atoms], self.lasto[atoms] + 1
                ):
                    for io in range(io1, io2):
                        yield ia, io

    def iR(self, na: int = 1000, iR: int = 20, R: Optional[float] = None) -> int:
        """Return an integer number of maximum radii (``self.maxR()``) which holds approximately `na` atoms

        Parameters
        ----------
        na :
           number of atoms within the radius
        iR :
           initial `iR` value, which the sphere is estitametd from
        R :
           the value used for atomic range (defaults to ``self.maxR()``)

        Returns
        -------
        int
            number of radius needed to contain `na` atoms. Minimally 2 will be returned.
        """
        ia = np.random.randint(len(self))

        # default block iterator
        if R is None:
            R = self.maxR() + 0.001
        if R < 0:
            raise ValueError(
                f"{self.__class__.__name__}.iR unable to determine a number of atoms within a sphere with negative radius, is maxR() defined?"
            )

        # Number of atoms within 20 * R
        naiR = max(1, len(self.close(ia, R=R * iR)))

        # Convert to na atoms spherical radii
        iR = int(4 / 3 * np.pi * R**3 / naiR * na)

        return max(2, iR)

    def iter_block_rand(
        self,
        iR: int = 20,
        R: Optional[float] = None,
        atoms: AtomsIndex = None,
    ) -> Iterator[tuple[ndarray, ndarray]]:
        """Perform the *random* block-iteration by randomly selecting the next center of block"""

        # We implement yields as we can then do nested iterators
        # create a boolean array
        na = len(self)
        if atoms is not None:
            not_passed = np.zeros(na, dtype=bool)
            # Reverse the values
            not_passed[atoms] = True
        else:
            not_passed = np.ones(na, dtype=bool)

        # Figure out how many we need to loop on
        not_passed_N = np.sum(not_passed)

        if iR < 2:
            raise SislError(f"{self.__class__.__name__}.iter_block_rand too small iR!")

        if R is None:
            R = self.maxR() + 0.001
        # The boundaries (ensure complete overlap)
        R = np.array([iR - 0.5, iR + 0.501]) * R

        # loop until all passed are true
        while not_passed_N > 0:
            # Take a random non-passed element
            all_true = not_passed.nonzero()[0]

            # Shuffle should increase the chance of hitting a
            # completely "fresh" segment, thus we take the most
            # atoms at any single time.
            # Shuffling will cut down needed iterations.
            np.random.shuffle(all_true)
            # take one element, after shufling, we can take the first
            idx = all_true[0]
            del all_true

            # Now we have found a new index, from which
            # we want to create the index based stuff on

            # get all elements within two radii
            all_idx = self.close(idx, R=R)

            # Get unit-cell atoms, we are drawing a circle, and this
            # circle only encompasses those already in the unit-cell.
            all_idx[1] = np.union1d(
                self.asc2uc(all_idx[0], unique=True),
                self.asc2uc(all_idx[1], unique=True),
            )
            # If we translated stuff into the unit-cell, we could end up in situations
            # where the supercell atom is in the circle, but not the UC-equivalent
            # of that one.
            all_idx[0] = all_idx[0][all_idx[0] < na]

            # Only select those who have not been runned yet
            all_idx[0] = all_idx[0][not_passed[all_idx[0]].nonzero()[0]]
            if len(all_idx[0]) == 0:
                continue

            # Tell the next loop to skip those passed
            not_passed[all_idx[0]] = False
            # Update looped variables
            not_passed_N -= len(all_idx[0])

            # Now we want to yield the stuff revealed
            # all_idx[0] contains the elements that should be looped
            # all_idx[1] contains the indices that can be searched
            yield all_idx[0], all_idx[1]

        if np.any(not_passed):
            print(not_passed.nonzero()[0])
            print(np.sum(not_passed), len(self))
            raise SislError(
                f"{self.__class__.__name__}.iter_block_rand error on iterations. Not all atoms have been visited."
            )

    def iter_block_shape(
        self, shape=None, iR: int = 20, atoms: AtomsIndex = None
    ) -> Iterator[tuple[ndarray, ndarray]]:
        """Perform the *grid* block-iteration by looping a grid"""

        # We implement yields as we can then do nested iterators
        # create a boolean array
        na = len(self)
        if atoms is not None:
            not_passed = np.zeros(na, dtype=bool)
            # Reverse the values
            not_passed[atoms] = True
        else:
            not_passed = np.ones(na, dtype=bool)

        # Figure out how many we need to loop on
        not_passed_N = np.sum(not_passed)

        if iR < 2:
            raise SislError(f"{self.__class__.__name__}.iter_block_shape too small iR!")

        R = self.maxR() + 0.001
        if shape is None:
            # we default to the Cube shapes
            dS = (Cube((iR - 0.5) * R), Cube((iR + 1.501) * R))
        else:
            if isinstance(shape, Shape):
                dS = (shape,)
            else:
                dS = tuple(shape)
            if len(dS) == 1:
                dS += (dS[0].expand(R),)
        if len(dS) != 2:
            raise ValueError(
                f"{self.__class__.__name__}.iter_block_shape, number of Shapes *must* be one or two"
            )

        # Now create the Grid
        # convert the radius to a square Grid
        # We do this by examining the x, y, z coordinates
        xyz_m = np.amin(self.xyz, axis=0)
        xyz_M = np.amax(self.xyz, axis=0)
        dxyz = xyz_M - xyz_m

        # Currently iterating different shapes only works for
        # Sphere and Cube
        for s in dS:
            if not isinstance(s, (Cube, Sphere)):
                raise ValueError(
                    f"{self.__class__.__name__}.iter_block_shape currently only works for "
                    "Cube or Sphere objects. Please change sources."
                )

        # Retrieve the internal diameter
        if isinstance(dS[0], Cube):
            ir = dS[0].edge_length
        elif isinstance(dS[0], Sphere):
            ir = [dS[0].radius * 0.5**0.5 * 2] * 3
        elif isinstance(dS[0], Shape):
            # Convert to spheres (which probably should be cubes for performance)
            dS = [s.to.Sphere() for s in dS]
            # Now do the same with spheres
            ir = [dS[0].radius * 0.5**0.5 * 2] * 3

        # Figure out number of segments in each iteration
        # (minimum 1)
        ixyz = _a.arrayi(ceil(dxyz / ir + 0.0001))

        # Calculate the steps required for each iteration
        for i in (0, 1, 2):
            dxyz[i] = dxyz[i] / ixyz[i]

            # Correct the initial position to offset the initial displacement
            # so that we are at the border.
            xyz_m[i] += min(dxyz[i], ir[i]) / 2

            if xyz_m[i] > xyz_M[i]:
                # This is the case where one of the cell dimensions
                # is far too great.
                # In this case ixyz[i] should be 1
                xyz_m[i] = (xyz_M[i] - xyz_m[i]) / 2

        # Shorthand function
        where = np.where

        # Now we loop in each direction
        for x, y, z in product(range(ixyz[0]), range(ixyz[1]), range(ixyz[2])):
            # Create the new center
            center = xyz_m + [x * dxyz[0], y * dxyz[1], z * dxyz[2]]
            # Correct in case the iteration steps across the maximum
            center = where(center < xyz_M, center, xyz_M)
            dS[0].center = center[:]
            dS[1].center = center[:]

            # Now perform the iteration
            # get all elements within two radii
            all_idx = self.within(dS)

            # Get unit-cell atoms, we are drawing a circle, and this
            # circle only encompasses those already in the unit-cell.
            all_idx[1] = np.union1d(
                self.asc2uc(all_idx[0], unique=True),
                self.asc2uc(all_idx[1], unique=True),
            )
            # If we translated stuff into the unit-cell, we could end up in situations
            # where the supercell atom is in the circle, but not the UC-equivalent
            # of that one.
            all_idx[0] = all_idx[0][all_idx[0] < na]

            # Only select those who have not been runned yet
            all_idx[0] = all_idx[0][not_passed[all_idx[0]].nonzero()[0]]
            if len(all_idx[0]) == 0:
                continue

            # Tell the next loop to skip those passed
            not_passed[all_idx[0]] = False
            # Update looped variables
            not_passed_N -= len(all_idx[0])

            # Now we want to yield the stuff revealed
            # all_idx[0] contains the elements that should be looped
            # all_idx[1] contains the indices that can be searched
            yield all_idx[0], all_idx[1]

        if np.any(not_passed):
            not_passed = not_passed.nonzero()[0]
            raise SislError(
                f"{self.__class__.__name__}.iter_block_shape error on iterations. Not all atoms have been visited "
                f"{not_passed}"
            )

    def iter_block(
        self,
        iR: int = 20,
        R: Optional[float] = None,
        atoms: AtomsIndex = None,
        method: str = "rand",
    ) -> Iterator[tuple[ndarray, ndarray]]:
        """Iterator for performance critical loops

        NOTE: This requires that `R` has been set correctly as the maximum interaction range.

        I.e. the loop would look like this:

        >>> for ias, idxs in self.iter_block():
        ...    for ia in ias:
        ...        idx_a = self.close(ia, R = R, idx = idxs)

        This iterator is intended for systems with more than 1000 atoms.

        Remark that the iterator used is non-deterministic, i.e. any two iterators need
        not return the same atoms in any way.

        Parameters
        ----------
        iR :
            the number of `R` ranges taken into account when doing the iterator
        R :
            enables overwriting the local R quantity. Defaults to ``self.maxR() + 0.001``
        atoms :
            enables only effectively looping a subset of the full geometry
        method : {'rand', 'sphere', 'cube'}
            select the method by which the block iteration is performed.
            Possible values are:

             `rand`: a spherical object is constructed with a random center according to the internal atoms
             `sphere`: a spherical equispaced shape is constructed and looped
             `cube`: a cube shape is constructed and looped

        Yields
        -------
        numpy.ndarray
            current list of atoms currently searched
        numpy.ndarray
            atoms that needs searching
        """
        if iR < 2:
            raise SislError(f"{self.__class__.__name__}.iter_block too small iR!")

        method = method.lower()
        if method in ("rand", "random"):
            yield from self.iter_block_rand(iR, R, atoms)
        elif method in ("sphere", "cube"):
            if R is None:
                R = self.maxR() + 0.001

            # Create shapes
            if method == "sphere":
                dS = (Sphere((iR - 0.5) * R), Sphere((iR + 0.501) * R))
            elif method == "cube":
                dS = (
                    Cube((2 * iR - 0.5) * R),
                    # we need an extra R here since it needs to extend on both sides
                    Cube((2 * iR + 1.501) * R),
                )

            yield from self.iter_block_shape(dS)
        else:
            raise ValueError(
                f"{self.__class__.__name__}.iter_block got unexpected 'method' argument: {method}"
            )

    @deprecate_argument(
        "eps",
        "atol",
        "argument eps has been deprecated in favor of atol",
        "0.15",
        "0.17",
    )
    def overlap(
        self,
        other: GeometryLikeType,
        atol: float = 0.1,
        offset: Sequence[float] = (0.0, 0.0, 0.0),
        offset_other: Sequence[float] = (0.0, 0.0, 0.0),
    ) -> tuple[ndarray, ndarray]:
        """Calculate the overlapping indices between two geometries

        Find equivalent atoms (in the primary unit-cell only) in two geometries.
        This routine finds which atoms have the same atomic positions in `self` and `other`.

        Note that this will return duplicate overlapping atoms if one atoms lies within `eps`
        of more than 1 atom in `other`.

        Parameters
        ----------
        other :
           Geometry to compare with `self`
        atol :
           atoms within this distance will be considered *equivalent*
        offset :
           offset for `self.xyz` before comparing
        offset_other :
           offset for `other.xyz` before comparing

        Examples
        --------
        >>> gr22 = sisl.geom.graphene().tile(2, 0).tile(2, 1)
        >>> gr44 = gr22.tile(2, 0).tile(2, 1)
        >>> offset = np.array([0.2, 0.4, 0.4])
        >>> gr22 = gr22.translate(offset)
        >>> idx = np.arange(len(gr22))
        >>> np.random.shuffle(idx)
        >>> gr22 = gr22.sub(idx)
        >>> idx22, idx44 = gr22.overlap(gr44, offset=-offset)
        >>> assert idx22 == np.arange(len(gr22))
        >>> assert idx44 == idx

        Returns
        -------
        idx_self : numpy.ndarray of int
             indices in `self` that are equivalent with `idx_other`
        idx_other : numpy.ndarray of int
             indices in `other` that are equivalent with `idx_self`
        """
        # sanitize `other`
        other = self.new(other)
        s_xyz = self.xyz + (_a.arrayd(offset) - _a.arrayd(offset_other))
        idx_self = []
        self_extend = idx_self.extend
        idx_other = []
        other_extend = idx_other.extend

        for ia, xyz in enumerate(s_xyz):
            # only search in the primary unit-cell
            idx = other.close_sc(xyz, R=(atol,))
            self_extend([ia] * idx.size)
            other_extend(idx)
        return _a.arrayi(idx_self), _a.arrayi(idx_other)

    def find_nsc(
        self,
        axes: Optional[CellAxes] = None,
        R: Optional[float] = None,
        method: Literal["atoms", "cell", "overlap"] = "atoms",
    ) -> ndarray:
        """Find number of supercells for the geometry, depending on certain criteria

        This can find the optimal ``nsc`` values for a given method.

        The important parameter, `method` determines how ``nsc`` is found.
        The method are shown here, from method that produces the smallest ``nsc``, up
        to the largest ``nsc``.

        ``method=atoms``
            here only the atoms ranges are taken into account, and only
            whether atoms in the primary unit cell can connect to others in neigboring
            cells.

        ``method=cell``
            only the atoms ranges are taken into account.
            For instance if a lattice vector is as long as the orbital range
            it will have 3 supercells (it can only connect to its neighboring
            cells).

        ``method=overlap``
            determine nsc by examining at what range two orbitals overlaps.

        Parameters
        ----------
        axes :
           only discover new ``nsc`` the specified axes (defaults to all)
        R :
           the maximum connection radius for each atom, defaults to ``self.maxR()``.
        method:
            See discussion above.

        Returns
        -------
        numpy.ndarray: the found nsc that obeys `method`

        See Also
        --------
        optimize_nsc: same as this, but equivalent to also doing ``self.set_nsc(self.find_nsc(...))``
        """
        method = method.lower()

        nsc = self.nsc.copy()

        if axes is None:
            axes = [0, 1, 2]
        else:
            axes = map(direction, listify(axes)) | listify

        if len(axes) == 0:
            # requesting no search space
            return nsc

        if R is None:
            R = self.maxR() + 0.001
        if R < 0:
            R = 0.00001
            warn(
                f"{self.__class__.__name__}"
                ".find_nsc could not determine the radius from the "
                "internal atoms (defaulting to zero radius)."
            )

        cell = self.cell
        length, angles = self.lattice.parameters()

        # TODO check that angles below 60 degrees are
        # important.

        # Half-nsc (only 1 direction)
        hsc = nsc // 2

        # determine the maximum hsc values
        if method in ("atoms", "cell"):
            R_actual = R
        elif method in ("overlap",):
            R_actual = R * 2
        else:
            raise ValueError(
                f"{self.__class__.__name__}.find_nsc got wrong 'method' argument, got {method}"
            )

        # Determine the actual range depending on the actual R
        hsc[axes] = ceil(R_actual / length[axes])

        if method == "atoms":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                isc = _a.emptyi(3)

                for ax in axes:
                    # Initialize the isc for this direction
                    # (note we do not take non-orthogonal directions
                    #  into account)
                    isc[:] = 0

                    # Initialize the actual number of supercell connections
                    # along this direction.
                    prev_isc = 0

                    while prev_isc == isc[ax]:
                        # Try next supercell connection
                        isc[ax] += 1

                        for ia in self:
                            idx = self.close_sc(ia, isc=isc, R=R)
                            if len(idx) > 0:
                                prev_isc = isc[ax]
                                break

                    hsc[ax] = prev_isc

        nsc[axes] = hsc[axes] * 2 + 1

        return nsc

    @deprecation(
        "optimize_nsc is deprecated, update the code to use 'find_nsc' and then 'set_nsc'",
        "0.15.0",
        "0.16.0",
    )
    def optimize_nsc(
        self,
        axes: Optional[CellAxes] = None,
        R: Optional[float] = None,
    ) -> ndarray:
        """Optimize the number of supercell connections based on ``self.maxR()``

        After this routine the number of supercells may not necessarily be the same.

        This is an in-place operation.

        Deprecated method!

        Parameters
        ----------
        axes :
           only optimize the specified axes (default to all)
        R :
           the maximum connection radius for each atom
        """
        nsc = self.find_nsc(axes, R, method="atoms")
        self.set_nsc(nsc)
        return nsc

    def sub_orbital(self, atoms: AtomsIndex, orbitals: OrbitalsIndex) -> Geometry:
        r"""Retain only a subset of the orbitals on `atoms` according to `orbitals`

        This allows one to retain only a given subset of geometry.

        Parameters
        ----------
        atoms :
            indices of atoms or `Atom` that will be reduced in size according to `orbitals`
        orbitals :
            indices of the orbitals on `atoms` that are retained in the geometry, the list of
            orbitals will be sorted.

        Notes
        -----
        Future implementations may allow one to re-arrange orbitals using this method.

        When using this method the internal species list will be populated by another specie
        that is named after the orbitals removed. This is to distinguish different atoms.

        Examples
        --------

        >>> # a Carbon atom with 2 orbitals
        >>> C = sisl.Atom('C', [1., 2.])
        >>> # an oxygen atom with 3 orbitals
        >>> O = sisl.Atom('O', [1., 2., 2.4])
        >>> geometry = sisl.Geometry([[0] * 3, [1] * 3]], 2, [C, O])

        Now ``geometry`` is a geometry with 2 different species and 6 atoms (3 of each).
        They are ordered ``[C, O, C, O, C, O]``. In the following we
        will note species that are different from the original by a ``'`` in the list.

        Retain 2nd orbital on the 2nd atom: ``[C, O', C, O, C, O]``

        >>> new_geom = geometry.sub_orbital(1, 1)

        Retain 2nd orbital on 1st and 2nd atom: ``[C', O', C, O, C, O]``

        >>> new_geom = geometry.sub_orbital([0, 1], 1)

        Retain 2nd orbital on the 1st atom and 3rd orbital on 4th atom: ``[C', O, C, O', C, O]``

        >>> new_geom = geometry.sub_orbital(0, 1).sub_orbital(3, 2)

        Retain 2nd orbital on all atoms equivalent to the first atom: ``[C', O, C', O, C', O]``

        >>> new_geom = geometry.sub_orbital(obj.geometry.atoms[0], 1)

        Retain 1st orbital on 1st atom, and 2nd orbital on 3rd and 5th atom: ``[C', O, C'', O, C'', O]``

        >>> new_geom = geometry.sub_orbital(0, 0).sub_orbital([2, 4], 1)

        See Also
        --------
        remove_orbital : removing a set of orbitals (opposite of this)
        """
        atoms = self._sanitize_atoms(atoms).ravel()

        # Figure out if all atoms have the same species
        species = self.atoms.species[atoms]
        uniq_species, indices = unique(species, return_inverse=True)
        if len(uniq_species) > 1:
            # In case there are multiple different species but one wishes to
            # retain the same orbital index, then we loop on the unique species
            new = self
            for i in range(uniq_species.size):
                idx = (indices == i).nonzero()[0]
                # now determine whether it is the whole atom
                # or only part of the geometry
                new = new.sub_orbital(atoms[idx], orbitals)
            return new

        # At this point we are sure that uniq_species is *only* one specie!
        geom = self.copy()

        # Get the atom object we wish to reduce
        old_atom = geom.atoms[atoms[0]]
        old_atom_species = geom.atoms.species_index(old_atom)
        old_atom_count = (geom.atoms.species == old_atom_species).sum()

        if isinstance(orbitals, (Orbital, Integral)):
            orbitals = [orbitals]
        if isinstance(orbitals[0], Orbital):
            orbitals = [old_atom.index(orb) for orb in orbitals]
        orbitals = np.sort(orbitals)

        if len(orbitals) == 0:
            raise ValueError(
                f"{self.__class__.__name__}.sub_orbital trying to retain 0 orbitals on a given atom. This is not allowed!"
            )

        # create the new atom
        new_atom = old_atom.sub(orbitals)
        # Rename the new-atom to <>_1_2 for orbital == [1, 2]
        new_atom._tag += "_" + "_".join(map(str, orbitals))

        # There are now 2 cases.
        #  1. we replace all atoms of a given specie
        #  2. we replace a subset of atoms of a given specie
        if len(atoms) == old_atom_count:
            # We catch the warning about reducing the number of orbitals!
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                # this is in-place operation and we don't need to worry about
                geom.atoms.replace_atom(old_atom, new_atom)

        else:
            # we have to add the new one (in case it does not exist)
            try:
                new_atom_species = geom.atoms.species_index(new_atom)
            except Exception:
                new_atom_species = geom.atoms.nspecies
                # the above checks that it is indeed a new atom
                geom._atoms._atom.append(new_atom)
            # transfer specie index
            geom.atoms._species[atoms] = new_atom_species
            geom.atoms._update_orbitals()

        return geom

    def remove_orbital(self, atoms: AtomsIndex, orbitals: OrbitalsIndex) -> Geometry:
        """Remove a subset of orbitals on `atoms` according to `orbitals`

        For more detailed examples, please see the equivalent (but opposite) method
        `sub_orbital`.

        Parameters
        ----------
        atoms : array_like of int or Atom
            indices of atoms or `Atom` that will be reduced in size according to `orbitals`
        orbitals : array_like of int or Orbital
            indices of the orbitals on `atoms` that are removed from the geometry.

        See Also
        --------
        sub_orbital : retaining a set of orbitals (see here for examples)
        """
        # Get specie index of the atom (convert to list of indices)
        atoms = self._sanitize_atoms(atoms).ravel()

        # Figure out if all atoms have the same species
        species = self.atoms.species[atoms]
        uniq_species, indices = unique(species, return_inverse=True)
        if len(uniq_species) > 1:
            # In case there are multiple different species but one wishes to
            # retain the same orbital index, then we loop on the unique species
            new = self
            for i in range(uniq_species.size):
                idx = (indices == i).nonzero()[0]
                # now determine whether it is the whole atom
                # or only part of the geometry
                new = new.remove_orbital(atoms[idx], orbitals)
            return new

        # Get the atom object we wish to reduce
        # We know np.all(geom.atoms[atom] == old_atom)
        old_atom = self.atoms[atoms[0]]

        if isinstance(orbitals, (Orbital, Integral)):
            orbitals = [orbitals]
        if isinstance(orbitals[0], Orbital):
            orbitals = [old_atom.index(orb) for orb in orbitals]
        orbitals = np.delete(_a.arangei(len(old_atom)), np.asarray(orbitals).ravel())
        orbitals = np.sort(orbitals)

        # now call sub_orbital
        return self.sub_orbital(atoms, orbitals)

    def __mul__(self, m, method="tile") -> Geometry:
        """Implement easy tile/repeat function

        Parameters
        ----------
        m : int or tuple or list or (tuple, str) or (list, str)
           a tuple/list may be of length 2 or 3. A length of 2 corresponds
           to tuple[0] == *number of multiplications*, tuple[1] is the
           axis.
           A length of 3 corresponds to each of the directions.
           An optional string may be used to specify the `tile` or `repeat` function.
           The default is the `tile` function.

        Examples
        --------
        >>> geometry = Geometry([0.] * 3, lattice=[1.5, 3, 4])
        >>> geometry * 2 == geometry.tile(2, 0).tile(2, 1).tile(2, 2)
        True
        >>> geometry * [2, 1, 2] == geometry.tile(2, 0).tile(2, 2)
        True
        >>> geometry * [2, 2] == geometry.tile(2, 2)
        True
        >>> geometry * ([2, 1, 2], 'repeat') == geometry.repeat(2, 0).repeat(2, 2)
        True
        >>> geometry * ([2, 1, 2], 'r') == geometry.repeat(2, 0).repeat(2, 2)
        True
        >>> geometry * ([2, 0], 'r') == geometry.repeat(2, 0)
        True
        >>> geometry * ([2, 2], 'r') == geometry.repeat(2, 2)
        True

        See Also
        --------
        tile : specific method to enlarge the geometry
        repeat : specific method to enlarge the geometry
        """
        # Simple form
        if isinstance(m, Integral):
            return self * [m, m, m]

        # Error in argument, fall-back
        if len(m) == 1:
            return self * m[0]

        # Look-up table
        method_tbl = {"r": "repeat", "repeat": "repeat", "t": "tile", "tile": "tile"}

        # Determine the type
        if len(m) == 2:
            # either
            #  (r, axis)
            #  ((...), method
            if isinstance(m[1], str):
                method = method_tbl[m[1]]
                m = m[0]

        g = self
        if len(m) == 1:
            #  r
            m = m[0]
            for i in range(3):
                g = getattr(g, method)(m, i)

        elif len(m) == 2:
            #  (r, axis)
            g = getattr(g, method)(m[0], m[1])

        elif len(m) == 3:
            #  (r, r, r)
            for i in range(3):
                g = getattr(g, method)(m[i], i)

        else:
            raise ValueError(f"Multiplying a geometry got an unexpected value: {m}")

        return g

    def __rmul__(self, m) -> Geometry:
        """Default to repeating the atomic structure"""
        return self.__mul__(m, "repeat")

    def angle(
        self,
        atoms: AtomsIndex,
        dir: Union[str, int, Sequence[int]] = (1.0, 0, 0),
        ref: Optional[Union[int, Sequence[float]]] = None,
        rad: bool = False,
    ) -> Union[float, ndarray]:
        r"""The angle between atom `atoms` and the direction `dir`, with possibility of a reference coordinate `ref`

        The calculated angle can be written as this

        .. math::
            \theta = \arccos \frac{(\mathbf r^I - \mathbf{r^{\mathrm{ref}}})\cdot \mathbf{d}}
            {|\mathbf r^I-\mathbf{r^{\mathrm{ref}}}||\mathbf{d}|}

        and thus lies in the interval :math:`[0 ; \pi]` as one cannot distinguish orientation without
        additional vectors.

        Parameters
        ----------
        atoms :
           indices/boolean of all atoms where angles should be calculated on
        dir :
           the direction from which the angle is calculated from, default to ``x``.
           An integer specifies the corresponding lattice vector as the direction.
        ref :
           the reference point from which the vectors are drawn, default to origin
           An integer specifies an atomic index.
        rad :
           whether the returned value is in radians
        """
        xi = self.axyz(atoms)
        if isinstance(dir, (str, Integral)):
            dir = direction(dir, abc=self.cell, xyz=np.diag([1] * 3))
        else:
            dir = _a.asarrayd(dir)
        # Normalize so we don't have to have this in the
        # below formula
        dir = dir / fnorm(dir)

        if ref is None:
            pass
        elif isinstance(ref, Integral):
            xi -= self.axyz(ref)[None, :]
        else:
            xi -= _a.asarrayd(ref)[None, :]
        nx = sqrt(square(xi).sum(1))
        ang = np.zeros_like(nx)
        idx = (nx > 1e-6).nonzero()[0]
        ang[idx] = np.arccos(xi[idx] @ dir / nx[idx])
        if rad:
            return ang
        return np.degrees(ang)

    def dihedral(
        self,
        atoms: AtomsIndex,
        rad: bool = False,
    ) -> Union[float, ndarray]:
        r"""Calculate the dihedral angle defined by four atoms.

        The dihehral angle is defined between 2 half-planes.

        The first 3 atoms define the first plane
        The last 3 atoms define the second.

        The dihedral angle is calculated using this formula:

        .. math::

            \mathbf u_0 &= \mathbf r_1 - \mathbf r_0
            \\
            \mathbf u_1 &= \mathbf r_2 - \mathbf r_1
            \\
            \mathbf u_2 &= \mathbf r_3 - \mathbf r_2
            \\
            \phi &= \operatorname{atan2}\Big(
                 \hat{\mathbf u}_0\cdot
                (\hat{\mathbf u}_1\times\hat{\mathbf u}_2),
                (\hat{\mathbf u}_0\times\hat{\mathbf u}_1)\cdot
                (\hat{\mathbf u}_1\times\hat{\mathbf u}_2)
                \Big)

        Where :math:`\hat{\cdot}` means the unit-vector.

        Parameters
        ----------
        atoms :
           An array of shape `(4,)` or `(*, 4)` representing the indices of 4 atoms forming the dihedral angle
        rad :
           whether the returned value is in radians
        """
        atoms = self._sanitize_atoms(atoms)
        ndim = atoms.ndim
        if ndim == 1:
            if len(atoms) != 4:
                raise ValueError(
                    f"{self.__class__.__name__}.dihedral requires atoms to be 4 indices"
                )
            atoms = [atoms]
        elif ndim == 2:
            if atoms.shape[1] != 4:
                raise ValueError(
                    f"{self.__class__.__name__}.dihedral requires atoms to be (*, 4) indices"
                )
        else:
            raise ValueError(
                f"{self.__class__.__name__}.dihedral requires atoms index of shape (4,) or (*, 4)"
            )

        # The 2 planes are defined by
        #  r0, r1, r2
        # and
        #  r1, r2, r3
        #   we know that atoms has a dimension of 2!
        u = diff(self.axyz(atoms), axis=1)
        # normalize to make algorithm easier
        u /= fnorm(u)[..., None]
        # calculate the two planes normal vector
        n0 = np.cross(u[:, 0], u[:, 1])
        n1 = np.cross(u[:, 1], u[:, 2])

        # Prepare arguments for atan2
        y = (u[:, 0] * n1).sum(axis=-1)
        x = (n0 * n1).sum(axis=-1)

        # see https://en.wikipedia.org/wiki/Dihedral_angle
        angles = np.arctan2(y, x)

        if not rad:
            angles = np.degrees(angles)

        if ndim == 1:
            return angles[0]

        return angles

    def rotate_miller(self, m, v) -> Geometry:
        """Align Miller direction along ``v``

        Rotate geometry and cell such that the Miller direction
        points along the Cartesian vector ``v``.
        """
        # Create normal vector to miller direction and cartesian
        # direction
        cp = _a.arrayd(
            [
                m[1] * v[2] - m[2] * v[1],
                m[2] * v[0] - m[0] * v[2],
                m[0] * v[1] - m[1] * v[0],
            ]
        )
        cp /= fnorm(cp)

        lm = _a.arrayd(m)
        lm /= fnorm(lm)
        lv = _a.arrayd(v)
        lv /= fnorm(lv)

        # Now rotate the angle between them
        a = acos(np.sum(lm * lv))
        return self.rotate(a, cp, rad=True)

    def translate2uc(
        self,
        atoms: AtomsIndex = None,
        axes: Optional[Union[int, bool, Sequence[int]]] = None,
    ) -> Geometry:
        """Translates atoms in the geometry into the unit cell

        One can translate a subset of the atoms or axes by appropriate arguments.

        Warning
        -------
        When coordinates are lying on one of the edges, they may move to the other
        side of the unit-cell due to small rounding errors.
        In such situations you are encouraged to shift all coordinates by a small
        amount to remove numerical errors, in the following case we have atomic
        coordinates lying close to the lower side of each lattice vector.

        >>> geometry.translate(1e-8).translate2uc().translate(-1e-8)

        Notes
        -----
        By default only the periodic axes (``self.pbc``) will be translated to the UC. If
        translation is required for all axes, supply them directly.

        Parameters
        ----------
        atoms :
             only translate the given atomic indices, if not specified, all
             atoms will be translated
        axes :
             only translate certain lattice directions, `None` specifies
             only the directions with supercells, `True` specifies all
             directions.
        """
        if axes is None:
            axes = self.pbc.nonzero()[0]
        elif isinstance(axes, bool):
            if axes:
                axes = (0, 1, 2)
            else:
                raise ValueError(
                    "translate2uc with a bool argument can only be True to signal all axes"
                )
        axes = map(direction, listify(axes)) | listify

        fxyz = self.fxyz
        # move to unit-cell
        fxyz[:, axes] %= 1
        g = self.copy()
        # convert back
        if atoms is None:
            g.xyz[:, :] = fxyz @ self.cell
        else:
            idx = self._sanitize_atoms(atoms).ravel()
            g.xyz[idx] = fxyz[idx] @ self.cell
        return g

    def add_vacuum(
        self, vacuum: float, axis: int, offset: Sequence[float] = (0, 0, 0)
    ) -> Geometry:
        """Add vacuum along the `axis` lattice vector

        When the vacuum is bigger than the maximum orbital ranges the
        number of supercells along that axis will be truncated to 1 (de-couple
        images).

        Parameters
        ----------
        vacuum :
           amount of vacuum added, in Ang
        axis :
           the lattice vector to add vacuum along
        offset :
            offset in geometry when adding the vacuum.

        Returns
        -------
        Geometry : a new geometry with added vacuum
        """
        new = self.copy()
        new.xyz += _a.arrayd(offset)
        new.set_lattice(self.lattice.add_vacuum(vacuum, axis))
        if vacuum > self.maxR() + 0.001:
            # only overwrite along axis
            nsc = [None for _ in range(3)]
            nsc[axis] = 1
            new.lattice.set_nsc(nsc)
        return new

    def __add__(self, b) -> Geometry:
        """Merge two geometries (or geometry and supercell)

        Parameters
        ----------
        self, b : Geometry or Lattice or tuple or list
           when adding a Geometry with a Geometry it defaults to using `add` function
           with the LHS retaining the cell-vectors.
           a tuple/list may be of length 2 with the first element being a Geometry and the second
           being an integer specifying the lattice vector where it is appended.
           One may also use a `Lattice` instead of a `Geometry` which behaves similarly.

        Examples
        --------

        >>> A + B == A.add(B)
        >>> A + (B, 1) == A.append(B, 1)
        >>> A + (B, 2) == A.append(B, 2)
        >>> (A, 1) + B == A.append(B, 1)

        See Also
        --------
        add : add geometries
        append : appending geometries
        prepend : prending geometries
        """
        if isinstance(b, (Lattice, Geometry)):
            return self.add(b)
        return self.append(b[0], b[1])

    def __radd__(self, b) -> Geometry:
        """Merge two geometries (or geometry and supercell)

        Parameters
        ----------
        self, b : Geometry or Lattice or tuple or list
           when adding a Geometry with a Geometry it defaults to using `add` function
           with the LHS retaining the cell-vectors.
           a tuple/list may be of length 2 with the first element being a Geometry and the second
           being an integer specifying the lattice vector where it is appended.
           One may also use a `Lattice` instead of a `Geometry` which behaves similarly.

        Examples
        --------

        >>> A + B == A.add(B)
        >>> A + (B, 1) == A.append(B, 1)
        >>> A + (B, 2) == A.append(B, 2)
        >>> (A, 1) + B == A.append(B, 1)

        See Also
        --------
        add : add geometries
        append : appending geometries
        prepend : prending geometries
        """
        if isinstance(b, (Lattice, Geometry)):
            return b.add(self)
        return self + b

    def attach(
        self,
        atom: int,
        other: GeometryLike,
        other_atom: int,
        dist="calc",
        axis: Optional[int] = None,
    ) -> Geometry:
        """Attaches another `Geometry` at the `atom` index with respect to `other_atom` using different methods.

        The attached geometry will be inserted at the end of the geometry via `add`.

        Parameters
        ----------
        atom : int
           atomic index which is the base position of the attachment. The distance
           between `atom` and `other_atom` is `dist`.
        other : Geometry
           the other Geometry to attach at the given point. In this case `dist` from
           `atom`.
        other_atom : int
           the index of the atom in `other` that is inserted at `atom`.
        dist : array_like or float or str, optional
           the distance (in `Ang`) between the attached coordinates.
           If `dist` is `array_like` it should be the vector between
           the atoms;
           if `dist` is `float` the argument `axis` is required
           and the vector will be calculated along the corresponding latticevector;
           else if `dist` is `str` this will correspond to the
           `method` argument of the `Atom.radius` class of the two
           atoms. Here `axis` is also required.
        axis : int
           specify the direction of the lattice vectors used.
           Not used if `dist` is an array-like argument.
        """
        other = self.new(other)
        if isinstance(dist, Real):
            # We have a single rational number
            if axis is None:
                raise ValueError(
                    f"{self.__class__.__name__}.attach, `axis` has not been specified, please specify the axis when using a distance"
                )

            # Now calculate the vector that we should have
            # between the atoms
            v = self.cell[axis, :]
            v = v / (v @ v) ** 0.5 * dist

        elif isinstance(dist, str):
            # We have a single rational number
            if axis is None:
                raise ValueError(
                    f"{self.__class__.__name__}.attach, `axis` has not been specified, please specify the axis when using a distance"
                )

            # This is the empirical distance between the atoms
            d = self.atoms[atom].radius(dist) + other.atoms[other_atom].radius(dist)
            if isinstance(axis, Integral):
                v = self.cell[axis, :]
            else:
                v = np.array(axis)

            v = v / (v @ v) ** 0.5 * d

        else:
            # The user *must* have supplied a vector
            v = np.array(dist)

        # Now create a copy of the other geometry
        # so that we move it...
        # Translate to origin, then back to position in new cell
        o = other.translate(-other.xyz[other_atom] + self.xyz[atom] + v)

        # We do not know how to handle the lattice-vectors,
        # so we will do nothing...
        return self.add(o)

    def replace(
        self,
        atoms: AtomsIndex,
        other: GeometryLike,
        offset: Sequence[float] = (0.0, 0.0, 0.0),
    ) -> Geometry:
        """Create a new geometry from `self` and replace `atoms` with `other`

        Parameters
        ----------
        atoms :
            atoms in `self` to be removed and replaced by other
            `other` will be placed in the geometry at the lowest index of `atoms`
        other :
            the other Geometry to insert instead, the unit-cell will not
            be used.
        offset :
            the offset for `other` when adding its coordinates, default to no offset
        """
        # Find lowest value in atoms
        atoms = self._sanitize_atoms(atoms)
        index = atoms.min()
        if offset is None:
            offset = _a.zerosd(3)

        # remove atoms, preparing for inserting new geometry
        out = self.remove(atoms)

        other = self.new(other)

        # insert new positions etc.
        out.xyz = np.insert(out.xyz, index, other.xyz + offset, axis=0)
        out._atoms = out.atoms.insert(index, other.atoms)
        return out

    def reverse(self, atoms: AtomsIndex = None) -> Geometry:
        """Returns a reversed geometry

        Also enables reversing a subset of the atoms.

        Parameters
        ----------
        atoms :
             only reverse the given atomic indices, if not specified, all
             atoms will be reversed
        """
        if atoms is None:
            xyz = self.xyz[::-1, :]
        else:
            atoms = self._sanitize_atoms(atoms).ravel()
            xyz = np.copy(self.xyz)
            xyz[atoms, :] = self.xyz[atoms[::-1], :]
        return self.__class__(
            xyz, atoms=self.atoms.reverse(atoms), lattice=self.lattice.copy()
        )

    def mirror(
        self,
        method,
        atoms: AtomsIndex = None,
        point: Sequence[float] = (0, 0, 0),
    ) -> Geometry:
        r"""Mirrors the atomic coordinates about a plane given by its normal vector

        This will typically move the atomic coordinates outside of the unit-cell.
        This method should be used with care.

        Parameters
        ----------
        method : {'xy'/'z', ..., 'ab', ..., v}
           mirror the structure about a Cartesian direction (``x``, ``y``, ``z``),
           plane (``xy``, ``xz``, ``yz``) or about user defined vectors (``v``).
           A vector may also be specified by ``'ab'`` which is the vector normal
           to the plane spanned by the first and second lattice vector.
           or user defined vector (`v`) which is defining a plane.
        atoms :
           only mirror a subset of atoms
        point:
           mirror coordinates around the plane that intersects the *method* vector
           and this point

        Examples
        --------
        >>> geom = geom.graphene()
        >>> out = geom.mirror('x')
        >>> out.xyz[:, 0]
        [0.  -1.42]
        >>> out = geom.mirror('x', point=(1.42/2, 0, 0))
        >>> out.xyz[:, 0]
        [1.42  0.]
        """
        atoms = self._sanitize_atoms(atoms)
        point = _a.asarrayd(point)

        if isinstance(method, str):
            method = "".join(sorted(method.lower()))
            if method in ("z", "xy"):
                method = _a.arrayd([0, 0, 1])
            elif method in ("x", "yz"):
                method = _a.arrayd([1, 0, 0])
            elif method in ("y", "xz"):
                method = _a.arrayd([0, 1, 0])
            elif method == "a":
                method = self.cell[0]
            elif method == "b":
                method = self.cell[1]
            elif method == "c":
                method = self.cell[2]
            elif method == "ab":
                method = cross3(self.cell[0], self.cell[1])
            elif method == "ac":
                method = cross3(self.cell[0], self.cell[2])
            elif method == "bc":
                method = cross3(self.cell[1], self.cell[2])
            else:
                raise ValueError(
                    f"{self.__class__.__name__}.mirror unrecognized 'method' value"
                )

        # it has to be an array of length 3
        # Mirror about a user defined vector
        method = _a.asarrayd(method).copy()
        method /= fnorm(method)

        # project onto vector
        vp = (self.xyz[atoms, :] - point).dot(method) * 2

        # convert coordinates
        # first subtract the projection, then its mirror position
        g = self.copy()
        g.xyz[atoms, :] -= vp.reshape(-1, 1) * method.reshape(1, 3)
        return g

    def axyz(self, atoms: AtomsIndex = None, isc=None) -> ndarray:
        """Return the atomic coordinates in the supercell of a given atom.

        The ``Geometry[...]`` slicing is calling this function with appropriate options.

        Parameters
        ----------
        atoms :
          atom(s) from which we should return the coordinates, the atomic indices
          may be in supercell format.
        isc : array_like, optional
            Returns the atomic coordinates shifted according to the integer
            parts of the cell. Defaults to the unit-cell

        Examples
        --------
        >>> geom = Geometry([[0, 0, 0], [0.5, 0, 0]], lattice=1.)
        >>> print(geom.axyz(isc=[1,0,0]))
        [[1.   0.   0. ]
         [1.5  0.   0. ]]

        >>> geom = Geometry([[0, 0, 0], [0.5, 0, 0]], lattice=1.)
        >>> print(geom.axyz(0))
        [0.  0.  0.]

        """
        if atoms is None and isc is None:
            return self.xyz
        atoms = self._sanitize_atoms(atoms)

        # If only atoms has been specified
        if isc is None:
            # get offsets from atomic indices (note that this will be per atom)
            isc = self.a2isc(atoms)
            offset = self.lattice.offset(isc)
            return self.xyz[self.asc2uc(atoms)] + offset

        # Neither of atoms, or isc are `None`, we add the offset to all coordinates
        return self.axyz(atoms) + self.lattice.offset(isc)

    def within_sc(
        self,
        shapes,
        isc=None,
        atoms: AtomsIndex = None,
        atoms_xyz=None,
        ret_xyz: bool = False,
        ret_rij: bool = False,
    ):
        """Indices of atoms in a given supercell within a given shape from a given coordinate

        This returns a set of atomic indices which are within a
        sphere of radius ``R``.

        If R is a tuple/list/array it will return the indices:
        in the ranges:

        >>> ( x <= R[0] , R[0] < x <= R[1], R[1] < x <= R[2] )

        Parameters
        ----------
        shapes : Shape or list of Shape
            A list of increasing shapes that define the extend of the geometric
            volume that is searched.
            It is vital that::

               shapes[0] in shapes[1] in shapes[2] ...
        isc : array_like, optional
            The super-cell which the coordinates are checked in. Defaults to ``[0, 0, 0]``
        atoms :
            List of atoms that will be considered. This can
            be used to only take out a certain atoms.
        atoms_xyz : array_like, optional
            The atomic coordinates of the equivalent `idx` variable (`idx` must also be passed)
        ret_xyz :
            If True this method will return the coordinates
            for each of the couplings.
        ret_rij :
            If True this method will return the distance to the center of the shapes

        Returns
        -------
        index
            indices of atoms (in supercell indices) within the shape
        xyz
            atomic coordinates of the indexed atoms (only for true `ret_xyz`)
        rij
            distance of the indexed atoms to the center of the shape (only for true `ret_rij`)
        """

        # Ensure that `shapes` is a list
        if isinstance(shapes, Shape):
            shapes = [shapes]
        nshapes = len(shapes)

        # Convert to actual array
        if atoms is not None:
            atoms = self._sanitize_atoms(atoms)
        else:
            # If idx is None, then idx_xyz cannot be used!
            # So we force it to None
            atoms_xyz = None

        # Get shape centers
        off = shapes[-1].center[:]
        # Get the supercell offset
        soff = self.lattice.offset(isc)[:]

        # Get atomic coordinate in principal cell
        if atoms_xyz is None:
            xa = self[atoms, :] + soff[None, :]
        else:
            # For extremely large systems re-using the
            # idx_xyz is faster than indexing
            # a very large array
            # However, this idx_xyz should not
            # be offset by any supercell
            xa = atoms_xyz[:, :] + soff[None, :]

        # Get indices and coordinates of the largest shape
        # The largest part of the calculation are to calculate
        # the content in the largest shape.
        ix = shapes[-1].within_index(xa)
        # Reduce search space
        xa = xa[ix, :]

        if atoms is None:
            # This is because of the pre-check of the distance checks
            atoms = ix
        else:
            atoms = atoms[ix]

        if len(xa) == 0:
            # Quick return if there are no entries...

            ret = [[np.empty([0], np.int32)] * nshapes]
            if ret_xyz:
                ret.append([np.empty([0, 3], np.float64)] * nshapes)
            if ret_rij:
                ret.append([np.empty([0], np.float64)] * nshapes)

            if nshapes == 1:
                if ret_xyz and ret_rij:
                    return [ret[0][0], ret[1][0], ret[2][0]]
                elif ret_xyz or ret_rij:
                    return [ret[0][0], ret[1][0]]
                return ret[0][0]
            if ret_xyz or ret_rij:
                return ret
            return ret[0]

        # Calculate distance
        if ret_rij:
            d = sqrt(square(xa - off[None, :]).sum(1))

        # Create the initial lists that we will build up
        # Then finally, we will return the reversed lists

        # Quick return
        if nshapes == 1:
            ret = [[atoms]]
            if ret_xyz:
                ret.append([xa])
            if ret_rij:
                ret.append([d])
            if ret_xyz or ret_rij:
                return ret
            return ret[0]

        # TODO Check that all shapes coincide with the following shapes

        # Now we create a list of indices which coincide
        # in each of the shapes
        # Do a reduction on each of the list elements
        ixS = []
        cum = np.array([], atoms.dtype)
        for i, s in enumerate(shapes):
            x = s.within_index(xa)
            if i > 0:
                x = np.setdiff1d(x, cum, assume_unique=True)
            # Update elements to remove in next loop
            cum = np.append(cum, x)
            ixS.append(x)

        # Do for the first shape
        ret = [[_a.asarrayi(atoms[ixS[0]]).ravel()]]
        rc = 0
        if ret_xyz:
            rc = rc + 1
            ret.append([xa[ixS[0], :]])
        if ret_rij:
            rd = rc + 1
            ret.append([d[ixS[0]]])
        for i in range(1, nshapes):
            ret[0].append(_a.asarrayi(atoms[ixS[i]]).ravel())
            if ret_xyz:
                ret[rc].append(xa[ixS[i], :])
            if ret_rij:
                ret[rd].append(d[ixS[i]])

        if ret_xyz or ret_rij:
            return ret
        return ret[0]

    def close_sc(
        self,
        xyz_ia,
        isc=(0, 0, 0),
        R=None,
        atoms: AtomsIndex = None,
        atoms_xyz=None,
        ret_xyz: bool = False,
        ret_rij: bool = False,
    ):
        """Indices of atoms in a given supercell within a given radius from a given coordinate

        This returns a set of atomic indices which are within a
        sphere of radius `R`.

        If `R` is a tuple/list/array it will return the indices:
        in the ranges:

        >>> ( x <= R[0] , R[0] < x <= R[1], R[1] < x <= R[2] )

        Parameters
        ----------
        xyz_ia : array_like of floats or int
            Either a point in space or an index of an atom.
            If an index is passed it is the equivalent of passing
            the atomic coordinate ``close_sc(self.xyz[xyz_ia,:])``.
        isc : (3,), optional
            Integer super-cell offsets in which the coordinates are checked in.
            I.e. ``isc=[0, 0, 0]`` is the primary cell (default).
        R : float or array_like, optional
            The radii parameter to where the atomic connections are found.
            If `R` is an array it will return the indices:
            in the ranges ``( x <= R[0] , R[0] < x <= R[1], R[1] < x <= R[2] )``.
            If a single float it will return ``x <= R``.
        atoms :
            List of atoms that will be considered. This can
            be used to only take out a certain atom.
        atoms_xyz : array_like of float, optional
            The atomic coordinates of the equivalent `atoms` variable (`atoms` must also be passed)
        ret_xyz :
            If True this method will return the coordinates
            for each of the couplings.
        ret_rij :
            If True this method will return the distance
            for each of the couplings.

        Returns
        -------
        index
            indices of atoms (in supercell indices) within the shells of radius `R`
        xyz
            atomic coordinates of the indexed atoms (only for true `ret_xyz`)
        rij
            distance of the indexed atoms to the center coordinate (only for true `ret_rij`)
        """
        maxR = self.maxR() + 0.001
        if R is None:
            R = np.array([maxR], np.float64)
        elif not isndarray(R):
            R = _a.asarrayd(R).ravel()

        # Maximum distance queried
        max_R = R[-1]
        if atoms is not None and max_R > maxR + 0.1:
            warn(
                f"{self.__class__.__name__}.close_sc has been passed an 'atoms' argument "
                "together with an R value larger than the orbital ranges. "
                "If used together with 'sparse-matrix.construct' this can result in wrong couplings.",
                register=True,
            )

        # Convert to actual array
        if atoms is not None:
            atoms = self._sanitize_atoms(atoms).ravel()
        else:
            # If atoms is None, then atoms_xyz cannot be used!
            atoms_xyz = None

        if isinstance(xyz_ia, Integral):
            off = self.xyz[xyz_ia]
        elif not isndarray(xyz_ia):
            off = _a.asarrayd(xyz_ia)
        elif xyz_ia.ndim == 0:
            off = self.xyz[xyz_ia]
        else:
            off = xyz_ia

        # Calculate the complete offset
        foff = self.lattice.offset(isc) - off

        # Get distances between `xyz_ia` and `atoms`
        if atoms_xyz is None:
            dxa = self.axyz(atoms) + foff
        else:
            # For extremely large systems re-using the
            # atoms_xyz is faster than indexing
            # a very large array
            dxa = atoms_xyz + foff

        # Immediately downscale by easy checking
        # This will reduce the computation of the vector-norm
        # which is the main culprit of the time-consumption
        # This abstraction will _only_ help very large
        # systems.
        # For smaller ones this will actually be a slower
        # method..
        if atoms is None:
            atoms, d = indices_in_sphere_with_dist(dxa, max_R)
            dxa = dxa[atoms].reshape(-1, 3)
        else:
            ix, d = indices_in_sphere_with_dist(dxa, max_R)
            atoms = atoms[ix]
            dxa = dxa[ix].reshape(-1, 3)
            del ix

        if len(atoms) == 0:
            # Create default return
            ret = [[_a.emptyi([0]) for _ in R]]
            if ret_xyz:
                ret.append([_a.emptyd([0, 3]) for _ in R])
            if ret_rij:
                ret.append([_a.emptyd([0]) for _ in R])

            # Quick return if there are
            # no entries...
            if len(R) == 1:
                if ret_xyz and ret_rij:
                    return [ret[0][0], ret[1][0], ret[2][0]]
                elif ret_xyz or ret_rij:
                    return [ret[0][0], ret[1][0]]
                return ret[0][0]
            if ret_xyz or ret_rij:
                return ret
            return ret[0]

        if ret_xyz:
            xa = dxa + off
        del dxa  # just because this array could be very big...

        # Check whether we only have one range to check.
        # If so, we need not reduce the index space
        if len(R) == 1:
            ret = [atoms]
            if ret_xyz:
                ret.append(xa)
            if ret_rij:
                ret.append(d)
            if ret_xyz or ret_rij:
                return ret
            return ret[0]

        if not is_ascending(R):
            raise ValueError(
                f"{self.__class__.__name__}.close_sc proximity checks for several "
                "quantities at a time requires ascending R values."
            )

        # The more neigbours you wish to find the faster this becomes
        # We only do "one" heavy duty search,
        # then we immediately reduce search space to this subspace
        tidx = indices_le(d, R[0])
        ret = [[atoms[tidx]]]
        r_app = ret[0].append
        if ret_xyz:
            ret.append([xa[tidx]])
            r_appx = ret[1].append
        if ret_rij:
            ret.append([d[tidx]])
            r_appd = ret[-1].append

        if ret_xyz and ret_rij:
            for i in range(1, len(R)):
                # Search in the sub-space
                # Notice that this sub-space reduction will never
                # allow the same indice to be in two ranges (due to
                # numerics)
                tidx = indices_gt_le(d, R[i - 1], R[i])
                r_app(atoms[tidx])
                r_appx(xa[tidx])
                r_appd(d[tidx])
        elif ret_xyz:
            for i in range(1, len(R)):
                tidx = indices_gt_le(d, R[i - 1], R[i])
                r_app(atoms[tidx])
                r_appx(xa[tidx])
        elif ret_rij:
            for i in range(1, len(R)):
                tidx = indices_gt_le(d, R[i - 1], R[i])
                r_app(atoms[tidx])
                r_appd(d[tidx])
        else:
            for i in range(1, len(R)):
                tidx = indices_gt_le(d, R[i - 1], R[i])
                r_app(atoms[tidx])

        if ret_xyz or ret_rij:
            return ret
        return ret[0]

    def bond_correct(
        self, ia: int, atoms: AtomsIndex, method: Union[str, float] = "calc"
    ) -> None:
        """Corrects the bond between `ia` and the `atoms`.

        Corrects the bond-length between atom `ia` and `atoms` in such
        a way that the atomic radius is preserved.
        I.e. the sum of the bond-lengths minimizes the distance matrix.

        Only atom `ia` is moved.

        Parameters
        ----------
        ia :
            The atom to be displaced according to the atomic radius
        atoms :
            The atom(s) from which the radius should be reduced.
        method :
            If str will use that as lookup in `Atom.radius`.
            Else it will be the new bond-length.
        """

        # Decide which algorithm to choose from
        atoms = self._sanitize_atoms(atoms).ravel()
        if len(atoms) == 1:
            algo = atoms[0]
        else:
            # signal a list of atoms
            algo = -1

        if algo >= 0:
            # We have a single atom
            # Get bond length in the closest direction
            # A bond-length HAS to be below 10
            atoms, c, d = self.close(
                ia, R=(0.1, 10.0), atoms=algo, ret_xyz=True, ret_rij=True
            )
            i = np.argmin(d[1])
            # Convert to unitcell atom (and get the one atom)
            atoms = self.asc2uc(atoms[1][i])
            c = c[1][i]
            d = d[1][i]

            # Calculate the bond vector
            bv = self.xyz[ia, :] - c

            try:
                # If it is a number, we use that.
                rad = float(method)
            except Exception:
                # get radius
                rad = self.atoms[atoms].radius(method) + self.atoms[ia].radius(method)

            # Update the coordinate
            self.xyz[ia, :] = c + bv / d * rad

        else:
            raise NotImplementedError(
                "Changing bond-length dependent on several lacks implementation."
            )

    def within(
        self,
        shapes,
        atoms: AtomsIndex = None,
        atoms_xyz=None,
        ret_xyz: bool = False,
        ret_rij: bool = False,
        ret_isc: bool = False,
    ):
        """Indices of atoms in the entire supercell within a given shape from a given coordinate

        This heavily relies on the `within_sc` method.

        Note that if a connection is made in a neighboring super-cell
        then the atomic index is shifted by the super-cell index times
        number of atoms.
        This allows one to decipher super-cell atoms from unit-cell atoms.

        Parameters
        ----------
        shapes : Shape, list of Shape
        atoms :
            List of indices for atoms that are to be considered
        atoms_xyz : array_like, optional
            The atomic coordinates of the equivalent `atoms` variable (`atoms` must also be passed)
        ret_xyz :
            If true this method will return the coordinates
            for each of the couplings.
        ret_rij :
            If true this method will return the distances from the `xyz_ia`
            for each of the couplings.
        ret_isc :
            If true this method will return the supercell offsets for each of the couplings.

        Returns
        -------
        index
            indices of atoms (in supercell indices) within the shape
        xyz
            atomic coordinates of the indexed atoms (only for true `ret_xyz`)
        rij
            distance of the indexed atoms to the center of the shape (only for true `ret_rij`)
        isc
            supercell indices of the couplings (only for true `ret_isc`)
        """

        # Ensure that `shapes` is a list
        if isinstance(shapes, Shape):
            shapes = [shapes]
        nshapes = len(shapes)

        ret = [[np.empty([0], np.int32)] * nshapes]
        i = 0
        if ret_xyz:
            ixyz = i + 1
            i += 1
            ret.append([np.empty([0, 3], np.float64)] * nshapes)
        if ret_rij:
            irij = i + 1
            i += 1
            ret.append([np.empty([0], np.float64)] * nshapes)
        if ret_isc:
            iisc = i + 1
            i += 1
            ret.append([np.empty([0, 3], np.int32)] * nshapes)

        # number of special returns
        n_ret = i
        listify = n_ret == 0 or (n_ret == 1 and ret_isc)

        def isc_tile(isc, n):
            return tile(isc.reshape(1, -1), (n, 1))

        for s in range(self.n_s):
            na = self.na * s
            isc = self.lattice.sc_off[s, :]
            sret = self.within_sc(
                shapes,
                self.lattice.sc_off[s, :],
                atoms=atoms,
                atoms_xyz=atoms_xyz,
                ret_xyz=ret_xyz,
                ret_rij=ret_rij,
            )

            if listify:
                # This is to "fake" the return
                # of a list (we will do indexing!)
                sret = [sret]

            if isinstance(sret[0], list):
                # we have a list of arrays (nshapes > 1)
                for i, x in enumerate(sret[0]):
                    ret[0][i] = concatenate((ret[0][i], x + na), axis=0)
                    if ret_xyz:
                        ret[ixyz][i] = concatenate(
                            (ret[ixyz][i], sret[ixyz][i]), axis=0
                        )
                    if ret_rij:
                        ret[irij][i] = concatenate(
                            (ret[irij][i], sret[irij][i]), axis=0
                        )
                    if ret_isc:
                        ret[iisc][i] = concatenate(
                            (ret[iisc][i], isc_tile(isc, len(x))), axis=0
                        )
            elif len(sret[0]) > 0:
                # We can add it to the list (nshapes == 1)
                # We add the atomic offset for the supercell index
                ret[0][0] = concatenate((ret[0][0], sret[0] + na), axis=0)
                if ret_xyz:
                    ret[ixyz][0] = concatenate((ret[ixyz][0], sret[ixyz]), axis=0)
                if ret_rij:
                    ret[irij][0] = concatenate((ret[irij][0], sret[irij]), axis=0)
                if ret_isc:
                    ret[iisc][0] = concatenate(
                        (ret[iisc][0], isc_tile(isc, len(sret[0]))), axis=0
                    )

        if nshapes == 1:
            if n_ret == 0:
                return ret[0][0]
            return tuple(ret[i][0] for i in range(n_ret + 1))

        if n_ret == 0:
            return ret[0]
        return ret

    def close(
        self,
        xyz_ia,
        R=None,
        atoms: AtomsIndex = None,
        atoms_xyz=None,
        ret_xyz: bool = False,
        ret_rij: bool = False,
        ret_isc: bool = False,
    ):
        """Indices of atoms in the entire supercell within a given radius from a given coordinate

        This heavily relies on the `close_sc` method.

        Note that if a connection is made in a neighboring super-cell
        then the atomic index is shifted by the super-cell index times
        number of atoms.
        This allows one to decipher super-cell atoms from unit-cell atoms.

        Parameters
        ----------
        xyz_ia : coordinate/index
            Either a point in space or an index of an atom.
            If an index is passed it is the equivalent of passing
            the atomic coordinate ``close_sc(self.xyz[xyz_ia,:])``.
        R : (None), float/tuple of float
            The radii parameter to where the atomic connections are found.
            If `R` is an array it will return the indices:
            in the ranges:

            >>> ( x <= R[0] , R[0] < x <= R[1], R[1] < x <= R[2] )

            If a single float it will return:

            >>> x <= R

        atoms :
            List of indices for atoms that are to be considered
        atoms_xyz : array_like, optional
            The atomic coordinates of the equivalent `atoms` variable (`atoms` must also be passed)
        ret_xyz :
            If true this method will return the coordinates
            for each of the couplings.
        ret_rij :
            If true this method will return the distances from the `xyz_ia`
            for each of the couplings.
        ret_isc :
            If true this method will return the lattice offset from `xyz_ia`
            for each of the couplings.

        Returns
        -------
        index
            indices of atoms (in supercell indices) within the shells of radius `R`
        xyz
            atomic coordinates of the indexed atoms (only for true `ret_xyz`)
        rij
            distance of the indexed atoms to the center coordinate (only for true `ret_rij`)
        isc
            integer lattice offsets for the couplings (related to `rij` without atomic coordinates)
        """
        if R is None:
            R = self.maxR() + 0.001
        R = _a.asarrayd(R).ravel()
        nR = R.size

        # Convert index coordinate to point
        if isinstance(xyz_ia, Integral):
            xyz_ia = self.xyz[xyz_ia]
        elif not isndarray(xyz_ia):
            xyz_ia = _a.asarrayd(xyz_ia)

        ret = [[np.empty([0], np.int32)] * nR]
        i = 0
        if ret_xyz:
            ixyz = i + 1
            i += 1
            ret.append([np.empty([0, 3], np.float64)] * nR)
        if ret_rij:
            irij = i + 1
            i += 1
            ret.append([np.empty([0], np.float64)] * nR)
        if ret_isc:
            iisc = i + 1
            i += 1
            ret.append([np.empty([0, 3], np.int32)] * nR)

        # number of special returns
        n_ret = i
        listify = n_ret == 0 or (n_ret == 1 and ret_isc)

        def isc_tile(isc, n):
            return tile(isc.reshape(1, -1), (n, 1))

        for s in range(self.n_s):
            na = self.na * s
            isc = self.lattice.sc_off[s]
            sret = self.close_sc(
                xyz_ia,
                isc,
                R=R,
                atoms=atoms,
                atoms_xyz=atoms_xyz,
                ret_xyz=ret_xyz,
                ret_rij=ret_rij,
            )

            if listify:
                # This is to "fake" the return
                # of a list (we will do indexing!)
                sret = [sret]

            if isinstance(sret[0], list):
                # we have a list of arrays (len(R) > 1)
                for i, x in enumerate(sret[0]):
                    ret[0][i] = concatenate((ret[0][i], x + na), axis=0)
                    if ret_xyz:
                        ret[ixyz][i] = concatenate(
                            (ret[ixyz][i], sret[ixyz][i]), axis=0
                        )
                    if ret_rij:
                        ret[irij][i] = concatenate(
                            (ret[irij][i], sret[irij][i]), axis=0
                        )
                    if ret_isc:
                        ret[iisc][i] = concatenate(
                            (ret[iisc][i], isc_tile(isc, len(x))), axis=0
                        )
            elif len(sret[0]) > 0:
                # We can add it to the list (len(R) == 1)
                # We add the atomic offset for the supercell index
                ret[0][0] = concatenate((ret[0][0], sret[0] + na), axis=0)
                if ret_xyz:
                    ret[ixyz][0] = concatenate((ret[ixyz][0], sret[ixyz]), axis=0)
                if ret_rij:
                    ret[irij][0] = concatenate((ret[irij][0], sret[irij]), axis=0)
                if ret_isc:
                    ret[iisc][0] = concatenate(
                        (ret[iisc][0], isc_tile(isc, len(sret[0]))), axis=0
                    )

        if nR == 1:
            if n_ret == 0:
                return ret[0][0]
            return tuple(ret[i][0] for i in range(n_ret + 1))

        if n_ret == 0:
            return ret[0]
        return ret

    def a2transpose(
        self, atoms1: AtomsIndex, atoms2: AtomsIndex = None
    ) -> tuple[ndarray, ndarray]:
        """Transposes connections from `atoms1` to `atoms2` such that supercell connections are transposed

        When handling supercell indices it is useful to get the *transposed* connection. I.e. if you have
        a connection from site ``i`` (in unit cell indices) to site ``j`` (in supercell indices) it may be
        useful to get the equivalent supercell connection such for site ``j`` (in unit cell indices) to
        site ``i`` (in supercell indices) such that they correspond to the transposed coupling.

        Note that since this transposes couplings the indices returned are always expanded to the full
        length if either of the inputs are a single index.

        Examples
        --------
        >>> gr = geom.graphene()
        >>> atoms = gr.close(0, 1.5)
        >>> atoms
        array([0, 1, 5, 9], dtype=int32)
        >>> gr.a2transpose(0, atoms)
        (array([0, 1, 1, 1], dtype=int32), array([ 0,  0, 14, 10], dtype=int32))

        Parameters
        ----------
        atoms1 :
            atomic indices must have same length as `atoms2` or length 1
        atoms2 :
            atomic indices must have same length as `atoms1` or length 1.
            If not present then only `atoms1` will be returned in transposed indices.

        Returns
        -------
        atoms2 : array_like
            transposed indices for atoms2 (only returned if `atoms2` is not None)
        atoms1 : array_like
            transposed indices for atoms1
        """
        # First check whether they have the same size, if so then do not pre-process
        atoms1 = self._sanitize_atoms(atoms1)
        if atoms2 is None:
            # we only need to transpose atoms1
            offset = self.lattice.sc_index(-self.a2isc(atoms1)) * self.na
            return atoms1 % self.na + offset

        atoms2 = self._sanitize_atoms(atoms2)
        if atoms1.size == atoms2.size:
            pass
        elif atoms1.size == 1:  # typical case where atoms1 is a single number
            atoms1 = np.tile(atoms1, atoms2.size)
        elif atoms2.size == 1:
            atoms2 = np.tile(atoms2, atoms1.size)
        else:
            raise ValueError(
                f"{self.__class__.__name__}.a2transpose only allows length 1 or same length arrays."
            )

        # Now convert atoms
        na = self.na
        sc_index = self.lattice.sc_index
        isc1 = self.a2isc(atoms1)
        isc2 = self.a2isc(atoms2)

        atoms1 = atoms1 % na + sc_index(-isc2) * na
        atoms2 = atoms2 % na + sc_index(-isc1) * na
        return atoms2, atoms1

    def o2transpose(
        self, orb1: OrbitalsIndex, orb2: Optional[OrbitalsIndex] = None
    ) -> tuple[ndarray, ndarray]:
        """Transposes connections from `orb1` to `orb2` such that supercell connections are transposed

        When handling supercell indices it is useful to get the *transposed* connection. I.e. if you have
        a connection from site ``i`` (in unit cell indices) to site ``J`` (in supercell indices) it may be
        useful to get the equivalent supercell connection such for site ``j`` (in unit cell indices) to
        site ``I`` (in supercell indices) such that they correspond to the transposed coupling.

        Note that since this transposes couplings the indices returned are always expanded to the full
        length if either of the inputs are a single index.

        Examples
        --------
        >>> gr = geom.graphene() # one orbital per site
        >>> atoms = gr.close(0, 1.5)
        >>> atoms
        array([0, 1, 5, 9], dtype=int32)
        >>> gr.o2transpose(0, atoms)
        (array([0, 1, 1, 1], dtype=int32), array([ 0,  0, 14, 10], dtype=int32))

        Parameters
        ----------
        orb1 :
            orbital indices must have same length as `orb2` or length 1
        orb2 :
            orbital indices must have same length as `orb1` or length 1.
            If not present then only `orb1` will be returned in transposed indices.

        Returns
        -------
        orb2 : array_like
            transposed indices for orb2 (only returned if `orb2` is not None)
        orb1 : array_like
            transposed indices for orb1
        """
        # First check whether they have the same size, if so then do not pre-process
        orb1 = self._sanitize_orbs(orb1)
        if orb2 is None:
            # we only need to transpose orb1
            offset = self.lattice.sc_index(-self.o2isc(orb1)) * self.no
            return orb1 % self.no + offset

        orb2 = self._sanitize_orbs(orb2)
        if orb1.size == orb2.size:
            pass
        elif orb1.size == 1:  # typical case where orb1 is a single number
            orb1 = np.tile(orb1, orb2.size)
        elif orb2.size == 1:
            orb2 = np.tile(orb2, orb1.size)
        else:
            raise ValueError(
                f"{self.__class__.__name__}.o2transpose only allows length 1 or same length arrays."
            )

        # Now convert orbs
        no = self.no
        sc_index = self.lattice.sc_index
        isc1 = self.o2isc(orb1)
        isc2 = self.o2isc(orb2)

        orb1 = orb1 % no + sc_index(-isc2) * no
        orb2 = orb2 % no + sc_index(-isc1) * no
        return orb2, orb1

    def a2o(self, atoms: AtomsIndex, all: bool = False) -> ndarray:
        """
        Returns an orbital index of the first orbital of said atom.
        This is particularly handy if you want to create
        TB models with more than one orbital per atom.

        Note that this will preserve the super-cell offsets.

        Parameters
        ----------
        atoms :
             Atomic indices
        all :
             ``False``, return only the first orbital corresponding to the atom,
             ``True``, returns list of the full atom(s), will always return a 1D array.
        """
        # we must not alter `atoms` as it may come from outside
        off, atoms = np.divmod(self._sanitize_atoms(atoms), self.na)
        is_integral = isinstance(atoms, Integral)
        off *= self.no
        if not all:
            return self.firsto[atoms] + off
        ob = (self.firsto[atoms] + off).ravel()
        oe = (self.lasto[atoms] + off + 1).ravel()

        # Create ranges
        if is_integral:
            return _a.arangei(ob[0], oe[0])

        return _a.array_arange(ob, oe)

    def o2a(self, orbitals: OrbitalsIndex, unique: bool = False) -> ndarray:
        """Atomic index corresponding to the orbital indicies.

        Note that this will preserve the super-cell offsets.

        Parameters
        ----------
        orbitals :
             List of orbital indices to return the atoms for
        unique :
             If True only return the unique atoms.
        """
        orbitals = self._sanitize_orbs(orbitals)
        if orbitals.ndim == 0:
            # must only be 1 number (an Integral)
            return (
                np.argmax(orbitals % self.no <= self.lasto)
                + (orbitals // self.no) * self.na
            )

        isc, orbitals = np.divmod(_a.asarrayi(orbitals), self.no)
        a = list_index_le(orbitals.ravel(), self.lasto).reshape(orbitals.shape)
        if unique:
            return np.unique(a + isc * self.na)
        return a + isc * self.na

    def auc2sc(self, atoms: AtomsIndex, unique: bool = False) -> ndarray:
        """Returns atom from unit-cell indices to supercell indices, possibly removing duplicates

        Parameters
        ----------
        atoms :
           the atomic unit-cell indices to be converted to supercell indices
        unique :
           If True the returned indices are unique and sorted.
        """
        atoms = self._sanitize_atoms(atoms) % self.na
        atoms = (atoms[..., None] + _a.arangei(self.n_s) * self.na).reshape(
            *atoms.shape[:-1], -1
        )
        if unique:
            return np.unique(atoms)
        return atoms

    uc2sc = deprecation(
        "uc2sc is deprecated, update the code to use the explicit form auc2sc",
        "0.15.0",
        "0.16.0",
    )(auc2sc)

    def asc2uc(self, atoms: AtomsIndex, unique: bool = False) -> ndarray:
        """Returns atoms from supercell indices to unit-cell indices, possibly removing duplicates

        Parameters
        ----------
        atoms :
           the atomic supercell indices to be converted to unit-cell indices
        unique :
           If True the returned indices are unique and sorted.
        """
        atoms = self._sanitize_atoms(atoms) % self.na
        if unique:
            return np.unique(atoms)
        return atoms

    sc2uc = deprecation(
        "sc2uc is deprecated, update the code to use the explicit form asc2uc",
        "0.15.0",
        "0.16.0",
    )(asc2uc)

    def osc2uc(self, orbitals: OrbitalsIndex, unique: bool = False) -> ndarray:
        """Orbitals from supercell indices to unit-cell indices, possibly removing duplicates

        Parameters
        ----------
        orbitals :
           the orbital supercell indices to be converted to unit-cell indices
        unique :
           If True the returned indices are unique and sorted.
        """
        orbitals = self._sanitize_orbs(orbitals) % self.no
        if unique:
            return np.unique(orbitals)
        return orbitals

    def ouc2sc(self, orbitals: OrbitalsIndex, unique: bool = False) -> ndarray:
        """Orbitals from unit-cell indices to supercell indices, possibly removing duplicates

        Parameters
        ----------
        orbitals :
           the orbital unit-cell indices to be converted to supercell indices
        unique :
           If True the returned indices are unique and sorted.
        """
        orbitals = self._sanitize_orbs(orbitals) % self.no
        orbitals = (orbitals[..., None] + _a.arangei(self.n_s) * self.no).reshape(
            *orbitals.shape[:-1], -1
        )
        if unique:
            return np.unique(orbitals)
        return orbitals

    def a2isc(self, atoms: AtomsIndex) -> ndarray:
        """Super-cell indices for a specific/list atom

        Returns a vector of 3 numbers with integers.
        Any multi-dimensional input will be flattened before return.

        The returned indices will thus always be a 2D matrix or a 1D vector.

        Parameters
        ----------
        atoms :
            atom indices to extract the supercell locations of
        """
        atoms = self._sanitize_atoms(atoms) // self.na
        return self.lattice.sc_off[atoms, :]

    # This function is a bit weird, it returns a real array,
    # however, there should be no ambiguity as it corresponds to th
    # offset and "what else" is there to query?
    def a2sc(self, atoms: AtomsIndex) -> ndarray:
        """Returns the super-cell offset for a specific atom

        Parameters
        ----------
        atoms :
            atom indices to extract the supercell offsets of
        """
        return self.lattice.offset(self.a2isc(atoms))

    def o2isc(self, orbitals: OrbitalsIndex) -> ndarray:
        """
        Returns the super-cell index for a specific orbital.

        Returns a vector of 3 numbers with integers.
        """
        orbitals = self._sanitize_orbs(orbitals) // self.no
        return self.lattice.sc_off[orbitals, :]

    def o2sc(self, orbitals: OrbitalsIndex) -> ndarray:
        """
        Returns the super-cell offset for a specific orbital.
        """
        return self.lattice.offset(self.o2isc(orbitals))

    @deprecate_argument(
        "tol",
        "atol",
        "argument tol has been deprecated in favor of atol, please update your code.",
        "0.15",
        "0.17",
    )
    def equal(self, other: GeometryLike, R: bool = True, atol: float = 1e-4) -> bool:
        """Whether two geometries are the same (optional not check of the orbital radius)

        Parameters
        ----------
        other :
            the other Geometry to check against
        R :
            if True also check if the orbital radii are the same (see `Atom.equal`)
        atol :
            tolerance for checking the atomic coordinates
        """
        other = self.new(other)
        if not isinstance(other, Geometry):
            return False
        same = self.lattice.equal(other.lattice, atol=atol)
        same = same and np.allclose(self.xyz, other.xyz, atol=atol)
        same = same and self.atoms.equal(other.atoms, R)
        return same

    def __eq__(self, other):
        return self.equal(other)

    def __ne__(self, other):
        return not (self == other)

    def sparserij(self, dtype=np.float64, na_iR: int = 1000, method: str = "rand"):
        """Return the sparse matrix with all distances in the matrix
        The sparse matrix will only be defined for the elements which have
        orbitals overlapping with other atoms.

        Parameters
        ----------
        dtype : numpy.dtype, numpy.float64
           the data-type of the sparse matrix
        na_iR :
           number of atoms within the sphere for speeding
           up the `iter_block` loop.
        method :
           see `iter_block` for details

        Returns
        -------
        SparseAtom
           sparse matrix with all rij elements

        See Also
        --------
        iter_block : the method for looping the atoms
        distance : create a list of distances
        """
        from .sparse_geometry import SparseAtom

        rij = SparseAtom(self, nnzpr=20, dtype=dtype)

        # Get R
        R = (0.1, self.maxR() + 0.001)
        iR = self.iR(na_iR)

        # Do the loop
        for ias, atoms in self.iter_block(iR=iR, method=method):
            # Get all the indexed atoms...
            # This speeds up the searching for
            # coordinates...
            atoms_xyz = self[atoms, :]

            # Loop the atoms inside
            for ia in ias:
                idx, r = self.close(
                    ia, R=R, atoms=atoms, atoms_xyz=atoms_xyz, ret_rij=True
                )
                rij[ia, ia] = 0.0
                rij[ia, idx[1]] = r[1]

        return rij

    @deprecate_argument(
        "tol",
        "atol",
        "argument tol has been deprecated in favor of atol, please update your code.",
        "0.15",
        "0.17",
    )
    def distance(
        self,
        atoms: AtomsIndex = None,
        R: Optional[float] = None,
        atol: Union[float, Sequence[float]] = 0.1,
        method: Union[
            Callable[[Sequence[float]], float],
            Literal["average", "mode", "<numpy.method>"],
        ] = "average",
    ) -> Union[float, ndarray]:
        """Calculate the distances for all atoms in shells of radius `tol` within `max_R`

        Parameters
        ----------
        atoms :
           only create list of distances from the given atoms, default to all atoms
        R :
           the maximum radius to consider, default to ``self.maxR()``.
           To retrieve all distances for atoms within the supercell structure
           you can pass `numpy.inf`.
        atol :
           the tolerance for grouping a set of atoms.
           This parameter sets the shell radius for each shell.
           I.e. the returned distances between two shells will be maximally
           ``2*atol``, but only if atoms are within two consecutive lists.
           If this is a list, the shells will be of unequal size.

           The first shell size will be ``atol * .5`` or ``atol[0] * .5`` if `atol` is a list.

        method :
           How the distance in each shell is determined.
           A list of distances within each shell is gathered and the equivalent
           method will be used to extract a single quantity from the list of
           distances in the shell.
           If `'mode'` is chosen it will use `scipy.stats.mode`.
           If another string is given it will correspond to ``getattr(numpy, method)``,
           while any callable function may be passed. The passed function
           will only be passed a list of unsorted distances that needs to be
           processed.

        Notes
        -----
        Using ``method='mode'`` requires ``scipy>=1.9``.


        Examples
        --------
        >>> geom = Geometry([0]*3, Atom(1, R=1.), lattice=Lattice(1., nsc=[5, 5, 1]))
        >>> geom.distance()
        array([1.])
        >>> geom.distance(atol=[0.5, 0.4, 0.3, 0.2])
        array([1.])
        >>> geom.distance(R=2, atol=[0.5, 0.4, 0.3, 0.2])
        array([1.        ,  1.41421356,  2.        ])
        >>> geom.distance(R=2, atol=[0.5, 0.7]) # the R = 1 and R = 2 ** .5 gets averaged
        array([1.20710678,  2.        ])

        Returns
        -------
        numpy.ndarray
           an array of positive numbers yielding the distances from the atoms in reduced form

        See Also
        --------
        sparserij : return a sparse matrix will all distances between atoms
        """
        atoms = self._sanitize_atoms(atoms).ravel()

        # Figure out maximum distance
        if R is None:
            R = self.maxR()
            if R < 0:
                raise ValueError(
                    f"{self.__class__.__name__}"
                    ".distance cannot determine the `R` parameter. "
                    "The internal `maxR()` is negative and thus not set. "
                    "Set an explicit value for `R`."
                )
        elif np.any(self.pbc):
            maxR = fnorm(self.cell).max()
            # These loops could be leveraged if we look at angles...
            for i, j, k in product(
                [0, self.nsc[0] // 2], [0, self.nsc[1] // 2], [0, self.nsc[2] // 2]
            ):
                if i == 0 and j == 0 and k == 0:
                    continue
                sc = [i, j, k]
                off = self.lattice.offset(sc)

                for ii, jj, kk in product([0, 1], [0, 1], [0, 1]):
                    o = self.cell[0] * ii + self.cell[1] * jj + self.cell[2] * kk
                    maxR = max(maxR, fnorm(off + o))

            if R > maxR:
                R = maxR

        # Convert to list
        atol = _a.asarray(atol).ravel()
        if len(atol) == 1:
            # Now we are in a position to determine the sizes
            dR = _a.aranged(atol[0] * 0.5, R + atol[0] * 0.55, atol[0])
        else:
            dR = atol.copy()
            dR[0] *= 0.5
            # The first tolerance, is for it-self, the second
            # has to have the first tolerance as the field
            dR = _a.cumsumd(np.insert(dR, 1, atol[0]))

            if dR[-1] < R:
                # Now finalize dR by ensuring all remaining segments are captured
                t = atol[-1]

                dR = concatenate((dR, _a.aranged(dR[-1] + t, R + t * 0.55, t)))

            # Reduce to the largest value above R
            # This ensures that R, truly is the largest considered element
            dR = dR[: (dR > R).nonzero()[0][0] + 1]

        # Now we can figure out the list of atoms in each shell
        # First create the initial lists of shell atoms
        # The inner shell will never be used, because it should correspond
        # to the atom it-self.
        shells = [[] for i in range(len(dR) - 1)]

        for a in atoms:
            _, r = self.close(a, R=dR, ret_rij=True)

            for i, rlist in enumerate(r[1:]):
                shells[i].extend(rlist)

        # Now parse all of the shells with the correct routine
        # First we grap the routine:
        if isinstance(method, str):
            if method == "median":

                def func(lst):
                    return np.median(lst, overwrite_input=True)

            elif method == "mode":
                from scipy.stats import mode

                def func(lst):
                    # We don't need keepdims=False, because an array of size 1
                    # can be broadcasted to a single element.
                    return mode(lst)[0]

            else:
                try:
                    func = getattr(np, method)
                except AttributeError as e:
                    raise ValueError(
                        f"{self.__class__.__name__}.distance `method` got wrong input value."
                    ) from e
        else:
            func = method

        # Reduce lists
        for i in range(len(shells)):
            lst = shells[i]
            if len(lst) == 0:
                continue

            # Reduce elements
            shells[i] = func(lst)

        # Convert to flattened numpy array and ensure shape
        d = np.hstack(shells).ravel()

        return d

    @deprecate_argument(
        "tol",
        "atol",
        "argument tol has been deprecated in favor of atol, please update your code.",
        "0.15",
        "0.17",
    )
    def within_inf(
        self,
        lattice: Lattice,
        periodic: Optional[Union[Sequence[bool], CellAxes]] = None,
        atol: float = 1e-5,
        origin: Sequence[float] = (0.0, 0.0, 0.0),
    ) -> tuple[ndarray, ndarray, ndarray]:
        """Find all atoms within a provided supercell

        Note this function is rather different from `close` and `within`.
        Specifically this routine is returning *all* indices for the infinite
        periodic system. The default periodic directions are ``self.pbc``,
        unless `periodic` is provided.

        Atomic coordinates lying on the boundary of the supercell will be duplicated
        on the neighboring supercell images. Thus performing ``geom.within_inf(geom.lattice)``
        may result in more atoms than in the structure.

        Notes
        -----
        The name of this function may change. Currently it should only be used
        internally in sisl.

        Parameters
        ----------
        lattice : LatticeLike
            the supercell in which this geometry should be expanded into.
        periodic :
            explicitly define the periodic directions, by default the periodic
            directions are only where ``self.pbc``.
        atol :
            length tolerance for the coordinates to be on a duplicate site (in Ang).
            This allows atoms within `atol` of the cell boundaries to be taken as *inside* the
            cell.
        origin :
            origin that is the basis for comparison, default to 0.

        Returns
        -------
        numpy.ndarray
           unit-cell atomic indices which are inside the `lattice` cell
        numpy.ndarray
           atomic coordinates for the `ia` atoms (including supercell offsets)
        numpy.ndarray
           integer supercell offsets for `ia` atoms
        """
        lattice = self.lattice.__class__.new(lattice)
        if periodic is None:
            periodic = self.pbc.nonzero()[0]
        elif isinstance(periodic, bool):
            if periodic:
                periodic = (0, 1, 2)
            else:
                periodic = ()
        else:
            try:
                periodic = map(direction, listify(periodic)) | listify
            except:
                periodic = np.asarray(periodic).nonzero()[0]

        # extract the non-periodic directions
        non_periodic = filter(lambda i: i not in periodic, range(3)) | listify

        if origin is None:
            origin = _a.zerosd(3)

        # Our first task is to construct a geometry large
        # enough to fully encompass the supercell
        # The supercell here defines how big `self` needs to be
        # to be fully located inside `lattice`.

        # 1. Number of times each lattice vector must be expanded to fit
        #    inside the "possibly" larger `lattice`.
        idx = lattice.cell @ self.icell.T
        tile_min = floor(idx.min(0)).astype(dtype=int32)
        tile_max = ceil(idx.max(0)).astype(dtype=int32)

        # Intrinsic offset (when atomic coordinates are outside primary unit-cell)
        fxyz = np.round(self.fxyz, decimals=5)
        # We don't collapse this as it is necessary for correcting isc further below
        fxyz_ifloor = floor(fxyz).astype(dtype=int32)
        fxyz_iceil = ceil(fxyz).max(0).astype(dtype=int32)
        tile_min = np.minimum(tile_min, fxyz_ifloor.min(0))
        tile_max = np.maximum(tile_max, fxyz_iceil)
        del idx, fxyz

        # 1a) correct for origin displacement
        idx = floor(lattice.origin @ self.icell.T)
        tile_min = np.minimum(tile_min, idx).astype(dtype=int32)
        idx = floor(origin @ self.icell.T)
        tile_min = np.minimum(tile_min, idx).astype(dtype=int32)

        # 2. Reduce tiling along non-periodic directions
        tile_min[non_periodic] = 0
        tile_max[non_periodic] = 1

        # 3. Find the *new* origin according to the *negative* tilings.
        #    This is important for skewed cells as the placement of the new
        #    larger geometry has to be shifted to have lattice inside
        big_origin = (tile_min.reshape(3, 1) * self.cell).sum(0)

        # The xyz geometry that fully encompass the (possibly) larger supercell
        tile = tile_max - tile_min

        full_geom = (self * tile).translate(big_origin - origin)

        # Now we have to figure out all atomic coordinates within
        cuboid = lattice.to.Cuboid()

        # Make sure that full_geom doesn't return coordinates outside the unit cell
        # for non periodic directions
        nsc = full_geom.nsc.copy() // 2

        # If we have atoms outside the primary unit-cell in the original
        # cell, then we should consider an nsc large enough to encompass this
        nsc = np.maximum(nsc, fxyz_iceil)
        nsc = np.maximum(nsc, -fxyz_ifloor.min(0))
        nsc = nsc * 2 + 1

        nsc[non_periodic] = 1
        full_geom.set_nsc(nsc)

        # Now retrieve all atomic coordinates from the full geometry
        xyz = full_geom.axyz(_a.arangei(full_geom.na_s))
        idx = cuboid.within_index(xyz)
        xyz = xyz[idx, :]
        del full_geom

        # Figure out supercell connections in the smaller indices
        # Since we have shifted all coordinates into the primary unit cell we
        # are sure that these fxyz are [0:1[
        fxyz = xyz @ self.icell.T

        # Since there are numerical errors for the above operation
        # we *have* to account for possible sign-errors
        # This is done by a length tolerance
        ftol = atol / fnorm(self.cell).reshape(1, 3)
        isc = floor(fxyz - ftol).astype(int32)

        # Now we can extract the indices where the two are non-matching.
        # At these indices we have some "errors" that we have to fix and
        # thus select the correct isc.
        idx_diff = (isc - floor(fxyz + ftol).astype(int32)).nonzero()

        # For these indices we can use the nearest integer as that
        # selects the closest. floor will ONLY be wrong for -0.0000, 0.99999, ...
        isc[idx_diff] = np.rint(fxyz[idx_diff]).astype(int32)

        # Convert indices to unit-cell indices and also return coordinates and
        # infinite supercell indices
        ia = self.asc2uc(idx)
        return ia, xyz, isc - fxyz_ifloor[ia]

    def _orbital_values(
        self, grid_shape: tuple[int, int, int], truncate_with_nsc: bool = False
    ):
        r"""Calculates orbital values for a given grid.

        Parameters
        ----------
        grid_shape:
           the grid shape (i.e. resolution) in which to calculate the orbital values.
        truncate_with_nsc:
            if True, only consider atoms within the geometry's auxiliary cell.

        Notes
        -----
        This method does not belong on this geometry. It will be removed eventually.
        """
        # We need to import these here to avoid circular imports.
        from sisl import Grid
        from sisl._sparse_grid import SparseGridOrbitalBZ

        # In the following we don't care about division
        # So 1) save error state, 2) turn off divide by 0, 3) calculate, 4) turn on old error state
        old_err = np.seterr(divide="ignore", invalid="ignore")

        # Instead of looping all atoms in the supercell we find the exact atoms
        # and their supercell indices.
        add_R = _a.fulld(3, self.maxR())
        # Calculate the required additional vectors required to increase the fictitious
        # supercell by add_R in each direction.
        # For extremely skewed lattices this will be way too much, hence we make
        # them square.
        o = self.lattice.to.Cuboid(orthogonal=True)
        lattice = Lattice(o._v + np.diag(2 * add_R), origin=o.origin - add_R)

        # Retrieve all atoms within the grid supercell
        # (and the neighbours that connect into the cell)
        IA, XYZ, ISC = self.within_inf(lattice, periodic=self.pbc)
        XYZ -= self.lattice.origin.reshape(1, 3)

        # Don't consider atoms that are outside of the geometry's auxiliary cell.
        if truncate_with_nsc:
            mask = (abs(ISC) <= self.nsc // 2).all(axis=1)
            IA, XYZ, ISC = IA[mask], XYZ[mask], ISC[mask]

        def xyz2spherical(xyz, offset):
            """Calculate the spherical coordinates from indices"""
            rx = xyz[:, 0] - offset[0]
            ry = xyz[:, 1] - offset[1]
            rz = xyz[:, 2] - offset[2]

            xyz_to_spherical_cos_phi(rx, ry, rz)
            return rx, ry, rz

        def sphere_grid_index(grid, center, R):

            corners = np.mgrid[-1:2:2, -1:2:2, -1:2:2].T * R + center
            corners = corners.reshape(-1, 3)

            corners_i = grid.index(corners)

            cmin = corners_i.min(axis=0)
            cmax = corners_i.max(axis=0) + 1
            sh = grid.shape

            # direct if-statements are 4-5 times faster than min+max
            # These subsequent 25 lines are equivalent to:
            # cmin = np.maximum(0, np.minimum(cmin, sh))
            # the numpy equivalents are way too slow in this case.
            if cmin[0] < 0:
                cmin[0] = 0
            elif sh[0] < cmin[0]:
                cmin[0] = sh[0]
            if cmin[1] < 0:
                cmin[1] = 0
            elif sh[1] < cmin[1]:
                cmin[1] = sh[1]
            if cmin[2] < 0:
                cmin[2] = 0
            elif sh[2] < cmin[2]:
                cmin[2] = sh[2]

            if cmax[0] < 0:
                cmax[0] = 0
            elif sh[0] < cmax[0]:
                cmax[0] = sh[0]
            if cmax[1] < 0:
                cmax[1] = 0
            elif sh[1] < cmax[1]:
                cmax[1] = sh[1]
            if cmax[2] < 0:
                cmax[2] = 0
            elif sh[2] < cmax[2]:
                cmax[2] = sh[2]

            rx = slice(cmin[0], cmax[0])
            ry = slice(cmin[1], cmax[1])
            rz = slice(cmin[2], cmax[2])

            indices = np.mgrid[rx, ry, rz].reshape(3, -1).T

            return indices

        # Get the size of the auxiliary supercell needed to store orbital values.
        nsc = abs(ISC).max(axis=0) * 2 + 1
        sp_grid_geom = self.copy()
        sp_grid_geom.set_nsc(nsc)

        # Initialize a fake grid to compute some quantities related to the grid distribution
        grid = Grid(grid_shape, geometry=self)

        # Estimate a top limit on how many values we need to store. We estimate it by expecting
        # each orbital to fill a sphere of radius R, being R the radius of the orbital. We also
        # add a margin of 1 voxel so that we don't underestimate because of rounding.
        dvolume = grid.dvolume
        margin_R = np.linalg.norm(grid.dcell.sum(axis=0))
        vol = 0.0
        for atom, indices in self.sub(IA).atoms.iter(species=True):
            vol += (4 / 3 * np.pi * (atom.R + margin_R) ** 3).sum() * len(indices)

        max_vals = int(vol / dvolume)

        # Array storing all the grid values
        grid_values = np.zeros(max_vals, dtype=np.float64)
        # Orbital indices for each orbital that has a nonzero value in the grid.
        orbital_indices = np.full(max_vals, -1, dtype=np.int32)
        # For each value, its index of the grid. Even if the grid is 3 dimensional,
        # we store the raveled index. That is, a single integer representing the position
        # of the point. One can always unravel the index if needed.
        grid_indices = np.zeros(max_vals, dtype=np.int32)

        # print(
        #     f"Estimated memory required:",
        #     (orbital_indices.size * 32 + grid_values.size * 64 + grid_indices.size * 32) / 8 / 1024 / 1024,
        #     "MB"
        # )

        # Temporal variables that will help us keep track of the construction of the arrays.
        i_value = 0
        first_orbs = self.firsto
        isc_off = sp_grid_geom.isc_off

        # Loop over all atoms in the grid-cell
        for ia, ia_xyz, isc in zip(IA, XYZ, ISC):
            # Get current atom
            atom = self.atoms[ia]

            # Get the index of the cell where this atom is in the auxiliary supercell
            index_sc = isc_off[isc[0], isc[1], isc[2]]
            # And use it to calculate the offset on the orbital index.
            io_offset = self.no * index_sc

            # Extract maximum R
            R = atom.maxR()

            if R <= 0.0:
                warn(f"Atom '{atom}' does not have a wave-function, skipping atom.")
                continue

            idx = sphere_grid_index(grid, ia_xyz, R)

            if len(idx) == 0:
                continue

            # Get real-space coordinates for the atom
            grid_xyz = dot(idx, grid.dcell)
            # Convert them to spherical coordinates
            at_r, at_theta, at_cos_phi = xyz2spherical(grid_xyz, ia_xyz)

            del grid_xyz
            # Merge the three components of spherical coordinates into one array.
            at_spherical = np.array([at_r, at_theta, at_cos_phi]).T

            # Filter out points where the distance to the atom is less than its max R.
            at_nonzero = at_spherical[:, 0] < R
            idx = idx[at_nonzero]
            at_spherical = at_spherical[at_nonzero]

            if len(idx) == 0:
                continue

            # Ravel multi index to save space. That is, convert the 3D grid index
            # into a single integer. One can always unravel them if needed.
            idx = (
                idx[:, 0] * grid.shape[1] * grid.shape[2]
                + idx[:, 1] * grid.shape[2]
                + idx[:, 2]
            )

            # Loop over the orbitals
            for io, orb in enumerate(atom.orbitals):
                # Get the index of this orbital
                uc_io = first_orbs[ia] + io

                orb_spherical = at_spherical
                orb_indices = idx

                # The orbital's R might not be the maximum R of the atom. In that case,
                # we don't need to calculate the values for all the grid points that are within
                # the atom's range.
                if R - orb.R > 1e-6:
                    # Check which coordinates are not within this orbital's range (the radius is bigger than orbital radius)
                    orb_nonzero = orb_spherical[:, 0] < orb.R

                    orb_spherical = orb_spherical[orb_nonzero]
                    orb_indices = orb_indices[orb_nonzero]

                # Number of grid values that we are going to compute for this orbital
                orb_nvals = orb_spherical.shape[0]

                # If there are no values to add, go to the next orbital
                if orb_nvals == 0:
                    continue

                # Compute the psi values for the grid points we are interested in
                psi = orb.psi_spher(*orb_spherical.T, cos_phi=True)

                # Update the data structure
                values_i = slice(i_value, i_value + orb_nvals)
                grid_values[values_i] = psi
                grid_indices[values_i] = orb_indices
                orbital_indices[values_i] = uc_io + io_offset

                # Update the index where new values should be stored
                i_value += orb_nvals

        # Reset the error code for division
        np.seterr(**old_err)

        # Cut the arrays to return only the parts that have been filled
        grid_values = grid_values[:i_value]
        grid_indices = grid_indices[:i_value]
        orbital_indices = orbital_indices[:i_value]

        psi_values = csr_matrix(
            (grid_values, (grid_indices, orbital_indices)),
            shape=(np.prod(grid.shape), sp_grid_geom.no_s),
        )

        return SparseGridOrbitalBZ(grid.shape, psi_values, geometry=sp_grid_geom)

    # Create pickling routines
    def __getstate__(self):
        """Returns the state of this object"""
        d = self.lattice.__getstate__()
        d["xyz"] = self.xyz
        d["atoms"] = self.atoms.__getstate__()
        return d

    def __setstate__(self, d):
        """Re-create the state of this object"""
        lattice = Lattice([1, 1, 1])
        lattice.__setstate__(d)
        atoms = Atoms()
        atoms.__setstate__(d["atoms"])
        self.__init__(d["xyz"], atoms=atoms, lattice=lattice)

    @classmethod
    def _ArgumentParser_args_single(cls):
        """Returns the options for `Geometry.ArgumentParser` in case they are the only options"""
        return {
            "limit_arguments": False,
            "short": True,
            "positional_out": True,
        }

    # Hook into the Geometry class to create
    # an automatic ArgumentParser which makes actions
    # as the options are read.
    @default_ArgumentParser(description="Manipulate a Geometry object in sisl.")
    def ArgumentParser(self, p=None, *args, **kwargs):
        """Create and return a group of argument parsers which manipulates it self `Geometry`.

        Parameters
        ----------
        parser : ArgumentParser, optional
           in case the arguments should be added to a specific parser. It defaults
           to create a new.
        limit_arguments : bool, optional
           If ``False`` additional options will be created which are similar to other options.
           For instance ``--repeat-x <>`` which is equivalent to ``--repeat <> x``.
           Default `True`.
        short : bool, optional
           Create short options for a selected range of options.
        positional_out : bool, optional
           If ``True``, adds a positional argument which acts as --out. This may be handy if only the geometry is in the argument list.
        """
        limit_args = kwargs.get("limit_arguments", True)
        short = kwargs.get("short", False)

        if short:

            def opts(*args):
                return args

        else:

            def opts(*args):
                return [arg for arg in args if arg.startswith("--")]

        # We limit the import to occur here
        import argparse

        # The first thing we do is adding the geometry to the NameSpace of the
        # parser.
        # This will enable custom actions to interact with the geometry in a
        # straight forward manner.
        if isinstance(self, Geometry):
            g = self.copy()
        else:
            g = None
        namespace = default_namespace(
            _geometry=g,
            _stored_geometry=False,
        )

        # Create actions
        class Format(argparse.Action):
            def __call__(self, parser, ns, value, option_string=None):
                ns._geom_fmt = value[0]

        p.add_argument(
            *opts("--format"),
            action=Format,
            nargs=1,
            default=".8f",
            help="Specify output format for coordinates.",
        )

        class MoveOrigin(argparse.Action):
            def __call__(self, parser, ns, no_value, option_string=None):
                ns._geometry.xyz[:, :] -= np.amin(ns._geometry.xyz, axis=0)[None, :]

        p.add_argument(
            *opts("--origin", "-O"),
            action=MoveOrigin,
            nargs=0,
            help="Move all atoms such that the smallest value along each Cartesian direction will be at the origin.",
        )

        class MoveCenterOf(argparse.Action):
            def __call__(self, parser, ns, value, option_string=None):
                xyz = ns._geometry.center(what="xyz")
                ns._geometry = ns._geometry.translate(
                    ns._geometry.center(what=value) - xyz
                )

        p.add_argument(
            *opts("--center-of", "-co"),
            choices=["mass", "mass:pbc", "xyz", "position", "cell", "mm:xyz"],
            action=MoveCenterOf,
            help="Move coordinates to the center of the designated choice.",
        )

        class MoveUnitCell(argparse.Action):
            def __call__(self, parser, ns, value, option_string=None):
                if value in ("translate", "tr", "t"):
                    # Simple translation
                    tmp = np.amin(ns._geometry.xyz, axis=0)
                    ns._geometry = ns._geometry.translate(-tmp)
                elif value == "mod":
                    g = ns._geometry
                    # Change all coordinates using the reciprocal cell and move to unit-cell (% 1.)
                    fxyz = g.fxyz % 1.0
                    ns._geometry.xyz[:, :] = dot(fxyz, g.cell)

        p.add_argument(
            *opts("--unit-cell", "-uc"),
            choices=["translate", "tr", "t", "mod"],
            action=MoveUnitCell,
            help="Moves the coordinates into the unit-cell by translation or the mod-operator",
        )

        # Rotation
        class Rotation(argparse.Action):
            def __call__(self, parser, ns, values, option_string=None):
                # Convert value[0] to the direction
                # The rotate function expects degree
                ang = angle(values[0], rad=False, in_rad=False)
                ns._geometry = ns._geometry.rotate(ang, values[1], what="abc+xyz")

        p.add_argument(
            *opts("--rotate", "-R"),
            nargs=2,
            metavar=("ANGLE", "DIR"),
            action=Rotation,
            help='Rotate coordinates and lattice vectors around given axis (x|y|z|a|b|c). ANGLE defaults to be specified in degree. Prefix with "r" for input in radians.',
        )

        if not limit_args:

            class RotationX(argparse.Action):
                def __call__(self, parser, ns, value, option_string=None):
                    # The rotate function expects degree
                    ang = angle(value, rad=False, in_rad=False)
                    ns._geometry = ns._geometry.rotate(ang, "x", what="abc+xyz")

            p.add_argument(
                *opts("--rotate-x", "-Rx"),
                metavar="ANGLE",
                action=RotationX,
                help='Rotate coordinates and lattice vectors around x axis. ANGLE defaults to be specified in degree. Prefix with "r" for input in radians.',
            )

            class RotationY(argparse.Action):
                def __call__(self, parser, ns, value, option_string=None):
                    # The rotate function expects degree
                    ang = angle(value, rad=False, in_rad=False)
                    ns._geometry = ns._geometry.rotate(ang, "y", what="abc+xyz")

            p.add_argument(
                *opts("--rotate-y", "-Ry"),
                metavar="ANGLE",
                action=RotationY,
                help='Rotate coordinates and lattice vectors around y axis. ANGLE defaults to be specified in degree. Prefix with "r" for input in radians.',
            )

            class RotationZ(argparse.Action):
                def __call__(self, parser, ns, value, option_string=None):
                    # The rotate function expects degree
                    ang = angle(value, rad=False, in_rad=False)
                    ns._geometry = ns._geometry.rotate(ang, "z", what="abc+xyz")

            p.add_argument(
                *opts("--rotate-z", "-Rz"),
                metavar="ANGLE",
                action=RotationZ,
                help='Rotate coordinates and lattice vectors around z axis. ANGLE defaults to be specified in degree. Prefix with "r" for input in radians.',
            )

        # Reduce size of geometry
        class ReduceSub(argparse.Action):
            def __call__(self, parser, ns, value, option_string=None):
                # Get atomic indices
                rng = lstranges(strmap(int, value))
                ns._geometry = ns._geometry.sub(rng)

        p.add_argument(
            "--sub",
            metavar="RNG",
            action=ReduceSub,
            help="Retains specified atoms, can be complex ranges.",
        )

        class ReduceRemove(argparse.Action):
            def __call__(self, parser, ns, value, option_string=None):
                # Get atomic indices
                rng = lstranges(strmap(int, value))
                ns._geometry = ns._geometry.remove(rng)

        p.add_argument(
            "--remove",
            metavar="RNG",
            action=ReduceRemove,
            help="Removes specified atoms, can be complex ranges.",
        )

        # Swaps atoms
        class AtomSwap(argparse.Action):
            def __call__(self, parser, ns, value, option_string=None):
                # Get atomic indices
                a = lstranges(strmap(int, value[0]))
                b = lstranges(strmap(int, value[1]))
                if len(a) != len(b):
                    raise ValueError(
                        "swapping atoms requires equal number of LHS and RHS atomic ranges"
                    )
                ns._geometry = ns._geometry.swap(a, b)

        p.add_argument(
            *opts("--swap"),
            metavar=("A", "B"),
            nargs=2,
            action=AtomSwap,
            help="Swaps groups of atoms (can be complex ranges). The groups must be of equal length.",
        )

        # Add an atom
        class AtomAdd(argparse.Action):
            def __call__(self, parser, ns, values, option_string=None):
                # Create an atom from the input
                g = Geometry(
                    [float(x) for x in values[0].split(",")], atoms=Atom(values[1])
                )
                ns._geometry = ns._geometry.add(g)

        p.add_argument(
            *opts("--add"),
            nargs=2,
            metavar=("COORD", "Z"),
            action=AtomAdd,
            help="Adds an atom, coordinate is comma separated (in Ang). Z is the atomic number.",
        )

        class Translate(argparse.Action):
            def __call__(self, parser, ns, values, option_string=None):
                # Create an atom from the input
                if "," in values[0]:
                    xyz = [float(x) for x in values[0].split(",")]
                else:
                    xyz = [float(x) for x in values[0].split()]
                ns._geometry = ns._geometry.translate(xyz)

        p.add_argument(
            *opts("--translate", "-t"),
            nargs=1,
            metavar="COORD",
            action=Translate,
            help="Translates the coordinates via a comma separated list (in Ang).",
        )

        # Periodicly increase the structure
        class PeriodRepeat(argparse.Action):
            def __call__(self, parser, ns, values, option_string=None):
                r = int(values[0])
                d = direction(values[1])
                ns._geometry = ns._geometry.repeat(r, d)

        p.add_argument(
            *opts("--repeat", "-r"),
            nargs=2,
            metavar=("TIMES", "DIR"),
            action=PeriodRepeat,
            help="Repeats the geometry in the specified direction.",
        )

        if not limit_args:

            class PeriodRepeatX(argparse.Action):
                def __call__(self, parser, ns, value, option_string=None):
                    ns._geometry = ns._geometry.repeat(int(value), 0)

            p.add_argument(
                *opts("--repeat-x", "-rx"),
                metavar="TIMES",
                action=PeriodRepeatX,
                help="Repeats the geometry along the first cell vector.",
            )

            class PeriodRepeatY(argparse.Action):
                def __call__(self, parser, ns, value, option_string=None):
                    ns._geometry = ns._geometry.repeat(int(value), 1)

            p.add_argument(
                *opts("--repeat-y", "-ry"),
                metavar="TIMES",
                action=PeriodRepeatY,
                help="Repeats the geometry along the second cell vector.",
            )

            class PeriodRepeatZ(argparse.Action):
                def __call__(self, parser, ns, value, option_string=None):
                    ns._geometry = ns._geometry.repeat(int(value), 2)

            p.add_argument(
                *opts("--repeat-z", "-rz"),
                metavar="TIMES",
                action=PeriodRepeatZ,
                help="Repeats the geometry along the third cell vector.",
            )

        class ReduceUnrepeat(argparse.Action):
            def __call__(self, parser, ns, values, option_string=None):
                s = int(values[0])
                d = direction(values[1])
                ns._geometry = ns._geometry.unrepeat(s, d)

        p.add_argument(
            *opts("--unrepeat", "-ur"),
            nargs=2,
            metavar=("REPS", "DIR"),
            action=ReduceUnrepeat,
            help="Unrepeats the geometry into `reps` parts along the unit-cell direction `dir` (opposite of --repeat).",
        )

        class PeriodTile(argparse.Action):
            def __call__(self, parser, ns, values, option_string=None):
                r = int(values[0])
                d = direction(values[1])
                ns._geometry = ns._geometry.tile(r, d)

        p.add_argument(
            *opts("--tile"),
            nargs=2,
            metavar=("TIMES", "DIR"),
            action=PeriodTile,
            help="Tiles the geometry in the specified direction.",
        )

        if not limit_args:

            class PeriodTileX(argparse.Action):
                def __call__(self, parser, ns, value, option_string=None):
                    ns._geometry = ns._geometry.tile(int(value), 0)

            p.add_argument(
                *opts("--tile-x", "-tx"),
                metavar="TIMES",
                action=PeriodTileX,
                help="Tiles the geometry along the first cell vector.",
            )

            class PeriodTileY(argparse.Action):
                def __call__(self, parser, ns, value, option_string=None):
                    ns._geometry = ns._geometry.tile(int(value), 1)

            p.add_argument(
                *opts("--tile-y", "-ty"),
                metavar="TIMES",
                action=PeriodTileY,
                help="Tiles the geometry along the second cell vector.",
            )

            class PeriodTileZ(argparse.Action):
                def __call__(self, parser, ns, value, option_string=None):
                    ns._geometry = ns._geometry.tile(int(value), 2)

            p.add_argument(
                *opts("--tile-z", "-tz"),
                metavar="TIMES",
                action=PeriodTileZ,
                help="Tiles the geometry along the third cell vector.",
            )

        class ReduceUntile(argparse.Action):
            def __call__(self, parser, ns, values, option_string=None):
                s = int(values[0])
                d = direction(values[1])
                ns._geometry = ns._geometry.untile(s, d)

        p.add_argument(
            *opts("--untile", "--cut", "-ut"),
            nargs=2,
            metavar=("REPS", "DIR"),
            action=ReduceUntile,
            help="Untiles the geometry into `reps` parts along the unit-cell direction `dir` (opposite of --tile).",
        )

        # append another geometry
        class Geometryend(argparse.Action):
            def __call__(self, parser, ns, values, option_string=None):
                # Create an atom from the input
                f = Path(values[0])
                geom = Geometry.read(values[0])
                d = direction(values[1])
                ns._geometry = getattr(ns._geometry, self._method_pend)(geom, d)

        class GeometryAppend(Geometryend):
            _method_pend = "append"

        p.add_argument(
            *opts("--append"),
            nargs=2,
            metavar=("GEOM", "DIR"),
            action=GeometryAppend,
            help="Appends another Geometry along direction DIR.",
        )

        class GeometryPrepend(Geometryend):
            _method_pend = "prepend"

        p.add_argument(
            *opts("--prepend"),
            nargs=2,
            metavar=("GEOM", "DIR"),
            action=GeometryPrepend,
            help="Prepends another Geometry along direction DIR.",
        )

        # Sort
        class Sort(argparse.Action):
            def __call__(self, parser, ns, values, option_string=None):
                # call geometry.sort(...) using appropriate keywords (and ordered dict)
                kwargs = OrderedDict()
                opts = values[0].split(";")
                for i, opt in enumerate(opts):
                    # Split for equal
                    opt = opt.split("=", 1)
                    if len(opt) > 1:
                        opt, val = opt
                    else:
                        opt = opt[0]
                        val = "True"
                    if val.lower() in ("t", "true"):
                        val = True
                    elif val.lower() in ("f", "false"):
                        val = False
                    elif opt == "atol":
                        # float values
                        val = float(val)
                    elif opt == "group":
                        pass
                    else:
                        # it must be a range/tuple
                        val = lstranges(strmap(int, val))

                    # we always add integers to allow users to use the same keywords on commandline
                    kwargs[opt.strip() + str(i)] = val
                ns._geometry = ns._geometry.sort(**kwargs)

        p.add_argument(
            *opts("--sort"),
            nargs=1,
            metavar="SORT",
            action=Sort,
            help='Semi-colon separated options for sort, please always encapsulate in quotation ["axes=0;descend;lattice=(1, 2);group=Z"].',
        )

        # Print some common information about the
        # geometry (to stdout)
        class PrintInfo(argparse.Action):
            def __call__(self, parser, ns, no_value, option_string=None):
                # We fake that it has been stored...
                ns._stored_geometry = True
                print(ns._geometry)

        p.add_argument(
            *opts("--info"),
            nargs=0,
            action=PrintInfo,
            help="Print, to stdout, some regular information about the geometry.",
        )

        class Out(argparse.Action):
            def __call__(self, parser, ns, value, option_string=None):
                if value is None:
                    return
                if len(value) == 0:
                    return
                # If the vector, exists, we should write it
                kwargs = {}
                if hasattr(ns, "_geom_fmt"):
                    kwargs["fmt"] = ns._geom_fmt
                if hasattr(ns, "_vector"):
                    v = getattr(ns, "_vector")
                    vs = getattr(ns, "_vector_scale")
                    if isinstance(vs, bool):
                        if vs:
                            vs = 1.0 / np.max(sqrt(square(v).sum(1)))
                            info(f"Scaling vector by: {vs}")
                        else:
                            vs = 1.0

                    # Store the vectors with the scaling
                    kwargs["data"] = v * vs
                ns._geometry.write(value[0], **kwargs)
                # Issue to the namespace that the geometry has been written, at least once.
                ns._stored_geometry = True

        p.add_argument(
            *opts("--out", "-o"),
            nargs=1,
            action=Out,
            help="Store the geometry (at its current invocation) to the out file.",
        )

        # If the user requests positional out arguments, we also add that.
        if kwargs.get("positional_out", False):
            p.add_argument(
                "out",
                nargs="*",
                default=None,
                action=Out,
                help="Store the geometry (at its current invocation) to the out file.",
            )

        # We have now created all arguments
        return p, namespace


new_dispatch = Geometry.new
to_dispatch = Geometry.to


# Define base-class for this
class GeometryNewDispatch(AbstractDispatch):
    """Base dispatcher from class passing arguments to Geometry class"""


# Bypass regular Geometry to be returned as is
class GeometryNewGeometryDispatch(GeometryNewDispatch):
    def dispatch(self, geometry, copy: bool = False) -> Geometry:
        """Return Geometry, for sanitization purposes"""
        cls = self._get_class()
        if cls != geometry.__class__:
            geometry = cls(
                geometry.xyz.copy(),
                atoms=geometry.atoms.copy(),
                lattice=geometry.lattice.copy(),
            )
            copy = False
        if copy:
            return geometry.copy()
        return geometry


new_dispatch.register(Geometry, GeometryNewGeometryDispatch)


class GeometryNewFileDispatch(GeometryNewDispatch):
    def dispatch(self, *args, **kwargs) -> Geometry:
        """Defer the `Geometry.read` method by passing down arguments"""
        cls = self._get_class()
        return cls.read(*args, **kwargs)


new_dispatch.register(str, GeometryNewFileDispatch)
new_dispatch.register(Path, GeometryNewFileDispatch)
# see sisl/__init__.py for new_dispatch.register(BaseSile, GeometryNewFileDispatcher)


class GeometryNewAseDispatch(GeometryNewDispatch):
    def dispatch(self, aseg, **kwargs) -> Geometry:
        """Convert an `ase.Atoms` object into a `Geometry`"""
        cls = self._get_class()
        Z = aseg.get_atomic_numbers()
        xyz = aseg.get_positions()
        cell = aseg.get_cell()
        nsc = [3 if pbc else 1 for pbc in aseg.pbc]
        lattice = Lattice(cell, nsc=nsc)
        return cls(xyz, atoms=Z, lattice=lattice, **kwargs)


new_dispatch.register("ase", GeometryNewAseDispatch)

# currently we can't ensure the ase Atoms type
# to get it by type(). That requires ase to be importable.
try:
    from ase import Atoms as ase_Atoms

    new_dispatch.register(ase_Atoms, GeometryNewAseDispatch)
    # ensure we don't pollute name-space
    del ase_Atoms
except Exception:
    pass


class GeometryNewpymatgenDispatch(GeometryNewDispatch):
    def dispatch(self, struct, **kwargs) -> Geometry:
        """Convert a ``pymatgen`` structure/molecule object into a `Geometry`"""
        from pymatgen.core import Structure

        cls = self._get_class(allow_instance=True)

        Z = []
        xyz = []
        for site in struct.sites:
            Z.append(site.specie.Z)
            xyz.append(site.coords)
        xyz = np.array(xyz)

        if isinstance(struct, Structure):
            # we also have the lattice
            cell = struct.lattice.matrix
            nsc = [3, 3, 3]  # really, this is unknown
        else:
            cell = xyz.max() - xyz.min(0) + 15.0
            nsc = [1, 1, 1]
        lattice = Lattice(cell, nsc=nsc)
        return cls(xyz, atoms=Z, lattice=lattice, **kwargs)


new_dispatch.register("pymatgen", GeometryNewpymatgenDispatch)

# currently we can't ensure the pymatgen classes
# to get it by type(). That requires pymatgen to be importable.
try:
    from pymatgen.core import Molecule as pymatgen_Molecule
    from pymatgen.core import Structure as pymatgen_Structure

    new_dispatch.register(pymatgen_Molecule, GeometryNewpymatgenDispatch)
    new_dispatch.register(pymatgen_Structure, GeometryNewpymatgenDispatch)
    # ensure we don't pollute name-space
    del pymatgen_Molecule, pymatgen_Structure
except Exception:
    pass


class GeometryToDispatch(AbstractDispatch):
    """Base dispatcher from class passing from Geometry class"""


class GeometryToSileDispatch(GeometryToDispatch):
    def dispatch(self, *args, **kwargs) -> None:
        """Writes the geometry to a sile with any optional arguments.

        Examples
        --------

        >>> geom = si.geom.graphene()
        >>> geom.to("hello.xyz")
        >>> geom.to(pathlib.Path("hello.xyz"))
        """
        geom = self._get_object()
        return geom.write(*args, **kwargs)


to_dispatch.register("str", GeometryToSileDispatch)
to_dispatch.register("Path", GeometryToSileDispatch)
# to do geom.to[Path](path)
to_dispatch.register(str, GeometryToSileDispatch)
to_dispatch.register(Path, GeometryToSileDispatch)


class GeometryToAseDispatch(GeometryToDispatch):
    def dispatch(self, **kwargs) -> ase.Atoms:
        """Conversion of `Geometry` to an `ase.Atoms` object"""
        from ase import Atoms as ase_Atoms

        geom = self._get_object()
        return ase_Atoms(
            symbols=geom.atoms.Z,
            positions=geom.xyz.tolist(),
            cell=geom.cell.tolist(),
            pbc=geom.pbc,
            **kwargs,
        )


to_dispatch.register("ase", GeometryToAseDispatch)
if has_module("ase"):
    from ase import Atoms as ase_Atoms

    to_dispatch.register(ase_Atoms, GeometryToAseDispatch)
    del ase_Atoms


class GeometryTopymatgenDispatch(GeometryToDispatch):
    def dispatch(
        self, **kwargs
    ) -> Union[pymatgen.core.Molecule, pymatgen.core.Structure]:
        """Conversion of `Geometry` to a `pymatgen` object.

        Depending on the periodicity, it can be `Molecule` or `Structure`.
        """
        from pymatgen.core import Lattice, Molecule, Structure

        from sisl._core import PeriodicTable

        # ensure we have an object
        geom = self._get_object()

        lattice = Lattice(geom.cell)
        # get atomic letters and coordinates
        PT = PeriodicTable()
        xyz = geom.xyz
        species = [PT.Z_label(Z) for Z in geom.atoms.Z]

        if np.any(self.pbc):
            return Structure(lattice, species, xyz, coords_are_cartesian=True, **kwargs)
        # we define a molecule
        return Molecule(species, xyz, **kwargs)


to_dispatch.register("pymatgen", GeometryTopymatgenDispatch)


class GeometryToDataframeDispatch(GeometryToDispatch):
    def dispatch(self, *args, **kwargs) -> pandas.DataFrame:
        """Convert the geometry to a `pandas.DataFrame` with values stored in columns"""

        import pandas as pd

        geom = self._get_object()

        # Now create data-frame
        # Currently we will populate it with
        # - xyz
        # - symbol
        # - Z
        # - tag
        # - R
        # - mass
        # - valence q
        # - norbs
        data = {}
        x, y, z = geom.xyz.T
        data["x"] = x
        data["y"] = y
        data["z"] = z

        atoms = geom.atoms
        data["Z"] = atoms.Z
        data["mass"] = atoms.mass
        data["R"] = atoms.maxR(all=True)
        data["q0"] = atoms.q0
        data["norbitals"] = atoms.orbitals

        return pd.DataFrame(data)


to_dispatch.register("dataframe", GeometryToDataframeDispatch)
if has_module("pandas"):
    from pandas import DataFrame as pd_DataFrame

    to_dispatch.register(pd_DataFrame, GeometryToDataframeDispatch)
    del pd_DataFrame


# Clean up
del new_dispatch, to_dispatch


@set_module("sisl")
def sgeom(geometry=None, argv=None, ret_geometry=False):
    """Main script for sgeom.

    This routine may be called with `argv` and/or a `Sile` which is the geometry at hand.

    Parameters
    ----------
    geom : Geometry or BaseSile
       this may either be the geometry, as-is, or a `Sile` which contains
       the geometry.
    argv : list of str
       the arguments passed to sgeom
    ret_geometry : bool, optional
       whether the function should return the geometry
    """
    import argparse
    import sys
    from pathlib import Path

    from sisl.io import BaseSile, get_sile

    # The geometry-file *MUST* be the first argument
    # (except --help|-h)
    exe = Path(sys.argv[0]).name

    # We cannot create a separate ArgumentParser to retrieve a positional arguments
    # as that will grab the first argument for an option!

    # Start creating the command-line utilities that are the actual ones.
    description = f"""
This manipulation utility is highly advanced and one should note that the ORDER of
options is determining the final structure. For instance:

   {exe} geom.xyz --repeat 2 x --repeat 2 y

is NOT equivalent to:

   {exe} geom.xyz --repeat 2 y --repeat 2 x

This may be unexpected but enables one to do advanced manipulations.

Additionally, in between arguments, one may store the current state of the geometry
by writing to a standard file.

   {exe} geom.xyz --repeat 2 y geom_repy.xyz --repeat 2 x geom_repy_repx.xyz

will create two files:
   geom_repy.xyz
will only be repeated 2 times along the second lattice vector, while:
   geom_repy_repx.xyz
will be repeated 2 times along the second lattice vector, and then the first
lattice vector.
    """

    if argv is not None:
        if len(argv) == 0:
            argv = ["--help"]
    elif len(sys.argv) == 1:
        # no arguments
        # fake a help
        argv = ["--help"]
    else:
        argv = sys.argv[1:]

    # Ensure that the arguments have pre-pended spaces
    argv = cmd.argv_negative_fix(argv)

    p = argparse.ArgumentParser(
        exe,
        formatter_class=SislHelpFormatter,
        description=description,
    )

    # Add default sisl version stuff
    cmd.add_sisl_version_cite_arg(p)

    # First read the input "Sile"
    stdout_geom = True
    if geometry is None:
        from os.path import isfile

        argv, input_file = cmd.collect_input(argv)

        if input_file is None:
            stdout_geom = False
            geometry = Geometry([0] * 3)
        else:
            # Extract specification of the input file
            i_file, spec = str_spec(input_file)

            if isfile(i_file):
                geometry = get_sile(input_file).read_geometry()
            else:
                info(f"Cannot find file '{input_file}'!")
                geometry = Geometry
                stdout_geom = False

    elif isinstance(geometry, Geometry):
        # Do nothing, the geometry is already created
        pass

    elif isinstance(geometry, BaseSile):
        geometry = geometry.read_geometry()
        # Store the input file...
        input_file = geometry.file

    # Do the argument parser
    p, ns = geometry.ArgumentParser(p, **geometry._ArgumentParser_args_single())

    # Now the arguments should have been populated
    # and we will sort out if the input options
    # is only a help option.
    try:
        if not hasattr(ns, "_input_file"):
            setattr(ns, "_input_file", input_file)
    except Exception:
        pass

    # Now try and figure out the actual arguments
    p, ns, argv = cmd.collect_arguments(
        argv, input=False, argumentparser=p, namespace=ns
    )

    # We are good to go!!!
    args = p.parse_args(argv, namespace=ns)
    g = args._geometry

    if stdout_geom and not args._stored_geometry:
        # We should write out the information to the stdout
        # This is merely for testing purposes and may not be used for anything.
        print("Cell:")
        for i in (0, 1, 2):
            print("  {:10.6f} {:10.6f} {:10.6f}".format(*g.cell[i, :]))
        print("Lattice:")
        print("  {:d} {:d} {:d}".format(*g.nsc))
        print(" {:>10s} {:>10s} {:>10s}  {:>3s}".format("x", "y", "z", "Z"))
        for ia in g:
            print(
                " {1:10.6f} {2:10.6f} {3:10.6f}  {0:3d}".format(
                    g.atoms[ia].Z, *g.xyz[ia, :]
                )
            )

    if ret_geometry:
        return g
    return 0
