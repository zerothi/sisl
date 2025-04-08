# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""Define a lattice with cell-parameters and supercells

This class is the basis of many different objects.
"""
from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from enum import IntEnum, auto
from numbers import Integral
from pathlib import Path
from typing import Optional, Union

import numpy as np
import numpy.typing as npt
from numpy import dot, ndarray

import sisl._array as _a
from sisl._dispatch_class import _Dispatchs
from sisl._dispatcher import AbstractDispatch, ClassDispatcher, TypeDispatcher
from sisl._internal import set_module
from sisl._math_small import cross3, dot3
from sisl.messages import SislError, deprecate, deprecate_argument, deprecation, warn
from sisl.shape.prism4 import Cuboid
from sisl.typing import CellAxes, CellAxis, LatticeLike
from sisl.utils.mathematics import fnorm
from sisl.utils.misc import direction, listify

from ._lattice import cell_invert, cell_reciprocal

__all__ = ["Lattice", "SuperCell", "LatticeChild", "BoundaryCondition"]

_log = logging.getLogger(__name__)


class BoundaryCondition(IntEnum):
    """Enum for boundary conditions"""

    UNKNOWN = auto()
    PERIODIC = auto()
    DIRICHLET = auto()
    NEUMANN = auto()
    OPEN = auto()

    @classmethod
    def getitem(cls, key):
        """Search for a specific integer entry by value, and not by name"""
        if isinstance(key, cls):
            return key
        if isinstance(key, bool):
            if key:
                return cls.PERIODIC
            raise ValueError(
                f"{cls.__name__}.getitem does not allow False, which BC should this refer to?"
            )
        if isinstance(key, str):
            key = key.upper()
            if len(key) == 1:
                key = {
                    "U": "UNKNOWN",
                    "P": "PERIODIC",
                    "D": "DIRICHLET",
                    "N": "NEUMANN",
                    "O": "OPEN",
                }[key]
            for bc in cls:
                if bc.name.startswith(key):
                    return bc
        else:
            for bc in cls:
                if bc == key:
                    return bc
        raise KeyError(f"{cls.__name__}.getitem could not find key={key}")


BoundaryConditionType = Union[BoundaryCondition, int, str, bool]
SeqBoundaryConditionType = Union[BoundaryConditionType, Sequence[BoundaryConditionType]]


@set_module("sisl")
class Lattice(
    _Dispatchs,
    dispatchs=[
        ClassDispatcher("new", instance_dispatcher=TypeDispatcher),
        ClassDispatcher("to", type_dispatcher=None),
    ],
    when_subclassing="copy",
):
    r"""A cell class to retain lattice vectors and a supercell structure

    The supercell structure is comprising the *primary* unit-cell and neighboring
    unit-cells. The number of supercells is given by the attribute `nsc` which
    is a vector with 3 elements, one per lattice vector. It describes *how many*
    times the primary unit-cell is extended along the i'th lattice vector.
    For ``nsc[i] == 3`` the supercell is made up of 3 unit-cells. One *behind*, the
    primary unit-cell and one *after*.

    Parameters
    ----------
    cell :
       the lattice parameters of the unit cell (the actual cell
       is returned from `tocell`.
    nsc :
       number of supercells along each lattice vector
    origin : (3,) of float, optional
       the origin of the supercell.
    boundary_condition :
        the boundary conditions for each of the cell's planes. Defaults to periodic boundary condition.
        See `BoundaryCondition` for valid enumerations.
    """

    # We limit the scope of this Lattice object.
    __slots__ = ("cell", "_origin", "nsc", "n_s", "_sc_off", "_isc_off", "_bc")

    #: Internal reference to `BoundaryCondition` for simpler short-hands
    BC = BoundaryCondition

    def __init__(
        self,
        cell: CellLike,
        nsc: npt.ArrayLike = None,
        origin=None,
        boundary_condition: SeqBoundaryConditionType = BoundaryCondition.PERIODIC,
    ):
        if nsc is None:
            nsc = [1, 1, 1]

        # If the length of cell is 6 it must be cell-parameters, not
        # actual cell coordinates
        self.cell = self.tocell(cell)
        if np.any(self.length < 1e-7):
            warn(
                f"{self.__class__.__name__} got initialized with one or more "
                "lattice vector(s) with 0 length. Use with care."
            )

        if origin is None:
            self._origin = _a.zerosd(3)
        else:
            self._origin = _a.arrayd(origin)
            if self._origin.size != 3:
                raise ValueError("Origin *must* be 3 numbers.")

        self.nsc = _a.onesi(3)
        # Set the super-cell
        self.set_nsc(nsc=nsc)
        self.set_boundary_condition(boundary_condition)

    @property
    def length(self) -> ndarray:
        """Length of each lattice vector"""
        return fnorm(self.cell)

    def lengthf(self, axes: CellAxes = (0, 1, 2)) -> ndarray:
        """Length of specific lattice vectors (as a function)
        Parameters
        ----------
        axes:
            only calculate the volume based on a subset of axes

        Examples
        --------
        Only get lengths of two lattice vectors:

        >>> lat = Lattice(1)
        >>> lat.lengthf([0, 1])
        """
        axes = map(direction, listify(axes)) | listify
        return fnorm(self.cell[axes])

    @property
    def volume(self) -> float:
        """Volume of cell"""
        return self.volumef((0, 1, 2))

    def volumef(self, axes: CellAxes = (0, 1, 2)) -> float:
        """Volume of cell (as a function)

        Default to the 3D volume.
        For `axes` with only 2 elements, it corresponds to an area.
        For `axes` with only 1 element, it corresponds to a length.

        Parameters
        ----------
        axes:
            only calculate the volume based on a subset of axes

        Examples
        --------
        Only get the volume of the periodic directions:

        >>> lat = Lattice(1)
        >>> lat.pbc = (True, False, True)
        >>> lat.volumef(lat.pbc.nonzero()[0])
        """
        axes = map(direction, listify(axes)) | listify

        cell = self.cell
        if len(axes) == 3:
            return abs(dot3(cell[axes[0]], cross3(cell[axes[1]], cell[axes[2]])))
        if len(axes) == 2:
            return fnorm(cross3(cell[axes[0]], cell[axes[1]]))
        if len(axes) == 1:
            return fnorm(cell[axes])
        return 0.0

    def area(self, axis1: CellAxis, axis2: CellAxis) -> float:
        """Calculate the area spanned by the two axis `ax0` and `ax1`"""
        axis1 = direction(axis1)
        axis2 = direction(axis2)
        return fnorm(cross3(self.cell[axis1], self.cell[axis2]))

    @property
    def boundary_condition(self) -> np.ndarray:
        """Boundary conditions for each lattice vector (lower/upper) sides ``(3, 2)``"""
        return self._bc

    @boundary_condition.setter
    def boundary_condition(self, boundary_condition):
        """Boundary conditions for each lattice vector (lower/upper) sides ``(3, 2)``"""
        self.set_boundary_condition(boundary_condition)

    @property
    def pbc(self) -> np.ndarray:
        """Boolean array to specify whether the boundary conditions are periodic`"""
        # set_boundary_condition does not allow to have PERIODIC and non-PERIODIC
        # along the same lattice vector. So checking one should suffice
        return self._bc[:, 0] == BoundaryCondition.PERIODIC

    @pbc.setter
    def pbc(self, pbc) -> None:
        """Boolean array to specify whether the boundary conditions are periodic`"""
        # set_boundary_condition does not allow to have PERIODIC and non-PERIODIC
        # along the same lattice vector. So checking one should suffice
        assert len(pbc) == 3

        PERIODIC = BoundaryCondition.PERIODIC
        for axis, bc in enumerate(pbc):

            # Simply skip those that are not T|F
            if not isinstance(bc, bool):
                continue

            if bc:
                self._bc[axis] = PERIODIC
            elif self._bc[axis, 0] == PERIODIC:
                self._bc[axis] = BoundaryCondition.UNKNOWN

    @property
    def origin(self) -> ndarray:
        """Origin for the cell"""
        return self._origin

    @origin.setter
    def origin(self, origin):
        """Set origin for the cell"""
        self._origin[:] = origin

    @deprecation(
        "toCuboid is deprecated, please use lattice.to['cuboid'](...) instead.",
        "0.15",
        "0.17",
    )
    def toCuboid(self, *args, **kwargs):
        """A cuboid with vectors as this unit-cell and center with respect to its origin

        Parameters
        ----------
        orthogonal : bool, optional
           if true the cuboid has orthogonal sides such that the entire cell is contained
        """
        return self.to[Cuboid](*args, **kwargs)

    def set_boundary_condition(
        self,
        boundary: Optional[SeqBoundaryConditionType] = None,
        a: Optional[SeqBoundaryConditionType] = None,
        b: Optional[SeqBoundaryConditionType] = None,
        c: Optional[SeqBoundaryConditionType] = None,
    ):
        """Set the boundary conditions on the grid

        Parameters
        ----------
        boundary : (3, 2) or (3, ) or int, optional
           boundary condition for all boundaries (or the same for all)
        a : int or list of int, optional
           boundary condition for the first unit-cell vector direction
        b : int or list of int, optional
           boundary condition for the second unit-cell vector direction
        c : int or list of int, optional
           boundary condition for the third unit-cell vector direction

        Raises
        ------
        ValueError
            if specifying periodic one one boundary, so must the opposite side.
        """
        getitem = BoundaryCondition.getitem

        def conv(v):
            if v is None:
                return v
            if isinstance(v, (np.ndarray, list, tuple)):
                return list(map(getitem, v))
            return getitem(v)

        if not hasattr(self, "_bc"):
            self._bc = _a.fulli([3, 2], getitem("Unknown"))
        old = self._bc.copy()

        if not boundary is None:
            if isinstance(boundary, (Integral, str, bool)):
                try:
                    getitem(boundary)
                    self._bc[:, :] = conv(boundary)
                except KeyError:
                    for d, bc in enumerate(boundary):
                        bc = conv(bc)
                        if bc is not None:
                            self._bc[d] = conv(bc)

            else:
                for d, bc in enumerate(boundary):
                    bc = conv(bc)
                    if bc is not None:
                        self._bc[d] = bc

        for d, v in enumerate([a, b, c]):
            v = conv(v)
            if v is not None:
                self._bc[d] = v

        # shorthand for bc
        for nsc, bc, changed in zip(
            self.nsc, self._bc == BoundaryCondition.PERIODIC, self._bc != old
        ):
            if bc.any() and not bc.all():
                raise ValueError(
                    f"{self.__class__.__name__}.set_boundary_condition has a one non-periodic and "
                    "one periodic direction. If one direction is periodic, both instances "
                    "must have that BC."
                )
            if changed.any() and (~bc).all() and nsc > 1:
                warn(
                    f"{self.__class__.__name__}.set_boundary_condition is having image connections (nsc={nsc}>1) "
                    "while having a non-periodic boundary condition."
                )

    def parameters(
        self, rad: bool = False
    ) -> tuple[float, float, float, float, float, float]:
        r"""Cell parameters of this cell in 3 lengths and 3 angles

        Notes
        -----
        Since we return the length and angles between vectors it may not be possible to
        recreate the same cell. Only in the case where the first lattice vector *only*
        has a Cartesian :math:`x` component will this be the case.

        Parameters
        ----------
        rad : bool, optional
           whether the angles are returned in radians (otherwise in degree)

        Returns
        -------
        length : numpy.ndarray
            length of each lattice vector
        angles : numpy.ndarray
            angles between the lattice vectors (in Voigt notation)
            ``[0]`` is between 2nd and 3rd lattice vector, etc.
        """
        if rad:
            f = 1.0
        else:
            f = 180 / np.pi

        # Calculate length of each lattice vector
        cell = self.cell.copy()
        abc = fnorm(cell)

        cell = cell / abc.reshape(-1, 1)
        angles = np.empty(3)
        angles[0] = math.acos(dot3(cell[1], cell[2])) * f
        angles[1] = math.acos(dot3(cell[0], cell[2])) * f
        angles[2] = math.acos(dot3(cell[0], cell[1])) * f

        return abc, angles

    def _fill(self, non_filled, dtype=None):
        """Return a zero filled array of length 3"""

        if len(non_filled) == 3:
            return non_filled

        # Fill in zeros
        # This will purposefully raise an exception
        # if the dimensions of the periodic one
        # are not consistent.
        if dtype is None:
            try:
                dtype = non_filled.dtype
            except Exception:
                dtype = np.dtype(non_filled[0].__class__)
                if dtype == np.dtype(int):
                    # Never go higher than int32 for default
                    # guesses on integer lists.
                    dtype = np.int32
        f = np.zeros(3, dtype)
        i = 0
        if self.nsc[0] > 1:
            f[0] = non_filled[i]
            i += 1
        if self.nsc[1] > 1:
            f[1] = non_filled[i]
            i += 1
        if self.nsc[2] > 1:
            f[2] = non_filled[i]
        return f

    def _fill_sc(self, supercell_index):
        """Return a filled supercell index by filling in zeros where needed"""
        return self._fill(supercell_index, dtype=np.int32)

    def set_nsc(
        self,
        nsc=None,
        a: Optional[int] = None,
        b: Optional[int] = None,
        c: Optional[int] = None,
    ) -> None:
        """Sets the number of supercells in the 3 different cell directions

        Parameters
        ----------
        nsc : list of int, optional
           number of supercells in each direction
        a : int, optional
           number of supercells in the first unit-cell vector direction
        b : int, optional
           number of supercells in the second unit-cell vector direction
        c : int, optional
           number of supercells in the third unit-cell vector direction
        """
        if not nsc is None:
            for i in range(3):
                if not nsc[i] is None:
                    self.nsc[i] = nsc[i]
        if a:
            self.nsc[0] = a
        if b:
            self.nsc[1] = b
        if c:
            self.nsc[2] = c
        # Correct for misplaced number of unit-cells
        for i in range(3):
            if self.nsc[i] == 0:
                self.nsc[i] = 1
        if np.sum(self.nsc % 2) != 3:
            raise ValueError(
                "Supercells has to be of un-even size. The primary cell counts "
                + "one, all others count 2"
            )

        # We might use this very often, hence we store it
        self.n_s = _a.prodi(self.nsc)
        self._sc_off = _a.zerosi([self.n_s, 3])
        self._isc_off = _a.zerosi(self.nsc)

        n = self.nsc
        # We define the following ones like this:

        def ret_range(val):
            i = val // 2
            return range(-i, i + 1)

        x = ret_range(n[0])
        y = ret_range(n[1])
        z = ret_range(n[2])
        i = 0
        for iz in z:
            for iy in y:
                for ix in x:
                    if ix == 0 and iy == 0 and iz == 0:
                        continue
                    # Increment index
                    i += 1
                    # The offsets for the supercells in the
                    # sparsity pattern
                    self._sc_off[i, 0] = ix
                    self._sc_off[i, 1] = iy
                    self._sc_off[i, 2] = iz

        self._update_isc_off()

    def _update_isc_off(self):
        """Internal routine for updating the supercell indices"""
        for i in range(self.n_s):
            d = self.sc_off[i]
            self._isc_off[d[0], d[1], d[2]] = i

    @property
    def sc_off(self) -> ndarray:
        """Integer supercell offsets"""
        return self._sc_off

    @sc_off.setter
    def sc_off(self, sc_off):
        """Set the supercell offset"""
        self._sc_off[:, :] = _a.arrayi(sc_off, order="C")
        self._update_isc_off()

    @property
    def isc_off(self) -> ndarray:
        """Internal indexed supercell ``[ia, ib, ic] == i``"""
        return self._isc_off

    def __iter__(self):
        """Iterate the supercells and the indices of the supercells"""
        yield from enumerate(self.sc_off)

    @deprecate_argument(
        "axis",
        "axes",
        "argument axis has been deprecated in favor of axes, please update your code.",
        "0.15",
        "0.17",
    )
    @deprecate_argument(
        "tol",
        "atol",
        "argument tol has been deprecated in favor of atol, please update your code.",
        "0.15",
        "0.17",
    )
    def fit(self, xyz, axes: CellAxes = (0, 1, 2), atol: float = 0.05) -> Lattice:
        """Fit the supercell to `xyz` such that the unit-cell becomes periodic in the specified directions

        The fitted supercell tries to determine the unit-cell parameters by solving a set of linear equations
        corresponding to the current supercell vectors.

        >>> numpy.linalg.solve(self.cell.T, xyz.T)

        It is important to know that this routine will *only* work if at least some of the atoms are
        integer offsets of the lattice vectors. I.e. the resulting fit will depend on the translation
        of the coordinates.

        Parameters
        ----------
        xyz : array_like ``shape(*, 3)``
           the coordinates that we will wish to encompass and analyze.
        axes :
           only the cell-vectors along the provided axes will be used
        atol :
           tolerance (in Angstrom) of the positions. I.e. we neglect coordinates
           which are not within the radius of this magnitude

        Raises
        ------
        RuntimeError :
            when the cell-parameters does not fit within the given tolerance (`atol`).
        """
        # In case the passed coordinates are from a Geometry
        from .geometry import Geometry

        if isinstance(xyz, Geometry):
            xyz = xyz.xyz

        cell = np.copy(self.cell)

        # Get fractional coordinates to get the divisions in the current cell
        x = dot(xyz, self.icell.T)

        # Now we should figure out the correct repetitions
        # by rounding to integer positions of the cell vectors
        ix = np.rint(x)

        # Figure out the displacements from integers
        # Then reduce search space by removing those coordinates
        # that are more than the tolerance.
        dist = np.sqrt((dot(cell.T, (x - ix).T) ** 2).sum(0))
        idx = (dist <= atol).nonzero()[0]
        if len(idx) == 0:
            raise RuntimeError(
                "Could not fit the cell parameters to the coordinates "
                "due to insufficient accuracy (try to increase the tolerance)"
            )

        # Reduce problem to allowed values below the tolerance
        ix = ix[idx]

        # Reduce to total repetitions
        ireps = np.amax(ix, axis=0) - np.amin(ix, axis=0) + 1

        # Reduce the non-set axis
        if not axes is None:
            axes = map(direction, listify(axes))
            for ax in (0, 1, 2):
                if ax not in axes:
                    ireps[ax] = 1

        # Enlarge the cell vectors
        cell[0] *= ireps[0]
        cell[1] *= ireps[1]
        cell[2] *= ireps[2]

        return self.copy(cell)

    def plane(
        self, axis1: CellAxis, axis2: CellAxis, origin: bool = True
    ) -> tuple[ndarray, ndarray]:
        """Query point and plane-normal for the plane spanning `ax1` and `ax2`

        Parameters
        ----------
        axis1 :
           the first axis vector
        axis2 :
           the second axis vector
        origin : bool, optional
           whether the plane intersects the origin or the opposite corner of the
           unit-cell.

        Returns
        -------
        normal_V : numpy.ndarray
           planes normal vector (pointing outwards with regards to the cell)
        p : numpy.ndarray
           a point on the plane

        Examples
        --------

        All 6 faces of the supercell can be retrieved like this:

        >>> lattice = Lattice(4)
        >>> n1, p1 = lattice.plane(0, 1, True)
        >>> n2, p2 = lattice.plane(0, 1, False)
        >>> n3, p3 = lattice.plane(0, 2, True)
        >>> n4, p4 = lattice.plane(0, 2, False)
        >>> n5, p5 = lattice.plane(1, 2, True)
        >>> n6, p6 = lattice.plane(1, 2, False)

        However, for performance critical calculations it may be advantageous to
        do this:

        >>> lattice = Lattice(4)
        >>> uc = lattice.cell.sum(0)
        >>> n1, p1 = lattice.plane(0, 1)
        >>> n2 = -n1
        >>> p2 = p1 + uc
        >>> n3, p3 = lattice.plane(0, 2)
        >>> n4 = -n3
        >>> p4 = p3 + uc
        >>> n5, p5 = lattice.plane(1, 2)
        >>> n6 = -n5
        >>> p6 = p5 + uc

        Secondly, the variables ``p1``, ``p3`` and ``p5`` are always ``[0, 0, 0]`` and
        ``p2``, ``p4`` and ``p6`` are always ``uc``.
        Hence this may be used to further reduce certain computations.
        """
        axis1 = direction(axis1)
        axis2 = direction(axis2)

        cell = self.cell
        n = cross3(cell[axis1], cell[axis2])
        # Normalize
        n /= dot3(n, n) ** 0.5
        # Now we need to figure out if the normal vector
        # is pointing outwards
        # Take the cell center
        up = cell.sum(0)
        # Calculate the distance from the plane to the center of the cell

        # If d is positive then the normal vector is pointing towards
        # the center, so rotate 180
        if dot3(n, up / 2) > 0.0:
            n *= -1

        if origin:
            return n, _a.zerosd([3])
        # We have to reverse the normal vector
        return -n, up

    def __mul__(self, m) -> Lattice:
        """Implement easy repeat function

        Parameters
        ----------
        m : int or array_like of length 3
           a single integer may be regarded as [m, m, m].
           A list will expand the unit-cell along the equivalent lattice vector.

        Returns
        -------
        Lattice
             enlarged supercell
        """
        # Simple form
        if isinstance(m, Integral):
            return self.tile(m, 0).tile(m, 1).tile(m, 2)

        lattice = self.copy()
        for i, r in enumerate(m):
            lattice = lattice.tile(r, i)
        return lattice

    @property
    def icell(self) -> ndarray:
        """Returns the reciprocal (inverse) cell for the `Lattice`.

        Note: The returned vectors are still in ``[0, :]`` format
        and not as returned by an inverse LAPACK algorithm.
        """
        return cell_invert(self.cell)

    @property
    def rcell(self) -> ndarray:
        """Returns the reciprocal cell for the `Lattice` with ``2*np.pi``

        Note: The returned vectors are still in [0, :] format
        and not as returned by an inverse LAPACK algorithm.
        """
        return cell_reciprocal(self.cell)

    def cell2length(self, length, axes: CellAxes = (0, 1, 2)) -> ndarray:
        """Calculate cell vectors such that they each have length `length`

        Parameters
        ----------
        length : float or array_like
            length for cell vectors, if an array it corresponds to the individual
            vectors and it must have length equal to `axes`
        axes :
            which axes the `length` variable refers too.

        Returns
        -------
        numpy.ndarray
             cell-vectors with prescribed length, same order as `axes`
        """
        axes = map(direction, listify(axes)) | listify

        length = _a.asarray(length).ravel()
        if len(length) != len(axes):
            if len(length) == 1:
                length = np.tile(length, len(axes))
            else:
                raise ValueError(
                    f"{self.__class__.__name__}.cell2length length parameter should be a single "
                    "float, or an array of values according to axes argument."
                )

        return self.cell[axes] * (length / self.length[axes]).reshape(-1, 1)

    def offset(self, isc=None) -> tuple[float, float, float]:
        """Returns the supercell offset of the supercell index"""
        if isc is None:
            return _a.arrayd([0, 0, 0])
        return dot(isc, self.cell)

    def __add__(self, other) -> Lattice:
        return self.add(other)

    __radd__ = __add__

    def add_vacuum(
        self, vacuum: float, axis: CellAxis, orthogonal_to_plane: bool = False
    ) -> Lattice:
        """Returns a new object with vacuum along the `axis` lattice vector

        Parameters
        ----------
        vacuum : float
           amount of vacuum added, in Ang
        axis :
           the lattice vector to add vacuum along
        orthogonal_to_plane : bool, optional
           whether the lattice vector should be elongated so that it is `vacuum` longer
           when projected onto the normal vector of the other two axis.
        """
        axis = direction(axis)
        cell = np.copy(self.cell)
        d = cell[axis].copy()
        d /= fnorm(d)
        if orthogonal_to_plane:
            # first calculate the normal vector of the other plane
            n = cross3(cell[axis - 1], cell[axis - 2])
            n /= fnorm(n)
            # now project onto cell
            projection = n @ d

            # calculate how long it should be so that the normal vector
            # is `vacuum` longer
            scale = vacuum / abs(projection)
        else:
            scale = vacuum
        # normalize to get direction vector
        cell[axis] += d * scale
        return self.copy(cell)

    def sc_index(self, sc_off) -> Union[int, Sequence[int]]:
        """Returns the integer index in the sc_off list that corresponds to `sc_off`

        Returns the index for the supercell in the global offset.

        Parameters
        ----------
        sc_off : (3,) or list of (3,)
            super cell specification. For each axis having value ``None`` all supercells
            along that axis is returned.
        """

        def _assert(m, v):
            if np.any(np.abs(v) > m):
                raise ValueError("Requesting a non-existing supercell index")

        hsc = self.nsc // 2

        if len(sc_off) == 0:
            return _a.arrayi([[]])

        elif isinstance(sc_off[0], ndarray):
            _assert(hsc[0], sc_off[:, 0])
            _assert(hsc[1], sc_off[:, 1])
            _assert(hsc[2], sc_off[:, 2])
            return self._isc_off[sc_off[:, 0], sc_off[:, 1], sc_off[:, 2]]

        elif isinstance(sc_off[0], (tuple, list)):
            # We are dealing with a list of lists
            sc_off = np.asarray(sc_off)
            _assert(hsc[0], sc_off[:, 0])
            _assert(hsc[1], sc_off[:, 1])
            _assert(hsc[2], sc_off[:, 2])
            return self._isc_off[sc_off[:, 0], sc_off[:, 1], sc_off[:, 2]]

        # Fall back to the other routines
        sc_off = self._fill_sc(sc_off)
        if sc_off[0] is not None and sc_off[1] is not None and sc_off[2] is not None:
            _assert(hsc[0], sc_off[0])
            _assert(hsc[1], sc_off[1])
            _assert(hsc[2], sc_off[2])
            return self._isc_off[sc_off[0], sc_off[1], sc_off[2]]

        # We build it because there are 'none'
        if sc_off[0] is None:
            idx = _a.arangei(self.n_s)
        else:
            idx = (self.sc_off[:, 0] == sc_off[0]).nonzero()[0]

        if not sc_off[1] is None:
            idx = idx[(self.sc_off[idx, 1] == sc_off[1]).nonzero()[0]]

        if not sc_off[2] is None:
            idx = idx[(self.sc_off[idx, 2] == sc_off[2]).nonzero()[0]]

        return idx

    def vertices(self) -> ndarray:
        """Vertices of the cell

        Returns
        --------
        array of shape (2, 2, 2, 3):
            The coordinates of the vertices of the cell. The first three dimensions
            correspond to each cell axis (off, on), and the last one contains the xyz coordinates.
        """
        verts = np.zeros([2, 2, 2, 3])
        verts[1, :, :, 0] = 1
        verts[:, 1, :, 1] = 1
        verts[:, :, 1, 2] = 1
        return verts @ self.cell

    @classmethod
    def tocell(cls, *args) -> Lattice:
        r"""Returns a 3x3 unit-cell dependent on the input

        1 argument
          a unit-cell along Cartesian coordinates with side-length
          equal to the argument.

        3 arguments
          the diagonal components of a Cartesian unit-cell

        6 arguments
          the cell parameters given by :math:`a`, :math:`b`, :math:`c`,
          :math:`\alpha`, :math:`\beta` and :math:`\gamma` (angles
          in degrees).

        9 arguments
          a 3x3 unit-cell.

        Parameters
        ----------
        *args : float
            May be either, 1, 3, 6 or 9 elements.
            Note that the arguments will be put into an array and flattened
            before checking the number of arguments.

        Examples
        --------
        >>> cell_1_1_1 = Lattice.tocell(1.)
        >>> cell_1_2_3 = Lattice.tocell(1., 2., 3.)
        >>> cell_1_2_3 = Lattice.tocell([1., 2., 3.]) # same as above
        """
        # Convert into true array (flattened)
        args = _a.arrayd(args, order="C").ravel()
        nargs = len(args)

        # A square-box
        if nargs == 1:
            return np.diag([args[0]] * 3)

        # Diagonal components
        if nargs == 3:
            return np.diag(args)

        # Cell parameters
        if nargs == 6:
            cell = _a.zerosd([3, 3])
            a = args[0]
            b = args[1]
            c = args[2]
            alpha = args[3]
            beta = args[4]
            gamma = args[5]

            from math import cos, pi, sin, sqrt

            pi180 = pi / 180.0

            cell[0, 0] = a
            g = gamma * pi180
            cg = cos(g)
            sg = sin(g)
            cell[1, 0] = b * cg
            cell[1, 1] = b * sg
            b = beta * pi180
            cb = cos(b)
            sb = sin(b)
            cell[2, 0] = c * cb
            a = alpha * pi180
            d = (cos(a) - cb * cg) / sg
            cell[2, 1] = c * d
            cell[2, 2] = c * sqrt(sb**2 - d**2)
            return cell

        # A complete cell
        if nargs == 9:
            return args.reshape(3, 3)

        raise ValueError(
            f"Creating a unit cell has to have 1, 3, 6 or 9 arguments, got {nargs}."
        )

    @deprecate_argument(
        "tol",
        "rtol",
        "argument tol has been deprecated in favor of rtol, please update your code.",
        "0.15",
        "0.17",
    )
    def is_orthogonal(self, rtol: float = 0.001) -> bool:
        """
        Returns true if the cell vectors are orthogonal.

        Internally this will be done on the normalized lattice vectors
        to ensure no numerical instability.

        Parameters
        -----------
        rtol: float, optional
            the threshold above which the scalar product of two normalized cell
            vectors will be considered non-zero.
        """
        # Convert to unit-vector cell
        cell = self.cell
        cl = fnorm(cell)
        cell[0] = cell[0] / cl[0]
        cell[1] = cell[1] / cl[1]
        cell[2] = cell[2] / cl[2]
        i_s = dot3(cell[0], cell[1]) < rtol
        i_s = dot3(cell[0], cell[2]) < rtol and i_s
        i_s = dot3(cell[1], cell[2]) < rtol and i_s
        return i_s

    @deprecate_argument(
        "tol",
        "atol",
        "argument tol has been deprecated in favor of atol, please update your code.",
        "0.15",
        "0.17",
    )
    def is_cartesian(self, atol: float = 0.001) -> bool:
        """
        Checks if cell vectors a,b,c are multiples of the cartesian axis vectors (x, y, z)

        Parameters
        -----------
        atol: float, optional
            the threshold above which an off diagonal term will be considered non-zero.
        """
        # Get the off diagonal terms of the cell
        off_diagonal = self.cell.ravel()[:-1].reshape(2, 4)[:, 1:]

        # Check if all are bolew the threshold tolerance
        return np.all(np.abs(off_diagonal) <= atol)

    def parallel(self, other, axes: CellAxes = (0, 1, 2)) -> bool:
        """Returns true if the cell vectors are parallel to `other`

        Parameters
        ----------
        other : Lattice
           the other object to check whether the axis are parallel
        axes :
           only check the specified axes (default to all)
        """
        axis = map(direction, listify(axes))

        # Convert to unit-vector cell
        for i in axis:
            a = self.cell[i] / fnorm(self.cell[i])
            b = other.cell[i] / fnorm(other.cell[i])
            if abs(dot3(a, b) - 1) > 0.001:
                return False
        return True

    def angle(self, axis1: CellAxis, axis2: CellAxis, rad: bool = False) -> float:
        """The angle between two of the cell vectors

        Parameters
        ----------
        axis1 :
           the first cell vector
        axis2 :
           the second cell vector
        rad : bool, optional
           whether the returned value is in radians
        """
        axis1 = direction(axis1)
        axis2 = direction(axis2)
        n = fnorm(self.cell[[axis1, axis2]])
        ang = math.acos(dot3(self.cell[axis1], self.cell[axis2]) / (n[0] * n[1]))
        if rad:
            return ang
        return math.degrees(ang)

    @staticmethod
    def read(sile, *args, **kwargs) -> Lattice:
        """Reads the supercell from the `Sile` using ``Sile.read_lattice``

        Parameters
        ----------
        sile : Sile, str or pathlib.Path
            a `Sile` object which will be used to read the supercell
            if it is a string it will create a new sile using `sisl.io.get_sile`.
        """
        # This only works because, they *must*
        # have been imported previously
        from sisl.io import BaseSile, get_sile

        if isinstance(sile, BaseSile):
            return sile.read_lattice(*args, **kwargs)
        else:
            with get_sile(sile, mode="r") as fh:
                return fh.read_lattice(*args, **kwargs)

    @deprecate_argument(
        "tol",
        "atol",
        "argument tol has been deprecated in favor of atol, please update your code.",
        "0.15",
        "0.17",
    )
    def equal(self, other, atol: float = 1e-4) -> bool:
        """Check whether two lattices are equivalent

        Parameters
        ----------
        other : Lattice
           the other object to check whether the lattice is equivalent
        atol : float, optional
            tolerance value for the cell vectors and origin
        """
        if not isinstance(other, (Lattice, LatticeChild)):
            return False
        same = np.allclose(self.cell, other.cell, atol=atol)
        same = same and np.allclose(self.nsc, other.nsc)
        same = same and np.allclose(self.origin, other.origin, atol=atol)
        return same

    def __str__(self) -> str:
        """Returns a string representation of the object"""

        # Create format for lattice vectors
        def bcstr(bc):
            left = BoundaryCondition.getitem(bc[0]).name.capitalize()
            if bc[0] == bc[1]:
                # single string
                return left
            right = BoundaryCondition.getitem(bc[1]).name.capitalize()
            return f"[{left}, {right}]"

        s = ",\n ".join(
            [
                "ABC"[i] + "=[{:.4f}, {:.4f}, {:.4f}]".format(*self.cell[i])
                for i in (0, 1, 2)
            ]
        )
        origin = "[{:.4f}, {:.4f}, {:.4f}]".format(*self.origin)
        bc = ",\n     ".join(map(bcstr, self.boundary_condition))
        return f"{self.__class__.__name__}{{nsc: {self.nsc},\n origin={origin},\n {s},\n bc=[{bc}]\n}}"

    def __repr__(self) -> str:
        abc, abg = self.parameters()
        a, b, c = map(lambda r: round(r, 4), abc.tolist())
        alpha, beta, gamma = map(lambda r: round(r, 4), abg.tolist())

        BC = BoundaryCondition
        bc = self.boundary_condition

        def bcstr(bc):
            left = BC.getitem(bc[0]).name[0]
            if bc[0] == bc[1]:
                # single string
                return left
            right = BC.getitem(bc[1]).name[0]
            return f"[{left}, {right}]"

        bc = ", ".join(map(bcstr, self.boundary_condition))
        return f"<{self.__module__}.{self.__class__.__name__} a={a}, b={b}, c={c}, α={alpha}, β={beta}, γ={gamma}, bc=[{bc}], nsc={self.nsc}>"

    def __eq__(self, other) -> bool:
        """Equality check"""
        return self.equal(other)

    def __ne__(self, b) -> bool:
        """In-equality check"""
        return not (self == b)

    # Create pickling routines
    def __getstate__(self):
        """Returns the state of this object"""
        return {
            "cell": self.cell,
            "nsc": self.nsc,
            "sc_off": self.sc_off,
            "origin": self.origin,
        }

    def __setstate__(self, d):
        """Re-create the state of this object"""
        self.__init__(d["cell"], d["nsc"], d["origin"])
        self.sc_off = d["sc_off"]


new_dispatch = Lattice.new
to_dispatch = Lattice.to


# Define base-class for this
class LatticeNewDispatch(AbstractDispatch):
    """Base dispatcher from class passing arguments to Lattice class"""


class LatticeNewLatticeDispatch(LatticeNewDispatch):
    def dispatch(self, lattice, copy: bool = False) -> Lattice:
        """Return Lattice as-is, for sanitization purposes"""
        cls = self._get_class()
        if cls != lattice.__class__:
            lattice = cls(
                lattice.cell.copy(),
                nsc=lattice.nsc.copy(),
                origin=lattice.origin.copy(),
                boundary_condition=lattice.boundary_condition.copy(),
            )
            copy = False
        if copy:
            return lattice.copy()
        return lattice


new_dispatch.register(Lattice, LatticeNewLatticeDispatch)


class LatticeNewListLikeDispatch(LatticeNewDispatch):
    def dispatch(self, cell, *args, **kwargs) -> Lattice:
        """Converts simple `array-like` variables to a `Lattice`

        Examples
        --------

        >>> Lattice.new([1, 2, 3]) == Lattice([1, 2, 3])
        """
        return Lattice(cell, *args, **kwargs)


# A cell can be created form a ndarray/list/tuple

new_dispatch.register("ndarray", LatticeNewListLikeDispatch)
new_dispatch.register(np.ndarray, LatticeNewListLikeDispatch)
new_dispatch.register(int, LatticeNewListLikeDispatch)
new_dispatch.register(float, LatticeNewListLikeDispatch)
new_dispatch.register(list, LatticeNewListLikeDispatch)
new_dispatch.register(tuple, LatticeNewListLikeDispatch)


class LatticeNewAseDispatch(LatticeNewDispatch):
    def dispatch(self, aseg) -> Lattice:
        """`ase.Cell` conversion to `Lattice`"""
        cls = self._get_class(allow_instance=True)
        cell = aseg.get_cell()
        nsc = [3 if pbc else 1 for pbc in aseg.pbc]
        return cls(cell, nsc=nsc)


new_dispatch.register("ase", LatticeNewAseDispatch)

# currently we can't ensure the ase Atoms type
# to get it by type(). That requires ase to be importable.
try:
    from ase import Cell as ase_Cell

    new_dispatch.register(ase_Cell, LatticeNewAseDispatch)

    # ensure we don't pollute name-space
    del ase_Cell

except Exception:
    pass


class LatticeNewFileDispatch(LatticeNewDispatch):
    def dispatch(self, *args, **kwargs) -> Lattice:
        """Defer the `Lattice.read` method by passing down arguments"""
        cls = self._get_class()
        return cls.read(*args, **kwargs)


new_dispatch.register(str, LatticeNewFileDispatch)
new_dispatch.register(Path, LatticeNewFileDispatch)
# see sisl/__init__.py for new_dispatch.register(BaseSile, ...)


class LatticeToDispatch(AbstractDispatch):
    """Base dispatcher from class passing from Lattice class"""


class LatticeToAseDispatch(LatticeToDispatch):
    def dispatch(self, **kwargs) -> ase.Cell:
        """`Lattice` conversion to an `ase.Cell` object."""
        from ase import Cell as ase_Cell

        lattice = self._get_object()
        return ase_Cell(lattice.cell.copy())


to_dispatch.register("ase", LatticeToAseDispatch)


class LatticeToSileDispatch(LatticeToDispatch):
    def dispatch(self, *args, **kwargs) -> Any:
        """`Lattice` writing to a sile.

        Examples
        --------

        >>> geom = si.geom.graphene()
        >>> geom.lattice.to("hello.xyz")
        >>> geom.lattice.to(pathlib.Path("hello.xyz"))
        """
        lattice = self._get_object()
        return lattice.write(*args, **kwargs)


to_dispatch.register("str", LatticeToSileDispatch)
to_dispatch.register("Path", LatticeToSileDispatch)
# to do geom.to[Path](path)
to_dispatch.register(str, LatticeToSileDispatch)
to_dispatch.register(Path, LatticeToSileDispatch)


class LatticeToCuboidDispatch(LatticeToDispatch):
    def dispatch(self, center=None, origin=None, orthogonal=False) -> Cuboid:
        """Convert lattice parameters to a `Cuboid`"""
        lattice = self._get_object()

        cell = lattice.cell.copy()

        if center is None:
            center = lattice.center()
        center = _a.asarray(center)

        if origin is None:
            origin = lattice.origin
        origin = _a.asarray(origin)

        center_off = center + origin

        if not orthogonal:
            return Cuboid(cell, center_off)

        def find_min_max(cmin, cmax, new):
            for i in range(3):
                cmin[i] = min(cmin[i], new[i])
                cmax[i] = max(cmax[i], new[i])

        cmin = cell.min(0)
        cmax = cell.max(0)
        find_min_max(cmin, cmax, cell[[0, 1]].sum(0))
        find_min_max(cmin, cmax, cell[[0, 2]].sum(0))
        find_min_max(cmin, cmax, cell[[1, 2]].sum(0))
        find_min_max(cmin, cmax, cell.sum(0))
        return Cuboid(cmax - cmin, center_off)


to_dispatch.register("Cuboid", LatticeToCuboidDispatch)
to_dispatch.register(Cuboid, LatticeToCuboidDispatch)


@set_module("sisl")
class SuperCell(Lattice):
    """Deprecated class, please use `Lattice` instead"""

    def __init__(self, *args, **kwargs):
        deprecate(
            f"{self.__class__.__name__} is deprecated; please use 'Lattice' class instead",
            "0.15",
            "0.17",
        )
        super().__init__(*args, **kwargs)


@set_module("sisl")
class LatticeChild:
    """Class to be inherited by using the ``self.lattice`` as a `Lattice` object

    Initialize by a `Lattice` object and get access to several different
    routines directly related to the `Lattice` class.
    """

    @property
    def sc(self):
        """[deprecated] Return the lattice object associated with the `Lattice`."""
        deprecate(
            f"{self.__class__.__name__}.sc is deprecated; please use 'lattice' instead",
            "0.15",
            "0.17",
        )
        return self.lattice

    def set_nsc(self, *args, **kwargs):
        """Set the number of super-cells in the `Lattice` object

        See `set_nsc` for allowed parameters.

        See Also
        --------
        Lattice.set_nsc : the underlying called method
        """
        self.lattice.set_nsc(*args, **kwargs)

    def set_lattice(self, lattice: LatticeLike):
        """Overwrites the local lattice."""
        if lattice is None:
            # Default supercell is a simple
            # 1x1x1 unit-cell
            lattice = [1.0, 1.0, 1.0]
        self.lattice = Lattice.new(lattice)

    set_supercell = deprecation(
        "set_sc is deprecated; please use set_lattice instead", "0.15", "0.17"
    )(set_lattice)

    @property
    def length(self) -> float:
        """Returns the inherent `Lattice.length`"""
        return self.lattice.length

    @property
    def volume(self) -> float:
        """Returns the inherent `Lattice.volume`"""
        return self.lattice.volume

    def area(self, ax0, ax1) -> float:
        """Calculate the area spanned by the two axis `ax0` and `ax1`"""
        return self.lattice.area(ax0, ax1)

    @property
    def cell(self) -> ndarray:
        """Returns the inherent `Lattice.cell`"""
        return self.lattice.cell

    @property
    def icell(self) -> ndarray:
        """Returns the inherent `Lattice.icell`"""
        return self.lattice.icell

    @property
    def rcell(self) -> ndarray:
        """Returns the inherent `Lattice.rcell`"""
        return self.lattice.rcell

    @property
    def origin(self) -> ndarray:
        """Returns the inherent `Lattice.origin`"""
        return self.lattice.origin

    @property
    def n_s(self) -> int:
        """Returns the inherent `Lattice.n_s`"""
        return self.lattice.n_s

    @property
    def nsc(self) -> ndarray:
        """Returns the inherent `Lattice.nsc`"""
        return self.lattice.nsc

    @property
    def sc_off(self) -> ndarray:
        """Returns the inherent `Lattice.sc_off`"""
        return self.lattice.sc_off

    @property
    def isc_off(self) -> ndarray:
        """Returns the inherent `Lattice.isc_off`"""
        return self.lattice.isc_off

    def sc_index(self, *args, **kwargs) -> Union[int, Sequence[int]]:
        """Call local `Lattice.sc_index` function"""
        return self.lattice.sc_index(*args, **kwargs)

    @property
    def boundary_condition(self) -> np.ndarray:
        f"""{Lattice.boundary_condition.__doc__}"""
        return self.lattice.boundary_condition

    @boundary_condition.setter
    def boundary_condition(self, boundary_condition: Sequence[BoundaryConditionType]):
        f"""{Lattice.boundary_condition.__doc__}"""
        raise SislError(
            f"Cannot use property to set boundary conditions of LatticeChild"
        )

    @property
    def pbc(self) -> np.ndarray:
        __doc__ = Lattice.pbc.__doc__
        return self.lattice.pbc


class LatticeNewLatticeChildDispatch(LatticeNewDispatch):
    def dispatch(self, obj, copy: bool = False) -> Lattice:
        """Extraction of `Lattice` object from a `LatticeChild` object."""
        # for sanitation purposes
        if copy:
            return obj.lattice.copy()
        return obj.lattice


new_dispatch.register(LatticeChild, LatticeNewLatticeChildDispatch)

# Remove references
del new_dispatch, to_dispatch
