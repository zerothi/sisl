# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
""" Define a lattice with cell-parameters and supercells

This class is the basis of many different objects.
"""
from __future__ import annotations

import math
import warnings
from numbers import Integral
from typing import TYPE_CHECKING, Tuple, Union

import numpy as np
from numpy import dot, ndarray

from . import _array as _a
from . import _plot as plt
from ._internal import set_module
from ._lattice import cell_invert, cell_reciprocal
from ._math_small import cross3, dot3
from .messages import deprecate, deprecate_argument, deprecation
from .quaternion import Quaternion
from .shape.prism4 import Cuboid
from .utils.mathematics import fnorm

__all__ = ["Lattice", "SuperCell", "LatticeChild"]


@set_module("sisl")
class Lattice:
    r""" A cell class to retain lattice vectors and a supercell structure

    The supercell structure is comprising the *primary* unit-cell and neighbouring
    unit-cells. The number of supercells is given by the attribute `nsc` which
    is a vector with 3 elements, one per lattice vector. It describes *how many*
    times the primary unit-cell is extended along the i'th lattice vector.
    For ``nsc[i] == 3`` the supercell is made up of 3 unit-cells. One *behind*, the
    primary unit-cell and one *after*.

    Parameters
    ----------
    cell : array_like
       the lattice parameters of the unit cell (the actual cell
       is returned from `tocell`.
    nsc : array_like of int
       number of supercells along each latticevector
    origin : (3,) of float, optional
       the origin of the supercell.
    """

    # We limit the scope of this Lattice object.
    __slots__ = ('cell', '_origin', 'nsc', 'n_s', '_sc_off', '_isc_off')

    def __init__(self, cell, nsc=None, origin=None):

        if nsc is None:
            nsc = [1, 1, 1]

        # If the length of cell is 6 it must be cell-parameters, not
        # actual cell coordinates
        self.cell = self.tocell(cell)

        if origin is None:
            self._origin = _a.zerosd(3)
        else:
            self._origin = _a.arrayd(origin)
            if self._origin.size != 3:
                raise ValueError("Origin *must* be 3 numbers.")

        self.nsc = _a.onesi(3)
        # Set the super-cell
        self.set_nsc(nsc=nsc)

    @property
    def length(self) -> ndarray:
        """ Length of each lattice vector """
        return fnorm(self.cell)

    @property
    def volume(self):
        """ Volume of cell """
        return abs(dot3(self.cell[0, :], cross3(self.cell[1, :], self.cell[2, :])))

    def area(self, ax0, ax1):
        """ Calculate the area spanned by the two axis `ax0` and `ax1` """
        return (cross3(self.cell[ax0, :], self.cell[ax1, :]) ** 2).sum() ** 0.5

    @property
    def origin(self) -> ndarray:
        """ Origin for the cell """
        return self._origin

    @origin.setter
    def origin(self, origin):
        """ Set origin for the cell """
        self._origin[:] = origin

    def toCuboid(self, orthogonal=False):
        """ A cuboid with vectors as this unit-cell and center with respect to its origin

        Parameters
        ----------
        orthogonal : bool, optional
           if true the cuboid has orthogonal sides such that the entire cell is contained
        """
        if not orthogonal:
            return Cuboid(self.cell.copy(), self.center() + self.origin)
        def find_min_max(cmin, cmax, new):
            for i in range(3):
                cmin[i] = min(cmin[i], new[i])
                cmax[i] = max(cmax[i], new[i])
        cell = self.cell
        cmin = cell.min(0)
        cmax = cell.max(0)
        find_min_max(cmin, cmax, cell[[0, 1], :].sum(0))
        find_min_max(cmin, cmax, cell[[0, 2], :].sum(0))
        find_min_max(cmin, cmax, cell[[1, 2], :].sum(0))
        find_min_max(cmin, cmax, cell.sum(0))
        return Cuboid(cmax - cmin, self.center() + self.origin)

    def parameters(self, rad=False) -> Tuple[float, float, float, float, float, float]:
        r""" Cell parameters of this cell in 3 lengths and 3 angles

        Notes
        -----
        Since we return the length and angles between vectors it may not be possible to
        recreate the same cell. Only in the case where the first lattice vector *only*
        has a Cartesian :math:`x` component will this be the case

        Parameters
        ----------
        rad : bool, optional
           whether the angles are returned in radians (otherwise in degree)

        Returns
        -------
        float
            length of first lattice vector
        float
            length of second lattice vector
        float
            length of third lattice vector
        float
            angle between b and c vectors
        float
            angle between a and c vectors
        float
            angle between a and b vectors
        """
        if rad:
            f = 1.
        else:
            f = 180 / np.pi

        # Calculate length of each lattice vector
        cell = self.cell.copy()
        abc = fnorm(cell)

        from math import acos
        cell = cell / abc.reshape(-1, 1)
        alpha = acos(dot3(cell[1, :], cell[2, :])) * f
        beta = acos(dot3(cell[0, :], cell[2, :])) * f
        gamma = acos(dot3(cell[0, :], cell[1, :])) * f

        return abc[0], abc[1], abc[2], alpha, beta, gamma

    def _fill(self, non_filled, dtype=None):
        """ Return a zero filled array of length 3 """

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
        """ Return a filled supercell index by filling in zeros where needed """
        return self._fill(supercell_index, dtype=np.int32)

    def set_nsc(self, nsc=None, a=None, b=None, c=None):
        """ Sets the number of supercells in the 3 different cell directions

        Parameters
        ----------
        nsc : list of int, optional
           number of supercells in each direction
        a : integer, optional
           number of supercells in the first unit-cell vector direction
        b : integer, optional
           number of supercells in the second unit-cell vector direction
        c : integer, optional
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
                "Supercells has to be of un-even size. The primary cell counts " +
                "one, all others count 2")

        # We might use this very often, hence we store it
        self.n_s = _a.prodi(self.nsc)
        self._sc_off = _a.zerosi([self.n_s, 3])
        self._isc_off = _a.zerosi(self.nsc)

        n = self.nsc
        # We define the following ones like this:

        def ret_range(val):
            i = val // 2
            return range(-i, i+1)
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
        """ Internal routine for updating the supercell indices """
        for i in range(self.n_s):
            d = self.sc_off[i, :]
            self._isc_off[d[0], d[1], d[2]] = i

    @property
    def sc_off(self) -> ndarray:
        """ Integer supercell offsets """
        return self._sc_off

    @sc_off.setter
    def sc_off(self, sc_off):
        """ Set the supercell offset """
        self._sc_off[:, :] = _a.arrayi(sc_off, order='C')
        self._update_isc_off()

    @property
    def isc_off(self) -> ndarray:
        """ Internal indexed supercell ``[ia, ib, ic] == i`` """
        return self._isc_off

    def __iter__(self):
        """ Iterate the supercells and the indices of the supercells """
        yield from enumerate(self.sc_off)

    def copy(self, cell=None, origin=None):
        """ A deepcopy of the object

        Parameters
        ----------
        cell : array_like
           the new cell parameters
        origin : array_like
           the new origin
        """
        if origin is None:
            origin = self.origin.copy()
        if cell is None:
            copy = self.__class__(np.copy(self.cell), nsc=np.copy(self.nsc), origin=origin)
        else:
            copy = self.__class__(np.copy(cell), nsc=np.copy(self.nsc), origin=origin)
        # Ensure that the correct super-cell information gets carried through
        if not np.allclose(copy.sc_off, self.sc_off):
            copy.sc_off = self.sc_off
        return copy

    def fit(self, xyz, axis=None, tol=0.05):
        """ Fit the supercell to `xyz` such that the unit-cell becomes periodic in the specified directions

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
        axis : None or array_like
           if ``None`` equivalent to ``[0, 1, 2]``, else only the cell-vectors
           along the provided axis will be used
        tol : float
           tolerance (in Angstrom) of the positions. I.e. we neglect coordinates
           which are not within the radius of this magnitude
        """
        # In case the passed coordinates are from a Geometry
        from .geometry import Geometry

        if isinstance(xyz, Geometry):
            xyz = xyz.xyz[:, :]

        cell = np.copy(self.cell[:, :])

        # Get fractional coordinates to get the divisions in the current cell
        x = dot(xyz, self.icell.T)

        # Now we should figure out the correct repetitions
        # by rounding to integer positions of the cell vectors
        ix = np.rint(x)

        # Figure out the displacements from integers
        # Then reduce search space by removing those coordinates
        # that are more than the tolerance.
        dist = np.sqrt((dot(cell.T, (x - ix).T) ** 2).sum(0))
        idx = (dist <= tol).nonzero()[0]
        if len(idx) == 0:
            raise ValueError('Could not fit the cell parameters to the coordinates '
                              'due to insufficient accuracy (try increase the tolerance)')

        # Reduce problem to allowed values below the tolerance
        ix = ix[idx, :]

        # Reduce to total repetitions
        ireps = np.amax(ix, axis=0) - np.amin(ix, axis=0) + 1

        # Only repeat the axis requested
        if isinstance(axis, Integral):
            axis = [axis]

        # Reduce the non-set axis
        if not axis is None:
            for ax in (0, 1, 2):
                if ax not in axis:
                    ireps[ax] = 1

        # Enlarge the cell vectors
        cell[0, :] *= ireps[0]
        cell[1, :] *= ireps[1]
        cell[2, :] *= ireps[2]

        return self.copy(cell)

    def swapaxes(self, axes_a: Union[int, str],
                 axes_b: Union[int, str],
                 what: str="abc") -> Lattice:
        r""" Swaps axes `axes_a` and `axes_b`

        Swapaxes is a versatile method for changing the order
        of axes elements, either lattice vector order, or Cartesian
        coordinate orders.

        Parameters
        ----------
        axes_a : int or str
           the old axis indices (or labels if `str`)
           A string will translate each character as a specific
           axis index.
           Lattice vectors are denoted by ``abc`` while the
           Cartesian coordinates are denote by ``xyz``.
           If `str`, then `what` is not used.
        axes_b : int or str
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
                raise ValueError(f"{self.__class__.__name__}.swapaxes could not understand 'what' "
                                 "must contain abc and/or xyz.")
        elif (not isinstance(axes_a, str)) or (not isinstance(axes_b, str)):
            raise ValueError(f"{self.__class__.__name__}.swapaxes axes arguments must be either all int or all str, not a mix.")

        cell = self.cell
        nsc = self.nsc
        origin = self.origin

        if len(axes_a) != len(axes_b):
            raise ValueError(f"{self.__class__.__name__}.swapaxes expects axes_a and axes_b to have the same lengeth {len(axes_a)}, {len(axes_b)}.")

        for a, b in zip(axes_a, axes_b):
            idx = [0, 1, 2]

            aidx = "abcxyz".index(a)
            bidx = "abcxyz".index(b)

            if aidx // 3 != bidx // 3:
                raise ValueError(f"{self.__class__.__name__}.swapaxes expects axes_a and axes_b to belong to the same category, do not mix lattice vector swaps with Cartesian coordinates.")

            if aidx < 3:
                idx[aidx], idx[bidx] = idx[bidx], idx[aidx]
                # we are dealing with lattice vectors
                cell = cell[idx]
                nsc = nsc[idx]

            else:
                aidx -= 3
                bidx -= 3
                idx[aidx], idx[bidx] = idx[bidx], idx[aidx]

                # we are dealing with cartesian coordinates
                cell = cell[:, idx]
                origin = origin[idx]

        return self.__class__(cell.copy(), nsc=nsc.copy(), origin=origin.copy())

    def plane(self, ax1, ax2, origin=True):
        """ Query point and plane-normal for the plane spanning `ax1` and `ax2`

        Parameters
        ----------
        ax1 : int
           the first axis vector
        ax2 : int
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
        cell = self.cell
        n = cross3(cell[ax1, :], cell[ax2, :])
        # Normalize
        n /= dot3(n, n) ** 0.5
        # Now we need to figure out if the normal vector
        # is pointing outwards
        # Take the cell center
        up = cell.sum(0)
        # Calculate the distance from the plane to the center of the cell

        # If d is positive then the normal vector is pointing towards
        # the center, so rotate 180
        if dot3(n, up / 2) > 0.:
            n *= -1

        if origin:
            return n, _a.zerosd([3])
        # We have to reverse the normal vector
        return -n, up

    def __mul__(self, m):
        """ Implement easy repeat function

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
    def icell(self):
        """ Returns the reciprocal (inverse) cell for the `Lattice`.

        Note: The returned vectors are still in ``[0, :]`` format
        and not as returned by an inverse LAPACK algorithm.
        """
        return cell_invert(self.cell)

    @property
    def rcell(self):
        """ Returns the reciprocal cell for the `Lattice` with ``2*np.pi``

        Note: The returned vectors are still in [0, :] format
        and not as returned by an inverse LAPACK algorithm.
        """
        return cell_reciprocal(self.cell)

    def cell2length(self, length, axes=(0, 1, 2)) -> ndarray:
        """ Calculate cell vectors such that they each have length `length`

        Parameters
        ----------
        length : float or array_like
            length for cell vectors, if an array it corresponds to the individual
            vectors and it must have length equal to `axes`
        axes : int or array_like, optional
            which axes the `length` variable refers too.

        Returns
        -------
        numpy.ndarray
             cell-vectors with prescribed length, same order as `axes`
        """
        if isinstance(axes, Integral):
            # ravel
            axes = [axes]
        else:
            axes = list(axes)

        length = _a.asarray(length).ravel()
        if len(length) != len(axes):
            if len(length) == 1:
                length = np.tile(length, len(axes))
            else:
                raise ValueError(f"{self.__class__.__name__}.cell2length length parameter should be a single "
                                 "float, or an array of values according to axes argument.")
        return self.cell[axes] * (length / self.length[axes]).reshape(-1, 1)

    @deprecate_argument("only", "what",
                        "argument only has been deprecated in favor of what, please update your code.",
                        "0.14.0")
    def rotate(self, angle, v,
               rad: bool=False,
               what: str="abc") -> Lattice:
        """ Rotates the supercell, in-place by the angle around the vector

        One can control which cell vectors are rotated by designating them
        individually with ``only='[abc]'``.

        Parameters
        ----------
        angle : float
             the angle of which the geometry should be rotated
        v     : array_like or str or int
             the vector around the rotation is going to happen
             ``v = [1,0,0]`` will rotate in the ``yz`` plane
        what : combination of ``"abc"``, str, optional
             only rotate the designated cell vectors.
        rad : bool, optional
             Whether the angle is in radians (True) or in degrees (False)
        """
        if isinstance(v, Integral):
            v = direction(v, abc=self.cell, xyz=np.diag([1, 1, 1]))
        elif isinstance(v, str):
            v = reduce(lambda a, b: a + direction(b, abc=self.cell, xyz=np.diag([1, 1, 1])), v, 0)
        # flatten => copy
        vn = _a.asarrayd(v).flatten()
        vn /= fnorm(vn)
        q = Quaternion(angle, vn, rad=rad)
        q /= q.norm()  # normalize the quaternion
        cell = np.copy(self.cell)
        idx = []
        for i, d in enumerate('abc'):
            if d in what:
                idx.append(i)
        if idx:
            cell[idx, :] = q.rotate(self.cell[idx, :])
        return self.copy(cell)

    def offset(self, isc=None):
        """ Returns the supercell offset of the supercell index """
        if isc is None:
            return _a.arrayd([0, 0, 0])
        return dot(isc, self.cell)

    def add(self, other):
        """ Add two supercell lattice vectors to each other

        Parameters
        ----------
        other : Lattice, array_like
           the lattice vectors of the other supercell to add
        """
        if not isinstance(other, Lattice):
            other = Lattice(other)
        cell = self.cell + other.cell
        origin = self.origin + other.origin
        nsc = np.where(self.nsc > other.nsc, self.nsc, other.nsc)
        return self.__class__(cell, nsc=nsc, origin=origin)

    def __add__(self, other):
        return self.add(other)

    __radd__ = __add__

    def add_vacuum(self, vacuum, axis, orthogonal_to_plane=False):
        """ Add vacuum along the `axis` lattice vector

        Parameters
        ----------
        vacuum : float
           amount of vacuum added, in Ang
        axis : int
           the lattice vector to add vacuum along
        orthogonal_to_plane : bool, optional
           whether the lattice vector should be elongated so that it is `vacuum` longer
           when projected onto the normal vector of the other two axis.
        """
        cell = np.copy(self.cell)
        d = cell[axis, :].copy()
        d /= fnorm(d)
        if orthogonal_to_plane:
            # first calculate the normal vector of the other plane
            n = cross3(cell[axis-1], cell[axis-2])
            n /= fnorm(n)
            # now project onto cell
            projection = n @ d

            # calculate how long it should be so that the normal vector
            # is `vacuum` longer
            scale = vacuum / abs(projection)
        else:
            scale = vacuum
        # normalize to get direction vector
        cell[axis, :] += d * scale
        return self.copy(cell)

    def sc_index(self, sc_off):
        """ Returns the integer index in the sc_off list that corresponds to `sc_off`

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

    def vertices(self):
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

    def scale(self, scale, what="abc"):
        """ Scale lattice vectors

        Does not scale `origin`.

        Parameters
        ----------
        scale : float or (3,)
           the scale factor for the new lattice vectors.
        what: {"abc", "xyz"}
           If three different scale factors are provided, whether each scaling factor
           is to be applied on the corresponding lattice vector ("abc") or on the
           corresponding cartesian coordinate ("xyz").
        """
        if what == "abc":
            return self.copy((self.cell.T * scale).T)
        if what == "xyz":
            return self.copy(self.cell * scale)
        raise ValueError(f"{self.__class__.__name__}.scale argument what='{what}' is not in ['abc', 'xyz'].")

    def tile(self, reps, axis):
        """ Extend the unit-cell `reps` times along the `axis` lattice vector

        Notes
        -----
        This is *exactly* equivalent to the `repeat` routine.

        Parameters
        ----------
        reps : int
            number of times the unit-cell is repeated along the specified lattice vector
        axis : int
            the lattice vector along which the repetition is performed
        """
        cell = np.copy(self.cell)
        nsc = np.copy(self.nsc)
        origin = np.copy(self.origin)
        cell[axis, :] *= reps
        # Only reduce the size if it is larger than 5
        if nsc[axis] > 3 and reps > 1:
            # This is number of connections for the primary cell
            h_nsc = nsc[axis] // 2
            # The new number of supercells will then be
            nsc[axis] = max(1, int(math.ceil(h_nsc / reps))) * 2 + 1
        return self.__class__(cell, nsc=nsc, origin=origin)

    def repeat(self, reps, axis):
        """ Extend the unit-cell `reps` times along the `axis` lattice vector

        Notes
        -----
        This is *exactly* equivalent to the `tile` routine.

        Parameters
        ----------
        reps : int
            number of times the unit-cell is repeated along the specified lattice vector
        axis : int
            the lattice vector along which the repetition is performed
        """
        return self.tile(reps, axis)

    def untile(self, reps, axis):
        """Reverses a `Lattice.tile` and returns the segmented version

        Notes
        -----
        Untiling will not correctly re-calculate nsc since it has no
        knowledge of connections.

        See Also
        --------
        tile : opposite of this method
        """
        cell = np.copy(self.cell)
        cell[axis, :] /= reps
        return self.copy(cell)

    unrepeat = untile

    def append(self, other, axis):
        """ Appends other `Lattice` to this grid along axis """
        cell = np.copy(self.cell)
        cell[axis, :] += other.cell[axis, :]
        # TODO fix nsc here
        return self.copy(cell)

    def prepend(self, other, axis):
        """ Prepends other `Lattice` to this grid along axis

        For a `Lattice` object this is equivalent to `append`.
        """
        return self.append(other, axis)

    def translate(self, v):
        """ Appends additional space to the object """
        # check which cell vector resembles v the most,
        # use that
        cell = np.copy(self.cell)
        p = np.empty([3], np.float64)
        cl = fnorm(cell)
        for i in range(3):
            p[i] = abs(np.sum(cell[i, :] * v)) / cl[i]
        cell[np.argmax(p), :] += v
        return self.copy(cell)
    move = translate

    def center(self, axis=None):
        """ Returns center of the `Lattice`, possibly with respect to an axis """
        if axis is None:
            return self.cell.sum(0) * 0.5
        return self.cell[axis, :] * 0.5

    @classmethod
    def tocell(cls, *args):
        r""" Returns a 3x3 unit-cell dependent on the input

        1 argument
          a unit-cell along Cartesian coordinates with side-length
          equal to the argument.

        3 arguments
          the diagonal components of a Cartesian unit-cell

        6 arguments
          the cell parameters give by :math:`a`, :math:`b`, :math:`c`,
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
        args = _a.asarrayd(args).ravel()
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
            pi180 = pi / 180.

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
            cell[2, 2] = c * sqrt(sb ** 2 - d ** 2)
            return cell

        # A complete cell
        if nargs == 9:
            return args.copy().reshape(3, 3)

        raise ValueError(
            "Creating a unit-cell has to have 1, 3 or 6 arguments, please correct.")

    def is_orthogonal(self, tol=0.001):
        """
        Returns true if the cell vectors are orthogonal.

        Parameters
        -----------
        tol: float, optional
            the threshold above which the scalar product of two cell vectors will be considered non-zero.
        """
        # Convert to unit-vector cell
        cell = np.copy(self.cell)
        cl = fnorm(cell)
        cell[0, :] = cell[0, :] / cl[0]
        cell[1, :] = cell[1, :] / cl[1]
        cell[2, :] = cell[2, :] / cl[2]
        i_s = dot3(cell[0, :], cell[1, :]) < tol
        i_s = dot3(cell[0, :], cell[2, :]) < tol and i_s
        i_s = dot3(cell[1, :], cell[2, :]) < tol and i_s
        return i_s

    def is_cartesian(self, tol=0.001):
        """
        Checks if cell vectors a,b,c are multiples of the cartesian axis vectors (x, y, z)

        Parameters
        -----------
        tol: float, optional
            the threshold above which an off diagonal term will be considered non-zero.
        """
        # Get the off diagonal terms of the cell
        off_diagonal = self.cell.ravel()[:-1].reshape(2, 4)[:, 1:]
        # Check if any of them are above the threshold tolerance
        return ~np.any(np.abs(off_diagonal) > tol)

    def parallel(self, other, axis=(0, 1, 2)):
        """ Returns true if the cell vectors are parallel to `other`

        Parameters
        ----------
        other : Lattice
           the other object to check whether the axis are parallel
        axis : int or array_like
           only check the specified axis (default to all)
        """
        axis = _a.asarrayi(axis).ravel()
        # Convert to unit-vector cell
        for i in axis:
            a = self.cell[i, :] / fnorm(self.cell[i, :])
            b = other.cell[i, :] / fnorm(other.cell[i, :])
            if abs(dot3(a, b) - 1) > 0.001:
                return False
        return True

    def angle(self, i, j, rad=False):
        """ The angle between two of the cell vectors

        Parameters
        ----------
        i : int
           the first cell vector
        j : int
           the second cell vector
        rad : bool, optional
           whether the returned value is in radians
        """
        n = fnorm(self.cell[[i, j], :])
        ang = math.acos(dot3(self.cell[i, :], self.cell[j, :]) / (n[0] * n[1]))
        if rad:
            return ang
        return math.degrees(ang)

    @staticmethod
    def read(sile, *args, **kwargs):
        """ Reads the supercell from the `Sile` using ``Sile.read_lattice``

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
            with get_sile(sile, mode='r') as fh:
                return fh.read_lattice(*args, **kwargs)

    def equal(self, other, tol=1e-4):
        """ Check whether two lattices are equivalent

        Parameters
        ----------
        tol : float, optional
            tolerance value for the cell vectors and origin
        """
        if not isinstance(other, (Lattice, LatticeChild)):
            return False
        same = np.allclose(self.cell, other.cell, atol=tol)
        same = same and np.allclose(self.nsc, other.nsc)
        same = same and np.allclose(self.origin, other.origin, atol=tol)
        return same

    def __str__(self):
        """ Returns a string representation of the object """
        # Create format for lattice vectors
        s = ',\n '.join(['ABC'[i] + '=[{:.4f}, {:.4f}, {:.4f}]'.format(*self.cell[i]) for i in (0, 1, 2)])
        origin = "{:.4f}, {:.4f}, {:.4f}".format(*self.origin)
        return f"{self.__class__.__name__}{{nsc: {self.nsc},\n origin={origin},\n {s}\n}}"

    def __repr__(self):
        a, b, c, alpha, beta, gamma = map(lambda r: round(r, 4), self.parameters())
        return f"<{self.__module__}.{self.__class__.__name__} a={a}, b={b}, c={c}, α={alpha}, β={beta}, γ={gamma}, nsc={self.nsc}>"

    def __eq__(self, other):
        """ Equality check """
        return self.equal(other)

    def __ne__(self, b):
        """ In-equality check """
        return not (self == b)

    # Create pickling routines
    def __getstate__(self):
        """ Returns the state of this object """
        return {'cell': self.cell, 'nsc': self.nsc, 'sc_off': self.sc_off, 'origin': self.origin}

    def __setstate__(self, d):
        """ Re-create the state of this object """
        self.__init__(d['cell'], d['nsc'], d['origin'])
        self.sc_off = d['sc_off']

    def __plot__(self, axis=None, axes=False, *args, **kwargs):
        """ Plot the supercell in a specified ``matplotlib.Axes`` object.

        Parameters
        ----------
        axis : array_like, optional
           only plot a subset of the axis, defaults to all axis
        axes : bool or matplotlib.Axes, optional
           the figure axes to plot in (if ``matplotlib.Axes`` object).
           If ``True`` it will create a new figure to plot in.
           If ``False`` it will try and grap the current figure and the current axes.
        """
        # Default dictionary for passing to newly created figures
        d = dict()

        # Try and default the color and alpha
        if 'color' not in kwargs and len(args) == 0:
            kwargs['color'] = 'k'
        if 'alpha' not in kwargs:
            kwargs['alpha'] = 0.5

        if axis is None:
            axis = [0, 1, 2]

        # Ensure we have a new 3D Axes3D
        if len(axis) == 3:
            d['projection'] = '3d'

        axes = plt.get_axes(axes, **d)

        # Create vector objects
        o = self.origin
        v = []
        for a in axis:
            v.append(np.vstack((o[axis], o[axis] + self.cell[a, axis])))
        v = np.array(v)

        if axes.__class__.__name__.startswith('Axes3D'):
            # We should plot in 3D plots
            for vv in v:
                axes.plot(vv[:, 0], vv[:, 1], vv[:, 2], *args, **kwargs)

            v0, v1 = v[0], v[1] - o
            axes.plot(v0[1, 0] + v1[:, 0], v0[1, 1] + v1[:, 1], v0[1, 2] + v1[:, 2], *args, **kwargs)

            axes.set_zlabel('Ang')

        else:
            for vv in v:
                axes.plot(vv[:, 0], vv[:, 1], *args, **kwargs)

            v0, v1 = v[0], v[1] - o[axis]
            axes.plot(v0[1, 0] + v1[:, 0], v0[1, 1] + v1[:, 1], *args, **kwargs)
            axes.plot(v1[1, 0] + v0[:, 0], v1[1, 1] + v0[:, 1], *args, **kwargs)

        axes.set_xlabel('Ang')
        axes.set_ylabel('Ang')

        return axes


# same reference
class SuperCell(Lattice):
    """ Deprecated class, please use `Lattice` instead """
    def __init__(self, *args, **kwargs):
        deprecate(f"{self.__class__.__name__} is deprecated; please use 'Lattice' class instead", "0.15")
        super().__init__(*args, **kwargs)


class LatticeChild:
    """ Class to be inherited by using the ``self.lattice`` as a `Lattice` object

    Initialize by a `Lattice` object and get access to several different
    routines directly related to the `Lattice` class.
    """

    @property
    def sc(self):
        """ Returns the lattice object (deprecated method) """
        deprecate(f"{self.__class__.__name__}.sc is deprecated; please use 'lattice' instead", "0.15")
        return self.lattice

    def set_nsc(self, *args, **kwargs):
        """ Set the number of super-cells in the `Lattice` object

        See `set_nsc` for allowed parameters.

        See Also
        --------
        Lattice.set_nsc : the underlying called method
        """
        self.lattice.set_nsc(*args, **kwargs)

    def set_lattice(self, lattice):
        """ Overwrites the local supercell """
        if lattice is None:
            # Default supercell is a simple
            # 1x1x1 unit-cell
            self.lattice = Lattice([1., 1., 1.])
        elif isinstance(lattice, Lattice):
            self.lattice = lattice
        elif isinstance(lattice, LatticeChild):
            self.lattice = lattice.lattice
        else:
            # The supercell is given as a cell
            self.lattice = Lattice(lattice)

    set_sc = deprecation("set_sc is deprecated; please use set_lattice instead", "0.14")(set_lattice)
    set_supercell = deprecation("set_sc is deprecated; please use set_lattice instead", "0.15")(set_lattice)

    @property
    def length(self):
        """ Returns the inherent `Lattice` objects `length` """
        return self.lattice.length

    @property
    def volume(self):
        """ Returns the inherent `Lattice` objects `volume` """
        return self.lattice.volume

    def area(self, ax0, ax1):
        """ Calculate the area spanned by the two axis `ax0` and `ax1` """
        return self.lattice.area(ax0, ax1)

    @property
    def cell(self):
        """ Returns the inherent `Lattice` objects `cell` """
        return self.lattice.cell

    @property
    def icell(self):
        """ Returns the inherent `Lattice` objects `icell` """
        return self.lattice.icell

    @property
    def rcell(self):
        """ Returns the inherent `Lattice` objects `rcell` """
        return self.lattice.rcell

    @property
    def origin(self):
        """ Returns the inherent `Lattice` objects `origin` """
        return self.lattice.origin

    @property
    def n_s(self):
        """ Returns the inherent `Lattice` objects `n_s` """
        return self.lattice.n_s

    @property
    def nsc(self):
        """ Returns the inherent `Lattice` objects `nsc` """
        return self.lattice.nsc

    @property
    def sc_off(self):
        """ Returns the inherent `Lattice` objects `sc_off` """
        return self.lattice.sc_off

    @property
    def isc_off(self):
        """ Returns the inherent `Lattice` objects `isc_off` """
        return self.lattice.isc_off

    def sc_index(self, *args, **kwargs):
        """ Call local `Lattice` object `sc_index` function """
        return self.lattice.sc_index(*args, **kwargs)

