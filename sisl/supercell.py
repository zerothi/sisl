""" Define a supercell

This class is the basis of many different objects.
"""
from __future__ import print_function, division

import math
from numbers import Integral
import numpy as np
from numpy import dot

from sisl.utils.mathematics import fnorm
import sisl._array as _a
import sisl._plot as plt
from .quaternion import Quaternion

__all__ = ['SuperCell', 'SuperCellChild']


def _cross(u, v):
    """ Cross product u x v """
    return np.array([u[1] * v[2] - u[2] * v[1],
                     u[2] * v[0] - u[0] * v[2],
                     u[0] * v[1] - u[1] * v[0]])


def _dot(u, v):
    """ Dot product u . v """
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]


class SuperCell(object):
    """ Object to retain a super-cell and its nested values.

    This supercell object handles cell vectors and its supercell mirrors.
    """

    # We limit the scope of this SuperCell object.
    __slots__ = ('cell', 'origo', 'volume', 'nsc', 'n_s', '_sc_off', '_isc_off')

    def __init__(self, cell, nsc=None, origo=None):
        """ Initialize a `SuperCell` object from initial quantities

        Initialize a `SuperCell` object with cell information
        and number of supercells in each direction.
        """
        if nsc is None:
            nsc = [1, 1, 1]

        # If the length of cell is 6 it must be cell-parameters, not
        # actual cell coordinates
        self.cell = self.tocell(cell)

        if origo is None:
            self.origo = _a.zerosd(3)
        else:
            self.origo = _a.arrayd(origo)
            if self.origo.size != 3:
                raise ValueError("Origo *must* be 3 numbers.")

        # Set the volume
        self._update_vol()

        self.nsc = _a.onesi(3)
        # Set the super-cell
        self.set_nsc(nsc=nsc)

    def parameters(self, rad=False, sort_norm=False):
        r""" Return the cell-parameters of this cell. I.e. lattice vector lengths and angles

        Parameters
        ----------
        rad : bool, optional
           whether the angles are returned in radians (otherwise in degree)
        sort_norm : bool, optional
           if True the cell parameters are calculated as though the lattice vectors
           are swapped to have the largest Cartesian :math:`x` component as the first
           lattice vector, the second lattice vector has the largest :math:`y` component.
        """
        if rad:
            f = 1.
        else:
            f = 180 / np.pi

        # Figure out which vectors has the largest component along Cartesian x
        cell = self.cell.copy()
        abc = fnorm(cell)
        if sort_norm:
            x_frac = np.abs(cell[:, 0]) / abc
            y_frac = np.abs(cell[:, 1]) / abc
            # Figure out which has the largest component
            ix = np.argmax(x_frac)
            y_frac[ix] = -1.
            # Figure out which has the largest component
            iy = np.argmax(y_frac)
            # The last component has to be this:
            iz = 3 - ix - iy
        else:
            ix = 0
            iy = 1
            iz = 2

        # Now we have the largest x-component.
        # Rotate 'ix' vector to be along x vector
        from math import acos
        cell = cell / abc.reshape(-1, 1)
        alpha = acos(_dot(cell[iy, :], cell[iz, :])) * f
        beta = acos(_dot(cell[ix, :], cell[iz, :])) * f
        gamma = acos(_dot(cell[ix, :], cell[iy, :])) * f

        return abc[ix], abc[iy], abc[iz], alpha, beta, gamma

    def _update_vol(self):
        self.volume = np.abs(dot(self.cell[0, :],
                                 np.cross(self.cell[1, :], self.cell[2, :])
        ))

    def _fill(self, non_filled, dtype=None):
        """ Return a zero filled array of length 3 """

        if len(non_filled) == 3:
            return non_filled

        # Fill in zeros
        # This will purposefully raise an exception
        # if the dimensions of the periodic ones
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

        nsc: list of int, optional
           number of supercells in each direction
        a: integer, optional
           number of supercells in the first unit-cell vector direction
        b: integer, optional
           number of supercells in the second unit-cell vector direction
        c: integer, optional
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
    def sc_off(self):
        """ Integer supercell offsets """
        return self._sc_off

    @sc_off.setter
    def sc_off(self, sc_off):
        """ Set the supercell offset """
        self._sc_off[:, :] = _a.arrayi(sc_off, order='C')
        self._update_isc_off()

    @property
    def isc_off(self):
        """ Internal indexed supercell ``[ia, ib, ic] == i`` """
        return self._isc_off

    # Aliases
    set_supercell = set_nsc

    def __iter__(self):
        """ Iterate the supercells and the indices of the supercells """
        for i, sc in enumerate(self.sc_off):
            yield i, sc

    def copy(self, cell=None, origo=None):
        """ Returns a copy of the object with possibly some quantities overwritten

        Parameters
        ----------
        cell : array_like
           the new cell parameters
        origo : array_like
           the new origo
        """
        if origo is None:
            origo = self.origo.copy()
        if cell is None:
            copy = self.__class__(np.copy(self.cell), nsc=np.copy(self.nsc), origo=origo)
        else:
            copy = self.__class__(np.copy(cell), nsc=np.copy(self.nsc), origo=origo)
        # Ensure that the correct super-cell information gets carried through
        if not np.all(copy.sc_off == self.sc_off):
            copy.sc_off = self.sc_off
        return copy

    def fit(self, xyz, axis=None, tol=0.05):
        """ Fit the supercell to `xyz` such that the unit-cell becomes periodic in the specified directions

        The fitted supercell tries to determine the unit-cell parameters by solving a set of linear equations
        corresponding to the current supercell vectors.

        >>> numpy.linalg.solve(self.cell.T, xyz.T) # doctest: +SKIP

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
        x = dot(xyz, self.rcell.T / (2 * np.pi))

        # Now we should figure out the correct repetitions
        # by rounding to integer positions of the cell vectors
        ix = np.rint(x)

        # Figure out the displacements from integers
        # Then reduce search space by removing those coordinates
        # that are more than the tolerance.
        dist = np.sqrt((dot(cell.T, (x - ix).T) ** 2).sum(0))
        idx = (dist <= tol).nonzero()[0]
        if len(idx) == 0:
            raise ValueError(('Could not fit the cell parameters to the coordinates '
                              'due to insufficient accuracy (try increase the tolerance)'))

        # Reduce problem to allowed values below the tolerance
        x = x[idx, :]
        ix = ix[idx, :]

        # Reduce to total repetitions
        ireps = np.amax(ix, axis=0) - np.amin(ix, axis=0) + 1

        # Only repeat the axis requested
        if isinstance(axis, Integral):
            axis = [axis]

        # Reduce the non-set axis
        if not axis is None:
            for ax in [0, 1, 2]:
                if ax not in axis:
                    ireps[ax] = 1

        # Enlarge the cell vectors
        cell[0, :] *= ireps[0]
        cell[1, :] *= ireps[1]
        cell[2, :] *= ireps[2]

        return self.copy(cell)

    def swapaxes(self, a, b):
        """ Returns `SuperCell` with swapped axis

        If ``swapaxes(0,1)`` it returns the 0 in the 1 values.
        """
        # Create index vector
        idx = _a.arrayi([0, 1, 2])
        idx[b] = a
        idx[a] = b
        # There _can_ be errors when sc_off isn't created by sisl
        return self.__class__(np.copy(self.cell[idx, :], order='C'),
                              nsc=self.nsc[idx],
                              origo=np.copy(self.origo[idx], order='C'))

    def plane(self, ax1, ax2, origo=True):
        """ Query point and plane-normal for the plane spanning `ax1` and `ax2`

        Parameters
        ----------
        ax1 : int
           the first axis vector
        ax2 : int
           the second axis vector
        origo : bool, optional
           whether the plane intersects the origo or the opposite corner of the
           unit-cell.

        Returns
        -------
        n : array_like
           planes normal vector (pointing outwards with regards to the cell)
        p : array_like
           a point on the plane

        Examples
        --------

        All 6 faces of the supercell can be retrieved like this:

        >>> n1, p1 = self.plane(0, 1, True) # doctest: +SKIP
        >>> n2, p2 = self.plane(0, 1, False) # doctest: +SKIP
        >>> n3, p3 = self.plane(0, 2, True) # doctest: +SKIP
        >>> n4, p4 = self.plane(0, 2, False) # doctest: +SKIP
        >>> n5, p5 = self.plane(1, 2, True) # doctest: +SKIP
        >>> n6, p6 = self.plane(1, 2, False) # doctest: +SKIP

        However, for performance critical calculations it may be advantageous to
        do this:

        >>> uc = self.cell.sum(0) # doctest: +SKIP
        >>> n1, p1 = self.sc.plane(0, 1) # doctest: +SKIP
        >>> n2 = -n1 # doctest: +SKIP
        >>> p2 = p1 + uc # doctest: +SKIP
        >>> n3, p3 = self.sc.plane(0, 2) # doctest: +SKIP
        >>> n4 = -n3 # doctest: +SKIP
        >>> p4 = p3 + uc # doctest: +SKIP
        >>> n5, p5 = self.sc.plane(1, 2) # doctest: +SKIP
        >>> n6 = -n5 # doctest: +SKIP
        >>> p6 = p5 + uc # doctest: +SKIP

        Secondly, the variables ``p1``, ``p3`` and ``p5`` are always ``[0, 0, 0]`` and
        ``p2``, ``p4`` and ``p6`` are always ``uc``.
        Hence this may be used to further reduce certain computations.
        """
        cell = self.cell
        n = _cross(cell[ax1, :], cell[ax2, :])
        # Normalize
        n /= _dot(n, n) ** .5
        # Now we need to figure out if the normal vector
        # is pointing outwards
        # Take the cell center
        up = cell.sum(0)
        # Calculate the distance from the plane to the center of the cell

        # If d is positive then the normal vector is pointing towards
        # the center, so rotate 180
        if _dot(n, up / 2) > 0.:
            n *= -1

        if origo:
            return n, _a.zerosd([3])
        # We have to reverse the normal vector
        return -n, up

    @property
    def rcell(self):
        """ Returns the reciprocal cell for the `SuperCell` without ``2*np.pi``

        Note: The returned vectors are still in [0, :] format
        and not as returned by an inverse LAPACK algorithm.
        """
        # Calculate the reciprocal cell
        # This should probably be changed and checked for
        # transposition
        cell = self.cell
        rcell = np.empty([3, 3], dtype=cell.dtype)
        rcell[0, 0] = cell[1, 1] * cell[2, 2] - cell[1, 2] * cell[2, 1]
        rcell[0, 1] = cell[1, 2] * cell[2, 0] - cell[1, 0] * cell[2, 2]
        rcell[0, 2] = cell[1, 0] * cell[2, 1] - cell[1, 1] * cell[2, 0]
        rcell[1, 0] = cell[2, 1] * cell[0, 2] - cell[2, 2] * cell[0, 1]
        rcell[1, 1] = cell[2, 2] * cell[0, 0] - cell[2, 0] * cell[0, 2]
        rcell[1, 2] = cell[2, 0] * cell[0, 1] - cell[2, 1] * cell[0, 0]
        rcell[2, 0] = cell[0, 1] * cell[1, 2] - cell[0, 2] * cell[1, 1]
        rcell[2, 1] = cell[0, 2] * cell[1, 0] - cell[0, 0] * cell[1, 2]
        rcell[2, 2] = cell[0, 0] * cell[1, 1] - cell[0, 1] * cell[1, 0]
        rcell[0, :] = rcell[0, :] / dot(rcell[0, :], cell[0, :])
        rcell[1, :] = rcell[1, :] / dot(rcell[1, :], cell[1, :])
        rcell[2, :] = rcell[2, :] / dot(rcell[2, :], cell[2, :])
        return rcell * 2. * np.pi

    def rotatea(self, angle, only='abc', rad=False):
        return self.rotate(angle, self.cell[0, :], only=only, rad=rad)

    def rotateb(self, angle, only='abc', rad=False):
        return self.rotate(angle, self.cell[1, :], only=only, rad=rad)

    def rotatec(self, angle, only='abc', rad=False):
        return self.rotate(angle, self.cell[2, :], only=only, rad=rad)

    def rotate(self, angle, v, only='abc', rad=False):
        """ Rotates the supercell, in-place by the angle around the vector

        One can control which cell vectors are rotated by designating them
        individually with ``only='[abc]'``.

        Parameters
        ----------
        angle : float
             the angle of which the geometry should be rotated
        v     : array_like [3]
             the vector around the rotation is going to happen
             v = [1,0,0] will rotate in the ``yz`` plane
        rad : bool, optional
             Whether the angle is in radians (True) or in degrees (False)
        only : ('abc'), str, optional
             only rotate the designated cell vectors.
        """
        # flatte => copy
        vn = np.asarray(v, dtype=np.float64).flatten()
        vn /= fnorm(vn)
        q = Quaternion(angle, vn, rad=rad)
        q /= q.norm()  # normalize the quaternion
        cell = np.copy(self.cell)
        if 'a' in only:
            cell[0, :] = q.rotate(self.cell[0, :])
        if 'b' in only:
            cell[1, :] = q.rotate(self.cell[1, :])
        if 'c' in only:
            cell[2, :] = q.rotate(self.cell[2, :])
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
        other : SuperCell, array_like
           the lattice vectors of the other supercell to add
        """
        if not isinstance(other, SuperCell):
            other = self.tocell(other)
        cell = self.cell + other.cell
        origo = self.origo + other.origo
        nsc = np.where(self.nsc > other.nsc, self.nsc, other.nsc)
        return self.__class__(cell, nsc=nsc, origo=origo)

    def __add__(self, other):
        return self.add(other)

    __radd__ = __add__

    def add_vacuum(self, vacuum, axis):
        """ Add vacuum along the `axis` lattice vector

        Parameters
        ----------
        vacuum : float
           amount of vacuum added, in Ang
        axis : int
           the lattice vector to add vacuum along
        """
        cell = np.copy(self.cell)
        d = cell[axis, :].copy()
        # normalize to get direction vector
        cell[axis, :] += d * (vacuum / fnorm(d))
        return self.copy(cell)

    def sc_index(self, sc_off):
        """ Returns the integer index in the sc_off list that corresponds to `sc_off`

        Returns the integer for the supercell
        """
        def _assert(m, v):
            if np.all(np.abs(v) > m):
                raise ValueError("Requesting a non-existing supercell index")
        hsc = self.nsc // 2

        if isinstance(sc_off[0], np.ndarray):
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
        idx = []
        for i in range(self.n_s):
            if (sc_off[0] == self.sc_off[i, 0] or sc_off[0] is None) and \
               (sc_off[1] == self.sc_off[i, 1] or sc_off[1] is None) and \
               (sc_off[2] == self.sc_off[i, 2] or sc_off[2] is None):
                idx.append(i)
        return idx

    def scale(self, scale):
        """ Scale lattice vectors

        Does not scale `origo`.

        Parameters
        ----------
        scale : ``float``
           the scale factor for the new lattice vectors
        """
        return self.copy(self.cell * scale)

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
        origo = np.copy(self.origo)
        cell[axis, :] *= reps
        # Only reduce the size if it is larger than 5
        if nsc[axis] > 3 and reps > 1:
            nsc[axis] = max(1, nsc[axis] // 2 - (reps - 1)) * 2 + 1
        return self.__class__(cell, nsc=nsc, origo=origo)

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

    def cut(self, seps, axis):
        """ Cuts the cell into several different sections. """
        cell = np.copy(self.cell)
        cell[axis, :] /= seps
        return self.copy(cell)

    def append(self, other, axis):
        """ Appends other `SuperCell` to this grid along axis """
        cell = np.copy(self.cell)
        cell[axis, :] += other.cell[axis, :]
        return self.copy(cell)

    def prepend(self, other, axis):
        """ Prepends other `SuperCell` to this grid along axis

        For a `SuperCell` object this is equivalent to `append`.
        """
        return self.append(other, axis)

    def move(self, v):
        """ Appends additional space in the SuperCell object """
        # check which cell vector resembles v the most,
        # use that
        cell = np.copy(self.cell)
        p = np.empty([3], np.float64)
        cl = fnorm(cell)
        for i in range(3):
            p[i] = abs(np.sum(cell[i, :] * v)) / cl[i]
        cell[np.argmax(p), :] += v
        return self.copy(cell)
    translate = move

    def center(self, axis=None):
        """ Returns center of the `SuperCell`, possibly with respect to an axis """
        if axis is None:
            return np.sum(self.cell, axis=0) / 2
        return self.cell[axis, :] / 2

    @classmethod
    def tocell(cls, *args):
        """ Returns a 3x3 unit-cell dependent on the input

        If you supply a single argument it is regarded as either
        a) a proper unit-cell
        b) the diagonal elements in the unit-cell

        If you supply 3 arguments it will be the same as the
        diagonal elements of the unit-cell

        If you supply 6 arguments it will be the same as the
        cell parameters, a, b, c, alpha, beta, gamma.
        The angles should be provided in degree (not radians).
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

            from math import sqrt, cos, sin, pi
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

    def is_orthogonal(self):
        """ Returns true if the cell vectors are orthogonal """
        # Convert to unit-vector cell
        cell = np.copy(self.cell)
        cl = fnorm(cell)
        cell[0, :] = cell[0, :] / cl[0]
        cell[1, :] = cell[1, :] / cl[1]
        cell[2, :] = cell[2, :] / cl[2]
        i_s = dot(cell[0, :], cell[1, :]) < 0.001
        i_s = dot(cell[0, :], cell[2, :]) < 0.001 and i_s
        i_s = dot(cell[1, :], cell[2, :]) < 0.001 and i_s
        return i_s

    def parallel(self, other, axis=(0, 1, 2)):
        """ Returns true if the cell vectors are parallel to `other`

        Parameters
        ----------
        other : SuperCell
           the other object to check whether the axis are parallel
        axis : int or array_like
           only check the specified axis (default to all)
        """
        axis = _a.asarrayi(axis).ravel()
        # Convert to unit-vector cell
        for i in axis:
            a = self.cell[i, :] / fnorm(self.cell[i, :])
            b = other.cell[i, :] / fnorm(other.cell[i, :])
            if abs(dot(a, b) - 1) > 0.001:
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
        ang = math.acos(dot(self.cell[i, :], self.cell[j, :]) / (n[0] * n[1]))
        if rad:
            return ang
        return math.degrees(ang)

    @staticmethod
    def read(sile, *args, **kwargs):
        """ Reads SuperCell from the `Sile` using `Sile.read_supercell`

        Parameters
        ----------
        sile : `Sile` or str
            a `Sile` object which will be used to read the supercell
            if it is a string it will create a new sile using `sisl.io.get_sile`.
        """
        # This only works because, they *must*
        # have been imported previously
        from sisl.io import get_sile, BaseSile
        if isinstance(sile, BaseSile):
            return sile.read_supercell(*args, **kwargs)
        else:
            with get_sile(sile) as fh:
                return fh.read_supercell(*args, **kwargs)

    def __repr__(self):
        """ Returns a string representation of the object """
        return self.__class__.__name__ + '{{volume: {:.4e}, nsc: {} {} {}}}'.format(self.volume, *self.nsc)

    def __eq__(a, b):
        """ Equality check """
        if not isinstance(b, SuperCell):
            return False
        same = np.allclose(a.cell, b.cell)
        same = same and np.all(a.nsc == b.nsc)
        same = same and np.allclose(a.origo, b.origo)
        return same

    def __ne__(a, b):
        """ In-equality check """
        return not (a == b)

    # Create pickling routines
    def __getstate__(self):
        """ Returns the state of this object """
        return {'cell': self.cell, 'nsc': self.nsc, 'sc_off': self.sc_off, 'origo': self.origo}

    def __setstate__(self, d):
        """ Re-create the state of this object """
        self.__init__(d['cell'], d['nsc'], d['origo'])
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

        if axis is None:
            axis = [0, 1, 2]

        # Ensure we have a new 3D Axes3D
        if len(axis) == 3:
            d['projection'] = '3d'

        if axes is False:
            try:
                axes = plt.mlibplt.gca()
            except:
                axes = plt.mlibplt.figure().add_subplot(111, **d)
        elif axes is True:
            axes = plt.mlibplt.figure().add_subplot(111, **d)

        # Create vector objects
        v = []
        for a in axis:
            v.append(np.vstack(([0.]*len(axis), self.cell[a, axis])))
        v = np.array(v)

        if isinstance(axes, plt.mlib3d.Axes3D):
            # We should plot in 3D plots
            for vv in v:
                axes.plot(vv[:, 0], vv[:, 1], vv[:, 2], *args, **kwargs)

            v0, v1 = v[0], v[1]
            axes.plot(v0[1, 0] + v1[:, 0], v0[1, 1] + v1[:, 1], v0[1, 2] + v1[:, 2], *args, **kwargs)

            axes.set_zlabel('Ang')

        else:
            for vv in v:
                axes.plot(vv[:, 0], vv[:, 1], *args, **kwargs)

            v0, v1 = v[0], v[1]
            axes.plot(v0[1, 0] + v1[:, 0], v0[1, 1] + v1[:, 1], *args, **kwargs)
            axes.plot(v1[1, 0] + v0[:, 0], v1[1, 1] + v0[:, 1], *args, **kwargs)

        axes.set_xlabel('Ang')
        axes.set_ylabel('Ang')


class SuperCellChild(object):
    """ Class to be inherited by using the ``self.sc`` as a `SuperCell` object

    Initialize by a `SuperCell` object and get access to several different
    routines directly related to the `SuperCell` class.
    """

    def set_nsc(self, *args, **kwargs):
        """ Set the number of super-cells in the `SuperCell` object

        See `set_nsc` for allowed parameters.

        See Also
        --------
        SuperCell.set_nsc : the underlying called method
        """
        self.sc.set_nsc(*args, **kwargs)

    def set_supercell(self, sc):
        """ Overwrites the local supercell """
        if sc is None:
            # Default supercell is a simple
            # 1x1x1 unit-cell
            self.sc = SuperCell([1., 1., 1.])
        elif isinstance(sc, SuperCell):
            self.sc = sc
        elif isinstance(sc, SuperCellChild):
            self.sc = sc.sc
        else:
            # The supercell is given as a cell
            self.sc = SuperCell(sc)

        # Loop over attributes in this class
        # if it inherits SuperCellChild, we call
        # set_sc on that too.
        # Sadly, getattr fails for @property methods
        # which forces us to use try ... except
        for a in dir(self):
            try:
                if isinstance(getattr(self, a), SuperCellChild):
                    getattr(self, a).set_supercell(self.sc)
            except:
                pass

    set_sc = set_supercell

    @property
    def volume(self):
        """ Returns the inherent `SuperCell` objects `vol` """
        return self.sc.volume

    @property
    def cell(self):
        """ Returns the inherent `SuperCell` objects `cell` """
        return self.sc.cell

    @property
    def rcell(self):
        """ Returns the inherent `SuperCell` objects `rcell` """
        return self.sc.rcell

    @property
    def origo(self):
        """ Returns the inherent `SuperCell` objects `origo` """
        return self.sc.origo

    @property
    def n_s(self):
        """ Returns the inherent `SuperCell` objects `n_s` """
        return self.sc.n_s

    @property
    def nsc(self):
        """ Returns the inherent `SuperCell` objects `nsc` """
        return self.sc.nsc

    @property
    def sc_off(self):
        """ Returns the inherent `SuperCell` objects `sc_off` """
        return self.sc.sc_off

    @property
    def isc_off(self):
        """ Returns the inherent `SuperCell` objects `isc_off` """
        return self.sc.isc_off

    def add_vacuum(self, vacuum, axis):
        """ Add vacuum along the `axis` lattice vector

        Parameters
        ----------
        vacuum : float
           amount of vacuum added, in Ang
        axis : int
           the lattice vector to add vacuum along
        """
        copy = self.copy()
        copy.set_supercell(self.sc.add_vacuum(vacuum, axis))
        return copy

    def _fill(self, non_filled, dtype=None):
        return self.sc._fill(non_filled, dtype)

    def _fill_sc(self, supercell_index):
        return self.sc._fill_sc(supercell_index)

    def sc_index(self, *args, **kwargs):
        """ Call local `SuperCell` object `sc_index` function """
        return self.sc.sc_index(*args, **kwargs)

    def is_orthogonal(self):
        """ Return true if all cell vectors are linearly independent"""
        return self.sc.is_orthogonal()
