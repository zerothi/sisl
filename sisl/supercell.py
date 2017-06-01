""" Define a supercell

This class is the basis of many different objects.
"""
from __future__ import print_function, division

import numpy as np

from .quaternion import Quaternion

__all__ = ['SuperCell', 'SuperCellChild']


class SuperCell(object):
    """ Object to retain a super-cell and its nested values.

    This supercell object handles cell vectors and its supercell mirrors.
    """

    # We limit the scope of this SuperCell object.
    __slots__ = ('cell', 'vol', 'nsc', 'n_s', 'sc_off')

    def __init__(self, cell, nsc=None):
        """ Initialize a `SuperCell` object from initial quantities

        Initialize a `SuperCell` object with cell information
        and number of supercells in each direction.
        """
        if nsc is None:
            nsc = [1, 1, 1]

        # If the length of cell is 6 it must be cell-parameters, not
        # actual cell coordinates
        self.cell = self.tocell(cell)

        # Set the volume
        self._update_vol()

        # Set the super-cell
        self.set_nsc(nsc=nsc)

    def _update_vol(self):
        self.vol = np.abs(np.dot(self.cell[0, :],
                                 np.cross(self.cell[1, :], self.cell[2, :])
                             )
                      )

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
            except:
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

        nsc: [3], integer, optional
           number of supercells in each direction
        a: integer, optional
           number of supercells in the first unit-cell vector direction
        b: integer, optional
           number of supercells in the second unit-cell vector direction
        c: integer, optional
           number of supercells in the third unit-cell vector direction
        """
        if not nsc is None:
            self.nsc = np.asarray(nsc, np.int32)
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
        self.n_s = np.prod(self.nsc)
        self.sc_off = np.zeros([self.n_s, 3], np.int32)

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
                    self.sc_off[i, 0] = ix
                    self.sc_off[i, 1] = iy
                    self.sc_off[i, 2] = iz

    # Aliases
    set_supercell = set_nsc

    def __iter__(self):
        """ Iterate the supercells and the indices of the supercells """
        for i, sc in enumerate(self.sc_off):
            yield i, sc

    def copy(self):
        """
        Returns a copy of the object.
        """
        return self.__class__(np.copy(self.cell), nsc=np.copy(self.nsc))

    def swapaxes(self, a, b):
        """ Returns `SuperCell` with swapped axis

        If ``swapaxes(0,1)`` it returns the 0 in the 1 values.
        """
        # Create index vector
        idx = np.arange(3)
        idx[b] = a
        idx[a] = b
        return self.__class__(np.copy(self.cell[idx, :], order='C'),
                              nsc=self.nsc[idx])

    @property
    def rcell(self):
        """ Returns the reciprocal cell for the `SuperCell` without `2*np.pi`

        Note: The returned vectors are still in [0,:] format
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
        dot = np.dot
        rcell[0, :] = rcell[0, :] / dot(rcell[0, :], cell[0, :])
        rcell[1, :] = rcell[1, :] / dot(rcell[1, :], cell[1, :])
        rcell[2, :] = rcell[2, :] / dot(rcell[2, :], cell[2, :])
        return rcell * 2. * np.pi

    def rotatea(self, angle, only='abc', radians=False):
        return self.rotate(angle, self.cell[0, :], only=only, radians=radians)

    def rotateb(self, angle, only='abc', radians=False):
        return self.rotate(angle, self.cell[1, :], only=only, radians=radians)

    def rotatec(self, angle, only='abc', radians=False):
        return self.rotate(angle, self.cell[2, :], only=only, radians=radians)

    def rotate(self, angle, v, only='abc', radians=False):
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
        radians : bool, False
             Whether the angle is in radians (True) or in degrees (False)
        only : ('abc'), str, optional
             only rotate the designated cell vectors.
        """
        vn = np.copy(np.asarray(v, dtype=np.float64)[:])
        vn /= np.sum(vn ** 2) ** .5
        q = Quaternion(angle, vn, radians=radians)
        q /= q.norm()  # normalize the quaternion
        cell = np.copy(self.cell)
        if 'a' in only:
            cell[0, :] = q.rotate(self.cell[0, :])
        if 'b' in only:
            cell[1, :] = q.rotate(self.cell[1, :])
        if 'c' in only:
            cell[2, :] = q.rotate(self.cell[2, :])
        return self.__class__(cell, nsc=np.copy(self.nsc))

    def offset(self, isc=None):
        """ Returns the supercell offset of the supercell index """
        if isc is None:
            return np.array([0, 0, 0], np.float64)
        return np.dot(isc, self.cell)

    def add_vacuum(self, vacuum, axis):
        """ Add vacuum along the `axis` lattice vector

        Parameters
        ----------
        vacuum : float
           amount of vacuum added, in Ang
        axis : int
           the lattice vector to add vacuum along
        """
        d = self.cell[axis, :]
        # normalize to get direction vector
        d = d / np.sum(d ** 2) ** .5
        self.cell[axis, :] += d * vacuum
        self._update_vol()

    def sc_index(self, sc_off):
        """ Returns the integer index in the sc_off list that corresponds to `sc_off`

        Returns the integer for the supercell
        """
        sc_off = self._fill_sc(sc_off)
        if sc_off[0] is not None and sc_off[1] is not None and sc_off[2] is not None:
            for i in range(self.n_s):
                if (sc_off[0] == self.sc_off[i, 0] or sc_off[0] is None) and \
                   (sc_off[1] == self.sc_off[i, 1] or sc_off[1] is None) and \
                   (sc_off[2] == self.sc_off[i, 2] or sc_off[2] is None):
                    return i
            raise Exception(
                'Could not find supercell index, number of super-cells not big enough')
        idx = []
        for i in range(self.n_s):
            if (sc_off[0] == self.sc_off[i, 0] or sc_off[0] is None) and \
               (sc_off[1] == self.sc_off[i, 1] or sc_off[1] is None) and \
               (sc_off[2] == self.sc_off[i, 2] or sc_off[2] is None):
                idx.append(i)
        return idx

    def scale(self, scale):
        """ Scale lattice vectors

        Parameters
        ----------
        scale : ``float``
           the scale factor for the new lattice vectors
        """
        cell = self.cell * scale
        return self.__class__(cell, np.copy(self.nsc))

    def cut(self, seps, axis):
        """ Cuts the cell into several different sections. """
        cell = np.copy(self.cell)
        cell[axis, :] /= seps
        return self.__class__(cell, np.copy(self.nsc))

    def append(self, other, axis):
        """ Appends other `SuperCell` to this grid along axis """
        cell = np.copy(self.cell)
        cell[axis, :] += other.cell[axis, :]
        return self.__class__(cell, nsc=np.copy(self.nsc))

    def prepend(self, other, axis):
        """ Prepends other `SuperCell` to this grid along axis 

        For a `SuperCell` object this is equivalent to `append`.
        """
        cell = np.copy(self.cell)
        cell[axis, :] += other.cell[axis, :]
        return self.__class__(cell, nsc=np.copy(self.nsc))

    def move(self, v):
        """ Appends additional space in the SuperCell object """
        # check which cell vector resembles v the most,
        # use that
        cell = np.copy(self.cell)
        p = np.empty([3], np.float64)
        for i in range(3):
            p[i] = abs(np.sum(cell[i, :] * v)) / np.sum(cell[i, :]**2)**.5
        cell[np.argmax(p), :] += v
        return self.__class__(cell, np.copy(self.nsc))
    translate = move

    def center(self, axis=None):
        """ Returns center of the `SuperCell`, possibly with respect to an axis
        """
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
        args = np.asarray(args, np.float64).flatten()
        nargs = len(args)

        # A square-box
        if nargs == 1:
            return np.diag([args[0]] * 3)

        # Diagonal components
        if nargs == 3:
            return np.diag(args)

        # Cell parameters
        if nargs == 6:
            cell = np.zeros([3, 3], np.float64)
            a = args[0]
            b = args[1]
            c = args[2]
            alpha = args[3]
            beta = args[4]
            gamma = args[5]

            cell[0, 0] = a
            g = gamma * np.pi / 180.
            cg = np.cos(g)
            sg = np.sin(g)
            cell[1, 0] = b * cg
            cell[1, 1] = b * sg
            b = beta * np.pi / 180.
            cb = np.cos(b)
            sb = np.sin(b)
            cell[2, 0] = c * cb
            a = alpha * np.pi / 180.
            d = (np.cos(a) - cb * cg) / sg
            cell[2, 1] = c * d
            cell[2, 2] = c * np.sqrt(sb**2 - d**2)
            return cell

        # A complete cell
        if nargs == 9:
            args.shape = (3, 3)
            return np.copy(args)

        raise ValueError(
            "Creating a unit-cell has to have 1, 3 or 6 arguments, please correct.")

    def is_orthogonal(self):
        """ Returns true if the cell vectors are orthogonal """
        # Convert to unit-vector cell
        cell = np.copy(self.cell)
        cell[0, :] = cell[0, :] / np.sum(cell[0, :]**2) ** .5
        cell[1, :] = cell[1, :] / np.sum(cell[1, :]**2) ** .5
        cell[2, :] = cell[2, :] / np.sum(cell[2, :]**2) ** .5
        i_s = np.dot(cell[0, :], cell[1, :]) < 0.001
        i_s = np.dot(cell[0, :], cell[2, :]) < 0.001 and i_s
        i_s = np.dot(cell[1, :], cell[2, :]) < 0.001 and i_s
        return i_s

    @staticmethod
    def read(sile, *args, **kwargs):
        """ Reads SuperCell from the `Sile` using `Sile.read_supercell`

        Parameters
        ----------
        sile : `Sile`, str
            a `Sile` object which will be used to read the supercell
            if it is a string it will create a new sile using `get_sile`.
        """
        # This only works because, they *must*
        # have been imported previously
        from sisl.io import get_sile, BaseSile
        if isinstance(sile, BaseSile):
            return sile.read_supercell(*args, **kwargs)
        else:
            return get_sile(sile).read_supercell(*args, **kwargs)

    def __repr__(self):
        """ Returns a string representation of the object """
        return 'SuperCell[{} {} {}]'.format(*self.nsc)

    def __eq__(a, b):
        """ Equality check """
        if not isinstance(b, SuperCell):
            return False
        same = np.allclose(a.cell, b.cell)
        same = same and np.all(a.nsc == b.nsc)
        return same

    def __ne__(a, b):
        return not (a == b)

    # Create pickling routines
    def __getstate__(self):
        """ Returns the state of this object """
        return {'cell': self.cell, 'nsc': self.nsc}

    def __setstate__(self, d):
        """ Re-create the state of this object """
        self.__init__(d['cell'], d['nsc'])


class SuperCellChild(object):
    """ Class to be inherited by using the `self.sc` as a `SuperCell` object

    Initialize by a `SuperCell` object and get access to several different
    routines directly related to the `SuperCell` class.
    """

    def set_nsc(self, nsc):
        """ Set the number of super-cells in the `SuperCell` object """
        self.sc.set_nsc(nsc)

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
    def vol(self):
        """ Returns the inherent `SuperCell` objects `vol` """
        return self.sc.vol

    @property
    def cell(self):
        """ Returns the inherent `SuperCell` objects `cell` """
        return self.sc.cell

    @property
    def rcell(self):
        """ Returns the inherent `SuperCell` objects `rcell` """
        return self.sc.rcell

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

    def add_vacuum(self, vacuum, axis):
        """ Add vacuum along the `axis` lattice vector

        Parameters
        ----------
        vacuum : float
           amount of vacuum added, in Ang
        axis : int
           the lattice vector to add vacuum along
        """
        self.sc.add_vacuum(vacuum, axis)

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
