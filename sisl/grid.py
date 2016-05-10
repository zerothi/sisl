""" Define a grid

This grid is the basis of different used models.
"""
from __future__ import print_function, division

from numbers import Integral

import numpy as np

from .quaternion import Quaternion
from .supercell import SuperCell, SuperCellChild
from .atom import Atom
from .geometry import Geometry

__all__ = ['Grid']


# Default nsc variable
_nsc = np.array([1] * 3, np.int32)
_size = np.array([0] * 3, np.int32)
_origo = np.array([0] * 3, np.float64)
_bc = 1  # Periodic (Grid.Periodic)
_dtype = np.float64


class Grid(SuperCellChild):
    """ Object to retain grid information

    This grid object handles cell vectors and divisions of said grid.

    A grid can be periodic and non-periodic.
    """

    # Constant (should never be changed)
    Periodic = 1
    Neumann = 2
    Dirichlet = 3

    def __init__(self, size=_size, bc=_bc, sc=None, dtype=_dtype, geom=None):
        """ Initialize a `Grid` object.

        Initialize a `Grid` object.
        """

        self.set_supercell(sc)

        # Create the grid
        self.set_grid(size, dtype=dtype)

        # Create the grid boundary conditions
        self.set_bc(bc)

        # Create the atomic structure in the grid, if possible
        self.set_geom(geom)

        # If the user sets the super-cell, that has precedence.
        if sc is not None:
            self.geom.set_sc(sc)
            self.set_sc(sc)

    def __getitem__(self, key):
        """ Returns the grid contained """
        return self.grid[key]

    def __setitem__(self, key, val):
        """ Updates the grid contained """
        self.grid[key] = val

    def set_geom(self, geom):
        """ Sets the `Geometry` for the grid.

        Setting the `Geometry` for the grid is a possibility
        to attach atoms to the grid.

        It is not a necessary entity.
        """
        if geom is None:
            # Fake geometry
            self.set_geom(Geometry([0, 0, 0], Atom['H'], sc=self.sc))
        else:
            self.geom = geom
            self.set_sc(geom.sc)

    def interp(self, size, *args, **kwargs):
        """ Returns an interpolated version of the grid

        Parameters
        ----------
        size : int, array_like
            the new size of the grid
        *args, **kwargs :
            optional arguments passed to the interpolation algorithm
            The interpolation routine is `scipy.interpolate.interpn`
        """
        # Get grid spacing
        dold = []
        dnew = []
        for i in range(3):
            dold.append(np.linspace(0, 1, self.size[i]))
            dnew.append(np.linspace(0, 1, size[i]))

        # Interpolate
        from scipy.interpolate import interpn

        # Create new grid
        grid = self.__class__(size, bc=np.copy(self.bc), sc=self.sc.copy())
        grid.grid = interpn(dold, self.grid, dnew, *args, **kwargs)

        return grid

    @property
    def size(self):
        """ Returns size of the grid """
        return np.asarray(self.grid.shape, np.int32)

    def set_grid(self, size, dtype=_dtype):
        """ Create the internal grid of certain size.
        """
        self.grid = np.empty(size, dtype=dtype)

    def set_bc(self, boundary=None, a=None, b=None, c=None):
        """ Set the boundary conditions on the grid
        boundary: [3], integer, optional
           boundary condition for all boundaries (or the same for all)
        a: integer, optional
           boundary condition for the first unit-cell vector direction
        b: integer, optional
           boundary condition for the second unit-cell vector direction
        c: integer, optional
           boundary condition for the third unit-cell vector direction
        """
        if not boundary is None:
            if isinstance(boundary, Integral):
                self.bc = np.array([boundary] * 3, np.int32)
            else:
                self.bc = np.asarray(boundary, np.int32)
        if not a is None:
            self.bc[0] = a
        if not b is None:
            self.bc[1] = b
        if not c is None:
            self.bc[2] = c

    # Aliases
    set_boundary = set_bc
    set_boundary_condition = set_bc

    def copy(self):
        """
        Returns a copy of the object.
        """
        grid = self.__class__(np.copy(self.size), bc=np.copy(self.bc),
                              dtype=self.grid.dtype,
                              geom=self.geom.copy())
        grid.grid[:, :, :] = self.grid[:, :, :]
        return grid

    def swapaxes(self, a, b):
        """ Returns Grid with swapped axis

        If ``swapaxes(0,1)`` it returns the 0 in the 1 values.
        """
        # Create index vector
        idx = np.arange(3)
        idx[b] = a
        idx[a] = b
        s = np.copy(self.size)
        grid = self.__class__(s[idx], bc=self.bc[idx],
                              sc=self.sc.swapaxes(a, b), dtype=self.grid.dtype,
                              geom=self.geom.copy())
        # We need to force the C-order or we loose the contiguity
        grid.grid = np.copy(np.swapaxes(self.grid, a, b), order='C')
        return grid

    @property
    def dcell(self):
        """ Returns the delta-cell """
        # Calculate the grid-distribution
        dcell = np.empty([3, 3], np.float64)
        dcell[0, :] = self.cell[0, :] / self.size[0]
        dcell[1, :] = self.cell[1, :] / self.size[1]
        dcell[2, :] = self.cell[2, :] / self.size[2]
        return dcell

    @property
    def dvol(self):
        """ Returns the delta-volume """
        return self.sc.vol / np.prod(self.size)

    def cross_section(self, idx, axis):
        """ Takes a cross-section of the grid along axis ``axis``

        Remark: This API entry might change to handle arbitrary
        cuts via rotation of the axis """

        # First calculate the new size
        size = self.size
        cell = np.copy(self.cell)
        # Down-scale cell
        cell[axis, :] /= size[axis]
        size[axis] = 1
        grid = self.__class__(size, bc=np.copy(self.bc), geom=self.geom.copy())
        # Update cell size (the cell is smaller now)
        grid.set_sc(cell)

        if axis == 0:
            grid.grid[:, :, :] = self.grid[idx, :, :]
        elif axis == 1:
            grid.grid[:, :, :] = self.grid[:, idx, :]
        elif axis == 2:
            grid.grid[:, :, :] = self.grid[:, :, idx]
        else:
            raise ValueError('Unknown axis specification in cross_section')

        return grid

    def sum(self, axis):
        """ Returns the grid summed along axis ``axis``. """
        # First calculate the new size
        size = self.size
        cell = np.copy(self.cell)
        # Down-scale cell
        cell[axis, :] /= size[axis]
        size[axis] = 1

        grid = self.__class__(size, bc=np.copy(self.bc), geom=self.geom.copy())
        # Update cell size (the cell is smaller now)
        grid.set_sc(cell)

        # Calculate sum (retain dimensions)
        grid.grid[:, :, :] = np.sum(self.grid, axis=axis, keepdims=True)
        return grid

    def mean(self, axis):
        """ Returns the average grid along direction ``axis`` """
        n = self.size[axis]
        g = self.sum(axis)
        g /= float(n)
        return g

    # for compatibility
    average = mean

    def remove_part(self, idx, axis, above):
        """ Removes parts of the grid via above/below designations.

        Works exactly opposite to `sub_part`

        Parameters
        ----------
        idx : array_like
           the indices of the grid axis ``axis`` to be removed
           for ``above=True`` grid[:idx,...]
           for ``above=False`` grid[idx:,...]
        axis : int
           the axis segment from which we retain the indices `idx`
        above: bool
           if `True` will retain the grid:
              `grid[:idx,...]`
           else it will retain the grid:
              `grid[idx:,...]`
        """
        return self.sub_part(idx, axis, not above)

    def sub_part(self, idx, axis, above):
        """ Retains parts of the grid via above/below designations.

        Works exactly opposite to `remove_part`

        Parameters
        ----------
        idx : array_like
           the indices of the grid axis ``axis`` to be retained
           for ``above=True`` grid[idx:,...]
           for ``above=False`` grid[:idx,...]
        axis : int
           the axis segment from which we retain the indices ``idx``
        above: bool
           if ``True`` will retain the grid:
              ``grid[idx:,...]``
           else it will retain the grid:
              ``grid[:idx,...]``
        """
        if above:
            sub = np.arange(idx, self.size[axis])
        else:
            sub = np.arange(0, idx)
        return self.sub(sub, axis)

    def sub(self, idx, axis):
        """ Retains certain indices from a specified axis.

        Works exactly opposite to `remove`.

        Parameters
        ----------
        idx : array_like
           the indices of the grid axis ``axis`` to be retained
        axis : int
           the axis segment from which we retain the indices ``idx``
        """
        uidx = np.unique(np.clip(idx, 0, self.size[axis] - 1))

        # Calculate new size
        size = self.size
        cell = np.copy(self.cell)
        old_N = size[axis]

        # Calculate new size
        size[axis] = len(idx)
        if size[axis] < 1:
            raise ValueError('You cannot retain no indices.')

        # Down-scale cell
        cell[axis, :] = cell[axis, :] / old_N * size[axis]

        grid = self.__class__(size, bc=np.copy(self.bc), geom=self.geom.copy())
        # Update cell size (the cell is smaller now)
        grid.set_sc(cell)

        # Remove the indices
        # First create the opposite, index
        if axis == 0:
            grid.grid[:, :, :] = self.grid[idx, :, :]
        elif axis == 1:
            grid.grid[:, :, :] = self.grid[:, idx, :]
        elif axis == 2:
            grid.grid[:, :, :] = self.grid[:, :, idx]

        return grid

    def remove(self, idx, axis):
        """ Removes certain indices from a specified axis.

        Works exactly opposite to `sub`.

        Parameters
        ----------
        idx : array_like
           the indices of the grid axis ``axis`` to be removed
        axis : int
           the axis segment from which we remove all indices ``idx``
        """
        uidx = np.unique(np.clip(idx, 0, self.size[axis] - 1))
        ret_idx = np.setdiff1d(
            np.arange(
                self.size[axis]),
            uidx,
            assume_unique=True)
        return self.sub(ret_idx, axis)

    def index(self, coord, axis=None):
        """ Returns the index along the axis ``axis`` where ``coord`` exists

        Parameters
        ----------
        coord : array_like / float
           the coordinate of the axis
        axis : int
           the axis direction of the index
        """

        # if the axis is none, we do this for all axes
        if axis is None:
            # Loop over each direction
            idx = np.empty([3], np.int32)
            for i in [0, 1, 2]:
                # Normalize cell vector
                ca = self.cell[i, :] / np.sum(self.cell[i, :]**2)**.5
                # get the coordinate along the direction of the cell vector
                c = np.dot(self.cell[i, :], coord) * coord
                # Calculate the index corresponding to this placement
                idx[i] = self.index(c, i)
            return idx

        # Ensure a 1D array
        ac = np.atleast_1d(np.asarray(coord, np.float64))

        # Calculate the index of the coord in the cell
        dax = self.dcell[axis, :]

        # Calculate how many indices are required to fulfil
        # the correct line cut
        return int(np.rint((np.sum(ac ** 2) / np.sum(dax ** 2)) ** .5))

    def append(self, other, axis):
        """ Appends other `Grid` to this grid along axis

        """
        size = np.copy(self.size)
        size[axis] += other.size[axis]
        return self.__class__(size, bc=np.copy(self.bc),
                              geom=self.geom.append(other.geom, axis))

    @staticmethod
    def read(sile, *args, **kwargs):
        """ Reads grid from the `Sile` using `read_grid`

        Parameters
        ----------
        sile : Sile, str
            a `Sile` object which will be used to read the grid
            if it is a string it will create a new sile using `get_sile`.
        * : args passed directly to ``read_grid(,**)``
        """
        # This only works because, they *must*
        # have been imported previously
        from sisl.io import get_sile, BaseSile
        if isinstance(sile, BaseSile):
            return sile.read_grid(*args, **kwargs)
        else:
            return get_sile(sile).read_grid(*args, **kwargs)

    def write(self, sile):
        """ Writes grid to the `Sile` using `write_grid`

        Parameters
        ----------
        sile : Sile, str
            a `Sile` object which will be used to write the grid
            if it is a string it will create a new sile using `get_sile`
        """

        # This only works because, they *must*
        # have been imported previously
        from sisl.io import get_sile, BaseSile
        if isinstance(sile, BaseSile):
            sile.write_grid(self)
        else:
            get_sile(sile, 'w').write_grid(self)

    def __repr__(self):
        """ Representation of object """
        return 'Grid[{} {} {}]'.format(*self.size)

    def _check_compatibility(self, other, msg):
        """ Internal check for asserting two grids are commensurable """
        if self == other:
            return True
        else:
            s1 = repr(self)
            s2 = repr(other)
            raise ValueError(
                'Grids are not compatible, ' +
                s1 +
                '-' +
                s2 +
                '. ',
                msg)

    def _compatible_copy(self, other, *args, **kwargs):
        """ Returns a copy of self with an additional check of commensurable """
        if isinstance(other, Grid):
            if self._check_compatibility(other, *args, **kwargs):
                return self.copy()
        else:
            return self.copy()

    def __eq__(self, other):
        """ Returns true if the two grids are commensurable

        There will be no check of the values _on_ the grid. """
        return bool(np.all(self.size == other.size))

    def __ne__(self, other):
        """ Returns whether two grids have the same size """
        return not (self == other)

    def __add__(self, other):
        """ Returns a new grid with the addition of two grids

        Returns same size with same cell as the first"""
        if isinstance(other, Grid):
            grid = self._compatible_copy(other, 'they cannot be added')
            grid.grid = self.grid + other.grid
        else:
            grid = self.copy()
            grid.grid = self.grid + other
        return grid

    def __iadd__(self, other):
        """ Returns a new grid with the addition of two grids

        Returns same size with same cell as the first"""
        if isinstance(other, Grid):
            self._check_compatibility(other, 'they cannot be added')
            self.grid += other.grid
        else:
            self.grid += other
        return self

    def __sub__(self, other):
        """ Returns a new grid with the difference of two grids

        Returns same size with same cell as the first"""
        if isinstance(other, Grid):
            grid = self._compatible_copy(other, 'they cannot be subtracted')
            grid.grid = self.grid - other.grid
        else:
            grid = self.copy()
            grid.grid = self.grid - other
        return grid

    def __isub__(self, other):
        """ Returns a same grid with the difference of two grids

        Returns same size with same cell as the first"""
        if isinstance(other, Grid):
            self._check_compatibility(other, 'they cannot be subtracted')
            self.grid -= other.grid
        else:
            self.grid -= other
        return self

    def __div__(self, other):
        return self.__truediv__(other)

    def __idiv__(self, other):
        return self.__itruediv__(other)

    def __truediv__(self, other):
        if isinstance(other, Grid):
            grid = self._compatible_copy(other, 'they cannot be divided')
            grid.grid = self.grid / other.grid
        else:
            grid = self.copy()
            grid.grid = self.grid / other
        return grid

    def __itruediv__(self, other):
        if isinstance(other, Grid):
            self._check_compatibility(other, 'they cannot be divided')
            self.grid /= other.grid
        else:
            self.grid /= other
        return self

    def __mul__(self, other):
        if isinstance(other, Grid):
            grid = self._compatible_copy(other, 'they cannot be multiplied')
            grid.grid = self.grid * other.grid
        else:
            grid = self.copy()
            grid.grid = self.grid * other
        return grid

    def __imul__(self, other):
        if isinstance(other, Grid):
            self._check_compatibility(other, 'they cannot be multiplied')
            self.grid *= other.grid
        else:
            self.grid *= other
        return self


if __name__ == "__main__":
    pass
