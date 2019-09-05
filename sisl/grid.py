from __future__ import print_function, division

from functools import partial
from numbers import Integral, Real
from math import pi

import numpy as np
from numpy import int32
from numpy import floor, dot, add, cos, sin
from numpy import ogrid, take
from scipy.sparse import diags as sp_diags
from scipy.sparse import SparseEfficiencyWarning

from . import _array as _a
from ._help import dtype_complex_to_real, wrap_filterwarnings
from .shape import Shape
from .utils import default_ArgumentParser, default_namespace
from .utils import cmd, strseq, direction, str_spec
from .utils import array_arange
from .utils.mathematics import fnorm

from .supercell import SuperCellChild
from .geometry import Geometry

__all__ = ['Grid', 'sgrid']


class Grid(SuperCellChild):
    """ Real-space grid information with associated geometry.

    This grid object handles cell vectors and divisions of said grid.

    Parameters
    ----------
    shape : float or (3,) of int
        the shape of the grid. A ``float`` specifies the grid spacing in Angstrom, while
        a list of integers specifies the exact grid size.
    bc : list of int (3, 2) or (3, ), optional
        the boundary conditions for each of the cell's planes. Default to periodic BC.
    sc : SuperCell, optional
        the supercell that this grid represents. `sc` has precedence if both `geometry` and `sc`
        has been specified. Default to ``[1, 1, 1]``.
    dtype : numpy.dtype, optional
        the data-type of the grid, default to `numpy.float64`.
    geometry : Geometry, optional
        associated geometry with the grid. If `sc` has not been passed the supercell will
        be taken from this geometry.

    Examples
    --------
    >>> grid1 = Grid(0.1, sc=10)
    >>> grid2 = Grid(0.1, sc=SuperCell(10))
    >>> grid3 = Grid(0.1, sc=SuperCell([10] * 3))
    >>> grid1 == grid2
    True
    >>> grid1 == grid3
    True
    >>> grid = Grid(0.1, sc=10, dtype=np.complex128)
    >>> grid == grid1
    False
    """

    #: Constant for defining a periodic boundary condition
    PERIODIC = 1
    #: Constant for defining a Neumann boundary condition
    NEUMANN = 2
    #: Constant for defining a Dirichlet boundary condition
    DIRICHLET = 3
    #: Constant for defining an open boundary condition
    OPEN = 4

    def __init__(self, shape, bc=None, sc=None, dtype=None, geometry=None):
        if bc is None:
            bc = [[self.PERIODIC] * 2] * 3

        self.set_supercell(sc)

        # Create the atomic structure in the grid, if possible
        self.set_geometry(geometry)

        if isinstance(shape, Real):
            d = (self.cell ** 2).sum(1) ** 0.5
            shape = list(map(int, np.rint(d / shape)))

        # Create the grid
        self.set_grid(shape, dtype=dtype)

        # Create the grid boundary conditions
        self.set_bc(bc)

        # If the user sets the super-cell, that has precedence.
        if sc is not None:
            if not self.geometry is None:
                self.geometry.set_sc(sc)
            self.set_sc(sc)

    def __getitem__(self, key):
        """ Grid value at `key` """
        return self.grid[key]

    def __setitem__(self, key, val):
        """ Updates the grid contained """
        self.grid[key] = val

    @property
    def geom(self):
        """ Short-hand for `self.geometry`, may be deprecated later """
        return self.geometry

    def set_geometry(self, geometry):
        """ Sets the `Geometry` for the grid.

        Setting the `Geometry` for the grid is a possibility
        to attach atoms to the grid.

        It is not a necessary entity, so passing `None` is a viable option.
        """
        if geometry is None:
            # Fake geometry
            self.geometry = None
        else:
            self.geometry = geometry
            self.set_sc(geometry.sc)
    set_geom = set_geometry

    def fill(self, val):
        """ Fill the grid with this value

        Parameters
        ----------
        val : numpy.dtype
           all grid-points will have this value after execution
        """
        self.grid.fill(val)

    def interp(self, shape, method='linear', **kwargs):
        """ Interpolate grid values to a new grid

        Parameters
        ----------
        shape : int, array_like
            the new shape of the grid
        method : str
            the method used to perform the interpolation,
            see `scipy.interpolate.interpn` for further details.
        **kwargs : dict
            optional arguments passed to the interpolation algorithm
            The interpolation routine is `scipy.interpolate.interpn`
        """
        # Interpolate function
        from scipy.interpolate import RegularGridInterpolator

        # Get current grid spacing
        dold = (
            _a.linspacef(0, 1, self.shape[0]),
            _a.linspacef(0, 1, self.shape[1]),
            _a.linspacef(0, 1, self.shape[2])
        )

        # Create new grid and clean-up to reduce memory
        grid = self.copy()
        del grid.grid

        # Create the interpolator!
        f = RegularGridInterpolator(dold, self.grid, method=method)
        del dold

        # Create new interpolation points
        dnew = (
            _a.linspacef(0, 1, shape[0]),
            _a.linspacef(0, 1, shape[1]),
            _a.linspacef(0, 1, shape[2])
        )

        grid.grid = np.empty(shape, dtype=self.dtype)
        for iz, dz in enumerate(dnew[2]):
            dnew = np.mgrid[0:1:shape[0] * 1j, 0:1:shape[1] * 1j, dz:dz*1.0001:1j].reshape(3, -1).T
            grid.grid[:, :, iz] = f(dnew).reshape((shape[0], shape[1]))

        del dnew

        return grid

    @property
    def size(self):
        """ Total number of elements in the grid """
        return np.prod(self.grid.shape)

    @property
    def shape(self):
        r""" Grid shape in :math:`x`, :math:`y`, :math:`z` directions """
        return self.grid.shape

    @property
    def dtype(self):
        """ Data-type used in grid """
        return self.grid.dtype

    @property
    def dkind(self):
        """ The data-type of the grid (in str) """
        return np.dtype(self.grid.dtype).kind

    def set_grid(self, shape, dtype=None):
        """ Create the internal grid of certain size. """
        shape = _a.asarrayi(shape).ravel()
        if dtype is None:
            dtype = np.float64
        if shape.size != 3:
            raise ValueError(self.__class__.__name__ + '.set_grid requires shape to be of length 3')
        self.grid = np.zeros(shape, dtype=dtype)

    def set_bc(self, boundary=None, a=None, b=None, c=None):
        """ Set the boundary conditions on the grid

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
        ValueError : if specifying periodic one one boundary, so must the opposite side.
        """
        if not boundary is None:
            if isinstance(boundary, Integral):
                self.bc = _a.fulli([3, 2], boundary)
            else:
                self.bc = _a.asarrayi(boundary)
        if not a is None:
            self.bc[0, :] = a
        if not b is None:
            self.bc[1, :] = b
        if not c is None:
            self.bc[2, :] = c

        # shorthand for bc
        bc = self.bc[:, :]
        for i in range(3):
            if (bc[i, 0] == self.PERIODIC and bc[i, 1] != self.PERIODIC) or \
               (bc[i, 0] != self.PERIODIC and bc[i, 1] == self.PERIODIC):
                raise ValueError(self.__class__.__name__ + '.set_bc has a one non-periodic and '
                                 'one periodic direction. If one direction is periodic, both instances '
                                 'must have that BC.')

    # Aliases
    set_boundary = set_bc
    set_boundary_condition = set_bc

    def __sc_geometry_dict(self):
        """ Internal routine for copying the SuperCell and Geometry """
        d = dict()
        d['sc'] = self.sc.copy()
        if not self.geometry is None:
            d['geometry'] = self.geometry.copy()
        return d

    def copy(self):
        """ Copy the object """
        d = self.__sc_geometry_dict()
        d['dtype'] = self.dtype
        grid = self.__class__([1] * 3, bc=np.copy(self.bc), **d)
        # This also ensures the shape is copied!
        grid.grid = self.grid.copy()
        return grid

    def swapaxes(self, a, b):
        """ Swap two axes in the grid (also swaps axes in the supercell)

        If ``swapaxes(0,1)`` it returns the 0 in the 1 values.

        Parameters
        ----------
        a, b : int
            axes indices to be swapped
        """
        # Create index vector
        idx = _a.arangei(3)
        idx[b] = a
        idx[a] = b
        s = np.copy(self.shape)
        d = self.__sc_geometry_dict()
        d['sc'] = d['sc'].swapaxes(a, b)
        d['dtype'] = self.dtype
        grid = self.__class__(s[idx], bc=self.bc[idx], **d)
        # We need to force the C-order or we loose the contiguity
        grid.grid = np.copy(np.swapaxes(self.grid, a, b), order='C')
        return grid

    @property
    def dcell(self):
        """ Voxel cell size """
        # Calculate the grid-distribution
        return self.cell / _a.asarrayi(self.shape).reshape(3, 1)

    @property
    def dvolume(self):
        """ Volume of the grid voxel elements """
        return self.sc.volume / self.size

    def _copy_sub(self, n, axis, scale_geometry=False):
        # First calculate the new shape
        shape = list(self.shape)
        cell = np.copy(self.cell)
        # Down-scale cell
        cell[axis, :] = (cell[axis, :] / shape[axis]) * n
        shape[axis] = n
        if n < 1:
            raise ValueError('You cannot retain no indices.')
        grid = self.__class__(shape, bc=np.copy(self.bc), dtype=self.dtype, **self.__sc_geometry_dict())
        # Update cell shape (the cell is smaller now)
        grid.set_sc(cell)
        if scale_geometry and not self.geometry is None:
            geom = self.geometry.copy()
            fxyz = geom.fxyz.copy()
            geom.set_supercell(grid.sc)
            geom.xyz[:, :] = np.dot(fxyz, grid.sc.cell)
            grid.set_geometry(geom)

        return grid

    def cross_section(self, idx, axis):
        """ Takes a cross-section of the grid along axis `axis`

        Remark: This API entry might change to handle arbitrary
        cuts via rotation of the axis """
        idx = _a.asarrayi(idx).ravel()
        grid = self._copy_sub(1, axis)

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
        """ Sum grid values along axis `axis`.

        Parameters
        ----------
        axis : int
            unit-cell direction to sum across
        """
        grid = self._copy_sub(1, axis, scale_geometry=True)
        # Calculate sum (retain dimensions)
        np.sum(self.grid, axis=axis, keepdims=True, out=grid.grid)
        return grid

    def average(self, axis, weights=None):
        """ Average grid values along direction `axis`.

        Parameters
        ----------
        axis : int
            unit-cell direction to average across
        weights : array_like, optional
            the weights for the individual axis elements, if boolean it corresponds to 0 and 1
            for false/true.

        See Also
        --------
        numpy.average : for details regarding the `weights` argument
        """
        grid = self._copy_sub(1, axis, scale_geometry=True)

        if weights is None:
            # Calculate sum (retain dimensions)
            np.sum(self.grid, axis=axis, keepdims=True, out=grid.grid)
            grid.grid /= self.shape[axis]
        elif axis == 0:
            grid.grid[0, :, :] = np.average(self.grid, axis=axis, weights=weights)
        elif axis == 1:
            grid.grid[:, 0, :] = np.average(self.grid, axis=axis, weights=weights)
        elif axis == 2:
            grid.grid[:, :, 0] = np.average(self.grid, axis=axis, weights=weights)
        else:
            raise ValueError(self.__class__.__name__ + '.average requires `axis` to be in [0, 1, 2]')

        return grid

    # for compatibility
    mean = average

    def remove_part(self, idx, axis, above):
        """ Removes parts of the grid via above/below designations.

        Works exactly opposite to `sub_part`

        Parameters
        ----------
        idx : int
           the index of the grid axis `axis` to be removed
           for ``above=True`` grid[:idx,...]
           for ``above=False`` grid[idx:,...]
        axis : int
           the axis segment from which we retain the indices `idx`
        above : bool
           if ``True`` will retain the grid:
              ``grid[:idx,...]``
           else it will retain the grid:
              ``grid[idx:,...]``
        """
        return self.sub_part(idx, axis, not above)

    def sub_part(self, idx, axis, above):
        """ Retains parts of the grid via above/below designations.

        Works exactly opposite to `remove_part`

        Parameters
        ----------
        idx : int
           the index of the grid axis `axis` to be retained
           for ``above=True`` grid[idx:,...]
           for ``above=False`` grid[:idx,...]
        axis : int
           the axis segment from which we retain the indices `idx`
        above : bool
           if ``True`` will retain the grid:
              ``grid[idx:,...]``
           else it will retain the grid:
              ``grid[:idx,...]``
        """
        if above:
            sub = _a.arangei(idx, self.shape[axis])
        else:
            sub = _a.arangei(0, idx)
        return self.sub(sub, axis)

    def sub(self, idx, axis):
        """ Retains certain indices from a specified axis.

        Works exactly opposite to `remove`.

        Parameters
        ----------
        idx : array_like
           the indices of the grid axis `axis` to be retained
        axis : int
           the axis segment from which we retain the indices `idx`
        """
        idx = _a.asarrayi(idx).ravel()
        shift_geometry = False
        if len(idx) > 1:
            if np.allclose(np.diff(idx), 1):
                shift_geometry = not self.geometry is None

        if shift_geometry:
            grid = self._copy_sub(len(idx), axis)
            min_xyz = self.dcell[axis, :] * idx[0]
            # Now shift the geometry according to what is retained
            geom = self.geometry.translate(-min_xyz)
            geom.set_supercell(grid.sc)
            grid.set_geometry(geom)
        else:
            grid = self._copy_sub(len(idx), axis, scale_geometry=True)

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
           the indices of the grid axis `axis` to be removed
        axis : int
           the axis segment from which we remove all indices `idx`
        """
        ret_idx = np.delete(_a.arangei(self.shape[axis]), _a.asarrayi(idx))
        return self.sub(ret_idx, axis)

    def index2xyz(self, index):
        """ Real-space coordinates of indices related to the grid

        Parameters
        ----------
        index : array_like
           indices for grid-positions

        Returns
        -------
        numpy.ndarray
           coordinates of the indices with respect to this grid spacing
        """
        return dot(np.asarray(index), self.dcell)

    def index_fold(self, index, unique=True):
        """ Converts indices from *any* placement to only exist in the "primary" grid

        Examples
        --------
        >>> grid = Grid([10, 10, 10])
        >>> assert np.all(grid.index2primary([-1, -1, -1]) == 9)

        Parameters
        ----------
        index : array_like
           indices for grid-positions
        unique : bool, optional
           if true the returned indices are made unique after having folded the index points

        Returns
        -------
        numpy.ndarray
            all indices are then within the shape of the grid
        """
        index = _a.asarrayi(index)
        ndim = index.ndim

        # Convert to internal
        if unique:
            index = np.unique(index.reshape(-1, 3) % _a.asarrayi(self.shape)[None, :], axis=0)
        else:
            index = index.reshape(-1, 3) % _a.asarrayi(self.shape)[None, :]

        if ndim == 1:
            return index.ravel()
        return index

    def _index_shape(self, shape):
        """ Internal routine for shape-indices """
        # First grab the sphere, subsequent indices will be reduced
        # by the actual shape
        cuboid = shape.toCuboid()
        ellipsoid = shape.toEllipsoid()
        if ellipsoid.volume() > cuboid.volume():
            idx = self._index_shape_cuboid(cuboid)
        else:
            idx = self._index_shape_ellipsoid(ellipsoid)

        # Get min/max
        imin = idx.min(0)
        imax = idx.max(0)
        del idx

        dc = self.dcell

        # Now to find the actual points inside the shape
        # First create all points in the square and then retrieve all indices
        # within.
        ix = _a.aranged(imin[0], imax[0] + 0.5)
        iy = _a.aranged(imin[1], imax[1] + 0.5)
        iz = _a.aranged(imin[2], imax[2] + 0.5)
        output_shape = (ix.size, iy.size, iz.size, 3)
        rxyz = _a.emptyd(output_shape)
        ao = add.outer
        ao(ao(ix * dc[0, 0], iy * dc[1, 0]), iz * dc[2, 0], out=rxyz[:, :, :, 0])
        ao(ao(ix * dc[0, 1], iy * dc[1, 1]), iz * dc[2, 1], out=rxyz[:, :, :, 1])
        ao(ao(ix * dc[0, 2], iy * dc[1, 2]), iz * dc[2, 2], out=rxyz[:, :, :, 2])
        idx = shape.within_index(rxyz.reshape(-1, 3))
        del rxyz
        i = _a.emptyi(output_shape)
        i[:, :, :, 0] = ix.reshape(-1, 1, 1)
        i[:, :, :, 1] = iy.reshape(1, -1, 1)
        i[:, :, :, 2] = iz.reshape(1, 1, -1)
        del ix, iy, iz
        i.shape = (-1, 3)
        i = take(i, idx, axis=0)
        del idx

        return i

    def _index_shape_cuboid(self, cuboid):
        """ Internal routine for cuboid shape-indices """
        # Construct all points on the outer rim of the cuboids
        min_d = fnorm(self.dcell).min()

        # Retrieve cuboids edge-lengths
        v = cuboid.edge_length
        # Create normalized cuboid vectors (because we expan via the lengths below
        vn = cuboid._v / fnorm(cuboid._v).reshape(-1, 1)
        LL = (cuboid.center - cuboid._v.sum(0) / 2).reshape(1, 3)
        UR = (cuboid.center + cuboid._v.sum(0) / 2).reshape(1, 3)

        # Create coordinates
        a = vn[0, :].reshape(1, -1) * _a.aranged(0, v[0] + min_d, min_d).reshape(-1, 1)
        b = vn[1, :].reshape(1, -1) * _a.aranged(0, v[1] + min_d, min_d).reshape(-1, 1)
        c = vn[2, :].reshape(1, -1) * _a.aranged(0, v[2] + min_d, min_d).reshape(-1, 1)

        # Now create all sides
        sa = a.shape[0]
        sb = b.shape[0]
        sc = c.shape[0]

        def plane(v1, v2):
            return (v1.reshape(-1, 1, 3) + v2.reshape(1, -1, 3)).reshape(1, -1, 3)

        # Allocate for the 6 faces of the cuboid
        rxyz = _a.emptyd([2, sa * sb + sa * sc + sb * sc, 3])
        # Define the LL and UR
        rxyz[0, :, :] = LL
        rxyz[1, :, :] = UR

        i = 0
        rxyz[:, i:i + sa * sb, :] += plane(a, b)
        i += sa * sb
        rxyz[:, i:i + sa * sc, :] += plane(a, c)
        i += sa * sc
        rxyz[:, i:i + sb * sc, :] += plane(b, c)
        del a, b, c, sa, sb, sc
        rxyz.shape = (-1, 3)

        # Get all indices of the cuboid planes
        return self.index(rxyz)

    def _index_shape_ellipsoid(self, ellipsoid):
        """ Internal routine for ellipsoid shape-indices """
        # Figure out the points on the ellipsoid
        rad1 = pi / 180
        theta, phi = ogrid[-pi:pi:rad1, 0:pi:rad1]

        rxyz = _a.emptyd([theta.size, phi.size, 3])
        rxyz[..., 2] = cos(phi)
        sin(phi, out=phi)
        rxyz[..., 0] = cos(theta) * phi
        rxyz[..., 1] = sin(theta) * phi
        rxyz = dot(rxyz, ellipsoid._v) + ellipsoid.center.reshape(1, 3)
        del theta, phi

        # Get all indices of the ellipsoid circumference
        return self.index(rxyz)

    def index(self, coord, axis=None):
        """ Find the grid index for a given coordinate (possibly only along a given lattice vector `axis`)

        Parameters
        ----------
        coord : (:, 3) or float or Shape
            the coordinate of the axis. If a float is passed `axis` is
            also required in which case it corresponds to the length along the
            lattice vector corresponding to `axis`.
            If a Shape a list of coordinates that fits the voxel positions
            are returned (all internal points also).
        axis : int, optional
            the axis direction of the index, or for all axes if none.
        """
        if isinstance(coord, Shape):
            # We have to do something differently
            return self._index_shape(coord)

        coord = _a.asarrayd(coord)
        if coord.size == 1: # float
            if axis is None:
                raise ValueError(self.__class__.__name__ + '.index requires the '
                                 'coordinate to be 3 values when an axis has not '
                                 'been specified.')

            c = (self.dcell[axis, :] ** 2).sum() ** 0.5
            return int(floor(coord / c))

        icell = self.icell

        # Ensure we return values in the same dimensionality
        ndim = coord.ndim

        shape = np.array(self.shape).reshape(3, -1)

        # dot(icell, coord) is the fraction in the
        # cell. So * l / (l / self.shape) will
        # give the float of dcell lattice vectors (where l is the length of
        # each lattice vector)
        if axis is None:
            if ndim == 1:
                return floor(dot(icell, coord.reshape(-1, 3).T) * shape).reshape(3).astype(int32, copy=False)
            else:
                return floor(dot(icell, coord.reshape(-1, 3).T) * shape).T.astype(int32, copy=False)
        if ndim == 1:
            return floor(dot(icell[axis, :], coord.reshape(-1, 3).T) * shape[axis]).astype(int32, copy=False)[0]
        else:
            return floor(dot(icell[axis, :], coord.reshape(-1, 3).T) * shape[axis]).T.astype(int32, copy=False)

    def append(self, other, axis):
        """ Appends other `Grid` to this grid along axis """
        shape = list(self.shape)
        shape[axis] += other.shape[axis]
        d = self.__sc_geometry_dict()
        if 'geometry' in d:
            if not other.geometry is None:
                d['geometry'] = d['geometry'].append(other.geometry, axis)
        else:
            d['geometry'] = other.geometry
        d['dtype'] = self.dtype
        return self.__class__(shape, bc=np.copy(self.bc), **d)

    @staticmethod
    def read(sile, *args, **kwargs):
        """ Reads grid from the `Sile` using `read_grid`

        Parameters
        ----------
        sile : Sile, str or pathlib.Path
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
            sile = str(sile)
            sile, spec = str_spec(sile)
            if spec is not None:
                if ',' in spec:
                    kwargs['index'] = list(map(float, spec.split(',')))
                else:
                    kwargs['index'] = int(spec)
            with get_sile(sile) as fh:
                return fh.read_grid(*args, **kwargs)

    def write(self, sile, *args, **kwargs):
        """ Writes grid to the `Sile` using `write_grid`

        Parameters
        ----------
        sile : Sile, str or pathlib.Path
            a `Sile` object which will be used to write the grid
            if it is a string it will create a new sile using `get_sile`
        """

        # This only works because, they *must*
        # have been imported previously
        from sisl.io import get_sile, BaseSile
        if isinstance(sile, BaseSile):
            sile.write_grid(self, *args, **kwargs)
        else:
            with get_sile(sile, 'w') as fh:
                fh.write_grid(self, *args, **kwargs)

    def __str__(self):
        """ String of object """
        s = self.__class__.__name__ + '{{kind: {kind}, shape: [{shape[0]} {shape[1]} {shape[2]}],\n'.format(kind=self.dkind, shape=self.shape)
        bc = {self.PERIODIC: 'periodic',
              self.NEUMANN: 'neumann',
              self.DIRICHLET: 'dirichlet',
              self.OPEN: 'open'
        }
        s += ' bc: [{}, {},\n      {}, {},\n      {}, {}],\n '.format(bc[self.bc[0, 0]], bc[self.bc[0, 1]],
                                                                      bc[self.bc[1, 0]], bc[self.bc[1, 1]],
                                                                      bc[self.bc[2, 0]], bc[self.bc[2, 1]])
        if not self.geometry is None:
            s += '{}\n}}'.format(str(self.geometry).replace('\n', '\n '))
        else:
            s += '{}\n}}'.format(str(self.sc).replace('\n', '\n '))
        return s

    def _check_compatibility(self, other, msg):
        """ Internal check for asserting two grids are commensurable """
        if self == other:
            return True
        s1 = str(self)
        s2 = str(other)
        raise ValueError('Grids are not compatible, ' +
                         s1 + '-' + s2 + '. ', msg)

    def _compatible_copy(self, other, *args, **kwargs):
        """ Internally used copy function that also checks whether the two grids are compatible """
        if isinstance(other, Grid):
            self._check_compatibility(other, *args, **kwargs)
        return self.copy()

    def __eq__(self, other):
        """ Whether two grids are commensurable (no value checks, only grid shape)

        There will be no check of the values _on_ the grid. """
        return self.shape == other.shape

    def __ne__(self, other):
        """ Whether two grids are incommensurable (no value checks, only grid shape) """
        return not (self == other)

    def __abs__(self):
        r""" Take the absolute value of the grid :math:`|grid|` """
        dtype = dtype_complex_to_real(self.dtype)
        a = self.copy()
        a.grid = np.absolute(self.grid).astype(dtype, copy=False)
        return a

    def __add__(self, other):
        """ Add two grid values (or add a single value to all grid values)

        Raises
        ------
        ValueError: if the grids are not compatible (different shapes)
        """
        if isinstance(other, Grid):
            grid = self._compatible_copy(other, 'they cannot be added')
            grid.grid = self.grid + other.grid
        else:
            grid = self.copy()
            grid.grid = self.grid + other
        return grid

    def __iadd__(self, other):
        """ Add, in-place, values from another grid

        Raises
        ------
        ValueError: if the grids are not compatible (different shapes)
        """
        if isinstance(other, Grid):
            self._check_compatibility(other, 'they cannot be added')
            self.grid += other.grid
        else:
            self.grid += other
        return self

    def __sub__(self, other):
        """ Subtract two grid values (or subtract a single value from all grid values)

        Raises
        ------
        ValueError: if the grids are not compatible (different shapes)
        """
        if isinstance(other, Grid):
            grid = self._compatible_copy(other, 'they cannot be subtracted')
            np.subtract(self.grid, other.grid, out=grid.grid)
        else:
            grid = self.copy()
            np.subtract(self.grid, other, out=grid.grid)
        return grid

    def __isub__(self, other):
        """ Subtract, in-place, values from another grid

        Raises
        ------
        ValueError: if the grids are not compatible (different shapes)
        """
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
            np.divide(self.grid, other.grid, out=grid.grid)
        else:
            grid = self.copy()
            np.divide(self.grid, other, out=grid.grid)
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
            np.multiply(self.grid, other.grid, out=grid.grid)
        else:
            grid = self.copy()
            np.multiply(self.grid, other, out=grid.grid)
        return grid

    def __imul__(self, other):
        if isinstance(other, Grid):
            self._check_compatibility(other, 'they cannot be multiplied')
            self.grid *= other.grid
        else:
            self.grid *= other
        return self

    # Here comes additional supplementary routines which enables an easy
    # work-through case with other programs.
    @classmethod
    def mgrid(cls, *slices):
        """ Return a list of indices corresponding to the slices

        The returned values are equivalent to `numpy.mgrid` but they are returned
        in a (:, 3) array.

        Parameters
        ----------
        *slices : slice or list of int or int
            return a linear list of indices that points to the collective slice
            made by the passed arguments

        Returns
        -------
        numpy.ndarray
            linear indices for each of the sliced values, shape ``(*, 3)``
        """
        if len(slices) == 1:
            g = np.mgrid[slices[0]]
        else:
            g = np.mgrid[slices]
        indices = _a.emptyi(g.size).reshape(-1, 3)
        indices[:, 0] = g[0].flatten()
        indices[:, 1] = g[1].flatten()
        indices[:, 2] = g[2].flatten()
        del g
        return indices

    def pyamg_index(self, index):
        r""" Calculate `pyamg` matrix indices from a list of grid indices

        Parameters
        ----------
        index : (:, 3) of int
            a list of indices of the grid along each grid axis

        Returns
        -------
        numpy.ndarray
            linear indices for the matrix

        See Also
        --------
        index : query indices from coordinates (directly passable to this method)
        mgrid : Grid equivalent to `numpy.mgrid`. Grid.mgrid returns indices in shapes (:, 3), contrary to numpy's `numpy.mgrid`

        Raises
        ------
        ValueError : if any of the passed indices are below 0 or above the number of elements per axis
        """
        index = _a.asarrayi(index).reshape(-1, 3)
        grid = _a.arrayi(self.shape[:])
        if np.any(index < 0) or np.any(index >= grid.reshape(1, 3)):
            raise ValueError(self.__class__.__name__ + '.pyamg_index erroneous values for grid indices')
        # Skipping factor per element
        cp = _a.arrayi([[grid[1] * grid[2], grid[2], 1]])
        return (cp * index).sum(1)

    @classmethod
    def pyamg_source(cls, b, pyamg_indices, value):
        r""" Fix the source term to `value`.

        Parameters
        ----------
        b : numpy.ndarray
           a vector containing RHS of :math:`A x = b` for the solution of the grid stencil
        pyamg_indices : list of int
           the linear pyamg matrix indices where the value of the grid is fixed. I.e. the indices should
           correspond to returned quantities from `pyamg_indices`.
        """
        b[pyamg_indices] = value

    def pyamg_fix(self, A, b, pyamg_indices, value):
        r""" Fix values for the stencil to `value`.

        Parameters
        ----------
        A : `~scipy.sparse.csr_matrix`/`~scipy.sparse.csc_matrix`
           sparse matrix describing the LHS for the linear system of equations
        b : numpy.ndarray
           a vector containing RHS of :math:`A x = b` for the solution of the grid stencil
        pyamg_indices : list of int
           the linear pyamg matrix indices where the value of the grid is fixed. I.e. the indices should
           correspond to returned quantities from `pyamg_indices`.
        value : float
           the value of the grid to fix the value at
        """
        if not A.format in ['csc', 'csr']:
            raise ValueError(self.__class__.__name__ + '.pyamg_fix only works for csr/csc sparse matrices')

        # Clean all couplings between the respective indices and all other data
        s = array_arange(A.indptr[pyamg_indices], A.indptr[pyamg_indices+1])
        A.data[s] = 0.
        # clean-up
        del s

        # Specify that these indices are not to be tampered with
        d = np.zeros(A.shape[0], dtype=A.dtype)
        d[pyamg_indices] = 1.
        # BUG in scipy, sparse matrix += does not do in-place operations
        # hence we need to overwrite the `A` matrix afterward
        AA = A + sp_diags(d, format=A.format)
        del d
        # Restore data in the A array
        A.indices = AA.indices
        A.indptr = AA.indptr
        A.data = AA.data
        del AA
        A.eliminate_zeros()

        # force RHS value
        self.pyamg_source(b, pyamg_indices, value)

    @wrap_filterwarnings("ignore", category=SparseEfficiencyWarning)
    def pyamg_boundary_condition(self, A, b, bc=None):
        r""" Attach boundary conditions to the `pyamg` grid-matrix `A` with default boundary conditions as specified for this `Grid`

        Parameters
        ----------
        A : scipy.sparse.csr_matrix
           sparse matrix describing the grid
        b : numpy.ndarray
           a vector containing RHS of :math:`A x = b` for the solution of the grid stencil
        bc : list of BC, optional
           the specified boundary conditions.
           Default to the grid's boundary conditions, else `bc` *must* be a list of elements
           with elements corresponding to `Grid.PERIODIC`/`Grid.NEUMANN`...
        """
        C = -1

        def Neumann(idx_bc, idx_p1):
            # TODO check this BC
            # Set all boundary equations to 0
            s = array_arange(A.indptr[idx_bc], A.indptr[idx_bc+1])
            A.data[s] = 0
            # force the boundary cells to equal the neighbouring cell
            A[idx_bc, idx_bc] = -C # I am not sure this is correct, but setting it to 0 does NOT work
            A[idx_bc, idx_p1] = C
            # ensure the neighbouring cell doesn't connect to the boundary (no back propagation)
            A[idx_p1, idx_bc] = 0
            # Ensure the sum of the source for the neighbouring cells equals 0
            # To make it easy to figure out the diagonal elements we first
            # set it to 0, then sum off-diagonal terms, and set the diagonal
            # equal to the sum.
            A[idx_p1, idx_p1] = 0
            n = A.indptr[idx_p1+1] - A.indptr[idx_p1]
            s = array_arange(A.indptr[idx_p1], n=n)
            n = np.split(A.data[s], np.cumsum(n)[:-1])
            n = _a.fromiteri(map(np.sum, n))
            # update diagonal
            A[idx_p1, idx_p1] = -n
            del s, n
            A.eliminate_zeros()
            b[idx_bc] = 0.
        def Dirichlet(idx):
            # Default pyamg Poisson matrix has Dirichlet BC
            b[idx] = 0.
        def Periodic(idx1, idx2):
            A[idx1, idx2] = C
            A[idx2, idx1] = C

        def sl2idx(sl):
            return self.pyamg_index(self.mgrid(sl))

        # Create slices
        sl = [slice(0, g) for g in self.shape]

        for i in range(3):
            # We have a periodic direction
            new_sl = sl[:]

            # LOWER BOUNDARY
            bc = self.bc[i, 0]
            new_sl[i] = slice(0, 1)
            idx1 = sl2idx(new_sl) # lower

            if self.bc[i, 0] == self.PERIODIC or \
               self.bc[i, 1] == self.PERIODIC:
                if self.bc[i, 0] != self.bc[i, 1]:
                    raise ValueError(self.__class__.__name__ + '.pyamg_boundary_condition found a periodic and non-periodic direction in the same direction!')
                new_sl[i] = slice(self.shape[i]-1, self.shape[i])
                idx2 = sl2idx(new_sl) # upper
                Periodic(idx1, idx2)
                del idx2
                continue

            if bc == self.NEUMANN:
                # Retrieve next index
                new_sl[i] = slice(1, 2)
                idx2 = sl2idx(new_sl) # lower + 1
                Neumann(idx1, idx2)
                del idx2
            elif bc == self.DIRICHLET:
                Dirichlet(idx1)

            # UPPER BOUNDARY
            bc = self.bc[i, 1]
            new_sl[i] = slice(self.shape[i]-1, self.shape[i])
            idx1 = sl2idx(new_sl) # upper

            if bc == self.NEUMANN:
                # Retrieve next index
                new_sl[i] = slice(self.shape[i]-2, self.shape[i]-1)
                idx2 = sl2idx(new_sl) # upper - 1
                Neumann(idx1, idx2)
                del idx2
            elif bc == self.DIRICHLET:
                Dirichlet(idx1)

        A.eliminate_zeros()

    def topyamg(self, dtype=None):
        r""" Create a `pyamg` stencil matrix to be used in pyamg

        This allows retrieving the grid matrix equivalent of the real-space grid.
        Subsequently the returned matrix may be used in pyamg for solutions etc.

        The `pyamg` suite is it-self a rather complicated code with many options.
        For details we refer to `pyamg <pyamg https://github.com/pyamg/pyamg/>`_.

        Parameters
        ----------
        dtype : numpy.dtype, optional
           data-type used for the sparse matrix, default to use the grid data-type

        Returns
        -------
        scipy.sparse.csr_matrix
            the stencil for the `pyamg` solver
        numpy.ndarray
            RHS of the linear system of equations

        Examples
        --------
        This example proves the best method for a variety of cases in regards of the 3D Poisson problem:

        >>> grid = Grid(0.01)
        >>> A, b = grid.topyamg() # automatically setups the current boundary conditions
        >>> # add terms etc. to A and/or b
        >>> import pyamg
        >>> from scipy.sparse.linalg import cg
        >>> ml = pyamg.aggregation.smoothed_aggregation_solver(A, max_levels=1000)
        >>> M = ml.aspreconditioner(cycle='W') # pre-conditioner
        >>> x, info = cg(A, b, tol=1e-12, M=M)

        See Also
        --------
        pyamg_index : convert grid indices into the sparse matrix indices for ``A``
        pyamg_fix : fixes stencil for indices and fixes the source for the RHS matrix (uses `pyamg_source`)
        pyamg_source : fix the RHS matrix ``b`` to a constant value
        pyamg_boundary_condition : setup the sparse matrix ``A`` to given boundary conditions (called in this routine)
        """
        from pyamg.gallery import poisson
        if dtype is None:
            dtype = self.dtype
        # Initially create the CSR matrix
        A = poisson(self.shape, dtype=dtype, format='csr')
        b = np.zeros(A.shape[0], dtype=A.dtype)

        # Now apply the boundary conditions
        self.pyamg_boundary_condition(A, b)
        return A, b

    @classmethod
    def _ArgumentParser_args_single(cls):
        """ Returns the options for `Grid.ArgumentParser` in case they are the only options """
        return {'limit_arguments': False,
                'short': True,
                'positional_out': True,
            }

    # Hook into the Grid class to create
    # an automatic ArgumentParser which makes actions
    # as the options are read.
    @default_ArgumentParser(description="Manipulate a Grid object in sisl.")
    def ArgumentParser(self, p=None, *args, **kwargs):
        """ Create and return a group of argument parsers which manipulates it self `Grid`.

        Parameters
        ----------
        p : ArgumentParser, None
           in case the arguments should be added to a specific parser. It defaults
           to create a new.
        limit_arguments : bool, True
           If `False` additional options will be created which are similar to other options.
           For instance `--repeat-x` which is equivalent to `--repeat x`.
        short : bool, False
           Create short options for a selected range of options
        positional_out : bool, False
           If `True`, adds a positional argument which acts as --out. This may be handy if only the geometry is in the argument list.
        """
        limit_args = kwargs.get('limit_arguments', True)
        short = kwargs.get('short', False)

        def opts(*args):
            if short:
                return args
            return [args[0]]

        # We limit the import to occur here
        import argparse

        # The first thing we do is adding the Grid to the NameSpace of the
        # parser.
        # This will enable custom actions to interact with the grid in a
        # straight forward manner.
        d = {
            "_grid": self.copy(),
            "_stored_grid": False,
        }
        namespace = default_namespace(**d)

        # Define actions
        class SetGeometry(argparse.Action):

            def __call__(self, parser, ns, value, option_string=None):
                ns._geometry = Geometry.read(value)
                ns._grid.set_geometry(ns._geometry)
        p.add_argument(*opts('--geometry', '-G'), action=SetGeometry,
                       help='Define the geometry attached to the Grid.')

        # Define size of grid
        class InterpGrid(argparse.Action):

            def __call__(self, parser, ns, values, option_string=None):
                ns._grid = ns._grid.interp([int(x) for x in values])
        p.add_argument(*opts('--interp'), nargs=3, metavar=('NX', 'NY', 'NZ'),
                       action=InterpGrid,
                       help='Interpolate the grid.')

        # substract another grid
        # They *MUST* be conmensurate.
        class DiffGrid(argparse.Action):

            def __call__(self, parser, ns, value, option_string=None):
                grid = Grid.read(value)
                ns._grid -= grid
                del grid
        p.add_argument(*opts('--diff', '-d'), action=DiffGrid,
                       help='Subtract another grid (they must be commensurate).')

        class AverageGrid(argparse.Action):

            def __call__(self, parser, ns, value, option_string=None):
                ns._grid = ns._grid.average(direction(value))
        p.add_argument(*opts('--average'), metavar='DIR',
                       action=AverageGrid,
                       help='Take the average of the grid along DIR.')

        class SumGrid(argparse.Action):

            def __call__(self, parser, ns, value, option_string=None):
                ns._grid = ns._grid.sum(direction(value))
        p.add_argument(*opts('--sum'), metavar='DIR',
                       action=SumGrid,
                       help='Take the sum of the grid along DIR.')

        # Create-subsets of the grid
        class SubDirectionGrid(argparse.Action):

            def __call__(self, parser, ns, values, option_string=None):
                # The unit-cell direction
                axis = direction(values[1])
                # Figure out whether this is a fractional or
                # distance in Ang
                is_frac = 'f' in values[0]
                rng = strseq(float, values[0].replace('f', ''))
                if isinstance(rng, tuple):
                    if is_frac:
                        rng = tuple(rng)
                    # we have bounds
                    if rng[0] is None:
                        idx1 = 0
                    else:
                        idx1 = ns._grid.index(rng[0], axis=axis)
                    if rng[1] is None:
                        idx2 = ns._grid.shape[axis]
                    else:
                        idx2 = ns._grid.index(rng[1], axis=axis)
                    ns._grid = ns._grid.sub(_a.arangei(idx1, idx2), axis)
                    return
                elif rng < 0.:
                    if is_frac:
                        rng = ns._grid.cell[axis, :] * abs(rng)
                    b = False
                else:
                    if is_frac:
                        rng = ns._grid.cell[axis, :] * rng
                    b = True
                idx = ns._grid.index(rng, axis=axis)
                ns._grid = ns._grid.sub_part(idx, axis, b)
        p.add_argument(*opts('--sub'), nargs=2, metavar=('COORD', 'DIR'),
                       action=SubDirectionGrid,
                       help='Reduce the grid by taking a subset of the grid (along DIR).')

        # Create-subsets of the grid
        class RemoveDirectionGrid(argparse.Action):

            def __call__(self, parser, ns, values, option_string=None):
                # The unit-cell direction
                axis = direction(values[1])
                # Figure out whether this is a fractional or
                # distance in Ang
                is_frac = 'f' in values[0]
                rng = strseq(float, values[0].replace('f', ''))
                if isinstance(rng, tuple):
                    # we have bounds
                    if not (rng[0] is None or rng[1] is None):
                        raise NotImplementedError('Can not figure out how to apply mid-removal of grids.')
                    if rng[0] is None:
                        idx1 = 0
                    else:
                        idx1 = ns._grid.index(rng[0], axis=axis)
                    if rng[1] is None:
                        idx2 = ns._grid.shape[axis]
                    else:
                        idx2 = ns._grid.index(rng[1], axis=axis)
                    ns._grid = ns._grid.remove(_a.arangei(idx1, idx2), axis)
                    return
                elif rng < 0.:
                    if is_frac:
                        rng = ns._grid.cell[axis, :] * abs(rng)
                    b = True
                else:
                    if is_frac:
                        rng = ns._grid.cell[axis, :] * rng
                    b = False
                idx = ns._grid.index(rng, axis=axis)
                ns._grid = ns._grid.remove_part(idx, axis, b)
        p.add_argument(*opts('--remove'), nargs=2, metavar=('COORD', 'DIR'),
                       action=RemoveDirectionGrid,
                       help='Reduce the grid by removing a subset of the grid (along DIR).')

        # Define size of grid
        class PrintInfo(argparse.Action):

            def __call__(self, parser, ns, values, option_string=None):
                ns._stored_grid = True
                print(ns._grid)
        p.add_argument(*opts('--info'), nargs=0,
                       action=PrintInfo,
                       help='Print, to stdout, some regular information about the grid.')

        class Out(argparse.Action):

            def __call__(self, parser, ns, value, option_string=None):
                if value is None:
                    return
                if len(value) == 0:
                    return
                ns._grid.write(value[0])
                # Issue to the namespace that the grid has been written, at least once.
                ns._stored_grid = True
        p.add_argument(*opts('--out', '-o'), nargs=1, action=Out,
                       help='Store the grid (at its current invocation) to the out file.')

        # If the user requests positional out arguments, we also add that.
        if kwargs.get('positional_out', False):
            p.add_argument('out', nargs='*', default=None,
                           action=Out,
                           help='Store the grid (at its current invocation) to the out file.')

        # We have now created all arguments
        return p, namespace


def sgrid(grid=None, argv=None, ret_grid=False):
    """ Main script for sgrid.

    This routine may be called with `argv` and/or a `Sile` which is the grid at hand.

    Parameters
    ----------
    grid : Grid or BaseSile
       this may either be the grid, as-is, or a `Sile` which contains
       the grid.
    argv : list of str
       the arguments passed to sgrid
    ret_grid : bool, optional
       whether the function should return the grid
    """
    import sys
    import os.path as osp
    import argparse

    from sisl.io import get_sile, BaseSile

    # The file *MUST* be the first argument
    # (except --help|-h)

    # We cannot create a separate ArgumentParser to retrieve a positional arguments
    # as that will grab the first argument for an option!

    # Start creating the command-line utilities that are the actual ones.
    description = """
This manipulation utility is highly advanced and one should note that the ORDER of
options is determining the final structure. For instance:

   $> {0} Reference.grid.nc --diff Other.grid.nc --sub 0.:0.2f z

is NOT equivalent to:

   $> {0} Reference.grid.nc --sub 0.:0.2f z --diff Other.grid.nc

This may be unexpected but enables one to do advanced manipulations.
    """.format(osp.basename(sys.argv[0]))

    if argv is not None:
        if len(argv) == 0:
            argv = ['--help']
    elif len(sys.argv) == 1:
        # no arguments
        # fake a help
        argv = ['--help']
    else:
        argv = sys.argv[1:]

    # Ensure that the arguments have pre-pended spaces
    argv = cmd.argv_negative_fix(argv)

    p = argparse.ArgumentParser('Manipulates real-space grids.',
                           formatter_class=argparse.RawDescriptionHelpFormatter,
                           description=description)

    # Add default sisl version stuff
    cmd.add_sisl_version_cite_arg(p)

    # First read the input "Sile"
    stdout_grid = True
    if grid is None:
        from os.path import isfile
        argv, input_file = cmd.collect_input(argv)

        kwargs = {}
        if input_file is None:
            stdout_grid = False
            grid = Grid(0.1, geometry=Geometry([0] * 3, sc=1))
        else:
            # Extract specification of the input file
            input_file, spec = str_spec(input_file)
            if spec is not None:
                if ',' in spec:
                    kwargs['index'] = list(map(float, spec.split(',')))
                else:
                    kwargs['index'] = int(spec)

            if isfile(input_file):
                grid = get_sile(input_file).read_grid(**kwargs)

            elif not isfile(input_file):
                from .messages import info
                info("Cannot find file '{}'!".format(input_file))

    elif isinstance(grid, BaseSile):
        # Store the input file...
        input_file = grid.file
        grid = grid.read_grid()

    # Do the argument parser
    p, ns = grid.ArgumentParser(p, **grid._ArgumentParser_args_single())

    # Now the arguments should have been populated
    # and we will sort out if the input options
    # is only a help option.
    try:
        if not hasattr(ns, '_input_file'):
            setattr(ns, '_input_file', input_file)
    except:
        pass

    # Now try and figure out the actual arguments
    p, ns, argv = cmd.collect_arguments(argv, input=False,
                                        argumentparser=p,
                                        namespace=ns)

    # We are good to go!!!
    args = p.parse_args(argv, namespace=ns)
    g = args._grid

    if stdout_grid and not args._stored_grid:
        # We should write out the information to the stdout
        # This is merely for testing purposes and may not be used for anything.
        print(g)

    if ret_grid:
        return g
    return 0
