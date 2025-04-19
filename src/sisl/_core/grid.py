# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import logging
from math import pi
from numbers import Real
from pathlib import Path
from typing import Optional

import numpy as np
from numpy import add, asarray, cos, dot, floor, int32, ogrid, sin, take
from scipy.ndimage import zoom as ndimage_zoom
from scipy.sparse import SparseEfficiencyWarning
from scipy.sparse import diags as sp_diags

import sisl._array as _a
from sisl._dispatch_class import _Dispatchs
from sisl._dispatcher import AbstractDispatch, ClassDispatcher, TypeDispatcher
from sisl._help import dtype_complex_to_float, wrap_filterwarnings
from sisl._internal import set_module
from sisl._lib._argparse import SislHelpFormatter
from sisl.messages import deprecate_argument, deprecation
from sisl.shape import Shape
from sisl.utils import (
    cmd,
    default_ArgumentParser,
    default_namespace,
    direction,
    import_attr,
    str_spec,
    strseq,
)
from sisl.utils.mathematics import fnorm

from .geometry import Geometry
from .lattice import BoundaryCondition, Lattice, LatticeChild

__all__ = ["Grid", "sgrid"]

_log = logging.getLogger(__name__)


@set_module("sisl")
class Grid(
    LatticeChild,
    _Dispatchs,
    dispatchs=[
        ClassDispatcher("new", obj_getattr="error", instance_dispatcher=TypeDispatcher),
        ClassDispatcher("to", obj_getattr="error", type_dispatcher=None),
    ],
    when_subclassing="copy",
):
    """Real-space grid information with associated geometry.

    This grid object handles cell vectors and divisions of said grid.

    Parameters
    ----------
    shape : float or (3,) of int
        the shape of the grid. A ``float`` specifies the grid spacing in Angstrom, while
        a list of integers specifies the exact grid size.
    bc : list of int (3, 2) or (3, ), optional
        the boundary conditions for each of the cell's planes. Default to periodic BC.
    lattice :
        the lattice that this grid represents. `lattice` has precedence if both `geometry` and `lattice`
        has been specified. Defaults to ``[1, 1, 1]``.
    dtype : numpy.dtype, optional
        the data-type of the grid, default to `numpy.float64`.
    geometry :
        associated geometry with the grid. If `lattice` has not been passed the lattice will
        be taken from this geometry.

    Examples
    --------
    >>> grid1 = Grid(0.1, lattice=10)
    >>> grid2 = Grid(0.1, lattice=Lattice(10))
    >>> grid3 = Grid(0.1, lattice=Lattice([10] * 3))
    >>> grid1 == grid2
    True
    >>> grid1 == grid3
    True
    >>> grid = Grid(0.1, lattice=10, dtype=np.complex128)
    >>> grid == grid1
    False

    It is possible to provide a geometry *and* a different lattice to make a smaller (or bigger) lattice
    based on a geometry. This might be useful when creating wavefunctions or expanding densities to grids.
    Here we create a square grid based on a hexagonal graphene lattice. Expanding wavefunctions from
    this ``geometry`` will automatically convert to the ``lattice`` size.
    >>> lattice = Lattice(10) # square lattice 10x10x10 Ang
    >>> geometry = geom.graphene()
    >>> grid = Grid(0.1, lattice=lattice, geometry=geometry)
    """

    #: Constant for defining a periodic boundary condition (deprecated, use Lattice.BC.PERIODIC)
    PERIODIC = BoundaryCondition.PERIODIC
    #: Constant for defining a Neumann boundary condition (deprecated, use Lattice.BC.NEUMANN)
    NEUMANN = BoundaryCondition.NEUMANN
    #: Constant for defining a Dirichlet boundary condition (deprecated, use Lattice.BC.DIRICHLET)
    DIRICHLET = BoundaryCondition.DIRICHLET
    #: Constant for defining an open boundary condition (deprecated, use Lattice.BC.OPEN)
    OPEN = BoundaryCondition.OPEN

    @deprecate_argument(
        "sc",
        "lattice",
        "argument sc has been deprecated in favor of lattice, please update your code.",
        "0.15",
        "0.17",
    )
    @deprecate_argument(
        "bc",
        None,
        "argument bc has been deprecated (removed) in favor of the boundary conditions in Lattice, please update your code.",
        "0.15",
        "0.17",
    )
    def __init__(
        self,
        shape,
        bc=None,
        lattice: Optional[Lattice] = None,
        dtype=None,
        geometry: Optional[Geometry] = None,
    ):
        self.set_lattice(None)

        # Create the atomic structure in the grid, if possible
        self.set_geometry(geometry)
        if lattice is not None:
            if bc is None:
                bc = [[self.PERIODIC] * 2] * 3
            self.set_lattice(lattice)

        if isinstance(shape, Real):
            d = (self.cell**2).sum(1) ** 0.5
            shape = list(map(int, np.rint(d / shape)))

        # Create the grid
        self.set_grid(shape, dtype=dtype)

        # Create the grid boundary conditions
        if bc is not None:
            self.lattice.set_boundary_condition(bc)

    @deprecation(
        "Grid.set_bc is deprecated since boundary conditions are moved to Lattice (see github issue #626)",
        "0.15",
        "0.17",
    )
    def set_bc(self, bc):
        self.lattice.set_boundary_condition(bc)

    @deprecation(
        "Grid.set_boundary is deprecated since boundary conditions are moved to Lattice (see github issue #626)",
        "0.15",
        "0.17",
    )
    def set_boundary(self, bc):
        self.lattice.set_boundary_condition(bc)

    @deprecation(
        "Grid.set_boundary_condition is deprecated since boundary conditions are moved to Lattice (see github issue #626)",
        "0.15",
        "0.17",
    )
    def set_boundary_condition(self, bc):
        self.lattice.set_boundary_condition(bc)

    def __getitem__(self, key):
        """Grid value at `key`"""
        return self.grid[key]

    def __setitem__(self, key, val):
        """Updates the grid contained"""
        self.grid[key] = val

    def _is_commensurate(self):
        """Determine whether the contained geometry and lattice are commensurate"""
        if self.geometry is None:
            return True
        # ideally this should be checked that they are integer equivalent
        l_len = self.lattice.length
        g_len = self.geometry.lattice.length
        reps = np.ones(3)
        for i, (l, g) in enumerate(zip(l_len, g_len)):
            if l >= g:
                reps[i] = l / g
            else:
                return False
        return np.all(abs(reps - np.round(reps)) < 1e-5)

    def set_geometry(self, geometry, also_lattice: bool = True):
        """Sets the `Geometry` for the grid.

        Setting the `Geometry` for the grid is a possibility
        to attach atoms to the grid.

        It is not a necessary entity, so passing `None` is a viable option.

        Parameters
        ----------
        geometry : Geometry or None
            specify the new geometry in the `Grid`. If ``None`` will
            remove the geometry (but not the lattice)
        also_lattice : bool, optional
            whether to also set the lattice for the grid according to the
            lattice of the `geometry`, if ``False`` it will keep the lattice
            as it was.
        """
        if geometry is None:
            # Fake geometry
            self.geometry = None
        else:
            self.geometry = geometry
            if also_lattice:
                self.set_lattice(geometry.lattice)

    def fill(self, val):
        """Fill the grid with this value

        Parameters
        ----------
        val : numpy.dtype
           all grid-points will have this value after execution
        """
        self.grid.fill(val)

    def interp(self, shape, order=1, mode="wrap", **kwargs):
        """Interpolate grid values to a new resolution (retaining lattice vectors)

        It uses the `scipy.ndimage.zoom`, which creates a finer or
        more spaced grid using spline interpolation.
        The lattice vectors remains unchanged.

        Parameters
        ----------
        shape : int, array_like of len 3
            the new shape of the grid.
        order : int 0-5, optional
            the order of the spline interpolation.
            1 means linear, 2 quadratic, etc...
        mode: {'wrap', 'mirror', 'constant', 'reflect', 'nearest'}
            determines how to compute the borders of the grid.
            The default is ``'wrap'``, which accounts for periodic conditions.
        **kwargs :
            optional arguments passed to the interpolation algorithm
            The interpolation routine is `scipy.ndimage.zoom`

        See Also
        --------
        scipy.ndimage.zoom : method used for interpolation
        """
        # For backwards compatibility
        method = kwargs.pop("method", None)
        # Maybe the method was passed as a positional argument
        if isinstance(order, str):
            method = order
        if method is not None:
            order = {"linear": 1}.get(method, 3)

        # And now we do the actual interpolation
        # Calculate the zoom_factors
        zoom_factors = _a.arrayd(shape) / self.shape

        # Apply the scipy.ndimage.zoom function and return a new grid
        return self.apply(ndimage_zoom, zoom_factors, mode=mode, order=order, **kwargs)

    def isosurface(self, level: float, step_size: int = 1, **kwargs):
        """Calculates the isosurface for a given value

        It uses `skimage.measure.marching_cubes`, so you need to have scikit-image installed.

        Parameters
        ----------
        level:
            contour value to search for isosurfaces in the grid.
            If not given or None, the average of the min and max of the grid is used.
        step_size:
            step size in voxels. Larger steps yield faster but coarser results.
            The result will always be topologically correct though.
        **kwargs:
            optional arguments passed directly to `skimage.measure.marching_cubes`
            for the calculation of isosurfaces.

        Returns
        ----------
        numpy array of shape (V, 3)
            Verts. Spatial coordinates for V unique mesh vertices.

        numpy array of shape (n_faces, 3)
            Faces. Define triangular faces via referencing vertex indices from verts.
            This algorithm specifically outputs triangles, so each face has exactly three indices.

        numpy array of shape (V, 3)
            Normals. The normal direction at each vertex, as calculated from the data.

        numpy array of shape (V, 3)
            Values. Gives a measure for the maximum value of the data in the local region near each vertex.
            This can be used by visualization tools to apply a colormap to the mesh.

        See Also
        --------
        skimage.measure.marching_cubes : method used to calculate the isosurface.
        """
        try:
            import skimage.measure
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                f"{self.__class__.__name__}.isosurface requires scikit-image to be installed"
            )

        if np.iscomplexobj(self.grid):
            raise NotImplementedError(
                f"{self.__class__.__name__}.isosurface requires real grid values."
            )

        # Run the marching cubes algorithm to calculate the vertices and faces
        # of the requested isosurface.
        verts, *returns = skimage.measure.marching_cubes(
            self.grid, level=level, step_size=step_size, **kwargs
        )

        # The verts cordinates are in fractional coordinates of unit-length.
        verts = self.index2xyz(verts)

        return (verts, *returns)

    def smooth(self, r=0.7, method="gaussian", mode="wrap", **kwargs):
        """Make a smoother grid by applying a filter

        Parameters
        -----------
        r: float or array-like of len 3, optional
            the radius of the filter in Angstrom for each axis.
            If the method is ``"gaussian"``, this is the standard deviation!

            If a single float is provided, then the same distance will be used for all axes.
        method: {'gaussian', 'uniform'}
            the type of filter to apply to smoothen the grid.
        mode: {'wrap', 'mirror', 'constant', 'reflect', 'nearest'}
            determines how to compute the borders of the grid.
            The default is wrap, which accounts for periodic conditions.

        See Also
        --------
        scipy.ndimage.gaussian_filter
        """

        # Normalize the radius input to a list of radius
        if isinstance(r, Real):
            r = [r, r, r]

        # Calculate the size of the kernel in pixels (in case the
        # gaussian filter is used, this is the standard deviation)
        pixels_r = np.round(r / fnorm(self.dcell)).astype(np.int32)

        # Update the kwargs accordingly
        if method == "gaussian":
            kwargs["sigma"] = pixels_r
        elif method == "uniform":
            kwargs["size"] = pixels_r * 2

        # This should raise an import error if the method does not exist
        func = import_attr(f"scipy.ndimage.{method}_filter")
        return self.apply(func, mode=mode, **kwargs)

    @property
    def size(self):
        """Total number of elements in the grid"""
        return np.prod(self.grid.shape)

    @property
    def shape(self):
        r"""Grid shape along the lattice vectors"""
        return self.grid.shape

    @property
    def dtype(self):
        """Data-type used in grid"""
        return self.grid.dtype

    @property
    def dkind(self):
        """The data-type of the grid (in str)"""
        return np.dtype(self.grid.dtype).kind

    def set_grid(self, shape, dtype=None):
        """Create the internal grid of certain size."""
        shape = _a.asarrayi(shape).ravel()
        if dtype is None:
            dtype = np.float64
        if shape.size != 3:
            raise ValueError(
                f"{self.__class__.__name__}.set_grid requires shape to be of length 3"
            )
        self.grid = np.zeros(shape, dtype=dtype)

    def _sc_geometry_dict(self):
        """Internal routine for copying the Lattice and Geometry"""
        d = dict()
        d["lattice"] = self.lattice.copy()
        if not self.geometry is None:
            d["geometry"] = self.geometry.copy()
        return d

    @property
    def dcell(self):
        """Voxel cell size"""
        # Calculate the grid-distribution
        return self.cell / _a.asarrayi(self.shape).reshape(3, 1)

    @property
    def dvolume(self):
        """Volume of the grid voxel elements"""
        return self.lattice.volume / self.size

    def _copy_sub(self, n, axis, scale_geometry=False):
        # First calculate the new shape
        shape = list(self.shape)
        cell = np.copy(self.cell)
        # Down-scale cell
        cell[axis, :] = (cell[axis, :] / shape[axis]) * n
        shape[axis] = n
        if n < 1:
            raise ValueError("You cannot retain no indices.")
        grid = self.__class__(shape, dtype=self.dtype, **self._sc_geometry_dict())
        # Update cell shape (the cell is smaller now)
        grid.set_lattice(cell)
        if scale_geometry and not self.geometry is None:
            geom = self.geometry.copy()
            fxyz = geom.fxyz.copy()
            geom.set_lattice(grid.lattice)
            geom.xyz[:, :] = np.dot(fxyz, grid.lattice.cell)
            grid.set_geometry(geom)

        return grid

    def cross_section(self, idx, axis):
        """Takes a cross-section of the grid along axis `axis`

        Remark: This API entry might change to handle arbitrary
        cuts via rotation of the axis"""
        idx = _a.asarrayi(idx).ravel()
        grid = self._copy_sub(1, axis)

        if axis == 0:
            grid.grid[:, :, :] = self.grid[idx, :, :]
        elif axis == 1:
            grid.grid[:, :, :] = self.grid[:, idx, :]
        elif axis == 2:
            grid.grid[:, :, :] = self.grid[:, :, idx]
        else:
            raise ValueError(f"Unknown axis specification in cross_section {axis}")

        return grid

    def sum(self, axis):
        """Sum grid values along axis `axis`.

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
        """Average grid values along direction `axis`.

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
            raise ValueError(
                f"{self.__class__.__name__}.average requires axis to be in [0, 1, 2]"
            )

        return grid

    # for compatibility
    mean = average

    def remove_part(self, idx, axis, above):
        """Removes parts of the grid via above/below designations.

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
        """Retains parts of the grid via above/below designations.

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

    def index2xyz(self, index):
        """Real-space coordinates of indices related to the grid

        Parameters
        ----------
        index : array_like
           indices for grid-positions

        Returns
        -------
        numpy.ndarray
           coordinates of the indices with respect to this grid spacing
        """
        return asarray(index).dot(self.dcell)

    def index_fold(self, index, unique=True):
        """Converts indices from *any* placement to only exist in the "primary" grid

        Examples
        --------
        >>> grid = Grid([10, 10, 10])
        >>> assert np.all(grid.index_fold([-1, -1, -1]) == 9)

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

        See Also
        --------
        index_truncate : truncate indices by removing indices outside the primary cell
        """
        index = _a.asarrayi(index)
        ndim = index.ndim

        # Convert to internal
        if unique:
            index = np.unique(
                index.reshape(-1, 3) % _a.asarrayi(self.shape)[None, :], axis=0
            )
        else:
            index = index.reshape(-1, 3) % _a.asarrayi(self.shape)[None, :]

        if ndim == 1:
            return index.ravel()
        return index

    def index_truncate(self, index):
        """Remove indices from *outside* the grid to only retain indices in the "primary" grid

        Examples
        --------
        >>> grid = Grid([10, 10, 10])
        >>> assert len(grid.index_truncate([-1, -1, -1])) == 0

        Parameters
        ----------
        index : array_like
           indices for grid-positions

        Returns
        -------
        numpy.ndarray
            all indices are then within the shape of the grid (others have been removed

        See Also
        --------
        index_fold : fold indices into the primary cell
        """
        index = _a.asarrayi(index)
        ndim = index.ndim

        index.shape = (-1, 3)
        log_and_reduce = np.logical_and.reduce
        index = index[log_and_reduce(0 <= index, axis=1), :]
        s = _a.asarrayi(self.shape).reshape(1, 3)
        index = index[log_and_reduce(index < s, axis=1), :]

        if ndim == 1:
            return index.ravel()
        return index

    def _index_shape(self, shape):
        """Internal routine for shape-indices"""
        # First grab the sphere, subsequent indices will be reduced
        # by the actual shape
        cuboid = shape.to.Cuboid()
        ellipsoid = shape.to.Ellipsoid()
        if ellipsoid.volume > cuboid.volume:
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
        """Internal routine for cuboid shape-indices"""
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
        rxyz[:, i : i + sa * sb, :] += plane(a, b)
        i += sa * sb
        rxyz[:, i : i + sa * sc, :] += plane(a, c)
        i += sa * sc
        rxyz[:, i : i + sb * sc, :] += plane(b, c)
        del a, b, c, sa, sb, sc
        rxyz.shape = (-1, 3)

        # Get all indices of the cuboid planes
        return self.index(rxyz)

    def _index_shape_ellipsoid(self, ellipsoid):
        """Internal routine for ellipsoid shape-indices"""
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
        """Find the grid index for a given coordinate (possibly only along a given lattice vector `axis`)

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
        if coord.size == 1:  # float
            if axis is None:
                raise ValueError(
                    f"{self.__class__.__name__}.index requires the "
                    "coordinate to be 3 values when an axis has not "
                    "been specified."
                )

            c = self.dcell[axis]
            c = (c @ c) ** 0.5
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
                return (
                    floor(dot(icell, coord.reshape(-1, 3).T) * shape)
                    .reshape(3)
                    .astype(int32, copy=False)
                )
            else:
                return floor(dot(icell, coord.reshape(-1, 3).T) * shape).T.astype(
                    int32, copy=False
                )
        if ndim == 1:
            return floor(
                dot(icell[axis, :], coord.reshape(-1, 3).T) * shape[axis]
            ).astype(int32, copy=False)[0]
        else:
            return floor(
                dot(icell[axis, :], coord.reshape(-1, 3).T) * shape[axis]
            ).T.astype(int32, copy=False)

    @staticmethod
    def read(sile, *args, **kwargs):
        """Reads grid from the `Sile` using `read_grid`

        Parameters
        ----------
        sile : Sile, str or pathlib.Path
            a `Sile` object which will be used to read the grid
            if it is a string it will create a new sile using `get_sile`.
        * : args passed directly to ``read_grid(,**)``
        """
        # This only works because, they *must*
        # have been imported previously
        from sisl.io import BaseSile, get_sile

        if isinstance(sile, BaseSile):
            return sile.read_grid(*args, **kwargs)
        else:
            sile = str(sile)
            sile, spec = str_spec(sile)
            if spec is not None:
                if "," in spec:
                    kwargs["index"] = list(map(float, spec.split(",")))
                else:
                    kwargs["index"] = int(spec)
            with get_sile(sile, mode="r") as fh:
                return fh.read_grid(*args, **kwargs)

    def __str__(self):
        """String of object"""
        s = "{name}{{kind: {kind}, shape: [{shape[0]} {shape[1]} {shape[2]}],\n".format(
            kind=self.dkind, shape=self.shape, name=self.__class__.__name__
        )
        if self._is_commensurate() and self.geometry is not None:
            l = np.round(self.lattice.length / self.geometry.lattice.length).astype(
                np.int32
            )
            s += f"commensurate: [{l[0]} {l[1]} {l[2]}]"
        else:
            s += "{}".format(str(self.lattice).replace("\n", "\n "))
        if not self.geometry is None:
            s += ",\n {}".format(str(self.geometry).replace("\n", "\n "))
        return f"{s}\n}}"

    def _check_compatibility(self, other, msg):
        """Internal check for asserting two grids are commensurable"""
        if self == other:
            return True
        s1 = str(self)
        s2 = str(other)
        raise ValueError(f"Grids are not compatible, {s1}-{s2}. {msg}")

    def _compatible_copy(self, other, *args, **kwargs):
        """Internally used copy function that also checks whether the two grids are compatible"""
        if isinstance(other, Grid):
            self._check_compatibility(other, *args, **kwargs)
        return self.copy()

    def __eq__(self, other):
        """Whether two grids are commensurable (no value checks, only grid shape)

        There will be no check of the values _on_ the grid."""
        return self.shape == other.shape

    def __ne__(self, other):
        """Whether two grids are incommensurable (no value checks, only grid shape)"""
        return not (self == other)

    def __abs__(self):
        r"""Take the absolute value of the grid :math:`|\mathrm{grid}|`"""
        dtype = dtype_complex_to_float(self.dtype)
        a = self.copy()
        a.grid = np.absolute(self.grid).astype(dtype, copy=False)
        return a

    def __add__(self, other):
        """Add two grid values (or add a single value to all grid values)

        Raises
        ------
        ValueError
            if the grids are not compatible (different shapes)
        """
        if isinstance(other, Grid):
            grid = self._compatible_copy(other, "they cannot be added")
            grid.grid = self.grid + other.grid
        else:
            grid = self.copy()
            grid.grid = self.grid + other
        return grid

    def __iadd__(self, other):
        """Add, in-place, values from another grid

        Raises
        ------
        ValueError
            if the grids are not compatible (different shapes)
        """
        if isinstance(other, Grid):
            self._check_compatibility(other, "they cannot be added")
            self.grid += other.grid
        else:
            self.grid += other
        return self

    def __sub__(self, other):
        """Subtract two grid values (or subtract a single value from all grid values)

        Raises
        ------
        ValueError
            if the grids are not compatible (different shapes)
        """
        if isinstance(other, Grid):
            grid = self._compatible_copy(other, "they cannot be subtracted")
            np.subtract(self.grid, other.grid, out=grid.grid)
        else:
            grid = self.copy()
            np.subtract(self.grid, other, out=grid.grid)
        return grid

    def __isub__(self, other):
        """Subtract, in-place, values from another grid

        Raises
        ------
        ValueError
            if the grids are not compatible (different shapes)
        """
        if isinstance(other, Grid):
            self._check_compatibility(other, "they cannot be subtracted")
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
            grid = self._compatible_copy(other, "they cannot be divided")
            np.divide(self.grid, other.grid, out=grid.grid)
        else:
            grid = self.copy()
            np.divide(self.grid, other, out=grid.grid)
        return grid

    def __itruediv__(self, other):
        if isinstance(other, Grid):
            self._check_compatibility(other, "they cannot be divided")
            self.grid /= other.grid
        else:
            self.grid /= other
        return self

    def __mul__(self, other):
        if isinstance(other, Grid):
            grid = self._compatible_copy(other, "they cannot be multiplied")
            np.multiply(self.grid, other.grid, out=grid.grid)
        else:
            grid = self.copy()
            np.multiply(self.grid, other, out=grid.grid)
        return grid

    def __imul__(self, other):
        if isinstance(other, Grid):
            self._check_compatibility(other, "they cannot be multiplied")
            self.grid *= other.grid
        else:
            self.grid *= other
        return self

    # Here comes additional supplementary routines which enables an easy
    # work-through case with other programs.
    @classmethod
    def mgrid(cls, *slices):
        """Return a list of indices corresponding to the slices

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
        r"""Calculate `pyamg` matrix indices from a list of grid indices

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
        ValueError
            if any of the passed indices are below 0 or above the number of elements per axis
        """
        index = _a.asarrayi(index).reshape(-1, 3)
        grid = _a.arrayi(self.shape[:])
        if np.any(index < 0) or np.any(index >= grid.reshape(1, 3)):
            raise ValueError(
                f"{self.__class__.__name__}.pyamg_index erroneous values for grid indices"
            )
        # Skipping factor per element
        cp = _a.arrayi([[grid[1] * grid[2], grid[2], 1]])
        return (cp * index).sum(1)

    @classmethod
    def pyamg_source(cls, b, pyamg_indices, value):
        r"""Fix the source term to `value`.

        Parameters
        ----------
        b : numpy.ndarray
           a vector containing RHS of :math:`\mathbf A \mathbf x = \mathbf b` for the solution of the grid stencil
        pyamg_indices : list of int
           the linear pyamg matrix indices where the value of the grid is fixed. I.e. the indices should
           correspond to returned quantities from `pyamg_indices`.
        """
        b[pyamg_indices] = value

    def pyamg_fix(self, A, b, pyamg_indices, value):
        r"""Fix values for the stencil to `value`.

        Parameters
        ----------
        A : `~scipy.sparse.csr_matrix`/`~scipy.sparse.csc_matrix`
           sparse matrix describing the LHS for the linear system of equations
        b : numpy.ndarray
           a vector containing RHS of :math:`\mathbf A \mathbf x = \mathbf b` for the solution of the grid stencil
        pyamg_indices : list of int
           the linear pyamg matrix indices where the value of the grid is fixed. I.e. the indices should
           correspond to returned quantities from `pyamg_indices`.
        value : float
           the value of the grid to fix the value at
        """
        if not A.format in ("csc", "csr"):
            raise ValueError(
                f"{self.__class__.__name__}.pyamg_fix only works for csr/csc sparse matrices"
            )

        # Clean all couplings between the respective indices and all other data
        s = _a.array_arange(A.indptr[pyamg_indices], A.indptr[pyamg_indices + 1])
        A.data[s] = 0.0
        # clean-up
        del s

        # Specify that these indices are not to be tampered with
        d = np.zeros(A.shape[0], dtype=A.dtype)
        d[pyamg_indices] = 1.0
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
    def pyamg_boundary_condition(self, A, b):
        r"""Attach boundary conditions to the `pyamg` grid-matrix `A` with default boundary conditions as specified for this `Grid`

        Parameters
        ----------
        A : scipy.sparse.csr_matrix
           sparse matrix describing the grid
        b : numpy.ndarray
           a vector containing RHS of :math:`\mathbf A \mathbf x = \mathbf b` for the solution of the grid stencil
        """

        def Neumann(idx_bc, idx_p1):
            # Set all boundary equations to 0
            s = _a.array_arange(A.indptr[idx_bc], A.indptr[idx_bc + 1])
            A.data[s] = 0
            # force the boundary cells to equal the neighboring cell
            A[idx_bc, idx_bc] = 1
            A[idx_bc, idx_p1] = -1
            A.eliminate_zeros()
            b[idx_bc] = 0.0

        def Dirichlet(idx):
            # Default pyamg Poisson matrix has Dirichlet BC
            b[idx] = 0.0

        def Periodic(idx1, idx2):
            A[idx1, idx2] = -1
            A[idx2, idx1] = -1

        def sl2idx(sl):
            return self.pyamg_index(self.mgrid(sl))

        for i in range(3):
            # We have a periodic direction
            # Create slices
            sl = [slice(0, g) for g in self.shape]

            # LOWER BOUNDARY
            bci = self.lattice.boundary_condition[i]
            sl[i] = slice(0, 1)
            idx1 = sl2idx(sl)  # lower

            bc = bci[0]
            if bci[0] == self.PERIODIC:
                sl[i] = slice(self.shape[i] - 1, self.shape[i])
                idx2 = sl2idx(sl)  # upper
                Periodic(idx1, idx2)
                del idx2
                # rest has been parsed as well
                continue

            if bc == self.NEUMANN:
                # Retrieve next index
                sl[i] = slice(1, 2)
                idx2 = sl2idx(sl)  # lower + 1
                Neumann(idx1, idx2)
                del idx2
            elif bc == self.DIRICHLET:
                Dirichlet(idx1)

            # UPPER BOUNDARY
            bc = bci[1]
            sl[i] = slice(self.shape[i] - 1, self.shape[i])
            idx1 = sl2idx(sl)  # upper

            if bc == self.NEUMANN:
                # Retrieve next index
                sl[i] = slice(self.shape[i] - 2, self.shape[i] - 1)
                idx2 = sl2idx(sl)  # upper - 1
                Neumann(idx1, idx2)
                del idx2
            elif bc == self.DIRICHLET:
                Dirichlet(idx1)

        A.eliminate_zeros()

    @deprecation(
        "Grid.topyamg is deprecated in favor of Grid.to.pyamg",
        "0.15",
        "0.17",
    )
    def topyamg(self, dtype=None):
        r"""Create a `pyamg` stencil matrix to be used in pyamg

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
        A = poisson(self.shape, dtype=dtype, format="csr")
        b = np.zeros(A.shape[0], dtype=A.dtype)

        # Now apply the boundary conditions
        self.pyamg_boundary_condition(A, b)
        return A, b

    @classmethod
    def _ArgumentParser_args_single(cls):
        """Returns the options for `Grid.ArgumentParser` in case they are the only options"""
        return {
            "limit_arguments": False,
            "short": True,
            "positional_out": True,
        }

    # Hook into the Grid class to create
    # an automatic ArgumentParser which makes actions
    # as the options are read.
    @default_ArgumentParser(description="Manipulate a Grid object in sisl.")
    def ArgumentParser(self, p=None, *args, **kwargs):
        """Create and return a group of argument parsers which manipulates it self `Grid`.

        Parameters
        ----------
        p : ArgumentParser, None
           in case the arguments should be added to a specific parser. It defaults
           to create a new.
        limit_arguments : bool, True
           If `False` additional options will be created which are similar to other options.
        short : bool, False
           Create short options for a selected range of options
        positional_out : bool, False
           If `True`, adds a positional argument which acts as --out. This may be handy if only the geometry is in the argument list.
        """
        short = kwargs.get("short", False)

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

        p.add_argument(
            *opts("--geometry", "-G"),
            action=SetGeometry,
            help="Define the geometry attached to the Grid.",
        )

        # subtract another grid
        # They *MUST* be comensurate.
        class DiffGrid(argparse.Action):
            def __call__(self, parser, ns, value, option_string=None):
                grid = Grid.read(value)
                ns._grid -= grid

        p.add_argument(
            *opts("--diff", "-d"),
            action=DiffGrid,
            help="Subtract another grid (they must be commensurate).",
        )

        class AverageGrid(argparse.Action):
            def __call__(self, parser, ns, value, option_string=None):
                ns._grid = ns._grid.average(direction(value))

        p.add_argument(
            *opts("--average"),
            metavar="DIR",
            action=AverageGrid,
            help="Take the average of the grid along DIR.",
        )

        class SumGrid(argparse.Action):
            def __call__(self, parser, ns, value, option_string=None):
                ns._grid = ns._grid.sum(direction(value))

        p.add_argument(
            *opts("--sum"),
            metavar="DIR",
            action=SumGrid,
            help="Take the sum of the grid along DIR.",
        )

        # Create-subsets of the grid
        class SubDirectionGrid(argparse.Action):
            def __call__(self, parser, ns, values, option_string=None):
                # The unit-cell direction
                axis = direction(values[1])
                # Figure out whether this is a fractional or
                # distance in Ang
                is_frac = "f" in values[0]
                rng = strseq(float, values[0].replace("f", ""))
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
                elif rng < 0.0:
                    if is_frac:
                        rng = ns._grid.cell[axis, :] * abs(rng)
                    b = False
                else:
                    if is_frac:
                        rng = ns._grid.cell[axis, :] * rng
                    b = True
                idx = ns._grid.index(rng, axis=axis)
                ns._grid = ns._grid.sub_part(idx, axis, b)

        p.add_argument(
            *opts("--sub"),
            nargs=2,
            metavar=("COORD", "DIR"),
            action=SubDirectionGrid,
            help="Reduce the grid by taking a subset of the grid (along DIR).",
        )

        # Create-subsets of the grid
        class RemoveDirectionGrid(argparse.Action):
            def __call__(self, parser, ns, values, option_string=None):
                # The unit-cell direction
                axis = direction(values[1])
                # Figure out whether this is a fractional or
                # distance in Ang
                is_frac = "f" in values[0]
                rng = strseq(float, values[0].replace("f", ""))
                if isinstance(rng, tuple):
                    # we have bounds
                    if not (rng[0] is None or rng[1] is None):
                        raise NotImplementedError(
                            "Can not figure out how to apply mid-removal of grids."
                        )
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
                elif rng < 0.0:
                    if is_frac:
                        rng = ns._grid.cell[axis, :] * abs(rng)
                    b = True
                else:
                    if is_frac:
                        rng = ns._grid.cell[axis, :] * rng
                    b = False
                idx = ns._grid.index(rng, axis=axis)
                ns._grid = ns._grid.remove_part(idx, axis, b)

        p.add_argument(
            *opts("--remove"),
            nargs=2,
            metavar=("COORD", "DIR"),
            action=RemoveDirectionGrid,
            help="Reduce the grid by removing a subset of the grid (along DIR).",
        )

        class Tile(argparse.Action):
            def __call__(self, parser, ns, values, option_string=None):
                r = int(values[0])
                d = direction(values[1])
                ns._grid = ns._grid.tile(r, d)

        p.add_argument(
            *opts("--tile"),
            nargs=2,
            metavar=("TIMES", "DIR"),
            action=Tile,
            help="Tiles the grid in the specified direction.",
        )

        # Scale the grid with this value
        class ScaleGrid(argparse.Action):
            def __call__(self, parser, ns, value, option_string=None):
                ns._grid.grid *= value

        p.add_argument(
            *opts("--scale", "-S"),
            type=float,
            action=ScaleGrid,
            help="Scale grid values with a factor",
        )

        # Define size of grid
        class InterpGrid(argparse.Action):
            def __call__(self, parser, ns, values, option_string=None):
                def _conv_shape(length, value):
                    if "." in value:
                        return int(round(length / float(value)))
                    return int(value)

                shape = list(map(_conv_shape, ns._grid.lattice.length, values[:3]))
                # shorten list for easier arguments
                values = values[3:]
                if len(values) > 0:
                    values[0] = int(values[0])
                ns._grid = ns._grid.interp(shape, *values)

        p.add_argument(
            *opts("--interp"),
            nargs="*",
            metavar="NX NY NZ *ORDER *MODE",
            action=InterpGrid,
            help="""Interpolate grid for higher or lower density (minimum 3 arguments)
Requires at least 3 arguments, number of points along 1st, 2nd and 3rd lattice vector. These may contain a "." to signal a distance in angstrom of each voxel.
For instance --interp 0.1 10 100 will result in an interpolated shape of [nint(grid.lattice.length / 0.1), 10, 100].

The 4th optional argument is the order of interpolation; an integer 0<=i<=5 (default 1)
The 5th optional argument is the mode to interpolate; wrap/mirror/constant/reflect/nearest
""",
        )

        # Smoothen the grid
        class SmoothGrid(argparse.Action):
            def __call__(self, parser, ns, values, option_string=None):
                if len(values) > 0:
                    values[0] = float(values[0])
                ns._grid = ns._grid.smooth(*values)

        p.add_argument(
            *opts("--smooth"),
            nargs="*",
            metavar="*R *METHOD *MODE",
            action=SmoothGrid,
            help="""Smoothen grid values according to methods by applying a filter, all arguments are optional.
The 1st argument is the radius of the filter for smoothening, a larger value means a larger volume which is agglomerated
The 2nd argument is the method to use; gaussian/uniform
The 3rd argument is the mode to use; wrap/mirror/constant/reflect/nearest
""",
        )

        class PrintInfo(argparse.Action):
            def __call__(self, parser, ns, values, option_string=None):
                ns._stored_grid = True
                print(ns._grid)

        p.add_argument(
            *opts("--info"),
            nargs=0,
            action=PrintInfo,
            help="Print, to stdout, some regular information about the grid.",
        )

        class Plot(argparse.Action):
            def __call__(self, parser, ns, values, option_string=None):
                ns._stored_grid = True
                import matplotlib.pyplot as plt

                grid = ns._grid

                axs = []
                idx = []
                for ax in (0, 1, 2):
                    shape = grid.shape[ax]
                    if shape > 1:
                        axs.append(
                            np.linspace(
                                0, grid.lattice.length[ax], shape, endpoint=False
                            )
                        )
                        idx.append(ax)

                # Now plot data
                if len(idx) == 3:
                    raise ValueError("Cannot plot a 3D grid (yet!)")
                elif len(idx) == 2:
                    X, Y = np.meshgrid(*axs)
                    plt.contourf(X, Y, np.squeeze(grid.grid).T)
                    plt.xlabel(f"Distance along {'ABC'[idx[0]]} [Ang]")
                    plt.ylabel(f"Distance along {'ABC'[idx[1]]} [Ang]")
                elif len(idx) == 1:
                    plt.plot(axs[0], grid.grid.ravel())
                    plt.xlabel(f"Distance along {'ABC'[idx[0]]} [Ang]")
                    plt.ylabel(f"Arbitrary unit")
                plt.show()

        p.add_argument(
            *opts("--plot", "-P"),
            nargs=0,
            action=Plot,
            help="Plot the grid (currently only enabled if at least one dimension has been averaged out",
        )

        class Out(argparse.Action):
            def __call__(self, parser, ns, value, option_string=None):
                if value is None:
                    return
                if len(value) == 0:
                    return
                from sisl.io import get_sile

                grid = ns._grid

                # determine whether the write-out file has *write_grid* as a methd
                # if not, and the grid only have 1 dimension, we allow it to be
                # written to a datafile
                sile = get_sile(value[0], "w")
                if hasattr(sile, "write_grid"):
                    grid.write(sile)
                elif np.prod(grid.shape) == np.amax(grid.shape):
                    # this means that 2 dimensions have a length of 1
                    # figure out which dimensions it is and add calculate
                    # the distance along the lattice vector
                    idx = np.argmax(grid.shape)

                    dx = np.linspace(
                        0, grid.lattice.length[idx], grid.shape[idx], endpoint=False
                    )
                    sile.write_data(dx, grid.grid.ravel())
                else:
                    raise ValueError(
                        f"""Either of these two cases are not fullfilled:

1. {sile} do not have the `write_grid` method

2. The grid is not 1D data; averaged or summed along 2 directions."""
                    )

                # Issue to the namespace that the grid has been written, at least once.
                ns._stored_grid = True

        p.add_argument(
            *opts("--out", "-o"),
            nargs=1,
            action=Out,
            help="Store the grid (at its current invocation) to the out file.",
        )

        # If the user requests positional out arguments, we also add that.
        if kwargs.get("positional_out", False):
            p.add_argument(
                "out",
                nargs="*",
                default=None,
                action=Out,
                help="Store the grid (at its current invocation) to the out file.",
            )

        # We have now created all arguments
        return p, namespace


new_dispatch = Grid.new
to_dispatch = Grid.to


# Define base-class for this
class GridNewDispatch(AbstractDispatch):
    """Base dispatcher from class passing arguments to Grid class

    This forwards all `__call__` calls to `dispatch`
    """

    def __call__(self, *args, **kwargs):
        return self.dispatch(*args, **kwargs)


class GridNewGridDispatch(GridNewDispatch):
    def dispatch(self, grid, copy=False):
        """Return Grid, for sanitization purposes"""
        cls = self._get_class()
        if cls != grid.__class__:
            out = cls(shape=grid.shape, lattice=grid.lattice, geometry=grid.geometry)
            out.grid = grid.grid.copy()
            grid = out
            copy = False
        if copy:
            return grid.copy()
        return grid


new_dispatch.register(Grid, GridNewGridDispatch)


class GridNewFileDispatch(GridNewDispatch):
    def dispatch(self, *args, **kwargs):
        """Defer the `Grid.read` method by passing down arguments"""
        # can work either on class or instance
        cls = self._get_class()
        return cls.read(*args, **kwargs)


new_dispatch.register(str, GridNewFileDispatch)
new_dispatch.register(Path, GridNewFileDispatch)


class GridToDispatch(AbstractDispatch):
    """Base dispatcher from class passing from Grid class"""


class GridToSileDispatch(GridToDispatch):
    def dispatch(self, *args, **kwargs):
        grid = self._get_object()
        return grid.write(*args, **kwargs)


to_dispatch.register("str", GridToSileDispatch)
to_dispatch.register("Path", GridToSileDispatch)
# to do grid.to[Path](path)
to_dispatch.register(str, GridToSileDispatch)
to_dispatch.register(Path, GridToSileDispatch)


class GridTopyamgDispatch(GridToDispatch):
    def dispatch(self, dtype=None):
        grid = self._get_object()
        from pyamg.gallery import poisson

        if dtype is None:
            dtype = grid.dtype
        # Initially create the CSR matrix
        A = poisson(grid.shape, dtype=dtype, format="csr")
        b = np.zeros(A.shape[0], dtype=A.dtype)

        # Now apply the boundary conditions
        grid.pyamg_boundary_condition(A, b)
        return A, b


to_dispatch.register("pyamg", GridTopyamgDispatch)


# Clean up
del new_dispatch, to_dispatch


@set_module("sisl")
def sgrid(grid=None, argv=None, ret_grid=False):
    """Main script for sgrid.

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
    import argparse
    import sys
    from pathlib import Path

    from sisl.io import BaseSile

    # The file *MUST* be the first argument
    # (except --help|-h)
    exe = Path(sys.argv[0]).name

    # We cannot create a separate ArgumentParser to retrieve a positional arguments
    # as that will grab the first argument for an option!

    # Start creating the command-line utilities that are the actual ones.
    description = """
This manipulation utility is highly advanced and one should note that the ORDER of
options is determining the final structure. For instance:

   {exe} Reference.grid.nc --diff Other.grid.nc --sub 0.:0.2f z

is NOT equivalent to:

   {exe} Reference.grid.nc --sub 0.:0.2f z --diff Other.grid.nc

This may be unexpected but enables one to do advanced manipulations.
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
    stdout_grid = True
    if grid is None:

        argv, input_file = cmd.collect_input(argv)

        kwargs = {}
        if input_file is None:
            stdout_grid = False
            grid = Grid(0.1, geometry=Geometry([0] * 3, lattice=1))
        else:
            grid = Grid.read(input_file)

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
    g = args._grid

    if stdout_grid and not args._stored_grid:
        # We should write out the information to the stdout
        # This is merely for testing purposes and may not be used for anything.
        print(g)

    if ret_grid:
        return g
    return 0
