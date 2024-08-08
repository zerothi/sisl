# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal, Optional, Union

import numpy as np
import numpy.typing as npt
from scipy.ndimage import affine_transform
from xarray import DataArray

from sisl import _array as _a
from sisl._core import Geometry, Grid
from sisl._core._lattice import cell_invert

from .cell import infer_cell_axes, is_1D_cartesian, is_cartesian_unordered

# from ...types import Axis, PathLike
# from ..data_sources import DataSource


def get_grid_representation(
    grid: Grid,
    represent: Literal["real", "imag", "mod", "phase", "rad_phase", "deg_phase"],
) -> Grid:
    """Returns a representation of the grid

    Parameters
    ------------
    grid:
        the grid for which we want return
    represent: {"real", "imag", "mod", "phase", "deg_phase", "rad_phase"}
        the type of representation. "phase" is equivalent to "rad_phase"

    Returns
    ------------
    sisl.Grid
    """

    # note we have removed the typing of np.float_ etc since numpy 2.0.0
    # removed this. One should be explicit, and this would mean writing
    # all possibilities, it is not worth the effort here.
    def _func(
        values: npt.NDArray,  # *any* data type, int, float, complex
    ) -> npt.NDArray:
        if represent == "real":
            new_values = values.real
        elif represent == "imag":
            new_values = values.imag
        elif represent == "mod":
            new_values = np.absolute(values)
        elif represent in ["phase", "rad_phase", "deg_phase"]:
            new_values = np.angle(values, deg=represent.startswith("deg"))
        else:
            raise ValueError(
                f"'{represent}' is not a valid value for the `represent` argument"
            )

        return new_values

    return grid.apply(_func)


def tile_grid(grid: Grid, nsc: tuple[int, int, int] = (1, 1, 1)) -> Grid:
    """Tiles the grid"""
    for ax, reps in enumerate(nsc):
        grid = grid.tile(reps, ax)
    return grid


def transform_grid_cell(
    grid: Grid,
    cell: npt.NDArray[np.float64] = np.eye(3),
    output_shape: Optional[nuple[int, int, int]] = None,
    mode: str = "constant",
    order: int = 1,
    **kwargs,
) -> Grid:
    """Applies a linear transformation to the grid to get it relative to an arbitrary cell.

    This method can be used, for example to get the values of the grid with respect to
    the standard basis, so that you can easily visualize it or overlap it with other grids
    (e.g. to perform integrals).

    Parameters
    -----------
    cell: array-like of shape (3,3)
        these cell represent the directions that you want to use as references for
        the new grid.

        The length of the axes does not have any effect! They will be rescaled to create
        the minimum bounding box necessary to accomodate the unit cell.
    output_shape: array-like of int of shape (3,), optional
        the shape of the final output. If not provided, the current shape of the grid
        will be used.

        Notice however that if the transformation applies a big shear to the image (grid)
        you will probably need to have a bigger output_shape.
    mode: str, optional
        determines how to handle borders. See scipy docs for more info on the possible values.
    order : int 0-5, optional
        the order of the spline interpolation to calculate the values (since we are applying
        a transformation, we don't actually have values for the new locations and we need to
        interpolate them)
        1 means linear, 2 quadratic, etc...
    **kwargs:
        the rest of keyword arguments are passed directly to `scipy.ndimage.affine_transform`

    See also
    ----------
    scipy.ndimage.affine_transform : method used to apply the linear transformation.
    """
    # Take the current shape of the grid if no output shape was provided
    if output_shape is None:
        output_shape = grid.shape

    # Make sure the cell has type float
    cell = np.asarray(cell, dtype=float)

    # Get the current cell in coordinates of the destination axes
    inv_cell = cell_invert(cell).T
    projected_cell = grid.cell.dot(inv_cell)

    # From that, infere how long will the bounding box of the cell be
    lengths = abs(projected_cell).sum(axis=0)

    # Create the transformation matrix. Since we want to control the shape
    # of the output, we can not use grid.dcell directly, we need to modify it.
    scales = output_shape / lengths
    forward_t = (grid.dcell.dot(inv_cell) * scales).T

    # Scipy's affine transform asks for the inverse transformation matrix, to
    # map from output pixels to input pixels. By taking the inverse of our
    # transformation matrix, we get exactly that.
    tr = cell_invert(forward_t).T

    # Calculate the offset of the image so that all points of the grid "fall" inside
    # the output array.
    # For this we just calculate the centers of the input and output images
    center_input = 0.5 * (_a.asarrayd(grid.shape) - 1)
    center_output = 0.5 * (_a.asarrayd(output_shape) - 1)

    # And then make sure that the input center that is interpolated from the output
    # falls in the actual input's center
    offset = center_input - tr.dot(center_output)

    # We pass all the parameters to scipy's affine_transform
    transformed_image = affine_transform(
        grid.grid,
        tr,
        order=1,
        offset=offset,
        output_shape=output_shape,
        mode=mode,
        **kwargs,
    )

    # Create a new grid with the new shape and the new cell (notice how the cell
    # is rescaled from the input cell to fit the actual coordinates of the system)
    new_grid = grid.__class__((1, 1, 1), lattice=cell * lengths.reshape(3, 1))
    new_grid.grid = transformed_image
    new_grid.geometry = grid.geometry
    new_grid.lattice.origin = grid.origin + new_grid.dcell.dot(forward_t.dot(offset))

    # Find the offset between the origin before and after the transformation
    return new_grid


def orthogonalize_grid(
    grid: Grid,
    interp: tuple[int, int, int] = (1, 1, 1),
    mode: str = "constant",
    **kwargs,
) -> Grid:
    """Transform grid cell to be orthogonal.

    Uses `transform_grid_cell`.

    Parameters
    -----------
    grid: sisl.Grid
        The grid to transform.
    interp: array-like of int of shape (3,), optional
        Number of times that the grid should be augmented for each
        lattice vector.
    mode: str, optional
        determines how to handle borders.
        See `transform_grid_cell` for more info on the possible values.
    **kwargs:
        the rest of keyword arguments are passed directly to `transform_grid_cell`
    """
    return transform_grid_cell(
        grid,
        mode=mode,
        output_shape=tuple(interp[i] * grid.shape[i] for i in range(3)),
        cval=np.nan,
        **kwargs,
    )


def orthogonalize_grid_if_needed(
    grid: Grid,
    axes: Sequence[str],
    tol: float = 1e-3,
    interp: tuple[int, int, int] = (1, 1, 1),
    mode: str = "constant",
    **kwargs,
) -> Grid:
    """Same as `orthogonalize_grid`, but first checks if it is really needed.

    Parameters
    -----------
    grid: sisl.Grid
        The grid to transform.
    axes: list of str
        axes that will be plotted.
    tol: float, optional
        tolerance to determine whether the grid should be transformed.
    interp: array-like of int of shape (3,), optional
        Number of times that the grid should be augmented for each
        lattice vector.
    mode: str, optional
        determines how to handle borders.
        See `transform_grid_cell` for more info on the possible values.
    **kwargs:
        the rest of keyword arguments are passed directly to `transform_grid_cell`
    """

    should_ortogonalize = should_transform_grid_cell_plotting(
        grid=grid, axes=axes, tol=tol
    )

    if should_ortogonalize:
        grid = orthogonalize_grid(grid, interp=interp, mode=mode, **kwargs)

    return grid


def apply_transform(grid: Grid, transform: Union[Callable, str]) -> Grid:
    """applies a transformation to the grid.

    Parameters
    -----------
    grid: sisl.Grid
        The grid to transform.
    transform: callable or str
        The transformation to apply. If it is a string, it will be
        interpreted as a numpy function unless it contains a dot, in
        which case it will be interpreted as a path to a function.
    """
    if isinstance(transform, str):
        # Since this may come from the GUI, there might be extra spaces
        transform = transform.strip()

        # If is a string with no dots, we will assume it is a numpy function
        if len(transform.split(".")) == 1:
            transform = f"numpy.{transform}"

    return grid.apply(transform)


def apply_transforms(grid: Grid, transforms: Sequence[Union[Callable, str]]) -> Grid:
    """Applies multiple transformations sequentially

    Parameters
    -----------
    grid: sisl.Grid
        The grid to transform.
    transforms: list of callable or str
        The transformations to apply. If a transformation it is a string, it will be
        interpreted as a numpy function unless it contains a dot, in which case it will
        be interpreted as a path to a function.
    """
    for transform in transforms:
        grid = apply_transform(grid, transform)
    return grid


def reduce_grid(
    grid: Grid, reduce_method: Literal["average", "sum"], keep_axes: Sequence[int]
) -> Grid:
    """Reduces the grid along multiple axes

    Parameters
    -----------
    grid: sisl.Grid
        The grid to reduce.
    reduce_method: {"average", "sum"}
        The method to use to reduce the grid.
    keep_axes: list of int
        Lattice vectors to maintain (not reduce).
    """
    old_origin = grid.origin

    # Reduce the dimensions that are not going to be displayed
    for ax in [0, 1, 2]:
        if ax not in keep_axes:
            grid = getattr(grid, reduce_method)(ax)

    grid.origin[:] = old_origin

    return grid


def sub_grid(
    grid: Grid,
    x_range: Optional[tuple[float, float]] = None,
    y_range: Optional[tuple[float, float]] = None,
    z_range: Optional[tuple[float, float]] = None,
    cart_tol: float = 1e-3,
) -> Grid:
    """Returns only the part of the grid that is within the specified ranges.

    Only works for cartesian dimensions that correspond to some lattice vector. For
    example, if the grid is skewed in XY but not in Z, this function can sub along Z
    but not along X or Y.

    If there's no point that coincides with the limits, the closest point will be
    taken. This means that the returned grid might not be limited exactly by the bounds
    provided.

    Parameters
    -----------
    grid: sisl.Grid
        The grid to sub.
    x_range: tuple of float, optional
        The range of the x coordinate.
    y_range: tuple of float, optional
        The range of the y coordinate.
    z_range: tuple of float, optional
        The range of the z coordinate.
    cart_tol: float, optional
        Tolerance to determine whether a dimension is cartesian or not.
    """

    cell = grid.lattice.cell

    origin = grid.origin.copy()

    # Get only the part of the grid that we need
    ax_ranges = [x_range, y_range, z_range]
    directions = ["x", "y", "z"]
    for ax, (ax_range, direction) in enumerate(zip(ax_ranges, directions)):
        if ax_range is not None:
            # Cartesian check
            if not is_1D_cartesian(cell, direction, tol=cart_tol):
                raise ValueError(
                    f"Cannot sub grid along '{direction}', since there is no unique lattice vector that represents this direction. Cell: {cell}"
                )

            # Find out which lattice vector represents the direction
            lattice_ax = np.where(cell[:, ax] > cart_tol)[0][0]

            # Build an array with the limits
            lims = np.zeros((2, 3))
            # If the cell was transformed, then we need to modify
            # the range to get what the user wants.
            lims[:, ax] = (
                ax_range  # + self.offsets["cell_transform"][ax] - self.offsets["origin"][ax]
            )

            origin[ax] += ax_range[0]

            # Get the indices of those points
            indices = np.array([grid.index(lim) for lim in lims], dtype=int)

            # And finally get the subpart of the grid
            grid = grid.sub(
                np.arange(indices[0, lattice_ax], indices[1, lattice_ax] + 1),
                lattice_ax,
            )

    grid.origin[:] = origin

    return grid


def interpolate_grid(
    grid: Grid, interp: tuple[int, int, int] = (1, 1, 1), force: bool = False
) -> Grid:
    """Interpolates the grid.

    It also makes sure that the grid is not interpolated over dimensions that only
    contain one value, unless `force` is True.

    If the interpolation factors are all 1, the grid is returned unchanged.

    Parameters
    -----------
    grid: sisl.Grid
        The grid to interpolate.
    interp: array-like of int of shape (3,), optional
        Number of times that the grid should be augmented for each
        lattice vector.
    force: bool, optional
        Whether to force the interpolation over dimensions that only
        contain one value.
    """

    grid_shape = np.array(grid.shape)

    interp_factors = np.array(interp)
    if not force:
        # No need to interpolate over dimensions that only contain one value.
        interp_factors[grid_shape == 1] = 1

    interp_factors = interp_factors * grid_shape
    if (interp_factors != 1).any():
        grid = grid.interp(interp_factors.astype(int))

    return grid


def grid_geometry(
    grid: Grid, geometry: Optional[Geometry] = None
) -> Union[Geometry, None]:
    """Returns the geometry associated with the grid.

    Parameters
    -----------
    grid: sisl.Grid
        The grid for which we want to get the geometry.
    geometry: sisl.Geometry, optional
        If provided, this geometry will be returned instead of the one
        associated with the grid.
    """
    if geometry is None:
        geometry = getattr(grid, "geometry", None)

    return geometry


def should_transform_grid_cell_plotting(
    grid: Grid, axes: Sequence[str], tol: float = 1e-3
) -> bool:
    """Determines whether the grid should be transformed for plotting.

    It takes into account the axes that will be plotted and checks if the grid
    is skewed in any of those directions. If it is, it will return True, meaning
    that the grid should be transformed before plotting.

    Parameters
    -----------
    grid: sisl.Grid
        grid to check.
    axes: list of str
        axes that will be plotted.
    """
    ndim = len(axes)

    # Determine whether we should transform the grid to cartesian axes. This will be needed
    # if the grid is skewed. However, it is never needed for the 3D representation, since we
    # compute the coordinates of each point in the isosurface, and we don't need to reduce the
    # grid.
    should_orthogonalize = not is_cartesian_unordered(grid, tol=tol) and len(axes) < 3
    # We also don't need to orthogonalize if cartesian coordinates are not requested
    # (this would mean that axes is a combination of "a", "b" and "c")
    should_orthogonalize = should_orthogonalize and bool(
        set(axes).intersection(["x", "y", "z"])
    )

    if should_orthogonalize and ndim == 1:
        # In 1D representations, even if the cell is skewed, we might not need to transform.
        # An example of a cell that we don't need to transform is:
        # a = [1, 1, 0], b = [1, -1, 0], c = [0, 0, 1]
        # If the user wants to display the values on the z coordinate, we can safely reduce the
        # first two axes, as they don't contribute in the Z direction. Also, it is required that
        # "c" doesn't contribute to any of the other two directions.
        should_orthogonalize &= not is_1D_cartesian(grid, axes[0], tol=tol)

    return should_orthogonalize


def get_grid_axes(grid: Grid, axes: Sequence[str]) -> list[int]:
    """Returns the indices of the lattice vectors that correspond to the axes.

    If axes is of length 3 (i.e. a 3D view), this function always returns [0, 1, 2]
    regardless of what the axes are.

    Parameters
    -----------
    grid: sisl.Grid
        The grid for which we want to get the axes.
    axes: list of str
        axes that will be plotted. Either cartesian or "a", "b", "c".
    """

    ndim = len(axes)

    if ndim < 3:
        grid_axes = infer_cell_axes(grid, axes)
    elif ndim == 3:
        grid_axes = [0, 1, 2]
    else:
        raise ValueError(f"Invalid number of axes: {ndim}")

    return grid_axes


def get_ax_vals(
    grid: Grid,
    ax: Literal[0, 1, 2, "a", "b", "c", "x", "y", "z"],
    nsc: tuple[int, int, int],
) -> npt.NDArray[np.float64]:
    """Returns the values of a given axis on all grid points.

    These can be used for example as axes ticks on a plot.

    Parameters
    ----------
    grid: sisl.Grid
        The grid for which we want to get the axes values.
    ax: {"x", "y", "z", "a", "b", "c", 0, 1, 2}
        The axis for which we want the values.
    nsc: array-like of int of shape (3,)
        Number of times that the grid has been tiled in each direction, so that
        if a fractional axis is requested, the values are correct.
    """
    if isinstance(ax, int) or ax in ("a", "b", "c"):
        ax = {"a": 0, "b": 1, "c": 2}.get(ax, ax)
        ax_vals = np.linspace(0, nsc[ax], grid.shape[ax])
    else:
        ax_index = {"x": 0, "y": 1, "z": 2}[ax]

        ax_vals = np.arange(
            0, grid.cell[ax_index, ax_index], grid.dcell[ax_index, ax_index]
        ) + get_offset(grid, ax)

        if len(ax_vals) == grid.shape[ax_index] + 1:
            ax_vals = ax_vals[:-1]

    return ax_vals


def get_offset(grid: Grid, ax: Literal[0, 1, 2, "a", "b", "c", "x", "y", "z"]) -> float:
    """Returns the offset of the grid along a certain axis.

    Parameters
    -----------
    grid: sisl.Grid
        The grid for which we want to get the offset.
    ax: {"x", "y", "z", "a", "b", "c", 0, 1, 2}
        The axis for which we want the offset.
    """

    if isinstance(ax, int) or ax in ("a", "b", "c"):
        return 0
    else:
        coord_index = "xyz".index(ax)
        return grid.origin[coord_index]


GridDataArray = DataArray


def grid_to_dataarray(
    grid: Grid, axes: Sequence[str], grid_axes: Sequence[int], nsc: tuple[int, int, int]
) -> GridDataArray:
    transpose_grid_axes = [*grid_axes]
    for ax in (0, 1, 2):
        if ax not in transpose_grid_axes:
            transpose_grid_axes.append(ax)

    values = np.squeeze(grid.grid.transpose(*transpose_grid_axes))

    arr = DataArray(
        values,
        coords=[
            (k, get_ax_vals(grid, ax, nsc=nsc)) for k, ax in zip(["x", "y", "z"], axes)
        ],
    )

    arr.attrs["grid"] = grid

    return arr


def get_isos(data: GridDataArray, isos: Sequence[dict]) -> list[dict]:
    """Gets the iso surfaces or isocontours of an array of data.

    Parameters
    -----------
    data: DataArray
        The data for which we want to get the iso surfaces.
    isos: list of dict
        List of isosurface specifications.
    """
    from skimage.measure import find_contours

    # values = data['values'].values
    values = data.values
    isos_to_draw = []

    # Get the dimensionality of the data
    ndim = values.ndim

    if len(isos) > 0 or ndim == 3:
        minval = np.nanmin(values)
        maxval = np.nanmax(values)

    # Prepare things for each possible dimensionality
    if ndim == 1:
        # For now, we don't calculate 1D "isopoints"
        return []
    elif ndim == 2:
        # Get the partition size
        dx = data.x[1] - data.x[0]
        dy = data.y[1] - data.y[0]

        # Function to get the coordinates from indices
        def _indices_to_2Dspace(contour_coords):
            return contour_coords.dot([[dx, 0, 0], [0, dy, 0]])

        def _calc_iso(isoval):
            contours = find_contours(values, isoval)

            contour_xs = []
            contour_ys = []
            for contour in contours:
                # Swap the first and second columns so that we have [x,y] for each
                # contour point (instead of [row, col], which means [y, x])
                contour_coords = contour[:, [1, 0]]
                # Then convert from indices to coordinates in the 2D space
                contour_coords = _indices_to_2Dspace(contour_coords)
                contour_xs = [*contour_xs, None, *contour_coords[:, 0]]
                contour_ys = [*contour_ys, None, *contour_coords[:, 1]]

            # Add the information about this isoline to the list of isolines
            return {
                "x": contour_xs,
                "y": contour_ys,
                "width": iso.get("width"),
            }

    elif ndim == 3:
        # In 3D, use default isosurfaces if none were provided.
        if len(isos) == 0 and maxval != minval:
            default_iso_frac = 0.3  # isos_param["frac"].default

            # If the default frac is 0.3, they will be displayed at 0.3 and 0.7
            isos = [{"frac": default_iso_frac}, {"frac": 1 - default_iso_frac}]

        # Define the function that will calculate each isosurface
        def _calc_iso(isoval):
            vertices, faces, normals, intensities = data.grid.isosurface(
                isoval, iso.get("step_size", 1)
            )

            # vertices = vertices + self._get_offsets(grid) + self.offsets["origin"]

            return {"vertices": vertices, "faces": faces}

    else:
        raise ValueError(f"Dimensionality must be lower than 3, but is {ndim}")

    # Now loop through all the isos
    for iso in isos:
        if not iso.get("active", True):
            continue

        # Infer the iso value either from val or from frac
        isoval = iso.get("val")
        if isoval is None:
            frac = iso.get("frac")
            if frac is None:
                raise ValueError(
                    f"You are providing an iso query without 'val' and 'frac'. There's no way to know the isovalue!\nquery: {iso}"
                )
            isoval = minval + (maxval - minval) * frac

        isos_to_draw.append(
            {
                "color": iso.get("color"),
                "opacity": iso.get("opacity"),
                "name": iso.get("name", "Iso: $isoval$").replace(
                    "$isoval$", f"{isoval:.4f}"
                ),
                **_calc_iso(isoval),
            }
        )

    return isos_to_draw
