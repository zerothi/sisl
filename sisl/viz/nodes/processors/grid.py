from __future__ import annotations

from typing import Callable, Literal, Union, Optional, Sequence, List
import numpy as np
import numpy.typing as npt
from scipy.ndimage import affine_transform
from xarray import DataArray

import sisl
from sisl import Grid
from sisl import _array as _a
from sisl._supercell import cell_invert

from .geometry import GeometryHelper
from .cell import infer_cell_axes, is_cartesian_unordered, is_1D_cartesian
from ..node import Node
from ...types import Axis, PathLike
from ..data_sources import DataSource

class GridGetter(DataSource):
    pass

@GridGetter.from_func
def GridDataGrid(grid: Grid) -> Grid:
    return grid

@GridGetter.from_func
def GridDataFile(file: PathLike, **kwargs) -> Grid:
    return sisl.get_sile(file).read_grid(**kwargs)

class GridProcessor(Node):
    pass

@GridProcessor.from_func
def get_grid_representation(grid: Grid, represent: Literal['real', 'imag', 'mod', 'phase', 'rad_phase', 'deg_phase']) -> Grid:
    """Returns a representation of the grid

    Parameters
    ------------
    grid: sisl.Grid
        the grid for which we want return
    represent: {"real", "imag", "mod", "phase", "deg_phase", "rad_phase"}
        the type of representation. "phase" is equivalent to "rad_phase"

    Returns
    ------------
    np.ndarray of shape = grid.shape
    """
    def _func(values: npt.NDArray[Union[np.int_, np.float_, np.complex_]]) -> npt.NDArray:
        if represent == 'real':
            new_values = values.real
        elif represent == 'imag':
            new_values = values.imag
        elif represent == 'mod':
            new_values = np.absolute(values)
        elif represent in ['phase', 'rad_phase', 'deg_phase']:
            new_values = np.angle(values, deg=represent.startswith("deg"))
        else:
            raise ValueError(f"'{represent}' is not a valid value for the `represent` argument")

        return new_values

    return grid.apply(_func)

@GridProcessor.from_func
def tile_grid(grid: Grid, nsc: Sequence[int] = [1, 1, 1]) -> Grid:
    """Tiles the grid"""
    for ax, reps in enumerate(nsc):
        grid = grid.tile(reps, ax)
    return grid

@GridProcessor.from_func
def transform_grid_cell(grid: Grid, cell: npt.NDArray[np.float_] = np.eye(3), output_shape: Optional[Sequence[int]] = None, mode: str = "constant", order: int = 1, **kwargs):
    """
    Applies a linear transformation to the grid to get it relative to an arbitrary cell.

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

    # Get the current cell in coordinates of the destination axes
    inv_cell = cell_invert(cell).T
    projected_cell = grid.cell.dot(inv_cell)

    # From that, infere how long will the bounding box of the cell be
    lengths = abs(projected_cell).sum(axis=0)

    # Create the transformation matrix. Since we want to control the shape
    # of the output, we can not use grid.dcell directly, we need to modify it.
    scales = output_shape / lengths
    forward_t = (grid.dcell.dot(inv_cell)*scales).T

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
    transformed_image = affine_transform(grid.grid, tr, order=1, offset=offset,
        output_shape=output_shape, mode=mode, **kwargs)

    # Create a new grid with the new shape and the new cell (notice how the cell
    # is rescaled from the input cell to fit the actual coordinates of the system)
    new_grid = grid.__class__((1, 1, 1), sc=cell*lengths.reshape(3, 1))
    new_grid.grid = transformed_image
    new_grid.geometry = grid.geometry
    #new_grid.origin = grid.origin + new_grid.dcell.dot(forward_t.dot(offset))

    # Find the offset between the origin before and after the transformation
    return new_grid #,new_grid.dcell.dot(forward_t.dot(offset))

@GridProcessor.from_func
def orthogonalize_grid(grid:Grid, interp: Sequence[int] = [1, 1, 1], mode: str = "constant", **kwargs) -> Grid:
    return transform_grid_cell(
        grid, mode=mode, output_shape=(np.array(interp)*grid.shape).astype(int), cval=np.nan, **kwargs
    )

@GridProcessor.from_func
def orthogonalize_grid_if_needed(grid: Grid, axes: Sequence[str], tol: float = 1e-3, 
    interp: Sequence[int] = [1, 1, 1], mode: str = "constant", **kwargs) -> Grid:

    should_ortogonalize = should_transform_grid_cell_plotting(grid=grid, axes=axes, tol=tol)

    if should_ortogonalize:
        grid = orthogonalize_grid(grid, interp=interp, mode=mode, **kwargs)

    return grid

@GridProcessor.from_func
def apply_transform(grid: Grid, transform: Union[Callable, str]) -> Grid:
    if isinstance(transform, str):
        # Since this may come from the GUI, there might be extra spaces
        transform = transform.strip()

        # If is a string with no dots, we will assume it is a numpy function
        if len(transform.split(".")) == 1:
            transform = f"numpy.{transform}"

    return grid.apply(transform)

@GridProcessor.from_func
def apply_transforms(grid: Grid, transforms: Sequence[Union[Callable, str]]) -> Grid:
    for transform in transforms:
        grid = apply_transform(grid, transform)
    return grid

@GridProcessor.from_func
def reduce_grid(grid: Grid, reduce_method: Literal["average", "sum"], axes: Sequence[int]) -> Grid:
    # Reduce the dimensions that are not going to be displayed
    for ax in [0, 1, 2]:
        if ax not in axes:
            grid = getattr(grid, reduce_method)(ax)
    
    return grid

AxRange = Optional["Sequence[float]"]

@GridProcessor.from_func
def sub_grid(grid: Grid, x_range: AxRange = None, y_range: AxRange = None, z_range: AxRange = None) -> Grid:
    # Get only the part of the grid that we need
    ax_ranges = [x_range, y_range, z_range]
    for ax, ax_range in enumerate(ax_ranges):
        if ax_range is not None:
            # Build an array with the limits
            lims = np.zeros((2, 3))
            # If the cell was transformed, then we need to modify
            # the range to get what the user wants.
            lims[:, ax] = ax_range #+ self.offsets["cell_transform"][ax] - self.offsets["origin"][ax]

            # Get the indices of those points
            indices = np.array([grid.index(lim) for lim in lims], dtype=int)

            # And finally get the subpart of the grid
            grid = grid.sub(np.arange(indices[0, ax], indices[1, ax] + 1), ax)
    
    return grid

@GridProcessor.from_func
def interpolate_grid(grid:Grid, interp: Sequence[int] = [1, 1, 1], force: bool = False) -> Grid:
    grid_shape = np.array(grid.shape)

    interp_factors = np.array(interp)
    if not force:
        # No need to interpolate over dimensions that only contain one value.
        interp_factors[grid_shape == 1] = 1

    interp_factors = interp_factors * grid_shape
    if (interp_factors != 1).any():
        grid = grid.interp(interp_factors.astype(int))

    return grid


class GridHelper(Node):
    pass

@GridHelper.from_func
def should_transform_grid_cell_plotting(grid: Grid, axes: Sequence[str], tol: float = 1e-3) -> bool:
    ndim = len(axes)

    # Determine whether we should transform the grid to cartesian axes. This will be needed
    # if the grid is skewed. However, it is never needed for the 3D representation, since we
    # compute the coordinates of each point in the isosurface, and we don't need to reduce the
    # grid.
    should_orthogonalize = not is_cartesian_unordered(grid, tol=tol) and len(axes) < 3
    # We also don't need to orthogonalize if cartesian coordinates are not requested
    # (this would mean that axes is a combination of "a", "b" and "c")
    should_orthogonalize = should_orthogonalize and bool(set(axes).intersection(["x", "y", "z"]))

    if should_orthogonalize and ndim == 1:
        # In 1D representations, even if the cell is skewed, we might not need to transform.
        # An example of a cell that we don't need to transform is:
        # a = [1, 1, 0], b = [1, -1, 0], c = [0, 0, 1]
        # If the user wants to display the values on the z coordinate, we can safely reduce the
        # first two axes, as they don't contribute in the Z direction. Also, it is required that
        # "c" doesn't contribute to any of the other two directions.
        should_orthogonalize &= not is_1D_cartesian(grid, axes[0], tol=tol)
    
    return should_orthogonalize

@GridHelper.from_func
def get_grid_axes(grid: Grid, axes: Sequence[str]) -> List[int]:
    ndim = len(axes)

    if ndim < 3:
        # If we are not transforming the grid, we need to get the axes of the grid that contribute to the
        # directions we have to plot.
        grid_axes = infer_cell_axes(grid, axes)
    elif ndim == 3:
        grid_axes = [0, 1, 2]
    else:
        raise ValueError(f"Invalid number of axes: {ndim}")

    return grid_axes

@GridHelper.from_func
def get_ax_range(grid: Grid, ax: Literal[0,1,2,"a","b","c","x","y","z"], nsc: Sequence[int]) -> npt.NDArray[np.float_]:
    if isinstance(ax, int) or ax in ("a", "b", "c"):
        ax = {"a": 0, "b": 1, "c": 2}.get(ax, ax)
        ax_vals = np.linspace(0, nsc[ax], grid.shape[ax])
    else:
        offset = grid.origin

        ax = {"x": 0, "y": 1, "z": 2}[ax]

        ax_vals = np.arange(0, grid.cell[ax, ax], grid.dcell[ax, ax]) + get_offset(grid, ax)

        if len(ax_vals) == grid.shape[ax] + 1:
            ax_vals = ax_vals[:-1]

    return ax_vals

@GeometryHelper.from_func
def get_offset(grid: Grid, ax: Literal[0,1,2,"a","b","c","x","y","z"]) -> float:
    if isinstance(ax, int) or ax in ("a", "b", "c"):
        return 0
    else:
        coord_index = "xyz".index(ax)
        return grid.origin[coord_index]

GridDataArray = DataArray

@GridHelper.from_func
def get_isos(data: GridDataArray, isos: Sequence[dict]) -> List[dict]:
    """Gets the iso surfaces or isocontours of an array of data.
    
    Parameters
    -----------
    data: DataArray
    isos: list of dict
    """
    from skimage.measure import find_contours
    
    #values = data['values'].values
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
                "x": contour_xs, "y": contour_ys, "width": iso.get("width"),
            }
        
    elif ndim == 3:
        # In 3D, use default isosurfaces if none were provided.
        if len(isos) == 0 and maxval != minval:
            default_iso_frac = 0.3 #isos_param["frac"].default

            # If the default frac is 0.3, they will be displayed at 0.3 and 0.7
            isos = [
                {"frac": default_iso_frac},
                {"frac": 1-default_iso_frac}
            ]
            
        # Define the function that will calculate each isosurface
        def _calc_iso(isoval):
            vertices, faces, normals, intensities = data.grid.isosurface(isoval, iso.get("step_size", 1))

            #vertices = vertices + self._get_offsets(grid) + self.offsets["origin"]
            
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
                raise ValueError(f"You are providing an iso query without 'val' and 'frac'. There's no way to know the isovalue!\nquery: {iso}")
            isoval = minval + (maxval-minval)*frac
        
        isos_to_draw.append({
            "color": iso.get("color"), "opacity": iso.get("opacity"),
            "name": iso.get("name", "Iso: $isoval$").replace("$isoval$", f"{isoval:.4f}"),
            **_calc_iso(isoval),
        })

    return isos_to_draw

class GridData(Node):
    pass

@GridData.from_func
def project_grid(grid: Grid, axes: Sequence[str], grid_axes: Sequence[int], nsc: Sequence[int]) -> GridDataArray:

    values = np.squeeze(grid.grid)

    arr = DataArray(
        values,
        coords=[
            (k, get_ax_range(grid, ax, nsc=nsc)) 
            for k, ax in zip(["x", "y", "z"], axes)
        ]
    )

    arr.attrs['grid'] = grid

    return arr

"""# We will tile the grid now, as at the moment there's no good way to tile it afterwards
# Note that this means extra computation, as we are transforming (skewed_2d) or calculating
# the isosurfaces (3d) using more than one unit cell (FIND SMARTER WAYS!)
for ax, reps in enumerate(nsc):
    grid = grid.tile(reps, ax)

# Determine whether we should transform the grid to cartesian axes. This will be needed
# if the grid is skewed. However, it is never needed for the 3D representation, since we
# compute the coordinates of each point in the isosurface, and we don't need to reduce the
# grid.
should_orthogonalize = ~self._is_cartesian_unordered(grid.cell) and self._ndim < 3
# We also don't need to orthogonalize if cartesian coordinates are not requested
# (this would mean that axes is a combination of "a", "b" and "c")
should_orthogonalize = should_orthogonalize and bool(set(axes).intersection(["x", "y", "z"]))

if should_orthogonalize and self._ndim == 1:
    # In 1D representations, even if the cell is skewed, we might not need to transform.
    # An example of a cell that we don't need to transform is:
    # a = [1, 1, 0], b = [1, -1, 0], c = [0, 0, 1]
    # If the user wants to display the values on the z coordinate, we can safely reduce the
    # first two axes, as they don't contribute in the Z direction. Also, it is required that
    # "c" doesn't contribute to any of the other two directions.
    should_orthogonalize &= not self._is_1D_cartesian(grid.cell, axes[0])

if should_orthogonalize:
    grid, self.offsets["cell_transform"] = self._transform_grid_cell(
        grid, mode=transform_bc, output_shape=(np.array(interp)*grid.shape).astype(int), cval=np.nan
    )
    # The interpolation has already happened, so just set it to [1,1,1] for the rest of the method
    interp = [1, 1, 1]

    # Now the grid axes correspond to the cartesian coordinates.
    grid_axes = [{"x": 0, "y": 1, "z": 2}[ax] for ax in axes]
elif self._ndim < 3:
    # If we are not transforming the grid, we need to get the axes of the grid that contribute to the
    # directions we have to plot.
    grid_axes = self._infer_grid_axes(axes, grid.cell)
elif self._ndim == 3:
    grid_axes = [0, 1, 2]

# Apply all transforms requested by the user
for transform in transforms:
    grid = self._transform_grid(grid, transform)

# Get only the part of the grid that we need
ax_ranges = [x_range, y_range, z_range]
for ax, ax_range in enumerate(ax_ranges):
    if ax_range is not None:
        # Build an array with the limits
        lims = np.zeros((2, 3))
        # If the cell was transformed, then we need to modify
        # the range to get what the user wants.
        lims[:, ax] = ax_range + self.offsets["cell_transform"][ax] - self.offsets["origin"][ax]

        # Get the indices of those points
        indices = np.array([grid.index(lim) for lim in lims], dtype=int)

        # And finally get the subpart of the grid
        grid = grid.sub(np.arange(indices[0, ax], indices[1, ax] + 1), ax)

# Reduce the dimensions that are not going to be displayed
for ax in [0, 1, 2]:
    if ax not in grid_axes:
        grid = getattr(grid, reduce_method)(ax)

# Interpolate the grid to a different shape, if needed
interp_factors = np.array([factor if ax in grid_axes else 1 for ax, factor in enumerate(interp)], dtype=int)
interpolate = (interp_factors != 1).any()
if interpolate:
    grid = grid.interp((np.array(interp_factors)*grid.shape).astype(int))

# Remove the leftover dimensions
values = np.squeeze(grid.grid)

# Choose which function we need to use to prepare the data
prepare_func = getattr(self, f"_prepare{self._ndim}D")

# Use it
backend_info = prepare_func(grid, values, axes, grid_axes, nsc, trace_name, showlegend=bool(trace_name) or values.ndim == 3)

backend_info["ndim"] = self._ndim

# Add also the geometry if the user requested it
# This should probably not work like this. It should make use
# of MultiplePlot somehow. The problem is that right now, the bonds
# are calculated each time this method is called, for example
geom_plot = None
if plot_geom:
    geom = getattr(self.grid, 'geometry', None)
    if geom is None:
        warn('You asked to plot the geometry, but the grid does not contain any geometry')
    else:
        geom_plot = geom.plot(**{'axes': axes, "nsc": self.get_setting("nsc"), **geom_kwargs})

backend_info["geom_plot"] = geom_plot

# Define the axes titles
backend_info["axes_titles"] = {
    f"{ax_name}axis": GeometryPlot._get_ax_title(ax) for ax_name, ax in zip(("x", "y", "z"), axes)
}
if self._ndim == 1:
    backend_info["axes_titles"]["yaxis"] = "Values"

return backend_info"""