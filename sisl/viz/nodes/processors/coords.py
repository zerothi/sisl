from typing import Optional, Union, Callable
import numpy as np
import numpy.typing as npt
import re
from xarray import Dataset

from sisl.supercell import SuperCell, SuperCellChild
from sisl._supercell import cell_invert
from sisl.utils import direction
from sisl.utils.mathematics import fnorm

from sisl.viz.nodes.node import Node
from ...types import Axes, CellLike, Axis

CoordsDataset = Dataset

class AxesHelper(Node):
    pass

@AxesHelper.from_func
def axis_direction(ax: Axis, cell: Optional[Union[npt.ArrayLike, SuperCell]] = None) -> npt.NDArray[np.float64]:
    if isinstance(ax, (int, str)):
        sign = 1
        # If the axis contains a -, we need to mirror the direction.
        if isinstance(ax, str) and ax[0] == "-":
            sign = -1
            ax = ax[1]
        ax = sign * direction(ax, abc=cell, xyz=np.diag([1., 1., 1.]))

    return ax

@AxesHelper.from_func
def axes_cross_product(v1: Axis, v2: Axis, cell: Optional[Union[npt.ArrayLike, SuperCell]] = None):
    """An enhanced version of the cross product.

    It is an enhanced version because both bectors accept strings that represent
    the cartesian axes or the lattice vectors (see `v1`, `v2` below). It has been built
    so that cross product between lattice vectors (-){"a", "b", "c"} follows the same rules
    as (-){"x", "y", "z"}
    Parameters
    ----------
    v1, v2: array-like of shape (3,) or (-){"x", "y", "z", "a", "b", "c"}
        The vectors to take the cross product of.
    cell: array-like of shape (3, 3)
        The cell of the structure, only needed if lattice vectors {"a", "b", "c"}
        are passed for `v1` and `v2`.
    """
    # Make abc follow the same rules as xyz to find the orthogonal direction
    # That is, a X b = c; -a X b = -c and so on.
    if isinstance(v1, str) and isinstance(v2, str):
        if re.match("([+-]?[abc]){2}", v1 + v2):
            v1 = v1.replace("a", "x").replace("b", "y").replace("c", "z")
            v2 = v2.replace("a", "x").replace("b", "y").replace("c", "z")
            ort = axes_cross_product(v1, v2)
            ort_ax = "abc"[np.where(ort != 0)[0][0]]
            if ort.sum() == -1:
                ort_ax = "-" + ort_ax
            return axis_direction(ort_ax, cell)

    # If the vectors are not abc, we just need to take the cross product.
    return np.cross(axis_direction(v1, cell), axis_direction(v2, cell))

@AxesHelper.from_func
def get_ax_title(ax: Union[Axis, Callable]) -> str:
    """Generates the title for a given axis"""
    if hasattr(ax, "__name__"):
        title = ax.__name__
    elif isinstance(ax, np.ndarray) and ax.shape == (3,):
        title = str(ax)
    elif not isinstance(ax, str):
        title = ""
    elif re.match("[+-]?[xXyYzZ]", ax):
        title = f'{ax.upper()} axis [Ang]'
    elif re.match("[+-]?[aAbBcC]", ax):
        title = f'{ax.upper()} lattice vector'
    else:
        title = ax

    return title

class CoordsHelper(Node):
    pass

@CoordsHelper.from_func
def projected_2Dcoords(cell: CellLike, xyz: npt.NDArray[np.float64], xaxis: Axis = "x", yaxis: Axis = "y") -> npt.NDArray[np.float64]:
    """
    Moves the 3D positions of the atoms to a 2D supspace.

    In this way, we can plot the structure from the "point of view" that we want.

    NOTE: If xaxis/yaxis is one of {"a", "b", "c", "1", "2", "3"} the function doesn't
    project the coordinates in the direction of the lattice vector. The fractional
    coordinates, taking in consideration the three lattice vectors, are returned
    instead.

    Parameters
    ------------
    geometry: sisl.Geometry
        the geometry for which you want the projected coords
    xyz: array-like of shape (natoms, 3), optional
        the 3D coordinates that we want to project.
        otherwise they are taken from the geometry. 
    xaxis: {"x", "y", "z", "a", "b", "c"} or array-like of shape 3, optional
        the direction to be displayed along the X axis. 
    yaxis: {"x", "y", "z", "a", "b", "c"} or array-like of shape 3, optional
        the direction to be displayed along the X axis.

    Returns
    ----------
    np.ndarray of shape (2, natoms)
        the 2D coordinates of the geometry, with all positions projected into the plane
        defined by xaxis and yaxis.
    """
    if isinstance(cell, (SuperCell, SuperCellChild)):
        cell = cell.cell

    try:
        all_lattice_vecs = len(set([xaxis, yaxis]).intersection(["a", "b", "c"])) == 2
    except:
        # If set fails it is because xaxis/yaxis is unhashable, which means it
        # is a numpy array
        all_lattice_vecs = False

    if all_lattice_vecs:
        coord_indices = ["abc".index(ax) for ax in (xaxis, yaxis)]

        icell = cell_invert(cell)
    else:
        # Get the directions that these axes represent
        xaxis = axis_direction(xaxis, cell)
        yaxis = axis_direction(yaxis, cell)

        fake_cell = np.array([xaxis, yaxis, np.cross(xaxis, yaxis)], dtype=np.float64)
        icell = cell_invert(fake_cell)
        coord_indices = [0, 1]

    return np.dot(xyz, icell.T)[..., coord_indices]

@CoordsHelper.from_func
def projected_1Dcoords(cell: CellLike, xyz: npt.NDArray[np.float64], axis: Axis = "x"):
    """
    Moves the 3D positions of the atoms to a 2D supspace.

    In this way, we can plot the structure from the "point of view" that we want.

    NOTE: If axis is one of {"a", "b", "c", "1", "2", "3"} the function doesn't
    project the coordinates in the direction of the lattice vector. The fractional
    coordinates, taking in consideration the three lattice vectors, are returned
    instead.

    Parameters
    ------------
    geometry: sisl.Geometry
        the geometry for which you want the projected coords
    xyz: array-like of shape (natoms, 3), optional
        the 3D coordinates that we want to project.
        otherwise they are taken from the geometry. 
    axis: {"x", "y", "z", "a", "b", "c", "1", "2", "3"} or array-like of shape 3, optional
        the direction to be displayed along the X axis.
    nsc: array-like of shape (3, ), optional
        only used if `axis` is a lattice vector. It is used to rescale everything to the unit
        cell lattice vectors, otherwise `GeometryPlot` doesn't play well with `GridPlot`.

    Returns
    ----------
    np.ndarray of shape (natoms, )
        the 1D coordinates of the geometry, with all positions projected into the line
        defined by axis.
    """
    if isinstance(cell, (SuperCell, SuperCellChild)):
        cell = cell.cell

    if isinstance(axis, str) and axis in ("a", "b", "c", "0", "1", "2"):
        return projected_2Dcoords(cell, xyz, xaxis=axis, yaxis="a" if axis == "c" else "c")[..., 0]

    # Get the direction that the axis represents
    axis = axis_direction(axis, cell)

    return xyz.dot(axis/fnorm(axis)) / fnorm(axis)

@CoordsHelper.from_func
def coords_depth(coords_data: CoordsDataset, axes: Axes) -> npt.NDArray[np.float64]:
    cell = _get_cell_from_dataset(coords_data=coords_data)
    
    depth_vector = axes_cross_product(axes[0], axes[1], cell)
    depth = project_to_axes(coords_data, axes=[depth_vector]).x
    
    return depth

@CoordsHelper.from_func
def sphere(center: npt.ArrayLike = [0, 0, 0], r: float = 1, vertices: int = 10) -> dict:
    phi, theta = np.mgrid[0.0:np.pi: 1j*vertices, 0.0:2.0*np.pi: 1j*vertices]
    center = np.array(center)

    x = center[0] + r*np.sin(phi)*np.cos(theta)
    y = center[1] + r*np.sin(phi)*np.sin(theta)
    z = center[2] + r*np.cos(phi)

    return {'x': x, 'y': y, 'z': z}

class CoordsDataProcessor(Node):
    pass

def _get_cell_from_dataset(coords_data: CoordsDataset) -> npt.NDArray[np.float64]:
    cell = coords_data.attrs.get("cell")
    if cell is None:
        if "sc" in coords_data.attrs:
            cell = coords_data.sc.cell
        else:
            cell = coords_data.geometry.cell
    
    return cell

@CoordsDataProcessor.from_func
def projected_1D_data(coords_data: CoordsDataset, axis: Axis = "x", dataaxis_1d: Union[Callable, npt.NDArray, None] = None) -> CoordsDataset:
    cell = _get_cell_from_dataset(coords_data=coords_data)

    xyz = coords_data.xyz.values

    x = projected_1Dcoords(cell, xyz=xyz, axis=axis)

    dims = coords_data.xyz.dims[:-1]

    if dataaxis_1d is None:
        y = np.zeros_like(x)
    else:
        if callable(dataaxis_1d):
            y = dataaxis_1d(x)
        else:
            y = dataaxis_1d
    coords_data = coords_data.assign(x=(dims, x), y=(dims, y))

    return coords_data

@CoordsDataProcessor.from_func
def projected_2D_data(coords_data: CoordsDataset, xaxis: Axis = "x", yaxis: Axis = "y", sort_by_depth: bool = False) -> CoordsDataset:
    cell = _get_cell_from_dataset(coords_data=coords_data)

    xyz = coords_data.xyz.values

    xy = projected_2Dcoords(cell, xyz, xaxis=xaxis, yaxis=yaxis)

    x, y = xy[..., 0], xy[..., 1]
    dims = coords_data.xyz.dims[:-1]

    coords_data = coords_data.assign(x=(dims, x), y=(dims, y))

    coords_data = coords_data.assign(
        {"depth": (dims, coords_depth(coords_data, [xaxis, yaxis]).data)}
    )
    if sort_by_depth:
        coords_data = coords_data.sortby("depth")

    return coords_data

@CoordsDataProcessor.from_func
def projected_3D_data(coords_data: CoordsDataset) -> CoordsDataset:
    x, y, z = np.moveaxis(coords_data.xyz.values, -1, 0)
    dims = coords_data.xyz.dims[:-1]

    coords_data = coords_data.assign(x=(dims, x), y=(dims, y), z=(dims, z))

    return coords_data

@CoordsDataProcessor.from_func
def project_to_axes(coords_data: CoordsDataset, axes: Axes, 
    dataaxis_1d: Optional[Union[npt.ArrayLike, Callable]] = None, sort_by_depth: bool = False) -> CoordsDataset:
    ndim = len(axes) 
    if ndim == 3:
        xaxis, yaxis, zaxis = axes
        coords_data = projected_3D_data(coords_data)
    elif ndim == 2:
        xaxis, yaxis = axes
        coords_data = projected_2D_data(coords_data, xaxis=xaxis, yaxis=yaxis, sort_by_depth=sort_by_depth)
    elif ndim == 1:
        xaxis = axes[0]
        yaxis = dataaxis_1d
        coords_data = projected_1D_data(coords_data, axis=xaxis, dataaxis_1d=dataaxis_1d)

    plot_axes = ["x", "y", "z"][:ndim]

    for ax, plot_ax in zip(axes, plot_axes):
        coords_data[plot_ax].attrs["axis"] = {
            "title": get_ax_title(ax),
        }
    
    coords_data.attrs['ndim'] = ndim

    return coords_data
