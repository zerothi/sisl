from typing import Literal, Sequence
from sisl.viz.nodes.processors.cell import gen_cell_dataset, cell_to_lines
from sisl.viz.nodes.processors.coords import project_to_axes

from .plotter import PlotterNode, PlotterNodeXY
from ...types import Axes, CellLike
from ..node import Node

@Node.from_func
def get_ndim(axes: Axes) -> int:
    return len(axes)

@Node.from_func
def get_z(ndim: int) -> Literal["z", False]:
    if ndim == 3:
        z = "z"
    else:
        z = False
    return z

@PlotterNode.from_func
def cell_plot_actions(cell: CellLike = None, show_cell: Literal[False, "box", "axes"] = "box", axes=["x", "y", "z"], cell_style={}, dataaxis_1d=None):    
    if show_cell == False:
        cell_plottings = []
    else:
        cell_ds = gen_cell_dataset(cell)
        cell_lines = cell_to_lines(cell_ds, show_cell, cell_style)
        projected_cell_lines = project_to_axes(cell_lines, axes=axes, dataaxis_1d=dataaxis_1d)
        
        ndim = get_ndim(axes)
        z = get_z(ndim)
        cell_plottings = PlotterNodeXY(data=projected_cell_lines, x="x", y="y", z=z, set_axequal=True)
    
    return cell_plottings