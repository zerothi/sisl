from typing import Literal, Optional, Sequence, Union

from sisl.viz.types import AtomSpec, AtomsStyleSpec, Axes, GeometryLike
from ..node import Node
from ..processors.geometry import sanitize_atoms, parse_atoms_style, gen_atoms_dataset, filter_atoms,\
    gen_bonds_dataset, find_all_bonds, parse_bonds_style, filter_bonds, bonds_to_lines,\
    tile_data_sc, stack_sc_data
from ..processors.coords import project_to_axes
from ..plotters import PlotterNodeXY, CompositePlotterNode
from ..plotters.cell import cell_plot_actions, get_ndim, get_z

from .plot import Plot

@Node.from_func
def _get_atom_mode(drawing_mode, ndim):

    if drawing_mode is None:
        if ndim == 3:
            return 'balls'
        else:
            return 'scatter'
    
    return drawing_mode
    
@Plot.from_func
def GeometryPlot(geometry: GeometryLike, axes: Axes = ["x", "y", "z"], atoms: AtomSpec = None, 
    atoms_style: Sequence[AtomsStyleSpec] = [], atoms_scale: float = 1., atoms_colorscale: Optional[str] = None,
    drawing_mode: Literal["scatter", "balls", None] = None,
    bind_bonds_to_ats: bool = True, points_per_bond: int = 2, bonds_style={}, 
    show_cell: Literal["box", "axes", False] = "box", cell_style={}, nsc: Sequence[int] = [1,1,1],
    dataaxis_1d=None
):
    # INPUTS ARE NOT GETTING PARSED BECAUSE WORKFLOWS RUN GET ON FINAL NODE
    # SO PARSING IS DELEGATED TO NODES.
    
    sanitized_atoms = sanitize_atoms(geometry, atoms=atoms)
    ndim = get_ndim(axes)
    z = get_z(ndim)

    # Atoms and bonds are processed in parallel paths, which means that one is able
    # to update without requiring the other. This means: 1) Faster updates if only one
    # of them needs to update; 2) It should be possible to run each path in a different
    # thread/process, potentially increasing speed.
    parsed_atom_style = parse_atoms_style(geometry, atoms_style=atoms_style, scale=atoms_scale)
    atoms_dataset = gen_atoms_dataset(parsed_atom_style)
    filtered_atoms = filter_atoms(atoms_dataset, sanitized_atoms)
    tiled_atoms = tile_data_sc(filtered_atoms, nsc=nsc)
    sc_atoms = stack_sc_data(tiled_atoms, newname="sc_atom", dims=["atom"])
    projected_atoms = project_to_axes(sc_atoms, axes=axes, sort_by_depth=True, dataaxis_1d=dataaxis_1d)

    atom_mode = _get_atom_mode(drawing_mode, ndim)
    atom_plottings = PlotterNodeXY(
        data=projected_atoms, x="x", y="y", z=z, width="size", what=atom_mode, colorscale=atoms_colorscale,
        set_axequal=True
    )
    
    # Here we start to process bonds
    bonds = find_all_bonds(geometry)
    parsed_bonds_style = parse_bonds_style(bonds, bonds_style)
    bonds_dataset = gen_bonds_dataset(bonds, parsed_bonds_style)
    filtered_bonds = filter_bonds(bonds_dataset, sanitized_atoms, bind_bonds_to_ats=bind_bonds_to_ats)
    tiled_bonds = tile_data_sc(filtered_bonds, nsc=nsc)
    
    projected_bonds = project_to_axes(tiled_bonds, axes=axes, dataaxis_1d=dataaxis_1d)
    sc_bonds = stack_sc_data(projected_bonds, newname="sc_bond_index", dims=["bond_index"])
    bond_lines = bonds_to_lines(projected_bonds, points_per_bond=points_per_bond)

    bond_plottings = PlotterNodeXY(data=bond_lines, x="x", y="y", z=z, set_axequal=True)
    
    # And now the cell
    cell_plottings = cell_plot_actions(
        cell=geometry, show_cell=show_cell, cell_style=cell_style,
        axes=axes, dataaxis_1d=dataaxis_1d
    )
        
    # And the arrows
    
    return CompositePlotterNode(bond_plottings, atom_plottings, cell_plottings, composite_method=None)