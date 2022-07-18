import itertools
from typing import Optional, Sequence, TypedDict, Any, Union
import numpy as np
import numpy.typing as npt
from xarray import Dataset

from sisl import Geometry, PeriodicTable
from sisl.supercell import SuperCell, SuperCellChild
from sisl.utils.mathematics import fnorm
from sisl.viz.nodes.data_sources.atom_data import AtomDefaultColors, AtomIsGhost, AtomPeriodicTable
from sisl.viz.nodes.data_sources.data_source import DataSource
from sisl.viz.nodes.node import Node

from .coords import CoordsDataset
from ...types import AtomSpec, GeometryLike, PathLike

GeometryDataset = CoordsDataset
AtomsDataset = GeometryDataset
BondsDataset = GeometryDataset

class GeometryData(DataSource):
    pass

@GeometryData.from_func
def geometry_from_file(file: PathLike) -> Geometry:
    return Geometry.new(file)

@GeometryData.from_func
def geometry_from_obj(obj: GeometryLike) -> Geometry:
    return Geometry.new(obj)

class GeometryModifier(Node):
    pass

@GeometryModifier.from_func
def tile_geometry(geometry: Geometry, nsc: npt.ArrayLike) -> Geometry:
    tiled_geometry = geometry
    nsc = np.array(nsc)
    for ax, reps in enumerate(nsc):
        tiled_geometry = tiled_geometry.tile(reps, ax)

    return tiled_geometry

class GeometryMeasure(Node):
    pass

@GeometryMeasure.from_func
def find_all_bonds(geometry: Geometry, tol: float = 0.2) -> BondsDataset:
    """
    Finds all bonds present in a geometry.

    Parameters
    -----------
    geometry: sisl.Geometry
        the structure where the bonds should be found.
    tol: float
        the fraction that the distance between atoms is allowed to differ from
        the "standard" in order to be considered a bond.

    Return
    ---------
    np.ndarray of shape (nbonds, 2)
        each item of the array contains the 2 indices of the atoms that participate in the
        bond.
    """
    pt = PeriodicTable()

    bonds = []
    for at in geometry:
        neighs: npt.NDArray[np.int32] = geometry.close(at, R=[0.1, 3])[-1]

        for neigh in neighs[neighs > at]:
            summed_radius = pt.radius([abs(geometry.atoms[at].Z), abs(geometry.atoms[neigh % geometry.na].Z)]).sum()
            bond_thresh = (1+tol) * summed_radius
            if  bond_thresh > fnorm(geometry[neigh] - geometry[at]):
                bonds.append([at, neigh])

    return Dataset({
            "bonds": (("bond_index", "bond_atom"), np.array(bonds, dtype=np.int64))
        },
        coords={"bond_index": np.arange(len(bonds)), "bond_atom": [0, 1]}, 
        attrs={"geometry": geometry}
    )

@GeometryMeasure.from_func
def get_cell_corners(cell: Union[npt.ArrayLike, SuperCell, SuperCellChild], unique: bool = False) -> npt.NDArray[np.float64]:
    """Gets the coordinates of a cell's corners.

    Parameters
    ----------
    cell: np.ndarray of shape (3, 3)
        the cell for which you want the corner's coordinates.
    unique: bool, optional
        if `False`, a full path to draw a cell is returned.
        if `True`, only unique points are returned, in no particular order.

    Returns
    ---------
    np.ndarray of shape (x, 3)
        where x is 16 if unique=False and 8 if unique=True.
    """
    if isinstance(cell, (SuperCell, SuperCellChild)):
        cell = cell.cell

    if unique:
        verts = list(itertools.product([0, 1], [0, 1], [0, 1]))
    else:
        # Define the vertices of the cube. They follow an order so that we can
        # draw a line that represents the cell's box
        verts = [
            (0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 1, 1), (0, 1, 1), (0, 1, 0),
            (np.nan, np.nan, np.nan),
            (0, 1, 1), (0, 0, 1), (0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1),
            (np.nan, np.nan, np.nan),
            (1, 1, 0), (1, 0, 0),
            (np.nan, np.nan, np.nan),
            (1, 1, 1), (1, 0, 1)
        ]

    verts = np.array(verts, dtype=np.float64)

    return verts.dot(cell)

class GeometryHelper(Node):
    pass

@GeometryHelper.from_func
def get_atoms_bonds(bonds: npt.NDArray[np.int32], atoms: npt.ArrayLike, ret_mask: bool = False) -> npt.NDArray[Union[np.float64, np.bool8]]:
    """
    Gets the bonds where the given atoms are involved
    """
    # For each bond, we check if one of the desired atoms is involved
    mask = np.isin(bonds, atoms).any(axis=-1)
    if ret_mask:
        return mask
    
    return bonds[mask]

@GeometryHelper.from_func
def sanitize_atoms(geometry: Geometry, atoms: AtomSpec = None) -> npt.NDArray[np.int32]:
    atoms = geometry._sanitize_atoms(atoms)
    return np.atleast_1d(atoms)

class GeometryProcessor(Node):
    pass

@GeometryProcessor.from_func
def tile_data_sc(geometry_data: GeometryDataset, nsc: npt.NDArray[np.int32] = [1,1,1]) -> GeometryDataset:
    # Get the total number of supercells
    total_sc = np.prod(nsc)
    
    # Generate the offsets for each cell
    offsets = np.moveaxis(np.mgrid[0:nsc[0], 0:nsc[1], 0:nsc[2]], 0, -1).reshape(-1, 3)
    offsets = offsets @ geometry_data.geometry.cell
    
    # Calculate all the coordinates of the atoms for each cell, adding the offsets
    # We add an extra dimension on the first axis for the index of the sc.
    xyz = geometry_data.xyz.values
    tiles = np.ones(len(xyz.shape), dtype=int)
    tiles[-2] = total_sc
    sc_xyz = np.tile(np.expand_dims(xyz, -2), tiles)
    sc_xyz = np.moveaxis(sc_xyz + offsets, -2, 0)
    
    # Build the new dataset
    sc_atoms = geometry_data.assign({"xyz": (("isc", *geometry_data.xyz.dims), sc_xyz)})
    sc_atoms = sc_atoms.assign_coords(isc=range(total_sc))
    
    return sc_atoms

@GeometryProcessor.from_func
def stack_sc_data(geometry_data: GeometryDataset, newname: str, dims: Sequence[str]) -> GeometryDataset:
    return geometry_data.stack(**{newname: ["isc", *dims]}).transpose(newname, ...)

class AtomsProcessor(Node):
    pass

class AtomsStyleSpec(TypedDict):
    color: Any
    size: Any
    opacity: Any
    vertices: Any

@AtomsProcessor.from_func
def parse_atoms_style(geometry: Geometry, atoms_style: Sequence[AtomsStyleSpec], scale: float = 1.) -> AtomsDataset:
    """Parses the `atoms_style` setting to a dictionary of style specifications.

    Parameters
    -----------
    atoms_style:
        the value of the atoms_style setting.
    """

    # Add the default styles first
    atoms_style = [
        {
            "color": AtomDefaultColors(),
            "size": AtomPeriodicTable(what="radius"),
            "opacity": AtomIsGhost(fill_true=0.4, fill_false=1.),
            "vertices": 15,
        },
        *atoms_style
    ]

    def _tile_if_needed(atoms, spec):
        """Function that tiles an array style specification.

        It does so if the specification needs to be applied to more atoms
        than items are in the array."""
        if isinstance(spec, (tuple, list, np.ndarray)):
            n_ats = len(atoms)
            n_spec = len(spec)
            if n_ats != n_spec and n_ats % n_spec == 0:
                spec = np.tile(spec, n_ats // n_spec)
        return spec

    # Initialize the styles.
    parsed_atoms_style = {
        "color": np.empty((geometry.na, ), dtype=object),
        "size": np.empty((geometry.na, ), dtype=float),
        "vertices": np.empty((geometry.na, ), dtype=int),
        "opacity": np.empty((geometry.na), dtype=float),
    }

    # Go specification by specification and apply the styles
    # to the corresponding atoms.
    for style_spec in atoms_style:
        atoms = geometry._sanitize_atoms(style_spec.get("atoms"))
        for key in parsed_atoms_style:
            if style_spec.get(key) is not None:
                style = style_spec[key]

                if isinstance(style, Node):
                    style = style.get(geometry=geometry, atoms=atoms)

                parsed_atoms_style[key][atoms] = _tile_if_needed(atoms, style)

    # Apply the scale
    parsed_atoms_style['size'] = parsed_atoms_style['size'] * scale
    # Convert colors to numbers if possible
    try:
        parsed_atoms_style['color'] = parsed_atoms_style['color'].astype(float)
    except:
        pass

    # Add coordinates to the values according to their unique dimensionality.
    data_vars = {}
    for k, value in parsed_atoms_style.items():
        if (k != "color" or value.dtype not in (float, int)):
            unique = np.unique(value)
            if len(unique) == 1:
                data_vars[k] = unique[0]
                continue

        data_vars[k] = ("atom", value)
    
    return Dataset(
        data_vars,
        coords={"atom": range(geometry.na)},
        attrs={"geometry": geometry},
    )

@AtomsProcessor.from_func
def gen_atoms_dataset(parsed_atoms_style: AtomsDataset) -> AtomsDataset:
    geometry = parsed_atoms_style.attrs['geometry']

    xyz_ds = Dataset({"xyz": (("atom", "axis"), geometry.xyz)}, coords={"axis": [0,1,2]}, attrs={"geometry": geometry})

    return xyz_ds.merge(parsed_atoms_style, combine_attrs="no_conflicts")

@AtomsProcessor.from_func
def filter_atoms(atoms_data: AtomsDataset, atoms: Union[npt.NDArray[np.int32], None] = None, show_atoms: bool = True) -> AtomsDataset:
    if show_atoms == False:
        atoms = []
    if atoms is not None:
        atoms_data = atoms_data.sel(atom=atoms)
    return atoms_data

class BondsProcessor(Node):
    pass

class BondsStyleSpec(TypedDict):
    color: Any
    width: Any
    opacity: Any

@BondsProcessor.from_func
def parse_bonds_style(bonds_data: BondsDataset, bonds_style: BondsStyleSpec) -> BondsDataset:
    """Parses the `bonds_style` setting to a dictionary of style specifications.

    Parameters
    -----------
    bonds_style:
        the value of the bonds_style setting.
    """
    geometry = bonds_data.geometry

    nbonds = bonds_data.bonds.shape[0]

    # Add the default styles first
    bonds_styles: Sequence[BondsStyleSpec] = [
        {
            "color": "gray",
            "width": 1,
            "opacity": 1,
        },
        bonds_style
    ]

    # Initialize the styles.
    # Potentially bond styles could have two styles, one for each halve.
    parsed_bonds_style = {
        "color": np.empty((nbonds, ), dtype=object),
        "width": np.empty((nbonds, ), dtype=float),
        "opacity": np.empty((nbonds, ), dtype=float),
    }

    # Go specification by specification and apply the styles
    # to the corresponding bonds. Note that we still have no way of
    # selecting bonds, so for now we just apply the styles to all bonds.
    for style_spec in bonds_styles:
        for key in parsed_bonds_style:
            style = style_spec.get(key)
            if style is None:
                continue
            if isinstance(style, Node):
                style = style.get(geometry, bonds_data.bonds)

            parsed_bonds_style[key][:] = style
    
    # Add coordinates to the values according to their unique dimensionality.
    data_vars = {}
    for k, value in parsed_bonds_style.items():
        if (k != "color" or value.dtype not in (float, int)):
            unique = np.unique(value)
            if len(unique) == 1:
                data_vars[k] = unique[0]
                continue

        data_vars[k] = ("bond_index", value)

    return Dataset(
        data_vars,
        coords={"bond_index": range(nbonds)},
        attrs=bonds_data.attrs
    )

@BondsProcessor.from_func
def gen_bonds_dataset(bonds_data: BondsDataset, parsed_bonds_style: BondsDataset) -> BondsDataset:
    geometry = bonds_data.attrs['geometry']

    bonds_ds = bonds_data.assign({"xyz": lambda ds: (("bond_index", "bond_atom", "axis"), geometry[ds.bonds.values])})

    return bonds_ds.merge(parsed_bonds_style, combine_attrs="no_conflicts")

@BondsProcessor.from_func
def filter_bonds(bonds_data: BondsDataset, atoms: Optional[npt.NDArray[np.int32]] = None, bind_bonds_to_ats: bool = False) -> BondsDataset:
    if bind_bonds_to_ats and atoms is not None:
        bonds_mask = get_atoms_bonds(bonds_data.bonds, atoms, ret_mask=True)
        return bonds_data.sel(bond_index=bonds_mask)
    return bonds_data

@BondsProcessor.from_func
def bonds_to_lines(bonds_data: BondsDataset, points_per_bond: int = 2) -> BondsDataset:
    if points_per_bond > 2:
        bonds_data = bonds_data.interp(bond_atom=np.linspace(0, 1, points_per_bond))

    bonds_data = bonds_data.reindex({"bond_atom": [*bonds_data.bond_atom.values, 2]}).stack(point_index=bonds_data.xyz.dims[:-1])
    
    return bonds_data


