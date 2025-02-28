# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import asdict
from typing import Any, Optional, TypedDict, Union

import numpy as np
import numpy.typing as npt
from xarray import Dataset

from sisl import BrillouinZone, Geometry, PeriodicTable
from sisl.messages import warn
from sisl.typing import AtomsIndex
from sisl.utils.mathematics import fnorm
from sisl.viz.types import AtomArrowSpec

from ..data_sources.atom_data import AtomIsGhost, AtomPeriodicTable, JMolAtomColors
from .coords import CoordsDataset, projected_1Dcoords, projected_2Dcoords

# from ...types import AtomsIndex, GeometryLike, PathLike

GeometryDataset = CoordsDataset
AtomsDataset = GeometryDataset
BondsDataset = GeometryDataset

# class GeometryData(DataSource):
#     pass

# @GeometryData.from_func
# def geometry_from_file(file: PathLike) -> Geometry:
#     return Geometry.new(file)

# @GeometryData.from_func
# def geometry_from_obj(obj: GeometryLike) -> Geometry:
#     return Geometry.new(obj)


def tile_geometry(geometry: Geometry, nsc: tuple[int, int, int]) -> Geometry:
    """Tiles a geometry along the three lattice vectors.

    Parameters
    -----------
    geometry: sisl.Geometry
        the geometry to be tiled.
    nsc: tuple[int, int, int]
        the number of repetitions along each lattice vector.
    """

    tiled_geometry = geometry.copy()
    for ax, reps in enumerate(nsc):
        tiled_geometry = tiled_geometry.tile(reps, ax)

    return tiled_geometry


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
            summed_radius = pt.radius(
                [abs(geometry.atoms[at].Z), abs(geometry.atoms[neigh % geometry.na].Z)]
            ).sum()
            bond_thresh = (1 + tol) * summed_radius
            if bond_thresh > fnorm(geometry[neigh] - geometry[at]):
                bonds.append([at, neigh])

    if len(bonds) == 0:
        bonds = np.empty((0, 2), dtype=np.int64)

    return Dataset(
        {"bonds": (("bond_index", "bond_atom"), np.array(bonds, dtype=np.int64))},
        coords={"bond_index": np.arange(len(bonds)), "bond_atom": [0, 1]},
        attrs={"geometry": geometry},
    )


def get_atoms_bonds(
    bonds: npt.NDArray[np.int32], atoms: npt.ArrayLike, ret_mask: bool = False
) -> npt.NDArray[Union[np.float64, np.bool8]]:
    """Gets the bonds where the given atoms are involved.

    Parameters
    -----------
    bonds: np.ndarray of shape (nbonds, 2)
        Pairs of indices of atoms that are bonded.
    atoms: np.ndarray of shape (natoms,)
        Indices of the atoms for which we want to keep the bonds.
    """
    # For each bond, we check if one of the desired atoms is involved
    mask = np.isin(bonds, atoms).any(axis=-1)
    if ret_mask:
        return mask

    return bonds[mask]


def sanitize_atoms(
    geometry: Geometry, atoms: AtomsIndex = None
) -> npt.NDArray[np.int32]:
    """Sanitizes the atoms argument to a np.ndarray of shape (natoms,).

    This is the same as `geometry._sanitize_atoms` but ensuring that the
    result is a numpy array of 1 dimension.

    Parameters
    -----------
    geometry: sisl.Geometry
        geometry that will sanitize the atoms
    atoms: AtomsIndex
        anything that `Geometry` can sanitize.
    """
    atoms = geometry._sanitize_atoms(atoms)
    return np.atleast_1d(atoms)


def tile_data_sc(
    geometry_data: GeometryDataset, nsc: tuple[int, int, int] = (1, 1, 1)
) -> GeometryDataset:
    """Tiles coordinates from unit cell to a supercell.

    Parameters
    -----------
    geometry_data: GeometryDataset
        the dataset containing the coordinates to be tiled.
    nsc: np.ndarray of shape (3,)
        the number of repetitions along each lattice vector.
    """
    # Get the total number of supercells
    total_sc = np.prod(nsc)

    xyz_shape = geometry_data.xyz.shape

    # Create a fake geometry
    fake_geom = Geometry(
        xyz=geometry_data.xyz.values.reshape(-1, 3),
        lattice=geometry_data.geometry.lattice.copy(),
        atoms=1,
    )

    sc_offs = np.array(list(itertools.product(*[range(n) for n in nsc])))

    sc_xyz = np.array([fake_geom.axyz(isc=sc_off) for sc_off in sc_offs]).reshape(
        (total_sc, *xyz_shape)
    )

    # Build the new dataset
    sc_atoms = geometry_data.assign({"xyz": (("isc", *geometry_data.xyz.dims), sc_xyz)})
    sc_atoms = sc_atoms.assign_coords(isc=range(total_sc))

    return sc_atoms


def stack_sc_data(
    geometry_data: GeometryDataset, newname: str, dims: Sequence[str]
) -> GeometryDataset:
    """Stacks the supercell coordinate with others.

    Parameters
    -----------
    geometry_data: GeometryDataset
        the dataset for which we want to stack the supercell coordinates.
    newname: str
    """

    return geometry_data.stack(**{newname: ["isc", *dims]}).transpose(newname, ...)


class AtomsStyleSpec(TypedDict):
    color: Any
    size: Any
    opacity: Any
    vertices: Any


def parse_atoms_style(
    geometry: Geometry, atoms_style: Sequence[AtomsStyleSpec], scale: float = 1.0
) -> AtomsDataset:
    """Parses atom style specifications to a dataset of styles.

    Parameters
    -----------
    geometry: sisl.Geometry
        the geometry for which the styles are parsed.
    atoms_style: Sequence[AtomsStyleSpec]
        the styles to be parsed.
    scale: float
        the scale to be applied to the size of the atoms.
    """
    if isinstance(atoms_style, dict):
        atoms_style = [atoms_style]

    # Add the default styles first
    atoms_style = [
        {
            "color": JMolAtomColors(),
            "size": AtomPeriodicTable(what="radius"),
            "opacity": AtomIsGhost(fill_true=0.4, fill_false=1.0),
            "vertices": 15,
            "border_width": 1,
            "border_color": "black",
        },
        *atoms_style,
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
        "color": np.empty((geometry.na,), dtype=object),
        "size": np.empty((geometry.na,), dtype=float),
        "vertices": np.empty((geometry.na,), dtype=int),
        "opacity": np.empty((geometry.na), dtype=float),
        "border_width": np.empty((geometry.na,), dtype=int),
        "border_color": np.empty((geometry.na,), dtype=object),
    }

    # Go specification by specification and apply the styles
    # to the corresponding atoms.
    for style_spec in atoms_style:
        atoms = geometry._sanitize_atoms(style_spec.get("atoms"))
        for key in parsed_atoms_style:
            if style_spec.get(key) is not None:
                style = style_spec[key]

                if callable(style):
                    style = style(geometry=geometry, atoms=atoms)

                parsed_atoms_style[key][atoms] = _tile_if_needed(atoms, style)

    # Apply the scale
    parsed_atoms_style["size"] = parsed_atoms_style["size"] * scale
    # Convert colors to numbers if possible
    try:
        parsed_atoms_style["color"] = parsed_atoms_style["color"].astype(float)
    except:
        pass

    # Add coordinates to the values according to their unique dimensionality.
    data_vars = {}
    for k, value in parsed_atoms_style.items():
        if k != "color" or value.dtype not in (float, int):
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


def sanitize_arrows(
    geometry: Geometry,
    arrows: Sequence[AtomArrowSpec],
    atoms: AtomsIndex,
    ndim: int,
    axes: Sequence[str],
) -> list[dict]:
    """Sanitizes a list of arrow specifications.

    Each arrow specification in the output has the atoms sanitized and
    the data with the shape (natoms, ndim).

    Parameters
    ----------
    geometry: sisl.Geometry
        the geometry for which the arrows are sanitized.
    arrows: Sequence[AtomArrowSpec]
        unsanitized arrow specifications.
    atoms: AtomsIndex
        atoms for which we want the data. This means that data
        will be filtered to only contain the atoms in this argument.
    ndim: int
        dimensionality of the space into which arrows must be projected.
    axes: Sequence[str]
        Axes onto which the arrows must be projected.
    """
    atoms: np.ndarray = geometry._sanitize_atoms(atoms)

    def _sanitize_spec(arrow_spec):
        arrow_spec = AtomArrowSpec(**arrow_spec)
        arrow_spec = asdict(arrow_spec)

        arrow_spec["atoms"] = np.atleast_1d(
            geometry._sanitize_atoms(arrow_spec["atoms"])
        )
        arrow_atoms = arrow_spec["atoms"]

        not_displayed = set(arrow_atoms) - set(atoms)
        if not_displayed:
            warn(
                f"Arrow data for atoms {not_displayed} will not be displayed because these atoms are not displayed."
            )
        if set(atoms) == set(atoms) - set(arrow_atoms):
            # Then it makes no sense to store arrows, as nothing will be drawn
            return None

        arrow_data = np.full((geometry.na, ndim), np.nan, dtype=np.float64)
        provided_data = np.array(arrow_spec["data"])

        # Get the projected directions if we are not in 3D.
        if ndim == 1:
            provided_data = projected_1Dcoords(geometry, provided_data, axis=axes[0])
            provided_data = np.expand_dims(provided_data, axis=-1)
        elif ndim == 2:
            provided_data = projected_2Dcoords(
                geometry, provided_data, xaxis=axes[0], yaxis=axes[1]
            )

        arrow_data[arrow_atoms] = provided_data
        arrow_spec["data"] = arrow_data[atoms]

        # arrow_spec["data"] = self._tile_atomic_data(arrow_spec["data"])

        return arrow_spec

    if isinstance(arrows, dict):
        if arrows == {}:
            arrows = []
        else:
            arrows = [arrows]

    san_arrows = [_sanitize_spec(arrow_spec) for arrow_spec in arrows]

    return [arrow_spec for arrow_spec in san_arrows if arrow_spec is not None]


def add_xyz_to_dataset(dataset: AtomsDataset) -> AtomsDataset:
    """Adds the xyz data variable to a dataset with associated geometry.

    The new xyz data variable contains the coordinates of the atoms.

    Parameters
    -----------
    dataset: AtomsDataset
        the dataset to be augmented with xyz data.
    """
    geometry = dataset.attrs["geometry"]

    xyz_ds = Dataset(
        {"xyz": (("atom", "axis"), geometry.xyz)},
        coords={"axis": [0, 1, 2]},
        attrs={"geometry": geometry},
    )

    return xyz_ds.merge(dataset, combine_attrs="no_conflicts")


class BondsStyleSpec(TypedDict):
    color: Any
    width: Any
    opacity: Any


def style_bonds(
    bonds_data: BondsDataset, bonds_style: BondsStyleSpec, scale: float = 1.0
) -> BondsDataset:
    """Adds styles to a bonds dataset.

    Parameters
    -----------
    bonds_data: BondsDataset
        the bonds that need to be styled.
        This can come from the `find_all_bonds` function.
    bonds_style: Sequence[BondsStyleSpec]
        the styles to be parsed.
    scale: float
        the scale to be applied to the width of the bonds.
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
        bonds_style,
    ]

    # Initialize the styles.
    # Potentially bond styles could have two styles, one for each halve.
    parsed_bonds_style = {
        "color": np.empty((nbonds,), dtype=object),
        "width": np.empty((nbonds,), dtype=float),
        "opacity": np.empty((nbonds,), dtype=float),
    }

    # Go specification by specification and apply the styles
    # to the corresponding bonds. Note that we still have no way of
    # selecting bonds, so for now we just apply the styles to all bonds.
    for style_spec in bonds_styles:
        for key in parsed_bonds_style:
            style = style_spec.get(key)
            if style is None:
                continue
            if callable(style):
                style = style(geometry=geometry, bonds=bonds_data.bonds)

            parsed_bonds_style[key][:] = style

    # Apply the scale
    parsed_bonds_style["width"] = parsed_bonds_style["width"] * scale
    # Convert colors to float datatype if possible
    try:
        parsed_bonds_style["color"] = parsed_bonds_style["color"].astype(float)
    except ValueError:
        pass

    # Add coordinates to the values according to their unique dimensionality.
    data_vars = {}
    for k, value in parsed_bonds_style.items():
        if k != "color" or value.dtype not in (float, int):
            unique = np.unique(value)
            if len(unique) == 1:
                data_vars[k] = unique[0]
                continue

        data_vars[k] = ("bond_index", value)

    return bonds_data.assign(data_vars)


def add_xyz_to_bonds_dataset(bonds_data: BondsDataset) -> BondsDataset:
    """Adds the coordinates of the bonds endpoints to a bonds dataset.

    Parameters
    -----------
    bonds_data: BondsDataset
        the bonds dataset to be augmented with xyz data.
    """
    geometry = bonds_data.attrs["geometry"]

    def _bonds_xyz(ds):
        bonds_shape = ds.bonds.shape
        bonds_xyz = geometry[ds.bonds.values.reshape(-1)].reshape((*bonds_shape, 3))
        return (("bond_index", "bond_atom", "axis"), bonds_xyz)

    return bonds_data.assign({"xyz": _bonds_xyz})


def sanitize_bonds_selection(
    bonds_data: BondsDataset,
    atoms: Optional[npt.NDArray[np.int32]] = None,
    bind_bonds_to_ats: bool = False,
    show_bonds: bool = True,
) -> Union[np.ndarray, None]:
    """Sanitizes bonds selection, unifying multiple parameters into a single value

    Parameters
    -----------
    bonds_data: BondsDataset
        the bonds dataset containing the already computed bonds.
    atoms: np.ndarray of shape (natoms,)
        the atoms for which we want to keep the bonds.
    bind_bonds_to_ats: bool
        if True, the bonds will be bound to the atoms,
        so that if an atom is not displayed, its bonds
        will not be displayed either.
    show_bonds: bool
        if False, no bonds will be displayed.
    """
    if not show_bonds:
        return np.array([], dtype=np.int64)
    elif bind_bonds_to_ats and atoms is not None:
        return get_atoms_bonds(bonds_data.bonds, atoms, ret_mask=True)
    else:
        return None


def bonds_to_lines(bonds_data: BondsDataset, points_per_bond: int = 2) -> BondsDataset:
    """Computes intermediate points between the endpoints of the bonds by interpolation.

    Bonds are concatenated into a single dimension "point index", and NaNs
    are added between bonds.

    Parameters
    -----------
    bonds_data: BondsDataset
        the bonds dataset containing the endpoints of the bonds.
    points_per_bond: int
        the number of points to be computed between the endpoints,
        including the endpoints.
    """
    if points_per_bond > 2:
        bonds_data = bonds_data.interp(bond_atom=np.linspace(0, 1, points_per_bond))

    bonds_data = bonds_data.reindex(
        {"bond_atom": [*bonds_data.bond_atom.values, 2]}
    ).stack(point_index=bonds_data.xyz.dims[:-1])

    return bonds_data


def sites_obj_to_geometry(sites_obj: BrillouinZone):
    """Converts anything that contains sites into a geometry.

    Possible conversions:
        - BrillouinZone object to geometry, kpoints to atoms.

    Parameters
    -----------
    sites_obj
        the object to be converted.
    """

    if isinstance(sites_obj, BrillouinZone):
        return Geometry(sites_obj.k.dot(sites_obj.rcell), lattice=sites_obj.rcell)
    else:
        raise ValueError(
            f"Cannot convert {sites_obj.__class__.__name__} to a geometry."
        )


def get_sites_units(sites_obj: BrillouinZone):
    """Units of space for an object that is to be converted into a geometry"""
    if isinstance(sites_obj, BrillouinZone):
        return "1/Ang"
    else:
        return ""
