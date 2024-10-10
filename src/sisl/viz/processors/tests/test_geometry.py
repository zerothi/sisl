# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

import sisl
from sisl.viz.processors.geometry import (
    add_xyz_to_bonds_dataset,
    add_xyz_to_dataset,
    bonds_to_lines,
    find_all_bonds,
    get_atoms_bonds,
    parse_atoms_style,
    sanitize_arrows,
    sanitize_atoms,
    sanitize_bonds_selection,
    stack_sc_data,
    style_bonds,
    tile_data_sc,
    tile_geometry,
)

pytestmark = [pytest.mark.viz, pytest.mark.processors]


@pytest.fixture(scope="module")
def geometry():
    return sisl.geom.bcc(2.93, "Au", True)


@pytest.fixture(scope="module")
def coords_dataset(geometry):
    return xr.Dataset(
        {"xyz": (("atom", "axis"), geometry.xyz)},
        coords={"axis": [0, 1, 2]},
        attrs={"geometry": geometry},
    )


def test_tile_geometry():
    geom = sisl.geom.graphene()

    tiled_geometry = tile_geometry(geom, (2, 3, 5))

    assert np.allclose(tiled_geometry.cell.T, geom.cell.T * (2, 3, 5))


def test_find_all_bonds():
    geom = sisl.geom.graphene()

    bonds = find_all_bonds(geom, 1.5)

    assert isinstance(bonds, xr.Dataset)

    assert "geometry" in bonds.attrs
    assert bonds.attrs["geometry"] is geom

    assert "bonds" in bonds.data_vars

    assert bonds.bonds.shape == (23, 2)

    # Now get bonds only for the unit cell
    geom.set_nsc([1, 1, 1])
    bonds = find_all_bonds(geom, 1.5)

    assert bonds.bonds.shape == (1, 2)
    assert np.all(bonds.bonds == (0, 1))

    # Run function with just one atom
    bonds = find_all_bonds(geom.sub(0), 1.5)


def test_get_atom_bonds():
    bonds = np.array([[0, 1], [0, 2], [1, 2]])

    mask = get_atoms_bonds(bonds, [0], ret_mask=True)

    assert isinstance(mask, np.ndarray)
    assert mask.dtype == bool
    assert np.all(mask == [True, True, False])

    atom_bonds = get_atoms_bonds(bonds, [0])

    assert isinstance(atom_bonds, np.ndarray)
    assert atom_bonds.shape == (2, 2)
    assert np.all(atom_bonds == [[0, 1], [0, 2]])


def test_sanitize_atoms():
    geom = sisl.geom.graphene()

    sanitized = sanitize_atoms(geom, 3)

    assert len(sanitized) == 1
    assert sanitized[0] == 3


def test_data_sc(coords_dataset):
    assert "isc" not in coords_dataset.dims

    # First, check that not tiling works as expected
    tiled = tile_data_sc(coords_dataset, nsc=(1, 1, 1))
    assert "isc" in tiled.dims
    assert len(tiled.isc) == 1
    assert np.all(tiled.sel(isc=0).xyz == coords_dataset.xyz)

    # Now, check that tiling works as expected
    tiled = tile_data_sc(coords_dataset, nsc=(2, 1, 1))

    assert "isc" in tiled.dims
    assert len(tiled.isc) == 2
    assert np.allclose(tiled.sel(isc=0).xyz, coords_dataset.xyz)
    assert np.allclose(
        tiled.sel(isc=1).xyz,
        coords_dataset.xyz + coords_dataset.attrs["geometry"].cell[0],
    )


def test_stack_sc_data(coords_dataset):
    tiled = tile_data_sc(coords_dataset, nsc=(3, 3, 1))

    assert "isc" in tiled.dims

    stacked = stack_sc_data(tiled, newname="sc_atom", dims=["atom"])

    assert "isc" not in stacked.dims
    assert "sc_atom" in stacked.dims
    assert len(stacked.sc_atom) == 9 * len(coords_dataset.atom)


@pytest.mark.parametrize("data_type", [list, dict])
def test_parse_atoms_style_empty(data_type):
    g = sisl.geom.graphene()
    styles = parse_atoms_style(g, data_type())

    assert isinstance(styles, xr.Dataset)

    assert "atom" in styles.coords
    assert len(styles.coords["atom"]) == 2

    for data_var in styles.data_vars:
        assert len(styles[data_var].shape) == 0


@pytest.mark.parametrize("data_type", [list, dict])
def test_parse_atoms_style_single_values(data_type):
    g = sisl.geom.graphene()

    unparsed = {"color": "green", "size": 14}
    if data_type == list:
        unparsed = [unparsed]

    styles = parse_atoms_style(g, unparsed)

    assert isinstance(styles, xr.Dataset)

    assert "atom" in styles.coords
    assert len(styles.coords["atom"]) == 2

    for data_var in styles.data_vars:
        assert len(styles[data_var].shape) == 0

        if data_var == "color":
            assert styles[data_var].values == "green"
        elif data_var == "size":
            assert styles[data_var].values == 14


def test_add_xyz_to_dataset(geometry):
    parsed_atoms_style = parse_atoms_style(geometry, {"color": "green", "size": 14})

    atoms_dataset = add_xyz_to_dataset(parsed_atoms_style)

    assert isinstance(atoms_dataset, xr.Dataset)

    assert "xyz" in atoms_dataset.data_vars
    assert atoms_dataset.xyz.shape == (geometry.na, 3)
    assert np.allclose(atoms_dataset.xyz, geometry.xyz)


@pytest.mark.parametrize("data_type", [list, dict])
def test_sanitize_arrows_empty(data_type):
    g = sisl.geom.graphene()
    arrows = sanitize_arrows(g, data_type(), atoms=None, ndim=3, axes="xyz")

    assert isinstance(arrows, list)

    assert len(arrows) == 0


def test_sanitize_arrows():
    data = np.array([[0, 0, 0], [1, 1, 1]])

    g = sisl.geom.graphene()

    unparsed = [{"data": data}]
    arrows = sanitize_arrows(g, unparsed, atoms=None, ndim=3, axes="xyz")

    assert isinstance(arrows, list)
    assert np.allclose(arrows[0]["data"], data)

    arrows_from_dict = sanitize_arrows(g, unparsed[0], atoms=None, ndim=3, axes="xyz")
    assert isinstance(arrows_from_dict, list)

    for k, v in arrows[0].items():
        if not isinstance(v, np.ndarray):
            assert arrows[0][k] == arrows_from_dict[0][k]


def test_style_bonds(geometry):
    bonds = find_all_bonds(geometry, 1.5)

    # Test no styles
    styled_bonds = style_bonds(bonds, {})

    assert isinstance(styled_bonds, xr.Dataset)
    assert "bonds" in styled_bonds.data_vars
    for k in ("color", "width", "opacity"):
        assert k in styled_bonds.data_vars, f"Missing {k}"
        assert styled_bonds[k].shape == (), f"Wrong shape for {k}"

    # Test single values
    styles = {"color": "green", "width": 14, "opacity": 0.2}
    styled_bonds = style_bonds(bonds, styles)
    assert isinstance(styled_bonds, xr.Dataset)
    assert "bonds" in styled_bonds.data_vars
    for k in ("color", "width", "opacity"):
        assert k in styled_bonds.data_vars, f"Missing {k}"
        assert styled_bonds[k].shape == (), f"Wrong shape for {k}"
        assert styled_bonds[k].values == styles[k], f"Wrong value for {k}"

    # Test callable
    def some_property(geometry, bonds):
        return np.arange(len(bonds))

    styles = {"color": some_property, "width": some_property, "opacity": some_property}
    styled_bonds = style_bonds(bonds, styles)
    assert isinstance(styled_bonds, xr.Dataset)
    assert "bonds" in styled_bonds.data_vars
    for k in ("color", "width", "opacity"):
        assert k in styled_bonds.data_vars, f"Missing {k}"
        assert styled_bonds[k].shape == (len(bonds.bonds),), f"Wrong shape for {k}"
        assert np.all(
            styled_bonds[k].values == np.arange(len(bonds.bonds))
        ), f"Wrong value for {k}"

    # Test scale
    styles = {"color": some_property, "width": some_property, "opacity": some_property}
    styled_bonds = style_bonds(bonds, styles, scale=2)
    assert isinstance(styled_bonds, xr.Dataset)
    assert "bonds" in styled_bonds.data_vars
    for k in ("color", "width", "opacity"):
        assert k in styled_bonds.data_vars, f"Missing {k}"
        assert styled_bonds[k].shape == (len(bonds.bonds),), f"Wrong shape for {k}"
        if k == "width":
            assert np.all(
                styled_bonds[k].values == 2 * np.arange(len(bonds.bonds))
            ), f"Wrong value for {k}"
        else:
            assert np.all(
                styled_bonds[k].values == np.arange(len(bonds.bonds))
            ), f"Wrong value for {k}"


def test_add_xyz_to_bonds_dataset(geometry):
    bonds = find_all_bonds(geometry, 1.5)

    xyz_bonds = add_xyz_to_bonds_dataset(bonds)

    assert isinstance(xyz_bonds, xr.Dataset)
    assert "xyz" in xyz_bonds.data_vars
    assert xyz_bonds.xyz.shape == (len(bonds.bonds), 2, 3)
    assert np.allclose(xyz_bonds.xyz[:, 0], geometry.xyz[np.ravel(bonds.bonds[:, 0])])


def test_sanitize_bonds_selection(geometry):
    bonds = find_all_bonds(geometry, 1.5)

    # No selection
    assert sanitize_bonds_selection(bonds) is None

    # No bonds
    bonds_sel = sanitize_bonds_selection(bonds, show_bonds=False)
    assert isinstance(bonds_sel, np.ndarray)
    assert len(bonds_sel) == 0

    # Assert not bound to atoms
    assert sanitize_bonds_selection(bonds, atoms=[0], bind_bonds_to_ats=False) is None

    # Assert bind to atoms. We check that all selected bonds have the only
    # requested atom.
    bonds_sel = sanitize_bonds_selection(bonds, atoms=[0], bind_bonds_to_ats=True)

    assert isinstance(bonds_sel, np.ndarray)
    assert (bonds.sel(bond_index=bonds_sel) == 0).any("bond_atom").all("bond_index")


def test_bonds_to_lines(geometry):
    bonds = find_all_bonds(geometry, 1.5)
    xyz_bonds = add_xyz_to_bonds_dataset(bonds)

    assert isinstance(xyz_bonds, xr.Dataset)
    assert "bond_atom" in xyz_bonds.dims
    assert len(xyz_bonds.bond_atom) == 2

    # No interpolation. Nan is added between bonds.
    bond_lines = bonds_to_lines(xyz_bonds)
    assert isinstance(bond_lines, xr.Dataset)
    assert "point_index" in bond_lines.dims
    assert len(bond_lines.point_index) == len(xyz_bonds.bond_index) * 3

    # Interpolation.
    bond_lines = bonds_to_lines(xyz_bonds, points_per_bond=10)
    assert isinstance(bond_lines, xr.Dataset)
    assert "point_index" in bond_lines.dims
    assert len(bond_lines.point_index) == len(xyz_bonds.bond_index) * 11
