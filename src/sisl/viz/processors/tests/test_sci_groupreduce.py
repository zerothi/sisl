# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

import sisl
from sisl.viz.processors.atom import reduce_atom_data

pytestmark = [pytest.mark.viz, pytest.mark.processors]


@pytest.fixture(scope="module")
def atom_x():
    geom = sisl.geom.graphene().tile(20, 1).tile(20, 0)

    return xr.DataArray(
        geom.xyz[:, 0], coords=[("atom", range(geom.na))], attrs={"geometry": geom}
    )


@pytest.fixture(scope="module")
def atom_xyz():
    geom = sisl.geom.graphene().tile(20, 1).tile(20, 0)

    return xr.Dataset(
        {
            "x": xr.DataArray(geom.xyz[:, 0], coords=[("atom", range(geom.na))]),
            "y": xr.DataArray(geom.xyz[:, 1], coords=[("atom", range(geom.na))]),
            "z": xr.DataArray(geom.xyz[:, 2], coords=[("atom", range(geom.na))]),
        },
        attrs={"geometry": geom},
    )


def test_reduce_atom_dataarray(atom_x):
    grouped = reduce_atom_data(
        atom_x,
        [{"atoms": [0, 1], "name": "first"}, {"atoms": [5, 6], "name": "second"}],
        reduce_func=np.sum,
        groups_dim="group",
    )

    assert isinstance(grouped, xr.DataArray)
    assert float(grouped.sel(group="first")) == np.sum(atom_x.values[0:2].sum())
    assert float(grouped.sel(group="second")) == np.sum(atom_x.values[5:7].sum())


def test_reduce_atom_dataarray_cat(atom_x):
    """We test that the atoms field is correctly sanitized using the geometry attached to the xarray object."""
    grouped = reduce_atom_data(
        atom_x,
        [
            {"atoms": {"x": (0, 10)}, "name": "first"},
            {"atoms": {"x": (10, None)}, "name": "second"},
        ],
        reduce_func=np.max,
        groups_dim="group",
    )

    assert isinstance(grouped, xr.DataArray)
    assert float(grouped.sel(group="first")) <= 10
    assert float(grouped.sel(group="second")) == atom_x.max()


def test_reduce_atom_cat_nogeom(atom_x):
    """We test that the atoms field is correctly sanitized using the geometry attached to the xarray object."""
    atom_x = atom_x.copy()
    geometry = atom_x.attrs["geometry"]

    # Remove the geometry
    atom_x.attrs = {}

    # Without a geometry, it should fail to sanitize atoms specifications
    with pytest.raises(Exception):
        grouped = reduce_atom_data(
            atom_x,
            [
                {"atoms": {"x": (0, 10)}, "name": "first"},
                {"atoms": {"x": (10, None)}, "name": "second"},
            ],
            reduce_func=np.max,
            groups_dim="group",
        )

    # If we explicitly pass the geometry it should again be able to sanitize the atoms
    grouped = reduce_atom_data(
        atom_x,
        [
            {"atoms": {"x": (0, 10)}, "name": "first"},
            {"atoms": {"x": (10, None)}, "name": "second"},
        ],
        geometry=geometry,
        reduce_func=np.max,
        groups_dim="group",
    )

    assert isinstance(grouped, xr.DataArray)
    assert float(grouped.sel(group="first")) <= 10
    assert float(grouped.sel(group="second")) == atom_x.max()


def test_reduce_atom_dataset_cat(atom_xyz):
    """We test that the atoms field is correctly sanitized using the geometry attached to the xarray object."""
    grouped = reduce_atom_data(
        atom_xyz,
        [
            {"atoms": {"x": (0, 10), "y": (1, 3)}, "name": "first"},
            {"atoms": {"x": (10, None)}, "name": "second"},
        ],
        reduce_func=np.max,
        groups_dim="group",
    )

    assert isinstance(grouped, xr.Dataset)
    assert float(grouped.sel(group="first").x) <= 10
    assert float(grouped.sel(group="second").x) == atom_xyz.x.max()
    assert float(grouped.sel(group="first").y) <= 3
    assert float(grouped.sel(group="second").y) == atom_xyz.y.max()
