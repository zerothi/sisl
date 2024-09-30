# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

import sisl
from sisl import AtomicOrbital, Geometry
from sisl.messages import SislError
from sisl.viz.data import PDOSData
from sisl.viz.processors.orbital import (
    OrbitalQueriesManager,
    atom_data_from_orbital_data,
    reduce_orbital_data,
)

pytestmark = [pytest.mark.viz, pytest.mark.processors]


@pytest.fixture(scope="module")
def geometry():
    orbs = [
        AtomicOrbital("2sZ1"),
        AtomicOrbital("2sZ2"),
        AtomicOrbital("2pxZ1"),
        AtomicOrbital("2pyZ1"),
        AtomicOrbital("2pzZ1"),
        AtomicOrbital("2pxZ2"),
        AtomicOrbital("2pyZ2"),
        AtomicOrbital("2pzZ2"),
    ]

    atoms = [
        sisl.Atom(5, orbs),
        sisl.Atom(7, orbs),
    ]
    return sisl.geom.graphene(atoms=atoms)


@pytest.fixture(
    scope="module", params=["unpolarized", "polarized", "noncolinear", "spinorbit"]
)
def spin(request):
    return sisl.Spin(request.param)


@pytest.fixture(scope="module")
def orb_manager(geometry, spin):
    return OrbitalQueriesManager(geometry, spin=spin)


def test_get_orbitals(orb_manager, geometry: Geometry):
    orbs = orb_manager.get_orbitals({"atoms": [0]})
    assert len(orbs) == geometry.atoms.atom[0].no
    assert np.all(orbs == np.arange(geometry.atoms.atom[0].no))

    orbs = orb_manager.get_orbitals({"orbitals": [0, 1]})
    assert len(orbs) == 2
    assert np.all(orbs == np.array([0, 1]))


def test_get_atoms(orb_manager, geometry: Geometry):
    ats = orb_manager.get_atoms({"atoms": [0]})
    assert len(ats) == 1
    assert ats[0] == 0

    if geometry.orbitals[0] > 1:
        at_orbitals = [0, 1]
    else:
        at_orbitals = [0]

    ats = orb_manager.get_atoms({"orbitals": at_orbitals})
    assert len(ats) == 1
    assert np.all(ats == np.array([0]))


def test_split(orb_manager, geometry: Geometry):
    # Check that it can split over species
    queries = orb_manager.generate_queries(split="species")

    assert len(queries) == geometry.atoms.nspecies

    atom_tags = [atom.tag for atom in geometry.atoms.atom]

    for query in queries:
        assert isinstance(query, dict), f"Query is not a dict: {query}"

        assert "species" in query, f"Query does not have species: {query}"
        assert isinstance(query["species"], list)
        assert len(query["species"]) == 1
        assert query["species"][0] in atom_tags

    # Check that it can split over atoms
    queries = orb_manager.generate_queries(split="atoms")

    assert len(queries) == geometry.na

    for i_atom in range(geometry.na):
        query = queries[i_atom]

        assert isinstance(query, dict), f"Query is not a dict: {query}"

        assert "atoms" in query, f"Query does not have atoms: {query}"
        assert isinstance(query["atoms"], list)
        assert len(query["atoms"]) == 1
        assert query["atoms"][0] == i_atom

    # Check that it can split over n
    queries = orb_manager.generate_queries(split="l")

    assert len(queries) != 0

    for query in queries:
        assert isinstance(query, dict), f"Query is not a dict: {query}"

        assert "l" in query, f"Query does not have l: {query}"
        assert isinstance(query["l"], list)
        assert len(query["l"]) == 1
        assert isinstance(query["l"][0], int)


def test_double_split(orb_manager):
    # Check that it can split over two things at the same time
    queries = orb_manager.generate_queries(split="l+m")

    assert len(queries) != 0

    for query in queries:
        assert isinstance(query, dict), f"Query is not a dict: {query}"

        assert "l" in query, f"Query does not have l: {query}"
        assert isinstance(query["l"], list)
        assert len(query["l"]) == 1
        assert isinstance(query["l"][0], int)

        assert "l" in query, f"Query does not have l: {query}"
        assert isinstance(query["m"], list)
        assert len(query["m"]) == 1
        assert isinstance(query["m"][0], int)

        assert abs(query["m"][0]) <= query["l"][0]


def test_split_only(orb_manager, geometry):
    queries = orb_manager.generate_queries(
        split="species", only=[geometry.atoms.atom[0].tag]
    )

    assert len(queries) == 1
    assert queries[0]["species"] == [geometry.atoms.atom[0].tag]


def test_split_exclude(orb_manager, geometry):
    queries = orb_manager.generate_queries(
        split="species", exclude=[geometry.atoms.atom[0].tag]
    )

    assert len(queries) == geometry.atoms.nspecies - 1
    assert geometry.atoms.atom[0].tag not in [query["species"][0] for query in queries]


def test_constrained_split(orb_manager, geometry):
    queries = orb_manager.generate_queries(split="species", atoms=[0])

    assert len(queries) == 1
    assert queries[0]["species"] == [geometry.atoms.atom[0].tag]


def test_split_name(orb_manager, geometry):
    queries = orb_manager.generate_queries(split="species", name="Tag: $species")

    assert len(queries) == geometry.atoms.nspecies

    for query in queries:
        assert "name" in query, f"Query does not have name: {query}"
        assert query["name"] == f"Tag: {query['species'][0]}"


def test_sanitize_query(orb_manager, geometry):
    san_query = orb_manager.sanitize_query({"atoms": [0]})

    atom_orbitals = geometry.atoms.atom[0].orbitals

    assert len(san_query["orbitals"]) == len(atom_orbitals)
    assert np.all(san_query["orbitals"] == np.arange(len(atom_orbitals)))


def test_reduce_orbital_data(geometry, spin):
    data = PDOSData.toy_example(geometry=geometry, spin=spin)._data

    reduced = reduce_orbital_data(data, [{"name": "all"}])

    assert isinstance(reduced, xr.DataArray)

    for dim in data.dims:
        if dim == "orb":
            assert dim not in reduced.dims
        else:
            assert dim in reduced.dims
            assert len(data[dim]) == len(reduced[dim])

    assert "group" in reduced.dims
    assert len(reduced.group) == 1
    assert reduced.group[0] == "all"
    assert np.allclose(reduced.sel(group="all").values, data.sum("orb").values)

    data_no_geometry = data.copy()
    data_no_geometry.attrs.pop("geometry")

    with pytest.raises(SislError):
        reduced = reduce_orbital_data(data_no_geometry, [{"name": "all"}])


def test_reduce_orbital_data_spin(geometry, spin):
    data = PDOSData.toy_example(geometry=geometry, spin=spin)._data

    if spin.is_polarized:
        sel_total = reduce_orbital_data(data, [{"name": "all", "spin": "total"}])
        red_total = reduce_orbital_data(data, [{"name": "all"}], spin_reduce=np.sum)

        assert np.allclose(sel_total.values, red_total.values)


def test_reduce_orbital_data_no_spin(geometry, spin):
    """Reduce orbital data without reducing the spin dimension"""
    data = PDOSData.toy_example(geometry=geometry, spin=spin)._data

    reduced = reduce_orbital_data(
        data, [{"name": "all", "spin": "total"}], spin_reduce=False
    )

    assert np.all(data.spin == reduced.spin)


def test_atom_data_from_orbital_data(geometry: Geometry, spin):
    data = PDOSData.toy_example(geometry=geometry, spin=spin)._data

    atom_data = atom_data_from_orbital_data(data, geometry)

    assert isinstance(atom_data, xr.DataArray)

    for dim in data.dims:
        if dim == "orb":
            assert dim not in atom_data.dims
        else:
            assert dim in atom_data.dims
            assert len(data[dim]) == len(atom_data[dim])

    assert "atom" in atom_data.dims
    assert len(atom_data.atom) == geometry.na
    assert np.all(atom_data.atom == np.arange(geometry.na))

    atom_values = []
    firsto = geometry.firsto
    lasto = geometry.lasto
    for i in range(geometry.na):
        atom_values.append(data.sel(orb=slice(firsto[i], lasto[i])).sum("orb").values)

    assert np.allclose(atom_data.values, np.array(atom_values))
