import pytest

from dataclasses import fields

import sisl
from sisl.viz.input_fields import OrbitalQueries
from sisl.viz.types import OrbitalStyleQuery

def test_orbital_queries_init():
    field = OrbitalQueries()

def test_orbital_queries_fields():

    field = OrbitalQueries()

    # Check that all the orbital query fields have been transferred.
    item_input = field.params.item_input
    for query_field in fields(OrbitalStyleQuery):
        assert query_field.name in item_input.params.fields
    
def test_update_options():

    field = OrbitalQueries()

    geom = sisl.geom.graphene(atoms=[
        sisl.Atom("B", orbitals=["1s", "2s"]),
        sisl.Atom("N", orbitals=["1p", "2p"])
    ])
    # Provide geometry and spin so that the options of the field can be updated.
    field.update_options(geom, "polarized")

    assert set(field.get_query_param("species").get_options()) == set(["B", "N"])
    assert set(field.get_query_param("orbitals").get_options()) == set(["1s", "2s", "1p", "2p"])
    assert set(field.get_query_param("spin").get_options()) == set([0, 1])


split_params = pytest.mark.parametrize(
    ["on", "expected_keys"], 
    [("spin", [0, 1]), ("species", ["B", "N"]), ("orbitals", ["1s", "2s", "1p", "2p"])]
)

@split_params
def test_split_empty_query(on, expected_keys):

    field = OrbitalQueries()

    geom = sisl.geom.graphene(atoms=[
        sisl.Atom("B", orbitals=["1s", "2s"]),
        sisl.Atom("N", orbitals=["1p", "2p"])
    ])
    # Provide geometry and spin so that the options of the field can be updated.
    field.update_options(geom, "polarized")

    # Create a splitted query
    queries = field._split_query({}, on=on, name=f"${on}")

    # Check that the splitting has been done correctly
    assert len(queries) == len(expected_keys)
    for query, key in zip(queries, expected_keys):
        assert query[on] == [key]
        assert query['name'] == str(key)

@split_params
def test_split_query(on, expected_keys):

    field = OrbitalQueries()

    geom = sisl.geom.graphene(atoms=[
        sisl.Atom("B", orbitals=["1s", "2s"]),
        sisl.Atom("N", orbitals=["1p", "2p"])
    ])
    # Provide geometry and spin so that the options of the field can be updated.
    field.update_options(geom, "polarized")

    # Split a query where we require the first spin component
    queries = field._split_query({"spin": [0]}, on=on, name=f"${on}")

    if on == "spin":
        expected_keys = [0]

    # Check that the splitting has been done correctly
    assert len(queries) == len(expected_keys)
    for query, key in zip(queries, expected_keys):
        assert query[on] == [key]
        assert query['name'] == str(key)
        # Check that the spin requirement is maintained
        assert query['spin'] == [0]
        
    


