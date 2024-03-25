# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import itertools

import numpy as np
import pytest

import sisl
from sisl.viz.plotters.matrix import draw_matrix_separators, set_matrix_axes


def test_draw_matrix_separators_empty():
    C = sisl.Atom(
        "C",
        orbitals=[
            sisl.AtomicOrbital("2s"),
            sisl.AtomicOrbital("2px"),
            sisl.AtomicOrbital("2py"),
            sisl.AtomicOrbital("2pz"),
        ],
    )
    geom = sisl.geom.graphene(atoms=C)

    # Check combinations that should give no lines
    assert draw_matrix_separators(False, geom, "orbitals", "orbitals") == []
    assert draw_matrix_separators(True, geom, "atoms", "orbitals") == []


@pytest.mark.parametrize(
    "draw_supercells,separator_mode",
    itertools.product([True, False], ["atoms", "orbitals", "supercells"]),
)
def test_draw_matrix_separators(draw_supercells, separator_mode):
    C = sisl.Atom(
        "C",
        orbitals=[
            sisl.AtomicOrbital("2s"),
            sisl.AtomicOrbital("2px"),
            sisl.AtomicOrbital("2py"),
            sisl.AtomicOrbital("2pz"),
        ],
    )
    geom = sisl.geom.graphene(atoms=C)

    lines = draw_matrix_separators(
        {"color": "red"},
        geom,
        "orbitals",
        separator_mode=separator_mode,
        draw_supercells=draw_supercells,
    )

    if not draw_supercells and separator_mode == "supercells":
        assert len(lines) == 0
        return

    assert len(lines) == 1
    assert isinstance(lines[0], dict)
    action = lines[0]
    assert action["method"] == "draw_line"
    # Check that the number of points in the line is fine
    n_expected_points = {
        ("atoms", False): 6,
        ("atoms", True): 30,
        ("orbitals", False): 12,
        ("orbitals", True): 60,
        ("supercells", True): 24,
    }[separator_mode, draw_supercells]

    assert action["kwargs"]["x"].shape == (n_expected_points,)
    assert action["kwargs"]["y"].shape == (n_expected_points,)

    assert action["kwargs"]["line"]["color"] == "red"


def test_set_matrix_axes():

    C = sisl.Atom(
        "C",
        orbitals=[
            sisl.AtomicOrbital("2s"),
            sisl.AtomicOrbital("2px"),
            sisl.AtomicOrbital("2py"),
            sisl.AtomicOrbital("2pz"),
        ],
    )
    geom = sisl.geom.graphene(atoms=C)

    matrix = np.zeros((geom.no, geom.no * geom.n_s))

    actions = set_matrix_axes(
        matrix, geom, "orbitals", constrain_axes=False, set_labels=False
    )
    assert len(actions) == 1
    assert actions[0]["method"] == "set_axes_equal"

    # Test without labels
    actions = set_matrix_axes(
        matrix, geom, "orbitals", constrain_axes=True, set_labels=False
    )
    assert len(actions) == 3
    assert actions[0]["method"] == "set_axes_equal"
    assert actions[1]["method"] == "set_axis"
    assert actions[1]["kwargs"]["axis"] == "x"
    assert actions[1]["kwargs"]["range"] == [-0.5, geom.no * geom.n_s - 0.5]
    assert "tickvals" not in actions[1]["kwargs"]
    assert "ticktext" not in actions[1]["kwargs"]

    assert actions[2]["method"] == "set_axis"
    assert actions[2]["kwargs"]["axis"] == "y"
    assert actions[2]["kwargs"]["range"] == [geom.no - 0.5, -0.5]
    assert "tickvals" not in actions[2]["kwargs"]
    assert "ticktext" not in actions[2]["kwargs"]

    # Test with labels
    actions = set_matrix_axes(
        matrix, geom, "orbitals", constrain_axes=True, set_labels=True
    )
    assert len(actions) == 3
    assert actions[0]["method"] == "set_axes_equal"
    assert actions[1]["method"] == "set_axis"
    assert actions[1]["kwargs"]["axis"] == "x"
    assert actions[1]["kwargs"]["range"] == [-0.5, geom.no * geom.n_s - 0.5]
    assert np.all(actions[1]["kwargs"]["tickvals"] == np.arange(geom.no * geom.n_s))
    assert len(actions[1]["kwargs"]["ticktext"]) == geom.no * geom.n_s

    assert actions[2]["method"] == "set_axis"
    assert actions[2]["kwargs"]["axis"] == "y"
    assert actions[2]["kwargs"]["range"] == [geom.no - 0.5, -0.5]
    assert np.all(actions[2]["kwargs"]["tickvals"] == np.arange(geom.no))
    assert len(actions[2]["kwargs"]["ticktext"]) == geom.no
