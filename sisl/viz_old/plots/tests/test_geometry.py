# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

Tests specific functionality of the bands plot.

Different inputs are tested (siesta .bands and sisl Hamiltonian).

"""
from sisl.viz.plots.geometry import GeometryPlot
from sisl.messages import SislWarning
import numpy as np

import pytest

import sisl
from sisl.viz.plots.tests.conftest import _TestPlot


pytestmark = [pytest.mark.viz, pytest.mark.plotly]


def test_cross_product():
    cell = np.eye(3) * 2
    z_dir = np.array([0, 0, 1])

    products = [
        ["x", "y", z_dir], ["-x", "y", -z_dir], ["-x", "-y", z_dir],
        ["b", "c", cell[0]], ["c", "b", -cell[0]],
        np.eye(3)
    ]

    for v1, v2, result in products:
        assert np.all(GeometryPlot._cross_product(v1, v2, cell) == result)


class TestGeometry(_TestPlot):

    @pytest.fixture(scope="class", params=["sisl_geom", "ghost_atoms"])
    def init_func_and_attrs(self, request):
        name = request.param

        if name == "sisl_geom":
            init_func = sisl.geom.graphene(orthogonal=True).plot
        elif name == "ghost_atoms":
            init_func = sisl.Geometry([[0, 0, 1], [1, 0, 0]], atoms=[sisl.Atom(6), sisl.Atom(-6)]).plot

        attrs = {}

        return init_func, attrs

    @pytest.fixture(scope="class", params=[None, *sisl.viz.GeometryPlot.get_class_param("backend").options])
    def backend(self, request):
        return request.param

    @pytest.fixture(params=[1, 2, 3])
    def ndim(self, request, backend):
        if backend == "matplotlib" and request.param == 3:
            pytest.skip("Matplotlib 3D representations are not available yet")
        return request.param

    @pytest.fixture(params=["cartesian", "lattice", "explicit"])
    def axes(self, request, ndim):
        if request.param == "cartesian":
            return {1: "x", 2: "x-y", 3: "xyz"}[ndim]
        elif request.param == "lattice":
            # We don't test the 3D case because it doesn't work
            if ndim == 3:
                pytest.skip("3D view doesn't support fractional coordinates")
            return {1: "a", 2: "a-b"}[ndim]
        elif request.param == "explicit":
            if ndim == 3:
                pytest.skip("3D view doesn't support explicit directions")
            return {
                1: [[1, 1, 0]],
                2: [[1, 1, 0], [0, 1, 1]],
            }[ndim]

    @pytest.fixture(params=["Unit cell", "supercell"])
    def nsc(self, request):
        return {"Unit cell": [1, 1, 1], "supercell": [2, 1, 1]}[request.param]

    def _check_all_atomic_props_shape(self, backend_info, na, nsc_val):
        na_sc = na*nsc_val[0]*nsc_val[1]*nsc_val[2]

        for key, value in backend_info["atoms_props"].items():
            if not isinstance(value, np.ndarray):
                continue

            assert value.shape[0] == na_sc, f"'{key}' doesn't have the appropiate shape"

            if key == "xy":
                assert value.shape[1] == 2
            elif key == "xyz":
                assert value.shape[1] == 3

    @pytest.mark.parametrize("atoms, na", [([], 0), (0, 1), (None, "na")])
    def test_atoms(self, plot, axes, nsc, atoms, na):
        plot.update_settings(axes=axes, nsc=nsc, show_bonds=False, show_cell=False, atoms=atoms)

        if na == "na":
            na = plot.geometry.na

        backend_info = plot._for_backend
        self._check_all_atomic_props_shape(backend_info, na, nsc)

    @pytest.mark.parametrize("show_bonds", [False, True])
    def test_toggle_bonds(self, plot, axes, ndim, nsc, show_bonds, test_attrs):
        plot.update_settings(axes=axes, nsc=nsc, show_bonds=show_bonds, bind_bonds_to_ats=True, show_cell=False, atoms=[])
        assert len(plot._for_backend["bonds_props"]) == 0

        plot.update_settings(bind_bonds_to_ats=False)

        backend_info = plot._for_backend
        bonds_props = backend_info["bonds_props"]
        if not test_attrs.get("no_bonds", False):
            n_bonds = len(bonds_props)
            if show_bonds and ndim > 1:
                assert n_bonds > 0
                if ndim == 2:
                    assert bonds_props[0]["xys"].shape == (2, 2)
                elif ndim == 3:
                    assert bonds_props[0]["xyz1"].shape == (3,)
                    assert bonds_props[0]["xyz2"].shape == (3,)
            else:
                assert n_bonds == 0

    @pytest.mark.parametrize("show_cell", [False, "box", "axes"])
    def test_cell(self, plot, axes, show_cell):
        plot.update_settings(axes=axes, show_cell=show_cell)

        assert plot._for_backend["show_cell"] == show_cell

    @pytest.mark.parametrize("show_cell", [False, "box", "axes"])
    def test_cell_styles(self, plot, axes, show_cell):
        cell_style = {"color": "red", "width": 2, "opacity": 0.6}
        plot.update_settings(axes=axes, show_cell=show_cell, cell_style=cell_style)

        assert plot._for_backend["cell_style"] == cell_style

    def test_atoms_sorted_2d(self, plot):
        plot.update_settings(atoms=None, axes="yz", nsc=[1, 1, 1])

        # Check that atoms are sorted along x
        assert np.allclose(plot.geometry.xyz[:, 1:][plot.geometry.xyz[:, 0].argsort()], plot._for_backend["atoms_props"]["xy"])

    def test_atoms_style(self, plot, axes, ndim, nsc):
        plot.update_settings(atoms=None, axes=axes, nsc=nsc)

        rand_values = np.random.random(plot.geometry.na)
        atoms_style = {"color": rand_values, "size": rand_values, "opacity": rand_values}

        new_atoms_style = {"atoms": 0, "color": 2, "size": 2, "opacity": 0.3}

        if ndim == 2:
            depth_vector = plot._cross_product(*plot.get_setting("axes"), plot.geometry.cell)
            sorted_atoms = np.concatenate(plot.geometry.sort(vector=depth_vector, ret_atoms=True)[1])
        else:
            sorted_atoms = plot.geometry._sanitize_atoms(None)

        # Try both passing a dictionary and a list with one dictionary
        for i, atoms_style_val in enumerate((atoms_style, [atoms_style], [atoms_style, new_atoms_style])):
            plot.update_settings(atoms_style=atoms_style_val)

            backend_info = plot._for_backend
            self._check_all_atomic_props_shape(backend_info, plot.geometry.na, nsc)

            if i != 2:
                for key in atoms_style:
                    if not (ndim == 3 and key == "color"):
                        assert np.allclose(
                            backend_info["atoms_props"][key].astype(float),
                            np.tile(atoms_style[key][sorted_atoms], nsc[0]*nsc[1]*nsc[2])
                        )
            else:
                for key in atoms_style:
                    if not (ndim == 3 and key == "color"):
                        assert np.isclose(
                            backend_info["atoms_props"][key].astype(float),
                            np.tile(atoms_style[key][sorted_atoms], nsc[0]*nsc[1]*nsc[2])
                        ).sum() == (plot.geometry.na - 1) * nsc[0]*nsc[1]*nsc[2]

    def test_bonds_style(self, plot, axes, ndim, nsc):
        if ndim == 1:
            return

        bonds_style = {"width": 2, "opacity": 0.6}

        plot.update_settings(atoms=None, axes=axes, nsc=nsc, bonds_style=bonds_style)

        bonds_props = plot._for_backend["bonds_props"]

        assert bonds_props[0]["width"] == 2
        assert bonds_props[0]["opacity"] == 0.6

        plot.update_settings(bonds_style={})

    def test_arrows(self, plot, axes, ndim, nsc):
        # Check that arrows accepts both a dictionary and a list and the data is properly transferred
        for arrows in ({"data": [0, 0, 2]}, [{"data": [0, 0, 2]}]):
            plot.update_settings(axes=axes, arrows=arrows, atoms=None, nsc=nsc, atoms_style=[])
            arrow_data = plot._for_backend["arrows"][0]["data"]
            assert arrow_data.shape == (plot.geometry.na * nsc[0]*nsc[1]*nsc[2], ndim)
            assert not np.isnan(arrow_data).any()

        # Now check that atom selection works
        plot.update_settings(arrows=[{"atoms": 0, "data": [0, 0, 2]}])
        arrow_data = plot._for_backend["arrows"][0]["data"]
        assert arrow_data.shape == (plot.geometry.na * nsc[0]*nsc[1]*nsc[2], ndim)
        assert np.isnan(arrow_data).any()
        assert not np.isnan(arrow_data[0]).any()

        # Check that if atoms is provided, data is only stored for those atoms that are going to be
        # displayed
        plot.update_settings(atoms=0, arrows=[{"atoms": 0, "data": [0, 0, 2]}])
        arrow_data = plot._for_backend["arrows"][0]["data"]
        assert arrow_data.shape == (nsc[0]*nsc[1]*nsc[2], ndim)
        assert not np.isnan(arrow_data).any()

        # Check that if no data is provided for the atoms that are displayed, arrow data is not stored
        # We also check that a warning is being raised because we are providing arrow data for atoms that
        # are not being displayed.
        with pytest.warns(SislWarning):
            plot.update_settings(atoms=1, arrows=[{"atoms": 0, "data": [0, 0, 2]}])
        assert len(plot._for_backend["arrows"]) == 0

        # Finally, check that multiple arrows are passed to the backend
        plot.update_settings(atoms=None, arrows=[{"data": [0, 0, 2]}, {"data": [1, 0, 0]}])
        assert len(plot._for_backend["arrows"]) == 2
