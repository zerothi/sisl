# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

Tests specific functionality of the bands plot.

Different inputs are tested (siesta .bands and sisl Hamiltonian).

"""
from sisl.messages import SislWarning
import numpy as np

import pytest

import sisl
from sisl.viz.plots.tests.conftest import _TestPlot


pytestmark = [pytest.mark.viz, pytest.mark.plotly]


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

    @pytest.fixture(params=[1, 2, 3])
    def ndim(self, request):
        return request.param

    @pytest.fixture(params=["cartesian", "lattice"])
    def axes(self, request, ndim):
        if request.param == "cartesian":
            return {1: "x", 2: "xy", 3: "xyz"}[ndim]
        elif request.param == "lattice":
            # We don't test the 3D case because it doesn't work
            return {1: "a", 2: "ab", 3: "xyz"}[ndim]

    @pytest.fixture(scope="function", params=["xy", "ab", [[1, 1, 0], [1, -1, 0]], ["x", [1, -1, 0]]])
    def axes_2D(self, request):
        """Fixture returning all valid combinations of axes in 2D"""
        return request.param

    def test_1d(self):
        # Remains untested for now as there is no clear behavior
        pass

    def test_2d(self, plot, axes_2D):

        plot.update_settings(axes=axes_2D, show_bonds=True, show_cell='box', atoms=None)

        # Check that the first trace is 2d
        assert np.all([hasattr(plot.data[0], ax) for ax in ('x', 'y')])
        assert not hasattr(plot.data[0], 'z')

        # Check that there is a cell and we can toggle it
        with_box_cell = len(plot.data)

        plot.update_settings(show_cell=False)
        assert len(plot.data) == with_box_cell - 1

        plot.update_settings(show_cell='axes')
        assert len(plot.data) == with_box_cell + 2

        # Check that we can toggle the bonds
        with_bonds = len(plot.data)

        plot.update_settings(show_bonds=False)
        assert len(plot.data) == with_bonds - 1

        # Check that we can ask for specific atoms
        plot.update_settings(atoms=[0], show_bonds=False, show_cell=False)
        assert len(plot.data) == 1

        # Check that we can toggle bonds being bound to atoms
        if plot.geometry.na > 2:
            plot.update_settings(atoms=[0], show_bonds=True, bind_bonds_to_ats=True)
            #First trace is the bonds
            prev_len = len(plot.data[0].x)

            plot.update_settings(atoms=[0], show_bonds=True, bind_bonds_to_ats=False)
            assert len(plot.data[0].x) > prev_len

    def test_3d(self, plot):

        plot.update_settings(axes=[0, 1, 2], show_cell='box', show_bonds=True, atoms=None)

        # Check that the first trace is 3d
        assert np.all([hasattr(plot.data[0], ax) for ax in ('x', 'y', 'z')])

        # Check that there is a cell and we can toggle it
        with_box_cell = len(plot.data)

        plot.update_settings(show_cell=False)
        assert len(plot.data) == with_box_cell - 1

        plot.update_settings(show_cell='axes')
        assert len(plot.data) == with_box_cell + 2

        # Check that we can toggle the bonds
        with_bonds = len(plot.data)

        plot.update_settings(show_bonds=False)
        assert len(plot.data) < with_bonds

        # Check that we can ask for specific atoms
        plot.update_settings(atoms=[0], show_bonds=False, show_cell=False)
        assert len(plot.data) == 1

        # Check that we can toggle bonds being bound to atoms
        if plot.geometry.na > 2:
            plot.update_settings(atoms=[0], show_bonds=True, bind_bonds_to_ats=True)
            #First trace is the bonds
            prev_len = len(plot.data[0].x)

            plot.update_settings(atoms=[0], show_bonds=True, bind_bonds_to_ats=False)
            assert len(plot.data[0].x) > prev_len

    def test_nsc(self, plot):
        plot.update_settings(axes=[0, 1], atoms=None)

        na = plot.geometry.na

        atom_traces = [trace for trace in plot.data if trace.name == "Atoms"]
        assert len(atom_traces) == 1
        assert len(atom_traces[0].y) == na

        plot.update_settings(nsc=[2, 2, 1])

        atom_traces = [trace for trace in plot.data if trace.name == "Atoms"]
        assert len(atom_traces[0].y) == na * 4, f"Supercell is not correctly displayed in {plot.__class__.__name__}"

        plot.update_settings(nsc=[1, 1, 1])

    def test_atoms_styles(self, plot):
        # Check that atoms_style accepts a dictionary and it's properly transferred
        rand_color = np.random.random(plot.geometry.na)
        plot.update_settings(atoms_style={"color": rand_color}, atoms=None, nsc=[1, 1, 1])
        assert np.all(plot._for_backend["atoms_props"]["color"] == rand_color)

        # Same for a list with just one dictionary
        plot.update_settings(atoms_style=[{"color": rand_color}])
        assert np.all(plot._for_backend["atoms_props"]["color"] == rand_color)

        # Now check if adding a new rule for styles overwrites the previous values
        plot.update_settings(atoms_style=[{"color": rand_color}, {"atoms": 0, "color": 2}])
        assert plot._for_backend["atoms_props"]["color"][0] == 2
        assert np.all(plot._for_backend["atoms_props"]["color"][1:] == rand_color[1:])

    def test_atoms_styles_sc(self, plot):
        """
        We need to check that atom styles are handled correctly
        when a supercell is requested.
        """
        geom = plot.geometry

        # Just check that they work, i.e. the arrays have been properly extended.
        # Otherwise an index error would be raised.
        plot.update_settings(atoms_style={"color": np.random.random(geom.na)}, nsc=[2, 1, 1])

        plot.update_settings(atoms_style={"size": np.random.random(geom.na)}, nsc=[2, 1, 1])

    def test_atom_colors_2d(self, plot):

        plot.update_settings(axes="xy", atoms=None, atoms_style=[], nsc=[1,1,1])

        geom = plot.geometry

        atom_traces = [trace for trace in plot.data if trace.name == "Atoms"]
        assert len(atom_traces) == 1

        colors = np.random.random(geom.na)

        plot.update_settings(atoms_style={"color": colors}, atoms_colorscale="RdBu")
        rdbu_atom_traces = [trace for trace in plot.data if trace.name == "Atoms"]

        assert np.all(rdbu_atom_traces[0].marker.color.astype(float) == colors)

        plot.update_settings(atoms_colorscale="viridis")
        viridis_atom_traces = [trace for trace in plot.data if trace.name == "Atoms"]

        assert rdbu_atom_traces[0].marker.colorscale[0][1] != viridis_atom_traces[0].marker.colorscale[0][1]

    def test_atom_colors_3d(self, plot):

        geom = plot.geometry

        plot.update_settings(axes="xyz", atoms=None, atoms_style=[])

        atom_traces = [trace for trace in plot.data if trace["legendgroup"] == "Atoms"]
        assert len(atom_traces) == geom.na

        colors = np.random.random(geom.na)
        plot.update_settings(atoms_style={"color": colors}, atoms_colorscale="viridis")
        viridis_atom_traces = [trace for trace in plot.data if trace["legendgroup"] == "Atoms"]

        assert len(viridis_atom_traces) == len(atom_traces)
        assert np.all([old.color != new.color for old, new in zip(atom_traces, viridis_atom_traces)])

        plot.update_settings(atoms_colorscale="RdBu")
        rdbu_atom_traces = [trace for trace in plot.data if trace["legendgroup"] == "Atoms"]

        assert len(viridis_atom_traces) == len(atom_traces)
        assert np.all([old.color != new.color for old, new in zip(viridis_atom_traces, rdbu_atom_traces)])

    def test_atom_sizes_2d(self, plot):

        geom = plot.geometry

        plot.update_settings(axes=[0, 1], atoms=None, atoms_style=[])

        atom_traces = [trace for trace in plot.data if trace.name == "Atoms"]
        assert len(atom_traces) == 1

        plot.update_settings(atoms_style={"size":geom.atoms.Z+1})
        sized_atom_traces = [trace for trace in plot.data if trace.name == "Atoms"]

        assert len(atom_traces) == len(sized_atom_traces)
        assert np.all([old.marker.size != new.marker.size for old, new in zip(atom_traces, sized_atom_traces)])

    def test_atom_sizes_3d(self, plot):

        geom = plot.geometry

        plot.update_settings(axes=[0, 1, 2], atoms=None, atoms_style=[])

        atom_traces = [trace for trace in plot.data if trace["legendgroup"] == "Atoms"]
        assert len(atom_traces) == geom.na

        plot.update_settings(atoms_style={"size":geom.atoms.Z+1})
        sized_atom_traces = [trace for trace in plot.data if trace["legendgroup"] == "Atoms"]

        assert len(atom_traces) == len(sized_atom_traces)
        assert np.all([np.any(old.x != new.x) for old, new in zip(atom_traces, sized_atom_traces)])
    
    def test_cell_styles(self, plot):
        cell_style = {"color": "red", "width": 2}
        plot.update_settings(cell_style=cell_style)

        assert plot._for_backend["cell_style"] == cell_style

    def test_arrows(self, plot, axes, ndim):
        # Check that arrows accepts both a dictionary and a list and the data is properly transferred
        for arrows in ({"data": [0,0,2]}, [{"data": [0,0,2]}]):
            plot.update_settings(axes=axes, arrows=arrows, atoms=None, nsc=[1, 1, 1])
            arrow_data = plot._for_backend["arrows"][0]["data"]
            assert arrow_data.shape == (plot.geometry.na, ndim)
            assert not np.isnan(arrow_data).any()

        # Now check that atom selection works
        plot.update_settings(arrows=[{"atoms": 0, "data": [0,0,2]}])
        arrow_data = plot._for_backend["arrows"][0]["data"]
        assert arrow_data.shape == (plot.geometry.na, ndim)
        assert np.isnan(arrow_data).any()
        assert not np.isnan(arrow_data[0]).any()

        # Check that if atoms is provided, data is only stored for those atoms that are going to be
        # displayed
        plot.update_settings(atoms=0, arrows=[{"atoms": 0, "data": [0,0,2]}])
        arrow_data = plot._for_backend["arrows"][0]["data"]
        assert arrow_data.shape == (1, ndim)
        assert not np.isnan(arrow_data).any()

        # Check that if no data is provided for the atoms that are displayed, arrow data is not stored
        # We also check that a warning is being raised because we are providing arrow data for atoms that
        # are not being displayed.
        with pytest.warns(SislWarning):
            plot.update_settings(atoms=1, arrows=[{"atoms": 0, "data": [0,0,2]}])
        assert len(plot._for_backend["arrows"]) == 0

        # Finally, check that multiple arrows are passed to the backend
        plot.update_settings(atoms=None, arrows=[{"data": [0,0,2]}, {"data": [1,0,0]}])
        assert len(plot._for_backend["arrows"]) == 2
    
    def test_arrows_sc(self, plot):
        plot.update_settings(atoms=None, arrows={"data": [0,0,2]}, nsc=[2,1,1])

    def test_no_atoms(self, plot, axes):
        plot.update_settings(atoms=[], axes=axes, arrows=[])

        plot.update_settings(atoms=None, show_atoms=False)
