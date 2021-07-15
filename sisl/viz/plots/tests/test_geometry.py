# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

Tests specific functionality of the bands plot.

Different inputs are tested (siesta .bands and sisl Hamiltonian).

"""
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

    def test_1d(self):
        # Remains untested for now as there is no clear behavior
        pass

    def test_2d(self, plot):

        plot.update_settings(axes=[0, 1], show_bonds=True, show_cell='box', atoms=None)

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

    def test_atom_colors_2d(self, plot):

        geom = plot.geometry

        plot.update_settings(axes=[0, 1], atoms=None)

        atom_traces = [trace for trace in plot.data if trace.name == "Atoms"]
        assert len(atom_traces) == 1

        colors = np.random.random(geom.na)

        plot.update_settings(atoms_color=colors, atoms_colorscale="RdBu")
        rdbu_atom_traces = [trace for trace in plot.data if trace.name == "Atoms"]

        assert np.all(rdbu_atom_traces[0].marker.color == colors)

        plot.update_settings(atoms_colorscale="viridis")
        viridis_atom_traces = [trace for trace in plot.data if trace.name == "Atoms"]

        assert rdbu_atom_traces[0].marker.colorscale[0][1] != viridis_atom_traces[0].marker.colorscale[0][1]

    def test_atom_colors_3d(self, plot):

        geom = plot.geometry

        plot.update_settings(axes=[0, 1, 2], atoms=None, atoms_color=None)

        atom_traces = [trace for trace in plot.data if trace["legendgroup"] == "Atoms"]
        assert len(atom_traces) == geom.na

        plot.update_settings(atoms_color=np.random.random(geom.na), atoms_colorscale="viridis")
        viridis_atom_traces = [trace for trace in plot.data if trace["legendgroup"] == "Atoms"]

        assert len(viridis_atom_traces) == len(atom_traces)
        assert np.all([old.color != new.color for old, new in zip(atom_traces, viridis_atom_traces)])

        plot.update_settings(atoms_colorscale="RdBu")
        rdbu_atom_traces = [trace for trace in plot.data if trace["legendgroup"] == "Atoms"]

        assert len(viridis_atom_traces) == len(atom_traces)
        assert np.all([old.color != new.color for old, new in zip(viridis_atom_traces, rdbu_atom_traces)])

    def test_atom_sizes_2d(self, plot):

        geom = plot.geometry

        plot.update_settings(axes=[0, 1], atoms=None, atoms_size=None)

        atom_traces = [trace for trace in plot.data if trace.name == "Atoms"]
        assert len(atom_traces) == 1

        plot.update_settings(atoms_size=geom.atoms.Z+1)
        sized_atom_traces = [trace for trace in plot.data if trace.name == "Atoms"]

        assert len(atom_traces) == len(sized_atom_traces)
        assert np.all([old.marker.size != new.marker.size for old, new in zip(atom_traces, sized_atom_traces)])

    def test_atom_sizes_3d(self, plot):

        geom = plot.geometry

        plot.update_settings(axes=[0, 1, 2], atoms=None, atoms_size=None)

        atom_traces = [trace for trace in plot.data if trace["legendgroup"] == "Atoms"]
        assert len(atom_traces) == geom.na

        plot.update_settings(atoms_size=geom.atoms.Z+1)
        sized_atom_traces = [trace for trace in plot.data if trace["legendgroup"] == "Atoms"]

        assert len(atom_traces) == len(sized_atom_traces)
        assert np.all([np.any(old.x != new.x) for old, new in zip(atom_traces, sized_atom_traces)])

    def test_atom_properties_nsc(self, plot):
        """
        We need to check that atoms_color and atoms_size are handled correctly
        when a supercell is requested.
        """
        geom = plot.geometry

        # Just check that they work, i.e. the arrays have been properly extended.
        # Otherwise an index error would be raised.
        plot.update_settings(atoms_color=np.random.random(geom.na), nsc=[2, 1, 1])

        plot.update_settings(atoms_size=geom.atoms.Z+1, atoms_color=None, nsc=[2, 1, 1])    
