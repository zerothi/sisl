"""

Tests specific functionality of the bands plot.

Different inputs are tested (siesta .bands and sisl Hamiltonian).

"""

import numpy as np
import plotly.graph_objs as go

import sisl
from sisl.viz import GeometryPlot
from sisl.viz.plotly.plots.tests.get_files import from_files
from sisl.viz.plotly.plots.tests.helpers import PlotTester


class GeometryPlotTester(PlotTester):

    def test_1d(self):
        # Remains untested for now as there is no clear behavior
        pass

    def test_2d(self):

        plot = self.plot

        plot.update_settings(axes=[0, 1], bonds=True, cell='box', atoms=None)

        # Check that the first trace is 2d
        assert np.all([hasattr(plot.data[0], ax) for ax in ('x', 'y')])
        assert not hasattr(plot.data[0], 'z')

        # Check that there is a cell and we can toggle it
        with_box_cell = len(plot.data)

        plot.update_settings(cell=False)
        assert len(plot.data) == with_box_cell - 1

        plot.update_settings(cell='axes')
        assert len(plot.data) == with_box_cell + 2

        # Check that we can toggle the bonds
        with_bonds = len(plot.data)

        plot.update_settings(bonds=False)
        assert len(plot.data) == with_bonds - 1

        # Check that we can ask for specific atoms
        plot.update_settings(atoms=[0], bonds=False, cell=False)
        assert len(plot.data) == 1

        # Check that we can toggle bonds being bound to atoms
        if plot.geometry.na > 2:
            plot.update_settings(atoms=[0], bonds=True, bind_bonds_to_ats=True)
            #First trace is the bonds
            prev_len = len(plot.data[0].x)

            plot.update_settings(atoms=[0], bonds=True, bind_bonds_to_ats=False)
            assert len(plot.data[0].x) > prev_len

    def test_3d(self):

        plot = self.plot

        plot.update_settings(axes=[0, 1, 2], cell='box', bonds=True, atoms=None)

        # Check that the first trace is 3d
        assert np.all([hasattr(plot.data[0], ax) for ax in ('x', 'y', 'z')])

        # Check that there is a cell and we can toggle it
        with_box_cell = len(plot.data)

        plot.update_settings(cell=False)
        assert len(plot.data) == with_box_cell - 1

        plot.update_settings(cell='axes')
        assert len(plot.data) == with_box_cell + 2

        # Check that we can toggle the bonds
        with_bonds = len(plot.data)

        plot.update_settings(bonds=False)
        assert len(plot.data) < with_bonds

        # Check that we can ask for specific atoms
        plot.update_settings(atoms=[0], bonds=False, cell=False)
        assert len(plot.data) == 1

        # Check that we can toggle bonds being bound to atoms
        if plot.geometry.na > 2:
            plot.update_settings(atoms=[0], bonds=True, bind_bonds_to_ats=True)
            #First trace is the bonds
            prev_len = len(plot.data[0].x)

            plot.update_settings(atoms=[0], bonds=True, bind_bonds_to_ats=False)
            assert len(plot.data[0].x) > prev_len


class TestGeometryPlot(GeometryPlotTester):

    run_for = {
        "sisl_geom": {"init_func": sisl.geom.graphene(orthogonal=True).plot.bind()},
        "ghost_atoms": {"init_func": sisl.Geometry([[0, 0, 1], [1, 0, 0]], atoms=[sisl.Atom(6), sisl.Atom(-6)]).plot.bind()}
    }
