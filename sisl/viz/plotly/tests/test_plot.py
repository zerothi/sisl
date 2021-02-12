"""

This file tests general Plot behavior.

"""
from copy import deepcopy
import os

import pytest
import numpy as np

import sisl
from sisl.viz.plotly.plot import Plot, MultiplePlot, SubPlots, Animation
from sisl.viz.plotly.plots import *
from sisl.viz.plotly.plotutils import get_plotable_siles, load
from sisl.viz.plotly._presets import PRESETS

# ------------------------------------------------------------
# Checks that will be available to be used on any plot class
# ------------------------------------------------------------

pytestmark = [pytest.mark.viz, pytest.mark.plotly]


class BasePlotTester:

    PlotClass = Plot

    def test_documentation(self):

        doc = self.PlotClass.__doc__

        # Check that it has documentation
        assert doc is not None, f'{self.PlotClass.__name__} does not have documentation'

        # Check that all params are in the documentation
        params = [param.key for param in self.PlotClass._get_class_params()[0]]
        missing_params = list(filter(lambda key: key not in doc, params))
        assert len(missing_params) == 0, f"The following parameters are missing in the documentation of {self.PlotClass.__name__}: {missing_params}"

        missing_help = list(map(lambda p: p.key,
                                filter(lambda p: not getattr(p, "help", None), self.PlotClass._parameters)
        ))
        assert len(missing_help) == 0, f"Parameters {missing_help} in {self.PlotClass.__name__} are missing a help message. Don't be lazy!"

    def test_plot_settings(self):

        plot = self.PlotClass()
        # Check that all the parameters have been passed to the settings
        assert np.all([param.key in plot.settings for param in self.PlotClass._parameters])
        # Build some test settings
        new_settings = {'root_fdf': 'Test'}
        # Update settings and check they have been succesfully updated
        old_settings = deepcopy(plot.settings)
        plot.update_settings(**new_settings, run_updates=False)
        assert np.all([plot.settings[key] == val for key, val in new_settings.items()])
        # Undo settings and check if they go back to the previous ones
        plot.undo_settings(run_updates=False)

        assert np.all([plot.settings[key] ==
                    val for key, val in old_settings.items()])

        # Build a plot directly with test settings and check if it works
        plot = self.PlotClass(**new_settings)
        assert np.all([plot.settings[key] == val for key, val in new_settings.items()])

    def test_plot_shortcuts(self):

        plot = self.PlotClass()
        # Build a fake shortcut and test it.
        def dumb_shortcut(a=2):
            plot.a_value = a
        # Add it without extra parameters
        plot.add_shortcut("ctrl+a", "Dumb shortcut", dumb_shortcut)
        # Call it without extra parameters
        plot.call_shortcut("ctrl+a")
        assert plot.a_value == 2
        # Call it with extra parameters
        plot.call_shortcut("ctrl+a", a=5)
        assert plot.a_value == 5
        # Add the shortcut directly with extra parameters
        plot.add_shortcut("ctrl+alt+a", "Dumb shortcut 2", dumb_shortcut, a=8)
        # And test that it works
        plot.call_shortcut("ctrl+alt+a")
        assert plot.a_value == 8

    def test_presets(self):

        plot = self.PlotClass(presets="dark")

        assert np.all([key not in plot.settings or plot.settings[key] == val for key, val in PRESETS["dark"].items()])

    def test_save_and_load(self, obj=None):

        file_name = "./__sislsaving_test"

        if obj is None:
            obj = self.PlotClass()

        obj.save(file_name)

        try:
            plot = load(file_name)
        except Exception as e:
            os.remove(file_name)
            raise e

        os.remove(file_name)

# ------------------------------------------------------------
#          Actual tests on the Plot parent class
# ------------------------------------------------------------


class TestPlot(BasePlotTester):

    PlotClass = Plot

    # def test_plotable_siles(self):
    #     """
    #     Checks that all the siles that are registered as plotables get the corresponding plot.
    #     """

    #     for sile_rule in get_plotable_siles(rules=True):

    #         file_name = f"file.{sile_rule.suffix}"

    #         plot = Plot(file_name)


# ------------------------------------------------------------
#            Tests for the MultiplePlot class
# ------------------------------------------------------------

class TestMultiplePlot(BasePlotTester):

    PlotClass = MultiplePlot

    def test_init_from_kw(self):

        kw = MultiplePlot._kw_from_cls(self.PlotClass)

        geom = sisl.geom.graphene()

        multiple_plot = geom.plot(show_cell=["box", False, False], axes=[0, 1], **{kw: "show_cell"})

        assert isinstance(multiple_plot, self.PlotClass), f"{self.PlotClass} was not correctly initialized using the {kw} keyword argument"
        assert len(multiple_plot.child_plots) == 3, "Child plots were not properly generated"

    def test_object_sharing(self):

        kw = MultiplePlot._kw_from_cls(self.PlotClass)

        geom = sisl.geom.graphene()

        multiple_plot = geom.plot(show_cell=["box", False, False], axes=[0, 1], **{kw: "show_cell"})
        geoms_ids = [id(plot.geometry) for plot in multiple_plot]
        assert len(set(geoms_ids)) == 1, f"{self.PlotClass} is not properly sharing objects"

        multiple_plot = GeometryPlot(geometry=[sisl.geom.graphene(bond=bond) for bond in (1.2, 1.6)], axes=[0, 1], **{kw: "geometry"})
        geoms_ids = [id(plot.geometry) for plot in multiple_plot]
        assert len(set(geoms_ids)) > 1, f"{self.PlotClass} is sharing objects that should not be shared"

    def test_update_settings(self):

        kw = MultiplePlot._kw_from_cls(self.PlotClass)

        geom = sisl.geom.graphene()

        multiple_plot = geom.plot(show_cell=["box", False, False], axes=[0, 1], **{kw: "show_cell"})
        assert len(multiple_plot.child_plots) == 3

        prev_data_lens = [len(plot.data) for plot in multiple_plot]
        assert prev_data_lens[0] > prev_data_lens[1]

        multiple_plot.update_child_settings(show_cell="box", childs_sel=[1])
        data_lens = [len(plot.data) for plot in multiple_plot]
        assert data_lens[0] == data_lens[1]
        assert data_lens[1] > data_lens[2]

# ------------------------------------------------------------
#            Tests for the SubPlots class
# ------------------------------------------------------------


class TestSubPlots(TestMultiplePlot):

    PlotClass = SubPlots

    def test_subplots_arrangement(self):

        geom = sisl.geom.graphene()

        # We are going to try some things here and check that they don't fail
        # as we have no way of checking the actual layout of the subplots
        plot = GeometryPlot.subplots('show_bonds', [True, False],
            fixed={'geometry': geom, 'axes': [0, 1]}, _debug=True)

        plot.update_settings(cols=2)

        plot.update_settings(rows=2)

        plot.update_settings(cols=1, rows=1)

        plot.update_settings(cols=None, rows=None, arrange='square')

# ------------------------------------------------------------
#              Tests for the Animation class
# ------------------------------------------------------------


class TestAnimation(TestMultiplePlot):

    PlotClass = Animation
