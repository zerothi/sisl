# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

This file tests general Plot behavior.

"""
from copy import deepcopy
import os
from sisl.messages import SislInfo, SislWarning

import pytest
import numpy as np

import warnings

import sisl
from sisl.viz.plot import Plot, MultiplePlot, SubPlots, Animation
from sisl.viz.plots import *
from sisl.viz.plotutils import load
from sisl.viz._presets import PRESETS

try:
    import dill
    skip_dill = pytest.mark.skipif(False, reason="dill not available")
except ImportError:
    skip_dill = pytest.mark.skipif(True, reason="dill not available")

# ------------------------------------------------------------
# Checks that will be available to be used on any plot class
# ------------------------------------------------------------

pytestmark = [pytest.mark.viz, pytest.mark.plotly]


class _TestPlotClass:

    _cls = Plot

    def _init_plot_without_warnings(self, *args, **kwargs):

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self._cls(*args, **kwargs)

    def test_documentation(self):

        doc = self._cls.__doc__

        # Check that it has documentation
        assert doc is not None, f'{self._cls.__name__} does not have documentation'

        # Check that all params are in the documentation
        params = [param.key for param in self._cls._get_class_params()[0]]
        missing_params = list(filter(lambda key: key not in doc, params))
        assert len(missing_params) == 0, f"The following parameters are missing in the documentation of {self._cls.__name__}: {missing_params}"

        missing_help = list(map(lambda p: p.key,
                                filter(lambda p: not getattr(p, "help", None), self._cls._parameters)
        ))
        assert len(missing_help) == 0, f"Parameters {missing_help} in {self._cls.__name__} are missing a help message. Don't be lazy!"

    def test_plot_settings(self):

        plot = self._init_plot_without_warnings()
        # Check that all the parameters have been passed to the settings
        assert np.all([param.key in plot.settings for param in self._cls._parameters])
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
        plot = self._init_plot_without_warnings(**new_settings)
        assert np.all([plot.settings[key] == val for key, val in new_settings.items()])

    def test_plot_shortcuts(self):

        plot = self._init_plot_without_warnings()
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

        plot = self._init_plot_without_warnings(presets="dark")

        assert np.all([key not in plot.settings or plot.settings[key] == val for key, val in PRESETS["dark"].items()])

    @skip_dill
    def test_save_and_load(self, obj=None):

        file_name = "./__sislsaving_test"

        if obj is None:
            obj = self._init_plot_without_warnings()

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


class TestPlot(_TestPlotClass):

    _cls = Plot


# ------------------------------------------------------------
#            Tests for the MultiplePlot class
# ------------------------------------------------------------

class TestMultiplePlot(_TestPlotClass):

    _cls = MultiplePlot

    def test_init_from_kw(self):

        kw = MultiplePlot._kw_from_cls(self._cls)

        geom = sisl.geom.graphene()

        multiple_plot = geom.plot(show_cell=["box", False, False], backend=None, axes=[0, 1], **{kw: "show_cell"})

        assert isinstance(multiple_plot, self._cls), f"{self._cls} was not correctly initialized using the {kw} keyword argument"
        assert len(multiple_plot.children) == 3, "Child plots were not properly generated"

    def test_object_sharing(self):

        kw = MultiplePlot._kw_from_cls(self._cls)

        geom = sisl.geom.graphene()

        multiple_plot = geom.plot(show_cell=["box", False, False], backend=None, axes=[0, 1], **{kw: "show_cell"})
        geoms_ids = [id(plot.geometry) for plot in multiple_plot]
        assert len(set(geoms_ids)) == 1, f"{self._cls} is not properly sharing objects"

        multiple_plot = GeometryPlot(geometry=[sisl.geom.graphene(bond=bond) for bond in (1.2, 1.6)], backend=None, axes=[0, 1], **{kw: "geometry"})
        geoms_ids = [id(plot.geometry) for plot in multiple_plot]
        assert len(set(geoms_ids)) > 1, f"{self._cls} is sharing objects that should not be shared"

    def test_update_settings(self):

        kw = MultiplePlot._kw_from_cls(self._cls)

        geom = sisl.geom.graphene()
        show_cell = ["box", False, False]

        multiple_plot = geom.plot(show_cell=show_cell, backend=None, axes=[0, 1], **{kw: "show_cell"})
        assert len(multiple_plot.children) == 3

        for i, show_cell_val in enumerate(show_cell):
            assert multiple_plot[i]._for_backend["show_cell"] == show_cell_val

        multiple_plot.update_children_settings(show_cell="box", children_sel=[1])
        for i, show_cell_val in enumerate(show_cell):
            if i == 1:
                show_cell_val = "box"
            assert multiple_plot[i]._for_backend["show_cell"] == show_cell_val

# ------------------------------------------------------------
#            Tests for the SubPlots class
# ------------------------------------------------------------


class TestSubPlots(TestMultiplePlot):

    _cls = SubPlots

    def test_subplots_arrangement(self):

        geom = sisl.geom.graphene()

        # We are going to try some things here and check that they don't fail
        # as we have no way of checking the actual layout of the subplots
        plot = GeometryPlot.subplots('show_bonds', [True, False], backend=None,
            fixed={'geometry': geom, 'axes': [0, 1], "backend": None}, _debug=True)

        plot.update_settings(cols=2)

        plot.update_settings(rows=2)

        # This should issue a warning stating that one plot will be missing
        with pytest.warns(SislWarning):
            plot.update_settings(cols=1, rows=1)

        plot.update_settings(cols=None, rows=None, arrange='square')

# ------------------------------------------------------------
#              Tests for the Animation class
# ------------------------------------------------------------


class _TestAnimation(TestMultiplePlot):

    PlotClass = Animation


def test_calling_Plot():
    # Just check that it doesn't raise any error
    with pytest.warns(SislInfo):
        plot = Plot("nonexistent.LDOS")

    assert isinstance(plot, GridPlot)
