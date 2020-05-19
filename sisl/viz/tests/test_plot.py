'''

This file tests general Plot behavior.

'''

from copy import deepcopy
import os
import tempfile

import pytest
import numpy as np

from sisl.viz.plot import Plot, MultiplePlot, Animation

from sisl.viz.plots import *
from sisl.viz.plotutils import get_plotable_siles, load
from sisl.viz._presets import PRESETS

# ------------------------------------------------------------
# Checks that will be available to be used on any plot class
# ------------------------------------------------------------
class BasePlotTester:

    PlotClass = Plot

    def test_documentation(self):

        doc = self.PlotClass.__doc__

        # Check that it has documentation
        assert doc is not None, f'{self.PlotClass.__name__} does not have documentation'

        # Check that all params are in the documentation
        params = np.array([param.key for param in self.PlotClass._get_class_params()[0]])
        is_indoc = np.array([key in doc for key in params])
        assert np.all(is_indoc), f'The following parameters are missing in the documentation of {self.PlotClass.__name__}: {params[~is_indoc]} '

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
        print(old_settings)
        print(plot.settings)
        assert np.all([plot.settings[key] ==
                    val for key, val in old_settings.items()])
        
        # Build a plot directly with test settings and check if it works
        plot = self.PlotClass(**new_settings)
        assert np.all([plot.settings[key] == val for key, val in new_settings.items()])

    def test_plot_connected(self):

        plot = self.PlotClass()

        # Check that the plot has a socketio attribute and that it can be changed
        # Seems dumb, but socketio is really a property that uses functions to set the
        # real attribute, so they may be broken by something
        assert hasattr(plot, 'socketio'), f"Socketio connectivity is not initialized correctly in {plot.__class__}"
        assert plot.socketio is None
        plot.socketio = 2
        assert plot.socketio == 2, f'Problems setting a new socketio for {plot.__class__}'

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

        assert np.all([ key not in plot.settings or plot.settings[key] == val for key, val in PRESETS["dark"].items()])

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
    #     '''
    #     Checks that all the siles that are registered as plotables get the corresponding plot.
    #     '''

    #     for sile_rule in get_plotable_siles(rules=True):

    #         file_name = f"file.{sile_rule.suffix}"

    #         plot = Plot(file_name)

    
# ------------------------------------------------------------
#            Tests for the MultiplePlot class
# ------------------------------------------------------------

class TestMultiplePlot(BasePlotTester):

    PlotClass = MultiplePlot

# ------------------------------------------------------------
#              Tests for the Animation class
# ------------------------------------------------------------

class TestAnimation(BasePlotTester):

    PlotClass = Animation

    

