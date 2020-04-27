'''

This file tests general Plot behavior.

'''

from copy import deepcopy

import numpy as np

from sisl.viz.plot import Plot, MultiplePlot, Animation

from sisl.viz.plots import *
from sisl.viz.plotutils import get_plotable_siles

# ------------------------------------------------------------
# Checks that will be available to be used on any plot class
# ------------------------------------------------------------
class BasePlotTester:

    PlotClass = Plot

    def test_plot_settings(self):

        plot = self.PlotClass()
        # Check that all the parameters have been passed to the settings
        assert np.all([param.key in plot.settings for param in self.PlotClass._parameters])
        # Build some test settings
        new_settings = {'title': 'Test', 'root_fdf': 'Another test'}
        # Update settings and check they have been succesfully updated
        old_settings = deepcopy(plot.settings)
        plot.update_settings(**new_settings, update_fig=False)
        assert np.all([plot.settings[key] == val for key, val in new_settings.items()])
        # Undo settings and check if they go back to the previous ones
        plot.undo_settings(update_fig=False)
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
        assert hasattr(plot, 'socketio')
        assert plot.socketio is None
        plot.socketio = 2
        assert plot.socketio == 2

    def test_plot_shortcuts(self):

        plot = self.PlotClass()
        # Build a fake shortcut and test it.
        def dumb_shortcut(self, a=2):
            self.a_value = a
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

# ------------------------------------------------------------
#          Actual tests on the Plot parent class
# ------------------------------------------------------------

class TestPlot(BasePlotTester):

    PlotClass = Plot

    def test_plotable_siles(self):
        '''
        Checks that all the siles that are registered as plotables get the corresponding plot.
        '''

        for sile_rule in get_plotable_siles(rules=True):

            file_name = f"file.{sile_rule.suffix}"
            plot = Plot(file_name)

            corresponding_class = sile_rule.cls._plot[0]
            assert plot.__class__ == corresponding_class

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

    

