"""

These tests check that all plot subclasses fulfill at least the most basic stuff

More tests should be run on each plot, but these are the most basic ones to
ensure that at least they do not break basic plot functionality.
"""

from sisl.viz.plotly.tests.test_plot import BasePlotTester
from sisl.viz.plotly.plots import *
from sisl.viz.plotly.plotutils import get_plot_classes

# ------------------------------------------------------------
# Factory that returns a basic functionality test for a class
# ------------------------------------------------------------


def get_basic_functionality_test(PlotSubClass):

    class BasicSubClassTest(BasePlotTester):

        PlotClass = PlotSubClass

        def test_compulsory_methods(self):

            assert hasattr(self.PlotClass, "_set_data")
            assert callable(self.PlotClass._set_data)

            assert hasattr(self.PlotClass, "_plot_type")
            assert isinstance(self.PlotClass._plot_type, str)

        def test_param_groups(self):

            plot = self.PlotClass()

            for group in plot.param_groups:
                for key in ("key", "name", "icon", "description"):
                    assert key in group, f'{self.PlotClass} is missing {key} in parameters group {group}'

    return BasicSubClassTest

# ------------------------------------------------------------
#               Let's test all plot subclasses
# ------------------------------------------------------------

for PlotSubClass in get_plot_classes():

    globals()[f"Test{PlotSubClass.__name__}"] = get_basic_functionality_test(PlotSubClass)
