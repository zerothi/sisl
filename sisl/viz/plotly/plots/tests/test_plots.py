"""

These tests check that all plot subclasses fulfill at least the most basic stuff

More tests should be run on each plot, but these are the most basic ones to
ensure that at least they do not break basic plot functionality.
"""
import pytest

from sisl.viz.plotly.tests.test_plot import BasePlotTester
from sisl.viz.plotly.plots import *
from sisl.viz.plotly.plotutils import get_plot_classes


pytestmark = [pytest.mark.viz, pytest.mark.plotly]

# Test all plot subclasses with the subclass tester

# The following function basically tells pytest to run TestPlotSubClass
# once for each plot class. It takes care of setting the PlotClass attribute
# to the corresponding plot class.


@pytest.fixture(autouse=True, scope="class", params=get_plot_classes())
def plot_class(request):
    request.cls.PlotClass = request.param


class TestPlotSubClass(BasePlotTester):

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
