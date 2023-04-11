# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

These tests check that all plot subclasses fulfill at least the most basic stuff

More tests should be run on each plot, but these are the most basic ones to
ensure that at least they do not break basic plot functionality.
"""
import pytest

from sisl.viz.tests.test_plot import _TestPlotClass
from sisl.viz.plots import *
from sisl.viz.plotutils import get_plot_classes


pytestmark = [pytest.mark.viz, pytest.mark.plotly]

# Test all plot subclasses with the subclass tester

# The following function basically tells pytest to run TestPlotSubClass
# once for each plot class. It takes care of setting the _cls attribute
# to the corresponding plot class.


@pytest.fixture(autouse=True, scope="class", params=get_plot_classes())
def plot_class(request):
    request.cls._cls = request.param


class TestPlotSubClass(_TestPlotClass):

    def test_compulsory_methods(self):

        assert hasattr(self._cls, "_set_data")
        assert callable(self._cls._set_data)

        assert hasattr(self._cls, "_plot_type")
        assert isinstance(self._cls._plot_type, str)

    def test_param_groups(self):

        plot = self._init_plot_without_warnings()

        for group in plot.param_groups:
            for key in ("key", "name", "icon", "description"):
                assert key in group, f'{self._cls.__name__} is missing {key} in parameters group {group}'
