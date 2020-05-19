'''

Tests specific functionality of the bands plot.

Different inputs are tested (siesta .bands and sisl Hamiltonian).

'''

from xarray import DataArray
import numpy as np

import sisl
from sisl.viz.plots import GeometryPlot
from sisl.viz.plots.tests.get_files import from_files

# ------------------------------------------------------------
#      Build a generic tester for the geometry plot
# ------------------------------------------------------------


class GeometryPlotTester:

    plot = None
    
    def test_1d(self):
        pass

    def test_2d(self):
        pass

    def test_3d(self):
        pass