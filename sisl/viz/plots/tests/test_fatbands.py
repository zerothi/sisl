'''

Tests specific functionality of a fatbands plot

'''

from xarray import DataArray
import numpy as np

import sisl
from sisl.viz.plots import FatbandsPlot
from sisl.viz.plots.tests.test_bands import BandsPlotTester

# ------------------------------------------------------------
#         Build a generic tester for the bands plot
# ------------------------------------------------------------

class FatbandsPlotTester(BandsPlotTester):

    def test_weights_dataarray(self):
        '''
        Check that the data array was created and contains the correct information.
        '''

        # Check that there is a weights attribute
        assert hasattr(self.plot, 'weights')

        # Check that it is a dataarray containing the right information
        weights = self.plot.weights
        assert isinstance(weights, DataArray)
        assert weights.dims == ('k', 'band', 'orb')
        assert weights.shape == self.weights_shape
    
    def test_groups(self):
        '''
        Check that we can request groups
        '''

        color = "green"
        name = "Nice group"

        self.plot.update_settings(groups=[{"atoms": [1], "color": color, "name": name}])

        fatbands_traces = [ trace for trace in self.plot.data if trace.fill == 'toself']

        assert len(fatbands_traces) > 0
        assert fatbands_traces[0].line.color == color
        assert fatbands_traces[0].name == name



# ------------------------------------------------------------
#    Test the fatbands plot reading from a sisl Hamiltonian
# ------------------------------------------------------------

gr = sisl.geom.graphene()
H = sisl.Hamiltonian(gr)
H.construct([(0.1, 1.44), (0, -2.7)])
bz = sisl.BandStructure(H, [[0,0,0], [2/3, 1/3, 0], [1/2, 0, 0]], 9, ["Gamma", "M", "K"])

class TestFatbandsSislHamiltonian(FatbandsPlotTester):

    plot = FatbandsPlot(H=H, band_structure=bz)
    bands_shape = (9, 1, 2)
    weights_shape = (9, 2, 2)
    gap = 0
    ticktext = ["Gamma", "M", "K"]
    tickvals = [0., 1.70309799, 2.55464699]
    groups = {}


class TestFatbandsBandStructure(TestFatbandsSislHamiltonian):

    plot = FatbandsPlot(band_structure=bz)
