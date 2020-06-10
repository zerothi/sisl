'''

Tests specific functionality of the bands plot.

Different inputs are tested (siesta .bands and sisl Hamiltonian).

'''

from xarray import DataArray
import numpy as np

import sisl
from sisl.viz import BandsPlot
from sisl.viz.plotly.plots.tests.get_files import from_files

# ------------------------------------------------------------
#         Build a generic tester for the bands plot
# ------------------------------------------------------------

class BandsPlotTester:

    plot = None
    bands_shape = ()
    gap = None
    ticktext = None
    tickvals = None

    def test_bands_dataarray(self):
        '''
        Check that the data array was created and contains the correct information.
        '''

        # Check that there is a bands attribute
        assert hasattr(self.plot, 'bands')

        # Check that it is a dataarray containing the right information
        bands = self.plot.bands
        assert isinstance(bands, DataArray)
        assert bands.dims == ('k', 'spin', 'band')
        assert bands.shape == self.bands_shape
    
    def test_bands_in_figure(self):

        # Check if all bands are plotted
        self.plot.update_settings(bands_range=[0, self.bands_shape[-1]], Erange=None)
        assert len(self.plot.data) >= self.bands_shape[-1]

        # Now check if the ticks are correctly set
        assert np.allclose(list(self.tickvals), self.plot.figure.layout.xaxis.tickvals, rtol=0.01)
        assert np.all(list(self.ticktext) == list(self.plot.figure.layout.xaxis.ticktext))

    def test_gap(self):

        # Check that we can calculate the gap correctly
        # Allow for a small variability just in case there
        # are precision differences
        assert abs(self.plot.gap - self.gap) < 0.01

        # Check that the gap can be drawn correctly
        self.plot.update_settings(gap=True)
        assert len([True for trace in self.plot.data if trace.name == "Gap"]) > 0

# ------------------------------------------------------------
#       Test the bands plot reading from siesta .bands
# ------------------------------------------------------------
bands_file = from_files("SrTiO3.bands")

class TestBandsSiestaOutput(BandsPlotTester):

    plot = BandsPlot(bands_file=bands_file)
    bands_shape = (150, 1, 72)
    ticktext = ('Gamma', 'X', 'M', 'Gamma', 'R', 'X')
    tickvals = [0.0, 0.429132, 0.858265, 1.465149, 2.208428, 2.815313]
    gap = 1.677

# ------------------------------------------------------------
#     Test the bands plot reading from a sisl Hamiltonian
# ------------------------------------------------------------

gr = sisl.geom.graphene()
H = sisl.Hamiltonian(gr)
H.construct([(0.1, 1.44), (0, -2.7)])
bz = sisl.BandStructure(H, [[0,0,0], [2/3, 1/3, 0], [1/2, 0, 0]], 9, ["Gamma", "M", "K"])

class TestBandsSislHamiltonian(BandsPlotTester):

    plot = BandsPlot(H=H, band_structure=bz)
    bands_shape = (9, 1, 2)
    gap = 0
    ticktext = ["Gamma", "M", "K"]
    tickvals = [0., 1.70309799, 2.55464699]

path = [{"active": True, "x": x, "y": y, "z": z, "divisions": 3,
            "tick": tick} for tick, (x, y, z) in zip(["Gamma", "M", "K"], [[0, 0, 0], [2/3, 1/3, 0], [1/2, 0, 0]])]
            
class TestBandsPathSislHamiltonian(TestBandsSislHamiltonian):

    plot = BandsPlot(H=H, path=path)
    bands_shape = (6, 1, 2)
