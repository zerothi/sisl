"""

Tests specific functionality of the bands plot.

Different inputs are tested (siesta .bands and sisl Hamiltonian).

"""

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
        """
        Check that the data array was created and contains the correct information.
        """

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
bz = sisl.BandStructure(H, [[0, 0, 0], [2/3, 1/3, 0], [1/2, 0, 0]], 9, ["Gamma", "M", "K"])


class TestBandsSislHamiltonian(BandsPlotTester):

    plot = BandsPlot(band_structure=bz)
    bands_shape = (9, 1, 2)
    gap = 0
    ticktext = ["Gamma", "M", "K"]
    tickvals = [0., 1.70309799, 2.55464699]

path = [{"active": True, "x": x, "y": y, "z": z, "divisions": 3,
            "tick": tick} for tick, (x, y, z) in zip(["Gamma", "M", "K"], [[0, 0, 0], [2/3, 1/3, 0], [1/2, 0, 0]])]


class TestBandsPathSislHamiltonian(TestBandsSislHamiltonian):

    plot = BandsPlot(H=H, path=path)
    bands_shape = (6, 1, 2)


H = sisl.get_sile(from_files("fe_clust_noncollinear.TSHS")).read_hamiltonian()
bz = sisl.BandStructure(H, [[0, 0, 0], [0.5,0,0]], 3, ["Gamma", "X"])

class TestNCSpinbands(BandsPlotTester):

    plot = bz.plot()
    bands_shape = (3, 1, 90)
    gap = 0.40
    ticktext = ["Gamma", "X"]
    tickvals = [0., 0.49472934]

    def test_spin_moments(self):

        plot = self.plot

        # Check that spin moments have been calculated
        assert hasattr(plot, "spin_moments")

        # Check that it is a dataarray containing the right information
        spin_moments = plot.spin_moments
        assert isinstance(spin_moments, DataArray)
        assert spin_moments.dims == ('k', 'band', 'axis')
        assert spin_moments.shape == (self.bands_shape[0], self.bands_shape[-1], 3)

    def test_spin_texture(self):

        plot = self.plot

        plot.update_settings(spin="x")

        # Check that spin texture has been 
        for band in range(*plot.settings["bands_range"]):
            expected = plot.spin_moments.sel(band=band, axis="x").values
            displayed = plot.data[band].marker.color

            assert np.all(expected == displayed), f"Colors of spin textured bands not correctly set (band {band})"

        plot.update_settings(spin=None)



        
