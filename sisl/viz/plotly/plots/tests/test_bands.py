"""

Tests specific functionality of the bands plot.

Different inputs are tested (siesta .bands and sisl Hamiltonian).

"""
import os.path as osp
import itertools
from functools import partial
import pytest
from xarray import DataArray
import numpy as np

import sisl
from sisl.viz.plotly import BandsPlot
from sisl.viz.plotly.plots.tests.conftest import PlotTester


pytestmark = [pytest.mark.viz, pytest.mark.plotly]
_dir = osp.join('sisl', 'io', 'siesta')


class BandsPlotTester(PlotTester):

    _required_attrs = [
        "bands_shape", # Tuple specifying the shape of the bands dataarray
        "gap", # Float. The value of the gap in eV
        "ticklabels", # Array-like with the tick labels
        "tickvals" # Array-like with the expected positions of the ticks
    ]

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
        assert np.all(list(self.ticklabels) == list(self.plot.figure.layout.xaxis.ticktext))

    def test_gap(self):

        # Check that we can calculate the gap correctly
        # Allow for a small variability just in case there
        # are precision differences
        assert abs(self.plot.gap - self.gap) < 0.01

        # Check that the gap can be drawn correctly
        self.plot.update_settings(gap=True)
        assert len([True for trace in self.plot.data if trace.name == "Gap"]) > 0

    def test_custom_gaps(self):

        plot = self.plot

        plot.update_settings(gap=False, custom_gaps=[])

        prev_traces = len(plot.data)

        gaps = list(itertools.combinations(self.ticklabels, 2))

        plot.update_settings(custom_gaps=[{"from": gap[0], "to": gap[1]} for gap in gaps])

        assert len(plot.data) == prev_traces + len(gaps)

        # Get the traces that have been generated and assert that they are
        # exactly the same as if we define the gaps with numerical values for the ks
        from_labels = plot.data[-len(gaps):]
        gaps = list(itertools.combinations(self.tickvals, 2))

        plot.update_settings(
            custom_gaps=[{"from": gap[0], "to": gap[1]} for gap in gaps])

        assert len(plot.data) == prev_traces + len(gaps)
        assert np.all([
            np.allclose(old_trace.y, new_trace.y)
            for old_trace, new_trace in zip(from_labels, plot.data[-len(gaps):])])


class NCSpinBandsTester(BandsPlotTester):

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

        # If this is a fatbands plot, the first traces are drawing the weights
        # (the actual fatbands). Therefore we need to find where the traces that
        # belong to the bands begin.
        if plot.data[0].fill is None:
            first_band_trace = 0
        else:
            for i, trace in enumerate(plot.data):
                if trace.fill is None:
                    first_band_trace = i
                    break
            else:
                raise Exception(f"We didn't find any band traces in the plot")

        # Check that spin texture has been
        for band in range(*plot.settings["bands_range"]):
            expected = plot.spin_moments.sel(band=band, axis="x").values
            displayed = plot.data[first_band_trace + band].marker.color

            assert np.all(expected == displayed), f"Colors of spin textured bands not correctly set (band {band})"

        plot.update_settings(spin=None)

# Define the dictionary where we will store all the plots that we want to try out
bands_plots = {}

# ---- From siesta.bands

bands_plots["siesta_output"] = {
    "plot_file": osp.join(_dir, "SrTiO3.bands"),
    "bands_shape": (150, 1, 72),
    "ticklabels": ('Gamma', 'X', 'M', 'Gamma', 'R', 'X'),
    "tickvals": [0.0, 0.429132, 0.858265, 1.465149, 2.208428, 2.815313],
    "gap": 1.677
}

# ---- From a hamiltonian generated in sisl

gr = sisl.geom.graphene()
H = sisl.Hamiltonian(gr)
H.construct([(0.1, 1.44), (0, -2.7)])
bz = sisl.BandStructure(H, [[0, 0, 0], [2/3, 1/3, 0], [1/2, 0, 0]], 9, ["Gamma", "M", "K"])

bands_plots["sisl_H"] = {
    "init_func": bz.plot,
    "bands_shape": (9, 1, 2),
    "ticklabels": ["Gamma", "M", "K"],
    "tickvals": [0., 1.70309799, 2.55464699],
    "gap": 0
}

# ---- From a hamiltonian generated in sisl but passing a path
# ---- (as if we were providing the input from the GUI)

path = [{"active": True, "x": x, "y": y, "z": z, "divisions": 3,
            "tick": tick} for tick, (x, y, z) in zip(["Gamma", "M", "K"], [[0, 0, 0], [2/3, 1/3, 0], [1/2, 0, 0]])]

bands_plots["sisl_H_path"] = {
    "init_func": partial(H.plot.bands, path=path),
    "bands_shape": (6, 1, 2),
    "ticklabels": ["Gamma", "M", "K"],
    "tickvals": [0., 1.70309799, 2.55464699],
    "gap": 0,
}

# ---- From a non collinear calculation in SIESTA


def NC_init_func(sisl_files, **kwargs):
    TSHS_path = osp.join(_dir, "fe_clust_noncollinear.TSHS")
    H = sisl.get_sile(sisl_files(TSHS_path)).read_hamiltonian()
    bz = sisl.BandStructure(H, [[0, 0, 0], [0.5, 0, 0]], 3, ["Gamma", "X"])

    return bz.plot(**kwargs)


class TestBandsPlot(BandsPlotTester):
    run_for = bands_plots


class TestNCSpinBands(NCSpinBandsTester):

    run_for = {
        "NCspin_H": {
            "init_func": NC_init_func,
            "bands_shape": (3, 1, 90),
            "ticklabels": ["Gamma", "X"],
            "tickvals": [0., 0.49472934],
            "gap": 0.40109,
        }
    }
