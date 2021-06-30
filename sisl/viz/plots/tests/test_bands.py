# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

Tests specific functionality of the bands plot.

Different inputs are tested (siesta .bands and sisl Hamiltonian).

"""
import itertools
from functools import partial
import pytest
from xarray import DataArray
import numpy as np

import sisl
from sisl.viz import BandsPlot
from sisl.viz.plots.tests.conftest import _TestPlot

pytestmark = [pytest.mark.viz, pytest.mark.plotly]

@pytest.fixture(params=BandsPlot.get_class_param("backend").options)
def backend(request):
    return request.param
    
class TestBandsPlot(_TestPlot):

    _required_attrs = [
        "bands_shape", # Tuple specifying the shape of the bands dataarray
        "gap", # Float. The value of the gap in eV
        "ticklabels", # Array-like with the tick labels
        "tickvals" # Array-like with the expected positions of the ticks
    ]

    @pytest.fixture(scope="class", params=["siesta_output", "sisl_H", "sisl_H_path"])
    def init_func_and_attrs(self, request, siesta_test_files):
        name = request.param

        if name == "siesta_output":
            # From a siesta .bands file
            init_func = sisl.get_sile(siesta_test_files("SrTiO3.bands")).plot
            attrs = {
                "bands_shape": (150, 1, 72),
                "ticklabels": ('Gamma', 'X', 'M', 'Gamma', 'R', 'X'),
                "tickvals": [0.0, 0.429132, 0.858265, 1.465149, 2.208428, 2.815313],
                "gap": 1.677
            }
        elif "sisl_H" in name:
            gr = sisl.geom.graphene()
            H = sisl.Hamiltonian(gr)
            H.construct([(0.1, 1.44), (0, -2.7)])

            if name == "sisl_H":
                # From a hamiltonian generated in sisl
                bz = sisl.BandStructure(H, [[0, 0, 0], [2/3, 1/3, 0], [1/2, 0, 0]], 9, ["Gamma", "M", "K"])
                init_func = bz.plot
                attrs = {
                    "bands_shape": (9, 1, 2),
                    "ticklabels": ["Gamma", "M", "K"],
                    "tickvals": [0., 1.70309799, 2.55464699],
                    "gap": 0
                }
            elif name == "sisl_H_path":
                # From a hamiltonian generated in sisl but passing a path
                # (as if we were providing the input from the GUI)
                path = [{"active": True, "x": x, "y": y, "z": z, "divisions": 3,
                    "tick": tick} for tick, (x, y, z) in zip(["Gamma", "M", "K"], [[0, 0, 0], [2/3, 1/3, 0], [1/2, 0, 0]])]

                init_func = partial(H.plot.bands, path=path)
                attrs =  {
                    "bands_shape": (6, 1, 2),
                    "ticklabels": ["Gamma", "M", "K"],
                    "tickvals": [0., 1.70309799, 2.55464699],
                    "gap": 0,
                }
            
        return init_func, attrs

    def test_bands_dataarray(self, plot, test_attrs):
        """
        Check that the data array was created and contains the correct information.
        """

        # Check that there is a bands attribute
        assert hasattr(plot, 'bands')

        # Check that it is a dataarray containing the right information
        bands = plot.bands
        assert isinstance(bands, DataArray)
        assert bands.dims == ('k', 'spin', 'band')
        assert bands.shape == test_attrs['bands_shape']

    def test_bands_in_figure(self, plot, test_attrs):

        # Check if all bands are plotted
        plot.update_settings(bands_range=[0, test_attrs['bands_shape'][-1]], Erange=None)
        assert len(plot.data) >= test_attrs['bands_shape'][-1]

        # Now check if the ticks are correctly set
        assert np.allclose(list(test_attrs['tickvals']), plot.figure.layout.xaxis.tickvals, rtol=0.01)
        assert np.all(list(test_attrs['ticklabels']) == list(plot.figure.layout.xaxis.ticktext))

    def test_gap(self, plot, test_attrs):

        # Check that we can calculate the gap correctly
        # Allow for a small variability just in case there
        # are precision differences
        assert abs(plot.gap - test_attrs['gap']) < 0.01

    def test_gap_in_figure(self, plot, backend):
        # Check that the gap can be drawn correctly
        plot.update_settings(backend=backend, gap=False)
        assert not plot._test_is_gap_drawn(), f"Test for gap doesn't work properly in {backend} backend"

        plot.update_settings(gap=True)
        assert plot._test_is_gap_drawn(), f"Gap is not drawn by {backend} backend"

    def test_custom_gaps_in_figure(self, plot, test_attrs, backend):

        plot.update_settings(gap=False, custom_gaps=[], backend=backend)

        prev_traces = plot._test_number_of_items_drawn()

        gaps = list(itertools.combinations(test_attrs['ticklabels'], 2))

        plot.update_settings(custom_gaps=[{"from": gap[0], "to": gap[1]} for gap in gaps])

        assert plot._test_number_of_items_drawn() == prev_traces + len(gaps)
    
    def test_custom_gaps_correct(self, plot, test_attrs):

        # We only test this with plotly.
        gaps = list(itertools.combinations(test_attrs['ticklabels'], 2))
        plot.update_settings(custom_gaps=[{"from": gap[0], "to": gap[1]} for gap in gaps], backend="plotly")

        n_items = plot._test_number_of_items_drawn()

        # Get the traces that have been generated and assert that they are
        # exactly the same as if we define the gaps with numerical values for the ks
        from_labels = plot.data[-len(gaps):]
        gaps = list(itertools.combinations(test_attrs['tickvals'], 2))

        plot.update_settings(
            custom_gaps=[{"from": gap[0], "to": gap[1]} for gap in gaps])

        assert plot._test_number_of_items_drawn() == n_items
        assert np.all([
            np.allclose(old_trace.y, new_trace.y)
            for old_trace, new_trace in zip(from_labels, plot.data[-len(gaps):])])


class TestBandsPlotNC(TestBandsPlot):

    @pytest.fixture(scope="class")
    def init_func_and_attrs(self, siesta_test_files):

        H = sisl.get_sile(siesta_test_files("fe_clust_noncollinear.TSHS")).read_hamiltonian()
        bz = sisl.BandStructure(H, [[0, 0, 0], [0.5, 0, 0]], 3, ["Gamma", "X"])
        init_func = bz.plot

        attrs = { 
            "bands_shape": (3, 1, 90),
            "ticklabels": ["Gamma", "X"],
            "tickvals": [0., 0.49472934],
            "gap": 0.40109,
        }
        return init_func, attrs 

    def test_spin_moments(self, plot, test_attrs):

        # Check that spin moments have been calculated
        assert hasattr(plot, "spin_moments")

        # Check that it is a dataarray containing the right information
        spin_moments = plot.spin_moments
        assert isinstance(spin_moments, DataArray)
        assert spin_moments.dims == ('k', 'band', 'axis')
        assert spin_moments.shape == (test_attrs['bands_shape'][0], test_attrs['bands_shape'][-1], 3)

    def test_spin_texture(self, plot):

        plot.update_settings(spin="x", backend="plotly")

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

