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
    
class TestBandsPlot(_TestPlot):

    _required_attrs = [
        "bands_shape", # Tuple specifying the shape of the bands dataarray
        "gap", # Float. The value of the gap in eV
        "ticklabels", # Array-like with the tick labels
        "tickvals", # Array-like with the expected positions of the ticks
        "spin_texture" # Whether spin texture should be possible to draw or not.
    ]

    @pytest.fixture(params=BandsPlot.get_class_param("backend").options)
    def backend(self, request):
        return request.param

    @pytest.fixture(scope="class", params=[
        "siesta_output", 
        "sisl_H_unpolarized", "sisl_H_polarized", "sisl_H_noncolinear", "sisl_H_spinorbit",
        "sisl_H_path_unpolarized",
    ])
    def init_func_and_attrs(self, request, siesta_test_files):
        name = request.param

        if name == "siesta_output":
            # From a siesta .bands file
            init_func = sisl.get_sile(siesta_test_files("SrTiO3.bands")).plot
            attrs = {
                "bands_shape": (150, 1, 72),
                "ticklabels": ('Gamma', 'X', 'M', 'Gamma', 'R', 'X'),
                "tickvals": [0.0, 0.429132, 0.858265, 1.465149, 2.208428, 2.815313],
                "gap": 1.677,
                "spin_texture": False
            }
        elif name.startswith("sisl_H"):
            gr = sisl.geom.graphene()
            H = sisl.Hamiltonian(gr)
            H.construct([(0.1, 1.44), (0, -2.7)])

            spin_type = name.split("_")[-1]
            n_spin, H = {
                "unpolarized": (1, H),
                "polarized": (2, H.transform(spin=sisl.Spin.POLARIZED)),
                "noncolinear": (1, H.transform(spin=sisl.Spin.NONCOLINEAR)),
                "spinorbit": (1, H.transform(spin=sisl.Spin.SPINORBIT))
            }.get(spin_type)

            n_states = 2
            if H.spin.is_spinorbit or H.spin.is_noncolinear:
                n_states *= 2

            # Let's create the same graphene bands plot using the hamiltonian
            # from two different prespectives    
            if name.startswith("sisl_H_path"):
                # Passing a list of points (as if we were interacting from a GUI)
                path = [{"active": True, "x": x, "y": y, "z": z, "divisions": 3,
                    "name": tick} for tick, (x, y, z) in zip(["Gamma", "M", "K"], [[0, 0, 0], [2/3, 1/3, 0], [1/2, 0, 0]])]

                init_func = partial(H.plot.bands, band_structure=path)
            else:
                # Directly creating a BandStructure object
                bz = sisl.BandStructure(H, [[0, 0, 0], [2/3, 1/3, 0], [1/2, 0, 0]], 6, ["Gamma", "M", "K"])
                init_func = bz.plot
            
            attrs = {
                "bands_shape": (6, n_spin, n_states),
                "ticklabels": ["Gamma", "M", "K"],
                "tickvals": [0., 1.70309799, 2.55464699],
                "gap": 0,
                "spin_texture": H.spin.is_spinorbit or H.spin.is_noncolinear
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

        plot.update_settings(custom_gaps=[{"from": gap[0], "to": gap[1], "spin": [0]} for gap in gaps])

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
        
        # We have finished with all the gaps tests here, so just clean up before continuing
        plot.update_settings(custom_gaps=[], gap=False)

    def test_spin_moments(self, plot, test_attrs):
        if not test_attrs["spin_texture"]:
            return

        # Check that spin moments have been calculated
        assert hasattr(plot, "spin_moments")

        # Check that it is a dataarray containing the right information
        spin_moments = plot.spin_moments
        assert isinstance(spin_moments, DataArray)
        assert spin_moments.dims == ('k', 'band', 'axis')
        assert spin_moments.shape == (test_attrs['bands_shape'][0], test_attrs['bands_shape'][-1], 3)

    def test_spin_texture(self, plot, test_attrs):
        if not test_attrs["spin_texture"]:
            return

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

