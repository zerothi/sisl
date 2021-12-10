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
        "spin_texture", # Whether spin texture should be possible to draw or not.
        "spin", # The spin class of the calculation
    ]

    @pytest.fixture(scope="class", params=[None, *BandsPlot.get_class_param("backend").options])
    def backend(self, request):
        return request.param

    @pytest.fixture(scope="class", params=[
        # From .bands file
        "siesta_output",
        # From a hamiltonian
        "sisl_H_unpolarized", "sisl_H_polarized", "sisl_H_noncolinear", "sisl_H_spinorbit",
        "sisl_H_path_unpolarized",
        # From a .bands.WFSX file
        "wfsx_file"
    ])
    def init_func_and_attrs(self, request, siesta_test_files):
        name = request.param

        if name == "siesta_output":
            # From a siesta .bands file
            init_func = sisl.get_sile(siesta_test_files("SrTiO3.bands")).plot
            attrs = {
                "bands_shape": (150, 72),
                "ticklabels": ('Gamma', 'X', 'M', 'Gamma', 'R', 'X'),
                "tickvals": [0.0, 0.429132, 0.858265, 1.465149, 2.208428, 2.815313],
                "gap": 1.677,
                "spin_texture": False,
                "spin": sisl.Spin("")
            }
        elif name == "wfsx_file":
            # From the SIESTA .bands.WFSX file
            fdf = sisl.get_sile(siesta_test_files("bi2se3_3ql.fdf"))
            wfsx = siesta_test_files("bi2se3_3ql.bands.WFSX")
            init_func = partial(fdf.plot.bands, wfsx_file=wfsx, E0=-51.68, entry_points_order=["wfsx file"])
            attrs = {
                "bands_shape": (16, 8),
                "ticklabels": None,
                "tickvals": None,
                "gap": 0.0575,
                "spin_texture": False,
                "spin": sisl.Spin("nc")
            }

        elif name.startswith("sisl_H"):
            gr = sisl.geom.graphene()
            H = sisl.Hamiltonian(gr)
            H.construct([(0.1, 1.44), (0, -2.7)])

            spin_type = name.split("_")[-1]
            n_spin, H = {
                "unpolarized": (0, H),
                "polarized": (2, H.transform(spin=sisl.Spin.POLARIZED)),
                "noncolinear": (0, H.transform(spin=sisl.Spin.NONCOLINEAR)),
                "spinorbit": (0, H.transform(spin=sisl.Spin.SPINORBIT))
            }.get(spin_type)

            n_states = 2
            if not H.spin.is_diagonal:
                n_states *= 2

            # Let's create the same graphene bands plot using the hamiltonian
            # from two different prespectives
            if name.startswith("sisl_H_path"):
                # Passing a list of points (as if we were interacting from a GUI)
                # We want 6 points in total. This is the path that we want to get:
                # [0,0,0] --2-- [2/3, 1/3, 0] --1-- [1/2, 0, 0]
                path = [{"active": True, "x": x, "y": y, "z": z, "divisions": 3,
                    "name": tick} for tick, (x, y, z) in zip(["Gamma", "M", "K"], [[0, 0, 0], [2/3, 1/3, 0], [1/2, 0, 0]])]
                path[-1]['divisions'] = 2

                init_func = partial(H.plot.bands, band_structure=path)
            else:
                # Directly creating a BandStructure object
                bz = sisl.BandStructure(H, [[0, 0, 0], [2/3, 1/3, 0], [1/2, 0, 0]], 6, ["Gamma", "M", "K"])
                init_func = bz.plot

            attrs = {
                "bands_shape": (6, n_spin, n_states) if n_spin != 0 else (6, n_states),
                "ticklabels": ["Gamma", "M", "K"],
                "tickvals": [0., 1.70309799, 2.55464699],
                "gap": 0,
                "spin_texture": not H.spin.is_diagonal,
                "spin": H.spin
            }

        return init_func, attrs

    def _check_bands_array(self, bands, spin, expected_shape):
        pytest.importorskip("xarray")
        from xarray import DataArray

        assert isinstance(bands, DataArray)

        if spin.is_polarized:
            expected_coords = ('k', 'spin', 'band')
        else:
            expected_coords = ('k', 'band')

        assert set(bands.dims) == set(expected_coords)
        assert bands.transpose(*expected_coords).shape == expected_shape

    def test_bands_dataarray(self, plot, test_attrs):
        """
        Check that the data array was created and contains the correct information.
        """
        # Check that there is a bands attribute
        assert hasattr(plot, 'bands')

        self._check_bands_array(plot.bands, test_attrs["spin"], test_attrs['bands_shape'])

    def test_bands_filtered(self, plot, test_attrs):
        # Check that we can correctly filter the bands to draw.
        plot.update_settings(bands_range=[0, 1], Erange=None)

        # Check that the filtered bands are correctly passed to the backend
        assert "draw_bands" in plot._for_backend
        assert "filtered_bands" in plot._for_backend["draw_bands"]

        # Check that everything is fine with the dimensions of the filtered bands. Since we filtered,
        # it should contain only one band
        filtered_bands = plot._for_backend["draw_bands"]["filtered_bands"]
        self._check_bands_array(filtered_bands, test_attrs["spin"], (*test_attrs["bands_shape"][:-1], 1))

    def test_gap(self, plot, test_attrs):
        # Check that we can calculate the gap correctly
        # Allow for a small variability just in case there
        # are precision differences
        assert abs(plot.gap - test_attrs['gap']) < 0.01

    def test_gap_to_backend(self, plot, test_attrs):
        # Check that the gap is correctly transmitted to the backend
        plot.update_settings(gap=False, custom_gaps=[])
        assert len(plot._for_backend["gaps"]) == 0

        plot.update_settings(gap=True)
        assert len(plot._for_backend["gaps"]) > 0
        for gap in plot._for_backend["gaps"]:
            assert len(set(["ks", "Es", "color", "name"]) - set(gap)) == 0
            assert abs(np.diff(gap["Es"]) - test_attrs['gap']) < 0.01

    def test_custom_gaps_to_backend(self, plot, test_attrs):
        if test_attrs['ticklabels'] is None:
            return

        plot.update_settings(gap=False, custom_gaps=[])
        assert len(plot._for_backend["gaps"]) == 0

        gaps = list(itertools.combinations(test_attrs['ticklabels'], 2))

        plot.update_settings(custom_gaps=[{"from": gap[0], "to": gap[1], "spin": [0]} for gap in gaps])

        assert len(plot._for_backend["gaps"]) + len(gaps)
        for gap in plot._for_backend["gaps"]:
            assert len(set(["ks", "Es", "color", "name"]) - set(gap)) == 0

    def test_custom_gaps_correct(self, plot, test_attrs):
        if test_attrs['ticklabels'] is None:
            return

        # Generate custom gaps from labels
        gaps = list(itertools.combinations(test_attrs['ticklabels'], 2))
        plot.update_settings(custom_gaps=[{"from": gap[0], "to": gap[1]} for gap in gaps])

        gaps_from_labels = np.unique([np.diff(gap["Es"]) for gap in plot._for_backend["gaps"]])

        # Generate custom gaps from k values
        gaps = list(itertools.combinations(test_attrs['tickvals'], 2))
        plot.update_settings(custom_gaps=[{"from": gap[0], "to": gap[1]} for gap in gaps])

        # Check that we get the same values for the gaps
        assert abs(gaps_from_labels.sum() - np.unique([np.diff(gap["Es"]) for gap in plot._for_backend["gaps"]]).sum()) < 0.03

        # We have finished with all the gaps tests here, so just clean up before continuing
        plot.update_settings(custom_gaps=[], gap=False)

    def test_spin_moments(self, plot, test_attrs):
        if not test_attrs["spin_texture"]:
            return
        pytest.importorskip("xarray")
        from xarray import DataArray

        # Check that spin moments have been calculated
        assert hasattr(plot, "spin_moments")

        # Check that it is a dataarray containing the right information
        spin_moments = plot.spin_moments
        assert isinstance(spin_moments, DataArray)
        assert set(spin_moments.dims) == set(('k', 'band', 'axis'))
        assert spin_moments.shape == (test_attrs['bands_shape'][0], test_attrs['bands_shape'][-1], 3)

    def test_spin_texture(self, plot, test_attrs):
        assert plot._for_backend["draw_bands"]["spin_texture"]["show"] is False

        if not test_attrs["spin_texture"]:
            return

        plot.update_settings(spin="x", bands_range=[0, 1], Erange=None)

        spin_texture = plot._for_backend["draw_bands"]["spin_texture"]
        assert spin_texture["show"] is True
        assert "colorscale" in spin_texture
        assert "values" in spin_texture

        spin_texture_arr = spin_texture["values"]

        self._check_bands_array(spin_texture_arr, test_attrs["spin"], (*test_attrs["bands_shape"][:-1], 1))
        assert "axis" in spin_texture_arr.coords
        assert str(spin_texture_arr.axis.values) == "x"

        plot.update_settings(spin=None)
