# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

Tests specific functionality of a fatbands plot

"""
import pytest
import numpy as np
from xarray import DataArray

import sisl
from sisl.viz.plots.tests.test_bands import TestBandsPlot as _TestBandsPlot


pytestmark = [pytest.mark.viz, pytest.mark.plotly]

# ------------------------------------------------------------
#         Build a generic tester for the bands plot
# ------------------------------------------------------------


class TestFatbandsPlot(_TestBandsPlot):

    _required_attrs = [
        *_TestBandsPlot._required_attrs,
        "weights_shape", # Tuple. The shape that self.weights dataarray is expected to have
    ]

    @pytest.fixture(scope="class", params=[
        "sisl_H_unpolarized", "sisl_H_polarized", "sisl_H_noncolinear", "sisl_H_spinorbit",
    ])
    def init_func_and_attrs(self, request, siesta_test_files):
        name = request.param

        if name.startswith("sisl_H"):
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

            # Directly creating a BandStructure object
            bz = sisl.BandStructure(H, [[0, 0, 0], [2/3, 1/3, 0], [1/2, 0, 0]], 6, ["Gamma", "M", "K"])
            init_func = bz.plot.fatbands
            
            attrs = {
                "bands_shape": (6, n_spin, n_states),
                "weights_shape": (n_spin, 6, n_states, 2),
                "ticklabels": ["Gamma", "M", "K"],
                "tickvals": [0., 1.70309799, 2.55464699],
                "gap": 0,
                "spin_texture": H.spin.is_spinorbit or H.spin.is_noncolinear,
                "soc_or_nc": H.spin.is_spinorbit or H.spin.is_noncolinear,
            }
            
        return init_func, attrs

    def test_weights_dataarray_avail(self, plot, test_attrs):
        """
        Check that the data array was created and contains the correct information.
        """

        # Check that there is a weights attribute
        assert hasattr(plot, "weights")

        # Check that it is a dataarray containing the right information
        weights = plot.weights
        assert isinstance(weights, DataArray)
        assert weights.dims == ("spin", "k", "band", "orb")
        assert weights.shape == test_attrs["weights_shape"]
    
    def test_weights_values(self, plot, test_attrs):
        assert np.allclose(plot.weights.sum("orb"), 1), "Weight values do not sum 1 for all states."
        assert np.allclose(plot.weights.sum("band"), 2 if test_attrs["soc_or_nc"] else 1)

    def test_groups(self, plot):
        """
        Check that we can request groups
        """
        color = "green"
        name = "Nice group"

        plot.update_settings(groups=[{"atoms": [1], "color": color, "name": name}])

        fatbands_traces = [trace for trace in plot.data if trace.fill == 'toself']

        assert len(fatbands_traces) > 0
        assert fatbands_traces[0].line.color == color
        assert fatbands_traces[0].name == name

    def test_split_groups(self, plot):

        # Number of groups that each splitting should give
        expected_splits = [
            ('species', len(plot.geometry.atoms.atom)),
            ('atoms', plot.geometry.na),
            ('orbitals', plot.geometry.no)
        ]

        # Check how many traces are there before generating groups
        # (these traces correspond to bands)
        plot.update_settings(groups=[])
        traces_before = len(plot.data)

        # Check that each splitting works as expected
        for group_by, length in expected_splits:

            plot.split_groups(group_by)
            err_message = f'Not correctly grouping by {group_by}'
            assert len(plot.data) - traces_before, err_message

