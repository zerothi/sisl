# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

Tests specific functionality of a fatbands plot

"""
import pytest
import numpy as np
from functools import partial

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

    @pytest.fixture(scope="class", params=[None, *sisl.viz.FatbandsPlot.get_class_param("backend").options])
    def backend(self, request):
        return request.param

    @pytest.fixture(scope="class", params=[
        "sisl_H_unpolarized", "sisl_H_polarized", "sisl_H_noncolinear", "sisl_H_spinorbit",
        "sisl_H_unpolarized_jump",
        "wfsx file",
    ])
    def init_func_and_attrs(self, request, siesta_test_files):
        name = request.param

        if name.startswith("sisl_H"):
            gr = sisl.geom.graphene()
            H = sisl.Hamiltonian(gr)
            H.construct([(0.1, 1.44), (0, -2.7)])

            spin_type = name.split("_")[2]
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
            if name.endswith("jump"):
                names = ["Gamma", "M", "M", "K"]
                bz = sisl.BandStructure(H, [[0, 0, 0], [2/3, 1/3, 0], None, [2/3, 1/3, 0], [1/2, 0, 0]], 6, names)
                nk = 7
                tickvals = [0., 1.70309799, 1.83083034, 2.68237934]
            else:
                names = ["Gamma", "M", "K"]
                bz = sisl.BandStructure(H, [[0, 0, 0], [2/3, 1/3, 0], [1/2, 0, 0]], 6, names)
                nk = 6
                tickvals = [0., 1.70309799, 2.55464699]
            init_func = bz.plot.fatbands

            attrs = {
                "bands_shape": (nk, n_spin, n_states) if H.spin.is_polarized else (nk, n_states),
                "weights_shape": (n_spin, nk, n_states, 2) if H.spin.is_polarized else (nk, n_states, 2),
                "ticklabels": names,
                "tickvals": tickvals,
                "gap": 0,
                "spin_texture": not H.spin.is_diagonal,
                "spin": H.spin
            }
        elif name == "wfsx file":
            # From a siesta bands.WFSX file
            # Since there is no hamiltonian for bi2se3_3ql.fdf, we create a dummy one
            wfsx = sisl.get_sile(siesta_test_files("bi2se3_3ql.bands.WFSX"))

            geometry = sisl.get_sile(siesta_test_files("bi2se3_3ql.fdf")).read_geometry()
            geometry = sisl.Geometry(geometry.xyz, atoms=wfsx.read_basis())

            H = sisl.Hamiltonian(geometry, dim=4)

            init_func = partial(H.plot.fatbands, wfsx_file=wfsx, E0=-51.68, entry_points_order=["wfsx file"])
            attrs = {
                "bands_shape": (16, 8),
                "weights_shape": (16, 8, 195),
                "ticklabels": None,
                "tickvals": None,
                "gap": 0.0575,
                "spin_texture": False,
                "spin": sisl.Spin("nc")
            }

        return init_func, attrs

    def test_weights_dataarray_avail(self, plot, test_attrs):
        """
        Check that the data array was created and contains the correct information.
        """
        pytest.importorskip("xarray")
        from xarray import DataArray

        # Check that there is a weights attribute
        assert hasattr(plot, "weights")

        # Check that it is a dataarray containing the right information
        weights = plot.weights
        assert isinstance(weights, DataArray)

        if test_attrs["spin"].is_polarized:
            expected_dims = ("spin", "k", "band", "orb")
        else:
            expected_dims = ("k", "band", "orb")
        assert weights.dims == expected_dims
        assert weights.shape == test_attrs["weights_shape"]

    def test_group_weights(self, plot):
        pytest.importorskip("xarray")
        from xarray import DataArray

        total_weights = plot._get_group_weights({})

        assert isinstance(total_weights, DataArray)
        assert set(total_weights.dims) == set(("spin", "band", "k"))

    def test_weights_values(self, plot, test_attrs):
        # Check that all states are normalized.
        assert np.allclose(plot.weights.dropna("k", "all").sum("orb"), 1, atol=0.05), "Weight values do not sum 1 for all states."

        # If we have all the bands of the system, assert that orbitals are also "normalized".
        factor = 2 if not test_attrs["spin"].is_diagonal else 1
        if len(plot.weights.band) * factor == len(plot.weights.orb):
            assert np.allclose(plot.weights.dropna("k", "all").sum("band"), factor)

    def test_groups(self, plot, test_attrs):
        """
        Check that we can request groups
        """
        pytest.importorskip("xarray")
        from xarray import DataArray

        color = "green"
        name = "Nice group"

        plot.update_settings(
            groups=[{"atoms": [1], "color": color, "name": name}],
            bands_range=None, Erange=None
        )

        assert "groups_weights" in plot._for_backend
        assert len(plot._for_backend["groups_weights"]) == 1
        assert name in plot._for_backend["groups_weights"]

        group_weights = plot._for_backend["groups_weights"][name]
        assert isinstance(group_weights, DataArray)
        assert set(group_weights.dims) == set(("spin", "k", "band"))
        group_weights_shape = test_attrs["weights_shape"][:-1]
        if not test_attrs["spin"].is_polarized:
            group_weights_shape = (1, *group_weights_shape)
        assert group_weights.transpose("spin", "k", "band").shape == group_weights_shape

        assert "groups_metadata" in plot._for_backend
        assert len(plot._for_backend["groups_metadata"]) == 1
        assert name in plot._for_backend["groups_metadata"]
        assert plot._for_backend["groups_metadata"][name]["style"]["line"]["color"] == color

    @pytest.mark.parametrize("request_atoms", [None, {"index": 0}])
    def _test_split_groups(self, plot, constraint_atoms):

        # Number of groups that each splitting should give
        expected_splits = [
            ('species', len(plot.geometry.atoms.atom)),
            ('atoms', plot.geometry.na),
            ('orbitals', plot.geometry.no)
        ]

        plot.update_settings(groups=[])
        # Check that there are no groups
        assert len(plot._for_backend["groups_weights"]) == 0
        assert len(plot._for_backend["groups_metadata"]) == 0

        # Check that each splitting works as expected
        for group_by, n_groups in expected_splits:
            plot.split_groups(group_by, atoms=constraint_atoms)
            if constraint_atoms is None:
                err_message = f'Not correctly grouping by {group_by}'
                assert len(plot._for_backend["groups_weights"]) == n_groups, err_message
                assert len(plot._for_backend["groups_metadata"]) == n_groups, err_message
