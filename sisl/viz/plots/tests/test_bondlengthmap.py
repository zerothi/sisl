# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from functools import partial

import pytest
import numpy as np

import sisl

from sisl.viz.plots.tests.test_geometry import TestGeometry as _TestGeometry


pytestmark = [pytest.mark.viz, pytest.mark.plotly]

# ------------------------------------------------------------
#      Build a generic tester for bond length plot
# ------------------------------------------------------------


class TestBondLengthMap(_TestGeometry):

    _required_attrs = ["has_strain_ref"]

    @pytest.fixture(scope="class", params=[None, *sisl.viz.BondLengthMap.get_class_param("backend").options])
    def backend(self, request):
        return request.param

    @pytest.fixture(scope="class", params=["sisl_geom", "sisl_geom_strain"])
    def init_func_and_attrs(self, request):
        name = request.param

        if name.startswith("sisl_geom"):
            geometry = sisl.geom.graphene(orthogonal=True, bond=1.35)

            if name.endswith("strain"):
                kwargs = {"strain_ref": sisl.geom.graphene(orthogonal=True)}
                attrs = {"has_strain_ref": True}
            else:
                kwargs = {}
                attrs = {"has_strain_ref": False}

            init_func = partial(geometry.plot.bondlengthmap, **kwargs)

        return init_func, attrs

    def test_strain_ref(self, plot, test_attrs):
        if test_attrs["has_strain_ref"]:
            plot.update_settings(axes=[0, 1, 2], strain=True, show_bonds=True)

            strains = [bond["color"] for bond in plot._for_backend["bonds_props"]]

            plot.update_settings(strain=False)
            assert not np.allclose([bond["color"] for bond in plot._for_backend["bonds_props"]], strains)
