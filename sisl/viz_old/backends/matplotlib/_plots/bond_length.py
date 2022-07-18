# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from ....plots import BondLengthMap
from .geometry import MatplotlibGeometryBackend
from ...templates import BondLengthMapBackend


class MatplotlibBondLengthMapBackend(BondLengthMapBackend, MatplotlibGeometryBackend):

    def draw_2D(self, backend_info, **kwargs):
        self._colorscale = None
        if "bonds_coloraxis" in backend_info:
            self._colorscale = backend_info["bonds_coloraxis"]["colorscale"]

        super().draw_2D(backend_info, **kwargs)

    def _draw_bonds_2D_multi_color_size(self, *args, **kwargs):
        kwargs["colorscale"] = self._colorscale
        super()._draw_bonds_2D_multi_color_size(*args, **kwargs)

BondLengthMap.backends.register("matplotlib", MatplotlibBondLengthMapBackend)
