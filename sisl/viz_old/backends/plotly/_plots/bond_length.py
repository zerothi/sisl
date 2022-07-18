# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from ....plots import BondLengthMap
from .geometry import PlotlyGeometryBackend
from ...templates import BondLengthMapBackend


class PlotlyBondLengthMapBackend(BondLengthMapBackend, PlotlyGeometryBackend):

    def draw_2D(self, backend_info, **kwargs):
        super().draw_2D(backend_info, **kwargs)
        self._setup_coloraxis(backend_info)

    def draw_3D(self, backend_info, **kwargs):
        super().draw_3D(backend_info, **kwargs)
        self._setup_coloraxis(backend_info)

    def _setup_coloraxis(self, backend_info):
        if "bonds_coloraxis" in backend_info:
            self.update_layout(coloraxis=backend_info["bonds_coloraxis"])

        self.update_layout(legend_orientation='h')

BondLengthMap.backends.register("plotly", PlotlyBondLengthMapBackend)
