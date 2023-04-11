# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from ....plots import PdosPlot
from ..backend import PlotlyBackend
from ...templates import PdosBackend


class PlotlyPDOSBackend(PlotlyBackend, PdosBackend):

    _layout_defaults = {
        'xaxis_title': 'Density of states [1/eV]',
        'xaxis_mirror': True,
        'yaxis_mirror': True,
        'yaxis_title': 'Energy [eV]',
        'showlegend': True
    }

    def draw_PDOS_lines(self, drawer_info):
        super().draw_PDOS_lines(drawer_info)

        Es = drawer_info["Es"]
        self.update_layout(yaxis_range=[min(Es), max(Es)])


PdosPlot.backends.register("plotly", PlotlyPDOSBackend)
