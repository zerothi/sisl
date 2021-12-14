# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import numpy as np

from ....plots import FatbandsPlot
from .bands import PlotlyBandsBackend
from ...templates import FatbandsBackend


class PlotlyFatbandsBackend(PlotlyBandsBackend, FatbandsBackend):

    def draw(self, backend_info):
        super().draw(backend_info)

        if backend_info["draw_bands"]["spin_texture"]["show"]:
            self.update_layout(legend_orientation="h")

    def _draw_band_weights(self, x, y, weights, name, color, is_group_first):

        for i_chunk, chunk in enumerate(self._yield_band_chunks(x, y, weights)):

            # Removing the parts of the band where y is nan handles bands that
            # flow outside the plot.
            chunk_x, chunk_y, chunk_weights = chunk[:, ~np.isnan(chunk[1])]

            self.add_trace({
                "type": "scatter",
                "mode": "lines",
                "x": [*chunk_x, *reversed(chunk_x)],
                "y": [*(chunk_y + chunk_weights), *reversed(chunk_y - chunk_weights)],
                "line": {"width": 0, "color": color},
                "showlegend": is_group_first and i_chunk == 0,
                "name": name,
                "legendgroup": name,
                "fill": "toself"
            })

FatbandsPlot.backends.register("plotly", PlotlyFatbandsBackend)
