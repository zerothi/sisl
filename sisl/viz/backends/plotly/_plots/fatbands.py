from ....plots import FatbandsPlot
from .bands import PlotlyBandsBackend
from ...templates import FatbandsBackend


class PlotlyFatbandsBackend(PlotlyBandsBackend, FatbandsBackend):

    def draw(self, backend_info):
        # We are going to need a trace that goes forward and then back so that
        # it is self-fillable
        xs = backend_info["draw_bands"]["filtered_bands"].k.values
        self._area_xs = [*xs, *reversed(xs)]

        super().draw(backend_info)

        if backend_info["draw_bands"]["spin_texture"]["show"]:
            self.update_layout(legend_orientation="h")

    def _draw_band_weights(self, x, y, weights, name, color, is_group_first):

        self.add_trace({
            "type": "scatter",
            "mode": "lines",
            "x": self._area_xs,
            "y": [*(y + weights), *reversed(y - weights)],
            "line": {"width": 0, "color": color},
            "showlegend": is_group_first,
            "name": name,
            "legendgroup": name,
            "fill": "toself"
        })

FatbandsPlot.backends.register("plotly", PlotlyFatbandsBackend)
