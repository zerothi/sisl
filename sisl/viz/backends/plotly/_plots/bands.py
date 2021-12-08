# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import numpy as np

from ....plots import BandsPlot
from ..backend import PlotlyBackend
from ...templates import BandsBackend


class PlotlyBandsBackend(PlotlyBackend, BandsBackend):

    _layout_defaults = {
        'xaxis_title': 'K',
        'xaxis_mirror': True,
        'yaxis_mirror': True,
        'xaxis_showgrid': True,
        'yaxis_title': 'Energy [eV]'
    }

    def draw_bands(self, filtered_bands, spin_texture, **kwargs):
        super().draw_bands(filtered_bands=filtered_bands, spin_texture=spin_texture, **kwargs)

        # Add the ticks
        tickvals = getattr(filtered_bands, "ticks", None)
        # We need to convert tick values to a list, otherwise sometimes plotly fails to display them
        self.figure.layout.xaxis.tickvals = list(tickvals) if tickvals is not None else None
        self.figure.layout.xaxis.ticktext = getattr(filtered_bands, "ticklabels", None)
        self.figure.layout.yaxis.range = [filtered_bands.min(), filtered_bands.max()]
        self.figure.layout.xaxis.range = filtered_bands.k.values[[0, -1]]

        # If we are showing spin textured bands, customize the colorbar
        if spin_texture["show"]:
            self.layout.coloraxis.colorbar = {"title": f"Spin texture ({str(spin_texture['values'].axis.item())})"}
            self.update_layout(coloraxis = {"cmin": spin_texture["values"].min().item(), "cmax": spin_texture["values"].max().item(), "colorscale": spin_texture["colorscale"]})

    def _draw_band(self, x, y, *args, **kwargs):
        kwargs = {
            "hovertemplate": '%{y:.2f} eV',
            "hoverinfo": "name",
            **kwargs
        }
        return super()._draw_band(x, y, *args, **kwargs)

    def _draw_spin_textured_band(self, *args, spin_texture_vals=None, **kwargs):
        kwargs.update({
            "mode": "markers",
            "marker": {"color": spin_texture_vals, "size": kwargs["line"]["width"], "showscale": True, "coloraxis": "coloraxis"},
            "hovertemplate": '%{y:.2f} eV (spin moment: %{marker.color:.2f})',
            "showlegend": False
        })
        return self._draw_band(*args, **kwargs)

    def draw_gap(self, ks, Es, color, name, **kwargs):

        self.add_trace({
            'type': 'scatter',
            'mode': 'lines+markers+text',
            'x': ks,
            'y': Es,
            'text': [f'Gap: {Es[1] - Es[0]:.3f} eV', ''],
            'marker': {'color': color},
            'line': {'color': color},
            'name': name,
            'textposition': 'top right',
            **kwargs
        })

    def _test_is_gap_drawn(self):
        return len([True for trace in self.figure.data if trace.name == "Gap"]) > 0

BandsPlot.backends.register("plotly", PlotlyBandsBackend)
