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
    
    def _draw_band(self, x, y, *args, **kwargs):
        kwargs["hovertemplate"] = '%{y:.2f} eV (spin moment: %{marker.color:.2f})'
        kwargs["hoverinfo"] = "name"
        return super()._draw_band(x, y, *args, **kwargs)
    
    def _draw_spin_textured_band(self, *args, spin_texture_vals=None, **kwargs):
        kwargs.update({
            "mode": "markers",
            "marker": {"color": spin_texture_vals,  "size": kwargs["line"]["width"], "showscale": True, "coloraxis": "coloraxis"},
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

    def after_get_figure(self, plot, Erange, spin, spin_texture_colorscale):
        # Add the ticks
        tickvals = getattr(plot.bands, "ticks", None)
        # We need to convert tick values to a list, otherwise sometimes plotly fails to display them
        self.figure.layout.xaxis.tickvals = list(tickvals) if tickvals is not None else None
        self.figure.layout.xaxis.ticktext = getattr(plot.bands, "ticklabels", None)
        self.figure.layout.yaxis.range = Erange
        self.figure.layout.xaxis.range = plot.bands.k.values[[0, -1]]

        # If we are showing spin textured bands, customize the colorbar
        if plot.spin_texture:
            self.layout.coloraxis.colorbar = {"title": f"Spin texture ({spin[0]})"}
            self.update_layout(coloraxis = {"cmin": -1, "cmax": 1, "colorscale": spin_texture_colorscale})
    
    def _test_is_gap_drawn(self):
        return len([True for trace in self.figure.data if trace.name == "Gap"]) > 0

BandsPlot._backends.register("plotly", PlotlyBandsBackend)