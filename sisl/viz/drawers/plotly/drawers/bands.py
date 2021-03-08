import numpy as np

from ....plots import BandsPlot
from ..drawer import PlotlyDrawer

class PlotlyBandsDrawer(PlotlyDrawer):

    _layout_defaults = {
        'xaxis_title': 'K',
        'xaxis_mirror': True,
        'yaxis_mirror': True,
        'xaxis_showgrid': True,
        'yaxis_title': 'Energy [eV]'
    }
    
    def draw_bands(self, filtered_bands, spin_texture, spin_moments, spin_polarized, bands_color, spindown_color, bands_width, spin, add_band_trace_data):

        if not callable(add_band_trace_data):
            add_band_trace_data = lambda band, plot: {}

        if spin_texture:

            def scatter_additions(band, spin_index):

                return {
                    "mode": "markers",
                    "marker": {"color": spin_moments, "size": bands_width, "showscale": True, "coloraxis": "coloraxis"},
                    "showlegend": False
                }
        else:

            def scatter_additions(band, spin_index):

                return {
                    "mode": "lines",
                    'line': {"color": [bands_color, spindown_color][spin_index], 'width': bands_width},
                }

        # Define the data of the plot as a list of dictionaries {x, y, 'type', 'name'}
        self.add_traces(np.ravel([[{
            'type': 'scatter',
            'x': band.k.values,
            'y': (band).values,
            'mode': 'lines',
            'name': "{} spin {}".format(band.band.values, ["up", "down"][spin]) if spin_polarized else str(band.band.values),
            **scatter_additions(band.band.values, spin),
            'hoverinfo':'name',
            "hovertemplate": '%{y:.2f} eV',
            **add_band_trace_data(band, self)
            } for band in spin_bands] for spin_bands, spin in zip(filtered_bands.transpose('spin', 'band', 'k'), filtered_bands.spin.values)]).tolist())

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
        self.figure.layout.xaxis.tickvals = getattr(plot.bands, "ticks", None)
        self.figure.layout.xaxis.ticktext = getattr(plot.bands, "ticklabels", None)
        self.figure.layout.yaxis.range = Erange
        self.figure.layout.xaxis.range = plot.bands.k.values[[0, -1]]

        # If we are showing spin textured bands, customize the colorbar
        if plot.spin_texture:
            self.layout.coloraxis.colorbar = {"title": f"Spin texture ({spin[0]})"}
            self.update_layout(coloraxis = {"cmin": -1, "cmax": 1, "colorscale": spin_texture_colorscale})

BandsPlot._drawers.register("plotly", PlotlyBandsDrawer)