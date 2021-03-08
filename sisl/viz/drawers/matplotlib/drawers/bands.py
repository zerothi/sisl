import numpy as np

from ....plots import BandsPlot
from ..drawer import MatplotlibDrawer

class MatplotlibBandsDrawer(MatplotlibDrawer):

    _ax_defaults = {
        'xlabel': 'K',
        'ylabel': 'Energy [eV]'
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ax.grid(axis="x")
    
    def draw_bands(self, filtered_bands, spin_texture, spin_moments, spin_polarized, bands_color, spindown_color, bands_width, spin, add_band_trace_data):

        # if spin_texture:

        #     def scatter_additions(band, spin_index):

        #         return {
        #             "mode": "markers",
        #             "marker": {"color": spin_moments, "size": bands_width, "showscale": True, "coloraxis": "coloraxis"},
        #             "showlegend": False
        #         }
        # else:

        #     def scatter_additions(band, spin_index):

        #         return {
        #             "mode": "lines",
        #             'line': {"color": , 'width': },
        #         }

        
        for spin_bands, spin in zip(filtered_bands.transpose('spin', 'band', 'k'), filtered_bands.spin.values):
            for band in spin_bands:
                self.ax.plot(band.k.values, band.values, color=[bands_color, spindown_color][spin], linewidth=bands_width)

        #Define the data of the plot as a list of dictionaries {x, y, 'type', 'name'}
        # self.add_traces(np.ravel([[{
        #     'type': 'scatter',
        #     'x': band.k.values,
        #     'y': (band).values,
        #     'mode': 'lines',
        #     'name': "{} spin {}".format(band.band.values, ["up", "down"][spin]) if spin_polarized else str(band.band.values),
        #     **scatter_additions(band.band.values, spin),
        #     'hoverinfo':'name',
        #     "hovertemplate": '%{y:.2f} eV',
        #     **add_band_trace_data(band, self)
        #     } for band in spin_bands] for spin_bands, spin in zip(filtered_bands.transpose('spin', 'band', 'k'), filtered_bands.spin.values)]).tolist())

    def draw_gap(self, ks, Es, color, name, **kwargs):

        self.ax.plot(
            ks, Es, color=color, marker=".", label=f"{name} ({Es[1] - Es[0]:.2f} eV)"
        )

        self.ax.legend()

        # self.add_trace({
        #     'type': 'scatter',
        #     'mode': 'lines+markers+text',
        #     'x': ks,
        #     'y': Es,
        #     'text': [f'Gap: {Es[1] - Es[0]:.3f} eV', ''],
        #     'marker': {'color': color},
        #     'line': {'color': color},
        #     'name': name,
        #     'textposition': 'top right',
        #     **kwargs
        # })

    def after_get_figure(self, plot, Erange, spin, spin_texture_colorscale):
        #Add the ticks
        self.ax.set_xticks(getattr(plot.bands, "ticks", None))
        self.ax.set_xticklabels(getattr(plot.bands, "ticklabels", None))
        self.ax.set_xlim(*plot.bands.k.values[[0, -1]])
        self.ax.set_ylim(*Erange)

        # # If we are showing spin textured bands, customize the colorbar
        # if plot.spin_texture:
        #     self.layout.coloraxis.colorbar = {"title": f"Spin texture ({spin[0]})"}
        #     self.update_layout(coloraxis = {"cmin": -1, "cmax": 1, "colorscale": spin_texture_colorscale})
        pass

    # def get_ipywidget(self):
    #     return self.figure

BandsPlot._drawers.register("matplotlib", MatplotlibBandsDrawer)