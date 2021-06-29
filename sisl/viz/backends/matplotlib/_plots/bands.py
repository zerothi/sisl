from ....plots import BandsPlot
from ..backend import MatplotlibBackend
from ...templates import BandsBackend

import numpy as np
from matplotlib.pyplot import Normalize
from matplotlib.collections import LineCollection

class MatplotlibBandsBackend(MatplotlibBackend, BandsBackend):

    _ax_defaults = {
        'xlabel': 'K',
        'ylabel': 'Energy [eV]'
    }

    def _init_ax(self):
        super()._init_ax()
        self.ax.grid(axis="x")
    
    def draw_bands(self, filtered_bands, spin_texture, spin_moments, spin_texture_colorscale, spin_polarized, bands_color, spindown_color, bands_width, spin, add_band_trace_data):

        if spin_texture:
            spin_texture_norm = Normalize(spin_moments.min(), spin_moments.max())

        for spin_bands, spin in zip(filtered_bands.transpose('spin', 'band', 'k'), filtered_bands.spin.values):
            for band in spin_bands:
                x = band.k.values
                y = band.values

                if spin_texture:
                    color = spin_moments.sel(band=band.band.values).values

                    # Create a set of line segments so that we can color them individually
                    # This creates the points as a N x 1 x 2 array so that we can stack points
                    # together easily to get the segments. The segments array for line collection
                    # needs to be (numlines) x (points per line) x 2 (for x and y)
                    points = np.array([x, y]).T.reshape(-1, 1, 2)
                    segments = np.concatenate([points[:-1], points[1:]], axis=1)
                    
                    lc = LineCollection(segments, cmap=spin_texture_colorscale, norm=spin_texture_norm)
                    # Set the values used for colormapping
                    lc.set_array(color)
                    lc.set_linewidth(bands_width)
                    line = self.ax.add_collection(lc)
                else:
                    self.ax.plot(x, y, color=[bands_color, spindown_color][spin], linewidth=bands_width)

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
    
    def _test_is_gap_drawn(self):
        return self.ax.lines[-1].get_label().startswith("Gap")

BandsPlot._backends.register("matplotlib", MatplotlibBandsBackend)