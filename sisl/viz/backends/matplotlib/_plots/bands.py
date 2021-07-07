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
            # Create the normalization for the colorscale of spin_moments.
            self._spin_texture_norm = Normalize(spin_moments.min(), spin_moments.max())
            self._spin_texture_colorscale = spin_texture_colorscale
        
        super().draw_bands(filtered_bands, spin_texture, spin_moments, spin_texture_colorscale, spin_polarized, bands_color, spindown_color, bands_width, spin, add_band_trace_data)

        if spin_texture:
            # Add the colorbar for spin texture.
            self.figure.colorbar(self._colorbar)

    
    def _draw_spin_textured_band(self, x, y, spin_texture_vals=None, **kwargs):
        # This is heavily based on 
        # https://matplotlib.org/stable/gallery/lines_bars_and_markers/multicolored_line.html

        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        lc = LineCollection(segments, cmap=self._spin_texture_colorscale, norm=self._spin_texture_norm)
        
        # Set the values used for colormapping
        lc.set_array(spin_texture_vals)
        lc.set_linewidth(kwargs["line"].get("width", 1))
        self._colorbar = self.ax.add_collection(lc)
                    

    def draw_gap(self, ks, Es, color, name, **kwargs):

        name = f"{name} ({Es[1] - Es[0]:.2f} eV)"
        gap = self.ax.plot(
            ks, Es, color=color, marker=".", label=name
        )

        self.ax.legend(gap, [name])

    def after_get_figure(self, plot, Erange, spin, spin_texture_colorscale):
        #Add the ticks
        self.ax.set_xticks(getattr(plot.bands, "ticks", None))
        self.ax.set_xticklabels(getattr(plot.bands, "ticklabels", None))
        self.ax.set_xlim(*plot.bands.k.values[[0, -1]])
        self.ax.set_ylim(*Erange) 
    
    def _test_is_gap_drawn(self):
        return self.ax.lines[-1].get_label().startswith("Gap")

BandsPlot.backends.register("matplotlib", MatplotlibBandsBackend)