# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from ....plots import BandsPlot
from ..backend import MatplotlibBackend
from ...templates import BandsBackend

import numpy as np
from matplotlib.pyplot import Normalize
from matplotlib.collections import LineCollection


class MatplotlibBandsBackend(MatplotlibBackend, BandsBackend):

    _axes_defaults = {
        'xlabel': 'K',
        'ylabel': 'Energy [eV]'
    }

    def _init_ax(self):
        super()._init_ax()
        self.axes.grid(axis="x")

    def draw_bands(self, filtered_bands, spin_texture, **kwargs):

        if spin_texture["show"]:
            # Create the normalization for the colorscale of spin_moments.
            self._spin_texture_norm = Normalize(spin_texture["values"].min(), spin_texture["values"].max())
            self._spin_texture_colorscale = spin_texture["colorscale"]

        super().draw_bands(filtered_bands=filtered_bands, spin_texture=spin_texture, **kwargs)

        if spin_texture["show"]:
            # Add the colorbar for spin texture.
            self.figure.colorbar(self._colorbar)

        # Add the ticks
        tick_vals = getattr(filtered_bands, "ticks", None)
        if tick_vals is not None:
            self.axes.set_xticks(tick_vals)
        tick_labels = getattr(filtered_bands, "ticklabels", None)
        if tick_labels is not None:
            self.axes.set_xticklabels(tick_labels)
        # Set the limits
        self.axes.set_xlim(*filtered_bands.k.values[[0, -1]])
        self.axes.set_ylim(filtered_bands.min(), filtered_bands.max())

    def _draw_spin_textured_band(self, x, y, spin_texture_vals=None, **kwargs):
        # This is heavily based on
        # https://matplotlib.org/stable/gallery/lines_bars_and_markers/multicolored_line.html

        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        lc = LineCollection(segments, cmap=self._spin_texture_colorscale, norm=self._spin_texture_norm)

        # Set the values used for colormapping
        lc.set_array(spin_texture_vals)
        lc.set_linewidth(kwargs["line"].get("width", 1))
        self._colorbar = self.axes.add_collection(lc)

    def draw_gap(self, ks, Es, color, name, **kwargs):

        name = f"{name} ({Es[1] - Es[0]:.2f} eV)"
        gap = self.axes.plot(
            ks, Es, color=color, marker=".", label=name
        )

        self.axes.legend(gap, [name])

    def _test_is_gap_drawn(self):
        return self.axes.lines[-1].get_label().startswith("Gap")

BandsPlot.backends.register("matplotlib", MatplotlibBandsBackend)
