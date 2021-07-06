from abc import abstractmethod
from ..backend import Backend

from ....plots import BandsPlot

class BandsBackend(Backend):

    def draw(self, backend_info):
        self.draw_bands(*backend_info["draw_bands"])

        self._draw_gaps(backend_info["gaps"])

    def draw_bands(self, filtered_bands, spin_texture, spin_moments, spin_texture_colorscale, spin_polarized, bands_color, spindown_color, bands_width, spin, add_band_trace_data):

        if spin_texture:
            draw_band_func = self._draw_spin_textured_band
        else:
            draw_band_func = self._draw_band

        if not callable(add_band_trace_data):
            add_band_trace_data = lambda band, drawer: {}

        # Now loop through all bands to draw them
        for spin_bands, spin in zip(filtered_bands.transpose('spin', 'band', 'k'), filtered_bands.spin.values):
            for band in spin_bands:
                # Get the xy values for the band
                x = band.k.values
                y = band.values
                kwargs = {
                    "name": "{} spin {}".format(band.band.values, ["up", "down"][spin]) if spin_polarized else str(band.band.values),
                    "line": {"color": [bands_color, spindown_color][spin], "width": bands_width},
                    **add_band_trace_data(band, self)
                }

                # And plot it differently depending on whether we need to display spin texture or not.
                if not spin_texture:
                    draw_band_func(x, y, **kwargs)
                else:
                    spin_texture_vals = spin_moments.sel(band=band.band.values).values
                    draw_band_func(x, y, spin_texture_vals=spin_texture_vals, **kwargs)
    
    def _draw_band(self, *args, **kwargs):
        return self.draw_line(*args, **kwargs)

    def _draw_spin_textured_band(self, *args, **kwargs):
        return NotImplementedError(f"{self.__class__.__name__} doesn't implement plotting spin_textured bands.")

    def _draw_gaps(self, gaps_info):
        for gap_info in gaps_info:
            self.draw_gap(**gap_info)

    @abstractmethod
    def draw_gap(self, ks, Es, color, name, **kwargs):
        """This method should draw a gap, given the k and E coordinates.

        The color of the line should be determined by `color`, and `name` should be used for labeling.
        """

    # Methods needed for testing

    def _test_is_gap_drawn(self):
        """
        Should return `True` if the gap is currently drawn, otherwise `False`.
        """
        raise NotImplementedError

BandsPlot._backends.register_template(BandsBackend)