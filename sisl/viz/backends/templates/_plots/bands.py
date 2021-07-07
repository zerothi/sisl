from abc import abstractmethod
from ..backend import Backend

from ....plots import BandsPlot

class BandsBackend(Backend):
    """Draws the bands provided by a `BandsPlot`

    The workflow implemented by it is as follows:
        First, `self.draw_bands` draws all bands like:
            for band in bands:
                if (spin texture needs to be drawn):
                    `self._draw_spin_textured_band()`, NO GENERIC IMPLEMENTATION (optional) 
                else:
                    `self._draw_band()`, generic implementation that calls `self._draw_line`
        Once all bands are drawn, `self.draw_gaps` loops through all the gaps to be drawn:
            for gap in gaps:
                `self.draw_gap()`, MUST BE IMPLEMENTED!
    """

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
        """Iterates over all gaps to draw them"""
        for gap_info in gaps_info:
            self.draw_gap(**gap_info)

    @abstractmethod
    def draw_gap(self, ks, Es, color, name, **kwargs):
        """This method should draw a gap, given the k and E coordinates.

        The color of the line should be determined by `color`, and `name` should be used for labeling.

        Parameters
        -----------
        ks: numpy array of shape (2,)
            The two k coordinates of the gap.
        Es: numpy array of shape (2,)
            The two E coordinates of the gap, sorted from minor to major.
        color: str
            Color with which the gap should be drawn.
        name: str
            Label that should be asigned to the gap.
        """

    # Methods needed for testing

    def _test_is_gap_drawn(self):
        """
        Should return `True` if the gap is currently drawn, otherwise `False`.
        """
        raise NotImplementedError

BandsPlot.backends.register_template(BandsBackend)