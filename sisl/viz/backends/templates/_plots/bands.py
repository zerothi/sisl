from abc import abstractmethod
from ..backend import Backend

from ....plots import BandsPlot

class BandsBackend(Backend):
    
    @abstractmethod
    def draw_bands(self, filtered_bands, spin_texture, spin_moments, spin_polarized, bands_color, spindown_color, bands_width, spin, add_band_trace_data):
        """This method should draw all bands contained in filtered_bands, using the settings correctly.

        If `spin_texture` is True, then it should use `spin_moments` to draw the spin texture.
        """

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