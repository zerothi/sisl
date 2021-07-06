from abc import abstractmethod
from .bands import BandsBackend

from ....plots import FatbandsPlot

class FatbandsBackend(BandsBackend):

    def draw(self, backend_info):
        """Controls the flow for drawing Fatbands, so that specific backends are easy to implement.

        It draws first all the weights and then the bands.
        """

        groups_weights = backend_info["groups_weights"]
        groups_metadata = backend_info["groups_metadata"]
        existing_bands = backend_info["draw_bands"][0]

        x = backend_info["draw_bands"][0].k.values

        for group_name in groups_weights:
            self.draw_group_weights(
                weights=groups_weights[group_name], metadata=groups_metadata[group_name],
                name=group_name, bands=existing_bands, x=x
            )

        self.draw_bands(*backend_info["draw_bands"])

        self._draw_gaps(backend_info["gaps"])

    def draw_group_weights(self, weights, metadata, name, bands, x):
        """Draws all weights for a group
        
        It will iterate over all the bands that need to be drawn for a certain group
        and ask the backend to draw them. The backend should implement `_draw_band_weights`
        as specified below.
        """

        for ispin, spin_weights in enumerate(weights):
            for i, band_weights in enumerate(spin_weights):
                band_values = bands.sel(band=band_weights.band, spin=band_weights.spin)

                self._draw_band_weights(
                    x=x, y=band_values, weights=band_weights.values, 
                    color=metadata["style"]["line"]["color"], name=name,
                    is_group_first=i==0
                )

    @abstractmethod
    def _draw_band_weights(self, x, y, weights, color, name):
        """Implement this method to draw a fatband"""

FatbandsPlot._backends.register_template(FatbandsBackend)