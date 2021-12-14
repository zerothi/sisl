# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import numpy as np

from .bands import BandsBackend

from ....plots import FatbandsPlot


class FatbandsBackend(BandsBackend):
    """Draws fatbands provided by `FatbandsPlot`

    The flow implemented by it is as follows:
        First, it draws all the weights, iterating through all the requests:
            for weight_request in weight_requests:
                Call `self.draw_group_weights`, which loops through all the bands to be drawn:
                for band in bands:
                    `self._draw_band_weights`
        Then it just calls the `draw` method of `BandsBackend`, which takes care of the rest. See
        the documentation of `BandsBackend` to understand the rest of the workflow.

    """

    def draw(self, backend_info):
        """Controls the flow for drawing Fatbands.

        It draws first all the weights and then the bands.
        """

        groups_weights = backend_info["groups_weights"]
        groups_metadata = backend_info["groups_metadata"]
        filtered_bands = backend_info["draw_bands"]["filtered_bands"]

        x = filtered_bands.k.values

        for group_name in groups_weights:
            self.draw_group_weights(
                weights=groups_weights[group_name], metadata=groups_metadata[group_name],
                name=group_name, bands=filtered_bands, x=x
            )

        super().draw(backend_info)

    def draw_group_weights(self, weights, metadata, name, bands, x):
        """Draws all weights for a group

        It will iterate over all the bands that need to be drawn for a certain group
        and ask the backend to draw them. The backend should implement `_draw_band_weights`
        as specified below.

        Parameters
        -----------
        weights: xarray.DataArray with indices (spin, band, k)
            Contains all the weights to be drawn.
        metadata: dict
            Contains extra data specifying how the contributions of a group must be drawn.
        name: str
            The name of the group to which the weights correspond
        bands: xarray.DataArray with indices (spin, band, k)
            Contains all the eigenvalues of the band structure.
        """
        if "spin" not in bands.coords:
            bands = bands.expand_dims("spin")

        # Find where the discontinuities are
        self._discontinuities = np.where(np.isnan(x))[0]

        # Loop over spin
        for ispin, spin_weights in enumerate(weights.transpose("spin", "band", "k")):
            # Loop over bands
            for i, band_weights in enumerate(spin_weights):
                # For each band, draw the fatband
                band_values = bands.sel(band=band_weights.band, spin=ispin)

                self._draw_band_weights(
                    x=x, y=band_values, weights=band_weights.values,
                    color=metadata["style"]["line"]["color"], name=name,
                    is_group_first=i==0 and ispin == 0
                )

    def _draw_band_weights(self, x, y, weights, color, name, is_group_first):
        """Default implementation to draw a fatband.

        It uses a scatter plot, where the size of each scatter point is proportional
        to the weight.

        Parameters
        -----------
        x: np.ndarray of shape (nk,)
            Contains the k coordinates of the band
        y: np.ndarray of shape (nk,)
            Contains the energy values of the band
        weights: np.ndarray of shape (nk,)
            Contains the weight values for each k coordinate
        color: str
            The color with which the contribution must be drawn.
        name: str
            The name of the group to which these band weights correspond
        is_group_first: bool
            Whether this is the first fatband plotted for a given request. This might
            be useful for grouping items drawn, for example.
        """
        size = weights
        size[np.isnan(size)] = 0
        self.draw_scatter(x, y, name=name, marker={"color": color, "size": weights})

    def _yield_band_chunks(self, *arrays):
        """For backends that can not handle fatbands discontinuities
        out of the box, this method can be used to yield continuous chunks
        to be drawn.

        Parameters
        -----------
        *arrays:
            All the arrays that must be divided into chunks.
            They must all have the same datatype.

        Yields
        -----------
        np.ndarray of shape (chunk_nk, n_arrays)
            A continous chunk of data of the potentially discontinuous band.
        """
        # If there are discontinuities, we need to split
        chunks = np.split(np.array([*arrays]).T, self._discontinuities)

        for i_chunk, chunk in enumerate(chunks):
            # Chunks other than the first one begin with np.nan (which was the signal for a discontinuity)
            if i_chunk > 0:
                chunk = chunk[1:]

            yield chunk.T

FatbandsPlot.backends.register_template(FatbandsBackend)
