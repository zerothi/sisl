# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import matplotlib.pyplot as plt

from ....plots.grid import GridPlot
from ..backend import MatplotlibBackend
from ...templates import GridBackend


class MatplotlibGridBackend(MatplotlibBackend, GridBackend):

    def draw_1D(self, backend_info, **kwargs):
        super().draw_1D(backend_info, **kwargs)

        self.axes.set_xlabel(backend_info["axes_titles"]["xaxis"])
        self.axes.set_ylabel(backend_info["axes_titles"]["yaxis"])

    def draw_2D(self, backend_info, **kwargs):

        # Define the axes values
        x = backend_info["x"]
        y = backend_info["y"]

        extent = [x[0], x[-1], y[0], y[-1]]

        # Draw the values of the grid
        self.axes.imshow(
            backend_info["values"], vmin=backend_info["cmin"], vmax=backend_info["cmax"],
            label=backend_info["name"], cmap=backend_info["colorscale"], extent=extent,
            origin="lower"
        )

        # Draw the isocontours
        for contour in backend_info["contours"]:
            self.axes.plot(
                contour["x"], contour["y"],
                color=contour["color"],
                alpha=contour["opacity"],
                label=contour["name"]
            )

        self.axes.set_xlabel(backend_info["axes_titles"]["xaxis"])
        self.axes.set_ylabel(backend_info["axes_titles"]["yaxis"])

    def draw_3D(self, backend_info, **kwargs):
        # This will basically raise the NotImplementedError
        super().draw_3D(backend_info, **kwargs)

        # The following code is just here as reference of how this MIGHT
        # be done in matplotlib.
        self.figure = plt.figure()
        self.axes = self.figure.add_subplot(projection="3d")

        for isosurf in backend_info["isosurfaces"]:

            x, y, z = isosurf["vertices"].T
            I, J, K = isosurf["faces"].T

            self.axes.plot_trisurf(x, y, z, linewidth=0, antialiased=True)

GridPlot.backends.register("matplotlib", MatplotlibGridBackend)
