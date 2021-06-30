import matplotlib.pyplot as plt

from ....plots import GridPlot
from ..backend import MatplotlibBackend
from ...templates import GridBackend


class MatplotlibGridBackend(MatplotlibBackend, GridBackend):

    def draw_1D(self, backend_info, **kwargs):

        self.ax.plot(backend_info["ax_range"], backend_info["values"], label=backend_info["name"], **kwargs)

        self.ax.set_xlabel(f'{("X","Y", "Z")[backend_info["ax"]]} axis [Ang]')
        self.ax.set_ylabel('Values')

    def draw_2D(self, backend_info, **kwargs):

        # Define the axes values
        x = backend_info["x"]
        y = backend_info["y"]

        extent = [x[0], x[-1], y[0], y[-1]]

        # Draw the values of the grid
        self.ax.imshow(
            backend_info["values"], vmin=backend_info["cmin"], vmax=backend_info["cmax"],
            label=backend_info["name"], cmap=backend_info["colorscale"], extent=extent,
            origin="lower"
        )

        # Draw the isocontours
        for contour in backend_info["contours"]:
            self.ax.plot(
                contour["x"], contour["y"],
                color=contour["color"],
                alpha=contour["opacity"],
                label=contour["name"]
            )

        self.ax.set_xlabel(f'{("X","Y", "Z")[backend_info["xaxis"]]} axis [Ang]')
        self.ax.set_ylabel(f'{("X","Y", "Z")[backend_info["yaxis"]]} axis [Ang]')

    def draw_3D(self, backend_info, **kwargs):
        # This will basically raise the NotImplementedError
        super().draw3D(backend_info, **kwargs)

        # The following code is just here as reference of how this MIGHT
        # be done in matplotlib.
        self.figure = plt.figure()
        self.ax = self.figure.add_subplot(projection="3d")

        for isosurf in backend_info["isosurfaces"]:

            x, y, z = isosurf["vertices"].T
            I, J, K = isosurf["faces"].T

            self.ax.plot_trisurf(x, y, z, linewidth=0, antialiased=True)

GridPlot._backends.register("matplotlib", MatplotlibGridBackend)