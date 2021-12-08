# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from ..backend import Backend

from ....plots.grid import GridPlot


class GridBackend(Backend):
    """Draws a grid as provided by `GridPlot`.

    Checks the dimensionality of the grid and then calls:
        - 1D case: `self.draw_1D`, generic implementation that uses `self.draw_line`
        - 2D case: `self.draw_2D`, NOT IMPLEMENTED (optional)
        - 3D case: `self.draw_3D`, NOT IMPLEMENTED (optional)

    Then, if the geometry needs to be plotted, it plots the geometry. This will use
    the `GeometryBackend` with the same name as your grid backend, so make sure it is implemented
    if you want to allow showing geometries along with the grid.
    """

    def draw(self, backend_info):
        # Choose which function we need to use to plot
        drawing_func = getattr(self, f"draw_{backend_info['ndim']}D")

        drawing_func(backend_info)

        if backend_info["geom_plot"] is not None:
            self.draw_other_plot(backend_info["geom_plot"])

    def draw_1D(self, backend_info, **kwargs):
        """Draws the grid in 1D"""
        self.draw_line(backend_info["ax_range"], backend_info["values"], name=backend_info["name"], **kwargs)

    def draw_2D(self, backend_info, **kwargs):
        """Should draw the grid in 2D, and draw contours if requested."""
        raise NotImplementedError(f"{self.__class__.__name__} does not implement displaying grids in 2D")

    def draw_3D(self, backend_info, **kwargs):
        """Should draw all the isosurfaces of the grid in 3D"""
        raise NotImplementedError(f"{self.__class__.__name__} does not implement displaying grids in 3D")

GridPlot.backends.register_template(GridBackend)
