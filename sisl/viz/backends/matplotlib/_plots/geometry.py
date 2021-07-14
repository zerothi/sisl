from collections.abc import Iterable
import numpy as np

from ....plots import GeometryPlot
from ..backend import MatplotlibBackend
from ...templates import GeometryBackend


class MatplotlibGeometryBackend(MatplotlibBackend, GeometryBackend):

    def draw_1D(self, backend_info, **kwargs):
        super().draw_1D(backend_info, **kwargs)

        xaxis = backend_info["xaxis"]
        yaxis = backend_info["yaxis"]

        self.axes.set_xlabel(f'{("X","Y","Z")[xaxis]} axis [Ang]')
        self.axes.set_ylabel(yaxis)

    def draw_2D(self, backend_info, **kwargs):
        super().draw_2D(backend_info, **kwargs)

        self.axes.set_xlabel(f'{("X","Y", "Z")[backend_info["xaxis"]]} axis [Ang]')
        self.axes.set_ylabel(f'{("X","Y", "Z")[backend_info["yaxis"]]} axis [Ang]')
        self.axes.axis("equal")
    
    def _draw_atoms_2D_scatter(self, *args, **kwargs):
        kwargs["zorder"] = 2.1
        super()._draw_atoms_2D_scatter(*args, **kwargs)

    def draw_3D(self, backend_info):
        return NotImplementedError(f"3D geometry plots are not implemented by {self.__class__.__name__}")

GeometryPlot.backends.register("matplotlib", MatplotlibGeometryBackend)