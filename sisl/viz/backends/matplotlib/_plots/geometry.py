# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from collections.abc import Iterable
import numpy as np

from ....plots import GeometryPlot
from ..backend import MatplotlibBackend
from ...templates import GeometryBackend


class MatplotlibGeometryBackend(MatplotlibBackend, GeometryBackend):

    def draw_1D(self, backend_info, **kwargs):
        super().draw_1D(backend_info, **kwargs)

        self.axes.set_xlabel(backend_info["axes_titles"]["xaxis"])
        self.axes.set_ylabel(backend_info["axes_titles"]["yaxis"])

    def draw_2D(self, backend_info, **kwargs):
        super().draw_2D(backend_info, **kwargs)

        self.axes.set_xlabel(backend_info["axes_titles"]["xaxis"])
        self.axes.set_ylabel(backend_info["axes_titles"]["yaxis"])
        self.axes.axis("equal")

    def _draw_atoms_2D_scatter(self, *args, **kwargs):
        kwargs["zorder"] = 2.1
        super()._draw_atoms_2D_scatter(*args, **kwargs)

    def draw_3D(self, backend_info):
        return NotImplementedError(f"3D geometry plots are not implemented by {self.__class__.__name__}")

GeometryPlot.backends.register("matplotlib", MatplotlibGeometryBackend)
