# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

from .figure import BACKENDS, Figure, get_figure


class NotAvailableFigure(Figure):
    _package: str = ""

    def __init__(self, *args, **kwargs):
        raise ModuleNotFoundError(
            f"{self.__class__.__name__} is not available because {self._package} is not installed."
        )


try:
    import plotly  # noqa: F401
except ModuleNotFoundError:

    class PlotlyFigure(NotAvailableFigure):
        _package = "plotly"

else:
    from .plotly import PlotlyFigure

try:
    import matplotlib  # noqa: F401
except ModuleNotFoundError:

    class MatplotlibFigure(NotAvailableFigure):
        _package = "matplotlib"

else:
    from .matplotlib import MatplotlibFigure

try:
    import py3Dmol  # noqa: F401
except ModuleNotFoundError:

    class Py3DmolFigure(NotAvailableFigure):
        _package = "py3Dmol"

else:
    from .py3dmol import Py3DmolFigure

try:
    import bpy  # noqa: F401
except ModuleNotFoundError:

    class BlenderFigure(NotAvailableFigure):
        _package = "blender (bpy)"

else:
    from .blender import BlenderFigure


BACKENDS["plotly"] = PlotlyFigure
BACKENDS["matplotlib"] = MatplotlibFigure
BACKENDS["py3dmol"] = Py3DmolFigure
BACKENDS["blender"] = BlenderFigure
