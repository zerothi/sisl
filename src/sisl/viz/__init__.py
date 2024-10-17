# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

"""
Visualization utilities
=======================

"""

try:
    import nodify as _  # noqa: F401
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        """\
sisl.viz requires additional packages.
Install them with pip:
   pip install sisl[viz]
Or conda (only possible if inside a conda environment):
   conda install nodify plotly netCDF4 scikit-image pathos
"""
    ) from e

# Placeholders for 'plot' attributes are set in the classes while
# sisl.viz is not loaded. Now we are loading it, so just remove those
# placeholders.
from sisl._lazy_viz import clear_viz_placeholders

clear_viz_placeholders()

del clear_viz_placeholders

from . import _xarray_accessor
from ._plotables import register_plotable
from ._plotables_register import *
from .figure import Figure, get_figure
from .plot import Plot
from .plots import *
from .plotters import plot_actions
