# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

"""
Visualization utilities
=======================

"""

import os

try:
    import nodify as _
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
from sisl._environ import register_environ_variable
from sisl._lazy_viz import clear_viz_placeholders

register_environ_variable(
    "SISL_VIZ_NUM_PROCS",
    1,
    description="Maximum number of processors used for parallel plotting",
    process=int,
)

clear_viz_placeholders()

del os, register_environ_variable, clear_viz_placeholders

from . import _xarray_accessor
from ._plotables import register_plotable
from ._plotables_register import *
from .figure import Figure, get_figure
from .plot import Plot
from .plots import *
from .plotters import plot_actions
