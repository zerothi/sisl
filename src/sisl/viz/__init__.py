# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""
Visualization utilities
=======================

"""

import os

from sisl._environ import register_environ_variable

try:
    _nprocs = len(os.sched_getaffinity(0))
except Exception:
    _nprocs = 1

register_environ_variable("SISL_VIZ_NUM_PROCS", min(1, _nprocs),
                          description="Maximum number of processors used for parallel plotting",
                          process=int)

from . import _xarray_accessor
from ._plotables import register_plotable
from ._plotables_register import *
from .figure import Figure, get_figure
from .plot import Plot
from .plots import *
from .plotters import plot_actions
