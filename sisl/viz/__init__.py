# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""
Visualization utilities
=======================

Various visualization modules are described here.


Plotly
======

The plotly backend.
"""
# from ._presets import *
# from ._templates import *
# from ._user_customs import import_user_plots, import_user_presets, import_user_sessions, import_user_plugins
import os
from sisl._environ import register_environ_variable

try:
    _nprocs = len(os.sched_getaffinity(0))
except:
    _nprocs = 1

register_environ_variable("SISL_VIZ_NUM_PROCS", min(1, _nprocs),
                          description="Maximum number of processors used for parallel plotting",
                          process=int)

from .plot import Plot, Animation, MultiplePlot, SubPlots
from .plots import *
from .session import Session
from .sessions import *
from .plotutils import load
from ._plotables import register_plotable
from ._plotables_register import *

from .backends import load_backends
load_backends()
