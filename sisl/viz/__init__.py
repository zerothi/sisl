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

from .plot import Plot, Animation, MultiplePlot, SubPlots
from .plots import *
from .session import Session
from .sessions import *
from .plotutils import load
from ._plotables import register_plotable
from ._plotables_register import *

from .backends import load_backends
load_backends()
