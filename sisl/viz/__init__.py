"""
====================================
Visualization utilities (:mod:`sisl.viz`)
====================================

.. module:: sisl.viz
   :noindex:

Classes that are used for visualizing simulations results

Range routines
==============

Miscellaneous routines
======================

"""

from .plot import Plot, Animation, MultiplePlot, SubPlots
from .plots import *
from .configurable import *
from .session import Session
from .sessions import *
from .plotutils import load
from ._plotables import register_plotable
from ._presets import get_preset, add_presets
from ._user_customs import import_user_plots, import_user_presets, import_user_sessions, import_user_plugins

import sisl.viz._templates

user_plots = import_user_plots()
user_presets = import_user_presets()
user_sessions = import_user_sessions()
user_plugins = import_user_plugins()
