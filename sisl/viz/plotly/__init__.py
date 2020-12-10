"""
====================================
Plotly utilities (:mod:`sisl.viz.plotly`)
====================================

.. module:: sisl.viz.plotly
   :noindex:

Classes that are used for visualizing simulations results

Range routines
==============

Miscellaneous routines
======================

"""
from ._presets import *
from ._templates import *
from ._user_customs import import_user_plots, import_user_presets, import_user_sessions, import_user_plugins

from .plot import Plot, Animation, MultiplePlot, SubPlots
from .plots import *
from .session import Session
from .sessions import *
from .plotutils import load
from ._plotables import register_plotable

from ._express import sx as express
