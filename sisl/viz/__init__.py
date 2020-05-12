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

from .plot import Plot, Animation, MultiplePlot
from .plots import *
from .configurable import *
from .session import Session
from .sessions import *
from .plotutils import load
from ._plotables import register_plotable_sile
from ._presets import get_preset

import sisl.viz._templates