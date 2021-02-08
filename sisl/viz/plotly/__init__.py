r"""Plotly
==========

Plotly is a backend that provides expert plotting utilities using `plotly`.
It features a rich set of settings enabling fine-tuning of many parameters.

   GeometryPlot
   BandsPlot
   FatbandsPlot
   PdosPlot
   BondLengthMap
   ForcesPlot
   GridPlot
   WavefunctionPlot

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

# And then import user customs (we need to do it here
# to allow the user importing sisl.viz.plotly.Plot, for example)
user_plots = import_user_plots()
user_presets = import_user_presets()
user_sessions = import_user_sessions()
user_plugins = import_user_plugins()
