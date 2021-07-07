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
import plotly

from .backend import PlotlyBackend, PlotlyMultiplePlotBackend, PlotlySubPlotsBackend, PlotlyAnimationBackend
from ._plots import *
from ._templates import *
