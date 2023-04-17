# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
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
