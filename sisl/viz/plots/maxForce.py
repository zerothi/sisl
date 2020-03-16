import numpy as np
import xarray as xr
import itertools

import os
import shutil

import sisl
from ..plot import Plot, MultiplePlot, Animation, PLOTS_CONSTANTS
from ..plotutils import sortOrbitals, initMultiplePlots, copyParams, findFiles, runMultiple, calculateGap
from ..inputFields import InputField, TextInput, SwitchInput, ColorPicker, DropdownInput, IntegerInput, FloatInput, RangeSlider, QueriesInput, ProgramaticInput

class MaxForcePlot(Plot):

    _plotType = "Max Force"

    def _readSiesOut(self):

        rootFdf = self.setting("rootFdf")

        outFile = os.path.splitext(rootFdf)[0] + ".out"

        outSile = sisl.get_sile(outFile) 

        self.forces = outSile.read_force(all=True)

        self.simulation_ended = outSile.job_completed

        if outSile.job_completed:
            self.updateSettings(paper_bgcolor="lightgreen", plot_bgcolor="lightgreen" ,updateFig=False)

        return [outFile]

    def _setData(self):

        color = "green" if self.simulation_ended else "darkred" 

        self.data = [{
            'type': 'scatter',
            'y': [ np.max(abs(MDstep)) for MDstep in self.forces],
            'marker': {'color': color},
            'line': {"color": color }
        }]
