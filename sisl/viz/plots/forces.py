import numpy as np
import xarray as xr
import itertools

import os
import shutil

import sisl
from ..plot import Plot, MultiplePlot, Animation, PLOTS_CONSTANTS
from ..plotutils import sortOrbitals, initMultiplePlots, copyParams, findFiles, runMultiple, calculateGap
from ..inputFields import TextInput, SwitchInput, ColorPicker, DropdownInput, IntegerInput, FloatInput, RangeSlider, QueriesInput, ProgramaticInput

class ForcesPlot(Plot):

    _plotType = "Forces"

    _parameters = (

        TextInput(
            key="outFile", name="Output log file",
            default = None,
            group="readdata",
            params = {
                "placeholder": "Write the path to your output file here..."
            },
            help = "The path to the output file that contains logs where forces are written."
        ),

        DropdownInput(
            key="type", name="Type of forces",
            default='max',
            params={
                'options': [
                    {'label': 'Atomic', 'value': 'atomic'},
                    {'label': 'Maximum', 'value': 'max'},
                    {'label': 'Cell total', 'value': 'total'},
                ],
                'isMulti': False,
                'isClearable': False,
                'isSearchable': True
            },
            help="The type of forces that should be displayed."
        ),

        SwitchInput(
            key = "all", name = "All steps",
            default = True,
            params = {
                "offLabel": "No",
                "onLabel": "Yes",
            },
            help = "Whether forces for all steps should be displayed."
        ),

        ColorPicker(
            key = "linecolor", name = "Line color",
            default = "black",
            help = "Color of the line that displays the forces"
        ),

    )

    def _readSiesOut(self):

        rootFdf = self.setting("rootFdf")

        outFile = self.setting("outFile") or os.path.splitext(rootFdf)[0] + ".out"

        outSile = sisl.get_sile(outFile) 
        
        self.atomic_forces = outSile.read_force(all=True)

        print(self.atomic_forces)
        if len(self.atomic_forces[0]) == 0:
            self.total_forces, self.max_forces = outSile.read_force(all=True, total=True, max=True)
        else:
            self.total_forces, self.max_forces = outSile.read_force(all=True, total=True, max=True)
            
        self.simulation_ended = outSile.job_completed

        return [outFile]
    
    def _afterRead(self):

        self.updateSettings(updateFig=False, xaxis_title="MD step", yaxis_title="Force (eV/Ang)")

    def _setData(self):

        forces_type = self.setting('type')

        if forces_type == 'total':
            self.data = [{
                'type': 'scatter',
                'mode': 'lines+markers',
                'y': axis_forces,
                'name': f'{axis_name} axis',
            } for axis_forces, axis_name in zip(self.total_forces.T, ("X", "Y", "Z"))]

        elif forces_type == 'max':
            self.data = [{
                'type': 'scatter',
                'mode': 'lines+markers',
                'y': self.max_forces,
                'marker': {'color': self.setting("linecolor")},
                'line': {"color": self.setting("linecolor")},
                'name': 'Max force'
            }]
