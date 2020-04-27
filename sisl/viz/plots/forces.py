import numpy as np

import os

import sisl
from ..plot import Plot
from ..input_fields import TextInput, SwitchInput, ColorPicker, DropdownInput, IntegerInput, FloatInput, RangeSlider, QueriesInput, ProgramaticInput

class ForcesPlot(Plot):

    _plot_type = "Forces"

    _parameters = (

        TextInput(
            key="out_file", name="Output log file",
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

    def _read_siesta_output(self):

        root_fdf = self.setting("root_fdf")

        out_file = self.setting("out_file") or os.path.splitext(root_fdf)[0] + ".out"

        outSile = self.get_sile(out_file) 
        
        self.atomic_forces = outSile.read_force(all=True)

        if len(self.atomic_forces[0]) == 0:
            self.total_forces, self.max_forces = outSile.read_force(all=True, total=True, max=True)
        else:
            self.total_forces, self.max_forces = outSile.read_force(all=True, total=True, max=True)
            
        self.simulation_ended = outSile.job_completed
    
    def _after_read(self):

        self.update_settings(update_fig=False, xaxis_title="MD step", yaxis_title="Force (eV/Ang)")

    def _set_data(self):

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
