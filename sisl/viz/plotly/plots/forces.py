import numpy as np

import sisl
from ..plot import Plot, entry_point
from ..input_fields import TextInput, SileInput, SwitchInput, ColorPicker, DropdownInput, \
    IntegerInput, FloatInput, RangeSlider, QueriesInput, ProgramaticInput


class ForcesPlot(Plot):
    """
    Display of atomic forces.

    Parameters
    -------------
    out_file: outSileSiesta, optional
        The path to the output file that contains logs where forces are
        written.
    type:  optional
        The type of forces that should be displayed.
    all: bool, optional
        Whether forces for all steps should be displayed.
    linecolor: str, optional
        Color of the line that displays the forces
    root_fdf: fdfSileSiesta, optional
        Path to the fdf file that is the 'parent' of the results.
    results_path: str, optional
        Directory where the files with the simulations results are
        located. This path has to be relative to the root fdf.
    """

    _plot_type = "Forces"

    _parameters = (

        SileInput(
            key="out_file", name="Output log file",
            dtype=sisl.io.siesta.outSileSiesta,
            default = None,
            group="dataread",
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

    @entry_point("siesta_output")
    def _read_siesta_output(self, root_fdf, out_file):

        out_file = out_file or root_fdf.with_suffix(".out")

        outSile = self.get_sile(out_file)

        self.atomic_forces = outSile.read_force(all=True)

        if len(self.atomic_forces[0]) == 0:
            self.total_forces, self.max_forces = outSile.read_force(all=True, total=True, max=True)
        else:
            self.total_forces, self.max_forces = outSile.read_force(all=True, total=True, max=True)

        self.simulation_ended = outSile.job_completed

    def _after_read(self):

        self.update_layout(xaxis_title="MD step", yaxis_title="Force [eV/Ang]")

    def _set_data(self, type, linecolor):

        forces_type = type

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
                'marker': {'color': linecolor},
                'line': {"color": linecolor},
                'name': 'Max force'
            }]
