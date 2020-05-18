from collections import defaultdict

import numpy as np
import pandas as pd
from xarray import DataArray

import sisl
from ..plot import Plot
from .bands import BandsPlot
from ..plotutils import random_color
from ..input_fields import OrbitalQueries, TextInput, DropdownInput, SwitchInput, ColorPicker, FloatInput 
from ..input_fields.range import ErangeInput

class FatbandsPlot(BandsPlot):
    '''
    Colorful representation of orbital weights in bands.

    Parameters
    -------------
    %%configurable_settings%%
    '''

    _plot_type = 'Fatbands'

    _parameters = (

        OrbitalQueries(
            key="groups", name="Fatbands groups",
            default=[],
            help='''The different groups that are displayed in the fatbands''',
            queryForm=[

                TextInput(
                    key="name", name="Name",
                    default="Group",
                    width="s100% m50% l20%",
                    params={
                        "placeholder": "Name of the line..."
                    },
                ),

                'species', 'atoms', 'orbitals', 'spin',

                SwitchInput(
                    key="normalize", name="Normalize",
                    default=True,
                    params={
                        "offLabel": "No",
                        "onLabel": "Yes"
                    }
                ),

                ColorPicker(
                    key="color", name="Color",
                    default=None,
                ),

                FloatInput(
                    key="factor", name="Width factor",
                    default=1,
                ),
            ]
        ),

    )

    def _read_siesta_output(self):
        raise NotImplementedError

    def _read_from_H(self):

        self.weights = []

        # Define the function that will "catch" each eigenstate and 
        # build the weights array. See BandsPlot._read_from_H to understand where
        # this will go exactly
        def _weights_from_eigenstate(eigenstate, plot):
            plot.weights.append(eigenstate.norm2(sum=False))

        # We make bands plot read the bands, which will also populate the weights
        # thanks to the above step
        bands_read = False; err = None
        try:
            BandsPlot._read_from_H(self, eigenstate_map=_weights_from_eigenstate)
            bands_read = True
        except Exception as e:
            # Let's keep this error, we are going to at least set the group options so that the
            # user knows what can they choose (specially important for the GUI)
            err = e

        self._set_group_options()
        if not bands_read:
            raise e
        
        # Then we just convert the weights to a DataArray
        self.weights = np.array(self.weights).real

        self.weights = DataArray(
            self.weights,
            coords={
                'k': self.bands.k,
                'band': np.arange(0, self.weights.shape[1]),
                'orb': np.arange(0, self.weights.shape[2]),
            },
            dims=('k', 'band', 'orb')
        )
    
    def _set_group_options(self):

        if not hasattr(self, "geom"):
            self.geom = self.setting("band_structure").parent.geom
        
        self.get_param('groups').update_options(self.geom)

    def _set_data(self):

        # We let the bands plot draw the bands
        BandsPlot._set_data(
            self, draw_before_bands=self._draw_fatbands,
            # Avoid bands being displayed in the legend individually (it would be a mess)
            add_band_trace_data=lambda band, plot: {'showlegend': False}
        )
    
    def _draw_fatbands(self):

        # We get the bands range that is going to be plotted
        # Remember that the BandsPlot will have updated this setting accordingly,
        # so it's safe to use it directly
        plotted_bands = self.setting("bands_range")
        #plotted_bands[-1] -= 1
        
        #Get the bands that matter (spin polarization currently not implemented)
        plot_eigvals = self.bands.sel(band=np.arange(*plotted_bands), spin=0) - self.fermi
        # Get the weights that matter
        plot_weights = self.weights.sel(band=np.arange(*plotted_bands))
        
        # Get the groups of orbitals whose bands are requested
        groups = self.setting('groups')

        # We are going to need a trace that goes forward and then back so that
        # it is self-fillable
        xs = self.bands.k.values
        area_xs = [*xs, *np.flip(xs)]

        prev_traces = len(self.data)

        groups_param = self.get_param("groups")

        # Let's plot each group of orbitals as an area that surrounds each band
        for i ,group in enumerate(groups):

            group = groups_param.complete_query(group, name=f"Group {i+1}")

            #Use only the active requests
            if not group.get("active", True):
                continue

            orb = groups_param.get_orbitals(group)
            
            weights = plot_weights.sel(orb=orb)
            if group["normalize"]:
                weights = weights.mean("orb")
            else:
                weights = weights.sum("orb")

            if group["color"] is None:
                group["color"] = random_color()

            self.add_traces([{
                'type': 'scatter',
                'mode': 'lines',
                'x': area_xs,
                'y': [*(band + band_weights*group["factor"]), *np.flip(band - band_weights*group["factor"])],
                'line':{'width': 0, "color": group["color"]},
                'showlegend': i == 0,
                'name': group["name"],
                'legendgroup':group["name"],
                'fill': 'toself'
            } for i, (band, band_weights) in enumerate(zip(plot_eigvals.transpose('band', 'k'), weights.transpose('band', 'k')))])
