import numpy as np
from xarray import DataArray

import sisl
from ..plot import Plot
from .bands import BandsPlot
from ..input_fields import ProgramaticInput, RangeSlider
from ..input_fields.range import ErangeInput

class FatbandsPlot(BandsPlot):

    _plot_type = 'Fatbands'

    _parameters = (

        ProgramaticInput(
            key="groups", name="Atom groups",
            default=[]
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

        self.update_settings(eigenstate_map=_weights_from_eigenstate, update_fig=False, _nolog=True)

        # We make bands plot read the bands, which will also populate the weights
        # thanks to the above step
        BandsPlot._read_from_H(self)

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

    def _set_data(self):

        # Avoid bands being displayed in the legend individually (it would be a mess)
        self.update_settings(add_band_trace_data=lambda band, plot: {'showlegend': False}, update_fig=False, _nolog=True)

        # We let the bands plot draw the bands
        BandsPlot._set_data(self, draw_before_bands=self._draw_fatbands)
    
    def _draw_fatbands(self):

        # We get the bands range that is going to be plotted
        # Remember that the BandsPlot will have updated this setting accordingly,
        # so it's safe to use it directly
        plotted_bands = self.setting("bands_range")
        plotted_bands[-1] -= 1
        
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

        # Let's plot each group of orbitals as an area that surrounds each band
        for group in groups:

            weights = plot_weights.sel(orb=group["orbitals"]).mean("orb")

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
            } for i, (band, band_weights) in enumerate(zip(plot_eigvals.T, weights.T))])
        
        #self.data = [*self.data[prev_traces:],*self.data[0:prev_traces]]
