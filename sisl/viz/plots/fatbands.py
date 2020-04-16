import numpy as np
from xarray import DataArray

import sisl
from ..plot import Plot
from .bands import BandsPlot
from ..input_fields import ProgramaticInput, RangeSlider
from ..input_fields.range import ErangeInput

class FatbandsPlot(BandsPlot):

    _plotType = 'Fatbands'

    _parameters = (

        ProgramaticInput(
            key="groups", name="Atom groups",
            default=[]
        ),

    )

    def _readSiesOut(self):
        raise NotImplementedError

    def _readfromH(self):

        self.weights = []

        # Define the function that will "catch" each eigenstate and 
        # build the weights array. See BandsPlot._readFromH to understand where
        # this will go exactly
        def _weights_from_eigenstate(eigenstate, plot):
            plot.weights.append(eigenstate.norm2(sum=False))

        self.updateSettings(eigenstate_map=_weights_from_eigenstate, updateFig=False, _nolog=True)

        # We make bands plot read the bands, which will also populate the weights
        # thanks to the above step
        BandsPlot._readfromH(self)

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

    def _setData(self):

        # Avoid bands being displayed in the legend individually (it would be a mess)
        self.updateSettings(add_band_trace_data=lambda band, plot: {'showlegend': False}, updateFig=False, _nolog=True)

        # We let the bands plot draw the bands
        BandsPlot._setData(self, draw_before_bands=self._draw_fatbands)
    
    def _draw_fatbands(self):

        # We get the bands range that is going to be plotted
        # Remember that the BandsPlot will have updated this setting accordingly,
        # so it's safe to use it directly
        plotted_bands = self.setting("bandsRange")
        
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
