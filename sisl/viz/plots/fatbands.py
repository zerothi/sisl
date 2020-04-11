import numpy as np
from xarray import DataArray

import sisl
from ..plot import Plot
from ..input_fields import ProgramaticInput
from ..input_fields.range import ErangeInput

class FatbandsPlot(Plot):

    _plotType = 'Fatbands'

    _parameters = (

        ProgramaticInput(
            key="groups", name="Atom groups",
            default=[]
        ),

        ErangeInput(
            key="Erange",
            help="Energy range where the bands are displayed."
        ),
    )

    def _readfromH(self):

        self.bands_path = sisl.BandStructure(self.H, [[0, 0, 0], [0.5, 0, 0]], 10)

        # Inform that we will want results as datarrays
        self.bands_path.asdataarray()

        eigvals = []

        def wrap_weights(eigenstate):
            eigvals.append(eigenstate.eig)
            return eigenstate.norm2(sum=False)

        self.weights = self.bands_path.eigenstate(
            coords=('band', 'orb'),
            wrap=wrap_weights
        )

        eigvals = np.array(eigvals)

        self.eigvals = DataArray(
            eigvals,
            coords={
                'k': np.arange(0, eigvals.shape[0]),
                'band': np.arange(0, eigvals.shape[1])
            },
            dims=('k', 'band')
        )

    def _setData(self):

        groups = self.setting('groups')
        Erange = self.setting('Erange')

        plot_eigvals = self.eigvals.where((self.eigvals > Erange[0]) & (self.eigvals < Erange[1]), drop=True)
        plot_weights = self.weights.sel(band=plot_eigvals['band'])

        self.data = []

        xs = self.bands_path.lineark()
        area_xs = [*xs, *np.flip(xs)]

        colors = {group["name"]: group["color"] for group in groups}

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

        self.add_traces([{
            'type': 'scatter',
            'mode': 'lines',
            'x': xs,
            'y': band,
            'line': {'color': 'black'},
            'name': int(band['band']),
            'showlegend': False,
        } for band in plot_eigvals.T])
