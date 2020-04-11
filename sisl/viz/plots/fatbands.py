import numpy as np
from xarray import DataArray

import sisl
from ..plot import Plot
from ..input_fields import ProgramaticInput, RangeSlider
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

        RangeSlider(
            key="bandsRange", name="Bands range",
            default=None,
            width="s90%",
            params={
                'step': 1,
            },
            help="The bands that should be displayed. Only relevant if Erange is None."
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

    def _afterRead(self):

        # Make sure that the bandsRange control knows which bands are available
        iBands = self.eigvals.band.values

        if len(iBands) > 30:
            iBands = iBands[np.linspace(0, len(iBands)-1, 20, dtype=int)]

        self.modifyParam('bandsRange', 'inputField.params', {
            **self.getParam('bandsRange')["inputField"]["params"],
            "min": min(iBands),
            "max": max(iBands),
            "allowCross": False,
            "marks": {int(i): str(i) for i in iBands},
        })

    def _setData(self):

        Erange = self.setting('Erange')
        # Get the bands that matter for the plot
        if Erange is None:
            bandsRange = self.setting("bandsRange")

            if bandsRange is None:
                # If neither E range or bandsRange was provided, we will just plot the 15 bands below and above the fermi level
                CB = int(self.eigvals.where(self.eigvals <= 0).argmax('band').max())
                bandsRange = [int(max(self.eigvals["band"].min(), CB - 15)),
                              int(min(self.eigvals["band"].max(), CB + 16))]

            iBands = np.arange(*bandsRange)
            plot_eigvals = self.eigvals.where(
                self.eigvals.band.isin(iBands), drop=True)
            self.updateSettings(updateFig=False, Erange=[float(f'{val:.3f}') for val in [float(
                plot_eigvals.min()), float(plot_eigvals.max())]], bandsRange=bandsRange, no_log=True)
        else:
            Erange = np.array(Erange)
            plot_eigvals = self.eigvals.where((self.eigvals <= Erange[1]) & (
                self.eigvals >= Erange[0])).dropna("band", "all")
            self.updateSettings(updateFig=False, bandsRange=[int(
                plot_eigvals['band'].min()), int(plot_eigvals['band'].max())], no_log=True)
        
        plot_weights = self.weights.sel(band=plot_eigvals['band'])

        groups = self.setting('groups')
        
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
