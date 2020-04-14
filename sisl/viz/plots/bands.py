import numpy as np
import pandas as pd
import xarray as xr
import itertools

import os
import shutil

import sisl
from ..plot import Plot, PLOTS_CONSTANTS
from ..plotutils import sortOrbitals, copyParams, findFiles, runMultiple, calculateGap
from ..input_fields import TextInput, SwitchInput, ColorPicker, DropdownInput, IntegerInput, FloatInput, RangeInput, RangeSlider, QueriesInput, ProgramaticInput, FunctionInput
from ..input_fields.range import ErangeInput

class BandsPlot(Plot):

    '''
    Plot representation of the bands.
    '''

    _plotType = "Bands"
    
    _requirements = {
        "siesOut": {
            "files": ["$struct$.bands", "*.bands"]
        }
    }
    
    _parameters = (

        TextInput(key = "bandsFile", name = "Path to bands file",
            width = "s100% m50% l33%",
            group="readdata",
            params = {
                "placeholder": "Write the path to your bands file here...",
            },
            help = '''This parameter explicitly sets a .bands file. Otherwise, the bands file is attempted to read from the fdf file '''
        ),

        ProgramaticInput(key = "bandStructure", name = "bandStruct structure object",
            default= None,
            help = "The bandStruct structure object to be used."
        ),

        FunctionInput(key="add_band_trace_data", name="Additional data for band traces",
            default=None,
            positional=["band", "plot"],
            returns=[dict],
            help='''A function that receives each band (as a DataArray) and adds data to the trace. It also recieves the plot object. 
            The returned data may even overwrite the existing one, therefore it can be useful to fully customize your bands plot (individual style for each band if you want).'''
        ),

        FunctionInput(key="eigenstate_map", name="Eigenstate map function",
            default=None,
            positional=["eigenstate", "plot"],
            returns=[],
            help='''This function receives the eigenstate object for each k value when the bands are being extracted from a hamiltonian.
            You can do whatever you want with it, the point of this function is to avoid running the diagonalization process twice.'''
        ),

        ErangeInput(key="Erange",
            help = "Energy range where the bands are displayed."
        ),

        RangeSlider(key = "bandsRange", name = "Bands range",
            default = None,
            width = "s90%",
            params = {
                'step': 1,
            },
            help = "The bands that should be displayed. Only relevant if Erange is None."
        ),

        QueriesInput(key = "path", name = "Bands path",
            default = [],
            help='''Path along which bands are drawn in units of reciprocal lattice vectors.<br>
            Note that if you want to provide a path programatically you can do it more easily with the `bandStructure` setting''',
            queryForm=[

                FloatInput(
                    key="x", name="X",
                    width = "s50% m20% l10%",
                    default=0,
                    params={
                        "step": 0.01
                    }
                ),

                FloatInput(
                    key="y", name="Y",
                    width = "s50% m20% l10%",
                    default=0,
                    params={
                        "step": 0.01
                    }
                ),

                FloatInput(
                    key="z", name="Z",
                    width="s50% m20% l10%",
                    default=0,
                    params={
                        "step": 0.01
                    }
                ),

                IntegerInput(
                    key="divisions", name="Divisions",
                    width="s50% m20% l10%",
                    default=50,
                    params={
                        "min": 0,
                        "step": 10
                    }
                ),

                TextInput(
                    key="tick", name="Tick",
                    width = "s50% m20% l10%",
                    default=None,
                    params = {
                        "placeholder": "Tick..."
                    },
                    help = "Tick that should be displayed at this corner of the path."
                )

            ]
        ),

        SwitchInput(key="gap", name="Show gap",
            default=False,
            params={
                'onLabel': 'Yes',
                'offLabel': 'No'
            },
            help="Whether the gap should be displayed in the plot"
        ),

        SwitchInput(key="directGapsOnly", name="Only direct gaps",
            default=False,
            params={
                'onLabel': 'Yes',
                'offLabel': 'No'
            },
            help="Whether to show only gaps that are direct, according to the gap tolerance"
        ),

        FloatInput(key="gapTol", name="Gap tolerance",
            default=0.01,
            params={
                'step': 0.001
            },
            help='''The difference in k that must exist to consider to gaps different.<br>
            If two gaps' positions differ in less than this, only one gap will be drawn.<br>
            Useful in cases where there are degenerated bands with exactly the same values.'''
        ),

        ColorPicker(key = "gapColor", name = "Gap color",
            default = None,
            help = "Color to display the gap"
        ),

        FloatInput(key = "bandsWidth", name = "Band lines width",
            default = 1,
            help = "Width of the lines that represent the bands"
        ),

        ColorPicker(key = "spinUpColor", name = "No spin/spin up line color",
            default = "black",
            help = "Choose the color to display the bands. <br> This will be used for the spin up bands if the calculation is spin polarized"
        ),

        ColorPicker(key = "spinDownColor", name = "Spin down line color",
            default = "blue",
            help = "Choose the color for the spin down bands.<br>Only used if the calculation is spin polarized."
        ),

    )

    _overwrite_defaults = {
        'xaxis_title': 'K',
        'xaxis_mirror': True,
        'yaxis_mirror': True,
        'xaxis_showgrid': True,
        'yaxis_title': 'Energy (eV)'
    }

    @classmethod
    def _defaultAnimation(self, wdir = None, frameNames = None, **kwargs):
        
        bandsFiles = findFiles(wdir, "*.bands", sort = True)

        def _getFrameNames(self):

            return [os.path.basename( childPlot.setting("bandsFile")) for childPlot in self.childPlots]

        return self.__class__.animated("bandsFile", bandsFiles, frameNames = _getFrameNames, wdir = wdir, **kwargs)

    def _readfromH(self):

        bandStruct = self.setting("bandStructure")
        
        #If no bandStruct structure is provided, then build it ourselves
        if bandStruct is None:

            if not hasattr(self, "H"):
                self.setupHamiltonian()

            #Get the requested path
            self.path = [point for point in self.setting("path") if point["active"]]

            bandStruct = sisl.BandStructure(
                self.H,
                point=np.array([[point["x"] or 0, point["y"] or 0, point["z"] or 0] for point in self.path], dtype=float),
                division=np.array([point["divisions"] for point in self.path][1:], dtype=int) ,
                name=np.array([point["tick"] for point in self.path])
            )
        else:
            self.fermi = 0

            if not hasattr(bandStruct, "H"):
                self.setupHamiltonian()
                bandStruct.set_parent(self.H)
            
        self.ticks = bandStruct.lineartick()
        self.kPath = bandStruct._k

        # We define a wrapper to get the values out of the eigenstates
        # to give the possibility to the user to do something inbetween
        # NOTE THAT THIS IS USED BY FAT BANDS TO GET THE WEIGHTS SIMULTANEOUSLY
        eig_map = self.setting('eigenstate_map')
        def bands_wrapper(eigenstate):
            if callable(eig_map):
                eig_map(eigenstate, self)
            return eigenstate.eig

        # THIS DOES NOT SUPPORT SPIN!!!!!!!!!!!!!!!! (I think)
        self.bands = bandStruct.asdataarray().eigenstate(
            wrap=bands_wrapper,
            coords=('band',),
        )

        self.bands = self.bands.expand_dims('spin', axis=2)
        self.bands['k'] = bandStruct.lineark()

        self.bands.attrs = {"ticks": self.ticks[0], "ticklabels": self.ticks[1]}

    def _readSiesOut(self):
        
        #Get the info from the bands file
        self.path = self.setting("path")

        if self.path and self.path != getattr(self, "siestaPath", None) or self.setting("bandStructure"):
            raise Exception("A path was provided, therefore we can not use the .bands file even if there is one")

        bandsFile = self.setting("bandsFile") or self.requiredFiles[0]

        self.bands = self.get_sile(bandsFile).read_data(as_dataarray=True)
        self.fermi = 0.0 #Energies are already shifted

        # Inform of the path that it's being used if we can
        # THIS IS ONLY WORKING PROPERLY FOR FRACTIONAL UNITS OF THE BAND POINTS RN
        if hasattr(self, "fdfSile") and self.fdfSile.get("BandLines"):

            try:
                self.siestaPath = []
                points = self.fdfSile.get("BandLines")

                for i, point in enumerate(points):

                    divisions, x, y, z, *others = point.split()
                    divisions = int(divisions) - int(points[i-1].split()[0]) if i > 0 else None
                    tick = others[0] if len(others) > 0 else None

                    self.siestaPath.append({"active": True, "x": float(x), "y": float(y), "z": float(z), "divisions": divisions, "tick": tick})
                    
                    self.updateSettings(path=self.siestaPath, updateFig=False, no_log=True)
            except Exception as e:
                print(f"Could not correctly read the bands path from siesta.\n Error {e}")
    
    def _afterRead(self):

        self.isSpinPolarized = len(self.bands.spin.values) == 2
        self._calculateGaps()

        # Make sure that the bandsRange control knows which bands are available
        iBands = self.bands.band.values

        if len(iBands) > 30:
            iBands = iBands[np.linspace(0, len(iBands)-1, 20, dtype=int)]

        self.modifyParam('bandsRange', 'inputField.params', {
            **self.getParam('bandsRange')["inputField"]["params"],
            "min": min(iBands),
            "max": max(iBands),
            "allowCross": False,
            "marks": { int(i): str(i) for i in iBands },
        })
    
    def _setData(self, draw_before_bands=None):
        
        '''
        Converts the bands dataframe into a data object for plotly.

        It stores the data under self.data, so that it can be accessed by posterior methods.

        Returns
        ---------
        self.data: list of dicts
            contains a dictionary for each bandStruct with all its information.
        '''

        Erange = self.setting('Erange')
        # Get the bands that matter for the plot
        if Erange is None:
            bandsRange = self.setting("bandsRange")

            if bandsRange is None:
                # If neither E range or bandsRange was provided, we will just plot the 15 bands below and above the fermi level
                CB = int(self.bands.where(self.bands <= 0).argmax('band').max())
                bandsRange = [int(max(self.bands["band"].min(), CB - 15)), int(min(self.bands["band"].max(), CB + 16))]

            iBands = np.arange(*bandsRange)
            filtered_bands = self.bands.where(self.bands.band.isin(iBands), drop=True)
            self.updateSettings(updateFig=False, Erange=[float(f'{val:.3f}') for val in [float(filtered_bands.min()), float(filtered_bands.max())]], bandsRange=bandsRange, no_log=True)
        else:
            Erange = np.array(Erange) + self.fermi
            filtered_bands = self.bands.where( (self.bands <= Erange[1]) & (self.bands >= Erange[0])).dropna("band", "all")
            self.updateSettings(updateFig=False, bandsRange=[int(filtered_bands['band'].min()), int(filtered_bands['band'].max())], no_log=True)

        add_band_trace_data = self.setting("add_band_trace_data") 
        if not callable(add_band_trace_data):
            add_band_trace_data = lambda * args, **kwargs: {}
        
        # Give the oportunity to draw before bands are drawn (used by Fatbands)
        if callable(draw_before_bands):
            draw_before_bands()

        #Define the data of the plot as a list of dictionaries {x, y, 'type', 'name'}
        self.add_traces(np.ravel([[{
                        'type': 'scatter',
                        'x': band.k.values,
                        'y': (band - self.fermi).values,
                        'mode': 'lines', 
                        'name': "{} spin {}".format( band.band.values, PLOTS_CONSTANTS["spins"][spin]) if self.isSpinPolarized else str(band.band.values) , 
                        'line': {"color": [self.setting("spinUpColor"),self.setting("spinDownColor")][spin], 'width' : self.setting("bandsWidth")},
                        'hoverinfo':'name',
                        "hovertemplate": '%{y:.2f} eV',
                        **add_band_trace_data(band, self)
                    } for band in spin_bands] for spin_bands, spin in zip(filtered_bands.transpose(), filtered_bands.spin.values)]).tolist())
        
        self._drawGaps()

    def _afterGetFigure(self):

        #Add the ticks
        self.figure.layout.xaxis.tickvals = getattr(self.bands, "ticks", None)
        self.figure.layout.xaxis.ticktext = getattr(self.bands, "ticklabels", None)
        self.figure.layout.yaxis.range = np.array(self.setting("Erange")) + self.fermi
    
    def _calculateGaps(self):

        # Calculate the band gap to store it
        above_fermi = self.bands.where(self.bands > 0)
        below_fermi = self.bands.where(self.bands < 0)
        CBbot = above_fermi.min()
        VBtop = below_fermi.max()

        CB = above_fermi.where(above_fermi==CBbot, drop=True).squeeze()
        VB = below_fermi.where(below_fermi==VBtop, drop=True).squeeze()

        self.gap = float(CBbot - VBtop)
        
        self.gap_info = {
            'k': (VB["k"].values, CB['k'].values),
            'bands': (VB["band"].values, CB["band"].values),
            'spin': (VB["spin"].values, CB["spin"].values) if self.isSpinPolarized else (0,0),
            'Es': [float(VBtop), float(CBbot)]
        }

    def _drawGaps(self):

        # Draw gaps
        if self.setting("gap"):

            gap_tolerance = self.setting('gapTol')
            gap_color = self.setting('gapColor')
            only_direct = self.setting('directGapsOnly')

            gapKs = [np.atleast_1d(k) for k in self.gap_info['k']]

            # Remove "equivalent" gaps
            def clear_equivalent(ks):
                if len(ks) == 1:
                    return ks

                uniq = [ks[0]]
                for k in ks[1:]:
                    if abs(min(np.array(uniq) - k)) > gap_tolerance:
                        uniq.append(k)
                return uniq

            all_gapKs = itertools.product(*[clear_equivalent(ks) for ks in gapKs])

            for gapKs in all_gapKs:

                if only_direct and abs(gapKs[1] - gapKs[0]) > gap_tolerance:
                    continue

                self.add_trace({
                    'type': 'scatter',
                    'mode': 'lines+markers+text',
                    'x': gapKs,
                    'y': self.gap_info["Es"],
                    'text': [f'Gap: {self.gap:.3f} eV', ''],
                    'marker':{'color': gap_color },
                    'line': {'color': gap_color},
                    'name': 'Gap',
                    'textposition': 'top right',
                })
