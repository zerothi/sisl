import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
import itertools

import os
import shutil

import sisl
from ..plot import Plot, PLOTS_CONSTANTS
from ..plotutils import sortOrbitals, copyParams, findFiles, runMultiple, calculateGap
from ..inputFields import TextInput, SwitchInput, ColorPicker, DropdownInput, IntegerInput, FloatInput, RangeInput, RangeSlider, QueriesInput, ProgramaticInput

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

        TextInput(
            key = "bandsFile", name = "Path to bands file",
            width = "s100% m50% l33%",
            group="readdata",
            params = {
                "placeholder": "Write the path to your bands file here...",
            },
            help = '''This parameter explicitly sets a .bands file. Otherwise, the bands file is attempted to read from the fdf file '''
        ),

        ProgramaticInput(
            key = "bandStructure", name = "bandStruct structure object",
            default=None,
            help = "The bandStruct structure object to be used."
        ),

        RangeInput(
            key = "Erange", name = "Energy range",
            default = [-20,20],
            params = {
                "step": 1,
            },
            help = "Energy range where the bands are displayed."
        ),

        DropdownInput(
            key="usedRange", name = "Range to use",
            default="E",
            params={
                'options': [
                    {'label': 'Energy', 'value': 'E'},
                    {'label': 'Bands', 'value': 'bands'}
                ],
                'isSearchable': True,
                'isMulti': False,
                'isClearable': False
            }
        ),

        RangeSlider(
            key = "bandsRange", name = "Bands range",
            default = [1, 10**4],
            width = "s90%",
            params = {
                'step': 1,
            },
            help = "The bands that should be displayed."
        ),

        QueriesInput(
            key = "path", name = "Bands path",
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

        SwitchInput(
            key="showGap", name="Show gap",
            default=False,
            params={
                'onLabel': 'Yes',
                'offLabel': 'No'
            },
            help="Whether the gap should be displayed in the plot"
        ),

        FloatInput(
            key = "bandsWidth", name = "Band lines width",
            default = 1,
            help = "Width of the lines that represent the bands"
        ),

        ColorPicker(
            key = "spinUpColor", name = "No spin/spin up line color",
            default = "black",
            help = "Choose the color to display the bands. <br> This will be used for the spin up bands if the calculation is spin polarized"
        ),

        ColorPicker(
            key = "spinDownColor", name = "Spin down line color",
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

        return BandsPlot.animated("bandsFile", bandsFiles, frameNames = _getFrameNames, wdir = wdir, **kwargs)

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
        self.Ks = bandStruct.lineark()
        self.kPath = bandStruct._k

        bands = bandStruct.eigh()

        self._bandsToXArray(bands.expand_dims(axis=1) if bands.ndim == 2 else bands)

    def _readSiesOut(self):
        
        #Get the info from the bands file
        self.path = self.setting("path")
        if self.path and self.path != getattr(self, "siestaPath", None) or self.setting("bandStructure"):
            raise Exception("A path was provided, therefore we can not use the .bands file even if there is one")

        bandsFile = self.setting("bandsFile") or self.requiredFiles[0]
        self.ticks, self.Ks, bands = sisl.get_sile(bandsFile).read_data()
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
                    
                    self.updateSettings(path=self.siestaPath, updateFig=False)
            except Exception as e:
                print(f"Could not correctly read the bands path from siesta.\n Error {e}")

        self._bandsToXArray(bands)

        #Inform that the bandsFile has been read so that it can be followed if the user wants
        return [bandsFile]
    
    def _bandsToXArray(self, bands):

        ticks = {"tick_vals": self.ticks[0], "tick_labels": self.ticks[1]}
        self.arr = xr.DataArray(
            name="Energy",
            data=bands,
            coords=[
                ("K", self.Ks),
                ("spin", np.arange(0,bands.shape[1])),
                ("iBand", np.arange(0,bands.shape[2]) + 1)
            ],
            attrs= {**ticks}
        )

        self.isSpinPolarized = len(self.arr.spin.values) == 2

        # Calculate the band gap to store it
        above_fermi = self.arr.where(self.arr > 0)
        below_fermi = self.arr.where(self.arr < 0)
        CBbot = above_fermi.min()
        VBtop = below_fermi.max()

        CB = above_fermi.where(above_fermi==CBbot, drop=True).squeeze()
        VB = below_fermi.where(below_fermi==VBtop, drop=True).squeeze()

        self.gap = float(CBbot - VBtop)
        
        self.gap_info = {
            'k': (VB["K"].values, CB['K'].values),
            'bands': (VB["iBand"].values, CB["iBand"].values),
            'spin': (VB["spin"].values, CB["spin"].values),
            'Es': [float(VBtop), float(CBbot)]
        }
    
    def _afterRead(self):

        # Make sure that the iBands control knows which bands are available
        iBands = self.arr.iBand.values

        self.modifyParam('bandsRange', 'inputField.params', {
            **self.getParam('bandsRange')["inputField"]["params"],
            "min": min(iBands),
            "max": max(iBands),
            "allowCross": False,
            "marks": { int(i): str(i) for i in iBands },
        })
    
    def _setData(self):
        
        '''
        Converts the bands dataframe into a data object for plotly.

        It stores the data under self.data, so that it can be accessed by posterior methods.

        Returns
        ---------
        self.data: list of dicts
            contains a dictionary for each bandStruct with all its information.
        '''

        #Get the bands that matter for the plot
        #self.plotDF = self.df[ (self.df["Emin"] <= Erange[1]) & (self.df["Emax"] >= Erange[0]) ].dropna(axis = 0, how = "all")
        if self.setting("usedRange") == 'bands':
            iBands = np.arange(*self.setting("bandsRange"))
            filtered_bands = self.arr.where(self.arr.iBand.isin(iBands), drop=True)
            self.updateSettings(updateFig=False, Erange=[float(f'{val:.3f}') for val in [float(filtered_bands.min()), float(filtered_bands.max())]])
        else:
            Erange = np.array(self.setting("Erange")) + self.fermi
            filtered_bands = self.arr.where( (self.arr <= Erange[1]) & (self.arr >= Erange[0])).dropna("iBand", "all")

        #Define the data of the plot as a list of dictionaries {x, y, 'type', 'name'}
        self.data = np.ravel([[{
                        'type': 'scatter',
                        'x': self.Ks,
                        'y': (band - self.fermi).values,
                        'mode': 'lines', 
                        'name': "{} spin {}".format( band.iBand.values, PLOTS_CONSTANTS["spins"][band.spin.values]) if self.isSpinPolarized else str(band.iBand.values) , 
                        'line': {"color": [self.setting("spinUpColor"),self.setting("spinDownColor")][band.spin.values], 'width' : self.setting("bandsWidth")},
                        'hoverinfo':'name',
                        "hovertemplate": '%{y:.2f} eV',
                    } for band in spin_bands] for spin_bands in filtered_bands.transpose()]).tolist()
        
        if self.setting("showGap"):

            all_gapKs = itertools.product(*[np.atleast_1d(k) for k in self.gap_info['k']] )

            for gapKs in all_gapKs:
                self.add_trace({
                    'type': 'scatter',
                    'mode': 'lines+markers+text',
                    'x': gapKs,
                    'y': self.gap_info["Es"],
                    'text': [f'Gap: {self.gap:.3f} eV', ''],
                    'textposition': 'top right',
                })

    def _afterGetFigure(self):

        #Add the ticks
        self.figure.layout.xaxis.tickvals = self.ticks[0]
        self.figure.layout.xaxis.ticktext = self.ticks[1]
        self.figure.layout.yaxis.range = np.array(self.setting("Erange")) + self.fermi
