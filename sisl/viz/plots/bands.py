import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
import itertools

import os
import shutil

import sisl
from ..plot import Plot, MultiplePlot, Animation, PLOTS_CONSTANTS
from ..plotutils import sortOrbitals, initMultiplePlots, copyParams, findFiles, runMultiple, calculateGap
from ..inputFields import InputField, TextInput, SwitchInput, ColorPicker, DropdownInput, IntegerInput, FloatInput, RangeSlider, QueriesInput, ProgramaticInput

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

        RangeSlider(
            key = "Erange", name = "Energy range",
            default = [-2,4],
            width = "s90%",
            params = {
                "min": -10,
                "max": 10,
                "allowCross": False,
                "step": 0.1,
                "marks": { **{ i: str(i) for i in range(-10,11) }, 0: "Ef",},
            },
            help = "Energy range where the bands are displayed."
        ),

        TextInput(
            key = "path", name = "Bands path",
            default = "0,0,0/100/0.5,0,0",
            width = "s100% m50% l33%",
            params = {
                "placeholder": "Write your path here..."
            },
            help = '''Path along which bands are drawn in format:
                            <br>p1x,p1y,p1z/<number of points from P1 to P2>/p2x,p2y,p2z/...'''
        ),

        TextInput(
            key = "ticks", name = "K ticks",
            default = "A,B",
            width = "s100% m50%",
            params = {
                "placeholder": "Write your ticks..."
            },
            help = "Ticks that should be displayed at the corners of the path (separated by commas)."
        ),

        FloatInput(
            key = "bandsWidth", name = "bandStruct lines width",
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

    @classmethod
    def _defaultAnimation(self, wdir = None, frameNames = None, **kwargs):
        
        bandsFiles = findFiles(wdir, "*.bands", sort = True)

        def _getFrameNames(self):

            return [os.path.basename( childPlot.setting("bandsFile")) for childPlot in self.childPlots]

        return BandsPlot.animated("bandsFile", bandsFiles, frameNames = _getFrameNames, wdir = wdir, **kwargs)

    def _afterInit(self):

        self.updateSettings(updateFig = False, xaxis_title = 'K', yaxis_title = "Energy (eV)")

    def _readfromH(self):

        bandStruct = self.setting("bandStructure")
        
        #If no bandStruct structure is provided, then build it ourselves
        if bandStruct is None:

            if not hasattr(self, "H"):
                self.setupHamiltonian()

            #Get the requested path
            self.path = self.setting("path")
            bandPoints, divisions = [], []
            for item in self.path.split("/"):
                splittedItem = item.split(",")
                if splittedItem == [item]:
                    divisions.append(item)
                elif len(splittedItem) == 3:
                    bandPoints.append(splittedItem)
            bandPoints, divisions = np.array(bandPoints, dtype=float), np.array(divisions, dtype=int)

            bandStruct = sisl.BandStructure(self.geom, bandPoints, divisions)
            bandStruct.set_parent(self.H)
        else:
            self.fermi = 0

        self.ticks = bandStruct.lineartick()
        self.Ks = bandStruct.lineark()
        self.kPath = bandStruct._k

        bands = bandStruct.eigh()

        self._bandsToDfs(np.array([bands]))

    def _readSiesOut(self):
        
        #Get the info from the bands file
        self.path = self.setting("path") #This should be modified at some point, it's just so that setData works correctly

        bandsFile = self.setting("bandsFile") or self.requiredFiles[0]
        self.ticks, self.Ks, bands = sisl.get_sile(bandsFile).read_data()
        self.fermi = 0.0 #Energies are already shifted

        #Axes are switched so that the returned array is a list like [spinUpBands, spinDownBands]
        self._bandsToDfs(np.rollaxis(bands, 1))

        #Inform that the bandsFile has been read so that it can be followed if the user wants
        return [bandsFile]
    
    def _bandsToDfs(self, bands):
        '''
        Gets the bands read and stores them in a convenient way into self.df
        '''

        self.isSpinPolarized = bands.shape[0] == 2
        self.df = pd.DataFrame()

        for iSpin, spinComponentBands in enumerate(bands):
            
            df = pd.DataFrame(spinComponentBands.T)
            df.columns = self.Ks
            #We insert these columns at the beggining so that the user can see them if it prints the dataframe
            df.insert(0, "iBand", range(1, spinComponentBands.shape[1] + 1))
            df.insert(1, "iSpin", iSpin)
            df.insert(2, "Emin", np.min(spinComponentBands, axis = 0))
            df.insert(3, "Emax", np.max(spinComponentBands, axis = 0))

            #Append the dataframe to the main dataframe
            self.df = self.df.append(df, ignore_index = True)
        
            self.gap = calculateGap(spinComponentBands)
        
        return self
    
    def _setData(self):
        
        '''
        Converts the bands dataframe into a data object for plotly.

        It stores the data under self.data, so that it can be accessed by posterior methods.

        Returns
        ---------
        self.data: list of dicts
            contains a dictionary for each bandStruct with all its information.
        '''

        #If the path has changed we need to produce the bandStruct structure again
        if self.path != self.setting("path"):
            self.order = ["fromH"]
            self.readData()

        self.data = []

        Erange = np.array(self.setting("Erange")) + self.fermi

        #Get the bands that matter for the plot
        self.plotDF = self.df[ (self.df["Emin"] <= Erange[1]) & (self.df["Emax"] >= Erange[0]) ].dropna(axis = 0, how = "all")

        #Define the data of the plot as a list of dictionaries {x, y, 'type', 'name'}
        self.data = [ *self.data, *[{
                        'type': 'scatter',
                        'x': self.Ks,
                        'y': bandStruct.loc[self.Ks] - self.fermi,
                        'mode': 'lines', 
                        'name': "{} spin {}".format( bandStruct["iBand"], PLOTS_CONSTANTS["spins"][int(bandStruct["iSpin"])]) if self.isSpinPolarized else str(int(bandStruct["iBand"])) , 
                        'line': {"color": [self.setting("spinUpColor"),self.setting("spinDownColor")][int(bandStruct["iSpin"])], 'width' : self.setting("bandsWidth")},
                        'hoverinfo':'name',
                        "hovertemplate": '%{y:.2f} eV',
                    } for i, bandStruct in self.plotDF.sort_values("iBand").iterrows() ] ]

    def _afterGetFigure(self):

        #Add the ticks
        self.figure.layout.xaxis.tickvals = self.ticks[0]
        self.figure.layout.xaxis.ticktext = self.ticks[1]
        self.figure.layout.yaxis.range = np.array(self.setting("Erange")) + self.fermi
