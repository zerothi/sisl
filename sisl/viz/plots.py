'''
This file contains all the plot subclasses
'''

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import os

import sisl
from .configurable import *
from .plot import Plot, PLOTS_CONSTANTS

class BandsPlot(Plot):

    '''
    Plot representation of the bands.
    '''

    _plotType = "Bands"
    
    _requirements = {
        "files": ["$struct$.bands", "*.bands"]
    }
    
    _parameters = (
    
        {
            "key": "Erange" ,
            "name": "Energy range",
            "default": [-2,4],
            "inputField": {
                "type": "rangeslider",
                "width": "s90%",
                "params": {
                    "min": -10,
                    "max": 10,
                    "allowCross": False,
                    "step": 0.1,
                    "marks": { **{ i: str(i) for i in range(-10,11) }, 0: "Ef",},
                    "updatemode": "drag",
                    "units": "eV",
                }
            },
            "tooltip": {
                "message": "Energy range where the bands are displayed. Default: [-2,4]",
                "position": "top"
            },
            "onUpdate": "setData",
        },

        {
            "key": "path" ,
            "name": "Bands path",
            "default": "0,0,0/100/0.5,0,0",
            "inputField": {
                "type": "textinput",
                "width": "s100% m50% l33%",
                "params": {
                    "placeholder": "Write your path here...",
                }
            },
            "tooltip": {
                "message": '''Path along which bands are drawn in format:
                            <br>p1x,p1y,p1z/<number of points from P1 to P2>/p2x,p2y,p2z/...
                            <br>Default: 0,0,0/100/0.5,0,0''',
                "position": "top"
            },
            "onUpdate": "readData",
        },

        {
            "key": "ticks" ,
            "name": "K ticks",
            "default": "A,B",
            "inputField": {
                "type": "textinput",
                "width": "s100% m50%",
                "params": {
                    "placeholder": "Write your ticks...",
                }
            },
            "tooltip": {
                "message": "Ticks that should be displayed at the corners of the path (separated by commas). Default: A,B",
                "position": "top"
            },
            "onUpdate": "getFigure",
        },

        {
            "key": "bandsWidth",
            "name": "Band lines width",
            "default": 1,
            "inputField": {
                "type": "number",
                "width": "s50% m30% l30%",
                "params": {
                    "min": 0,
                    "step": 0.1
                }
            },
            "onUpdate": "setData",
        },
        
        {
            "key": "spinUpColor",
            "name": "No spin/spin up line color",
            "default": "black",
            "inputField": {
                "type": "color",
                "width": "s50% m33% l15%",
            },
            "tooltip": {
                "message": "Choose the color to display the bands. <br> This will be used for the spin up bands if the calculation is spin polarized",
                "position": "top"
            },
            "onUpdate": "setData",
        },

        {
            "key": "spinDownColor",
            "name": "Spin down line color",
            "default": "blue",
            "inputField": {
                "type": "color",
                "width": "s50% m33% l15%",
            },
            "tooltip": {
                "message": "Choose the color for the spin down bands.<br>Only used if the calculation is spin polarized.",
                "position": "top"
            },
            "onUpdate": "setData",
        },

    )

    def __init__(self, **kwargs):
            
        super().__init__(**kwargs)
    
    def _readfromH(self):
        
        #Get the path requested
        self.path = self.settings["path"]
        bandPoints, divisions = [], []
        for item in self.path.split("/"):
            splittedItem = item.split(",")
            if splittedItem == [item]:
                divisions.append(item)
            elif len(splittedItem) == 3:
                bandPoints.append(splittedItem)
        bandPoints, divisions = np.array(bandPoints, dtype = float), np.array(divisions, dtype = int)


        band = sisl.BandStructure(self.geom, bandPoints , divisions )
        band.set_parent(self.H)

        self.ticks = band.lineartick()
        self.Ks = band.lineark()
        self.kPath = band._k


        bands = band.eigh()

        return [bands]

    def _readSiesOut(self):
        
        #Get the info from the bands file
        self.path = self.settings["path"] #This should be modified at some point, it's just so that setData works correctly
        self.ticks, self.Ks, bands = sisl.get_sile(self.requiredFiles[0]).read_data()
        self.fermi = 0.0 #Energies are already shifted

        #Axes are switched so that the returned array is a list like [spinUpBands, spinDownBands]
        return np.rollaxis(bands, 1)
    
    @afterSettingsUpdate
    def readData(self, updateFig = True, **kwargs):
        '''
        Gets the information for the bands plot and stores it into self.df

        Returns
        -----------
        dataRead: boolean
            whether data has been read succesfully or not
        '''

        self.setFiles()
        
        #We try to read from the different sources using the _readFromSources method of the parent Plot class.
        bands = self._readFromSources()

        #Save the bands to dataframes so that we can easily query them
        self.dfs = []
        for spinComponentBands in bands:
            df = pd.DataFrame(spinComponentBands)

            #Set the column headers as strings instead of int (These are the wavefunctions numbers)
            df.columns = df.columns.astype(str)

            self.dfs.append(df)

        if updateFig:
            self.setData(updateFig = updateFig)
        
        return self
    
    @afterSettingsUpdate
    def setData(self, updateFig = True, **kwargs):
        
        '''
        Converts the bands dataframe into a data object for plotly.

        It stores the data under self.data, so that it can be accessed by posterior methods.

        Returns
        ---------
        self.data: list of dicts
            contains a dictionary for each band with all its information.
        '''

        self.reqBandsDfs = []; self.data = []

        for iSpin, df in enumerate(self.dfs):
            #If the path has changed we need to produce the band structure again
            if self.path != self.settings["path"]:
                self.order = ["fromH"]
                self.readData()

            Erange = np.array(self.settings["Erange"]) + self.fermi
            reqBandsDf = df[ df < Erange[1] + 3 ][ df > Erange[0] - 3 ].dropna(axis = 1, how = "all")

            #Define the data of the plot as a list of dictionaries {x, y, 'type', 'name'}
            self.data = [ *self.data, *[{
                            'x': self.Ks[~np.isnan(reqBandsDf[str(column)] - self.fermi)].tolist(),
                            'y': (reqBandsDf[str(column)] - self.fermi)[~np.isnan(reqBandsDf[str(column)] - self.fermi)].tolist(),
                            'mode': 'lines', 
                            'name': "{} spin {}".format(int(column) + 1, PLOTS_CONSTANTS["spins"][iSpin]) if len(self.dfs) == 2 else str(int(column) + 1), 
                            'line': {"color": [self.settings["spinUpColor"],self.settings["spinDownColor"]][iSpin], 'width' : self.settings["bandsWidth"]},
                            'hoverinfo':'name',
                            "hovertemplate": '%{y:.2f} eV',
                        } for column in reqBandsDf.columns ] ]
            
            self.reqBandsDfs.append(reqBandsDf)

        self.data = sorted(self.data, key = lambda x: x["name"])

        if updateFig:
            self.getFigure()
        
        return self