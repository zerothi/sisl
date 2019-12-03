'''
This file contains all the plot subclasses
'''

import numpy as np
import pandas as pd
import multiprocessing as mp
import itertools
import tqdm

import os

import sisl
from .plot import Plot, PLOTS_CONSTANTS
from.plotutils import sortOrbitals, initMultiplePlots

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
            "key": "bandsFile" ,
            "name": "Path to bands file",
            "default": None,
            "inputField": {
                "type": "textinput",
                "width": "s100% m50% l33%",
                "params": {
                    "placeholder": "Write the path to your bands file here...",
                }
            },
            "help": '''This parameter explicitly sets a .bands file. Otherwise, the bands file is attempted to read from the fdf file ''',
            "onUpdate": "readData",
        },
    
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
                }
            },
            "help": "Energy range where the bands are displayed.",
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
            "help": '''Path along which bands are drawn in format:
                            <br>p1x,p1y,p1z/<number of points from P1 to P2>/p2x,p2y,p2z/...''',
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
            "help": "Ticks that should be displayed at the corners of the path (separated by commas).",
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
            "help": "Width of the lines that represent the bands",
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
            "help":"Choose the color to display the bands. <br> This will be used for the spin up bands if the calculation is spin polarized",
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
            "help": "Choose the color for the spin down bands.<br>Only used if the calculation is spin polarized.",
            "onUpdate": "setData",
        },

    )

    def _afterInit(self):

        self.updateSettings(updateFig = False, xaxis_title = 'K', yaxis_title = "Energy (eV)")

    def _readfromH(self):

        if not hasattr(self, "H"):
            self.setupHamiltonian()
        
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

        return self._bandsToDfs([bands]) 

    def _readSiesOut(self):
        
        #Get the info from the bands file
        self.path = self.settings["path"] #This should be modified at some point, it's just so that setData works correctly

        bandsFile = self.settings["bandsFile"] or self.requiredFiles[0]
        self.ticks, self.Ks, bands = sisl.get_sile(bandsFile).read_data()
        self.fermi = 0.0 #Energies are already shifted

        #Axes are switched so that the returned array is a list like [spinUpBands, spinDownBands]
        return self._bandsToDfs(np.rollaxis(bands, 1))
    
    def _bandsToDfs(self, bands):
        '''
        Gets the bands read and stores them in a convenient way into self.dfs
        '''

        self.dfs = []
        for spinComponentBands in bands:
            df = pd.DataFrame(spinComponentBands)

            #Set the column headers as strings instead of int (These are the wavefunctions numbers)
            df.columns = df.columns.astype(str)

        self.dfs.append(df)

        # y = {
        #     'tickvals': self.ticks[0],
        #     'ticktext': self.settings["ticks"].split(",") if self.source != "siesOut" else self.ticks[1],
        # }
        
        return self
    
    def _setData(self):
        
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
            reqBandsDf = df[ df < Erange[1]][ df > Erange[0]].dropna(axis = 1, how = "all")

            #Define the data of the plot as a list of dictionaries {x, y, 'type', 'name'}
            self.data = [ *self.data, *[{
                            'type': 'scatter',
                            'x': self.Ks,
                            'y': reqBandsDf[str(column)] - self.fermi,
                            'mode': 'lines', 
                            'name': "{} spin {}".format(int(column) + 1, PLOTS_CONSTANTS["spins"][iSpin]) if len(self.dfs) == 2 else str(int(column) + 1), 
                            'line': {"color": [self.settings["spinUpColor"],self.settings["spinDownColor"]][iSpin], 'width' : self.settings["bandsWidth"]},
                            'hoverinfo':'name',
                            "hovertemplate": '%{y:.2f} eV',
                        } for column in reqBandsDf.columns ] ]
            
            self.reqBandsDfs.append(reqBandsDf)

        self.data = sorted(self.data, key = lambda x: x["name"])

class BandsAnimation(Plot):

    #Define all the class attributes
    _plotType = "Bands animation"

    _parameters = BandsPlot._parameters

    def _readSiesOut(self):

        #Get the relevant files
        wdir = os.path.join(self.rootDir, self.settings["resultsPath"])
        files = os.listdir( wdir )
        self.bandsFiles = sorted( [ fileName for fileName in files if (".bands" in fileName)] )

        kwargsList = [{ "bandsFile": bandsFile } for bandsFile in self.bandsFiles]

        self.singlePlots = initMultiplePlots(BandsPlot, kwargsList = kwargsList)
            
    def _setData(self):

        self.frames=[]
        self.data = self.singlePlots[0].data

        for plot in self.singlePlots:

            #plot.updateSettings(**{**self.settings, "bandsFile": plot.settings["bandsFile"]} )
            
            #Define the frames of the animation
            self.frames.append({'name': os.path.basename(plot.settings["bandsFile"]), 'data': plot.data})

class PdosPlot(Plot):

    '''
    Plot representation of the projected density of states.
    '''

    #Define all the class attributes
    _plotType = "PDOS"

    _requirements = {"files": ["$struct$.PDOS"]}

    _parameters = (
        
        {
            "key": "PDOSFile" ,
            "name": "Path to PDOS file",
            "default": None,
            "inputField": {
                "type": "textinput",
                "width": "s100% m50% l33%",
                "params": {
                    "placeholder": "Write the path to your PDOS file here...",
                }
            },
            "help": '''This parameter explicitly sets a .PDOS file. Otherwise, the PDOS file is attempted to read from the fdf file ''',
            "onUpdate": "readData",
        },

        {
            "key": "Erange",
            "name": "Energy range" ,
            "default": [-2,4],
            "inputField": {
                "type": "rangeslider",
                "width": "s100%",
                "params": {
                    "min": -10,
                    "max": 10,
                    "step": 0.1,
                    "marks": { **{ i: str(i) for i in range(-10,11) }, 0: "Ef",},
                }
            },
            "help": "Energy range where the PDOS is displayed. Default: [-2,4]",
            "onUpdate": "setData",
        },

        {
            "key": "requests",
            "name": "PDOS queries" ,
            "default": [{"active": True, "linename": "DOS", "species": None, "atoms": None, "orbitals": None, "spin": None, "normalize": False}],
            "inputField": {
                "type": "queries",
                "width": "s100%",
                "queryForm": [
                    {
                        "key": "linename",
                        "name": "Name",
                        "default": "DOS",
                        "inputField": {
                            "type": "textinput",
                            "width": "s100% m50% l20%",
                            "params": {
                                "placeholder": "Name of the line..."
                            },
                        }
                    },

                    {
                        "key" : "species",
                        "name" : "Species",
                        "default": None,
                        "inputField": {
                            "type": "dropdown",
                            "width": "s100% m50% l40%",
                            "params": {
                                "options":  [],
                                "isMulti": True,
                                "placeholder": "",
                                "isClearable": True,
                                "isSearchable": True,    
                            },
                        },
                    },

                    {
                        "key" : "atoms",
                        "name" : "Atoms",
                        "default": None,
                        "inputField": {
                            "type": "dropdown",
                            "width": "s100% m50% l40%",
                            "params": {
                                "options":  [],
                                "isMulti": True,
                                "placeholder": "",
                                "isClearable": True,
                                "isSearchable": True,    
                            },
                        },
                    },

                    {
                        "key" : "orbitals",
                        "name" : "Orbitals",
                        "default": None,
                        "inputField": {
                            "type": "dropdown",
                            "width": "s100% m50% l50%",
                            "params": {
                                "options":  [],
                                "isMulti": True,
                                "placeholder": "",
                                "isClearable": True,
                                "isSearchable": True,    
                            },
                        },
                    },

                    {
                        "key" : "spin",
                        "name" : "Spin",
                        "default": None,
                        "inputField": {
                            "type": "dropdown",
                            "width": "s100% m50% l25%",
                            "params": {
                                "options":  [],
                                "placeholder": "",
                                "isMulti": False,
                                "isClearable": True,
                            },
                            "style": {
                                "width": 200
                            }
                        },
                    },

                    {
                        "key" : "normalize",
                        "name": "Normalize",
                        "default": False,
                        "inputField": {
                            "type": "switch",
                            "width": "s100% m50% l25%",
                        },
                        
                    },

                ]
            },

            "help": '''Here you can ask for the specific PDOS that you need. 
                    <br>TIP: Queries can be activated and deactivated.''',
            "onUpdate": "setData",
        },
    
    )
    
    def _afterInit(self):

        self.updateSettings(updateFig = False, xaxis_title = 'Density of states (1/eV)', yaxis_title = "Energy (eV)")

    def _readfromH(self):

        if not hasattr(self, "H"):
            self.setupHamiltonian()

        #Calculate the pdos with sisl using the last geometry and the hamiltonian
        self.monkhorstPackGrid = [15, 1, 1]
        Erange = self.settings["Erange"]
        self.E = np.linspace( Erange[0], Erange[-1], 1000) 

        mp = sisl.MonkhorstPack(self.H, self.monkhorstPackGrid)
        self.PDOSinfo = mp.asaverage().PDOS(self.E + self.fermi , eta=True)

    def _readSiesOut(self):

        PDOSFile = self.settings["PDOSFile"] or self.requiredFiles[0]
        #Get the info from the .PDOS file
        self.geom, self.E, self.PDOSinfo = sisl.get_sile(PDOSFile).read_data()

        self.fermi = 0    

    def _afterRead(self):

        '''

        Gets the information out of the .pdos and processes it into self.PDOSdicts so that it can be accessed by 
        the self.setData() method once the orbitals/atoms to display PDOS are selected.

        The method stores all the information in a pandas dataframe that looks like this:

             |  species  |     1s     |    2s      |
        _____|___________|____________|____________|......
             |           |            |            |
        iAt  |  string   | PDOS array | PDOS array |
        _____|___________|____________|____________|......
             .           .            .            .
             .           .            .            .

        Returns
        ---------
        self.df:
            The dataframe obtained
        self.E:
            The energy values where PDOS is calculated

        '''

        #Get the orbital where each atom starts
        orbitals = np.array([0] + [len(atom) for atom in self.geom.atoms]).cumsum()[:-1]

        #Initialize a dataframe to store all the information
        self.df = pd.DataFrame()
        
        #Normalize self.PDOSinfo to do the same treatment for both spin-polarized and spinless simulations
        self.hasSpin = len(self.PDOSinfo.shape) == 3
        self.PDOSinfo = np.moveaxis(self.PDOSinfo, 0, -1) if self.hasSpin else self.PDOSinfo

        #Save the information of each atom
        for at, initOrb in zip( self.geom.atoms, orbitals ):
    
            atDict = {orb.name() : self.PDOSinfo[ initOrb + iOrb , :] for iOrb, orb in enumerate(at) }
            atDict["species"] = at.tag
            
            self.df = self.df.append(atDict, ignore_index = True)
        
        #"Inform" the queries of the available options
        for i, param in enumerate(self.params):

            if param["key"] == "requests":
                for iParam, reqParam in enumerate(self.params[i]["inputField"]["queryForm"]):

                    options = []
                    
                    if reqParam["key"] == "atoms":

                        options = [{ "label": "{} ({})".format(i, self.df.iloc[i]["species"]), "value": i } 
                            for i in range( self.df.shape[0] )]
                
                    elif reqParam["key"] == "species":

                        options = [{ "label": spec, "value": spec } for spec in self.df.species.unique()]
                    
                    elif reqParam["key"] == "orbitals":

                        orbitals = sortOrbitals([column for column in self.df.columns.tolist() if column != "species"])

                        options = [{ "label": orbName, "value": orbName } for orbName in orbitals]
                    
                    elif reqParam["key"] == "spin":

                        options = [{ "label": "↑", "value": 0 },{ "label": "↓", "value": 1 }] if self.hasSpin else []

                    if options:
                        self.params[i]["inputField"]["queryForm"][iParam]["inputField"]["params"]["options"] = options

    def _setData(self):
        '''

        Uses the information processed by the self.readData() method and converts it into a data object for plotly.

        It stores the data under self.data, so that it can be accessed by posterior methods.

        Arguments
        ---------
        requests: list of [ list of (int, str and dict) ]
            contains all the user's requests for the PDOS display.

            The contributions of all the requests under a request group [ ] will be summed and displayed together.
        
        normalize: bool
            whether the contribution is normalized by the number of atoms.

        Returns
        ---------
        self.data: list of dicts
            contains a dictionary for each band with all its information.

        '''

        #Get only the energies we are interested in
        if not self.fermi:
            Emin, Emax = np.array(self.settings["Erange"])
            Estep = ( max(self.E) - min(self.E) ) / len(self.E)
            iEmax = int( ( Emax - min(self.E) ) / Estep )
            iEmin = int( ( Emin - min(self.E) ) / Estep )
            self.Ecut = self.E[ iEmin : iEmax+1 ]
        #else:
        #    self.readData( fromH = True )

        #Initialize the data object for the plotly graph
        self.data = []

        #Go request by request and plot the corresponding PDOS contribution
        for request in self.settings["requests"]:

            #Use only the active requests
            if not request["active"]:
                continue

            #Get the part of the dataframe that the request is asking for
            reqDf = self.df.copy()

            if request.get("atoms"):
                reqDf = reqDf.iloc[np.array(request["atoms"])]

            if request.get("species"):
                reqDf = reqDf[ reqDf["species"].isin(request["species"]) ]

            if request.get("orbitals"):
                reqDf = reqDf[ request["orbitals"] ]
            
            #Remove the species column if it is still there
            try:
                PDOS = reqDf.drop("species", axis = 1)
            except KeyError:
                PDOS = reqDf

            if PDOS.isnull().values.all(axis=0)[0]:

                PDOS = []
                log.warning("No PDOS for the following request: {}".format(request.params))
            else:
                
                PDOS = PDOS.sum(1, skipna = True)

                if request.get("normalize"):
                    PDOS = PDOS.mean()
                else:
                    PDOS = PDOS.sum(0)
            
            #Get the spin component (or the sum of both components)
            if self.hasSpin:
                spinReq = request.get("spin")
                if type(spinReq) == int:
                    PDOS = PDOS[:, spinReq]
                else:
                    PDOS = PDOS.sum(1)
            
            self.data.append({
                        'type': 'scatter',
                        'x': PDOS[ iEmin : iEmax + 1 ] if not self.fermi else PDOS,
                        'y': self.Ecut if not self.fermi else self.E ,
                        'mode': 'lines', 
                        'name': request["linename"], 
                        'line': {'width' : 1},
                        "hoverinfo": "name",
                    })
        
        return self.data

    def addRequest(self, newReq = {}, **kwargs):
        '''
        Adds a new PDOS request. The new request can be passed as a dict or as a list of keyword arguments.
        The keyword arguments will overwrite what has been passed as a dict if there is conflict.
        '''

        self.updateSettings(requests = [*self.settings["requests"], {"active": True, "linename": len(self.settings["requests"]), **newReq, **kwargs}])

        return self

class PdosAnimation(Plot):
    
    '''
    Plot representation of the projected density of states.
    '''

    #Define all the class attributes
    _plotType = "PDOS animation"

    _requirements = {"files": ["$struct$.PDOS"]}

    _parameters = PdosPlot._parameters

    def _readSiesOut(self):

        #At the moment read data only from PDOS files (maybe in a future generate it also with Hs)

        #Get the relevant files
        wdir = os.path.join(self.rootDir, self.settings["resultsPath"])
        files = os.listdir( wdir )
        self.PDOSFiles = sorted( [ fileName for fileName in files if (".PDOS" in fileName)] )

        kwargsList = [{ "PDOSFile": PDOSFile } for PDOSFile in self.PDOSFiles]

        self.singlePlots = initMultiplePlots(PdosPlot, kwargsList = kwargsList)

    def _setData(self):

        self.frames = []
        self.data = self.singlePlots[0].data

        for i, plot in enumerate(self.singlePlots):

            #plot.updateSettings(**{**self.settings, "PDOSFile": plot.settings["PDOSFile"]})
            
            #Define the frames of the animation
            self.frames.append({'name': plot.settings["PDOSFile"], 'data': plot.data})

