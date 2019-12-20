'''
This file contains all the plot subclasses
'''

import numpy as np
import pandas as pd
import multiprocessing as mp
import itertools
from copy import deepcopy
import tqdm

import os
import shutil

import sisl
from .plot import Plot, MultiplePlot, Animation, PLOTS_CONSTANTS
from .plotutils import sortOrbitals, initMultiplePlots, copyParams, findFiles

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

    def _afterGetFigure(self):

        #Add the ticks
        self.figure.layout.xaxis.tickvals = self.ticks[0]
        self.figure.layout.xaxis.ticktext = self.ticks[1]

class BandsAnimation(Animation):

    #Define all the class attributes
    _plotType = "Bands animation"

    _parameters = copyParams( BandsPlot._parameters, exclude = ["bandsFile"])

    #All these are variables used by MultiplePlot
    _plotClasses = BandsPlot

    def _getInitKwargsList(self):

        self.bandsFiles = findFiles(self.wdir, "*.bands", sort = True)

        return [{ "bandsFile": bandsFile } for bandsFile in self.bandsFiles]

    def _getFrameNames(self):

        return [os.path.basename( childPlot.settings["bandsFile"] ) for childPlot in self.childPlots]

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
            
            if len(self.Ecut) <= 1:
                print("PDOS Plot error: There is no data for the provided energy range ({}).\n The energy range of the read data is: [{},{}]"
                    .format(self.settings["Erange"], min(self.E), max(self.E))
                )
        #else:
        #    self.readData()

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
                print("PDOS Plot warning: No PDOS for the following request: {}".format(request.params))
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

class PdosAnimation(Animation):
    
    '''
    Plot representation of the projected density of states.
    '''

    #Define all the class attributes
    _plotType = "PDOS animation"

    _parameters = copyParams( PdosPlot._parameters, exclude = ["PDOSFile"])

    #All these are variables used by MultiplePlot
    _plotClasses = PdosPlot

    def _getInitKwargsList(self):

        self.PDOSFiles = findFiles(self.wdir, "*.PDOS", sort = True)

        return [{ "PDOSFile": PDOSFile } for PDOSFile in self.PDOSFiles]

    def _getFrameNames(self):

        return [os.path.basename( childPlot.settings["PDOSFile"] ) for childPlot in self.childPlots]

    def _afterChildsUpdated(self):

        #This will make sure that the correct options are available for the PDOS requests of the parent plot.
        self.params = copyParams( self.childPlots[0].params, exclude = ["PDOSFile"])

class LDOSmap(Plot):
    '''
    Generates a heat map with the STS spectra along a path.
    '''
    
    _plotType = "LDOS map"
    
    _requirements = {
        "files": ["$struct$.DIM", "$struct$.PLD", "*.ion" , "$struct$.selected.WFSX"]
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
                }
            },
            "help": "Energy range where the STS spectra are computed.",
            "onUpdate": "readData",           
        },
        
        {
            "key": "nE",
            "name": "Energy points",
            "default": 100,
            "inputField": {
                "type": "number",
                "width": "s30%",
                "params": {
                    "min": 1,
                    "step": 1
                }
            },
            "onUpdate": "readData", 
        },

        {
            "key": "STSEta",
            "name": "Smearing factor (eV)",
            "default": 0.05,
            "inputField": {
                "type": "number",
                "width": "s30%",
                "params": {
                    "min": 1,
                    "step": 1
                }
            },
            "help": '''This determines the smearing factor of each STS spectra. You can play with this to modify sensibility in the vertical direction.
                <br> If the smearing value is too high, your map will have a lot of vertical noise''',
            "onUpdate": "readData", 
        },

        {
            "key": "distStep",
            "name": "Distance step (Ang)",
            "default": 0.1,
            "inputField": {
                "type": "number",
                "width": "s30%",
                "params": {
                    "min": 1,
                    "step": 1
                }
            },
            "onUpdate": "readData", 
        },

        {
            "key": "points",
            "name": "Path points" ,
            "default": [{"x": 0, "y": 0, "z": 0, "active": True}],
            "inputField":{
                "type": "queries",
                "width": "s100%",
                "queryForm": [
                    {
                        "key": "x",
                        "name": "x",
                        "default": 0,
                        "inputField": {
                            "type": "number",
                            "width": "s30%",
                            "params": {
                                "step": 0.01
                            }
                        },
                    },

                    {
                        "key": "y",
                        "name": "y",
                        "default": 0,
                        "inputField": {
                            "type": "number",
                            "width": "s30%",
                            "params": {
                                "step": 0.01
                            }
                        },
                    },

                    {
                        "key": "z",
                        "name": "z",
                        "default": 0,
                        "inputField": {
                            "type": "number",
                            "width": "s30%",
                            "params": {
                                "step": 0.01
                            }
                        },
                    },

                ]
            },

            "help": '''Provide the points to generate the path through which STS need to be calculated.''',
            "onUpdate": "readData",
        },

        {
            "key": "zmin",
            "name": "Lower Z bound",
            "default": 0,
            "inputField": {
                "type": "number",
                "width": "s30%",
                "params": {
                    "min": 0,
                    "step": 10*-6
                }
            },
            "onUpdate": "setData", 
        },

        {
            "key": "zmax",
            "name": "Upper Z bound",
            "default": 0,
            "inputField": {
                "type": "number",
                "width": "s30%",
                "params": {
                    "min": 0,
                    "step": 10*-6
                }
            },
            "onUpdate": "setData", 
        },
    
    )

    def _afterInit(self):

        self.updateSettings(updateFig = False, xaxis_title = "Path coordinate", yaxis_title = "E-Ef (eV)")

    def _getdencharSTSfdf(self, stsPosition):
        
        return '''
            Denchar.PlotSTS .true.
            Denchar.PlotWaveFunctions   .false.
            Denchar.PlotCharge .false.

            %block Denchar.STSposition
                {} {} {}
            %endblock Denchar.STSposition

            Denchar.STSEmin {} eV
            Denchar.STSEmax {} eV
            Denchar.STSEnergyPoints {}
            Denchar.CoorUnits Ang
            Denchar.STSEta {} eV
            '''.format(*stsPosition, *(np.array(self.settings["Erange"]) + self.fermi), self.settings["nE"], self.settings["STSEta"])

    def _readSiesOut(self):
        '''Function that uses denchar to get STSpecra along a path'''

        #Find fermi level
        self.fermi = False
        for line in open(os.path.join(self.rootDir, "{}.out".format(self.struct)) ):
            if "Fermi =" in line:
                self.fermi = float(line.split()[-1])
                print("\nFERMI LEVEL FOUND: {} eV\n Energies will be relative to this level (E-Ef)\n".format(self.fermi))

        if not self.fermi:
            print("\nFERMI LEVEL NOT FOUND IN THE OUTPUT FILE. \nEnergy values will be absolute\n")
            self.fermi = 0

        Emin, Emax = np.array(self.settings["Erange"]) + self.fermi

        Estep = (Emax - Emin)/self.settings["nE"]

        points = np.array([[point["x"],point["y"],point["z"]] for point in self.settings["points"] if point["active"]])
        if len(points) < 2:
            raise Exception("You need more than 1 point to generate a path! You better provide 2 next time...\n")

        #Generate an evenly distributed path along the points provided
        self.path = []
        #This array will store the number of points that each stage has
        pointsByStage = np.zeros(len(points) - 1)

        for i, point in enumerate(points[1:]):

            prevPoint = points[i]

            distance = np.linalg.norm(point - prevPoint)
            nSteps = int(round(distance/self.settings["distStep"])) + 1

            #Add the trajectory from the previous point to this one to the path
            self.path = [*self.path, *np.linspace(prevPoint, point, nSteps)]

            pointsByStage[i] = nSteps
        
        self.path = np.array(self.path)

        #Get the number of points for each stage and the total number of points in the path
        totalPoints = int(pointsByStage.sum())
        iCorners = pointsByStage.cumsum()

        #Prepare the array that will store all the spectra
        self.spectra = np.zeros((totalPoints, self.settings["nE"]))

        #Copy selected WFSX into WFSX if it exists (denchar reads from .WFSX)
        shutil.copyfile(os.path.join(self.rootDir, '{}.selected.WFSX'.format(self.struct)),
            os.path.join(self.rootDir, '{}.WFSX'.format(self.struct) ) )

        tempFdf = os.path.join(self.rootDir, '{}STS.fdf'.format(self.struct))
        outputFile = os.path.join(self.rootDir, '{}.STS'.format(self.struct))

        #Denchar needs to be run from the directory where everything is stored
        cwd = os.getcwd()
        os.chdir(self.rootDir)

        for iPoint, point in enumerate(self.path):

            #Generate the appropiate input file

            #Copy the root fdf
            shutil.copyfile( self.settings["rootFdf"], tempFdf )

            #And then append flags for denchar
            with open(tempFdf, "a") as fh:
                fh.write(self._getdencharSTSfdf(point))

            #Do the STS calculation for the point
            os.system("denchar < {}".format(tempFdf))

            #Retrieve and save the output appropiately
            spectrum = np.loadtxt(outputFile)

            self.spectra[iPoint,:] = np.flip(spectrum[:,1])

        os.chdir(cwd)
        os.remove(outputFile)
        os.remove(tempFdf)

        #Update the values for the limits so that they are automatically set
        self.updateSettings(updateFig = False, zmin = 0, zmax = 0)

    def _setData(self):

        self.data = [{
            'type': 'heatmap',
            'z': self.spectra.T,
            #These limits determine the contrast of the image
            'zmin': self.settings["zmin"],
            'zmax': self.settings["zmax"],
            #Yaxis is the energy axis
            'y': np.linspace(*self.settings["Erange"], self.settings["nE"])}]

    def plotSTSpectra(spectra, path):

        Emin, Emax = -6, 1
        Ef = -4.18

        distances = np.zeros(len(path))
        for iStage, (stage, stageSpectra) in enumerate(zip(path, spectra)):

            if iStage == 0:
                spectraToPlot = stageSpectra
            else:    
                spectraToPlot = np.concatenate((spectraToPlot, stageSpectra))

            distances[iStage] = np.linalg.norm(stage[-1] - stage[0])

        fig = plt.figure()
        plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = True
        plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = False

        # Add funcionality
        def keypress(event, im):

            #Change saturation
            if event.key in ["up", "down"]:

                vmin, vmax = im.get_clim()
                factor = [0.99, 1.01][["up","down"].index(event.key)]
                im.set_clim(vmax=vmax*factor)
                fig.canvas.draw()

            if event.key == "r":

                data = im.get_array()
                im.set_clim(vmin=np.amin(data), vmax=np.amax(data))
                fig.canvas.draw()

            #Save file
            elif event.key == "t":

                data = im.get_array()
                [minX, maxX, minE, maxE] = im.get_extent()

                fileName = input("\nPlease provide a file name (or path) to save the data:  ")

                df = pd.DataFrame(data, np.linspace(minE, maxE, data.shape[0]), np.linspace(minX, maxX, data.shape[1]))

                with open(fileName, "w") as f:
                    f.write("#LDOS spectra for different positions\n")
                    f.write("#First column contains the energies (eV), first row contains the path coordinate (Ang)\n")
                    df.to_csv(f, float_format = "%.3e")
                print(f"Saved to {fileName}!\n")

        #Print the point in the material that corresponds to the clicked spot
        def onClick(event, path, distances):

            distances = np.cumsum(distances)
            if not event.xdata:
                return

            pathCoordinate = event.xdata

            for i, distance in enumerate(distances):

                if pathCoordinate < distance :

                    stage = path[i]
                    nPoints = len(stage)

                    if i > 0:
                        prevDistance = distances[i-1]
                    else:
                        prevDistance = 0.0

                    distStep = (distance - prevDistance)/(nPoints-1)
                    clickedPoint = stage[int(round((pathCoordinate - prevDistance)/distStep))]

                    print(f"\nClicked zone corresponds to {clickedPoint} in the material.\n")
                    break

        im = plt.imshow(spectraToPlot.T, extent=[0, distances.sum(), Emin - Ef, Emax - Ef], aspect = "auto", origin='upper')
        plt.xlabel("Path coordinate (Ang)")
        plt.ylabel("Energy (eV)")

        for i, _ in enumerate(distances[:-1]):
            plt.axvline(distances[0:i+1].sum(), color = "r")

        #Listen to events
        fig.canvas.mpl_connect('key_press_event', lambda event: keypress(event, im))
        fig.canvas.mpl_connect('button_press_event', lambda event: onClick(event, path, distances))

        print("\nINSTRUCTIONS\n.............")
        print("-  You can increase or decrease the saturation by pressing the up and down arrows (r to reset)")
        print('-  Press "t" to save the data to a file.')
        print('-  Click on a part of the image to get the point of the material to which it corresponds.')
        plt.show()

    def showSTS(struct, directory, templatePath):

        #Get directories to switch between them
        home = os.getcwd()

        os.chdir(directory)
        spectra, path = getSTSpectra(struct, templatePath)
        os.chdir(home)

        plotSTSpectra(spectra, path)