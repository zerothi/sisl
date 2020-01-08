'''
This file contains all the plot subclasses
'''

import numpy as np
import pandas as pd
import xarray as xr
import multiprocessing as mp
import itertools
from copy import deepcopy
import tqdm

import os
import shutil

import sisl
from .plot import Plot, MultiplePlot, Animation, PLOTS_CONSTANTS
from .plotutils import sortOrbitals, initMultiplePlots, copyParams, findFiles, runMultiple

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

        #If the path has changed we need to produce the band structure again
        if self.path != self.settings["path"]:
            self.order = ["fromH"]
            self.readData()

        self.data = []

        Erange = np.array(self.settings["Erange"]) + self.fermi

        #Get the bands that matter for the plot
        self.plotDF = self.df[ (self.df["Emin"] <= Erange[1]) & (self.df["Emax"] >= Erange[0]) ].dropna(axis = 0, how = "all")

        #Define the data of the plot as a list of dictionaries {x, y, 'type', 'name'}
        self.data = [ *self.data, *[{
                        'type': 'scatter',
                        'x': self.Ks,
                        'y': band.loc[self.Ks] - self.fermi,
                        'mode': 'lines', 
                        'name': "{} spin {}".format( band["iBand"], PLOTS_CONSTANTS["spins"][int(band["iSpin"])]) if self.isSpinPolarized else str(int(band["iBand"])) , 
                        'line': {"color": [self.settings["spinUpColor"],self.settings["spinDownColor"]][int(band["iSpin"])], 'width' : self.settings["bandsWidth"]},
                        'hoverinfo':'name',
                        "hovertemplate": '%{y:.2f} eV',
                    } for i, band in self.plotDF.sort_values("iBand").iterrows() ] ]

    def _afterGetFigure(self):

        #Add the ticks
        self.figure.layout.xaxis.tickvals = self.ticks[0]
        self.figure.layout.xaxis.ticktext = self.ticks[1]
        self.figure.layout.yaxis.range = np.array(self.settings["Erange"]) + self.fermi

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

    _requirements = {
        "siesOut": {
            "files": ["$struct$.PDOS"]
        }
    }

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
        orbitals = self.geom.orbitals.cumsum()[:-1]
        
        #Normalize self.PDOSinfo to do the same treatment for both spin-polarized and spinless simulations
        self.isSpinPolarized = len(self.PDOSinfo.shape) == 3

        #Initialize the dataframe to store all the info
        self.df = pd.DataFrame()

        #Normalize the PDOSinfo array
        self.PDOSinfo = [self.PDOSinfo] if not self.isSpinPolarized else self.PDOSinfo 

        #Loop over all spin components
        for iSpin, spinComponentPDOS in enumerate(self.PDOSinfo):

            #Initialize the dictionary with all the properties of the orbital
            orbProperties = {
                "iAtom": [],
                "Species": [],
                "Atom Z": [],
                "Spin": [],
                "Orbital name": [],
                "Z shell": [],
                "n": [],
                "l": [],
                "m": [],
                "Polarized": [],
                "Initial charge": [],
            }     

            #Loop over all orbitals of the basis
            for iAt, iOrb in self.geom.iter_orbitals():

                atom = self.geom.atoms[iAt]
                orb = atom[iOrb]

                orbProperties["iAtom"].append(iAt + 1)
                orbProperties["Species"].append(atom.symbol)
                orbProperties["Atom Z"].append(atom.Z)
                orbProperties["Spin"].append(iSpin)
                orbProperties["Orbital name"].append(orb.name())
                orbProperties["Z shell"].append(orb.Z)
                orbProperties["n"].append(orb.n)
                orbProperties["l"].append(orb.l)
                orbProperties["m"].append(orb.m)
                orbProperties["Polarized"].append(orb.P)
                orbProperties["Initial charge"].append(orb.q0)
            
            #Append this part of the dataframe (a full spin component)
            self.df = self.df.append( pd.concat([pd.DataFrame(orbProperties), pd.DataFrame(spinComponentPDOS, columns = self.E)], axis=1, sort = False), ignore_index = True)
        
        #"Inform" the queries of the available options
        for i, param in enumerate(self.params):

            if param["key"] == "requests":
                for iParam, reqParam in enumerate(self.params[i]["inputField"]["queryForm"]):

                    options = []
                    
                    if reqParam["key"] == "atoms":
                        
                        options = [{ "label": "{} ({})".format(iAt, self.geom.atoms[iAt - 1].symbol), "value": iAt } 
                            for iAt in self.df["iAtom"].unique()]
                        
                
                    elif reqParam["key"] == "species":
                        
                        options = [{ "label": spec, "value": spec } for spec in self.df.Species.unique()]
                        
                    
                    elif reqParam["key"] == "orbitals":
                        
                        options = [{ "label": orbName, "value": orbName } for orbName in self.df["Orbital name"].unique()]
                        
                    
                    elif reqParam["key"] == "spin":

                        options = [{ "label": "↑", "value": 0 },{ "label": "↓", "value": 1 }] if self.isSpinPolarized else []
                        

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

        #Initialize the data object for the plotly graph
        self.data = []

        #Get only the energies we are interested in 
        Emin, Emax = np.array(self.settings["Erange"])
        plotEvals = [Evalue for Evalue in self.E if Emin < Evalue < Emax]

        #Inform and abort if there is no data
        if len(plotEvals) == 0:
            print("PDOS Plot error: There is no data for the provided energy range ({}).\n The energy range of the read data is: [{},{}]"
                .format(self.settings["Erange"], min(self.E), max(self.E))
            )

            return self.data

        #If there is data, get it (drop the columns that we don't want)
        self.reqDf = self.df.drop([Evalue for Evalue in self.E if Evalue not in plotEvals], axis = 1)

        #Go request by request and plot the corresponding PDOS contribution
        for request in self.settings["requests"]:

            #Use only the active requests
            if not request["active"]:
                continue

            reqDf = self.reqDf.copy()

            if request.get("atoms"):
                reqDf = reqDf[reqDf["iAtom"].isin(request["atoms"])]

            if request.get("species"):
                reqDf = reqDf[reqDf["Species"].isin(request["species"])]

            if request.get("orbitals"):
                reqDf = reqDf[reqDf["Orbital name"].isin(request["orbitals"])]

            if request.get("spin") != None:
                reqDf = reqDf[reqDf["Spin"].isin(request["spin"])]


            if reqDf.empty:
                print("PDOS Plot warning: No PDOS for the following request: {}".format(request.params))
                PDOS = []
            else:
                PDOS = reqDf[plotEvals].values

                if request.get("normalize"):
                    PDOS = PDOS.mean(axis = 0)
                else:
                    PDOS = PDOS.sum(axis = 0)   
            
            self.data.append({
                'type': 'scatter',
                'x': PDOS,
                'y': plotEvals ,
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

        request = {"active": True, "linename": len(self.settings["requests"]), **newReq, **kwargs}

        try:
            self.updateSettings(requests = [*self.settings["requests"], request ])
        except Exception as e:
            print("There was a problem with your new request ({}): \n\n {}".format(request, e))
            self.undoSettings()

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
        "siesOut": {
            "files": ["$struct$.DIM", "$struct$.PLD", "*.ion" , "$struct$.selected.WFSX"],
            "codes": {

                "denchar": {
                    "reason": "The 'denchar' code is used in this case to generate STS spectra."
                }

            }
        },
        
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
            "key": "trajectory",
            "name": "Trajectory",
            "default": [],
            "help": '''You can directly provide a trajectory instead of the corner points.<br>
                    This option has preference over 'points', but can't be used through the GUI.<br>
                    It is useful if you want a non-straight trajectory.''',
            "onUpdate": "readData", 
        },

        {
            "key": "widenFunc",
            "name": "Widen function",
            "default": None,
            "help": '''You can widen the path with this parameter. 
                    This option has preference over 'widenX', 'widenY' and 'widenZ', but can't be used through the GUI.<br>
                    This must be a function that gets a point of the path and returns a set of points surrounding it (including the point itself).<br>
                    All points of the path must be widened with the same amount of points, otherwise you will get an error.''',
            "onUpdate": "readData", 
        },

        {
            "key" : "widenMethod",
            "name" : "Widen Method",
            "default": "sum",
            "inputField": {
                "type": "dropdown",
                "width": "s100% m50% l40%",
                "params": {
                    "options":  [{"label": "Sum", "value": "sum"}, {"label" : "Average", "value": "average"}],
                    "isMulti": True,
                    "placeholder": "",
                    "isClearable": True,
                    "isSearchable": True,    
                },
            },
            "help": "Determines whether values surrounding a point should be summed or averaged",
            "onUpdate": "setData",
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

        #Get the path (this also sets some attributes: 'distances', 'pointsByStage', 'totalPoints')
        self._getPath()
        
        #Prepare the array that will store all the spectra
        self.spectra = np.zeros((self.path.shape[0], self.path.shape[1], self.settings["nE"]))
        #Other helper arrays
        pathIs = np.linspace(0, self.path.shape[0] - 1, self.path.shape[0] )
        Epoints = np.linspace( *(np.array(self.settings["Erange"]) + self.fermi), self.settings["nE"] )

        #Copy selected WFSX into WFSX if it exists (denchar reads from .WFSX)
        shutil.copyfile(os.path.join(self.rootDir, '{}.selected.WFSX'.format(self.struct)),
            os.path.join(self.rootDir, '{}.WFSX'.format(self.struct) ) )

        #Denchar needs to be run from the directory where everything is stored
        cwd = os.getcwd()
        os.chdir(self.rootDir)

        def getSpectraForPath(argsTuple):

            path, nE, iPath, rootDir, struct, STSflags, args, kwargs = argsTuple

            #Generate a temporal directory so that we don't interfere with the other processes
            tempDir = "{}tempSTS".format(iPath)

            os.makedirs(tempDir, exist_ok = True)
            os.chdir(tempDir)

            tempFdf = os.path.join('{}STS.fdf'.format(struct))
            outputFile = os.path.join('{}.STS'.format(struct))

            #Link all the needed files to this directory
            os.system("ln -s ../*fdf ../*out ../*ion* ../*WFSX ../*DIM ../*PLD . ")

            spectra = []; failedPoints = 0

            for i, point in enumerate(path):

                #Generate the appropiate input file

                #Copy the root fdf
                shutil.copyfile( os.path.basename(kwargs["rootFdf"]), tempFdf )

                #And then append flags for denchar
                with open(tempFdf, "a") as fh:
                    fh.write(STSflags[i])
                    
                #Do the STS calculation for the point
                os.system("denchar < {} > /dev/null".format(tempFdf))

                if i%100 == 0 and i != 0:
                    print("PATH {}. Points calculated: {}".format(int(iPath), i))

                #Retrieve and save the output appropiately
                try:
                    spectrum = np.loadtxt(outputFile)

                    spectra.append(spectrum[:,1])
                except Exception as e:
                    
                    print("Error calculating the spectra for point {}: \n{}".format(point, e))
                    failedPoints += 1
                    #If any spectrum was read, just fill it with zeros
                    spectra.append(np.zeros(nE))
            
            if failedPoints:
                print("Path {} finished with {} error{} ({}/{} points succesfully calculated)".format(int(iPath), failedPoints, "s" if failedPoints > 1 else "", len(path) - failedPoints, len(path)))

            
            os.chdir("..")
            shutil.rmtree(tempDir, ignore_errors=True)

            return spectra

        self.spectra = runMultiple(
            getSpectraForPath,
            self.path,
            self.settings["nE"],
            pathIs,
            self.rootDir, self.struct,
            #All the strings that need to be added to each file
            [ [self._getdencharSTSfdf(point) for point in points] for points in self.path ],
            kwargsList = {"rootFdf" : self.settings["rootFdf"]},
            messageFn = lambda nTasks, nodes: "Calculating {} simultaneous paths in {} nodes".format(nTasks, nodes)
        )

        self.spectra = np.array(self.spectra)

        #WITH XARRAY
        self.xarr = xr.DataArray(
            name = "LDOSmap",
            data = self.spectra,
            dims = ["iPath", "x", "E"],
            coords = [pathIs, list(range(self.path.shape[1])), Epoints] 
        )

        os.chdir(cwd)
        
        #Update the values for the limits so that they are automatically set
        self.updateSettings(updateFig = False, zmin = 0, zmax = 0)
    
    def _getPath(self):

        if list(self.settings["trajectory"]):
            #If the user provides a trajectory, we are going to use that without questioning it
            self.path = np.array(self.settings["trajectory"])

            #At the moment these make little sense, but in the future there will be the possibility to add breakpoints
            self.pointsByStage = np.array([len(self.path)])
            self.distances = np.array( [np.linalg.norm(self.path[-1] - self.path[0])] )
        else:
            #Otherwise, we will calculate the trajectory according to the points provided
            points = np.array([[point["x"],point["y"],point["z"]] for point in self.settings["points"] if point["active"]])

            nCorners = len(points)
            if nCorners < 2:
                raise Exception("You need more than 1 point to generate a path! You better provide 2 next time...\n")

            #Generate an evenly distributed path along the points provided
            self.path = []
            #This array will store the number of points that each stage has
            self.pointsByStage = np.zeros(nCorners - 1)
            self.distances = np.zeros(nCorners - 1)

            for i, point in enumerate(points[1:]):

                prevPoint = points[i]

                self.distances[i] = np.linalg.norm(point - prevPoint)
                nSteps = int(round(self.distances[i]/self.settings["distStep"])) + 1

                #Add the trajectory from the previous point to this one to the path
                self.path = [*self.path, *np.linspace(prevPoint, point, nSteps)]

                self.pointsByStage[i] = nSteps
            
            self.path = np.array(self.path)
        
        #Then, let's widen the path if the user wants to do it (check also points that surround the path)
        if callable(self.settings["widenFunc"]):
            self.path = self.settings["widenFunc"](self.path)
        else:
            #This is just to normalize path
            self.path = np.expand_dims(self.path, 0)
        
        #Store the total number of points of the path
        self.nPathPoints = self.path.shape[1]
        self.totalPoints = self.path.shape[0] * self.path.shape[1]
        self.iCorners = self.pointsByStage.cumsum()

    def _setData(self):

        #With xarray
        if self.settings["widenMethod"] == "sum":
            spectraToPlot = self.xarr.sum(dim = "iPath")
        elif self.settings["widenMethod"] == "average":
            spectraToPlot = self.xarr.mean(dim = "iPath")
        
        self.data = [{
            'type': 'heatmap',
            'z': spectraToPlot.transpose("E", "x").values,
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