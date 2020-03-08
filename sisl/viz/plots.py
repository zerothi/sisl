'''
This file contains all the plot subclasses
'''

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
import itertools

import os
import shutil

import sisl
from .plot import Plot, MultiplePlot, Animation, PLOTS_CONSTANTS
from .plotutils import sortOrbitals, initMultiplePlots, copyParams, findFiles, runMultiple, calculateGap
from .inputFields import InputField, TextInput, SwitchInput, ColorPicker, DropdownInput, IntegerInput, FloatInput, RangeSlider, QueriesInput

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

    @classmethod
    def _defaultAnimation(self, wdir = None, frameNames = None, **kwargs):
        
        bandsFiles = findFiles(wdir, "*.bands", sort = True)

        def _getFrameNames(self):

            return [os.path.basename( childPlot.setting("bandsFile")) for childPlot in self.childPlots]

        return BandsPlot.animated("bandsFile", bandsFiles, frameNames = _getFrameNames, wdir = wdir, **kwargs)

    def _afterInit(self):

        self.updateSettings(updateFig = False, xaxis_title = 'K', yaxis_title = "Energy (eV)")

    def _readfromH(self):

        if not hasattr(self, "H"):
            self.setupHamiltonian()
        
        #Get the path requested
        self.path = self.setting("path")
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
        self.path = self.setting("path") #This should be modified at some point, it's just so that setData works correctly

        bandsFile = self.setting("bandsFile") or self.requiredFiles[0]
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
        
            self.gap = calculateGap(spinComponentBands)
        
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
                        'y': band.loc[self.Ks] - self.fermi,
                        'mode': 'lines', 
                        'name': "{} spin {}".format( band["iBand"], PLOTS_CONSTANTS["spins"][int(band["iSpin"])]) if self.isSpinPolarized else str(int(band["iBand"])) , 
                        'line': {"color": [self.setting("spinUpColor"),self.setting("spinDownColor")][int(band["iSpin"])], 'width' : self.setting("bandsWidth")},
                        'hoverinfo':'name',
                        "hovertemplate": '%{y:.2f} eV',
                    } for i, band in self.plotDF.sort_values("iBand").iterrows() ] ]

    def _afterGetFigure(self):

        #Add the ticks
        self.figure.layout.xaxis.tickvals = self.ticks[0]
        self.figure.layout.xaxis.ticktext = self.ticks[1]
        self.figure.layout.yaxis.range = np.array(self.setting("Erange")) + self.fermi

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
        
        TextInput(
            key = "PDOSFile", name = "Path to PDOS file",
            width = "s100% m50% l33%",
            params = {
                "placeholder": "Write the path to your PDOS file here...",
            },
            help = '''This parameter explicitly sets a .PDOS file. Otherwise, the PDOS file is attempted to read from the fdf file '''
        ),

        RangeSlider(
            key = "Erange", name = "Energy range",
            default = [-2,4],
            width = "s100%",
            params = {
                "min": -10,
                "max": 10,
                "step": 0.1,
                "marks": { **{ i: str(i) for i in range(-10,11) }, 0: "Ef",}
            },
            help = "Energy range where the PDOS is displayed."
        ),

        QueriesInput(
            key = "requests", name = "PDOS queries",
            default = [{"active": True, "linename": "DOS", "species": None, "atoms": None, "orbitals": None, "spin": None, "normalize": False, "color": "black", "linewidth": 1}],
            help = '''Here you can ask for the specific PDOS that you need. 
                    <br>TIP: Queries can be activated and deactivated.''',
            queryForm = [

                TextInput(
                    key = "linename", name = "Name",
                    default = "DOS",
                    width = "s100% m50% l20%",
                    params = {
                        "placeholder": "Name of the line..."
                    },
                ),

                DropdownInput(
                    key = "species", name = "Species",
                    default = None,
                    width = "s100% m50% l40%",
                    params = {
                        "options":  [],
                        "isMulti": True,
                        "placeholder": "",
                        "isClearable": True,
                        "isSearchable": True,  
                    },
                ),

                DropdownInput(
                    key = "atoms", name = "Atoms",
                    default = None,
                    width = "s100% m50% l40%",
                    params = {
                        "options":  [],
                        "isMulti": True,
                        "placeholder": "",
                        "isClearable": True,
                        "isSearchable": True,  
                    },
                ),

                DropdownInput(
                    key = "orbitals", name = "Orbitals",
                    default = None,
                    width = "s100% m50% l50%",
                    params = {
                        "options":  [],
                        "isMulti": True,
                        "placeholder": "",
                        "isClearable": True,
                        "isSearchable": True,  
                    },
                ),

                DropdownInput(
                    key = "spin", name = "Spin",
                    default = None,
                    width = "s100% m50% l25%",
                    params = {
                        "options":  [],
                        "isMulti": False,
                        "placeholder": "",
                        "isClearable": True, 
                    },
                    style = {
                        "width": 200
                    }
                ),

                SwitchInput(
                    key = "normalize", name = "Normalize",
                    default = False,
                    params = {
                        "offLabel": "No",
                        "onLabel": "Yes"
                    }
                ),

                ColorPicker(
                    key = "color", name = "Line color",
                    default = None,
                ),

                FloatInput(
                    key = "linewidth", name = "Line width",
                    default = 1,
                )
            ]
        )

    )
    
    @classmethod
    def _defaultAnimation(self, wdir = None, frameNames = None, **kwargs):
        
        PDOSfiles = findFiles(wdir, "*.PDOS", sort = True)

        def _getFrameNames(self):

            return [os.path.basename( childPlot.setting("PDOSFile")) for childPlot in self.childPlots]

        return PdosPlot.animated("PDOSFile", bandsFiles, frameNames = _getFrameNames, wdir = wdir, **kwargs)

    def _afterInit(self):

        self.updateSettings(updateFig = False, xaxis_title = 'Density of states (1/eV)', yaxis_title = "Energy (eV)")

    def _readfromH(self):

        if not hasattr(self, "H") or self.PROVIDED_H:
            self.setupHamiltonian()

        #Calculate the pdos with sisl using the last geometry and the hamiltonian
        self.monkhorstPackGrid = [15, 1, 1]
        Erange = self.setting("Erange")
        self.E = np.linspace( Erange[0], Erange[-1], 1000) 

        mp = sisl.MonkhorstPack(self.H, self.monkhorstPackGrid)
        self.PDOSinfo = mp.asaverage().PDOS(self.E, eta=True)

    def _readSiesOut(self):

        PDOSFile = self.setting("PDOSFile") or self.requiredFiles[0]
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
        #First define the function that will modify all the fields of the query form
        def modifier(requestsInput):

            options = {
                "atoms": [{ "label": "{} ({})".format(iAt, self.geom.atoms[iAt - 1].symbol), "value": iAt } 
                    for iAt in self.df["iAtom"].unique()],
                "species": [{ "label": spec, "value": spec } for spec in self.df.Species.unique()],
                "orbitals": [{ "label": orbName, "value": orbName } for orbName in self.df["Orbital name"].unique()],
                "spin": [{ "label": "↑", "value": 0 },{ "label": "↓", "value": 1 }] if self.isSpinPolarized else []
            }

            for key, val in options.items():
                requestsInput.modifyQueryParam(key, "inputField.params.options", val)

        #And then apply it
        self.modifyParam("requests", modifier)

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
        Emin, Emax = np.array(self.setting("Erange"))
        plotEvals = [Evalue for Evalue in self.E if Emin < Evalue < Emax]

        #Inform and abort if there is no data
        if len(plotEvals) == 0:
            print("PDOS Plot error: There is no data for the provided energy range ({}).\n The energy range of the read data is: [{},{}]"
                .format(self.setting("Erange"), min(self.E), max(self.E))
            )

            return self.data

        #If there is data, get it (drop the columns that we don't want)
        self.reqDf = self.df.drop([Evalue for Evalue in self.E if Evalue not in plotEvals], axis = 1)

        #Go request by request and plot the corresponding PDOS contribution
        for request in self.setting("requests"):

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
                'line': {'width' : request["linewidth"], "color": request["color"]},
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

""" class PdosAnimation(Animation):
    
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
        self.params = copyParams( self.childPlots[0].params, exclude = ["PDOSFile"]) """

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
            help = "Energy range where the STS spectra are computed."
        ),

        IntegerInput(
            key = "nE", name = "Energy points",
            default = 100,
            params = {
                "min": 1
            },
            help = "The number of energy points that are calculated for each spectra"
        ),

        FloatInput(
            key = "STSEta", name = "Smearing factor (eV)",
            default = 0.05,
            params = {
                "min": 0.01,
                "step": 0.01
            },
            help = '''This determines the smearing factor of each STS spectra. You can play with this to modify sensibility in the vertical direction.
                <br> If the smearing value is too high, your map will have a lot of vertical noise'''
        ),

        FloatInput(
            key = "distStep", name = "Distance step (Ang)",
            default = 0.1,
            params = {
                "min": 0,
                "step": 0.01,
            },
            help = "The step in distance between one point and the next one in the path."
        ),

        InputField(
            key = "trajectory", name = "Trajectory",
            default = [],
            help = '''You can directly provide a trajectory instead of the corner points.<br>
                    This option has preference over 'points', but can't be used through the GUI.<br>
                    It is useful if you want a non-straight trajectory.'''
        ),

        InputField(
            key = "widenFunc", name = "Widen function",
            default = None,
            help = '''You can widen the path with this parameter. 
                    This option has preference over 'widenX', 'widenY' and 'widenZ', but can't be used through the GUI.<br>
                    This must be a function that gets a point of the path and returns a set of points surrounding it (including the point itself).<br>
                    All points of the path must be widened with the same amount of points, otherwise you will get an error.'''
        ),

        DropdownInput(
            key = "widenMethod", name = "Widen method",
            default = "sum",
            width = "s100% m50% l40%",
            params = {
                "options":  [{"label": "Sum", "value": "sum"}, {"label" : "Average", "value": "average"}],
                "isMulti": False,
                "placeholder": "",
                "isClearable": False,
                "isSearchable": True, 
            },
            help = "Determines whether values surrounding a point should be summed or averaged"
        ),

        QueriesInput(
            key = "points", name = "Path corners",
            default = [{"x": 0, "y": 0, "z": 0, "atom": None, "active": True}],
            queryForm = [

                *[FloatInput(
                    key = key, name = key.upper(),
                    default = 0,
                    width = "s30%",
                    params = {
                        "step" : 0.01
                    }
                ) for key in ("x", "y", "z")],

                DropdownInput(
                    key = "atom", name = "Atom index",
                    default = None,
                    params = {
                        "options":  [],
                        "isMulti": False,
                        "placeholder": "",
                        "isClearable": True,
                        "isSearchable": True, 
                    },
                    help = '''You can provide an atom index instead of the coordinates<br>
                    If an atom is provided, x, y and z will be interpreted as the supercell indices.<br>
                    That is: atom 23 [x=0,y=0,z=0] is atom 23 in the primary cell, while atom 23 [x=1,y=0,z=0]
                    is the image of atom 23 in the adjacent cell in the direction of x'''
                )
            ],
            help = '''Provide the points to generate the path through which STS need to be calculated.'''
        ),

        FloatInput(
            key = "cmin", name = "Lower color limit",
            default = 0,
            params = {
                "step": 10*-6
            },
            help = "All points below this value will be displayed as 0."
        ),

        FloatInput(
            key = "cmax", name = "Upper color limit",
            default = 0,
            params = {
                "step": 10*-6
            },
            help = "All points above this value will be displayed as the maximum.<br> Decreasing this value will increase saturation."
        ),
    
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
            '''.format(*stsPosition, *(np.array(self.setting("Erange")) + self.fermi), self.setting("nE"), self.setting("STSEta"))

    def _readSiesOut(self):
        '''Function that uses denchar to get STSpecra along a path'''

        self.geom = self.fdfSile.read_geometry(output = True)

        #Find fermi level
        self.fermi = False
        for outFileName in (self.struct, self.fdfSile.base_file.replace(".fdf", "")):
            try:
                for line in open(os.path.join(self.rootDir, "{}.out".format(outFileName)) ):
                    if "Fermi =" in line:
                        self.fermi = float(line.split()[-1])
                        print("\nFERMI LEVEL FOUND: {} eV\n Energies will be relative to this level (E-Ef)\n".format(self.fermi))
                break
            except FileNotFoundError:
                pass
        

        if not self.fermi:
            print("\nFERMI LEVEL NOT FOUND IN THE OUTPUT FILE. \nEnergy values will be absolute\n")
            self.fermi = 0

        #Get the path (this also sets some attributes: 'distances', 'pointsByStage', 'totalPoints')
        self._getPath()
        
        #Prepare the array that will store all the spectra
        self.spectra = np.zeros((self.path.shape[0], self.path.shape[1], self.setting("nE")))
        #Other helper arrays
        pathIs = np.linspace(0, self.path.shape[0] - 1, self.path.shape[0] )
        Epoints = np.linspace( *(np.array(self.setting("Erange")) + self.fermi), self.setting("nE") )

        #Copy selected WFSX into WFSX if it exists (denchar reads from .WFSX)
        shutil.copyfile(os.path.join(self.rootDir, '{}.selected.WFSX'.format(self.struct)),
            os.path.join(self.rootDir, '{}.WFSX'.format(self.struct) ) )
        
        #Get the fdf file and replace include paths so that they work
        with open(self.setting("rootFdf"), "r") as f:
            self.fdfLines = f.readlines()
        
        for i, line in enumerate(self.fdfLines):
            if "%include" in line and not os.path.isabs(line.split()[-1]):

                self.fdfLines[i] = "%include {}\n".format(os.path.join("../", line.split()[-1]))

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

                #Write the fdf
                with open(tempFdf, "w") as fh:
                    fh.writelines(kwargs["fdfLines"])
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
            self.setting("nE"),
            pathIs,
            self.rootDir, self.struct,
            #All the strings that need to be added to each file
            [ [self._getdencharSTSfdf(point) for point in points] for points in self.path ],
            kwargsList = {"rootFdf" : self.setting("rootFdf"), "fdfLines": self.fdfLines },
            messageFn = lambda nTasks, nodes: "Calculating {} simultaneous paths in {} nodes".format(nTasks, nodes),
            serial = self.isChildPlot
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
        self.updateSettings(updateFig = False, cmin = 0, cmax = 0)
    
    def _getPath(self):

        if list(self.setting("trajectory")):
            #If the user provides a trajectory, we are going to use that without questioning it
            self.path = np.array(self.setting("trajectory"))

            #At the moment these make little sense, but in the future there will be the possibility to add breakpoints
            self.pointsByStage = np.array([len(self.path)])
            self.distances = np.array( [np.linalg.norm(self.path[-1] - self.path[0])] )
        else:
            #Otherwise, we will calculate the trajectory according to the points provided
            points = []
            for reqPoint in self.setting("points"):

                if reqPoint.get("atom"):
                    translate = np.array([reqPoint["x"],reqPoint["y"],reqPoint["z"]]).dot(self.geom.cell)
                    points.append(self.geom[reqPoint["atom"]] + translate)
                else:
                    points.append([reqPoint["x"],reqPoint["y"],reqPoint["z"]])
            points = np.array(points)

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
                nSteps = int(round(self.distances[i]/self.setting("distStep"))) + 1

                #Add the trajectory from the previous point to this one to the path
                self.path = [*self.path, *np.linspace(prevPoint, point, nSteps)]

                self.pointsByStage[i] = nSteps
            
            self.path = np.array(self.path)
        
        #Then, let's widen the path if the user wants to do it (check also points that surround the path)
        if callable(self.setting("widenFunc")):
            self.path = self.setting("widenFunc")(self.path)
        else:
            #This is just to normalize path
            self.path = np.expand_dims(self.path, 0)
        
        #Store the total number of points of the path
        self.nPathPoints = self.path.shape[1]
        self.totalPoints = self.path.shape[0] * self.path.shape[1]
        self.iCorners = self.pointsByStage.cumsum()

    def _setData(self):

        #With xarray
        if self.setting("widenMethod") == "sum":
            spectraToPlot = self.xarr.sum(dim = "iPath")
        elif self.setting("widenMethod") == "average":
            spectraToPlot = self.xarr.mean(dim = "iPath")
        
        self.data = [{
            'type': 'heatmap',
            'z': spectraToPlot.transpose("E", "x").values,
            #These limits determine the contrast of the image
            'zmin': self.setting("cmin"),
            'zmax': self.setting("cmax"),
            #Yaxis is the energy axis
            'y': np.linspace(*self.setting("Erange"), self.setting("nE"))}]
    
class BondLengthMap(Plot):
    
    '''
    Colorful representation of bond lengths.
    '''

    _plotType = "Bond length"
    
    _requirements = {
        
    }
    
    _parameters = (
        
        SwitchInput(
            key = "geomFromOutput", name = "Geometry from output",
            default = True,
            group = "readdata",
            params = {
                "offLabel": "No",
                "onLabel": "Yes",
            },
            help = "In case the geometry is read from the fdf file, this will determine whether the input or the output geometry is taken.<br>This setting will be ignored if geomFile is passed"
        ),
        
        TextInput(
            key = "geomFile", name = "Path to the geometry file",
            group = "readdata",
            width = "s100% m50% l33%",
            params = {
                "placeholder": "Write the path to your geometry file here..."
            },
            help = '''This parameter explicitly sets a geometry file. Otherwise, the geometry is attempted to read from the fdf file '''
        ),

        TextInput(
            key = "strainRef", name = "Strain reference geometry",
            default = None,
            group = "readdata",
            width = "s100% m50% l33%",
            params = {
                "placeholder": "Write the path to your strain reference file here..."
            },
            help = '''The path to a geometry used to calculate strain from.<br>
            This geometry will probably be the relaxed one<br>
            If provided, colors can indicate strain values. Otherwise they are just bond length'''
        ),

        SwitchInput(
            key = "showStrain", name = "Bond display mode",
            default = True,
            params = {
                "offLabel": "Length",
                "onLabel": "Strain"
            },
            help = '''Determines whether, <b>IF POSSIBLE</b>, strain values should be displayed instead of lengths<br>
            If this is set to show strain, but no strain reference is set, <b>it will be ignored</b>
            '''
        ),
        
        FloatInput(
            key = "bondThreshold", name = "Bond length threshold",
            default = 1.7,
            params = {
                "step": 0.01
            },
            help = "Maximum distance between two atoms to draw a bond"
        ),
        
        TextInput(
            key = "cmap", name = "Plotly colormap",
            default = "solar",
            width = "s100% m50% l33%",
            params = {
                "placeholder": "Write a valid plotly colormap here..."
            },
            help = '''This determines the colormap to be used for the bond lengths display.<br>
            You can see all valid colormaps here: <a>https://plot.ly/python/builtin-colorscales/<a/><br>
            Note that you can reverse a color map by adding _r'''
        ),
        
        IntegerInput(
            key = "tileX", name = "Tile first axis",
            default = 1,
            params = {
                "min": 1
            },
            help = "Number of unit cells to display along the first axis"
        ),
        
        IntegerInput(
            key = "tileY", name = "Tile second axis",
            default = 1,
            params = {
                "min": 1
            },
            help = "Number of unit cells to display along the second axis"
        ),
        
        IntegerInput(
            key = "tileZ", name = "Tile third axis",
            default = 1,
            params = {
                "min": 1
            },
            help = "Number of unit cells to display along the third axis"
        ),
            
        DropdownInput(
            key = "xAxis", name = "Coordinate in X axis",
            default = "X",
            width = "s100% m50% l33%",
            params = {
                "placeholder": "Choose the coordinate of the X axis...",
                "options": [
                    {"label": ax, "value": ax} for ax in ("X", "Y", "Z")
                ],
                "isClearable": False,
                "isSearchable": True,
            },
            help = "This is the coordinate that will be shown in the X axis of the plot "
        ),

        DropdownInput(
            key = "yAxis", name = "Coordinate in Y axis",
            default = "Y",
            width = "s100% m50% l33%",
            params = {
                "placeholder": "Choose the coordinate of the Y axis...",
                "options": [
                    {"label": ax, "value": ax} for ax in ("X", "Y", "Z")
                ],
                "isClearable": False,
                "isSearchable": True,
            },
            help = "This is the coordinate that will be shown in the Y axis of the plot "
        ),
        
        FloatInput(
            key = "cmin", name = "Color scale low limit",
            default = 0,
            params = {
                "step": 0.01
            }
        ),
        
        FloatInput(
            key = "cmax", name = "Color scale high limit",
            default = 0,
            params = {
                "step": 0.01
            }
        ),

        FloatInput(
            key = "cmid", name = "Color scale mid point",
            default = None,
            params = {
                "step": 0.01
            },
            help = '''Sets the middle point of the color scale. Only meaningful in diverging colormaps<br>
            If this is set 'cmin' and 'cmax' are ignored. In strain representations this might be set to 0.
            '''
        ),
        
        IntegerInput(
            key = "pointsPerBond", name = "Points per bond",
            default = 5,
            help = "Number of points that fill a bond <br>More points will make it look more like a line but will slow plot rendering down."
        )
    
    )
    
    @classmethod
    def _defaultAnimation(self, wdir = None, frameNames = None, **kwargs):
        
        geomsFiles = findFiles(wdir, "*.XV", sort = True)

        #def _getFrameNames(self):

            #return [os.path.basename( childPlot.setting("bandsFile")) for childPlot in self.childPlots]

        return BondLengthMap.animated("geomFile", geomsFiles, wdir = wdir, **kwargs)

    def _afterInit(self):

        self.updateSettings(updateFig = False, xaxis_title = 'X (Ang)', yaxis_title = "Y (Ang)", yaxis_zeroline = False)
        pass

    def _readSiesOut(self):
        
        if self.setting("geomFile"):
            self.geom = sisl.get_sile(self.setting("geomFile")).read_geometry()
        else:
            self.geom = sisl.get_sile(self.setting("rootFdf")).read_geometry(output = self.setting("geomFromOutput"))
        
        self.isStrain = False
        if self.setting("strainRef"):
            self.relaxedGeom = sisl.get_sile(self.setting("strainRef")).read_geometry()
            self.isStrain = True

            self.relaxedGeom.set_nsc([3,3,3])

        
        #If there isn't a supercell in all directions define it
        self.geom.set_nsc([3,3,3])
        
        #Build the dataframe with all the bonds info
        dfKeys = ("From", "To", "From Species", "To Species", "Bond Length",
            "initX", "initY", "initZ", "finalX", "finalY", "finalZ")
        strainKeys = ("Relaxed Length", "Strain") if self.isStrain else ()

        dfKeys = (*dfKeys, *strainKeys)
        bondsDict = { key: [] for key in dfKeys}

        for at in self.geom:

            #If there is a strain reference we take the neighbors of each atom from it
            if self.isStrain:
                geom = self.relaxedGeom
            else:
                geom = self.geom
            
            _, neighs = geom.close(at, R = (0.1, self.setting("bondThreshold")))

            for neigh in neighs:

                bondsDict["From"].append(at)
                bondsDict["To"].append(neigh)
                bondsDict["From Species"].append(self.geom.atoms[at].symbol)
                bondsDict["To Species"].append(self.geom.atom[neigh % self.geom.na].symbol)
                bondsDict["Bond Length"].append(np.linalg.norm(self.geom[at] - self.geom[neigh]))

                if self.isStrain:
                    relLength = np.linalg.norm(self.relaxedGeom[at] - self.relaxedGeom[neigh])
                    bondsDict["Relaxed Length"].append(relLength)
                    bondsDict["Strain"].append( (bondsDict["Bond Length"][-1] - relLength)/relLength )

                bondsDict["initX"].append(self.geom[at][0])
                bondsDict["initY"].append(self.geom[at][1])
                bondsDict["initZ"].append(self.geom[at][2])
                bondsDict["finalX"].append(self.geom[neigh][0])
                bondsDict["finalY"].append(self.geom[neigh][1])
                bondsDict["finalZ"].append(self.geom[neigh][2])

        

        self.df = pd.DataFrame(bondsDict)
    
    def _setData(self):
        
        """ #Define a colormap
        cmap = plt.cm.get_cmap(self.setting("cmap"))
        
        #Get the normalizer
        cmin = self.setting("cmin") or self.df["Bond Length"].min()
        cmax = self.setting("cmax") or self.df["Bond Length"].max()
        norm = matplotlib.colors.Normalize(cmin, cmax) """
        
        self.data = []
        tileCombs = itertools.product(*[range(self.setting(tile)) for tile in ("tileX", "tileY", "tileZ")])
        pointsPerBond = self.setting("pointsPerBond")
        self.showStrain = self.isStrain and self.setting("showStrain")
        colorColumn = "Strain" if self.showStrain else "Bond Length"
        xAxis = self.setting("xAxis"); yAxis = self.setting("yAxis")
        
        for tiles in tileCombs :
            
            #Get the translation vector
            translate = np.array(tiles).dot(self.geom.cell)
        
            #Draw bonds
            self.data = [*self.data, *[{
                            'type': 'scatter',
                            'x': np.linspace(bond["init{}".format(xAxis)], bond["final{}".format(xAxis)], pointsPerBond) + translate[["X","Y","Z"].index(xAxis)],
                            'y': np.linspace(bond["init{}".format(yAxis)], bond["final{}".format(yAxis)], pointsPerBond) + translate[["X","Y","Z"].index(yAxis)],
                            'mode': 'markers', 
                            'name': "{}{}-{}{}".format(bond["From Species"], bond["From"], bond["To Species"], bond["To"]), 
                            #'line': {"color": "rgba{}".format(cmap(norm(bond["Bond Length"])) ), "width": 3},
                            'marker': {
                                "size": 3, 
                                "color": [bond[colorColumn]]*pointsPerBond, 
                                "coloraxis": "coloraxis"
                            },
                            "showlegend": False,
                            'hoverinfo': "name",
                            'hovertemplate':'{:.2f} Ang{}'.format(bond["Bond Length"], ". Strain: {:.3f}".format(bond["Strain"]) if self.isStrain else "" ),
                        } for i, bond in self.df.iterrows() ]]

    def _afterGetFigure(self):

        #Add the ticks
        self.figure.layout.yaxis.scaleratio = 1
        self.figure.layout.yaxis.scaleanchor = "x"
        
        colorColumn = "Strain" if self.showStrain else "Bond Length"
        cmap = self.setting("cmap")
        reverse = "_r" in cmap
        cmap = cmap[:-2] if reverse else cmap
        self.figure.update_layout(coloraxis = {
            'colorbar': {
                'title': "Strain" if self.showStrain else "Length (Ang)"
            },
            'colorscale': cmap,
            'reversescale': reverse ,
            "cmin": (self.setting("cmin") or self.df[colorColumn].min()) if self.setting("cmid") == None else None,
            "cmax": (self.setting("cmax") or self.df[colorColumn].max()) if self.setting("cmid") == None else None,
            "cmid": self.setting("cmid"),
        })
        
        self.updateSettings(updateFig = False, xaxis_title = 'X (Ang)', yaxis_title = "Y (Ang)")