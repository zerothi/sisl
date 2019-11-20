'''
This file contains the Plot class, which should be inherited by all plot classes
'''
import time
import os
import plotly.graph_objects as go

import sisl

from .configurable import *

PLOTS_CONSTANTS = {
    "spins": ["up", "down"],
    "readFuncs": {
        "fromH": lambda obj: obj._readfromH, 
        "siesOut": lambda obj: obj._readSiesOut
    }
}

class Plot(Configurable):
    
    _onSettingsUpdate = {
        "__config":{
            "multipleFunc": False,
            "importanceOrder": ["readData", "setData", "getFigure"]
        },
        "readData": lambda obj: obj.readData,
        "setData": lambda obj: obj.setData,
        "getFigure": lambda obj: obj.getFigure,
    }
    
    _parameters = (
        
        {
            "key": "readingOrder",
            "name": "Output reading/generating order",
            "default": ("guiOut", "siesOut", "fromH"),
            "onUpdate": "readData"
        },

        {
            "key": "rootFdf",
            "name": "Path to fdf file",
            "default": None,
            "onUpdate": "readData"
        },

        {
            "key": "showLegend",
            "name": "Show Legend",
            "group": "layout",
            "default": True,
            "onUpdate": "getFigure",
        },

        {
            "key": "backgroundColor",
            "name": "Background color",
            "group": "layout",
            "default": "white",
            "onUpdate": "getFigure",
        },
        
        
        *[
            param for iAxis, axis in enumerate(["xaxis", "yaxis"]) for param in [
            
            {
              "key": "{}_title".format(axis),
              "name": "Title",
              "group": "layout",
              "subGroup": axis,
              "default": "",
              "onUpdate": "getFigure",
            },
            
            {
              "key": "{}_type".format(axis),
              "name": "Scale",
              "group": "layout",
              "subGroup": axis,
              "default": "linear",
              "onUpdate": "getFigure",
            },
            
            {
              "key": "{}_showline".format(axis),
              "name": "Show axis line",
              "group": "layout",
              "subGroup": axis,
              "default": False,
              "onUpdate": "getFigure",
            },
                
            {
              "key": "{}_linecolor".format(axis),
              "name": "Axis line color",
              "group": "layout",
              "subGroup": axis,
              "default": "black",
              "onUpdate": "getFigure",
            },
                
            {
              "key": "{}_linewidth".format(axis),
              "name": "Axis line width",
              "group": "layout",
              "subGroup": axis,
              "default": 1,
              "onUpdate": "getFigure",
            },
                
            {
              "key": "{}_visible".format(axis),
              "name": "Visible",
              "group": "layout",
              "subGroup": axis,
              "default": True,
              "onUpdate": "getFigure",
            },
                
            {
              "key": "{}_color".format(axis),
              "name": "Color",
              "group": "layout",
              "subGroup": axis,
              "default": "black",
              "onUpdate": "getFigure",
            },
            
            {
              "key": "{}_showgrid".format(axis),
              "name": "Show grid",
              "group": "layout",
              "subGroup": axis,
              "default": False,
              "onUpdate": "getFigure",
            },
                
            {
              "key": "{}_gridcolor".format(axis),
              "name": "Grid color",
              "group": "layout",
              "subGroup": axis,
              "default": "#ccc",
              "onUpdate": "getFigure",
            },
            
            {
              "key": "{}_zeroline".format(axis),
              "name": "Zero line",
              "group": "layout",
              "subGroup": axis,
              "default": [False, True][iAxis],
              "onUpdate": "getFigure",
            },
                
            {
              "key": "{}_zerolinecolor".format(axis),
              "name": "Zero line color",
              "group": "layout",
              "subGroup": axis,
              "default": "#ccc",
              "onUpdate": "getFigure",
            },
            
            {
              "key": "{}_tickcolor".format(axis),
              "name": "Tick color",
              "group": "layout",
              "subGroup": axis,
              "default": ["black", "white"][iAxis],
              "onUpdate": "getFigure",
            },
            
            {
              "key": "{}_ticklen".format(axis),
              "name": "Tick length",
              "group": "layout",
              "subGroup": axis,
              "default": 20,
              "onUpdate": "getFigure",
            }]
            
        ]
        
    )
    
    @afterSettingsInit
    def __init__(self, **kwargs):
        
        #Give an ID to the plot
        self.id = time.time()

        if self.settings["rootFdf"]:
            
            #Set the other relevant files
            self.setFiles()

            #Try to read the hamiltonian
            if "readHamiltonian" in kwargs.keys() and kwargs["readHamiltonian"]:
                try:
                    self.setupHamiltonian()
                except Exception:
                    log.warning("Unable to find or read {}.HSX".format(self.struct))
                    pass
            
            #Process data in the required files, optimally to build a dataframe that can be queried afterwards
            self.readData(**kwargs)
    
    def __str__(self):
        
        string = '''
    Plot class: {}    Plot type: {}
        
    Settings: 
    {}
        
        '''.format(
            self.__class__.__name__,
            self._plotType,
            "\n".join([ "\t- {}: {}".format(key,value) for key, value in self.settings.items()])
        )
        
        return string
    
    @afterSettingsUpdate
    def setFiles(self, **kwargs):
        '''
        Checks if the required files are available and then builds a list with them
        '''
        #Set the fdfSile
        rootFdf = self.settings["rootFdf"]
        self.rootDir, fdfFile = os.path.split( rootFdf )
        self.fdfSile = sisl.get_sile(rootFdf)
        self.struct = self.fdfSile.get("SystemLabel")
            
        #Check that the required files are there
        #if RequirementsFilter().check(self.rootFdf, self.__class__.__name__ ):
        if True:
            #If they are there, we can confidently build this list
            self.requiredFiles = [ os.path.join( self.rootDir, req.replace("$struct$", self.struct) ) for req in self.__class__._requirements["files"] ]
        else:
            log.error("\t the required files were not found, please check your file system.")
            raise Exception("The required files were not found, please check your file system.")

        return self
    
    @afterSettingsUpdate
    def setupHamiltonian(self, **kwargs):
        '''
        Sets up the hamiltonian for calculations with sisl.
        '''
        
        self.geom = self.fdfSile.read_geometry(output = True)

        #Try to read the hamiltonian in two different ways
        try:
            #This one is favoured because it may read from TSHS file, which contains all the information of the geometry and basis already
            self.H = self.fdfSile.read_hamiltonian()
        except Exception:
            Hsile = sisl.get_sile(os.path.join(self.rootDir, self.struct + ".HSX"))
            self.H = Hsile.read_hamiltonian(geom = self.geom)

        self.fermi = self.H.fermi_level()

        return self

    def _readFromSources(self):
        
        '''
        Tries to read the data from the different possible sources in the order 
        determined by self.settings["readingOrder"].
        '''
        
        errors = []
        #Try to read in the order specified by the user
        for source in self.settings["readingOrder"]:
            try:
                #Get the reading function
                readingFunc = PLOTS_CONSTANTS["readFuncs"][source](self)
                #Execute it
                data = readingFunc()
                self.source = source
                return data
            except Exception as e:
                errors.append("\t- {}: {}.{}".format(source, type(e).__name__, e))
                
        else:
            raise Exception("Could not read or generate data for {} from any of the possible sources.\n\n Here are the errors for each source:\n\n {}  "
                            .format(self.__class__.__name__, "\n".join(errors)) )
    
    @afterSettingsUpdate
    def getFigure(self, **kwargs):

        '''
        Define the plot object using the actual data. 
        
        This method can be applied after updating the data so that the plot object is refreshed.

        Returns
        ---------
        self.figure: go.Figure()
            the updated version of the figure.

        '''

        #Get all the settings that correspond to the layout of the axes
        axesParams = {}
        for axis in "xaxis","yaxis":
            
            axisParams = {setting["key"].split("_")[-1]: self.settings[setting["key"]] for setting in self.params if setting.get("subGroup", None) == axis}
            
            axesParams = {**axesParams, axis: axisParams}

        self.layout = {
            'title': '{} {}'.format(self.struct, self._plotType),
            'showlegend': self.settings["showLegend"],
            'hovermode': 'closest',
            'plot_bgcolor': self.settings["backgroundColor"],
            'xaxis' : {
                'tickvals': self.ticks[0],
                'ticktext': self.settings["ticks"].split(",") if self.source != "siesOut" else self.ticks[1],
                **axesParams["xaxis"]
                },
            'yaxis' : { 
                'range': self.settings["Erange"],
                **axesParams["yaxis"]
                },
        }
            
        self.figure = go.Figure({
            'data': [go.Scatter(**lineData) for lineData in self.data],
            'layout': self.layout,
        })
        
        return self.figure
    
    #-------------------------------------------
    #       PLOT MANIPULATION METHODS
    #-------------------------------------------

    def show(self):
        
        return self.figure.show()
    
    def merge(self, plotsToMerge, inplace = True, **kwargs):
        '''
        Merges this plot's instance with the list of plots provided (EXPERIMENTAL)
        '''
        
        #Make sure we deal with a list (user can provide a single plot)
        if not isinstance(plotsToMerge, list):
            plotsToMerge = [plotsToMerge]
            
        if inplace:
            for plot in plotsToMerge:
                self.data = [*self.data, *plot.data]
        
            self.getFigure()
            
            return self