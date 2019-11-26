'''
This file contains the Plot class, which should be inherited by all plot classes
'''
import uuid
import os
import numpy as np
import json

import plotly
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

    _paramGroups = (

        {
            "key": "dataread",
            "name": "Data reading settings",
            "icon": "import_export",
            "description": "In such a busy world, one may forget how the files are structured in their computer. Please take a moment to <b>make sure your data is being read exactly in the way you expect<b>."
        },

        {
            "key": "layout",
            "name": "Layout settings",
            "icon": "format_paint",
            "subGroups":[
                {"key": "xaxis", "name": "X axis"},
                {"key": "yaxis", "name": "Y axis"}
            ],
            "description": "Data may loose its value if it is not well presented. Play with this parameters to <b>make your plot as beautiful and easy to understand as you can</b>."
        },

        {
            "key": None,
            "name": "Other settings",
            "icon": "settings",
            "description": "Here are some unclassified settings. Even if they don't belong to any group, they might still be important :) They may be here just because the developer was too lazy to categorize them or forgot to do so. <b>If you are the developer</b> and it's the first case, <b>shame on you<b>."
        }
    )
    
    _parameters = (
        
        {
            "key": "readingOrder",
            "name": "Output reading/generating order",
            "group": "dataread",
            "default": ("guiOut", "siesOut", "fromH"),
            "help": "Order in which the plot tries to read the data it needs.",
            "onUpdate": "readData",
            
        },

        {
            "key": "rootFdf",
            "name": "Path to fdf file",
            "group": "dataread",
            "inputField": {
                "type": "textinput",
                "width": "s100%",
                "params": {
                    "placeholder": "Write the path here...",
                }
            },
            "default": None,
            "help": "Path to the fdf file that is the 'parent' of the results.",
            "onUpdate": "readData"
        },

        {
            "key": "resultsPath",
            "name": "Path to your results",
            "group": "dataread",
            "inputField": {
                "type": "textinput",
                "width": "s100% m50% l33%",
                "params": {
                    "placeholder": "Write the path here...",
                }
            },
            "default": ".",
            "help": "Directory where the files with the simulations results are located.<br> This path has to be relative to the root fdf.",
            "onUpdate": "readData"
        },

        {
            "key": "showlegend",
            "name": "Show Legend",
            "group": "layout",
            "default": True,
            "inputField": {
                "type": "switch",
                "width": "s50% m30% l15%",
                "params": {
                    "offLabel": "No",
                    "onLabel": "Yes"
                }
            },
            "onUpdate": "getFigure",
        },

        {
            "key": "paper_bgcolor",
            "name": "Figure color",
            "group": "layout",
            "default": "white",
            "inputField": {
                "type": "color",
                "width": "s50% m30% l15%",
            },
            "onUpdate": "getFigure",
        },

        {
            "key": "plot_bgcolor",
            "name": "Plot color",
            "group": "layout",
            "default": "white",
            "inputField": {
                "type": "color",
                "width": "s50% m30% l15%",
            },
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
              "inputField": {
                    "type": "textinput",
                    "width": "s100% m50%",
                    "params": {
                        "placeholder": "Write the axis title...",
                    }
                },
              "onUpdate": "getFigure",
            },
            
            {
              "key": "{}_type".format(axis),
              "name": "Type",
              "group": "layout",
              "subGroup": axis,
              "default": "-",
              "inputField": {
                    "type": "dropdown",
                    "width": "s100% m50% l33%",
                    "params": {
                        "placeholder": "Choose the axis scale...",
                        "options": [
                            {"label": "Automatic", "value": "-"},
                            {"label": "Linear", "value": "linear"},
                            {"label": "Logarithmic", "value": "log"},
                            {"label": "Date", "value": "date"},
                            {"label": "Category", "value": "category"},
                            {"label": "Multicategory", "value": "multicategory"}
                        ],
                        "isClearable": False,
                        "isSearchable": False,
                    }
                },
              "onUpdate": "getFigure",
            },

            {
              "key": "{}_visible".format(axis),
              "name": "Visible",
              "group": "layout",
              "subGroup": axis,
              "default": True,
              "inputField": {
                    "type": "switch",
                    "width": "s50% m50% l25%",
                    "params": {
                        "offLabel": "No",
                        "onLabel": "Yes"
                    }
                },
              "onUpdate": "getFigure",
            },
                
            {
              "key": "{}_color".format(axis),
              "name": "Color",
              "group": "layout",
              "subGroup": axis,
              "default": "black",
              "inputField": {
                    "type": "color",
                    "width": "s50% m50% l25%",
                },
              "onUpdate": "getFigure",
            },
            
            {
              "key": "{}_showgrid".format(axis),
              "name": "Show grid",
              "group": "layout",
              "subGroup": axis,
              "default": False,
              "inputField": {
                    "type": "switch",
                    "width": "s50% m50% l25%",
                    "params": {
                        "offLabel": "No",
                        "onLabel": "Yes"
                    }
                },
              "onUpdate": "getFigure",
            },
                
            {
              "key": "{}_gridcolor".format(axis),
              "name": "Grid color",
              "group": "layout",
              "subGroup": axis,
              "default": "#ccc",
              "inputField": {
                    "type": "color",
                    "width": "s50% m50% l25%",
                },
              "onUpdate": "getFigure",
            },

            {
              "key": "{}_showline".format(axis),
              "name": "Show axis line",
              "group": "layout",
              "subGroup": axis,
              "default": False,
              "inputField": {
                "type": "switch",
                "width": "s50% m30% l30%",
                "params": {
                    "offLabel": "No",
                    "onLabel": "Yes"
                }
            },
              "onUpdate": "getFigure",
            },

            {
              "key": "{}_linewidth".format(axis),
              "name": "Axis line width",
              "group": "layout",
              "subGroup": axis,
              "default": 1,
              "inputField": {
                    "type": "number",
                    "width": "s50% m30% l30%",
                    "params": {
                        "min": 0,
                        "step": 0.1
                    }
                },
              "onUpdate": "getFigure",
            },
                
            {
              "key": "{}_linecolor".format(axis),
              "name": "Axis line color",
              "group": "layout",
              "subGroup": axis,
              "default": "black",
              "inputField": {
                    "type": "color",
                    "width": "s50% m30% l30%",
                },
              "onUpdate": "getFigure",
            },
            
            {
              "key": "{}_zeroline".format(axis),
              "name": "Zero line",
              "group": "layout",
              "subGroup": axis,
              "default": [False, True][iAxis],
              "inputField": {
                    "type": "switch",
                    "width": "s50% m30% l15%",
                    "params": {
                        "offLabel": "Hide",
                        "onLabel": "Show"
                    }
                },
              "onUpdate": "getFigure",
            },
                
            {
              "key": "{}_zerolinecolor".format(axis),
              "name": "Zero line color",
              "group": "layout",
              "subGroup": axis,
              "default": "#ccc",
              "inputField": {
                    "type": "color",
                    "width": "s50% m30% l15%",
                },
              "onUpdate": "getFigure",
            },

            {
              "key": "{}_ticks".format(axis),
              "name": "Ticks position",
              "group": "layout",
              "subGroup": axis,
              "default": "outside",
              "inputField": {
                    "type": "dropdown",
                    "width": "s100% m50% l33%",
                    "params": {
                        "placeholder": "Choose the ticks positions...",
                        "options": [
                            {"label": "Outside", "value": "outside"},
                            {"label": "Inside", "value": "Inside"},
                            {"label": "No ticks", "value": ""},
                        ],
                        "isClearable": False,
                        "isSearchable": False,
                    }
                },
              "onUpdate": "getFigure",
            },
            
            {
              "key": "{}_tickcolor".format(axis),
              "name": "Tick color",
              "group": "layout",
              "subGroup": axis,
              "default": "white",
              "inputField": {
                    "type": "color",
                    "width": "s50% m30% l15%",
                },
              "onUpdate": "getFigure",
            },
            
            {
              "key": "{}_ticklen".format(axis),
              "name": "Tick length",
              "group": "layout",
              "subGroup": axis,
              "default": 5,
              "inputField": {
                    "type": "number",
                    "width": "s50% m30% l15%",
                    "params": {
                        "min": 0,
                        "step": 0.1
                    }
                },
              "onUpdate": "getFigure",
            }]
            
        ]
        
    )
    
    @afterSettingsInit
    def __init__(self, **kwargs):
        
        #Give an ID to the plot
        self.id = str(uuid.uuid4())

        #Try to generate the figure (if the settings required are still not there, it won't be generated)
        try:
            self.readData()
        except Exception:
            pass
    
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
    def readData(self, updateFig = True, **kwargs):
        '''
        Gets the information for the bands plot and stores it into self.df

        Returns
        -----------
        dataRead: boolean
            whether data has been read succesfully or not
        '''

        if callable( getattr(self, "_beforeRead", None )):
            self._beforeRead()

        self.setFiles()
        
        #We try to read from the different sources using the _readFromSources method of the parent Plot class.
        result = self._readFromSources()

        if callable( getattr(self, "_afterRead", None )):
            self._afterRead(result)

        if updateFig:
            self.setData(updateFig = updateFig)
        
        return self

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
            self.requiredFiles = [ os.path.join( self.rootDir, self.settings["resultsPath"], req.replace("$struct$", self.struct) ) for req in self.__class__._requirements["files"] ]
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
    def setData(self, updateFig = True, **kwargs):
        
        '''
        Method to process the data that has been read beforehand by readData() and prepare the figure.
        '''

        self.data = []

        self._setData()

        if updateFig:
            self.getFigure()
        
        return self

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

        self.layout = {
            'title': '{} {}'.format(self.struct, self._plotType),
            'hovermode': 'closest',
            'xaxis' : {
                'tickvals': self.ticks[0],
                'ticktext': self.settings["ticks"].split(",") if self.source != "siesOut" else self.ticks[1],
            },
            **self.getSettingsGroup("layout")
        }
            
        self.figure = go.Figure({
            'data': self.data,
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
    
    #-------------------------------------------
    #           GUI ORIENTED METHODS
    #-------------------------------------------

    def _getJsonifiableInfo(self):
        '''
        This method is thought mainly to prepare data to be sent through the API to the GUI.
        Data has to be sent as JSON, so this method can only return JSONifiable objects. (no numpy arrays, no NaN,...)
        '''

        if hasattr(self, "figure"):

            #This will take care of converting to JSON wierd datatypes such as np arrays and np.nan 
            figure = json.dumps(self.figure, cls=plotly.utils.PlotlyJSONEncoder)

        else:
            figure = {
                "data": [],
                "layout": {}
            }

        infoDict = {
            "id": self.id,
            "figure": figure,
            "settings": self.settings,
            "params": self.params,
            "paramGroups": self._paramGroups
        }

        return infoDict
    
