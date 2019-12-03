import sisl
from sisl.viz import Plot, Configurable, afterSettingsInit

import pickle

import uuid
import os

from copy import deepcopy

from fileScrapper import findFiles

class Session(Configurable):

    _onSettingsUpdate = {
        "__config":{
            "multipleFunc": False,
            "importanceOrder": ["getStructures"]
        },
        "getStructures": lambda obj: obj.getStructures,
    }

    _paramGroups = (

        {
            "key": "gui",
            "name": "Interface tweaks",
            "icon": "aspect_ratio",
            "description": "There's a spanish saying that goes something like: <i>for each taste there's a color</i>. Since we know this is true, you can tweak these parameters too <b>make the interface feel as comfortable as possible</b>. "
        },

        {
            "key": "filesystem",
            "name": "File system settings",
            "icon": "folder",
            "description": "Your computer is pretty big and most of it is not important for the analysis of simulations (e.g. the folder with your holidays pictures). Also, everyone likes to store things differently. Please <b>indicate how exactly do you want the interface to look for simulations results</b> in your filesystem. "
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
            "key": "rootDir",
            "name": "Root directory",
            "group": "filesystem",
            "default": os.getcwd(),
            "inputField": {
                "type": "textinput",
                "width": "s100% l50%",
                "params": {
                    "placeholder": "Write the path here...",
                }
            },
            #"onUpdate": "getStructures"
        },

        {
            "key": "searchDepth",
            "name": "Search depth",
            "group": "filesystem",
            "default": [0,3],
            "inputField": {
                "type": "rangeslider",
                "width": "s100% l50%",
                "params": {
                    "min": 0,
                    "max": 15,
                    "allowCross": False,
                    "step": 1,
                    "marks": { i: str(i) for i in range(0,16) },
                    "updatemode": "drag",
                    "units": "eV",
                },
                "styles": {}
            },
            "help": "Determines the depth limits of the search for structures (from the root directory).",
            #"onUpdate": "getStructures"
        },

        {
            "key": "showTooltips",
            "name": "Show Tooltips",
            "default": True,
            "group": "gui",
            "inputField": {
                "type": "switch",
                "width": "s50% m30% l15%",
                "params": {
                    "offLabel": "No",
                    "onLabel": "Yes"
                }
            },
            "help": "Tooltips help you understand how something works or what something will do.<br>If you are already familiar with the interface, you can turn this off.",
            "onUpdate": "getFigure",
        },

    )

    @afterSettingsInit
    def __init__(self):

        self.id = str(uuid.uuid4())

        self.initWarehouse()
        
    def initWarehouse(self):

        self.warehouse = {
            "plots": {},
            "structs": {},
            "tabs": [
                {"id": str(uuid.uuid4()), "name": "First Tab", "plots": []}
            ],
        }

    #-----------------------------------------
    #            PLOT MANAGEMENT
    #-----------------------------------------

    def getPlotClasses(self):
        '''
        This method provides all the plot subclasses, even the nested ones
        '''

        def get_all_subclasses(cls):
            all_subclasses = []

            for subclass in cls.__subclasses__():
                all_subclasses.append(subclass)
                all_subclasses.extend(get_all_subclasses(subclass))

            return all_subclasses
        
        return get_all_subclasses(sisl.viz.Plot) 

    def newPlot(self, plotClass, tabID = None, structID = None, **kwargs):
        '''
        Get a new plot from the specified class

        Arguments
        -----------
        plotClass: str
            The name of the desired class
        tabID: str
            Tab where the plot should be stored
        structID: str (optional, None)
            The ID of the structure for which we want the plot
        **kwargs:
            Passed directly to plot initialization

        Returns
        -----------
        newPlot: sisl.viz.Plot()
            The initialized new plot
        '''

        for PlotClass in self.getPlotClasses():
            if PlotClass.__name__ == plotClass:
                ReqPlotClass = PlotClass
                break
        else:
            raise Exception("Didn't find the desired plot class: {}".format(plotClass))

        if structID:
            kwargs = {**kwargs, "rootFdf": self.warehouse["structs"][structID]["path"] }

        newPlot = ReqPlotClass(**kwargs)

        self.warehouse["plots"][newPlot.id] = newPlot

        if tabID:
            self.addPlotToTab(newPlot.id, tabID)

        return newPlot
    
    def getPlot(self, plotID):
        '''
        Method to get a plot that is already in the session's warehouse

        Arguments
        -----------
        plotID: str
            The ID of the desired plot
        
        Returns
        ---------
        plot: sisl.viz.Plot()
            The instance of the desired plot
        '''

        return self.warehouse["plots"][plotID]
    
    def updatePlot(self, plotID, newSettings):
        '''
        Method to update the settings of a plot that is in the session's warehouse

        Arguments
        -----------
        plotID: str
            The ID of the plot whose settings need to be updated
        newSettings: dict
            Dictionary with the key and new value of all the settings that need to be updated.

        Returns
        ---------
        plot: sisl.viz.Plot()
            The instance of the updated plot
        '''

        return self.getPlot(plotID).updateSettings(**newSettings)
    
    def undoPlotSettings(self, plotID):
        '''
        Method undo the settings of a plot that is in the session's warehouse

        Arguments
        -----------
        plotID: str
            The ID of the plot whose settings need to be undone

        Returns
        ---------
        plot: sisl.viz.Plot()
            The instance of the plot with the settings rolled back.
        '''

        return self.getPlot(plotID).undoSettings()
    
    def removePlot(self, plotID):
        '''
        Method to remove a plot
        '''

        del self.warehouse["plots"][plotID]

        self.removePlotFromAllTabs(plotID)

        return self

    #-----------------------------------------
    #            TABS MANAGEMENT
    #-----------------------------------------
    def getTabs(self):
        '''
        Returns the list of current tabs
        '''

        return self.warehouse["tabs"]
    
    def addTab(self):
        '''
        Adds a new tab to the session
        '''

        newTab = {"id": str(uuid.uuid4()), "name": "New tab", "plots": []}

        self.warehouse["tabs"].append(newTab)

        return self
    
    def updateTab(self, tabID, newParams = {}):
        '''
        Method to update the parameters of a given tab
        '''

        for iTab, tab in enumerate(self.warehouse["tabs"]):
            if tab["id"] == tabID:
                self.warehouse["tabs"][iTab] = {**tab, **newParams}
                break

        return self

    def removeTab(self, tabID):
        '''
        Removes a tab from the current session
        '''

        for iTab, tab in enumerate(self.warehouse["tabs"]):
            if tab["id"] == tabID:
                del self.warehouse["tabs"][iTab]
                break

        return self

    def addPlotToTab(self, plotID, tabID):
        '''
        Adds a plot to the requested tab
        '''

        for tab in self.warehouse["tabs"]:
            if tab["id"] == tabID:
                tab["plots"].append(plotID)
                break
        
        return self
    
    def removePlotFromAllTabs(self, plotID):
        '''
        Removes a given plot from all tabs where it is located
        '''

        for iTab, tab in enumerate( self.warehouse["tabs"] ):

            self.warehouse["tabs"][iTab] = {**tab, "plots": [plot for plot in tab["plots"] if plot != plotID]}
        
        return self

    def getTabPlots(self, tabID):
        '''
        Returns all the plots of a given tab
        '''

        for tab in self.warehouse["tabs"]:
            if tab["id"] == tabID:
                return [self.getPlot(plotID) for plotID in tab["plots"]]
        
        return []

    #-----------------------------------------
    #         STRUCTURES MANAGEMENT
    #-----------------------------------------

    def getStructures(self):

        #Get the structures
        self.warehouse["structs"] = {
            str(uuid.uuid4()): {"name": os.path.basename(path), "path": path } for path in findFiles(self.settings["rootDir"], "*fdf", self.settings["searchDepth"])
        }

        #Avoid passing unnecessary info to the browser.
        return {structID: {k: struct[k] for k in ["name"]} for structID, struct in deepcopy(self.warehouse["structs"]).items() }
    
    #-----------------------------------------
    #      NOTIFY CURRENT STATE TO GUI
    #-----------------------------------------
    
    def _getJsonifiableInfo(self):
        '''
        This method is thought mainly to prepare data to be sent through the API to the GUI.
        Data has to be sent as JSON, so this method can only return JSONifiable objects. (no numpy arrays, no NaN,...)
        '''

        infoDict = {
            "id": self.id,
            "tabs": self.warehouse["tabs"],
            "settings": self.settings,
            "params": self.params,
            "paramGroups": self._paramGroups
        }

        return infoDict
    
    def save(self, path):

        for plotID, plot in self.warehouse["plots"].items():
            plot._getPickleable()
        
        with open(path, 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return self