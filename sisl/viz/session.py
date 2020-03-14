import pickle

import uuid
import os
import glob

from copy import deepcopy

import sisl
from .plot import Plot, MultiplePlot, Animation
from .configurable import Configurable, afterSettingsInit
from .plotutils import findFiles

from .inputFields import InputField, TextInput, SwitchInput, ColorPicker, DropdownInput, IntegerInput, FloatInput, RangeSlider, QueriesInput

class Session(Configurable):

    _onSettingsUpdate = {
        "functions": ["getStructures"],
        "config":{
            "multipleFunc": False,
            "order": False,
        },
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
        
        TextInput(
            key = "rootDir", name = "Root directory",
            group = "filesystem",
            default = os.getcwd(),
            width = "s100% l50%",
            params = {
                "placeholder": "Write the path here..."
            }
        ),

        RangeSlider(
            key = "searchDepth", name = "Search depth",
            group = "filesystem",
            default = [0,3],
            width = "s100% l50%",
            params = {
                "min": 0,
                "max": 15,
                "allowCross": False,
                "step": 1,
                "marks": { i: str(i) for i in range(0,16) },
                "updatemode": "drag",
                "units": "eV",
            },
            help = "Determines the depth limits of the search for structures (from the root directory)."
        ),

        SwitchInput(
            key = "showTooltips", name = "Show Tooltips",
            group = "gui",
            default = True,
            params = {
                "offLabel": "No",
                "onLabel": "Yes"
            },
            help = "Tooltips help you understand how something works or what something will do.<br>If you are already familiar with the interface, you can turn this off."
        ),

        SwitchInput(
            key = "listenForUpdates", name = "Listen for updates",
            group = "gui",
            default = True,
            params = {
                "offLabel": "No",
                "onLabel": "Yes"
            },
            help = "Determines whether the session updates plots when files change <br> This is very useful to track progress. It is only meaningful in the GUI."
        ),

        IntegerInput(
            key="updateInterval", name="Update time interval",
            group="gui",
            default=2000,
            params={
                "min": 0
            },
            help="The time in ms between consecutive checks for updates."
        )


    )

    @afterSettingsInit
    def __init__(self, **kwargs):

        self.id = str(uuid.uuid4())

        self.initWarehouse( kwargs.get("firstTab", False) )

        if callable( getattr(self, "_afterInit", None )):
            self._afterInit()
        
    def initWarehouse(self, firstTab):

        self.warehouse = {
            "plots": {},
            "structs": {},
            "tabs": [],
        }

    #-----------------------------------------
    #            PLOT MANAGEMENT
    #-----------------------------------------
    @property
    def tabs(self):
        return self.warehouse["tabs"]

    @property
    def plots(self):
        return self.warehouse["plots"]

    def getPlotClasses(self):
        '''
        This method provides all the plot subclasses, even the nested ones
        '''

        def get_all_subclasses(cls):

            all_subclasses = []

            for subclass in cls.__subclasses__():

                if subclass not in [MultiplePlot, Animation]:
                    all_subclasses.append(subclass)

                all_subclasses.extend(get_all_subclasses(subclass))

            return all_subclasses
        
        return sorted(get_all_subclasses(sisl.viz.Plot), key = lambda clss: clss._plotType) 
    
    def addPlot(self, plot, tabID = None, noTab = False): 
        '''
        Adds an already initialized plot object to the session

        Parameters
        -----
        plot: Plot()
            the plot object that we want to add to the session
        tabID: str, optional
            the ID of the tab where we want to add the plot. If not provided,
            it will be appended to the first tab
        noTab: boolean, optional
            if set to true, prevents the plot from being added to a tab
        '''

        self.warehouse["plots"][plot.id] = plot

        if not noTab:
            tabID = tabID if tabID is not None else self.tabs[0]["id"]
            self.addPlotToTab(plot.id, tabID)
        
        return self

    def newPlot(self, plotClass, tabID = None, structID = None, animation = False ,**kwargs):
        '''
        Get a new plot from the specified class

        Arguments
        -----------
        plotClass: str
            The name of the desired class
        tabID: str
            Tab where the plot should be stored
        structID: str, optional
            The ID of the structure for which we want the plot
        animation: bool, optional
            Whether the initialized plot should be an animation.
            
            If true, it uses the `Plot.animated` method to initialize the plot
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

        if animation:
            wdir = os.path.dirname(self.warehouse["structs"][structID]["path"]) if structID else self.setting("rootDir")
            newPlot = ReqPlotClass.animated(wdir = wdir)
        else:
            newPlot = ReqPlotClass(**kwargs)

        self.addPlot(newPlot, tabID)

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

    def updates_available(self):
        '''
        Checks if the session's plots have pending updates due to changes in files.
        '''

        updates_avail = [plotID for plotID, plot in self.plots.items() if plot.updates_available()]

        return updates_avail
    
    def commit_updates(self):
        '''
        Updates the plots that can be updated according to `updates_available`.

        Note that this method can be safely called since it has no effect when no updates are available.
        '''

        for plotID in self.updates_available():
            self.plots[plotID].readData(updateFig=True)
        
        return self

    def listen(self, forever=False):
        '''
        Listens for updates in the followed files (see the `updates_available` method)

        Parameters
        ---------
        forever: boolean, optional
            whether to keep listening after the first plot updates.
        '''

        while True:
            
            time.sleep(1)

            updates_avail = self.updates_available()
            
            if len(updates_avail) != 0:
                
                for plotID in updates_avail:
                    self.plots[plotID].readData(updateFig=True)
                
                if not forever:
                    break
    
    #-----------------------------------------
    #            TABS MANAGEMENT
    #-----------------------------------------
    def getTabs(self):
        '''
        Returns the list of current tabs
        '''

        return self.warehouse["tabs"]
    
    def addTab(self, tabName = "New tab", plots = []):
        '''
        Adds a new tab to the session

        Arguments
        ----------
        tabName: optional, str ("New tab")
            The name of the new tab
        plots: optional, array-like
            Array of ids (as strings) that identify the plots that you want to put inside your tab.
            Keep in mind that the plots with these ids must be present in self.plots.
        '''

        newTab = {"id": str(uuid.uuid4()), "name": tabName, "plots": deepcopy(plots)}

        print("WITH THIS PLOTS:", plots)

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
        Adds a plot that is already in the session to the requested tab
        '''

        for tab in self.warehouse["tabs"]:
            if tab["id"] == tabID:
                tab["plots"] = [*tab["plots"], plotID]
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
            str(uuid.uuid4()): {"name": os.path.basename(path), "path": path } for path in findFiles(self.setting("rootDir"), "*fdf", self.setting("searchDepth"))
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
            "paramGroups": self._paramGroups,
            "updatesAvailable": self.updates_available()
        }

        return infoDict
    
    def save(self, path):

        for plotID, plot in self.warehouse["plots"].items():
            plot._getPickleable()
        
        with open(path, 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return self