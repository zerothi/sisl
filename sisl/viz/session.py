import pickle

import uuid
import os
import glob

from copy import deepcopy

import sisl
from .plot import Plot, MultiplePlot, Animation
from .configurable import Configurable, afterSettingsInit
from .plotutils import findFiles, get_plotable_siles

from .input_fields import TextInput, SwitchInput, ColorPicker, DropdownInput, IntegerInput, FloatInput, RangeSlider, QueriesInput, Array1dInput

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
        ),

        Array1dInput(
            key="plotDims", name="Initial plot dimensions",
            default=[4, 30],
            group="gui",
            help='''The initial width and height of a new plot. <br> Width is in columns (out of a total of 12). For height, you really should try what works best for you'''
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
            "plotables": {},
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

    def plot(self, plotID):
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

        plot = self.plots[plotID]

        if not hasattr(plot, "grid_dims"):
            plot.grid_dims = self.setting('plotDims')

        return plot

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
        tab: str, optional
            the name of the tab where we want to add the plot or
            the ID of the tab where we want to add the plot.

            If neither tab or tabID are provided, it will be appended to the first tab
        noTab: boolean, optional
            if set to true, prevents the plot from being added to a tab
        '''

        self.warehouse["plots"][plot.id] = plot

        if not noTab:
            tabID = self._tab_id(tabID) if tabID is not None else self.tabs[0]["id"]

            self.addPlotToTab(plot.id, tabID)
        
        return self

    def newPlot(self, plotClass=None, tabID = None, structID = None, plotableID=None, animation = False ,**kwargs):
        '''
        Get a new plot from the specified class

        Arguments
        -----------
        plotClass: str, optional
            The name of the desired class.
            If not provided, the session will try to initialize from the `Plot` parent class. 
            This may be useful if the keyword argument filename is provided, for example, to let
            `Plot` guess which type of plot to use. 
        tabID: str, optional
            Tab where the plot should be stored
        structID: str, optional
            The ID of the structure for which we want the plot.
        plotableID: str, optional
            The ID of the plotable file that we want to plot.
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

        if plotClass is None:
            ReqPlotClass = Plot
        else:
            for PlotClass in self.getPlotClasses():
                if PlotClass.__name__ == plotClass:
                    ReqPlotClass = PlotClass
                    break
            else:
                raise Exception("Didn't find the desired plot class: {}".format(plotClass))

        if plotableID is not None:
            kwargs = {**kwargs, "filename": self.warehouse["plotables"][plotableID]["path"] }
        if structID:
            kwargs = {**kwargs, "rootFdf": self.warehouse["structs"][structID]["path"] }

        if animation:
            wdir = os.path.dirname(self.warehouse["structs"][structID]["path"]) if structID else self.setting("rootDir")
            newPlot = ReqPlotClass.animated(wdir = wdir)
        else:
            newPlot = ReqPlotClass(**kwargs)

        self.addPlot(newPlot, tabID)

        return self.plot(newPlot.id)
    
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

        return self.plot(plotID).updateSettings(**newSettings)
    
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

        return self.plot(plotID).undoSettings()
    
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
            try:
                self.plots[plotID].readData(updateFig=True)
            except Exception as e:
                print(f"Could not update plot {plotID}. \n Error: {e}")
        
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
    
    def tab(self, tab):
        '''
        Get a tab by its name or ID. 
        
        If it does not exist, it will be created (this acts as a shortcut for addTab in that case)

        Parameters
        --------
        tab: str
            The name or ID of the tab you want to get
        '''

        tab_str = tab

        tabID = self._tab_id(tab_str)

        for tab in self.tabs:
            if tab["id"] == tabID:
                return tab
        else:
            self.addTab(tab_str)
            return self.tab(tab_str)

    def addTab(self, name = "New tab", plots = []):
        '''
        Adds a new tab to the session

        Arguments
        ----------
        name: optional, str ("New tab")
            The name of the new tab
        plots: optional, array-like
            Array of ids (as strings) that identify the plots that you want to put inside your tab.
            Keep in mind that the plots with these ids must be present in self.plots.
        '''

        newTab = {"id": str(uuid.uuid4()), "name": name, "plots": deepcopy(plots), "layouts": {"lg":[]}}

        self.warehouse["tabs"].append(newTab)

        return self
    
    def updateTab(self, tabID, newParams = {}, **kwargs):
        '''
        Method to update the parameters of a given tab
        '''

        tab = self.tab(tabID)

        for key, val in {**newParams, **kwargs}.items():
            tab[key] = val

        return self

    def removeTab(self, tabID):
        '''
        Removes a tab from the current session
        '''

        tabID = self._tab_id(tabID)

        for iTab, tab in enumerate(self.warehouse["tabs"]):
            if tab["id"] == tabID:
                del self.warehouse["tabs"][iTab]
                break

        return self

    def addPlotToTab(self, plot, tab):
        '''
        Adds a plot to the requested tab.

        If the plot is not part of the session already, it will be added.

        Parameters
        ----------
        plot: str or sisl.viz.Plot
            the plot's ID or the plot's instance
        tab: str
            the tab's id or the tab's name.
        '''

        if isinstance(plot, Plot):
            plotID = plot.id
            if plotID not in self.plots:
                self.addPlot(plot, tab)
        else:
            plotID = plot

        tab = self.tab(tab)

        tab["plots"] = [*tab["plots"], plotID]
        
        return self
    
    def removePlotFromAllTabs(self, plotID):
        '''
        Removes a given plot from all tabs where it is located
        '''

        for iTab, tab in enumerate( self.warehouse["tabs"] ):

            self.warehouse["tabs"][iTab] = {**tab, "plots": [plot for plot in tab["plots"] if plot != plotID]}
        
        return self

    def getTabPlots(self, tab):
        '''
        Returns all the plots of a given tab
        '''

        tab = self.tab(tab)

        return [self.plot(plotID) for plotID in tab["plots"]] if tab else None

    def set_tab_plots(self, tab, plots):
        '''
        Sets the plots list of a tab

        Parameters
        --------
        tab: str
            tab's id or name
        plots: array-like of str or sisl.viz.Plot (or combination of the two)
            plots ids or plot instances.
        '''

        tab = self.tab(tab)

        tab["plots"] = []

        for plot in plots:
            self.addPlotToTab(plot, tab)
    
    def tab_id(self, tab_name):

        for tab in self.tabs:
            if tab["name"] == tab_name:
                return tab["id"]
    
    def _tab_id(self, tab_id_or_name):

        try:
            uuid.UUID(str(tab_id_or_name))
            return tab_id_or_name
        except Exception:
            return self.tab_id(tab_id_or_name)

    #-----------------------------------------
    #         STRUCTURES MANAGEMENT
    #-----------------------------------------

    def getStructures(self, path=None):

        path = path or self.setting("rootDir")

        #Get the structures
        self.warehouse["structs"] = {
            str(uuid.uuid4()): {"name": os.path.basename(path), "path": path} for path in findFiles(self.setting("rootDir"), "*fdf", self.setting("searchDepth"))
        }

        #Avoid passing unnecessary info to the browser.
        return {structID: {"id": structID, **{k: struct[k] for k in ["name", "path"]}} for structID, struct in deepcopy(self.warehouse["structs"]).items() }
    
    def getPlotables(self, path=None):

        # Empty the plotables dictionary
        self.warehouse["plotables"] = {}
        path = path or self.setting("rootDir")

        # Start filling it
        for rule in get_plotable_siles(rules=True):
            searchString = f"*.{rule.suffix}"
            plotType = rule.cls._plot[0].plotName()

            # Extend the plotables dict with the files that we find that belong to this sile
            self.warehouse["plotables"] = { **self.warehouse["plotables"], **{
                str(uuid.uuid4()): {"name": os.path.basename(path), "path": path, "plot": plotType} for path in findFiles(path, searchString, self.setting("searchDepth"), case_insensitive=True)
            }}

        #Avoid passing unnecessary info to the browser.
        return {id: {"id": id, **{k: struct[k] for k in ["name", "path", "plot"]}} for id, struct in deepcopy(self.warehouse["plotables"]).items() }

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