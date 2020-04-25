import dill as pickle

import uuid
import os
import glob

from copy import deepcopy, copy

import sisl
from sisl.viz.GUI.api_utils.sync import Connected
from .plot import Plot, MultiplePlot, Animation
from .configurable import Configurable, afterSettingsInit
from .plotutils import findFiles, get_plotable_siles, call_method_if_present

from .input_fields import TextInput, SwitchInput, ColorPicker, DropdownInput, IntegerInput, FloatInput, RangeSlider, QueriesInput, Array1dInput

class Warehouse:
    '''
    Class to store everything related to a session.

    A warehouse can be shared between multiple sessions.
    THIS SHOULD ONLY CONTAIN PLOTS!!!! 
    (The rest: tabs, structures and plotables should be session-specific)
    '''

    def __init__(self):

        self._warehouse = {
            "plots": {},
            "structs": {},
            "plotables": {},
            "tabs": []
        }

    def __getitem__(self, item):
        return self._warehouse[item]

    def __setitem__(self, item, value):
        self._warehouse[item] = value

    
class Session(Configurable, Connected):
    '''
    Represents a session of the graphical interface.

    Plots are organized in different tabs and each tab has a layout
    that defines how plots are displayed in the dashboard.

    Contains different methods that help managing the session and that are
    directly called by the front end of the graphical interface.
    Therefore: IF A METHOD NAME IS CHANGED, THE SAME CHANGE MUST BE DONE
    IN THE GUI FILE "apis/PythonApi".

    '''

    _onSettingsUpdate = {
        "functions": ["get_structures"],
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
    def __init__(self, *args, **kwargs):

        self.id = str(uuid.uuid4())

        self.before_plot_update = None
        self.on_plot_change = None
        self.on_plot_change_error = None

        self.warehouse = Warehouse()

        call_method_if_present(self, "_after_init")
        
        super().__init__(*args, **kwargs)

    #-----------------------------------------
    #            PLOT MANAGEMENT
    #-----------------------------------------

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

    def get_plot_classes(self):
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
    
    def add_plot(self, plot, tabID = None, noTab = False): 
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

        # Make sure the plot is connected to the same socketio as the session
        plot.socketio = self.socketio

        self.warehouse["plots"][plot.id] = plot

        if not noTab:
            tabID = self._tab_id(tabID) if tabID is not None else self.tabs[0]["id"]

            self._add_plot_to_tab(plot.id, tabID)
        
        return self
    
    def new_plot(self, plotClass=None, tabID = None, structID = None, plotableID=None, animation = False ,**kwargs):
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
        new_plot: sisl.viz.Plot()
            The initialized new plot
        '''

        if plotClass is None:
            ReqPlotClass = Plot
        else:
            for PlotClass in self.get_plot_classes():
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
            new_plot = ReqPlotClass.animated(wdir = wdir)
        else:
            new_plot = ReqPlotClass(**kwargs)

        self.add_plot(new_plot, tabID)

        return self.plot(new_plot.id)
    
    def update_plot(self, plotID, newSettings):
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
    
    def undo_plot_settings(self, plotID):
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
    
    def remove_plot_from_tab(self, plotID, tabID):
        '''
        Method to remove a plot only from a given tab.
        '''

        tab = self.tab(tabID)

        tab["plots"] = [plot for plot in tab["plots"] if plot != plotID]

    def remove_plot(self, plotID):
        '''
        Method to remove a plot
        '''

        plot = self.plot(plotID)

        plot.socketio = None

        self.warehouse["plots"] = { ID: plot for ID, plot in self.plots.items() if ID != plotID}

        self.remove_plot_from_all_tabs(plotID)

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
    
    def figures_only(self):
        '''
        Removes all plot data from this session's plots except the actual figure.

        This is very useful to save just for display, since it can decrease the size of the session
        DRAMATICALLY.
        '''

        for plotID, plot in self.plots.items():

            plot = Plot.from_plotly(plot.figure)
            plot.id = plotID

            self.warehouse["plots"][plotID] = plot

    def _run_plot_method(self, plotID, method_name, *args, **kwargs):
        '''
        Generic private method to run methods on plots that belong to this session.

        Any public method that runs plot methods should use this private method under the hood.

        In this way, the session will be able to consistently respond to plot updates. E.g. 
        '''

        plot = self.plot(plotID)

        method = getattr(plot.autosync, method_name)

        return method(*args, **kwargs)

    #-----------------------------------------
    #            TABS MANAGEMENT
    #-----------------------------------------
    @property
    def tabs(self):
        return self.warehouse["tabs"]
    
    def tab(self, tab):
        '''
        Get a tab by its name or ID. 
        
        If it does not exist, it will be created (this acts as a shortcut for add_tab in that case)

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
            self.add_tab(tab_str)
            return self.tab(tab_str)

    def add_tab(self, name = "New tab", plots = []):
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

        self.tabs.append(newTab)

        return self
    
    def update_tab(self, tabID, newParams = {}, **kwargs):
        '''
        Method to update the parameters of a given tab
        '''

        tab = self.tab(tabID)

        for key, val in {**newParams, **kwargs}.items():
            tab[key] = val

        return self

    def remove_tab(self, tabID):
        '''
        Removes a tab from the current session
        '''

        tabID = self._tab_id(tabID)

        for iTab, tab in enumerate(self.warehouse["tabs"]):
            if tab["id"] == tabID:
                del self.warehouse["tabs"][iTab]
                break

        return self

    def _add_plot_to_tab(self, plot, tab):
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
                self.add_plot(plot, tab)
        else:
            plotID = plot

        tab = self.tab(tab)

        tab["plots"] = [*tab["plots"], plotID]
        
        return self
    
    def remove_plot_from_all_tabs(self, plotID):
        '''
        Removes a given plot from all tabs where it is located
        '''

        for tab in self.tabs:
            self.remove_plot_from_tab(plotID, tab["id"])
        
        return self

    def get_tab_plots(self, tab):
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
            self.add_plot(plot, tab)
    
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

    def get_structures(self, path=None):

        path = path or self.setting("rootDir")

        #Get the structures
        self.warehouse["structs"] = {
            str(uuid.uuid4()): {"name": os.path.basename(path), "path": path} for path in findFiles(self.setting("rootDir"), "*fdf", self.setting("searchDepth"))
        }

        #Avoid passing unnecessary info to the browser.
        return {structID: {"id": structID, **{k: struct[k] for k in ["name", "path"]}} for structID, struct in self.warehouse["structs"].items() }
    
    def get_plotables(self, path=None):

        # Empty the plotables dictionary
        self.warehouse["plotables"] = {}
        path = path or self.setting("rootDir")

        # Start filling it
        for rule in get_plotable_siles(rules=True):
            searchString = f"*.{rule.suffix}"
            plotType = rule.cls._plot[0].plotName()

            files = findFiles(path, searchString, self.setting(
                "searchDepth"), case_insensitive=True)

            if files:
                # Extend the plotables dict with the files that we find that belong to this sile
                self.warehouse["plotables"] = { **self.warehouse["plotables"], **{
                    str(uuid.uuid4()): {"name": os.path.basename(path), "path": path, "plot": plotType} for path in files
                }}

        #Avoid passing unnecessary info to the browser.
        return {id: {"id": id, **{k: struct[k] for k in ["name", "path", "plot"]}} for id, struct in self.warehouse["plotables"].items() }

    #-----------------------------------------
    #      NOTIFY CURRENT STATE TO GUI
    #-----------------------------------------
    
    def _get_dict_for_GUI(self):
        '''
        This method is thought mainly to prepare data to be sent through the API to the GUI.
        Data has to be sent as JSON, so this method can only return JSONifiable objects. (no numpy arrays, no NaN,...)
        '''

        infoDict = {
            "id": self.id,
            "tabs": self.tabs,
            "settings": self.settings,
            "params": self.params,
            "paramGroups": self._paramGroups,
            "updatesAvailable": self.updates_available(),
            "plotOptions": [
                {"value": subclass.__name__, "label": subclass._plotType} 
                for subclass in self.get_plot_classes()
            ],
            "structures": self.get_structures(),
            "plotables": self.get_plotables()
        }

        return infoDict
    
    def _on_socketio_change(self):
        '''
        Transmit the socketio change to all the plots
        '''

        for _, plot in self.plots.items():
            plot.socketio = self.socketio

    def save(self, path, figs_only=False):
        '''
        Stores the session in disk.

        Parameters
        ----------
        path: str
            Path where the session should be saved.
        figs_only: boolean, optional
            Whether only figures should be saved, the rest of plot's data will be ignored.
        '''

        session = copy(self)
        if figs_only:
            session.figures_only()

        for plotID, plot in session.plots.items():
            plot._getPickleable()
        
        with open(path, 'wb') as handle:
            pickle.dump(session, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return self
