import dill

import uuid
import os
from pathlib import Path

from copy import deepcopy, copy

import sisl
from sisl.messages import warn
from sisl._environ import register_environ_variable, get_environ_variable
from .plot import Plot
from .configurable import Configurable, vizplotly_settings
from .plotutils import find_files, find_plotable_siles, call_method_if_present, get_plot_classes

from .input_fields import TextInput, FilePathInput, SwitchInput, ColorPicker, DropdownInput, IntegerInput, FloatInput, RangeSlider, QueriesInput, Array1DInput

__all__ = ["Session"]


class Warehouse:
    """ Class to store everything related to a session.

    A warehouse can be shared between multiple sessions.
    THIS SHOULD ONLY CONTAIN PLOTS!!!! 
    (The rest: tabs, structures and plotables should be session-specific)
    """

    def __init__(self):
        self._warehouse = {
            "plots": {},
            "structs": {},
            "plotables": {},
            "tabs": []
        }

    def __getitem__(self, item):
        """ Gets an item from the warehouse """
        return self._warehouse[item]

    def __setitem__(self, item, value):
        """ Stores an item to the warehouse """
        self._warehouse[item] = value


class Session(Configurable):
    """ Represents a session of the graphical interface

    Plots are organized in different tabs and each tab has a layout
    that defines how plots are displayed in the dashboard.

    Contains different methods that help managing the session and that are
    directly called by the front end of the graphical interface.
    Therefore: IF A METHOD NAME IS CHANGED, THE SAME CHANGE MUST BE DONE
    IN THE GUI FILE "apis/PythonApi".

    Parameters
    -----------
    root_dir: str, optional

    file_storage_dir: str, optional
        Directory where files uploaded in the GUI will be stored
    keep_uploaded: bool, optional
        Whether uploaded files should be kept in disk or directly removed
        after plotting them.
    search_depth: array-like of shape (2,), optional
        Determines the depth limits of the search for structures (from the
        root directory).
    showTooltips: bool, optional
        Tooltips help you understand how something works or what something
        will do.If you are already familiar with the interface, you can
        turn this off.
    listenForUpdates: bool, optional
        Determines whether the session updates plots when files change 
        This is very useful to track progress. It is only meaningful in the
        GUI.
    updateInterval: int, optional
        The time in ms between consecutive checks for updates.
    plot_dims: array-like, optional
        The initial width and height of a new plot.  Width is in columns
        (out of a total of 12). For height, you really should try what works
        best for you
    plot_preset: str, optional
        Preset that is passed directly to each plot initialization
    plotly_template: str, optional
        Plotly template that should be used as the default for this session
    """

    _param_groups = (

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
            key = "root_dir", name = "Root directory",
            group = "filesystem",
            default = os.getcwd(),
            width = "s100% l50%",
            params = {
                "placeholder": "Write the path here..."
            }
        ),

        FilePathInput(
            key="file_storage_dir", name="File storage directory",
            group="filesystem",
            default= get_environ_variable("SISL_TMP"),
            width="s100% l50%",
            params={
                "placeholder": "Write the path here..."
            },
            help="Directory where files uploaded in the GUI will be stored"
        ),

        SwitchInput(
            key="keep_uploaded", name="Keep uploaded files",
            group="filesystem",
            default=False,
            width="s100% l50%",
            help="Whether uploaded files should be kept in disk or directly removed after plotting them."
        ),

        RangeSlider(
            key = "search_depth", name = "Search depth",
            group = "filesystem",
            default = [0, 3],
            width = "s100% l50%",
            params = {
                "min": 0,
                "max": 15,
                "allowCross": False,
                "step": 1,
                "marks": {i: str(i) for i in range(0, 16)},
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

        Array1DInput(
            key="plot_dims", name="Initial plot dimensions",
            default=[4, 30],
            group="gui",
            help="""The initial width and height of a new plot. <br> Width is in columns (out of a total of 12). For height, you really should try what works best for you"""
        ),

        TextInput(key="plot_preset", name="Plot presets",
            default=None,
            help="Preset that is passed directly to each plot initialization"
        ),

        TextInput(key="plotly_template", name="Plotly template",
            default=None,
            help="Plotly template that should be used as the default for this session"
        )

    )

    @vizplotly_settings('before', init=True)
    def __init__(self, *args, **kwargs):
        self.id = str(uuid.uuid4())

        self.before_plot_update = None
        self.on_plot_change = None
        self.on_plot_change_error = None

        self.warehouse = Warehouse()

        call_method_if_present(self, "_after_init")

    #-----------------------------------------
    #            PLOT MANAGEMENT
    #-----------------------------------------

    @property
    def plots(self):
        """ The plots that this session contains """
        return self.warehouse["plots"]

    def plot(self, plotID):
        """ Method to get a plot that is already in the session's warehouse

        Arguments
        -----------
        plotID: str
            The ID of the desired plot

        Returns
        ---------
        plot: sisl.viz.plotly.Plot()
            The instance of the desired plot
        """
        plot = self.plots[plotID]

        if not hasattr(plot, "grid_dims"):
            plot.grid_dims = self.get_setting("plot_dims")

        return plot

    @staticmethod
    def get_plot_classes():
        """ This method provides all the plot subclasses, even the nested ones

        Returns
        -------
        list
            all the plot classes that the module is aware of.
        """
        return get_plot_classes()

    def add_plot(self, plot, tabID = None, noTab = False):
        """ Adds an already initialized plot object to the session

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
        """
        self.warehouse["plots"][plot.id] = plot

        if not noTab:
            tabID = self._tab_id(tabID) if tabID is not None else self.tabs[0]["id"]

            self._add_plot_to_tab(plot.id, tabID)

        call_method_if_present(self, "_on_plot_added", plot, tabID)

        return self

    def new_plot(self, plotClass=None, tabID=None, structID=None, plotable_path=None, animation = False, **kwargs):
        """ Get a new plot from the specified class

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
        new_plot: sisl.viz.plotly.Plot()
            The initialized new plot
        """
        args = []

        if plotClass is None:
            ReqPlotClass = Plot
        else:
            for PlotClass in self.get_plot_classes():
                if PlotClass.__name__ == plotClass:
                    ReqPlotClass = PlotClass
                    break
            else:
                raise ValueError(f"Didn't find the desired plot class: {plotClass}")

        if plotable_path is not None:
            args = (plotable_path,)
        if structID:
            kwargs = {**kwargs, "root_fdf": self.warehouse["structs"][structID]["path"]}

        if animation:
            wdir = self.warehouse["structs"][structID]["path"].parent if structID else self.get_setting("root_dir")
            new_plot = ReqPlotClass.animated(wdir = wdir)
        else:
            plot_preset = self.get_setting("plot_preset")
            if plot_preset is not None:
                kwargs["presets"] = [*[plot_preset], *kwargs.get("presets", [])]
            plotly_template = self.get_setting("plotly_template")
            if plotly_template is not None:
                layout = kwargs.get("layout", {})
                template = layout.get("template", "")
                kwargs["layout"] = {"template": f'{plotly_template}{"+" + template if template else ""}', **layout}
            new_plot = ReqPlotClass(*args, **kwargs)

        self.add_plot(new_plot, tabID)

        return self.plot(new_plot.id)

    def update_plot(self, plotID, newSettings):
        """ Method to update the settings of a plot that is in the session's warehouse

        Arguments
        -----------
        plotID: str
            The ID of the plot whose settings need to be updated
        newSettings: dict
            Dictionary with the key and new value of all the settings that need to be updated.

        Returns
        ---------
        plot: sisl.viz.plotly.Plot()
            The instance of the updated plot
        """
        return self.plot(plotID).update_settings(**newSettings)

    def undo_plot_settings(self, plotID):
        """ Method undo the settings of a plot that is in the session's warehouse

        Arguments
        -----------
        plotID: str
            The ID of the plot whose settings need to be undone

        Returns
        ---------
        plot: sisl.viz.plotly.Plot()
            The instance of the plot with the settings rolled back.
        """
        return self.plot(plotID).undo_settings()

    def remove_plot_from_tab(self, plotID, tab):
        """ Method to remove a plot only from a given tab.

        Parameters
        -----------
        plotID: str
            the ID of the plot that you want to remove.
        tab: str
            the ID or name of the tab that you want to remove the plot from.
        """
        tab = self.tab(tab)

        tab["plots"] = [plot for plot in tab["plots"] if plot != plotID]

    def remove_plot(self, plotID):
        """ Method to remove a plot from all tabs.

        Parameters
        ----------
        plotID: str
            the ID of the plot that you want to remove.
        """
        plot = self.plot(plotID)

        self.warehouse["plots"] = {ID: plot for ID, plot in self.plots.items() if ID != plotID}

        self.remove_plot_from_all_tabs(plotID)

        call_method_if_present(self, "_on_plot_removed", plot)

        return self

    def merge_plots(self, plots, to="multiple", tab=None, remove=True, **kwargs):
        """ Merges two or more plots present in the session using `Plot.merge`.

        Parameters
        -----------
        plots: array-like of (str and/or Plot)
            A list with the ids of the plots (or the actual plots) that you want to merge.
            Note that THE PLOTS PASSED HERE ARE NOT NECESSARILY IN THE SESSION beforehand.
        to: {"multiple", "subplots", "animation"}, optional
            the merge method. Each option results in a different way of putting all the plots
            together:
            - "multiple": All plots are shown in the same canvas at the same time. Useful for direct
            comparison.
            - "subplots": The layout is divided in different subplots.
            - "animation": Each plot is converted into the frame of an animation.
        tab: str, optional
            the name or id of the tab where you want the new plot to go. 
            If not provided it will go to the tab where the first plot belongs.
        remove: boolean, optional
            whether the plots used to do the merging should be removed from the session's layout.
            Remember that you are always in time to split the merged plots into individual plots
            again.
        **kwargs: 
            go directly extra arguments that are directly passed to `MultiplePlot`, `Subplots`
            or `Animation` initialization. (see `Plot.merge`)
        """
        # Get the plots if ids where passed. Note that we can accept plots that are not in the warehouse yet
        plots = [self.plot(plot) if isinstance(plot, str) else plot for plot in plots]

        merged = plots[0].merge(plots[1:], to=to, **kwargs)

        if tab is None:
            for session_tab in self.tabs:
                if plots[0].id in session_tab["plots"]:
                    tab = session_tab["id"]
                    break

        if remove:
            for plot in plots:
                self.remove_plot(plot.id)

        self.add_plot(merged, tabID=tab)

        return self

    def updates_available(self):
        """ Checks if the session's plots have pending updates due to changes in files.

        Returns
        ---------
        list
            the ids of the plots where an update is available.
        """
        updates_avail = [plotID for plotID, plot in self.plots.items() if plot.updates_available()]

        return updates_avail

    def commit_updates(self):
        """ Updates the plots that can be updated according to `updates_available`.

        Note that this method can be safely called since it has no effect when no updates are available.
        """
        for plotID in self.updates_available():
            try:
                self.plots[plotID].read_data(update_fig=True)
            except Exception as e:
                warn(f"Could not update plot {plotID}.\nError: {e}")

        return self

    def listen(self, forever=False):
        """ Listens for updates in the followed files (see the `updates_available` method)

        Parameters
        ---------
        forever: boolean, optional
            whether to keep listening after the first plot updates.
        """
        from threading import Event

        exit_event = Event()

        while exit_event.is_set():

            exit_event.wait(1)

            updates_avail = self.updates_available()

            if len(updates_avail) != 0:

                for plotID in updates_avail:
                    self.plots[plotID].read_data(update_fig=True)

                if not forever:
                    exit_event.set()

    def figures_only(self):
        """ Removes all plot data from this session's plots except the actual figure.

        This is very useful to save just for display, since it can decrease the size of the session
        DRAMATICALLY.
        """
        for plotID, plot in self.plots.items():

            plot = Plot.from_plotly(plot.figure)
            plot.id = plotID

            self.warehouse["plots"][plotID] = plot

    def _run_plot_method(self, plotID, method_name, *args, **kwargs):
        """ Generic private method to run methods on plots that belong to this session.

        Any public method that runs plot methods should use this private method under the hood.

        In this way, the session will be able to consistently respond to plot updates. E.g. 
        """
        plot = self.plot(plotID)

        method = getattr(plot.autosync, method_name)

        return method(*args, **kwargs)

    #-----------------------------------------
    #            TABS MANAGEMENT
    #-----------------------------------------
    @property
    def tabs(self):
        """ The tabs that this session contains """
        return self.warehouse["tabs"]

    def tab(self, tab):
        """ Get a tab by its name or ID. 

        If it does not exist, it will be created (this acts as a shortcut for add_tab in that case)

        Parameters
        --------
        tab: str
            The name or ID of the tab you want to get
        """
        tab_str = tab

        tabID = self._tab_id(tab_str)

        for tab in self.tabs:
            if tab["id"] == tabID:
                return tab
        else:
            self.add_tab(tab_str)
            return self.tab(tab_str)

    def add_tab(self, name="New tab", plots=[]):
        """ Adds a new tab to the session

        Arguments
        ----------
        name: optional, str ("New tab")
            The name of the new tab
        plots: optional, array-like
            Array of ids (as strings) that identify the plots that you want to put inside your tab.
            Keep in mind that the plots with these ids must be present in self.plots.
        """
        new_tab = {"id": str(uuid.uuid4()), "name": name, "plots": deepcopy(plots), "layouts": {"lg": []}}

        self.tabs.append(new_tab)

        return self

    def update_tab(self, tabID, newParams={}, **kwargs):
        """ Method to update the parameters of a given tab """
        tab = self.tab(tabID)

        for key, val in {**newParams, **kwargs}.items():
            tab[key] = val

        return self

    def remove_tab(self, tabID):
        """ Removes a tab from the current session """
        tabID = self._tab_id(tabID)

        for iTab, tab in enumerate(self.warehouse["tabs"]):
            if tab["id"] == tabID:
                del self.warehouse["tabs"][iTab]
                break

        return self

    def move_plot(self, plot, tab, keep=False):
        """ Moves a plot to a tab

        Parameters
        ----------
        plot: str or sisl.viz.plotly.Plot
            the plot's ID or the plot's instance
        tab: str
            the tab's id or the tab's name.
        keep: boolean, optional
            if True the plot is also kept in the previous tab.
            This doesn't waste any additional memory,
            since the tabs only hold references of the plots they have,
            each plot is stored only once
        """
        plotID = plot
        if isinstance(plot, Plot):
            plotID = plot.id

        if not keep:
            self.remove_plot_from_all_tabs(plotID)

        self._add_plot_to_tab(plotID, tab)

        return self

    def _add_plot_to_tab(self, plot, tab):
        """ Adds a plot to the requested tab.

        If the plot is not part of the session already, it will be added.

        Parameters
        ----------
        plot: str or sisl.viz.plotly.Plot
            the plot's ID or the plot's instance
        tab: str
            the tab's id or the tab's name.
        """
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
        """ Removes a given plot from all tabs where it is located.

        Parameters
        -----------
        plotID: str
            the id of the plot you want to remove.
        """
        for tab in self.tabs:
            self.remove_plot_from_tab(plotID, tab["id"])

        return self

    def get_tab_plots(self, tab):
        """ Returns all the plots of a given tab.

        Parameters
        ------------
        tab: str
            the id or name of the tab.
        """
        tab = self.tab(tab)

        return [self.plot(plotID) for plotID in tab["plots"]] if tab else None

    def set_tab_plots(self, tab, plots):
        """ Sets the plots list of a tab

        Parameters
        --------
        tab: str
            tab's id or name
        plots: array-like of str or sisl.viz.plotly.Plot (or combination of the two)
            plots ids or plot instances.
        """
        tab = self.tab(tab)

        tab["plots"] = []

        for plot in plots:
            self.add_plot(plot, tab)

    def tab_id(self, tab_name):
        """ Gets the id of a given tab.

        Parameters
        -----------
        tab_name: str
            the name of the tab

        Returns
        ---------
        str or None.
            the ID of the tab. None if there's no such tab.
        """
        for tab in self.tabs:
            if tab["name"] == tab_name:
                return tab["id"]

    def _tab_id(self, tab_id_or_name):
        """ Gets the id of a given tab.

        Parameters
        -----------
        tab_id_or_name: str
            the id or name of the tab.

        Returns
        ---------
        str or None.
            the ID of the tab. None if there's no such tab.
        """
        try:
            uuid.UUID(str(tab_id_or_name))
            return tab_id_or_name
        except Exception:
            return self.tab_id(tab_id_or_name)

    #-----------------------------------------
    #         STRUCTURES MANAGEMENT
    #-----------------------------------------

    def get_structures(self, path=None, root_dir=".", search_depth=None):
        """ Gets all the structures that are in the scope of this session

        Parameters
        -----------
        path: str, optional
            the path where to start looking for structures.

            If not provided, the session's "root_dir" will be used.

        Returns
        ----------
        dict
            keys are the structure ID and values are info about each structure.
        """
        path = Path(path or root_dir)

        #Get the structures
        self.warehouse["structs"] = {
            str(uuid.uuid4()): {"name": path.name, "path": path} for path in find_files(root_dir, "*fdf", search_depth)
        }

        #Avoid passing unnecessary info to the browser.
        return {structID: {"id": structID, **{k: struct[k] for k in ["name", "path"]}} for structID, struct in self.warehouse["structs"].items()}

    def get_plotables(self, path=None, root_dir=".", search_depth=None):
        """ Gets all the plotables that are in the scope of this session.

        Parameters
        -----------
        path: str, optional
            the path where to start looking for plotables.

            If not provided, the session's "root_dir" will be used.

        Returns
        ----------
        dict
            keys are the plotable ID and values are info about each structure.
        """
        # Empty the plotables dictionary
        self.warehouse["plotables"] = {}
        path = Path(path or root_dir)

        # Get all the files that correspond to registered plotable siles
        files = find_plotable_siles(path, search_depth)

        for SileClass, filepaths in files.items():

            avail_plots = list(SileClass.plot.plotly._raw_methods)
            default_plot = SileClass.plot.plotly._default

            # Extend the plotables dict with the files that we find that belong to this sile
            self.warehouse["plotables"] = {**self.warehouse["plotables"], **{
                str(uuid.uuid4()): {"name": path.name, "path": path, "plots": avail_plots, "default_plot": default_plot} for path in filepaths
            }}

        #Avoid passing unnecessary info to the browser.
        return {id: {"id": id, **{k: plotable[k] for k in ["name", "path", "plots", "default_plot"]}, "chosenPlots": [plotable["default_plot"]]}
                for id, plotable in self.warehouse["plotables"].items()}

    def save(self, path, figs_only=False):
        """ Stores the session in disk.

        Parameters
        ----------
        path: str
            Path where the session should be saved.
        figs_only: boolean, optional
            Whether only figures should be saved, the rest of plot's data will be ignored.
        """
        session = copy(self)

        if figs_only:
            session.figures_only()

        for plotID, plot in session.plots.items():
            plot._get_pickleable()

        with open(path, 'wb') as handle:
            dill.dump(session, handle, protocol=dill.HIGHEST_PROTOCOL)

        return self
