'''
This file contains the Plot class, which should be inherited by all plot classes
'''
import uuid
import os
import sys
import numpy as np
import json
import dill as pickle
#import pickle
import time
from types import MethodType, FunctionType
import itertools
from functools import partial

import plotly
import plotly.graph_objects as go

import sisl

from .configurable import *
from .plotutils import initMultiplePlots, repeatIfChilds, dictOfLists2listOfDicts, trigger_notification, spoken_message, running_in_notebook, check_widgets
from .inputFields import TextInput, SwitchInput, ColorPicker, DropdownInput, IntegerInput, FloatInput, RangeSlider, QueriesInput, ProgramaticInput
from ._shortcuts import ShortCutable

PLOTS_CONSTANTS = {
    "spins": ["up", "down"],
    "readFuncs": {
        "fromH": lambda obj: obj._readfromH, 
        "siesOut": lambda obj: obj._readSiesOut,
        "noSource": lambda obj: obj._readNoSource
    }
}

#Wrapper to time methods
def timeit(method):

    def timed(obj, *args, **kwargs):

        start = time.time()

        result = method(obj, *args, **kwargs)

        print("{}: {} seconds".format(method.__name__, time.time() - start))

        return result
    
    return timed

#------------------------------------------------
#                 PLOT CLASS
#------------------------------------------------    
class Plot(ShortCutable, Configurable):
    
    _onSettingsUpdate = {
        "functions": ["readData", "setData", "getFigure"],
        "config":{
            "multipleFunc": False,
            "order": True,
        },
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
                {"key": "yaxis", "name": "Y axis"},
                {"key": "animation", "name": "Animation"}
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

        ProgramaticInput(
            key = "readingOrder", name = "Output reading/generating order",
            group = "dataread",
            default = ("guiOut", "siesOut", "fromH", "noSource"),
            help = "Order in which the plot tries to read the data it needs."
        ),

        TextInput(
            key = "rootFdf", name = "Path to fdf file",
            group = "dataread",
            help = "Path to the fdf file that is the 'parent' of the results.",
            params = {
                "placeholder": "Write the path here..."
            }
        ),

        TextInput(
            key = "resultsPath", name = "Path to your results",
            group = "dataread",
            default = "",
            params = {
                "placeholder": "Write the path here..."
            },
            width = "s100% m50% l33%",
            help = "Directory where the files with the simulations results are located.<br> This path has to be relative to the root fdf.",
        ),

        TextInput(
            key = "title", name = "Title",
            group = "layout",
            params = {
                "placeholder": "Title of your plot..."
            },
            width = "s100% l40%",
            help = "Think of a memorable title for your plot!",
        ),

        SwitchInput(
            key = "showlegend", name = "Show Legend",
            group = "layout",
            default = True,
            params = {
                "offLabel": "No",
                "onLabel": "Yes"
            }
        ),

        ColorPicker(
            key = "paper_bgcolor", name = "Figure color",
            group = "layout",
            default = "white",
        ),

        ColorPicker(
            key = "plot_bgcolor", name = "Plot color",
            group = "layout",
            default = "white",
        ),
        
        *[
            param for iAxis, axis in enumerate(["xaxis", "yaxis"]) for param in [
            
            TextInput(
                key = "{}_title".format(axis), name = "Title",
                group = "layout", subGroup = axis,
                params = {
                    "placeholder": "Write the axis title..."
                },
                width = "s100% m50%"
            ),

            DropdownInput(
                key = "{}_type".format(axis), name = "Type",
                group = "layout", subGroup = axis,
                default = "-",
                params = {
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
                },
                width = "s100% m50%"
            ),

            SwitchInput(
                key = "{}_visible".format(axis), name = "Visible",
                group = "layout", subGroup = axis,
                default = True,
                params = {
                    "offLabel": "No",
                    "onLabel": "Yes"
                },
                width = "s50% m50% l25%"
            ),

            ColorPicker(
                key = "{}_color".format(axis), name = "Color",
                group = "layout", subGroup = axis,
                default = "black",
                width = "s50% m50% l25%"
            ),

            SwitchInput(
                key = "{}_showgrid".format(axis), name = "Show grid",
                group = "layout", subGroup = axis,
                default = False,
                params = {
                    "offLabel": "No",
                    "onLabel": "Yes"
                },
                width = "s50% m50% l25%"
            ),

            ColorPicker(
                key = "{}_gridcolor".format(axis), name = "Grid color",
                group = "layout", subGroup = axis,
                default = "#ccc",
                width = "s50% m50% l25%"
            ),

            SwitchInput(
                key = "{}_showline".format(axis), name = "Show axis line",
                group = "layout", subGroup = axis,
                default = True,
                params = {
                    "offLabel": "No",
                    "onLabel": "Yes"
                },
                width = "s50% m30% l30%",
            ),

            FloatInput(
                key = "{}_linewidth".format(axis), name = "Axis line width",
                group = "layout", subGroup = axis,
                default = 1,
            ),

            ColorPicker(
                key = "{}_linecolor".format(axis), name = "Axis line color",
                group = "layout", subGroup = axis,
                default = "black",
                width = "s50% m30% l30%",
            ),

            SwitchInput(
                key = "{}_zeroline".format(axis), name = "Zero line",
                group = "layout", subGroup = axis,
                default = [False, True][iAxis],
                params = {
                    "offLabel": "Hide",
                    "onLabel": "Show"
                }
            ),

            ColorPicker(
                key = "{}_zerolinecolor".format(axis), name = "Zero line color",
                group = "layout", subGroup = axis,
                default = "#ccc",
            ),

            DropdownInput(
                key = "{}_ticks".format(axis), name = "Ticks position",
                group = "layout", subGroup = axis,
                default = "outside",
                params = {
                    "placeholder": "Choose the ticks positions...",
                    "options": [
                        {"label": "Outside", "value": "outside"},
                        {"label": "Inside", "value": "Inside"},
                        {"label": "No ticks", "value": ""},
                    ],
                    "isClearable": False,
                    "isSearchable": False,
                },
                width = "s100% m50% l33%"
            ),

            ColorPicker(
                key = "{}_tickcolor".format(axis), name = "Tick color",
                group = "layout", subGroup = axis,
                default = "white",
            ),

            FloatInput(
                key = "{}_ticklen".format(axis), name = "Tick length",
                group = "layout", subGroup = axis,
                default = 5,
                width = "s50% m30% l15%"
            ),
            
            SwitchInput(
                key = "{}_mirror".format(axis), name = "Mirror",
                group = "layout", subGroup = axis,
                default = False,
                params = {
                    "offLabel": "False",
                    "onLabel": "True"
                },
                width = "s50% m50% l25%"
            ),
            
            TextInput(
                key = "{}_scaleanchor".format(axis), name = "Scale anchor",
                group = "layout", subGroup = axis,
                default=None,
                width ="s100% m50%",
                help="Axis to which the scale of this axis should be proportional. The ratio is set by scaleratio"
            ),
            
            FloatInput(
                key = "{}_scaleratio".format(axis), name = "Scale ratio",
                group = "layout", subGroup = axis,
                default = 1,
                width = "s50% m30% l15%"
            ),]
            
        ]
        
    )

    @classmethod
    def from_plotly(cls, plotly_fig):
        '''
        Converts a plotly plot to a Plot object
        '''

        plot = cls(only_init=True)
        plot.figure = plotly_fig

        return plot

    @classmethod
    def plotName(cls):
        return getattr(cls, "_plotType", cls.__name__)

    @property
    def plotType(self):
        return self.__class__.plotName()
    
    @property
    def _innotebook(self):
        return running_in_notebook()
    
    @property
    def _widgets(self):
        return check_widgets()

    @classmethod
    def animated(cls, *args, fixed = {}, frameNames = None, **kwargs):
        '''Creates an animation out of a class.

        This class method returns an animation with frames belonging to a given plot class.

        For example, if you run `BandsPlot.animated()` you will get an animation made of bands plots.

        If no arguments are passed, you will get the default animation for that plot, if there is any.

        Parameters
        -----------
        *args:
            Depending on what you pass the arguments will be interpreted differently:
                - Two arguments:
                    First: str
                        Key of the setting that you want to animate.
                    Second: array-like
                        Values that you want the setting to have at each animation frame.

                    Ex: BandsPlot.animated("bandsFile", ["file1", "file2", "file3"] )
                    will produce an animation where each frame uses a different bandsFile.

                - One argument and it is a dictionary:
                    First: dict
                        The keys of this dictionary will be the setting keys you want to animate
                        and the values are of course the values for each frame for that setting. 

                    It works exactly as the previous case, but in this case we have multiple settings to animate.
                
                - One argument and it is a function:
                    First: function
                        With this function you can produce any settings you want without limitations.

                        It will be used as the `_getInitKwargsList` method of the animation, so it needs to
                        accept self (the animation object) and return a list of dictionaries which are the 
                        settings for each frame.

                    the function will recieve the parameter and can act on it in any way you like.
                    It doesn't need to return the parameter, just modify it.
                    In this function, you can call predefined methods of the parameter, for example.

                    Ex: obj.modifyParam("length", lambda param: param.incrementByOne() )

                    given that you know that this type of parameter has this method.
        fixed: dict, optional
            A dictionary containing values for settings that will be fixed along the animation.
            For the settings that you don't specify here you will get the defaults.
        frameNames: list of str or function, optional
            If it is a list of strings, each string will be used as the name for the corresponding frame.

            If it is a function, it should accept `self` (the animation object) and return a list of strings
            with the frame names. Note that you can access the plot instance responsible for each frame under
            `self.childPlots`. The function will be run each time the figure is generated, so in this way your
            frame names will be dynamic.

            FRAME NAMES SHOULD BE UNIQUE, OTHERWISE THE ANIMATION WILL HAVE A WEIRD BEHAVIOR.

            If this is not provided, frame names will be generated automatically.
        **kwargs:
            Will be passed directly to animation initialization, so it can contain the settings for the animation, for example.

            If args are not passed and the default animation is being created. Some keyword arguments may be used by the method
            that generates the default animation. One recurrent example of this is the keyword `wdir`. 

        Returns
        --------
        Animation
            The Animation that you asked for
        
        '''

        #Try to retrieve the default animation if no arguments are provided
        if len(args) == 0:

            return cls._defaultAnimation(fixed = fixed, frameNames = frameNames, **kwargs) if hasattr(cls, "_defaultAnimation" ) else None

        #Define how the getInitkwargsList method will look like
        if callable(args[0]):
            _getInitKwargsList = args[0]
        else:
            if len(args) == 2:
                animated_settings = {args[0]: args[1]}
            elif isinstance(args[0], dict):
                animated_settings = args[0]

            def _getInitKwargsList(self):

                #Adding the fixed values to the list
                vals = { 
                    **{key: itertools.repeat(val) for key, val in fixed.items()},
                    **animated_settings
                }

                return dictOfLists2listOfDicts(vals)
        
        _getFrameNames = None
        #Define how to get the framenames
        if frameNames:

            if callable(frameNames):
                _getFrameNames = frameNames
            else:
                def _getFrameNames(self,i):
                    return frameNames[i]
        
        #Return the initialized animation
        return Animation( _plugins = {
            "_getInitKwargsList": _getInitKwargsList,
            "_getFrameNames": _getFrameNames,
            "_plotClasses": cls
        })

    def __new__(cls, filename=None, **kwargs):
        '''
        This method decides what to return when the plot class is instantiated.

        It is supposed to help the users by making the plot class very functional
        without the need for the users to use extra methods.
        '''

        #If a filename is recieved, we will try to find a plot for it
        if isinstance(filename, str):
            SileClass = sisl.get_sile_class(filename)

            if SileClass == sisl.io.siesta.fdfSileSiesta:
                kwargs["rootFdf"] = filename
                plot = cls(**kwargs)
            else:

                if hasattr(SileClass, "__plot__"):
                    plot = SileClass.__plot__(**kwargs)
                elif hasattr(SileClass, "_plot"):
                    PlotClass, kwarg_key = SileClass._plot
                    kwargs[kwarg_key] = filename
                    plot = PlotClass(**kwargs, title=os.path.basename(filename))
                else:
                    raise NotImplementedError(f'There is no plot implementation for {os.path.splitext(filename)[-1]} extensions yet.')

                if cls != Plot and cls != plot.__class__:
                    print((
                        f"You requested a {cls.__name__} and we got you a {plot.__class__.__name__} instead.\nWe hope you don't mind :)\n\n"
                        f'We did this because "{filename}" is a {SileClass.__name__}. If you are not happy with this, please specify the key of the setting where "{filename}" should go.'
                    ))
            
            # Inform that we don't want to run the __init__ method anymore
            # See the beggining of __init__()
            plot.INIT_ON_NEW = True
            plot.AVOID_SETTINGS_INIT = True

            return plot

        return object.__new__(cls)

    @afterSettingsInit
    def __init__(self, *args, H = None, attrs_for_plot={}, only_init=False,**kwargs):

        if getattr(self, "INIT_ON_NEW", False):
            delattr(self, "INIT_ON_NEW")
            return
        
        #Initialize shortcut management
        ShortCutable.__init__(self)
        
        #Give an ID to the plot
        self.id = str(uuid.uuid4())

        #Give the user the possibility to do things before initialization (IDK why)
        if callable( getattr(self, "_beforeInit", None )):
            self._beforeInit()

        # Set all the attributes that have been passed
        for key, val in attrs_for_plot.items():
            setattr(self, key, val)

        # Check if the user has provided a hamiltonian (which can contain a geometry)
        # This is not meant to be used by the GUI (in principle), just programatically
        self.PROVIDED_H = False
        self.PROVIDED_GEOM = False
        if H is not None:
            self.PROVIDED_H = True
            self.H = H
            self.geom = getattr(H, "geom", None)
            self.PROVIDED_GEOM = self.geom is not None

        #Set the isChildPlot attribute to let the plot know if it is part of a bigger picture (e.g. Animation)
        self.isChildPlot = kwargs.get("isChildPlot", False)
        
        #Initialize the variable to store when has been the last data read (0 means never basically)
        self.last_dataread = 0
        self._filesToFollow = []

        # Initialize the figure
        self.figure = go.Figure()

        #If plugins have been provided, then add them.
        #Plugins are an easy way of extending a plot. They can be methods, variables...
        #They are added to the object instance, not the whole class.
        if kwargs.get("_plugins"):
            for name, plugin in kwargs.get("_plugins").items():
                if isinstance(plugin, FunctionType):
                    plugin = MethodType(plugin, self)
                setattr(self, name, plugin)

        self._general_plot_shortcuts()
        #Give the user the possibility to overwrite default settings
        if callable( getattr(self, "_afterInit", None )):
            self._afterInit()
        
        # If we were supposed to only initialize the plot, stop here
        if only_init:
            return

        #Try to generate the figure (if the settings required are still not there, it won't be generated)
        try:
            if MultiplePlot in type.mro(self.__class__):
                #If its a multiple plot try to inititialize all its child plots
                if self.PLOTS_PROVIDED:
                    self.getFigure()
                else:
                    self.initAllPlots()
            else:
                self.readData()
                
        except Exception as e:
            print("The plot has been initialized correctly, but the current settings were not enough to generate the figure.\n (Error: {})".format(e))
            pass
    
    def __str__(self):
        
        string = (
            f'Plot class: {self.plotType}    Plot type: {getattr(self, "_plotType", None)}\n\n'
            'Settings:\n{}'.format("\n".join([ "\t- {}: {}".format(key,value) for key, value in self.settings.items()]))
        )
        
        return string
    
    def __getattr__(self, key):
        '''
        This method is executed only after python has found that there is no such attribute in the instance

        So let's try to find elsewhere. There are two options:
            - The attribute is in the figure object (self.figure)
            - The attribute is currently being shared with other plots (only possible if it's a childplot)
        '''

        if key in ["figure", "shared_attr"]:
            pass
        elif hasattr(self.figure, key):
            return getattr(self.figure, key)
        else:
            #If it is a childPlot, maybe the attribute is in the shared storage to save memory and time
            try:
                return self.shared_attr(key)
            except (KeyError, AttributeError):
                pass

        raise AttributeError(f"The attribute '{key}' was not found either in the plot, its figure, or in shared attributes.")

    def __setattr__(self, key, val):
        '''
        If the attribute is one of ["data", "layout", "frames"] we are going to store it directly in `self.figure` for convenience
        and in order to save memory.

        If is a childplot and it has the attribute `_SHOULD_SHARE_WITH_SIBLINGS` set to True, we will submit the attribute to the shared store.
        This happens in animations/multiple plots. There's a "leading plot" that reads the data and then shares it with the rest
        so that they don't need to read it again, in a collective effort to save memory and time.

        Otherwise
        '''

        if key in ["data", "layout", "frames"]:
            self.figure.update(**{key: val})
        elif key != '_SHOULD_SHARE_WITH_SIBLINGS' and getattr(self, '_SHOULD_SHARE_WITH_SIBLINGS', False):
            self.share_attr(key, val)
        else:
            self.__dict__[key] = val

    def _general_plot_shortcuts(self):

        self._listening_shortcut()

    @repeatIfChilds
    @afterSettingsUpdate
    def readData(self, updateFig = True, **kwargs):
        '''
        Gets the information for the bands plot and stores it into self.df

        Returns
        -----------
        dataRead: boolean
            whether data has been read succesfully or not
        '''

        # Restart the filesToFollow variable so that we can start to fill it with the new files
        # Apart from the explicit call in this method, setFiles and setUpHamiltonian also add files to follow
        self._filesToFollow = []

        if callable( getattr(self, "_beforeRead", None )):
            self._beforeRead()
        
        try:    
            self.setFiles()
        except Exception:
            pass

        #Update the title of the plot if there is none
        if not self.getSetting("title"):
            self.updateSettings(updateFig = False, title = '{} {}'.format(getattr(self, "struct", ""), self.plotType) )
        
        #We try to read from the different sources using the _readFromSources method of the parent Plot class.
        filesToFollow = self._readFromSources()

        #Follow the files that are returned by the method that succesfully read the plot's data
        self.follow(*filesToFollow, unfollow=False)

        # We don't update the last dataread here in case there has been a succesful data read because we want to
        # wait for the afterRead() method to be succesful
        if self.source is None:
            self.last_dataread = 0

        if callable( getattr(self, "_afterRead", None )):
            self._afterRead()

        if self.source is not None:
            self.last_dataread = time.time()

        if updateFig:
            self.setData(updateFig = updateFig)
        
        return self
    
    def _readFromSources(self):
        
        '''
        Tries to read the data from the different possible sources in the order 
        determined by self.settings["readingOrder"].
        '''
        
        errors = []
        #Try to read in the order specified by the user
        for source in self.setting("readingOrder"):
            try:
                #Get the reading function
                readingFunc = PLOTS_CONSTANTS["readFuncs"][source](self)
                #Execute it
                filesToFollow = readingFunc()
                self.source = source
                return filesToFollow
            except Exception as e:
                errors.append("\t- {}: {}.{}".format(source, type(e).__name__, e))
                
        else:
            self.source = None
            raise Exception("Could not read or generate data for {} from any of the possible sources.\n\n Here are the errors for each source:\n\n {}  "
                            .format(self.__class__.__name__, "\n".join(errors)) )
    
    def follow(self, *files, to_abs=True, unfollow=False):
        '''
        Makes sure that the object knows which files to follow in order to trigger updates.

        Parameters
        ----------
        *files: str
            a string that represents the path to the file that needs to be followed.

            You can pass as many as you want as separate arguments. Note that if you have a list of
            files you can pass them separately by doing `follow(*my_list_of_files)`, you don't need to
            (and you shouldn't) build a loop :)
        to_abs: boolean, optional
            whether the paths should be converted to absolute paths to make file following procedures
            more robust. It is better to leave it as True unless you have a good reason to change it.
        unfollow: boolean, optional
            whether the previous files should be unfollowed. If set to False, we are just adding more files.
        '''

        newFilesToFollow = [os.path.abspath(filePath) if to_abs else filePath for filePath in files or []]

        self._filesToFollow = newFilesToFollow if unfollow else [*self._filesToFollow, *newFilesToFollow]

    def updates_available(self):
        '''
        This function checks whether the read files have changed.

        For it to work properly, one should specify the files that have been read by
        their _read*() methods. This is done by returning a list of the files (see example).

        Note that the `setFiles` and `setUpHamiltonian` methods are already responsible for
        informing about the files they read, so you only need to specify those that you are
        "explicitly" reading in your method.

        Examples
        -------
        If I'm developing a class with a `_readSiesOut` method that reads the "structure.bands" file and I want it to
        be reactive to file changes, it should look something like this:

        ```python
        def _readSiesOut(self):

            ...Do all the stuff

            return ["structure.bands"]
        ```
        '''

        filesModified = np.array([ os.path.getmtime(filePath) > self.last_dataread for filePath in self._filesToFollow])

        return filesModified.any()
    
    def listen(self, forever=True, show=True, as_animation=False, return_animation=True, return_figWidget=False, clearPrevious=True, notify=False, speak=False, notify_title=None, notify_message=None, speak_message=None, fig_widget=None):
        '''
        Listens for updates in the followed files (see the `updates_available` method)

        Parameters
        ---------
        forever: boolean, optional
            whether to keep listening after the first plot update
        show: boolean, optional
            whether to show the plot at the beggining and update the layout when the plot is updated.
        as_animation: boolean, optional
            will add a new frame each time the plot is updated.

            The resulting animation is returned unless return_animation is set to False. This is done because
            the Plot object iself is not converted to an animation. Instead, a new animation is created and if you
            don't save it in a variable it will be lost, you will have no way to access it later.

            If you are seeing two figures at the beggining, it is because you are not storing the animation figure.
            Set the return_animation parameter to False if you understand that you are going to "lose" the animation,
            you will only be able to see a display of it while it is there.
        return_animation: boolean, optional
            if as_animation is `True`, whether the animation should be returned.
            Important: see as_animation for an explanation on why this is the case
        return_figWidget: boolean, optional
            it returns the figure widget that is in display in a jupyter notebook in case the plot has
            succeeded to display it. Note that, even if you are in a jupyter notebook, you won't get a figure
            widget if you don't have the plotly notebook extension enabled. Check `<your_plot>._widgets` to see
            if you are missing witget support.

            if return_animation is True, both the animation and the figure widget will be returned in a tuple.
            Although right now, this does not make much sense because figure widgets don't support frames. You will get None.
        clearPrevious: boolean, optional
            in case show is True, whether the previous version of the plot should be hidden or kept in display.
        notify: boolean, optional
            trigger a notification everytime the plot updates.
        speak: boolean, optional
            trigger a spoken message everytime the plot updates.
        notify_title: str, optional
            the title of the notification.
        notify_message: str, optional
            the message of the notification.
        speak_message: str, optional
            the spoken message. Feel free to get creative here!
        '''
        from IPython.display import clear_output
        import asyncio

        # This is a weird limitation, because multiple listeners could definitely
        # be implemented, but I don't have time now, and I need to ensure that no listeners are left untracked
        # If you need it, ask me! (Pol)
        self.stop_listening()

        pt = self

        if as_animation:
            pt = Animation(
                plots = [self.clone()]
            )

        if show and fig_widget is None:
            fig = pt.show(return_figWidget=True)
            if isinstance(fig, go.FigureWidget):
                fig_widget = fig
        
        if notify:
            trigger_notification("SISL", "Notifications will appear here")
        if speak:
            spoken_message("I will speak when there is an update.")

        async def listen():
            while True:
                if self.updates_available():
                    try:

                        self.readData(updateFig=True)

                        if as_animation:
                            new_plot = self.clone()
                            pt.add_childplots(new_plot)
                            pt.getFigure()

                        if clearPrevious and fig_widget is None:
                            clear_output()

                        if show and fig_widget is None:
                            pt.show()
                        else:
                            pt._update_FigureWidget(fig_widget)

                        if not forever:
                            self._listening_task.cancel()

                        if notify:
                            title = notify_title or "SISL PLOT UPDATE"
                            message = notify_message or f"{getattr(self, 'struct', '')} {self.__class__.__name__} updated"
                            trigger_notification(title, message )
                        if speak:
                            spoken_message(speak_message if speak_message is not None else f"Your {self.__class__.__name__} is updated. Check it out")

                    except Exception as e:
                        pass
                    
                await asyncio.sleep(1)

        loop = asyncio.get_event_loop()
        self._listening_task = loop.create_task(listen())

        self.add_shortcut("ctrl+alt+l", "Stop listening", self.stop_listening, fig_widget=fig_widget, _description="Tell the plot to stop listening for updates")

        if as_animation and return_animation:
            if return_figWidget:
                return pt, fig_widget
            else:
                return pt
        elif return_figWidget:
            return fig_widget

    def _listening_shortcut(self, fig_widget=None):

        self.add_shortcut(
            "ctrl+alt+l", "Listen for updates", 
            self.listen, fig_widget=fig_widget,
            _description="Make the plot listen for changes in the files that it reads"
        )

    def stop_listening(self, fig_widget=None):
        '''
        Makes the plot stop listening for updates.

        Using this method only makes sense if you have previously made the plot listen
        either through `Plot.listen()` or `Plot.show(listen=True)`
        '''

        task = getattr(self, "_listening_task", None)

        if task is not None:
            task.cancel()
            self._listening_task = None
        
            self._listening_shortcut(fig_widget=fig_widget)

    @afterSettingsUpdate
    def setFiles(self, **kwargs):
        '''
        Checks if the required files are available and then builds a list with them
        '''
        #Set the fdfSile
        rootFdf = self.setting("rootFdf")
        self.rootDir, fdfFile = os.path.split( rootFdf )
        self.rootDir = "." if self.rootDir == "" else self.rootDir
        
        self.wdir = os.path.join(self.rootDir, self.setting("resultsPath"))
        self.fdfSile = sisl.get_sile(rootFdf)
        self.struct = self.fdfSile.get("SystemLabel", "")
            
        #Check that the required files are there
        #if RequirementsFilter().check(self.rootFdf, self.__class__.__name__ ):
        if hasattr(self, "_requirements"):
            #If they are there, we can confidently build this list
            self.requiredFiles = [ os.path.join( self.rootDir, self.setting("resultsPath"), req.replace("$struct$", self.struct) ) for req in self.__class__._requirements["siesOut"]["files"] ]
        #else:
            #raise Exception("The required files were not found, please check your file system.")

        #Inform that we have read the fdf file
        self.follow(rootFdf, unfollow=False)

        return self
    
    @afterSettingsUpdate
    def setupHamiltonian(self, **kwargs):
        '''
        Sets up the hamiltonian for calculations with sisl.
        '''

        NEW_FDF = self.did_setting_update("rootFdf")
        
        if not self.PROVIDED_GEOM and (not hasattr(self, "geom") or NEW_FDF):
            self.geom = self.fdfSile.read_geometry(output = True)
        
        if not self.PROVIDED_H and (not hasattr(self, "H") or NEW_FDF):
            #Try to read the hamiltonian in two different ways
            try:
                #This one is favoured because it may read from TSHS file, which contains all the information of the geometry and basis already
                self.H = self.fdfSile.read_hamiltonian()

                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                # WE SHOULD FOLLOW THE FILES HERE SOMEHOW
                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            except Exception:

                HSXfile = os.path.join(self.rootDir, self.struct + ".HSX")
                Hsile = sisl.get_sile(HSXfile)
                self.H = Hsile.read_hamiltonian(geom = self.geom)

                #Inform that we have read the hamiltonian from the HSX file
                self._followFiles([HSXfile], unfollow=False)

        # Sisl is hanging on this step, so for now we are not going to calculate the fermi level
        # self.fermi = self.H.fermi_level()
        self.fermi = 0

        return self
    
    @repeatIfChilds
    @afterSettingsUpdate
    def setData(self, updateFig = True, **kwargs):
        
        '''
        Method to process the data that has been read beforehand by readData() and prepare the figure.
        '''

        self.clear()

        # This is used to know how many traces have been added, so that the user can clear only
        # the traces written by the plot methods and keep traces that they have added later, if they want
        self._starting_traces = len(self.data)

        self._setData()

        if updateFig:
            self.getFigure()
        
        # The explanation for this is above (in the definition of _starting_traces)
        self._own_traces_slice = slice(self._starting_traces, len(self.data))

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

        if getattr(self, "childPlots", None):
            #Then it is a multiple plot and we need to create the figure from the child plots

            self.clear(); frames = []

            if getattr(self, "_isAnimation", False):

                self.data = self.childPlots[0].data

                # Get the names for each frame
                frameNames = []
                for i, plot in enumerate(self.childPlots):
                    try:
                        frame_name = self._getFrameNames(i)
                    except Exception:
                        frame_name = f"Frame {i+1}"
                    frameNames.append(frame_name)
                
                maxN = np.max([[len(plot.data) for plot in self.childPlots]])
                
                for frameName , plot in zip(frameNames , self.childPlots):
                    
                    data = plot.data
                    nTraces = len(data)
                    if nTraces < maxN:
                        nAddTraces = maxN - nTraces
                        data = [*data, *np.full(nAddTraces, {"type": "scatter", "x":  [0], "y": [0], "visible": False})]

                    frames = [*frames, {'name': frameName, 'data': data, "layout": plot.settingsGroup("layout")}]
                
                self.frames = frames

            else:

                data = []
                for plot in self.childPlots:
                    data = [*data, *plot.data]
                
                self.data = data
            
        framesLayout = {}

        #If it is an animation, extra work needs to be done.
        if getattr(self, 'frames', False):

            #This will create the buttons needed no control the animation

            #Animate method
            steps = [
                {"args": [
                [frame["name"]],
                {"frame": {"duration": int(self.setting("frameDuration")), "redraw": True},
                "mode": "immediate",
                "transition": {"duration": 300}}
            ],
                "label": frame["name"],
                "method": "animate"} for frame in self.frames
            ]

            #Update method
            """ steps = []
            breakpoints = np.array([len(frame["data"]) for frame in self.frames]).cumsum()
            for i, frame in enumerate(self.frames):
                
                steps.append({
                    "label": frame["name"],
                    "method": "restyle",
                    "args": [
                        {"x": [data["x"] for data in frame["data"]],
                        "y": [data["y"] for data in frame["data"]],
                        "visible": [data.get("visible", True) for data in frame["data"]],
                        "mode": [data.get("mode",None) for data in frame["data"]],
                        "marker": [data.get("marker", {}) for data in frame["data"]],
                        "line": [data.get("line", {}) for data in frame["data"]]},
                    ]
                }) """
            
            framesLayout = {

                "sliders": [
                    {
                        "active": 0,
                        "yanchor": "top",
                        "xanchor": "left",
                        "currentvalue": {
                            "font": {"size": 20},
                            #"prefix": "Bands file:",
                            "visible": True,
                            "xanchor": "right"
                        },
                        #"transition": {"duration": 300, "easing": "cubic-in-out"},
                        "pad": {"b": 10, "t": 50},
                        "len": 0.9,
                        "x": 0.1,
                        "y": 0,
                        "steps": steps
                    }
                ],
                
                "updatemenus": [

                    {'type': 'buttons',
                    'buttons': [
                        {
                            'label': 'Play',
                            'method': 'animate',
                            'args': [None, {"frame": {"duration": int(self.setting("frameDuration")), "redraw": True},
                                            "fromcurrent": True, "transition": {"duration": 100,
                                                                                "easing": "quadratic-in-out"}}],
                        },

                        {
                            'label': 'Pause',
                            'method': 'animate',
                            'args': [ [None], {"frame": {"duration": 0}, "redraw": True,
                                            'mode': 'immediate',
                                            "transition": {"duration": 0}}],
                        }
                    ]}
                ]
            }

        self.layout = {
            'hovermode': 'closest',
            #Need to register this on whatToRunOnUpdate somehow
            **self.settingsGroup("layout"),
            **framesLayout
        }

        if callable( getattr(self, "_afterGetFigure", None )):
            self._afterGetFigure()

        return self
    
    #-------------------------------------------
    #       PLOT MANIPULATION METHODS
    #-------------------------------------------

    def show(self, *args, listen=False, return_figWidget=False, **kwargs):
        '''
        Shows the plot.

        Parameters
        ------
        listen: boolean, optional
            after showing, keeps listening for file changes to update the plot.
            This is nice for monitoring.
        '''

        if listen:
            self.listen(show=True, **kwargs)

        if not hasattr(self, "figure"):
            self.getFigure()

        if self._innotebook and len(args) == 0:
            try:
                return self._show_in_jupyternb(listen=listen, return_figWidget = return_figWidget, **kwargs)
            except Exception as e:
                print(e)
                pass
        
        return self.figure.show(*args, **kwargs)
    
    def _show_in_jupyternb(self, listen=False, return_figWidget=False, **kwargs):

        if self._widgets["plotly"]:

            from IPython.display import display
            import ipywidgets as widgets

            f = go.FigureWidget(self.figure)

            if self._widgets["events"]:
                # If ipyevents is available, show with shortcut support
                self._show_jupnb_with_shortcuts(f, **kwargs)
            else:
                # Else, show without shortcut support
                display(f)

            self._listening_shortcut(fig_widget=f)

            if return_figWidget:
                return f

        else:
            self.figure.show(**kwargs)
                                    
    def _show_jupnb_with_shortcuts(self, fig_widget, **kwargs):

        from ipyevents import Event
        from ipywidgets import HTML, Output

        h = HTML("") # This is to display help such as available shortcuts
        messages = HTML("") # This is to inform about current status
        styles = HTML( "<style>.ipyevents-watched:focus {outline: none}</style>")
        d = Event(source=fig_widget, watched_events=['keydown', 'keyup'])
       
        def handle_dom_events(event, keys_down=[], last_timestamp=[0]):
            # We will keep track of keydowns because then we will be able to support multiple keys shortcuts

            try:
                # Clear the list
                timestamp = event.get("timeStamp")
                duplicates = len(keys_down) != len(set(keys_down))
                if timestamp - last_timestamp[0]> 2000 or duplicates:
                    keys_down *= 0 #Clear the list
                last_timestamp[0] = timestamp

                # This means that the key has been held down for a long time
                if event.get("repeat", False):
                    return

                key = event.get("key", "").lower()
                key_code = event.get("code", "")

                ev_type = event.get("type", None)

                if ev_type == "keydown":
                    if key == "control":
                        key = "ctrl"
                    # If it's a key down event, record it
                    keys_down.append(key)
                elif ev_type == "keyup" and key in keys_down:
                    # If it's a key up event, anounce that the key is not down anymore
                    keys_down.remove(key)
                
                shortcut_key = "+".join(keys_down)

                # Get the help message
                if shortcut_key == "shift+?":
                    h.value = self.shortcuts_summary("html") if not h.value else ""
                
                shortcut = self.shortcut(shortcut_key)
                if shortcut is not None:
                    
                    messages.value = f'Executing "{shortcut["name"]}" because you pressed "{shortcut_key}"...'
                    self.call_shortcut(shortcut_key)
                    messages.value = ""
                    keys_down = []

                    self._update_FigureWidget(fig_widget)

            except Exception as e:
                messages.value = f'<span style="color:darkred; font-weight: bold">{e}</span>'

        d.on_dom_event(partial(handle_dom_events))

        display(fig_widget, messages, h, styles, Output())

    def _update_FigureWidget(self, fig_widget, plot = None):

        fig_widget.data = []
        fig_widget.add_traces(self.data)
        fig_widget.update(layout=self.layout, frames=self.frames)

    def merge(self, others, asAnimation = False, **kwargs):
        '''
        Merges this plot's instance with the list of plots provided

        Parameters
        -------
        others: array-like of Plot() or Plot()
            the plots that we want to merge with this plot instance.
        asAnimation: boolean, optional
            whether you want to merge them as an animation.
            If `False` the plots are all merged into a single plot
        kwargs:
            extra arguments that are directly passed to `Animation` or `MultiplePlot`
            initialization.

        Returns
        -------
        Animation() or MultiplePlot():
            depending on the value of `asAnimation`.
        '''
        
        #Make sure we deal with a list (user can provide a single plot)
        if not isinstance(others, (list, tuple, np.ndarray)):
            others = [others]

        childPlots = [self, *others]

        if asAnimation:
            newPlot = Animation(plots=childPlots, **kwargs)
        else:
            newPlot = MultiplePlot(plots=childPlots, **kwargs)
        
        return newPlot
    
    def clear(self, plot_traces=True, added_traces=True, frames=True):
        '''
        Clears the plot canvas so that data can be reset

        Parameters
        --------
        plot_traces: boolean, optional
            whether traces added by the plot's code should be cleared.
        added_traces: boolean, optional
            whether traces added externally (by the user) should be cleared.
        frames: boolean, optional
            whether frames should also be deleted
        '''

        own_slice = getattr(self, '_own_traces_slice', slice(0,0))

        if not plot_traces and added_traces:
            self.figure.data = self.figure.data[own_slice]
        elif not added_traces and plot_traces:
            self.figure.data = [ trace for i, trace in enumerate(self.data) if i < own_slice.start or i >= own_slice.stop ]
        else:
            self.figure.data = []
        
        if frames:
            self.figure.frames = []
        
        return self

    def normalize(self, axis = "y"):
        '''
        Normalizes all data between 0 and 1 along the requested axis
        '''
        
        self.data = [{**lineData.to_plotly_json(), 
            axis: (np.array(lineData[axis]) - np.min(lineData[axis]))/(np.max(lineData[axis]) - np.min(lineData[axis]))
        } for lineData in self.data]

        return self
    
    def swapAxes(self):

        self.data = [{**lineData.to_plotly_json(), 
            "x": lineData["y"], "y": lineData["x"]
        } for lineData in self.data]

        return self
    
    def vLine(self, x):
        '''
        Draws a vertical line in the figure (NOT WORKING YET!)
        '''

        yrange = self.figure.layout.yaxis.range or [0, 7000]
    
        self.figure.add_scatter(mode = "lines", x = [x,x], y = yrange, hoverinfo = 'none', showlegend = False)

        return self

    def clone(self, *args, **kwargs):
        '''
        Gets you and exact clone of this plot

        You can pass extra args that will overwrite the previous parameters though, if you don't want it to be that exact.
        '''

        return self.__class__(*args, **self.settings, **kwargs) 

    #-------------------------------------------
    #       DATA TRANSFER/STORAGE METHODS
    #-------------------------------------------

    def _getDictForGUI(self):
        '''
        This method is thought mainly to prepare data to be sent through the API to the GUI.
        Data has to be sent as JSON, so this method can only return JSONifiable objects. (no numpy arrays, no NaN,...)
        '''
        
        figure = json.dumps(self.figure, cls=plotly.utils.PlotlyJSONEncoder)

        infoDict = {
            "id": self.id,
            "plotClass": self.__class__.__name__,
            "struct": getattr(self, "struct", None),
            "figure": figure,
            "settings": self.settings,
            "params": self.params,
            "paramGroups": self.paramGroups,
            "grid_dims": getattr(self, "grid_dims", None)
        }

        return infoDict
    
    def _getPickleable(self):
        '''
        Removes from the instance the attributes that are not pickleable.
        '''

        unpickleableAttrs = ['geom', 'fdfSile']

        for attr in ['geom', 'fdfSile']:
            if hasattr(self, attr):
                delattr(self, attr)

        return self
    
    def save(self, path, html = False):
        '''
        Saves the plot so that it can be loaded in the future.

        Parameters
        ---------
        path: str
            The path to the file where you want to save the plot
        html: bool
            If set to true, saves just an html file of the plot visualization.

        Returns
        ---------
        self
        '''

        if html or os.path.splitext(path)[-1] == "html":
            self.figure.write_html('{}.html'.format(path.replace(".html", "")))
            return self

        #The following method actually modifies 'self', so there's no need to get the return
        self._getPickleable()

        with open(path, 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return self
    
    def html(self, path):

        '''
        Just a shortcut for save( html = True )

        Arguments
        --------
        path: str
            The path to the file where you want to save the plot.
        '''

        return self.save(path, html = True)
    
    def to_chart_studio(self, *args, **kwargs):
        '''
        Sends the plot to chart studio if it is possible.

        For it to work, the user should have their credentials correctly set up.

        It is a shortcut for chart_studio.plotly.plot(self.figure, ...etc) so you can pass any extra arguments as if
        you were using `py.plot`
        '''
        import chart_studio.plotly as py

        return py.plot(self.figure, *args, **kwargs)
#------------------------------------------------
#               ANIMATION CLASS
#------------------------------------------------

class MultiplePlot(Plot):

    def __init__(self, *args, plots=None, **kwargs):

        self.shared = {}

        # Take the plots if they have already been created and are provided by the user
        self.PLOTS_PROVIDED = plots is not None
        if self.PLOTS_PROVIDED:

            self.set_child_plots(plots)
        
        super().__init__(*args, **kwargs)
    
    def __getitem__(self, i):

        return self.childPlots[i]
    
    @property
    def _attrs_for_childplots(self):
        '''
        Returns all the attributes that its childplots should have
        '''

        return {
            'isChildPlot': True,
            'shared_attr': lambda key: self.shared_attr(key),
            'share_attr': lambda key, val: self.set_shared_attr(key, val)
        }

    def initAllPlots(self, updateFig = True, try_sharing=True):

        try:
            self.setFiles()
        except Exception:
            pass

        if not self.PLOTS_PROVIDED:

            SINGLE_CLASS = isinstance(self._plotClasses, type)

            # Initialize all the plots
            # In case there is only one class only initialize them, avoid reading data
            # In this case, it is extremely important to initialize them all in serial mode, because
            # with multiprocessing they won't know that the current instance is their parent
            # (objects get copied in multiprocessing) and they won't be able to share data
            plots = initMultiplePlots(
                self._plotClasses, 
                kwargsList = [
                    {**kwargs, "attrs_for_plot": self._attrs_for_childplots, "only_init": SINGLE_CLASS and try_sharing} 
                    for kwargs in self._getInitKwargsList()
                ],
                serial=SINGLE_CLASS and try_sharing
            )

            if SINGLE_CLASS:

                # Then, we read the data of a leading plot
                # This leading plot will share attributes with the rest in case it is needed
                plots[0]._SHOULD_SHARE_WITH_SIBLINGS = True
                plots[0].readData(updateFig=False)
                plots[0]._SHOULD_SHARE_WITH_SIBLINGS = False

                # Now, we get the settings of the first plot
                read_data_settings = {key: plots[0].getSetting(key) for key, func in plots[0].whatToRunOnUpdate.items() if func == "readData"}

                for i, plot in enumerate(plots):
                    if not plot.has_this_settings(read_data_settings):
                        self.initAllPlots(try_sharing=False)
                else:
                    # In case there is no plot that has different settings
                    self.set_child_plots(plots)

                    self.setData()
                
            if callable( getattr(self, "_afterChildsUpdated", None )):
                self._afterChildsUpdated()

        if updateFig:
            self.getFigure()

        return self
    
    def updateSettings(self, **kwargs):
        '''
        This method takes into account that on plots that contain childs, one may want to update only the parent settings or all the child's settings.

        Use
        ------
        Call update settings
        '''

        if kwargs.get("onlyOnParent", False) or kwargs.get("exFromDecorator", False):

            return super().updateSettings(**kwargs)
        
        else:

            repeatIfChilds(Configurable.updateSettings)(self, **kwargs)

            if callable( getattr(self, "_afterChildsUpdated", None )):
                self._afterChildsUpdated()

            return self

    def set_child_plots(self, plots, keep=False):
        '''
        Sets the childplots of a multiple plot

        Parameters
        --------
        plots: array-like of sisl.viz.Plot or plotly Figure
            the plots that should be set as childplots for the animation. 
        keep: boolean, optional
            whether the existing childplots should be kept.

            If `True`, `plots` is added after them.
        '''

        # Maybe one of the plots is a plotly figure, normalize all to the plot class
        plots = [Plot.from_plotly(plot) if isinstance(plot, go.Figure) else plot for plot in plots]

        for plot in plots:

            for key, val in self._attrs_for_childplots.items():
                setattr(plot, key, val)

        self.childPlots = plots if not keep else [*self.childPlots, *plots]

        return self
    
    def add_childplots(self, *plots):
        '''
        Append childplots to the existing ones
        '''

        self.set_child_plots(plots, keep=True)
    
    def insert_childplot(self, index, plot):
        '''
        Inserts a plot in a given position of the childplots list

        Parameters
        ----------
        index: int
            The position where the plot should be inserted
        plot: sisl Plot or plotly Figure
            The plot to insert in the list
        '''

        if isinstance(plot, go.Figure):
            plot = Plot.from_plotly(plot)
        
        self.childPlots.insert(index, plot)
           
    def shared_attr(self, key):

        return self.shared[key]
    
    def set_shared_attr(self, key, val):
        
        self.shared[key] = val

        return self

class Animation(MultiplePlot):

    _isAnimation = True

    _paramGroups = (

        {
            "key": "animation",
            "name": "Animation specific settings",
            "icon": "videocam",
            "description": "The fact that you have not studied cinematography is not a good excuse for creating ugly animations. <b>Customize your animation with these settings</b>"
        },

    )

    _parameters = (

        IntegerInput(
            key = "frameDuration", name = "Frame duration",
            default = 500,
            group = "animation",
            params = {
                "step": 100
            },
            help = "Time (in ms) that each frame will be displayed. <br> This is only meaningful if you have an animation"
        ),

    )

    def __init__(self, *args, frameNames=None, _plugins={}, **kwargs):
        
        if frameNames is not None:
            _plugins["_getFrameNames"] = frameNames if callable(frameNames) else lambda self: frameNames
        
        super().__init__(*args, **kwargs, _plugins=_plugins)
