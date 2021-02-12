"""
This file contains the Plot class, which should be inherited by all plot classes
"""
import uuid
from io import StringIO, BytesIO
import inspect
import numpy as np
from copy import deepcopy
from collections import defaultdict
import time
from types import MethodType, FunctionType
import itertools
from functools import partial
from pathlib import Path

import dill
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import sisl
from sisl.messages import info, warn

from .configurable import (
    Configurable, ConfigurableMeta,
    vizplotly_settings, _populate_with_settings
)
from ._presets import get_preset
from .plotutils import (
    init_multiple_plots, repeat_if_childs, dictOfLists2listOfDicts,
    trigger_notification, spoken_message,
    running_in_notebook, check_widgets, call_method_if_present
)
from .input_fields import (
    TextInput, SileInput, SwitchInput,
    ColorPicker, DropdownInput, IntegerInput,
    FloatInput, RangeSlider, QueriesInput,
    ProgramaticInput, PlotableInput
)
from ._shortcuts import ShortCutable


__all__ = ["Plot", "MultiplePlot", "Animation", "SubPlots"]


class PlotMeta(ConfigurableMeta):

    def __call__(cls, *args, **kwargs):
        """ This method decides what to return when the plot class is instantiated.

        It is supposed to help the users by making the plot class very functional
        without the need for the users to use extra methods.

        It will catch the first argument and initialize the corresponding plot
        if the first argument is:
            - A string, it will be assumed that it is a path to a file.
            - A plotable object (has a _plot attribute)

        Note that both cases are registered in the _plotables.py file, and you
        can register new siles/plotables by using the register functions.
        """
        if args:

            # This is just so that the plotable framework knows from which plot class
            # it is being called so that it can build the corresponding plot.
            # Only relevant if the plot is built with obj.plot()
            plot_method = kwargs.get("plot_method", cls.suffix())

            # If a filename is recieved, we will try to find a plot for it
            if isinstance(args[0], (str, Path)):

                filename = args[0]
                sile = sisl.get_sile(filename)

                if sile.__class__ == sisl.io.siesta.fdfSileSiesta:
                    kwargs["root_fdf"] = filename
                    plot = cls(**kwargs)
                else:
                    if hasattr(sile, "plot"):
                        plot = sile.plot(**{**kwargs, "method": plot_method})
                    else:
                        raise NotImplementedError(
                            f'There is no plot implementation for {sile.__class__} yet.')

            elif isinstance(args[0], go.Figure):
                plot = Plot.from_plotly(args[0]).update_settings(**kwargs)
            elif isinstance(args[0], Plot):
                plot = args[0].update_settings(**kwargs)
            else:
                obj = args[0]
                # Maybe the first argument is a plotable object (e.g. a geometry)
                if hasattr(obj, "plot"):
                    plot = obj.plot(**{**kwargs, "method": plot_method})
                else:
                    return object.__new__(cls)

            return plot

        elif 'animate' in kwargs or 'varying' in kwargs or 'subplots' in kwargs:

            methods = {'animate': cls.animated, 'varying': cls.multiple, 'subplots': cls.subplots}
            # Retrieve the keyword that was actually passed
            # and choose the appropiate method
            for keyword in ('animate', 'varying', 'subplots'):
                variable_settings = kwargs.pop(keyword, None)
                if variable_settings is not None:
                    method = methods[keyword]
                    break

            # Normalize all accepted input types to a dict
            if isinstance(variable_settings, str):
                variable_settings = [variable_settings]
            if isinstance(variable_settings, (list, tuple, np.ndarray)):
                variable_settings = {key: kwargs.pop(key) for key in variable_settings}

            # Just run the method that will get us the desired plot
            plot = method(variable_settings, fixed=kwargs, **kwargs)

            return plot

        return super().__call__(cls, *args, **kwargs)


class Plot(ShortCutable, Configurable, metaclass=PlotMeta):
    """ Parent class of all plot classes

    Implements general things needed by all plots such as settings and shortcut
    management.

    Parameters
    ----------
    reading_order: None, optional
        Order in which the plot tries to read the data it needs.
    root_fdf: str, optional
        Path to the fdf file that is the 'parent' of the results.
    results_path: str, optional
        Directory where the files with the simulations results are
        located. This path has to be relative to the root fdf.
    animate: str, array-like of str or dict, optional
        the settings to animate. 
        If it's a dict it should contain the values of the settings for each frame.
        You only need to pass the ones that are changing! The rest of settings you can
        still specify them as usual.

        If it's a string or a list of strings, it will basically create the above mentioned
        dictionary using the values that you passed. E.g:
            object.plot(i=[1,4,5], animate="i")
        will create a plot with 3 frames where i=1, i=4 and i=5. Note that of course you should
        provide a list, tuple, etc... for the values of the settings that you want to animate.
    varying: str, array-like of str or dict, optional
        works exactly like `animate` but will create a MultiplePlot instead of an Animation. That is,
        it will render all the different plots in the same canvas.
    subplots: str, array-like of str or dict, optional
        works exactly like `animate` but will create a SubPlots instead of an Animation. That is,
        it will display a layout with subplots corresponding to each configuration.

    Attributes
    ----------
    settings: dict
        contains the values for each setting of the plot.
    params: dict
        for each setting, contains information about their input field.

        This information might include valid values, for example.
    paramGroups: tuple of dicts
        contains the different setting groups present in the plot.

        Each group is a dict like { "key": , "name": ,"icon": , "description": }
    ...

    """
    _update_methods = {
        "read_data": [],
        "set_data": [],
        "get_figure": []
    }

    _param_groups = (
        {
            "key": "dataread",
            "name": "Data reading settings",
            "icon": "import_export",
            "description": "In such a busy world, one may forget how the files are structured in their computer. Please take a moment to <b>make sure your data is being read exactly in the way you expect<b>."
        },

    )

    _parameters = (

        SileInput(
            key = "root_fdf", name = "Path to fdf file",
            dtype=sisl.io.siesta.fdfSileSiesta,
            group = "dataread",
            help = "Path to the fdf file that is the 'parent' of the results.",
            params = {
                "placeholder": "Write the path here..."
            }
        ),

        TextInput(
            key = "results_path", name = "Path to your results",
            group = "dataread",
            default = "",
            params = {
                "placeholder": "Write the path here..."
            },
            width = "s100% m50% l33%",
            help = "Directory where the files with the simulations results are located.<br> This path has to be relative to the root fdf.",
        ),

    )

    @property
    def read_data_methods(self):
        entry_points_names = [entry_point._method.__name__ for entry_point in self.entry_points]

        return ["_before_read", "_after_read", *entry_points_names, *self._update_methods["read_data"]]

    @property
    def set_data_methods(self):
        return ["_set_data", *self._update_methods["set_data"]]

    @property
    def get_figure_methods(self):
        return ["_after_get_figure", *self._update_methods["get_figure"]]

    def _parse_update_funcs(self, func_names):
        """ Decides which functions to run when the settings of the plot are updated

        This is called in self._run_updates as a final oportunity to decide what functions to run.

        In the case of plots, all we basically want to know is if we need to read the data again (execute
        `read_data`), set new data for the plot without needing to read (execute `set_data`) or just update 
        some aesthetic aspects of the plot (execute `get_figure`).

        When Plot sees one of the following functions in the list of functions with updated parameters:
            - "_before_read", "_after_read", any entry point function, functions in cls._update_methods["read_data"]

        it knows that `read_data` needs to be executed. (which afterwards triggers `set_data` and `get_figure`).

        Otherwise, if any of this functions is present:
            - "_set_data", functions in cls._update_methods["set_data"]

        it executes `set_data` (and `get_figure` subsequentially)

        Finally, if it finds:
            - "_after_get_figure", functions in cls._update_methods["get_figure"]

        it executes `get_figure`.

        WARNING: If it doesn't find any of these, it will return the unparsed list of functions,
        and all functions will get executed.

        Parameters
        -----------
        func_names: set of str
            the unique functions names that are to be executed unless you modify them.

        Returns
        -----------
        array-like of str
            the final list of functions that will be executed.

        See also
        ------------
        ``Configurable._run_updates``
        """
        if len(func_names.intersection(self.read_data_methods)) > 0:
            return ["read_data"]

        if len(func_names.intersection(self.set_data_methods)) > 0:
            return ["set_data"]

        if len(func_names.intersection(self.get_figure_methods)) > 0:
            return ["get_figure"]

        return func_names

    @classmethod
    def from_plotly(cls, plotly_fig):
        """ Converts a plotly plot to a Plot object

        Parameters
        -----------
        plotly_fig: plotly.graph_objs.Figure
            the figure that we want to convert into a sisl plot

        Returns
        -----------
        Plot
            the converted plot that contains the plotly figure.
        """
        plot = cls(only_init=True)
        plot.figure = plotly_fig

        return plot

    @classmethod
    def plot_name(cls):
        """ The name of the plot. Used to be displayed in the GUI, for example """
        return getattr(cls, "_plot_type", cls.__name__)

    @classmethod
    def suffix(cls):
        """ Get the suffix that this class adds to plotting functions

        See sisl/viz/_plotables.py and particularly the `register_plotable`
        function to understand this better.
        """
        if cls is Plot:
            return None
        return getattr(cls, "_suffix", cls.__name__.lower().replace("plot", ""))

    @classmethod
    def entry_points_help(cls):
        """ Generates a helpful message about the entry points of the plot class """
        string = ""

        for entry_point in cls.entry_points:

            string += f"{entry_point._name.capitalize()}\n------------\n\n"
            string += (entry_point.help or "").lstrip()

            string += "\nSettings used:\n\t- "
            string += '\n\t- '.join(map(lambda ab: ab[1], entry_point._method._settings))
            string += "\n\n"

        return string

    @property
    def _innotebook(self):
        """ Boolean indicating whether this plot is being used in a notebook

        Used to understand how we should display the plot.
        """
        return running_in_notebook()

    @property
    def _widgets(self):
        """ Dictionary that informs of which jupyter notebook widgets are available """
        return check_widgets()

    @classmethod
    def multiple(cls, *args, fixed={}, template_plot=None, merge_method='together', **kwargs):
        """ Creates a multiple plot out of a class

        This class method returns a multiple plot that contains several plots of the same class.
        It is a general method that serves as a common denominator for building MultiplePlot, SubPlot
        or Animation objects. Therefore, it is used by the `animated` and `subplots` methods

        If no arguments are passed, you will get the default multiple plot for the class, if there is any.

        Parameters
        -----------
        *args:
            Depending on what you pass the arguments will be interpreted differently:
                - Two arguments:
                    First: str
                        Key of the setting that you want to be variable.
                    Second: array-like
                        Values that you want the setting to have in each individual plot.

                    Ex: BandsPlot.multiple("bands_file", ["file1", "file2", "file3"] )
                    will produce a multiple plot where each plot uses a different bands_file.

                - One argument and it is a dictionary:
                    First: dict
                        The keys of this dictionary will be the setting keys you want to be variable
                        and the values are of course the values for each plot for that setting. 

                    It works exactly as the previous case, but in this case we have multiple settings that vary.

                - One argument and it is a function:
                    First: function
                        With this function you can produce any settings you want without limitations.

                        It will be used as the `_getInitKwargsList` method of the MultiplePlot object, so it needs to
                        accept self (the MultiplePlot object) and return a list of dictionaries which are the 
                        settings for each plot.

        fixed: dict, optional
            A dictionary containing values for settings that will be fixed for all plots.
            For the settings that you don't specify here you will get the defaults.
        template_plot: sisl Plot, optional
            If provided this plot will be used as a template.

            It is important to know that it will not act only as the settings template,
            but it will also PROVIDE DATA FOR THE OTHER PLOTS in case the data reading
            settings are not being varied through the different plots.

            This is extremely important to provide when possible, because in some cases the data
            that a plot gathers can be very large and therefore it may not be even feasable to store
            the repeated data in terms of memory/time.
        merge_method: {'together', 'subplots', 'animation'}, optional
            the way in which the multiple plots are 'related' to each other (i.e. how they should be displayed).
            In most cases, instead of using this argument, you should probably use the specific method (`animated` or
            `subplots`). They set this argument accordingly but also do some other work to make your life easier.
        **kwargs:
            Will be passed directly to initialization, so it can contain the settings for the MultiplePlot object, for example.

            If args are not passed and the default multiple plot is being created, some keyword arguments may be used by the method
            that generates the default multiple plot. One recurrent example of this is the keyword `wdir`. 

        Returns
        --------
        MultiplePlot, SubPlots or Animation
            The plot that you asked for. 
        """
        #Try to retrieve the default animation if no arguments are provided
        if len(args) == 0:

            return call_method_if_present(cls, "_default_animation", fixed=fixed, **kwargs)

        #Define how the getInitkwargsList method will look like
        if callable(args[0]):
            _getInitKwargsList = args[0]
        else:
            if len(args) == 2:
                variable_settings = {args[0]: args[1]}
            elif isinstance(args[0], dict):
                variable_settings = args[0]

            def _getInitKwargsList(self):

                #Adding the fixed values to the list
                vals = {
                    **{key: itertools.repeat(val) for key, val in fixed.items()},
                    **variable_settings
                }

                return dictOfLists2listOfDicts(vals)

        # Choose the specific class that we want to initialize
        MultipleClass = {'together': MultiplePlot, 'subplots': SubPlots, 'animation': Animation}[merge_method]

        #Return the initialized multiple plot
        return MultipleClass(_plugins={
            "_getInitKwargsList": _getInitKwargsList,
            "_plot_classes": cls,
            **kwargs.pop('_plugins', {})
        }, template_plot=template_plot, **kwargs)

    @classmethod
    def subplots(cls, *args, fixed={}, template_plot=None, rows=None, cols=None, arrange="rows", **kwargs):
        """ Creates subplots where each plot has different settings

        Mainly, it uses the `multiple` method to generate them.

        Parameters
        -----------
        *args:
            Depending on what you pass the arguments will be interpreted differently:
                - Two arguments:
                    First: str
                        Key of the setting that you want to vary across subplots.
                    Second: array-like
                        Values that you want the setting to have in each subplot.

                    Ex: BandsPlot.multiple("bands_file", ["file1", "file2", "file3"] )
                    will produce a layout where each subplot uses a different bands_file.

                - One argument and it is a dictionary:
                    First: dict
                        The keys of this dictionary will be the setting keys you want to vary across subplots
                        and the values are of course the values for each plot for that setting. 

                    It works exactly as the previous case, but in this case we have multiple settings that vary.

                - One argument and it is a function:
                    First: function
                        With this function you can produce any settings you want without limitations.

                        It will be used as the `_getInitKwargsList` method of the MultiplePlot object, so it needs to
                        accept self (the MultiplePlot object) and return a list of dictionaries which are the 
                        settings for each plot.

        fixed: dict, optional
            A dictionary containing values for settings that will be fixed for all subplots.
            For the settings that you don't specify here you will get the defaults.
        template_plot: sisl Plot, optional
            If provided this plot will be used as a template.

            It is important to know that it will not act only as the settings template,
            but it will also PROVIDE DATA FOR THE OTHER PLOTS in case the data reading
            settings are not being varied through the different plots.

            This is extremely important to provide when possible, because in some cases the data
            that a plot gathers can be very large and therefore it may not be even feasable to store
            the repeated data in terms of memory/time.
        rows: int, optional
            The number of rows of the plot grid. If not provided, it will be inferred from `cols`
            and the number of plots. If neither `cols` or `rows` are provided, the `arrange` parameter will decide
            how the layout should look like.
        cols: int, optional
            The number of columns of the subplot grid. If not provided, it will be inferred from `rows`
            and the number of plots. If neither `cols` or `rows` are provided, the `arrange` parameter will decide
            how the layout should look like.
        arrange: {'rows', 'col', 'square'}, optional
            The way in which subplots should be aranged if the `rows` and/or `cols`
            parameters are not provided.
        **kwargs:
            Will be passed directly to SubPlots initialization, so it can contain the settings for it, for example.

            If args are not passed and the default multiple plot is being created, some keyword arguments may be used by the method
            that generates the default multiple plot. One recurrent example of this is the keyword `wdir`. 
        """
        return cls.multiple(*args, fixed=fixed, template_plot=None, merge_method='subplots',
                rows=rows, cols=cols, arrange=arrange, **kwargs)

    @classmethod
    def animated(cls, *args, fixed={}, frame_names=None, template_plot=None, **kwargs):
        """ Creates an animation out of a class

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

                    Ex: BandsPlot.animated("bands_file", ["file1", "file2", "file3"] )
                    will produce an animation where each frame uses a different bands_file.

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

                    Ex: obj.modify_param("length", lambda param: param.incrementByOne() )

                    given that you know that this type of parameter has this method.
        fixed: dict, optional
            A dictionary containing values for settings that will be fixed along the animation.
            For the settings that you don't specify here you will get the defaults.
        frame_names: list of str or function, optional
            If it is a list of strings, each string will be used as the name for the corresponding frame.

            If it is a function, it should accept `self` (the animation object) and return a list of strings
            with the frame names. Note that you can access the plot instance responsible for each frame under
            `self.child_plots`. The function will be run each time the figure is generated, so in this way your
            frame names will be dynamic.

            FRAME NAMES SHOULD BE UNIQUE, OTHERWISE THE ANIMATION WILL HAVE A WEIRD BEHAVIOR.

            If this is not provided, frame names will be generated automatically.
        template_plot: sisl Plot, optional
            If provided this plot will be used as a template.

            It is important to know that it will not act only as the settings template,
            but it also will PROVIDE DATA FOR THE OTHER PLOTS in case the data reading
            settings are not animated.

            This is extremely important to provide when possible, because in some cases the data
            that a plot gathers can be very large and therefore it may not be even feasable to store
            the repeated data in terms of memory/time. 
        **kwargs:
            Will be passed directly to animation initialization, so it can contain the settings for the animation, for example.

            If args are not passed and the default animation is being created. Some keyword arguments may be used by the method
            that generates the default animation. One recurrent example of this is the keyword `wdir`. 

        Returns
        --------
        Animation
            The Animation that you asked for
        """
        # And just let the general multiple plot creator do the work
        return cls.multiple(*args, fixed=fixed, template_plot=template_plot, merge_method='animation',
                            frame_names=frame_names, **kwargs)

    def __init_subclass__(cls):
        """ Whenever a plot class is defined, this method is called

        We will use this opportunity to:
            - Register entry points.
            - Generate a more helpful __init__ method that exposes all the settings.

        We could use this to register plotables (see commented code).
        However, there is one major problem: how to specify defaults.
        This is a problem because sometimes an input field is inherited from one plot
        to another, therefore you can not say: "this is the default plotable input".

        Probably, defaults should be centralized, but I don't know where just yet.
        """
        super().__init_subclass__()

        # Register the entry points of this class.
        cls.entry_points = []
        for key, val in inspect.getmembers(cls, lambda x: isinstance(x, EntryPoint)):
            cls.entry_points.append(val)
            # After registering an entry point, we will just set the method
            setattr(cls, key, _populate_with_settings(val._method, [param["key"] for param in cls._get_class_params()[0]]))

        # from ._plotables import register_plotly_plotable

        # for param in cls._get_class_params()[0]:
        #     if isinstance(param, PlotableInput):
        #         register_plotly_plotable(param.dtype, cls, param.key)

    @vizplotly_settings('before', init=True)
    def __init__(self, *args, H = None, attrs_for_plot={}, only_init=False, presets=None, layout={}, _debug=False, **kwargs):
        # Give an ID to the plot
        self.id = str(uuid.uuid4())

        # Inform whether the plot is in debug mode or not:
        self._debug = _debug

        # Initialize the figure
        self.figure = go.Figure()

        # Initialize shortcut management
        ShortCutable.__init__(self)

        #Give the user the possibility to do things before initialization (IDK why)
        call_method_if_present(self, "_before_init")

        #Set the isChildPlot attribute to let the plot know if it is part of a bigger picture (e.g. Animation)
        self.isChildPlot = kwargs.get("isChildPlot", False)

        #Initialize the variable to store when has been the last data read (0 means never basically)
        self.last_dataread = 0
        self._files_to_follow = []

        # Check if the user has provided a hamiltonian (which can contain a geometry)
        # This is not meant to be used by the GUI (in principle), just programatically
        self.PROVIDED_H = False
        self.PROVIDED_GEOM = False
        if H is not None:
            self.PROVIDED_H = True
            self.H = H
            self.setup_hamiltonian()

        # Update its layout if a layout is provided
        self.update_layout(**getattr(self.__class__, "_layout_defaults", {}), **layout)

        if presets is not None:
            if isinstance(presets, str):
                presets = [presets]
            for preset in presets:
                self.update_layout(**get_preset(preset)['layout'])

        # on_figure_change is triggered after get_figure.
        self.on_figure_change = None

        # Set all the attributes that have been passed
        # It is important that this is here so that it can overwrite any of
        # the already written attributes
        for key, val in attrs_for_plot.items():
            setattr(self, key, val)

        #If plugins have been provided, then add them.
        #Plugins are an easy way of extending a plot. They can be methods, variables...
        #They are added to the object instance, not the whole class.
        if kwargs.get("_plugins"):
            for name, plugin in kwargs.get("_plugins").items():
                if isinstance(plugin, FunctionType):
                    plugin = MethodType(plugin, self)
                setattr(self, name, plugin)

        # Add the general plot shortcuts
        self._general_plot_shortcuts()

        #Give the user the possibility to overwrite default settings
        call_method_if_present(self, "_after_init")

        # If we were supposed to only initialize the plot, stop here
        if only_init:
            return

        #Try to generate the figure (if the settings required are still not there, it won't be generated)
        try:
            if MultiplePlot in type.mro(self.__class__):
                #If its a multiple plot try to inititialize all its child plots
                if self.PLOTS_PROVIDED:
                    self.get_figure()
                else:
                    self.init_all_plots()
            else:
                self.read_data()

        except Exception as e:
            if self._debug:
                raise e
            info(f"The plot has been initialized correctly, but the current settings were not enough to generate the figure.\nError: {e}")

    def __str__(self):
        """ Information to print about the plot """
        string = (
            f'Plot class: {self.__class__.__name__}    Plot type: {self.plot_name()}\n\n'
            'Settings:\n{}'.format("\n".join(["\t- {}: {}".format(key, value) for key, value in self.settings.items()]))
        )

        return string

    def __getattr__(self, key):
        """ This method is executed only after python has found that there is no such attribute in the instance

        So let's try to find it elsewhere. There are two options:
            - The attribute is in the figure object (self.figure)
            - The attribute is currently being shared with other plots (only possible if it's a childplot)
        """
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
        """ If the attribute is one of ["data", "layout", "frames"] we are going to store it directly in `self.figure` for convenience and in order to save memory.

        If is a childplot and it has the attribute `_SHOULD_SHARE_WITH_SIBLINGS` set to True, we will submit the attribute to the shared store.
        This happens in animations/multiple plots. There's a "leading plot" that reads the data and then shares it with the rest
        so that they don't need to read it again, in a collective effort to save memory and time.

        Otherwise we set the attribute to the plot itself.
        """
        if key in ["data", "layout", "frames"]:
            self.figure.update(**{key: val})
        elif key != '_SHOULD_SHARE_WITH_SIBLINGS' and getattr(self, '_SHOULD_SHARE_WITH_SIBLINGS', False):
            self.share_attr(key, val)
        else:
            object.__setattr__(self, key, val)

    def __getitem__(self, key):
        """ Getting an item from plot returns the trace(s) that correspond to the requested indices """
        if isinstance(key, (int, slice)):
            return self.data[key]

    def _general_plot_shortcuts(self):
        """ In this method we set the shortcuts that are general to all plots

        This is called in `__init__`
        """
        self._listening_shortcut()

        self.add_shortcut("ctrl+z", "Undo settings", self.undo_settings, _description="Takes the settings of the plot one step back")

    @repeat_if_childs
    @vizplotly_settings('before')
    def read_data(self, update_fig=True, **kwargs):
        """ This method is responsible for organizing the data-reading step

        If everything is done succesfully, it calls the next step (`set_data`)
        """
        # Restart the files_to_follow variable so that we can start to fill it with the new files
        # Apart from the explicit call in this method, setFiles and setup_hamiltonian also add files to follow
        self._files_to_follow = []

        call_method_if_present(self, "_before_read")

        # Update the title of the plot if there is none
        if not self.figure.layout["title"]:
            self.update_layout(title = '{} {}'.format(getattr(self, "struct", ""), self.plot_name()))

        # We try to read from the different entry points available
        self._read_from_sources()

        # We don't update the last dataread here in case there has been a succesful data read because we want to
        # wait for the after_read() method to be succesful
        if self.source is None:
            self.last_dataread = 0

        call_method_if_present(self, "_after_read")

        if self.source is not None:
            self.last_dataread = time.time()

        if update_fig:
            self.set_data(update_fig = update_fig)

        return self

    def _read_from_sources(self):
        """ Tries to read the data from the different available entry points in the plot class

        If it fails to read from all entry points, it raises an exception.
        """
        # It is possible that the class does not implement any entry points,
        # because it doesn't need to read any data. Then the plotting process
        # will basically start at set_data.
        if not self.entry_points:
            return

        errors = []
        # Try to read data using all the different entry points
        # This is just a first implementation. One of the reasons entry points
        # have been implemented is that we can do smarter things than this.
        for entry_point in self.entry_points:
            try:
                returns = getattr(self, entry_point._method_attr)()
                self.source = entry_point
                return returns
            except Exception as e:
                errors.append("\t- {}: {}.{}".format(entry_point._name, type(e).__name__, e))
        else:
            self.source = None
            raise ValueError("Could not read or generate data for {} from any of the possible sources.\nHere are the errors for each source:\n{}"
                             .format(self.__class__.__name__, "\n".join(errors)))

    def follow(self, *files, to_abs=True, unfollow=False):
        """ Makes sure that the object knows which files to follow in order to trigger updates

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
        """
        new_files = [Path(file_path).resolve() if to_abs else Path(file_path) for file_path in files or []]

        self._files_to_follow = new_files if unfollow else [*self._files_to_follow, *new_files]

    def get_sile(self, path, results_path, root_fdf, *args, follow=True, follow_kwargs={}, file_contents=None, **kwargs):
        """ A wrapper around get_sile so that the reading of the file is registered

        It has to main functions:
            - Automatically following files that are read, so that you don't neet to go always like:

                ```
                self.follow(file)
                sisl.get_sile(file)
                ```
            - Infering files from a root file. For example, using the root_fdf. 

        Parameters
        ----------
        path: str
            the path to the file that you want to read.
            It can also be the setting key that you want to read.
        *args:
            passed to sisl.get_sile
        follow: boolean, optional
            whether the path should be followed.
        follow_kwargs: dict, optional
            dictionary of keywords that are passed directly to the follow method.
        **kwargs:
            passed to sisl.get_sile
        """
        # If path is a setting name, retrieve it
        if path in self.settings:
            setting_key = path
            path = self.get_setting(path)

            # However, if it wasn't provided, try to infer it.
            # For example, if it is a siesta sile, we will try to infer it
            # from the fdf file
            if not path:

                sile_type = self.get_param(setting_key).dtype
                # We need to check here if it is a SIESTA related sile!

                fdf_sile = sisl.get_sile(root_fdf)

                for rule in sisl.get_sile_rules(cls=sile_type):
                    filename = fdf_sile.get('SystemLabel', default='siesta') + f'.{rule.suffix}'
                    try:
                        path = fdf_sile.dir_file(filename, results_path)
                        return self.get_sile(path, *args, follow=True, follow_kwargs={}, file_contents=None, **kwargs)
                    except:
                        pass
                else:
                    raise FileNotFoundError(f"Tried to infer {setting_key} from the 'root_fdf', "
                    f"but didn't find any {sile_type.__name__} in {Path(fdf_sile._directory) / results_path }")

        if follow:
            self.follow(path, **follow_kwargs)

        return sisl.get_sile(path, *args, **kwargs)

    def updates_available(self):
        """ This function checks whether the read files have changed

        For it to work properly, one should specify the files that have been read by
        their reading methods (usually, the entry points). This is done by using the 
        `follow()` method or by reading files with `self.get_sile()` instead of `sisl.get_sile()`.
        """
        def modified(filepath):

            try:
                return filepath.stat().st_mtime > self.last_dataread
            except FileNotFoundError:
                return False  # This probably should implement better logic

        files_modified = np.array([modified(file_path) for file_path in self._files_to_follow])

        return files_modified.any()

    def listen(self, forever=True, show=True, as_animation=False, return_animation=True, return_figWidget=False,
               clear_previous=True, notify=False, speak=False, notify_title=None, notify_message=None, speak_message=None, fig_widget=None):
        """ Listens for updates in the followed files (see the `updates_available` method)

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
        clear_previous: boolean, optional
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
        """
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

                        self.read_data(update_fig=True)

                        if as_animation:
                            new_plot = self.clone()
                            pt.add_child_plots(new_plot)
                            pt.get_figure()

                        if clear_previous and fig_widget is None:
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
                            trigger_notification(title, message)
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
        """ Adds the shortcut to start listening for updates

        This is done here and not in `_general_plot_settings` because
        we need to be able to toggle this shortcut each time the plot
        starts/stops listening.

        Maybe at some point we can have a rule to automatically disable
        shortcuts based on state in `ShortCutable`.
        """
        self.add_shortcut(
            "ctrl+alt+l", "Listen for updates",
            self.listen, fig_widget=fig_widget,
            _description="Make the plot listen for changes in the files that it reads"
        )

    def stop_listening(self, fig_widget=None):
        """ Makes the plot stop listening for updates

        Using this method only makes sense if you have previously made the plot listen
        either through `Plot.listen()` or `Plot.show(listen=True)`

        Parameters
        -----------
        fig_widget: plotly FigureWidget, optional
            the figure widget where the plot is currently being displayed.

            This is just used to reset the listening shortcut.

            YOU WILL MOST LIKELY NOT USE THIS because `Plot` already knows
            where is it being displayed in normal situations.
        """
        task = getattr(self, "_listening_task", None)

        if task is not None:
            task.cancel()
            self._listening_task = None
            self._listening_shortcut(fig_widget=fig_widget)

        return self

    @vizplotly_settings('before')
    def setup_hamiltonian(self, **kwargs):
        """ Sets up the hamiltonian for calculations with sisl """
        NEW_FDF = True
        if len(self.settings_history) > 1:
            NEW_FDF = self.settings_history.was_updated("root_fdf")

        if not hasattr(self, "geometry") or NEW_FDF:
            try:
                fdf_sile = self.get_sile("root_fdf")
                self.geometry = fdf_sile.read_geometry(output = True)
            except:
                pass

        if not self.PROVIDED_H and (not hasattr(self, "H") or NEW_FDF):
            #Read the hamiltonian
            fdf_sile = self.get_sile("root_fdf")
            self.H = fdf_sile.read_hamiltonian()
        else:
            if isinstance(self.H, (str, Path)):
                self.H = self.get_sile(self.H)

            if isinstance(self.H, sisl.BaseSile):
                self.H = self.H.read_hamiltonian(geometry=getattr(self, "geometry", None))

        if not hasattr(self, "geometry"):
            self.geometry = self.H.geometry

        return self

    @repeat_if_childs
    @vizplotly_settings('before')
    def set_data(self, update_fig = True, **kwargs):
        """ Method to process the data that has been read beforehand by read_data() and prepare the figure

        If everything is succesful, it calls the next step in plotting (`get_figure`)
        """
        # Clear all the traces from the figure before drawing the new ones
        self.clear()

        # This is used to know how many traces have been added, so that the user can clear only
        # the traces written by the plot methods and keep traces that they have added later, if they want
        self._starting_traces = len(self.data)

        self._set_data()

        if update_fig:
            self.get_figure()

        # The explanation for this is above (in the definition of _starting_traces)
        self._own_traces_slice = slice(self._starting_traces, len(self.data))

        return self

    @vizplotly_settings('before')
    def get_figure(self, **kwargs):
        """
        This is here mainly for historic reasons. At the beggining, the plot's figure was not
        constructed until this point (each time a new figure was constructed). However, it seemed
        better to construct a figure only once on `__init__` so that it can be manipulated at any
        point of the plotting process.

        Therefore, there is nothing left to do here.

        HOWEVER, `MultiplePlot` and its child classes overwrite this method and actually make use of it,
        because they need to collect the data from all their child_plots and build the figure.

        This method can be applied after updating the data so that the plot object is refreshed.

        Returns
        ---------
        self.figure: plotly.graph_objs.Figure
            the plotly figure.
        """
        call_method_if_present(self, '_after_get_figure')

        call_method_if_present(self, 'on_figure_change')

        return self.figure

    #-------------------------------------------
    #       PLOT DISPLAY METHODS
    #-------------------------------------------

    def show(self, *args, listen=False, return_figWidget=False, **kwargs):
        """ Displays the plot

        Parameters
        ------
        listen: bool, optional
            after showing, keeps listening for file changes to update the plot.
            This is nice for monitoring.
        return_figureWidget: bool, optional
            if the plot is displayed in a jupyter notebook, whether you want to
            get the figure widget as a return so that you can act on it.
        """
        if listen:
            self.listen(show=True, **kwargs)

        if self._innotebook and (len(args) == 0 or 'config' in kwargs):
            try:
                return self._ipython_display_(listen=listen, return_figWidget=return_figWidget, **kwargs)
            except Exception as e:
                warn(e)

        return self.figure.show(*args, **kwargs)

    def _ipython_display_(self, return_figWidget=False, **kwargs):
        """ Handles all things needed to display the plot in a jupyter notebook

        Plotly already knows how to show a plot in the jupyter notebook, however
        here we try to extend it to support shortcuts if the appropiate widget is 
        there (ipyevents, https://github.com/mwcraig/ipyevents).

        Parameters
        ------
        return_figureWidget: bool, optional
            if the plot is displayed in a jupyter notebook, whether you want to
            get the figure widget as a return so that you can act on it.
        """

        if self._widgets["plotly"] and not isinstance(self, Animation):

            from IPython.display import display
            import ipywidgets as widgets

            f = go.FigureWidget(self.figure, )

            if self._widgets["events"]:
                # If ipyevents is available, show with shortcut support
                self._ipython_display_with_shortcuts(f, **kwargs)
            else:
                # Else, show without shortcut support
                display(f)

            self._listening_shortcut(fig_widget=f)

            if return_figWidget:
                return f

        else:
            kwargs.pop("listen", None)
            self.figure._ipython_display_(**kwargs)

    def _ipython_display_with_shortcuts(self, fig_widget, **kwargs):
        """
        If the appropiate widget is there (ipyevents, https://github.com/mwcraig/ipyevents),
        we extend plotly's FigureWidget to support keypress events so that we can trigger
        shortcuts from the notebook.

        Parameters
        ------
        fig_widget: plotly.graph_objs.FigureWidget
            The figure widget that we need to extend.
        """
        from ipyevents import Event
        from ipywidgets import HTML, Output

        h = HTML("") # This is to display help such as available shortcuts
        messages = HTML("") # This is to inform about current status
        styles = HTML("<style>.ipyevents-watched:focus {outline: none}</style>")
        d = Event(source=fig_widget, watched_events=['keydown', 'keyup'])

        def handle_dom_events(event, keys_down=[], last_timestamp=[0], keys_up=[]):
            # We will keep track of keydowns because then we will be able to support multiple keys shortcuts
            time_threshold = 500 #To remove key up events

            try:
                # Clear the list
                timestamp = event.get("timeStamp")
                duplicates = len(keys_down) != len(set(keys_down))
                time_diff = timestamp - last_timestamp[0]
                if time_diff > 2000 or duplicates:
                    keys_down *= 0 #Clear the list
                if time_diff > time_threshold:
                    keys_up *= 0

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
                    if key == "control":
                        key = "ctrl"
                    if len(keys_down) == 1:
                        keys_up.append(key)
                    # If it's a key up event, anounce that the key is not down anymore
                    keys_down.remove(key)

                only_down = "+".join(keys_down)
                shortcut_key = f'{" ".join(keys_up)} {only_down}'.strip()

                if shortcut_key:
                    messages.value = f'<span style="border-radius: 3px; background: #ccc; padding: 5px 10px">{shortcut_key}</span>'

                # Get the help message
                if shortcut_key == "shift+?":
                    h.value = self.shortcuts_summary("html") if not h.value else ""

                shortcut = self.shortcut(shortcut_key)
                if shortcut is not None:
                    keys_down *= 0
                    keys_up *= 0

                    messages.value = f'Executing "{shortcut["name"]}" because you pressed "{shortcut_key}"...'
                    self.call_shortcut(shortcut_key)
                    messages.value = ""

                    self._update_FigureWidget(fig_widget)

            except Exception as e:
                messages.value = f'<span style="color:darkred; font-weight: bold">{e}</span>'

        d.on_dom_event(partial(handle_dom_events))

        display(fig_widget, messages, h, styles, Output())

    def _update_FigureWidget(self, fig_widget, plot = None):
        """ Updates a figure widget so that it is in sync with this plot's data

        Parameters
        ----------
        fig_widget: plotly.graph_objs.FigureWidget
            The figure widget that we need to extend.
        """
        fig_widget.data = []
        fig_widget.add_traces(self.data)
        fig_widget.layout = self.layout
        fig_widget.update(frames=self.frames)

    #-------------------------------------------
    #       PLOT MANIPULATION METHODS
    #-------------------------------------------

    def merge(self, others, to="multiple", extend_multiples=True, **kwargs):
        """ Merges this plot's instance with the list of plots provided

        Parameters
        -------
        others: array-like of Plot() or Plot()
            the plots that we want to merge with this plot instance.
        to: {"multiple", "subplots", "animation"}, optional
            the merge method. Each option results in a different way of putting all the plots
            together:
            - "multiple": All plots are shown in the same canvas at the same time. Useful for direct
            comparison.
            - "subplots": The layout is divided in different subplots.
            - "animation": Each plot is converted into the frame of an animation.
        extend_multiples: boolean, optional
            if True, if `MultiplePlot`s are passed, they are splitted into their child_plots, so that the result
            is the merge of its child_plots with the rest.
            If False, a `MultiplePlot` is treated as a solid unit.
        kwargs:
            extra arguments that are directly passed to `MultiplePlot`, `Subplots`
            or `Animation` initialization.

        Returns
        -------
        MultiplePlot, Subplots or Animation
            depending on the value of the `to` parameter.
        """
        #Make sure we deal with a list (user can provide a single plot)
        if not isinstance(others, (list, tuple, np.ndarray)):
            others = [others]

        child_plots = [self, *others]
        if extend_multiples:
            child_plots = [[pt] if not isinstance(pt, MultiplePlot) else pt.child_plots for pt in child_plots]
            # Flatten the list
            child_plots = [pt for plots in child_plots for pt in plots]

        PlotClass = {
            "multiple": MultiplePlot,
            "subplots": SubPlots,
            "animation": Animation
        }[to]

        return PlotClass(plots=child_plots, **kwargs)

    def group_legend(self, by=None, names=None, show_all=False, extra_updates=None, **kwargs):
        """ Joins plot traces in groups in the legend

        As the result of this method, plot traces end up with a legendgroup attribute.
        You can use that for selecting traces in further processing of your plot.

        This also provides the ability to toggle the whole group from the legend, which is nice.

        Parameters
        ---------
        by: str or function, optional
            it defines what are the criteria to group the traces.

            If it's a string:
                It is the name of the trace attribute. Remember that plotly allows you to
                lookup for nested attributes using underscores. E.g: "line_color" gets {line: {color: THIS VALUE}}
            If it's a function:
                It will recieve each trace and needs to decide which group to put it in by returning the group value.
                Note that the value will also be used as the group name if `names` is not provided, so you can save yourself
                some code and directly return the group's name.
            If not provided:
                All traces will be put in the same group
        names: array-like, dict or function, optional
            it defines what the names of the generated groups will be.

            If it's an array:
                When a new group is found, the name will be taken from this array (order can be very arbitrary)
            If it's a dict:
                When a new group is found, the value of the group will be used as a key to get the name from this dictionary.
                If the key is not found, the name will just be the value.
                E.g.: If grouping by `line_color` and `blue` is found, the name will be `names.get('blue', 'blue')`
            If it's a function:
                It will recieve the group value and the trace and needs to return the name of the TRACE.
                NOTE: If `show_all` is set to `True` all traces will appear in the legend, so it would be nice
                to give them different names. Otherwise, you can just return the group's name.
                If you provided a grouping function and `show_all` is False you don't need this, as you can return
                directly the group name from there.
            If not provided:
                the values will be used as names.
        show_all: boolean, optional
            whether all the items of the group should be displayed in the legend.
            If `False`, only one item per group will be displayed.
            If `True`, all the items of the group will be displayed.
        extra_updates: dict, optional
            A dict stating extra updates that you want to do for each group.

            E.g.: `{"blue": {"line_width": 4}}`

            would also convert the lines with a group VALUE (not name) of "blue" to a width of 4.

            This is just for convenience so that you can run other methods after this one.
            Note that you can always do something like this by doing

            ```
            plot.update_traces(
                selector={"line_width": "blue"}, # Selects the traces that you will update
                line_width=4,
            ) 
            ```

            If you use a function to return the group values, there is probably no point on using this
            argument. Since you recieve the trace, you can run `trace.update(...)` inside your function.
        **kwargs:
            like extra_updates but they are passed to all groups without distinction
        """
        unique_values = []

        # Normalize the "by" parameter to a function
        if by is None:
            if show_all:
                name = names[0] if names is not None else "Group"
                self.figure.update_traces(showlegend=True, legendgroup=name, name=name)
                return self
            else:
                func = lambda trace: 0
        if isinstance(by, str):
            def func(trace):
                try:
                    return trace[by]
                except Exception:
                    return None
        else:
            func = by

        # Normalize also the names parameter to a function
        if names is None:
            def get_name(val, trace):
                return str(val) if not show_all else f'{val}: {trace.name}'
        elif callable(names):
            get_name = names
        elif isinstance(names, dict):
            def get_name(val, trace):
                name = names.get(val, val)
                return str(name) if not show_all else f'{name}: {trace.name}'
        else:
            def get_name(val, trace):
                name = names[len(unique_values) - 1]
                return str(name) if not show_all else f'{name}: {trace.name}'

        # And finally normalize the extra updates
        if extra_updates is None:
            get_extra_updates = lambda *args, **kwargs: {}
        elif isinstance(extra_updates, dict):
            get_extra_updates = lambda val, trace: extra_updates.get(val, {})
        elif callable(extra_updates):
            get_extra_updates = extra_updates

        # Build the function that will apply the change
        def check_and_apply(trace):

            val = func(trace)

            if isinstance(val, np.ndarray):
                val =  val.tolist()
            if isinstance(val, list):
                val = ", ".join([str(item) for item in val])

            if val in unique_values:
                showlegend = show_all
            else:
                unique_values.append(val)
                showlegend = True

            customdata = trace.customdata if trace.customdata is not None else [{}]

            trace.update(
                showlegend=showlegend,
                legendgroup=str(val),
                name=get_name(val, trace=trace),
                customdata=[{**customdata[0], "name": trace.name}, *customdata[1:]],
                **get_extra_updates(val, trace=trace),
                **kwargs
            )

        # And finally apply all the changes
        self.figure.for_each_trace(
            lambda trace: check_and_apply(trace)
        )

        return self

    def ungroup_legend(self):
        """ Ungroups traces if a legend contains groups """
        self.figure.for_each_trace(
            lambda trace: trace.update(
                legendgroup=None,
                showlegend=True,
                name=trace.customdata[0]["name"]
            )
        )

        return self

    def clear(self, plot_traces=True, added_traces=True, frames=True, layout=False):
        """ Clears the plot canvas so that data can be reset

        Parameters
        --------
        plot_traces: boolean, optional
            whether traces added by the plot's code should be cleared.
        added_traces: boolean, optional
            whether traces added externally (by the user) should be cleared.
        frames: boolean, optional
            whether frames should also be deleted
        """
        own_slice = getattr(self, '_own_traces_slice', slice(0, 0))

        if not plot_traces and added_traces:
            self.figure.data = self.figure.data[own_slice]
        elif not added_traces and plot_traces:
            self.figure.data = [trace for i, trace in enumerate(self.data) if i < own_slice.start or i >= own_slice.stop]
        else:
            self.figure.data = []

        if frames:
            self.figure.frames = []

        if layout:
            self.figure.layout = {}

        return self

    def normalize(self, min_val=0, max_val=1, axis="y", **kwargs):
        """ Normalizes traces to a given range along an axis

        Parameters
        -----------
        min_val: float, optional
            The lower bound of the range.
        max_val: float, optional
            The upper part of the range
        axis: {"x", "y", "z"}, optional
            The axis along which we want to normalize.
        **kwargs:
            keyword arguments that are passed directly to plotly's Figure `for_each_trace`
            method. You can check its documentation. One important thing is that you can pass a
            'selector', which will choose if the trace is updated or not. 
        """
        from .plotutils import normalize_trace

        self.for_each_trace(partial(normalize_trace, min_val=min_val, max_val=max_val, axis=axis), **kwargs)

        return self

    def swap_axes(self, ax1='x', ax2='y', **kwargs):
        """ Swaps two axes in the plot

        Parameters
        -----------
        ax1, ax2: str, {'x', 'x*', 'y', 'y*', 'z', 'z*'}
            The names of the axes that you want to swap. 
        **kwargs:
            keyword arguments that are passed directly to plotly's Figure `for_each_trace`
            method. You can check its documentation. One important thing is that you can pass a
            'selector', which will choose if the trace is updated or not. 
        """
        from .plotutils import swap_trace_axes
        # Swap the traces
        self.for_each_trace(partial(swap_trace_axes, ax1=ax1, ax2=ax2), **kwargs)

        # Try to also swap the axes
        try:
            self.update_layout({
                f'{ax1}axis': self.layout[f'{ax2}axis'].to_plotly_json(),
                f'{ax2}axis': self.layout[f'{ax1}axis'].to_plotly_json(),
            }, overwrite=True)
        except:
            pass

        return self

    def shift(self, shift, axis="y", **kwargs):
        """ Shifts the traces of the plot by a given value in the given axis

        Parameters
        -----------
        shift: float or array-like
            If it's a float, it will be a solid shift (i.e. all points moved equally).
            If it's an array, an element-wise sum will be performed
        axis: {"x","y","z"}, optional
            The axis along which we want to shift the traces.
        **kwargs:
            keyword arguments that are passed directly to plotly's Figure `for_each_trace`
            method. You can check its documentation. One important thing is that you can pass a
            'selector', which will choose if the trace is updated or not. 
        """
        from .plotutils import shift_trace

        self.for_each_trace(partial(shift_trace, shift=shift, axis=axis), **kwargs)

        return self

    def v_line(self, x, yrange=None, **kwargs):
        """ Draws a vertical line in the figure (NOT WORKING YET!) """
        if yrange is None:
            yrange = self.layout.yaxis.range
            if yrange is None:
                yrange = [0, 7000]

        self.add_scatter(mode = "lines", x = [x, x], y = yrange, **kwargs)

        return self

    def copy(self):
        """ Returns a copy of the plot

        If you want a plot with the exact plot configuration but newly initialized,
        use `clone()` instead.
        """
        return deepcopy(self)

    def clone(self, *args, **kwargs):
        """ Gets you and exact clone of this plot

        You can pass extra args that will overwrite the previous parameters though, if you don't want it to be that exact.

        IMPORTANT: IT WILL INITIALIZE A NEW PLOT, THEREFORE IT WILL READ NEW DATA.
        IF YOU JUST WANT A COPY, USE THE `copy()` method.
        """
        return deepcopy(self)

        return self.__class__(*args, **self.settings, **kwargs)

    #-------------------------------------------
    #          LISTENING TO EVENTS
    #-------------------------------------------

    def dispatch_event(self, event, *args, **kwargs):
        """ Not functional yet """
        warn((event, args, kwargs))
        # Of course this needs to be done
        raise NotImplementedError

    #-------------------------------------------
    #       DATA TRANSFER/STORAGE METHODS
    #-------------------------------------------

    def _get_pickleable(self):
        """ Removes from the instance the attributes that are not pickleable """
        # Currently there is nothing unpickleable in plots :)
        return self

    def save(self, path, html=False):
        """ Saves the plot so that it can be loaded in the future

        Parameters
        ---------
        path: str
            The path to the file where you want to save the plot
        html: bool
            If set to true, saves just an html file of the plot visualization.

        Returns
        ---------
        self
        """
        if isinstance(path, str):
            path = Path(path)

        if html or path.suffix == ".html":
            self.figure.write_html(str(path.with_suffix(".html")))
            return self

        #The following method actually modifies 'self', so there's no need to get the return
        self._get_pickleable()

        with open(path, 'wb') as handle:
            dill.dump(self, handle, protocol=dill.HIGHEST_PROTOCOL)

        return True

    def html(self, path):
        """ Just a shortcut for save( html = True )

        Arguments
        --------
        path: str
            The path to the file where you want to save the plot.
        """
        return self.save(path, html = True)

    def to_chart_studio(self, *args, **kwargs):
        """ Sends the plot to chart studio if it is possible

        For it to work, the user should have their credentials correctly set up.

        It is a shortcut for chart_studio.plotly.plot(self.figure, ...etc) so you can pass any extra arguments as if
        you were using `py.plot`
        """
        import chart_studio.plotly as py

        return py.plot(self.figure, *args, **kwargs)


class EntryPoint:

    def __init__(self, name, setting_key, method, instance=None):
        self._name = name
        self._method_attr = method.__name__
        self._setting_key = setting_key
        self._method = method
        self.help = method.__doc__


def entry_point(name):
    """ Helps registering entry points for plots

    See the usage section to get a fast intuitive way of how to use it.

    Basically, you need to provide some parameters (which are described
    in the parameters section), and this function will return a decorator that
    you can use in the functions of your plot class that do the reading part.

    A function that is meant to read data but it's not marked as an entry_point
    will be invisible to Plot.

    NOTE: A plot class can have no entry points. This is perfectly fine if the 
    class does not need to read data for some reason. In this case, we will go straight
    into the data setting methods (i.e. set_data).

    Examples
    -----------

    >>> class MyPlot(Plot):
    >>>     @entry_point('siesta_output')
    >>>     def _lets_read_from_siesta_output(self):
    >>>         ...do some work here
    >>> 
    >>>     @entry_point('ask_mum'):
    >>>     def _we_are_quite_lost_so_we_better_ask_mum(self):
    >>>         self.call_mum()

    Parameters
    -----------
    name: str
        the name of the entry point that the decorated function implements.
    """
    return partial(EntryPoint, name, ())

#------------------------------------------------
#       CLASSES TO SUPPORT COMPOSITE PLOTS
#------------------------------------------------


class MultiplePlot(Plot):
    """ General handler of a group of plots that need to be rendered together

    Parameters
    ----------
    reading_order: None, optional
        Order in which the plot tries to read the data it needs.
    root_fdf: str, optional
        Path to the fdf file that is the 'parent' of the results.
    results_path: str, optional
        Directory where the files with the simulations results are
        located. This path has to be relative to the root fdf.
    """

    _trigger_kw = "varying"

    def __init__(self, *args, plots=None, template_plot=None, **kwargs):
        self.shared = {}

        # Take the plots if they have already been created and are provided by the user
        self.PLOTS_PROVIDED = plots is not None
        if self.PLOTS_PROVIDED:

            self.set_child_plots(plots)

        self.has_template_plot = False
        if isinstance(template_plot, Plot):
            self.template_plot = template_plot
            self.has_template_plot = True

        super().__init__(*args, **kwargs)

    def __getitem__(self, i):
        """ Gets a given child plot """
        return self.child_plots[i]

    @staticmethod
    def _kw_from_cls(cls):
        return cls._trigger_kw

    @staticmethod
    def _cls_from_kw(key):
        for cls in MultiplePlot.__subclasses__():
            if cls._trigger_kw == key:
                return cls
        else:
            return None

    @property
    def _attrs_for_child_plots(self):
        """ Returns all the attributes that its child_plots should have """
        return {
            'isChildPlot': True,
            'shared_attr': lambda key: self.shared_attr(key),
            'share_attr': lambda key, val: self.set_shared_attr(key, val)
        }

    def init_all_plots(self, update_fig=True, try_sharing=True):
        """ Initializes all child plots

        Parameters
        -----------
        update_fig: boolean, optional
            whether we should build the figure if this step is succesful.
        try_sharing: boolean, optional
            If `True`, we will check if all plots have exactly the same settings to read
            data and, in that case, they will share attributes to avoid memory waste.

            This is specially essential for plots that read big amounts of data (e.g. `GridPlot`),
            but also for those that take significant time to read it. 
        """
        if not self.PLOTS_PROVIDED:

            # If there is a template plot, take its settings as the starting point
            template_settings={}
            if self.has_template_plot:
                self._plot_classes = self.template_plot.__class__
                template_settings = self.template_plot.settings

            SINGLE_CLASS = isinstance(self._plot_classes, type)

            # Initialize all the plots
            # In case there is only one class only initialize them, avoid reading data
            # In this case, it is extremely important to initialize them all in serial mode, because
            # with multiprocessing they won't know that the current instance is their parent
            # (objects get copied in multiprocessing) and they won't be able to share data
            plots = init_multiple_plots(
                self._plot_classes,
                kwargsList = [
                    {**template_settings, **kwargs, "attrs_for_plot": self._attrs_for_child_plots, "only_init": SINGLE_CLASS and try_sharing}
                    for kwargs in self._getInitKwargsList()
                ],
                serial=SINGLE_CLASS and try_sharing
            )

            if SINGLE_CLASS and try_sharing:

                if not self.has_template_plot:
                    # Our leading plot will be the first one
                    leading_plot = plots[0]
                else:
                    leading_plot = self.template_plot

                # Now, we get the settings of the first plot
                read_data_settings = {
                    key: leading_plot.get_setting(key) for key, funcs in leading_plot._run_on_update.items()
                    if set(funcs).intersection(leading_plot.read_data_methods)
                }

                for i, plot in enumerate(plots):
                    if not plot.has_these_settings(read_data_settings):
                        # If there is a plot that needs to read different data, we will just
                        # make each of them read their own data. (this could be optimized by grouping plots)
                        self.init_all_plots(try_sharing=False)
                        break
                else:
                    # In case there is no plot that has different settings, we will
                    # happily set the data, avoiding the read data step. Plots will take
                    # their missing attributes from the shared store or from the plot
                    # template
                    self.set_child_plots(plots)

                    if not self.has_template_plot:
                        leading_plot._SHOULD_SHARE_WITH_SIBLINGS = True
                        leading_plot.read_data(update_fig=False)
                        leading_plot._SHOULD_SHARE_WITH_SIBLINGS = False

                    self.set_data()

            else:
                # If we haven't tried sharing data, the plots are already prepared (with read data of their own)
                self.set_child_plots(plots)

            call_method_if_present(self, "_after_childs_updated")

        if update_fig:
            self.get_figure()

        return self

    def update_child_settings(self, childs_sel=None, **kwargs):
        """ Updates the settings of all child plots

        Parameters
        -----------
        childs_sel: array-like of int, optional
            The indices of the child plots that you want to update.
        **kwargs
            Keyword arguments specifying the settings that you want to update
            and the values you want them to have
        """
        return self.update_settings(on_child_plots=True, on_parent_plot=False, childs_sel=childs_sel, **kwargs)

    def _update_settings(self, on_child_plots=False, on_parent_plot=True, childs_sel=None, **kwargs):
        """ This method takes into account that on plots that contain childs, one may want to update only the parent settings or all the child's settings.

        Parameters
        -----------
        on_child_plots: boolean, optional
            whether the settings should be updated on child plots
        on_parent_plot: boolean, optional
            whether the settings should be updated on the parent plot.
        childs_sel: array-like of int, optional
            The indices of the child plots that you want to update.
        """
        if on_parent_plot:
            super()._update_settings(**kwargs)

        if on_child_plots:

            repeat_if_childs(Configurable._update_settings)(self, childs_sel=childs_sel, **kwargs)

            call_method_if_present(self, "_after_childs_updated")

        return self

    def set_child_plots(self, plots, keep=False):
        """ Sets the child_plots of a multiple plot

        Parameters
        --------
        plots: array-like of sisl.viz.plotly.Plot or plotly Figure
            the plots that should be set as child_plots for the animation. 
        keep: boolean, optional
            whether the existing child_plots should be kept.

            If `True`, `plots` is added after them.
        """
        # Maybe one of the plots is a plotly figure, normalize all to the plot class
        plots = [Plot.from_plotly(plot) if isinstance(plot, go.Figure) else plot for plot in plots]

        for plot in plots:

            for key, val in self._attrs_for_child_plots.items():
                setattr(plot, key, val)

        self.child_plots = plots if not keep else [*self.child_plots, *plots]

        return self

    def add_child_plots(self, *plots):
        """ Append child_plots to the existing ones

        Parameters
        -----------
        *plots: Plot
            all the plots that you want to add as child plots of this one.
        """
        self.set_child_plots(plots, keep=True)

    def insert_childplot(self, index, plot):
        """ Inserts a plot in a given position of the child_plots list

        Parameters
        ----------
        index: int
            The position where the plot should be inserted
        plot: sisl Plot or plotly Figure
            The plot to insert in the list
        """
        if isinstance(plot, go.Figure):
            plot = Plot.from_plotly(plot)

        self.child_plots.insert(index, plot)

    def shared_attr(self, key):
        """ Gets an attribute that is located in the shared storage of the MultiplePlot

        This method will be given to all child_plots so that they can retreive the shared
        attributes. This is done in `set_child_plots`.

        Parameters
        ------------
        key: str
            the name of the attribute that you want to retrieve

        Returns
        -----------
        any
            the value that you asked for
        """
        # If from the beggining there is a template plot, the shared
        # storage is actually that plot.
        if self.has_template_plot:
            return getattr(self.template_plot, key)

        return self.shared[key]

    def set_shared_attr(self, key, val):
        """ Sets the value of a shared attribute

        Parameters
        ------------
        key: str
            the key of the attribute that is to be set.
        val: any
            the new value for the attribute
        """
        self.shared[key] = val

        return self

    def to_animation(self):
        """ Converts the multiple plot into an animation by splitting its plots into frames """
        self._isAnimation = True
        self.clear().get_figure()
        return self

    def get_figure(self):
        """ This method is responsible of building the figure from the child plots """
        self.clear()

        if getattr(self, "_isAnimation", False):
            frames_layout = Animation._build_frames(self)
            self.update_layout(**frames_layout)
        elif getattr(self, "_is_subplots", False):
            SubPlots._get_figure(self)

        else:
            data = []
            for plot in self.child_plots:
                data = [*data, *plot.data]

            self.data = data

        call_method_if_present(self, '_after_get_figure')

        call_method_if_present(self, 'on_figure_change')

        return self


class Animation(MultiplePlot):
    """ Version of MultiplePlot that renders each plot in a different animation frame

    Parameters
    ----------
    frame_duration: int, optional
        Time (in ms) that each frame will be displayed.  This is only
        meaningful if you have an animation
    redraw: bool, optional
        Whether each frame of the animation should be redrawn
        If False, the animation will try to interpolate between one frame and
        the other             Set this to False if you are sure that the
        frames contain the same number of traces, otherwise new traces will
        not appear.
    ani_method: None, optional
        It determines how the animation is rendered.
    reading_order: None, optional
        Order in which the plot tries to read the data it needs.
    root_fdf: str, optional
        Path to the fdf file that is the 'parent' of the results.
    results_path: str, optional
        Directory where the files with the simulations results are
        located. This path has to be relative to the root fdf.
    """

    _trigger_kw = "animate"

    _isAnimation = True

    _param_groups = (

        {
            "key": "animation",
            "name": "Animation specific settings",
            "icon": "videocam",
            "description": "The fact that you have not studied cinematography is not a good excuse for creating ugly animations. <b>Customize your animation with these settings</b>"
        },

    )

    _parameters = (

        IntegerInput(
            key = "frame_duration", name = "Frame duration",
            default = 500,
            group = "animation",
            params = {
                "step": 100
            },
            help = "Time (in ms) that each frame will be displayed. <br> This is only meaningful if you have an animation"
        ),

        SwitchInput(
            key='redraw', name='Redraw each frame',
            default=True,
            group='animation',
            help="""Whether each frame of the animation should be redrawn<br>
            If False, the animation will try to interpolate between one frame and the other<br>
            Set this to False if you are sure that the frames contain the same number of traces, otherwise new traces will not appear."""
        ),

        DropdownInput(
            key='ani_method', name="Animation method",
            default=None,
            group='animation',
            params={
                "placeholder": "Choose the animation method...",
                "options": [
                    {"label": "Update", "value": "update"},
                    {"label": "Animate", "value": "animate"},
                ],
                "isClearable": True,
                "isSearchable": True,
                "isMulti": False
            },
            help="""It determines how the animation is rendered. """
        )

    )

    def __init__(self, *args, frame_names=None, _plugins={}, **kwargs):
        if frame_names is not None:
            _plugins["_get_frame_names"] = frame_names if callable(frame_names) else lambda self, i: frame_names[i]

        super().__init__(*args, **kwargs, _plugins=_plugins)

    def _build_frames(self, ani_method):
        """ Builds the frames of the plotly figure from the child plots' data

        It actually sets the frames of the figure.

        Returns
        -----------
        dict
            keys and values that need to be added to the layout
            in order for frames to work.
        """
        # Get the names for each frame
        frame_names = []
        for i, plot in enumerate(self.child_plots):
            try:
                frame_name = self._get_frame_names(i)
            except Exception:
                frame_name = f"Frame {i+1}"
            frame_names.append(frame_name)

        if ani_method is None:
            same_traces = np.unique(
                [len(plot.data) for plot in self.child_plots]
            ).shape[0] == 1

            ani_method = "animate" if same_traces else "update"

        # Choose the method that we need to run in order to get the figure
        if ani_method == "animate":
            figure_builder = self._figure_animate_method
        elif ani_method == "update":
            figure_builder = self._figure_update_method

        steps, updatemenus = figure_builder(frame_names)

        frames_layout = {

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

            "updatemenus": updatemenus
        }

        return frames_layout

    def _figure_update_method(self, frame_names):
        """
        In the update method, we give all the traces to data, and we are just going to toggle
        their visibility depending on which 'frame' needs to be displayed.
        """
        # Add all the traces
        for i, (frame_name, plot) in enumerate(zip(frame_names, self.child_plots)):

            visible = i == 0

            self.add_traces([{
                **trace.to_plotly_json(),
                'customdata': [{'frame': frame_name, "iFrame": i}],
                'visible': visible
            } for trace in plot.data])

        # Generate the steps
        steps = []
        for i, frame_name in enumerate(frame_names):

            steps.append({
                "label": frame_name,
                "method": "restyle",
                "args": [{"visible": [trace.customdata[0]["iFrame"] == i for trace in self.data]}]
            })

        # WE SHOULD DEFINE PLAY AND PAUSE BUTTONS TO BE RENDERED IN JUPYTER'S NOTEBOOK HERE
        # IT IS IMPOSSIBLE TO PASS CONDITIONS TO DECIDE WHAT TO DISPLAY USING PLOTLY JSON
        self.animate_widgets = []

        return steps, []

    def _figure_animate_method(self, frame_names, frame_duration, redraw):
        """
        In the animate method, we explicitly define frames, And the transition from one to the other
        will be animated
        """
        # Data will actually only be the first frame
        self.data = self.child_plots[0].data

        frames = []

        maxN = np.max([len(plot.data) for plot in self.child_plots])
        for frame_name, plot in zip(frame_names, self.child_plots):

            data = plot.data
            nTraces = len(data)
            if nTraces < maxN:
                nAddTraces = maxN - nTraces
                data = [
                    *data, *np.full(nAddTraces, {"type": "scatter", "x":  [0], "y": [0], "visible": False})]

            frames = [
                *frames, {'name': frame_name, 'data': data, "layout": plot.get_settings_group("layout")}]

        self.frames = frames

        steps = [
            {"args": [
            [frame["name"]],
            {"frame": {"duration": int(frame_duration), "redraw": redraw},
            "mode": "immediate",
            "transition": {"duration": 300}}
        ],
            "label": frame["name"],
            "method": "animate"} for frame in self.frames
        ]

        updatemenus = [

            {'type': 'buttons',
            'buttons': [
                {
                    'label': '▶',
                    'method': 'animate',
                    'args': [None, {"frame": {"duration": int(frame_duration), "redraw": True},
                                    "fromcurrent": True, "transition": {"duration": 100,
                                                                        "easing": "quadratic-in-out"}}],
                },

                {
                    'label': '⏸',
                    'method': 'animate',
                    'args': [[None], {"frame": {"duration": 0}, "redraw": True,
                                    'mode': 'immediate',
                                    "transition": {"duration": 0}}],
                }
            ]}
        ]

        return steps, updatemenus

    def merge_frames(self):
        """ Merges all frames of an animation into one """
        self._isAnimation = False
        self.clear().get_figure()
        return self

    def zip(*animations):
        """ Zips multiple animations together

        This can work both as a static method or an object method:
        It can be called from the Animation class. However, if called from an Animation, 
        it will automatically include itself in the animations to zip.

        YOU NEED TO MAKE SURE THAT ALL THE ANIMATIONS HAVE THE SAME NUMBER OF FRAMES

        Parameters
        -----------
        *animations: sisl Animation
            the animations that you want to zip together.
            YOU NEED TO MAKE SURE THAT ALL THE ANIMATIONS HAVE THE SAME NUMBER OF FRAMES
        """
        frames = []
        for (*old_frames,) in zip(*animations):

            new_frame = MultiplePlot(plots=old_frames)

            frames.append(new_frame)

        return Animation(
            plots=frames
        )

    def unzip(self):
        """ Unzips the animation

        In order for this method to make sense, the animation needs to be previously zipped.
        This basically means that each frame needs to be a multiple plot and all frames are made
        of the same number of plots.
        """
        # Basically we just need to get the plots for each frame and then transpose it
        # so that we have the "frames for each plot"
        new_animations = np.array([frame.child_plots for frame in self]).T
        return [Animation(plots=plots) for plots in new_animations]


class SubPlots(MultiplePlot):
    """ Version of MultiplePlot that renders each plot in a separate subplot

    Parameters
    -----------
    arrange:  optional
        The way in which subplots should be aranged if the `rows` and/or
        `cols` parameters are not provided.
    rows: int, optional
        The number of rows of the plot grid. If not provided, it will be
        inferred from `cols` and the number of plots. If neither
        `cols` or `rows` are provided, the `arrange` parameter will decide
        how the layout should look like.
    cols: int, optional
        The number of columns of the subplot grid. If not provided, it will
        be inferred from `rows` and the number of plots. If
        neither `cols` or `rows` are provided, the `arrange` parameter will
        decide how the layout should look like.
    make_subplot_kwargs: dict, optional
        Extra keyword arguments that will be passed to make_subplots.
    reading_order:  optional
        Order in which the plot tries to read the data it needs.
    root_fdf: str, optional
        Path to the fdf file that is the 'parent' of the results.
    results_path: str, optional
        Directory where the files with the simulations results are
        located. This path has to be relative to the root fdf.
    """

    _trigger_kw = "subplots"

    _is_subplots = True

    _parameters = (

        DropdownInput(key='arrange', name='Automatic arrangement method',
            default='rows',
            params={
                'options': [
                    {'value': option, 'label': option} for option in ('rows', 'cols', 'square')
                ],
                'placeholder': 'Choose a subplot arrangement method...',
                'isMulti': False,
                'isSearchable': True,
                'isClearable': True,
            },
            help="""The way in which subplots should be aranged if the `rows` and/or `cols`
            parameters are not provided."""
        ),

        IntegerInput(key='rows', name='Rows',
            default=None,
            help="""The number of rows of the plot grid. If not provided, it will be inferred from `cols`
            and the number of plots. If neither `cols` or `rows` are provided, the `arrange` parameter will decide
            how the layout should look like."""
        ),

        IntegerInput(key='cols', name='Columns',
            default=None,
            help="""The number of columns of the subplot grid. If not provided, it will be inferred from `rows`
            and the number of plots. If neither `cols` or `rows` are provided, the `arrange` parameter will decide
            how the layout should look like."""
        ),

        ProgramaticInput(key='make_subplot_kwargs', name='make_subplot additional arguments',
            dtype=dict,
            default={},
            help="""Extra keyword arguments that will be passed to make_subplots."""
        )
    )

    def _get_figure(self, rows, cols, arrange, make_subplot_kwargs):
        """ Builds the subplots layout from the child plots' data """
        nplots = len(self.child_plots)
        if rows is None and cols is None:
            if arrange == 'rows':
                rows = nplots
                cols = 1
            elif arrange == 'cols':
                cols = nplots
                rows = 1
            elif arrange == 'square':
                cols = nplots ** 0.5
                rows = nplots ** 0.5
                # we will correct so it *fits*, always have more columns
                rows, cols = int(rows), int(cols)
                cols = nplots // rows + min(1, nplots % rows)
        elif rows is None:
            # ensure it is large enough by adding 1 if they don't add up
            rows = nplots // cols + min(1, nplots % cols)
        elif cols is None:
            # ensure it is large enough by adding 1 if they don't add up
            cols = nplots // rows + min(1, nplots % rows)

        rows, cols = int(rows), int(cols)

        if cols * rows < nplots:
            warn(f"requested {nplots} on a {rows}x{cols} grid layout. {nplots - cols*rows} plots will be missing.")

        # Check if all childplots have the same xaxis or yaxis titles.
        axes_titles = defaultdict(list)
        for child_plot in self.child_plots:
            axes_titles["x"].append(child_plot.layout.xaxis.title.text)
            axes_titles["y"].append(child_plot.layout.yaxis.title.text)

        # If so, we will set the subplots figure x_title and/or y_title so that it looks cleaner.
        # See how we remove the titles from the axis layout below when we allocate each plot.
        axes_titles = {f"{key}_title": val[0] for key, val in axes_titles.items() if len(set(val)) == 1}

        self.figure = make_subplots(**{
            "rows": rows, "cols": cols,
            **axes_titles,
            **make_subplot_kwargs
        })

        # Start assigning each plot to a position of the layout
        for (row, col), plot in zip(itertools.product(range(1, rows + 1), range(1, cols + 1)), self.child_plots):

            ntraces = len(plot.data)

            self.add_traces(plot.data, rows=[row]*ntraces, cols=[col]*ntraces)

            for ax in "x", "y":
                ax_layout = getattr(plot.layout, f"{ax}axis").to_plotly_json()

                # If we have set a global title for this axis, just remove it from the plot
                if axes_titles.get(f"{ax}_title"):
                    ax_layout["title"] = None

                update_axis = getattr(self, f"update_{ax}axes")

                update_axis(ax_layout, row=row, col=col)

        # Since we have directly copied the layouts of the child plots, there may be some references
        # between axes that we need to fix. E.g.: if yaxis was set to follow xaxis in the second child plot,
        # since the second child plot is put in (xaxes2, yaxes2) the reference will be now to the first child
        # plot xaxis, not itself. This is best understood by printing the figure of a subplot :)
        new_layouts = {}
        for ax, layout in self.figure.layout.to_plotly_json().items():
            if "axis" in ax:
                ax_name, ax_num = ax.split("axis")

                # Go over all possible problematic keys
                for key in ["anchor", "scaleanchor"]:
                    val = layout.get(key)
                    if val in ["x", "y"]:
                        layout[key] = f"{val}{ax_num}"

                new_layouts[ax] = layout

        self.update_layout(**new_layouts)
