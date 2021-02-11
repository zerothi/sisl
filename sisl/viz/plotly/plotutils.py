import os
import glob
import dill
import sys
from pathlib import Path

import numpy as np
import itertools
from pathos.pools import ProcessPool as Pool
import tqdm

from copy import deepcopy

from sisl.messages import info
from sisl.io.sile import get_siles, get_sile_rules
from sisl._environ import register_environ_variable, get_environ_variable

__all__ = ["running_in_notebook", "check_widgets",
           "get_plot_classes", "get_plotable_siles", "get_plotable_variables",
           "get_session_classes", "get_avail_presets",
           "get_nested_key", "modify_nested_dict", "dictOfLists2listOfDicts",
           "get_avail_presets", "random_color",
           "load", "find_files", "find_plotable_siles",
           "shift_trace", "normalize_trace", "swap_trace_axes"
]

#-------------------------------------
#            Ipython
#-------------------------------------


def running_in_notebook():
    """ Finds out whether the code is being run on a notebook.

    Returns
    --------
    bool
        whether the code is running in a notebook
    """
    try:
        return get_ipython().__class__.__name__ == 'ZMQInteractiveShell'
    except NameError:
        return False


def check_widgets():
    """ Checks if some jupyter notebook widgets are there.

    This will be helpful to know how the figures should be displayed.

    Returns
    -------
    dict
        contains whether widgets are available and if there was any error
        loading them.
    """
    import subprocess

    widgets = {
        'plotly_avail': False,
        'plotly_error': False,
        'events_avail': False,
        'events_error': False
    }

    out, err = subprocess.Popen(['jupyter', 'nbextension', 'list'], stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()
    out = str(out)
    err = str(err)

    if 'plotlywidget' in out:
        widgets['plotly_avail'] = True
    if 'plotlywidget' in err:
        widgets['plotly_error'] = True

    if 'ipyevents' in out:
        try:
            import ipyevents
            widgets['events_avail'] = True
        except:
            pass
    if 'ipyevents' in err:
        widgets['events_error'] = True

    widgets['plotly'] = widgets['plotly_avail'] and not widgets['plotly_error']
    widgets['events'] = widgets['events_avail'] and not widgets['events_error']

    return widgets

#-------------------------------------
#            Informative
#-------------------------------------


def get_plot_classes():
    """ This method returns all the plot subclasses, even the nested ones.

    Returns
    ---------
    list
        all the plot classes that the module is aware of.
    """
    from . import Plot, MultiplePlot, Animation, SubPlots

    def get_all_subclasses(cls):

        all_subclasses = []

        for Subclass in cls.__subclasses__():

            if Subclass not in [MultiplePlot, Animation, SubPlots] and not getattr(Subclass, 'is_only_base', False):
                all_subclasses.append(Subclass)

            all_subclasses.extend(get_all_subclasses(Subclass))

        return all_subclasses

    return sorted(get_all_subclasses(Plot), key = lambda clss: clss.plot_name())


def get_plotable_siles(rules=False):
    """ Gets the subset of siles that are plotable.

    Returns
    ---------
    list
        all the siles that the module knows how to plot.
    """
    if rules:
        sile_getter = get_sile_rules
    else:
        sile_getter = get_siles

    return sile_getter(["plot"])


def get_plotable_variables(variables):
    """ Retrieves all plotable variables that are in the global scope.

    Examples
    -----------
    >>> get_plotable_variables(locals())
    >>> get_plotable_variables(globals())

    Parameters
    ----------
    variables: dict
        The variables dictionary of the namespace. Usually this will
        be retrieved by doing `locals()` or `globals()`

    Returns
    --------
    dict:
        A dict that contains the variable names and objects of the
        that are in the global variables scope and are plotables.
    """
    from types import ModuleType

    plotables = {}
    for vname, obj in list(variables.items()):

        if vname.startswith("_"):
            continue

        is_object = not isinstance(obj, (type, ModuleType))
        is_plotable = isinstance(getattr(obj, "_plot", None), tuple)
        if is_object and is_plotable:
            plotables[vname] = obj

    return plotables


def get_configurable_docstring(cls):
    """ Builds the docstring for a class that inherits from Configurable

    Parameters
    -----------
    cls:
        the class you want the docstring for

    Returns
    -----------
    str:
        the docs with the settings added.
    """
    import re

    if isinstance(cls, type):
        params = cls._parameters
        doc = cls.__doc__
        if doc is None:
            doc = ""
    else:
        # It's really an instance, not the class
        params = cls.params
        doc = ""

    configurable_settings = "\n".join([param._get_docstring() for param in params])

    html_cleaner = re.compile('<.*?>')
    configurable_settings = re.sub(html_cleaner, '', configurable_settings)

    if "Parameters\n--" not in doc:
        doc += f'\n\nParameters\n-----------\n{configurable_settings}'
    else:
        doc += f'\n{configurable_settings}'

    return doc


def get_configurable_kwargs(cls_or_inst, fake_default):
    """ Builds a string to help you define all the kwargs coming from the settings.

    The main point is to avoid wasting time writing all the kwargs manually, and
    at the same time makes it easy to keep it consistent with the defaults.

    This may be useful, for example, for the __init__ method of plots.

    Parameters
    ------------
    cls_or_inst:
        the class (or instance) you want the kwargs for.
    fake_default: str
        only floats, ints, bools and strings can be parsed safely into strings and then into values again.
        For this reason, the rest of the settings will just be given a fake default that you need to handle.

    Returns
    -----------
    str:
        the string containing the described kwargs.
    """
    # TODO why not just repr(val)? that seems to be the same in all cases?
    def get_string(val):
        if isinstance(val, (float, int, bool)) or val is None:
            return val
        elif isinstance(val, str):
            return val.__repr__()
        else:
            return fake_default.__repr__()

    if isinstance(cls_or_inst, type):
        params = cls_or_inst._parameters
        return ", ".join([f'{param.key}={get_string(param.default)}' for param in params])

    # It's really an instance, not the class
    # In this case, the defaults for the method will be the current values.
    params = cls_or_inst.params
    return ", ".join([f'{param.key}={get_string(cls_or_inst.settings[param.key])}' for param in params])


def get_configurable_kwargs_to_pass(cls):
    """ Builds a string to help you pass kwargs that you got from the function `get_configurable_kwargs`.

    E.g.: If `get_configurable_kwargs` gives you 'param1=None, param2="nothing"'
    `get_configurable_kwargs_to_pass` will give you param1=param1, param2=param2

    Parameters
    ------------
    cls:
        the class you want the kwargs for

    Returns
    -----------
    str:
        the string containing the described kwargs.
    """
    if isinstance(cls, type):
        params = cls._parameters
    else:
        # It's really an instance, not the class
        params = cls.params

    return ", ".join([f'{param.key}={param.key}' for param in params])


def get_session_classes():
    """ Returns the available session classes

    Returns
    --------
    dict
        keys are the name of the class and values are the class itself.
    """
    from .session import Session

    return {sbcls.__name__: sbcls for sbcls in Session.__subclasses__()}


def get_avail_presets():
    """ Gets the names of the currently available presets.

    Returns
    ---------
    list
        a list with all the presets names
    """
    from ._presets import PRESETS

    return list(PRESETS.keys())

#-------------------------------------
#           Python helpers
#-------------------------------------


def get_nested_key(obj, nestedKey, separator="."):
    """ Gets a nested key from a dictionary using a given separator.

    Parameters
    --------
    obj: dict
        The dictionary to search.
    nestedKey: str
        The key to get. See the separator argument for how it should look like.

        The function will work too if this is a simple key, without any nesting
    separator: str, optional (".")
        It defines how hierarchy is indicated in the provided key.

        For example:
            if separator is "." and nestedKey is "xaxis.length"

            {
                "xaxis: {
                    "aKey": whatever,
                    "anotherKey": whatever,
                    "length": ---This is the value that will be retrieved---
                },
                "moreKeys": whatever,
                "notRelevant": whatever
            }
    """

    ref = obj
    splitted = nestedKey.split(separator)
    for key in splitted[:-1]:
        ref = ref[key]

    return ref[splitted[-1]]


def modify_nested_dict(obj, nestedKey, val, separator="."):
    """ Use it to modify a nested dictionary with ease. 

    It modifies the dictionary itself, does not return anything.

    Arguments
    ----------
    obj: dict
        The dictionary to modify.
    nestedKey: str
        The key to modify. See the separator argument for how it should look like.

        The function will work too if this is a simple key, without any nesting
    val: 
        The new value to give to the target key.
    separator: str, optional (".")
        It defines how hierarchy is indicated in the provided key.

        For example:
            if separator is "." and nestedKey is "xaxis.length"

            {
                "xaxis: {
                    "aKey": whatever,
                    "anotherKey": whatever,
                    "length": ---This is the value that will be modified---
                },
                "moreKeys": whatever,
                "notRelevant": whatever
            }
    """

    ref = obj
    splitted = nestedKey.split(separator)
    for key in splitted[:-1]:
        ref = ref[key]

    ref[splitted[-1]] = val


def dictOfLists2listOfDicts(dictOfLists):
    """ Converts a dictionary of lists to a list of dictionaries.

    The example will make it quite clear.

    Examples
    ---------
    >>> list_of_dicts = dictOfLists2listOfDicts({"a": [0,1,2], "b": [3,4,5]})
    >>> assert list_of_dicts == [{"a": 0, "b": 3}, {"a": 1, "b": 4}, {"a": 2, "b": 5}]

    Parameters
    ---------
    dictOfLists: dict of array-like
        The dictionary of lists that you want to convert

    Returns
    ---------
    list of dicts:
        A list with the individual dicts generated by the function.
    """

    return [dict(zip(dictOfLists, t)) for t in zip(*dictOfLists.values())]


def call_method_if_present(obj, method_name, *args, **kwargs):
    """ Calls a method of the object if it is present.

    If the method is not there, it just does nothing.

    Parameters
    -----------
    method_name: str
        the name of the method that you want to call.
    *args and **kwargs:
        arguments passed to the method call.
    """

    method = getattr(obj, method_name, None)
    if callable(method):
        return method(*args, **kwargs)


def copy_params(params, only=(), exclude=()):
    """ Function that returns a copy of the provided plot parameters.

    Arguments
    ----------
    params: tuple
        The parameters that have to be copied. This will come presumably from the "_parameters" variable of some plot class.
    only: array-like
        Use this if you only want a certain set of parameters. Pass the wanted keys as a list.
    exclude: array-like
        Use this if there are some parameters that you don't want. Pass the unwanted keys as a list.
        This argument will not be used if "only" is present.

    Returns
    ----------
    copiedParams: tuple
        The params that the user asked for. They are not linked to the input params, so they can be modified independently.
    """
    if only:
        return tuple(param for param in deepcopy(params) if param.key in only)
    return tuple(param for param in deepcopy(params) if param.key not in exclude)


def copy_dict(dictInst, only=(), exclude=()):
    """ Function that returns a copy of a dict. This function is thought to be used for the settings dictionary, for example.

    Arguments
    ----------
    dictInst: dict
        The dictionary that needs to be copied.
    only: array-like
        Use this if you only want a certain set of values. Pass the wanted keys as a list.
    exclude: array-like
        Use this if there are some values that you don't want. Pass the unwanted keys as a list.
        This argument will not be used if "only" is present.

    Returns
    ----------
    copiedDict: dict
        The dictionary that the user asked for. It is not linked to the input dict, so it can be modified independently.
    """
    if only:
        return {k: v for k, v  in deepcopy(dictInst).iteritems() if k in only}
    return {k: v for k, v  in deepcopy(dictInst).iteritems() if k not in exclude}

#-------------------------------------
#            Filesystem
#-------------------------------------


# TODO load seems extremely generic. Could we have another name?
#      consider users doing from dill import *, and same here?
def load(path):
    """
    Loads a previously saved python object using pickle. To be used for plots, sessions, etc...

    Arguments
    ----------
    path: str
        The path to the saved object.

    Returns
    ----------
    loadedObj: object
        The object that was saved.
    """

    with open(path, 'rb') as handle:
        loadedObj = dill.load(handle)

    return loadedObj


def find_files(root_dir=Path("."), search_string = "*", depth = [0, 0], sort = True, sort_func = None, case_insensitive=False):
    """
    Function that finds files (or directories) according to some conditions.

    Arguments
    -----------
    root_dir: str or Path, optional
        Path of the directory from which the search will start.
    search_string: str, optional
        This is the string that will be passed to glob.glob() to find files or directories. 
        It works mostly like bash, so you can use wildcards, for example.
    depth: array-like of length 2 or int, optional
        If it is an array:

            It will specify the limits of the search. 
            For example, depth = [1,3] will make the function search for the search_string from 1 to 3 directories deep from root_dir.
            (0 depth means to look for files in the root_dir)

        If it is an int:
            Only that depth level will be searched.
            That is, depth = 1 is the same as depth = [1,1].
    sort: boolean, optional
        Whether the returned list of paths should be sorted.
    sort_func: function, optional
        The function that has to be used for sorting the paths. Only meaningful if sort is True.
    case_insensitive: boolean, optional
        whether the search should be case insensitive

    Returns
    -----------
    list
        A list with all the paths found for the given conditions and sorted according to the provided arguments.
    """
    # Normalize the depth parameter
    if isinstance(depth, int):
        depth = [depth, depth]

    # Normalize the root path to a pathlib path
    if isinstance(root_dir, str):
        root_dir = Path(root_dir)

    if case_insensitive:
        search_string = "".join([f"[{char.upper()}{char.lower()}]" if char.isalpha() else char for char in search_string])

    files = []
    for depth in range(depth[0], depth[1] + 1):
        # Path.glob returns a generator
        new_files = root_dir.glob(f"*{os.path.sep}" * depth + search_string)

        # And we just iterate over all the found paths (if any)
        files.extend([path.resolve() for path in new_files])

    if sort:
        return sorted(files, key=sort_func)
    return files


def find_plotable_siles(dir_path=None, depth=0):
    """ Spans the filesystem to look for files that are registered as plotables.

    Parameters
    -----------
    dir_path: str, optional
        the directory where to look for the files.
        If not provided, the current working directory will be used.
    depth: int or array-like of length 2, optional
        how deep into directories we should go to look for files.

        If it is an array:

            It will specify the limits of the search. 
            For example, depth = [1,3] will make the function search for the searchString from 1 to 3 directories deep from root_dir.
            (0 depth means to look for files in the root_dir)

        If it is an int:
            Only that depth level will be searched.
            That is, depth = 1 is the same as depth = [1,1].

    Returns
    -----------
    dict
        A dict containing all the files found sorted by sile (the keys are the siles)
    """

    files = {}
    for rule in get_plotable_siles(rules=True):
        search_string = f"*.{rule.suffix}"

        sile_files = find_files(dir_path, search_string, depth, case_insensitive=True)

        if sile_files:
            files[rule.cls] = sile_files

    return files

#-------------------------------------
#         Multiprocessing
#-------------------------------------

register_environ_variable("SISL_NPROCS_VIZ", max(os.cpu_count() - 1, 1),
                          description="Maximum number of processors used for parallel plotting",
                          process=int)
_MAX_NPROCS = get_environ_variable("SISL_NPROCS_VIZ")


def _apply_method(args_tuple):
    """ Apply a method to an object. This function is meant for multiprocessing """

    method, obj, args, kwargs = args_tuple

    if args is None:
        args = []

    method(obj, *args, **kwargs)

    return obj._get_pickleable()


def _init_single_plot(args_tuple):
    """ Initialize a single plot. This function is meant to be used in multiprocessing, when multiple plots need to be initialized """

    PlotClass, args, kwargs = args_tuple

    return PlotClass(**kwargs)._get_pickleable()


def run_multiple(func, *args, argsList = None, kwargsList = None, messageFn = None, serial = False):
    """
    Makes use of the pathos.multiprocessing module to run a function simultanously multiple times.
    This is meant mainly to update multiple plots at the same time, which can accelerate significantly the process of visualizing data.

    All arguments passed to the function, except func, can be passed as specified in the arguments section of this documentation
    or as a list containing multiple instances of them.
    If a list is passed, each time the function needs to be run it will take the next item of the list.
    If a single item is passed instead, this item will be repeated for each function run.
    However, at least one argument must be a list, so that the number of times that the function has to be ran is defined.

    Arguments
    ----------
    func: function
        The function to be executed. It has to be prepared to recieve the arguments as they are provided to it (zipped).

        See the applyMethod() function as an example.
    *args:
        Contains all the arguments that are specific to the individual function that we want to run.
        See each function separately to understand what you need to pass (you may not need this parameter).
    argsList: array-like
        An array of arguments that have to be passed to the executed function.

        Can also be a list of arrays (see this function's description).

        WARNING: Currently it only works properly for a list of arrays. Didn't fix this because the lack of interest
        of argsList on Plot's methods (everything is passed as keyword arguments).
    kwargsList: dict
        A dictionary with the keyword arguments that have to be passed to the executed function.

        If the executed function is a Plot's method, these can be the settings, for example.

        Can also be a list of dicts (see this function's description).

    messageFn: function
        Function that recieves the number of tasks and nodes and needs to return a string to display as a description of the progress bar.
    serial: bool
        If set to true, multiprocessing is not used.

        This seems to have little sense, but it is useful to switch easily between multiprocessing and serial with the same code.

    Returns
    ----------
    results: list
        A list with all the returned values or objects from each function execution.
        This list is ordered, so results[0] is the result of executing the function with argsList[0] and kwargsList[0].  
    """

    #Prepare the arguments to be passed to the initSinglePlot function
    toZip = [*args, argsList, kwargsList]
    for i, arg in enumerate(toZip):
        if not isinstance(arg, (list, tuple, np.ndarray)):
            toZip[i] = itertools.repeat(arg)
        else:
            nTasks = len(arg)

    # Run things in serial mode in case it is demanded
    serial = serial or _MAX_NPROCS == 1 or nTasks == 1
    if serial:
        return [func(argsTuple) for argsTuple in zip(*toZip)]

    #Create a pool with the appropiate number of processes
    pool = Pool(min(nTasks, _MAX_NPROCS))
    #Define the plots array to store all the plots that we initialize
    results = [None]*nTasks

    #Initialize the pool iterator and the progress bar that controls it
    progress = tqdm.tqdm(pool.imap(func, zip(*toZip)), total = nTasks)

    #Set a description for the progress bar
    if not callable(messageFn):
        message = "Updating {} plots in {} processes".format(nTasks, pool.nodes)
    else:
        message = messageFn(nTasks, pool.nodes)

    progress.set_description(message)

    #Run the processes and store each result in the plots array
    for i, res in enumerate(progress):
        results[i] = res

    pool.close()
    pool.join()
    pool.clear()

    return results


def init_multiple_plots(PlotClass, argsList = None, kwargsList = None, **kwargs):
    """ Initializes a set of plots in multiple processes simultanously making use of runMultiple()

    All arguments passed to the function, can be passed as specified in the arguments section of this documentation
    or as a list containing multiple instances of them.
    If a list is passed, each time the function needs to be run it will take the next item of the list.
    If a single item is passed instead, this item will be repeated for each function run.
    However, at least one argument must be a list, so that the number of times that the function has to be ran is defined.

    Arguments
    ----------
    PlotClass: child class of sisl.viz.plotly.Plot
        The plot class that must be initialized

        Can also be a list of classes (see this function's description).
    argsList: array-like
        An array of arguments that have to be passed to the executed function.

        Can also be a list of arrays (see this function's description).

        WARNING: Currently it only works properly for a list of arrays. Didn't fix this because the lack of interest
        of argsList on Plot's methods (everything is passed as keyword arguments).
    kwargsList: dict
        A dictionary with the keyword arguments that have to be passed to the executed function.

        If the executed function is a Plot's method, these can be the settings, for example.

        Can also be a list of dicts (see this function's description).

    Returns
    ----------
    plots: list
        A list with all the initialized plots.
        This list is ordered, so plots[0] is the plot initialized with argsList[0] and kwargsList[0].  
    """

    return run_multiple(_init_single_plot, PlotClass, argsList = argsList, kwargsList = kwargsList, **kwargs)


def apply_method_on_multiple_objs(method, objs, argsList = None, kwargsList = None, **kwargs):
    """ Applies a given method to the objects provided on multiple processes simultanously making use of the runMultiple() function.

    This is useful in principle for any kind of object and any method, but has been tested only on plots.

    All arguments passed to the function, except method, can be passed as specified in the arguments section of this documentation
    or as a list containing multiple instances of them.
    If a list is passed, each time the function needs to be run it will take the next item of the list.
    If a single item is passed instead, this item will be repeated for each function run.
    However, at least one argument must be a list, so that the number of times that the function has to be ran is defined.

    Arguments
    ----------
    method: func
        The method to be executed.
    objs: object
        The object to which we need to apply the method (e.g. a plot)

        Can also be a list of objects (see this function's description).
    argsList: array-like
        An array of arguments that have to be passed to the executed function.

        Can also be a list of arrays (see this function's description).

        WARNING: Currently it only works properly for a list of arrays. Didn't fix this because the lack of interest
        of argsList on Plot's methods (everything is passed as keyword arguments).
    kwargsList: dict
        A dictionary with the keyword arguments that have to be passed to the executed function.

        If the executed function is a Plot's method, these can be the settings, for example.

        Can also be a list of dicts (see this function's description).

    Returns
    ----------
    plots: list
        A list with all the initialized plots.
        This list is ordered, so plots[0] is the plot initialized with argsList[0] and kwargsList[0].  
    """

    return run_multiple(_apply_method, method, objs, argsList = argsList, kwargsList = kwargsList, **kwargs)


def repeat_if_childs(method):
    """ Decorator that will force a method to be run on all the plot's child_plots in case there are any """

    def apply_to_all_plots(obj, *args, childs_sel=None, **kwargs):

        if hasattr(obj, "child_plots"):

            kwargs_list = kwargs.get("kwargs_list", kwargs)

            # Get all the child plots that we are going to modify
            childs = obj.child_plots
            if childs_sel is not None:
                childs = np.array(childs)[childs_sel].tolist()
            else:
                childs_sel = range(len(childs))

            new_childs = apply_method_on_multiple_objs(method, childs, kwargsList=kwargs_list, serial=True)

            # Set the new plots. We need to do this because apply_method_on_multiple_objs
            # can use multiprocessing, and therefore will not modify the plot in place.
            for i, new_child in zip(childs_sel, new_childs):
                obj.child_plots[i] = new_child

            obj.get_figure()

        else:

            return method(obj, *args, **kwargs)

    return apply_to_all_plots

#-------------------------------------
#             Fun stuff
#-------------------------------------

# TODO these would be ideal to put in the sisl configdir so users can
# alter the commands used ;)
# However, not really needed now.


def trigger_notification(title, message, sound="Submarine"):
    """ Triggers a notification.

    Will not do anything in Windows (oops!)

    Parameters
    -----------
    title: str
    message: str
    sound: str
    """

    if sys.platform == 'linux':
        os.system(f"""notify-send "{title}" "{message}" """)
    elif sys.platform == 'darwin':
        sound_string = f'sound name "{sound}"' if sound else ''
        os.system(f"""osascript -e 'display notification "{message}" with title "{title}" {sound_string}' """)
    else:
        info(f"sisl cannot issue notifications through the operating system ({sys.platform})")


def spoken_message(message):
    """ Trigger a spoken message.

    In linux espeak must be installed (sudo apt-get install espeak)

    Will not do anything in Windows (oops!)

    Parameters
    -----------
    title: str
    message: str
    sound: str
    """

    if sys.platform == 'linux':
        os.system(f"""espeak -s 150 "{message}" 2>/dev/null""")
    elif sys.platform == 'darwin':
        os.system(f"""osascript -e 'say "{message}"' """)
    else:
        info(f"sisl cannot issue notifications through the operating system ({sys.platform})")

#-------------------------------------
#        Plot manipulation
#-------------------------------------


def shift_trace(trace, shift, axis="y"):
    """ Shifts a trace by a given value in the given axis.

    Parameters
    -----------
    shift: float or array-like
        If it's a float, it will be a solid shift (i.e. all points moved equally).
        If it's an array, an element-wise sum will be performed
    axis: {"x","y","z"}, optional
        The axis along which we want to shift the traces.
    """
    trace[axis] = np.array(trace[axis]) + shift


def normalize_trace(trace, min_val=0, max_val=1, axis='y'):
    """ Normalizes a trace to a given range along an axis.

    Parameters
    -----------
    min_val: float, optional
        The lower bound of the range.
    max_val: float, optional
        The upper part of the range
    axis: {"x", "y", "z"}, optional
        The axis along which we want to normalize.
    """
    t = np.array(trace[axis])
    tmin = t.min()
    trace[axis] = (t - tmin) / (t.max() - tmin) * (max_val - min_val) + min_val


def swap_trace_axes(trace, ax1='x', ax2='y'):
    """ Swaps two axes of a trace.

    Parameters
    -----------
    ax1, ax2: str, {'x', 'x*', 'y', 'y*', 'z', 'z*'}
        The names of the axes that you want to swap. 
    """
    ax1_data = trace[ax1]
    trace[ax1] = trace[ax2]
    trace[ax2] = ax1_data


#-------------------------------------
#            Colors
#-------------------------------------

def random_color():
    """ Returns a random color in hex format

    Returns
    --------
    str
        the color in HEX format
    """
    import random
    return "#"+"%06x" % random.randint(0, 0xFFFFFF)


def values_to_colors(values, scale):
    """ Maps an array of numbers to colors using a colorscale.

    Parameters
    -----------
    values: array-like of float or int
        the values to map to colors.
    scale: str or list
        the color scale to use for the mapping.

        If it's a string, it is interpreted as a plotly scale (the supported names are
        the same accepted by the "colorscale" key in plotly)

        Otherwise, it must be a list of colors.

    Returns
    -----------
    list
        the corresponding colors in "rgb(r,g,b)" format.
    """
    import plotly
    import matplotlib

    v_min = np.min(values)
    values = (values - v_min) / (np.max(values) - v_min)

    scale_colors = plotly.colors.convert_colors_to_same_type(scale, colortype="tuple")[0]

    if not scale_colors and isinstance(scale, str):
        scale_colors = plotly.colors.convert_colors_to_same_type(scale[0].upper() + scale[1:], colortype="tuple")[0]

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("my color map", scale_colors)

    return plotly.colors.convert_colors_to_same_type([cmap(c) for c in values])[0]
