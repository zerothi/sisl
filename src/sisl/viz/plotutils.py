# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

try:

    pathos_avail = True
except Exception:
    pathos_avail = False
try:

    tqdm_avail = True
except Exception:
    tqdm_avail = False

from sisl.io.sile import get_sile_rules, get_siles

from .types import Colorscale

__all__ = [
    "running_in_notebook",
    "check_widgets",
    "get_plot_classes",
    "get_plotable_siles",
    "get_plotable_variables",
    "get_avail_presets",
    "get_nested_key",
    "modify_nested_dict",
    "dictOfLists2listOfDicts",
    "get_avail_presets",
    "random_color",
    "find_files",
    "find_plotable_siles",
]

# -------------------------------------
#            Ipython
# -------------------------------------


def running_in_notebook():
    """Finds out whether the code is being run on a notebook.

    Returns
    --------
    bool
        whether the code is running in a notebook
    """
    try:
        return get_ipython().__class__.__name__ == "ZMQInteractiveShell"
    except NameError:
        return False


def check_widgets():
    """Checks if some jupyter notebook widgets are there.

    This will be helpful to know how the figures should be displayed.

    Returns
    -------
    dict
        contains whether widgets are available and if there was any error
        loading them.
    """
    import subprocess

    widgets = {
        "plotly_avail": False,
        "plotly_error": False,
        "events_avail": False,
        "events_error": False,
    }

    out, err = subprocess.Popen(
        ["jupyter", "nbextension", "list"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).communicate()
    out = str(out)
    err = str(err)

    if "plotlywidget" in out:
        widgets["plotly_avail"] = True
    if "plotlywidget" in err:
        widgets["plotly_error"] = True

    if "ipyevents" in out:
        try:

            widgets["events_avail"] = True
        except Exception:
            pass
    if "ipyevents" in err:
        widgets["events_error"] = True

    widgets["plotly"] = widgets["plotly_avail"] and not widgets["plotly_error"]
    widgets["events"] = widgets["events_avail"] and not widgets["events_error"]

    return widgets


# -------------------------------------
#            Informative
# -------------------------------------


def get_plot_classes():
    """This method returns all the plot subclasses, even the nested ones.

    Returns
    ---------
    list
        all the plot classes that the module is aware of.
    """
    from . import Plot

    def get_all_subclasses(cls):
        all_subclasses = []

        for Subclass in cls.__subclasses__():
            all_subclasses.append(Subclass)

            all_subclasses.extend(get_all_subclasses(Subclass))

        return all_subclasses

    return sorted(get_all_subclasses(Plot), key=lambda clss: clss.plot_name())


def get_plotable_siles(rules=False):
    """Gets the subset of siles that are plotable.

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
    """Retrieves all plotable variables that are in the global scope.

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


def get_avail_presets():
    """Gets the names of the currently available presets.

    Returns
    ---------
    list
        a list with all the presets names
    """
    from ._presets import PRESETS

    return list(PRESETS.keys())


# -------------------------------------
#           Python helpers
# -------------------------------------


def get_nested_key(obj, nestedKey, separator="."):
    """Gets a nested key from a dictionary using a given separator.

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
    """Use it to modify a nested dictionary with ease.

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
    """Converts a dictionary of lists to a list of dictionaries.

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


# -------------------------------------
#            Filesystem
# -------------------------------------


def find_files(
    root_dir=Path("."),
    search_string="*",
    depth=[0, 0],
    sort=True,
    sort_func=None,
    case_insensitive=False,
):
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
        search_string = "".join(
            [
                f"[{char.upper()}{char.lower()}]" if char.isalpha() else char
                for char in search_string
            ]
        )

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
    """Spans the filesystem to look for files that are registered as plotables.

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


# -------------------------------------
#            Colors
# -------------------------------------


def random_color():
    """Returns a random color in hex format

    Returns
    --------
    str
        the color in HEX format
    """
    import random

    return "#" + "%06x" % random.randint(0, 0xFFFFFF)


def values_to_colors(values, scale: Colorscale):
    """Maps an array of numbers to colors using a colorscale.

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

    from plotly.colors import sample_colorscale

    # Normalize values
    min_value = np.min(values)
    values = (np.array(values) - min_value) / (np.max(values) - min_value)

    return sample_colorscale(scale, values)
