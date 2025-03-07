# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

"""
This file provides tools to handle plotability of objects.

Registering `ClassA` as a plotable means that given an object of `ClassA`,
one can plot it like:

... code-block:: python

    object.plot()
    # or
    object.plot.some_plot_function()

In practice, what is registered is `(plotable_object, plot_function)` pairs.
When one of this pairs is registered:

   - A plot handler is attached to the object's class, if not already there.
   - The plotting function is attached to the plot handler.

The module has three main functions that should be used to register plotable objects. The
simplest of them is `register_plotable, which simply registers a object-function pair.
However, one tipically wants to merge a function that generates data with the function that
plots it. This module defines two functions that help with this by creating the merged functions
automatically, given that you provide the data function and the plot function:

    - `register_data_source`: Registers all the possible ways of getting a given data class,
        combining them with a plot class.
    - `register_sile_method`: Registers reading data from a sile using a certain method and then
        plotting it with a plot class.
"""
import inspect
from collections import ChainMap
from collections.abc import Sequence
from typing import Any, Callable, Optional

from sisl._dispatcher import AbstractDispatch, ClassDispatcher, ObjectDispatcher
from sisl._lib._docscrape import FunctionDoc
from sisl.io.sile import BaseSile
from sisl.viz.data import Data
from sisl.viz.plot import Plot

__all__ = ["register_plotable", "register_data_source", "register_sile_method"]

ALL_PLOT_HANDLERS = []

# --------------------------------------
#          Dispatcher classes
# --------------------------------------


class ClassPlotHandler(ClassDispatcher):
    """Handles all plotting possibilities for a class"""

    def __init__(self, cls, *args, inherited_handlers=(), **kwargs):
        self._cls = cls
        if not "instance_dispatcher" in kwargs:
            kwargs["instance_dispatcher"] = ObjectPlotHandler
        kwargs["type_dispatcher"] = None
        super().__init__(*args, inherited_handlers=inherited_handlers, **kwargs)

        ALL_PLOT_HANDLERS.append(self)

        self.__doc__ = f"Plotting functions for the `{cls.__name__}` class."

        self._dispatchs = ChainMap(
            self._dispatchs, *[handler._dispatchs for handler in inherited_handlers]
        )

    def set_default(self, key: str):
        """Sets the default plotting function for the class."""
        if key not in self._dispatchs:
            raise KeyError(f"Cannot set {key} as default since it is not registered.")
        self._default = key


class ObjectPlotHandler(ObjectDispatcher):
    """Handles all plotting possibilities for an object."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self._default is not None:
            default_call = getattr(self, self._default)

            self.__doc__ = default_call.__doc__
            self.__signature__ = default_call.__signature__

    def __call__(self, *args, **kwargs):
        """If the plot handler is called, we will run the default plotting function
        unless the keyword method has been passed."""
        if self._default is None:
            raise TypeError(
                f"No default plotting function has been defined for {self._obj.__class__.__name__}."
            )
        return getattr(self, self._default)(*args, **kwargs)


class PlotDispatch(AbstractDispatch):
    """Wraps a plotting function to be used in the dispatchers framework"""

    def dispatch(self, *args, **kwargs):
        """Runs the plotting function by passing the object instance to it."""
        return self._plot(self._obj, *args, **kwargs)


# --------------------------------------
#   Functions to register dispatchs
# --------------------------------------


def create_plot_dispatch(function: Callable, name: str, plot_cls=None):
    """From a function, creates a dispatch class that will be used by the dispatchers.

    By generating a different class for each function, we can have a different docstring
    for each of them. And this allows us to document each dispatch function properly.

    Parameters
    -----------
    function :
        function that will be executed when the dispatch is called. It will receive
        the object from the dispatch.
    name :
        the class name
    """
    return type(
        f"Plot{name.capitalize()}Dispatch",
        (PlotDispatch,),
        {
            "_plot": staticmethod(function),
            "__doc__": function.__doc__,
            "__signature__": inspect.signature(function),
            "_plot_class": plot_cls,
        },
    )


def _get_plotting_func(plot_cls: type[Plot], obj_input_key: str) -> callable:
    """Given a plot class and the key where the object should be passed, creates a plotting function.

    This is used in the `register_plotable` function to create a plotting function
    automatically from the plot class and the key where the object should be passed.

    It simply creates a function that accepts the object as first argument and then
    calls the plot class, passing the object to the appropiate input key.

    Parameters
    -----------
    plot_cls:
        the plot class that you want to use to plot the object.
    obj_input_key:
        the argument where the object should be passed.

    Returns
    -----------
    function
        a function that accepts the object as first argument and then generates the plot.
    """

    def _plot(obj, *args, **kwargs):
        return plot_cls(*args, **{obj_input_key: obj, **kwargs})

    fdoc = FunctionDoc(plot_cls)
    fdoc["Parameters"] = list(
        filter(lambda p: p.name.replace(":", "") != obj_input_key, fdoc["Parameters"])
    )
    docstring = str(fdoc)
    docstring = docstring[docstring.find("\n") :].lstrip()

    _plot.__doc__ = f"""Builds a ``{plot_cls.__name__}`` by setting the value of "{obj_input_key}" to the current object."""
    _plot.__doc__ += "\n\n" + docstring

    sig = inspect.signature(plot_cls)

    # The signature will be the same as the plot class, but without the input key, which
    # will be added by the _plot function
    _plot.__signature__ = sig.replace(
        parameters=[p for p in sig.parameters.values() if p.name != obj_input_key],
        return_annotation=plot_cls,
    )

    return _plot


def register_plotable(
    plotable,
    plot_cls: Optional[type[Plot]] = None,
    obj_input_key: Optional[str] = None,
    plotting_func: Optional[callable] = None,
    name: str = None,
    default: bool = False,
    plot_handler_attr: str = "plot",
    **kwargs,
):
    """Registers a pair of (plotable_class, plotting function).

    When one of this pairs is registered:

    * A plot handler is attached to the object's class, if not already there.
    * The plotting function is attached to the plot handler.

    Registering ``ClassA`` as a plotable means that given an object of ``ClassA``,
    one can plot it like:

    .. code-block:: python

       object.plot()
       # or
       object.plot.some_plot_function()

    Effectively, the plotting function becomes a method of the class so that when
    you call ``object.plot()``, the object is passed to the plotting function.

    Parameters
    ------------
    plotable:
        any class or object that you want to make plotable. Note that, if it's an object, the plotting
        capabilities will be attributed to all instances of the object's class.
    plot_cls:
        The class of the Plot that we want this object to use.
        If this is not provided, the `plotting_func` argument must be provided.
    obj_input_key:
        If the plotting function is generated from `plot_cls`, this is the key where
        the object will be passed.
    plotting_func:
        the function that takes care of the plotting. It should accept the object as
        first argument and then the rest of the arguments.

        If not provided, the plotting function is automatically generated from the
        `plot_cls` and `obj_input_key` arguments.
    name:
        name that will be used to identify the particular plot function that is being registered.

        E.g.: If name is "nicely", the plotting function will be registered under
        ``obj.plot.nicely()``

        If not provided:
        * If `plotting_func` is provided, the name of the function will be used.
        * If `plot_cls` is provided, ``plot_cls.plot_class_key()`` will be used, which
           by default removes the "Plot" suffix from the class name.
    default:
        whether this way of plotting the class should be the default one.
    plot_handler_attr:
        the attribute where the plot handler is or should be located in the plotable class.
    """

    # If no plotting function is provided, we will try to create one by using the plot_cls
    # and the obj_input_key that have been provided
    if plotting_func is None:
        plotting_func = _get_plotting_func(plot_cls, obj_input_key)

    if name is None and plot_cls is not None:
        # We will take the name of the plot class as the name
        name = plot_cls.plot_class_key()

    # Check if we already have a plot_handler
    plot_handler = getattr(plotable, plot_handler_attr, None)

    # If it's the first time that the class is being registered,
    # let's give the class a plot handler
    if (
        not isinstance(plot_handler, ClassPlotHandler)
        or plot_handler._cls is not plotable
    ):
        if isinstance(plot_handler, ClassPlotHandler):
            inherited_handlers = [plot_handler]
        else:
            inherited_handlers = []

        # If the user is passing an instance, we get the class
        if not isinstance(plotable, type):
            plotable = type(plotable)

        setattr(
            plotable,
            plot_handler_attr,
            ClassPlotHandler(
                plotable, plot_handler_attr, inherited_handlers=inherited_handlers
            ),
        )

        plot_handler = getattr(plotable, plot_handler_attr)

    plot_dispatch = create_plot_dispatch(plotting_func, name, plot_cls=plot_cls)
    # Register the function in the plot_handler
    plot_handler.register(name, plot_dispatch, default=default, **kwargs)


def _get_merged_parameters(
    doc1,
    doc2,
    excludedoc1: list = (),
    replacedoc1: dict = {},
    excludedoc2: list = (),
    replacedoc2: dict = {},
) -> str:
    """Merges the documentation of the parameters of two functions.

    Parameters
    ----------
    doc1:
        the documentation of the first function.
    doc2:
        the documentation of the second function.
    excludedoc1:
        the parameters of the first function that should not be included in the merged documentation.
    replacedoc1:
        a dictionary with the names of the parameters of the first function that should be replaced.
        Keys are the original names, values are the new names to use.
    excludedoc2:
        the parameters of the second function that should not be included in the merged documentation.
    replacedoc2:
        a dictionary with the names of the parameters of the second function that should be replaced.
        Keys are the original names, values are the new names to use.
    """

    def filter_and_replace(params, exclude, replace):
        filtered = list(
            filter(lambda p: p.name.replace(":", "") not in exclude, params)
        )

        replaced = []
        for p in filtered:
            name = p.name.removesuffix(":").removeprefix("**")
            if name in replace:
                p = p.__class__(name=replace[name], type=p.type, desc=p.desc)
            else:
                p = p.__class__(name=name, type=p.type, desc=p.desc)

            replaced.append(p)
        return replaced

    fdoc1 = FunctionDoc(doc1)

    fdoc2 = FunctionDoc(doc2)
    fdoc1["Parameters"] = [
        *filter_and_replace(fdoc1["Parameters"], excludedoc1, replacedoc1),
        *filter_and_replace(fdoc2["Parameters"], excludedoc2, replacedoc2),
    ]
    for k in fdoc1:
        if k == "Parameters":
            continue
        fdoc1[k] = fdoc1[k].__class__()

    docstring = str(fdoc1)
    docstring = docstring[docstring.find("\n") :].lstrip()

    return docstring


def get_merged_signature(
    func1: Callable,
    func2: Callable,
    func1_slice: slice = slice(None),
    func1_prefix: str = "_",
    remove_func2_inputs: list[str] = [],
    ret_annotation: Optional[Any] = None,
) -> tuple[inspect.Signature, dict]:
    """Creates a signature for the merging of two functions.

    This function resolves name clashes between the two functions by
    adding a prefix to the arguments of the first function if they have
    the same name as an argument of the second function.

    It makes the arguments of the second function keyword-only so that the
    signature shows an asterisk ``*`` between the arguments of the first and second
    functions, which visually helps to distinguish between the two.

    Since there can't be two ``**kwargs`` arguments, if function 1 contains a ``**kwargs``
    argument, it will be converted into an argument that accepts a dictionary.
    This dictionary should then be expanded when calling function 1. The ``**kwargs``
    argument of function 1 is also prefixed with `func1_prefix`.

    The function also returns a dictionary with useful information about the parameters
    that can be used to recreate the merged function, as done in `get_merged_function`.

    Parameters
    ----------
    func1 :
        the first function
    func2 :
        the second function
    func1_slice :
        the slice that will be used to get the arguments of the first function.
        E.g. `slice(1, None)` will get all arguments of the first function except the first one.
    func1_prefix :
        the prefix that will be added to the arguments of the first function
        when there is a name clash.
    remove_func2_inputs :
        the arguments of the second function that should not be included in the
        merged signature.
    ret_annotation :
        the return annotation of the merged function.

    Returns
    -------
    signature
        the signature of the merged function.
    params_info
        a dictionary with information about the parameters of the merged function.

    See also
    --------
    get_merged_function
        The function that builds the merged function out of the two functions.
        It needs the information generated by this function.
    """
    # Get the signatures of the functions
    signature1 = inspect.signature(func1)
    signature2 = inspect.signature(func2)

    # Get the parameters of the second function
    func2_params = {
        name: param.replace(kind=inspect.Parameter.KEYWORD_ONLY)
        for name, param in signature2.parameters.items()
        if name not in remove_func2_inputs
    }

    # Then go over the parameters of the first function to see
    # if they have to be modified

    # Initialize tracking variables
    merged_parameters = []
    replaced_func1_args = {}
    func1_var_kwarg = None

    # Loop through the parameters of the first function
    for param in list(signature1.parameters.values())[func1_slice]:
        # If the parameter is a **kwargs parameter, we have to convert it into an
        # argument that accepts a dictionary. This dictionary is to be expanded
        # when calling function 1.
        if param.kind == param.VAR_KEYWORD:
            func1_var_kwarg = param.name
            replaced_func1_args[f"{func1_prefix}{param.name}"] = param.name
            param = param.replace(
                name=f"{func1_prefix}{param.name}", kind=param.KEYWORD_ONLY, default={}
            )
        # If the name clashes, add prefix
        elif param.name in func2_params:
            replaced_func1_args[f"{func1_prefix}{param.name}"] = param.name
            param = param.replace(name=f"{func1_prefix}{param.name}")

        merged_parameters.append(param)

    # Store the new names of the arguments of the first function
    func1_args = [p.name for p in merged_parameters]

    # Add the arguments of the second function
    merged_parameters.extend(list(func2_params.values()))

    # Build the merged signature
    signature = inspect.Signature(
        parameters=merged_parameters, return_annotation=ret_annotation
    )

    # Store information about the parameters
    params_info = {
        "func1_args": func1_args,
        "replaced_func1_args": replaced_func1_args,
        "func1_var_kwarg": func1_var_kwarg,
        "func2_var_kwarg": (
            merged_parameters[-1].name
            if merged_parameters[-1].kind == inspect.Parameter.VAR_KEYWORD
            else None
        ),
    }

    return signature, params_info


def get_merged_function(
    data_func: Callable,
    data_defaults: dict,
    plot_cls: type[Plot],
    data_input_key: str,
    signature: inspect.Signature,
    params_info: dict,
) -> callable:
    """Builds a merged function out of two functions.

    It handles the splitting of the received arguments between those that
    go to the data function and those that go to the plot class.

    Parameters
    ----------
    data_func :
        the function that generates the data.
    data_defaults :
        the default values that should be passed to the data function. They
        will be overridden by the arguments passed when the merged function
        is called.
    plot_cls :
        the plot class that will be used to plot the data.
    data_input_key :
        the name of the plot class' argument where the data should be passed.
    signature :
        the signature of the merged function, this is built by the
        `get_merged_signature` function.
    params_info :
        information about the parameters of the merged function, this is built by
        the `get_merged_signature` function.

    Returns
    -------
    callable
        the merged function that will generate the data and then plot it.
    get_merged_signature
        The function that builds the merged signature and the dictionary with the
        information about the parameters. It is probably being called before
        this function.
    """

    # Copy the defaults so that we can override its arguments
    data_kwargs = data_defaults.copy()

    # Define the merged function
    def _plot(obj, *args, **kwargs):
        # Get the arguments that have been passed to the function
        bound = signature.bind_partial(**kwargs)

        # Determine which of those have to go to the data function
        try:
            # Loop through arguments of the data function
            for k in params_info["func1_args"]:
                # This argument has not been passed, skip it.
                if k not in bound.arguments:
                    continue

                # This argument has been passed.

                # The argument might have been renamed to avoid clashes,
                # translate back to the real name.
                data_key = params_info["replaced_func1_args"].get(k, k)

                # If it is the **kwargs argument expand it, else just add it.
                if params_info["func1_var_kwarg"] == data_key:
                    data_kwargs.update(bound.arguments.pop(k, {}))
                else:
                    data_kwargs[data_key] = bound.arguments.pop(k)
        except:
            raise TypeError(
                f"Error while parsing arguments for the merged function: {data_func.__name__} and {plot_cls.__name__}"
            )

        # Once we have all the arguments, get the data
        data = data_func(obj, *args, **data_kwargs)

        plot_kwargs = bound.arguments.pop(params_info["func2_var_kwarg"], {})

        # With the data, get the plot. Note that data arguments have been removed from bound.arguments
        return plot_cls(**{data_input_key: data, **bound.arguments, **plot_kwargs})

    _plot.__signature__ = signature

    return _plot


def register_data_source(
    data_source_cls: type[Data],
    plot_cls: type[Plot],
    data_input_key: str,
    name: Optional[str] = None,
    default: Sequence[type] = [],
    plot_handler_attr: str = "plot",
    data_source_defaults: dict = {},
    **kwargs,
):
    """Registers a data source as a plotable object.

    This function attaches a plotting method to the data source class.

    This function also goes through all the possible entry points in the data source
    (registered in the `new` class method) and appends a plotting method to each of them
    using `register_plotable`.

    The plotting function registered will be a merge of the data source and the plot class,
    like:

    .. code-block:: python

        def _plot(obj, ...):
            data = data_class.new(obj, ...)
            return plot_cls(**{data_input_key: data, ...})

    The resulting merged function will have a signature and docstring that is a merge of the
    data source and the plot class. See `get_merged_signature` and `get_merged_function` for more
    information on how this is done (for example, how name clashes are solved).

    Parameters
    ----------
    data_source_cls :
        the class to register as plotable.
    plot_cls :
        the plot class to be used to plot the data
    data_input_key :
        the name of the plot's argument where the data should be passed.
    name :
        the name that will be used to identify the particular plot function that is being registered.
        If not provided, the name of the plot class will be used, removing the "Plot" suffix.
    default :
        if there is an entry point for which the default plot should be this one,
        include in the list the class of the object that defines the entry point.

        E.g.: If the data source has an entry point that is triggered by calling `new` with
        `sisl.io.pdosSileSiesta` as a first argument, you can do
        ``default=[sisl.io.pdosSileSiesta]``
        to make the plot the default for `sisl.io.pdosSileSiesta`.
    plot_handler_attr :
        the attribute where the plot handler is or should be located in the class that you want to register.
    data_source_defaults :
        the default values that should be passed to the data source.

        NOTE: If an entry point does not support one of the keys in the defaults, it will not be registered.
    kwargs:
        passed directly to `register_plotable`

    See also
    --------
    register_plotable
        The method used to register the plotable object, once the merged (data+plot) function has been created.
    get_merged_signature, get_merged_function
        Helpers used to create the merged (data+plot) function.
    """

    # First register the data source itself
    register_plotable(
        data_source_cls,
        plot_cls=plot_cls,
        obj_input_key=data_input_key,
        name=name,
        plot_handler_attr=plot_handler_attr,
        **kwargs,
    )

    # And then all its entry points
    for plotable, cls_method in data_source_cls.new.dispatcher.registry.items():
        # Get this entry point's function
        func = cls_method.__get__(None, data_source_cls)

        # Get merged signature and function
        signature, params_info = get_merged_signature(
            func,
            plot_cls,
            func1_slice=slice(1, None),
            func1_prefix="data_",
            remove_func2_inputs=[data_input_key],
            ret_annotation=plot_cls,
        )
        _plot = get_merged_function(
            func, data_source_defaults, plot_cls, data_input_key, signature, params_info
        )

        # Check if the entry point supports the provided defaults.
        register_this = True
        for k in data_source_defaults.keys():
            if k not in signature.parameters:
                register_this = False
                break
        # If not, skip it.
        if not register_this:
            continue

        # Build the documentation
        doc = (
            # Short description
            f"Creates a ``{data_source_cls.__name__}`` object and then plots a ``{plot_cls.__name__}`` from it.\n\n"
            + "\n"
            # Parameters section
            + _get_merged_parameters(
                func,
                plot_cls,
                excludedoc1=(list(inspect.signature(func).parameters)[0],),
                replacedoc1={
                    v: k for k, v in params_info["replaced_func1_args"].items()
                },
                excludedoc2=(data_input_key,),
            )
            # See also section
            + "\n\nSee also\n--------\n"
            + plot_cls.__name__
            + "\n    The plot class used to generate the plot.\n"
            + data_source_cls.__name__
            + "\n    The class to which data is converted."
        )

        _plot.__doc__ = doc

        # Determine whether the plot should be the default for this entry point
        try:
            this_default = plotable in default
        except:
            this_default = False

        # Try to register the plotable object
        # It might not be possible (e.g. the object does not accept setting an attribute
        # so we can not add a plot handler to it)
        try:
            register_plotable(
                plotable,
                plot_cls=plot_cls,
                plotting_func=_plot,
                name=name,
                default=this_default,
                plot_handler_attr=plot_handler_attr,
                **kwargs,
            )
        except TypeError:
            pass


def register_sile_method(
    sile_cls: type[BaseSile],
    method: str,
    plot_cls: type[Plot],
    data_input_key: str,
    name: Optional[str] = None,
    default: bool = False,
    plot_handler_attr: str = "plot",
    **kwargs,
):
    """Registers a sile object as a plotable object.

    This function attaches a plotting method to the data source class.

    The plotting function registered will be a merge of the some previous step to
    read data from the sile object and the plot class, like (if `method="read_geometry"`):

    .. code-block:: python

        def _plot(obj, ...):
            data = sile_cls.read_geometry(obj, ...)
            return plot_cls(**{data_input_key: data, ...})

    The resulting merged function will have a signature and docstring that is a merge of the
    data source and the plot class. See `get_merged_signature` and `get_merged_function` for more
    information on how this is done (for example, how name clashes are solved).

    Parameters
    -----------
    sile_cls:
        the sile class to register as plotable.
    method:
        the method to use to read the data from the sile object, e.g. ``"read_geometry"``.
    plot_cls:
        the plot class to be used to plot the data.
    data_input_key:
        the name of the plot's argument where the data should be passed.
    name:
        the name that will be used to identify the particular plot function that is being registered.
        If not provided, the name of the plot class will be used, removing the "Plot" suffix.
    default:
        whether the plot being registered should be the default for this sile class.
    plot_handler_attr:
        the attribute where the plot handler is or should be located in the class that you want to register.
    kwargs:
        passed directly to `register_plotable`

    See also
    --------
    register_plotable
        The method used to register the plotable object, once the merged (data+plot) function has been created.
    get_merged_signature, get_merged_function
        Helpers used to create the merged (data+plot) function.
    """

    func = getattr(sile_cls, method)

    # Get merged signature and function
    signature, params_info = get_merged_signature(
        func,
        plot_cls,
        func1_slice=slice(1, None),
        func1_prefix="data_",
        remove_func2_inputs=[data_input_key],
        ret_annotation=plot_cls,
    )
    _plot = get_merged_function(
        func, {}, plot_cls, data_input_key, signature, params_info
    )

    # Build the documentation
    doc = (
        # Short description
        f"Calls ``{method}`` and creates a ``{plot_cls.__name__}`` from its output.\n\n"
        "\n"
        # Parameters section
        + _get_merged_parameters(
            func,
            plot_cls,
            excludedoc1=(list(inspect.signature(func).parameters)[0],),
            replacedoc1={v: k for k, v in params_info["replaced_func1_args"].items()},
            excludedoc2=(data_input_key,),
        )
        # See also section
        + "\n\nSee also\n--------\n"
        + plot_cls.__name__
        + "\n    The plot class used to generate the plot.\n"
        + method
        + "\n    The method called to get the data."
    )

    _plot.__doc__ = doc

    # Register the sile class as a plotable object
    register_plotable(
        sile_cls,
        plot_cls=plot_cls,
        plotting_func=_plot,
        name=name,
        default=default,
        plot_handler_attr=plot_handler_attr,
        **kwargs,
    )
