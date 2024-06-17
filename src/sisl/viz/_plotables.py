# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from __future__ import annotations

"""
This file provides tools to handle plotability of objects
"""
import inspect
from collections import ChainMap
from typing import Sequence, Type

from sisl._dispatcher import AbstractDispatch, ClassDispatcher, ObjectDispatcher

__all__ = ["register_plotable", "register_data_source", "register_sile_method"]


class ClassPlotHandler(ClassDispatcher):
    """Handles all plotting possibilities for a class"""

    def __init__(self, cls, *args, inherited_handlers=(), **kwargs):
        self._cls = cls
        if not "instance_dispatcher" in kwargs:
            kwargs["instance_dispatcher"] = ObjectPlotHandler
        kwargs["type_dispatcher"] = None
        super().__init__(*args, inherited_handlers=inherited_handlers, **kwargs)

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


def create_plot_dispatch(function, name):
    """From a function, creates a dispatch class that will be used by the dispatchers.

    Parameters
    -----------
    function: function
        function that will be executed when the dispatch is called. It will receive
        the object from the dispatch.
    name: str
        the class name
    """
    return type(
        f"Plot{name.capitalize()}Dispatch",
        (PlotDispatch,),
        {
            "_plot": staticmethod(function),
            "__doc__": function.__doc__,
            "__signature__": inspect.signature(function),
        },
    )


def _get_plotting_func(plot_cls, setting_key):
    """Generates a plotting function for an object.

    Parameters
    -----------
    plot_cls: subclass of Plot
        the plot class that you want to use to plot the object.
    setting_key: str
        the setting where the plotable should go

    Returns
    -----------
    function
        a function that accepts the object as first argument and then generates the plot.

        It sends the object to the appropiate setting key. The rest works exactly the same as
        calling the plot class. I.e. you can provide all the extra settings/keywords that you want.
    """

    def _plot(obj, *args, **kwargs):
        return plot_cls(*args, **{setting_key: obj, **kwargs})

    _plot.__doc__ = f"""Builds a {plot_cls.__name__} by setting the value of "{setting_key}" to the current object.

    Documentation for {plot_cls.__name__}
    ===========
    {inspect.cleandoc(plot_cls.__doc__) if plot_cls.__doc__ is not None else None}
    """

    sig = inspect.signature(plot_cls)

    # The signature will be the same as the plot class, but without the setting key, which
    # will be added by the _plot function
    _plot.__signature__ = sig.replace(
        parameters=[p for p in sig.parameters.values() if p.name != setting_key]
    )

    return _plot


def register_plotable(
    plotable,
    plot_cls=None,
    setting_key=None,
    plotting_func=None,
    name=None,
    default=False,
    plot_handler_attr="plot",
    **kwargs,
):
    """
    Makes the sisl.viz module aware of which sisl objects can be plotted and how to do it.

    The implementation uses plot handlers. The only thing that this function does is to check
    if there is a plot handler, and if not, create it. The rest is handled by the plot handler.

    Parameters
    ------------
    plotable: any
        any class or object that you want to make plotable. Note that, if it's an object, the plotting
        capabilities will be attributed to all instances of the object's class.
    plot_cls: child of sisl.Plot, optional
        The class of the Plot that we want this object to use.
    setting_key: str, optional
        The key of the setting where the object must go. This works together with
        the plot_cls parameter.
    plotting_func: function
        the function that takes care of the plotting.
        It should accept (self, *args, **kwargs) and return a plot object.
    name: str, optional
        name that will be used to identify the particular plot function that is being registered.

        E.g.: If name is "nicely", the plotting function will be registered under "obj.plot.nicely()"

        If not provided, the name of the function will be used
    default: boolean, optional
        whether this way of plotting the class should be the default one.
    plot_handler_attr: str, optional
        the attribute where the plot handler is or should be located in the class that you want to register.
    """

    # If no plotting function is provided, we will try to create one by using the plot_cls
    # and the setting_key that have been provided
    if plotting_func is None:
        plotting_func = _get_plotting_func(plot_cls, setting_key)

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

    plot_dispatch = create_plot_dispatch(plotting_func, name)
    # Register the function in the plot_handler
    plot_handler.register(name, plot_dispatch, default=default, **kwargs)


def register_data_source(
    data_source_cls,
    plot_cls,
    setting_key,
    name=None,
    default: Sequence[Type] = [],
    plot_handler_attr="plot",
    data_source_init_kwargs: dict = {},
    **kwargs,
):
    # First register the data source itself
    register_plotable(
        data_source_cls,
        plot_cls=plot_cls,
        setting_key=setting_key,
        name=name,
        plot_handler_attr=plot_handler_attr,
        **kwargs,
    )

    # And then all its entry points
    plot_cls_params = {
        name: param.replace(kind=inspect.Parameter.KEYWORD_ONLY)
        for name, param in inspect.signature(plot_cls).parameters.items()
        if name != setting_key
    }

    for plotable, cls_method in data_source_cls.new.dispatcher.registry.items():
        func = cls_method.__get__(None, data_source_cls)

        signature = inspect.signature(func)

        register_this = True
        for k in data_source_init_kwargs.keys():
            if k not in signature.parameters:
                register_this = False
                break

        if not register_this:
            continue

        new_parameters = []
        data_args = []
        replaced_data_args = {}
        data_var_kwarg = None
        for param in list(signature.parameters.values())[1:]:
            if param.kind == param.VAR_KEYWORD:
                data_var_kwarg = param.name
                replaced_data_args[f"data_{param.name}"] = param.name
                param = param.replace(
                    name=f"data_{param.name}", kind=param.KEYWORD_ONLY, default={}
                )
            elif param.name in plot_cls_params:
                replaced_data_args[f"data_{param.name}"] = param.name
                param = param.replace(name=f"data_{param.name}")

            data_args.append(param.name)
            new_parameters.append(param)

        new_parameters.extend(list(plot_cls_params.values()))

        signature = signature.replace(parameters=new_parameters)

        params_info = {
            "data_args": data_args,
            "replaced_data_args": replaced_data_args,
            "data_var_kwarg": data_var_kwarg,
            "plot_var_kwarg": (
                new_parameters[-1].name
                if new_parameters[-1].kind == inspect.Parameter.VAR_KEYWORD
                else None
            ),
        }

        def _plot(
            obj, *args, __params_info=params_info, __signature=signature, **kwargs
        ):
            sig = __signature
            params_info = __params_info

            bound = sig.bind_partial(**kwargs)

            try:
                data_kwargs = {}
                for k in params_info["data_args"]:
                    if k not in bound.arguments:
                        continue

                    data_key = params_info["replaced_data_args"].get(k, k)
                    if params_info["data_var_kwarg"] == data_key:
                        data_kwargs.update(bound.arguments[k])
                    else:
                        data_kwargs[data_key] = bound.arguments.pop(k)
            except Exception as e:
                raise TypeError(
                    f"Error while parsing arguments to create the {data_source_cls.__name__}"
                )

            for k, v in data_source_init_kwargs.items():
                if k not in data_kwargs:
                    data_kwargs[k] = v

            data = data_source_cls.new(obj, *args, **data_kwargs)

            plot_kwargs = bound.arguments.pop(params_info["plot_var_kwarg"], {})

            return plot_cls(**{setting_key: data, **bound.arguments, **plot_kwargs})

        _plot.__signature__ = signature
        doc = f"Read data into {data_source_cls.__name__} and create a {plot_cls.__name__} from it.\n\n"

        doc += (
            "This function accepts the arguments for creating both the data source and the plot. The following"
            " arguments of the data source have been renamed so that they don't clash with the plot arguments:\n"
            + "\n".join(f" - {v} -> {k}" for k, v in replaced_data_args.items())
            + f"\n\nDocumentation for the {data_source_cls.__name__} creator ({func.__name__})"
            f"\n=============\n{inspect.cleandoc(func.__doc__) if func.__doc__ is not None else None}"
            f"\n\nDocumentation for {plot_cls.__name__}:"
            f"\n=============\n{inspect.cleandoc(plot_cls.__doc__) if plot_cls.__doc__ is not None else None}"
        )

        _plot.__doc__ = doc

        try:
            this_default = plotable in default
        except:
            this_default = False

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
    sile_cls,
    method: str,
    plot_cls,
    setting_key,
    name=None,
    default=False,
    plot_handler_attr="plot",
    **kwargs,
):
    plot_cls_params = {
        name: param.replace(kind=inspect.Parameter.KEYWORD_ONLY)
        for name, param in inspect.signature(plot_cls).parameters.items()
        if name != setting_key
    }

    func = getattr(sile_cls, method)

    signature = inspect.signature(getattr(sile_cls, method))

    new_parameters = []
    data_args = []
    replaced_data_args = {}
    data_var_kwarg = None
    for param in list(signature.parameters.values())[1:]:
        if param.kind == param.VAR_KEYWORD:
            data_var_kwarg = param.name
            replaced_data_args[param.name] = f"data_{param.name}"
            param = param.replace(
                name=f"data_{param.name}", kind=param.KEYWORD_ONLY, default={}
            )
        elif param.name in plot_cls_params:
            replaced_data_args[param.name] = f"data_{param.name}"
            param = param.replace(name=f"data_{param.name}")

        data_args.append(param.name)
        new_parameters.append(param)

    new_parameters.extend(list(plot_cls_params.values()))

    params_info = {
        "data_args": data_args,
        "replaced_data_args": replaced_data_args,
        "data_var_kwarg": data_var_kwarg,
        "plot_var_kwarg": (
            new_parameters[-1].name
            if len(new_parameters) > 0
            and new_parameters[-1].kind == inspect.Parameter.VAR_KEYWORD
            else None
        ),
    }

    signature = signature.replace(parameters=new_parameters)

    def _plot(obj, *args, **kwargs):
        bound = signature.bind_partial(**kwargs)

        try:
            data_kwargs = {}
            for k in params_info["data_args"]:
                if k not in bound.arguments:
                    continue

                data_key = params_info["replaced_data_args"].get(k, k)
                if params_info["data_var_kwarg"] == data_key:
                    data_kwargs.update(bound.arguments[k])
                else:
                    data_kwargs[data_key] = bound.arguments.pop(k)
        except:
            raise TypeError(
                f"Error while parsing arguments to create the call {method}"
            )

        data = func(obj, *args, **data_kwargs)

        plot_kwargs = bound.arguments.pop(params_info["plot_var_kwarg"], {})

        return plot_cls(**{setting_key: data, **bound.arguments, **plot_kwargs})

    _plot.__signature__ = signature
    doc = f"Calls {method} and creates a {plot_cls.__name__} from its output.\n\n"

    doc += (
        f"This function accepts the arguments both for calling {method} and creating the plot. The following"
        f" arguments of {method} have been renamed so that they don't clash with the plot arguments:\n"
        + "\n".join(f" - {k} -> {v}" for k, v in replaced_data_args.items())
        + f"\n\nDocumentation for {method} "
        f"\n=============\n{inspect.cleandoc(func.__doc__) if func.__doc__ is not None else None}"
        f"\n\nDocumentation for {plot_cls.__name__}:"
        f"\n=============\n{inspect.cleandoc(plot_cls.__doc__) if plot_cls.__doc__ is not None else None}"
    )

    _plot.__doc__ = doc

    register_plotable(
        sile_cls,
        plot_cls=plot_cls,
        plotting_func=_plot,
        name=name,
        default=default,
        plot_handler_attr=plot_handler_attr,
        **kwargs,
    )
