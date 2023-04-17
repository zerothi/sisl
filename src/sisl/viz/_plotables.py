# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""
This file provides tools to handle plotability of objects
"""
from sisl._dispatcher import ClassDispatcher, AbstractDispatch, ObjectDispatcher

__all__ = ["register_plotable"]


class ClassPlotHandler(ClassDispatcher):
    """Handles all plotting possibilities for a class"""

    def __init__(self, *args, **kwargs):
        if not "instance_dispatcher" in kwargs:
            kwargs["instance_dispatcher"] = ObjectPlotHandler
        kwargs["type_dispatcher"] = None
        super().__init__(*args, **kwargs)


class ObjectPlotHandler(ObjectDispatcher):
    """Handles all plotting possibilities for an object."""

    def __call__(self, *args, **kwargs):
        """If the plot handler is called, we will run the default plotting function
        unless the keyword method has been passed."""
        return getattr(self, kwargs.pop("method", self._default) or self._default)(*args, **kwargs)


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
        (PlotDispatch, ),
        {"_plot": staticmethod(function), "__doc__": function.__doc__}
    )


def _get_plotting_func(PlotClass, setting_key):
    """
    Generates a plotting function for an object.

    Parameters
    -----------
    PlotClass: child of Plot
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
        return PlotClass(*args, **{setting_key: obj, **kwargs})

    _plot.__doc__ = f"""Builds a {PlotClass.__name__} by setting the value of "{setting_key}" to the current object.

    Apart from this specific parameter ,it accepts the same arguments as {PlotClass.__name__}.
    
    Documentation for {PlotClass.__name__}
    -------------
    
    {PlotClass.__doc__}
    """
    return _plot


def register_plotable(plotable, PlotClass=None, setting_key=None, plotting_func=None, name=None, default=False, plot_handler_attr='plot', engine=None, **kwargs):
    """
    Makes the sisl.viz module aware of which sisl objects can be plotted and how to do it.

    The implementation uses plot handlers. The only thing that this function does is to check
    if there is a plot handler, and if not, create it. The rest is handled by the plot handler.

    Parameters
    ------------
    plotable: any
        any class or object that you want to make plotable. Note that, if it's an object, the plotting
        capabilities will be attributed to all instances of the object's class.
    PlotClass: child of sisl.Plot, optional
        The class of the Plot that we want this object to use.
    setting_key: str, optional
        The key of the setting where the object must go. This works together with
        the PlotClass parameter.
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

    # If no plotting function is provided, we will try to create one by using the PlotClass
    # and the setting_key that have been provided
    if plotting_func is None:
        plotting_func = _get_plotting_func(PlotClass, setting_key)

    if name is None:
        # We will take the name of the plot class as the name
        name = PlotClass.suffix()

    # Check if we already have a plot_handler
    plot_handler = plotable.__dict__.get(plot_handler_attr, None)

    # If it's the first time that the class is being registered,
    # let's give the class a plot handler
    if not isinstance(plot_handler, ClassPlotHandler):

        # If the user is passing an instance, we get the class
        if not isinstance(plotable, type):
            plotable = type(plotable)

        setattr(plotable, plot_handler_attr, ClassPlotHandler(plot_handler_attr))

        plot_handler = getattr(plotable, plot_handler_attr)

    plot_dispatch = create_plot_dispatch(plotting_func, name)
    # Register the function in the plot_handler
    plot_handler.register(name, plot_dispatch, default=default, **kwargs)
