"""
This file provides tools to handle plotability of objects
"""

from copy import copy
from functools import wraps

__all__ = ["register_plotable"]


class PlotEngine:
    """
    Stores and takes care of calling all methods of a plotting engine.
    """

    def __init__(self, name, handler, default=None, methods={}):

        self._raw_methods = {}
        self._handler = handler

        for name, method in methods.items():
            self._add(name, method)

        self._name = name
        self._default = default

    def copy(self, **kwargs):
        return self.__class__(self._name, kwargs.get("handler", self._handler),
            default=self._default, methods=self._raw_methods)

    def __dir__(self):
        return list(self._raw_methods)

    def _add(self, name, function, default=False):
        """
        Makes a new function available to this plotting engine.

        Parameters
        -----------
        name: str
            the name of the plotting method. 

            This will be the name of the attribute where the function is stored.
        function: function
            the plotting method.
        default: bool, optional
            whether this method is the default for the engine.

            The default method can be called by simply calling the engine.        
        """
        if default:
            self._default = name

        self._raw_methods[name] = function

    def _get_wrapped_function(self, name):
        function = self._raw_methods[name]

        # This is the function that really does the plotting and goes
        # into the engine
        @wraps(function)
        def real_function(*args, **kwargs):
            return function(self._handler._obj, *args, **kwargs)

        return real_function

    def set_handler(self, handler):
        self._handler = handler

        return self

    def __getattr__(self, key):
        return self._get_wrapped_function(key)

    def get(self, method=None, otherwise='raise'):
        """
        Returns a method of the engine.

        Parameters
        -----------
        method: str, optional
            the name of the method that we want.

            If not provided, the default will be returned
        otherwise: any, optional
            what to return in case that the method is not found.

            If you want the method to raise an exception, set this to 'raise'.
        """

        if method is None:
            if self._default is None:
                raise ValueError('There is no default plotting method registered in this plotting engine')
            method = self._default

        if otherwise == 'raise':
            return getattr(self, method)
        else:
            return getattr(self, method, otherwise)

    def __call__(self, method=None, **kwargs):
        """
        Handles a call to the engine, which will trigger calling plot methods.

        Note that you can also call the methods directly, since they are stored
        in attributes.

        Parameters
        -----------
        method: str, optional
            the method that we want to execute.

            If not provided, the default will be executed.
        **kwargs:
            all the keyword arguments that will go into executing the method
        """

        return self.get(method)(**kwargs)


class PlotHandler:
    """
    Handles all plotting possibilities for a class.

    It supports handling multiple plotting engines while keeping autocompletion
    and help messages meaningful to make the plotting experience smooth.

    Parameters
    ----------
    default_engine: str, optional
        the plotting engine that should be used as the default.

        With this, the plotting methods of the engine get exposed also as
        attributes of the plot handler. Calling the plot handler without
        specifying an engine will call the default engine.

    Usage
    --------
    class A:

        plot = PlotHandler()

    or

    class A:
        pass

    A.plot = PlotHandler()

    NOTE: It only works if it is set as an attribute of the class!

    You can not do:

    A().plot = PlotHandler()

    """

    def __init__(self, obj=None, engines=None, default_engine='plotly'):
        self._obj = obj

        self._engines = engines or {}
        self._default_engine = default_engine

    def __dir__(self):
        return ["register", *list(self._engines), *dir(self._engines[self._default_engine])]

    def register(self, function, name=None, engine=None, default=False):
        """
        Registers plotting functions to the plot handler.

        Parameters
        -----------
        function: function
            the plotting function that we want to register.
        name: str, optional
            the name that should be used to identify this plotting method through
            the plot handler and the plot engines.

            If not provided, the name of the function will be used.
        engine: str, optional
            the engine where we should register this plotting function. 

            If it doesn't exist yet, a new plotting engine is created.

            If not provided, the default engine is used.
        default: boolean, optional
            whether this should be set as the default plotting method
            for the engine.

            The default method can be accessed by just calling the engine
            with no "method" argument.
            The default method of the default engine can be accessed by calling the
            plot handler with no "engine" and "method" arguments.
        """
        if name is None:
            name = function.__name__

        if engine is None:
            engine = self._default_engine

        # Initialize a new plot engine, if it isn't already present
        if engine not in self._engines:
            self._engines[engine] = PlotEngine(engine, handler=self)

        engine = self._engines[engine]

        engine._add(name, function, default=default)

    def __getattr__(self, key):

        if key in self._engines:
            return self._engines[key]

        return getattr(self._engines[self._default_engine], key)

    def __get__(self, instance, owner):
        """
        Makes the plot handler aware of what is the instance that it is handling
        plots for.
        """
        if instance is not None:

            new_handler = self.__class__(obj=instance, default_engine=self._default_engine)

            for name, engine in self._engines.items():
                new_handler._engines[name] = engine.copy(handler=new_handler)

            return new_handler

        return self

    def __call__(self, engine=None, method=None, **kwargs):
        """
        Handles a call to the engine, which will trigger calling plot methods.

        Note that you can also call the methods directly, since they are stored
        in attributes in the plot handler (for the default engine) or in the engines,
        which are attributes of the plot handler.

        Parameters
        -----------
        engine: str
            the engine where we should look for methods.

            If not provided the default engine will be used
        method: str, optional
            the method that we want to execute.

            If not provided, the default will be executed.
        **kwargs:
            all the keyword arguments that will go into executing the method
        """

        if engine is None:
            engine = self._default_engine

        return self._engines[engine](method=method, **kwargs)


def register_plotable(plotable, plotting_func, name=None,
                      engine='plotly', default=False, plot_handler_attr='plot'):
    """
    Makes the sisl.viz module aware of which sisl objects can be plotted and how to do it.

    The implementation uses plot handlers. The only thing that this function does is to check
    if there is a plot handler, and if not, create it. The rest is handled by the plot handler.

    Parameters
    ------------
    plotable: any
        any class or object that you want to make plotable. Note that, if it's an object, the plotting
        capabilities will be attributed to all instances of the object's class.
    plotting_func: function
        the function that takes care of the plotting.
        It should accept (self, *args, **kwargs) and return a plot object.
    name: str, optional
        name that will be used to identify the particular plot function that is being registered.

        E.g.: If name is "nicely", the plotting function will be registered under "obj.plot.nicely()"

        If not provided, the name of the function will be used
    default: boolean, optional 
        whether this way of plotting the class should be the default one.
    engine: str, optional
        the engine that the function uses.
    plot_handler_attr: str, optional
        the attribute where the plot handler is or should be located in the class that you want to register.
    """
    # Check if we already have a plot_handler
    plot_handler = getattr(plotable, plot_handler_attr, None)

    # If it's the first time that the class is being registered,
    # let's give the class a plot handler
    if not isinstance(plot_handler, PlotHandler):

        # If the user is passing an instance, we get the class
        if not isinstance(plotable, type):
            plotable = type(plotable)

        setattr(plotable, plot_handler_attr, PlotHandler())

        plot_handler = getattr(plotable, plot_handler_attr)

    # Register the function in the plot_handler
    plot_handler.register(plotting_func, name=name, engine=engine, default=default)
