from abc import ABC, abstractmethod

from ...plot import MultiplePlot, SubPlots, Animation

class Backend(ABC):

    def __init__(self, plot):
        # Let's store our parent plot, we might need it.
        self._plot = plot

    @abstractmethod
    def draw(self, backend_info):
        """Draws the plot, given the info passed by it. 
        
        This is plot specific and is implemented in the templates."""
       
    def draw_other_plot(self, plot, backend=None):
        """Method that draws a different plot in the current canvas.

        Note that the other plot might have a different active backend, which might be incompatible.
        We take care of it in this method.

        This method will be used by `MultiplePlotBackend`, but it's also used in some cases by regular plots.

        NOTE: This needs the `draw_on` method, which is specific to each framework. See below.

        Parameters
        ------------
        plot: Plot
            The plot we want to draw in the current canvas
        backend: str, optional
            The name of the backend that we want to force on the plot to be drawn. If not provided, we use
            the name of the current backend.
        """
        backend_name = backend or self._backend_name

        # Get the current backend of the plot that we have to draw
        plot_backend = getattr(plot, "_backend", None)
        
        # If the current backend of the plot is incompatible with this backend, we are going to
        # setup a compatible backend. Note that here we assume a backend to be compatible if its
        # prefixed with the name of the current backend. I.e. if the current backend is "plotly"
        # "plotly_*" backends are assumed to be compatible.
        if plot_backend is None or not plot_backend._backend_name.startswith(backend_name):
            plot._backends.setup(plot, backend_name)

        # Make the plot draw in this backend instance
        plot.draw_on(self)

        # Restore the initial backend of the plot, so that it doesn't feel affected
        plot._backend = plot_backend

    @abstractmethod
    def draw_on(self, figure):
        """Should draw the method in another instance of a compatible backend.
        
        Parameters
        -----------
        figure:
            The types of objects accepted by this argument are dependent on each backend.
            However, it should always be able to accept a compatible backend. See `PlotlyBackend`
            or `MatplotlibBackend` as examples.
        """

    @abstractmethod
    def clear(self):
        """Clears the figure so that we can draw again."""

    # Methods needed for testing
    def _test_number_of_items_drawn(self):
        """Returns the number of items drawn currently in the plot."""
        raise NotImplementedError
    
    def draw_line(self, x, y, name=None, line={}, marker={}, text=None, **kwargs):
        raise NotImplementedError(f"{self.__class__.__name__} doesn't implement a draw_line method.")

    def draw_scatter(self, x, y, name=None, marker={}, text=None, **kwargs):
        raise NotImplementedError(f"{self.__class__.__name__} doesn't implement a draw_scatter method.")

    def draw_line3D(self, x, y, z, name=None, line={}, marker={}, text=None, **kwargs):
        raise NotImplementedError(f"{self.__class__.__name__} doesn't implement a draw_line3D method.")
    
    def draw_scatter3D(self, x, y, z, name=None, marker={}, text=None, **kwargs):
        raise NotImplementedError(f"{self.__class__.__name__} doesn't implement a draw_scatter3D method.")

class MultiplePlotBackend(Backend):

    def draw(self, backend_info, childs):
        """Recieves the child plots and is responsible for drawing all of them in the same canvas"""
        for child in childs:
            self.draw_other_plot(child)

class SubPlotsBackend(Backend):

    @abstractmethod
    def draw(self, drawer_info, rows, cols, childs, **make_subplots_kwargs):
        """Draws the subplots layout

        It must use `rows` and `cols`, and draw the childs row by row.
        """

class AnimationBackend(Backend):

    @abstractmethod
    def draw(self, drawer_info, childs, get_frame_names):
        """Generates an animation out of the child plots.
        """

MultiplePlot._backends.register_template(MultiplePlotBackend)
SubPlots._backends.register_template(SubPlotsBackend)
Animation._backends.register_template(AnimationBackend)