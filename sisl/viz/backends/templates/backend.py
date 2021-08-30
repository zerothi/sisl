from abc import ABC, abstractmethod

from ...plot import MultiplePlot, SubPlots, Animation

import numpy as np


class Backend(ABC):
    """Base backend class that all backends should inherit from.

    This class contains various methods that need to be implemented by its subclasses.

    Methods that MUST be implemented are marked as abstract methods, therefore you won't
    even be able to use the class if you don't implement them. On the other hand, there are
    methods that are not absolutely essential to the general workings of the framework. 
    These are written in this class to raise a NotImplementedError. Therefore, the backend 
    will be instantiable but errors may happen during the plotting process.

    Below are all methods that need to be implemented by...

    (1) the generic backend of the framework:
        - `clear`, MUST
        - `draw_on`, optional (highly recommended, otherwise no multiple plot functionality)
        - `draw_line`, optional (highly recommended for 2D)
        - `draw_scatter`, optional (highly recommended for 2D)
        - `draw_line3D`, optional
        - `draw_scatter3D`, optional
        - `draw_arrows3D`, optional
        - `show`, optional

    (2) specific backend of a plot:
        - `draw`, MUST

    Also, you probably need to write an `__init__` method to initialize the state of the plot.
    Usually drawing methods will add to the state and finally on `show` you display the full
    plot.
    """

    def __init__(self, plot):
        # Let's store our parent plot, we might need it.
        self._plot = plot

    @abstractmethod
    def draw(self, backend_info):
        """Draws the plot, given the info passed by it. 

        This is plot specific and is implemented in the templates, you don't need to worry about it!
        For example: if you inherit from `BandsBackend`, this class already contains a draw method that
        manages the flow of drawing the bands.
        """

    def draw_other_plot(self, plot, backend=None, **kwargs):
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
        **kwargs:
            passed directly to `draw_on`
        """
        backend_name = backend or self._backend_name

        # Get the current backend of the plot that we have to draw
        plot_backend = getattr(plot, "_backend", None)

        # If the current backend of the plot is incompatible with this backend, we are going to
        # setup a compatible backend. Note that here we assume a backend to be compatible if its
        # prefixed with the name of the current backend. I.e. if the current backend is "plotly"
        # "plotly_*" backends are assumed to be compatible.
        if plot_backend is None or not plot_backend._backend_name.startswith(backend_name):
            plot.backends.setup(plot, backend_name)

        # Make the plot draw in this backend instance
        plot.draw_on(self, **kwargs)

        # Restore the initial backend of the plot, so that it doesn't feel affected
        plot._backend = plot_backend

    def draw_on(self, figure, **kwargs):
        """Should draw the method in another instance of a compatible backend.

        Parameters
        -----------
        figure:
            The types of objects accepted by this argument are dependent on each backend.
            However, it should always be able to accept a compatible backend. See `PlotlyBackend`
            or `MatplotlibBackend` as examples.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not implement a 'draw_on' method and therefore doesn't know"+
            "how to draw outside its own instance.")

    @abstractmethod
    def clear(self):
        """Clears the figure so that we can draw again."""

    def show(self):
        pass

    # Methods needed for testing
    def _test_number_of_items_drawn(self):
        """Returns the number of items drawn currently in the plot."""
        raise NotImplementedError

    def draw_line(self, x, y, name=None, line={}, marker={}, text=None, **kwargs):
        """Should draw a line satisfying the specifications

        Parameters
        -----------
        x: array-like
            the coordinates of the points along the X axis.
        y: array-like
            the coordinates of the points along the Y axis.
        name: str, optional
            the name of the line
        line: dict, optional
            specifications for the line style, following plotly standards. The backend
            should at least be able to implement `line["color"]` and `line["width"]`
        marker: dict, optional
            specifications for the markers style, following plotly standards. The backend
            should at least be able to implement `marker["color"]` and `marker["size"]`
        text: str, optional
            contains the text asigned to each marker. On plotly this is seen on hover,
            other options could be annotating. However, it is not necessary that this
            argument is supported.
        **kwargs:
            should allow other keyword arguments to be passed directly to the creation of
            the line. This will of course be framework specific
        """
        raise NotImplementedError(f"{self.__class__.__name__} doesn't implement a draw_line method.")

    def draw_scatter(self, x, y, name=None, marker={}, text=None, **kwargs):
        """Should draw a scatter satisfying the specifications

        Parameters
        -----------
        x: array-like
            the coordinates of the points along the X axis.
        y: array-like
            the coordinates of the points along the Y axis.
        name: str, optional
            the name of the scatter
        marker: dict, optional
            specifications for the markers style, following plotly standards. The backend
            should at least be able to implement `marker["color"]` and `marker["size"]`, but
            it is very advisable that it supports also `marker["opacity"]` and `marker["colorscale"]`
        text: str, optional
            contains the text asigned to each marker. On plotly this is seen on hover,
            other options could be annotating. However, it is not necessary that this
            argument is supported.
        **kwargs:
            should allow other keyword arguments to be passed directly to the creation of
            the scatter. This will of course be framework specific
        """
        raise NotImplementedError(f"{self.__class__.__name__} doesn't implement a draw_scatter method.")
    
    def draw_arrows(self, xy, dxy, arrowhead_scale=0.2, arrowhead_angle=np.pi / 9, **kwargs):
        """Draws multiple arrows using the generic draw_line method.

        Parameters
        -----------
        xy: np.ndarray of shape (n_arrows, 2)
            the positions where the atoms start.
        dxy: np.ndarray of shape (n_arrows, 2)
            the arrow vector.
        arrow_head_scale: float, optional
            how big is the arrow head in comparison to the arrow vector.
        arrowhead_angle: angle
            the angle that the arrow head forms with the direction of the arrow.
        """
        # Get the destination of the arrows
        final_xy = xy + dxy

        # Get the rotation matrices to get the tips of the arrowheads
        rot_matrix = np.array([[np.cos(arrowhead_angle), -np.sin(arrowhead_angle)], [np.sin(arrowhead_angle), np.cos(arrowhead_angle)]])
        inv_rot = np.linalg.inv(rot_matrix)

        # Calculate the tips of the arrow heads
        arrowhead_tips1 = final_xy - (dxy*arrowhead_scale).dot(rot_matrix)
        arrowhead_tips2 = final_xy - (dxy*arrowhead_scale).dot(inv_rot)

        # Now build an array with all the information to draw the arrows
        # This has shape (n_arrows * 7, 2). The information to draw an arrow
        # occupies 7 rows and the columns are the x and y coordinates.
        arrows = np.empty((xy.shape[0]*7, xy.shape[1]), dtype=np.float64)

        arrows[0::7] = xy
        arrows[1::7] = final_xy
        arrows[2::7] = np.nan
        arrows[3::7] = arrowhead_tips1
        arrows[4::7] = final_xy
        arrows[5::7] = arrowhead_tips2
        arrows[6::7] = np.nan
        
        return self.draw_line(arrows[:, 0], arrows[:, 1], **kwargs)

    def draw_line3D(self, x, y, z, name=None, line={}, marker={}, text=None, **kwargs):
        """Should draw a 3D line satisfying the specifications

        Parameters
        -----------
        x: array-like
            the coordinates of the points along the X axis.
        y: array-like
            the coordinates of the points along the Y axis.
        z: array-like
            the coordinates of the points along the Z axis.
        name: str, optional
            the name of the line
        line: dict, optional
            specifications for the line style, following plotly standards. The backend
            should at least be able to implement `line["color"]` and `line["width"]`
        marker: dict, optional
            specifications for the markers style, following plotly standards. The backend
            should at least be able to implement `marker["color"]` and `marker["size"]`
        text: str, optional
            contains the text asigned to each marker. On plotly this is seen on hover,
            other options could be annotating. However, it is not necessary that this
            argument is supported.
        **kwargs:
            should allow other keyword arguments to be passed directly to the creation of
            the line. This will of course be framework specific
        """
        raise NotImplementedError(f"{self.__class__.__name__} doesn't implement a draw_line3D method.")

    def draw_scatter3D(self, x, y, z, name=None, marker={}, text=None, **kwargs):
        """Should draw a 3D scatter satisfying the specifications

        Parameters
        -----------
        x: array-like
            the coordinates of the points along the X axis.
        y: array-like
            the coordinates of the points along the Y axis.
        z: array-like
            the coordinates of the points along the Z axis.
        name: str, optional
            the name of the scatter
        marker: dict, optional
            specifications for the markers style, following plotly standards. The backend
            should at least be able to implement `marker["color"]` and `marker["size"]`
        text: str, optional
            contains the text asigned to each marker. On plotly this is seen on hover,
            other options could be annotating. However, it is not necessary that this
            argument is supported.
        **kwargs:
            should allow other keyword arguments to be passed directly to the creation of
            the scatter. This will of course be framework specific
        """
        raise NotImplementedError(f"{self.__class__.__name__} doesn't implement a draw_scatter3D method.")
    
    def draw_arrows3D(self, xyzs, dxyzs, **kwargs):
        for xyz, dxyz in zip(xyzs, dxyzs):
            self.draw_arrow3D(xyz, dxyz, **kwargs)

    def draw_arrow3D(self, xyz, dxyz, **kwargs):
        raise NotImplementedError(f"{self.__class__.__name__} doesn't implement a draw_arrows3D method.")


class MultiplePlotBackend(Backend):

    def draw(self, backend_info):
        """Recieves the child plots and is responsible for drawing all of them in the same canvas"""
        for child in backend_info["children"]:
            self.draw_other_plot(child)


class SubPlotsBackend(Backend):

    @abstractmethod
    def draw(self, backend_info):
        """Draws the subplots layout

        It must use `rows` and `cols`, and draw the children row by row.
        """


class AnimationBackend(Backend):

    @abstractmethod
    def draw(self, backend_info):
        """Generates an animation out of the child plots.
        """

MultiplePlot.backends.register_template(MultiplePlotBackend)
SubPlots.backends.register_template(SubPlotsBackend)
Animation.backends.register_template(AnimationBackend)
