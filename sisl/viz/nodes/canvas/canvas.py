import numpy as np

from sisl.viz.plotutils import values_to_colors

from ..node import Node

class CanvasNode(Node):
    """Base plotter class that all backends should inherit from.

    It contains all the plotting actions that should be supported by a canvas.

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
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _get(self, plot_actions, *args, **kwargs):

        if isinstance(plot_actions, dict):
            composite_method = plot_actions['composite_method']
            plot_actions = plot_actions["plot_actions"]
        else:
            composite_method = None
            plot_actions = [plot_actions]

        fig = self.init_figure(composite_method=composite_method)
        self._coloraxes = {}

        for section_actions in plot_actions:
            for action in section_actions:
                getattr(self, action['method'])(*action['args'], **action['kwargs'])
            
        return fig
    
    def init_figure(self, composite_method=None):
        if composite_method is None:
            return self._init_figure()
        elif composite_method == "multiple":
            return self._init_figure_multiple()
        elif composite_method == "animate":
            return self._init_figure_animated()
        elif composite_method == "subplot":
            return self._init_figure_subplots()
        else:
            raise ValueError(f"Unknown composite method '{composite_method}'")

    def _init_figure(self):
        raise NotImplementedError(f"{self.__class__.__name__} doesn't implement a _init_figure method.")

    def _init_figure_multiple(self):
        raise NotImplementedError(f"{self.__class__.__name__} doesn't implement a _init_figure_multiple method.")
    
    def _init_figure_animated(self):
        raise NotImplementedError(f"{self.__class__.__name__} doesn't implement a _init_figure_animated method.")
    
    def _init_figure_subplots(self):
        raise NotImplementedError(f"{self.__class__.__name__} doesn't implement a _init_figure_subplots method.")
    
    # def _ipython_display_(self, return_figWidget=False, **kwargs):
    #     """ Handles all things needed to display the plot in a jupyter notebook.

    #     Parameters
    #     ------
    #     return_figureWidget: bool, optional
    #         if the plot is displayed in a jupyter notebook, whether you want to
    #         get the figure widget as a return so that you can act on it.
    #     """
    #     from IPython.display import display

    #     widget = self.get_ipywidget()

    #     # Else, show without shortcut support
    #     display(widget)

    #     if return_figWidget:
    #         return widget

    def get_ipywidget(self):
        raise NotImplementedError(f"{self.__class__.__name__} doesn't implement a get_ipywidget method.")

    # def draw_other_plot(self, plot, backend=None, **kwargs):
    #     """Method that draws a different plot in the current canvas.

    #     Note that the other plot might have a different active backend, which might be incompatible.
    #     We take care of it in this method.

    #     This method will be used by `MultiplePlotBackend`, but it's also used in some cases by regular plots.

    #     NOTE: This needs the `draw_on` method, which is specific to each framework. See below.

    #     Parameters
    #     ------------
    #     plot: Plot
    #         The plot we want to draw in the current canvas
    #     backend: str, optional
    #         The name of the backend that we want to force on the plot to be drawn. If not provided, we use
    #         the name of the current backend.
    #     **kwargs:
    #         passed directly to `draw_on`
    #     """
    #     backend_name = backend or self._backend_name

    #     # Get the current backend of the plot that we have to draw
    #     plot_backend = getattr(plot, "_backend", None)

    #     # If the current backend of the plot is incompatible with this backend, we are going to
    #     # setup a compatible backend. Note that here we assume a backend to be compatible if its
    #     # prefixed with the name of the current backend. I.e. if the current backend is "plotly"
    #     # "plotly_*" backends are assumed to be compatible.
    #     if plot_backend is None or not plot_backend._backend_name.startswith(backend_name):
    #         plot.backends.setup(plot, backend_name)

    #     # Make the plot draw in this backend instance
    #     plot.draw_on(self, **kwargs)

    #     # Restore the initial backend of the plot, so that it doesn't feel affected
    #     plot._backend = plot_backend

    # def draw_on(self, figure, **kwargs):
    #     """Should draw the method in another instance of a compatible backend.

    #     Parameters
    #     -----------
    #     figure:
    #         The types of objects accepted by this argument are dependent on each backend.
    #         However, it should always be able to accept a compatible backend. See `PlotlyBackend`
    #         or `MatplotlibBackend` as examples.
    #     """
    #     raise NotImplementedError(f"{self.__class__.__name__} does not implement a 'draw_on' method and therefore doesn't know"+
    #         "how to draw outside its own instance.")

    def clear(self):
        """Clears the figure so that we can draw again."""
        pass

    def show(self):
        pass

    def init_3D(self):
        """Called if functions that draw in 3D are going to be called."""
        return

    def init_coloraxis(self, name, cmin, cmax, colorscale, **kwargs):
        """Initializes a color axis to be used by the drawing functions"""
        self._coloraxes[name] = {
            'cmin': cmin,
            'cmax': cmax,
            'colorscale': colorscale,
            **kwargs
        }

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

    def draw_multicolor_line(self, *args, line={}, **kwargs):
        """By default, multicoloured lines are drawn simply by drawing the points."""
        marker = {
            **kwargs.pop('marker', {}),
            'color': line.get('color'),
            'size': line.get('width'),
            'opacity': line.get('opacity'),
            'coloraxis': line.get('coloraxis')
        }
        self.draw_multicolor_scatter(*args, marker=marker, **kwargs)

    def draw_area_line(self, x, y, name=None, line={}, text=None, **kwargs):
        """Same as draw line, but to draw a line with an area. This is for example used to draw fatbands.

        Parameters
        -----------
        x: array-like
            the coordinates of the points along the X axis.
        y: array-like
            the coordinates of the points along the Y axis.
        name: str, optional
            the name of the scatter
        line: dict, optional
            specifications for the line style, following plotly standards. The backend
            should at least be able to implement `line["color"]` and `line["width"]`, but
            it is very advisable that it supports also `line["opacity"]`.
        text: str, optional
            contains the text asigned to each marker. On plotly this is seen on hover,
            other options could be annotating. However, it is not necessary that this
            argument is supported.
        **kwargs:
            should allow other keyword arguments to be passed directly to the creation of
            the scatter. This will of course be framework specific
        """
        raise NotImplementedError(f"{self.__class__.__name__} doesn't implement a draw_area_line method.")

    def draw_multicolor_area_line(self, x, y, name=None, line={}, text=None, **kwargs):
        """Same as draw line, but to draw a line with an area. This is for example used to draw fatbands.

        Parameters
        -----------
        x: array-like
            the coordinates of the points along the X axis.
        y: array-like
            the coordinates of the points along the Y axis.
        name: str, optional
            the name of the scatter
        line: dict, optional
            specifications for the line style, following plotly standards. The backend
            should at least be able to implement `line["color"]` and `line["width"]`, but
            it is very advisable that it supports also `line["opacity"]`.
        text: str, optional
            contains the text asigned to each marker. On plotly this is seen on hover,
            other options could be annotating. However, it is not necessary that this
            argument is supported.
        **kwargs:
            should allow other keyword arguments to be passed directly to the creation of
            the scatter. This will of course be framework specific
        """
        raise NotImplementedError(f"{self.__class__.__name__} doesn't implement a draw_multicolor_area_line method.")

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

    def draw_multicolor_scatter(self, *args, **kwargs):
        # Usually, multicoloured scatter plots are already supported.
        return self.draw_scatter(*args, **kwargs)

    def draw_arrows(self, xy, dxy, arrowhead_scale=0.2, arrowhead_angle=20, **kwargs):
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
            the angle that the arrow head forms with the direction of the arrow (in degrees).
        """
        # Get the destination of the arrows
        final_xy = xy + dxy

        # Convert from degrees to radians.
        arrowhead_angle = np.radians(arrowhead_angle)

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

    def draw_line_3D(self, x, y, z, name=None, line={}, marker={}, text=None, **kwargs):
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
        raise NotImplementedError(f"{self.__class__.__name__} doesn't implement a draw_line_3D method.")

    def draw_multicolor_line_3D(self, *args, **kwargs):
        self.draw_line_3D(*args, **kwargs)

    def draw_scatter_3D(self, x, y, z, name=None, marker={}, text=None, **kwargs):
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
        raise NotImplementedError(f"{self.__class__.__name__} doesn't implement a draw_scatter_3D method.")

    def draw_multicolor_scatter_3D(self, *args, **kwargs):
        # Usually, multicoloured scatter plots are already supported.
        return self.draw_scatter_3D(*args, **kwargs)

    def draw_balls_3D(self, x, y, z, name=None, markers={}, **kwargs):
        return NotImplementedError(f"{self.__class__.__name__} doesn't implement a draw_balls_3D method.")

    def draw_multicolor_balls_3D(self, x, y, z, name=None, marker={}, **kwargs):

        kwargs['marker'] = marker.copy()

        if 'color' in marker and np.array(marker['color']).dtype in (int, float):
            coloraxis = kwargs['marker']['coloraxis']
            coloraxis = self._coloraxes[coloraxis]

            kwargs['marker']['color'] = values_to_colors(kwargs['marker']['color'], coloraxis['colorscale'] or "viridis")

        return self.draw_balls_3D(x, y, z, name=name, **kwargs)

    def draw_arrows_3D(self, xyz, dxyz, arrowhead_scale=0.3, arrowhead_angle=15, **kwargs):
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
            the angle that the arrow head forms with the direction of the arrow (in degrees).
        """
        # Get the destination of the arrows
        final_xyz = xyz + dxyz

        # Convert from degrees to radians.
        arrowhead_angle = np.radians(arrowhead_angle)

        # Calculate the arrowhead positions. This is a bit more complex than the 2D case,
        # since there's no unique plane to rotate all vectors.
        # First, we get a unitary vector that is perpendicular to the direction of the arrow in xy.
        dxy_norm = np.linalg.norm(dxyz[:, :2], axis=1)
        # Some vectors might be only in the Z direction, which will result in dxy_norm being 0.
        # We avoid problems by dividinc
        dx_p = np.divide(dxyz[:, 1], dxy_norm, where=dxy_norm != 0, out=np.zeros(dxyz.shape[0], dtype=np.float64))
        dy_p = np.divide(-dxyz[:, 0], dxy_norm, where=dxy_norm != 0, out=np.ones(dxyz.shape[0], dtype=np.float64))

        # And then we build the rotation matrices. Since each arrow needs a unique rotation matrix,
        # we will have n 3x3 matrices, where n is the number of arrows, for each arrowhead tip.
        c = np.cos(arrowhead_angle)
        s = np.sin(arrowhead_angle)

        # Rotation matrix to build the first arrowhead tip positions.
        rot_matrices = np.array(
            [[c + (dx_p ** 2) * (1 - c), dx_p * dy_p * (1 - c), dy_p * s],
            [dy_p * dx_p * (1 - c), c + (dy_p ** 2) * (1 - c), -dx_p * s],
            [-dy_p * s, dx_p * s, np.full_like(dx_p, c)]])

        # The opposite rotation matrix, to get the other arrowhead's tip positions.
        inv_rots = rot_matrices.copy()
        inv_rots[[0, 1, 2, 2], [2, 2, 0, 1]] *= -1

        # Calculate the tips of the arrow heads.
        arrowhead_tips1 = final_xyz - np.einsum("ij...,...j->...i", rot_matrices, dxyz * arrowhead_scale)
        arrowhead_tips2 = final_xyz - np.einsum("ij...,...j->...i", inv_rots, dxyz * arrowhead_scale)

        # Now build an array with all the information to draw the arrows
        # This has shape (n_arrows * 7, 3). The information to draw an arrow
        # occupies 7 rows and the columns are the x and y coordinates.
        arrows = np.empty((xyz.shape[0]*7, 3))

        arrows[0::7] = xyz
        arrows[1::7] = final_xyz
        arrows[2::7] = np.nan
        arrows[3::7] = arrowhead_tips1
        arrows[4::7] = final_xyz
        arrows[5::7] = arrowhead_tips2
        arrows[6::7] = np.nan

        return self.draw_line_3D(arrows[:, 0], arrows[:, 1], arrows[:, 2], **kwargs)

    def draw_heatmap(self, values, x=None, y=None, name=None, zsmooth=False, coloraxis=None):
        raise NotImplementedError(f"{self.__class__.__name__} doesn't implement a draw_heatmap method.")

    def draw_mesh_3D(self, vertices, faces, color=None, opacity=None, name=None, **kwargs):
        raise NotImplementedError(f"{self.__class__.__name__} doesn't implement a draw_mesh_3D method.")

    def set_axis(self, **kwargs):
        """Sets the axis parameters.

        The specification for the axes is exactly the plotly one. This is to have a good
        reference for consistency. Other frameworks should translate the calls to their
        functionality.
        """
    
    def set_axes_equal(self):
        """Sets the axes equal."""
        raise NotImplementedError(f"{self.__class__.__name__} doesn't implement a set_axes_equal method.")