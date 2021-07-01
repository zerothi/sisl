from abc import ABC, abstractmethod

from ...plot import MultiplePlot, SubPlots, Animation

class Backend(ABC):

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

    @abstractmethod
    def draw(self, drawer_info, childs):
        """Recieves the child plots and is responsible for drawing all of them in the same canvas"""

class SubPlotsBackend(Backend):

    @abstractmethod
    def draw_subplots(self, drawer_info, rows, cols, childs, **make_subplots_kwargs):
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