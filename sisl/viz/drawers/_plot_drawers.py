from abc import ABC, abstractmethod

from ..configurable import _populate_with_settings

__all__ = []

class Drawers:

    def __init__(self, plot_cls):
        self._drawers = {}

        self._plot_cls = plot_cls
        self._plot_cls_params = self._plot_cls._get_class_params()
        self._plot_cls_params_keys = [param["key"] for param in self._plot_cls_params[0]]

        self._plot_cls._drawer = None

    def register(self, drawer_name, drawer):
        drawer._drawer_name = drawer_name
        self._drawers[drawer_name] = drawer

    def setup(self, plot, drawer_name):
        current_drawer = getattr(plot, "_drawer", None)
        if current_drawer is None or current_drawer._drawer_name != drawer_name:
            if drawer_name not in self._drawers:
                raise NotImplementedError(f"There is no '{drawer_name}' drawer implemented for {self._plot_cls.__name__} or the drawer has not been loaded.")
            plot._drawer = self._drawers[drawer_name]()

class Drawer(ABC):

    @abstractmethod
    def clear(self):
        """Clears the figure so that we can draw again."""