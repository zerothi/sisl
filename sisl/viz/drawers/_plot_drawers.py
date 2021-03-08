from abc import ABC

from ..configurable import _populate_with_settings

__all__ = []

class Drawers:

    def __init__(self, plot_cls):
        self._drawers = {}
        self._drawer = None

        self._plot_cls = plot_cls
        self._plot_cls_params = self._plot_cls._get_class_params()
        self._plot_cls_params_keys = [param["key"] for param in self._plot_cls_params[0]]

    def register(self, drawer_name, drawer):
        self._drawers[drawer_name] = drawer

    def setup(self, plot, drawer_name):
        self._drawer = self._drawers[drawer_name]()

    def __getattr__(self, key):
        return getattr(self._drawer, key)

class Drawer(ABC):

    def setup(self):
        """Initializes all the things that the drawer needs"""

    def clear(self):
        """Clears the figure so that we can draw again."""