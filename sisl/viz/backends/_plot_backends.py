from abc import ABC, abstractmethod

__all__ = []

class Backends:

    def __init__(self, plot_cls):
        self._backends = {}

        self._cls = plot_cls
        self._cls_params = self._cls._get_class_params()
        self._cls_params_keys = [param["key"] for param in self._cls_params[0]]

        self._cls._backend = None

    def register(self, backend_name, backend):
        backend._backend_name = backend_name
        self._backends[backend_name] = backend

    def setup(self, plot, backend_name):
        current_backend = getattr(plot, "_backend", None)
        if current_backend is None or current_backend._backend_name != backend_name:
            if backend_name not in self._backends:
                raise NotImplementedError(f"There is no '{backend_name}' backend implemented for {self._plot_cls.__name__} or the backend has not been loaded.")
            plot._backend = self._backends[backend_name]()

class Backend(ABC):

    @abstractmethod
    def clear(self):
        """Clears the figure so that we can draw again."""