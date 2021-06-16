__all__ = []

class Backends:
    """The backends manager for a plot class"""

    def __init__(self, plot_cls):
        self._backends = {}
        self._template = None

        self._cls = plot_cls
        self._cls_params = self._cls._get_class_params()
        self._cls_params_keys = [param["key"] for param in self._cls_params[0]]

        self._cls._backend = None

    def register(self, backend_name, backend):

        if self._template is not None:
            if not issubclass(backend, self._template):
                raise TypeError(f"Error registering '{backend_name}': Backends for {self._cls.__name__} should inherit from {self._template.__name__}")

        backend._backend_name = backend_name
        self._backends[backend_name] = backend

    def setup(self, plot, backend_name):
        current_backend = getattr(plot, "_backend", None)
        if current_backend is None or current_backend._backend_name != backend_name:
            if backend_name not in self._backends:
                raise NotImplementedError(f"There is no '{backend_name}' backend implemented for {self._plot_cls.__name__} or the backend has not been loaded.")
            plot._backend = self._backends[backend_name]()
    
    def register_template(self, template):
        self._template = template