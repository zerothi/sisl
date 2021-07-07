__all__ = []

class Backends:
    """The backends manager for a plot class"""

    def __init__(self, plot_cls):
        self._backends = {}
        self._template = None

        self._cls = plot_cls

        self._cls._backend = None

    def register(self, backend_name, backend, default=False):
        """Register a new backend to the available backends.

        Note that if there is a template registered, you can only register backends that
        inherit from that template, otherwise a `TypeError` will be raised.

        Parameters
        -----------
        backend_name: str
            The name of the backend being registered. Users will need to pass this value
            in order to choose this backend.
        backend: Backend
            The backend class to be registered
        default: bool, optional
            Whether this backend should be the default one.
        """
        if self._template is not None:
            if not issubclass(backend, self._template):
                raise TypeError(f"Error registering '{backend_name}': Backends for {self._cls.__name__} should inherit from {self._template.__name__}")

        # Update the options of the backend setting
        backend_param = self._cls.get_class_param("backend")
        backend_param.options = [*backend_param.get_options(raw=True), {"label": backend_name, "value": backend_name}]
        if backend_param.default is None or default:
            backend_param.default = backend_name

        backend._backend_name = backend_name
        self._backends[backend_name] = backend

    def setup(self, plot, backend_name):
        """Sets up the backend for a given plot.

        Note that if the current backend of the plot is already `backend_name`, then nothing is done.
        Also, if the requested `backend_name` is not available, a `NotImplementedError` is raised.
        Parameters
        -----------
        plot: Plot
            The plot for which we want to set up a backend.
        backend_name: str
            The name of the backend we want to initialize.
        """
        current_backend = getattr(plot, "_backend", None)
        if current_backend is None or current_backend._backend_name != backend_name:
            if backend_name not in self._backends:
                raise NotImplementedError(f"There is no '{backend_name}' backend implemented for {self._cls.__name__} or the backend has not been loaded.")
            plot._backend = self._backends[backend_name](plot)
    
    def register_template(self, template):
        """Sets a template that all registered backends have to satisfy.

        That is, any backend that you want to register here needs to inherit from this template.
        
        Parameters
        -----------
        template: Backend
            The backend class that should be used as a template.
        """
        self._template = template
    
    @property
    def options(self):
        return list(self._backends)