from .context import lazy_context
from .node import Node


class Dispatcher(Node):
    
    _dispatchs = {}
    _node = None
    _dispatch_input = "key"

    def __init_subclass__(cls):
        cls._dispatchs = {}
        cls._node = None
        cls._default_dispatch = None
        return super().__init_subclass__()
    
    def _get(self, *args, **kwargs):

        key = kwargs.pop(self._dispatch_input, None)

        if key is None:
            key = self._default_dispatch

        if isinstance(key, type) and issubclass(key, Node):
            kls = key
        else:
            if key not in self._dispatchs:
                raise ValueError(f"Registered nodes have keys: {list(self._dispatchs)}, but {key} was requested")
            
            kls = self._dispatchs[key]
        
        with lazy_context(nodes=True):
            self._node = kls(*args, **kwargs)
        
        return self._node.get()

    def __getattr__(self, key):
        if key != "_node":
            if self._node is None:
                self.get()
            return getattr(self._node, key)
    
    @classmethod
    def register(cls, key, node_cls, default=False):
        if default or len(cls._dispatchs) == 0:
            cls._default_dispatch = key

        cls._dispatchs[key] = node_cls