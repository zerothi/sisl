from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass

from typing import Any, Dict, Type, Optional
import typing

@dataclass
class InputParams:
    key: str | None = None
    name: str | None = None
    doc: str | None = None
    default: Any = None

    def __init_subclass__(cls) -> None:
        _params_subclasses[cls.__name__] = cls

_params_subclasses: Dict[str, Type[InputParams]] = {
    "InputParams": InputParams,
}

class InputField:
    params: InputParams
    _parsers: typing.Dict[str, Type[Parser]] = {}
    _current_parser: Optional[str] | None = None

    def __init__(self, params: InputParams | None = None):
        # Make sure the input field has its params initialized.
        if params is None:
            if not hasattr(self, "params"):
                # Use the default parameters for this class
                self.params = self._params_init()
        else:
            self.params = params

    @classmethod
    def _params_init(cls, *args, **kwargs) -> InputParams:
        # We parse the annotation of the params variable ourselves because typing.get_type_hints
        # is not able to correctly find the InputParams classes when the annotation is a string
        # that must be evaluated.
        param_cls = cls.__annotations__['params']
        if isinstance(param_cls, str):
            param_cls = _params_subclasses[param_cls]
        return param_cls(*args, **kwargs)

    def __init_subclass__(cls) -> None:
        # Initialize the parsers for this class
        cls._parsers = {}

    @classmethod
    def register_parser(cls, key: str, parser: Type[Parser]) -> None:
        cls._parsers[key] = parser
        # We set the attribute because in this way it can be found from subclasses.
        setattr(cls, f"_{key}_parser", parser)

    @classmethod
    def get_parser(cls, key: str):
        return getattr(cls, f"_{key}_parser")

    @classmethod
    def get_current_parser(cls):
        key = cls._current_parser
        if key is None:
            return None
        return cls.get_parser(key)

    @classmethod
    def from_typehint(cls, type_):
        return cls()

    def __getattr__(self, key):
        if key != "params" and hasattr(self.params, key):
            return getattr(self.params, key)

        raise AttributeError(f"{self.__class__.__name__} has no attribute {key}")

    def parse(self, val: Any) -> Any:
        parser = self.get_current_parser()

        if parser is None:
            return val
        else:
            return parser().parse(val)
    
    def _raise_type_error(self, val) -> None:
        raise TypeError(f"{self.__class__.__name__} received input of type {type(val)}: {val}")

class Parser(ABC):
    """Parsers are required to parse the input
    """
    @abstractmethod
    def parse(self, val: Any) -> Any:
        ...
    
    @abstractmethod
    def unparse(self, val: Any) -> Any:
        ...
