"""This module implements a series of parsers to go from and to strings.

They are to be used, for example, when calling functions from the CLI.
"""
from typing import Callable, Any, Type
from sisl.viz.input_fields.basic.dict import DictInput
from sisl.viz.input_fields.basic.list import ListInput
from sisl.viz.input_fields.basic.number import FloatInput, IntegerInput, NumericInput

from sisl.viz.input_fields.basic.text import TextInput

from .._input_field import InputField, Parser
from .. import BoolInput

class BaseStringParser(Parser):
    _dtype: Callable[[str], Any] = lambda x: x

    def parse(self, val: str) -> Any:
        return self._dtype(val)
    
    def unparse(self, val: Any) -> str:
        return str(val)


def create_simple_parser(dtype: Callable[[str], Any]) -> Type[BaseStringParser]:
    return type(f"", (BaseStringParser,), {'_dtype': dtype})

StringParser = create_simple_parser(str)
IntegerParser = create_simple_parser(int)
FloatParser = create_simple_parser(float)

class BoolParser(BaseStringParser):
    _false_strings = ("f", "false")
    _true_strings = ("t", "true")
    
    def parse(self, val: str):
        val = val.lower()
        if val in self._true_strings:
            return True
        elif val in self._false_strings:
            return False
        else:
            raise ValueError(f"The input value must be either {self._true_strings} or {self._false_strings} (case insensitive), but was: {val}")

class ListParser(BaseStringParser):
    def parse(self, val: str):
        val = val.strip()
        if val[0] == "[" and val[-1] == "]":
            val = val[1:-1]
        return val.split(",")

class DictParser(BaseStringParser):
    def parse(self, val: str):
        val = val.strip()
        if val[0] == "{" and val[-1] == "}":
            val = val[1:-1]
        
        key_val_pairs = val.split(";")
        parsed = {}
        for key_val_pair in key_val_pairs:
            key, val = key_val_pair.split(":")
            parsed[key.strip()] = val.strip()
        
        return parsed


parsers_name = "cli"

parsers_to_register = [
    (InputField, BaseStringParser),
    (TextInput, StringParser),
    (NumericInput, FloatParser),
    (IntegerInput, IntegerParser),
    (FloatInput, FloatParser),
    (BoolInput, BoolParser),
    (ListInput, ListParser),
    (DictInput, DictParser),
]

for field, parser in parsers_to_register:
    field.register_parser(parsers_name, parser)
