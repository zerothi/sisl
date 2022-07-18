import contextlib
from .cli import *


@contextlib.contextmanager
def context_parser(parser: str):
    from .._input_field import InputField

    old_parser = InputField._current_parser
    InputField._current_parser = parser
    try:
        yield
    except Exception as e:
        InputField._current_parser = old_parser
        raise e
    InputField._current_parser = old_parser


__all__ = []