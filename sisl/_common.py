""" Global common files """
from enum import Enum, Flag, auto, unique

__all__ = ["Opt"]


@unique
class Opt(Flag):
    """ Global option arguments used throughout sisl

    These flags may be combined via bit-wise operations
    """
    NONE = auto()
    ANY = auto()
    ALL = auto()
