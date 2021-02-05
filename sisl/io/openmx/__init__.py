"""
OpenMX
======

   omxSileOpenMX - input file
"""
from .sile import *

from .omx import *

__all__ = [s for s in dir() if not s.startswith('_')]
