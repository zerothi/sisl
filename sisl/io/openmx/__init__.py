"""
==============================
OpenMX (:mod:`sisl.io.openmx`)
==============================

.. module:: sisl.io.openmx
   :noindex:

`OpenMX`_ software is an LCAO code.

.. autosummary::
   :toctree:

   omxSileOpenMX - input file

"""
from .sile import *

from .omx import *

__all__ = [s for s in dir() if not s.startswith('_')]
