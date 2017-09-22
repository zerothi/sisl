"""
Utility routines (:mod:`sisl.utils`)
====================================

.. currentmodule:: sisl.utils

Several utility functions are used throughout sisl.

Range routines (:mod:`sisl.utils.ranges`)
-----------------------------------------

.. module:: sisl.utils.ranges

.. autosummary::
   :toctree: api-sisl/

   array_arange - fast creation of sub-aranges
   strmap
   strseq
   lstranges
   erange
   list2range
   fileindex

Miscellaneous routines (:mod:`sisl.utils.misc`)
-----------------------------------------------

.. module:: sisl.utils.misc

.. autosummary::
   :toctree: api-sisl/

   str_spec
   direction - abc/012 -> 012
   angle - radian to degree
   iter_shape
   math_eval

Command line utilites (:mod:`sisl.utils.cmd`)
---------------------------------------------

.. module:: sisl.utils.cmd

.. autosummary::
   :toctree: api-sisl/

   default_namespace
   ensure_namespace
   dec_default_AP
   dec_collect_action
   dec_collect_and_run_action
   dec_run_actions

"""

from .cmd import *
from .misc import *
from .ranges import *

__all__ = [s for s in dir() if not s.startswith('_')]
