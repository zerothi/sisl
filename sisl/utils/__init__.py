"""
====================================
Utility routines (:mod:`sisl.utils`)
====================================

.. currentmodule:: sisl.utils

Several utility functions are used throughout sisl.

Range routines
==============

.. autosummary::
   :toctree: api-generated/

   array_arange - fast creation of sub-aranges
   strmap
   strseq
   lstranges
   erange
   list2range
   fileindex

Miscellaneous routines
======================

.. autosummary::
   :toctree: api-generated/

   str_spec
   direction - abc/012 -> 012
   angle - radian to degree
   iter_shape
   math_eval

Command line utilites
=====================

.. autosummary::
   :toctree: api-generated/

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

#for rm in ['cmd', 'misc', 'ranges']:
#    __all__.remove(rm)
