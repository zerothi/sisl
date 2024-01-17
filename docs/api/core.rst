.. _core:

**********
Functional
**********

.. currentmodule:: sisl

sisl provides simple functionality that may be used by various
sisl objects.

.. list of methods that currently are dispatched can be created via

     grep "def " ../../src/**/_ufuncs_*.py | tr '(' ' ' | awk '{print "  ",$2}' | sort | uniq >> core.rst


.. autosummary::
   :toctree: generated/

   add
   append
   center
   copy
   prepend
   remove
   repeat
   rotate
   sub
   swapaxes
   tile
   translate
   unrepeat
   untile
