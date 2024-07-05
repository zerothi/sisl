.. _core:

**********************
Functional programming
**********************

.. currentmodule:: sisl

sisl provides simple functionality that may be used by various
sisl objects.

.. list of methods that currently are dispatched can be created via

     grep -A 2 register_sisl_dispatch ../../src/**/_ufuncs_*.py | grep def | tr '(' ' ' | awk '{print "  ",$2}' | sort | uniq >> core.rst


.. autosummary::
   :toctree: generated/

   add
   append
   apply
   ~physics.berry_curvature
   center
   copy
   insert
   prepend
   remove
   repeat
   rotate
   scale
   sort
   ~physics.spin_berry_curvature
   sub
   swap
   swapaxes
   tile
   translate
   ~physics.velocity
   unrepeat
   untile
   write
