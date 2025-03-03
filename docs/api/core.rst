.. _core:

**********************
Functional programming
**********************

.. currentmodule:: sisl

sisl provides simple functionality that may be used by various
sisl objects.

.. tip::

   All of these *functional* methods can automatically handle external
   classes. But only if `sisl` has an internal conversion of that
   object.

   E.g. one can do:

   .. code::

      from ase.build import bulk
      import sisl as si

      bulk_rotated = si.rotate(bulk("Au"), 30, [1, 1, 1])

   Note that `sisl` will automatically convert the `ase.Atoms` object
   to a `sisl.Geometry`, and then do the rotation call.

   Hence, the returned object will be a `sisl.Geometry` object.


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
