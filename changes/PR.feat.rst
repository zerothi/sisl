Enabled implicit conversion of unknown objects

Now users can automatically convert unknown objects
and use them directly in `sisl` methods that are
implemented for various methods.

E.g.

.. code::

   import ase
   import sisl

   gr = sisl.geom.graphene()
   sisl.rotate(gr, ...)

   gr = ase.Atoms(...)
   sisl.rotate(gr, ...)

will both work. The latter will return a sisl
geometry.
