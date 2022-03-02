ASE
---

A sisl `Geometry` object may easily be converted to ASE objects and thus directly
plotted.


.. code::

   import sisl
   from ase.visualize import view

   geom = sisl.geom.graphene()
   view(geom.to.ase())

will open a new window showing the atoms.
