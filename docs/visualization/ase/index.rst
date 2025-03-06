:orphan:

ASE
---

A sisl `Geometry` object may easily be converted to ASE objects and thus directly
plotted.


.. code-block::

   import sisl as si
   from ase.visualize import view

   geom = si.geom.graphene()
   view(geom.to.ase())

will open a new window showing the atoms.
