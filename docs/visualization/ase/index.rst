ASE
---

A sisl `Geometry` object may easily be converted to ASE objects and thus directly
plotted.

   import sisl
   import ase.visualize.view as view

   geom = sisl.geom.graphene()
   view(geom.toASE())

will open a new window showing the atoms.
