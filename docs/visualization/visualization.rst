
Visualization
=============

sisl's strength lies in its post-processing capabilities of DFT outputs and/or manipulating
geometries.

However, quite frequently one is in need for good looking graphics. This document tries
to explain and show how one may use sisl and related tools for showing publication ready
images.


ASE
---

A sisl `Geometry` object may easily be converted to ASE objects and thus directly
plotted.

   import sisl
   import ase.visualize.view as view

   geom = sisl.geom.graphene()
   view(geom.toASE())

will open a new window showing the atoms.
