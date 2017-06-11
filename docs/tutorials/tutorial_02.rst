
.. _tutorial-02:

Geometry creation -- part 2
---------------------------

Many geometries are intrinsically enabled via the `sisl.geom` submodule.

The default geometries that may be created may easily be used.

* `honeycomb` (graphene unit-cell)::

     hBN = geom.honeycomb(1.5, [Atom('B'), Atom('N')])
  
* `graphene` (equivalent to `honeycomb` with Carbon atoms)::

     graphene = geom.graphene(1.42)

* Simple, body-centered and face-centered cubic as well as HCP
  All have the same interface::

     sc = geom.sc(2.5)
     bcc = geom.bcc(2.5)
     fcc = geom.fcc(2.5)
     hcp = geom.hcp(2.5)

* Nanotubes with different chirality::

     ntb = geom.nanotube(1.54, chirality=(n, m))
  
* Diamond::

     d = geom.diamond(3.57)


