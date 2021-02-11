.. _shapes:

Shapes
======

.. module:: sisl.shape

Shapes are geometric objects that enables one to locate positions in- or
out-side a given shape.

Shapes are constructed from basic geometric shapes, such as cubes, ellipsoids
and spheres, and they enable any _set_ operation on them,
intersection (``A & B``), union (``A | B``), disjunction (``A ^ B``)
or complementary (``A - B``).

All shapes in sisl allows one to perform arithmetic on them.
I.e. one may *add* two shapes to accomblish what would be equivalent
to an ``&`` operation. The resulting shape will be a `CompositeShape` which
implements the necessary routines to ensure correct operation.


.. autosummary::
   :toctree: generated/

   ~sisl.shape.Shape - base class
   ~sisl.shape.Cuboid - 3d cube
   ~sisl.shape.Cube - 3d box
   ~sisl.shape.Ellipsoid
   ~sisl.shape.Sphere
   ~sisl.shape.NullShape
