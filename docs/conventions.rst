
.. _conventions:

Conventions in sisl
===================

sisl is a joint effort to enable easy usage of electronic structure
concepts in a scientific setting. Ranging between introductory levels
up to, and including, full scientific employees.

There are certain conventions that we want to enforce in sisl, and
we strive them as far as possible.



.. _conventions.units:

Units
-----

All units in sisl are standardized to:

Length
   All units internally and returned are in Angstrom.

Energy
   All units internally and returned are in electron volts.


.. _convention.mathematics:

Mathematical notation
---------------------

Here is a list of rules that sisl will strive to adhere to.
If you find any inconsistencies in the documentation,
please let us know!

* upper case characters such as :math:`I` and :math:`J` refer
  to atomic indices
* lower case characters such as :math:`i` and :math:`j` refer
  to orbital indices, e.g. :math:`i\in I`
* scalars are represented via lower case italics, such
  as :math:`a`
* vectors are represented via lower case bold faced, such
  as :math:`\mathbf a`

  One may use :math:`\hat{\mathbf a}` to signal normal vectors

  * dot-products between vectors should be explicit :math:`\mathbf a\cdot\mathbf b`

* matrices are represented via upper case bold faced, such
  as :math:`\mathbf A`

  * vector-matrix products are implicit :math:`\mathbf a\mathbf B`

  * matrix-matrix products are implicit :math:`\mathbf A\mathbf B`

* Greek letters are used for other indices, such as spin (:math:`\sigma`),
  Cartesian or lattice vectors.

* range of indices are denoted with :math:`\{ \}`, such that,
  :math:`\{i\}` is an orbital index range, :math:`\{I\}`
  is an atomic index range and :math:`\{\alpha\}` refers
  to some *other* range which should be inferred from
  the context

* the imaginary number is generally referred to as :math:`i` in
  physics, its meaning should be implicit from the context.
