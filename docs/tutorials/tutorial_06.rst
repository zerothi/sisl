
.. _tutorial-06:

Electronic structure setup -- part 2
------------------------------------

Following :ref:`part 1 <tutorial-05>` we focus on how to generalize the specification
of the hopping parameters in a more generic way.

First, we re-create the square geometry (with one orbital per atom). However,
to generalize the specification of the hopping parameters it is essential that
we specify how long range the orbitals interact. In the following we set the
atomic specie to be a Hydrogen atom with a single orbital with a range of :math:`1\,Å`

  >>> Hydrogen = Atom(1, R=1.)
  >>> square = Geometry([[0.5, 0.5, 0]], Hydrogen,
                        sc=SuperCell([1, 1, 10], [3, 3, 1]))
  >>> H = Hamiltonian(square)
  >>> print(H)
  {spin: 1, non-zero: 0
   {na: 1, no: 1, species:
    {Atoms(1):
      (1) == [H, Z: 1, orbs: 1, mass(au): 1.00794, maxR: 1.00000], 
    },
    nsc: [3, 3, 1], maxR: 1.0
   }
  }

Note how the ``maxR`` variable has changed from ``-1.0`` to ``1.0``. This corresponds to the
maximal orbital range in the geometry. Here there is only one type of orbital, but for
geometries with several different orbitals, there may be different orbital ranges.

Now one can assign the generalized parameters::

  >>> for ia in square: # loop atomic indices (here equivalent to the orbital indices)
  ...     idx_a = square.close(ia, R=[0.1, 1.1])
  ...     H[ia, idx_a[0]] = -4.
  ...     H[ia, idx_a[1]] = 1.

The `Geometry.close` function is a convenience function to return atomic indices of
atoms within a certain radius. For instance ``close(0, R=1.)`` returns all atomic
indices within a spherical radius of :math:`1\,Å` from the first atom in the geometry,
including it-self.
``close([0., 0., 1.], R=1.)`` will return all atomic indices within :math:`1\,Å` of the
coordinate ``[0., 0., 1.]``.
If one specifies a list of ``R`` it will return the atomic indices in the sphere within the
first element; and for the later values it will return the atomic indices in the spherical
shell between the corresponding radii and the previous radii.

The above code is the preferred method of creating a Hamiltonian. It is safe because it ensures
that all parameters are set, and symmetrized.

For very large geometries (larger than 50,000 atoms) the above code will be *extremely* slow.
Hence, the preferred method to setup the Hamiltonian for these large geometries is::

  >>> for ias, idxs in square.iter_block():
  ...    for ia in ias:
  ...        idx_a = square.close(ia, R=[0.1, 1.1], idx=idxs)
  ...        H[ia, idx_a[0]] = -4.
  ...        H[ia, idx_a[1]] = 1.

The code above is the preferred method of specifying the Hamiltonian
parameters.

The complete code for this example (plus the band-structure) can be found
:download:`here <tutorial_06_square.py>`.
