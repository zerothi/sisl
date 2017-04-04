
Specifying super-cell information
---------------------------------

An important thing when dealing with geometries in how the *super-cell* is
used. First, recall that the number of supercells can be retrieved by::

   >>> geometry = Geometry([[0, 0, 0]])
   >>> print(geometry)
   {na: 1, no: 1, species:
    {{Atoms(1):
       (1) == [H, Z: 1, orbs: 1, mass(au): 1.00794, dR: -1.00000], 
      }
    },
    nsc: [1, 1, 1], dR: -1.0
   }
   >>> geometry.nsc # or geometry.sc.nsc
   array([1, 1, 1], dtype=int32)

where ``nsc`` is the specific super-cell information. In the default
case only the unit-cell is taken into consideration. However when using
the `Geometry.close` or `Geometry.within` functions the atomic indices it
becomes important how large the supercell is.

Specifying the number of super-cells may be done when creating the geometry,
or after it has been created::

   >>> geometry = Geometry([[0, 0, 0]], sc=SuperCell(5, [3, 3, 3]))
   >>> geometry.nsc
   array([3, 3, 3], dtype=int32)
   >>> geometry.set_nsc([3, 1, 5])
   >>> geometry.nsc
   array([3, 1, 5], dtype=int32)

The final geometry enables intrinsic routines to interact with the 2 closest neighbouring cells
along the first lattice vector (`1 + 2 == 3`), and the 4 closest neighbouring cells
along the third lattice vector (`1 + 2 + 2 == 5`). 
