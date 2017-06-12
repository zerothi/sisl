
.. _tutorial-01:

Geometry creation -- part 1
---------------------------

To create a `Geometry` one needs to define a set of attributes.
The *only* required information is the atomic coordinates::

   >>> single_hydrogen = Geometry([[0., 0., 0.]])
   >>> print(single_hydrogen)
   {na: 1, no: 1, species:
    {Atoms(1):
       (1) == [H, Z: 1, orbs: 1, mass(au): 1.00794, maxR: -1.00000], 
    },
    nsc: [1, 1, 1], maxR: -1.0
   }

this will create a `Geometry` object with 1 Hydrogen atom with a single orbital
(default if not specified), and a supercell of 10 A in each Cartesian direction.
When printing a `Geometry` object a list of information is printed in an
XML-like fashion. ``na`` corresponds to the total number of atoms in the
geometry, while ``no`` refers to the total number of orbitals.
The species are printed in a sub-tree and ``Atoms(1)`` means that there is
one distinct atomic specie in the geometry. That atom is a Hydrogen, with mass
listed in atomic-units. ``maxR`` refers to the maximum range of all the orbitals
associated with that atom. A negative number means that there is no specified
range.
Lastly ``nsc`` refers to the number of neighbouring super-cells that is represented
by the object. In this case ``[1, 1, 1]`` means that it is a molecule and there
are no super-cells (only the unit-cell).

To specify the atomic specie one may do::

   >>> single_carbon = Geometry([[0., 0., 0.]], Atom('C'))

which changes the Hydrogen to a Carbon atom. See <link to atom_01.rst> on how to create different atoms.
   
To create a geometry with two different atomic species, for instance a chain
of alternating Natrium an Chloride atoms, separated by 1.6 A one may do::

   >>> chain = Geometry([[0. , 0., 0.],
                         [1.6, 0., 0.]], [Atom('Na'), Atom('Cl')],
	                 [3.2, 10., 10.])

note the last argument which specifies the Cartesian lattice vectors.
sisl is clever enough to repeat atomic species if the number of atomic
coordinates is a multiple of the number of passed atoms, i.e.::

   >>> chainx2 = Geometry([[0. , 0., 0.],
                           [1.6, 0., 0.],
                           [3.2, 0., 0.],
                           [4.8, 0., 0.]]], [Atom('Na'), Atom('Cl')],
		           [6.4, 10., 10.])

which is twice the length of the first chain with alternating Natrium and Chloride atoms,
but otherwise identical.

This is the most basic form of creating geometries in sisl and is the starting
point of almost anything related to sisl. 

