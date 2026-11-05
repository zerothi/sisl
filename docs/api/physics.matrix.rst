.. _physics.matrix:

Physical quantites
==================

.. currentmodule:: sisl.physics

Physical quantities such as Hamiltonian and density matrices are representated
through specific classes enabling various handlings.


Spin
----

.. autosummary::
   :toctree: generated/

   Spin

Matrices
--------

.. autosummary::
   :toctree: generated/

   EnergyDensityMatrix
   DensityMatrix
   Hamiltonian
   DynamicalMatrix
   Overlap


Tight-binding models
--------------------

A tight-binding model from Slater-Koster integrals, used with ``H.construct(sk)``.

.. autosummary::
   :toctree: generated/

   SlaterKoster


Models from DFTB parameter sets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A DFTB parameter set, for example ``mio`` from `dftb.org <https://dftb.org/parameters>`_,
has one Slater-Koster file (``.skf``) for each ordered pair of elements.
:meth:`~sisl.io.dftb.skfSileDFTB.read_slater_koster` reads the integrals of one pair
into a `SlaterKoster` with the Hamiltonian and the overlap, interpolated as in DFTB+.
It reads ``O-H.skf`` and ``H-O.skf`` in one call. `SlaterKoster.merge` joins the pairs
into one model:

>>> import sisl
>>> sk = sisl.SlaterKoster.merge(
...     *(sisl.get_sile(f).read_slater_koster() for f in ["O-O.skf", "O-H.skf", "H-H.skf"])
... )
>>> O, H = sk.atoms["O"], sk.atoms["H"]
>>> water = sisl.Geometry(
...     [[0, 0, 0], [0.757, 0.586, 0], [-0.757, 0.586, 0]], [O, H, H], lattice=[10, 10, 10]
... )
>>> Hw = sisl.Hamiltonian(water, orthogonal=False)
>>> Hw.construct(sk)
>>> Hw.eigh().round(2)
array([-24.85, -12.6 , -10.39,  -9.04,  10.2 ,  15.46])

- **Tags.** The atoms are matched by their `~sisl.Atom.tag`, which must be the element
  of the files (``"O"`` and ``"H"`` above). A file with a suffix is another model of the
  same pair: read ``N-H_short.skf`` in place of ``N-H.skf``, with ``N-N.skf`` and ``H-H.skf``.
- **Shells.** By default, the shells of an element are the basis that its homonuclear file
  (``O-O.skf``) documents, here ``2s 2p`` for O and ``1s`` for H. The orbitals of the atoms
  must have these labels: `SlaterKoster.atoms` has the atoms with these orbitals, and a range
  that gives the supercell of the model. Use the ``shells`` argument (a shell list for each tag) to select
  other shells, or when the folder has no homonuclear file (for example the ``hyb`` set
  has no ``H-H.skf``, thus ``Ga-H.skf`` needs the shells of H):

  >>> sk_OH = sisl.get_sile("O-H.skf").read_slater_koster({"O": ["2s", "2p"], "H": ["1s"]})

- **On-site energies** come from the homonuclear files. Thus read ``O-O.skf`` and ``H-H.skf`` too.
- **Documentation.** :meth:`~sisl.io.dftb.skfSileDFTB.read_documentation` returns the XML
  documentation at the end of a file: the author, the elements and the basis of the set.
- Files with f shells (the extended format, first character ``@``) are not supported.
  The repulsive energy (``Spline``) is not part of the Hamiltonian and is not read.


Self energies
-------------

Self-energies are specific physical quantities that enables integrating
out semi-infinite regions.

.. autosummary::
   :toctree: generated/

   SelfEnergy
   WideBandSE
   SemiInfinite
   RecursiveSI
   RealSpaceSE
   RealSpaceSI


Bloch's theorem
---------------

Bloch's theorem is a very powerful procedure that enables one to utilize
the periodicity of a given direction to describe the complete system.

.. autosummary::
   :toctree: generated/

   Bloch
