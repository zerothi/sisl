.. _introduction:

Introduction
============

sisl has a number of features which makes it easy to jump right into
and perform a large variation of tasks.

1. Easy creation of geometries. Similar to `ASE`_ sisl provides an
   easy scripting engine to create and manipulate geometries.
   The goal of sisl is not specifically DFT-related software which
   typically only targets a limited number of atoms. One of the main
   features of sisl is the enourmously fast creation and manipulation of
   very large geometries such as attaching two geometries together,
   rotating atoms, removing atoms, changing bond-lengths etc. 
   Everything is optimized for extremely large scale systems `>1,000,000` atoms
   such that creating geometries for tight-binding models becomes a breeze.

2. Easy creation of tight-binding Hamiltonians via intrinsic and very fast
   algorithms for creating sparse matrices.
   One of the key-points is that the Hamiltonian is treated as a matrix.
   I.e. one may easily specify couplings without using routine calls.
   For large systems, `>100,000`, it also becomes advantegeous to iterate on
   sub-grids of atoms to speed up the creation by orders of magnitudes.
   sisl intrinsically implements such algorithms.

3. Post-processing of data from DFT software. One may easily add additional
   post-processing tools to use sisl on non-implemented data-files.

