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
   For large systems, `>10,000` atoms, it becomes advantegeous to iterate on
   sub-grids of atoms to speed up the creation by orders of magnitudes.
   sisl intrinsically implements such algorithms.

3. Post-processing of data from DFT software. One may easily add additional
   post-processing tools to use sisl on non-implemented data-files.



Package
-------

sisl is mainly a Python package with many intrinsic capabilities.

Follow `these <installation>` instructions for installing sisl.

DFT
~~~

Many intrinsic DFT program files are handled by sisl and extraction of the necessary
physical quantities are easily performed.

Its main focus has been `Siesta`_ which thus has the largest amount of implemented
output files.


Geometry manipulation
~~~~~~~~~~~~~~~~~~~~~

Geometries can easily be generated from basic routines and enables easy repetitions,
additions, removal etc. of different atoms/geometries, for instance to generate a
graphene flake one can use this small snippet:


   >>> import sisl
   >>> graphene = sisl.geom.graphene(1.42).repeat(100, 0).repeat(100, 1)

which generates a graphene flake of :math:`2 \cdot 100 \cdot 100 = 20000` atoms.


Command line usage
------------------

The functionality of sisl is also extended to command line utilities for easy manipulation
of data from DFT programs. There are a variety of commands to manipulate generic data (`sdata`),
geometries (`sgeom`) or grid-related quantities (`sgrid`).

Additionally there are user-contributed toolboxes which are exposed in the module `sisl_toolbox`
and through the `stoolbox` command.
