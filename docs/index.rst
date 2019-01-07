.. sisl documentation master file, created by
   sphinx-quickstart on Wed Dec  2 19:55:34 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


|pypi|_
|conda|_
|zenodo|_
|license|_
|donate|_

|buildstatus|_
|gitter|_
|codecov|_


.. title:: Scientific Python toolbox for electronic structure calculations
.. meta::
   :description: sisl is a tool to manipulate density functional
		 theory code input and/or output. It also implements tight-binding
		 tools to create and manipulate multi-orbital (non)-orthogonal basis sets.
   :keywords: LCAO, Siesta, TranSiesta, TBtrans, VASP, GULP, DFT, tight-binding, electrons, phonons


sisl: a tight-binding and DFT postprocessing library
====================================================

sisl is an open source Python library to manipulate density functional
theory code input and/or output.
It also allows constructing and analyzing custom made tight-binding models with high
throughput for millions of orbitals/atoms.
Any user tight-binding model may be used as input to the non-equilibrium Green function
transport calculator `TBtrans`_.
sisl deals with both electrons and phonons.

sisl is hosted `here <http://github.com/zerothi/sisl>`_.

Tight-binding and localized orbitals are an essential part of the electronic structure
community. A great deal of codes are implementing either of the afore mentioned methodologies.
However, every code implements their own analysis and post-processing utilities which typically
turns out to be equivalent utilities only having the interface differently.

sisl tries to solve some of the analysis issues by creating a unified scripting approach
in Python which does analysis using the same interface, regardless of code being used.
For instance one may read the Kohn-Sham eigenvalue spectrum from various codes and return them
in a consistent manner so the post-processing is the same, regardless of code being used.

Tight-binding models are created dynamically in sisl using a custom sparse Hamiltonian.
Having *any* electronic structure one can calculate physical properties such as (projected)
density of states, wavefunctions, spin moments, band velocities and Berry phases.


Features
--------

sisl consists of several distinct features:

* Geometries; create, extend, combine, manipulate different geometries readed from
  a large variety of DFT-codes and/or from generically used file formats.

* Hamiltonian; easily create tight-binding Hamiltonians with user chosen number of
  orbitals per atom. Or read in Hamiltonians from DFT software such as `Siesta`_,
  `Wannier90`_, `GULP`_ etc. Secondly, there is intrinsic capability of orthogonal
  *and* non-orthogonal Hamiltonians.

* Post-process output from several DFT codes: `Siesta`_,
  `Wannier90`_, `VASP`_, `OpenMX`_, `GULP`_, `BigDFT`_, see `sisl.io` for a complete
  list of allowed files.

* Command line utilities for processing of data files for a wide
  variety of file formats.


.. toctree::
   :hidden:
   :maxdepth: 2

   introduction
   contribute
   other

.. toctree::
   :maxdepth: 2
   :caption: Publications

   cite
   publications

.. toctree::
   :maxdepth: 2
   :caption: User Guide
   
   installation
   tutorials.rst
   examples.rst
   scripts/scripts
   rst/files

.. toctree::
   :maxdepth: 3
   :caption: Reference documentation
   
   api


A table of contents for all methods may be found :ref:`here <genindex>` while
a table of contents for the sub-modules may be found :ref:`here <modindex>`.


.. |buildstatus| image:: https://travis-ci.org/zerothi/sisl.svg
.. _buildstatus: https://travis-ci.org/zerothi/sisl

.. |pypi| image:: https://badge.fury.io/py/sisl.svg
.. _pypi: https://badge.fury.io/py/sisl

.. |license| image:: https://img.shields.io/badge/License-LGPL%20v3-blue.svg
.. _license: https://www.gnu.org/licenses/lgpl-3.0

.. |conda| image:: https://anaconda.org/conda-forge/sisl/badges/installer/conda.svg
.. _conda: https://anaconda.org/conda-forge/sisl

.. |codecov| image:: https://codecov.io/gh/zerothi/sisl/branch/master/graph/badge.svg
.. _codecov: https://codecov.io/gh/zerothi/sisl

.. |donate| image:: https://img.shields.io/badge/Donate-PayPal-green.svg
.. _donate: https://www.paypal.com/cgi-bin/webscr?cmd=_donations&business=NGNU2AA3JXX94&lc=DK&item_name=Papior%2dCodes&item_number=codes&currency_code=EUR&bn=PP%2dDonationsBF%3abtn_donate_SM%2egif%3aNonHosted

.. |zenodo| image:: https://zenodo.org/badge/doi/10.5281/zenodo.597181.svg
.. _zenodo: http://dx.doi.org/10.5281/zenodo.597181

.. |gitter| image:: https://img.shields.io/gitter/room/nwjs/nw.js.svg
.. _gitter: https://gitter.im/sisl-tool/Lobby
