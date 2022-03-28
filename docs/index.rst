.. sisl documentation master file, created by
   sphinx-quickstart on Wed Dec  2 19:55:34 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


|pypi|_
|conda|_
|zenodo|_
|discord|_

|license|_
|buildstatus|_
|codecov|_


.. title:: sisl: Toolbox for electronic structure calculations
.. meta::
   :description: sisl is a tool to manipulate density functional
		 theory code input and/or output. It also implements tight-binding
		 tools to create and manipulate multi-orbital (non)-orthogonal basis sets.
   :keywords: LCAO, Siesta, TranSiesta, OpenMX, TBtrans,
	      VASP, GULP, BigDFT,
	      DFT,
	      tight-binding, electron, electrons, phonon, phonons


sisl: tight-binding and DFT interface library
=============================================

The Python library `sisl <https://github.com/zerothi/sisl>`_ was born out of a need to handle (create and read), manipulate and analyse output from DFT programs.
It was initially developed by Nick Papior (co-developer of `Siesta`_) as a side-project to `TranSiesta`_
and `TBtrans`_ to efficiently analyse TBtrans output for N-electrode calculations.  
Since then it has expanded to accommodate a rich set of DFT code input/outputs such as (but not limited to)
`VASP`_, `OpenMX`_, `BigDFT`_, `Wannier90`_.

A great deal of codes are implementing, roughly, the same thing.
However, every code implements their own analysis and post-processing utilities which typically
turns out to be equivalent utilities only having the interface differently.

sisl tries to solve some of the analysis issues by creating a unified scripting approach
in Python which does analysis using the same interface, regardless of the code being used.
For instance, one may read the Kohn-Sham eigenvalue spectrum from various codes and return them
in a consistent manner so the post-processing is the same, regardless of the code being used.

sisl is also part of the training material for a series of workshops hosted `here <workshop_>`_.

In some regards it has overlap with `ASE`_ and sisl also interfaces with ASE.


First time use
--------------

Here we show 2 examples of using sisl together with `Siesta`_.

To read in a Hamiltonian from a Siesta calculation and calculate the DOS for a given Monkhorst-Pack grid
one would do::

    import sisl
    import numpy as np
    H = sisl.get_sile('RUN.fdf').read_hamiltonian()
    mp = sisl.MonkhorstPack(H, [13, 13, 13])
    E = np.linspace(-4, 4, 500)
    DOS = mp.apply.average.DOS(E)
    from matplotlib import pyplot as plt
    plt.plot(E, DOS)

Which calculates the DOS for a 13x13x13 Monkhorst-Pack grid.

Another common analysis is real-space charge analysis. The following command line subtracts two real-space
charge grids and writes them to a CUBE file:

.. code-block:: bash

   sgrid reference/Rho.grid.nc --diff Rho.grid.nc --geometry RUN.fdf --out diff.cube

which may be analysed using VMD, XCrySDen or other tools.


Every use of sisl
-----------------

There are different places for getting information on using sisl. Here is a short list
of places to search/ask for answers:

- This page for the documentation!
- Workshop examples showing different uses, see `workshop`_
- Ask questions on its use on the Github `issue page <issue_>`_
- Ask questions on `Discord <sisl-discord_>`_


.. toctree::
   :hidden:
   :maxdepth: 2

   introduction
   contribute
   other
   cite
   publications

.. toctree::
   :maxdepth: 2
   :caption: User Guide
   
   installation
   tutorials.rst
   scripts/scripts

.. toctree::
   :maxdepth: 2
   :caption: Toolboxes

   toolbox/index

.. toctree::
   :maxdepth: 2
   :caption: Visualization
   
   visualization/ase/index
   visualization/viz_module/index

.. toctree::
   :maxdepth: 3
   :caption: Reference documentation
   
   api/index


A table of contents for all methods may be found :ref:`here <genindex>` while
a table of contents for the sub-modules may be found :ref:`here <modindex>`.


.. |buildstatus| image:: https://github.com/zerothi/sisl/actions/workflows/test.yaml/badge.svg?branch=main
.. _buildstatus: https://github.com/zerothi/sisl/actions/workflows/test.yaml

.. |pypi| image:: https://badge.fury.io/py/sisl.svg
.. _pypi: https://badge.fury.io/py/sisl

.. |license| image:: https://img.shields.io/badge/License-MPL%202.0-brightgreen.svg
.. _license: https://www.mozilla.org/en-US/MPL/2.0/

.. |conda| image:: https://anaconda.org/conda-forge/sisl/badges/version.svg
.. _conda: https://anaconda.org/conda-forge/sisl

.. |codecov| image:: https://codecov.io/gh/zerothi/sisl/branch/main/graph/badge.svg
.. _codecov: https://codecov.io/gh/zerothi/sisl

.. |zenodo| image:: https://zenodo.org/badge/doi/10.5281/zenodo.597181.svg
.. _zenodo: https://doi.org/10.5281/zenodo.597181

.. |discord| image:: https://img.shields.io/discord/742636379871379577.svg?label=&logo=discord&logoColor=ffffff&color=green&labelColor=red
.. _discord: https://discord.gg/5XnFXFdkv2
